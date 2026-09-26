import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeCredentials } from "../llm/credentials.js";
import { ToolExecutor, isProcessGroupAlive } from "../tools/executor.js";
import type { ResourceLimits } from "../tools/resource-limits.js";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

interface StubTerminal {
  id: string;
  waitForExit: () => Promise<{ exitCode: number }>;
  currentOutput: () => Promise<{ output: string }>;
  release: () => Promise<void>;
}

function createConnectionStub(opts: {
  permission?: "allow" | "reject" | "cancelled" | "missing" | "unoffered" | "unknown";
  readError?: boolean;
  writeError?: boolean;
  terminalOutput?: string;
  /** When set, client readTextFile returns this instead of the on-disk content (simulates a dirty buffer). */
  clientFileContent?: string;
  /** Called when a permission request arrives — use it to mutate files mid-prompt. */
  onPermission?: () => void;
} = {}) {
  const updates: Array<Record<string, unknown>> = [];
  const permissionRequests: Array<unknown> = [];
  const terminalCalls: Array<{ command: string; args?: string[] }> = [];
  const writeTextFileCalls: Array<{ sessionId: string; path: string; content: string }> = [];
  const readTextFileCalls: Array<{ sessionId: string; path: string }> = [];

  return {
    updates,
    permissionRequests,
    terminalCalls,
    writeTextFileCalls,
    readTextFileCalls,
    async sessionUpdate(payload: Record<string, unknown>) {
      updates.push(payload);
    },
    async readTextFile(params: { sessionId: string; path: string }) {
      if (opts.readError) throw new Error("file not found");
      readTextFileCalls.push(params);
      if (opts.clientFileContent !== undefined) return { content: opts.clientFileContent };
      // Mirror a real client: readTextFile serves the file's current on-disk contents.
      return { content: readFileSync(params.path, "utf8") };
    },
    async writeTextFile(params: { sessionId: string; path: string; content: string }) {
      writeTextFileCalls.push(params);
      if (opts.writeError) throw new Error("permission denied");
      writeFileSync(params.path, params.content, "utf8");
    },
    async createTerminal(params: { command: string; args?: string[] }): Promise<StubTerminal> {
      terminalCalls.push(params);
      return {
        id: "term-1",
        async waitForExit() {
          return { exitCode: 0 };
        },
        async currentOutput() {
          return { output: opts.terminalOutput ?? "(stub)" };
        },
        async release() {
          /* noop */
        },
      };
    },
    async requestPermission(params: unknown) {
      permissionRequests.push(params);
      if (opts.onPermission) opts.onPermission();
      switch (opts.permission ?? "allow") {
        case "allow":
          return { outcome: { outcome: "selected", optionId: "allow" } };
        case "reject":
          return { outcome: { outcome: "selected", optionId: "reject" } };
        case "cancelled":
          return { outcome: { outcome: "cancelled" } };
        case "missing":
          return { outcome: { outcome: "selected" } };
        case "unoffered":
          return { outcome: { outcome: "selected", optionId: "allow_always" } };
        case "unknown":
          return { outcome: { outcome: "unknown" } };
      }
    },
  };
}

const FULL_CAPS = {
  fs: { readTextFile: true, writeTextFile: true },
  terminal: true,
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

// Git Bash translates native Windows paths embedded in `sh -c` command text.
// Keep fixture files relative to the shell cwd there, while retaining absolute
// paths on POSIX where no translation occurs.
function shellNodeCommand(): string {
  return process.platform === "win32" ? "node" : shellQuote(process.execPath);
}

function shellFixturePath(absolutePath: string, relativePath: string): string {
  return JSON.stringify(process.platform === "win32" ? relativePath : absolutePath);
}

async function removeTestDirectory(path: string): Promise<void> {
  const attempts = process.platform === "win32" ? 40 : 1;
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      await rm(path, { recursive: true, force: true });
      return;
    } catch (error) {
      lastError = error;
      const code = (error as NodeJS.ErrnoException).code;
      if (process.platform !== "win32" || !["EBUSY", "EPERM", "ENOTEMPTY"].includes(code ?? "")) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw lastError;
}

type FetchCall = {
  url: string;
  init: RequestInit;
  body: Record<string, unknown>;
  headers: Headers;
};

function jsonResponse(
  body: unknown,
  init: ResponseInit & { sessionId?: string } = {}
): Response {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (init.sessionId) headers.set("MCP-Session-Id", init.sessionId);
  return new Response(JSON.stringify(body), { ...init, headers });
}

function createFetchStub(responses: Response[]) {
  const calls: FetchCall[] = [];
  const fetchStub = async (url: string | URL | Request, init?: RequestInit) => {
    assert.ok(init, "fetch init is required");
    const body =
      typeof init.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : {};
    const headers = new Headers(init.headers);
    calls.push({ url: String(url), init, body, headers });
    const response = responses.shift();
    assert.ok(response, "unexpected fetch call");
    return response;
  };
  return { calls, fetchStub };
}

async function withStoredApiKey<T>(fn: () => Promise<T>): Promise<T> {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-creds-"));
  const oldEnv = process.env["Z_AI_API_KEY"];
  const oldXdg = process.env["XDG_CONFIG_HOME"];
  try {
    delete process.env["Z_AI_API_KEY"];
    process.env["XDG_CONFIG_HOME"] = dir;
    writeCredentials("from-disk", join(dir, "glm-acp-agent", "credentials.json"));
    return await fn();
  } finally {
    if (oldEnv === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = oldEnv;
    if (oldXdg === undefined) delete process.env["XDG_CONFIG_HOME"];
    else process.env["XDG_CONFIG_HOME"] = oldXdg;
    rmSync(dir, { recursive: true, force: true });
  }
}

async function withMockedFetch<T>(
  responses: Response[],
  fn: (calls: FetchCall[]) => Promise<T>
): Promise<T> {
  const oldFetch = globalThis.fetch;
  const { calls, fetchStub } = createFetchStub(responses);
  try {
    globalThis.fetch = fetchStub as typeof fetch;
    return await fn(calls);
  } finally {
    globalThis.fetch = oldFetch;
  }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

test("invalid JSON arguments yield a failed tool_call notification", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "read_file", "{ not json");
  assert.match(result.content, /could not parse tool arguments as JSON/);
  const last = conn.updates.at(-1) as { update: { sessionUpdate: string; status?: string } };
  assert.equal(last.update.sessionUpdate, "tool_call");
  assert.equal(last.update.status, "failed");
});

test("invalid roots and write/edit arguments fail before permissions or filesystem mutations", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-invalid-write-"));
  const path = join(dir, "note.txt");
  writeFileSync(path, "remove this", "utf8");
  const cases = [
    ["write_file", "null"],
    ["write_file", "[]"],
    ["write_file", "\"text\""],
    ["write_file", "1"],
    ["write_file", "true"],
    ["write_file", "{"],
    ["write_file", JSON.stringify({ path })],
    ["write_file", JSON.stringify({ path, content: null })],
    ["edit_file", JSON.stringify({ path, old_text: "remove this" })],
    ["edit_file", JSON.stringify({ path, old_text: "remove this", new_text: null })],
    ["edit_file", JSON.stringify({ path, old_text: "", new_text: "replacement" })],
  ] as const;
  try {
    for (const [toolName, rawArguments] of cases) {
      const conn = createConnectionStub({ permission: "allow" });
      const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
      const result = await exec.execute(`tc-${toolName}-${rawArguments.length}`, toolName, rawArguments);
      assert.match(result.content, /required|string|JSON object|could not parse/i);
      assert.equal(conn.permissionRequests.length, 0);
      assert.equal(conn.writeTextFileCalls.length, 0);
      const update = conn.updates.at(-1) as { update: { sessionUpdate: string; status?: string } };
      assert.equal(update.update.sessionUpdate, "tool_call");
      assert.equal(update.update.status, "failed");
    }
    assert.equal(readFileSync(path, "utf8"), "remove this");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("explicit empty write and edit text remain valid destructive operations", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-empty-write-"));
  const writePath = join(dir, "write.txt");
  const editPath = join(dir, "edit.txt");
  writeFileSync(writePath, "existing", "utf8");
  writeFileSync(editPath, "before unique snippet after", "utf8");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const write = await exec.execute("tc-write-empty", "write_file", JSON.stringify({ path: writePath, content: "" }));
    const edit = await exec.execute(
      "tc-edit-empty",
      "edit_file",
      JSON.stringify({ path: editPath, old_text: "unique snippet", new_text: "" })
    );
    assert.match(write.content, /written successfully/);
    assert.match(edit.content, /edited successfully/);
    assert.equal(readFileSync(writePath, "utf8"), "");
    assert.equal(readFileSync(editPath, "utf8"), "before  after");
    assert.equal(conn.permissionRequests.length, 2);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("unknown tool name yields a failed tool_call notification", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "frobnicate", "{}");
  assert.match(result.content, /unknown tool/);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("empty arguments string is accepted as empty object", async () => {
  const conn = createConnectionStub({ readError: true });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "read_file", "");
  // Path is empty so the tool reports an error, not a JSON parse error.
  assert.match(result.content, /path.*required/);
});

// ---------------------------------------------------------------------------
// Client capability independence
// ---------------------------------------------------------------------------

test("read_file reads from the agent process without fs.readTextFile capability", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-"));
  const path = join(dir, "note.txt");
  writeFileSync(path, "from disk", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} });
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    assert.equal(result.content, "from disk");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file reads the client's unsaved buffer when both fs capabilities are available", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-client-"));
  const path = join(dir, "note.txt");
  writeFileSync(path, "stale disk", "utf8");
  const conn = createConnectionStub({ clientFileContent: "unsaved buffer" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    assert.equal(result.content, "unsaved buffer");
    assert.deepEqual(conn.readTextFileCalls.map((call) => call.path), [path]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file writes from the agent process without fs.writeTextFile capability", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-"));
  const path = join(dir, "out.txt");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: { readTextFile: true } });
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "hi" })
    );
    assert.match(result.content, /written successfully/);
    assert.equal(readFileSync(path, "utf8"), "hi");
    assert.equal(conn.writeTextFileCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file routes through fs.writeTextFile when the client advertises it", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-client-"));
  const path = join(dir, "out.txt");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "via client" })
    );
    assert.match(result.content, /written successfully/);
    assert.equal(readFileSync(path, "utf8"), "via client");
    assert.deepEqual(
      conn.writeTextFileCalls.map((c) => ({ path: c.path, content: c.content })),
      [{ path, content: "via client" }]
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("list_files and run_command execute in the agent process without terminal capability", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-local-"));
  writeFileSync(join(dir, "entry.txt"), "data", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir);
  try {
    const ls = await exec.execute("tc1", "list_files", JSON.stringify({ path: "." }));
    assert.match(ls.content, /entry\.txt/);
    const rc = await exec.execute("tc2", "run_command", JSON.stringify({ command: "pwd" }));
    assert.match(rc.content, new RegExp(escapeRegExp(basename(dir))));
    assert.equal(conn.terminalCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

test("read_file truncates large files with a range marker", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-truncate-"));
  const path = join(dir, "big.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const first = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, limit: 4 })
    );
    assert.match(first.content, /line-1/);
    assert.match(first.content, /showing lines 1-4 of (10|11)/);
    assert.match(first.content, /pass offset=5/);

    const second = await exec.execute(
      "tc2",
      "read_file",
      JSON.stringify({ path, offset: 5, limit: 4 })
    );
    assert.match(second.content, /line-5/);
    assert.doesNotMatch(second.content, /line-4\b/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file treats editor line pagination as pagination, not byte truncation", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-editor-page-"));
  const path = join(dir, "paged.txt");
  writeFileSync(path, "line-1\nline-2\nline-3\nline-4", "utf8");
  const updates: Array<Record<string, unknown>> = [];
  const conn = {
    async sessionUpdate(payload: Record<string, unknown>) { updates.push(payload); },
    async readTextFile(params: { line?: number; limit?: number }) {
      const lines = ["line-1", "line-2", "line-3", "line-4"];
      const line = params.line ?? 1;
      const limit = params.limit ?? lines.length;
      return { content: lines.slice(line - 1, line - 1 + limit).join("\n") };
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, limit: 2 }),
    );
    assert.match(result.content, /showing lines 1-2 \(total unknown\); pass offset=3/);
    assert.doesNotMatch(result.content, /scan stopped at .*byte read limit/);

    const next = await exec.execute(
      "tc2",
      "read_file",
      JSON.stringify({ path, offset: 3, limit: 2 }),
    );
    assert.match(next.content, /^line-3\nline-4(?:\n|$)/);
    assert.doesNotMatch(next.content, /line-1|line-2/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file renders EOF for a conforming editor at a short or empty offset", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-editor-eof-"));
  const path = join(dir, "paged.txt");
  writeFileSync(path, "line-1\nline-2", "utf8");
  const conn = {
    async sessionUpdate() {},
    async readTextFile(params: { line?: number; limit?: number }) {
      const lines = ["line-1", "line-2"];
      const line = params.line ?? 1;
      const limit = params.limit ?? lines.length;
      return { content: lines.slice(line - 1, line - 1 + limit).join("\n") };
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const short = await exec.execute("tc1", "read_file", JSON.stringify({ path, offset: 1, limit: 4 }));
    assert.equal(short.content, "line-1\nline-2");
    const empty = await exec.execute("tc2", "read_file", JSON.stringify({ path, offset: 3, limit: 2 }));
    assert.match(empty.content, /end of file: offset 3 is beyond the end of/);
    assert.doesNotMatch(empty.content, /\(2 lines\)/);
    const farPast = await exec.execute("tc3", "read_file", JSON.stringify({ path, offset: 100, limit: 2 }));
    assert.match(farPast.content, /end of file: offset 100 is beyond the end of/);
    assert.doesNotMatch(farPast.content, /99 lines/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("read_file paginates a legacy editor full buffer for a short offset page", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-editor-legacy-page-"));
  const path = join(dir, "paged.txt");
  writeFileSync(path, "line-1\nline-2\nline-3", "utf8");
  const updates: Array<Record<string, unknown>> = [];
  const conn = {
    async sessionUpdate(payload: Record<string, unknown>) { updates.push(payload); },
    // Older ACP clients ignore both line and limit and return the full buffer.
    async readTextFile() {
      return { content: "line-1\nline-2\nline-3" };
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, offset: 2, limit: 2 }),
    );
    assert.match(result.content, /^line-2\nline-3\n/);
    assert.doesNotMatch(result.content, /^line-1\n/);
    assert.match(result.content, /showing lines 2-3 .*end of file/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file treats a one-line legacy editor buffer as EOF past its only line", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-editor-legacy-single-line-"));
  const path = join(dir, "single-line.txt");
  writeFileSync(path, "only-line", "utf8");
  const updates: Array<Record<string, unknown>> = [];
  const conn = {
    async sessionUpdate(payload: Record<string, unknown>) { updates.push(payload); },
    // This client ignores both line and limit, including the far-beyond-EOF probe.
    async readTextFile() {
      return { content: "only-line" };
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, offset: 2, limit: 2 }),
    );
    assert.match(result.content, /offset 2 is beyond the last line of .* \(1 line\)/);
    assert.doesNotMatch(result.content, /only-line/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file final page reports end of file without a next-offset hint", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-final-page-"));
  const path = join(dir, "paged.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const last = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, offset: 7, limit: 4 })
    );
    assert.match(last.content, /line-10/);
    assert.match(last.content, /showing lines 7-10 of 10/);
    assert.match(last.content, /end of file/);
    // The hint must disappear once the page reaches EOF — advertising a next
    // offset there sent the model into an endless last-line loop.
    assert.doesNotMatch(last.content, /pass offset=/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file offset beyond EOF returns an EOF result without clamping or a hint", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-past-eof-"));
  const path = join(dir, "paged.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "read_file",
      JSON.stringify({ path, offset: 11, limit: 4 })
    );
    assert.match(result.content, /end of file/);
    assert.match(result.content, /offset 11 is beyond the last line/);
    // No clamp back into the final line, no next-page advertising.
    assert.doesNotMatch(result.content, /line-10/);
    assert.doesNotMatch(result.content, /pass offset=/);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "completed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file labels a bounded partial line without reporting an impossible range", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-partial-line-"));
  const path = join(dir, "partial.txt");
  writeFileSync(path, "abc\ndefgh", "utf8");
  const conn = createConnectionStub();
  const limits: ResourceLimits = {
    toolResultBytes: 262_144, fileReadBytes: 5, listEntries: 2000, listBytes: 262_144,
    fsConcurrency: 16,
  };
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir, () => "default", () => undefined, limits);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path, offset: 2, limit: 1 }));
    assert.match(result.content, /showing complete lines none/);
    assert.match(result.content, /line 2 is incomplete/);
    assert.doesNotMatch(result.content, /complete lines 2-1/);
    assert.match(result.content, /^d\n/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file elides the client content channel while the tool result stays full", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-elide-content-"));
  const path = join(dir, "long.txt");
  writeFileSync(path, `${"a".repeat(400)}\nsecond line`, "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    // The model receives the full page through the tool result...
    assert.ok(result.content.startsWith("a".repeat(400)));
    assert.match(result.content, /second line/);
    // ...but the client-facing content channel is elided.
    const completed = conn.updates.find(
      (u) =>
        (u.update as { sessionUpdate?: string }).sessionUpdate === "tool_call_update" &&
        Array.isArray((u.update as { content?: unknown[] }).content)
    ) as { update: { content: Array<{ content: { text: string } }> } };
    const text = completed.update.content[0]?.content.text ?? "";
    assert.match(text, /chars\]$/);
    assert.ok(text.length < 300);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file bounds a real large local result before it can enter model history", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-bounded-result-"));
  const path = join(dir, "large.txt");
  writeFileSync(path, "🙂".repeat(150_000), "utf8");
  const conn = createConnectionStub();
  const limits: ResourceLimits = {
    toolResultBytes: 128, fileReadBytes: 8 * 1024 * 1024, listEntries: 2000, listBytes: 262_144,
    fsConcurrency: 16,
  };
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir, () => "default", () => undefined, limits);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    assert.ok(Buffer.byteLength(result.content, "utf8") <= limits.toolResultBytes);
    assert.match(result.content, /bytes omitted/);
    assert.ok(!result.content.includes("\uFFFD"));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file success path emits in_progress and completed updates", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-success-"));
  const path = join(dir, "x.txt");
  writeFileSync(path, "hello", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    assert.equal(result.content, "hello");

    const sequence = conn.updates.map(
      (u) => ({
        type: (u.update as { sessionUpdate: string }).sessionUpdate,
        status: (u.update as { status?: string }).status,
      })
    );
    assert.deepEqual(sequence, [
      { type: "tool_call", status: "in_progress" },
      { type: "tool_call_update", status: "completed" },
    ]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file failure is reported with status=failed and an error message", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-read-fail-"));
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: join(dir, "missing.txt") }));
    assert.match(result.content, /Error reading file:/);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// write_file (permission flow)
// ---------------------------------------------------------------------------

test("write_file requests permission, then transitions through pending → in_progress → completed", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-success-"));
  const path = join(dir, "y.txt");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "data" })
    );
    assert.match(result.content, /written successfully/);
    assert.equal(readFileSync(path, "utf8"), "data");

    assert.equal(conn.permissionRequests.length, 1);
    const sequence = conn.updates.map((u) => ({
      type: (u.update as { sessionUpdate: string }).sessionUpdate,
      status: (u.update as { status?: string }).status,
    }));
    assert.deepEqual(sequence, [
      { type: "tool_call", status: "pending" },
      { type: "tool_call_update", status: "in_progress" },
      { type: "tool_call_update", status: "completed" },
    ]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file rejected by user marks call failed and skips writing", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-reject-"));
  const path = join(dir, "y.txt");
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "data" })
    );
    assert.match(result.content, /rejected by user/i);
    assert.equal(existsSync(path), false);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

for (const permission of ["missing", "unoffered", "unknown"] as const) {
  test(`unexpected ${permission} permission outcome never writes or executes`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-permission-invalid-"));
    try {
      const path = join(dir, "denied.txt");
      const conn = createConnectionStub({ permission });
      const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
      const write = await executor.execute("write", "write_file", JSON.stringify({ path, content: "secret" }));
      assert.match(write.content, /rejected/i);
      assert.equal(existsSync(path), false);
      const command = await executor.execute("command", "run_command", JSON.stringify({ command: "echo unsafe" }));
      assert.match(command.content, /rejected/i);
      assert.deepEqual(conn.terminalCalls, []);
      assert.equal(conn.permissionRequests.length, 2);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

test("session MCP notifications elide large content and aggregate small fields", async () => {
  const conn = createConnectionStub();
  const text = "x".repeat(50_000);
  const raw = { content: [{ type: "text", text }], metadata: Array.from({ length: 1_000 }, (_, i) => ({ key: i, value: "v".repeat(30) })) };
  const tools = { hasTool: () => true, callTool: async () => raw };
  const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, tools as never);
  const result = await executor.execute("mcp", "custom_tool", JSON.stringify({ fields: Array.from({ length: 1_000 }, (_, i) => ({ key: i })) }));
  assert.equal(result.content, text);
  for (const event of conn.updates) {
    assert.ok(Buffer.byteLength(JSON.stringify(event), "utf8") < 18_000);
  }
  const final = conn.updates.at(-1)?.update as { rawOutput: unknown; content: Array<{ content: { text: string } }> };
  assert.ok(final.content[0]?.content.text.length < 240);
  assert.notDeepEqual(final.rawOutput, raw);
});
test("failed tool cards bound both invalid arguments and upstream error bodies", async () => {
  const text = "z".repeat(50_000);
  const conn = createConnectionStub();
  const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const invalid = await executor.execute("invalid", "write_file", JSON.stringify({ path: "", content: text }));
  assert.match(invalid.content, /path.*required/i);
  const tools = { hasTool: () => true, callTool: async () => { throw new Error(text); } };
  const mcpExecutor = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, tools as never);
  const failed = await mcpExecutor.execute("error", "custom_tool", "{}");
  assert.match(failed.content, /Error calling MCP tool/);
  for (const event of conn.updates) {
    assert.ok(Buffer.byteLength(JSON.stringify(event), "utf8") < 18_000);
  }
});
test("tool-card titles and locations stay bounded without shortening approval payloads", async () => {
  const path = "p".repeat(50_000);
  const command = "echo " + "c".repeat(50_000);
  const conn = createConnectionStub({ permission: "reject" });
  const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  await executor.execute("read", "read_file", JSON.stringify({ path }));
  await executor.execute("write", "write_file", JSON.stringify({ path, content: "body" }));
  await executor.execute("run", "run_command", JSON.stringify({ command }));
  for (const event of conn.updates) {
    assert.ok(Buffer.byteLength(JSON.stringify(event), "utf8") < 18_000);
  }
  const first = conn.updates[0]?.update as { locations?: unknown[] };
  assert.deepEqual(first.locations, []);
  const writeApproval = conn.permissionRequests[0] as { toolCall: { title: string; rawInput: { path: string } } };
  assert.equal(writeApproval.toolCall.title, `Write file: ${path}`);
  assert.equal(writeApproval.toolCall.rawInput.path, path);
  const executeApproval = conn.permissionRequests[1] as { toolCall: { title: string; rawInput: { command: string } } };
  assert.equal(executeApproval.toolCall.title, `Run command: ${command}`);
  assert.equal(executeApproval.toolCall.rawInput.command, command);
});
test("write_file cancelled by user marks call failed", async () => {
  const conn = createConnectionStub({ permission: "cancelled" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/y.txt", content: "data" })
  );
  assert.match(result.content, /cancelled by user/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

// ---------------------------------------------------------------------------
// edit_file
// ---------------------------------------------------------------------------

test("edit_file replaces a unique snippet and goes through the permission flow", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-success-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "const a = 1;\nconst b = 2;\n", "utf8");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "const b = 2;", new_text: "const b = 3;" })
    );
    assert.match(result.content, /edited successfully/);
    assert.equal(readFileSync(path, "utf8"), "const a = 1;\nconst b = 3;\n");
    assert.deepEqual(
      conn.writeTextFileCalls.map((c) => ({ path: c.path, content: c.content })),
      [{ path, content: "const a = 1;\nconst b = 3;\n" }]
    );

    assert.equal(conn.permissionRequests.length, 1);
    const sequence = conn.updates.map((u) => ({
      type: (u.update as { sessionUpdate: string }).sessionUpdate,
      status: (u.update as { status?: string }).status,
    }));
    assert.deepEqual(sequence, [
      { type: "tool_call", status: "pending" },
      { type: "tool_call_update", status: "in_progress" },
      { type: "tool_call_update", status: "completed" },
    ]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file replaces a unique blank-line snippet through the permission flow", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-blank-lines-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "before\n\nafter\n", "utf8");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "\n\n", new_text: "\ninserted\n" })
    );

    assert.match(result.content, /edited successfully/);
    assert.equal(readFileSync(path, "utf8"), "before\ninserted\nafter\n");
    assert.equal(conn.permissionRequests.length, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file inserts replacement text literally when it contains replace tokens", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-literal-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "before needle after", "utf8");
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} });
  try {
    const newText = "$& $$ $' $`";
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "needle", new_text: newText })
    );
    assert.match(result.content, /edited successfully/);
    assert.equal(readFileSync(path, "utf8"), `before ${newText} after`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file does not mutate after the turn aborts while permission is pending", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-abort-permission-"));
  const path = join(dir, "out.txt");
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  let resolvePermission!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  let writeCalls = 0;
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<{ outcome: { outcome: "selected"; optionId: "allow" } }>((resolve) => {
        resolvePermission = () => resolve({ outcome: { outcome: "selected", optionId: "allow" } });
      });
    },
    async readTextFile() {
      return { content: readFileSync(path, "utf8") };
    },
    async writeTextFile() {
      writeCalls++;
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal);
  try {
    const pending = exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "must not write" })
    );
    await started;
    abortController.abort();
    resolvePermission();
    const result = await pending;
    assert.match(result.content, /cancelled by turn/i);
    assert.equal(writeCalls, 0);
    assert.equal(existsSync(path), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file does not mutate after the turn aborts while permission is pending", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-abort-permission-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "before", "utf8");
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  let resolvePermission!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  let writeCalls = 0;
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<{ outcome: { outcome: "selected"; optionId: "allow" } }>((resolve) => {
        resolvePermission = () => resolve({ outcome: { outcome: "selected", optionId: "allow" } });
      });
    },
    async readTextFile() {
      return { content: readFileSync(path, "utf8") };
    },
    async writeTextFile() {
      writeCalls++;
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal);
  try {
    const pending = exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "before", new_text: "after" })
    );
    await started;
    abortController.abort();
    resolvePermission();
    const result = await pending;
    assert.match(result.content, /cancelled by turn/i);
    assert.equal(writeCalls, 0);
    assert.equal(readFileSync(path, "utf8"), "before");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file settles when a pending permission request never responds", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-abort-never-responds-"));
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<never>(() => {});
    },
    async writeTextFile() {
      throw new Error("write must not start after abort");
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal);
  try {
    const pending = exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path: join(dir, "out.txt"), content: "must not write" })
    );
    await started;
    abortController.abort();
    const result = await pending;
    assert.match(result.content, /cancelled/i);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file fails fast when old_text is absent, without requesting permission", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-missing-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "hello", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "nope", new_text: "x" })
    );
    assert.match(result.content, /was not found/);
    assert.equal(readFileSync(path, "utf8"), "hello");
    assert.equal(conn.permissionRequests.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file refuses ambiguous old_text matches", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-ambiguous-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "return 1;\nreturn 1;\n", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "return 1;", new_text: "return 2;" })
    );
    assert.match(result.content, /occurs 2 times/);
    assert.equal(readFileSync(path, "utf8"), "return 1;\nreturn 1;\n");
    assert.equal(conn.permissionRequests.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file rejected by user leaves the file untouched", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-reject-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "before", "utf8");
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "before", new_text: "after" })
    );
    assert.match(result.content, /rejected by user/i);
    assert.equal(readFileSync(path, "utf8"), "before");
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file computes the edit against the client's buffer, not stale disk", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-client-read-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "const a = 1;\n", "utf8"); // stale disk content
  // The client's buffer has unsaved edits the agent must build on.
  const conn = createConnectionStub({ permission: "allow", clientFileContent: "const a = 1;\nconst b = 2;\n" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "const b = 2;", new_text: "const b = 3;" })
    );
    assert.match(result.content, /edited successfully/);
    // The write went back through the client with the merged content.
    assert.deepEqual(
      conn.writeTextFileCalls.map((c) => c.content),
      ["const a = 1;\nconst b = 3;\n"]
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file falls back to agent-process disk I/O without the writeTextFile capability", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-disk-fallback-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "alpha beta\n", "utf8");
  // readTextFile advertised but writeTextFile not: a client buffer we cannot
  // write back must not be the edit source either, so this stays disk↔disk.
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", { fs: { readTextFile: true } });
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "beta", new_text: "gamma" })
    );
    assert.match(result.content, /edited successfully/);
    assert.equal(readFileSync(path, "utf8"), "alpha gamma\n");
    assert.equal(conn.writeTextFileCalls.length, 0);
    assert.equal(conn.readTextFileCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("edit_file re-validates after the permission prompt and refuses a file changed underfoot", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-stale-"));
  const path = join(dir, "code.txt");
  writeFileSync(path, "keep\nold snippet\n", "utf8");
  const conn = createConnectionStub({
    permission: "allow",
    // The user edits the buffer while the permission prompt is up.
    onPermission: () => writeFileSync(path, "keep\nuser rewrote this\n", "utf8"),
  });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "edit_file",
      JSON.stringify({ path, old_text: "old snippet", new_text: "new snippet" })
    );
    assert.match(result.content, /changed while waiting for permission/);
    // The user's concurrent edit is intact and nothing was written back.
    assert.equal(readFileSync(path, "utf8"), "keep\nuser rewrote this\n");
    assert.equal(conn.writeTextFileCalls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file surfaces client writeTextFile failures as a failed tool result", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-write-client-fail-"));
  const path = join(dir, "y.txt");
  const conn = createConnectionStub({ permission: "allow", writeError: true });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "data" })
    );
    assert.match(result.content, /Error writing file: permission denied/);
    assert.equal(existsSync(path), false);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("todowrite stores the task list and renders it back to the model", async () => {
  const conn = createConnectionStub();
  const stored: Array<Array<{ content: string; status: string }>> = [];
  const exec = new ToolExecutor(
    conn as never,
    "s1",
    FULL_CAPS,
    undefined,
    null,
    null,
    process.cwd(),
    () => "default",
    (todos) => stored.push(todos)
  );
  const result = await exec.execute(
    "tc1",
    "todowrite",
    JSON.stringify({
      todos: [
        { content: "Reproduce the failure", status: "completed" },
        { content: "Patch the parser", status: "in_progress", active_form: "Patching the parser" },
        { content: "Add regression tests", status: "pending" },
      ],
    })
  );
  assert.match(result.content, /Todo list updated/);
  assert.match(result.content, /\[x\] Reproduce the failure/);
  assert.match(result.content, /\[>\] Patch the parser/);
  assert.match(result.content, /\[ \] Add regression tests/);
  assert.equal(stored.length, 1);
  assert.equal(stored[0]?.length, 3);

  const bad = await exec.execute(
    "tc2",
    "todowrite",
    JSON.stringify({ todos: [{ content: "x", status: "wat" }] })
  );
  assert.match(bad.content, /status.*must be one of/i);
  assert.equal(stored.length, 1);
});

test("write_file elides long content in client previews but writes the full file", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-preview-elide-"));
  const path = join(dir, "out.txt");
  const big = "x".repeat(5000);
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: big })
    );
    assert.match(result.content, /written successfully/);
    // The disk write is complete...
    assert.equal(readFileSync(path, "utf8"), big);
    // ...but the client-facing rawInput is elided.
    const announce = conn.updates.find(
      (u) => (u.update as { rawInput?: { content?: unknown } }).rawInput?.content !== undefined
    ) as { update: { rawInput: { content: string } } };
    assert.match(announce.update.rawInput.content, /5000 chars\]$/);
    assert.ok(announce.update.rawInput.content.length < 300);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("permission prompts receive the full payload while UI cards stay elided", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-permission-full-"));
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
  try {
    const big = "x".repeat(5000);
    await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path: join(dir, "written.txt"), content: big })
    );

    const editPath = join(dir, "editable.txt");
    writeFileSync(editPath, `${"y".repeat(400)} old ${"y".repeat(400)}`, "utf8");
    const replacement = "z".repeat(5000);
    await exec.execute(
      "tc2",
      "edit_file",
      JSON.stringify({ path: editPath, old_text: "old", new_text: replacement })
    );

    await exec.execute("tc3", "run_command", JSON.stringify({ command: "printf '%s' ok" }));

    // Every permission request carries the complete, unelided payload — the
    // user approves exactly what will run, never a truncated prefix.
    assert.equal(conn.permissionRequests.length, 3);
    const [writeReq, editReq, runReq] = conn.permissionRequests as Array<{
      toolCall: { rawInput: Record<string, unknown> };
    }>;
    assert.equal(writeReq.toolCall.rawInput["content"], big);
    assert.equal(editReq.toolCall.rawInput["new_text"], replacement);
    assert.equal(editReq.toolCall.rawInput["old_text"], "old");
    assert.equal(runReq.toolCall.rawInput["command"], "printf '%s' ok");

    // The sessionUpdate announcement cards, in contrast, stay elided.
    const announce = conn.updates.find(
      (u) => (u.update as { rawInput?: { content?: unknown } }).rawInput?.content !== undefined
    ) as { update: { rawInput: { content: string } } };
    assert.match(announce.update.rawInput.content, /5000 chars\]$/);
    assert.ok(announce.update.rawInput.content.length < 300);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// run_command
// ---------------------------------------------------------------------------

test("run_command rejects empty input", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "   " })
  );
  assert.match(result.content, /non-empty string/);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("run_command runs through sh -c so quoting/pipes work", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-cmd-"));
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf 'mixed' | tr a-z A-Z && pwd" })
  );
  assert.match(result.content, /Exit code: 0/);
  assert.match(result.content, /MIXED/);
  assert.match(result.content, new RegExp(escapeRegExp(basename(dir))));
  assert.equal(conn.terminalCalls.length, 0);
  rmSync(dir, { recursive: true, force: true });
});

test("run_command includes stderr and non-zero exit code in the tool result", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf 'bad' >&2; exit 7" })
  );
  assert.match(result.content, /Exit code: 7/);
  assert.match(result.content, /STDERR:\nbad/);
  const last = conn.updates.at(-1) as {
    update: { status?: string; rawOutput?: { exitCode?: number; stderr?: string } };
  };
  assert.equal(last.update.status, "completed");
  assert.equal(last.update.rawOutput?.exitCode, 7);
  assert.equal(last.update.rawOutput?.stderr, "bad");
});

test(
  "run_command returns promptly when a script spawns a background process (#67)",
  { timeout: 10_000 },
  async () => {
    // Regression test for #67, fixed by PR #66.
    // A backgrounded process (`&`, `nohup`, `disown`) inherits the stdout/stderr
    // pipe FDs and holds them open after the shell exits; resolving on "close"
    // hung forever. The fix detaches the child and force-destroys the streams
    // shortly after the shell exits. Without the fix this test never resolves
    // and fails via the test timeout.
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);

    const start = Date.now();
    const result = await exec.execute(
      "tc1",
      "run_command",
      // `sleep 3` outlives the shell and keeps the inherited pipes open.
      JSON.stringify({ command: "sleep 3 & echo spawned; exit 0" })
    );
    const elapsed = Date.now() - start;

    assert.match(result.content, /Exit code: 0/);
    assert.match(result.content, /STDOUT:\nspawned/);
    // Must return well before the 3s background sleep finishes — proves we did
    // not block on the inherited pipe FDs.
    assert.ok(elapsed < 2000, `run_command took ${elapsed}ms; expected < 2000ms`);

    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "completed");
  }
);

test(
  "run_command abort terminates foreground descendants in the shell process group",
  { timeout: 5_000 },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-cmd-abort-tree-"));
    const ready = join(dir, "ready");
    const marker = join(dir, "should-not-exist");
    const abortController = new AbortController();
    const conn = createConnectionStub();
    const exec = new ToolExecutor(
      conn as never,
      "s1",
      FULL_CAPS,
      abortController.signal,
      null,
      null,
      dir
    );
    try {
      const command = `${shellNodeCommand()} -e 'const fs = require("node:fs"); process.on("SIGTERM", () => {}); fs.writeFileSync(${shellFixturePath(ready, "ready")}, "ready"); setTimeout(() => fs.writeFileSync(${shellFixturePath(marker, "should-not-exist")}, "late"), 1000)' ; echo shell-finished`;
      const pending = exec.execute("tc1", "run_command", JSON.stringify({ command }));
      const deadline = Date.now() + 1_000;
      while (!existsSync(ready) && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 10));
      }
      assert.equal(existsSync(ready), true, "foreground child never reached the ready point");
      abortController.abort();
      const result = await pending;
      assert.match(result.content, /cancelled|signal|exit code/i);
      await new Promise((resolve) => setTimeout(resolve, 1_400));
      assert.equal(existsSync(marker), false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
);

const RESISTANT_ABORT_HELPER = `
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [executorModule, cwd, ready, marker] = process.argv.slice(2);
const { ToolExecutor } = await import(pathToFileURL(executorModule).href);
const connection = {
  async sessionUpdate() {},
  async requestPermission() {
    return { outcome: { outcome: "selected", optionId: "allow" } };
  },
};
const abortController = new AbortController();
const exec = new ToolExecutor(connection, "s1", { fs: {} }, abortController.signal, null, null, cwd);
const quote = (value) => "'" + value.replaceAll("'", "'\\\\''") + "'";
const shellNode = process.platform === "win32" ? "node" : quote(process.execPath);
const shellReady = process.platform === "win32" ? "ready" : ready;
const shellMarker = process.platform === "win32" ? "should-not-exist" : marker;
const command = shellNode + " -e 'const fs = require(\\"node:fs\\"); process.on(\\"SIGTERM\\", () => {}); fs.writeFileSync(" + JSON.stringify(shellReady) + ", \\"ready\\"); setTimeout(() => fs.writeFileSync(" + JSON.stringify(shellMarker) + ", \\"late\\"), 700)' ; echo shell-finished";
const pending = exec.execute("tc1", "run_command", JSON.stringify({ command }));
const deadline = Date.now() + 1000;
while (!existsSync(ready) && Date.now() < deadline) {
  await new Promise((resolve) => setTimeout(resolve, 5));
}
if (!existsSync(ready)) process.exit(2);
abortController.abort();
await pending;
process.stdout.write("SETTLED\\n");
`;

test(
  "run_command keeps cancellation escalation alive after the shell closes",
  { timeout: 5_000 },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-cmd-abort-escalation-"));
    const ready = join(dir, "ready");
    const marker = join(dir, "should-not-exist");
    const helperPath = join(dir, "abort-probe.mjs");
    writeFileSync(helperPath, RESISTANT_ABORT_HELPER, "utf8");
    const executorModule = fileURLToPath(new URL("../tools/executor.js", import.meta.url));
    try {
      const output = execFileSync(
        process.execPath,
        [helperPath, executorModule, dir, ready, marker],
        { encoding: "utf8", timeout: 4_000 }
      );
      assert.match(output, /SETTLED/);
      await new Promise((resolve) => setTimeout(resolve, 900));
      assert.equal(existsSync(marker), false);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
);

test(
  "run_command leaves intentionally backgrounded processes alive after normal shell exit",
  { timeout: 5_000 },
  async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-cmd-background-survival-"));
    const marker = join(dir, "background-finished");
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
    try {
      const command = `${shellNodeCommand()} -e 'setTimeout(() => require("node:fs").writeFileSync(${shellFixturePath(marker, "background-finished")}, "done"), 250)' >/dev/null 2>&1 & echo started`;
      const result = await exec.execute("tc1", "run_command", JSON.stringify({ command }));
      assert.match(result.content, /Exit code: 0/);
      const deadline = Date.now() + 2_500;
      while (!existsSync(marker) && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 20));
      }
      assert.equal(readFileSync(marker, "utf8"), "done");
    } finally {
      await removeTestDirectory(dir);
    }
  }
);

test(
  "isProcessGroupAlive reports live groups as alive and exited groups as gone",
  { timeout: 5_000 },
  async () => {
    if (process.platform === "win32") {
      // No POSIX process groups on Windows: the probe deliberately reports
      // "alive" so the escalation timer stays armed (taskkill on a dead pid
      // is a harmless no-op). Only the missing-pid case reports "gone".
      assert.equal(isProcessGroupAlive(12345), true, "win32 probe must stay permissive");
      assert.equal(isProcessGroupAlive(undefined), false, "missing pid reported alive");
      return;
    }
    const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 800)"], {
      detached: true,
      stdio: "ignore",
    });
    const exited = new Promise<void>((resolve) => child.on("exit", () => resolve()));
    assert.ok(child.pid, "detached child never started");
    try {
      await new Promise((resolve) => setTimeout(resolve, 100));
      assert.equal(isProcessGroupAlive(child.pid), true, "live group reported dead");
    } finally {
      try {
        process.kill(-child.pid!, "SIGKILL");
      } catch {
        /* already gone */
      }
    }
    await exited;
    assert.equal(isProcessGroupAlive(child.pid), false, "exited group reported alive");
    assert.equal(isProcessGroupAlive(undefined), false, "missing pid reported alive");
  }
);

// The helpers run serially inside one test, so the test timeout must exceed
// attempts × per-helper budget — otherwise a loaded runner can kill the test
// while every helper is still individually within budget (greptile, PR #85).
const PROBE_ATTEMPTS = 8;
const PROBE_HELPER_TIMEOUT_MS = 8_000;

// Runs inside a bare node process whose only pending work is a sequence of
// run_command calls: with an unref'd child the event loop can drain mid-await
// on fast commands and the process exits with the promise still unsettled.
const LOOP_PROBE_HELPER = `
import { pathToFileURL } from "node:url";
const [executorModule, cwd] = process.argv.slice(2);
const { ToolExecutor } = await import(pathToFileURL(executorModule).href);
const connection = {
  async sessionUpdate() {},
  async requestPermission() {
    return { outcome: { outcome: "selected", optionId: "allow" } };
  },
};
const exec = new ToolExecutor(connection, "s1", { fs: {} }, undefined, null, null, cwd);
for (let i = 1; i <= 15; i++) {
  const result = await exec.execute(
    "tc" + i,
    "run_command",
    JSON.stringify({ command: "echo loop-alive-" + i })
  );
  if (!result.content.includes("loop-alive-" + i)) {
    process.stdout.write("BAD OUTPUT AT " + i + "\\n");
    process.exit(1);
  }
}
process.stdout.write("SETTLED 15\\n");
`;

test(
  "run_command keeps the event loop alive until the command settles (#82)",
  { timeout: PROBE_ATTEMPTS * PROBE_HELPER_TIMEOUT_MS + 5_000 },
  async () => {
    // Regression test for #82. runShellCommand unref()ed the sh -c child, so a
    // node:test file process with no other pending work could drain its event
    // loop while a fast run_command was still in flight. The runner then
    // cancelled every remaining test in the file with "Promise resolution is
    // still pending but the event loop has already resolved" — flaky,
    // load-dependent blocks of 6–30 cancelled subtests per run. The unfixed
    // race is per command (~40% observed), so several bare helper processes
    // are probed; each must settle all of its tool calls rather than exit
    // early. With the child ref'd every process settles deterministically.
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-loop-"));
    try {
      const helperPath = join(dir, "loop-probe.mjs");
      writeFileSync(helperPath, LOOP_PROBE_HELPER, "utf8");
      const executorModule = fileURLToPath(new URL("../tools/executor.js", import.meta.url));
      for (let attempt = 1; attempt <= PROBE_ATTEMPTS; attempt++) {
        let stdout = "";
        try {
          stdout = execFileSync(process.execPath, [helperPath, executorModule, dir], {
            encoding: "utf8",
            timeout: PROBE_HELPER_TIMEOUT_MS,
          });
        } catch (err) {
          assert.fail(
            `helper ${attempt}/${PROBE_ATTEMPTS} exited before the tool calls settled: ` +
              `${(err as Error).message}`
          );
        }
        assert.match(stdout, /SETTLED 15/, `helper ${attempt}/${PROBE_ATTEMPTS} output`);
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
);

test("run_command rejected by user marks call failed and skips execution", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf should-not-run" })
  );
  assert.match(result.content, /rejected by user/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
  assert.equal(conn.terminalCalls.length, 0);
});

test("run_command cancelled by user marks call failed and skips execution", async () => {
  const conn = createConnectionStub({ permission: "cancelled" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf should-not-run" })
  );
  assert.match(result.content, /cancelled by user/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
  assert.equal(conn.terminalCalls.length, 0);
});

test("run_command turn abort during permission prompt reports cancelled by turn", async () => {
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    terminalCalls: [] as Array<{ command: string; args?: string[] }>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<never>(() => {});
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal);
  const pending = exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf should-not-run" })
  );
  await started;
  abortController.abort();
  const result = await pending;
  assert.match(result.content, /cancelled by turn/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
  assert.equal(conn.terminalCalls.length, 0);
});

// ---------------------------------------------------------------------------
// list_files
// ---------------------------------------------------------------------------

test("list_files resolves relative paths against the session cwd", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-list-"));
  writeFileSync(join(dir, "with space.txt"), "data", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
  const result = await exec.execute(
    "tc1",
    "list_files",
    JSON.stringify({ path: "." })
  );
  assert.match(result.content, /with space\.txt/);
  assert.equal(conn.terminalCalls.length, 0);
  rmSync(dir, { recursive: true, force: true });
});

test("list_files rejects empty path", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "list_files", JSON.stringify({ path: "" }));
  assert.match(result.content, /non-empty string/);
});

test("list_files caps an oversized empty-directory header and marks it truncated", async () => {
  const dir = mkdtempSync(join(tmpdir(), `glm-executor-list-header-${"x".repeat(90)}-`));
  const conn = createConnectionStub();
  const limits: ResourceLimits = {
    toolResultBytes: 262_144, fileReadBytes: 8 * 1024 * 1024, listEntries: 2000, listBytes: 128,
    fsConcurrency: 16,
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "default", () => undefined, limits);
  try {
    const result = await exec.execute("tc1", "list_files", JSON.stringify({ path: "." }));
    assert.ok(Buffer.byteLength(result.content, "utf8") <= limits.listBytes);
    assert.match(result.content, /\[listing truncated:/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// web_search / web_reader via Z.AI Coding Plan MCP
// ---------------------------------------------------------------------------

test("web_search uses stored credentials and calls the Coding Plan MCP search tool", async () => {
  await withStoredApiKey(async () => {
    await withMockedFetch(
      [
        jsonResponse(
          {
            jsonrpc: "2.0",
            id: 1,
            result: { protocolVersion: "2025-06-18", capabilities: {} },
          },
          { sessionId: "search-session" }
        ),
        new Response(null, { status: 202 }),
        jsonResponse({
          jsonrpc: "2.0",
          id: 2,
          result: { tools: [{ name: "webSearchPrime", inputSchema: { properties: { search_query: { type: "string" } } } }] },
        }),
        jsonResponse({
          jsonrpc: "2.0",
          id: 3,
          result: {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  search_result: [
                    {
                      title: "GLM Coding Plan",
                      link: "https://z.ai/",
                      media: "Z.AI",
                      publish_date: "2026-04-29",
                      content: "MCP quota path",
                    },
                  ],
                }),
              },
            ],
          },
        }),
      ],
      async (calls) => {
        const conn = createConnectionStub();
        const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
        const result = await exec.execute(
          "tc-web-search",
          "web_search",
          JSON.stringify({ query: "glm coding plan", count: 1 })
        );

        assert.match(result.content, /\[1\] GLM Coding Plan/);
        assert.match(result.content, /URL: https:\/\/z\.ai\//);
        assert.equal(calls.length, 4);
        assert.ok(calls.every((call) => call.url === "https://api.z.ai/api/mcp/web_search_prime/mcp"));
        assert.equal(calls[0]?.headers.get("Authorization"), "Bearer from-disk");
        assert.equal(calls[3]?.headers.get("Mcp-Method"), "tools/call");
        assert.equal(calls[3]?.headers.get("Mcp-Name"), "webSearchPrime");
        assert.deepEqual(calls[3]?.body.params, {
          name: "webSearchPrime",
          arguments: { search_query: "glm coding plan", count: 1 },
        });
        const last = conn.updates.at(-1) as { update: { status?: string } };
        assert.equal(last.update.status, "completed");
      }
    );
  });
});

test("web_reader calls the Coding Plan MCP reader tool and formats reader_result", async () => {
  const oldEnv = process.env["Z_AI_API_KEY"];
  try {
    process.env["Z_AI_API_KEY"] = "from-env";
    await withMockedFetch(
      [
        jsonResponse({
          jsonrpc: "2.0",
          id: 1,
          result: { protocolVersion: "2025-06-18", capabilities: {} },
        }),
        new Response(null, { status: 202 }),
        jsonResponse({
          jsonrpc: "2.0",
          id: 2,
          result: { tools: [{ name: "webReader" }] },
        }),
        jsonResponse({
          jsonrpc: "2.0",
          id: 3,
          result: {
            content: [
              {
                type: "text",
                text: JSON.stringify({
                  reader_result: {
                    title: "Example",
                    url: "https://example.com/",
                    description: "Short description",
                    content: "Main body",
                  },
                }),
              },
            ],
          },
        }),
      ],
      async (calls) => {
        const conn = createConnectionStub();
        const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
        const result = await exec.execute(
          "tc-web-reader",
          "web_reader",
          JSON.stringify({ url: "https://example.com/", return_format: "markdown" })
        );

        assert.match(result.content, /^# Example/);
        assert.match(result.content, /URL: https:\/\/example\.com\//);
        assert.match(result.content, /Main body/);
        assert.ok(calls.every((call) => call.url === "https://api.z.ai/api/mcp/web_reader/mcp"));
        assert.equal(calls[3]?.headers.get("Mcp-Name"), "webReader");
        assert.deepEqual(calls[3]?.body.params, {
          name: "webReader",
          arguments: { url: "https://example.com/", return_format: "markdown" },
        });
      }
    );
  } finally {
    if (oldEnv === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = oldEnv;
  }
});

test("web_search reports Coding Plan 1113 MCP errors as actionable failed tool results", async () => {
  const oldEnv = process.env["Z_AI_API_KEY"];
  try {
    process.env["Z_AI_API_KEY"] = "from-env";
    await withMockedFetch(
      [
        jsonResponse(
          { error: { code: "1113", message: "No permission for current API key" } },
          { status: 429 }
        ),
      ],
      async () => {
        const conn = createConnectionStub();
        const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
        const result = await exec.execute(
          "tc-web-search-error",
          "web_search",
          JSON.stringify({ query: "glm" })
        );

        assert.match(result.content, /Coding Plan quota\/base URL\/tool eligibility/);
        assert.match(result.content, /1113/);
        const last = conn.updates.at(-1) as { update: { status?: string } };
        assert.equal(last.update.status, "failed");
      }
    );
  } finally {
    if (oldEnv === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = oldEnv;
  }
});

// ---------------------------------------------------------------------------
// Permission transport errors
// ---------------------------------------------------------------------------

test("write_file converts requestPermission transport errors into a failed tool result", async () => {
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) {
      this.updates.push(payload);
    },
    async writeTextFile() {
      throw new Error("should not be called");
    },
    async requestPermission(): Promise<never> {
      throw new Error("connection lost");
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/y.txt", content: "data" })
  );
  assert.match(result.content, /requesting permission.*connection lost/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("run_command converts requestPermission transport errors into a failed tool result", async () => {
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) {
      this.updates.push(payload);
    },
    async createTerminal() {
      throw new Error("should not be called");
    },
    async requestPermission(): Promise<never> {
      throw new Error("connection lost");
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "echo hi" })
  );
  assert.match(result.content, /requesting permission.*connection lost/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

// ---------------------------------------------------------------------------
// Session mode permission behavior
// ---------------------------------------------------------------------------

test("write_file prompts for permission in default mode", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, undefined, () => "default");
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/y.txt", content: "data" })
  );
  assert.equal(result.content, "Write rejected by user.");
  assert.equal(conn.permissionRequests.length, 1);
});

test("write_file skips permission prompt in accept_edits mode", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-accept-edits-"));
  const path = join(dir, "out.txt");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "accept_edits");
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "hi" })
    );
    // In accept_edits mode, writes should NOT request permission
    assert.equal(conn.permissionRequests.length, 0);
    // The write should succeed
    assert.match(result.content, /written successfully/);
    assert.equal(readFileSync(path, "utf8"), "hi");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("write_file skips permission prompt in bypass_permissions mode", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-executor-bypass-"));
  const path = join(dir, "out.txt");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "bypass_permissions");
  try {
    const result = await exec.execute(
      "tc1",
      "write_file",
      JSON.stringify({ path, content: "hi" })
    );
    assert.equal(conn.permissionRequests.length, 0);
    assert.match(result.content, /written successfully/);
    assert.equal(readFileSync(path, "utf8"), "hi");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("run_command prompts for permission in default mode", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, undefined, () => "default");
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "echo hi" })
  );
  assert.equal(result.content, "Command rejected by user.");
  assert.equal(conn.permissionRequests.length, 1);
});

test("run_command prompts for permission in accept_edits mode", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, undefined, () => "accept_edits");
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "echo hi" })
  );
  assert.equal(result.content, "Command rejected by user.");
  // In accept_edits mode, commands should still prompt
  assert.equal(conn.permissionRequests.length, 1);
});

test("run_command skips permission prompt in bypass_permissions mode", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, undefined, () => "bypass_permissions");
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "echo hi" })
  );
  assert.equal(conn.permissionRequests.length, 0);
  assert.match(result.content, /Exit code: 0/);
  assert.match(result.content, /STDOUT:\nhi/);
});

// ---------------------------------------------------------------------------
// Whitespace path normalization
// ---------------------------------------------------------------------------

test("read_file rejects whitespace-only paths", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "   " }));
  assert.match(result.content, /path.*required/);
});

test("write_file rejects whitespace-only paths", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: " \t\n", content: "x" })
  );
  assert.match(result.content, /path.*required/);
});

// ---------------------------------------------------------------------------
// image_analysis (Vision MCP)
// ---------------------------------------------------------------------------

function fakeVisionClient(impl: VisionMcpClient["callTool"]): VisionMcpClient {
  return {
    callTool: impl,
    async dispose() { /* noop */ },
  };
}

test("image_analysis routes through the injected vision client and returns the text", async () => {
  const conn = createConnectionStub();
  const vision = fakeVisionClient(async (toolName, args) => {
    assert.equal(toolName, "image_analysis");
    assert.equal(args["image_source"], "/tmp/cat.png");
    return { content: [{ type: "text", text: "A tabby cat." }] };
  });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const result = await exec.execute(
    "tc1",
    "image_analysis",
    JSON.stringify({ image_source: "/tmp/cat.png", prompt: "describe" })
  );
  assert.equal(result.content, "A tabby cat.");
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "completed");
});

test("image_analysis surfaces vision errors as a failed tool result", async () => {
  const conn = createConnectionStub();
  const vision = fakeVisionClient(async () => {
    throw new Error("Vision MCP image_analysis failed: quota exceeded");
  });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const result = await exec.execute(
    "tc1",
    "image_analysis",
    JSON.stringify({ image_source: "/tmp/x.png" })
  );
  assert.match(result.content, /quota exceeded/);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("image_analysis is unavailable when no vision client is configured", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "image_analysis",
    JSON.stringify({ image_source: "/tmp/x.png" })
  );
  assert.match(result.content, /vision[^.]*not configured/i);
});
