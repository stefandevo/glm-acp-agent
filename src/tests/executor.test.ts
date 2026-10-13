import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { getEventListeners } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import fsPromises from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import { realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import { writeCredentials } from "../llm/credentials.js";
import { takeUtf8Prefix } from "../tools/tool-output.js";
import { ToolExecutor, formatDirectoryListing, isProcessGroupAlive } from "../tools/executor.js";
import type { ResourceLimits } from "../tools/resource-limits.js";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

interface StubTerminal {
  id: string;
  waitForExit: () => Promise<{ exitCode: number }>;
  currentOutput: () => Promise<{ output: string }>;
  release: () => Promise<void>;
}

function createConnectionStub(opts: {
  permission?: "allow" | "reject" | "cancelled" | "missing" | "unoffered" | "unknown" | "null" | "empty" | "nonstring";
  permissionError?: boolean;
  readError?: boolean;
  writeError?: boolean;
  terminalOutput?: string;
  /** When set, client readTextFile returns this instead of the on-disk content (simulates a dirty buffer). */
  clientFileContent?: string;
  /**
   * Live editor buffer: when set, every client readTextFile serves this
   * object's current `content` (NOT disk), so tests can mutate the unsaved
   * buffer independently of the file on disk.
   */
  clientBuffer?: { content: string };
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
      if (opts.clientBuffer) return { content: opts.clientBuffer.content };
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
      if (opts.permissionError) throw new Error("permission transport failed");
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
        case "null":
          return null;
        case "empty":
          return {};
        case "nonstring":
          return { outcome: { outcome: "selected", optionId: 42 } };
      }
    },
  };
}

const FULL_CAPS = {
  fs: { readTextFile: true, writeTextFile: true },
  terminal: true,
};

function deferredRead<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

async function withinReadDeadline<T>(promise: Promise<T>): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("read did not settle within test deadline")), 2_000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Remove a tempdir that spawned children ran in. On Windows a just-exited
 * child's cwd handle (or an antivirus scan) can hold the directory for a
 * moment, so a bare rmSync intermittently fails with EBUSY/ENOTEMPTY — retry
 * briefly before giving up.
 */
function rmTempDir(dir: string): void {
  for (let attempt = 1; ; attempt += 1) {
    try {
      rmSync(dir, { recursive: true, force: true });
      return;
    } catch (err) {
      const code = (err as NodeJS.ErrnoException).code;
      if (attempt >= 5 || (code !== "EBUSY" && code !== "ENOTEMPTY" && code !== "EPERM")) {
        throw err;
      }
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 50 * attempt);
    }
  }
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

type FetchResponse = Response | ((request: Record<string, unknown>) => Response);

function createFetchStub(responses: FetchResponse[]) {
  const calls: FetchCall[] = [];
  const fetchStub = async (url: string | URL | Request, init?: RequestInit) => {
    assert.ok(init, "fetch init is required");
    const body =
      typeof init.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : {};
    const headers = new Headers(init.headers);
    calls.push({ url: String(url), init, body, headers });
    const response = responses.shift();
    assert.ok(response, "unexpected fetch call");
    return typeof response === "function" ? response(body) : response;
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
  responses: FetchResponse[],
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-")));
  const path = join(dir, "note.txt");
  writeFileSync(path, "from disk", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    assert.equal(result.content, "from disk");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file reads the client's unsaved buffer when both fs capabilities are available", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-client-")));
  const path = join(dir, "note.txt");
  writeFileSync(path, "stale disk", "utf8");
  const conn = createConnectionStub({ clientFileContent: "unsaved buffer" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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

test("read_file rejects paths and symlinks that escape the session workspace", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-confined-")));
  const outsidePath = `${dir}-secret.txt`;
  try {
    writeFileSync(outsidePath, "secret outside the workspace", "utf8");
    symlinkSync(outsidePath, join(dir, "escape.txt"), "file");
    for (const caps of [{ fs: {} }, { fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub({ clientFileContent: "must not read" });
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, null, null, dir);
      for (const path of [outsidePath, join("..", basename(outsidePath)), "escape.txt"]) {
        const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
        assert.match(result.content, /path must remain inside the session workspace/);
        assert.equal(conn.readTextFileCalls.length, 0);
        const last = conn.updates.at(-1) as { update: { status?: string } };
        assert.equal(last.update.status, "failed");
      }
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outsidePath, { force: true });
  }
});

test("read_file rejects a missing editor leaf beneath an existing outside parent", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-read-missing-leaf-workspace-")));
  const outside = realpathSync(mkdtempSync(join(tmpdir(), "glm-read-missing-leaf-outside-")));
  const path = join(outside, "unsaved.txt");
  try {
    assert.equal(existsSync(path), false);
    for (const caps of [{ fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub({ clientFileContent: "must not read outside buffer" });
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, null, null, dir);
      const result = await exec.execute("outside-leaf", "read_file", JSON.stringify({ path }));
      assert.match(result.content, /path must remain inside the session workspace/);
      assert.equal(conn.readTextFileCalls.length, 0);
      const last = conn.updates.at(-1) as { update: { status?: string } };
      assert.equal(last.update.status, "failed");
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  }
});

test("read_file accepts absolute paths that resolve inside the workspace", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-absolute-")));
  const path = join(dir, "note.txt");
  try {
    writeFileSync(path, "workspace disk content", "utf8");
    for (const caps of [{ fs: {} }, FULL_CAPS]) {
      const conn = createConnectionStub({ clientFileContent: "workspace buffer content" });
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, null, null, dir);
      const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
      assert.equal(result.content, caps === FULL_CAPS ? "workspace buffer content" : "workspace disk content");
      assert.equal(conn.readTextFileCalls.length, caps === FULL_CAPS ? 1 : 0);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file uses a read-only client's unsaved buffer and preserves pagination", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-only-client-")));
  const path = join(dir, "note.txt");
  try {
    writeFileSync(path, "stale disk content", "utf8");
    const conn = createConnectionStub({ clientFileContent: "buffer-1\nbuffer-2\nbuffer-3\nbuffer-4" });
    const exec = new ToolExecutor(conn as never, "s1", { fs: { readTextFile: true } }, undefined, null, null, dir);
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path, offset: 2, limit: 2 }));
    assert.ok(result.content.startsWith("buffer-2\nbuffer-3\n"));
    assert.match(result.content, /pass offset=4/);
    assert.doesNotMatch(result.content, /buffer-1|buffer-4|stale disk content/);
    assert.ok(conn.readTextFileCalls.length > 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file preserves a symlink workspace path when reading an editor buffer", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-workspace-alias-")));
  const alias = `${dir}-alias`;
  try {
    writeFileSync(join(dir, "note.txt"), "stale disk content", "utf8");
    symlinkSync(dir, alias, process.platform === "win32" ? "junction" : "dir");
    const conn = createConnectionStub();
    conn.readTextFile = async (params) => {
      assert.equal(params.path, join(alias, "note.txt"));
      conn.readTextFileCalls.push(params);
      return { content: "buffer through workspace alias" };
    };
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, alias);
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "note.txt" }));
    assert.equal(result.content, "buffer through workspace alias");
    assert.equal(conn.readTextFileCalls.length, 1);
  } finally {
    rmSync(alias, { recursive: true, force: true });
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file reads a new editor buffer whose disk file does not exist yet", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-new-buffer-")));
  const parent = join(dir, "buffers");
  try {
    mkdirSync(parent);
    const conn = createConnectionStub({ clientFileContent: "new unsaved buffer" });
    const exec = new ToolExecutor(conn as never, "s1", { fs: { readTextFile: true } }, undefined, null, null, dir);
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: join(parent, "new.txt") }));
    assert.equal(result.content, "new unsaved buffer");
    assert.equal(conn.readTextFileCalls.length, 1);

    const danglingPath = join(parent, "dangling.txt");
    symlinkSync(join(dir, "missing-target.txt"), danglingPath, "file");
    const deniedConn = createConnectionStub({ clientFileContent: "must not read" });
    const deniedExec = new ToolExecutor(deniedConn as never, "s1", { fs: { readTextFile: true } }, undefined, null, null, dir);
    const denied = await deniedExec.execute("tc2", "read_file", JSON.stringify({ path: danglingPath }));
    assert.match(denied.content, /^Error reading file:/);
    assert.match(denied.content, /no such file/i);
    assert.equal(deniedConn.readTextFileCalls.length, 0);
    const last = deniedConn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("read_file follows an in-workspace symlink to another workspace file", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-internal-symlink-")));
  try {
    const target = join(dir, "target.txt");
    writeFileSync(target, "in-workspace symlink content", "utf8");
    symlinkSync(target, join(dir, "linked.txt"), "file");
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir);
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "linked.txt" }));
    assert.equal(result.content, "in-workspace symlink content");
    assert.equal(conn.readTextFileCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

for (const caps of [{ fs: { readTextFile: true } }, FULL_CAPS]) {
  const clientKind = caps === FULL_CAPS ? "read+write" : "read-only";

  test(`read_file cancellation during confinement prevents client reads (${clientKind})`, async () => {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-read-abort-confinement-")));
    const controller = new AbortController();
    const held = deferredRead<void>();
    const started = deferredRead<void>();
    const conn = createConnectionStub({ clientFileContent: "late buffer contents" });
    const exec = new ToolExecutor(conn as never, "s1", caps, controller.signal, null, null, dir);
    const paths = exec as unknown as {
      resolveConfinedReadPath(path: string, options?: { clientRead: boolean }): Promise<string>;
    };
    const resolvePath = paths.resolveConfinedReadPath.bind(exec);
    paths.resolveConfinedReadPath = async (path, options) => {
      started.resolve();
      await held.promise;
      return resolvePath(path, options);
    };
    const pending = exec.execute("read-abort-path", "read_file", JSON.stringify({ path: "buffer.txt" }));
    try {
      await withinReadDeadline(started.promise);
      controller.abort();
      held.resolve();
      const result = await withinReadDeadline(pending);
      assert.match(result.content, /The operation was aborted/);
      assert.equal(conn.readTextFileCalls.length, 0);
      const statuses = conn.updates.map(update => (update.update as { status?: string }).status);
      assert.deepEqual(statuses, ["in_progress", "failed"]);
      assert.doesNotMatch(JSON.stringify(conn.updates), /late buffer contents/);
    } finally {
      controller.abort();
      held.resolve();
      await withinReadDeadline(Promise.allSettled([pending]));
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test(`read_file cancellation in the first editor page prevents the EOF probe (${clientKind})`, async () => {
    const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-read-abort-first-page-")));
    const controller = new AbortController();
    const requestedLines: Array<number | undefined> = [];
    const conn = createConnectionStub();
    conn.readTextFile = async (params: { sessionId: string; path: string; line?: number }) => {
      conn.readTextFileCalls.push(params);
      requestedLines.push(params.line);
      controller.abort();
      return { content: "late first-page content" };
    };
    const exec = new ToolExecutor(conn as never, "s1", caps, controller.signal, null, null, dir);
    try {
      const result = await withinReadDeadline(exec.execute("read-abort-page", "read_file", JSON.stringify({
        path: "buffer.txt", offset: 2, limit: 2,
      })));
      assert.match(result.content, /The operation was aborted/);
      assert.deepEqual(requestedLines, [2]);
      assert.equal(getEventListeners(controller.signal, "abort").length, 0);
      const statuses = conn.updates.map(update => (update.update as { status?: string }).status);
      assert.deepEqual(statuses, ["in_progress", "failed"]);
      assert.doesNotMatch(JSON.stringify(conn.updates), /late first-page content/);
    } finally {
      controller.abort();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  for (const lateOutcome of ["resolve", "reject"] as const) {
    test(`read_file cancellation settles a pending editor read and ignores late ${lateOutcome} (${clientKind})`, async () => {
      const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-read-abort-pending-")));
      const controller = new AbortController();
      const held = deferredRead<{ content: string }>();
      const started = deferredRead<void>();
      const unhandled: unknown[] = [];
      const onUnhandled = (reason: unknown) => { unhandled.push(reason); };
      process.on("unhandledRejection", onUnhandled);
      const conn = createConnectionStub();
      conn.readTextFile = params => {
        conn.readTextFileCalls.push(params);
        started.resolve();
        return held.promise;
      };
      const exec = new ToolExecutor(conn as never, "s1", caps, controller.signal, null, null, dir);
      const pending = exec.execute("read-abort-pending", "read_file", JSON.stringify({ path: "buffer.txt" }));
      try {
        await withinReadDeadline(started.promise);
        controller.abort();
        const result = await withinReadDeadline(pending);
        assert.equal(result.content, "Error reading file: The operation was aborted");
        assert.equal(conn.readTextFileCalls.length, 1, "the already-admitted RPC is not undone");
        assert.equal(getEventListeners(controller.signal, "abort").length, 0);
        const statuses = conn.updates.map(update => (update.update as { status?: string }).status);
        assert.deepEqual(statuses, ["in_progress", "failed"]);
        const beforeLateSettlement = JSON.stringify(conn.updates);
        if (lateOutcome === "resolve") held.resolve({ content: "late remote secret" });
        else held.reject(new Error("late editor transport rejection"));
        await new Promise<void>(resolve => setImmediate(resolve));
        await new Promise<void>(resolve => setImmediate(resolve));
        assert.equal(JSON.stringify(conn.updates), beforeLateSettlement);
        assert.doesNotMatch(JSON.stringify(conn.updates), /late remote secret/);
        assert.deepEqual(unhandled, []);
      } finally {
        controller.abort();
        held.resolve({ content: "teardown" });
        await withinReadDeadline(Promise.allSettled([pending]));
        await new Promise<void>(resolve => setImmediate(resolve));
        process.off("unhandledRejection", onUnhandled);
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }
}

test("read_file truncates large files with a range marker", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-truncate-")));
  const path = join(dir, "big.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-editor-page-")));
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
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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

for (const legacy of [false, true]) {
  test(`read_file rejects an oversized ${legacy ? "legacy full buffer" : "conforming editor long line"} before returning it`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-editor-byte-limit-"));
    const path = join(dir, "buffer.txt");
    writeFileSync(path, "on disk");
    const updates: Array<{ update: { status?: string } }> = [];
    const conn = {
      async sessionUpdate(payload: { update: { status?: string } }) { updates.push(payload); },
      async readTextFile(params: { line?: number; limit?: number }) {
        const content = legacy ? "a\nb\nc\nd\ne\nf\ng\nh\ni" : "🙂".repeat(512);
        return { content: legacy || params.line === 1 ? content : "" };
      },
    };
    const limits: ResourceLimits = {
      toolResultBytes: 262_144, fileReadBytes: 16, listEntries: 2000, listBytes: 262_144,
      fsConcurrency: 16,
    };
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "default", () => undefined, limits);
    try {
      const result = await exec.execute("tc1", "read_file", JSON.stringify({ path, limit: 1 }));
      assert.match(result.content, /editor buffer exceeds the 16-byte read\/edit limit/);
      assert.doesNotMatch(result.content, /🙂|pass offset=/);
      assert.equal(updates.at(-1)?.update.status, "failed");
    } finally { rmSync(dir, { recursive: true, force: true }); }
  });
}

test("read_file checks the byte budget on the legacy EOF probe too", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-editor-probe-byte-limit-"));
  const path = join(dir, "buffer.txt");
  writeFileSync(path, "on disk");
  const conn = {
    async sessionUpdate() {},
    async readTextFile(params: { line?: number; limit?: number }) {
      return { content: params.line === 2 ? "short" : "x".repeat(2048) };
    },
  };
  const limits: ResourceLimits = {
    toolResultBytes: 262_144, fileReadBytes: 16, listEntries: 2000, listBytes: 262_144,
    fsConcurrency: 16,
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "default", () => undefined, limits);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path, offset: 2, limit: 1 }));
    assert.match(result.content, /editor buffer exceeds the 16-byte read\/edit limit/);
    assert.doesNotMatch(result.content, /showing lines|pass offset=/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("read_file accepts editor pages exactly at their UTF-8 byte budget", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-editor-exact-byte-limit-"));
  const path = join(dir, "buffer.txt");
  writeFileSync(path, "on disk");
  const conn = {
    async sessionUpdate() {},
    async readTextFile() { return { content: "🙂🙂🙂🙂" }; },
  };
  const limits: ResourceLimits = {
    toolResultBytes: 262_144, fileReadBytes: 16, listEntries: 2000, listBytes: 262_144,
    fsConcurrency: 16,
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir, () => "default", () => undefined, limits);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path, limit: 1 }));
    assert.equal(result.content, "🙂🙂🙂🙂");
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("read_file renders EOF for a conforming editor at a short or empty offset", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-editor-eof-")));
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
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-editor-legacy-page-")));
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
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-editor-legacy-single-line-")));
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
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-final-page-")));
  const path = join(dir, "paged.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-past-eof-")));
  const path = join(dir, "paged.txt");
  writeFileSync(path, Array.from({ length: 10 }, (_, i) => `line-${i + 1}`).join("\n"), "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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

test("read_file keeps small card content fully readable while the tool result stays full", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-elide-content-")));
  const path = join(dir, "long.txt");
  writeFileSync(path, `${"a".repeat(400)}\nsecond line`, "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
  try {
    const result = await exec.execute("tc1", "read_file", JSON.stringify({ path }));
    // The model receives the full page through the tool result...
    assert.ok(result.content.startsWith("a".repeat(400)));
    assert.match(result.content, /second line/);
    // ...and the client-facing content channel now carries the same full
    // text: boundCardText is a passthrough within the 16 KiB preview budget.
    const completed = conn.updates.find(
      (u) =>
        (u.update as { sessionUpdate?: string }).sessionUpdate === "tool_call_update" &&
        Array.isArray((u.update as { content?: unknown[] }).content)
    ) as { update: { content: Array<{ content: { text: string } }> } };
    const text = completed.update.content[0]?.content.text ?? "";
    assert.equal(text, result.content);
    assert.doesNotMatch(text, /bytes elided in preview/);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-success-")));
  const path = join(dir, "x.txt");
  writeFileSync(path, "hello", "utf8");
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-read-fail-")));
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
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

for (const permission of [
  "missing", "unoffered", "unknown", "null", "empty", "nonstring",
] as const) {
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
  // The card carries the 16 KiB UTF-8 prefix plus the elision marker, while
  // the model-facing tool result keeps the full 50,000-byte text.
  const cardText = final.content[0]?.content.text ?? "";
  assert.ok(cardText.startsWith("x".repeat(16_384)));
  assert.match(cardText, /\[… 33616 bytes elided in preview; full result sent to model\]$/);
  assert.ok(Buffer.byteLength(cardText, "utf8") <= 16_384 + 200);
  assert.notDeepEqual(final.rawOutput, raw);
});

test("completed cards over 16 KiB keep a readable UTF-8 prefix while the model result stays full", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-elide-")));
  try {
    writeFileSync(join(dir, "big.png"), "fake png bytes", "utf8");
    const conn = createConnectionStub();
    // Multibyte tail proves the prefix cut is UTF-8-safe (no replacement chars).
    const full = `${"y".repeat(20_000)}结尾`;
    const vision = fakeVisionClient(async () => ({ content: [{ type: "text", text: full }] }));
    const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await executor.execute(
      "tc1",
      "image_analysis",
      JSON.stringify({ image_source: join(dir, "big.png") })
    );
    // The model-facing tool result keeps the complete text.
    assert.equal(result.content, full);
    const completed = conn.updates.find(
      (u) =>
        (u.update as { sessionUpdate?: string }).sessionUpdate === "tool_call_update" &&
        Array.isArray((u.update as { content?: unknown[] }).content)
    ) as { update: { status?: string; content: Array<{ content: { text: string } }> } };
    assert.equal(completed.update.status, "completed");
    const cardText = completed.update.content[0]?.content.text ?? "";
    assert.ok(cardText.startsWith("y".repeat(16_384)));
    assert.doesNotMatch(cardText, /\uFFFD/);
    assert.match(cardText, /\[… 3622 bytes elided in preview; full result sent to model\]$/);
    assert.ok(Buffer.byteLength(cardText, "utf8") <= 16_384 + 200);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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

for (const scenario of [
  { name: "overlapping snippets", content: "ababa", oldText: "aba", expected: "changedba" },
  { name: "overlapping astral text", content: "🙂🙂🙂", oldText: "🙂🙂", expected: "changed🙂" },
  { name: "a literal UTF-16 surrogate", content: "🙂", oldText: "\ud83d", expected: "changed\ude42" },
]) {
  test(`edit_file counts non-overlapping literal matches for ${scenario.name}`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-overlap-"));
    const path = join(dir, "code.txt");
    writeFileSync(path, scenario.content, "utf8");
    const conn = createConnectionStub();
    const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    try {
      const result = await executor.execute(
        "tc1",
        "edit_file",
        JSON.stringify({ path, old_text: scenario.oldText, new_text: "changed" })
      );
      assert.match(result.content, /edited successfully/);
      assert.deepEqual(conn.writeTextFileCalls.map(call => call.content), [scenario.expected]);
      assert.equal(conn.permissionRequests.length, 1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

for (const afterPermission of [false, true]) {
  test(`edit_file reports the full non-overlapping ambiguity count ${afterPermission ? "after" : "before"} permission`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-overlap-ambiguous-"));
    const path = join(dir, "code.txt");
    const ambiguous = "🙂🙂🙂🙂🙂🙂🙂";
    writeFileSync(path, afterPermission ? "🙂🙂" : ambiguous, "utf8");
    const conn = createConnectionStub({
      onPermission: () => writeFileSync(path, ambiguous, "utf8"),
    });
    const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    try {
      const result = await executor.execute(
        "tc1",
        "edit_file",
        JSON.stringify({ path, old_text: "🙂🙂", new_text: "changed" })
      );
      assert.match(result.content, afterPermission ? /now occurs 3 times/ : /occurs 3 times/);
      assert.equal(readFileSync(path, "utf8"), ambiguous);
      assert.equal(conn.writeTextFileCalls.length, 0);
      assert.equal(conn.permissionRequests.length, afterPermission ? 1 : 0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

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
    assert.match(result.content, /changed while waiting for permission or progress delivery/);
    // The user's concurrent edit is intact and nothing was written back.
    assert.equal(readFileSync(path, "utf8"), "keep\nuser rewrote this\n");
    assert.equal(conn.writeTextFileCalls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

for (const scenario of [
  { name: "preserves unrelated changes", concurrent: "user changed this\nold snippet\n", expected: "user changed this\nnew snippet\n", writes: 1 },
  { name: "refuses a changed target", concurrent: "keep\nuser rewrote this\n", expected: "keep\nuser rewrote this\n", writes: 0 },
  { name: "refuses a newly ambiguous target", concurrent: "old snippet\nold snippet\n", expected: "old snippet\nold snippet\n", writes: 0 },
  { name: "cancels before dispatch", concurrent: "keep\nold snippet\n", expected: "keep\nold snippet\n", writes: 0, abort: true },
]) {
  test(`edit_file ${scenario.name} while the in_progress notification is pending`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-notification-"));
    const path = join(dir, "code.txt");
    writeFileSync(path, "keep\nold snippet\n", "utf8");
    const conn = createConnectionStub();
    let entered!: () => void;
    const notificationEntered = new Promise<void>(resolve => { entered = resolve; });
    let release!: () => void;
    const notificationReleased = new Promise<void>(resolve => { release = resolve; });
    const sessionUpdate = conn.sessionUpdate.bind(conn);
    conn.sessionUpdate = async payload => {
      await sessionUpdate(payload);
      const update = payload["update"] as { status?: string };
      if (update.status === "in_progress") {
        entered();
        await notificationReleased;
      }
    };
    const controller = new AbortController();
    const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS, controller.signal);
    try {
      const editing = executor.execute("tc1", "edit_file", JSON.stringify({ path, old_text: "old snippet", new_text: "new snippet" }));
      await notificationEntered;
      writeFileSync(path, scenario.concurrent, "utf8");
      if (scenario.abort) controller.abort();
      // Reads dispatched before the held notification is released (the initial
      // buffer read). Once the turn is aborted, the cancellation check must
      // skip the post-notification re-read entirely, so this count may only
      // grow while the turn is still alive.
      const readsBeforeDispatch = conn.readTextFileCalls.length;
      release();
      const result = await editing;
      assert.equal(readFileSync(path, "utf8"), scenario.expected);
      assert.equal(conn.writeTextFileCalls.length, scenario.writes);
      assert.equal(
        conn.readTextFileCalls.length,
        readsBeforeDispatch + (scenario.abort ? 0 : 1)
      );
      assert.match(result.content, scenario.abort ? /cancelled (?:by turn|before execution)/i : scenario.writes ? /File edited successfully/ : /changed while waiting/);
    } finally {
      release();
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

// The family above mutates the file on disk while the stub reads disk, so it
// cannot catch a regression that re-reads disk instead of the client's editor
// buffer during the post-notification revalidation. Here the client serves an
// independently mutable in-memory buffer while disk stays stale: the edit must
// be computed against the buffer the user actually sees.
for (const scenario of [
  {
    name: "preserves an unrelated unsaved buffer change",
    concurrent: "keep\nuser tweaked this\nold snippet\n",
    expected: "keep\nuser tweaked this\nnew snippet\n",
    diskAfter: "keep\nuser tweaked this\nnew snippet\n",
    writes: 1,
  },
  {
    name: "refuses an unsaved buffer rewrite of the target",
    concurrent: "keep\nuser rewrote this\n",
    expected: "keep\nuser rewrote this\n",
    diskAfter: "keep\nold snippet\n",
    writes: 0,
  },
]) {
  test(`edit_file ${scenario.name} while the in_progress notification is pending`, async () => {
    const dir = mkdtempSync(join(tmpdir(), "glm-executor-edit-buffer-"));
    const path = join(dir, "code.txt");
    // Disk keeps the original contents for the whole test; only the client's
    // unsaved buffer changes while the notification is held.
    writeFileSync(path, "keep\nold snippet\n", "utf8");
    const buffer = { content: "keep\nold snippet\n" };
    const conn = createConnectionStub({ clientBuffer: buffer });
    let entered!: () => void;
    const notificationEntered = new Promise<void>(resolve => { entered = resolve; });
    let release!: () => void;
    const notificationReleased = new Promise<void>(resolve => { release = resolve; });
    const sessionUpdate = conn.sessionUpdate.bind(conn);
    conn.sessionUpdate = async payload => {
      await sessionUpdate(payload);
      const update = payload["update"] as { status?: string };
      if (update.status === "in_progress") {
        entered();
        await notificationReleased;
      }
    };
    const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    try {
      const editing = executor.execute("tc1", "edit_file", JSON.stringify({ path, old_text: "old snippet", new_text: "new snippet" }));
      await notificationEntered;
      // Mutate ONLY the unsaved editor buffer; disk stays at "keep\nold snippet\n".
      buffer.content = scenario.concurrent;
      const readsBeforeDispatch = conn.readTextFileCalls.length;
      release();
      const result = await editing;
      // The post-notification re-read must serve the client buffer, so the
      // delivered edit carries the user's unsaved change instead of stale disk.
      assert.equal(conn.writeTextFileCalls.length, scenario.writes);
      if (scenario.writes > 0) {
        assert.equal(conn.writeTextFileCalls[0]?.content, scenario.expected);
      }
      assert.equal(readFileSync(path, "utf8"), scenario.diskAfter);
      assert.equal(conn.readTextFileCalls.length, readsBeforeDispatch + 1);
      assert.match(result.content, scenario.writes ? /File edited successfully/ : /changed while waiting/);
    } finally {
      release();
      rmSync(dir, { recursive: true, force: true });
    }
  });
}

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
  rmTempDir(dir);
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
    const staging = join(dir, "background-finished.tmp");
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, dir);
    try {
      // Existence must mean a closed, complete write, rather than the empty
      // file becoming visible between writeFileSync's open and write steps.
      const fixture = `setTimeout(() => { const fs = require("node:fs"); const staging = ${shellFixturePath(staging, "background-finished.tmp")}; fs.writeFileSync(staging, "done"); fs.renameSync(staging, ${shellFixturePath(marker, "background-finished")}); }, 250)`;
      const command = `${shellNodeCommand()} -e '${fixture}' >/dev/null 2>&1 & echo started`;
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
      rmTempDir(dir);
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

test("directory listing formatting preserves UTF-8 truncation with linear byte measurements", () => {
  const header = "Listing for . (中 path)";
  const marker = "[listing truncated: returned a subset; entries=2000, bytes=96]";
  const lines = Array.from({ length: 700 }, (_, index) => `${index % 2 ? "file" : "dir"}\t${index}\t${index % 3 ? "entry" : "🚀中"}-${index}`);
  const reference = (byteLimit: number, entriesTruncated: boolean): string => {
    const outputLines = [header];
    let byteLimitReached = Buffer.byteLength(outputLines[0]!, "utf8") > byteLimit;
    if (!byteLimitReached) {
      for (const line of lines) {
        if (Buffer.byteLength([...outputLines, line].join("\n"), "utf8") > byteLimit) {
          byteLimitReached = true;
          break;
        }
        outputLines.push(line);
      }
    }
    if (!entriesTruncated && !byteLimitReached) return outputLines.join("\n");
    const markerBytes = Buffer.byteLength(marker, "utf8");
    const prefixBudget = Math.max(0, byteLimit - markerBytes - 1);
    const prefix = takeUtf8Prefix(outputLines.join("\n"), prefixBudget);
    return `${prefix ? `${prefix}\n` : ""}${marker}`;
  };

  let byteMeasurements = 0;
  const full = formatDirectoryListing(header, lines, 1_000_000, marker, false, text => {
    byteMeasurements += 1;
    return Buffer.byteLength(text, "utf8");
  });
  assert.equal(full, reference(1_000_000, false));
  assert.equal(byteMeasurements, lines.length + 1, "header and each entry are measured once");

  for (const byteLimit of [1, 80, 180, 900, 100_000]) {
    for (const entriesTruncated of [false, true]) {
      assert.equal(
        formatDirectoryListing(header, lines, byteLimit, marker, entriesTruncated),
        reference(byteLimit, entriesTruncated),
        `byteLimit=${byteLimit}, entriesTruncated=${entriesTruncated}`,
      );
    }
  }
});

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

test("list_files opens the canonical directory with read-capable clients", async (t) => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-canonical-")));
  const target = join(dir, "target");
  const alias = join(dir, "alias");
  const openedPaths: string[] = [];
  const originalOpendir = fsPromises.opendir;
  try {
    mkdirSync(target);
    writeFileSync(join(target, "note.txt"), "data", "utf8");
    symlinkSync(target, alias, process.platform === "win32" ? "junction" : "dir");
    t.mock.method(fsPromises, "opendir", async (path: string) => {
      openedPaths.push(path);
      return originalOpendir(path);
    });
    syncBuiltinESMExports();
    for (const caps of [{ fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub();
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, null, null, dir);
      const result = await exec.execute("tc-canonical-list", "list_files", JSON.stringify({ path: "alias" }));
      assert.match(result.content, /note.txt/);
      // Match the production API: on Windows, async realpath expands 8.3
      // aliases that realpathSync may retain in the temporary fixture path.
      assert.equal(openedPaths.at(-1), await realpath(target));
      assert.equal(conn.readTextFileCalls.length, 0);
    }
    assert.equal(openedPaths.length, 2);
  } finally {
    t.mock.restoreAll();
    syncBuiltinESMExports();
    rmSync(dir, { recursive: true, force: true });
  }
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

test("list_files rejects an absolute path outside the session workspace", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-confined-")));
  const outside = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-outside-")));
  writeFileSync(join(outside, "secret.txt"), "secret", "utf8");
  try {
    for (const caps of [{ fs: {} }, { fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub();
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, null, null, dir);
      const result = await exec.execute("tc1", "list_files", JSON.stringify({ path: outside }));
      assert.match(result.content, /must remain inside the session workspace/);
      assert.ok(!result.content.includes("secret.txt"), "must not enumerate the outside directory");
      const completed = (conn.updates as Array<{ update: { status?: string } }>).filter(
        (u) => u.update.status === "completed"
      );
      assert.equal(completed.length, 0);
      const last = conn.updates.at(-1) as { update: { status?: string } };
      assert.equal(last.update.status, "failed");
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  }
});

test("list_files rejects parent traversal that escapes the session workspace", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-traversal-")));
  const outside = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-sibling-")));
  writeFileSync(join(outside, "sibling-secret.txt"), "secret", "utf8");
  try {
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir);
    for (const path of ["..", join("..", basename(outside))]) {
      const result = await exec.execute("tc1", "list_files", JSON.stringify({ path }));
      assert.match(result.content, /must remain inside the session workspace/);
      assert.ok(!result.content.includes("sibling-secret.txt"), "must not enumerate the escaped directory");
      const completed = (conn.updates as Array<{ update: { status?: string } }>).filter(
        (u) => u.update.status === "completed"
      );
      assert.equal(completed.length, 0);
      const last = conn.updates.at(-1) as { update: { status?: string } };
      assert.equal(last.update.status, "failed");
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  }
});

test("list_files rejects a symlink that escapes the session workspace", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-symlink-")));
  const outside = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-list-symlink-outside-")));
  writeFileSync(join(outside, "linked-secret.txt"), "secret", "utf8");
  symlinkSync(outside, join(dir, "escape"), "dir");
  try {
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", { fs: {} }, undefined, null, null, dir);
    const result = await exec.execute("tc1", "list_files", JSON.stringify({ path: "escape" }));
    assert.match(result.content, /must remain inside the session workspace/);
    assert.ok(!result.content.includes("linked-secret.txt"), "must not enumerate the linked directory");
    const completed = (conn.updates as Array<{ update: { status?: string } }>).filter(
      (u) => u.update.status === "completed"
    );
    assert.equal(completed.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// web_search / web_reader via Z.AI Coding Plan MCP
// ---------------------------------------------------------------------------

test("web_search uses stored credentials and calls the Coding Plan MCP search tool", async () => {
  await withStoredApiKey(async () => {
    await withMockedFetch(
      [
        request => jsonResponse(
          {
            jsonrpc: "2.0",
            id: request.id,
            result: { protocolVersion: "2025-06-18", capabilities: {} },
          },
          { sessionId: "search-session" }
        ),
        new Response(null, { status: 202 }),
        request => jsonResponse({
          jsonrpc: "2.0",
          id: request.id,
          result: { tools: [{ name: "webSearchPrime", inputSchema: { properties: { search_query: { type: "string" } } } }] },
        }),
        request => jsonResponse({
          jsonrpc: "2.0",
          id: request.id,
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
        request => jsonResponse({
          jsonrpc: "2.0",
          id: request.id,
          result: { protocolVersion: "2025-06-18", capabilities: {} },
        }),
        new Response(null, { status: 202 }),
        request => jsonResponse({
          jsonrpc: "2.0",
          id: request.id,
          result: { tools: [{ name: "webReader" }] },
        }),
        request => jsonResponse({
          jsonrpc: "2.0",
          id: request.id,
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
        assert.equal(conn.permissionRequests.length, 1);
        assert.deepEqual(conn.permissionRequests[0], {
          sessionId: "s1",
          toolCall: {
            toolCallId: "tc-web-reader",
            title: "Read URL: https://example.com/",
            kind: "fetch",
            status: "pending",
            locations: [{ path: "https://example.com/" }],
            rawInput: { url: "https://example.com/", return_format: "markdown" },
          },
          options: [
            { kind: "allow_once", name: "Allow", optionId: "allow" },
            { kind: "reject_once", name: "Skip", optionId: "reject" },
          ],
        });
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

test("web_reader rejected by user makes no network requests", async () => {
  await withMockedFetch([], async (calls) => {
    const conn = createConnectionStub({ permission: "reject" });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    const result = await exec.execute(
      "tc-web-reader-reject",
      "web_reader",
      JSON.stringify({ url: "https://example.com/" })
    );

    assert.equal(result.content, "URL read rejected by user.");
    assert.equal(conn.permissionRequests.length, 1);
    assert.equal(calls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  });
});

test("web_reader rejects a long URL with full approval payload and bounded tool cards", async () => {
  const url = `https://example.com/${"s".repeat(50_000)}`;
  await withMockedFetch([], async calls => {
    const conn = createConnectionStub({ permission: "reject" });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    const result = await exec.execute("long-url", "web_reader", JSON.stringify({ url }));
    assert.equal(result.content, "URL read rejected by user.");
    assert.equal(calls.length, 0);
    assert.equal(conn.permissionRequests.length, 1);
    const permission = conn.permissionRequests[0] as { toolCall: Record<string, unknown> };
    assert.deepEqual(permission.toolCall, {
      toolCallId: "long-url",
      title: `Read URL: ${url}`,
      kind: "fetch",
      status: "pending",
      locations: [{ path: url }],
      rawInput: { url },
    });
    const card = conn.updates[0]?.update as { title: string; locations: unknown[]; rawInput: { url: string } };
    assert.ok(card.title.length <= 240);
    assert.ok(card.rawInput.url.length <= 240);
    assert.notEqual(card.title, permission.toolCall.title);
    assert.notEqual(card.rawInput.url, url);
    assert.deepEqual(card.locations, []);
    for (const update of conn.updates) assert.ok(Buffer.byteLength(JSON.stringify(update), "utf8") < 16_384);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  });
});

test("web_reader cancelled by user makes no network requests", async () => {
  await withMockedFetch([], async (calls) => {
    const conn = createConnectionStub({ permission: "cancelled" });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    const result = await exec.execute(
      "tc-web-reader-cancelled",
      "web_reader",
      JSON.stringify({ url: "https://example.com/" })
    );

    assert.equal(result.content, "URL read cancelled by user.");
    assert.equal(conn.permissionRequests.length, 1);
    assert.equal(calls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  });
});

for (const permission of [
  "missing", "unoffered", "unknown", "null", "empty", "nonstring",
] as const) {
  test(`web_reader rejects unexpected ${permission} permission outcomes without network requests`, async () => {
    await withMockedFetch([], async (calls) => {
      const conn = createConnectionStub({ permission });
      const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
      const result = await exec.execute(
        "tc-web-reader-invalid-permission",
        "web_reader",
        JSON.stringify({ url: "https://example.com/" })
      );

      assert.equal(result.content, "URL read rejected by user.");
      assert.equal(conn.permissionRequests.length, 1);
      assert.equal(calls.length, 0);
      const last = conn.updates.at(-1) as { update: { status?: string } };
      assert.equal(last.update.status, "failed");
    });
  });
}

test("web_reader permission transport errors make no network requests", async () => {
  await withMockedFetch([], async (calls) => {
    const conn = createConnectionStub({ permissionError: true });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
    const result = await exec.execute(
      "tc-web-reader-permission-error",
      "web_reader",
      JSON.stringify({ url: "https://example.com/" })
    );

    assert.match(result.content, /Error requesting permission: permission transport failed/);
    assert.equal(conn.permissionRequests.length, 1);
    assert.equal(calls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  });
});

test("web_reader follows session modes for permission requests", async () => {
  const oldEnv = process.env["Z_AI_API_KEY"];
  try {
    for (const mode of ["default", "accept_edits", "bypass_permissions"] as const) {
      // The MCP client caches by endpoint + API key; isolate each four-call fixture.
      process.env["Z_AI_API_KEY"] = `from-env-web-reader-${mode}`;
      await withMockedFetch(
        [
          request => jsonResponse({
            jsonrpc: "2.0",
            id: request.id,
            result: { protocolVersion: "2025-06-18", capabilities: {} },
          }),
          new Response(null, { status: 202 }),
          request => jsonResponse({
            jsonrpc: "2.0",
            id: request.id,
            result: { tools: [{ name: "webReader" }] },
          }),
          request => jsonResponse({
            jsonrpc: "2.0",
            id: request.id,
            result: {
              content: [
                {
                  type: "text",
                  text: JSON.stringify({
                    reader_result: {
                      title: "Example",
                      url: "https://example.com/",
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
          const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, null, undefined, () => mode);
          const result = await exec.execute(
            "tc-web-reader-mode",
            "web_reader",
            JSON.stringify({ url: "https://example.com/", return_format: "markdown" })
          );

          assert.equal(conn.permissionRequests.length, mode === "bypass_permissions" ? 0 : 1);
          assert.match(result.content, /^# Example/);
          assert.equal(calls.length, 4);
          assert.deepEqual(calls[3]?.body.params, {
            name: "webReader",
            arguments: { url: "https://example.com/", return_format: "markdown" },
          });
        }
      );
    }
  } finally {
    if (oldEnv === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = oldEnv;
  }
});

test("web_reader makes no network requests after the turn aborts while the permission prompt stays unanswered", async () => {
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  let resolvePermission!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<{ outcome: { outcome: "selected"; optionId: "allow" } }>((resolve) => {
        resolvePermission = () => resolve({ outcome: { outcome: "selected", optionId: "allow" } });
      });
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal);
  await withMockedFetch([], async (calls) => {
    const pending = exec.execute(
      "tc1",
      "web_reader",
      JSON.stringify({ url: "https://example.com/" })
    );
    await started;
    abortController.abort();

    // Cancellation must settle on its own while the prompt is still unanswered.
    // A regression that waits for the permission answer trips this timeout
    // instead of silently passing.
    let settleTimer: ReturnType<typeof setTimeout> | undefined;
    try {
      const result = await Promise.race([
        pending,
        new Promise<never>((_, reject) => {
          settleTimer = setTimeout(
            () => reject(new Error("execute() did not settle after abort; cancellation waits for the pending permission")),
            500
          );
        }),
      ]);

      assert.equal(result.content, "URL read cancelled by turn.");
      assert.equal(calls.length, 0);
      const last = conn.updates.at(-1) as { update: { status?: string } };
      assert.equal(last.update.status, "failed");
    } finally {
      if (settleTimer !== undefined) clearTimeout(settleTimer);
    }

    // Answering the stale prompt afterwards is a no-op: no network request and
    // no further status update may appear.
    resolvePermission();
    await pending;
    assert.equal(calls.length, 0);
    const finalUpdate = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(finalUpdate.update.status, "failed");
  });
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

function assertInitialImageAnnouncement(
  conn: { updates: Array<Record<string, unknown>> },
  toolCallId: string,
  args: Record<string, unknown>,
  status: "pending" | "in_progress" = "in_progress",
): void {
  const source = String(args["image_source"]).trim();
  assert.deepEqual(conn.updates[0], {
    sessionId: "s1",
    update: {
      sessionUpdate: "tool_call",
      toolCallId,
      title: `Analyze image: ${source}`,
      kind: "fetch",
      status,
      locations: [{ path: source }],
      rawInput: args,
    },
  });
}

test("image_analysis routes through the injected vision client and returns the text", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-route-")));
  try {
    writeFileSync(join(dir, "cat.png"), "fake png bytes", "utf8");
    const conn = createConnectionStub();
    const vision = fakeVisionClient(async (toolName, args) => {
      assert.equal(toolName, "image_analysis");
      assert.equal(args["image_source"], await realpath(join(dir, "cat.png")));
      return { content: [{ type: "text", text: "A tabby cat." }] };
    });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await exec.execute(
      "tc1",
      "image_analysis",
      JSON.stringify({ image_source: join(dir, "cat.png"), prompt: "describe" })
    );
    assert.equal(result.content, "A tabby cat.");
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "completed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("image_analysis surfaces vision errors as a failed tool result", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-error-")));
  try {
    writeFileSync(join(dir, "x.png"), "fake png bytes", "utf8");
    const conn = createConnectionStub();
    const vision = fakeVisionClient(async () => {
      throw new Error("Vision MCP image_analysis failed: quota exceeded");
    });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await exec.execute(
      "tc1",
      "image_analysis",
      JSON.stringify({ image_source: join(dir, "x.png") })
    );
    assert.match(result.content, /quota exceeded/);
    assertInitialImageAnnouncement(conn, "tc1", { image_source: join(dir, "x.png") });
    assert.equal(conn.updates.length, 2);
    const last = conn.updates.at(-1) as { update: { sessionUpdate?: string; status?: string } };
    assert.equal(last.update.sessionUpdate, "tool_call_update");
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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

// ---------------------------------------------------------------------------
// image_analysis local-source confinement (mirrors the read_file battery)
// ---------------------------------------------------------------------------

test("image_analysis rejects a local image_source outside the workspace and makes no vision calls", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-confined-")));
  const outsidePath = `${dir}-secret.png`;
  try {
    writeFileSync(outsidePath, "fake png bytes", "utf8");
    const conn = createConnectionStub();
    const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const vision = recordingVisionClient(visionCalls);
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    for (const path of [outsidePath, join("..", basename(outsidePath))]) {
      conn.updates.length = 0;
      const result = await exec.execute(
        "tc-image-outside",
        "image_analysis",
        JSON.stringify({ image_source: path, prompt: "describe" })
      );
      assert.match(result.content, /path must remain inside the session workspace/);
      assertInitialImageAnnouncement(conn, "tc-image-outside", { image_source: path, prompt: "describe" });
      assert.equal(conn.updates.length, 2);
      const last = conn.updates.at(-1) as { update: { sessionUpdate?: string; status?: string } };
      assert.equal(last.update.sessionUpdate, "tool_call_update");
      assert.equal(last.update.status, "failed");
    }
    assert.equal(visionCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outsidePath, { force: true });
  }
});

test("image_analysis passes the resolved workspace path to vision for a local image_source", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-inside-")));
  try {
    writeFileSync(join(dir, "cat.png"), "fake png bytes", "utf8");
    const conn = createConnectionStub();
    const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const vision = recordingVisionClient(visionCalls);
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await exec.execute(
      "tc-image-inside",
      "image_analysis",
      JSON.stringify({ image_source: "cat.png", prompt: "describe" })
    );
    assert.equal(result.content, "A tabby cat.");
    assert.deepEqual(
      visionCalls.map((c) => ({ toolName: c.toolName, source: c.args["image_source"] })),
      [{ toolName: "image_analysis", source: await realpath(join(dir, "cat.png")) }]
    );
    assertInitialImageAnnouncement(conn, "tc-image-inside", { image_source: "cat.png", prompt: "describe" });
    assert.equal(conn.updates.length, 2);
    const last = conn.updates.at(-1) as { update: { sessionUpdate?: string; status?: string } };
    assert.equal(last.update.sessionUpdate, "tool_call_update");
    assert.equal(last.update.status, "completed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("image_analysis forwards the canonical file with read-capable clients", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-canonical-")));
  const target = join(dir, "target.png");
  const alias = join(dir, "alias.png");
  try {
    writeFileSync(target, "fake png bytes", "utf8");
    symlinkSync(target, alias, "file");
    for (const caps of [{ fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub();
      const calls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, recordingVisionClient(calls), null, dir);
      const args = { image_source: "alias.png" };
      const result = await exec.execute("tc-canonical-image", "image_analysis", JSON.stringify(args));
      assert.equal(result.content, "A tabby cat.");
      assert.deepEqual(calls, [{ toolName: "image_analysis", args: { image_source: await realpath(target) } }]);
      assertInitialImageAnnouncement(conn, "tc-canonical-image", args);
      assert.equal(conn.readTextFileCalls.length, 0);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("agent-side reads reject missing leaves before enumeration or Vision dispatch", async (t) => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-agent-missing-")));
  const openedPaths: string[] = [];
  const calls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const originalOpendir = fsPromises.opendir;
  try {
    t.mock.method(fsPromises, "opendir", async (path: string) => {
      openedPaths.push(path);
      return originalOpendir(path);
    });
    syncBuiltinESMExports();
    for (const caps of [{ fs: { readTextFile: true } }, FULL_CAPS]) {
      const conn = createConnectionStub();
      const exec = new ToolExecutor(conn as never, "s1", caps, undefined, recordingVisionClient(calls), null, dir);
      for (const tool of ["list_files", "image_analysis"]) {
        const args = tool === "list_files" ? { path: "missing" } : { image_source: "missing.png" };
        const result = await exec.execute("tc-missing", tool, JSON.stringify(args));
        assert.match(result.content, /ENOENT|no such file/i);
        assert.equal(openedPaths.length, 0);
        assert.equal(calls.length, 0);
        assert.equal(conn.readTextFileCalls.length, 0);
      }
    }
  } finally {
    t.mock.restoreAll();
    syncBuiltinESMExports();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("image_analysis rejects a workspace symlink pointing outside and makes no vision calls", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-symlink-")));
  const outsidePath = `${dir}-outside.png`;
  try {
    writeFileSync(outsidePath, "fake png bytes", "utf8");
    symlinkSync(outsidePath, join(dir, "escape.png"), "file");
    const conn = createConnectionStub();
    const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const vision = recordingVisionClient(visionCalls);
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await exec.execute(
      "tc-image-symlink",
      "image_analysis",
      JSON.stringify({ image_source: "escape.png", prompt: "describe" })
    );
    assert.match(result.content, /path must remain inside the session workspace/);
    assert.equal(visionCalls.length, 0);
    assertInitialImageAnnouncement(conn, "tc-image-symlink", { image_source: "escape.png", prompt: "describe" });
    assert.equal(conn.updates.length, 2);
    const last = conn.updates.at(-1) as { update: { sessionUpdate?: string; status?: string } };
    assert.equal(last.update.sessionUpdate, "tool_call_update");
    assert.equal(last.update.status, "failed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(outsidePath, { force: true });
  }
});

// ---------------------------------------------------------------------------
// image_analysis remote-source approval gate (mirrors the web_reader battery)
// ---------------------------------------------------------------------------

function recordingVisionClient(
  visionCalls: Array<{ toolName: string; args: Record<string, unknown> }>
): VisionMcpClient {
  return fakeVisionClient(async (toolName, args) => {
    visionCalls.push({ toolName, args });
    return { content: [{ type: "text", text: "A tabby cat." }] };
  });
}

/**
 * Reject after `ms` unless the promise it is raced against settles first. The
 * rejection is pre-handled so losing the race never surfaces as an unhandled
 * rejection.
 */
function failAfter(ms: number, message: string): Promise<never> {
  const failure = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(message)), ms);
  });
  failure.catch(() => { /* handled by whoever wins the race */ });
  return failure;
}

const imagePathBackslash = String.fromCharCode(92);
const nonLocalImageSources = [
  ["whitespace-padded HTTP source", " " + String.fromCharCode(9) + "https:attacker.example/x.png" + String.fromCharCode(10)],
  ["HTTP URL without slashes", "https:attacker.example/x.png"],
  ["HTTP URL with backslashes", ["http:", "", "attacker.example", "x.png"].join(imagePathBackslash)],
  ["FTP scheme", "ftp://attacker.example/x.png"],
  ["file scheme", "file:///outside-workspace.png"],
  ["data scheme", "data:image/png;base64,AA=="],
  ["custom scheme", "vision+remote:attacker.example/x.png"],
  ["drive-relative scheme-like source", "C:cat.png"],
  ["embedded tab", "ht" + String.fromCharCode(9) + "tps://attacker.example/x.png"],
  ["embedded newline", "ht" + String.fromCharCode(10) + "tps://attacker.example/x.png"],
  ["embedded carriage return", "ht" + String.fromCharCode(13) + "tps://attacker.example/x.png"],
  ["leading NUL", String.fromCharCode(0) + "https://attacker.example/x.png"],
  ["DEL in a path", "images/" + String.fromCharCode(127) + "cat.png"],
  ["control in a Windows drive path", "C:/images/" + String.fromCharCode(9) + "cat.png"],
  ["protocol-relative source", "//attacker.example/x.png"],
  ["UNC source", imagePathBackslash.repeat(2) + "attacker.example" + imagePathBackslash + "x.png"],
  ["mixed network-authority prefix", "/" + imagePathBackslash + "attacker.example/x.png"],
] as const;

for (const [label, source] of nonLocalImageSources) {
  test(`image_analysis requires approval for ${label}`, async () => {
    const args = { image_source: source, prompt: "describe" };
    for (const mode of ["default", "accept_edits"] as const) {
      const conn = createConnectionStub({ permission: "reject" });
      const calls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
      const exec = new ToolExecutor(
        conn as never, "s1", FULL_CAPS, undefined, recordingVisionClient(calls),
        null, process.cwd(), () => mode,
      );
      const result = await exec.execute("tc-non-local", "image_analysis", JSON.stringify(args));
      assert.equal(result.content, "Image analysis rejected by user.");
      assert.equal(conn.permissionRequests.length, 1);
      const request = conn.permissionRequests[0] as { toolCall: { rawInput: unknown; locations: unknown } };
      assert.deepEqual(request.toolCall.rawInput, args);
      assert.deepEqual(request.toolCall.locations, [{ path: source.trim() }]);
      assert.equal(calls.length, 0);
      assertInitialImageAnnouncement(conn, "tc-non-local", args, "pending");
    }

    const conn = createConnectionStub({ permission: "allow" });
    const calls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, recordingVisionClient(calls));
    const result = await exec.execute("tc-non-local-allowed", "image_analysis", JSON.stringify(args));
    assert.equal(result.content, "A tabby cat.");
    assert.equal(conn.permissionRequests.length, 1);
    assert.deepEqual(calls, [{ toolName: "image_analysis", args: { ...args, image_source: source.trim() } }]);
    assertInitialImageAnnouncement(conn, "tc-non-local-allowed", args, "pending");
    assert.deepEqual(conn.updates.map(({ update }) => {
      const event = update as { sessionUpdate: string; status: string };
      return [event.sessionUpdate, event.status];
    }), [["tool_call", "pending"], ["tool_call_update", "in_progress"], ["tool_call_update", "completed"]]);
  });
}

for (const [label, source] of [
  ["relative", "images/cat.png"],
  ["absolute", "/workspace/images/cat.png"],
  ["boundary whitespace", String.fromCharCode(9) + "images/cat.png" + String.fromCharCode(10)],
  ["spaces", "images/my cat.png"],
  ["Windows forward slashes", "C:/images/cat.png"],
  ["Windows backslashes", ["C:", "images", "cat.png"].join(imagePathBackslash)],
] as const) {
  test(`image_analysis does not request approval for ${label} local path syntax`, async () => {
    const conn = createConnectionStub({ permission: "reject" });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, fakeVisionClient(async () => ({
      content: [{ type: "text", text: "A tabby cat." }],
    })));
    const args = { image_source: source };
    await exec.execute("tc-local-syntax", "image_analysis", JSON.stringify(args));
    assert.equal(conn.permissionRequests.length, 0);
    assertInitialImageAnnouncement(conn, "tc-local-syntax", args);
  });
}

test("image_analysis rejected by user makes no vision calls for remote image sources", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const url = "https://attacker.example/x.png?d=secret";
  const result = await exec.execute(
    "tc-image-url-reject",
    "image_analysis",
    JSON.stringify({ image_source: url, prompt: "describe" })
  );

  assert.equal(result.content, "Image analysis rejected by user.");
  assert.equal(conn.permissionRequests.length, 1);
  // The full, unelided arguments reach the approval prompt: the URL can carry
  // sensitive data, so the user must approve exactly what would be sent.
  assert.deepEqual(conn.permissionRequests[0], {
    sessionId: "s1",
    toolCall: {
      toolCallId: "tc-image-url-reject",
      title: `Analyze image: ${url}`,
      kind: "fetch",
      status: "pending",
      locations: [{ path: url }],
      rawInput: { image_source: url, prompt: "describe" },
    },
    options: [
      { kind: "allow_once", name: "Allow", optionId: "allow" },
      { kind: "reject_once", name: "Skip", optionId: "reject" },
    ],
  });
  assert.equal(visionCalls.length, 0);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("image_analysis cancelled by user makes no vision calls for remote image sources", async () => {
  const conn = createConnectionStub({ permission: "cancelled" });
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const result = await exec.execute(
    "tc-image-url-cancelled",
    "image_analysis",
    JSON.stringify({ image_source: "https://example.com/pic.png" })
  );

  assert.equal(result.content, "Image analysis cancelled by user.");
  assert.equal(conn.permissionRequests.length, 1);
  assert.equal(visionCalls.length, 0);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

for (const permission of [
  "missing", "unoffered", "unknown", "null", "empty", "nonstring",
] as const) {
  test(`image_analysis rejects unexpected ${permission} permission outcomes for remote image sources without vision calls`, async () => {
    const conn = createConnectionStub({ permission });
    const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const vision = recordingVisionClient(visionCalls);
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
    const result = await exec.execute(
      "tc-image-url-invalid-permission",
      "image_analysis",
      JSON.stringify({ image_source: "https://example.com/pic.png" })
    );

    assert.equal(result.content, "Image analysis rejected by user.");
    assert.equal(conn.permissionRequests.length, 1);
    assert.equal(visionCalls.length, 0);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "failed");
  });
}

test("image_analysis permission transport errors make no vision calls for remote image sources", async () => {
  const conn = createConnectionStub({ permissionError: true });
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const result = await exec.execute(
    "tc-image-url-permission-error",
    "image_analysis",
    JSON.stringify({ image_source: "https://example.com/pic.png" })
  );

  assert.match(result.content, /Error requesting permission: permission transport failed/);
  assert.equal(conn.permissionRequests.length, 1);
  assert.equal(visionCalls.length, 0);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("image_analysis requires approval for uppercase HTTP remote image sources", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision);
  const url = "HTTP://insecure.example/pic.jpg";
  const result = await exec.execute(
    "tc-image-url-uppercase-scheme",
    "image_analysis",
    JSON.stringify({ image_source: url })
  );

  assert.equal(result.content, "Image analysis rejected by user.");
  assert.equal(conn.permissionRequests.length, 1);
  assert.deepEqual(conn.permissionRequests[0], {
    sessionId: "s1",
    toolCall: {
      toolCallId: "tc-image-url-uppercase-scheme",
      title: `Analyze image: ${url}`,
      kind: "fetch",
      status: "pending",
      locations: [{ path: url }],
      rawInput: { image_source: url },
    },
    options: [
      { kind: "allow_once", name: "Allow", optionId: "allow" },
      { kind: "reject_once", name: "Skip", optionId: "reject" },
    ],
  });
  assert.equal(visionCalls.length, 0);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("image_analysis follows session modes for remote image sources", async () => {
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  for (const mode of ["default", "accept_edits", "bypass_permissions"] as const) {
    visionCalls.length = 0;
    const conn = createConnectionStub();
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, undefined, () => mode);
    const result = await exec.execute(
      "tc-image-url-mode",
      "image_analysis",
      JSON.stringify({ image_source: "https://example.com/pic.png", prompt: "describe" })
    );

    assert.equal(conn.permissionRequests.length, mode === "bypass_permissions" ? 0 : 1);
    if (mode !== "bypass_permissions") {
      const request = conn.permissionRequests[0] as { toolCall: { kind: string } };
      // "fetch" is never auto-approved, even in accept_edits: only writes skip prompting.
      assert.equal(request.toolCall.kind, "fetch");
    }
    assert.equal(result.content, "A tabby cat.");
    assert.equal(visionCalls.length, 1);
    const last = conn.updates.at(-1) as { update: { status?: string } };
    assert.equal(last.update.status, "completed");
  }
});

test("image_analysis makes no vision calls after the turn aborts while permission is pending", async () => {
  const abortController = new AbortController();
  let permissionStarted!: () => void;
  let resolvePermission!: () => void;
  const started = new Promise<void>((resolve) => { permissionStarted = resolve; });
  const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
  const vision = recordingVisionClient(visionCalls);
  const conn = {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(payload: Record<string, unknown>) { this.updates.push(payload); },
    async requestPermission() {
      permissionStarted();
      return new Promise<{ outcome: { outcome: "selected"; optionId: "allow" } }>((resolve) => {
        resolvePermission = () => resolve({ outcome: { outcome: "selected", optionId: "allow" } });
      });
    },
  };
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, abortController.signal, vision);
  const pending = exec.execute(
    "tc-image-url-abort",
    "image_analysis",
    JSON.stringify({ image_source: "https://example.com/pic.png" })
  );
  // Wait until the approval prompt is actually pending; a missing gate fails
  // fast here instead of hanging the suite.
  await Promise.race([
    started,
    failAfter(500, "image_analysis never requested permission for a remote image source (approval gate missing)"),
  ]);
  abortController.abort();
  // The execute promise must settle from the abort alone; a regression that
  // waits on the user hangs here instead of passing silently.
  await Promise.race([
    pending,
    failAfter(500, "image_analysis did not settle after the turn aborted while permission was pending"),
  ]);
  // The user's late allow arrives only after the turn already aborted.
  resolvePermission();
  const result = await pending;

  assert.equal(result.content, "Image analysis cancelled by turn.");
  assert.equal(visionCalls.length, 0);
  const completed = conn.updates.filter((u) => (u.update as { status?: string }).status === "completed");
  assert.equal(completed.length, 0);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

test("image_analysis does not prompt for local image sources", async () => {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "glm-executor-image-noprompt-")));
  try {
    writeFileSync(join(dir, "cat.png"), "fake png bytes", "utf8");
    const conn = createConnectionStub({ permission: "reject" });
    const visionCalls: Array<{ toolName: string; args: Record<string, unknown> }> = [];
    const vision = fakeVisionClient(async (toolName, args) => {
      assertInitialImageAnnouncement(conn, "tc-image-local", { image_source: join(dir, "cat.png"), prompt: "describe" });
      assert.equal(conn.updates.length, 1, "local analysis is announced before Vision dispatch");
      visionCalls.push({ toolName, args });
      assert.equal(args["image_source"], await realpath(join(dir, "cat.png")));
      return { content: [{ type: "text", text: "A tabby cat." }] };
    });
    const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, vision, null, dir);
    const result = await exec.execute(
      "tc-image-local",
      "image_analysis",
      JSON.stringify({ image_source: join(dir, "cat.png"), prompt: "describe" })
    );

    assert.equal(result.content, "A tabby cat.");
    assert.equal(conn.permissionRequests.length, 0);
    assert.equal(visionCalls.length, 1);
    assertInitialImageAnnouncement(conn, "tc-image-local", { image_source: join(dir, "cat.png"), prompt: "describe" });
    assert.equal(conn.updates.length, 2);
    const last = conn.updates.at(-1) as { update: { sessionUpdate?: string; status?: string } };
    assert.equal(last.update.sessionUpdate, "tool_call_update");
    assert.equal(last.update.status, "completed");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("session MCP tool with ~10k-deep result completes with bounded preview markers", async () => {
  const conn = createConnectionStub();
  let deep: Record<string, unknown> = { leaf: "end" };
  for (let i = 0; i < 10_000; i++) deep = { nested: deep };
  const raw = { content: [{ type: "text", text: "deep ok" }], data: deep };
  const tools = { hasTool: () => true, callTool: async () => raw };
  const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, tools as never);
  const result = await executor.execute("mcp-deep", "custom_tool", JSON.stringify({ q: 1 }));
  // The tool already ran — the preview walk must not flip it to failed.
  assert.equal(result.content, "deep ok");
  const failed = conn.updates.filter((u) => (u.update as { status?: string }).status === "failed");
  assert.equal(failed.length, 0);
  const completed = conn.updates.filter((u) => (u.update as { status?: string }).status === "completed");
  assert.equal(completed.length, 1);
  const final = completed[0]!.update as { rawOutput: unknown };
  const serialized = JSON.stringify(final.rawOutput)!;
  assert.ok(serialized.includes("[nested]"), "depth marker present");
  assert.ok((serialized.match(/"nested"/g) ?? []).length <= 40, "walk did not descend the full depth");
  for (const event of conn.updates) {
    assert.ok(Buffer.byteLength(JSON.stringify(event), "utf8") < 18_000);
  }
});

test("session MCP preview construction failure cannot fail an executed tool call", async () => {
  const conn = createConnectionStub();
  const raw: Record<string, unknown> = {
    content: [{ type: "text", text: "computed fine" }],
    get data(): never {
      throw new Error("preview exploded");
    },
  };
  const tools = { hasTool: () => true, callTool: async () => raw };
  const executor = new ToolExecutor(conn as never, "s1", FULL_CAPS, undefined, null, tools as never);
  const result = await executor.execute("mcp-throw", "custom_tool", "{}");
  assert.equal(result.content, "computed fine");
  const last = conn.updates.at(-1)?.update as { status?: string; rawOutput: unknown };
  assert.equal(last.status, "completed");
  assert.deepEqual(last.rawOutput, { truncated: "Preview unavailable" });
});
