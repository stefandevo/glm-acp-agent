import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { writeCredentials } from "../llm/credentials.js";
import { ToolExecutor } from "../tools/executor.js";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

interface StubTerminal {
  id: string;
  waitForExit: () => Promise<{ exitCode: number }>;
  currentOutput: () => Promise<{ output: string }>;
  release: () => Promise<void>;
}

function createConnectionStub(opts: {
  permission?: "allow" | "reject" | "cancelled";
  readError?: boolean;
  writeError?: boolean;
  terminalOutput?: string;
} = {}) {
  const updates: Array<Record<string, unknown>> = [];
  const permissionRequests: Array<unknown> = [];
  const terminalCalls: Array<{ command: string; args?: string[] }> = [];

  return {
    updates,
    permissionRequests,
    terminalCalls,
    async sessionUpdate(payload: Record<string, unknown>) {
      updates.push(payload);
    },
    async readTextFile(_params: { sessionId: string; path: string }) {
      if (opts.readError) throw new Error("file not found");
      return { content: "hello" };
    },
    async writeTextFile(_params: { sessionId: string; path: string; content: string }) {
      if (opts.writeError) throw new Error("permission denied");
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
      switch (opts.permission ?? "allow") {
        case "allow":
          return { outcome: { outcome: "selected", optionId: "allow" } };
        case "reject":
          return { outcome: { outcome: "selected", optionId: "reject" } };
        case "cancelled":
          return { outcome: { outcome: "cancelled" } };
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
    assert.match(rc.content, new RegExp(escapeRegExp(dir)));
    assert.equal(conn.terminalCalls.length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

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
  assert.match(result.content, new RegExp(escapeRegExp(dir)));
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
