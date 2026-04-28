import test from "node:test";
import assert from "node:assert/strict";
import { ToolExecutor } from "../tools/executor.js";

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
// Capability gating
// ---------------------------------------------------------------------------

test("read_file is unavailable without fs.readTextFile capability", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} });
  const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "/x" }));
  assert.match(result.content, /does not advertise the fs.readTextFile capability/);
});

test("write_file is unavailable without fs.writeTextFile capability", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: { readTextFile: true } });
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/x", content: "hi" })
  );
  assert.match(result.content, /does not advertise the fs.writeTextFile capability/);
});

test("list_files / run_command are unavailable without terminal capability", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", { fs: {} });
  const ls = await exec.execute("tc1", "list_files", JSON.stringify({ path: "/" }));
  assert.match(ls.content, /does not advertise the terminal capability/);
  const rc = await exec.execute("tc2", "run_command", JSON.stringify({ command: "ls" }));
  assert.match(rc.content, /does not advertise the terminal capability/);
});

// ---------------------------------------------------------------------------
// read_file
// ---------------------------------------------------------------------------

test("read_file success path emits in_progress and completed updates", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "/x.txt" }));
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
});

test("read_file failure is reported with status=failed and an error message", async () => {
  const conn = createConnectionStub({ readError: true });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "read_file", JSON.stringify({ path: "/x.txt" }));
  assert.match(result.content, /Error reading file:/);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
});

// ---------------------------------------------------------------------------
// write_file (permission flow)
// ---------------------------------------------------------------------------

test("write_file requests permission, then transitions through pending → in_progress → completed", async () => {
  const conn = createConnectionStub({ permission: "allow" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/y.txt", content: "data" })
  );
  assert.match(result.content, /written successfully/);

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
});

test("write_file rejected by user marks call failed and skips writing", async () => {
  const conn = createConnectionStub({ permission: "reject" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "write_file",
    JSON.stringify({ path: "/y.txt", content: "data" })
  );
  assert.match(result.content, /rejected by user/i);
  const last = conn.updates.at(-1) as { update: { status?: string } };
  assert.equal(last.update.status, "failed");
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
  const conn = createConnectionStub({ terminalOutput: "/tmp" });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "echo $HOME | tr a-z A-Z" })
  );
  assert.equal(result.content, "/tmp");
  const call = conn.terminalCalls[0];
  assert.equal(call?.command, "sh");
  assert.deepEqual(call?.args, ["-c", "echo $HOME | tr a-z A-Z"]);
});

// ---------------------------------------------------------------------------
// list_files
// ---------------------------------------------------------------------------

test("list_files runs through sh -c with shell-quoted path", async () => {
  const conn = createConnectionStub({ terminalOutput: "drwxr-xr-x ..." });
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute(
    "tc1",
    "list_files",
    JSON.stringify({ path: "/tmp/with space" })
  );
  assert.match(result.content, /drwxr-xr-x/);
  const call = conn.terminalCalls[0];
  assert.equal(call?.command, "sh");
  assert.equal(call?.args?.[0], "-c");
  assert.match(call?.args?.[1] ?? "", /^ls -la --/);
  assert.match(call?.args?.[1] ?? "", /'\/tmp\/with space'/);
});

test("list_files rejects empty path", async () => {
  const conn = createConnectionStub();
  const exec = new ToolExecutor(conn as never, "s1", FULL_CAPS);
  const result = await exec.execute("tc1", "list_files", JSON.stringify({ path: "" }));
  assert.match(result.content, /non-empty string/);
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
