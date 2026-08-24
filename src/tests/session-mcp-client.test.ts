import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { Readable, Writable } from "node:stream";
import type { McpServerStdio } from "@agentclientprotocol/sdk";
import { StdioMcpClient } from "../tools/session-mcp-client.js";

interface FakeChild extends EventEmitter {
  stdin: Writable;
  stdout: Readable;
  stderr: Readable;
  pid: number;
  exitCode: number | null;
  kill: (signal?: string) => boolean;
}

function makeFakeChild(): {
  child: FakeChild;
  written: string[];
  pushStdout: (line: string) => void;
  pushStderr: (line: string) => void;
  getKillCount: () => number;
} {
  const written: string[] = [];
  let killCount = 0;
  const stdin = new Writable({
    write(chunk, _enc, cb) {
      written.push(chunk.toString("utf8"));
      cb();
    },
  });
  const stdout = new Readable({ read() { /* push manually */ } });
  const stderr = new Readable({ read() { /* push manually */ } });
  const child = Object.assign(new EventEmitter(), {
    stdin,
    stdout,
    stderr,
    pid: 5150,
    exitCode: null,
    kill: () => {
      killCount += 1;
      return true;
    },
  }) as FakeChild;
  return {
    child,
    written,
    pushStdout: (line: string) => stdout.push(line),
    pushStderr: (line: string) => stderr.push(line),
    getKillCount: () => killCount,
  };
}

function stdioServer(overrides: Partial<McpServerStdio> = {}): McpServerStdio {
  return {
    name: "docs",
    command: "npx",
    args: ["-y", "@example/mcp-docs"],
    env: [],
    ...overrides,
  };
}

const tick = () => new Promise((r) => setImmediate(r));

/** Drive a fake child through initialize + tools/list so the client is ready for tools/call. */
async function completeHandshake(
  written: string[],
  pushStdout: (line: string) => void,
  tools: { name: string }[] = [{ name: "search" }]
): Promise<void> {
  await tick();
  const init = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: init.id, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await tick();
  const list = JSON.parse(written[2]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: list.id, result: { tools } }) + "\n");
  await tick();
}

test("StdioMcpClient rejects an asynchronous spawn error instead of hanging", async () => {
  const { child } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", Object.assign(new Error("spawn npx ENOENT"), { code: "ENOENT" }));

  await assert.rejects(listPromise, /could not launch `npx`/i);
  await client.dispose();
});

test("StdioMcpClient rejects a pending tools/call when the child errors asynchronously", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  const callPromise = client.callTool("search", { q: "x" });
  await tick();
  child.emit("error", Object.assign(new Error("spawn npx EINVAL"), { code: "EINVAL" }));

  await assert.rejects(callPromise, /EINVAL/);
  await client.dispose();
});

test("StdioMcpClient times out initialization and terminates the child", async () => {
  const { child, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => child as never,
    initializationTimeoutMs: 50,
  });

  await assert.rejects(() => client.listTools(), /timed out after \d+ms/i);
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioMcpClient times out an individual tools/call and terminates the child", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => child as never,
    requestTimeoutMs: 50,
  });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  await assert.rejects(() => client.callTool("search", { q: "x" }), /timed out after \d+ms/i);
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioMcpClient launches npx through cmd.exe on Windows", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  let windowsHide = false;
  const client = new StdioMcpClient(stdioServer(), {
    platform: "win32",
    comSpec: "C:\\Windows\\System32\\cmd.exe",
    killProcessTree: () => true,
    spawn: (capturedCommand, capturedArgs, options) => {
      command = capturedCommand;
      args = capturedArgs;
      windowsHide = options.windowsHide === true;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "C:\\Windows\\System32\\cmd.exe");
  assert.deepEqual(args, ["/d", "/s", "/c", "npx", "-y", "@example/mcp-docs"]);
  assert.equal(windowsHide, true);
  await client.dispose();
});

test("StdioMcpClient launches a .cmd shim through cmd.exe on Windows", async () => {
  const { child } = makeFakeChild();
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "C:\\tools\\mcp-docs.cmd", args: ["--stdio"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    killProcessTree: () => true,
    spawn: (_command, capturedArgs) => {
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.deepEqual(args, ["/d", "/s", "/c", "C:\\tools\\mcp-docs.cmd", "--stdio"]);
  await client.dispose();
});

test("StdioMcpClient spawns a real executable directly on Windows", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "node", args: ["server.js"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    killProcessTree: () => true,
    spawn: (capturedCommand, capturedArgs) => {
      command = capturedCommand;
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "node");
  assert.deepEqual(args, ["server.js"]);
  await client.dispose();
});

test("StdioMcpClient rejects cmd.exe metacharacters in client-supplied args before spawning", async () => {
  let spawned = false;
  const client = new StdioMcpClient(stdioServer({ args: ["-y", "@example/mcp-docs & calc.exe"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => {
      spawned = true;
      return makeFakeChild().child as never;
    },
  });

  await assert.rejects(() => client.listTools(), /unsafe .*cmd\.exe/i);
  assert.equal(spawned, false);
});

test("StdioMcpClient rejects cmd.exe metacharacters in the command before spawning", async () => {
  let spawned = false;
  const client = new StdioMcpClient(stdioServer({ command: "C:\\tools & calc.exe\\mcp-docs.cmd", args: [] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => {
      spawned = true;
      return makeFakeChild().child as never;
    },
  });

  await assert.rejects(() => client.listTools(), /unsafe .*cmd\.exe/i);
  assert.equal(spawned, false);
});

test("StdioMcpClient allows cmd.exe metacharacters on POSIX where no shell is involved", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "npx", args: ["-y", "@example/weird&name"] }), {
    platform: "linux",
    spawn: (capturedCommand, capturedArgs) => {
      command = capturedCommand;
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "npx");
  assert.deepEqual(args, ["-y", "@example/weird&name"]);
  await client.dispose();
});

test("StdioMcpClient drains stderr and redacts secret env values on failure", async () => {
  const secret = "sk-super-secret-token";
  const { child, pushStderr } = makeFakeChild();
  const client = new StdioMcpClient(
    stdioServer({
      env: [
        { name: "DOCS_API_KEY", value: secret },
        { name: "DOCS_LOCALE", value: "en-US" },
      ],
    }),
    { spawn: () => child as never }
  );
  const listPromise = client.listTools();

  await tick();
  pushStderr(`auth failed for key ${secret} (locale en-US)\n`);
  child.emit("exit", 1, null);

  await assert.rejects(listPromise, (error: Error) => {
    assert.match(error.message, /stderr: auth failed for key/);
    assert.match(error.message, /\[REDACTED\]/);
    assert.doesNotMatch(error.message, new RegExp(secret));
    assert.match(error.message, /locale en-US/);
    return true;
  });
  await client.dispose();
});

test("StdioMcpClient terminates the Windows process tree on dispose", async () => {
  const { child, getKillCount } = makeFakeChild();
  let terminatedPid: number | undefined;
  const client = new StdioMcpClient(stdioServer(), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => child as never,
    killProcessTree: (pid) => {
      terminatedPid = pid;
      return true;
    },
  });
  const listPromise = client.listTools();

  await tick();
  await client.dispose();

  await assert.rejects(listPromise, /disposed/i);
  assert.equal(terminatedPid, 5150);
  assert.equal(getKillCount(), 0);
});

test("StdioMcpClient keeps a shared initialization alive when one caller aborts", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });

  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  const survivingCall = client.callTool("search", { q: "surviving" });

  await tick();
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled/i);
  assert.equal(getKillCount(), 0);

  // Only the handshake is shared; each caller issues its own tools/call afterwards.
  const init = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: init.id, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await tick();
  const callBody = JSON.parse(written.at(-1)?.trim() ?? "{}") as {
    id: number;
    params: { arguments: { q: string } };
  };
  assert.equal(callBody.params.arguments.q, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [] } }) + "\n");

  await assert.doesNotReject(survivingCall);
  await client.dispose();
});

test("StdioMcpClient does not kill an initialized server when a caller aborts", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  const controller = new AbortController();
  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled|aborted/i);
  assert.equal(getKillCount(), 0);

  // The server is still usable for the next caller.
  const survivingCall = client.callTool("search", { q: "surviving" });
  await tick();
  const callBody = JSON.parse(written.at(-1)?.trim() ?? "{}") as {
    id: number;
    params: { arguments: { q: string } };
  };
  assert.equal(callBody.params.arguments.q, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [] } }) + "\n");
  await assert.doesNotReject(survivingCall);

  await client.dispose();
});

test("StdioMcpClient keeps a shared initialization alive for a concurrent listTools", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });

  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  const listPromise = client.listTools();

  await tick();
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled/i);
  assert.equal(getKillCount(), 0);

  await completeHandshake(written, pushStdout);
  assert.deepEqual(await listPromise, [{ name: "search", description: undefined, inputSchema: undefined }]);
  await client.dispose();
});

test("StdioMcpClient does not resurrect a disposed server", async () => {
  const { child } = makeFakeChild();
  let spawnCount = 0;
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => {
      spawnCount += 1;
      return child as never;
    },
  });

  await client.dispose();
  await assert.rejects(() => client.callTool("search", {}), /disposed/i);
  assert.equal(spawnCount, 0);
});
