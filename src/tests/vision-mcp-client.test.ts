import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { Readable, Writable } from "node:stream";
import { StdioVisionMcpClient } from "../tools/vision-mcp-client.js";

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
  const stderr = new Readable({ read() { /* noop */ } });
  const child = Object.assign(new EventEmitter(), {
    stdin,
    stdout,
    stderr,
    pid: 4242,
    exitCode: null,
    kill: () => {
      killCount += 1;
      return true;
    },
  }) as FakeChild;
  const pushStdout = (line: string) => stdout.push(line);
  const pushStderr = (line: string) => stderr.push(line);
  return { child, written, pushStdout, pushStderr, getKillCount: () => killCount };
}

test("StdioVisionMcpClient initializes once and forwards tools/call", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioVisionMcpClient({
    apiKey: "key-1",
    spawn: () => child as never,
  });

  const callPromise = client.callTool("image_analysis", {
    image_source: "/tmp/x.png",
    prompt: "describe",
  });

  // Wait a tick so the client has written initialize.
  await new Promise((r) => setImmediate(r));
  const initLine = written[0] ?? "";
  const initBody = JSON.parse(initLine.trim()) as { id: number; method: string };
  assert.equal(initBody.method, "initialize");

  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: initBody.id, result: { protocolVersion: "2025-06-18" } }) + "\n");

  // notifications/initialized and tools/list should follow.
  await new Promise((r) => setImmediate(r));
  const initialized = JSON.parse(written[1]?.trim() ?? "{}") as { method: string };
  assert.equal(initialized.method, "notifications/initialized");

  const toolsListBody = JSON.parse(written[2]?.trim() ?? "{}") as { id: number; method: string };
  assert.equal(toolsListBody.method, "tools/list");

  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: toolsListBody.id, result: { tools: [{ name: "image_analysis" }] } }) + "\n");

  // tools/call should follow after tools/list completes.
  await new Promise((r) => setImmediate(r));
  const callBody = JSON.parse(written[3]?.trim() ?? "{}") as { id: number; method: string; params: { name: string; arguments: Record<string, unknown> } };
  assert.equal(callBody.method, "tools/call");
  assert.equal(callBody.params.name, "image_analysis");
  assert.equal(callBody.params.arguments["image_source"], "/tmp/x.png");

  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [{ type: "text", text: "a cat" }] } }) + "\n");

  const result = await callPromise;
  assert.deepEqual(result, { content: [{ type: "text", text: "a cat" }] });

  await client.dispose();
});

test("StdioVisionMcpClient surfaces JSON-RPC errors with method context", async () => {
  const { child, pushStdout } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });

  const callPromise = client.callTool("image_analysis", { image_source: "x" });

  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }) + "\n"); // init response
  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 2, result: { tools: [{ name: "image_analysis" }] } }) + "\n"); // tools/list
  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 3, error: { code: -32000, message: "quota exceeded" } }) + "\n"); // tools/call error

  await assert.rejects(callPromise, /Vision MCP image_analysis failed.*quota exceeded/);
  await client.dispose();
});

test("StdioVisionMcpClient explains a missing npx as an actionable error", async () => {
  const client = new StdioVisionMcpClient({
    apiKey: "k",
    spawn: () => {
      const err = Object.assign(new Error("spawn npx ENOENT"), { code: "ENOENT" });
      throw err;
    },
  });
  await assert.rejects(
    () => client.callTool("image_analysis", { image_source: "x" }),
    /npx.*not found/i
  );
});

test("StdioVisionMcpClient rejects an asynchronous spawn error instead of hanging", async () => {
  const { child } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });
  const callPromise = client.callTool("image_analysis", { image_source: "x" });

  await new Promise((r) => setImmediate(r));
  child.emit("error", Object.assign(new Error("spawn npx ENOENT"), { code: "ENOENT" }));

  await assert.rejects(callPromise, /could not launch npx/i);
  await client.dispose();
});

test("StdioVisionMcpClient times out initialization and terminates the child", async () => {
  const { child, getKillCount } = makeFakeChild();
  const client = new StdioVisionMcpClient({
    apiKey: "k",
    spawn: () => child as never,
    initializationTimeoutMs: 50,
  });

  await assert.rejects(
    () => client.callTool("image_analysis", { image_source: "x" }),
    /timed out after \d+ms/i,
  );
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioVisionMcpClient aborts initialization and terminates the child", async () => {
  const { child, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });
  const callPromise = client.callTool("image_analysis", { image_source: "x" }, controller.signal);

  await new Promise((r) => setImmediate(r));
  controller.abort();

  await assert.rejects(callPromise, /Vision MCP call cancelled/i);
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioVisionMcpClient keeps shared initialization alive for another caller", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });
  const cancelledCall = client.callTool("image_analysis", { image_source: "cancelled" }, controller.signal);
  const survivingCall = client.callTool("image_analysis", { image_source: "surviving" });

  await new Promise((r) => setImmediate(r));
  controller.abort();
  await assert.rejects(cancelledCall, /Vision MCP call cancelled/i);
  assert.equal(getKillCount(), 0);

  const initBody = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: initBody.id, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await new Promise((r) => setImmediate(r));
  const toolsListBody = JSON.parse(written[2]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: toolsListBody.id, result: { tools: [{ name: "image_analysis" }] } }) + "\n");
  await new Promise((r) => setImmediate(r));
  const callBody = JSON.parse(written[3]?.trim() ?? "{}") as { id: number; params: { arguments: { image_source: string } } };
  assert.equal(callBody.params.arguments.image_source, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [{ type: "text", text: "ok" }] } }) + "\n");

  await assert.doesNotReject(survivingCall);
  await client.dispose();
});

test("StdioVisionMcpClient does not kill an initialized server when a caller aborts", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });
  const firstCall = client.callTool("image_analysis", { image_source: "first" });

  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 2, result: { tools: [{ name: "image_analysis" }] } }) + "\n");
  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 3, result: { content: [{ type: "text", text: "first ok" }] } }) + "\n");
  await assert.doesNotReject(firstCall);

  const controller = new AbortController();
  const cancelledCall = client.callTool("image_analysis", { image_source: "cancelled" }, controller.signal);
  controller.abort();
  await assert.rejects(cancelledCall, /Vision MCP call cancelled|aborted/i);
  assert.equal(getKillCount(), 0);

  const survivingCall = client.callTool("image_analysis", { image_source: "surviving" });
  await new Promise((r) => setImmediate(r));
  const callBody = JSON.parse(written.at(-1)?.trim() ?? "{}") as { id: number; params: { arguments: { image_source: string } } };
  assert.equal(callBody.params.arguments.image_source, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [{ type: "text", text: "still ok" }] } }) + "\n");
  await assert.doesNotReject(survivingCall);

  await client.dispose();
});

test("StdioVisionMcpClient launches npx through cmd.exe on Windows", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  let windowsHide = false;
  const client = new StdioVisionMcpClient({
    apiKey: "k",
    platform: "win32",
    comSpec: "C:\\Windows\\System32\\cmd.exe",
    spawn: (capturedCommand, capturedArgs, options) => {
      command = capturedCommand;
      args = capturedArgs;
      windowsHide = options.windowsHide === true;
      return child as never;
    },
  });
  const callPromise = client.callTool("image_analysis", { image_source: "x" });

  await new Promise((r) => setImmediate(r));
  child.emit("error", new Error("test stop"));
  await assert.rejects(callPromise, /test stop/i);

  assert.equal(command, "C:\\Windows\\System32\\cmd.exe");
  assert.deepEqual(args.slice(0, 5), ["/d", "/s", "/c", "npx", "-y"]);
  assert.equal(windowsHide, true);
  await client.dispose();
});

test("StdioVisionMcpClient rejects unsafe Windows package specs before spawning", async () => {
  let spawned = false;
  const client = new StdioVisionMcpClient({
    apiKey: "k",
    platform: "win32",
    packageSpec: "@z_ai/mcp-server@latest & echo injected",
    spawn: () => {
      spawned = true;
      return makeFakeChild().child as never;
    },
  });

  await assert.rejects(
    () => client.callTool("image_analysis", { image_source: "x" }),
    /unsafe npm package spec/i,
  );
  assert.equal(spawned, false);
});

test("StdioVisionMcpClient terminates the Windows process tree on dispose", async () => {
  const { child, getKillCount } = makeFakeChild();
  let terminatedPid: number | undefined;
  const client = new StdioVisionMcpClient({
    apiKey: "k",
    platform: "win32",
    spawn: () => child as never,
    killProcessTree: (pid) => {
      terminatedPid = pid;
      return true;
    },
  });
  const callPromise = client.callTool("image_analysis", { image_source: "x" });

  await new Promise((r) => setImmediate(r));
  await client.dispose();

  await assert.rejects(callPromise, /client disposed/i);
  assert.equal(terminatedPid, 4242);
  assert.equal(getKillCount(), 0);
});

test("StdioVisionMcpClient drains and redacts stderr on failure", async () => {
  const apiKey = "secret-vision-key";
  const { child, pushStderr } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey, spawn: () => child as never });
  const callPromise = client.callTool("image_analysis", { image_source: "x" });

  await new Promise((r) => setImmediate(r));
  pushStderr(`provider failed for ${apiKey}\n`);
  child.emit("exit", 1, null);

  await assert.rejects(callPromise, (error: Error) => {
    assert.match(error.message, /stderr: provider failed/);
    assert.match(error.message, /\[REDACTED\]/);
    assert.doesNotMatch(error.message, new RegExp(apiKey));
    return true;
  });
  await client.dispose();
});

test("StdioVisionMcpClient resolves tool name via keyword fallback when server uses a different name", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });

  const callPromise = client.callTool("image_analysis", { image_source: "/img.png" });

  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await new Promise((r) => setImmediate(r));
  // Server uses "analyzeImage" instead of "image_analysis"
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 2, result: { tools: [{ name: "analyzeImage" }] } }) + "\n");
  await new Promise((r) => setImmediate(r));
  const callBody = JSON.parse(written[3]?.trim() ?? "{}") as { id: number; params: { name: string } };
  assert.equal(callBody.params.name, "analyzeImage");

  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [{ type: "text", text: "dog" }] } }) + "\n");
  const result = await callPromise;
  assert.deepEqual(result, { content: [{ type: "text", text: "dog" }] });
  await client.dispose();
});

test("StdioVisionMcpClient retries once on tool-not-found after re-discovering tools", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioVisionMcpClient({ apiKey: "k", spawn: () => child as never });

  const callPromise = client.callTool("image_analysis", { image_source: "/img.png" });

  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await new Promise((r) => setImmediate(r));
  // Initial discovery: tool name contains "image" so keyword matches
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 2, result: { tools: [{ name: "old_image_tool" }] } }) + "\n");
  await new Promise((r) => setImmediate(r));
  // written[3] = first tools/call (name resolved to "old_image_tool" via keyword)
  const firstCall = JSON.parse(written[3]?.trim() ?? "{}") as { id: number; params: { name: string } };
  assert.equal(firstCall.params.name, "old_image_tool");
  // tools/call fails with tool-not-found
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: firstCall.id, error: { code: -32601, message: "Tool old_image_tool not found" } }) + "\n");
  await new Promise((r) => setImmediate(r));
  // written[4] = re-discovery tools/list
  const rediscoverBody = JSON.parse(written[4]?.trim() ?? "{}") as { id: number; method: string };
  assert.equal(rediscoverBody.method, "tools/list");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: rediscoverBody.id, result: { tools: [{ name: "analyzeImage" }] } }) + "\n");
  await new Promise((r) => setImmediate(r));
  // written[5] = retry tools/call with updated name
  const retryCallBody = JSON.parse(written[5]?.trim() ?? "{}") as { id: number; params: { name: string } };
  assert.equal(retryCallBody.params.name, "analyzeImage");

  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: retryCallBody.id, result: { content: [{ type: "text", text: "cat" }] } }) + "\n");
  const result = await callPromise;
  assert.deepEqual(result, { content: [{ type: "text", text: "cat" }] });
  await client.dispose();
});

