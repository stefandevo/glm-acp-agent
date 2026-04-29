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
  kill: (signal?: string) => boolean;
}

function makeFakeChild(): { child: FakeChild; written: string[]; pushStdout: (line: string) => void } {
  const written: string[] = [];
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
    kill: () => true,
  }) as FakeChild;
  const pushStdout = (line: string) => stdout.push(line);
  return { child, written, pushStdout };
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

