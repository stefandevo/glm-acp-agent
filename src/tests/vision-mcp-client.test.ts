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

  // Initialized notification + tools/call should follow.
  await new Promise((r) => setImmediate(r));
  const initialized = JSON.parse(written[1]?.trim() ?? "{}") as { method: string };
  assert.equal(initialized.method, "notifications/initialized");
  const callBody = JSON.parse(written[2]?.trim() ?? "{}") as { id: number; method: string; params: { name: string; arguments: Record<string, unknown> } };
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
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }) + "\n");
  await new Promise((r) => setImmediate(r));
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: 2, error: { code: -32000, message: "quota exceeded" } }) + "\n");

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
