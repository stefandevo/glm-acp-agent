import test from "node:test";
import assert from "node:assert/strict";
import { MCP_RESPONSE_LIMIT_BYTES, readMcpResponseText } from "../tools/mcp-response-limit.js";

test("MCP response reader preserves multibyte characters split across chunks", async () => {
  const encoder = new TextEncoder();
  const bytes = encoder.encode("A🌟B");
  const response = new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes.slice(0, 2));
      controller.enqueue(bytes.slice(2, 4));
      controller.enqueue(bytes.slice(4));
      controller.close();
    },
  }));
  assert.equal(await readMcpResponseText(response), "A🌟B");
});

test("MCP response reader cancels the body before accumulating oversized input", async () => {
  let cancelled = false;
  const response = new Response(new ReadableStream<Uint8Array>({
    pull(controller) { controller.enqueue(new Uint8Array(1_048_576)); },
    cancel() { cancelled = true; },
  }));
  await assert.rejects(readMcpResponseText(response), new RegExp(String(MCP_RESPONSE_LIMIT_BYTES)));
  assert.equal(cancelled, true);
});
