import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { preprocessImageBlocks } from "../protocol/image-preprocessor.js";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

function makeClient(impl: VisionMcpClient["callTool"]): VisionMcpClient {
  return { callTool: impl, async dispose() {} };
}

test("preprocessImageBlocks returns the input unchanged when no images are present", async () => {
  const result = await preprocessImageBlocks(
    [{ type: "text", text: "hi" }],
    makeClient(async () => { throw new Error("should not be called"); })
  );
  assert.deepEqual(result.blocks, [{ type: "text", text: "hi" }]);
  assert.equal(result.cleanups.length, 0);
});

test("preprocessImageBlocks forwards a remote URI directly to the vision client", async () => {
  let seen: Record<string, unknown> | null = null;
  const result = await preprocessImageBlocks(
    [
      { type: "text", text: "What is this?" },
      { type: "image", data: "", mimeType: "image/png", uri: "https://example.com/cat.png" },
    ],
    makeClient(async (_name, args) => {
      seen = args;
      return { content: [{ type: "text", text: "A cat." }] };
    })
  );
  assert.equal(seen?.["image_source"], "https://example.com/cat.png");
  // Image block must have been replaced with a text annotation.
  assert.equal(result.blocks.length, 2);
  const last = result.blocks[1] as { type: string; text?: string };
  assert.equal(last.type, "text");
  assert.match(last.text ?? "", /<image_analysis index="1">[\s\S]*A cat\.[\s\S]*<\/image_analysis>/);
});

test("preprocessImageBlocks materializes inline data to a temp file and cleans it up", async () => {
  let seenPath = "";
  const result = await preprocessImageBlocks(
    [{ type: "image", data: "AAAA", mimeType: "image/png" }],
    makeClient(async (_name, args) => {
      seenPath = String(args["image_source"]);
      assert.ok(existsSync(seenPath), "temp file must exist while vision MCP is invoked");
      const bytes = readFileSync(seenPath);
      assert.equal(bytes.length, 3); // base64 "AAAA" = 3 bytes
      return { content: [{ type: "text", text: "blank" }] };
    })
  );
  // Run cleanups after the call returns and verify the file is gone.
  for (const c of result.cleanups) await c();
  assert.equal(existsSync(seenPath), false);
});

test("preprocessImageBlocks degrades gracefully when the vision client fails", async () => {
  const result = await preprocessImageBlocks(
    [
      { type: "text", text: "Look:" },
      { type: "image", data: "", mimeType: "image/png", uri: "https://example.com/x.png" },
    ],
    makeClient(async () => { throw new Error("quota exceeded"); })
  );
  const annotation = result.blocks.at(-1) as { type: string; text?: string };
  assert.equal(annotation.type, "text");
  assert.match(annotation.text ?? "", /image_analysis_error/);
  assert.match(annotation.text ?? "", /quota exceeded/);
});

test("preprocessImageBlocks skips vision when no client is given and notes the image", async () => {
  const result = await preprocessImageBlocks(
    [{ type: "image", data: "AAAA", mimeType: "image/png" }],
    null
  );
  const block = result.blocks[0] as { type: string; text?: string };
  assert.equal(block.type, "text");
  assert.match(block.text ?? "", /image attached \(not analyzed/i);
  assert.equal(result.cleanups.length, 0);
});
