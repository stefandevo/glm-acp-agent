import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { preprocessImageBlocks, buildPromptBlockDiagnosticLines } from "../protocol/image-preprocessor.js";
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

// ---------------------------------------------------------------------------
// buildPromptBlockDiagnosticLines
// ---------------------------------------------------------------------------

test("buildPromptBlockDiagnosticLines summarizes block types without leaking base64", () => {
  const base64Data = Buffer.from("fake image payload data").toString("base64");
  const lines = buildPromptBlockDiagnosticLines([
    { type: "text", text: "hello" },
    { type: "image", data: base64Data, mimeType: "image/jpeg" },
  ]);
  const summary = lines[0] ?? "";
  assert.ok(summary.includes("text×1"), "summary must count text blocks");
  assert.ok(summary.includes("image×1"), "summary must count image blocks");
  for (const line of lines) {
    assert.ok(!line.includes(base64Data), "base64 payload must not appear in diagnostic output");
  }
  const imageLine = lines.find((l) => l.includes("image block"));
  assert.ok(imageLine, "image block diagnostic line must be present");
  assert.ok(imageLine!.includes("data_bytes"), "approximate byte count must be logged");
  assert.ok(imageLine!.includes("uri=false"), "URI presence must be false when uri is absent");
});

test("buildPromptBlockDiagnosticLines logs approximate decoded byte count for base64 data", () => {
  // 4 base64 chars ≈ 3 decoded bytes; Math.floor(4 * 0.75) = 3
  const lines = buildPromptBlockDiagnosticLines([
    { type: "image", data: "AAAA", mimeType: "image/png" },
  ]);
  const imageLine = lines.find((l) => l.includes("image block")) ?? "";
  assert.ok(imageLine.includes("data_bytes≈3"), "should log approximate decoded byte count");
});

test("buildPromptBlockDiagnosticLines marks URI presence when uri is set", () => {
  const lines = buildPromptBlockDiagnosticLines([
    { type: "image", data: "", mimeType: "image/png", uri: "https://example.com/cat.png" },
  ]);
  const imageLine = lines.find((l) => l.includes("image block")) ?? "";
  assert.ok(imageLine.includes("uri=true"), "URI presence must be true when uri is set");
});

test("buildPromptBlockDiagnosticLines logs safe URI basename for resource_link", () => {
  const lines = buildPromptBlockDiagnosticLines([
    { type: "resource_link", uri: "file:///home/user/private/secret-config.ts", name: "secret-config.ts" },
  ]);
  const rl = lines.find((l) => l.includes("resource_link block")) ?? "";
  assert.ok(rl.length > 0, "resource_link diagnostic line must be present");
  assert.ok(!rl.includes("/home/user/private/"), "full directory path must not appear");
  assert.ok(rl.includes("secret-config.ts"), "basename should appear");
});

test("buildPromptBlockDiagnosticLines redacts data: URIs for resources", () => {
  const lines = buildPromptBlockDiagnosticLines([
    {
      type: "resource",
      resource: { uri: "data:text/plain;base64,SGVsbG8=", text: "Hello" },
    },
  ]);
  const rl = lines.find((l) => l.includes("resource block")) ?? "";
  assert.ok(rl.includes("data:<redacted>"), "data: URI must be redacted");
  assert.ok(!rl.includes("SGVsbG8="), "base64 payload in data URI must not appear");
});
