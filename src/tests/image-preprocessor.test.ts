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
  assert.equal(typeof seen?.["prompt"], "string", "callTool must always receive a prompt field");
  // Image block must have been replaced with a text annotation.
  assert.equal(result.blocks.length, 2);
  const last = result.blocks[1] as { type: string; text?: string };
  assert.equal(last.type, "text");
  assert.match(last.text ?? "", /<image_analysis index="1">[\s\S]*A cat\.[\s\S]*<\/image_analysis>/);
});

test("preprocessImageBlocks prefers base64 data over URI when both are present", async () => {
  let seenSource = "";
  let seenPrompt: unknown;
  const result = await preprocessImageBlocks(
    [{ type: "image", data: "AAAA", mimeType: "image/png", uri: "https://example.com/should-not-use.png" }],
    makeClient(async (_name, args) => {
      seenSource = String(args["image_source"]);
      seenPrompt = args["prompt"];
      return { content: [{ type: "text", text: "blank image" }] };
    })
  );
  for (const c of result.cleanups) await c();
  assert.ok(!seenSource.startsWith("https://"), "URI must not be used when base64 data is available");
  assert.ok(seenSource.length > 0, "a temp file path must have been passed");
  assert.equal(typeof seenPrompt, "string", "callTool must receive a prompt field");
});

test("preprocessImageBlocks materializes inline data to a temp file and cleans it up", async () => {
  let seenPath = "";
  let seenPrompt: unknown;
  const result = await preprocessImageBlocks(
    [{ type: "image", data: "AAAA", mimeType: "image/png" }],
    makeClient(async (_name, args) => {
      seenPath = String(args["image_source"]);
      seenPrompt = args["prompt"];
      assert.ok(existsSync(seenPath), "temp file must exist while vision MCP is invoked");
      const bytes = readFileSync(seenPath);
      assert.equal(bytes.length, 3); // base64 "AAAA" = 3 bytes
      return { content: [{ type: "text", text: "blank" }] };
    })
  );
  // Run cleanups after the call returns and verify the file is gone.
  for (const c of result.cleanups) await c();
  assert.equal(existsSync(seenPath), false);
  assert.equal(typeof seenPrompt, "string", "callTool must receive a prompt field");
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


test("preprocessImageBlocks bounds parallel analysis while preserving original block order", async () => {
  let active = 0;
  let peak = 0;
  const images = Array.from({ length: 5 }, (_, index) => ({
    type: "image" as const,
    data: "",
    mimeType: "image/png",
    uri: `fixture://image-${index + 1}`,
  }));
  const result = await preprocessImageBlocks(
    [{ type: "text", text: "before" }, ...images, { type: "text", text: "after" }],
    makeClient(async (_name, args) => {
      active += 1;
      peak = Math.max(peak, active);
      const source = String(args["image_source"]);
      const index = Number(source.at(-1));
      await new Promise((resolve) => setTimeout(resolve, (6 - index) * 5));
      active -= 1;
      return { content: [{ type: "text", text: source }] };
    }),
  );
  assert.equal(peak, 3);
  assert.equal(result.blocks.length, 7);
  assert.deepEqual(result.blocks.slice(0, 1), [{ type: "text", text: "before" }]);
  const annotations = result.blocks.slice(1, 6).map((block) => (block as { text: string }).text);
  assert.deepEqual(annotations, images.map((_, index) =>
    `<image_analysis index="${index + 1}">\nfixture://image-${index + 1}\n</image_analysis>`));
  assert.deepEqual(result.blocks.slice(6), [{ type: "text", text: "after" }]);
});

test("preprocessImageBlocks stops scheduling after inline preparation fails and drains in-flight URI analyses", async () => {
  let visionCalls = 0;
  let cleaned = 0;
  let resolveTwoAnalysesStarted!: () => void;
  const twoAnalysesStarted = new Promise<void>((resolve) => { resolveTwoAnalysesStarted = resolve; });
  let releaseAnalyses!: () => void;
  const analysesReleased = new Promise<void>((resolve) => { releaseAnalyses = resolve; });
  let resolvePreparationAttempted!: () => void;
  const preparationAttempted = new Promise<void>((resolve) => { resolvePreparationAttempted = resolve; });
  const preparationError = new Error("cannot materialize inline image");

  const result = preprocessImageBlocks(
    [
      { type: "image", data: "AAAA", mimeType: "image/png" },
      { type: "image", data: "", mimeType: "image/png", uri: "fixture://uri-1" },
      { type: "image", data: "", mimeType: "image/png", uri: "fixture://uri-2" },
      { type: "image", data: "", mimeType: "image/png", uri: "fixture://uri-3" },
    ],
    makeClient(async () => {
      visionCalls += 1;
      if (visionCalls === 2) resolveTwoAnalysesStarted();
      await analysesReleased;
      return { content: [{ type: "text", text: "analysis" }] };
    }),
    undefined,
    {
      mkdtemp: async () => "/tmp/phase5-image-preparation-failure",
      writeFile: async () => {
        await twoAnalysesStarted;
        resolvePreparationAttempted();
        throw preparationError;
      },
      rm: async () => { cleaned += 1; },
    },
  );

  await preparationAttempted;
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(visionCalls, 2, "only the already-running URI analyses should have started");
  assert.equal(cleaned, 0, "temporary files must remain until in-flight analyses drain");

  releaseAnalyses();
  await assert.rejects(result, (error: unknown) => error === preparationError);
  assert.equal(visionCalls, 2, "queued URI images must not be analyzed after preparation fails");
  assert.equal(cleaned, 1, "the failed inline image directory must be cleaned after draining");
});

test("preprocessImageBlocks waits for aborted parallel calls and cleans up every image file", async () => {
  const controller = new AbortController();
  let started = 0;
  let cleaned = 0;
  const result = preprocessImageBlocks(
    Array.from({ length: 5 }, () => ({ type: "image" as const, data: "AAAA", mimeType: "image/png" })),
    makeClient(async (_name, _args, signal) => {
      started += 1;
      return new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new Error("fixture cancelled")), { once: true });
      });
    }),
    controller.signal,
    {
      mkdtemp: async () => `/tmp/phase5-image-${started}`,
      writeFile: async () => {},
      rm: async () => { cleaned += 1; },
    },
  );
  while (started < 3) await new Promise((resolve) => setImmediate(resolve));
  controller.abort();
  await assert.rejects(result, /cancelled/);
  assert.equal(started, 3, "cancellation must not schedule more analyses");
  assert.equal(cleaned, 3, "all materialized image directories must be removed");
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
