import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join as pathJoin } from "node:path";
import type { PromptRequest } from "@agentclientprotocol/sdk";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

type Block = PromptRequest["prompt"][number];
type TextBlock = { type: "text"; text: string };

export interface PreprocessedPrompt {
  blocks: Block[];
  cleanups: Array<() => Promise<void>>;
}

/** File operations are injectable so preparation failures can be exercised without filesystem races. */
export interface ImagePreprocessorFileOps {
  mkdtemp: (prefix: string) => Promise<string>;
  writeFile: (path: string, data: Uint8Array) => Promise<void>;
  rm: (path: string, options: { recursive: boolean; force: boolean }) => Promise<void>;
}

const defaultFileOps: ImagePreprocessorFileOps = { mkdtemp, writeFile, rm };

const MAX_CONCURRENT_IMAGE_ANALYSES = 3;

/**
 * Replace every ACP image block with a text annotation containing the result
 * of a Vision MCP `image_analysis` call. ACP image blocks may carry a usable
 * URI (passed through as-is) or only inline base64 `data` (we materialize that
 * to an OS temp file and pass the path; cleanup callbacks remove it later).
 *
 * Failures from the vision client are degraded into `<image_analysis_error>`
 * annotations so a vision outage cannot crash a prompt that happens to
 * include an image.
 */
export async function preprocessImageBlocks(
  blocks: ReadonlyArray<Block>,
  visionClient: VisionMcpClient | null,
  signal?: AbortSignal,
  fileOps: ImagePreprocessorFileOps = defaultFileOps
): Promise<PreprocessedPrompt> {
  if (!blocks.some((b) => b.type === "image")) {
    return { blocks: [...blocks], cleanups: [] };
  }

  const out = new Array<Block>(blocks.length);
  const cleanups: Array<() => Promise<void>> = [];
  const imageIndexes = new Map<number, number>();
  let imageIndex = 0;
  blocks.forEach((block, blockIndex) => {
    if (block.type === "image") imageIndexes.set(blockIndex, ++imageIndex);
  });
  let nextBlockIndex = 0;
  let fatalFailure = false;

  const processBlock = async (blockIndex: number): Promise<void> => {
    const block = blocks[blockIndex];
    if (block.type !== "image") {
      out[blockIndex] = block;
      return;
    }
    const index = imageIndexes.get(blockIndex);
    if (index === undefined) throw new Error("Image index was not assigned");

    throwIfAborted(signal);

    if (!visionClient) {
      out[blockIndex] = textBlock(`<image_attached index="${index}" mime="${block.mimeType}">image attached (not analyzed; Vision MCP unavailable)</image_attached>`);
      return;
    }

    let imageSource: string;
    if (typeof block.data === "string" && block.data.length > 0) {
      const dir = await fileOps.mkdtemp(pathJoin(tmpdir(), "glm-acp-image-"));
      cleanups.push(async () => {
        try { await fileOps.rm(dir, { recursive: true, force: true }); } catch { /* best effort */ }
      });
      throwIfAborted(signal);
      const ext = guessExtension(block.mimeType);
      const path = pathJoin(dir, `image-${index}${ext}`);
      await fileOps.writeFile(path, Buffer.from(block.data, "base64"));
      throwIfAborted(signal);
      imageSource = path;
    } else if (typeof block.uri === "string" && block.uri.length > 0) {
      imageSource = block.uri;
    } else {
      out[blockIndex] = textBlock(`<image_analysis_error index="${index}">image block has neither a uri nor base64 data</image_analysis_error>`);
      return;
    }

    try {
      throwIfAborted(signal);
      const visionOperation = Promise.resolve().then(() => {
        throwIfAborted(signal);
        return visionClient.callTool("image_analysis", {
          image_source: imageSource,
          prompt: "Describe this image in detail, including any text, code, UI elements, or other visible content.",
        }, signal);
      });
      const result = await waitForAbort(visionOperation, signal);
      const text = extractText(result);
      out[blockIndex] = textBlock(`<image_analysis index="${index}">\n${text}\n</image_analysis>`);
    } catch (err) {
      if (signal?.aborted) throw err;
      const message = err instanceof Error ? err.message : String(err);
      out[blockIndex] = textBlock(`<image_analysis_error index="${index}">${message}</image_analysis_error>`);
    }
  };

  const runWorker = async (): Promise<void> => {
    while (true) {
      if (fatalFailure) return;
      throwIfAborted(signal);
      const blockIndex = nextBlockIndex++;
      if (blockIndex >= blocks.length) return;
      try {
        await processBlock(blockIndex);
      } catch (err) {
        // Any unexpected preparation failure makes this batch unusable. Let current work drain, but do not schedule more blocks.
        fatalFailure = true;
        throw err;
      }
    }
  };

  try {
    const workerCount = Math.min(MAX_CONCURRENT_IMAGE_ANALYSES, blocks.length);
    const results = await Promise.allSettled(Array.from({ length: workerCount }, () => runWorker()));
    const failure = results.find((result): result is PromiseRejectedResult => result.status === "rejected");
    if (failure) throw failure.reason;
  } catch (err) {
    await runCleanups(cleanups);
    throw err;
  }

  return { blocks: out, cleanups };
}

/**
 * Wait for the operation, rejecting on cancellation — but never before the
 * operation itself has settled. Cleanups that delete materialized images, the
 * prompt lifecycle, and session close all follow this rejection, so letting it
 * run ahead of the request would let cleanup race a still-active Vision MCP
 * call. (The vision client rejects an aborted call promptly, so this wait is
 * short in practice.) Late operation rejections are consumed either way.
 */
function waitForAbort<T>(operation: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return operation;
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const cancelError = () => new Error("Vision MCP call cancelled");
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      // Hold the rejection until the request settles so cleanup cannot race
      // the in-flight Vision MCP call. Its outcome is discarded — the prompt
      // was cancelled — but observing it here also consumes a late rejection.
      operation.then(
        () => reject(cancelError()),
        () => reject(cancelError()),
      );
    };
    signal.addEventListener("abort", onAbort, { once: true });
    operation.then(
      (value) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(error);
      },
    );
    // Attach the operation handlers before observing an already-aborted
    // signal. The operation may already be scheduled and must still have its
    // late rejection consumed even when cancellation wins immediately.
    if (signal.aborted) onAbort();
  });
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new Error("Vision MCP call cancelled");
}

async function runCleanups(cleanups: Array<() => Promise<void>>): Promise<void> {
  for (const cleanup of cleanups.splice(0)) {
    try { await cleanup(); } catch { /* best effort */ }
  }
}

/**
 * Build a list of safe, redacted diagnostic lines describing the inbound ACP
 * prompt blocks. Intended for debug logging — callers must guard with
 * `isDebugEnabled()` before calling so the string work is skipped in prod.
 *
 * Safety rules:
 * - Image blocks: log MIME, URI presence, and approximate decoded byte length.
 *   Never log the base64 payload.
 * - Resource/resource-link blocks: log the URI scheme + path basename only
 *   (no full paths, no query strings, no data: payloads).
 */
export function buildPromptBlockDiagnosticLines(blocks: ReadonlyArray<Block>): string[] {
  const counts = new Map<string, number>();
  for (const b of blocks) counts.set(b.type, (counts.get(b.type) ?? 0) + 1);

  const lines: string[] = [
    `prompt blocks: ${[...counts.entries()].map(([t, n]) => `${t}×${n}`).join(", ")}`,
  ];

  for (const b of blocks) {
    if (b.type === "image") {
      const hasUri = typeof b.uri === "string" && b.uri.length > 0;
      // Approximate decoded byte length from base64 length to avoid overstating size.
      const dataBytes = typeof b.data === "string" ? Math.floor(b.data.length * 0.75) : 0;
      lines.push(`  image block: mime=${b.mimeType} uri=${hasUri} data_bytes≈${dataBytes}`);
    } else if (b.type === "resource_link") {
      const uriSafe = safeUriSummary(b.uri);
      const mimePart = b.mimeType ? ` mime=${b.mimeType}` : "";
      lines.push(`  resource_link block: name=${b.name} uri=${uriSafe}${mimePart}`);
    } else if (b.type === "resource") {
      const res = b.resource;
      const uriSafe = safeUriSummary(res.uri);
      const mimePart = res.mimeType ? ` mime=${res.mimeType}` : "";
      lines.push(`  resource block: uri=${uriSafe}${mimePart}`);
    }
  }

  return lines;
}

function textBlock(text: string): TextBlock {
  return { type: "text", text };
}

function safeUriSummary(uri: string): string {
  if (uri.startsWith("data:")) return "data:<redacted>";
  // Normalize backslashes so Windows-style paths don't mislead URL parsing.
  const normalized = uri.replace(/\\/g, "/");
  try {
    const u = new URL(normalized);
    const segments = u.pathname.split("/").filter(Boolean);
    const basename = segments.at(-1) ?? "";
    return `${u.protocol}//...${basename ? "/" + basename : ""}`;
  } catch {
    // Not a parseable URL (e.g. a bare local path without scheme).
    const basename = normalized.split("/").filter(Boolean).at(-1) ?? normalized;
    return `.../${basename}`;
  }
}

function guessExtension(mime: string): string {
  switch (mime.toLowerCase()) {
    case "image/png": return ".png";
    case "image/jpeg":
    case "image/jpg": return ".jpg";
    case "image/gif": return ".gif";
    case "image/webp": return ".webp";
    default: return ".bin";
  }
}

function extractText(result: unknown): string {
  if (typeof result === "string") return result;
  if (typeof result !== "object" || result === null) return "";
  const content = (result as { content?: unknown }).content;
  if (!Array.isArray(content)) return "";
  return content
    .map((entry) => (typeof entry === "object" && entry !== null && typeof (entry as { text?: unknown }).text === "string" ? (entry as { text: string }).text : ""))
    .filter((s) => s.length > 0)
    .join("\n");
}
