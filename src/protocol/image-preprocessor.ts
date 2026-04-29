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
  signal?: AbortSignal
): Promise<PreprocessedPrompt> {
  if (!blocks.some((b) => b.type === "image")) {
    return { blocks: [...blocks], cleanups: [] };
  }

  const out: Block[] = [];
  const cleanups: Array<() => Promise<void>> = [];
  let imageIndex = 0;

  for (const block of blocks) {
    if (block.type !== "image") {
      out.push(block);
      continue;
    }
    imageIndex += 1;

    if (!visionClient) {
      out.push(textBlock(`<image_attached index="${imageIndex}" mime="${block.mimeType}">image attached (not analyzed; Vision MCP unavailable)</image_attached>`));
      continue;
    }

    let imageSource = "";
    if (typeof block.uri === "string" && block.uri.length > 0) {
      imageSource = block.uri;
    } else if (typeof block.data === "string" && block.data.length > 0) {
      const dir = await mkdtemp(pathJoin(tmpdir(), "glm-acp-image-"));
      const ext = guessExtension(block.mimeType);
      const path = pathJoin(dir, `image-${imageIndex}${ext}`);
      await writeFile(path, Buffer.from(block.data, "base64"));
      imageSource = path;
      cleanups.push(async () => {
        try { await rm(dir, { recursive: true, force: true }); } catch { /* best effort */ }
      });
    } else {
      out.push(textBlock(`<image_analysis_error index="${imageIndex}">image block has neither a uri nor base64 data</image_analysis_error>`));
      continue;
    }

    try {
      const result = await visionClient.callTool("image_analysis", { image_source: imageSource }, signal);
      const text = extractText(result);
      out.push(textBlock(`<image_analysis index="${imageIndex}">\n${text}\n</image_analysis>`));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      out.push(textBlock(`<image_analysis_error index="${imageIndex}">${message}</image_analysis_error>`));
    }
  }

  return { blocks: out, cleanups };
}

function textBlock(text: string): TextBlock {
  return { type: "text", text };
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
