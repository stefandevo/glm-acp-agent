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
