import { open } from "node:fs/promises";
import { StringDecoder } from "node:string_decoder";

export interface TextPage {
  text: string;
  firstLine: number;
  lastCompleteLine: number;
  totalLines?: number;
  nextLine?: number;
  truncated: boolean;
  incompleteLine?: number;
}

/**
 * Read at most maxReadBytes from disk. The byte bound applies to bytes consumed
 * from the handle too, rather than trusting a pre-read stat result.
 */
export async function readLocalTextPage(
  path: string,
  offset: number,
  limit: number,
  maxReadBytes: number,
  signal?: AbortSignal,
): Promise<TextPage> {
  const handle = await open(path, "r");
  try {
    const chunks: Buffer[] = [];
    let consumed = 0;
    let eof = false;
    while (consumed < maxReadBytes) {
      if (signal?.aborted) throw new Error("The operation was aborted");
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, maxReadBytes - consumed));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (bytesRead === 0) { eof = true; break; }
      chunks.push(Buffer.from(buffer.subarray(0, bytesRead)));
      consumed += bytesRead;
    }
    const bytes = Buffer.concat(chunks, consumed);
    const completeLines: Buffer[] = [];
    let lineStart = 0;
    for (let i = 0; i < bytes.length; i++) {
      if (bytes[i] === 0x0a) {
        completeLines.push(bytes.subarray(lineStart, i));
        lineStart = i + 1;
      }
    }
    const hasPartial = lineStart < bytes.length;
    if (eof && !hasPartial && completeLines.length === 0 && bytes.length > 0) completeLines.push(bytes);
    if (eof && hasPartial) completeLines.push(bytes.subarray(lineStart));
    const totalLines = eof ? completeLines.length : undefined;
    const safeOffset = Math.max(1, Math.floor(offset));
    const safeLimit = Math.max(1, Math.floor(limit));
    if (totalLines !== undefined && safeOffset > totalLines) {
      return { text: "", firstLine: safeOffset, lastCompleteLine: totalLines, totalLines, truncated: false };
    }
    const end = Math.min(completeLines.length, safeOffset - 1 + safeLimit);
    const selected = completeLines.slice(safeOffset - 1, end).map(line => decodeUtf8Safely(line));
    const lastCompleteLine = !eof && safeOffset > completeLines.length
      ? completeLines.length
      : safeOffset + selected.length - 1;
    const currentLine = completeLines.length + 1;
    const pageEndsAtKnownPartial = !eof && hasPartial && safeOffset <= currentLine && safeOffset + safeLimit - 1 >= currentLine;
    const pageEndsAtBudget = !eof && (hasPartial || bytes.length === maxReadBytes);
    return {
      text: selected.join("\n") + (pageEndsAtKnownPartial && selected.length === 0 ? decodeUtf8Safely(bytes.subarray(lineStart)) : ""),
      firstLine: safeOffset,
      lastCompleteLine,
      ...(totalLines === undefined ? {} : { totalLines }),
      // Advertise a next line only when the whole file is known (eof): the
      // scan always restarts from byte zero, so a budget-bound page can never
      // serve lines beyond its own prefix — following such an offset would
      // dead-end on an empty result.
      ...(totalLines !== undefined && end < totalLines ? { nextLine: end + 1 } : {}),
      truncated: pageEndsAtBudget,
      ...(pageEndsAtKnownPartial ? { incompleteLine: currentLine } : {}),
    };
  } finally {
    await handle.close();
  }
}

/** Read a complete local file only when it stays inside the configured edit budget. */
export async function readLocalTextFileBounded(path: string, maxReadBytes: number, signal?: AbortSignal): Promise<string> {
  const handle = await open(path, "r");
  try {
    const chunks: Buffer[] = [];
    let consumed = 0;
    while (true) {
      if (signal?.aborted) throw new Error("The operation was aborted");
      const buffer = Buffer.allocUnsafe(Math.min(64 * 1024, maxReadBytes + 1 - consumed));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, null);
      if (bytesRead === 0) return decodeUtf8Safely(Buffer.concat(chunks, consumed));
      consumed += bytesRead;
      if (consumed > maxReadBytes) {
        throw new Error(`file exceeds the ${maxReadBytes}-byte read/edit limit`);
      }
      chunks.push(Buffer.from(buffer.subarray(0, bytesRead)));
    }
  } finally {
    await handle.close();
  }
}

function decodeUtf8Safely(bytes: Buffer): string {
  // StringDecoder retains an incomplete trailing sequence instead of emitting
  // U+FFFD; by not calling end(), the bounded scan drops that incomplete tail.
  return new StringDecoder("utf8").write(bytes);
}
