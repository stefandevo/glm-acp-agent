/** Keep a string within a UTF-8 byte budget without creating replacement characters. */
export function boundToolResult(text: string, maxBytes: number): string {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 128) {
    throw new Error("tool result limit must be an integer of at least 128 bytes");
  }
  const totalBytes = Buffer.byteLength(text, "utf8");
  if (totalBytes <= maxBytes) return text;

  let prefixBudget = Math.floor(maxBytes * 0.75);
  let suffixBudget = maxBytes - prefixBudget;
  for (let attempt = 0; attempt < 4; attempt++) {
    const prefix = takeUtf8Prefix(text, prefixBudget);
    const suffix = takeUtf8Suffix(text, suffixBudget);
    const omitted = totalBytes - Buffer.byteLength(prefix, "utf8") - Buffer.byteLength(suffix, "utf8");
    const marker = `\n[… ${Math.max(0, omitted)} bytes omitted …]\n`;
    const markerBytes = Buffer.byteLength(marker, "utf8");
    const used = Buffer.byteLength(prefix, "utf8") + markerBytes + Buffer.byteLength(suffix, "utf8");
    if (used <= maxBytes) return `${prefix}${marker}${suffix}`;
    const excess = used - maxBytes;
    prefixBudget = Math.max(0, prefixBudget - Math.ceil(excess * 0.75));
    suffixBudget = Math.max(0, suffixBudget - Math.floor(excess * 0.25));
  }
  // The configured minimum leaves ample room for this constant marker.
  const marker = `\n[… ${totalBytes} bytes omitted …]\n`;
  return marker;
}

export function takeUtf8Prefix(text: string, maxBytes: number): string {
  let used = 0;
  let end = 0;
  for (const character of text) {
    const bytes = Buffer.byteLength(character, "utf8");
    if (used + bytes > maxBytes) break;
    used += bytes;
    end += character.length;
  }
  return text.slice(0, end);
}

export function takeUtf8Suffix(text: string, maxBytes: number): string {
  let used = 0;
  let start = text.length;
  while (start > 0) {
    let characterStart = start - 1;
    const codeUnit = text.charCodeAt(characterStart);
    if (
      codeUnit >= 0xdc00 &&
      codeUnit <= 0xdfff &&
      characterStart > 0 &&
      text.charCodeAt(characterStart - 1) >= 0xd800 &&
      text.charCodeAt(characterStart - 1) <= 0xdbff
    ) {
      characterStart -= 1;
    }
    const character = text.slice(characterStart, start);
    const bytes = Buffer.byteLength(character, "utf8");
    if (used + bytes > maxBytes) break;
    used += bytes;
    start = characterStart;
  }
  return text.slice(start);
}
