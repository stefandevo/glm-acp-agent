/** Maximum bytes accepted from one MCP HTTP response or stdio JSON-RPC frame. */
export const MCP_RESPONSE_LIMIT_BYTES = 8 * 1024 * 1024;

/** Read a response without letting a peer force an unbounded text() allocation. */
export async function readMcpResponseText(response: Response): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parts: string[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > MCP_RESPONSE_LIMIT_BYTES) {
        throw new Error(`MCP response exceeds ${MCP_RESPONSE_LIMIT_BYTES}-byte limit`);
      }
      parts.push(decoder.decode(value, { stream: true }));
    }
    parts.push(decoder.decode());
    return parts.join("");
  } catch (err) {
    await reader.cancel().catch(() => undefined);
    throw err;
  } finally {
    reader.releaseLock();
  }
}
