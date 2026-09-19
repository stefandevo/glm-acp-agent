/** A transport EOF is not proof that a provider completed its response. */
export class IncompleteModelStreamError extends Error {
  override name = "IncompleteModelStreamError";
}

export type TerminalReason = "stop" | "tool_calls" | "length" | "content_filter";

export function validateStreamCompletion(
  reason: string | undefined,
  calls: ReadonlyArray<{ id: string; name: string }>,
): TerminalReason {
  if (reason !== "stop" && reason !== "tool_calls" && reason !== "length" && reason !== "content_filter") {
    throw new IncompleteModelStreamError("Incomplete model stream: no supported terminal reason.");
  }
  // Truncated/filtered calls are never executable, even when their arguments
  // happen to look complete. Their fragments are intentionally discarded.
  if (reason === "length" || reason === "content_filter") return reason;
  if (reason === "stop" && calls.length > 0) {
    throw new IncompleteModelStreamError("Inconsistent model stream: stop with tool-call fragments.");
  }
  if (reason === "tool_calls") {
    const ids = new Set<string>();
    if (calls.length === 0) throw new IncompleteModelStreamError("Incomplete model tool-call batch.");
    for (const call of calls) {
      if (!call.id.trim() || !call.name.trim() || ids.has(call.id)) {
        throw new IncompleteModelStreamError("Incomplete or duplicate model tool-call batch.");
      }
      ids.add(call.id);
    }
  }
  return reason;
}
