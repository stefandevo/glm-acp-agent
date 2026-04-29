import { debug } from "../llm/logger.js";

/** Maps agent-side argument names to upstream MCP property names. */
const ARG_ALIASES: Record<string, string> = {
  query: "search_query",
};

/**
 * Remaps argument keys to match the upstream MCP tool's `inputSchema.properties`.
 * - If a key already matches a target property, it is passed through unchanged.
 * - Otherwise, the alias table is consulted; if the alias exists in the target properties, the key is remapped.
 * - If no schema is available (empty `targetProperties`), all arguments are passed through unchanged.
 */
export function remapArguments(
  requestedArgs: Record<string, unknown>,
  targetProperties: string[]
): Record<string, unknown> {
  if (!targetProperties.length) return requestedArgs;
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(requestedArgs)) {
    if (targetProperties.includes(key)) {
      result[key] = value;
    } else {
      const alias = ARG_ALIASES[key];
      if (alias && targetProperties.includes(alias)) {
        debug(`mcp-arg-remap: remapped arg "${key}" → "${alias}"`);
        result[alias] = value;
      } else {
        result[key] = value;
      }
    }
  }
  return result;
}

/**
 * Resolves a requested tool name against the list of available tools discovered via `tools/list`.
 * - Exact match takes priority.
 * - Falls back to keyword-based search (e.g. "search", "reader", "image").
 * - If `availableTools` is empty (discovery not available), returns `requestedName` unchanged.
 * - Throws a descriptive error if no match is found.
 */
export function resolveToolName(
  requestedName: string,
  availableTools: string[],
  context: string
): string {
  if (!availableTools.length) return requestedName;
  if (availableTools.includes(requestedName)) return requestedName;

  const keywords = extractToolKeywords(requestedName);
  for (const keyword of keywords) {
    const match = availableTools.find((t) => t.toLowerCase().includes(keyword));
    if (match) return match;
  }

  throw new Error(
    `Tool "${requestedName}" not available on ${context}. Available tools: [${availableTools.join(", ")}]`
  );
}

function extractToolKeywords(name: string): string[] {
  const lower = name.toLowerCase();
  const keywords: string[] = [];
  if (lower.includes("search")) keywords.push("search");
  if (lower.includes("reader")) keywords.push("reader");
  if (
    lower.includes("image") ||
    lower.includes("vision") ||
    lower.includes("analysis") ||
    lower.includes("recognition")
  ) {
    keywords.push("image", "vision", "analysis", "recognition");
  }
  return keywords;
}
