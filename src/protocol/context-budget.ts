import type { GlmMessage } from "../llm/glm-client.js";
import { boundToolResult } from "../tools/tool-output.js";
import type { ToolDefinition } from "../tools/definitions.js";

export interface ContextBudget {
  contextWindow: number;
  maxOutputTokens: number;
  toolSchemaTokens: number;
  safetyTokens: number;
}

export interface CompactionResult {
  messages: GlmMessage[];
  changed: boolean;
  removedExchanges: number;
  reducedToolResults: number;
  estimatedTokens: number;
}

const IMAGE_PART_TOKENS = 1600;
const MESSAGE_OVERHEAD_TOKENS = 4;
const ACTIVE_TOOL_RESULT_BYTES = 4096;

/** A deliberately labelled heuristic; providers do not expose their tokenizer. */
export function estimateMessagesTokens(messages: readonly GlmMessage[]): number {
  return messages.reduce((total, message) => total + estimateMessageTokens(message), 0);
}

export function availableInputTokens(budget: ContextBudget): number {
  return budget.contextWindow - budget.maxOutputTokens - budget.toolSchemaTokens - budget.safetyTokens;
}

export function estimateSerializedTokens(value: unknown): number {
  const text = JSON.stringify(value);
  return Math.max(Math.ceil(text.length / 4), Math.ceil(Buffer.byteLength(text, "utf8") / 2));
}

/** Cache schema cost by catalog identity; replace the catalog or invalidate it after in-place edits. */
export class ToolSchemaTokenCache {
  private readonly estimates = new WeakMap<readonly ToolDefinition[], number>();

  constructor(private readonly estimate: (value: unknown) => number = estimateSerializedTokens) {}

  get(definitions: readonly ToolDefinition[]): number {
    const cached = this.estimates.get(definitions);
    if (cached !== undefined) return cached;
    const estimate = this.estimate(definitions);
    this.estimates.set(definitions, estimate);
    return estimate;
  }

  invalidate(definitions: readonly ToolDefinition[]): void {
    this.estimates.delete(definitions);
  }
}

/** Guard against retaining a tool result whose matching assistant call was removed. */
export function assertValidHistory(messages: readonly GlmMessage[]): void {
  const calls = new Set<string>();
  for (const message of messages) {
    if (message.role === "assistant") for (const call of message.tool_calls ?? []) calls.add(call.id);
    if (message.role === "tool" && !calls.has(message.tool_call_id)) {
      throw new Error(`invalid history: tool result ${message.tool_call_id} has no retained assistant call`);
    }
  }
}

/** Compact only complete user exchanges; the final user exchange remains live. */
export interface CompactionMetrics {
  messageEstimateCount: number;
}

export function compactToBudget(
  messages: readonly GlmMessage[],
  budget: ContextBudget,
  force: boolean,
  metrics?: CompactionMetrics,
): CompactionResult {
  const available = availableInputTokens(budget);
  const messageCosts = new Map<GlmMessage, number>();
  let initial = 0;
  for (const message of messages) {
    const cost = estimateOneMessage(message, metrics);
    messageCosts.set(message, cost);
    initial += cost;
  }
  if (messages.length <= 1 || (!force && initial <= Math.floor(available * 0.9))) {
    return { messages: [...messages], changed: false, removedExchanges: 0, reducedToolResults: 0, estimatedTokens: initial };
  }
  const target = Math.floor(available * 0.8);
  const desired = force ? Math.min(target, Math.floor(initial / 2)) : target;
  const system = messages[0]?.role === "system" ? messages[0] : undefined;
  const turns = groupTurns(system ? messages.slice(1) : messages);
  if (turns.length === 0) return { messages: [...messages], changed: false, removedExchanges: 0, reducedToolResults: 0, estimatedTokens: initial };

  const currentTurns = turns.map(turn => [...turn]);
  const currentTurnCosts = turns.map(turn => turn.map(message => messageCosts.get(message)!));
  const turnCosts = currentTurnCosts.map(costs => costs.reduce((total, cost) => total + cost, 0));
  let reducedToolResults = 0;
  let removedExchanges = 0;
  const currentMessages = () => [...(system ? [system] : []), ...currentTurns.flat()];
  let estimate = initial;

  // A single user request can span many assistant/tool rounds. Remove only
  // complete oldest batches, retaining the current user message, retained
  // arguments/reasoning, and the newest completed batch when it fits.
  // Message costs are computed once above and subtracted for each removed batch.
  while (estimate > desired) {
    const activeIndex = currentTurns.length - 1;
    const active = currentTurns[activeIndex]!;
    const batches = completeToolBatches(active);
    if (batches.length <= 1) break;
    const [start, end] = batches[0]!;
    const activeCosts = currentTurnCosts[activeIndex]!;
    let batchCost = 0;
    for (let index = start; index <= end; index++) batchCost += activeCosts[index]!;
    currentTurns[activeIndex] = active.filter((_, index) => index < start || index > end);
    currentTurnCosts[activeIndex] = activeCosts.filter((_, index) => index < start || index > end);
    turnCosts[activeIndex] -= batchCost;
    estimate -= batchCost;
    removedExchanges += 1;
  }

  // Preserve the newest complete exchange whenever it fits. Older completed
  // exchanges are removed whole, so tool IDs never dangle from their calls.
  // Ten recent user turns are normally retained for conversational cohesion.
  // An actual provider overflow uses force and may yield this preference.
  const minimumTurns = 1;
  while (estimate > desired && currentTurns.length > minimumTurns) {
    estimate -= turnCosts.shift()!;
    currentTurns.shift();
    currentTurnCosts.shift();
    removedExchanges += 1;
  }

  // Last resort within the still-live request: retain assistant calls and
  // reasoning verbatim and shorten oversized tool-result bodies — but only
  // after every older exchange and batch has already been removed, so a
  // freshly returned result is never truncated while older history remains.
  if (estimate > desired) {
    const activeIndex = currentTurns.length - 1;
    const active = currentTurns[activeIndex]!;
    const activeCosts = currentTurnCosts[activeIndex]!;
    currentTurns[activeIndex] = active.map((message, index) => {
      if (message.role !== "tool" || typeof message.content !== "string" || Buffer.byteLength(message.content, "utf8") <= ACTIVE_TOOL_RESULT_BYTES) return message;
      reducedToolResults += 1;
      const shortened = { ...message, content: boundToolResult(message.content, ACTIVE_TOOL_RESULT_BYTES) } as GlmMessage;
      const oldCost = activeCosts[index]!;
      const newCost = estimateOneMessage(shortened, metrics);
      activeCosts[index] = newCost;
      const difference = newCost - oldCost;
      turnCosts[activeIndex] += difference;
      estimate += difference;
      return shortened;
    });
  }

  const compacted = currentMessages();
  return {
    messages: compacted,
    changed: removedExchanges > 0 || reducedToolResults > 0,
    removedExchanges,
    reducedToolResults,
    estimatedTokens: estimate,
  };
}

export function appendCompactionNote(message: GlmMessage, removed: number, reduced: number): GlmMessage {
  const pattern = /\n?\n?\[Context compaction: omitted (\d+) completed exchanges?; shortened (\d+) tool results?\. The original user request remains above\.\]/;
  const previous = typeof message.content === "string" ? message.content.match(pattern) : undefined;
  const totalRemoved = removed + Number(previous?.[1] ?? 0);
  const totalReduced = reduced + Number(previous?.[2] ?? 0);
  const note = `[Context compaction: omitted ${totalRemoved} completed exchange${totalRemoved === 1 ? "" : "s"}; shortened ${totalReduced} tool result${totalReduced === 1 ? "" : "s"}. The original user request remains above.]`;
  if (typeof message.content === "string") return { ...message, content: `${message.content.replace(pattern, "")}\n\n${note}` } as GlmMessage;
  if (Array.isArray(message.content)) return { ...message, content: [...message.content.filter(part => !(part.type === "text" && typeof part.text === "string" && pattern.test(part.text))), { type: "text", text: note }] } as GlmMessage;
  return { ...message, content: note } as GlmMessage;
}

function groupTurns(messages: readonly GlmMessage[]): GlmMessage[][] {
  const turns: GlmMessage[][] = [];
  let current: GlmMessage[] = [];
  for (const message of messages) {
    if (message.role === "user" && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(message);
  }
  if (current.length > 0) turns.push(current);
  return turns;
}

function completeToolBatches(messages: readonly GlmMessage[]): Array<[number, number]> {
  const batches: Array<[number, number]> = [];
  for (let start = 0; start < messages.length; start++) {
    const assistant = messages[start];
    if (assistant?.role !== "assistant" || !assistant.tool_calls?.length) continue;
    let end = start;
    while (end + 1 < messages.length && messages[end + 1]?.role === "tool") end++;
    const ids = new Set(assistant.tool_calls.map(call => call.id));
    const returned = new Set(messages.slice(start + 1, end + 1).filter(message => message.role === "tool").map(message => message.tool_call_id));
    if ([...ids].every(id => returned.has(id))) batches.push([start, end]);
  }
  return batches;
}

function estimateMessageTokens(message: GlmMessage): number {
  let bytes = 0;
  let chars = 0;
  let images = 0;
  const add = (text: string) => { bytes += Buffer.byteLength(text, "utf8"); chars += text.length; };
  if (typeof message.content === "string") add(message.content);
  else if (Array.isArray(message.content)) {
    for (const part of message.content) {
      if (part.type === "image_url") images += 1;
      else if ("text" in part && typeof part.text === "string") add(part.text);
    }
  }
  if (message.role === "assistant") {
    if (message.reasoning_content) add(message.reasoning_content);
    for (const call of message.tool_calls ?? []) {
      if ("function" in call) { add(call.function.name); add(call.function.arguments); }
    }
  }
  // This uses both UTF-16 characters and UTF-8 bytes to avoid treating CJK as
  // cheap English text. It remains a heuristic, not a tokenizer guarantee.
  return MESSAGE_OVERHEAD_TOKENS + images * IMAGE_PART_TOKENS + Math.max(Math.ceil(chars / 4), Math.ceil(bytes / 2));
}

function estimateOneMessage(message: GlmMessage, metrics?: CompactionMetrics): number {
  if (metrics) metrics.messageEstimateCount += 1;
  return estimateMessageTokens(message);
}
