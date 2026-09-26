import test from "node:test";
import assert from "node:assert/strict";
import { appendCompactionNote, assertValidHistory, compactToBudget, estimateMessagesTokens, estimateSerializedTokens, ToolSchemaTokenCache } from "../protocol/context-budget.js";
import { boundToolResult } from "../tools/tool-output.js";
import type { GlmMessage } from "../llm/glm-client.js";
import type { ToolDefinition } from "../tools/definitions.js";

const budget = { contextWindow: 20_000, maxOutputTokens: 2_000, toolSchemaTokens: 200, safetyTokens: 4_096 };

// Original full-rescan implementation retained as an equivalence oracle.
function referenceCompactToBudget(messages: readonly GlmMessage[], currentBudget: typeof budget, force: boolean) {
  const available = currentBudget.contextWindow - currentBudget.maxOutputTokens - currentBudget.toolSchemaTokens - currentBudget.safetyTokens;
  const initial = estimateMessagesTokens(messages);
  if (messages.length <= 1 || (!force && initial <= Math.floor(available * 0.9))) {
    return { messages: [...messages], changed: false, removedExchanges: 0, reducedToolResults: 0, estimatedTokens: initial };
  }
  const target = Math.floor(available * 0.8);
  const desired = force ? Math.min(target, Math.floor(initial / 2)) : target;
  const system = messages[0]?.role === "system" ? messages[0] : undefined;
  const turns: GlmMessage[][] = [];
  let current: GlmMessage[] = [];
  for (const message of system ? messages.slice(1) : messages) {
    if (message.role === "user" && current.length > 0) {
      turns.push(current);
      current = [];
    }
    current.push(message);
  }
  if (current.length > 0) turns.push(current);
  if (turns.length === 0) {
    return { messages: [...messages], changed: false, removedExchanges: 0, reducedToolResults: 0, estimatedTokens: initial };
  }

  const currentTurns = turns.map(turn => [...turn]);
  let reducedToolResults = 0;
  let removedExchanges = 0;
  const currentMessages = () => [...(system ? [system] : []), ...currentTurns.flat()];
  let estimate = estimateMessagesTokens(currentMessages());
  while (estimate > desired) {
    const activeIndex = currentTurns.length - 1;
    const active = currentTurns[activeIndex]!;
    const batches: Array<[number, number]> = [];
    for (let start = 0; start < active.length; start++) {
      const assistant = active[start];
      if (assistant?.role !== "assistant" || !assistant.tool_calls?.length) continue;
      let end = start;
      while (end + 1 < active.length && active[end + 1]?.role === "tool") end++;
      const ids = new Set(assistant.tool_calls.map(call => call.id));
      const returned = new Set(active.slice(start + 1, end + 1).filter(message => message.role === "tool").map(message => message.tool_call_id));
      if ([...ids].every(id => returned.has(id))) batches.push([start, end]);
    }
    if (batches.length <= 1) break;
    const [start, end] = batches[0]!;
    currentTurns[activeIndex] = active.filter((_, index) => index < start || index > end);
    removedExchanges += 1;
    estimate = estimateMessagesTokens(currentMessages());
  }
  while (estimate > desired && currentTurns.length > 1) {
    currentTurns.shift();
    removedExchanges += 1;
    estimate = estimateMessagesTokens(currentMessages());
  }
  if (estimate > desired) {
    const activeIndex = currentTurns.length - 1;
    const active = currentTurns[activeIndex]!;
    currentTurns[activeIndex] = active.map(message => {
      if (message.role !== "tool" || typeof message.content !== "string" || Buffer.byteLength(message.content, "utf8") <= 4096) return message;
      reducedToolResults += 1;
      return { ...message, content: boundToolResult(message.content, 4096) } as GlmMessage;
    });
    estimate = estimateMessagesTokens(currentMessages());
  }
  const compacted = currentMessages();
  return { messages: compacted, changed: removedExchanges > 0 || reducedToolResults > 0, removedExchanges, reducedToolResults, estimatedTokens: estimate };
}

test("tool schema estimates are reused and invalidated when the catalog changes", () => {
  const definitions: ToolDefinition[] = [{
    type: "function",
    function: { name: "first", description: "initial", parameters: { type: "object" } },
  }];
  let serializations = 0;
  const cache = new ToolSchemaTokenCache(value => {
    serializations += 1;
    return estimateSerializedTokens(value);
  });

  const initial = cache.get(definitions);
  assert.equal(cache.get(definitions), initial);
  assert.equal(serializations, 1, "unchanged catalog is serialized once");

  definitions[0]!.function.description = "changed schema description";
  cache.invalidate(definitions);
  const refreshed = cache.get(definitions);
  assert.equal(refreshed, estimateSerializedTokens(definitions));
  assert.notEqual(refreshed, initial);
  assert.equal(serializations, 2, "catalog invalidation recomputes exactly once");

  const replacement = [...definitions];
  cache.get(replacement);
  assert.equal(serializations, 3, "a replacement catalog has a distinct cache identity");
});

test("incremental compaction accounting matches full-rescan decisions with one estimate per message", () => {
  const messages: GlmMessage[] = [{ role: "system", content: "rules" }];
  for (let turn = 0; turn < 18; turn++) {
    messages.push({ role: "user", content: `request ${turn} ${"中".repeat(160)}` });
    messages.push({
      role: "assistant",
      content: null,
      reasoning_content: `reasoning ${turn}`,
      tool_calls: [{ id: `old-${turn}`, type: "function", function: { name: "read_file", arguments: JSON.stringify({ turn }) } }],
    });
    messages.push({ role: "tool", tool_call_id: `old-${turn}`, content: `old result ${turn} ` + "x".repeat(320) });
  }
  messages.push({ role: "user", content: "current request" });
  for (let batch = 0; batch < 7; batch++) {
    const id = `current-${batch}`;
    messages.push({
      role: "assistant",
      content: null,
      reasoning_content: `current reasoning ${batch}`,
      tool_calls: [{ id, type: "function", function: { name: "read_file", arguments: JSON.stringify({ batch }) } }],
    });
    messages.push({ role: "tool", tool_call_id: id, content: `result ${batch} ` + "y".repeat(batch === 6 ? 7_000 : 300) });
  }

  let sawRemovedHistory = false;
  let sawShortenedToolResult = false;
  for (const force of [false, true]) {
    for (const contextWindow of [4_700, 5_200, 8_000, 14_000]) {
      const currentBudget = { contextWindow, maxOutputTokens: 500, toolSchemaTokens: 0, safetyTokens: 4_096 };
      const expected = referenceCompactToBudget(messages, currentBudget, force);
      const metrics = { messageEstimateCount: 0 };
      const actual = compactToBudget(messages, currentBudget, force, metrics);
      assert.deepEqual(actual, expected, `force=${force}, contextWindow=${contextWindow}`);
      assert.equal(actual.estimatedTokens, estimateMessagesTokens(actual.messages));
      assertValidHistory(actual.messages);
      assert.equal(metrics.messageEstimateCount, messages.length + actual.reducedToolResults, "removed history and batches are not re-estimated");
      sawRemovedHistory ||= actual.removedExchanges > 0;
      sawShortenedToolResult ||= actual.reducedToolResults > 0;
    }
  }
  assert.ok(sawRemovedHistory, "fixtures exercise complete-exchange removal");
  assert.ok(sawShortenedToolResult, "fixtures exercise the active tool-result shortening fallback");
});

test("compaction removes whole completed exchanges and keeps reasoning intact", () => {
  const messages: GlmMessage[] = [{ role: "system", content: "rules" }];
  for (let i = 0; i < 4; i++) {
    messages.push({ role: "user", content: `request ${i} ${"中".repeat(5000)}` });
    messages.push({ role: "assistant", content: "calling", reasoning_content: `reasoning ${i}`, tool_calls: [{ id: String(i), type: "function", function: { name: "read_file", arguments: "{}" } }] });
    messages.push({ role: "tool", tool_call_id: String(i), content: "result".repeat(5000) });
  }
  const compacted = compactToBudget(messages, budget, true);
  assert.ok(compacted.changed);
  assert.ok(compacted.removedExchanges > 0);
  assert.ok(compacted.messages.some(message => message.role === "assistant" && message.reasoning_content === "reasoning 3"));
  assert.ok(!compacted.messages.some(message => message.role === "tool" && message.tool_call_id === "0"));
});

test("compaction note preserves string user content and identifies omissions", () => {
  const source: GlmMessage = { role: "user", content: "original user request" };
  const noted = appendCompactionNote(source, 2, 1);
  assert.match(String(noted.content), /original user request/);
  assert.match(String(noted.content), /omitted 2 completed exchanges/);
});

test("compaction updates one note instead of appending another", () => {
  const once = appendCompactionNote({ role: "user", content: "request" }, 1, 0);
  const twice = appendCompactionNote(once, 2, 1);
  assert.equal(String(twice.content).match(/Context compaction/g)?.length, 1);
  assert.match(String(twice.content), /omitted 3 completed exchanges/);
});

test("compaction evicts old complete tool batches within one live user turn", () => {
  const messages: GlmMessage[] = [{ role: "system", content: "rules" }, { role: "user", content: "current" }];
  for (const id of ["old", "new"]) {
    messages.push({ role: "assistant", content: null, tool_calls: [{ id, type: "function", function: { name: "read_file", arguments: JSON.stringify({ id }) } }] });
    messages.push({ role: "tool", tool_call_id: id, content: "x".repeat(30_000) });
  }
  const result = compactToBudget(messages, { contextWindow: 7_000, maxOutputTokens: 1_000, toolSchemaTokens: 0, safetyTokens: 4_096 }, true);
  assert.ok(result.removedExchanges >= 1);
  assert.ok(!result.messages.some(message => message.role === "tool" && message.tool_call_id === "old"));
  assert.ok(result.messages.some(message => message.role === "assistant" && message.tool_calls?.some(call => call.id === "new")));
});

test("compaction removes older history before shortening fresh results", () => {
  const messages: GlmMessage[] = [{ role: "system", content: "rules" }];
  messages.push({ role: "user", content: "old request" });
  messages.push({ role: "assistant", content: null, tool_calls: [{ id: "old", type: "function", function: { name: "read_file", arguments: "{}" } }] });
  messages.push({ role: "tool", tool_call_id: "old", content: "x".repeat(40_000) });
  messages.push({ role: "user", content: "current request" });
  messages.push({ role: "assistant", content: null, tool_calls: [{ id: "new", type: "function", function: { name: "read_file", arguments: "{}" } }] });
  messages.push({ role: "tool", tool_call_id: "new", content: "y".repeat(10_000) });
  const result = compactToBudget(messages, budget, true);
  assert.ok(result.removedExchanges >= 1);
  assert.equal(result.reducedToolResults, 0, "a fresh result must not be shortened while older history can go");
  assert.ok(!result.messages.some(message => message.role === "tool" && message.tool_call_id === "old"));
  const fresh = result.messages.find(message => message.role === "tool" && message.tool_call_id === "new");
  assert.equal(fresh?.content, "y".repeat(10_000), "the just-returned result stays verbatim");
});
