import test from "node:test";
import assert from "node:assert/strict";
import { appendCompactionNote, compactToBudget } from "../protocol/context-budget.js";
import type { GlmMessage } from "../llm/glm-client.js";

const budget = { contextWindow: 20_000, maxOutputTokens: 2_000, toolSchemaTokens: 200, safetyTokens: 4_096 };

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
    messages.push({ role: "assistant", content: null, reasoning_content: `${id} reasoning`, tool_calls: [{ id, type: "function", function: { name: "read_file", arguments: `{"id":"${id}"}` } }] });
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
