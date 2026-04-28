import test from "node:test";
import assert from "node:assert/strict";
import { GlmAcpAgent } from "../protocol/agent.js";

function createConnectionStub() {
  return {
    updates: [] as Array<Record<string, unknown>>,
    async sessionUpdate(update: Record<string, unknown>) {
      this.updates.push(update);
    },
  };
}

test("runPromptLoop maps content_filter stop reason to refusal", async () => {
  const connection = createConnectionStub();
  const agent = new GlmAcpAgent(connection as never);
  (agent as any)._glm = {
    async *streamChat() {
      yield { text: "I can't help with that." };
      yield { done: true, stopReason: "content_filter" };
    },
  };

  const result = await (agent as any).runPromptLoop(
    "s1",
    {
      cwd: "/tmp",
      messages: [],
      abortController: null,
      title: null,
      updatedAt: new Date().toISOString(),
    },
    new AbortController().signal
  );

  assert.equal(result.stopReason, "refusal");
});

test("runPromptLoop returns max_turn_requests after tool loop limit", async () => {
  const connection = createConnectionStub();
  const agent = new GlmAcpAgent(connection as never);
  (agent as any)._glm = {
    async *streamChat() {
      yield { toolCall: { id: "tc1", name: "unknown_tool", arguments: "{}" } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };

  const result = await (agent as any).runPromptLoop(
    "s2",
    {
      cwd: "/tmp",
      messages: [],
      abortController: null,
      title: null,
      updatedAt: new Date().toISOString(),
    },
    new AbortController().signal
  );

  assert.equal(result.stopReason, "max_turn_requests");
});
