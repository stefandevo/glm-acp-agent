import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import type { GlmMessage, GlmStreamChunk, StreamChatOptions } from "../llm/glm-client.js";
import { checkModelTransition } from "../protocol/model-transition.js";

for (const entry of ["legacy", "config"]) {
  test(`${entry} model switch rejects retained images without changing state`, async () => {
    const cwd = mkdtempSync(join(tmpdir(), "glm-model-transition-"));
    const store = new SessionStore(join(cwd, "sessions"));
    const requests: Array<{ messages: GlmMessage[]; model?: string }> = [];
    const updates: unknown[] = [];
    const agent = new GlmAcpAgent({ sessionUpdate: async (update: unknown) => { updates.push(update); } } as never,
      { sessionStore: store, visionClient: null, glm: {
        async *streamChat(messages: GlmMessage[], _signal?: AbortSignal, options?: StreamChatOptions): AsyncGenerator<GlmStreamChunk> {
          requests.push({ messages: structuredClone(messages), model: options?.model });
          yield { text: "done" }; yield { done: true, stopReason: "stop" };
        },
      } });
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    try {
      await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3-flash" });
      await agent.prompt({ sessionId, prompt: [{ type: "image", data: "AAAA", mimeType: "image/png" }] });
      await agent.closeSession({ sessionId });
      await agent.resumeSession({ sessionId, cwd, mcpServers: [] });
      const before = store.load(sessionId);
      const emitted = updates.length;
      await assert.rejects(entry === "legacy"
        ? agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3" })
        : agent.setSessionConfigOption({ sessionId, configId: "model", value: "glm-5.3" }), /image/i);
      assert.deepEqual(store.load(sessionId), before);
      assert.equal(updates.length, emitted, "rejected changes must not advertise new config");
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
      assert.equal(requests.at(-1)?.model, "glm-5.3-flash");
      assert.ok(requests.at(-1)?.messages.some(message => message.role === "user" && Array.isArray(message.content) && message.content.some(part => part.type === "image_url")));
      await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5v-turbo" });
      assert.equal(store.load(sessionId)?.model, "glm-5v-turbo");
    } finally {
      await agent.closeSession({ sessionId });
      rmSync(cwd, { recursive: true, force: true });
    }
  });
}

test("text-only history can change models while an existing call retains its captured model", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-model-boundary-"));
  let started!: () => void;
  const ready = new Promise<void>(resolve => { started = resolve; });
  let release!: () => void;
  const waiting = new Promise<void>(resolve => { release = resolve; });
  const models: Array<string | undefined> = [];
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, { sessionStore: null, visionClient: null, glm: {
    async *streamChat(_messages, _signal, options): AsyncGenerator<GlmStreamChunk> {
      models.push(options?.model); started(); await waiting;
      yield { text: "done" }; yield { done: true, stopReason: "stop" };
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3" });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });
    await ready;
    await agent.unstable_setSessionModel({ sessionId, modelId: "glm-4.7" });
    release(); await prompt;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
    assert.deepEqual(models, ["glm-5.3", "glm-4.7"]);
  } finally { release(); await agent.closeSession({ sessionId }); rmSync(cwd, { recursive: true, force: true }); }
});

test("restored incompatible image history fails before making a provider request", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-restored-model-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let calls = 0;
  const messages: GlmMessage[] = [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://example.invalid/image.png" } }] }];
  await store.save({ sessionId: "incompatible", cwd, title: null, updatedAt: new Date().toISOString(), model: "glm-5.3", mode: "default", messages });
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, { sessionStore: store, visionClient: null,
    glm: { async *streamChat() { calls++; yield { done: true, stopReason: "stop" }; } },
  });
  try {
    await agent.resumeSession({ sessionId: "incompatible", cwd, mcpServers: [] });
    await assert.rejects(agent.prompt({ sessionId: "incompatible", prompt: [{ type: "text", text: "continue" }] }), /image/i);
    assert.equal(calls, 0);
    assert.equal(checkModelTransition(messages, "glm-5.3").ok, false);
    assert.equal(checkModelTransition([{ role: "user", content: "text-only retained history" }], "glm-5.3").ok, true);
  } finally {
    await agent.closeSession({ sessionId: "incompatible" });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("active native-capability prompt must settle before switching to a text-only model", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-active-native-"));
  let ready!: () => void;
  const started = new Promise<void>(resolve => { ready = resolve; });
  let release!: () => void;
  const wait = new Promise<void>(resolve => { release = resolve; });
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, { sessionStore: null, visionClient: null,
    glm: { async *streamChat() { ready(); await wait; yield { text: "done" }; yield { done: true, stopReason: "stop" }; } },
  });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3-flash" });
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });
  try {
    await started;
    await assert.rejects(agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3" }), /active prompt/i);
    release(); await prompt;
    await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3" });
  } finally {
    release(); await prompt; await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});
