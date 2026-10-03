import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import OpenAI from "openai";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";

test("a connection error cancels the turn and the next prompt continues the session", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) throw new OpenAI.APIConnectionError({ message: "Connection error." });
        yield { text: "still here" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send the message again when you are back online/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("add the missing row")),
      false
    );

    const continued = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(continued.stopReason, "end_turn");
    assert.equal(notices.at(-1), "still here");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("partial text from a dropped stream is kept and the session accepts another prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-partial-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let calls = 0;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      if (update.update.sessionUpdate === "agent_message_chunk" && update.update.content?.text) {
        notices.push(update.update.content.text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: "partial answer" };
          throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
        }
        yield { text: "continued" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "compare them" }],
    });
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send a message to continue/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "partial answer"),
      true
    );

    const continued = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "continue" }],
    });
    assert.equal(continued.stopReason, "end_turn");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

// Larger than the default model's compaction threshold, so the next prompt
// rewrites the live user message before the provider call.
const PRIOR_TURN = "prior-turn ".repeat(200_000);

test("a connection error after compaction drops the unsent prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let sawCompaction = false;
  let seeded = false;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      const text = update.update.content?.text;
      if (update.update.sessionUpdate === "agent_message_chunk" && text && !text.startsWith("prior-turn")) {
        notices.push(text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        if (!seeded) {
          seeded = true;
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        throw new OpenAI.APIConnectionError({ message: "Connection error." });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send the message again when you are back online/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")),
      false
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true,
      "the prior user turn must survive a no-output drop after compaction"
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true,
      "the prior assistant turn must survive a no-output drop after compaction"
    );
    await agent.closeSession({ sessionId });
    const afterClose = store.load(sessionId);
    assert.equal(
      afterClose?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true,
      "closing the session must not replace the checkpoint with the shortened transcript"
    );
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a queued prompt during a no-output drop after compaction keeps the prior exchange", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-queued-"));
  const store = new SessionStore(join(cwd, "sessions"));
  let calls = 0;
  let sawCompaction = false;
  let release = (): void => {};
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let markEntered = (): void => {};
  const entered = new Promise<void>((resolve) => {
    markEntered = resolve;
  });
  const agent = new GlmAcpAgent({
    sessionUpdate: async () => {},
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        if (calls === 2) {
          sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
          markEntered();
          await gate;
          throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
        }
        throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failing = agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "add the missing row" }],
    });
    await entered;
    const queued = agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "queued follow up" }],
    });
    release();
    const failed = await failing;
    const queuedResult = await queued;
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.equal(queuedResult.stopReason, "cancelled");
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("remember the bulk")),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true
    );
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("add the missing row")), false);
    assert.equal(saved?.messages.some((message) => JSON.stringify(message).includes("queued follow up")), false);
    await agent.closeSession({ sessionId });
    const afterClose = store.load(sessionId);
    assert.equal(
      afterClose?.messages.some((message) => message.role === "assistant" && String(message.content).startsWith("prior-turn")),
      true,
      "closing the session must not replace the checkpoint with the shortened transcript"
    );
  } finally {
    release();
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("a dropped stream after compaction keeps partial work", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-compact-partial-"));
  const store = new SessionStore(join(cwd, "sessions"));
  const notices: string[] = [];
  let sawCompaction = false;
  let seeded = false;
  const agent = new GlmAcpAgent({
    sessionUpdate: async (update: { update: { sessionUpdate: string; content?: { text?: string } } }) => {
      const text = update.update.content?.text;
      if (update.update.sessionUpdate === "agent_message_chunk" && text && !text.startsWith("prior-turn")) {
        notices.push(text);
      }
    },
  } as never, {
    sessionStore: store,
    visionClient: null,
    glm: {
      async *streamChat(messages: { role?: string; content?: unknown }[]): AsyncGenerator<GlmStreamChunk> {
        if (!seeded) {
          seeded = true;
          yield { text: PRIOR_TURN };
          yield { done: true, stopReason: "stop" };
          return;
        }
        sawCompaction = messages.some((message) => message.role === "user" && String(message.content).includes("Context compaction"));
        yield { text: "partial answer" };
        throw Object.assign(new Error("socket hang up"), { code: "ECONNRESET" });
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    const prior = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "remember the bulk" }],
    });
    assert.equal(prior.stopReason, "end_turn");
    const failed = await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "compare them" }],
    });
    assert.equal(sawCompaction, true);
    assert.equal(failed.stopReason, "cancelled");
    assert.match(notices.join("\n"), /Send a message to continue/);
    const saved = store.load(sessionId);
    assert.equal(
      saved?.messages.some((message) => message.role === "assistant" && message.content === "partial answer"),
      true
    );
    assert.equal(
      saved?.messages.some((message) => message.role === "user" && JSON.stringify(message).includes("compare them")),
      true
    );
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("an HTTP provider error still fails the prompt", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-transport-http-"));
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: null,
    visionClient: null,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        const fail = { status: 502 };
        if (fail.status >= 500) throw Object.assign(new Error("upstream"), fail);
        yield { text: "unreachable" };
      },
    },
  });
  const sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
  try {
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] }),
      /upstream/
    );
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});
