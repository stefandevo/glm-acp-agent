import test from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent } from "../protocol/agent.js";
import { preprocessImageBlocks } from "../protocol/image-preprocessor.js";
import { SessionStore } from "../protocol/session-store.js";
import { SessionMcpTools } from "../tools/session-mcp-client.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";
import type { VisionMcpClient } from "../tools/vision-mcp-client.js";

function connection() {
  const updates: Array<Record<string, unknown>> = [];
  return {
    updates,
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) {
      updates.push(params);
    },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
}

function textGlm(onCall: (messages: unknown[]) => void = () => {}) {
  return {
    async *streamChat(messages: unknown[]): AsyncGenerator<GlmStreamChunk> {
      onCall(messages);
      yield { text: "done" };
      yield { done: true, stopReason: "stop" };
    },
  };
}

function imagePrompt() {
  return [{ type: "image" as const, data: "AAAA", mimeType: "image/png" }];
}

test("cancelled delayed vision preprocessing settles without starting the model", async () => {
  const conn = connection();
  let visionStarted!: () => void;
  const started = new Promise<void>((resolve) => { visionStarted = resolve; });
  let releaseVision!: () => void;
  const visionDone = new Promise<void>((resolve) => { releaseVision = resolve; });
  let sourcePath = "";
  const vision: VisionMcpClient = {
    async callTool(_name, args) {
      sourcePath = String(args["image_source"]);
      visionStarted();
      await visionDone;
      return { content: [{ type: "text", text: "late" }] };
    },
    async dispose() {},
  };
  let modelCalls = 0;
  const agent = new GlmAcpAgent(conn as never, {
    visionClient: vision,
    sessionStore: null,
    glm: textGlm(() => { modelCalls += 1; }),
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: imagePrompt(), messageId: "cancelled-1" });
  await started;
  assert.equal(existsSync(sourcePath), true);
  await agent.cancel({ sessionId });
  // The prompt lifecycle cannot settle while the vision request is still in
  // flight, so release it before awaiting the cancelled prompt.
  releaseVision();
  const result = await prompt;
  assert.equal(result.stopReason, "cancelled");
  assert.equal(result.userMessageId, "cancelled-1");
  assert.equal(modelCalls, 0);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(existsSync(sourcePath), false, "abort must remove materialized image data");
});

test("cancelled vision request defers materialized-image cleanup until it settles", async () => {
  const conn = connection();
  let visionStarted!: () => void;
  const started = new Promise<void>((resolve) => { visionStarted = resolve; });
  let releaseVision!: () => void;
  const visionDone = new Promise<void>((resolve) => { releaseVision = resolve; });
  let sourcePath = "";
  const vision: VisionMcpClient = {
    async callTool(_name, args) {
      sourcePath = String(args["image_source"]);
      visionStarted();
      await visionDone;
      return { content: [{ type: "text", text: "late" }] };
    },
    async dispose() {},
  };
  const agent = new GlmAcpAgent(conn as never, {
    visionClient: vision,
    sessionStore: null,
    glm: textGlm(),
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: imagePrompt(), messageId: "defer-cleanup-1" });
  await started;
  await agent.cancel({ sessionId });
  await new Promise((resolve) => setImmediate(resolve));
  // Cancellation has been observed, but the vision request is still running:
  // its materialized image must survive until the request settles.
  assert.equal(existsSync(sourcePath), true, "cleanup must not race the in-flight vision request");
  releaseVision();
  const result = await prompt;
  assert.equal(result.stopReason, "cancelled");
  assert.equal(
    existsSync(sourcePath),
    false,
    "materialized image is removed once the request settles and the lifecycle unwinds",
  );
});

test("preprocessing abort before vision startup observes a rejecting late operation and makes no vision call", async () => {
  const controller = new AbortController();
  const root = await mkdtemp(join(tmpdir(), "glm-acp-abort-test-"));
  let writeStarted!: () => void;
  const writeReady = new Promise<void>((resolve) => { writeStarted = resolve; });
  let releaseWrite!: () => void;
  const writeDone = new Promise<void>((resolve) => { releaseWrite = resolve; });
  let visionCalls = 0;
  const vision: VisionMcpClient = {
    async callTool() {
      visionCalls += 1;
      throw new Error("late vision failure");
    },
    async dispose() {},
  };
  const prepared = preprocessImageBlocks(imagePrompt(), vision, controller.signal, {
    async mkdtemp(prefix: string) {
      void prefix;
      return mkdtemp(join(root, "image-"));
    },
    async writeFile() {
      writeStarted();
      await writeDone;
      controller.abort();
    },
    rm,
  });
  await writeReady;
  releaseWrite();
  await assert.rejects(prepared, /cancelled|aborted/i);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(visionCalls, 0, "abort during materialization must prevent vision startup");
  await rm(root, { recursive: true, force: true });
});

test("already-aborted preprocessing does not start a rejecting vision stub", async () => {
  const controller = new AbortController();
  controller.abort();
  let visionCalls = 0;
  const vision: VisionMcpClient = {
    async callTool() {
      visionCalls += 1;
      throw new Error("late vision failure");
    },
    async dispose() {},
  };
  await assert.rejects(
    preprocessImageBlocks([{ type: "image", data: "", mimeType: "image/png", uri: "https://example.test/image.png" }], vision, controller.signal),
    /cancelled|aborted/i,
  );
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(visionCalls, 0, "an already-aborted prompt must not start vision");
});

test("abort between URI setup and queued vision startup prevents the call", async () => {
  const controller = new AbortController();
  let visionCalls = 0;
  const vision: VisionMcpClient = {
    async callTool() {
      visionCalls += 1;
      throw new Error("late vision failure");
    },
    async dispose() {},
  };
  const prepared = preprocessImageBlocks(
    [{ type: "image", data: "", mimeType: "image/png", uri: "https://example.test/image.png" }],
    vision,
    controller.signal,
  );
  controller.abort();
  await assert.rejects(prepared, /cancelled|aborted/i);
  assert.equal(visionCalls, 0);
});

test("cancellation after vision starts still observes its late rejection", async () => {
  const controller = new AbortController();
  let rejectVision!: (error: Error) => void;
  const visionOperation = new Promise<unknown>((_resolve, reject) => { rejectVision = reject; });
  let visionCalls = 0;
  const vision: VisionMcpClient = {
    async callTool() {
      visionCalls += 1;
      controller.abort();
      return visionOperation;
    },
    async dispose() {},
  };
  const prepared = preprocessImageBlocks([{ type: "image", data: "", mimeType: "image/png", uri: "https://example.test/image.png" }], vision, controller.signal);
  // Let the queued vision startup run so the abort races an in-flight operation.
  await new Promise((resolve) => setImmediate(resolve));
  // The rejection is held until the operation settles: settle it, then observe
  // both the cancellation error and that the late rejection was consumed.
  rejectVision(new Error("late vision failure"));
  await assert.rejects(prepared, /cancelled|aborted/i);
  assert.equal(visionCalls, 1);
  await new Promise((resolve) => setImmediate(resolve));
});

test("preprocessing write failures remove the directory created before the write", async () => {
  const created: string[] = [];
  const root = await mkdtemp(join(tmpdir(), "glm-acp-prep-test-"));
  const result = await assert.rejects(
    preprocessImageBlocks(imagePrompt(), { callTool: async () => ({}), dispose: async () => {} }, undefined, {
      async mkdtemp(prefix: string) {
        void prefix;
        const dir = await mkdtemp(join(root, "image-"));
        created.push(dir);
        return dir;
      },
      async writeFile() {
        throw new Error("synthetic write failure");
      },
      rm,
    }),
  );
  assert.equal(result, undefined);
  assert.equal(created.length, 1);
  assert.equal(existsSync(created[0]!), false);
  await rm(root, { recursive: true, force: true });
});

test("close waits for preprocessing cleanup before persisting and removing the session", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-store-"));
  const sessionStore = new SessionStore(storeRoot);
  let started!: () => void;
  const prepStarted = new Promise<void>((resolve) => { started = resolve; });
  let releaseVision!: () => void;
  const visionDone = new Promise<void>((resolve) => { releaseVision = resolve; });
  let sourcePath = "";
  const vision: VisionMcpClient = {
    async callTool(_name, args) {
      sourcePath = String(args["image_source"]);
      started();
      await visionDone;
      return { content: [{ type: "text", text: "late" }] };
    },
    async dispose() {},
  };
  let saveSawCleanedImage = false;
  const save = sessionStore.save.bind(sessionStore);
  sessionStore.save = (session) => {
    saveSawCleanedImage = !existsSync(sourcePath);
    save(session);
  };
  const agent = new GlmAcpAgent(conn as never, {
    visionClient: vision,
    sessionStore,
    glm: textGlm(),
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: imagePrompt() });
  await prepStarted;
  const closing = agent.closeSession({ sessionId });
  // The lifecycle — including the in-flight vision request — must settle
  // before close can persist, so release the request before awaiting.
  releaseVision();
  await prompt;
  await closing;
  assert.equal(saveSawCleanedImage, true, "close must persist only after lifecycle cleanup");
  assert.equal(existsSync(sourcePath), false);
  await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "late" }] }), /Session not found/);
  await rm(storeRoot, { recursive: true, force: true });
});

test("a follow-up prompt starts after the cancelled image turn has fully unwound", async () => {
  const conn = connection();
  let releaseVision!: () => void;
  const visionDone = new Promise<void>((resolve) => { releaseVision = resolve; });
  let visionCall = 0;
  let modelCalls = 0;
  const vision: VisionMcpClient = {
    async callTool() {
      visionCall += 1;
      if (visionCall === 1) {
        await visionDone;
        return { content: [{ type: "text", text: "cancelled" }] };
      }
      return { content: [{ type: "text", text: "follow-up" }] };
    },
    async dispose() {},
  };
  const agent = new GlmAcpAgent(conn as never, {
    visionClient: vision,
    sessionStore: null,
    glm: textGlm(() => { modelCalls += 1; }),
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const cancelled = agent.prompt({ sessionId, prompt: imagePrompt(), messageId: "first" });
  await new Promise((resolve) => setImmediate(resolve));
  await agent.cancel({ sessionId });
  releaseVision();
  assert.equal((await cancelled).stopReason, "cancelled");
  const followUp = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "next" }], messageId: "second" });
  assert.equal(followUp.stopReason, "end_turn");
  assert.equal(followUp.userMessageId, "second");
  assert.equal(modelCalls, 1);
});

test("queued prompts form a chain so only the newest prompt can reach the model", async () => {
  const conn = connection();
  let firstStarted!: () => void;
  const firstStartedPromise = new Promise<void>((resolve) => { firstStarted = resolve; });
  let releaseFirst!: () => void;
  const firstDone = new Promise<void>((resolve) => { releaseFirst = resolve; });
  const calls: string[] = [];
  const glm = {
    async *streamChat(messages: Array<{ role: string; content?: unknown }>, signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
      const last = messages.at(-1)?.content;
      calls.push(typeof last === "string" ? last : "unknown");
      if (calls.length === 1) {
        firstStarted();
        await firstDone;
      }
      if (signal?.aborted) return;
      yield { text: "done" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "one" }] });
  await firstStartedPromise;
  const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "two" }] });
  const third = agent.prompt({ sessionId, prompt: [{ type: "text", text: "three" }] });
  releaseFirst();
  const results = await Promise.all([first, second, third]);
  assert.deepEqual(results.map((r) => r.stopReason), ["cancelled", "cancelled", "end_turn"]);
  assert.deepEqual(calls, ["one", "three"]);
});

for (const restoreKind of ["load", "resume"] as const) {
test(`live ${restoreKind} drains the reserved prompt chain before replacing its history`, async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-live-resume-"));
  const store = new SessionStore(storeRoot);
  let secondStarted!: () => void;
  const secondReady = new Promise<void>((resolve) => { secondStarted = resolve; });
  let releaseSecond!: () => void;
  const secondDone = new Promise<void>((resolve) => { releaseSecond = resolve; });
  let call = 0;
  let thirdHistory: unknown[] = [];
  const glm = {
    async *streamChat(messages: Array<{ content?: unknown }>): AsyncGenerator<GlmStreamChunk> {
      call += 1;
      if (call === 2) {
        secondStarted();
        await secondDone;
      }
      if (call === 3) thirdHistory = structuredClone(messages);
      yield { text: `reply-${call}` };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "second" }] });
    await secondReady;

    const restoreParams = { sessionId, cwd: tmpdir(), mcpServers: [] };
    const restoring = restoreKind === "load"
      ? agent.loadSession(restoreParams)
      : agent.resumeSession(restoreParams);
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "third" }] }),
      /transition in progress/i,
    );
    releaseSecond();
    assert.equal((await second).stopReason, "cancelled");
    await restoring;

    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "third" }] });
    const userTurns = thirdHistory
      .filter((message): message is { role?: string; content?: unknown } => typeof message === "object" && message !== null)
      .filter((message) => message.role === "user")
      .map((message) => String(message.content));
    assert.deepEqual(userTurns, ["first", "second", "third"]);
  } finally {
    releaseSecond();
    await rm(storeRoot, { recursive: true, force: true });
  }
});
}

test("a restore drain timeout leaves the original session usable after its prompt unwinds", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-resume-timeout-"));
  const store = new SessionStore(storeRoot);
  let started!: () => void;
  const ready = new Promise<void>((resolve) => { started = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  let postTimeoutHistory: Array<{ role?: string; content?: unknown }> = [];
  const glm = {
    async *streamChat(messages: Array<{ role?: string; content?: unknown }>): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 1) {
        yield { text: "partial before timeout" };
        started();
        await blocked;
      } else {
        postTimeoutHistory = structuredClone(messages);
      }
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, {
    glm,
    sessionStore: store,
    sessionDrainTimeoutMs: 0,
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "blocked" }] });
    await ready;
    await assert.rejects(
      agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
      /timed out/i,
    );
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "too-early" }] }),
      /transition in progress/i,
    );
    release();
    await first;
    assert.equal(
      (await agent.prompt({ sessionId, prompt: [{ type: "text", text: "after-timeout" }] })).stopReason,
      "end_turn",
    );
    assert.ok(postTimeoutHistory.some(
      (message) => message.role === "assistant" && message.content === "partial before timeout"
    ));
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a mode change remains responsive and survives a live restore drain", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-resume-mode-"));
  const store = new SessionStore(storeRoot);
  let secondStarted!: () => void;
  const secondReady = new Promise<void>((resolve) => { secondStarted = resolve; });
  let releaseSecond!: () => void;
  const secondDone = new Promise<void>((resolve) => { releaseSecond = resolve; });
  let calls = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 2) {
        secondStarted();
        await secondDone;
      }
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "second" }] });
    await secondReady;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await agent.setSessionMode({ sessionId, modeId: "bypass_permissions" });
    releaseSecond();
    await second;
    assert.equal((await resume).modes?.currentModeId, "bypass_permissions");
  } finally {
    releaseSecond();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close during restore cancels replacement and leaves no in-memory session", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-restore-"));
  const store = new SessionStore(storeRoot);
  let started!: () => void;
  const ready = new Promise<void>((resolve) => { started = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 2) {
        started();
        await blocked;
      }
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "second" }] });
    await ready;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    const closing = agent.closeSession({ sessionId });
    release();
    await second;
    await assert.rejects(resume, /cancelled/i);
    await closing;
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "after-close" }] }),
      /Session not found/i,
    );
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("failed replacement setup keeps the original session available", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-setup-failure-"));
  const store = new SessionStore(storeRoot);
  let calls = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      yield { text: `reply-${calls}` };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const previousFetch = globalThis.fetch;
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    globalThis.fetch = (async () => {
      throw new Error("synthetic MCP setup failure");
    }) as typeof fetch;
    await assert.rejects(
      agent.resumeSession({
        sessionId,
        cwd: tmpdir(),
        mcpServers: [{ type: "http", name: "broken", url: "https://mcp.example.test/broken", headers: [] }],
      }),
      /synthetic MCP setup failure/i,
    );
    globalThis.fetch = previousFetch;
    assert.equal(
      (await agent.prompt({ sessionId, prompt: [{ type: "text", text: "after-failure" }] })).stopReason,
      "end_turn",
    );
  } finally {
    globalThis.fetch = previousFetch;
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close is retained while an unloaded resume is setting up replacement resources", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-unloaded-close-"));
  const store = new SessionStore(storeRoot);
  const sessionId = "44444444-4444-4444-4444-444444444444";
  store.save({
    sessionId,
    cwd: "/tmp",
    messages: [{ role: "system", content: "system" }],
    title: null,
    updatedAt: "2026-01-01T00:00:00.000Z",
    model: "glm-5.3",
    mode: "default",
  });
  let setupStarted!: () => void;
  const setupReady = new Promise<void>((resolve) => { setupStarted = resolve; });
  let releaseSetup!: () => void;
  const setupDone = new Promise<void>((resolve) => { releaseSetup = resolve; });
  let disposed = 0;
  const replacement = new SessionMcpTools([]);
  replacement.dispose = async () => { disposed += 1; };
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    connectSessionMcpServers: async () => {
      setupStarted();
      await setupDone;
      return replacement;
    },
  });
  try {
    const resume = agent.resumeSession({ sessionId, cwd: "/tmp", mcpServers: [] });
    await setupReady;
    const close = agent.closeSession({ sessionId });
    releaseSetup();
    await assert.rejects(resume, /cancelled/i);
    await close;
    assert.equal(disposed, 1);
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "late" }] }),
      /Session not found/i,
    );
  } finally {
    releaseSetup();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close aborts stalled restore MCP setup and disposes a late empty-catalog replacement once", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-abort-restore-setup-"));
  const store = new SessionStore(storeRoot);
  const sessionId = "55555555-5555-5555-5555-555555555555";
  store.save({
    sessionId,
    cwd: "/tmp",
    messages: [{ role: "system", content: "system" }],
    title: null,
    updatedAt: "2026-01-01T00:00:00.000Z",
    model: "glm-5.3",
    mode: "default",
  });
  let setupStarted!: () => void;
  const setupReady = new Promise<void>((resolve) => { setupStarted = resolve; });
  let releaseLateResult!: () => void;
  const lateResult = new Promise<void>((resolve) => { releaseLateResult = resolve; });
  let abortObserved = false;
  let disposed = 0;
  const replacement = new SessionMcpTools([], [{
    async listTools() { return []; },
    async callTool() { return undefined; },
    async dispose() { disposed += 1; },
  }]);
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    connectSessionMcpServers: async (_servers, signal) => {
      setupStarted();
      await new Promise<void>((resolve) => signal?.addEventListener("abort", () => resolve(), { once: true }));
      abortObserved = signal?.aborted === true;
      await lateResult;
      return replacement;
    },
  });
  try {
    const resume = agent.resumeSession({ sessionId, cwd: "/tmp", mcpServers: [] });
    await setupReady;
    const close = agent.closeSession({ sessionId });
    await Promise.race([
      close,
      new Promise<never>((_resolve, reject) => {
        setTimeout(() => reject(new Error("close waited for stalled MCP setup")), 50);
      }),
    ]);
    assert.equal(abortObserved, true, "close must abort the restore connector");
    await assert.rejects(resume, /cancelled/i);
    assert.equal(disposed, 0, "a late replacement is not available to dispose yet");
    releaseLateResult();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(disposed, 1, "a late replacement is disposed exactly once");
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "late" }] }),
      /Session not found/i,
    );
  } finally {
    releaseLateResult();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a load replay failure disposes provisional resources and keeps the live original", async () => {
  const updates: Array<Record<string, unknown>> = [];
  let failReplay = false;
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) {
      if (failReplay) throw new Error("synthetic replay failure");
      updates.push(params);
    },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-replay-rollback-"));
  const store = new SessionStore(storeRoot);
  let replacementDisposed = 0;
  const replacement = new SessionMcpTools([]);
  replacement.dispose = async () => { replacementDisposed += 1; };
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    glm: textGlm(),
    connectSessionMcpServers: async () => replacement,
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "original" }] });
    failReplay = true;
    await assert.rejects(
      agent.loadSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
      /synthetic replay failure/i,
    );
    failReplay = false;
    assert.equal(replacementDisposed, 1);
    assert.equal(
      (await agent.prompt({ sessionId, prompt: [{ type: "text", text: "still-original" }] })).stopReason,
      "end_turn",
    );
  } finally {
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("restore snapshots assistant text already received before it aborts the draining stream", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-partial-assistant-"));
  const store = new SessionStore(storeRoot);
  let receivedPartial!: () => void;
  const partialReady = new Promise<void>((resolve) => { receivedPartial = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  let restoredHistory: Array<{ role?: string; content?: unknown }> = [];
  const glm = {
    async *streamChat(messages: Array<{ role?: string; content?: unknown }>): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 1) {
        yield { text: "partial assistant" };
        receivedPartial();
        await blocked;
        yield { text: "late assistant" };
      } else {
        restoredHistory = structuredClone(messages);
        yield { text: "continued" };
      }
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    await partialReady;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    release();
    assert.equal((await first).stopReason, "cancelled");
    await resume;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
    assert.ok(restoredHistory.some((message) => message.role === "assistant" && message.content === "partial assistant"));
    assert.ok(!conn.updates.some((update) => JSON.stringify(update).includes("late assistant")));
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("restore suppresses late tool updates while retaining an active tool result in history", async () => {
  const updates: Array<Record<string, unknown>> = [];
  let readStarted!: () => void;
  const readReady = new Promise<void>((resolve) => { readStarted = resolve; });
  let releaseRead!: () => void;
  const readDone = new Promise<void>((resolve) => { releaseRead = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) { updates.push(params); },
    async readTextFile() {
      readStarted();
      await readDone;
      return { content: "contents" };
    },
    async writeTextFile() {},
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-tool-"));
  const store = new SessionStore(storeRoot);
  let calls = 0;
  let restoredHistory: Array<{ role?: string; tool_call_id?: string }> = [];
  const glm = {
    async *streamChat(messages: Array<{ role?: string; tool_call_id?: string }>): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 1) {
        yield { toolCall: { id: "read-1", name: "read_file", arguments: JSON.stringify({ path: "a.txt" }) } };
        yield { done: true, stopReason: "tool_calls" };
      } else {
        restoredHistory = structuredClone(messages);
        yield { text: "continued" };
        yield { done: true, stopReason: "stop" };
      }
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  } as never);
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read" }] });
    await readReady;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    releaseRead();
    assert.equal((await first).stopReason, "cancelled");
    await resume;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
    assert.ok(restoredHistory.some((message) => message.role === "tool" && message.tool_call_id === "read-1"));
    const lateUpdates = updates.filter((update) => {
      const body = update.update as { sessionUpdate?: string; toolCallId?: string };
      return body.sessionUpdate === "tool_call_update" && body.toolCallId === "read-1";
    });
    assert.equal(lateUpdates.length, 0);
  } finally {
    releaseRead();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close persists assistant text received before the stream is aborted", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-stream-history-"));
  const store = new SessionStore(storeRoot);
  let partialReceived!: () => void;
  const partialReady = new Promise<void>((resolve) => { partialReceived = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { text: "partial before close" };
      partialReceived();
      await blocked;
      yield { text: "late after close" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
    await partialReady;
    const close = agent.closeSession({ sessionId });
    release();
    assert.equal((await prompt).stopReason, "cancelled");
    await close;
    const persisted = store.load(sessionId);
    assert.ok(persisted?.messages.some(
      (message) => message.role === "assistant" && message.content === "partial before close"
    ));
    assert.ok(!conn.updates.some((update) => JSON.stringify(update).includes("late after close")));
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close persists an in-flight tool result without late tool updates", async () => {
  const updates: Array<Record<string, unknown>> = [];
  let readStarted!: () => void;
  const readReady = new Promise<void>((resolve) => { readStarted = resolve; });
  let releaseRead!: () => void;
  const readDone = new Promise<void>((resolve) => { releaseRead = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) { updates.push(params); },
    async readTextFile() {
      readStarted();
      await readDone;
      return { content: "contents before close" };
    },
    async writeTextFile() {},
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-tool-history-"));
  const store = new SessionStore(storeRoot);
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { toolCall: { id: "read-close", name: "read_file", arguments: JSON.stringify({ path: "a.txt" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  } as never);
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read" }] });
    await readReady;
    const close = agent.closeSession({ sessionId });
    releaseRead();
    assert.equal((await prompt).stopReason, "cancelled");
    await close;
    const persisted = store.load(sessionId);
    assert.ok(persisted?.messages.some(
      (message) => message.role === "tool" && message.tool_call_id === "read-close" &&
        message.content === "contents before close"
    ));
    assert.equal(updates.filter((update) => {
      const body = update.update as { sessionUpdate?: string; toolCallId?: string };
      return body.sessionUpdate === "tool_call_update" && body.toolCallId === "read-close";
    }).length, 0);
  } finally {
    releaseRead();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("restore completes a streamed tool batch with cancelled tool results", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-tool-batch-"));
  const store = new SessionStore(storeRoot);
  let batchReceived!: () => void;
  const batchReady = new Promise<void>((resolve) => { batchReceived = resolve; });
  let releaseStream!: () => void;
  const streamDone = new Promise<void>((resolve) => { releaseStream = resolve; });
  let calls = 0;
  let restoredHistory: Array<{ role?: string; tool_call_id?: string; content?: unknown }> = [];
  const glm = {
    async *streamChat(messages: Array<{ role?: string; tool_call_id?: string; content?: unknown }>): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 1) {
        yield { toolCall: { id: "read-1", name: "read_file", arguments: JSON.stringify({ path: "a.txt" }) } };
        yield { toolCall: { id: "read-2", name: "read_file", arguments: JSON.stringify({ path: "b.txt" }) } };
        batchReceived();
        await streamDone;
      } else {
        restoredHistory = structuredClone(messages);
        yield { text: "continued" };
      }
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read both" }] });
    await batchReady;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    releaseStream();
    assert.equal((await first).stopReason, "cancelled");
    await resume;
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] });
    const toolResults = restoredHistory.filter((message) => message.role === "tool");
    assert.deepEqual(toolResults.map((message) => message.tool_call_id), ["read-1", "read-2"]);
    assert.ok(toolResults.every((message) => String(message.content).includes("cancelled before execution")));
  } finally {
    releaseStream();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close after swap disposes the current replacement rather than the stale original", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-swapped-"));
  const store = new SessionStore(storeRoot);
  let oldDisposeStarted!: () => void;
  const oldDisposeReady = new Promise<void>((resolve) => { oldDisposeStarted = resolve; });
  let releaseOldDispose!: () => void;
  const oldDisposeDone = new Promise<void>((resolve) => { releaseOldDispose = resolve; });
  let oldDisposals = 0;
  let replacementDisposals = 0;
  const originalTools = new SessionMcpTools([]);
  originalTools.dispose = async () => {
    oldDisposals += 1;
    oldDisposeStarted();
    await oldDisposeDone;
  };
  const replacementTools = new SessionMcpTools([]);
  replacementTools.dispose = async () => { replacementDisposals += 1; };
  let connections = 0;
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    connectSessionMcpServers: async () => {
      connections += 1;
      return connections === 1 ? originalTools : replacementTools;
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await oldDisposeReady;
    const close = agent.closeSession({ sessionId });
    releaseOldDispose();
    await assert.rejects(resume, /cancelled/i);
    await close;
    assert.equal(oldDisposals, 1);
    assert.equal(replacementDisposals, 1);
    await assert.rejects(
      agent.prompt({ sessionId, prompt: [{ type: "text", text: "late" }] }),
      /Session not found/i,
    );
  } finally {
    releaseOldDispose();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a configuration change during load replay is carried into the replacement", async () => {
  let replayArmed = false;
  let gateUsed = false;
  let blockReplay!: () => void;
  const replayBlocked = new Promise<void>((resolve) => { blockReplay = resolve; });
  let releaseReplay!: () => void;
  const replayGate = new Promise<void>((resolve) => { releaseReplay = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) {
      // Block exactly once on a replayed message (armed only once the load
      // starts), so a configuration change can land inside the replay window.
      if (
        replayArmed && !gateUsed &&
        String((params as { update?: { sessionUpdate?: string } }).update?.sessionUpdate) === "agent_message_chunk"
      ) {
        gateUsed = true;
        blockReplay();
        await replayGate;
      }
    },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-replay-config-"));
  const store = new SessionStore(storeRoot);
  const agent = new GlmAcpAgent(conn as never, { glm: textGlm(), sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    replayArmed = true;
    const loading = agent.loadSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await replayBlocked;
    await agent.setSessionMode({ sessionId, modeId: "bypass_permissions" });
    releaseReplay();
    const loaded = await loading;
    assert.equal(loaded.modes?.currentModeId, "bypass_permissions");
    assert.equal(store.load(sessionId)?.mode, "bypass_permissions");
  } finally {
    releaseReplay();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a live restore checkpoints its merged history at the swap", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-checkpoint-"));
  const store = new SessionStore(storeRoot);
  let receivedPartial!: () => void;
  const partialReady = new Promise<void>((resolve) => { receivedPartial = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { text: "partial assistant" };
      receivedPartial();
      await blocked;
      yield { text: "late assistant" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    await partialReady;
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    release();
    assert.equal((await first).stopReason, "cancelled");
    await resume;
    // No further prompt or close: the swap itself must have written the
    // retained partial turn to disk.
    assert.ok(store.load(sessionId)?.messages.some(
      (message) => message.role === "assistant" && message.content === "partial assistant"
    ));
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});
