import test from "node:test";
import assert from "node:assert/strict";
import { GlmAcpAgent } from "../protocol/agent.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";
import { PROTOCOL_VERSION } from "@agentclientprotocol/sdk";

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

interface ConnectionStub {
  updates: Array<Record<string, unknown>>;
  permissionRequests: Array<unknown>;
  reads: string[];
  writes: Array<{ path: string; content: string }>;
  terminalCommands: Array<{ command: string; args?: string[] }>;
  permissionResponse: () => { outcome: { outcome: "selected"; optionId: string } | { outcome: "cancelled" } };
  fileResponses: Map<string, string>;
  sessionUpdate: (params: Record<string, unknown>) => Promise<void>;
  requestPermission: (params: unknown) => Promise<unknown>;
  readTextFile: (params: { sessionId: string; path: string }) => Promise<{ content: string }>;
  writeTextFile: (params: { sessionId: string; path: string; content: string }) => Promise<void>;
  createTerminal: (params: { command: string; args?: string[] }) => Promise<unknown>;
}

function createConnectionStub(): ConnectionStub {
  const stub: ConnectionStub = {
    updates: [],
    permissionRequests: [],
    reads: [],
    writes: [],
    terminalCommands: [],
    permissionResponse: () => ({ outcome: { outcome: "selected", optionId: "allow" } }),
    fileResponses: new Map(),
    async sessionUpdate(params) {
      this.updates.push(params);
    },
    async requestPermission(params) {
      this.permissionRequests.push(params);
      return this.permissionResponse();
    },
    async readTextFile({ path }) {
      this.reads.push(path);
      const content = this.fileResponses.get(path);
      if (content === undefined) throw new Error(`file not found: ${path}`);
      return { content };
    },
    async writeTextFile({ path, content }) {
      this.writes.push({ path, content });
    },
    async createTerminal({ command, args }) {
      this.terminalCommands.push({ command, args });
      return {
        id: "term-1",
        async waitForExit() {
          return { exitCode: 0 };
        },
        async currentOutput() {
          return { output: "(stub output)" };
        },
        async release() {
          /* noop */
        },
      };
    },
  };
  return stub;
}

function makeStreamingGlm(steps: Array<GlmStreamChunk[]>) {
  let i = 0;
  return {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      const step = steps[i++];
      if (!step) throw new Error("streamChat called more times than expected");
      for (const chunk of step) yield chunk;
    },
  };
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

test("initialize returns negotiated protocol version, agent info, and auth methods", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });

  const result = await agent.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: {
      fs: { readTextFile: true, writeTextFile: true },
      terminal: true,
    },
  });

  assert.equal(result.protocolVersion, PROTOCOL_VERSION);
  assert.equal(result.agentInfo?.name, "glm-acp-agent");
  assert.equal(result.agentCapabilities?.loadSession, true);
  assert.equal(result.agentCapabilities?.promptCapabilities?.embeddedContext, true);
  assert.equal(result.agentCapabilities?.promptCapabilities?.image, true);
  assert.ok(result.agentCapabilities?.sessionCapabilities?.close);
  assert.ok(result.agentCapabilities?.sessionCapabilities?.list);
  assert.ok(result.agentCapabilities?.sessionCapabilities?.fork);
  assert.ok(result.agentCapabilities?.sessionCapabilities?.resume);
  assert.ok(Array.isArray(result.authMethods));
  const envVarMethod = result.authMethods?.find(
    (m) => (m as { type?: string }).type === "env_var"
  );
  assert.ok(envVarMethod, "env_var auth method should be advertised");
  // The ACP registry verifier requires at least one method of type `agent`
  // (no discriminator) or `terminal`. We advertise an `agent` method since
  // the agent reads Z_AI_API_KEY itself at startup with no extra UI.
  const agentMethod = result.authMethods?.find(
    (m) => (m as { type?: string }).type === undefined
  );
  assert.ok(agentMethod, "agent-default auth method should be advertised");
  assert.equal((agentMethod as { id: string }).id, "z-ai-api-key");
});

test("initialize negotiates lower protocol version when client requests one", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });

  const result = await agent.initialize({
    protocolVersion: 0,
    clientCapabilities: {},
  });

  assert.equal(result.protocolVersion, 0);
});

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

test("newSession returns a unique session id and seeds a system prompt", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await agent.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true }, terminal: true },
  });

  const a = await agent.newSession({ cwd: "/tmp/a", mcpServers: [] });
  const b = await agent.newSession({ cwd: "/tmp/b", mcpServers: [] });

  assert.notEqual(a.sessionId, b.sessionId);

  const list = await agent.listSessions({});
  assert.equal(list.sessions.length, 2);
});

test("system prompt advertises only the tools matching client capabilities", async () => {
  const conn = createConnectionStub();
  let captured = "";
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown }>
    ): AsyncGenerator<GlmStreamChunk> {
      const sys = messages.find((m) => m.role === "system");
      captured = typeof sys?.content === "string" ? sys.content : "";
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  // Explicitly empty capabilities: the client has no fs, no terminal.
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });

  // Web tools are unconditional, but file/terminal tools must NOT appear.
  assert.ok(captured.includes("web_search"));
  assert.ok(captured.includes("web_reader"));
  assert.ok(!captured.includes("read_file"));
  assert.ok(!captured.includes("write_file"));
  assert.ok(!captured.includes("list_files"));
  assert.ok(!captured.includes("run_command"));
});

test("system prompt falls back to all tools when client never sent capabilities", async () => {
  const conn = createConnectionStub();
  let captured = "";
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown }>
    ): AsyncGenerator<GlmStreamChunk> {
      const sys = messages.find((m) => m.role === "system");
      captured = typeof sys?.content === "string" ? sys.content : "";
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  // Skip initialize on purpose so clientCapabilities stays null.
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });

  for (const name of [
    "read_file",
    "write_file",
    "list_files",
    "run_command",
    "web_search",
    "web_reader",
  ]) {
    assert.ok(captured.includes(name), `expected system prompt to mention ${name}`);
  }
});

test("listSessions can filter by cwd", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });

  await agent.newSession({ cwd: "/tmp/a", mcpServers: [] });
  await agent.newSession({ cwd: "/tmp/a", mcpServers: [] });
  await agent.newSession({ cwd: "/tmp/b", mcpServers: [] });

  const filtered = await agent.listSessions({ cwd: "/tmp/a" });
  assert.equal(filtered.sessions.length, 2);
  for (const s of filtered.sessions) assert.equal(s.cwd, "/tmp/a");
});

test("closeSession removes the session and a subsequent prompt fails", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.closeSession({ sessionId });
  const list = await agent.listSessions({});
  assert.equal(list.sessions.length, 0);

  await assert.rejects(
    () =>
      agent.prompt({
        sessionId,
        prompt: [{ type: "text", text: "hi" }],
      }),
    /Session not found/
  );
});

test("authenticate is a no-op", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  const result = await agent.authenticate({ methodId: "z_ai_api_key" });
  assert.deepEqual(result, {});
});

test("setSessionMode is a no-op", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  const result = await agent.setSessionMode({ sessionId, modeId: "ask" });
  assert.deepEqual(result, {});
});

// ---------------------------------------------------------------------------
// Prompt loop
// ---------------------------------------------------------------------------

test("prompt streams agent_message_chunk and returns end_turn", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([[{ text: "Hello, world!" }, { done: true, stopReason: "stop" }]]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "Hi" }],
  });

  assert.equal(result.stopReason, "end_turn");

  const messageChunks = conn.updates.filter(
    (u) => (u.update as { sessionUpdate: string }).sessionUpdate === "agent_message_chunk"
  );
  assert.equal(messageChunks.length, 1);
  const sessionInfo = conn.updates.filter(
    (u) => (u.update as { sessionUpdate: string }).sessionUpdate === "session_info_update"
  );
  assert.equal(sessionInfo.length, 1);
  assert.ok((sessionInfo[0] as { update: { title?: string } }).update.title);
});

test("prompt forwards reasoning_content as agent_thought_chunk", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([
    [
      { thinking: "Let me think..." },
      { text: "Done." },
      { done: true, stopReason: "stop" },
    ],
  ]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "Hi" }] });

  const thoughts = conn.updates.filter(
    (u) => (u.update as { sessionUpdate: string }).sessionUpdate === "agent_thought_chunk"
  );
  assert.equal(thoughts.length, 1);
});

test("prompt maps content_filter stop reason to refusal", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([
    [{ text: "I can't help." }, { done: true, stopReason: "content_filter" }],
  ]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "..." }],
  });
  assert.equal(result.stopReason, "refusal");
});

test("prompt maps length stop reason to max_tokens", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([
    [{ text: "..." }, { done: true, stopReason: "length" }],
  ]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "..." }],
  });
  assert.equal(result.stopReason, "max_tokens");
});

test("prompt loop returns max_turn_requests after exhausting tool turns", async () => {
  const conn = createConnectionStub();
  // Each turn produces a tool call; we'll stub maxTurns=2 so we hit the limit fast.
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { toolCall: { id: "tc1", name: "unknown_tool", arguments: "{}" } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, maxTurns: 2, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "do things" }],
  });
  assert.equal(result.stopReason, "max_turn_requests");
});

test("prompt cancellation returns cancelled stop reason", async () => {
  const conn = createConnectionStub();
  // Latches that make the cancel point deterministic: the test waits until
  // the model has yielded at least one chunk before calling cancel, and the
  // model waits for the cancel signal to fire before completing.
  let resolveStarted!: () => void;
  const started = new Promise<void>((r) => (resolveStarted = r));

  const glm = {
    async *streamChat(_msgs: unknown, signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
      yield { text: "starting..." };
      resolveStarted();
      // Suspend until cancellation fires.
      await new Promise<void>((resolve) => {
        if (signal?.aborted) return resolve();
        signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const promptPromise = agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "hi" }],
    messageId: "abcd1234-abcd-abcd-abcd-abcdabcd1234",
  });
  await started;
  await agent.cancel({ sessionId });
  const result = await promptPromise;
  assert.equal(result.stopReason, "cancelled");
  assert.equal(result.userMessageId, "abcd1234-abcd-abcd-abcd-abcdabcd1234");
});

test("prompt echoes userMessageId and reports usage", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([
    [
      { text: "ok" },
      { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } },
      { done: true, stopReason: "stop" },
    ],
  ]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "hi" }],
    messageId: "12345678-1234-1234-1234-123456789abc",
  });
  assert.equal(result.userMessageId, "12345678-1234-1234-1234-123456789abc");
  assert.deepEqual(result.usage, { inputTokens: 10, outputTokens: 5, totalTokens: 15 });
});

test("prompt converts resource_link and embedded resource blocks", async () => {
  const conn = createConnectionStub();
  let captured = "";
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown }>
    ): AsyncGenerator<GlmStreamChunk> {
      const userMsg = messages.find((m) => m.role === "user");
      captured = typeof userMsg?.content === "string" ? userMsg.content : "";
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({
    sessionId,
    prompt: [
      { type: "text", text: "Look at:" },
      { type: "resource_link", uri: "file:///tmp/a.ts", name: "a.ts" },
      {
        type: "resource",
        resource: { uri: "file:///tmp/b.ts", text: "console.log(1)", mimeType: "text/plain" },
      },
    ],
  });

  assert.ok(captured.includes("Look at:"));
  assert.ok(captured.includes("[a.ts](file:///tmp/a.ts)"));
  assert.ok(captured.includes("<resource uri=\"file:///tmp/b.ts\">"));
  assert.ok(captured.includes("console.log(1)"));
});

test("tool call result is fed back into the next streamChat call", async () => {
  const conn = createConnectionStub();
  conn.fileResponses.set("/tmp/x.ts", "export const x = 1;");

  let callIndex = 0;
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown; tool_call_id?: string }>
    ): AsyncGenerator<GlmStreamChunk> {
      callIndex++;
      if (callIndex === 1) {
        yield {
          toolCall: { id: "tc1", name: "read_file", arguments: JSON.stringify({ path: "/tmp/x.ts" }) },
        };
        yield { done: true, stopReason: "tool_calls" };
      } else {
        // Validate that the tool result was fed back in.
        const toolMsg = messages.find((m) => m.role === "tool");
        assert.ok(toolMsg, "expected a tool role message in the second call");
        assert.equal(toolMsg?.tool_call_id, "tc1");
        assert.equal(toolMsg?.content, "export const x = 1;");
        yield { text: "Done." };
        yield { done: true, stopReason: "stop" };
      }
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const result = await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "read it" }],
  });
  assert.equal(result.stopReason, "end_turn");
  assert.equal(callIndex, 2);
  assert.deepEqual(conn.reads, ["/tmp/x.ts"]);
});

test("prompt without title sets title from first user message", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([
    [{ text: "ok" }, { done: true, stopReason: "stop" }],
  ]);
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({
    sessionId,
    prompt: [{ type: "text", text: "How do I write a TypeScript function?" }],
  });

  const list = await agent.listSessions({});
  assert.ok(list.sessions[0]?.title?.includes("How do I write a TypeScript function?"));
});

test("a follow-up prompt waits for the previous loop to fully unwind", async () => {
  const conn = createConnectionStub();

  // The first call suspends until aborted; the second one must not start
  // until the first has fully exited.
  let firstStartedResolve!: () => void;
  const firstStarted = new Promise<void>((r) => (firstStartedResolve = r));
  let firstReturned = false;
  let secondStartedBeforeFirstReturned = false;
  let callCount = 0;

  const glm = {
    async *streamChat(_msgs: unknown, signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
      callCount++;
      if (callCount === 1) {
        yield { text: "first" };
        firstStartedResolve();
        await new Promise<void>((resolve) => {
          if (signal?.aborted) return resolve();
          signal?.addEventListener("abort", () => resolve(), { once: true });
        });
        // simulate some unwinding work
        await new Promise<void>((r) => setImmediate(r));
        firstReturned = true;
        yield { done: true, stopReason: "stop" };
      } else {
        if (!firstReturned) secondStartedBeforeFirstReturned = true;
        yield { text: "second" };
        yield { done: true, stopReason: "stop" };
      }
    },
  };

  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  const first = agent.prompt({ sessionId, prompt: [{ type: "text", text: "go 1" }] });
  await firstStarted;
  const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "go 2" }] });

  const [r1, r2] = await Promise.all([first, second]);
  assert.equal(r1.stopReason, "cancelled");
  assert.equal(r2.stopReason, "end_turn");
  assert.equal(secondStartedBeforeFirstReturned, false);
});

// ---------------------------------------------------------------------------
// Per-session model + unstable_setSessionModel
// ---------------------------------------------------------------------------

test("newSession returns a SessionModelState with availableModels and currentModelId", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });

  const result = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  assert.ok(result.models, "expected models field on NewSessionResponse");
  assert.ok(Array.isArray(result.models?.availableModels));
  assert.ok((result.models?.availableModels.length ?? 0) >= 2);
  assert.equal(typeof result.models?.currentModelId, "string");
});

test("unstable_setSessionModel updates the model used on the next prompt", async () => {
  const conn = createConnectionStub();
  const seenModels: Array<string | undefined> = [];
  const glm = {
    async *streamChat(
      _msgs: unknown,
      _signal?: AbortSignal,
      options?: { model?: string }
    ): AsyncGenerator<GlmStreamChunk> {
      seenModels.push(options?.model);
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId, models } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
  await agent.unstable_setSessionModel({ sessionId, modelId: "glm-4.5-air" });
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "second" }] });

  assert.equal(seenModels[0], models?.currentModelId);
  assert.equal(seenModels[1], "glm-4.5-air");
});

test("unstable_setSessionModel rejects unknown sessions", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await assert.rejects(
    () => agent.unstable_setSessionModel({ sessionId: "missing", modelId: "glm-5.1" }),
    /Session not found/
  );
});

// ---------------------------------------------------------------------------
// Image content
// ---------------------------------------------------------------------------

test("prompt with image block forwards a multimodal user message", async () => {
  const conn = createConnectionStub();
  let captured: unknown;
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown }>
    ): AsyncGenerator<GlmStreamChunk> {
      const userMsg = messages.find((m) => m.role === "user");
      captured = userMsg?.content;
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({
    sessionId,
    prompt: [
      { type: "text", text: "What is in this image?" },
      { type: "image", data: "AAAA", mimeType: "image/png" },
    ],
  });

  assert.ok(Array.isArray(captured), "expected multimodal content array");
  const arr = captured as Array<{ type: string; image_url?: { url: string }; text?: string }>;
  const textPart = arr.find((p) => p.type === "text");
  const imagePart = arr.find((p) => p.type === "image_url");
  assert.equal(textPart?.text, "What is in this image?");
  assert.equal(imagePart?.image_url?.url, "data:image/png;base64,AAAA");
});

test("prompt without image blocks keeps content as a plain string", async () => {
  const conn = createConnectionStub();
  let captured: unknown;
  const glm = {
    async *streamChat(
      messages: ReadonlyArray<{ role: string; content?: unknown }>
    ): AsyncGenerator<GlmStreamChunk> {
      const userMsg = messages.find((m) => m.role === "user");
      captured = userMsg?.content;
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });

  assert.equal(typeof captured, "string");
  assert.equal(captured, "hello");
});

test("invalid maxTurns falls back to the default", async () => {
  const conn = createConnectionStub();
  const glm = makeStreamingGlm([[{ text: "ok" }, { done: true, stopReason: "stop" }]]);

  for (const bad of [0, -1, NaN, Number.POSITIVE_INFINITY]) {
    const agent = new GlmAcpAgent(conn as never, { glm: { ...glm }, maxTurns: bad, sessionStore: null });
    assert.equal((agent as unknown as { maxTurns: number }).maxTurns, 20);
  }
});

// ---------------------------------------------------------------------------
// Session persistence (loadSession / fork / resume)
// ---------------------------------------------------------------------------

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SessionStore } from "../protocol/session-store.js";

function makeTempStore(): { store: SessionStore; cleanup: () => void } {
  const dir = mkdtempSync(join(tmpdir(), "glm-acp-test-"));
  const store = new SessionStore(dir);
  return {
    store,
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
  };
}

test("prompt persists session state to the SessionStore", async () => {
  const { store, cleanup } = makeTempStore();
  try {
    const conn = createConnectionStub();
    const glm = makeStreamingGlm([[{ text: "hi back" }, { done: true, stopReason: "stop" }]]);
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
    await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hi" }] });

    const persisted = store.load(sessionId);
    assert.ok(persisted, "expected session to be persisted");
    assert.equal(persisted?.cwd, "/tmp");
    // system + user + assistant
    assert.ok((persisted?.messages.length ?? 0) >= 3);
    assert.ok(persisted?.title);
  } finally {
    cleanup();
  }
});

test("loadSession restores messages and replays them as session updates", async () => {
  const { store, cleanup } = makeTempStore();
  try {
    store.save({
      sessionId: "abcd1234-abcd-abcd-abcd-abcdabcd1234",
      cwd: "/tmp",
      messages: [
        { role: "system", content: "you are a coding assistant" },
        { role: "user", content: "ping" },
        { role: "assistant", content: "pong" },
      ],
      title: "ping pong",
      updatedAt: "2026-01-01T00:00:00.000Z",
      model: "glm-5.1",
    });

    const conn = createConnectionStub();
    const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
    const result = await agent.loadSession({
      sessionId: "abcd1234-abcd-abcd-abcd-abcdabcd1234",
      cwd: "/tmp",
      mcpServers: [],
    });

    assert.ok(result.models, "expected models in load response");
    assert.equal(result.models?.currentModelId, "glm-5.1");

    const userChunks = conn.updates.filter(
      (u) => (u.update as { sessionUpdate: string }).sessionUpdate === "user_message_chunk"
    );
    const assistantChunks = conn.updates.filter(
      (u) => (u.update as { sessionUpdate: string }).sessionUpdate === "agent_message_chunk"
    );
    assert.equal(userChunks.length, 1);
    assert.equal(assistantChunks.length, 1);
  } finally {
    cleanup();
  }
});

test("unstable_forkSession creates a new sessionId with a deep-copied history", async () => {
  const { store, cleanup } = makeTempStore();
  try {
    const conn = createConnectionStub();
    const glm = makeStreamingGlm([[{ text: "ok" }, { done: true, stopReason: "stop" }]]);
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
    await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });

    const fork = await agent.unstable_forkSession({
      sessionId,
      cwd: "/tmp",
      mcpServers: [],
    });

    assert.notEqual(fork.sessionId, sessionId);
    assert.ok(fork.models);

    // Mutating the fork shouldn't affect the original.
    const original = store.load(sessionId);
    const forked = store.load(fork.sessionId);
    assert.ok(original);
    assert.ok(forked);
    assert.notEqual(original?.messages, forked?.messages);
    assert.equal(original?.messages.length, forked?.messages.length);
  } finally {
    cleanup();
  }
});

test("resumeSession restores in-memory state without replaying messages", async () => {
  const { store, cleanup } = makeTempStore();
  try {
    store.save({
      sessionId: "abcd1234-abcd-abcd-abcd-abcdabcd1234",
      cwd: "/tmp",
      messages: [
        { role: "system", content: "you are a coding assistant" },
        { role: "user", content: "ping" },
        { role: "assistant", content: "pong" },
      ],
      title: "ping pong",
      updatedAt: "2026-01-01T00:00:00.000Z",
      model: "glm-5.1",
    });

    const conn = createConnectionStub();
    const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
    const result = await agent.resumeSession({
      sessionId: "abcd1234-abcd-abcd-abcd-abcdabcd1234",
      cwd: "/tmp",
      mcpServers: [],
    });

    assert.equal(result.models?.currentModelId, "glm-5.1");
    // No replay updates expected.
    assert.equal(conn.updates.length, 0);
  } finally {
    cleanup();
  }
});

test("loadSession throws when persistence is disabled", async () => {
  const conn = createConnectionStub();
  const agent = new GlmAcpAgent(conn as never, { sessionStore: null });
  await assert.rejects(
    () => agent.loadSession({ sessionId: "x", cwd: "/tmp", mcpServers: [] }),
    /persistence is disabled/
  );
});

test("listSessions surfaces persisted-but-not-in-memory sessions", async () => {
  const { store, cleanup } = makeTempStore();
  try {
    store.save({
      sessionId: "11111111-1111-1111-1111-111111111111",
      cwd: "/tmp",
      messages: [{ role: "system", content: "" }],
      title: "Saved earlier",
      updatedAt: "2026-01-01T00:00:00.000Z",
      model: "glm-5.1",
    });

    const conn = createConnectionStub();
    const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
    const list = await agent.listSessions({});
    assert.equal(list.sessions.length, 1);
    assert.equal(list.sessions[0]?.sessionId, "11111111-1111-1111-1111-111111111111");
    assert.equal(list.sessions[0]?.title, "Saved earlier");
  } finally {
    cleanup();
  }
});
