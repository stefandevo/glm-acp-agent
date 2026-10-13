import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, realpathSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
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
  const permissionRequests: unknown[] = [];
  return {
    updates,
    permissionRequests,
    signal: new AbortController().signal,
    async sessionUpdate(params: Record<string, unknown>) {
      updates.push(params);
    },
    async requestPermission(params: unknown) {
      permissionRequests.push(params);
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((onResolve, onReject) => {
    resolve = onResolve;
    reject = onReject;
  });
  return { promise, resolve, reject };
}

async function withoutNotificationRelease<T>(promise: Promise<T>): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("prompt/close waited for held notification")), 2_000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

for (const cancelAtNotice of [false, true]) {
  test(`${cancelAtNotice ? "cancelling" : "completing"} the turn-limit notice preserves completed writes and usage before close`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-turn-limit-checkpoint-"));
    const storeRoot = join(cwd, "sessions");
    const store = new SessionStore(storeRoot);
    const conn = connection();
    const held = deferred<void>();
    const started = deferred<void>();
    const sendUpdate = conn.sessionUpdate;
    conn.sessionUpdate = async params => {
      await sendUpdate(params);
      const update = params.update as { sessionUpdate?: string; content?: { text?: string } };
      if (update.sessionUpdate === "agent_message_chunk" && update.content?.text?.includes("[stopped: reached")) {
        started.resolve();
        await held.promise;
      }
    };
    let modelCalls = 0;
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        yield { usage: { inputTokens: 7, outputTokens: 3, totalTokens: 10 } };
        yield { toolCall: { id: "completed-write", name: "write_file", arguments: JSON.stringify({
          path: "written.txt", content: "completed side effect",
        }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store, maxTurns: 1 });
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await store.flush();
    const prompt = agent.prompt({ sessionId, messageId: "write-once", prompt: [{ type: "text", text: "write once" }] });
    try {
      await withoutNotificationRelease(started.promise);
      assert.equal(await readFile(join(cwd, "written.txt"), "utf8"), "completed side effect");
      if (cancelAtNotice) await agent.cancel({ sessionId });
      else held.resolve();
      const response = await withoutNotificationRelease(prompt);

      // A fresh store must see the prompt's checkpoint before close, restore,
      // or shutdown can supply an unrelated save and hide a missing write.
      const history = new SessionStore(storeRoot).load(sessionId)?.messages;
      const toolResult = history?.find(message => message.role === "tool" && message.tool_call_id === "completed-write");
      const toolBatch = history?.find(message => message.role === "assistant" && message.tool_calls?.some(call => call.id === "completed-write"));
      assert.deepEqual({ response, toolResult, toolCallIds: toolBatch?.role === "assistant" ? toolBatch.tool_calls?.map(call => call.id) : undefined }, {
        response: {
          stopReason: cancelAtNotice ? "cancelled" : "max_turn_requests",
          usage: { inputTokens: 7, outputTokens: 3, totalTokens: 10 },
          userMessageId: "write-once",
        },
        toolResult: { role: "tool", tool_call_id: "completed-write", content: "File written successfully: written.txt" },
        toolCallIds: ["completed-write"],
      });
      const updateCount = conn.updates.length;
      const permissionCount = conn.permissionRequests.length;
      held.reject(new Error("late turn-limit notification rejection"));
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(conn.updates.length, updateCount, "late settlement must not resume notifications");
      assert.equal(conn.permissionRequests.length, permissionCount);
      assert.equal(modelCalls, 1, "the terminal notice must not start another model request");
    } finally {
      held.resolve();
      await Promise.allSettled([prompt]);
      await agent.closeSession({ sessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

test("a turn-limit notification failure without cancellation still rejects the prompt", async () => {
  const conn = connection();
  const failure = new Error("turn-limit transport failed");
  const sendUpdate = conn.sessionUpdate;
  conn.sessionUpdate = async params => {
    await sendUpdate(params);
    const update = params.update as { content?: { text?: string } };
    if (update.content?.text?.includes("[stopped: reached")) throw failure;
  };
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { toolCall: { id: "completed-todo", name: "todowrite", arguments: JSON.stringify({
        todos: [{ content: "Completed task", status: "completed" }],
      }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null, maxTurns: 1 });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "finish the task" }] }), failure);
  } finally {
    await agent.closeSession({ sessionId });
  }
});

for (const kind of ["text", "thought", "metadata", "error", "max-turn", "todo"] as const) {
  test(`cancel and close unwind a held ${kind} notification and consume its late rejection`, async () => {
    const conn = connection();
    const held = deferred<void>();
    const started = deferred<void>();
    const sendUpdate = conn.sessionUpdate;
    conn.sessionUpdate = async (params) => {
      await sendUpdate(params);
      const update = params.update as { sessionUpdate?: string; status?: string };
      const target = kind === "thought" ? "agent_thought_chunk"
        : kind === "metadata" ? "session_info_update"
        : kind === "todo" ? "tool_call" : "agent_message_chunk";
      if (update.sessionUpdate === target) {
        started.resolve();
        await held.promise;
      }
    };
    let modelCalls = 0;
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        if (kind === "error") throw new Error("provider fixture failed");
        if (kind === "todo" || kind === "max-turn") {
          yield { toolCall: { id: "todo-held", name: "todowrite", arguments: JSON.stringify({
            todos: [{ content: "Completed task", status: "completed" }],
          }) } };
          yield { done: true, stopReason: "tool_calls" };
        } else {
          yield kind === "thought" ? { thinking: "thought before cancel", text: "stale same-chunk text" }
            : { text: "text before cancel" };
          yield { done: true, stopReason: "stop", usage: { inputTokens: 3, outputTokens: 2, totalTokens: 5 } };
        }
        if (kind === "text" || kind === "thought") yield { text: "stale after cancel" };
      },
    };
    const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-held-notification-"));
    const store = new SessionStore(storeRoot);
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store, maxTurns: 1 });
    const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
    await new Promise<void>(resolve => setImmediate(resolve));
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
    let close: Promise<unknown> | undefined;
    try {
      await started.promise;
      await agent.cancel({ sessionId });
      close = agent.closeSession({ sessionId });
      const [response] = await withoutNotificationRelease(Promise.all([prompt, close]));
      assert.equal(response.stopReason, "cancelled");
      if (kind === "metadata") assert.deepEqual(response.usage, { inputTokens: 3, outputTokens: 2, totalTokens: 5 });
      const updateCount = conn.updates.length;
      held.reject(new Error("late notification rejection"));
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(conn.updates.length, updateCount, "late settlement must not resume UI updates");
      assert.equal(modelCalls, 1, "cancel must not start another model request");
      assert.ok(!conn.updates.some(update => JSON.stringify(update).includes("stale")));
      const history = store.load(sessionId)?.messages;
      if (kind === "text" || kind === "metadata") {
        assert.ok(history?.some(message => message.role === "assistant" && message.content === "text before cancel"));
      }
      if (kind === "todo") {
        assert.ok(history?.some(message => message.role === "tool" && message.tool_call_id === "todo-held" &&
          String(message.content).includes("Todo list updated:")));
      }
    } finally {
      held.resolve();
      await Promise.allSettled([prompt, close]);
      await agent.closeSession({ sessionId });
      await rm(storeRoot, { recursive: true, force: true });
    }
  });
}

for (const status of ["pending", "in_progress", "completed", "failed"] as const) {
  test(`cancel and close unwind a held ${status} tool card without losing known writes`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-held-tool-card-"));
    const store = new SessionStore(join(cwd, "sessions"));
    const conn = connection();
    const held = deferred<void>();
    const started = deferred<void>();
    const sendUpdate = conn.sessionUpdate;
    conn.sessionUpdate = async params => {
      await sendUpdate(params);
      const update = params.update as { status?: string };
      if (update.status === status) {
        started.resolve();
        await held.promise;
      }
    };
    let modelCalls = 0;
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        yield { toolCall: { id: "write-held", name: "write_file", arguments: status === "failed" ? "null"
          : JSON.stringify({ path: "written.txt", content: "completed side effect" }) } };
        yield { toolCall: { id: "write-stale", name: "write_file", arguments: JSON.stringify({ path: "stale.txt", content: "stale" }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await new Promise<void>(resolve => setImmediate(resolve));
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "write" }] });
    let close: Promise<unknown> | undefined;
    try {
      await started.promise;
      await agent.cancel({ sessionId });
      close = agent.closeSession({ sessionId });
      const [response] = await withoutNotificationRelease(Promise.all([prompt, close]));
      assert.equal(response.stopReason, "cancelled");
      assert.equal(modelCalls, 1);
      assert.equal(existsSync(join(cwd, "stale.txt")), false, "the next write must not run");
      const history = store.load(sessionId)?.messages;
      const result = history?.find(message => message.role === "tool" && message.tool_call_id === "write-held");
      assert.ok(result && result.role === "tool");
      if (status === "completed") {
        assert.equal(await readFile(join(cwd, "written.txt"), "utf8"), "completed side effect");
        assert.equal(result.content, "File written successfully: written.txt");
      } else {
        assert.equal(existsSync(join(cwd, "written.txt")), false);
      }
      if (status === "failed") assert.match(String(result.content), /tool arguments must be a JSON object/i);
      const updateCount = conn.updates.length;
      const permissionCount = conn.permissionRequests.length;
      held.resolve();
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(conn.updates.length, updateCount);
      assert.equal(conn.permissionRequests.length, permissionCount);
      if (status === "pending") assert.equal(permissionCount, 0, "cancel must not start a permission request");
    } finally {
      held.resolve();
      await Promise.allSettled([prompt, close]);
      await agent.closeSession({ sessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

test("cancelled pending permission publishes a failed card without waiting for its delivery", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-acp-cancelled-permission-card-"));
  const storeRoot = join(cwd, "sessions");
  const permission = deferred<{ outcome: { outcome: "selected"; optionId: string } }>();
  const permissionStarted = deferred<void>();
  const terminalDelivery = deferred<void>();
  let writes = 0;
  let modelCalls = 0;
  const conn = {
    ...connection(),
    async requestPermission(params: unknown) {
      conn.permissionRequests.push(params);
      permissionStarted.resolve();
      return permission.promise;
    },
    async writeTextFile() { writes += 1; },
  };
  conn.sessionUpdate = async params => {
    conn.updates.push(params);
    if ((params.update as { status?: string }).status === "failed") await terminalDelivery.promise;
  };
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      modelCalls += 1;
      yield { toolCall: { id: "permission-held", name: "write_file", arguments: JSON.stringify({ path: "written.txt", content: "do not write" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: new SessionStore(storeRoot) });
  await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { writeTextFile: true } } } as never);
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "write" }] });
  try {
    await withoutNotificationRelease(permissionStarted.promise);
    await agent.cancel({ sessionId });
    assert.equal((await withoutNotificationRelease(prompt)).stopReason, "cancelled");
    const cards = conn.updates.map(params => params.update as { toolCallId?: string; status?: string; rawOutput?: unknown })
      .filter(update => update.toolCallId === "permission-held");
    assert.deepEqual(cards.map(card => card.status), ["pending", "failed"]);
    assert.deepEqual(cards[1].rawOutput, { error: "Cancelled by turn." });
    const result = new SessionStore(storeRoot).load(sessionId)?.messages.find(message => message.role === "tool" && message.tool_call_id === "permission-held");
    assert.equal(result?.content, "Write cancelled by turn.", "the cancelled result must be durable before close");
    assert.equal(writes, 0);
    assert.equal(modelCalls, 1);
    const updateCount = conn.updates.length;
    terminalDelivery.reject(new Error("late failed-card delivery rejection"));
    permission.resolve({ outcome: { outcome: "selected", optionId: "allow" } });
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(conn.updates.length, updateCount, "late settlement must not resume notification work");
    assert.equal(writes, 0, "late permission must not start an editor write");
    assert.equal(conn.permissionRequests.length, 1);
  } finally {
    terminalDelivery.resolve();
    permission.resolve({ outcome: { outcome: "selected", optionId: "allow" } });
    await Promise.allSettled([prompt]);
    await agent.closeSession({ sessionId });
    await rm(cwd, { recursive: true, force: true });
  }
});

for (const outcome of ["completed", "failed"] as const) {
  test(`an editor write ${outcome} after cancel publishes its actual terminal card without waiting for delivery`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-cancelled-write-card-"));
    const storeRoot = join(cwd, "sessions");
    const writeStarted = deferred<void>();
    const writeRelease = deferred<void>();
    const terminalDelivery = deferred<void>();
    let writes = 0;
    let modelCalls = 0;
    const conn = {
      ...connection(),
      async writeTextFile(params: { path: string; content: string }) {
        writes += 1;
        writeStarted.resolve();
        await writeRelease.promise;
        if (outcome === "failed") throw new Error("editor refused the admitted write");
        await writeFile(params.path, params.content);
      },
    };
    conn.sessionUpdate = async params => {
      conn.updates.push(params);
      if ((params.update as { status?: string }).status === outcome) await terminalDelivery.promise;
    };
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        yield { toolCall: { id: "write-admitted", name: "write_file", arguments: JSON.stringify({ path: "written.txt", content: "admitted write" }) } };
        yield { toolCall: { id: "write-stale", name: "write_file", arguments: JSON.stringify({ path: "stale.txt", content: "do not write" }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: new SessionStore(storeRoot) });
    await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { writeTextFile: true } } } as never);
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "write" }] });
    try {
      await withoutNotificationRelease(writeStarted.promise);
      await agent.cancel({ sessionId });
      writeRelease.resolve();
      assert.equal((await withoutNotificationRelease(prompt)).stopReason, "cancelled");
      const cards = conn.updates.map(params => params.update as { toolCallId?: string; status?: string; rawOutput?: unknown })
        .filter(update => update.toolCallId === "write-admitted");
      assert.deepEqual(cards.map(card => card.status), ["pending", "in_progress", outcome]);
      assert.deepEqual(cards[2].rawOutput, outcome === "completed"
        ? { success: true } : { error: "editor refused the admitted write" });
      const messages = new SessionStore(storeRoot).load(sessionId)?.messages;
      const result = messages?.find(message => message.role === "tool" && message.tool_call_id === "write-admitted");
      assert.equal(result?.content, outcome === "completed"
        ? "File written successfully: written.txt" : "Error writing file: editor refused the admitted write",
      "the known editor outcome must be durable before close");
      const skipped = messages?.find(message => message.role === "tool" && message.tool_call_id === "write-stale");
      assert.equal(skipped?.content, "Tool call cancelled before execution.");
      assert.equal(writes, 1);
      assert.equal(modelCalls, 1);
      assert.equal(conn.permissionRequests.length, 1);
      assert.equal(existsSync(join(cwd, "stale.txt")), false);
      if (outcome === "completed") assert.equal(await readFile(join(cwd, "written.txt"), "utf8"), "admitted write");
      else assert.equal(existsSync(join(cwd, "written.txt")), false);
      const updateCount = conn.updates.length;
      terminalDelivery.reject(new Error("late terminal write-card delivery rejection"));
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(conn.updates.length, updateCount);
      assert.equal(writes, 1);
    } finally {
      writeRelease.resolve();
      terminalDelivery.resolve();
      await Promise.allSettled([prompt]);
      await agent.closeSession({ sessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

for (const { toolName, status, args, statuses, permissionCount } of [
  { toolName: "write_file", status: "pending", args: { path: "written.txt", content: "do not write" }, statuses: ["pending", "failed"], permissionCount: 0 },
  { toolName: "write_file", status: "in_progress", args: { path: "written.txt", content: "do not write" }, statuses: ["pending", "in_progress", "failed"], permissionCount: 1 },
  { toolName: "run_command", status: "in_progress", args: { command: "printf do-not-run > written.txt" }, statuses: ["pending", "in_progress", "failed"], permissionCount: 1 },
  { toolName: "read_file", status: "in_progress", args: { path: "written.txt" }, statuses: ["in_progress", "failed"], permissionCount: 0 },
  { toolName: "custom_mcp_tool", status: "in_progress", args: {}, statuses: ["in_progress", "failed"], permissionCount: 0 },
] as const) {
  test(`cancelled held ${status} ${toolName} announcement publishes a failed card and a cancelled-before-execution result`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-cancelled-announcement-card-"));
    const storeRoot = join(cwd, "sessions");
    const announcement = deferred<void>();
    const announcementStarted = deferred<void>();
    const terminalDelivery = deferred<void>();
    let startedOperations = 0;
    let modelCalls = 0;
    const conn = {
      ...connection(),
      async readTextFile() { startedOperations += 1; return { content: "do not read" }; },
      async writeTextFile() { startedOperations += 1; },
    };
    conn.sessionUpdate = async params => {
      conn.updates.push(params);
      const update = params.update as { status?: string };
      if (update.status === status) {
        announcementStarted.resolve();
        await announcement.promise;
      }
      if (update.status === "failed") await terminalDelivery.promise;
    };
    const tools = {
      toolDefinitions: [{ type: "function", function: { name: "custom_mcp_tool", description: "fixture", parameters: { type: "object" } } }],
      hasTool: (name: string) => name === "custom_mcp_tool",
      async callTool() { startedOperations += 1; return { content: [{ type: "text", text: "do not call" }] }; },
      async dispose() {},
    } as unknown as SessionMcpTools;
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        yield { toolCall: { id: "announcement-held", name: toolName, arguments: JSON.stringify(args) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: new SessionStore(storeRoot), mcpConnector: async () => tools });
    await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } } } as never);
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
    try {
      await withoutNotificationRelease(announcementStarted.promise);
      await agent.cancel({ sessionId });
      assert.equal((await withoutNotificationRelease(prompt)).stopReason, "cancelled");
      const cards = conn.updates.map(params => params.update as { toolCallId?: string; status?: string })
        .filter(update => update.toolCallId === "announcement-held");
      assert.deepEqual(cards.map(card => card.status), statuses);
      const result = new SessionStore(storeRoot).load(sessionId)?.messages.find(message => message.role === "tool" && message.tool_call_id === "announcement-held");
      assert.equal(result?.content, "Tool call cancelled before execution.", "an interrupted announcement must not imply an unknown side effect");
      assert.equal(startedOperations, 0);
      assert.equal(existsSync(join(cwd, "written.txt")), false, "cancelled command/write must not mutate the filesystem");
      assert.equal(conn.permissionRequests.length, permissionCount);
      assert.equal(modelCalls, 1);
      const updateCount = conn.updates.length;
      terminalDelivery.reject(new Error("late announcement failed-card delivery rejection"));
      announcement.reject(new Error("late announcement delivery rejection"));
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(conn.updates.length, updateCount);
      assert.equal(conn.permissionRequests.length, permissionCount);
      assert.equal(startedOperations, 0);
    } finally {
      announcement.resolve();
      terminalDelivery.resolve();
      await Promise.allSettled([prompt]);
      await agent.closeSession({ sessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

test("an abort during notification dispatch consumes the late rejected notification", async () => {
  const conn = connection();
  const held = deferred<void>();
  const started = deferred<void>();
  conn.sessionUpdate = params => {
    conn.updates.push(params);
    if ((params.update as { sessionUpdate?: string }).sessionUpdate === "agent_message_chunk") {
      void agent.cancel({ sessionId: String(params.sessionId) });
      started.resolve();
      return held.promise;
    }
    return Promise.resolve();
  };
  const agent = new GlmAcpAgent(conn as never, { glm: textGlm(), sessionStore: null });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
  try {
    await started.promise;
    assert.equal((await withoutNotificationRelease(prompt)).stopReason, "cancelled");
    held.reject(new Error("late synchronously aborted notification"));
    await new Promise<void>(resolve => setImmediate(resolve));
  } finally {
    held.resolve();
    await Promise.allSettled([prompt]);
    await agent.closeSession({ sessionId });
  }
});

test("cancellation after a thought notification settles blocks its same-chunk text", async () => {
  const conn = connection();
  conn.sessionUpdate = params => {
    conn.updates.push(params);
    const delivered = Promise.resolve();
    if ((params.update as { sessionUpdate?: string }).sessionUpdate === "agent_thought_chunk") {
      void delivered.then(() => queueMicrotask(() => queueMicrotask(() => {
        void agent.cancel({ sessionId: String(params.sessionId) });
      })));
    }
    return delivered;
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-settled-notification-"));
  const store = new SessionStore(storeRoot);
  let advancedAfterCancellation = false;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { thinking: "delivered thought", text: "stale same-chunk text" };
      advancedAfterCancellation = true;
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] })).stopReason, "cancelled");
    assert.equal(advancedAfterCancellation, false, "an aborted owner must not request another stream chunk");
    assert.ok(!store.load(sessionId)?.messages.some(message => String(message.content).includes("stale")));
    assert.ok(!conn.updates.some(update => JSON.stringify(update).includes("stale")));
  } finally {
    await agent.closeSession({ sessionId });
    await rm(storeRoot, { recursive: true, force: true });
  }
});

for (const { toolName, transfer } of [
  { toolName: "read_file", transfer: "cancel" },
  { toolName: "list_files", transfer: "cancel" },
  { toolName: "custom_mcp_tool", transfer: "cancel" },
  { toolName: "read_file", transfer: "fork" },
  { toolName: "read_file", transfer: "load" },
  { toolName: "custom_mcp_tool", transfer: "fork" },
  { toolName: "custom_mcp_tool", transfer: "load" },
] as const) {
  test(`cancellation after a ${toolName} announcement settles blocks its operation (${transfer})`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-settled-tool-announcement-"));
    const store = new SessionStore(join(cwd, "sessions"));
    let startedOperations = 0;
    let transferPromise: Promise<unknown> | undefined;
    let forkedSessionId: string | undefined;
    let announced = false;
    const conn = {
      ...connection(),
      async readTextFile() {
        startedOperations += 1;
        return { content: "stale file read" };
      },
      async writeTextFile() {},
    };
    conn.sessionUpdate = params => {
      conn.updates.push(params);
      const delivered = Promise.resolve();
      if (!announced && (params.update as { status?: string }).status === "in_progress") {
        announced = true;
        void delivered.then(() => queueMicrotask(() => queueMicrotask(() => {
          const sessionId = String(params.sessionId);
          if (transfer === "cancel") {
            void agent.cancel({ sessionId });
          } else {
            transferPromise = transfer === "fork"
              ? agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] })
              : agent.loadSession({ sessionId, cwd, mcpServers: [] });
            void transferPromise.catch(() => undefined);
          }
        })));
      }
      return delivered;
    };
    const tools = {
      toolDefinitions: [{ type: "function", function: { name: "custom_mcp_tool", description: "fixture", parameters: { type: "object" } } }],
      hasTool: (name: string) => name === "custom_mcp_tool",
      async callTool() {
        startedOperations += 1;
        return { content: [{ type: "text", text: "stale MCP side effect" }] };
      },
      async dispose() {},
    } as unknown as SessionMcpTools;
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield { toolCall: { id: "announced", name: toolName, arguments: JSON.stringify({ path: "." }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store, mcpConnector: async () => tools });
    await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } } } as never);
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    try {
      assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] })).stopReason, "cancelled");
      if (transferPromise) {
        const transferred = await transferPromise as { sessionId?: string };
        if (transfer === "fork") forkedSessionId = transferred.sessionId;
      }
      assert.equal(startedOperations, 0, "cancel must not dispatch editor/MCP operations after announcement");
      const result = store.load(sessionId)?.messages.find(message => message.role === "tool" && message.tool_call_id === "announced");
      assert.equal(result?.content, "Tool call cancelled before execution.");
    } finally {
      await agent.closeSession({ sessionId });
      if (forkedSessionId) await agent.closeSession({ sessionId: forkedSessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

for (const transfer of ["close", "fork", "load"] as const) {
  test(`${transfer} of an unresolved editor write waits for its actual result before checkpointing success`, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "glm-acp-held-editor-write-"));
    const store = new SessionStore(join(cwd, "sessions"));
    const held = deferred<void>();
    const started = deferred<void>();
    const conn = {
      ...connection(),
      async writeTextFile() {
        started.resolve();
        await held.promise;
      },
    };
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield { toolCall: { id: "editor-write", name: "write_file", arguments: JSON.stringify({ path: "buffer.txt", content: "written" }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: store });
    await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { writeTextFile: true } } } as never);
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "write" }] });
    let close: Promise<unknown> | undefined;
    let forkedSessionId: string | undefined;
    try {
      await started.promise;
      if (transfer === "close") {
        await agent.cancel({ sessionId });
        close = agent.closeSession({ sessionId });
      } else {
        close = transfer === "fork"
          ? agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] })
          : agent.loadSession({ sessionId, cwd, mcpServers: [] });
      }
      let promptSettled = false;
      let closeSettled = false;
      void prompt.then(() => { promptSettled = true; });
      void close.then(() => { closeSettled = true; });
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(promptSettled, false);
      assert.equal(closeSettled, false);
      assert.ok(!store.load(sessionId)?.messages.some(message => message.role === "tool" && message.tool_call_id === "editor-write"));
      held.resolve();
      assert.equal((await prompt).stopReason, "cancelled");
      const transferred = await close as { sessionId?: string };
      if (transfer === "fork") forkedSessionId = transferred.sessionId;
      assert.ok(store.load(sessionId)?.messages.some(message => message.role === "tool" &&
        message.tool_call_id === "editor-write" && message.content === "File written successfully: buffer.txt"));
    } finally {
      held.resolve();
      await Promise.allSettled([prompt, close]);
      await agent.closeSession({ sessionId });
      if (forkedSessionId) await agent.closeSession({ sessionId: forkedSessionId });
      await rm(cwd, { recursive: true, force: true });
    }
  });
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
  sessionStore.save = async (session) => {
    saveSawCleanedImage = !existsSync(sourcePath);
    await save(session);
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

test("a malformed tool argument result is paired in history before the model and next prompt continue", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-invalid-tool-history-"));
  const sessionStore = new SessionStore(storeRoot);
  const modelInputs: Array<Array<{ role: string; tool_call_id?: string; content?: unknown; tool_calls?: unknown[] }>> = [];
  let calls = 0;
  const glm = {
    async *streamChat(messages: Array<{ role: string; tool_call_id?: string; content?: unknown; tool_calls?: unknown[] }>): AsyncGenerator<GlmStreamChunk> {
      modelInputs.push(structuredClone(messages));
      calls += 1;
      if (calls === 1) {
        yield { toolCall: { id: "bad-root", name: "write_file", arguments: "null" } };
        yield { done: true, stopReason: "tool_calls" };
        return;
      }
      yield { text: calls === 2 ? "recovered" : "follow-up" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "write it" }] })).stopReason, "end_turn");
    assert.equal(conn.permissionRequests.length, 0);
    const historyForRecovery = modelInputs[1]!;
    const assistantIndex = historyForRecovery.findIndex((message) => message.role === "assistant" && message.tool_calls);
    assert.ok(assistantIndex >= 0);
    assert.deepEqual(historyForRecovery[assistantIndex + 1], {
      role: "tool",
      tool_call_id: "bad-root",
      content: "Error: Tool arguments must be a JSON object.",
    });

    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "continue" }] })).stopReason, "end_turn");
    assert.equal(calls, 3);
    await agent.closeSession({ sessionId });
    const persisted = sessionStore.load(sessionId);
    assert.ok(persisted);
    assert.ok(persisted.messages.some((message) => message.role === "tool" && message.tool_call_id === "bad-root"));
  } finally {
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("an unexpected executor failure settles history and stops until a user follow-up", async () => {
  const conn = connection();
  const sessionUpdate = conn.sessionUpdate.bind(conn);
  let failFailedNotification = true;
  conn.sessionUpdate = async (params: Record<string, unknown>) => {
    const update = params["update"] as { sessionUpdate?: string; status?: string };
    if (failFailedNotification && update.sessionUpdate === "tool_call" && update.status === "failed") {
      failFailedNotification = false;
      throw new Error("synthetic tool notification interruption");
    }
    await sessionUpdate(params);
  };
  const modelInputs: Array<Array<{ role: string; tool_call_id?: string; content?: unknown; tool_calls?: unknown[] }>> = [];
  let calls = 0;
  const glm = {
    async *streamChat(messages: Array<{ role: string; tool_call_id?: string; content?: unknown; tool_calls?: unknown[] }>): AsyncGenerator<GlmStreamChunk> {
      modelInputs.push(structuredClone(messages));
      calls += 1;
      if (calls === 1) {
        yield { toolCall: { id: "interrupted", name: "write_file", arguments: "null" } };
        yield { done: true, stopReason: "tool_calls" };
        return;
      }
      yield { text: "recovered" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "write it" }] }), /outcome.*unknown/i);
  assert.equal(calls, 1, "unexpected failure must not automatically continue the model");
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "inspect what happened" }] });
  assert.equal(calls, 2);
  const tool = modelInputs[1]!.find(message => message.tool_call_id === "interrupted");
  assert.match(String(tool?.content), /outcome.*unknown/i);
});

test("an error after the first write stops remaining tools and records uncertain versus unstarted outcomes", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "glm-tool-uncertain-"));
  const conn = connection();
  const update = conn.sessionUpdate.bind(conn);
  conn.sessionUpdate = async params => {
    const event = params["update"] as { toolCallId?: string; status?: string };
    if (event.toolCallId === "first" && ["completed", "failed"].includes(event.status ?? "")) {
      throw new Error("fixture notification failed after write");
    }
    await update(params);
  };
  const store = new SessionStore(join(cwd, "sessions"));
  let calls = 0;
  const agent = new GlmAcpAgent(conn as never, { sessionStore: store, glm: {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls++;
      for (const id of ["first", "second"]) yield { toolCall: { id, name: "write_file", arguments: JSON.stringify({ path: join(cwd, id), content: "written" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  } });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
    await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "write both" }] }), /outcome.*unknown/i);
    assert.equal(calls, 1);
    assert.equal(existsSync(join(cwd, "first")), true);
    assert.equal(existsSync(join(cwd, "second")), false);
    await agent.closeSession({ sessionId });
    const tools = store.load(sessionId)?.messages.filter(message => message.role === "tool");
    assert.equal(tools?.length, 2);
    assert.match(String(tools?.[0]?.content), /outcome.*unknown/i);
    assert.match(String(tools?.[1]?.content), /not started/i);
  } finally {
    await agent.closeSession({ sessionId });
    await rm(cwd, { recursive: true, force: true });
  }
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
  await store.save({
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
  await store.save({
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
        // Generous bound: the raced close resolves through a real fsync'd
        // session write, which can exceed 100 ms on slow CI runners. The
        // property under test is that close never blocks on the held stall.
        setTimeout(() => reject(new Error("close waited for stalled MCP setup")), 2_000);
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

for (const writeTextFile of [false, true]) {
  test(`cancelling a pending editor read settles the prompt before the RPC (${writeTextFile ? "read+write" : "read-only"})`, async () => {
    const cwd = realpathSync(await mkdtemp(join(tmpdir(), "glm-acp-abort-pending-read-")));
    await writeFile(join(cwd, "buffer.txt"), "on disk", "utf8");
    const held = deferred<{ content: string }>();
    const started = deferred<void>();
    let readCalls = 0;
    let modelCalls = 0;
    const conn = {
      ...connection(),
      readTextFile() {
        readCalls += 1;
        started.resolve();
        return held.promise;
      },
    };
    const glm = {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        modelCalls += 1;
        yield { toolCall: { id: "pending-read", name: "read_file", arguments: JSON.stringify({ path: "buffer.txt" }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    };
    const agent = new GlmAcpAgent(conn as never, { glm, sessionStore: null });
    await agent.initialize({ protocolVersion: 1, clientCapabilities: { fs: { readTextFile: true, writeTextFile } } } as never);
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read buffer" }] });
    try {
      await withoutNotificationRelease(started.promise);
      await agent.cancel({ sessionId });
      assert.equal((await withoutNotificationRelease(prompt)).stopReason, "cancelled");
      assert.equal(readCalls, 1);
      assert.equal(modelCalls, 1);
      assert.equal(conn.permissionRequests.length, 0);
      assert.ok(!conn.updates.some(update => (update.update as { status?: string }).status === "completed"));
      const beforeLateSettlement = JSON.stringify(conn.updates);
      held.resolve({ content: "late cancelled buffer contents" });
      await new Promise<void>(resolve => setImmediate(resolve));
      assert.equal(JSON.stringify(conn.updates), beforeLateSettlement);
      assert.ok(!beforeLateSettlement.includes("late cancelled buffer contents"));
    } finally {
      await agent.cancel({ sessionId });
      held.resolve({ content: "teardown" });
      await withoutNotificationRelease(Promise.allSettled([prompt]));
      await withoutNotificationRelease(agent.closeSession({ sessionId }));
      await rm(cwd, { recursive: true, force: true });
    }
  });
}

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
        message.content === "Error reading file: The operation was aborted"
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

test("fork waits for an active tool and clones a complete assistant/tool batch", async () => {
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
      return { content: "forked contents" };
    },
    async writeTextFile() {},
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-tool-history-"));
  const store = new SessionStore(storeRoot);
  let connections = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      yield { toolCall: { id: "fork-read", name: "read_file", arguments: JSON.stringify({ path: "a.txt" }) } };
      yield { done: true, stopReason: "tool_calls" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, {
    glm,
    sessionStore: store,
    connectSessionMcpServers: async () => {
      connections += 1;
      return new SessionMcpTools([]);
    },
  });
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  } as never);
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read" }] });
    await readReady;
    const fork = agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(connections, 1, "child resources must not be created before the parent drains");
    releaseRead();
    assert.equal((await prompt).stopReason, "cancelled");
    const forked = await fork;
    const persisted = store.load(forked.sessionId);
    const assistant = persisted?.messages.find(
      (message) => message.role === "assistant" && message.tool_calls?.some((call) => call.id === "fork-read")
    );
    const result = persisted?.messages.find(
      (message) => message.role === "tool" && message.tool_call_id === "fork-read"
    );
    assert.ok(assistant);
    assert.equal(result?.content, "Error reading file: The operation was aborted");
    assert.equal(connections, 2);
    assert.equal(updates.filter((update) => {
      const body = update.update as { sessionUpdate?: string; toolCallId?: string };
      return body.sessionUpdate === "tool_call_update" && body.toolCallId === "fork-read";
    }).length, 0);
  } finally {
    releaseRead();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("fork checkpoints a drained second tool turn so its parent survives restart", async () => {
  let readStarted!: () => void;
  const readReady = new Promise<void>((resolve) => { readStarted = resolve; });
  let releaseRead!: () => void;
  const readDone = new Promise<void>((resolve) => { releaseRead = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate() {},
    async readTextFile() {
      readStarted();
      await readDone;
      return { content: "second turn contents" };
    },
    async writeTextFile() {},
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-parent-restart-"));
  const store = new SessionStore(storeRoot);
  let calls = 0;
  const agent = new GlmAcpAgent(conn as never, {
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: "first turn" };
        } else {
          yield { toolCall: { id: "second-read", name: "read_file", arguments: JSON.stringify({ path: "second.txt" }) } };
        }
        yield { done: true, stopReason: calls === 1 ? "stop" : "tool_calls" };
      },
    },
    sessionStore: store,
    connectSessionMcpServers: async () => new SessionMcpTools([]),
  });
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  } as never);
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "first" }] });
    const second = agent.prompt({ sessionId, prompt: [{ type: "text", text: "second" }] });
    await readReady;
    const fork = agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    releaseRead();
    assert.equal((await second).stopReason, "cancelled");
    await fork;

    const persisted = store.load(sessionId);
    assert.ok(persisted?.messages.some(
      (message) => message.role === "tool" && message.tool_call_id === "second-read" &&
        message.content === "Error reading file: The operation was aborted"
    ));

    const restarted = new GlmAcpAgent(connection() as never, {
      sessionStore: store,
      connectSessionMcpServers: async () => new SessionMcpTools([]),
    });
    await restarted.loadSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await restarted.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
  } finally {
    releaseRead();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("fork checkpoints a settled parent after configuration persists during drain", async () => {
  let readStarted!: () => void;
  const readReady = new Promise<void>((resolve) => { readStarted = resolve; });
  let releaseRead!: () => void;
  const readDone = new Promise<void>((resolve) => { releaseRead = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate() {},
    async readTextFile() {
      readStarted();
      await readDone;
      return { content: "racing contents" };
    },
    async writeTextFile() {},
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-config-race-"));
  const store = new SessionStore(storeRoot);
  const agent = new GlmAcpAgent(conn as never, {
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield { toolCall: { id: "racing-read", name: "read_file", arguments: JSON.stringify({ path: "race.txt" }) } };
        yield { done: true, stopReason: "tool_calls" };
      },
    },
    sessionStore: store,
    connectSessionMcpServers: async () => new SessionMcpTools([]),
  });
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  } as never);
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "read" }] });
    await readReady;
    const fork = agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await new Promise((resolve) => setImmediate(resolve));
    await agent.setSessionMode({ sessionId, modeId: "bypass_permissions" });
    releaseRead();
    assert.equal((await prompt).stopReason, "cancelled");
    await fork;

    const persisted = store.load(sessionId);
    assert.equal(persisted?.mode, "bypass_permissions");
    assert.ok(persisted?.messages.some(
      (message) => message.role === "tool" && message.tool_call_id === "racing-read" &&
        message.content === "Error reading file: The operation was aborted"
    ));

    const restarted = new GlmAcpAgent(connection() as never, {
      sessionStore: store,
      connectSessionMcpServers: async () => new SessionMcpTools([]),
    });
    await restarted.loadSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await restarted.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
  } finally {
    releaseRead();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a timed-out fork creates no child resources and reopens after the prompt drains", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-timeout-"));
  const store = new SessionStore(storeRoot);
  let started!: () => void;
  const ready = new Promise<void>((resolve) => { started = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  let connections = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      calls += 1;
      if (calls === 1) {
        started();
        await blocked;
      }
      yield { text: "ok" };
      yield { done: true, stopReason: "stop" };
    },
  };
  const agent = new GlmAcpAgent(conn as never, {
    glm,
    sessionStore: store,
    sessionDrainTimeoutMs: 0,
    connectSessionMcpServers: async () => {
      connections += 1;
      return new SessionMcpTools([]);
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "blocked" }] });
    await ready;
    await assert.rejects(
      agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
      /fork timed out/i,
    );
    assert.equal(connections, 1);
    release();
    await prompt;
    await new Promise((resolve) => setImmediate(resolve));
    const persisted = store.load(sessionId);
    assert.ok(
      persisted?.messages.some((message) => message.role === "user" && message.content === "blocked"),
      "the deferred fork rollback must checkpoint the drained turn before another prompt",
    );
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "again" }] })).stopReason, "end_turn");
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close cancels stalled fork MCP setup and disposes its late result once", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-fork-setup-"));
  const store = new SessionStore(storeRoot);
  let setupStarted!: () => void;
  const setupReady = new Promise<void>((resolve) => { setupStarted = resolve; });
  let releaseLateResult!: () => void;
  const lateResult = new Promise<void>((resolve) => { releaseLateResult = resolve; });
  let setupSignal: AbortSignal | undefined;
  let connections = 0;
  let disposed = 0;
  const replacement = new SessionMcpTools([]);
  replacement.dispose = async () => { disposed += 1; };
  const agent = new GlmAcpAgent(conn as never, {
    glm: textGlm(),
    sessionStore: store,
    connectSessionMcpServers: async (_servers, signal) => {
      connections += 1;
      if (connections === 1) return new SessionMcpTools([]);
      setupSignal = signal;
      setupStarted();
      await lateResult;
      return replacement;
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  let fork: Promise<unknown> | undefined;
  let forkRejected: Promise<void> | undefined;
  let closing: Promise<void> | undefined;
  try {
    fork = agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    forkRejected = assert.rejects(fork, /fork cancelled/i);
    await setupReady;
    closing = agent.closeSession({ sessionId });
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(setupSignal?.aborted, true, "close must abort the fork connector");
    await forkRejected;
    const watchdog = new Promise<never>((_resolve, reject) => {
      // Generous bound: the raced close resolves through a real fsync'd
      // session write, which can exceed 100 ms on slow CI runners. The
      // property under test is that close never blocks on the held stall.
      setTimeout(() => reject(new Error("close waited for stalled fork MCP setup")), 2_000).unref();
    });
    await Promise.race([closing, watchdog]);
    assert.equal(disposed, 0, "the late setup result is not available yet");
    releaseLateResult();
    await new Promise<void>((resolve) => setImmediate(resolve));
    assert.equal(disposed, 1, "a late setup result is disposed exactly once");
  } finally {
    releaseLateResult();
    await Promise.allSettled([fork, closing].filter((value): value is Promise<unknown> => value !== undefined));
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("an unloaded fork cannot race a restore of the same persisted session", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-unloaded-race-"));
  const store = new SessionStore(storeRoot);
  let restoreSetupStarted!: () => void;
  const restoreSetupReady = new Promise<void>((resolve) => { restoreSetupStarted = resolve; });
  let releaseRestoreSetup!: () => void;
  const restoreSetupDone = new Promise<void>((resolve) => { releaseRestoreSetup = resolve; });
  let connections = 0;
  const agent = new GlmAcpAgent(conn as never, {
    glm: textGlm(),
    sessionStore: store,
    connectSessionMcpServers: async () => {
      connections += 1;
      if (connections === 2) {
        restoreSetupStarted();
        await restoreSetupDone;
      }
      return new SessionMcpTools([]);
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    await agent.closeSession({ sessionId });
    const resume = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] });
    await restoreSetupReady;
    await assert.rejects(
      agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
      /transition in progress/i,
    );
    releaseRestoreSetup();
    await resume;
  } finally {
    releaseRestoreSetup();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("fork rejects persisted history with an unmatched assistant tool call", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-fork-invalid-history-"));
  const store = new SessionStore(storeRoot);
  const sessionId = "33333333-3333-4333-8333-333333333333";
  await store.save({
    sessionId,
    cwd: tmpdir(),
    messages: [
      { role: "system", content: "rules" },
      { role: "user", content: "read" },
      { role: "assistant", content: null, tool_calls: [
        { id: "missing-result", type: "function", function: { name: "read_file", arguments: "{}" } },
      ] },
    ],
    title: null,
    updatedAt: new Date().toISOString(),
    model: "glm-4.7",
    mode: "default",
  });
  const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
  try {
    await assert.rejects(
      agent.unstable_forkSession({ sessionId, cwd: tmpdir(), mcpServers: [] }),
      /missing tool result.*missing-result/i,
    );
  } finally {
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

test("a restore timeout checkpoints the settled original after the prompt unwinds", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-timeout-checkpoint-"));
  const store = new SessionStore(storeRoot);
  let partialReceived!: () => void;
  const partialReady = new Promise<void>((resolve) => { partialReceived = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    sessionDrainTimeoutMs: 5,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield { text: "retained before timeout" };
        partialReceived();
        await blocked;
        yield { text: "late" };
        yield { done: true, stopReason: "stop" };
      },
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
    await partialReady;
    await assert.rejects(agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [] }), /timed out/i);
    release();
    await prompt;
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(store.load(sessionId)?.messages.some(
      (message) => message.role === "assistant" && message.content === "retained before timeout",
    ));
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a failed restore checkpoints drained history before leaving the original usable", async () => {
  const conn = connection();
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-restore-failure-checkpoint-"));
  const store = new SessionStore(storeRoot);
  let partialReceived!: () => void;
  const partialReady = new Promise<void>((resolve) => { partialReceived = resolve; });
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let calls = 0;
  const agent = new GlmAcpAgent(conn as never, {
    sessionStore: store,
    glm: {
      async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        calls += 1;
        if (calls === 1) {
          yield { text: "retained before setup failure" };
          partialReceived();
          await blocked;
        }
        yield { text: "ok" };
        yield { done: true, stopReason: "stop" };
      },
    },
    connectSessionMcpServers: async (servers) => {
      if (servers.length > 0) throw new Error("setup failed");
      return new SessionMcpTools([]);
    },
  });
  const { sessionId } = await agent.newSession({ cwd: tmpdir(), mcpServers: [] });
  try {
    const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "start" }] });
    await partialReady;
    const restore = agent.resumeSession({ sessionId, cwd: tmpdir(), mcpServers: [{ type: "http", name: "broken", url: "https://mcp.example.test", headers: [] }] });
    release();
    await prompt;
    await assert.rejects(restore, /setup failed/i);
    assert.ok(store.load(sessionId)?.messages.some(
      (message) => message.role === "assistant" && message.content === "retained before setup failure",
    ));
    assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "after" }] })).stopReason, "end_turn");
  } finally {
    release();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("close interrupts a stalled unloaded restore replay", async () => {
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-close-replay-"));
  const store = new SessionStore(storeRoot);
  const sessionId = "66666666-6666-6666-6666-666666666666";
  await store.save({
    sessionId,
    cwd: "/tmp",
    messages: [
      { role: "system", content: "system" },
      { role: "user", content: "replay me" },
    ],
    title: null,
    updatedAt: "2026-01-01T00:00:00.000Z",
    model: "glm-5.3",
    mode: "default",
  });
  let replayStarted!: () => void;
  const replayReady = new Promise<void>((resolve) => { replayStarted = resolve; });
  let releaseReplay!: () => void;
  const replayDone = new Promise<void>((resolve) => { releaseReplay = resolve; });
  const conn = {
    signal: new AbortController().signal,
    async sessionUpdate() {
      replayStarted();
      await replayDone;
    },
  };
  const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
  try {
    const load = agent.loadSession({ sessionId, cwd: "/tmp", mcpServers: [] });
    await replayReady;
    const close = agent.closeSession({ sessionId });
    await Promise.race([
      close,
      // Generous bound: the raced close resolves through a real fsync'd
      // session write, which can exceed 100 ms on slow CI runners. The
      // property under test is that close never blocks on the held replay.
      new Promise<never>((_resolve, reject) => setTimeout(() => reject(new Error("close waited for replay")), 2_000)),
    ]);
    await assert.rejects(load, /cancelled/i);
  } finally {
    releaseReplay();
    await rm(storeRoot, { recursive: true, force: true });
  }
});

test("a failed unloaded restore removes its transition record", async () => {
  const storeRoot = await mkdtemp(join(tmpdir(), "glm-acp-unloaded-failure-record-"));
  const store = new SessionStore(storeRoot);
  const sessionId = "77777777-7777-7777-7777-777777777777";
  await store.save({
    sessionId,
    cwd: "/tmp",
    messages: [{ role: "system", content: "system" }],
    title: null,
    updatedAt: "2026-01-01T00:00:00.000Z",
    model: "glm-5.3",
    mode: "default",
  });
  const agent = new GlmAcpAgent(connection() as never, {
    sessionStore: store,
    connectSessionMcpServers: async () => { throw new Error("setup failed"); },
  });
  try {
    await assert.rejects(agent.resumeSession({ sessionId, cwd: "/tmp", mcpServers: [] }), /setup failed/i);
    assert.equal((agent as unknown as { transitions: Map<string, unknown> }).transitions.size, 0);
  } finally {
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
    const timestampBeforeChange = store.load(sessionId)!.updatedAt;
    while (Date.now() <= Date.parse(timestampBeforeChange)) {
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
    await agent.setSessionMode({ sessionId, modeId: "bypass_permissions" });
    const timestampAfterChange = store.load(sessionId)!.updatedAt;
    assert.notEqual(timestampAfterChange, timestampBeforeChange);
    releaseReplay();
    const loaded = await loading;
    assert.equal(loaded.modes?.currentModeId, "bypass_permissions");
    assert.equal(store.load(sessionId)?.mode, "bypass_permissions");
    assert.equal(store.load(sessionId)?.updatedAt, timestampAfterChange);
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
