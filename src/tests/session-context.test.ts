import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PROTOCOL_VERSION } from "@agentclientprotocol/sdk";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import type { GlmAcpAgentOptions } from "../protocol/agent.js";
import type { GlmMessage, GlmStreamChunk } from "../llm/glm-client.js";

interface ConnectionStub {
  updates: Array<Record<string, unknown>>;
  sessionUpdate: (params: Record<string, unknown>) => Promise<void>;
}

function createConnectionStub(): ConnectionStub {
  return {
    updates: [],
    async sessionUpdate(params) {
      this.updates.push(params);
    },
  };
}

function makeStreamingGlm(captured: {
  messages: GlmMessage[];
  tools: string[];
}): NonNullable<GlmAcpAgentOptions["glm"]> {
  return {
    async *streamChat(
      messages: GlmMessage[],
      _signal?: AbortSignal,
      options?: { tools?: Array<{ function: { name: string } }> }
    ): AsyncGenerator<GlmStreamChunk> {
      captured.messages = messages.map((message) => structuredClone(message));
      captured.tools = options?.tools?.map((tool) => tool.function.name) ?? [];
      yield { text: "ack" };
      yield { done: true, stopReason: "stop" };
    },
  };
}

function workspace(label: string, command = false): { cwd: string; cleanup: () => void } {
  const cwd = mkdtempSync(join(tmpdir(), `glm-context-${label}-`));
  writeFileSync(join(cwd, "AGENTS.md"), `${label} project rules`);
  if (command) {
    mkdirSync(join(cwd, ".claude", "commands"), { recursive: true });
    writeFileSync(
      join(cwd, ".claude", "commands", "deploy.md"),
      "---\ndescription: deploy\n---\nRun the deploy playbook.\n"
    );
  }
  return { cwd, cleanup: () => rmSync(cwd, { recursive: true, force: true }) };
}

function mcpServer(label: string) {
  return {
    type: "http" as const,
    name: `${label}-mcp`,
    url: `https://mcp.example.test/${label}`,
    headers: [],
  };
}

function installMcpFetch(): () => void {
  const previous = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL | Request, init?: RequestInit) => {
    const body = typeof init?.body === "string" ? (JSON.parse(init.body) as { method?: string }) : {};
    const label = String(url).split("/").at(-1) ?? "unknown";
    if (body.method === "initialize") {
      return new Response(
        JSON.stringify({ jsonrpc: "2.0", id: 1, result: { protocolVersion: "2025-06-18" } }),
        { status: 200, headers: { "Content-Type": "application/json", "MCP-Session-Id": `${label}-session` } }
      );
    }
    if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
    if (body.method === "tools/list") {
      return new Response(
        JSON.stringify({
          jsonrpc: "2.0",
          id: 2,
          result: {
            tools: [{ name: `${label}_tool`, description: `${label} connected tool`, inputSchema: { type: "object" } }],
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }
    throw new Error(`unexpected MCP request: ${String(body.method)}`);
  }) as typeof fetch;
  return () => {
    globalThis.fetch = previous;
  };
}

async function seedSession(
  store: SessionStore,
  cwd: string,
  captured: { messages: GlmMessage[]; tools: string[] }
): Promise<{ agent: GlmAcpAgent; sessionId: string }> {
  const agent = new GlmAcpAgent(createConnectionStub() as never, {
    glm: makeStreamingGlm(captured),
    sessionStore: store,
  });
  await agent.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [mcpServer("alpha")] });
  await agent.unstable_setSessionModel({ sessionId, modelId: "glm-5.3" });
  await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
  await agent.setSessionConfigOption({ sessionId, configId: "thought_level", value: "low" });
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/deploy staging" }] });
  return { agent, sessionId };
}

function systemContent(messages: GlmMessage[]): string {
  const system = messages.find((message) => message.role === "system");
  return typeof system?.content === "string" ? system.content : "";
}

function replayedUserTexts(conn: ConnectionStub): string[] {
  return conn.updates
    .filter((item) => (item.update as { sessionUpdate?: string }).sessionUpdate === "user_message_chunk")
    .map((item) => (item.update as { content: { text: string } }).content.text);
}

for (const restoreKind of ["load", "resume"] as const) {
  test(`${restoreKind} rebuilds system context from B while preserving A history and state`, async () => {
    const storeDir = mkdtempSync(join(tmpdir(), "glm-context-store-"));
    const store = new SessionStore(storeDir);
    const a = workspace("alpha", true);
    const b = workspace("beta");
    const captured = { messages: [] as GlmMessage[], tools: [] as string[] };
    const restoreFetch = installMcpFetch();
    try {
      const seeded = await seedSession(store, a.cwd, captured);
      const persistedBefore = store.load(seeded.sessionId);
      assert.ok(persistedBefore);
      assert.match(systemContent(persistedBefore.messages), /alpha project rules/);
      assert.deepEqual(persistedBefore.displayText, { "1": "/deploy staging" });

      const conn = createConnectionStub();
      const restored = new GlmAcpAgent(conn as never, {
        glm: makeStreamingGlm(captured),
        sessionStore: store,
      });
      const restoreParams = {
        sessionId: seeded.sessionId,
        cwd: b.cwd,
        mcpServers: [mcpServer("beta")],
      };
      const result = restoreKind === "load"
        ? await restored.loadSession(restoreParams)
        : await restored.resumeSession(restoreParams);

      assert.equal(result.models?.currentModelId, "glm-5.3");
      assert.equal(result.modes?.currentModeId, "accept_edits");
      assert.equal(
        result.configOptions?.find((option) => option.id === "thought_level")?.currentValue,
        "low"
      );
      if (restoreKind === "load") assert.deepEqual(replayedUserTexts(conn), ["/deploy staging"]);

      await restored.prompt({ sessionId: seeded.sessionId, prompt: [{ type: "text", text: "continue" }] });
      const prompt = systemContent(captured.messages);
      assert.match(prompt, new RegExp(`Working directory: ${b.cwd.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
      assert.match(prompt, /beta project rules/);
      assert.match(prompt, /beta_tool/);
      assert.doesNotMatch(prompt, /alpha project rules/);
      assert.doesNotMatch(prompt, /alpha_tool/);
      const restoredUser = captured.messages.find((message) => message.role === "user");
      assert.match(String(restoredUser?.content), /<slash_command name="deploy"/);
      assert.match(String(restoredUser?.content), /Run the deploy playbook/);
      assert.ok(captured.messages.some((message) => message.role === "assistant" && message.content === "ack"));
      assert.ok(captured.tools.includes("beta_tool"));
      assert.ok(!captured.tools.includes("alpha_tool"));
    } finally {
      restoreFetch();
      a.cleanup();
      b.cleanup();
      rmSync(storeDir, { recursive: true, force: true });
    }
  });
}

test("load shifts display mapping when an old record lacks a leading system message", async () => {
  const storeDir = mkdtempSync(join(tmpdir(), "glm-context-store-"));
  const store = new SessionStore(storeDir);
  const b = workspace("beta");
  const sessionId = "33333333-3333-3333-3333-333333333333";
  try {
    await store.save({
      sessionId,
      cwd: "/old/project",
      messages: [{ role: "user", content: "expanded command body" }],
      displayText: { "0": "/deploy staging" },
      title: "deploy",
      updatedAt: "2026-01-01T00:00:00.000Z",
      model: "glm-5.3",
      mode: "default",
    });
    const conn = createConnectionStub();
    const agent = new GlmAcpAgent(conn as never, { sessionStore: store });
    await agent.loadSession({ sessionId, cwd: b.cwd, mcpServers: [] });
    assert.deepEqual(replayedUserTexts(conn), ["/deploy staging"]);
  } finally {
    b.cleanup();
    rmSync(storeDir, { recursive: true, force: true });
  }
});

test("fork refreshes B context without mutating the A parent or its display mapping", async () => {
  const storeDir = mkdtempSync(join(tmpdir(), "glm-context-store-"));
  const store = new SessionStore(storeDir);
  const a = workspace("alpha", true);
  const b = workspace("beta");
  const captured = { messages: [] as GlmMessage[], tools: [] as string[] };
  const restoreFetch = installMcpFetch();
  try {
    const seeded = await seedSession(store, a.cwd, captured);
    const parentBefore = store.load(seeded.sessionId);
    assert.ok(parentBefore);

    const fork = await seeded.agent.unstable_forkSession({
      sessionId: seeded.sessionId,
      cwd: b.cwd,
      mcpServers: [mcpServer("beta")],
    });
    const forked = store.load(fork.sessionId);
    assert.ok(forked);
    assert.match(systemContent(forked.messages), /beta project rules/);
    assert.match(systemContent(forked.messages), /beta_tool/);
    assert.deepEqual(forked.displayText, { "1": "/deploy staging" });
    assert.equal(fork.models?.currentModelId, "glm-5.3");
    assert.equal(fork.modes?.currentModeId, "accept_edits");
    assert.equal(
      fork.configOptions?.find((option) => option.id === "thought_level")?.currentValue,
      "low"
    );

    const parentAfter = store.load(seeded.sessionId);
    assert.equal(systemContent(parentAfter?.messages ?? []), systemContent(parentBefore.messages));
    assert.deepEqual(parentAfter?.displayText, { "1": "/deploy staging" });

    const conn = createConnectionStub();
    const reopened = new GlmAcpAgent(conn as never, {
      glm: makeStreamingGlm(captured),
      sessionStore: store,
    });
    await reopened.loadSession({ sessionId: fork.sessionId, cwd: b.cwd, mcpServers: [mcpServer("beta")] });
    assert.deepEqual(replayedUserTexts(conn), ["/deploy staging"]);
  } finally {
    restoreFetch();
    a.cleanup();
    b.cleanup();
    rmSync(storeDir, { recursive: true, force: true });
  }
});
