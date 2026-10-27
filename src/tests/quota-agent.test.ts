import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AgentSideConnection, ClientSideConnection, ndJsonStream, PROTOCOL_VERSION } from "@agentclientprotocol/sdk";
import { GlmAcpAgent } from "../protocol/agent.js";
import type { GlmMessage, GlmStreamChunk } from "../llm/glm-client.js";
import type { QuotaSnapshot } from "../llm/quota-client.js";

const home = mkdtempSync(join(tmpdir(), "glm-quota-home-"));
process.env.HOME = home;
process.env.USERPROFILE = home;
test.after(() => rmSync(home, { recursive: true, force: true }));

const snapshot: QuotaSnapshot = {
  version: 1, checkedAt: "2026-10-24T12:00:00.000Z", stale: false,
  windows: [{ id: "five-hour", label: "5-hour model quota", kind: "session", usedPercent: 25, remaining: 1500, unit: "credits" }],
};

function setup(getUsage: () => Promise<QuotaSnapshot> = async () => snapshot) {
  const updates: Array<{ update: { sessionUpdate: string; content?: { text?: string }; availableCommands?: Array<{ name: string }> } }> = [];
  const modelCalls: GlmMessage[][] = [];
  const agent = new GlmAcpAgent({ async sessionUpdate(update: typeof updates[number]) { updates.push(update); } } as never, {
    quotaClient: { getUsage }, sessionStore: null, visionClient: null,
    glm: { async *streamChat(messages: GlmMessage[]): AsyncGenerator<GlmStreamChunk> {
      modelCalls.push(structuredClone(messages));
      yield { text: "hello" };
      yield { done: true, stopReason: "stop" };
    } },
  });
  return { agent, updates, modelCalls };
}

test("/usage is advertised once, reserves a conflicting file command, and never enters model history", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-quota-command-"));
  try {
    mkdirSync(join(cwd, ".claude", "commands"), { recursive: true });
    writeFileSync(join(cwd, ".claude", "commands", "usage.md"), "Send this to the model");
    let probes = 0;
    const { agent, updates, modelCalls } = setup(async () => { probes++; return snapshot; });
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    await new Promise(resolve => setTimeout(resolve, 5));
    const commands = updates.find(update => update.update.sessionUpdate === "available_commands_update")?.update.availableCommands;
    assert.equal(commands?.filter(command => command.name === "usage").length, 1);
    const result = await agent.prompt({ sessionId, messageId: "quota-request", prompt: [{ type: "text", text: " /usage\n" }] });
    assert.deepEqual(result, { stopReason: "end_turn", userMessageId: "quota-request" });
    assert.equal(probes, 1);
    assert.equal(modelCalls.length, 0);
    assert.match(updates.at(-1)?.update.content?.text ?? "", /1500 credits/);
    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });
    assert.equal(modelCalls.length, 1);
    assert.deepEqual(modelCalls[0]?.filter(message => message.role !== "system"), [{ role: "user", content: "hello" }]);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("/usage rejects arguments and attachments without model, vision, or quota calls", async () => {
  let probes = 0;
  const { agent, updates, modelCalls } = setup(async () => { probes++; return snapshot; });
  const { sessionId } = await agent.newSession({ cwd: home, mcpServers: [] });
  for (const prompt of [
    [{ type: "text" as const, text: "/usage refresh" }],
    [{ type: "text" as const, text: "/usage" }, { type: "text" as const, text: "hello" }],
    [{ type: "image" as const, data: "invalid", mimeType: "image/png" }, { type: "text" as const, text: "/usage" }],
  ]) {
    assert.equal((await agent.prompt({ sessionId, prompt })).stopReason, "end_turn");
    assert.match(updates.at(-1)?.update.content?.text ?? "", /without arguments or attachments/);
  }
  assert.equal(probes, 0);
  assert.equal(modelCalls.length, 0);
});

test("cancelling /usage releases the prompt while an account-wide lookup is pending", async () => {
  let release!: (value: QuotaSnapshot) => void;
  let started!: () => void;
  const ready = new Promise<void>(resolve => { started = resolve; });
  const { agent, updates } = setup(() => {
    started();
    return new Promise(resolve => { release = resolve; });
  });
  const { sessionId } = await agent.newSession({ cwd: home, mcpServers: [] });
  const pending = agent.prompt({ sessionId, messageId: "cancel-quota", prompt: [{ type: "text", text: "/usage" }] });
  await ready;
  await agent.cancel({ sessionId });
  assert.deepEqual(await pending, { stopReason: "cancelled", userMessageId: "cancel-quota" });
  release(snapshot);
  await Promise.resolve();
  assert.equal(updates.some(update => update.update.sessionUpdate === "agent_message_chunk"), false);
});

test("quota unavailability remains informational and leaves the next normal prompt usable", async () => {
  const { agent, updates, modelCalls } = setup(async () => ({
    version: 1, checkedAt: snapshot.checkedAt, stale: false, windows: [],
    unavailable: { reason: "requestFailed", message: "Quota lookup failed or timed out. Try again later." },
  }));
  const { sessionId } = await agent.newSession({ cwd: home, mcpServers: [] });
  assert.equal((await agent.prompt({ sessionId, prompt: [{ type: "text", text: "/usage" }] })).stopReason, "end_turn");
  assert.match(updates.at(-1)?.update.content?.text ?? "", /failed or timed out/);
  await agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });
  assert.equal(modelCalls.length, 1);
});

test("real ACP transport advertises and serves _glm/usage without a session or model call", async () => {
  const toAgent = new TransformStream<Uint8Array, Uint8Array>();
  const toClient = new TransformStream<Uint8Array, Uint8Array>();
  let probes = 0;
  const agentConnection = new AgentSideConnection(conn => new GlmAcpAgent(conn, {
    sessionStore: null, quotaClient: { async getUsage() { probes++; return snapshot; } },
  }), ndJsonStream(toClient.writable, toAgent.readable));
  const clientConnection = new ClientSideConnection(() => ({
    async sessionUpdate() {}, async requestPermission() { return { outcome: { outcome: "cancelled" as const } }; },
  }), ndJsonStream(toAgent.writable, toClient.readable));
  const init = await clientConnection.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  assert.deepEqual(init.agentCapabilities?._meta?.["glm-acp-agent/usage"], { version: 1, method: "_glm/usage" });
  assert.equal(probes, 0); // Discovery must not make quota requests.
  assert.deepEqual(await clientConnection.extMethod("_glm/usage", {}), snapshot);
  await assert.rejects(clientConnection.extMethod("_glm/unknown", {}), { code: -32601 });
  await assert.rejects(clientConnection.extMethod("_glm/usage", { token: "must not be accepted" }), { code: -32602 });
  await assert.rejects(clientConnection.extMethod("_glm/usage", null as never), { code: -32602 });
  await assert.rejects(clientConnection.extMethod("_glm/usage", [] as never), { code: -32602 });
  assert.equal(probes, 1);
  // Keep both ends referenced until the response round-trip completes.
  assert.equal(agentConnection.signal.aborted, false);
});
