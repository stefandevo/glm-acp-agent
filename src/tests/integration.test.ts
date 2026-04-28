import test from "node:test";
import assert from "node:assert/strict";
import {
  AgentSideConnection,
  ClientSideConnection,
  ndJsonStream,
  PROTOCOL_VERSION,
  type Client,
} from "@agentclientprotocol/sdk";
import { GlmAcpAgent } from "../protocol/agent.js";
import type { GlmStreamChunk } from "../llm/glm-client.js";

/**
 * End-to-end style test: wire two `TransformStream`s together so that the
 * AgentSideConnection (running our GlmAcpAgent) and a fake ClientSideConnection
 * exchange real JSON-RPC newline-delimited frames over in-memory streams.
 * This validates that the agent is wire-compatible with the official SDK.
 */
function pairedStreams() {
  const aToB = new TransformStream<Uint8Array, Uint8Array>();
  const bToA = new TransformStream<Uint8Array, Uint8Array>();
  return {
    a: ndJsonStream(aToB.writable, bToA.readable),
    b: ndJsonStream(bToA.writable, aToB.readable),
  };
}

class StubClient implements Client {
  updates: Array<Record<string, unknown>> = [];
  reads: Array<{ path: string }> = [];
  fileContents = new Map<string, string>();
  permissionResponses: Array<{ outcome: { outcome: "selected"; optionId: string } | { outcome: "cancelled" } }> = [];

  async sessionUpdate(params: Parameters<Client["sessionUpdate"]>[0]): Promise<void> {
    this.updates.push(params as unknown as Record<string, unknown>);
  }
  async requestPermission(): Promise<ReturnType<NonNullable<Client["requestPermission"]>>> {
    const next = this.permissionResponses.shift();
    return next ?? { outcome: { outcome: "selected", optionId: "allow" } };
  }
  async readTextFile(params: { sessionId: string; path: string }) {
    this.reads.push({ path: params.path });
    const content = this.fileContents.get(params.path);
    if (content === undefined) throw new Error(`file not found: ${params.path}`);
    return { content };
  }
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

test("end-to-end initialize / new session / prompt round-trip via real SDK transport", async () => {
  const { a, b } = pairedStreams();
  const stub = new StubClient();

  // Agent side
  const glm = makeStreamingGlm([
    [
      { thinking: "Let me think." },
      { text: "Hello!" },
      { done: true, stopReason: "stop" },
    ],
  ]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _agentConn = new AgentSideConnection(
    (conn) => new GlmAcpAgent(conn, { glm }),
    a
  );

  // Client side
  const clientConn = new ClientSideConnection(() => stub, b);

  const initResp = await clientConn.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: {
      fs: { readTextFile: true, writeTextFile: true },
      terminal: false,
    },
  });
  assert.equal(initResp.protocolVersion, PROTOCOL_VERSION);
  assert.equal(initResp.agentInfo?.name, "glm-acp-agent");

  const session = await clientConn.newSession({ cwd: "/tmp", mcpServers: [] });
  assert.ok(typeof session.sessionId === "string" && session.sessionId.length > 0);

  const prompt = await clientConn.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "Hi there" }],
  });

  assert.equal(prompt.stopReason, "end_turn");

  // Confirm the client received streaming updates including a thought chunk and a message chunk.
  const updateKinds = stub.updates.map(
    (u) => (u.update as { sessionUpdate: string }).sessionUpdate
  );
  assert.ok(updateKinds.includes("agent_thought_chunk"));
  assert.ok(updateKinds.includes("agent_message_chunk"));
  assert.ok(updateKinds.includes("session_info_update"));
});

test("end-to-end tool call: agent reads a file via the stub client", async () => {
  const { a, b } = pairedStreams();
  const stub = new StubClient();
  stub.fileContents.set("/tmp/x.ts", "export const x = 1;");

  let callIndex = 0;
  const glm = {
    async *streamChat(): AsyncGenerator<GlmStreamChunk> {
      callIndex++;
      if (callIndex === 1) {
        yield {
          toolCall: {
            id: "tc1",
            name: "read_file",
            arguments: JSON.stringify({ path: "/tmp/x.ts" }),
          },
        };
        yield { done: true, stopReason: "tool_calls" };
      } else {
        yield { text: "Read it." };
        yield { done: true, stopReason: "stop" };
      }
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _agentConn = new AgentSideConnection(
    (conn) => new GlmAcpAgent(conn, { glm }),
    a
  );
  const clientConn = new ClientSideConnection(() => stub, b);

  await clientConn.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: { fs: { readTextFile: true, writeTextFile: true } },
  });
  const session = await clientConn.newSession({ cwd: "/tmp", mcpServers: [] });
  const result = await clientConn.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "read it" }],
  });

  assert.equal(result.stopReason, "end_turn");
  assert.deepEqual(stub.reads, [{ path: "/tmp/x.ts" }]);

  // The client should have seen `tool_call` and `tool_call_update` notifications.
  const updateKinds = stub.updates.map(
    (u) => (u.update as { sessionUpdate: string }).sessionUpdate
  );
  assert.ok(updateKinds.includes("tool_call"));
  assert.ok(updateKinds.includes("tool_call_update"));
});

test("end-to-end cancellation via session/cancel notification", async () => {
  const { a, b } = pairedStreams();
  const stub = new StubClient();

  const glm = {
    async *streamChat(_messages: unknown, signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
      yield { text: "starting" };
      await new Promise((resolve) => setTimeout(resolve, 30));
      if (signal?.aborted) {
        yield { done: true, stopReason: "stop" };
        return;
      }
      yield { text: "more" };
      yield { done: true, stopReason: "stop" };
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _agentConn = new AgentSideConnection(
    (conn) => new GlmAcpAgent(conn, { glm }),
    a
  );
  const clientConn = new ClientSideConnection(() => stub, b);

  await clientConn.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
  const session = await clientConn.newSession({ cwd: "/tmp", mcpServers: [] });

  const promptPromise = clientConn.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "go" }],
  });
  await new Promise((resolve) => setTimeout(resolve, 5));
  await clientConn.cancel({ sessionId: session.sessionId });
  const result = await promptPromise;
  assert.equal(result.stopReason, "cancelled");
});

test("end-to-end session/list and session/close advertised on initialize and routable", async () => {
  const { a, b } = pairedStreams();
  const stub = new StubClient();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _agentConn = new AgentSideConnection(
    (conn) => new GlmAcpAgent(conn, { glm: makeStreamingGlm([]) }),
    a
  );
  const clientConn = new ClientSideConnection(() => stub, b);

  const initResp = await clientConn.initialize({
    protocolVersion: PROTOCOL_VERSION,
    clientCapabilities: {},
  });
  assert.ok(initResp.agentCapabilities?.sessionCapabilities?.list);
  assert.ok(initResp.agentCapabilities?.sessionCapabilities?.close);

  const a1 = await clientConn.newSession({ cwd: "/tmp/a", mcpServers: [] });
  await clientConn.newSession({ cwd: "/tmp/b", mcpServers: [] });

  const list = await clientConn.listSessions({});
  assert.equal(list.sessions.length, 2);

  await clientConn.closeSession({ sessionId: a1.sessionId });
  const list2 = await clientConn.listSessions({});
  assert.equal(list2.sessions.length, 1);
});
