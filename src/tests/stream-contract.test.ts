import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmClient, type GlmStreamChunk } from "../llm/glm-client.js";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";

async function withProvider(
  frames: unknown[], run: (client: GlmClient, requests: unknown[]) => Promise<void>,
): Promise<void> {
  const requests: unknown[] = [];
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push(JSON.parse(body));
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    for (const frame of frames) res.write(`data: ${JSON.stringify(frame)}\n\n`);
    res.end();
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedUrl = process.env["ACP_GLM_BASE_URL"];
  process.env["Z_AI_API_KEY"] = "fixture-key";
  process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v4`;
  try {
    await run(new GlmClient(), requests);
  } finally {
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedUrl === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedUrl;
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
  }
}

function delta(content: Record<string, unknown>, reason: string | null = null) {
  return { choices: [{ index: 0, delta: content, finish_reason: reason }] };
}

for (const frames of [[], [delta({ content: "partial answer" })]]) {
  test(`HTTP EOF without terminal reason rejects (${frames.length} frames)`, async () => {
    await withProvider(frames, async client => {
      await assert.rejects(async () => {
        for await (const chunk of client.streamChat([{ role: "user", content: "hello" }])) {
          assert.equal(chunk.toolCall, undefined);
          assert.notEqual(chunk.done, true);
        }
      }, /incomplete|terminal/i);
    });
  });
}

for (const reason of ["length", "content_filter"]) {
  test(`HTTP ${reason} never executes an assembled write call`, async () => {
    const cwd = mkdtempSync(join(tmpdir(), "glm-stream-contract-"));
    const path = join(cwd, "keep.txt");
    writeFileSync(path, "original");
    try {
      await withProvider([delta({ content: "partial", tool_calls: [{ index: 0, id: "write-1", type: "function",
        function: { name: "write_file", arguments: JSON.stringify({ path, content: "changed" }) },
      }] }, reason)], async (client, requests) => {
        const updates: unknown[] = [];
        const agent = new GlmAcpAgent({ sessionUpdate: async (update: unknown) => { updates.push(update); } } as never,
          { glm: client, sessionStore: null, maxTurns: 1, visionClient: null });
        const session = await agent.newSession({ cwd, mcpServers: [] });
        try {
          await agent.setSessionMode({ sessionId: session.sessionId, modeId: "accept_edits" });
          const result = await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "write" }] });
          assert.equal(result.stopReason, reason === "length" ? "max_tokens" : "refusal");
          assert.equal(readFileSync(path, "utf8"), "original");
          assert.equal(requests.length, 1);
          assert.ok(updates.length > 0);
        } finally {
          await agent.closeSession({ sessionId: session.sessionId });
        }
      });
    } finally {
      rmSync(cwd, { recursive: true, force: true });
    }
  });
}

for (const calls of [
  [{ index: 0, function: { arguments: "{}" } }],
  [0, 1].map(index => ({ index, id: "duplicate", function: { name: "read_file", arguments: "{}" } })),
]) {
  test(`HTTP tool_calls rejects an invalid ${calls.length}-call batch`, async () => {
    await withProvider([delta({ tool_calls: calls }, "tool_calls")], async client => {
      await assert.rejects(async () => {
        for await (const chunk of client.streamChat([])) assert.equal(chunk.toolCall, undefined);
      }, /tool|batch|incomplete/i);
    });
  });
}

test("HTTP complete tool batch permits repeated function names and retains trailing usage", async () => {
  await withProvider([
    delta({ tool_calls: [0, 1].map(index => ({ index, id: `read-${index}`, function: { name: "read_file", arguments: "{}" } })) }, "tool_calls"),
    { choices: [], usage: { prompt_tokens: 4, completion_tokens: 2, total_tokens: 6 } },
  ], async client => {
    const chunks: GlmStreamChunk[] = [];
    for await (const chunk of client.streamChat([])) chunks.push(chunk);
    assert.deepEqual(chunks.flatMap(chunk => chunk.toolCall ? [chunk.toolCall.id] : []), ["read-0", "read-1"]);
    assert.equal(chunks.find(chunk => chunk.usage)?.usage?.totalTokens, 6);
    assert.equal(chunks.at(-1)?.stopReason, "tool_calls");
  });
});

test("HTTP stream rejects a choice after a terminal tool batch", async () => {
  await withProvider([
    delta({ tool_calls: [{ index: 0, id: "read-1", type: "function",
      function: { name: "read_file", arguments: JSON.stringify({ path: "/tmp/keep.txt" }) },
    }] }, "tool_calls"),
    delta({ tool_calls: [{ index: 1, id: "write-1", type: "function",
      function: { name: "write_file", arguments: JSON.stringify({ path: "/tmp/keep.txt", content: "changed" }) },
    }] }),
  ], async client => {
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) void chunk;
    }, /after terminal|incomplete/i);
  });
});

test("HTTP stream rejects a multi-choice frame with a nonzero terminal reason", async () => {
  await withProvider([
    { choices: [
      { index: 0, delta: {}, finish_reason: null },
      { index: 1, delta: { tool_calls: [{ index: 0, id: "read-1", type: "function",
        function: { name: "read_file", arguments: JSON.stringify({ path: "/tmp/keep.txt" }) },
      }] }, finish_reason: "tool_calls" },
    ] },
    delta({ tool_calls: [{ index: 0, id: "write-1", type: "function",
      function: { name: "write_file", arguments: JSON.stringify({ path: "/tmp/keep.txt", content: "changed" }) },
    }] }, "tool_calls"),
  ], async client => {
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) void chunk;
    }, /multiple model choices/i);
  });
});

test("HTTP post-terminal write delta never changes a file in accept_edits mode", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-post-terminal-stream-"));
  const path = join(cwd, "keep.txt");
  writeFileSync(path, "original");
  try {
    await withProvider([
      delta({ tool_calls: [{ index: 0, id: "read-1", type: "function",
        function: { name: "read_file", arguments: JSON.stringify({ path }) },
      }] }, "tool_calls"),
      delta({ tool_calls: [{ index: 1, id: "write-1", type: "function",
        function: { name: "write_file", arguments: JSON.stringify({ path, content: "changed" }) },
      }] }),
    ], async client => {
      const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never,
        { glm: client, sessionStore: null, maxTurns: 1, visionClient: null });
      const session = await agent.newSession({ cwd, mcpServers: [] });
      try {
        await agent.setSessionMode({ sessionId: session.sessionId, modeId: "accept_edits" });
        let promptError: unknown;
        try {
          await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "write" }] });
        } catch (err) {
          promptError = err;
        }
        assert.equal(readFileSync(path, "utf8"), "original");
        assert.match(String(promptError), /after terminal|incomplete/i);
      } finally {
        await agent.closeSession({ sessionId: session.sessionId });
      }
    });
  } finally {
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("incomplete HTTP response preserves partial text for close and restore", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-partial-history-"));
  const store = new SessionStore(join(cwd, "sessions"));
  try {
    await withProvider([delta({ content: "received before EOF" })], async client => {
      const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never,
        { glm: client, sessionStore: store, visionClient: null });
      const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
      await assert.rejects(agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] }), /incomplete/i);
      await agent.closeSession({ sessionId });
      assert.ok(store.load(sessionId)?.messages.some(message => message.role === "assistant" && message.content === "received before EOF"));
    });
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("agent validates injected provider completion before executing its calls", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-injected-stream-"));
  const path = join(cwd, "keep.txt");
  writeFileSync(path, "original");
  const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
    sessionStore: null, visionClient: null, maxTurns: 1,
    glm: { async *streamChat() {
      yield { toolCall: { id: "write", name: "write_file", arguments: JSON.stringify({ path, content: "changed" }) } };
      yield { done: true, stopReason: "length" };
    } },
  });
  const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
  try {
    await agent.setSessionMode({ sessionId, modeId: "accept_edits" });
    const result = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "write" }] });
    assert.equal(readFileSync(path, "utf8"), "original");
    assert.equal(result.stopReason, "max_tokens");
  } finally {
    await agent.closeSession({ sessionId });
    rmSync(cwd, { recursive: true, force: true });
  }
});
