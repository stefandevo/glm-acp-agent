import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmClient, type GlmMessage, type GlmStreamChunk } from "../llm/glm-client.js";
import { GlmAcpAgent } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";

for (const display of ["true", "false"]) {
  test(`reasoning survives HTTP tool round, save/load and fork with display=${display}`, async () => {
    const cwd = mkdtempSync(join(tmpdir(), "glm-reasoning-"));
    const source = join(cwd, "read.txt");
    writeFileSync(source, "file contents");
    const requests: Array<{ messages: Array<{ role: string; reasoning_content?: string }> }> = [];
    const reasoning = "  First\n思考🙂 second  ";
    const server = createServer(async (req, res) => {
      let body = "";
      for await (const chunk of req) body += chunk;
      requests.push(JSON.parse(body));
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      const frame = (delta: unknown, finish_reason: string | null = null) => res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason }] })}\n\n`);
      if (requests.length === 1) {
        frame({ reasoning_content: "  First\n" });
        frame({ reasoning_content: "思考🙂 second  " });
        frame({ tool_calls: [{ index: 0, id: "read-1", function: { name: "read_file", arguments: JSON.stringify({ path: source }) } }] }, "tool_calls");
      } else {
        frame({ reasoning_content: "Final reasoning" });
        frame({ content: "done" }, "stop");
      }
      res.end();
    });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const keys = ["Z_AI_API_KEY", "ACP_GLM_BASE_URL", "ACP_GLM_STREAM_THINKING"] as const;
    const before = keys.map(key => process.env[key]);
    process.env["Z_AI_API_KEY"] = "fixture-key";
    process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v4`;
    process.env["ACP_GLM_STREAM_THINKING"] = display;
    const updates: Array<{ update: { sessionUpdate: string } }> = [];
    const store = new SessionStore(join(cwd, "sessions"));
    const agent = new GlmAcpAgent({ sessionUpdate: async (update: { update: { sessionUpdate: string } }) => { updates.push(update); } } as never,
      { glm: new GlmClient(), sessionStore: store, visionClient: null });
    const { sessionId } = await agent.newSession({ cwd, mcpServers: [] });
    let forkId: string | undefined;
    try {
      await agent.prompt({ sessionId, prompt: [{ type: "text", text: "read" }] });
      assert.equal(requests[1]?.messages.find(message => message.role === "assistant")?.reasoning_content, reasoning);
      const stored = store.load(sessionId);
      assert.ok(stored);
      const assistants = stored.messages.filter(message => message.role === "assistant") as Array<{ reasoning_content?: string }>;
      assert.deepEqual(assistants.map(message => message.reasoning_content), [reasoning, "Final reasoning"]);
      await agent.closeSession({ sessionId });
      await agent.loadSession({ sessionId, cwd, mcpServers: [] });
      const fork = await agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] });
      forkId = fork.sessionId;
      await agent.prompt({ sessionId: fork.sessionId, prompt: [{ type: "text", text: "continue" }] });
      assert.equal(requests[2]?.messages.find(message => message.role === "assistant")?.reasoning_content, reasoning);
      assert.equal(updates.some(({ update }) => update.sessionUpdate === "agent_thought_chunk"), display === "true");
    } finally {
      await agent.closeSession({ sessionId });
      if (forkId) await agent.closeSession({ sessionId: forkId });
      keys.forEach((key, index) => { if (before[index] === undefined) delete process.env[key]; else process.env[key] = before[index]; });
      server.closeAllConnections();
      await new Promise<void>((resolve, reject) => server.close(err => err ? reject(err) : resolve()));
      rmSync(cwd, { recursive: true, force: true });
    }
  });
}

test("session store rejects non-string reasoning and preserves valid empty reasoning", async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-reasoning-shape-"));
  const store = new SessionStore(cwd);
  try {
    for (const value of [null, 42, {}, [], "", "unchanged\n思考"]) {
      const record = { sessionId: "shape", cwd, model: "glm-5.3", mode: "default" as const, title: null,
        updatedAt: new Date().toISOString(), messages: [
          { role: "assistant", content: "answer", reasoning_content: value } as GlmMessage,
        ] };
      if (typeof value === "string") await store.save(record);
      else {
        assert.throws(() => store.save(record), /invalid persisted/i);
        writeFileSync(join(cwd, "shape.json"), JSON.stringify(record));
      }
      assert.equal(store.load("shape") !== undefined, typeof value === "string");
    }
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

for (const ending of ["error", "cancel", "length"] as const) {
  test(`incomplete reasoning is not replayed after ${ending}`, async () => {
    const cwd = mkdtempSync(join(tmpdir(), "glm-reasoning-interrupted-"));
    const store = new SessionStore(join(cwd, "sessions"));
    let sessionId = "";
    const agent = new GlmAcpAgent({ sessionUpdate: async () => {} } as never, {
      sessionStore: store, visionClient: null,
      glm: { async *streamChat(): AsyncGenerator<GlmStreamChunk> {
        yield { thinking: "unfinished reasoning" };
        yield { text: "partial text" };
        if (ending === "error") throw new Error("fixture interrupted");
        if (ending === "cancel") await agent.cancel({ sessionId });
        yield { done: true, stopReason: ending === "length" ? "length" : "stop" };
      } },
    });
    sessionId = (await agent.newSession({ cwd, mcpServers: [] })).sessionId;
    try {
      const prompt = agent.prompt({ sessionId, prompt: [{ type: "text", text: "test" }] });
      if (ending === "error") await assert.rejects(prompt, /fixture interrupted/);
      else assert.equal((await prompt).stopReason, ending === "cancel" ? "cancelled" : "max_tokens");
      await agent.closeSession({ sessionId });
      const messages = store.load(sessionId)?.messages;
      assert.ok(messages?.some(message => message.role === "assistant" && message.content === "partial text"));
      assert.ok(messages?.every(message => !("reasoning_content" in message)));
    } finally {
      await agent.closeSession({ sessionId });
      rmSync(cwd, { recursive: true, force: true });
    }
  });
}
