import test from "node:test";
import assert from "node:assert/strict";
import { GlmClient } from "../llm/glm-client.js";

// We don't want to actually hit the network, but the OpenAI SDK still
// constructs a client when we instantiate GlmClient. We stub the
// `client.chat.completions.create` method on the underlying OpenAI client.

function fakeStream(chunks: unknown[]) {
  return {
    [Symbol.asyncIterator]() {
      let i = 0;
      return {
        next() {
          if (i < chunks.length) {
            return Promise.resolve({ value: chunks[i++], done: false });
          }
          return Promise.resolve({ value: undefined, done: true });
        },
      };
    },
  };
}

function makeClient(stream: ReturnType<typeof fakeStream>): GlmClient {
  process.env["Z_AI_API_KEY"] = "test-key";
  const c = new GlmClient();
  // The OpenAI SDK exposes `chat.completions.create`. Replace it with a stub.
  (c as unknown as { client: { chat: { completions: { create: () => Promise<unknown> } } } }).client = {
    chat: { completions: { create: () => Promise.resolve(stream) } },
  };
  return c;
}

test("streamChat yields text chunks for delta.content", async () => {
  const stream = fakeStream([
    { choices: [{ delta: { content: "Hel" }, finish_reason: null }] },
    { choices: [{ delta: { content: "lo" }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);
  const c = makeClient(stream);
  const out: string[] = [];
  let lastStop: string | undefined;
  for await (const chunk of c.streamChat([])) {
    if (chunk.text) out.push(chunk.text);
    if (chunk.done) lastStop = chunk.stopReason;
  }
  assert.deepEqual(out, ["Hel", "lo"]);
  assert.equal(lastStop, "stop");
});

test("streamChat surfaces reasoning_content as thinking chunks", async () => {
  const stream = fakeStream([
    { choices: [{ delta: { reasoning_content: "Hmm..." }, finish_reason: null }] },
    { choices: [{ delta: { content: "Answer" }, finish_reason: "stop" }] },
  ]);
  const c = makeClient(stream);
  const thoughts: string[] = [];
  const text: string[] = [];
  for await (const chunk of c.streamChat([])) {
    if (chunk.thinking) thoughts.push(chunk.thinking);
    if (chunk.text) text.push(chunk.text);
  }
  assert.deepEqual(thoughts, ["Hmm..."]);
  assert.deepEqual(text, ["Answer"]);
});

test("streamChat assembles tool calls across deltas", async () => {
  const stream = fakeStream([
    {
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: "tc1", function: { name: "read_file", arguments: "" } },
            ],
          },
          finish_reason: null,
        },
      ],
    },
    {
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, function: { arguments: '{"path":"/x' } }],
          },
          finish_reason: null,
        },
      ],
    },
    {
      choices: [
        {
          delta: { tool_calls: [{ index: 0, function: { arguments: '"}' } }] },
          finish_reason: "tool_calls",
        },
      ],
    },
  ]);
  const c = makeClient(stream);
  const calls: Array<{ id: string; name: string; arguments: string }> = [];
  for await (const chunk of c.streamChat([])) {
    if (chunk.toolCall) calls.push(chunk.toolCall);
  }
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.id, "tc1");
  assert.equal(calls[0]?.name, "read_file");
  assert.equal(calls[0]?.arguments, '{"path":"/x"}');
});

test("streamChat captures usage from the trailing usage chunk", async () => {
  const stream = fakeStream([
    { choices: [{ delta: { content: "ok" }, finish_reason: "stop" }] },
    {
      choices: [],
      usage: { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 },
    },
  ]);
  const c = makeClient(stream);
  let usage: { inputTokens: number; outputTokens: number; totalTokens: number } | undefined;
  for await (const chunk of c.streamChat([])) {
    if (chunk.usage) {
      usage = {
        inputTokens: chunk.usage.inputTokens,
        outputTokens: chunk.usage.outputTokens,
        totalTokens: chunk.usage.totalTokens,
      };
    }
  }
  assert.deepEqual(usage, { inputTokens: 12, outputTokens: 7, totalTokens: 19 });
});

test("constructor throws if Z_AI_API_KEY is missing", () => {
  const old = process.env["Z_AI_API_KEY"];
  delete process.env["Z_AI_API_KEY"];
  try {
    assert.throws(() => new GlmClient(), /Z_AI_API_KEY/);
  } finally {
    if (old !== undefined) process.env["Z_AI_API_KEY"] = old;
  }
});

test("streamChat does not flush partial tool calls (missing id or name)", async () => {
  // Stream a tool_call delta that only contains arguments – no id, no name.
  // The client should silently drop the partial entry instead of yielding it.
  const stream = fakeStream([
    {
      choices: [
        {
          delta: { tool_calls: [{ index: 0, function: { arguments: '{"foo":1}' } }] },
          finish_reason: null,
        },
      ],
    },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);
  const c = makeClient(stream);
  const calls: Array<{ id: string; name: string }> = [];
  for await (const chunk of c.streamChat([])) {
    if (chunk.toolCall) calls.push({ id: chunk.toolCall.id, name: chunk.toolCall.name });
  }
  assert.equal(calls.length, 0);
});
