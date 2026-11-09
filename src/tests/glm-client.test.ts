import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir as osTmpdir } from "node:os";
import { join as pathJoin } from "node:path";
import OpenAI from "openai";
import {
  GlmClient,
  ModelStreamIdleTimeoutError,
  PROVIDER_MAX_RETRIES,
  getAvailableModels,
  getDefaultModel,
  getContextWindow,
  getThoughtLevels,
  resolveThoughtLevel,
  buildThinkingParams,
  isVisionNativeModel,
  isRecoverableTransportError,
} from "../llm/glm-client.js";

test("constructor uses the coding endpoint by default", () => {
  process.env["Z_AI_API_KEY"] = "test-key";
  delete process.env["ACP_GLM_BASE_URL"];
  try {
    const c = new GlmClient();
    const url = (c as unknown as { client: { baseURL: string } }).client.baseURL;
    assert.equal(url, "https://api.z.ai/api/coding/paas/v4");
  } finally {
    delete process.env["Z_AI_API_KEY"];
  }
});

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

function makeClientWithCreate(
  create: (body: Record<string, unknown>) => Promise<unknown>
): GlmClient {
  process.env["Z_AI_API_KEY"] = "test-key";
  const c = new GlmClient();
  (c as unknown as {
    client: { chat: { completions: { create: (body: Record<string, unknown>) => Promise<unknown> } } };
  }).client = {
    chat: { completions: { create } },
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
  const oldXdg = process.env["XDG_CONFIG_HOME"];
  delete process.env["Z_AI_API_KEY"];
  // resolveApiKey() also reads $XDG_CONFIG_HOME/glm-acp-agent/credentials.json,
  // so point it at an empty dir — otherwise this fails on any machine where the
  // developer has actually run `--setup`.
  const emptyConfig = mkdtempSync(pathJoin(osTmpdir(), "glm-acp-noconfig-"));
  process.env["XDG_CONFIG_HOME"] = emptyConfig;
  try {
    assert.throws(() => new GlmClient(), /Z_AI_API_KEY/);
  } finally {
    if (old !== undefined) process.env["Z_AI_API_KEY"] = old;
    if (oldXdg === undefined) delete process.env["XDG_CONFIG_HOME"];
    else process.env["XDG_CONFIG_HOME"] = oldXdg;
    rmSync(emptyConfig, { recursive: true, force: true });
  }
});

// Minimal stalled SSE provider (same pattern as stream-contract.test.ts):
// one content frame, then the connection stays open and silent forever.
// The `construct` callback lets tests set env vars before GlmClient reads
// them, since the idle timeout is resolved at construction time.
async function withStalledProvider(
  run: (construct: () => GlmClient) => Promise<void>
): Promise<void> {
  const server = createServer((_req, res) => {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.write(`data: ${JSON.stringify({
      choices: [{ index: 0, delta: { content: "partial output" }, finish_reason: null }],
    })}\n\n`);
    res.flushHeaders();
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedUrl = process.env["ACP_GLM_BASE_URL"];
  const savedIdle = process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
  process.env["Z_AI_API_KEY"] = "fixture-key";
  process.env["ACP_GLM_BASE_URL"] = `http://127.0.0.1:${address.port}/v4`;
  delete process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
  try {
    await run(() => new GlmClient());
  } finally {
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedUrl === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedUrl;
    if (savedIdle === undefined) delete process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
    else process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"] = savedIdle;
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(err => (err ? reject(err) : resolve())));
  }
}

test("stream idle timeout is resolved at construction from the environment", async () => {
  await withStalledProvider(async construct => {
    // Bake a small idle window into the client at construction.
    process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"] = "60";
    const client = construct();

    // The stalled provider trips the constructed timeout...
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) void chunk;
    }, (err: unknown) => err instanceof ModelStreamIdleTimeoutError);

    // ...and changing the env var afterwards must not raise the timeout:
    // a second stalled call still rejects via the constructed 60ms window.
    process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"] = "60000";
    const start = Date.now();
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) void chunk;
    }, (err: unknown) => err instanceof ModelStreamIdleTimeoutError);
    const elapsed = Date.now() - start;
    assert.ok(elapsed < 3000, `env was re-read after construction (took ${elapsed}ms)`);
  });
});

test("invalid ACP_GLM_MAX_TOKENS falls back to the default with a stderr warning", () => {
  const savedKey = process.env["Z_AI_API_KEY"];
  const savedTokens = process.env["ACP_GLM_MAX_TOKENS"];
  const savedIdle = process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
  delete process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
  process.env["Z_AI_API_KEY"] = "test-key";
  process.env["ACP_GLM_MAX_TOKENS"] = "abc";
  const written: string[] = [];
  const originalWrite = process.stderr.write.bind(process.stderr);
  process.stderr.write = ((chunk: unknown) => {
    written.push(String(chunk));
    return true;
  }) as typeof process.stderr.write;
  try {
    const c = new GlmClient();
    assert.equal(c.getMaxOutputTokens(), 32_768);
    const warning = written.find((line) => line.includes("ACP_GLM_MAX_TOKENS"));
    assert.ok(warning, `expected a stderr warning, got: ${JSON.stringify(written)}`);
    assert.match(warning!, /invalid/);
  } finally {
    process.stderr.write = originalWrite;
    if (savedKey === undefined) delete process.env["Z_AI_API_KEY"];
    else process.env["Z_AI_API_KEY"] = savedKey;
    if (savedTokens === undefined) delete process.env["ACP_GLM_MAX_TOKENS"];
    else process.env["ACP_GLM_MAX_TOKENS"] = savedTokens;
    if (savedIdle === undefined) delete process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"];
    else process.env["ACP_GLM_STREAM_IDLE_TIMEOUT_MS"] = savedIdle;
  }
});

test("getAvailableModels returns the Coding Plan allowlist by default", () => {
  const old = process.env["ACP_GLM_AVAILABLE_MODELS"];
  delete process.env["ACP_GLM_AVAILABLE_MODELS"];
  try {
    const ids = getAvailableModels().map((m) => m.modelId);
    assert.deepEqual(ids, ["glm-5.3", "glm-5.3-flash", "glm-5-turbo", "glm-4.7"]);
    assert.ok(!ids.includes("glm-4v-plus"), "glm-4v-plus must not be advertised");
    assert.ok(!ids.includes("glm-4.6"), "glm-4.6 must not be advertised");
    assert.ok(!ids.includes("glm-4.5"), "glm-4.5 must not be advertised");
    // Aliases: the Coding Plan endpoint answers these with a different model
    // (5.2/5.1 → glm-5.3, 4.5-air → glm-4.7), so advertising them would report
    // a model the user is not actually talking to.
    assert.ok(!ids.includes("glm-5.2"), "glm-5.2 is an alias for glm-5.3");
    assert.ok(!ids.includes("glm-5.1"), "glm-5.1 is an alias for glm-5.3");
    assert.ok(!ids.includes("glm-4.5-air"), "glm-4.5-air is an alias for glm-4.7");
    // Not on the Coding Plan allowlist — selecting it fails with code 1311.
    assert.ok(!ids.includes("glm-5v-turbo"), "glm-5v-turbo is not Coding Plan accessible");
  } finally {
    if (old !== undefined) process.env["ACP_GLM_AVAILABLE_MODELS"] = old;
  }
});

test("getDefaultModel returns glm-5.3 without an env override", () => {
  const old = process.env["ACP_GLM_MODEL"];
  delete process.env["ACP_GLM_MODEL"];
  try {
    assert.equal(getDefaultModel(), "glm-5.3");
  } finally {
    if (old !== undefined) process.env["ACP_GLM_MODEL"] = old;
  }
});

test("getContextWindow reports the 1M window for glm-5.3", () => {
  assert.equal(getContextWindow("glm-5.3"), 1_000_000);
});

test("getContextWindow reports the 1M window for glm-5.3-flash", () => {
  assert.equal(getContextWindow("glm-5.3-flash"), 1_000_000);
});

test("getContextWindow reports the 1M window for glm-5.2", () => {
  assert.equal(getContextWindow("glm-5.2"), 1_000_000);
});

test("getContextWindow reports the routed window for de-listed aliases", () => {
  // glm-5.1 is served by glm-5.3 (1M) and glm-4.5-air by glm-4.7 (200K).
  // Reporting the old standalone windows would compact these sessions early.
  assert.equal(getContextWindow("glm-5.1"), 1_000_000);
  assert.equal(getContextWindow("glm-4.5-air"), 200_000);
});

test("ACP_GLM_AVAILABLE_MODELS env override still wins over the built-in list", () => {
  const old = process.env["ACP_GLM_AVAILABLE_MODELS"];
  process.env["ACP_GLM_AVAILABLE_MODELS"] = "custom-a, custom-b";
  try {
    const ids = getAvailableModels().map((m) => m.modelId);
    assert.deepEqual(ids, ["custom-a", "custom-b"]);
  } finally {
    if (old === undefined) delete process.env["ACP_GLM_AVAILABLE_MODELS"];
    else process.env["ACP_GLM_AVAILABLE_MODELS"] = old;
  }
});

test("streamChat auto-enables thinking for glm-5v-turbo", async () => {
  const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
  let requestBody: Record<string, unknown> | undefined;
  const c = makeClientWithCreate((body) => {
    requestBody = body;
    return Promise.resolve(stream);
  });

  for await (const chunk of c.streamChat([], undefined, { model: "glm-5v-turbo" })) {
    // Drain the stream so the request is made.
    void chunk;
  }

  assert.equal(requestBody?.["model"], "glm-5v-turbo");
  assert.deepEqual(requestBody?.["thinking"], { type: "enabled" });
});

test("streamChat sends reasoning_effort when level is set on glm-5.3", async () => {
  const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
  let requestBody: Record<string, unknown> | undefined;
  const c = makeClientWithCreate((body) => {
    requestBody = body;
    return Promise.resolve(stream);
  });

  // Spot-check across the ladder: endpoints, a middle level, and the one
  // non-capitalized id ("xhigh") that would break on naive name mangling.
  for (const effort of ["max", "medium", "minimal", "xhigh"] as const) {
    for await (const chunk of c.streamChat([], undefined, { model: "glm-5.3", reasoningEffort: effort })) {
      void chunk;
    }

    assert.deepEqual(requestBody?.["thinking"], { type: "enabled" });
    assert.equal(requestBody?.["reasoning_effort"], effort);
  }
});

test("streamChat sends reasoning_effort on the de-listed aliases routed to 5.3", async () => {
  for (const model of ["glm-5.2", "glm-5.1"]) {
    const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
    let requestBody: Record<string, unknown> | undefined;
    const c = makeClientWithCreate((body) => {
      requestBody = body;
      return Promise.resolve(stream);
    });

    for await (const chunk of c.streamChat([], undefined, { model, reasoningEffort: "high" })) {
      void chunk;
    }

    assert.equal(requestBody?.["reasoning_effort"], "high", `expected effort for ${model}`);
  }
});

test("streamChat sends reasoning_effort when level is set on glm-5.2", async () => {
  const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
  let requestBody: Record<string, unknown> | undefined;
  const c = makeClientWithCreate((body) => {
    requestBody = body;
    return Promise.resolve(stream);
  });

  for await (const chunk of c.streamChat([], undefined, { model: "glm-5.2", reasoningEffort: "high" })) {
    void chunk;
  }

  assert.deepEqual(requestBody?.["thinking"], { type: "enabled" });
  assert.equal(requestBody?.["reasoning_effort"], "high");
});

test("streamChat does not send reasoning_effort for models that ignore it", async () => {
  const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
  let requestBody: Record<string, unknown> | undefined;
  const c = makeClientWithCreate((body) => {
    requestBody = body;
    return Promise.resolve(stream);
  });

  for await (const chunk of c.streamChat([], undefined, { model: "glm-4.7", reasoningEffort: "high" })) {
    void chunk;
  }

  assert.deepEqual(requestBody?.["thinking"], { type: "enabled" });
  assert.equal(requestBody?.["reasoning_effort"], undefined);
});

test("streamChat disables thinking when effort is none", async () => {
  const stream = fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }]);
  let requestBody: Record<string, unknown> | undefined;
  const c = makeClientWithCreate((body) => {
    requestBody = body;
    return Promise.resolve(stream);
  });

  for await (const chunk of c.streamChat([], undefined, { model: "glm-4.7", reasoningEffort: "none" })) {
    void chunk;
  }

  assert.deepEqual(requestBody?.["thinking"], { type: "disabled" });
  assert.equal(requestBody?.["reasoning_effort"], undefined);
});

test("getThoughtLevels omits none for reasoning-effort models", () => {
  // Thinking cannot be turned off on glm-5.3 / glm-5.3-flash: the endpoint
  // errors on `thinking: { type: "disabled" }`, so offering an "Off" option
  // would lie. Everything else on the validated ladder is exposed, in order.
  const ladder = ["minimal", "low", "medium", "high", "xhigh", "max"];
  assert.deepEqual(getThoughtLevels("glm-5.3"), ladder);
  // glm-5.2 and glm-5.1 are both served by glm-5.3, so they behave identically
  // — including the fact that "Off" would not actually turn thinking off.
  assert.deepEqual(getThoughtLevels("glm-5.2"), ladder);
  assert.deepEqual(getThoughtLevels("glm-5.1"), ladder);
});

test("getThoughtLevels advertises only low/high/max for glm-5.3-flash", () => {
  // Z.AI Chat Completions: GLM-5.3-Flash reasoning_effort is only low/high/max.
  assert.deepEqual(getThoughtLevels("glm-5.3-flash"), ["low", "high", "max"]);
  assert.deepEqual(getThoughtLevels("GLM-5.3-Flash"), ["low", "high", "max"]);
});

test("getThoughtLevels returns none/on for models without reasoning_effort", () => {
  assert.deepEqual(getThoughtLevels("glm-4.7"), ["none", "on"]);
  assert.deepEqual(getThoughtLevels("glm-5-turbo"), ["none", "on"]);
});

test("resolveThoughtLevel clamps invalid levels to the model default", () => {
  // Effort-ladder levels only apply to the reasoning-effort models; others
  // get "on".
  assert.equal(resolveThoughtLevel("glm-4.7", "max"), "on");
  assert.equal(resolveThoughtLevel("glm-4.7", "high"), "on");
  assert.equal(resolveThoughtLevel("glm-4.7", "xhigh"), "on");
  // Valid levels are preserved, across the whole ladder.
  assert.equal(resolveThoughtLevel("glm-5.3", "high"), "high");
  assert.equal(resolveThoughtLevel("glm-5.3", "minimal"), "minimal");
  assert.equal(resolveThoughtLevel("glm-5.3", "xhigh"), "xhigh");
  assert.equal(resolveThoughtLevel("glm-4.7", "none"), "none");
});

test("resolveThoughtLevel clamps a persisted none up to max on glm-5.3", () => {
  // A session saved on a model where "Off" was valid must still load cleanly
  // when switched to glm-5.3, which has no "none" level.
  assert.equal(resolveThoughtLevel("glm-5.3", "none"), "max");
  assert.equal(resolveThoughtLevel("glm-5.3", "on"), "max");
});

test("resolveThoughtLevel maps the 5.3 ladder onto Flash's three levels", () => {
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "low"), "low");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "high"), "high");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "max"), "max");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "minimal"), "low");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "medium"), "high");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "xhigh"), "max");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "none"), "max");
  assert.equal(resolveThoughtLevel("glm-5.3-flash", "on"), "max");
});

test("buildThinkingParams omits fields for non-thinking models", () => {
  const old = process.env["ACP_GLM_THINKING"];
  delete process.env["ACP_GLM_THINKING"];
  try {
    assert.deepEqual(buildThinkingParams("some-other-model"), {});
  } finally {
    if (old !== undefined) process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams defaults to enabled with no effort", () => {
  const old = process.env["ACP_GLM_THINKING"];
  delete process.env["ACP_GLM_THINKING"];
  try {
    assert.deepEqual(buildThinkingParams("glm-5.2"), { thinking: { type: "enabled" } });
  } finally {
    if (old !== undefined) process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams treats glm-5.3 as a thinking model", () => {
  const old = process.env["ACP_GLM_THINKING"];
  delete process.env["ACP_GLM_THINKING"];
  try {
    // Proves modelSupportsThinking() matches glm-5.3 — no reasoning_effort is
    // attached without an explicit level, so the model default applies.
    assert.deepEqual(buildThinkingParams("glm-5.3"), { thinking: { type: "enabled" } });
  } finally {
    if (old !== undefined) process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams attaches reasoning_effort on glm-5.3", () => {
  const old = process.env["ACP_GLM_THINKING"];
  delete process.env["ACP_GLM_THINKING"];
  try {
    // Every level on the exposed ladder maps to its own reasoning_effort.
    for (const level of ["minimal", "low", "medium", "high", "xhigh", "max"] as const) {
      assert.deepEqual(buildThinkingParams("glm-5.3", level), {
        thinking: { type: "enabled" },
        reasoning_effort: level,
      });
    }
    // "on" carries no effort, so it stays a plain thinking toggle.
    assert.deepEqual(buildThinkingParams("glm-5.3", "on"), {
      thinking: { type: "enabled" },
    });
  } finally {
    if (old !== undefined) process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams never emits reasoning_effort='none' when thinking is force-enabled", () => {
  const old = process.env["ACP_GLM_THINKING"];
  process.env["ACP_GLM_THINKING"] = "true";
  try {
    // Force-on + level "none" on glm-5.2: thinking is enabled, but "none" is
    // not a valid reasoning_effort value, so the field must be omitted.
    // Exact deep-equality proves reasoning_effort is absent (not "none").
    assert.deepEqual(buildThinkingParams("glm-5.2", "none"), { thinking: { type: "enabled" } });
  } finally {
    if (old === undefined) delete process.env["ACP_GLM_THINKING"];
    else process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams honours ACP_GLM_THINKING=false override", () => {
  const old = process.env["ACP_GLM_THINKING"];
  process.env["ACP_GLM_THINKING"] = "false";
  try {
    // Forced off even when a level requests thinking — models that still
    // accept thinking.type: "disabled" (not the GLM-5.3 family).
    assert.deepEqual(buildThinkingParams("glm-4.7", "on"), { thinking: { type: "disabled" } });
    assert.deepEqual(buildThinkingParams("glm-5-turbo", "on"), { thinking: { type: "disabled" } });
  } finally {
    if (old === undefined) delete process.env["ACP_GLM_THINKING"];
    else process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams attaches reasoning_effort on glm-5.3-flash", () => {
  const old = process.env["ACP_GLM_THINKING"];
  delete process.env["ACP_GLM_THINKING"];
  try {
    for (const level of ["low", "high", "max"] as const) {
      assert.deepEqual(buildThinkingParams("glm-5.3-flash", level), {
        thinking: { type: "enabled" },
        reasoning_effort: level,
      });
    }
    // Unadvertised 5.3-ladder values must not be forwarded as-is.
    assert.deepEqual(buildThinkingParams("glm-5.3-flash", "minimal"), {
      thinking: { type: "enabled" },
    });
    assert.deepEqual(buildThinkingParams("glm-5.3-flash", "medium"), {
      thinking: { type: "enabled" },
    });
    assert.deepEqual(buildThinkingParams("glm-5.3-flash", "xhigh"), {
      thinking: { type: "enabled" },
    });
  } finally {
    if (old !== undefined) process.env["ACP_GLM_THINKING"] = old;
  }
});

test("buildThinkingParams does not send thinking.type=disabled on GLM-5.3 family", () => {
  const old = process.env["ACP_GLM_THINKING"];
  process.env["ACP_GLM_THINKING"] = "false";
  try {
    // Latest Z.AI docs: thinking.type: "disabled" errors on GLM-5.3 / Flash.
    for (const model of ["glm-5.3", "glm-5.3-flash"]) {
      const params = buildThinkingParams(model, "max");
      assert.notDeepEqual(
        params["thinking"],
        { type: "disabled" },
        `${model} must not send thinking.type: disabled`
      );
      assert.deepEqual(params["thinking"], { type: "enabled" });
      assert.equal(params["reasoning_effort"], "max");
    }
  } finally {
    if (old === undefined) delete process.env["ACP_GLM_THINKING"];
    else process.env["ACP_GLM_THINKING"] = old;
  }
});

test("isVisionNativeModel is true for Flash and glm-5v-turbo only", () => {
  assert.equal(isVisionNativeModel("glm-5.3-flash"), true);
  assert.equal(isVisionNativeModel("GLM-5.3-Flash"), true);
  assert.equal(isVisionNativeModel("glm-5v-turbo"), true);
  assert.equal(isVisionNativeModel("glm-5.3"), false);
  assert.equal(isVisionNativeModel("glm-5-turbo"), false);
  assert.equal(isVisionNativeModel("glm-4.7"), false);
});

test("streamChat rejects inconsistent partial tool calls instead of reporting success", async () => {
  // Stream a tool_call delta that only contains arguments – no id, no name.
  // It must not turn a malformed tool response into a successful text stop.
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
  await assert.rejects(async () => {
    for await (const chunk of c.streamChat([])) {
      if (chunk.toolCall) calls.push({ id: chunk.toolCall.id, name: chunk.toolCall.name });
    }
  }, /inconsistent|incomplete/i);
  assert.equal(calls.length, 0);
});

test("isRecoverableTransportError accepts connection drops and rejects HTTP statuses", () => {
  assert.equal(
    isRecoverableTransportError(new OpenAI.APIConnectionError({ message: "Connection error." })),
    true
  );
  assert.equal(
    isRecoverableTransportError(new OpenAI.APIConnectionTimeoutError({ message: "Request timed out." })),
    true
  );
  const withStatus = new OpenAI.APIConnectionError({ message: "Connection error." });
  Object.assign(withStatus, { status: 502 });
  assert.equal(isRecoverableTransportError(withStatus), false);
  assert.equal(isRecoverableTransportError(Object.assign(new Error("reset"), { code: "ECONNRESET" })), true);
  assert.equal(isRecoverableTransportError(Object.assign(new Error("upstream"), { status: 502 })), false);
  assert.equal(isRecoverableTransportError(new Error("fixture interrupted")), false);
});

interface OwnedDispatcher {
  closed: boolean;
  destroyed: boolean;
  close: () => Promise<void>;
  destroy: () => Promise<void>;
}

function clientInternals(client: GlmClient): {
  client: unknown;
  dispatcher: OwnedDispatcher;
  discardTransport: () => void;
} {
  return client as unknown as {
    client: unknown;
    dispatcher: OwnedDispatcher;
    discardTransport: () => void;
  };
}

function withSavedEnv(update: Record<string, string | undefined>): () => void {
  const saved = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(update)) {
    saved.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return () => {
    for (const [key, value] of saved) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };
}

async function listenLocal(onRequest: (req: import("node:http").IncomingMessage, res: import("node:http").ServerResponse) => void): Promise<{
  baseURL: string;
  close: () => Promise<void>;
}> {
  const server = createServer(onRequest);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("test server has no port");
  return {
    baseURL: `http://127.0.0.1:${address.port}/v1`,
    close: () => new Promise((resolve, reject) => {
      server.close((err) => err ? reject(err) : resolve());
    }),
  };
}

function captureStderr(): { text: () => string; restore: () => void } {
  const chunks: string[] = [];
  const original = process.stderr.write.bind(process.stderr);
  process.stderr.write = ((chunk: string | Uint8Array) => {
    chunks.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8"));
    return true;
  }) as typeof process.stderr.write;
  return {
    text: () => chunks.join(""),
    restore: () => {
      process.stderr.write = original;
    },
  };
}

test("each provider client owns a connection pool and a failed rebuild keeps the old one", () => {
  // resolveApiKey() falls back to $XDG_CONFIG_HOME/glm-acp-agent/credentials.json
  // when the env var is unset, so an empty config dir is required or a
  // machine that has run --setup would rebuild successfully.
  const emptyConfig = mkdtempSync(pathJoin(osTmpdir(), "glm-acp-noconfig-"));
  const restore = withSavedEnv({
    Z_AI_API_KEY: "test-key",
    XDG_CONFIG_HOME: emptyConfig,
  });
  try {
    const first = new GlmClient();
    const second = new GlmClient();
    const firstPool = clientInternals(first).dispatcher;
    const secondPool = clientInternals(second).dispatcher;
    assert.notEqual(firstPool, secondPool);
    const logs = captureStderr();
    try {
      const restoreKey = withSavedEnv({ Z_AI_API_KEY: undefined });
      try {
        clientInternals(first).discardTransport();
      } finally {
        restoreKey();
      }
      assert.match(logs.text(), /keeping the existing client/);
    } finally {
      logs.restore();
    }
    assert.equal(clientInternals(first).dispatcher, firstPool);
    assert.equal(firstPool.closed, false);
    assert.equal(firstPool.destroyed, false);
  } finally {
    restore();
    rmSync(emptyConfig, { recursive: true, force: true });
  }
});

test("streamChat closes the connection pool when the body fails with a socket reset", async () => {
  const restore = withSavedEnv({ Z_AI_API_KEY: "test-key" });
  const client = new GlmClient();
  const internals = clientInternals(client);
  const previous = internals.dispatcher;
  const broken = {
    [Symbol.asyncIterator]() {
      return {
        next() {
          return Promise.reject(Object.assign(new Error("socket hang up"), { code: "ECONNRESET" }));
        },
      };
    },
  };
  internals.client = { chat: { completions: { create: () => Promise.resolve(broken) } } };
  try {
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) {
        void chunk;
      }
    });
    assert.notEqual(internals.dispatcher, previous);
    assert.equal(previous.closed, true);
  } finally {
    await internals.dispatcher.destroy().catch(() => undefined);
    restore();
  }
});

test("a dropped connection retries twice, logs each attempt, and then replaces the pool", async () => {
  assert.equal(PROVIDER_MAX_RETRIES, 2);
  let attempts = 0;
  const server = await listenLocal((req) => {
    attempts += 1;
    req.socket.destroy();
  });
  const restore = withSavedEnv({ Z_AI_API_KEY: "test-key", ACP_GLM_BASE_URL: server.baseURL });
  const client = new GlmClient();
  const previous = clientInternals(client).dispatcher;
  // The client captured the base URL. Put the process env back before the
  // await so other tests are not pointed at this server. The key stays until
  // the failure has rebuilt the client.
  const savedBase = process.env["ACP_GLM_BASE_URL"];
  delete process.env["ACP_GLM_BASE_URL"];
  const logs = captureStderr();
  try {
    await assert.rejects(async () => {
      for await (const chunk of client.streamChat([])) {
        void chunk;
      }
    }, OpenAI.APIConnectionError);
    assert.equal(attempts, PROVIDER_MAX_RETRIES + 1);
    const stderr = logs.text();
    assert.match(stderr, /provider request attempt 1 failed: code=\S+/);
    assert.match(stderr, /provider request attempt 2 failed: code=\S+/);
    assert.match(stderr, /provider request attempt 3 failed: code=\S+/);
    assert.notEqual(clientInternals(client).dispatcher, previous);
    assert.equal(previous.closed, true);
  } finally {
    if (savedBase === undefined) delete process.env["ACP_GLM_BASE_URL"];
    else process.env["ACP_GLM_BASE_URL"] = savedBase;
    logs.restore();
    await clientInternals(client).dispatcher.destroy().catch(() => undefined);
    restore();
    await server.close();
  }
});

test("HTTP 408, 409, 429, and 5xx use the same retry budget and 400 does not", async () => {
  const statuses = [408, 409, 429, 500, 502, 400];
  let status = 408;
  let attempts = 0;
  const server = await listenLocal((_req, res) => {
    attempts += 1;
    res.writeHead(status, {
      "content-type": "application/json",
      "retry-after-ms": "0",
    });
    res.end(JSON.stringify({ error: { message: `status ${status}` } }));
  });
  const restore = withSavedEnv({ Z_AI_API_KEY: "test-key", ACP_GLM_BASE_URL: server.baseURL });
  const client = new GlmClient();
  delete process.env["ACP_GLM_BASE_URL"];
  const pool = clientInternals(client).dispatcher;
  try {
    for (const code of statuses) {
      status = code;
      attempts = 0;
      await assert.rejects(async () => {
        for await (const chunk of client.streamChat([])) {
          void chunk;
        }
      });
      const expected = code === 400 ? 1 : PROVIDER_MAX_RETRIES + 1;
      assert.equal(attempts, expected, `status ${code} should make ${expected} HTTP attempts`);
      assert.equal(clientInternals(client).dispatcher, pool, `status ${code} must not replace the connection pool`);
    }
  } finally {
    restore();
    await pool.destroy().catch(() => undefined);
    await server.close();
  }
});

test("request max_tokens follows the model's documented output limit", async () => {
  delete process.env["ACP_GLM_MAX_TOKENS"];
  try {
    const bodies: Array<Record<string, unknown>> = [];
    const c = makeClientWithCreate((body) => {
      bodies.push(body);
      return Promise.resolve(
        fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }])
      );
    });
    for await (const chunk of c.streamChat([], undefined, { model: "glm-5.3-flash" })) {
      void chunk;
    }
    assert.equal(bodies[0]!["max_tokens"], 131_072);
  } finally {
    delete process.env["Z_AI_API_KEY"];
  }
});

test("ACP_GLM_MAX_TOKENS still overrides the per-model default", async () => {
  const saved = process.env["ACP_GLM_MAX_TOKENS"];
  process.env["ACP_GLM_MAX_TOKENS"] = "4096";
  try {
    const bodies: Array<Record<string, unknown>> = [];
    const c = makeClientWithCreate((body) => {
      bodies.push(body);
      return Promise.resolve(
        fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }])
      );
    });
    for await (const chunk of c.streamChat([], undefined, { model: "glm-5.3" })) {
      void chunk;
    }
    assert.equal(bodies[0]!["max_tokens"], 4096);
  } finally {
    if (saved === undefined) delete process.env["ACP_GLM_MAX_TOKENS"];
    else process.env["ACP_GLM_MAX_TOKENS"] = saved;
    delete process.env["Z_AI_API_KEY"];
  }
});

test("uncatalogued models keep the flat 32768 default", async () => {
  delete process.env["ACP_GLM_MAX_TOKENS"];
  try {
    const bodies: Array<Record<string, unknown>> = [];
    const c = makeClientWithCreate((body) => {
      bodies.push(body);
      return Promise.resolve(
        fakeStream([{ choices: [{ delta: {}, finish_reason: "stop" }] }])
      );
    });
    for await (const chunk of c.streamChat([], undefined, { model: "glm-not-in-catalog" })) {
      void chunk;
    }
    assert.equal(bodies[0]!["max_tokens"], 32_768);
  } finally {
    delete process.env["Z_AI_API_KEY"];
  }
});
