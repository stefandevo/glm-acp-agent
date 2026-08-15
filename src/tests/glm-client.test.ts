import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir as osTmpdir } from "node:os";
import { join as pathJoin } from "node:path";
import {
  GlmClient,
  getAvailableModels,
  getDefaultModel,
  getContextWindow,
  getThoughtLevels,
  resolveThoughtLevel,
  buildThinkingParams,
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

test("getAvailableModels returns the Coding Plan allowlist by default", () => {
  const old = process.env["ACP_GLM_AVAILABLE_MODELS"];
  delete process.env["ACP_GLM_AVAILABLE_MODELS"];
  try {
    const ids = getAvailableModels().map((m) => m.modelId);
    assert.deepEqual(ids, ["glm-5.3", "glm-5-turbo", "glm-4.7"]);
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
  // Thinking cannot be turned off on glm-5.3: the endpoint accepts
  // `thinking: { type: "disabled" }` and `reasoning_effort: "none"` but keeps
  // emitting reasoning either way, so offering an "Off" option would lie.
  // Everything else on the validated ladder is exposed, in ladder order.
  const ladder = ["minimal", "low", "medium", "high", "xhigh", "max"];
  assert.deepEqual(getThoughtLevels("glm-5.3"), ladder);
  // glm-5.2 and glm-5.1 are both served by glm-5.3, so they behave identically
  // — including the fact that "Off" would not actually turn thinking off.
  assert.deepEqual(getThoughtLevels("glm-5.2"), ladder);
  assert.deepEqual(getThoughtLevels("glm-5.1"), ladder);
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
    // Forced off even when a level requests thinking.
    assert.deepEqual(buildThinkingParams("glm-5.2", "max"), { thinking: { type: "disabled" } });
  } finally {
    if (old === undefined) delete process.env["ACP_GLM_THINKING"];
    else process.env["ACP_GLM_THINKING"] = old;
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
