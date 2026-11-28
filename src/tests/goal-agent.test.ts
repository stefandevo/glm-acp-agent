import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GlmAcpAgent, type GlmAcpAgentOptions } from "../protocol/agent.js";
import { SessionStore } from "../protocol/session-store.js";
import type { GlmMessage, StreamChatOptions } from "../llm/glm-client.js";

const home = mkdtempSync(join(tmpdir(), "glm-goal-home-"));
process.env.HOME = home;
process.env.USERPROFILE = home;
delete process.env["ACP_GLM_GOAL_JUDGE_MODEL"];
delete process.env["ACP_GLM_GOAL_MAX_ROUNDS"];
test.after(() => rmSync(home, { recursive: true, force: true }));

type Update = { update: { sessionUpdate: string; content?: { text?: string }; availableCommands?: Array<{ name: string; description?: string; input?: { hint?: string } }> } };

interface ScriptedTurn {
  text?: string;
  /** A read-only tool call keeps the no-progress streak at zero. */
  toolCall?: boolean;
  /** Calls ask_user_question with one question instead of a text/tool turn. */
  ask?: boolean;
}

interface Script {
  /** Main-model turns, consumed in order; later calls repeat the last one. */
  main: ScriptedTurn[];
  /** Evaluator replies, consumed in order; an Error is thrown by the stream. */
  judge: Array<string | Error>;
  /** Awaited before a main-model turn streams, so a test can interleave a prompt. */
  beforeMain?: (index: number) => Promise<void>;
  /** Thrown by the main model before a turn streams, like a dropped connection. */
  mainError?: (index: number) => Error | undefined;
}

const ASK_ARGS = JSON.stringify({
  questions: [{
    question: "Which database should the migration target?",
    header: "Target",
    multiSelect: false,
    options: [
      { label: "Postgres", description: "The staging database." },
      { label: "SQLite", description: "Local only." },
    ],
  }],
});

/** Client-side ACP surface the fakes need: a dismissed form and a dismissed permission prompt. */
function clientStub(updates: Update[]) {
  return {
    async sessionUpdate(update: Update) { updates.push(update); },
    async requestPermission() { return { outcome: { outcome: "cancelled" as const } }; },
    async unstable_createElicitation() { return { action: "cancel" as const }; },
  };
}

const NOT_MET = (reason: string) => `{"verdict":"not_met","reason":"${reason}"}`;
const MET = (reason: string) => `{"verdict":"met","reason":"${reason}"}`;
const GOAL_ACHIEVED = /\*\*Goal: achieved\*\*/;
/** Token usage the fake model and the fake evaluator report per call. */
const MAIN_USAGE = { inputTokens: 1_000, outputTokens: 100, totalTokens: 1_100 };
const JUDGE_USAGE = { inputTokens: 100, outputTokens: 20, totalTokens: 120 };
const summedUsage = (mainCalls: number, judgeCalls: number) => ({
  inputTokens: mainCalls * MAIN_USAGE.inputTokens + judgeCalls * JUDGE_USAGE.inputTokens,
  outputTokens: mainCalls * MAIN_USAGE.outputTokens + judgeCalls * JUDGE_USAGE.outputTokens,
  totalTokens: mainCalls * MAIN_USAGE.totalTokens + judgeCalls * JUDGE_USAGE.totalTokens,
});

function setup(script: Script, options: Partial<GlmAcpAgentOptions> = {}, store: SessionStore | null = null) {
  // The store is handed back so tests can read what a fork or load persisted.
  const updates: Update[] = [];
  const modelCalls: GlmMessage[][] = [];
  const judgeCalls: GlmMessage[][] = [];
  const judgeOptions: Array<StreamChatOptions | undefined> = [];
  let mainIndex = 0;
  let judgeIndex = 0;
  const glm: NonNullable<GlmAcpAgentOptions["glm"]> = {
    async *streamChat(messages, signal, options) {
      if (options?.tools === null) {
        judgeCalls.push(structuredClone(messages));
        judgeOptions.push(options);
        const reply = script.judge[judgeIndex++] ?? NOT_MET("no verdict scripted");
        if (reply instanceof Error) throw reply;
        yield { text: reply, usage: { inputTokens: 100, outputTokens: 20, totalTokens: 120 } };
        yield { done: true, stopReason: "stop", model: options?.model };
        return;
      }
      const index = mainIndex++;
      modelCalls.push(structuredClone(messages));
      await script.beforeMain?.(index);
      const failure = script.mainError?.(index);
      if (failure) throw failure;
      signal?.throwIfAborted();
      const turn = script.main[Math.min(index, script.main.length - 1)] ?? { text: "done" };
      if (turn.ask) {
        yield { toolCall: { id: `ask_${index}`, name: "ask_user_question", arguments: ASK_ARGS } };
      }
      if (turn.toolCall) {
        yield { toolCall: { id: `call_${index}`, name: "list_files", arguments: JSON.stringify({ path: "." }) } };
      }
      if (turn.text) yield { text: turn.text };
      yield { usage: { inputTokens: 1_000, outputTokens: 100, totalTokens: 1_100 } };
      yield { done: true, stopReason: turn.toolCall || turn.ask ? "tool_calls" : "stop" };
    },
  };
  const agent = new GlmAcpAgent(clientStub(updates) as never, {
    glm, visionClient: null, sessionStore: store, ...options,
  });
  const texts = (): string[] => updates.map(update => update.update.content?.text ?? "").filter(text => text.length > 0);
  const lastUserText = (messages: GlmMessage[]): string => {
    const last = [...messages].reverse().find(message => message.role === "user");
    return typeof last?.content === "string" ? last.content : JSON.stringify(last?.content);
  };
  /** Run a `/goal` control word and return the agent's reply. */
  const status = async (sessionId: string, text = "/goal"): Promise<string> => {
    const before = updates.length;
    const result = await agent.prompt({ sessionId, prompt: [{ type: "text", text }] });
    assert.equal(result.stopReason, "end_turn");
    return updates.slice(before).map(update => update.update.content?.text ?? "").join("\n");
  };
  return { agent, updates, texts, modelCalls, judgeCalls, judgeOptions, lastUserText, status, store };
}

function tempDir(prefix: string): string {
  return mkdtempSync(join(tmpdir(), prefix));
}

const tick = () => new Promise(resolve => setTimeout(resolve, 5));

test("/goal is advertised once with an input hint and shadows a file-backed goal command", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    mkdirSync(join(cwd, ".claude", "commands"), { recursive: true });
    writeFileSync(join(cwd, ".claude", "commands", "goal.md"), "Send this to the model");
    const h = setup({ main: [], judge: [] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    await tick();
    const commands = h.updates.find(update => update.update.sessionUpdate === "available_commands_update")?.update.availableCommands ?? [];
    const goals = commands.filter(command => command.name === "goal");
    assert.equal(goals.length, 1);
    assert.match(goals[0]?.description ?? "", /evaluator/);
    assert.equal(goals[0]?.input?.hint, "<condition | status | pause | resume | clear>");
    assert.ok(commands.some(command => command.name === "usage"));
    // A file-backed goal.md never shadows the built-in: the typed text becomes the directive, not that body.
    const result = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal all tests pass" }] });
    assert.equal(result.stopReason, "end_turn");
    assert.doesNotMatch(h.lastUserText(h.modelCalls[0]!), /Send this to the model/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("setting a goal sends the directive, keeps the typed text for replay, and loops until the evaluator says met", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    // A tool-call turn and the turn that reads its result are two model calls
    // inside one round; the second round is the text-only "Suite is green.".
    const h = setup({
      main: [{ toolCall: true, text: "Running the suite." }, { text: "Still running." }, { text: "Suite is green." }],
      judge: [NOT_MET("refresh still fails"), MET("npm test exits 0")],
    });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const result = await h.agent.prompt({ sessionId, messageId: "m1", prompt: [{ type: "text", text: "/goal all tests in test/auth pass" }] });
    assert.equal(result.stopReason, "end_turn");
    assert.equal(result.userMessageId, "m1");
    assert.deepEqual(result.usage, summedUsage(3, 2));

    assert.equal(h.modelCalls.length, 3);
    const directive = h.lastUserText(h.modelCalls[0]!);
    assert.match(directive, /^<goal>\n/);
    assert.match(directive, /<condition>\nall tests in test\/auth pass\n<\/condition>/);
    assert.doesNotMatch(directive, /^\/goal/);
    const continuation = h.lastUserText(h.modelCalls[2]!);
    assert.match(continuation, /^<goal_continuation round="1\/20">/);
    assert.match(continuation, /<reason>\nrefresh still fails\n<\/reason>/);
    assert.match(continuation, /all tests in test\/auth pass/);

    assert.equal(h.judgeCalls.length, 2);
    assert.match(String(h.judgeCalls[0]?.[1]?.content), /<condition>\nall tests in test\/auth pass\n<\/condition>/);
    assert.match(String(h.judgeCalls[0]?.[1]?.content), /Still running\./);
    assert.match(String(h.judgeCalls[1]?.[1]?.content), /Suite is green\./);
    assert.equal(h.judgeOptions[0]?.model, "glm-5.4");
    assert.equal(h.judgeOptions[0]?.tools, null);

    assert.match(h.texts().join("\n"), /\[goal 1\/20\] Continuing — not met yet: refresh still fails/);
    assert.match(h.texts().at(-1) ?? "", /Goal achieved after 1 continuation round: npm test exits 0/);
    const replay = h.updates.filter(update => update.update.sessionUpdate === "user_message_chunk").map(update => update.update.content?.text);
    assert.deepEqual(replay, ["[goal 1/20] Continuing — not met yet: refresh still fails"]);
    assert.match(await h.status(sessionId), GOAL_ACHIEVED);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("the goal persists, survives session/load with its counters reset, and resumes", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  const dir = tempDir("glm-goal-store-");
  try {
    const store = new SessionStore(dir);
    const first = setup({ main: [{ text: "one" }, { text: "two" }], judge: [NOT_MET("a"), NOT_MET("b")] }, { goalMaxRounds: 2 }, store);
    const { sessionId } = await first.agent.newSession({ cwd, mcpServers: [] });
    const result = await first.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal lint is clean" }] });
    assert.equal(result.stopReason, "end_turn");
    assert.match(first.texts().at(-1) ?? "", /Goal paused — round budget reached \(2\/2\)/);
    const saved = store.load(sessionId);
    assert.equal(saved?.goal?.status, "paused");
    assert.equal(saved?.goal?.pauseReason, "budget");
    assert.equal(saved?.goal?.rounds, 2);
    assert.deepEqual(saved?.goal?.tokens, summedUsage(2, 2));
    // The directive and the one continuation before the budget paused are user turns on disk.
    assert.equal(saved?.messages.filter(message => message.role === "user").length, 2);

    // A fresh process restores the session from disk.
    const second = setup({ main: [{ toolCall: true }, { text: "fixed" }], judge: [MET("ruff exits 0")] }, {}, store);
    await second.agent.loadSession({ sessionId, cwd, mcpServers: [] });
    const restored = await second.status(sessionId);
    assert.match(restored, /\*\*Goal: paused \(budget\)\*\*/);
    assert.match(restored, /- Rounds: 0\/2/);
    assert.match(restored, /- Evaluator: glm-5\.4 \(session model\)/);
    assert.match(restored, /- Tokens: 2\.4k/);
    assert.equal(second.modelCalls.length, 0);

    const resumed = await second.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.equal(resumed.stopReason, "end_turn");
    assert.equal(second.modelCalls.length, 2);
    assert.match(second.lastUserText(second.modelCalls[0]!), /^<goal_resumed round="0\/2">/);
    // The resumed round satisfied the goal on its first try.
    assert.match(second.texts().join("\n"), /Goal achieved after 0 continuation rounds: ruff exits 0/);
    assert.match(await second.status(sessionId), GOAL_ACHIEVED);
    assert.equal(store.load(sessionId)?.goal?.rounds, 0);
  } finally { rmSync(cwd, { recursive: true, force: true }); rmSync(dir, { recursive: true, force: true }); }
});

test("a fork carries the goal with its counters reset", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  const dir = tempDir("glm-goal-store-");
  try {
    const store = new SessionStore(dir);
    const h = setup({ main: [{ text: "one" }, { text: "two" }], judge: [NOT_MET("a"), NOT_MET("b")] }, { goalMaxRounds: 2 }, store);
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal lint is clean" }] });
    const fork = await h.agent.unstable_forkSession({ sessionId, cwd, mcpServers: [] });
    const forked = await h.status(fork.sessionId);
    assert.match(forked, /\*\*Goal: paused \(budget\)\*\*/);
    assert.match(forked, /- Rounds: 0\/2/);
    // The parent keeps its own accounting, even after the fork spends more.
    const forkResult = await h.agent.prompt({ sessionId: fork.sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.equal(forkResult.stopReason, "end_turn");
    assert.deepEqual(h.store!.load(fork.sessionId)?.goal?.tokens, summedUsage(4, 4));
    assert.deepEqual(h.store!.load(sessionId)?.goal?.tokens, summedUsage(2, 2));
    assert.match(await h.status(sessionId), /- Rounds: 2\/2/);
  } finally { rmSync(cwd, { recursive: true, force: true }); rmSync(dir, { recursive: true, force: true }); }
});

test("/goal pause and /goal resume steer a running goal; status and clear never reach the model", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    let release!: () => void;
    let started!: () => void;
    const running = new Promise<void>(resolve => { started = resolve; });
    const gate = new Promise<void>(resolve => { release = resolve; });
    const h = setup({
      main: [{ toolCall: true }, { text: "working" }, { toolCall: true }, { text: "done" }],
      judge: [MET("all green")],
      beforeMain: async index => { if (index === 1) { started(); await gate; } },
    });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const first = h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal all tests pass" }] });
    await running;
    // Pause preempts the running round; the superseded round leaves the goal
    // alone. The gated turn only observes its abort once it is released.
    const pausing = h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal pause" }] });
    release();
    assert.equal((await first).stopReason, "cancelled");
    assert.equal((await pausing).stopReason, "end_turn");
    assert.match(h.texts().at(-1) ?? "", /Goal paused: all tests pass/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(user\)\*\*/);
    assert.equal(h.modelCalls.length, 2);
    assert.equal(h.judgeCalls.length, 0);

    assert.match(await h.status(sessionId, "/goal pause"), /Goal already paused/);
    const resumed = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.equal(resumed.stopReason, "end_turn");
    assert.equal(h.modelCalls.length, 4);
    assert.equal(h.judgeCalls.length, 1);
    // The resumed round satisfied the goal on its first try, so no continuation was needed.
    assert.match(h.texts().join("\n"), /Goal achieved after 0 continuation rounds: all green/);

    assert.match(await h.status(sessionId, "/goal pause"), /Goal is achieved; \/goal pause applies to an active goal/);
    assert.match(await h.status(sessionId, "/goal resume"), /Goal already achieved: all tests pass/);
    assert.match(await h.status(sessionId, "/goal clear"), /Goal cleared: all tests pass/);
    assert.match(await h.status(sessionId), /No goal set/);
    assert.match(await h.status(sessionId, "/goal clear"), /No goal set/);
    assert.match(await h.status(sessionId, "/goal resume"), /No goal set\. Start one with \/goal <condition>/);
    assert.equal(h.modelCalls.length, 4);
    assert.equal(h.judgeCalls.length, 1);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("three answer-only rounds pause the goal; a tool call resets the streak", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ text: "one" }, { text: "two" }, { text: "three" }], judge: [NOT_MET("a"), NOT_MET("b"), NOT_MET("c")] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(h.modelCalls.length, 3);
    assert.match(h.texts().at(-1) ?? "", /Goal paused — 3 rounds in a row without tool calls/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(no progress\)\*\*/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }

  const cwd2 = tempDir("glm-goal-cwd-");
  try {
    const h = setup({
      main: [{ text: "one" }, { toolCall: true }, { text: "three" }, { text: "four" }, { text: "five" }],
      judge: [NOT_MET("a"), NOT_MET("b"), NOT_MET("c"), NOT_MET("d"), NOT_MET("e")],
    });
    const { sessionId } = await h.agent.newSession({ cwd: cwd2, mcpServers: [] });
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    // Rounds: "one" (streak 1), tool call + "three" (0), "four" (1), "five"
    // (2), "five" again (3). The tool round has two model calls of its own.
    assert.equal(h.modelCalls.length, 6);
    assert.match(await h.status(sessionId), /- Rounds: 5\/20/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(no progress\)\*\*/);
  } finally { rmSync(cwd2, { recursive: true, force: true }); }
});

test("cancelling a goal turn pauses the goal and /goal resume continues it", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    let started!: () => void;
    const running = new Promise<void>(resolve => { started = resolve; });
    const h = setup({
      main: [{ toolCall: true }, { text: "interrupted" }, { text: "status reply" }, { toolCall: true }, { text: "done" }],
      judge: [MET("green")],
      beforeMain: async index => { if (index === 1) { started(); await new Promise(resolve => setTimeout(resolve, 100)); } },
    });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const pending = h.agent.prompt({ sessionId, messageId: "cancel-me", prompt: [{ type: "text", text: "/goal x" }] });
    await running;
    await h.agent.cancel({ sessionId });
    const cancelled = await pending;
    assert.equal(cancelled.stopReason, "cancelled");
    assert.equal(cancelled.userMessageId, "cancel-me");
    assert.deepEqual(cancelled.usage, summedUsage(1, 0));
    assert.match(h.texts().at(-1) ?? "", /Goal paused — turn cancelled/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(cancelled\)\*\*/);
    assert.equal(h.modelCalls.length, 2);
    // A plain message while paused runs one turn and does not restart the loop.
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "status?" }] });
    assert.equal(h.modelCalls.length, 3);
    assert.equal(h.judgeCalls.length, 0);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(cancelled\)\*\*/);
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.equal(h.modelCalls.length, 5);
    assert.equal(h.judgeCalls.length, 1);
    assert.match(await h.status(sessionId), GOAL_ACHIEVED);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a newer prompt supersedes a goal round without pausing; its turn is judged and the loop continues", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    let started!: () => void;
    const running = new Promise<void>(resolve => { started = resolve; });
    const h = setup({
      main: [{ toolCall: true }, { text: "interrupted" }, { text: "answer to your message" }, { toolCall: true }, { text: "finished" }],
      judge: [NOT_MET("still not done"), MET("green")],
      beforeMain: async index => { if (index === 1) { started(); await new Promise(resolve => setTimeout(resolve, 100)); } },
    });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const superseded = h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    await running;
    const steering = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "also fix the README" }] });
    assert.equal((await superseded).stopReason, "cancelled");
    assert.equal(steering.stopReason, "end_turn");
    // The superseded round never reached its judge. The steering turn ran, was
    // judged (not met), and the goal round continued to completion.
    assert.equal(h.modelCalls.length, 5);
    assert.equal(h.judgeCalls.length, 2);
    assert.match(h.lastUserText(h.modelCalls[2]!), /also fix the README/);
    assert.match(h.lastUserText(h.modelCalls[3]!), /^<goal_continuation round="1\/20">/);
    const status = await h.status(sessionId);
    assert.match(status, GOAL_ACHIEVED);
    assert.doesNotMatch(status, /paused/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("an impossible verdict blocks the goal until re-scoped or resumed", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ text: "I cannot reach the staging DB" }], judge: ['{"verdict":"impossible","reason":"needs staging credentials"}', MET("overridden")] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal deploy to staging" }] });
    assert.match(h.texts().at(-1) ?? "", /Goal judged unachievable — paused: needs staging credentials/);
    const status = await h.status(sessionId);
    assert.match(status, /\*\*Goal: blocked\*\*/);
    assert.match(status, /- Last check: impossible — needs staging credentials/);
    assert.match(status, /\/goal resume to override/);
    assert.match(await h.status(sessionId, "/goal pause"), /Goal is blocked; \/goal pause applies to an active goal/);
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.match(await h.status(sessionId), GOAL_ACHIEVED);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("evaluator failure pauses the goal after one retry", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ text: "done?" }], judge: [new Error("503 upstream"), new Error("503 upstream")] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const result = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(result.stopReason, "end_turn");
    assert.equal(h.modelCalls.length, 1);
    assert.equal(h.judgeCalls.length, 2);
    assert.match(h.texts().at(-1) ?? "", /Goal paused — evaluator unavailable: 503 upstream/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(judge unavailable\)\*\*/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a round that stops on its turn limit pauses the goal with that reason", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ toolCall: true }, { text: "done" }], judge: [MET("green")] }, { maxTurns: 1 });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const result = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(result.stopReason, "max_turn_requests");
    assert.match(h.texts().at(-1) ?? "", /Goal paused — the round hit its turn limit \(1\)\./);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(turn limit\)\*\*/);
    assert.equal((await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] })).stopReason, "end_turn");
    assert.match(await h.status(sessionId), GOAL_ACHIEVED);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("the evaluator model comes from --judge, then the environment, then the session", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const flagged = setup({ main: [{ text: "ok" }], judge: [MET("yes")] });
    const { sessionId } = await flagged.agent.newSession({ cwd, mcpServers: [] });
    await flagged.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal --judge glm-4.7 all tests pass" }] });
    assert.equal(flagged.judgeOptions[0]?.model, "glm-4.7");
    assert.equal(flagged.judgeOptions[0]?.reasoningEffort, "on");
    assert.match(flagged.lastUserText(flagged.modelCalls[0]!), /<condition>\nall tests pass\n<\/condition>/);
    // The provider's own answer about which model judged is part of the status.
    assert.match(await flagged.status(sessionId), /- Evaluator: glm-4\.7 \(last verdict answered by glm-4\.7\)\n/);

    process.env["ACP_GLM_GOAL_JUDGE_MODEL"] = "glm-5.3-flash";
    try {
      const env = setup({ main: [{ text: "ok" }], judge: [MET("yes"), MET("yes")] });
      const { sessionId: envSession } = await env.agent.newSession({ cwd, mcpServers: [] });
      await env.agent.prompt({ sessionId: envSession, prompt: [{ type: "text", text: "/goal x" }] });
      assert.equal(env.judgeOptions[0]?.model, "glm-5.3-flash");
      assert.equal(env.judgeOptions[0]?.reasoningEffort, "low");
      // The flag wins over the environment.
      await env.agent.prompt({ sessionId: envSession, prompt: [{ type: "text", text: "/goal clear" }] });
      await env.agent.prompt({ sessionId: envSession, prompt: [{ type: "text", text: "/goal --judge=glm-5.3 y" }] });
      assert.equal(env.judgeOptions[1]?.model, "glm-5.3");
      assert.match(await env.status(envSession), /- Evaluator: glm-5\.3 \(last verdict answered by glm-5\.3\)\n/);
    } finally { delete process.env["ACP_GLM_GOAL_JUDGE_MODEL"]; }

    const sessionModel = setup({ main: [{ text: "ok" }], judge: [MET("yes")] });
    const { sessionId: sessionModelId } = await sessionModel.agent.newSession({ cwd, mcpServers: [] });
    await sessionModel.agent.prompt({ sessionId: sessionModelId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(sessionModel.judgeOptions[0]?.model, "glm-5.4");
    assert.equal(sessionModel.judgeOptions[0]?.reasoningEffort, "minimal");
    assert.match(await sessionModel.status(sessionModelId), /- Evaluator: glm-5\.4 \(session model\)/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("invalid goals are answered without a model call; a control word followed by words is a condition", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ text: "ok" }], judge: [MET("yes")] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    assert.match(await h.status(sessionId, "/goal --rounds 5 x"), /Unknown \/goal option/);
    assert.match(await h.status(sessionId, `/goal ${"x".repeat(4_001)}`), /limit is 4000/);
    assert.match(await h.status(sessionId, "/goal --judge glm-5.3-flash"), /A goal needs a condition/);
    assert.match(await h.status(sessionId), /No goal set/);
    assert.equal(h.modelCalls.length, 0);
    assert.equal(h.judgeCalls.length, 0);
    // "pause" followed by words is a condition, not a pause.
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal pause after verification" }] });
    assert.match(h.lastUserText(h.modelCalls[0]!), /<condition>\npause after verification\n<\/condition>/);
    assert.match(await h.status(sessionId), GOAL_ACHIEVED);
    assert.match(await h.status(sessionId, "/goal pause"), /Goal is achieved; \/goal pause applies to an active goal/);
    // Control words are case-insensitive and accept the clear aliases.
    assert.match(await h.status(sessionId, "/goal STOP"), /Goal cleared: pause after verification/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a goal round stops on the agent's turn limit unless the goal raises or removes it", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    // Five tool turns, then a text turn; the agent default here is 2, so a
    // plain goal trips the limit before the model ever gets to "done".
    const script = () => ({
      main: [{ toolCall: true }, { toolCall: true }, { toolCall: true }, { toolCall: true }, { toolCall: true }, { text: "done" }],
      judge: [MET("green")],
    });
    const plain = setup(script(), { maxTurns: 2 });
    const { sessionId } = await plain.agent.newSession({ cwd, mcpServers: [] });
    const result = await plain.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(result.stopReason, "max_turn_requests");
    assert.equal(plain.modelCalls.length, 2);
    assert.match(plain.texts().join("\n"), /\[stopped: reached the 2-turn limit/);
    assert.match(plain.texts().at(-1) ?? "", /Goal paused — the round hit its turn limit \(2\)\./);
    const status = await plain.status(sessionId);
    assert.match(status, /\*\*Goal: paused \(turn limit\)\*\*/);
    assert.match(status, /- Turn limit: 2 per round/);

    const raised = setup(script(), { maxTurns: 2 });
    const { sessionId: raisedSession } = await raised.agent.newSession({ cwd, mcpServers: [] });
    const raisedResult = await raised.agent.prompt({ sessionId: raisedSession, prompt: [{ type: "text", text: "/goal --turns 6 x" }] });
    assert.equal(raisedResult.stopReason, "end_turn");
    assert.equal(raised.modelCalls.length, 6);
    assert.match(await raised.status(raisedSession), /- Turn limit: 6 per round/);
    assert.match(await raised.status(raisedSession), GOAL_ACHIEVED);

    const unlimited = setup(script(), { maxTurns: 2 });
    const { sessionId: unlimitedSession } = await unlimited.agent.newSession({ cwd, mcpServers: [] });
    const unlimitedResult = await unlimited.agent.prompt({ sessionId: unlimitedSession, prompt: [{ type: "text", text: "/goal --turns unlimited x" }] });
    assert.equal(unlimitedResult.stopReason, "end_turn");
    assert.equal(unlimited.modelCalls.length, 6);
    assert.match(await unlimited.status(unlimitedSession), /- Turn limit: unlimited per round/);

    // /goal resume keeps the goal's own (default) limit, so it trips again.
    const resumed = await plain.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.equal(resumed.stopReason, "max_turn_requests");
    assert.equal(plain.modelCalls.length, 4);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("ACP_GLM_GOAL_MAX_TURNS sets the agent-wide goal turn limit; --turns still wins", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    process.env["ACP_GLM_GOAL_MAX_TURNS"] = "6";
    try {
      const h = setup({
        main: [{ toolCall: true }, { toolCall: true }, { toolCall: true }, { toolCall: true }, { toolCall: true }, { text: "done" }],
        judge: [MET("green"), MET("green")],
      }, { maxTurns: 2 });
      const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
      assert.equal((await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] })).stopReason, "end_turn");
      assert.equal(h.modelCalls.length, 6);
      assert.match(await h.status(sessionId), /- Turn limit: 6 per round/);
      // A lower per-goal limit still wins over the environment.
      const lowered = setup({
        main: [{ toolCall: true }, { toolCall: true }, { text: "done" }],
        judge: [MET("green")],
      }, { maxTurns: 2 });
      const { sessionId: loweredSession } = await lowered.agent.newSession({ cwd, mcpServers: [] });
      const loweredResult = await lowered.agent.prompt({ sessionId: loweredSession, prompt: [{ type: "text", text: "/goal --turns 1 x" }] });
      assert.equal(loweredResult.stopReason, "max_turn_requests");
      assert.equal(lowered.modelCalls.length, 1);
      assert.match(await lowered.status(loweredSession), /- Turn limit: 1 per round/);
    } finally { delete process.env["ACP_GLM_GOAL_MAX_TURNS"]; }
    // ...and ordinary prompts keep the agent's own limit, whatever goals use.
    const plain = setup({ main: [{ toolCall: true }, { text: "done" }], judge: [] }, { maxTurns: 2 });
    const { sessionId } = await plain.agent.newSession({ cwd, mcpServers: [] });
    assert.equal((await plain.agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] })).stopReason, "end_turn");
    assert.equal(plain.modelCalls.length, 2);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("--judge is validated against the advertised models and the id is canonicalised", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ text: "ok" }], judge: [MET("yes"), MET("yes")] });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    // A bogus id is rejected with the list, before any model call; nothing is set.
    const rejected = await h.status(sessionId, "/goal --judge umbrella fix the build");
    assert.match(rejected, /Unknown evaluator model "umbrella"\. Advertised models: glm-5\.4, glm-5\.3, glm-5\.3-flash, glm-5-turbo, glm-4\.7/);
    assert.match(rejected, /ACP_GLM_AVAILABLE_MODELS/);
    assert.match(await h.status(sessionId), /No goal set/);
    assert.equal(h.modelCalls.length, 0);
    assert.equal(h.judgeCalls.length, 0);
    // A multi-word trailing value is still a (bad) model name, never goal text.
    const uncleBob = await h.status(sessionId, "/goal test 2+2 --judge Uncle Bob");
    assert.match(uncleBob, /Unknown evaluator model "Uncle Bob"\./);
    assert.match(await h.status(sessionId), /No goal set/);
    // Typed after the condition, in a different case: the flag is honoured and the id canonicalised.
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal fix the build --judge GLM-5.3-Flash" }] });
    assert.equal(h.judgeOptions[0]?.model, "glm-5.3-flash");
    assert.equal(h.judgeOptions[0]?.reasoningEffort, "low");
    assert.match(h.lastUserText(h.modelCalls[0]!), /<condition>\nfix the build\n<\/condition>/);
    assert.match(await h.status(sessionId), /- Evaluator: glm-5\.3-flash \(last verdict answered by glm-5\.3-flash\)\n/);
    // ACP_GLM_AVAILABLE_MODELS extends what --judge accepts.
    process.env["ACP_GLM_AVAILABLE_MODELS"] = "glm-5.4,glm-5.3,glm-5.3-flash,glm-5-turbo,glm-4.7,glm-5.2";
    try {
      await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal fix it --judge glm-5.2" }] });
      assert.equal(h.judgeOptions[1]?.model, "glm-5.2");
    } finally { delete process.env["ACP_GLM_AVAILABLE_MODELS"]; }
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("an invalid ACP_GLM_GOAL_JUDGE_MODEL is ignored and the session model judges", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    process.env["ACP_GLM_GOAL_JUDGE_MODEL"] = "umbrella";
    try {
      const h = setup({ main: [{ text: "ok" }], judge: [MET("yes")] });
      const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
      await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
      assert.equal(h.judgeOptions[0]?.model, "glm-5.4");
      assert.match(await h.status(sessionId), /- Evaluator: glm-5\.4 \(session model\)/);
    } finally { delete process.env["ACP_GLM_GOAL_JUDGE_MODEL"]; }
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a dismissed question form during a goal round pauses the goal by name", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    const h = setup({ main: [{ ask: true }, { text: "unused" }], judge: [MET("yes")] });
    await h.agent.initialize({ protocolVersion: 1, clientCapabilities: { elicitation: { form: {} } } });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const result = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal migrate the database" }] });
    // #154 turns the dismissed form into a cancelled round without an abort;
    // the goal loop names the cause instead of a bare "stop reason cancelled".
    assert.equal(result.stopReason, "cancelled");
    assert.equal(h.modelCalls.length, 1);
    assert.equal(h.judgeCalls.length, 0);
    assert.match(h.texts().at(-1) ?? "", /Goal paused — you dismissed the agent's question\.\nAnswer in a message and send \/goal resume to continue/);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(question dismissed\)\*\*/);
    // Resuming asks the model to continue; the directive sanctions the tool.
    await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal resume" }] });
    assert.match(h.lastUserText(h.modelCalls[1]!), /use ask_user_question/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a declined question is not progress: repeated declines stop after three rounds", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    // No elicitation capability: the question renders as permission buttons and
    // the client's dismissed prompt is a Decline, which completes the call.
    // Each round: ask once, get the decline, end the turn.
    const h = setup({
      main: [{ ask: true }, { text: "still blocked" }, { ask: true }, { text: "still blocked" }, { ask: true }, { text: "still blocked" }],
      judge: [NOT_MET("still blocked"), NOT_MET("still blocked"), NOT_MET("still blocked")],
    });
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    const result = await h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal migrate the database" }] });
    assert.equal(result.stopReason, "end_turn");
    assert.equal(h.modelCalls.length, 6);
    assert.equal(h.judgeCalls.length, 3);
    assert.match(await h.status(sessionId), /\*\*Goal: paused \(no progress\)\*\*/);
    assert.match(await h.status(sessionId), /- Rounds: 3\/20/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a round that dies on a thrown error pauses the goal; a plain message does not restart the loop", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  try {
    // A dropped connection in the first continuation round: prompt() recovers
    // it as a cancelled turn, and the goal must not be left active.
    const dropped = setup({
      main: [{ text: "one" }, { text: "two" }, { text: "three" }],
      judge: [NOT_MET("keep going"), MET("green")],
      mainError: index => (index === 1 ? Object.assign(new Error("socket hang up"), { code: "ECONNRESET" }) : undefined),
    });
    const { sessionId } = await dropped.agent.newSession({ cwd, mcpServers: [] });
    const result = await dropped.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    assert.equal(result.stopReason, "cancelled");
    assert.match(dropped.texts().join("\n"), /Connection lost/);
    assert.match(dropped.texts().join("\n"), /Goal paused — send \/goal resume to continue/);
    assert.match(await dropped.status(sessionId), /\*\*Goal: paused \(error\)\*\*/);
    // The next ordinary message runs one turn and is not judged.
    await dropped.agent.prompt({ sessionId, prompt: [{ type: "text", text: "hello" }] });
    assert.equal(dropped.modelCalls.length, 3);
    assert.equal(dropped.judgeCalls.length, 1);
    assert.match(await dropped.status(sessionId), /\*\*Goal: paused \(error\)\*\*/);

    // A non-recoverable error rejects the prompt and still pauses the goal.
    const boom = setup({
      main: [{ text: "one" }, { text: "two" }],
      judge: [NOT_MET("keep going")],
      mainError: index => (index === 1 ? new Error("boom") : undefined),
    });
    const { sessionId: boomSession } = await boom.agent.newSession({ cwd, mcpServers: [] });
    await assert.rejects(boom.agent.prompt({ sessionId: boomSession, prompt: [{ type: "text", text: "/goal x" }] }), /boom/);
    assert.match(boom.texts().join("\n"), /\[error\] boom\n\nGoal paused — send \/goal resume to continue/);
    assert.match(await boom.status(boomSession), /\*\*Goal: paused \(error\)\*\*/);
  } finally { rmSync(cwd, { recursive: true, force: true }); }
});

test("a cancelled later round keeps the completed rounds before it, even after compaction", async () => {
  const cwd = tempDir("glm-goal-cwd-");
  const dir = tempDir("glm-goal-store-");
  try {
    const store = new SessionStore(dir);
    let started!: () => void;
    const running = new Promise<void>(resolve => { started = resolve; });
    const h = setup({
      main: [{ toolCall: true }, { text: "round one done" }, { text: "interrupted" }],
      judge: [NOT_MET("keep going")],
      beforeMain: async index => { if (index === 2) { started(); await new Promise(resolve => setTimeout(resolve, 100)); } },
    }, {}, store);
    const { sessionId } = await h.agent.newSession({ cwd, mcpServers: [] });
    // A long history on a 128K model compacts at the start of the first round,
    // which installs a rollback checkpoint for that prompt.
    await h.agent.unstable_setSessionModel({ sessionId, modelId: "glm-5-turbo" });
    const session = (h.agent as unknown as { sessions: Map<string, { messages: GlmMessage[] }> }).sessions.get(sessionId)!;
    for (let i = 0; i < 40; i++) {
      session.messages.push({ role: "user", content: "Very long message filler ".repeat(1000) });
      session.messages.push({ role: "assistant", content: "Intermediate response filler ".repeat(1000) });
    }
    const pending = h.agent.prompt({ sessionId, prompt: [{ type: "text", text: "/goal x" }] });
    await running;
    assert.equal(h.modelCalls.length, 3, "round one ran to completion before the gated second round");
    assert.ok(h.modelCalls[0]!.length < session.messages.length + 40, "the first request was compacted");
    await h.agent.cancel({ sessionId });
    const result = await pending;
    assert.equal(result.stopReason, "cancelled");
    // Usage from the finished round and its evaluation survives on the response.
    assert.deepEqual(result.usage, summedUsage(2, 1));
    // Round one's output is still in the saved history: the checkpoint taken
    // before it was dropped once the round completed.
    const saved = store.load(sessionId);
    assert.ok(saved?.messages.some(message => message.role === "assistant" && message.content === "round one done"), "round one's reply is kept");
    assert.ok(saved?.messages.some(message => message.role === "tool"), "round one's tool result is kept");
    assert.equal(saved?.goal?.pauseReason, "cancelled");
  } finally { rmSync(cwd, { recursive: true, force: true }); rmSync(dir, { recursive: true, force: true }); }
});
