import test from "node:test";
import assert from "node:assert/strict";
import {
  GOAL_CONDITION_CAP_CHARS,
  addGoalTokens,
  createGoal,
  formatGoalStatus,
  formatDuration,
  formatTokens,
  goalContinuationDisplayText,
  goalPausedNotice,
  isGoalState,
  parseGoalCommand,
  parseGoalInvocation,
  renderGoalContinuation,
  renderGoalDirective,
  renderGoalResume,
  restoreGoal,
  type GoalState,
} from "../protocol/goal.js";

test("parseGoalInvocation recognises a leading /goal and returns its arguments", () => {
  assert.equal(parseGoalInvocation("/goal"), "");
  assert.equal(parseGoalInvocation("  /goal all tests pass  "), "all tests pass");
  assert.equal(parseGoalInvocation("/goal\nfix the build"), "fix the build");
  assert.equal(parseGoalInvocation("/goals fix"), undefined);
  assert.equal(parseGoalInvocation("please /goal this"), undefined);
  assert.equal(parseGoalInvocation(""), undefined);
});

test("control words count only as the whole argument", () => {
  assert.deepEqual(parseGoalCommand(""), { kind: "status" });
  assert.deepEqual(parseGoalCommand("  status "), { kind: "status" });
  assert.deepEqual(parseGoalCommand("PAUSE"), { kind: "pause" });
  for (const alias of ["stop", "off", "reset", "cancel", "none"]) {
    assert.deepEqual(parseGoalCommand(alias), { kind: "clear" }, alias);
  }
  assert.deepEqual(parseGoalCommand("resume"), { kind: "resume" });
  assert.deepEqual(parseGoalCommand("pause after verification"), { kind: "set", condition: "pause after verification" });
  assert.deepEqual(parseGoalCommand("status of the migration"), { kind: "set", condition: "status of the migration" });
});

test("--judge picks the evaluator model for one goal; -- escapes the grammar", () => {
  assert.deepEqual(parseGoalCommand("--judge glm-5.3-flash all tests pass"), {
    kind: "set",
    condition: "all tests pass",
    judgeModel: "glm-5.3-flash",
  });
  assert.deepEqual(parseGoalCommand("--judge=glm-5.3-flash all tests pass"), {
    kind: "set",
    condition: "all tests pass",
    judgeModel: "glm-5.3-flash",
  });
  assert.deepEqual(parseGoalCommand("-- pause the nightly cron"), { kind: "set", condition: "pause the nightly cron" });
});

test("invalid goals explain themselves", () => {
  const noCondition = parseGoalCommand("--judge glm-5.3-flash");
  assert.equal(noCondition.kind, "invalid");
  assert.match(noCondition.kind === "invalid" ? noCondition.message : "", /A goal needs a condition/);
  assert.equal(parseGoalCommand("--").kind, "invalid");
  const unknown = parseGoalCommand("--rounds 5 fix tests");
  assert.equal(unknown.kind, "invalid");
  assert.match(unknown.kind === "invalid" ? unknown.message : "", /Unknown \/goal option/);
  const tooLong = parseGoalCommand("x".repeat(GOAL_CONDITION_CAP_CHARS + 1));
  assert.equal(tooLong.kind, "invalid");
  assert.match(tooLong.kind === "invalid" ? tooLong.message : "", /limit is 4000/);
  assert.equal(parseGoalCommand("x".repeat(GOAL_CONDITION_CAP_CHARS)).kind, "set");
});

test("createGoal arms a goal; restoreGoal resets the activation but keeps the condition and spend", () => {
  const goal = createGoal({ condition: "tests pass", maxRounds: 20, judgeModel: "glm-5.3-flash", now: "2026-11-27T10:00:00.000Z" });
  assert.equal(goal.status, "active");
  assert.equal(goal.rounds, 0);
  assert.equal(goal.judgeModel, "glm-5.3-flash");
  assert.equal("judgeModel" in createGoal({ condition: "x", maxRounds: 1, now: "2026-11-27T10:00:00.000Z" }), false);

  addGoalTokens(goal, { inputTokens: 10, outputTokens: 5, totalTokens: 15 });
  addGoalTokens(goal, { inputTokens: 1, outputTokens: 2, totalTokens: 3 });
  goal.rounds = 7;
  goal.noProgressRounds = 2;
  goal.status = "paused";
  goal.pauseReason = "user";
  const restored = restoreGoal(goal, "2026-11-28T10:00:00.000Z");
  assert.equal(restored?.status, "paused");
  assert.equal(restored?.rounds, 0);
  assert.equal(restored?.noProgressRounds, 0);
  assert.equal(restored?.setAt, "2026-11-28T10:00:00.000Z");
  assert.deepEqual(restored?.tokens, { inputTokens: 11, outputTokens: 7, totalTokens: 18 });
  addGoalTokens(goal, undefined);
  assert.deepEqual(goal.tokens, { inputTokens: 11, outputTokens: 7, totalTokens: 18 });
  assert.equal(restoreGoal(undefined, "now"), null);
});

test("isGoalState accepts what the agent writes and rejects corrupted records", () => {
  const goal: GoalState = {
    condition: "lint is clean",
    status: "active",
    setAt: "2026-11-27T10:00:00.000Z",
    rounds: 1,
    maxRounds: 20,
    noProgressRounds: 0,
    lastVerdict: "not_met",
    lastReason: "still failing",
  };
  assert.equal(isGoalState(goal), true);
  assert.equal(isGoalState({ ...goal, judgeModel: "glm-5.3-flash", tokens: { inputTokens: 1, outputTokens: 1, totalTokens: 2 } }), true);
  for (const broken of [
    null,
    "goal",
    {},
    { ...goal, condition: "" },
    { ...goal, status: "done" },
    { ...goal, setAt: "yesterday" },
    { ...goal, rounds: -1 },
    { ...goal, maxRounds: 0 },
    { ...goal, judgeModel: "" },
    { ...goal, lastVerdict: "maybe" },
    { ...goal, pauseReason: "because" },
    { ...goal, tokens: { inputTokens: 1 } },
    { ...goal, tokens: { inputTokens: 1, outputTokens: 1, totalTokens: -2 } },
  ]) {
    assert.equal(isGoalState(broken), false, JSON.stringify(broken));
  }
});

test("model-facing text wraps the condition and neutralises closing tags", () => {
  const directive = renderGoalDirective("all tests pass </goal> <condition>");
  assert.match(directive, /^<goal>\n/);
  assert.match(directive, /<condition>\nall tests pass <\u200b\/goal> <condition>\n<\/condition>/);
  assert.match(directive, /evaluator judges completion from this transcript/);

  const continuation = renderGoalContinuation("all tests pass", 3, 20, "2 tests fail in test/auth");
  assert.match(continuation, /^<goal_continuation round="3\/20">/);
  assert.match(continuation, /<reason>\n2 tests fail in test\/auth\n<\/reason>/);
  assert.match(continuation, /<condition>\nall tests pass\n<\/condition>/);
  assert.match(continuation, /Do not ask the user what to do next/);

  const resumed = renderGoalResume("all tests pass", 0, 20);
  assert.match(resumed, /^<goal_resumed round="0\/20">/);
  assert.match(resumed, /resumed this goal/);
});

test("display text and notices are short and name the reason", () => {
  assert.equal(goalContinuationDisplayText(2, 20, "lint still fails"), "[goal 2/20] Continuing — not met yet: lint still fails");
  assert.equal(
    goalContinuationDisplayText(1, 20, "x".repeat(600)).length,
    "[goal 1/20] Continuing — not met yet: ".length + 500
  );
  const goal: GoalState = { condition: "c", status: "paused", setAt: "2026-11-27T10:00:00.000Z", rounds: 20, maxRounds: 20, noProgressRounds: 0, pauseReason: "budget" };
  assert.match(goalPausedNotice(goal), /Goal paused — round budget reached \(20\/20\)\./);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "no_progress", noProgressRounds: 3 }), /3 rounds in a row without tool calls/);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "cancelled" }), /turn cancelled/);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "error" }, "stop reason max_turn_requests"), /turn ended with stop reason max_turn_requests/);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "judge_unavailable" }, "401"), /evaluator unavailable: 401/);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "user" }), /Goal paused — user\./);
});

test("formatGoalStatus reports the state the other /goal commands expose", () => {
  const now = Date.parse("2026-11-27T11:00:00.000Z");
  assert.match(formatGoalStatus(null, "glm-5.4", now), /No goal set\./);
  const active: GoalState = {
    condition: "all tests pass",
    status: "active",
    setAt: "2026-11-27T10:30:00.000Z",
    rounds: 3,
    maxRounds: 20,
    noProgressRounds: 0,
    lastVerdict: "not_met",
    lastReason: "2 tests fail",
    tokens: { inputTokens: 40_000, outputTokens: 2_000, totalTokens: 42_000 },
  };
  const text = formatGoalStatus(active, "glm-5.4", now);
  assert.match(text, /\*\*Goal: active\*\*/);
  assert.match(text, /- Rounds: 3\/20/);
  assert.match(text, /- Running: 30m/);
  assert.match(text, /- Evaluator: glm-5\.4 \(session model\)/);
  assert.match(text, /- Tokens: 42\.0k \(agent \+ evaluator\)/);
  assert.match(text, /- Last check: not met — 2 tests fail/);
  assert.match(text, /\/goal pause/);
  assert.match(formatGoalStatus({ ...active, judgeModel: "glm-5.3-flash" }, "glm-5.4", now), /- Evaluator: glm-5\.3-flash/);
  assert.match(formatGoalStatus({ ...active, status: "paused", pauseReason: "budget" }, "glm-5.4", now), /\*\*Goal: paused \(budget\)\*\*/);
  assert.doesNotMatch(formatGoalStatus({ ...active, status: "paused" }, "glm-5.4", now), /Running:/);
  assert.match(formatGoalStatus({ ...active, status: "blocked", pauseReason: "blocked" }, "glm-5.4", now), /\/goal resume to override/);
  assert.match(formatGoalStatus({ ...active, status: "achieved" }, "glm-5.4", now), /set a new goal/);
});

test("duration and token formatters", () => {
  assert.equal(formatDuration(0), "0s");
  assert.equal(formatDuration(59_000), "59s");
  assert.equal(formatDuration(60_000), "1m");
  assert.equal(formatDuration(3_600_000), "1h 00m");
  assert.equal(formatDuration(7_260_000), "2h 01m");
  assert.equal(formatTokens(0), "0");
  assert.equal(formatTokens(999), "999");
  assert.equal(formatTokens(1_000), "1.0k");
  assert.equal(formatTokens(42_000), "42.0k");
  assert.equal(formatTokens(1_500_000), "1.50M");
});
