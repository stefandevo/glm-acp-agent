import test from "node:test";
import assert from "node:assert/strict";
import {
  GOAL_CONDITION_CAP_CHARS,
  addGoalTokens,
  createGoal,
  formatGoalStatus,
  formatTurnLimit,
  formatVerdict,
  formatDuration,
  formatTokens,
  goalContinuationDisplayText,
  goalPausedNotice,
  isGoalState,
  mergeUsage,
  parseGoalCommand,
  parseGoalInvocation,
  parseTurnLimit,
  renderGoalContinuation,
  renderGoalDirective,
  renderGoalInterrupted,
  renderGoalResume,
  goalInterruptedDisplayText,
  restoreGoal,
  resumeGoalState,
  turnLimitForLoop,
  type GoalState,
} from "../protocol/goal.js";
import { isDeclinedAskResult } from "../tools/ask-question.js";

test("parseGoalInvocation recognises a leading /goal and returns its arguments", () => {
  assert.equal(parseGoalInvocation("/goal"), "");
  assert.equal(parseGoalInvocation("  /goal all tests pass  "), "all tests pass");
  assert.equal(parseGoalInvocation("/goal\nfix the build"), "fix the build");
  assert.equal(parseGoalInvocation("/goals fix"), undefined);
  assert.equal(parseGoalInvocation("please /goal this"), undefined);
  assert.equal(parseGoalInvocation(""), undefined);
});

test("control words are recognised case-insensitively, with aliases; followed by words they are an error", () => {
  assert.deepEqual(parseGoalCommand(""), { kind: "status" });
  assert.deepEqual(parseGoalCommand("  status "), { kind: "status" });
  assert.deepEqual(parseGoalCommand("PAUSE"), { kind: "pause" });
  for (const alias of ["stop", "off", "reset", "cancel", "none"]) {
    assert.deepEqual(parseGoalCommand(alias), { kind: "clear" }, alias);
  }
  assert.deepEqual(parseGoalCommand("resume"), { kind: "resume" });
  assert.match(invalidMessage("pause after verification"), /\/goal pause takes no arguments/);
  assert.match(invalidMessage("status of the migration"), /\/goal status takes no arguments/);
  assert.match(invalidMessage("clear the debt"), /\/goal clear takes no arguments/);
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

/** Parse `args`, assert it was rejected, and return the message. */
function invalidMessage(args: string): string {
  const parsed = parseGoalCommand(args);
  assert.equal(parsed.kind, "invalid", args);
  return parsed.kind === "invalid" ? parsed.message : "";
}

test("--turns sets the per-round turn limit; it composes with --judge in either order", () => {
  assert.deepEqual(parseGoalCommand("--turns 500 fix the build"), { kind: "set", condition: "fix the build", turnLimit: 500 });
  assert.deepEqual(parseGoalCommand("--turns=unlimited fix the build"), { kind: "set", condition: "fix the build", turnLimit: "unlimited" });
  assert.deepEqual(parseGoalCommand("--turns NONE fix the build"), { kind: "set", condition: "fix the build", turnLimit: "unlimited" });
  assert.deepEqual(parseGoalCommand("--turns 300 --judge glm-5.3-flash fix the build"), {
    kind: "set",
    condition: "fix the build",
    judgeModel: "glm-5.3-flash",
    turnLimit: 300,
  });
  assert.deepEqual(parseGoalCommand("--judge glm-5.3-flash --turns=unlimited fix the build"), {
    kind: "set",
    condition: "fix the build",
    judgeModel: "glm-5.3-flash",
    turnLimit: "unlimited",
  });
  for (const bad of ["--turns 0 x", "--turns abc x", "--turns -5 x", "--turns 1.5 x", "--turns x"]) {
    const parsed = parseGoalCommand(bad);
    assert.equal(parsed.kind, "invalid", bad);
    assert.match(parsed.kind === "invalid" ? parsed.message : "", /--turns needs a positive number or "unlimited"/);
  }
  assert.match(invalidMessage("--judge"), /--judge needs a value/);
  assert.match(invalidMessage("--turns"), /--turns needs a value/);
  assert.match(invalidMessage("--rounds 5 x"), /--turns <n\|unlimited>/);
});

test("resume takes --turns and --judge; other control words take nothing; nothing becomes a goal named 'resume'", () => {
  assert.deepEqual(parseGoalCommand("resume"), { kind: "resume" });
  assert.deepEqual(parseGoalCommand("resume --turns 500"), { kind: "resume", turnLimit: 500 });
  assert.deepEqual(parseGoalCommand("RESUME --judge=glm-5.3-flash"), { kind: "resume", judgeModel: "glm-5.3-flash" });
  assert.deepEqual(parseGoalCommand("resume --judge glm-5.3-flash --turns=unlimited"), { kind: "resume", judgeModel: "glm-5.3-flash", turnLimit: "unlimited" });
  assert.deepEqual(parseGoalCommand("resume --turns unlimited --judge glm-5.3"), { kind: "resume", judgeModel: "glm-5.3", turnLimit: "unlimited" });
  // Invalid values are errors that name the option, never a new goal.
  assert.match(invalidMessage("resume --turns abc"), /--turns needs a positive number or "unlimited", not "abc"/);
  assert.match(invalidMessage("resume --turns 0"), /--turns needs a positive number/);
  assert.match(invalidMessage("resume --judge"), /--judge needs a value/);
  assert.match(invalidMessage("resume --turns"), /--turns needs a value/);
  assert.match(invalidMessage("resume --rounds 5"), /Unknown \/goal option "--rounds"/);
  assert.match(invalidMessage("resume fix it"), /\/goal resume takes only --turns <n\|unlimited> and --judge <model>, not "fix it"\. Use \/goal <condition> to set a new goal/);
  assert.match(invalidMessage("resume -- x"), /takes only --turns/);
  // Other control words take no arguments at all, and still parse alone.
  for (const word of ["pause", "status", "show", "clear", "stop", "off", "reset", "cancel", "none"]) {
    assert.match(invalidMessage(`${word} now`), new RegExp(`\\/goal ${word} takes no arguments`));
    assert.notEqual(parseGoalCommand(word).kind, "invalid");
    assert.notEqual(parseGoalCommand(word).kind, "set");
  }
});

test("options may follow the condition; -- turns option parsing off for the remainder", () => {
  assert.deepEqual(parseGoalCommand("test 3+3 --judge umbrella"), { kind: "set", condition: "test 3+3", judgeModel: "umbrella" });
  // A trailing option's value runs to the end, so a multi-word model name is
  // a (rejected) model name rather than prose silently appended to the goal.
  assert.deepEqual(parseGoalCommand("test 2+2 --judge Uncle Bob"), { kind: "set", condition: "test 2+2", judgeModel: "Uncle Bob" });
  assert.deepEqual(parseGoalCommand("fix the build --turns 300 --judge Uncle Bob"), { kind: "set", condition: "fix the build", judgeModel: "Uncle Bob", turnLimit: 300 });
  assert.deepEqual(parseGoalCommand("--judge x -- fix --watch"), { kind: "set", condition: "fix --watch", judgeModel: "x" });
  assert.match(invalidMessage("fix it --judge="), /--judge needs a value/);
  assert.match(invalidMessage("fix it --turns three hundred"), /--turns needs a positive number/);
  assert.deepEqual(parseGoalCommand("fix the build --turns=unlimited --judge glm-5.3-flash"), {
    kind: "set",
    condition: "fix the build",
    judgeModel: "glm-5.3-flash",
    turnLimit: "unlimited",
  });
  assert.deepEqual(parseGoalCommand("--judge glm-5.3-flash fix the build --turns 300"), {
    kind: "set",
    condition: "fix the build",
    judgeModel: "glm-5.3-flash",
    turnLimit: 300,
  });
  // The parser keeps the id as typed; the agent matches it case-insensitively.
  assert.deepEqual(parseGoalCommand("--judge GLM-5.3-Flash x"), { kind: "set", condition: "x", judgeModel: "GLM-5.3-Flash" });
  // Any other `--word` is an unknown option, wherever it sits; escape with `-- `.
  assert.match(invalidMessage("make the --verbose flag work"), /Unknown \/goal option "--verbose"/);
  assert.deepEqual(parseGoalCommand("-- make the --verbose flag work"), { kind: "set", condition: "make the --verbose flag work" });
  assert.deepEqual(parseGoalCommand("-- fix -- this --judge x"), { kind: "set", condition: "fix -- this --judge x" });
  assert.match(invalidMessage("fix it --judge"), /--judge needs a value/);
  assert.match(invalidMessage("fix it --turns"), /--turns needs a value/);
  assert.match(invalidMessage("fix it --turns 0"), /--turns needs a positive number/);
  assert.match(invalidMessage("fix it --rounds 5"), /Unknown \/goal option "--rounds"/);
  assert.match(invalidMessage("fix it --watch"), /Unknown \/goal option "--watch"/);
  assert.match(invalidMessage("--watch fix it"), /Unknown \/goal option "--watch"/);
});

test("turn limits parse, validate, and resolve to the loop bound", () => {
  assert.equal(parseTurnLimit("1"), 1);
  assert.equal(parseTurnLimit("007"), 7);
  assert.equal(parseTurnLimit("unlimited"), "unlimited");
  assert.equal(parseTurnLimit("Unlimited"), "unlimited");
  for (const bad of ["", "0", "-1", "1.5", "1e3", "many", "unlimited "]) assert.equal(parseTurnLimit(bad), undefined, bad);
  assert.equal(turnLimitForLoop(undefined, 100), 100);
  assert.equal(turnLimitForLoop(500, 100), 500);
  assert.equal(turnLimitForLoop("unlimited", 100), Number.POSITIVE_INFINITY);
  assert.equal(turnLimitForLoop(undefined, "unlimited"), Number.POSITIVE_INFINITY);
  assert.equal(formatTurnLimit("unlimited"), "unlimited");
  assert.equal(formatTurnLimit(250), "250");
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
  const limited = createGoal({ condition: "x", maxRounds: 1, turnLimit: "unlimited", now: "2026-11-27T10:00:00.000Z" });
  assert.equal(limited.turnLimit, "unlimited");
  assert.equal("turnLimit" in createGoal({ condition: "x", maxRounds: 1, now: "2026-11-27T10:00:00.000Z" }), false);

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
  // The copy is independent: a live fork must not move its parent's total.
  assert.notEqual(restored?.tokens, goal.tokens);
  addGoalTokens(restored!, { inputTokens: 1, outputTokens: 1, totalTokens: 2 });
  assert.deepEqual(goal.tokens, { inputTokens: 11, outputTokens: 7, totalTokens: 18 });
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
  assert.equal(isGoalState({ ...goal, turnLimit: 500 }), true);
  assert.equal(isGoalState({ ...goal, turnLimit: "unlimited" }), true);
  assert.equal(isGoalState({ ...goal, lastJudgeModel: "glm-5.3" }), true);
  assert.equal(isGoalState({ ...goal, pauseReason: "question_dismissed" }), true);
  assert.equal(isGoalState({ ...goal, pauseReason: "awaiting_input", lastVerdict: "needs_input" }), true);
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
    { ...goal, turnLimit: 0 },
    { ...goal, turnLimit: "many" },
    { ...goal, turnLimit: 1.5 },
    { ...goal, lastJudgeModel: "" },
    { ...goal, lastJudgeModel: 7 },
    { ...goal, lastVerdict: "maybe" },
    { ...goal, pauseReason: "because" },
    { ...goal, pauseReason: "dismissed" },
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
  // #154 landed: the tool is the sanctioned way to get a decision, so the
  // "do not ask" rule must not fight it.
  assert.match(directive, /ask it with ask_user_question/);
  // Without the tool the model asks by ending its turn; the goal then waits.
  const noTool = renderGoalDirective("fix the build", false);
  assert.doesNotMatch(noTool, /ask_user_question/);
  assert.match(noTool, /end your turn with the question and wait/);
  const continuationNoTool = renderGoalContinuation("c", 1, 20, "r", false);
  assert.match(continuationNoTool, /end your turn with the question and wait for the answer/);
  assert.doesNotMatch(continuationNoTool, /ask_user_question/);
  assert.match(renderGoalResume("c", 0, 20, false), /end your turn with the question and wait/);

  const continuation = renderGoalContinuation("all tests pass", 3, 20, "2 tests fail in test/auth");
  assert.match(continuation, /^<goal_continuation round="3\/20">/);
  assert.match(continuation, /<reason>\n2 tests fail in test\/auth\n<\/reason>/);
  assert.match(continuation, /<condition>\nall tests pass\n<\/condition>/);
  assert.match(continuation, /Do not ask the user what to do next/);
  assert.match(continuation, /use ask_user_question/);
  assert.match(continuation, /declined to answer, proceed with the safest reasonable choice/);

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
  assert.match(goalPausedNotice({ ...goal, pauseReason: "turn_limit" }, "100"), /Goal paused — the round hit its turn limit \(100\)\./);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "turn_limit" }, "100"), /\/goal --turns <n\|unlimited> to set a new goal/);
  assert.match(goalPausedNotice({ ...goal, pauseReason: "awaiting_input" }, "Which DB?"), /Goal paused — the agent needs your answer — Which DB\?\nReply to continue, or \/goal clear to drop the goal\./);
  assert.equal(formatVerdict("needs_input"), "waiting for input");
  assert.match(
    goalPausedNotice({ ...goal, pauseReason: "question_dismissed" }),
    /Goal paused — you dismissed the agent's question\.\nAnswer in a message and send \/goal resume to continue, or \/goal clear to drop the goal\./
  );
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
  assert.match(text, /- Turn limit: 100 per round/);
  assert.match(formatGoalStatus({ ...active, turnLimit: 500 }, "glm-5.4", now), /- Turn limit: 500 per round/);
  assert.match(formatGoalStatus({ ...active, turnLimit: "unlimited" }, "glm-5.4", now), /- Turn limit: unlimited per round/);
  assert.match(formatGoalStatus(active, "glm-5.4", now, 40), /- Turn limit: 40 per round/);
  assert.match(text, /- Running: 30m/);
  assert.match(text, /- Evaluator: glm-5\.4 \(session model\)/);
  assert.match(text, /- Tokens: 42\.0k \(agent \+ evaluator\)/);
  assert.match(text, /- Last check: not met — 2 tests fail/);
  assert.match(text, /\/goal pause/);
  assert.match(formatGoalStatus({ ...active, judgeModel: "glm-5.3-flash" }, "glm-5.4", now), /- Evaluator: glm-5\.3-flash\n/);
  assert.match(formatGoalStatus({ ...active, lastVerdict: "needs_input", lastReason: "Which DB?" }, "glm-5.4", now), /- Last check: waiting for input — Which DB\?/);
  assert.match(
    formatGoalStatus({ ...active, judgeModel: "glm-5.2", lastJudgeModel: "glm-5.3" }, "glm-5.4", now),
    /- Evaluator: glm-5\.2 \(last verdict answered by glm-5\.3\)\n/
  );
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

test("mergeUsage sums snapshots and keeps optional counters", () => {
  assert.equal(mergeUsage(undefined, undefined), undefined);
  assert.deepEqual(mergeUsage(undefined, { inputTokens: 1, outputTokens: 2, totalTokens: 3 }), { inputTokens: 1, outputTokens: 2, totalTokens: 3 });
  assert.deepEqual(
    mergeUsage(
      { inputTokens: 1, outputTokens: 2, totalTokens: 3, cachedReadTokens: 10 },
      { inputTokens: 5, outputTokens: 7, totalTokens: 12, thoughtTokens: 4 }
    ),
    { inputTokens: 6, outputTokens: 9, totalTokens: 15, cachedReadTokens: 10, thoughtTokens: 4 }
  );
});

test("isDeclinedAskResult tells a decline from an answered or blank form", () => {
  assert.equal(isDeclinedAskResult(JSON.stringify({ answers: {}, annotations: { "Which?": { unanswered: "dismissed" } } })), true);
  assert.equal(isDeclinedAskResult(JSON.stringify({ answers: {}, annotations: { "A?": { unanswered: "dismissed" }, "B?": { unanswered: "dismissed" } } })), true);
  assert.equal(isDeclinedAskResult(JSON.stringify({ answers: { "Which?": "Postgres" } })), false);
  assert.equal(isDeclinedAskResult(JSON.stringify({ answers: {}, annotations: { "Which?": { unanswered: "skipped" } } })), false);
  assert.equal(isDeclinedAskResult(JSON.stringify({ answers: { "A?": "x" }, annotations: { "B?": { unanswered: "dismissed" } } })), false);
  assert.equal(isDeclinedAskResult("not json"), false);
});

test("an interrupted-round pickup restates the condition; resumeGoalState resets counters and keeps spend", () => {
  const pickup = renderGoalInterrupted("fix the build", "/goal status");
  assert.match(pickup, /^<goal_interrupted>\n/);
  assert.match(pickup, /interrupted by \/goal status\. Continue working toward the condition from where it left off; do not redo finished work/);
  assert.match(pickup, /<condition>\nfix the build\n<\/condition>/);
  assert.equal(goalInterruptedDisplayText("/goal status"), "[goal] Continuing after /goal status — picking up the interrupted round");
  const paused: GoalState = {
    condition: "c",
    status: "paused",
    pauseReason: "budget",
    setAt: "2026-11-27T10:00:00.000Z",
    rounds: 7,
    maxRounds: 20,
    noProgressRounds: 2,
    tokens: { inputTokens: 5, outputTokens: 1, totalTokens: 6 },
  };
  resumeGoalState(paused, "2026-11-28T10:00:00.000Z");
  assert.equal(paused.status, "active");
  assert.equal(paused.pauseReason, undefined);
  assert.equal(paused.rounds, 0);
  assert.equal(paused.noProgressRounds, 0);
  assert.equal(paused.setAt, "2026-11-28T10:00:00.000Z");
  assert.deepEqual(paused.tokens, { inputTokens: 5, outputTokens: 1, totalTokens: 6 });
});

test("options are read across line breaks: a multi-line condition never swallows its options", () => {
  // Trailing options after a condition that spans lines (the Greptile finding:
  // a `.*` scan left "--turns 500" inside the goal text and used the default).
  assert.deepEqual(parseGoalCommand("fix the build\nand update docs --turns 500"), { kind: "set", condition: "fix the build\nand update docs", turnLimit: 500 });
  // Line endings are the user's; only the option is peeled off.
  assert.deepEqual(parseGoalCommand("fix the build\r\nand docs --judge glm-5.3-flash"), { kind: "set", condition: "fix the build\r\nand docs", judgeModel: "glm-5.3-flash" });
  assert.deepEqual(parseGoalCommand("a\nb\nc --turns=unlimited --judge glm-5.3"), { kind: "set", condition: "a\nb\nc", judgeModel: "glm-5.3", turnLimit: "unlimited" });
  // Leading options, then a multi-line condition.
  assert.deepEqual(parseGoalCommand("--turns 500\nfix the build\nand docs"), { kind: "set", condition: "fix the build\nand docs", turnLimit: 500 });
  assert.deepEqual(parseGoalCommand("--judge glm-5.3-flash\nfix\n--turns 7"), { kind: "set", condition: "fix", judgeModel: "glm-5.3-flash", turnLimit: 7 });
  // The escape still wins on a later line, and an unknown option on a later
  // line is still an error rather than goal text.
  assert.deepEqual(parseGoalCommand("-- fix\n--this --that"), { kind: "set", condition: "fix\n--this --that" });
  assert.match(invalidMessage("fix the build\nand docs --rounds 5"), /Unknown \/goal option "--rounds"/);
  // A trailing multi-word value on a later line is still a (bad) model name
  // for the agent to reject, never goal text.
  assert.deepEqual(parseGoalCommand("first line\n--judge Uncle Bob"), { kind: "set", condition: "first line", judgeModel: "Uncle Bob" });
});

test("invariant: no option survives into a condition unless `-- ` turned parsing off", () => {
  // Every `--word` in the input is either consumed as an option, reported in
  // an error, or explicitly escaped. This is the class of bug behind
  // "--judge Uncle Bob" and the multi-line swallow: an option silently
  // becoming goal text. Any new sample must keep that true.
  const samples = [
    "fix it", "fix it --turns 500", "--turns 500 fix it", "fix it --judge glm-5.3-flash", "fix\nit --turns 500",
    "--judge glm-5.3-flash fix\nit --turns unlimited", "fix --turns=7 --judge=glm-5.3 it", "resume --turns 500", "status",
    "fix it --watch", "--watch fix it", "fix\n--judge Uncle Bob", "pause now", "-- fix --this", "--turns 1 -- fix --that",
    "a\r\nb --turns 2", "x --turns abc", "y --judge", "z --turns", "make the --verbose flag work", "make the --verbose flag\nwork",
  ];
  const optionInText = /(^|\s)--[A-Za-z]/;
  for (const sample of samples) {
    const parsed = parseGoalCommand(sample);
    const escaped = /^--(?:\s|$)/.test(sample.trim()) || /(^|\s)--\s/.test(sample);
    if (parsed.kind === "set") {
      assert.equal(optionInText.test(parsed.condition), escaped, `${JSON.stringify(sample)}: condition ${JSON.stringify(parsed.condition)}`);
    } else if (parsed.kind === "resume") {
      assert.equal(optionInText.test(JSON.stringify(parsed)), false, sample);
    } else if (parsed.kind === "invalid") {
      assert.ok(parsed.message.length > 0 && parsed.message.includes("Usage:"), sample);
    } else {
      assert.ok(["status", "pause", "clear"].includes(parsed.kind), `${sample}: ${parsed.kind}`);
    }
  }
});
