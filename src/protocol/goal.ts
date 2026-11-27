/**
 * Session goal — the state, command grammar, and prompt text behind `/goal`.
 *
 * `/goal <condition>` pins one objective for the session. The agent works on
 * it across turns: after each turn an evaluator decides whether the condition
 * holds (see `goal-evaluator.ts`), and while it does not, the agent feeds a
 * continuation message back into the same prompt instead of returning to the
 * user. This mirrors the `/goal` commands of Claude Code, Codex CLI, Kimi Code
 * and Hermes Agent, adapted to ACP: the agent cannot start a turn on its own,
 * so the loop runs inside the `session/prompt` call that set or resumed the
 * goal — or any later prompt while the goal is active.
 *
 * Pure functions only. The agent owns I/O, persistence, and the loop itself.
 */

export type GoalStatus = "active" | "paused" | "achieved" | "blocked";
export type GoalVerdict = "met" | "not_met" | "impossible";
/**
 * Model/tool turns one goal round may take before it stops on
 * `max_turn_requests`. A positive integer, or "unlimited" for no cap.
 */
export type GoalTurnLimit = number | "unlimited";
export type GoalPauseReason =
  /** `/goal pause`. */
  | "user"
  /** The client cancelled a goal turn. */
  | "cancelled"
  /** The round budget ran out. */
  | "budget"
  /** Several rounds in a row without a single tool call. */
  | "no_progress"
  /** A round used up its turn limit (`max_turn_requests`). */
  | "turn_limit"
  /** A turn ended on another stop reason. */
  | "error"
  /** The evaluator model could not be reached or did not answer in the verdict format. */
  | "judge_unavailable"
  /** The evaluator judged the condition impossible to satisfy. */
  | "blocked";

/** Token spend on a goal, main loop and evaluator alike. */
export interface GoalTokens {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

/**
 * One goal per session. Persisted with the session (schema v5) so a goal
 * survives `session/load`, `session/resume` and `session/fork`.
 */
export interface GoalState {
  condition: string;
  status: GoalStatus;
  /** Evaluator model id; absent means the session's own model. */
  judgeModel?: string;
  /** Model/tool turns each round may take; absent means the agent-wide goal default. */
  turnLimit?: GoalTurnLimit;
  /** Start of the current activation; reset by `/goal resume` and by a restore. */
  setAt: string;
  /** Continuation rounds consumed in the current activation. */
  rounds: number;
  maxRounds: number;
  /** Consecutive continuation rounds in which the model made no tool call. */
  noProgressRounds: number;
  lastVerdict?: GoalVerdict;
  lastReason?: string;
  pauseReason?: GoalPauseReason;
  /** Spend since the goal was set. */
  tokens?: GoalTokens;
}

export const GOAL_COMMAND_NAME = "goal";
/** Matches the Claude Code and Kimi Code limits. */
export const GOAL_CONDITION_CAP_CHARS = 4_000;
export const DEFAULT_GOAL_MAX_ROUNDS = 20;
/** Shown in status when nothing else is known; the agent passes its real default. */
export const DEFAULT_GOAL_TURN_LIMIT = 100;
/** Claude Code, Kimi Code and Qwen Code all stop after about three answer-only turns. */
export const GOAL_NO_PROGRESS_ROUNDS = 3;
/** Evaluator reasons become the next continuation's instructions; keep them short. */
export const GOAL_REASON_CAP_CHARS = 500;

export const GOAL_USAGE =
  "Usage: /goal <condition> [--judge <model>] [--turns <n|unlimited>]  |  /goal status  |  /goal pause  |  /goal resume  |  /goal clear";

export type GoalCommand =
  | { kind: "status" }
  | { kind: "pause" }
  | { kind: "resume" }
  | { kind: "clear" }
  | { kind: "set"; condition: string; judgeModel?: string; turnLimit?: GoalTurnLimit }
  | { kind: "invalid"; message: string };

const CONTROL_WORDS = new Map<string, "status" | "pause" | "resume" | "clear">([
  ["status", "status"],
  ["show", "status"],
  ["pause", "pause"],
  ["resume", "resume"],
  ["clear", "clear"],
  ["stop", "clear"],
  ["off", "clear"],
  ["reset", "clear"],
  ["cancel", "clear"],
  ["none", "clear"],
]);

/**
 * Extract the text after a leading `/goal` in a prompt block, or `undefined`
 * when the block is not a `/goal` invocation. Any whitespace separates the
 * name from its arguments, matching `parseSlashCommand`.
 */
export function parseGoalInvocation(text: string): string | undefined {
  const match = /^\/goal(?:\s+([\s\S]*))?$/.exec(text.trim());
  return match ? (match[1] ?? "") : undefined;
}

/**
 * Parse the text after `/goal`.
 *
 * A control word counts only when it is the entire argument — the DeepSeek
 * Harness grammar — so `/goal pause after verification` is a goal with that
 * text rather than a pause. `--judge <model>` picks the evaluator model and
 * `--turns <n|unlimited>` the per-round turn limit for this goal alone. Both
 * may come before or after the condition, in any order, so
 * `/goal fix the build --judge glm-5.3-flash` reads the way it is typed.
 * Every `--word` is read as an option — a mistyped one is rejected rather
 * than becoming goal text — and a trailing option's value runs to the end of
 * the input. A `--` (before or after leading options) turns option parsing
 * off for the remainder, for a condition that contains `--word` prose or
 * starts with a control word.
 */
export function parseGoalCommand(args: string): GoalCommand {
  const trimmed = args.trim();
  if (trimmed.length === 0) return { kind: "status" };
  const control = CONTROL_WORDS.get(trimmed.toLowerCase());
  if (control) return { kind: control };

  let rest = trimmed;
  let judgeModel: string | undefined;
  let turnLimit: GoalTurnLimit | undefined;
  /** Apply one `--name value` pair; returns the rejection text when it is not a known option. */
  const applyOption = (name: string, value: string | undefined): string | undefined => {
    if (name !== "judge" && name !== "turns") {
      return `Unknown /goal option "--${name}". Supported: --judge <model>, --turns <n|unlimited>, or -- before the condition to turn option parsing off.\n${GOAL_USAGE}`;
    }
    const trimmedValue = value?.trim();
    if (!trimmedValue) return `--${name} needs a value.\n${GOAL_USAGE}`;
    if (name === "judge") {
      judgeModel = trimmedValue;
      return undefined;
    }
    const parsed = parseTurnLimit(trimmedValue);
    if (parsed === undefined) {
      return `--turns needs a positive number or "unlimited", not "${trimmedValue}".\n${GOAL_USAGE}`;
    }
    turnLimit = parsed;
    return undefined;
  };

  // Every `--word` is an option, wherever it appears, so a mistyped one is
  // rejected rather than silently becoming goal text. A leading `--` turns
  // option parsing off for the whole remainder. Leading options take a
  // single-token value (the condition follows); a trailing option takes
  // everything after it, so `--judge Uncle Bob` is a rejected model name
  // rather than prose. The right-most option is peeled first, which lets
  // several trailing options chain.
  let escaped = false;
  if (rest === "--" || rest.startsWith("-- ")) {
    rest = rest.slice(2).trim();
    escaped = true;
  }
  while (!escaped && rest.startsWith("--")) {
    if (rest === "--" || rest.startsWith("-- ")) {
      rest = rest.slice(2).trim();
      escaped = true;
      break;
    }
    const match = /^--([A-Za-z][\w-]*)(?:[=\s]+(\S+))?(?:\s+([\s\S]*))?$/.exec(rest);
    const rejection = match ? applyOption(match[1]!, match[2]) : undefined;
    if (!match || rejection) {
      return { kind: "invalid", message: rejection ?? `Unknown /goal option in "${rest}".\n${GOAL_USAGE}` };
    }
    rest = (match[3] ?? "").trim();
  }
  while (!escaped) {
    const match = /^(.*)\s--([A-Za-z][\w-]*)(?:[=\s]+([\s\S]*))?$/.exec(rest);
    if (!match) break;
    const rejection = applyOption(match[2]!, match[3]);
    if (rejection) return { kind: "invalid", message: rejection };
    rest = match[1]!.trim();
  }

  const condition = rest.trim();
  if (condition.length === 0) {
    return { kind: "invalid", message: `A goal needs a condition.\n${GOAL_USAGE}` };
  }
  if (condition.length > GOAL_CONDITION_CAP_CHARS) {
    return {
      kind: "invalid",
      message: `Goal condition is ${condition.length} characters; the limit is ${GOAL_CONDITION_CAP_CHARS}.`,
    };
  }
  return {
    kind: "set",
    condition,
    ...(judgeModel ? { judgeModel } : {}),
    ...(turnLimit !== undefined ? { turnLimit } : {}),
  };
}

/**
 * Parse a per-round turn limit: a positive integer, or "unlimited" / "none"
 * for no cap. `undefined` for anything else.
 */
export function parseTurnLimit(raw: string): GoalTurnLimit | undefined {
  const lowered = raw.toLowerCase();
  if (lowered === "unlimited" || lowered === "none") return "unlimited";
  if (!/^\d+$/.test(raw)) return undefined;
  const parsed = Number(raw);
  return Number.isSafeInteger(parsed) && parsed >= 1 ? parsed : undefined;
}

/**
 * The bound the prompt loop iterates against. An unset goal limit falls back
 * to the agent-wide goal default; "unlimited" becomes Infinity, so the loop
 * only ends when the model stops, the client cancels, or a request fails.
 */
export function turnLimitForLoop(limit: GoalTurnLimit | undefined, fallback: GoalTurnLimit): number {
  const resolved = limit ?? fallback;
  return resolved === "unlimited" ? Number.POSITIVE_INFINITY : resolved;
}

/** `500` or `unlimited`, for notices and status lines. */
export function formatTurnLimit(limit: GoalTurnLimit): string {
  return limit === "unlimited" ? "unlimited" : String(limit);
}

export function createGoal(input: {
  condition: string;
  maxRounds: number;
  judgeModel?: string;
  turnLimit?: GoalTurnLimit;
  now: string;
}): GoalState {
  return {
    condition: input.condition,
    status: "active",
    ...(input.judgeModel ? { judgeModel: input.judgeModel } : {}),
    ...(input.turnLimit !== undefined ? { turnLimit: input.turnLimit } : {}),
    setAt: input.now,
    rounds: 0,
    maxRounds: input.maxRounds,
    noProgressRounds: 0,
  };
}

/**
 * Carry a persisted goal into a restored or forked session. The condition,
 * status and spend carry over; the round counter, no-progress streak and
 * timer restart, so a restored goal gets a fresh budget (Claude Code's
 * resume semantics). Terminal goals stay for `/goal` status.
 */
export function restoreGoal(goal: GoalState | undefined, now: string): GoalState | null {
  if (!goal) return null;
  return { ...goal, rounds: 0, noProgressRounds: 0, setAt: now };
}

/** Add a model call's usage to the goal's spend. */
export function addGoalTokens(
  goal: GoalState,
  usage: { inputTokens: number; outputTokens: number; totalTokens: number } | undefined
): void {
  if (!usage) return;
  const tokens = goal.tokens ?? { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
  tokens.inputTokens += usage.inputTokens;
  tokens.outputTokens += usage.outputTokens;
  tokens.totalTokens += usage.totalTokens;
  goal.tokens = tokens;
}

/** Type guard for a goal read back from disk. */
export function isGoalState(value: unknown): value is GoalState {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const goal = value as Record<string, unknown>;
  if (typeof goal.condition !== "string" || goal.condition.length === 0 || goal.condition.length > GOAL_CONDITION_CAP_CHARS) {
    return false;
  }
  if (typeof goal.status !== "string" || !VALID_STATUSES.has(goal.status)) return false;
  if (goal.judgeModel !== undefined && (typeof goal.judgeModel !== "string" || goal.judgeModel.length === 0)) return false;
  if (typeof goal.setAt !== "string" || Number.isNaN(Date.parse(goal.setAt))) return false;
  if (!isCount(goal.rounds) || !isCount(goal.maxRounds) || goal.maxRounds < 1 || !isCount(goal.noProgressRounds)) return false;
  if (goal.lastVerdict !== undefined && (typeof goal.lastVerdict !== "string" || !VALID_VERDICTS.has(goal.lastVerdict))) {
    return false;
  }
  if (goal.turnLimit !== undefined &&
      !(goal.turnLimit === "unlimited" || (isCount(goal.turnLimit) && goal.turnLimit >= 1))) {
    return false;
  }
  if (goal.lastReason !== undefined && typeof goal.lastReason !== "string") return false;
  if (goal.pauseReason !== undefined && (typeof goal.pauseReason !== "string" || !VALID_PAUSE_REASONS.has(goal.pauseReason))) {
    return false;
  }
  if (goal.tokens !== undefined) {
    if (typeof goal.tokens !== "object" || goal.tokens === null || Array.isArray(goal.tokens)) return false;
    const tokens = goal.tokens as Record<string, unknown>;
    for (const key of ["inputTokens", "outputTokens", "totalTokens"] as const) {
      if (!isCount(tokens[key])) return false;
    }
  }
  return true;
}

const VALID_STATUSES = new Set<string>(["active", "paused", "achieved", "blocked"]);
const VALID_VERDICTS = new Set<string>(["met", "not_met", "impossible"]);
const VALID_PAUSE_REASONS = new Set<string>([
  "user",
  "cancelled",
  "budget",
  "no_progress",
  "turn_limit",
  "error",
  "judge_unavailable",
  "blocked",
]);

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

// ---------------------------------------------------------------------------
// Model-facing text
// ---------------------------------------------------------------------------

/**
 * The user message that replaces `/goal <condition>` on the first turn.
 * Adapted from the directive Claude Code's Stop hook injects: acknowledge,
 * start immediately, never ask the user what to do next, and never point the
 * user at `/goal clear` — the goal clears itself when met.
 */
export function renderGoalDirective(condition: string): string {
  return [
    "<goal>",
    "A goal is now active for this session. Briefly acknowledge it, then immediately start working toward it:",
    "<condition>",
    neutralize(condition, "condition", "goal"),
    "</condition>",
    "Treat the condition itself as your directive and do not pause to ask the user what to do next. Before ending a turn, run the verification the condition names (tests, lint, a build, a file or command check) and show the output: an evaluator judges completion from this transcript, so surface concrete evidence rather than claims. The goal clears itself once the condition is met; do not tell the user to run /goal clear.",
    "</goal>",
  ].join("\n");
}

/** The user-role message that starts continuation round `round`. */
export function renderGoalContinuation(condition: string, round: number, maxRounds: number, reason: string): string {
  return [
    `<goal_continuation round="${round}/${maxRounds}">`,
    "The evaluator judged the goal not yet met. Outstanding work:",
    "<reason>",
    neutralize(cap(reason, GOAL_REASON_CAP_CHARS), "reason", "goal_continuation"),
    "</reason>",
    "Continue working toward the condition and address the outstanding work above. Do not ask the user what to do next, and show verification evidence before stopping.",
    "<condition>",
    neutralize(condition, "condition", "goal_continuation"),
    "</condition>",
    "</goal_continuation>",
  ].join("\n");
}

/** The user-role message that restarts a paused goal after `/goal resume`. */
export function renderGoalResume(condition: string, round: number, maxRounds: number): string {
  return [
    `<goal_resumed round="${round}/${maxRounds}">`,
    "The user resumed this goal. Continue working toward the condition from where the previous round left off. Do not ask the user what to do next, and show verification evidence before stopping.",
    "<condition>",
    neutralize(condition, "condition", "goal_resumed"),
    "</condition>",
    "</goal_resumed>",
  ].join("\n");
}

/** What the client transcript shows for an injected continuation; replayed on `session/load`. */
export function goalContinuationDisplayText(round: number, maxRounds: number, reason: string): string {
  return `[goal ${round}/${maxRounds}] Continuing — not met yet: ${cap(reason, GOAL_REASON_CAP_CHARS)}`;
}

// ---------------------------------------------------------------------------
// Client-facing text
// ---------------------------------------------------------------------------

/**
 * Reply to `/goal` (status). `defaultTurnLimit` is the agent-wide goal turn
 * limit, shown when the goal did not set its own.
 */
export function formatGoalStatus(
  goal: GoalState | null | undefined,
  sessionModel: string,
  now: number = Date.now(),
  defaultTurnLimit: GoalTurnLimit = DEFAULT_GOAL_TURN_LIMIT
): string {
  if (!goal) return `No goal set.\n\n${GOAL_USAGE}`;
  const pause = goal.status === "paused" && goal.pauseReason ? ` (${goal.pauseReason.replaceAll("_", " ")})` : "";
  const lines = [`**Goal: ${goal.status}${pause}**`, "", goal.condition, "", `- Rounds: ${goal.rounds}/${goal.maxRounds}`];
  if (goal.status === "active") lines.push(`- Running: ${formatDuration(Math.max(0, now - Date.parse(goal.setAt)))}`);
  lines.push(`- Turn limit: ${formatTurnLimit(goal.turnLimit ?? defaultTurnLimit)} per round`);
  lines.push(`- Evaluator: ${goal.judgeModel ?? `${sessionModel} (session model)`}`);
  if (goal.tokens) lines.push(`- Tokens: ${formatTokens(goal.tokens.totalTokens)} (agent + evaluator)`);
  if (goal.lastVerdict) {
    lines.push(`- Last check: ${formatVerdict(goal.lastVerdict)}${goal.lastReason ? ` — ${goal.lastReason}` : ""}`);
  }
  lines.push("");
  switch (goal.status) {
    case "active":
      lines.push("Send /goal pause to stop auto-continuation, or /goal clear to drop the goal.");
      break;
    case "paused":
      lines.push("Send /goal resume to continue, or /goal clear to drop the goal.");
      break;
    case "blocked":
      lines.push("Send /goal <condition> to re-scope, or /goal resume to override the evaluator.");
      break;
    case "achieved":
      lines.push("Send /goal <condition> to set a new goal.");
      break;
  }
  return lines.join("\n");
}

export function goalPauseNotice(goal: GoalState): string {
  return `Goal paused: ${goal.condition}\nSend /goal resume to continue, or /goal clear to drop the goal.`;
}

export function goalClearedNotice(condition: string): string {
  return `Goal cleared: ${condition}`;
}

/** Terminal and pause notices emitted while the loop runs. */
export function goalAchievedNotice(goal: GoalState, reason: string): string {
  return `Goal achieved after ${goal.rounds} continuation round${goal.rounds === 1 ? "" : "s"}: ${reason}`;
}

export function goalBlockedNotice(reason: string): string {
  return `Goal judged unachievable — paused: ${reason}\nSend /goal <condition> to re-scope, or /goal resume to override.`;
}

export function goalPausedNotice(goal: GoalState, detail?: string): string {
  const cause = goal.pauseReason ?? "user";
  let line: string;
  switch (cause) {
    case "budget":
      line = `round budget reached (${goal.rounds}/${goal.maxRounds})`;
      break;
    case "no_progress":
      line = `${goal.noProgressRounds} rounds in a row without tool calls`;
      break;
    case "cancelled":
      line = "turn cancelled";
      break;
    case "turn_limit":
      line = `the round hit its turn limit (${detail ?? "default"})`;
      break;
    case "error":
      line = `turn ended with ${detail ?? "an error"}`;
      break;
    case "judge_unavailable":
      line = `evaluator unavailable${detail ? `: ${detail}` : ""}`;
      break;
    default:
      line = cause;
  }
  const hint = cause === "turn_limit"
    ? "Send /goal resume to continue, /goal --turns <n|unlimited> to set a new goal with a higher limit, or /goal clear to drop the goal."
    : "Send /goal resume to continue, or /goal clear to drop the goal.";
  return `Goal paused — ${line}.\n${hint}`;
}

export function formatVerdict(verdict: GoalVerdict): string {
  switch (verdict) {
    case "not_met":
      return "not met";
    case "impossible":
      return "impossible";
    default:
      return verdict;
  }
}

export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1_000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
}

export function formatTokens(total: number): string {
  if (total < 1_000) return String(total);
  if (total < 1_000_000) return `${(total / 1_000).toFixed(1)}k`;
  return `${(total / 1_000_000).toFixed(2)}M`;
}

function cap(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/**
 * Stop supplied text from closing one of the XML-style wrappers it sits in.
 * Mirrors `renderProjectContext`: a zero-width space after the `<` keeps the
 * text readable while making the closing tag unmatchable.
 */
function neutralize(text: string, ...tags: string[]): string {
  let safe = text;
  for (const tag of tags) safe = safe.replaceAll(`</${tag}>`, `<\u200b/${tag}>`);
  return safe;
}
