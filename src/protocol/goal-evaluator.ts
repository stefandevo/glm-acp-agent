/**
 * Goal evaluator: a separate, tool-less model call that judges whether the
 * session goal is met, given the condition and a bounded window of the recent
 * transcript.
 *
 * Adapted from the judges behind Claude Code's, Codex's, Kimi's and Hermes'
 * `/goal` commands: strict, evidence-based, and conservative — a missing proof
 * is "not met", never "met". The evaluator never runs tools, so it can only
 * judge what the agent has surfaced in the conversation; the directive in
 * `goal.ts` tells the agent to show verification output for that reason.
 */
import type { Usage } from "@agentclientprotocol/sdk";
import {
  resolveThoughtLevel,
  type GlmMessage,
  type GlmStreamChunk,
  type StreamChatOptions,
  type ThoughtLevel,
} from "../llm/glm-client.js";
import { GOAL_REASON_CAP_CHARS, type GoalVerdict } from "./goal.js";

/** Anything with the GLM client's `streamChat` — the real client or a test fake. */
export interface GoalJudgeClient {
  streamChat(
    messages: GlmMessage[],
    signal?: AbortSignal,
    options?: StreamChatOptions
  ): AsyncIterable<GlmStreamChunk>;
}

export interface GoalVerdictResult {
  verdict: GoalVerdict;
  reason: string;
  usage?: Usage;
}

/** Raised when no attempt produced a parseable verdict. The agent pauses the goal. */
export class GoalJudgeUnavailableError extends Error {}

/** How many messages of transcript the evaluator sees, oldest dropped first. */
const WINDOW_MESSAGES = 8;
/** Per-item caps keep one huge tool result from crowding out the rest of the window. */
const TOOL_RESULT_CAP_CHARS = 2_000;
const TOOL_CALL_CAP_CHARS = 300;
const WINDOW_CAP_CHARS = 16_000;
/** One retry covers a dropped stream or a reply that is not valid JSON. */
export const GOAL_JUDGE_ATTEMPTS = 2;

/**
 * The lowest reasoning effort the model supports: verdicts are short
 * classifications, and the evaluator runs once per round on top of the main
 * loop's spend.
 */
export function judgeReasoningEffort(model: string): ThoughtLevel {
  return resolveThoughtLevel(model, "minimal");
}

const JUDGE_SYSTEM_PROMPT = `You are a strict goal evaluator for an autonomous coding agent. You receive a goal condition and the recent transcript of the agent's work, and decide whether the condition has been met, based only on evidence in the transcript. You cannot run commands or read files; judge what the agent has shown.

Verdicts:
- "met" only when the transcript shows concrete evidence that the whole condition holds: command output (tests, lint, a build), file contents, a diff, or a produced artifact. A claim without evidence is not met. Partial completion is not met.
- "impossible" only when the condition cannot be satisfied as stated: it is contradictory, outside what this session can do, or needs something the agent cannot obtain (credentials, a decision only the user can make, a missing system).
- "not_met" otherwise.

"reason" is one or two sentences. For "not_met", name the outstanding work concretely — it is handed to the agent as its next instruction. For "met" or "impossible", state the evidence or the blocker that decided it.

Reply with only a JSON object of this shape: {"verdict":"met"|"not_met"|"impossible","reason":"..."}`;

/** The messages sent to the evaluator: instructions plus a fenced transcript window. */
export function buildJudgeMessages(input: {
  condition: string;
  rounds: number;
  maxRounds: number;
  messages: ReadonlyArray<GlmMessage>;
}): GlmMessage[] {
  const transcript = renderTranscriptWindow(input.messages);
  const user = [
    "<goal_evaluation>",
    "<rounds>",
    `${input.rounds}/${input.maxRounds}`,
    "</rounds>",
    "<condition>",
    neutralize(input.condition, "condition", "goal_evaluation"),
    "</condition>",
    "<transcript>",
    "```",
    neutralize(transcript, "transcript", "goal_evaluation").replaceAll("```", "`\u200b``"),
    "```",
    "</transcript>",
    "</goal_evaluation>",
  ].join("\n");
  return [
    { role: "system", content: JUDGE_SYSTEM_PROMPT },
    { role: "user", content: user },
  ];
}

/**
 * Parse the evaluator's reply. Tolerates a Markdown fence or prose around
 * the object, and the common synonyms agents use for the three verdicts;
 * anything else is unparseable so the caller can retry or pause.
 */
export function parseGoalVerdict(text: string): GoalVerdictResult | undefined {
  const stripped = text.trim().replace(/^```[a-z]*\s*/i, "").replace(/```\s*$/, "");
  const start = stripped.indexOf("{");
  const end = stripped.lastIndexOf("}");
  if (start < 0 || end <= start) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(stripped.slice(start, end + 1));
  } catch {
    return undefined;
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return undefined;
  const record = parsed as Record<string, unknown>;
  const verdict = normalizeVerdict(record.verdict ?? record.status ?? record.decision ?? record.result);
  if (!verdict) return undefined;
  const reason = typeof record.reason === "string" ? record.reason.trim() : "";
  return { verdict, reason: reason.length > GOAL_REASON_CAP_CHARS ? `${reason.slice(0, GOAL_REASON_CAP_CHARS - 1)}…` : reason };
}

/**
 * Ask the evaluator model for a verdict. Retries once on a failed stream or an
 * unparseable reply, then raises {@link GoalJudgeUnavailableError}. An abort
 * signal propagates as the abort error so the caller can tell a cancel from an
 * outage.
 */
export async function evaluateGoal(
  client: GoalJudgeClient,
  input: {
    condition: string;
    rounds: number;
    maxRounds: number;
    messages: ReadonlyArray<GlmMessage>;
    model: string;
    signal?: AbortSignal;
  }
): Promise<GoalVerdictResult> {
  const judgeMessages = buildJudgeMessages(input);
  const options: StreamChatOptions = {
    model: input.model,
    tools: null,
    reasoningEffort: judgeReasoningEffort(input.model),
  };
  let lastFailure: unknown;
  for (let attempt = 0; attempt < GOAL_JUDGE_ATTEMPTS; attempt++) {
    input.signal?.throwIfAborted();
    let text = "";
    let usage: Usage | undefined;
    try {
      for await (const chunk of client.streamChat(judgeMessages, input.signal, options)) {
        if (chunk.text) text += chunk.text;
        if (chunk.usage) usage = chunk.usage;
      }
    } catch (err) {
      input.signal?.throwIfAborted();
      lastFailure = err;
      continue;
    }
    const verdict = parseGoalVerdict(text);
    if (verdict) return { ...verdict, ...(usage ? { usage } : {}) };
    lastFailure = new Error(`reply was not a verdict: ${JSON.stringify(text.slice(0, 200))}`);
  }
  throw new GoalJudgeUnavailableError(
    lastFailure instanceof Error ? lastFailure.message : String(lastFailure)
  );
}

function normalizeVerdict(value: unknown): GoalVerdict | undefined {
  if (typeof value !== "string") return undefined;
  switch (value.toLowerCase().replace(/[^a-z]/g, "")) {
    case "met":
    case "done":
    case "complete":
    case "completed":
    case "achieved":
      return "met";
    case "notmet":
    case "notyetmet":
    case "continue":
    case "incomplete":
    case "pending":
      return "not_met";
    case "impossible":
    case "unachievable":
    case "blocked":
      return "impossible";
    default:
      return undefined;
  }
}

/**
 * Render the last {@link WINDOW_MESSAGES} non-system messages, oldest first,
 * as plain text. Tool calls and results are summarised and capped: the
 * evaluator needs to see that a check ran and what it printed, not the whole
 * payload.
 */
function renderTranscriptWindow(messages: ReadonlyArray<GlmMessage>): string {
  const blocks: string[] = [];
  let total = 0;
  for (let index = messages.length - 1; index >= 0 && blocks.length < WINDOW_MESSAGES; index--) {
    const block = renderMessage(messages[index]!);
    if (block === undefined) continue;
    if (blocks.length > 0 && total + block.length > WINDOW_CAP_CHARS) break;
    blocks.push(block);
    total += block.length;
  }
  return blocks.reverse().join("\n\n");
}

function renderMessage(message: GlmMessage): string | undefined {
  switch (message.role) {
    case "user":
      return `[user]\n${cap(renderContent(message.content), TOOL_RESULT_CAP_CHARS)}`;
    case "assistant": {
      const parts: string[] = [];
      const text = renderContent(message.content);
      if (text) parts.push(cap(text, TOOL_RESULT_CAP_CHARS));
      for (const call of message.tool_calls ?? []) {
        if (call.type !== "function") continue;
        parts.push(`[tool call] ${call.function.name} ${cap(call.function.arguments, TOOL_CALL_CAP_CHARS)}`);
      }
      return parts.length > 0 ? `[assistant]\n${parts.join("\n")}` : undefined;
    }
    case "tool":
      return `[tool result]\n${cap(renderContent(message.content), TOOL_RESULT_CAP_CHARS)}`;
    default:
      return undefined;
  }
}

function renderContent(content: GlmMessage["content"]): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if ("text" in part && typeof part.text === "string") return part.text;
        return `[${part.type}]`;
      })
      .join("\n");
  }
  return "";
}

function cap(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function neutralize(text: string, ...tags: string[]): string {
  let safe = text;
  for (const tag of tags) safe = safe.replaceAll(`</${tag}>`, `<\u200b/${tag}>`);
  return safe;
}
