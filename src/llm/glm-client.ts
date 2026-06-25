import OpenAI from "openai";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "openai/resources/index.js";
import type { ModelInfo, Usage } from "@agentclientprotocol/sdk";
import { TOOL_DEFINITIONS, type ToolDefinition } from "../tools/definitions.js";
import { resolveApiKey } from "./credentials.js";
import { debug, error } from "./logger.js";

/**
 * Reasoning effort levels exposed to ACP clients via the thought_level
 * SessionConfigOption. These map onto Z.AI's thinking / reasoning_effort
 * parameters (see buildThinkingParams).
 *
 * GLM-5.2 supports three levels: none, high, max.
 * Other thinking-capable models (GLM-5.1, 5-turbo, 4.7, etc.) only
 * distinguish between thinking on and off, so they use "none" and "on".
 */
export type ThoughtLevel = "none" | "on" | "high" | "max";

/** Levels shown when the selected model is GLM-5.2. */
const LEVELS_52: ThoughtLevel[] = ["none", "high", "max"];

/** Levels shown for every other thinking-capable model. */
const LEVELS_DEFAULT: ThoughtLevel[] = ["none", "on"];

/**
 * Resolve which thought-level options a model supports.
 *
 * reasoning_effort is a GLM-5.2 exclusive per the Z.AI docs — other models
 * accept the field but it has no effect, so we only expose high/max for 5.2.
 */
export function getThoughtLevels(model: string): ThoughtLevel[] {
  return model.toLowerCase().startsWith("glm-5.2") ? LEVELS_52 : LEVELS_DEFAULT;
}

/**
 * Resolve a stored ThoughtLevel to one that's valid for the given model.
 * Used when switching models or restoring a persisted session: if the old
 * level isn't in the new model's option list, fall back to the model's
 * default (max for 5.2, on for everything else).
 */
export function resolveThoughtLevel(model: string, level: ThoughtLevel): ThoughtLevel {
  const valid = getThoughtLevels(model);
  return valid.includes(level) ? level : valid[valid.length - 1];
}

/**
 * A single message in the GLM conversation history.
 */
export type GlmMessage = ChatCompletionMessageParam;

/**
 * A streamed chunk from the GLM API.
 */
export interface GlmStreamChunk {
  /** Incremental assistant text */
  text?: string;
  /** Incremental reasoning/thinking text */
  thinking?: string;
  /** A complete tool call (assembled from streaming deltas) */
  toolCall?: {
    id: string;
    name: string;
    arguments: string;
  };
  /** Token usage reported when the stream finishes */
  usage?: Usage;
  /** Set when the stream is done */
  done?: boolean;
  /** Stop reason when done */
  stopReason?: string;
}

/** Options applied to a single `streamChat` call. */
export interface StreamChatOptions {
  /** GLM model identifier to use for this call. */
  model: string;
  /** Tool schemas available in this specific session. */
  tools?: ToolDefinition[];
  /** Reasoning effort for this call, or undefined to use defaults. */
  reasoningEffort?: ThoughtLevel;
}

/** Default base URL for the Z.AI / Zhipu OpenAI-compatible API (Coding endpoint). */
const DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4";

/** Default GLM model when neither client nor user has chosen one. */
export const DEFAULT_MODEL = "glm-5.2";

/**
 * Curated list of GLM models the agent advertises to ACP clients via
 * `SessionModelState.availableModels`. Users can override this list via the
 * `ACP_GLM_AVAILABLE_MODELS` environment variable (comma-separated model ids).
 *
 * The descriptions are intentionally short — clients render them in compact
 * model pickers.
 */
const BUILTIN_AVAILABLE_MODELS: ModelInfo[] = [
  {
    modelId: "glm-5.2",
    name: "GLM-5.2",
    description: "1M-context coding model with thinking mode",
  },
  {
    modelId: "glm-5.1",
    name: "GLM-5.1",
    description: "Long-horizon coding model with thinking mode",
  },
  {
    modelId: "glm-5-turbo",
    name: "GLM-5 Turbo",
    description: "Faster Coding Plan reasoning model",
  },
  {
    modelId: "glm-5v-turbo",
    name: "GLM-5V Turbo",
    description: "Multimodal Coding Plan model with native vision",
  },
  {
    modelId: "glm-4.7",
    name: "GLM-4.7",
    description: "200K-context reasoning model",
  },
  {
    modelId: "glm-4.5-air",
    name: "GLM-4.5 Air",
    description: "Lightweight, lower-latency model",
  },
];

/**
 * Z.AI / Zhipu AI error code returned when the total prompt length (messages +
 * tools) exceeds the model's context window.
 */
export const ERR_CONTEXT_OVERFLOW = 1261;

/**
 * Context window sizes (in tokens) for GLM series models.
 */
const MODEL_METADATA: Record<string, { contextWindow: number }> = {
  "glm-5.2": { contextWindow: 1_000_000 },
  "glm-5.1": { contextWindow: 128_000 },
  "glm-5-turbo": { contextWindow: 128_000 },
  "glm-5v-turbo": { contextWindow: 200_000 },
  "glm-4.7": { contextWindow: 200_000 },
  "glm-4.5-air": { contextWindow: 128_000 },
};

/** Models that accept image content parts directly through chat completions. */
const VISION_NATIVE_MODELS = new Set(["glm-5v-turbo"]);

export function isVisionNativeModel(modelId: string): boolean {
  return VISION_NATIVE_MODELS.has(modelId.toLowerCase());
}

/**
 * Resolve the context window size for a given model ID. Falls back to a safe
 * default (128K) for uncatalogued models.
 */
export function getContextWindow(modelId: string): number {
  return MODEL_METADATA[modelId]?.contextWindow ?? 128_000;
}

/**
 * Resolve the list of advertised models, allowing the user to override the
 * built-in list via `ACP_GLM_AVAILABLE_MODELS`.
 */
export function getAvailableModels(): ModelInfo[] {
  const override = process.env["ACP_GLM_AVAILABLE_MODELS"];
  if (!override) return BUILTIN_AVAILABLE_MODELS;
  const ids = override
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  if (ids.length === 0) return BUILTIN_AVAILABLE_MODELS;
  return ids.map((id) => {
    const builtin = BUILTIN_AVAILABLE_MODELS.find((m) => m.modelId === id);
    return builtin ?? { modelId: id, name: id };
  });
}

/** Default model for new sessions: env override → built-in default. */
export function getDefaultModel(): string {
  return process.env["ACP_GLM_MODEL"] ?? DEFAULT_MODEL;
}

/**
 * Wrapper around the OpenAI-compatible Zhipu AI (Z.AI) API.
 *
 * Uses the standard `openai` npm package pointed at `https://api.z.ai/api/paas/v4`
 * so no Zhipu-specific SDK is required. The Z.AI service speaks the OpenAI
 * Chat Completions wire format, plus a few GLM-specific extras (like the
 * `thinking` field and `delta.reasoning_content` for reasoning tokens).
 */
export class GlmClient {
  private client: OpenAI;
  private maxTokens: number;

  constructor() {
    const apiKey = resolveApiKey();
    if (!apiKey) {
      throw new Error(
        "No API key found. Set the Z_AI_API_KEY environment variable, or run `glm-acp-agent --setup` to store one."
      );
    }

    const baseURL = process.env["ACP_GLM_BASE_URL"] ?? DEFAULT_BASE_URL;
    this.maxTokens = parseIntEnv("ACP_GLM_MAX_TOKENS", 8192);

    this.client = new OpenAI({ apiKey, baseURL });
  }

  /**
   * Stream a chat completion from the GLM model, yielding chunks as they arrive.
   *
   * Reasoning/thinking tokens (from GLM "thinking" mode) are mapped to
   * `thinking` chunks so the ACP agent can forward them as `agent_thought_chunk`
   * blocks.
   */
  async *streamChat(
    messages: GlmMessage[],
    signal?: AbortSignal,
    options?: StreamChatOptions
  ): AsyncGenerator<GlmStreamChunk> {
    const model = options?.model ?? getDefaultModel();
    const thinkingParams = buildThinkingParams(model, options?.reasoningEffort);
    const tools: ChatCompletionTool[] = (options?.tools ?? TOOL_DEFINITIONS).map((t) => ({
      type: "function" as const,
      function: {
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters as Record<string, unknown>,
      },
    }));

    debug(
      `streamChat: model=${model} baseURL=${this.client.baseURL} messages=${messages.length} tools=${tools.length} thinking=${JSON.stringify(thinkingParams)}`
    );

    // The OpenAI SDK forwards unknown extra body fields verbatim, so we use
    // that to pass GLM-specific fields like `thinking` and `reasoning_effort`.
    const extraBody: Record<string, unknown> = { ...thinkingParams };

    let stream: Awaited<ReturnType<typeof this.client.chat.completions.create>>;
    try {
      stream = await this.client.chat.completions.create(
        {
          model,
          messages,
          tools,
          tool_choice: "auto",
          stream: true,
          // Always ask the API to include final usage in the streaming response.
          stream_options: { include_usage: true },
          max_tokens: this.maxTokens,
          ...extraBody,
        },
        { signal }
      );
    } catch (err) {
      const status = (err as { status?: number })?.status;
      const body = (err as { error?: unknown })?.error;
      error(`streamChat request failed: status=${status}`, JSON.stringify(body) ?? String(err));
      throw err;
    }

    // Tool call deltas arrive interleaved across chunks; assemble by index.
    const pendingToolCalls: Map<
      number,
      { id: string; name: string; arguments: string }
    > = new Map();

    let lastFinishReason: string | undefined;

    for await (const chunk of stream) {
      const choice = chunk.choices[0];

      if (choice) {
        const delta = choice.delta as Record<string, unknown>;

        // Reasoning / thinking tokens (GLM "thinking" mode).
        const reasoning = delta["reasoning_content"];
        if (typeof reasoning === "string" && reasoning.length > 0) {
          yield { thinking: reasoning };
        }

        // Regular assistant text.
        const content = delta["content"];
        if (typeof content === "string" && content.length > 0) {
          yield { text: content };
        }

        // Tool call deltas.
        const toolCallDeltas = delta["tool_calls"] as
          | Array<{
              index: number;
              id?: string;
              function?: { name?: string; arguments?: string };
            }>
          | undefined;

        if (Array.isArray(toolCallDeltas)) {
          for (const tc of toolCallDeltas) {
            let pending = pendingToolCalls.get(tc.index);
            if (!pending) {
              pending = { id: "", name: "", arguments: "" };
              pendingToolCalls.set(tc.index, pending);
            }
            if (tc.id) pending.id = tc.id;
            if (tc.function?.name) pending.name = tc.function.name;
            if (tc.function?.arguments) pending.arguments += tc.function.arguments;
          }
        }

        if (choice.finish_reason) {
          lastFinishReason = choice.finish_reason;
        }
      }

      // The final chunk on most providers (GLM included when
      // include_usage: true) ships only a `usage` object with no choices.
      const rawUsage = (chunk as { usage?: unknown }).usage as
        | {
            prompt_tokens?: number;
            completion_tokens?: number;
            total_tokens?: number;
            prompt_tokens_details?: { cached_tokens?: number };
            completion_tokens_details?: { reasoning_tokens?: number };
          }
        | null
        | undefined;

      if (rawUsage) {
        const usage: Usage = {
          inputTokens: rawUsage.prompt_tokens ?? 0,
          outputTokens: rawUsage.completion_tokens ?? 0,
          totalTokens: rawUsage.total_tokens ?? 0,
        };
        if (typeof rawUsage.prompt_tokens_details?.cached_tokens === "number") {
          usage.cachedReadTokens = rawUsage.prompt_tokens_details.cached_tokens;
        }
        if (
          typeof rawUsage.completion_tokens_details?.reasoning_tokens === "number"
        ) {
          usage.thoughtTokens = rawUsage.completion_tokens_details.reasoning_tokens;
        }
        debug(`streamChat usage: input=${usage.inputTokens} output=${usage.outputTokens} total=${usage.totalTokens}`);
        yield { usage };
      }
    }

    // Flush any assembled tool calls and emit a final done chunk. Only emit
    // calls that have both an id and a name – partial entries can be left
    // behind by upstream errors and would just confuse the agent loop.
    for (const [, tc] of pendingToolCalls) {
      if (tc.id && tc.name) yield { toolCall: tc };
    }
    pendingToolCalls.clear();

    yield { done: true, stopReason: lastFinishReason };
  }
}

/** Parse an integer environment variable, falling back to a default. */
function parseIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/**
 * Decide whether a model supports GLM thinking mode based on its name.
 *
 * The user can force on/off via `ACP_GLM_THINKING=true|false`. Otherwise,
 * we enable thinking for any model whose name suggests it supports it
 * (the GLM-4.5 / GLM-4.6 / GLM-4.7 / GLM-5.x families).
 */
function supportsThinking(model: string): boolean {
  const override = process.env["ACP_GLM_THINKING"];
  if (override !== undefined) {
    return override.toLowerCase() === "true" || override === "1";
  }
  return /^glm-(?:4\.[567]|5)/i.test(model);
}

/**
 * Build the Z.AI thinking / reasoning_effort extra-body params for a call.
 *
 * Z.AI has two parameters:
 * - `thinking` — on/off gate: `{"type":"enabled"}` or `{"type":"disabled"}`
 * - `reasoning_effort` — GLM-5.2 only; controls thinking depth.
 *
 * ThoughtLevel values map as follows:
 * - "none"  → thinking disabled
 * - "on"    → thinking enabled (no reasoning_effort)
 * - "high"  → thinking enabled + reasoning_effort=high (5.2 only)
 * - "max"   → thinking enabled + reasoning_effort=max (5.2 only)
 *
 * When `effort` is unset, the model defaults are used.
 */
export function buildThinkingParams(
  model: string,
  effort?: ThoughtLevel
): Record<string, unknown> {
  const params: Record<string, unknown> = {};

  const canThink = supportsThinking(model);

  if (effort === "none") {
    if (canThink) params["thinking"] = { type: "disabled" };
    return params;
  }

  if (!canThink) return params;

  params["thinking"] = { type: "enabled" };

  // reasoning_effort is only meaningful for GLM-5.2 per the Z.AI docs.
  if (effort && effort !== "on" && model.toLowerCase().startsWith("glm-5.2")) {
    params["reasoning_effort"] = effort;
  }

  return params;
}
