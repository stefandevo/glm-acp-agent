import OpenAI from "openai";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "openai/resources/index.js";
import type { ModelInfo, Usage } from "@agentclientprotocol/sdk";
import { TOOL_DEFINITIONS } from "../tools/definitions.js";
import { resolveApiKey } from "./credentials.js";
import { debug, error } from "./logger.js";

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
}

/** Default base URL for the Z.AI / Zhipu OpenAI-compatible API (Coding endpoint). */
const DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4";

/** Default GLM model when neither client nor user has chosen one. */
export const DEFAULT_MODEL = "glm-5.1";

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
    modelId: "glm-5.1",
    name: "GLM-5.1",
    description: "Latest GLM reasoning model with thinking mode",
  },
  {
    modelId: "glm-5-turbo",
    name: "GLM-5 Turbo",
    description: "Faster Coding Plan reasoning model",
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
    const thinkingEnabled = parseThinking(model);
    const tools: ChatCompletionTool[] = TOOL_DEFINITIONS.map((t) => ({
      type: "function" as const,
      function: {
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters as Record<string, unknown>,
      },
    }));

    debug(
      `streamChat: model=${model} baseURL=${this.client.baseURL} messages=${messages.length} tools=${tools.length} thinking=${thinkingEnabled}`
    );

    // The OpenAI SDK forwards unknown extra body fields verbatim, so we use
    // that to pass GLM-specific fields like `thinking`.
    const extraBody: Record<string, unknown> = {};
    if (thinkingEnabled) {
      extraBody["thinking"] = { type: "enabled" };
    }

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
 * Decide whether to enable GLM "thinking" mode for a given model name.
 *
 * The user can force on/off via `ACP_GLM_THINKING=true|false`. Otherwise,
 * we enable thinking for any model whose name suggests it supports it
 * (the GLM-4.5 / GLM-4.6 / GLM-4.7 / GLM-5.x families).
 */
function parseThinking(model: string): boolean {
  const override = process.env["ACP_GLM_THINKING"];
  if (override !== undefined) {
    return override.toLowerCase() === "true" || override === "1";
  }
  return /^glm-(?:4\.[567]|5)/i.test(model);
}
