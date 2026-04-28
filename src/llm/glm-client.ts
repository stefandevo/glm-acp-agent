import OpenAI from "openai";
import type { ChatCompletionMessageParam, ChatCompletionTool } from "openai/resources/index.js";
import { TOOL_DEFINITIONS } from "../tools/definitions.js";

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
  usage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
  /** Set when the stream is done */
  done?: boolean;
  /** Stop reason when done */
  stopReason?: string;
}

/**
 * Wrapper around the OpenAI-compatible Zhipu AI (Z.AI) API.
 *
 * Uses the standard `openai` npm package pointed at `https://api.z.ai/v1`
 * so no Zhipu-specific SDK is required.
 */
export class GlmClient {
  private client: OpenAI;
  private model: string;

  constructor() {
    const apiKey = process.env["Z_AI_API_KEY"];
    if (!apiKey) {
      throw new Error(
        "Z_AI_API_KEY environment variable is required but not set."
      );
    }

    this.model = process.env["ACP_GLM_MODEL"] ?? "glm-5-1";

    this.client = new OpenAI({
      apiKey,
      baseURL: "https://api.z.ai/v1",
    });
  }

  /**
   * Stream a chat completion from the GLM model, yielding chunks as they arrive.
   *
   * Reasoning/thinking tokens (from GLM-5.1's "Rethink" mode) are mapped to
   * `thinking` chunks so the ACP agent can forward them as `agent_thought_chunk`
   * blocks.
   */
  async *streamChat(messages: GlmMessage[], signal?: AbortSignal): AsyncGenerator<GlmStreamChunk> {
    const tools: ChatCompletionTool[] = TOOL_DEFINITIONS.map((t) => ({
      type: "function" as const,
      function: {
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters as Record<string, unknown>,
      },
    }));

    // Extra params for GLM-5.1 thinking mode. The openai SDK passes through
    // unknown extra body fields so this works transparently.
    const extraBody: Record<string, unknown> = {};
    if (this.model.startsWith("glm-5")) {
      extraBody["thinking"] = { type: "enabled" };
    }

    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages,
      tools,
      tool_choice: "auto",
      stream: true,
      max_tokens: 8192,
      ...extraBody,
    }, { signal });

    // Accumulate tool call deltas keyed by index
    const pendingToolCalls: Map<
      number,
      { id: string; name: string; arguments: string }
    > = new Map();

    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      if (!choice) continue;

      const delta = choice.delta as Record<string, unknown>;

      // Thinking / reasoning tokens (GLM-5.1 "Rethink" mode)
      // The API returns these in delta.reasoning_content
      if (typeof delta["reasoning_content"] === "string" && delta["reasoning_content"].length > 0) {
        yield { thinking: delta["reasoning_content"] as string };
      }

      // Regular assistant text
      if (typeof delta["content"] === "string" && delta["content"].length > 0) {
        yield { text: delta["content"] as string };
      }

      // Tool call deltas
      const toolCallDeltas = delta["tool_calls"] as
        | Array<{
            index: number;
            id?: string;
            function?: { name?: string; arguments?: string };
          }>
        | undefined;

      if (toolCallDeltas) {
        for (const tc of toolCallDeltas) {
          if (!pendingToolCalls.has(tc.index)) {
            pendingToolCalls.set(tc.index, { id: "", name: "", arguments: "" });
          }
          const pending = pendingToolCalls.get(tc.index)!;
          if (tc.id) pending.id = tc.id;
          if (tc.function?.name) pending.name = tc.function.name;
          if (tc.function?.arguments) pending.arguments += tc.function.arguments;
        }
      }

      // When the finish reason is set, emit any completed tool calls and usage
      if (choice.finish_reason) {
        for (const [, tc] of pendingToolCalls) {
          yield { toolCall: tc };
        }
        pendingToolCalls.clear();

        const rawUsage = chunk.usage as
          | { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }
          | undefined;
        const usage = rawUsage
          ? {
              inputTokens: rawUsage.prompt_tokens ?? 0,
              outputTokens: rawUsage.completion_tokens ?? 0,
              totalTokens: rawUsage.total_tokens ?? 0,
            }
          : undefined;

        yield { done: true, stopReason: choice.finish_reason, usage };
      }
    }
  }
}
