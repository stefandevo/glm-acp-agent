import { randomUUID } from "node:crypto";
import type {
  Agent,
  AgentSideConnection,
  InitializeRequest,
  InitializeResponse,
  NewSessionRequest,
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  CancelNotification,
  AuthenticateRequest,
  AuthenticateResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  CloseSessionRequest,
  ListSessionsRequest,
  ListSessionsResponse,
} from "@agentclientprotocol/sdk";
import { PROTOCOL_VERSION as VERSION } from "@agentclientprotocol/sdk";
import { GlmClient, type GlmMessage } from "../llm/glm-client.js";
import { ToolExecutor } from "../tools/executor.js";

/** Per-session state */
interface SessionState {
  cwd: string;
  messages: GlmMessage[];
  abortController: AbortController | null;
  title: string | null;
  updatedAt: string;
}

/**
 * GlmAcpAgent implements the ACP `Agent` interface.
 *
 * It bridges the ACP protocol (via `AgentSideConnection`) and the Zhipu AI
 * GLM series models (via `GlmClient`), providing a full prompt loop with
 * tool-calling and streaming support.
 */
export class GlmAcpAgent implements Agent {
  private sessions: Map<string, SessionState> = new Map();
  private _glm: GlmClient | null = null;

  constructor(private connection: AgentSideConnection) {}

  private get glm(): GlmClient {
    if (this._glm === null) {
      this._glm = new GlmClient();
    }
    return this._glm;
  }

  // ---------------------------------------------------------------------------
  // ACP Lifecycle
  // ---------------------------------------------------------------------------

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    return {
      protocolVersion: params.protocolVersion <= VERSION ? params.protocolVersion : VERSION,
      agentInfo: {
        name: "glm-acp-agent",
        version: "1.0.0",
      },
      agentCapabilities: {
        loadSession: false,
        promptCapabilities: {
          embeddedContext: true,
        },
        sessionCapabilities: {
          close: {},
          list: {},
        },
      },
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const sessionId = randomUUID();

    const systemPrompt: GlmMessage = {
      role: "system",
      content:
        "You are an expert software engineer and coding assistant. " +
        "You help users read, write, and modify code across their projects. " +
        "Use the available tools (read_file, write_file, list_files, run_command, web_search, web_reader) " +
        "to interact with the client's file system, terminal, and the web. " +
        "Always explain what you are doing before taking any action. " +
        `Working directory: ${params.cwd}`,
    };

    this.sessions.set(sessionId, {
      cwd: params.cwd,
      messages: [systemPrompt],
      abortController: null,
      title: null,
      updatedAt: new Date().toISOString(),
    });

    return { sessionId };
  }

  async authenticate(
    _params: AuthenticateRequest
  ): Promise<AuthenticateResponse> {
    // No authentication required
    return {};
  }

  async setSessionMode(
    _params: SetSessionModeRequest
  ): Promise<SetSessionModeResponse> {
    return {};
  }

  // ---------------------------------------------------------------------------
  // Prompt Turn
  // ---------------------------------------------------------------------------

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }

    // Abort any previous in-flight prompt for this session
    session.abortController?.abort();
    const abortController = new AbortController();
    session.abortController = abortController;

    // Convert ACP content blocks to a GLM user message.
    // Baseline: text blocks. Extended: resource_link (URI reference) and
    // embedded resource blocks (inline text content).
    const userParts: string[] = [];
    for (const block of params.prompt) {
      if (block.type === "text") {
        userParts.push(block.text);
      } else if (block.type === "resource_link") {
        userParts.push(`[${block.name}](${block.uri})`);
      } else if (block.type === "resource") {
        const res = block.resource;
        if ("text" in res) {
          userParts.push(`<resource uri="${res.uri}">\n${res.text}\n</resource>`);
        }
      }
    }
    const userText = userParts.join("\n");

    session.messages.push({ role: "user", content: userText });

    try {
      const { stopReason, usage } = await this.runPromptLoop(
        params.sessionId,
        session,
        abortController.signal
      );

      session.abortController = null;
      session.updatedAt = new Date().toISOString();

      // Derive a session title from the first user message (first prompt only)
      if (!session.title) {
        session.title = userText.slice(0, 80).replace(/\n/g, " ") || "New conversation";
        await this.connection.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "session_info_update",
            title: session.title,
            updatedAt: session.updatedAt,
          },
        });
      } else {
        await this.connection.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "session_info_update",
            updatedAt: session.updatedAt,
          },
        });
      }

      // Report token usage if the model provided it
      if (usage) {
        await this.connection.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "usage_update",
            usage,
          },
        });
      }

      return {
        stopReason,
        userMessageId: params.messageId ?? undefined,
      };
    } catch (err) {
      if (abortController.signal.aborted) {
        session.abortController = null;
        return { stopReason: "cancelled" };
      }
      throw err;
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions.get(params.sessionId);
    session?.abortController?.abort();
  }

  async closeSession(params: CloseSessionRequest): Promise<void> {
    const session = this.sessions.get(params.sessionId);
    session?.abortController?.abort();
    this.sessions.delete(params.sessionId);
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const allSessions = Array.from(this.sessions.entries());

    // Optional cwd filter
    const filtered = params.cwd
      ? allSessions.filter(([, s]) => s.cwd === params.cwd)
      : allSessions;

    return {
      sessions: filtered.map(([sessionId, s]) => ({
        sessionId,
        cwd: s.cwd,
        title: s.title ?? undefined,
        updatedAt: s.updatedAt,
      })),
    };
  }

  // ---------------------------------------------------------------------------
  // Internal prompt loop
  // ---------------------------------------------------------------------------

  /**
   * Runs the full prompt/tool-calling loop until the model stops or is cancelled.
   *
   * Returns the ACP stop reason and optional token usage reported by the model.
   */
  private async runPromptLoop(
    sessionId: string,
    session: SessionState,
    signal: AbortSignal
  ): Promise<{
    stopReason: "end_turn" | "cancelled" | "max_tokens" | "max_turn_requests" | "refusal";
    usage?: { inputTokens: number; outputTokens: number; totalTokens: number };
  }> {
    const MAX_TURNS = 20;
    const executor = new ToolExecutor(this.connection, sessionId, signal);
    let lastUsage: { inputTokens: number; outputTokens: number; totalTokens: number } | undefined;
    let lastStopReason: string | undefined;

    for (let turn = 0; turn < MAX_TURNS; turn++) {
      if (signal.aborted) return { stopReason: "cancelled" };

      const toolCalls: Array<{
        id: string;
        name: string;
        arguments: string;
      }> = [];

      let assistantText = "";

      // Stream the GLM response
      for await (const chunk of this.glm.streamChat(session.messages, signal)) {
        if (signal.aborted) return { stopReason: "cancelled" };

        if (chunk.thinking) {
          // Forward reasoning tokens as ACP thought chunks
          await this.connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_thought_chunk",
              content: {
                type: "text",
                text: chunk.thinking,
              },
            },
          });
        }

        if (chunk.text) {
          assistantText += chunk.text;
          await this.connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: {
                type: "text",
                text: chunk.text,
              },
            },
          });
        }

        if (chunk.toolCall) {
          toolCalls.push(chunk.toolCall);
        }

        if (chunk.usage) {
          lastUsage = chunk.usage;
        }

        if (chunk.done) {
          lastStopReason = chunk.stopReason;
        }
      }

      // Record the assistant message in history
      if (toolCalls.length > 0) {
        session.messages.push({
          role: "assistant",
          content: assistantText || null,
          tool_calls: toolCalls.map((tc) => ({
            id: tc.id,
            type: "function" as const,
            function: {
              name: tc.name,
              arguments: tc.arguments,
            },
          })),
        });
      } else if (assistantText) {
        session.messages.push({
          role: "assistant",
          content: assistantText,
        });
      }

      // If there are no tool calls, we're done
      if (toolCalls.length === 0) {
        const stopReason = this.mapStopReason(lastStopReason);
        return {
          stopReason,
          usage: lastUsage,
        };
      }

      // Execute tool calls and feed results back
      for (const tc of toolCalls) {
        if (signal.aborted) return { stopReason: "cancelled" };

        const result = await executor.execute(tc.id, tc.name, tc.arguments);

        session.messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result.content,
        });
      }

      // Continue the loop (tool_calls finish reason means GLM wants another turn)
    }

    // MAX_TURNS reached: we exceeded internal model/tool requests for a single user turn.
    return { stopReason: "max_turn_requests", usage: lastUsage };
  }

  private mapStopReason(
    stopReason: string | undefined
  ): "end_turn" | "max_tokens" | "refusal" {
    if (stopReason === "length") return "max_tokens";
    if (stopReason === "content_filter") return "refusal";
    return "end_turn";
  }
}
