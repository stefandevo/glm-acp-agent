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
  PROTOCOL_VERSION,
} from "@agentclientprotocol/sdk";
import { PROTOCOL_VERSION as VERSION } from "@agentclientprotocol/sdk";
import { GlmClient, type GlmMessage } from "../llm/glm-client.js";
import { ToolExecutor } from "../tools/executor.js";

/** Per-session state */
interface SessionState {
  messages: GlmMessage[];
  abortController: AbortController | null;
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
  private glm: GlmClient;

  constructor(private connection: AgentSideConnection) {
    this.glm = new GlmClient();
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
        sessionCapabilities: {
          close: {},
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
        "Use the available tools (read_file, write_file, list_files, run_command) " +
        "to interact with the client's file system and terminal. " +
        "Always explain what you are doing before taking any action. " +
        `Working directory: ${params.cwd}`,
    };

    this.sessions.set(sessionId, {
      messages: [systemPrompt],
      abortController: null,
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

    // Convert ACP content blocks to a GLM user message
    const userText = params.prompt
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("\n");

    session.messages.push({ role: "user", content: userText });

    try {
      const stopReason = await this.runPromptLoop(
        params.sessionId,
        session,
        abortController.signal
      );

      session.abortController = null;
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

  // ---------------------------------------------------------------------------
  // Internal prompt loop
  // ---------------------------------------------------------------------------

  /**
   * Runs the full prompt/tool-calling loop until the model stops or is cancelled.
   *
   * Returns the ACP stop reason.
   */
  private async runPromptLoop(
    sessionId: string,
    session: SessionState,
    signal: AbortSignal
  ): Promise<"end_turn" | "cancelled" | "max_tokens"> {
    const MAX_TURNS = 20;
    const executor = new ToolExecutor(this.connection, sessionId);

    for (let turn = 0; turn < MAX_TURNS; turn++) {
      if (signal.aborted) return "cancelled";

      const toolCalls: Array<{
        id: string;
        name: string;
        arguments: string;
      }> = [];

      let assistantText = "";

      // Stream the GLM response
      for await (const chunk of this.glm.streamChat(session.messages)) {
        if (signal.aborted) return "cancelled";

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

        if (chunk.done) {
          // finish_reason recorded; tool calls are emitted by the stream generator
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
        return "end_turn";
      }

      // Execute tool calls and feed results back
      for (const tc of toolCalls) {
        if (signal.aborted) return "cancelled";

        const result = await executor.execute(tc.id, tc.name, tc.arguments);

        session.messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result.content,
        });
      }

      // Continue the loop (tool_calls finish reason means GLM wants another turn)
    }

    return "max_tokens";
  }
}
