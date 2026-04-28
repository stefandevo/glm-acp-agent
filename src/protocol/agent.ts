import { randomUUID } from "node:crypto";
import type {
  Agent,
  AgentSideConnection,
  ClientCapabilities,
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
  StopReason,
  Usage,
} from "@agentclientprotocol/sdk";
import { PROTOCOL_VERSION as VERSION } from "@agentclientprotocol/sdk";
import { GlmClient, type GlmMessage, type GlmStreamChunk } from "../llm/glm-client.js";
import { ToolExecutor } from "../tools/executor.js";
import { TOOL_DEFINITIONS } from "../tools/definitions.js";

/** Per-session state */
interface SessionState {
  cwd: string;
  messages: GlmMessage[];
  abortController: AbortController | null;
  title: string | null;
  updatedAt: string;
}

/** ACP stop reasons that the prompt loop can produce internally. */
type InternalStopReason = StopReason;

/**
 * Optional dependencies for tests.
 */
export interface GlmAcpAgentOptions {
  /** Override the GLM client (used in tests). */
  glm?: { streamChat: (messages: GlmMessage[], signal?: AbortSignal) => AsyncIterable<GlmStreamChunk> };
  /** Maximum number of model/tool turns per single prompt. Default 20. */
  maxTurns?: number;
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
  private _glm: NonNullable<GlmAcpAgentOptions["glm"]> | null;
  private maxTurns: number;
  private clientCapabilities: ClientCapabilities | null = null;

  constructor(
    private connection: AgentSideConnection,
    options: GlmAcpAgentOptions = {}
  ) {
    this._glm = options.glm ?? null;
    this.maxTurns = options.maxTurns ?? 20;
  }

  private get glm(): NonNullable<GlmAcpAgentOptions["glm"]> {
    if (this._glm === null) {
      this._glm = new GlmClient();
    }
    return this._glm;
  }

  // ---------------------------------------------------------------------------
  // ACP Lifecycle
  // ---------------------------------------------------------------------------

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    // Negotiate the lowest version both sides support.
    const negotiatedVersion =
      params.protocolVersion <= VERSION ? params.protocolVersion : VERSION;

    // Track client capabilities so we can adapt tool calls to what the client
    // actually supports.
    this.clientCapabilities = params.clientCapabilities ?? null;

    return {
      protocolVersion: negotiatedVersion,
      agentInfo: {
        name: "glm-acp-agent",
        version: "1.0.0",
      },
      // Declare an environment-variable auth method so clients that respect the
      // auth-methods proposal know what to ask the user for.
      authMethods: [
        {
          type: "env_var",
          id: "z_ai_api_key",
          name: "Z.AI API key",
          description:
            "API key for the Z.AI / Zhipu AI service. Generate one at https://z.ai/manage-apikey/apikey-list",
          link: "https://z.ai/manage-apikey/apikey-list",
          vars: [
            {
              name: "Z_AI_API_KEY",
              label: "Z.AI API key",
              secret: true,
              optional: false,
            },
          ],
        },
      ],
      agentCapabilities: {
        loadSession: false,
        promptCapabilities: {
          // Baseline (text + resource_link) is implicit; we additionally accept
          // embedded resources for inline file context.
          embeddedContext: true,
        },
        sessionCapabilities: {
          close: {},
          list: {},
        },
      },
    };
  }

  async authenticate(
    _params: AuthenticateRequest
  ): Promise<AuthenticateResponse> {
    // Authentication is configured via the Z_AI_API_KEY environment variable
    // (advertised as an `env_var` auth method). The agent has nothing to do
    // here; failures will surface when the model is first called.
    return {};
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const sessionId = randomUUID();

    const tools = this.availableToolNames().join(", ");

    const systemPrompt: GlmMessage = {
      role: "system",
      content:
        "You are an expert software engineer and coding assistant. " +
        "You help users read, write, and modify code across their projects. " +
        `Use the available tools (${tools}) to interact with the client's ` +
        "file system, terminal, and the web. " +
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

  async setSessionMode(
    _params: SetSessionModeRequest
  ): Promise<SetSessionModeResponse> {
    // We don't advertise modes; this is a no-op accepting any request the
    // client might send.
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

    // The ACP spec serializes prompts per-session – the client should not send
    // a new prompt while another is running. Defensively cancel any stale
    // controller.
    session.abortController?.abort();
    const abortController = new AbortController();
    session.abortController = abortController;

    // Abort the prompt automatically if the underlying connection closes.
    const onConnectionClose = () => abortController.abort();
    const connSignal = this.connection.signal;
    if (connSignal && !connSignal.aborted) {
      connSignal.addEventListener("abort", onConnectionClose, { once: true });
    } else if (connSignal?.aborted) {
      abortController.abort();
    }

    // Convert ACP content blocks into a GLM user message.
    // Baseline: text + resource_link. Optional: embedded resources (text only).
    const userText = renderPromptBlocks(params.prompt);
    session.messages.push({ role: "user", content: userText });

    try {
      const { stopReason, usage } = await this.runPromptLoop(
        params.sessionId,
        session,
        abortController.signal
      );

      session.abortController = null;
      session.updatedAt = new Date().toISOString();

      // Emit a session_info_update with the (possibly first-set) title and
      // updated timestamp so clients can show fresh metadata.
      const titleUpdate: { title?: string | null } =
        session.title === null
          ? (() => {
              const derived = userText.slice(0, 80).replace(/\s+/g, " ").trim();
              session.title = derived.length > 0 ? derived : "New conversation";
              return { title: session.title };
            })()
          : {};

      await this.connection.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "session_info_update",
          updatedAt: session.updatedAt,
          ...titleUpdate,
        },
      });

      const response: PromptResponse = { stopReason };
      if (usage) response.usage = usage;
      if (params.messageId) response.userMessageId = params.messageId;
      return response;
    } catch (err) {
      // If the abort happened concurrently with another error, prefer the
      // cancelled stop reason – that's what the spec asks for.
      if (abortController.signal.aborted) {
        session.abortController = null;
        return { stopReason: "cancelled" };
      }
      session.abortController = null;
      // Surface the error to the user as an agent message so the IDE displays
      // something instead of a silent JSON-RPC error.
      const message = err instanceof Error ? err.message : String(err);
      await safeSessionUpdate(this.connection, {
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `\n\n[error] ${message}` },
        },
      });
      throw err;
    } finally {
      connSignal?.removeEventListener("abort", onConnectionClose);
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
  ): Promise<{ stopReason: InternalStopReason; usage?: Usage }> {
    const executor = new ToolExecutor(
      this.connection,
      sessionId,
      this.clientCapabilities,
      signal
    );

    let lastUsage: Usage | undefined;

    for (let turn = 0; turn < this.maxTurns; turn++) {
      if (signal.aborted) return { stopReason: "cancelled" };

      const toolCalls: Array<{
        id: string;
        name: string;
        arguments: string;
      }> = [];

      let assistantText = "";
      let lastStopReason: string | undefined;

      // Stream the GLM response.
      for await (const chunk of this.glm.streamChat(session.messages, signal)) {
        if (signal.aborted) return { stopReason: "cancelled" };

        if (chunk.thinking) {
          await this.connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_thought_chunk",
              content: { type: "text", text: chunk.thinking },
            },
          });
        }

        if (chunk.text) {
          assistantText += chunk.text;
          await this.connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: chunk.text },
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

      // Record the assistant turn in history so the model has full context for
      // the next iteration.
      if (toolCalls.length > 0) {
        session.messages.push({
          role: "assistant",
          content: assistantText.length > 0 ? assistantText : null,
          tool_calls: toolCalls.map((tc) => ({
            id: tc.id,
            type: "function" as const,
            function: { name: tc.name, arguments: tc.arguments },
          })),
        });
      } else if (assistantText.length > 0) {
        session.messages.push({ role: "assistant", content: assistantText });
      }

      // No tool calls => model is done.
      if (toolCalls.length === 0) {
        return { stopReason: this.mapStopReason(lastStopReason), usage: lastUsage };
      }

      // Execute tool calls in declaration order and feed each result back.
      for (const tc of toolCalls) {
        if (signal.aborted) return { stopReason: "cancelled", usage: lastUsage };

        const result = await executor.execute(tc.id, tc.name, tc.arguments);

        session.messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result.content,
        });
      }

      // Loop and continue – GLM expects a follow-up completion now that it has
      // tool results.
    }

    // Reached MAX_TURNS without resolution.
    return { stopReason: "max_turn_requests", usage: lastUsage };
  }

  private mapStopReason(stopReason: string | undefined): InternalStopReason {
    switch (stopReason) {
      case "length":
        return "max_tokens";
      case "content_filter":
        return "refusal";
      case "stop":
      case "tool_calls":
      case undefined:
      case null:
      case "":
        return "end_turn";
      default:
        return "end_turn";
    }
  }

  /** Names of tools we expose given the client's capabilities. */
  private availableToolNames(): string[] {
    const fs = this.clientCapabilities?.fs ?? {};
    const terminal = this.clientCapabilities?.terminal ?? false;
    const names: string[] = [];
    if (fs.readTextFile) names.push("read_file");
    if (fs.writeTextFile) names.push("write_file");
    if (terminal) names.push("list_files", "run_command");
    // Web tools always run inside the agent process, so they are unconditional.
    names.push("web_search", "web_reader");
    // If everything is missing fall back to the full list so the system prompt
    // still mentions tools by name (the executor will surface a clean error if
    // they are invoked).
    if (names.length === 2) return TOOL_DEFINITIONS.map((t) => t.function.name);
    return names;
  }
}

/** Render a list of ACP content blocks into a single user message string. */
function renderPromptBlocks(blocks: PromptRequest["prompt"]): string {
  const parts: string[] = [];
  for (const block of blocks) {
    switch (block.type) {
      case "text":
        parts.push(block.text);
        break;
      case "resource_link":
        parts.push(`[${block.name}](${block.uri})`);
        break;
      case "resource": {
        const res = block.resource;
        if ("text" in res && typeof res.text === "string") {
          parts.push(`<resource uri="${res.uri}">\n${res.text}\n</resource>`);
        } else if ("blob" in res) {
          // Binary resources can't be inlined into a chat message; keep a link
          // reference so the model knows it exists.
          parts.push(`[binary resource](${res.uri})`);
        }
        break;
      }
      case "image":
      case "audio":
        // We don't advertise image/audio capabilities, but defensively render
        // a placeholder so a misbehaving client doesn't crash the agent.
        parts.push(`[unsupported ${block.type} block]`);
        break;
      default:
        // Future block types: best-effort placeholder.
        parts.push(`[unknown block type ${(block as { type: string }).type}]`);
    }
  }
  return parts.join("\n");
}

/** sessionUpdate that swallows transport errors during error reporting. */
async function safeSessionUpdate(
  connection: AgentSideConnection,
  params: Parameters<AgentSideConnection["sessionUpdate"]>[0]
): Promise<void> {
  try {
    await connection.sessionUpdate(params);
  } catch {
    // best-effort
  }
}
