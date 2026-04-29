import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join as pathJoin } from "node:path";
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
  LoadSessionRequest,
  LoadSessionResponse,
  ForkSessionRequest,
  ForkSessionResponse,
  ResumeSessionRequest,
  ResumeSessionResponse,
  SetSessionModelRequest,
  SetSessionModelResponse,
  ModelInfo,
  StopReason,
  Usage,
} from "@agentclientprotocol/sdk";
import { PROTOCOL_VERSION as VERSION } from "@agentclientprotocol/sdk";
import {
  GlmClient,
  getAvailableModels,
  getDefaultModel,
  type GlmMessage,
  type GlmStreamChunk,
  type StreamChatOptions,
} from "../llm/glm-client.js";
import { ToolExecutor } from "../tools/executor.js";
import { TOOL_DEFINITIONS } from "../tools/definitions.js";
import { SessionStore, type PersistedSession } from "./session-store.js";
import { buildSystemPrompt } from "./system-prompt.js";
import { debug, error } from "../llm/logger.js";

/**
 * Maximum bytes of AGENTS.md / CLAUDE.md to embed in the system prompt.
 * Caps the input-token cost of a single project's context. Z.AI prompt
 * caching will absorb repeated reads across turns of the same session, but
 * this still bounds the worst case for projects that ship enormous spec
 * files in their AGENTS.md.
 */
const PROJECT_CONTEXT_CAP_CHARS = 8 * 1024;

/** Per-session state */
interface SessionState {
  cwd: string;
  messages: GlmMessage[];
  abortController: AbortController | null;
  /**
   * Promise that resolves once the currently-running prompt loop has fully
   * unwound. Tracking this lets a follow-up prompt wait for the previous loop
   * to observe its abort before mutating shared session state.
   */
  promptPromise: Promise<void> | null;
  title: string | null;
  updatedAt: string;
  /** Active model for this session (clients can change via `session/set_model`). */
  model: string;
}

/** ACP stop reasons that the prompt loop can produce internally. */
type InternalStopReason = StopReason;

/**
 * Optional dependencies for tests.
 */
export interface GlmAcpAgentOptions {
  /** Override the GLM client (used in tests). */
  glm?: {
    streamChat: (
      messages: GlmMessage[],
      signal?: AbortSignal,
      options?: StreamChatOptions
    ) => AsyncIterable<GlmStreamChunk>;
  };
  /** Maximum number of model/tool turns per single prompt. Default 20. */
  maxTurns?: number;
  /**
   * Override the session store (used in tests). When undefined the agent
   * uses an on-disk store rooted at `$ACP_GLM_SESSION_DIR` /
   * `$XDG_STATE_HOME/glm-acp-agent/sessions` / `~/.local/state/glm-acp-agent/sessions`.
   * Pass `null` to disable persistence entirely.
   */
  sessionStore?: SessionStore | null;
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
  private sessionStore: SessionStore | null;

  constructor(
    private connection: AgentSideConnection,
    options: GlmAcpAgentOptions = {}
  ) {
    this._glm = options.glm ?? null;
    const candidateMaxTurns = options.maxTurns ?? 20;
    this.maxTurns =
      Number.isFinite(candidateMaxTurns) && candidateMaxTurns > 0
        ? Math.floor(candidateMaxTurns)
        : 20;
    this.sessionStore =
      options.sessionStore === null
        ? null
        : (options.sessionStore ?? new SessionStore());
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
      // Advertise auth methods so the ACP registry verifier and capable
      // clients can discover how to configure us. The `agent`-default method
      // (no `type` discriminator) signals that the agent reads its credentials
      // itself at startup; the experimental `env_var` method gives clients
      // that support it the metadata to prompt the user for the right var.
      authMethods: [
        {
          id: "z-ai-api-key",
          name: "Z.AI API key",
          description:
            "Set Z_AI_API_KEY in the environment, or run `glm-acp-agent --setup` once to store the key on disk. Generate one at https://z.ai/manage-apikey/apikey-list",
        },
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
        loadSession: true,
        promptCapabilities: {
          // Baseline (text + resource_link) is implicit; we additionally accept
          // embedded resources for inline file context, plus images for
          // vision-capable GLM models (e.g. glm-4v-plus).
          embeddedContext: true,
          image: true,
        },
        sessionCapabilities: {
          close: {},
          list: {},
          fork: {},
          resume: {},
        },
      },
    };
  }

  async authenticate(
    _params: AuthenticateRequest
  ): Promise<AuthenticateResponse> {
    // Authentication is configured externally — either via Z_AI_API_KEY in the
    // environment or via the credentials file written by `glm-acp-agent --setup`.
    // The agent has nothing to do here; failures will surface when the model
    // is first called.
    return {};
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    const sessionId = randomUUID();
    debug(`newSession: id=${sessionId} cwd=${params.cwd} model=${getDefaultModel()}`);

    const systemPrompt: GlmMessage = {
      role: "system",
      content: buildSystemPrompt({
        cwd: params.cwd,
        tools: this.availableToolNames(),
        agentsMd: loadProjectContext(params.cwd),
      }),
    };

    const model = getDefaultModel();

    this.sessions.set(sessionId, {
      cwd: params.cwd,
      messages: [systemPrompt],
      abortController: null,
      promptPromise: null,
      title: null,
      updatedAt: new Date().toISOString(),
      model,
    });

    return {
      sessionId,
      models: this.modelsState(model),
    };
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest
  ): Promise<SetSessionModelResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }
    const available = getAvailableModels();
    const known = available.find((m) => m.modelId === params.modelId);
    if (!known) {
      // Allow the caller to pick any model id even if not in the curated list
      // (Z.AI may offer models we haven't catalogued), but log a stderr hint.
      process.stderr.write(
        `[glm-acp-agent] warning: model "${params.modelId}" is not in the advertised list; using as-is.\n`
      );
    }
    session.model = params.modelId;
    session.updatedAt = new Date().toISOString();
    // Notify clients so any UI that displays the active model refreshes
    // immediately, instead of waiting for the next prompt to complete.
    await safeSessionUpdate(this.connection, {
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "session_info_update",
        updatedAt: session.updatedAt,
      },
    });
    return {};
  }

  /** Build the SessionModelState we advertise on session create/load/resume/fork. */
  private modelsState(currentModelId: string): {
    availableModels: ModelInfo[];
    currentModelId: string;
  } {
    return {
      availableModels: getAvailableModels(),
      currentModelId,
    };
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
    // controller and wait for the previous loop to fully unwind so it stops
    // mutating session.messages or emitting updates before we start the new
    // one.
    if (session.abortController) {
      session.abortController.abort();
    }
    if (session.promptPromise) {
      try {
        await session.promptPromise;
      } catch {
        // The previous loop's failure is already reported back to its caller.
      }
    }

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

    // Convert ACP content blocks into a GLM user message. Baseline: text +
    // resource_link. Optional: embedded resources, plus image blocks (forwarded
    // as OpenAI-shaped data-URL parts so vision-capable GLM models like
    // glm-4v-plus can ingest them).
    const { content: userContent, plainText: userText } = renderPromptBlocks(
      params.prompt
    );
    session.messages.push({ role: "user", content: userContent });

    // Echo back the client-supplied messageId on every response (success,
    // cancelled, or error) so the client can correlate the turn.
    const userMessageId = params.messageId ?? undefined;

    let resolvePromptPromise!: () => void;
    session.promptPromise = new Promise<void>((resolve) => {
      resolvePromptPromise = resolve;
    });

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

      this.persistSession(params.sessionId, session);

      const response: PromptResponse = { stopReason };
      if (usage) response.usage = usage;
      if (userMessageId) response.userMessageId = userMessageId;
      return response;
    } catch (err) {
      error(`prompt error: session=${params.sessionId}`, err instanceof Error ? err.message : String(err));
      // If the abort happened concurrently with another error, prefer the
      // cancelled stop reason – that's what the spec asks for.
      if (abortController.signal.aborted) {
        session.abortController = null;
        const cancelled: PromptResponse = { stopReason: "cancelled" };
        if (userMessageId) cancelled.userMessageId = userMessageId;
        return cancelled;
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
      // Always resolve the promptPromise so a subsequent prompt can proceed.
      session.promptPromise = null;
      resolvePromptPromise();
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions.get(params.sessionId);
    session?.abortController?.abort();
  }

  async closeSession(params: CloseSessionRequest): Promise<void> {
    const session = this.sessions.get(params.sessionId);
    session?.abortController?.abort();
    if (session) {
      // Persist final state on close so a subsequent loadSession/resume can
      // pick the conversation back up. closeSession only releases in-memory
      // resources; the on-disk record is intentionally retained.
      this.persistSession(params.sessionId, session);
    }
    this.sessions.delete(params.sessionId);
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    // Merge in-memory sessions with anything previously persisted to disk. The
    // store is the source of truth for closed/restarted sessions; in-memory
    // state takes precedence when both exist (it has the freshest title /
    // updatedAt before persistence has fired).
    //
    // We use the metadata-only store API so we don't load every conversation's
    // full message history just to render a session picker.
    const merged = new Map<
      string,
      { cwd: string; title: string | null; updatedAt: string }
    >();

    if (this.sessionStore) {
      for (const meta of this.sessionStore.listMetadata()) {
        merged.set(meta.sessionId, {
          cwd: meta.cwd,
          title: meta.title,
          updatedAt: meta.updatedAt,
        });
      }
    }
    for (const [sessionId, s] of this.sessions) {
      merged.set(sessionId, {
        cwd: s.cwd,
        title: s.title,
        updatedAt: s.updatedAt,
      });
    }

    const all = Array.from(merged.entries());
    const filtered = params.cwd
      ? all.filter(([, s]) => s.cwd === params.cwd)
      : all;
    // Newest-first so clients can render the picker without re-sorting.
    filtered.sort(([, a], [, b]) =>
      a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0
    );

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
  // Session load / fork / resume
  // ---------------------------------------------------------------------------

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const persisted = this.requirePersisted(params.sessionId);

    // Restore in-memory state. We do NOT carry over the abortController /
    // promptPromise — those are transient.
    const restored: SessionState = {
      cwd: params.cwd,
      messages: persisted.messages,
      abortController: null,
      promptPromise: null,
      title: persisted.title,
      updatedAt: persisted.updatedAt,
      model: persisted.model,
    };
    this.sessions.set(params.sessionId, restored);

    // Replay user/assistant text turns so the client can rehydrate its UI.
    // Tool / system messages are skipped — they're internal and the client
    // doesn't render them on its own.
    await this.replayMessages(params.sessionId, persisted.messages);

    return { models: this.modelsState(persisted.model) };
  }

  async unstable_forkSession(
    params: ForkSessionRequest
  ): Promise<ForkSessionResponse> {
    const source = this.sessions.get(params.sessionId);
    const persisted = source
      ? this.snapshot(params.sessionId, source)
      : this.requirePersisted(params.sessionId);

    const newSessionId = randomUUID();
    const forkedTitle =
      persisted.title === null ? null : `${persisted.title} (fork)`;
    const forked: SessionState = {
      cwd: params.cwd,
      // Deep-clone messages so the fork doesn't share state with the parent.
      messages: structuredClone(persisted.messages),
      abortController: null,
      promptPromise: null,
      title: forkedTitle,
      updatedAt: new Date().toISOString(),
      model: persisted.model,
    };
    this.sessions.set(newSessionId, forked);
    this.persistSession(newSessionId, forked);

    return {
      sessionId: newSessionId,
      models: this.modelsState(forked.model),
    };
  }

  async resumeSession(
    params: ResumeSessionRequest
  ): Promise<ResumeSessionResponse> {
    const persisted = this.requirePersisted(params.sessionId);

    const restored: SessionState = {
      cwd: params.cwd,
      messages: persisted.messages,
      abortController: null,
      promptPromise: null,
      title: persisted.title,
      updatedAt: persisted.updatedAt,
      model: persisted.model,
    };
    this.sessions.set(params.sessionId, restored);

    // Resume does NOT replay history — the client keeps its own UI state and
    // just wants the agent to pick up where it left off.
    return { models: this.modelsState(persisted.model) };
  }

  // ---------------------------------------------------------------------------
  // Persistence helpers
  // ---------------------------------------------------------------------------

  private snapshot(sessionId: string, session: SessionState): PersistedSession {
    return {
      sessionId,
      cwd: session.cwd,
      messages: session.messages,
      title: session.title,
      updatedAt: session.updatedAt,
      model: session.model,
    };
  }

  private persistSession(sessionId: string, session: SessionState): void {
    if (!this.sessionStore) return;
    try {
      this.sessionStore.save(this.snapshot(sessionId, session));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      process.stderr.write(
        `[glm-acp-agent] warning: failed to persist session ${sessionId}: ${msg}\n`
      );
    }
  }

  private requirePersisted(sessionId: string): PersistedSession {
    if (!this.sessionStore) {
      throw new Error("Session persistence is disabled");
    }
    const persisted = this.sessionStore.load(sessionId);
    if (!persisted) {
      throw new Error(`Session not found: ${sessionId}`);
    }
    return persisted;
  }

  private async replayMessages(
    sessionId: string,
    messages: GlmMessage[]
  ): Promise<void> {
    for (const msg of messages) {
      if (msg.role === "user") {
        const text = stringifyUserMessage(msg.content);
        if (text.length === 0) continue;
        await this.connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "user_message_chunk",
            content: { type: "text", text },
          },
        });
      } else if (msg.role === "assistant") {
        const content = msg.content;
        const text =
          typeof content === "string"
            ? content
            : Array.isArray(content)
              ? content
                  .filter(
                    (p): p is { type: "text"; text: string } =>
                      typeof p === "object" &&
                      p !== null &&
                      (p as { type?: unknown }).type === "text" &&
                      typeof (p as { text?: unknown }).text === "string"
                  )
                  .map((p) => p.text)
                  .join("")
              : "";
        if (text.length === 0) continue;
        await this.connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text },
          },
        });
      }
      // system / tool messages are not replayed — they're internal.
    }
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

      debug(`promptLoop: turn=${turn} session=${sessionId} model=${session.model} messages=${session.messages.length}`);

      const toolCalls: Array<{
        id: string;
        name: string;
        arguments: string;
      }> = [];

      let assistantText = "";
      let lastStopReason: string | undefined;

      // Stream the GLM response. The session's currently-selected model wins;
      // it's mutated by `unstable_setSessionModel` between turns.
      for await (const chunk of this.glm.streamChat(session.messages, signal, {
        model: session.model,
      })) {
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
          debug(`promptLoop: toolCall id=${chunk.toolCall.id} name=${chunk.toolCall.name}`);
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
        debug(`promptLoop: toolResult id=${tc.id} name=${tc.name} contentLength=${result.content.length}`);

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
    // If the client never sent capabilities at all (initialize wasn't called,
    // or it omitted the field), advertise the full set so the system prompt
    // still mentions every tool. The executor will surface a clean error if
    // the model invokes one whose capability is missing.
    if (!this.clientCapabilities) {
      return TOOL_DEFINITIONS.map((t) => t.function.name);
    }

    const fs = this.clientCapabilities.fs ?? {};
    const terminal = this.clientCapabilities.terminal ?? false;
    const names: string[] = [];
    if (fs.readTextFile) names.push("read_file");
    if (fs.writeTextFile) names.push("write_file");
    if (terminal) names.push("list_files", "run_command");
    // Web tools always run inside the agent process, so they are unconditional.
    names.push("web_search", "web_reader");
    return names;
  }
}

/** OpenAI-shaped content part for multimodal user messages. */
type UserContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

/**
 * Render a list of ACP content blocks into a GLM user message.
 *
 * When the prompt contains only text/resource blocks we return a plain string
 * (the broadest-compatible shape). When at least one image block is present
 * we switch to OpenAI's multimodal array shape so vision-capable GLM models
 * receive the image as a data URL.
 *
 * `plainText` always contains the text-only flattening of the prompt and is
 * used by the agent to derive a session title.
 */
function renderPromptBlocks(blocks: PromptRequest["prompt"]): {
  content: string | UserContentPart[];
  plainText: string;
} {
  const textParts: string[] = [];
  const imageParts: UserContentPart[] = [];

  for (const block of blocks) {
    switch (block.type) {
      case "text":
        textParts.push(block.text);
        break;
      case "resource_link":
        textParts.push(`[${block.name}](${block.uri})`);
        break;
      case "resource": {
        const res = block.resource;
        if ("text" in res && typeof res.text === "string") {
          textParts.push(`<resource uri="${res.uri}">\n${res.text}\n</resource>`);
        } else if ("blob" in res) {
          // Binary resources can't be inlined into a chat message; keep a link
          // reference so the model knows it exists.
          textParts.push(`[binary resource](${res.uri})`);
        }
        break;
      }
      case "image": {
        const dataUrl = `data:${block.mimeType};base64,${block.data}`;
        imageParts.push({ type: "image_url", image_url: { url: dataUrl } });
        break;
      }
      case "audio":
        textParts.push("[unsupported audio block]");
        break;
      default:
        textParts.push(`[unknown block type ${(block as { type: string }).type}]`);
    }
  }

  const plainText = textParts.join("\n");

  if (imageParts.length === 0) {
    return { content: plainText, plainText };
  }

  // Multimodal: combine text + images. OpenAI requires at least one part, so
  // we always include the text part (even when empty, the array form is valid).
  const content: UserContentPart[] = [];
  if (plainText.length > 0) {
    content.push({ type: "text", text: plainText });
  }
  content.push(...imageParts);
  return { content, plainText };
}

/**
 * Flatten the `content` of a user message into a plain string for replay.
 *
 * Multimodal user messages (text + image parts) are rendered as their text
 * portions only; images are noted with a placeholder so the client knows
 * something visual was there but doesn't try to re-render an opaque
 * `image_url` part it never produced.
 */
function stringifyUserMessage(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const part of content) {
    if (typeof part !== "object" || part === null) continue;
    const type = (part as { type?: unknown }).type;
    if (type === "text") {
      const text = (part as { text?: unknown }).text;
      if (typeof text === "string") parts.push(text);
    } else if (type === "image_url") {
      parts.push("[image]");
    }
  }
  return parts.join("\n");
}

/**
 * Read an `AGENTS.md` (preferred) or `CLAUDE.md` from the session's cwd, returning
 * its contents capped to {@link PROJECT_CONTEXT_CAP_CHARS} characters. Read errors
 * (file missing, no permission, directory missing) are intentionally swallowed —
 * project context is optional, and a missing file is the common case.
 *
 * Called once at `newSession` time (not per prompt) so the project context is
 * stable across the conversation.
 */
function loadProjectContext(cwd: string): string | undefined {
  for (const filename of ["AGENTS.md", "CLAUDE.md"] as const) {
    let contents: string;
    try {
      contents = readFileSync(pathJoin(cwd, filename), { encoding: "utf-8" });
    } catch {
      continue;
    }
    if (contents.length > PROJECT_CONTEXT_CAP_CHARS) {
      contents = contents.slice(0, PROJECT_CONTEXT_CAP_CHARS);
    }
    return contents;
  }
  return undefined;
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
