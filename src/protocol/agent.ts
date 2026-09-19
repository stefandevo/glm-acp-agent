import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join as pathJoin } from "node:path";
import type { ChatCompletionContentPart } from "openai/resources/index.js";
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
  SetSessionConfigOptionRequest,
  SetSessionConfigOptionResponse,
  SessionConfigOption,
  AvailableCommand,
  ModelInfo,
  StopReason,
  Usage,
} from "@agentclientprotocol/sdk";
import { PROTOCOL_VERSION as VERSION } from "@agentclientprotocol/sdk";
import {
  GlmClient,
  getAvailableModels,
  getDefaultModel,
  getContextWindow,
  isVisionNativeModel,
  ERR_CONTEXT_OVERFLOW,
  getThoughtLevels,
  resolveThoughtLevel,
  isThoughtLevel,
  type GlmMessage,
  type GlmStreamChunk,
  type StreamChatOptions,
  type ThoughtLevel,
} from "../llm/glm-client.js";
import { ToolExecutor, type TodoItem } from "../tools/executor.js";
import { ProcessSupervisor } from "../tools/process-supervisor.js";
import { TOOL_DEFINITIONS, type ToolDefinition } from "../tools/definitions.js";
import { connectSessionMcpServers, type SessionMcpTools } from "../tools/session-mcp-client.js";
import { SessionStore, type PersistedSession } from "./session-store.js";
import { SessionLifecycle, type TransitionLease } from "./session-lifecycle.js";
import { buildSystemPrompt } from "./system-prompt.js";
import {
  discoverSlashCommands,
  parseSlashCommand,
  renderSlashCommand,
  type SlashCommand,
} from "./slash-commands.js";
import { preprocessImageBlocks, buildPromptBlockDiagnosticLines } from "./image-preprocessor.js";
import { StdioVisionMcpClient, type VisionMcpClient } from "../tools/vision-mcp-client.js";
import { resolveApiKey } from "../llm/credentials.js";
import { debug, error, isDebugEnabled } from "../llm/logger.js";

/**
 * Maximum bytes of AGENTS.md / CLAUDE.md to embed in the system prompt.
 * Caps the input-token cost of a single project's context. Z.AI prompt
 * caching will absorb repeated reads across turns of the same session, but
 * this still bounds the worst case for projects that ship enormous spec
 * files in their AGENTS.md.
 */
const PROJECT_CONTEXT_CAP_CHARS = 8 * 1024;

/**
 * ACP session mode identifiers. These control when the agent requests user
 * permission for tool calls that mutate state.
 */
export type SessionModeId = "default" | "accept_edits" | "bypass_permissions";

/**
 * The session modes we advertise, in the order clients should display them.
 *
 * Single source of truth for both the ACP `modes` state (`session/set_mode`)
 * and the `mode`-category SessionConfigOption. Clients like Zed suppress the
 * legacy mode selector as soon as an agent advertises any config option, so
 * the same list has to reach the UI through both channels.
 */
const SESSION_MODES: ReadonlyArray<{
  id: SessionModeId;
  name: string;
  description: string;
}> = [
  {
    id: "default",
    name: "Ask for permission",
    description: "Prompt before edits and commands.",
  },
  {
    id: "accept_edits",
    name: "Auto-approve edits",
    description: "Edits run without prompting. Commands still prompt.",
  },
  {
    id: "bypass_permissions",
    name: "Bypass all permissions",
    description: "Edits and commands run without prompting.",
  },
];

const SESSION_MODE_IDS: SessionModeId[] = SESSION_MODES.map((mode) => mode.id);

function isSessionModeId(value: unknown): value is SessionModeId {
  return typeof value === "string" && SESSION_MODE_IDS.includes(value as SessionModeId);
}

/**
 * Display names for thought levels shown in client UIs. Kept as an exhaustive
 * map (not capitalize-first-letter) so `xhigh` renders as "X-High" and adding
 * a level forces a conscious naming decision here.
 */
const THOUGHT_LEVEL_NAMES: Record<ThoughtLevel, string> = {
  none: "Off",
  on: "On",
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
  xhigh: "X-High",
  max: "Max",
};

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
  /** True while closeSession is waiting for the active prompt to unwind. */
  closing: boolean;
  /** True after closeSession has disposed and removed this session. */
  closed: boolean;
  /** De-duplicates concurrent closeSession calls. */
  closePromise: Promise<void> | null;
  /** De-duplicates MCP cleanup between closeSession and agent shutdown. */
  disposePromise: Promise<void> | null;
  title: string | null;
  updatedAt: string;
  /** Active model for this session (clients can change via `session/set_model`). */
  model: string;
  /** Per-session tool schemas, including client-supplied MCP tools. */
  toolDefinitions: ToolDefinition[];
  /** Connected client-supplied MCP tools for this session. */
  mcpTools: SessionMcpTools | null;
  /** Active permission mode for this session (clients can change via `session/set_mode`). */
  mode: SessionModeId;
  /** Reasoning effort for this session, controlled via the `thought_level` config option. */
  thoughtLevel: ThoughtLevel;
  /** Slash commands discovered under this session's cwd, advertised to the client. */
  commands: SlashCommand[];
  /**
   * Replay text for user messages whose stored content is not what the user
   * typed — a slash command expanded into its body, an image swapped for its
   * vision annotation. The model reads `messages`; the client's transcript
   * reads this.
   *
   * Keyed by message identity rather than by position: `compactMessages`
   * evicts whole turns from `messages`, and an index-keyed map would then
   * point at the wrong message. A WeakMap also drops evicted entries on its
   * own. Serialized to indices at save time and rebuilt on load.
   */
  displayText: WeakMap<GlmMessage, string>;
  /** Synchronous gate for replacement and close transitions. */
  lifecycle: SessionLifecycle;
}

interface SessionTransition {
  lifecycle: SessionLifecycle;
  promise: Promise<unknown> | null;
  closePromise: Promise<void> | null;
  original: SessionState | null;
  restoreAbortController: AbortController | null;
}

/** A connected MCP client that remains agent-owned until its session installs it. */
interface PendingMcpSetup {
  tools: Promise<SessionMcpTools>;
  release: () => void;
  dispose: () => Promise<void>;
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
  /**
   * Maximum number of model/tool turns per single prompt. Default 100,
   * overridable via `$ACP_GLM_MAX_TURNS`.
   */
  maxTurns?: number;
  /**
   * Override the session store (used in tests). When undefined the agent
   * uses an on-disk store rooted at `$ACP_GLM_SESSION_DIR` /
   * `$XDG_STATE_HOME/glm-acp-agent/sessions` / `~/.local/state/glm-acp-agent/sessions`.
   * Pass `null` to disable persistence entirely.
   */
  sessionStore?: SessionStore | null;
  /**
   * Vision MCP client used to analyze ACP image blocks via @z_ai/mcp-server.
   * Pass `null` to disable vision entirely (image blocks degrade to a text
   * placeholder). When undefined the agent lazy-creates a StdioVisionMcpClient
   * on first use.
   */
  visionClient?: VisionMcpClient | null;
  /** Test-only override for the bounded live-restore prompt drain. */
  sessionDrainTimeoutMs?: number;
  /** Override session MCP setup for deterministic lifecycle tests. */
  connectSessionMcpServers?: typeof connectSessionMcpServers;
  /** Override session MCP setup for shutdown/lifecycle tests. */
  mcpConnector?: (servers: Parameters<typeof connectSessionMcpServers>[0], signal?: AbortSignal) => Promise<SessionMcpTools>;
  /** Test-only override for the bounded shutdown prompt-drain deadline. */
  shutdownDrainTimeoutMs?: number;
}

/**
 * GlmAcpAgent implements the ACP `Agent` interface.
 *
 * It bridges the ACP protocol (via `AgentSideConnection`) and the Zhipu AI
 * GLM series models (via `GlmClient`), providing a full prompt loop with
 * tool-calling and streaming support.
 */
// 20 turns is a handful of tool calls; real editing sessions routinely need
// more, and hitting the cap mid-task surfaces as Zed's "reached the turn
// limit — send a message to continue". Still overridable via ACP_GLM_MAX_TURNS.
export const DEFAULT_MAX_TURNS = 100;

/**
 * Resolve the fallback maxTurns from `$ACP_GLM_MAX_TURNS` when the caller did
 * not pass an explicit value. Returns `undefined` when unset or invalid (the
 * constructor then applies the default, and logs a warning on invalid input).
 * Values that floor below 1 (e.g. 0.5) count as invalid — flooring them to 0
 * would just be silently replaced by the default in the constructor.
 */
function envMaxTurns(): number | undefined {
  const raw = process.env["ACP_GLM_MAX_TURNS"];
  if (raw === undefined || raw === "") return undefined;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || Math.floor(parsed) < 1) {
    process.stderr.write(
      `glm-acp-agent: ignoring invalid ACP_GLM_MAX_TURNS="${raw}"\n`
    );
    return undefined;
  }
  return Math.floor(parsed);
}

export class GlmAcpAgent implements Agent {
  private sessions: Map<string, SessionState> = new Map();
  private sessionTodos: Map<string, TodoItem[]> = new Map();
  private _glm: NonNullable<GlmAcpAgentOptions["glm"]> | null;
  private maxTurns: number;
  /** Forward GLM reasoning to the client as agent_thought_chunk. Default on; disable with ACP_GLM_STREAM_THINKING=false. */
  private streamThinking: boolean;
  private clientCapabilities: ClientCapabilities | null = null;
  private sessionStore: SessionStore | null;
  private _visionClient: VisionMcpClient | null;
  private visionClientExplicit: boolean;
  private sessionDrainTimeoutMs: number;
  private connectMcpServers: typeof connectSessionMcpServers;
  private transitions = new Map<string, SessionTransition>();
  private readonly mcpConnector: NonNullable<GlmAcpAgentOptions["mcpConnector"]>;
  private readonly pendingSetups = new Set<PendingMcpSetup>();
  private readonly processSupervisor = new ProcessSupervisor();
  private shuttingDown = false;
  private shutdownPromise: Promise<void> | null = null;
  private readonly shutdownDrainTimeoutMs: number;

  constructor(
    private connection: AgentSideConnection,
    options: GlmAcpAgentOptions = {}
  ) {
    this._glm = options.glm ?? null;
    const candidateMaxTurns = options.maxTurns ?? envMaxTurns() ?? DEFAULT_MAX_TURNS;
    const floored = Math.floor(candidateMaxTurns);
    this.maxTurns =
      Number.isFinite(floored) && floored >= 1 ? floored : DEFAULT_MAX_TURNS;
    this.sessionStore =
      options.sessionStore === null
        ? null
        : (options.sessionStore ?? new SessionStore());
    this.visionClientExplicit = "visionClient" in options;
    this._visionClient = options.visionClient ?? null;
    this.sessionDrainTimeoutMs = options.sessionDrainTimeoutMs ?? 30_000;
    this.mcpConnector = options.mcpConnector ?? options.connectSessionMcpServers ?? connectSessionMcpServers;
    this.connectMcpServers = this.mcpConnector;
    this.shutdownDrainTimeoutMs = options.shutdownDrainTimeoutMs ?? 4_000;
    this.streamThinking = process.env["ACP_GLM_STREAM_THINKING"]?.toLowerCase() !== "false";
  }

  private get glm(): NonNullable<GlmAcpAgentOptions["glm"]> {
    if (this._glm === null) {
      this._glm = new GlmClient();
    }
    return this._glm;
  }

  private get visionClient(): VisionMcpClient | null {
    if (this.visionClientExplicit) return this._visionClient;
    if (!this._visionClient) {
      const apiKey = resolveApiKey();
      if (!apiKey) return null;
      this._visionClient = new StdioVisionMcpClient({ apiKey });
    }
    return this._visionClient;
  }

  // ---------------------------------------------------------------------------
  // ACP Lifecycle
  // ---------------------------------------------------------------------------

  async initialize(params: InitializeRequest): Promise<InitializeResponse> {
    // Negotiate the lowest version both sides support.
    const negotiatedVersion =
      params.protocolVersion <= VERSION ? params.protocolVersion : VERSION;

    // Keep client capabilities for compatibility with ACP initialize payloads.
    // Built-in local tools run in this agent process and are not gated on
    // client fs/terminal support.
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
        mcpCapabilities: {
          http: true,
        },
        promptCapabilities: {
          // Baseline (text + resource_link) is implicit; we additionally accept
          // embedded resources for inline file context, plus images for
          // glm-5v-turbo native vision and the Vision MCP fallback path.
          // Set ACP_GLM_PROMPT_IMAGES=false (or =0) to stop advertising image
          // support so clients like Zed won't offer the image-attachment UI.
          embeddedContext: true,
          image: process.env["ACP_GLM_PROMPT_IMAGES"] !== "false" &&
                 process.env["ACP_GLM_PROMPT_IMAGES"] !== "0",
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
    params: AuthenticateRequest
  ): Promise<AuthenticateResponse> {
    // Authentication is configured externally — either via Z_AI_API_KEY in the
    // environment or via the credentials file written by `glm-acp-agent --setup`.
    // The agent has nothing to do here; failures will surface when the model
    // is first called.
    void params;
    return {};
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    if (this.shuttingDown) throw new Error("Agent is shutting down");
    const sessionId = randomUUID();
    debug(`newSession: id=${sessionId} cwd=${params.cwd} model=${getDefaultModel()}`);
    const setup = this.connectMcpForSession(params.mcpServers);
    try {
      const mcpTools = await setup.tools;
      const toolDefinitions = this.availableToolDefinitions(mcpTools);

    const systemPrompt: GlmMessage = {
      role: "system",
      content: buildSystemPrompt({
        cwd: params.cwd,
        tools: toolDefinitions.map((tool) => tool.function.name),
        agentsMd: loadProjectContext(params.cwd),
      }),
    };

    const model = getDefaultModel();
    // Default to the model's own default effort ("max" on the 5.3 family,
    // "on" otherwise) so out-of-the-box behaviour matches the pre-thought-level
    // default (thinking on, no explicit reasoning_effort).
    const thoughtLevel = resolveThoughtLevel(model, "max");

    const lifecycle = new SessionLifecycle();
    this.transitions.set(sessionId, {
      lifecycle,
      promise: null,
      closePromise: null,
      original: null,
      restoreAbortController: null,
    });
    this.sessions.set(sessionId, {
      cwd: params.cwd,
      messages: [systemPrompt],
      abortController: null,
      promptPromise: null,
      closing: false,
      closed: false,
      closePromise: null,
      disposePromise: null,
      title: null,
      updatedAt: new Date().toISOString(),
      model,
      toolDefinitions,
      mcpTools,
      mode: "default",
      thoughtLevel,
      commands: discoverSlashCommands(params.cwd),
      displayText: new WeakMap(),
      lifecycle,
      });

      this.scheduleAvailableCommands(sessionId);

      return {
        sessionId,
        models: this.modelsState(model),
        modes: this.modesState("default"),
        configOptions: this.configOptionsState(model, thoughtLevel, "default"),
      };
    } catch (error) {
      await setup.dispose();
      throw error;
    } finally {
      setup.release();
    }
  }

  /**
   * Queue an `available_commands_update` snapshot for a session.
   *
   * The notification is the only channel ACP gives us for slash-command
   * autocomplete — it is not part of any method's response — so every session
   * entry point (create / load / fork / resume) has to send one.
   *
   * It is deliberately *deferred* rather than awaited inline: a client learns a
   * session's id from the `session/new` / `session/fork` response, so a
   * notification written ahead of that response arrives for a session the client
   * has never heard of, and clients drop those. Sending on the next macrotask
   * puts it behind the response the caller is about to return (and, on load,
   * behind the replayed transcript), which is also when a client is ready to
   * paint the menu. Each send replaces the previous list wholesale.
   */
  private scheduleAvailableCommands(sessionId: string): void {
    setTimeout(() => {
      const session = this.sessions.get(sessionId);
      if (!session) return;
      void safeSessionUpdate(this.connection, {
        sessionId,
        update: {
          sessionUpdate: "available_commands_update",
          availableCommands: availableCommandsState(session.commands),
        },
      });
    }, 0);
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest
  ): Promise<SetSessionModelResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }
    this.assertConfigurable(params.sessionId, session);
    await this.applySessionModel(params.sessionId, session, params.modelId);
    return {};
  }

  /**
   * Switch a session to a new model id — shared by `session/set_model` and the
   * `model` config option. Uncatalogued ids are allowed on purpose (Z.AI may
   * offer models we haven't catalogued), but log a stderr hint. Persists,
   * notifies clients via `session_info_update`, and pushes a
   * `config_option_update` because the valid thought levels may differ between
   * models (e.g. switching from 5.3 to 4.7 drops the effort ladder).
   */
  private async applySessionModel(
    sessionId: string,
    session: SessionState,
    modelId: string
  ): Promise<void> {
    const available = getAvailableModels();
    const known = available.find((m) => m.modelId === modelId);
    if (!known) {
      process.stderr.write(
        `[glm-acp-agent] warning: model "${modelId}" is not in the advertised list; using as-is.\n`
      );
    }
    session.model = modelId;
    // The valid thought levels differ per model, so clamp the current level to
    // what the new model supports.
    session.thoughtLevel = resolveThoughtLevel(modelId, session.thoughtLevel);
    session.updatedAt = new Date().toISOString();
    // Persist immediately so a fork/reload before the next prompt doesn't
    // resurrect the previous model or thought level.
    this.persistSession(sessionId, session);
    // Notify clients so any UI that displays the active model refreshes
    // immediately, instead of waiting for the next prompt to complete.
    await safeSessionUpdate(this.connection, {
      sessionId,
      update: {
        sessionUpdate: "session_info_update",
        updatedAt: session.updatedAt,
      },
    });
    await safeSessionUpdate(this.connection, {
      sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions: this.configOptionsState(
          session.model,
          session.thoughtLevel,
          session.mode
        ),
      },
    });
  }

  /**
   * Build the SessionConfigOptions we advertise: `thought_level` (levels depend
   * on the model, see {@link getThoughtLevels}), `mode`, and `model`.
   *
   * The `mode` option mirrors {@link modesState} as a `category: "mode"`
   * selector. Clients that render config options (Zed) suppress the legacy mode
   * selector once any config option is advertised, so the permission mode would
   * otherwise be unreachable from their UI. `currentMode` is always read from
   * the live session state, so a mode set through `session/set_mode` shows up
   * here too.
   *
   * The `model` option mirrors {@link modelsState} as a `category: "model"`
   * selector for the same reason: clients that render config options suppress
   * the legacy model selector too, so the active model would otherwise be
   * unreachable. `currentModel` is read from the live session state, so a model
   * set through `session/set_model` shows up here as well.
   */
  private configOptionsState(
    model: string,
    thoughtLevel: ThoughtLevel,
    currentMode: SessionModeId
  ): SessionConfigOption[] {
    const levels = getThoughtLevels(model);
    const modelOptions = availableModelsWith(model);
    return [
      {
        id: "thought_level",
        name: "Thinking",
        description: "Reasoning effort",
        category: "thought_level",
        type: "select" as const,
        currentValue: thoughtLevel,
        options: levels.map((level) => ({
          value: level,
          name: THOUGHT_LEVEL_NAMES[level],
        })),
      },
      {
        id: "mode",
        name: "Mode",
        description: "Tool permission mode",
        category: "mode",
        type: "select" as const,
        currentValue: currentMode,
        options: SESSION_MODES.map((mode) => ({
          value: mode.id,
          name: mode.name,
        })),
      },
      {
        id: "model",
        name: "Model",
        description: "GLM model for this session",
        category: "model",
        type: "select" as const,
        currentValue: model,
        options: modelOptions.map((m) => ({
          value: m.modelId,
          name: m.name,
        })),
      },
    ];
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest
  ): Promise<SetSessionConfigOptionResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }
    this.assertConfigurable(params.sessionId, session);
    if (params.configId === "mode") {
      // Same reject-don't-coerce policy as thought_level below.
      if (!isSessionModeId(params.value)) {
        throw new Error(`Invalid mode value: ${String(params.value)}`);
      }
      session.mode = params.value;
      session.updatedAt = new Date().toISOString();
      this.persistSession(params.sessionId, session);
      // Mirror setSessionMode so clients tracking the ACP mode state (rather
      // than the config option) stay in sync with the dropdown.
      await safeSessionUpdate(this.connection, {
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "current_mode_update",
          currentModeId: session.mode,
        },
      });
      return {
        configOptions: this.configOptionsState(
          session.model,
          session.thoughtLevel,
          session.mode
        ),
      };
    }
    if (params.configId === "model") {
      // Same reject-don't-coerce policy as thought_level below, but only for
      // values that aren't model ids at all. Uncatalogued ids are allowed on
      // purpose, mirroring unstable_setSessionModel / ACP_GLM_MODEL.
      if (typeof params.value !== "string" || params.value.length === 0) {
        throw new Error(`Invalid model value: ${String(params.value)}`);
      }
      await this.applySessionModel(params.sessionId, session, params.value);
      return {
        configOptions: this.configOptionsState(
          session.model,
          session.thoughtLevel,
          session.mode
        ),
      };
    }
    if (params.configId !== "thought_level") {
      throw new Error(`Unknown config option: ${params.configId}`);
    }
    // Reject values that aren't a thought level at all (typos, stale ids from
    // another agent version) rather than silently coercing them.
    if (typeof params.value !== "string" || !isThoughtLevel(params.value)) {
      throw new Error(`Invalid thought_level value: ${String(params.value)}`);
    }
    // Auto-resolve a *known* level that isn't valid for the current model to
    // that model's default. This happens when a client caches a thoughtLevel
    // from a previous session (or model) and re-sends it for a model with a
    // different level set (e.g. "max" from glm-5.3 sent while on glm-4.7).
    session.thoughtLevel = resolveThoughtLevel(session.model, params.value);
    session.updatedAt = new Date().toISOString();
    this.persistSession(params.sessionId, session);
    return {
      configOptions: this.configOptionsState(
        session.model,
        session.thoughtLevel,
        session.mode
      ),
    };
  }

  /**
   * Build the SessionModelState we advertise on session create/load/resume/fork.
   *
   * The active model is always included in `availableModels`, even when it
   * isn't in the advertised list — a session restored from disk can be pinned
   * to a de-listed id (`glm-5.2` was the previous default), and `ACP_GLM_MODEL`
   * / `session/set_model` both accept uncatalogued ids on purpose. Returning a
   * `currentModelId` outside the advertised set leaves pickers unable to
   * represent the selection the agent is actually using.
   */
  private modelsState(currentModelId: string): {
    availableModels: ModelInfo[];
    currentModelId: string;
  } {
    return { availableModels: availableModelsWith(currentModelId), currentModelId };
  }

  /** Build the SessionModeState we advertise on session create/load/resume/fork. */
  private modesState(currentModeId: SessionModeId): {
    availableModes: Array<{ id: SessionModeId; name: string; description: string }>;
    currentModeId: SessionModeId;
  } {
    return {
      availableModes: SESSION_MODES.map((mode) => ({ ...mode })),
      currentModeId,
    };
  }

  async setSessionMode(
    params: SetSessionModeRequest
  ): Promise<SetSessionModeResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }
    this.assertConfigurable(params.sessionId, session);
    if (!isSessionModeId(params.modeId)) {
      throw new Error(
        `Invalid modeId: ${params.modeId}. Valid modes are: ${SESSION_MODE_IDS.join(", ")}`
      );
    }
    const newMode = params.modeId;
    session.mode = newMode;
    session.updatedAt = new Date().toISOString();
    this.persistSession(params.sessionId, session);
    await safeSessionUpdate(this.connection, {
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "current_mode_update",
        currentModeId: newMode,
      },
    });
    // Config options are a separate state channel: re-publish them so a mode
    // dropdown (category "mode") doesn't keep a stale currentValue when the
    // mode changes through the classic ACP path.
    await safeSessionUpdate(this.connection, {
      sessionId: params.sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions: this.configOptionsState(
          session.model,
          session.thoughtLevel,
          session.mode
        ),
      },
    });
    return {};
  }

  // ---------------------------------------------------------------------------
  // Prompt Turn
  // ---------------------------------------------------------------------------

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions.get(params.sessionId);
    if (!session || session.closing || session.closed) {
      throw new Error(`Session not found: ${params.sessionId}`);
    }
    if (!session.lifecycle.acceptsPrompts()) {
      throw new Error(`Session transition in progress: ${params.sessionId}`);
    }
    const generation = session.lifecycle.generation;
    const ownsPrompt = () => this.ownsSession(params.sessionId, session, generation);
    const preservesDrainingHistory = () => this.isDrainingOriginal(params.sessionId, session);

    // Reserve this lifecycle synchronously, before preprocessing can await.
    // Every queued prompt gets a distinct promise and chains behind the prompt
    // that was current when it arrived; otherwise several callers can all
    // resume from one old promise and enter the model concurrently.
    const predecessor = session.promptPromise;
    session.abortController?.abort();
    const abortController = new AbortController();
    session.abortController = abortController;
    let resolvePromptPromise!: () => void;
    const promptPromise = new Promise<void>((resolve) => {
      resolvePromptPromise = resolve;
    });
    session.promptPromise = promptPromise;

    // Abort the prompt automatically if the underlying connection closes.
    const onConnectionClose = () => abortController.abort();
    const connSignal = this.connection.signal;
    if (connSignal && !connSignal.aborted) {
      connSignal.addEventListener("abort", onConnectionClose, { once: true });
    } else if (connSignal?.aborted) {
      abortController.abort();
    }

    // Echo back the client-supplied messageId on every response (success,
    // cancelled, or error) so the client can correlate the turn.
    const userMessageId = params.messageId ?? undefined;

    let preprocessed: { blocks: PromptRequest["prompt"]; cleanups: Array<() => Promise<void>> } | undefined;
    const cancelledResponse = (): PromptResponse => {
      const cancelled: PromptResponse = { stopReason: "cancelled" };
      if (userMessageId) cancelled.userMessageId = userMessageId;
      return cancelled;
    };

    try {
      if (predecessor) {
        try {
          await predecessor;
        } catch {
          // A previous loop's failure is already reported to its caller.
        }
      }
      if (abortController.signal.aborted || !ownsPrompt()) {
        return cancelledResponse();
      }

      // Convert ACP content blocks into a GLM user message. Most models receive
      // plain text, with images preprocessed through Vision MCP. Native vision
      // models receive OpenAI-style multimodal content parts directly.
      if (isDebugEnabled()) {
        for (const line of buildPromptBlockDiagnosticLines(params.prompt)) {
          debug(line);
        }
      }
      // Clients invoke an advertised command by sending `/name …` as ordinary
      // prompt text, so expand it here into the instructions its definition
      // holds. Unknown `/foo` is left alone and reaches the model as prose.
      const promptBlocks = expandPromptCommand(params.prompt, session.commands);
      const visionNative = isVisionNativeModel(session.model);
      preprocessed = visionNative
        ? { blocks: promptBlocks, cleanups: [] }
        : await preprocessImageBlocks(
            promptBlocks,
            this.visionClient,
            abortController.signal
          );
      if (abortController.signal.aborted || !ownsPrompt()) {
        return cancelledResponse();
      }
      const userContent = visionNative
        ? renderVisionNativePromptBlocks(preprocessed.blocks)
        : renderPromptBlocks(preprocessed.blocks).content;
      const userMessage: GlmMessage = { role: "user", content: userContent };
      session.messages.push(userMessage);

      // What the user typed, rendered from the blocks as they arrived — before
      // command expansion and image analysis rewrote them for the model. Kept
      // separately so `session/load` replays the conversation the user had.
      const displayText = renderPromptBlocks(params.prompt).plainText;
      if (displayText !== stringifyUserMessage(userContent)) {
        session.displayText.set(userMessage, displayText);
      }

      const { stopReason, usage } = await this.runPromptLoop(
        params.sessionId,
      session,
      abortController.signal,
      ownsPrompt,
      preservesDrainingHistory,
      this.processSupervisor
      );

      if (!ownsPrompt()) {
        return cancelledResponse();
      }
      if (session.abortController === abortController) session.abortController = null;
      session.updatedAt = new Date().toISOString();

      // Emit a session_info_update with the (possibly first-set) title and
      // updated timestamp so clients can show fresh metadata.
      const titleUpdate: { title?: string | null } =
        session.title === null
          ? (() => {
              const derived = displayText
                .slice(0, 80)
                .replace(/\s+/g, " ")
                .trim();
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
      if (abortController.signal.aborted || !ownsPrompt()) {
        return cancelledResponse();
      }
      // Surface the error to the user as an agent message so the IDE displays
      // something instead of a silent JSON-RPC error.
      const message = err instanceof Error ? err.message : String(err);
      if (ownsPrompt()) {
        await safeSessionUpdate(this.connection, {
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: `\n\n[error] ${message}` },
          },
        });
      }
      throw err;
    } finally {
      connSignal?.removeEventListener("abort", onConnectionClose);
      for (const cleanup of preprocessed?.cleanups ?? []) {
        try { await cleanup(); } catch { /* best effort */ }
      }
      // A newer queued prompt may own these fields already. Only the owner
      // may clear them, and resolution follows all preprocessing/cleanup.
      if (session.abortController === abortController) session.abortController = null;
      if (session.promptPromise === promptPromise) session.promptPromise = null;
      resolvePromptPromise();
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions.get(params.sessionId);
    session?.abortController?.abort();
  }

  async closeSession(params: CloseSessionRequest): Promise<void> {
    const transition = this.transitions.get(params.sessionId);
    const initial = this.sessions.get(params.sessionId);
    if (!transition && !initial) return;
    const record = transition ?? this.registerTransition(params.sessionId, initial!.lifecycle);
    if (record.closePromise) return record.closePromise;

    if (record.lifecycle.phase === "open") {
      record.lifecycle.begin("closing");
    } else {
      // This also records a close for an unloaded session: the record exists
      // even while restore has not yet installed its replacement state.
      record.lifecycle.requestClose();
    }
    record.restoreAbortController?.abort();
    initial?.abortController?.abort();
    const closePromise = (async () => {
      const pendingTransition = record.promise;
      if (pendingTransition) {
        try { await pendingTransition; } catch { /* close owns final cleanup */ }
      }

      // Restore may have swapped while this close was waiting. Resolve the
      // current map entry now; never dispose or delete a stale captured state.
      const current = this.sessions.get(params.sessionId);
      if (current) {
        current.closing = true;
        current.abortController?.abort();
        if (current.promptPromise) {
          try { await current.promptPromise; } catch { /* resolves in prompt finally */ }
        }
        current.closed = true;
        this.persistSession(params.sessionId, current);
        await this.disposeSessionTools(current);
        if (this.sessions.get(params.sessionId) === current) {
          this.sessions.delete(params.sessionId);
        }
        this.sessionTodos.delete(params.sessionId);
      }
      if (this.transitions.get(params.sessionId) === record) {
        this.transitions.delete(params.sessionId);
      }
    })();
    record.closePromise = closePromise;
    return closePromise;
  }

  /**
   * Stop admitting work and release all resources owned by this ACP runtime.
   * Concurrent close/signal paths share one promise so disposals and process
   * termination happen once.
   */
  shutdown(_reason: "disconnect" | "sigterm" | "sigint" | "fatal"): Promise<void> {
    void _reason;
    if (this.shutdownPromise) return this.shutdownPromise;
    this.shuttingDown = true;
    this.shutdownPromise = (async () => {
      const sessions = [...this.sessions.entries()];

      // Close the lifecycle gates synchronously before aborting prompts. This
      // makes the in-flight loop retain canonical partial history while its
      // owned connection suppresses all late UI chunks.
      for (const [sessionId, record] of this.transitions) {
        if (record.lifecycle.phase === "open") record.lifecycle.begin("closing");
        else record.lifecycle.requestClose();
        record.restoreAbortController?.abort();
        void sessionId;
      }
      for (const [, session] of sessions) {
        const record = [...this.transitions.values()].find((candidate) => candidate.lifecycle === session.lifecycle);
        if (record && record.lifecycle.phase === "open") record.lifecycle.begin("closing");
        else record?.lifecycle.requestClose();
        record?.restoreAbortController?.abort();
        session.closing = true;
        session.abortController?.abort();
      }

      await this.processSupervisor.terminateAll();
      const deadlineMs = this.shutdownDrainTimeoutMs;
      const promptsSettled = await Promise.all(sessions.map(([, session]) =>
        session.promptPromise
          ? settlesWithin(session.promptPromise, deadlineMs)
          : Promise.resolve(true),
      ));

      const transitions = [...this.transitions.values()]
        .map((record) => record.promise)
        .filter((promise): promise is Promise<unknown> => promise !== null);
      const closes = sessions
        .map(([, session]) => session.closePromise)
        .filter((close): close is Promise<void> => close !== null);
      const setups = [...this.pendingSetups].map((setup) => setup.dispose());
      const auxDrained = await settlesWithin(
        Promise.allSettled([...transitions, ...closes, ...setups]),
        deadlineMs,
      );

      for (let index = 0; index < sessions.length; index += 1) {
        const [sessionId, session] = sessions[index]!;
        session.closed = true;
        // A prompt that missed the drain deadline may still hold an unmatched
        // assistant tool call. Leave the last valid on-disk checkpoint intact.
        if (promptsSettled[index] === true) this.persistSession(sessionId, session);
        await this.disposeSessionTools(session).catch(() => undefined);
        if (this.sessions.get(sessionId) === session) this.sessions.delete(sessionId);
        this.sessionTodos.delete(sessionId);
      }
      this.transitions.clear();
      await this._visionClient?.dispose();

      if (this.processSupervisor.hasActiveProcesses()) {
        await this.processSupervisor.forceTerminateAll();
      }
      if (this.processSupervisor.hasActiveProcesses()) {
        throw new Error("Agent shutdown left active command processes");
      }
      if (!auxDrained || promptsSettled.some((settled) => !settled)) {
        throw new Error("Agent shutdown timed out waiting for prompt cleanup");
      }
    })();
    return this.shutdownPromise;
  }

  /** Used by the CLI deadline path before reporting a nonzero exit. */
  async forceShutdown(): Promise<void> {
    await this.processSupervisor.forceTerminateAll();
    if (this.processSupervisor.hasActiveProcesses()) {
      throw new Error("Agent shutdown left active command processes");
    }
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
    return this.restoreSession(params, true);
  }

  async unstable_forkSession(
    params: ForkSessionRequest
  ): Promise<ForkSessionResponse> {
    if (this.shuttingDown) throw new Error("Agent is shutting down");
    const source = this.sessions.get(params.sessionId);
    const persisted = source
      ? this.snapshot(params.sessionId, source)
      : this.requirePersisted(params.sessionId);
    const setup = this.connectMcpForSession(params.mcpServers ?? []);
    let installed: SessionState | null = null;
    let newSessionId: string | null = null;
    try {
      const mcpTools = await setup.tools;
      const toolDefinitions = this.availableToolDefinitions(mcpTools);

    newSessionId = randomUUID();
    const forkedTitle =
      persisted.title === null ? null : `${persisted.title} (fork)`;
    const forkedMessages = rebuildRestoredMessages(
      structuredClone(persisted.messages),
      params.cwd,
      toolDefinitions
    );
    const forkLifecycle = new SessionLifecycle();
    const forked: SessionState = {
      cwd: params.cwd,
      // Deep-clone messages so the fork doesn't share state with the parent.
      messages: forkedMessages,
      abortController: null,
      promptPromise: null,
      closing: false,
      closed: false,
      closePromise: null,
      disposePromise: null,
      title: forkedTitle,
      updatedAt: new Date().toISOString(),
      model: persisted.model,
      toolDefinitions,
      mcpTools,
      mode: persisted.mode,
      thoughtLevel: resolveThoughtLevel(persisted.model, persisted.thoughtLevel ?? "max"),
      commands: discoverSlashCommands(params.cwd),
      // Re-key onto the cloned messages: the parent's map is keyed by the
      // originals, which the fork no longer holds.
      displayText: deserializeRestoredDisplayText(
        persisted.messages,
        forkedMessages,
        persisted.displayText
      ),
      lifecycle: forkLifecycle,
    };
    this.sessions.set(newSessionId, forked);
    installed = forked;
    this.transitions.set(newSessionId, {
      lifecycle: forkLifecycle,
      promise: null,
      closePromise: null,
      original: null,
      restoreAbortController: null,
    });
    this.persistSession(newSessionId, forked);

    // Notify on the *created* session id — the parent thread's command list is
    // unchanged and a notify there would repaint the wrong menu.
    this.scheduleAvailableCommands(newSessionId);

    return {
      sessionId: newSessionId,
      models: this.modelsState(forked.model),
      modes: this.modesState(forked.mode),
      configOptions: this.configOptionsState(
        forked.model,
        forked.thoughtLevel,
        forked.mode
      ),
    };
    } catch (error) {
      if (installed) {
        if (this.sessions.get(newSessionId!) === installed) this.sessions.delete(newSessionId!);
        await this.disposeSessionTools(installed);
      } else {
        await setup.dispose();
      }
      throw error;
    } finally {
      setup.release();
    }
  }

  async resumeSession(
    params: ResumeSessionRequest
  ): Promise<ResumeSessionResponse> {
    return this.restoreSession(params, false);
  }

  private async restoreSession(
    params: { sessionId: string; cwd: string; mcpServers?: LoadSessionRequest["mcpServers"] },
    replay: boolean
  ): Promise<LoadSessionResponse> {
    if (this.shuttingDown) throw new Error("Agent is shutting down");
    const original = this.sessions.get(params.sessionId);
    const record = this.transitions.get(params.sessionId)
      ?? this.registerTransition(params.sessionId, original?.lifecycle ?? new SessionLifecycle());
    const lifecycle = record.lifecycle;
    const lease = lifecycle.begin("restoring");
    record.original = original ?? null;
    const restoreAbortController = new AbortController();
    record.restoreAbortController = restoreAbortController;
    let deferLeaseRelease = false;

    const transition = Promise.resolve().then(async (): Promise<LoadSessionResponse> => {
      let provisional: SessionMcpTools | null = null;
      let provisionalDisposal: Promise<void> | null = null;
      let swapped = false;
      const disposeProvisional = (tools: SessionMcpTools): Promise<void> => {
        if (!provisionalDisposal) provisionalDisposal = tools.dispose();
        return provisionalDisposal;
      };
      try {
        let persisted: PersistedSession;
        if (original) {
          original.abortController?.abort();
          const { drained, pending } = await this.drainPrompt(original);
          if (!drained) {
            deferLeaseRelease = true;
            this.releaseAfterPromptDrain(pending, original, lifecycle, lease, record);
            throw new Error(`Session restore timed out waiting for prompt cleanup: ${lease.generation}`);
          }
          this.assertRestoreOwner(lifecycle, lease);
          // This must be taken only after the reserved prompt chain settles:
          // disk can be older than a just-finished live turn.
          persisted = this.snapshot(params.sessionId, original);
        } else {
          persisted = this.requirePersisted(params.sessionId);
        }

        this.assertRestoreOwner(lifecycle, lease);
        const setup = this.connectMcpServers(params.mcpServers ?? [], restoreAbortController.signal);
        void setup.then(
          (tools) => {
            if (restoreAbortController.signal.aborted) {
              void disposeProvisional(tools).catch(() => undefined);
            }
          },
          () => undefined
        );
        provisional = await waitForAbort(
          setup,
          restoreAbortController.signal,
          `Session restore cancelled: ${params.sessionId}`
        );
        if (this.shuttingDown) throw new Error("Agent is shutting down");
        if (!lifecycle.owns(lease) || lifecycle.closeRequested) {
          throw new Error(`Session restore cancelled: ${params.sessionId}`);
        }

        const toolDefinitions = this.availableToolDefinitions(provisional);
        // Configuration updates remain responsive while MCP setup is in
        // flight. Prompts are gated, so this second live projection keeps
        // those latest settings without reopening the history race.
        const restoreSource = original ? this.snapshot(params.sessionId, original) : persisted;
        const restoredMessages = rebuildRestoredMessages(
          restoreSource.messages,
          params.cwd,
          toolDefinitions
        );
        const restored: SessionState = {
          cwd: params.cwd,
          messages: restoredMessages,
          abortController: null,
          promptPromise: null,
          closing: false,
          closed: false,
          closePromise: null,
          disposePromise: null,
          title: restoreSource.title,
          updatedAt: restoreSource.updatedAt,
          model: restoreSource.model,
          toolDefinitions,
          mcpTools: provisional,
          mode: restoreSource.mode,
          thoughtLevel: resolveThoughtLevel(restoreSource.model, restoreSource.thoughtLevel ?? "max"),
          commands: discoverSlashCommands(params.cwd),
          displayText: deserializeRestoredDisplayText(
            restoreSource.messages,
            restoredMessages,
            restoreSource.displayText
          ),
          lifecycle,
        };

        // Replaying a load is part of preparing the replacement. If the
        // client rejects a replay notification, leave the live original in
        // place and dispose the provisional resources below.
        if (replay) {
          await this.replayMessages(params.sessionId, restoredMessages, restored.displayText);
        }
        if (this.shuttingDown) throw new Error("Agent is shutting down");
        if (lifecycle.closeRequested) {
          throw new Error(`Session restore cancelled: ${params.sessionId}`);
        }
        if (original && this.sessions.get(params.sessionId) === original) {
          // Configuration setters stay responsive while a load replays, and
          // they mutate the still-installed original. Carry those latest
          // values into the replacement instead of installing the values
          // captured before replay started.
          restored.model = original.model;
          restored.mode = original.mode;
          restored.thoughtLevel = original.thoughtLevel;
        }
        this.sessions.set(params.sessionId, restored);
        this.sessionTodos.delete(params.sessionId);
        swapped = true;
        // Checkpoint the merged state immediately: it can retain a partially
        // received turn from the drained prompt, which otherwise exists only
        // in memory until the next prompt or close.
        this.persistSession(params.sessionId, restored);
        // Ownership transfers only after the replacement is installed. The
        // old resources are released afterwards, so setup and replay failures
        // still retain a valid original session.
        if (original) await this.disposeSessionTools(original);
        if (this.shuttingDown) throw new Error("Agent is shutting down");
        if (lifecycle.closeRequested) {
          throw new Error(`Session restore cancelled: ${params.sessionId}`);
        }
        this.scheduleAvailableCommands(params.sessionId);
        return {
          models: this.modelsState(restored.model),
          modes: this.modesState(restored.mode),
          configOptions: this.configOptionsState(restored.model, restored.thoughtLevel, restored.mode),
        };
      } catch (err) {
        if (!swapped) {
          if (provisional) await disposeProvisional(provisional);
        }
        throw err;
      } finally {
        if (!deferLeaseRelease) lease.release();
      }
    });
    record.promise = transition;
    try {
      return await transition;
    } finally {
      if (record.promise === transition) record.promise = null;
      if (!deferLeaseRelease && record.original === original) record.original = null;
      if (record.restoreAbortController === restoreAbortController) {
        record.restoreAbortController = null;
      }
    }
  }

  private async drainPrompt(
    session: SessionState
  ): Promise<{ drained: boolean; pending: Promise<void> | null }> {
    const pending = session.promptPromise;
    if (!pending) return { drained: true, pending: null };
    let timer: NodeJS.Timeout | undefined;
    const timeout = new Promise<"timeout">((resolve) => {
      if (this.sessionDrainTimeoutMs <= 0) queueMicrotask(() => resolve("timeout"));
      else timer = setTimeout(() => resolve("timeout"), this.sessionDrainTimeoutMs);
    });
    const outcome = await Promise.race([
      pending.then(() => "drained" as const),
      timeout,
    ]);
    if (timer) clearTimeout(timer);
    return { drained: outcome === "drained", pending };
  }

  private releaseAfterPromptDrain(
    pending: Promise<void> | null,
    session: SessionState,
    lifecycle: SessionLifecycle,
    lease: TransitionLease,
    record: SessionTransition
  ): void {
    // The promise captured when the drain started must be the one observed
    // here: the prompt's cleanup can null `session.promptPromise` right after
    // the timeout resolves, and re-reading the field would never attach the
    // release callback, leaving the lease stuck in `restoring` forever.
    void pending?.then(() => {
      if (!lifecycle.closeRequested) lease.release();
      if (record.original === session) record.original = null;
    });
  }

  private registerTransition(sessionId: string, lifecycle: SessionLifecycle): SessionTransition {
    const record: SessionTransition = {
      lifecycle,
      promise: null,
      closePromise: null,
      original: null,
      restoreAbortController: null,
    };
    this.transitions.set(sessionId, record);
    return record;
  }

  private assertRestoreOwner(lifecycle: SessionLifecycle, lease: TransitionLease): void {
    if (!lifecycle.owns(lease) || lifecycle.closeRequested) {
      throw new Error("Session restore cancelled");
    }
  }

  // ---------------------------------------------------------------------------
  // Persistence helpers
  // ---------------------------------------------------------------------------

  private ownsSession(sessionId: string, session: SessionState, generation: number): boolean {
    return this.sessions.get(sessionId) === session
      && !session.closing
      && !session.closed
      && session.lifecycle.phase === "open"
      && session.lifecycle.generation === generation;
  }

  private isDrainingOriginal(sessionId: string, session: SessionState): boolean {
    const record = this.transitions.get(sessionId);
    if (record?.original === session && record.lifecycle.phase === "restoring") return true;
    // A normal close aborts the prompt and waits for it before persisting. The
    // closing generation must still finish its canonical history (without UI
    // notifications), otherwise already-received text or an in-flight tool
    // result disappears from the final checkpoint.
    return this.sessions.get(sessionId) === session
      && record?.lifecycle === session.lifecycle
      && record.lifecycle.phase === "closing"
      && !session.closed;
  }

  private ownedPromptConnection(ownsPrompt: () => boolean): AgentSideConnection {
    const connection = this.connection;
    return new Proxy(connection, {
      get(target, property, receiver) {
        if (property === "sessionUpdate") {
          return async (params: Parameters<AgentSideConnection["sessionUpdate"]>[0]) => {
            if (ownsPrompt()) await connection.sessionUpdate(params);
          };
        }
        const value = Reflect.get(target, property, receiver);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
  }

  private assertConfigurable(sessionId: string, session: SessionState): void {
    if (session.closing || session.closed || session.lifecycle.closeRequested) {
      throw new Error(`Session is closing: ${sessionId}`);
    }
  }

  /** Track MCP setup until it installs into a session or shutdown disposes it. */
  private connectMcpForSession(
    servers: Parameters<typeof connectSessionMcpServers>[0]
  ): PendingMcpSetup {
    let disposePromise: Promise<void> | null = null;
    const source = Promise.resolve().then(() => this.mcpConnector(servers));
    const setup: PendingMcpSetup = {
      tools: source.then(async (mcpTools) => {
        if (this.shuttingDown) {
          await setup.dispose();
          throw new Error("Agent is shutting down");
        }
        return mcpTools;
      }),
      release: () => this.pendingSetups.delete(setup),
      dispose: () => {
        if (!disposePromise) {
          disposePromise = source.then(
            (mcpTools) => mcpTools.dispose(),
            () => undefined,
          ).finally(() => setup.release());
        }
        return disposePromise;
      },
    };
    this.pendingSetups.add(setup);
    void source.catch(() => setup.release());
    return setup;
  }

  private disposeSessionTools(session: SessionState): Promise<void> {
    if (!session.disposePromise) {
      session.disposePromise = session.mcpTools?.dispose() ?? Promise.resolve();
    }
    return session.disposePromise;
  }

  private snapshot(sessionId: string, session: SessionState): PersistedSession {
    const displayText = serializeDisplayText(session.messages, session.displayText);
    return {
      sessionId,
      cwd: session.cwd,
      messages: session.messages,
      title: session.title,
      updatedAt: session.updatedAt,
      model: session.model,
      mode: session.mode,
      thoughtLevel: session.thoughtLevel,
      // Absent for the common case where nothing diverges, so an ordinary
      // session gains no on-disk weight.
      ...(displayText ? { displayText } : {}),
    };
  }

  private persistSession(sessionId: string, session: SessionState): void {
    if (this.sessions.get(sessionId) !== session) return;
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
    messages: GlmMessage[],
    displayText: SessionState["displayText"]
  ): Promise<void> {
    for (const msg of messages) {
      if (msg.role === "user") {
        const text = displayText.get(msg) ?? stringifyUserMessage(msg.content);
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
    signal: AbortSignal,
    ownsPrompt: () => boolean,
    preservesDrainingHistory: () => boolean,
    processSupervisor: ProcessSupervisor
  ): Promise<{ stopReason: InternalStopReason; usage?: Usage }> {
    const executor = new ToolExecutor(
      this.ownedPromptConnection(ownsPrompt),
      sessionId,
      this.clientCapabilities,
      signal,
      this.visionClient,
      session.mcpTools,
      session.cwd,
      // Use a thunk so mode changes mid-turn take effect on the next tool call.
      () => this.sessions.get(sessionId)?.mode ?? "default",
      (todos) => {
        if (ownsPrompt()) this.sessionTodos.set(sessionId, todos);
      },
      processSupervisor
    );

    let totalUsage: Usage | undefined;
    let overflowRetryCount = 0;

    const addUsage = (usage: Usage | undefined): void => {
      if (!usage) return;
      if (!totalUsage) {
        totalUsage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
      }
      totalUsage.inputTokens += usage.inputTokens;
      totalUsage.outputTokens += usage.outputTokens;
      totalUsage.totalTokens += usage.totalTokens;
      for (const key of ["cachedReadTokens", "cachedWriteTokens", "thoughtTokens"] as const) {
        const value = usage[key];
        if (typeof value !== "number") continue;
        totalUsage[key] = (totalUsage[key] ?? 0) + value;
      }
    };

    const cancelledToolResult = (toolCallId: string): void => {
      session.messages.push({
        role: "tool",
        tool_call_id: toolCallId,
        content: "Tool call cancelled before execution.",
      });
    };

    for (let turn = 0; turn < this.maxTurns; turn++) {
      if (signal.aborted || !ownsPrompt()) return { stopReason: "cancelled", usage: totalUsage };

      // Proactive compaction: check if history exceeds 90% of context window.
      const window = getContextWindow(session.model);
      const limit = Math.floor(window * 0.9);
      if (estimateTokens(session.messages) > limit) {
        session.messages = compactMessages(session.messages, Math.floor(window * 0.8));
      }

      debug(`promptLoop: turn=${turn} session=${sessionId} model=${session.model} messages=${session.messages.length}`);

      const toolCalls: Array<{
        id: string;
        name: string;
        arguments: string;
      }> = [];

      let assistantText = "";
      let lastStopReason: string | undefined;
      let turnUsage: Usage | undefined;
      let usageCommitted = false;
      let cancelledDuringStream = false;
      const commitTurnUsage = (): void => {
        if (usageCommitted) return;
        usageCommitted = true;
        addUsage(turnUsage);
      };

      let retryTurn = false;
      try {
        // Stream the GLM response. The session's currently-selected model wins;
        // it's mutated by `unstable_setSessionModel` between turns.
        for await (const chunk of this.glm.streamChat(session.messages, signal, {
          model: session.model,
          tools: session.toolDefinitions,
          reasoningEffort: session.thoughtLevel,
        })) {
          if (signal.aborted || !ownsPrompt()) {
            cancelledDuringStream = true;
            break;
          }

          if (chunk.thinking && this.streamThinking && ownsPrompt()) {
            await this.connection.sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "agent_thought_chunk",
                content: { type: "text", text: chunk.thinking },
              },
            });
          }

          if (chunk.text && ownsPrompt()) {
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
            // A streamed response can repeat its final usage snapshot. Keep
            // the last snapshot for this model call and merge it once below.
            turnUsage = chunk.usage;
          }

          if (chunk.done) {
            lastStopReason = chunk.stopReason;
          }
        }
      } catch (err) {
        commitTurnUsage();
        if (signal.aborted) {
          cancelledDuringStream = true;
        }

        const body = (err as { error?: { code?: string | number } })?.error;
        const isOverflow =
          body?.code === ERR_CONTEXT_OVERFLOW ||
          body?.code === String(ERR_CONTEXT_OVERFLOW);

        if (!cancelledDuringStream && isOverflow && overflowRetryCount < 1) {
          debug(`promptLoop: context overflow (1261) detected, performing emergency compaction`);
          const window = getContextWindow(session.model);
          session.messages = compactMessages(session.messages, Math.floor(window * 0.7), {
            force: true,
          });
          overflowRetryCount++;
          retryTurn = true;
        } else if (!cancelledDuringStream && isOverflow) {
          throw new Error("Context overflow persisted after emergency compaction", { cause: err });
        } else if (!cancelledDuringStream) {
          throw err;
        }
      }

      if (retryTurn) {
        turn--; // Re-run the same turn index
        continue;
      }

      commitTurnUsage();

      // Record the assistant turn in history so the model has full context for
      // the next iteration.
      const retainDrainedHistory = preservesDrainingHistory();
      if (!ownsPrompt() && !retainDrainedHistory) {
        return { stopReason: "cancelled", usage: totalUsage };
      }
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

      if (cancelledDuringStream || signal.aborted) {
        if (ownsPrompt() || retainDrainedHistory) {
          for (const tc of toolCalls) cancelledToolResult(tc.id);
        }
        return { stopReason: "cancelled", usage: totalUsage };
      }

      // No tool calls => model is done.
      if (toolCalls.length === 0) {
        return { stopReason: this.mapStopReason(lastStopReason), usage: totalUsage };
      }

      // Execute tool calls in declaration order and feed each result back.
      let cancellationObserved = false;
      for (const tc of toolCalls) {
        if (signal.aborted || !ownsPrompt()) {
          if (ownsPrompt() || preservesDrainingHistory()) cancelledToolResult(tc.id);
          cancellationObserved = true;
          continue;
        }

        let result: { content: string };
        try {
          result = await executor.execute(tc.id, tc.name, tc.arguments);
        } catch (err) {
          if (!signal.aborted && !preservesDrainingHistory()) throw err;
          if (ownsPrompt() || preservesDrainingHistory()) cancelledToolResult(tc.id);
          cancellationObserved = true;
          continue;
        }
        debug(`promptLoop: toolResult id=${tc.id} name=${tc.name} contentLength=${result.content.length}`);

        const retainToolResult = preservesDrainingHistory();
        if (!ownsPrompt() && !retainToolResult) {
          cancellationObserved = true;
          continue;
        }
        session.messages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result.content,
        });
        if (signal.aborted || !ownsPrompt()) cancellationObserved = true;
      }

      if (cancellationObserved || signal.aborted || !ownsPrompt()) {
        return { stopReason: "cancelled", usage: totalUsage };
      }

      // Loop and continue – GLM expects a follow-up completion now that it has
      // tool results.
    }

    // Reached MAX_TURNS without resolution. Tell the user why we stopped —
    // without this, hitting the cap is indistinguishable from a normal end.
    if (!ownsPrompt()) return { stopReason: "cancelled", usage: totalUsage };
    await this.connection.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "agent_message_chunk",
        content: {
          type: "text",
          text: `\n[stopped: reached the ${this.maxTurns}-turn limit — send a message to continue]`,
        },
      },
    });
    return { stopReason: "max_turn_requests", usage: totalUsage };
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

  /** Tool schemas we expose for agent-owned local tools plus session MCP tools. */
  private availableToolDefinitions(mcpTools: SessionMcpTools | null = null): ToolDefinition[] {
    const names = ["read_file", "write_file", "edit_file", "todowrite", "list_files", "run_command", "web_search", "web_reader"];
    if (this.visionClientExplicit ? this._visionClient !== null : true) {
      names.push("image_analysis");
    }
    const allowed = new Set(names);
    return [
      ...TOOL_DEFINITIONS.filter((tool) => allowed.has(tool.function.name)),
      ...(mcpTools?.toolDefinitions ?? []),
    ];
  }
}

/**
 * Rebuild the leading system prompt for a session entering a new process or
 * working directory while retaining the conversation messages that follow it.
 * The returned array is always new so a fork can refresh its prompt without
 * changing the source session's message history.
 */
function rebuildRestoredMessages(
  messages: GlmMessage[],
  cwd: string,
  toolDefinitions: ReadonlyArray<ToolDefinition>
): GlmMessage[] {
  const systemPrompt: GlmMessage = {
    role: "system",
    content: buildSystemPrompt({
      cwd,
      tools: toolDefinitions.map((tool) => tool.function.name),
      agentsMd: loadProjectContext(cwd),
    }),
  };
  return messages[0]?.role === "system"
    ? [systemPrompt, ...messages.slice(1)]
    : [systemPrompt, ...messages];
}

/**
 * Render a list of ACP content blocks (after image preprocessing) into the
 * plain-string user message we send to the chat-completions endpoint.
 *
 * `plainText` mirrors `content` and is used by the agent to derive a session
 * title.
 */
function renderPromptBlocks(blocks: ReadonlyArray<PromptRequest["prompt"][number]>): {
  content: string;
  plainText: string;
} {
  const textParts: string[] = [];
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
          textParts.push(`[binary resource](${res.uri})`);
        }
        break;
      }
      case "image": {
        // Only the display-text pass sees these; the model-facing pass has had
        // its images preprocessed into `<image_analysis>` annotations upstream.
        textParts.push(`[image: ${block.mimeType}]`);
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
  return { content: plainText, plainText };
}

function renderVisionNativePromptBlocks(
  blocks: ReadonlyArray<PromptRequest["prompt"][number]>
): ChatCompletionContentPart[] {
  const content: ChatCompletionContentPart[] = [];
  let imageIndex = 0;

  const pushText = (text: string) => {
    if (text.length === 0) return;
    content.push({ type: "text", text });
  };

  for (const block of blocks) {
    switch (block.type) {
      case "text":
        pushText(block.text);
        break;
      case "resource_link":
        pushText(`[${block.name}](${block.uri})`);
        break;
      case "resource": {
        const res = block.resource;
        if ("text" in res && typeof res.text === "string") {
          pushText(`<resource uri="${res.uri}">\n${res.text}\n</resource>`);
        } else if ("blob" in res) {
          pushText(`[binary resource](${res.uri})`);
        }
        break;
      }
      case "image": {
        imageIndex += 1;
        const imageUrl = toVisionNativeImageUrl(block);
        if (imageUrl) {
          content.push({ type: "image_url", image_url: { url: imageUrl } });
        } else {
          pushText(
            `<image_unsupported_format index="${imageIndex}" mime="${escapeAttribute(block.mimeType)}">Only image/jpeg, image/jpg, image/png inputs can be sent to the native vision model. Attach a supported HTTPS image URL or supported base64 image data.</image_unsupported_format>`
          );
        }
        break;
      }
      case "audio":
        pushText("[unsupported audio block]");
        break;
      default:
        pushText(`[unknown block type ${(block as { type: string }).type}]`);
    }
  }

  return content;
}

function toVisionNativeImageUrl(block: Extract<PromptRequest["prompt"][number], { type: "image" }>): string | null {
  if (!isSupportedVisionMime(block.mimeType)) return null;

  if (typeof block.uri === "string" && block.uri.length > 0) {
    if (block.uri.startsWith("data:")) return block.uri;
    try {
      const url = new URL(block.uri);
      if (url.protocol === "https:") return block.uri;
    } catch {
      // Fall through to inline data when available.
    }
  }

  if (typeof block.data === "string" && block.data.length > 0) {
    return `data:${block.mimeType.toLowerCase()};base64,${block.data}`;
  }

  return null;
}

function isSupportedVisionMime(mimeType: string): boolean {
  switch (mimeType.toLowerCase()) {
    case "image/jpeg":
    case "image/jpg":
    case "image/png":
      return true;
    default:
      return false;
  }
}

function escapeAttribute(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

/**
 * Project the identity-keyed display map onto indices into `messages`, ready
 * to persist. Returns `undefined` when nothing diverges so the field can be
 * left off the record entirely.
 */
function serializeDisplayText(
  messages: ReadonlyArray<GlmMessage>,
  displayText: SessionState["displayText"]
): Record<string, string> | undefined {
  let out: Record<string, string> | undefined;
  for (const [index, message] of messages.entries()) {
    const text = displayText.get(message);
    if (text === undefined) continue;
    out ??= {};
    out[String(index)] = text;
  }
  return out;
}

/** Rebuild the identity-keyed display map from a persisted index map. */
function deserializeDisplayText(
  messages: ReadonlyArray<GlmMessage>,
  persisted: Record<string, string> | undefined
): SessionState["displayText"] {
  const out = new WeakMap<GlmMessage, string>();
  if (!persisted) return out;
  for (const [key, text] of Object.entries(persisted)) {
    const message = messages[Number(key)];
    // A hand-edited or truncated record can point past the end of the array;
    // dropping the entry just falls back to replaying the stored content.
    if (message) out.set(message, text);
  }
  return out;
}

/** Re-key display text when restoring a record that needed a new system message prepended. */
function deserializeRestoredDisplayText(
  sourceMessages: ReadonlyArray<GlmMessage>,
  restoredMessages: ReadonlyArray<GlmMessage>,
  persisted: Record<string, string> | undefined
): SessionState["displayText"] {
  const offset = restoredMessages.length - sourceMessages.length;
  if (!persisted || offset === 0) {
    return deserializeDisplayText(restoredMessages, persisted);
  }
  const shifted = Object.fromEntries(
    Object.entries(persisted).map(([index, text]) => [String(Number(index) + offset), text])
  );
  return deserializeDisplayText(restoredMessages, shifted);
}

/** Flatten the `content` of a user message into a plain string for replay. */
function stringifyUserMessage(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (
        typeof part === "object" &&
        part !== null &&
        (part as { type?: unknown }).type === "text" &&
        typeof (part as { text?: unknown }).text === "string"
      ) {
        return (part as { text: string }).text;
      }
      if (
        typeof part === "object" &&
        part !== null &&
        (part as { type?: unknown }).type === "image_url"
      ) {
        return "[image]";
      }
      return "";
    })
    .filter((text) => text.length > 0)
    .join("\n");
}

/**
 * The advertised model list, always including `currentModelId` — a session
 * restored from disk can be pinned to a de-listed id (`glm-5.2` was the
 * previous default), and `ACP_GLM_MODEL` / `session/set_model` / the `model`
 * config option all accept uncatalogued ids on purpose. Dropping the active id
 * from the list would leave pickers unable to represent the selection in use.
 */
function availableModelsWith(currentModelId: string): ModelInfo[] {
  const available = getAvailableModels();
  return available.some((m) => m.modelId === currentModelId)
    ? available
    : [...available, { modelId: currentModelId, name: currentModelId }];
}

/**
 * Map discovered commands onto the ACP wire shape. Names are sent *without* a
 * leading slash — the client prepends it for display and sends `/name …` back
 * as prompt text. `input` is omitted for commands that declare no
 * `argument-hint`, which is how a client learns the command takes no extra text.
 */
function availableCommandsState(
  commands: ReadonlyArray<SlashCommand>
): AvailableCommand[] {
  return commands.map((command) => ({
    name: command.name,
    description: command.description,
    ...(command.argumentHint !== undefined
      ? { input: { hint: command.argumentHint } }
      : {}),
  }));
}

/**
 * Expand a leading `/name` in the prompt's first written text into the
 * command's body.
 *
 * The target is the first text block that actually has content, not block 0:
 * a client may place an image or resource block ahead of what the user typed,
 * and the command would otherwise reach the model as a literal `/name`.
 *
 * Returns the blocks unchanged when that text does not start with an
 * advertised command. The caller keeps the original blocks around to derive
 * the replay/title text, so nothing here needs to report the typed form back.
 */
function expandPromptCommand(
  blocks: PromptRequest["prompt"],
  commands: ReadonlyArray<SlashCommand>
): PromptRequest["prompt"] {
  const index = blocks.findIndex(
    (block) => block.type === "text" && block.text.trim().length > 0
  );
  const typed = blocks[index];
  if (typed?.type !== "text") return blocks;
  const parsed = parseSlashCommand(typed.text, commands);
  if (!parsed) return blocks;
  const expanded = [...blocks];
  expanded[index] = { ...typed, text: renderSlashCommand(parsed) };
  return expanded;
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

/** Reject the owner immediately while retaining a handler for a late setup result. */
function waitForAbort<T>(promise: Promise<T>, signal: AbortSignal, message: string): Promise<T> {
  if (signal.aborted) return Promise.reject(new Error(message));
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const claim = (): boolean => {
      if (settled) return false;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      return true;
    };
    const onAbort = () => {
      if (claim()) reject(new Error(message));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        if (claim()) resolve(value);
      },
      (error) => {
        if (claim()) reject(error);
      }
    );
  });
}

/**
 * Token cost charged to a single `image_url` content part.
 *
 * Vision-native GLM models tile an image into 14px patches and merge them 2x2,
 * so a full-size input (capped at 1120x1120) costs (1120/14)^2 / 4 = 1600
 * tokens. The wire form — an HTTPS URL or a base64 data URL — says nothing
 * about the decoded dimensions, so charge every image that ceiling:
 * over-counting only makes compaction fire early, while under-counting is what
 * lets an image-heavy session sail past the window and get rejected.
 */
const IMAGE_PART_TOKENS = 1600;

/**
 * Heuristically estimate the number of tokens in a list of messages.
 * Uses a simple 4-character-per-token rule, which is a safe baseline for
 * English and code, plus a flat per-image charge for native `image_url` parts
 * (see {@link IMAGE_PART_TOKENS}).
 */
function estimateTokens(messages: GlmMessage[]): number {
  let chars = 0;
  let tokens = 0;
  for (const m of messages) {
    if (typeof m.content === "string") {
      chars += m.content.length;
    } else if (Array.isArray(m.content)) {
      for (const part of m.content) {
        if ("text" in part && typeof part.text === "string") {
          chars += part.text.length;
        } else if (part.type === "image_url") {
          tokens += IMAGE_PART_TOKENS;
        }
      }
    }
    if (m.role === "assistant" && m.tool_calls) {
      for (const tc of m.tool_calls) {
        if ("function" in tc) {
          chars += tc.function.name.length;
          chars += tc.function.arguments.length;
        }
      }
    }
  }
  return tokens + Math.ceil(chars / 4);
}

async function settlesWithin(promise: Promise<unknown>, timeoutMs: number): Promise<boolean> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise.then(() => true, () => true),
      new Promise<false>((resolve) => { timer = setTimeout(() => resolve(false), timeoutMs); }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Prune message history to stay within a target token limit.
 *
 * Strategy:
 * 1. Always keep the System prompt (index 0).
 * 2. Always keep the last `preserveTurns` interaction groups (default 10) to
 *    maintain conversation flow. An interaction group (turn) typically starts
 *    with a user message followed by assistant and tool responses.
 * 3. Evict the largest remaining interaction groups until the total estimate
 *    is below `targetTokens`.
 *
 * `force` is for the emergency path after the provider itself rejected the
 * history. There {@link estimateTokens} has been proven wrong, so it can set
 * neither the stopping point nor what is off limits: the target is halved, and
 * the preserved tail becomes evictable from its oldest end. Only the final turn
 * is sacred — it carries the live user message, and a request without it is not
 * a retry of anything.
 */
function compactMessages(
  messages: GlmMessage[],
  targetTokens: number,
  { preserveTurns = 10, force = false }: { preserveTurns?: number; force?: boolean } = {}
): GlmMessage[] {
  if (messages.length <= 1) return messages;

  const systemPrompt = messages[0];
  const remaining = messages.slice(1);

  // Group messages into interaction turns. A turn starts with a "user" message.
  const turns: GlmMessage[][] = [];
  let currentTurn: GlmMessage[] = [];

  for (const m of remaining) {
    if (m.role === "user" && currentTurn.length > 0) {
      turns.push(currentTurn);
      currentTurn = [];
    }
    currentTurn.push(m);
  }
  if (currentTurn.length > 0) {
    turns.push(currentTurn);
  }

  // The final turn holds the live user message, so it is never evictable —
  // with nothing else to drop there is no compaction to do.
  if (turns.length < 2) return messages;
  if (!force && turns.length <= preserveTurns) return messages;

  let currentEstimate = estimateTokens(messages);
  if (!force && currentEstimate <= targetTokens) return messages;

  // An estimate already above target names a real reduction to aim for, forced
  // or not. One that sits *below* target while the provider is rejecting the
  // payload has been disproven, and stopping on it would spend the single retry
  // on another oversized request — so halve it instead. Wrong by an unknown
  // factor still shrinks geometrically, and only that case pays the extra loss.
  const estimateDisproven = force && currentEstimate <= targetTokens;
  const effectiveTarget = estimateDisproven
    ? Math.floor(currentEstimate / 2)
    : targetTokens;

  debug(
    `compactMessages: currentEstimate=${currentEstimate} target=${effectiveTarget} turns=${turns.length} force=${force}`
  );

  const sized = turns.map((turn, index) => ({ index, tokens: estimateTokens(turn) }));
  const protectedFrom = turns.length - preserveTurns;
  const evictedIndices = new Set<number>();

  // Largest first, among the turns outside the preserved tail.
  const candidateTurns = sized
    .filter((c) => c.index < protectedFrom)
    .sort((a, b) => b.tokens - a.tokens);
  for (const c of candidateTurns) {
    if (currentEstimate <= effectiveTarget) break;
    evictedIndices.add(c.index);
    currentEstimate -= c.tokens;
  }

  // Still over, and forced? Then the preserved tail is itself the problem — a
  // run of image-heavy prompts can exceed the window on its own, leaving the
  // candidates above unable to reach the target however many are dropped. Eat
  // into the tail from its oldest end so the freshest context survives.
  if (force) {
    for (let i = Math.max(protectedFrom, 0); i < turns.length - 1; i++) {
      if (currentEstimate <= effectiveTarget) break;
      evictedIndices.add(i);
      currentEstimate -= sized[i].tokens;
    }
  }

  const compacted: GlmMessage[] = [systemPrompt];
  for (let i = 0; i < turns.length; i++) {
    if (!evictedIndices.has(i)) {
      compacted.push(...turns[i]);
    }
  }

  debug(`compactMessages: done, newEstimate=${estimateTokens(compacted)} messageCount=${compacted.length}`);
  return compacted;
}
