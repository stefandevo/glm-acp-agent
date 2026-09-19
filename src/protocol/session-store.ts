import { randomUUID } from "node:crypto";
import {
  closeSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, join } from "node:path";
import type { GlmMessage, ThoughtLevel } from "../llm/glm-client.js";

/**
 * Schema version embedded in every persisted session file. Bump whenever the
 * shape of `PersistedSession` changes incompatibly so future loaders can
 * migrate (or reject) old records instead of silently producing garbage.
 */
export const SESSION_SCHEMA_VERSION = 4 as const;

/**
 * On-disk representation of a session. Only fields that need to survive a
 * process restart are persisted — `abortController` / `promptPromise` are
 * transient state that has no meaning across processes.
 */
export interface PersistedSession {
  /** Schema version of this on-disk record (see SESSION_SCHEMA_VERSION). */
  schemaVersion?: number;
  sessionId: string;
  cwd: string;
  messages: GlmMessage[];
  title: string | null;
  updatedAt: string;
  model: string;
  /**
   * Permission mode for this session. Defaults to "default" for persisted
   * sessions from schema versions that didn't include this field.
   */
  mode: "default" | "accept_edits" | "bypass_permissions";
  /**
   * Reasoning effort level, controlled via the `thought_level` config option.
   * Optional so sessions persisted before this field was added still parse;
   * the migration (and load-time resolution) default it to "max".
   */
  thoughtLevel?: ThoughtLevel;
  /**
   * Replay text for user messages whose stored `content` differs from what the
   * user actually typed — a slash command expanded into its body, an image
   * replaced by its vision annotation. Keys are indices into `messages`;
   * entries are written only where the two texts diverge, so an ordinary
   * conversation persists no sidecar at all.
   *
   * Indices are re-derived from message identity on every save, so they stay
   * correct across the compaction that drops turns from `messages`.
   */
  displayText?: Record<string, string>;
}

/** Light-weight summary of a persisted session — used by `listSessions`. */
export interface PersistedSessionMetadata {
  sessionId: string;
  cwd: string;
  title: string | null;
  updatedAt: string;
  model: string;
  mode: "default" | "accept_edits" | "bypass_permissions";
}

const VALID_MODES = new Set<PersistedSession["mode"]>([
  "default",
  "accept_edits",
  "bypass_permissions",
]);
const VALID_THOUGHT_LEVELS = new Set<ThoughtLevel>([
  "none",
  "on",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);
const VALID_MESSAGE_ROLES = new Set(["system", "developer", "user", "assistant", "tool", "function"]);

type UnknownRecord = Record<string, unknown>;

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

type ContentPartKind = "text" | "refusal" | "image_url" | "input_audio" | "file";

function isContent(value: unknown, allowed: readonly ContentPartKind[], allowNull = false): boolean {
  if (typeof value === "string") return true;
  if (value === null) return allowNull;
  return Array.isArray(value) && value.every((part) => isContentPart(part, allowed));
}

function isContentPart(value: unknown, allowed: readonly ContentPartKind[]): boolean {
  if (!isRecord(value) || typeof value.type !== "string" || !allowed.includes(value.type as ContentPartKind)) {
    return false;
  }
  switch (value.type) {
    case "text":
    case "refusal":
      return typeof value[value.type] === "string";
    case "image_url":
      return isRecord(value.image_url) && typeof value.image_url.url === "string";
    case "input_audio":
      return (
        isRecord(value.input_audio) &&
        typeof value.input_audio.data === "string" &&
        (value.input_audio.format === "wav" || value.input_audio.format === "mp3")
      );
    case "file":
      {
        const file = value.file;
        if (!isRecord(file)) return false;
        let hasValue = false;
        for (const key of ["file_data", "file_id", "filename"] as const) {
          if (!(key in file)) continue;
          if (typeof file[key] !== "string" || file[key].length === 0) return false;
          hasValue = true;
        }
        return hasValue;
      }
    default:
      return false;
  }
}

/** Validate the message shapes emitted by OpenAI's Chat Completions API. */
function isGlmMessage(value: unknown): value is GlmMessage {
  if (!isRecord(value) || typeof value.role !== "string" || !VALID_MESSAGE_ROLES.has(value.role)) {
    return false;
  }

  if (value.role !== "assistant" && !("content" in value)) return false;
  if (value.role === "system" && !isContent(value.content, ["text"])) return false;
  if (value.role === "developer" && !isContent(value.content, ["text"])) return false;
  if (value.role === "user" && !isContent(value.content, ["text", "image_url", "input_audio", "file"])) {
    return false;
  }
  if (value.role === "tool" && (!isContent(value.content, ["text"]) || typeof value.tool_call_id !== "string")) {
    return false;
  }
  if (value.role === "function" &&
      (typeof value.name !== "string" || (value.content !== null && typeof value.content !== "string"))) {
    return false;
  }
  if (value.role === "assistant" && "content" in value && !isContent(value.content, ["text", "refusal"], true)) {
    return false;
  }

  if ("tool_call_id" in value && (value.role !== "tool" || typeof value.tool_call_id !== "string")) return false;
  if ("name" in value && typeof value.name !== "string") return false;
  if ("reasoning_content" in value && (value.role !== "assistant" || typeof value.reasoning_content !== "string")) return false;
  if ("tool_calls" in value) {
    if (value.role !== "assistant") return false;
    if (!Array.isArray(value.tool_calls)) return false;
    for (const call of value.tool_calls) {
      if (!isRecord(call) || typeof call.id !== "string") return false;
      if (call.type === "function") {
        if (!isRecord(call.function)) return false;
        if (typeof call.function.name !== "string" || typeof call.function.arguments !== "string") return false;
      } else if (call.type === "custom") {
        if (!isRecord(call.custom)) return false;
        if (typeof call.custom.name !== "string" || typeof call.custom.input !== "string") return false;
      } else {
        return false;
      }
    }
  }

  // Assistant tool-call messages may legitimately omit content. A normal
  // assistant message still needs content (including explicit null).
  if (value.role === "assistant" && !("content" in value) && !("tool_calls" in value) && !("function_call" in value)) {
    return false;
  }
  if ("function_call" in value) {
    if (value.role !== "assistant") return false;
    if (value.function_call !== null && !isRecord(value.function_call)) return false;
    if (
      value.function_call !== null &&
      (typeof value.function_call.name !== "string" ||
        typeof value.function_call.arguments !== "string")
    ) {
      return false;
    }
  }
  return true;
}

function parsePersistedSession(value: unknown, expectedSessionId: string): PersistedSession | undefined {
  if (!isRecord(value)) return undefined;

  const rawVersion = value.schemaVersion;
  if (rawVersion !== undefined &&
      (typeof rawVersion !== "number" || !Number.isInteger(rawVersion) || rawVersion < 1 || rawVersion > SESSION_SCHEMA_VERSION)) {
    return undefined;
  }
  const version = rawVersion === undefined ? 1 : rawVersion;
  if (typeof value.sessionId !== "string" || value.sessionId !== expectedSessionId) return undefined;
  if (
    typeof value.cwd !== "string" ||
    typeof value.updatedAt !== "string" ||
    Number.isNaN(Date.parse(value.updatedAt)) ||
    typeof value.model !== "string"
  ) {
    return undefined;
  }
  if (!Array.isArray(value.messages) || !value.messages.every(isGlmMessage)) return undefined;
  if (value.title !== null && typeof value.title !== "string") return undefined;

  const mode = value.mode;
  if (version === 1) {
    if (mode !== undefined && (typeof mode !== "string" || !VALID_MODES.has(mode as PersistedSession["mode"]))) {
      return undefined;
    }
  } else if (typeof mode !== "string" || !VALID_MODES.has(mode as PersistedSession["mode"])) {
    return undefined;
  }

  if (value.thoughtLevel !== undefined &&
      (typeof value.thoughtLevel !== "string" || !VALID_THOUGHT_LEVELS.has(value.thoughtLevel as ThoughtLevel))) {
    return undefined;
  }

  if (value.displayText !== undefined) {
    if (!isRecord(value.displayText)) return undefined;
    for (const [index, text] of Object.entries(value.displayText)) {
      if (!/^\d+$/.test(index) || typeof text !== "string") return undefined;
    }
  }

  const parsed = value as unknown as PersistedSession;
  if (version === 1) {
    return {
      ...parsed,
      mode: "default",
      thoughtLevel: parsed.thoughtLevel ?? "max",
      schemaVersion: SESSION_SCHEMA_VERSION,
    };
  }
  if (version === 2) {
    return {
      ...parsed,
      thoughtLevel: parsed.thoughtLevel ?? "max",
      schemaVersion: SESSION_SCHEMA_VERSION,
    };
  }
  return {
    ...parsed,
    schemaVersion: SESSION_SCHEMA_VERSION,
  };
}

/** Resolve the directory we write session files to, honouring overrides. */
function defaultSessionDir(): string {
  const explicit = process.env["ACP_GLM_SESSION_DIR"];
  if (explicit && explicit.length > 0) return explicit;
  const xdg = process.env["XDG_STATE_HOME"];
  const base = xdg && xdg.length > 0 ? xdg : join(homedir(), ".local", "state");
  return join(base, "glm-acp-agent", "sessions");
}

/**
 * File-backed session store. Each session lives in its own JSON file so we can
 * grow/shrink linearly with the number of conversations and avoid locking a
 * single shared file.
 */
export class SessionStore {
  private dir: string;

  constructor(dir: string = defaultSessionDir()) {
    this.dir = dir;
  }

  /** Resolve the path for a given sessionId. */
  private pathFor(sessionId: string): string {
    // sessionId is generated via randomUUID() so it's path-safe; reject
    // anything else defensively to avoid path traversal.
    if (!/^[a-zA-Z0-9_-]+$/.test(sessionId)) {
      throw new Error(`Invalid sessionId: ${sessionId}`);
    }
    return join(this.dir, `${sessionId}.json`);
  }

  /** Persist a session, creating directories as needed. */
  save(session: PersistedSession): void {
    const path = this.pathFor(session.sessionId);
    mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    // Write the schema version *after* the spread so the constant always wins,
    // even if a caller accidentally sets `schemaVersion` on the input.
    const body: PersistedSession = {
      ...session,
      schemaVersion: SESSION_SCHEMA_VERSION,
    };
    if (!parsePersistedSession(body, session.sessionId)) {
      throw new Error(`Invalid persisted session: ${session.sessionId}`);
    }

    const tempPath = join(this.dir, `.${basename(path)}.${randomUUID()}.tmp`);
    let fd: number | undefined;
    try {
      fd = openSync(tempPath, "wx", 0o600);
      const contents = JSON.stringify(body, null, 2) + "\n";
      const bytes = Buffer.from(contents, "utf8");
      for (let offset = 0; offset < bytes.byteLength;) {
        offset += writeSync(fd, bytes, offset, bytes.byteLength - offset);
      }
      fsyncSync(fd);
      closeSync(fd);
      fd = undefined;
      renameSync(tempPath, path);
    } finally {
      if (fd !== undefined) {
        try {
          closeSync(fd);
        } catch {
          // Preserve the original save error.
        }
      }
      try {
        unlinkSync(tempPath);
      } catch {
        // The rename succeeded, or the temp file was never created.
      }
    }
  }

  /** Load a session by id, returning undefined if no such file exists. */
  load(sessionId: string): PersistedSession | undefined {
    let path: string;
    try {
      path = this.pathFor(sessionId);
    } catch {
      return undefined;
    }
    let raw: string;
    try {
      raw = readFileSync(path, "utf8");
    } catch {
      return undefined;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw) as unknown;
    } catch {
      return undefined;
    }
    // Handle schema migrations. Valid v1 records have no mode or thought
    // level; v2 added mode, v3 added thoughtLevel, and v4 added displayText.
    return parsePersistedSession(parsed, sessionId);
  }

  /**
   * List metadata for all persisted sessions, sorted newest-first by
   * `updatedAt`. This is the hot path for `session/list`; we still parse each
   * file (single-file-per-session has no shared index), but discard the
   * `messages` array immediately so memory usage scales with the number of
   * sessions, not their length.
   */
  listMetadata(): PersistedSessionMetadata[] {
    let entries: string[];
    try {
      entries = readdirSync(this.dir);
    } catch {
      return [];
    }
    const out: PersistedSessionMetadata[] = [];
    for (const name of entries) {
      if (!name.endsWith(".json")) continue;
      const sessionId = name.slice(0, -".json".length);
      const sess = this.load(sessionId);
      if (!sess) continue;
      out.push({
        sessionId: sess.sessionId,
        cwd: sess.cwd,
        title: sess.title,
        updatedAt: sess.updatedAt,
        model: sess.model,
        mode: sess.mode,
      });
    }
    out.sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : a.updatedAt > b.updatedAt ? -1 : 0));
    return out;
  }
}
