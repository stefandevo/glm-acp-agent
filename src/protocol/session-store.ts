import { mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { GlmMessage } from "../llm/glm-client.js";

/**
 * Schema version embedded in every persisted session file. Bump whenever the
 * shape of `PersistedSession` changes incompatibly so future loaders can
 * migrate (or reject) old records instead of silently producing garbage.
 */
export const SESSION_SCHEMA_VERSION = 3 as const;

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
   * Reasoning effort level. Defaults to "max" for sessions persisted before
   * the thought_level config option was added.
   */
  thoughtLevel?: "none" | "high" | "max";
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
    mkdirSync(this.dir, { recursive: true, mode: 0o700 });
    const path = this.pathFor(session.sessionId);
    // Write the schema version *after* the spread so the constant always wins,
    // even if a caller accidentally sets `schemaVersion` on the input.
    const body: PersistedSession = {
      ...session,
      schemaVersion: SESSION_SCHEMA_VERSION,
    };
    writeFileSync(path, JSON.stringify(body, null, 2) + "\n", { mode: 0o600 });
  }

  /** Load a session by id, returning undefined if no such file exists. */
  load(sessionId: string): PersistedSession | undefined {
    let raw: string;
    try {
      raw = readFileSync(this.pathFor(sessionId), "utf8");
    } catch {
      return undefined;
    }
    let parsed: PersistedSession;
    try {
      parsed = JSON.parse(raw) as PersistedSession;
    } catch {
      return undefined;
    }
    // Handle schema migrations. We support v1 (pre-modes), v2 (with mode),
    // and v3 (with thoughtLevel).
    const version = parsed.schemaVersion ?? 1;
    if (version === 1) {
      return {
        ...parsed,
        mode: "default",
        thoughtLevel: "max",
        schemaVersion: SESSION_SCHEMA_VERSION,
      };
    }
    if (version === 2) {
      return {
        ...parsed,
        thoughtLevel: "max",
        schemaVersion: SESSION_SCHEMA_VERSION,
      };
    }
    if (version !== SESSION_SCHEMA_VERSION) {
      // Forward-incompatible record written by a newer agent build.
      return undefined;
    }
    return parsed;
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
