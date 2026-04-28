import { mkdirSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { GlmMessage } from "../llm/glm-client.js";

/**
 * On-disk representation of a session. Only fields that need to survive a
 * process restart are persisted — `abortController` / `promptPromise` are
 * transient state that has no meaning across processes.
 */
export interface PersistedSession {
  sessionId: string;
  cwd: string;
  messages: GlmMessage[];
  title: string | null;
  updatedAt: string;
  model: string;
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
    writeFileSync(path, JSON.stringify(session, null, 2) + "\n", { mode: 0o600 });
  }

  /** Load a session by id, returning undefined if no such file exists. */
  load(sessionId: string): PersistedSession | undefined {
    let raw: string;
    try {
      raw = readFileSync(this.pathFor(sessionId), "utf8");
    } catch {
      return undefined;
    }
    const parsed = JSON.parse(raw) as PersistedSession;
    return parsed;
  }

  /** List all persisted sessions (lightweight metadata only). */
  list(): PersistedSession[] {
    let entries: string[];
    try {
      entries = readdirSync(this.dir);
    } catch {
      return [];
    }
    const out: PersistedSession[] = [];
    for (const name of entries) {
      if (!name.endsWith(".json")) continue;
      const sessionId = name.slice(0, -".json".length);
      const sess = this.load(sessionId);
      if (sess) out.push(sess);
    }
    return out;
  }
}
