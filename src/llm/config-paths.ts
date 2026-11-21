import { realpath } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, resolve, sep } from "node:path";

/** Directory holding the agent's own configuration. Honours `$XDG_CONFIG_HOME`. */
export function agentConfigDir(): string {
  const xdg = process.env["XDG_CONFIG_HOME"];
  const base = xdg && xdg.length > 0 ? xdg : join(homedir(), ".config");
  return join(base, "glm-acp-agent");
}

export interface GlobalContextSetting {
  /** Absolute path to read, or undefined when global context is disabled or invalid. */
  path: string | undefined;
  /** True when `ACP_GLM_GLOBAL_CONTEXT_FILE` was set, so a missing file is worth reporting. */
  explicit: boolean;
  /** Human-readable reason the configured value was rejected. */
  problem?: string;
}

/**
 * Resolve where the global context file lives. Unset means
 * `<agent config dir>/AGENTS.md`; `none` or an empty value disables it.
 * `~/` is expanded. Anything else must be absolute: a relative value would
 * resolve against the process cwd and could promote a repository file to
 * trusted global instructions.
 */
export function globalContextSetting(): GlobalContextSetting {
  const configured = process.env["ACP_GLM_GLOBAL_CONTEXT_FILE"];
  if (configured === undefined) {
    return { path: join(agentConfigDir(), "AGENTS.md"), explicit: false };
  }
  const trimmed = configured.trim();
  if (trimmed === "" || trimmed.toLowerCase() === "none") return { path: undefined, explicit: true };
  const expanded = trimmed.startsWith("~/") ? join(homedir(), trimmed.slice(2)) : trimmed;
  if (!isAbsolute(expanded)) {
    return {
      path: undefined,
      explicit: true,
      problem: `ACP_GLM_GLOBAL_CONTEXT_FILE must be an absolute path or start with "~/" (got "${trimmed}")`,
    };
  }
  return { path: expanded, explicit: true };
}

async function resolveLoose(path: string): Promise<string> {
  try {
    return await realpath(path);
  } catch {
    try {
      return join(await realpath(dirname(path)), basename(path));
    } catch {
      return resolve(path);
    }
  }
}

/**
 * True when `absolutePath` is the global context file or lies inside the
 * agent's config directory. Writes there change instructions or credentials
 * that outlive the session, so they always need an explicit prompt.
 */
export async function isProtectedConfigPath(absolutePath: string): Promise<boolean> {
  const target = await resolveLoose(absolutePath);
  const protectedRoots = [agentConfigDir()];
  const globalPath = globalContextSetting().path;
  const candidates: Array<{ path: string; dir: boolean }> = protectedRoots.map((path) => ({ path, dir: true }));
  if (globalPath !== undefined) candidates.push({ path: globalPath, dir: false });
  for (const candidate of candidates) {
    const resolved = await resolveLoose(candidate.path);
    if (target === resolved) return true;
    if (candidate.dir && target.startsWith(resolved + sep)) return true;
  }
  return false;
}
