import {
  closeSync,
  constants,
  fchmodSync,
  ftruncateSync,
  mkdirSync,
  openSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { debug, maskSecret } from "./logger.js";

/**
 * Credential management for the Z.AI API key.
 *
 * Resolution order on startup:
 *   1. `Z_AI_API_KEY` environment variable (highest priority — easiest for CI
 *      and clients that pass env vars).
 *   2. The credentials file written by `glm-acp-agent --setup`. This lets users
 *      configure the agent once on their machine without leaking the key into
 *      shell history.
 *
 * The credentials file lives under `$XDG_CONFIG_HOME/glm-acp-agent/credentials.json`
 * (or `~/.config/glm-acp-agent/credentials.json` if `XDG_CONFIG_HOME` is unset).
 *
 * The file and environment can also hold an optional Z.AI team seat
 * (organization ID + project ID), used only for quota lookups.
 */

interface CredentialsFile {
  z_ai_api_key?: string;
  z_ai_organization_id?: string;
  z_ai_project_id?: string;
}

/**
 * Optional Z.AI team-seat identifiers. Both must be present to select team
 * quota; a lone value is a misconfiguration that callers must not send.
 */
export interface TeamSeat {
  organizationId?: string;
  projectId?: string;
}

/** Path to the credentials JSON file. Honours `$XDG_CONFIG_HOME`. */
export function credentialsPath(): string {
  const xdg = process.env["XDG_CONFIG_HOME"];
  const base = xdg && xdg.length > 0 ? xdg : join(homedir(), ".config");
  return join(base, "glm-acp-agent", "credentials.json");
}

/** Read the API key from the credentials file, returning undefined if absent. */
export function readCredentialsKey(path: string = credentialsPath()): string | undefined {
  try {
    const raw = readFileSync(path, "utf8");
    const parsed = JSON.parse(raw) as CredentialsFile;
    const key = parsed.z_ai_api_key;
    return typeof key === "string" && key.length > 0 ? key : undefined;
  } catch {
    return undefined;
  }
}

function nonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

/** Read the optional team-seat IDs from the credentials file. */
export function readCredentialsTeamSeat(path: string = credentialsPath()): TeamSeat {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as CredentialsFile;
    const organizationId = nonEmptyString(parsed.z_ai_organization_id);
    const projectId = nonEmptyString(parsed.z_ai_project_id);
    return {
      ...(organizationId ? { organizationId } : {}),
      ...(projectId ? { projectId } : {}),
    };
  } catch {
    return {};
  }
}

/**
 * Resolve the optional team seat. The environment pair
 * (`Z_AI_ORGANIZATION_ID` / `Z_AI_PROJECT_ID`) wins as a unit when either
 * variable is set, so a stale file value is never mixed with an env value.
 */
export function resolveTeamSeat(): TeamSeat {
  const organizationId = nonEmptyString(process.env["Z_AI_ORGANIZATION_ID"]);
  const projectId = nonEmptyString(process.env["Z_AI_PROJECT_ID"]);
  if (organizationId || projectId) {
    debug("resolveTeamSeat: source=env");
    return {
      ...(organizationId ? { organizationId } : {}),
      ...(projectId ? { projectId } : {}),
    };
  }
  const fromFile = readCredentialsTeamSeat();
  debug(`resolveTeamSeat: source=${fromFile.organizationId || fromFile.projectId ? "file" : "none"}`);
  return fromFile;
}

/**
 * Resolve the API key: env var > credentials file. Returns undefined when
 * neither source has one set.
 */
export function resolveApiKey(): string | undefined {
  const fromEnv = process.env["Z_AI_API_KEY"];
  if (typeof fromEnv === "string" && fromEnv.length > 0) {
    debug(`resolveApiKey: source=env key=${maskSecret(fromEnv)}`);
    return fromEnv;
  }
  const fromFile = readCredentialsKey();
  if (fromFile) {
    debug(`resolveApiKey: source=file key=${maskSecret(fromFile)}`);
  } else {
    debug("resolveApiKey: no key found");
  }
  return fromFile;
}

/**
 * Write the API key (and optional team seat) to the credentials file.
 * Restricts the file to mode 0600 so it isn't world-readable on shared machines.
 */
export function writeCredentials(apiKey: string, path: string = credentialsPath(), teamSeat: TeamSeat = {}): void {
  if (!apiKey || apiKey.length === 0) {
    throw new Error("Refusing to write empty API key");
  }
  const organizationId = nonEmptyString(teamSeat.organizationId);
  const projectId = nonEmptyString(teamSeat.projectId);
  if (!organizationId !== !projectId) {
    throw new Error("Refusing to write a team seat without both an organization ID and a project ID");
  }
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  const body: CredentialsFile = {
    z_ai_api_key: apiKey,
    ...(organizationId && projectId ? { z_ai_organization_id: organizationId, z_ai_project_id: projectId } : {}),
  };
  const fd = openSync(path, constants.O_WRONLY | constants.O_CREAT, 0o600);
  try {
    fchmodSync(fd, 0o600);
    ftruncateSync(fd, 0);
    writeFileSync(fd, JSON.stringify(body, null, 2) + "\n", "utf8");
  } finally {
    closeSync(fd);
  }
}
