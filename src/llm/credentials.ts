import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";

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
 */

interface CredentialsFile {
  z_ai_api_key?: string;
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

/**
 * Resolve the API key: env var > credentials file. Returns undefined when
 * neither source has one set.
 */
export function resolveApiKey(): string | undefined {
  const fromEnv = process.env["Z_AI_API_KEY"];
  if (typeof fromEnv === "string" && fromEnv.length > 0) return fromEnv;
  return readCredentialsKey();
}

/**
 * Write the API key to the credentials file. Restricts the file to mode 0600
 * so it isn't world-readable on shared machines.
 */
export function writeCredentials(apiKey: string, path: string = credentialsPath()): void {
  if (!apiKey || apiKey.length === 0) {
    throw new Error("Refusing to write empty API key");
  }
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  const body: CredentialsFile = { z_ai_api_key: apiKey };
  writeFileSync(path, JSON.stringify(body, null, 2) + "\n", { mode: 0o600 });
}
