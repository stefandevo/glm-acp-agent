export interface ResourceLimits {
  toolResultBytes: number;
  fileReadBytes: number;
  listEntries: number;
  listBytes: number;
  fsConcurrency: number;
}

type Environment = Record<string, string | undefined>;
type Warn = (message: string) => void;

export const DEFAULT_TOOL_RESULT_BYTES = 262_144;
export const DEFAULT_FILE_READ_BYTES = 8 * 1024 * 1024;
export const DEFAULT_LIST_ENTRIES = 2_000;
export const DEFAULT_LIST_BYTES = 262_144;

function positiveInteger(env: Environment, name: string, fallback: number, minimum: number, maximum: number, warn: Warn): number {
  const raw = env[name];
  if (raw === undefined) return fallback;
  const value = /^\d+$/.test(raw.trim()) ? Number(raw) : NaN;
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    warn(`glm-acp-agent: ignoring invalid ${name} value "${raw}"; using ${fallback}`);
    return fallback;
  }
  return value;
}

export function readResourceLimits(
  env: Environment = process.env,
  warn: Warn = message => process.stderr.write(`${message}\n`),
): ResourceLimits {
  return {
    toolResultBytes: positiveInteger(env, "ACP_GLM_TOOL_RESULT_LIMIT_BYTES", DEFAULT_TOOL_RESULT_BYTES, 128, Number.MAX_SAFE_INTEGER, warn),
    fileReadBytes: positiveInteger(env, "ACP_GLM_READ_FILE_LIMIT_BYTES", DEFAULT_FILE_READ_BYTES, 1, 64 * 1024 * 1024, warn),
    listEntries: positiveInteger(env, "ACP_GLM_LIST_FILES_MAX_ENTRIES", DEFAULT_LIST_ENTRIES, 1, DEFAULT_LIST_ENTRIES, warn),
    listBytes: positiveInteger(env, "ACP_GLM_LIST_FILES_LIMIT_BYTES", DEFAULT_LIST_BYTES, 128, Number.MAX_SAFE_INTEGER, warn),
    fsConcurrency: 16,
  };
}
