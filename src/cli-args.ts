import { DEFAULT_MAX_TURNS } from "./protocol/agent.js";

/**
 * Parse `--max-turns <n>` (or `--max-turns=<n>`) from CLI args.
 * Returns `undefined` only when the flag is absent; invalid input selects the
 * default (20), never `$ACP_GLM_MAX_TURNS` — explicit-but-bad CLI input must
 * not silently fall through to the env var.
 */
export function parseMaxTurnsFlag(argv: string[]): number | undefined {
  const idx = argv.indexOf("--max-turns");
  if (idx !== -1) {
    if (idx + 1 >= argv.length) {
      process.stderr.write("glm-acp-agent: --max-turns requires a value\n");
      return DEFAULT_MAX_TURNS;
    }
    const parsed = Number(argv[idx + 1]);
    if (Number.isFinite(parsed) && Math.floor(parsed) >= 1) {
      return Math.floor(parsed);
    }
    process.stderr.write(
      `glm-acp-agent: ignoring invalid --max-turns "${argv[idx + 1]}"\n`
    );
    return DEFAULT_MAX_TURNS;
  }
  const eq = argv.find((a) => a.startsWith("--max-turns="));
  if (eq !== undefined) {
    const raw = eq.slice("--max-turns=".length);
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && Math.floor(parsed) >= 1) {
      return Math.floor(parsed);
    }
    process.stderr.write(`glm-acp-agent: ignoring invalid --max-turns "${raw}"\n`);
    return DEFAULT_MAX_TURNS;
  }
  return undefined;
}
