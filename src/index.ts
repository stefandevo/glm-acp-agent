#!/usr/bin/env node
/**
 * GLM ACP Agent entry point.
 *
 * Starts an ACP agent over stdio that uses the Zhipu AI GLM model family
 * as its reasoning core. Pass `--setup` instead of starting the protocol
 * loop to interactively store a Z.AI API key on disk.
 *
 * Environment variables:
 *   Z_AI_API_KEY      - API key for the Z.AI / Zhipu AI service. If unset,
 *                       falls back to the credentials file written by --setup.
 *   ACP_GLM_MODEL     - (optional) Override the default model (default: glm-5.3)
 *   ACP_GLM_MAX_TURNS - (optional) Max model/tool turns per prompt (default: 20)
 */
import { startConnection } from "./protocol/connection.js";
import { runSetup } from "./setup.js";

const args = process.argv.slice(2);

/** Parse `--max-turns <n>` (or `--max-turns=<n>`) from the CLI args. */
function parseMaxTurnsFlag(argv: string[]): number | undefined {
  const idx = argv.indexOf("--max-turns");
  if (idx !== -1) {
    if (idx + 1 >= argv.length) {
      process.stderr.write("glm-acp-agent: --max-turns requires a value\n");
      return undefined;
    }
    const parsed = Number(argv[idx + 1]);
    if (Number.isFinite(parsed) && parsed > 0) return Math.floor(parsed);
    process.stderr.write(
      `glm-acp-agent: ignoring invalid --max-turns "${argv[idx + 1]}"\n`
    );
    return undefined;
  }
  const eq = argv.find((a) => a.startsWith("--max-turns="));
  if (eq !== undefined) {
    const parsed = Number(eq.slice("--max-turns=".length));
    if (Number.isFinite(parsed) && parsed > 0) return Math.floor(parsed);
    process.stderr.write(
      `glm-acp-agent: ignoring invalid --max-turns "${eq.slice("--max-turns=".length)}"\n`
    );
  }
  return undefined;
}

if (args.includes("--setup")) {
  runSetup()
    .then(() => process.exit(0))
    .catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      process.stderr.write(`Setup failed: ${message}\n`);
      process.exit(1);
    });
} else if (args.includes("--help") || args.includes("-h")) {
  process.stdout.write(
    [
      "glm-acp-agent — ACP agent using Zhipu AI's GLM models",
      "",
      "Usage:",
      "  glm-acp-agent           Start the ACP stdio loop (run by an ACP client)",
      "  glm-acp-agent --setup   Interactively store your Z.AI API key on disk",
      "  glm-acp-agent --max-turns <n>  Max model/tool turns per prompt",
      "  glm-acp-agent --help    Show this message",
      "",
      "Environment variables:",
      "  Z_AI_API_KEY                   API key (overrides the stored credentials)",
      "  ACP_GLM_MODEL                  Default model id (e.g. glm-5.3)",
      "  ACP_GLM_MAX_TURNS              Max model/tool turns per prompt (default 20)",
      "  ACP_GLM_AVAILABLE_MODELS       Comma-separated list of advertised models",
      "  ACP_GLM_BASE_URL               Override the Z.AI API base URL",
      "  ACP_GLM_MAX_TOKENS             Per-call max output tokens (default 8192)",
      "  ACP_GLM_THINKING               Force thinking mode (true / false)",
      "  ACP_GLM_SESSION_DIR            Where to persist sessions (default: ~/.local/state/glm-acp-agent/sessions)",
      "  ACP_GLM_DEBUG                  Enable verbose stderr logging (true or 1)",
      "  XDG_CONFIG_HOME                Where to read/write credentials.json (default: ~/.config)",
      "",
    ].join("\n")
  );
  process.exit(0);
} else {
  const connection = startConnection({ maxTurns: parseMaxTurnsFlag(args) });

  // Keep the process alive until the connection closes
  connection.closed
    .then(() => {
      process.exit(0);
    })
    .catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      process.stderr.write(`Fatal error: ${message}\n`);
      process.exit(1);
    });
}
