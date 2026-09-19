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
 *   ACP_GLM_MAX_TURNS - (optional) Max model/tool turns per prompt (default: 100)
 *   ACP_GLM_COMMAND_TIMEOUT_MS - (optional) run_command deadline in milliseconds (default: 120000)
 *   ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES - (optional) combined run_command stdout/stderr capture limit (default: 65536)
 */
import { startAgentRuntime } from "./protocol/connection.js";
import { parseMaxTurnsFlag } from "./cli-args.js";
import { runSetup } from "./setup.js";

const args = process.argv.slice(2);

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
      "  ACP_GLM_MAX_TURNS              Max model/tool turns per prompt (default 100)",
      "  ACP_GLM_COMMAND_TIMEOUT_MS     run_command deadline in milliseconds (default 120000)",
      "  ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES  Combined run_command stdout/stderr limit in bytes (default 65536)",
      "  ACP_GLM_AVAILABLE_MODELS       Comma-separated list of advertised models",
      "  ACP_GLM_BASE_URL               Override the Z.AI API base URL",
      "  ACP_GLM_MAX_TOKENS             Per-call max output tokens (default 32768)",
      "  ACP_GLM_THINKING               Force thinking mode (true / false)",
      "  ACP_GLM_STREAM_THINKING        Forward reasoning to the client as thought chunks (default true)",
      "  ACP_GLM_SESSION_DIR            Where to persist sessions (default: ~/.local/state/glm-acp-agent/sessions)",
      "  ACP_GLM_DEBUG                  Enable verbose stderr logging (true or 1)",
      "  XDG_CONFIG_HOME                Where to read/write credentials.json (default: ~/.config)",
      "",
    ].join("\n")
  );
  process.exit(0);
} else {
  const runtime = startAgentRuntime({ maxTurns: parseMaxTurnsFlag(args) });
  let finishing: Promise<void> | null = null;
  let signalExitCode: number | null = null;

  const finish = (reason: "disconnect" | "sigterm" | "sigint" | "fatal", exitCode: number) => {
    if (reason === "sigterm" || reason === "sigint") signalExitCode = exitCode;
    if (finishing) return finishing;

    finishing = (async () => {
      const clean = await settlesWithin(runtime.shutdown(reason), 5_000);
      if (clean) {
        runtime.closeTransport();
        process.exitCode = signalExitCode ?? exitCode;
        return;
      }
      process.stderr.write("glm-acp-agent: shutdown deadline exceeded; forcing active command cleanup\n");
      await settlesWithin(runtime.forceShutdown(), 1_000);
      process.exitCode = signalExitCode ?? 1;
      // An unresolved client/MCP handle would otherwise keep the CLI alive
      // indefinitely after the documented deadline. Command cleanup had its
      // forced attempt above; report the incomplete shutdown as nonzero.
      process.exit(process.exitCode);
    })().catch((err: unknown) => {
      const message = err instanceof Error ? err.message : String(err);
      process.stderr.write(`Fatal shutdown error: ${message}\n`);
      process.exitCode = signalExitCode ?? 1;
    });
    return finishing;
  };

  void runtime.connection.closed
    .then(() => finish("disconnect", 0))
    .catch(() => finish("fatal", 1));
  process.on("SIGINT", () => { void finish("sigint", 130); });
  process.on("SIGTERM", () => { void finish("sigterm", 143); });
}

async function settlesWithin(promise: Promise<unknown>, timeoutMs: number): Promise<boolean> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise.then(() => true, () => false),
      new Promise<false>((resolve) => { timer = setTimeout(() => resolve(false), timeoutMs); }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
