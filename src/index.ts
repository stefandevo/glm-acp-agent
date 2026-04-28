#!/usr/bin/env node
/**
 * GLM ACP Agent entry point.
 *
 * Starts an ACP agent over stdio that uses the Zhipu AI GLM model family
 * as its reasoning core.
 *
 * Environment variables:
 *   Z_AI_API_KEY      - (required) API key for the Z.AI / Zhipu AI service
 *   ACP_GLM_MODEL     - (optional) Override the default model (default: glm-5-1)
 */
import { startConnection } from "./protocol/connection.js";

const connection = startConnection();

// Keep the process alive until the connection closes
connection.closed.then(() => {
  process.exit(0);
}).catch((err: unknown) => {
  const message = err instanceof Error ? err.message : String(err);
  process.stderr.write(`Fatal error: ${message}\n`);
  process.exit(1);
});
