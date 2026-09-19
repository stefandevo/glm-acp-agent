import { Readable, Writable } from "node:stream";
import { AgentSideConnection, ndJsonStream } from "@agentclientprotocol/sdk";
import { GlmAcpAgent, type GlmAcpAgentOptions } from "./agent.js";

/**
 * Sets up the ACP stdio connection and starts the agent.
 *
 * Uses `ndJsonStream` for newline-delimited JSON transport over stdin/stdout,
 * as specified in the ACP SDK documentation.
 */
export function startConnection(
  agentOptions: GlmAcpAgentOptions = {}
): AgentSideConnection {
  return startAgentRuntime(agentOptions).connection;
}

export interface AgentRuntime {
  connection: AgentSideConnection;
  shutdown: (reason: "disconnect" | "sigterm" | "sigint" | "fatal") => Promise<void>;
  forceShutdown: () => Promise<void>;
  closeTransport: () => void;
}

/**
 * Starts the stdio connection together with its agent-owned shutdown path.
 * `startConnection` remains available for embedders that only need the SDK
 * connection; the CLI uses this runtime to await cleanup before exiting.
 */
export function startAgentRuntime(
  agentOptions: GlmAcpAgentOptions = {}
): AgentRuntime {
  // Convert Node.js streams to Web Streams API
  const output = Writable.toWeb(process.stdout) as WritableStream<Uint8Array>;
  const input = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;

  const stream = ndJsonStream(output, input);

  let agent: GlmAcpAgent | null = null;
  const connection = new AgentSideConnection(
    (conn) => (agent = new GlmAcpAgent(conn, agentOptions)),
    stream
  );

  const shutdown = (reason: "disconnect" | "sigterm" | "sigint" | "fatal") =>
    agent?.shutdown(reason) ?? Promise.resolve();
  const forceShutdown = () => agent?.forceShutdown() ?? Promise.resolve();
  const closeTransport = () => {
    process.stdin.destroy();
  };

  // There is exactly one connection-close hook. `GlmAcpAgent.shutdown` is
  // memoized, so this composes safely with a simultaneous signal handler.
  void connection.closed.then(
    () => { void shutdown("disconnect").catch(() => undefined); },
    () => { void shutdown("fatal").catch(() => undefined); }
  );

  return { connection, shutdown, forceShutdown, closeTransport };
}
