import { Readable, Writable } from "node:stream";
import { AgentSideConnection, ndJsonStream } from "@agentclientprotocol/sdk";
import { GlmAcpAgent } from "./agent.js";

/**
 * Sets up the ACP stdio connection and starts the agent.
 *
 * Uses `ndJsonStream` for newline-delimited JSON transport over stdin/stdout,
 * as specified in the ACP SDK documentation.
 */
export function startConnection(): AgentSideConnection {
  // Convert Node.js streams to Web Streams API
  const output = Writable.toWeb(process.stdout) as WritableStream<Uint8Array>;
  const input = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;

  const stream = ndJsonStream(output, input);

  const connection = new AgentSideConnection(
    (conn) => new GlmAcpAgent(conn),
    stream
  );

  return connection;
}
