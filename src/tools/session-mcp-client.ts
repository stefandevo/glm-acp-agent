import { spawn as nodeSpawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { McpServer, McpServerHttp, McpServerStdio } from "@agentclientprotocol/sdk";
import { TOOL_DEFINITIONS, type ToolDefinition } from "./definitions.js";

const MCP_PROTOCOL_VERSION = "2025-06-18";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id?: number;
  method: string;
  params?: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc?: string;
  id?: number;
  result?: unknown;
  error?: {
    code?: string | number;
    message?: string;
    [key: string]: unknown;
  };
}

interface McpTool {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

interface ToolBinding {
  exposedName: string;
  sourceName: string;
  client: ConnectedMcpClient;
  definition: ToolDefinition;
}

interface ConnectedMcpClient {
  listTools(): Promise<McpTool[]>;
  callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
  dispose(): Promise<void>;
}

export class SessionMcpTools {
  private bindings = new Map<string, ToolBinding>();

  constructor(bindings: ToolBinding[]) {
    for (const binding of bindings) {
      this.bindings.set(binding.exposedName, binding);
    }
  }

  get toolDefinitions(): ToolDefinition[] {
    return Array.from(this.bindings.values()).map((binding) => binding.definition);
  }

  get toolNames(): string[] {
    return Array.from(this.bindings.keys());
  }

  hasTool(name: string): boolean {
    return this.bindings.has(name);
  }

  async callTool(
    exposedName: string,
    args: Record<string, unknown>,
    signal?: AbortSignal
  ): Promise<unknown> {
    const binding = this.bindings.get(exposedName);
    if (!binding) throw new Error(`Unknown MCP tool: ${exposedName}`);
    return binding.client.callTool(binding.sourceName, args, signal);
  }

  async dispose(): Promise<void> {
    const clients = new Set(Array.from(this.bindings.values()).map((binding) => binding.client));
    await Promise.all(Array.from(clients).map((client) => client.dispose().catch(() => undefined)));
    this.bindings.clear();
  }
}

export async function connectSessionMcpServers(
  servers: ReadonlyArray<McpServer>
): Promise<SessionMcpTools> {
  const usedNames = new Set(TOOL_DEFINITIONS.map((tool) => tool.function.name));
  const bindings: ToolBinding[] = [];
  const clients: ConnectedMcpClient[] = [];

  try {
    for (const server of servers) {
      const client = createClient(server);
      clients.push(client);
      const tools = await client.listTools();
      for (const tool of tools) {
        const exposedName = chooseToolName(tool.name, server.name, usedNames);
        usedNames.add(exposedName);
        bindings.push({
          exposedName,
          sourceName: tool.name,
          client,
          definition: {
            type: "function",
            function: {
              name: exposedName,
              description: tool.description ?? `Call ${tool.name} on the ${server.name} MCP server.`,
              parameters: normalizeSchema(tool.inputSchema),
            },
          },
        });
      }
    }
  } catch (err) {
    await Promise.all(clients.map((client) => client.dispose().catch(() => undefined)));
    throw err;
  }

  return new SessionMcpTools(bindings);
}

function createClient(server: McpServer): ConnectedMcpClient {
  if ("type" in server && server.type === "http") {
    return new HttpMcpClient(server);
  }
  if ("type" in server && server.type === "sse") {
    throw new Error(`MCP server "${server.name}" uses SSE transport, which is not supported yet.`);
  }
  return new StdioMcpClient(server);
}

class HttpMcpClient implements ConnectedMcpClient {
  private nextId = 1;
  private initialized: Promise<void> | null = null;
  private mcpSessionId: string | undefined;

  constructor(private server: McpServerHttp & { type: "http" }) {}

  async listTools(): Promise<McpTool[]> {
    await this.ensureInitialized();
    const result = await this.request("tools/list", {}, "tools/list");
    return extractTools(result);
  }

  async callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    await this.ensureInitialized();
    return this.request("tools/call", { name, arguments: args }, "tools/call", signal, name);
  }

  async dispose(): Promise<void> {
    this.initialized = null;
    this.mcpSessionId = undefined;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return this.initialized;
    this.initialized = this.initialize();
    try {
      await this.initialized;
    } catch (err) {
      this.initialized = null;
      throw err;
    }
  }

  private async initialize(): Promise<void> {
    const response = await this.fetchJsonRpc(
      "initialize",
      {
        jsonrpc: "2.0",
        id: this.nextId++,
        method: "initialize",
        params: {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: { name: "glm-acp-agent", version: "1.0.0" },
        },
      },
      "initialize"
    );
    this.mcpSessionId = response.sessionId;
    await this.sendNotification({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    });
  }

  private async request(
    method: string,
    params: Record<string, unknown>,
    stage: string,
    signal?: AbortSignal,
    mcpName?: string
  ): Promise<unknown> {
    const response = await this.fetchJsonRpc(
      method,
      {
        jsonrpc: "2.0",
        id: this.nextId++,
        method,
        params,
      },
      stage,
      signal,
      mcpName
    );
    return response.body.result;
  }

  private async sendNotification(body: JsonRpcRequest): Promise<void> {
    const response = await fetch(this.server.url, {
      method: "POST",
      headers: this.headers("notifications/initialized"),
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`MCP ${this.server.name} notifications/initialized failed: HTTP ${response.status}: ${await response.text()}`);
    }
  }

  private async fetchJsonRpc(
    mcpMethod: string,
    body: JsonRpcRequest,
    stage: string,
    signal?: AbortSignal,
    mcpName?: string
  ): Promise<{ body: JsonRpcResponse; sessionId?: string }> {
    const response = await fetch(this.server.url, {
      method: "POST",
      headers: this.headers(mcpMethod, mcpName),
      body: JSON.stringify(body),
      signal,
    });
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`MCP ${this.server.name} ${stage} failed: HTTP ${response.status}: ${text}`);
    }
    const parsed = parseMcpResponse(text, response.headers.get("Content-Type") ?? "");
    if (parsed.error) {
      throw new Error(`MCP ${this.server.name} ${stage} failed: ${JSON.stringify(parsed.error)}`);
    }
    return {
      body: parsed,
      sessionId: response.headers.get("MCP-Session-Id") ?? undefined,
    };
  }

  private headers(mcpMethod: string, mcpName?: string): Headers {
    const headers = new Headers();
    for (const header of this.server.headers) {
      headers.set(header.name, header.value);
    }
    headers.set("Accept", "application/json, text/event-stream");
    headers.set("Content-Type", "application/json");
    headers.set("MCP-Protocol-Version", MCP_PROTOCOL_VERSION);
    headers.set("Mcp-Method", mcpMethod);
    if (this.mcpSessionId) headers.set("MCP-Session-Id", this.mcpSessionId);
    if (mcpName) headers.set("Mcp-Name", mcpName);
    return headers;
  }
}

class StdioMcpClient implements ConnectedMcpClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private initialized: Promise<void> | null = null;
  private nextId = 1;
  private pending = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (reason: Error) => void;
      label: string;
    }
  >();
  private buffer = "";
  private exited = false;
  private exitReason: string | null = null;

  constructor(private server: McpServerStdio) {}

  async listTools(): Promise<McpTool[]> {
    await this.ensureInitialized();
    return extractTools(await this.request("tools/list", {}, "tools/list"));
  }

  async callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    await this.ensureInitialized();
    return this.request("tools/call", { name, arguments: args }, `tools/call ${name}`, signal);
  }

  async dispose(): Promise<void> {
    if (this.child && !this.exited) {
      try {
        this.child.kill();
      } catch {
        // ignore
      }
    }
    this.child = null;
    this.initialized = null;
    for (const [, pending] of this.pending) {
      pending.reject(new Error("MCP client disposed"));
    }
    this.pending.clear();
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return this.initialized;
    this.initialized = this.startAndInitialize();
    try {
      await this.initialized;
    } catch (err) {
      this.initialized = null;
      throw err;
    }
  }

  private async startAndInitialize(): Promise<void> {
    try {
      this.child = nodeSpawn(this.server.command, this.server.args, {
        env: buildStdioEnv(this.server),
      });
    } catch (err) {
      throw new Error(`MCP ${this.server.name} startup failed: ${(err as Error).message}`);
    }
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.handleStdout(chunk));
    this.child.on("exit", (code, sig) => {
      this.exited = true;
      this.exitReason = `exit code=${code} signal=${sig ?? "(none)"}`;
      for (const [, pending] of this.pending) {
        pending.reject(new Error(`server exited (${this.exitReason}).`));
      }
      this.pending.clear();
    });
    this.child.on("error", (err) => {
      this.exited = true;
      this.exitReason = err.message;
    });

    await this.request(
      "initialize",
      {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: "glm-acp-agent", version: "1.0.0" },
      },
      "initialize"
    );
    this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
  }

  private request(
    method: string,
    params: Record<string, unknown>,
    label: string,
    signal?: AbortSignal
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, {
        label,
        resolve,
        reject: (err) => reject(new Error(`MCP ${this.server.name} ${label} failed: ${err.message}`)),
      });
      const onAbort = () => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        pending.reject(new Error("aborted"));
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        this.send({ jsonrpc: "2.0", id, method, params });
      } catch (err) {
        this.pending.delete(id);
        reject(new Error(`MCP ${this.server.name} ${label} failed: ${(err as Error).message}`));
      }
    });
  }

  private send(message: Record<string, unknown>): void {
    if (!this.child || this.exited) {
      throw new Error(`MCP server is not running${this.exitReason ? ` (${this.exitReason})` : ""}.`);
    }
    this.child.stdin.write(JSON.stringify(message) + "\n");
  }

  private handleStdout(chunk: string): void {
    this.buffer += chunk;
    let idx: number;
    while ((idx = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, idx).trim();
      this.buffer = this.buffer.slice(idx + 1);
      if (!line) continue;
      let parsed: JsonRpcResponse;
      try {
        parsed = JSON.parse(line) as JsonRpcResponse;
      } catch {
        continue;
      }
      if (typeof parsed.id !== "number") continue;
      const pending = this.pending.get(parsed.id);
      if (!pending) continue;
      this.pending.delete(parsed.id);
      if (parsed.error) {
        pending.reject(new Error(parsed.error.message ?? `code ${parsed.error.code ?? "?"}`));
      } else {
        pending.resolve(parsed.result);
      }
    }
  }
}

function buildStdioEnv(server: McpServerStdio): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const entry of server.env) {
    env[entry.name] = entry.value;
  }
  return env;
}

function extractTools(result: unknown): McpTool[] {
  if (!isRecord(result)) return [];
  const tools = result["tools"];
  if (!Array.isArray(tools)) return [];
  return tools
    .filter((tool): tool is Record<string, unknown> => isRecord(tool) && typeof tool["name"] === "string")
    .map((tool) => ({
      name: tool["name"] as string,
      description: typeof tool["description"] === "string" ? tool["description"] : undefined,
      inputSchema: isRecord(tool["inputSchema"]) ? tool["inputSchema"] : undefined,
    }));
}

function chooseToolName(sourceName: string, serverName: string, usedNames: Set<string>): string {
  const safeSource = sanitizeToolName(sourceName) || "tool";
  if (!usedNames.has(safeSource)) return safeSource;
  const safeServer = sanitizeToolName(serverName) || "mcp";
  const base = `${safeServer}_${safeSource}`.slice(0, 60);
  let candidate = base;
  let suffix = 2;
  while (usedNames.has(candidate)) {
    candidate = `${base}_${suffix++}`.slice(0, 64);
  }
  return candidate;
}

function sanitizeToolName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^_+|_+$/g, "").slice(0, 64);
}

function normalizeSchema(schema: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!schema) return { type: "object", properties: {} };
  return schema;
}

function parseMcpResponse(text: string, contentType: string): JsonRpcResponse {
  if (!text.trim()) {
    throw new Error("MCP response was empty.");
  }
  if (contentType.toLowerCase().includes("text/event-stream")) {
    return parseSseJsonRpc(text);
  }
  return JSON.parse(text) as JsonRpcResponse;
}

function parseSseJsonRpc(text: string): JsonRpcResponse {
  const dataLines: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  for (const data of dataLines) {
    if (!data || data === "[DONE]") continue;
    const parsed = JSON.parse(data) as JsonRpcResponse;
    if (parsed.result !== undefined || parsed.error !== undefined) return parsed;
  }
  throw new Error("MCP SSE response did not contain a JSON-RPC result.");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
