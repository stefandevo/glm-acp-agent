import { spawn as nodeSpawn, spawnSync as nodeSpawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { McpServer, McpServerHttp, McpServerStdio } from "@agentclientprotocol/sdk";
import { TOOL_DEFINITIONS, type ToolDefinition } from "./definitions.js";
import {
  collectToolPages,
  assertValidToolPage,
  DEFAULT_MCP_MAX_PAGES,
  DEFAULT_MCP_MAX_SCHEMA_BYTES,
  DEFAULT_MCP_MAX_TOOLS,
} from "./mcp-pagination.js";

const MCP_PROTOCOL_VERSION = "2025-06-18";
/** Generous: a cold `npx -y` fetch on Windows Defender can take well over a minute. */
const DEFAULT_INITIALIZATION_TIMEOUT_MS = 120_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const STDERR_TAIL_LIMIT = 16_384;
const STDERR_MESSAGE_LIMIT = 2_000;
/** Extensionless launchers that resolve to a `.cmd` shim on Windows, which Node cannot spawn directly. */
const WINDOWS_SHIM_COMMANDS = new Set(["npx", "npm", "pnpm", "yarn", "bunx"]);
/** Characters cmd.exe treats specially; any of them in a client-supplied token is a launch injection risk. */
const CMD_METACHARACTERS = /[&|<>^"%!\r\n\0]/;
const SECRET_ENV_NAME = /key|token|secret|password|passwd|pwd|credential|auth|cookie/i;
/** Exact names that match SECRET_ENV_NAME but are never credentials — exempt these, not the substring. */
const NON_SECRET_ENV_NAMES = new Set(["PWD", "OLDPWD"]);
const CHILD_TERM_GRACE_MS = 250;
const CHILD_KILL_SETTLE_MS = 500;
const MAX_CONCURRENT_MCP_SERVER_STARTUPS = 3;

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

export interface ToolBinding {
  exposedName: string;
  sourceName: string;
  client: ConnectedMcpClient;
  definition: ToolDefinition;
}

export interface ConnectedMcpClient {
  listTools(signal?: AbortSignal): Promise<McpTool[]>;
  callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
  dispose(): Promise<void>;
}

export class SessionMcpTools {
  private bindings = new Map<string, ToolBinding>();
  private clients: Set<ConnectedMcpClient>;
  private disposePromise: Promise<void> | null = null;

  constructor(bindings: ToolBinding[], clients: ConnectedMcpClient[] = []) {
    this.clients = new Set(clients);
    for (const binding of bindings) {
      this.bindings.set(binding.exposedName, binding);
      this.clients.add(binding.client);
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
    if (!this.disposePromise) {
      this.disposePromise = (async () => {
        const clients = Array.from(this.clients);
        const results = await Promise.allSettled(clients.map((client) => client.dispose()));
        this.clients.clear();
        this.bindings.clear();
        const failure = results.find((result): result is PromiseRejectedResult => result.status === "rejected");
        if (failure) throw failure.reason;
      })();
    }
    await this.disposePromise;
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function closeChildPipes(child: ChildProcessWithoutNullStreams): void {
  for (const stream of [child.stdin, child.stdout, child.stderr]) {
    try { stream.destroy(); } catch { /* already closed */ }
  }
}

export async function connectSessionMcpServers(
  servers: ReadonlyArray<McpServer>,
  signal?: AbortSignal
): Promise<SessionMcpTools> {
  const usedNames = new Set(TOOL_DEFINITIONS.map((tool) => tool.function.name));
  const bindings: ToolBinding[] = [];
  const clients: ConnectedMcpClient[] = [];
  const clientsByServer: Array<ConnectedMcpClient | undefined> = new Array(servers.length);
  const discoveredTools: Array<McpTool[] | undefined> = new Array(servers.length);
  let nextServerIndex = 0;
  let setupFailed = false;

  const connectNextServer = async (): Promise<void> => {
    while (!setupFailed) {
      const index = nextServerIndex++;
      if (index >= servers.length) return;
      const server = servers[index];
      try {
        if (signal?.aborted) throw new Error("MCP session setup cancelled");
        const client = createClient(server);
        clients.push(client);
        clientsByServer[index] = client;
        const tools = await client.listTools(signal);
        assertUniqueSourceNames(tools, server.name);
        discoveredTools[index] = tools;
      } catch (err) {
        setupFailed = true;
        throw err;
      }
    }
  };

  try {
    const workerCount = Math.min(MAX_CONCURRENT_MCP_SERVER_STARTUPS, servers.length);
    await Promise.all(Array.from({ length: workerCount }, () => connectNextServer()));
    for (let index = 0; index < servers.length; index += 1) {
      const server = servers[index];
      const serverClient = clientsByServer[index];
      const tools = discoveredTools[index];
      if (!serverClient || !tools) throw new Error(`MCP server "${server.name}" did not return a tool catalog`);
      for (const tool of tools) {
        const exposedName = chooseToolName(tool.name, server.name, usedNames);
        usedNames.add(exposedName);
        bindings.push({
          exposedName,
          sourceName: tool.name,
          client: serverClient,
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

  return new SessionMcpTools(bindings, clients);
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

export class HttpMcpClient implements ConnectedMcpClient {
  private nextId = 1;
  private initialized: Promise<void> | null = null;
  private initializationAbortController: AbortController | null = null;
  private initializationWaiters = 0;
  private mcpSessionId: string | undefined;
  private activeRequests = new Set<AbortController>();
  private disposed = false;

  constructor(private server: McpServerHttp & { type: "http" }) {}

  async listTools(signal?: AbortSignal): Promise<McpTool[]> {
    await this.awaitInitialization(signal);
    return collectToolPages({
      signal: signal ?? new AbortController().signal,
      maxPages: DEFAULT_MCP_MAX_PAGES,
      maxTools: DEFAULT_MCP_MAX_TOOLS,
      maxSchemaBytes: DEFAULT_MCP_MAX_SCHEMA_BYTES,
      requestPage: async (cursor, pageSignal) => {
        const result = await this.request(
          "tools/list",
          cursor === undefined ? {} : { cursor },
          "tools/list",
          pageSignal,
        );
        return extractToolPage(result);
      },
    });
  }

  async callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    await this.awaitInitialization(signal);
    return this.request("tools/call", { name, arguments: args }, "tools/call", signal, name);
  }

  async dispose(): Promise<void> {
    const sessionId = this.mcpSessionId;
    this.disposed = true;
    for (const controller of this.activeRequests) controller.abort();
    this.initialized = null;
    this.initializationAbortController = null;
    this.mcpSessionId = undefined;
    if (!sessionId) return;

    const controller = new AbortController();
    let resolveTimeout: (() => void) | undefined;
    const timeout = new Promise<void>((resolve) => { resolveTimeout = resolve; });
    const timer = setTimeout(() => {
      controller.abort();
      resolveTimeout?.();
    }, 1_000);
    try {
      const request = fetch(this.server.url, {
        method: "DELETE",
        headers: this.headers("DELETE", undefined, sessionId),
        signal: controller.signal,
      }).catch(() => undefined);
      await Promise.race([request, timeout]);
    } catch {
      // A server may not support DELETE, and shutdown must remain bounded.
    } finally {
      clearTimeout(timer);
      // The DELETE response body is never consumed. Aborting here also closes
      // it when the fetch settled before the deadline, so an idle-but-open
      // stream cannot hold socket resources after disposal returns.
      controller.abort();
    }
  }

  private async awaitInitialization(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) throw new Error(`MCP ${this.server.name} call cancelled`);
    const initialization = this.ensureInitialized();
    const initializationController = this.initializationAbortController;
    const waiting = this.initializationAbortController !== null;
    if (waiting) this.initializationWaiters += 1;
    const abortSharedInitialization = () => {
      if (
        !waiting
        || this.initializationWaiters !== 1
        || this.initialized !== initialization
        || this.initializationAbortController !== initializationController
        || !initializationController
      ) return;
      // Clear the identity before aborting the transport. A new caller can
      // arrive synchronously from the old caller's abort path and must not
      // attach to this rejected initialization promise.
      this.initialized = null;
      this.initializationAbortController = null;
      this.mcpSessionId = undefined;
      initializationController.abort();
    };
    try {
      await waitForAbort(
        initialization,
        signal,
        `MCP ${this.server.name} call cancelled`,
        abortSharedInitialization,
      );
    } catch (err) {
      // One aborted waiter must not interrupt a handshake another caller needs.
      // The abort callback handles the synchronous case; retry here after a
      // concurrent waiter may have finished between the callback and catch.
      if (signal?.aborted) abortSharedInitialization();
      throw err;
    } finally {
      if (waiting) this.initializationWaiters -= 1;
    }
  }

  private ensureInitialized(): Promise<void> {
    if (this.disposed) throw new Error(`MCP ${this.server.name} client disposed`);
    if (this.initialized) return this.initialized;
    const controller = new AbortController();
    this.initializationAbortController = controller;
    const initialization = this.initialize(controller.signal);
    this.initialized = initialization;
    void initialization.then(
      () => {
        if (this.initialized === initialization) this.initializationAbortController = null;
      },
      () => {
        if (this.initialized === initialization) {
          this.initialized = null;
          this.initializationAbortController = null;
          this.mcpSessionId = undefined;
        }
      },
    );
    return initialization;
  }

  private async initialize(signal: AbortSignal): Promise<void> {
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
      "initialize",
      signal
    );
    this.mcpSessionId = response.sessionId;
    await this.sendNotification({
      jsonrpc: "2.0",
      method: "notifications/initialized",
    }, signal);
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

  private async sendNotification(body: JsonRpcRequest, signal?: AbortSignal): Promise<void> {
    await this.fetchWithLifecycle({
      method: "POST",
      headers: this.headers("notifications/initialized"),
      body: JSON.stringify(body),
    }, async response => {
      if (!response.ok) {
        throw new Error(`MCP ${this.server.name} notifications/initialized failed: HTTP ${response.status}: ${await response.text()}`);
      }
      await response.body?.cancel();
    }, signal);
  }

  private async fetchJsonRpc(
    mcpMethod: string,
    body: JsonRpcRequest,
    stage: string,
    signal?: AbortSignal,
    mcpName?: string
  ): Promise<{ body: JsonRpcResponse; sessionId?: string }> {
    return this.fetchWithLifecycle({
      method: "POST",
      headers: this.headers(mcpMethod, mcpName),
      body: JSON.stringify(body),
    }, async response => {
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
    }, signal);
  }

  private async fetchWithLifecycle<T>(
    init: RequestInit, consume: (response: Response) => Promise<T>, signal?: AbortSignal,
  ): Promise<T> {
    if (this.disposed) throw new Error(`MCP ${this.server.name} client disposed`);
    const controller = new AbortController();
    this.activeRequests.add(controller);
    const onAbort = () => controller.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) controller.abort();
    try {
      const response = await fetch(this.server.url, { ...init, signal: controller.signal });
      // Keep cancellation and disposal ownership until body consumption ends.
      return await consume(response);
    } finally {
      signal?.removeEventListener("abort", onAbort);
      this.activeRequests.delete(controller);
    }
  }

  private headers(mcpMethod: string, mcpName?: string, sessionId = this.mcpSessionId): Headers {
    const headers = new Headers();
    for (const header of this.server.headers) {
      headers.set(header.name, header.value);
    }
    headers.set("Accept", "application/json, text/event-stream");
    headers.set("Content-Type", "application/json");
    headers.set("MCP-Protocol-Version", MCP_PROTOCOL_VERSION);
    headers.set("Mcp-Method", mcpMethod);
    if (sessionId) headers.set("MCP-Session-Id", sessionId);
    if (mcpName) headers.set("Mcp-Name", mcpName);
    return headers;
  }
}

export interface StdioMcpClientOptions {
  /** Maximum time for the MCP handshake (spawn + initialize). Generous by default for cold `npx -y` fetches. */
  initializationTimeoutMs?: number;
  /** Maximum time for an individual JSON-RPC request. */
  requestTimeoutMs?: number;
  /** Platform override for tests. */
  platform?: NodeJS.Platform;
  /** Windows command interpreter override for tests. */
  comSpec?: string;
  /** Windows process-tree terminator override for tests. */
  killProcessTree?: (pid: number) => boolean;
  /** Override the spawn function for tests. */
  spawn?: (
    command: string,
    args: string[],
    options: {
      env: NodeJS.ProcessEnv;
      windowsHide?: boolean;
      windowsVerbatimArguments?: boolean;
    }
  ) => ChildProcessWithoutNullStreams;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
}

export class StdioMcpClient implements ConnectedMcpClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private initialized: Promise<void> | null = null;
  private initializingChild: ChildProcessWithoutNullStreams | null = null;
  private initializationWaiters = 0;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private buffer = "";
  private stderrTail = "";
  private exited = false;
  private exitReason: string | null = null;
  private disposed = false;
  private secrets: string[] = [];
  private readonly killProcessTree: (pid: number) => boolean;

  constructor(
    private server: McpServerStdio,
    private opts: StdioMcpClientOptions = {}
  ) {
    this.killProcessTree = opts.killProcessTree ?? taskkillTree;
  }

  async listTools(signal?: AbortSignal): Promise<McpTool[]> {
    // Counted as an initialization waiter too, so a concurrent callTool abort cannot
    // tear down the handshake this call is still waiting on.
    await this.awaitInitialization(signal);
    return collectToolPages({
      signal: signal ?? new AbortController().signal,
      maxPages: DEFAULT_MCP_MAX_PAGES,
      maxTools: DEFAULT_MCP_MAX_TOOLS,
      maxSchemaBytes: DEFAULT_MCP_MAX_SCHEMA_BYTES,
      requestPage: async (cursor, pageSignal) => extractToolPage(
        await this.request(
          "tools/list",
          cursor === undefined ? {} : { cursor },
          "tools/list",
          pageSignal,
        )
      ),
    });
  }

  async callTool(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    if (signal?.aborted) throw new Error(`MCP ${this.server.name} call cancelled`);
    await this.awaitInitialization(signal);
    return this.request("tools/call", { name, arguments: args }, `tools/call ${name}`, signal);
  }

  async dispose(): Promise<void> {
    const child = this.child;
    this.disposed = true;
    this.child = null;
    this.initialized = null;
    this.initializingChild = null;
    this.exited = true;
    this.exitReason = "client disposed";
    if (child) await this.terminateChildAndWait(child);
    this.rejectAllPending(new Error("cancelled (client disposed)"));
  }

  /**
   * Wait for the shared handshake without letting one caller's abort tear it down for the others:
   * the child is only killed when the aborting caller is the last one still waiting on it.
   */
  private async awaitInitialization(signal?: AbortSignal): Promise<void> {
    const initialization = this.ensureInitialized();
    const waiting = this.initializingChild !== null;
    if (waiting) this.initializationWaiters += 1;
    try {
      await waitForAbort(initialization, signal, `MCP ${this.server.name} call cancelled`);
    } catch (err) {
      const child = this.initializingChild;
      if (signal?.aborted && waiting && this.initializationWaiters === 1 && child) {
        this.failConnection(new Error("initialization aborted"), child, true);
      }
      throw err;
    } finally {
      if (waiting) this.initializationWaiters -= 1;
    }
  }

  private ensureInitialized(): Promise<void> {
    if (this.disposed) throw new Error(`MCP ${this.server.name} client disposed`);
    if (this.initialized && this.child && !this.exited) return this.initialized;
    const initialization = this.startAndInitialize();
    this.initialized = initialization;
    void initialization.catch(() => {
      if (this.initialized === initialization) this.initialized = null;
    });
    return initialization;
  }

  private async startAndInitialize(): Promise<void> {
    const { command, args, spawnOptions } = this.resolveLaunch();
    const spawnFn = this.opts.spawn ?? nodeSpawn;
    // Redact against the env the child actually gets — it inherits process.env, so a
    // credential the parent holds can surface in the child's stderr.
    const env = buildStdioEnv(this.server);
    this.secrets = collectSecretEnvValues(env);
    this.exited = false;
    this.exitReason = null;
    this.buffer = "";
    this.stderrTail = "";

    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawnFn(command, args, { env, windowsHide: true, ...spawnOptions });
    } catch (err) {
      throw this.launchError(err as NodeJS.ErrnoException);
    }
    this.child = child;
    this.initializingChild = child;

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      if (this.child === child) this.handleStdout(chunk);
    });
    // Drain stderr even when we never read it: an undrained pipe eventually blocks the child.
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      if (this.child === child) this.handleStderr(chunk);
    });
    child.on("exit", (code, sig) => {
      this.failConnection(new Error(`server exited (exit code=${code} signal=${sig ?? "(none)"}).`), child, false);
    });
    child.on("error", (err) => {
      this.failConnection(this.launchError(err as NodeJS.ErrnoException), child, true);
    });

    const deadline = Date.now() + (this.opts.initializationTimeoutMs ?? DEFAULT_INITIALIZATION_TIMEOUT_MS);
    try {
      await this.request(
        "initialize",
        {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: { name: "glm-acp-agent", version: "1.0.0" },
        },
        "initialize",
        undefined,
        Math.max(1, deadline - Date.now())
      );
      this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
      if (this.initializingChild === child) this.initializingChild = null;
    } catch (err) {
      this.failConnection(err instanceof Error ? err : new Error(String(err)), child, true);
      throw err;
    }
  }

  /**
   * Windows cannot `spawn` a `.cmd`/`.bat` shim directly (bare `npx` yields async ENOENT,
   * `npx.cmd` yields EINVAL), so those go through `cmd.exe /d /s /c`. Because the command and
   * args are client-supplied, every token routed through cmd.exe is validated first — we reject
   * rather than try to escape.
   *
   * cmd.exe strips the first and last quote after `/s /c`, so passing the command as its own
   * argv entry breaks a spaced path Node quoted for it (e.g. `C:\Program Files\nodejs\npx.cmd`
   * becomes `'C:\Program' is not recognized`). The whole line is therefore built here —
   * whitespace tokens quoted, one outer quote pair for cmd.exe to strip — and passed verbatim
   * so Node does not re-escape it. Tokens are metacharacter-free, so this stays safe.
   */
  private resolveLaunch(): {
    command: string;
    args: string[];
    spawnOptions?: { windowsVerbatimArguments: boolean };
  } {
    const platform = this.opts.platform ?? process.platform;
    const command = this.server.command;
    const args = [...this.server.args];
    if (platform !== "win32" || !needsWindowsShim(command)) {
      return { command, args };
    }
    for (const token of [command, ...args]) {
      if (CMD_METACHARACTERS.test(token)) {
        throw new Error(
          `MCP ${this.server.name} startup failed: unsafe token for the cmd.exe launch of \`${command}\` ` +
            `(contains a shell metacharacter): ${token}`
        );
      }
    }
    const comSpec = this.opts.comSpec ?? process.env["ComSpec"] ?? "cmd.exe";
    // Quote whitespace tokens (cmd.exe must keep them as one argv entry) and empty
    // tokens (an unquoted "" would vanish in join, shifting the child's argv).
    const line = [command, ...args]
      .map((token) => (token.length === 0 || /\s/.test(token) ? `"${token}"` : token))
      .join(" ");
    return {
      command: comSpec,
      args: ["/d", "/s", "/c", `"${line}"`],
      spawnOptions: { windowsVerbatimArguments: true },
    };
  }

  private launchError(err: NodeJS.ErrnoException): Error {
    if (err.code === "ENOENT") {
      return new Error(
        `MCP ${this.server.name} failed: could not launch \`${this.server.command}\`. ` +
          `Ensure it is installed and available on PATH.`,
        { cause: err }
      );
    }
    if (err.code === "EINVAL") {
      return new Error(
        `MCP ${this.server.name} failed: could not launch \`${this.server.command}\` (EINVAL). ` +
          `On Windows a .cmd/.bat launcher must be started through cmd.exe.`,
        { cause: err }
      );
    }
    return new Error(`MCP ${this.server.name} process error: ${err.message}`, { cause: err });
  }

  private request(
    method: string,
    params: Record<string, unknown>,
    label: string,
    signal?: AbortSignal,
    timeoutMs = this.opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      let settled = false;
      const onAbort = () => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        pending.reject(new Error("aborted"));
      };
      const cleanup = () => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      };
      const settle = (fn: (value: unknown) => void, value: unknown) => {
        if (settled) return;
        settled = true;
        cleanup();
        fn(value);
      };
      this.pending.set(id, {
        resolve: (value) => settle(resolve, value),
        reject: (err) =>
          settle(reject, new Error(`MCP ${this.server.name} ${label} failed: ${this.withStderr(err.message)}`)),
      });
      const timer = setTimeout(() => {
        const pending = this.pending.get(id);
        if (!pending) return;
        const timeout = new Error(`request timed out after ${timeoutMs}ms`);
        const child = this.child;
        if (child) {
          this.failConnection(timeout, child, true);
          return;
        }
        this.pending.delete(id);
        pending.reject(timeout);
      }, timeoutMs);
      signal?.addEventListener("abort", onAbort, { once: true });
      if (signal?.aborted) {
        onAbort();
        return;
      }
      try {
        this.send({ jsonrpc: "2.0", id, method, params });
      } catch (err) {
        const pending = this.pending.get(id);
        this.pending.delete(id);
        pending?.reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  private rejectAllPending(error: Error): void {
    const pending = [...this.pending.values()];
    this.pending.clear();
    for (const request of pending) request.reject(error);
  }

  /** Tear the connection down once and propagate the reason into every in-flight request. */
  private failConnection(error: Error, child: ChildProcessWithoutNullStreams, kill: boolean): void {
    if (this.child !== child || this.exited) return;
    this.child = null;
    if (this.initializingChild === child) this.initializingChild = null;
    this.exited = true;
    this.exitReason = error.message;
    this.initialized = null;
    this.rejectAllPending(error);
    if (kill) this.terminateChild(child);
  }

  private terminateChild(child: ChildProcessWithoutNullStreams): void {
    void this.terminateChildAndWait(child).catch(() => undefined);
  }

  private async terminateChildAndWait(child: ChildProcessWithoutNullStreams): Promise<void> {
    if (child.exitCode !== null) {
      closeChildPipes(child);
      return;
    }
    let settled = false;
    let resolveExit!: () => void;
    const exited = new Promise<void>((resolve) => { resolveExit = resolve; });
    const onExit = () => {
      if (settled) return;
      settled = true;
      resolveExit();
    };
    child.once("exit", onExit);
    child.once("close", onExit);
    child.once("error", onExit);
    closeChildPipes(child);
    if (this.sendChildSignal(child, "SIGTERM")) settled = true;
    await Promise.race([exited, delay(CHILD_TERM_GRACE_MS)]);
    if (!settled) {
      this.sendChildSignal(child, "SIGKILL");
      await Promise.race([exited, delay(CHILD_KILL_SETTLE_MS)]);
    }
    child.removeListener("exit", onExit);
    child.removeListener("close", onExit);
    child.removeListener("error", onExit);
    closeChildPipes(child);
    if (!settled) throw new Error("MCP child process did not exit after termination");
  }

  private sendChildSignal(child: ChildProcessWithoutNullStreams, signal: NodeJS.Signals): boolean {
    const platform = this.opts.platform ?? process.platform;
    // `child.kill()` only reaches cmd.exe, orphaning the npx -> node tree underneath it.
    if (platform === "win32" && child.pid && child.exitCode === null) {
      try {
        if (this.killProcessTree(child.pid)) return true;
      } catch {
        // fall back to the direct child below
      }
    }
    try {
      child.kill(signal);
    } catch {
      // ignore
    }
    return false;
  }

  private handleStderr(chunk: string): void {
    const tail = chunk.length >= STDERR_TAIL_LIMIT ? chunk.slice(-STDERR_TAIL_LIMIT) : this.stderrTail + chunk;
    this.stderrTail = tail.slice(-STDERR_TAIL_LIMIT);
  }

  private withStderr(message: string): string {
    let stderr = this.stderrTail.trim();
    if (!stderr) return message;
    for (const secret of this.secrets) stderr = stderr.split(secret).join("[REDACTED]");
    stderr = stderr.replace(/\s+/g, " ").slice(-STDERR_MESSAGE_LIMIT);
    return `${message}; stderr: ${stderr}`;
  }

  private send(message: Record<string, unknown>): void {
    if (!this.child || this.exited) {
      throw new Error(`MCP ${this.server.name} server is not running${this.exitReason ? ` (${this.exitReason})` : ""}.`);
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

function needsWindowsShim(command: string): boolean {
  const base = command.replace(/^.*[\\/]/, "").toLowerCase();
  return WINDOWS_SHIM_COMMANDS.has(base) || base.endsWith(".cmd") || base.endsWith(".bat");
}

/** Windows: kill the whole cmd.exe -> npx -> node tree, not just the interpreter we spawned. */
function taskkillTree(pid: number): boolean {
  return nodeSpawnSync("taskkill", ["/pid", String(pid), "/t", "/f"], {
    stdio: "ignore",
    windowsHide: true,
  }).status === 0;
}

function collectSecretEnvValues(env: NodeJS.ProcessEnv): string[] {
  const values = new Set<string>();
  for (const [name, value] of Object.entries(env)) {
    if (NON_SECRET_ENV_NAMES.has(name.toUpperCase())) continue;
    if (value && value.length >= 4 && SECRET_ENV_NAME.test(name)) values.add(value);
  }
  // Longest first, so a secret that contains another is replaced before its substring.
  return [...values].sort((a, b) => b.length - a.length);
}

function waitForAbort<T>(
  promise: Promise<T>,
  signal: AbortSignal | undefined,
  message: string,
  onAbortCleanup?: () => void,
): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new Error(message));
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const claim = (): boolean => {
      if (settled) return false;
      settled = true;
      signal.removeEventListener("abort", handleAbort);
      return true;
    };
    const handleAbort = () => {
      if (!claim()) return;
      onAbortCleanup?.();
      reject(new Error(message));
    };
    signal.addEventListener("abort", handleAbort, { once: true });
    promise.then(
      (value) => {
        if (claim()) resolve(value);
      },
      (error) => {
        if (claim()) reject(error);
      }
    );
  });
}

function buildStdioEnv(server: McpServerStdio): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const entry of server.env) {
    env[entry.name] = entry.value;
  }
  return env;
}

function extractToolPage(result: unknown): { tools: McpTool[]; nextCursor?: string | null } {
  assertValidToolPage(result);
  const tools = result["tools"];
  const nextCursor = result["nextCursor"];
  return {
    tools: Array.isArray(tools) ? tools
    .filter((tool): tool is Record<string, unknown> => isRecord(tool) && typeof tool["name"] === "string")
    .map((tool) => ({
      name: tool["name"] as string,
      description: typeof tool["description"] === "string" ? tool["description"] : undefined,
      inputSchema: isRecord(tool["inputSchema"]) ? tool["inputSchema"] : undefined,
    })) : [],
    nextCursor: nextCursor === null || nextCursor === undefined || typeof nextCursor === "string"
      ? nextCursor
      : nextCursor as never,
  };
}

function assertUniqueSourceNames(tools: McpTool[], serverName: string): void {
  const names = new Set<string>();
  for (const tool of tools) {
    if (names.has(tool.name)) {
      throw new Error(`MCP server "${serverName}" returned duplicate tool name "${tool.name}"`);
    }
    names.add(tool.name);
  }
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
