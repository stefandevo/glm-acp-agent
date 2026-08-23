import { spawn as nodeSpawn, spawnSync as nodeSpawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { remapArguments, resolveToolName, type DiscoveredTool } from "./mcp-arg-remap.js";

const MCP_PROTOCOL_VERSION = "2025-06-18";
const DEFAULT_INITIALIZATION_TIMEOUT_MS = 30_000;
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;
const STDERR_TAIL_LIMIT = 16_384;

export interface VisionMcpClient {
  callTool(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
  dispose(): Promise<void>;
}

interface StdioVisionMcpClientOptions {
  apiKey: string;
  /** Override the package spec for tests/pinning. Defaults to `@z_ai/mcp-server@latest`. */
  packageSpec?: string;
  /** Maximum time for Vision MCP initialization and tool discovery. */
  initializationTimeoutMs?: number;
  /** Maximum time for an individual Vision MCP request. */
  requestTimeoutMs?: number;
  /** Platform override for tests. */
  platform?: NodeJS.Platform;
  /** Windows command interpreter override for tests. */
  comSpec?: string;
  /** Windows process-tree terminator override for tests. */
  killProcessTree?: (pid: number) => boolean;
  /** Override the spawn function for tests. */
  spawn?: (command: string, args: string[], options: { env: NodeJS.ProcessEnv; windowsHide?: boolean }) => ChildProcessWithoutNullStreams;
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  method: string;
}

export class StdioVisionMcpClient implements VisionMcpClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private initialized: Promise<void> | null = null;
  private initializingChild: ChildProcessWithoutNullStreams | null = null;
  private initializationWaiters = 0;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private buffer = "";
  private exited = false;
  private exitReason: string | null = null;
  private discoveredTools: DiscoveredTool[] = [];
  private stderrTail = "";

  constructor(private opts: StdioVisionMcpClientOptions) {}

  async callTool(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    if (signal?.aborted) throw new Error("Vision MCP call cancelled");
    const initialization = this.ensureInitialized();
    const waitingForInitialization = this.initializingChild !== null;
    if (waitingForInitialization) this.initializationWaiters += 1;
    try {
      await waitForAbort(initialization, signal, "Vision MCP call cancelled");
    } catch (err) {
      const child = this.initializingChild;
      if (signal?.aborted && waitingForInitialization && this.initializationWaiters === 1 && child) {
        this.failConnection(new Error("initialization aborted"), child, true);
      }
      throw err;
    } finally {
      if (waitingForInitialization) this.initializationWaiters -= 1;
    }
    return this.callToolInternal(toolName, args, signal);
  }

  private async callToolInternal(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    const { name: resolvedName, args: remappedArgs } = this.resolveAndRemap(toolName, args);
    try {
      return await this.request("tools/call", { name: resolvedName, arguments: remappedArgs }, `Vision MCP ${toolName}`, signal);
    } catch (err) {
      if (!isVisionRetryableError(err)) throw err;
      await this.rediscoverTools(signal);
      const { name: resolvedName2, args: remappedArgs2 } = this.resolveAndRemap(toolName, args);
      return this.request("tools/call", { name: resolvedName2, arguments: remappedArgs2 }, `Vision MCP ${toolName}`, signal);
    }
  }

  private resolveAndRemap(toolName: string, args: Record<string, unknown>): { name: string; args: Record<string, unknown> } {
    const toolNames = this.discoveredTools.map((t) => t.name);
    const resolvedName = resolveToolName(toolName, toolNames, "@z_ai/mcp-server");
    const toolSchema = this.discoveredTools.find((t) => t.name === resolvedName);
    return { name: resolvedName, args: remapArguments(args, toolSchema?.properties ?? []) };
  }

  private async rediscoverTools(signal?: AbortSignal, timeoutMs?: number): Promise<void> {
    const result = await this.request("tools/list", {}, "Vision MCP tools/list", signal, timeoutMs) as
      | { tools?: { name: string; inputSchema?: { properties?: Record<string, unknown> } }[] }
      | undefined;
    const tools = result?.tools ?? [];
    if (tools.length > 0) {
      this.discoveredTools = tools.map((t) => ({
        name: t.name,
        properties: t.inputSchema?.properties ? Object.keys(t.inputSchema.properties) : [],
      }));
    }
  }

  async dispose(): Promise<void> {
    const child = this.child;
    this.child = null;
    this.initialized = null;
    this.initializingChild = null;
    this.discoveredTools = [];
    this.exited = true;
    this.exitReason = "client disposed";
    if (child) {
      this.terminateChild(child);
    }
    this.rejectAllPending(new Error("cancelled (client disposed)"));
  }

  private ensureInitialized(): Promise<void> {
    if (this.initialized && this.child && !this.exited) return this.initialized;
    const initialization = this.startAndInitialize();
    this.initialized = initialization;
    void initialization.catch(() => {
      if (this.initialized === initialization) this.initialized = null;
    });
    return initialization;
  }

  private async startAndInitialize(): Promise<void> {
    const packageSpec = this.opts.packageSpec ?? "@z_ai/mcp-server@latest";
    const spawnFn = this.opts.spawn ?? nodeSpawn;
    const platform = this.opts.platform ?? process.platform;
    const isWindows = platform === "win32";
    if (isWindows && !isSafeNpmPackageSpec(packageSpec)) {
      throw new Error(`Vision MCP startup failed: unsafe npm package spec for Windows: ${packageSpec}`);
    }
    const command = isWindows ? (this.opts.comSpec ?? process.env.ComSpec ?? "cmd.exe") : "npx";
    const args = isWindows
      ? ["/d", "/s", "/c", "npx", "-y", packageSpec]
      : ["-y", packageSpec];
    this.exited = false;
    this.exitReason = null;
    this.buffer = "";
    this.stderrTail = "";
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawnFn(command, args, {
        env: { ...process.env, Z_AI_API_KEY: this.opts.apiKey, Z_AI_MODE: "ZAI" },
        windowsHide: true,
      });
    } catch (err) {
      const code = (err as { code?: string }).code;
      if (code === "ENOENT") {
        throw new Error("Vision MCP startup failed: `npx` not found on PATH. Install Node.js / npm 9+ and ensure `npx` is available.", {
          cause: err,
        });
      }
      throw new Error(`Vision MCP startup failed: ${(err as Error).message}`, {
        cause: err,
      });
    }
    this.child = child;
    this.initializingChild = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      if (this.child === child) this.handleStdout(chunk);
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      if (this.child === child) this.handleStderr(chunk);
    });
    child.on("exit", (code, sig) => {
      const reason = `exit code=${code} signal=${sig ?? "(none)"}`;
      this.failConnection(new Error(`server exited (${reason}).`), child, false);
    });
    child.on("error", (err) => {
      const startupError = (err as NodeJS.ErrnoException).code === "ENOENT"
        ? new Error("Vision MCP startup failed: could not launch npx. Ensure Node.js/npm are installed and npx is on PATH.", { cause: err })
        : new Error(`Vision MCP startup failed: ${err.message}`, { cause: err });
      this.failConnection(startupError, child, true);
    });

    const initializationTimeoutMs = this.opts.initializationTimeoutMs ?? DEFAULT_INITIALIZATION_TIMEOUT_MS;
    const deadline = Date.now() + initializationTimeoutMs;
    const remaining = () => Math.max(1, deadline - Date.now());
    try {
      await this.request("initialize", {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: "glm-acp-agent", version: "1.0.0" },
      }, "Vision MCP initialize", undefined, remaining());
      this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
      await this.rediscoverTools(undefined, remaining());
      if (this.initializingChild === child) this.initializingChild = null;
    } catch (err) {
      this.failConnection(err instanceof Error ? err : new Error(String(err)), child, true);
      throw err;
    }
  }

  private request(
    method: string,
    params: Record<string, unknown>,
    label: string,
    signal?: AbortSignal,
    timeoutMs = this.opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS,
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      let settled = false;
      const onAbort = () => {
        const pending = this.pending.get(id);
        if (pending) {
          this.pending.delete(id);
          pending.reject(new Error("aborted"));
        }
      };
      const cleanup = () => {
        if (timer) clearTimeout(timer);
        signal?.removeEventListener("abort", onAbort);
      };
      const settle = (fn: (value: unknown) => void, value: unknown) => {
        if (settled) return;
        settled = true;
        cleanup();
        fn(value);
      };
      this.pending.set(id, {
        method: label,
        resolve: (value) => settle(resolve, value),
        reject: (err) => settle(reject, new Error(`${label} failed: ${this.withStderr(err.message)}`)),
      });
      const timer = setTimeout(() => {
        const child = this.child;
        if (child && this.pending.has(id)) {
          this.failConnection(new Error(`request timed out after ${timeoutMs}ms`), child, true);
        }
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

  private failConnection(error: Error, child: ChildProcessWithoutNullStreams, kill: boolean): void {
    if (this.child !== child || this.exited) return;
    this.child = null;
    if (this.initializingChild === child) this.initializingChild = null;
    this.exited = true;
    this.exitReason = error.message;
    this.initialized = null;
    this.rejectAllPending(error);
    if (kill) {
      this.terminateChild(child);
    }
  }

  private terminateChild(child: ChildProcessWithoutNullStreams): void {
    const platform = this.opts.platform ?? process.platform;
    if (platform === "win32" && child.pid && child.exitCode === null && (!this.opts.spawn || this.opts.killProcessTree)) {
      try {
        const killed = this.opts.killProcessTree
          ? this.opts.killProcessTree(child.pid)
          : nodeSpawnSync("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
              stdio: "ignore",
              windowsHide: true,
            }).status === 0;
        if (killed) return;
      } catch {
        // fall back to the direct child below
      }
    }
    try {
      child.kill();
    } catch {
      // ignore
    }
  }

  private handleStderr(chunk: string): void {
    const tail = chunk.length >= STDERR_TAIL_LIMIT ? chunk.slice(-STDERR_TAIL_LIMIT) : this.stderrTail + chunk;
    this.stderrTail = tail.slice(-STDERR_TAIL_LIMIT);
  }

  private withStderr(message: string): string {
    let stderr = this.stderrTail.trim();
    if (this.opts.apiKey) stderr = stderr.split(this.opts.apiKey).join("[REDACTED]");
    stderr = stderr.replace(/\s+/g, " ").slice(-2_000);
    return stderr ? `${message}; stderr: ${stderr}` : message;
  }

  private send(message: Record<string, unknown>): void {
    if (!this.child || this.exited) {
      throw new Error(`Vision MCP server is not running${this.exitReason ? ` (${this.exitReason})` : ""}.`);
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
      let parsed: { id?: number; result?: unknown; error?: { code?: number; message?: string } };
      try {
        parsed = JSON.parse(line) as typeof parsed;
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

function waitForAbort<T>(promise: Promise<T>, signal: AbortSignal | undefined, message: string): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new Error(message));
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error(message));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(value);
      },
      (error) => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(error);
      },
    );
  });
}

function isSafeNpmPackageSpec(packageSpec: string): boolean {
  return /^(?:@[a-z0-9._-]+\/)?[a-z0-9._-]+(?:@[a-z0-9._-]+)?$/i.test(packageSpec);
}

function isVisionRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  if (/-32601/.test(msg)) return true;
  if (/tool.*not.*found|not.*found.*tool|unknown.*tool/.test(msg)) return true;
  return false;
}
