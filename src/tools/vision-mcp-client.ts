import { spawn as nodeSpawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { remapArguments, resolveToolName } from "./mcp-arg-remap.js";

const MCP_PROTOCOL_VERSION = "2025-06-18";

export interface VisionMcpClient {
  callTool(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown>;
  dispose(): Promise<void>;
}

interface StdioVisionMcpClientOptions {
  apiKey: string;
  /** Override the package spec for tests/pinning. Defaults to `@z_ai/mcp-server@latest`. */
  packageSpec?: string;
  /** Override the spawn function for tests. */
  spawn?: (command: string, args: string[], options: { env: NodeJS.ProcessEnv }) => ChildProcessWithoutNullStreams;
}

interface DiscoveredTool {
  name: string;
  properties: string[];
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  method: string;
}

export class StdioVisionMcpClient implements VisionMcpClient {
  private child: ChildProcessWithoutNullStreams | null = null;
  private initialized: Promise<void> | null = null;
  private nextId = 1;
  private pending = new Map<number, PendingRequest>();
  private buffer = "";
  private exited = false;
  private exitReason: string | null = null;
  private discoveredTools: DiscoveredTool[] = [];

  constructor(private opts: StdioVisionMcpClientOptions) {}

  async callTool(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    if (signal?.aborted) throw new Error("Vision MCP call cancelled");
    await this.ensureInitialized();
    return this.callToolInternal(toolName, args, signal);
  }

  private async callToolInternal(toolName: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<unknown> {
    const toolNames = this.discoveredTools.map((t) => t.name);
    const resolvedName = resolveToolName(toolName, toolNames, "@z_ai/mcp-server");
    const toolSchema = this.discoveredTools.find((t) => t.name === resolvedName);
    const remappedArgs = remapArguments(args, toolSchema?.properties ?? []);
    try {
      return await this.request("tools/call", { name: resolvedName, arguments: remappedArgs }, `Vision MCP ${toolName}`, signal);
    } catch (err) {
      if (!isVisionRetryableError(err)) throw err;
      await this.rediscoverTools();
      const resolvedName2 = resolveToolName(toolName, this.discoveredTools.map((t) => t.name), "@z_ai/mcp-server");
      const toolSchema2 = this.discoveredTools.find((t) => t.name === resolvedName2);
      const remappedArgs2 = remapArguments(args, toolSchema2?.properties ?? []);
      return this.request("tools/call", { name: resolvedName2, arguments: remappedArgs2 }, `Vision MCP ${toolName}`, signal);
    }
  }

  private async rediscoverTools(): Promise<void> {
    const result = await this.request("tools/list", {}, "Vision MCP tools/list") as
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
    if (this.child && !this.exited) {
      try {
        this.child.kill();
      } catch {
        // ignore
      }
    }
    this.child = null;
    this.initialized = null;
    this.discoveredTools = [];
    for (const [, p] of this.pending) {
      p.reject(new Error(`cancelled (client disposed)`));
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
    const packageSpec = this.opts.packageSpec ?? "@z_ai/mcp-server@latest";
    const spawnFn = this.opts.spawn ?? nodeSpawn;
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawnFn("npx", ["-y", packageSpec], {
        env: { ...process.env, Z_AI_API_KEY: this.opts.apiKey, Z_AI_MODE: "ZAI" },
      });
    } catch (err) {
      const code = (err as { code?: string }).code;
      if (code === "ENOENT") {
        throw new Error("Vision MCP startup failed: `npx` not found on PATH. Install Node.js / npm 9+ and ensure `npx` is available.");
      }
      throw new Error(`Vision MCP startup failed: ${(err as Error).message}`);
    }
    this.child = child;
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.handleStdout(chunk));
    child.on("exit", (code, sig) => {
      this.exited = true;
      this.exitReason = `exit code=${code} signal=${sig ?? "(none)"}`;
      for (const [, p] of this.pending) {
        p.reject(new Error(`server exited (${this.exitReason}).`));
      }
      this.pending.clear();
    });
    child.on("error", (err) => {
      this.exited = true;
      this.exitReason = err.message;
    });

    await this.request("initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "glm-acp-agent", version: "1.0.0" },
    }, "Vision MCP initialize");
    this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
    await this.rediscoverTools();
  }

  private request(method: string, params: Record<string, unknown>, label: string, signal?: AbortSignal): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, {
        method: label,
        resolve,
        reject: (err) => reject(new Error(`${label} failed: ${err.message}`)),
      });
      const onAbort = () => {
        const pending = this.pending.get(id);
        if (pending) {
          this.pending.delete(id);
          pending.reject(new Error("aborted"));
        }
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      try {
        this.send({ jsonrpc: "2.0", id, method, params });
      } catch (err) {
        this.pending.delete(id);
        reject(new Error(`${label} failed: ${(err as Error).message}`));
      }
    });
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

function isVisionRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  if (/-32601/.test(msg)) return true;
  if (/tool.*not.*found|not.*found.*tool|unknown.*tool/.test(msg)) return true;
  return false;
}
