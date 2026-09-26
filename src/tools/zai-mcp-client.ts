import { remapArguments, resolveToolName, type DiscoveredTool } from "./mcp-arg-remap.js";
import {
  collectToolPages,
  assertValidToolPage,
  DEFAULT_MCP_MAX_PAGES,
  DEFAULT_MCP_MAX_SCHEMA_BYTES,
  DEFAULT_MCP_MAX_TOOLS,
} from "./mcp-pagination.js";
import { readMcpResponseText } from "./mcp-response-limit.js";

const MCP_PROTOCOL_VERSION = "2025-06-18";

export const ZAI_WEB_SEARCH_MCP_ENDPOINT =
  "https://api.z.ai/api/mcp/web_search_prime/mcp";
export const ZAI_WEB_READER_MCP_ENDPOINT =
  "https://api.z.ai/api/mcp/web_reader/mcp";

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

export interface ZaiMcpToolCall {
  endpoint: string;
  toolName: string;
  arguments: Record<string, unknown>;
  apiKey: string;
  signal?: AbortSignal;
}

export interface ZaiMcpClientOptions {
  maxPages?: number;
  maxTools?: number;
  maxSchemaBytes?: number;
}

export class ZaiMcpClient {
  private sessions = new Map<string, { sessionId?: string; initialized: boolean; tools: DiscoveredTool[] }>();
  private nextId = 1;

  constructor(
    private fetchImpl: typeof fetch = ((...args: Parameters<typeof fetch>) =>
      fetch(...args)) as typeof fetch,
    private limits: ZaiMcpClientOptions = {}
  ) {}

  async callTool(call: ZaiMcpToolCall): Promise<unknown> {
    try {
      return await this.callToolInternal(call);
    } catch (err) {
      if (!isRetryableError(err)) throw err;
      const cacheKey = `${call.endpoint}\n${call.apiKey}`;
      this.sessions.delete(cacheKey);
      return this.callToolInternal(call);
    }
  }

  private async callToolInternal(call: ZaiMcpToolCall): Promise<unknown> {
    const session = await this.ensureInitialized(call);
    const toolNames = session.tools.map((t) => t.name);
    const resolvedName = resolveToolName(call.toolName, toolNames, call.endpoint);
    const toolSchema = session.tools.find((t) => t.name === resolvedName);
    const remappedArgs = remapArguments(call.arguments, toolSchema?.properties ?? []);
    const result = await this.sendRequest(
      call.endpoint,
      call.apiKey,
      "tools/call",
      {
        jsonrpc: "2.0",
        id: this.nextId++,
        method: "tools/call",
        params: {
          name: resolvedName,
          arguments: remappedArgs,
        },
      },
      "tools/call",
      call.signal,
      session.sessionId,
      resolvedName
    );
    return result;
  }

  private async ensureInitialized(call: ZaiMcpToolCall) {
    const cacheKey = `${call.endpoint}\n${call.apiKey}`;
    const cached = this.sessions.get(cacheKey);
    if (cached?.initialized) return cached;

    const initializeResponse = await this.fetchJsonRpc(
      call.endpoint,
      call.apiKey,
      "initialize",
      {
        jsonrpc: "2.0",
        id: this.nextId++,
        method: "initialize",
        params: {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: {},
          clientInfo: {
            name: "glm-acp-agent",
            version: "1.0.0",
          },
        },
      },
      "initialize",
      call.signal
    );

    const sessionId = initializeResponse.sessionId;
    await this.sendNotification(
      call.endpoint,
      call.apiKey,
      {
        jsonrpc: "2.0",
        method: "notifications/initialized",
      },
      "notifications/initialized",
      call.signal,
      sessionId
    );

    const tools = await this.discoverTools(
      call.endpoint,
      call.apiKey,
      sessionId,
      call.signal
    );
    const session = { sessionId, initialized: true, tools };
    this.sessions.set(cacheKey, session);
    return session;
  }

  private async discoverTools(
    endpoint: string,
    apiKey: string,
    sessionId: string | undefined,
    signal?: AbortSignal
  ): Promise<DiscoveredTool[]> {
    const rawTools = await collectToolPages({
      signal: signal ?? new AbortController().signal,
      maxPages: this.limits.maxPages ?? DEFAULT_MCP_MAX_PAGES,
      maxTools: this.limits.maxTools ?? DEFAULT_MCP_MAX_TOOLS,
      maxSchemaBytes: this.limits.maxSchemaBytes ?? DEFAULT_MCP_MAX_SCHEMA_BYTES,
      requestPage: async (cursor, pageSignal) => {
        const response = await this.fetchJsonRpc(
          endpoint,
          apiKey,
          "tools/list",
          {
            jsonrpc: "2.0",
            id: this.nextId++,
            method: "tools/list",
            ...(cursor === undefined ? {} : { params: { cursor } }),
          },
          "tools/list",
          pageSignal,
          sessionId
        );
        return parseToolPage(response.body.result);
      },
    });
    const tools = rawTools.map((tool) => ({
      name: tool.name,
      properties: tool.inputSchema?.properties ? Object.keys(tool.inputSchema.properties) : [],
    }));
    assertUniqueToolNames(tools, endpoint);
    return tools;
  }

  private async sendRequest(
    endpoint: string,
    apiKey: string,
    mcpMethod: string,
    body: JsonRpcRequest,
    stage: string,
    signal?: AbortSignal,
    sessionId?: string,
    mcpName?: string
  ): Promise<unknown> {
    const response = await this.fetchJsonRpc(
      endpoint,
      apiKey,
      mcpMethod,
      body,
      stage,
      signal,
      sessionId,
      mcpName
    );
    return response.body.result;
  }

  private async sendNotification(
    endpoint: string,
    apiKey: string,
    body: JsonRpcRequest,
    mcpMethod: string,
    signal?: AbortSignal,
    sessionId?: string
  ): Promise<void> {
    const response = await this.fetchImpl(endpoint, {
      method: "POST",
      headers: buildHeaders(apiKey, mcpMethod, sessionId),
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok) {
      const text = await readMcpResponseText(response);
      throw new Error(formatMcpError(mcpMethod, response.status, text));
    }
  }

  private async fetchJsonRpc(
    endpoint: string,
    apiKey: string,
    mcpMethod: string,
    body: JsonRpcRequest,
    stage: string,
    signal?: AbortSignal,
    sessionId?: string,
    mcpName?: string
  ): Promise<{ body: JsonRpcResponse; sessionId?: string }> {
    const response = await this.fetchImpl(endpoint, {
      method: "POST",
      headers: buildHeaders(apiKey, mcpMethod, sessionId, mcpName),
      body: JSON.stringify(body),
      signal,
    });
    const text = await readMcpResponseText(response);
    if (!response.ok) {
      throw new Error(formatMcpError(stage, response.status, text));
    }

    const parsed = parseMcpResponse(text, response.headers.get("Content-Type") ?? "");
    if (parsed.error) {
      throw new Error(formatJsonRpcError(stage, parsed.error));
    }

    return {
      body: parsed,
      sessionId: response.headers.get("MCP-Session-Id") ?? undefined,
    };
  }
}

const defaultClient = new ZaiMcpClient();

export function callZaiMcpTool(call: ZaiMcpToolCall): Promise<unknown> {
  return defaultClient.callTool(call);
}

interface RawZaiTool {
  name: string;
  inputSchema?: { properties?: Record<string, unknown> };
  [key: string]: unknown;
}

function parseToolPage(result: unknown): { tools: RawZaiTool[]; nextCursor?: string | null } {
  assertValidToolPage(result);
  const record = result as Record<string, unknown>;
  const tools = Array.isArray(record.tools)
    ? record.tools.filter((tool): tool is RawZaiTool =>
        Boolean(tool) && typeof tool === "object" && typeof (tool as Record<string, unknown>).name === "string"
      )
    : [];
  const nextCursor = record.nextCursor;
  return {
    tools,
    nextCursor: nextCursor === undefined || nextCursor === null || typeof nextCursor === "string"
      ? nextCursor
      : nextCursor as never,
  };
}

function assertUniqueToolNames(tools: DiscoveredTool[], endpoint: string): void {
  const seen = new Set<string>();
  for (const tool of tools) {
    if (seen.has(tool.name)) {
      throw new Error(`MCP server "${endpoint}" returned duplicate tool name "${tool.name}"`);
    }
    seen.add(tool.name);
  }
}

function buildHeaders(
  apiKey: string,
  mcpMethod: string,
  sessionId?: string,
  mcpName?: string
): Headers {
  const headers = new Headers({
    Authorization: `Bearer ${apiKey}`,
    Accept: "application/json, text/event-stream",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
    "Mcp-Method": mcpMethod,
  });
  if (sessionId) headers.set("MCP-Session-Id", sessionId);
  if (mcpName) headers.set("Mcp-Name", mcpName);
  return headers;
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

function formatMcpError(stage: string, status: number, body: string): string {
  if (isCodingPlanEligibilityError(body)) {
    return `MCP ${stage} failed: HTTP ${status}. Coding Plan quota/base URL/tool eligibility likely is not being met (business code 1113). ${body}`;
  }
  return `MCP ${stage} failed: HTTP ${status}: ${body}`;
}

function formatJsonRpcError(stage: string, error: NonNullable<JsonRpcResponse["error"]>): string {
  const details = JSON.stringify(error);
  if (isCodingPlanEligibilityError(details)) {
    return `MCP ${stage} failed: Coding Plan quota/base URL/tool eligibility likely is not being met (business code 1113). ${details}`;
  }
  return `MCP ${stage} failed: ${details}`;
}

function isCodingPlanEligibilityError(body: string): boolean {
  return /(^|["\s:])1113($|["\s,}])/.test(body);
}

function isRetryableError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  if (/-32601/.test(msg)) return true;
  if (/-32602/.test(msg)) return true;
  if (/tool.*not.*found|not.*found.*tool|unknown.*tool/.test(msg)) return true;
  return false;
}
