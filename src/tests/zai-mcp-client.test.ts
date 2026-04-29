import test from "node:test";
import assert from "node:assert/strict";
import { ZaiMcpClient } from "../tools/zai-mcp-client.js";

type FetchCall = {
  url: string;
  init: RequestInit;
  body: Record<string, unknown>;
  headers: Headers;
};

function jsonResponse(
  body: unknown,
  init: ResponseInit & { sessionId?: string } = {}
): Response {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (init.sessionId) headers.set("MCP-Session-Id", init.sessionId);
  return new Response(JSON.stringify(body), { ...init, headers });
}

function sseResponse(body: unknown): Response {
  return new Response(`event: message\ndata: ${JSON.stringify(body)}\n\n`, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

function createFetchStub(responses: Response[]) {
  const calls: FetchCall[] = [];
  const fetchStub = async (url: string | URL | Request, init?: RequestInit) => {
    assert.ok(init, "fetch init is required");
    const body =
      typeof init.body === "string" ? (JSON.parse(init.body) as Record<string, unknown>) : {};
    const headers = new Headers(init.headers);
    calls.push({ url: String(url), init, body, headers });
    const response = responses.shift();
    assert.ok(response, "unexpected fetch call");
    return response;
  };
  return { calls, fetchStub };
}

test("ZaiMcpClient initializes, sends initialized, discovers tools, and calls a tool", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { calls, fetchStub } = createFetchStub([
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "zai" } },
      },
      { sessionId: "session-123" }
    ),
    new Response(null, { status: 202 }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "webSearchPrime" }] },
    }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 3,
      result: { content: [{ type: "text", text: "Search result text" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  const result = await client.callTool({
    endpoint,
    toolName: "webSearchPrime",
    arguments: { query: "glm coding plan", count: 3 },
    apiKey: "test-key",
  });

  assert.deepEqual(result, { content: [{ type: "text", text: "Search result text" }] });
  assert.equal(calls.length, 4);
  assert.equal(calls[0]?.body.method, "initialize");
  assert.equal(calls[0]?.headers.get("Authorization"), "Bearer test-key");
  assert.equal(calls[0]?.headers.get("Accept"), "application/json, text/event-stream");
  assert.equal(calls[0]?.headers.get("Mcp-Method"), "initialize");
  assert.equal(calls[0]?.headers.get("MCP-Protocol-Version"), "2025-06-18");

  assert.equal(calls[1]?.body.method, "notifications/initialized");
  assert.equal(calls[1]?.headers.get("MCP-Session-Id"), "session-123");
  assert.equal(calls[1]?.headers.get("Mcp-Method"), "notifications/initialized");

  assert.equal(calls[2]?.body.method, "tools/list");
  assert.equal(calls[2]?.headers.get("MCP-Session-Id"), "session-123");

  assert.equal(calls[3]?.body.method, "tools/call");
  assert.deepEqual(calls[3]?.body.params, {
    name: "webSearchPrime",
    arguments: { query: "glm coding plan", count: 3 },
  });
  assert.equal(calls[3]?.headers.get("MCP-Session-Id"), "session-123");
  assert.equal(calls[3]?.headers.get("Mcp-Method"), "tools/call");
  assert.equal(calls[3]?.headers.get("Mcp-Name"), "webSearchPrime");
});

test("ZaiMcpClient discovers tools after initialization and caches them", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { calls, fetchStub } = createFetchStub([
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: "zai" } },
      },
      { sessionId: "session-discover" }
    ),
    new Response(null, { status: 202 }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: {
        tools: [
          { name: "webSearchPrime", description: "Search the web" },
          { name: "otherTool", description: "Something else" },
        ],
      },
    }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 3,
      result: { content: [{ type: "text", text: "ok" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  await client.callTool({
    endpoint,
    toolName: "webSearchPrime",
    arguments: { query: "test" },
    apiKey: "test-key",
  });

  assert.equal(calls.length, 4);
  assert.equal(calls[2]?.body.method, "tools/list");
  assert.equal(calls[2]?.headers.get("MCP-Session-Id"), "session-discover");
  assert.equal(calls[2]?.headers.get("Mcp-Method"), "tools/list");
});

test("ZaiMcpClient parses SSE wrapped JSON-RPC responses", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_reader/mcp";
  const { fetchStub } = createFetchStub([
    sseResponse({
      jsonrpc: "2.0",
      id: 1,
      result: { protocolVersion: "2025-06-18", capabilities: {} },
    }),
    new Response(null, { status: 202 }),
    sseResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "webReader" }] },
    }),
    sseResponse({
      jsonrpc: "2.0",
      id: 3,
      result: { content: [{ type: "text", text: "# Title\nBody" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  const result = await client.callTool({
    endpoint,
    toolName: "webReader",
    arguments: { url: "https://example.com" },
    apiKey: "test-key",
  });

  assert.deepEqual(result, { content: [{ type: "text", text: "# Title\nBody" }] });
});

test("ZaiMcpClient resolves tool name via keyword fallback when exact name is unavailable", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { calls, fetchStub } = createFetchStub([
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {} },
      },
      { sessionId: "session-fallback" }
    ),
    new Response(null, { status: 202 }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "webSearchV2" }, { name: "otherTool" }] },
    }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 3,
      result: { content: [{ type: "text", text: "fallback result" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  const result = await client.callTool({
    endpoint,
    toolName: "webSearchPrime",
    arguments: { query: "test" },
    apiKey: "test-key",
  });

  assert.deepEqual(result, { content: [{ type: "text", text: "fallback result" }] });
  assert.equal(calls[3]?.body.method, "tools/call");
  assert.deepEqual(calls[3]?.body.params, {
    name: "webSearchV2",
    arguments: { query: "test" },
  });
  assert.equal(calls[3]?.headers.get("Mcp-Name"), "webSearchV2");
});

test("ZaiMcpClient throws with diagnostics when no matching tool is found", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { fetchStub } = createFetchStub([
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {} },
      },
      { sessionId: "session-no-match" }
    ),
    new Response(null, { status: 202 }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "totallyUnrelated" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  await assert.rejects(
    () =>
      client.callTool({
        endpoint,
        toolName: "webSearchPrime",
        arguments: { query: "test" },
        apiKey: "test-key",
      }),
    /Tool "webSearchPrime" not available/
  );
});

test("ZaiMcpClient explains Coding Plan eligibility when Z.AI returns 1113", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { fetchStub } = createFetchStub([
    jsonResponse(
      {
        error: {
          code: "1113",
          message: "No permission for current API key",
        },
      },
      { status: 429 }
    ),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  await assert.rejects(
    () =>
      client.callTool({
        endpoint,
        toolName: "webSearchPrime",
        arguments: { query: "glm" },
        apiKey: "test-key",
      }),
    /Coding Plan quota\/base URL\/tool eligibility.*1113/
  );
});

test("ZaiMcpClient retries once on tool-not-found error with re-initialization", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { calls, fetchStub } = createFetchStub([
    // First attempt: init
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {} },
      },
      { sessionId: "session-old" }
    ),
    new Response(null, { status: 202 }),
    // First attempt: tools/list returns stale list
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "staleSearchTool" }] },
    }),
    // First attempt: tools/call fails with tool-not-found
    jsonResponse({
      jsonrpc: "2.0",
      id: 3,
      error: { code: -32601, message: "Tool not found: staleSearchTool" },
    }),
    // Retry: re-initialize
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 4,
        result: { protocolVersion: "2025-06-18", capabilities: {} },
      },
      { sessionId: "session-new" }
    ),
    new Response(null, { status: 202 }),
    // Retry: tools/list returns updated list
    jsonResponse({
      jsonrpc: "2.0",
      id: 5,
      result: { tools: [{ name: "webSearchV2" }] },
    }),
    // Retry: tools/call succeeds
    jsonResponse({
      jsonrpc: "2.0",
      id: 6,
      result: { content: [{ type: "text", text: "retry success" }] },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  const result = await client.callTool({
    endpoint,
    toolName: "webSearchPrime",
    arguments: { query: "test" },
    apiKey: "test-key",
  });

  assert.deepEqual(result, { content: [{ type: "text", text: "retry success" }] });
  assert.equal(calls.length, 8);
  assert.equal(calls[4]?.body.method, "initialize");
  assert.equal(calls[6]?.body.method, "tools/list");
  assert.equal(calls[7]?.body.method, "tools/call");
  assert.deepEqual(calls[7]?.body.params, {
    name: "webSearchV2",
    arguments: { query: "test" },
  });
});

test("ZaiMcpClient does not retry on non-tool-not-found errors", async () => {
  const endpoint = "https://api.z.ai/api/mcp/web_search_prime/mcp";
  const { calls, fetchStub } = createFetchStub([
    jsonResponse(
      {
        jsonrpc: "2.0",
        id: 1,
        result: { protocolVersion: "2025-06-18", capabilities: {} },
      },
      { sessionId: "session-no-retry" }
    ),
    new Response(null, { status: 202 }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 2,
      result: { tools: [{ name: "webSearchPrime" }] },
    }),
    jsonResponse({
      jsonrpc: "2.0",
      id: 3,
      error: { code: -32000, message: "Internal server error" },
    }),
  ]);

  const client = new ZaiMcpClient(fetchStub as typeof fetch);
  await assert.rejects(
    () =>
      client.callTool({
        endpoint,
        toolName: "webSearchPrime",
        arguments: { query: "test" },
        apiKey: "test-key",
      }),
    /Internal server error/
  );
  assert.equal(calls.length, 4);
});
