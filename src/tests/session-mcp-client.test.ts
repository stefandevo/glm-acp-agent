import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { Readable, Writable } from "node:stream";
import { createServer, type Server } from "node:http";
import { once } from "node:events";
import type { McpServer, McpServerStdio } from "@agentclientprotocol/sdk";
import {
  HttpMcpClient,
  SessionMcpTools,
  StdioMcpClient,
  type ConnectedMcpClient,
  type ToolBinding,
  connectSessionMcpServers,
} from "../tools/session-mcp-client.js";

interface FakeChild extends EventEmitter {
  stdin: Writable;
  stdout: Readable;
  stderr: Readable;
  pid: number;
  exitCode: number | null;
  kill: (signal?: string) => boolean;
}

function makeFakeChild(): {
  child: FakeChild;
  written: string[];
  pushStdout: (line: string) => void;
  pushStderr: (line: string) => void;
  getKillCount: () => number;
} {
  const written: string[] = [];
  let killCount = 0;
  const stdin = new Writable({
    write(chunk, _enc, cb) {
      written.push(chunk.toString("utf8"));
      cb();
    },
  });
  const stdout = new Readable({ read() { /* push manually */ } });
  const stderr = new Readable({ read() { /* push manually */ } });
  const child = Object.assign(new EventEmitter(), {
    stdin,
    stdout,
    stderr,
    pid: 5150,
    exitCode: null,
    kill: () => {
      killCount += 1;
      child.exitCode = 137;
      queueMicrotask(() => child.emit("exit", 137, "SIGTERM"));
      return true;
    },
  }) as FakeChild;
  return {
    child,
    written,
    pushStdout: (line: string) => stdout.push(line),
    pushStderr: (line: string) => stderr.push(line),
    getKillCount: () => killCount,
  };
}

function stdioServer(overrides: Partial<McpServerStdio> = {}): McpServerStdio {
  return {
    name: "docs",
    command: "npx",
    args: ["-y", "@example/mcp-docs"],
    env: [],
    ...overrides,
  };
}

const tick = () => new Promise((r) => setImmediate(r));

test("connectSessionMcpServers aborts stalled HTTP initialization", async () => {
  const previousFetch = globalThis.fetch;
  let setupSignal: AbortSignal | undefined;
  globalThis.fetch = ((_: RequestInfo | URL, init?: RequestInit) => {
    setupSignal = init?.signal ?? undefined;
    return new Promise<Response>((_resolve, reject) => {
      setupSignal?.addEventListener("abort", () => reject(new Error("synthetic setup abort")), { once: true });
    });
  }) as typeof fetch;
  const controller = new AbortController();
  try {
    const pending = connectSessionMcpServers([
      { type: "http", name: "stalled", url: "https://mcp.example.test/stalled", headers: [] },
    ], controller.signal);
    await tick();
    assert.ok(setupSignal, "HTTP setup must receive an abort signal");
    controller.abort();
    await assert.rejects(pending, /synthetic setup abort|cancelled/i);
    assert.equal(setupSignal.aborted, true);
  } finally {
    globalThis.fetch = previousFetch;
  }
});

function httpServer() {
  return { type: "http" as const, name: "fixture", url: "https://fixture.invalid", headers: [] };
}

interface DelayedMcpFixtureConfig {
  name: string;
  delayMs: number;
  toolsListDelayMs?: number;
  failToolsList?: boolean;
}

async function createDelayedMcpFixtures(configs: DelayedMcpFixtureConfig[]) {
  let activeRequests = 0;
  let peakActiveRequests = 0;
  const deleteCounts = new Map<string, number>();
  const fixtures: Server[] = [];
  const servers: McpServer[] = [];

  for (const config of configs) {
    deleteCounts.set(config.name, 0);
    const fixture = createServer((request, response) => {
      void (async () => {
        if (request.method === "DELETE") {
          deleteCounts.set(config.name, (deleteCounts.get(config.name) ?? 0) + 1);
          response.writeHead(202).end();
          return;
        }
        let text = "";
        for await (const chunk of request) text += chunk;
        const message = JSON.parse(text) as { id?: number; method?: string };
        activeRequests += 1;
        peakActiveRequests = Math.max(peakActiveRequests, activeRequests);
        try {
          const delayMs = message.method === "tools/list" ? config.toolsListDelayMs ?? config.delayMs : config.delayMs;
          await new Promise((resolve) => setTimeout(resolve, delayMs));
          if (response.destroyed) return;
          if (message.method === "notifications/initialized") {
            response.writeHead(202).end();
            return;
          }
          if (message.method === "tools/list" && config.failToolsList) {
            response.writeHead(500).end("fixture discovery failure");
            return;
          }
          const result = message.method === "initialize"
            ? { protocolVersion: "2025-06-18", capabilities: {}, serverInfo: { name: config.name, version: "1" } }
            : { tools: [{ name: "phase5_fixture_tool", inputSchema: { type: "object", properties: {} } }] };
          if (message.method === "initialize") response.setHeader("MCP-Session-Id", config.name);
          response.writeHead(200, { "Content-Type": "application/json" })
            .end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
        } finally {
          activeRequests -= 1;
        }
      })().catch((error: unknown) => response.destroy(error instanceof Error ? error : undefined));
    });
    fixture.listen(0, "127.0.0.1");
    await once(fixture, "listening");
    const address = fixture.address();
    if (!address || typeof address === "string") throw new Error("MCP fixture did not bind a TCP port");
    fixtures.push(fixture);
    servers.push({ type: "http", name: config.name, url: "http://127.0.0.1:" + address.port, headers: [] });
  }

  return {
    servers,
    get peakActiveRequests() { return peakActiveRequests; },
    deleteCount: (name: string) => deleteCounts.get(name) ?? 0,
    close: async () => {
      await Promise.all(fixtures.map((fixture) => new Promise<void>((resolve, reject) => {
        fixture.close((error) => error ? reject(error) : resolve());
      })));
    },
  };
}

test("connectSessionMcpServers bounds parallel discovery and preserves configured naming order", async () => {
  const fixtures = await createDelayedMcpFixtures([
    { name: "first", delayMs: 35 },
    { name: "second", delayMs: 25 },
    { name: "third", delayMs: 5 },
    { name: "fourth", delayMs: 15 },
    { name: "fifth", delayMs: 1 },
  ]);
  let tools: SessionMcpTools | undefined;
  try {
    tools = await connectSessionMcpServers(fixtures.servers);
    assert.equal(fixtures.peakActiveRequests, 3);
    assert.deepEqual(tools.toolNames, [
      "phase5_fixture_tool",
      "second_phase5_fixture_tool",
      "third_phase5_fixture_tool",
      "fourth_phase5_fixture_tool",
      "fifth_phase5_fixture_tool",
    ]);
  } finally {
    await tools?.dispose();
    await fixtures.close();
  }
});

test("connectSessionMcpServers disposes every started client after parallel discovery failure", async () => {
  const fixtures = await createDelayedMcpFixtures([
    { name: "broken", delayMs: 5, toolsListDelayMs: 5, failToolsList: true },
    { name: "slow-one", delayMs: 5, toolsListDelayMs: 100 },
    { name: "slow-two", delayMs: 5, toolsListDelayMs: 100 },
  ]);
  try {
    await assert.rejects(connectSessionMcpServers(fixtures.servers), /broken tools\/list failed/);
    assert.equal(fixtures.peakActiveRequests, 3);
    assert.equal(fixtures.deleteCount("broken"), 1);
    assert.equal(fixtures.deleteCount("slow-one"), 1);
    assert.equal(fixtures.deleteCount("slow-two"), 1);
  } finally {
    await fixtures.close();
  }
});

test("HTTP MCP cancels a stalled shared initialization when its only waiter aborts", async () => {
  const originalFetch = globalThis.fetch;
  let initializeSignal: AbortSignal | undefined;
  let initializationAborted = false;
  const initializationStarted = new Promise<void>((resolve) => {
    globalThis.fetch = (async (_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { method?: string };
      if (body.method !== "initialize") throw new Error(`unexpected method ${String(body.method)}`);
      initializeSignal = init?.signal ?? undefined;
      resolve();
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          initializationAborted = true;
          reject(new Error("initialization aborted"));
        }, { once: true });
      });
    }) as typeof fetch;
  });
  try {
    const client = new HttpMcpClient(httpServer());
    const controller = new AbortController();
    const listPromise = client.listTools(controller.signal);
    await initializationStarted;
    controller.abort();
    const watchdog = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("HTTP initialization cancellation hung")), 250).unref();
    });
    await assert.rejects(Promise.race([listPromise, watchdog]), /cancelled|aborted/i);
    assert.equal(initializeSignal?.aborted, true);
    assert.equal(initializationAborted, true);
    await client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("HTTP MCP retries initialization for a caller started immediately after cancellation", async () => {
  const originalFetch = globalThis.fetch;
  let initializeCalls = 0;
  let firstInitializationStarted!: () => void;
  let secondInitializationStarted!: () => void;
  const firstStarted = new Promise<void>((resolve) => { firstInitializationStarted = resolve; });
  const secondStarted = new Promise<void>((resolve) => { secondInitializationStarted = resolve; });
  globalThis.fetch = (async (_url, init) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { method?: string; id?: number };
    if (body.method === "initialize") {
      initializeCalls += 1;
      if (initializeCalls === 1) {
        firstInitializationStarted();
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("initialization aborted")), { once: true });
        });
      }
      secondInitializationStarted();
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {} }), {
        headers: { "Content-Type": "application/json", "MCP-Session-Id": "retry-session" },
      });
    }
    if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
    if (body.method === "tools/list") {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { tools: [{ name: "search" }] } }), {
        headers: { "Content-Type": "application/json" },
      });
    }
    if (init?.method === "DELETE") return new Response(null, { status: 202 });
    throw new Error(`unexpected method ${String(body.method)}`);
  }) as typeof fetch;
  try {
    const client = new HttpMcpClient(httpServer());
    const controller = new AbortController();
    const cancelled = client.listTools(controller.signal);
    void cancelled.catch(() => {});
    await firstStarted;
    controller.abort();

    // Start the replacement before the cancelled caller's promise continuation
    // runs. This is the race that must not attach to the doomed initialization.
    const retry = client.listTools();
    const watchdog = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("replacement initialization did not start")), 250).unref();
    });
    await Promise.race([secondStarted, watchdog]);
    await assert.rejects(cancelled, /cancelled|aborted/i);
    assert.deepEqual(await retry, [{ name: "search", description: undefined, inputSchema: undefined }]);
    assert.equal(initializeCalls, 2);
    await client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("HTTP MCP keeps shared initialization alive for a concurrent non-aborted waiter", async () => {
  const originalFetch = globalThis.fetch;
  let initializeSignal: AbortSignal | undefined;
  let resolveInitialize!: (response: Response) => void;
  let initializeCalls = 0;
  const initializationStarted = new Promise<void>((resolve) => {
    globalThis.fetch = (async (_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { method?: string; id?: number };
      if (body.method === "initialize") {
        initializeCalls += 1;
        initializeSignal = init?.signal ?? undefined;
        resolve();
        return new Promise<Response>((responseResolve) => { resolveInitialize = responseResolve; });
      }
      if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (body.method === "tools/list") {
        return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { tools: [{ name: "search" }] } }), {
          headers: { "Content-Type": "application/json" },
        });
      }
      throw new Error(`unexpected method ${String(body.method)}`);
    }) as typeof fetch;
  });
  try {
    const client = new HttpMcpClient(httpServer());
    const controller = new AbortController();
    const cancelled = client.listTools(controller.signal);
    const surviving = client.listTools();
    await initializationStarted;
    controller.abort();
    await assert.rejects(cancelled, /cancelled|aborted/i);
    assert.equal(initializeSignal?.aborted, false);
    resolveInitialize(new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }), {
      headers: { "Content-Type": "application/json", "MCP-Session-Id": "shared-session" },
    }));
    assert.deepEqual(await surviving, [{ name: "search", description: undefined, inputSchema: undefined }]);
    assert.equal(initializeCalls, 1);
    await client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("HTTP MCP forwards cancellation to a stalled initialized notification", async () => {
  const originalFetch = globalThis.fetch;
  let notificationSignal: AbortSignal | undefined;
  let notificationAborted = false;
  const notificationStarted = new Promise<void>((resolve) => {
    globalThis.fetch = (async (_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { method?: string; id?: number };
      if (body.method === "initialize") {
        return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {} }), {
          headers: { "Content-Type": "application/json", "MCP-Session-Id": "notification-session" },
        });
      }
      if (body.method !== "notifications/initialized") throw new Error(`unexpected method ${String(body.method)}`);
      notificationSignal = init?.signal ?? undefined;
      resolve();
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          notificationAborted = true;
          reject(new Error("initialized notification aborted"));
        }, { once: true });
      });
    }) as typeof fetch;
  });
  try {
    const client = new HttpMcpClient(httpServer());
    const controller = new AbortController();
    const listPromise = client.listTools(controller.signal);
    await notificationStarted;
    controller.abort();
    const watchdog = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error("initialized notification cancellation hung")), 250).unref();
    });
    await assert.rejects(Promise.race([listPromise, watchdog]), /cancelled|aborted/i);
    assert.equal(notificationSignal?.aborted, true);
    assert.equal(notificationAborted, true);
    await client.dispose();
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("HTTP discovery rejects malformed later pages rather than exposing a partial catalog", async () => {
  const savedFetch = globalThis.fetch;
  try {
    for (const invalid of [null, { tools: {} }, { tools: [null] }, { tools: [{ name: 42 }] }]) {
      let pages = 0;
      globalThis.fetch = (async (_url, init) => {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
        const result = body.method === "tools/list"
          ? (++pages === 1 ? { tools: [{ name: "first" }], nextCursor: "next" } : invalid)
          : {};
        return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }), { headers: { "Content-Type": "application/json" } });
      }) as typeof fetch;
      await assert.rejects(connectSessionMcpServers([{ type: "http", name: "fixture", url: "https://fixture.invalid", headers: [] }]), /malformed.*tools\/list/i);
    }
  } finally { globalThis.fetch = savedFetch; }
});

for (const operation of ["cancel", "dispose"]) {
  test(`HTTP ${operation} still aborts after response headers while body is pending`, async () => {
    const originalFetch = globalThis.fetch;
    let bodyStarted!: () => void;
    const started = new Promise<void>(resolve => { bodyStarted = resolve; });
    let bodyController: ReadableStreamDefaultController<Uint8Array> | undefined;
    globalThis.fetch = (async (_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}"));
      if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
      if (body.method === "tools/call") {
        return new Response(new ReadableStream<Uint8Array>({
          pull(controller) {
            bodyController = controller;
            init?.signal?.addEventListener("abort", () => controller.error(new Error("body aborted")), { once: true });
            bodyStarted();
            return new Promise<void>(() => {});
          },
        }, { highWaterMark: 0 }), { headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id,
        result: body.method === "tools/list" ? { tools: [{ name: "read" }] } : {},
      }), { headers: { "Content-Type": "application/json" } });
    }) as typeof fetch;
    let tools: SessionMcpTools | undefined;
    let call: Promise<unknown> | undefined;
    let timer: NodeJS.Timeout | undefined;
    try {
      tools = await connectSessionMcpServers([{ type: "http", name: "fixture", url: "https://fixture.invalid", headers: [] }]);
      const controller = new AbortController();
      call = tools.callTool("read", {}, controller.signal);
      void call.catch(() => {});
      await started;
      if (operation === "cancel") controller.abort();
      else await tools.dispose();
      const watchdog = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("pending body was not interrupted")), 200);
      });
      await assert.rejects(Promise.race([call, watchdog]), /body aborted/);
    } finally {
      if (timer) clearTimeout(timer);
      bodyController?.error(new Error("fixture cleanup"));
      await call?.catch(() => {});
      await tools?.dispose();
      globalThis.fetch = originalFetch;
    }
  });
}

/** Drive a fake child through initialize + tools/list so the client is ready for tools/call. */
async function completeHandshake(
  written: string[],
  pushStdout: (line: string) => void,
  tools: { name: string }[] = [{ name: "search" }]
): Promise<void> {
  await tick();
  const init = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: init.id, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await tick();
  const list = JSON.parse(written[2]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: list.id, result: { tools } }) + "\n");
  await tick();
}

test("StdioMcpClient rejects an asynchronous spawn error instead of hanging", async () => {
  const { child } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", Object.assign(new Error("spawn npx ENOENT"), { code: "ENOENT" }));

  await assert.rejects(listPromise, /could not launch `npx`/i);
  await client.dispose();
});

test("StdioMcpClient collects every tools/list page and forwards the opaque cursor", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();

  await new Promise((r) => setImmediate(r));
  const init = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: init.id, result: {} }) + "\n");
  await new Promise((r) => setImmediate(r));
  const firstList = JSON.parse(written[2]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({
    jsonrpc: "2.0",
    id: firstList.id,
    result: { tools: [{ name: "first" }], nextCursor: "opaque/std-2" },
  }) + "\n");
  await new Promise((r) => setImmediate(r));
  const secondList = JSON.parse(written[3]?.trim() ?? "{}") as { id: number; params: { cursor: string } };
  assert.equal(secondList.params.cursor, "opaque/std-2");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: secondList.id, result: { tools: [{ name: "second" }] } }) + "\n");

  assert.deepEqual((await listPromise).map((tool) => tool.name), ["first", "second"]);
  const secondCall = client.callTool("second", {});
  await new Promise((r) => setImmediate(r));
  const secondCallBody = JSON.parse(written[4]?.trim() ?? "{}") as { id: number; params: { name: string } };
  assert.equal(secondCallBody.params.name, "second");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: secondCallBody.id, result: { content: [{ type: "text", text: "ok" }] } }) + "\n");
  await secondCall;
  await client.dispose();
});

test("SessionMcpTools owns and disposes zero-tool clients exactly once", async () => {
  let disposeCount = 0;
  const client: ConnectedMcpClient = {
    listTools: async () => [],
    callTool: async () => undefined,
    dispose: async () => { disposeCount += 1; },
  };
  const tools = new SessionMcpTools([] as ToolBinding[], [client, client]);
  await Promise.all([tools.dispose(), tools.dispose()]);
  assert.equal(disposeCount, 1);
});

test("HTTP MCP discovery collects a second page with its opaque cursor", async () => {
  const savedFetch = globalThis.fetch;
  const calls: Array<Record<string, unknown>> = [];
  globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
    calls.push(body);
    if (body.method === "initialize") {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {} }), {
        headers: { "Content-Type": "application/json", "MCP-Session-Id": "session-http" },
      });
    }
    if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
    if (body.method === "tools/list" && calls.filter((entry) => entry.method === "tools/list").length === 1) {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { tools: [{ name: "first" }], nextCursor: "opaque/http-2" } }), {
        headers: { "Content-Type": "application/json" },
      });
    }
    if (body.method === "tools/list") {
      assert.deepEqual(body.params, { cursor: "opaque/http-2" });
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { tools: [{ name: "second" }] } }), {
        headers: { "Content-Type": "application/json" },
      });
    }
    if (body.method === "tools/call") {
      assert.equal((body.params as { name: string }).name, "second");
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { content: [{ type: "text", text: "ok" }] } }), {
        headers: { "Content-Type": "application/json" },
      });
    }
    throw new Error(`unexpected method ${String(body.method)}`);
  }) as typeof fetch;
  try {
    const tools = await connectSessionMcpServers([{
      type: "http",
      name: "http",
      url: "https://mcp.example.test",
      headers: [],
    }]);
    assert.deepEqual(tools.toolNames, ["first", "second"]);
    await tools.callTool("second", {});
    await tools.dispose();
  } finally {
    globalThis.fetch = savedFetch;
  }
});

test("StdioMcpClient rejects a pending tools/call when the child errors asynchronously", async () => {
  const { child, written, pushStdout } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  const callPromise = client.callTool("search", { q: "x" });
  await tick();
  child.emit("error", Object.assign(new Error("spawn npx EINVAL"), { code: "EINVAL" }));

  await assert.rejects(callPromise, /EINVAL/);
  await client.dispose();
});

test("StdioMcpClient times out initialization and terminates the child", async () => {
  const { child, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => child as never,
    initializationTimeoutMs: 50,
  });

  await assert.rejects(() => client.listTools(), /timed out after \d+ms/i);
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioMcpClient times out an individual tools/call and terminates the child", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => child as never,
    requestTimeoutMs: 50,
  });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  await assert.rejects(() => client.callTool("search", { q: "x" }), /timed out after \d+ms/i);
  assert.equal(getKillCount(), 1);
  await client.dispose();
});

test("StdioMcpClient disposal escalates a stubborn child and waits for exit", async () => {
  const { child } = makeFakeChild();
  let disposeResolved = false;
  child.kill = (signal?: string) => {
    assert.equal(signal, "SIGTERM");
    child.kill = (nextSignal?: string) => {
      assert.equal(nextSignal, "SIGKILL");
      child.exitCode = 137;
      child.emit("exit", 137, "SIGKILL");
      return true;
    };
    return true;
  };
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  (client as unknown as { child: FakeChild }).child = child;
  const disposing = client.dispose().then(() => { disposeResolved = true; });
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(disposeResolved, false);
  await disposing;
  assert.equal(child.stdin.destroyed, true);
  assert.equal(child.stdout.destroyed, true);
  assert.equal(child.stderr.destroyed, true);
});

test("StdioMcpClient disposal rejects when KILL cannot prove child exit", async () => {
  const { child } = makeFakeChild();
  child.kill = () => true;
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  (client as unknown as { child: FakeChild }).child = child;
  await assert.rejects(client.dispose(), /did not exit after termination/i);
  assert.equal(child.stdin.destroyed, true);
  assert.equal(child.stdout.destroyed, true);
  assert.equal(child.stderr.destroyed, true);
});

test("StdioMcpClient launches npx through cmd.exe on Windows", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  let windowsHide = false;
  let verbatim = false;
  const client = new StdioMcpClient(stdioServer(), {
    platform: "win32",
    comSpec: "C:\\Windows\\System32\\cmd.exe",
    killProcessTree: () => true,
    spawn: (capturedCommand, capturedArgs, options) => {
      command = capturedCommand;
      args = capturedArgs;
      windowsHide = options.windowsHide === true;
      verbatim = options.windowsVerbatimArguments === true;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "C:\\Windows\\System32\\cmd.exe");
  assert.deepEqual(args, ["/d", "/s", "/c", '"npx -y @example/mcp-docs"']);
  assert.equal(windowsHide, true);
  assert.equal(verbatim, true);
  await client.dispose();
});

test("StdioMcpClient launches a .cmd shim through cmd.exe on Windows", async () => {
  const { child } = makeFakeChild();
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "C:\\tools\\mcp-docs.cmd", args: ["--stdio"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    killProcessTree: () => true,
    spawn: (_command, capturedArgs) => {
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.deepEqual(args, ["/d", "/s", "/c", '"C:\\tools\\mcp-docs.cmd --stdio"']);
  await client.dispose();
});

test("StdioMcpClient quotes a spaced .cmd path so cmd.exe keeps it intact", async () => {
  const { child } = makeFakeChild();
  let args: string[] = [];
  let verbatim = false;
  const client = new StdioMcpClient(
    stdioServer({ command: "C:\\Program Files\\nodejs\\npx.cmd", args: ["-y", "@example/mcp-docs"] }),
    {
      platform: "win32",
      comSpec: "cmd.exe",
      killProcessTree: () => true,
      spawn: (_command, capturedArgs, options) => {
        args = capturedArgs;
        verbatim = options.windowsVerbatimArguments === true;
        return child as never;
      },
    }
  );
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.deepEqual(args, ["/d", "/s", "/c", '""C:\\Program Files\\nodejs\\npx.cmd" -y @example/mcp-docs"']);
  assert.equal(verbatim, true);
  await client.dispose();
});

test("StdioMcpClient quotes an argument that itself contains spaces", async () => {
  const { child } = makeFakeChild();
  let args: string[] = [];
  const client = new StdioMcpClient(
    stdioServer({ command: "npx", args: ["-y", "@example/mcp-docs", "--prompt", "hello world"] }),
    {
      platform: "win32",
      comSpec: "cmd.exe",
      killProcessTree: () => true,
      spawn: (_command, capturedArgs) => {
        args = capturedArgs;
        return child as never;
      },
    }
  );
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.deepEqual(args, ["/d", "/s", "/c", '"npx -y @example/mcp-docs --prompt "hello world""']);
  await client.dispose();
});

test("StdioMcpClient preserves an empty-string argument in the cmd.exe line", async () => {
  const { child } = makeFakeChild();
  let args: string[] = [];
  const client = new StdioMcpClient(
    stdioServer({ command: "npx", args: ["-y", "@example/mcp-docs", "--suffix", ""] }),
    {
      platform: "win32",
      comSpec: "cmd.exe",
      killProcessTree: () => true,
      spawn: (_command, capturedArgs) => {
        args = capturedArgs;
        return child as never;
      },
    }
  );
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  // The empty token must survive as "" rather than vanishing into the join and
  // shifting the child's positional argv.
  assert.deepEqual(args, ["/d", "/s", "/c", '"npx -y @example/mcp-docs --suffix """']);
  await client.dispose();
});

test("StdioMcpClient spawns a real executable directly on Windows", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "node", args: ["server.js"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    killProcessTree: () => true,
    spawn: (capturedCommand, capturedArgs) => {
      command = capturedCommand;
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "node");
  assert.deepEqual(args, ["server.js"]);
  await client.dispose();
});

test("StdioMcpClient rejects cmd.exe metacharacters in client-supplied args before spawning", async () => {
  let spawned = false;
  const client = new StdioMcpClient(stdioServer({ args: ["-y", "@example/mcp-docs & calc.exe"] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => {
      spawned = true;
      return makeFakeChild().child as never;
    },
  });

  await assert.rejects(() => client.listTools(), /unsafe .*cmd\.exe/i);
  assert.equal(spawned, false);
});

test("StdioMcpClient rejects cmd.exe metacharacters in the command before spawning", async () => {
  let spawned = false;
  const client = new StdioMcpClient(stdioServer({ command: "C:\\tools & calc.exe\\mcp-docs.cmd", args: [] }), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => {
      spawned = true;
      return makeFakeChild().child as never;
    },
  });

  await assert.rejects(() => client.listTools(), /unsafe .*cmd\.exe/i);
  assert.equal(spawned, false);
});

test("StdioMcpClient allows cmd.exe metacharacters on POSIX where no shell is involved", async () => {
  const { child } = makeFakeChild();
  let command = "";
  let args: string[] = [];
  const client = new StdioMcpClient(stdioServer({ command: "npx", args: ["-y", "@example/weird&name"] }), {
    platform: "linux",
    spawn: (capturedCommand, capturedArgs) => {
      command = capturedCommand;
      args = capturedArgs;
      return child as never;
    },
  });
  const listPromise = client.listTools();

  await tick();
  child.emit("error", new Error("test stop"));
  await assert.rejects(listPromise, /test stop/i);

  assert.equal(command, "npx");
  assert.deepEqual(args, ["-y", "@example/weird&name"]);
  await client.dispose();
});

test("StdioMcpClient drains stderr and redacts secret env values on failure", async () => {
  const secret = "sk-super-secret-token";
  const { child, pushStderr } = makeFakeChild();
  const client = new StdioMcpClient(
    stdioServer({
      env: [
        { name: "DOCS_API_KEY", value: secret },
        { name: "DOCS_LOCALE", value: "en-US" },
      ],
    }),
    { spawn: () => child as never }
  );
  const listPromise = client.listTools();

  await tick();
  pushStderr(`auth failed for key ${secret} (locale en-US)\n`);
  child.emit("exit", 1, null);

  await assert.rejects(listPromise, (error: Error) => {
    assert.match(error.message, /stderr: auth failed for key/);
    assert.match(error.message, /\[REDACTED\]/);
    assert.doesNotMatch(error.message, new RegExp(secret));
    assert.match(error.message, /locale en-US/);
    return true;
  });
  await client.dispose();
});

test("StdioMcpClient redacts inherited environment secrets from stderr", async () => {
  const inherited = "inherited-parent-credential";
  const suffixed = "suffixed-db-password";
  process.env["GLM_TEST_UPSTREAM_TOKEN"] = inherited;
  process.env["GLM_TEST_DB_PWD"] = suffixed;
  try {
    const { child, pushStderr } = makeFakeChild();
    const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
    const listPromise = client.listTools();

    await tick();
    // The child inherits process.env, so a parent-held credential can reach its stderr.
    pushStderr(`upstream rejected ${inherited} / ${suffixed} in ${process.cwd()}\n`);
    child.emit("exit", 1, null);

    await assert.rejects(listPromise, (error: Error) => {
      assert.match(error.message, /\[REDACTED\]/);
      assert.doesNotMatch(error.message, new RegExp(inherited));
      // A `_PWD`-suffixed name is a credential (MYSQL_PWD and friends), not a path.
      assert.doesNotMatch(error.message, new RegExp(suffixed));
      // Bare PWD/OLDPWD are exempt, or every diagnostic loses its cwd.
      assert.match(error.message, new RegExp(process.cwd().replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
      return true;
    });
    await client.dispose();
  } finally {
    delete process.env["GLM_TEST_UPSTREAM_TOKEN"];
    delete process.env["GLM_TEST_DB_PWD"];
  }
});

test("StdioMcpClient terminates the Windows process tree on dispose", async () => {
  const { child, getKillCount } = makeFakeChild();
  let terminatedPid: number | undefined;
  const client = new StdioMcpClient(stdioServer(), {
    platform: "win32",
    comSpec: "cmd.exe",
    spawn: () => child as never,
    killProcessTree: (pid) => {
      terminatedPid = pid;
      return true;
    },
  });
  const listPromise = client.listTools();

  await tick();
  await client.dispose();

  await assert.rejects(listPromise, /disposed/i);
  assert.equal(terminatedPid, 5150);
  assert.equal(getKillCount(), 0);
});

test("StdioMcpClient keeps a shared initialization alive when one caller aborts", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });

  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  const survivingCall = client.callTool("search", { q: "surviving" });

  await tick();
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled/i);
  assert.equal(getKillCount(), 0);

  // Only the handshake is shared; each caller issues its own tools/call afterwards.
  const init = JSON.parse(written[0]?.trim() ?? "{}") as { id: number };
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: init.id, result: { protocolVersion: "2025-06-18" } }) + "\n");
  await tick();
  const callBody = JSON.parse(written.at(-1)?.trim() ?? "{}") as {
    id: number;
    params: { arguments: { q: string } };
  };
  assert.equal(callBody.params.arguments.q, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [] } }) + "\n");

  await assert.doesNotReject(survivingCall);
  await client.dispose();
});

test("StdioMcpClient does not kill an initialized server when a caller aborts", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });
  const listPromise = client.listTools();
  await completeHandshake(written, pushStdout);
  await listPromise;

  const controller = new AbortController();
  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled|aborted/i);
  assert.equal(getKillCount(), 0);

  // The server is still usable for the next caller.
  const survivingCall = client.callTool("search", { q: "surviving" });
  await tick();
  const callBody = JSON.parse(written.at(-1)?.trim() ?? "{}") as {
    id: number;
    params: { arguments: { q: string } };
  };
  assert.equal(callBody.params.arguments.q, "surviving");
  pushStdout(JSON.stringify({ jsonrpc: "2.0", id: callBody.id, result: { content: [] } }) + "\n");
  await assert.doesNotReject(survivingCall);

  await client.dispose();
});

test("StdioMcpClient keeps a shared initialization alive for a concurrent listTools", async () => {
  const { child, written, pushStdout, getKillCount } = makeFakeChild();
  const controller = new AbortController();
  const client = new StdioMcpClient(stdioServer(), { spawn: () => child as never });

  const cancelledCall = client.callTool("search", { q: "cancelled" }, controller.signal);
  const listPromise = client.listTools();

  await tick();
  controller.abort();
  await assert.rejects(cancelledCall, /cancelled/i);
  assert.equal(getKillCount(), 0);

  await completeHandshake(written, pushStdout);
  assert.deepEqual(await listPromise, [{ name: "search", description: undefined, inputSchema: undefined }]);
  await client.dispose();
});

test("StdioMcpClient does not resurrect a disposed server", async () => {
  const { child } = makeFakeChild();
  let spawnCount = 0;
  const client = new StdioMcpClient(stdioServer(), {
    spawn: () => {
      spawnCount += 1;
      return child as never;
    },
  });

  await client.dispose();
  await assert.rejects(() => client.callTool("search", {}), /disposed/i);
  assert.equal(spawnCount, 0);
});

test("HTTP dispose aborts its DELETE request once disposal returns", async () => {
  const originalFetch = globalThis.fetch;
  // A server can return DELETE headers while leaving the body pending. The
  // fetch promise settles on headers, so the only observable contract of the
  // fix is that the request signal is aborted by the time dispose() returns.
  let deleteSignal: AbortSignal | undefined;
  globalThis.fetch = (async (_url, init) => {
    const body = JSON.parse(String(init?.body ?? "{}"));
    if (body.method === "initialize") {
      return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: {} }), {
        headers: { "Content-Type": "application/json", "MCP-Session-Id": "session-delete" },
      });
    }
    if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
    if (init?.method === "DELETE") {
      deleteSignal = init.signal ?? undefined;
      return new Response(new ReadableStream<Uint8Array>({
        pull() { return new Promise<void>(() => {}); },
      }, { highWaterMark: 0 }), { status: 200 });
    }
    return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id,
      result: body.method === "tools/list" ? { tools: [{ name: "read" }] } : {},
    }), { headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  try {
    const tools = await connectSessionMcpServers([{ type: "http", name: "fixture", url: "https://fixture.invalid", headers: [] }]);
    await tools.dispose();
    assert.ok(deleteSignal, "a DELETE request was made");
    assert.equal(deleteSignal.aborted, true, "dispose must abort the DELETE request so its body cannot hold resources");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

for (const scenario of ["within a page", "across pages"] as const) {
  test(`HTTP discovery rejects duplicate tool names ${scenario}`, async () => {
    const savedFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async (_url, init) => {
        const body = JSON.parse(String(init?.body ?? "{}"));
        if (body.method === "notifications/initialized") return new Response(null, { status: 202 });
        const page = scenario === "within a page"
          ? { tools: [{ name: "dup" }, { name: "dup" }] }
          : { tools: [{ name: "dup" }], nextCursor: "page2" };
        if (body.method === "tools/list" && body.params?.cursor === "page2") {
          return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: { tools: [{ name: "dup" }] } }),
            { headers: { "Content-Type": "application/json" } });
        }
        return new Response(JSON.stringify({ jsonrpc: "2.0", id: body.id, result: page }),
          { headers: { "Content-Type": "application/json" } });
      }) as typeof fetch;
      await assert.rejects(
        connectSessionMcpServers([{ type: "http", name: "fixture", url: "https://fixture.invalid", headers: [] }]),
        /duplicate tool name/i,
      );
    } finally { globalThis.fetch = savedFetch; }
  });
}
