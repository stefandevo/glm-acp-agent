import test from "node:test";
import assert from "node:assert/strict";
import { QuotaClient, formatQuotaUsage, parseQuotaWindows } from "../llm/quota-client.js";

const reset = Date.parse("2026-10-25T10:00:00Z");
const credit = { type: "CREDIT_LIMIT", unit: 3, number: 5, usage: 2000, currentValue: 500, remaining: 1500, percentage: 25, nextResetTime: reset };
const payload = (limits: unknown[] = [credit]) => ({ code: 200, success: true, data: { limits } });
const json = (value: unknown) => Response.json(value);

test("normalizes credit, legacy model, and MCP quotas using provider durations", () => {
  const windows = parseQuotaWindows(payload([
    credit,
    { ...credit, number: 1, unit: 6, percentage: 40 },
    { type: "TOKENS_LIMIT", percentage: 12 },
    { type: "TIME_LIMIT", usage: 100, currentValue: 5 },
    { type: "FUTURE_LIMIT", percentage: 99 },
  ]));
  assert.deepEqual(windows[0], {
    id: "credit_limit-3-5", label: "5-hour Model quota", kind: "session",
    usedPercent: 25, windowDurationMins: 300, used: 500, limit: 2000, remaining: 1500,
    unit: "credits", resetsAt: "2026-10-25T10:00:00.000Z",
  });
  assert.equal(windows[1]?.kind, "weekly");
  assert.equal(windows[1]?.windowDurationMins, 10080);
  assert.equal(windows[2]?.unit, undefined); // Legacy quota counts are not necessarily raw tokens.
  assert.equal(windows[2]?.remaining, undefined);
  assert.equal(windows[2]?.windowDurationMins, undefined);
  assert.equal(windows[3]?.usedPercent, 5);
  assert.equal(windows[3]?.remaining, 95);
  assert.equal(windows[3]?.unit, "calls");
  assert.equal(windows.length, 4);
});

test("does not turn missing, invalid, or empty quota responses into zero usage", () => {
  for (const value of [null, {}, payload([]), payload([{ type: "CREDIT_LIMIT" }]),
    payload([{ type: "CREDIT_LIMIT", percentage: -1 }]), payload([{ type: "CREDIT_LIMIT", percentage: "50" }]),
    payload([{ type: "CREDIT_LIMIT", percentage: Infinity }]), payload([{ type: "NEW_LIMIT", percentage: 20 }]),
    payload(Array.from({ length: 65 }, () => credit))]) {
    assert.throws(() => parseQuotaWindows(value), /no usable quota/);
  }
});

test("keeps zero usage, bounds overruns, and omits invalid reset dates", () => {
  const [zero, overrun] = parseQuotaWindows(payload([
    { type: "CREDIT_LIMIT", percentage: 0, nextResetTime: 1e100 },
    { type: "CREDIT_LIMIT", usage: 100, currentValue: 120 },
  ]));
  assert.equal(zero?.usedPercent, 0);
  assert.equal(zero?.resetsAt, undefined);
  assert.equal(overrun?.usedPercent, 100);
  assert.equal(overrun?.remaining, 0);
  assert.notEqual(zero?.id, overrun?.id);
});

test("fetches once for concurrent callers, caches for a minute, and isolates caller mutations", async () => {
  let time = reset;
  let calls = 0;
  const client = new QuotaClient({
    resolveKey: () => "test-key", now: () => time,
    baseUrl: () => "https://api.z.ai/api/coding/paas/v4",
    fetch: async (url, init) => {
      calls++;
      assert.equal(url, "https://api.z.ai/api/monitor/usage/quota/limit");
      assert.equal(new Headers(init?.headers).get("Authorization"), "test-key");
      assert.equal(init?.redirect, "error");
      return json(payload());
    },
  });
  const [a, b] = await Promise.all([client.getUsage(), client.getUsage()]);
  assert.equal(calls, 1);
  a.windows[0]!.usedPercent = 99;
  assert.equal(b.windows[0]?.usedPercent, 25);
  assert.equal((await client.getUsage()).windows[0]?.usedPercent, 25);
  assert.equal(calls, 1);
  time += 60_000;
  await client.getUsage();
  assert.equal(calls, 2);
});

test("a failed refresh retains dated stale data and caches failures to avoid repeated probes", async () => {
  let time = reset;
  let calls = 0;
  const client = new QuotaClient({ resolveKey: () => "key", now: () => time,
    fetch: async () => { if (++calls > 1) throw new Error("secret provider response"); return json(payload()); },
  });
  const first = await client.getUsage();
  time += 60_000;
  const stale = await client.getUsage();
  assert.equal(stale.stale, true);
  assert.equal(stale.checkedAt, first.checkedAt);
  assert.deepEqual(stale.windows, first.windows);
  assert.equal(stale.unavailable?.reason, "requestFailed");
  assert.doesNotMatch(JSON.stringify(stale), /secret/);
  await client.getUsage();
  assert.equal(calls, 2);
  assert.match(formatQuotaUsage(stale), /stale data/);
});

test("credential or region changes invalidate cached data", async () => {
  let key: string | undefined = "first";
  let base = "https://api.z.ai/api/coding/paas/v4";
  const urls: string[] = [];
  const client = new QuotaClient({ resolveKey: () => key, baseUrl: () => base,
    fetch: async url => { urls.push(String(url)); return urls.length === 1 ? json(payload()) : new Response("private", { status: 503 }); },
  });
  await client.getUsage();
  key = "second";
  const changed = await client.getUsage();
  assert.equal(changed.stale, false);
  assert.deepEqual(changed.windows, []);
  base = "https://open.bigmodel.cn/api/coding/paas/v4";
  await client.getUsage();
  assert.equal(urls[2], "https://open.bigmodel.cn/api/monitor/usage/quota/limit");
  key = undefined;
  assert.equal((await client.getUsage()).unavailable?.reason, "missingCredentials");
  assert.equal(urls.length, 3);
});

test("never sends a key to custom, insecure, credential-bearing, or lookalike origins", async () => {
  for (const base of ["http://api.z.ai", "https://api.z.ai.attacker.invalid", "https://example.com",
    "https://user:pass@api.z.ai", "https://api.z.ai:8443", "invalid"]) {
    const client = new QuotaClient({ baseUrl: () => base, resolveKey: () => "private-key",
      fetch: async () => { throw new Error("must not fetch"); },
    });
    const result = await client.getUsage();
    assert.equal(result.unavailable?.reason, "unsupportedEndpoint");
    assert.doesNotMatch(JSON.stringify(result), /private-key|user:pass|attacker/);
  }
});

test("handles HTTP and business authentication errors and clears previous quota", async () => {
  for (const response of [() => new Response("private", { status: 401 }), () => new Response("private", { status: 403 }),
    () => json({ code: 401, msg: "private" }), () => json({ code: "403", success: false, msg: "private" })]) {
    let calls = 0;
    const client = new QuotaClient({ resolveKey: () => "private-key", cacheMs: 0,
      fetch: async () => ++calls === 1 ? json(payload()) : response(),
    });
    await client.getUsage();
    const result = await client.getUsage();
    assert.equal(result.unavailable?.reason, "authenticationFailed");
    assert.deepEqual(result.windows, []);
    assert.equal(result.stale, false);
    assert.doesNotMatch(JSON.stringify(result), /private/);
  }
});

test("handles rate limiting, business failures, malformed JSON, and oversized bodies without leaking responses", async () => {
  for (const [response, reason] of [
    [() => new Response("private", { status: 429 }), "requestFailed"],
    [() => json({ code: 500, success: false, msg: "private" }), "requestFailed"],
    [() => new Response("private"), "invalidResponse"],
    [() => new Response("x".repeat(256 * 1024 + 1)), "invalidResponse"],
  ] as const) {
    const client = new QuotaClient({ resolveKey: () => "private-key", fetch: async () => response() });
    const result = await client.getUsage();
    assert.equal(result.unavailable?.reason, reason);
    assert.deepEqual(result.windows, []);
    assert.doesNotMatch(JSON.stringify(result), /private/);
  }
});

test("bounds a stalled response body with the request deadline", async () => {
  let cancelled = false;
  const client = new QuotaClient({ resolveKey: () => "key", timeoutMs: 20,
    fetch: async () => new Response(new ReadableStream({ cancel() { cancelled = true; } })),
  });
  // AbortSignal.timeout is unref'ed; keep the test alive while the response stalls.
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    assert.equal((await client.getUsage()).unavailable?.reason, "requestFailed");
    assert.equal(cancelled, true);
  } finally { clearTimeout(keepAlive); }
});

test("transport disconnect aborts a shared in-flight quota request", async () => {
  const controller = new AbortController();
  let started!: () => void;
  const ready = new Promise<void>(resolve => { started = resolve; });
  const client = new QuotaClient({ resolveKey: () => "key", signal: () => controller.signal,
    fetch: async (_url, init) => {
      const signal = init!.signal!;
      started();
      return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    },
  });
  const pending = client.getUsage();
  await ready;
  controller.abort();
  assert.equal((await pending).unavailable?.reason, "requestFailed");
});

test("formats actual usage, remaining credits, resets, and timestamps", async () => {
  const client = new QuotaClient({ resolveKey: () => "key", fetch: async () => json(payload()) });
  const text = formatQuotaUsage(await client.getUsage());
  assert.match(text, /25% used/);
  assert.match(text, /Remaining: 1500 credits/);
  assert.match(text, /Used: 500 \/ 2000 credits/);
  assert.match(text, /Resets: 2026-10-25T10:00:00.000Z/);
  assert.match(text, /Last updated:/);
});
