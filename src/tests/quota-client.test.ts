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
    resolveKey: () => "test-key", resolveTeam: () => ({}), now: () => time,
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
  const client = new QuotaClient({ resolveKey: () => "key", resolveTeam: () => ({}), now: () => time,
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
  const client = new QuotaClient({ resolveKey: () => key, resolveTeam: () => ({}), baseUrl: () => base,
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
    const client = new QuotaClient({ baseUrl: () => base, resolveKey: () => "private-key", resolveTeam: () => ({}),
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
    const client = new QuotaClient({ resolveKey: () => "private-key", resolveTeam: () => ({}), cacheMs: 0,
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
    const client = new QuotaClient({ resolveKey: () => "private-key", resolveTeam: () => ({}), fetch: async () => response() });
    const result = await client.getUsage();
    assert.equal(result.unavailable?.reason, reason);
    assert.deepEqual(result.windows, []);
    assert.doesNotMatch(JSON.stringify(result), /private/);
  }
});

test("bounds a stalled response body with the request deadline", async () => {
  let cancelled = false;
  const client = new QuotaClient({ resolveKey: () => "key", resolveTeam: () => ({}), timeoutMs: 20,
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
  const client = new QuotaClient({ resolveKey: () => "key", resolveTeam: () => ({}), signal: () => controller.signal,
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
  const client = new QuotaClient({ resolveKey: () => "key", resolveTeam: () => ({}), fetch: async () => json(payload()) });
  const text = formatQuotaUsage(await client.getUsage());
  assert.match(text, /25% used/);
  assert.match(text, /Remaining: 1500 credits/);
  assert.match(text, /Used: 500 \/ 2000 credits/);
  assert.match(text, /Resets: 2026-10-25T10:00:00.000Z/);
  assert.match(text, /Last updated:/);
});

const ORG = "org-placeholder-0001";
const PROJECT = "proj-placeholder-0002";
const teamReply = () => ({
  code: 200, success: true,
  data: {
    level: "max",
    limits: [
      { type: "CREDIT_LIMIT", unit: 3, number: 5, usage: 4000, currentValue: 1000, remaining: 3000, percentage: 25, nextResetTime: reset },
      { type: "CREDIT_LIMIT", unit: 6, number: 1, usage: 40000, currentValue: 30000, remaining: 10000, percentage: 75, nextResetTime: reset + 6 * 86_400_000 },
    ],
  },
});

test("personal keys send the unchanged request when no team seat is configured", async () => {
  const seen: { url: string; headers: [string, string][] }[] = [];
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}),
    fetch: async (url, init) => {
      seen.push({ url: String(url), headers: [...new Headers(init?.headers)].sort() });
      assert.equal(init?.redirect, "error");
      return json(payload());
    },
  });
  await client.getUsage();
  assert.deepEqual(seen, [{
    url: "https://api.z.ai/api/monitor/usage/quota/limit",
    headers: [["accept", "application/json"], ["authorization", "test-key"]],
  }]);
});

test("team seats add type=2 and the organization and project headers", async () => {
  const seen: { url: string; init?: RequestInit }[] = [];
  const client = new QuotaClient({ resolveKey: () => "test-key",
    resolveTeam: () => ({ organizationId: ` ${ORG} `, projectId: PROJECT }),
    fetch: async (url, init) => { seen.push({ url: String(url), init }); return json(teamReply()); },
  });
  const result = await client.getUsage();
  assert.equal(result.unavailable, undefined);
  assert.equal(seen.length, 1);
  assert.equal(seen[0]!.url, "https://api.z.ai/api/monitor/usage/quota/limit?type=2");
  const headers = new Headers(seen[0]!.init?.headers);
  assert.equal(headers.get("Authorization"), "test-key");
  assert.equal(headers.get("bigmodel-organization"), ORG);
  assert.equal(headers.get("bigmodel-project"), PROJECT);
  assert.equal(seen[0]!.init?.redirect, "error");
});

test("team seats keep the allowed-host rule", async () => {
  const client = new QuotaClient({ baseUrl: () => "https://example.com", resolveKey: () => "private-key",
    resolveTeam: () => ({ organizationId: ORG, projectId: PROJECT }),
    fetch: async () => { throw new Error("must not fetch"); },
  });
  const result = await client.getUsage();
  assert.equal(result.unavailable?.reason, "unsupportedEndpoint");
});

test("a half-configured or malformed team seat is unavailable and sends nothing", async () => {
  for (const seat of [{ organizationId: ORG }, { projectId: PROJECT }, { organizationId: ORG, projectId: "  " },
    { organizationId: `${ORG}\r\nx-injected: 1`, projectId: PROJECT }, { organizationId: ORG, projectId: "has space" }]) {
    let calls = 0;
    const client = new QuotaClient({ resolveKey: () => "private-key", resolveTeam: () => seat,
      fetch: async () => { calls++; return json(payload()); },
    });
    const result = await client.getUsage();
    assert.equal(calls, 0);
    assert.equal(result.unavailable?.reason, "missingCredentials");
    assert.match(result.unavailable!.message, /Z_AI_ORGANIZATION_ID/);
    assert.deepEqual(result.windows, []);
    assert.doesNotMatch(JSON.stringify(result), /private-key|placeholder|injected|has space/);
  }
});

test("changing the team seat invalidates cached quota", async () => {
  let seat: { organizationId?: string; projectId?: string } = {};
  const requests: string[] = [];
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => seat,
    fetch: async (url, init) => {
      const headers = new Headers(init?.headers);
      requests.push(`${String(url)} ${headers.get("bigmodel-organization")} ${headers.get("bigmodel-project")}`);
      return json(teamReply());
    },
  });
  await client.getUsage();
  await client.getUsage();
  assert.equal(requests.length, 1); // cached within the minute
  seat = { organizationId: ORG, projectId: PROJECT };
  await client.getUsage();
  seat = { organizationId: ORG, projectId: "proj-placeholder-0003" };
  await client.getUsage();
  await client.getUsage();
  seat = {};
  await client.getUsage();
  assert.deepEqual(requests, [
    "https://api.z.ai/api/monitor/usage/quota/limit null null",
    `https://api.z.ai/api/monitor/usage/quota/limit?type=2 ${ORG} ${PROJECT}`,
    `https://api.z.ai/api/monitor/usage/quota/limit?type=2 ${ORG} proj-placeholder-0003`,
    "https://api.z.ai/api/monitor/usage/quota/limit null null",
  ]);
});

test("parses a team reply with 5-hour and 7-day credit windows", async () => {
  const client = new QuotaClient({ resolveKey: () => "test-key",
    resolveTeam: () => ({ organizationId: ORG, projectId: PROJECT }),
    fetch: async () => json(teamReply()),
  });
  const { windows } = await client.getUsage();
  assert.deepEqual(windows, [{
    id: "credit_limit-3-5", label: "5-hour Model quota", kind: "session",
    usedPercent: 25, windowDurationMins: 300, used: 1000, limit: 4000, remaining: 3000,
    unit: "credits", resetsAt: "2026-10-25T10:00:00.000Z",
  }, {
    id: "credit_limit-6-1", label: "1-week Model quota", kind: "weekly",
    usedPercent: 75, windowDurationMins: 7 * 24 * 60, used: 30000, limit: 40000, remaining: 10000,
    unit: "credits", resetsAt: "2026-10-31T10:00:00.000Z",
  }]);
});

test("team-seat business failures point at the seat settings without echoing IDs", async () => {
  const client = new QuotaClient({ resolveKey: () => "private-key",
    resolveTeam: () => ({ organizationId: ORG, projectId: PROJECT }),
    fetch: async () => json({ code: 500, success: false, msg: "private" }),
  });
  const result = await client.getUsage();
  assert.equal(result.unavailable?.reason, "requestFailed");
  assert.match(result.unavailable!.message, /team-seat quota/);
  assert.doesNotMatch(JSON.stringify(result), /private|placeholder/);
});

const QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit";
const CUSTOMER_URL = "https://api.z.ai/api/biz/customer/getCustomerInfo";
const noPlan = () => ({ code: 500, success: false, msg: "private no coding plan" });
const project = (projectId: string, organizationId: string, projectType: number, status = "active") =>
  ({ projectId, organizationId, projectType, status, projectName: "Private Project Name" });
const customerInfo = (organizations: unknown[]) => ({
  code: 200, success: true,
  data: { email: "private-person@example.invalid", phone: "+00-private-phone", customerNumber: "private-cust-001", organizations },
});
const personalAndTeam = () => customerInfo([
  { organizationId: "org-personal-placeholder", organizationName: "Private Personal Org", isDefault: true,
    projects: [project("proj-personal-placeholder", "org-personal-placeholder", 1)] },
  { organizationId: ORG, organizationName: "Private Team Org", isDefault: false, projects: [project(PROJECT, ORG, 2)] },
]);
const PRIVATE = /private|placeholder|example\.invalid/i;

type Seen = { url: string; headers: Headers; init?: RequestInit };
/** Route mocked requests by URL and record each request's URL, headers, and init. */
function discoveryFetch(routes: { quota?: () => Response; customer?: () => Response; team?: (seen: Seen) => Response }) {
  const seen: Seen[] = [];
  const fetch = async (url: string | URL | Request, init?: RequestInit) => {
    const entry = { url: String(url), headers: new Headers(init?.headers), init };
    seen.push(entry);
    assert.equal(init?.redirect, "error");
    assert.equal(entry.headers.get("Authorization"), "test-key");
    assert.equal(entry.headers.get("Accept"), "application/json");
    if (entry.url === QUOTA_URL) return (routes.quota ?? (() => json(noPlan())))();
    if (entry.url === CUSTOMER_URL) return (routes.customer ?? (() => json(personalAndTeam())))();
    if (entry.url === `${QUOTA_URL}?type=2`) return (routes.team ?? (() => json(teamReply())))(entry);
    throw new Error(`unexpected ${entry.url}`);
  };
  return { seen, fetch };
}

test("discovers the single active team seat when the personal lookup reports no plan", async () => {
  let time = reset;
  const { seen, fetch } = discoveryFetch({});
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), now: () => time, fetch });
  const result = await client.getUsage();
  assert.equal(result.unavailable, undefined);
  assert.deepEqual(seen.map(s => s.url), [QUOTA_URL, CUSTOMER_URL, `${QUOTA_URL}?type=2`]);
  assert.equal(seen[0]!.headers.get("bigmodel-organization"), null);
  assert.equal(seen[1]!.headers.get("bigmodel-organization"), null);
  assert.equal(seen[2]!.headers.get("bigmodel-organization"), ORG);
  assert.equal(seen[2]!.headers.get("bigmodel-project"), PROJECT);
  assert.ok(seen.every(s => s.init?.signal === seen[0]!.init?.signal), "one deadline covers the whole refresh");
  assert.deepEqual(result.windows.map(w => [w.id, w.windowDurationMins, w.usedPercent]),
    [["credit_limit-3-5", 300, 25], ["credit_limit-6-1", 10080, 75]]);
  assert.doesNotMatch(JSON.stringify(result), PRIVATE);

  await client.getUsage();
  assert.equal(seen.length, 3); // cached within the minute
  time += 60_000;
  assert.equal((await client.getUsage()).unavailable, undefined);
  assert.deepEqual(seen.slice(3).map(s => s.url), [`${QUOTA_URL}?type=2`]);
  assert.equal(seen[3]!.headers.get("bigmodel-project"), PROJECT);
});

test("discovery uses the configured allowed origin", async () => {
  const urls: string[] = [];
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}),
    baseUrl: () => "https://open.bigmodel.cn/api/coding/paas/v4",
    fetch: async url => {
      urls.push(String(url));
      return json(urls.length === 1 ? noPlan() : urls.length === 2 ? personalAndTeam() : teamReply());
    },
  });
  assert.equal((await client.getUsage()).unavailable, undefined);
  assert.deepEqual(urls, [
    "https://open.bigmodel.cn/api/monitor/usage/quota/limit",
    "https://open.bigmodel.cn/api/biz/customer/getCustomerInfo",
    "https://open.bigmodel.cn/api/monitor/usage/quota/limit?type=2",
  ]);
});

test("discovery skips inactive and non-team projects in favor of the single active team project", async () => {
  const { seen, fetch } = discoveryFetch({ customer: () => json(customerInfo([
    { organizationId: "org-old-placeholder", projects: [project("proj-old-placeholder", "org-old-placeholder", 2, "disabled")] },
    // The project's organization ID is missing, so it comes from the organization; IDs are trimmed.
    { organizationId: ` ${ORG} `, projects: [{ projectId: ` ${PROJECT} `, projectType: 2, status: "ACTIVE" },
      project(PROJECT, ORG, 2), project("proj-personal-placeholder", ORG, 1), { projectId: "bad id", projectType: 2 }, null] },
    "not-an-org",
  ])) });
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), fetch });
  assert.equal((await client.getUsage()).unavailable, undefined);
  assert.equal(seen.length, 3);
  assert.equal(seen[2]!.headers.get("bigmodel-organization"), ORG);
  assert.equal(seen[2]!.headers.get("bigmodel-project"), PROJECT);
});

test("multiple active team projects ask for an explicit seat and send no team request", async () => {
  const { seen, fetch } = discoveryFetch({ customer: () => json(customerInfo([
    { organizationId: ORG, projects: [project(PROJECT, ORG, 2), project("proj-placeholder-0003", ORG, 2)] },
  ])) });
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), fetch });
  const result = await client.getUsage();
  assert.deepEqual(seen.map(s => s.url), [QUOTA_URL, CUSTOMER_URL]);
  assert.equal(result.unavailable?.reason, "requestFailed");
  assert.match(result.unavailable!.message, /Multiple Z\.AI team projects/);
  assert.match(result.unavailable!.message, /Z_AI_ORGANIZATION_ID and Z_AI_PROJECT_ID/);
  assert.deepEqual(result.windows, []);
  assert.doesNotMatch(JSON.stringify(result), PRIVATE);
});

test("truncated organization or project lists decline automatic selection without calling type=2", async () => {
  const tooManyOrgs = customerInfo([
    { organizationId: ORG, projects: [project(PROJECT, ORG, 2)] },
    ...Array.from({ length: 63 }, (_, i) => ({
      organizationId: `org-personal-placeholder-${i}`,
      projects: [project(`proj-personal-placeholder-${i}`, `org-personal-placeholder-${i}`, 1)],
    })),
    { organizationId: "org-hidden-placeholder", projects: [project("proj-hidden-placeholder", "org-hidden-placeholder", 2)] },
  ]);
  const tooManyProjects = customerInfo([
    {
      organizationId: ORG,
      projects: [
        project(PROJECT, ORG, 2),
        ...Array.from({ length: 63 }, (_, i) => project(`proj-personal-placeholder-${i}`, ORG, 1)),
        project("proj-hidden-placeholder", ORG, 2),
      ],
    },
  ]);
  for (const reply of [tooManyOrgs, tooManyProjects]) {
    const { seen, fetch } = discoveryFetch({ customer: () => json(reply) });
    const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), cacheMs: 0, fetch });
    const result = await client.getUsage();
    assert.deepEqual(seen.map(s => s.url), [QUOTA_URL, CUSTOMER_URL]);
    assert.equal(result.unavailable?.reason, "requestFailed");
    assert.match(result.unavailable!.message, /Z_AI_ORGANIZATION_ID and Z_AI_PROJECT_ID/);
    assert.deepEqual(result.windows, []);
    assert.doesNotMatch(JSON.stringify(result), PRIVATE);
    // Confirm no seat was remembered for the next refresh.
    await client.getUsage();
    assert.deepEqual(seen.slice(2).map(s => s.url), [QUOTA_URL, CUSTOMER_URL]);
  }
});

test("discovery still auto-selects at the exact 64-organization and 64-project boundary", async () => {
  const atLimit = customerInfo([
    {
      organizationId: ORG,
      projects: [
        project(PROJECT, ORG, 2),
        ...Array.from({ length: 63 }, (_, i) => project(`proj-personal-placeholder-${i}`, ORG, 1)),
      ],
    },
    ...Array.from({ length: 63 }, (_, i) => ({
      organizationId: `org-personal-placeholder-${i}`,
      projects: [project(`proj-personal-placeholder-${i}`, `org-personal-placeholder-${i}`, 1)],
    })),
  ]);
  const { seen, fetch } = discoveryFetch({ customer: () => json(atLimit) });
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), fetch });
  const result = await client.getUsage();
  assert.equal(result.unavailable, undefined);
  assert.deepEqual(seen.map(s => s.url), [QUOTA_URL, CUSTOMER_URL, `${QUOTA_URL}?type=2`]);
  assert.equal(seen[2]!.headers.get("bigmodel-organization"), ORG);
  assert.equal(seen[2]!.headers.get("bigmodel-project"), PROJECT);
});

test("no team project or a failed discovery falls back to the original failure without leaking the reply", async () => {
  for (const customer of [
    () => json(customerInfo([{ organizationId: ORG, projects: [project("proj-personal-placeholder", ORG, 1)] }])),
    () => json(customerInfo([])),
    () => json({ ...personalAndTeam(), code: 500 }),
    () => json({ ...personalAndTeam(), success: false }),
    () => new Response(JSON.stringify(personalAndTeam()), { status: 500 }),
    () => new Response(JSON.stringify(personalAndTeam()), { status: 401 }),
    () => new Response("private-person@example.invalid not json"),
    () => { throw new Error("private-person@example.invalid network"); },
  ]) {
    const { seen, fetch } = discoveryFetch({ customer });
    const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), fetch });
    const result = await client.getUsage();
    assert.deepEqual(seen.map(s => s.url), [QUOTA_URL, CUSTOMER_URL]);
    assert.equal(result.unavailable?.reason, "requestFailed");
    assert.match(result.unavailable!.message, /could not return Coding Plan quota/);
    assert.doesNotMatch(JSON.stringify(result), PRIVATE);
  }
});

test("authentication, HTTP, and malformed personal lookups never call getCustomerInfo", async () => {
  for (const [quota, reason] of [
    [() => new Response("private", { status: 401 }), "authenticationFailed"],
    [() => new Response("private", { status: 403 }), "authenticationFailed"],
    [() => json({ code: 401, msg: "private" }), "authenticationFailed"],
    [() => new Response("private", { status: 500 }), "requestFailed"],
    [() => new Response("private", { status: 429 }), "requestFailed"],
    [() => new Response("private"), "invalidResponse"],
    [() => json(payload([])), "invalidResponse"],
    [() => { throw new Error("private network"); }, "requestFailed"],
  ] as const) {
    const { seen, fetch } = discoveryFetch({ quota });
    const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), fetch });
    assert.equal((await client.getUsage()).unavailable?.reason, reason);
    assert.deepEqual(seen.map(s => s.url), [QUOTA_URL]);
  }
});

test("an explicit team seat skips discovery entirely", async () => {
  const { seen, fetch } = discoveryFetch({ team: () => json(noPlan()) });
  const client = new QuotaClient({ resolveKey: () => "test-key",
    resolveTeam: () => ({ organizationId: ORG, projectId: PROJECT }), fetch });
  const result = await client.getUsage();
  assert.deepEqual(seen.map(s => s.url), [`${QUOTA_URL}?type=2`]);
  assert.equal(result.unavailable?.reason, "requestFailed");
  assert.match(result.unavailable!.message, /team-seat quota/);
});

test("a discovered seat is forgotten when the key, origin, or explicit seat changes", async () => {
  let key = "test-key";
  let base = "https://api.z.ai/api/coding/paas/v4";
  let seat: { organizationId?: string; projectId?: string } = {};
  const urls: string[] = [];
  const client = new QuotaClient({ resolveKey: () => key, baseUrl: () => base, resolveTeam: () => seat, cacheMs: 0,
    fetch: async (url, init) => {
      const path = new URL(String(url));
      const team = new Headers(init?.headers).get("bigmodel-project");
      urls.push(`${path.host}${path.pathname}${path.search}${team ? ` ${team}` : ""}`);
      if (path.pathname.endsWith("getCustomerInfo")) return json(personalAndTeam());
      return json(path.search ? teamReply() : noPlan());
    },
  });
  const probe = (host: string) => [`${host}/api/monitor/usage/quota/limit`, `${host}/api/biz/customer/getCustomerInfo`,
    `${host}/api/monitor/usage/quota/limit?type=2 ${PROJECT}`];
  await client.getUsage();
  await client.getUsage();
  assert.deepEqual(urls.splice(0), [...probe("api.z.ai"), `api.z.ai/api/monitor/usage/quota/limit?type=2 ${PROJECT}`]);
  key = "other-key";
  await client.getUsage();
  assert.deepEqual(urls.splice(0), probe("api.z.ai"));
  base = "https://open.bigmodel.cn/api/coding/paas/v4";
  await client.getUsage();
  assert.deepEqual(urls.splice(0), probe("open.bigmodel.cn"));
  seat = { organizationId: ORG, projectId: "proj-placeholder-0003" };
  await client.getUsage();
  assert.deepEqual(urls.splice(0), ["open.bigmodel.cn/api/monitor/usage/quota/limit?type=2 proj-placeholder-0003"]);
  seat = {};
  await client.getUsage();
  assert.deepEqual(urls.splice(0), probe("open.bigmodel.cn"));
});

test("a discovered seat that later fails is forgotten and rediscovered on the next refresh", async () => {
  let teamStatus = 200;
  const { seen, fetch } = discoveryFetch({
    team: () => teamStatus === 200 ? json(teamReply()) : new Response("private", { status: teamStatus }),
  });
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), cacheMs: 0, fetch });
  await client.getUsage();
  assert.equal(seen.length, 3);
  teamStatus = 503;
  const failed = await client.getUsage();
  assert.deepEqual(seen.slice(3).map(s => s.url), [`${QUOTA_URL}?type=2`]);
  assert.equal(failed.unavailable?.reason, "requestFailed");
  assert.equal(failed.stale, true);
  teamStatus = 200;
  assert.equal((await client.getUsage()).unavailable, undefined);
  assert.deepEqual(seen.slice(4).map(s => s.url), [QUOTA_URL, CUSTOMER_URL, `${QUOTA_URL}?type=2`]);
});

test("a discovered seat whose team lookup fails is not remembered", async () => {
  let attempt = 0;
  const { seen, fetch } = discoveryFetch({ team: () => ++attempt === 1 ? json(noPlan()) : json(teamReply()) });
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), cacheMs: 0, fetch });
  const first = await client.getUsage();
  assert.equal(first.unavailable?.reason, "requestFailed");
  assert.match(first.unavailable!.message, /team-seat quota/);
  assert.doesNotMatch(JSON.stringify(first), PRIVATE);
  await client.getUsage();
  assert.deepEqual(seen.slice(3).map(s => s.url), [QUOTA_URL, CUSTOMER_URL, `${QUOTA_URL}?type=2`]);
});

test("a deadline during discovery reports a timeout instead of the original failure", async () => {
  const client = new QuotaClient({ resolveKey: () => "test-key", resolveTeam: () => ({}), timeoutMs: 20,
    fetch: async (url, init) => {
      if (String(url) === QUOTA_URL) return json(noPlan());
      const signal = init!.signal!;
      return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(signal.reason), { once: true }));
    },
  });
  const keepAlive = setTimeout(() => {}, 1000);
  try {
    const result = await client.getUsage();
    assert.equal(result.unavailable?.reason, "requestFailed");
    assert.match(result.unavailable!.message, /timed out/);
  } finally { clearTimeout(keepAlive); }
});
