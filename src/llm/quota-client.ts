import { resolveApiKey } from "./credentials.js";

export interface QuotaWindow {
  id: string;
  label: string;
  kind: "session" | "weekly" | "monthly" | "other";
  usedPercent: number;
  windowDurationMins?: number;
  used?: number;
  limit?: number;
  remaining?: number;
  unit?: "credits" | "calls";
  resetsAt?: string;
}

export interface QuotaSnapshot {
  version: 1;
  /** Time of the last successful read, or the attempt when no data is available. */
  checkedAt: string;
  stale: boolean;
  windows: QuotaWindow[];
  unavailable?: {
    reason: "missingCredentials" | "unsupportedEndpoint" | "authenticationFailed" | "requestFailed" | "invalidResponse";
    message: string;
  };
}

interface QuotaClientOptions {
  fetch?: typeof fetch;
  resolveKey?: () => string | undefined;
  baseUrl?: () => string;
  now?: () => number;
  cacheMs?: number;
  timeoutMs?: number;
  signal?: () => AbortSignal | undefined;
}

const DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4";
const MAX_RESPONSE_BYTES = 256 * 1024;
const MAX_WINDOWS = 64;

class QuotaError extends Error {
  constructor(readonly reason: NonNullable<QuotaSnapshot["unavailable"]>["reason"], message: string) {
    super(message);
  }
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

function nonNegative(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function invalidResponse(): never {
  throw new QuotaError("invalidResponse", "Z.AI returned no usable quota data.");
}

/** Normalize provider values without guessing plan allowances or treating missing data as zero. */
export function parseQuotaWindows(value: unknown): QuotaWindow[] {
  const envelope = record(value);
  if (!envelope) return invalidResponse();
  if (envelope.code === 401 || envelope.code === 403 || envelope.code === "401" || envelope.code === "403") {
    throw new QuotaError("authenticationFailed", "Z.AI rejected the quota credentials. Check your Coding Plan API key.");
  }
  if (envelope.success === false || (envelope.code !== undefined && ![0, 200, "0", "200"].includes(envelope.code as number | string))) {
    throw new QuotaError("requestFailed", "Z.AI could not return Coding Plan quota. Check your subscription in the Z.AI dashboard.");
  }
  const limits = record(envelope.data)?.limits;
  if (!Array.isArray(limits) || limits.length > MAX_WINDOWS) return invalidResponse();
  const windows: QuotaWindow[] = [];
  const ids = new Map<string, number>();
  for (const raw of limits) {
    const row = record(raw);
    if (!row || !["CREDIT_LIMIT", "TOKENS_LIMIT", "TIME_LIMIT"].includes(row.type as string)) continue;
    const used = nonNegative(row.currentValue);
    const limit = nonNegative(row.usage);
    const percentage = nonNegative(row.percentage) ?? (
      used !== undefined && limit !== undefined && limit > 0 ? used / limit * 100 : undefined
    );
    if (percentage === undefined) return invalidResponse();
    const remaining = nonNegative(row.remaining) ?? (
      used !== undefined && limit !== undefined ? Math.max(0, limit - used) : undefined
    );
    // Known provider duration units: hour=3, day=4, week=6. Leave others unspecified.
    const mins = ({ 3: 60, 4: 1440, 6: 10080 } as Record<string, number>)[String(row.unit)];
    const count = nonNegative(row.number);
    const duration = mins && count && Number.isSafeInteger(count) && Number.isSafeInteger(mins * count)
      ? mins * count : undefined;
    const baseId = `${row.type}-${nonNegative(row.unit) ?? "unknown"}-${count ?? "unknown"}`.toLowerCase();
    const occurrence = ids.get(baseId) ?? 0;
    ids.set(baseId, occurrence + 1);
    const toolQuota = row.type === "TIME_LIMIT";
    const durationLabel = duration === undefined ? "" : duration % 10080 === 0
      ? `${duration / 10080}-week ` : duration % 1440 === 0 ? `${duration / 1440}-day ` : `${duration / 60}-hour `;
    const reset = nonNegative(row.nextResetTime);
    const date = reset !== undefined && reset > 0 ? new Date(reset) : undefined;
    windows.push({
      id: occurrence ? `${baseId}-${occurrence}` : baseId,
      label: `${durationLabel}${toolQuota ? "MCP" : "Model"} quota`,
      kind: duration === undefined ? "other" : duration >= 43200 ? "monthly" : duration >= 10080 ? "weekly" : "session",
      usedPercent: Math.min(100, percentage),
      ...(duration !== undefined ? { windowDurationMins: duration } : {}),
      ...(used !== undefined ? { used } : {}),
      ...(limit !== undefined ? { limit } : {}),
      ...(remaining !== undefined ? { remaining } : {}),
      ...(row.type === "CREDIT_LIMIT" ? { unit: "credits" as const } : toolQuota ? { unit: "calls" as const } : {}),
      ...(date && Number.isFinite(date.getTime()) ? { resetsAt: date.toISOString() } : {}),
    });
  }
  if (!windows.length) return invalidResponse();
  return windows;
}

/** Only send quota credentials to supported provider origins, never an arbitrary model proxy. */
function quotaUrl(baseUrl: string): string {
  let url: URL;
  try { url = new URL(baseUrl); } catch { throw new QuotaError("unsupportedEndpoint", "Quota lookup requires a supported Z.AI or Zhipu HTTPS endpoint."); }
  if (url.protocol !== "https:" || url.username || url.password || url.port ||
      !["api.z.ai", "open.bigmodel.cn", "dev.bigmodel.cn"].includes(url.hostname)) {
    throw new QuotaError("unsupportedEndpoint", "Quota lookup is unavailable for this custom API endpoint.");
  }
  return new URL("/api/monitor/usage/quota/limit", url.origin).href;
}

async function readQuotaJson(response: Response, signal: AbortSignal): Promise<unknown> {
  if (!response.body) return invalidResponse();
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => undefined); };
  signal.addEventListener("abort", cancel, { once: true });
  if (signal.aborted) cancel();
  try {
    const chunks: Uint8Array[] = [];
    let size = 0;
    while (true) {
      const { value, done } = await reader.read();
      signal.throwIfAborted();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_RESPONSE_BYTES) { cancel(); return invalidResponse(); }
      chunks.push(value);
    }
    try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { return invalidResponse(); }
  } finally {
    signal.removeEventListener("abort", cancel);
    reader.releaseLock();
  }
}

/** Account-wide, memory-only cache shared by prompts and extension requests in this process. */
export class QuotaClient {
  private cached?: { key: string; attemptedAt: number; snapshot: QuotaSnapshot };
  private pending?: { key: string; promise: Promise<QuotaSnapshot> };

  constructor(private readonly options: QuotaClientOptions = {}) {}

  async getUsage(): Promise<QuotaSnapshot> {
    const now = this.options.now ?? Date.now;
    const time = now();
    let key: string;
    let url: string;
    let apiKey: string | undefined;
    try {
      url = quotaUrl(this.options.baseUrl?.() ?? process.env["ACP_GLM_BASE_URL"] ?? DEFAULT_BASE_URL);
      apiKey = (this.options.resolveKey ?? resolveApiKey)();
      if (!apiKey) throw new QuotaError("missingCredentials", "Set Z_AI_API_KEY or run glm-acp-agent --setup to check quota.");
      key = `${url}\n${apiKey}`;
    } catch (err) {
      this.cached = undefined;
      return this.failure(err, time);
    }
    // Changing credentials or region invalidates both successes and failures.
    if (this.cached?.key !== key) this.cached = undefined;
    if (this.cached && time - this.cached.attemptedAt < (this.options.cacheMs ?? 60_000)) {
      return structuredClone(this.cached.snapshot);
    }
    if (this.pending?.key === key) return structuredClone(await this.pending.promise);
    const previous = this.cached?.snapshot;
    const promise = this.fetchUsage(url, apiKey, now).catch(err => this.failure(err, now(), previous));
    const pending = { key, promise };
    this.pending = pending;
    try {
      const snapshot = await promise;
      if (this.pending === pending) this.cached = { key, attemptedAt: now(), snapshot };
      return structuredClone(snapshot);
    } finally {
      if (this.pending === pending) this.pending = undefined;
    }
  }

  private async fetchUsage(url: string, apiKey: string, now: () => number): Promise<QuotaSnapshot> {
    const timeout = AbortSignal.timeout(this.options.timeoutMs ?? 10_000);
    const parentSignal = this.options.signal?.();
    const signal = parentSignal ? AbortSignal.any([timeout, parentSignal]) : timeout;
    const response = await (this.options.fetch ?? fetch)(url, {
      // The provider's official usage script sends the key without a Bearer prefix.
      headers: { Authorization: apiKey, Accept: "application/json" },
      redirect: "error",
      signal,
    });
    if (!response.ok) {
      void response.body?.cancel().catch(() => undefined);
      if (response.status === 401 || response.status === 403) {
        throw new QuotaError("authenticationFailed", "Z.AI rejected the quota credentials. Check your Coding Plan API key.");
      }
      throw new QuotaError("requestFailed", `Quota lookup failed (HTTP ${response.status}). Try again later.`);
    }
    const windows = parseQuotaWindows(await readQuotaJson(response, signal));
    return { version: 1, checkedAt: new Date(now()).toISOString(), stale: false, windows };
  }

  private failure(error: unknown, time: number, previous?: QuotaSnapshot): QuotaSnapshot {
    // Never surface fetch errors, provider bodies, or credential-bearing URLs.
    const unavailable = error instanceof QuotaError
      ? { reason: error.reason, message: error.message }
      : { reason: "requestFailed" as const, message: "Quota lookup failed or timed out. Try again later." };
    const retain = previous?.windows.length && unavailable.reason !== "authenticationFailed";
    return {
      version: 1,
      checkedAt: retain ? previous.checkedAt : new Date(time).toISOString(),
      stale: !!retain,
      windows: retain ? previous.windows : [],
      unavailable,
    };
  }
}

export function formatQuotaUsage(snapshot: QuotaSnapshot): string {
  const lines = ["**GLM Coding Plan usage**", ""];
  if (snapshot.unavailable) lines.push(snapshot.unavailable.message, "");
  if (snapshot.stale) lines.push("Showing stale data from the last successful refresh.", "");
  for (const window of snapshot.windows) {
    lines.push(`**${window.label}: ${Number(window.usedPercent.toFixed(2))}% used**`, "");
    if (window.remaining !== undefined) lines.push(`- Remaining: ${window.remaining}${window.unit ? ` ${window.unit}` : ""}`);
    if (window.used !== undefined && window.limit !== undefined) lines.push(`- Used: ${window.used} / ${window.limit}${window.unit ? ` ${window.unit}` : ""}`);
    if (window.resetsAt) lines.push(`- Resets: ${window.resetsAt}`);
    lines.push("");
  }
  lines.push(`${snapshot.windows.length ? "Last updated" : "Checked"}: ${snapshot.checkedAt}`);
  return lines.join("\n");
}
