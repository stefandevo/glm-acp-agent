import test, { mock } from "node:test";
import assert from "node:assert/strict";
import childProcess from "node:child_process";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { rm } from "node:fs/promises";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ToolExecutor } from "../tools/executor.js";
import {
  DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
  DEFAULT_COMMAND_TIMEOUT_MS,
  readCommandLimits,
} from "../tools/command-limits.js";

function createConnectionStub() {
  const updates: Array<Record<string, unknown>> = [];
  return {
    updates,
    async sessionUpdate(payload: Record<string, unknown>) {
      updates.push(payload);
    },
  };
}

function withEnv<T>(values: Record<string, string | undefined>, fn: () => Promise<T>): Promise<T> {
  const old = new Map<string, string | undefined>();
  for (const [key, value] of Object.entries(values)) {
    old.set(key, process.env[key]);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  return fn().finally(() => {
    for (const [key, value] of old) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });
}

function commandExecutor(
  connection: ReturnType<typeof createConnectionStub>,
  cwd = process.cwd(),
  signal?: AbortSignal
) {
  return new ToolExecutor(
    connection as never,
    "s1",
    { fs: {} },
    signal,
    null,
    null,
    cwd,
    () => "bypass_permissions"
  );
}

function lastUpdate(connection: ReturnType<typeof createConnectionStub>) {
  return connection.updates.at(-1)?.update as {
    status?: string;
    rawOutput?: Record<string, unknown>;
  };
}

async function waitForFile(path: string, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(path)) return true;
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  return existsSync(path);
}

async function settlesWithinRealTime(promise: Promise<unknown>, timeoutMs: number): Promise<boolean> {
  let settled = false;
  void promise.then(
    () => { settled = true; },
    () => { settled = true; },
  );
  const deadline = Date.now() + timeoutMs;
  while (!settled && Date.now() < deadline) {
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  return settled;
}

test("command limits use the documented defaults when env is absent", () => {
  const warnings: string[] = [];
  const limits = readCommandLimits({}, (message) => warnings.push(message));
  assert.deepEqual(limits, {
    timeoutMs: DEFAULT_COMMAND_TIMEOUT_MS,
    outputLimitBytes: DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
  });
  assert.deepEqual(warnings, []);
});

test("command limits accept positive integer env overrides", () => {
  const limits = readCommandLimits(
    {
      ACP_GLM_COMMAND_TIMEOUT_MS: "2500",
      ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "123",
    },
    () => undefined
  );
  assert.deepEqual(limits, { timeoutMs: 2500, outputLimitBytes: 123 });
});

test("invalid and unsafe command limits fall back with warnings", () => {
  const warnings: string[] = [];
  const limits = readCommandLimits(
    {
      ACP_GLM_COMMAND_TIMEOUT_MS: "2147483648",
      ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "9007199254740992",
    },
    (message) => warnings.push(message)
  );
  assert.deepEqual(limits, {
    timeoutMs: DEFAULT_COMMAND_TIMEOUT_MS,
    outputLimitBytes: DEFAULT_COMMAND_OUTPUT_LIMIT_BYTES,
  });
  assert.equal(warnings.length, 2);
  assert.match(warnings[0] ?? "", /ACP_GLM_COMMAND_TIMEOUT_MS/);
  assert.match(warnings[1] ?? "", /ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES/);
});

test("finite command output is capped across stdout and stderr with a truncation notice", async () => {
  const connection = createConnectionStub();
  const cwd = mkdtempSync(join(tmpdir(), "glm-command-limits-output-"));
  try {
    const result = await withEnv(
      {
        // This test exercises output accounting, not startup latency. Windows
        // CI can take more than a second to start Git Bash under full-suite load.
        ACP_GLM_COMMAND_TIMEOUT_MS: "10000",
        ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "10",
      },
      () =>
        commandExecutor(connection, cwd).execute(
          "tc1",
          "run_command",
          JSON.stringify({ command: "printf 12345678; printf abcdefgh >&2" })
        )
    );
    assert.match(result.content, /Output truncated: command output exceeded 10 bytes/);
    const update = lastUpdate(connection);
    assert.equal(update.status, "completed");
    const raw = update.rawOutput ?? {};
    const capturedBytes =
      Buffer.byteLength(String(raw.stdout ?? ""), "utf8") +
      Buffer.byteLength(String(raw.stderr ?? ""), "utf8");
    assert.ok(capturedBytes <= 10, `captured ${capturedBytes} bytes`);
    assert.equal(raw.outputTruncated, true);
  } finally {
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

test("a byte cap never exposes a partial UTF-8 code point", async () => {
  const connection = createConnectionStub();
  const cwd = mkdtempSync(join(tmpdir(), "glm-command-limits-utf8-"));
  try {
    await withEnv(
      {
        ACP_GLM_COMMAND_TIMEOUT_MS: "10000",
        ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "1",
      },
      () =>
        commandExecutor(connection, cwd).execute(
          "tc1",
          "run_command",
          JSON.stringify({ command: "printf '\\303\\251'" })
        )
    );
    const raw = lastUpdate(connection).rawOutput ?? {};
    assert.ok(Buffer.byteLength(String(raw.stdout ?? ""), "utf8") <= 1);
    assert.equal(raw.stdout, "");
  } finally {
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

test("a command timeout terminates the process tree and marks the tool failed", async () => {
  const connection = createConnectionStub();
  const abortController = new AbortController();
  const cwd = mkdtempSync(join(tmpdir(), "glm-command-limits-timeout-"));
  const ready = join(cwd, "ready");
  const fire = join(cwd, "fire");
  const marker = join(cwd, "late");
  writeFileSync(
    join(cwd, "timeout-fixture.cjs"),
      'const fs = require("node:fs");\n' +
      'fs.writeFileSync("ready", "ready");\n' +
      'setInterval(() => process.stdout.write("x"), 1);\n' +
      'setInterval(() => {\n' +
      '  if (fs.existsSync("fire")) fs.writeFileSync("late", "late");\n' +
      '}, 1);\n',
    "utf8"
  );
  mock.timers.enable({ apis: ["setTimeout"] });
  let pendingCleanup: Promise<unknown> | null = null;
  try {
    const pending = withEnv(
      {
        ACP_GLM_COMMAND_TIMEOUT_MS: "1000",
        ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "1024",
      },
      () =>
        commandExecutor(connection, cwd, abortController.signal).execute(
          "tc1",
          "run_command",
          JSON.stringify({
            command: "node timeout-fixture.cjs",
          })
        )
    );
    pendingCleanup = pending;
    assert.equal(await waitForFile(ready, 5_000), true);
    mock.timers.tick(1_000);
    const result = await pending;
    assert.match(result.content, /timed out after 1000 ms/i);
    assert.equal(lastUpdate(connection).status, "failed");
    writeFileSync(fire, "fire");
    assert.equal(await waitForFile(marker, 800), false);
  } finally {
    abortController.abort();
    mock.timers.tick(250);
    mock.timers.reset();
    await pendingCleanup?.catch(() => undefined);
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

test("a normal background shell exit survives a longer deadline", async () => {
  const connection = createConnectionStub();
  const cwd = mkdtempSync(join(tmpdir(), "glm-command-limits-background-deadline-"));
  const marker = join(cwd, "background-finished");
  writeFileSync(
    join(cwd, "background-fixture.cjs"),
    'setTimeout(() => require("node:fs").writeFileSync("background-finished", "done"), 1100);\n',
    "utf8"
  );
  try {
    const result = await withEnv(
      {
        ACP_GLM_COMMAND_TIMEOUT_MS: "10000",
        ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "1024",
      },
      () =>
        commandExecutor(connection, cwd).execute(
          "tc1",
          "run_command",
          JSON.stringify({
            command: "node background-fixture.cjs & echo started",
          })
        )
    );
    assert.match(result.content, /Exit code: 0/);
    assert.equal(await waitForFile(marker, 2_500), true);
    assert.equal(lastUpdate(connection).status, "completed");
  } finally {
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

test("shell-exit cleanup prevents a deadline from killing inherited pipes", async () => {
  const connection = createConnectionStub();
  const abortController = new AbortController();
  const cwd = mkdtempSync(join(tmpdir(), "glm-command-limits-background-race-"));
  const ready = join(cwd, "ready");
  const release = join(cwd, "release");
  const marker = join(cwd, "background-finished");
  writeFileSync(
    join(cwd, "foreground-fixture.cjs"),
    'const fs = require("node:fs");\n' +
      'fs.writeFileSync("ready", "ready");\n' +
      'const wait = setInterval(() => {\n' +
      '  if (fs.existsSync("release")) { clearInterval(wait); process.exit(0); }\n' +
      '}, 1);\n',
    "utf8"
  );
  writeFileSync(
    join(cwd, "background-race-fixture.cjs"),
    'setTimeout(() => require("node:fs").writeFileSync("background-finished", "done"), 1500);\n',
    "utf8"
  );
  mock.timers.enable({ apis: ["setTimeout"] });
  let shellExitResolve!: () => void;
  const shellExitEvent = new Promise<void>((resolve) => {
    shellExitResolve = resolve;
  });
  const originalSpawn = childProcess.spawn;
  const spawnMock = mock.method(
    childProcess,
    "spawn",
    ((...args: Parameters<typeof originalSpawn>) => {
      const child = originalSpawn(...args);
      if (args[0] === "sh") child.once("exit", shellExitResolve);
      return child;
    }) as typeof childProcess.spawn
  );
  syncBuiltinESMExports();
  let pendingCleanup: Promise<unknown> | null = null;
  try {
    const pending = withEnv(
      {
        ACP_GLM_COMMAND_TIMEOUT_MS: "1000",
        ACP_GLM_COMMAND_OUTPUT_LIMIT_BYTES: "1024",
      },
      () =>
        commandExecutor(connection, cwd, abortController.signal).execute(
          "tc1",
          "run_command",
          JSON.stringify({
            command: "node foreground-fixture.cjs; node background-race-fixture.cjs & echo started",
          })
        )
    );
    pendingCleanup = pending;

    assert.equal(await waitForFile(ready, 2_000), true);
    mock.timers.tick(950);
    writeFileSync(release, "release");
    // Wait for the actual ChildProcess exit event. Its continuation runs after
    // the executor's exit handler has cleared the command deadline.
    assert.equal(
      await settlesWithinRealTime(shellExitEvent, 5_000),
      true,
      "shell did not exit after the foreground fixture was released",
    );
    mock.timers.tick(100);
    const result = await pending;
    assert.match(result.content, /Exit code: 0/);
    assert.doesNotMatch(result.content, /timed out/i);
    assert.equal(await waitForFile(marker, 2_500), true);
    assert.equal(lastUpdate(connection).status, "completed");
  } finally {
    abortController.abort();
    mock.timers.tick(250);
    spawnMock.mock.restore();
    syncBuiltinESMExports();
    mock.timers.reset();
    await pendingCleanup?.catch(() => undefined);
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
});

test("a pre-aborted command is rejected without spawning work", async () => {
  const connection = createConnectionStub();
  const controller = new AbortController();
  controller.abort();
  const result = await commandExecutor(connection, process.cwd(), controller.signal).execute(
    "tc1",
    "run_command",
    JSON.stringify({ command: "printf should-not-run" })
  );
  assert.match(result.content, /cancelled by (turn|user)|aborted/i);
  assert.equal(lastUpdate(connection).status, "failed");
});
