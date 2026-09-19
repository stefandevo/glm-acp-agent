import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable, Writable } from "node:stream";
import { ClientSideConnection, ndJsonStream, PROTOCOL_VERSION } from "@agentclientprotocol/sdk";

const AGENT_ENTRY = fileURLToPath(new URL("../index.js", import.meta.url));

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

async function startMockLlm(command: string) {
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      res.writeHead(200, { "content-type": "text/event-stream" });
      const chunk = {
        id: "mock",
        object: "chat.completion.chunk",
        choices: [{
          delta: {
            tool_calls: [{
              index: 0,
              id: "tool-1",
              function: { name: "run_command", arguments: JSON.stringify({ command }) },
            }],
          },
          finish_reason: "tool_calls",
        }],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      res.end("data: [DONE]\n\n");
    });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  return { server, baseUrl: `http://127.0.0.1:${address.port}/v1` };
}

async function runCliUntilCommand(command: string, cwd: string): Promise<{ child: ChildProcess; started: Promise<void>; close: () => void; closeServer: () => void }> {
  const { server, baseUrl } = await startMockLlm(command);
  const child = spawn(process.execPath, [AGENT_ENTRY], {
    cwd,
    stdio: ["pipe", "pipe", "pipe"],
    env: {
      ...process.env,
      Z_AI_API_KEY: "cli-shutdown-test-key",
      ACP_GLM_BASE_URL: baseUrl,
      ACP_GLM_SESSION_DIR: join(cwd, "sessions"),
    },
  });
  let resolveStarted!: () => void;
  const started = new Promise<void>((resolve) => { resolveStarted = resolve; });
  const client = {
    async sessionUpdate({ update }: { update: { sessionUpdate?: string; status?: string } }) {
      if (update.sessionUpdate === "tool_call_update" && update.status === "in_progress") resolveStarted();
    },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
  const stream = ndJsonStream(
    Writable.toWeb(child.stdin!) as WritableStream<Uint8Array>,
    Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>
  );
  const connection = new ClientSideConnection(() => client, stream);
  void (async () => {
    await connection.initialize({ protocolVersion: PROTOCOL_VERSION, clientCapabilities: {} });
    const session = await connection.newSession({ cwd, mcpServers: [] });
    await connection.setSessionMode({ sessionId: session.sessionId, modeId: "bypass_permissions" });
    await connection.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "run" }] });
  })().catch(() => undefined);
  return {
    child,
    started,
    close: () => {
      child.stdin?.end();
      server.close();
    },
    closeServer: () => server.close(),
  };
}

async function waitForExit(child: ChildProcess): Promise<{ code: number | null; signal: NodeJS.Signals | null }> {
  const [code, signal] = await once(child, "exit") as [number | null, NodeJS.Signals | null];
  return { code, signal };
}

test("CLI stdin close waits for SIGTERM-resistant command cleanup", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-shutdown-"));
  const ready = join(cwd, "ready");
  const marker = join(cwd, "marker");
  const code = [
    'const fs=require("node:fs");',
    'process.on("SIGTERM",()=>{});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(()=>fs.writeFileSync(${JSON.stringify(marker)}, "survived"),700);`,
  ].join("");
  const { child, started, close } = await runCliUntilCommand(`${shellQuote(process.execPath)} -e ${shellQuote(code)}`, cwd);
  try {
    await started;
    const deadline = Date.now() + 2_000;
    while (!existsSync(ready)) {
      assert.ok(Date.now() < deadline, "command child did not reach ready state");
      await wait(10);
    }
    close();
    const [exitCode] = await once(child, "exit");
    assert.equal(exitCode, 0);
    await wait(900);
    assert.equal(existsSync(marker), false);
  } finally {
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("CLI SIGTERM closes its stdin transport after cleanup even when the client keeps stdin open", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-sigterm-"));
  const ready = join(cwd, "ready");
  const marker = join(cwd, "marker");
  const code = [
    'const fs=require("node:fs");',
    'process.on("SIGTERM",()=>{});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(()=>fs.writeFileSync(${JSON.stringify(marker)}, "survived"),700);`,
  ].join("");
  const { child, started, closeServer } = await runCliUntilCommand(`${shellQuote(process.execPath)} -e ${shellQuote(code)}`, cwd);
  try {
    await started;
    while (!existsSync(ready)) await wait(10);
    child.kill("SIGTERM");
    const [exitCode] = await once(child, "exit");
    assert.equal(exitCode, 143);
    await wait(900);
    assert.equal(existsSync(marker), false);
  } finally {
    closeServer();
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("CLI keeps cleanup alive across repeated SIGTERM signals", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-repeat-sigterm-"));
  const ready = join(cwd, "ready");
  const marker = join(cwd, "marker");
  const code = [
    'const fs=require("node:fs");',
    'process.on("SIGTERM",()=>{});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(()=>fs.writeFileSync(${JSON.stringify(marker)}, "survived"),700);`,
  ].join("");
  const { child, started, closeServer } = await runCliUntilCommand(`${shellQuote(process.execPath)} -e ${shellQuote(code)}`, cwd);
  try {
    await started;
    while (!existsSync(ready)) await wait(10);
    child.kill("SIGTERM");
    // Let the first handler start shutdown before delivering the second signal.
    await wait(50);
    child.kill("SIGTERM");
    const exited = await waitForExit(child);
    assert.equal(exited.code, 143);
    assert.equal(exited.signal, null);
    await wait(900);
    assert.equal(existsSync(marker), false);
  } finally {
    closeServer();
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("CLI SIGTERM upgrades a disconnect shutdown exit code", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-disconnect-sigterm-"));
  const ready = join(cwd, "ready");
  const marker = join(cwd, "marker");
  const code = [
    'const fs=require("node:fs");',
    'process.on("SIGTERM",()=>{});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(()=>fs.writeFileSync(${JSON.stringify(marker)}, "survived"),700);`,
  ].join("");
  const { child, started, close, closeServer } = await runCliUntilCommand(`${shellQuote(process.execPath)} -e ${shellQuote(code)}`, cwd);
  try {
    await started;
    while (!existsSync(ready)) await wait(10);
    close();
    child.kill("SIGTERM");
    const exited = await waitForExit(child);
    assert.equal(exited.code, 143);
    assert.equal(exited.signal, null);
    await wait(900);
    assert.equal(existsSync(marker), false);
  } finally {
    closeServer();
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("CLI SIGINT upgrades a disconnect shutdown exit code", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-disconnect-sigint-"));
  const ready = join(cwd, "ready");
  const marker = join(cwd, "marker");
  const code = [
    'const fs=require("node:fs");',
    'process.on("SIGTERM",()=>{});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(()=>fs.writeFileSync(${JSON.stringify(marker)}, "survived"),700);`,
  ].join("");
  const { child, started, close, closeServer } = await runCliUntilCommand(`${shellQuote(process.execPath)} -e ${shellQuote(code)}`, cwd);
  try {
    await started;
    while (!existsSync(ready)) await wait(10);
    close();
    child.kill("SIGINT");
    const exited = await waitForExit(child);
    assert.equal(exited.code, 130);
    assert.equal(exited.signal, null);
    await wait(900);
    assert.equal(existsSync(marker), false);
  } finally {
    closeServer();
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});

test("CLI shutdown preserves intentionally backgrounded commands after normal shell exit", { skip: process.platform === "win32", timeout: 10_000 }, async () => {
  const cwd = mkdtempSync(join(tmpdir(), "glm-cli-background-"));
  const marker = join(cwd, "marker");
  const code = `setTimeout(()=>require("node:fs").writeFileSync(${JSON.stringify(marker)}, "done"), 300)`;
  const command = `${shellQuote(process.execPath)} -e ${shellQuote(code)} >/dev/null 2>&1 & echo started`;
  const { child, started, close } = await runCliUntilCommand(command, cwd);
  try {
    await started;
    await wait(100);
    close();
    const [exitCode] = await once(child, "exit");
    assert.equal(exitCode, 0);
    const deadline = Date.now() + 1_500;
    while (!existsSync(marker)) {
      assert.ok(Date.now() < deadline, "background command was terminated during shutdown");
      await wait(10);
    }
  } finally {
    if (child.exitCode === null) child.kill("SIGKILL");
    rmSync(cwd, { recursive: true, force: true });
  }
});
