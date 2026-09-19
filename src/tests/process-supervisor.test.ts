import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ProcessSupervisor } from "../tools/process-supervisor.js";

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(condition: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!condition()) {
    assert.ok(Date.now() < deadline, "timed out waiting for child process state");
    await wait(10);
  }
}

test("ProcessSupervisor terminates a SIGTERM-resistant owned process before shutdown completes", { skip: process.platform === "win32", timeout: 5_000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-process-supervisor-"));
  const ready = join(dir, "ready");
  const marker = join(dir, "marker");
  const script = [
    'const fs = require("node:fs");',
    'process.on("SIGTERM", () => {});',
    `fs.writeFileSync(${JSON.stringify(ready)}, "ready");`,
    `setTimeout(() => fs.writeFileSync(${JSON.stringify(marker)}, "survived"), 700);`,
  ].join("");
  const child = spawn(process.execPath, ["-e", script], { detached: true, stdio: "ignore" });
  const supervisor = new ProcessSupervisor();
  supervisor.register(child);

  try {
    await waitFor(() => existsSync(ready), 1_000);
    await supervisor.terminateAll();
    assert.equal(supervisor.hasActiveProcesses(), false);
    await wait(800);
    assert.equal(existsSync(marker), false);
  } finally {
    if (child.exitCode === null) {
      try { process.kill(-child.pid!, "SIGKILL"); } catch { /* already gone */ }
    }
    rmSync(dir, { recursive: true, force: true });
  }
});

test("ProcessSupervisor releases background work after a normal shell exit", { skip: process.platform === "win32", timeout: 5_000 }, async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-process-supervisor-background-"));
  const marker = join(dir, "marker");
  const child = spawn("sh", ["-c", `${JSON.stringify(process.execPath)} -e ${JSON.stringify(`setTimeout(() => require("node:fs").writeFileSync(${JSON.stringify(marker)}, "done"), 250)`)} >/dev/null 2>&1 &`], {
    detached: true,
    stdio: "ignore",
  });
  const supervisor = new ProcessSupervisor();
  const managed = supervisor.register(child);

  try {
    await new Promise<void>((resolve) => child.once("exit", () => resolve()));
    managed.releaseAfterNormalExit();
    await supervisor.terminateAll();
    await waitFor(() => existsSync(marker), 1_000);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
