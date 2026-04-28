import test from "node:test";
import assert from "node:assert/strict";
import { ToolExecutor } from "../tools/executor.js";

function createConnectionStub() {
  const updates: Array<Record<string, unknown>> = [];
  return {
    updates,
    async sessionUpdate(payload: Record<string, unknown>) {
      updates.push(payload);
    },
    async readTextFile() {
      throw new Error("file not found");
    },
    async requestPermission() {
      return { outcome: { outcome: "selected", optionId: "allow" } };
    },
  };
}

test("read_file failures are reported with failed status", async () => {
  const connection = createConnectionStub();
  const executor = new ToolExecutor(connection as never, "session-1");
  const result = await executor.execute("tc1", "read_file", JSON.stringify({ path: "missing.ts" }));

  assert.match(result.content, /Error reading file:/);
  const lastUpdate = connection.updates[connection.updates.length - 1] as {
    update?: { status?: string };
  };
  assert.equal(lastUpdate.update?.status, "failed");
});

test("run_command rejects empty command input", async () => {
  const connection = createConnectionStub();
  const executor = new ToolExecutor(connection as never, "session-2");
  const result = await executor.execute("tc2", "run_command", JSON.stringify({ command: "   " }));

  assert.match(result.content, /non-empty string/);
  const lastUpdate = connection.updates[connection.updates.length - 1] as {
    update?: { status?: string };
  };
  assert.equal(lastUpdate.update?.status, "failed");
});
