import test from "node:test";
import assert from "node:assert/strict";
import { readResourceLimits } from "../tools/resource-limits.js";

test("resource limits use safe defaults and reject invalid overrides", () => {
  const warnings: string[] = [];
  const limits = readResourceLimits({
    ACP_GLM_TOOL_RESULT_LIMIT_BYTES: "127",
    ACP_GLM_READ_FILE_LIMIT_BYTES: "bad",
    ACP_GLM_LIST_FILES_MAX_ENTRIES: "2001",
  }, message => warnings.push(message));

  assert.equal(limits.toolResultBytes, 262_144);
  assert.equal(limits.fileReadBytes, 8 * 1024 * 1024);
  assert.equal(limits.listEntries, 2000);
  assert.equal(warnings.length, 3);
});
