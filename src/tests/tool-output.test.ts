import test from "node:test";
import assert from "node:assert/strict";
import { boundToolResult, takeUtf8Suffix } from "../tools/tool-output.js";

test("tool output fits an inclusive UTF-8 budget without splitting characters", () => {
  const result = boundToolResult("🙂".repeat(100_000), 262_144);
  assert.ok(Buffer.byteLength(result, "utf8") <= 262_144);
  assert.match(result, /bytes omitted/);
  assert.ok(!result.includes("\uFFFD"));
});

test("tool output returns the original string when it fits", () => {
  assert.equal(boundToolResult("hé", 128), "hé");
});

test("tool output keeps complete characters at both retained boundaries", () => {
  const result = boundToolResult("α".repeat(1_000) + "🙂".repeat(1_000), 128);
  assert.ok(!result.includes("\uFFFD"));
  assert.equal(Buffer.byteLength(result, "utf8") <= 128, true);
});

test("suffix helper keeps a trailing supplementary character intact", () => {
  assert.equal(takeUtf8Suffix("x🙂", 4), "🙂");
});

test("suffix helper scans a large single line without materializing all code points", () => {
  const text = `${"x".repeat(1_000_000)}🙂tail`;
  const originalFrom = Object.getOwnPropertyDescriptor(Array, "from");
  assert.ok(originalFrom);
  Object.defineProperty(Array, "from", {
    ...originalFrom,
    value: (...args: Parameters<typeof Array.from>) => {
      if (args[0] === text) throw new Error("whole-input code-point materialization is forbidden");
      return Reflect.apply(originalFrom.value as typeof Array.from, Array, args);
    },
  });
  try {
    assert.equal(takeUtf8Suffix(text, 8), "🙂tail");
  } finally {
    Object.defineProperty(Array, "from", originalFrom);
  }
});
