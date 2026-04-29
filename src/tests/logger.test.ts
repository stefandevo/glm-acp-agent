import test from "node:test";
import assert from "node:assert/strict";
import { maskSecret } from "../llm/logger.js";

test("maskSecret masks all but last 4 chars", () => {
  assert.equal(maskSecret("abcdefgh"), "****efgh");
});

test("maskSecret returns **** for short strings", () => {
  assert.equal(maskSecret("ab"), "****");
  assert.equal(maskSecret("abcd"), "****");
});

test("maskSecret handles exactly 5 chars", () => {
  assert.equal(maskSecret("abcde"), "****bcde");
});
