import test from "node:test";
import assert from "node:assert/strict";
import { parseMaxTurnsFlag } from "../cli-args.js";

/** Run the parser with stderr capture; returns [result, warnings]. */
function parse(argv: string[]): [number | undefined, string[]] {
  const warnings: string[] = [];
  const original = process.stderr.write;
  process.stderr.write = (chunk: unknown) => {
    warnings.push(String(chunk));
    return true;
  };
  try {
    return [parseMaxTurnsFlag(argv), warnings];
  } finally {
    process.stderr.write = original;
  }
}

test("--max-turns <n> space form parses a valid integer", () => {
  const [result, warnings] = parse(["--max-turns", "150"]);
  assert.equal(result, 150);
  assert.equal(warnings.length, 0);
});

test("--max-turns=<n> equals form parses a valid integer", () => {
  const [result, warnings] = parse(["--max-turns=7"]);
  assert.equal(result, 7);
  assert.equal(warnings.length, 0);
});

test("--max-turns floors a valid fraction to a whole turn count", () => {
  const [result, warnings] = parse(["--max-turns", "2.9"]);
  assert.equal(result, 2);
  assert.equal(warnings.length, 0);
});

test("absent flag returns undefined so the env var can apply", () => {
  const [result, warnings] = parse(["--setup"]);
  assert.equal(result, undefined);
  assert.equal(warnings.length, 0);
});

test("invalid value warns and selects the default, not undefined", () => {
  for (const bad of ["nope", "-3", "0", "0.5"]) {
    const [result, warnings] = parse(["--max-turns", bad]);
    assert.equal(result, 20, `value ${bad}`);
    assert.match(warnings.join(""), /ignoring invalid --max-turns/);
  }
});

test("invalid value in equals form warns and selects the default", () => {
  const [result, warnings] = parse(["--max-turns=banana"]);
  assert.equal(result, 20);
  assert.match(warnings.join(""), /ignoring invalid --max-turns/);
});

test("valueless flag warns and selects the default", () => {
  const [result, warnings] = parse(["--setup", "--max-turns"]);
  assert.equal(result, 20);
  assert.match(warnings.join(""), /--max-turns requires a value/);
});
