import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readLocalTextPage } from "../tools/file-reader.js";

test("local reader caps a 600 KB single line without treating it as a next page", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-file-reader-"));
  const path = join(dir, "large.txt");
  writeFileSync(path, "x".repeat(600 * 1024));
  try {
    const page = await readLocalTextPage(path, 1, 20, 1024);
    assert.equal(page.truncated, true);
    assert.equal(page.incompleteLine, 1);
    assert.equal(page.nextLine, undefined);
    assert.ok(Buffer.byteLength(page.text, "utf8") <= 1024);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("local reader keeps ordinary pagination and knows EOF", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-file-reader-"));
  const path = join(dir, "lines.txt");
  writeFileSync(path, "one\ntwo\nthree\nfour\n", "utf8");
  try {
    const page = await readLocalTextPage(path, 2, 2, 1024);
    assert.deepEqual(page, { text: "two\nthree", firstLine: 2, lastCompleteLine: 3, totalLines: 4, nextLine: 4, truncated: false });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("local reader does not emit a replacement character when the scan stops in UTF-8", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-file-reader-"));
  const path = join(dir, "utf8.txt");
  writeFileSync(path, "🙂".repeat(100), "utf8");
  try {
    const page = await readLocalTextPage(path, 1, 1, 129);
    assert.ok(!page.text.includes("\uFFFD"));
    assert.equal(page.incompleteLine, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("local reader discloses a scan ending after a complete line without a dead-end offset", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-file-reader-")); const path = join(dir, "cap.txt");
  writeFileSync(path, "a\nb\nc\n");
  try {
    const page = await readLocalTextPage(path, 1, 20, 4);
    assert.equal(page.truncated, true);
    // The scan restarts from byte zero, so no offset beyond this prefix can
    // ever be served; advertising one would dead-end the next request.
    assert.equal(page.nextLine, undefined);
    const followUp = await readLocalTextPage(path, page.lastCompleteLine + 1, 20, 4);
    assert.equal(followUp.text, "", "following a budget-bound boundary must not re-serve old lines");
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("local reader reports only observed complete lines beyond a truncated scan", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-file-reader-"));
  const path = join(dir, "cap.txt");
  writeFileSync(path, "a\nb\nc\n");
  try {
    const page = await readLocalTextPage(path, 100, 20, 4);
    assert.deepEqual(page, {
      text: "",
      firstLine: 100,
      lastCompleteLine: 2,
      truncated: true,
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
