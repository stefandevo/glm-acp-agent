import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir as osTmpdir } from "node:os";
import { join as pathJoin } from "node:path";
import { fileURLToPath } from "node:url";
import { syncManifestContent } from "../registry/sync-manifest.js";

const PACKAGE = { name: "glm-acp-agent", version: "1.4.1" };

/** Mirrors registry/glm-acp-agent/agent.json, including its hand formatting. */
const MANIFEST = `{
  "id": "glm-acp-agent",
  "name": "GLM Agent",
  "version": "1.3.0",
  "description": "ACP agent powered by Zhipu AI's GLM Coding Plan models.",
  "repository": "https://github.com/stefandevo/glm-acp-agent",
  "authors": ["Stefan de Vogelaere"],
  "license": "Apache-2.0",
  "icon": "icon.svg",
  "distribution": {
    "npx": {
      "package": "glm-acp-agent@1.3.0"
    }
  }
}
`;

// ---------------------------------------------------------------------------
// syncManifestContent
// ---------------------------------------------------------------------------

test("syncManifestContent: rewrites the top-level version field", () => {
  const result = syncManifestContent(MANIFEST, PACKAGE);
  assert.match(result, /"version": "1\.4\.1"/);
});

test("syncManifestContent: rewrites the npx distribution pin", () => {
  const result = syncManifestContent(MANIFEST, PACKAGE);
  assert.match(result, /"package": "glm-acp-agent@1\.4\.1"/);
});

test("syncManifestContent: preserves unrelated formatting (inline arrays, field order)", () => {
  const result = syncManifestContent(MANIFEST, PACKAGE);
  // The surgical rewrite must not reformat the whole file — only the two
  // version occurrences may change.
  assert.equal(
    result,
    MANIFEST.replaceAll("1.3.0", "1.4.1"),
    "everything except the two version pins should be byte-identical"
  );
});

test("syncManifestContent: is idempotent when already in sync", () => {
  const synced = syncManifestContent(MANIFEST, PACKAGE);
  assert.equal(syncManifestContent(synced, PACKAGE), synced);
});

test("syncManifestContent: throws on invalid JSON", () => {
  assert.throws(
    () => syncManifestContent("{ not json", PACKAGE),
    /agent\.json[\s\S]*parse/u
  );
});

test("syncManifestContent: throws when the version field is missing", () => {
  const noVersion = MANIFEST.replace(/^ {2}"version": ".*",\n/m, "");
  assert.throws(
    () => syncManifestContent(noVersion, PACKAGE),
    /"version"[\s\S]*not found/u
  );
});

test("syncManifestContent: throws when the npx package pin is missing", () => {
  const noPin = MANIFEST.replace(
    /^(\s*)"package": ".*"\n/m,
    '$1"package_placeholder": true\n'
  );
  assert.throws(
    () => syncManifestContent(noPin, PACKAGE),
    /distribution\.npx\.package[\s\S]*not found/u
  );
});

// ---------------------------------------------------------------------------
// CLI (built script, run against a fixture package in a temp git repo)
// ---------------------------------------------------------------------------

const SCRIPT_PATH = fileURLToPath(
  new URL("../registry/sync-manifest.js", import.meta.url)
);

interface Fixture {
  dir: string;
  manifestPath: string;
}

/** A temp directory shaped like the repo: package.json + registry manifest, git-initialised. */
function makeFixture(manifestContent: string): Fixture {
  const dir = mkdtempSync(pathJoin(osTmpdir(), "glm-acp-sync-test-"));
  writeFileSync(
    pathJoin(dir, "package.json"),
    `${JSON.stringify({ name: PACKAGE.name, version: PACKAGE.version }, null, 2)}\n`
  );
  const manifestPath = pathJoin(dir, "registry", "glm-acp-agent", "agent.json");
  mkdirSync(pathJoin(manifestPath, ".."), { recursive: true });
  writeFileSync(manifestPath, manifestContent);
  const git = (args: string[]) =>
    spawnSync("git", args, { cwd: dir, encoding: "utf8" });
  git(["init", "--quiet"]);
  git(["add", "package.json", "registry"]);
  git([
    "-c",
    "user.name=test",
    "-c",
    "user.email=test@example.com",
    "-c",
    "commit.gpgsign=false",
    "commit",
    "--quiet",
    "-m",
    "fixture",
  ]);
  return { dir, manifestPath };
}

function runScript(fixture: Fixture, args: string[]) {
  return spawnSync(process.execPath, [SCRIPT_PATH, ...args], {
    cwd: fixture.dir,
    encoding: "utf8",
  });
}

test("CLI: default mode rewrites the manifest and stages it for the release commit", () => {
  const fixture = makeFixture(MANIFEST);
  try {
    const run = runScript(fixture, []);
    assert.equal(run.status, 0, `stderr: ${run.stderr}`);

    const onDisk = readFileSync(fixture.manifestPath, "utf8");
    assert.match(onDisk, /"version": "1\.4\.1"/);
    assert.match(onDisk, /"package": "glm-acp-agent@1\.4\.1"/);

    // The `version` lifecycle hook relies on the rewrite landing in the git
    // index so `npm version` folds it into the release commit.
    const status = spawnSync("git", ["status", "--porcelain"], {
      cwd: fixture.dir,
      encoding: "utf8",
    }).stdout.trim();
    assert.equal(status, "M  registry/glm-acp-agent/agent.json");
  } finally {
    rmSync(fixture.dir, { recursive: true, force: true });
  }
});

test("CLI: --check passes (exit 0) and leaves the file untouched when in sync", () => {
  const inSync = syncManifestContent(MANIFEST, PACKAGE);
  const fixture = makeFixture(inSync);
  try {
    const run = runScript(fixture, ["--check"]);
    assert.equal(run.status, 0, `stderr: ${run.stderr}`);
    assert.equal(readFileSync(fixture.manifestPath, "utf8"), inSync);
  } finally {
    rmSync(fixture.dir, { recursive: true, force: true });
  }
});

test("CLI: --check fails (exit 1) with a diagnostic when the manifest drifted", () => {
  const fixture = makeFixture(MANIFEST);
  try {
    const run = runScript(fixture, ["--check"]);
    assert.notEqual(run.status, 0);
    assert.match(run.stderr, /1\.3\.0[\s\S]*1\.4\.1/);
    // Check mode must never rewrite the file.
    assert.equal(readFileSync(fixture.manifestPath, "utf8"), MANIFEST);
  } finally {
    rmSync(fixture.dir, { recursive: true, force: true });
  }
});
