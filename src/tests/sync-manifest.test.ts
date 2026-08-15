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

test("syncManifestContent: preserves field order and unrelated values, normalizing formatting", () => {
  const result = syncManifestContent(MANIFEST, PACKAGE);
  const parsed = JSON.parse(result) as Record<string, unknown>;
  // JSON.stringify keeps parse insertion order, so after the one-time
  // normalization the field layout is stable and diffs stay reviewable.
  assert.deepEqual(Object.keys(parsed), [
    "id",
    "name",
    "version",
    "description",
    "repository",
    "authors",
    "license",
    "icon",
    "distribution",
  ]);
  assert.deepEqual(parsed.authors, ["Stefan de Vogelaere"]);
  assert.equal(parsed.description, "ACP agent powered by Zhipu AI's GLM Coding Plan models.");
  assert.equal(result, `${JSON.stringify(parsed, null, 2)}\n`);
});

test("syncManifestContent: leaves a nested version field alone even when the top-level pin is already current", () => {
  // Regression (PR #83 review): a line-based rewrite hit the FIRST "version"
  // line, so with the top-level field already current it silently rewrote
  // unrelated nested metadata while validation still passed.
  const nested = MANIFEST.replace(
    /^ {2}"id": "glm-acp-agent",\n/m,
    `  "metadata": { "version": "9.9.9" },\n  "id": "glm-acp-agent",\n`
  ).replace(/^ {2}"version": "1\.3\.0",\n/m, `  "version": "1.4.1",\n`);
  const parsed = JSON.parse(syncManifestContent(nested, PACKAGE)) as {
    version: string;
    metadata: { version: string };
    distribution: { npx: { package: string } };
  };
  assert.equal(parsed.metadata.version, "9.9.9");
  assert.equal(parsed.version, "1.4.1");
  assert.equal(parsed.distribution.npx.package, "glm-acp-agent@1.4.1");
});

test("syncManifestContent: updates the npx pin without touching a sibling distribution's pin", () => {
  // Regression (PR #83 review): a line-based rewrite hit the FIRST "package"
  // line, so a sibling distribution carrying the same pin made the sync throw
  // on an otherwise valid manifest and blocked CI / `npm version`.
  const sibling = MANIFEST.replace(
    /^ {4}"npx": \{\n/m,
    `    "pip": {\n      "package": "glm-acp-agent@1.3.0"\n    },\n    "npx": {\n`
  );
  const parsed = JSON.parse(syncManifestContent(sibling, PACKAGE)) as {
    distribution: { pip: { package: string }; npx: { package: string } };
  };
  assert.equal(parsed.distribution.npx.package, "glm-acp-agent@1.4.1");
  // Only the npx distribution is managed by this script.
  assert.equal(parsed.distribution.pip.package, "glm-acp-agent@1.3.0");
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
