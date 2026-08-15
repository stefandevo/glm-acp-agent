import { spawnSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const MANIFEST_PATH = "registry/glm-acp-agent/agent.json";

/** The npm identity the registry manifest must mirror. */
export interface PackageIdentity {
  name: string;
  version: string;
}

interface AgentManifest {
  version?: unknown;
  distribution?: { npx?: { package?: unknown } };
}

const escapeRegExp = (value: string) =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const VERSION_FIELD = /^(\s*"version"\s*:\s*")[^"]*(")/;

/**
 * Rewrite an agent.json manifest so both version pins match the package.
 *
 * The edit is surgical — only the `"version"` field and the
 * `distribution.npx.package` pin change, so hand formatting (inline arrays,
 * field order) survives and release diffs stay reviewable. Anything
 * unexpected (unparseable JSON, missing fields, a pin that no longer targets
 * this package) throws instead of shipping a half-synced manifest.
 */
export function syncManifestContent(
  content: string,
  pkg: PackageIdentity
): string {
  parseManifest(content);

  const pin = new RegExp(
    `^(\\s*"package"\\s*:\\s*")${escapeRegExp(pkg.name)}@[^"]*(")`
  );
  let versionRewritten = false;
  let pinRewritten = false;
  const rewritten = content
    .split("\n")
    .map((line) => {
      if (!versionRewritten && VERSION_FIELD.test(line)) {
        versionRewritten = true;
        return line.replace(VERSION_FIELD, `$1${pkg.version}$2`);
      }
      if (!pinRewritten && pin.test(line)) {
        pinRewritten = true;
        return line.replace(pin, `$1${pkg.name}@${pkg.version}$2`);
      }
      return line;
    })
    .join("\n");

  // Belt over the line matching above: verify the result semantically.
  const updated = parseManifest(rewritten);
  if (updated.version !== pkg.version || !versionRewritten) {
    throw new Error(
      `"version" field not found (or not rewritable) in ${MANIFEST_PATH} — ` +
        `expected it to become ${pkg.version}`
    );
  }
  if (
    updated.distribution?.npx?.package !== `${pkg.name}@${pkg.version}` ||
    !pinRewritten
  ) {
    throw new Error(
      `distribution.npx.package pin not found (or not targeting ${pkg.name}) ` +
        `in ${MANIFEST_PATH} — expected it to become ${pkg.name}@${pkg.version}`
    );
  }
  return rewritten;
}

function parseManifest(content: string): AgentManifest {
  try {
    return JSON.parse(content) as AgentManifest;
  } catch (error) {
    throw new Error(
      `${MANIFEST_PATH} could not be parsed: ${(error as Error).message}`,
      { cause: error }
    );
  }
}

/**
 * CLI entry. Default mode rewrites the manifest and `git add`s it so the
 * `version` lifecycle hook can fold it into the `npm version` release commit;
 * `--check` only verifies (used by CI) and never writes.
 */
function main(argv: string[]): number {
  const check = argv.includes("--check");
  const pkg = JSON.parse(
    readFileSync("package.json", "utf8")
  ) as PackageIdentity;
  const content = readFileSync(MANIFEST_PATH, "utf8");
  const updated = syncManifestContent(content, pkg);

  if (updated === content) {
    if (!check) {
      console.log(`${MANIFEST_PATH} already at ${pkg.version} — nothing to do`);
    }
    return 0;
  }
  if (check) {
    const manifest = JSON.parse(content) as AgentManifest;
    console.error(`${MANIFEST_PATH} is out of sync with package.json:`);
    console.error(`  version:        ${String(manifest.version)} (expected ${pkg.version})`);
    console.error(
      `  npx package:    ${String(manifest.distribution?.npx?.package)} (expected ${pkg.name}@${pkg.version})`
    );
    console.error(
      `Fix: run \`npm run build && node dist/registry/sync-manifest.js\` and commit the result.`
    );
    return 1;
  }

  writeFileSync(MANIFEST_PATH, updated);
  console.log(`${MANIFEST_PATH} synced to ${pkg.version}`);
  const staged = spawnSync("git", ["add", MANIFEST_PATH]);
  if (staged.status !== 0) {
    console.error(
      `synced ${MANIFEST_PATH} but \`git add\` failed — stage it manually`
    );
    return 1;
  }
  return 0;
}

const invokedDirectly =
  import.meta.url === pathToFileURL(process.argv[1] ?? "").href;
if (invokedDirectly) {
  try {
    process.exitCode = main(process.argv.slice(2));
  } catch (error) {
    console.error((error as Error).message);
    process.exitCode = 1;
  }
}
