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

/**
 * Rewrite an agent.json manifest so both version pins match the package.
 *
 * The pins are updated through the parsed object — `version` and
 * `distribution.npx.package` by path — so a nested `"version"` field or a
 * sibling distribution pin elsewhere in the manifest can neither be mistaken
 * for the real pins nor silently clobbered. Anything unexpected —
 * unparseable JSON, a missing field, or a pin no longer targeting this
 * package — throws instead of shipping a half-synced manifest.
 *
 * Output uses normalized `JSON.stringify` formatting: an unnormalized input
 * is rewritten once, and every later sync is byte-stable.
 */
export function syncManifestContent(
  content: string,
  pkg: PackageIdentity
): string {
  const manifest = parseManifest(content);
  if (typeof manifest.version !== "string") {
    throw new Error(
      `"version" field not found (or not a string) in ${MANIFEST_PATH}`
    );
  }
  const npx = manifest.distribution?.npx;
  if (
    !npx ||
    typeof npx.package !== "string" ||
    !npx.package.startsWith(`${pkg.name}@`)
  ) {
    throw new Error(
      `distribution.npx.package pin not found (or not targeting ${pkg.name}) in ${MANIFEST_PATH}`
    );
  }
  manifest.version = pkg.version;
  npx.package = `${pkg.name}@${pkg.version}`;
  return `${JSON.stringify(manifest, null, 2)}\n`;
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

const describeValue = (value: unknown) =>
  typeof value === "string" ? value : "(missing)";

/**
 * CLI entry. Default mode rewrites the manifest and `git add`s it so the
 * `version` lifecycle hook can fold it into the `npm version` release commit;
 * `--check` only verifies (used by CI) and never writes. Both modes compare
 * pin values, not file formatting.
 */
function main(argv: string[]): number {
  const check = argv.includes("--check");
  const pkg = JSON.parse(
    readFileSync("package.json", "utf8")
  ) as PackageIdentity;
  const content = readFileSync(MANIFEST_PATH, "utf8");

  if (check) {
    const manifest = parseManifest(content);
    const pin = manifest.distribution?.npx?.package;
    if (
      manifest.version === pkg.version &&
      pin === `${pkg.name}@${pkg.version}`
    ) {
      return 0;
    }
    console.error(`${MANIFEST_PATH} is out of sync with package.json:`);
    console.error(
      `  version:        ${describeValue(manifest.version)} (expected ${pkg.version})`
    );
    console.error(
      `  npx package:    ${describeValue(pin)} (expected ${pkg.name}@${pkg.version})`
    );
    console.error(
      `Fix: run \`npm run build && node dist/registry/sync-manifest.js\` and commit the result.`
    );
    return 1;
  }

  const updated = syncManifestContent(content, pkg);
  if (updated === content) {
    console.log(`${MANIFEST_PATH} already at ${pkg.version} — nothing to do`);
    return 0;
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
