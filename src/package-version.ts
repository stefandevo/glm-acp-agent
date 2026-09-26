import { readFileSync } from "node:fs";

interface PackageMetadata {
  version: string;
}

const packageMetadata = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8")
) as PackageMetadata;

export const PACKAGE_VERSION = packageMetadata.version;
