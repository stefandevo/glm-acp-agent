import { readdir } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const testsDirectory = join(repositoryRoot, "dist", "tests");
const entries = await readdir(testsDirectory, { withFileTypes: true });
const testFiles = entries
  .filter((entry) => entry.isFile() && entry.name.endsWith(".test.js"))
  .map((entry) => join(testsDirectory, entry.name))
  .sort();

if (testFiles.length === 0) {
  throw new Error(`No compiled test files found in ${testsDirectory}`);
}

const child = spawn(process.execPath, ["--test", ...testFiles], {
  cwd: repositoryRoot,
  stdio: "inherit",
  // Keep tests independent of any global context file in the developer's home.
  env: { ...process.env, ACP_GLM_GLOBAL_CONTEXT_FILE: "none" },
});

child.on("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});

child.on("exit", (code, signal) => {
  if (signal !== null) {
    console.error(`Test runner terminated by ${signal}`);
    process.exitCode = 1;
  } else {
    process.exitCode = code ?? 1;
  }
});
