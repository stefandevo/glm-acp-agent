import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  credentialsPath,
  readCredentialsKey,
  resolveApiKey,
  writeCredentials,
} from "../llm/credentials.js";

function withTmp<T>(fn: (dir: string) => T): T {
  const dir = mkdtempSync(join(tmpdir(), "glm-creds-"));
  try {
    return fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

test("credentialsPath honours XDG_CONFIG_HOME", () => {
  const old = process.env["XDG_CONFIG_HOME"];
  try {
    process.env["XDG_CONFIG_HOME"] = "/custom/xdg";
    assert.equal(
      credentialsPath(),
      "/custom/xdg/glm-acp-agent/credentials.json"
    );
  } finally {
    if (old === undefined) delete process.env["XDG_CONFIG_HOME"];
    else process.env["XDG_CONFIG_HOME"] = old;
  }
});

test("readCredentialsKey returns undefined for a missing file", () => {
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    assert.equal(readCredentialsKey(path), undefined);
  });
});

test("readCredentialsKey returns undefined for malformed JSON", () => {
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    writeFileSync(path, "not json");
    assert.equal(readCredentialsKey(path), undefined);
  });
});

test("writeCredentials and readCredentialsKey round-trip", () => {
  withTmp((dir) => {
    const path = join(dir, "nested", "credentials.json");
    writeCredentials("test-key", path);
    assert.equal(readCredentialsKey(path), "test-key");
  });
});

test("writeCredentials writes a 0600-permission file", () => {
  if (process.platform === "win32") return; // POSIX-only
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    writeCredentials("k", path);
    const mode = statSync(path).mode & 0o777;
    assert.equal(mode, 0o600);
  });
});

test("writeCredentials refuses an empty key", () => {
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    assert.throws(() => writeCredentials("", path), /empty/);
  });
});

test("resolveApiKey prefers Z_AI_API_KEY over the credentials file", () => {
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    writeCredentials("from-disk", path);
    const oldEnv = process.env["Z_AI_API_KEY"];
    const oldXdg = process.env["XDG_CONFIG_HOME"];
    try {
      process.env["Z_AI_API_KEY"] = "from-env";
      process.env["XDG_CONFIG_HOME"] = dir; // makes credentialsPath() point here
      // Path doesn't actually need to match; readCredentialsKey() falls through
      // anyway when env wins.
      assert.equal(resolveApiKey(), "from-env");
    } finally {
      if (oldEnv === undefined) delete process.env["Z_AI_API_KEY"];
      else process.env["Z_AI_API_KEY"] = oldEnv;
      if (oldXdg === undefined) delete process.env["XDG_CONFIG_HOME"];
      else process.env["XDG_CONFIG_HOME"] = oldXdg;
    }
  });
});

test("resolveApiKey falls back to the credentials file when env is unset", () => {
  withTmp((dir) => {
    const xdg = join(dir, "xdg");
    const credPath = join(xdg, "glm-acp-agent", "credentials.json");
    writeCredentials("from-disk", credPath);
    const oldEnv = process.env["Z_AI_API_KEY"];
    const oldXdg = process.env["XDG_CONFIG_HOME"];
    try {
      delete process.env["Z_AI_API_KEY"];
      process.env["XDG_CONFIG_HOME"] = xdg;
      assert.equal(resolveApiKey(), "from-disk");
    } finally {
      if (oldEnv !== undefined) process.env["Z_AI_API_KEY"] = oldEnv;
      if (oldXdg === undefined) delete process.env["XDG_CONFIG_HOME"];
      else process.env["XDG_CONFIG_HOME"] = oldXdg;
    }
  });
});
