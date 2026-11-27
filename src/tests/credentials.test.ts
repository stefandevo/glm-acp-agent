import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough, Readable } from "node:stream";
import {
  credentialsPath,
  readCredentialsKey,
  readCredentialsTeamSeat,
  resolveApiKey,
  resolveTeamSeat,
  writeCredentials,
} from "../llm/credentials.js";
import { runSetup } from "../setup.js";

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
  const configuredRoot = join(tmpdir(), "custom", "xdg");
  try {
    process.env["XDG_CONFIG_HOME"] = configuredRoot;
    assert.equal(
      credentialsPath(),
      join(configuredRoot, "glm-acp-agent", "credentials.json")
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

test("writeCredentials restricts an existing credentials file before replacing it", () => {
  if (process.platform === "win32") return; // POSIX-only
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    writeFileSync(path, '{"z_ai_api_key":"old-key"}\n', { mode: 0o644 });

    writeCredentials("new-key", path);

    assert.equal(statSync(path).mode & 0o777, 0o600);
    assert.equal(readCredentialsKey(path), "new-key");
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

function withEnv<T>(vars: Record<string, string | undefined>, fn: () => T): T {
  const old = Object.fromEntries(Object.keys(vars).map((name) => [name, process.env[name]]));
  try {
    for (const [name, value] of Object.entries(vars)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
    return fn();
  } finally {
    for (const [name, value] of Object.entries(old)) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  }
}

test("writeCredentials stores an optional team seat and refuses half of one", () => {
  withTmp((dir) => {
    const path = join(dir, "credentials.json");
    writeCredentials("k", path, { organizationId: "org-placeholder", projectId: "proj-placeholder" });
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), {
      z_ai_api_key: "k", z_ai_organization_id: "org-placeholder", z_ai_project_id: "proj-placeholder",
    });
    assert.deepEqual(readCredentialsTeamSeat(path), { organizationId: "org-placeholder", projectId: "proj-placeholder" });
    assert.throws(() => writeCredentials("k", path, { organizationId: "org-placeholder" }), /both/);
    writeCredentials("k", path);
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), { z_ai_api_key: "k" });
    assert.deepEqual(readCredentialsTeamSeat(path), {});
  });
});

test("resolveTeamSeat takes the env pair as a unit over the credentials file", () => {
  withTmp((dir) => {
    writeCredentials("k", join(dir, "glm-acp-agent", "credentials.json"),
      { organizationId: "org-file", projectId: "proj-file" });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: undefined, Z_AI_PROJECT_ID: undefined }, () => {
      assert.deepEqual(resolveTeamSeat(), { organizationId: "org-file", projectId: "proj-file" });
    });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: "org-env", Z_AI_PROJECT_ID: "proj-env" }, () => {
      assert.deepEqual(resolveTeamSeat(), { organizationId: "org-env", projectId: "proj-env" });
    });
    // A lone env value is reported as-is (half-configured) rather than mixed with the file.
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: "org-env", Z_AI_PROJECT_ID: "" }, () => {
      assert.deepEqual(resolveTeamSeat(), { organizationId: "org-env" });
    });
  });
});

test("resolveTeamSeat lets blank env overrides clear a stored credentials-file seat", () => {
  withTmp((dir) => {
    writeCredentials("k", join(dir, "glm-acp-agent", "credentials.json"),
      { organizationId: "org-file", projectId: "proj-file" });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: "", Z_AI_PROJECT_ID: "" }, () => {
      assert.deepEqual(resolveTeamSeat(), {});
    });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: "   ", Z_AI_PROJECT_ID: "\t" }, () => {
      assert.deepEqual(resolveTeamSeat(), {});
    });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: "", Z_AI_PROJECT_ID: undefined }, () => {
      assert.deepEqual(resolveTeamSeat(), {});
    });
    withEnv({ XDG_CONFIG_HOME: dir, Z_AI_ORGANIZATION_ID: undefined, Z_AI_PROJECT_ID: "  " }, () => {
      assert.deepEqual(resolveTeamSeat(), {});
    });
  });
});

test("runSetup stores a team seat from piped input, or none when left blank", async () => {
  const dir = mkdtempSync(join(tmpdir(), "glm-setup-"));
  const old = process.env["XDG_CONFIG_HOME"];
  try {
    process.env["XDG_CONFIG_HOME"] = dir;
    const path = join(dir, "glm-acp-agent", "credentials.json");
    await runSetup(Readable.from(["setup-key\norg-placeholder\nproj-placeholder\n"]), new PassThrough());
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), {
      z_ai_api_key: "setup-key", z_ai_organization_id: "org-placeholder", z_ai_project_id: "proj-placeholder",
    });
    await runSetup(Readable.from(["setup-key\n"]), new PassThrough());
    assert.deepEqual(JSON.parse(readFileSync(path, "utf8")), { z_ai_api_key: "setup-key" });
    await assert.rejects(runSetup(Readable.from(["setup-key\norg-placeholder\n\n"]), new PassThrough()), /both/);
  } finally {
    if (old === undefined) delete process.env["XDG_CONFIG_HOME"];
    else process.env["XDG_CONFIG_HOME"] = old;
    rmSync(dir, { recursive: true, force: true });
  }
});
