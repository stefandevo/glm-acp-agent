import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir as osTmpdir } from "node:os";
import { join as pathJoin } from "node:path";

// `discoverSlashCommands` also scans `~/.claude`, so point HOME at an isolated
// tempdir before importing it: without this the developer's own commands would
// leak into every assertion below.
const fakeHome = mkdtempSync(pathJoin(osTmpdir(), "glm-acp-test-home-"));
process.env["HOME"] = fakeHome;

const { discoverSlashCommands, parseSlashCommand, renderSlashCommand } =
  await import("../protocol/slash-commands.js");

/** Create a temp cwd seeded with files given as cwd-relative paths. */
function makeTree(files: Record<string, string>): {
  cwd: string;
  cleanup: () => void;
} {
  const cwd = mkdtempSync(pathJoin(osTmpdir(), "glm-acp-test-cmds-"));
  writeTree(cwd, files);
  return { cwd, cleanup: () => rmSync(cwd, { recursive: true, force: true }) };
}

function writeTree(root: string, files: Record<string, string>): void {
  for (const [relative, content] of Object.entries(files)) {
    const path = pathJoin(root, relative);
    mkdirSync(pathJoin(path, ".."), { recursive: true });
    writeFileSync(path, content);
  }
}

test("discoverSlashCommands reads .claude/commands markdown files", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/commands/commit.md":
      "---\ndescription: Commit staged changes\nargument-hint: <message>\n---\nWrite a conventional commit.\n",
  });
  try {
    const commands = discoverSlashCommands(cwd);
    assert.equal(commands.length, 1);
    assert.equal(commands[0]?.name, "commit");
    assert.equal(commands[0]?.description, "Commit staged changes");
    assert.equal(commands[0]?.argumentHint, "<message>");
    assert.equal(commands[0]?.body, "Write a conventional commit.");
  } finally {
    cleanup();
  }
});

test("discoverSlashCommands reads .claude/skills SKILL.md files", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/skills/review/SKILL.md":
      "---\nname: review\ndescription: Review the working tree\n---\nLook for bugs.\n",
  });
  try {
    const commands = discoverSlashCommands(cwd);
    assert.deepEqual(
      commands.map((c) => c.name),
      ["review"]
    );
    assert.equal(commands[0]?.description, "Review the working tree");
  } finally {
    cleanup();
  }
});

test("discoverSlashCommands namespaces nested command files with a colon", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/commands/review/pr.md": "# Review a pull request\n",
  });
  try {
    const commands = discoverSlashCommands(cwd);
    assert.deepEqual(
      commands.map((c) => c.name),
      ["review:pr"]
    );
  } finally {
    cleanup();
  }
});

test("discoverSlashCommands falls back to the first heading, then the first line", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/commands/headed.md": "# Ship the release\n\nDo the thing.\n",
    ".claude/commands/prose.md": "Just do the thing.\n",
    ".claude/commands/empty.md": "",
  });
  try {
    const byName = new Map(discoverSlashCommands(cwd).map((c) => [c.name, c]));
    assert.equal(byName.get("headed")?.description, "Ship the release");
    assert.equal(byName.get("prose")?.description, "Just do the thing.");
    assert.equal(byName.get("empty")?.description, "Run the /empty command.");
  } finally {
    cleanup();
  }
});

test("discoverSlashCommands omits argumentHint when the definition declares none", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/commands/status.md": "---\ndescription: Show status\n---\nRun git status.\n",
  });
  try {
    assert.equal(discoverSlashCommands(cwd)[0]?.argumentHint, undefined);
  } finally {
    cleanup();
  }
});

test("discoverSlashCommands lets a project command shadow a same-named user command", () => {
  const { cwd, cleanup } = makeTree({
    ".claude/commands/commit.md": "---\ndescription: Project commit\n---\nproject body\n",
  });
  writeTree(fakeHome, {
    ".claude/commands/commit.md": "---\ndescription: User commit\n---\nuser body\n",
    ".claude/commands/publish.md": "---\ndescription: User publish\n---\nuser body\n",
  });
  try {
    const byName = new Map(discoverSlashCommands(cwd).map((c) => [c.name, c]));
    assert.equal(byName.get("commit")?.description, "Project commit");
    // User-level commands the project doesn't shadow still show up.
    assert.equal(byName.get("publish")?.description, "User publish");
  } finally {
    rmSync(pathJoin(fakeHome, ".claude"), { recursive: true, force: true });
    cleanup();
  }
});

test("discoverSlashCommands returns nothing for a cwd with no .claude directory", () => {
  const { cwd, cleanup } = makeTree({ "README.md": "hello" });
  try {
    assert.deepEqual(discoverSlashCommands(cwd), []);
  } finally {
    cleanup();
  }
});

test("parseSlashCommand matches known names and ignores everything else", () => {
  const commands = [
    { name: "commit", description: "d", body: "b", source: "/s" },
    { name: "review:pr", description: "d", body: "b", source: "/s" },
  ];
  assert.equal(parseSlashCommand("/commit", commands)?.command.name, "commit");
  assert.equal(parseSlashCommand("/commit group by feature", commands)?.args, "group by feature");
  assert.equal(parseSlashCommand("/review:pr", commands)?.command.name, "review:pr");
  assert.equal(parseSlashCommand("/commit", commands)?.args, "");
  assert.equal(parseSlashCommand("/commit\nspanning a newline", commands)?.args, "spanning a newline");
  assert.equal(parseSlashCommand("/unknown do a thing", commands), undefined);
  assert.equal(parseSlashCommand("what does /commit do?", commands), undefined);
  assert.equal(parseSlashCommand("plain prose", commands), undefined);
});

test("renderSlashCommand injects the body and appends free-form arguments", () => {
  const command = {
    name: "commit",
    description: "d",
    body: "Write a conventional commit.",
    source: "/tmp/.claude/commands/commit.md",
  };
  const rendered = renderSlashCommand({ command, args: "group by feature" });
  assert.match(rendered, /The user invoked the \/commit command/);
  assert.match(rendered, /Write a conventional commit\./);
  assert.match(rendered, /Arguments: group by feature/);
});

test("renderSlashCommand substitutes $ARGUMENTS instead of appending them", () => {
  const command = {
    name: "fix",
    description: "d",
    body: "Fix the bug described as: $ARGUMENTS",
    source: "/tmp/.claude/commands/fix.md",
  };
  const rendered = renderSlashCommand({ command, args: "flaky login test" });
  assert.match(rendered, /Fix the bug described as: flaky login test/);
  assert.doesNotMatch(rendered, /Arguments:/);
  assert.doesNotMatch(rendered, /\$ARGUMENTS/);
});
