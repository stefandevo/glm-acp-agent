/**
 * Discovers the slash commands a session can offer and expands them when the
 * user invokes one.
 *
 * ACP clients fill their `/` autocomplete from the `available_commands_update`
 * notification an agent sends after `session/new` (and load / fork / resume).
 * Commands are then invoked as ordinary prompt text — the client sends
 * `"/commit group by feature"` as a text block and the agent is responsible for
 * recognising the prefix.
 *
 * We read the same on-disk layout Claude Code uses, so a project that already
 * ships commands or skills works without extra configuration:
 *
 *   <cwd>/.claude/commands/<name>.md     → `/name` (nested files become `/dir:name`)
 *   <cwd>/.claude/skills/<name>/SKILL.md → `/name`
 *
 * plus the same two directories under `~/.claude` for user-level commands.
 * Project definitions win over user-level ones with the same name.
 *
 * All I/O errors are swallowed: having no commands at all is the common case,
 * and a broken command file should never fail `session/new`.
 */
import { readdirSync, readFileSync, type Dirent } from "node:fs";
import { homedir } from "node:os";
import { join as pathJoin } from "node:path";

/** A command discovered on disk, ready to advertise and to expand. */
export interface SlashCommand {
  /** Command name *without* a leading slash — the client prepends it. */
  name: string;
  /** One-line description rendered in the client's slash menu. */
  description: string;
  /** Placeholder for the free text typed after the name; absent when the command takes none. */
  argumentHint?: string;
  /** Markdown body injected into the prompt when the command is invoked. */
  body: string;
  /** Absolute path the definition was read from, quoted back to the model. */
  source: string;
}

/**
 * Cap on a single command body embedded into a prompt. Command files are
 * usually a screenful of instructions; this bounds the worst case for a project
 * that keeps an entire playbook in one.
 */
const COMMAND_BODY_CAP_CHARS = 16 * 1024;

/**
 * Cap on the advertised list so a pathological tree can't flood the client — or
 * make `session/new` read tens of thousands of files. Roots are visited in
 * precedence order, so a truncated list keeps the project's own commands.
 */
const MAX_COMMANDS = 200;

/** How deep to walk `.claude/commands`; deeper files are ignored. */
const MAX_COMMAND_DEPTH = 3;

/** Descriptions are shown in a one-line menu entry, so keep them short. */
const DESCRIPTION_CAP_CHARS = 200;

/**
 * Names have to survive being typed after a `/` and split on whitespace, and we
 * build some of them from path segments — so restrict to the characters Claude
 * Code itself allows.
 */
const COMMAND_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_:.-]*$/;

/**
 * Collect the commands available to a session rooted at `cwd`, sorted by name.
 *
 * Project commands shadow user-level ones, and within a root a `commands/` file
 * shadows a same-named skill — first writer wins, so roots are visited in
 * precedence order.
 */
export function discoverSlashCommands(cwd: string): SlashCommand[] {
  const found = new Map<string, SlashCommand>();
  for (const root of [pathJoin(cwd, ".claude"), pathJoin(homedir(), ".claude")]) {
    collectCommandFiles(pathJoin(root, "commands"), [], found);
    collectSkillFiles(pathJoin(root, "skills"), found);
  }
  return [...found.values()].sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Recursively read `*.md` files under a `commands/` directory. Nested files are
 * namespaced with `:` (`commands/review/pr.md` → `review:pr`), matching how
 * Claude Code names subdirectory commands.
 */
function collectCommandFiles(
  dir: string,
  prefix: string[],
  found: Map<string, SlashCommand>
): void {
  if (prefix.length >= MAX_COMMAND_DEPTH) return;
  for (const entry of readDirSafe(dir)) {
    const path = pathJoin(dir, entry.name);
    if (entry.isDirectory()) {
      collectCommandFiles(path, [...prefix, entry.name], found);
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
    addCommand(found, [...prefix, entry.name.slice(0, -".md".length)].join(":"), path);
  }
}

/** Read `<skills>/<name>/SKILL.md` definitions; the directory name is the command name. */
function collectSkillFiles(dir: string, found: Map<string, SlashCommand>): void {
  for (const entry of readDirSafe(dir)) {
    if (!entry.isDirectory()) continue;
    addCommand(found, entry.name, pathJoin(dir, entry.name, "SKILL.md"));
  }
}

function readDirSafe(dir: string): Dirent<string>[] {
  try {
    return readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

function addCommand(
  found: Map<string, SlashCommand>,
  name: string,
  path: string
): void {
  if (found.size >= MAX_COMMANDS) return;
  if (!COMMAND_NAME_PATTERN.test(name) || found.has(name)) return;
  let contents: string;
  try {
    contents = readFileSync(path, { encoding: "utf-8" });
  } catch {
    return;
  }
  const { frontmatter, body } = splitFrontmatter(contents);
  found.set(name, {
    name,
    description: describe(frontmatter["description"], body, name),
    ...(frontmatter["argument-hint"] !== undefined
      ? { argumentHint: frontmatter["argument-hint"] }
      : {}),
    body: body.slice(0, COMMAND_BODY_CAP_CHARS).trim(),
    source: path,
  });
}

/**
 * Split a leading `---` YAML frontmatter block off a markdown file.
 *
 * Deliberately not a YAML parser: command frontmatter in the wild is flat
 * `key: value` lines, and pulling in a dependency to read `description` and
 * `argument-hint` would not pay for itself. Anything we can't parse is simply
 * dropped, and the description falls back to the body.
 */
function splitFrontmatter(contents: string): {
  frontmatter: Record<string, string>;
  body: string;
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(contents);
  if (!match) return { frontmatter: {}, body: contents };
  const frontmatter: Record<string, string> = {};
  for (const line of match[1]!.split(/\r?\n/)) {
    const separator = line.indexOf(":");
    if (separator <= 0) continue;
    const key = line.slice(0, separator).trim();
    const value = line
      .slice(separator + 1)
      .trim()
      .replace(/^["'](.*)["']$/, "$1")
      .trim();
    if (key.length > 0 && value.length > 0) frontmatter[key] = value;
  }
  return { frontmatter, body: contents.slice(match[0].length) };
}

/**
 * ACP requires a non-empty description, so fall back through the body's first
 * heading, then its first prose line, then the command name itself.
 */
function describe(
  fromFrontmatter: string | undefined,
  body: string,
  name: string
): string {
  const candidates = [fromFrontmatter, firstHeading(body), firstProseLine(body)];
  for (const candidate of candidates) {
    const text = candidate?.replace(/\s+/g, " ").trim();
    if (text) return text.slice(0, DESCRIPTION_CAP_CHARS);
  }
  return `Run the /${name} command.`;
}

function firstHeading(body: string): string | undefined {
  return /^#{1,6}[ \t]+(.+)$/m.exec(body)?.[1];
}

function firstProseLine(body: string): string | undefined {
  for (const line of body.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.length > 0 && !trimmed.startsWith("#")) return trimmed;
  }
  return undefined;
}

/** A parsed `/name rest` prompt. */
export interface ParsedSlashCommand {
  command: SlashCommand;
  /** Everything typed after the command name, trimmed; `""` when nothing was. */
  args: string;
}

/**
 * Match a prompt against the known commands. Returns `undefined` for anything
 * that isn't a leading `/name` we advertised, so prose that happens to start
 * with a slash (and unknown commands) still reaches the model untouched.
 */
export function parseSlashCommand(
  text: string,
  commands: ReadonlyArray<SlashCommand>
): ParsedSlashCommand | undefined {
  // Any whitespace separates the name from its arguments: prompt editors let a
  // user press shift-enter after `/name`, so the arguments can start on the
  // next line rather than after a space.
  const match = /^\/(\S+)(?:\s+([\s\S]*))?$/.exec(text.trim());
  if (!match) return undefined;
  const command = commands.find((c) => c.name === match[1]);
  if (!command) return undefined;
  return { command, args: match[2]?.trim() ?? "" };
}

/**
 * Render an invoked command as the user message the model actually sees.
 *
 * The body is presented as instructions rather than as opaque data (unlike
 * `<project_context>`): the user explicitly typed `/name`, so running what that
 * file says *is* the request. `$ARGUMENTS` is substituted where the definition
 * asks for it, which is the convention `.claude/commands` files are written
 * against; otherwise the arguments are appended as their own line.
 */
export function renderSlashCommand(parsed: ParsedSlashCommand): string {
  const { command, args } = parsed;
  const usesPlaceholder = command.body.includes("$ARGUMENTS");
  const body = usesPlaceholder
    ? command.body.replaceAll("$ARGUMENTS", args)
    : command.body;
  const lines = [
    `<slash_command name="${command.name}" source="${command.source}">`,
    `The user invoked the /${command.name} command. Follow the instructions below.`,
    "",
    body,
  ];
  if (args.length > 0 && !usesPlaceholder) {
    lines.push("", `Arguments: ${args}`);
  }
  lines.push("</slash_command>");
  return lines.join("\n");
}
