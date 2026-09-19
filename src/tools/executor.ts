import type {
  AgentSideConnection,
  ClientCapabilities,
} from "@agentclientprotocol/sdk";
import { spawn } from "node:child_process";
import type { Dirent } from "node:fs";
import { lstat, opendir, writeFile } from "node:fs/promises";
import { join as pathJoin, resolve as pathResolve } from "node:path";
import { resolveApiKey } from "../llm/credentials.js";
import {
  callZaiMcpTool,
  ZAI_WEB_READER_MCP_ENDPOINT,
  ZAI_WEB_SEARCH_MCP_ENDPOINT,
} from "./zai-mcp-client.js";
import type { SessionMcpTools } from "./session-mcp-client.js";
import type { VisionMcpClient } from "./vision-mcp-client.js";
import type { SessionModeId } from "../protocol/agent.js";
import {
  readCommandLimits,
  type CommandLimits,
} from "./command-limits.js";
import { readResourceLimits, type ResourceLimits } from "./resource-limits.js";
import { boundToolResult, takeUtf8Prefix } from "./tool-output.js";
import { readLocalTextFileBounded, readLocalTextPage, type TextPage } from "./file-reader.js";

/**
 * Result returned after executing a tool call against the ACP client.
 */
export interface ToolResult {
  content: string;
}

/**
 * Executes GLM tool calls from inside the agent process.
 *
 * Permission is requested from the user before any write or execute operation,
 * while read/list operations and approved writes/commands run locally with
 * paths resolved relative to the ACP session cwd.
 */
export interface TodoItem {
  content: string;
  status: "pending" | "in_progress" | "completed";
  activeForm?: string;
}

/** Default page size for read_file results fed back to the model. */
const DEFAULT_READ_LIMIT = 2000;
/** Upper bound for an explicit limit — keeps one call from flooding the context. */
const HARD_READ_LIMIT = 5000;
/** Valid ACP line number that is beyond any practical editor buffer. */
const EDITOR_EOF_PROBE_LINE = 0xffffffff;
/** Strings longer than this are elided in client-facing previews (UI cards), never in tool results. */
const PREVIEW_STRING_LIMIT = 240;
const PREVIEW_HEAD = 120;

/**
 * Elide a single long string for client-facing display (UI cards, read
 * previews) — never for tool results or permission prompts.
 */
function elideStringForPreview(value: string): string {
  if (value.length <= PREVIEW_STRING_LIMIT) return value;
  return `${value.slice(0, PREVIEW_HEAD)}… [${value.length} chars]`;
}

/**
 * Elide long strings in a rawInput/rawOutput payload so client UI cards stay
 * compact (a whole-file write would otherwise render the entire file in chat).
 * The full payload still reaches the model through the tool result channel.
 */
function elideForPreview(value: unknown): unknown {
  if (typeof value === "string") {
    return elideStringForPreview(value);
  }
  if (Array.isArray(value)) return value.map(elideForPreview);
  if (typeof value === "object" && value !== null) {
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      out[key] = elideForPreview(item);
    }
    return out;
  }
  return value;
}

export class ToolExecutor {
  constructor(
    private connection: AgentSideConnection,
    private sessionId: string,
    private clientCapabilities: ClientCapabilities | null = null,
    private signal?: AbortSignal,
    private visionClient: VisionMcpClient | null = null,
    private sessionMcpTools: SessionMcpTools | null = null,
    private sessionCwd: string = process.cwd(),
    private getMode: () => SessionModeId = () => "default",
    private setTodos: (todos: TodoItem[]) => void = () => undefined,
    private resourceLimits: ResourceLimits = readResourceLimits(),
  ) {}

  /**
   * Dispatch a tool call from GLM to the appropriate ACP Client method.
   *
   * Returns a plain text result that can be fed back to GLM as a tool message.
   */
  async execute(
    toolCallId: string,
    toolName: string,
    rawArguments: string
  ): Promise<ToolResult> {
    let args: Record<string, unknown>;
    try {
      args =
        rawArguments.trim().length === 0
          ? {}
          : (JSON.parse(rawArguments) as Record<string, unknown>);
    } catch {
      const message = `Error: could not parse tool arguments as JSON: ${rawArguments}`;
      await this.failedToolCall(toolCallId, toolName, {}, message);
      return { content: boundToolResult(message, this.resourceLimits.toolResultBytes) };
    }

    const result = await (async (): Promise<ToolResult> => { switch (toolName) {
      case "read_file":
        return this.readFile(toolCallId, args);
      case "write_file":
        return this.writeFile(toolCallId, args);
      case "edit_file":
        return this.editFile(toolCallId, args);
      case "list_files":
        return this.listFiles(toolCallId, args);
      case "run_command":
        return this.runCommand(toolCallId, args);
      case "web_search":
        return this.webSearch(toolCallId, args);
      case "web_reader":
        return this.webReader(toolCallId, args);
      case "image_analysis":
        return this.imageAnalysis(toolCallId, args);
      case "todowrite":
        return this.todoWrite(toolCallId, args);
      default: {
        if (this.sessionMcpTools?.hasTool(toolName)) {
          return this.sessionMcpTool(toolCallId, toolName, args);
        }
        const message = `Error: unknown tool "${toolName}"`;
        await this.failedToolCall(toolCallId, toolName, args, message);
        return { content: message };
      }
    }} )();
    // This is the sole boundary before a result becomes a model-history tool
    // message. Permission arguments and write payloads never cross this path.
    return { content: boundToolResult(result.content, this.resourceLimits.toolResultBytes) };
  }

  // ---------------------------------------------------------------------------
  // Private tool implementations
  // ---------------------------------------------------------------------------

  private async readFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? "").trim();
    if (!path) {
      return this.failAndReturn(toolCallId, "read_file", args, "Error: `path` is required.");
    }
    const offset = Math.max(1, Math.floor(Number(args["offset"] ?? 1)) || 1);
    const limitArg = Number(args["limit"] ?? DEFAULT_READ_LIMIT);
    const limit = Math.min(
      HARD_READ_LIMIT,
      Math.max(1, Math.floor(limitArg) || DEFAULT_READ_LIMIT)
    );
    const absolutePath = this.resolvePath(path);

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Read file: ${path}`,
        kind: "read",
        status: "in_progress",
        locations: [{ path }],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const page = await this.readTextPage(absolutePath, offset, limit);
      if (page.totalLines !== undefined && offset > page.totalLines) {
        const content = `[end of file: offset ${offset} is beyond the last line of ${path} (${page.totalLines} line${page.totalLines === 1 ? "" : "s"})]`;
        await this.connection.sessionUpdate({
          sessionId: this.sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId,
            status: "completed",
            content: [{ type: "content", content: { type: "text", text: content } }],
            rawOutput: elideForPreview({ content }),
          },
        });
        return { content };
      }

      let content = page.text;
      const shown = page.lastCompleteLine >= page.firstLine
        ? `${page.firstLine}-${page.lastCompleteLine}`
        : "none";
      if (page.incompleteLine !== undefined) {
        content += `${content ? "\n" : ""}[showing complete lines ${shown}; line ${page.incompleteLine} is incomplete because the ${this.resourceLimits.fileReadBytes}-byte scan limit was reached. Narrow the input or use an explicitly bounded command for byte-level inspection.]`;
      } else if (page.truncated) {
        content += `\n[scan stopped at the ${this.resourceLimits.fileReadBytes}-byte read limit after line ${page.lastCompleteLine}; total lines are unknown${page.nextLine === undefined ? ". Narrow the input or use an explicitly bounded command for byte-level inspection." : `; pass offset=${page.nextLine} to continue`} ]`;
      } else if (page.nextLine !== undefined) {
        content += `\n[showing lines ${shown}${page.totalLines === undefined ? " (total unknown)" : ` of ${page.totalLines}`}; pass offset=${page.nextLine} to read the next chunk]`;
      } else if (page.totalLines !== undefined && offset > 1) {
        content += `\n[showing lines ${shown} of ${page.totalLines}; end of file]`;
      }

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [
            { type: "content", content: { type: "text", text: elideStringForPreview(content) } },
          ],
          rawOutput: elideForPreview({ content }),
        },
      });

      return { content };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error reading file: ${message}` };
    }
  }

  private async todoWrite(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const rawTodos = args["todos"];
    if (!Array.isArray(rawTodos)) {
      return this.failAndReturn(
        toolCallId,
        "todowrite",
        args,
        "Error: `todos` must be an array of { content, status, activeForm? }."
      );
    }
    const todos: TodoItem[] = [];
    for (const raw of rawTodos) {
      if (typeof raw !== "object" || raw === null) {
        return this.failAndReturn(toolCallId, "todowrite", args, "Error: each todo must be an object.");
      }
      const content = String((raw as Record<string, unknown>)["content"] ?? "").trim();
      const status = String((raw as Record<string, unknown>)["status"] ?? "");
      const activeFormRaw = (raw as Record<string, unknown>)["active_form"] ?? (raw as Record<string, unknown>)["activeForm"];
      const activeForm = activeFormRaw === undefined ? undefined : String(activeFormRaw);
      if (!content) {
        return this.failAndReturn(toolCallId, "todowrite", args, "Error: each todo requires non-empty `content`.");
      }
      if (status !== "pending" && status !== "in_progress" && status !== "completed") {
        return this.failAndReturn(
          toolCallId,
          "todowrite",
          args,
          "Error: `status` must be one of pending, in_progress, completed."
        );
      }
      todos.push({ content, status, activeForm });
    }
    if (todos.length === 0) {
      return this.failAndReturn(toolCallId, "todowrite", args, "Error: `todos` must not be empty.");
    }
    this.setTodos(todos);

    const rendered = todos
      .map((todo, index) => {
        const marker =
          todo.status === "completed" ? "[x]" : todo.status === "in_progress" ? "[>]" : "[ ]";
        return `${index + 1}. ${marker} ${todo.content}${todo.activeForm ? ` (${todo.activeForm})` : ""}`;
      })
      .join("\n");

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: todos.some((t) => t.status === "in_progress")
          ? `Task list: ${todos.find((t) => t.status === "in_progress")?.activeForm ?? todos.find((t) => t.status === "in_progress")?.content ?? ""}`
          : `Task list: ${todos.length} item${todos.length === 1 ? "" : "s"}`,
        kind: "other",
        status: "completed",
        rawInput: elideForPreview(args),
        rawOutput: elideForPreview({ todos }),
      },
    });

    return { content: `Todo list updated:\n${rendered}` };
  }

  private async writeFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? "").trim();
    const content = String(args["content"] ?? "");
    if (!path) {
      return this.failAndReturn(toolCallId, "write_file", args, "Error: `path` is required.");
    }
    const absolutePath = this.resolvePath(path);

    // Step 1: announce the pending tool call so the client can show it.
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Write file: ${path}`,
        kind: "edit",
        status: "pending",
        locations: [{ path }],
        rawInput: elideForPreview(args),
      },
    });

    // Step 2: request user permission based on the current session mode. The
    // prompt must show the full payload: an approval decides on exactly what
    // will run, so elision is reserved for sessionUpdate UI cards (step 1).
    const permissionResult = await this.maybeRequestPermission({
      toolCallId,
      kind: "write",
      rawInput: args,
      title: `Write file: ${path}`,
      locations: [{ path }],
    });

    if (permissionResult.type === "error") {
      const message = `Error requesting permission: ${permissionResult.message}`;
      await this.markFailed(toolCallId, message);
      return { content: message };
    }
    if (permissionResult.type === "cancelled") {
      await this.markFailed(toolCallId, "Cancelled by user.");
      return { content: "Write cancelled by user." };
    }
    if (permissionResult.type === "aborted") {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Write cancelled by turn." };
    }
    if (permissionResult.type === "reject") {
      await this.markFailed(toolCallId, "Rejected by user.");
      return { content: "Write rejected by user." };
    }
    if (this.signal?.aborted) {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Write cancelled by turn." };
    }

    // Step 3: move to in_progress and execute.
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "in_progress",
      },
    });

    try {
      if (this.signal?.aborted) {
        await this.markFailed(toolCallId, "Cancelled by turn.");
        return { content: "Write cancelled by turn." };
      }
      await this.performWrite(absolutePath, content);

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          rawOutput: { success: true },
        },
      });

      return { content: `File written successfully: ${path}` };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error writing file: ${message}` };
    }
  }

  /**
   * Route the actual write through the ACP client when it advertises
   * `fs.writeTextFile` (e.g. Zed), so edits land in the client's buffer and
   * render as native editor diffs. Fall back to writing from the agent process
   * when the client has no fs capability.
   */
  private async performWrite(path: string, content: string): Promise<void> {
    if (this.clientCapabilities?.fs?.writeTextFile) {
      await this.connection.writeTextFile({ sessionId: this.sessionId, path, content });
      return;
    }
    await writeFile(path, content, "utf8");
  }

  /**
   * Mirror of performWrite for reads, used by read_file and edit_file: when the client
   * advertises BOTH `fs.readTextFile` and `fs.writeTextFile`, read through the
   * client so the edit is computed against the same contents the user sees (a
   * dirty editor buffer). Reading a client buffer we cannot write back would
   * leave the editor showing stale content while disk diverges, so a
   * read-without-write capability falls back to plain agent-process disk I/O.
   */
  private async performRead(path: string): Promise<string> {
    if (
      this.clientCapabilities?.fs?.readTextFile &&
      this.clientCapabilities?.fs?.writeTextFile
    ) {
      const response = await this.connection.readTextFile({ sessionId: this.sessionId, path });
      if (Buffer.byteLength(response.content, "utf8") > this.resourceLimits.fileReadBytes) {
        throw new Error(`editor buffer exceeds the ${this.resourceLimits.fileReadBytes}-byte read/edit limit`);
      }
      return response.content;
    }
    return readLocalTextFileBounded(path, this.resourceLimits.fileReadBytes, this.signal);
  }

  private async readTextPage(path: string, offset: number, limit: number): Promise<TextPage> {
    if (this.clientCapabilities?.fs?.readTextFile && this.clientCapabilities?.fs?.writeTextFile) {
      // ACP's line/limit form makes the editor responsible for paging. One
      // lookahead line tells us whether to advertise another request; no page
      // is misrepresented as a whole-buffer line count.
      const readEditorLines = async (line: number, pageLimit: number): Promise<string[]> => {
        const response = await this.connection.readTextFile({
          sessionId: this.sessionId, path, line, limit: pageLimit,
        } as never);
        const lines = response.content.split("\n");
        if (lines.length > 0 && lines.at(-1) === "") lines.pop();
        return lines;
      };
      let lines = await readEditorLines(offset, limit + 1);
      let legacyFullBuffer = false;
      if (offset > 1 && lines.length > 0 && lines.length <= limit + 1) {
        // A few older ACP clients ignore line/limit and return a short full
        // buffer. A short file is indistinguishable from a conforming page,
        // so probe a far-beyond-EOF line before deciding which line numbers
        // the response represents. ACP defines line as a uint32, so this is
        // valid for conforming clients and cannot be a real file line here.
        const probe = await readEditorLines(EDITOR_EOF_PROBE_LINE, 1);
        if (probe.length > 0) {
          lines = probe;
          legacyFullBuffer = true;
        }
      }
      // Older ACP clients ignore line/limit and return the complete buffer.
      // Retain their correct local pagination instead of treating the first
      // lines as the requested offset; conforming clients stay on the bounded
      // lookahead path below.
      if (legacyFullBuffer || lines.length > limit + 1) {
        const totalLines = lines.length;
        if (offset > totalLines) return { text: "", firstLine: offset, lastCompleteLine: totalLines, totalLines, truncated: false };
        const visible = lines.slice(offset - 1, offset - 1 + limit);
        const end = offset + visible.length - 1;
        return { text: visible.join("\n"), firstLine: offset, lastCompleteLine: end, totalLines,
          ...(end < totalLines ? { nextLine: end + 1 } : {}), truncated: false };
      }
      const hasNext = lines.length > limit;
      const visible = lines.slice(0, limit);
      return { text: visible.join("\n"), firstLine: offset, lastCompleteLine: offset + visible.length - 1,
        ...(hasNext ? { nextLine: offset + visible.length } : {}), truncated: false };
    }
    return readLocalTextPage(path, offset, limit, this.resourceLimits.fileReadBytes, this.signal);
  }

  private async editFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? "").trim();
    const oldText = String(args["old_text"] ?? "");
    const newText = String(args["new_text"] ?? "");
    if (!path) {
      return this.failAndReturn(toolCallId, "edit_file", args, "Error: `path` is required.");
    }
    if (oldText.length === 0) {
      return this.failAndReturn(
        toolCallId,
        "edit_file",
        args,
        "Error: `old_text` is required and must be a non-empty exact snippet from the file."
      );
    }
    const absolutePath = this.resolvePath(path);

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Edit file: ${path}`,
        kind: "edit",
        status: "pending",
        locations: [{ path }],
        rawInput: elideForPreview(args),
      },
    });

    let current: string;
    try {
      current = await this.performRead(absolutePath);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error editing file: cannot read ${path}: ${message}` };
    }

    const occurrences = current.split(oldText).length - 1;
    if (occurrences === 0) {
      await this.markFailed(toolCallId, "old_text not found in file");
      return {
        content: `Error editing file: \`old_text\` was not found in ${path}. Re-read the file and copy the snippet exactly, including whitespace.`,
      };
    }
    if (occurrences > 1) {
      await this.markFailed(toolCallId, `old_text matches ${occurrences} locations`);
      return {
        content: `Error editing file: \`old_text\` occurs ${occurrences} times in ${path}. Add surrounding lines to make it unique, then retry.`,
      };
    }

    // Full payload in the prompt — the user approves the exact edit (see
    // writeFile; elision is for sessionUpdate cards only).
    const permissionResult = await this.maybeRequestPermission({
      toolCallId,
      kind: "write",
      rawInput: args,
      title: `Edit file: ${path}`,
      locations: [{ path }],
    });

    if (permissionResult.type === "error") {
      const message = `Error requesting permission: ${permissionResult.message}`;
      await this.markFailed(toolCallId, message);
      return { content: message };
    }
    if (permissionResult.type === "cancelled") {
      await this.markFailed(toolCallId, "Cancelled by user.");
      return { content: "Edit cancelled by user." };
    }
    if (permissionResult.type === "aborted") {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Edit cancelled by turn." };
    }
    if (permissionResult.type === "reject") {
      await this.markFailed(toolCallId, "Rejected by user.");
      return { content: "Edit rejected by user." };
    }
    if (this.signal?.aborted) {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Edit cancelled by turn." };
    }

    // The permission prompt can sit in front of the user for a while; re-read
    // and re-validate so a buffer edited while deciding is not silently
    // overwritten by this stale snapshot.
    let latest: string;
    try {
      latest = await this.performRead(absolutePath);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error editing file: cannot re-read ${path}: ${message}` };
    }
    const latestOccurrences = latest.split(oldText).length - 1;
    if (latestOccurrences !== 1) {
      const reason =
        latestOccurrences === 0
          ? "`old_text` is no longer present"
          : `\`old_text\` now occurs ${latestOccurrences} times`;
      await this.markFailed(toolCallId, `file changed while waiting for permission (${reason})`);
      return {
        content: `Error editing file: ${path} changed while waiting for permission (${reason}). Re-read the file and retry.`,
      };
    }

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "in_progress",
      },
    });

    try {
      if (this.signal?.aborted) {
        await this.markFailed(toolCallId, "Cancelled by turn.");
        return { content: "Edit cancelled by turn." };
      }
      await this.performWrite(absolutePath, latest.replace(oldText, () => newText));

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          rawOutput: { success: true },
        },
      });

      return { content: `File edited successfully: ${path}` };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error editing file: ${message}` };
    }
  }

  private async listFiles(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const rawPath = args["path"];
    if (typeof rawPath !== "string" || rawPath.trim().length === 0) {
      return this.failAndReturn(
        toolCallId,
        "list_files",
        args,
        "Error listing files: `path` must be a non-empty string."
      );
    }
    const path = rawPath.trim();
    const absolutePath = this.resolvePath(path);

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `List files: ${path}`,
        kind: "read",
        status: "in_progress",
        locations: [{ path }],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const directory = await opendir(absolutePath);
      const entries: Dirent[] = [];
      let entryLimitReached = false;
      try {
        for await (const entry of directory) {
          if (entries.length >= this.resourceLimits.listEntries) {
            entryLimitReached = true;
            break;
          }
          entries.push(entry);
        }
      } finally {
        await directory.close().catch(() => undefined);
      }
      const sorted = entries.sort((a, b) => a.name.localeCompare(b.name));
      const lines = await mapWithConcurrency(sorted, this.resourceLimits.fsConcurrency, async entry => {
        const info = await lstat(pathJoin(absolutePath, entry.name));
        const type = entry.isDirectory() ? "dir" : entry.isSymbolicLink() ? "link" : "file";
        return `${type}\t${info.size}\t${entry.name}`;
      });
      const outputLines = [`Listing for ${path} (${absolutePath})`];
      let byteLimitReached = Buffer.byteLength(outputLines[0]!, "utf8") > this.resourceLimits.listBytes;
      if (!byteLimitReached) {
        for (const line of lines) {
          if (Buffer.byteLength([...outputLines, line].join("\n"), "utf8") > this.resourceLimits.listBytes) {
            byteLimitReached = true;
            break;
          }
          outputLines.push(line);
        }
      }
      const marker = `[listing truncated: returned a subset; entries=${this.resourceLimits.listEntries}, bytes=${this.resourceLimits.listBytes}]`;
      const listingTruncated = entryLimitReached || byteLimitReached;
      const output = listingTruncated
        ? (() => {
            const markerBytes = Buffer.byteLength(marker, "utf8");
            const prefixBudget = Math.max(0, this.resourceLimits.listBytes - markerBytes - 1);
            const prefix = takeUtf8Prefix(outputLines.join("\n"), prefixBudget);
            return `${prefix ? `${prefix}\n` : ""}${marker}`;
          })()
        : outputLines.join("\n");

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: elideForPreview({ output }),
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error listing files: ${message}` };
    }
  }

  private async runCommand(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const command = String(args["command"] ?? "").trim();
    if (!command) {
      return this.failAndReturn(
        toolCallId,
        "run_command",
        args,
        "Error running command: command must be a non-empty string."
      );
    }

    // Step 1: announce.
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Run command: ${command}`,
        kind: "execute",
        status: "pending",
        locations: [],
        rawInput: elideForPreview(args),
      },
    });

    // Step 2: request permission based on the current session mode. Full
    // payload again: the approval must see the whole command line.
    const permissionResult = await this.maybeRequestPermission({
      toolCallId,
      kind: "execute",
      rawInput: args,
      title: `Run command: ${command}`,
      locations: [],
    });

    if (permissionResult.type === "error") {
      const message = `Error requesting permission: ${permissionResult.message}`;
      await this.markFailed(toolCallId, message);
      return { content: message };
    }
    if (permissionResult.type === "cancelled") {
      await this.markFailed(toolCallId, "Cancelled by user.");
      return { content: "Command cancelled by user." };
    }
    if (permissionResult.type === "aborted") {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Command cancelled by turn." };
    }
    if (permissionResult.type === "reject") {
      await this.markFailed(toolCallId, "Rejected by user.");
      return { content: "Command rejected by user." };
    }

    return this.runLocalCommand(toolCallId, command);
  }

  private async runLocalCommand(
    toolCallId: string,
    command: string
  ): Promise<ToolResult> {
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "in_progress",
      },
    });

    if (this.signal?.aborted) {
      await this.markFailed(toolCallId, "Cancelled by turn.");
      return { content: "Command cancelled by turn." };
    }

    try {
      const limits = readCommandLimits();
      const result = await runShellCommand(
        command,
        this.sessionCwd,
        this.signal,
        limits
      );
      const output = formatCommandOutput(result, limits);

      if (result.timedOut) {
        await this.connection.sessionUpdate({
          sessionId: this.sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId,
            status: "failed",
            content: [{ type: "content", content: { type: "text", text: output } }],
            rawOutput: elideForPreview(result),
          },
        });
        return { content: output };
      }

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: elideForPreview(result),
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error running command: ${message}` };
    }
  }

  private resolvePath(path: string): string {
    return pathResolve(this.sessionCwd, path);
  }

  private async webSearch(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const query = String(args["query"] ?? "").trim();
    const count = typeof args["count"] === "number" ? args["count"] : undefined;
    if (!query) {
      return this.failAndReturn(
        toolCallId,
        "web_search",
        args,
        "Error: `query` is required."
      );
    }

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Web search: ${query}`,
        kind: "fetch",
        status: "in_progress",
        locations: [],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const apiKey = requireResolvedApiKey();
      const toolArgs: Record<string, unknown> = { query };
      if (count !== undefined) toolArgs["count"] = count;

      const mcpResult = await callZaiMcpTool({
        endpoint: ZAI_WEB_SEARCH_MCP_ENDPOINT,
        toolName: "webSearchPrime",
        arguments: toolArgs,
        apiKey,
        signal: this.signal,
      });

      const { output, resultCount } = formatSearchOutput(mcpResult);

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: elideForPreview({ resultCount }),
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error performing web search: ${message}` };
    }
  }

  private async webReader(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const url = String(args["url"] ?? "").trim();
    const returnFormat = String(args["return_format"] ?? "markdown");
    if (!url) {
      return this.failAndReturn(
        toolCallId,
        "web_reader",
        args,
        "Error: `url` is required."
      );
    }

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Read URL: ${url}`,
        kind: "fetch",
        status: "in_progress",
        locations: [{ path: url }],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const apiKey = requireResolvedApiKey();

      const mcpResult = await callZaiMcpTool({
        endpoint: ZAI_WEB_READER_MCP_ENDPOINT,
        toolName: "webReader",
        arguments: { url, return_format: returnFormat },
        apiKey,
        signal: this.signal,
      });

      const { output, title, resultUrl } = formatReaderOutput(mcpResult);

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: elideForPreview({ title, url: resultUrl }),
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error reading URL: ${message}` };
    }
  }

  private async imageAnalysis(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const imageSource = String(args["image_source"] ?? "").trim();
    const prompt = typeof args["prompt"] === "string" ? args["prompt"] : undefined;
    if (!imageSource) {
      return this.failAndReturn(
        toolCallId,
        "image_analysis",
        args,
        "Error: `image_source` is required."
      );
    }
    if (!this.visionClient) {
      return this.failAndReturn(
        toolCallId,
        "image_analysis",
        args,
        "Error: vision is not configured on this agent process. Vision MCP requires `npx` and the Z.AI Coding Plan."
      );
    }

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Analyze image: ${imageSource}`,
        kind: "fetch",
        status: "in_progress",
        locations: [{ path: imageSource }],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const visionArgs: Record<string, unknown> = { image_source: imageSource };
      if (prompt) visionArgs["prompt"] = prompt;
      const mcpResult = await this.visionClient.callTool("image_analysis", visionArgs, this.signal);
      const text = unwrapVisionText(mcpResult);

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text } }],
          rawOutput: elideForPreview({ text }),
        },
      });
      return { content: text };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error analyzing image: ${message}` };
    }
  }

  private async sessionMcpTool(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: toolName,
        kind: "other",
        status: "in_progress",
        locations: [],
        rawInput: elideForPreview(args),
      },
    });

    try {
      const mcpResult = await this.sessionMcpTools!.callTool(toolName, args, this.signal);
      const text = unwrapToolText(mcpResult);
      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text } }],
          rawOutput: mcpResult,
        },
      });
      return { content: text };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error calling MCP tool ${toolName}: ${message}` };
    }
  }

  // ---------------------------------------------------------------------------
  // Notification helpers
  // ---------------------------------------------------------------------------

  /**
   * Request permission from the user based on the current session mode.
   *
   * Returns a result indicating whether to allow, reject, or cancel the operation,
   * or whether a transport error occurred.
   */
  private async maybeRequestPermission(args: {
    toolCallId: string;
    kind: "write" | "execute";
    rawInput: unknown;
    title: string;
    locations?: Array<{ path: string }>;
  }): Promise<
    | { type: "allow" }
    | { type: "reject" }
    | { type: "cancelled" }
    | { type: "aborted" }
    | { type: "error"; message: string }
  > {
    const mode = this.getMode();

    if (this.signal?.aborted) {
      return { type: "aborted" };
    }

    // bypass_permissions: allow everything without prompting
    if (mode === "bypass_permissions") {
      return { type: "allow" };
    }

    // accept_edits: allow writes without prompting, still prompt for commands
    if (mode === "accept_edits" && args.kind === "write") {
      return { type: "allow" };
    }

    // default mode (or accept_edits with execute): prompt for permission
    try {
      const permissionPromise = this.connection.requestPermission({
        sessionId: this.sessionId,
        toolCall: {
          toolCallId: args.toolCallId,
          title: args.title,
          kind: args.kind === "write" ? "edit" : "execute",
          status: "pending",
          locations: args.locations ?? [],
          // Passed through verbatim: callers hand us the full payload so the
          // approval prompt shows exactly what will run. UI-card elision
          // happens on the sessionUpdate channel, never here.
          rawInput: args.rawInput,
        },
        options: [
          { kind: "allow_once", name: "Allow", optionId: "allow" },
          { kind: "reject_once", name: "Skip", optionId: "reject" },
        ],
      });

      if (!this.signal) {
        return this.permissionOutcome(await permissionPromise);
      }

      let abortHandler: (() => void) | undefined;
      const abortPromise = new Promise<"aborted">((resolve) => {
        abortHandler = () => resolve("aborted");
        this.signal!.addEventListener("abort", abortHandler, { once: true });
        if (this.signal!.aborted) abortHandler();
      });
      try {
        const outcome = await Promise.race([
          permissionPromise.then((response) => ({ kind: "response" as const, response })),
          abortPromise.then(() => ({ kind: "aborted" as const })),
        ]);
        if (outcome.kind === "aborted") {
          return { type: "aborted" };
        }
        return this.permissionOutcome(outcome.response);
      } finally {
        if (abortHandler) this.signal.removeEventListener("abort", abortHandler);
      }
    } catch (err) {
      // Transport failure: return error so caller can handle appropriately
      const message = err instanceof Error ? err.message : String(err);
      return { type: "error", message };
    }
  }

  private permissionOutcome(permissionResponse: {
    outcome: { outcome: string; optionId?: string };
  }): { type: "allow" } | { type: "reject" } | { type: "cancelled" } {
    if (permissionResponse.outcome.outcome === "cancelled") {
      return { type: "cancelled" };
    }
    if (
      permissionResponse.outcome.outcome === "selected" &&
      permissionResponse.outcome.optionId === "reject"
    ) {
      return { type: "reject" };
    }
    return { type: "allow" };
  }

  /** Mark an in-progress tool call as failed. */
  private async markFailed(toolCallId: string, message: string): Promise<void> {
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call_update",
        toolCallId,
        status: "failed",
        rawOutput: { error: message },
      },
    });
  }

  /**
   * Emit a brand new failed tool_call (for situations where we never made it
   * to in_progress, e.g. invalid arguments / missing capabilities).
   */
  private async failedToolCall(
    toolCallId: string,
    toolName: string,
    rawInput: Record<string, unknown>,
    message: string
  ): Promise<void> {
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: toolName,
        kind: "other",
        status: "failed",
        locations: [],
        rawInput,
        rawOutput: { error: message },
      },
    });
  }

  private async failAndReturn(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>,
    message: string
  ): Promise<ToolResult> {
    await this.failedToolCall(toolCallId, toolName, args, message);
    return { content: message };
  }
}

/** Resolve the API key from env or stored credentials, throwing a clear error if missing. */
function requireResolvedApiKey(): string {
  const apiKey = resolveApiKey();
  if (!apiKey) {
    throw new Error(
      "No API key found. Set Z_AI_API_KEY, or run `glm-acp-agent --setup` to store one."
    );
  }
  return apiKey;
}

function formatSearchOutput(mcpResult: unknown): { output: string; resultCount: number } {
  const payload = unwrapMcpPayload(mcpResult);
  const results = isRecord(payload) && Array.isArray(payload["search_result"])
    ? payload["search_result"]
    : [];

  if (results.length === 0) {
    return {
      output: typeof payload === "string" && payload.length > 0 ? payload : "No results found.",
      resultCount: 0,
    };
  }

  const output = results
    .map((raw, i) => {
      const r = isRecord(raw) ? raw : {};
      const lines = [`[${i + 1}] ${stringValue(r["title"]) ?? "(no title)"}`];
      const link = stringValue(r["link"]);
      const media = stringValue(r["media"]);
      const publishDate = stringValue(r["publish_date"]);
      const content = stringValue(r["content"]);
      if (link) lines.push(`URL: ${link}`);
      if (media) lines.push(`Source: ${media}`);
      if (publishDate) lines.push(`Date: ${publishDate}`);
      if (content) lines.push(`Summary: ${content}`);
      return lines.join("\n");
    })
    .join("\n\n");

  return { output, resultCount: results.length };
}

function formatReaderOutput(mcpResult: unknown): {
  output: string;
  title?: string;
  resultUrl?: string;
} {
  const payload = unwrapMcpPayload(mcpResult);
  const result = isRecord(payload) && isRecord(payload["reader_result"])
    ? payload["reader_result"]
    : undefined;

  if (!result) {
    return {
      output: typeof payload === "string" && payload.length > 0 ? payload : "No content returned.",
    };
  }

  const title = stringValue(result["title"]);
  const resultUrl = stringValue(result["url"]);
  const description = stringValue(result["description"]);
  const content = stringValue(result["content"]);
  const lines: string[] = [];
  if (title) lines.push(`# ${title}`);
  if (resultUrl) lines.push(`URL: ${resultUrl}`);
  if (description) lines.push(`\n${description}`);
  if (content) lines.push(`\n${content}`);

  return { output: lines.join("\n") || "No content returned.", title, resultUrl };
}

function unwrapMcpPayload(mcpResult: unknown): unknown {
  if (!isRecord(mcpResult)) return mcpResult;
  const content = mcpResult["content"];
  if (!Array.isArray(content)) return mcpResult;

  const texts = content
    .map((entry) => {
      if (!isRecord(entry)) return undefined;
      const text = entry["text"];
      return typeof text === "string" ? text : undefined;
    })
    .filter((text): text is string => typeof text === "string");

  if (texts.length === 0) return mcpResult;
  if (texts.length === 1) return parseJsonIfPossible(texts[0]);
  return texts.join("\n");
}

function parseJsonIfPossible(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

async function mapWithConcurrency<T, R>(items: readonly T[], concurrency: number, mapper: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (true) {
      const index = next++;
      if (index >= items.length) return;
      results[index] = await mapper(items[index]!);
    }
  });
  await Promise.all(workers);
  return results;
}

function runShellCommand(
  command: string,
  cwd: string,
  signal?: AbortSignal,
  limits: CommandLimits = readCommandLimits()
): Promise<ShellCommandResult> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("The operation was aborted"));
      return;
    }
    const child = spawn("sh", ["-c", command], {
      cwd,
      // Run the shell in its own process group so that background processes
      // (nohup, disown, &) survive after the main sh -c exits and don't
      // receive signals aimed at this agent. The child must stay ref'd: while
      // the tool call is in flight it is real pending work, and an unref'd
      // child let the event loop drain mid-await whenever nothing else was
      // pending (#82). Combined with the post-"exit" stream destroy below,
      // daemons that inherit the pipes still can't keep this process alive
      // after the shell exits.
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let capturedBytes = 0;
    let outputTruncated = false;
    let settled = false;
    let abortRequested = false;
    let timedOut = false;
    let forceKillTimer: NodeJS.Timeout | undefined;
    let streamDestroyTimer: NodeJS.Timeout | undefined;

    const capture = (target: Buffer[], chunk: Buffer | Uint8Array) => {
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const remaining = limits.outputLimitBytes - capturedBytes;
      if (remaining <= 0) {
        if (buffer.length > 0) outputTruncated = true;
        return;
      }
      const bytes = Math.min(remaining, buffer.length);
      if (bytes > 0) {
        // Copy the retained prefix. A subarray would keep the entire incoming
        // chunk alive, allowing one noisy write to bypass the memory bound.
        target.push(Buffer.from(buffer.subarray(0, bytes)));
        capturedBytes += bytes;
      }
      if (bytes < buffer.length) outputTruncated = true;
    };

    const terminateAndEscalate = () => {
      terminateProcessTree(child);
      // A process can ignore SIGTERM. Escalate after a short grace period so
      // an aborted or timed-out tool cannot keep the prompt turn alive. The
      // guard keeps the SIGKILL from ever landing on a process group the OS
      // has recycled for unrelated work.
      if (!forceKillTimer) {
        forceKillTimer = setTimeout(() => {
          forceKillTimer = undefined;
          if (!isProcessGroupAlive(child.pid)) return;
          terminateProcessTree(child, true);
        }, 250);
      }
    };

    const onAbort = () => {
      if (settled) return;
      abortRequested = true;
      terminateAndEscalate();
    };
    const onTimeout = () => {
      if (settled) return;
      timedOut = true;
      terminateAndEscalate();
    };
    const cleanup = () => {
      if (timeoutTimer) clearTimeout(timeoutTimer);
      if (streamDestroyTimer) clearTimeout(streamDestroyTimer);
      if (forceKillTimer) {
        // Keep escalation armed only while the detached group may still hold
        // SIGTERM-resistant descendants; once the group is gone the delayed
        // SIGKILL must never fire (stale process-group ID).
        if ((!abortRequested && !timedOut) || !isProcessGroupAlive(child.pid)) {
          clearTimeout(forceKillTimer);
        }
      }
      signal?.removeEventListener("abort", onAbort);
    };

    const timeoutTimer = setTimeout(onTimeout, limits.timeoutMs);
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();

    child.stdout.on("data", (chunk: Buffer) => capture(stdout, chunk));
    child.stderr.on("data", (chunk: Buffer) => capture(stderr, chunk));
    child.on("error", (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err);
    });
    child.on("exit", () => {
      // The shell is the command's foreground process. Once it exits normally,
      // only inherited pipes from intentionally backgrounded work may remain;
      // do not let the deadline kill that work during the short drain grace.
      clearTimeout(timeoutTimer);
      // The shell exited. Normal commands will close their streams immediately,
      // firing "close" within milliseconds. For daemons that inherit stdio and
      // keep pipes open, forcefully destroy the streams after a brief grace
      // period so "close" fires and the Promise can resolve.
      streamDestroyTimer = setTimeout(() => {
        if (!settled) {
          child.stdout.destroy();
          child.stderr.destroy();
        }
      }, 50);
    });
    child.on("close", (exitCode, closeSignal) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve({
        stdout: decodeCapturedOutput(stdout),
        stderr: decodeCapturedOutput(stderr),
        exitCode,
        signal: closeSignal,
        outputTruncated,
        timedOut,
      });
    });
  });
}

export function isProcessGroupAlive(pid: number | undefined): boolean {
  if (!pid) return false;
  if (process.platform === "win32") {
    // No POSIX process groups here: keep the previous behavior of leaving the
    // escalation timer armed (taskkill on a dead pid fails harmlessly).
    return true;
  }
  try {
    process.kill(-pid, 0);
    return true;
  } catch (err) {
    // ESRCH: the process group no longer exists. EPERM: it exists but is owned
    // by another user — still alive, so leave escalation armed.
    return (err as NodeJS.ErrnoException).code !== "ESRCH";
  }
}

interface ShellCommandResult {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  outputTruncated: boolean;
  timedOut: boolean;
}

function terminateProcessTree(child: ReturnType<typeof spawn>, force = false): void {
  if (!child.pid) return;
  const signal: NodeJS.Signals = force ? "SIGKILL" : "SIGTERM";
  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
      stdio: "ignore",
      windowsHide: true,
    });
    killer.on("error", () => {
      try {
        child.kill(signal);
      } catch {
        /* already exited */
      }
    });
    killer.unref();
    return;
  }

  try {
    // detached=true makes the shell the process-group leader. A negative PID
    // targets the whole group, including foreground descendants.
    process.kill(-child.pid, signal);
  } catch {
    // The shell may have exited between the abort event and this call. Fall
    // back to the direct child so the cancellation still settles promptly.
    try {
      child.kill(signal);
    } catch {
      /* already exited */
    }
  }
}

function formatCommandOutput(result: {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  outputTruncated?: boolean;
  timedOut?: boolean;
}, limits?: CommandLimits): string {
  const lines = [`Exit code: ${result.exitCode ?? "unknown"}`];
  if (result.signal) lines.push(`Signal: ${result.signal}`);
  if (result.timedOut && limits) lines.push(`Command timed out after ${limits.timeoutMs} ms.`);
  if (result.outputTruncated && limits) {
    lines.push(`Output truncated: command output exceeded ${limits.outputLimitBytes} bytes.`);
  }
  lines.push("", "STDOUT:", result.stdout.length > 0 ? result.stdout : "(empty)");
  lines.push("", "STDERR:", result.stderr.length > 0 ? result.stderr : "(empty)");
  return lines.join("\n");
}

/**
 * Decode captured bytes without allowing malformed or partial UTF-8 to turn
 * into a replacement character that is larger than the bytes we retained.
 * Invalid bytes are discarded; valid UTF-8 sequences are copied unchanged.
 */
function decodeCapturedOutput(chunks: Buffer[]): string {
  const bytes = Buffer.concat(chunks);
  const valid: number[] = [];
  let index = 0;
  while (index < bytes.length) {
    const first = bytes[index]!;
    let length = 0;
    if (first <= 0x7f) {
      length = 1;
    } else if (first >= 0xc2 && first <= 0xdf) {
      length = validUtf8Continuation(bytes, index, 2) ? 2 : 0;
    } else if (first === 0xe0) {
      length = validUtf8Continuation(bytes, index, 3, 0xa0) ? 3 : 0;
    } else if (first >= 0xe1 && first <= 0xec) {
      length = validUtf8Continuation(bytes, index, 3) ? 3 : 0;
    } else if (first === 0xed) {
      length = validUtf8Continuation(bytes, index, 3, undefined, 0x9f) ? 3 : 0;
    } else if (first >= 0xee && first <= 0xef) {
      length = validUtf8Continuation(bytes, index, 3) ? 3 : 0;
    } else if (first === 0xf0) {
      length = validUtf8Continuation(bytes, index, 4, 0x90) ? 4 : 0;
    } else if (first >= 0xf1 && first <= 0xf3) {
      length = validUtf8Continuation(bytes, index, 4) ? 4 : 0;
    } else if (first === 0xf4) {
      length = validUtf8Continuation(bytes, index, 4, undefined, 0x8f) ? 4 : 0;
    }

    if (length > 0) {
      for (let offset = 0; offset < length; offset++) valid.push(bytes[index + offset]!);
      index += length;
    } else {
      index++;
    }
  }
  return Buffer.from(valid).toString("utf8");
}

function validUtf8Continuation(
  bytes: Buffer,
  start: number,
  length: number,
  minimumSecond?: number,
  maximumSecond?: number
): boolean {
  if (start + length > bytes.length) return false;
  const second = bytes[start + 1]!;
  if (minimumSecond !== undefined && second < minimumSecond) return false;
  if (maximumSecond !== undefined && second > maximumSecond) return false;
  if (second < 0x80 || second > 0xbf) return false;
  for (let offset = 2; offset < length; offset++) {
    const value = bytes[start + offset]!;
    if (value < 0x80 || value > 0xbf) return false;
  }
  return true;
}

function unwrapVisionText(mcpResult: unknown): string {
  if (!isRecord(mcpResult)) return typeof mcpResult === "string" ? mcpResult : "";
  const content = mcpResult["content"];
  if (Array.isArray(content)) {
    const texts = content
      .map((entry) => (isRecord(entry) && typeof entry["text"] === "string" ? (entry["text"] as string) : ""))
      .filter((s) => s.length > 0);
    if (texts.length > 0) return texts.join("\n");
  }
  return JSON.stringify(mcpResult);
}

function unwrapToolText(mcpResult: unknown): string {
  if (typeof mcpResult === "string") return mcpResult;
  if (isRecord(mcpResult)) {
    const content = mcpResult["content"];
    if (Array.isArray(content)) {
      const texts = content
        .map((entry) => (isRecord(entry) && typeof entry["text"] === "string" ? (entry["text"] as string) : ""))
        .filter((s) => s.length > 0);
      if (texts.length > 0) return texts.join("\n");
    }
  }
  return JSON.stringify(mcpResult);
}
