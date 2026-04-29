import type {
  AgentSideConnection,
  ClientCapabilities,
} from "@agentclientprotocol/sdk";
import { resolveApiKey } from "../llm/credentials.js";
import {
  callZaiMcpTool,
  ZAI_WEB_READER_MCP_ENDPOINT,
  ZAI_WEB_SEARCH_MCP_ENDPOINT,
} from "./zai-mcp-client.js";
import type { VisionMcpClient } from "./vision-mcp-client.js";

/**
 * Result returned after executing a tool call against the ACP client.
 */
export interface ToolResult {
  content: string;
}

/**
 * Executes GLM tool calls by delegating to the corresponding ACP Client methods.
 *
 * Permission is requested from the user before any write or execute operation.
 * Tools that require capabilities the client did not advertise return a clear
 * error string instead of throwing, so the model can recover by trying a
 * different approach.
 */
export class ToolExecutor {
  constructor(
    private connection: AgentSideConnection,
    private sessionId: string,
    private clientCapabilities: ClientCapabilities | null = null,
    private signal?: AbortSignal,
    private visionClient: VisionMcpClient | null = null
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
      return { content: message };
    }

    switch (toolName) {
      case "read_file":
        return this.readFile(toolCallId, args);
      case "write_file":
        return this.writeFile(toolCallId, args);
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
      default: {
        const message = `Error: unknown tool "${toolName}"`;
        await this.failedToolCall(toolCallId, toolName, args, message);
        return { content: message };
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Capability helpers
  // ---------------------------------------------------------------------------

  private requireCap(
    toolCallId: string,
    toolName: string,
    args: Record<string, unknown>,
    available: boolean,
    capName: string
  ): Promise<ToolResult> | null {
    if (available) return null;
    const message = `Error: client does not advertise the ${capName} capability; this tool is unavailable.`;
    return (async () => {
      await this.failedToolCall(toolCallId, toolName, args, message);
      return { content: message };
    })();
  }

  // ---------------------------------------------------------------------------
  // Private tool implementations
  // ---------------------------------------------------------------------------

  private async readFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const cap = this.requireCap(
      toolCallId,
      "read_file",
      args,
      Boolean(this.clientCapabilities?.fs?.readTextFile),
      "fs.readTextFile"
    );
    if (cap) return cap;

    const path = String(args["path"] ?? "").trim();
    if (!path) {
      return this.failAndReturn(toolCallId, "read_file", args, "Error: `path` is required.");
    }

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Read file: ${path}`,
        kind: "read",
        status: "in_progress",
        locations: [{ path }],
        rawInput: args,
      },
    });

    try {
      const response = await this.connection.readTextFile({
        sessionId: this.sessionId,
        path,
      });

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: response.content } }],
          rawOutput: { content: response.content },
        },
      });

      return { content: response.content };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error reading file: ${message}` };
    }
  }

  private async writeFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const cap = this.requireCap(
      toolCallId,
      "write_file",
      args,
      Boolean(this.clientCapabilities?.fs?.writeTextFile),
      "fs.writeTextFile"
    );
    if (cap) return cap;

    const path = String(args["path"] ?? "").trim();
    const content = String(args["content"] ?? "");
    if (!path) {
      return this.failAndReturn(toolCallId, "write_file", args, "Error: `path` is required.");
    }

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
        rawInput: args,
      },
    });

    // Step 2: request user permission. Treat transport failures here as a
    // failed tool call instead of letting them escape and abort the loop.
    let permissionResponse;
    try {
      permissionResponse = await this.connection.requestPermission({
        sessionId: this.sessionId,
        toolCall: {
          toolCallId,
          title: `Write file: ${path}`,
          kind: "edit",
          status: "pending",
          locations: [{ path }],
          rawInput: args,
        },
        options: [
          { kind: "allow_once", name: "Allow write", optionId: "allow" },
          { kind: "reject_once", name: "Skip write", optionId: "reject" },
        ],
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error requesting permission: ${message}` };
    }

    if (permissionResponse.outcome.outcome === "cancelled") {
      await this.markFailed(toolCallId, "Cancelled by user.");
      return { content: "Write cancelled by user." };
    }
    if (
      permissionResponse.outcome.outcome === "selected" &&
      permissionResponse.outcome.optionId === "reject"
    ) {
      await this.markFailed(toolCallId, "Rejected by user.");
      return { content: "Write rejected by user." };
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
      await this.connection.writeTextFile({
        sessionId: this.sessionId,
        path,
        content,
      });

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

  private async listFiles(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const cap = this.requireCap(
      toolCallId,
      "list_files",
      args,
      Boolean(this.clientCapabilities?.terminal),
      "terminal"
    );
    if (cap) return cap;

    const rawPath = args["path"];
    if (typeof rawPath !== "string" || rawPath.trim().length === 0) {
      return this.failAndReturn(
        toolCallId,
        "list_files",
        args,
        "Error listing files: `path` must be a non-empty string."
      );
    }
    const path = rawPath;

    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `List files: ${path}`,
        kind: "read",
        status: "in_progress",
        locations: [{ path }],
        rawInput: args,
      },
    });

    let terminal;
    try {
      terminal = await this.connection.createTerminal({
        sessionId: this.sessionId,
        // Run via a shell so quoting / wildcards behave like a real ls call.
        command: "sh",
        args: ["-c", `ls -la -- ${shellQuote(path)}`],
      });

      await terminal.waitForExit();
      const outputResponse = await terminal.currentOutput();
      const output = outputResponse.output;

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "terminal", terminalId: terminal.id }],
          rawOutput: { output },
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error listing files: ${message}` };
    } finally {
      try {
        await terminal?.release();
      } catch {
        // ignore release errors
      }
    }
  }

  private async runCommand(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const cap = this.requireCap(
      toolCallId,
      "run_command",
      args,
      Boolean(this.clientCapabilities?.terminal),
      "terminal"
    );
    if (cap) return cap;

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
        rawInput: args,
      },
    });

    // Step 2: request permission. Transport failures here become a failed
    // tool call so the agent loop can continue.
    let permissionResponse;
    try {
      permissionResponse = await this.connection.requestPermission({
        sessionId: this.sessionId,
        toolCall: {
          toolCallId,
          title: `Run command: ${command}`,
          kind: "execute",
          status: "pending",
          locations: [],
          rawInput: args,
        },
        options: [
          { kind: "allow_once", name: "Run command", optionId: "allow" },
          { kind: "reject_once", name: "Skip command", optionId: "reject" },
        ],
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error requesting permission: ${message}` };
    }

    if (permissionResponse.outcome.outcome === "cancelled") {
      await this.markFailed(toolCallId, "Cancelled by user.");
      return { content: "Command cancelled by user." };
    }
    if (
      permissionResponse.outcome.outcome === "selected" &&
      permissionResponse.outcome.optionId === "reject"
    ) {
      await this.markFailed(toolCallId, "Rejected by user.");
      return { content: "Command rejected by user." };
    }

    return this.runTerminalCommand(toolCallId, command);
  }

  private async runTerminalCommand(
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

    let terminal;
    try {
      // Run via `sh -c` so the command string is interpreted by a real shell –
      // this preserves quoting, pipes, redirects and environment expansion.
      terminal = await this.connection.createTerminal({
        sessionId: this.sessionId,
        command: "sh",
        args: ["-c", command],
      });

      await terminal.waitForExit();
      const outputResponse = await terminal.currentOutput();
      const output = outputResponse.output;

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "terminal", terminalId: terminal.id }],
          rawOutput: { output },
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error running command: ${message}` };
    } finally {
      try {
        await terminal?.release();
      } catch {
        // ignore release errors
      }
    }
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
        rawInput: args,
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
          rawOutput: { resultCount },
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
        rawInput: args,
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
          rawOutput: { title, url: resultUrl },
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
        rawInput: args,
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
          rawOutput: { text },
        },
      });
      return { content: text };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error analyzing image: ${message}` };
    }
  }

  // ---------------------------------------------------------------------------
  // Notification helpers
  // ---------------------------------------------------------------------------

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

/**
 * Quote a string for safe inclusion in a `sh -c` command line.
 * We use single quotes and escape any embedded single quotes via `'\''`.
 */
function shellQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
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
