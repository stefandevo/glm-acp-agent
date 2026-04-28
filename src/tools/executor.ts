import type {
  AgentSideConnection,
  ClientCapabilities,
} from "@agentclientprotocol/sdk";

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
    private signal?: AbortSignal
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

    const path = String(args["path"] ?? "");
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

    const path = String(args["path"] ?? "");
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

    // Step 2: request user permission.
    const permissionResponse = await this.connection.requestPermission({
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

    // Step 2: request permission.
    const permissionResponse = await this.connection.requestPermission({
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
      const apiKey = requireApiKey();
      const body: Record<string, unknown> = {
        search_engine: "search-prime",
        search_query: query,
      };
      if (count !== undefined) body["count"] = count;

      const response = await fetch("https://api.z.ai/api/paas/v4/web_search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: this.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as {
        search_result?: Array<{
          title?: string;
          link?: string;
          content?: string;
          media?: string;
          publish_date?: string;
        }>;
      };

      const results = data.search_result ?? [];
      const formatted = results
        .map((r, i) => {
          const lines = [`[${i + 1}] ${r.title ?? "(no title)"}`];
          if (r.link) lines.push(`URL: ${r.link}`);
          if (r.media) lines.push(`Source: ${r.media}`);
          if (r.publish_date) lines.push(`Date: ${r.publish_date}`);
          if (r.content) lines.push(`Summary: ${r.content}`);
          return lines.join("\n");
        })
        .join("\n\n");

      const output = formatted || "No results found.";

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: { resultCount: results.length },
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
      const apiKey = requireApiKey();

      const response = await fetch("https://api.z.ai/api/paas/v4/reader", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({ url, return_format: returnFormat }),
        signal: this.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = (await response.json()) as {
        reader_result?: {
          title?: string;
          description?: string;
          url?: string;
          content?: string;
        };
      };

      const result = data.reader_result;
      const lines: string[] = [];
      if (result?.title) lines.push(`# ${result.title}`);
      if (result?.url) lines.push(`URL: ${result.url}`);
      if (result?.description) lines.push(`\n${result.description}`);
      if (result?.content) lines.push(`\n${result.content}`);
      const output = lines.join("\n") || "No content returned.";

      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "completed",
          content: [{ type: "content", content: { type: "text", text: output } }],
          rawOutput: { title: result?.title, url: result?.url },
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.markFailed(toolCallId, message);
      return { content: `Error reading URL: ${message}` };
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

/** Read the API key from env, throwing a clear error if it's missing. */
function requireApiKey(): string {
  const apiKey = process.env["Z_AI_API_KEY"];
  if (!apiKey) {
    throw new Error("Z_AI_API_KEY environment variable is required but not set.");
  }
  return apiKey;
}

/**
 * Quote a string for safe inclusion in a `sh -c` command line.
 * We use single quotes and escape any embedded single quotes via `'\''`.
 */
function shellQuote(s: string): string {
  return `'${s.replace(/'/g, "'\\''")}'`;
}
