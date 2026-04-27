import type { AgentSideConnection } from "@agentclientprotocol/sdk";

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
 */
export class ToolExecutor {
  constructor(
    private connection: AgentSideConnection,
    private sessionId: string
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
      args = JSON.parse(rawArguments) as Record<string, unknown>;
    } catch {
      return { content: `Error: could not parse tool arguments: ${rawArguments}` };
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
      default:
        return { content: `Error: unknown tool "${toolName}"` };
    }
  }

  // ---------------------------------------------------------------------------
  // Private tool implementations
  // ---------------------------------------------------------------------------

  private async readFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? "");

    // Notify the client about the pending read tool call
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title: `Read file: ${path}`,
        kind: "read",
        status: "pending",
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
          rawOutput: { content: response.content },
        },
      });

      return { content: response.content };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "error",
          rawOutput: { error: message },
        },
      });
      return { content: `Error reading file: ${message}` };
    }
  }

  private async writeFile(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? "");
    const content = String(args["content"] ?? "");

    // Request permission before writing
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
        {
          kind: "allow_once",
          name: "Allow write",
          optionId: "allow",
        },
        {
          kind: "reject_once",
          name: "Skip write",
          optionId: "reject",
        },
      ],
    });

    if (permissionResponse.outcome.outcome === "cancelled") {
      return { content: "Write cancelled by user." };
    }
    if (
      permissionResponse.outcome.outcome === "selected" &&
      permissionResponse.outcome.optionId === "reject"
    ) {
      return { content: "Write rejected by user." };
    }

    // Notify pending state
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
      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "error",
          rawOutput: { error: message },
        },
      });
      return { content: `Error writing file: ${message}` };
    }
  }

  private async listFiles(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const path = String(args["path"] ?? ".");

    // list_files is implemented as a terminal command to `ls -la`
    return this.runTerminalCommand(
      toolCallId,
      `ls -la ${JSON.stringify(path)}`,
      `List files: ${path}`,
      args
    );
  }

  private async runCommand(
    toolCallId: string,
    args: Record<string, unknown>
  ): Promise<ToolResult> {
    const command = String(args["command"] ?? "");

    // Request permission before executing arbitrary commands
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
        {
          kind: "allow_once",
          name: "Run command",
          optionId: "allow",
        },
        {
          kind: "reject_once",
          name: "Skip command",
          optionId: "reject",
        },
      ],
    });

    if (permissionResponse.outcome.outcome === "cancelled") {
      return { content: "Command cancelled by user." };
    }
    if (
      permissionResponse.outcome.outcome === "selected" &&
      permissionResponse.outcome.optionId === "reject"
    ) {
      return { content: "Command rejected by user." };
    }

    return this.runTerminalCommand(toolCallId, command, `Run: ${command}`, args);
  }

  private async runTerminalCommand(
    toolCallId: string,
    command: string,
    title: string,
    rawInput: Record<string, unknown>
  ): Promise<ToolResult> {
    await this.connection.sessionUpdate({
      sessionId: this.sessionId,
      update: {
        sessionUpdate: "tool_call",
        toolCallId,
        title,
        kind: "execute",
        status: "pending",
        locations: [],
        rawInput,
      },
    });

    let terminal;
    try {
      terminal = await this.connection.createTerminal({
        sessionId: this.sessionId,
        command: "/bin/sh",
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
          content: [
            {
              type: "terminal",
              terminalId: terminal.id,
            },
          ],
          rawOutput: { output },
        },
      });

      return { content: output };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await this.connection.sessionUpdate({
        sessionId: this.sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId,
          status: "error",
          rawOutput: { error: message },
        },
      });
      return { content: `Error running command: ${message}` };
    } finally {
      await terminal?.release();
    }
  }
}
