/**
 * Tool JSON schemas exposed to the GLM model.
 *
 * These definitions follow the OpenAI function-calling format and map
 * directly to ToolExecutor implementations. Local file and shell tools run in
 * the agent process; writes and command execution still ask the ACP client for
 * permission before doing anything.
 */

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description:
        "Read text from a file or editor buffer. Relative paths resolve against the ACP session working directory. Use offset (1-based) and limit (default 2000, maximum 5000) to page. Local scans are byte-bounded, so total lines can be unknown and an incomplete line must be narrowed or inspected with a bounded command.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file to read.",
          },
          offset: {
            type: "number",
            description: "1-based line number to start reading from. Default 1.",
          },
          limit: {
            type: "number",
            description: "Maximum number of lines to return. Default 2000.",
          },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description:
        "Replace one exact snippet inside an existing text file. old_text must match exactly once, including whitespace. The complete local file or editor buffer must fit the configured read/edit budget; use write_file for an intentional full rewrite.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file to edit.",
          },
          old_text: {
            type: "string",
            description: "The exact existing text to replace. Must appear exactly once in the file.",
          },
          new_text: {
            type: "string",
            description: "The replacement text. Use an empty string to delete the snippet.",
          },
        },
        required: ["path", "old_text", "new_text"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description:
        "Write or overwrite a text file from the agent process after asking the user for permission. Relative paths resolve against the ACP session working directory. Use this for new files or full rewrites; prefer edit_file for small changes to existing files.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file to write.",
          },
          content: {
            type: "string",
            description: "The full text content to write to the file.",
          },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_files",
      description:
        "List files and directories at the given path. Relative paths resolve against the ACP session working directory. Results are a bounded, sorted subset and state when listing limits are reached.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path of the directory to list.",
          },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_command",
      description:
        "Execute a shell command via `sh -c` in the ACP session working directory and return stdout, stderr, and exit code. The user is asked for permission before each invocation.",
      parameters: {
        type: "object",
        properties: {
          command: {
            type: "string",
            description:
              "The shell command line to execute (interpreted by `sh -c`, so quoting, pipes, and redirects all work).",
          },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description:
        "Search the web using Z.AI's premium search engine and return relevant results, including titles, URLs, sources, dates, and content summaries.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query.",
          },
          count: {
            type: "integer",
            description: "Number of results to return (1–50). Default is 10.",
            minimum: 1,
            maximum: 50,
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_reader",
      description:
        "Fetch and parse the content of a web page at the given URL via Z.AI's reader, returning the main text content as markdown or plain text.",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "The URL of the page to read.",
          },
          return_format: {
            type: "string",
            description: "Return format: 'markdown' (default) or 'text'.",
            enum: ["markdown", "text"],
          },
        },
        required: ["url"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "image_analysis",
      description:
        "Analyze an image (local file path or remote URL) using Z.AI Coding Plan Vision MCP. Returns a textual description / answer. Use this to extract text from screenshots, describe diagrams, or answer questions about images the user has referenced.",
      parameters: {
        type: "object",
        properties: {
          image_source: {
            type: "string",
            description: "Local file path or remote URL of the image to analyze.",
          },
          prompt: {
            type: "string",
            description: "Optional question or instruction guiding the analysis. Defaults to a general description.",
          },
        },
        required: ["image_source"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "todowrite",
      description:
        "Create or replace the session's structured task list. Use it for multi-step work so progress is tracked in the task list instead of narrated in text; skip it for single-step, trivial, or purely conversational requests. Keep exactly one task list: each call replaces the previous one entirely. Mark a task in_progress before starting it and completed as soon as it is done — never batch status updates after the fact.",
      parameters: {
        type: "object",
        properties: {
          todos: {
            type: "array",
            description: "The full task list, replacing any previous one.",
            items: {
              type: "object",
              properties: {
                content: {
                  type: "string",
                  description: "Short imperative description of the task.",
                },
                status: {
                  type: "string",
                  enum: ["pending", "in_progress", "completed"],
                  description: "pending = not started, in_progress = currently working on it, completed = done.",
                },
                active_form: {
                  type: "string",
                  description: "Present-progressive form shown while the task runs, e.g. 'Renaming the entry point'.",
                },
              },
              required: ["content", "status"],
            },
          },
        },
        required: ["todos"],
      },
    },
  },
];
