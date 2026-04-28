/**
 * Tool JSON schemas exposed to the GLM model.
 *
 * These definitions follow the OpenAI function-calling format and map
 * directly to ACP Client capabilities. The agent advertises the same set
 * of tools on every call; whether a tool actually works at runtime depends
 * on the client capabilities advertised at `initialize` time.
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
        "Read the text content of a file from the client's file system. Requires the `fs.readTextFile` client capability.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file to read.",
          },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description:
        "Write or overwrite a text file in the client's file system. The user is asked for permission before any write occurs. Requires the `fs.writeTextFile` client capability.",
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
        "List the files and directories at the given path by running `ls -la` through a shell on the client machine. Requires the `terminal` client capability and a POSIX-compatible shell.",
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
        "Execute a shell command via `sh -c` on the client machine and return its output. The user is asked for permission before each invocation. Requires the `terminal` client capability.",
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
];
