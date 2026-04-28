# glm-acp-agent

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com) agent written in TypeScript that uses the **Z.AI / Zhipu AI GLM** model family (GLM-5.1, GLM-4.7, GLM-4.6, …) as its reasoning core.

The agent connects to any ACP-compatible IDE or client over **stdio**, streams responses back in real time, and can call a rich set of tools to interact with the user's file system, terminal, and the web.

---

## Features

- **Full ACP compliance** – implements `initialize`, `authenticate`, `session/new`, `session/set_mode`, `session/prompt`, `session/cancel`, `session/close`, and `session/list`
- **Streaming** – assistant text and reasoning tokens are forwarded as incremental ACP chunks
- **Tool calling** – agentic loop with up to 20 turns of GLM function calling
- **Thinking mode** – GLM's `reasoning_content` tokens are surfaced as `agent_thought_chunk` blocks so the client can show the model's chain of thought
- **Six built-in tools** (see below)
- **Capability-aware** – every tool is gated on the client capabilities advertised at `initialize` time; tools the client can't run return a clear error to the model instead of crashing
- **Permissioned writes / commands** – every `write_file` and `run_command` call asks the user via `session/request_permission` before doing anything
- **Protocol-correct stop reasons** – maps model and runtime conditions to ACP `end_turn`, `max_tokens`, `max_turn_requests`, `refusal`, and `cancelled`
- **Protocol-correct tool statuses** – `pending` → `in_progress` → `completed` / `failed`
- **Token usage reporting** – aggregated usage is returned on the `session/prompt` response

---

## Architecture

```text
ACP Client (IDE plugin, CLI, …)
        │  stdio (ndjson)
        ▼
  GlmAcpAgent          ← ACP protocol layer  (src/protocol/)
        │
        ├─ GlmClient   ← Z.AI / Zhipu AI Chat Completions  (src/llm/)
        │    └─ streams chat completions with tool schemas
        │
        └─ ToolExecutor ← executes tool calls  (src/tools/)
             ├─ read_file / write_file       → ACP client (fs.*)
             ├─ list_files / run_command     → ACP client (terminal)
             ├─ web_search                   → Z.AI /paas/v4/web_search
             └─ web_reader                   → Z.AI /paas/v4/reader
```

The agent process itself only needs network access to `api.z.ai`. All file-system and shell operations are **delegated to the ACP client** — the agent never touches the local disk directly. `web_search` and `web_reader` are different: they run **inside the agent process** and call the Z.AI Tools API directly.

---

## Available Tools

| Tool | Runs on | Requires client capability | Description |
|------|---------|----------------------------|-------------|
| `read_file` | ACP client | `fs.readTextFile` | Read the text content of a file |
| `write_file` | ACP client | `fs.writeTextFile` | Write or overwrite a text file (asks for permission) |
| `list_files` | ACP client | `terminal` | List a directory via `ls -la` (POSIX shell required) |
| `run_command` | ACP client | `terminal` | Run an arbitrary shell command via `sh -c` (asks for permission) |
| `web_search` | Agent (Z.AI) | – | Search the web — returns titles, URLs, and summaries |
| `web_reader` | Agent (Z.AI) | – | Fetch and parse a web page (markdown or plain text) |

---

## Prerequisites

- **Node.js** 20 or later (native `fetch` and Web Streams required)
- **npm** 9 or later
- A **Z.AI API key** — obtain one at <https://z.ai/manage-apikey/apikey-list>

---

## Installation

```bash
git clone https://github.com/stefandevo/glm-acp-agent.git
cd glm-acp-agent
npm install
npm run build
```

---

## Configuration

The agent is configured entirely through environment variables.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z_AI_API_KEY` | **Yes** | — | API key for the Z.AI / Zhipu AI service |
| `ACP_GLM_MODEL` | No | `glm-5.1` | Override the GLM model |
| `ACP_GLM_BASE_URL` | No | `https://api.z.ai/api/paas/v4` | Override the API base URL |
| `ACP_GLM_MAX_TOKENS` | No | `8192` | Cap on `max_tokens` for each completion |
| `ACP_GLM_THINKING` | No | auto-detected | Force thinking mode `true` / `false` |

### Supported models

Any model exposed by the Z.AI Chat Completions API can be used. Recommended choices:

| Model | Notes |
|-------|-------|
| `glm-5.1` | **Default.** Long-horizon coding model; thinking mode auto-enabled |
| `glm-4.7` | Strong reasoning, 200K context |
| `glm-4.6` | Faster, slightly cheaper |
| `glm-4.5` | Cost-efficient general-purpose |
| `glm-4-long` | Extended-context variant |

When the model name matches `glm-4.5`, `glm-4.6`, `glm-4.7`, or `glm-5.x`, the agent enables Z.AI's `thinking: { type: "enabled" }` extension and forwards reasoning tokens to the client as `agent_thought_chunk` blocks. Override with `ACP_GLM_THINKING=false` if you want plain completions only.

---

## Running

### Standalone (stdio)

```bash
export Z_AI_API_KEY=your_key_here
node dist/index.js
```

The agent speaks the ACP newline-delimited JSON protocol over stdin/stdout. You can connect any ACP-compatible client to it.

### As a global CLI

```bash
npm install -g .
export Z_AI_API_KEY=your_key_here
glm-acp-agent
```

### Development mode (watch)

```bash
export Z_AI_API_KEY=your_key_here
npm run dev        # tsc --watch
```

---

## Connecting to an ACP Client

### Zed

`~/.config/zed/settings.json`:

```json
{
  "agent_servers": {
    "glm": {
      "command": "node",
      "args": ["/absolute/path/to/glm-acp-agent/dist/index.js"],
      "env": { "Z_AI_API_KEY": "sk-…" }
    }
  }
}
```

### Neovim / VS Code / JetBrains / any ACP client

Any client that supports configuring an ACP agent via a `command` + `args` invocation works the same way: point it at `node /absolute/path/to/glm-acp-agent/dist/index.js` and supply `Z_AI_API_KEY` in the environment.

### Authentication

The agent advertises a single `env_var` authentication method. ACP clients that support the auth-methods proposal will prompt the user for `Z_AI_API_KEY` automatically; clients that don't yet handle auth methods should set the variable themselves before launching the agent.

---

## Project Structure

```text
src/
├── index.ts                  # Entry point – starts stdio connection
├── llm/
│   └── glm-client.ts         # OpenAI-compatible client for Z.AI / Zhipu AI
├── protocol/
│   ├── connection.ts         # Sets up the ACP stdio connection
│   └── agent.ts              # GlmAcpAgent – ACP protocol implementation
├── tools/
│   ├── definitions.ts        # Tool JSON schemas (function-calling format)
│   └── executor.ts           # ToolExecutor – dispatches tool calls
└── tests/
    ├── agent.test.ts         # Protocol-level tests for GlmAcpAgent
    ├── executor.test.ts      # Tests for ToolExecutor
    ├── glm-client.test.ts    # Tests for streaming / tool-call assembly
    └── integration.test.ts   # End-to-end tests over the real ACP ndjson transport
```

---

## Building & Testing

```bash
npm run build   # one-shot TypeScript compilation → dist/
npm run dev     # watch mode
npm test        # build + run unit tests with the node:test runner
```

The test suite covers:

- ACP protocol-version negotiation, capability advertising, and auth method shape
- Session lifecycle (`new` / `list` / `close`, filter by `cwd`)
- Prompt loop streaming, tool-call assembly, max-turn cap, cancellation, stop-reason mapping (`stop` → `end_turn`, `length` → `max_tokens`, `content_filter` → `refusal`, `tool_calls` exhausted → `max_turn_requests`)
- Content-block conversion (`text`, `resource_link`, embedded `resource`)
- Token usage reporting on the `PromptResponse`
- Tool call lifecycle (`pending` → `in_progress` → `completed` / `failed`)
- Capability gating (tools refuse cleanly when the client did not advertise the required capability)
- Permission flows for `write_file` and `run_command` (allow / reject / cancel)
- Shell-quoted argument handling for `list_files` and `run_command`
- GLM streaming: text deltas, `reasoning_content` deltas, multi-chunk tool-call assembly, and trailing usage

---

## Troubleshooting

- **`Z_AI_API_KEY environment variable is required but not set.`** — set the env var or configure your ACP client to forward it.
- **`HTTP 401: Invalid API key`** — your key is wrong or expired; rotate it on <https://z.ai/manage-apikey/apikey-list>.
- **The agent says "client does not advertise the … capability".** — your ACP client doesn't expose that capability (e.g. terminal). Ask the model to use a different tool, or upgrade the client.
- **Tools never get to run.** — make sure the client is sending the `clientCapabilities` field in `initialize`; the agent uses it to decide which tools to expose to the model.

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
