# glm-acp-agent

An [Agent Client Protocol (ACP)](https://agentclientprotocol.org) agent written in TypeScript that uses the **Zhipu AI GLM** model family (GLM-5.1, GLM-4.7, and others) as its reasoning core.

The agent connects to any ACP-compatible IDE or client over **stdio**, streams responses back in real time, and can call a rich set of tools to interact with the user's file system, terminal, and the web.

---

## Features

- **Full ACP compliance** – implements `initialize`, `authenticate`, `newSession`, `setSessionMode`, `prompt`, `cancel`, `closeSession`, and `listSessions`
- **Streaming** – assistant text and reasoning tokens are forwarded as incremental ACP chunks
- **Tool calling** – agentic loop with up to 20 turns of GLM function-calling
- **Thinking mode** – GLM-5.1's `reasoning_content` tokens are surfaced as `agent_thought_chunk` blocks so the client can show the model's thinking
- **Six built-in tools** (see below)
- **Protocol-correct stop reasons** – maps model and runtime conditions to ACP `end_turn`, `max_tokens`, `max_turn_requests`, `refusal`, and `cancelled`
- **Protocol-correct tool statuses** – uses ACP tool lifecycle statuses (`pending`, `in_progress`, `completed`, `failed`)

---

## Architecture

```text
ACP Client (IDE plugin, CLI, …)
        │  stdio (ndjson)
        ▼
  GlmAcpAgent          ← ACP protocol layer  (src/protocol/)
        │
        ├─ GlmClient   ← Zhipu AI / Z.AI API  (src/llm/)
        │    └─ streams chat completions with tool schemas
        │
        └─ ToolExecutor ← executes tool calls  (src/tools/)
             ├─ read_file / write_file / list_files  → ACP client (user's machine)
             ├─ run_command                          → ACP client (user's machine)
             ├─ web_search                           → Z.AI /paas/v4/web_search
             └─ web_reader                           → Z.AI /paas/v4/reader
```

The agent process itself only needs network access to `api.z.ai`. All file-system and shell operations are **delegated to the ACP client** (the IDE or tool running on the user's machine) – the agent never touches the local disk directly for those tools.

`web_search` and `web_reader` are different: they run **inside the agent process** and call the Z.AI Tools API directly.

---

## Available Tools

| Tool | Runs on | Description |
|------|---------|-------------|
| `read_file` | ACP client | Read the text content of a file |
| `write_file` | ACP client | Write or overwrite a text file |
| `list_files` | ACP client | List directory contents |
| `run_command` | ACP client | Execute a shell command and capture output |
| `web_search` | Agent (Z.AI) | Search the web – returns titles, URLs, and summaries |
| `web_reader` | Agent (Z.AI) | Fetch and parse a web page (markdown or plain text) |

---

## Prerequisites

- **Node.js** 20 or later (native `fetch` required)
- **npm** 9 or later
- A **Z.AI API key** – obtain one at <https://z.ai/manage-apikey/apikey-list>

---

## Installation

```bash
# Clone the repository
git clone https://github.com/stefandevo/glm-acp-agent.git
cd glm-acp-agent

# Install dependencies
npm install

# Build
npm run build
```

---

## Configuration

The agent is configured entirely through environment variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z_AI_API_KEY` | **Yes** | – | API key for the Z.AI / Zhipu AI service |
| `ACP_GLM_MODEL` | No | `glm-5-1` | Override the GLM model to use |

### Supported models

Any model available on the Z.AI platform can be used. Recommended choices:

| Model | Notes |
|-------|-------|
| `glm-5-1` | Default. Enables "Rethink" thinking mode automatically |
| `glm-4-5` | Fast, cost-efficient |
| `glm-4-long` | Extended context window (up to 1M tokens) |

When the model name starts with `glm-5`, the agent enables GLM's `thinking` mode and forwards reasoning tokens to the client as thought blocks.

---

## Running

### Standalone (stdio)

```bash
export Z_AI_API_KEY=your_key_here
node dist/index.js
```

The agent speaks the ACP ndjson protocol over stdin/stdout. You can connect any ACP-compatible client to it.

### As a global CLI

```bash
npm install -g .
export Z_AI_API_KEY=your_key_here
glm-acp-agent
```

### Development mode (watch)

```bash
export Z_AI_API_KEY=your_key_here
npm run dev        # tsc --watch in one terminal
node dist/index.js # in another terminal
```

---

## Connecting to an ACP Client

### VS Code (via the ACP extension)

Add the following to your VS Code `settings.json`:

```json
{
  "acp.agents": [
    {
      "name": "GLM Agent",
      "command": "node",
      "args": ["/path/to/glm-acp-agent/dist/index.js"],
      "env": {
        "Z_AI_API_KEY": "${env:Z_AI_API_KEY}"
      }
    }
  ]
}
```

### Any ACP-compatible client

Point the client at the `dist/index.js` entry point with `node` as the runtime and set `Z_AI_API_KEY` in the environment.

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
└── tools/
    ├── definitions.ts        # Tool JSON schemas (function-calling format)
    └── executor.ts           # ToolExecutor – dispatches tool calls
```

---

## Building

```bash
npm run build   # one-shot TypeScript compilation → dist/
npm run dev     # watch mode
npm test        # build + run protocol-focused unit tests
```

TypeScript output lands in `dist/` with source maps and declaration files.

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
