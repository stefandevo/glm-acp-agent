# glm-acp-agent

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com) agent written in TypeScript that uses the **Z.AI / Zhipu AI GLM** model family (GLM-5.1, GLM-4.7, GLM-4.6, …) as its reasoning core.

The agent connects to any ACP-compatible IDE or client over **stdio**, streams responses back in real time, and can call a rich set of tools to interact with the user's file system, terminal, and the web.

---

## Features

- **Full ACP compliance** – implements `initialize`, `authenticate`, `session/new`, `session/set_mode`, `session/prompt`, `session/cancel`, `session/close`, `session/list`, `session/load`, `session/fork`, `session/resume`, and `session/set_model`
- **Streaming** – assistant text and reasoning tokens are forwarded as incremental ACP chunks
- **Tool calling** – agentic loop with up to 20 turns of GLM function calling
- **Thinking mode** – GLM's `reasoning_content` tokens are surfaced as `agent_thought_chunk` blocks so the client can show the model's chain of thought
- **Per-session model switching** – `session/set_model` lets clients change the active GLM model mid-conversation; `session/new` returns the curated `availableModels` list
- **Image input** – `promptCapabilities.image` is advertised; image content blocks are forwarded as data-URL parts so vision-capable models (e.g. `glm-4v-plus`) can ingest them
- **Session persistence** – conversations are written to `~/.local/state/glm-acp-agent/sessions/` and can be reloaded via `session/load`, branched via `session/fork`, or resumed without replay via `session/resume`
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

The agent reads its configuration from environment variables, plus an optional credentials file written by `glm-acp-agent --setup`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z_AI_API_KEY` | One of env / `--setup` | — | API key for the Z.AI / Zhipu AI service. If unset, the credentials file is consulted. |
| `ACP_GLM_MODEL` | No | `glm-5.1` | Default GLM model for new sessions |
| `ACP_GLM_AVAILABLE_MODELS` | No | built-in list | Comma-separated list of model ids advertised in `session/set_model` |
| `ACP_GLM_BASE_URL` | No | `https://api.z.ai/api/paas/v4` | Override the API base URL |
| `ACP_GLM_MAX_TOKENS` | No | `8192` | Cap on `max_tokens` for each completion |
| `ACP_GLM_THINKING` | No | auto-detected | Force thinking mode `true` / `false` |
| `ACP_GLM_SESSION_DIR` | No | `$XDG_STATE_HOME/glm-acp-agent/sessions` | Where session JSON files are persisted |
| `XDG_CONFIG_HOME` | No | `~/.config` | Where the credentials file is read/written |

### One-time setup

If you'd rather not pass `Z_AI_API_KEY` through your ACP client's environment block, run the interactive setup once and the agent will read the key from disk on subsequent launches:

```bash
glm-acp-agent --setup
```

The key is written to `$XDG_CONFIG_HOME/glm-acp-agent/credentials.json` (default: `~/.config/glm-acp-agent/credentials.json`) with `0600` permissions. The `Z_AI_API_KEY` environment variable, when set, always wins over the file.

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

### Zed (recommended for local testing)

[Zed](https://zed.dev) is currently the most polished editor for trying out ACP agents locally — it spawns the agent process over stdio and surfaces it in the agent panel. This is the fastest way to iterate on `glm-acp-agent` before publishing it to the ACP registry or to npm.

#### 1. Prerequisites

- A recent build of [Zed](https://zed.dev/download) that supports the `agent_servers` setting
- Node.js 20 or later on your `PATH` (`node --version`)
- A Z.AI API key — create one at <https://z.ai/manage-apikey/apikey-list>

#### 2. Build the agent

```bash
git clone https://github.com/stefandevo/glm-acp-agent.git
cd glm-acp-agent
npm install
npm run build
```

Take note of the absolute path to the freshly built entry point — Zed needs it (no `~`, no `$HOME` shortcuts):

```bash
echo "$(pwd)/dist/index.js"
```

#### 3. Wire it into Zed

Open (or create) `~/.config/zed/settings.json` and add an `agent_servers` entry. Pick **one** of the two API-key strategies below.

**Option A — inline `env` block (simplest):**

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

**Option B — credentials file (no key in your editor settings):**

Run the interactive setup once. From the project directory:

```bash
node dist/index.js --setup
```

…or, if you `npm install -g .`'d the package:

```bash
glm-acp-agent --setup
```

The key is written to `~/.config/glm-acp-agent/credentials.json` with `0600` permissions. Then drop the `env` block from the Zed entry — the agent will read the file on launch:

```json
{
  "agent_servers": {
    "glm": {
      "command": "node",
      "args": ["/absolute/path/to/glm-acp-agent/dist/index.js"]
    }
  }
}
```

If `Z_AI_API_KEY` is set in the environment **and** a credentials file exists, the environment variable wins.

#### 4. Verify it works

1. Save `settings.json`. Zed reloads settings automatically.
2. Open the **agent panel** (use the command palette: `agent panel: toggle focus`).
3. In the agent picker, select **glm** — Zed labels external agents by their `agent_servers` key.
4. Start a new thread and send a small prompt that exercises a tool, e.g. `Read package.json and tell me the project name.`
5. You should see streaming text, a `read_file` tool call awaiting permission, and (with a thinking-capable model like `glm-5.1`) reasoning surfaced as a separate thought block.

#### 5. Iterating on the agent

After editing the source, rebuild:

```bash
npm run build
```

Zed spawns a fresh agent process per thread, so the easiest way to pick up changes is to **start a new thread** with the glm agent. If you see stale behavior, fully quit and reopen Zed — that guarantees no in-flight process is reused.

For a tighter inner loop, run `npm run dev` in a terminal so `dist/` rebuilds on every save; you only need to start a new Zed thread to test the latest code.

#### 6. Troubleshooting

- **The "glm" agent doesn't appear in the picker.** — `settings.json` likely has a JSON parse error, or your Zed build is older than `agent_servers` support. Open the Zed log via the command palette (`zed: open log`) and look for settings errors.
- **`Error: Cannot find module '/.../dist/index.js'`** — you skipped `npm run build`, or the path in `args` is wrong. It must be absolute and point at a file that exists.
- **`No API key found.`** — neither `Z_AI_API_KEY` nor `~/.config/glm-acp-agent/credentials.json` is set. Use Option A or Option B in step 3.
- **`HTTP 401: Invalid API key`** — your key is wrong, expired, or for the wrong region. Rotate it on <https://z.ai/manage-apikey/apikey-list>.
- **`client does not advertise the … capability`** — Zed didn't expose that capability to the agent (e.g. `terminal` for `run_command`/`list_files`). Ask the model to use a different tool, or upgrade Zed.

### Neovim / VS Code / JetBrains / any ACP client

Any client that supports configuring an ACP agent via a `command` + `args` invocation works the same way: point it at `node /absolute/path/to/glm-acp-agent/dist/index.js` and supply `Z_AI_API_KEY` in the environment.

### Authentication

The agent advertises two authentication methods at `initialize` time:

1. An **`agent`-default** method — the agent will read the API key itself, either from the `Z_AI_API_KEY` environment variable or from the credentials file written by `glm-acp-agent --setup`.
2. An **`env_var`** method (experimental SDK extension) describing the `Z_AI_API_KEY` variable so capable clients can prompt the user and inject it.

ACP clients that support the auth-methods proposal will use whichever method they recognise; clients that don't handle auth methods should set `Z_AI_API_KEY` themselves before launching the agent (or run `glm-acp-agent --setup` once and let the agent read it from disk).

---

## Project Structure

```text
src/
├── index.ts                  # Entry point – starts stdio connection or --setup flow
├── setup.ts                  # Interactive credential setup (`--setup`)
├── llm/
│   ├── glm-client.ts         # OpenAI-compatible client for Z.AI / Zhipu AI
│   └── credentials.ts        # API-key resolution (env var > credentials.json)
├── protocol/
│   ├── connection.ts         # Sets up the ACP stdio connection
│   ├── agent.ts              # GlmAcpAgent – ACP protocol implementation
│   └── session-store.ts      # On-disk persistence for load/fork/resume
├── tools/
│   ├── definitions.ts        # Tool JSON schemas (function-calling format)
│   └── executor.ts           # ToolExecutor – dispatches tool calls
└── tests/
    ├── agent.test.ts         # Protocol-level tests for GlmAcpAgent
    ├── credentials.test.ts   # Credential resolution and --setup persistence
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

- **`No API key found.`** — either set `Z_AI_API_KEY` in the environment, or run `glm-acp-agent --setup` once to store the key on disk.
- **`HTTP 401: Invalid API key`** — your key is wrong or expired; rotate it on <https://z.ai/manage-apikey/apikey-list>.
- **The agent says "client does not advertise the … capability".** — your ACP client doesn't expose that capability (e.g. terminal). Ask the model to use a different tool, or upgrade the client.
- **Tools never get to run.** — make sure the client is sending the `clientCapabilities` field in `initialize`; the agent uses it to decide which tools to expose to the model.

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
