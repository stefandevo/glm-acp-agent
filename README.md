# glm-acp-agent

An [Agent Client Protocol (ACP)](https://agentclientprotocol.com) agent written in TypeScript that uses the **Z.AI / Zhipu AI GLM** model family (GLM-5.2, GLM-5.1, GLM-4.7, …) as its reasoning core.

The agent connects to any ACP-compatible IDE or client over **stdio**, streams responses back in real time, and can call a rich set of tools to interact with the user's file system, terminal, and the web.

---

## Coding Plan Only

`glm-acp-agent` is intentionally built for the **Z.AI GLM Coding Plan**. It is not a general-purpose Z.AI Open Platform API client.

By default, model calls use the Coding Plan endpoint:

```text
https://api.z.ai/api/coding/paas/v4
```

The same Coding Plan API key is used for the agent's GLM model calls and supported Coding Plan tools. General Z.AI API/resource-package billing surfaces, such as direct `/api/paas/v4` Tool API calls, are intentionally out of scope for this ACP agent.

Built-in web tools use Coding Plan-compatible MCP endpoints, not the general `/api/paas/v4` Tool API. If you need general Z.AI API billing, separate resource packages, or non-Coding Plan endpoints, use a different provider configuration or fork this agent for that purpose.

---

## Features

- **Full ACP compliance** – implements `initialize`, `authenticate`, `session/new`, `session/set_mode`, `session/prompt`, `session/cancel`, `session/close`, `session/list`, `session/load`, `session/fork`, `session/resume`, and `session/set_model`
- **Streaming** – assistant text and reasoning tokens are forwarded as incremental ACP chunks
- **Tool calling** – agentic loop with up to 20 turns of GLM function calling
- **Thinking mode** – GLM's `reasoning_content` tokens are surfaced as `agent_thought_chunk` blocks so the client can show the model's chain of thought
- **Session permission modes** – supports `default`, `accept_edits`, and `bypass_permissions` via `session/set_mode`. Clients like DevFlow can use this to toggle between prompting for every edit, auto-approving edits while prompting for commands, or bypassing permissions entirely.
- **Per-session model switching** – `session/set_model` lets clients change the active GLM model mid-conversation; `session/new` returns the curated `availableModels` list
- **Image input via Coding Plan-native vision or Vision MCP** – `promptCapabilities.image` is advertised; `glm-5v-turbo` sessions send supported image parts directly to the model, while non-native coding models route pasted ACP image blocks through Z.AI Vision MCP (`@z_ai/mcp-server`). Direct chat-image-only models (e.g. `glm-4v-plus`) are intentionally not used.
- **Session persistence** – conversations are written to `~/.local/state/glm-acp-agent/sessions/` and can be reloaded via `session/load`, branched via `session/fork`, or resumed without replay via `session/resume`
- **Six built-in tools** (see below)
- **Self-sufficient local tools** – file reads/writes, directory listings, and shell commands run in the agent process, so they do not depend on ACP client `fs` or `terminal` capabilities
- **Configurable permissions** – `write_file` and `run_command` behavior depends on the active session mode (prompts by default)
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
        ├─ GlmClient   ← Z.AI / Zhipu AI Coding Plan Chat Completions  (src/llm/)
        │
        ├─ ToolExecutor ← executes tool calls  (src/tools/)
        │    ├─ read_file / write_file       → Agent process (Node fs)
        │    ├─ list_files / run_command     → Agent process (Node fs / child_process)
        │    ├─ web_search / web_reader      → Z.AI Coding Plan Web MCP (HTTP)
        │    └─ image_analysis               → Z.AI Coding Plan Vision MCP (stdio)
        │
        └─ VisionMcpClient ← spawns `npx @z_ai/mcp-server` on demand
```

The agent process needs network access to `api.z.ai` for chat completions and Web MCP, plus `npx` available on `PATH` so it can launch `@z_ai/mcp-server` for vision. Filesystem and shell operations run inside the agent process with paths resolved against the ACP session working directory. Writes and arbitrary shell commands still go through ACP `session/request_permission`, so clients can render an approval prompt before the operation runs.

---

## Available Tools

| Tool | Runs on | Permission behavior | Description |
|------|---------|---------------------|-------------|
| `read_file` | Agent process | Always silent | Read the text content of a file |
| `write_file` | Agent process | Mode-dependent | Write or overwrite a text file. Silent in `accept_edits` and `bypass_permissions`. |
| `list_files` | Agent process | Always silent | List a directory using Node filesystem APIs |
| `run_command` | Agent process | Mode-dependent | Run an arbitrary shell command. Silent only in `bypass_permissions`. |
| `web_search` | Agent (Z.AI Coding Plan MCP) | Always silent | Search the web — returns titles, URLs, and summaries |
| `web_reader` | Agent (Z.AI Coding Plan MCP) | Always silent | Fetch and parse a web page (markdown or plain text) |
| `image_analysis` | Agent (Z.AI Vision MCP, stdio) | Always silent | Analyze a local image path or remote URL using `@z_ai/mcp-server` |

### Session Modes

Clients can use `session/set_mode` to drive the permission policy:

| Mode ID | Name | `write_file` | `run_command` |
|---|---|---|---|
| `default` | Ask for permission | **Prompt** | **Prompt** |
| `accept_edits` | Auto-approve edits | Silent | **Prompt** |
| `bypass_permissions` | Bypass all permissions | Silent | Silent |

Reads, listings, and MCP tool calls are always silent across all modes.

---

## Prerequisites

- **Node.js** 20 or later (native `fetch` and Web Streams required)
- **npm** 9 or later
- A **Z.AI API key** — obtain one at <https://z.ai/manage-apikey/apikey-list>

---

## Installation

### Quick Start

Install the published package globally to get the `glm-acp-agent` command on your `PATH`:

```bash
npm install -g glm-acp-agent@latest
```

Then either export your Z.AI API key and run it directly:

```bash
export Z_AI_API_KEY=your_key_here
glm-acp-agent
```

…or run the interactive setup once to persist the key to disk (see [One-time setup](#one-time-setup)) and point any ACP-compatible client at the `glm-acp-agent` command.

### From Source (Development)

Clone the repository if you want to hack on the agent or pin a specific commit:

```bash
git clone https://github.com/stefandevo/glm-acp-agent.git
cd glm-acp-agent
npm install
npm run build
```

The build output lands in `dist/`; the rest of this README uses `node dist/index.js` whenever it refers to the source-build entry point.

---

## Configuration

The agent reads its configuration from environment variables, plus an optional credentials file written by `glm-acp-agent --setup`.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `Z_AI_API_KEY` | One of env / `--setup` | — | API key for the Z.AI / Zhipu AI service. If unset, the credentials file is consulted. |
| `ACP_GLM_MODEL` | No | `glm-5.2` | Default GLM model for new sessions |
| `ACP_GLM_AVAILABLE_MODELS` | No | built-in list | Comma-separated list of model ids advertised in `session/set_model` |
| `ACP_GLM_BASE_URL` | No | `https://api.z.ai/api/coding/paas/v4` | Override the API base URL |
| `ACP_GLM_MAX_TOKENS` | No | `8192` | Cap on `max_tokens` for each completion |
| `ACP_GLM_THINKING` | No | auto-detected | Force thinking mode `true` / `false` |
| `ACP_GLM_SESSION_DIR` | No | `$XDG_STATE_HOME/glm-acp-agent/sessions` | Where session JSON files are persisted |
| `ACP_GLM_DEBUG` | No | — | Set to `true` or `1` to enable verbose debug logging to stderr (shows model selection, API key resolution, tool calls, and usage stats) |
| `XDG_CONFIG_HOME` | No | `~/.config` | Where the credentials file is read/written |

### One-time setup

If you'd rather not pass `Z_AI_API_KEY` through your ACP client's environment block, run the interactive setup once and the agent will read the key from disk on subsequent launches:

```bash
glm-acp-agent --setup
```

If you're working from a source clone instead of the published package, use the build output directly:

```bash
node dist/index.js --setup
```

The key is written to `$XDG_CONFIG_HOME/glm-acp-agent/credentials.json` (default: `~/.config/glm-acp-agent/credentials.json`) with `0600` permissions. The `Z_AI_API_KEY` environment variable, when set, always wins over the file.

### Supported models

The agent advertises only the models on the current Z.AI Coding Plan allowlist:

| Model | Notes |
|-------|-------|
| `glm-5.2` | **Default.** Newest 1M-context coding model; thinking mode auto-enabled |
| `glm-5.1` | Long-horizon coding model; thinking mode auto-enabled |
| `glm-5-turbo` | Faster Coding Plan reasoning model |
| `glm-5v-turbo` | Multimodal Coding Plan model; native image understanding for jpg / jpeg / png inputs |
| `glm-4.7` | 200K-context reasoning model |
| `glm-4.5-air` | Lightweight, lower-latency model |

`ACP_GLM_AVAILABLE_MODELS` still lets you advertise custom IDs, but custom IDs sit outside the supported Coding Plan list — the Coding Plan endpoint will reject any model code Z.AI hasn't whitelisted (business code `1211`). If you override the model list, include `glm-5v-turbo` yourself to keep it visible in the picker.

Vision-only chat models (`glm-4v-plus` etc.) are **not** advertised. The multimodal coding model `glm-5v-turbo` is advertised because it is on the Coding Plan and receives supported ACP image blocks directly as native `image_url` content parts. Other advertised models keep using the [Vision MCP](#vision-mcp) path for image analysis.

When the model name matches `glm-4.5`, `glm-4.6`, `glm-4.7`, or the `glm-5` family, the agent enables Z.AI's `thinking: { type: "enabled" }` extension and forwards reasoning tokens to the client as `agent_thought_chunk` blocks. This includes `glm-5v-turbo`. Override with `ACP_GLM_THINKING=false` if you want plain completions only.

`ACP_GLM_PROMPT_IMAGES=false` still hides the image-attachment capability at session startup. With that flag set, users can pick `glm-5v-turbo` for text work but clients should not offer image attachments.

### Vision MCP

For `glm-5v-turbo`, pasted ACP image blocks with `image/jpeg`, `image/jpg`, or `image/png` are sent directly to chat completions as `image_url` content parts. HTTPS image URLs are forwarded as URLs; inline base64 data is sent as a `data:<mime>;base64,...` URI. Unsupported image MIME types are rejected client-side with an inline `<image_unsupported_format>` annotation so the prompt can continue without a provider 4xx.

For non-native models, pasted ACP image blocks are not sent to the chat-completions endpoint. Instead, the agent boots `@z_ai/mcp-server` over stdio (via `npx -y @z_ai/mcp-server@latest`) and calls its `image_analysis` tool. The text result is spliced into the user message as `<image_analysis index="N">…</image_analysis>` so the regular Coding Plan model can reason about it.

Prerequisites:

- `npx` on `PATH` (Node 18+ / npm 9+).
- The same `Z_AI_API_KEY` used for chat completions; the agent forwards it to the MCP server as `Z_AI_API_KEY` plus `Z_AI_MODE=ZAI`.

The model can also call `image_analysis` explicitly with `{ image_source: "/path/or/url", prompt?: "…" }`. Vision failures (missing `npx`, MCP startup, quota) are surfaced as actionable errors but never abort the prompt; an inline `<image_analysis_error>` annotation is used so the conversation can continue.

---

## Running

### Standalone (stdio)

If you installed the published package globally, just run the CLI:

```bash
export Z_AI_API_KEY=your_key_here
glm-acp-agent
```

If you built from source, invoke the entry point directly:

```bash
export Z_AI_API_KEY=your_key_here
node dist/index.js
```

The agent speaks the ACP newline-delimited JSON protocol over stdin/stdout. You can connect any ACP-compatible client to it.

### Development mode (watch)

When working from a source clone, run `tsc` in watch mode so `dist/` rebuilds on every save:

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

#### 2. Install the agent

Follow either path from the [Installation](#installation) section above:

- **Quick Start** — `npm install -g glm-acp-agent@latest`. Zed can then spawn the agent with `"command": "glm-acp-agent"`.
- **From source** — clone, `npm install`, `npm run build`. Zed needs the absolute path to the built entry point (no `~`, no `$HOME` shortcuts):

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
      "command": "glm-acp-agent",
      "env": { "Z_AI_API_KEY": "sk-…" }
    }
  }
}
```

If you installed from source instead of the published package, replace `command` / `args` with the absolute path to the build output:

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

Run the interactive setup once:

```bash
glm-acp-agent --setup
```

If you're working from a source clone, use the build output directly: `node dist/index.js --setup`.

The key is written to `~/.config/glm-acp-agent/credentials.json` with `0600` permissions. Then drop the `env` block from the Zed entry — the agent will read the file on launch:

```json
{
  "agent_servers": {
    "glm": {
      "command": "glm-acp-agent"
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
5. You should see streaming text, a `read_file` tool call awaiting permission, and (with a thinking-capable model like `glm-5.2`) reasoning surfaced as a separate thought block.

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
- **A write or command did not run.** — approve the ACP permission prompt for that tool call. Rejected or cancelled prompts are reported back to the model as skipped operations.

### Neovim / VS Code / JetBrains / any ACP client

Any client that supports configuring an ACP agent via a `command` + `args` invocation works the same way:

- If you installed via `npm install -g glm-acp-agent@latest`, set `command` to `glm-acp-agent` (no args required).
- If you built from source, set `command` to `node` and `args` to `["/absolute/path/to/glm-acp-agent/dist/index.js"]`.

Supply `Z_AI_API_KEY` in the environment, or run `glm-acp-agent --setup` once so the agent reads the key from disk on launch.

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
- Agent-owned local file and shell tools that work without ACP `fs` / `terminal` client capabilities
- Permission flows for `write_file` and `run_command` (allow / reject / cancel)
- Shell-quoted argument handling for `list_files` and `run_command`
- GLM streaming: text deltas, `reasoning_content` deltas, multi-chunk tool-call assembly, and trailing usage
- Image preprocessing through a mocked Vision MCP client and graceful degradation on Vision MCP failures

---

## Troubleshooting

- **`No API key found.`** — either set `Z_AI_API_KEY` in the environment, or run `glm-acp-agent --setup` (or `node dist/index.js --setup` from a source clone) once to store the key on disk.
- **`HTTP 401: Invalid API key`** — your key is wrong or expired; rotate it on <https://z.ai/manage-apikey/apikey-list>.
- **Writes or commands never get to run.** — make sure your ACP client supports `session/request_permission` and that you approve the prompt for the specific tool call.

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
