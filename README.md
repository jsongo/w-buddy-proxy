# CodeBuddy Proxy

> A lightweight local proxy that turns CodeBuddy's underlying chat interface into standard **OpenAI Chat Completions**, **Responses**, and **Anthropic Messages** protocols — so you can plug CodeBuddy models into Codex CLI, Claude Code / CC Switch, OpenCode, Grok, Oh My Pi and any OpenAI-compatible client.

> **中文文档见 [README_zh.md](README_zh.md).**

---

## Features

- **Protocol conversion** — `/v1/chat/completions` (OpenAI), `/v1/responses` (Codex CLI), `/v1/messages` (Anthropic / Claude Code)
- **Model list** — `/v1/models` returns an OpenAI-compatible model list plus rich per-model metadata (context window, capabilities)
- **Desensitization** (`--desensitize`) — inserts zero-width spaces into compliance terms inside system messages to avoid backend false-blocking by keyword review
- **Message compression** (`--optimize-context`) — compresses long histories / large schemas / oversized tool output for `/v1/responses`, cutting token usage dramatically
- **Tool calls** — full function calling support with automatic filtering of invalid tool definitions
- **DSML parsing** — detects and converts DeepSeek Markup Language tool calls
- **Streaming** — SSE output with idle / total-duration timeout protection
- **Multi-account** — isolated session files for work / personal accounts

---

## Install & run

The proxy is a plain Python package under `src/`. Run from source with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -m codebuddy_proxy --desensitize
```

First-time login (opens a browser):

```bash
uv run python -m codebuddy_proxy --login --desensitize
```

It listens on `http://127.0.0.1:8787` by default.

> A handy `start.sh` is included: `uv run python -m codebuddy_proxy --desensitize --port 8787`.

## Quick check

```bash
curl http://127.0.0.1:8787/health      # service + auth status
curl http://127.0.0.1:8787/v1/models    # model list
```

## Connect clients

### Codex CLI

`~/.codex/config.toml`:

```toml
[model_providers.codebuddy]
name = "CodeBuddy (via local proxy)"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"

[profiles.codebuddy]
model = "glm-5.2"
model_provider = "codebuddy"
```

```bash
codex --profile codebuddy "your task"
```

### Claude Code / CC Switch

```json
{
  "DeepSeek-V4": {
    "base_url": "http://127.0.0.1:8787/v1/messages",
    "api_key": "",
    "model": "deepseek-v4-pro"
  }
}
```

### OpenCode

`opencode.json`:

```json
{
  "model": "codebuddy/glm-5.2",
  "providers": {
    "codebuddy": {
      "name": "CodeBuddy (via local proxy)",
      "package": "@opencode-ai/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8787/v1", "apiKey": "noop" },
      "models": {
        "glm-5.2":         { "modelID": "glm-5.2",         "name": "GLM-5.2" },
        "deepseek-v4-pro": { "modelID": "deepseek-v4-pro", "name": "DeepSeek V4 Pro" },
        "kimi-k2.7":       { "modelID": "kimi-k2.7",       "name": "Kimi K2.7" }
      }
    }
  }
}
```

### Grok CLI

`~/.grok/config.toml`:

```toml
[models]
default = "hy3"

[model.hy3]
model = "hy3"
base_url = "http://127.0.0.1:8787/v1"
name = "HY3 Main"
api_key = "noop"

[model.dv4f]
model = "deepseek-v4-flash"
base_url = "http://127.0.0.1:8787/v1"
name = "DeepSeek V4 Flash"
api_key = "noop"
```

### Oh My Pi (OMP)

`~/.omp/agent/models.yml`:

```yaml
providers:
  codebuddy:
    baseUrl: http://127.0.0.1:8787/v1
    api: openai-completions
    auth: none
    models:
      - id: hy3
        name: Hy3 (CodeBuddy)
        reasoning: true
        contextWindow: 192000
        maxTokens: 64000
      - id: deepseek-v4-flash
        name: DeepSeek V4 Flash (CodeBuddy)
        reasoning: true
        contextWindow: 1000000
        maxTokens: 50000
```

### Other OpenAI-compatible clients

- Base URL: `http://127.0.0.1:8787/v1`
- API Key: blank (or your `--api-key`)
- Model: any id from `/v1/models`, e.g. `glm-5.2`, `deepseek-v4-pro`, `kimi-k2.7`, `auto`

---

## Command-line options

```
--host HOST               bind address (default 127.0.0.1)
--port PORT               bind port (default 8787)
--endpoint ENDPOINT       CodeBuddy backend address
--session-file PATH       session file (default ~/.codebuddy-session.json)
--log-file PATH           JSONL log (default ~/.codebuddy-proxy/codebuddy-proxy.jsonl)
--desensitize             enable desensitization (recommended)
--optimize-context        enable message compression (recommended for Codex)
--login                   browser login at startup
--no-browser              don't auto-open the browser on login
--verbose-llm             log full request/response bodies
--mock-dir DIR            serve recorded fixtures (testing)
```

Env vars: `CODEBUDDY_PROXY_HOST`, `CODEBUDDY_PROXY_PORT`, `CODEBUDDY_ENDPOINT`, `CODEBUDDY_PROXY_LOG_FILE`.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/health`             | service + auth status |
| GET  | `/v1/models`          | model list |
| POST | `/v1/chat/completions` | OpenAI chat (tools + streaming) |
| POST | `/v1/responses`       | Responses API (Codex CLI) |
| POST | `/v1/messages`        | Anthropic Messages (Claude Code) |

Endpoints authenticate using the local session; no extra token is required.

## Disclaimer

For learning and research purposes only. Please comply with CodeBuddy's Terms of Service. Use at your own risk.
