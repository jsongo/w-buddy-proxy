# CodeBuddy Proxy

> A lightweight local proxy that turns CodeBuddy's underlying chat interface into standard **OpenAI Chat Completions**, **Responses**, and **Anthropic Messages** protocols — so you can plug CodeBuddy models into Codex CLI, Claude Code / CC Switch, OpenCode, Grok, Oh My Pi and any OpenAI-compatible client.

> **中文文档见 [README_zh.md](README_zh.md).**

---

## Features

- **Protocol conversion** — `/v1/chat/completions` (OpenAI), `/v1/responses` (Codex CLI), `/v1/messages` (Anthropic / Claude Code)
- **Admin UI** — built-in web console at `/ui`: browse models grouped by provider, one-click "set as default model", one-click test per model (sends a "hi"), and per-model request stats with charts
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
uv run python -m buddy_proxy --desensitize
```

### The `buddy` command (recommended)

`buddy` is the day-to-day entry point: one command starts the proxy and opens the admin UI. It can also register the proxy as a macOS launchd service (auto-start at login, automatic restart on crash).

```bash
./buddy start              # start (if not running) and open http://127.0.0.1:8787/ui
./buddy stop / restart / status / logs
./buddy login [provider]   # upstream login (codebuddy(=workbuddy)/trae/zcode/doubao)
./buddy ui                 # just open the admin UI (starts the proxy if needed)

# one-time install: put buddy on your PATH so it works from anywhere
./buddy install            # -> /usr/local/bin (falls back to ~/.local/bin)

# register as a system service (launchd)
./buddy service install    # start/stop/restart now route through launchctl
./buddy service status
./buddy service uninstall
```

- When the service is installed, `buddy start/stop/restart` automatically use `launchctl`; otherwise they fall back to `proxy.sh`'s pid management.
- Env vars `PROXY_HOST` (default `0.0.0.0`), `PROXY_PORT` (default `8787`) and `PROXY_EXTRA_ARGS` apply to both `start` and `service install`.

### The `proxy.sh` management script

`buddy` calls `proxy.sh` internally; you can drive it directly for finer control:

```bash
./proxy.sh start          # background start, returns immediately
./proxy.sh stop           # stop the running instance
./proxy.sh restart        # restart (with the same args)
./proxy.sh status         # show PID and listening address
./proxy.sh logs           # tail -F the log file
./proxy.sh ui             # ensure it's running, then open the admin UI
./proxy.sh login codebuddy  # upstream login (also: trae / zcode / doubao)

# customize host / port / args
./proxy.sh start -p 9000 -H 0.0.0.0
PROXY_PORT=9000 PROXY_EXTRA_ARGS="--desensitize --optimize-context" ./proxy.sh start
```

The script:
- Detects `.venv/bin/python` automatically (preferring the project venv).
- Uses `nohup ... &` so `start` returns immediately — terminal is **not** blocked.
- Writes the PID to `logs/proxy.pid` and startup output to `logs/proxy.sh.log`; app logs rotate daily (`logs/proxy.log` + `logs/buddy-proxy.jsonl`, 30 days kept).
- Stops cleanly with `kill`; falls back to `kill -9` after 10s if the process doesn't exit.

---

## Models

The model catalog is maintained in `src/buddy_proxy/models_config.json` — `/v1/models` always serves it (offline-reliable, no remote dependency). The catalog currently ships **11 models** with their credit multiplier (× base cost). `GET /v1/models` → `data[].credits` / `models[].credits` exposes the multiplier:

| id | name | credits |
|---|---|---|
| `auto` | Auto (fast / balanced / ultimate → 0.21 / 0.65 / 1.20) | dynamic |
| `default` | Default | x2.20 |
| `glm-5.3` | GLM-5.3 | x0.79 |
| `glm-5.3-flash` | GLM-5.3-Flash | x0.06 |
| `hy3` | Hy3 (限时免费) | x0.00 |
| `hy4-preview` | Hy4 preview | x0.29 |
| `minimax-m3` | MiniMax-M3 | x0.25 |
| `kimi-k3` | Kimi-K3 | x1.62 |
| `kimi-k2.7` | Kimi-K2.7-Code | x0.57 |
| `deepseek-v4-flash` | Deepseek-V4-Flash | x0.17 |
| `deepseek-v4-pro` | Deepseek-V4-Pro | x0.51 |

Edit `src/buddy_proxy/models_config.json` to add or tweak entries — changes take effect on restart.

First-time login (opens a browser):

```bash
uv run python -m buddy_proxy --login --desensitize
```

It listens on `http://127.0.0.1:8787` by default; the admin UI lives at **http://127.0.0.1:8787/ui** — see [Admin UI](#admin-ui-ui) below.

## Quick check

```bash
curl http://127.0.0.1:8787/health      # service + auth status
curl http://127.0.0.1:8787/v1/models    # model list
```

## Admin UI (`/ui`)

Open <http://127.0.0.1:8787/ui> in a browser (or just run `buddy start` / `buddy ui`):

- **Default model** — models are grouped by provider; click "Set as default" (设为默认) on any model to make it the proxy default. Client requests **without a `model` field** are routed to it automatically. Settings persist in `~/.buddy-proxy/settings.json` (override via `BUDDY_PROXY_SETTINGS`) and survive restarts; `--default-model zcode/glm-5.3` seeds the initial value (an existing settings file wins).
- **One-click test** — every model row has a "Test" (测试) button that sends a real `hi` upstream and shows latency, token usage and the reply preview (non-streaming, `max_tokens=256` — a real, billable upstream call).
- **Stats & charts** — per-provider/per-model request counts, errors, average latency and token usage: a 14-day stacked daily chart, a top-models bar list, and the latest 50 requests. Each completed request appends one line to `logs/metrics.jsonl`; the tail is reloaded on startup so history survives restarts (30 days kept).
- **Provider health** — login/config status at a glance (CodeBuddy session, zcode key, ...).
- **Auto check-in & calendar** — both Trae and CodeBuddy expose daily sign-in: tick "Auto check-in" (自动打卡) and the proxy claims them every day at the configured time (default 09:30; if the proxy starts later it catches up immediately). The last 35 days are shown as a calendar; "Check in now" (立即打卡) claims manually. Campaigns can be seasonal — when CodeBuddy's is closed the UI shows "no sign-in activity today" and skips it. History is appended to `logs/checkin.jsonl`. ZCode (GLM Coding Plan) / Doubao have no sign-in API.
- **Quota** — remaining allowance per provider at a glance: CodeBuddy credit packs (total remaining + per-pack detail), Trae total allowance + entitlement pack expiry, ZCode 5-hour / weekly windows with reset times. Quota responses are cached for 5 minutes.

The admin UI also lets you switch the **fallback provider** (used when a request matches no model); it persists in the same settings file.

Security: the `/ui/api/*` admin endpoints are **restricted to localhost (127.0.0.1)**. To manage the proxy from the LAN, set `BUDDY_PROXY_ADMIN_OPEN=1` (at your own risk). `/v1/*` proxy endpoints are unaffected.

## Connect clients

### Codex CLI

`~/.codex/config.toml`:

```toml
[model_providers.codebuddy]
name = "CodeBuddy (via local proxy)"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"

[profiles.codebuddy]
model = "glm-5.3"
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
  "model": "codebuddy/glm-5.3",
  "providers": {
    "codebuddy": {
      "name": "CodeBuddy (via local proxy)",
      "package": "@opencode-ai/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8787/v1", "apiKey": "noop" },
      "models": {
        "glm-5.3":         { "modelID": "glm-5.3",         "name": "GLM-5.3" },
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
- Model: any id from `/v1/models`, e.g. `glm-5.3`, `deepseek-v4-pro`, `kimi-k2.7`, `auto`

---

## Command-line options

```
--host HOST               bind address (default 127.0.0.1)
--port PORT               bind port (default 8787)
--endpoint ENDPOINT       CodeBuddy backend address
--session-file PATH       session file (default ~/.codebuddy-session.json)
--log-file PATH           JSONL log (default <project>/logs/buddy-proxy.jsonl, override via BUDDY_PROXY_LOG_FILE)
--desensitize             enable desensitization (recommended)
--optimize-context        enable message compression (recommended for Codex)
--default-model MODEL     default model, e.g. zcode/glm-5.3; used when a request has no `model`
                          field. Seeded into the settings file on first start; afterwards
                          ~/.buddy-proxy/settings.json (editable from the admin UI) wins
--login                   browser login at startup
--no-browser              don't auto-open the browser on login
--verbose-llm             log full request/response bodies
--mock-dir DIR            serve recorded fixtures (testing)
```

Env vars: `BUDDY_PROXY_HOST`, `BUDDY_PROXY_PORT`, `CODEBUDDY_ENDPOINT`, `BUDDY_PROXY_LOG_FILE`, `BUDDY_PROXY_SETTINGS` (settings file path), `BUDDY_PROXY_ADMIN_OPEN=1` (lift the localhost-only restriction on admin endpoints), `WB_TRAE_HEARTBEAT_INTERVAL` (Trae stream heartbeat while waiting for the buffered upstream response, seconds, default 45, `0` disables — keeps clients with per-chunk timeouts like Ethan's 120s from aborting long generations). `WB_TRAE_NATIVE_TOOLS` (Trae native function-calling channel, default `1`; `0` falls back to the legacy prompt-taught text protocol). `WB_TRAE_IDE_VERSION_CODE` (Trae client version header, default `20260906` — the upstream gates per-model capabilities by this header; raise it when a model suddenly 4001s).

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET  | `/ui`                 | admin UI (`/` 302-redirects here) |
| GET  | `/ui/api/*`           | admin API (overview / models / stats / benefits / settings / checkin / test, localhost only) |
| GET  | `/health`             | service + auth status |
| GET  | `/v1/models`          | model list |
| POST | `/v1/chat/completions` | OpenAI chat (tools + streaming) |
| POST | `/v1/responses`       | Responses API (Codex CLI) |
| POST | `/v1/messages`        | Anthropic Messages (Claude Code) |

Endpoints authenticate using the local session; no extra token is required.

## Disclaimer

For learning and research purposes only. Please comply with CodeBuddy's Terms of Service. Use at your own risk.
