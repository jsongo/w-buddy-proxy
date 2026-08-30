# CodeBuddy Proxy

> 一个轻量级本地代理，把 CodeBuddy 底层的聊天接口转换成标准的 **OpenAI Chat Completions**、**Responses** 和 **Anthropic Messages** 协议——让你能把 CodeBuddy 模型接到 Codex CLI、Claude Code / CC Switch、OpenCode、Grok、Oh My Pi 以及任意 OpenAI 兼容客户端上。

> **English docs: [README.md](README.md).**

---

## 特性

- **协议转换** — `/v1/chat/completions`(OpenAI)、`/v1/responses`(Codex CLI)、`/v1/messages`(Anthropic / Claude Code)
- **模型列表** — `/v1/models` 返回 OpenAI 兼容的模型列表，附带每个模型的完整元数据（上下文窗口、能力）
- **脱敏**（`--desensitize`）— 向 system 消息里的合规关键词插入零宽空格，避免后端关键词审核误拦
- **消息压缩**（`--optimize-context`）— 压缩长历史 / 大 schema / 超大工具输出，大幅降低 token 消耗
- **工具调用** — 完整的 function calling 支持，自动过滤无效工具定义
- **DSML 解析** — 自动识别并转换 DeepSeek Markup Language 工具调用
- **流式输出** — SSE 实时返回，带空闲 / 总时长双重超时保护
- **多账号** — 隔离的 session 文件，方便工作 / 个人账号切换

---

## 安装与启动

代理是 `src/` 下的一个普通 Python 包，用 [uv](https://docs.astral.sh/uv/) 从源码运行：

```bash
uv sync
uv run python -m codebuddy_proxy --desensitize
```

首次使用需要登录（会打开浏览器）：

```bash
uv run python -m codebuddy_proxy --login --desensitize
```

默认监听 `http://127.0.0.1:8787`。

> 仓库自带的 `start.sh`：`uv run python -m codebuddy_proxy --desensitize --port 8787`

## 快速验证

```bash
curl http://127.0.0.1:8787/health      # 服务 + 认证状态
curl http://127.0.0.1:8787/v1/models    # 模型列表
```

## 接入客户端

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

### 其它 OpenAI 兼容客户端

- Base URL：`http://127.0.0.1:8787/v1`
- API Key：留空（或启动时设置的 `--api-key`）
- Model：`/v1/models` 里的任意 id，如 `glm-5.2`、`deepseek-v4-pro`、`kimi-k2.7`、`auto`

---

## 命令行参数

```
--host HOST               绑定地址（默认 127.0.0.1）
--port PORT               绑定端口（默认 8787）
--endpoint ENDPOINT       CodeBuddy 后端地址
--session-file PATH       会话文件（默认 ~/.codebuddy-session.json）
--log-file PATH           JSONL 日志（默认 ~/.codebuddy-proxy/codebuddy-proxy.jsonl）
--desensitize             启用脱敏（推荐）
--optimize-context        启用消息压缩（Codex 场景推荐）
--login                   启动时浏览器登录
--no-browser              登录时不自动打开浏览器
--verbose-llm             记录完整请求/响应体
--mock-dir DIR            使用录制的响应（测试用）
```

环境变量：`CODEBUDDY_PROXY_HOST`、`CODEBUDDY_PROXY_PORT`、`CODEBUDDY_ENDPOINT`、`CODEBUDDY_PROXY_LOG_FILE`。

## API 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET  | `/health`             | 服务 + 认证状态 |
| GET  | `/v1/models`          | 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 对话（工具 + 流式） |
| POST | `/v1/responses`       | Responses API（Codex CLI） |
| POST | `/v1/messages`        | Anthropic Messages（Claude Code） |

端点使用本地 session 认证，无需额外 token。

## 免责声明

本项目仅供学习与研究使用，请遵守 CodeBuddy 的服务条款，使用风险自负。
