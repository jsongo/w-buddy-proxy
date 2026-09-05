# CodeBuddy Proxy

> 一个轻量级本地代理，把 CodeBuddy 底层的聊天接口转换成标准的 **OpenAI Chat Completions**、**Responses** 和 **Anthropic Messages** 协议——让你能把 CodeBuddy 模型接到 Codex CLI、Claude Code / CC Switch、OpenCode、Grok、Oh My Pi 以及任意 OpenAI 兼容客户端上。

> **English docs: [README.md](README.md).**

---

## 特性

- **协议转换** — `/v1/chat/completions`(OpenAI)、`/v1/responses`(Codex CLI)、`/v1/messages`(Anthropic / Claude Code)
- **管理界面** — 内置 Web 控制台 `/ui`：按 provider 分组管理模型、一键「设为默认启用模型」、每个模型一键测试（发条 hi）、按模型维度聚合请求统计图表
- **模型列表** — `/v1/models` 返回 OpenAI 兼容的模型列表，附带每个模型的完整元数据（上下文窗口、能力）
- **脱敏**（`--desensitize`）— 向 system 消息里的合规关键词插入零宽空格，避免后端关键词审核误拦
- **消息压缩**（`--optimize-context`）— 压缩长历史 / 大 schema / 超大工具输出，大幅降低 token 消耗
- **工具调用** — 完整的 function calling 支持，自动过滤无效工具定义
- **DSML 解析** — 自动识别并转换 DeepSeek Markup Language 工具调用
- **流式输出** — SSE 实时返回，带空闲 / 总时长双重超时保护
- **多账号** — 隔离的 session 文件，方便工作 / 个人账号切换
- **多 Provider** — 除 CodeBuddy 外，内置豆包（纯 stdlib CDP 直连豆包工作 App），统一经 `/v1/models` 列出、按模型名路由

---

## 安装与启动

代理是 `src/` 下的一个普通 Python 包，用 [uv](https://docs.astral.sh/uv/) 从源码运行：

```bash
uv sync
uv run python -m buddy_proxy --desensitize
```

首次使用需要登录（会打开浏览器）：

```bash
uv run python -m buddy_proxy --login --desensitize
```

默认监听 `http://127.0.0.1:8787`，管理界面在 **http://127.0.0.1:8787/ui**。

### `buddy` 命令（推荐）

`buddy` 是日常使用的入口：一条命令启动 + 自动打开管理页，也可以把代理注册成 macOS 系统服务（launchd，登录自启 + 崩溃自动拉起）。

```bash
./buddy start              # 启动（未运行时）并打开 http://127.0.0.1:8787/ui
./buddy stop / restart / status / logs
./buddy login [provider]   # 登录上游账号（codebuddy(=workbuddy)/trae/zcode/doubao）
./buddy ui                 # 仅打开管理页（必要时先启动）

# 一次性安装：把 buddy 放进 PATH，之后任意目录敲 buddy 即可
./buddy install            # 装到 /usr/local/bin（不可写时退回 ~/.local/bin）

# 注册为系统服务（launchd）：登录自启、崩溃自动拉起
./buddy service install    # 之后 start/stop/restart 自动走 launchctl
./buddy service status
./buddy service uninstall
```

- 有系统服务时 `buddy start/stop/restart` 自动转 `launchctl`，否则走 `proxy.sh` 的 pid 管理
- 环境变量 `PROXY_HOST`（默认 0.0.0.0）、`PROXY_PORT`（默认 8787）、`PROXY_EXTRA_ARGS` 对 `start` 与 `service install` 生效

### 豆包 Provider（可选）

除 CodeBuddy 外，内置豆包 provider（纯 stdlib CDP 直连本地「豆包工作」App，零额外依赖）：

```bash
# 启用豆包 provider（需要本机已安装并登录豆包工作 App）
uv run python -m buddy_proxy --desensitize --doubao
```

- **原理**：复用豆包工作 App 的登录态与内置 Chromium（CDP 直连），在页面 JS 环境 fetch
  自动注入 a_bogus 风控签名，无需扫码、无需额外凭证
- **模型**：
  - 经典通道：`doubao`（默认模型快速）、`doubao-pro`（旧别名）、`doubao-think`（深度思考）、
    `doubao-expert`（专家）——服务端固定路由到当前默认豆包模型
  - agent 通道（App 模型菜单同款，2026-09 实测）：`doubao-auto`（App「自动」）、
    `doubao-2.1-turbo`、`doubao-2.1-pro`、`orange-5.0`、`gemini-3.7-flash`、`gpt-5.6-sol`；
    请求体可带 `reasoning_effort`（3低/4中/5高/6极高/7最高，默认 5）
- **依赖**：纯 Python 标准库，不需要 Playwright / chromium
- **CDP 模式**：默认优先复用主 App（`open -a DoubaoWork --args --remote-debugging-port=9223`，
  不杀用户进程）；主 App 不可用时回退独立 Helper

### Trae Provider（可选）

除 CodeBuddy 和豆包外，内置 Trae provider（解密 Trae IDE 登录态，直连底层模型）：

```bash
# 启用 Trae provider（需要本机已安装并登录 Trae IDE）
uv run python -m buddy_proxy --desensitize --trae
```

- **原理**：自动解密 Trae IDE 本地存储的 tc 加密登录态（AES-128-CBC + SHA-512），
  或从 `.env` 读 `TRAE_TOKEN` / `TRAE_USER_ID`，直连 `trae-api-cn.mchost.guru`
- **模型**：T1-T5 分级（glm-5.2 / qwen-3.7-plus / kimi-k2.6 / DeepSeek-V4-Pro 等），
  支持外部名别名（如 `claude-sonnet-4-5` → `glm-5.2`）
- **原生通道（2026-09 起）**：全部请求（含纯聊天）默认走 `chat_v3` 直通——
  带 `tools` 时为原生 function calling（结构化 `tool_calls` + `role:"tool"` 历史回放），
  纯聊天无服务端 agent 预设、不再注入压制指令与泄漏清洗；实测 17 个模型
  16 个原生可用（仅 glm-5-turbo 不在通道，自动回落文本协议）；通道还带真实 token usage
- **依赖**：纯 Python 标准库（含零依赖 AES 兜底实现），不需要 Node.js
- **注意**：免费账号有日/周调用额度，耗尽时报 `4011`（今日用量已达上限），
  错误会以友好中文文案透传

#### Trae 账号工具（`trae-cli`）

安装后（`uv sync` / `pip install .`）自带 `trae-cli` 命令，可查询/领取签到积分、查看权益、测试对话：

```bash
uv run trae-cli status            # 签到/积分状态（剩余积分、今日是否已签到）
uv run trae-cli claim             # 领取今日签到积分
uv run trae-cli usage             # 权益/用量（总额、已用比例、权益包列表）
uv run trae-cli chat -m glm-5.2 -q "你好"    # 发一条对话测试
```

认证自动加载：优先 Work 凭证 `~/.ethan/trae_work.json`（`trae_work_login.py` 登录生成），
其次解密本机 Trae IDE `storage.json`，无需手动配置 token。

### 用 `proxy.sh` 后台管理

`buddy` 内部即调用 `proxy.sh`；想手动精细控制时可以直接用它：

```bash
./proxy.sh start          # 后台启动，立刻返回
./proxy.sh stop           # 停止
./proxy.sh restart        # 重启
./proxy.sh status         # 显示 PID 和监听地址
./proxy.sh logs           # tail -F 日志
./proxy.sh ui             # 确保在跑并打开管理页

# 自定义 host / port / 额外参数
./proxy.sh start -p 9000 -H 0.0.0.0
PROXY_PORT=9000 PROXY_EXTRA_ARGS="--desensitize --optimize-context" ./proxy.sh start
```

脚本行为：
- 自动检测 `.venv/bin/python`（优先使用项目 venv）
- 用 `nohup ... &` 启动，`start` 命令**立即返回**，不会阻塞终端
- PID 写到 `logs/proxy.pid`，启动输出写到 `logs/proxy.sh.log`；应用日志按天滚动（`logs/proxy.log` 与 `logs/buddy-proxy.jsonl`，保留 30 天）
- 停止用 `kill` 优雅退出；10s 内未退出会 fallback 到 `kill -9`

## 模型列表

模型目录由 `src/buddy_proxy/models_config.json` 维护（启动时与 `/v1/models` 都从这里读取，离线可靠）。当前内置 **11 个模型**，`GET /v1/models` 的 `data[].credits` / `models[].credits` 会返回积分倍率（消费 × 倍率）：

| id | name | credits |
|---|---|---|
| `auto` | Auto（快速 / 均衡 / 极致 → 0.21 / 0.65 / 1.20） | 动态 |
| `default` | Default | x2.20 |
| `glm-5.3` | GLM-5.3 | x0.79 |
| `glm-5.3-flash` | GLM-5.3-Flash | x0.06 |
| `hy3` | Hy3（限时免费） | x0.00 |
| `hy4-preview` | Hy4 preview | x0.29 |
| `minimax-m3` | MiniMax-M3 | x0.25 |
| `kimi-k3` | Kimi-K3 | x1.62 |
| `kimi-k2.7` | Kimi-K2.7-Code | x0.57 |
| `deepseek-v4-flash` | Deepseek-V4-Flash | x0.17 |
| `deepseek-v4-pro` | Deepseek-V4-Pro | x0.51 |

要新增 / 调整模型，直接编辑 `src/buddy_proxy/models_config.json` 后重启即生效。

## 管理界面（/ui）

浏览器打开 <http://127.0.0.1:8787/ui>（或 `buddy start` / `buddy ui` 自动打开）：

- **默认启用模型** — 按 provider 分组浏览所有模型，点「设为默认」即可把某个模型设为默认；
  客户端请求**不带 `model` 字段**时自动用它补齐。设置持久化在 `~/.buddy-proxy/settings.json`
  （可用 `BUDDY_PROXY_SETTINGS` 覆盖），重启后仍生效；启动时 `--default-model zcode/glm-5.3`
  可提供初始值（设置文件里已有的值优先）
- **一键测试** — 每个模型一个「测试」按钮，向上游发一条 `hi`，弹窗里返回延迟、token 用量和回复预览
  （非流式、`max_tokens=256`，走真实上游、会产生真实调用）
- **自动打卡 & 打卡日历** — Trae 与 CodeBuddy 上游都提供每日签到：勾选「自动打卡」后代理每天定时
  （默认 09:30，启动时当天未签会立即补签）自动领取，最近 35 天的打卡情况在日历里展示；也可点
  「立即打卡」手动领。签到活动有档期——CodeBuddy 档期未开时界面显示「今日无签到活动」且不会误打。
  打卡历史逐行落在 `logs/checkin.jsonl`。ZCode（智谱 Coding Plan）/ 豆包没有签到 API
- **额度查询** — 各通道剩余额度一目了然：CodeBuddy 积分包余额合计 + 各资源包明细（credits）；
  Trae 总额度剩余 + 权益包到期时间；ZCode 的 5 小时 / 每周用量窗口与重置时间。额度数据带
  5 分钟缓存，避免频繁请求上游
- **统计图表** — 按 provider/模型维度聚合请求数、错误数、平均耗时、token 用量：
  近 14 天按通道堆叠的柱状图、模型请求 Top 榜、最近 50 条请求明细。
  每次请求完成追加一行到 `logs/metrics.jsonl`，重启后自动回读恢复历史（保留 30 天）
- **通道健康** — 各 provider 的登录/配置状态一目了然（CodeBuddy 是否登录、zcode key 是否配置等）

管理页里还可以切换**兜底通道**（请求未命中任何模型时的默认路由），设置持久化在同一文件。

安全约定：`/ui/api/*` 管理接口**仅允许本机（127.0.0.1）访问**；确需从局域网操作管理页时设置
`BUDDY_PROXY_ADMIN_OPEN=1` 放开（自担风险）。`/v1/*` 代理端点不受此限制。

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

### 其它 OpenAI 兼容客户端

- Base URL：`http://127.0.0.1:8787/v1`
- API Key：留空（或启动时设置的 `--api-key`）
- Model：`/v1/models` 里的任意 id，如 `glm-5.3`、`deepseek-v4-pro`、`kimi-k2.7`、`auto`

---

## 命令行参数

```
--host HOST               绑定地址（默认 127.0.0.1）
--port PORT               绑定端口（默认 8787）
--endpoint ENDPOINT       CodeBuddy 后端地址
--session-file PATH       会话文件（默认 ~/.codebuddy-session.json）
--log-file PATH           JSONL 日志（默认 <项目根>/logs/buddy-proxy.jsonl，可用 BUDDY_PROXY_LOG_FILE 覆盖）
--desensitize             启用脱敏（推荐）
--optimize-context        启用消息压缩（Codex 场景推荐）
--default-model MODEL     默认启用模型（如 zcode/glm-5.3）；请求未带 model 时使用，
                          首次启动写入设置文件，此后以 ~/.buddy-proxy/settings.json 为准
--login                   启动时浏览器登录
--no-browser              登录时不自动打开浏览器
--verbose-llm             记录完整请求/响应体
--mock-dir DIR            使用录制的响应（测试用）
```

环境变量：`BUDDY_PROXY_HOST`、`BUDDY_PROXY_PORT`、`CODEBUDDY_ENDPOINT`、`BUDDY_PROXY_LOG_FILE`、`BUDDY_PROXY_SETTINGS`（设置文件路径）、`BUDDY_PROXY_ADMIN_OPEN=1`（放开管理接口的本机限制）。

## API 端点

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET  | `/ui`                 | 管理界面（`/` 302 跳转到 `/ui`） |
| GET  | `/ui/api/*`           | 管理接口（overview/models/stats/benefits/settings/checkin/test，仅限本机） |
| GET  | `/health`             | 服务 + 认证状态 |
| GET  | `/v1/models`          | 模型列表 |
| POST | `/v1/chat/completions` | OpenAI 对话（工具 + 流式） |
| POST | `/v1/responses`       | Responses API（Codex CLI） |
| POST | `/v1/messages`        | Anthropic Messages（Claude Code） |

端点使用本地 session 认证，无需额外 token。

---

## Provider 接口一览

三种 provider 统一注册到 `providers.py` 的抽象层，`forward_chat` 按模型名路由。

### 1. CodeBuddy Provider（`codebuddy_provider.py`）

| 接口 | 说明 |
| --- | --- |
| `GET /v1/models` | 模型列表（`models_config.json` 本地配置） |
| `POST /v1/chat/completions` | OpenAI 对话（`stream_upstream` 流式 / `collect_upstream` 非流式） |
| `POST /v1/responses` | Codex CLI Responses 协议适配 |
| `POST /v1/messages` | Claude Code Anthropic Messages 协议适配 |
| `forward_chat(body, "openai"/"codex"/"anthropic")` | 按协议转换 + 按模型路由到对应 provider |
| session / 多账号 | 隔离的 session 文件，`--session-file` 指定 |

### 2. 豆包 Provider（`doubao_provider.py` + `doubao/cdp_client.py`）

| 接口 | 说明 |
| --- | --- |
| `CDPDoubaoClient.start()` | 确保 CDP：优先复用主 App（`open -a DoubaoWork --args --remote-debugging-port=9223`，不杀进程），兜底独立 Helper |
| `CDPDoubaoClient.chat_completion()` | 页面内 fetch `/chat/completion`（自动带 a_bogus 签名），流式 yield SSE；`model_spec` 传入时走 agent 管线（`_build_agent_payload`，App 同款 `model_item_key` 路由），否则经典管线 |
| `DoubaoProvider.forward()` | 流式 / 非流式转换，返回 OpenAI 标准格式 |
| 模型 | 经典：`doubao`（快速）/ `doubao-pro`（旧别名）/ `doubao-think`（深度思考）/ `doubao-expert`（专家）；agent（App 菜单同款）：`doubao-auto` / `doubao-2.1-turbo` / `doubao-2.1-pro` / `orange-5.0` / `gemini-3.7-flash` / `gpt-5.6-sol` |
| 依赖 | 纯 Python 标准库，无 Playwright / chromium |

### 3. Trae Provider（`trae/` 子包 + `trae_work_login*.py`）

实现已从单体 `trae_provider.py`（3400+ 行）拆分为 `trae/` 子包，按职责分 12 个模块：
`config`（常量/版本头/模型映射）、`auth_storage`（IDE 存储解密）、`benefits_api`（签到/权益）、
`credentials`（凭证与请求头）、`leak_guard`（预设泄漏防护）、`text_toolcall`（文本协议解析兜底）、
`text_protocol`（教学注入/请求改写）、`native_tools`（原生 function calling）、`transport`（HTTP 发送）、
`sse`（SSE 解析/Anthropic 包装）、`provider`（编排入口）、`cli`（账号工具）。
`trae_provider.py` 保留为兼容 shim，历史导入与 `python -m buddy_proxy.trae_provider` 不受影响。

| 类别 | 接口 | 说明 |
| --- | --- | --- |
| 认证 | `auth_storage.decrypt_auth_data()` / `find_auth_data()` | 解密 Trae IDE 本地 `storage.json`（AES-128-CBC + SHA-512 派生） |
| 认证 | `trae_work_login.py` / `trae_work_login_server.py` | Work 登录（ExchangeToken 换 token，回调自动捕获，落盘 `~/.ethan/trae_work.json`） |
| 认证 | `credentials._auth()` / `_load_work_cred()` | 凭证加载：`.env` → Work 凭证 → 本地解密 |
| Chat（原生） | `native_tools._send_native_chat()` | **默认通道**：`function=chat_v3` 直通 + 原生 tools（`parameters` 需 JSON 字符串），结构化 `tool_calls` 与 `role:"tool"` 历史回放，带真实 usage；上游按 `X-Ide-Version-Code` 逐模型门控（`WB_TRAE_IDE_VERSION_CODE` 可覆盖） |
| Chat（兜底） | `transport.send_trae_chat()` / `_send_trae_work_chat()` | 文本协议通道（IDE 3 级端点回退 / Work `solo_work_lite`）；原生通道 4001 拒绝时自动回落（`WB_TRAE_NATIVE_TOOLS=0` 可强制全走此路） |
| 解析 | `text_toolcall._parse_tool_calls()` / `_StreamToolCallSplitter()` | 文本协议工具调用解析（教学格式 + 泄漏闸门），仅兜底路径使用 |
| 签到 | `benefits_api.fetch_checkin_status()` / `claim_checkin_credits()` | 查询/领取签到积分（`/trae/api/v2/ug/checkin_credits/*`） |
| 权益 | `benefits_api.fetch_ent_usage()` | 查询积分总额 / 已用量 / 权益包 |
| 模型 | `config._map_model()` | T1-T5 分级 + 外部名别名 |
| 错误 | `sse._trae_error_text()` | 14+ 个官方错误码 → 中文文案（4011 今日额度 / 1005 plan 权益不足等） |

## 免责声明

本项目仅供学习与研究使用，请遵守 CodeBuddy 的服务条款，使用风险自负。
