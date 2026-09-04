"""2026-09-05 实测泄漏 badcase 回归（三个真实 payload 原文内联）。

来源（~/.ethan/db/sessions.db）：
- msg 8595 / s_20260905_0141_3a19 / glm-5.3-flash：裸 JSON 调用被上游断流
  截断，最外层 } 缺失，修复链全部跳过 → 4.8KB 坏 JSON 整段泄漏进正文。
  修复：_repair_call_json 截断修复（补引号/括号 + 尾部回退）。
- msg 8567 / s_20260804_0200_eb1d / glm-5.3-flash：command 里嵌 python 脚本，
  markdown 转义 \$ 是 JSON 非法转义（Invalid \escape），引号修复管不了。
  修复：_repair_invalid_escapes（\X -> \\X，解码后无损还原）。
- msg 8611 / s_20260902_2130_c8bd / deepseek-v4-flash：外层大括号全省，
  name: "shell" 与 arguments: {...} 散装两行，任何既有模式都不匹配。
  修复：散装 kv 兜底（_LOOSE_KV_RE 解析 + _loose_kv_pos 流式扣留）。
- msg 8628 / s_20260905_0233_f291 / doubao-seed-evolving：<seed:tool_call>
  <function name="..."><parameter name="..." string="true">值</parameter>
  </function></seed:tool_call> seed 协议块，任何既有模式都不匹配。
  修复：seed 块 XML 提取（解析 + _TOOL_STREAM_TAGS 扣留 + 聊天清洗）。
"""
import json
import sys

sys.path.insert(0, "src")
from buddy_proxy.trae_provider import (  # noqa: E402
    _parse_tool_calls,
    _StreamToolCallSplitter,
)

KNOWN = frozenset({"shell", "file_read", "file_write", "web_search", "recall_memory"})

fails = []


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


# ---------------- badcase 1：截断裸 JSON（glm-5.3-flash） ----------------
P1 = 'M-5-Turbo（从你 zcode 配置里看到的）+ static/远程均可\n\n先写 provider 主体：\n\n\n\n{"name": "file_write", "arguments": {"content": "\\"\\"\\"ZCode / BigModel (GLM) provider —— Anthropic 兼容直通转发。\\n\\n上游是智谱 Z.ai / BigModel 的 coding-plan Anthropic 兼容端点：\\n    - BigModel（国内）: https://open.bigmodel.cn/api/anthropic\\n    - Z.AI（国际）:     https://api.z.ai/api/anthropic\\n\\n认证走 ZCode CLI 同款 OAuth 流程（参考 router-for-me/CLIProxyAPI PR #3928）：\\n    1. BigModel 分支：本地起 loopback callback server → 浏览器打开\\n       bigmodel.cn/login?redirect=http://127.0.0.1:<port>/callback&appId=zcode\\n       → 登录后回调带 authCode → POST zcode.z.ai/api/v1/oauth/token 换\\n       OAuth access_token\\n    2. 用 OAuth token 调 biz API（getCustomerInfo → 找 org/project →\\n       find-or-create 名为 zcode-api-key 的 API key → copy secretKey）\\n       铸造出 \\"<apiKey>.<secretKey>\\" 形式的标准 coding-plan API key\\n    3. 之后请求带 x-api-key: <apiKey>.<secretKey> 打标准 Anthropic 端点，\\n       无需 captcha（ZCode 专用 zcode-plan 端点才有人机校验）\\n\\n也支持手动路径：直接配置 \\"<apiKey>.<secretKey>\\" API key（在\\n~/.ethan/.secrets/zcode_api_key 或环境变量 ZCODE_API_KEY）。\\n\\n协议侧：上游说 Anthropic Messages 协议，因此 /v1/messages 请求**直通转发**\\n（model 名透传），OpenAI/responses 协议的请求经现有 anthropic_adapter 反向\\n转换后同样直通。\\n\\"\\"\\"\\n\\nfrom __future__ import annotations\\n\\nimport asyncio\\nimport json\\nimport logging\\nimport os\\nimport secrets as pysecrets\\nimport threading\\nimport time\\nimport urllib.parse\\nimport urllib.request\\nimport urllib.error\\nimport webbrowser\\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\\nfrom pathlib import Path\\nfrom typing import Any, AsyncIterator, Sequence\\n\\nimport httpx\\nfrom fastapi import HTTPException\\nfrom fastapi.responses import JSONResponse, StreamingResponse\\n\\nfrom .providers import BaseProvider\\n\\nlog = logging.getLogger(__name__)\\n\\n# ---------------------------------------------------------------- 常量\\n\\nZCODE_OAUTH_BASE = \\"https://zcode.z.ai/api/v1\\"\\nBIGMODEL_BIZ_HOST = \\"https://bigmodel.cn\\"\\nZAI_BIZ_HOST = \\"https://api.z.ai\\"\\nBIGMODEL_API_BASE = \\"https://open.bigmodel.cn/api/anthropic\\"\\nZAI_API_BASE = \\"https://api.z.ai/api/anthropic\\"\\n\\nBIGMODEL_LOGIN_URL = \\"https://bigmodel.cn/login\\"\\nBIGMODEL_APP_ID = \\"zcode\\"\\n\\nMINT_KEY_NAME = \\"zcode-api-key\\"\\nDEFAULT_ORG_NAME = \\"默认机构\\"\\nDEFAULT_PROJECT_NAME = \\"默认项目\\"\\n\\n# 默认模型表（与官方 ZCode 客户端内置一致，zcode config.json 中可见）\\nDEFAULT_MODELS: dict[str, str] = {\\n    \\"glm-5.3\\": \\"GLM-5.3 (thinking, 1M ctx)\\",\\n    \\"glm-5.3-flash\\": \\"GLM-5.3-Flash (thinking, multimodal, 1M ctx)\\",\\n    \\"glm-5-turbo\\": \\"GLM-5-Turbo (200K ctx)\\",\\n}\\n\\n# ---------------------------------------------------------------- 凭证存取\\n\\n\\ndef _cred_path() -> Path:\\n    \\"\\"\\"凭证落盘路径（与 trae provider 风格一致：项目 logs 下）。\\"\\"\\"\\n    return Path(__file__).resolve().parents[2] / \\"logs\\" / \\"zcode-credentials.json\\"\\n\\n\\ndef _load_stored_creds() -> dict[str, Any] | None:\\n    try:\\n        return json.loads(_cred_path().read_text())\\n    except Exception:\\n        return None\\n\\n\\ndef _save_stored_creds(data: dict[str, Any]) -> None:\\n    path = _cred_path()<path> / \\"logs\\" / \\"zcode-credentials.json\\"\\n    path.parent.mkdir(parents=True, exist_ok=True)\\n    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))\\n    try:\\n        path.chmod(0o600)\\n    except Exception:\\n        pass\\n\\n\\n# ---------------------------------------------------------------- OAuth 铸 key\\n\\n\\nclass ZCodeOAuthError(RuntimeError):\\n    pass\\n\\n\\ndef _biz_request(\\n    method: str,\\n    url: str,\\n    authorization: str,\\n    body: dict | list | None = None,\\n    timeout: float = 30.0,\\n) -> Any:\\n    \\"\\"\\"调 biz API 并解 {code,msg,data} 信封。信封 code=0 视为成功。\\"\\"\\"\\n    headers = {\\n        \\"Authorization\\": authorization,\\n        \\"Accept\\": \\"application/json\\",\\n        \\"Content-Type\\": \\"application/json\\",\\n    }\\n    data = json.dumps(body).encode() if body is not None else None\\n    req = urllib.request.Request(url, data=data, method=method, headers=headers)\\n    try:\\n        with urllib.request.urlopen(req, timeout=timeout) as r:\\n            payload = json.loads(r.read())\\n    except urllib.error.HTTPError as e:\\n        detail = e.read()[:200].decode(errors=\\"replace\\")\\n        raise ZCodeOAuthError(f\\"biz API {method} {url} -> HTTP {e.code}: {detail}\\") from e\\n    # 信封式响应 {code, msg, data}\\n    if isinstance(payload, dict) and \\"code\\" in payload and \\"data\\" in payload:\\n        if payload.get(\\"code\\") == 0:\\n            return payload.get(\\"data\\")\\n        raise ZCodeOAuthError(f\\"biz API {method} {url} -> code={payload.get(\'code\')} msg={payload.get(\'msg\')}\\")\\n    return payload\\n\\n\\ndef _oauth_token_exchange(code: str, redirect_uri: string, state: str) -> dict:\\n    \\"\\"\\"BigModel 分支：authCode 换 OAuth token（zcode oauth/token 端点）。\\"\\"\\"\\n    body = {\\n        \\"provider\\": \\"bigmodel\\",\\n    }\\n    raise NotImplementedError\\n", "path": "/Users/jsongo/code/life/w-buddy-proxy/src/buddy_proxy/zcode_provider.py"}'

rest, calls = _parse_tool_calls(P1, KNOWN)
check("b1截断: 打捞出调用", len(calls) == 1 and calls[0]["function"]["name"] == "file_write",
      f"n={len(calls)}")
if calls:
    a1 = json.loads(calls[0]["function"]["arguments"])
    check("b1截断: path 参数完整", a1.get("path") == "/Users/jsongo/code/life/w-buddy-proxy/src/buddy_proxy/zcode_provider.py", f"path={a1.get('path')!r}")
    check("b1截断: content 主体保住", "ZCode / BigModel (GLM) provider" in a1.get("content", "") and "DEFAULT_MODELS" in a1.get("content", ""))
    check("b1截断: 前导正文保留", "先写 provider 主体" in rest, f"rest_head={rest[:60]!r}")
check("b1截断: 无泄漏", '{"name"' not in rest, f"rest_len={len(rest)}")

# 流式（任意切点扣留后 flush 打捞）
for size in (7, 33, 128):
    sp = _StreamToolCallSplitter(KNOWN)
    out = []
    for k in range(0, len(P1), size):
        out.append(sp.feed(P1[k:k + size]))
    tail, calls = sp.flush()
    leaked = '{"name"' in "".join(out) + tail
    check(f"b1截断: 流式切{size}无泄漏+打捞", not leaked and len(calls) == 1,
          f"n={len(calls)} leak={leaked}")

# ---------------- badcase 2：非法 \$ 转义（glm-5.3-flash） ----------------
P2 = 'user 身份无权限（之前几天也一样），按惯例切换 bot 身份重发。\n\n\n{"name": "shell", "arguments": {"command": "python3 -c \\"\\nimport json, subprocess\\ncontent = json.dumps({\'zh_cn\': {\'title\': \'9.5 数据同步\', \'content\': [[{\'tag\': \'md\', \'text\': \'\'\'**9.5 数据同步**\\n- credits 当前水位：**7477.1 / 21,000 (35.6%)**\\n- 共 21 个账号全部正常\\n- **4台机子** RAM=24% & 87% & 82% & 64%，Disk=82% & 35% & 42% & 35%\\n- 今日消耗：\\$**0**（客户 \\$0，团队内 \\$0）\\n- SLA：0.0% (0/0)，客户体感 **100.0**% (0/0)\\n- 异常分析：今天服务运行非常稳定，没有任何异常\\n\\n> 注：现在是 9.5 凌晨 00:00，当天数据刚开始统计，消耗和 SLA 为 0 属正常（credits 水位 7477.1，较昨日 7720.8 略降约 3%）。\'\'\'}}]]}}, ensure_ascii=False)\\nr = subprocess.run([\'lark-cli\', \'im\', \'+messages-send\', \'--identity\', \'bot\', \'--chat-id\', \'oc_70a8f3ab325ee9236b54a0269a88e111\', \'--msg-type\', \'post\', \'--content\', content], capture_output=True, text=True)\\nprint(\'STDOUT:\', r.stdout)\\nprint(\'STDERR:\', r.stderr)\\nprint(\'RC:\', r.returncode)\\n\\"", "intent": "bot身份post推送9.5数据", "timeout": 60}}'

rest, calls = _parse_tool_calls(P2, KNOWN)
check("b2转义: 打捞出调用", len(calls) == 1 and calls[0]["function"]["name"] == "shell",
      f"n={len(calls)} rest={rest[:80]!r}")
if calls:
    a2 = json.loads(calls[0]["function"]["arguments"])
    cmd = a2.get("command", "")
    check("b2转义: 脚本头完整", cmd.startswith("python3 -c"))
    check("b2转义: 脚本主体完整", "lark-cli" in cmd and "oc_70a8f3ab325ee9236b54a0269a88e111" in cmd)
    check("b2转义: \$ 无损还原", "\\$" in cmd and "7477.1" in cmd)
    check("b2转义: 其余参数保留", a2.get("intent") == "bot身份post推送9.5数据" and a2.get("timeout") == 60, f"args_keys={sorted(a2)}")
check("b2转义: 无泄漏", "arguments" not in rest, f"rest={rest[:60]!r}")

for size in (7, 33, 128):
    sp = _StreamToolCallSplitter(KNOWN)
    out = []
    for k in range(0, len(P2), size):
        out.append(sp.feed(P2[k:k + size]))
    tail, calls = sp.flush()
    leaked = '{"name"' in "".join(out) + tail
    check(f"b2转义: 流式切{size}无泄漏+打捞", not leaked and len(calls) == 1)

# ---------------- badcase 3：散装 kv（deepseek-v4-flash） ----------------
P3 = '9月4日完全没产出。让我查一下当天的调度日志和服务状态。\n\n\n\n\nname: "shell"\narguments: {"command": "find ~/.ethan -name \'*.log\' -newer ~/.ethan/output/eigenflux/2026-09-03-evening 2>/dev/null | head -10; echo \'===\'; ls ~/.ethan/logs/ 2>/dev/null | head -20; echo \'===\'; cat ~/.ethan/system/tools.md | grep -A5 \'eigenflux\' | tail -20", "intent": "查调度日志和EigenFlux状态"}'

rest, calls = _parse_tool_calls(P3, KNOWN)
check("b3散装: 打捞出调用", len(calls) == 1 and calls[0]["function"]["name"] == "shell",
      f"n={len(calls)}")
if calls:
    a3 = json.loads(calls[0]["function"]["arguments"])
    check("b3散装: command 完整", a3.get("command", "").startswith("find ~/.ethan") and "tools.md" in a3.get("command", ""))
    check("b3散装: intent 保留", a3.get("intent") == "查调度日志和EigenFlux状态")
    check("b3散装: 前导正文保留", "9月4日完全没产出" in rest and "name:" not in rest, f"rest={rest[:60]!r}")

for size in (7, 33, 128):
    sp = _StreamToolCallSplitter(KNOWN)
    out = []
    for k in range(0, len(P3), size):
        out.append(sp.feed(P3[k:k + size]))
    tail, calls = sp.flush()
    combined = "".join(out) + tail
    leaked = 'name: "shell"' in combined or '"command"' in combined
    check(f"b3散装: 流式切{size}无泄漏+打捞", not leaked and len(calls) == 1,
          f"n={len(calls)} leak={leaked} out_head={combined[:40]!r}")


# ---------------- badcase 4：seed 协议块（doubao-seed-evolving） ----------------
P4 = 'gh 认证正常。开始深度 review，先拉 PR 元信息和文件清单。<seed:tool_call>\n<function name="shell"><parameter name="command" string="true">gh api repos/llm011/ethan-agent/pulls/305 --jq \'{title: .title, body: .body, head_sha: .head.sha, base: .base.ref, additions: .additions, deletions: .deletions, changed_files: .changed_files, state: .state, draft: .draft}\' > /tmp/pr_305_meta.json && cat /tmp/pr_305_meta.json && gh api repos/llm011/ethan-agent/pulls/305 --jq \'.head.sha\' > /tmp/pr_305_sha.txt && echo "SHA: $(cat /tmp/pr_305_sha.txt)"</parameter><parameter name="intent" string="true">拉PR元信息和head sha</parameter></function>\n<function name="shell"><parameter name="command" string="true">gh api repos/llm011/ethan-agent/pulls/305/files --paginate > /tmp/pr_305_files.json && jq -r \'.[] | "\\(.status)\\t\\(.additions)+\\(.deletions)-\\t\\(.filename)"\' /tmp/pr_305_files.json</parameter><parameter name="intent" string="true">拉PR文件列表</parameter></function>\n</seed:tool_call>'

rest, calls = _parse_tool_calls(P4, KNOWN)
check("b4seed: 解析2个调用", len(calls) == 2
      and all(c["function"]["name"] == "shell" for c in calls),
      f"n={len(calls)}")
if len(calls) == 2:
    a40 = json.loads(calls[0]["function"]["arguments"])
    a41 = json.loads(calls[1]["function"]["arguments"])
    check("b4seed: command1 完整", a40.get("command", "").startswith("gh api repos/llm011/ethan-agent/pulls/305")
          and "pr_305_meta.json" in a40.get("command", ""))
    check("b4seed: intent1 保留", a40.get("intent") == "拉PR元信息和head sha")
    check("b4seed: command2 完整", "--paginate" in a41.get("command", "")
          and a41.get("command", "").endswith("/tmp/pr_305_files.json"))
    check("b4seed: intent2 保留", a41.get("intent") == "拉PR文件列表")
check("b4seed: 前导正文保留", "深度 review" in rest and "seed:tool_call" not in rest
      and "<function" not in rest, f"rest={rest[:60]!r}")

for size in (7, 17, 64, 200):
    sp = _StreamToolCallSplitter(KNOWN)
    out = []
    for k in range(0, len(P4), size):
        out.append(sp.feed(P4[k:k + size]))
    tail, calls = sp.flush()
    combined = "".join(out) + tail
    leaked = "seed:tool_call" in combined or "<function" in combined or "<parameter" in combined
    check(f"b4seed: 流式切{size}无泄漏+打捞", not leaked and len(calls) == 2,
          f"n={len(calls)} leak={leaked}")

# 未闭合块（流截断在 </seed:tool_call> 之前）
P4t = P4.replace("</seed:tool_call>", "")
rest, calls = _parse_tool_calls(P4t, KNOWN)
check("b4seed: 截断块打捞", len(calls) == 2 and "seed:tool_call" not in rest,
      f"n={len(calls)} rest={rest[:40]!r}")

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
