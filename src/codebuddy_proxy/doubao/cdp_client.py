"""豆包工作 CDP 客户端 —— 直连本地 DoubaoWork.app 的内置 Chromium。

与 ``BrowserClient``（Playwright，自己开 Chromium 连 doubao.com）不同，
本客户端**复用豆包工作 App 自带的 Helper（Chromium 147）**：

1. 独立启动 Helper（``--saman-from-chat=1 --remote-debugging-port=<port>``），
   从磁盘 profile 自动加载完整登录态，无需扫码、无需主 App。
2. 纯 stdlib WebSocket 直连 CDP（避开 playwright 在沙箱被 SIGTERM 的问题）。
3. 在 ``chrome://doubaowork-chat/chat`` 页面的 JS 环境里 ``fetch``
   ``/chat/completion``，自动带 httpOnly cookie + a_bogus 签名。
4. 流式：JS 逐块读 SSE 并通过 ``console.log`` 回传，Python 监听
   ``Runtime.consoleAPICalled`` 事件实时取回。

关键坑（踩坑总结）：
- 独立 Helper 必须加 ``--disable-features=SpareRendererForSitePerProcess``，
  否则 10 秒后 spare renderer 崩溃连锁拖垮 network service。
- 启动前必须 ``pkill -KILL -f DoubaoWork``，否则单例锁拦截伪 pid 启动。
- CDP 客户端帧必须 masked（RFC6455 规定）。

接口与 ``BrowserClient`` 对齐，供 ``DoubaoProvider`` 无缝切换。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import socket
import struct
import subprocess
import time
import urllib.request
import uuid
from typing import Any, AsyncGenerator, Optional

log = logging.getLogger(__name__)

# 豆包工作 Helper 二进制路径
HELPER_BIN = (
    "/Applications/DoubaoWork.app/Contents/Helpers/"
    "DoubaoWork Browser.app/Contents/MacOS/DoubaoWork Browser"
)

DEFAULT_BOT_ID = "7338286299411103781"
DEFAULT_PORT = 9223

# 一方模型 -> use_deep_think（与 doubao_provider._DOUBAO_CHAT_MODELS 对齐）
# 这里只保留常量，模型表仍由 DoubaoProvider 维护。

_STABLE_FLAGS = [
    "--saman-from-chat=1",
    "--no-first-run",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-software-rasterizer",
    "--disable-dev-shm-usage",
    "--disable-background-networking",
    "--disable-sync",
    "--disable-extensions",
    "--disable-default-apps",
    "--disable-features=SpareRendererForSitePerProcess",
]


class _WS:
    """最小 WebSocket 客户端（纯 stdlib，RFC6455）。"""

    def __init__(self, host: str, port: int, path: str, timeout: float = 15.0):
        self._s = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._s.sendall(req.encode())
        data = b""
        while b"\r\n\r\n" not in data:
            data += self._s.recv(4096)
        self._id = 0
        self._timeout = timeout

    def send(self, payload: bytes | str, opcode: int = 0x1) -> None:
        if isinstance(payload, str):
            payload = payload.encode()
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack(">H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack(">Q", n)
        header += mask
        self._s.sendall(bytes(header) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _read_exact(self, n: int) -> bytes:
        data = b""
        while len(data) < n:
            c = self._s.recv(n - len(data))
            if not c:
                raise EOFError("websocket closed")
            data += c
        return data

    def recv_frame(self) -> tuple[int, bytes]:
        b1 = self._read_exact(2)
        length = b1[1] & 0x7F
        opcode = b1[0] & 0x0F
        masked = (b1[1] & 0x80) != 0
        if length == 126:
            length = struct.unpack(">H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else b""
        payload = self._read_exact(length)
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return opcode, payload

    def cmd(self, method: str, params: dict[str, Any] | None = None,
            timeout: float = 20.0) -> dict[str, Any]:
        self._id += 1
        msg: dict[str, Any] = {"id": self._id, "method": method}
        if params:
            msg["params"] = params
        self._s.settimeout(timeout)
        self.send(json.dumps(msg))
        while True:
            op, payload = self.recv_frame()
            if op == 0x9:  # ping
                self.send(payload, opcode=0xA)  # pong
                continue
            if op == 0x1:  # text
                m = json.loads(payload)
                if m.get("id") == self._id:
                    return m

    def close(self) -> None:
        try:
            self._s.close()
        except Exception:
            pass


class CDPDoubaoClient:
    """直连豆包工作 Helper 的 CDP 客户端。"""

    def __init__(self, port: int = DEFAULT_PORT, helper_bin: str = HELPER_BIN):
        self.port = port
        self.helper_bin = helper_bin
        self._proc: subprocess.Popen | None = None
        self._ws: _WS | None = None
        self._ready = False
        self._web_id: str = ""
        self._consecutive_failures = 0
        self._last_error_code = 0

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error_code(self) -> int:
        return self._last_error_code

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self, error_code: int = 0) -> None:
        self._consecutive_failures += 1
        self._last_error_code = error_code

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动独立 Helper 并连上 CDP。"""
        # 1. 清理残留（否则单例锁拦截）
        subprocess.run(["pkill", "-KILL", "-f", "DoubaoWork"], check=False)
        await asyncio.sleep(2)

        # 2. 启动 Helper
        cmd = [self.helper_bin, f"--remote-debugging-port={self.port}", *_STABLE_FLAGS]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        log.info("CDPDoubaoClient: Helper started pid=%s port=%d", self._proc.pid, self.port)

        # 3. 等 CDP target 就绪
        chat = await self._wait_for_target()
        if not chat:
            raise RuntimeError("CDP target not found")

        # 4. 连接 WS
        ws_url = chat["webSocketDebuggerUrl"]
        path = ws_url.split(f":{self.port}", 1)[1]
        self._ws = _WS("127.0.0.1", self.port, path)

        # 5. 等 bdms 就绪 + 提取 web_id
        await self._wait_for_bdms()
        self._web_id = await self._extract_web_id() or ""
        self._ready = True
        log.info("CDPDoubaoClient: ready (web_id=%s)", self._web_id[:20])

    async def stop(self) -> None:
        if self._ws:
            self._ws.close()
            self._ws = None
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._ready = False

    async def is_alive(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    async def restart(self) -> None:
        await self.stop()
        await self.start()

    # ------------------------------------------------------------------
    # 内部：CDP 辅助
    # ------------------------------------------------------------------

    def _evaluate(self, expr: str, timeout: float = 30.0) -> Any:
        """同步执行 Runtime.evaluate（返回 JS 值或错误标记）。"""
        if not self._ws:
            raise RuntimeError("CDP not connected")
        r = self._ws.cmd(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
            timeout=timeout,
        )
        result = r.get("result", {})
        if "exceptionDetails" in result:
            desc = (
                result["exceptionDetails"].get("exception", {}).get("description")
                or result["exceptionDetails"].get("text")
            )
            return {"ERROR": str(desc)[:500]}
        return result.get("result", {}).get("value")

    async def _wait_for_target(self, timeout: float = 60.0) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ts = json.loads(
                    urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/json/list", timeout=2
                    ).read()
                )
                for t in ts:
                    if t.get("type") == "page" or "chat" in t.get("url", ""):
                        return t
            except Exception:
                pass
            await asyncio.sleep(0.2)
        return None

    async def _wait_for_bdms(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = await asyncio.to_thread(self._evaluate, "typeof window.bdms")
            if r == "object":
                return
            await asyncio.sleep(0.5)
        log.warning("bdms not ready after %ss", timeout)

    async def _extract_web_id(self) -> str:
        expr = (
            "(() => { try { const v = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');"
            " return v.web_id || ''; } catch(e) { return ''; } })()"
        )
        r = await asyncio.to_thread(self._evaluate, expr)
        return r if isinstance(r, str) else ""

    def _build_query_params(self) -> dict[str, str]:
        return {
            "aid": "1044603",
            "device_platform": "web",
            "doubao_device_platform": "desktop",
            "web_id": self._web_id or "",
            "version_code": "20800",
            "language": "zh",
            "real_aid": "1044603",
        }

    # ------------------------------------------------------------------
    # 登录辅助
    # ------------------------------------------------------------------

    async def wait_for_login(self, timeout: int = 120) -> bool:
        """独立 Helper 从磁盘自动加载登录态，通常无需扫码。"""
        if not self._ready:
            await self.start()
        # 检查登录态
        expr = "localStorage.getItem('flow_web_has_login')"
        r = await asyncio.to_thread(self._evaluate, expr)
        return r == "true"

    # ------------------------------------------------------------------
    # 核心：chat_completion（对齐 BrowserClient 接口）
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """发送消息并 yield 解析后的 SSE 事件（与 BrowserClient 格式一致）。"""
        if not self._ready:
            raise RuntimeError("CDP client not ready")

        need_create = conversation_id is None or conversation_id == ""
        effective_bot_id = bot_id or DEFAULT_BOT_ID
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        payload = {
            "client_meta": {
                "local_conversation_id": f"local_{uuid.uuid4().int % 10**16}" if need_create else "",
                "conversation_id": conversation_id or "",
                "bot_id": effective_bot_id,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": [{
                "local_message_id": str(uuid.uuid4()),
                "content_block": [{
                    "block_type": 10000,
                    "content": {
                        "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                        "pc_event_block": "",
                    },
                    "block_id": str(uuid.uuid4()),
                    "parent_id": "",
                    "meta_info": [],
                    "append_fields": [],
                }],
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "",
                "create_time_ms": now_ms,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": False,
                "tts_switch": False,
                "need_deep_think": use_deep_think,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": str(uuid.uuid4()),
                "start_seq": 0,
                "need_create_conversation": need_create,
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "recovery_option": {
                    "is_recovery": False,
                    "req_create_time_sec": now_sec,
                    "append_sse_event_scene": 0,
                },
                "message_storage_type": 0,
            },
            "ext": {
                "use_deep_think": str(use_deep_think),
                "fp": "",
                "collection_id": "",
                "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1" if need_create else "0",
            },
        }

        query = self._build_query_params()
        query_string = "&".join(f"{k}={v}" for k, v in sorted(query.items()))
        url = f"https://www.doubao.com/chat/completion?{query_string}"

        log.info("CDP POST /chat/completion (conv=%s, deep_think=%s)",
                 conversation_id or "new", use_deep_think)

        # 流式：JS 读 SSE 逐块 console.log，Python 监听 consoleAPICalled
        async for event in self._stream_fetch(url, payload):
            yield event

    async def _stream_fetch(
        self, url: str, payload: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """在页面 JS 里 fetch 并流式读取 SSE，逐块回传。

        用「window 队列 + evaluate 轮询」作为桥：
        - JS 在后台 async IIFE 里 fetch，逐块解析 SSE，把事件 push 到
          ``window.__dwChunks`` 数组（元素是 JSON 字符串）。
        - Python 循环 ``Runtime.evaluate`` 读取 ``shift()`` 取走事件。

        不用 console.log 桥——实测该 Helper 不派发 consoleAPICalled 事件。
        关键：``awaitPromise=False`` 启动的 async IIFE 会在后台持续执行
        （已实测验证），fetch 不受 evaluate 返回影响。
        """
        if not self._ws:
            raise RuntimeError("CDP not connected")

        queue_name = f"__dwChunks_{uuid.uuid4().hex[:10]}"
        payload_json = json.dumps(payload, ensure_ascii=False)

        js = f"""
        (async () => {{
          const q = {json.dumps(queue_name)};
          window[q] = [];
          try {{
            const res = await fetch({json.dumps(url)}, {{
              method: 'POST',
              credentials: 'include',
              headers: {{'Content-Type': 'application/json'}},
              body: {json.dumps(payload_json)},
            }});
            window[q].push('__HTTP_STATUS__:' + res.status);
            if (!res.ok) {{
              const errBody = await res.text();
              window[q].push('__HTTP_ERROR__:' + res.status + ':' + errBody.slice(0, 500));
              window[q].push('__DONE__');
              return;
            }}
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let currentEvent = '';
            let buffer = '';
            while (true) {{
              const {{done, value}} = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, {{stream: true}});
              const lines = buffer.split('\\n');
              buffer = lines.pop();
              for (const line of lines) {{
                const t = line.trim();
                if (!t) continue;
                if (t.startsWith('event: ')) {{ currentEvent = t.slice(7); continue; }}
                if (t.startsWith('id: ')) continue;
                if (!t.startsWith('data: ')) continue;
                const ds = t.slice(6);
                if (!ds || ds === '{{}}') continue;
                try {{
                  const obj = JSON.parse(ds);
                  obj._event = currentEvent;
                  window[q].push(JSON.stringify(obj));
                }} catch(e) {{}}
              }}
            }}
            if (buffer.trim()) {{
              const t = buffer.trim();
              if (t.startsWith('data: ')) {{
                const ds = t.slice(6);
                if (ds && ds !== '{{}}') {{
                  try {{
                    const obj = JSON.parse(ds);
                    obj._event = currentEvent;
                    window[q].push(JSON.stringify(obj));
                  }} catch(e) {{}}
                }}
              }}
            }}
            window[q].push('__DONE__');
          }} catch(e) {{
            window[q].push('__ERROR__:' + e.message);
          }}
        }})()
        """

        # 启动后台 JS（awaitPromise=False，后台持续跑）
        self._ws.cmd("Runtime.evaluate", {"expression": js, "awaitPromise": False})

        # 轮询队列：每次取走一批事件
        drain_js = (
            f"(() => {{ const q = window[{json.dumps(queue_name)}];"
            f" if (!q || q.length === 0) return '[]';"
            f" const batch = q.splice(0, q.length);"
            f" return JSON.stringify(batch); }})()"
        )
        deadline = time.time() + 180.0
        while time.time() < deadline:
            await asyncio.sleep(0.15)
            r = await asyncio.to_thread(self._evaluate, drain_js, 10.0)
            if not isinstance(r, str):
                continue
            try:
                batch = json.loads(r)
            except json.JSONDecodeError:
                continue
            for item in batch:
                if item == "__DONE__":
                    return
                if item.startswith("__ERROR__:"):
                    yield {"error": True, "status": 0, "body": item[len("__ERROR__:"):]}
                    return
                if item.startswith("__HTTP_ERROR__:"):
                    rest = item[len("__HTTP_ERROR__:"):]
                    status = int(rest.split(":", 1)[0])
                    body = rest.split(":", 1)[1] if ":" in rest else ""
                    yield {"error": True, "status": status, "body": body}
                    return
                if item.startswith("__HTTP_STATUS__:"):
                    continue
                try:
                    obj = json.loads(item)
                    yield obj
                except json.JSONDecodeError:
                    continue
        # 超时
        yield {"error": True, "status": 0, "body": "Stream timeout (180s)"}

    # ------------------------------------------------------------------
    # SSE 解析辅助（与 BrowserClient 对齐）
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str:
        """从 SSE 事件提取文本（复用 BrowserClient 逻辑）。"""
        event_type = event.get("_event", "")

        if event_type == "CHUNK_DELTA" and isinstance(event.get("text"), str):
            return event["text"]

        if "patch_op" in event:
            for op in event["patch_op"]:
                pv = op.get("patch_value", {})
                for block in pv.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                if op.get("patch_object") == 102:
                    raw = pv.get("content", "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if parsed.get("text"):
                                return parsed["text"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        if event_type == "STREAM_MSG_NOTIFY":
            content = event.get("content", {})
            if isinstance(content, dict):
                for block in content.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]

        return ""

    @staticmethod
    def extract_conversation_id(event: dict[str, Any]) -> Optional[str]:
        ack = event.get("ack_client_meta", {})
        if ack.get("conversation_id"):
            return ack["conversation_id"]
        meta = event.get("meta", {})
        if meta.get("conversation_id"):
            return meta["conversation_id"]
        return None
