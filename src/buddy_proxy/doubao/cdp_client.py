"""豆包工作 CDP 客户端 —— 直连本地 DoubaoWork.app 的内置 Chromium。

与 ``BrowserClient``（Playwright，自己开 Chromium 连 doubao.com）不同，
本客户端**复用豆包工作 App 自带的 Helper（Chromium 147）**：

1. 优先**复用主 App**：探测 ``--remote-debugging-port`` 已开则直接连接；
   未开时用 ``open -a DoubaoWork --args --remote-debugging-port=<port>``
   拉起主 App（不杀用户进程，主 App 自带 ``--saman-from-chat=<主AppPID>``
   开 CDP），用户界面正常可用。
2. 主 App 不可用时才退回**独立 Helper**（``--saman-from-chat=1``），
   从磁盘 profile 自动加载完整登录态，无需扫码。
3. 纯 stdlib WebSocket 直连 CDP（避开 playwright 在沙箱被 SIGTERM 的问题）。
4. 在 ``chrome://doubaowork-chat/chat`` 页面的 JS 环境里 ``fetch``
   ``/chat/completion``，自动带 httpOnly cookie + a_bogus 签名。
5. 流式：JS 后台 fetch 逐块读 SSE push 到 window 队列，Python 轮询取回
   （该 Helper 不派发 ``Runtime.consoleAPICalled`` 事件，不能走 console 桥）。

关键坑（踩坑总结）：
- 独立 Helper 必须加 ``--disable-features=SpareRendererForSitePerProcess``，
  否则 10 秒后 spare renderer 崩溃连锁拖垮 network service。
- 只在**退回独立 Helper** 时才 ``pkill -KILL -f DoubaoWork``（单例锁拦截
  伪 pid 启动）；复用主 App 时绝不杀进程。
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
import threading
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
        # 所有 CDP 命令（含 to_thread 并发调用）都经此锁串行化，
        # 避免多请求同时读写同一 WebSocket 导致 id 串号 / 响应丢失 / 串流。
        self._ws_lock = threading.Lock()

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

    def _probe_cdp(self) -> bool:
        """探测端口上是否已有可用的 CDP（主 App 已开调试端口）。"""
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/json/version", timeout=2
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _connect_chat(self) -> None:
        """连接 chat 页面 target，等 bdms 就绪并提取 web_id。"""
        chat = await self._wait_for_target()
        if not chat:
            raise RuntimeError("CDP target not found")
        ws_url = chat["webSocketDebuggerUrl"]
        path = ws_url.split(f":{self.port}", 1)[1]
        self._ws = _WS("127.0.0.1", self.port, path)

        # 连到的可能只是 Helper 的 chrome:// 占位页（如
        # chrome://doubaowork-chat/cross-site-support/，无 bdms/cookie，
        # 直接发请求会报 710020202 invalid param）—— 主动导航到聊天页。
        href = await asyncio.to_thread(self._evaluate, "location.href")
        if "doubao.com" not in (href or ""):
            log.info("CDPDoubaoClient: target is %s, navigating to chat page", href)
            with self._ws_lock:
                self._ws.cmd("Page.navigate", {"url": "https://www.doubao.com/chat/"})
            # 等导航完成（readyState=complete 且已到豆包域名）
            deadline = time.time() + 30
            while time.time() < deadline:
                href = await asyncio.to_thread(self._evaluate, "location.href")
                rs = await asyncio.to_thread(self._evaluate, "document.readyState")
                if "doubao.com" in (href or "") and rs == "complete":
                    break
                await asyncio.sleep(0.5)

        await self._wait_for_bdms()
        self._web_id = await self._extract_web_id() or ""
        self._ready = True
        log.info("CDPDoubaoClient: ready (web_id=%s)", self._web_id[:20])

    async def start(self) -> None:
        """确保 CDP 可用：优先复用主 App（共存，不杀用户进程），兜底独立 Helper。"""
        # 1) 端口已有 CDP 直接复用（主 App 正在跑且开了调试端口）
        if await asyncio.to_thread(self._probe_cdp):
            log.info("CDPDoubaoClient: reuse existing CDP on port %d", self.port)
            await self._connect_chat()
            return

        # 2) 尝试拉起主 App（不杀进程；主 App 自带 saman-from-chat 开 CDP）
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["open", "-a", "DoubaoWork", "--args", f"--remote-debugging-port={self.port}"],
                capture_output=True,
                timeout=10,
            )
            for _ in range(30):
                await asyncio.sleep(1)
                if await asyncio.to_thread(self._probe_cdp):
                    log.info("CDPDoubaoClient: 主 App CDP ready on port %d", self.port)
                    await self._connect_chat()
                    return
            log.warning("CDPDoubaoClient: 主 App 未开 CDP，回退独立 Helper")
        except Exception as e:
            log.warning("CDPDoubaoClient: 主 App 拉起异常（%s），回退独立 Helper", e)

        # 3) 兜底：独立 Helper。
        # 注意：主 App 正在运行但没开调试口时（open -a 对已运行实例只激活、
        # 不传参），绝不能 pkill 用户正在用的豆包 App —— 明确报错让用户决策。
        main_app_running = (
            await asyncio.to_thread(
                subprocess.run,
                ["pgrep", "-f", "DoubaoWork.app/Contents/MacOS/DoubaoWork"],
                capture_output=True,
            )
        ).returncode == 0
        if main_app_running:
            raise RuntimeError(
                "豆包主 App 正在运行但未开启 CDP 调试端口（无法给已运行的实例"
                "追加启动参数）。请先完全退出豆包（Cmd+Q）后重试；代理不会强杀"
                "正在使用的豆包 App。"
            )
        # 主 App 确认未运行时才清理残留 Helper 进程（否则单例锁拦截）
        await asyncio.to_thread(
            subprocess.run, ["pkill", "-KILL", "-f", "DoubaoWork"], check=False
        )
        await asyncio.sleep(2)
        cmd = [self.helper_bin, f"--remote-debugging-port={self.port}", *_STABLE_FLAGS]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        log.info("CDPDoubaoClient: Helper started pid=%s port=%d", self._proc.pid, self.port)
        await self._connect_chat()

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

    def _evaluate(self, expr: str, timeout: float = 30.0, await_promise: bool = True) -> Any:
        """同步执行 Runtime.evaluate（返回 JS 值或错误标记）。

        经 ``_ws_lock`` 串行化：并发请求（to_thread 调用）不会交叉读写
        同一 WebSocket，避免 CDP 响应 id 串号。
        """
        if not self._ws:
            raise RuntimeError("CDP not connected")
        with self._ws_lock:
            r = self._ws.cmd(
                "Runtime.evaluate",
                {"expression": expr, "returnByValue": True, "awaitPromise": await_promise},
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

    def _fetch_targets(self) -> list[dict[str, Any]]:
        """同步拉取 /json/list（失败返回空列表）。"""
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/json/list", timeout=2
            ) as resp:
                return json.loads(resp.read())
        except Exception:
            return []

    async def _wait_for_target(self, timeout: float = 60.0) -> dict[str, Any] | None:
        """找可用的 chat target。

        优先豆包域名页面（主 App 复用场景下已存在）；独立 Helper 只开
        chrome://doubaowork-chat/... 占位页（URL 含 "chat" 会误匹配旧逻辑），
        等一小段宽限期后返回该占位页，由 _connect_chat 导航到聊天页。
        """
        deadline = time.time() + timeout
        grace = time.time() + 5  # 最多等 5s 让豆包域名页面出现
        fallback: dict[str, Any] | None = None
        while time.time() < deadline:
            for t in await asyncio.to_thread(self._fetch_targets):
                url = t.get("url", "")
                if not t.get("webSocketDebuggerUrl"):
                    continue
                if "doubao.com" in url:
                    return t
                if fallback is None and t.get("type") == "page":
                    fallback = t
            await asyncio.sleep(0.3)
            if fallback is not None and time.time() > grace:
                return fallback
        return fallback

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

        # 启动后台 JS（awaitPromise=False，后台持续跑；经 _ws_lock 串行化，
        # 不阻塞事件循环）
        await asyncio.to_thread(self._evaluate, js, 10.0, False)

        # 轮询队列：每次取走一批事件
        drain_js = (
            f"(() => {{ const q = window[{json.dumps(queue_name)}];"
            f" if (!q || q.length === 0) return '[]';"
            f" const batch = q.splice(0, q.length);"
            f" return JSON.stringify(batch); }})()"
        )
        # 空闲超时：以「最近一次收到数据」计，长回复（>180s 仍在出数据）不会被误杀
        idle_timeout = 180.0
        last_activity = time.time()
        while time.time() - last_activity < idle_timeout:
            await asyncio.sleep(0.15)
            r = await asyncio.to_thread(self._evaluate, drain_js, 10.0)
            if not isinstance(r, str):
                continue
            try:
                batch = json.loads(r)
            except json.JSONDecodeError:
                continue
            if batch:
                last_activity = time.time()
            for item in batch:
                if item == "__DONE__":
                    await self._cleanup_queue(queue_name)
                    return
                if item.startswith("__ERROR__:"):
                    yield {"error": True, "status": 0, "body": item[len("__ERROR__:"):]}
                    await self._cleanup_queue(queue_name)
                    return
                if item.startswith("__HTTP_ERROR__:"):
                    rest = item[len("__HTTP_ERROR__:"):]
                    status = int(rest.split(":", 1)[0])
                    body = rest.split(":", 1)[1] if ":" in rest else ""
                    yield {"error": True, "status": status, "body": body}
                    await self._cleanup_queue(queue_name)
                    return
                if item.startswith("__HTTP_STATUS__:"):
                    continue
                try:
                    obj = json.loads(item)
                    yield obj
                except json.JSONDecodeError:
                    continue
        # 超时
        await self._cleanup_queue(queue_name)
        yield {"error": True, "status": 0, "body": f"Stream idle timeout ({idle_timeout:.0f}s)"}

    async def _cleanup_queue(self, queue_name: str) -> None:
        """删除页面侧轮询队列，避免长会话下浏览器内存无界增长。"""
        try:
            await asyncio.to_thread(
                self._evaluate,
                f"delete window[{json.dumps(queue_name)}]",
                5.0,
            )
        except Exception:
            pass

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
