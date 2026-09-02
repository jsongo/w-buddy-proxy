"""Trae Work 一键登录服务 —— 监听 127.0.0.1:18080 捕获回调，自动完成换 token。

用法：
  1. 先运行本脚本（后台/前台均可，监听 18080）
  2. 打开登录链接（trae_work_login.py 生成的），浏览器登录
  3. 登录成功后浏览器自动跳到 http://127.0.0.1:18080/authorize?...
  4. 本服务捕获回调 → ExchangeToken → GetUserInfo → 落盘 ~/.ethan/trae_work.json
  5. 自动用 Work 通道（solo_work_lite）发一条测试消息验证
"""
from __future__ import annotations

import html
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

CLIENT_ID = "en1oxy7wnw8j9n"
APP_VERSION = "0.1.43"
API_HOST = "https://api.trae.com.cn"
OUT_PATH = Path.home() / ".ethan" / "trae_work.json"
# 与 trae_work_login.py 共享的本次登录状态（nonce + machine_id/device_id）
STATE_PATH = Path("/tmp/trae_work_login_state.json")
STATE_TTL = 900  # 状态有效期（秒）
PORT = 18080


def _load_login_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _write_cred_secure(path: Path, data: dict) -> None:
    """以 0600 权限原子创建凭证文件（避免 write_text 后 chmod 前的短暂 0644 窗口）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def _http_post_json(url: str, body: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def exchange_token(refresh_token: str) -> dict:
    body = {"ClientID": CLIENT_ID, "RefreshToken": refresh_token, "ClientSecret": "-", "UserID": ""}
    resp = _http_post_json(
        API_HOST + "/cloudide/api/v3/trae/oauth/ExchangeToken",
        body,
        {"Content-Type": "application/json", "User-Agent": f"Trae/{APP_VERSION}"},
    )
    result = resp.get("Result") or {}
    token = result.get("Token") or ""
    if not token:
        raise RuntimeError(f"ExchangeToken 失败: {json.dumps(resp, ensure_ascii=False)[:300]}")
    new_refresh = result.get("RefreshToken") or refresh_token
    expires_at = int(result.get("TokenExpireAt") or 0)
    if expires_at > 10**12:
        expires_at //= 1000
    if expires_at <= time.time():
        expires_at = int(time.time()) + int(result.get("TokenExpireDuration") or 1209600)
    return {"access_token": token, "refresh_token": new_refresh, "expires_at": expires_at}


def get_user_info(token: str) -> dict:
    try:
        ui = _http_post_json(
            API_HOST + "/cloudide/api/v3/trae/GetUserInfo",
            {"ReqSource": "IDE", "IDEVersion": APP_VERSION},
            {"Content-Type": "application/json", "x-cloudide-token": token,
             "User-Agent": f"Trae/{APP_VERSION}"},
        )
        u = ui.get("Result") or ui
        return {
            "uid": str(u.get("UserID") or ""),
            "nickname": str(u.get("ScreenName") or ""),
            "enterprise_id": str(u.get("EnterpriseID") or ""),
        }
    except Exception as e:
        print(f"[*] GetUserInfo 失败: {e}", file=sys.stderr)
        return {}


def test_work_chat(work: dict) -> str:
    """用 Work 通道发一条测试消息。"""
    import hashlib
    import uuid

    token = work["access_token"]
    machine_id = work.get("machine_id") or uuid.uuid4().hex
    device_id = work.get("device_id") or hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "1+1等于几"}]}],
        "model": "glm-5.2",
        "function": "solo_work_lite",
        "stream": True,
        "request_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
    }
    headers = {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "x-uid": work.get("uid") or "",
        "x-app-id": "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8",
        "x-device-id": device_id,
        "x-machine-id": machine_id,
        "x-request-id": str(uuid.uuid4()),
        "x-ide-version": "3.3.67",
        "x-ide-version-code": "20260401",
        "x-device-type": "windows",
        "x-os-version": "Windows 10",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    url = work.get("api_host", API_HOST).rstrip("/") + "/api/agent/v3/llm_utils_chat"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


class Handler(BaseHTTPRequestHandler):
    def _reject(self, msg: str) -> None:
        self.send_response(403)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(msg.encode())
        print(f"[!] 拒绝回调: {msg}", file=sys.stderr)

    def do_GET(self):
        """捕获 /authorize 回调。"""
        if not self.path.startswith("/authorize"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        refresh_token = (qs.get("refreshToken") or [""])[0]
        # 容错：缺 refreshToken 时试 userJwt
        if not refresh_token:
            try:
                user_jwt = json.loads(urllib.parse.unquote((qs.get("userJwt") or [""])[0]))
                refresh_token = str(user_jwt.get("RefreshToken") or "")
            except Exception:
                pass

        # nonce 校验：回调必须携带 trae_work_login.py 生成的 state 文件中
        # 的一次性 nonce，防止本机恶意网页直接 GET 注入伪造 refreshToken
        # 覆盖用户凭证（否则后续对话会被发到攻击者账号）
        login_state = _load_login_state()
        expected_nonce = login_state.get("nonce") or ""
        created_at = int(login_state.get("created_at") or 0)
        callback_nonce = (qs.get("nonce") or [""])[0]
        if not expected_nonce:
            self._reject(
                "<h3>回调被拒绝</h3><p>未找到登录状态（nonce 缺失）。"
                "请先运行 trae_work_login.py 生成登录链接后再走服务器回调流程</p>"
            )
            return
        if time.time() - created_at > STATE_TTL:
            self._reject(
                "<h3>回调被拒绝</h3><p>登录状态已过期（超过 15 分钟），"
                "请重新运行 trae_work_login.py 生成新的登录链接</p>"
            )
            return
        if not callback_nonce or not secrets.compare_digest(callback_nonce, expected_nonce):
            self._reject(
                "<h3>回调被拒绝</h3><p>nonce 校验失败（回调可能被伪造）</p>"
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        if not refresh_token:
            msg = "<h3>回调缺少 refreshToken</h3><p>请检查链接是否完整</p>"
            self.wfile.write(msg.encode())
            print("[!] 回调缺少 refreshToken", file=sys.stderr)
            return

        try:
            cred = exchange_token(refresh_token)
            user = get_user_info(cred["access_token"])
            out = {
                "uid": user.get("uid") or "",
                "nickname": user.get("nickname") or "",
                "enterprise_id": user.get("enterprise_id") or "",
                "access_token": cred["access_token"],
                "refresh_token": cred["refresh_token"],
                "expires_at": cred["expires_at"],
                "api_host": API_HOST,
                "machine_id": login_state.get("machine_id", ""),
                "device_id": login_state.get("device_id", ""),
            }
            _write_cred_secure(OUT_PATH, out)
            # 一次性消费：成功落盘后删除状态文件，重放/重复回调一律拒绝
            STATE_PATH.unlink(missing_ok=True)
            msg = f"<h3>登录成功！</h3><p>uid={html.escape(out['uid'])} nickname={html.escape(out['nickname'])}</p><p>凭证已保存，可以关闭此页面</p>"
            self.wfile.write(msg.encode())
            print(f"\n[OK] 凭证已保存: {OUT_PATH}")
            print(f"    uid={out['uid']} nickname={out['nickname']}")
            print(f"    expires_at={out['expires_at']}")

            # 自动测试 Work 通道
            print("\n[*] 测试 Work 通道 (solo_work_lite)...")
            try:
                raw = test_work_chat(out)
                print("=== Work chat 响应 ===")
                print(raw[:500])
            except Exception as e:
                print(f"[!] Work chat 测试失败: {e}")
        except Exception as e:
            msg = f"<h3>换 token 失败</h3><p>{html.escape(str(e))}</p>"
            self.wfile.write(msg.encode())
            print(f"[!] 失败: {e}", file=sys.stderr)

    def log_message(self, *args):
        pass


def main() -> int:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[*] 回调监听启动: http://127.0.0.1:{PORT}/authorize")
    print("[*] 现在打开登录链接，登录成功后会自动回调到这里")
    print("[*] 等待回调...（Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] 已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
