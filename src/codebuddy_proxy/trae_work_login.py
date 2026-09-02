"""Trae Work (SOLO) 登录脚本 —— 生成登录链接 → 换 token → 落盘凭证。

流程（对齐 traework2api/login.sh）：
1. 生成 machine_id / device_id
2. 构造登录 URL（带 127.0.0.1 回调 + client_id=en1oxy7wnw8j9n）
3. 用户在浏览器登录 → 复制回调链接粘贴回来
4. ExchangeToken（refreshToken → access_token）
5. GetUserInfo 确认 uid → 落盘 ~/.ethan/trae_work.json

用法：
  python3 trae_work_login.py
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CLIENT_ID = "en1oxy7wnw8j9n"  # SOLO stable
APP_VERSION = "0.1.43"
API_HOST = "https://api.trae.com.cn"
AUTH_HOST = "https://www.trae.cn/authorization"
OUT_PATH = Path.home() / ".ethan" / "trae_work.json"
# 供 trae_work_login_server.py 读取的本次登录状态（随机 id，非机密）
STATE_PATH = Path("/tmp/trae_work_login_state.json")
# 登录状态有效期（秒）：超时后 server 拒绝回调
STATE_TTL = 900


def _http_post_json(url: str, body: dict, headers: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


def build_login_url() -> tuple[str, str, str]:
    machine_id = secrets.token_hex(16)
    device_id = secrets.token_hex(16)
    # 一次性 nonce：server 只接受带匹配 nonce 的回调，防止本机恶意网页
    # 直接 GET /authorize 注入伪造 refreshToken 覆盖凭证
    nonce = secrets.token_hex(16)
    params = {
        "login_version": "1",
        "auth_from": "solo",
        "login_channel": "native_ide",
        "plugin_version": "2.3.62834",
        "auth_type": "local",
        "client_id": CLIENT_ID,
        "redirect": "0",
        "login_trace_id": secrets.token_hex(8),
        "auth_callback_url": f"http://127.0.0.1:18080/authorize?nonce={nonce}",
        "machine_id": machine_id,
        "device_id": device_id,
        "x_device_id": device_id,
        "x_machine_id": machine_id,
        "x_device_brand": "PC",
        "x_device_type": "PC",
        "x_os_version": "1.0",
        "x_app_version": APP_VERSION,
        "x_app_type": "stable",
    }
    url = AUTH_HOST + "?" + urllib.parse.urlencode(params)
    # 状态文件供 trae_work_login_server.py 使用：machine_id/device_id 必须
    # 复用同一对（避免每请求随机指纹触发风控），nonce 用于回调防伪造
    STATE_PATH.write_text(json.dumps({
        "machine_id": machine_id,
        "device_id": device_id,
        "nonce": nonce,
        "created_at": int(time.time()),
    }))
    return url, machine_id, device_id


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


def main() -> int:
    url, machine_id, device_id = build_login_url()
    print("=" * 60)
    print("Trae Work (SOLO) 登录")
    print("=" * 60)
    print()
    print("步骤：")
    print("  1. 在浏览器打开下面链接，用手机号/验证码登录")
    print("  2. 登录成功后浏览器会跳到打不开的 127.0.0.1 地址")
    print("  3. 复制浏览器地址栏的完整链接，粘贴到下面（不回显）")
    print()
    print("登录链接：")
    print(f"  {url}")
    print()

    import getpass

    callback = getpass.getpass("登录完成后，粘贴回调链接: ")
    if not callback:
        print("未输入，已取消")
        return 1

    qs = urllib.parse.parse_qs(urllib.parse.urlparse(callback).query)
    refresh_token = (qs.get("refreshToken") or [""])[0]

    # 容错：回调缺 refreshToken 时尝试 userJwt
    if not refresh_token:
        try:
            user_jwt = json.loads(urllib.parse.unquote((qs.get("userJwt") or [""])[0]))
            refresh_token = str(user_jwt.get("RefreshToken") or "")
        except Exception:
            pass
    if not refresh_token:
        print("[!] 回调链接缺少 refreshToken", file=sys.stderr)
        return 1

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
        "machine_id": machine_id,
        "device_id": device_id,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 直接以 0600 权限创建（避免 write_text 后 chmod 前的短暂 0644 窗口）
    fd = os.open(OUT_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print()
    print(f"[OK] 凭证已保存: {OUT_PATH}")
    print(f"    uid={out['uid']} nickname={out['nickname']} expires={out['expires_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
