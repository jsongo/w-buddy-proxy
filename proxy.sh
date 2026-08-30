#!/usr/bin/env bash
# proxy.sh - manage the w-buddy-proxy (codebuddy_proxy) FastAPI server
#
# Usage:
#   ./proxy.sh start    [-p PORT] [-H HOST]     # 后台启动 (默认 127.0.0.1:8787)
#   ./proxy.sh stop                              # 停止
#   ./proxy.sh restart  [-p PORT] [-H HOST]     # 重启
#   ./proxy.sh status                            # 查看状态
#   ./proxy.sh logs                              # 跟踪日志
#
# 支持环境变量（start 命令生效）：
#   PROXY_PORT, PROXY_HOST, PROXY_EXTRA_ARGS (附加传给 python -m codebuddy_proxy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认配置
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8787}"
EXTRA_ARGS="${PROXY_EXTRA_ARGS:---desensitize}"

# 优先使用项目自带 .venv（uv 已装好依赖），否则退回系统 python
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_BIN="uv run python"
else
    PYTHON_BIN="python3"
fi

# 状态文件
RUNTIME_DIR="$HOME/.codebuddy-proxy"
PID_FILE="$RUNTIME_DIR/proxy.pid"
LOG_FILE="$RUNTIME_DIR/proxy.log"

mkdir -p "$RUNTIME_DIR"

log() { printf '[proxy.sh] %s\n' "$*"; }

read_pid() {
    if [[ -f "$PID_FILE" ]]; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

is_running() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

usage() {
    grep '^# ' "$0" | sed 's/^# //'
    exit 1
}

parse_common() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--port)  PROXY_PORT="$2"; shift 2 ;;
            -H|--host)  PROXY_HOST="$2"; shift 2 ;;
            --port=*)   PROXY_PORT="${1#*=}"; shift ;;
            --host=*)   PROXY_HOST="${1#*=}"; shift ;;
            *)          EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
        esac
    done
}

cmd_start() {
    parse_common "$@"
    local pid
    pid="$(read_pid)"
    if is_running "$pid"; then
        log "already running (pid=$pid, $PROXY_HOST:$PROXY_PORT)"
        return 0
    fi
    # 清理残留 pid
    [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"

    log "starting w-buddy-proxy on $PROXY_HOST:$PROXY_PORT ..."
    # nohup + & : 立刻返回，不阻塞
    # 走源文件 src/ 运行，避开本机 setuptools env 的 EEXIST 环境 bug（uv sync 构建可编辑安装时触发）
    PYTHONPATH="$SCRIPT_DIR/src" \
    nohup $PYTHON_BIN -m codebuddy_proxy \
        --host "$PROXY_HOST" \
        --port "$PROXY_PORT" \
        --static-models \
        $EXTRA_ARGS \
        >>"$LOG_FILE" 2>&1 &
    local new_pid=$!

    echo "$new_pid" > "$PID_FILE"
    # 等最多 5s 看是否存活
    for _ in 1 2 3 4 5; do
        sleep 1
        if ! is_running "$new_pid"; then
            log "FAILED to start, see $LOG_FILE"
            rm -f "$PID_FILE"
            tail -n 20 "$LOG_FILE" || true
            return 1
        fi
    done
    log "started (pid=$new_pid), log: $LOG_FILE"
    return 0
}

cmd_stop() {
    local pid
    pid="$(read_pid)"
    if [[ -z "$pid" ]] || ! is_running "$pid"; then
        log "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    log "stopping pid=$pid ..."
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if ! is_running "$pid"; then
            rm -f "$PID_FILE"
            log "stopped"
            return 0
        fi
    done
    log "force killing pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    return 0
}

cmd_restart() {
    cmd_stop || true
    sleep 1
    cmd_start "$@"
}

cmd_status() {
    local pid
    pid="$(read_pid)"
    if is_running "$pid"; then
        # 试着从 ps 里读当前端口（best-effort，失败回退到默认）
        local actual
        actual=$(ps -o command= -p "$pid" 2>/dev/null | grep -oE -- '--port[[:space:]]+[0-9]+' | awk '{print $2}' | head -1)
        local actual_host
        actual_host=$(ps -o command= -p "$pid" 2>/dev/null | grep -oE -- '--host[[:space:]]+[^[:space:]]+' | awk '{print $2}' | head -1)
        local host="${actual_host:-$PROXY_HOST}"
        local port="${actual:-$PROXY_PORT}"
        log "running (pid=$pid, $host:$port)"
        log "log: $LOG_FILE"
        return 0
    fi
    log "not running (pid_file=${pid:-none})"
    return 1
}

cmd_logs() {
    tail -n 100 -F "$LOG_FILE"
}

case "${1:-}" in
    start)   shift; cmd_start "$@" ;;
    stop)    cmd_stop ;;
    restart) shift; cmd_restart "$@" ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    -h|--help|help|"") usage ;;
    *)       usage ;;
esac
