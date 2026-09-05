# ============================================================================
# common.sh — scripts/dsl/install_*.sh 共享工具函数 (由各脚本 source, 不单独执行)
# ============================================================================
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DSL_PYTHON="${DSL_PYTHON:-3.11}"   # torch_npu 2.8.0rc1 仅发布 cp311 wheel

info() { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*" >&2; return 1; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_uv() {
  if have uv; then return 0; fi
  info "未检测到 uv, 安装中 (https://github.com/astral-sh/uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { fail "uv 安装失败"; return 1; }
  export PATH="$HOME/.local/bin:$PATH"
  have uv || { fail "uv 安装后仍不可用, 请检查 PATH"; return 1; }
}

load_cann() {
  local p
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then info "CANN: $ASCEND_HOME_PATH"; return 0; fi
  for p in /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
           "$HOME/Ascend/ascend-toolkit/latest/set_env.sh"; do
    if [[ -f "$p" ]]; then
      # shellcheck disable=SC1090
      source "$p"; break
    fi
  done
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    warn "未找到 CANN (ASCEND_HOME_PATH 为空)。请先运行:"
    warn "  sudo $REPO_ROOT/scripts/install_npu_toolchain.sh   # 或手动 source set_env.sh"
    return 1
  fi
  info "CANN: $ASCEND_HOME_PATH"
}

need_linux() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    warn "$1 需要 Linux (torch_npu 无 mac/win wheel)"
    return 1
  fi
}
