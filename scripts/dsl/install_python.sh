#!/usr/bin/env bash
# ============================================================================
# install_python.sh — DSL 1/4: python/ NumPy 正确性基准环境 (CPU, 无 NPU 依赖)
#
# 用法:
#   ./install_python.sh            # 安装 (默认)
#   ./install_python.sh verify     # 仅验证已装环境
#
# 依赖: uv (自动安装), Python >= 3.12 (uv 自动获取); 任意机器可跑
# ============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DSL_DIR="$REPO_ROOT/examples/python"

do_verify() {
  cd "$DSL_DIR" || return 1
  [[ -d .venv ]] || { warn "python 基准: venv 不存在 (先 install)"; return 1; }
  if uv run python -c "import numpy, termcolor, pytest; print('python 基准: OK, numpy', numpy.__version__)"; then
    info "运行: cd examples/python && uv run python src/gemm.py src/softmax.py ..."
    return 0
  fi
  warn "python 基准: 不可用"
  return 1
}

do_install() {
  info "== python 基准 (examples/python) =="
  ensure_uv || return 1
  cd "$DSL_DIR" || return 1
  uv sync || { fail "uv sync 失败 (需要 Python >= 3.12)"; return 1; }
  do_verify || return 1
}

case "${1:-install}" in
  install) do_install ;;
  verify)  do_verify ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
