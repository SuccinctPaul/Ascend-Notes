#!/usr/bin/env bash
# ============================================================================
# install_ascend_c.sh — DSL 2/4: ascend_c/ CANN 原生 C++ 环境检查 + 试编译
#
# 用法:
#   ./install_ascend_c.sh            # 检查工具链 + cmake 试编译 (无需 NPU 设备)
#   ./install_ascend_c.sh verify     # 仅验证工具链与已有构建产物
#
# 前置: CANN toolkit (sudo ../../install_npu_toolchain.sh, 或已装好并 source
#       /usr/local/Ascend/ascend-toolkit/latest/set_env.sh)
# 环境变量: SKIP_BUILD=1 跳过试编译
# ============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DSL_DIR="$REPO_ROOT/examples/ascend_c"
SKIP_BUILD="${SKIP_BUILD:-0}"

check_tools() {
  local cmd
  for cmd in cmake g++ bisheng; do
    have "$cmd" || { warn "ascend_c: 缺少 $cmd (source set_env.sh 后 bisheng 应在 PATH)"; return 1; }
  done
}

do_verify() {
  load_cann || return 1
  check_tools || return 1
  if [[ -f "$DSL_DIR/build/ascend_gemm" ]]; then
    info "ascend_c: OK (工具链在位, build/ 产物已生成)"
    info "运行: cd examples/ascend_c/build && ./ascend_gemm"
    return 0
  fi
  warn "ascend_c: 工具链在位但 build/ascend_gemm 未构建 — 重跑 install 试编译"
  return 1
}

do_install() {
  info "== ascend_c (examples/ascend_c) =="
  load_cann || return 1
  check_tools || return 1
  if [[ "$SKIP_BUILD" == "1" ]]; then
    info "SKIP_BUILD=1, 跳过试编译"
    do_verify >/dev/null 2>&1 || true
    info "ascend_c: OK (工具链在位)"
    return 0
  fi
  cd "$DSL_DIR" || return 1
  info "试编译 (ascendc.cmake: bisheng 编 kernel → 打包 → 编 host, 不需要 NPU 设备)..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/ascend_c_cmake.log 2>&1 \
    || { tail -20 /tmp/ascend_c_cmake.log; fail "cmake configure 失败"; return 1; }
  cmake --build build -j"$(nproc)" >/tmp/ascend_c_build.log 2>&1 \
    || { tail -30 /tmp/ascend_c_build.log; fail "构建失败 (日志 /tmp/ascend_c_build.log)"; return 1; }
  do_verify || return 1
}

case "${1:-install}" in
  install) do_install ;;
  verify)  do_verify ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
