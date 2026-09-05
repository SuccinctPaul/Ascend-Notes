#!/usr/bin/env bash
# ============================================================================
# install_dsl_envs.sh — 编排入口: 调用 scripts/dsl/ 下四种 DSL 的独立安装脚本
#                       与 scripts/install_npu_toolchain.sh (NPU 工具链)
#
# 每种 DSL 也可脱离本脚本单独安装/验证 (独立场景直接用对应脚本):
#   scripts/dsl/install_python.sh     [install|verify]   # NumPy 基准 (无 NPU 依赖)
#   scripts/dsl/install_ascend_c.sh   [install|verify]   # CANN 原生 C++
#   scripts/dsl/install_triton.sh     [install|verify]   # triton-ascend + torch_npu
#   scripts/dsl/install_tilelang.sh   [install|verify]   # tilelang-ascend + torch_npu
#
# 用法:
#   ./install_dsl_envs.sh all                  # 四种 DSL 全部安装 (默认)
#   ./install_dsl_envs.sh <python|ascend_c|triton|tilelang>
#   ./install_dsl_envs.sh verify               # 依次验证四个环境
#   ./install_dsl_envs.sh toolchain [args...]  # 转调 NPU 工具链安装 (透传参数,
#                                              # 如 --download-only / --driver ...)
#   ./install_dsl_envs.sh all --with-toolchain # 先装 NPU 工具链, 再装四种 DSL
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN="$SCRIPT_DIR/install_npu_toolchain.sh"

info() { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }

ORDER=(python ascend_c triton tilelang)
declare -A DSL_SCRIPT=(
  [python]="$SCRIPT_DIR/dsl/install_python.sh"
  [ascend_c]="$SCRIPT_DIR/dsl/install_ascend_c.sh"
  [triton]="$SCRIPT_DIR/dsl/install_triton.sh"
  [tilelang]="$SCRIPT_DIR/dsl/install_tilelang.sh"
)

usage() { sed -n '2,22p' "$0"; exit 1; }

# ---- 参数解析: <action> [--with-toolchain] [passthrough...] -----------------
# 第一个位置参数 = action, 其余参数 (含 -* 选项) 原样透传给工具链脚本
ACTION=""
WITH_TOOLCHAIN=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-toolchain) WITH_TOOLCHAIN=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    -*) PASSTHROUGH+=("$1") ;;
    *)
      if [[ -z "$ACTION" ]]; then ACTION="$1"; else PASSTHROUGH+=("$1"); fi
      ;;
  esac
  shift
done
ACTION="${ACTION:-all}"

case "$ACTION" in
  all|python|ascend_c|triton|tilelang|verify|toolchain) ;;
  *) usage ;;
esac

# ---- 工具链转调 -------------------------------------------------------------
if [[ "$ACTION" == "toolchain" ]]; then
  exec "$TOOLCHAIN" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
fi

if [[ "$WITH_TOOLCHAIN" == "1" ]]; then
  info "== 前置: 安装 NPU 工具链 (install_npu_toolchain.sh) =="
  if bash "$TOOLCHAIN" ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh 2>/dev/null \
      || source "$HOME/Ascend/ascend-toolkit/latest/set_env.sh" 2>/dev/null || true
  else
    warn "工具链安装失败, 继续装 DSL (python 基准不受影响; NPU 系 DSL 会因缺 CANN 失败)"
  fi
fi

# ---- DSL 分发 ----------------------------------------------------------------
NAMES=(); RESULTS=()
run_dsl() { # run_dsl <name> <install|verify>
  local name="$1" act="$2"
  echo
  if bash "${DSL_SCRIPT[$name]}" "$act"; then
    NAMES+=("$name"); RESULTS+=("OK")
  else
    NAMES+=("$name"); RESULTS+=("FAILED")
  fi
}

case "$ACTION" in
  all)
    for name in "${ORDER[@]}"; do run_dsl "$name" install; done
    ;;
  verify)
    for name in "${ORDER[@]}"; do run_dsl "$name" verify; done
    ;;
  *)
    run_dsl "$ACTION" install
    ;;
esac

# ---- 汇总 --------------------------------------------------------------------
echo
echo "=========== 安装结果汇总 ==========="
rc=0
for i in "${!NAMES[@]}"; do
  printf '  %-10s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}"
  [[ "${RESULTS[$i]}" == "FAILED" ]] && rc=1
done
echo "===================================="
if [[ $rc -eq 0 ]]; then
  info "全部完成。运行前置(每次 shell): source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
fi
exit $rc
