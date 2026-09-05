#!/usr/bin/env bash
# ============================================================================
# install_dsl_envs.sh — 一键安装四种 DSL 的开发环境 (版本锁 = 仓库实测组合)
#
# 实测可用组合 (详见 examples/*/README.md 与 docs/pages/dsl/):
#   CANN 9.0.0 + torch 2.8.0 + torch_npu 2.8.0rc1 + triton-ascend 3.2.0
#   + tilelang-ascend v0.1.1.010 (cann900 wheel)
#
# 目标机: Linux aarch64 / x86_64 + 已装 CANN (先跑 scripts/install_npu_toolchain.sh)。
# python 基准 (examples/python) 是纯 NumPy, 任意机器可装。
#
# 用法:
#   ./install_dsl_envs.sh all        # 四种 DSL 全部安装 (默认)
#   ./install_dsl_envs.sh python     # 仅 NumPy 基准 (无 NPU 依赖)
#   ./install_dsl_envs.sh ascend_c   # 仅检查/构建 Ascend C 工具链 (cmake+bisheng)
#   ./install_dsl_envs.sh triton     # 仅 triton_ascend venv
#   ./install_dsl_envs.sh tilelang   # 仅 tilelang_ascend venv
#   ./install_dsl_envs.sh verify     # 只验证四个环境是否可用
#
# 可用环境变量覆盖:
#   TORCH_NPU_VERSION=2.8.0rc1   必须与 torch 2.8.0 严格一致 (pyproject 锁定)
#   TILELANG_WHEEL_URL=...       手动指定 tilelang-ascend wheel 直链
#   SKIP_BUILD=1                 ascend_c 安装时跳过试编译
# ============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.8.0rc1}"
DSL_PYTHON="3.11"          # torch_npu 2.8.0rc1 仅发布 cp311 wheel
SKIP_BUILD="${SKIP_BUILD:-0}"

info() { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; return 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- uv 就绪 ----------------------------------------------------------------
ensure_uv() {
  if have uv; then return 0; fi
  info "未检测到 uv, 安装中 (https://github.com/astral-sh/uv)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv 安装失败"
  export PATH="$HOME/.local/bin:$PATH"
  have uv || fail "uv 安装后仍不可用, 请检查 PATH"
}

# ---- CANN 环境 (NPU 系 DSL 前置) --------------------------------------------
load_cann() {
  local p
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then return 0; fi
  for p in /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
           "$HOME/Ascend/ascend-toolkit/latest/set_env.sh"; do
    if [[ -f "$p" ]]; then
      # shellcheck disable=SC1090
      source "$p"; break
    fi
  done
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    warn "未找到 CANN (ASCEND_HOME_PATH 为空)。请先运行:"
    warn "  sudo scripts/install_npu_toolchain.sh   # 或手动 source set_env.sh"
    return 1
  fi
  info "CANN: $ASCEND_HOME_PATH"
}

need_linux() {
  [[ "$(uname -s)" == "Linux" ]] || { warn "$1 需要 Linux (torch_npu 无 mac/win wheel)"; return 1; }
}

# ============================================================================
# 1. python 基准 (NumPy, CPU, 无 NPU 依赖)
# ============================================================================
install_python() {
  info "== python 基准 (examples/python) =="
  ensure_uv || return 1
  cd "$REPO_ROOT/examples/python" || return 1
  uv sync || fail "uv sync 失败 (需要 Python >= 3.12)" || return 1
  uv run python -c "import numpy, termcolor, pytest; print('python 基准环境 OK, numpy', numpy.__version__)" \
    || fail "python 基准导入失败" || return 1
  info "python 基准 OK —— 运行: cd examples/python && uv run python src/gemm.py ..."
}

verify_python() {
  cd "$REPO_ROOT/examples/python" || return 1
  [[ -d .venv ]] || { warn "python 基准: venv 不存在 (先 install)"; return 1; }
  if uv run python -c "import numpy, termcolor, pytest"; then
    info "python 基准: OK"
  else
    warn "python 基准: 不可用"; return 1
  fi
}

# ============================================================================
# 2. ascend_c (CANN 原生 C++, 只需工具链; 试编译不需要 NPU 设备)
# ============================================================================
install_ascend_c() {
  info "== ascend_c (examples/ascend_c) =="
  load_cann || return 1
  for cmd in cmake g++ bisheng; do
    have "$cmd" || { warn "缺少 $cmd (source set_env.sh 后 bisheng 应在 PATH)"; return 1; }
  done
  if [[ "$SKIP_BUILD" == "1" ]]; then
    info "SKIP_BUILD=1, 跳过试编译"
    return 0
  fi
  cd "$REPO_ROOT/examples/ascend_c" || return 1
  info "试编译 (ascendc.cmake: bisheng 编 kernel → 打包 → 编 host, 不需要 NPU 设备)..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/ascend_c_cmake.log 2>&1 \
    || { tail -20 /tmp/ascend_c_cmake.log; fail "cmake configure 失败"; return 1; }
  cmake --build build -j"$(nproc)" >/tmp/ascend_c_build.log 2>&1 \
    || { tail -30 /tmp/ascend_c_build.log; fail "构建失败 (日志 /tmp/ascend_c_build.log)"; return 1; }
  ls build/ascend_gemm >/dev/null 2>&1 || { fail "未找到 build/ascend_gemm"; return 1; }
  info "ascend_c OK —— 运行: cd examples/ascend_c/build && ./ascend_gemm"
}

# ============================================================================
# 3. triton_ascend (torch_npu + triton-ascend)
# ============================================================================
install_triton() {
  info "== triton_ascend (examples/triton_ascend) =="
  need_linux "triton_ascend" || return 1
  load_cann || return 1
  ensure_uv || return 1
  cd "$REPO_ROOT/examples/triton_ascend" || return 1

  uv venv --python "$DSL_PYTHON" || fail "创建 py$DSL_PYTHON venv 失败" || return 1
  uv sync || fail "uv sync 失败 (应装好 numpy + torch==2.8.0)" || return 1

  # torch_npu 必须与 torch 2.8.0 严格一致, 否则 import 报 undefined symbol
  uv pip install "torch-npu==${TORCH_NPU_VERSION}" --prerelease=allow \
    || fail "torch-npu==${TORCH_NPU_VERSION} 安装失败" || return 1
  # triton-ascend 3.2.0 (PyPI 双架构 wheel); 失败则源码:
  #   git clone https://gitcode.com/Ascend/triton-ascend.git && uv pip install -e ./triton-ascend
  uv pip install triton-ascend \
    || fail "triton-ascend 安装失败 (备选: 源码安装, 见 README)" || return 1
  # torch_npu/triton-ascend 运行时依赖 (裸环境必装, 2026-09 新容器实测)
  uv pip install pyyaml decorator attrs psutil scipy pybind11 || return 1

  # triton-ascend 3.2.0 + CANN 9.0.0 的 enum 重命名补丁 (见 README 常见问题)
  local cpp
  for cpp in .venv/lib/python3.*/site-packages/triton/backends/ascend/npu_utils.cpp; do
    if [[ -f "$cpp" ]] && grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$cpp"; then
      sed -i 's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE/g' "$cpp"
      rm -rf "$HOME/.triton/cache"
      info "已打 CANN 9.0.0 enum 补丁: $cpp"
    fi
  done

  verify_triton
}

verify_triton() {
  cd "$REPO_ROOT/examples/triton_ascend" || return 1
  [[ -d .venv ]] || { warn "triton_ascend: venv 不存在 (先 install)"; return 1; }
  load_cann || return 1
  if uv run python -c "
import torch, torch_npu, triton
print('triton_ascend 环境 OK: torch', torch.__version__, '/ triton-ascend', triton.__version__)
print('NPU 可见:', torch.npu.is_available())
"; then
    info "triton_ascend: OK —— 运行: cd examples/triton_ascend && ASCEND_RT_VISIBLE_DEVICES=<卡号> uv run python src/test_gemm.py"
    return 0
  fi
  warn "triton_ascend: import 失败 — 检查 torch_npu 与 torch 版本是否严格一致 / CANN 是否 source"
  return 1
}

# ============================================================================
# 4. tilelang_ascend (torch_npu + tilelang-ascend 预编译 wheel)
# ============================================================================
install_tilelang() {
  info "== tilelang_ascend (examples/tilelang_ascend) =="
  need_linux "tilelang_ascend" || return 1
  load_cann || return 1
  ensure_uv || return 1
  cd "$REPO_ROOT/examples/tilelang_ascend" || return 1

  uv venv --python "$DSL_PYTHON" || fail "创建 py$DSL_PYTHON venv 失败" || return 1
  uv sync || fail "uv sync 失败" || return 1
  uv pip install "torch-npu==${TORCH_NPU_VERSION}" --prerelease=allow \
    || fail "torch-npu==${TORCH_NPU_VERSION} 安装失败" || return 1

  # tilelang-ascend 预编译 wheel: v0.1.1.010-release 只有 cann900 资产 (aarch64/x86_64)
  # 注意: PyPI 的 tilelang 主包是 CUDA 版, 不含 ascend 后端, 千万别装错
  local wheel_url="${TILELANG_WHEEL_URL:-}"
  if [[ -z "$wheel_url" ]]; then
    local wheel_name
    case "$(uname -m)" in
      aarch64|arm64) wheel_name="tilelang-0.1.1.10%2Bubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl" ;;
      x86_64)        wheel_name="tilelang-0.1.1.10%2Blinux.cann900-cp311-cp311-linux_x86_64.whl" ;;
      *) warn "不支持的架构: $(uname -m)"; return 1 ;;
    esac
    wheel_url="https://github.com/tile-ai/tilelang-ascend/releases/download/v0.1.1.010-release/${wheel_name}"
  fi
  local whl="/tmp/${wheel_url##*/}"
  whl="${whl//%2B/+}"   # 解码 %2B -> '+' (uv/pip 校验 wheel 文件命名规范)
  if [[ ! -s "$whl" ]]; then
    info "下载 tilelang-ascend wheel: $wheel_url"
    curl -fL --retry 3 --progress-bar -o "$whl" "$wheel_url" \
      || fail "wheel 下载失败 (CANN 非 9.0.x 时请到 https://github.com/tile-ai/tilelang-ascend/releases 选对应 wheel, 并 TILELANG_WHEEL_URL=... 重跑)" || return 1
  fi
  uv pip install "$whl" || fail "tilelang wheel 安装失败" || return 1
  # torch_npu 运行时依赖 (裸环境必装)
  uv pip install pyyaml decorator attrs psutil scipy || return 1

  verify_tilelang
}

verify_tilelang() {
  cd "$REPO_ROOT/examples/tilelang_ascend" || return 1
  [[ -d .venv ]] || { warn "tilelang_ascend: venv 不存在 (先 install)"; return 1; }
  load_cann || return 1
  # ACL_OP_INIT_MODE=1 跳过 torch_npu TBE 初始化, 规避 tilelang 与 CANN 的 TVM FFI 冲突
  if ACL_OP_INIT_MODE=1 uv run python -c "
import torch, torch_npu, tilelang
print('tilelang_ascend 环境 OK: tilelang', tilelang.__version__, '/ NPU 可见:', torch.npu.is_available())
"; then
    info "tilelang_ascend: OK —— 运行: cd examples/tilelang_ascend && ACL_OP_INIT_MODE=1 uv run python src/test_gemm.py"
    return 0
  fi
  warn "tilelang_ascend: import 失败 — 确认装的是 ascend wheel 而非 PyPI CUDA 版 tilelang"
  return 1
}

verify_ascend_c() {
  load_cann || return 1
  for cmd in cmake g++ bisheng; do
    have "$cmd" || { warn "ascend_c: 缺少 $cmd"; return 1; }
  done
  if ls "$REPO_ROOT/examples/ascend_c/build/ascend_gemm" >/dev/null 2>&1; then
    info "ascend_c: OK (工具链在位, build/ 产物已生成)"
    return 0
  fi
  warn "ascend_c: 工具链在位但 build/ascend_gemm 未构建 — 重跑 install_ascend_c 试编译"
  return 1
}

# ============================================================================
# 入口
# ============================================================================
TARGET="${1:-all}"
declare -A RESULTS=()

run_step() { # run_step <name> <cmd...>
  local name="$1"; shift
  if "$@"; then RESULTS[$name]="OK"; else RESULTS[$name]="FAILED"; fi
}

case "$TARGET" in
  all)
    run_step "python"     install_python
    run_step "ascend_c"   install_ascend_c
    run_step "triton"     install_triton
    run_step "tilelang"   install_tilelang
    ;;
  python)   run_step "python"   install_python ;;
  ascend_c) run_step "ascend_c" install_ascend_c ;;
  triton)   run_step "triton"   install_triton ;;
  tilelang) run_step "tilelang" install_tilelang ;;
  verify)
    run_step "python"     verify_python
    run_step "triton"     verify_triton
    run_step "tilelang"   verify_tilelang
    run_step "ascend_c"   verify_ascend_c
    ;;
  *) sed -n '2,27p' "$0"; exit 1 ;;
esac

echo
echo "=========== 安装结果汇总 ==========="
rc=0
for name in python ascend_c triton tilelang; do
  r="${RESULTS[$name]:-SKIP}"
  printf '  %-10s %s\n' "$name" "$r"
  [[ "$r" == "FAILED" ]] && rc=1
done
echo "===================================="
[[ $rc -eq 0 ]] && info "全部完成。运行前置(每次 shell): source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
exit $rc
