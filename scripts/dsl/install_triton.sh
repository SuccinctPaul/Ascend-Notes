#!/usr/bin/env bash
# ============================================================================
# install_triton.sh — DSL 3/4: triton_ascend/ (triton-ascend 3.2.0 + torch_npu)
#
# 版本锁 (与 CANN 9.0.0 实测配套, 踩坑细节见 examples/triton_ascend/README.md):
#   torch 2.8.0 (pyproject 锁定) + torch_npu 2.8.0rc1 + triton-ascend 3.2.0
#
# 用法:
#   ./install_triton.sh            # 安装 (默认)
#   ./install_triton.sh verify     # 仅验证 (import torch/torch_npu/triton)
#
# 环境变量: TORCH_NPU_VERSION=2.8.0rc1   DSL_PYTHON=3.11 (torch_npu 仅 cp311 wheel)
# ============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DSL_DIR="$REPO_ROOT/examples/triton_ascend"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.8.0rc1}"

apply_cann9_enum_patch() {
  # triton-ascend 3.2.0 + CANN 9.0.0: CANN 把该 enum 重命名为
  # RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE, 而 triton-ascend 仍用旧名
  # (见 triton_ascend/README.md 常见问题); 补丁需清 ~/.triton 缓存生效
  local cpp
  for cpp in "$DSL_DIR"/.venv/lib/python3.*/site-packages/triton/backends/ascend/npu_utils.cpp; do
    if [[ -f "$cpp" ]] && grep -q "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE" "$cpp"; then
      sed -i 's/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE/g' "$cpp"
      rm -rf "$HOME/.triton/cache"
      info "已打 CANN 9.0.0 enum 补丁: $cpp"
    fi
  done
}

do_verify() {
  cd "$DSL_DIR" || return 1
  [[ -d .venv ]] || { warn "triton_ascend: venv 不存在 (先 install)"; return 1; }
  load_cann || return 1
  if uv run python -c "
import torch, torch_npu, triton
print('triton_ascend: OK — torch', torch.__version__, '/ triton-ascend', triton.__version__)
print('NPU 可见:', torch.npu.is_available())
"; then
    info "运行: cd examples/triton_ascend && ASCEND_RT_VISIBLE_DEVICES=<卡号> uv run python src/test_gemm.py"
    return 0
  fi
  warn "triton_ascend: import 失败 — 检查 torch_npu 与 torch 版本是否严格一致 / CANN 是否 source"
  return 1
}

do_install() {
  info "== triton_ascend (examples/triton_ascend) =="
  need_linux "triton_ascend" || return 1
  load_cann || return 1
  ensure_uv || return 1
  cd "$DSL_DIR" || return 1

  uv venv --python "$DSL_PYTHON" || { fail "创建 py$DSL_PYTHON venv 失败"; return 1; }
  uv sync || { fail "uv sync 失败 (应装好 numpy + torch==2.8.0)"; return 1; }
  # torch_npu 必须与 torch 2.8.0 严格一致, 否则 import 报 undefined symbol
  uv pip install "torch-npu==${TORCH_NPU_VERSION}" --prerelease=allow \
    || { fail "torch-npu==${TORCH_NPU_VERSION} 安装失败"; return 1; }
  # triton-ascend 3.2.0 (PyPI 双架构 wheel); 失败则源码:
  #   git clone https://gitcode.com/Ascend/triton-ascend.git && uv pip install -e ./triton-ascend
  uv pip install triton-ascend \
    || { fail "triton-ascend 安装失败 (备选: 源码安装, 见 examples/triton_ascend/README.md)"; return 1; }
  # torch_npu/triton-ascend 运行时依赖 (裸环境必装, 2026-09 新容器实测)
  uv pip install pyyaml decorator attrs psutil scipy pybind11 || return 1

  apply_cann9_enum_patch
  do_verify || return 1
}

case "${1:-install}" in
  install) do_install ;;
  verify)  do_verify ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
