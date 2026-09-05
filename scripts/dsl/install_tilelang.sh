#!/usr/bin/env bash
# ============================================================================
# install_tilelang.sh — DSL 4/4: tilelang_ascend/ (tilelang-ascend wheel + torch_npu)
#
# 版本锁 (与 CANN 9.0.0 实测配套, 踩坑细节见 examples/tilelang_ascend/README.md):
#   torch 2.8.0 (pyproject 锁定) + torch_npu 2.8.0rc1 + tilelang-ascend v0.1.1.010
#   (cann900 预编译 wheel, aarch64/x86_64 资产均已核实;
#    PyPI 的 tilelang 主包是 CUDA 版, 不含 ascend 后端, 勿装错)
#
# 用法:
#   ./install_tilelang.sh            # 安装 (默认)
#   ./install_tilelang.sh verify     # 仅验证 (ACL_OP_INIT_MODE=1 下 import)
#
# 环境变量: TORCH_NPU_VERSION=2.8.0rc1   TILELANG_WHEEL_URL=<直链覆盖>
#           DSL_PYTHON=3.11 (torch_npu 仅 cp311 wheel)
# ============================================================================
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

DSL_DIR="$REPO_ROOT/examples/tilelang_ascend"
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.8.0rc1}"

tilelang_wheel_url() {
  if [[ -n "${TILELANG_WHEEL_URL:-}" ]]; then
    echo "$TILELANG_WHEEL_URL"
    return 0
  fi
  local wheel_name
  case "$(uname -m)" in
    aarch64|arm64) wheel_name="tilelang-0.1.1.10%2Bubuntu.20.4.cann900-cp311-cp311-linux_aarch64.whl" ;;
    x86_64)        wheel_name="tilelang-0.1.1.10%2Blinux.cann900-cp311-cp311-linux_x86_64.whl" ;;
    *) warn "不支持的架构: $(uname -m)"; return 1 ;;
  esac
  echo "https://github.com/tile-ai/tilelang-ascend/releases/download/v0.1.1.010-release/${wheel_name}"
}

do_verify() {
  cd "$DSL_DIR" || return 1
  [[ -d .venv ]] || { warn "tilelang_ascend: venv 不存在 (先 install)"; return 1; }
  load_cann || return 1
  # ACL_OP_INIT_MODE=1 跳过 torch_npu TBE 初始化, 规避 tilelang 与 CANN 的 TVM FFI 冲突
  if ACL_OP_INIT_MODE=1 uv run python -c "
import torch, torch_npu, tilelang
print('tilelang_ascend: OK — tilelang', tilelang.__version__, '/ NPU 可见:', torch.npu.is_available())
"; then
    info "运行: cd examples/tilelang_ascend && ACL_OP_INIT_MODE=1 uv run python src/test_gemm.py"
    return 0
  fi
  warn "tilelang_ascend: import 失败 — 确认装的是 ascend wheel 而非 PyPI CUDA 版 tilelang"
  return 1
}

do_install() {
  info "== tilelang_ascend (examples/tilelang_ascend) =="
  need_linux "tilelang_ascend" || return 1
  load_cann || return 1
  ensure_uv || return 1
  cd "$DSL_DIR" || return 1

  uv venv --python "$DSL_PYTHON" || { fail "创建 py$DSL_PYTHON venv 失败"; return 1; }
  uv sync || { fail "uv sync 失败"; return 1; }
  # torch_npu 必须与 torch 2.8.0 严格一致 (不 pin 会被连带解析成 2.12.0 → 符号不匹配)
  uv pip install "torch-npu==${TORCH_NPU_VERSION}" --prerelease=allow \
    || { fail "torch-npu==${TORCH_NPU_VERSION} 安装失败"; return 1; }

  local url whl
  url="$(tilelang_wheel_url)" || return 1
  whl="/tmp/${url##*/}"
  whl="${whl//%2B/+}"   # 解码 %2B -> '+' (uv/pip 校验 wheel 文件命名规范)
  if [[ ! -s "$whl" ]]; then
    info "下载 tilelang-ascend wheel: $url"
    curl -fL --retry 3 --progress-bar -o "$whl" "$url" \
      || { fail "wheel 下载失败 (CANN 非 9.0.x 时请到 https://github.com/tile-ai/tilelang-ascend/releases 选对应 wheel, 并 TILELANG_WHEEL_URL=... 重跑)"; return 1; }
  fi
  uv pip install "$whl" || { fail "tilelang wheel 安装失败"; return 1; }
  # torch_npu 运行时依赖 (裸环境必装)
  uv pip install pyyaml decorator attrs psutil scipy || return 1

  do_verify || return 1
}

case "${1:-install}" in
  install) do_install ;;
  verify)  do_verify ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
