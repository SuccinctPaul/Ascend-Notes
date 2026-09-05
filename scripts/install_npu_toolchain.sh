#!/usr/bin/env bash
# ============================================================================
# install_npu_toolchain.sh — 安装 Ascend NPU 开发工具链 (驱动/固件 + CANN toolkit)
#
# 仓库实测组合 (见 examples/*/README.md): Ascend 910B2 + CANN 9.0.0, aarch64 Ubuntu。
# 本脚本跑在 NPU 服务器/容器上 (Linux aarch64 / x86_64), 做三件事:
#   1. (可选) 安装本地驱动/固件 .run 包 —— 容器场景通常跳过 (驱动在宿主机)
#   2. 自动下载并安装 CANN toolkit (昇腾官方 OBS 镜像, 双架构均已验证)
#   3. source 环境并验证: ASCEND_HOME_PATH / bisheng / acl 头文件 / npu-smi
#
# 用法:
#   sudo ./install_npu_toolchain.sh                  # 安装 CANN toolkit (驱动缺失时仅提示)
#   ./install_npu_toolchain.sh --download-only       # 只下载 .run 包到 DOWNLOAD_DIR
#   sudo ./install_npu_toolchain.sh \
#        --driver   /path/Ascend-hdk-910b-npu-driver_24.1.rc3_linux-aarch64.run \
#        --firmware /path/Ascend-hdk-910b-npu-firmware_8.0.0.5.001_linux-aarch64.run
#        # 驱动/固件需登录昇腾官方渠道下载 (无公开直链):
#        #   https://www.hiascend.com/developer/download
#        # 或 https://gitee.com/ascend -> "Ascend HDK" 版本说明
#
# 可用环境变量覆盖:
#   CANN_VERSION=9.0.0          CANN toolkit 版本
#   ARCH=aarch64                目标架构 (默认取 uname -m, 支持 x86_64)
#   DOWNLOAD_DIR=/tmp/ascend_pkgs
# ============================================================================
set -euo pipefail

CANN_VERSION="${CANN_VERSION:-9.0.0}"
ARCH="${ARCH:-$(uname -m)}"
case "$ARCH" in
  aarch64|arm64) ARCH=aarch64 ;;
  x86_64)        ARCH=x86_64 ;;
  *) echo "不支持的架构: $ARCH (仅支持 aarch64 / x86_64)"; exit 1 ;;
esac
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp/ascend_pkgs}"
MIRROR_BASE="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com"
TOOLKIT_URL="$MIRROR_BASE/CANN/CANN%20${CANN_VERSION}/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"
TOOLKIT_RUN="$DOWNLOAD_DIR/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"

MODE="install"
DRIVER_RUN=""
FIRMWARE_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-only) MODE="download" ;;
    --driver)        DRIVER_RUN="$2"; shift 2 ;;
    --firmware)      FIRMWARE_RUN="$2"; shift 2 ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1 (见脚本头部用法)"; exit 1 ;;
  esac
done

info()  { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m[FAIL]\033[0m %s\n' "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# 安装必须在 NPU 服务器 (Linux) 上; 仅下载允许在任意机器 (下载后 scp 到服务器)
if [[ "$(uname -s)" != "Linux" && "$MODE" != "download" ]]; then
  fail "本脚本需在 NPU 服务器 (Linux) 上执行安装; 如仅下载 .run 包请加 --download-only"
fi

# ---- 0. 基础依赖 -----------------------------------------------------------
have curl || fail "缺少 curl, 请先安装 (apt-get install -y curl)"
mkdir -p "$DOWNLOAD_DIR"

download() { # download <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    info "已存在, 跳过下载: $dest"
  else
    info "下载: $url"
    curl -fL --retry 3 --progress-bar -o "$dest" "$url" \
      || fail "下载失败: $url (可手动下载后放到 $dest)"
  fi
}

# ---- 1. CANN toolkit 下载 --------------------------------------------------
info "== 1/3 下载 CANN toolkit $CANN_VERSION ($ARCH) =="
download "$TOOLKIT_URL" "$TOOLKIT_RUN"

[[ "$MODE" == "download" ]] && { info "download-only 模式结束, 包在 $DOWNLOAD_DIR"; exit 0; }

# ---- 2. 驱动/固件 (可选) ---------------------------------------------------
info "== 2/3 驱动/固件 =="
if have npu-smi; then
  info "npu-smi 已可用, 跳过驱动安装 (当前状态):"
  npu-smi info | head -15 || true
else
  if [[ -n "$DRIVER_RUN" ]]; then
    [[ "$(id -u)" -eq 0 ]] || fail "安装驱动需要 root (sudo)"
    [[ -f "$DRIVER_RUN" ]] || fail "驱动包不存在: $DRIVER_RUN"
    info "安装驱动: $DRIVER_RUN"
    bash "$DRIVER_RUN" --full
    [[ -n "$FIRMWARE_RUN" ]] && { info "安装固件: $FIRMWARE_RUN"; bash "$FIRMWARE_RUN" --install; }
    warn "驱动/固件安装后建议重启机器再继续 (reboot)"
  else
    warn "本机没有 npu-smi (无驱动)。两种情况:"
    warn "  · 云容器: 驱动在宿主机, 容器内无法安装 —— 正常, CANN 装完即可编译;"
    warn "    能否跑 NPU 用例取决于宿主驱动是否挂载进来 (挂载后 npu-smi 会自动可见)。"
    warn "  · 物理机: 请从 https://www.hiascend.com/developer/download 下载驱动+固件"
    warn "    .run 包后重跑: sudo $0 --driver <driver.run> --firmware <firmware.run>"
  fi
fi

# ---- 3. 安装 CANN toolkit --------------------------------------------------
info "== 3/3 安装 CANN toolkit =="
if [[ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]] \
   || [[ -f "$HOME/Ascend/ascend-toolkit/latest/version.cfg" ]]; then
  warn "检测到已有 CANN toolkit 安装, 直接复用 (如需重装请先卸载 /usr/local/Ascend)"
else
  if [[ "$(id -u)" -eq 0 ]]; then
    bash "$TOOLKIT_RUN" --install --install-for-all
  else
    bash "$TOOLKIT_RUN" --install     # 非 root 装到 $HOME/Ascend
  fi
fi

# ---- 验证 -------------------------------------------------------------------
SET_ENV=""
for p in /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
         "$HOME/Ascend/ascend-toolkit/latest/set_env.sh"; do
  [[ -f "$p" ]] && SET_ENV="$p" && break
done
[[ -n "$SET_ENV" ]] || fail "安装后找不到 set_env.sh"
# shellcheck disable=SC1090
source "$SET_ENV"

if ! grep -qs "ascend-toolkit/latest/set_env.sh" "$HOME/.bashrc"; then
  echo "source $SET_ENV" >> "$HOME/.bashrc"
  info "已把 'source $SET_ENV' 追加到 ~/.bashrc (当前 shell 仍需手动 source)"
fi

info "CANN toolkit 安装完成并验证:"
echo "  set_env.sh        : $SET_ENV"
echo "  ASCEND_HOME_PATH  : ${ASCEND_HOME_PATH:-未设置}"
if [[ -n "${ASCEND_HOME_PATH:-}" ]] && [[ -f "$ASCEND_HOME_PATH/include/acl/acl.h" ]]; then
  echo "  acl/acl.h         : OK"
else
  warn "  acl/acl.h 缺失"
fi
if have bisheng || [[ -x "${ASCEND_HOME_PATH:-/nonexistent}/bin/bisheng" ]]; then
  echo "  bisheng 编译器    : OK"
else
  warn "  bisheng 缺失 (ascend_c 无法编译)"
fi

for cmd in cmake g++ python3; do
  if have "$cmd"; then echo "  $cmd : $(command -v "$cmd")"
  else warn "  缺少 $cmd (ascend_c 构建需要; sudo apt-get install -y cmake g++ python3)"; fi
done

if have npu-smi; then
  echo "  npu-smi           : OK"
else
  warn "  npu-smi 不可用: 编译可进行, 但 NPU 运行用例需宿主机驱动 (见上文)"
fi
info "后续步骤: source $SET_ENV && ./scripts/install_dsl_envs.sh all"
