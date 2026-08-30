cmake_minimum_required(VERSION 3.16.0)
project(ascend_gemm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- CANN 环境变量 ----
# 使用前必须先 source CANN 的 set_env.sh, 它会设置 ASCEND_HOME_PATH
# (例如 /usr/local/Ascend/cann-9.0.0)
set(ASCEND_HOME $ENV{ASCEND_HOME_PATH})
if(NOT ASCEND_HOME)
    message(FATAL_ERROR
        "ASCEND_HOME_PATH 未设置, 请先 source CANN 的 set_env.sh:\n"
        "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh")
endif()

# CMAKE_BUILD_TYPE 必须设置: ascendc.cmake 的 merge_device_obj.py 需要 --build-type 参数
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- SoC 型号 (可被 -D 覆盖) ----
# 远程 NPU: npu-smi 显示 910B2。
# ascendc.cmake 的 host_config.cmake 把 ascend910b* 家族 → BUILD_MODE=c220
set(SOC_VERSION "Ascend910B2" CACHE STRING
    "Ascend SoC version (Ascend910B2, Ascend310B1, Ascend310P3, ...)")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_HOME})

# ---- 引入官方 Ascend-C 构建框架 (ascendc.cmake) ----
# ascendc.cmake 提供 ascendc_library() 函数, 它自动完成整套构建+打包流程:
#
#   1. bisheng -E 预处理 kernel 源码 → 提取 kernel 函数签名
#   2. bisheng 编译 device kernel → device_aic.o / device_aiv.o (按 Cube/Vector 分类)
#   3. extract_host_stub.py 生成 host_stub.cpp (含 device 二进制嵌入 + launch stub)
#   4. update_host_stub.py 补丁 host_stub.cpp (填充 SoC 专属符号, 如 .ascend.kernel.ascend910b2.gemm)
#   5. g++ 编译 host_stub.cpp → host_stub.o
#   6. ascendc_pack_kernel 把 device.o 合并进 host_stub.o → 可被 ACL 运行时识别的二进制
#   7. ar 打包 → lib/libgemm.a
#
# 最终 libgemm.a 导出: aclrtlaunch_gemm_kernel(numBlocks, stream, a, b, c, workspace, tiling)
# —— host 侧只需调用此函数即可启动 kernel, 无需手动 aclrtBinaryLoadFromFile 等。
#
# 路径: $ASCEND_HOME/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake
include(${ASCEND_HOME}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# ---- 构建 kernel 静态库 ----
# 必须用 STATIC 而非 OBJECT:
#   - STATIC: 经过 link 步骤, 触发 PRE_LINK 阶段的 ascendc_pack_kernel 打包
#   - OBJECT: 不经过 link, pack 步骤不会执行 (产出无法被 ACL 运行时加载)
ascendc_library(gemm STATIC ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/gemm_kernel.cpp)

# ---- host 侧可执行文件 ----
# 链接 gemm (内含 aclrtlaunch_gemm_kernel + 打包好的 device 二进制)
#   ascendcl : ACL 主库 (aclInit / aclrtMalloc / aclrtMemcpy / aclrtSynchronizeStream 等)
#   runtime  : 底层 rt 库 (LaunchAscendKernel 等运行时符号)
add_executable(ascend_gemm src/gemm_host.cpp)
target_compile_options(ascend_gemm PRIVATE -fno-fast-math)
target_link_libraries(ascend_gemm PRIVATE gemm ascendcl runtime)

# 运行: cd build && ./ascend_gemm
