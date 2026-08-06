#include <iostream>
#include <vector>
#include "acl/acl.h"

int main() {
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtContext ctx;
    aclrtCreateContext(&ctx, 0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    int M = 32, K = 32, N = 32;

    std::vector<float> A(M*K), B(K*N), C(M*N);
    for (auto &x : A) x = 1.0f;
    for (auto &x : B) x = 2.0f;

    float *A_dev, *B_dev, *C_dev;
    aclrtMalloc((void**)&A_dev, M*K*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc((void**)&B_dev, K*N*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc((void**)&C_dev, M*N*sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);

    aclrtMemcpy(A_dev, A.data(), M*K*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(B_dev, B.data(), K*N*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    uint32_t tiling[3] = {M, K, N};
    uint32_t *tiling_dev;
    aclrtMalloc((void**)&tiling_dev, sizeof(tiling), ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(tiling_dev, tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);

    // 调用 Ascend-C kernel（名字必须和 ascendc 编译出来的符号一致）
    void* args[] = {A_dev, B_dev, C_dev, nullptr, tiling_dev};
    aclrtLaunchKernel("gemm_kernel", 1, 1, args, 0, stream);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(C.data(), C_dev, M*N*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    std::cout << "C[0] = " << C[0] << std::endl;

    return 0;
}
