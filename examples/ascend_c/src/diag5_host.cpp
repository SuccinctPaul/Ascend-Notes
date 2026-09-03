// Host for diag5 v3: 打印常数检查
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <random>
#include "acl/acl.h"
using half_t = __fp16;
extern "C" int aclrtlaunch_diag5_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
static void check(const char* w, aclError e){ if(e!=ACL_ERROR_NONE){std::cerr<<"[ACL] "<<w<<": code="<<int(e)<<"\n"; std::exit(1);} }
int main(){
    check("init", aclInit(nullptr));
    check("setDevice", aclrtSetDevice(0));
    aclrtStream s; check("createStream", aclrtCreateStream(&s));
    void *dx=nullptr, *dy=nullptr;
    uint32_t tiling[1]={8u};
    void* dt=nullptr;
    check("Mx",aclrtMalloc(&dx,1024,ACL_MEM_MALLOC_HUGE_FIRST));
    check("My",aclrtMalloc(&dy,1024,ACL_MEM_MALLOC_HUGE_FIRST));
    check("Mt",aclrtMalloc(&dt,sizeof(tiling),ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2Dt",aclrtMemcpy(dt,sizeof(tiling),tiling,sizeof(tiling),ACL_MEMCPY_HOST_TO_DEVICE));
    int rc=aclrtlaunch_diag5_kernel(1u,s,dx,dy,nullptr,dt);
    if(rc!=0){std::cerr<<"launch rc="<<rc<<"\n"; return 2;}
    check("sync",aclrtSynchronizeStream(s));
    half_t y[16]={};
    check("D2H",aclrtMemcpy(y,32,dy,32,ACL_MEMCPY_DEVICE_TO_HOST));
    float cbig=1.5957691216057308f, ccub=0.044715f, cone=1.0f, m1=-1.0f, m3=-3.5f, zero=0.0f, ninf=-1e20f;
    printf("slot   expected(hw_fp16->fp32)      actual\n");
    auto prn=[&](int i, float ex){ float a=static_cast<float>(y[i]); printf("y[%d]  %-24.7f  %.7f (match=%d)\n",i,ex,a, (std::fabs(ex-a)<1e-3f)); };
    prn(0, cbig);  prn(1, ccub);  prn(2, cone);  prn(3, m1);  prn(4, m3);
    prn(5, zero);  prn(6, ninf);
    printf("y[7]=%f  expected=8\n", (float)y[7]);
    aclrtFree(dx); aclrtFree(dy); aclrtFree(dt);
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
