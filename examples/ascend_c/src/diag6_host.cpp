// Host for diag6: load tiling constants [N=8, pad, CBIG=1.5958, CCUB=0.044715, CONE=1.0, zeros...], print y.
#include <iostream>
#include <cmath>
#include <cstdint>
#include "acl/acl.h"
using half_t = __fp16;
extern "C" int aclrtlaunch_diag6_kernel(uint32_t, aclrtStream, void*, void*, void*, void*);
static void check(const char* w, aclError e){ if(e!=ACL_ERROR_NONE){std::cerr<<"[ACL] "<<w<<": code="<<int(e)<<"\n"; std::exit(1);} }

struct alignas(8) Tiling { uint32_t N; uint32_t pad; float cf[8]; };

int main(){
    check("init", aclInit(nullptr)); check("dev", aclrtSetDevice(0));
    aclrtStream s; check("cs", aclrtCreateStream(&s));
    Tiling t{}; t.N = 8u; t.cf[0]=1.5957691216057308f; t.cf[1]=0.044715f; t.cf[2]=1.0f;
    void *dx=nullptr, *dy=nullptr, *dt=nullptr;
    check("Mx",aclrtMalloc(&dx,256,ACL_MEM_MALLOC_HUGE_FIRST));
    check("My",aclrtMalloc(&dy,256,ACL_MEM_MALLOC_HUGE_FIRST));
    check("Mt",aclrtMalloc(&dt,sizeof(t),ACL_MEM_MALLOC_HUGE_FIRST));
    check("H2D",aclrtMemcpy(dt,sizeof(t),&t,sizeof(t),ACL_MEMCPY_HOST_TO_DEVICE));
    int rc=aclrtlaunch_diag6_kernel(1u,s,dx,dy,nullptr,dt);
    if(rc!=0){std::cerr<<"launch rc="<<rc<<"\n";return 2;}
    check("sync",aclrtSynchronizeStream(s));
    half_t y[16]={};
    check("D2H",aclrtMemcpy(y,32,dy,32,ACL_MEMCPY_DEVICE_TO_HOST));
    float expc[]={1.5957691f,0.044715f,1.0f,0,0,0,0,0,1.5957691f,0.0f};
    for(int i=0;i<10;++i) printf("y[%2d]=%-12.7f  expected=%-12.7f\n",i,(float)y[i],expc[i]);
    aclrtFree(dx);aclrtFree(dy);aclrtFree(dt);
    aclrtDestroyStream(s); aclrtResetDevice(0); aclFinalize();
    return 0;
}
