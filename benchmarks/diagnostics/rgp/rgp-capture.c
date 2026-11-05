#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
// Redirect only Mesa RGP output; avoid its same-second file overwrite.
FILE *fopen(const char *path,const char *mode){
 FILE *(*real_open)(const char *,const char *)=dlsym(RTLD_NEXT,"fopen");
 const char *dir=getenv("RGP_OUTPUT_DIR");size_t len=strlen(path);
 if(dir && len>4 && !strncmp(path,"/tmp/",5) && !strcmp(path+len-4,".rgp")){
  static _Atomic unsigned int serial;char dst[4096];const char *phase=getenv("RGP_KEEP_PHASE");
  snprintf(dst,sizeof(dst),"%s/%s-%u-%u.rgp",dir,phase?phase:"setup",(unsigned)getpid(),atomic_fetch_add(&serial,1));return real_open(dst,mode);
 }
 return real_open(path,mode);
}
FILE *fopen64(const char *path,const char *mode){return fopen(path,mode);}
#include <vulkan/vulkan.h>
// Mesa 26.0.8 per-submit tracing omits the actual submission when count==0,
// so the caller's fence is never signaled. Send one empty VkSubmitInfo instead.
// Diagnostic process only: GPU work, kernels and production loader unchanged.
static PFN_vkQueueSubmit next_submit;
static PFN_vkQueueSubmit2 next_submit2;
VKAPI_ATTR VkResult VKAPI_CALL capture_submit(VkQueue q,uint32_t n,const VkSubmitInfo *s,VkFence f){
 if(n==0 && f){VkSubmitInfo empty={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO};fprintf(stderr,"RGP_COMPAT empty submit fence preserved\n");return next_submit(q,1,&empty,f);}
 return next_submit(q,n,s,f);
}
VKAPI_ATTR VkResult VKAPI_CALL capture_submit2(VkQueue q,uint32_t n,const VkSubmitInfo2 *s,VkFence f){
 if(n==0 && f){VkSubmitInfo2 empty={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO_2};fprintf(stderr,"RGP_COMPAT empty submit2 fence preserved\n");return next_submit2(q,1,&empty,f);}
 return next_submit2(q,n,s,f);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice d,const char *name){
 PFN_vkGetDeviceProcAddr fn=dlsym(RTLD_NEXT,"vkGetDeviceProcAddr");PFN_vkVoidFunction p=fn(d,name);
 if(!strcmp(name,"vkQueueSubmit")){next_submit=(PFN_vkQueueSubmit)p;return (PFN_vkVoidFunction)capture_submit;}
 if(!strcmp(name,"vkQueueSubmit2") || !strcmp(name,"vkQueueSubmit2KHR")){next_submit2=(PFN_vkQueueSubmit2)p;return (PFN_vkVoidFunction)capture_submit2;}
 return p;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance i,const char *name){
 PFN_vkGetInstanceProcAddr fn=dlsym(RTLD_NEXT,"vkGetInstanceProcAddr");
 if(!strcmp(name,"vkGetInstanceProcAddr"))return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
 if(!strcmp(name,"vkGetDeviceProcAddr"))return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
 PFN_vkVoidFunction p=fn(i,name);
 if(!strcmp(name,"vkQueueSubmit")){next_submit=(PFN_vkQueueSubmit)p;return (PFN_vkVoidFunction)capture_submit;}
 if(!strcmp(name,"vkQueueSubmit2") || !strcmp(name,"vkQueueSubmit2KHR")){next_submit2=(PFN_vkQueueSubmit2)p;return (PFN_vkVoidFunction)capture_submit2;}
 return p;
}
