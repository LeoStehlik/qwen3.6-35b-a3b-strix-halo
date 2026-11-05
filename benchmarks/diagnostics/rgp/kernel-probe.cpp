// Diagnostic-only synthetic graphs linked to the unchanged serving libraries.
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool x,const char *s){if(!x)throw std::runtime_error(s);}
static void init_tensor(ggml_tensor *t) {
    if(t->view_src)return;
    if(t->type==GGML_TYPE_F32){std::vector<float>x(ggml_nelements(t));for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+19)%127)-63)/128;ggml_backend_tensor_set(t,x.data(),0,ggml_nbytes(t));}
    else if(t->type==GGML_TYPE_F16){std::vector<ggml_fp16_t>x(ggml_nelements(t));for(size_t i=0;i<x.size();++i)x[i]=ggml_fp32_to_fp16(float(int((i*31+7)%127)-63)/128);ggml_backend_tensor_set(t,x.data(),0,ggml_nbytes(t));}
    else if(t->type==GGML_TYPE_Q4_0_ROCMFP4_FAST){std::vector<uint8_t>x(ggml_nbytes(t));uint32_t rng=0x73653821;for(size_t b=0;b<x.size()/17;++b){for(int j=0;j<16;++j){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x[b*17+j]=uint8_t(rng);}x[b*17+16]=64+(b%16);}ggml_backend_tensor_set(t,x.data(),0,x.size());}
}
int main(int argc,char **argv){
    require(argc>=2,"mode required");std::string mode=argv[1];
    const char * backend_dir = std::getenv("GGML_BACKEND_DIR"); require(backend_dir, "Set GGML_BACKEND_DIR to the tested build/bin directory");
    ggml_backend_load_all_from_path(backend_dir);auto backend=ggml_backend_init_by_name("Vulkan0",nullptr);require(backend,"Vulkan unavailable");
    ggml_init_params ip={ggml_tensor_overhead()*64+ggml_graph_overhead(),nullptr,true};auto ctx=ggml_init(ip);
    ggml_tensor *out=nullptr,*ids=nullptr,*mask=nullptr;int n=0;std::vector<int> counts;
    if(mode=="attention"){
        require(argc==4,"attention Q KV");int qn=std::stoi(argv[2]),kv=std::stoi(argv[3]);
        auto q=ggml_permute(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F32,256,16,qn,1),0,2,1,3);
        auto k=ggml_permute(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F16,256,2,kv,1),0,2,1,3);
        auto v=ggml_permute(ctx,ggml_new_tensor_4d(ctx,GGML_TYPE_F16,256,2,kv,1),0,2,1,3);
        mask=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,kv,qn,1,1);
        out=ggml_flash_attn_ext(ctx,q,k,v,mask,1.0f/16,0,0);ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
    }else if(mode=="expert"){
        require(argc==5,"expert M K ROWS-or-countfile");int m=std::stoi(argv[2]),k=std::stoi(argv[3]);std::string r=argv[4];
        counts.resize(256);if(r.find('/')!=std::string::npos){std::ifstream f(r);for(int &x:counts)require(bool(f>>x),"count file");}else std::fill(counts.begin(),counts.end(),std::stoi(r));
        int sum=0;for(int x:counts)sum+=x;require(sum%8==0,"route population");n=sum/8;
        auto a=ggml_new_tensor_3d(ctx,GGML_TYPE_Q4_0_ROCMFP4_FAST,k,m,256);
        auto b=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,1,n);ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,8,n);out=ggml_mul_mat_id(ctx,a,b,ids);
    }else if(mode=="dense"){
        require(argc==5,"dense M K N");int m=std::stoi(argv[2]),k=std::stoi(argv[3]);n=std::stoi(argv[4]);
        out=ggml_mul_mat(ctx,ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_0_ROCMFP4_FAST,k,m),ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n));
    }else throw std::runtime_error("unknown mode");
    ggml_set_name(out,mode.c_str());auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buf,"allocation");ggml_backend_buffer_clear(buf,0);
    for(auto t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t))if(t->op==GGML_OP_NONE)init_tensor(t);
    if(mask){int kv=mask->ne[0],qn=mask->ne[1];std::vector<ggml_fp16_t>x(ggml_nelements(mask),ggml_fp32_to_fp16(0));for(int q=0;q<qn;++q)for(int j=kv-qn+q+1;j<kv;++j)x[q*size_t(kv)+j]=ggml_fp32_to_fp16(-INFINITY);ggml_backend_tensor_set(mask,x.data(),0,ggml_nbytes(mask));}
    if(ids){std::vector<int> remain=counts;std::vector<int32_t> routes(n*8),order(256);for(int i=0;i<256;++i)order[i]=i;for(int token=0;token<n;++token){std::stable_sort(order.begin(),order.end(),[&](int a,int b){return remain[a]>remain[b];});for(int slot=0;slot<8;++slot){int e=order[slot];require(remain[e]>0,"routing degree");routes[token*8+slot]=e;--remain[e];}}for(int x:remain)require(x==0,"route ownership");ggml_backend_tensor_set(ids,routes.data(),0,ggml_nbytes(ids));}
    for(int run=0;run<2;++run){setenv("RGP_KEEP_PHASE",run==0?"warm":"measured",1);require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");ggml_backend_synchronize(backend);}
    setenv("RGP_KEEP_PHASE","readback",1);std::vector<float> result(ggml_nelements(out));ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out));
    size_t finite=0;for(float x:result)finite+=std::isfinite(x);require(finite==result.size(),"nonfinite output");printf("PASS %s finite=%zu\n",mode.c_str(),finite);
    ggml_backend_buffer_free(buf);ggml_free(ctx);ggml_backend_free(backend);
}
