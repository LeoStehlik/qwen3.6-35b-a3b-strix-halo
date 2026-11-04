// Same-input cross-process model logits for FA_MR=0/1; no sampling or KV reuse.
#include "llama.h"
#include "llama-ext.h"
#include "json.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}


struct capture_state {
    std::string prefix;
    std::string kind;
    std::map<std::string,int> occurrences;
    std::ofstream raw, records;
    size_t offset=0;
    std::map<std::pair<std::string,int>,std::vector<float>> replay;
};
static bool trace_tensor(ggml_tensor * t, bool ask, void * user) {
    auto & s=*(capture_state *)user;
    const std::string name=t->name;
    const bool layer=name.rfind("l_out-",0)==0;
    const bool attention=name.rfind("__fattn__-",0)==0;
    const bool router=name.rfind("ffn_moe_logits-",0)==0;
    const bool residual=name.rfind("attn_residual-",0)==0;
    const bool wanted=layer || (s.kind=="stages" && (attention || router || residual));
    if(ask) return wanted;
    if(!wanted) return true;
    require(t->type==GGML_TYPE_F32,"trace expects F32 activation");
    require(t->ne[3]==1,"trace only supports single outer batch");
    int axis=attention?2:1;
    int n=t->ne[axis];
    require(n>0 && n<=1024,"unexpected trace token dimension");
    std::vector<int> rows;
    if(router) { rows.resize(n); std::iota(rows.begin(),rows.end(),0); }
    else for(int r: {0,1,31,63,127,255,511,767,1023}) if(r<n) rows.push_back(r);
    std::vector<unsigned char> all(ggml_nbytes(t));
    ggml_backend_tensor_get(t,all.data(),0,all.size());
    std::vector<float> packed;
    for(int r:rows) {
        if(axis==1) {
            require(t->ne[2]==1,"unexpected layer dimensions");
            for(int i=0;i<t->ne[0];++i) {
                float value; std::memcpy(&value,all.data()+i*t->nb[0]+r*t->nb[1],4); packed.push_back(value);
            }
        } else {
            for(int h=0;h<t->ne[1];++h) for(int i=0;i<t->ne[0];++i) {
                float value; std::memcpy(&value,all.data()+i*t->nb[0]+h*t->nb[1]+r*t->nb[2],4);packed.push_back(value);
            }
        }
    }
    for(float value:packed) require(std::isfinite(value),"nonfinite trace activation");
    int occurrence=s.occurrences[name]++;
    s.raw.write((const char *)packed.data(),packed.size()*4);
    nlohmann::json record={{"name",name},{"op",ggml_op_name(t->op)},{"occurrence",occurrence},
      {"token_axis",axis},{"ne",{t->ne[0],t->ne[1],t->ne[2],t->ne[3]}},
      {"rows",rows},{"offset",s.offset},{"floats",packed.size()}};

    if(attention && name=="__fattn__-3" && std::getenv("FA_CAPTURE_UBATCH") && occurrence==std::atoi(std::getenv("FA_CAPTURE_UBATCH"))) {
        nlohmann::json metadata;
        for(int i=0;i<5;++i) {
            ggml_tensor * x=i==4?t:t->src[i]; if(!x) continue;
            std::vector<unsigned char> bytes(ggml_nbytes(x));
            ggml_backend_tensor_get(x,bytes.data(),0,bytes.size());
            std::string suffix=i==4?"out":std::string("qkvm").substr(i,1);
            std::ofstream file(s.prefix+".deep."+suffix,std::ios::binary);
            file.write((const char *)bytes.data(),bytes.size());
            require(file.good(),"deep capture failed");
            metadata[suffix]={{"type",ggml_type_name(x->type)},{"ne",{x->ne[0],x->ne[1],x->ne[2],x->ne[3]}},{"nb",{x->nb[0],x->nb[1],x->nb[2],x->nb[3]}}};
        }
        std::ofstream file(s.prefix+".deep.json");file<<metadata.dump(2)<<'\n';
    }
    auto it=s.replay.find({name,occurrence});
    if(it!=s.replay.end()) {
        require(router && ggml_is_contiguous(t),"replay expects contiguous router tensor");
        require(it->second.size()*4==ggml_nbytes(t),"replay shape mismatch");
        ggml_backend_tensor_set(t,it->second.data(),0,it->second.size()*4);
        record["router_replayed"]=true;
    }
    s.records<<record.dump()<<'\n';s.offset+=packed.size()*4;
    require(s.raw.good() && s.records.good(),"trace write failed");
    return true;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 4, "usage: test-fa-mr-numeric model.gguf tokens.json output-prefix");
        std::ifstream input(argv[2]);
        nlohmann::json data; input >> data;
        auto tokens = data.get<std::vector<llama_token>>();
        require(tokens.size() == 65536, "expected fixed 64K prompt");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999; mp.use_mmap = false;
        auto model = llama_model_load_from_file(argv[1], mp);
        require(model, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 131072; cp.n_batch = 2048; cp.n_ubatch = 1024;
        cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = 16;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

        const int count=std::getenv("FA_NUMERIC_TOKENS")?std::atoi(std::getenv("FA_NUMERIC_TOKENS")):2048;
        require(count>0 && count<=65536 && count%2048==0,"invalid capture token count");
        capture_state capture;
        capture.prefix=argv[3];
        const char * kind=std::getenv("FA_TRACE_KIND");
        capture.kind=kind?kind:"none";
        if(capture.kind!="none") {
            require(capture.kind=="layers" || capture.kind=="stages","invalid trace kind");
            capture.raw.open(capture.prefix+".trace",std::ios::binary);
            capture.records.open(capture.prefix+".trace.jsonl");
            require(capture.raw.good() && capture.records.good(),"trace open failed");
            cp.cb_eval=trace_tensor;cp.cb_eval_user_data=&capture;
        }

        if(const char * prefix=std::getenv("FA_REPLAY_ROUTERS")) {
            require(capture.kind=="stages","router replay needs stages tracing");
            std::ifstream records(std::string(prefix)+".trace.jsonl");
            std::ifstream raw(std::string(prefix)+".trace",std::ios::binary);
            require(records.good() && raw.good(),"replay input missing");
            std::string line;
            while(std::getline(records,line)) {
                auto record=nlohmann::json::parse(line);
                std::string name=record["name"];
                if(name.rfind("ffn_moe_logits-",0)!=0) continue;
                std::vector<float> values(record["floats"].get<size_t>());
                raw.seekg(record["offset"].get<size_t>());
                raw.read((char *)values.data(),values.size()*4);
                require(raw.good(),"replay read failed");
                capture.replay[{name,record["occurrence"].get<int>()}]=std::move(values);
            }
            require(capture.replay.size()==size_t(count/1024*40),"replay record count mismatch");
        }
        auto ctx = llama_init_from_model(model, cp);
        require(ctx, "context creation failed");
        llama_set_embeddings_pre_norm(ctx, true, false);
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const int hidden = llama_model_n_embd(model);
        std::ofstream logits(std::string(argv[3])+".logits",std::ios::binary);
        std::ofstream embeddings(std::string(argv[3])+".hidden",std::ios::binary);
        require(logits.good() && embeddings.good(), "output open failed");
        auto batch = llama_batch_init(2048,0,1);
        const std::vector<int> sample_rows = {255,767,1279,2047};
        std::vector<int> positions;
        for (int start = 0; start < count; start += 2048) {
            batch.n_tokens = 2048;
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = tokens[start+i]; batch.pos[i] = start+i;
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = false;
            }
            for (int row : sample_rows) batch.logits[row] = true;
            require(llama_decode(ctx,batch)==0,"decode failed");
            const float * h = llama_get_embeddings_pre_norm(ctx);
            require(h, "hidden state missing");
            for (int row : sample_rows) {
                const float * p = llama_get_logits_ith(ctx,row);
                require(p,"selected logits missing");
                logits.write((const char *)p, vocab*sizeof(float));
                embeddings.write((const char *)(h+row*hidden),hidden*sizeof(float));
                positions.push_back(start+row);
            }
            if ((start+2048)%8192 == 0) {
                std::printf("captured through %d tokens\n",start+2048); std::fflush(stdout);
            }
        }
        require(logits.good() && embeddings.good(),"output write failed");
        std::ofstream metadata(std::string(argv[3])+".json");
        metadata << nlohmann::json({{"vocab",vocab},{"hidden",hidden},{"positions",positions}}).dump(2) << '\n';
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1;
    }
}
