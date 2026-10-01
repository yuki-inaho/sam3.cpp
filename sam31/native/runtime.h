#pragma once
#include "json.h"
#include "sam31_gguf.h"
#include "sam31_quant.h"
#include <algorithm>
#include <cstring>
#include <deque>
#include <chrono>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
namespace sam31 {
    namespace native {
        inline void require(bool value,const std::string&message) {
            if(!value)throw std::runtime_error(message);
        }
        inline size_t product(const std::vector<size_t>&shape) {
            size_t n=1;
            for(size_t s:shape) {
                require(s>0&&s<=std::numeric_limits<size_t>::max()/n,"invalid/overflowing shape");
                n*=s;
            }
            return n;
        }
        struct Config {
            size_t image=1008,patch=14,vit=1024,vit_heads=16,vit_depth=32,vit_mlp=4736;
            size_t window=24,pretrain=24,dim=256,heads=8,decoder_depth=2,ff=2048;
            size_t memory_depth=4,slots=16,down_base=4,mem_frames=7,ptr_frames=16;
            static Config smoke() {
                Config c;
                c.image=16;
                c.patch=4;
                c.vit=32;
                c.vit_heads=4;
                c.vit_mlp=148;
                c.window=2;
                c.pretrain=2;
                c.dim=32;
                c.heads=4;
                c.ff=256;
                c.down_base=1;
                return c;
            }
            size_t grid() const {
                return image/patch;
            }
            Json json() const;
            static Config from_json(const Json&j);
            void validate() const;
        };
        struct Spec {
            std::string name;
            std::vector<size_t>shape;
            bool quantizable=false;
            std::string kind;
        };
        std::vector<Spec> specs(const Config&c);
        inline size_t tensor_size(size_t h, size_t w, size_t c) {
            const size_t n = product( {
                h, w, c
            }
            );
            require(n <= (size_t(1) << 30), "tensor allocation limit");
            return n;
        }
        struct Tensor {
            size_t h=0,w=0,c=0;
            std::vector<float>v;
            Tensor()=default;
            Tensor(size_t h_,size_t w_,size_t c_,float fill=0):h(h_),w(w_),c(c_),v(tensor_size(h_,w_,c_),fill) {
            }
            size_t rows()const {
                return h*w;
            }
            float*row(size_t n) {
                return v.data()+n*c;
            }
            const float*row(size_t n)const {
                return v.data()+n*c;
            }
            bool finite()const {
                for(float a:v)if(!std::isfinite(a))return false;
                return true;
            }
        };
        struct Weight {
            std::vector<size_t>shape;
            std::vector<float>v;
        };
        class Weights {
            struct Desc {
                std::string physical,dtype;
                std::vector<size_t>shape;
            };
            gguf_file file_;
            std::map<std::string,Desc> desc_;
            std::unordered_map<std::string,std::shared_ptr<Weight>>cache_;
            std::deque<std::string>cache_order_;
            size_t bytes_=0,limit_;
            std::vector<float> float_data(const Desc&d)const;
            public: Config config;
            size_t required_count=0,restorations=0,convrot_restorations=0;
            std::set<std::string>used;
            Weights(const std::string&path,bool fixture,size_t cache_mb);
            bool fixture()const {
                return file_.is_test_fixture();
            }
            const std::string hash()const {
                return file_.source_sha256();
            }
            std::shared_ptr<Weight> get(const std::string&name);
        };
        struct Counters {
            size_t image_encoder=0,interactive_head=0,multiplex_head=0,memory_encoder=0,memory_attention=0;
            size_t linear=0,conv=0,deconv=0,sdpa=0,vit_blocks=0,two_way_blocks=0,memory_blocks=0;
            bool profile=false;
            double linear_ms=0,conv_ms=0,deconv_ms=0,attention_ms=0,norm_ms=0;
            Json json()const;
        };
        struct OperatorTimer {
            double* total;
            std::chrono::steady_clock::time_point start;
            OperatorTimer(bool enabled,double&sum):total(enabled?&sum:nullptr) {
                if(total)start=std::chrono::steady_clock::now();
            }
            ~OperatorTimer() {
                if(total)*total+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            }
        };
        Tensor add(Tensor x,const Tensor&y,float scale=1);
        Tensor add_vector(Tensor x,const std::vector<float>&v,float scale=1);
        Tensor slice_channels(const Tensor&x,size_t begin,size_t count);
        Tensor slice_rows(const Tensor&x,size_t begin,size_t count);
        Tensor concat(const std::vector<Tensor>&xs);
        Tensor gelu(Tensor x);
        Tensor relu(Tensor x);
        Tensor resize(const Tensor&x,size_t h,size_t w,bool antialias=false);
        Tensor sine_position(size_t h,size_t w,size_t c);
        Tensor random_position(const Weight&gaussian,size_t h,size_t w);
        void rope(Tensor&x,size_t heads,size_t grid,float scale=1,size_t exclude=0);
        Tensor attention(const Tensor&q,const Tensor&k,const Tensor&v,size_t heads);
        class Ops {
            public: Weights&weights;
            Counters&calls;
            Ops(Weights&w,Counters&c):weights(w),calls(c) {
            }
            Tensor linear(const Tensor&x,const std::string&p,bool bias=true);
            Tensor norm(const Tensor&x,const std::string&p,float eps=1e-5f);
            Tensor conv(const Tensor&x,const std::string&p,size_t stride=1,size_t pad=0,bool bias=true,bool depthwise=false);
            Tensor deconv(const Tensor&x,const std::string&p);
            Tensor mlp(Tensor x,const std::string&p,size_t layers=3);
            Tensor projected_attention(const Tensor&q,const Tensor&k,const Tensor&v,const std::string&p,size_t heads);
            Tensor sdpa(const Tensor&q,const Tensor&k,const Tensor&v,size_t heads) {
                OperatorTimer timer(calls.profile,calls.attention_ms);
                ++calls.sdpa;
                return attention(q,k,v,heads);
            }
        };
    }
}
