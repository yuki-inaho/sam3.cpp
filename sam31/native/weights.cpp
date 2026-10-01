#include "runtime.h"
#include "sam31_native.h"
#include <fstream>
namespace sam31 {
    namespace native {
        Json Config::json() const {
            Json j;
#define FIELD(n) j[#n]=n
            FIELD(image);
            FIELD(patch);
            FIELD(vit);
            FIELD(vit_heads);
            FIELD(vit_depth);
            FIELD(vit_mlp);
            FIELD(window);
            FIELD(pretrain);
            FIELD(dim);
            FIELD(heads);
            FIELD(decoder_depth);
            FIELD(ff);
            FIELD(memory_depth);
            FIELD(slots);
            FIELD(down_base);
            FIELD(mem_frames);
            FIELD(ptr_frames);
#undef FIELD
            return j;
        }
        Config Config::from_json(const Json&j) {
            Config c;
#define FIELD(n) c.n=j.at(#n).integer()
            FIELD(image);
            FIELD(patch);
            FIELD(vit);
            FIELD(vit_heads);
            FIELD(vit_depth);
            FIELD(vit_mlp);
            FIELD(window);
            FIELD(pretrain);
            FIELD(dim);
            FIELD(heads);
            FIELD(decoder_depth);
            FIELD(ff);
            FIELD(memory_depth);
            FIELD(slots);
            FIELD(down_base);
            FIELD(mem_frames);
            FIELD(ptr_frames);
#undef FIELD
            require(j.obj().size()==c.json().obj().size(),"unknown native configuration field");
            c.validate();
            return c;
        }
        void Config::validate()const {
            const Json values=json();
            for(const auto&kv:values.obj())require(kv.second.integer()>0&&kv.second.integer()<=8192,"configuration range: "+kv.first);
            require(image<=2048&&image%patch==0&&grid()<=144,"invalid image/patch dimensions");
            require(vit%vit_heads==0&&(vit/vit_heads)%4==0&&dim%heads==0&&(dim/heads)%4==0&&dim%8==0&&dim/2%heads==0,"invalid attention dimensions");
            require(vit_depth==32&&memory_depth==4&&decoder_depth==2&&slots==16,"native profile requires 32 ViT, 4 memory, 2 decoder layers, 16 slots");
            require(window<=grid()&&pretrain<=grid()&&mem_frames==7&&ptr_frames==16,"unsupported spatial/temporal profile");
            require(down_base<=4,"mask downsampler channel bound");
        }
        std::vector<Spec> specs(const Config&c) {
            std::vector<Spec>s;
            auto add=[&](std::string p,std::vector<size_t>sh,bool q=false,std::string kind="parameter") {
                s.push_back( {
                    std::move(p),std::move(sh),q,std::move(kind)
                }
                );
            };
            auto lin=[&](const std::string&p,size_t in,size_t out,bool bias=true) {
                add(p+".weight", {
                    out,in
                }
                ,true,"linear");
                if(bias)add(p+".bias", {
                    out
                }
                ,false,"bias");
            };
            auto norm=[&](const std::string&p,size_t d) {
                add(p+".weight", {
                    d
                }
                ,false,"norm");
                add(p+".bias", {
                    d
                }
                ,false,"bias");
            };
            auto conv=[&](const std::string&p,size_t in,size_t out,size_t k,bool bias=true) {
                add(p+".weight", {
                    out,in,k,k
                }
                ,true,"conv");
                if(bias)add(p+".bias", {
                    out
                }
                ,false,"bias");
            };
            auto deconv=[&](const std::string&p,size_t in,size_t out) {
                add(p+".weight", {
                    in,out,2,2
                }
                ,false,"deconv");
                add(p+".bias", {
                    out
                }
                ,false,"bias");
            };
            auto mlp=[&](const std::string&p,size_t in,size_t hidden,size_t out) {
                lin(p+".layers.0",in,hidden);
                lin(p+".layers.1",hidden,hidden);
                lin(p+".layers.2",hidden,out);
            };
            auto attn=[&](const std::string&p,size_t inner) {
                lin(p+".q_proj",c.dim,inner);
                lin(p+".k_proj",c.dim,inner);
                lin(p+".v_proj",c.dim,inner);
                lin(p+".out_proj",inner,c.dim);
            };
            const std::string trunk="backbone.vision_backbone.trunk";
            conv(trunk+".patch_embed.proj",3,c.vit,c.patch,false);
            add(trunk+".pos_embed", {
                1,c.pretrain*c.pretrain+1,c.vit
            }
            );
            norm(trunk+".ln_pre",c.vit);
            for(size_t i=0; i<c.vit_depth; ++i) {
                std::string p=trunk+".blocks."+std::to_string(i);
                norm(p+".norm1",c.vit);
                lin(p+".attn.qkv",c.vit,3*c.vit);
                lin(p+".attn.proj",c.vit,c.vit);
                norm(p+".norm2",c.vit);
                lin(p+".mlp.fc1",c.vit,c.vit_mlp);
                lin(p+".mlp.fc2",c.vit_mlp,c.vit);
            }
            for(const auto&branch: {
                "interactive_convs","propagation_convs"
            }
            )for(size_t level=0; level<3; ++level) {
                std::string p="backbone.vision_backbone."+std::string(branch)+"."+std::to_string(level);
                size_t in=c.vit;
                if(level==0) {
                    deconv(p+".dconv_2x2_0",c.vit,c.vit/2);
                    deconv(p+".dconv_2x2_1",c.vit/2,c.vit/4);
                    in=c.vit/4;
                }
                else if(level==1) {
                    deconv(p+".dconv_2x2",c.vit,c.vit/2);
                    in=c.vit/2;
                }
                conv(p+".conv_1x1",in,c.dim,1);
                conv(p+".conv_3x3",c.dim,c.dim,3);
            }
            add("interactivity_no_mem_embed", {
                1,1,c.dim
            }
            );
            const std::string prompt="interactive_sam_prompt_encoder";
            add(prompt+".pe_layer.positional_encoding_gaussian_matrix", {
                2,c.dim/2
            }
            );
            for(size_t i=0; i<2; ++i)add(prompt+".point_embeddings."+std::to_string(i)+".weight", {
                1,c.dim
            }
            );
            add(prompt+".not_a_point_embed.weight", {
                1,c.dim
            }
            );
            add(prompt+".no_mask_embed.weight", {
                1,c.dim
            }
            );
            add("image_pe_layer.positional_encoding_gaussian_matrix", {
                2,c.dim/2
            }
            );
            for(bool mux: {
                false,true
            }
            ) {
                const std::string p=mux?"sam_mask_decoder":"interactive_sam_mask_decoder";
                const size_t objects=mux?c.slots:1, masks=mux?3:4;
                add(p+".obj_score_token.weight", {
                    objects,c.dim
                }
                );
                add(p+".iou_token.weight", {
                    objects,c.dim
                }
                );
                add(p+".mask_tokens.weight", {
                    objects*masks,c.dim
                }
                );
                for(size_t i=0; i<c.decoder_depth; ++i) {
                    std::string t=p+".transformer.layers."+std::to_string(i);
                    attn(t+".self_attn",c.dim);
                    attn(t+".cross_attn_token_to_image",c.dim/2);
                    attn(t+".cross_attn_image_to_token",c.dim/2);
                    for(int n=1; n<=4; ++n)norm(t+".norm"+std::to_string(n),c.dim);
                    lin(t+".mlp.lin1",c.dim,c.ff);
                    lin(t+".mlp.lin2",c.ff,c.dim);
                }
                attn(p+".transformer.final_attn_token_to_image",c.dim/2);
                norm(p+".transformer.norm_final_attn",c.dim);
                deconv(p+".output_upscaling.0",c.dim,c.dim/4);
                norm(p+".output_upscaling.1",c.dim/4);
                deconv(p+".output_upscaling.3",c.dim/4,c.dim/8);
                conv(p+".conv_s0",c.dim,c.dim/8,1);
                conv(p+".conv_s1",c.dim,c.dim/4,1);
                for(size_t i=0; i<masks; ++i)mlp(p+".output_hypernetworks_mlps."+std::to_string(i),c.dim,c.dim,c.dim/8);
                mlp(p+".iou_prediction_head",c.dim,c.dim,masks);
                mlp(p+".pred_obj_score_head",c.dim,c.dim,1);
            }
            mlp("interactive_obj_ptr_proj",c.dim,c.dim,c.dim);
            mlp("obj_ptr_proj",c.dim,c.dim,c.dim);
            lin("no_obj_ptr_linear",c.dim,c.dim);
            lin("obj_ptr_tpos_proj",c.dim,c.dim);
            add("output_valid_embed", {
                c.slots,c.dim
            }
            );
            add("output_invalid_embed", {
                c.slots,c.dim
            }
            );
            add("no_obj_embed_spatial", {
                c.slots,c.dim
            }
            );
            add("maskmem_tpos_enc", {
                c.mem_frames,1,1,c.dim
            }
            );
            const std::string me="maskmem_backbone";
            size_t in=2*c.slots,out=c.down_base;
            for(size_t i=0; i<4; ++i) {
                out*=4;
                std::string p=me+".mask_downsampler.encoder.";
                conv(p+std::to_string(3*i),in,out,3);
                norm(p+std::to_string(3*i+1),out);
                in=out;
            }
            conv(me+".mask_downsampler.encoder.12",in,c.dim,1);
            conv(me+".pix_feat_proj",c.dim,c.dim,1);
            for(size_t i=0; i<2; ++i) {
                std::string p=me+".fuser.layers."+std::to_string(i);
                conv(p+".dwconv",1,c.dim,7);
                norm(p+".norm",c.dim);
                lin(p+".pwconv1",c.dim,4*c.dim);
                lin(p+".pwconv2",4*c.dim,c.dim);
                add(p+".gamma", {
                    c.dim
                }
                ,false,"gamma");
            }
            for(size_t i=0; i<c.memory_depth; ++i) {
                std::string p="transformer.encoder.layers."+std::to_string(i);
                for(auto suffix: {
                    "self_attn_q_proj","self_attn_k_proj","self_attn_v_proj","self_attn_out_proj","cross_attn_q_proj","cross_attn_k_proj","cross_attn_v_proj","cross_attn_out_proj","image_cross_attn_q_proj","image_cross_attn_k_proj"
                }
                )lin(p+"."+suffix,c.dim,c.dim);
                for(int n=1; n<=3; ++n)norm(p+".norm"+std::to_string(n),c.dim);
                lin(p+".linear1",c.dim,c.ff);
                lin(p+".linear2",c.ff,c.dim);
            }
            norm("transformer.encoder.norm",c.dim);
            std::set<std::string>names;
            for(const auto&w:s)require(names.insert(w.name).second,"duplicate native specification");
            return s;
        }
        std::string manifest_json(bool smoke) {
            Config c=smoke?Config::smoke():Config {
            };
            c.validate();
            Json j;
            j["config"]=c.json();
            j["test_fixture"]=smoke;
            Json::array a;
            for(const auto&w:specs(c)) {
                Json d;
                d["name"]=w.name;
                Json::array sh;
                for(auto v:w.shape)sh.emplace_back(v);
                d["shape"]=sh;
                d["quantizable"]=w.quantizable;
                d["kind"]=w.kind;
                a.push_back(d);
            }
            j["tensors"]=a;
            return j.dump();
        }
        Weights::Weights(const std::string&path,bool fixture,size_t cache_mb):file_(path,fixture),limit_(cache_mb*1024*1024) {
            require(cache_mb>0&&cache_mb<=4096,"cache must be 1..4096 MiB");
            auto meta=Json::parse(file_.source_metadata_json());
            if(file_.is_test_fixture()) {
                require(meta.has("sam31.native.config"),"test GGUF is not a native graph fixture");
                config=Config::from_json(Json::parse(meta.at("sam31.native.config").str()));
            }
            else {
                require(!meta.has("sam31.native.config"),"custom dimensions are forbidden for trained model mode");
                config.validate();
            }
            auto mapping=Json::parse(file_.tensor_map_json());
            std::map<std::string,tensor_descriptor>physical;
            for(const auto&t:file_.tensors())require(physical.emplace(t.name,t).second,"duplicate physical tensor name");
            std::map<std::string,Desc>all;
            for(const auto&kv:mapping.obj()) {
                const auto&t=kv.second;
                Desc d;
                d.physical=kv.first;
                d.dtype=t.at("dtype").str();
                for(const auto&x:t.at("shape").arr())d.shape.push_back(x.integer());
                const auto it=physical.find(kv.first);
                require(it!=physical.end(),"tensor map refers to absent payload");
                const std::map<std::string,std::pair<int,size_t>>types= {
                    {
                        "F32", {
                            0,4
                        }
                    }
                    , {
                        "F16", {
                            1,2
                        }
                    }
                    , {
                        "BF16", {
                            30,2
                        }
                    }
                    , {
                        "I8", {
                            24,1
                        }
                    }
                    , {
                        "U8", {
                            24,1
                        }
                    }
                    , {
                        "BOOL", {
                            24,1
                        }
                    }
                    , {
                        "I16", {
                            25,2
                        }
                    }
                    , {
                        "I32", {
                            26,4
                        }
                    }
                    , {
                        "I64", {
                            27,8
                        }
                    }
                    , {
                        "F64", {
                            28,8
                        }
                    }
                };
                auto typ=types.find(d.dtype);
                require(typ!=types.end(),"unknown logical dtype");
                size_t n=product(d.shape);
                require(n<=std::numeric_limits<size_t>::max()/typ->second.second&&n*typ->second.second==it->second.size&&typ->second.first==it->second.type,"tensor dtype/shape byte count mismatch");
                require(all.emplace(t.at("name").str(),std::move(d)).second,"duplicate logical tensor name");
            }
            require(mapping.obj().size()==physical.size(),"incomplete tensor map");
            for(const auto&spec:specs(config)) {
                std::vector<std::string>aliases= {
                    spec.name,"tracker."+spec.name,"tracker.model."+spec.name
                };
                if(spec.name.rfind("backbone.",0)==0)aliases.insert(aliases.begin(),"detector."+spec.name);
                if(spec.name.find("sam_mask_decoder.transformer.")!=std::string::npos) {
                    auto renamed=spec.name;
                    for(const auto&pair: std::vector<std::pair<std::string,std::string>>{{".mlp.lin1.",".mlp.0."},{".mlp.lin2.",".mlp.2."},{".norm_final_attn.",".norm_final."}}) {
                        auto pos=renamed.find(pair.first);
                        if(pos!=std::string::npos)renamed.replace(pos,pair.first.size(),pair.second);
                    }
                    if(renamed!=spec.name) {
                        aliases.push_back(renamed);
                        aliases.push_back("tracker."+renamed);
                        aliases.push_back("tracker.model."+renamed);
                    }
                }
                std::string chosen;
                for(const auto&wrapper: {
                    "","model.","diffusion_model.","model.diffusion_model."
                }
                )for(const auto&alias:aliases) {
                    std::string name=std::string(wrapper)+alias;
                    if(!all.count(name))continue;
                    if(chosen.empty())chosen=name;
                    else if(chosen!=name) {
                        const auto&a=all.at(chosen);
                        const auto&b=all.at(name);
                        require(a.dtype==b.dtype&&a.shape==b.shape&&file_.read_tensor(a.physical)==file_.read_tensor(b.physical),"ambiguous checkpoint aliases for "+spec.name);
                        for(const auto&suffix: {
                            ".weight_scale",".comfy_quant"
                        }
                        )if(a.dtype=="I8") {
                            std::string l=chosen.substr(0,chosen.size()-7)+suffix,r=name.substr(0,name.size()-7)+suffix;
                            require(all.count(l)&&all.count(r)&&file_.read_tensor(all.at(l).physical)==file_.read_tensor(all.at(r).physical),"ambiguous quantized alias");
                        }
                    }
                }
                require(!chosen.empty(),"missing required SAM 3.1 tensor: "+spec.name);
                Desc d=all.at(chosen);
                require(d.shape==spec.shape,"shape mismatch: "+spec.name);
                require(d.dtype=="F32"||d.dtype=="F16"||d.dtype=="BF16"||d.dtype=="I8","unsupported learned weight dtype: "+spec.name);
                desc_[spec.name]=d;
                if(d.dtype=="I8") {
                    require(spec.quantizable || (spec.shape.size()==2 && spec.name.size()>=7 && spec.name.substr(spec.name.size()-7)==".weight"),"INT8 unsupported for this parameter: "+spec.name);
                    std::string source_base=chosen.substr(0,chosen.size()-7),target_base=spec.name.substr(0,spec.name.size()-7);
                    for(auto suffix: {
                        ".weight_scale",".comfy_quant"
                    }
                    ) {
                        require(all.count(source_base+suffix),"missing quantization tensor: "+source_base+suffix);
                        desc_[target_base+suffix]=all.at(source_base+suffix);
                    }
                }
                ++required_count;
            }
        }
        std::vector<float> Weights::float_data(const Desc&d)const {
            auto bytes=file_.read_tensor(d.physical);
            size_t n=product(d.shape);
            std::vector<float>out(n);
            require(d.dtype=="F32"||d.dtype=="F16"||d.dtype=="BF16","expected floating tensor");
            for(size_t i=0; i<n; ++i) {
                float x;
                if(d.dtype=="F32") {
                    uint32_t u=uint32_t(bytes[4*i])|(uint32_t(bytes[4*i+1])<<8)|(uint32_t(bytes[4*i+2])<<16)|(uint32_t(bytes[4*i+3])<<24);
                    std::memcpy(&x,&u,4);
                }
                else {
                    uint16_t u=uint16_t(bytes[2*i])|(uint16_t(bytes[2*i+1])<<8);
                    if(d.dtype=="BF16") {
                        uint32_t b=uint32_t(u)<<16;
                        std::memcpy(&x,&b,4);
                    }
                    else {
                        unsigned e=(u>>10)&31,m=u&1023;
                        float a=e==0?std::ldexp(float(m),-24):e==31?(m?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity()):std::ldexp(float(1024+m),int(e)-25);
                        x=(u&0x8000)?-a:a;
                    }
                }
                require(std::isfinite(x),"non-finite model value in "+d.physical);
                out[i]=x;
            }
            return out;
        }
        std::shared_ptr<Weight> Weights::get(const std::string&name) {
            used.insert(name);
            auto hit=cache_.find(name);
            if(hit!=cache_.end())return hit->second;
            auto it=desc_.find(name);
            require(it!=desc_.end(),"unregistered model parameter: "+name);
            const Desc&d=it->second;
            auto w=std::make_shared<Weight>();
            w->shape=d.shape;
            if(d.dtype!="I8")w->v=float_data(d);
            else {
                std::string base=name.substr(0,name.size()-7);
                const Desc&qdesc=desc_.at(base+".comfy_quant");
                require(qdesc.dtype=="U8"||qdesc.dtype=="I8","invalid quantization descriptor type");
                auto qb=file_.read_tensor(qdesc.physical);
                auto q=Json::parse(std::string(qb.begin(),qb.end()));
                require(q.at("format").str()=="int8_tensorwise","unsupported INT8 format");
                for(const auto&kv:q.obj())require(kv.first=="format"||kv.first=="convrot"||kv.first=="convrot_groupsize","unknown quantization field");
                bool rotate=q.has("convrot")&&q.at("convrot").boolean();
                size_t group=rotate?q.at("convrot_groupsize").integer():0;
                require(rotate||!q.has("convrot_groupsize"),"group size on non-rotated weights");
                auto scale=float_data(desc_.at(base+".weight_scale"));
                auto raw=file_.read_tensor(d.physical);
                w->v.resize(product(d.shape));
                int rc;
                if(d.shape.size()==2)rc=sam31_dequantize_linear(reinterpret_cast<const int8_t*>(raw.data()),raw.size(),scale.data(),scale.size(),d.shape[0],d.shape[1],group,w->v.data(),w->v.size());
                else {
                    require(d.shape.size()==4,"unsupported INT8 tensor rank");
                    rc=sam31_dequantize_conv2d(reinterpret_cast<const int8_t*>(raw.data()),raw.size(),scale.data(),scale.size(),d.shape[0],d.shape[1],d.shape[2],d.shape[3],group,w->v.data(),w->v.size());
                }
                require(rc==0,"ConvRot restoration rejected "+name);
                ++restorations;
                if(rotate)++convrot_restorations;
            }
            size_t sz=w->v.size()*sizeof(float);
            while(!cache_order_.empty()&&bytes_+sz>limit_) {
                auto key=cache_order_.front();
                cache_order_.pop_front();
                auto old=cache_.find(key);
                if(old!=cache_.end()) {
                    bytes_-=old->second->v.size()*sizeof(float);
                    cache_.erase(old);
                }
            }
            if(sz<=limit_) {
                cache_.emplace(name,w);
                cache_order_.push_back(name);
                bytes_+=sz;
            }
            return w;
        }
        Json Counters::json()const {
            Json j;
#define FIELD(n) j[#n]=n
            FIELD(image_encoder);
            FIELD(interactive_head);
            FIELD(multiplex_head);
            FIELD(memory_encoder);
            FIELD(memory_attention);
            FIELD(linear);
            FIELD(conv);
            FIELD(deconv);
            FIELD(sdpa);
            FIELD(vit_blocks);
            FIELD(two_way_blocks);
            FIELD(memory_blocks);
#undef FIELD
            return j;
        }
    }
}
