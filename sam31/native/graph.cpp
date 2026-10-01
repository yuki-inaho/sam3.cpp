// SAM 3.1 point-prompted multiplex tracker, native eager CPU execution.
// Tensor names and operators follow facebookresearch/sam3 at
// 2345a4ad109ac29c569da749c91d84f10dc08c40. See docs/NATIVE.ja.md.
#include "runtime.h"
#include "sam31_native.h"
#include <array>
#include <mutex>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace sam31 {
    namespace native {
        struct Model::Impl {
            Options options;
            Weights weights;
            std::mutex mutex;
            Impl(const std::string&path,const Options&o):options(o),weights(path,o.allow_test_fixture,o.weight_cache_mb) {
                require(o.threads>=1&&o.threads<=256,"threads must be 1..256");
                require(o.total_frames>0,"total_frames must be positive");
#ifndef _OPENMP
                require(o.threads == 1, "this build has no OpenMP; use --threads 1");
#endif
            }
        };
        namespace {
            Tensor embedding(const std::shared_ptr<Weight>&w) {
                require(w->shape.size()>=2,"embedding rank");
                Tensor x(product(w->shape)/w->shape.back(),1,w->shape.back());
                x.v=w->v;
                return x;
            }
            struct Features {
                std::array<Tensor,3>interactive,propagation;
            };
            struct Prediction {
                Tensor masks,pointers;
                std::vector<float>scores,quality;
            };
            struct Memory {
                size_t index;
                Tensor image,encoded,pointers;
            };
            class Graph {
                Model::Impl&m_;
                Weights&w_;
                const Config&c_;
                Counters&calls_;
                Ops op_;
                public: Graph(Model::Impl&m,Counters&calls):m_(m),w_(m.weights),c_(w_.config),calls_(calls),op_(w_,calls) {
                    calls_.profile=m_.options.profile;
#ifdef _OPENMP
                    omp_set_dynamic(0);
                    omp_set_num_threads(m_.options.threads);
#endif
                }
                Tensor preprocess(const Image&image) {
                    require(image.width&&image.height&&image.width<=16384&&image.height<=16384,"image dimensions must be 1..16384");
                    require(image.width<=size_t(1)<<26&&image.height<=((size_t(1)<<26)/image.width),"input image exceeds 64 megapixels");
                    require(image.rgb.size()==product( {
                        image.height,image.width,3
                    }
                    ),"RGB byte count mismatch");
                    Tensor x(image.height,image.width,3);
                    for(size_t i=0; i<x.v.size(); ++i)x.v[i]=image.rgb[i]/255.0f;
                    x=resize(x,c_.image,c_.image,true);
                    for(float&v:x.v)v=2*v-1;
                    return x;
                }
                Tensor vit_attention(const Tensor&x,const std::string&p,size_t grid) {
                    Tensor qkv=op_.linear(x,p+".qkv"),q=slice_channels(qkv,0,c_.vit),k=slice_channels(qkv,c_.vit,c_.vit),v=slice_channels(qkv,2*c_.vit,c_.vit);
                    float scale=float(c_.pretrain)/grid;
                    rope(q,c_.vit_heads,grid,scale);
                    rope(k,c_.vit_heads,grid,scale);
                    return op_.linear(op_.sdpa(q,k,v,c_.vit_heads),p+".proj");
                }
                Features encode(const Image&image,bool interactive) {
                    ++calls_.image_encoder;
                    const std::string trunk="backbone.vision_backbone.trunk";
                    Tensor x=op_.conv(preprocess(image),trunk+".patch_embed.proj",c_.patch,0,false);
                    auto pos=w_.get(trunk+".pos_embed");
                    for(size_t y=0; y<x.h; ++y)for(size_t z=0; z<x.w; ++z) {
                        size_t from=1+(y%c_.pretrain)*c_.pretrain+z%c_.pretrain;
                        for(size_t d=0; d<x.c; ++d)x.row(y*x.w+z)[d]+=pos->v[from*x.c+d];
                    }
                    x=op_.norm(x,trunk+".ln_pre",1e-6f);
                    for(size_t i=0; i<c_.vit_depth; ++i) {
                        ++calls_.vit_blocks;
                        std::string p=trunk+".blocks."+std::to_string(i);
                        Tensor n=op_.norm(x,p+".norm1",1e-6f),a(x.h,x.w,x.c);
                        if((i+1)%8==0)a=vit_attention(n,p+".attn",x.w);
                        else {
                            size_t win=c_.window;
                            for(size_t y=0; y<x.h; y+=win)for(size_t z=0; z<x.w; z+=win) {
                                Tensor tile(win,win,x.c);
                                for(size_t dy=0; dy<win&&y+dy<x.h; ++dy)for(size_t dx=0; dx<win&&z+dx<x.w; ++dx)std::copy_n(n.row((y+dy)*x.w+z+dx),x.c,tile.row(dy*win+dx));
                                auto out=vit_attention(tile,p+".attn",win);
                                for(size_t dy=0; dy<win&&y+dy<x.h; ++dy)for(size_t dx=0; dx<win&&z+dx<x.w; ++dx)std::copy_n(out.row(dy*win+dx),x.c,a.row((y+dy)*x.w+z+dx));
                            }
                        }
                        x=add(std::move(x),a);
                        n=op_.norm(x,p+".norm2",1e-6f);
                        x=add(std::move(x),op_.linear(gelu(op_.linear(n,p+".mlp.fc1")),p+".mlp.fc2"));
                    }
                    Features result;
                    for(bool inter: {
                        false,true
                    }
                    ) {
                        if(inter&&!interactive)continue;
                        auto&features=inter?result.interactive:result.propagation;
                        const std::string branch=inter?"interactive_convs":"propagation_convs";
                        for(size_t i=0; i<3; ++i) {
                            std::string p="backbone.vision_backbone."+branch+"."+std::to_string(i);
                            Tensor f=x;
                            if(i==0)f=op_.deconv(gelu(op_.deconv(f,p+".dconv_2x2_0")),p+".dconv_2x2_1");
                            else if(i==1)f=op_.deconv(f,p+".dconv_2x2");
                            features[i]=op_.conv(op_.conv(f,p+".conv_1x1"),p+".conv_3x3",1,1);
                        }
                    }
                    require(result.propagation[2].finite(),"non-finite image features");
                    return result;
                }
                std::pair<Tensor,Tensor> two_way(Tensor image,const Tensor&pe,Tensor tokens,const std::string&p) {
                    Tensor query_pe=tokens;
                    for(size_t i=0; i<c_.decoder_depth; ++i) {
                        ++calls_.two_way_blocks;
                        std::string l=p+".layers."+std::to_string(i);
                        if(i==0)tokens=op_.projected_attention(tokens,tokens,tokens,l+".self_attn",c_.heads);
                        else {
                            auto q=add(tokens,query_pe);
                            tokens=add(tokens,op_.projected_attention(q,q,tokens,l+".self_attn",c_.heads));
                        }
                        tokens=op_.norm(tokens,l+".norm1");
                        tokens=op_.norm(add(tokens,op_.projected_attention(add(tokens,query_pe),add(image,pe),image,l+".cross_attn_token_to_image",c_.heads)),l+".norm2");
                        tokens=op_.norm(add(tokens,op_.linear(relu(op_.linear(tokens,l+".mlp.lin1")),l+".mlp.lin2")),l+".norm3");
                        image=op_.norm(add(image,op_.projected_attention(add(image,pe),add(tokens,query_pe),tokens,l+".cross_attn_image_to_token",c_.heads)),l+".norm4");
                    }
                    tokens=op_.norm(add(tokens,op_.projected_attention(add(tokens,query_pe),add(image,pe),image,p+".final_attn_token_to_image",c_.heads)),p+".norm_final_attn");
                    return {
                        std::move(tokens),std::move(image)
                    };
                }
                Tensor sparse_prompt(const std::vector<Point>&points) {
                    const std::string p="interactive_sam_prompt_encoder";
                    auto g=w_.get(p+".pe_layer.positional_encoding_gaussian_matrix");
                    Tensor out(points.size()+1,1,c_.dim);
                    size_t f=c_.dim/2;
                    auto negative=w_.get(p+".point_embeddings.0.weight"),positive=w_.get(p+".point_embeddings.1.weight");
                    for(size_t n=0; n<points.size(); ++n) {
                        const auto&pt=points[n];
                        float x=2*(pt.x+0.5f/c_.image)-1,y=2*(pt.y+0.5f/c_.image)-1;
                        const auto&e=pt.label==1?positive->v:negative->v;
                        for(size_t i=0; i<f; ++i) {
                            float a=6.283185307179586f*(x*g->v[i]+y*g->v[f+i]);
                            out.row(n)[i]=std::sin(a)+e[i];
                            out.row(n)[f+i]=std::cos(a)+e[f+i];
                        }
                    }
                    auto padding=w_.get(p+".not_a_point_embed.weight");
                    std::copy(padding->v.begin(),padding->v.end(),out.row(points.size()));
                    return out;
                }
                Prediction decode(Tensor image,const std::array<Tensor,3>&features,bool multiplex,size_t active,const std::vector<Point>&points= {
                }
                ) {
                    if(multiplex)++calls_.multiplex_head;
                    else ++calls_.interactive_head;
                    const std::string p=multiplex?"sam_mask_decoder":"interactive_sam_mask_decoder";
                    size_t objects=multiplex?c_.slots:1,masks=multiplex?3:4;
                    Tensor object_tokens=embedding(w_.get(p+".obj_score_token.weight")),iou_tokens=embedding(w_.get(p+".iou_token.weight")),mask_tokens=embedding(w_.get(p+".mask_tokens.weight"));
                    if(multiplex) {
                        auto valid=w_.get("output_valid_embed"),invalid=w_.get("output_invalid_embed");
                        for(size_t o=0; o<c_.slots; ++o)for(size_t k=0; k<masks; ++k)for(size_t d=0; d<c_.dim; ++d)mask_tokens.row(o*masks+k)[d]+=(o<active?valid:invalid)->v[o*c_.dim+d];
                    }
                    std::vector<Tensor>seq= {
                        object_tokens,iou_tokens,mask_tokens
                    };
                    Tensor pe;
                    if(multiplex)pe=random_position(*w_.get("image_pe_layer.positional_encoding_gaussian_matrix"),image.h,image.w);
                    else {
                        seq.push_back(sparse_prompt(points));
                        image=add_vector(std::move(image),w_.get("interactivity_no_mem_embed")->v);
                        image=add_vector(std::move(image),w_.get("interactive_sam_prompt_encoder.no_mask_embed.weight")->v);
                        pe=random_position(*w_.get("interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"),image.h,image.w);
                    }
                    auto transformed=two_way(std::move(image),pe,concat(seq),p+".transformer");
                    auto&hs=transformed.first;
                    Tensor high0=op_.conv(features[0],p+".conv_s0"),high1=op_.conv(features[1],p+".conv_s1");
                    Tensor up=gelu(op_.norm(add(op_.deconv(transformed.second,p+".output_upscaling.0"),high1),p+".output_upscaling.1",1e-6f));
                    up=gelu(add(op_.deconv(up,p+".output_upscaling.3"),high0));
                    Tensor scores=op_.mlp(slice_rows(hs,0,objects),p+".pred_obj_score_head"),quality=op_.mlp(slice_rows(hs,objects,objects),p+".iou_prediction_head");
                    std::vector<Tensor>hypers;
                    for(size_t k=0; k<masks; ++k) {
                        Tensor t(objects,1,c_.dim);
                        for(size_t o=0; o<objects; ++o)std::copy_n(hs.row(2*objects+o*masks+k),c_.dim,t.row(o));
                        hypers.push_back(op_.mlp(std::move(t),p+".output_hypernetworks_mlps."+std::to_string(k)));
                    }
                    Prediction result;
                    result.masks=Tensor(up.h,up.w,objects);
                    result.pointers=Tensor(objects,1,c_.dim);
                    result.scores.resize(objects);
                    result.quality.resize(objects);
                    Tensor selected(objects,1,c_.dim);
                    auto candidate_logits=[&](size_t object,size_t candidate) {
                        std::vector<float> logits(up.rows());
                        for(size_t n=0; n<up.rows(); ++n) {
                            float value=0;
                            for(size_t d=0; d<up.c; ++d)value+=up.row(n)[d]*hypers[candidate].row(object)[d];
                            logits[n]=value;
                        }
                        return logits;
                    };
                    for(size_t o=0; o<objects; ++o) {
                        // One initial point uses the three ambiguity candidates. For
                        // multiple clicks, keep the single-mask token when stable.
                        size_t best=multiplex?0:1;
                        for(size_t k=best+1; k<masks; ++k)if(quality.row(o)[k]>quality.row(o)[best])best=k;
                        std::vector<float> logits;
                        if(!multiplex && points.size()>1) {
                            logits=candidate_logits(o,0);
                            size_t intersection=0,united=0;
                            for(float value:logits) {
                                intersection+=value>0.05f;
                                united+=value>-0.05f;
                            }
                            const double stability=united?double(intersection)/united:1.0;
                            if(stability>=0.98)best=0;
                            else logits.clear();
                        }
                        if(logits.empty())logits=candidate_logits(o,best);
                        result.scores[o]=scores.row(o)[0];
                        result.quality[o]=quality.row(o)[best];
                        std::copy_n(hs.row(2*objects+o*masks+best),c_.dim,selected.row(o));
                        for(size_t n=0; n<up.rows(); ++n)result.masks.row(n)[o]=result.scores[o]>0?logits[n]:-1024.0f;
                    }
                    result.pointers=op_.mlp(selected,multiplex?"obj_ptr_proj":"interactive_obj_ptr_proj");
                    Tensor no_object=op_.linear(result.pointers,"no_obj_ptr_linear");
                    for(size_t o=0; o<objects; ++o)if(result.scores[o]<=0)std::copy_n(no_object.row(o),c_.dim,result.pointers.row(o));
                    require(result.masks.finite()&&result.pointers.finite()&&scores.finite()&&quality.finite(),"non-finite decoder result");
                    return result;
                }
                Memory encode_memory(size_t index,const Tensor&image,const Prediction&prediction,size_t active) {
                    ++calls_.memory_encoder;
                    Tensor high=resize(prediction.masks,c_.image,c_.image);
                    Tensor masks(c_.image,c_.image,2*c_.slots);
                    for(size_t n=0; n<masks.rows(); ++n)for(size_t o=0; o<active; ++o) {
                        float value=high.row(n)[o];
                        float sig=value>=0?1.0f/(1.0f+std::exp(-value)):std::exp(value)/(1.0f+std::exp(value));
                        masks.row(n)[o]=2*sig-1;
                        masks.row(n)[c_.slots+o]=index==0?1.0f:0.0f;
                    }
                    masks=resize(masks,16*c_.grid(),16*c_.grid(),true);
                    const std::string p="maskmem_backbone";
                    for(size_t i=0; i<4; ++i) {
                        std::string base=p+".mask_downsampler.encoder.";
                        masks=gelu(op_.norm(op_.conv(masks,base+std::to_string(i*3),2,1),base+std::to_string(i*3+1),1e-6f));
                    }
                    masks=op_.conv(masks,p+".mask_downsampler.encoder.12");
                    Tensor x=add(op_.conv(image,p+".pix_feat_proj"),masks);
                    for(size_t i=0; i<2; ++i) {
                        std::string l=p+".fuser.layers."+std::to_string(i);
                        Tensor t=op_.norm(op_.conv(x,l+".dwconv",1,3,true,true),l+".norm",1e-6f);
                        t=op_.linear(gelu(op_.linear(t,l+".pwconv1")),l+".pwconv2");
                        auto gamma=w_.get(l+".gamma");
                        for(size_t n=0; n<t.rows(); ++n)for(size_t d=0; d<t.c; ++d)t.row(n)[d]*=gamma->v[d];
                        x=add(std::move(x),t);
                    }
                    auto noobj=w_.get("no_obj_embed_spatial");
                    std::vector<float>empty(c_.dim,0);
                    for(size_t o=0; o<c_.slots; ++o)if(o>=active||prediction.scores[o]<=0)for(size_t d=0; d<c_.dim; ++d)empty[d]+=noobj->v[o*c_.dim+d];
                    x=add_vector(std::move(x),empty);
                    require(x.finite(),"non-finite encoded memory");
                    return Memory {
                        index,image,std::move(x),prediction.pointers
                    };
                }
                Tensor condition(const Tensor&image,size_t index,const Memory&first,const std::deque<Memory>&past) {
                    ++calls_.memory_attention;
                    Tensor pe=sine_position(c_.grid(),c_.grid(),c_.dim);
                    std::vector<const Memory*>spatial= {
                        &first
                    };
                    for(const auto&mem:past)if(index-mem.index<c_.mem_frames)spatial.push_back(&mem);
                    std::vector<Tensor>mi,mv,mp;
                    auto tpos=w_.get("maskmem_tpos_enc");
                    for(const Memory*mem:spatial) {
                        size_t diff=index-mem->index;
                        size_t ti=mem->index==0?(diff>=c_.mem_frames?c_.mem_frames-1:c_.mem_frames-diff-1):diff-1;
                        require(ti<c_.mem_frames,"temporal index overflow");
                        std::vector<float>bias(c_.dim);
                        std::copy_n(tpos->v.data()+ti*c_.dim,c_.dim,bias.data());
                        mi.push_back(mem->image);
                        mv.push_back(mem->encoded);
                        mp.push_back(add_vector(pe,bias));
                    }
                    size_t pointer_limit=std::min(c_.ptr_frames,m_.options.total_frames);
                    pointer_limit=std::max(size_t(2),pointer_limit);
                    std::vector<const Memory*>pointer_mem= {
                        &first
                    };
                    for(auto it=past.rbegin(); it!=past.rend()&&index-it->index<pointer_limit; ++it)pointer_mem.push_back(&*it);
                    const size_t spatial_tokens=spatial.size()*pe.rows();
                    size_t pointer_tokens=0;
                    for(const Memory*mem:pointer_mem) {
                        Tensor temporal(1,1,c_.dim);
                        float position=float(index-mem->index)/float(pointer_limit-1);
                        size_t half=c_.dim/2;
                        for(size_t d=0; d<half; ++d) {
                            float den=std::pow(10000.0f,float(2*(d/2))/half);
                            float phase=position/den;
                            temporal.v[d]=std::sin(phase);
                            temporal.v[half+d]=std::cos(phase);
                        }
                        temporal=op_.linear(temporal,"obj_ptr_tpos_proj");
                        Tensor pos(c_.slots,1,c_.dim);
                        for(size_t o=0; o<c_.slots; ++o)std::copy(temporal.v.begin(),temporal.v.end(),pos.row(o));
                        mv.push_back(mem->pointers);
                        mi.emplace_back(c_.slots,1,c_.dim);
                        mp.push_back(std::move(pos));
                        pointer_tokens+=c_.slots;
                    }
                    Tensor memory=concat(mv),memory_image=concat(mi),memory_pos=concat(mp);
                    require(memory.rows()==spatial_tokens+pointer_tokens,"memory assembly mismatch");
                    Tensor target=add(image,pe,0.1f);
                    for(size_t i=0; i<c_.memory_depth; ++i) {
                        ++calls_.memory_blocks;
                        std::string p="transformer.encoder.layers."+std::to_string(i);
                        Tensor norm=op_.norm(target,p+".norm1"),q=op_.linear(norm,p+".self_attn_q_proj"),k=op_.linear(norm,p+".self_attn_k_proj"),v=op_.linear(norm,p+".self_attn_v_proj");
                        rope(q,c_.heads,c_.grid());
                        rope(k,c_.heads,c_.grid());
                        target=add(std::move(target),op_.linear(op_.sdpa(q,k,v,c_.heads),p+".self_attn_out_proj"));
                        norm=op_.norm(target,p+".norm2");
                        q=add(op_.linear(image,p+".image_cross_attn_q_proj"),op_.linear(norm,p+".cross_attn_q_proj"));
                        k=add(add(op_.linear(memory_image,p+".image_cross_attn_k_proj"),op_.linear(memory,p+".cross_attn_k_proj")),memory_pos);
                        v=op_.linear(memory,p+".cross_attn_v_proj");
                        rope(q,c_.heads,c_.grid());
                        rope(k,c_.heads,c_.grid(),1,pointer_tokens);
                        target=add(std::move(target),op_.linear(op_.sdpa(q,k,v,c_.heads),p+".cross_attn_out_proj"));
                        norm=op_.norm(target,p+".norm3");
                        target=add(std::move(target),op_.linear(gelu(op_.linear(norm,p+".linear1")),p+".linear2"));
                    }
                    target=op_.norm(target,"transformer.encoder.norm");
                    require(target.finite(),"non-finite conditioned features");
                    return target;
                }
                Result result(size_t index,const Image&image,const Prediction&p,const std::vector<int64_t>&ids) {
                    Tensor high=resize(resize(p.masks,c_.image,c_.image),image.height,image.width);
                    require(high.finite(),"non-finite output logits");
                    Result r;
                    r.frame_index=index;
                    r.width=image.width;
                    r.height=image.height;
                    for(size_t o=0; o<ids.size(); ++o) {
                        ObjectMask mask;
                        mask.object_id=ids[o];
                        mask.score=p.scores[o];
                        mask.quality=p.quality[o];
                        mask.logits.resize(image.width*image.height);
                        for(size_t n=0; n<high.rows(); ++n)mask.logits[n]=high.row(n)[o];
                        r.objects.push_back(std::move(mask));
                    }
                    return r;
                }
            };
        }
        struct Session::Impl {
            std::shared_ptr<Model::Impl>model;
            bool ready=false;
            size_t index=0,width=0,height=0,interventions=0;
            std::vector<int64_t>ids;
            Memory first;
            std::deque<Memory>past;
            Counters calls;
            explicit Impl(std::shared_ptr<Model::Impl>m):model(std::move(m)) {
            }
        };
        Model::Model(const std::string&path,const Options&o):p_(std::make_shared<Impl>(path,o)) {
        }
        Model::~Model()=default;
        Model::Model(Model&&)noexcept=default;
        Model&Model::operator=(Model&&)noexcept=default;
        std::string Model::info_json()const {
            std::lock_guard<std::mutex>lock(p_->mutex);
            Json j;
            j["backend"]="native-cpp-cpu";
            j["threads"]=size_t(p_->options.threads);
            j["python_inference"]=false;
            j["onnx_runtime"]=false;
            j["storage"]="GGUF v3 ConvRot INT8";
            j["compute"]="FP32 CPU; inverse-ConvRot restored weights";
            j["trained_checkpoint_verified"]=false;
            j["test_fixture"]=p_->weights.fixture();
            j["source_sha256"]=p_->weights.hash();
            j["config"]=p_->weights.config.json();
            j["required_tensors"]=p_->weights.required_count;
            j["used_tensors"]=p_->weights.used.size();
            j["int8_restorations"]=p_->weights.restorations;
            j["convrot_restorations"]=p_->weights.convrot_restorations;
            return j.dump();
        }
        Session::Session(Model&m):p_(new Impl(m.p_)) {
            require(bool(m.p_),"moved-from model");
        }
        Session::~Session()=default;
        Session::Session(Session&&)noexcept=default;
        Session&Session::operator=(Session&&)noexcept=default;
        Result Session::begin(const Image&image,const std::vector<Point>&points) {
            require(!p_->ready,"session already initialized; call reset before begin");
            require(!points.empty()&&points.size()<=4096,"supply 1..4096 prompt points");
            std::vector<int64_t>ids;
            std::vector<std::vector<Point>>groups;
            for(const auto&pt:points) {
                require(pt.object_id>0&&std::isfinite(pt.x)&&std::isfinite(pt.y)&&pt.x>=0&&pt.x<=1&&pt.y>=0&&pt.y<=1&&(pt.label==0||pt.label==1),"invalid point prompt");
                auto it=std::find(ids.begin(),ids.end(),pt.object_id);
                if(it==ids.end()) {
                    ids.push_back(pt.object_id);
                    groups.emplace_back();
                    it=ids.end()-1;
                }
                groups[size_t(it-ids.begin())].push_back(pt);
            }
            require(ids.size()<=16,"one multiplex bucket supports at most 16 objects");
            for(const auto&group:groups)require(std::any_of(group.begin(),group.end(),[](const Point&pt) {
                return pt.label==1;
            }
            ),"each object requires a positive point");
            std::lock_guard<std::mutex>lock(p_->model->mutex);
            Counters counters=p_->calls;
            Graph graph(*p_->model,counters);
            const auto&c=p_->model->weights.config;
            auto features=graph.encode(image,true);
            Prediction combined;
            combined.masks=Tensor(4*c.grid(),4*c.grid(),c.slots);
            combined.pointers=Tensor(c.slots,1,c.dim);
            combined.scores.resize(c.slots);
            combined.quality.resize(c.slots);
            for(size_t o=0; o<ids.size(); ++o) {
                auto pred=graph.decode(features.interactive[2],features.interactive,false,1,groups[o]);
                for(size_t n=0; n<combined.masks.rows(); ++n)combined.masks.row(n)[o]=pred.masks.row(n)[0];
                std::copy_n(pred.pointers.row(0),c.dim,combined.pointers.row(o));
                combined.scores[o]=pred.scores[0];
                combined.quality[o]=pred.quality[0];
            }
            auto memory=graph.encode_memory(0,features.propagation[2],combined,ids.size());
            auto output=graph.result(0,image,combined,ids);
            p_->ids=std::move(ids);
            p_->width=image.width;
            p_->height=image.height;
            p_->first=std::move(memory);
            p_->past.clear();
            p_->index=0;
            p_->ready=true;
            p_->calls=counters;
            return output;
        }
        Result Session::step(const Image&image) {
            require(p_->ready,"step requires begin");
            require(image.width==p_->width&&image.height==p_->height,"frame dimensions changed");
            require(p_->index<std::numeric_limits<size_t>::max()-1,"frame index overflow");
            std::lock_guard<std::mutex>lock(p_->model->mutex);
            Counters counters=p_->calls;
            Graph graph(*p_->model,counters);
            const auto&c=p_->model->weights.config;
            size_t index=p_->index+1;
            auto features=graph.encode(image,false);
            Tensor conditioned=graph.condition(features.propagation[2],index,p_->first,p_->past);
            auto pred=graph.decode(std::move(conditioned),features.propagation,true,p_->ids.size());
            for(size_t o=p_->ids.size(); o<c.slots; ++o) {
                pred.scores[o]=0;
                pred.quality[o]=0;
                std::fill_n(pred.pointers.row(o),c.dim,0);
                for(size_t n=0; n<pred.masks.rows(); ++n)pred.masks.row(n)[o]=0;
            }
            auto memory=graph.encode_memory(index,features.propagation[2],pred,p_->ids.size());
            auto output=graph.result(index,image,pred,p_->ids);
            p_->past.push_back(std::move(memory));
            while(p_->past.size()>std::max(c.mem_frames-1,c.ptr_frames-1))p_->past.pop_front();
            p_->index=index;
            p_->calls=counters;
            return output;
        }
        void Session::reset() {
            p_->ready=false;
            p_->index=0;
            p_->ids.clear();
            p_->first=Memory {
            };
            p_->past.clear();
            p_->calls=Counters {
            };
            p_->interventions=0;
        }
        void Session::zero_memory_for_test() {
            require(p_->ready,"ablation requires initialized session");
            auto zero=[](Memory&m) {
                std::fill(m.image.v.begin(),m.image.v.end(),0);
                std::fill(m.encoded.v.begin(),m.encoded.v.end(),0);
                std::fill(m.pointers.v.begin(),m.pointers.v.end(),0);
            };
            zero(p_->first);
            for(auto&m:p_->past)zero(m);
            ++p_->interventions;
        }
        std::string Session::stats_json()const {
            Json j=p_->calls.json();
            j["backend"]="native-cpp-cpu";
            j["frames"]=p_->ready?p_->index+1:0;
            j["memory_frames_stored"]=p_->ready?p_->past.size()+1:0;
            j["memory_interventions"]=p_->interventions;
            j["slot_capacity"]=size_t(16);
            Json::array ids;
            for(auto id:p_->ids)ids.emplace_back(std::to_string(id));
            j["object_ids"]=ids;
            return j.dump();
        }
    }
}
