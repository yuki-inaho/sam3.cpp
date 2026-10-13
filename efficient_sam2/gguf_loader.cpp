#include "gguf_loader.h"
#include "../sam31/native/json.h"
#include "../sam31/include/sam31_quant.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
using Json = sam31::native::Json;
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
bool ends(const std::string &name, const std::string &suffix) {
    return name.size() >= suffix.size() && name.compare(name.size()-suffix.size(), suffix.size(), suffix)==0;
}
// Validate the bounded directory before calling ggml: some ggml versions assert
// on a mistyped alignment or empty key, and divide by zero on zero dimensions.
void preflight(std::ifstream &input,uint64_t length,uint64_t tensors,uint64_t metadata) {
    uint64_t pos=24;
    const uint64_t limit=std::min<uint64_t>(length,64*1024*1024);
    auto number=[&](size_t bytes) {
        require(pos<=limit && bytes<=limit-pos,"Truncated/excessive GGUF directory");
        uint8_t data[8]={}; input.read(reinterpret_cast<char *>(data),bytes);
        require(bool(input),"Truncated GGUF directory"); pos+=bytes;
        uint64_t result=0; for(size_t i=0;i<bytes;++i) result|=uint64_t(data[i])<<(8*i);
        return result;
    };
    auto string=[&](uint64_t bound) {
        const auto n=number(8);
        require(n<=bound && pos<=limit && n<=limit-pos,"Invalid GGUF string length");
        std::string value(size_t(n),'\0'); input.read(value.data(),n);
        require(bool(input),"Truncated GGUF string"); pos+=n; return value;
    };
    const size_t sizes[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    std::set<std::string> keys;
    for(uint64_t i=0;i<metadata;++i) {
        const auto name=string(4096);
        require(!name.empty() && name.find('\0')==std::string::npos && keys.insert(name).second,"Invalid/duplicate GGUF key");
        auto type=number(4);
        require(type<13,"Unknown GGUF metadata type");
        if(name=="general.alignment") {
            require(type==4 && number(4)==32,"Expected UINT32 GGUF alignment 32");
            continue;
        }
        uint64_t count=1;
        if(type==9) {
            type=number(4); count=number(8);
            require(type<13 && type!=9 && count<=1000000,"Invalid/excessive GGUF metadata array");
        }
        for(uint64_t j=0;j<count;++j) {
            if(type==8) (void)string(limit);
            else {
                const auto value=number(sizes[type]);
                require(type!=7 || value<=1,"Invalid GGUF boolean");
            }
        }
    }
    std::set<std::string> names;
    const std::map<uint64_t,size_t> widths={{0,4},{1,2},{24,1},{25,2},{26,4},{27,8},{28,8},{30,2}};
    for(uint64_t i=0;i<tensors;++i) {
        const auto name=string(63);
        require(!name.empty() && name.find('\0')==std::string::npos && names.insert(name).second,"Invalid/duplicate GGUF tensor name");
        const auto dims=number(4); require(dims>=1 && dims<=4,"Invalid GGUF tensor rank");
        uint64_t n=1;
        for(uint64_t d=0;d<dims;++d) {
            const auto size=number(8);
            require(size>0 && size<uint64_t(INT64_MAX)/n,"Invalid GGUF tensor dimension"); n*=size;
        }
        const auto type=number(4); const auto width=widths.find(type);
        require(width!=widths.end() && n<=length/width->second,"Invalid/excessive GGUF tensor storage");
        const auto offset=number(8);
        require(offset%32==0 && offset<=length && n*width->second<=length-offset,"Invalid GGUF tensor offset");
    }
}
std::string text(const gguf_context *ctx, const char *key) {
    const auto i=gguf_find_key(ctx,key);
    require(i>=0 && gguf_get_kv_type(ctx,i)==GGUF_TYPE_STRING, std::string("Missing string metadata: ")+key);
    return gguf_get_val_str(ctx,i);
}
size_t elements(const sam2_gguf_tensor &t) {
    size_t n=1;
    for (size_t d:t.shape) {
        require(d>0 && n<=std::numeric_limits<size_t>::max()/d,"Invalid tensor dimensions: "+t.name);
        n*=d;
    }
    return n;
}
std::vector<uint8_t> raw(const sam2_gguf_file &file, const std::string &name) {
    auto it=file.tensors.find(name);
    require(it!=file.tensors.end(),"Missing tensor: "+name);
    const auto &t=it->second;
    require(t.bytes<=std::numeric_limits<size_t>::max(),"Tensor exceeds address space");
    std::vector<uint8_t> data(t.bytes);
    std::ifstream input(file.path,std::ios::binary);
    input.seekg(t.offset);
    input.read(reinterpret_cast<char *>(data.data()),data.size());
    require(bool(input),"Truncated tensor: "+name);
    return data;
}
std::vector<float> floating(const sam2_gguf_file &file,const std::string &name) {
    const auto &t=file.tensors.at(name);
    auto bytes=raw(file,name);
    std::vector<float> out(elements(t));
    for(size_t i=0;i<out.size();++i) {
        if(t.dtype=="F32") std::memcpy(&out[i],bytes.data()+4*i,4);
        else if(t.dtype=="F16") { ggml_fp16_t h; std::memcpy(&h,bytes.data()+2*i,2); out[i]=ggml_fp16_to_fp32(h); }
        else if(t.dtype=="BF16") { uint16_t h; std::memcpy(&h,bytes.data()+2*i,2); uint32_t u=uint32_t(h)<<16; std::memcpy(&out[i],&u,4); }
        else if(t.dtype=="F64") { double d; std::memcpy(&d,bytes.data()+8*i,8); out[i]=float(d); }
        else throw std::runtime_error("Expected floating-point tensor: "+name);
        require(std::isfinite(out[i]),"Non-finite value: "+name);
    }
    return out;
}
std::string key(const std::string &prefix,const std::string &suffix) {
    return prefix.empty()?suffix:prefix+"."+suffix;
}
bool parameter(const std::string &name,const std::string &suffix) {
    return name==suffix || ends(name,"."+suffix);
}
std::string prefix_of(const std::string &name,const std::string &suffix) {
    return name==suffix?"":name.substr(0,name.size()-suffix.size()-1);
}
struct Quant { size_t group=1; std::string scale; };
Quant quantization(const sam2_gguf_file &file,const sam2_gguf_tensor &t) {
    require(t.dtype=="I8" && parameter(t.name,"weight") && t.shape.size()==2,"Unsupported INT8 parameter: "+t.name);
    const std::string prefix=prefix_of(t.name,"weight");
    const std::string descriptor=key(prefix,"comfy_quant");
    auto d=file.tensors.find(descriptor);
    require(d!=file.tensors.end() && d->second.dtype=="U8" && d->second.shape.size()==1 && d->second.bytes<=4096,"Missing/invalid quantization descriptor: "+t.name);
    const auto bytes=raw(file,descriptor);
    const auto j=Json::parse(std::string(bytes.begin(),bytes.end()));
    require(j.at("format").str()=="int8_tensorwise","Unsupported quantization format");
    const std::set<std::string> keys={"format","convrot","convrot_groupsize","per_row"};
    for(const auto &p:j.obj()) require(keys.count(p.first),"Unknown quantization semantics: "+p.first);
    require(j.at("per_row").boolean(),"per_row must be true when supplied");
    Quant q; q.scale=key(prefix,"weight_scale");
    if(j.at("convrot").boolean()) {
        q.group=j.at("convrot_groupsize").integer();
        size_t g=q.group;
        require(g>=4 && t.shape[1]%g==0,"Invalid ConvRot input-channel group");
        while(g>1 && g%4==0) g/=4;
        require(g==1,"ConvRot requires a power-of-four group");
    } else require(!j.has("convrot_groupsize"),"Unexpected group on unrotated weight");
    auto s=file.tensors.find(q.scale);
    require(s!=file.tensors.end(),"Missing scale: "+q.scale);
    require(s->second.dtype=="F32" && s->second.shape==std::vector<size_t>{t.shape[0],1},
        "Expected F32 per-output-channel scale [out,1]");
    const auto scales=floating(file,q.scale);
    for(float value:scales) require(value>0,"Scale must be finite and positive");
    return q;
}
void replace(std::string &s,const std::string &from,const std::string &to) {
    size_t at=0;
    while((at=s.find(from,at))!=std::string::npos) { s.replace(at,from.size(),to); at+=to.size(); }
}
} // namespace

bool sam2_gguf_validate_native_profile(const sam2_gguf_file &file,std::string &error) {
    try {
        const auto config=Json::parse(file.config_json);
        require(file.sam2_version=="2.1" && file.image_size==1024 &&
            (file.variant=="base_plus" || file.variant=="large"),"Unsupported native model geometry/version");
        require(config.at("efficient_swr").boolean() && config.at("efficient_smr").boolean(),
            "Native Efficient profile requires explicit SWR and SMR");
        require(config.at("bypass_type").str()=="bottleneck","Unsupported native bypass type");
        const int first=file.variant=="base_plus"?6:9;
        const int last=file.variant=="base_plus"?20:43;
        require(config.at("final_global_layer").integer()==size_t(last),"Unsupported final global layer");
        const auto& selected=config.at("selected_layers").arr();
        require(selected.size()==size_t(last-first+1),"Unsupported selected layer count");
        for(int i=first;i<=last;++i) require(selected[i-first].integer()==size_t(i),"Unsupported selected layer sequence");
        require(config.at("memory_temporal_stride").integer()==1,"Native memory requires stride one");
        require(std::abs(config.at("memory_drop_ratio").number()-0.95)<1e-12,"Unsupported memory drop ratio");
        require(std::abs(config.at("wb_theta").number()-0.7)<1e-12,"Unsupported window routing threshold");
        require(config.at("dilate_mask").boolean() && config.at("dilate_kernel_size").integer()==5,
            "Native routing requires 5x5 mask dilation");
        return true;
    } catch(const std::exception& e) { error=e.what(); return false; }
}

bool sam2_gguf_open(const std::string &path,sam2_gguf_file &out,std::string &error) {
    try {
        sam2_gguf_file file; file.path=path;
        std::ifstream input(path,std::ios::binary|std::ios::ate);
        require(bool(input) && input.tellg()>=24,"Missing/truncated GGUF");
        const uint64_t length=uint64_t(input.tellg());
        uint8_t header[24]; input.seekg(0); input.read(reinterpret_cast<char *>(header),24);
        auto u64=[&](size_t at) { uint64_t v=0; for(size_t i=0;i<8;++i) v|=uint64_t(header[at+i])<<(8*i); return v; };
        require(std::memcmp(header,"GGUF",4)==0 && header[4]==3 && !header[5] && !header[6] && !header[7],"Expected little-endian GGUF v3");
        require(u64(8)>0 && u64(8)<=100000 && u64(16)>0 && u64(16)<=16384,"Excessive/empty GGUF directory");
        preflight(input,length,u64(8),u64(16));
        ggml_context *raw_ctx=nullptr;
        std::unique_ptr<gguf_context,decltype(&gguf_free)> ctx(gguf_init_from_file(path.c_str(),{true,&raw_ctx}),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors_ctx(raw_ctx,ggml_free);
        require(bool(ctx) && bool(tensors_ctx),"ggml rejected GGUF directory");
        require(text(ctx.get(),"general.architecture")=="efficientsam2","Expected Efficient-SAM2 architecture");
        auto schema=gguf_find_key(ctx.get(),"efficient_sam2.storage_version");
        require(schema>=0 && gguf_get_kv_type(ctx.get(),schema)==GGUF_TYPE_UINT32 && gguf_get_val_u32(ctx.get(),schema)==1,"Unsupported Efficient-SAM2 storage version");
        require(text(ctx.get(),"efficient_sam2.convrot_basis")=="regular-h4-kron","Unknown ConvRot basis");
        const auto fixture=gguf_find_key(ctx.get(),"efficient_sam2.test_fixture");
        require(fixture>=0 && gguf_get_kv_type(ctx.get(),fixture)==GGUF_TYPE_BOOL,"Missing test-fixture marker");
        file.test_fixture=gguf_get_val_bool(ctx.get(),fixture);
        require(gguf_get_alignment(ctx.get())==32 && gguf_get_data_offset(ctx.get())<=64*1024*1024,"Invalid GGUF alignment/header");
        file.config_json=text(ctx.get(),"efficient_sam2.model_config");
        const Json config=Json::parse(file.config_json);
        if(config.has("variant")) file.variant=config.at("variant").str();
        if(config.has("image_size")) file.image_size=config.at("image_size").integer();
        if(config.has("sam2_version")) file.sam2_version=config.at("sam2_version").str();
        const Json quant_meta=Json::parse(text(ctx.get(),"efficient_sam2.quantization"));
        require(quant_meta.at("scheme").str()=="symmetric-per-output-channel-int8","Unknown quantization scheme");
        const auto map=Json::parse(text(ctx.get(),"efficient_sam2.tensor_map"));
        require(map.obj().size()==size_t(gguf_get_n_tensors(ctx.get())),"Tensor map count mismatch");
        const std::map<std::string,std::pair<int,size_t>> types={{"F32",{0,4}},{"F16",{1,2}},{"I8",{24,1}},{"U8",{24,1}},{"BOOL",{24,1}},{"I16",{25,2}},{"I32",{26,4}},{"I64",{27,8}},{"F64",{28,8}},{"BF16",{30,2}}};
        std::vector<std::pair<uint64_t,uint64_t>> intervals;
        for(int64_t i=0;i<gguf_get_n_tensors(ctx.get());++i) {
            const std::string physical=gguf_get_tensor_name(ctx.get(),i);
            const auto &m=map.at(physical);
            sam2_gguf_tensor t; t.name=m.at("name").str(); t.dtype=m.at("dtype").str();
            require(!t.name.empty() && t.name.size()<=4096 && t.name.find('\0')==std::string::npos,"Invalid source tensor name");
            require(m.at("shape").arr().size()<=16,"Excessive tensor dimensions");
            for(const auto &d:m.at("shape").arr()) t.shape.push_back(d.integer());
            const auto type=types.find(t.dtype);
            require(type!=types.end() && type->second.first==gguf_get_tensor_type(ctx.get(),i),"Logical/physical dtype mismatch");
            const size_t n=elements(t);
            require(n<=std::numeric_limits<size_t>::max()/type->second.second,"Tensor byte overflow");
            t.bytes=gguf_get_tensor_size(ctx.get(),i);
            require(t.bytes==n*type->second.second,"Logical/physical size mismatch");
            const auto *tensor=ggml_get_tensor(tensors_ctx.get(),physical.c_str());
            require(tensor!=nullptr,"Missing GGUF tensor shape");
            std::vector<size_t> shape=t.shape;
            if(shape.empty() || shape.size()>4) shape={n}; else std::reverse(shape.begin(),shape.end());
            for(size_t d=0;d<4;++d) require(size_t(tensor->ne[d])==(d<shape.size()?shape[d]:1),"Logical/physical shape mismatch: "+t.name);
            const uint64_t offset=gguf_get_tensor_offset(ctx.get(),i), base=gguf_get_data_offset(ctx.get());
            require(offset%32==0 && base<=length && offset<=length-base && t.bytes<=length-base-offset,"Tensor outside payload");
            t.offset=base+offset; intervals.push_back({t.offset,t.offset+t.bytes});
            require(file.tensors.emplace(t.name,t).second,"Duplicate source tensor name");
        }
        std::sort(intervals.begin(),intervals.end());
        for(size_t i=1;i<intervals.size();++i) require(intervals[i].first>=intervals[i-1].second,"Overlapping GGUF tensors");
        size_t quant_count=0, rotated_count=0;
        for(const auto &kv:file.tensors) {
            if(kv.second.dtype=="I8" && parameter(kv.first,"weight")) {
                const auto q=quantization(file,kv.second);
                const std::string prefix=prefix_of(kv.first,"weight");
                const auto &layer=quant_meta.at("layers").at(prefix);
                require(layer.at("activation_mode").str()=="w8a16",
                    "Unsupported activation quantization: this CPU loader implements weight-only restoration, not W8A8");
                require(layer.at("in_features").integer()==kv.second.shape[1] &&
                    layer.at("out_features").integer()==kv.second.shape[0],"Quantization layer shape mismatch");
                require(layer.at("group_size").integer()==q.group,"Quantization group/marker mismatch");
                const auto &source_dtype=layer.at("source_dtype").str();
                require(source_dtype=="float32" || source_dtype=="float16" ||
                    source_dtype=="bfloat16" || source_dtype=="float64","Invalid source dtype");
                const auto bias=file.tensors.find(key(prefix,"bias"));
                require(layer.at("has_bias").boolean()==(bias!=file.tensors.end()),"Bias presence mismatch");
                if(bias!=file.tensors.end()) {
                    require(bias->second.dtype=="F32" && bias->second.shape==std::vector<size_t>{kv.second.shape[0]},
                        "Expected F32 bias [out]");
                    (void)floating(file,bias->first);
                }
                ++quant_count;
                if(q.group>1) ++rotated_count;
            }
            for(const auto &suffix:{"comfy_quant","weight_scale"}) if(parameter(kv.first,suffix)) {
                const auto weight=file.tensors.find(key(prefix_of(kv.first,suffix),"weight"));
                require(weight!=file.tensors.end() && weight->second.dtype=="I8","Orphan quantization side tensor");
            }
        }
        require(quant_count==quant_meta.at("layers").obj().size(),"Quantization layer directory mismatch");
        for(const auto &count:std::map<std::string,size_t>{{"int8_layers",quant_count},{"convrot_layers",rotated_count}}) {
            const auto i=gguf_find_key(ctx.get(),("efficient_sam2."+count.first).c_str());
            require(i>=0 && gguf_get_kv_type(ctx.get(),i)==GGUF_TYPE_UINT32 &&
                gguf_get_val_u32(ctx.get(),i)==count.second,"Quantized layer count metadata mismatch: "+count.first);
        }
        out=std::move(file); error.clear(); return true;
    } catch(const std::exception &e) { error=e.what(); return false; }
}

bool sam2_gguf_restore(const sam2_gguf_file &file,const std::string &name,std::vector<float> &out,std::string &error) {
    try {
        const auto &t=file.tensors.at(name);
        if(t.dtype!="I8") out=floating(file,name);
        else {
            const auto q=quantization(file,t);
            const auto data=raw(file,name);
            const auto scales=floating(file,q.scale);
            out.resize(elements(t));
            // Storage v1 calls identity group 1; the shared kernel uses 0.
            const int result=sam31_dequantize_linear(reinterpret_cast<const int8_t *>(data.data()),data.size(),scales.data(),scales.size(),t.shape[0],t.shape[1],q.group==1?0:q.group,out.data(),out.size());
            require(result==0,"ConvRot native restoration failed: "+name);
        }
        error.clear(); return true;
    } catch(const std::exception &e) { out.clear(); error=e.what(); return false; }
}

std::string sam2_gguf_runtime_name(std::string k) {
    replace(k,"image_encoder.trunk.","hiera.");
    replace(k,"hiera.patch_embed.proj.","hiera.patch_embed.");
    if(k.find("hiera.blocks.")!=std::string::npos) { replace(k,".mlp.layers.0.",".mlp.fc1."); replace(k,".mlp.layers.1.",".mlp.fc2."); }
    replace(k,"image_encoder.neck.convs.","fpn.convs."); replace(k,".conv.weight",".weight"); replace(k,".conv.bias",".bias");
    replace(k,"sam_prompt_encoder.","sam_pe."); replace(k,"sam_pe.pe_layer.positional_encoding_gaussian_matrix","sam_pe.pe_gaussian"); replace(k,"sam_pe.mask_downscaling.","sam_pe.mask_ds.");
    replace(k,"sam_mask_decoder.","sam_dec."); replace(k,"sam_dec.transformer.layers.","sam_dec.twoway."); replace(k,"sam_dec.transformer.final_attn_token_to_image.","sam_dec.final_attn."); replace(k,"sam_dec.transformer.norm_final_attn.","sam_dec.final_norm."); replace(k,"sam_dec.output_upscaling.","sam_dec.upscale."); replace(k,"sam_dec.output_hypernetworks_mlps.","sam_dec.hyper.");
    replace(k,".self_attn.",".sa.");
    if(k.find("sam_dec.twoway.")!=std::string::npos) { replace(k,".mlp.layers.0.",".mlp.lin1."); replace(k,".mlp.layers.1.",".mlp.lin2."); }
    replace(k,"memory_encoder.","mem_enc."); replace(k,"mem_enc.mask_downsampler.encoder.","mem_enc.ds."); replace(k,"mem_enc.fuser.layers.","mem_enc.fuser.");
    replace(k,"memory_attention.layers.","mem_attn.layers."); replace(k,"memory_attention.norm.","mem_attn.norm."); replace(k,".cross_attn_image.",".ca.");
    replace(k,"maskmem_tpos_enc","mem_enc.tpos_enc"); replace(k,"mask_downsample.","trk_mask_ds.");
    return k;
}
bool sam2_gguf_is_auxiliary(const std::string &name) {
    return parameter(name,"comfy_quant") || parameter(name,"weight_scale");
}
