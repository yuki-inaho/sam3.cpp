// Opt-in second quantization. Selection intersects strictly validated GGUF
// Linear metadata with the native graph's explicitly registered Linear set.
using sam2_packed_weights = std::map<std::string,sam2_cpu_quant::packed_weight>;
static bool sam2_prepare_q8_weights(const sam2_gguf_file& file, sam3_model& model,
                                   sam2_packed_weights& packed) {
    try {
        std::ofstream manifest;
        if(const char* path=std::getenv("EFFICIENT_SAM2_QUANT_MANIFEST")) {
            manifest.open(path);
            if(!manifest) throw std::runtime_error("Cannot write requested quantization manifest");
            manifest << "{\"schema\":1,\"candidate\":\"inverse-convrot-f32-then-ggml-q8_0\",\"layers\":[";
        }
        size_t selected=0, fallback=0;
        for(const auto& item:file.tensors) {
            if(item.second.dtype!="I8") continue;
            const auto name=sam2_gguf_runtime_name(item.first);
            const auto found=model.tensors.find(name);
            if(!model.efficient_linear_weights.count(name) || found==model.tensors.end())
                throw std::runtime_error("Q8 candidate target is not a registered Linear: "+item.first);
            auto* tensor=found->second;
            if(item.second.shape.size()!=2 || tensor->ne[0]!=int64_t(item.second.shape[1]) ||
               tensor->ne[1]!=int64_t(item.second.shape[0]) || tensor->ne[2]!=1 || tensor->ne[3]!=1)
                throw std::runtime_error("Q8 candidate Linear shape mismatch: "+item.first);
            std::vector<float> restored;
            std::string error;
            if(!sam2_gguf_restore(file,item.first,restored,error)) throw std::runtime_error(error);
            auto result=sam2_cpu_quant::pack(restored.data(),restored.size(),tensor->ne[0],tensor->ne[1],true);
            if(manifest) {
                if(selected+fallback) manifest << ',';
                manifest << "{\"source\":" << std::quoted(item.first)
                    << ",\"in_features\":" << tensor->ne[0] << ",\"out_features\":" << tensor->ne[1]
                    << ",\"candidate\":" << std::quoted(result.type==GGML_TYPE_Q8_0?"q8_0-second-quantization":"f32")
                    << ",\"reason\":" << std::quoted(result.reason) << '}';
            }
            if(result.type==GGML_TYPE_Q8_0) ++selected; else ++fallback;
            // Tensor descriptors have no allocated data or views at this point.
            // Keep the registered pointers while assigning packed row strides.
            tensor->type=result.type;
            tensor->nb[0]=ggml_type_size(result.type);
            tensor->nb[1]=ggml_row_size(result.type,tensor->ne[0]);
            tensor->nb[2]=tensor->nb[1]*tensor->ne[1];
            tensor->nb[3]=tensor->nb[2]*tensor->ne[2];
            packed.emplace(name,std::move(result));
        }
        if(manifest) manifest << "],\"selected\":" << selected << ",\"fp32_fallback\":" << fallback << "}\n";
        fprintf(stderr,"Efficient-SAM2 experimental Q8_0 compute: %zu selected Linear weights, %zu exact FP32 fallbacks; dynamic Q8_0 activations\n",selected,fallback);
        return true;
    } catch(const std::exception& error) {
        fprintf(stderr,"%s: %s\n",__func__,error.what());
        return false;
    }
}
