// Cache ownership follows the loaded model/session, not an individual track.
// Reset/reseed may therefore reuse deterministic prefix data, while each
// track's routing and temporal memory remain independent.
static void sam2_prepare_prefix_cache(const sam2_gguf_file& file, sam3_model& model,
                                     const sam3_params& params) {
    if(!params.efficient_prefix_cache && !params.efficient_fpn_cache) return;
    namespace cache=efficient_sam2::prefix_cache;
    if(file.variant!="base_plus") {
        fprintf(stderr,"Efficient-SAM2 prefix cache disabled: only base-plus prefix geometry is supported\n"); return;
    }
    cache::Options options;
    options.disk_directory=params.efficient_cache_directory;
    options.disk_budget_bytes=params.efficient_cache_disk_budget;
    if(!options.disk_directory.empty() && model.runtime_identity.digest.empty()) {
        fprintf(stderr,"Efficient-SAM2 persistent cache disabled: %s\n",model.runtime_identity_error.c_str());
        options.disk_directory.clear(); options.disk_budget_bytes=0;
    }
    std::string error;
    std::shared_ptr<cache::Cache> instance;
    if(params.efficient_prefix_cache) {
        instance=cache::create(options,error);
        if(!instance) fprintf(stderr,"Efficient-SAM2 prefix cache disabled: %s\n",error.c_str());
    }
    cache::Key key;
    key.model_sha256=cache::sha256_file(file.path,error);
    if(key.model_sha256.empty()) { fprintf(stderr,"Efficient-SAM2 prefix cache disabled: %s\n",error.c_str()); return; }
    key.preprocessing=sam3_get_preprocessing(model);
    key.precision=model.experimental_efficient_q8_0?"restored-then-ggml-q8_0-dynamic-activations-v1":"restored-f32-v1";
    const auto policy=file.config_json+"|"+key.preprocessing+"|"+key.precision;
    key.policy_sha256=cache::sha256_bytes(policy.data(),policy.size());
    key.build=std::string(SAM2_CACHE_BUILD_ID)+":"+SAM2_CACHE_CONFIGURATION;
    key.backend="ggml-cpu;avx2="+std::to_string(ggml_cpu_has_avx2())+
        ";fma="+std::to_string(ggml_cpu_has_fma())+";avx512="+std::to_string(ggml_cpu_has_avx512())+
        ";neon="+std::to_string(ggml_cpu_has_neon())+";dotprod="+std::to_string(ggml_cpu_has_dotprod())+
        ";matmul_int8="+std::to_string(ggml_cpu_has_matmul_int8());
    key.backend+=";loaded_elf_sha256="+model.runtime_identity.digest;
    key.threads=params.n_threads;
    model.efficient_cache_key=std::move(key);
    model.efficient_cache=std::move(instance);
    if(model.efficient_cache) fprintf(stderr,"Efficient-SAM2 exact prefix cache enabled: one 49MiB RAM entry; disk=%s\n",
        options.disk_directory.empty()?"disabled":"explicit bounded artifacts");
    if(params.efficient_fpn_cache) {
        model.efficient_fpn_cache=std::make_shared<efficient_sam2::fpn_cache::Cache>(4);
        fprintf(stderr,"Efficient-SAM2 exact full-FPN RAM cache enabled: four84MiB entries per image\n");
    }
}
