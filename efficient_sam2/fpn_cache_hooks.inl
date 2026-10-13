// Exact unconditioned FPN reuse. No decoder, memory attention, object pointer,
// mask or routing policy state is stored. Ownership stays with the model.
static efficient_sam2::fpn_cache::Entry sam2_capture_fpn(const sam3_state& state) {
    efficient_sam2::fpn_cache::Entry entry;
    for(int i=0;i<3;++i) {
        const auto* tensor=state.neck_trk[i];
        auto& out=entry.tensors[i];
        std::copy(tensor->ne,tensor->ne+4,out.shape.begin());
        out.values.resize(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,out.values.data(),0,out.values.size()*sizeof(float));
    }
    return entry;
}

static bool sam2_restore_fpn(sam3_state& state,const sam3_model& model,
                             const efficient_sam2::fpn_cache::Entry& entry) {
    if(state.buffer) {ggml_backend_buffer_free(state.buffer);state.buffer=nullptr;}
    if(state.pe_buf) {ggml_backend_buffer_free(state.pe_buf);state.pe_buf=nullptr;}
    if(state.pe_ctx) {ggml_free(state.pe_ctx);state.pe_ctx=nullptr;}
    if(state.ctx) {ggml_free(state.ctx);state.ctx=nullptr;}
    std::fill(state.neck_trk,state.neck_trk+4,nullptr);
    std::fill(state.neck_trk_pe,state.neck_trk_pe+4,nullptr);
    const ggml_init_params sp={ggml_tensor_overhead()*32,nullptr,true};
    const ggml_init_params pp={ggml_tensor_overhead()*16,nullptr,true};
    state.ctx=ggml_init(sp);state.pe_ctx=ggml_init(pp);
    if(!state.ctx || !state.pe_ctx) return false;
    for(int i=0;i<3;++i) {
        const auto& t=entry.tensors[i];const auto& s=t.shape;
        state.neck_trk[i]=ggml_new_tensor_4d(state.ctx,GGML_TYPE_F32,s[0],s[1],s[2],s[3]);
        state.neck_trk_pe[i]=ggml_new_tensor_4d(state.pe_ctx,GGML_TYPE_F32,model.hparams.neck_dim,s[1],s[2],1);
        char name[64];std::snprintf(name,sizeof(name),"neck_trk_%d",i);ggml_set_name(state.neck_trk[i],name);
        std::snprintf(name,sizeof(name),"neck_trk_pe_%d",i);ggml_set_name(state.neck_trk_pe[i],name);
    }
    state.buffer=ggml_backend_alloc_ctx_tensors(state.ctx,model.backend);
    state.pe_buf=ggml_backend_alloc_ctx_tensors(state.pe_ctx,model.backend);
    if(!state.buffer || !state.pe_buf) return false;
    for(int i=0;i<3;++i) {
        const auto& t=entry.tensors[i];
        ggml_backend_tensor_set(state.neck_trk[i],t.values.data(),0,t.values.size()*sizeof(float));
        const auto pe=sam3_sinusoidal_pe_2d(t.shape[2],t.shape[1],model.hparams.neck_dim);
        ggml_backend_tensor_set(state.neck_trk_pe[i],pe.data(),0,pe.size()*sizeof(float));
    }
    return true;
}

// Public numerical tests only: compact hashes, never full FPN tensor dumps.
static void sam2_trace_fpn(const sam3_state& state,const std::string& input_sha,bool hit) {
    if(!std::getenv("EFFICIENT_SAM2_FPN_TRACE")) return;
    const auto entry=sam2_capture_fpn(state);
    const auto ids=state.efficient_window_ids.empty()?efficient_sam2::all_windows(64,64,14):state.efficient_window_ids;
    std::ostringstream out;out << "{\"event\":\"efficient_fpn\",\"input_sha256\":\"" << input_sha << "\",\"hit\":" << (hit?"true":"false") << ",\"windows\":[";
    for(size_t i=0;i<ids.size();++i) {if(i)out<<',';out<<ids[i];}
    out << "],\"tensor_sha256\":[";
    for(int i=0;i<3;++i) {if(i)out<<',';const auto& v=entry.tensors[i].values;out << '"' << efficient_sam2::prefix_cache::sha256_bytes(v.data(),v.size()*sizeof(float)) << '"';}
    out << "]}\n";std::fputs(out.str().c_str(),stderr);
}
