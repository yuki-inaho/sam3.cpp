// SPDX-License-Identifier: Apache-2.0
// Efficient-SAM2 native adaptation; see NOTICE.md for upstream provenance.
// Upstream SAM2 portions: Copyright (c) Meta Platforms, Inc. and affiliates.
// Included after sam3_build_mem_attn_graph. Every memory layer owns a separate
// ggml graph. Sparse retrieval uses original raster indices, including RoPE.
static bool sam2_efficient_memory(sam3_tracker& tracker, sam3_state& state,
        const sam3_model& model, int instance_id, int start_frame,
        const sam3_prompt_data& prompt, const std::vector<int>& frame_ids,
        const std::vector<float>& rope_q, const std::vector<float>& rope_k,
        std::vector<float>& conditioned) {
    try {
        using namespace efficient_sam2;
        const int D=model.hparams.neck_dim, MD=model.hparams.mem_out_dim, N=4096;
        auto inserted=tracker.efficient_memory.emplace(instance_id,MemoryQueue(start_frame));
        auto& queue=inserted.first->second;
        const bool capture=queue.should_record(tracker.frame_index);
        conditioned.resize(D*N);
        ggml_backend_tensor_get(state.neck_trk[2],conditioned.data(),0,conditioned.size()*sizeof(float));
        std::vector<std::vector<float>> importances(4);
        for(int layer=0;layer<4;++layer) {
            sam2_profile_scope profile("memory.layer."+std::to_string(layer));
            const auto selection=queue.select(frame_ids,layer,prompt.num_obj_ptr_tokens);
            const auto values=gather_tokens(prompt.prompt,MD,selection.original_indices);
            const auto positions=gather_tokens(prompt.prompt_pos,MD,selection.original_indices);
            const std::vector<int> spatial_indices(selection.original_indices.begin(),
                selection.original_indices.begin()+selection.spatial_tokens);
            const auto selected_rope=gather_tokens(rope_k,D,spatial_indices);
            const int M=selection.original_indices.size();
            profile.lap("sparse_gather");
            const ggml_init_params init={ggml_tensor_overhead()*4096+ggml_graph_overhead()*2,nullptr,true};
            auto* ctx=ggml_init(init);
            if(!ctx) return false;
            auto* current=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,N,1);
            auto* pos=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,N,1);
            auto* memory=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,MD,M,1);
            auto* memory_pos=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,MD,M,1);
            auto* q_rope=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2,D/2,N);
            auto* k_rope=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2,D/2,selection.spatial_tokens);
            for(auto* tensor:{current,pos,memory,memory_pos,q_rope,k_rope}) ggml_set_input(tensor);
            ggml_tensor* importance=nullptr;
            auto* output=sam3_build_mem_attn_graph(ctx,model,current,pos,memory,memory_pos,q_rope,k_rope,
                selection.pointer_tokens,layer,layer+1,capture?&importance:nullptr);
            ggml_set_output(output);
            auto* graph=ggml_new_graph_custom(ctx,8192,false);
            ggml_build_forward_expand(graph,output);
            if(importance) ggml_build_forward_expand(graph,importance);
            auto* alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            bool ok=ggml_gallocr_alloc_graph(alloc,graph);
            profile.lap("build_allocate");
            if(ok) {
                ggml_backend_tensor_set(current,conditioned.data(),0,conditioned.size()*sizeof(float));
                // Later layers do not consume src_pos; uploading an unallocated
                // node would be invalid even though its tensor descriptor exists.
                if(layer==0) ggml_backend_tensor_set(pos,tracker.cached_sinpe_256.data(),0,D*N*sizeof(float));
                ggml_backend_tensor_set(memory,values.data(),0,values.size()*sizeof(float));
                ggml_backend_tensor_set(memory_pos,positions.data(),0,positions.size()*sizeof(float));
                ggml_backend_tensor_set(q_rope,rope_q.data(),0,rope_q.size()*sizeof(float));
                ggml_backend_tensor_set(k_rope,selected_rope.data(),0,selected_rope.size()*sizeof(float));
                profile.lap("upload");
                ok=sam3_graph_compute(model.backend,graph,state.n_threads);
                profile.lap("compute");
            }
            if(ok) {
                ggml_backend_tensor_get(output,conditioned.data(),0,conditioned.size()*sizeof(float));
                const auto diagnostic_label="conditioned_layer"+std::to_string(layer);
                sam2_diag_dump(tracker.frame_index,diagnostic_label.c_str(),conditioned.data(),conditioned.size());
                if(importance) {
                    importances[layer].resize(N);
                    ggml_backend_tensor_get(importance,importances[layer].data(),0,N*sizeof(float));
                }
                profile.lap("download");
            }
            ggml_gallocr_free(alloc); ggml_free(ctx);
            if(!ok) return false;
            fprintf(stderr,"Efficient-SAM2 SMR frame=%d layer=%d spatial=%d pointers=%d pruned=%d\n",
                tracker.frame_index,layer,selection.spatial_tokens,selection.pointer_tokens,int(selection.pruned));
        }
        if(capture) queue.record(tracker.frame_index,importances);
        return true;
    } catch(const std::exception& error) {
        fprintf(stderr,"%s: %s\n",__func__,error.what());
        return false;
    }
}

// Opt-in numerical audit only. Ordinary annotation writes compact masks and
// never materializes saliency or dense encoder tensors on disk.
static void sam2_efficient_trace(const sam3_tracker& tracker, const sam3_state& state,
                                int frame, int instance_id) {
    const char* directory=std::getenv("EFFICIENT_SAM2_TRACE_DIR");
    const bool to_stderr=std::getenv("EFFICIENT_SAM2_TRACE_STDERR")!=nullptr;
    if((!directory || !*directory) && !to_stderr) return;
    char suffix[80];
    std::snprintf(suffix,sizeof(suffix),"/frame_%05d_object_%03d.json",frame,instance_id);
    std::ostringstream out;
    auto integers=[&out](const std::vector<int>& values) {
        out << '[';
        for(size_t i=0;i<values.size();++i) { if(i) out << ','; out << values[i]; }
        out << ']';
    };
    out << "{\"frame\":" << frame << ",\"selected_before\":";
    const auto all=efficient_sam2::all_windows(64,64,14);
    integers(state.efficient_window_ids.empty()?all:state.efficient_window_ids);
    out << ",\"selected_after\":"; integers(tracker.efficient_windows.selected());
    out << ",\"memory_queue\":[";
    const auto queue=tracker.efficient_memory.find(instance_id);
    if(queue!=tracker.efficient_memory.end()) {
        bool first=true;
        for(const auto& item:queue->second.frames()) {
            if(!first) out << ',';
            first=false;
            out << "{\"frame\":" << item.frame_index << ",\"layers\":[";
            for(size_t layer=0;layer<item.layers.size();++layer) {
                if(layer) out << ',';
                std::vector<int> indices;
                for(size_t j=0;j<item.layers[layer].size();++j) if(item.layers[layer][j]) indices.push_back(j);
                integers(indices);
            }
            out << "]}";
        }
    }
    out << "]}\n";
    const auto text=out.str();
    if(directory && *directory) {
        std::ofstream file(std::string(directory)+suffix);file<<text;
        if(!file) fprintf(stderr,"Efficient-SAM2: cannot write requested trace\n");
    }
    if(to_stderr) fprintf(stderr,"Efficient-SAM2 routing trace: %s",text.c_str());
}
