// SPDX-License-Identifier: Apache-2.0
// Efficient-SAM2 native adaptation; see NOTICE.md for upstream provenance.
// Upstream SAM2 portions: Copyright (c) Meta Platforms, Inc. and affiliates.
// Included by sam3.cpp after the Hiera block builder. This is a separate
// Efficient-SAM2 execution path: padded windows persist between selected
// blocks, rather than falling back to the ordinary dense Hiera implementation.
struct sam2_efficient_array {
    std::vector<float> data;
    int64_t ne[4] = {1, 1, 1, 1};
};

static bool sam2_efficient_stage(const sam3_model& model, int n_threads,
        const sam2_efficient_array& input, sam2_efficient_array& output,
        const std::function<ggml_tensor*(ggml_context*, ggml_tensor*)>& build,
        const std::string& phase) {
    sam2_profile_scope profile(phase);
    const ggml_init_params params = {ggml_tensor_overhead()*4096 + ggml_graph_overhead()*2, nullptr, true};
    auto* ctx = ggml_init(params);
    if (!ctx) return false;
    auto* inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input.ne[0], input.ne[1], input.ne[2], input.ne[3]);
    ggml_set_input(inp);
    auto* out = build(ctx, inp);
    ggml_set_output(out);
    auto* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);
    auto* alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, graph);
    profile.lap("build_allocate");
    if (ok) {
        ggml_backend_tensor_set(inp, input.data.data(), 0, input.data.size()*sizeof(float));
        profile.lap("upload");
        ok = sam3_graph_compute(model.backend, graph, n_threads);
        profile.lap("compute");
    }
    if (ok) {
        for (int d=0; d<4; ++d) output.ne[d] = out->ne[d];
        output.data.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, output.data.data(), 0, output.data.size()*sizeof(float));
        profile.lap("download");
    }
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

static bool sam2_efficient_backbone(const sam3_model& model, const sam3_state& state,
        const std::vector<float>& pixels, int image_size, sam2_efficient_array stages[4],
        const efficient_sam2::prefix_cache::Key* cache_key=nullptr) {
    try {
        using namespace efficient_sam2;
        sam2_profile_scope backbone_profile("hiera.backbone");
        sam2_profile_scope prefix_profile("hiera.prefix");
        if (image_size != 1024) {
            fprintf(stderr, "%s: Efficient-SAM2 requires its trained 1024-square geometry\n", __func__);
            return false;
        }
        const auto& hp=model.hparams;
        sam2_efficient_array input, x;
        std::shared_ptr<const prefix_cache::Entry> cached;
        if(cache_key && model.efficient_cache) try {
            prefix_cache::Hit hit=prefix_cache::Hit::miss;
            std::string error;
            cached=prefix_cache::lookup(*model.efficient_cache,*cache_key,hit,error);
            fprintf(stderr,"Efficient-SAM2 prefix cache: %s%s%s\n",cached?(hit==prefix_cache::Hit::ram?"RAM hit":"disk hit"):"miss/recompute",
                error.empty()?"":": ",error.c_str());
        } catch(const std::exception& error) {
            fprintf(stderr,"Efficient-SAM2 prefix cache lookup failed; recomputing: %s\n",error.what());
        }
        if(cached) {
            auto restore=[](sam2_efficient_array& dst,const prefix_cache::Tensor& src) {
                dst.data=src.values;
                std::copy(src.shape.begin(),src.shape.end(),dst.ne);
            };
            restore(x,cached->tensors[0]); restore(stages[0],cached->tensors[1]); restore(stages[1],cached->tensors[2]);
        } else {
        input.data=pixels;
        input.ne[0]=input.ne[1]=image_size; input.ne[2]=3;
        if (!sam2_efficient_stage(model,state.n_threads,input,x,[&](ggml_context* ctx,ggml_tensor* inp) {
            auto* value=ggml_conv_2d(ctx,model.hiera.patch_embed_w,inp,4,4,3,3,1,1);
            value=ggml_add(ctx,value,ggml_reshape_4d(ctx,model.hiera.patch_embed_b,1,1,hp.hiera_embed_dim,1));
            return ggml_cont(ctx,ggml_permute(ctx,value,1,2,0,3));
        },"hiera.patch_embed")) return false;
        sam2_profile_scope position_profile("hiera.position");
        const auto pe=sam2_compute_pos_embed(model,image_size/4,image_size/4);
        if(pe.size()!=x.data.size()) return false;
        for(size_t i=0;i<pe.size();++i) x.data[i]+=pe[i];
        position_profile.lap("compute");
        position_profile.enabled=false;
        }

        const int first=hp.hiera_stages[0]+hp.hiera_stages[1]+1;
        const int last=hp.hiera_global_idx[hp.hiera_global_n-1];
        const int ws=hp.hiera_window_spec[2], C=4*hp.hiera_embed_dim;
        const int W=64,H=64,nw=(W+ws-1)/ws,total=nw*nw;
        std::vector<int> selected=state.efficient_window_ids;
        if(selected.empty()) for(int i=0;i<total;++i) selected.push_back(i);
        const auto background_ids=complement_windows(selected,total);
        std::vector<float> background;
        int stage_index=cached?2:0;
        for(int i=cached?first:0;i<hp.hiera_total_blocks();++i) {
            auto block=model.hiera.blocks[i];
            if(i==first) {
                prefix_profile.lap("patch_position_blocks0_to5");
                prefix_profile.enabled=false;
                if(!cached && cache_key && model.efficient_cache) try {
                    prefix_cache::Entry entry;
                    auto copy=[](prefix_cache::Tensor& dst,const sam2_efficient_array& src) {
                        dst.values=src.data; std::copy(src.ne,src.ne+4,dst.shape.begin());
                    };
                    copy(entry.tensors[0],x); copy(entry.tensors[1],stages[0]); copy(entry.tensors[2],stages[1]);
                    std::string error;
                    if(!prefix_cache::store(*model.efficient_cache,*cache_key,entry,error))
                        fprintf(stderr,"Efficient-SAM2 prefix cache store failed; inference continues: %s\n",error.c_str());
                } catch(const std::exception& error) {
                    fprintf(stderr,"Efficient-SAM2 prefix cache store failed; inference continues: %s\n",error.what());
                }
                background=pack_windows(x.data,C,W,H,ws,background_ids);
                x.data=pack_windows(x.data,C,W,H,ws,selected);
                x.ne[0]=C; x.ne[1]=ws; x.ne[2]=ws; x.ne[3]=selected.size();
            }
            if(i==last) {
                if(!background_ids.empty()) {
                    sam2_efficient_array old, restored;
                    old.data=std::move(background); old.ne[0]=C;
                    old.ne[1]=ws; old.ne[2]=ws; old.ne[3]=background_ids.size();
                    if(!sam2_efficient_stage(model,state.n_threads,old,restored,[&](ggml_context* ctx,ggml_tensor* inp) {
                        auto* flat=ggml_reshape_2d(ctx,inp,C,ggml_nelements(inp)/C);
                        auto* norm=ggml_norm(ctx,flat,1e-6f);
                        norm=ggml_add(ctx,ggml_mul(ctx,norm,model.efficient_norm_w),model.efficient_norm_b);
                        auto* down=ggml_add(ctx,ggml_mul_mat(ctx,model.efficient_down_w,norm),model.efficient_down_b);
                        auto* up=ggml_add(ctx,ggml_mul_mat(ctx,model.efficient_up_w,ggml_relu(ctx,down)),model.efficient_up_b);
                        return ggml_reshape_4d(ctx,ggml_add(ctx,flat,up),C,ws,ws,background_ids.size());
                    },"hiera.background_bypass")) return false;
                    background=std::move(restored.data);
                }
                x.data=merge_windows(x.data,selected,background,background_ids,C,W,H,ws);
                x.ne[0]=C; x.ne[1]=W; x.ne[2]=H; x.ne[3]=1;
            }
            std::vector<int> valid;
            const bool packed_global=i>=first && i<last && block.window_size==0;
            if(packed_global) {
                valid=valid_window_offsets(selected,W,H,ws);
                x.data=gather_tokens(x.data,C,valid);
                x.ne[1]=valid.size(); x.ne[2]=x.ne[3]=1;
            }
            if(i>=first && i<last) block.window_size=0; // Packed batch already partitions local windows.
            sam2_efficient_array next;
            const int spatial_w=x.ne[1],spatial_h=x.ne[2];
            if(!sam2_efficient_stage(model,state.n_threads,x,next,[&](ggml_context* ctx,ggml_tensor* inp) {
                return sam2_hiera_block_forward(ctx,inp,block,spatial_h,spatial_w,-1,true);
            },"hiera.block."+std::to_string(i))) return false;
            x=std::move(next);
            if(packed_global) {
                x.data=scatter_tokens(x.data,C,valid,selected.size()*ws*ws);
                x.ne[1]=ws; x.ne[2]=ws; x.ne[3]=selected.size();
            }
            if(i==model.hiera.stage_ends[stage_index]) stages[stage_index++]=x;
        }
        fprintf(stderr,"Efficient-SAM2 packed Hiera: %zu/%d windows, learned bypass on %zu windows\n",
                selected.size(),total,background_ids.size());
        return stage_index==4;
    } catch(const std::exception& error) {
        fprintf(stderr,"%s: %s\n",__func__,error.what());
        return false;
    }
}
