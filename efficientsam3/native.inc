// EfficientViT b1 and MobileCLIP-S0 forward stages. No external inference runtime.
static ggml_tensor * sam3_efficient_weight(const sam3_model & model, const std::string & name) {
    const auto found = model.tensors.find(name);
    return found == model.tensors.end() ? nullptr : found->second;
}

static ggml_tensor * sam3_efficient_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    if (!bias) return x;
    return ggml_add(ctx, x, ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]));
}

static ggml_tensor * sam3_efficient_bn(ggml_context * ctx, ggml_tensor * x,
                                      const sam3_model & model, const std::string & prefix) {
    auto * scale = sam3_efficient_weight(model, prefix + ".scale");
    if (!scale) return x;
    x = ggml_mul(ctx, x, ggml_reshape_3d(ctx, scale, 1, 1, scale->ne[0]));
    return sam3_efficient_bias(ctx, x, sam3_efficient_weight(model, prefix + ".shift"));
}

static ggml_tensor * sam3_efficient_conv_raw(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * weight, ggml_tensor * bias, int stride = 1, int groups = 1) {
    const int pw = (int)weight->ne[0] / 2, ph = (int)weight->ne[1] / 2;
    ggml_tensor * out = nullptr;
    if (groups == x->ne[2] && weight->ne[2] == 1) {
        out = ggml_conv_2d_dw_direct(ctx, weight, x, stride, stride, pw, ph, 1, 1);
    } else if (groups == 1 && weight->ne[0] == 1 && weight->ne[1] == 1 && stride == 1) {
        // A pointwise convolution is a matrix multiply; no im2col buffer is needed.
        const int64_t ci = x->ne[2], co = weight->ne[3];
        const int64_t width = x->ne[0], height = x->ne[1];
        auto * input = ggml_reshape_2d(ctx, x, width * height, ci);
        input = ggml_cont(ctx, ggml_transpose(ctx, input));
        auto * kernel = ggml_reshape_2d(ctx, weight, ci, co);
        out = ggml_mul_mat(ctx, kernel, input);
        out = ggml_cont(ctx, ggml_transpose(ctx, out));
        out = ggml_reshape_3d(ctx, out, width, height, co);
    } else if (groups == 1) {
        out = ggml_conv_2d(ctx, weight, x, stride, stride, pw, ph, 1, 1);
    } else {
        const int64_t ci = x->ne[2] / groups, co = weight->ne[3] / groups;
        for (int g = 0; g < groups; ++g) {
            auto * inp = ggml_view_3d(ctx, x, x->ne[0], x->ne[1], ci,
                                     x->nb[1], x->nb[2], g * ci * x->nb[2]);
            auto * w = ggml_view_4d(ctx, weight, weight->ne[0], weight->ne[1], ci, co,
                                    weight->nb[1], weight->nb[2], weight->nb[3], g * co * weight->nb[3]);
            auto * part = ggml_conv_2d(ctx, w, ggml_cont(ctx, inp), stride, stride, pw, ph, 1, 1);
            out = out ? ggml_concat(ctx, out, part, 2) : part;
        }
    }
    return sam3_efficient_bias(ctx, out, bias);
}

static ggml_tensor * sam3_efficient_conv(ggml_context * ctx, ggml_tensor * x,
        const sam3_model & model, const std::string & prefix, int stride = 1,
        bool activate = false, bool depthwise = false) {
    auto * weight = sam3_efficient_weight(model, prefix + ".conv.weight");
    x = sam3_efficient_conv_raw(ctx, x, weight, sam3_efficient_weight(model, prefix + ".conv.bias"),
                               stride, depthwise ? (int)x->ne[2] : 1);
    x = sam3_efficient_bn(ctx, x, model, prefix + ".norm");
    return activate ? ggml_hardswish(ctx, x) : x;
}

static ggml_tensor * sam3_efficient_mb(ggml_context * ctx, ggml_tensor * x,
        const sam3_model & model, const std::string & prefix, int stride) {
    if (sam3_efficient_weight(model, prefix + ".inverted_conv.conv.weight"))
        x = sam3_efficient_conv(ctx, x, model, prefix + ".inverted_conv", 1, true);
    x = sam3_efficient_conv(ctx, x, model, prefix + ".depth_conv", stride, true, true);
    return sam3_efficient_conv(ctx, x, model, prefix + ".point_conv");
}

static ggml_tensor * sam3_efficient_attention(ggml_context * ctx, ggml_tensor * x,
        const sam3_model & model, const std::string & prefix) {
    const int D = 16;
    const int64_t W = x->ne[0], H = x->ne[1], N = W * H, C = x->ne[2];
    auto * qkv = sam3_efficient_conv(ctx, x, model, prefix + ".qkv");
    auto * aggregated = sam3_efficient_conv_raw(ctx, qkv,
        sam3_efficient_weight(model, prefix + ".aggreg.0.0.weight"), nullptr, 1, (int)(3 * C));
    aggregated = sam3_efficient_conv_raw(ctx, aggregated,
        sam3_efficient_weight(model, prefix + ".aggreg.0.1.weight"), nullptr, 1, (int)(3 * C / D));
    auto * joined = ggml_concat(ctx, qkv, aggregated, 2);
    const int64_t heads = 2 * C / D;
    joined = ggml_reshape_3d(ctx, joined, N, 3 * D, heads);
    auto * q = ggml_relu(ctx, ggml_cont(ctx, ggml_view_3d(ctx, joined, N, D, heads, joined->nb[1], joined->nb[2], 0)));
    auto * k = ggml_relu(ctx, ggml_cont(ctx, ggml_view_3d(ctx, joined, N, D, heads, joined->nb[1], joined->nb[2], D * joined->nb[1])));
    auto * v = ggml_cont(ctx, ggml_view_3d(ctx, joined, N, D, heads, joined->nb[1], joined->nb[2], 2 * D * joined->nb[1]));
    auto * vk = ggml_mul_mat(ctx, k, v); // [Dk,Dv,heads] = V @ K^T
    q = ggml_cont(ctx, ggml_transpose(ctx, q)); // [D,N,heads]
    auto * numerator = ggml_mul_mat(ctx, vk, q);
    auto * denominator = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, ggml_sum_rows(ctx, k))), q);
    denominator = ggml_add1(ctx, denominator, sam3_efficient_weight(model, "ev.eps"));
    auto * result = ggml_div(ctx, numerator, ggml_repeat(ctx, denominator, numerator));
    result = ggml_cont(ctx, ggml_transpose(ctx, result));
    result = ggml_reshape_3d(ctx, result, W, H, 2 * C);
    return sam3_efficient_conv(ctx, result, model, prefix + ".proj");
}

// Every stage has fresh inputs, context, graph and allocator. Read the output
// to an owned CPU vector before destroying its allocator.
static bool sam3_efficient_run(const sam3_model & model, int threads,
        std::vector<float> & data, int64_t dims[3], const char * name,
        const std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> & build) {
    const auto begin = std::chrono::steady_clock::now();
    auto * ctx = ggml_init({ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true});
    if (!ctx) return false;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dims[0], dims[1], dims[2]);
    ggml_set_input(input);
    auto * output = build(ctx, input);
    ggml_set_name(output, name);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, output);
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
        ok = sam3_graph_compute(model.backend, graph, threads);
    }
    if (ok) {
        data.resize(ggml_nelements(output));
        ggml_backend_tensor_get(output, data.data(), 0, data.size() * sizeof(float));
        for (int i = 0; i < 3; ++i) dims[i] = output->ne[i];
        for (float value : data) if (!std::isfinite(value)) { ok = false; break; }
        const char * trace_dir = std::getenv("SAM3_EFFICIENT_TRACE_DIR");
        if (trace_dir && *trace_dir) {
            std::ofstream trace(std::string(trace_dir) + "/" + name + ".f32", std::ios::binary);
            trace.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
            std::ofstream shape(std::string(trace_dir) + "/" + name + ".shape");
            shape << dims[0] << ' ' << dims[1] << ' ' << dims[2] << '\n';
            if (!trace || !shape) ok = false;
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(ctx);
    if (std::getenv("SAM3_EFFICIENT_PROFILE")) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        fprintf(stderr, "EV-M profile %s: %.3f ms\n", name, ms);
    }
    if (!ok) fprintf(stderr, "EV-M stage failed: %s\n", name);
    return ok;
}

static bool sam3_efficient_text(const sam3_model & model, int threads,
        const std::vector<int32_t> & ids, std::vector<float> & features) {
    std::vector<float> data(ids.begin(), ids.end());
    int64_t dims[3] = {16, 1, 1};
    if (!sam3_efficient_run(model, threads, data, dims, "efficient_text_embedding", [&](ggml_context * ctx, ggml_tensor * input) {
        auto * tokens = ggml_reshape_1d(ctx, ggml_cast(ctx, input, GGML_TYPE_I32), 16);
        auto * x = ggml_get_rows(ctx, sam3_efficient_weight(model, "et.encoder.embedding_layer.weight"), tokens);
        return ggml_add(ctx, x, ggml_reshape_2d(ctx, sam3_efficient_weight(model, "et.encoder.positional_embedding.pos_embed.pos_embed"), 512, 16));
    })) return false;
    for (int block = 0; block < 6; ++block) {
        const std::string p = "et.b." + std::to_string(block);
        if (!sam3_efficient_run(model, threads, data, dims, p.c_str(), [&](ggml_context * ctx, ggml_tensor * input) {
            if (block == 0 || block == 5) {
                auto * x = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, input)), 16, 1, 512);
                auto * mixed = sam3_efficient_bn(ctx, x, model, p + ".tm.mixer.rbr_skip");
                // The pinned (1,11) MobileOne kernel has no rbr_scale branch.
                for (const auto & branch : {std::string(".rbr_conv.0")}) {
                    const std::string bp = p + ".tm.mixer" + branch;
                    auto * y = sam3_efficient_conv_raw(ctx, x, sam3_efficient_weight(model, bp + ".conv.weight"), nullptr, 1, 512);
                    y = sam3_efficient_bn(ctx, y, model, bp + ".bn");
                    mixed = ggml_add(ctx, mixed, y);
                }
                auto * norm = sam3_efficient_bn(ctx, x, model, p + ".tm.norm.rbr_skip");
                auto * scale = ggml_reshape_3d(ctx, sam3_efficient_weight(model, p + ".tm.layer_scale"), 1, 1, 512);
                x = ggml_add(ctx, x, ggml_mul(ctx, ggml_sub(ctx, mixed, norm), scale));
                auto * f = sam3_efficient_conv_raw(ctx, x, sam3_efficient_weight(model, p + ".ff.conv.conv.weight"), nullptr, 1, 512);
                f = sam3_efficient_bn(ctx, f, model, p + ".ff.conv.bn");
                f = sam3_efficient_conv_raw(ctx, f, sam3_efficient_weight(model, p + ".ff.fc1.weight"), sam3_efficient_weight(model, p + ".ff.fc1.bias"));
                f = ggml_gelu_erf(ctx, f);
                f = sam3_efficient_conv_raw(ctx, f, sam3_efficient_weight(model, p + ".ff.fc2.weight"), sam3_efficient_weight(model, p + ".ff.fc2.bias"));
                scale = ggml_reshape_3d(ctx, sam3_efficient_weight(model, p + ".layer_scale"), 1, 1, 512);
                x = ggml_add(ctx, x, ggml_mul(ctx, f, scale));
                return ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, x, 16, 512)));
            }
            sam3_text_block weights;
            weights.ln1_w = sam3_efficient_weight(model, p + ".ln1.weight");
            weights.ln1_b = sam3_efficient_weight(model, p + ".ln1.bias");
            weights.attn_in_proj_w = sam3_efficient_weight(model, p + ".qkv.weight");
            weights.attn_in_proj_b = sam3_efficient_weight(model, p + ".qkv.bias");
            weights.attn_out_proj_w = sam3_efficient_weight(model, p + ".out.weight");
            weights.attn_out_proj_b = sam3_efficient_weight(model, p + ".out.bias");
            weights.ln2_w = sam3_efficient_weight(model, p + ".ln2.weight");
            weights.ln2_b = sam3_efficient_weight(model, p + ".ln2.bias");
            weights.mlp_fc1_w = sam3_efficient_weight(model, p + ".fc1.weight");
            weights.mlp_fc1_b = sam3_efficient_weight(model, p + ".fc1.bias");
            weights.mlp_fc2_w = sam3_efficient_weight(model, p + ".fc2.weight");
            weights.mlp_fc2_b = sam3_efficient_weight(model, p + ".fc2.bias");
            return sam3_text_block_forward(ctx, ggml_reshape_2d(ctx, input, 512, 16), weights, model.hparams, nullptr, block);
        })) return false;
    }
    if (!sam3_efficient_run(model, threads, data, dims, "efficient_text_projection", [&](ggml_context * ctx, ggml_tensor * input) {
        auto * x = sam3_layer_norm(ctx, input, sam3_efficient_weight(model, "et.encoder.final_layer_norm.weight"), sam3_efficient_weight(model, "et.encoder.final_layer_norm.bias"));
        x = ggml_mul_mat(ctx, sam3_efficient_weight(model, "et.projector.weight"), x);
        return ggml_add(ctx, x, sam3_efficient_weight(model, "et.projector.bias"));
    })) return false;
    features = std::move(data);
    return true;
}

static bool sam3_efficient_encode(sam3_state & state, const sam3_model & model,
                                  const float * chw, int original_width, int original_height) {
    std::vector<float> data(chw, chw + 3 * 1008 * 1008);
    int64_t dims[3] = {1008, 1008, 3};
    if (!sam3_efficient_run(model, state.n_threads, data, dims, "efficient_stem", [&](ggml_context * ctx, ggml_tensor * input) {
        auto * x = sam3_efficient_conv(ctx, input, model, "ev.stem.0", 2, true);
        return ggml_add(ctx, x, sam3_efficient_mb(ctx, x, model, "ev.stem.1", 1));
    })) return false;
    const int depth[4] = {2, 3, 4, 5};
    for (int stage = 0; stage < 4; ++stage) {
        for (int block = 0; block < depth[stage]; ++block) {
            const std::string p = "ev.s." + std::to_string(stage) + "." + std::to_string(block);
            if (!sam3_efficient_run(model, state.n_threads, data, dims, p.c_str(), [&](ggml_context * ctx, ggml_tensor * input) {
                if (stage >= 2 && block > 0) {
                    auto * x = ggml_add(ctx, input, sam3_efficient_attention(ctx, input, model, p + ".att"));
                    return ggml_add(ctx, x, sam3_efficient_mb(ctx, x, model, p + ".mb", 1));
                }
                auto * x = sam3_efficient_mb(ctx, input, model, p, block == 0 ? 2 : 1);
                return block > 0 ? ggml_add(ctx, x, input) : x;
            })) return false;
        }
    }
    if (!sam3_efficient_run(model, state.n_threads, data, dims, "efficient_projection", [&](ggml_context * ctx, ggml_tensor * input) {
        auto * x = sam3_efficient_conv_raw(ctx, input, sam3_efficient_weight(model, "ev.head.0.weight"), nullptr);
        x = sam3_efficient_bn(ctx, x, model, "ev.head.1");
        x = ggml_gelu_erf(ctx, x);
        x = sam3_efficient_conv_raw(ctx, x, sam3_efficient_weight(model, "ev.head.3.weight"), sam3_efficient_weight(model, "ev.head.3.bias"));
        return ggml_interpolate(ctx, x, 72, 72, 1024, 1, GGML_SCALE_MODE_BILINEAR);
    })) return false;
    // Build the neck from a fresh CPU-backed trunk input. Persist only this
    // graph's FPN outputs; no previous graph tensors cross stage boundaries.
    auto * ctx = ggml_init({ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false), nullptr, true});
    if (!ctx) return false;
    auto * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 72, 72, 1024, 1);
    ggml_set_input(input);
    auto * trunk = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
    ggml_tensor * outputs[4] = {};
    sam3_build_neck_graph(ctx, trunk, model.neck_det, outputs);
    auto * graph = ggml_new_graph_custom(ctx, 8192, false);
    for (auto * out : outputs) { ggml_set_output(out); ggml_build_forward_expand(graph, out); }
    auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    if (ok) {
        ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
        ok = sam3_graph_compute(model.backend, graph, state.n_threads);
    }
    if (!ok) { ggml_gallocr_free(allocator); ggml_free(ctx); return false; }
    if (state.galloc) ggml_gallocr_free(state.galloc);
    if (state.ctx) ggml_free(state.ctx);
    state.ctx = ctx; state.galloc = allocator; state.vit_output = nullptr;
    state.orig_width = original_width; state.orig_height = original_height;
    for (int i = 0; i < 4; ++i) { state.neck_det[i] = outputs[i]; state.neck_trk[i] = nullptr; }
    if (state.pe_buf) ggml_backend_buffer_free(state.pe_buf);
    if (state.pe_ctx) ggml_free(state.pe_ctx);
    state.pe_ctx = ggml_init({ggml_tensor_overhead() * 4 + 256, nullptr, true});
    const int sizes[4] = {288, 144, 72, 36};
    for (int i = 0; i < 4; ++i) state.neck_det_pe[i] = ggml_new_tensor_3d(state.pe_ctx, GGML_TYPE_F32, 256, sizes[i], sizes[i]);
    state.pe_buf = ggml_backend_alloc_ctx_tensors(state.pe_ctx, model.backend);
    if (!state.pe_buf) return false;
    for (int i = 0; i < 4; ++i) {
        auto pe = sam3_sinusoidal_pe_2d(sizes[i], sizes[i], 256);
        ggml_backend_tensor_set(state.neck_det_pe[i], pe.data(), 0, pe.size() * sizeof(float));
    }
    return true;
}
