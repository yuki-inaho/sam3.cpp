// EV-M GGUF loading, included inside the sam3 implementation.
struct sam3_efficient_spec {
    const char * name;
    int type;
    int64_t ne[4];
};
static const sam3_efficient_spec sam3_efficient_specs[] = {
#include "ev_m_tensor_spec.inc"
};

static bool sam3_efficient_string(const gguf_context * ctx, const char * key, const char * expected) {
    const auto id = gguf_find_key(ctx, key);
    return id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING &&
           std::strcmp(gguf_get_val_str(ctx, id), expected) == 0;
}

static std::shared_ptr<sam3_model> sam3_load_efficient_gguf(const sam3_params & params) {
    ggml_context * file_ctx = nullptr;
    gguf_context * raw = gguf_init_from_file(params.model_path.c_str(), {true, &file_ctx});
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(raw, gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> file_tensors(file_ctx, ggml_free);
    if (!raw || !file_ctx) return nullptr;
    const auto count = sizeof(sam3_efficient_specs) / sizeof(sam3_efficient_specs[0]);
    const auto context_id = gguf_find_key(raw, "efficientsam3.context");
    if (!sam3_efficient_string(raw, "general.architecture", "efficientsam3") ||
        !sam3_efficient_string(raw, "efficientsam3.variant", "ev-m-stage3") ||
        !sam3_efficient_string(raw, "efficientsam3.source_revision", "bd0936c788fed8d51fa799437f05abd97b401b06") ||
        !sam3_efficient_string(raw, "efficientsam3.hf_revision", "85b05896928f974e308f889d7ccb2eefc069de98") ||
        !sam3_efficient_string(raw, "efficientsam3.checkpoint_sha256", "086b04b2e7da7cc98aa4621b70c7291608aa9d187357b98d03bd4d6533ed5a17") ||
        !sam3_efficient_string(raw, "efficientsam3.sequence_mode", "independent_detection") ||
        context_id < 0 || gguf_get_kv_type(raw, context_id) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(raw, context_id) != 16 || gguf_get_n_tensors(raw) != (int64_t)count) {
        fprintf(stderr, "EV-M GGUF: invalid model identity/context/tensor count\n");
        return nullptr;
    }
    for (const auto & spec : sam3_efficient_specs) {
        auto * tensor = ggml_get_tensor(file_ctx, spec.name);
        if (!tensor || tensor->type != (ggml_type)spec.type ||
            !std::equal(spec.ne, spec.ne + 4, tensor->ne)) {
            fprintf(stderr, "EV-M GGUF: missing or incompatible tensor %s\n", spec.name);
            return nullptr;
        }
    }
    auto model = std::make_shared<sam3_model>();
    model->hparams.model_type = SAM3_MODEL_EFFICIENTSAM3;
    model->hparams.text_width = 512;
    model->hparams.text_heads = 8;
    model->hparams.text_layers = 4;
    model->hparams.text_ctx_len = 16;
    model->weight_type = GGML_TYPE_F32;
    model->backend = ggml_backend_cpu_init();
    model->ctx = ggml_init({ggml_tensor_overhead() * 4096 + ggml_graph_overhead(), nullptr, true});
    if (!model->backend || !model->ctx) { sam3_free_model(*model); return nullptr; }
    sam3_register_tensors(*model);
    // Common PCS tensors are bound by the existing registration; student tensors
    // are created from the checked, compiled specification above.
    for (const auto & spec : sam3_efficient_specs) {
        auto * source = ggml_get_tensor(file_ctx, spec.name);
        const auto existing = model->tensors.find(spec.name);
        if (existing == model->tensors.end()) {
            auto * target = ggml_dup_tensor(model->ctx, source);
            ggml_set_name(target, spec.name);
            model->tensors[spec.name] = target;
        } else if (ggml_nbytes(existing->second) != ggml_nbytes(source)) {
            fprintf(stderr, "EV-M GGUF: incompatible PCS registration %s\n", spec.name);
            sam3_free_model(*model); return nullptr;
        }
    }
    for (const auto & entry : model->tensors) {
        if (!ggml_get_tensor(file_ctx, entry.first.c_str())) {
            fprintf(stderr, "EV-M GGUF: required PCS tensor absent: %s\n", entry.first.c_str());
            sam3_free_model(*model); return nullptr;
        }
    }
    // Persistent BN affine parameters avoid recomputing sqrt/scale in hot graphs.
    std::vector<std::string> bn;
    for (const auto & entry : model->tensors) {
        const auto & name = entry.first;
        if (name.size() > 3 && name.substr(name.size() - 3) == ".rm") bn.push_back(name.substr(0, name.size() - 3));
    }
    for (const auto & prefix : bn) {
        const int64_t channels = model->tensors.at(prefix + ".rm")->ne[0];
        for (const auto * suffix : {".scale", ".shift"})
            model->tensors[prefix + suffix] = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, channels);
    }
    model->tensors["ev.eps"] = ggml_new_tensor_1d(model->ctx, GGML_TYPE_F32, 1);
    model->buffer = ggml_backend_alloc_ctx_tensors(model->ctx, model->backend);
    if (!model->buffer) { sam3_free_model(*model); return nullptr; }
    std::ifstream file(params.model_path, std::ios::binary);
    for (const auto & spec : sam3_efficient_specs) {
        const auto id = gguf_find_tensor(raw, spec.name);
        const size_t size = gguf_get_tensor_size(raw, id);
        std::vector<char> bytes(size);
        file.seekg(gguf_get_data_offset(raw) + gguf_get_tensor_offset(raw, id));
        file.read(bytes.data(), size);
        if (!file) { sam3_free_model(*model); return nullptr; }
        if (spec.type == GGML_TYPE_F32) {
            for (size_t offset = 0; offset < size; offset += sizeof(float)) {
                float value; std::memcpy(&value, bytes.data() + offset, sizeof(float));
                if (!std::isfinite(value)) { sam3_free_model(*model); return nullptr; }
            }
        }
        ggml_backend_tensor_set(model->tensors.at(spec.name), bytes.data(), 0, size);
    }
    for (const auto & prefix : bn) {
        const int64_t channels = model->tensors.at(prefix + ".rm")->ne[0];
        std::vector<float> mean(channels), var(channels), weight(channels), bias(channels);
        ggml_backend_tensor_get(model->tensors.at(prefix + ".rm"), mean.data(), 0, channels * 4);
        ggml_backend_tensor_get(model->tensors.at(prefix + ".rv"), var.data(), 0, channels * 4);
        ggml_backend_tensor_get(model->tensors.at(prefix + ".weight"), weight.data(), 0, channels * 4);
        ggml_backend_tensor_get(model->tensors.at(prefix + ".bias"), bias.data(), 0, channels * 4);
        for (int64_t i = 0; i < channels; ++i) {
            if (var[i] < 0) { sam3_free_model(*model); return nullptr; }
            weight[i] /= std::sqrt(var[i] + 1e-5f);
            bias[i] -= mean[i] * weight[i];
        }
        ggml_backend_tensor_set(model->tensors.at(prefix + ".scale"), weight.data(), 0, channels * 4);
        ggml_backend_tensor_set(model->tensors.at(prefix + ".shift"), bias.data(), 0, channels * 4);
    }
    const float eps = 1e-15f;
    ggml_backend_tensor_set(model->tensors.at("ev.eps"), &eps, 0, sizeof(eps));
    const auto vocab = gguf_find_key(raw, "tokenizer.ggml.tokens");
    const auto merges = gguf_find_key(raw, "tokenizer.ggml.merges");
    if (vocab < 0 || merges < 0 || gguf_get_kv_type(raw, vocab) != GGUF_TYPE_ARRAY ||
        gguf_get_kv_type(raw, merges) != GGUF_TYPE_ARRAY || gguf_get_arr_type(raw, vocab) != GGUF_TYPE_STRING ||
        gguf_get_arr_type(raw, merges) != GGUF_TYPE_STRING || gguf_get_arr_n(raw, vocab) != 49408 ||
        gguf_get_arr_n(raw, merges) > 50000) { sam3_free_model(*model); return nullptr; }
    auto & tokenizer = model->tokenizer;
    for (size_t i = 0; i < gguf_get_arr_n(raw, vocab); ++i) {
        const std::string token = gguf_get_arr_str(raw, vocab, i);
        tokenizer.encoder[token] = (int)i;
        tokenizer.decoder[(int)i] = token;
    }
    for (size_t i = 0; i < gguf_get_arr_n(raw, merges); ++i) {
        const std::string text = gguf_get_arr_str(raw, merges, i);
        const auto split = text.find(' ');
        if (split == std::string::npos) { sam3_free_model(*model); return nullptr; }
        auto a = text.substr(0, split), b = text.substr(split + 1);
        tokenizer.merge_ranks[sam3_merge_key(a, b)] = (int)i;
        tokenizer.merges.push_back({std::move(a), std::move(b)});
    }
    sam3_init_byte_encoder(tokenizer.byte_encoder);
    fprintf(stderr, "EV-M GGUF: loaded %zu trained tensors; native ggml CPU\n", count);
    return model;
}
