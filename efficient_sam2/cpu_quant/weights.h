#pragma once
// Experimental second quantization of already-restored ConvRot weights.
// Opt-in only: this is not the original ConvRot W8A16 representation.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam2_cpu_quant {
struct packed_weight {
    ggml_type type = GGML_TYPE_F32;
    int64_t cols = 0, rows = 0;
    std::string reason;
    std::vector<uint8_t> bytes;
};
// No padding, custom ISA, activation transform, or per-inference conversion.
// Consumers keep these weights persistent and call ordinary ggml_mul_mat.
// values must point to exactly count contiguous row-major floats. A raw pointer
// cannot describe strides; callers must materialize noncontiguous tensors first.
inline packed_weight pack(const float * values, size_t count, int64_t cols,
                          int64_t rows, bool enable_q8 = false) {
    if (!values || cols <= 0 || rows <= 0 ||
        uint64_t(cols) > uint64_t(std::numeric_limits<int64_t>::max()) / uint64_t(rows) ||
        uint64_t(cols) * uint64_t(rows) != count ||
        count > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::invalid_argument("Invalid linear weight shape/count");
    bool scale_safe = true;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) throw std::invalid_argument("Non-finite linear weight");
        // Q8_0 stores its block scale in FP16. Avoid scale overflow.
        scale_safe = scale_safe && std::abs(values[i]) <= 65504.0f * 127.0f;
    }
    packed_weight out;
    out.cols = cols; out.rows = rows;
    const auto block = ggml_blck_size(GGML_TYPE_Q8_0);
    if (!enable_q8) out.reason = "disabled";
    else if (cols % block) out.reason = "input-width-not-block-aligned";
    else if (!scale_safe) out.reason = "fp16-scale-overflow";
    else {
        bool scale_underflow = false;
        for (size_t start = 0; start < count; start += size_t(block)) {
            float maximum = 0;
            for (int64_t j = 0; j < block; ++j) maximum = std::max(maximum, std::abs(values[start+size_t(j)]));
            if (maximum > 0 && ggml_fp16_to_fp32(ggml_fp32_to_fp16(maximum/127.0f)) == 0)
                scale_underflow = true;
        }
        if (scale_underflow) out.reason = "fp16-scale-underflow";
        else { out.type = GGML_TYPE_Q8_0; out.reason = "experimental-second-quantization"; }
    }
    out.bytes.resize(ggml_row_size(out.type, cols) * size_t(rows));
    if (out.type == GGML_TYPE_F32) std::memcpy(out.bytes.data(), values, out.bytes.size());
    else {
        const size_t written = ggml_quantize_chunk(out.type, values, out.bytes.data(), 0, rows, cols, nullptr);
        if (written != out.bytes.size()) throw std::runtime_error("ggml quantization size mismatch");
    }
    return out;
}
} // namespace sam2_cpu_quant
