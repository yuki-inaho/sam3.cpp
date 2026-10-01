#include "sam31_quant.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <vector>

namespace {
bool multiply(size_t a, size_t b, size_t & result) {
    if (!a || !b || a > std::numeric_limits<size_t>::max() / b) return false;
    result = a * b;
    return true;
}

bool valid_group(size_t group, size_t cols) {
    if (group == 0) return true;
    if (group < 4 || cols % group != 0) return false;
    while (group > 1 && group % 4 == 0) group /= 4;
    return group == 1;
}

bool valid_scales(const float * scales, size_t count, size_t rows) {
    if (!scales || (count != 1 && count != rows)) return false;
    for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(scales[i]) || scales[i] <= 0.0f) return false;
    return true;
}

void transform(float * values, size_t cols, size_t group) {
    if (group == 0) return;
    const float normalization = 1.0f / std::sqrt(static_cast<float>(group));
    for (size_t start = 0; start < cols; start += group) {
        for (size_t stride = 1; stride < group; stride *= 4) {
            for (size_t block = 0; block < group; block += 4 * stride) {
                for (size_t j = 0; j < stride; ++j) {
                    float * p = values + start + block + j;
                    const float a = p[0], b = p[stride], c = p[2 * stride], d = p[3 * stride];
                    // The regular H4 basis has its negative entries on the anti-diagonal.
                    const float sum = ((a + b) + c) + d;
                    p[0] = sum - 2.0f * d;
                    p[stride] = sum - 2.0f * c;
                    p[2 * stride] = sum - 2.0f * b;
                    p[3 * stride] = sum - 2.0f * a;
                }
            }
        }
        for (size_t j = 0; j < group; ++j) values[start + j] *= normalization;
    }
}
}

extern "C" int sam31_dequantize_linear(
    const int8_t * weights, size_t weight_count, const float * scales, size_t scale_count,
    size_t rows, size_t cols, size_t group_size, float * output, size_t output_count) {
    size_t total;
    if (!weights || !output || !multiply(rows, cols, total) || total != weight_count ||
        total != output_count || !valid_group(group_size, cols) ||
        !valid_scales(scales, scale_count, rows)) return 1;
    for (size_t row = 0; row < rows; ++row) {
        const float scale = scales[scale_count == 1 ? 0 : row];
        for (size_t col = 0; col < cols; ++col)
            output[row * cols + col] = static_cast<float>(weights[row * cols + col]) * scale;
        transform(output + row * cols, cols, group_size);
        for (size_t col = 0; col < cols; ++col)
            if (!std::isfinite(output[row * cols + col])) return 1;
    }
    return 0;
}

extern "C" int sam31_dequantize_conv2d(
    const int8_t * weights, size_t weight_count, const float * scales, size_t scale_count,
    size_t out_channels, size_t in_channels, size_t kernel_h, size_t kernel_w,
    size_t group_size, float * output, size_t output_count) {
    size_t spatial, channels, total;
    if (!weights || !output || !multiply(kernel_h, kernel_w, spatial) ||
        !multiply(out_channels, in_channels, channels) || !multiply(channels, spatial, total) ||
        total != weight_count || total != output_count || !valid_group(group_size, in_channels) ||
        !valid_scales(scales, scale_count, out_channels)) return 1;
    try {
        std::vector<float> row(in_channels);
        for (size_t o = 0; o < out_channels; ++o) {
            const float scale = scales[scale_count == 1 ? 0 : o];
            for (size_t xy = 0; xy < spatial; ++xy) {
                for (size_t i = 0; i < in_channels; ++i)
                    row[i] = static_cast<float>(weights[(o * in_channels + i) * spatial + xy]) * scale;
                transform(row.data(), in_channels, group_size);
                for (size_t i = 0; i < in_channels; ++i) {
                    if (!std::isfinite(row[i])) return 1;
                    output[(o * in_channels + i) * spatial + xy] = row[i];
                }
            }
        }
    } catch (const std::bad_alloc &) { return 2; }
    catch (...) { return 1; }
    return 0;
}

extern "C" int sam31_linear_w8a32(
    const int8_t * weights, size_t weight_count, const float * scales, size_t scale_count,
    const float * bias, size_t bias_count, const float * input, size_t input_count,
    size_t batch, size_t rows, size_t cols, size_t group_size,
    float * output, size_t output_count) {
    size_t nw, ni, no;
    if (!weights || !input || !output || !multiply(rows, cols, nw) || !multiply(batch, cols, ni) ||
        !multiply(batch, rows, no) || nw != weight_count || ni != input_count || no != output_count ||
        !valid_group(group_size, cols) || !valid_scales(scales, scale_count, rows) ||
        (bias ? bias_count != rows : bias_count != 0)) return 1;
    for (size_t i = 0; i < input_count; ++i) if (!std::isfinite(input[i])) return 1;
    for (size_t i = 0; i < bias_count; ++i) if (!std::isfinite(bias[i])) return 1;
    try {
        std::vector<float> activation(cols);
        for (size_t b = 0; b < batch; ++b) {
            std::copy(input + b * cols, input + (b + 1) * cols, activation.begin());
            transform(activation.data(), cols, group_size);
            for (size_t row = 0; row < rows; ++row) {
                double accumulator = 0.0;
                for (size_t col = 0; col < cols; ++col)
                    accumulator += static_cast<double>(activation[col]) * weights[row * cols + col];
                const float scale = scales[scale_count == 1 ? 0 : row];
                output[b * rows + row] = static_cast<float>(accumulator * scale + (bias ? bias[row] : 0.0f));
                if (!std::isfinite(output[b * rows + row])) return 1;
            }
        }
    } catch (const std::bad_alloc &) { return 2; }
    catch (...) { return 1; }
    return 0;
}
