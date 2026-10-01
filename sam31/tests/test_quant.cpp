#include "sam31_quant.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
size_t checks = 0;
void require(bool okay, const char * message) { ++checks; if (!okay) throw std::runtime_error(message); }
bool close(float got, double expected) {
    return std::isfinite(got) && std::fabs(got - expected) <= 2e-4 + std::fabs(expected) * 2e-5;
}
// Independent oracle: a product of matrix entries, not the radix-four implementation.
double h(size_t row, size_t col, size_t group) {
    double sign = 1;
    for (size_t digits = group; digits > 1; digits /= 4) {
        if (row % 4 + col % 4 == 3) sign = -sign;
        row /= 4; col /= 4;
    }
    return sign / std::sqrt(static_cast<double>(group));
}
void linear(size_t group) {
    const size_t rows = 5, cols = group ? 2 * group : 11, batch = 3;
    std::vector<int8_t> weights(rows * cols);
    std::vector<float> scales(rows), restored(rows * cols), input(batch * cols), out(batch * rows), bias(rows);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<int8_t>(int((i * 17 + 3) % 255) - 127);
    for (size_t i = 0; i < rows; ++i) { scales[i] = 0.003f * (i + 1); bias[i] = float(i) / 10; }
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) / 20;
    require(sam31_dequantize_linear(weights.data(), weights.size(), scales.data(), scales.size(),
        rows, cols, group, restored.data(), restored.size()) == 0, "linear dequantization failed");
    for (size_t r = 0; r < rows; ++r) for (size_t c = 0; c < cols; ++c) {
        double expected = 0;
        if (!group) expected = weights[r * cols + c] * scales[r];
        else for (size_t k = 0; k < group; ++k)
            expected += weights[r * cols + (c / group) * group + k] * scales[r] * h(k, c % group, group);
        require(close(restored[r * cols + c], expected), "regular Hadamard basis mismatch");
    }
    require(sam31_linear_w8a32(weights.data(), weights.size(), scales.data(), scales.size(), bias.data(), bias.size(),
        input.data(), input.size(), batch, rows, cols, group, out.data(), out.size()) == 0, "W8A32 failed");
    for (size_t b = 0; b < batch; ++b) for (size_t r = 0; r < rows; ++r) {
        double expected = bias[r];
        for (size_t c = 0; c < cols; ++c) expected += static_cast<double>(input[b * cols + c]) * restored[r * cols + c];
        require(close(out[b * rows + r], expected), "online rotation disagrees with restored weight");
    }
    require(sam31_dequantize_linear(weights.data(), weights.size() - 1, scales.data(), scales.size(),
        rows, cols, group, restored.data(), restored.size()) == 1, "bad weight size accepted");
    require(sam31_dequantize_linear(weights.data(), weights.size(), scales.data(), scales.size(),
        rows, cols, 2, restored.data(), restored.size()) == 1, "invalid group accepted");
    scales[0] = -1;
    require(sam31_dequantize_linear(weights.data(), weights.size(), scales.data(), scales.size(),
        rows, cols, group, restored.data(), restored.size()) == 1, "negative scale accepted");
}
void convolution() {
    const size_t oc = 3, ic = 16, kh = 3, kw = 2, xy = kh * kw;
    std::vector<int8_t> weights(oc * ic * xy);
    std::vector<float> result(weights.size()), scales(oc, 0.02f);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<int8_t>(int(i % 101) - 50);
    require(sam31_dequantize_conv2d(weights.data(), weights.size(), scales.data(), scales.size(),
        oc, ic, kh, kw, 16, result.data(), result.size()) == 0, "Conv2d dequantization failed");
    for (size_t o = 0; o < oc; ++o) for (size_t i = 0; i < ic; ++i) for (size_t p = 0; p < xy; ++p) {
        double expected = 0;
        for (size_t k = 0; k < ic; ++k) expected += weights[(o * ic + k) * xy + p] * scales[o] * h(k, i, 16);
        require(close(result[(o * ic + i) * xy + p], expected), "wrong Conv2d channel axis");
    }
    require(sam31_dequantize_conv2d(weights.data(), weights.size(), scales.data(), scales.size(),
        oc, ic, 0, kw, 16, result.data(), result.size()) == 1, "empty kernel accepted");
}
}
int main() {
    try {
        for (size_t group : {size_t(0), size_t(4), size_t(16), size_t(64), size_t(256)}) linear(group);
        convolution();
        std::cout << "PASS: " << checks << " native numerical assertions (synthetic weights; NOT SAM 3.1 inference)\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
