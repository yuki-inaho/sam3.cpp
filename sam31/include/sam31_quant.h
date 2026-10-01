#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(SAM31_QUANT_STATIC)
#  define SAM31_API
#elif defined(_WIN32)
#  if defined(SAM31_QUANT_BUILD)
#    define SAM31_API __declspec(dllexport)
#  else
#    define SAM31_API __declspec(dllimport)
#  endif
#else
#  define SAM31_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* These are native W8A32 reference primitives, not an entire SAM 3.1 graph.
 * Row-major arrays; group_size=0 means plain INT8, otherwise a power of four.
 * All lengths are element counts. Return 0 on success, 1 on invalid input,
 * 2 on allocation failure. Input/output buffers must not overlap.
 */
SAM31_API int sam31_dequantize_linear(
    const int8_t * weights, size_t weight_count,
    const float * scales, size_t scale_count,
    size_t rows, size_t cols, size_t group_size,
    float * output, size_t output_count);

SAM31_API int sam31_dequantize_conv2d(
    const int8_t * weights, size_t weight_count,
    const float * scales, size_t scale_count,
    size_t out_channels, size_t in_channels, size_t kernel_h, size_t kernel_w,
    size_t group_size, float * output, size_t output_count);

SAM31_API int sam31_linear_w8a32(
    const int8_t * weights, size_t weight_count,
    const float * scales, size_t scale_count,
    const float * bias, size_t bias_count,
    const float * input, size_t input_count,
    size_t batch, size_t rows, size_t cols, size_t group_size,
    float * output, size_t output_count);

#ifdef __cplusplus
}
#endif
