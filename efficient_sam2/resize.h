#pragma once
#include <cstdint>
#include <vector>

namespace efficient_sam2 {
// Pillow RGB Image.resize default: antialiased bicubic, a=-0.5, separable
// uint8 passes with 22-bit coefficient rounding. See NOTICE.md for provenance.
std::vector<uint8_t> resize_video_rgb(const uint8_t* rgb,int width,int height,
                                    int output_width,int output_height);
}
