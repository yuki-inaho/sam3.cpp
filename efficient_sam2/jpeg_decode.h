#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace efficient_sam2 {
// Full-resolution 8-bit grayscale/RGB/YCbCr JPEG -> interleaved RGB, matching
// Pillow Image.open(...).convert("RGB") without draft/EXIF/ICC transforms.
// Rejects CMYK/YCCK, corrupt/truncated input and decoded RGB above 512 MiB.
// On failure, dimensions are zero, rgb is empty, and error is nonempty.
// This opt-in video decoder does not change the legacy stb image loader.
bool decode_jpeg_rgb(const std::string & path, int & width, int & height,
                     std::vector<uint8_t> & rgb, std::string & error);
}
