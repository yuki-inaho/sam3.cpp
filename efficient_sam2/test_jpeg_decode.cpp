#include "jpeg_decode.h"
#include <cstdio>
#include <string>
#include <vector>

// Standalone oracle adapter: emit a binary PPM; Python checks every RGB byte.
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    int width = 99, height = 99;
    std::vector<uint8_t> rgb(3, 99);
    std::string error = "stale";
    if (!efficient_sam2::decode_jpeg_rgb(argv[1], width, height, rgb, error)) {
        if (width != 0 || height != 0 || !rgb.empty() || error.empty()) return 4;
        std::fprintf(stderr, "%s\n", error.c_str());
        return 3;
    }
    if (!error.empty() || width <= 0 || height <= 0 ||
        rgb.size() != size_t(width) * size_t(height) * 3) return 4;
    FILE * out = std::fopen(argv[2], "wb");
    if (!out) return 5;
    std::fprintf(out, "P6\n%d %d\n255\n", width, height);
    const bool ok = std::fwrite(rgb.data(), 1, rgb.size(), out) == rgb.size();
    return std::fclose(out) == 0 && ok ? 0 : 5;
}
