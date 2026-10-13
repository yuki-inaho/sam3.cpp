#include "jpeg_decode.h"

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>

extern "C" {
#include <jpeglib.h>
}

namespace {
// Keep every object mutated after setjmp on the heap. No C++ destructor is
// skipped by libjpeg's error longjmp, including partial create/decode failures.
struct sam3_jpeg_state {
    jpeg_decompress_struct decoder;
    jpeg_error_mgr error;
    jmp_buf recovery;
    FILE * file;
    uint8_t * pixels;
    char message[JMSG_LENGTH_MAX];
};

void sam3_jpeg_fail(j_common_ptr decoder) {
    auto * state = reinterpret_cast<sam3_jpeg_state *>(decoder);
    (*decoder->err->format_message)(decoder, state->message);
    longjmp(state->recovery, 1);
}

void sam3_jpeg_message(j_common_ptr decoder, int level) {
    // jpeg_stdio_src would otherwise silently synthesize EOI on truncation.
    // Strict failure is intentional; never use a partly decoded video frame.
    if (level < 0) sam3_jpeg_fail(decoder);
}

void sam3_jpeg_release(sam3_jpeg_state * state) {
    jpeg_destroy_decompress(&state->decoder);
    if (state->file) std::fclose(state->file);
    std::free(state->pixels);
    std::free(state);
}
}

namespace efficient_sam2 {
bool decode_jpeg_rgb(const std::string & path, int & width, int & height,
                     std::vector<uint8_t> & rgb, std::string & error) {
    width = height = 0;
    rgb.clear();
    error.clear();
    auto * state = static_cast<sam3_jpeg_state *>(std::calloc(1, sizeof(sam3_jpeg_state)));
    if (!state) { error = "JPEG decoder allocation failed"; return false; }
    state->decoder.err = jpeg_std_error(&state->error);
    state->error.error_exit = sam3_jpeg_fail;
    state->error.emit_message = sam3_jpeg_message;
    if (setjmp(state->recovery)) {
        error = state->message;
        sam3_jpeg_release(state);
        return false;
    }
    jpeg_create_decompress(&state->decoder);
    state->file = std::fopen(path.c_str(), "rb");
    if (!state->file) {
        error = "Cannot open JPEG: " + path;
        sam3_jpeg_release(state);
        return false;
    }
    jpeg_stdio_src(&state->decoder, state->file);
    jpeg_read_header(&state->decoder, TRUE);
    const auto space = state->decoder.jpeg_color_space;
    if (state->decoder.data_precision != 8 ||
        (space != JCS_GRAYSCALE && space != JCS_RGB && space != JCS_YCbCr)) {
        error = "Video JPEG requires 8-bit grayscale, RGB or YCbCr (CMYK/YCCK unsupported)";
        sam3_jpeg_release(state);
        return false;
    }
    // These are libjpeg/Pillow's normal (non-draft) defaults, made explicit.
    state->decoder.out_color_space = JCS_RGB;
    state->decoder.dct_method = JDCT_ISLOW;
    state->decoder.do_fancy_upsampling = TRUE;
    state->decoder.do_block_smoothing = TRUE;
    state->decoder.scale_num = state->decoder.scale_denom = 1;
    const size_t w = state->decoder.image_width;
    const size_t h = state->decoder.image_height;
    constexpr size_t max_rgb_bytes = 512u * 1024u * 1024u;
    if (!w || !h || w > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        h > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        w > max_rgb_bytes / 3 / h) {
        error = "JPEG dimensions exceed the 512 MiB video RGB limit";
        sam3_jpeg_release(state);
        return false;
    }
    const size_t bytes = w * h * 3;
    state->pixels = static_cast<uint8_t *>(std::malloc(bytes));
    if (!state->pixels) {
        error = "JPEG pixel allocation failed";
        sam3_jpeg_release(state);
        return false;
    }
    jpeg_start_decompress(&state->decoder);
    while (state->decoder.output_scanline < state->decoder.output_height) {
        JSAMPROW row = state->pixels + size_t(state->decoder.output_scanline) * w * 3;
        jpeg_read_scanlines(&state->decoder, &row, 1);
    }
    jpeg_finish_decompress(&state->decoder);
    // No further libjpeg call can fail/longjmp. Only now introduce a C++
    // destructor, so vector allocation exceptions release all codec resources
    // before propagating (consistent with ordinary std::vector allocation).
    const std::unique_ptr<sam3_jpeg_state, decltype(&sam3_jpeg_release)>
        cleanup(state, sam3_jpeg_release);
    rgb.assign(state->pixels, state->pixels + bytes);
    width = static_cast<int>(w);
    height = static_cast<int>(h);
    return true;
}
}
