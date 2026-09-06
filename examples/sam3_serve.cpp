/**
 * sam3_serve — persistent prompt server for interactive annotation.
 *
 * sam3_seg re-encodes the image on every invocation, which costs seconds and
 * makes click-to-segment unusable.  This process encodes once and then stays
 * alive, answering prompt after prompt from stdin.
 *
 * Protocol: one command per stdin line, one JSON object per stdout line.
 * Whitespace-delimited so no JSON parser is needed on this side (the library
 * may only depend on ggml, stb and the C++14 standard library).
 *
 *   load <path>                  encode an image (slow; do this once)
 *   point <x> <y> <label>         queue a prompt point, label 1=fg 0=bg
 *   box <x0> <y0> <x1> <y1>       queue a prompt box
 *   multimask <0|1>               ask the decoder for 3 candidates
 *   segment                       run the queued prompt, then clear it
 *                                 (needs a positive point or a box)
 *   reset                         clear the queued prompt
 *   quit                          exit
 *
 * Replies are {"ok":true,...} or {"ok":false,"error":"..."}.  Masks are
 * COCO-style uncompressed RLE: column-major run lengths starting with a run
 * of zeros, which is what COCO/CVAT consumers expect.
 *
 * Diagnostics go to stderr; stdout carries only protocol lines.
 */

#include "sam3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path> [--cpu] [--n-threads <n>]\n"
        "\n"
        "Reads commands on stdin, writes one JSON line per command on stdout.\n"
        "Commands: load <path> | point <x> <y> <label> | box <x0> <y0> <x1> <y1>\n"
        "          | multimask <0|1> | segment | reset | quit\n",
        prog);
}

// Minimal JSON string escaping — paths are the only free-form text we emit.
static std::string sam3_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static void sam3_reply_error(const std::string & msg) {
    printf("{\"ok\":false,\"error\":\"%s\"}\n", sam3_json_escape(msg).c_str());
    fflush(stdout);
}

// COCO uncompressed RLE: column-major, alternating run lengths, first run is
// background.  Emitted as a bare JSON array so the caller can drop it
// straight into a COCO `segmentation.counts`.
//
// The mask arrives row-major but COCO wants column-major runs.  Reading it
// column by column would stride by `width` on every pixel and miss the cache
// on nearly all of them; at 12 Mpx that dominates the whole reply.  Transpose
// in cache-sized tiles first, then the run scan is sequential.
static void sam3_transpose_tiled(const uint8_t * src, uint8_t * dst, int w, int h) {
    const int TILE = 64;
    for (int y0 = 0; y0 < h; y0 += TILE) {
        const int y1 = (y0 + TILE < h) ? y0 + TILE : h;
        for (int x0 = 0; x0 < w; x0 += TILE) {
            const int x1 = (x0 + TILE < w) ? x0 + TILE : w;
            for (int y = y0; y < y1; ++y) {
                const uint8_t * srow = src + (size_t)y * w;
                for (int x = x0; x < x1; ++x) {
                    dst[(size_t)x * h + y] = srow[x] ? 1 : 0;
                }
            }
        }
    }
}

// Append a non-negative integer without going through ostream formatting,
// which costs several times more per run than this does.
static void sam3_append_uint(std::string & out, uint64_t v) {
    char  buf[24];
    int   n = 0;
    do { buf[n++] = char('0' + (v % 10)); v /= 10; } while (v);
    while (n) out += buf[--n];
}

static std::string sam3_mask_to_rle(const sam3_mask & mask) {
    const int    w = mask.width;
    const int    h = mask.height;
    const size_t n = (size_t)w * (size_t)h;

    std::string out;
    if (w <= 0 || h <= 0 || mask.data.size() < n) {
        // Nothing trustworthy to encode; an empty counts array is still valid
        // COCO for an empty mask, and the caller sees mask_size to match.
        out = "[]";
        return out;
    }

    std::vector<uint8_t> col(n);
    sam3_transpose_tiled(mask.data.data(), col.data(), w, h);

    // Runs alternate and the first is background, so the count never exceeds
    // the pixel count; a few bytes per run is a safe starting reservation.
    out.reserve(1024);
    out += '[';

    uint8_t  run_value = 0;   // COCO always starts counting zeros
    uint64_t run_len   = 0;
    bool     first     = true;

    for (size_t i = 0; i < n; ++i) {
        const uint8_t v = col[i];
        if (v == run_value) {
            ++run_len;
        } else {
            if (!first) out += ',';
            sam3_append_uint(out, run_len);
            first     = false;
            run_value = v;
            run_len   = 1;
        }
    }
    if (!first) out += ',';
    sam3_append_uint(out, run_len);

    out += ']';
    return out;
}

int main(int argc, char** argv) {
    std::string model_path;
    int  n_threads = 4;
    bool use_gpu   = true;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--n-threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "ERROR: unknown argument '%s'\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n");
        usage(argv[0]);
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.n_threads  = n_threads;
    params.use_gpu    = use_gpu;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "ERROR: failed to create inference state\n");
        return 1;
    }

    // Announce readiness so the caller does not have to guess when the (slow)
    // model load has finished.
    printf("{\"ok\":true,\"event\":\"ready\",\"model_type\":%d}\n",
           (int)sam3_get_model_type(*model));
    fflush(stdout);

    bool            have_image = false;
    int             img_w = 0, img_h = 0;
    std::string     loaded_path;
    sam3_pvs_params pending;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream is(line);
        std::string cmd;
        if (!(is >> cmd)) continue;

        if (cmd == "quit") {
            printf("{\"ok\":true,\"event\":\"bye\"}\n");
            fflush(stdout);
            break;
        }

        if (cmd == "load") {
            std::string path;
            std::getline(is >> std::ws, path);   // keep spaces in the path
            if (path.empty()) { sam3_reply_error("load: missing path"); continue; }

            // Encoding the same file again costs seconds and produces the same
            // features, and the annotator re-picks the current image often.
            if (have_image && path == loaded_path) {
                printf("{\"ok\":true,\"event\":\"loaded\",\"width\":%d,\"height\":%d,\"cached\":true}\n",
                       img_w, img_h);
                fflush(stdout);
                continue;
            }

            // Drop the old image BEFORE touching the model state. A failed load
            // otherwise leaves have_image/img_w/img_h describing the previous
            // image while the library has already been repointed, and the next
            // `segment` answers ok:true with the wrong image's mask.
            have_image  = false;
            img_w       = 0;
            img_h       = 0;
            loaded_path.clear();
            pending     = sam3_pvs_params();

            sam3_image image = sam3_load_image(path);
            if (image.data.empty()) {
                sam3_reply_error("load: cannot read image '" + path + "'");
                continue;
            }
            if (!sam3_encode_image(*state, *model, image)) {
                sam3_reply_error("load: image encoding failed");
                continue;
            }

            have_image  = true;
            img_w       = image.width;
            img_h       = image.height;
            loaded_path = path;

            printf("{\"ok\":true,\"event\":\"loaded\",\"width\":%d,\"height\":%d,\"cached\":false}\n",
                   img_w, img_h);
            fflush(stdout);
            continue;
        }

        if (cmd == "point") {
            float x, y; int label;
            if (!(is >> x >> y >> label)) { sam3_reply_error("point: expected <x> <y> <label>"); continue; }
            if (label) pending.pos_points.push_back({x, y});
            else       pending.neg_points.push_back({x, y});
            printf("{\"ok\":true,\"event\":\"queued\",\"pos\":%zu,\"neg\":%zu}\n",
                   pending.pos_points.size(), pending.neg_points.size());
            fflush(stdout);
            continue;
        }

        if (cmd == "box") {
            float x0, y0, x1, y1;
            if (!(is >> x0 >> y0 >> x1 >> y1)) { sam3_reply_error("box: expected <x0> <y0> <x1> <y1>"); continue; }
            pending.box     = {x0, y0, x1, y1};
            pending.use_box = true;
            printf("{\"ok\":true,\"event\":\"queued\",\"box\":true}\n");
            fflush(stdout);
            continue;
        }

        if (cmd == "multimask") {
            int v;
            if (!(is >> v)) { sam3_reply_error("multimask: expected 0 or 1"); continue; }
            pending.multimask = (v != 0);
            printf("{\"ok\":true,\"event\":\"queued\",\"multimask\":%s}\n",
                   pending.multimask ? "true" : "false");
            fflush(stdout);
            continue;
        }

        if (cmd == "reset") {
            pending = sam3_pvs_params();
            printf("{\"ok\":true,\"event\":\"reset\"}\n");
            fflush(stdout);
            continue;
        }

        if (cmd == "segment") {
            if (!have_image) { sam3_reply_error("segment: no image loaded"); continue; }
            if (pending.pos_points.empty() && !pending.use_box) {
                // sam3_segment_pvs has nothing to anchor on without a positive
                // point or a box; answering "no detections" would blame the
                // image for what is really a malformed prompt.
                sam3_reply_error(pending.neg_points.empty()
                                     ? "segment: no prompt queued"
                                     : "segment: needs a positive point or a box, not only negative points");
                continue;
            }

            sam3_result result = sam3_segment_pvs(*state, *model, pending);
            pending = sam3_pvs_params();

            if (result.detections.empty()) {
                // Keep the reply shape identical to the non-empty case so a
                // consumer never has to special-case the field set.
                printf("{\"ok\":true,\"event\":\"segmented\",\"width\":%d,\"height\":%d,\"detections\":[]}\n",
                       img_w, img_h);
                fflush(stdout);
                continue;
            }

            printf("{\"ok\":true,\"event\":\"segmented\",\"width\":%d,\"height\":%d,\"detections\":[",
                   img_w, img_h);
            for (size_t i = 0; i < result.detections.size(); ++i) {
                const sam3_detection & d = result.detections[i];
                if (i) printf(",");
                printf("{\"score\":%.6f,\"iou\":%.6f,"
                       "\"box\":[%.3f,%.3f,%.3f,%.3f],"
                       "\"mask_size\":[%d,%d],\"counts\":%s}",
                       d.score, d.iou_score,
                       d.box.x0, d.box.y0, d.box.x1, d.box.y1,
                       d.mask.width, d.mask.height,
                       sam3_mask_to_rle(d.mask).c_str());
            }
            printf("]}\n");
            fflush(stdout);
            continue;
        }

        sam3_reply_error("unknown command '" + cmd + "'");
    }

    return 0;
}
