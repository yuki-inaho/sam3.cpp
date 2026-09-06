/**
 * sam3_seg — headless point/box prompt segmentation CLI.
 *
 * Loads a model and a still image, encodes once, then segments with a
 * point (or box) prompt and writes the resulting mask(s) as PNG.  No GUI
 * and no video dependencies — works in headless containers.
 *
 * Usage:
 *   sam3_seg --model <path> --image <path> [options]
 *
 * Options:
 *   --model <path>        .ggml model file (required)
 *   --image <path>        input image (jpg/png, required)
 *   --x <f> --y <f>       positive point prompt in image pixels (default 315, 250)
 *   --box x0 y0 x1 y1     box prompt instead of the point
 *   --out <path>          output mask PNG (default: mask.png; with --multimask
 *                         or multiple detections, indexed _0/_1 suffixes)
 *   --n-threads <n>       CPU threads (default: 4)
 *   --cpu                 force CPU backend
 *   --multimask           request 3 mask candidates from the decoder
 */

#include "sam3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path> --image <path> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>        .ggml model file (required)\n"
        "  --image <path>        input image (required)\n"
        "  --x <f> --y <f>       positive point prompt (default: 315, 250)\n"
        "  --box x0 y0 x1 y1     box prompt instead of the point\n"
        "  --out <path>          output mask PNG (default: mask.png)\n"
        "  --n-threads <n>       CPU threads (default: 4)\n"
        "  --cpu                 force CPU backend\n"
        "  --multimask           request 3 mask candidates\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_path;
    std::string image_path;
    std::string out_path = "mask.png";
    float px = 315.0f, py = 250.0f;
    float box[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool  use_box = false;
    int   n_threads = 4;
    bool  use_gpu = true;
    bool  multimask = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--x") == 0 && i + 1 < argc) {
            px = atof(argv[++i]);
        } else if (strcmp(argv[i], "--y") == 0 && i + 1 < argc) {
            py = atof(argv[++i]);
        } else if (strcmp(argv[i], "--box") == 0 && i + 4 < argc) {
            for (int k = 0; k < 4; ++k) box[k] = atof(argv[++i]);
            use_box = true;
        } else if (strcmp(argv[i], "--n-threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu") == 0) {
            use_gpu = false;
        } else if (strcmp(argv[i], "--multimask") == 0) {
            multimask = true;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || image_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu    = use_gpu;
    params.n_threads  = n_threads;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }

    auto image = sam3_load_image(image_path);
    if (image.data.empty()) {
        fprintf(stderr, "ERROR: failed to load image '%s'\n", image_path.c_str());
        return 1;
    }
    fprintf(stderr, "Model:      %s (type %d)\n", model_path.c_str(),
            (int)sam3_get_model_type(*model));
    fprintf(stderr, "Image:      %dx%d (%d ch)\n", image.width, image.height, image.channels);
    fprintf(stderr, "Backend:    %s\n", use_gpu ? "GPU (falls back to CPU if unavailable)" : "CPU");

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "ERROR: failed to create inference state\n");
        return 1;
    }

    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "ERROR: image encoding failed\n");
        return 1;
    }

    sam3_pvs_params pvs;
    pvs.multimask = multimask;
    if (use_box) {
        pvs.box    = {box[0], box[1], box[2], box[3]};
        pvs.use_box = true;
        fprintf(stderr, "Prompt:     box (%.1f, %.1f, %.1f, %.1f)\n", box[0], box[1], box[2], box[3]);
    } else {
        pvs.pos_points.push_back({px, py});
        fprintf(stderr, "Prompt:     point (%.1f, %.1f)\n", px, py);
    }

    sam3_result result = sam3_segment_pvs(*state, *model, pvs);
    if (result.detections.empty()) {
        fprintf(stderr, "No detections.\n");
        return 1;
    }

    fprintf(stderr, "Detections: %zu\n", result.detections.size());

    // Pick the highest-scoring detection.
    size_t best = 0;
    for (size_t i = 1; i < result.detections.size(); ++i) {
        if (result.detections[i].score > result.detections[best].score) best = i;
    }
    const sam3_detection& det = result.detections[best];
    fprintf(stderr, "Best:       score=%.3f iou=%.3f box=(%.1f, %.1f, %.1f, %.1f)\n",
            det.score, det.iou_score, det.box.x0, det.box.y0, det.box.x1, det.box.y1);

    std::vector<std::string> saved;
    for (size_t i = 0; i < result.detections.size(); ++i) {
        std::string path = out_path;
        if (result.detections.size() > 1) {
            size_t dot = path.rfind('.');
            char suffix[16];
            snprintf(suffix, sizeof(suffix), "_%zu", i);
            if (dot == std::string::npos) path += suffix;
            else path.insert(dot, suffix);
        }
        if (sam3_save_mask(result.detections[i].mask, path)) {
            fprintf(stderr, "Mask:       %s (%dx%d)\n", path.c_str(),
                    result.detections[i].mask.width, result.detections[i].mask.height);
            saved.push_back(path);
        } else {
            fprintf(stderr, "ERROR: failed to save mask '%s'\n", path.c_str());
        }
    }

    if (saved.empty()) return 1;

    // Machine-readable summary on stdout.
    for (const auto& p : saved) printf("%s\n", p.c_str());
    return 0;
}
