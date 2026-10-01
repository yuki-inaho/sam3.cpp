// Headless trained EV-M image/ordered-frame annotation with native ggml CPU.
#include "sam3.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

int main(int argc, char ** argv) {
    std::string model_path, text, output = "output/efficient", preprocessed;
    std::vector<std::string> images;
    int threads = 16;
    float threshold = 0.5f;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            std::fprintf(stderr, "Usage: efficientsam3 --model model.gguf --text dog --image image.png [--image next.png] --output existing-directory [--threads 16] [--threshold 0.5] [--preprocessed input.f32]\n");
            return 0;
        }
        if (i + 1 >= argc) return 2;
        const std::string value = argv[++i];
        if (argument == "--model") model_path = value;
        else if (argument == "--text") text = value;
        else if (argument == "--image") images.push_back(value);
        else if (argument == "--output") output = value;
        else if (argument == "--threads") {
            char * end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (value.empty() || !end || *end || parsed < 1 || parsed > 1024) return 2;
            threads = (int)parsed;
        } else if (argument == "--threshold") {
            char * end = nullptr;
            threshold = std::strtof(value.c_str(), &end);
            if (value.empty() || !end || *end) return 2;
        }
        else if (argument == "--preprocessed") preprocessed = value;
        else return 2;
    }
    if (model_path.empty() || text.empty() || images.empty() || threads < 1 ||
        !std::isfinite(threshold) || threshold < 0 || threshold > 1 ||
        (!preprocessed.empty() && images.size() != 1)) return 2;
    sam3_params parameters;
    parameters.model_path = model_path; parameters.use_gpu = false; parameters.n_threads = threads;
    auto model = sam3_load_model(parameters);
    if (!model) return 1;
    if (sam3_get_model_type(*model) != SAM3_MODEL_EFFICIENTSAM3) {
        sam3_free_model(*model); return 1;
    }
    auto state = sam3_create_state(*model, parameters);
    if (!state) { sam3_free_model(*model); return 1; }
#ifdef _WIN32
    _mkdir(output.c_str());
#else
    mkdir(output.c_str(), 0755);
#endif
    std::ofstream report(output + "/annotations.json");
    if (!report) { state.reset(); sam3_free_model(*model); return 1; }
    report << std::setprecision(9);
    report << "{\"backend\":\"ggml-cpu\",\"variant\":\"ev-m-stage3\",\"sequence_mode\":\"independent_detection\",\"frames\":[";
    int exit_status = 0;
    for (size_t frame = 0; frame < images.size(); ++frame) {
        auto image = sam3_load_image(images[frame]);
        if (image.data.empty()) { exit_status = 1; break; }
        auto begin = std::chrono::steady_clock::now();
        bool encoded;
        if (preprocessed.empty()) encoded = sam3_encode_image(*state, *model, image);
        else {
            std::vector<float> input(3 * 1008 * 1008);
            std::ifstream file(preprocessed, std::ios::binary);
            file.read(reinterpret_cast<char *>(input.data()), input.size() * sizeof(float));
            if (!file || file.peek() != std::ifstream::traits_type::eof()) { exit_status = 1; break; }
            encoded = sam3_encode_efficient_from_preprocessed(*state, *model, input.data(), image.width, image.height);
        }
        if (!encoded) { exit_status = 1; break; }
        if (std::getenv("SAM3_EFFICIENT_TRACE_DIR")) {
            const std::string directory = std::getenv("SAM3_EFFICIENT_TRACE_DIR");
            for (int level = 0; level < 3; ++level)
                if (!sam3_dump_state_tensor(*state, "neck_det_" + std::to_string(level), directory + "/fpn" + std::to_string(level))) { exit_status = 1; break; }
        }
        sam3_pcs_params prompt;
        prompt.text_prompt = text; prompt.score_threshold = threshold; prompt.nms_threshold = 1.0f;
        auto result = sam3_segment_pcs(*state, *model, prompt);
        auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (frame) report << ',';
        report << "{\"frame_index\":" << frame << ",\"width\":" << image.width << ",\"height\":" << image.height << ",\"elapsed_ms\":" << elapsed << ",\"detections\":[";
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto & detection = result.detections[i];
            const std::string name = "frame_" + std::to_string(frame) + "_mask_" + std::to_string(i) + ".png";
            if (!sam3_save_mask(detection.mask, output + "/" + name)) { exit_status = 1; break; }
            if (i) report << ',';
            report << "{\"score\":" << detection.score << ",\"mask\":\"" << name << "\",\"box_xyxy\":[" << detection.box.x0 << ',' << detection.box.y0 << ',' << detection.box.x1 << ',' << detection.box.y1 << "]}";
        }
        report << "]}";
        std::fprintf(stderr, "EV-M frame %zu: %zu detections, %.1f ms\n", frame, result.detections.size(), elapsed);
    }
    report << "]}\n";
    report.close();
    if (!report) exit_status = 1;
    state.reset();
    sam3_free_model(*model);
    return exit_status;
}
