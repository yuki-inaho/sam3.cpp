#pragma once
// Efficient-SAM2 v1 storage adapter. Native inference restores FP32 weights;
// this API does not claim INT8 arithmetic or efficient video execution.
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct sam2_gguf_tensor {
    std::string name, dtype;
    std::vector<size_t> shape;
    uint64_t offset = 0, bytes = 0;
};
struct sam2_gguf_file {
    std::string path, variant, sam2_version, config_json;
    size_t image_size = 0;
    bool test_fixture = false;
    std::map<std::string, sam2_gguf_tensor> tensors;
};
bool sam2_gguf_open(const std::string &path, sam2_gguf_file &file, std::string &error);
bool sam2_gguf_restore(const sam2_gguf_file &file, const std::string &name,
                      std::vector<float> &values, std::string &error);
// Storage alone is architecture-agnostic; executing the native sparse graph
// additionally requires its exact, explicitly declared model/policy profile.
bool sam2_gguf_validate_native_profile(const sam2_gguf_file &file, std::string &error);
std::string sam2_gguf_runtime_name(std::string source);
bool sam2_gguf_is_auxiliary(const std::string &name);
