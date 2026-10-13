#pragma once
// Exact, session-owned Efficient-SAM2 prefix features. C++14-compatible API.
// The caller must provide verified content SHA256 values, never path/mtime IDs.
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace efficient_sam2 { namespace prefix_cache {
constexpr uint64_t payload_bytes = 49ULL * 1024 * 1024;
struct Key {
    std::string model_sha256; // Actual loaded GGUF file content.
    std::string input_sha256; // Canonical original RGB bytes AND image geometry.
    std::string policy_sha256; // Full model/preprocessing/quantization configuration.
    std::string preprocessing; // Explicit image/video resize/normalization representation.
    std::string precision; // e.g. restored-f32 or exact Q8 selection manifest ID.
    std::string backend; // CPU ISA/backend variant, not merely "cpu".
    std::string build; // Inference implementation/ggml build identity.
    int threads = 0;
    int input_width = 0, input_height = 0;
};
struct Tensor {
    std::array<int64_t,4> shape{{1,1,1,1}};
    std::vector<float> values;
};
struct Entry {
    // Channel-fast FP32: x_after_block5, stage0, stage1. No live ggml pointers.
    std::array<Tensor,3> tensors;
};
struct Options {
    uint64_t ram_budget_bytes = payload_bytes; // At most one resident entry.
    std::string disk_directory; // Empty disables disk. Dedicated directory only.
    uint64_t disk_budget_bytes = 0; // Explicit opt-in, includes file headers.
};
struct Stats {
    uint64_t ram_hits=0, disk_hits=0, misses=0, invalid_entries=0;
    uint64_t writes=0, evictions=0, resident_bytes=0, disk_bytes=0;
};
enum class Hit { miss, ram, disk, rejected };
struct Cache;
// Separate sessions/processes may read immutable files. Only one writer/clearer
// may own a disk directory at a time; this helper does not implement a database.
std::shared_ptr<Cache> create(const Options&, std::string& error);
std::shared_ptr<const Entry> lookup(Cache&, const Key&, Hit&, std::string& error);
bool store(Cache&, const Key&, const Entry&, std::string& error);
bool clear(Cache&, bool include_disk, std::string& error);
Stats stats(Cache&);
// Import/export are atomic, exact-format operations, independent of residency.
// File imports reject mismatched keys, old versions, corrupt/partial data.
bool write_file(const std::string& path, const Key&, const Entry&, std::string& error);
std::shared_ptr<const Entry> read_file(const std::string& path, const Key&, std::string& error);
std::string artifact_path(Cache&, const Key&, std::string& error);
std::string canonical_key(const Key&); // Throws invalid_argument for invalid keys.
} }
