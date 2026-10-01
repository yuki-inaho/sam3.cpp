#include "sam31_gguf.h"
#include "gguf.h"
#include "ggml.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace sam31 {
namespace {
std::string string_value(const gguf_context * ctx, const char * key) {
    const int64_t index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_STRING)
        throw std::runtime_error(std::string("Missing or invalid string metadata: ") + key);
    return gguf_get_val_str(ctx, index);
}

bool boolean_value(const gguf_context * ctx, const char * key) {
    const int64_t index = gguf_find_key(ctx, key);
    if (index < 0 || gguf_get_kv_type(ctx, index) != GGUF_TYPE_BOOL)
        throw std::runtime_error(std::string("Missing or invalid boolean metadata: ") + key);
    return gguf_get_val_bool(ctx, index);
}
}

struct gguf_file::impl {
    std::string path;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> context{nullptr, &gguf_free};
    std::vector<tensor_descriptor> descriptors;
    uint64_t offset = 0;
    bool fixture = false;
};

gguf_file::gguf_file(const std::string & path, bool allow_test_fixture) : p_(new impl) {
    p_->path = path;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 24) throw std::runtime_error("Missing or truncated GGUF file");
    const uint64_t length = static_cast<uint64_t>(file.tellg());
    file.seekg(0);
    unsigned char header[24];
    file.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!file || std::memcmp(header, "GGUF", 4) != 0 || header[4] != 3 || header[5] || header[6] || header[7])
        throw std::runtime_error("Expected little-endian GGUF version 3");
    auto read_u64 = [&](size_t at) {
        uint64_t result = 0;
        for (size_t i = 0; i < 8; ++i) result |= uint64_t(header[at + i]) << (8 * i);
        return result;
    };
    if (!read_u64(8) || read_u64(8) > 100000 || !read_u64(16) || read_u64(16) > 16384)
        throw std::runtime_error("Excessive or empty GGUF tensor/metadata counts");
    const gguf_init_params params{true, nullptr};
    p_->context.reset(gguf_init_from_file(path.c_str(), params));
    auto * ctx = p_->context.get();
    if (!ctx) throw std::runtime_error("ggml rejected the GGUF directory");
    if (string_value(ctx, "general.architecture") != "sam3_1_multiplex")
        throw std::runtime_error("Not a SAM 3.1 multiplex GGUF");
    const int64_t schema = gguf_find_key(ctx, "sam31.storage_version");
    if (schema < 0 || gguf_get_kv_type(ctx, schema) != GGUF_TYPE_UINT32 || gguf_get_val_u32(ctx, schema) != 1)
        throw std::runtime_error("Unsupported SAM 3.1 storage version");
    if (gguf_get_alignment(ctx) != 32 || gguf_get_data_offset(ctx) > 64 * 1024 * 1024)
        throw std::runtime_error("Invalid alignment or excessive GGUF header");
    (void) string_value(ctx, "sam31.tensor_map");
    const auto hash = string_value(ctx, "sam31.source.sha256");
    if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("Invalid source SHA-256 metadata");
    p_->fixture = boolean_value(ctx, "sam31.test_fixture");
    if (p_->fixture && !allow_test_fixture)
        throw std::runtime_error("Synthetic test weights are not a SAM 3.1 checkpoint");
    p_->offset = gguf_get_data_offset(ctx);
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const uint64_t relative = gguf_get_tensor_offset(ctx, i);
        const uint64_t bytes = gguf_get_tensor_size(ctx, i);
        if (relative % 32 || p_->offset > length || relative > length - p_->offset ||
            bytes > length - p_->offset - relative || !bytes)
            throw std::runtime_error("Tensor outside the GGUF payload");
        p_->descriptors.push_back({gguf_get_tensor_name(ctx, i), static_cast<int>(gguf_get_tensor_type(ctx, i)),
                                   p_->offset + relative, bytes});
    }
    auto ordered = p_->descriptors;
    std::sort(ordered.begin(), ordered.end(), [](const tensor_descriptor & a, const tensor_descriptor & b) {
        return a.offset < b.offset;
    });
    for (size_t i = 1; i < ordered.size(); ++i)
        if (ordered[i].offset < ordered[i - 1].offset + ordered[i - 1].size)
            throw std::runtime_error("Overlapping GGUF tensors");
}

gguf_file::~gguf_file() = default;
gguf_file::gguf_file(gguf_file &&) noexcept = default;
gguf_file & gguf_file::operator=(gguf_file &&) noexcept = default;
const std::vector<tensor_descriptor> & gguf_file::tensors() const { return p_->descriptors; }
std::string gguf_file::tensor_map_json() const { return string_value(p_->context.get(), "sam31.tensor_map"); }
std::string gguf_file::source_sha256() const { return string_value(p_->context.get(), "sam31.source.sha256"); }
std::string gguf_file::source_metadata_json() const { return string_value(p_->context.get(), "sam31.source.metadata"); }
bool gguf_file::is_test_fixture() const { return p_->fixture; }
uint64_t gguf_file::data_offset() const { return p_->offset; }
std::vector<uint8_t> gguf_file::read_tensor(const std::string & storage_name) const {
    const auto it = std::find_if(p_->descriptors.begin(), p_->descriptors.end(), [&](const tensor_descriptor & item) {
        return item.name == storage_name;
    });
    if (it == p_->descriptors.end()) throw std::out_of_range("Unknown GGUF tensor: " + storage_name);
    if (it->size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::length_error("Tensor is too large for this platform");
    std::vector<uint8_t> data(static_cast<size_t>(it->size));
    std::ifstream file(p_->path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(it->offset));
    file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(it->size));
    if (!file) throw std::runtime_error("GGUF payload was truncated after opening");
    return data;
}
}
