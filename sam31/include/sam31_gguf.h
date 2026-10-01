#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sam31 {
struct tensor_descriptor {
    std::string name;
    int type;
    uint64_t offset;
    uint64_t size;
};

/* Native typed-GGUF reader. Its constructor validates bounds and schema before
 * any weight payload is read. Does not claim to implement the SAM 3.1 graph. */
class gguf_file {
public:
    explicit gguf_file(const std::string & path, bool allow_test_fixture = false);
    ~gguf_file();
    gguf_file(gguf_file &&) noexcept;
    gguf_file & operator=(gguf_file &&) noexcept;
    gguf_file(const gguf_file &) = delete;
    gguf_file & operator=(const gguf_file &) = delete;
    const std::vector<tensor_descriptor> & tensors() const;
    std::string tensor_map_json() const;
    std::string source_sha256() const;
    std::string source_metadata_json() const;
    bool is_test_fixture() const;
    uint64_t data_offset() const;
    std::vector<uint8_t> read_tensor(const std::string & storage_name) const;
private:
    struct impl;
    std::unique_ptr<impl> p_;
};
}
