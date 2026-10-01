#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sam31 { namespace native {
// RGB, interleaved HWC, [0,255]. An image is resized and normalized in C++.
struct Image { size_t width=0,height=0; std::vector<uint8_t> rgb; };
struct Point { int64_t object_id; float x,y; int label=1; }; // normalized [0,1]
struct ObjectMask {
    int64_t object_id=0;
    float score=0,quality=0;
    std::vector<float> logits; // row-major, input image size; threshold at zero
};
struct Result { size_t frame_index=0,width=0,height=0; std::vector<ObjectMask> objects; };
struct Options {
    bool allow_test_fixture=false; // required explicitly for the untrained smoke model
    size_t total_frames=16; // pointer temporal normalization; CLI supplies actual length
    size_t weight_cache_mb=128;
    int threads=1;
    bool profile=false; // emit operator elapsed times in session stats
};
// Reusable pure-C++ model. Storage: GGUF I8/F16/BF16/F32. Compute: float32 CPU.
// Every required tensor is checked before creating a session. No fallback weights.
class Model {
public:
    explicit Model(const std::string&gguf_path, const Options&options={});
    ~Model();
    Model(Model&&) noexcept; Model&operator=(Model&&) noexcept;
    Model(const Model&)=delete;Model&operator=(const Model&)=delete;
    std::string info_json() const;
    struct Impl;
private:
    std::shared_ptr<Impl> p_;
    friend class Session;
};
// One bucket, 1..16 stable IDs, first-frame point prompts; independent state per session.
// begin() and step() commit state only after a complete successful forward.
class Session {
public:
    explicit Session(Model&model); ~Session();
    Session(Session&&) noexcept;Session&operator=(Session&&) noexcept;
    Session(const Session&)=delete;Session&operator=(const Session&)=delete;
    Result begin(const Image&image,const std::vector<Point>&points);
    Result step(const Image&image);
    void reset();
    // Test-only intervention. Does not disable the attention calculation.
    void zero_memory_for_test();
    std::string stats_json() const;
private:
    struct Impl;std::unique_ptr<Impl> p_;
};
Image read_image(const std::string&path);
void write_image_png(const std::string&path,const Image&image);
void write_mask_png(const std::string&path,const ObjectMask&mask,size_t width,size_t height);
void write_logits(const std::string&path,const Result&result);
// Shape contracts for fixture generation and checkpoint audit, no weights generated here.
std::string manifest_json(bool smoke);
}}
