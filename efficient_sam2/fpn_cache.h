#pragma once

// Exact, session-owned Efficient-SAM2 FPN features. C++14, RAM only.
// A Cache belongs to one fixed encoder/model/configuration session. The caller
// supplies the complete canonical input identity, including image geometry and
// pixels; paths, timestamps, or a truncated identity are not sufficient.
#include "prefix_cache/cache.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace efficient_sam2 { namespace fpn_cache {

constexpr uint64_t payload_bytes = 84ULL * 1024 * 1024;

struct Entry {
    // Channel-fast FP32: neck0, neck1, neck2. No ggml or decoder state.
    std::array<prefix_cache::Tensor, 3> tensors;
};

struct Stats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    // Capacity-driven removals only; not image changes or key replacement.
    uint64_t evictions = 0;
    // Logical FP32 payload only, excluding allocator/vector spare capacity.
    uint64_t resident_bytes = 0;
};

// Not internally synchronized: serialize use within the owning session.
// Immutable lookup handles can outlive eviction, replacement, or the Cache.
// Such external handles extend allocation lifetimes beyond resident_bytes.
class Cache {
public:
    explicit Cache(std::size_t capacity = 4) : capacity_(capacity) {
        static_assert(sizeof(float) == 4 &&
                      std::numeric_limits<float>::is_iec559 &&
                      std::numeric_limits<float>::digits == 24,
                      "FPN cache requires IEEE-754 FP32");
        if (capacity == 0 || capacity > 4) {
            throw std::invalid_argument("FPN cache capacity must be between 1 and 4");
        }
    }

    std::shared_ptr<const Entry> lookup(const std::string& identity,
                                        const std::vector<int>& ids) {
        const std::vector<int> key = select_image(identity, ids);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->ids == key) {
                entries_.splice(entries_.begin(), entries_, it);
                ++stats_.hits;
                return entries_.front().entry;
            }
        }
        ++stats_.misses;
        return {};
    }

    // Invalid identity/IDs throw before changing the cache. A valid identity
    // selects the current image even if tensor validation subsequently fails.
    // Invalid tensors return false, leave the entry unconsumed, and set error.
    // Success moves the payload into immutable storage without a tensor copy.
    bool store(const std::string& identity, const std::vector<int>& ids,
               Entry&& entry, std::string& error) {
        error.clear();
        std::vector<int> key = select_image(identity, ids);
        if (!validate(entry, error)) {
            return false;
        }
        auto value = std::make_shared<const Entry>(std::move(entry));
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->ids == key) {
                it->entry = std::move(value);
                entries_.splice(entries_.begin(), entries_, it);
                return true;
            }
        }
        // Prepare the list node before removing an old entry. Splicing adds no
        // allocation and the resident list never exceeds the configured bound.
        std::list<Node> incoming;
        incoming.push_front(Node{std::move(key), std::move(value)});
        if (entries_.size() == capacity_) {
            entries_.pop_back();
            ++stats_.evictions;
        }
        entries_.splice(entries_.begin(), incoming, incoming.begin());
        return true;
    }

    Stats stats() const {
        Stats result = stats_;
        result.resident_bytes = static_cast<uint64_t>(entries_.size()) * payload_bytes;
        return result;
    }

private:
    struct Node {
        std::vector<int> ids;
        std::shared_ptr<const Entry> entry;
    };

    std::vector<int> select_image(const std::string& identity,
                                  const std::vector<int>& ids) {
        if (identity.empty()) {
            throw std::invalid_argument("FPN cache requires a nonempty canonical image identity");
        }
        std::vector<int> key = ids;
        if (key.empty()) {
            key.reserve(25);
            for (int id = 0; id < 25; ++id) {
                key.push_back(id);
            }
        } else {
            std::array<bool, 25> seen{};
            for (int id : key) {
                if (id < 0 || id >= 25 || seen[static_cast<std::size_t>(id)]) {
                    throw std::invalid_argument("FPN cache window IDs must be unique integers in [0, 24]");
                }
                seen[static_cast<std::size_t>(id)] = true;
            }
        }
        // Compare complete strings, including any embedded NUL bytes. Never
        // sort the window list: packed batch ordering is part of exact reuse.
        if (identity_ != identity) {
            identity_ = identity;
            entries_.clear();
        }
        return key;
    }

    static bool validate(const Entry& entry, std::string& error) {
        const std::array<std::array<int64_t, 4>, 3> shapes{{
            {{256, 256, 256, 1}},
            {{256, 128, 128, 1}},
            {{256, 64, 64, 1}}
        }};
        for (std::size_t i = 0; i < entry.tensors.size(); ++i) {
            const auto& tensor = entry.tensors[i];
            if (tensor.shape != shapes[i]) {
                error = "FPN cache neck" + std::to_string(i) + " has an invalid shape";
                return false;
            }
            const std::size_t count = static_cast<std::size_t>(shapes[i][0] *
                                                               shapes[i][1] *
                                                               shapes[i][2]);
            if (tensor.values.size() != count) {
                error = "FPN cache neck" + std::to_string(i) + " has an invalid FP32 element count";
                return false;
            }
            for (float value : tensor.values) {
                if (!std::isfinite(value)) {
                    error = "FPN cache neck" + std::to_string(i) + " contains a nonfinite FP32 value";
                    return false;
                }
            }
        }
        return true;
    }

    std::size_t capacity_;
    std::string identity_;
    // Most recently used at the front; at most four nodes for one image.
    std::list<Node> entries_;
    Stats stats_;
};

} } // namespace efficient_sam2::fpn_cache
