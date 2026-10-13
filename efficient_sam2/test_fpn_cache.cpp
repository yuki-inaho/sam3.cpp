#include "fpn_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace efficient_sam2::fpn_cache;

namespace {

void check(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

template<class Fn>
void invalid_argument(Fn fn, const char* reason) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(reason);
}

Entry fixture(float marker = 1.25f) {
    Entry entry;
    const int sizes[] = {256, 128, 64};
    for (std::size_t i = 0; i < entry.tensors.size(); ++i) {
        auto& tensor = entry.tensors[i];
        tensor.shape = {{256, sizes[i], sizes[i], 1}};
        tensor.values.assign(static_cast<std::size_t>(256 * sizes[i] * sizes[i]),
                             marker + static_cast<float>(i));
        tensor.values[1] = -0.0f;
        tensor.values[2] = 0.0f;
        tensor.values[3] = std::nextafter(1.0f, 2.0f);
        tensor.values[4] = std::numeric_limits<float>::denorm_min();
        tensor.values[5] = std::numeric_limits<float>::max();
        tensor.values[6] = std::numeric_limits<float>::lowest();
    }
    return entry;
}

void put(Cache& cache, const std::string& image, const std::vector<int>& ids,
         float marker = 1.25f) {
    std::string error = "stale error";
    check(cache.store(image, ids, fixture(marker), error), "Valid entry store failed");
    check(error.empty(), "Successful store did not clear error");
}

void test_exact_and_normalized() {
    Cache cache;
    check(!cache.lookup("image-A", {}), "Cold lookup unexpectedly hit");
    Entry entry = fixture();
    std::array<const float*, 3> buffers{};
    uint64_t bytes = 0;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        buffers[i] = entry.tensors[i].values.data();
        bytes += entry.tensors[i].values.size() * sizeof(float);
    }
    check(bytes == payload_bytes, "Payload is not exactly 84 MiB");
    std::string error;
    check(cache.store("image-A", {}, std::move(entry), error), "Exact entry store failed");
    std::vector<int> all;
    for (int id = 0; id < 25; ++id) {
        all.push_back(id);
    }
    auto found = cache.lookup("image-A", all);
    check(bool(found), "Empty window selection did not normalize to all 25 windows");
    const Entry expected = fixture();
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto& tensor = found->tensors[i];
        check(tensor.shape == expected.tensors[i].shape, "Tensor shape changed");
        check(tensor.values.data() == buffers[i], "Store copied rather than moved a tensor buffer");
        check(std::memcmp(tensor.values.data(), expected.tensors[i].values.data(),
                          tensor.values.size() * sizeof(float)) == 0,
              "FP32 tensor bits changed");
        check(std::signbit(tensor.values[1]) && !std::signbit(tensor.values[2]),
              "Signed zero changed");
    }
    check(cache.lookup("image-A", {}) == found, "Repeated lookup did not reuse immutable entry");
    std::reverse(all.begin(), all.end());
    check(!cache.lookup("image-A", all), "Reversed all-window order incorrectly hit");
    const Stats stats = cache.stats();
    check(stats.hits == 2 && stats.misses == 2 && stats.evictions == 0 &&
          stats.resident_bytes == payload_bytes, "Exact/normalization counters are incorrect");
}

void test_order_and_identity() {
    Cache cache;
    put(cache, "canonical-image-A", {1, 7, 3}, 2.0f);
    check(!cache.lookup("canonical-image-A", {3, 7, 1}), "Window IDs were incorrectly sorted");
    put(cache, "canonical-image-A", {3, 7, 1}, 3.0f);
    auto first = cache.lookup("canonical-image-A", {1, 7, 3});
    auto reverse = cache.lookup("canonical-image-A", {3, 7, 1});
    check(first && reverse && first != reverse, "Different window orders shared an entry");
    check(first->tensors[0].values[0] == 2.0f && reverse->tensors[0].values[0] == 3.0f,
          "Order-sensitive values were mixed");
    check(!cache.lookup("canonical-image-B", {1, 7, 3}), "Different image identity hit");
    check(cache.stats().resident_bytes == 0, "Image change did not clear resident entries");
    check(first->tensors[0].values[0] == 2.0f, "Image change invalidated an external handle");
    check(!cache.lookup("canonical-image-A", {1, 7, 3}), "Returning to an old image revived stale entries");
    check(cache.stats().evictions == 0, "Image invalidation counted as capacity eviction");
    put(cache, "canonical-image-A", {2});
    put(cache, "canonical-image-B", {2}, 4.0f);
    check(cache.stats().resident_bytes == payload_bytes, "Store did not invalidate the old image");
    check(cache.lookup("canonical-image-B", {2})->tensors[0].values[0] == 4.0f,
          "New-image store is missing");
    const std::string binary_a("image\0A", 7), binary_b("image\0B", 7);
    put(cache, binary_a, {2});
    check(!cache.lookup(binary_b, {2}), "Image identity comparison stopped at an embedded NUL");
}

void test_invalid_keys() {
    invalid_argument([] { Cache cache(0); }, "Zero capacity accepted");
    invalid_argument([] { Cache cache(5); }, "Capacity greater than four accepted");
    Cache cache;
    put(cache, "image-A", {0});
    invalid_argument([&] { cache.lookup("", {0}); }, "Empty image identity accepted");
    for (const auto& ids : std::vector<std::vector<int>>{{-1}, {25}, {0, 0}, {24, 1, 24}}) {
        invalid_argument([&] { cache.lookup("image-B", ids); }, "Malformed lookup IDs accepted");
        Entry entry;
        std::string error;
        invalid_argument([&] { cache.store("image-B", ids, std::move(entry), error); },
                         "Malformed store IDs accepted");
    }
    Entry empty;
    std::string error;
    invalid_argument([&] { cache.store("", {0}, std::move(empty), error); },
                     "Empty store identity accepted");
    check(cache.stats().resident_bytes == payload_bytes && cache.stats().misses == 0,
          "Invalid keys changed cache state or counters");
    check(bool(cache.lookup("image-A", {0})), "Invalid key discarded a valid resident entry");
}

void test_invalid_entries() {
    Cache cache(1);
    put(cache, "image-A", {0});
    Entry invalid = fixture(9.0f);
    std::string error;
    const auto reject = [&](const char* reason) {
        error = "stale error";
        check(!cache.store("image-A", {1}, std::move(invalid), error), reason);
        check(!error.empty() && error != "stale error", "Invalid entry lacks a useful diagnostic");
        check(cache.stats().resident_bytes == payload_bytes && cache.stats().evictions == 0,
              "Invalid entry displaced valid residency");
        check(bool(cache.lookup("image-A", {0})), "Invalid entry removed a valid cache key");
        check(invalid.tensors[0].values.size() == 256ULL * 256 * 256,
              "Rejected entry was consumed");
    };
    for (std::size_t i = 0; i < invalid.tensors.size(); ++i) {
        auto& tensor = invalid.tensors[i];
        ++tensor.shape[3];
        reject("Invalid shape accepted");
        --tensor.shape[3];
        const float tail = tensor.values.back();
        tensor.values.pop_back();
        // For neck0 this temporarily changes its size, so test directly.
        check(!cache.store("image-A", {1}, std::move(invalid), error) && !error.empty(),
              "Truncated tensor accepted");
        tensor.values.push_back(tail);
        tensor.values.push_back(tail);
        check(!cache.store("image-A", {1}, std::move(invalid), error) && !error.empty(),
              "Oversized tensor accepted");
        tensor.values.pop_back();
        for (float value : {std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
            tensor.values.back() = value;
            reject("Nonfinite tensor value accepted");
        }
        tensor.values.back() = tail;
    }
    check(!cache.lookup("image-A", {1}), "Rejected entry became visible");
    check(cache.store("image-A", {1}, std::move(invalid), error) && error.empty(),
          "Repaired entry could not be stored");
    check(cache.stats().evictions == 1, "Successful repaired store did not perform bounded eviction");
    Entry empty;
    check(!cache.store("image-B", {0}, std::move(empty), error), "Empty tensors accepted");
    check(cache.stats().resident_bytes == 0, "A valid new image key did not invalidate old residency");
}

void test_lru_and_replacement() {
    Cache cache;
    for (int id = 0; id < 4; ++id) {
        put(cache, "image-A", {id}, static_cast<float>(id));
        check(cache.stats().resident_bytes == static_cast<uint64_t>(id + 1) * payload_bytes,
              "Resident byte count is incorrect");
    }
    check(cache.stats().resident_bytes == 336ULL * 1024 * 1024, "Four entries do not occupy 336 MiB");
    auto pinned = cache.lookup("image-A", {0}); // Promote oldest; window 1 is now LRU.
    put(cache, "image-A", {4}, 4.0f);
    check(cache.stats().resident_bytes == 4 * payload_bytes && cache.stats().evictions == 1,
          "Default four-entry budget was exceeded");
    check(!cache.lookup("image-A", {1}), "Least-recently-used entry was not evicted");
    check(bool(cache.lookup("image-A", {0})), "Lookup did not promote the old entry");
    put(cache, "image-A", {2}, 12.0f); // Replacement promotes window 2; window 3 is LRU.
    check(cache.stats().resident_bytes == 4 * payload_bytes && cache.stats().evictions == 1,
          "Same-key replacement changed residency or eviction count");
    check(cache.lookup("image-A", {2})->tensors[0].values[0] == 12.0f,
          "Replacement data was not visible");
    put(cache, "image-A", {5}, 5.0f);
    check(!cache.lookup("image-A", {3}), "Replacement did not promote the key");
    check(cache.stats().resident_bytes == 4 * payload_bytes && cache.stats().evictions == 2,
          "Repeated insertion violated the resident budget");
    put(cache, "image-A", {0}, 10.0f);
    check(pinned->tensors[0].values[0] == 0.0f, "Replacement mutated an external immutable handle");
    check(cache.lookup("image-A", {0})->tensors[0].values[0] == 10.0f,
          "Replaced key did not resolve to the new entry");
    check(cache.stats().hits == 4 && cache.stats().misses == 2 && cache.stats().evictions == 2,
          "LRU hit/miss/eviction counters are incorrect");
}

void test_small_capacity_and_lifetime() {
    std::shared_ptr<const Entry> survivor;
    {
        Cache cache(1);
        put(cache, "image-A", {0}, 6.0f);
        survivor = cache.lookup("image-A", {0});
        put(cache, "image-A", {24}, 7.0f);
        check(!cache.lookup("image-A", {0}), "Capacity-one cache retained an evicted entry");
        check(cache.stats().resident_bytes == payload_bytes && cache.stats().evictions == 1,
              "Capacity-one bound is incorrect");
        check(survivor->tensors[0].values[0] == 6.0f, "Eviction invalidated an external handle");
    }
    check(survivor->tensors[0].values[0] == 6.0f, "Cache destruction invalidated an external handle");
}

} // namespace

int main() {
    try {
        test_exact_and_normalized();
        test_order_and_identity();
        test_invalid_keys();
        test_invalid_entries();
        test_lru_and_replacement();
        test_small_capacity_and_lifetime();
        std::cout << "PASS: FPN RAM cache exact FP32/move/normalization/order/identity/"
                     "invalid-key/shape/nonfinite/LRU/budget/lifetime tests\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
