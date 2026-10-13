// SPDX-License-Identifier: Apache-2.0
// Efficient-SAM2 native adaptation; see NOTICE.md for upstream provenance.
// Upstream SAM2 portions: Copyright (c) Meta Platforms, Inc. and affiliates.
#include "sparse_policy.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace efficient_sam2 {
std::vector<float> empty_video_prompt(const float* not_a_point, int channels) {
    if(!not_a_point || channels<=0) throw std::invalid_argument("empty point embedding shape");
    std::vector<float> result(size_t(channels)*2);
    std::copy(not_a_point,not_a_point+channels,result.begin());
    std::copy(not_a_point,not_a_point+channels,result.begin()+channels);
    return result;
}

void validate_sequence(int length, int source_start) {
    if(length<0 || source_start<0 || (length>0 && source_start>=length))
        throw std::invalid_argument("invalid initialized sequence length/source start");
}
bool sequence_frame_valid(int length, int source_start, int local_frame) {
    validate_sequence(length,source_start);
    return local_frame>=0 && (length==0 || int64_t(source_start)+local_frame<length);
}
int pointer_position_denominator(int length, int max_pointers) {
    if(length<0 || max_pointers<2 || length==1)
        throw std::invalid_argument("pointer positions require a multi-frame sequence");
    return (length==0?max_pointers:std::min(length,max_pointers))-1;
}

void gate_absent_logits(std::vector<float>& logits, float object_logit) {
    if (!(object_logit > 0.0f)) std::fill(logits.begin(), logits.end(), -1024.0f);
}

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void dimensions(int width, int height, int window_size) {
    require(width > 0 && height > 0 && window_size > 0,
            "Efficient-SAM2 dimensions must be positive");
    require(static_cast<int64_t>(width) * height <= std::numeric_limits<int>::max(),
            "Efficient-SAM2 spatial grid is too large");
    require(static_cast<int64_t>(window_size)*window_size <= std::numeric_limits<int>::max(),
            "Efficient-SAM2 window is too large");
}
int window_count(int width, int height, int window_size) {
    dimensions(width, height, window_size);
    return ((width - 1) / window_size + 1) * ((height - 1) / window_size + 1);
}
void valid_ids(const std::vector<int>& ids, int count) {
    int previous = -1;
    for (int id : ids) {
        require(id >= 0 && id < count && id > previous,
                "Efficient-SAM2 window IDs must be sorted, unique and in range");
        previous = id;
    }
}
void finite_values(const std::vector<float>& values) {
    for (float value : values)
        require(std::isfinite(value), "Efficient-SAM2 policy input must be finite");
}
void valid_config(const SparseConfig& config) {
    dimensions(config.feature_size, config.feature_size, config.window_size);
    require(config.mask_scale > 0, "Efficient-SAM2 mask scale must be positive");
    const int64_t mask_side = static_cast<int64_t>(config.feature_size)*config.mask_scale;
    require(mask_side <= std::numeric_limits<int>::max() &&
            mask_side*mask_side <= std::numeric_limits<int>::max(), "Mask grid is too large");
    require(static_cast<int64_t>(config.window_size)*config.mask_scale <= std::numeric_limits<int>::max(),
            "Scaled mask window is too large");
    require(config.dilation_kernel > 0 && config.dilation_kernel % 2 == 1,
            "Efficient-SAM2 dilation kernel must be positive and odd");
    require(std::isfinite(config.window_mass) && config.window_mass >= 0.0f &&
            config.window_mass <= 1.0f, "Efficient-SAM2 window mass must be in [0,1]");
    require(std::isfinite(config.object_threshold), "Invalid object threshold");
    require(config.memory_slots >= 3 && config.memory_layers > 0,
            "Efficient-SAM2 memory requires at least three slots and one layer");
    require(std::isfinite(config.drop_ratio) && config.drop_ratio >= 0.0 &&
            config.drop_ratio <= 1.0, "Efficient-SAM2 drop ratio must be in [0,1]");
}
std::vector<int> sorted_importance(const std::vector<float>& values) {
    finite_values(values);
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        return values[a] > values[b];
    });
    return indices;
}
} // namespace

std::vector<int> all_windows(int width, int height, int window_size) {
    std::vector<int> result(window_count(width, height, window_size));
    std::iota(result.begin(), result.end(), 0);
    return result;
}

std::vector<int> complement_windows(const std::vector<int>& ids, int count) {
    require(count >= 0, "Negative window count");
    valid_ids(ids, count);
    std::vector<int> result;
    std::size_t next = 0;
    for (int id = 0; id < count; ++id) {
        if (next < ids.size() && ids[next] == id) ++next;
        else result.push_back(id);
    }
    return result;
}

std::vector<float> pack_windows(const std::vector<float>& spatial, int channels,
                               int width, int height, int window_size,
                               const std::vector<int>& ids) {
    valid_ids(ids, window_count(width, height, window_size));
    require(channels > 0 && spatial.size() == static_cast<std::size_t>(channels) * width * height,
            "Invalid spatial tensor shape");
    const int nw = (width - 1) / window_size + 1;
    const std::size_t window_values = static_cast<std::size_t>(channels) * window_size * window_size;
    std::vector<float> result(ids.size() * window_values, 0.0f);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int x0 = ids[i] % nw * window_size;
        const int y0 = ids[i] / nw * window_size;
        for (int y = 0; y < window_size && y0 + y < height; ++y)
            for (int x = 0; x < window_size && x0 + x < width; ++x) {
                const std::size_t source = static_cast<std::size_t>(channels) * ((y0+y)*width+x0+x);
                const std::size_t target = i * window_values + channels * (y*window_size+x);
                std::copy_n(spatial.data()+source, channels, result.data()+target);
            }
    }
    return result;
}

std::vector<int> valid_window_offsets(const std::vector<int>& ids, int width,
                                      int height, int window_size) {
    valid_ids(ids, window_count(width, height, window_size));
    const int nw = (width - 1) / window_size + 1;
    std::vector<int> result;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int x0 = ids[i] % nw * window_size;
        const int y0 = ids[i] / nw * window_size;
        for (int y = 0; y < window_size && y0+y < height; ++y)
            for (int x = 0; x < window_size && x0+x < width; ++x)
                result.push_back(static_cast<int>(i)*window_size*window_size + y*window_size+x);
    }
    return result;
}

std::vector<int> window_raster_indices(const std::vector<int>& ids, int width,
                                      int height, int window_size) {
    valid_ids(ids, window_count(width, height, window_size));
    const int nw = (width - 1) / window_size + 1;
    std::vector<int> result;
    for (int id : ids) {
        const int x0 = id % nw * window_size;
        const int y0 = id / nw * window_size;
        for (int y = 0; y < window_size && y0+y < height; ++y)
            for (int x = 0; x < window_size && x0+x < width; ++x)
                result.push_back((y0+y)*width+x0+x);
    }
    return result;
}

std::vector<float> gather_tokens(const std::vector<float>& values, int channels,
                                const std::vector<int>& indices) {
    require(channels > 0 && values.size() % channels == 0, "Invalid gather tensor shape");
    const std::size_t count = values.size() / channels;
    std::vector<float> result(indices.size() * channels);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        require(indices[i] >= 0 && static_cast<std::size_t>(indices[i]) < count,
                "Token gather index out of range");
        std::copy_n(values.data()+static_cast<std::size_t>(indices[i])*channels,
                    channels, result.data()+i*channels);
    }
    return result;
}

std::vector<float> scatter_tokens(const std::vector<float>& values, int channels,
                                 const std::vector<int>& indices, int total_tokens) {
    require(channels > 0 && total_tokens >= 0 && values.size() == indices.size()*channels,
            "Invalid scatter tensor shape");
    std::vector<uint8_t> seen(total_tokens, 0);
    std::vector<float> result(static_cast<std::size_t>(total_tokens)*channels, 0.0f);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const int index = indices[i];
        require(index >= 0 && index < total_tokens && !seen[index],
                "Token scatter index duplicated or out of range");
        seen[index] = 1;
        std::copy_n(values.data()+i*channels, channels,
                    result.data()+static_cast<std::size_t>(index)*channels);
    }
    return result;
}

std::vector<float> merge_windows(const std::vector<float>& selected,
                                const std::vector<int>& selected_ids,
                                const std::vector<float>& background,
                                const std::vector<int>& background_ids,
                                int channels, int width, int height, int window_size) {
    const int count = window_count(width, height, window_size);
    valid_ids(selected_ids, count);
    valid_ids(background_ids, count);
    require(channels > 0 && complement_windows(selected_ids, count) == background_ids,
            "Merged window partitions must cover the full grid exactly once");
    const std::size_t per_window = static_cast<std::size_t>(channels)*window_size*window_size;
    require(selected.size() == selected_ids.size()*per_window &&
            background.size() == background_ids.size()*per_window,
            "Invalid packed tensor shape for merge");
    std::vector<float> result(static_cast<std::size_t>(channels)*width*height);
    const int nw = (width-1)/window_size+1;
    auto copy_partition = [&](const std::vector<float>& values, const std::vector<int>& ids) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const int x0 = ids[i]%nw*window_size, y0 = ids[i]/nw*window_size;
            for (int y = 0; y < window_size && y0+y < height; ++y)
                for (int x = 0; x < window_size && x0+x < width; ++x) {
                    const auto source = i*per_window+channels*(y*window_size+x);
                    const auto target = static_cast<std::size_t>(channels)*((y0+y)*width+x0+x);
                    std::copy_n(values.data()+source, channels, result.data()+target);
                }
        }
    };
    copy_partition(selected, selected_ids);
    copy_partition(background, background_ids);
    return result;
}

std::vector<int> mask_windows(const std::vector<float>& masks, int n_masks,
                             const SparseConfig& config) {
    valid_config(config);
    const int side = config.feature_size*config.mask_scale;
    require(n_masks > 0 && masks.size() == static_cast<std::size_t>(n_masks)*side*side,
            "Invalid multimask shape");
    finite_values(masks);
    std::vector<uint8_t> union_mask(static_cast<std::size_t>(side)*side, 0);
    for (int m = 0; m < n_masks; ++m)
        for (int i = 0; i < side*side; ++i)
            if (masks[static_cast<std::size_t>(m)*side*side+i] > 0.0f) union_mask[i] = 1;
    const int half = config.dilation_kernel/2;
    const int ws = config.window_size*config.mask_scale;
    const int nw = (side-1)/ws+1;
    std::vector<uint8_t> selected(nw*nw, 0);
    // A dilated pixel belongs to a window iff the source pixel's dilation box
    // overlaps it. Marking that box's window range is exactly max-pool dilation.
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) if (union_mask[y*side+x]) {
            const int xlo = std::max(0,x-half)/ws, xhi = std::min(side-1,x+half)/ws;
            const int ylo = std::max(0,y-half)/ws, yhi = std::min(side-1,y+half)/ws;
            for (int wy = ylo; wy <= yhi; ++wy)
                for (int wx = xlo; wx <= xhi; ++wx) selected[wy*nw+wx] = 1;
        }
    std::vector<int> result;
    for (int i = 0; i < nw*nw; ++i) if (selected[i]) result.push_back(i);
    return result;
}

std::vector<float> saliency_window_sums(const std::vector<float>& saliency,
                                      const SparseConfig& config) {
    valid_config(config);
    const int side = config.feature_size, ws = config.window_size;
    require(saliency.size() == static_cast<std::size_t>(side)*side, "Invalid saliency shape");
    finite_values(saliency);
    const int nw = (side-1)/ws+1;
    std::vector<float> sums(nw*nw, 0.0f);
    for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x) {
            require(saliency[y*side+x] >= 0.0f, "Saliency probabilities must be nonnegative");
            sums[(y/ws)*nw+x/ws] += saliency[y*side+x];
        }
    return sums;
}

std::vector<int> cumulative_windows(const std::vector<float>& weights, float threshold) {
    require(std::isfinite(threshold) && threshold >= 0.0f, "Invalid cumulative threshold");
    auto indices = sorted_importance(weights);
    for (float weight : weights)
        require(weight >= 0.0f, "Window weights must be nonnegative");
    double cumulative = 0.0;
    std::size_t keep = indices.size();
    for (std::size_t i = 0; i < indices.size(); ++i) {
        cumulative += weights[indices[i]];
        if (cumulative >= threshold) { keep = i+1; break; }
    }
    indices.resize(keep);
    std::sort(indices.begin(), indices.end());
    return indices;
}

WindowRouter::WindowRouter(const SparseConfig& config) : config_(config) {
    valid_config(config_);
    reset();
}
void WindowRouter::reset() {
    seen_frame_ = false;
    selected_ = all_windows(config_.feature_size, config_.feature_size, config_.window_size);
    pending_saliency_windows_.assign(selected_.size(),0);
    pending_saliency_count_ = 0;
}
const std::vector<int>& WindowRouter::update(const std::vector<float>& masks, int n_masks,
                                           float object_logit,
                                           const std::vector<float>& saliency) {
    auto ids = mask_windows(masks, n_masks, config_);
    // Validate before mutating state.
    const auto saliency_ids = cumulative_windows(saliency_window_sums(saliency, config_), config_.window_mass);
    require(std::isfinite(object_logit), "Object score must be finite");
    for (int id : saliency_ids) pending_saliency_windows_[id] = 1;
    ++pending_saliency_count_;
    if (!seen_frame_) { seen_frame_ = true; return selected_; }
    if (ids.empty() || object_logit < config_.object_threshold) {
        for (std::size_t id = 0; id < pending_saliency_windows_.size(); ++id)
            if (pending_saliency_windows_[id]) ids.push_back(static_cast<int>(id));
        std::fill(pending_saliency_windows_.begin(), pending_saliency_windows_.end(), 0);
        pending_saliency_count_ = 0;
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    selected_.swap(ids);
    return selected_;
}

std::vector<float> importance_from_logits(const std::vector<float>& logits,
                                        int n_queries, int n_keys) {
    require(n_queries > 0 && n_keys > 0 && logits.size() == static_cast<std::size_t>(n_queries)*n_keys,
            "Invalid latest-frame attention-logit shape");
    finite_values(logits);
    std::vector<double> sums(n_keys, 0.0);
    std::vector<double> exps(n_keys);
    for (int q = 0; q < n_queries; ++q) {
        const float* row = logits.data()+static_cast<std::size_t>(q)*n_keys;
        const float maximum = *std::max_element(row, row+n_keys);
        double denominator = 0.0;
        for (int k = 0; k < n_keys; ++k) {
            exps[k] = std::exp(static_cast<double>(row[k])-maximum);
            denominator += exps[k];
        }
        for (int k = 0; k < n_keys; ++k) sums[k] += exps[k]/denominator;
    }
    std::vector<float> result(n_keys);
    for (int k = 0; k < n_keys; ++k) result[k] = static_cast<float>(sums[k]/n_queries);
    return result;
}

std::vector<uint8_t> topk_mask(const std::vector<float>& importance, int keep_count) {
    require(keep_count >= 0 && static_cast<std::size_t>(keep_count) <= importance.size(),
            "Invalid retained memory-token count");
    auto indices = sorted_importance(importance);
    std::vector<uint8_t> mask(importance.size(), 0);
    for (int i = 0; i < keep_count; ++i) mask[indices[i]] = 1;
    return mask;
}

MemoryQueue::MemoryQueue(int start_frame, const SparseConfig& config) : config_(config) {
    valid_config(config_);
    reset(start_frame);
}
void MemoryQueue::reset(int start_frame) {
    require(start_frame >= 0, "Negative conditioning-frame index");
    start_frame_ = start_frame;
    last_recorded_query_ = -1;
    frames_.clear();
}
bool MemoryQueue::should_record(int current_frame) const {
    return current_frame > start_frame_+1;
}
int MemoryQueue::keep_count() const {
    // double matches the Python float configuration and int() truncation.
    return static_cast<int>((1.0-config_.drop_ratio)*config_.feature_size*config_.feature_size);
}
void MemoryQueue::record(int current_frame,
                         const std::vector<std::vector<float>>& layer_importance) {
    require(current_frame >= start_frame_, "Memory queue cannot run in reverse");
    if (!should_record(current_frame)) return;
    require((last_recorded_query_ < 0 && current_frame == start_frame_+2) ||
            current_frame == last_recorded_query_+1,
            "Memory queue requires contiguous forward frame updates");
    require(static_cast<int>(layer_importance.size()) == config_.memory_layers,
            "Wrong number of memory saliency layers");
    FrameMasks entry;
    entry.frame_index = current_frame-1;
    for (const auto& importance : layer_importance) {
        require(importance.size() == static_cast<std::size_t>(config_.feature_size)*config_.feature_size,
                "Wrong number of memory saliency tokens");
        entry.layers.push_back(topk_mask(importance, keep_count()));
    }
    if (static_cast<int>(frames_.size()) == config_.memory_slots-2) frames_.pop_front();
    frames_.push_back(std::move(entry));
    last_recorded_query_ = current_frame;
}
MemorySelection MemoryQueue::select(const std::vector<int>& frame_ids, int layer,
                                    int pointer_tokens) const {
    require(layer >= 0 && layer < config_.memory_layers && pointer_tokens >= 0,
            "Invalid memory selection layer/pointer count");
    require(!frame_ids.empty() && static_cast<int>(frame_ids.size()) <= config_.memory_slots,
            "Invalid number of memory frames");
    require(frame_ids.front() == start_frame_, "First memory slot must be the conditioning frame");
    for (std::size_t i = 1; i < frame_ids.size(); ++i)
        require(frame_ids[i] > frame_ids[i-1], "Memory frames must be unique and in increasing order");
    const int tokens = config_.feature_size*config_.feature_size;
    MemorySelection result;
    result.pointer_tokens = pointer_tokens;
    result.pruned = static_cast<int>(frames_.size()) == config_.memory_slots-2;
    if (result.pruned) {
        require(static_cast<int>(frame_ids.size()) == config_.memory_slots,
                "Full SMR queue requires the complete memory-slot set");
        require(frame_ids.back() == last_recorded_query_,
                "SMR selection must precede recording the current query frame");
        for (std::size_t i = 1; i+1 < frame_ids.size(); ++i)
            require(frames_[i-1].frame_index == frame_ids[i], "SMR masks do not match memory-frame identities");
    }
    for (std::size_t s = 0; s < frame_ids.size(); ++s) {
        int kept = 0;
        for (int t = 0; t < tokens; ++t) {
            const bool keep = !result.pruned || s == 0 || s+1 == frame_ids.size() ||
                              frames_[s-1].layers[layer][t];
            if (keep) { result.original_indices.push_back(static_cast<int>(s)*tokens+t); ++kept; }
        }
        result.spatial_tokens_per_frame.push_back(kept);
        result.spatial_tokens += kept;
    }
    const int pointer_start = static_cast<int>(frame_ids.size())*tokens;
    for (int i = 0; i < pointer_tokens; ++i) result.original_indices.push_back(pointer_start+i);
    return result;
}

int temporal_position(int current_frame, int memory_frame, bool conditioning, int memory_slots) {
    require(memory_slots >= 2 && memory_frame >= 0 && current_frame > memory_frame,
            "Invalid forward memory-frame distance");
    if (conditioning) return 0;
    const int distance = current_frame-memory_frame;
    require(distance < memory_slots, "Non-conditioning frame is outside the stride-1 memory horizon");
    return memory_slots-distance;
}

float round_bfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits |= 0x00400000u; // retain NaN, never round it to infinity
    } else {
        bits += 0x7fffu + ((bits >> 16) & 1u); // round-to-nearest, ties-to-even
    }
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace efficient_sam2
