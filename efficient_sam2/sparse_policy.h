// SPDX-License-Identifier: Apache-2.0
// Efficient-SAM2 native adaptation; see NOTICE.md for upstream provenance.
// Upstream SAM2 portions: Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef EFFICIENT_SAM2_SPARSE_POLICY_H
#define EFFICIENT_SAM2_SPARSE_POLICY_H

#include <cstdint>
#include <deque>
#include <vector>

// Pure CPU policy/layout helpers for the base-plus Efficient-SAM2 forward path.
// These functions do not implement inference or establish numerical model parity.
namespace efficient_sam2 {

// SAM2 video hard absence gate, applied before resizing and memory encoding.
// Conditioning image prediction without video semantics does not use it.
void gate_absent_logits(std::vector<float>& logits, float object_logit);

// Python video supplies an empty point and PromptEncoder appends padding.
std::vector<float> empty_video_prompt(const float* not_a_point, int channels);

// Known initialized sequence bounds. length=0 denotes an unknown stream.
void validate_sequence(int length, int source_start);
bool sequence_frame_valid(int length, int source_start, int local_frame);
int pointer_position_denominator(int length, int max_pointers=16);

struct SparseConfig {
    int feature_size = 64;
    int window_size = 14;
    int mask_scale = 4;
    int dilation_kernel = 5; // 1 disables dilation
    float window_mass = 0.7f;
    float object_threshold = 5.0f;
    int memory_slots = 7;
    int memory_layers = 4;
    double drop_ratio = 0.95;
};

// Layout convention: channel is the fastest dimension. Spatial data is
// [C,W,H]; packed data is [C,window_size,window_size,selected_window_count].
// IDs must be unique and ascending, in y-window-major, x-window-minor order.
std::vector<int> all_windows(int width, int height, int window_size);
std::vector<int> complement_windows(const std::vector<int>& ids, int count);
std::vector<float> pack_windows(const std::vector<float>& spatial, int channels,
                               int width, int height, int window_size,
                               const std::vector<int>& ids);
std::vector<int> valid_window_offsets(const std::vector<int>& ids, int width,
                                      int height, int window_size);
std::vector<int> window_raster_indices(const std::vector<int>& ids, int width,
                                      int height, int window_size);
// Gather/scatter preserve the supplied order. Scatter starts with zeros and
// rejects duplicate/out-of-range indices. This also gathers original RoPE rows
// when channels is the number of floats in one token's RoPE representation.
std::vector<float> gather_tokens(const std::vector<float>& values, int channels,
                                const std::vector<int>& indices);
std::vector<float> scatter_tokens(const std::vector<float>& values, int channels,
                                 const std::vector<int>& indices, int total_tokens);
std::vector<float> merge_windows(const std::vector<float>& selected,
                                const std::vector<int>& selected_ids,
                                const std::vector<float>& background,
                                const std::vector<int>& background_ids,
                                int channels, int width, int height, int window_size);

// masks: [n_masks,H*mask_scale,W*mask_scale], row-major; strict logit > 0.
std::vector<int> mask_windows(const std::vector<float>& masks, int n_masks,
                             const SparseConfig& config = SparseConfig());
std::vector<float> saliency_window_sums(const std::vector<float>& saliency,
                                      const SparseConfig& config = SparseConfig());
// Include the first item that reaches threshold. No normalization or cap.
// If mass never reaches threshold, return every index. Ties use index order.
std::vector<int> cumulative_windows(const std::vector<float>& weights, float threshold);

class WindowRouter {
public:
    explicit WindowRouter(const SparseConfig& config = SparseConfig());
    void reset();
    const std::vector<int>& selected() const { return selected_; }
    // Call once after a completed conditioning/propagation frame. Saliency is
    // the 64x64 mean of decoder token-to-image probabilities (heads, first
    // seven tokens, and three attention modules). The first update keeps all
    // windows. Reference behavior accumulates saliency until fallback uses it.
    const std::vector<int>& update(const std::vector<float>& masks, int n_masks,
                                  float object_logit,
                                  const std::vector<float>& saliency);
    std::size_t pending_saliency_count() const { return pending_saliency_count_; }
private:
    SparseConfig config_;
    bool seen_frame_ = false;
    std::vector<int> selected_;
    // The reference's eventual fallback unions independently selected windows
    // from every accumulated map. Cache that union rather than retaining an
    // unbounded history of 64x64 maps during a long high-confidence track.
    std::vector<uint8_t> pending_saliency_windows_;
    std::size_t pending_saliency_count_ = 0;
};

// Logits are [n_queries,n_keys] in row-major order. Important: pass only the
// last spatial memory frame's raw scaled QK scores, excluding pointer keys.
std::vector<float> importance_from_logits(const std::vector<float>& logits,
                                        int n_queries, int n_keys);
std::vector<uint8_t> topk_mask(const std::vector<float>& importance, int keep_count);

struct MemorySelection {
    // Indices into the original [all spatial frames, all pointer tokens] prompt.
    std::vector<int> original_indices;
    int spatial_tokens = 0;
    int pointer_tokens = 0;
    bool pruned = false;
    std::vector<int> spatial_tokens_per_frame;
};

struct FrameMasks {
    int frame_index = -1; // the recalled predecessor, not its query frame
    std::vector<std::vector<uint8_t>> layers;
};

class MemoryQueue {
public:
    explicit MemoryQueue(int start_frame = 0,
                         const SparseConfig& config = SparseConfig());
    void reset(int start_frame = 0);
    bool should_record(int current_frame) const;
    // Each layer_importance[l] is the mean re-softmax of latest-slot QK logits.
    // Single-object, contiguous forward, stride-1 only. Records current-1.
    // First two frame positions do not record. Later calls must be contiguous.
    void record(int current_frame,
                const std::vector<std::vector<float>>& layer_importance);
    // Frame order: sole conditioning frame first, followed by increasing
    // non-conditioning frame IDs, ending at the immediately previous frame.
    // Queue is used only once full. Full queues must match all middle frames;
    // inconsistent state throws rather than silently applying wrong masks.
    MemorySelection select(const std::vector<int>& frame_ids, int layer,
                           int pointer_tokens) const;
    const std::deque<FrameMasks>& frames() const { return frames_; }
    int keep_count() const;
private:
    SparseConfig config_;
    int start_frame_ = 0;
    int last_recorded_query_ = -1;
    std::deque<FrameMasks> frames_;
};

// Python's t_pos is NOT the encoding index. Encoding index = slots-t_pos-1.
// For stride 1: latest noncond t_pos=6 => encoding index 0 (slots=7).
int temporal_position(int current_frame, int memory_frame, bool conditioning,
                      int memory_slots = 7);

// Matches the explicit Python mask-memory float32 -> bfloat16 -> float32
// storage boundary. Float arithmetic elsewhere remains the caller's choice.
float round_bfloat16(float value);

} // namespace efficient_sam2
#endif
