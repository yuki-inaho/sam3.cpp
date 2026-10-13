// SPDX-License-Identifier: Apache-2.0
// Efficient-SAM2 native adaptation; see NOTICE.md for upstream provenance.
// Upstream SAM2 portions: Copyright (c) Meta Platforms, Inc. and affiliates.
#include "sparse_policy.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace efficient_sam2;
namespace {
int checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F action, const char* message) {
    bool threw = false;
    try { action(); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, message);
}
void close(float actual, float expected, const char* message, float tolerance = 1e-6f) {
    check(std::abs(actual-expected) <= tolerance, message);
}
std::vector<float> onehot(int y, int x) {
    std::vector<float> map(4096, 0.0f);
    map[y*64+x] = 1.0f;
    return map;
}

void layout_tests() {
    const int C = 2, W = 5, H = 3, ws = 2;
    std::vector<float> input(C*W*H);
    std::iota(input.begin(), input.end(), 1.0f);
    const std::vector<int> selected = {0,2,3,5};
    const auto background = complement_windows(selected, 6);
    check(background == std::vector<int>({1,4}), "window complement");
    const auto packed = pack_windows(input,C,W,H,ws,selected);
    check(packed.size() == 32, "packed shape includes zero padding");
    const auto offsets = valid_window_offsets(selected,W,H,ws);
    const auto raster = window_raster_indices(selected,W,H,ws);
    check(offsets == std::vector<int>({0,1,2,3,4,6,8,9,12}), "packed valid offsets");
    check(raster == std::vector<int>({0,1,5,6,4,9,10,11,14}), "Python window-major raster order");
    check(gather_tokens(packed,C,offsets) == gather_tokens(input,C,raster), "gather matches original pixels");
    const auto restored = scatter_tokens(gather_tokens(packed,C,offsets), C, offsets, 16);
    check(restored == packed, "scatter resets padded positions to zero");
    const auto bg = pack_windows(input,C,W,H,ws,background);
    check(merge_windows(packed,selected,bg,background,C,W,H,ws) == input,
          "selected/background merge round trip");
    for (int index : complement_windows(offsets,16)) {
        close(packed[index*C],0.0f,"padding is zero");
        close(packed[index*C+1],0.0f,"all padding channels are zero");
    }
    const auto all = all_windows(64,64,14);
    check(all.size() == 25 && all.back() == 24, "base-plus 25-window grid");
    check(valid_window_offsets(all,64,64,14).size() == 4096, "all-window valid token count");
    check(valid_window_offsets({24},64,64,14).size() == 64, "corner window only contains 8x8 pixels");
    check(valid_window_offsets({4,20,24},64,64,14).size() == 288, "edge and corner token counts");
    std::vector<float> dense(64*64,1.0f);
    auto empty = pack_windows(dense,1,64,64,14,{});
    check(empty.empty(), "empty selected path supported");
    check(merge_windows(empty,{},pack_windows(dense,1,64,64,14,all),all,1,64,64,14)==dense,
          "all-background merge supported");
    rejects([&]{pack_windows(input,C,W,H,ws,{2,0});}, "unsorted windows rejected");
    rejects([&]{pack_windows(input,C,W,H,ws,{6});}, "out-of-range windows rejected");
    rejects([&]{scatter_tokens({1,2},1,{0,0},2);}, "duplicate scatter indices rejected");
    rejects([&]{gather_tokens(input,C,{15});}, "out-of-range gather rejected");
    rejects([&]{merge_windows(packed,selected,bg,{1,3},C,W,H,ws);}, "overlapping merge rejected");
}

void routing_tests() {
    SparseConfig config;
    std::vector<float> masks(2*256*256,-1.0f);
    masks[55*256+55] = 1.0f;
    masks[256*256+255*256+255] = 1.0f;
    masks[100*256+200] = 0.0f;
    check(mask_windows(masks,2,config)==std::vector<int>({0,1,5,6,24}),
          "multimask union, strict positive threshold, dilation across windows");
    config.dilation_kernel=1;
    check(mask_windows(masks,2,config)==std::vector<int>({0,24}),"no dilation policy");
    const auto sums=saliency_window_sums(std::vector<float>(4096,1.0f/4096),config);
    close(sums[0],196.0f/4096,"interior-window mass");
    close(sums[4],112.0f/4096,"right edge zero padding");
    close(sums[24],64.0f/4096,"corner zero padding");
    close(std::accumulate(sums.begin(),sums.end(),0.0f),1.0f,"pooling preserves mass");
    check(cumulative_windows({0.2f,0.6f,0.2f},0.7f)==std::vector<int>({0,1}),
          "cumulative selector includes crossing item and sorts output IDs");
    check(cumulative_windows({0,0,0},0.7f)==std::vector<int>({0,1,2}),
          "insufficient total mass selects all");
    check(cumulative_windows({0.1f,0.9f},0.0f)==std::vector<int>({1}),
          "zero threshold still includes first element as Python does");
    rejects([]{cumulative_windows({1.0f,-0.1f},0.5f);}, "negative weights rejected even below cutoff");
    WindowRouter router(config);
    std::vector<float> positive(256*256,-1.0f);
    positive[10*256+10]=1.0f;
    check(router.update(positive,1,10,onehot(0,0)).size()==25,"conditioning forces all windows");
    check(router.update(positive,1,10,onehot(20,20))==std::vector<int>({0}),
          "second update routes following frame by mask");
    check(router.pending_saliency_count()==2,"source retains saliency until fallback");
    std::vector<float> negative(256*256,-1.0f);
    check(router.update(negative,1,10,onehot(63,63))==std::vector<int>({0,6,24}),
          "empty-mask fallback uses all accumulated saliency maps");
    check(router.pending_saliency_count()==0,"fallback clears accumulated saliency");
    check(router.update(positive,1,4.999f,onehot(63,63))==std::vector<int>({0,24}),
          "low object logit unions mask and saliency");
    check(router.update(positive,1,5.0f,onehot(63,63))==std::vector<int>({0}),
          "object logit exactly threshold does not trigger fallback");
    router.reset();
    check(router.selected().size()==25 && router.pending_saliency_count()==0,"router reset");
    auto invalid=onehot(0,0); invalid[1]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{router.update(positive,1,10,invalid);},"nonfinite saliency rejected");
    check(router.pending_saliency_count()==0,"failed update leaves router state unchanged");
}

void memory_tests() {
    auto importance=importance_from_logits({1000,1001,1002, -1000,-999,-998},2,3);
    const float denominator=1+std::exp(1.0f)+std::exp(2.0f);
    close(importance[0],1/denominator,"stable latest-slot softmax");
    close(importance[2],std::exp(2.0f)/denominator,"softmax query mean");
    check(topk_mask({0.1f,0.7f,0.2f},2)==std::vector<uint8_t>({0,1,1}),
          "top-k mask retains original raster ordering");
    check(topk_mask({1,1,1},1)==std::vector<uint8_t>({1,0,0}),"documented deterministic ties");
    check(topk_mask({1,2},0)==std::vector<uint8_t>({0,0}),"zero-token retention");
    MemoryQueue queue;
    check(queue.keep_count()==204,"default 95 percent drop retains 204 tokens");
    check(!queue.should_record(1) && queue.should_record(2),"SMR capture warmup boundary");
    std::vector<std::vector<float>> layer_scores(4,std::vector<float>(4096));
    for(int layer=0;layer<4;++layer)
        for(int token=0;token<4096;++token)
            layer_scores[layer][token]=static_cast<float>((token+layer*1000)%4096);
    for(int query=1;query<=6;++query) {
        std::vector<int> frame_ids(query);
        std::iota(frame_ids.begin(),frame_ids.end(),0);
        auto selection=queue.select(frame_ids,0,query*4);
        check(!selection.pruned,"SMR remains dense through query frame 6");
        check(selection.spatial_tokens==query*4096,"dense warmup token count");
        queue.record(query,layer_scores);
    }
    check(queue.frames().size()==5 && queue.frames().front().frame_index==1 &&
          queue.frames().back().frame_index==5,"five masks aligned to recalled frames 1 through 5");
    auto sparse=queue.select({0,1,2,3,4,5,6},0,28);
    check(sparse.pruned && sparse.spatial_tokens==9212,"first pruning at query frame 7");
    check(sparse.spatial_tokens_per_frame==std::vector<int>({4096,204,204,204,204,204,4096}),
          "conditioning and predecessor dense, five older frames sparse");
    check(sparse.original_indices.size()==9240 && sparse.original_indices.back()==28699,
          "all pointer tokens appended and retained");
    check(sparse.original_indices[4096]==4096+3892,"first sparse key keeps original coordinate");
    auto sparse_other=queue.select({0,1,2,3,4,5,6},1,28);
    check(sparse_other.original_indices!=sparse.original_indices,"each attention layer has independent mask");
    std::vector<float> fake_rope(28672*2);
    for(int i=0;i<28672;++i) {fake_rope[2*i]=static_cast<float>(i%4096);fake_rope[2*i+1]=-fake_rope[2*i];}
    std::vector<int> spatial_ids(sparse.original_indices.begin(),sparse.original_indices.begin()+sparse.spatial_tokens);
    auto gathered_rope=gather_tokens(fake_rope,2,spatial_ids);
    close(gathered_rope[2*4096],3892,"sparse RoPE retains original raster position, not compact index");
    queue.record(7,layer_scores);
    check(queue.frames().front().frame_index==2 && queue.frames().back().frame_index==6,"FIFO rolls one frame");
    check(queue.select({0,2,3,4,5,6,7},0,32).pruned,"next query uses rolled queue");
    rejects([&]{queue.select({0,1,2,3,4,5,7},0,32);},"stale frame-mask identities rejected");
    rejects([&]{queue.record(9,layer_scores);},"skipped query frame rejected");
    rejects([&]{queue.record(7,layer_scores);},"duplicate query frame rejected");
    queue.reset(10);
    check(queue.frames().empty() && !queue.should_record(11),"nonzero start reset");
    queue.record(12,layer_scores);
    check(queue.frames().front().frame_index==11,"nonzero start captures preceding frame");
    rejects([&]{queue.select({0,11},0,4);},"wrong conditioning frame rejected");
    check(temporal_position(7,6,false)==6,"latest memory temporal position 6");
    check(7-temporal_position(7,6,false)-1==0,"latest memory temporal embedding index 0");
    check(temporal_position(7,1,false)==1,"oldest memory temporal position 1");
    check(temporal_position(20,0,true)==0,"conditioning temporal position independent of age");
    rejects([]{temporal_position(8,1,false);},"out-of-horizon nonconditioning memory rejected");
    close(round_bfloat16(1.00390625f),1.0f,"BF16 halfway rounds even down");
    close(round_bfloat16(1.01171875f),1.015625f,"BF16 halfway rounds even up");
    check(std::isnan(round_bfloat16(std::numeric_limits<float>::quiet_NaN())),"BF16 preserves NaN");
    check(std::isinf(round_bfloat16(std::numeric_limits<float>::infinity())),"BF16 preserves infinity");
}
} // namespace

int main() {
    try {
        layout_tests(); routing_tests(); memory_tests();
        for(float score:{-1.0f,0.0f,-0.0f}) {
            std::vector<float> masks{1.0f,-5.0f,300.0f};
            efficient_sam2::gate_absent_logits(masks,score);
            check(masks==std::vector<float>(3,-1024.0f),"conditioning negative/zero object score gates every raw mask");
        }
        check(efficient_sam2::sequence_frame_valid(1,0,0),"single frame conditioning allowed");
        check(!efficient_sam2::sequence_frame_valid(1,0,1),"single frame propagation rejected");
        check(efficient_sam2::sequence_frame_valid(121,113,7),"arbitrary source start includes last frame");
        check(!efficient_sam2::sequence_frame_valid(121,113,8),"source end bound precedes state mutation");
        for(const auto& test:std::vector<std::pair<int,int>>{{0,15},{2,1},{8,7},{16,15},{17,15}})
            check(efficient_sam2::pointer_position_denominator(test.first)==test.second,"pointer PE uses full initialized sequence length");
        rejects([]{efficient_sam2::pointer_position_denominator(1);},"single-frame PE division is forbidden");
        rejects([]{efficient_sam2::validate_sequence(8,8);},"seed outside source rejected");
        const float embedding[8]={-.875f,-.625f,-.375f,-.125f,.125f,.375f,.625f,.875f};
        const auto empty=efficient_sam2::empty_video_prompt(embedding,8);
        check(empty.size()==16,"no-prompt video has two sparse tokens");
        check(std::equal(empty.begin(),empty.begin()+8,embedding),"first empty point content preserved");
        check(std::equal(empty.begin()+8,empty.end(),embedding),"padding empty point content preserved");
        rejects([]{efficient_sam2::empty_video_prompt(nullptr,8);},"missing empty-point embedding rejected");
        std::vector<float> present{1.0f,-5.0f,300.0f};
        efficient_sam2::gate_absent_logits(present,1.0f);
        check(present==std::vector<float>({1.0f,-5.0f,300.0f}),"positive conditioning score preserves logits exactly");
        std::cout << "Efficient-SAM2 sparse policy: " << checks << " checks passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "Efficient-SAM2 sparse policy failed after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
