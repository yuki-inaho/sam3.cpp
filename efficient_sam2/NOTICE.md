# Efficient-SAM2 source provenance

The Efficient-SAM2-specific packed-window encoder, sparse memory policy,
attention saliency extraction and memory storage semantics were adapted into
C++ from [Efficient-SAM2](https://github.com/yuki-inaho/Efficient-SAM2), commit
`8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5`, whose root license is Apache-2.0.
The source contains SAM2 material bearing:

> Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Relevant upstream files include `build_model/window_bypass.py`,
`sam2/modeling/sam2_base.py`, `sam2/modeling/memory_attention.py`,
`sam2/modeling/sam/transformer.py`, and
`sam2/sam2_video_predictor_with_bypass.py`.

The modified/adapted implementation is in `hiera.inl`, `memory.inl`,
`sparse_policy.{h,cpp}`, its policy tests, and the explicitly Efficient-SAM2
branches of `../sam3.cpp`. These additions retain Apache-2.0 terms; they are
not represented as exclusively covered by the repository's original MIT
license. The Apache-2.0 license is reproduced in `licenses/Apache-2.0.txt`.
No upstream NOTICE file was found in the inspected source root.

Changes include C++/GGML graph execution, graph isolation, bounded routing
state, deterministic tie ordering, explicit unsupported-mode rejection and
runtime validation. These changes do not imply upstream endorsement.

The GGML dependency and the original sam3.cpp code retain their existing
licenses. ConvRot storage code reuses the repository's existing SAM31
quantization implementation with its existing notices. Model checkpoints
and private annotation data are not distributed in this source change.

## Pillow-compatible video preprocessing

`resize.{h,cpp}` adapts the bicubic coefficient normalization, fixed-point
coefficient rounding and separable uint8-pass behavior of Pillow's
[`src/libImaging/Resample.c`](https://github.com/python-pillow/Pillow/blob/11.1.0/src/libImaging/Resample.c).
Pillow copyright and permission terms are preserved in `licenses/Pillow.txt`.
The adaptation is RGB-only, uses C++ vectors and 64-bit sums, and replaces the
lookup-table clip operation with a bounded integer operation. It is checked
against the installed Pillow oracle on upsample/downsample/edge geometries.
