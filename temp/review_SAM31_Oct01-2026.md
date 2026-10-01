# SAM31 review and corrections

## Scope

Review of merge `bb1617e`, followed by authorized E2E, fixes, refactoring and CPU optimization on `fix/sam31-e2e-performance`. Single agent. No model/credentials in Git.

## Findings

1. **P1: interactive decoder always excludes single-mask token 0.** `graph.cpp` selected from candidates1–3 regardless of click count. The pinned model uses multimask for0–1 clicks, stable single-mask0 for multiple clicks, and dynamic fallback when unstable. A controlled CLI E2E reproduced expected quality1000 vs actual100. Corrected candidate selection and matching object pointer; stable/unstable and single/multiple-point cases now pass.
2. **P2: result verification used Python assert.** Optimized Python could disable checks. Changed to explicit runtime validation and verified incomplete reports fail in normal and `-O` modes.
3. **P2: source packaging requires Git metadata and collected untracked files.** Added manifest-based packaging for source deliveries and limited checkout sources to tracked/staged files. Repack restoration test is recorded in workdoc.

## Investigation resolved without code change

The suspected temporal memory index difference was not a bug: official non-conditioning `t_pos` is `num_maskmem - frame_distance`; substituting it gives exactly the current `distance - 1` index. Conditioning frames use actual distance and out-of-range row6. The formula was retained.

## Verification and limits

Existing native quant/graph/operator/CLI/storage regression groups pass. New actual-trained-model E2E checks fixed GGUF bytes, C++ input generation, image/video, independent synthetic IoU, finite logits, IDs and memory20 blocks. Default CTest remains five fast groups; real E2E is explicitly enabled by `SAM31_E2E_MODEL`.

Exact GELU expression and row-local LayerNorm arithmetic are parallelized. Same model/input/16 OpenMP and16 BLAS threads:44.872→35.354 seconds,21.2% reduction; logits byte-exact. Measurements are single-run comparisons on this host, not cross-hardware guarantees. Official full PyTorch parity and natural-video accuracy are outside this smoke test.
