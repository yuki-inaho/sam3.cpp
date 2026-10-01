# SAM 3.1 validation evidence

These records come from the current native C++ run with the named trained ConvRot INT8 checkpoint. `conversion.json` verifies every payload byte. `real-*-validation.json` independently compares completed C++ predictions to synthetic input colors. The synthetic inputs are artificial; the weights are trained.

The runtime's `trained_checkpoint_verified=false` and `accuracy_validation_performed=false` describe checks the generic runtime does not perform. The external HF LFS identity check and synthetic IoU checks are recorded in `status.json` and the validation files. Official PyTorch parity and natural-video accuracy remain unverified.

The follow-up E2E (`e2e.json`) verifies the exact trained GGUF and runs native image/video end to end. `performance.json` records exact-logit image speedup; `parallel-operators.json` covers the large-tensor OpenMP path. Review fixes and measurements are in `temp/review_SAM31_Oct01-2026.md`.
