/**
 * RELEASE BUNDLE — sam3.cpp portable Linux x86_64
 *
 * Contents:
 *   bin/sam3_seg              headless point/box segmentation CLI (EdgeTAM etc.)
 *   bin/sam3_profile_edgetam  per-stage encoder latency profiler
 *   bin/sam3_benchmark        video tracking benchmark (needs ffmpeg CLI)
 *   bin/sam3_quantize         weight quantization tool
 *   models/edgetam_*.ggml     vendored EdgeTAM weights (f16 / q8_0 / q4_0)
 *   data/                     sample image + video
 *
 * Requirements: x86_64 CPU with AVX2 + FMA, glibc >= 2.17, any Linux.
 * The video benchmark additionally needs the `ffmpeg` CLI on PATH.
 * (Debian/Ubuntu: sudo apt install ffmpeg)
 *
 * Quick start:
 *   ./bin/sam3_seg --model models/edgetam_q8_0.ggml \
 *                  --image data/test_image.jpg --x 315 --y 250 \
 *                  --out mask.png --cpu
 *   ./bin/sam3_benchmark --models-dir models --video data/test_video.mp4 \
 *                        --filter edgetam --cpu-only --n-frames 5
 */
