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
 * Two bundles are published; pick by CPU:
 *   sam3-linux-x86_64.tar.gz         AVX2 + FMA. Runs on any x86_64 since
 *                                    Haswell. Use this if unsure.
 *   sam3-linux-x86_64-avx512.tar.gz  adds AVX-512 + VNNI, whose vpdpbusd
 *                                    carries the int8 dot products behind
 *                                    q8_0/q4_0 inference. Faster on a CPU
 *                                    that has it; SIGILLs on one that
 *                                    does not.
 *
 *   Check with:  grep -q avx512_vnni /proc/cpuinfo && echo "use -avx512"
 *
 * Requirements: x86_64 CPU with AVX2 + FMA (plus AVX-512 + VNNI for the
 * -avx512 bundle), glibc >= 2.38, any Linux.
 * Built and verified on Debian 13 (trixie, glibc 2.41); no libgomp or
 * libstdc++ runtime is needed (statically linked).
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
