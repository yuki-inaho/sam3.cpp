#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
namespace efficient_sam2 { namespace prefix_cache {
// Thin wrappers around unmodified, licensed llama.cpp-vendored SHA256.
std::string sha256_bytes(const void* data, size_t bytes);
std::string sha256_file(const std::string& path, std::string& error);
// Domain separator + uint64 little-endian width,height + contiguous RGB bytes.
std::string sha256_rgb(const uint8_t* rgb, size_t bytes, int width, int height);
} }
