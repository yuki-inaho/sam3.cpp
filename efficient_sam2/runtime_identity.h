#pragma once
#include <string>

namespace efficient_sam2 {
struct RuntimeIdentity {
    std::string digest;
    std::string json;
};
// Linux: SHA256 provenance of the running executable and all loaded file-backed
// ELF objects. Call after loading compute backends and before cache admission.
// On failure clears result and sets error; caller must disable persistent cache,
// not inference. Unsupported platforms explicitly return false.
bool runtime_identity(RuntimeIdentity & result, std::string & error);
}
