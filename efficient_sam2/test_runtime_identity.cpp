#include "runtime_identity.h"
#include <cstdio>
#if defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#endif
int main(int argc, char ** argv) {
#if defined(__linux__)
    // Optional disposable DSO lets the oracle test loaded/missing/replaced input.
    if (argc > 1) {
        if (!dlopen(argv[1], RTLD_NOW | RTLD_LOCAL)) return 2;
        if (argc > 2 && std::string(argv[2]) == "delete" && unlink(argv[1])) return 3;
        if (argc > 3 && std::string(argv[2]) == "replace" && std::rename(argv[3], argv[1])) return 3;
    }
#else
    (void) argc; (void) argv;
#endif
    efficient_sam2::RuntimeIdentity result{"stale", "stale"};
    std::string error;
    if (!efficient_sam2::runtime_identity(result, error)) {
        if (!result.digest.empty() || !result.json.empty() || error.empty()) return 4;
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!error.empty() || result.digest.size() != 64 || result.json.empty()) return 5;
    std::printf("%s\n%s\n", result.digest.c_str(), result.json.c_str());
    return 0;
}
