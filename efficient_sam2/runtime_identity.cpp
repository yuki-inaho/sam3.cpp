#include "runtime_identity.h"
#include "prefix_cache/digest.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <link.h>
#include <sys/auxv.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

namespace efficient_sam2 {
#if defined(__linux__)
namespace {
struct sam3_mapping {
    unsigned long long begin, end, inode;
    unsigned int device_major, device_minor;
};
struct sam3_runtime_file { std::string path, hash; };
struct sam3_runtime_scan {
    std::vector<sam3_mapping> mappings;
    std::vector<sam3_runtime_file> files;
    std::string error;
};

bool sam3_same_file(const struct stat & a, const struct stat & b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
        a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
        a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}

int sam3_runtime_object(dl_phdr_info * info, size_t, void * opaque) {
    auto & scan = *static_cast<sam3_runtime_scan *>(opaque);
    // Exclude only the actual kernel-provided synthetic vDSO, not a filename
    // that happens to resemble it. It has no regular on-disk ELF input.
    const auto vdso = getauxval(AT_SYSINFO_EHDR);
    if (vdso && info->dlpi_addr == vdso) return 0;
    const char * input = info->dlpi_name && info->dlpi_name[0]
        ? info->dlpi_name : "/proc/self/exe";
    // A relative loader name cannot safely be resolved after a cwd change.
    if (input[0] != '/') {
        scan.error = std::string("Relative loaded ELF path cannot be verified: ") + input;
        return 1;
    }
    char * resolved = realpath(input, nullptr);
    if (!resolved) {
        scan.error = std::string("Cannot resolve loaded ELF path: ") + input;
        return 1;
    }
    const std::string path(resolved);
    std::free(resolved);
    struct stat before{}, after{};
    if (stat(path.c_str(), &before) || !S_ISREG(before.st_mode)) {
        scan.error = "Loaded ELF is missing or not a regular file: " + path;
        return 1;
    }
    bool mapped = false;
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        const auto & header = info->dlpi_phdr[i];
        if (header.p_type != PT_LOAD || !header.p_filesz) continue;
        const auto address = info->dlpi_addr + header.p_vaddr;
        for (const auto & mapping : scan.mappings) {
            if (address >= mapping.begin && address < mapping.end &&
                mapping.inode == static_cast<unsigned long long>(before.st_ino) &&
                mapping.device_major == major(before.st_dev) &&
                mapping.device_minor == minor(before.st_dev)) mapped = true;
        }
        break;
    }
    if (!mapped) {
        scan.error = "Loaded ELF mapping does not match current file: " + path;
        return 1;
    }
    std::string hash_error;
    const std::string hash = prefix_cache::sha256_file(path, hash_error);
    if (hash.empty() || !hash_error.empty()) {
        scan.error = "Cannot hash loaded ELF " + path + ": " + hash_error;
        return 1;
    }
    if (stat(path.c_str(), &after) || !sam3_same_file(before, after)) {
        scan.error = "Loaded ELF changed during content hashing: " + path;
        return 1;
    }
    scan.files.push_back({path, hash});
    return 0;
}

std::string sam3_json_string(const std::string & value) {
    static const char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += char(c);
    }
    return result + '"';
}
}
#endif

bool runtime_identity(RuntimeIdentity & result, std::string & error) {
    result = {};
    error.clear();
#if defined(__linux__)
    sam3_runtime_scan scan;
    std::ifstream maps("/proc/self/maps");
    if (!maps) { error = "Cannot read loaded ELF mappings from /proc/self/maps"; return false; }
    std::string line;
    while (std::getline(maps, line)) {
        sam3_mapping mapping{};
        char permissions[5]{};
        unsigned long long offset = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu",
            &mapping.begin, &mapping.end, permissions, &offset,
            &mapping.device_major, &mapping.device_minor, &mapping.inode) == 7)
            scan.mappings.push_back(mapping);
    }
    if (!maps.eof() || scan.mappings.empty()) {
        error = "Cannot parse loaded ELF mappings"; return false;
    }
    if (dl_iterate_phdr(sam3_runtime_object, &scan) || scan.files.empty()) {
        error = scan.error.empty() ? "No file-backed loaded ELF identity" : scan.error;
        return false;
    }
    std::sort(scan.files.begin(), scan.files.end(),
        [](const sam3_runtime_file & a, const sam3_runtime_file & b) { return a.path < b.path; });
    std::string json = "{\"schema\":\"efficient-sam2-loaded-elf-v1\",\"platform\":\"linux\",\"objects\":[";
    for (size_t i = 0; i < scan.files.size(); ++i) {
        if (i) json += ',';
        json += "{\"path\":" + sam3_json_string(scan.files[i].path) +
            ",\"sha256\":" + sam3_json_string(scan.files[i].hash) + '}';
    }
    json += "]}";
    result.digest = prefix_cache::sha256_bytes(json.data(), json.size());
    result.json = std::move(json);
    return true;
#else
    error = "Loaded runtime content identity is unsupported on this platform";
    return false;
#endif
}
}
