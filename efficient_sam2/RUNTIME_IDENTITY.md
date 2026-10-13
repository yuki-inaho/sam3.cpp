# Loaded runtime identity

`runtime_identity.h/.cpp` provides one C++14 resolver for persistent prefix-cache
and resume provenance. It uses the existing licensed SHA256 wrapper; no new
cryptographic implementation or process-spawning `ldd` parser is introduced.

Linux enumerates the running executable and every currently loaded file-backed
ELF object through `dl_iterate_phdr`. This includes actual loaded ggml, JPEG,
BLAS/OpenMP/other compute libraries when present, plus the loader and C/C++
runtimes. There is no filename-based compute-library allowlist. The actual
kernel vDSO address is the only synthetic exclusion. Static code is covered by
the executable hash.

The result is `{digest,json}`. The JSON contains schema
`efficient-sam2-loaded-elf-v1`, platform `linux`, and sorted `objects` records
with resolved absolute `path` and lowercase SHA256. The digest hashes the exact
JSON bytes. Paths participate in identity, so relocating unchanged binaries can
conservatively invalidate the cache. JSON path characters encode individual
POSIX filename bytes (non-ASCII bytes use `\u00XX`); consumers needing native
paths must decode those characters back to bytes. No load addresses enter the
identity, so ASLR does not invalidate it.

The resolver checks the first file-backed PT_LOAD mapping's device/inode against
`stat`, hashes the resolved file, then verifies device, inode, size and
nanosecond modification/change timestamps are unchanged. Missing, deleted,
replaced, unreadable, nonregular or unresolved/relative loaded files fail closed.
An unreadable `/proc/self/maps` or unsupported platform also returns false with
an explicit error and empty result. It never invents a fallback identity.

Callers must disable persistent reuse/resume when identity is unavailable while
allowing inference and ordinary RAM-only work to continue. Call after all compute
backends/plugins have loaded and before admitting persistent cache entries.
This is a snapshot: the caller must refresh or invalidate it if libraries are
subsequently loaded or replaced. A lightweight preflight must load the same
compute libraries as inference or its identity may differ conservatively.

This is build-content provenance, not hostile-process memory attestation. It
does not hash relocated RAM, JIT code, GPU drivers/firmware, environment-selected
math modes, CPU ISA, thread count, model, policy or preprocessing configuration.
Those semantic inputs remain separate mandatory cache/resume key fields. A
concurrent in-place rewrite completed before collection or hostile ABA rewrite
is beyond this read-only file-identity check's threat model.

Link with the existing `efficient_sam2_prefix_cache` target and
`${CMAKE_DL_LIBS}` as needed. Non-Linux builds return an explicit unsupported
error rather than disabling model inference.

## Focused validation

`test_runtime_identity.cpp` and `test_runtime_identity.py` check native SHA256
against Python hashlib for every returned file and the aggregate, deterministic
identity across ASLR-enabled subprocesses, a deliberately dlopened disposable
DSO, and fail-closed deleted/replaced/relative-name DSOs. No model is loaded.

Official API references:
- https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html
- https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html
