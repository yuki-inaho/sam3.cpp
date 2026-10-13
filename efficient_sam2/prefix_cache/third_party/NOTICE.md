# Pinned source provenance

These files are copied **without modification** from official ggml-org/llama.cpp
commit `bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444`:

- `vendor/hash/sha256/sha256.c`
- `vendor/hash/sha256/sha256.h`
- `vendor/hash/sha256/LICENSE`
- `vendor/hash/rotate-bits/rotate-bits.h`
- `vendor/hash/rotate-bits/LICENSE.md`

Source: https://github.com/ggml-org/llama.cpp/tree/bfe65e1dfa8f1afed8a2158e83e5d2a3d6db5444/vendor/hash

SHA256: Igor Pavlov, 2010-06-11, public domain, based on public-domain code from
Wei Dai's Crypto++ library. See the preserved source header and LICENSE.
Rotate-bits: Copyright (c) 2021 William Casarin, MIT; complete license preserved.

`../digest.cpp` is only an adapter for byte, file and domain-separated RGB
identities. The checksum in `../cache.cpp` is separate CRC32 for accidental
local-file corruption detection and is not an authentication mechanism.
