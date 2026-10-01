"""Bounded, streaming SafeTensors <-> GGUF v3 storage.

The GGUF contains actual typed tensors, not an embedded model archive. I8 values,
scales and Comfy quantization descriptors are preserved bit-for-bit. Long source
names are mapped to short stable names because ggml limits tensor names to 63
bytes. Logical dtype and shape are retained in metadata (including U8 and scalars).
No pickle, PyTorch, GGUF Python package, or network access is used here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Protocol

import numpy as np

ARCHITECTURE = "sam3_1_multiplex"
ALIGNMENT = 32
MAX_HEADER = 64 * 1024 * 1024
MAX_TENSORS = 100_000
MAX_KV = 16_384
CHUNK = 4 * 1024 * 1024
# SafeTensors type -> (on-disk numpy dtype, standard ggml type ID).
DTYPES: dict[str, tuple[str, int]] = {
    "F32": ("<f4", 0), "F16": ("<f2", 1), "I8": ("i1", 24),
    "U8": ("u1", 24), "BOOL": ("?", 24), "I16": ("<i2", 25),
    "I32": ("<i4", 26), "I64": ("<i8", 27), "F64": ("<f8", 28),
    "BF16": ("<u2", 30),
}
TYPE_BYTES = {0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2}


class FormatError(ValueError):
    """Untrusted input violates the documented storage contract."""


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormatError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def decode_json(text: str | bytes) -> Any:
    try:
        return json.loads(text, object_pairs_hook=unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              FormatError(f"Non-finite JSON value: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"Invalid JSON: {exc}") from exc


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def shape_tuple(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or len(raw) > 16:
        raise FormatError("Tensor shape must be a list with at most 16 axes")
    if any(type(n) is not int or n < 1 or n > (1 << 40) for n in raw):
        raise FormatError(f"Invalid or empty tensor dimensions: {raw!r}")
    if math.prod(raw) > (1 << 60):
        raise FormatError("Tensor element count is too large")
    return tuple(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_read(file: BinaryIO, size: int) -> bytes:
    if size < 0:
        raise FormatError("Negative read length")
    data = file.read(size)
    if len(data) != size:
        raise FormatError(f"Truncated file: requested {size}, received {len(data)} bytes")
    return data


def unpack(file: BinaryIO, fmt: str) -> Any:
    return struct.unpack("<" + fmt, exact_read(file, struct.calcsize("<" + fmt)))[0]


def read_string(file: BinaryIO) -> str:
    length = unpack(file, "Q")
    if length > MAX_HEADER:
        raise FormatError("GGUF string exceeds the 64 MiB safety limit")
    try:
        return exact_read(file, length).decode("utf-8")
    except UnicodeError as exc:
        raise FormatError("Invalid UTF-8 in GGUF") from exc


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise FormatError("GGUF string exceeds the 64 MiB safety limit")
    return struct.pack("<Q", len(encoded)) + encoded


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int  # Absolute byte offset in its file.
    size: int
    storage_name: str = ""


class TensorStore(Protocol):
    path: Path
    tensors: dict[str, TensorInfo]
    metadata: dict[str, Any]

    def array(self, name: str) -> np.ndarray: ...
    def raw(self, name: str) -> bytes: ...


class FileStore:
    path: Path
    tensors: dict[str, TensorInfo]
    metadata: dict[str, Any]

    def raw(self, name: str) -> bytes:
        info = self.tensors[name]
        with self.path.open("rb") as file:
            file.seek(info.offset)
            return exact_read(file, info.size)

    def array(self, name: str) -> np.ndarray:
        """An owned array: it remains valid after closing the file."""
        info = self.tensors[name]
        value = np.frombuffer(self.raw(name), dtype=np.dtype(DTYPES[info.dtype][0]))
        return value.reshape(info.shape).copy()

    def float_array(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        value = self.array(name)
        if info.dtype == "BF16":
            return (value.astype(np.uint32) << 16).view(np.float32)
        return value.astype(np.float32)


def check_intervals(tensors: Mapping[str, TensorInfo], size: int, *, contiguous: bool) -> None:
    end = None
    for item in sorted(tensors.values(), key=lambda t: t.offset):
        if item.offset < 0 or item.size < 1 or item.offset + item.size > size:
            raise FormatError(f"Tensor lies outside the file: {item.name}")
        if end is not None and (item.offset < end or (contiguous and item.offset != end)):
            raise FormatError(f"Overlapping tensors or forbidden gaps at {item.name}")
        end = item.offset + item.size
    if contiguous and end is not None and end != size:
        raise FormatError("Unindexed bytes at the end of SafeTensors payload")


class SafeTensorFile(FileStore):
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        size = self.path.stat().st_size
        self.tensors = {}
        with self.path.open("rb") as file:
            length = unpack(file, "Q")
            if not 2 <= length <= min(MAX_HEADER, size - 8):
                raise FormatError("Invalid SafeTensors header length")
            raw_header = exact_read(file, length)
        self.header = raw_header.decode("utf-8")
        header = decode_json(self.header)
        if not isinstance(header, dict):
            raise FormatError("SafeTensors header must be an object")
        self.metadata = header.get("__metadata__", {})
        if not isinstance(self.metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in self.metadata.items()
        ):
            raise FormatError("SafeTensors metadata must contain string pairs")
        self.data_offset = 8 + length
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if not name or "\x00" in name or len(name.encode("utf-8")) > 4096:
                raise FormatError("Invalid SafeTensors tensor name")
            if not isinstance(descriptor, dict):
                raise FormatError(f"Invalid tensor descriptor: {name}")
            dtype = descriptor.get("dtype")
            if not isinstance(dtype, str) or dtype not in DTYPES:
                raise FormatError(f"Unsupported SafeTensors dtype {dtype!r} at {name}")
            shape = shape_tuple(descriptor.get("shape"))
            offsets = descriptor.get("data_offsets")
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(n) is not int or n < 0 for n in offsets)):
                raise FormatError(f"Invalid data offsets at {name}")
            expected = math.prod(shape) * np.dtype(DTYPES[dtype][0]).itemsize
            if offsets[1] - offsets[0] != expected:
                raise FormatError(f"Byte count does not match dtype and shape at {name}")
            self.tensors[name] = TensorInfo(name, dtype, shape,
                                             self.data_offset + offsets[0], expected)
        if not 1 <= len(self.tensors) <= MAX_TENSORS:
            raise FormatError("Invalid number of tensors")
        if min(t.offset for t in self.tensors.values()) != self.data_offset:
            raise FormatError("Unindexed bytes before the first tensor")
        check_intervals(self.tensors, size, contiguous=True)


SCALAR_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                  6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_value(file: BinaryIO, kind: int, depth: int = 0) -> Any:
    if kind == 8:
        return read_string(file)
    if kind in SCALAR_FORMATS:
        value = unpack(file, SCALAR_FORMATS[kind])
        if isinstance(value, float) and not math.isfinite(value):
            raise FormatError("Non-finite GGUF metadata value")
        return value
    if kind == 9 and depth == 0:
        subtype, count = unpack(file, "I"), unpack(file, "Q")
        if subtype == 9 or count > MAX_TENSORS:
            raise FormatError("Nested or excessive GGUF arrays are unsupported")
        return [read_value(file, subtype, 1) for _ in range(count)]
    raise FormatError(f"Unsupported GGUF metadata type: {kind}")


class GGUFFile(FileStore):
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        size = self.path.stat().st_size
        self.metadata = {}
        self.tensors = {}
        descriptors: dict[str, tuple[tuple[int, ...], int, int]] = {}
        with self.path.open("rb") as file:
            if exact_read(file, 4) != b"GGUF" or unpack(file, "I") != 3:
                raise FormatError("Expected little-endian GGUF version 3")
            count, metadata_count = unpack(file, "Q"), unpack(file, "Q")
            if not 1 <= count <= MAX_TENSORS or not 1 <= metadata_count <= MAX_KV:
                raise FormatError("GGUF tensor or metadata count is outside safety limits")
            for _ in range(metadata_count):
                key = read_string(file)
                if not key or "\x00" in key or key in self.metadata:
                    raise FormatError("Invalid or duplicate GGUF metadata key")
                self.metadata[key] = read_value(file, unpack(file, "I"))
                if file.tell() > MAX_HEADER:
                    raise FormatError("GGUF header exceeds safety limit")
            if self.metadata.get("general.architecture") != ARCHITECTURE:
                raise FormatError("GGUF is not a SAM 3.1 multiplex storage file")
            if type(self.metadata.get("sam31.storage_version")) is not int or self.metadata["sam31.storage_version"] != 1:
                raise FormatError("Unsupported SAM 3.1 GGUF storage schema")
            alignment = self.metadata.get("general.alignment", ALIGNMENT)
            if type(alignment) is not int or alignment != ALIGNMENT:
                raise FormatError("SAM 3.1 storage schema requires 32-byte alignment")
            for _ in range(count):
                name, rank = read_string(file), unpack(file, "I")
                if (not 1 <= len(name.encode("utf-8")) <= 63 or "\x00" in name
                        or name in descriptors or not 1 <= rank <= 4):
                    raise FormatError("Invalid GGUF tensor name or rank")
                dims = tuple(unpack(file, "Q") for _ in range(rank))
                if any(n < 1 for n in dims) or math.prod(dims) > (1 << 60):
                    raise FormatError("Invalid GGUF tensor dimensions")
                kind, offset = unpack(file, "I"), unpack(file, "Q")
                if kind not in TYPE_BYTES or offset % alignment:
                    raise FormatError("Unsupported tensor encoding or unaligned offset")
                descriptors[name] = (tuple(reversed(dims)), kind, offset)
            if file.tell() > MAX_HEADER:
                raise FormatError("GGUF header exceeds safety limit")
            self.data_offset = align(file.tell(), alignment)
        source_hash = self.metadata.get("sam31.source.sha256")
        if (not isinstance(source_hash, str) or len(source_hash) != 64
                or any(character not in "0123456789abcdef" for character in source_hash)):
            raise FormatError("Missing or invalid source SHA-256 metadata")
        mapping_text = self.metadata.get("sam31.tensor_map")
        if not isinstance(mapping_text, str):
            raise FormatError("Missing reversible source tensor map")
        mapping = decode_json(mapping_text)
        if not isinstance(mapping, dict) or set(mapping) != set(descriptors):
            raise FormatError("Tensor map does not match GGUF tensor directory")
        for stored, (physical_shape, kind, offset) in descriptors.items():
            entry = mapping[stored]
            if not isinstance(entry, dict):
                raise FormatError("Invalid tensor mapping entry")
            name, dtype = entry.get("name"), entry.get("dtype")
            logical_shape = shape_tuple(entry.get("shape"))
            if (not isinstance(name, str) or not name or "\x00" in name or
                    name in self.tensors or not isinstance(dtype, str) or dtype not in DTYPES or DTYPES[dtype][1] != kind):
                raise FormatError("Invalid source tensor name or dtype mapping")
            expected_physical = storage_shape(logical_shape)
            if physical_shape != expected_physical:
                raise FormatError(f"Physical and logical tensor shapes disagree at {name}")
            byte_size = math.prod(physical_shape) * TYPE_BYTES[kind]
            self.tensors[name] = TensorInfo(name, dtype, logical_shape,
                                             self.data_offset + offset, byte_size, stored)
        check_intervals(self.tensors, size, contiguous=False)

    @property
    def is_fixture(self) -> bool:
        return self.metadata.get("sam31.test_fixture") is not False


def storage_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape if 1 <= len(shape) <= 4 else (math.prod(shape),)


def stored_name(name: str) -> str:
    # The full 256-bit digest would be too long for ggml's 64-byte name buffer.
    return "t_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:60]


def encode_metadata(key: str, value: Any) -> bytes:
    if isinstance(value, str):
        kind, payload = 8, pack_string(value)
    elif isinstance(value, bool):
        kind, payload = 7, struct.pack("<?", value)
    elif type(value) is int and 0 <= value <= 0xFFFFFFFF:
        kind, payload = 4, struct.pack("<I", value)
    else:
        raise FormatError(f"Unsupported output metadata value for {key}")
    return pack_string(key) + struct.pack("<I", kind) + payload


def convert(input_path: str | Path, output_path: str | Path, *,
            allow_test_fixture: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Write atomically. No dequantization or second lossy quantization occurs."""
    from .quant import inspect_quantization

    source = SafeTensorFile(input_path)
    before = source.path.stat()
    output = Path(output_path).resolve()
    if source.path == output:
        raise FormatError("Input and output must be different files")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    fixture = source.metadata.get("sam31.test_fixture") == "true"
    if fixture and not allow_test_fixture:
        raise FormatError("Synthetic weights require --allow-test-fixture")
    quantization = inspect_quantization(source)
    if not quantization and not fixture:
        raise FormatError("No supported INT8 layers: this is not the requested INT8 model")
    names = sorted(source.tensors)
    mapping = {
        stored_name(name): {"name": name, "dtype": source.tensors[name].dtype,
                            "shape": list(source.tensors[name].shape)}
        for name in names
    }
    if len(mapping) != len(names):
        raise FormatError("Hashed tensor-name collision")
    source_hash = sha256_file(source.path)
    metadata = {
        "general.architecture": ARCHITECTURE,
        "general.name": "SAM 3.1 Multiplex ConvRot INT8" if not fixture else "TEST ONLY: synthetic tensors",
        "general.alignment": ALIGNMENT,
        "sam31.storage_version": 1,
        "sam31.storage": "typed-tensors-lossless-convrot-int8",
        "sam31.test_fixture": fixture,
        "sam31.source.filename": source.path.name,
        "sam31.source.sha256": source_hash,
        "sam31.source.header": source.header,
        "sam31.source.metadata": json_text(source.metadata),
        "sam31.tensor_map": json_text(mapping),
        "sam31.convrot_basis": "regular-h4-kron",
        "sam31.int8_layers": len(quantization),
        "sam31.convrot_layers": sum(spec.group_size is not None for spec in quantization.values()),
    }
    header = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(names), len(metadata)))
    for key, value in metadata.items():
        header.extend(encode_metadata(key, value))
    offsets: dict[str, int] = {}
    payload_size = 0
    for name in names:
        info = source.tensors[name]
        shape = storage_shape(info.shape)
        offsets[name] = payload_size
        header.extend(pack_string(stored_name(name)))
        header.extend(struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *reversed(shape)))
        header.extend(struct.pack("<IQ", DTYPES[info.dtype][1], payload_size))
        payload_size = align(payload_size + info.size)
    if len(header) > MAX_HEADER:
        raise FormatError("GGUF header exceeds safety limit")
    data_offset = align(len(header))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target, source.path.open("rb") as origin:
            target.write(header)
            target.write(b"\0" * (data_offset - len(header)))
            for name in names:
                info = source.tensors[name]
                if target.tell() != data_offset + offsets[name]:
                    raise RuntimeError("Internal alignment invariant failed")
                origin.seek(info.offset)
                remaining = info.size
                while remaining:
                    block = exact_read(origin, min(remaining, CHUNK))
                    target.write(block)
                    remaining -= len(block)
                target.write(b"\0" * (align(info.size) - info.size))
            target.flush()
            os.fsync(target.fileno())
        after = source.path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise FormatError("Input file changed during conversion")
        check = GGUFFile(temporary)
        if len(check.tensors) != len(names):
            raise RuntimeError("Output tensor count changed")
        if overwrite:
            os.replace(temporary, output)
        else:
            # Atomic no-clobber publication, including concurrent converters.
            os.link(temporary, output)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "input": str(source.path), "output": str(output),
        "source_sha256": source_hash, "gguf_sha256": sha256_file(output),
        "tensor_count": len(names), "int8_layers": len(quantization),
        "convrot_layers": metadata["sam31.convrot_layers"],
        "bytes": output.stat().st_size, "test_fixture": fixture,
        "numerical_requantization": False,
    }


def compare_stores(original: TensorStore, converted: TensorStore) -> dict[str, Any]:
    """Independent full-payload comparison; an E2E gate must call this on real weights."""
    if set(original.tensors) != set(converted.tensors):
        raise FormatError("Source and GGUF tensor names differ")
    checked_bytes = 0
    with original.path.open("rb") as left, converted.path.open("rb") as right:
        for name in sorted(original.tensors):
            a, b = original.tensors[name], converted.tensors[name]
            if (a.dtype, a.shape, a.size) != (b.dtype, b.shape, b.size):
                raise FormatError(f"Tensor descriptor mismatch: {name}")
            left.seek(a.offset)
            right.seek(b.offset)
            remaining = a.size
            while remaining:
                size = min(remaining, CHUNK)
                if exact_read(left, size) != exact_read(right, size):
                    raise FormatError(f"Tensor payload differs: {name}")
                remaining -= size
            checked_bytes += a.size
    return {"exact": True, "tensors": len(original.tensors), "bytes": checked_bytes}
