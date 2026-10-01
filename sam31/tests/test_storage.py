"""Checkpoint-independent unit tests. No test here proves SAM inference works."""
from __future__ import annotations

import ctypes
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_test_fixture import fixture_tensors, write_fixture
from sam31_gguf.format import (FormatError, GGUFFile, SafeTensorFile, compare_stores,
                              convert, decode_json, read_string, read_value, unpack)
from sam31_gguf.model import build_reference_model, match_model_keys
from sam31_gguf.quant import (inspect_quantization, inverse_weight_rotation,
                             regular_hadamard, restored_tensor, rotate_last_axis)


def independent_hadamard(size: int) -> np.ndarray:
    """Digit-sign definition independent of the production radix-four code."""
    matrix = np.empty((size, size), dtype=np.float64)
    for row in range(size):
        for col in range(size):
            r, c, sign = row, col, 1
            while r or c:
                if r % 4 + c % 4 == 3:
                    sign = -sign
                r //= 4
                c //= 4
            matrix[row, col] = sign / np.sqrt(size)
    return matrix


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "TEST_ONLY.safetensors"
        self.output = self.directory / "TEST_ONLY.gguf"
        write_fixture(self.source)

    def converted(self) -> GGUFFile:
        convert(self.source, self.output, allow_test_fixture=True)
        return GGUFFile(self.output)

    def test_exact_payload_and_logical_shapes(self) -> None:
        gguf = self.converted()
        original = SafeTensorFile(self.source)
        self.assertTrue(compare_stores(original, gguf)["exact"])
        self.assertEqual(len(gguf.tensors), 22)
        self.assertTrue(gguf.is_fixture)
        for name, (dtype, value) in fixture_tensors().items():
            with self.subTest(name=name):
                self.assertEqual(gguf.tensors[name].dtype, dtype)
                self.assertEqual(gguf.tensors[name].shape, value.shape)
                self.assertEqual(gguf.raw(name), value.tobytes())
                np.testing.assert_array_equal(gguf.array(name), value)
                self.assertLessEqual(len(gguf.tensors[name].storage_name), 63)
                self.assertEqual(gguf.tensors[name].offset % 32, 0)
        np.testing.assert_array_equal(gguf.float_array("test.bfloat16"), [1., 2., -1.])

    def test_official_ggml_cpp_reader(self) -> None:
        gguf = self.converted()
        binary = os.environ["SAM31_BINARY"]
        completed = subprocess.run([binary, "inspect", str(self.output), "--allow-test-fixture", "--read-all"],
                                   text=True, capture_output=True, check=True)
        report = json.loads(completed.stdout)
        self.assertTrue(report["test_fixture"])
        self.assertTrue(report["all_payloads_read"])
        self.assertTrue(report["native_backend_available"])
        self.assertFalse(report["native_model_validated"])
        self.assertEqual(report["tensor_count"], len(gguf.tensors))
        self.assertEqual(report["payload_bytes"], sum(item.size for item in gguf.tensors.values()))
        self.assertEqual(report["data_offset"], gguf.data_offset)

    def test_native_reader_rejects_fixture_by_default(self) -> None:
        self.converted()
        result = subprocess.run([os.environ["SAM31_BINARY"], "inspect", str(self.output)], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"synthetic test weights", result.stderr.lower())

    def test_fixture_cannot_be_converted_without_explicit_opt_in(self) -> None:
        with self.assertRaisesRegex(FormatError, "Synthetic weights"):
            convert(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_image_video_backend_rejects_fixture_before_import(self) -> None:
        self.converted()
        with self.assertRaisesRegex(FormatError, "Synthetic test weights"):
            build_reference_model(self.output, self.directory / "nonexistent-official-source")
        with self.assertRaisesRegex(FormatError, "Synthetic test weights"):
            build_reference_model(self.source, self.directory, safetensors_oracle=True)

    def test_no_overwrite_or_input_clobber(self) -> None:
        source_before = self.source.read_bytes()
        self.converted()
        old = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            convert(self.source, self.output, allow_test_fixture=True)
        with self.assertRaises(FormatError):
            convert(self.source, self.source, allow_test_fixture=True, overwrite=True)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(self.output.read_bytes(), old)
        convert(self.source, self.output, allow_test_fixture=True, overwrite=True)
        self.assertEqual(self.output.read_bytes(), old)

    def test_quantization_restoration_matches_independent_matrices(self) -> None:
        gguf = self.converted()
        specs = inspect_quantization(gguf)
        self.assertEqual(len(specs), 4)
        for name, spec in specs.items():
            with self.subTest(name=name):
                q = gguf.array(name).astype(np.float64)
                scale = gguf.float_array(spec.scale_name)
                scale = scale.reshape((-1,) + (1,) * (q.ndim - 1)) if scale.size > 1 else scale.reshape(())
                expected = q * scale
                if spec.group_size:
                    shape = expected.shape
                    transposed = expected.transpose(0, 2, 3, 1) if expected.ndim == 4 else expected
                    rotated = (transposed.reshape(-1, spec.group_size) @ independent_hadamard(spec.group_size)).reshape(transposed.shape)
                    expected = rotated.transpose(0, 3, 1, 2) if expected.ndim == 4 else rotated
                    self.assertEqual(expected.shape, shape)
                np.testing.assert_allclose(restored_tensor(gguf, name, specs), expected, atol=2e-6, rtol=2e-6)

    def test_hadamard_groups_and_inverse(self) -> None:
        random = np.random.default_rng(13)
        for group in (4, 16, 64, 256):
            with self.subTest(group=group):
                basis = independent_hadamard(group)
                np.testing.assert_array_equal(regular_hadamard(group), basis)
                matrix = random.normal(size=(3, group * 2)).astype(np.float32)
                expected = (matrix.reshape(-1, group) @ basis).reshape(matrix.shape)
                actual = rotate_last_axis(matrix, group)
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
                np.testing.assert_allclose(rotate_last_axis(actual, group), matrix, atol=2e-6, rtol=2e-6)
        for bad in (0, 1, 2, 8, 12, True, 16.0):
            with self.subTest(bad_group=bad), self.assertRaises(ValueError):
                regular_hadamard(bad)
        with self.assertRaises(ValueError):
            inverse_weight_rotation(np.zeros((2, 16, 2)), 16)

    def test_bad_quantization_metadata_fails_closed(self) -> None:
        cases = ("missing_scale", "negative_scale", "nan_scale", "missing_descriptor",
                 "bad_group", "unknown_format", "unknown_semantics", "orphan", "scale_shape")
        for case in cases:
            with self.subTest(case=case):
                tensors = fixture_tensors()
                prefix = "test.linear16"
                if case == "missing_scale": del tensors[prefix + ".weight_scale"]
                if case in ("negative_scale", "nan_scale"):
                    tensors[prefix + ".weight_scale"] = ("F32", np.asarray(-1. if case == "negative_scale" else np.nan, np.float32))
                if case == "missing_descriptor": del tensors[prefix + ".comfy_quant"]
                if case in ("bad_group", "unknown_format", "unknown_semantics"):
                    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
                    if case == "bad_group": config["convrot_groupsize"] = 8
                    if case == "unknown_format": config["format"] = "unknown"
                    if case == "unknown_semantics": config["unknown"] = 1
                    tensors[prefix + ".comfy_quant"] = ("U8", np.frombuffer(json.dumps(config).encode(), np.uint8))
                if case == "orphan": tensors["orphan.comfy_quant"] = tensors[prefix + ".comfy_quant"]
                if case == "scale_shape": tensors[prefix + ".weight_scale"] = ("F32", np.ones((5, 2), np.float32))
                path = self.directory / f"bad-{case}.safetensors"
                write_fixture(path, tensors)
                with self.assertRaises(FormatError):
                    convert(path, self.directory / f"bad-{case}.gguf", allow_test_fixture=True)
                self.assertFalse((self.directory / f"bad-{case}.gguf").exists())

    def test_corrupted_safetensors(self) -> None:
        raw = self.source.read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + length])
        names = sorted(key for key in header if key != "__metadata__")
        for case in ("truncated", "unhashable_dtype", "overlap", "negative_shape", "huge_header", "duplicate_json"):
            with self.subTest(case=case):
                copied = json.loads(json.dumps(header))
                if case == "unhashable_dtype": copied[names[0]]["dtype"] = []
                if case == "overlap":
                    copied[names[1]]["data_offsets"] = copied[names[0]]["data_offsets"]
                if case == "negative_shape": copied[names[0]]["shape"] = [-1]
                encoded = json.dumps(copied).encode()
                malformed = struct.pack("<Q", len(encoded)) + encoded + raw[8 + length:]
                if case == "truncated": malformed = raw[:-1]
                if case == "huge_header": malformed = struct.pack("<Q", 2**60)
                if case == "duplicate_json": malformed = struct.pack("<Q", 13) + b'{"a":1,"a":2}'
                path = self.directory / f"corrupt-{case}"
                path.write_bytes(malformed)
                with self.assertRaises(FormatError): SafeTensorFile(path)

    def test_corrupted_gguf_and_full_payload_verification(self) -> None:
        valid = self.converted()
        data = self.output.read_bytes()
        for case, malformed in (("magic", b"BAD!" + data[4:]),
                                ("version", data[:4] + struct.pack("<I", 2) + data[8:]),
                                ("header-truncated", data[:40]),
                                ("payload-truncated", data[:valid.data_offset + 4]),
                                ("architecture", data.replace(b"sam3_1_multiplex", b"sam3_0_multiplex", 1))):
            with self.subTest(case=case):
                path = self.directory / f"bad-{case}.gguf"
                path.write_bytes(malformed)
                with self.assertRaises(FormatError): GGUFFile(path)
                result = subprocess.run([os.environ["SAM31_BINARY"], "inspect", str(path), "--allow-test-fixture"], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
        mutated = bytearray(data)
        item = valid.tensors["test.linear16.weight"]
        mutated[item.offset] ^= 0x01
        self.output.write_bytes(mutated)
        with self.assertRaisesRegex(FormatError, "payload differs"):
            compare_stores(SafeTensorFile(self.source), GGUFFile(self.output))

    def test_duplicate_json_and_nested_gguf_arrays(self) -> None:
        with self.assertRaises(FormatError): decode_json('{"x":1,"x":2}')
        with self.assertRaises(FormatError): decode_json('{"x":NaN}')
        with self.assertRaises(FormatError): read_value(io.BytesIO(struct.pack("<IQ", 9, 1)), 9)

    def test_identical_int8_aliases_with_different_scales_rejected(self) -> None:
        tensors = fixture_tensors()
        for suffix in (".weight", ".weight_scale", ".comfy_quant"):
            dtype, value = tensors["test.linear16" + suffix]
            tensors["tracker.model.test.linear16" + suffix] = (dtype, value.copy())
        dtype, scale = tensors["tracker.model.test.linear16.weight_scale"]
        tensors["tracker.model.test.linear16.weight_scale"] = (dtype, scale * 2)
        path = self.directory / "alias-conflict.safetensors"
        write_fixture(path, tensors)
        with self.assertRaisesRegex(FormatError, "Ambiguous quantization"):
            match_model_keys({"test.linear16.weight": np.empty((5, 32))}, SafeTensorFile(path))

    def test_model_key_mapping_is_strict(self) -> None:
        store = self.converted()
        mapping, derived = match_model_keys({"test.linear16.weight": np.empty((5, 32))}, store)
        self.assertEqual(mapping, {"test.linear16.weight": "test.linear16.weight"})
        self.assertFalse(derived)
        with self.assertRaisesRegex(FormatError, "lacks"):
            match_model_keys({"missing.weight": np.zeros((2, 3))}, store)
        with self.assertRaisesRegex(FormatError, "Shape mismatch"):
            match_model_keys({"test.linear16.weight": np.zeros((5, 31))}, store)


class NativeNumericalInteropTests(unittest.TestCase):
    """NumPy oracle vs C ABI; these are operator tests, NOT model inference."""
    def test_cpp_linear_dequant_against_independent_oracle(self) -> None:
        library = ctypes.CDLL(os.environ["SAM31_QUANT_LIBRARY"])
        i8pointer = ctypes.POINTER(ctypes.c_int8)
        fpointer = ctypes.POINTER(ctypes.c_float)
        size = ctypes.c_size_t
        function = library.sam31_dequantize_linear
        function.argtypes = [i8pointer, size, fpointer, size, size, size, size, fpointer, size]
        function.restype = ctypes.c_int
        random = np.random.default_rng(1031)
        for group in (0, 4, 16, 64, 256):
            for scale_count in (1, 3):
                with self.subTest(group=group, scale_count=scale_count):
                    cols = group * 2 if group else 7
                    q = random.integers(-127, 128, (3, cols), dtype=np.int8)
                    scale = random.uniform(0.001, 0.02, scale_count).astype(np.float32)
                    output = np.empty(q.shape, dtype=np.float32)
                    expected = q * scale.reshape(-1, 1)
                    if group:
                        expected = (expected.reshape(-1, group) @ independent_hadamard(group)).reshape(q.shape)
                    args = [q.ctypes.data_as(i8pointer), q.size, scale.ctypes.data_as(fpointer), scale.size,
                            q.shape[0], q.shape[1], group, output.ctypes.data_as(fpointer), output.size]
                    self.assertEqual(function(*args), 0)
                    np.testing.assert_allclose(output, expected, atol=2e-6, rtol=2e-6)
                    args[-1] -= 1
                    self.assertEqual(function(*args), 1)


if __name__ == "__main__":
    unittest.main()
