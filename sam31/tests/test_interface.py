"""Image/video input validation and honest DoD status, without a SAM model."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam31_gguf.cli import doctor
from sam31_gguf.dod import run_gate
from sam31_gguf.format import FormatError
from sam31_gguf.runner import PointPrompt, input_frames, stage_frames, validate_prompts
from sam31_gguf.source import MARKER, OFFICIAL_COMMIT, python_hashes, verify_prepared_source, prepare_source
from sam31_gguf.synthetic import make_scene


class InterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_point_validation_and_slot_limit(self) -> None:
        self.assertEqual(PointPrompt.parse("7:0.25:0.75"), PointPrompt(7, .25, .75))
        validate_prompts([PointPrompt(i, .5, .5) for i in range(16)])
        for invalid in ("", "1:0", "1:nan:0", "1:inf:0", "-1:0:0", "1:1.01:0", "1:0:-.1", "2147483648:0:0"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): PointPrompt.parse(invalid)
        for invalid in ([], [PointPrompt(1, 0, 0)] * 2, [PointPrompt(i, 0, 0) for i in range(17)]):
            with self.assertRaises(ValueError): validate_prompts(invalid)

    def test_natural_frame_order(self) -> None:
        for name in ("10.jpg", "2.jpg", "1.jpg", "ignore.txt"):
            (self.root / name).touch()
        actual = input_frames(self.root, video=True)
        self.assertEqual([item.name for item in actual], ["1.jpg", "2.jpg", "10.jpg"])
        self.assertEqual(len(input_frames(self.root, video=True, max_frames=2)), 2)
        with self.assertRaises(ValueError): input_frames(self.root, video=True, max_frames=1)
        with self.assertRaises(ValueError): input_frames(self.root, video=True, max_frames=0)

    def test_empty_and_single_image_video_rejected(self) -> None:
        with self.assertRaises(ValueError): input_frames(self.root, video=True)
        (self.root / "1.jpg").touch()
        with self.assertRaises(ValueError): input_frames(self.root, video=True)
        self.assertEqual(len(input_frames(self.root / "1.jpg", video=False)), 1)
        with self.assertRaises(FileNotFoundError): input_frames(self.root / "unknown.txt", video=False)

    def test_staging_preserves_jpeg_bytes(self) -> None:
        first = self.root / "a.jpg"
        second = self.root / "b.png"
        Image.new("RGB", (32, 24), (200, 60, 10)).save(first)
        Image.new("RGBA", (32, 24), (30, 100, 60, 255)).save(second)
        staged = self.root / "stage"
        self.assertEqual(stage_frames([first, second], staged), (32, 24))
        self.assertEqual((staged / "000000.jpg").read_bytes(), first.read_bytes())
        with Image.open(staged / "000001.jpg") as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (32, 24))
        with self.assertRaises(FileExistsError): stage_frames([first], staged)

    def test_mismatched_frame_dimensions_rejected(self) -> None:
        a, b = self.root / "a.png", self.root / "b.png"
        Image.new("RGB", (16, 24)).save(a)
        Image.new("RGB", (24, 16)).save(b)
        with self.assertRaisesRegex(ValueError, "identical"):
            stage_frames([a, b], self.root / "stage")

    def test_synthetic_inputs_are_separate_from_ground_truth(self) -> None:
        output = self.root / "synthetic_inputs"
        scene = make_scene(output)
        self.assertEqual(scene["kind"], "synthetic-test-input-not-model-output")
        self.assertFalse(scene["ground_truth_used_as_model_input"])
        self.assertEqual(len(input_frames(output / "frames", video=True)), 6)
        self.assertEqual((output / "image.jpg").read_bytes(), (output / "frames/000000.jpg").read_bytes())
        with np.load(output / "ground_truth.npz", allow_pickle=False) as truth:
            self.assertEqual(truth["masks"].shape, (6, 2, 1, 384, 640))
            self.assertTrue(truth["masks"].any())
            self.assertFalse(truth["masks"].all())
            self.assertFalse(np.array_equal(truth["masks"][0], truth["masks"][-1]))
            for index, point in enumerate(scene["points"]):
                parsed = PointPrompt.parse(point)
                self.assertTrue(truth["masks"][0, index, 0, round(parsed.y * 384), round(parsed.x * 640)])
        self.assertFalse((output / "result.npz").exists())
        with self.assertRaises(FileExistsError): make_scene(output)

    def test_missing_model_never_passes_dod(self) -> None:
        report = self.root / "dod.json"
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_gate(checkpoint=self.root / "MISSING.safetensors", gguf=self.root / "MISSING.gguf",
                              source_root=self.root / "missing-source", binary=Path(os.environ["SAM31_BINARY"]),
                              work_dir=self.root / "uncreated-work", report_path=report)
        self.assertEqual(result, 2)
        actual = json.loads(report.read_text())
        self.assertEqual(actual["exit_code"], 2)
        self.assertFalse(actual["request_complete"])
        self.assertFalse(actual["reference_e2e_passed"])
        self.assertEqual(actual["native_sam31_graph_status"], "not_evaluated_by_reference_gate")
        self.assertEqual(actual["gates"][0]["status"], "BLOCKED")
        self.assertFalse((self.root / "uncreated-work").exists())

    def test_reference_only_does_not_hide_missing_model(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            code = run_gate(checkpoint=self.root / "missing", gguf=self.root / "missing.gguf",
                            source_root=self.root / "source", binary=Path(os.environ["SAM31_BINARY"]),
                            work_dir=self.root / "work", report_path=self.root / "report.json", reference_only=True)
        self.assertEqual(code, 2)
        self.assertFalse(json.loads((self.root / "report.json").read_text())["request_complete"])

    def test_unprepared_source_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError): verify_prepared_source(self.root)
        with self.assertRaises(ValueError): prepare_source(self.root, self.root / "nested")
        with self.assertRaises(FormatError): prepare_source(self.root, self.root.parent / (self.root.name + "-not-created"))

    def test_modified_prepared_source_rejected(self) -> None:
        # This temporary stub tests integrity checking only. It is never executed
        # or supplied to model inference and cannot meet the 900-key requirement.
        tracker = self.root / "sam3/model/video_tracking_multiplex.py"
        tracker.parent.mkdir(parents=True)
        tracker.write_text("# INERT UNIT TEST STUB; NOT A MODEL\n")
        marker = {"official_commit": OFFICIAL_COMMIT, "patch_schema": 1,
                  "python_sha256": python_hashes(self.root)}
        (self.root / MARKER).write_text(json.dumps(marker))
        verify_prepared_source(self.root)
        tracker.write_text("# changed\n")
        with self.assertRaisesRegex(FormatError, "changed"):
            verify_prepared_source(self.root)

    def test_native_cli_does_not_route_to_python(self) -> None:
        binary = os.environ["SAM31_BINARY"]
        for command in ("image", "video"):
            result = subprocess.run([binary, command, "--help"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--", result.stdout)
        env = dict(os.environ, SAM31_PYTHON=str(self.root / "missing-interpreter"))
        result = subprocess.run([binary, "image", "--help"], env=env, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"native C++", result.stdout)

    def test_reference_doctor_does_not_claim_native_validation(self) -> None:
        report = doctor(None, None)
        self.assertEqual(report["native_sam31_graph_status"], "not_evaluated_by_reference_gate")
        self.assertFalse(report["real_inference_prerequisites_ready"])
        self.assertTrue(report["storage_dependencies_ready"])


if __name__ == "__main__":
    unittest.main()
