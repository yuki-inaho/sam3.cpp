"""End-to-end tests of native executables. Python is the test driver, not inference."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SOURCE / "python"), str(SOURCE / "tools")]
from sam31_gguf.format import SafeTensorFile, convert
from make_test_fixture import write_fixture

BIN = Path(os.environ["SAM31_BINARY"]).resolve()
MODEL = Path(__file__).resolve().parents[1] / "fixtures/TEST_ONLY_native_sam31.gguf"

class NativeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="sam31-native-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(os.environ, PATH="/NO_EXECUTABLES", SAM31_PYTHON="/NOT_PYTHON", SAM31_SCRIPT="/NOT_A_SCRIPT")
        self.run_cli("synthetic", "--output", self.root / "inputs", "--frames", "6")

    def run_cli(self, *arguments: object, ok: bool = True, executable: Path = BIN) -> subprocess.CompletedProcess[str]:
        result = subprocess.run([str(executable), *map(str, arguments)], env=self.environment,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0 if ok else 2, result.stdout + result.stderr)
        return result

    def image_args(self, output: str = "result") -> list[object]:
        return ["image", "--model", MODEL, "--allow-test-fixture", "--image", self.root / "inputs/image.png",
                "--point", "1:0.25:0.32", "--point", "2:0.73:0.68", "--output", self.root / output]

    def video_args(self, output: str = "result") -> list[object]:
        return ["video", "--model", MODEL, "--allow-test-fixture", "--frames", self.root / "inputs/frames",
                "--point", "1:0.25:0.32", "--point", "2:0.73:0.68", "--output", self.root / output]

    def verify_outputs(self, folder: Path, count: int) -> dict:
        report = json.loads((folder / "report.json").read_text())
        self.assertTrue(report["complete"])
        self.assertTrue(report["model"]["test_fixture"])
        self.assertFalse(report["python_inference"])
        self.assertEqual(len(report["frames"]), count)
        for index, frame in enumerate(report["frames"]):
            self.assertEqual(frame["frame_index"], index)
            self.assertEqual([obj["object_id"] for obj in frame["objects"]], ["1", "2"])
            logits = np.fromfile(folder / frame["logits_file"], dtype="<f4").reshape(2,64,80)
            self.assertTrue(np.isfinite(logits).all())
            for slot, obj in enumerate(frame["objects"]):
                with Image.open(folder / obj["mask"]) as mask:
                    np.testing.assert_array_equal(np.asarray(mask), (logits[slot] > 0).astype(np.uint8) * 255)
        self.assertEqual(len(list(folder.glob("*.png"))), count * 2)
        return report

    def checkpoint_variant(self, *, rename=False, embedding=False, sequential=False):
        source = SafeTensorFile(SOURCE / "fixtures/TEST_ONLY_native_sam31.safetensors")
        tensors = {}
        for name, info in source.tensors.items():
            target = name.replace("tracker.model.", "tracker.") if rename else name
            if sequential and "sam_mask_decoder.transformer." in target:
                target = target.replace(".mlp.lin1.", ".mlp.0.").replace(".mlp.lin2.", ".mlp.2.").replace(".norm_final_attn.", ".norm_final.")
            tensors[target] = (info.dtype, source.array(name))
        if embedding:
            name = "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight"
            value = tensors[name][1].astype(np.float32)
            scale = np.maximum(np.abs(value).max(axis=1, keepdims=True), 1e-6) / 127
            tensors[name] = ("I8", np.rint(value / scale).astype(np.int8))
            base = name[:-7]
            tensors[base + ".weight_scale"] = ("F32", scale.astype(np.float32))
            tensors[base + ".comfy_quant"] = ("U8", np.frombuffer(b'{"format":"int8_tensorwise"}', dtype=np.uint8))
        path = self.root / "variant.safetensors"
        write_fixture(path, tensors, source.metadata)
        gguf = self.root / "variant.gguf"
        convert(path, gguf, allow_test_fixture=True)
        args = self.image_args()
        args[args.index("--model") + 1] = gguf
        self.run_cli(*args)
        self.verify_outputs(self.root / "result", 1)

    def test_real_checkpoint_tracker_prefix(self):
        self.checkpoint_variant(rename=True)

    def test_sequential_decoder_checkpoint_names(self):
        self.checkpoint_variant(sequential=True)

    def test_int8_token_embedding(self):
        self.checkpoint_variant(embedding=True)

    def controlled_selection(self, stable):
        source = SafeTensorFile(SOURCE / "fixtures/TEST_ONLY_native_sam31.safetensors")
        tensors = {name: (info.dtype, source.array(name)) for name, info in source.tensors.items()}
        prefix = "tracker.model.interactive_sam_mask_decoder"
        # Independent controlled oracle: candidate 0 has stable large logits and
        # quality 1000; multi-mask candidates have quality 100, 10, 1.
        tensors[prefix + ".iou_prediction_head.layers.2.bias"] = ("F32", np.asarray([1000, 100, 10, 1], dtype=np.float32))
        weight = prefix + ".output_hypernetworks_mlps.0.layers.2.weight"
        dtype, value = tensors[weight]
        tensors[weight] = (dtype, np.zeros_like(value))
        bias = prefix + ".output_hypernetworks_mlps.0.layers.2.bias"
        dtype, value = tensors[bias]
        tensors[bias] = (dtype, np.full_like(value, 1e6 if stable else 0))
        path = self.root / "selection.safetensors"
        write_fixture(path, tensors, source.metadata)
        gguf = self.root / "selection.gguf"
        convert(path, gguf, allow_test_fixture=True)
        args = self.image_args()
        args[args.index("--model") + 1] = gguf
        self.run_cli(*args, "--point", "1:0.05:0.05:0")
        report = self.verify_outputs(self.root / "result", 1)
        if stable:
            self.assertGreater(report["frames"][0]["objects"][0]["quality"], 900)
        else:
            self.assertLess(report["frames"][0]["objects"][0]["quality"], 200)
        self.assertLess(report["frames"][0]["objects"][1]["quality"], 200)

    def test_multiple_points_select_stable_single_mask(self):
        self.controlled_selection(stable=True)

    def test_multiple_points_unstable_mask_falls_back(self):
        self.controlled_selection(stable=False)

    def test_dedicated_image_binary_without_python(self) -> None:
        self.run_cli(*self.image_args()[1:], executable=BIN.with_name("sam31_image"))
        self.verify_outputs(self.root / "result", 1)

    def test_dedicated_video_binary_without_python(self) -> None:
        self.run_cli(*self.video_args()[1:], executable=BIN.with_name("sam31_video"))
        report = self.verify_outputs(self.root / "result", 6)
        self.assertEqual(report["model"]["used_tensors"], 867)
        self.assertEqual(report["stats"]["memory_blocks"], 20)

    def test_memory_ablation_changes_output(self) -> None:
        self.run_cli(*self.video_args("normal"))
        self.run_cli(*self.video_args("ablated"), "--ablate-memory")
        a = np.fromfile(self.root / "normal/frame_000001.f32", dtype="<f4")
        b = np.fromfile(self.root / "ablated/frame_000001.f32", dtype="<f4")
        self.assertGreater(float(np.abs(a-b).mean()), 1e-7)

    def test_two_threads_match_one(self) -> None:
        self.run_cli(*self.video_args("one"), "--threads", "1")
        other = subprocess.run([str(BIN), *map(str,self.video_args("two")), "--threads", "2"],
                               env=self.environment, capture_output=True, text=True, timeout=90)
        if other.returncode == 2 and "no OpenMP" in other.stderr:
            self.skipTest("OpenMP is optional and disabled in this build")
        self.assertEqual(other.returncode, 0, other.stderr)
        for index in range(6):
            name = f"frame_{index:06}.f32"
            self.assertEqual((self.root / "one" / name).read_bytes(), (self.root / "two" / name).read_bytes())

    def test_natural_frame_sort(self) -> None:
        frames = self.root / "inputs/frames"
        for path in frames.iterdir(): path.unlink()
        for number in (10,2,1): shutil.copyfile(self.root / "inputs/image.png", frames / f"{number}.png")
        result = self.run_cli(*self.video_args())
        report = json.loads(result.stdout)
        self.assertEqual([Path(f["input"]).name for f in report["frames"]], ["1.png", "2.png", "10.png"])

    def test_invalid_points_rejected_without_output(self) -> None:
        args = self.image_args()
        for value in ("", "1:nan:.5", "1:inf:.5", "1:-.1:.5", "1:1.01:.5", "0:.5:.5", "1:.5:.5:2", "1:.5:.5:"):
            with self.subTest(point=value):
                test_args = args.copy(); test_args[test_args.index("--point")+1] = value
                self.run_cli(*test_args, ok=False)
                self.assertFalse((self.root / "result").exists())

    def test_fixture_requires_opt_in(self) -> None:
        args = self.image_args(); args.remove("--allow-test-fixture")
        self.run_cli(*args, ok=False)
        self.assertFalse((self.root / "result").exists())

    def test_existing_output_not_overwritten(self) -> None:
        self.run_cli(*self.image_args())
        before = (self.root / "result/report.json").read_bytes()
        self.run_cli(*self.image_args(), ok=False)
        self.assertEqual(before, (self.root / "result/report.json").read_bytes())

    def test_mid_video_error_leaves_no_partial_output(self) -> None:
        (self.root / "inputs/frames/000003.png").write_bytes(b"not a PNG")
        self.run_cli(*self.video_args(), ok=False)
        self.assertFalse((self.root / "result").exists())
        self.assertEqual(list(self.root.glob(".result.sam31-*")), [])

    def test_single_frame_video_rejected(self) -> None:
        for file in (self.root / "inputs/frames").iterdir():
            if file.stem != "000000": file.unlink()
        self.run_cli(*self.video_args(), ok=False)

    def test_model_payload_corruption_is_not_success(self) -> None:
        path = self.root / "truncated.gguf"
        path.write_bytes(MODEL.read_bytes()[:-20])
        args = self.image_args(); args[args.index("--model")+1] = path
        self.run_cli(*args, ok=False)
        self.assertFalse((self.root / "result").exists())

if __name__ == "__main__":
    unittest.main(verbosity=2)
