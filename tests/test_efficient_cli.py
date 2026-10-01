import os
import subprocess
from pathlib import Path

import gguf
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(
    os.environ.get("EFFICIENTSAM3_BINARY", ROOT / "build/examples/efficientsam3")
)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--threads", "0"),
        ("--threads", "16x"),
        ("--threshold", "nan"),
        ("--threshold", "0.5x"),
    ],
)
def test_bad_numeric_arguments(option, value):
    p = subprocess.run(
        [
            str(BINARY),
            "--model",
            "missing",
            "--image",
            "missing",
            "--text",
            "dog",
            option,
            value,
        ],
        capture_output=True,
        check=False,
    )
    assert p.returncode == 2


@pytest.mark.parametrize("kind", ["identity", "tensor_count", "truncated"])
def test_invalid_gguf_rejected(tmp_path, kind):
    model = tmp_path / "invalid.gguf"
    if kind == "truncated":
        model.write_bytes(b"GGUF\x03\x00\x00\x00")
    else:
        writer = gguf.GGUFWriter(
            model, "efficientsam3" if kind == "tensor_count" else "wrong"
        )
        writer.add_tensor("invalid", np.zeros((1,), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    p = subprocess.run(
        [str(BINARY), "--model", str(model), "--image", "missing", "--text", "dog"],
        capture_output=True,
        check=False,
    )
    assert p.returncode == 1, p.stderr.decode()
