import importlib.util
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("EFFICIENTSAM3_REAL") != "1", reason="Set EFFICIENTSAM3_REAL=1"
)
def test_native_ev_m_image_and_six_frame_sequence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "efficientsam3"))
    spec = importlib.util.spec_from_file_location(
        "efficient_e2e", root / "efficientsam3/e2e.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_e2e(
        os.environ.get("EFFICIENTSAM3_SOURCE", "outputs/efficientsam3/source"),
        os.environ.get("EFFICIENTSAM3_CHECKPOINT", "models/efficientsam3_ev_m.pt"),
        os.environ.get("EFFICIENTSAM3_GGUF", "models/efficientsam3_ev_m.gguf"),
        os.environ.get("EFFICIENTSAM3_BINARY", "build/examples/efficientsam3"),
        tmp_path,
    )
    assert report["passed"] and len(report["frames"]) == 6
