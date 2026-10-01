"""Storage boundaries independent of the trained checkpoint."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ev_converter", ROOT / "convert_efficientsam3_to_gguf.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_vision_name_is_bounded():
    name = converter.rename(
        "backbone.vision_backbone.trunk.model.backbone.model."
        "stages.2.op_list.1.context_module.main.proj.norm.running_mean"
    )
    assert name == "ev.s.2.1.att.proj.norm.rm"


def test_common_pcs_mapping():
    assert (
        converter.rename("transformer.decoder.query_embed.weight")
        == "ddec.query_embed.weight"
    )


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        converter.convert_arrays(
            {
                "backbone.vision_backbone.trunk.model.foo.main.weight": np.ones(2),
                "backbone.vision_backbone.trunk.model.foo.weight": np.ones(2),
            }
        )


def test_scalar_is_storage_vector():
    values = converter.convert_arrays(
        {"backbone.language_backbone.encoder.test": np.array(1)}
    )
    assert values["et.encoder.test"].shape == (1,)


def test_bad_float_rejected():
    with pytest.raises(ValueError, match="finite"):
        converter.convert_arrays(
            {"geometry_encoder.norm.weight": np.array([float("inf")])}
        )
