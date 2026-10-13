"""Synthetic H4 tensor-power fixtures, no checkpoint/model-derived bytes."""

from types import SimpleNamespace

import numpy as np
import pytest

from restore_policy import native_h4_artifact, restore_native_h4


@pytest.mark.parametrize("group", [1, 4, 16, 64, 256])
def test_h4_tensor_power_mathematics(group):
    # This is the documented regular H4 matrix, not a trained model fixture.
    h4 = (
        np.array(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=np.float32,
        )
        / 2
    )
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < group:
        matrix = np.kron(matrix, h4)
    q = np.random.default_rng(17).integers(-127, 128, (3, 2 * group), dtype=np.int8)
    scales = np.array([[0.005], [0.007], [0.031]], dtype=np.float32)
    original = q.copy()
    restored = restore_native_h4(q, scales, group)
    expected = ((q.astype(np.float32) * scales).reshape(3, 2, group) @ matrix).reshape(
        q.shape
    )
    np.testing.assert_allclose(restored, expected, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(q, original)


@pytest.mark.parametrize("group", [0, 2, 3, 5, True, 32, 256])
def test_invalid_group(group):
    with pytest.raises(ValueError, match="group|Group"):
        restore_native_h4(np.ones((2, 16), np.int8), np.ones((2, 1), np.float32), group)


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_scale(scale):
    with pytest.raises(ValueError, match="Scales"):
        restore_native_h4(
            np.ones((2, 16), np.int8), np.full((2, 1), scale, np.float32), 16
        )


def test_k112_is_restored_without_padding_or_truncation():
    q = (np.arange(224) % 255 - 127).astype(np.int8).reshape(2, 112)
    scales = np.array([[0.013], [0.023]], np.float32)
    actual = restore_native_h4(q, scales, 16)
    assert actual.shape == (2, 112) and actual.dtype == np.float32
    for row in range(2):
        for start in range(0, 112, 16):
            np.testing.assert_array_equal(
                actual[row : row + 1, start : start + 16],
                restore_native_h4(
                    q[row : row + 1, start : start + 16], scales[row : row + 1], 16
                ),
            )


def test_artifact_view_preserves_original_and_residuals():
    class Artifact:
        def state_dict(self):
            return {
                **self.residuals,
                **{
                    name + ".weight": record.dequantized_weight()
                    for name, record in self.quantized.items()
                },
            }

    artifact = Artifact()
    q = np.arange(16, dtype=np.int8).reshape(1, 16)
    record = SimpleNamespace(
        weight_int8=q, weight_scale=np.ones((1, 1), np.float32), group_size=16
    )
    artifact.quantized = {"linear": record}
    artifact.residuals = {"bias": np.array([3.0], np.float32)}
    restored = native_h4_artifact(artifact)
    assert restored is not artifact and artifact.quantized["linear"] is record
    assert restored.residuals is artifact.residuals
    state = restored.state_dict()
    np.testing.assert_array_equal(
        state["linear.weight"], restore_native_h4(q, record.weight_scale, 16)
    )
    np.testing.assert_array_equal(state["bias"], artifact.residuals["bias"])
