"""Regression checks for the downloaded models' preprocessing contract."""
import json
from pathlib import Path

import numpy as np
import pytest

from depth_estimator import DepthEstimator
from tools.fetch_model_assets import fetch, sha256


def test_midas_input_is_rgb_unit_range_without_double_normalization(tmp_path):
    estimator = DepthEstimator(model_path=str(tmp_path / "absent.onnx"))
    image = np.full((40, 60, 3), (0, 127, 255), dtype=np.uint8)
    tensor = estimator._preprocess(image)
    assert tensor.shape == (1, 3, 256, 256) and tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 100, 100], [1., 127/255, 0.], atol=1e-6)


def test_constant_depth_output_does_not_become_negative(tmp_path):
    estimator = DepthEstimator(model_path=str(tmp_path / "absent.onnx"))
    depth = estimator._postprocess(np.full((1, 256, 256), 300., np.float32), 60, 40)
    assert depth.shape == (40, 60)
    assert np.isfinite(depth).all() and depth.min() >= 0


def test_download_checksum_failure_preserves_existing_file(tmp_path, monkeypatch):
    import io
    target = tmp_path / "model.onnx"
    target.write_bytes(b"original")
    class Response(io.BytesIO):
        headers = {"Content-Length": "8"}
    monkeypatch.setattr("tools.fetch_model_assets.urlopen", lambda *a, **kw: Response(b"changed!"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch("https://example.com/model", target, expected_sha256="0" * 64, minimum_size=1)
    assert target.read_bytes() == b"original"
    assert not target.with_suffix(".onnx.part").exists()


def test_model_manifest_matches_included_files():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "models" / "manifest.json").read_text())
    for asset in manifest["models"]:
        path = root / "models" / asset["filename"]
        assert path.stat().st_size == asset["size_bytes"]
        assert sha256(path) == asset["sha256"]
