"""Monocular depth estimator using MiDaS-Small (ONNX, CPU-only).

Adapted from the depth-estimation concepts in DeepPanoContext (ICCV 2021)
and holistic_scene_parsing (ECCV 2018), but re-implemented as a lightweight
ONNX inference wrapper that runs on Raspberry Pi 5 without a GPU.

Performance target (Raspberry Pi 5): ≤ 200 ms per 1280×960 image.

Usage::

    estimator = DepthEstimator()
    depth_map = estimator.estimate(bgr_image)   # returns float32 HxW array
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config

LOGGER = logging.getLogger("depth_estimator")

# ---------------------------------------------------------------------------
# MiDaS-Small ONNX: ~25 MB, runs in ~150-200 ms on Raspberry Pi 5 CPU
# ---------------------------------------------------------------------------
_MIDAS_ONNX_URL = (
    "https://huggingface.co/Intel/midas/resolve/main/midas_v21_small_256.onnx"
)
# Fallback if Hugging Face is also unreachable
_MIDAS_ONNX_URL_FALLBACK = (
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/"
    "midas_v21_small_256.onnx"
)


def _download_model(model_path: Path, url: str) -> bool:
    """Download ONNX model file with progress logging. Returns True on success."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading depth model from %s ...", url)
    LOGGER.info("Saving to: %s", model_path)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100.0)
            if block_num % 100 == 0:
                LOGGER.info("  %.1f %%  (%d / %d bytes)", pct, downloaded, total_size)

    try:
        urllib.request.urlretrieve(url, str(model_path), reporthook=_progress)
        LOGGER.info("Download complete: %s", model_path)
        return True
    except Exception as exc:
        LOGGER.warning("Download failed from %s: %s", url, exc)
        # Remove partial file
        if model_path.exists():
            model_path.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """Wrapper around MiDaS-Small ONNX for single-image monocular depth.

    The raw MiDaS output is a relative inverse depth map (arbitrary scale).
    We apply a configurable ``scale_factor`` to convert to approximate metres
    suitable for the 3D reconstruction pipeline.

    Args:
        model_path:   Path to the .onnx file.  Downloaded automatically if missing.
        input_size:   Square input resolution fed to MiDaS (default 256).
        scale_factor: Multiplier to convert relative depth → approximate metres.
    """

    def __init__(
        self,
        model_path: str = config.DEPTH_MODEL_PATH,
        input_size: int = config.DEPTH_INPUT_SIZE,
        scale_factor: float = config.DEPTH_SCALE_FACTOR,
    ) -> None:
        self.input_size = int(input_size)
        self.scale_factor = float(scale_factor)
        self._session = None  # lazy-loaded
        self._use_fallback = False  # True if model unavailable

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file
        if not model_file.exists():
            LOGGER.warning("Depth model not found at %s — trying to download...", model_file)
            # Try primary URL then fallback
            ok = config.AUTO_DOWNLOAD_MODELS and _download_model(model_file, _MIDAS_ONNX_URL)
            if not ok and config.AUTO_DOWNLOAD_MODELS:
                LOGGER.warning("Primary URL failed, trying fallback URL...")
                ok = _download_model(model_file, _MIDAS_ONNX_URL_FALLBACK)
            if not ok or not model_file.exists():
                LOGGER.warning(
                    "MiDaS model download failed. Reconstruction will use synthetic depth. "
                    "For best results, manually download midas_v21_small_256.onnx to: %s",
                    model_file
                )
                self._use_fallback = True

        self._model_path = str(model_file)
        LOGGER.info("DepthEstimator initialised (model=%s, fallback=%s)",
                    model_file.name, self._use_fallback)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, bgr_image: np.ndarray) -> np.ndarray:
        """Estimate a depth map for a single BGR image.

        Args:
            bgr_image: H×W×3 uint8 BGR image from OpenCV.

        Returns:
            float32 H×W array in approximate metres (same spatial size as input).
            Falls back to synthetic gradient depth if model is unavailable.
        """
        if self._use_fallback:
            if not config.ALLOW_SYNTHETIC_DEPTH:
                raise RuntimeError("Depth model unavailable. Install onnxruntime and a compatible MiDaS ONNX model at " + self._model_path + ". Synthetic reconstruction is available only in --demo mode.")
            return self._synthetic_depth(bgr_image)
        try:
            session = self._get_session()
        except Exception as exc:
            LOGGER.warning("ONNX session failed (%s) — using synthetic depth", exc)
            self._use_fallback = True
            if not config.ALLOW_SYNTHETIC_DEPTH:
                raise RuntimeError("Cannot load depth model; check ONNX Runtime and model compatibility") from exc
            return self._synthetic_depth(bgr_image)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        orig_h, orig_w = bgr_image.shape[:2]
        preprocessed = self._preprocess(bgr_image)

        t0 = time.perf_counter()
        try:
            raw = session.run([output_name], {input_name: preprocessed})[0]
            if not np.isfinite(raw).all():
                raise ValueError("Depth model returned non-finite values")
        except Exception as exc:
            raise RuntimeError("Depth inference failed; verify the ONNX model input and output shapes") from exc
        elapsed = (time.perf_counter() - t0) * 1000.0
        LOGGER.debug("MiDaS inference: %.1f ms", elapsed)

        depth = self._postprocess(raw, orig_w, orig_h)
        return depth

    def estimate_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Estimate depth maps for a list of BGR images (sequential)."""
        return [self.estimate(img) for img in images]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_session(self):
        """Lazy-load the ONNX Runtime session on first call."""
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for depth estimation. "
                    "Install with:  pip install onnxruntime"
                ) from exc

            # CPU-only provider is correct for Raspberry Pi 5
            providers = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4   # Pi 5 has 4 cores
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_path, sess_options=opts, providers=providers
            )
            LOGGER.info("ONNX Runtime session loaded for depth model.")
        return self._session

    def _preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """Resize, normalise, and add batch dimension for MiDaS."""
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0

        # MiDaS ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std

        # HWC → CHW → NCHW
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        return tensor

    def _postprocess(
        self, raw_output: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Convert raw MiDaS output to a metric depth map.

        MiDaS produces relative inverse-depth; we flip and scale to metres.
        """
        depth = raw_output.squeeze()  # (256, 256)

        # Normalise to [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)

        # Invert: MiDaS output is inverse-depth (larger = closer)
        depth = 1.0 - depth

        # Scale to approximate metres
        depth = depth * self.scale_factor

        # Resize back to original image dimensions
        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return depth_resized.astype(np.float32)

    def _synthetic_depth(self, bgr_image: np.ndarray) -> np.ndarray:
        """Fallback depth estimation without ONNX — uses simple CV heuristics.

        Produces a plausible depth map from edge gradients:
        - Areas with strong edges → likely nearby surfaces/objects
        - Smooth/homogeneous areas → likely walls/floors farther away
        - Centre-weighted radial gradient (rooms are wider in the centre)

        Good enough for basic room geometry extraction when MiDaS is unavailable.
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sobel gradient magnitude → edges = nearby objects
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx**2 + gy**2)
        edges = cv2.GaussianBlur(edges, (21, 21), 0)

        # Radial gradient: centre of image → farther, edges → closer (typical room view)
        cx, cy = w / 2, h / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        radial = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        radial = np.clip(radial, 0.0, 1.0)

        # Combine: normalise edges (0→1), invert radial (centre=far → high value)
        e_norm = edges / (edges.max() + 1e-6)
        depth = (1.0 - radial) * 0.6 + e_norm * 0.4  # blend

        # Normalise to 0.3–5.0 m range
        depth = 0.3 + depth * (self.scale_factor - 0.3)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="DepthEstimator self-test")
    parser.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    parser.add_argument("--output", type=str, default="depth_test.png", help="Output depth visualisation")
    args = parser.parse_args()

    estimator = DepthEstimator()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            LOGGER.error("Could not read: %s", args.image)
            return 1
    else:
        LOGGER.info("No image supplied — generating synthetic test frame.")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 240), 100, (255, 200, 100), -1)

    t0 = time.perf_counter()
    depth = estimator.estimate(img)
    elapsed = (time.perf_counter() - t0) * 1000.0

    LOGGER.info("Depth shape: %s | min=%.3f m | max=%.3f m | time=%.1f ms",
                depth.shape, float(depth.min()), float(depth.max()), elapsed)

    # Save coloured depth visualisation
    depth_uint8 = ((depth / depth.max()) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(args.output, depth_color)
    LOGGER.info("Depth visualisation saved: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
