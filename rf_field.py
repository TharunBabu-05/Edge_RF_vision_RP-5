"""RF field model for real-time WiFi intensity visualization.

This module maintains a low-resolution 2D field, accumulates incoming signal
intensity using a centered Gaussian spread, applies per-update decay, and
returns an upscaled full-resolution field for rendering.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from typing import Optional

import numpy as np

import config

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from scipy.ndimage import gaussian_filter

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    gaussian_filter = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

LOGGER = logging.getLogger("rf_field")


class RFFieldModel:
    """Accumulate and serve a decaying RF intensity field."""

    def __init__(
        self,
        frame_width: int = config.CAMERA_WIDTH,
        frame_height: int = config.CAMERA_HEIGHT,
        field_scale: int = config.FIELD_SCALE,
        sigma: float = config.GAUSSIAN_SIGMA,
        decay_factor: float = config.DECAY_FACTOR,
        use_scipy_kernel: bool = False,
    ) -> None:
        """Initialize RF field model.

        Args:
            frame_width: Output field width in pixels.
            frame_height: Output field height in pixels.
            field_scale: Downscale factor for low-resolution internal grid.
            sigma: Gaussian spread radius in output-pixel units.
            decay_factor: Multiplicative decay applied on each update.
            use_scipy_kernel: Use scipy-based kernel generation if available.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if field_scale <= 0:
            raise ValueError("field_scale must be positive")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.field_scale = int(field_scale)

        self.grid_width = max(1, self.frame_width // self.field_scale)
        self.grid_height = max(1, self.frame_height // self.field_scale)

        self.decay_factor = float(np.clip(decay_factor, 0.0, 1.0))
        self.sigma = float(max(0.1, sigma))
        self.sigma_lowres = float(max(0.5, self.sigma / float(self.field_scale)))

        self._center_x = self.grid_width // 2
        self._center_y = self.grid_height // 2

        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self._upscaled = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)

        self._use_scipy_kernel = bool(use_scipy_kernel and SCIPY_AVAILABLE)
        self._kernel = self._build_center_kernel()

        self._last_update_ms = 0.0

    def _build_center_kernel(self) -> np.ndarray:
        """Build a normalized center-weighted Gaussian kernel.

        Returns:
            A low-resolution 2D kernel with peak value 1.0 at center.
        """
        if self._use_scipy_kernel:
            impulse = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
            impulse[self._center_y, self._center_x] = 1.0
            kernel = gaussian_filter(  # type: ignore[misc]
                impulse,
                sigma=self.sigma_lowres,
                mode="nearest",
                truncate=3.0,
            ).astype(np.float32)
        else:
            y_coords = np.arange(self.grid_height, dtype=np.float32)[:, None]
            x_coords = np.arange(self.grid_width, dtype=np.float32)[None, :]
            dist_sq = (x_coords - float(self._center_x)) ** 2 + (y_coords - float(self._center_y)) ** 2
            kernel = np.exp(-dist_sq / (2.0 * (self.sigma_lowres ** 2))).astype(np.float32)

        peak = float(np.max(kernel))
        if peak > 0.0:
            kernel /= peak
        return kernel

    def update(self, intensity_value: float) -> None:
        """Update RF grid with a new normalized intensity sample.

        Update model:
            grid = decay_factor * grid + intensity * gaussian_kernel

        Args:
            intensity_value: New intensity in the range 0.0 to 1.0.
        """
        start = time.perf_counter()

        intensity = float(np.clip(float(intensity_value), 0.0, 1.0))

        np.multiply(self._grid, self.decay_factor, out=self._grid)
        if intensity > 0.0:
            self._grid += intensity * self._kernel

        np.clip(self._grid, 0.0, 1.0, out=self._grid)

        self._last_update_ms = (time.perf_counter() - start) * 1000.0

    def get_field(self) -> np.ndarray:
        """Return normalized full-resolution field as float32 array.

        Returns:
            2D float32 array shaped (frame_height, frame_width), range [0, 1].
        """
        if cv2 is not None:
            cv2.resize(
                self._grid,
                (self.frame_width, self.frame_height),
                dst=self._upscaled,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            y_repeat = int(math.ceil(self.frame_height / self.grid_height))
            x_repeat = int(math.ceil(self.frame_width / self.grid_width))
            repeated = np.repeat(np.repeat(self._grid, y_repeat, axis=0), x_repeat, axis=1)
            self._upscaled[:, :] = repeated[: self.frame_height, : self.frame_width]

        np.clip(self._upscaled, 0.0, 1.0, out=self._upscaled)
        return self._upscaled.copy()

    def get_lowres_field(self) -> np.ndarray:
        """Return a copy of the low-resolution internal field grid."""
        return self._grid.copy()

    def reset(self) -> None:
        """Clear field state to all zeros."""
        self._grid.fill(0.0)
        self._upscaled.fill(0.0)
        self._last_update_ms = 0.0

    def get_last_update_ms(self) -> float:
        """Return most recent update duration in milliseconds."""
        return float(self._last_update_ms)


def _configure_logging() -> None:
    """Configure simple console logging for standalone testing."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for RF field module test mode."""
    parser = argparse.ArgumentParser(description="RF field model self-test")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH, help="Output width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT, help="Output height")
    parser.add_argument("--scale", type=int, default=config.FIELD_SCALE, help="Internal field scale")
    parser.add_argument("--sigma", type=float, default=config.GAUSSIAN_SIGMA, help="Gaussian sigma")
    parser.add_argument("--decay", type=float, default=config.DECAY_FACTOR, help="Decay factor per update")
    parser.add_argument("--iterations", type=int, default=200, help="Number of update calls")
    parser.add_argument("--intensity", type=float, default=0.7, help="Base update intensity")
    parser.add_argument("--use-scipy", action="store_true", help="Use scipy gaussian kernel if available")
    return parser.parse_args()


def main() -> int:
    """Run a standalone performance and output-shape test."""
    _configure_logging()
    args = _parse_args()

    model = RFFieldModel(
        frame_width=args.width,
        frame_height=args.height,
        field_scale=args.scale,
        sigma=args.sigma,
        decay_factor=args.decay,
        use_scipy_kernel=args.use_scipy,
    )

    update_times: list[float] = []
    for idx in range(max(1, args.iterations)):
        # Small variation helps validate temporal accumulation and decay behavior.
        pulse = float(np.clip(args.intensity + 0.15 * math.sin(idx * 0.1), 0.0, 1.0))
        model.update(pulse)
        update_times.append(model.get_last_update_ms())

    field = model.get_field()

    avg_ms = float(np.mean(update_times))
    max_ms = float(np.max(update_times))
    min_ms = float(np.min(update_times))

    LOGGER.info("Low-res grid: %dx%d", model.grid_width, model.grid_height)
    LOGGER.info("Field shape: %s", field.shape)
    LOGGER.info("Field min/max: %.4f / %.4f", float(field.min()), float(field.max()))
    LOGGER.info("Update time ms (min/avg/max): %.3f / %.3f / %.3f", min_ms, avg_ms, max_ms)
    LOGGER.info("Target check (<10ms avg on Pi Zero 2W): %s", "PASS" if avg_ms < 10.0 else "CHECK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
