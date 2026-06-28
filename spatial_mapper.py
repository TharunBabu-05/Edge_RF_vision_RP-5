"""Spatial WiFi mapper for room-position heatmap accumulation.

This module stores a persistent 2D heatmap where each sample is attached to a
user-provided room position in normalized coordinates [0, 1].
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class MapperStatus:
    """Snapshot of mapper status for API responses."""

    has_position: bool
    position_x: float
    position_y: float
    sample_count: int
    last_sample_epoch: float
    last_strength_dbm: float


class SpatialWifiMapper:
    """Maintain a persistent low-resolution WiFi strength heatmap."""

    def __init__(
        self,
        grid_width: int = 64,
        grid_height: int = 48,
        sigma: float = 2.4,
        decay: float = 0.995,
    ) -> None:
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError("grid size must be positive")

        self.grid_width = int(grid_width)
        self.grid_height = int(grid_height)
        self.sigma = float(max(0.5, sigma))
        self.decay = float(np.clip(decay, 0.0, 1.0))

        self._lock = threading.Lock()
        self._grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self._position: Optional[Tuple[float, float]] = None
        self._sample_count = 0
        self._last_sample_epoch = 0.0
        self._last_strength_dbm = -100.0

    def set_position(self, x: float, y: float) -> Tuple[float, float]:
        """Set active room position in normalized coordinate space."""
        nx = float(np.clip(float(x), 0.0, 1.0))
        ny = float(np.clip(float(y), 0.0, 1.0))
        with self._lock:
            self._position = (nx, ny)
        return nx, ny

    def get_position(self) -> Optional[Tuple[float, float]]:
        """Return active normalized room position if set."""
        with self._lock:
            return None if self._position is None else (self._position[0], self._position[1])

    def clear(self) -> None:
        """Reset heatmap and sampling counters."""
        with self._lock:
            self._grid.fill(0.0)
            self._sample_count = 0
            self._last_sample_epoch = 0.0
            self._last_strength_dbm = -100.0

    def add_sample(self, intensity: float, dbm_value: float) -> bool:
        """Add one sample at active position using Gaussian spread.

        Returns True when the sample was added, or False when no position is set.
        """
        value = float(np.clip(float(intensity), 0.0, 1.0))
        with self._lock:
            if self._position is None:
                return False

            center_x, center_y = self._normalized_to_grid(self._position[0], self._position[1])
            self._apply_sample_locked(center_x, center_y, value, float(dbm_value))

        return True

    def add_sample_at_position(self, x: float, y: float, intensity: float, dbm_value: float) -> Tuple[float, float]:
        """Set active position and apply one sample immediately.

        Returns:
            Normalized clamped (x, y) position used for the sample.
        """
        nx = float(np.clip(float(x), 0.0, 1.0))
        ny = float(np.clip(float(y), 0.0, 1.0))
        value = float(np.clip(float(intensity), 0.0, 1.0))

        with self._lock:
            self._position = (nx, ny)
            center_x, center_y = self._normalized_to_grid(nx, ny)
            self._apply_sample_locked(center_x, center_y, value, float(dbm_value))

        return nx, ny

    def _normalized_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """Convert normalized [0, 1] room position to fractional grid coordinates."""
        gx = float(np.clip(x, 0.0, 1.0)) * float(self.grid_width - 1)
        gy = float(np.clip(y, 0.0, 1.0)) * float(self.grid_height - 1)
        return gx, gy

    def _apply_sample_locked(self, center_x: float, center_y: float, intensity: float, dbm_value: float) -> None:
        """Apply one sample update while lock is already held."""
        np.multiply(self._grid, self.decay, out=self._grid)
        self._apply_gaussian_stamp(self._grid, center_x, center_y, intensity, self.sigma)
        np.clip(self._grid, 0.0, 1.0, out=self._grid)

        self._sample_count += 1
        self._last_sample_epoch = time.time()
        self._last_strength_dbm = float(dbm_value)

    def _apply_gaussian_stamp(
        self,
        grid: np.ndarray,
        center_x: float,
        center_y: float,
        amplitude: float,
        sigma: float,
    ) -> None:
        """Apply an additive 2D Gaussian centered at one grid cell."""
        radius = max(1, int(round(3.0 * sigma)))
        center_x_i = int(round(center_x))
        center_y_i = int(round(center_y))
        x_start = max(0, center_x_i - radius)
        x_end = min(self.grid_width - 1, center_x_i + radius)
        y_start = max(0, center_y_i - radius)
        y_end = min(self.grid_height - 1, center_y_i + radius)

        x_coords = np.arange(x_start, x_end + 1, dtype=np.float32)
        y_coords = np.arange(y_start, y_end + 1, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)

        dist_sq = (xx - float(center_x)) ** 2 + (yy - float(center_y)) ** 2
        kernel = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
        grid[y_start : y_end + 1, x_start : x_end + 1] += float(amplitude) * kernel

    def get_heatmap(self, width: int, height: int) -> np.ndarray:
        """Return resized float32 heatmap in [0, 1]."""
        if width <= 0 or height <= 0:
            raise ValueError("target size must be positive")

        with self._lock:
            source = self._grid.copy()

        resized = cv2.resize(source, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        np.clip(resized, 0.0, 1.0, out=resized)
        return resized

    def get_status(self) -> MapperStatus:
        """Return immutable mapper status snapshot."""
        with self._lock:
            has_position = self._position is not None
            pos_x = self._position[0] if self._position is not None else 0.0
            pos_y = self._position[1] if self._position is not None else 0.0
            return MapperStatus(
                has_position=has_position,
                position_x=float(pos_x),
                position_y=float(pos_y),
                sample_count=int(self._sample_count),
                last_sample_epoch=float(self._last_sample_epoch),
                last_strength_dbm=float(self._last_strength_dbm),
            )

    def save_to_file(self, file_path: str) -> str:
        """Persist mapper state to a compressed NumPy snapshot file."""
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            grid = self._grid.copy()
            if self._position is None:
                position = np.array([-1.0, -1.0], dtype=np.float32)
            else:
                position = np.array([self._position[0], self._position[1]], dtype=np.float32)
            sample_count = int(self._sample_count)
            last_sample_epoch = float(self._last_sample_epoch)
            last_strength_dbm = float(self._last_strength_dbm)

        np.savez_compressed(
            str(target),
            grid=grid,
            position=position,
            sample_count=np.int64(sample_count),
            last_sample_epoch=np.float64(last_sample_epoch),
            last_strength_dbm=np.float32(last_strength_dbm),
            grid_width=np.int32(self.grid_width),
            grid_height=np.int32(self.grid_height),
            sigma=np.float32(self.sigma),
            decay=np.float32(self.decay),
        )
        return str(target)

    def load_from_file(self, file_path: str) -> None:
        """Load mapper state from a previously saved snapshot file."""
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Snapshot not found: {source}")

        with np.load(str(source)) as data:
            if "grid" not in data:
                raise ValueError("Invalid snapshot: missing grid")
            loaded_grid = np.array(data["grid"], dtype=np.float32)

            expected_shape = (self.grid_height, self.grid_width)
            if loaded_grid.shape != expected_shape:
                raise ValueError(
                    f"Snapshot grid shape mismatch: expected {expected_shape}, got {loaded_grid.shape}"
                )

            loaded_position = np.array(data["position"], dtype=np.float32) if "position" in data else np.array([-1.0, -1.0], dtype=np.float32)
            loaded_sample_count = int(data["sample_count"]) if "sample_count" in data else 0
            loaded_last_epoch = float(data["last_sample_epoch"]) if "last_sample_epoch" in data else 0.0
            loaded_last_dbm = float(data["last_strength_dbm"]) if "last_strength_dbm" in data else -100.0

        loaded_pos_tuple: Optional[Tuple[float, float]] = None
        if loaded_position.size >= 2 and loaded_position[0] >= 0.0 and loaded_position[1] >= 0.0:
            loaded_pos_tuple = (
                float(np.clip(float(loaded_position[0]), 0.0, 1.0)),
                float(np.clip(float(loaded_position[1]), 0.0, 1.0)),
            )

        with self._lock:
            np.copyto(self._grid, loaded_grid)
            self._position = loaded_pos_tuple
            self._sample_count = max(0, loaded_sample_count)
            self._last_sample_epoch = max(0.0, loaded_last_epoch)
            self._last_strength_dbm = loaded_last_dbm
