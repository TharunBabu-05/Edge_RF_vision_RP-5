"""Guided survey planner for deterministic room-sampling workflows.

This module defines a simple grid-based survey over normalized room coordinates.
It tracks capture progress and stores per-point measurements for later export.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SurveyStatus:
    """Snapshot of current guided survey state."""

    active: bool
    rows: int
    cols: int
    total_points: int
    current_index: int
    completed_points: int
    remaining_points: int
    completed: bool
    current_target_x: float
    current_target_y: float
    started_epoch: float
    last_capture_epoch: float


class GuidedSurvey:
    """Grid-based guided survey manager with capture history."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._points: List[Tuple[float, float]] = []
        self._captures: List[Dict[str, float | int]] = []
        self._rows = 0
        self._cols = 0
        self._active = False
        self._cursor = 0
        self._started_epoch = 0.0
        self._last_capture_epoch = 0.0

    def start(self, rows: int, cols: int, margin: float = 0.08, serpentine: bool = True) -> SurveyStatus:
        """Start a new survey and regenerate all target points."""
        with self._lock:
            self._rows = max(1, min(50, int(rows)))
            self._cols = max(1, min(50, int(cols)))
            safe_margin = max(0.0, min(0.45, float(margin)))
            self._points = self._build_points(self._rows, self._cols, safe_margin, bool(serpentine))
            self._captures = []
            self._cursor = 0
            self._active = len(self._points) > 0
            self._started_epoch = time.time()
            self._last_capture_epoch = 0.0
            return self._status_locked()

    def capture(self, intensity: float, dbm_value: float, network_count: int) -> SurveyStatus:
        """Record one measurement at the current target and advance."""
        with self._lock:
            if not self._active or self._cursor >= len(self._points):
                return self._status_locked()

            x, y = self._points[self._cursor]
            now = time.time()
            self._captures.append(
                {
                    "target_index": self._cursor + 1,
                    "x": float(x),
                    "y": float(y),
                    "intensity": float(intensity),
                    "dbm": float(dbm_value),
                    "network_count": int(network_count),
                    "timestamp": now,
                }
            )

            self._last_capture_epoch = now
            self._cursor += 1
            if self._cursor >= len(self._points):
                self._active = False

            return self._status_locked()

    def next_target(self) -> SurveyStatus:
        """Advance to the next target without recording a sample."""
        with self._lock:
            if self._active and self._cursor < len(self._points):
                self._cursor += 1
                if self._cursor >= len(self._points):
                    self._active = False
            return self._status_locked()

    def reset(self) -> SurveyStatus:
        """Stop and clear all survey state."""
        with self._lock:
            self._points = []
            self._captures = []
            self._rows = 0
            self._cols = 0
            self._active = False
            self._cursor = 0
            self._started_epoch = 0.0
            self._last_capture_epoch = 0.0
            return self._status_locked()

    def get_status(self) -> SurveyStatus:
        """Return current status as an immutable snapshot."""
        with self._lock:
            return self._status_locked()

    def get_points(self) -> List[Dict[str, float | int]]:
        """Return all planned target points."""
        with self._lock:
            return [
                {
                    "target_index": idx + 1,
                    "x": float(x),
                    "y": float(y),
                }
                for idx, (x, y) in enumerate(self._points)
            ]

    def get_captures(self) -> List[Dict[str, float | int]]:
        """Return recorded captures."""
        with self._lock:
            return [dict(item) for item in self._captures]

    def _status_locked(self) -> SurveyStatus:
        total = len(self._points)
        completed_points = len(self._captures)
        completed = total > 0 and completed_points >= total
        remaining = max(0, total - completed_points)

        if self._active and self._cursor < total:
            target_x, target_y = self._points[self._cursor]
            current_index = self._cursor + 1
        else:
            target_x, target_y = (0.0, 0.0)
            current_index = total if total > 0 else 0

        return SurveyStatus(
            active=bool(self._active),
            rows=int(self._rows),
            cols=int(self._cols),
            total_points=int(total),
            current_index=int(current_index),
            completed_points=int(completed_points),
            remaining_points=int(remaining),
            completed=bool(completed),
            current_target_x=float(target_x),
            current_target_y=float(target_y),
            started_epoch=float(self._started_epoch),
            last_capture_epoch=float(self._last_capture_epoch),
        )

    @staticmethod
    def _build_points(rows: int, cols: int, margin: float, serpentine: bool) -> List[Tuple[float, float]]:
        """Build row-major (optionally serpentine) normalized sampling points."""
        if rows <= 0 or cols <= 0:
            return []

        if cols == 1:
            x_values = [0.5]
        else:
            span_x = 1.0 - 2.0 * margin
            x_values = [margin + (span_x * i / float(cols - 1)) for i in range(cols)]

        if rows == 1:
            y_values = [0.5]
        else:
            span_y = 1.0 - 2.0 * margin
            y_values = [margin + (span_y * j / float(rows - 1)) for j in range(rows)]

        points: List[Tuple[float, float]] = []
        for row_idx, y_val in enumerate(y_values):
            row_x_values = list(x_values)
            if serpentine and (row_idx % 2 == 1):
                row_x_values.reverse()
            for x_val in row_x_values:
                points.append((float(x_val), float(y_val)))

        return points
