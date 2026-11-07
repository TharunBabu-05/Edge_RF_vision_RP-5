"""Guided 4-shot scene capture manager.

Orchestrates the 4-direction (N / E / S / W) image capture workflow
used by SceneReconstructor.  Exposes a thread-safe state machine that
Flask can poll and drive via REST endpoints.

State machine
-------------
  IDLE  →  CAPTURING (per direction) → RECONSTRUCTING → READY
                                     ↘ FAILED

Usage (from Flask)::

    capture = SceneCaptureManager(camera_source)
    capture.start_session()
    capture.take_shot("N", bgr_frame)
    capture.take_shot("E", bgr_frame)
    ...
    capture.take_shot("W", bgr_frame)
    geometry = capture.reconstruct()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import config
from scene_reconstructor import SceneReconstructor, SceneGeometry

LOGGER = logging.getLogger("scene_capture")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CaptureState(str, Enum):
    IDLE           = "idle"
    CAPTURING      = "capturing"
    RECONSTRUCTING = "reconstructing"
    READY          = "ready"
    FAILED         = "failed"


@dataclass
class CaptureStatus:
    """Immutable status snapshot for API responses."""
    state: str
    shots_taken: List[str]
    shots_remaining: List[str]
    progress_pct: float           # 0-100
    error: Optional[str]
    has_geometry: bool
    room_width: float
    room_depth: float
    room_height: float
    furniture_count: int
    timestamp: float


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SceneCaptureManager:
    """Thread-safe 4-shot capture + reconstruction manager.

    Args:
        output_dir:   Directory where captured images are stored.
        geometry_dir: Directory where room_geometry.json is written.
        reconstructor: SceneReconstructor instance (created lazily if None).
    """

    REQUIRED_SHOTS = config.SCENE_SHOTS  # ["N", "E", "S", "W"]

    def __init__(
        self,
        output_dir: str = config.SCENE_CAPTURE_DIR,
        geometry_dir: str = config.SCENE_OUTPUT_DIR,
        reconstructor: Optional[SceneReconstructor] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._output_dir   = Path(output_dir)
        self._geometry_dir = Path(geometry_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._geometry_dir.mkdir(parents=True, exist_ok=True)

        self._state: CaptureState = CaptureState.IDLE
        self._shots: Dict[str, np.ndarray] = {}
        self._error: Optional[str] = None
        self._geometry: Optional[SceneGeometry] = None
        self._recon_thread: Optional[threading.Thread] = None
        self._recon_start: float = 0.0
        self._generation = 0

        self._reconstructor = reconstructor  # lazy init in background thread

        LOGGER.info("SceneCaptureManager ready (output=%s, geometry=%s)",
                    self._output_dir, self._geometry_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self) -> CaptureStatus:
        """Begin a new capture session, clearing any previous shots."""
        with self._lock:
            if self._state == CaptureState.RECONSTRUCTING:
                return self._status_locked()
            self._shots.clear()
            self._generation += 1
            self._error = None
            self._geometry = None
            self._state = CaptureState.CAPTURING
            LOGGER.info("Capture session started.")
        return self.get_status()

    def take_shot(self, direction: str, frame: np.ndarray) -> CaptureStatus:
        """Record one directional shot.

        Args:
            direction: One of "N", "E", "S", "W".
            frame:     BGR image from camera.

        Returns:
            Updated CaptureStatus.
        """
        direction = direction.upper().strip()
        if direction not in self.REQUIRED_SHOTS:
            with self._lock:
                self._error = f"Unknown direction '{direction}'"
            return self.get_status()

        with self._lock:
            if self._state not in (CaptureState.CAPTURING,):
                self._error = "Start a capture session before taking shots"
                return self._status_locked()

            # Persist to disk
            img_path = self._output_dir / f"shot_{direction}.jpg"
            if not cv2.imwrite(
                str(img_path), frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
            ):
                self._error = "Could not save camera shot; check disk space and permissions"
                return self._status_locked()
            self._shots[direction] = frame.copy()
            self._error = None

        return self.get_status()

    def reconstruct(
        self,
        walk_path: Optional[List] = None,
        rf_samples: Optional[List] = None,
    ) -> CaptureStatus:
        """Trigger background reconstruction from the captured shots.

        Reconstruction runs in a background thread.  Poll ``get_status()``
        to check for completion.  When state == READY, geometry is available
        via ``get_geometry()``.

        Args:
            walk_path:  VO trajectory to embed in scene.
            rf_samples: RF samples to embed.
        """
        with self._lock:
            if self._state == CaptureState.RECONSTRUCTING or (self._recon_thread and self._recon_thread.is_alive()):
                self._error = "Previous reconstruction is still finishing. Try again shortly."
                return self._status_locked()
            missing = [d for d in self.REQUIRED_SHOTS if d not in self._shots]
            if len(missing) == len(self.REQUIRED_SHOTS):
                self._error = "No shots captured yet."
                return self._status_locked()
            if missing:
                LOGGER.warning("Reconstructing with partial shots — missing: %s", missing)

            self._state = CaptureState.RECONSTRUCTING
            self._error = None
            self._geometry = None
            self._generation += 1
            generation = self._generation
            self._recon_start = time.time()
            shots_copy = {k: v.copy() for k, v in self._shots.items()}

        def _expire():
            with self._lock:
                if generation == self._generation and self._state == CaptureState.RECONSTRUCTING:
                    self._generation += 1
                    self._state = CaptureState.FAILED
                    self._error = "Reconstruction timed out; its result will be discarded."

        timer = threading.Timer(config.SCENE_RECONSTRUCTION_TIMEOUT, _expire)
        timer.daemon = True

        def _worker():
            try:
                if self._reconstructor is None:
                    self._reconstructor = SceneReconstructor()
                geom = self._reconstructor.reconstruct(
                    shots_copy,
                    walk_path=walk_path or [],
                    rf_samples=rf_samples or [],
                )
                out_path = self._geometry_dir / config.SCENE_GEOMETRY_FILE
                with self._lock:
                    if generation != self._generation:
                        return
                    temp_path = out_path.with_suffix(".tmp.json")
                    self._reconstructor.save(geom, str(temp_path))
                    temp_path.replace(out_path)
                    self._geometry = geom
                    self._state = CaptureState.READY
                    self._error = None
                    LOGGER.info("Reconstruction complete. Room %.1f×%.1f×%.1f m | %d furniture items.",
                                geom.room_width, geom.room_depth, geom.room_height,
                                len(geom.furniture))
            except Exception as exc:
                LOGGER.exception("Reconstruction failed: %s", exc)
                with self._lock:
                    if generation == self._generation:
                        self._error = str(exc)
                        self._state = CaptureState.FAILED
            finally:
                timer.cancel()

        with self._lock:
            if generation == self._generation:
                self._recon_thread = threading.Thread(target=_worker, daemon=True, name="ReconWorker")
                timer.start()
                self._recon_thread.start()
        return self.get_status()

    def cancel(self) -> CaptureStatus:
        """Cancel the current session and return to IDLE."""
        with self._lock:
            self._generation += 1
            self._geometry = None
            self._shots.clear()
            self._error = None
            self._state = CaptureState.IDLE
        return self.get_status()

    def get_status(self) -> CaptureStatus:
        """Return immutable status snapshot."""
        with self._lock:
            return self._status_locked()

    def get_geometry(self) -> Optional[SceneGeometry]:
        """Return the reconstructed geometry, or None if not ready."""
        with self._lock:
            return self._geometry

    def get_geometry_dict(self) -> Optional[Dict]:
        """Return the geometry as a plain dict (JSON-serialisable)."""
        with self._lock:
            return self._geometry.to_dict() if self._geometry else None

    def load_geometry_from_disk(self) -> bool:
        """Try to reload a previously saved geometry from disk."""
        import json
        path = self._geometry_dir / config.SCENE_GEOMETRY_FILE
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            from scene_reconstructor import SceneGeometry, WallPlane, FurnitureBox
            walls = [WallPlane(**w) for w in data.get("walls", [])]
            furniture = [FurnitureBox(**fb) for fb in data.get("furniture", [])]
            geom = SceneGeometry(
                room_width=float(data["room_width"]),
                room_depth=float(data["room_depth"]),
                room_height=float(data["room_height"]),
                walls=walls,
                furniture=furniture,
                walk_path=data.get("walk_path", []),
                rf_samples=data.get("rf_samples", []),
                capture_directions=data.get("capture_directions", []),
                timestamp=float(data.get("timestamp", 0.0)),
                point_cloud_size=int(data.get("point_cloud_size", 0)),
                quality=data.get("quality", {"metric_calibrated": False, "note": "Legacy reconstruction; calibration unknown."}),
            )
            with self._lock:
                self._geometry = geom
                self._state = CaptureState.READY
            LOGGER.info("Loaded geometry from disk: %s", path)
            return True
        except Exception as exc:
            LOGGER.warning("Could not load geometry from disk: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------

    def get_preview_jpeg(self, direction: str, max_size: int = 320) -> Optional[bytes]:
        """Return a JPEG thumbnail of a captured shot, or None."""
        with self._lock:
            frame = self._shots.get(direction.upper())
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = min(max_size / w, max_size / h)
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None

    def get_shot_directions(self) -> List[str]:
        """Return directions for which shots have been captured."""
        with self._lock:
            return list(self._shots.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _status_locked(self) -> CaptureStatus:
        """Build status snapshot — call only while holding self._lock."""
        taken     = list(self._shots.keys())
        remaining = [d for d in self.REQUIRED_SHOTS if d not in self._shots]
        n_total   = len(self.REQUIRED_SHOTS)
        n_taken   = len(taken)

        if self._state == CaptureState.RECONSTRUCTING:
            elapsed = time.time() - self._recon_start
            # Estimate 0-80% during reconstruction phase
            progress = min(80.0, elapsed / config.SCENE_RECONSTRUCTION_TIMEOUT * 80.0)
        elif self._state == CaptureState.READY:
            progress = 100.0
        else:
            progress = (n_taken / max(1, n_total)) * 60.0   # capture phase is 0-60%

        geom = self._geometry
        return CaptureStatus(
            state=self._state.value,
            shots_taken=taken,
            shots_remaining=remaining,
            progress_pct=round(progress, 1),
            error=self._error,
            has_geometry=geom is not None,
            room_width=float(geom.room_width)  if geom else 0.0,
            room_depth=float(geom.room_depth)  if geom else 0.0,
            room_height=float(geom.room_height) if geom else 0.0,
            furniture_count=len(geom.furniture) if geom else 0,
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    mgr = SceneCaptureManager()
    mgr.start_session()

    for direction in ["N", "E", "S", "W"]:
        synth = np.zeros((480, 640, 3), dtype=np.uint8)
        synth[:] = (30, 50 + hash(direction) % 80, 60)
        cv2.putText(synth, direction, (250, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (200, 200, 200), 4)
        mgr.take_shot(direction, synth)
        LOGGER.info("Status after %s: %s", direction, mgr.get_status().state)

    LOGGER.info("Triggering reconstruction …")
    mgr.reconstruct()

    # Wait for completion
    for _ in range(config.SCENE_RECONSTRUCTION_TIMEOUT):
        time.sleep(1)
        status = mgr.get_status()
        LOGGER.info("[%.0f%%] %s", status.progress_pct, status.state)
        if status.state in ("ready", "failed"):
            break

    LOGGER.info("Final status: %s", mgr.get_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
