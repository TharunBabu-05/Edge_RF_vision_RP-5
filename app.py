"""Flask MJPEG server for real-time WiFi signal visualization.

This module wires together camera capture, WiFi scanning, signal processing,
RF field accumulation, visualization overlays, and HTTP streaming endpoints.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import config
from guided_survey import GuidedSurvey
from heatmap_3d_renderer import Ekahau3DRenderer
from pose_estimator import PoseEstimate, PoseEstimator
from rf_database import RFDatabase
from rf_field import RFFieldModel
from rf_interpolator import RFInterpolator
from room_reconstructor import FloorPlan, FloorPlanReconstructor
from signal_processor import SignalSmoother, get_combined_intensity, get_dominant_network
from spatial_mapper import SpatialWifiMapper
from visualizer import MODE_COMBINED, render
from wifi_scanner import WifiScanner, start_scanner
from scene_capture import SceneCaptureManager
from imu_reader import IMUReader

LOGGER = logging.getLogger("app")

# Cached floor plan (rebuilt on demand)
_floor_plan_cache: Optional[FloorPlan] = None
_floor_plan_lock = threading.Lock()

# Scene capture singleton (one per server process)
_scene_capture: Optional[SceneCaptureManager] = None
_scene_capture_lock = threading.Lock()

# IMU reader singleton
_imu_reader: Optional[IMUReader] = None
_imu_reader_lock = threading.Lock()


def _get_imu_reader() -> IMUReader:
    """Return (or lazily create and start) global IMUReader."""
    global _imu_reader
    with _imu_reader_lock:
        if _imu_reader is None:
            _imu_reader = IMUReader()
            _imu_reader.start()
    return _imu_reader


def _get_scene_capture() -> SceneCaptureManager:
    """Return (or lazily create) the global SceneCaptureManager."""
    global _scene_capture
    with _scene_capture_lock:
        if _scene_capture is None:
            _scene_capture = SceneCaptureManager()
            # Try to load a previous reconstruction from disk
            _scene_capture.load_geometry_from_disk()
    return _scene_capture


class CameraSource:
    """Camera abstraction with OpenCV primary and ffmpeg fallback backends."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        """Initialize camera source settings.

        Args:
            index: Camera index, usually 0 for /dev/video0.
            width: Desired frame width.
            height: Desired frame height.
            fps: Desired frame rate.
        """
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg_process: Optional[subprocess.Popen[bytes]] = None
        self._ffmpeg_buffer = bytearray()
        self._backend_name = "uninitialized"

    def open(self) -> bool:
        """Open camera source and choose first working backend."""
        self._apply_v4l2_performance_controls()
        if self._try_open_opencv():
            return True
        if self._try_open_ffmpeg():
            return True
        return False

    def _apply_v4l2_performance_controls(self) -> None:
        """Apply best-effort V4L2 controls to reduce frame-rate drops.

        Many UVC webcams reduce FPS in low light when exposure_auto_priority=1.
        Setting it to 0 prefers stable frame rate at the requested FPS.
        """
        device_path = f"/dev/video{self.index}"
        commands = (
            (
                "v4l2-ctl",
                "--device",
                device_path,
                "--set-ctrl=exposure_auto_priority=0",
            ),
            (
                "v4l2-ctl",
                "--device",
                device_path,
                f"--set-parm={self.fps}",
            ),
        )

        for command in commands:
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

    def _try_open_opencv(self) -> bool:
        """Try opening camera with OpenCV V4L2 and default backends."""
        backends = []
        cap_v4l2 = getattr(cv2, "CAP_V4L2", None)
        if cap_v4l2 is not None:
            backends.append(("opencv-v4l2", cap_v4l2))
        backends.append(("opencv-default", None))

        preferred_formats = ("MJPG", None)

        for backend_name, backend_id in backends:
            for fourcc in preferred_formats:
                cap = cv2.VideoCapture(self.index) if backend_id is None else cv2.VideoCapture(self.index, backend_id)
                if not cap.isOpened():
                    cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
                cap.set(cv2.CAP_PROP_FPS, float(self.fps))
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if fourcc is not None:
                    cap.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*fourcc)))

                warmup_ok = False
                for _ in range(8):
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size > 0:
                        warmup_ok = True
                        break

                if warmup_ok:
                    self._cap = cap
                    format_tag = fourcc if fourcc is not None else "AUTO"
                    self._backend_name = f"{backend_name}-{format_tag.lower()}"
                    LOGGER.info("Camera backend active: %s", self._backend_name)
                    return True

                cap.release()

        return False

    def _try_open_ffmpeg(self) -> bool:
        """Try opening camera via ffmpeg MJPEG image pipe fallback."""
        device_path = f"/dev/video{self.index}"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-f",
            "v4l2",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            device_path,
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except FileNotFoundError:
            return False

        if process.stdout is None:
            process.terminate()
            return False

        os.set_blocking(process.stdout.fileno(), False)
        self._ffmpeg_process = process
        self._ffmpeg_buffer.clear()

        first_frame = self._read_ffmpeg_frame(timeout_seconds=2.0)
        if first_frame is None:
            self.release()
            return False

        self._backend_name = "ffmpeg-v4l2"
        LOGGER.info("Camera backend active: %s", self._backend_name)
        return True

    def _read_ffmpeg_frame(self, timeout_seconds: float = 0.6) -> Optional[np.ndarray]:
        """Read one JPEG frame from ffmpeg stdout pipe."""
        process = self._ffmpeg_process
        if process is None or process.stdout is None:
            return None

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                break

            chunk = process.stdout.read(8192)
            if not chunk:
                break
            self._ffmpeg_buffer.extend(chunk)

            start = self._ffmpeg_buffer.find(b"\xff\xd8")
            if start < 0:
                if len(self._ffmpeg_buffer) > 2_000_000:
                    del self._ffmpeg_buffer[:-1_000_000]
                continue

            end = self._ffmpeg_buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                continue

            jpeg_bytes = bytes(self._ffmpeg_buffer[start : end + 2])
            del self._ffmpeg_buffer[: end + 2]

            array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is not None and frame.size > 0:
                return frame

        return None

    def read(self) -> Optional[np.ndarray]:
        """Read one frame from active backend, returning None on failure."""
        if self._cap is not None:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame

        if self._ffmpeg_process is not None:
            return self._read_ffmpeg_frame(timeout_seconds=0.8)

        return None

    def backend_name(self) -> str:
        """Return current backend name for diagnostics and API status."""
        return self._backend_name

    def release(self) -> None:
        """Release all camera resources cleanly."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        if self._ffmpeg_process is not None:
            self._ffmpeg_process.terminate()
            try:
                self._ffmpeg_process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._ffmpeg_process.kill()
            self._ffmpeg_process = None

        self._ffmpeg_buffer.clear()


@dataclass
class RuntimeContext:
    """Shared runtime state for API routes and video generator."""

    scanner: WifiScanner
    camera: CameraSource
    field_model: RFFieldModel
    smoother: SignalSmoother
    mapper: SpatialWifiMapper
    pose_estimator: PoseEstimator
    rf_db: RFDatabase
    rf_interpolator: RFInterpolator
    survey: GuidedSurvey = field(default_factory=GuidedSurvey)
    mode: str = MODE_COMBINED
    frame_width: int = config.CAMERA_WIDTH
    frame_height: int = config.CAMERA_HEIGHT
    jpeg_quality: int = config.JPEG_QUALITY
    start_epoch: float = field(default_factory=time.time)

    fps: float = 0.0
    combined_intensity: float = 0.0
    smoothed_intensity: float = 0.0
    dominant_dbm: float = -90.0
    frame_count: int = 0
    last_frame_epoch: float = 0.0

    # Current VO pose snapshot (updated each processed frame)
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_z: float = 0.0
    pose_confidence: float = 0.0
    pose_status: str = "INITIALIZING"
    pose_tracked_points: int = 0

    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_metrics(self, fps: float, combined: float, smoothed: float, dominant_dbm: float) -> None:
        """Update runtime metrics in a thread-safe way."""
        with self.lock:
            self.fps = float(fps)
            self.combined_intensity = float(combined)
            self.smoothed_intensity = float(smoothed)
            self.dominant_dbm = float(dominant_dbm)
            self.frame_count += 1
            self.last_frame_epoch = time.time()

    def update_pose(self, pose: PoseEstimate) -> None:
        """Update VO pose snapshot in a thread-safe way."""
        with self.lock:
            self.pose_x = float(pose.x)
            self.pose_y = float(pose.y)
            self.pose_z = float(pose.z)
            self.pose_confidence = float(pose.confidence)
            self.pose_status = str(pose.status)
            self.pose_tracked_points = int(pose.tracked_points)


class StreamWorker:
    """Three-stage threaded pipeline: capture -> process -> encode."""

    def __init__(self, runtime: RuntimeContext, process_every_n_frames: int) -> None:
        """Create streaming worker and preallocate reusable frame buffers."""
        self.runtime = runtime
        self.process_every_n_frames = max(1, int(process_every_n_frames))

        self._stop_event = threading.Event()
        self._capture_event = threading.Event()
        self._encode_event = threading.Event()

        self._raw_lock = threading.Lock()
        self._latest_raw_frame: Optional[np.ndarray] = None

        self._jpeg_lock = threading.Lock()
        self._jpeg_condition = threading.Condition(self._jpeg_lock)
        self._latest_jpeg: bytes = b""
        self._jpeg_sequence = 0

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._encode_thread: Optional[threading.Thread] = None

        self._blank_frame = _make_blank_frame(
            self.runtime.frame_width,
            self.runtime.frame_height,
            "Camera frame unavailable",
        )
        self._rendered_buffer = self._blank_frame.copy()
        self._encode_buffer = self._blank_frame.copy()

        self._frame_index = 0
        self._last_combined = 0.0
        self._last_smoothed = 0.0
        self._last_dominant_dbm = -90.0
        self._last_mapper_scan_epoch = 0.0

        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            self._blank_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.runtime.jpeg_quality)],
        )
        if encoded_ok:
            self._latest_jpeg = encoded.tobytes()

    def start(self) -> None:
        """Start capture, processing, and encoding threads."""
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, name="capture-thread", daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, name="process-thread", daemon=True)
        self._encode_thread = threading.Thread(target=self._encode_loop, name="encode-thread", daemon=True)

        self._capture_thread.start()
        self._process_thread.start()
        self._encode_thread.start()

    def stop(self) -> None:
        """Stop all pipeline threads and wait briefly for shutdown."""
        self._stop_event.set()
        self._capture_event.set()
        self._encode_event.set()

        for thread in (self._capture_thread, self._process_thread, self._encode_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def get_latest_jpeg(self, last_sequence: int, timeout: float = 1.0) -> Tuple[int, bytes]:
        """Return newest JPEG payload and sequence number.

        Args:
            last_sequence: Last sequence seen by caller.
            timeout: Wait duration for a newer frame.
        """
        with self._jpeg_condition:
            if self._jpeg_sequence <= last_sequence:
                self._jpeg_condition.wait(timeout=timeout)
            return self._jpeg_sequence, self._latest_jpeg

    def _capture_loop(self) -> None:
        """Capture frames continuously from camera backend."""
        while not self._stop_event.is_set():
            frame = self.runtime.camera.read()
            if frame is None:
                frame = self._blank_frame

            with self._raw_lock:
                self._latest_raw_frame = frame

            # Also expose on runtime so scene shot API can grab a clean frame
            with self.runtime.lock:
                self.runtime.latest_frame = frame

            self._capture_event.set()

    def _process_loop(self) -> None:
        """Apply signal processing and visualization with frame skipping."""
        previous_epoch = time.perf_counter()
        fps_ema = 0.0

        while not self._stop_event.is_set():
            self._capture_event.wait(timeout=0.5)
            self._capture_event.clear()

            with self._raw_lock:
                frame = self._latest_raw_frame

            if frame is None:
                frame = self._blank_frame

            now = time.perf_counter()
            dt = max(1e-3, now - previous_epoch)
            previous_epoch = now

            fps_instant = 1.0 / dt
            fps_ema = fps_instant if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * fps_instant)

            self._frame_index += 1
            should_process = (
                self._frame_index == 1
                or (self._frame_index % self.process_every_n_frames == 0)
            )

            if should_process:
                networks = self.runtime.scanner.get_networks()
                combined_intensity = get_combined_intensity(networks)
                smoothed_intensity = self.runtime.smoother.update(combined_intensity)
                self.runtime.field_model.update(smoothed_intensity)
                rf_field = self.runtime.field_model.get_field()

                dominant = get_dominant_network(networks)
                dominant_dbm = float(dominant.get("dbm", -90.0)) if dominant else -90.0

                # --- Visual Odometry: update pose from camera frame ---
                pose = self.runtime.pose_estimator.update(frame)
                self.runtime.update_pose(pose)

                # --- RF Database: add georeferenced sample on new scan ---
                scan_epoch = self.runtime.scanner.get_last_scan_epoch()
                if scan_epoch > self._last_mapper_scan_epoch:
                    self._last_mapper_scan_epoch = scan_epoch
                    self.runtime.mapper.add_sample(combined_intensity, dominant_dbm)
                    # Add to 3D RF database using VO world position
                    if networks and pose.confidence > 0.1:
                        self.runtime.rf_db.add_sample(
                            x=pose.x, y=pose.y, z=pose.z,
                            networks=networks,
                        )

                rendered = render(frame, rf_field, networks, mode=self.runtime.mode)
                # Draw VO keypoints overlay when tracking
                if pose.status == "TRACKING":
                    kpts = self.runtime.pose_estimator.get_tracked_keypoints()
                    if kpts is not None:
                        for pt in kpts[:50]:  # draw up to 50 feature points
                            cx, cy = int(pt[0]), int(pt[1])
                            if 0 <= cx < rendered.shape[1] and 0 <= cy < rendered.shape[0]:
                                import cv2 as _cv2
                                _cv2.circle(rendered, (cx, cy), 3, (0, 255, 100), -1)
                _draw_runtime_overlay(rendered, fps=fps_ema, camera_backend=self.runtime.camera.backend_name())
                # Draw VO pose overlay
                _draw_pose_overlay(rendered, pose)
                np.copyto(self._rendered_buffer, rendered)

                self._last_combined = combined_intensity
                self._last_smoothed = smoothed_intensity
                self._last_dominant_dbm = dominant_dbm

            np.copyto(self._encode_buffer, self._rendered_buffer)
            _draw_runtime_overlay(self._encode_buffer, fps=fps_ema, camera_backend=self.runtime.camera.backend_name())

            self.runtime.update_metrics(
                fps=fps_ema,
                combined=self._last_combined,
                smoothed=self._last_smoothed,
                dominant_dbm=self._last_dominant_dbm,
            )
            self._encode_event.set()

    def _encode_loop(self) -> None:
        """JPEG-encode latest prepared frame and publish to stream clients."""
        while not self._stop_event.is_set():
            self._encode_event.wait(timeout=0.5)
            self._encode_event.clear()

            ok, encoded = cv2.imencode(
                ".jpg",
                self._encode_buffer,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.runtime.jpeg_quality)],
            )
            if not ok:
                continue

            payload = encoded.tobytes()
            with self._jpeg_condition:
                self._latest_jpeg = payload
                self._jpeg_sequence += 1
                self._jpeg_condition.notify_all()


def _configure_logging() -> None:
    """Configure root logging for the Flask app process."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _make_blank_frame(width: int, height: int, text: str) -> np.ndarray:
    """Create fallback frame used when camera frame cannot be read."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2, cv2.LINE_AA)
    return frame


def _draw_runtime_overlay(frame: np.ndarray, fps: float, camera_backend: str) -> None:
    """Draw FPS and backend status at top-right corner."""
    height, width = frame.shape[:2]
    text_1 = f"FPS: {fps:.1f}"
    text_2 = f"CAM: {camera_backend}"

    cv2.putText(frame, text_1, (max(10, width - 180), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text_2, (max(10, width - 250), 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def _draw_pose_overlay(frame: np.ndarray, pose: PoseEstimate) -> None:
    """Draw VO pose readout and tracking status at bottom-left corner."""
    height, width = frame.shape[:2]
    status_color = {
        "TRACKING": (0, 220, 80),
        "INITIALIZING": (220, 180, 0),
        "LOST": (0, 50, 220),
    }.get(pose.status, (180, 180, 180))

    lines = [
        (f"VO: {pose.status}", status_color),
        (f"X={pose.x:.2f} Y={pose.y:.2f} Z={pose.z:.2f}", (200, 200, 200)),
        (f"pts={pose.tracked_points} conf={pose.confidence:.2f}", (160, 160, 160)),
    ]
    for idx, (text, color) in enumerate(lines):
        y_pos = height - 12 - (len(lines) - 1 - idx) * 18
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _generate_mjpeg(stream_worker: StreamWorker) -> Generator[bytes, None, None]:
    """Yield multipart MJPEG frames from pre-encoded stream worker output."""
    last_sequence = -1

    while True:
        sequence, payload = stream_worker.get_latest_jpeg(last_sequence, timeout=1.0)
        if not payload:
            continue
        last_sequence = sequence

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        )


def _build_mapper_png(runtime: RuntimeContext) -> bytes:
    """Render mapper heatmap image for dashboard polling."""
    heat = runtime.mapper.get_heatmap(config.MAPPER_VIEW_WIDTH, config.MAPPER_VIEW_HEIGHT)
    heat_uint8 = (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    status = runtime.mapper.get_status()
    if status.has_position:
        h, w = color.shape[:2]
        px = int(round(status.position_x * (w - 1)))
        py = int(round(status.position_y * (h - 1)))
        cv2.circle(color, (px, py), 6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(color, (px, py), 2, (0, 0, 0), -1, cv2.LINE_AA)

    cv2.putText(
        color,
        f"Samples: {status.sample_count}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(".png", color)
    if ok:
        return encoded.tobytes()

    fallback = np.zeros((config.MAPPER_VIEW_HEIGHT, config.MAPPER_VIEW_WIDTH, 3), dtype=np.uint8)
    cv2.putText(
        fallback,
        "Map unavailable",
        (12, max(24, config.MAPPER_VIEW_HEIGHT // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 120, 255),
        2,
        cv2.LINE_AA,
    )
    ok_fallback, encoded_fallback = cv2.imencode(".png", fallback)
    return encoded_fallback.tobytes() if ok_fallback else b""


def _snapshot_dir() -> Path:
    """Return mapper snapshot directory, creating it if needed."""
    path = Path(config.MAPPER_SNAPSHOT_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_snapshot_name(name: str) -> str:
    """Normalize snapshot name to a filesystem-safe token."""
    raw = "" if name is None else str(name).strip()
    if not raw:
        raw = time.strftime("map_%Y%m%d_%H%M%S")

    chars = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")

    token = "".join(chars).strip("_")
    if not token:
        token = "map"
    return token[:64]


def _list_snapshot_names() -> list[str]:
    """List available snapshot names, newest first."""
    snapshots = sorted(_snapshot_dir().glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [item.stem for item in snapshots]


def _snapshot_path(name: str) -> Path:
    """Resolve snapshot file path from a user-provided name."""
    safe_name = _sanitize_snapshot_name(name)
    return _snapshot_dir() / f"{safe_name}.npz"


def _build_dashboard_html() -> str:
    """Return Ekahau-style 3D Indoor RF Digital Twin dashboard HTML."""
    return """<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>EdgeRF 3D — Indoor RF Digital Twin</title>
    <meta name=\"description\" content=\"3D Indoor WiFi Signal Mapping System — Ekahau-style RF Digital Twin on Raspberry Pi\" />
    <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />
    <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap\" rel=\"stylesheet\" />
    <style>
        :root {
            --bg:       #070d14;
            --surface:  #0d1825;
            --surface2: #111f2e;
            --border:   #1a2f44;
            --accent:   #00c2ff;
            --accent2:  #7c3aed;
            --good:     #22d38a;
            --warn:     #f59e0b;
            --danger:   #ef4444;
            --text:     #dde6f0;
            --muted:    #7a9bb5;
            --font:     'Inter', sans-serif;
            --mono:     'JetBrains Mono', monospace;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; }
        body {
            font-family: var(--font);
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            background-image:
                radial-gradient(ellipse 900px 400px at 10% 0%, #0a1e35 0%, transparent 60%),
                radial-gradient(ellipse 600px 300px at 90% 100%, #0e0a2a 0%, transparent 60%);
        }

        /* ── Header ─────────────────────────────────── */
        header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
            background: rgba(7,13,20,0.9);
            backdrop-filter: blur(12px);
            position: sticky; top: 0; z-index: 100;
        }
        .header-logo {
            display: flex; align-items: center; gap: 10px;
        }
        .logo-icon {
            width: 32px; height: 32px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px;
        }
        .logo-text { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
        .logo-text span { color: var(--accent); }
        .header-badges { display: flex; gap: 8px; align-items: center; }
        .badge {
            padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600;
            letter-spacing: 0.5px; text-transform: uppercase;
        }
        .badge-green  { background: rgba(34,211,138,0.15); color: var(--good); border: 1px solid rgba(34,211,138,0.3); }
        .badge-blue   { background: rgba(0,194,255,0.12);  color: var(--accent); border: 1px solid rgba(0,194,255,0.25); }
        .badge-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
        .vo-badge { font-family: var(--mono); font-size: 11px; }

        /* ── Grid layout ─────────────────────────────── */
        .layout {
            display: grid;
            grid-template-columns: 1fr 380px;
            grid-template-rows: auto auto;
            gap: 12px;
            padding: 14px 16px;
            max-width: 1400px;
            margin: 0 auto;
        }
        @media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } }

        .col-main { display: flex; flex-direction: column; gap: 12px; }
        .col-side  { display: flex; flex-direction: column; gap: 12px; }

        /* ── Card ──────────────────────────────────────── */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }
        .card-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
        }
        .card-title {
            font-size: 13px; font-weight: 600; letter-spacing: 0.3px;
            display: flex; align-items: center; gap: 7px;
        }
        .card-icon { font-size: 15px; }
        .card-body { padding: 12px 14px; }

        /* ── Video stream ──────────────────────────────── */
        .stream-wrap {
            position: relative; background: #000; border-radius: 0 0 11px 11px; overflow: hidden;
        }
        #live-stream {
            width: 100%; display: block;
            border-radius: 0 0 11px 11px;
        }
        .vo-overlay {
            position: absolute; bottom: 10px; right: 10px;
            background: rgba(7,13,20,0.82);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 6px 10px;
            font-family: var(--mono);
            font-size: 11px;
            color: var(--muted);
            line-height: 1.7;
        }
        .vo-overlay .vo-status-tracking { color: var(--good); }
        .vo-overlay .vo-status-initializing { color: var(--warn); }
        .vo-overlay .vo-status-lost { color: var(--danger); }

        /* ── KPI strip ─────────────────────────────────── */
        .kpi-strip {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
        }
        @media (max-width: 700px) { .kpi-strip { grid-template-columns: repeat(2, 1fr); } }
        .kpi {
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 12px;
        }
        .kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; }
        .kpi-value { font-size: 22px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
        .kpi-unit  { font-size: 11px; color: var(--muted); margin-top: 1px; }

        /* ── RF Heatmap panel ──────────────────────────── */
        .heatmap-controls {
            display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
            padding: 8px 14px;
            border-bottom: 1px solid var(--border);
            background: rgba(255,255,255,0.015);
        }
        select, input[type=\"text\"], input[type=\"number\"] {
            background: #0a1520;
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 7px;
            padding: 5px 9px;
            font-size: 12px;
            font-family: var(--font);
            outline: none;
        }
        select:focus, input:focus { border-color: var(--accent); }
        .btn {
            background: var(--surface2);
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 7px;
            padding: 5px 11px;
            font-size: 12px;
            font-family: var(--font);
            cursor: pointer;
            transition: background 0.15s, border-color 0.15s;
            font-weight: 500;
        }
        .btn:hover { background: #1b2f44; border-color: var(--accent); }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-accent {
            background: linear-gradient(135deg, rgba(0,194,255,0.18), rgba(124,58,237,0.18));
            border-color: rgba(0,194,255,0.4);
            color: var(--accent);
        }
        .btn-accent:hover { background: linear-gradient(135deg, rgba(0,194,255,0.28), rgba(124,58,237,0.28)); }
        .btn-danger { border-color: rgba(239,68,68,0.4); color: var(--danger); }
        .btn-danger:hover { background: rgba(239,68,68,0.1); }

        .heatmap-wrap {
            position: relative; text-align: center; background: #060c14; min-height: 180px;
        }
        #rf-heatmap {
            width: 100%; max-height: 260px; object-fit: contain; display: block;
        }
        .heatmap-legend {
            display: flex; justify-content: space-between; padding: 4px 14px;
            font-size: 10px; color: var(--muted);
            border-top: 1px solid var(--border);
        }
        .legend-bar {
            height: 6px; flex: 1; margin: 0 8px; border-radius: 3px;
            background: linear-gradient(to right, #00f, #0ff, #0f0, #ff0, #f00);
        }

        /* ── Sample table ──────────────────────────────── */
        .sample-stat {
            display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }
        .sstat { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
        .sstat-label { font-size: 10px; color: var(--muted); }
        .sstat-val { font-size: 16px; font-weight: 700; }

        /* ── Network list ──────────────────────────────── */
        .net-list { max-height: 280px; overflow-y: auto; }
        .net-item {
            display: flex; align-items: center; gap: 10px;
            padding: 9px 14px;
            border-bottom: 1px solid rgba(26,47,68,0.6);
            transition: background 0.12s;
        }
        .net-item:hover { background: rgba(255,255,255,0.025); }
        .net-bars { display: flex; gap: 2px; align-items: flex-end; height: 16px; }
        .net-bar { width: 4px; border-radius: 2px; }
        .net-info { flex: 1; min-width: 0; }
        .net-ssid { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .net-meta { font-size: 11px; color: var(--muted); margin-top: 1px; }
        .net-dbm { font-family: var(--mono); font-size: 13px; font-weight: 600; white-space: nowrap; }

        /* ── Survey panel ──────────────────────────────── */
        .survey-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
        .survey-progress {
            height: 4px; background: var(--surface2); border-radius: 2px; margin: 8px 0;
            overflow: hidden;
        }
        .survey-progress-bar {
            height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2));
            border-radius: 2px; transition: width 0.4s ease;
        }
        .status-msg { font-size: 12px; color: var(--muted); line-height: 1.4; padding: 4px 0; }

        /* ── Scrollbar ─────────────────────────────────── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        /* ── Animations ────────────────────────────────── */
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
        .blink { animation: pulse 1.8s infinite; }
    </style>
</head>
<body>

<header>
    <div class=\"header-logo\">
        <div class=\"logo-icon\">📡</div>
        <div class=\"logo-text\">Edge<span>RF</span> 3D</div>
    </div>
    <div class=\"header-badges\">
        <span class=\"badge badge-blue\">3D RF Digital Twin</span>
        <span id=\"vo-status-badge\" class=\"badge badge-purple vo-badge blink\">VO INIT</span>
        <span id=\"tracking-badge\" class=\"badge badge-green\" style=\"display:none\">TRACKING</span>
    </div>
</header>

<div class=\"layout\">

    <!-- ── Left column ─────────────────────── -->
    <div class=\"col-main\">

        <!-- Live Camera + VO -->
        <div class=\"card\">
            <div class=\"card-header\">
                <span class=\"card-title\"><span class=\"card-icon\">🎥</span> Live Camera — Visual Odometry</span>
                <span id=\"fps-badge\" class=\"badge badge-blue\">0.0 fps</span>
            </div>
            <div class=\"stream-wrap\">
                <img id=\"live-stream\" src=\"/video\" alt=\"Live MJPEG stream\" />
                <div class=\"vo-overlay\">
                    <div><span class=\"blink\">●</span> VO &nbsp;<span id=\"vo-overlay-status\">INITIALIZING</span></div>
                    <div>X <span id=\"vo-x\">0.00</span> &nbsp;Y <span id=\"vo-y\">0.00</span> &nbsp;Z <span id=\"vo-z\">0.00</span></div>
                    <div>pts <span id=\"vo-pts\">0</span> &nbsp;conf <span id=\"vo-conf\">0.00</span></div>
                </div>
            </div>
            <div class=\"card-body\">
                <div class=\"kpi-strip\">
                    <div class=\"kpi\"><div class=\"kpi-label\">FPS</div><div id=\"kpi-fps\" class=\"kpi-value\">—</div></div>
                    <div class=\"kpi\"><div class=\"kpi-label\">Signal</div><div id=\"kpi-dbm\" class=\"kpi-value\">—</div><div class=\"kpi-unit\">dBm</div></div>
                    <div class=\"kpi\"><div class=\"kpi-label\">Intensity</div><div id=\"kpi-ci\" class=\"kpi-value\">—</div></div>
                    <div class=\"kpi\"><div class=\"kpi-label\">Uptime</div><div id=\"kpi-uptime\" class=\"kpi-value\">0</div><div class=\"kpi-unit\">s</div></div>
                </div>
            </div>
        </div>

        <!-- 3D RF Heatmap -->
        <div class=\"card\">
            <div class=\"card-header\">
                <span class=\"card-title\"><span class=\"card-icon\">🔥</span> 3D Indoor RF Heatmap</span>
                <span class=\"badge badge-purple\" id=\"db-sample-badge\">0 samples</span>
            </div>
            <div class=\"heatmap-controls\">
                <label style=\"font-size:12px;color:var(--muted)\">SSID</label>
                <select id=\"ssid-select\"><option value=\"\">All networks</option></select>
                <label style=\"font-size:12px;color:var(--muted)\">Method</label>
                <select id=\"method-select\">
                    <option value=\"gaussian\">Gaussian RBF</option>
                    <option value=\"idw\">IDW</option>
                    <option value=\"kriging\">Kriging</option>
                </select>
                <button class=\"btn btn-accent\" id=\"btn-refresh-heatmap\" onclick=\"refreshHeatmap()\">⟳ Refresh</button>
                <button class=\"btn btn-danger\" id=\"btn-clear-db\" onclick=\"clearRFDB()\">🗑 Clear DB</button>
                <a id=\"btn-export-csv\" class=\"btn\" href=\"/api/rf/export/csv\" download=\"rf_samples.csv\">⬇ CSV</a>
            </div>
            <div class=\"heatmap-wrap\">
                <img id=\"rf-heatmap\" src=\"/api/rf/heatmap\" alt=\"RF heatmap\" />
            </div>
            <div class=\"heatmap-legend\">
                <span>Weak (−90 dBm)</span>
                <div class=\"legend-bar\"></div>
                <span>Strong (−30 dBm)</span>
            </div>
            <div class=\"sample-stat\">
                <div class=\"sstat\"><div class=\"sstat-label\">Total Samples</div><div id=\"ss-total\" class=\"sstat-val\">0</div></div>
                <div class=\"sstat\"><div class=\"sstat-label\">SSIDs Tracked</div><div id=\"ss-ssids\" class=\"sstat-val\">0</div></div>
            </div>
            <div class=\"card-body\" style=\"padding-top:0\">
                <div style=\"display:flex;gap:8px;align-items:center;flex-wrap:wrap\">
                    <button class=\"btn\" onclick=\"resetVOPose()\">🔄 Reset VO Pose</button>
                    <span class=\"status-msg\" id=\"heatmap-status\">Walk through the room to collect RF samples automatically.</span>
                </div>
            </div>
        </div>

    </div>

    <!-- ── Right column ─────────────────────── -->
    <div class=\"col-side\">

        <!-- WiFi Networks -->
        <div class=\"card\">
            <div class=\"card-header\">
                <span class=\"card-title\"><span class=\"card-icon\">📶</span> Detected Networks</span>
                <button class=\"btn\" onclick=\"triggerScan()\" id=\"scan-now\">Scan now</button>
            </div>
            <div id=\"network-status\" class=\"card-body status-msg\">Initializing scanner...</div>
            <div class=\"net-list\" id=\"networks\"></div>
        </div>

        <!-- Room WiFi Map (legacy 2D mapper) -->
        <div class=\"card\">
            <div class=\"card-header\">
                <span class=\"card-title\"><span class=\"card-icon\">🗺️</span> Room WiFi Map (2D)</span>
            </div>
            <div style=\"padding:8px\">
                <img id=\"map-view\" src=\"/api/mapper/heatmap\" alt=\"2D room heatmap\"
                     style=\"width:100%;border-radius:8px;cursor:crosshair\" />
                <div class=\"status-msg\" style=\"margin-top:6px\">Click map to set your current position.</div>
                <div id=\"mapper-status\" class=\"status-msg\">Position not set</div>
                <div style=\"display:flex;gap:6px;flex-wrap:wrap;margin-top:8px\">
                    <button class=\"btn\" onclick=\"clearMapper()\">Clear</button>
                    <input id=\"map-name\" type=\"text\" placeholder=\"snapshot name\" style=\"flex:1\" />
                    <button class=\"btn btn-accent\" onclick=\"saveSnapshot()\">Save</button>
                    <select id=\"map-snapshot-list\" style=\"flex:1\"></select>
                    <button class=\"btn\" onclick=\"loadSnapshot()\">Load</button>
                </div>
                <div id=\"map-io-status\" class=\"status-msg\" style=\"margin-top:4px\"></div>
            </div>
        </div>

        <!-- Guided Survey -->
        <div class=\"card\">
            <div class=\"card-header\">
                <span class=\"card-title\"><span class=\"card-icon\">📋</span> Guided Survey</span>
            </div>
            <div class=\"card-body\">
                <div class=\"status-msg\">Workflow: Start → move to target → Capture → repeat → Export</div>
                <div id=\"survey-status\" class=\"status-msg\" style=\"margin-top:6px;font-weight:600\">Ready</div>
                <div class=\"survey-progress\"><div id=\"survey-bar\" class=\"survey-progress-bar\" style=\"width:0%\"></div></div>
                <div id=\"survey-detail\" class=\"status-msg\"></div>
                <div class=\"survey-grid\" style=\"margin-top:10px\">
                    <div>
                        <label style=\"font-size:11px;color:var(--muted)\">Rows</label>
                        <input id=\"survey-rows\" type=\"number\" min=\"1\" max=\"20\" value=\"3\" style=\"width:100%;margin-top:3px\" />
                    </div>
                    <div>
                        <label style=\"font-size:11px;color:var(--muted)\">Cols</label>
                        <input id=\"survey-cols\" type=\"number\" min=\"1\" max=\"20\" value=\"3\" style=\"width:100%;margin-top:3px\" />
                    </div>
                </div>
                <div style=\"display:flex;gap:6px;flex-wrap:wrap;margin-top:10px\">
                    <button class=\"btn btn-accent\" id=\"survey-start\" onclick=\"startSurvey()\">Start</button>
                    <button class=\"btn\" id=\"survey-capture\" onclick=\"capturePoint()\" disabled>Capture</button>
                    <button class=\"btn btn-danger\" id=\"survey-reset\" onclick=\"resetSurvey()\" disabled>Reset</button>
                </div>
                <div style=\"display:flex;gap:6px;flex-wrap:wrap;margin-top:8px\">
                    <button class=\"btn\" id=\"survey-export-png\" onclick=\"exportSurveyPNG()\" disabled>⬇ PNG</button>
                    <button class=\"btn\" id=\"survey-export-json\" onclick=\"exportSurveyJSON()\" disabled>⬇ JSON</button>
                    <button class=\"btn\" id=\"survey-export-csv\" onclick=\"exportSurveyCSV()\" disabled>⬇ CSV</button>
                </div>
                <div id=\"survey-feedback\" class=\"status-msg\" style=\"margin-top:6px\"></div>
            </div>
        </div>

    </div>
</div>

<script>
    // ─── State ────────────────────────────────────────────────────────
    let surveyActive = false, surveyCompleted = 0, surveyTotal = 0;
    let lastMapClickMs = 0;
    let scannerLastError = null;
    let currentSSIDs = [];

    // ─── Signal helpers ───────────────────────────────────────────────
    function dbmToColor(dbm) {
        const t = Math.max(0, Math.min(1, (dbm + 90) / 60));
        const r = Math.round(255 * (1 - t));
        const g = Math.round(255 * t);
        return `rgb(${r},${g},0)`;
    }
    function signalBars(dbm) {
        const t = Math.max(0, Math.min(1, (dbm + 90) / 60));
        const filled = Math.round(t * 4);
        const color = dbmToColor(dbm);
        let html = '';
        for (let i = 0; i < 4; i++) {
            const h = 4 + i * 4;
            const bg = i < filled ? color : 'rgba(255,255,255,0.12)';
            html += `<div class=\"net-bar\" style=\"height:${h}px;background:${bg}\"></div>`;
        }
        return html;
    }

    // ─── Status polling ───────────────────────────────────────────────
    async function pollStatus() {
        try {
            const s = await fetch('/api/status').then(r => r.json());
            document.getElementById('kpi-fps').textContent = (+s.fps || 0).toFixed(1);
            document.getElementById('fps-badge').textContent = (+s.fps || 0).toFixed(1) + ' fps';
            document.getElementById('kpi-dbm').textContent = (+s.signal_strength_dbm || -90).toFixed(0);
            document.getElementById('kpi-ci').textContent = (+s.combined_intensity || 0).toFixed(2);
            document.getElementById('kpi-uptime').textContent = Math.floor(s.uptime_seconds || 0);
            scannerLastError = s.scanner_last_error || null;
            const ns = document.getElementById('network-status');
            if (ns) ns.textContent = scannerLastError
                ? ('Scan error: ' + scannerLastError)
                : ('Scanner OK | latency: ' + (+s.scan_latency_ms || 0).toFixed(0) + ' ms');
        } catch (_) {}
    }

    // ─── VO pose polling ──────────────────────────────────────────────
    async function pollPose() {
        try {
            const p = await fetch('/api/rf/pose').then(r => r.json());
            document.getElementById('vo-x').textContent = (+p.x || 0).toFixed(2);
            document.getElementById('vo-y').textContent = (+p.y || 0).toFixed(2);
            document.getElementById('vo-z').textContent = (+p.z || 0).toFixed(2);
            document.getElementById('vo-pts').textContent = p.tracked_points || 0;
            document.getElementById('vo-conf').textContent = (+p.confidence || 0).toFixed(2);
            const status = (p.status || 'INITIALIZING').toUpperCase();
            const statusEl = document.getElementById('vo-overlay-status');
            if (statusEl) {
                statusEl.textContent = status;
                statusEl.className = 'vo-status-' + status.toLowerCase();
            }
            const badge = document.getElementById('vo-status-badge');
            const track = document.getElementById('tracking-badge');
            if (status === 'TRACKING') {
                badge.style.display = 'none';
                track.style.display = '';
                track.textContent = `✓ TRACKING (${p.tracked_points} pts)`;
            } else {
                badge.style.display = '';
                badge.textContent = 'VO ' + status;
                track.style.display = 'none';
            }
        } catch (_) {}
    }

    // ─── RF DB status ─────────────────────────────────────────────────
    async function pollRFDB() {
        try {
            const s = await fetch('/api/rf/db_status').then(r => r.json());
            document.getElementById('ss-total').textContent = s.total_samples || 0;
            document.getElementById('ss-ssids').textContent = s.ssid_count || 0;
            document.getElementById('db-sample-badge').textContent = (s.total_samples || 0) + ' samples';

            // Update SSID select
            const sel = document.getElementById('ssid-select');
            const prev = sel.value;
            const ssids = Array.isArray(s.ssids) ? s.ssids : [];
            if (JSON.stringify(ssids) !== JSON.stringify(currentSSIDs)) {
                currentSSIDs = ssids;
                sel.innerHTML = '<option value=\"\">All networks</option>';
                for (const id of ssids) {
                    const o = document.createElement('option');
                    o.value = id; o.textContent = id;
                    sel.appendChild(o);
                }
                if (prev) sel.value = prev;
            }
        } catch (_) {}
    }

    // ─── Heatmap ──────────────────────────────────────────────────────
    function refreshHeatmap() {
        const ssid = document.getElementById('ssid-select').value;
        const method = document.getElementById('method-select').value;
        const params = new URLSearchParams({ method });
        if (ssid) params.set('ssid', ssid);
        document.getElementById('rf-heatmap').src = '/api/rf/heatmap?' + params.toString() + '&t=' + Date.now();
        document.getElementById('heatmap-status').textContent = `Updated: ${method} interpolation${ssid ? ' for ' + ssid : ''}`;
    }

    async function clearRFDB() {
        if (!confirm('Clear all RF samples from the database?')) return;
        await fetch('/api/rf/clear', { method: 'POST' });
        refreshHeatmap();
        pollRFDB();
    }

    async function resetVOPose() {
        await fetch('/api/rf/reset_pose', { method: 'POST' });
        document.getElementById('heatmap-status').textContent = 'VO pose reset to origin (0,0,0).';
    }

    // ─── Networks ─────────────────────────────────────────────────────
    async function pollNetworks() {
        try {
            const nets = await fetch('/api/networks').then(r => r.json());
            const list = document.getElementById('networks');
            list.innerHTML = '';
            if (!Array.isArray(nets) || !nets.length) {
                list.innerHTML = '<div style=\"padding:12px 14px;font-size:12px;color:var(--muted)\">No networks detected</div>';
                return;
            }
            for (const n of nets) {
                const dbm = +(n.dbm || -100);
                const color = dbmToColor(dbm);
                const div = document.createElement('div');
                div.className = 'net-item';
                div.innerHTML =
                    '<div class=\"net-bars\">' + signalBars(dbm) + '</div>' +
                    '<div class=\"net-info\">' +
                      '<div class=\"net-ssid\">' + (n.ssid || '<hidden>') + '</div>' +
                      '<div class=\"net-meta\">' + (n.bssid || '') + ' · ' + (n.freq || 0) + ' MHz</div>' +
                    '</div>' +
                    '<div class=\"net-dbm\" style=\"color:' + color + '\">' + dbm.toFixed(0) + ' dBm</div>';
                list.appendChild(div);
            }
        } catch (_) {}
    }

    async function triggerScan() {
        const btn = document.getElementById('scan-now');
        if (btn) btn.disabled = true;
        try {
            await fetch('/api/scan/now', { method: 'POST' });
            await pollNetworks();
            await pollStatus();
        } catch (_) {}
        if (btn) btn.disabled = false;
    }

    // ─── 2D Mapper ────────────────────────────────────────────────────
    function refreshMapperImage() {
        const img = document.getElementById('map-view');
        if (img) img.src = '/api/mapper/heatmap?t=' + Date.now();
    }
    async function pollMapperStatus() {
        try {
            const s = await fetch('/api/mapper/status').then(r => r.json());
            const node = document.getElementById('mapper-status');
            if (node) node.textContent = s.has_position
                ? `Position: (${(+s.position_x).toFixed(2)}, ${(+s.position_y).toFixed(2)}) | Samples: ${s.sample_count}`
                : 'Position not set';
        } catch (_) {}
    }
    async function onMapClick(e) {
        const img = document.getElementById('map-view');
        if (!img) return;
        const now = Date.now();
        if (now - lastMapClickMs < 180) return;
        lastMapClickMs = now;
        const rect = img.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        try {
            await fetch('/api/mapper/position', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ x, y, sample_now: true }),
            });
            refreshMapperImage(); pollMapperStatus();
        } catch (_) {}
    }
    function setMapIoStatus(t) { const n=document.getElementById('map-io-status'); if(n) n.textContent=t||''; }
    async function refreshSnapshotList(pref) {
        try {
            const d = await fetch('/api/mapper/snapshots').then(r => r.json());
            const list = document.getElementById('map-snapshot-list');
            if (!list) return;
            const snaps = Array.isArray(d.snapshots) ? d.snapshots : [];
            list.innerHTML = snaps.length ? '' : '<option value=\"\">No snapshots</option>';
            for (const n of snaps) {
                const o = document.createElement('option');
                o.value = n; o.textContent = n; list.appendChild(o);
            }
            if (pref) list.value = pref;
        } catch (_) {}
    }
    async function saveSnapshot() {
        const input = document.getElementById('map-name');
        const name = input ? input.value || '' : '';
        try {
            const r = await fetch('/api/mapper/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name}) });
            const d = await r.json();
            setMapIoStatus(d.ok ? 'Saved: ' + d.name : 'Save failed');
            if (d.ok) await refreshSnapshotList(d.name);
        } catch(_) { setMapIoStatus('Save failed'); }
    }
    async function loadSnapshot() {
        const list = document.getElementById('map-snapshot-list');
        const sel = list ? list.value : '';
        if (!sel) { setMapIoStatus('Select a snapshot'); return; }
        try {
            const r = await fetch('/api/mapper/load', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name:sel}) });
            const d = await r.json();
            setMapIoStatus(d.ok ? 'Loaded: ' + sel : 'Load failed');
            if (d.ok) { refreshMapperImage(); pollMapperStatus(); }
        } catch(_) { setMapIoStatus('Load failed'); }
    }
    async function clearMapper() {
        await fetch('/api/mapper/clear', {method:'POST'});
        refreshMapperImage(); pollMapperStatus();
    }

    // ─── Guided Survey ────────────────────────────────────────────────
    function setSurveyFeedback(t) { const n=document.getElementById('survey-feedback'); if(n) n.textContent=t||''; }
    function updateSurveyButtons() {
        const has = surveyCompleted > 0;
        document.getElementById('survey-start').disabled   = surveyActive;
        document.getElementById('survey-capture').disabled = !surveyActive;
        document.getElementById('survey-reset').disabled   = !surveyActive;
        document.getElementById('survey-export-png').disabled  = !has;
        document.getElementById('survey-export-json').disabled = !has;
        document.getElementById('survey-export-csv').disabled  = !has;
        const bar = document.getElementById('survey-bar');
        if (bar && surveyTotal > 0) bar.style.width = (surveyCompleted/surveyTotal*100).toFixed(1) + '%';
    }
    async function pollSurveyStatus() {
        try {
            const s = await fetch('/api/survey/status').then(r => r.json());
            surveyActive = !!s.active;
            surveyCompleted = +s.completed_points || 0;
            surveyTotal = +s.total_points || 0;
            const statusEl = document.getElementById('survey-status');
            const detailEl = document.getElementById('survey-detail');
            if (s.completed) {
                if (statusEl) statusEl.textContent = '✓ Survey complete!';
                if (detailEl) detailEl.textContent = `Captured ${s.completed_points}/${s.total_points} points`;
            } else if (s.active) {
                if (statusEl) statusEl.textContent = `Step ${s.current_index}/${s.total_points}: move to target, then Capture`;
                if (detailEl) detailEl.textContent = `Target: (${(+s.target_x).toFixed(2)}, ${(+s.target_y).toFixed(2)}) | Done: ${s.completed_points}`;
            } else {
                if (statusEl) statusEl.textContent = 'Ready to start';
                if (detailEl) detailEl.textContent = 'Set rows and cols, then click Start.';
            }
            updateSurveyButtons();
        } catch (_) {}
    }
    async function startSurvey() {
        const rows = +document.getElementById('survey-rows').value || 3;
        const cols = +document.getElementById('survey-cols').value || 3;
        try {
            const r = await fetch('/api/survey/start', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({rows,cols,margin:0.08,serpentine:true}) });
            const d = await r.json();
            setSurveyFeedback(d.ok ? 'Survey started. Go to target and click Capture.' : 'Start failed: '+(d.error||'?'));
            pollSurveyStatus();
        } catch(_) {}
    }
    async function capturePoint() {
        try {
            const r = await fetch('/api/survey/capture', {method:'POST'});
            const d = await r.json();
            setSurveyFeedback(d.ok ? `Saved ${d.completed_points}/${surveyTotal}` : 'Capture failed: '+(d.error||'?'));
            pollSurveyStatus(); pollMapperStatus(); refreshMapperImage();
        } catch(_) {}
    }
    async function resetSurvey() {
        await fetch('/api/survey/reset', {method:'POST'});
        setSurveyFeedback('Survey reset.'); pollSurveyStatus();
    }
    function downloadBlob(url, filename) {
        fetch(url).then(r=>r.blob()).then(blob=>{
            const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=filename;
            document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(a.href);
        }).catch(_=>{});
    }
    function exportSurveyJSON() { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/survey/data','survey_data.json'); setSurveyFeedback('JSON downloaded.'); }
    function exportSurveyPNG()  { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/report/png','survey_report_map.png'); setSurveyFeedback('PNG downloaded.'); }
    function exportSurveyCSV()  { if(surveyCompleted<=0){setSurveyFeedback('No captures yet');return;} downloadBlob('/api/report/csv','survey_report.csv'); setSurveyFeedback('CSV downloaded.'); }

    // ─── Map click ────────────────────────────────────────────────────
    document.getElementById('map-view').addEventListener('click', onMapClick);

    // ─── Init & polling ───────────────────────────────────────────────
    pollStatus(); pollPose(); pollNetworks(); pollMapperStatus(); pollSurveyStatus(); pollRFDB();
    refreshSnapshotList(); refreshMapperImage();

    setInterval(pollStatus,   2000);
    setInterval(pollPose,     1000);
    setInterval(pollNetworks, 3000);
    setInterval(pollMapperStatus, 3000);
    setInterval(pollSurveyStatus, 3000);
    setInterval(pollRFDB, 4000);
    setInterval(refreshHeatmap, 8000);   // auto-refresh heatmap every 8s
    setInterval(refreshMapperImage, 3000);
</script>
</body>
</html>"""



def _build_3dmap_html() -> str:
    """Return the interactive 3D WiFi Heatmap viewer HTML."""
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>EdgeRF 3D — Spatial RF Digital Twin</title>
    <meta name="description" content="3D Indoor WiFi Signal Mapping System — Ekahau-style RF Digital Twin on Raspberry Pi" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <style>
        :root {
            --bg:       #070d14;
            --surface:  #0d1825;
            --surface2: #111f2e;
            --border:   #1a2f44;
            --accent:   #00c2ff;
            --accent2:  #7c3aed;
            --good:     #22d38a;
            --warn:     #f59e0b;
            --danger:   #ef4444;
            --text:     #dde6f0;
            --muted:    #7a9bb5;
            --font:     'Inter', sans-serif;
            --mono:     'JetBrains Mono', monospace;
        }
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; font-family: var(--font); background: var(--bg); color: var(--text); overflow: hidden; }
        
        /* ── Header ─────────────────────────────────── */
        header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
            background: rgba(7,13,20,0.9);
            backdrop-filter: blur(12px);
            height: 56px;
        }
        .header-logo { display: flex; align-items: center; gap: 10px; }
        .logo-icon {
            width: 32px; height: 32px;
            background: linear-gradient(135deg, var(--accent), var(--accent2));
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; font-weight: bold;
        }
        .logo-text { font-size: 17px; font-weight: 700; letter-spacing: -0.3px; }
        .logo-text span { color: var(--accent); }
        .header-btn {
            padding: 6px 14px; border-radius: 6px; font-size: 13px; font-weight: 600;
            background: var(--surface2); color: var(--text); border: 1px solid var(--border);
            text-decoration: none; display: flex; align-items: center; gap: 6px;
            transition: all 0.2s ease; cursor: pointer;
        }
        .header-btn:hover { background: var(--border); border-color: var(--accent); color: #fff; }

        /* ── Main Layout ────────────────────────────── */
        .layout {
            display: grid;
            grid-template-columns: 360px 1fr;
            height: calc(100vh - 56px);
            background: var(--bg);
        }

        /* ── Sidebar ────────────────────────────────── */
        aside {
            background: rgba(13,24,37,0.4);
            border-right: 1px solid var(--border);
            padding: 20px;
            display: flex; flex-direction: column; gap: 20px;
            overflow-y: auto;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex; flex-direction: column; gap: 14px;
        }
        .card-title {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.5px; color: var(--muted); border-bottom: 1px solid var(--border);
            padding-bottom: 6px; display: flex; align-items: center; justify-content: space-between;
        }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        label { font-size: 12px; font-weight: 600; color: var(--text); }
        select, input[type="range"] {
            width: 100%; padding: 8px 12px; border-radius: 6px;
            background: var(--surface2); border: 1px solid var(--border);
            color: var(--text); font-family: var(--font); font-size: 13px;
        }
        select:focus { outline: none; border-color: var(--accent); }
        
        .slider-container { display: flex; align-items: center; gap: 10px; }
        .slider-val { font-family: var(--mono); font-size: 12px; min-width: 32px; text-align: right; }

        .btn-group { display: flex; flex-direction: column; gap: 8px; }
        .btn {
            width: 100%; padding: 10px; border-radius: 6px; font-size: 13px; font-weight: 600;
            border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;
            transition: all 0.2s ease;
        }
        .btn-primary { background: var(--accent); color: #000; }
        .btn-primary:hover { background: #33d1ff; box-shadow: 0 0 12px rgba(0,194,255,0.3); }
        .btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
        .btn-secondary:hover { background: var(--border); border-color: var(--accent); }
        .btn-danger { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
        .btn-danger:hover { background: var(--danger); color: #fff; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* ── Status panel ───────────────────────────── */
        .stat-row { display: flex; justify-content: space-between; font-size: 12px; }
        .stat-lbl { color: var(--muted); }
        .stat-val { font-family: var(--mono); font-weight: 500; }

        /* ── Main View Panel ───────────────────────── */
        main {
            display: flex; flex-direction: column;
            padding: 20px; gap: 16px; overflow: hidden;
        }
        .tab-bar {
            display: flex; gap: 8px; border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }
        .tab-btn {
            padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600;
            background: transparent; border: 1px solid transparent; color: var(--muted);
            cursor: pointer; transition: all 0.2s ease;
        }
        .tab-btn:hover { color: var(--text); }
        .tab-btn.active {
            background: var(--surface); border: 1px solid var(--border);
            color: var(--accent);
        }

        .viewer-container {
            flex: 1; background: var(--surface); border: 1px solid var(--border);
            border-radius: 12px; position: relative; overflow: hidden;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }
        
        /* ── Canvas View ────────────────────────────── */
        #canvas-wrapper {
            position: absolute; width: 100%; height: 100%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
        }
        canvas {
            display: block; cursor: grab; background: #08101a;
            box-shadow: inset 0 0 40px rgba(0,0,0,0.6);
        }
        canvas:active { cursor: grabbing; }

        .canvas-toolbar {
            position: absolute; bottom: 16px; left: 16px; right: 16px;
            background: rgba(13,24,37,0.85); border: 1px solid var(--border);
            backdrop-filter: blur(8px); border-radius: 8px;
            padding: 10px 16px; display: flex; gap: 20px; align-items: center;
            justify-content: space-between;
        }
        .canvas-ctrl { display: flex; align-items: center; gap: 8px; font-size: 11px; }
        .canvas-ctrl label { font-size: 11px; color: var(--muted); }
        .canvas-ctrl input[type="range"] { width: 90px; }

        .instruction-overlay {
            position: absolute; top: 16px; left: 16px;
            font-size: 12px; color: var(--muted);
            background: rgba(7,13,20,0.6); padding: 6px 12px;
            border-radius: 6px; pointer-events: none;
            border: 1px solid rgba(26,47,68,0.5);
        }

        /* ── Static Matplotlib PNG View ─────────────── */
        #png-wrapper {
            position: absolute; width: 100%; height: 100%;
            display: none; align-items: center; justify-content: center;
            overflow: auto; padding: 20px; background: #08101a;
        }
        #heatmap-png {
            max-width: 100%; max-height: 100%; object-fit: contain;
            border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.7);
        }

        .loading-overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(7,13,20,0.85); backdrop-filter: blur(4px);
            display: none; flex-direction: column; align-items: center;
            justify-content: center; gap: 14px; z-index: 10;
        }
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid rgba(0,194,255,0.1);
            border-top: 3px solid var(--accent);
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .loading-text { font-size: 14px; font-weight: 500; color: var(--accent); }

        /* ── Legend ─────────────────────────────────── */
        .legend-bar {
            height: 8px; border-radius: 4px;
            background: linear-gradient(to right, var(--danger), var(--warn), var(--good));
            margin: 6px 0;
        }
        .legend-labels { display: flex; justify-content: space-between; font-size: 10px; color: var(--muted); }
    </style>
</head>
<body>
    <header>
        <div class="header-logo">
            <div class="logo-icon">📶</div>
            <div class="logo-text">EdgeRF<span>3D</span> Digital Twin</div>
        </div>
        <a href="/" class="header-btn">
            <span>←</span> Back to Live Dashboard
        </a>
    </header>

    <div class="layout">
        <!-- Sidebar Controls -->
        <aside>
            <!-- 3D Heatmap Config -->
            <div class="card">
                <div class="card-title">Heatmap Interpolation</div>
                
                <div class="form-group">
                    <label for="ssid-select">Filter by SSID</label>
                    <select id="ssid-select">
                        <option value="">All Networks</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="method-select">2D/3D Model Method</label>
                    <select id="method-select">
                        <option value="gaussian">Gaussian Propagation</option>
                        <option value="idw">Inverse Distance Weighting</option>
                        <option value="kriging">Kriging (Spherical Fit)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="grid-resolution">Matplotlib Grid Resolution</label>
                    <div class="slider-container">
                        <input type="range" id="grid-resolution" min="20" max="100" value="60" />
                        <div class="slider-val" id="grid-val">60</div>
                    </div>
                </div>

                <div>
                    <div class="legend-labels"><span>Signal Legend</span></div>
                    <div class="legend-bar"></div>
                    <div class="legend-labels">
                        <span>-85 dBm (Weak)</span>
                        <span>-65 dBm</span>
                        <span>-45 dBm (Strong)</span>
                    </div>
                </div>
            </div>

            <!-- Actions -->
            <div class="card">
                <div class="card-title">Digital Twin Actions</div>
                <div class="btn-group">
                    <button class="btn btn-primary" id="btn-render-png">
                        ⚡ Generate High-Res 3D Heatmap
                    </button>
                    <button class="btn btn-secondary" id="btn-rebuild-floorplan">
                        🧱 Force Rebuild Floor Plan
                    </button>
                    <button class="btn btn-secondary" id="btn-download-csv">
                        📥 Download RF samples.csv
                    </button>
                    <button class="btn btn-danger" id="btn-clear-db">
                        🗑️ Clear RF Database
                    </button>
                </div>
            </div>

            <!-- RF DB Stats -->
            <div class="card">
                <div class="card-title">Digital Twin Stats</div>
                <div class="stat-row">
                    <span class="stat-lbl">RF Samples Collected</span>
                    <span class="stat-val" id="stat-samples">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-lbl">Unique SSIDs Seen</span>
                    <span class="stat-val" id="stat-ssids">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-lbl">Floor Dimension (VO)</span>
                    <span class="stat-val" id="stat-dimensions">- x -</span>
                </div>
                <div class="stat-row">
                    <span class="stat-lbl">Estimated APs</span>
                    <span class="stat-val" id="stat-aps">0</span>
                </div>
            </div>
        </aside>

        <!-- Main Display Panel -->
        <main>
            <div class="tab-bar">
                <button class="tab-btn active" data-tab="canvas">3D Interactive Floor Plan</button>
                <button class="tab-btn" data-tab="png">High-Resolution Matplotlib 3D Heatmap</button>
            </div>

            <div class="viewer-container">
                <!-- Loading indicator -->
                <div class="loading-overlay" id="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text" id="loading-text">Reconstructing space...</div>
                </div>

                <!-- Tab 1: Isometric Interactive Canvas -->
                <div id="canvas-wrapper">
                    <div class="instruction-overlay">
                        🖱️ Click & Drag to Rotate / Tilt | Use Sliders for offsets
                    </div>
                    <canvas id="iso-canvas" width="800" height="520"></canvas>
                    
                    <div class="canvas-toolbar">
                        <div class="canvas-ctrl">
                            <label for="slider-rotation">Rotate</label>
                            <input type="range" id="slider-rotation" min="-180" max="180" value="45" />
                        </div>
                        <div class="canvas-ctrl">
                            <label for="slider-tilt">Tilt</label>
                            <input type="range" id="slider-tilt" min="15" max="80" value="35" />
                        </div>
                        <div class="canvas-ctrl">
                            <label for="slider-zoom">Zoom</label>
                            <input type="range" id="slider-zoom" min="10" max="120" value="35" />
                        </div>
                        <div class="canvas-ctrl">
                            <label for="slider-height">Wall Height</label>
                            <input type="range" id="slider-height" min="0" max="100" value="30" />
                        </div>
                        <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" id="btn-reset-view">
                            Reset View
                        </button>
                    </div>
                </div>

                <!-- Tab 2: Matplotlib Static PNG -->
                <div id="png-wrapper">
                    <img id="heatmap-png" src="" alt="Matplotlib static 3D projection" />
                </div>
            </div>
        </main>
    </div>

    <script>
        // Global variables for floor plan and samples
        let floorPlan = null;
        let samples = [];
        let activeSSID = '';

        // UI Element handles
        const canvas = document.getElementById('iso-canvas');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('loading-overlay');
        const overlayText = document.getElementById('loading-text');
        const heatmapImg = document.getElementById('heatmap-png');

        // Isometric View Projection Variables
        let rotation = 45 * Math.PI / 180; // default 45 degrees
        let tilt = 35 * Math.PI / 180;     // default 35 degrees
        let zoom = 35;
        let wallHeightScalar = 3.0;        // height factor in pixels
        
        let cx = 0; // relative origin centers (computed from floor plan bounds)
        let cy = 0;

        // Mouse Drag Interaction State
        let isDragging = false;
        let lastMouseX = 0;
        let lastMouseY = 0;

        // Init function
        window.addEventListener('DOMContentLoaded', async () => {
            setupEventListeners();
            resizeCanvas();
            await fetchMetadata();
            await loadDigitalTwinData();
        });

        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth - 10;
            canvas.height = container.clientHeight - 65;
            draw();
        }

        window.addEventListener('resize', resizeCanvas);

        function setupEventListeners() {
            // Tab switching
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
                    btn.classList.add('active');
                    
                    const tab = btn.dataset.tab;
                    if (tab === 'canvas') {
                        document.getElementById('canvas-wrapper').style.display = 'flex';
                        document.getElementById('png-wrapper').style.display = 'none';
                        draw();
                    } else {
                        document.getElementById('canvas-wrapper').style.display = 'none';
                        document.getElementById('png-wrapper').style.display = 'flex';
                        refreshStaticPng();
                    }
                });
            });

            // Input handlers
            document.getElementById('ssid-select').addEventListener('change', async (e) => {
                activeSSID = e.target.value;
                await loadDigitalTwinData();
                if (document.getElementById('png-wrapper').style.display === 'flex') {
                    refreshStaticPng();
                }
            });

            document.getElementById('grid-resolution').addEventListener('input', (e) => {
                document.getElementById('grid-val').textContent = e.target.value;
            });

            // Sliders manual adjustment
            document.getElementById('slider-rotation').addEventListener('input', (e) => {
                rotation = parseFloat(e.target.value) * Math.PI / 180;
                draw();
            });
            document.getElementById('slider-tilt').addEventListener('input', (e) => {
                tilt = parseFloat(e.target.value) * Math.PI / 180;
                draw();
            });
            document.getElementById('slider-zoom').addEventListener('input', (e) => {
                zoom = parseFloat(e.target.value);
                draw();
            });
            document.getElementById('slider-height').addEventListener('input', (e) => {
                wallHeightScalar = parseFloat(e.target.value) / 10;
                draw();
            });

            document.getElementById('btn-reset-view').addEventListener('click', () => {
                rotation = 45 * Math.PI / 180;
                tilt = 35 * Math.PI / 180;
                zoom = 35;
                wallHeightScalar = 3.0;

                document.getElementById('slider-rotation').value = 45;
                document.getElementById('slider-tilt').value = 35;
                document.getElementById('slider-zoom').value = 35;
                document.getElementById('slider-height').value = 30;
                draw();
            });

            // Action buttons
            document.getElementById('btn-render-png').addEventListener('click', () => {
                // Switch to png tab and reload
                document.querySelector('.tab-btn[data-tab="png"]').click();
            });

            document.getElementById('btn-rebuild-floorplan').addEventListener('click', rebuildFloorPlan);
            
            document.getElementById('btn-download-csv').addEventListener('click', () => {
                let url = '/api/rf/export/csv';
                if (activeSSID) url += '?ssid=' + encodeURIComponent(activeSSID);
                window.location.href = url;
            });

            document.getElementById('btn-clear-db').addEventListener('click', async () => {
                if (!confirm("Clear all RF samples? This will wipe the active data session.")) return;
                showLoading("Clearing Database...");
                try {
                    await fetch('/api/rf/clear', { method: 'POST' });
                    await loadDigitalTwinData();
                } catch (e) {
                    console.error(e);
                } finally {
                    hideLoading();
                }
            });

            // Canvas drag interaction
            canvas.addEventListener('mousedown', (e) => {
                isDragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            });

            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const deltaX = e.clientX - lastMouseX;
                const deltaY = e.clientY - lastMouseY;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;

                rotation += deltaX * 0.006;
                tilt = Math.max(15 * Math.PI / 180, Math.min(80 * Math.PI / 180, tilt - deltaY * 0.006));

                // update sliders to match
                document.getElementById('slider-rotation').value = Math.round(rotation * 180 / Math.PI);
                document.getElementById('slider-tilt').value = Math.round(tilt * 180 / Math.PI);

                draw();
            });

            window.addEventListener('mouseup', () => { isDragging = false; });
        }

        function showLoading(text) {
            overlayText.textContent = text || 'Loading...';
            overlay.style.display = 'flex';
        }

        function hideLoading() {
            overlay.style.display = 'none';
        }

        // Fetch unique SSIDs to fill select options
        async function fetchMetadata() {
            try {
                const s = await fetch('/api/rf/ssids').then(r => r.json());
                const sel = document.getElementById('ssid-select');
                
                // save selected value
                const prev = sel.value;
                sel.innerHTML = '<option value="">All Networks</option>';
                
                if (s.ssids && Array.isArray(s.ssids)) {
                    s.ssids.forEach(ssid => {
                        const o = document.createElement('option');
                        o.value = ssid;
                        o.textContent = ssid;
                        sel.appendChild(o);
                    });
                }
                sel.value = prev;
            } catch (err) {
                console.error("Failed to load metadata", err);
            }
        }

        // Load data from backend
        async function loadDigitalTwinData() {
            showLoading("Syncing spatial digital twin...");
            try {
                // Fetch floor plan bounds and path
                const fpResponse = await fetch('/api/rf/floorplan');
                if (fpResponse.status === 400 || fpResponse.status === 500) {
                    floorPlan = null;
                } else {
                    floorPlan = await fpResponse.json();
                }

                // Fetch samples
                const url = activeSSID ? `/api/rf/samples?ssid=${encodeURIComponent(activeSSID)}&max=2000` : '/api/rf/samples?max=2000';
                const sRes = await fetch(url).then(r => r.json());
                samples = sRes.samples || [];

                updateStats();
                draw();
            } catch (err) {
                console.error("Error loading twin data", err);
            } finally {
                hideLoading();
            }
        }

        // Force rebuild floor plan
        async function rebuildFloorPlan() {
            showLoading("Reconstructing room boundaries...");
            try {
                const r = await fetch('/api/rf/floorplan/rebuild', { method: 'POST' }).then(r => r.json());
                if (!r.ok) {
                    alert("Rebuild failed: " + (r.error || '?'));
                } else {
                    await loadDigitalTwinData();
                }
            } catch (e) {
                alert("Rebuild failed.");
            } finally {
                hideLoading();
            }
        }

        // Refresh static Matplotlib PNG
        async function refreshStaticPng() {
            showLoading("Generating Matplotlib 3D isometric surface...");
            const res = document.getElementById('grid-resolution').value;
            let url = `/api/rf/render3d?grid=${res}&t=${Date.now()}`;
            if (activeSSID) url += `&ssid=${encodeURIComponent(activeSSID)}`;
            
            heatmapImg.onload = () => {
                hideLoading();
            };
            heatmapImg.onerror = () => {
                hideLoading();
                alert("Failed to load 3D render. Verify that you have collected enough samples (>=2) and matplotlib is installed on the Pi.");
            };
            heatmapImg.src = url;
        }

        function updateStats() {
            document.getElementById('stat-samples').textContent = samples.length;
            
            if (floorPlan) {
                const w = (floorPlan.x_max - floorPlan.x_min).toFixed(1);
                const d = (floorPlan.y_max - floorPlan.y_min).toFixed(1);
                document.getElementById('stat-dimensions').textContent = `${w} x ${d} VO units`;
                document.getElementById('stat-aps').textContent = floorPlan.ap_locations ? floorPlan.ap_locations.length : 0;
            } else {
                document.getElementById('stat-dimensions').textContent = "Not reconstructed";
                document.getElementById('stat-aps').textContent = 0;
            }

            fetch('/api/rf/ssids')
                .then(r => r.json())
                .then(s => {
                    document.getElementById('stat-ssids').textContent = s.ssid_count || 0;
                }).catch(()=>{});
        }

        // Isometric projection formula: centers coords around (cx, cy) and rotates/tilts
        function project(x, y, z) {
            const dx = x - cx;
            const dy = y - cy;

            // Rotate about Z-axis
            const rx = dx * Math.cos(rotation) - dy * Math.sin(rotation);
            const ry = dx * Math.sin(rotation) + dy * Math.cos(rotation);

            // Apply tilt and vertical scale to Z
            const px = rx * zoom + (canvas.width / 2);
            const py = (ry * Math.sin(tilt) - z * wallHeightScalar) * zoom + (canvas.height / 2) + 20;

            return { x: px, y: py, depth: ry };
        }

        function dbmToColor(dbm) {
            if (dbm >= -45) return '#22d38a'; // pure green
            if (dbm <= -85) return '#ef4444'; // pure red
            
            const pct = (dbm - (-85)) / 40.0;
            if (pct < 0.5) {
                const factor = pct * 2;
                const r = Math.round(239 + (245 - 239) * factor);
                const g = Math.round(68 + (158 - 68) * factor);
                const b = Math.round(68 + (11 - 68) * factor);
                return `rgb(${r},${g},${b})`;
            } else {
                const factor = (pct - 0.5) * 2;
                const r = Math.round(245 + (34 - 245) * factor);
                const g = Math.round(158 + (211 - 158) * factor);
                const b = Math.round(11 + (138 - 11) * factor);
                return `rgb(${r},${g},${b})`;
            }
        }

        // 3D Canvas rendering engine
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!floorPlan || !floorPlan.rooms || floorPlan.rooms.length === 0) {
                ctx.fillStyle = 'var(--muted)';
                ctx.font = '14px var(--font)';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText("No spatial model available yet. Complete a survey in the dashboard to generate a floor plan.", canvas.width / 2, canvas.height / 2);
                return;
            }

            // Update projection origin
            cx = (floorPlan.x_min + floorPlan.x_max) / 2;
            cy = (floorPlan.y_min + floorPlan.y_max) / 2;

            // 1. Draw floor grid at z = 0
            drawGrid();

            // 2. Draw Walk path trace on the floor (z = 0)
            drawWalkPath();

            // 3. Compile all 3D drawables to sort them by depth (back to front)
            const drawables = [];

            // Add walls
            const wallHeight = 2.0; // wall height in meters
            floorPlan.rooms.forEach(room => {
                const segments = [
                    [room.x_min, room.y_min, room.x_max, room.y_min],
                    [room.x_max, room.y_min, room.x_max, room.y_max],
                    [room.x_max, room.y_max, room.x_min, room.y_max],
                    [room.x_min, room.y_max, room.x_min, room.y_min]
                ];
                segments.forEach(seg => {
                    const mx = (seg[0] + seg[2]) / 2;
                    const my = (seg[1] + seg[3]) / 2;
                    // Project center of segment to determine depth
                    const dx = mx - cx;
                    const dy = my - cy;
                    const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);
                    
                    drawables.push({
                        type: 'wall',
                        depth: depth,
                        data: seg,
                        height: wallHeight
                    });
                });
            });

            // Add RF samples
            samples.forEach(s => {
                const dx = s.x - cx;
                const dy = s.y - cy;
                const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);

                drawables.push({
                    type: 'sample',
                    depth: depth,
                    data: s
                });
            });

            // Add Access Points (APs)
            if (floorPlan.ap_locations) {
                floorPlan.ap_locations.forEach(ap => {
                    const dx = ap.x - cx;
                    const dy = ap.y - cy;
                    const depth = dx * Math.sin(rotation) + dy * Math.cos(rotation);

                    drawables.push({
                        type: 'ap',
                        depth: depth,
                        data: ap
                    });
                });
            }

            // Sort depth: smaller depth values are drawn first (painted over by closer things)
            drawables.sort((a, b) => a.depth - b.depth);

            // 4. Render drawables back-to-front
            drawables.forEach(d => {
                if (d.type === 'wall') {
                    renderWall(d.data[0], d.data[1], d.data[2], d.data[3], d.height);
                } else if (d.type === 'sample') {
                    renderSample(d.data);
                } else if (d.type === 'ap') {
                    renderAP(d.data);
                }
            });
        }

        function drawGrid() {
            ctx.strokeStyle = 'rgba(26, 47, 68, 0.25)';
            ctx.lineWidth = 1;

            const gridSpacing = 1.0;
            const x_min = Math.floor(floorPlan.x_min);
            const x_max = Math.ceil(floorPlan.x_max);
            const y_min = Math.floor(floorPlan.y_min);
            const y_max = Math.ceil(floorPlan.y_max);

            // Draw grid line intersections
            for (let x = x_min; x <= x_max; x += gridSpacing) {
                ctx.beginPath();
                const start = project(x, y_min, 0);
                const end = project(x, y_max, 0);
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }

            for (let y = y_min; y <= y_max; y += gridSpacing) {
                ctx.beginPath();
                const start = project(x_min, y, 0);
                const end = project(x_max, y, 0);
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.stroke();
            }
        }

        function drawWalkPath() {
            if (!floorPlan.walk_path || floorPlan.walk_path.length < 2) return;

            ctx.beginPath();
            const startPt = project(floorPlan.walk_path[0][0], floorPlan.walk_path[0][1], 0.02);
            ctx.moveTo(startPt.x, startPt.y);

            for (let i = 1; i < floorPlan.walk_path.length; i++) {
                const pt = project(floorPlan.walk_path[i][0], floorPlan.walk_path[i][1], 0.02);
                ctx.lineTo(pt.x, pt.y);
            }

            ctx.strokeStyle = 'rgba(124, 58, 237, 0.55)'; // violet trail
            ctx.lineWidth = 2.5;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]); // reset

            // Start & End markers
            const start = project(floorPlan.walk_path[0][0], floorPlan.walk_path[0][1], 0.02);
            const end = project(floorPlan.walk_path[floorPlan.walk_path.length - 1][0], floorPlan.walk_path[floorPlan.walk_path.length - 1][1], 0.02);
            
            ctx.beginPath();
            ctx.arc(start.x, start.y, 4, 0, 2*Math.PI);
            ctx.fillStyle = 'var(--good)';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(end.x, end.y, 4, 0, 2*Math.PI);
            ctx.fillStyle = 'var(--danger)';
            ctx.fill();
        }

        function renderWall(x1, y1, x2, y2, height) {
            const base1 = project(x1, y1, 0);
            const base2 = project(x2, y2, 0);
            const top1 = project(x1, y1, height);
            const top2 = project(x2, y2, height);

            // Draw filled wall polygon
            ctx.beginPath();
            ctx.moveTo(base1.x, base1.y);
            ctx.lineTo(base2.x, base2.y);
            ctx.lineTo(top2.x, top2.y);
            ctx.lineTo(top1.x, top1.y);
            ctx.closePath();

            // Translucent blue glass fill
            ctx.fillStyle = 'rgba(0, 194, 255, 0.06)';
            ctx.fill();

            // Wireframe borders
            ctx.strokeStyle = 'rgba(26, 47, 68, 0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();

            // Bright wall cap line
            ctx.beginPath();
            ctx.moveTo(top1.x, top1.y);
            ctx.lineTo(top2.x, top2.y);
            ctx.strokeStyle = 'rgba(0, 194, 255, 0.35)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        function renderSample(s) {
            const floorPt = project(s.x, s.y, 0);
            const zPt = project(s.x, s.y, s.z);

            // Floor anchor point
            ctx.beginPath();
            ctx.arc(floorPt.x, floorPt.y, 1.8, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(122, 155, 181, 0.3)';
            ctx.fill();

            // Vertical line representing sample height (z)
            if (Math.abs(s.z) > 0.05) {
                ctx.beginPath();
                ctx.moveTo(floorPt.x, floorPt.y);
                ctx.lineTo(zPt.x, zPt.y);
                ctx.strokeStyle = 'rgba(122, 155, 181, 0.25)';
                ctx.setLineDash([2, 2]);
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.setLineDash([]);
            }

            // Colored sample dot
            ctx.beginPath();
            ctx.arc(zPt.x, zPt.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = dbmToColor(s.dbm);
            ctx.fill();
            
            ctx.strokeStyle = 'rgba(7, 13, 20, 0.6)';
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }

        function renderAP(ap) {
            const floorPt = project(ap.x, ap.y, 0);
            const poleHeight = 1.3; // standard router placement height
            const topPt = project(ap.x, ap.y, poleHeight);

            // Pole
            ctx.beginPath();
            ctx.moveTo(floorPt.x, floorPt.y);
            ctx.lineTo(topPt.x, topPt.y);
            ctx.strokeStyle = 'var(--accent)';
            ctx.lineWidth = 1.8;
            ctx.stroke();

            // Antenna bulb
            ctx.beginPath();
            ctx.arc(topPt.x, topPt.y, 5.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'var(--accent)';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1;
            ctx.stroke();

            // White center dot
            ctx.beginPath();
            ctx.arc(topPt.x, topPt.y, 1.8, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();

            // SSID Name tags
            ctx.fillStyle = 'rgba(7, 13, 20, 0.85)';
            ctx.strokeStyle = 'var(--border)';
            ctx.lineWidth = 1;
            
            const label = ap.ssid || '<hidden>';
            ctx.font = '10px var(--mono)';
            const textWidth = ctx.measureText(label).width;
            
            const padX = 6;
            const padY = 4;
            const boxW = textWidth + padX * 2;
            const boxH = 14 + padY * 2;
            
            const boxX = topPt.x - boxW / 2;
            const boxY = topPt.y - 25;

            // Draw label box
            ctx.beginPath();
            if (ctx.roundRect) {
                ctx.roundRect(boxX, boxY, boxW, boxH, 4);
            } else {
                ctx.rect(boxX, boxY, boxW, boxH);
            }
            ctx.fill();
            ctx.stroke();

            // Draw label text
            ctx.fillStyle = 'var(--text)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, topPt.x, boxY + boxH / 2);
        }
    </script>
</body>
</html>"""


def create_app(runtime: RuntimeContext, stream_worker: StreamWorker) -> Flask:
    """Create and configure Flask application with all routes."""
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        """Serve dashboard UI page."""
        return Response(_build_dashboard_html(), mimetype="text/html")

    @app.get("/video")
    def video_stream() -> Response:
        """Serve MJPEG live stream."""
        return Response(
            _generate_mjpeg(stream_worker),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/networks")
    def api_networks() -> Response:
        """Return latest scanned WiFi networks as JSON."""
        return jsonify(runtime.scanner.get_networks())

    @app.get("/api/status")
    def api_status() -> Response:
        """Return runtime status metrics for dashboard polling."""
        with runtime.lock:
            data = {
                "fps": runtime.fps,
                "signal_strength_dbm": runtime.dominant_dbm,
                "combined_intensity": runtime.combined_intensity,
                "smoothed_intensity": runtime.smoothed_intensity,
                "uptime_seconds": time.time() - runtime.start_epoch,
                "frame_count": runtime.frame_count,
                "camera_backend": runtime.camera.backend_name(),
                "mode": runtime.mode,
                "last_frame_epoch": runtime.last_frame_epoch,
            }

        data["network_count"] = len(runtime.scanner.get_networks())
        data["scanner_last_error"] = runtime.scanner.get_last_error()
        data["last_scan_epoch"] = runtime.scanner.get_last_scan_epoch()
        data["scan_latency_ms"] = runtime.scanner.get_last_scan_duration_ms()
        data["process_every_n_frames"] = stream_worker.process_every_n_frames
        return jsonify(data)

    @app.post("/api/scan/now")
    def api_scan_now() -> Response:
        """Trigger an immediate WiFi scan and return diagnostics."""
        networks = runtime.scanner.scan_once()
        error = runtime.scanner.get_last_error()
        if error:
            return jsonify({"ok": False, "error": error, "network_count": len(networks)}), 500

        return jsonify(
            {
                "ok": True,
                "network_count": len(networks),
                "scan_latency_ms": runtime.scanner.get_last_scan_duration_ms(),
                "networks": networks,
            }
        )

    @app.get("/api/mapper/status")
    def api_mapper_status() -> Response:
        """Return current spatial mapper status."""
        status = runtime.mapper.get_status()
        return jsonify(
            {
                "has_position": status.has_position,
                "position_x": status.position_x,
                "position_y": status.position_y,
                "sample_count": status.sample_count,
                "last_sample_epoch": status.last_sample_epoch,
                "last_strength_dbm": status.last_strength_dbm,
            }
        )

    @app.post("/api/mapper/position")
    def api_mapper_position() -> Response:
        """Set active mapper sampling position in normalized coordinates."""
        payload = request.get_json(silent=True) or {}
        x = payload.get("x", 0.0)
        y = payload.get("y", 0.0)

        sample_now_raw = payload.get("sample_now", True)
        sample_now = bool(sample_now_raw)
        if isinstance(sample_now_raw, str):
            sample_now = sample_now_raw.strip().lower() in {"1", "true", "yes", "on"}

        if sample_now:
            with runtime.lock:
                intensity = float(runtime.combined_intensity)
                dominant_dbm = float(runtime.dominant_dbm)

            if intensity <= 0.0:
                networks = runtime.scanner.get_networks()
                intensity = float(get_combined_intensity(networks))
                dominant = get_dominant_network(networks)
                dominant_dbm = float(dominant.get("dbm", -90.0)) if dominant else dominant_dbm

            px, py = runtime.mapper.add_sample_at_position(float(x), float(y), intensity, dominant_dbm)
        else:
            px, py = runtime.mapper.set_position(float(x), float(y))

        status = runtime.mapper.get_status()
        return jsonify(
            {
                "ok": True,
                "position_x": px,
                "position_y": py,
                "sample_now": sample_now,
                "sample_count": status.sample_count,
                "last_sample_epoch": status.last_sample_epoch,
                "last_strength_dbm": status.last_strength_dbm,
            }
        )

    @app.post("/api/mapper/clear")
    def api_mapper_clear() -> Response:
        """Clear mapper heatmap and sample history."""
        runtime.mapper.clear()
        return jsonify({"ok": True})

    @app.get("/api/mapper/heatmap")
    def api_mapper_heatmap() -> Response:
        """Render current mapper heatmap as PNG image."""
        return Response(_build_mapper_png(runtime), mimetype="image/png")

    @app.get("/api/mapper/snapshots")
    def api_mapper_snapshots() -> Response:
        """List available mapper snapshots."""
        return jsonify({"snapshots": _list_snapshot_names()})

    @app.post("/api/mapper/save")
    def api_mapper_save() -> Response:
        """Save current mapper state to a named snapshot."""
        payload = request.get_json(silent=True) or {}
        raw_name = payload.get("name", "")
        safe_name = _sanitize_snapshot_name(str(raw_name))
        target = _snapshot_path(safe_name)

        runtime.mapper.save_to_file(str(target))
        return jsonify(
            {
                "ok": True,
                "name": safe_name,
                "path": str(target),
                "snapshots": _list_snapshot_names(),
            }
        )

    @app.post("/api/mapper/load")
    def api_mapper_load() -> Response:
        """Load mapper state from a named snapshot."""
        payload = request.get_json(silent=True) or {}
        raw_name = str(payload.get("name", "")).strip()
        if not raw_name:
            return jsonify({"ok": False, "error": "Snapshot name is required"}), 400

        safe_name = _sanitize_snapshot_name(raw_name)
        target = _snapshot_path(safe_name)
        if not target.exists():
            return jsonify({"ok": False, "error": f"Snapshot not found: {safe_name}"}), 404

        try:
            runtime.mapper.load_from_file(str(target))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        status = runtime.mapper.get_status()
        return jsonify(
            {
                "ok": True,
                "name": safe_name,
                "sample_count": status.sample_count,
                "has_position": status.has_position,
                "position_x": status.position_x,
                "position_y": status.position_y,
            }
        )

    @app.post("/api/survey/start")
    def api_survey_start() -> Response:
        """Start a guided survey with specified grid dimensions."""
        payload = request.get_json(silent=True) or {}

        try:
            rows = int(payload.get("rows", config.SURVEY_DEFAULT_ROWS))
            cols = int(payload.get("cols", config.SURVEY_DEFAULT_COLS))
            margin = float(payload.get("margin", config.SURVEY_DEFAULT_MARGIN))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Invalid survey parameters"}), 400

        rows = max(int(config.SURVEY_MIN_ROWS), min(int(config.SURVEY_MAX_ROWS), rows))
        cols = max(int(config.SURVEY_MIN_COLS), min(int(config.SURVEY_MAX_COLS), cols))

        serpentine_raw = payload.get("serpentine", config.SURVEY_DEFAULT_SERPENTINE)
        serpentine = bool(serpentine_raw)
        if isinstance(serpentine_raw, str):
            serpentine = serpentine_raw.strip().lower() in {"1", "true", "yes", "on"}

        status = runtime.survey.start(rows=rows, cols=cols, margin=margin, serpentine=serpentine)
        if status.active:
            runtime.mapper.set_position(status.current_target_x, status.current_target_y)

        return jsonify(
            {
                "ok": True,
                "active": status.active,
                "rows": status.rows,
                "cols": status.cols,
                "total_points": status.total_points,
                "current_index": status.current_index,
                "completed_points": status.completed_points,
                "remaining_points": status.remaining_points,
                "target_x": status.current_target_x,
                "target_y": status.current_target_y,
            }
        )

    @app.get("/api/survey/status")
    def api_survey_status() -> Response:
        """Get current survey status."""
        status = runtime.survey.get_status()
        return jsonify(
            {
                "active": status.active,
                "rows": status.rows,
                "cols": status.cols,
                "total_points": status.total_points,
                "current_index": status.current_index,
                "completed_points": status.completed_points,
                "remaining_points": status.remaining_points,
                "completed": status.completed,
                "target_x": status.current_target_x,
                "target_y": status.current_target_y,
                "started_epoch": status.started_epoch,
                "last_capture_epoch": status.last_capture_epoch,
            }
        )

    @app.post("/api/survey/capture")
    def api_survey_capture() -> Response:
        """Record one guided capture and advance to next target."""
        current_status = runtime.survey.get_status()
        if not current_status.active:
            return jsonify({"ok": False, "error": "Survey is not active"}), 400

        with runtime.lock:
            intensity = float(runtime.combined_intensity)
            dominant_dbm = float(runtime.dominant_dbm)

        if intensity <= 0.0:
            networks = runtime.scanner.get_networks()
            intensity = float(get_combined_intensity(networks))
            dominant = get_dominant_network(networks)
            dominant_dbm = float(dominant.get("dbm", -90.0)) if dominant else dominant_dbm

        # Keep mapper aligned with guided survey captures for spatial report exports.
        runtime.mapper.add_sample_at_position(
            current_status.current_target_x,
            current_status.current_target_y,
            intensity,
            dominant_dbm,
        )

        networks = runtime.scanner.get_networks()
        status = runtime.survey.capture(intensity=intensity, dbm_value=dominant_dbm, network_count=len(networks))

        if status.active:
            runtime.mapper.set_position(status.current_target_x, status.current_target_y)

        return jsonify(
            {
                "ok": True,
                "active": status.active,
                "completed_points": status.completed_points,
                "current_index": status.current_index,
                "target_x": status.current_target_x,
                "target_y": status.current_target_y,
            }
        )

    @app.post("/api/survey/next")
    def api_survey_next() -> Response:
        """Advance to next target without recording a capture."""
        status = runtime.survey.next_target()
        if status.active:
            runtime.mapper.set_position(status.current_target_x, status.current_target_y)
        return jsonify(
            {
                "ok": True,
                "active": status.active,
                "current_index": status.current_index,
                "target_x": status.current_target_x,
                "target_y": status.current_target_y,
            }
        )

    @app.post("/api/survey/reset")
    def api_survey_reset() -> Response:
        """Stop survey and clear all captures."""
        status = runtime.survey.reset()
        return jsonify({"ok": True, "active": status.active})

    @app.get("/api/survey/data")
    def api_survey_data() -> Response:
        """Export survey captures as JSON."""
        import json

        captures = runtime.survey.get_captures()
        points = runtime.survey.get_points()
        status = runtime.survey.get_status()

        export_data = {
            "survey": {
                "rows": status.rows,
                "cols": status.cols,
                "total_points": status.total_points,
                "completed": status.completed,
                "started_epoch": status.started_epoch,
            },
            "points": points,
            "captures": captures,
        }

        response = Response(json.dumps(export_data, indent=2), mimetype="application/json")
        response.headers["Content-Disposition"] = f'attachment; filename="{config.SURVEY_EXPORT_PREFIX}.json"'
        return response

    @app.get("/api/report/png")
    def api_report_png() -> Response:
        """Export current room heatmap as PNG."""
        response = Response(_build_mapper_png(runtime), mimetype="image/png")
        response.headers["Content-Disposition"] = f'attachment; filename="{config.SURVEY_EXPORT_PREFIX}_map.png"'
        return response

    @app.get("/api/report/csv")
    def api_report_csv() -> Response:
        """Export survey data as CSV."""
        import csv
        import io

        captures = runtime.survey.get_captures()
        if not captures:
            return Response("x,y,target_index,intensity,dbm,network_count,timestamp\n", mimetype="text/csv")

        output = io.StringIO()
        writer = csv.DictWriter(
            output, fieldnames=["target_index", "x", "y", "intensity", "dbm", "network_count", "timestamp"]
        )
        writer.writeheader()
        for cap in captures:
            writer.writerow(
                {
                    "target_index": int(cap.get("target_index", 0)),
                    "x": float(cap.get("x", 0.0)),
                    "y": float(cap.get("y", 0.0)),
                    "intensity": float(cap.get("intensity", 0.0)),
                    "dbm": float(cap.get("dbm", -90.0)),
                    "network_count": int(cap.get("network_count", 0)),
                    "timestamp": float(cap.get("timestamp", 0.0)),
                }
            )

        response = Response(output.getvalue(), mimetype="text/csv")
        response.headers["Content-Disposition"] = f'attachment; filename="{config.SURVEY_EXPORT_PREFIX}.csv"'
        return response

    # ---------------------------------------------------------------
    # 3D RF Digital Twin API endpoints
    # ---------------------------------------------------------------

    @app.get("/api/rf/pose")
    def api_rf_pose() -> Response:
        """Return current Visual Odometry pose estimate."""
        with runtime.lock:
            data = {
                "x": runtime.pose_x,
                "y": runtime.pose_y,
                "z": runtime.pose_z,
                "confidence": runtime.pose_confidence,
                "status": runtime.pose_status,
                "tracked_points": runtime.pose_tracked_points,
            }
        return jsonify(data)

    @app.post("/api/rf/reset_pose")
    def api_rf_reset_pose() -> Response:
        """Reset Visual Odometry pose to world origin."""
        runtime.pose_estimator.reset()
        return jsonify({"ok": True, "message": "Pose reset to origin."})

    @app.get("/api/rf/ssids")
    def api_rf_ssids() -> Response:
        """Return list of SSIDs seen in the RF database."""
        ssids = runtime.rf_db.get_ssids()
        status = runtime.rf_db.get_status()
        return jsonify({
            "ssids": ssids,
            "total_samples": status.total_samples,
            "ssid_count": status.ssid_count,
            "last_sample_epoch": status.last_sample_epoch,
        })

    @app.get("/api/rf/samples")
    def api_rf_samples() -> Response:
        """Return RF samples as JSON, optionally filtered by SSID."""
        ssid = request.args.get("ssid", None)
        max_items = request.args.get("max", 500, type=int)
        samples = runtime.rf_db.to_dict_list(ssid=ssid, max_items=max_items)
        return jsonify({"count": len(samples), "samples": samples})

    @app.get("/api/rf/heatmap")
    def api_rf_heatmap() -> Response:
        """Render RF heatmap as PNG for the selected SSID and interpolation method."""
        ssid = request.args.get("ssid", None)
        method = request.args.get("method", config.INTERPOLATION_METHOD)
        samples = runtime.rf_db.get_samples(ssid=ssid)

        view_w = int(config.MAPPER_VIEW_WIDTH)
        view_h = int(config.MAPPER_VIEW_HEIGHT)

        if len(samples) < 2:
            # Not enough data — return blank with message
            blank = np.zeros((view_h, view_w, 3), dtype=np.uint8)
            n = runtime.rf_db.sample_count(ssid=ssid)
            msg = f"Samples: {n} (need ≥2)"
            cv2.putText(blank, msg, (10, view_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 150, 255), 1)
            ok, enc = cv2.imencode(".png", blank)
            return Response(enc.tobytes() if ok else b"", mimetype="image/png")

        grid = runtime.rf_interpolator.interpolate_2d(
            samples, method=method,
            x_range=None, y_range=None,
        )
        grid_uint8 = (np.clip(grid, 0.0, 1.0) * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(grid_uint8, cv2.COLORMAP_JET)
        color_resized = cv2.resize(color, (view_w, view_h), interpolation=cv2.INTER_LINEAR)

        # Draw sample dots
        xs = [s.x for s in samples]
        ys = [s.y for s in samples]
        if xs and ys:
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            x_span = max(x_hi - x_lo, 0.01)
            y_span = max(y_hi - y_lo, 0.01)
            for s in samples[-100:]:
                px = int((s.x - x_lo) / x_span * (view_w - 1))
                py = int((s.y - y_lo) / y_span * (view_h - 1))
                cv2.circle(color_resized, (px, py), 3, (255, 255, 255), -1)

        # Label
        label = f"{ssid or 'All'} | {method} | n={len(samples)}"
        cv2.putText(color_resized, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)

        ok, enc = cv2.imencode(".png", color_resized)
        return Response(enc.tobytes() if ok else b"", mimetype="image/png")

    @app.get("/api/rf/export/csv")
    def api_rf_export_csv() -> Response:
        """Export all RF samples as a downloadable CSV file."""
        import csv as _csv
        import io
        ssid = request.args.get("ssid", None)
        samples = runtime.rf_db.get_samples(ssid=ssid)
        output = io.StringIO()
        writer = _csv.DictWriter(
            output,
            fieldnames=["x", "y", "z", "ssid", "bssid", "dbm", "freq", "timestamp"]
        )
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "x": f"{s.x:.4f}", "y": f"{s.y:.4f}", "z": f"{s.z:.4f}",
                "ssid": s.ssid, "bssid": s.bssid,
                "dbm": f"{s.dbm:.2f}", "freq": s.freq,
                "timestamp": f"{s.timestamp:.3f}",
            })
        resp = Response(output.getvalue(), mimetype="text/csv")
        resp.headers["Content-Disposition"] = 'attachment; filename="rf_samples.csv"'
        return resp

    @app.post("/api/rf/clear")
    def api_rf_clear() -> Response:
        """Clear RF database samples."""
        runtime.rf_db.clear()
        return jsonify({"ok": True})

    @app.get("/api/rf/db_status")
    def api_rf_db_status() -> Response:
        """Return RF database statistics."""
        status = runtime.rf_db.get_status()
        return jsonify({
            "total_samples": status.total_samples,
            "ssid_count": status.ssid_count,
            "ssids": status.ssids,
            "last_sample_epoch": status.last_sample_epoch,
            "last_position": {
                "x": status.last_position_x,
                "y": status.last_position_y,
                "z": status.last_position_z,
            },
        })

    # ---------------------------------------------------------------
    # Floor plan + 3D rendering routes
    # ---------------------------------------------------------------

    @app.get("/api/rf/floorplan")
    def api_rf_floorplan_get() -> Response:
        """Return the current floor plan JSON (auto-build if needed)."""
        global _floor_plan_cache
        samples = runtime.rf_db.to_dict_list()

        with _floor_plan_lock:
            if _floor_plan_cache is None or len(samples) < 2:
                if len(samples) >= 2:
                    try:
                        rec = FloorPlanReconstructor()
                        _floor_plan_cache = rec.from_samples(samples)
                    except Exception as exc:
                        LOGGER.warning("Floor plan build failed: %s", exc)
                        return jsonify({"ok": False, "error": str(exc)}), 500
                else:
                    return jsonify({
                        "ok": False,
                        "error": "Need at least 2 RF samples to build floor plan",
                        "sample_count": len(samples),
                    }), 400

            plan = _floor_plan_cache

        plan_dict = plan.to_dict()
        plan_dict["ok"] = True
        plan_dict["sample_count"] = len(samples)
        return jsonify(plan_dict)

    @app.post("/api/rf/floorplan/rebuild")
    def api_rf_floorplan_rebuild() -> Response:
        """Force-rebuild floor plan from current RF database."""
        global _floor_plan_cache
        samples = runtime.rf_db.to_dict_list()
        if len(samples) < 2:
            return jsonify({"ok": False, "error": "Not enough samples"}), 400
        try:
            rec = FloorPlanReconstructor()
            plan = rec.from_samples(samples)
            with _floor_plan_lock:
                _floor_plan_cache = plan
            return jsonify({"ok": True, "message": "Floor plan rebuilt.",
                            "bounds": {"x_min": plan.x_min, "x_max": plan.x_max,
                                       "y_min": plan.y_min, "y_max": plan.y_max},
                            "walk_path_points": len(plan.walk_path),
                            "ap_count": len(plan.ap_locations)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/rf/render3d")
    def api_rf_render3d() -> Response:
        """Render Ekahau-style 3D isometric heatmap PNG on demand."""
        global _floor_plan_cache
        ssid = request.args.get("ssid", None)
        grid = request.args.get("grid", 50, type=int)

        samples = runtime.rf_db.to_dict_list(ssid=ssid)
        if len(samples) < 2:
            return jsonify({"ok": False,
                            "error": f"Need ≥2 samples (have {len(samples)})"}), 400

        # Get or build floor plan
        with _floor_plan_lock:
            if _floor_plan_cache is None:
                try:
                    rec = FloorPlanReconstructor()
                    _floor_plan_cache = rec.from_samples(runtime.rf_db.to_dict_list())
                except Exception as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 500
            plan = _floor_plan_cache

        try:
            renderer = Ekahau3DRenderer(
                floor_plan=plan,
                samples=samples,
                grid_resolution=int(np.clip(grid, 20, 100)),
                ssid_filter=ssid,
            )
            png_bytes = renderer.render_png()
            if not png_bytes:
                return jsonify({"ok": False, "error": "Render returned empty bytes"}), 500
            resp = Response(png_bytes, mimetype="image/png")
            resp.headers["Content-Disposition"] = 'attachment; filename="rf_heatmap_3d.png"'
            resp.headers["Cache-Control"] = "no-cache"
            return resp
        except Exception as exc:
            LOGGER.exception("3D render failed")
            return jsonify({"ok": False, "error": str(exc)}), 500

    # ---------------------------------------------------------------
    # Persistent Room Layouts (multiple named layouts on Pi)
    # Stored in room_layouts.json: { active, layouts: { name: {rooms, nextId} } }
    # ---------------------------------------------------------------
    _ROOM_LAYOUTS_FILE = Path(__file__).parent / "room_layouts.json"
    _OLD_LAYOUT_FILE   = Path(__file__).parent / "room_layout.json"
    _DEFAULT_ROOMS = [
        {"id": 0, "name": "Living Room", "x": 0, "y": 0, "w": 6, "d": 5, "h": 2.6},
        {"id": 1, "name": "Bedroom",     "x": 6, "y": 0, "w": 4, "d": 5, "h": 2.6},
        {"id": 2, "name": "Kitchen",     "x": 0, "y": 5, "w": 10, "d": 3.5, "h": 2.6},
    ]

    def _load_all_layouts() -> dict:
        """Load all layouts from disk, migrating old single-layout file if needed."""
        import json
        if _ROOM_LAYOUTS_FILE.exists():
            try:
                return json.loads(_ROOM_LAYOUTS_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Migrate from old single-layout file
        default_layout = {"rooms": _DEFAULT_ROOMS, "nextId": 3}
        if _OLD_LAYOUT_FILE.exists():
            try:
                default_layout = json.loads(_OLD_LAYOUT_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "active": "My Home",
            "layouts": {"My Home": default_layout}
        }

    def _save_all_layouts(data: dict):
        import json
        _ROOM_LAYOUTS_FILE.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    @app.get("/api/rooms/layouts")
    def list_room_layouts() -> Response:
        """List all saved layout names and which is active."""
        data = _load_all_layouts()
        return jsonify({
            "ok": True,
            "active": data.get("active", "My Home"),
            "names": list(data.get("layouts", {}).keys())
        })

    @app.get("/api/rooms/layout")
    def get_room_layout() -> Response:
        """Return the active layout, or a specific one via ?name=..."""
        data = _load_all_layouts()
        name = request.args.get("name", data.get("active", "My Home"))
        layouts = data.get("layouts", {})
        if name in layouts:
            entry = layouts[name]
            return jsonify({"ok": True, "name": name,
                            "rooms": entry.get("rooms", _DEFAULT_ROOMS),
                            "nextId": entry.get("nextId", 3)})
        return jsonify({"ok": True, "name": name, "rooms": _DEFAULT_ROOMS, "nextId": 3})

    @app.post("/api/rooms/layout")
    def save_room_layout() -> Response:
        """Save rooms under a layout name. Body: {name, rooms, nextId}."""
        body = request.get_json(silent=True) or {}
        name = body.get("name", "My Home")
        rooms = body.get("rooms", [])
        next_id = body.get("nextId", 3)
        try:
            data = _load_all_layouts()
            data.setdefault("layouts", {})[name] = {"rooms": rooms, "nextId": next_id}
            data["active"] = name
            _save_all_layouts(data)
            LOGGER.info("Room layout '%s' saved (%d rooms)", name, len(rooms))
            return jsonify({"ok": True, "saved": len(rooms)})
        except Exception as exc:
            LOGGER.exception("Failed to save room layout")
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/api/rooms/layout/delete")
    def delete_room_layout() -> Response:
        """Delete a named layout. Body: {name}."""
        body = request.get_json(silent=True) or {}
        name = body.get("name", "")
        if not name:
            return jsonify({"ok": False, "error": "No name provided"}), 400
        try:
            data = _load_all_layouts()
            layouts = data.get("layouts", {})
            if name in layouts:
                del layouts[name]
            # If we deleted the active one, switch to the first remaining
            if data.get("active") == name:
                data["active"] = next(iter(layouts), "My Home")
            _save_all_layouts(data)
            return jsonify({"ok": True, "active": data["active"],
                            "names": list(layouts.keys())})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.post("/api/rooms/layout/rename")
    def rename_room_layout() -> Response:
        """Rename a layout. Body: {oldName, newName}."""
        body = request.get_json(silent=True) or {}
        old = body.get("oldName", "")
        new = body.get("newName", "").strip()
        if not old or not new:
            return jsonify({"ok": False, "error": "Names required"}), 400
        try:
            data = _load_all_layouts()
            layouts = data.get("layouts", {})
            if old in layouts:
                layouts[new] = layouts.pop(old)
                if data.get("active") == old:
                    data["active"] = new
                _save_all_layouts(data)
            return jsonify({"ok": True, "names": list(layouts.keys()),
                            "active": data.get("active", new)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/3dmap")
    def page_3dmap() -> Response:
        """Serve the Three.js interactive 3D WiFi Digital Twin page."""
        viewer_path = Path(__file__).parent / "threejs_viewer.html"
        if viewer_path.exists():
            return Response(viewer_path.read_text(encoding="utf-8"), mimetype="text/html")
        # Fallback: old matplotlib canvas page
        return Response(_build_3dmap_html(), mimetype="text/html")

    # ---------------------------------------------------------------
    # Holistic 3D Scene Reconstruction API
    # (adapted from DeepPanoContext / holistic_scene_parsing concepts)
    # Raspberry Pi 5 — CPU-only pipeline
    # ---------------------------------------------------------------

    @app.get("/scene")
    def page_scene_viewer() -> Response:
        """Serve the holistic 3D scene viewer (real room geometry + furniture)."""
        viewer_path = Path(__file__).parent / "scene_3d_viewer.html"
        if viewer_path.exists():
            return Response(viewer_path.read_text(encoding="utf-8"), mimetype="text/html")
        return Response("<h2>scene_3d_viewer.html not found</h2>", mimetype="text/html"), 404

    @app.post("/api/scene/start")
    def api_scene_start() -> Response:
        """Begin a new 4-shot capture session (clears previous shots)."""
        mgr = _get_scene_capture()
        status = mgr.start_session()
        return jsonify({
            "ok": True,
            "state": status.state,
            "shots_remaining": status.shots_remaining,
        })

    @app.post("/api/scene/shot/<direction>")
    def api_scene_shot(direction: str) -> Response:
        """Capture a directional shot using the current live camera frame.

        direction must be one of: N, E, S, W
        The latest frame from the stream worker is used.
        """
        mgr = _get_scene_capture()
        if mgr.get_status().state != "capturing":
            mgr.start_session()

        # Grab latest frame from stream worker
        with runtime.lock:
            latest = getattr(runtime, "latest_frame", None)

        if latest is None or latest.size == 0:
            # Fallback: read directly from camera
            # CameraSource.read() returns Optional[np.ndarray] (not a tuple!)
            latest = runtime.camera.read()
            if latest is None:
                return jsonify({"ok": False, "error": "No camera frame available. Is the camera streaming?"}), 503

        status = mgr.take_shot(direction.upper(), latest)
        return jsonify({
            "ok": True,
            "direction": direction.upper(),
            "state": status.state,
            "shots_taken": status.shots_taken,
            "shots_remaining": status.shots_remaining,
            "progress_pct": status.progress_pct,
        })

    @app.post("/api/scene/reconstruct")
    def api_scene_reconstruct() -> Response:
        """Trigger background room reconstruction from captured shots.

        Pass walk_path and rf_samples from the current VO + RF database
        so the reconstructed room includes real RF data.
        """
        mgr = _get_scene_capture()
        # Attach current VO walk path
        samples_list = runtime.rf_db.to_dict_list()
        walk = []
        try:
            plan_samples = samples_list
            if plan_samples:
                walk = [[float(s["x"]), float(s["y"]), float(s["z"])] for s in plan_samples[-200:]]
        except Exception:
            walk = []

        status = mgr.reconstruct(walk_path=walk, rf_samples=samples_list)
        return jsonify({
            "ok": True,
            "state": status.state,
            "progress_pct": status.progress_pct,
        })

    @app.get("/api/scene/status")
    def api_scene_status() -> Response:
        """Return current capture / reconstruction status."""
        mgr = _get_scene_capture()
        s = mgr.get_status()
        return jsonify({
            "state": s.state,
            "shots_taken": s.shots_taken,
            "shots_remaining": s.shots_remaining,
            "progress_pct": s.progress_pct,
            "error": s.error,
            "has_geometry": s.has_geometry,
            "room_width":   round(s.room_width, 2),
            "room_depth":   round(s.room_depth, 2),
            "room_height":  round(s.room_height, 2),
            "furniture_count": s.furniture_count,
        })

    @app.get("/api/scene/geometry")
    def api_scene_geometry() -> Response:
        """Return room_geometry.json for the Three.js scene viewer."""
        import json as _json
        mgr = _get_scene_capture()
        geom = mgr.get_geometry_dict()
        if geom is None:
            # Try loading from disk
            if mgr.load_geometry_from_disk():
                geom = mgr.get_geometry_dict()
        if geom is None:
            return jsonify({"ok": False, "error": "No scene geometry available. Please run a capture and reconstruct first."}), 404
        # Attach latest RF samples for real-time overlay
        geom["rf_samples"] = runtime.rf_db.to_dict_list(max_items=500)
        geom["ok"] = True
        return jsonify(geom)

    @app.get("/api/scene/preview/<direction>")
    def api_scene_preview(direction: str) -> Response:
        """Return JPEG thumbnail of a captured shot."""
        mgr = _get_scene_capture()
        jpeg = mgr.get_preview_jpeg(direction.upper(), max_size=320)
        if jpeg is None:
            return Response(b"", status=204)
        return Response(jpeg, mimetype="image/jpeg")

    @app.post("/api/scene/reset")
    def api_scene_reset() -> Response:
        """Cancel capture session and clear all data."""
        mgr = _get_scene_capture()
        mgr.cancel()
        return jsonify({"ok": True, "state": "idle"})

    # ---------------------------------------------------------------
    # IMU Sensor API
    # ---------------------------------------------------------------

    @app.get("/api/imu/orientation")
    def api_imu_orientation() -> Response:
        """Return real-time IMU orientation angles (Yaw, Pitch, Roll)."""
        reader = _get_imu_reader()
        ori = reader.get_orientation()
        return jsonify({
            "ok": True,
            "yaw": ori.yaw,
            "pitch": ori.pitch,
            "roll": ori.roll,
            "yaw_rad": ori.yaw_rad,
            "hardware_detected": ori.hardware_detected,
            "timestamp": ori.timestamp,
        })

    @app.get("/api/imu/debug")
    def api_imu_debug() -> Response:
        """Return raw IMU readings for hardware verification."""
        reader = _get_imu_reader()
        raw = reader._read_raw_data()
        ori = reader.get_orientation()
        if raw is None:
            return jsonify({
                "ok": False,
                "hardware_detected": reader.hardware_detected,
                "mode": "simulation",
                "orientation": {"yaw": ori.yaw, "pitch": ori.pitch, "roll": ori.roll},
                "message": "IMU hardware not responding. Check wiring (SDA=GPIO2, SCL=GPIO3) and run: pip install smbus2",
            })
        ax, ay, az, gx, gy, gz = raw
        return jsonify({
            "ok": True,
            "hardware_detected": reader.hardware_detected,
            "i2c_addr": f"0x{reader._addr:02X}",
            "mode": "hardware",
            "raw": {
                "accel_g":  {"x": round(ax, 4), "y": round(ay, 4), "z": round(az, 4)},
                "gyro_dps": {"x": round(gx, 4), "y": round(gy, 4), "z": round(gz, 4)},
            },
            "orientation": {"yaw": ori.yaw, "pitch": ori.pitch, "roll": ori.roll},
        })

    return app


def run_server(
    host: str = config.SERVER_HOST,
    port: int = config.SERVER_PORT,
    mode: str = config.DEFAULT_MODE,
    interface: str = config.WIFI_INTERFACE,
    scan_interval: float = config.SCAN_INTERVAL,
    camera_index: int = config.CAMERA_INDEX,
    camera_width: int = config.CAMERA_WIDTH,
    camera_height: int = config.CAMERA_HEIGHT,
    camera_fps: int = config.CAMERA_FPS,
    jpeg_quality: int = config.JPEG_QUALITY,
    process_every_n_frames: int = config.PROCESS_EVERY_N_FRAMES,
) -> None:
    """Initialize modules and run Flask server."""
    _configure_logging()
    cv2.setNumThreads(int(config.OPENCV_NUM_THREADS))
    cv2.setUseOptimized(True)

    scanner = start_scanner(interface=interface, scan_interval=scan_interval)
    camera = CameraSource(
        index=camera_index,
        width=camera_width,
        height=camera_height,
        fps=camera_fps,
    )

    if not camera.open():
        LOGGER.warning("Camera could not be opened; stream will use fallback frames.")

    runtime = RuntimeContext(
        scanner=scanner,
        camera=camera,
        field_model=RFFieldModel(
            frame_width=camera_width,
            frame_height=camera_height,
            field_scale=config.FIELD_SCALE,
            sigma=config.GAUSSIAN_SIGMA,
            decay_factor=config.DECAY_FACTOR,
        ),
        smoother=SignalSmoother(alpha=config.SMOOTHING_ALPHA),
        mapper=SpatialWifiMapper(
            grid_width=config.MAPPER_GRID_WIDTH,
            grid_height=config.MAPPER_GRID_HEIGHT,
            sigma=config.MAPPER_SIGMA,
            decay=config.MAPPER_DECAY,
        ),
        pose_estimator=PoseEstimator(
            width=camera_width,
            height=camera_height,
            max_features=config.VO_MAX_FEATURES,
            min_match_count=config.VO_MIN_MATCH_COUNT,
            scale_factor=config.VO_SCALE_FACTOR,
            confidence_threshold=config.VO_POSE_CONFIDENCE_THRESHOLD,
        ),
        rf_db=RFDatabase(
            max_samples=config.RF_DB_MAX_SAMPLES,
            min_displacement=config.RF_SAMPLE_MIN_DISTANCE,
        ),
        rf_interpolator=RFInterpolator(
            grid_size=config.INTERPOLATION_GRID_SIZE,
            method=config.INTERPOLATION_METHOD,
            idw_power=config.IDW_POWER,
        ),
        mode=mode,
        frame_width=camera_width,
        frame_height=camera_height,
        jpeg_quality=jpeg_quality,
    )

    stream_worker = StreamWorker(runtime=runtime, process_every_n_frames=process_every_n_frames)
    stream_worker.start()

    app = create_app(runtime, stream_worker)

    try:
        LOGGER.info("Starting server on http://%s:%d", host, port)
        app.run(host=host, port=port, threaded=True)
    finally:
        stream_worker.stop()
        camera.release()
        scanner.stop_scanner()


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for standalone app execution."""
    parser = argparse.ArgumentParser(description="WiFi visualizer Flask MJPEG server")
    parser.add_argument("--host", default=config.SERVER_HOST, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Bind port (default: 5000)")
    parser.add_argument("--mode", default=config.DEFAULT_MODE, help="Visualization mode")
    parser.add_argument("--interface", default=config.WIFI_INTERFACE, help="WiFi interface")
    parser.add_argument("--scan-interval", type=float, default=config.SCAN_INTERVAL, help="WiFi scan interval")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=config.JPEG_QUALITY,
        help="JPEG quality for MJPEG stream",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=config.PROCESS_EVERY_N_FRAMES,
        help="Run full visualization every N frames (default: 2)",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for running Flask app directly."""
    args = _parse_args()
    run_server(
        host=args.host,
        port=args.port,
        mode=args.mode,
        interface=args.interface,
        scan_interval=args.scan_interval,
        jpeg_quality=args.jpeg_quality,
        process_every_n_frames=args.process_every_n_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
