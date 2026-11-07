"""Main entry point for WiFi signal visualizer system.

Run this file to start the full pipeline:
- System info reporting
- Camera readiness check
- WiFi scan readiness check
- Flask MJPEG server launch
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Tuple

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

import app as web_app
import config
from wifi_scanner import WifiScanner

LOGGER = logging.getLogger("main")


def _configure_logging() -> None:
    """Configure process-wide logging format and level."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_resolution(value: str) -> Tuple[int, int]:
    """Parse a resolution string in the form WxH.

    Args:
        value: Resolution string, for example 640x480.

    Returns:
        Tuple of (width, height).
    """
    text = value.lower().strip()
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be in WxH format, for example 640x480")

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution values must be integers") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")

    return width, height


def _get_pi_model() -> str:
    """Read Raspberry Pi model string from system files if available."""
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        try:
            data = model_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = data.replace("\x00", "").strip()
            if cleaned:
                return cleaned
        except OSError:
            pass

    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model"):
                    _, _, value = line.partition(":")
                    value = value.strip()
                    if value:
                        return value
        except OSError:
            pass

    return "Unknown model"


def _get_local_ip() -> str:
    """Determine the primary local IPv4 address for display."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def _log_system_info() -> None:
    """Log system information required by startup sequence."""
    LOGGER.info("System model: %s", _get_pi_model())
    LOGGER.info("Python version: %s", platform.python_version())
    LOGGER.info("OpenCV version: %s", cv2.__version__)


def _test_camera(camera_index: int, width: int, height: int, fps: int) -> bool:
    """Validate that the camera can be opened and read before server launch."""
    camera = web_app.CameraSource(index=camera_index, width=width, height=height, fps=fps)
    try:
        if not camera.open():
            LOGGER.error(
                "Camera test failed for /dev/video%d at %dx%d. Check USB camera connection and permissions.",
                camera_index,
                width,
                height,
            )
            return False

        frame = camera.read()
        if frame is None:
            LOGGER.error("Camera opened but no frame was returned.")
            return False

        LOGGER.info("Camera test OK: shape=%s backend=%s", tuple(frame.shape), camera.backend_name())
        return True
    finally:
        camera.release()


def _test_wifi_scan(interface: str, scan_interval: float) -> None:
    """Run a one-shot WiFi scan test and log warning on failure."""
    scanner = WifiScanner(interface=interface, scan_interval=scan_interval, command_timeout=20)
    networks = scanner.scan_once()
    error = scanner.get_last_error()

    if error:
        LOGGER.warning("WiFi scan test warning: %s", error)
        LOGGER.warning("Continuing startup as requested.")
        return

    LOGGER.info("WiFi scan test OK: found %d network(s)", len(networks))


def _parse_args() -> argparse.Namespace:
    """Parse Stage 6 command-line arguments."""
    parser = argparse.ArgumentParser(description="WiFi signal visualizer main entry point")
    parser.add_argument("--demo", action="store_true", help="Run with simulated camera, WiFi and motion")
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--data-dir", default=".data")
    parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Server port (default: 5000)")
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        help="Camera resolution as WxH (default: 640x480)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=config.SCAN_INTERVAL,
        help="WiFi scan interval in seconds (default: 4)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=config.DEFAULT_MODE,
        choices=["heatmap", "pulse", "labels", "combined", "mapping"],
        help="Visualization mode (default: combined)",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=config.WIFI_INTERFACE,
        help="WiFi interface (default: wlan0)",
    )
    return parser.parse_args()


def main() -> int:
    """Execute startup sequence and launch Flask server."""
    _configure_logging()
    args = _parse_args()

    width, height = args.resolution

    LOGGER.info("=== WiFi Visualizer Startup ===")
    _log_system_info()

    # Sources are opened once by the server. Missing hardware is reported in the UI.

    local_ip = _get_local_ip()
    LOGGER.info("System ready. Open http://%s:%d in your browser", local_ip, args.port)

    try:
        web_app.run_server(
            host=args.host,
            demo=args.demo,
            data_dir=args.data_dir,
            port=args.port,
            mode=args.mode,
            interface=args.interface,
            scan_interval=args.scan_interval,
            camera_index=config.CAMERA_INDEX,
            camera_width=width,
            camera_height=height,
            camera_fps=config.CAMERA_FPS,
            jpeg_quality=config.JPEG_QUALITY,
            process_every_n_frames=config.PROCESS_EVERY_N_FRAMES,
        )
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested by user.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
