"""Stage 0 hardware verification for Raspberry Pi Zero 2W WiFi visualizer.

This script checks:
1. USB camera capture by reading a single frame.
2. WiFi scan capability via iw dev <iface> scan.

Run this script over SSH on the Raspberry Pi after installing dependencies.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")

import cv2

LOGGER = logging.getLogger("test_hardware")


def configure_logging() -> None:
    """Configure console logging with simple, readable output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _camera_backends() -> Iterable[Tuple[str, Optional[int]]]:
    """Yield camera backend options in preferred order.

    Many Logitech USB cameras work best with V4L2 on Raspberry Pi.
    """
    cap_v4l2 = getattr(cv2, "CAP_V4L2", None)
    if cap_v4l2 is not None:
        return [("CAP_V4L2", cap_v4l2)]
    return [("DEFAULT", None)]


def _discover_usb_camera_indices() -> Tuple[int, ...]:
    """Discover likely USB webcam indices from v4l2-ctl output."""
    command = ["v4l2-ctl", "--list-devices"]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return tuple()

    if result.returncode != 0:
        return tuple()

    indices = []
    current_is_usb_camera = False
    for line in (result.stdout or "").splitlines():
        if not line.strip():
            continue

        # Device headers are non-indented; paths are indented.
        if line[:1].isspace():
            if current_is_usb_camera and "/dev/video" in line:
                node = line.strip()
                suffix = node.replace("/dev/video", "")
                if suffix.isdigit():
                    indices.append(int(suffix))
            continue

        header = line.lower()
        current_is_usb_camera = "usb" in header or "webcam" in header or "camera" in header

    return tuple(sorted(set(indices)))


def _camera_indices(camera_index: int) -> Iterable[int]:
    """Yield likely camera indices in preferred order."""
    seen = set()

    candidates = [camera_index, camera_index + 1]
    candidates.extend(_discover_usb_camera_indices())

    for idx in candidates:
        if idx in seen:
            continue
        if not Path(f"/dev/video{idx}").exists():
            continue
        seen.add(idx)
        yield idx


def _preferred_capture_formats(device_path: Path) -> Tuple[Optional[str], ...]:
    """Discover likely pixel formats for a camera node via v4l2-ctl.

    Falls back to common formats if probing is unavailable.
    """
    fallback = ("MJPG", "YUYV", None)
    command = ["v4l2-ctl", "--device", str(device_path), "--list-formats-ext"]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return fallback

    if result.returncode != 0:
        return fallback

    probe_output = (result.stdout or "").upper()
    detected = []
    if "MJPG" in probe_output:
        detected.append("MJPG")
    if "YUYV" in probe_output or "YUY2" in probe_output:
        detected.append("YUYV")

    if not detected:
        return fallback

    detected.append(None)
    return tuple(detected)


def _preferred_resolutions(width: int, height: int) -> Tuple[Tuple[int, int], ...]:
    """Yield unique resolutions to try for capture initialization."""
    options = [(width, height), (1280, 720), (640, 480)]
    ordered = []
    seen = set()
    for item in options:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _configure_camera(cap: cv2.VideoCapture, width: int, height: int, fps: int, fourcc_name: Optional[str]) -> None:
    """Apply capture settings for stable USB camera frame reads."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000.0)
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000.0)

    if fourcc_name:
        cap.set(cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*fourcc_name)))


def _read_frame_with_retry(cap: cv2.VideoCapture, timeout_seconds: float = 5.0) -> Optional[object]:
    """Attempt multiple reads to allow camera warm-up before failing."""
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        grabbed = cap.grab()
        if grabbed:
            ok, frame = cap.retrieve()
        else:
            ok, frame = cap.read()

        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return frame
        time.sleep(0.05)

    return None


def _capture_frame_with_ffmpeg(device_path: Path, width: int, height: int, fps: int = 15) -> Optional[Tuple[int, ...]]:
    """Capture a single frame with ffmpeg and return decoded frame shape.

    This is a robust fallback for webcams that intermittently fail under OpenCV
    on Raspberry Pi while still working with V4L2.
    """
    with tempfile.NamedTemporaryFile(prefix="camera_probe_", suffix=".jpg", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "v4l2",
        "-framerate",
        str(fps),
        "-video_size",
        f"{width}x{height}",
        "-i",
        str(device_path),
        "-frames:v",
        "1",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None

        frame = cv2.imread(str(output_path))
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        return tuple(frame.shape)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass


def _query_v4l2_dimensions(device_path: Path, fallback_width: int, fallback_height: int) -> Tuple[int, int]:
    """Read active camera dimensions from v4l2-ctl output."""
    command = ["v4l2-ctl", "--device", str(device_path), "--get-fmt-video"]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return fallback_width, fallback_height

    if result.returncode != 0:
        return fallback_width, fallback_height

    text = result.stdout or ""
    match = re.search(r"Width/Height\s*:\s*(\d+)\s*/\s*(\d+)", text)
    if not match:
        return fallback_width, fallback_height

    width = int(match.group(1))
    height = int(match.group(2))
    return width, height


def _capture_frame_with_v4l2ctl(device_path: Path, width: int, height: int, fps: int = 15) -> Optional[Tuple[int, ...]]:
    """Capture one frame using v4l2-ctl stream mode and return frame shape.

    This avoids OpenCV backend issues and validates camera stream availability.
    """
    with tempfile.NamedTemporaryFile(prefix="camera_probe_", suffix=".raw", delete=False) as tmp_file:
        output_path = Path(tmp_file.name)

    set_command = [
        "v4l2-ctl",
        "--device",
        str(device_path),
        f"--set-fmt-video=width={width},height={height},pixelformat=YUYV",
        f"--set-parm={fps}",
    ]

    stream_command = [
        "v4l2-ctl",
        "--device",
        str(device_path),
        "--stream-mmap=3",
        "--stream-count=1",
        f"--stream-to={output_path}",
    ]

    try:
        subprocess.run(
            set_command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

        result = subprocess.run(
            stream_command,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if result.returncode != 0:
            return None

        if not output_path.exists() or output_path.stat().st_size <= 0:
            return None

        active_width, active_height = _query_v4l2_dimensions(device_path, width, height)
        return active_height, active_width, 3
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass


def test_camera(camera_index: int, width: int, height: int) -> Tuple[bool, Optional[Tuple[int, ...]], str]:
    """Open camera, capture one frame, and return result details.

    Returns:
        tuple:
            - success flag
            - captured frame shape if available
            - backend name used (or failure reason)
    """
    available_indices = list(_camera_indices(camera_index))
    if not available_indices:
        return False, None, "No usable webcam index found (/dev/videoX)"

    # Try native v4l2-ctl capture first for reliable hardware-level validation.
    for source_idx in available_indices:
        source_path = Path(f"/dev/video{source_idx}")
        for test_width, test_height in _preferred_resolutions(width, height):
            v4l2_shape = _capture_frame_with_v4l2ctl(source_path, test_width, test_height, fps=15)
            if v4l2_shape is not None:
                return (
                    True,
                    v4l2_shape,
                    f"V4L2CTL source={source_path} requested={test_width}x{test_height}",
                )

    # First try ffmpeg one-shot capture, which is usually more reliable on Pi.
    for source_idx in available_indices:
        source_path = Path(f"/dev/video{source_idx}")
        for test_width, test_height in _preferred_resolutions(width, height):
            ffmpeg_shape = _capture_frame_with_ffmpeg(source_path, test_width, test_height, fps=15)
            if ffmpeg_shape is not None:
                return (
                    True,
                    ffmpeg_shape,
                    f"FFMPEG source={source_path} requested={test_width}x{test_height}",
                )

    for backend_name, backend_id in _camera_backends():
        for source_idx in available_indices:
            source_path = Path(f"/dev/video{source_idx}")
            capture_formats = _preferred_capture_formats(source_path)

            cap = (
                cv2.VideoCapture(source_idx)
                if backend_id is None
                else cv2.VideoCapture(source_idx, backend_id)
            )
            try:
                if not cap.isOpened():
                    continue

                for fourcc_name in capture_formats:
                    for test_width, test_height in _preferred_resolutions(width, height):
                        _configure_camera(cap, test_width, test_height, fps=15, fourcc_name=fourcc_name)
                        time.sleep(0.2)
                        frame = _read_frame_with_retry(cap, timeout_seconds=2.0)
                        if frame is None:
                            continue

                        used_format = fourcc_name if fourcc_name is not None else "AUTO"
                        return (
                            True,
                            tuple(frame.shape),
                            f"{backend_name} index={source_idx} format={used_format} "
                            f"requested={test_width}x{test_height}",
                        )
                continue
            finally:
                cap.release()

    return False, None, "Unable to read frame from camera on all tested backends"


def run_scan_command(command: Sequence[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    """Execute a WiFi scan command with timeout and captured output."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def test_wifi_scan(interface: str, timeout_seconds: int) -> Tuple[bool, str, str]:
    """Run WiFi scan and return success flag, raw output, and source command.

    The function first tries direct iw and then non-interactive sudo (sudo -n) to
    avoid hanging on password prompts in SSH sessions.
    """
    commands = [
        ["iw", "dev", interface, "scan"],
        ["sudo", "-n", "iw", "dev", interface, "scan"],
    ]

    last_error = "Unknown scan failure"

    for command in commands:
        cmd_text = " ".join(command)
        try:
            result = run_scan_command(command, timeout_seconds)
        except FileNotFoundError as exc:
            last_error = f"Command not found: {exc}"
            continue
        except subprocess.TimeoutExpired:
            last_error = f"Scan timeout after {timeout_seconds}s using: {cmd_text}"
            continue

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode == 0 and stdout:
            return True, stdout, cmd_text

        if "Operation not permitted" in stderr or "permission" in stderr.lower():
            last_error = (
                "Permission denied for WiFi scan. Use sudo privileges or grant CAP_NET_ADMIN "
                "to iw."
            )
            continue

        if "No such device" in stderr:
            last_error = f"Interface '{interface}' not found."
            continue

        if "sudo: a password is required" in stderr:
            last_error = (
                "WiFi scan needs privileges. Run the script with sudo, for example: "
                "sudo -E .venv/bin/python test_hardware.py"
            )
            continue

        if stderr:
            last_error = f"Scan command failed ({cmd_text}): {stderr}"
        else:
            last_error = f"Scan command failed ({cmd_text}) with return code {result.returncode}"

    return False, "", last_error


def count_networks(raw_scan_output: str) -> int:
    """Count discovered networks from raw iw scan output."""
    return sum(1 for line in raw_scan_output.splitlines() if line.lstrip().startswith("BSS "))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for hardware test options."""
    parser = argparse.ArgumentParser(description="Test camera and WiFi scan on Raspberry Pi")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Requested capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Requested capture height (default: 480)")
    parser.add_argument("--interface", type=str, default="wlan0", help="WiFi interface name (default: wlan0)")
    parser.add_argument("--timeout", type=int, default=15, help="Scan timeout in seconds (default: 15)")
    return parser.parse_args()


def main() -> int:
    """Run hardware tests and return process exit code."""
    configure_logging()
    args = parse_args()

    LOGGER.info("=== Hardware Test Start ===")
    LOGGER.info("Camera device expected: /dev/video%s", args.camera_index)
    LOGGER.info("WiFi interface: %s", args.interface)

    camera_ok, frame_shape, backend_info = test_camera(args.camera_index, args.width, args.height)
    if camera_ok and frame_shape is not None:
        LOGGER.info("Camera OK: %s", frame_shape)
        LOGGER.info("Camera backend: %s", backend_info)
    else:
        LOGGER.error("Camera FAIL: %s", backend_info)

    wifi_ok, raw_output, wifi_info = test_wifi_scan(args.interface, args.timeout)
    if wifi_ok:
        network_count = count_networks(raw_output)
        LOGGER.info("WiFi scan command used: %s", wifi_info)
        LOGGER.info("WiFi scan raw output:\n%s", raw_output)
        LOGGER.info("WiFi scan OK: found %d networks", network_count)
    else:
        LOGGER.error("WiFi scan FAIL: %s", wifi_info)

    LOGGER.info("=== Hardware Test End ===")

    if camera_ok and wifi_ok:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
