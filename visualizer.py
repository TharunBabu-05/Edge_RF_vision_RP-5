"""Frame visualization engine for WiFi signal overlays.

This module renders camera frames with RF field heatmaps, animated pulse
indicators, and top-network text overlays for real-time monitoring.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np

import config
from signal_processor import get_dominant_network, normalize_dbm

LOGGER = logging.getLogger("visualizer")

MODE_HEATMAP_OVERLAY = "heatmap"
MODE_SIGNAL_PULSE = "pulse"
MODE_NETWORK_LABELS = "labels"
MODE_COMBINED = "combined"


def _clamp01(value: float) -> float:
    """Clamp numeric value into the range [0.0, 1.0]."""
    return float(max(0.0, min(1.0, value)))


def _normalize_mode(mode: str) -> str:
    """Normalize visualization mode string with fallback to combined mode."""
    if not mode:
        return MODE_COMBINED

    mode_normalized = str(mode).strip().lower()
    aliases = {
        "heatmap_overlay": MODE_HEATMAP_OVERLAY,
        "signal_pulse": MODE_SIGNAL_PULSE,
        "network_labels": MODE_NETWORK_LABELS,
    }
    mode_normalized = aliases.get(mode_normalized, mode_normalized)

    valid_modes = {
        MODE_HEATMAP_OVERLAY,
        MODE_SIGNAL_PULSE,
        MODE_NETWORK_LABELS,
        MODE_COMBINED,
    }

    if mode_normalized not in valid_modes:
        LOGGER.warning("Unknown mode '%s', falling back to combined.", mode)
        return MODE_COMBINED

    return mode_normalized


def _dbm_to_bgr(dbm_value: float) -> Tuple[int, int, int]:
    """Map dBm to color gradient: weak red -> medium yellow -> strong green."""
    strength = normalize_dbm(dbm_value)

    if strength < 0.5:
        t = strength / 0.5
        red = 255
        green = int(255 * t)
    else:
        t = (strength - 0.5) / 0.5
        red = int(255 * (1.0 - t))
        green = 255

    blue = 0
    return blue, green, red


def _strength_blocks(dbm_value: float, width: int = 6, use_unicode: bool = False) -> str:
    """Create text bar from dBm intensity.

    OpenCV Hershey fonts do not reliably render Unicode block glyphs on all
    platforms, so ASCII bars are used by default for compatibility.
    """
    strength = normalize_dbm(dbm_value)
    filled = int(round(strength * width))
    filled = max(0, min(width, filled))

    if use_unicode:
        return ("█" * filled) + ("░" * (width - filled))
    return ("#" * filled) + ("-" * (width - filled))


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw outlined text for readability over dynamic backgrounds."""
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _prepare_rf_uint8(rf_field: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize RF field if needed and convert from [0,1] float to uint8."""
    if rf_field.ndim != 2:
        raise ValueError("rf_field must be a 2D array")

    if rf_field.shape[1] != target_width or rf_field.shape[0] != target_height:
        rf_resized = cv2.resize(rf_field, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    else:
        rf_resized = rf_field

    rf_clamped = np.clip(rf_resized, 0.0, 1.0)
    return (rf_clamped * 255.0).astype(np.uint8)


def _apply_heatmap_overlay(frame: np.ndarray, rf_field: np.ndarray, alpha: float = config.OVERLAY_ALPHA) -> np.ndarray:
    """Apply COLORMAP_JET overlay on top of camera frame."""
    frame_height, frame_width = frame.shape[:2]
    rf_uint8 = _prepare_rf_uint8(rf_field, frame_width, frame_height)
    heatmap = cv2.applyColorMap(rf_uint8, cv2.COLORMAP_JET)

    overlay_alpha = _clamp01(float(alpha))
    return cv2.addWeighted(frame, 1.0 - overlay_alpha, heatmap, overlay_alpha, 0.0)


def _draw_signal_pulse(frame: np.ndarray, dbm_value: float, tick_seconds: float) -> None:
    """Draw animated concentric pulse rings centered in frame."""
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    strength = normalize_dbm(dbm_value)

    max_radius = max(20, int(min(width, height) * 0.35 * max(0.2, strength)))
    base_phase = tick_seconds * 80.0
    color = _dbm_to_bgr(dbm_value)

    for ring in range(3):
        radius = int((base_phase + ring * (max_radius / 3.0)) % max_radius) + 10
        thickness = 2
        cv2.circle(frame, center, radius, color, thickness, cv2.LINE_AA)

    cv2.circle(frame, center, 6, color, -1, cv2.LINE_AA)


def draw_signal_bar(frame: np.ndarray, dbm_value: float, position: Tuple[int, int]) -> None:
    """Draw a compact signal strength indicator with color and value.

    Args:
        frame: Target BGR frame.
        dbm_value: Signal strength in dBm.
        position: Top-left (x, y) position of the bar widget.
    """
    x, y = int(position[0]), int(position[1])
    width = 180
    height = 18

    strength = normalize_dbm(dbm_value)
    fill_width = int(width * strength)
    color = _dbm_to_bgr(dbm_value)

    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)
    if fill_width > 0:
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

    label = f"Signal {dbm_value:.0f} dBm"
    _draw_text(frame, label, (x, y - 6), (255, 255, 255), scale=0.5, thickness=1)


def draw_network_info(frame: np.ndarray, networks_list: Sequence[Dict[str, Any]]) -> None:
    """Draw top 3 network labels in top-left corner.

    Label format:
        MySSID: -65 dBm ####--
    """
    top_networks = sorted(
        (dict(item) for item in networks_list),
        key=lambda item: float(item.get("dbm", -100.0)),
        reverse=True,
    )[:3]

    panel_x = 10
    panel_y = 20
    line_height = 22

    if not top_networks:
        _draw_text(frame, "No networks", (panel_x, panel_y), (220, 220, 220), scale=0.55)
        return

    for idx, network in enumerate(top_networks):
        ssid = str(network.get("ssid", "<hidden>"))
        try:
            dbm = float(network.get("dbm", -100.0))
        except (TypeError, ValueError):
            dbm = -100.0

        blocks = _strength_blocks(dbm, use_unicode=False)
        color = _dbm_to_bgr(dbm)
        text = f"{ssid}: {dbm:.0f} dBm {blocks}"
        _draw_text(frame, text, (panel_x, panel_y + idx * line_height), color, scale=0.55, thickness=1)


def render(
    frame: np.ndarray,
    rf_field: np.ndarray,
    networks: Sequence[Dict[str, Any]],
    mode: str = config.DEFAULT_MODE,
) -> np.ndarray:
    """Render frame using selected visualization mode.

    Modes:
    - heatmap: RF colormap overlay only
    - pulse: animated concentric pulse only
    - labels: top network labels only
    - combined: heatmap + labels + signal strength indicator
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image array with shape (H, W, 3)")

    output = frame.copy()
    selected_mode = _normalize_mode(mode)

    dominant = get_dominant_network(networks)
    dominant_dbm = float(dominant.get("dbm", -90.0)) if dominant else -90.0

    if selected_mode in (MODE_HEATMAP_OVERLAY, MODE_COMBINED):
        output = _apply_heatmap_overlay(output, rf_field, alpha=config.OVERLAY_ALPHA)

    if selected_mode == MODE_SIGNAL_PULSE:
        _draw_signal_pulse(output, dominant_dbm, tick_seconds=time.monotonic())

    if selected_mode in (MODE_NETWORK_LABELS, MODE_COMBINED):
        draw_network_info(output, networks)

    if selected_mode == MODE_COMBINED:
        height, _ = output.shape[:2]
        draw_signal_bar(output, dominant_dbm, position=(10, max(30, height - 35)))

    return output


def _configure_logging() -> None:
    """Configure readable console logging for standalone test mode."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local visualizer test image generation."""
    parser = argparse.ArgumentParser(description="Visualizer module self-test")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH, help="Frame width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT, help="Frame height")
    parser.add_argument("--mode", type=str, default=config.DEFAULT_MODE, help="Mode: heatmap/pulse/labels/combined")
    parser.add_argument(
        "--output",
        type=str,
        default="visualizer_test.jpg",
        help="Output image path for rendered test frame",
    )
    return parser.parse_args()


def _build_demo_frame(width: int, height: int) -> np.ndarray:
    """Create synthetic BGR frame for module-level visualizer testing."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[..., 0] = (80 + 100 * xx).astype(np.uint8)
    base[..., 1] = (40 + 120 * yy).astype(np.uint8)
    base[..., 2] = (30 + 100 * (1.0 - xx)).astype(np.uint8)
    return base


def _build_demo_rf_field(width: int, height: int) -> np.ndarray:
    """Create synthetic RF field for quick local testing."""
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    field = np.exp(-((xx ** 2) + (yy ** 2)) / 0.2)
    return np.clip(field.astype(np.float32), 0.0, 1.0)


def main() -> int:
    """Generate one rendered demo frame and save it to disk."""
    _configure_logging()
    args = _parse_args()

    frame = _build_demo_frame(args.width, args.height)
    rf_field = _build_demo_rf_field(args.width, args.height)
    networks = [
        {"ssid": "Jack's", "dbm": -50.0, "bssid": "26:d3:23:a8:6d:a0", "freq": 2462},
        {"ssid": "realme 12x 5G", "dbm": -77.0, "bssid": "32:c3:09:51:bd:fe", "freq": 2437},
        {"ssid": "Navin", "dbm": -85.0, "bssid": "3e:9d:4e:04:4b:ee", "freq": 2437},
    ]

    rendered = render(frame, rf_field, networks, mode=args.mode)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), rendered)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    LOGGER.info("Visualizer test image written: %s", output_path)
    LOGGER.info("Mode used: %s", _normalize_mode(args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
