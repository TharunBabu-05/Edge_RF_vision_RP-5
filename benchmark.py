"""Performance benchmark for WiFi visualizer pipeline on Raspberry Pi.

Metrics collected:
- End-to-end processed FPS
- CPU usage percentage
- Memory usage (RSS MB)
- WiFi scan latency (ms)
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
import psutil

import app as web_app
import config
from rf_field import RFFieldModel
from signal_processor import SignalSmoother, get_combined_intensity
from visualizer import render
from wifi_scanner import start_scanner

LOGGER = logging.getLogger("benchmark")


@dataclass
class BenchmarkResult:
    """Container for benchmark output values."""

    fps: float
    cpu_avg_percent: float
    cpu_peak_percent: float
    memory_avg_mb: float
    memory_peak_mb: float
    scan_latency_avg_ms: float
    scan_latency_peak_ms: float


def _configure_logging() -> None:
    """Configure logging for benchmark output."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution string in WxH format."""
    parts = value.lower().strip().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be WxH, for example 640x480")
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution values must be integers") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Resolution values must be positive")

    return width, height


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark run."""
    parser = argparse.ArgumentParser(description="WiFi visualizer performance benchmark")
    parser.add_argument("--duration", type=float, default=30.0, help="Benchmark duration in seconds")
    parser.add_argument("--interface", type=str, default=config.WIFI_INTERFACE, help="WiFi interface")
    parser.add_argument("--scan-interval", type=float, default=config.SCAN_INTERVAL, help="WiFi scan interval")
    parser.add_argument(
        "--resolution",
        type=_parse_resolution,
        default=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        help="Frame resolution as WxH",
    )
    parser.add_argument("--mode", type=str, default=config.DEFAULT_MODE, help="Visualizer mode")
    parser.add_argument("--jpeg-quality", type=int, default=config.JPEG_QUALITY, help="JPEG quality")
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=config.PROCESS_EVERY_N_FRAMES,
        help="Apply full processing every N frames (default: 2)",
    )
    return parser.parse_args()


def _safe_mean(values: Sequence[float]) -> float:
    """Return mean value or 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_blank(width: int, height: int) -> np.ndarray:
    """Create fallback frame when camera data is unavailable."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera frame unavailable", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
    return frame


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    """Run benchmark loop and collect performance metrics."""
    width, height = args.resolution

    cv2.setNumThreads(int(config.OPENCV_NUM_THREADS))
    cv2.setUseOptimized(True)
    logging.getLogger("wifi_scanner").setLevel(logging.WARNING)

    scanner = start_scanner(interface=args.interface, scan_interval=args.scan_interval, command_timeout=20)
    camera = web_app.CameraSource(index=config.CAMERA_INDEX, width=width, height=height, fps=config.CAMERA_FPS)

    if not camera.open():
        raise RuntimeError("Failed to open camera for benchmark")

    smoother = SignalSmoother(alpha=config.SMOOTHING_ALPHA)
    field_model = RFFieldModel(
        frame_width=width,
        frame_height=height,
        field_scale=config.FIELD_SCALE,
        sigma=config.GAUSSIAN_SIGMA,
        decay_factor=config.DECAY_FACTOR,
    )

    blank = _build_blank(width, height)
    process = psutil.Process()

    cpu_samples: List[float] = []
    mem_samples_mb: List[float] = []
    scan_latencies_ms: List[float] = []

    frame_count = 0
    frame_index = 0
    start = time.perf_counter()
    next_sample = start

    networks = scanner.get_networks()
    last_scan_epoch = scanner.get_last_scan_epoch()
    if last_scan_epoch > 0.0:
        scan_latencies_ms.append(scanner.get_last_scan_duration_ms())
    rendered_cache = blank.copy()

    psutil.cpu_percent(interval=None)

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - start
            if elapsed >= float(args.duration):
                break

            scan_epoch = scanner.get_last_scan_epoch()
            if scan_epoch > last_scan_epoch:
                last_scan_epoch = scan_epoch
                networks = scanner.get_networks()
                scan_latencies_ms.append(scanner.get_last_scan_duration_ms())

            frame = camera.read()
            if frame is None:
                frame = blank

            frame_index += 1
            should_process = frame_index == 1 or (frame_index % max(1, int(args.process_every_n_frames)) == 0)

            if should_process:
                combined_intensity = get_combined_intensity(networks)
                smoothed_intensity = smoother.update(combined_intensity)
                field_model.update(smoothed_intensity)
                rf_field = field_model.get_field()
                rendered_cache = render(frame, rf_field, networks, mode=args.mode)

            rendered = rendered_cache

            ok, _encoded = cv2.imencode(
                ".jpg",
                rendered,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if ok:
                frame_count += 1

            if now >= next_sample:
                cpu_samples.append(float(psutil.cpu_percent(interval=None)))
                mem_mb = float(process.memory_info().rss) / (1024.0 * 1024.0)
                mem_samples_mb.append(mem_mb)
                next_sample = now + 0.5

    finally:
        camera.release()
        scanner.stop_scanner()

    total_elapsed = max(1e-6, time.perf_counter() - start)
    fps = float(frame_count) / total_elapsed

    result = BenchmarkResult(
        fps=fps,
        cpu_avg_percent=_safe_mean(cpu_samples),
        cpu_peak_percent=float(max(cpu_samples) if cpu_samples else 0.0),
        memory_avg_mb=_safe_mean(mem_samples_mb),
        memory_peak_mb=float(max(mem_samples_mb) if mem_samples_mb else 0.0),
        scan_latency_avg_ms=_safe_mean(scan_latencies_ms),
        scan_latency_peak_ms=float(max(scan_latencies_ms) if scan_latencies_ms else 0.0),
    )
    return result


def main() -> int:
    """Run benchmark and print pass/fail checks for Stage 7 targets."""
    _configure_logging()
    args = _parse_args()

    LOGGER.info("Running benchmark for %.1f seconds...", float(args.duration))
    result = run_benchmark(args)

    LOGGER.info("\n=== Benchmark Results ===")
    LOGGER.info("FPS achieved: %.2f", result.fps)
    LOGGER.info("CPU usage avg/peak: %.1f%% / %.1f%%", result.cpu_avg_percent, result.cpu_peak_percent)
    LOGGER.info("Memory avg/peak: %.1f MB / %.1f MB", result.memory_avg_mb, result.memory_peak_mb)
    LOGGER.info("WiFi scan latency avg/peak: %.1f ms / %.1f ms", result.scan_latency_avg_ms, result.scan_latency_peak_ms)

    fps_ok = result.fps >= 10.0
    cpu_ok = result.cpu_avg_percent < 80.0
    mem_ok = result.memory_peak_mb < 200.0

    LOGGER.info("\nTargets:")
    LOGGER.info("- FPS >= 10: %s", "PASS" if fps_ok else "CHECK")
    LOGGER.info("- CPU avg < 80%%: %s", "PASS" if cpu_ok else "CHECK")
    LOGGER.info("- Memory peak < 200 MB: %s", "PASS" if mem_ok else "CHECK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
