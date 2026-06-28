"""Signal processing utilities for WiFi dBm normalization and smoothing.

This module provides lightweight functions designed for Raspberry Pi Zero 2W:
- Convert dBm values into normalized 0.0-1.0 intensities.
- Smooth intensity changes using exponential moving average (EMA).
- Select dominant (strongest) network from scan results.
- Combine multiple network strengths into a single intensity metric.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, List, Optional, Sequence

import config

LOGGER = logging.getLogger("signal_processor")


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a float value into the inclusive [low, high] range."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def normalize_dbm(
    dbm_value: float,
    dbm_min: float = config.DBM_MIN,
    dbm_max: float = config.DBM_MAX,
) -> float:
    """Normalize dBm to intensity in the 0.0-1.0 range.

    Mapping rule:
    - dbm_max (default -30) -> 1.0 (strongest)
    - dbm_min (default -90) -> 0.0 (weakest)

    Values outside the range are clamped.

    Args:
        dbm_value: Input signal level in dBm.
        dbm_min: Lower bound that maps to 0.0.
        dbm_max: Upper bound that maps to 1.0.

    Returns:
        Normalized intensity between 0.0 and 1.0.
    """
    if dbm_max <= dbm_min:
        raise ValueError("dbm_max must be greater than dbm_min")

    ratio = (float(dbm_value) - float(dbm_min)) / (float(dbm_max) - float(dbm_min))
    return _clamp(ratio, 0.0, 1.0)


def smooth_signal(new_value: float, old_value: float, alpha: float = config.SMOOTHING_ALPHA) -> float:
    """Apply exponential moving average smoothing.

    Formula:
        smoothed = alpha * new + (1 - alpha) * old

    Args:
        new_value: Latest intensity sample.
        old_value: Previous smoothed intensity.
        alpha: Smoothing factor in [0.0, 1.0].

    Returns:
        Smoothed intensity in [0.0, 1.0].
    """
    alpha_clamped = _clamp(float(alpha), 0.0, 1.0)
    new_clamped = _clamp(float(new_value), 0.0, 1.0)
    old_clamped = _clamp(float(old_value), 0.0, 1.0)
    return _clamp(alpha_clamped * new_clamped + (1.0 - alpha_clamped) * old_clamped, 0.0, 1.0)


def get_dominant_network(networks_list: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the strongest network dictionary based on highest dBm.

    Args:
        networks_list: Iterable of network dicts with a `dbm` key.

    Returns:
        Copy of the strongest network dict, or None if input is empty/invalid.
    """
    if not networks_list:
        return None

    def _dbm_of(item: Dict[str, Any]) -> float:
        """Safely convert network dBm value to float for comparisons."""
        try:
            return float(item.get("dbm", -100.0))
        except (TypeError, ValueError):
            return -100.0

    strongest = max(networks_list, key=_dbm_of)
    return dict(strongest)


def get_network_intensities(networks_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return networks with per-network normalized intensity attached.

    This helper supports multi-network visualization paths.

    Args:
        networks_list: Network dicts containing at least `dbm`.

    Returns:
        New list of dicts with an added `intensity` key, sorted by intensity desc.
    """
    enriched: List[Dict[str, Any]] = []

    for network in networks_list:
        item = dict(network)
        try:
            dbm_value = float(item.get("dbm", -100.0))
        except (TypeError, ValueError):
            dbm_value = -100.0
        item["intensity"] = normalize_dbm(dbm_value)
        enriched.append(item)

    enriched.sort(key=lambda item: float(item.get("intensity", 0.0)), reverse=True)
    return enriched


def get_combined_intensity(networks_list: Sequence[Dict[str, Any]]) -> float:
    """Aggregate multiple networks into one combined intensity.

    Combination model:
    - Convert each network dBm to intensity in [0, 1].
    - Combine with probabilistic OR:
        combined = 1 - product(1 - intensity_i)

    This avoids over-amplifying many weak networks while still reflecting
    increased ambient RF presence when multiple networks are visible.

    Args:
        networks_list: Network dicts containing `dbm`.

    Returns:
        Combined intensity in [0.0, 1.0].
    """
    if not networks_list:
        return 0.0

    product = 1.0
    for network in networks_list:
        try:
            dbm_value = float(network.get("dbm", -100.0))
        except (TypeError, ValueError):
            dbm_value = -100.0
        intensity = normalize_dbm(dbm_value)
        product *= (1.0 - intensity)

    return _clamp(1.0 - product, 0.0, 1.0)


class SignalSmoother:
    """Stateful EMA helper for frame-to-frame signal stabilization."""

    def __init__(self, alpha: float = config.SMOOTHING_ALPHA, initial_value: float = 0.0) -> None:
        """Initialize smoother with alpha and initial intensity value."""
        self.alpha = _clamp(float(alpha), 0.0, 1.0)
        self._value = _clamp(float(initial_value), 0.0, 1.0)

    def update(self, new_value: float) -> float:
        """Update internal state using a new intensity sample and return smoothed value."""
        self._value = smooth_signal(new_value, self._value, self.alpha)
        return self._value

    def get_value(self) -> float:
        """Return current smoothed intensity value."""
        return self._value

    def reset(self, value: float = 0.0) -> None:
        """Reset the smoother state to a chosen intensity."""
        self._value = _clamp(float(value), 0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone module test."""
    parser = argparse.ArgumentParser(description="Signal processing module self-test")
    parser.add_argument(
        "--dbm",
        type=float,
        nargs="+",
        default=[-50.0, -77.0, -88.0],
        help="One or more dBm values for demo calculations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=config.SMOOTHING_ALPHA,
        help="EMA smoothing alpha in [0, 1].",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    """Configure readable logging for command-line demo mode."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    """Run a standalone demonstration of signal processing operations."""
    _configure_logging()
    args = _parse_args()

    networks = [{"ssid": f"net_{idx}", "dbm": dbm} for idx, dbm in enumerate(args.dbm, start=1)]

    LOGGER.info("Input dBm values: %s", args.dbm)

    dominant = get_dominant_network(networks)
    dominant_dbm = float(dominant["dbm"]) if dominant else -100.0
    dominant_intensity = normalize_dbm(dominant_dbm)
    combined_intensity = get_combined_intensity(networks)

    LOGGER.info("Dominant network: %s (%.2f dBm)", dominant.get("ssid", "none") if dominant else "none", dominant_dbm)
    LOGGER.info("Dominant intensity: %.3f", dominant_intensity)
    LOGGER.info("Combined intensity: %.3f", combined_intensity)

    smoother = SignalSmoother(alpha=args.alpha, initial_value=0.0)
    LOGGER.info("Smoothing demo (alpha=%.2f):", smoother.alpha)
    for dbm in args.dbm:
        raw = normalize_dbm(dbm)
        smoothed = smoother.update(raw)
        LOGGER.info("  dbm=%6.2f -> raw=%.3f smoothed=%.3f", dbm, raw, smoothed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
