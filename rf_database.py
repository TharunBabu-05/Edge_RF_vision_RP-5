"""Georeferenced RF sample database for 3D Indoor RF Digital Twin.

Stores WiFi scan results anchored to real-world (X, Y, Z) coordinates produced
by the Visual Odometry pose estimator. Every scan result creates one RFSample
per SSID found, tagged with:
    - World position (x, y, z) in metric units
    - SSID and BSSID identifiers
    - RSSI value in dBm
    - Unix timestamp of the measurement

Features:
    - Thread-safe add/query operations
    - Rolling window to cap memory usage
    - Min-displacement filter to avoid oversampling at stationary positions
    - Export to CSV and JSON for offline analysis
    - Load from JSON snapshot for session resume
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config

LOGGER = logging.getLogger("rf_database")


def csv_text(value: str) -> str:
    """Keep network names as text when opened in spreadsheet applications."""
    text = str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")) else text


@dataclass(frozen=True)
class RFSample:
    """One georeferenced WiFi measurement."""

    x: float          # World X coordinate (meters)
    y: float          # World Y coordinate (meters)
    z: float          # World Z coordinate (meters)
    ssid: str         # Network SSID
    bssid: str        # Network BSSID (MAC address)
    dbm: float        # Signal strength in dBm
    freq: int         # Channel frequency in MHz
    timestamp: float  # Unix timestamp of measurement

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z, self.dbm, self.timestamp)):
            raise ValueError("Sample coordinates, signal and timestamp must be finite")
        if not -150 <= self.dbm <= 0 or not 0 <= self.freq <= 100000 or self.timestamp < 0:
            raise ValueError("Sample signal, frequency or timestamp outside supported range")


@dataclass(frozen=True)
class DBStatus:
    """Immutable status snapshot of the RF database."""

    total_samples: int
    ssid_count: int
    ssids: List[str]
    last_sample_epoch: float
    last_position_x: float
    last_position_y: float
    last_position_z: float


def _euclidean_distance(ax: float, ay: float, az: float,
                         bx: float, by: float, bz: float) -> float:
    """Compute 3D Euclidean distance between two points."""
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


class RFDatabase:
    """Thread-safe store of georeferenced RF samples.

    Usage::

        db = RFDatabase()
        db.add_sample(pose, networks)
        samples = db.get_samples(ssid="Jack's")
        db.save_to_csv("survey_output.csv")
    """

    def __init__(
        self,
        max_samples: int = config.RF_DB_MAX_SAMPLES,
        min_displacement: float = config.RF_SAMPLE_MIN_DISTANCE,
        storage_path: Optional[str] = None,
    ) -> None:
        """Initialize empty RF database.

        Args:
            max_samples: Maximum total samples retained (rolling window).
            min_displacement: Minimum 3D displacement (meters) between consecutive
                              samples to avoid oversampling at a stationary position.
        """
        self.max_samples = max(10, int(max_samples))
        self.min_displacement = max(0.0, float(min_displacement))

        self._lock = threading.Lock()
        self._samples: List[RFSample] = []
        self._last_sample_x: float = float("nan")
        self._last_sample_y: float = float("nan")
        self._last_sample_z: float = float("nan")
        self._last_sample_epoch: float = 0.0
        self.revision = 0
        self.storage_path = storage_path
        if storage_path:
            Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(storage_path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
                rows = connection.execute("SELECT payload FROM samples ORDER BY id DESC LIMIT ?", (self.max_samples,)).fetchall()
            self._samples = [RFSample(**json.loads(row[0])) for row in reversed(rows)]
            if self._samples:
                last = self._samples[-1]
                self._last_sample_x, self._last_sample_y, self._last_sample_z = last.x, last.y, last.z
                self._last_sample_epoch = last.timestamp

    def _persist_locked(self, new_samples=(), replace=False):
        """Commit before publishing memory changes; SQLite rolls back failed writes."""
        if not self.storage_path:
            return
        with sqlite3.connect(self.storage_path, timeout=10) as connection:
            if replace:
                connection.execute("DELETE FROM samples")
            connection.executemany("INSERT INTO samples(payload) VALUES (?)", [(json.dumps(asdict(s), allow_nan=False),) for s in new_samples])
            connection.execute("DELETE FROM samples WHERE id NOT IN (SELECT id FROM samples ORDER BY id DESC LIMIT ?)", (self.max_samples,))

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add_sample(
        self,
        x: float,
        y: float,
        z: float,
        networks: Sequence[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> int:
        """Add RF measurements at a given world position.

        One RFSample is created per network entry in `networks`. The
        min_displacement filter is applied: if the current position is too
        close to the last recorded position, the sample is skipped.

        Args:
            x: World X coordinate.
            y: World Y coordinate.
            z: World Z coordinate.
            networks: List of network dicts with keys ssid, bssid, dbm, freq.
            timestamp: Unix timestamp; defaults to time.time().

        Returns:
            Number of new RFSample records added (0 if filtered out).
        """
        now = float(timestamp) if timestamp is not None else time.time()
        x, y, z = float(x), float(y), float(z)
        if not all(math.isfinite(v) for v in (x, y, z, now)):
            raise ValueError("Coordinates and timestamp must be finite")
        pending = []
        for net in networks:
            try:
                pending.append(RFSample(x, y, z, str(net.get("ssid", "<hidden>")), str(net.get("bssid", "")), float(net["dbm"]), int(net.get("freq", 0)), now))
            except (KeyError, ValueError, TypeError, OverflowError):
                continue
        if not pending:
            return 0

        with self._lock:
            # Displacement filter
            if not math.isnan(self._last_sample_x):
                dist = _euclidean_distance(
                    x, y, z,
                    self._last_sample_x,
                    self._last_sample_y,
                    self._last_sample_z,
                )
                if dist < self.min_displacement:
                    return 0

            self._persist_locked(pending)
            self._samples.extend(pending)
            added = len(pending)
            self.revision += 1

            # Rolling window: trim oldest samples if over capacity
            if len(self._samples) > self.max_samples:
                overflow = len(self._samples) - self.max_samples
                del self._samples[:overflow]

            # Update position tracking
            self._last_sample_x = float(x)
            self._last_sample_y = float(y)
            self._last_sample_z = float(z)
            self._last_sample_epoch = now

            return added

    def clear(self) -> None:
        """Remove all samples and reset position tracking."""
        with self._lock:
            self._persist_locked(replace=True)
            self.revision += 1
            self._samples.clear()
            self._last_sample_x = float("nan")
            self._last_sample_y = float("nan")
            self._last_sample_z = float("nan")
            self._last_sample_epoch = 0.0
        LOGGER.info("RFDatabase cleared.")

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_samples(
        self,
        ssid: Optional[str] = None,
        bssid: Optional[str] = None,
    ) -> List[RFSample]:
        """Return filtered list of samples.

        Args:
            ssid: If set, return only samples matching this SSID.
            bssid: If set, return only samples matching this BSSID.

        Returns:
            Copy of matching RFSample records.
        """
        with self._lock:
            result = list(self._samples)

        if ssid is not None:
            result = [s for s in result if s.ssid == ssid]
        if bssid is not None:
            result = [s for s in result if s.bssid == bssid]
        return result

    def get_ssids(self) -> List[str]:
        """Return sorted list of unique SSIDs seen across all samples."""
        with self._lock:
            ssids = {s.ssid for s in self._samples}
        return sorted(ssids)

    def get_status(self) -> DBStatus:
        """Return immutable snapshot of database statistics."""
        with self._lock:
            total = len(self._samples)
            ssids = sorted({s.ssid for s in self._samples})
            lx = self._last_sample_x if not math.isnan(self._last_sample_x) else 0.0
            ly = self._last_sample_y if not math.isnan(self._last_sample_y) else 0.0
            lz = self._last_sample_z if not math.isnan(self._last_sample_z) else 0.0
            epoch = self._last_sample_epoch

        return DBStatus(
            total_samples=total,
            ssid_count=len(ssids),
            ssids=ssids,
            last_sample_epoch=epoch,
            last_position_x=lx,
            last_position_y=ly,
            last_position_z=lz,
        )

    def sample_count(self, ssid: Optional[str] = None) -> int:
        """Return total samples, optionally filtered by SSID."""
        with self._lock:
            if ssid is None:
                return len(self._samples)
            return sum(1 for s in self._samples if s.ssid == ssid)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_to_csv(self, file_path: str) -> str:
        """Export all samples to a CSV file.

        CSV columns: x, y, z, ssid, bssid, dbm, freq, timestamp

        Args:
            file_path: Output file path.

        Returns:
            Resolved absolute path of written file.
        """
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            samples = list(self._samples)

        with target.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["x", "y", "z", "ssid", "bssid", "dbm", "freq", "timestamp"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "x": f"{s.x:.4f}",
                    "y": f"{s.y:.4f}",
                    "z": f"{s.z:.4f}",
                    "ssid": csv_text(s.ssid),
                    "bssid": csv_text(s.bssid),
                    "dbm": f"{s.dbm:.2f}",
                    "freq": s.freq,
                    "timestamp": f"{s.timestamp:.3f}",
                })

        LOGGER.info("Exported %d samples to CSV: %s", len(samples), target)
        return str(target.resolve())

    def save_to_json(self, file_path: str) -> str:
        """Export all samples to a JSON file for session resume or analysis.

        Args:
            file_path: Output file path.

        Returns:
            Resolved absolute path of written file.
        """
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            samples = list(self._samples)

        payload = {
            "version": 1,
            "exported_at": time.time(),
            "sample_count": len(samples),
            "samples": [asdict(s) for s in samples],
        }

        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(target)

        LOGGER.info("Exported %d samples to JSON: %s", len(samples), target)
        return str(target.resolve())

    def load_from_json(self, file_path: str) -> int:
        """Load samples from a previously saved JSON file.

        Existing samples are replaced.

        Args:
            file_path: Input JSON file path.

        Returns:
            Number of samples loaded.
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"RF database snapshot not found: {source}")

        with source.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise ValueError("Snapshot must contain a samples array")
        raw_samples = payload["samples"]
        loaded: List[RFSample] = []

        for item in raw_samples:
            try:
                loaded.append(RFSample(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item["z"]),
                    ssid=str(item.get("ssid", "<hidden>")),
                    bssid=str(item.get("bssid", "")),
                    dbm=float(item.get("dbm", -100.0)),
                    freq=int(item.get("freq", 0)),
                    timestamp=float(item.get("timestamp", 0.0)),
                ))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                LOGGER.warning("Skipped malformed sample record: %s", exc)

        with self._lock:
            self._persist_locked(loaded[-self.max_samples:], replace=True)
            self.revision += 1
            self._samples = loaded[-self.max_samples:]
            if loaded:
                last = loaded[-1]
                self._last_sample_x = last.x
                self._last_sample_y = last.y
                self._last_sample_z = last.z
                self._last_sample_epoch = last.timestamp
            else:
                self._last_sample_x = float("nan")
                self._last_sample_y = float("nan")
                self._last_sample_z = float("nan")
                self._last_sample_epoch = 0.0

        LOGGER.info("Loaded %d samples from JSON: %s", len(loaded), source)
        return min(len(loaded), self.max_samples)

    def to_dict_list(
        self,
        ssid: Optional[str] = None,
        max_items: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return samples as a list of plain dicts (for JSON API responses).

        Args:
            ssid: Optional SSID filter.
            max_items: If set, return only the most recent N items.

        Returns:
            List of serializable dicts.
        """
        samples = self.get_samples(ssid=ssid)
        if max_items is not None:
            samples = samples[-max_items:] if max_items > 0 else []
        return [asdict(s) for s in samples]


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> int:
    _configure_logging()
    LOGGER.info("=== RFDatabase Demo ===")

    db = RFDatabase(max_samples=100, min_displacement=0.05)

    # Simulate a walk through a room collecting WiFi data
    import random
    rng = random.Random(42)

    fake_networks = [
        {"ssid": "Jack's",       "bssid": "aa:bb:cc:dd:ee:01", "dbm": -45.0, "freq": 2462},
        {"ssid": "realme 12x",   "bssid": "aa:bb:cc:dd:ee:02", "dbm": -70.0, "freq": 2437},
        {"ssid": "TP-Link_5G",   "bssid": "aa:bb:cc:dd:ee:03", "dbm": -80.0, "freq": 5180},
    ]

    positions = [(x * 0.2, y * 0.2, 0.0) for x in range(10) for y in range(5)]

    added_total = 0
    for x, y, z in positions:
        # Add some RSSI variation per position
        nets = []
        for net in fake_networks:
            net_copy = dict(net)
            net_copy["dbm"] = float(net["dbm"]) + rng.uniform(-5.0, 5.0)
            nets.append(net_copy)

        n = db.add_sample(x, y, z, nets)
        added_total += n

    status = db.get_status()
    LOGGER.info("Total samples: %d", status.total_samples)
    LOGGER.info("SSIDs: %s", status.ssids)

    samples_jacks = db.get_samples(ssid="Jack's")
    LOGGER.info("Samples for 'Jack's': %d", len(samples_jacks))

    csv_path = db.save_to_csv("/tmp/rf_demo_export.csv")
    LOGGER.info("CSV saved: %s", csv_path)

    json_path = db.save_to_json("/tmp/rf_demo_export.json")
    LOGGER.info("JSON saved: %s", json_path)

    db2 = RFDatabase()
    db2.load_from_json(json_path)
    LOGGER.info("Reloaded database: %d samples", db2.sample_count())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
