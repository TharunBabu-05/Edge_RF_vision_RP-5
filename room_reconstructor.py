"""Floor plan reconstruction from Visual Odometry trajectory data.

Reads the georeferenced RF samples CSV (produced by the VO + WiFi pipeline)
and reconstructs a 2D floor plan:

  - Outer room boundary from bounding box of walk trajectory + margin
  - Walk path trace for visualization
  - AP location pins (peak-signal position per SSID)

This floor plan is then consumed by heatmap_3d_renderer.py to produce
the Ekahau-style 3D isometric WiFi coverage map.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("room_reconstructor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Room:
    """One rectangular room zone on the floor plan."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


@dataclass
class APLocation:
    """Estimated Access Point location (router position on the floor plan)."""

    x: float
    y: float
    ssid: str
    dbm_peak: float
    sample_count: int = 0


@dataclass
class FloorPlan:
    """Complete reconstructed floor plan.

    Attributes:
        x_min, x_max, y_min, y_max: Outer boundary including walls.
        rooms: List of rectangular room zones.
        walk_path: Downsampled walk trajectory [(x, y), ...].
        ap_locations: Estimated AP positions per SSID.
        wall_thickness: Wall thickness in VO units.
        scale: VO units per metre (1.0 = uncalibrated).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    rooms: List[Room]
    walk_path: List[Tuple[float, float]]
    ap_locations: List[APLocation]

    wall_thickness: float = 0.3
    scale: float = 1.0          # metres per VO unit
    source_csv: str = ""

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def depth(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "rooms": [asdict(r) for r in self.rooms],
            "walk_path": [[p[0], p[1]] for p in self.walk_path],
            "ap_locations": [asdict(a) for a in self.ap_locations],
            "wall_thickness": self.wall_thickness,
            "scale": self.scale,
            "source_csv": self.source_csv,
        }

    def to_json(self, path: str) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info("Floor plan saved → %s", target)
        return str(target.resolve())

    @classmethod
    def from_dict(cls, d: dict) -> "FloorPlan":
        rooms = [Room(**r) for r in d.get("rooms", [])]
        aps = [APLocation(**a) for a in d.get("ap_locations", [])]
        return cls(
            x_min=float(d["x_min"]),
            x_max=float(d["x_max"]),
            y_min=float(d["y_min"]),
            y_max=float(d["y_max"]),
            rooms=rooms,
            walk_path=[tuple(p) for p in d.get("walk_path", [])],
            ap_locations=aps,
            wall_thickness=float(d.get("wall_thickness", 0.3)),
            scale=float(d.get("scale", 1.0)),
            source_csv=str(d.get("source_csv", "")),
        )

    @classmethod
    def from_json(cls, path: str) -> "FloorPlan":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Reconstruction engine
# ---------------------------------------------------------------------------

class FloorPlanReconstructor:
    """Builds a FloorPlan from georeferenced RF sample data.

    Usage::

        rec = FloorPlanReconstructor()
        plan = rec.from_csv("results/rf_samples.csv")
        plan.to_json("results/floor_plan.json")
    """

    def __init__(
        self,
        wall_margin: float = 1.0,
        wall_thickness: float = 0.3,
        path_downsample: int = 80,
        min_extent: float = 2.0,
    ) -> None:
        """Initialise reconstructor.

        Args:
            wall_margin: Metres of space added around trajectory for walls.
            wall_thickness: Thickness of outer walls in VO units.
            path_downsample: Max number of points in the walk path output.
            min_extent: Minimum room dimension (prevents degenerate plans).
        """
        self.wall_margin = float(wall_margin)
        self.wall_thickness = float(wall_thickness)
        self.path_downsample = max(10, int(path_downsample))
        self.min_extent = float(min_extent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_csv(self, csv_path: str) -> FloorPlan:
        """Build FloorPlan from an rf_samples.csv file.

        Args:
            csv_path: Path to CSV with columns x,y,z,ssid,bssid,dbm,freq,timestamp.

        Returns:
            Reconstructed FloorPlan.

        Raises:
            ValueError: If the CSV has no valid rows.
        """
        samples = self._load_csv(csv_path)
        if not samples:
            raise ValueError(f"No valid samples in: {csv_path}")
        LOGGER.info("Loaded %d samples from %s", len(samples), csv_path)
        plan = self._build(samples)
        plan.source_csv = str(Path(csv_path).resolve())
        return plan

    def from_samples(self, samples: List[Dict]) -> FloorPlan:
        """Build FloorPlan from a list of sample dicts (from RFDatabase).

        Args:
            samples: List of dicts with keys x, y, z, ssid, dbm, etc.

        Returns:
            Reconstructed FloorPlan.
        """
        if not samples:
            raise ValueError("Empty samples list")
        return self._build(samples)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: str) -> List[Dict]:
        """Parse rf_samples.csv into list of dicts."""
        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row.get("z", 0.0)),
                        "ssid": str(row.get("ssid", "<hidden>")),
                        "bssid": str(row.get("bssid", "")),
                        "dbm": float(row.get("dbm", -90.0)),
                        "timestamp": float(row.get("timestamp", 0.0)),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
        return rows

    def _build(self, samples: List[Dict]) -> FloorPlan:
        """Core floor plan building logic."""
        xs = [s["x"] for s in samples]
        ys = [s["y"] for s in samples]

        x_lo = min(xs) - self.wall_margin
        x_hi = max(xs) + self.wall_margin
        y_lo = min(ys) - self.wall_margin
        y_hi = max(ys) + self.wall_margin

        # Ensure minimum extent
        if x_hi - x_lo < self.min_extent:
            cx = (x_lo + x_hi) / 2.0
            x_lo = cx - self.min_extent / 2.0
            x_hi = cx + self.min_extent / 2.0
        if y_hi - y_lo < self.min_extent:
            cy = (y_lo + y_hi) / 2.0
            y_lo = cy - self.min_extent / 2.0
            y_hi = cy + self.min_extent / 2.0

        # Single open room (inner boundary = outer - wall_thickness)
        inner_room = Room(
            name="Main Area",
            x_min=x_lo + self.wall_thickness,
            x_max=x_hi - self.wall_thickness,
            y_min=y_lo + self.wall_thickness,
            y_max=y_hi - self.wall_thickness,
        )

        # Walk path (downsampled)
        walk_path = self._downsample_path(list(zip(xs, ys)), self.path_downsample)

        # AP locations per SSID
        ap_locations = self._find_ap_locations(samples)

        return FloorPlan(
            x_min=x_lo,
            x_max=x_hi,
            y_min=y_lo,
            y_max=y_hi,
            rooms=[inner_room],
            walk_path=walk_path,
            ap_locations=ap_locations,
            wall_thickness=self.wall_thickness,
            scale=1.0,
        )

    @staticmethod
    def _downsample_path(
        path: List[Tuple[float, float]], target: int
    ) -> List[Tuple[float, float]]:
        """Evenly downsample walk path to at most `target` points."""
        if len(path) <= target:
            return path
        step = len(path) / target
        return [path[int(i * step)] for i in range(target)]

    @staticmethod
    def _find_ap_locations(samples: List[Dict]) -> List[APLocation]:
        """Estimate AP location as the position with peak RSSI per SSID."""
        by_ssid: Dict[str, List[Dict]] = {}
        for s in samples:
            ssid = s.get("ssid", "<hidden>")
            by_ssid.setdefault(ssid, []).append(s)

        aps: List[APLocation] = []
        for ssid, pts in by_ssid.items():
            # Peak signal position
            best = max(pts, key=lambda p: p.get("dbm", -100.0))
            aps.append(APLocation(
                x=float(best["x"]),
                y=float(best["y"]),
                ssid=ssid,
                dbm_peak=float(best.get("dbm", -90.0)),
                sample_count=len(pts),
            ))

        return aps


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main() -> int:
    import argparse

    _configure_logging()
    parser = argparse.ArgumentParser(description="Reconstruct floor plan from RF samples CSV")
    parser.add_argument("csv", nargs="?", default="rf_samples.csv",
                        help="Path to rf_samples CSV")
    parser.add_argument("--out", default="floor_plan.json",
                        help="Output JSON path")
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    LOGGER.info("Reconstructing floor plan from: %s", args.csv)
    try:
        rec = FloorPlanReconstructor(wall_margin=args.margin)
        plan = rec.from_csv(args.csv)

        LOGGER.info("Floor plan: %.1f × %.1f VO-units", plan.width, plan.depth)
        LOGGER.info("Walk path points: %d", len(plan.walk_path))
        LOGGER.info("APs found: %d  →  %s", len(plan.ap_locations),
                    [a.ssid for a in plan.ap_locations])

        out = plan.to_json(args.out)
        LOGGER.info("Saved → %s", out)
        return 0
    except FileNotFoundError:
        LOGGER.warning("CSV file not found: %s. (No data to reconstruct yet. Start a survey or capture data first.)", args.csv)
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to reconstruct floor plan")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
