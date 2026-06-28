"""Ekahau-style 3D isometric RF heatmap renderer (matplotlib).

Reads:
  - A FloorPlan (from room_reconstructor.py)
  - Georeferenced RF samples (from RFDatabase or CSV)

Produces:
  - A high-resolution PNG matching the Ekahau Site Survey visual style:
      · 3D isometric floor plan with white walls
      · RF signal heatmap draped on the floor
        (Blue=very weak → Red → Orange → Yellow → Green=strong)
      · Walk trajectory line
      · AP location pins (blue, with label)
      · Professional colorbar (dBm scale)

Performance note: matplotlib 3D rendering takes ~5–15 s on Pi Zero 2W.
This is acceptable for an on-demand export endpoint.
"""

from __future__ import annotations

import io
import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from room_reconstructor import APLocation, FloorPlan

LOGGER = logging.getLogger("heatmap_3d_renderer")

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _dbm_to_norm(dbm: float, dbm_min: float = -90.0, dbm_max: float = -30.0) -> float:
    """Map dBm value to 0–1 normalised intensity (0 = weak, 1 = strong)."""
    return float(np.clip((dbm - dbm_min) / (dbm_max - dbm_min), 0.0, 1.0))


def _signal_colormap(x: np.ndarray) -> np.ndarray:
    """Map 0–1 intensity to RGBA using a Ekahau-like ramp.

    0.0 = very weak → deep blue
    0.25 = weak      → cyan
    0.5  = medium    → green
    0.75 = good      → yellow
    1.0  = excellent → red  (reversed from typical heat — strong = green/yellow)

    Actually matching Ekahau convention:
      strong (high RSSI) = green
      medium             = yellow/orange
      weak (low RSSI)    = red/dark red
    """
    # Use matplotlib's RdYlGn: 0 → red, 0.5 → yellow, 1 → green
    try:
        import matplotlib.cm as cm
        return cm.RdYlGn(x)      # returns (N, 4) RGBA
    except ImportError:
        # Fallback: manual ramp
        r = np.clip(2 - 2 * x, 0, 1)
        g = np.clip(2 * x, 0, 1)
        b = np.zeros_like(x)
        a = np.ones_like(x)
        return np.stack([r, g, b, a], axis=-1)


# ---------------------------------------------------------------------------
# IDW interpolation (numpy-only, no scipy required)
# ---------------------------------------------------------------------------

def _idw_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Fast IDW interpolation from scatter points onto a regular grid.

    Args:
        xs, ys: Sample coordinates (N,).
        values: Sample values (N,).
        grid_x, grid_y: 1-D grid coordinate arrays.
        power: IDW power parameter.

    Returns:
        2-D array of shape (len(grid_y), len(grid_x)).
    """
    GX, GY = np.meshgrid(grid_x, grid_y)
    result = np.zeros_like(GX, dtype=np.float64)

    for gyi in range(len(grid_y)):
        for gxi in range(len(grid_x)):
            px, py = grid_x[gxi], grid_y[gyi]
            dists = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
            zero = dists < 1e-9
            if np.any(zero):
                result[gyi, gxi] = values[zero][0]
                continue
            w = 1.0 / (dists ** power)
            result[gyi, gxi] = np.dot(w, values) / w.sum()

    return result


def _fast_idw_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Vectorised IDW (may use more RAM but much faster for small grids)."""
    GX, GY = np.meshgrid(grid_x, grid_y)           # (H, W)
    QX = GX.reshape(-1, 1)                           # (H*W, 1)
    QY = GY.reshape(-1, 1)
    dists = np.sqrt((QX - xs) ** 2 + (QY - ys) ** 2)  # (H*W, N)
    dists = np.maximum(dists, 1e-9)
    w = 1.0 / (dists ** power)                       # (H*W, N)
    result_flat = (w @ values) / w.sum(axis=1)       # (H*W,)
    return result_flat.reshape(GX.shape)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

class Ekahau3DRenderer:
    """Renders Ekahau-style 3D isometric heatmap PNG.

    Usage::

        renderer = Ekahau3DRenderer(floor_plan, samples)
        png_bytes = renderer.render_png()
        with open("output.png", "wb") as f:
            f.write(png_bytes)
    """

    def __init__(
        self,
        floor_plan: FloorPlan,
        samples: List[dict],
        grid_resolution: int = 60,
        wall_height_ratio: float = 0.12,
        dbm_min: float = -90.0,
        dbm_max: float = -30.0,
        ssid_filter: Optional[str] = None,
    ) -> None:
        """Initialise renderer.

        Args:
            floor_plan:        Room geometry from FloorPlanReconstructor.
            samples:           List of dicts with x, y, dbm keys.
            grid_resolution:   N×N grid resolution (higher = smoother, slower).
            wall_height_ratio: Wall height as fraction of the largest room dim.
            dbm_min:           dBm that maps to 0 (weak).
            dbm_max:           dBm that maps to 1 (strong).
            ssid_filter:       If set, only use samples from this SSID.
        """
        self.fp = floor_plan
        self.grid_res = max(20, int(grid_resolution))
        self.wall_h_ratio = float(wall_height_ratio)
        self.dbm_min = float(dbm_min)
        self.dbm_max = float(dbm_max)

        # Filter samples
        if ssid_filter:
            self.samples = [s for s in samples if s.get("ssid") == ssid_filter]
        else:
            self.samples = list(samples)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_png(
        self,
        figsize: Tuple[float, float] = (16, 11),
        dpi: int = 130,
        elev: float = 32.0,
        azim: float = -52.0,
    ) -> bytes:
        """Render and return PNG bytes.

        Args:
            figsize: matplotlib figure size (width, height) in inches.
            dpi:     Output DPI (higher = sharper, slower).
            elev:    Camera elevation angle (degrees).
            azim:    Camera azimuth angle (degrees).

        Returns:
            PNG image as raw bytes.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")          # non-interactive backend (Pi safe)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as exc:
            LOGGER.error("matplotlib not available: %s", exc)
            return b""

        fp = self.fp
        wall_h = max(fp.width, fp.depth) * self.wall_h_ratio
        wall_h = max(wall_h, 0.5)

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#f8f8f6")
        ax = fig.add_subplot(111, projection="3d", facecolor="#f8f8f6")

        # ── Floor heatmap ────────────────────────────────────────────
        self._draw_floor(ax, wall_h)

        # ── Walls ────────────────────────────────────────────────────
        self._draw_walls(ax, wall_h, Poly3DCollection)

        # ── Walk path ────────────────────────────────────────────────
        self._draw_walk_path(ax)

        # ── AP pins ──────────────────────────────────────────────────
        self._draw_ap_pins(ax, wall_h)

        # ── Colorbar ─────────────────────────────────────────────────
        self._add_colorbar(fig, ax)

        # ── View & style ─────────────────────────────────────────────
        ax.view_init(elev=elev, azim=azim)

        # Box aspect ratio: keep room proportions
        aspect_x = fp.width
        aspect_y = fp.depth
        aspect_z = wall_h * 2.5
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])

        ax.set_axis_off()

        # Title
        n_samples = len(self.samples)
        ssid_names = list({s.get("ssid", "?") for s in self.samples})
        title = "Indoor RF Digital Twin"
        if ssid_names:
            title += f"  —  {', '.join(ssid_names[:3])}"
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color="#2c3e50", pad=12)

        plt.tight_layout(pad=1.0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        LOGGER.info("Rendered 3D heatmap PNG (%d samples, %dx%d grid)",
                    n_samples, self.grid_res, self.grid_res)
        return buf.read()

    def render_to_file(self, path: str, **kwargs) -> str:
        """Render and save PNG to file."""
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.render_png(**kwargs)
        target.write_bytes(data)
        LOGGER.info("Saved 3D render → %s", target)
        return str(target.resolve())

    # ------------------------------------------------------------------
    # Private drawing methods
    # ------------------------------------------------------------------

    def _make_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate samples onto a regular grid. Returns (GX, GY, intensity)."""
        fp = self.fp
        grid_x = np.linspace(fp.x_min, fp.x_max, self.grid_res)
        grid_y = np.linspace(fp.y_min, fp.y_max, self.grid_res)

        if len(self.samples) < 2:
            GX, GY = np.meshgrid(grid_x, grid_y)
            intensity = np.full(GX.shape, 0.5)
            return GX, GY, intensity

        xs = np.array([s["x"] for s in self.samples])
        ys = np.array([s["y"] for s in self.samples])
        dbms = np.array([s.get("dbm", -75.0) for s in self.samples])
        norms = np.clip(
            (dbms - self.dbm_min) / (self.dbm_max - self.dbm_min), 0.0, 1.0
        )

        try:
            intensity = _fast_idw_grid(xs, ys, norms, grid_x, grid_y)
        except MemoryError:
            intensity = _idw_grid(xs, ys, norms, grid_x, grid_y)

        GX, GY = np.meshgrid(grid_x, grid_y)
        return GX, GY, np.clip(intensity, 0.0, 1.0)

    def _draw_floor(self, ax, wall_h: float) -> None:
        """Draw RF heatmap surface on the floor (Z=0)."""
        import matplotlib.pyplot as plt

        GX, GY, intensity = self._make_grid()
        GZ = np.zeros_like(GX)
        face_colors = _signal_colormap(intensity)

        ax.plot_surface(
            GX, GY, GZ,
            facecolors=face_colors,
            shade=False,
            antialiased=True,
            linewidth=0,
            zorder=1,
        )

    def _draw_walls(self, ax, wall_h: float, Poly3DCollection) -> None:
        """Draw four white walls around the room boundary."""
        fp = self.fp
        x0, x1 = fp.x_min, fp.x_max
        y0, y1 = fp.y_min, fp.y_max
        h = wall_h

        # Each wall face as a polygon list for Poly3DCollection
        # Colors: faces facing camera are lighter; back faces are darker
        wall_specs = [
            # (vertices_list, face_color, edge_color)
            # Back wall (y = y_max) — typically top-left in isometric
            (
                [[x0, y1, 0], [x1, y1, 0], [x1, y1, h], [x0, y1, h]],
                "#e8e8e4",
            ),
            # Right wall (x = x_max) — typically top-right in isometric
            (
                [[x1, y0, 0], [x1, y1, 0], [x1, y1, h], [x1, y0, h]],
                "#d8d8d4",
            ),
            # Front wall (y = y_min) — bottom
            (
                [[x0, y0, 0], [x1, y0, 0], [x1, y0, h], [x0, y0, h]],
                "#efefea",
            ),
            # Left wall (x = x_min) — bottom-left
            (
                [[x0, y0, 0], [x0, y1, 0], [x0, y1, h], [x0, y0, h]],
                "#e0e0db",
            ),
        ]

        for verts, fc in wall_specs:
            poly = Poly3DCollection(
                [verts],
                facecolor=fc,
                edgecolor="#c0c0b8",
                linewidth=0.6,
                zorder=3,
                alpha=0.92,
            )
            ax.add_collection3d(poly)

        # Top edges of walls (outline)
        corners_top = [
            (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h), (x0, y0, h)
        ]
        ax.plot(
            [c[0] for c in corners_top],
            [c[1] for c in corners_top],
            [c[2] for c in corners_top],
            color="#a0a09a", linewidth=0.8, zorder=4,
        )

    def _draw_walk_path(self) -> None:
        pass  # used below

    def _draw_walk_path(self, ax) -> None:
        """Draw the VO walk trajectory as a line on the floor."""
        path = self.fp.walk_path
        if len(path) < 2:
            return
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [0.04] * len(path)
        ax.plot(xs, ys, zs,
                color="#ffffff", linewidth=1.2, alpha=0.70,
                linestyle="--", zorder=5)
        # Start dot
        ax.scatter([xs[0]], [ys[0]], [0.04],
                   color="#ffffff", s=40, zorder=6, marker="o")

    def _draw_ap_pins(self, ax, wall_h: float) -> None:
        """Draw AP location pins (blue vertical stems with circle markers)."""
        pin_h = wall_h * 0.9

        for ap in self.fp.ap_locations:
            x, y = ap.x, ap.y

            # Vertical stem
            ax.plot([x, x], [y, y], [0.04, pin_h],
                    color="#1565C0", linewidth=2.0, zorder=8)

            # Circle at top (in X-Y plane at height pin_h)
            r = max(self.fp.width, self.fp.depth) * 0.025
            theta = np.linspace(0, 2 * np.pi, 30)
            cx = x + r * np.cos(theta)
            cy = y + r * np.sin(theta)
            cz = np.full_like(cx, pin_h)
            ax.plot(cx, cy, cz, color="#1565C0", linewidth=2.0, zorder=9)

            # Fill circle (scatter disk)
            ax.scatter([x], [y], [pin_h],
                       color="#42A5F5", s=80, marker="o",
                       edgecolors="#1565C0", linewidths=1.5, zorder=10)

            # Label
            norm = _dbm_to_norm(ap.dbm_peak, self.dbm_min, self.dbm_max)
            label = f"{ap.ssid}\n{ap.dbm_peak:.0f} dBm"
            ax.text(x, y, pin_h + r * 1.5, label,
                    fontsize=7, color="#1a237e",
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffffcc",
                              ec="#1565C0", lw=0.8),
                    zorder=11)

    def _add_colorbar(self, fig, ax) -> None:
        """Add a horizontal colorbar showing the dBm → colour mapping."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            from matplotlib.colorbar import ColorbarBase

            norm = Normalize(vmin=self.dbm_min, vmax=self.dbm_max)
            sm = ScalarMappable(cmap="RdYlGn", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                                fraction=0.03, pad=0.01, shrink=0.5,
                                aspect=30)
            cbar.set_label("Signal Strength (dBm)", fontsize=9, color="#444")
            cbar.ax.tick_params(labelsize=8, colors="#555")
        except Exception:
            pass   # colorbar is cosmetic — don't crash on error


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def render_from_csv(
    csv_path: str,
    output_path: str = "results/rf_heatmap_3d.png",
    ssid: Optional[str] = None,
    grid_resolution: int = 60,
) -> str:
    """High-level helper: CSV → floor plan → 3D PNG in one call.

    Args:
        csv_path:        Path to rf_samples.csv.
        output_path:     Destination PNG path.
        ssid:            Optional SSID filter.
        grid_resolution: Grid quality (default 60).

    Returns:
        Absolute path of the saved PNG.
    """
    from room_reconstructor import FloorPlanReconstructor
    import csv as _csv

    rec = FloorPlanReconstructor()
    plan = rec.from_csv(csv_path)

    # Load samples
    samples = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                samples.append({
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "ssid": row.get("ssid", ""),
                    "dbm": float(row.get("dbm", -90.0)),
                })
            except (KeyError, ValueError):
                continue

    renderer = Ekahau3DRenderer(
        plan, samples,
        grid_resolution=grid_resolution,
        ssid_filter=ssid,
    )
    return renderer.render_to_file(output_path)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Render Ekahau-style 3D RF heatmap PNG")
    parser.add_argument("csv", nargs="?",
                        default="rf_samples.csv",
                        help="Path to rf_samples CSV")
    parser.add_argument("--out", default="rf_heatmap_3d.png")
    parser.add_argument("--ssid", default=None)
    parser.add_argument("--grid", type=int, default=60)
    args = parser.parse_args()

    LOGGER.info("Rendering 3D heatmap from: %s", args.csv)
    t0 = time.perf_counter()
    try:
        out = render_from_csv(args.csv, args.out, ssid=args.ssid,
                              grid_resolution=args.grid)
        elapsed = time.perf_counter() - t0
        LOGGER.info("Done in %.1f s → %s", elapsed, out)
        return 0
    except FileNotFoundError:
        LOGGER.warning("CSV file not found: %s. (No data to render yet. Start a survey or capture data first.)", args.csv)
        return 0
    except Exception as exc:
        LOGGER.exception("Failed to render 3D heatmap")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
