"""3D RF spatial interpolation engine for Indoor RF Digital Twin.

Converts sparse georeferenced RFSample measurements (from rf_database.py)
into dense continuous RF field grids using three selectable methods:

    IDW     — Inverse Distance Weighting (fast, no extra deps)
    gaussian — Gaussian Radial Basis Function via scipy (smooth, accurate)
    kriging  — Ordinary Kriging via scipy (statistically optimal, slowest)

Both 2D (top-down floor plan) and 3D (volumetric voxel grid) outputs
are supported.

Performance targets (Pi Zero 2W, 50×50 grid):
    IDW      ≤ 30 ms
    Gaussian ≤ 80 ms
    Kriging  ≤ 300 ms (export-only; not used for real-time streaming)
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

import config
from rf_database import RFDatabase, RFSample

LOGGER = logging.getLogger("rf_interpolator")

# Optional scipy imports — gracefully degrade if unavailable on Pi
try:
    from scipy.interpolate import RBFInterpolator
    SCIPY_RBF_AVAILABLE = True
except ImportError:
    RBFInterpolator = None  # type: ignore[assignment, misc]
    SCIPY_RBF_AVAILABLE = False

try:
    from scipy.spatial.distance import cdist
    SCIPY_CDIST_AVAILABLE = True
except ImportError:
    from numpy.linalg import norm as _np_norm
    SCIPY_CDIST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal distance utilities
# ---------------------------------------------------------------------------

def _pairwise_distances(query: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances efficiently.

    Args:
        query:   (M, D) array of query points.
        sources: (N, D) array of source points.

    Returns:
        (M, N) float64 distance matrix.
    """
    if SCIPY_CDIST_AVAILABLE:
        from scipy.spatial.distance import cdist  # noqa: F811
        return cdist(query, sources, metric="euclidean")
    # Fallback: vectorized NumPy
    diff = query[:, np.newaxis, :] - sources[np.newaxis, :, :]  # (M, N, D)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


# ---------------------------------------------------------------------------
# Core interpolation methods
# ---------------------------------------------------------------------------

def idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    power: float = config.IDW_POWER,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation.

    Args:
        points: (N, D) known sample coordinates.
        values: (N,) known sample values (dBm or normalized intensity).
        query:  (M, D) grid points to interpolate.
        power:  IDW power parameter (2 = classic).
        epsilon: Small value to prevent division by zero at exact sample positions.

    Returns:
        (M,) interpolated values at query points.
    """
    if len(points) == 0:
        return np.zeros(len(query), dtype=np.float64)

    dists = _pairwise_distances(query, points)  # (M, N)
    dists = np.maximum(dists, epsilon)
    weights = 1.0 / (dists ** float(power))    # (M, N)
    weight_sum = weights.sum(axis=1, keepdims=True)  # (M, 1)
    weight_sum = np.maximum(weight_sum, epsilon)
    interpolated = (weights @ values) / weight_sum.squeeze(axis=1)  # (M,)
    return interpolated.astype(np.float64)


def gaussian_rbf_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Gaussian Radial Basis Function interpolation via scipy.

    Falls back to IDW if scipy is unavailable.

    Args:
        points:    (N, D) known sample coordinates.
        values:    (N,) known sample values.
        query:     (M, D) grid query points.
        smoothing: RBFInterpolator smoothing parameter.

    Returns:
        (M,) interpolated values clipped to [min(values), max(values)].
    """
    if len(points) == 0:
        return np.zeros(len(query), dtype=np.float64)

    if not SCIPY_RBF_AVAILABLE:
        LOGGER.debug("scipy RBFInterpolator unavailable, falling back to IDW.")
        return idw_interpolate(points, values, query)

    try:
        rbf = RBFInterpolator(
            points,
            values,
            kernel="gaussian",
            smoothing=smoothing,
            epsilon=1.0,
        )
        result = rbf(query).astype(np.float64)
        v_min = float(values.min())
        v_max = float(values.max())
        return np.clip(result, v_min - abs(v_min) * 0.1, v_max + abs(v_max) * 0.1)
    except Exception as exc:
        LOGGER.warning("RBFInterpolator failed (%s), falling back to IDW.", exc)
        return idw_interpolate(points, values, query)


def kriging_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    length_scale: float = 1.0,
) -> np.ndarray:
    """Simple Ordinary Kriging interpolation (Gaussian variogram model).

    Uses only NumPy — no external Kriging library required.
    Implements the standard linear system: C * w = c0, result = w^T * values.

    Args:
        points:       (N, D) known sample coordinates.
        values:       (N,) known sample values.
        query:        (M, D) grid query points.
        length_scale: Gaussian variogram range parameter (same units as coords).

    Returns:
        (M,) kriged estimates.
    """
    n = len(points)
    if n == 0:
        return np.zeros(len(query), dtype=np.float64)
    if n == 1:
        return np.full(len(query), values[0], dtype=np.float64)

    def _gaussian_cov(d: np.ndarray, ls: float) -> np.ndarray:
        return np.exp(-0.5 * (d / max(ls, 1e-6)) ** 2)

    # Build covariance matrix between known points (N×N)
    dist_pp = _pairwise_distances(points, points)
    C = _gaussian_cov(dist_pp, length_scale)
    # Add nugget for numerical stability
    C += np.eye(n) * 1e-4

    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        LOGGER.warning("Kriging: covariance matrix is singular, falling back to IDW.")
        return idw_interpolate(points, values, query)

    # Covariance between known points and query points (M×N)
    dist_qp = _pairwise_distances(query, points)
    c0 = _gaussian_cov(dist_qp, length_scale)  # (M, N)

    # Kriging weights: w = C_inv @ c0^T → (N, M)
    weights = C_inv @ c0.T  # (N, M)

    # Kriged estimate: values^T @ weights → (M,)
    result = (values @ weights).astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# High-level interpolation interface
# ---------------------------------------------------------------------------

class RFInterpolator:
    """Manages RF field interpolation from scatter samples to dense grids.

    Supports 2D (floor plan) and 3D (volumetric) output grids.

    Usage::

        interp = RFInterpolator()
        grid_2d = interp.interpolate_2d(samples, method="gaussian")
        grid_3d = interp.interpolate_3d(samples, method="idw")
    """

    def __init__(
        self,
        grid_size: int = config.INTERPOLATION_GRID_SIZE,
        method: str = config.INTERPOLATION_METHOD,
        idw_power: float = config.IDW_POWER,
        kriging_length_scale: float = 1.0,
    ) -> None:
        """Initialize interpolator.

        Args:
            grid_size: NxN output grid resolution for 2D interpolation.
            method: Default interpolation method: "idw", "gaussian", "kriging".
            idw_power: IDW power parameter.
            kriging_length_scale: Kriging variogram length scale.
        """
        self.grid_size = max(4, int(grid_size))
        self.method = method.lower().strip()
        self.idw_power = float(idw_power)
        self.kriging_length_scale = float(kriging_length_scale)

    def interpolate_2d(
        self,
        samples: List[RFSample],
        method: Optional[str] = None,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        use_dbm: bool = False,
    ) -> np.ndarray:
        """Interpolate samples onto a 2D top-down floor plan grid.

        The output is a float32 array where values represent signal strength
        (either normalized 0-1 intensity or raw dBm, depending on use_dbm).

        Args:
            samples:  List of RFSample records (pre-filtered to one SSID).
            method:   Interpolation method override ("idw", "gaussian", "kriging").
            x_range:  Optional (min, max) world X bounds. Auto-detected if None.
            y_range:  Optional (min, max) world Y bounds. Auto-detected if None.
            use_dbm:  If True, output raw dBm values; else normalized 0-1 intensity.

        Returns:
            float32 array of shape (grid_size, grid_size), range [0,1] or [dBm].
            Returns zero grid if fewer than 2 samples provided.
        """
        chosen_method = (method or self.method).lower()

        if len(samples) < 2:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Build coordinate arrays
        xs = np.array([s.x for s in samples], dtype=np.float64)
        ys = np.array([s.y for s in samples], dtype=np.float64)
        dbm_vals = np.array([s.dbm for s in samples], dtype=np.float64)

        # Normalize dBm → 0–1 intensity using config range
        if use_dbm:
            values = dbm_vals
        else:
            values = (dbm_vals - float(config.DBM_MIN)) / (
                float(config.DBM_MAX) - float(config.DBM_MIN)
            )
            values = np.clip(values, 0.0, 1.0)

        # Determine bounds
        x_lo = float(xs.min()) if x_range is None else float(x_range[0])
        x_hi = float(xs.max()) if x_range is None else float(x_range[1])
        y_lo = float(ys.min()) if y_range is None else float(y_range[0])
        y_hi = float(ys.max()) if y_range is None else float(y_range[1])

        # Expand degenerate bounds
        if x_hi - x_lo < 0.01:
            x_lo -= 0.5
            x_hi += 0.5
        if y_hi - y_lo < 0.01:
            y_lo -= 0.5
            y_hi += 0.5

        # Build query grid
        gx = np.linspace(x_lo, x_hi, self.grid_size)
        gy = np.linspace(y_lo, y_hi, self.grid_size)
        grid_x, grid_y = np.meshgrid(gx, gy)  # both (G, G)
        query = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (G*G, 2)

        points = np.column_stack([xs, ys])  # (N, 2)

        # Run chosen interpolation
        result_flat = self._dispatch(chosen_method, points, values, query)

        grid = result_flat.reshape(self.grid_size, self.grid_size)
        if not use_dbm:
            grid = np.clip(grid, 0.0, 1.0)

        return grid.astype(np.float32)

    def interpolate_3d(
        self,
        samples: List[RFSample],
        method: Optional[str] = None,
        voxel_resolution: float = config.VIZ_VOXEL_RESOLUTION,
        use_dbm: bool = False,
    ) -> Tuple[np.ndarray, Tuple[float, float, float], float]:
        """Interpolate samples onto a 3D voxel grid.

        Args:
            samples:          List of RFSample records.
            method:           Interpolation method override.
            voxel_resolution: World-space size of each voxel (meters).
            use_dbm:          If True, output raw dBm; else normalized 0-1.

        Returns:
            Tuple of:
            - float32 array of shape (D, H, W) — voxel grid
            - origin (x0, y0, z0) — world coordinate of voxel [0,0,0]
            - voxel_resolution — resolution used

        Returns zero 1×1×1 grid if fewer than 2 samples.
        """
        chosen_method = (method or self.method).lower()

        if len(samples) < 2:
            return (
                np.zeros((1, 1, 1), dtype=np.float32),
                (0.0, 0.0, 0.0),
                voxel_resolution,
            )

        xs = np.array([s.x for s in samples], dtype=np.float64)
        ys = np.array([s.y for s in samples], dtype=np.float64)
        zs = np.array([s.z for s in samples], dtype=np.float64)
        dbm_vals = np.array([s.dbm for s in samples], dtype=np.float64)

        if use_dbm:
            values = dbm_vals
        else:
            values = np.clip(
                (dbm_vals - float(config.DBM_MIN)) / (
                    float(config.DBM_MAX) - float(config.DBM_MIN)
                ),
                0.0, 1.0,
            )

        res = max(0.01, float(voxel_resolution))
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        z0, z1 = float(zs.min()), float(zs.max())

        # Ensure minimum extent
        if x1 - x0 < res:
            x0 -= res; x1 += res
        if y1 - y0 < res:
            y0 -= res; y1 += res
        if z1 - z0 < res:
            z0 -= res; z1 += res

        W = max(2, int(round((x1 - x0) / res)) + 1)
        H = max(2, int(round((y1 - y0) / res)) + 1)
        D = max(2, int(round((z1 - z0) / res)) + 1)

        # Cap voxel grid to avoid OOM on Pi Zero 2W
        MAX_VOXELS = 50 * 50 * 10
        while W * H * D > MAX_VOXELS:
            W = max(2, W // 2)
            H = max(2, H // 2)
            D = max(2, D // 2)

        gx = np.linspace(x0, x1, W)
        gy = np.linspace(y0, y1, H)
        gz = np.linspace(z0, z1, D)
        vgx, vgy, vgz = np.meshgrid(gx, gy, gz, indexing="ij")  # (W, H, D)
        query = np.column_stack([vgx.ravel(), vgy.ravel(), vgz.ravel()])

        points = np.column_stack([xs, ys, zs])
        result_flat = self._dispatch(chosen_method, points, values, query)
        grid = result_flat.reshape(W, H, D)
        if not use_dbm:
            grid = np.clip(grid, 0.0, 1.0)

        return (
            grid.astype(np.float32),
            (x0, y0, z0),
            res,
        )

    def _dispatch(
        self,
        method: str,
        points: np.ndarray,
        values: np.ndarray,
        query: np.ndarray,
    ) -> np.ndarray:
        """Route to the appropriate interpolation backend."""
        if method == "idw":
            return idw_interpolate(points, values, query, power=self.idw_power)
        elif method == "gaussian":
            return gaussian_rbf_interpolate(points, values, query)
        elif method == "kriging":
            return kriging_interpolate(
                points, values, query,
                length_scale=self.kriging_length_scale,
            )
        else:
            LOGGER.warning("Unknown interpolation method '%s', using IDW.", method)
            return idw_interpolate(points, values, query, power=self.idw_power)


# ---------------------------------------------------------------------------
# Standalone demo / benchmark
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFInterpolator self-test")
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--method", type=str, default="gaussian",
                        choices=["idw", "gaussian", "kriging"])
    return parser.parse_args()


def main() -> int:
    _configure_logging()
    args = _parse_args()
    LOGGER.info("=== RFInterpolator Demo ===")
    LOGGER.info("Grid: %dx%d  Samples: %d  Method: %s",
                args.grid_size, args.grid_size, args.n_samples, args.method)

    # Synthetic samples along a diagonal path
    rng = np.random.default_rng(42)
    t = np.linspace(0, 5, args.n_samples)
    from rf_database import RFSample
    samples = [
        RFSample(
            x=float(t[i]),
            y=float(t[i] * 0.5 + rng.uniform(-0.2, 0.2)),
            z=0.0,
            ssid="TestNet",
            bssid="aa:bb:cc:dd:ee:ff",
            dbm=float(-40.0 - i * 1.5 + rng.uniform(-3, 3)),
            freq=2462,
            timestamp=float(i),
        )
        for i in range(args.n_samples)
    ]

    interp = RFInterpolator(grid_size=args.grid_size, method=args.method)

    for method in ["idw", "gaussian", "kriging"]:
        t0 = time.perf_counter()
        grid = interp.interpolate_2d(samples, method=method)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        LOGGER.info(
            "Method=%-8s  grid=%s  min=%.3f  max=%.3f  time=%.1f ms",
            method, grid.shape, float(grid.min()), float(grid.max()), elapsed_ms,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
