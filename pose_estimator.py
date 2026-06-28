"""Lightweight Visual Odometry (VO) for indoor spatial pose estimation.

Designed for Raspberry Pi Zero 2W constraints (512 MB RAM, 1 GHz quad-core).
Uses OpenCV ORB keypoints, Lucas-Kanade optical flow, and Essential Matrix
decomposition to estimate incremental camera pose (X, Y, Z) in world coordinates.

Pipeline per frame:
    1. Detect ORB keypoints in current frame.
    2. Track points to next frame using Lucas-Kanade optical flow.
    3. Compute Essential Matrix from matched correspondences.
    4. Recover relative rotation (R) and translation (t).
    5. Integrate into running world pose.

Performance target: ≤ 30 ms per frame at 640×480 on Pi Zero 2W.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

LOGGER = logging.getLogger("pose_estimator")


@dataclass(frozen=True)
class PoseEstimate:
    """Immutable snapshot of the current estimated world pose."""

    x: float               # World X position (arbitrary metric units)
    y: float               # World Y position
    z: float               # World Z position (depth / forward motion)
    yaw: float             # Rotation around vertical axis (radians)
    confidence: float      # 0.0 = tracking lost, 1.0 = full confidence
    frame_count: int       # Total frames processed
    tracked_points: int    # Number of points tracked in last frame
    status: str            # "TRACKING", "INITIALIZING", "LOST"


@dataclass
class _FrameState:
    """Internal mutable state between consecutive frames."""

    gray: np.ndarray
    keypoints: np.ndarray           # Tracked point coordinates shape (N, 1, 2)
    descriptors: Optional[np.ndarray]


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale efficiently."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _detect_orb_points(
    gray: np.ndarray,
    orb: cv2.ORB,  # type: ignore[name-defined]
    max_features: int,
) -> np.ndarray:
    """Detect ORB keypoints and return as float32 point array for LK tracking.

    Returns:
        Array of shape (N, 1, 2) float32, or empty array if none found.
    """
    keypoints = orb.detect(gray, None)
    if not keypoints:
        return np.empty((0, 1, 2), dtype=np.float32)

    # Sort by response strength and cap at max_features
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_features]
    pts = cv2.KeyPoint_convert(keypoints)  # shape (N, 2)
    return pts.reshape(-1, 1, 2).astype(np.float32)


def _build_lk_params() -> dict:
    """Return Lucas-Kanade optical flow parameters tuned for Pi Zero 2W."""
    return dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )


def _default_camera_matrix(width: int, height: int) -> np.ndarray:
    """Build a simple pinhole camera matrix assuming typical webcam parameters."""
    focal_length = float(max(width, height))
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return np.array(
        [[focal_length, 0.0, cx], [0.0, focal_length, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class PoseEstimator:
    """Incremental pose estimator using ORB + Lucas-Kanade + Essential Matrix.

    Usage::

        estimator = PoseEstimator(width=640, height=480)
        for frame in camera_frames:
            pose = estimator.update(frame)
            print(pose.x, pose.y, pose.z)
    """

    def __init__(
        self,
        width: int = config.CAMERA_WIDTH,
        height: int = config.CAMERA_HEIGHT,
        max_features: int = config.VO_MAX_FEATURES,
        min_match_count: int = config.VO_MIN_MATCH_COUNT,
        scale_factor: float = config.VO_SCALE_FACTOR,
        confidence_threshold: float = config.VO_POSE_CONFIDENCE_THRESHOLD,
        camera_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize pose estimator.

        Args:
            width: Camera frame width in pixels.
            height: Camera frame height in pixels.
            max_features: Maximum ORB features to detect per frame.
            min_match_count: Minimum tracked points to attempt Essential Matrix.
            scale_factor: Metric scale (1 unit ≈ 1 meter when calibrated).
            confidence_threshold: Minimum confidence to report TRACKING status.
            camera_matrix: Optional 3x3 camera intrinsics. Auto-built if None.
        """
        self.width = int(width)
        self.height = int(height)
        self.max_features = max(10, int(max_features))
        self.min_match_count = max(5, int(min_match_count))
        self.scale_factor = float(scale_factor)
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

        self._K: np.ndarray = (
            camera_matrix.astype(np.float64)
            if camera_matrix is not None
            else _default_camera_matrix(self.width, self.height)
        )

        self._orb: cv2.ORB = cv2.ORB_create(  # type: ignore[attr-defined]
            nfeatures=self.max_features,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            patchSize=31,
        )
        self._lk_params = _build_lk_params()

        # World pose state
        self._R_world: np.ndarray = np.eye(3, dtype=np.float64)
        self._t_world: np.ndarray = np.zeros((3, 1), dtype=np.float64)
        self._yaw: float = 0.0

        # Previous frame state
        self._prev_state: Optional[_FrameState] = None

        # Counters
        self._frame_count: int = 0
        self._consecutive_failures: int = 0
        self._last_tracked_points: int = 0

        # Redetect every N frames to keep tracking fresh
        self._redetect_interval: int = 20
        self._frames_since_redetect: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> PoseEstimate:
        """Process one camera frame and return updated pose estimate.

        Args:
            frame: BGR or grayscale frame from camera.

        Returns:
            Current world pose estimate.
        """
        start = time.perf_counter()
        gray = _to_gray(frame)
        self._frame_count += 1

        pose = self._process_frame(gray)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > 35.0:
            LOGGER.debug("VO update slow: %.1f ms (frame %d)", elapsed_ms, self._frame_count)

        return pose

    def get_pose(self) -> PoseEstimate:
        """Return current pose without processing a new frame."""
        return self._build_pose_estimate("TRACKING" if self._prev_state else "INITIALIZING")

    def reset(self) -> None:
        """Reset pose to world origin and clear tracking state."""
        self._R_world = np.eye(3, dtype=np.float64)
        self._t_world = np.zeros((3, 1), dtype=np.float64)
        self._yaw = 0.0
        self._prev_state = None
        self._frame_count = 0
        self._consecutive_failures = 0
        self._last_tracked_points = 0
        self._frames_since_redetect = 0
        LOGGER.info("PoseEstimator reset to world origin.")

    def get_tracked_keypoints(self) -> Optional[np.ndarray]:
        """Return currently tracked keypoint positions for visualization.

        Returns:
            Float32 array of shape (N, 2) or None if no points tracked.
        """
        if self._prev_state is None or self._prev_state.keypoints is None:
            return None
        pts = self._prev_state.keypoints
        if pts.size == 0:
            return None
        return pts.reshape(-1, 2)

    # ------------------------------------------------------------------
    # Internal processing
    # ------------------------------------------------------------------

    def _process_frame(self, gray: np.ndarray) -> PoseEstimate:
        """Core VO processing for one grayscale frame."""

        # --- Initialization: first frame ---
        if self._prev_state is None:
            pts = _detect_orb_points(gray, self._orb, self.max_features)
            self._prev_state = _FrameState(gray=gray, keypoints=pts, descriptors=None)
            self._last_tracked_points = len(pts)
            return self._build_pose_estimate("INITIALIZING")

        # --- Periodic redetection to refresh stale keypoints ---
        self._frames_since_redetect += 1
        if self._frames_since_redetect >= self._redetect_interval or self._prev_state.keypoints.size < self.min_match_count * 4:
            fresh_pts = _detect_orb_points(self._prev_state.gray, self._orb, self.max_features)
            self._prev_state = _FrameState(
                gray=self._prev_state.gray,
                keypoints=fresh_pts,
                descriptors=None,
            )
            self._frames_since_redetect = 0

        prev_pts = self._prev_state.keypoints
        if prev_pts is None or prev_pts.size == 0:
            # No points at all — redetect and mark as initializing
            pts = _detect_orb_points(gray, self._orb, self.max_features)
            self._prev_state = _FrameState(gray=gray, keypoints=pts, descriptors=None)
            self._consecutive_failures += 1
            return self._build_pose_estimate("INITIALIZING")

        # --- Lucas-Kanade optical flow ---
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_state.gray,
            gray,
            prev_pts,
            None,
            **self._lk_params,
        )

        if curr_pts is None or status is None:
            self._on_tracking_failure(gray)
            return self._build_pose_estimate("LOST")

        good_mask = status.ravel() == 1
        good_prev = prev_pts[good_mask]
        good_curr = curr_pts[good_mask]
        n_good = int(good_mask.sum())
        self._last_tracked_points = n_good

        if n_good < self.min_match_count:
            self._on_tracking_failure(gray)
            return self._build_pose_estimate("LOST" if self._consecutive_failures > 5 else "INITIALIZING")

        # --- Essential Matrix + Pose Recovery ---
        pose_updated = self._recover_pose(good_prev, good_curr)

        # Update previous frame state with good tracked points
        new_pts = _detect_orb_points(gray, self._orb, self.max_features) if n_good < self.min_match_count * 2 else good_curr.reshape(-1, 1, 2)
        self._prev_state = _FrameState(gray=gray, keypoints=new_pts, descriptors=None)

        if pose_updated:
            self._consecutive_failures = 0
            return self._build_pose_estimate("TRACKING")
        else:
            self._consecutive_failures += 1
            return self._build_pose_estimate("TRACKING" if self._consecutive_failures < 3 else "INITIALIZING")

    def _recover_pose(self, prev_pts: np.ndarray, curr_pts: np.ndarray) -> bool:
        """Compute Essential Matrix and recover incremental pose.

        Args:
            prev_pts: Matched points in previous frame, shape (N, 2).
            curr_pts: Matched points in current frame, shape (N, 2).

        Returns:
            True if pose was updated successfully.
        """
        try:
            E, mask = cv2.findEssentialMat(
                curr_pts,
                prev_pts,
                self._K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
        except cv2.error:
            return False

        if E is None or E.shape != (3, 3):
            return False

        inlier_mask = mask.ravel() if mask is not None else None
        n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if n_inliers < self.min_match_count:
            return False

        try:
            _, R_delta, t_delta, recover_mask = cv2.recoverPose(
                E,
                curr_pts,
                prev_pts,
                self._K,
                mask=mask,
            )
        except cv2.error:
            return False

        if R_delta is None or t_delta is None:
            return False

        # Integrate incremental translation into world frame
        # t_delta is unit-scale; apply user scale_factor
        t_world_delta = self._R_world @ (t_delta * self.scale_factor)
        self._t_world = self._t_world + t_world_delta

        # Integrate rotation
        self._R_world = R_delta @ self._R_world

        # Extract yaw from rotation matrix
        self._yaw = float(np.arctan2(self._R_world[1, 0], self._R_world[0, 0]))

        return True

    def _on_tracking_failure(self, gray: np.ndarray) -> None:
        """Handle tracking failure by redetecting keypoints."""
        self._consecutive_failures += 1
        pts = _detect_orb_points(gray, self._orb, self.max_features)
        self._prev_state = _FrameState(gray=gray, keypoints=pts, descriptors=None)
        self._last_tracked_points = len(pts)

    def _build_pose_estimate(self, status: str) -> PoseEstimate:
        """Construct a PoseEstimate snapshot from current internal state."""
        x = float(self._t_world[0, 0])
        y = float(self._t_world[1, 0])
        z = float(self._t_world[2, 0])

        # Confidence: based on tracked points and consecutive failures
        if status == "LOST":
            confidence = 0.0
        elif status == "INITIALIZING":
            confidence = 0.3
        else:
            pt_ratio = min(1.0, float(self._last_tracked_points) / float(self.max_features))
            failure_penalty = min(1.0, float(self._consecutive_failures) * 0.15)
            confidence = float(np.clip(pt_ratio - failure_penalty, 0.0, 1.0))

        return PoseEstimate(
            x=x,
            y=y,
            z=z,
            yaw=float(self._yaw),
            confidence=confidence,
            frame_count=int(self._frame_count),
            tracked_points=int(self._last_tracked_points),
            status=status,
        )


# ---------------------------------------------------------------------------
# Standalone demo / self-test
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoseEstimator self-test (synthetic frames)")
    parser.add_argument("--frames", type=int, default=100, help="Number of synthetic frames to process")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT)
    parser.add_argument("--demo", action="store_true", help="Run synthetic motion demo")
    return parser.parse_args()


def _build_synthetic_frame(width: int, height: int, offset_x: float, offset_y: float) -> np.ndarray:
    """Build a synthetic frame with a shifted checkerboard pattern."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cell = 40
    for row in range(height // cell + 1):
        for col in range(width // cell + 1):
            x0 = int(col * cell + offset_x) % width
            y0 = int(row * cell + offset_y) % height
            x1 = min(x0 + cell, width)
            y1 = min(y0 + cell, height)
            color = (200, 200, 200) if (row + col) % 2 == 0 else (60, 60, 60)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = color

    # Add random feature dots to give ORB something to detect
    rng = np.random.default_rng(seed=42)
    for _ in range(80):
        cx = int(rng.integers(10, width - 10) + offset_x) % width
        cy = int(rng.integers(10, height - 10) + offset_y) % height
        cv2.circle(frame, (cx, cy), 5, (255, 180, 50), -1)

    return frame


def main() -> int:
    _configure_logging()
    args = _parse_args()

    LOGGER.info("=== PoseEstimator Self-Test ===")
    LOGGER.info("Synthetic frames: %d  Resolution: %dx%d", args.frames, args.width, args.height)

    estimator = PoseEstimator(width=args.width, height=args.height)

    timings: List[float] = []
    final_pose = None

    for i in range(args.frames):
        # Simulate slow left-right drift
        offset_x = float(i) * 0.5
        offset_y = float(i) * 0.2
        frame = _build_synthetic_frame(args.width, args.height, offset_x, offset_y)

        t0 = time.perf_counter()
        pose = estimator.update(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        timings.append(elapsed_ms)
        final_pose = pose

        if i % 20 == 0:
            LOGGER.info(
                "Frame %3d | status=%-12s | x=%+.3f y=%+.3f z=%+.3f | conf=%.2f | pts=%d | %.1fms",
                i,
                pose.status,
                pose.x, pose.y, pose.z,
                pose.confidence,
                pose.tracked_points,
                elapsed_ms,
            )

    avg_ms = float(np.mean(timings))
    max_ms = float(np.max(timings))
    LOGGER.info("\n=== Results ===")
    LOGGER.info("Avg update time: %.2f ms", avg_ms)
    LOGGER.info("Max update time: %.2f ms", max_ms)
    LOGGER.info("Target (≤30ms avg): %s", "PASS" if avg_ms <= 30.0 else "CHECK")
    if final_pose:
        LOGGER.info(
            "Final pose: x=%.3f y=%.3f z=%.3f status=%s",
            final_pose.x, final_pose.y, final_pose.z, final_pose.status,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
