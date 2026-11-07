"""Holistic 3D room scene reconstructor.

Implements the full multi-stage pipeline inspired by:
  • DeepPanoContext – ICCV 2021 (panoramic 3D scene understanding,
    object layout estimation via context graph)
  • holistic_scene_parsing – ECCV 2018 (joint parsing of room layout
    and 3D objects from a single RGB image)

Adapted to run entirely on CPU (no CUDA) on Raspberry Pi 5 using
lightweight ONNX models.

Pipeline
--------
  Stage 1  Depth estimation      (MiDaS-Small ONNX, ~150 ms/image)
  Stage 2  Furniture detection   (YOLOv8-nano ONNX, ~100 ms/image)
  Stage 3  Point cloud build     (depth map + camera intrinsics → 3-D pts)
  Stage 4  Multi-view fusion     (rotate clouds by 0°/90°/180°/270°, merge)
  Stage 5  Room geometry         (RANSAC plane fitting for walls/floor/ceiling)
  Stage 6  Furniture placement   (back-project YOLO boxes with depth → 3-D box)
  Stage 7  JSON export           (room_geometry.json for Three.js viewer)

Usage::

    from scene_reconstructor import SceneReconstructor
    rec = SceneReconstructor()
    result = rec.reconstruct({"N": bgr_n, "E": bgr_e, "S": bgr_s, "W": bgr_w})
    rec.save(result, "scene_output/room_geometry.json")
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from depth_estimator import DepthEstimator
from furniture_detector import FurnitureDetector, Detection

LOGGER = logging.getLogger("scene_reconstructor")


def _native(value):
    """Convert NumPy scalars/arrays in geometry to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WallPlane:
    """One reconstructed wall (or floor/ceiling) as a polygon."""
    role: str            # "floor" | "ceiling" | "wall_N" | "wall_E" | "wall_S" | "wall_W"
    normal: List[float]  # unit normal vector [nx, ny, nz]
    d: float             # plane equation: dot(normal, p) = d
    corners: List[List[float]]  # 4 corners in world coords [ [x,y,z], ... ]
    color: str = "#E8E8E8"


@dataclass
class FurnitureBox:
    """3-D axis-aligned bounding box for one detected furniture piece."""
    label: str
    color: str
    confidence: float
    # World-space centre
    cx: float
    cy: float
    cz: float
    # Dimensions
    width: float
    height: float
    depth_size: float
    # Source direction (which of the 4 shots detected this)
    source_direction: str = ""


@dataclass
class SceneGeometry:
    """Complete reconstructed room geometry."""
    room_width: float        # metres (X axis)
    room_depth: float        # metres (Y axis)
    room_height: float       # metres (Z axis)
    walls: List[WallPlane]
    furniture: List[FurnitureBox]
    walk_path: List[List[float]] = field(default_factory=list)
    rf_samples: List[Dict] = field(default_factory=list)
    # Reconstruction metadata
    capture_directions: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    point_cloud_size: int = 0
    quality: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return _native({
            "room_width":   self.room_width,
            "room_depth":   self.room_depth,
            "room_height":  self.room_height,
            "walls":        [asdict(w) for w in self.walls],
            "furniture":    [asdict(f) for f in self.furniture],
            "walk_path":    self.walk_path,
            "rf_samples":   self.rf_samples,
            "capture_directions": self.capture_directions,
            "timestamp":    self.timestamp,
            "point_cloud_size": self.point_cloud_size,
            "quality": self.quality,
        })


# ---------------------------------------------------------------------------
# Camera intrinsics helpers
# ---------------------------------------------------------------------------

def _default_intrinsics(width: int, height: int) -> np.ndarray:
    """Estimate a plausible pinhole camera matrix for a typical wide-angle Pi camera.

    Returns 3×3 K matrix.
    """
    fx = fy = max(width, height) * 0.85   # ~85° horizontal FoV
    cx, cy = width / 2.0, height / 2.0
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]], dtype=np.float64)


def _depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    max_depth: float = 8.0,
    stride: int = 4,
) -> np.ndarray:
    """Back-project a depth map to a 3-D point cloud.

    Args:
        depth:     H×W float32 depth map in metres.
        K:         3×3 camera intrinsic matrix.
        max_depth: Points deeper than this are discarded.
        stride:    Spatial subsampling factor to keep cloud manageable on Pi 5.

    Returns:
        (N, 3) float32 array of XYZ points in camera coordinates.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    zs = depth[ys, xs].astype(np.float64)

    mask = (zs > 0.1) & (zs < max_depth)
    xs, ys, zs = xs[mask], ys[mask], zs[mask]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def _rotation_y(degrees: float) -> np.ndarray:
    """3×3 rotation matrix around the Y (up) axis."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def _ransac_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    min_inliers: int = 100,
    n_iter: int = 200,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit one dominant plane to points using RANSAC.

    Returns (normal_vec, inlier_mask) or (None, None) if no plane found.
    """
    n = len(points)
    if n < 3:
        return None, None

    best_normal = None
    best_mask   = None
    best_count  = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm

        d = np.dot(normal, p1)
        dists = np.abs(points @ normal - d)
        mask = dists < distance_threshold
        count = mask.sum()

        if count > best_count:
            best_count  = count
            best_normal = normal
            best_mask   = mask

    if best_count < min_inliers:
        return None, None

    # Refine normal with all inliers via SVD
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    refined_normal = Vt[-1]
    if refined_normal[1] < 0:
        refined_normal = -refined_normal

    return refined_normal.astype(np.float32), best_mask


def _plane_corners(
    points: np.ndarray,
    normal: np.ndarray,
    room_w: float,
    room_d: float,
    room_h: float,
) -> List[List[float]]:
    """Compute 4 bounding-box corners of a plane from its inlier points."""
    if len(points) == 0:
        return []
    centre = points.mean(axis=0)
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    projected = (points - centre) @ np.column_stack((u, v))
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    return [(centre + a*u + b*v).tolist() for a,b in
            ((lo[0],lo[1]), (hi[0],lo[1]), (hi[0],hi[1]), (lo[0],hi[1]))]


# ---------------------------------------------------------------------------
# Furniture 3D box lifting
# ---------------------------------------------------------------------------

def _lift_furniture_box(
    det: Detection,
    depth_map: np.ndarray,
    K: np.ndarray,
    R_world: np.ndarray,
    label: str,
) -> Optional[FurnitureBox]:
    """Lift a 2-D furniture detection to a 3-D world-space bounding box.

    Uses the median depth within the 2D bounding box to estimate distance,
    then back-projects the box corners through the camera intrinsics.

    Args:
        det:       2-D detection (pixel coords).
        depth_map: H×W float32 depth map for this view.
        K:         Camera intrinsic matrix.
        R_world:   3×3 rotation matrix rotating camera → world frame.
        label:     String label for display.
    """
    h, w = depth_map.shape
    x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    roi_depth = depth_map[y1:y2, x1:x2]
    valid = roi_depth[(roi_depth > 0.1) & (roi_depth < 8.0)]
    if len(valid) < 10:
        return None

    # Use 30th percentile depth (front surface of object)
    z_obj = float(np.percentile(valid, 30))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Project box corners
    cx_2d = (x1 + x2) / 2.0
    cy_2d = (y1 + y2) / 2.0

    # World-space 3-D centre (camera coords first)
    X_cam = (cx_2d - cx) * z_obj / fx
    Y_cam = (cy_2d - cy) * z_obj / fy
    cam_centre = np.array([X_cam, Y_cam, z_obj], dtype=np.float32)
    world_centre = R_world @ cam_centre

    # Estimate object size from pixel box + depth (simple pinhole)
    box_w_m = (x2 - x1) * z_obj / fx
    box_h_m = (y2 - y1) * z_obj / fy
    box_d_m = min(box_w_m, box_h_m) * 0.5   # rough depth estimate

    color = config.FURNITURE_COLORS.get(det.label, config.FURNITURE_COLORS["default"])

    return FurnitureBox(
        label=label,
        color=color,
        confidence=det.confidence,
        cx=float(world_centre[0]),
        cy=float(world_centre[1]),
        cz=float(world_centre[2]),
        width=float(box_w_m),
        height=float(box_h_m),
        depth_size=float(box_d_m),
    )


# ---------------------------------------------------------------------------
# Main reconstructor
# ---------------------------------------------------------------------------

class SceneReconstructor:
    """Full holistic 3D room reconstruction from 4 directional images.

    Pipeline stages mirror the approach of DeepPanoContext and
    holistic_scene_parsing, adapted for CPU inference on Raspberry Pi 5.

    Usage::

        rec = SceneReconstructor()
        images = {"N": bgr_n, "E": bgr_e, "S": bgr_s, "W": bgr_w}
        geometry = rec.reconstruct(images)
        rec.save(geometry, "scene_output/room_geometry.json")
    """

    # Direction → Y-axis rotation angle so all views align to one world frame
    _DIR_ANGLES = {"N": 0.0, "E": -90.0, "S": 180.0, "W": 90.0}

    def __init__(
        self,
        voxel_size: float = config.SCENE_VOXEL_SIZE,
        wall_distance_threshold: float = config.SCENE_WALL_DISTANCE_THRESHOLD,
        wall_min_inliers: int = config.SCENE_WALL_MIN_INLIERS,
        room_height_estimate: float = config.SCENE_ROOM_HEIGHT_ESTIMATE,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.wall_dist  = float(wall_distance_threshold)
        self.wall_min   = int(wall_min_inliers)
        self.room_h_est = float(room_height_estimate)

        self._depth_est   = DepthEstimator()
        self._furniture   = FurnitureDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        images: Dict[str, np.ndarray],
        walk_path: Optional[List] = None,
        rf_samples: Optional[List] = None,
    ) -> SceneGeometry:
        """Run the full reconstruction pipeline.

        Args:
            images:     Dict of direction → BGR numpy array.
                        Keys should be in {"N","E","S","W"}.
            walk_path:  Optional VO walk path to embed in the scene.
            rf_samples: Optional list of RF samples to embed.

        Returns:
            SceneGeometry with walls, furniture boxes, and metadata.
        """
        t_start = time.perf_counter()
        directions = [d for d in self._DIR_ANGLES if d in images]
        LOGGER.info("Starting reconstruction from %d views: %s", len(directions), directions)

        # ── Stage 1 & 2: Depth + Furniture per view ──────────────────
        depth_maps: Dict[str, np.ndarray] = {}
        all_detections: Dict[str, List[Detection]] = {}

        for direction in directions:
            img = images[direction]
            LOGGER.info("[%s] Running depth estimation …", direction)
            depth_maps[direction] = self._depth_est.estimate(img)
            LOGGER.info("[%s] Running furniture detection …", direction)
            all_detections[direction] = self._furniture.detect(img)
            LOGGER.info("[%s] Found %d furniture objects.", direction, len(all_detections[direction]))

        # ── Stage 3: Build per-view point clouds ──────────────────────
        h, w = list(images.values())[0].shape[:2]
        K = _default_intrinsics(w, h)
        all_clouds: List[np.ndarray] = []

        for direction in directions:
            cloud_cam = _depth_to_pointcloud(depth_maps[direction], K)
            angle = self._DIR_ANGLES[direction]
            R = _rotation_y(angle)
            cloud_world = (R @ cloud_cam.T).T
            all_clouds.append(cloud_world)
            LOGGER.info("[%s] Point cloud: %d pts (rotated %.0f°)", direction, len(cloud_world), angle)

        # ── Stage 4: Fuse all clouds ──────────────────────────────────
        if not all_clouds:
            LOGGER.error("No point clouds to fuse!")
            return self._empty_geometry()

        merged = np.vstack(all_clouds)
        if not len(merged):
            raise ValueError("Depth model produced no valid points")
        # A single room frame: X right, Y up, Z forward, origin at its bounds.
        merged[:, 1] *= -1
        room_origin = merged.min(axis=0)
        merged -= room_origin
        LOGGER.info("Merged cloud: %d points total", len(merged))

        # Voxel downsample for speed
        merged = self._voxel_downsample(merged, self.voxel_size)
        LOGGER.info("After voxel downsample (%0.2f m): %d points", self.voxel_size, len(merged))

        # ── Stage 5: Room geometry from RANSAC planes ──────────────────
        walls, room_w, room_d, room_h = self._extract_room_geometry(merged)

        # ── Stage 6: Furniture 3D boxes ──────────────────────────────
        furniture_boxes: List[FurnitureBox] = []
        for direction in directions:
            angle = self._DIR_ANGLES[direction]
            R_world = _rotation_y(angle)
            for det in all_detections[direction]:
                label = config.YOLO_FURNITURE_CLASSES.get(det.class_id, det.label)
                box = _lift_furniture_box(
                    det, depth_maps[direction], K, R_world, label
                )
                if box is not None:
                    centre = np.array([box.cx, -box.cy, box.cz]) - room_origin
                    box.cx, box.cy, box.cz = float(centre[0]), float(centre[2]), float(centre[1])
                    dimensions = np.abs(R_world) @ np.array([box.width, box.height, box.depth_size])
                    box.width, box.height, box.depth_size = map(float, dimensions)
                    box.source_direction = direction
                    furniture_boxes.append(box)

        # ── Stage 6b: OpenCV fallback furniture detection ─────────────
        # If YOLO found nothing (model unavailable), use pure CV detection
        if not furniture_boxes and self._furniture._use_fallback and config.ALLOW_HEURISTIC_FURNITURE:
            LOGGER.info("YOLO found no furniture — running OpenCV-based detection as fallback")
            for direction in directions:
                img = images[direction]
                depth = depth_maps[direction]
                angle = self._DIR_ANGLES[direction]
                R_world = _rotation_y(angle)
                cv_boxes = self._detect_furniture_opencv(img, depth, K, R_world, direction, room_w, room_d, room_h)
                furniture_boxes.extend(cv_boxes)
                LOGGER.info("[%s] OpenCV furniture: %d items", direction, len(cv_boxes))

        # Deduplicate nearby boxes (same object seen from multiple directions)
        furniture_boxes = self._deduplicate_furniture(furniture_boxes)
        LOGGER.info("Furniture boxes after dedup: %d", len(furniture_boxes))

        elapsed = time.perf_counter() - t_start
        LOGGER.info("Reconstruction completed in %.1f s", elapsed)

        return SceneGeometry(
            room_width=float(room_w),
            room_depth=float(room_d),
            room_height=float(room_h),
            walls=walls,
            furniture=furniture_boxes,
            walk_path=walk_path or [],
            rf_samples=rf_samples or [],
            capture_directions=directions,
            timestamp=time.time(),
            point_cloud_size=len(merged),
            quality={
                "metric_calibrated": False,
                "depth": "synthetic_demo" if self._depth_est._use_fallback else "relative_monocular",
                "furniture": "unavailable" if self._furniture._use_fallback else "model",
                "registration": "fixed_direction_angles",
                "note": "Experimental reconstruction; dimensions are estimates. RF positions are not registered to the reconstructed room.",
            },
        )

    def save(self, geometry: SceneGeometry, path: str) -> str:
        """Save SceneGeometry to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(geometry.to_dict(), f, indent=2)
        LOGGER.info("Scene geometry saved → %s", target)
        return str(target.resolve())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
        """Simple voxel-grid downsample (no Open3D dependency)."""
        if len(points) == 0:
            return points
        indices = np.floor(points / voxel).astype(np.int32)
        _, unique_idx = np.unique(indices, axis=0, return_index=True)
        return points[unique_idx]

    def _detect_furniture_opencv(
        self,
        img: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
        R_world: np.ndarray,
        direction: str,
        room_w: float,
        room_d: float,
        room_h: float,
    ) -> List[FurnitureBox]:
        """Detect furniture objects (tables, chairs, cabinets) using OpenCV contour & depth segmentation.

        Used as a lightweight, zero-dependency fallback when ONNX/YOLO models are offline.
        """
        boxes: List[FurnitureBox] = []
        h, w = img.shape[:2]

        # Convert to grayscale and apply blur + adaptive threshold/edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Morphological close to join contour fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = h * w
        min_area = img_area * 0.015  # min 1.5% of frame
        max_area = img_area * 0.45   # max 45% of frame

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx_pin, cy_pin = float(K[0, 2]), float(K[1, 2])

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            # Skip objects right at top (ceiling lights/vents) or extreme edges
            if y < h * 0.15:
                continue

            # ROI depth check
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            roi_depth = depth_map[y1:y2, x1:x2]
            valid_d = roi_depth[(roi_depth > 0.2) & (roi_depth < 6.0)]
            if len(valid_d) < 15:
                continue

            z_obj = float(np.percentile(valid_d, 35))

            # 3D bounding box dimensions
            box_w_m = max(0.3, (x2 - x1) * z_obj / fx)
            box_h_m = max(0.3, (y2 - y1) * z_obj / fy)
            box_d_m = max(0.3, min(box_w_m, box_h_m) * 0.7)

            # Camera 3D center
            cx_2d = (x1 + x2) / 2.0
            cy_2d = (y1 + y2) / 2.0
            X_cam = (cx_2d - cx_pin) * z_obj / fx
            Y_cam = (cy_2d - cy_pin) * z_obj / fy
            cam_centre = np.array([X_cam, Y_cam, z_obj], dtype=np.float32)
            world_centre = R_world @ cam_centre

            # Translate to room box coordinates (offset from room center)
            w_x = room_w / 2.0 + float(world_centre[0])
            w_y = room_d / 2.0 + float(world_centre[2])
            w_z = max(0.0, float(world_centre[1]))

            # Clamp coordinates within room boundaries
            w_x = max(0.2, min(room_w - 0.2, w_x))
            w_y = max(0.2, min(room_d - 0.2, w_y))

            # Determine probable object label based on aspect ratio & size
            aspect = box_w_m / max(0.01, box_h_m)
            if aspect > 1.3 and box_w_m > 0.7:
                label = "table"
                color = "#3498DB"  # Blue
            elif aspect < 0.95 and box_h_m > 0.8:
                label = "chair"
                color = "#E74C3C"  # Red
            else:
                label = "furniture"
                color = "#2ECC71"  # Green

            boxes.append(FurnitureBox(
                label=label,
                color=color,
                confidence=0.75,
                cx=w_x,
                cy=w_y,
                cz=w_z,
                width=box_w_m,
                height=box_h_m,
                depth_size=box_d_m,
                source_direction=direction,
            ))

        return boxes

    def _extract_room_geometry(
        self, cloud: np.ndarray
    ) -> Tuple[List[WallPlane], float, float, float]:
        """Extract walls, floor, and ceiling using iterative RANSAC.

        Inspired by the layout estimation stage of DeepPanoContext.
        Returns (planes, room_width, room_depth, room_height).
        """
        planes: List[WallPlane] = []
        remaining = cloud.copy()
        mid = (cloud.min(axis=0) + cloud.max(axis=0)) / 2 if len(cloud) else np.zeros(3)
        for _ in range(6):
            if len(remaining) < self.wall_min:
                break
            normal, mask = _ransac_plane(
                remaining,
                distance_threshold=self.wall_dist,
                min_inliers=self.wall_min,
            )
            if normal is None:
                break
            inliers = remaining[mask]
            remaining = remaining[~mask]
            axis = int(np.argmax(np.abs(normal)))
            if abs(normal[axis]) < 0.7:
                continue
            upper = inliers.mean(axis=0)[axis] > mid[axis]
            role = (("wall_W", "wall_E"), ("floor", "ceiling"), ("wall_N", "wall_S"))[axis][int(upper)]
            if any(p.role == role for p in planes):
                continue
            color = "#D4C5A9" if role == "floor" else "#E8E8E4"
            d_val = float(np.dot(normal, inliers.mean(axis=0)))
            corners = _plane_corners(inliers, normal, 0, 0, 0)

            planes.append(WallPlane(
                role=role,
                normal=normal.tolist(),
                d=d_val,
                corners=corners,
                color=color,
            ))
            LOGGER.info("Extracted plane '%s' with %d inliers.", role, mask.sum())

        # Compute room dimensions from cloud bounding box
        if len(cloud) > 0:
            mn = cloud.min(axis=0)
            mx = cloud.max(axis=0)
            room_w = max(float(mx[0] - mn[0]), 2.0)
            room_d = max(float(mx[2] - mn[2]), 2.0)
            room_h = max(float(mx[1] - mn[1]), self.room_h_est)
        else:
            room_w = room_d = 4.0
            room_h = self.room_h_est

        # If we couldn't detect floor/ceiling, synthesise them
        if not any(p.role == "floor" for p in planes):
            mn = cloud.min(axis=0) if len(cloud) > 0 else np.zeros(3)
            mx = cloud.max(axis=0) if len(cloud) > 0 else np.array([room_w, room_h, room_d])
            planes.insert(0, WallPlane(
                role="floor", normal=[0, 1, 0], d=float(mn[1]),
                corners=[[mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                         [mx[0], mn[1], mx[2]], [mn[0], mn[1], mx[2]]],
                color="#D4C5A9",
            ))
        if not any(p.role == "ceiling" for p in planes):
            mn = cloud.min(axis=0) if len(cloud) > 0 else np.zeros(3)
            mx = cloud.max(axis=0) if len(cloud) > 0 else np.array([room_w, room_h, room_d])
            planes.append(WallPlane(
                role="ceiling", normal=[0, -1, 0], d=float(mx[1]),
                corners=[[mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
                         [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]]],
                color="#F0EFEB",
            ))

        return planes, room_w, room_d, room_h

    @staticmethod
    def _deduplicate_furniture(boxes: List[FurnitureBox], dist_thresh: float = 0.8) -> List[FurnitureBox]:
        """Remove duplicate furniture boxes seen from multiple directions.

        Two boxes of the same label within dist_thresh metres of each other
        are merged (keeping the higher confidence one).
        """
        if not boxes:
            return []
        kept: List[FurnitureBox] = []
        for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
            duplicate = False
            for k in kept:
                if k.label != box.label:
                    continue
                dist = math.sqrt(
                    (k.cx - box.cx) ** 2 +
                    (k.cy - box.cy) ** 2 +
                    (k.cz - box.cz) ** 2
                )
                if dist < dist_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
        return kept

    def _empty_geometry(self) -> SceneGeometry:
        """Return a fallback empty-room geometry (simple cuboid)."""
        rw, rd, rh = 5.0, 5.0, 2.5
        return SceneGeometry(
            room_width=rw, room_depth=rd, room_height=rh,
            walls=self._make_cuboid_walls(rw, rd, rh),
            furniture=[], capture_directions=[],
            timestamp=time.time(), point_cloud_size=0,
        )

    @staticmethod
    def _make_cuboid_walls(w: float, d: float, h: float) -> List[WallPlane]:
        return [
            WallPlane("floor",   [0, 1, 0],  0.0, [[0,0,0],[w,0,0],[w,0,d],[0,0,d]], "#D4C5A9"),
            WallPlane("ceiling", [0,-1, 0],  h,   [[0,h,0],[w,h,0],[w,h,d],[0,h,d]], "#F0EFEB"),
            WallPlane("wall_N",  [0, 0,-1],  0.0, [[0,0,0],[w,0,0],[w,h,0],[0,h,0]], "#E8E8E4"),
            WallPlane("wall_S",  [0, 0, 1],  d,   [[0,0,d],[w,0,d],[w,h,d],[0,h,d]], "#E8E8E4"),
            WallPlane("wall_W",  [1, 0, 0],  0.0, [[0,0,0],[0,0,d],[0,h,d],[0,h,0]], "#DCDCD8"),
            WallPlane("wall_E",  [-1,0, 0],  w,   [[w,0,0],[w,0,d],[w,h,d],[w,h,0]], "#DCDCD8"),
        ]


# ---------------------------------------------------------------------------
# Standalone demo / test
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="SceneReconstructor self-test")
    parser.add_argument("--images-dir", type=str, default=None,
                        help="Directory with N.jpg E.jpg S.jpg W.jpg (optional)")
    parser.add_argument("--out", type=str, default="scene_output/room_geometry.json")
    args = parser.parse_args()

    rec = SceneReconstructor()
    images: Dict[str, np.ndarray] = {}

    if args.images_dir:
        img_dir = Path(args.images_dir)
        for direction in ["N", "E", "S", "W"]:
            for ext in [".jpg", ".jpeg", ".png"]:
                p = img_dir / f"{direction}{ext}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        images[direction] = img
                        LOGGER.info("Loaded %s: %s", direction, p)
                        break

    # Fill missing directions with synthetic images
    for direction in ["N", "E", "S", "W"]:
        if direction not in images:
            LOGGER.info("Generating synthetic frame for direction %s", direction)
            synth = np.zeros((480, 640, 3), dtype=np.uint8)
            synth[:] = (30 + hash(direction) % 50, 60, 80)
            cv2.putText(synth, f"Direction: {direction}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
            images[direction] = synth

    geometry = rec.reconstruct(images)
    LOGGER.info(
        "Room: %.1f m × %.1f m × %.1f m | walls=%d | furniture=%d",
        geometry.room_width, geometry.room_depth, geometry.room_height,
        len(geometry.walls), len(geometry.furniture),
    )

    out_path = rec.save(geometry, args.out)
    LOGGER.info("Saved → %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
