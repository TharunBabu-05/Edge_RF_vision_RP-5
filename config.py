"""Central configuration for the WiFi signal visualizer project.

Hardware target: Raspberry Pi 5 (16 GB RAM, quad-core Cortex-A76 @ 2.4 GHz).
Edit this file to tune camera, WiFi scanning, signal processing,
RF field behaviour, and server settings.
"""

# ── Camera ───────────────────────────────────────────────────────────────────
# Pi 5 targets: ≥ 25 FPS at 1280×720 on V3 Camera / USB UVC camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280              # Pi 5: 1280×720 (was 640×480 on Pi Zero 2W)
CAMERA_HEIGHT = 720
CAMERA_FPS = 30                  # Pi 5 can sustain 30 FPS (was 15 on Pi Zero 2W)
JPEG_QUALITY = 80                # Higher quality thanks to Pi 5 CPU
PROCESS_EVERY_N_FRAMES = 1       # Pi 5: process every frame (was 2 on Pi Zero 2W)
OPENCV_NUM_THREADS = 4           # Pi 5 has 4 cores (was 2 on Pi Zero 2W)

# ── WiFi ─────────────────────────────────────────────────────────────────────
WIFI_INTERFACE = "wlan0"
SCAN_INTERVAL = 2.0              # Pi 5: faster polling (was 4.0 s on Pi Zero 2W)

# ── Signal Processing ────────────────────────────────────────────────────────
DBM_MAX = -30                    # Strongest signal (maps to 1.0)
DBM_MIN = -90                    # Weakest signal (maps to 0.0)
SMOOTHING_ALPHA = 0.3

# ── RF Field Model ───────────────────────────────────────────────────────────
FIELD_SCALE = 10                 # Downsample factor (1280/10 = 128 wide)
GAUSSIAN_SIGMA = 8.0
DECAY_FACTOR = 0.95

# ── Visualization ─────────────────────────────────────────────────────────────
DEFAULT_MODE = "combined"
OVERLAY_ALPHA = 0.4              # Heatmap transparency

# ── Spatial Mapper ───────────────────────────────────────────────────────────
MAPPER_GRID_WIDTH = 160          # Larger grid for Pi 5 (was 120)
MAPPER_GRID_HEIGHT = 120         # Larger grid for Pi 5 (was 90)
MAPPER_SIGMA = 3.0
MAPPER_DECAY = 0.995
MAPPER_VIEW_WIDTH = 480          # Larger preview (was 320)
MAPPER_VIEW_HEIGHT = 360         # Larger preview (was 240)
MAPPER_SNAPSHOT_DIR = ".mapper_snapshots"

# ── Guided Survey ─────────────────────────────────────────────────────────────
SURVEY_DEFAULT_ROWS = 3
SURVEY_DEFAULT_COLS = 3
SURVEY_DEFAULT_MARGIN = 0.08
SURVEY_DEFAULT_SERPENTINE = True
SURVEY_MIN_ROWS = 1
SURVEY_MAX_ROWS = 20
SURVEY_MIN_COLS = 1
SURVEY_MAX_COLS = 20
SURVEY_EXPORT_PREFIX = "survey_report"

# ── Visual Odometry ─────────────────────────────────────────────────────────
# Pi 5 target: ≤ 15 ms per frame (was ≤ 30 ms on Pi Zero 2W)
VO_MAX_FEATURES = 500            # Pi 5: more features (was 300 on Pi Zero 2W)
VO_MIN_MATCH_COUNT = 20          # Minimum tracked points for Essential Matrix
VO_SCALE_FACTOR = 1.0            # Metric scale (1.0 unit ≈ 1 metre when calibrated)
VO_POSE_CONFIDENCE_THRESHOLD = 0.5

# ── Spatial RF Database ──────────────────────────────────────────────────────
RF_DB_MAX_SAMPLES = 5000         # Pi 5 RAM: more samples (was 2000 on Pi Zero 2W)
RF_SAMPLE_MIN_DISTANCE = 0.05    # Finer resolution (was 0.1 m on Pi Zero 2W)
RF_DB_SNAPSHOT_DIR = ".rf_snapshots"

# ── RF Interpolation ─────────────────────────────────────────────────────────
# Pi 5 target: Gaussian ≤ 30 ms (was ≤ 80 ms on Pi Zero 2W)
INTERPOLATION_METHOD = "gaussian"    # "idw" | "gaussian" | "kriging"
INTERPOLATION_GRID_SIZE = 80         # Higher res (was 50 on Pi Zero 2W)
IDW_POWER = 2.0

# ── 3D Visualization ─────────────────────────────────────────────────────────
VIZ_VOXEL_RESOLUTION = 0.15     # Finer voxels (was 0.2 m on Pi Zero 2W)
VIZ_HEIGHT_SLICES = 8           # More Z slices (was 5 on Pi Zero 2W)
VIZ_RF_UPDATE_INTERVAL = 1.0   # Faster update (was 2.0 s on Pi Zero 2W)

# ── Server ───────────────────────────────────────────────────────────────────
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# ── Scene Reconstruction (Holistic 3D Room + Furniture) ──────────────────────
# Replaces the simple cuboid room model with a real 3D reconstruction built
# from 4 camera images (N / E / S / W). Inspired by:
#   • DeepPanoContext – ICCV 2021 (panoramic layout + object 3D boxes)
#   • holistic_scene_parsing – ECCV 2018 (multi-cue room/object parsing)
# Adapted to CPU-only ARM using ONNX quantized models for Raspberry Pi 5.

# ── 4-image capture ──────────────────────────────────────────────────────────
SCENE_CAPTURE_DIR = "scene_images"         # Directory for the 4 captured shots
SCENE_SHOTS = ["N", "E", "S", "W"]        # Directions; each ~90° apart
SCENE_CAPTURE_WIDTH = 1280                 # Resolution for reconstruction shots
SCENE_CAPTURE_HEIGHT = 960

# ── Depth estimation: MiDaS-Small ONNX (CPU-only, ~30 ms on Pi 5) ───────────
DEPTH_MODEL_PATH = "models/midas_v21_small_256.onnx"
DEPTH_INPUT_SIZE = 256
DEPTH_SCALE_FACTOR = 5.0        # Rough metric scale multiplier (tune per lens)

# ── Furniture detection: YOLOv8-nano ONNX (~6 MB, ~80 ms on Pi 5) ───────────
YOLO_MODEL_PATH = "models/yolov8n.onnx"
YOLO_CONFIDENCE_THRESHOLD = 0.40
YOLO_NMS_THRESHOLD = 0.45
YOLO_INPUT_SIZE = 640
# COCO class IDs that correspond to indoor furniture / objects
YOLO_FURNITURE_CLASSES = {
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    66: "keyboard",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    74: "clock",
    75: "vase",
}

# ── 3D room reconstruction (Open3D RANSAC + TSDF) ────────────────────────────
SCENE_OUTPUT_DIR = "scene_output"
SCENE_GEOMETRY_FILE = "room_geometry.json"
SCENE_POINT_CLOUD_FILE = "room_cloud.ply"
SCENE_VOXEL_SIZE = 0.05                  # TSDF voxel size in metres
SCENE_WALL_DISTANCE_THRESHOLD = 0.05     # RANSAC inlier distance (m)
SCENE_WALL_MIN_INLIERS = 150            # Min points to accept a plane as a wall
SCENE_ROOM_HEIGHT_ESTIMATE = 2.5        # Fallback ceiling height (m)
SCENE_RECONSTRUCTION_TIMEOUT = 120      # Max seconds for full pipeline

# Model downloads are an explicit provisioning step, never a surprise in a capture.
AUTO_DOWNLOAD_MODELS = False
ALLOW_SYNTHETIC_DEPTH = False          # Enabled only by --demo
ALLOW_HEURISTIC_FURNITURE = False      # Contours cannot reliably identify furniture

# ── Furniture 3D box colors in the viewer (label → hex) ─────────────────────
FURNITURE_COLORS = {
    "chair":        "#4A90D9",
    "couch":        "#7B68EE",
    "bed":          "#9B59B6",
    "dining table": "#E67E22",
    "tv":           "#2C3E50",
    "refrigerator": "#1ABC9C",
    "sink":         "#3498DB",
    "oven":         "#E74C3C",
    "toilet":       "#BDC3C7",
    "default":      "#95A5A6",
}
