"""Central configuration for the WiFi signal visualizer project.

Edit this file to tune camera, WiFi scanning, signal processing,
RF field behavior, and server settings.
"""

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
JPEG_QUALITY = 65
PROCESS_EVERY_N_FRAMES = 2
OPENCV_NUM_THREADS = 2

# WiFi
WIFI_INTERFACE = "wlan0"
SCAN_INTERVAL = 4.0  # seconds between scans

# Signal Processing
DBM_MAX = -30  # strongest signal (maps to 1.0)
DBM_MIN = -90  # weakest signal (maps to 0.0)
SMOOTHING_ALPHA = 0.3

# RF Field Model
FIELD_SCALE = 10  # downsample factor (640/10 = 64 wide)
GAUSSIAN_SIGMA = 8.0
DECAY_FACTOR = 0.95

# Visualization
DEFAULT_MODE = "combined"
OVERLAY_ALPHA = 0.4  # heatmap transparency

# Stage 8 Spatial Mapper
MAPPER_GRID_WIDTH = 120
MAPPER_GRID_HEIGHT = 90
MAPPER_SIGMA = 3.0
MAPPER_DECAY = 0.995
MAPPER_VIEW_WIDTH = 320
MAPPER_VIEW_HEIGHT = 240
MAPPER_SNAPSHOT_DIR = ".mapper_snapshots"

# Stage 9 Guided Survey
SURVEY_DEFAULT_ROWS = 3
SURVEY_DEFAULT_COLS = 3
SURVEY_DEFAULT_MARGIN = 0.08
SURVEY_DEFAULT_SERPENTINE = True
SURVEY_MIN_ROWS = 1
SURVEY_MAX_ROWS = 20
SURVEY_MIN_COLS = 1
SURVEY_MAX_COLS = 20
SURVEY_EXPORT_PREFIX = "survey_report"

# Visual Odometry (pose_estimator.py)
VO_MAX_FEATURES = 300            # ORB features detected per frame
VO_MIN_MATCH_COUNT = 20          # Minimum tracked points for Essential Matrix
VO_SCALE_FACTOR = 1.0            # Metric scale (1.0 unit ≈ 1 metre when calibrated)
VO_POSE_CONFIDENCE_THRESHOLD = 0.5  # Confidence floor for TRACKING status

# Spatial RF Database (rf_database.py)
RF_DB_MAX_SAMPLES = 2000         # Rolling window total sample count
RF_SAMPLE_MIN_DISTANCE = 0.1     # Minimum displacement (m) between samples
RF_DB_SNAPSHOT_DIR = ".rf_snapshots"  # Persistent JSON snapshot directory

# RF Interpolation (rf_interpolator.py)
INTERPOLATION_METHOD = "gaussian"    # "idw" | "gaussian" | "kriging"
INTERPOLATION_GRID_SIZE = 50         # NxN grid resolution for 2D heatmap output
IDW_POWER = 2.0                       # IDW distance weighting exponent

# 3D Visualization
VIZ_VOXEL_RESOLUTION = 0.2      # Voxel size in world units (metres)
VIZ_HEIGHT_SLICES = 5            # Number of Z-level slices for 3D view
VIZ_RF_UPDATE_INTERVAL = 2.0    # Seconds between re-running interpolation

# Server
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
