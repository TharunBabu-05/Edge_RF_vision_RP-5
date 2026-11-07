# Running and validating EdgeRF

EdgeRF is an edge-device survey application with a Windows-compatible demo. The runtime now uses Waitress, persistent SQLite samples and local Three.js assets. One process owns the camera and radio. Do not launch multiple WSGI workers against the same device or database.

## Quick start on this Windows workspace

The project-local `.venv` is already installed. From the repository root:

```powershell
.\.venv\Scripts\python.exe main.py --demo --host 127.0.0.1
```

Open **http://127.0.0.1:5000**. Demo mode produces explicitly labeled synthetic camera frames, three WiFi networks, motion and IMU orientation. Its `.data/demo.sqlite3` is separate from `.data/live.sqlite3`. A recovered database starts paused because the camera's coordinate origin has changed. **New survey** archives the current samples, resets the origin and starts recording. **Past surveys** downloads archived JSON files.

For a clean installation, use Python 3.12+:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` pins the top-level versions actually tested on this Windows host. `requirements-lock-windows.txt` records the complete resolved runtime dependency set. Pi ARM64 wheel availability and performance still require validation; do not assume desktop test results establish Pi throughput.

## Raspberry Pi setup

Use 64-bit Raspberry Pi OS, Python 3.12+, a supported camera and a managed-mode WiFi interface. Install OS packages `iw`, `wireless-tools`, `v4l-utils` and, if the camera needs its fallback backend, `ffmpeg`. Enable I2C for the MPU sensor. Install `requirements-hardware.txt` in a virtual environment. Optional ONNX support is needed only for reconstruction.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-hardware.txt
.venv/bin/python main.py --host 127.0.0.1 --interface wlan0 --resolution 1280x720
```

Keep the IMU stationary during startup gyro calibration. Its complementary filter uses accelerometer/gyro samples; it does not read a magnetometer. Relative yaw therefore drifts. The browser now follows yaw, pitch and roll; **Recenter IMU** or **R** establishes a neutral orientation. The physical sensor mounting convention must be verified on the actual rig.

The dashboard remains available when hardware is missing, with degraded health. Hardware WiFi commands are Linux-specific. No real WiFi measurements are produced by the Windows demo.

## Deployment

`deploy/edgerf.service` is a template, not an installed service. Provision `/opt/edgerf`, its virtual environment and a dedicated `edgerf` user first. The service uses the `video` and `i2c` groups, grants only the scanner's `CAP_NET_ADMIN` capability, and binds loopback. Confirm these groups and `iw` capabilities work on the target OS. Place provisioned ONNX files in `/opt/edgerf/models` before starting.

Waitress uses 12 request threads; the app caps concurrent MJPEG clients at four so streams cannot consume every request thread. It is intended for a small local team, not hundreds of live viewers. The server has no user login or role system. For access beyond a trusted device/network, keep loopback binding and put an authenticated TLS reverse proxy or VPN in front of it. Origin checks protect browser writes, but are not authentication.

Use `--host 0.0.0.0` only when deliberately making the device accessible on your trusted LAN. The launch commands above use loopback. The original configuration default remains `0.0.0.0` for compatibility.

## Data and recovery

- Samples are transactionally committed to SQLite before being published in memory, capped at 5,000 records. A record represents one BSSID at a sample location.
- `.data/archives/` holds timestamped JSON survey archives. CSV/JSON exports remain available while recording is paused.
- A disk write error pauses acquisition and appears in diagnostics. Resolve the storage fault before resuming.
- Manual room layouts are stored beside the survey database using atomic replacement and serialized updates. Existing root `room_layouts.json` files remain compatible through migration.
- Room assignments in the older 3D editor are browser-session intervals, not durable room labels on each sample. Download raw surveys before reorganizing layouts.
- Back up the data directory. Stop the server before copying the database, or use SQLite's backup API for a consistent live backup.
- Reconstruction timeouts discard results and prevent overlapping workers. Python inference threads cannot be forcibly killed; a timed-out worker may need to finish before another job can start. Restart the process if a provider hangs indefinitely.

## Models and measurement limits

Live reconstruction no longer silently invents a depth map when MiDaS is missing. It reports an actionable failure. Provision a compatible `models/midas_v21_small_256.onnx` and optionally `models/yolov8n.onnx`; automatic downloads are disabled by default. A missing detector is reported as unavailable, not proof the room has no furniture. Synthetic depth is permitted only in explicit demo mode.

Monocular VO has unknown metric scale and drift. MiDaS estimates relative depth; fixed N/E/S/W rotations are approximate alignment, not multi-view registration. The dashboard uses raw measured sample coordinates, labeled as uncalibrated. The legacy room viewers rescale points into a layout and now label this explicitly. They no longer invent sample height from RSSI. In the reconstructed scene, RF points are displayed on a reference plane because RF-to-room registration is uncalibrated. Peaks in RSSI are not measured access-point coordinates. Do not use these outputs as a certified RF propagation or building survey.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node --check static/dashboard.js
node --check static/viewer-utils.js
.\.venv\Scripts\python.exe tools/browser_smoke.py
```

The browser script launches an isolated demo server on port 5056, uses installed Chrome headlessly, exercises the dashboard and both 3D viewers, and writes screenshots/logs to ignored `.dist/engineering/`. It terminates only the server process it created. It does not require real sensors. GitHub Actions runs unit/API tests and JavaScript syntax checks on Windows and Ubuntu; the workflow has been added locally, not run on GitHub yet.

`GET /api/overview` combines scan freshness, recording, storage, worker and camera status. `GET /healthz` returns 200 only with healthy camera/scanner/video stages; unavailable hardware returns 503. IMU readiness is reported separately by `/api/imu/orientation` because it is optional. An error response references the server log without exposing a traceback.

## Hardware acceptance before a production pilot

1. Verify camera recovery, WiFi permission/failure handling and MPU axis directions on the Pi.
2. Calibrate camera intrinsics and establish metric scale using known reference distances or a suitable VIO/SLAM implementation.
3. Compare repeat surveys against known positions and reference RSSI; measure drift and interpolation error for each band/BSSID.
4. Validate ONNX preprocessing/output compatibility and real room reconstruction against measured geometry.
5. Measure CPU/RAM, thermal throttling, dropped frames, disk behavior and four-client streaming over an extended run.
6. Validate backup/restore, access controls and deployment on the target network.

These are remaining acceptance tasks, not claims of completed hardware validation.
