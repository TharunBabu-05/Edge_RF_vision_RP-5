# Engineering update — 7 November 2026

This update improves the root Python application and its browser interfaces. The vendored research repositories are reference material, not part of the running pipeline. No hardware performance or accuracy certification is implied.

## User-facing additions

- Responsive dark survey dashboard with camera preview, health/freshness indicators, strongest-signal trend and animated isometric RF sample preview.
- Search networks by SSID/BSSID; filter by frequency band or select an SSID for the spatial preview.
- Pause/resume acquisition; archive the current survey and reset to a new origin; download past surveys and current CSV/JSON exports.
- Explicit `--demo` mode for Windows or disconnected development. Camera, WiFi, motion and IMU are labeled synthetic and stored separately from live samples.
- Quaternion-smoothed yaw, pitch and roll in both Three.js viewers, with neutral-orientation recentering and saved-view PNGs.
- Reduced-motion styling, hidden-tab rendering/poll suppression, bounded fetch timeouts and visible reconnection status.
- Pinned, locally served Three.js 0.164.1 and OrbitControls with the upstream MIT license. The new dashboard uses no remote fonts or scripts.

## Reliability fixes

| Area | Fix |
| --- | --- |
| WiFi scanner | Failed scans retain cached data without advancing the successful timestamp. Atomic snapshots expose freshness and generation. Manual scans cannot overlap a radio scan; subprocess fallbacks share a time budget. Empty successful scans are valid. |
| Samples | Immutable validated records, transactional SQLite persistence, rolling retention, correct zero/negative query limits, no displacement advancement for empty/invalid scans, atomic JSON archive writes. |
| Coordinate sessions | Recovered surveys start paused; resetting an origin with existing samples is blocked. Starting a new survey archives first and avoids mixing the previous scan into the new origin. |
| Video | Encode buffers are locked/copied; failed camera reads are throttled; frames are resized consistently; event timeouts do not reprocess frames; worker failures appear in health; MJPEG suppresses duplicate sequences and limits viewers. |
| VO | Rotation composition follows camera-to-world order; heading axis corrected; near-zero optical flow and failed triangulation cannot add a unit translation. Pose reset/update is synchronized. Samples require TRACKING and configured confidence. |
| Interpolation | IDW operates in bounded batches, RBF uses 64-neighbor support, ordinary kriging enforces the constant-mean constraint and caps support at 256 points. 3D grids use D/H/W order and report their actual isotropic spacing after coarsening. |
| Reconstruction | Cancellation and timeout invalidate late workers before save/publish. Output replacement is atomic; simultaneous jobs are prevented. Plane SVD avoids allocating an N×N matrix. |
| Geometry | Plane corners lie on their fitted plane. Plane roles are classified by orientation/location instead of assumed RANSAC order. Room clouds use a consistent up-axis/origin; furniture centers/dimensions are transformed consistently and JSON handles NumPy scalars. |
| Models | Missing live depth models fail explicitly; network downloads are not triggered by default. Demo synthetic depth is labeled. Missing/failed furniture inference is reported; heuristic object labels are disabled by default. |
| IMU | Supported chip IDs are checked; missing sensors do not fabricate motion. Integration uses elapsed time; timestamps reflect successful reads. Browser polling is serialized with stale-data handling. |
| API | JSON object/finite-value/room-layout validation, request size limit, cross-origin write rejection, structured errors, scan freshness checks for manual samples, atomic serialized layout writes, revision-based floorplan refresh. |
| Browser | SSIDs/room names are escaped or inserted as text. 3D groups dispose geometry/materials/textures on rebuild. RSSI no longer creates fake vertical sample positions. Reconstruction polling handles cancellation/failure. |

## Verification

Local validation completed: **34 unit/API/regression tests passed**, including PNG export. JavaScript syntax checks passed. The Chrome browser smoke test passed with **zero page errors**, exercising the dashboard, mobile network filtering, both 3D pages, full synthetic four-shot reconstruction, IMU sync, scene display controls and PNG downloads. The browser checks exposed and led to fixes for missing scene-view globals and undefined Auto-Fit/Export handlers.

Automated regression tests are in `tests/`, limited by `pytest.ini` so hardware diagnostics and vendor research tests are not accidentally collected. Run them with `python -m pytest -q`.

The browser smoke test in `tools/browser_smoke.py` runs an isolated synthetic server, verifies desktop/mobile layouts, filters, recording controls, archive/new-survey behavior, orientation synchronization and the four-shot reconstruction UI. Screenshots and server logs are saved under `.dist/engineering/` and excluded from Git. The local runtime dependency lock is `requirements-lock-windows.txt`.

The added GitHub Actions workflow runs tests on Windows and Ubuntu after a push. That remote workflow has not been executed as part of this local update. `deploy/edgerf.service` is prepared for review and target-specific provisioning; it has not been installed on a Pi.

## Remaining limits and next acceptance work

- Metric localization needs calibrated intrinsics, a known scale/reference and a validated SLAM/VIO approach. This update does not implement SLAM, loop closure or IMU-position fusion.
- The four directional clouds use prescribed yaw rotations and relative depth, not optimized multi-view registration. A generated cuboid or furniture box is an estimate.
- The older 3D room editor rescales coordinates into manually drawn rooms. Room assignment intervals are browser-session state; durable per-sample room labels remain future work.
- RF interpolation is an estimate between observations, not a wall-aware electromagnetic solver. Kriging support is capped for edge memory/compute; it has no fitted variogram or uncertainty output.
- Authentication, TLS termination, target hardware permissions, ARM64 installation, sustained-load testing and field accuracy validation must be completed for a production pilot. The provided deployment binds loopback and expects controlled access.
- A timed-out inference thread is invalidated but not forcibly terminated. Repeated native-provider hangs require process isolation in a future reconstruction service.
- The existing legacy dashboard and specialized viewers remain available. They are not a complete modular frontend rewrite; the new overview is separated into Flask templates and static assets.

The old `.dist/project_review/PROJECT_EXPLAINED.md` remains a historical baseline. Use this document and `docs/OPERATIONS.md` for the updated runtime.
