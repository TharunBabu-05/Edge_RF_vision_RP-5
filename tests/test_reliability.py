import json
import threading
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from wifi_scanner import WifiScanner
from rf_database import RFDatabase, RFSample
from rf_interpolator import RFInterpolator, idw_interpolate, kriging_interpolate
from scene_capture import SceneCaptureManager
from scene_reconstructor import SceneGeometry, _ransac_plane
from imu_reader import IMUReader

NET = dict(ssid="Lab", bssid="02:00:00:00:00:01", dbm=-55, freq=2437)
SCAN = "BSS 02:00:00:00:00:01\nSSID: Lab\nsignal: -55.00 dBm\nfreq: 2437"


def test_scan_failure_does_not_become_new_measurement(monkeypatch):
    scanner = WifiScanner()
    monkeypatch.setattr(scanner, "_run_scan_command", lambda: (True, SCAN, ""))
    scanner.scan_once()
    before = scanner.get_snapshot()
    monkeypatch.setattr(scanner, "_run_scan_command", lambda: (False, "", "Radio unavailable"))
    scanner.scan_once()
    after = scanner.get_snapshot()
    assert after["last_scan_epoch"] == before["last_scan_epoch"]
    assert after["generation"] == before["generation"]
    assert after["networks"] == before["networks"] and after["stale"]
    after["networks"][0]["ssid"] = "changed"
    assert scanner.get_networks()[0]["ssid"] == "Lab"


def test_empty_success_clears_networks(monkeypatch):
    scanner = WifiScanner()
    monkeypatch.setattr("wifi_scanner.subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert scanner._run_scan_command() == (True, "", "")
    scanner.scan_once()
    assert not scanner.get_snapshot()["stale"]
    assert scanner.get_snapshot()["generation"] == 1


def test_scan_lock_rejects_overlapping_manual_scan(monkeypatch):
    scanner = WifiScanner()
    scanner._scan_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            scanner.scan_once(blocking=False)
    finally:
        scanner._scan_lock.release()


def test_database_recovers_commits_and_clear(tmp_path):
    path = str(tmp_path / "survey.sqlite3")
    db = RFDatabase(max_samples=10, storage_path=path)
    for i in range(15):
        db.add_sample(i, 0, 0, [NET])
    recovered = RFDatabase(max_samples=10, storage_path=path)
    assert len(recovered.get_samples()) == 10
    assert recovered.get_samples()[0].x == 5
    assert recovered.add_sample(14, 0, 0, [NET]) == 0
    recovered.clear()
    assert RFDatabase(storage_path=path).sample_count() == 0


def test_storage_failure_does_not_publish_uncommitted_sample(monkeypatch):
    db = RFDatabase()
    def fail(*a, **kw): raise OSError("Disk full")
    monkeypatch.setattr(db, "_persist_locked", fail)
    with pytest.raises(OSError): db.add_sample(1, 0, 0, [NET])
    assert db.sample_count() == 0 and db.revision == 0


def test_bad_or_empty_samples_do_not_consume_position():
    db = RFDatabase()
    assert db.add_sample(0, 0, 0, []) == 0
    assert db.add_sample(0, 0, 0, [dict(NET, dbm=float("nan"))]) == 0
    assert db.add_sample(0, 0, 0, [NET]) == 1
    with pytest.raises(ValueError): db.add_sample(float("nan"), 0, 0, [NET])
    with pytest.raises(FrozenInstanceError): db.get_samples()[0].dbm = -20
    assert db.to_dict_list(max_items=0) == []
    assert db.to_dict_list(max_items=-1) == []


def test_ordinary_kriging_preserves_constant_even_far_away():
    points = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    result = kriging_interpolate(points, np.full(3, -65.), np.array([[100., 100.], [.3, .5]]))
    np.testing.assert_allclose(result, -65, atol=1e-7)


def test_idw_matches_reference_across_batches():
    rng = np.random.default_rng(5)
    points, query = rng.random((600, 2)), rng.random((500, 2))
    values = rng.uniform(-85, -35, len(points))
    distances = np.maximum(np.linalg.norm(query[:, None] - points, axis=2), 1e-6)
    weights = 1 / distances**2
    expected = weights @ values / weights.sum(axis=1)
    np.testing.assert_allclose(idw_interpolate(points, values, query), expected)


def test_voxel_axes_and_actual_spacing_after_capping(monkeypatch):
    samples = [RFSample(0, 0, 0, **NET, timestamp=1), RFSample(100, 50, 20, **NET, timestamp=2)]
    interpolator = RFInterpolator()
    monkeypatch.setattr(interpolator, "_dispatch", lambda method, points, values, query: query[:, 0] + 10*query[:, 1] + 100*query[:, 2])
    grid, origin, spacing = interpolator.interpolate_3d(samples, voxel_resolution=.1, use_dbm=True)
    assert grid.size <= 25000 and spacing > .1
    assert grid[0, 0, 1] - grid[0, 0, 0] == pytest.approx(spacing)
    assert grid[0, 1, 0] - grid[0, 0, 0] == pytest.approx(10*spacing)
    assert grid[1, 0, 0] - grid[0, 0, 0] == pytest.approx(100*spacing)


class SlowReconstructor:
    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
        self.saved = False

    def reconstruct(self, *a, **kw):
        self.started.set()
        assert self.release.wait(3)
        return SceneGeometry(4, 5, 2.5, [], [])

    def save(self, geom, path):
        self.saved = True
        from pathlib import Path
        Path(path).write_text(json.dumps(geom.to_dict()))


def make_scene(tmp_path):
    backend = SlowReconstructor()
    manager = SceneCaptureManager(output_dir=str(tmp_path / "shots"), geometry_dir=str(tmp_path / "geometry"), reconstructor=backend)
    manager.start_session()
    manager.take_shot("N", np.zeros((40, 40, 3), np.uint8))
    manager.reconstruct()
    assert backend.started.wait(2)
    return manager, backend


def test_cancelled_reconstruction_never_publishes_or_saves(tmp_path):
    manager, backend = make_scene(tmp_path)
    manager.cancel()
    manager.start_session()
    backend.release.set()
    manager._recon_thread.join(3)
    assert manager.get_status().state == "capturing"
    assert manager.get_geometry() is None and not backend.saved


def test_reconstruction_timeout_discards_late_output(tmp_path, monkeypatch):
    monkeypatch.setattr("config.SCENE_RECONSTRUCTION_TIMEOUT", .05)
    manager, backend = make_scene(tmp_path)
    deadline = time.monotonic()+2
    while manager.get_status().state == "reconstructing" and time.monotonic()<deadline:
        time.sleep(.01)
    assert manager.get_status().state == "failed"
    backend.release.set()
    manager._recon_thread.join(3)
    assert not backend.saved and manager.get_geometry() is None


def test_successful_reconstruction_publishes_atomically(tmp_path):
    manager, backend = make_scene(tmp_path)
    backend.release.set()
    manager._recon_thread.join(3)
    assert manager.get_status().state == "ready" and backend.saved
    assert (tmp_path / "geometry" / "room_geometry.json").exists()


def test_plane_fit_on_large_inlier_set_uses_compact_svd(monkeypatch):
    original = np.linalg.svd
    def checked(array, **kwargs):
        assert kwargs.get("full_matrices") is False
        return original(array, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", checked)
    points = np.random.default_rng(2).random((3000,3)).astype(np.float32)
    points[:,1] = 1
    normal, mask = _ransac_plane(points, min_inliers=50, n_iter=10)
    assert abs(normal[1]) == pytest.approx(1) and mask.sum() == len(points)


def test_missing_imu_does_not_simulate(monkeypatch):
    monkeypatch.setattr(IMUReader, "_init_hardware", lambda self: False)
    reader = IMUReader()
    reader.start()
    try:
        time.sleep(.06)
        orientation = reader.get_orientation()
        assert orientation.timestamp == 0 and orientation.yaw == 0 and not orientation.hardware_detected
    finally:
        reader.stop()


def test_missing_depth_fails_explicitly_without_network(tmp_path,monkeypatch):
    from depth_estimator import DepthEstimator
    monkeypatch.setattr('config.AUTO_DOWNLOAD_MODELS',False)
    monkeypatch.setattr('config.ALLOW_SYNTHETIC_DEPTH',False)
    estimator=DepthEstimator(model_path=str(tmp_path/'absent.onnx'))
    with pytest.raises(RuntimeError,match='Depth model unavailable'):
        estimator.estimate(np.zeros((40,40,3),np.uint8))


def test_demo_reconstruction_geometry_serializes(tmp_path,monkeypatch):
    from scene_reconstructor import SceneReconstructor
    from depth_estimator import DepthEstimator
    from furniture_detector import FurnitureDetector
    monkeypatch.setattr('config.AUTO_DOWNLOAD_MODELS',False)
    monkeypatch.setattr('config.ALLOW_SYNTHETIC_DEPTH',True)
    reconstructor=SceneReconstructor.__new__(SceneReconstructor)
    reconstructor.voxel_size=.1
    reconstructor.wall_dist=.05
    reconstructor.wall_min=30
    reconstructor.room_h_est=2.5
    reconstructor._depth_est=DepthEstimator(model_path=str(tmp_path/'missing-depth.onnx'))
    reconstructor._furniture=FurnitureDetector(model_path=str(tmp_path/'missing-yolo.onnx'))
    image=np.zeros((96,128,3),np.uint8)
    geometry=reconstructor.reconstruct({key:image for key in ('N','E','S','W')})
    reconstructor.save(geometry,str(tmp_path/'geometry.json'))
    data=json.loads((tmp_path/'geometry.json').read_text())
    assert data['quality']['depth']=='synthetic_demo' and not data['quality']['metric_calibrated']
    assert data['furniture']==[]
    for wall in data['walls']:
        corners=np.asarray(wall['corners'])
        # A rendered polygon must be planar, including on tilted surfaces.
        normal=np.asarray(wall['normal'])
        assert np.ptp(corners@normal)<1e-4


def test_plane_corners_lie_in_tilted_plane():
    from scene_reconstructor import _plane_corners
    xy=np.random.default_rng(3).uniform(-2,2,(100,2))
    points=np.column_stack((xy,xy[:,0]+2*xy[:,1]+3))
    normal=np.array([-1.,-2.,1.]);normal/=np.linalg.norm(normal)
    corners=np.asarray(_plane_corners(points,normal,0,0,0))
    np.testing.assert_allclose(corners@normal,np.full(4,3/np.sqrt(6)),atol=1e-6)


def test_stationary_camera_does_not_gain_unit_translation():
    from pose_estimator import PoseEstimator
    estimator=PoseEstimator(width=320,height=240)
    points=np.random.default_rng(0).random((30,1,2)).astype(np.float32)
    assert estimator._recover_pose(points,points) is False
    np.testing.assert_array_equal(estimator._t_world,np.zeros((3,1)))


def test_pose_rejects_invalid_triangulation(monkeypatch):
    from pose_estimator import PoseEstimator
    estimator=PoseEstimator(width=320,height=240)
    monkeypatch.setattr('cv2.findEssentialMat',lambda *a,**kw:(np.eye(3),np.ones((30,1),np.uint8)))
    monkeypatch.setattr('cv2.recoverPose',lambda *a,**kw:(0,np.eye(3),np.ones((3,1)),np.zeros((30,1),np.uint8)))
    assert estimator._recover_pose(np.zeros((30,1,2)),np.ones((30,1,2))) is False
    np.testing.assert_array_equal(estimator._t_world,np.zeros((3,1)))
