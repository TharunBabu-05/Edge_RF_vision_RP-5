"""Survey operations and health endpoints, independent of presentation markup."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time
import uuid
from urllib.parse import urlsplit

from flask import Response, g, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException


def register_dashboard(app, runtime, worker):
    app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
    layout_lock = threading.RLock()
    runtime.origin_requires_reset = bool(runtime.rf_db.sample_count())
    if runtime.origin_requires_reset:
        runtime.recording = False

    @app.before_request
    def validate_request():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and urlsplit(origin).netloc != request.host:
                return jsonify(ok=False, error="Cross-origin changes are not allowed"), 403
            if request.data:
                body = request.get_json(silent=True)
                if not isinstance(body, dict):
                    return jsonify(ok=False, error="Request body must be a JSON object"), 400
                def finite(value):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError("Numbers must be finite")
                    if isinstance(value, dict):
                        for item in value.values(): finite(item)
                    elif isinstance(value, list):
                        for item in value: finite(item)
                try:
                    finite(body)
                    for key in ("x", "y", "margin"):
                        if key in body:
                            number = float(body[key])
                            if not math.isfinite(number) or not 0 <= number <= 1:
                                raise ValueError(f"{key} must be between 0 and 1")
                    if request.path == "/api/rooms/layout":
                        name = body.get("name", "My Home")
                        rooms = body.get("rooms")
                        if not isinstance(name, str) or not name.strip() or len(name) > 80:
                            raise ValueError("Layout name must contain 1–80 characters")
                        if not isinstance(rooms, list) or not 1 <= len(rooms) <= 100:
                            raise ValueError("Provide 1–100 rooms")
                        ids = set()
                        for room in rooms:
                            if not isinstance(room, dict) or not isinstance(room.get("id"), int) or room["id"] in ids:
                                raise ValueError("Each room needs a unique integer id")
                            ids.add(room["id"])
                            if not isinstance(room.get("name"), str) or not 1 <= len(room["name"]) <= 80:
                                raise ValueError("Each room needs a name of 1–80 characters")
                            for key in ("x", "y", "w", "d", "h"):
                                value = room.get(key)
                                if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1000:
                                    raise ValueError("Room coordinates and dimensions must be finite, within 1000")
                                if key in ("w", "d", "h") and value <= 0:
                                    raise ValueError("Room dimensions must be positive")
                        if not isinstance(body.get("nextId", 3), int) or body.get("nextId", 3) <= max(ids):
                            raise ValueError("nextId must exceed all existing room ids")
                    for key in ("name", "newName", "oldName"):
                        if key in body and (not isinstance(body[key], str) or len(body[key]) > 80):
                            raise ValueError("Names must be strings, at most 80 characters")
                    for key in ("rows", "cols"):
                        if key in body and (not isinstance(body[key], int) or not 1 <= body[key] <= 20):
                            raise ValueError("Survey rows and columns must be integers between 1 and 20")
                except (TypeError, ValueError, RecursionError) as exc:
                    return jsonify(ok=False, error=str(exc)), 400
            body = request.get_json(silent=True) or {}
            if request.path == "/api/survey/capture" or (request.path == "/api/mapper/position" and body.get("sample_now", True)):
                if runtime.scanner.get_snapshot()["stale"]:
                    return jsonify(ok=False, error="A fresh successful WiFi scan is required before sampling"), 409
        if request.path.startswith("/api/rooms/"):
            layout_lock.acquire()
            g.layout_locked = True

    @app.teardown_request
    def release_layout_lock(error):
        if getattr(g, "layout_locked", False):
            g.layout_locked = False
            layout_lock.release()

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(ok=False, error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        reference = uuid.uuid4().hex[:10]
        app.logger.exception("Request failed [%s]", reference)
        return jsonify(ok=False, error=f"Operation failed. Check server log reference {reference}."), 500

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    def overview():
        scan = runtime.scanner.get_snapshot()
        now = time.time()
        with runtime.lock:
            capture_age = now - runtime.last_capture_epoch if runtime.last_capture_epoch else None
            metrics = dict(fps=round(runtime.fps, 1), uptime_seconds=int(now - runtime.start_epoch),
                           camera_backend=runtime.camera.backend_name(), camera_age_seconds=capture_age,
                           pose=dict(x=runtime.pose_x, y=runtime.pose_y, z=runtime.pose_z,
                                     status=runtime.pose_status, confidence=runtime.pose_confidence))
        threads = {name: bool(thread and thread.is_alive()) for name, thread in
                   (("capture", worker._capture_thread), ("process", worker._process_thread), ("encode", worker._encode_thread))}
        healthy = all(threads.values()) and capture_age is not None and capture_age < 3 and not scan["stale"] and not runtime.pipeline_error
        return dict(ok=True, healthy=healthy, mode="demo" if runtime.demo else "hardware",
                    recording=runtime.recording, requires_new_survey=runtime.origin_requires_reset,
                    storage="SQLite" if runtime.rf_db.storage_path else "memory", revision=runtime.rf_db.revision,
                    scan=scan, database=asdict(runtime.rf_db.get_status()), workers=threads,
                    error=runtime.pipeline_error, coordinate_units="synthetic metres" if runtime.demo else "uncalibrated VO units",
                    **metrics)

    @app.get("/api/overview")
    def get_overview():
        return jsonify(overview())

    @app.get("/healthz")
    def health():
        data = overview()
        return jsonify(data), 200 if data["healthy"] else 503

    @app.post("/api/recording")
    def recording():
        body = request.get_json(silent=True) or {}
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return jsonify(ok=False, error="enabled must be true or false"), 400
        with runtime.acquisition_lock:
            if enabled and runtime.origin_requires_reset:
                return jsonify(ok=False, error="Recovered survey uses a previous origin. Start a new survey to record."), 409
            runtime.recording = enabled
        return jsonify(ok=True, recording=enabled)

    @app.post("/api/session/new")
    def new_session():
        with runtime.acquisition_lock:
            runtime.recording = False
            archive = None
            if runtime.rf_db.sample_count():
                folder = Path(runtime.rf_db.storage_path).parent if runtime.rf_db.storage_path else Path(".data")
                archive = folder / "archives" / f"{'demo' if runtime.demo else 'live'}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.json"
                runtime.rf_db.save_to_json(str(archive))
            runtime.rf_db.clear()
            runtime.mapper.clear()
            runtime.pose_estimator.reset()
            worker._last_mapper_scan_epoch = runtime.scanner.get_last_scan_epoch()
            runtime.origin_requires_reset = False
            runtime.recording = True
        return jsonify(ok=True, archived=archive.name if archive else None)

    @app.get("/api/rf/export/json")
    def export_json():
        data = dict(version=2, mode="demo" if runtime.demo else "hardware", units=overview()["coordinate_units"],
                    exported_at=time.time(), samples=runtime.rf_db.to_dict_list())
        return Response(json.dumps(data, allow_nan=False), mimetype="application/json",
                        headers={"Content-Disposition": 'attachment; filename="edgerf-survey.json"'})

    def archive_folder():
        return (Path(runtime.rf_db.storage_path).parent if runtime.rf_db.storage_path else Path(".data")) / "archives"

    @app.get("/api/sessions")
    def sessions():
        prefix = "demo-" if runtime.demo else "live-"
        files = sorted(archive_folder().glob(prefix + "*.json"), reverse=True)[:100]
        return jsonify(ok=True, sessions=[dict(name=f.name, size_bytes=f.stat().st_size, modified=f.stat().st_mtime) for f in files])

    @app.get("/api/sessions/<name>")
    def download_session(name):
        folder = archive_folder().resolve()
        target = (folder / name).resolve()
        prefix = "demo-" if runtime.demo else "live-"
        if target.parent != folder or not name.startswith(prefix) or target.suffix != ".json" or not target.is_file():
            return jsonify(ok=False, error="Survey archive not found"), 404
        return send_file(target, mimetype="application/json", as_attachment=True)
