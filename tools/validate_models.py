"""Check model checksums and run the application's real ONNX CPU wrappers."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional real input image, otherwise a synthetic colour pattern")
    parser.add_argument("--report", type=Path, help="Save validation results to JSON")
    args = parser.parse_args()
    import cv2
    import numpy as np
    import onnxruntime as ort
    from depth_estimator import DepthEstimator
    from furniture_detector import FurnitureDetector
    import config

    config.ALLOW_SYNTHETIC_DEPTH = False
    config.AUTO_DOWNLOAD_MODELS = False
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    for asset in manifest["models"]:
        path = ROOT / "models" / asset["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        # onnx is an optional development dependency, not required on the Pi.
        try:
            import onnx
        except ImportError:
            pass
        else:
            onnx.checker.check_model(str(path))

    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Cannot read image: {args.image}")
    else:
        y, x = np.mgrid[0:360, 0:640]
        image = np.stack((x % 256, y % 256, (x+y) % 256), axis=-1).astype(np.uint8)

    depth = DepthEstimator()
    detector = FurnitureDetector()
    report = dict(python=platform.python_version(), platform=platform.platform(),
                  onnxruntime=ort.__version__, image=str(args.image) if args.image else "synthetic colour pattern", models=[])
    for wrapper, infer, label in ((depth, depth.estimate, "depth"), (detector, detector.detect, "furniture")):
        session = wrapper._get_session()
        started = time.perf_counter()
        result = infer(image)
        elapsed = (time.perf_counter() - started) * 1000
        if wrapper._use_fallback:
            raise RuntimeError(f"{label} used fallback instead of ONNX inference")
        item = dict(task=label, providers=session.get_providers(),
                    inputs=[dict(name=x.name, shape=x.shape, dtype=x.type) for x in session.get_inputs()],
                    outputs=[dict(name=x.name, shape=x.shape) for x in session.get_outputs()], inference_ms=round(elapsed, 2))
        if label == "depth":
            if result.shape != image.shape[:2] or not np.isfinite(result).all() or result.min() < 0 or result.max() > config.DEPTH_SCALE_FACTOR + .001:
                raise ValueError("Depth output has invalid shape, values or range")
            item.update(result_shape=list(result.shape), result_min=float(result.min()), result_max=float(result.max()))
        else:
            item["detections"] = [x.to_dict() for x in result]
        report["models"].append(item)
    report["result"] = "passed"
    serialized = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
