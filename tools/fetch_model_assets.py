"""Provision ONNX assets from documented upstream release/export locations.

This helper is also usable on the Raspberry Pi; it needs only Python stdlib.
Hashes are recorded in models/manifest.json after local validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LICENSES = {
    "LICENSE.MiDaS.txt": "https://raw.githubusercontent.com/isl-org/MiDaS/master/LICENSE",
    "LICENSE.YOLO.txt": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, target, expected_sha256=None, minimum_size=1000000):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(Request(url, headers={"User-Agent": "EdgeRF-model-provisioner/1.0"}), timeout=90) as response:
            expected = int(response.headers.get("Content-Length", 0))
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
        size = temporary.stat().st_size
        if size < minimum_size or (expected and size != expected):
            raise ValueError(f"Incomplete download: {size} bytes, expected {expected or minimum_size}")
        if expected_sha256 and sha256(temporary) != expected_sha256:
            raise ValueError(f"Checksum mismatch for {target.name}; existing file left untouched")
        temporary.replace(target)
        print(f"Downloaded {target.name}: {size:,} bytes", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace existing model files")
    parser.add_argument("--sample", action="store_true", help="Download a COCO sample to ignored .dist/engineering/ for inference validation")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    for asset in manifest["models"]:
        target = ROOT / "models" / asset["filename"]
        if target.exists() and not args.force:
            if sha256(target) != asset["sha256"]:
                raise ValueError(f"Checksum mismatch: {target.name}. Use --force to replace it from its pinned source.")
            print(f"Verified: {target.name}", flush=True)
            continue
        fetch(asset["url"], target, expected_sha256=asset["sha256"])
    for name, url in LICENSES.items():
        target = ROOT / "models" / name
        if not target.exists() or args.force:
            fetch(url, target, minimum_size=100)
    if args.sample:
        fetch("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png", ROOT / ".dist" / "engineering" / "coco-sample.png", minimum_size=1000)


if __name__ == "__main__":
    main()
