"""Download the exact Three.js modules used by the application, with its license."""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "static" / "vendor" / "three"
FILES = ("build/three.module.js", "examples/jsm/controls/OrbitControls.js", "LICENSE")
for name in FILES:
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(f"https://cdn.jsdelivr.net/npm/three@0.164.1/{name}", timeout=45) as response:
        data = response.read()
    target.write_bytes(data)
    print(f"{name}: {len(data)} bytes")
