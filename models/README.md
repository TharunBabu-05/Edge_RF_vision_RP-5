# EdgeRF 3D — ONNX Model Cache

This directory stores downloaded ONNX models used by the holistic 3D
scene reconstruction pipeline. Models are auto-downloaded on first run.

## Models

| File | Size | Purpose | Download Source |
|---|---|---|---|
| `midas_v21_small_256.onnx` | ~25 MB | Monocular depth estimation | MiDaS v2.1 (isl-org) |
| `yolov8n.onnx` | ~6 MB | Furniture object detection | Ultralytics YOLOv8-nano |

Both models are CPU-only and run without CUDA on Raspberry Pi 5 (ARM64).

## Performance on Raspberry Pi 5 (Cortex-A76 @ 2.4 GHz)

| Model | Input Size | Latency |
|---|---|---|
| MiDaS-Small ONNX | 256×256 | ~150–200 ms |
| YOLOv8n ONNX | 640×640 | ~80–150 ms |

## Licence

- MiDaS: MIT licence (see https://github.com/isl-org/MiDaS)
- YOLOv8: AGPL-3.0 licence (see https://github.com/ultralytics/ultralytics)
