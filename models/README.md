# EdgeRF ONNX models

Both model binaries are included in this project. They run with ONNX Runtime's CPU provider; PyTorch, Ultralytics, CUDA and Node.js are not runtime requirements.

| Local file | Downloaded bytes | Input / output | Source |
| --- | ---: | --- | --- |
| `midas_v21_small_256.onnx` | 66,764,249 (~63.7 MiB) | FP32 `1×3×256×256` → `1×256×256` inverse depth | Official [MiDaS v2.1 `model-small.onnx`](https://github.com/isl-org/MiDaS/releases/tag/v2_1), renamed locally |
| `yolov8n.onnx` | 12,845,642 (~12.3 MiB) | Dynamic FP32 NCHW; app uses `1×3×640×640` → `1×84×8400` raw detections | [Universidad de León robotics ONNX export](https://huggingface.co/unileon-robotics/YOLO8-ONNX/tree/66e5888954d3454da709847e77b36df31f756aa9/dynamic) of [Ultralytics YOLOv8n weights](https://huggingface.co/Ultralytics/YOLOv8) |

`manifest.json` records exact source URLs, the YOLO exporter revision, sizes, SHA-256 hashes, tensor contracts and licenses. The ONNX bytes are unmodified downloads. MiDaS's filename is an application alias, not a different export.

## Install and validate on the Pi

If the Pi was cloned before these files were added, copy the assets and updated wrappers from Windows PowerShell in the local repository root:

```powershell
scp -r .\models pi@pi.local:~/Edge_RF_vision_RP-5/
scp .\depth_estimator.py .\furniture_detector.py pi@pi.local:~/Edge_RF_vision_RP-5/
scp .\tools\validate_models.py .\tools\fetch_model_assets.py pi@pi.local:~/Edge_RF_vision_RP-5/tools/
```

The corrected depth wrapper is essential: the original wrapper normalized input a second time. These local additions are not automatically present on a previously cloned Pi or published to GitHub.

From the repository root, with its Linux virtual environment active:

```bash
python -m pip install -r requirements-hardware.txt
python tools/validate_models.py
python main.py --host 0.0.0.0 --interface wlan0 --resolution 640x480
```

To restore missing model files or detect a corrupted copy:

```bash
python tools/fetch_model_assets.py
```

The downloader verifies existing/downloaded binaries against the manifest and publishes downloads only after checksum verification. It will not silently replace a mismatched existing file; use `--force` to explicitly redownload. Normal server startup does not download files.

For a real-image smoke test:

```bash
python tools/fetch_model_assets.py --sample
python tools/validate_models.py --image .dist/engineering/coco-sample.png
```

The optional sample goes into ignored `.dist/engineering/`, not this folder. Validation needs NumPy, OpenCV and ONNX Runtime from the hardware requirements. If the optional `onnx` development package is installed, it also checks the ONNX graph. The command returns an error if checksums, inference or depth output checks fail, or either wrapper uses fallback mode.

## Preprocessing and measurement limits

- The bundled MiDaS graph embeds ImageNet normalization. Feed RGB floats in `[0,1]`; do not normalize again outside the graph. The Python wrapper has been corrected accordingly.
- The bundled YOLO model accepts dynamic image dimensions; the application keeps its existing 640×640 letterbox preprocessing and raw-output decoder/NMS. It uses the full 80-class COCO model and filters results to the indoor classes in `config.py`.
- MiDaS output is relative inverse depth. Application scaling does not make it a calibrated distance sensor.
- CPU inference was verified locally on Windows x64. These files are portable ONNX graphs, but Raspberry Pi 5 speed and field accuracy have not been measured. Earlier project timing estimates are not benchmarks.

Validation on 2026-11-10: both graphs passed the ONNX checker and the actual CPU wrappers with synthetic and real-image inputs. YOLO returned a `couch` detection on the COCO sample; MiDaS returned finite image-sized relative depth. The project regression suite passed all 38 tests. Detailed local reports are in ignored `.dist/engineering/model-validation-*.json`.

## Third-party licenses and corresponding sources

- MiDaS: [MIT](LICENSE.MiDaS.txt), Copyright © 2019 Intel ISL. Source and model release: <https://github.com/isl-org/MiDaS/tree/v2_1>.
- YOLOv8n: [AGPL-3.0](LICENSE.YOLO.txt). Original weights/source: <https://huggingface.co/Ultralytics/YOLOv8> and <https://github.com/ultralytics/ultralytics>. The ONNX exporter publisher, pinned files, and export recipe are linked in `manifest.json`. Preserve these license and source notices when redistributing the model files.
