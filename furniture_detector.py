"""Indoor furniture detector using YOLOv8-nano (ONNX, CPU-only).

Inspired by the object detection stage of DeepPanoContext (ICCV 2021)
but replaced with a lightweight YOLOv8-nano model that runs on
Raspberry Pi 5 CPU without CUDA.

Performance target (Raspberry Pi 5): ≤ 150 ms per image.

Output per image::

    [
      {
        "label":      "chair",
        "confidence": 0.82,
        "bbox_2d":    [x1, y1, x2, y2],   # pixel coords in original image
        "class_id":   56,
      },
      ...
    ]
"""

from __future__ import annotations

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

import config

LOGGER = logging.getLogger("furniture_detector")

# YOLOv8-nano ONNX (~6 MB) — official Ultralytics release
_YOLO_ONNX_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
)

# Full COCO class list (80 classes, 0-indexed)
_COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    """One detected object from the furniture detector."""
    label: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox_2d(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def center_2d(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> Dict:
        return {
            "label":      self.label,
            "class_id":   self.class_id,
            "confidence": round(float(self.confidence), 3),
            "bbox_2d":    self.bbox_2d,
        }


def _download_model(model_path: Path, url: str) -> bool:
    """Download ONNX model with progress logging. Returns True on success."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading YOLOv8n ONNX from %s ...", url)

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0 and block_num % 50 == 0:
            pct = min(100.0, downloaded / total_size * 100.0)
            LOGGER.info("  %.1f %%  (%d / %d bytes)", pct, downloaded, total_size)

    try:
        urllib.request.urlretrieve(url, str(model_path), reporthook=_progress)
        LOGGER.info("Download complete: %s", model_path)
        return True
    except Exception as exc:
        LOGGER.warning("YOLOv8n download failed: %s", exc)
        if model_path.exists():
            model_path.unlink(missing_ok=True)
        return False


class FurnitureDetector:
    """Detect indoor furniture objects using YOLOv8-nano ONNX.

    Only objects whose COCO class ID is in ``config.YOLO_FURNITURE_CLASSES``
    are returned; all other detections are silently discarded.

    Args:
        model_path:   Path to yolov8n.onnx. Downloaded automatically if missing.
        confidence:   Minimum detection confidence to keep.
        nms_threshold: IoU threshold for non-maximum suppression.
        input_size:   Square input resolution fed to YOLOv8 (default 640).
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL_PATH,
        confidence: float = config.YOLO_CONFIDENCE_THRESHOLD,
        nms_threshold: float = config.YOLO_NMS_THRESHOLD,
        input_size: int = config.YOLO_INPUT_SIZE,
    ) -> None:
        self.confidence = float(confidence)
        self.nms_threshold = float(nms_threshold)
        self.input_size = int(input_size)
        self._session = None  # lazy-loaded
        self._use_fallback = False  # True if model unavailable

        model_file = Path(model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).resolve().parent / model_file
        if not model_file.exists():
            LOGGER.warning("YOLO model not found at %s — attempting download...", model_file)
            ok = config.AUTO_DOWNLOAD_MODELS and _download_model(model_file, _YOLO_ONNX_URL)
            if not ok or not model_file.exists():
                LOGGER.warning(
                    "YOLOv8n model unavailable. Furniture detection disabled. "
                    "Reconstruction will continue with 0 furniture items detected."
                )
                self._use_fallback = True

        self._model_path = str(model_file)
        self._furniture_ids = set(config.YOLO_FURNITURE_CLASSES.keys())
        LOGGER.info(
            "FurnitureDetector initialised (model=%s, fallback=%s, classes=%d)",
            model_file.name,
            self._use_fallback,
            len(self._furniture_ids),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_image: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR image.

        Args:
            bgr_image: H×W×3 uint8 BGR image.

        Returns:
            List of Detection objects filtered to furniture classes only.
            Returns empty list if YOLO model is unavailable (fallback mode).
        """
        if self._use_fallback:
            return []  # No model — silently return no detections

        try:
            session = self._get_session()
        except Exception as exc:
            LOGGER.warning("YOLO session failed (%s) — furniture detection disabled", exc)
            self._use_fallback = True
            return []

        input_name  = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        orig_h, orig_w = bgr_image.shape[:2]
        blob, scale_x, scale_y, pad_x, pad_y = self._preprocess(bgr_image)

        t0 = time.perf_counter()
        try:
            raw = session.run([output_name], {input_name: blob})[0]
            if not np.isfinite(raw).all():
                raise ValueError("Detector returned non-finite values")
        except Exception as exc:
            LOGGER.warning("Furniture inference unavailable: %s", exc)
            self._use_fallback = True
            return []
        elapsed = (time.perf_counter() - t0) * 1000.0
        LOGGER.debug("YOLOv8n inference: %.1f ms", elapsed)

        detections = self._postprocess(raw, scale_x, scale_y, pad_x, pad_y, orig_w, orig_h)
        LOGGER.debug("Detected %d furniture objects.", len(detections))
        return detections

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Detect furniture in each image sequentially."""
        return [self.detect(img) for img in images]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_session(self):
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for furniture detection. "
                    "Install with:  pip install onnxruntime"
                ) from exc
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            LOGGER.info("ONNX Runtime session loaded for YOLOv8n.")
        return self._session

    def _preprocess(
        self, bgr_image: np.ndarray
    ):
        """Letterbox-resize + normalise image to YOLOv8 input format.

        Returns (blob, scale_x, scale_y, pad_x, pad_y).
        """
        orig_h, orig_w = bgr_image.shape[:2]
        target = self.input_size

        # Compute scale that preserves aspect ratio
        scale = min(target / orig_w, target / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target×target
        pad_x = (target - new_w) // 2
        pad_y = (target - new_h) // 2
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR → RGB, HWC → CHW → NCHW, normalise
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        scale_x = scale
        scale_y = scale
        return blob, scale_x, scale_y, pad_x, pad_y

    def _postprocess(
        self,
        raw: np.ndarray,
        scale_x: float,
        scale_y: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        """Parse YOLOv8 output tensor → list of Detection objects.

        YOLOv8 ONNX output shape: (1, 84, N) where 84 = 4 box + 80 class scores.
        """
        # (1, 84, N) → (N, 84)
        output = raw[0].transpose(1, 0)  # (N, 84)

        boxes_xywh  = output[:, :4]
        class_scores = output[:, 4:]     # (N, 80)

        class_ids   = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Filter by confidence and furniture class
        mask = (confidences >= self.confidence) & np.isin(class_ids, list(self._furniture_ids))
        boxes_xywh  = boxes_xywh[mask]
        class_ids   = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert CX,CY,W,H → X1,Y1,X2,Y2 in original image coordinates
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]

        x1 = ((cx - bw / 2) - pad_x) / scale_x
        y1 = ((cy - bh / 2) - pad_y) / scale_y
        x2 = ((cx + bw / 2) - pad_x) / scale_x
        y2 = ((cy + bh / 2) - pad_y) / scale_y

        x1 = np.clip(x1, 0, orig_w).astype(int)
        y1 = np.clip(y1, 0, orig_h).astype(int)
        x2 = np.clip(x2, 0, orig_w).astype(int)
        y2 = np.clip(y2, 0, orig_h).astype(int)

        # NMS
        nms_boxes  = [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])] for i in range(len(x1))]
        nms_scores = confidences.tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.confidence, self.nms_threshold)

        results: List[Detection] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid   = int(class_ids[idx])
                label = config.YOLO_FURNITURE_CLASSES.get(cid, _COCO_CLASSES[cid] if cid < len(_COCO_CLASSES) else "object")
                results.append(Detection(
                    label=label,
                    class_id=cid,
                    confidence=float(confidences[idx]),
                    x1=int(x1[idx]),
                    y1=int(y1[idx]),
                    x2=int(x2[idx]),
                    y2=int(y2[idx]),
                ))

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="FurnitureDetector self-test")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--output", type=str, default="furniture_detect.jpg", help="Output annotated image")
    args = parser.parse_args()

    detector = FurnitureDetector()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            LOGGER.error("Could not read: %s", args.image)
            return 1
    else:
        LOGGER.info("No image supplied — generating synthetic test frame.")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (50, 60, 70)

    t0 = time.perf_counter()
    detections = detector.detect(img)
    elapsed = (time.perf_counter() - t0) * 1000.0

    LOGGER.info("Found %d furniture object(s) in %.1f ms:", len(detections), elapsed)
    for det in detections:
        LOGGER.info("  %s (%.0f%%) @ %s", det.label, det.confidence * 100, det.bbox_2d)
        cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 100), 2)
        cv2.putText(img, f"{det.label} {det.confidence:.2f}", (det.x1, det.y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

    cv2.imwrite(args.output, img)
    LOGGER.info("Annotated image saved: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
