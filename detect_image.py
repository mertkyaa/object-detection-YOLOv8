"""Static image object detection using a pre-trained YOLOv8 model."""

import os
import cv2
from ultralytics import YOLO

from utils.config import CONFIDENCE_THRESHOLD, OUTPUT_DIR
from utils.draw import draw_detections

# Images wider or taller than this threshold get imgsz=1280 automatically.
_AUTO_IMGSZ_THRESHOLD = 2000


def _resolve_model_path(model_size: str) -> str:
    """Return the .pt file path for the given model size letter (n/s/m/l)."""
    return os.path.join("models", f"yolov8{model_size}.pt")


def _auto_imgsz(frame) -> int:
    """Return 1280 for large images, 640 otherwise."""
    h, w = frame.shape[:2]
    return 1280 if max(h, w) > _AUTO_IMGSZ_THRESHOLD else 640


def detect_image(
    source: str,
    save: bool = False,
    show: bool = True,
    classes: list[str] | None = None,
    model_size: str = "n",
    conf: float | None = None,
    imgsz: int | None = None,
) -> str | None:
    """
    Run YOLOv8 inference on a single static image.

    The pipeline:
      1. Load the YOLOv8 model (auto-downloads weights on first run).
      2. Read the source image with OpenCV.
      3. Auto-select inference resolution if imgsz is not given.
      4. Run model inference and filter by confidence threshold.
      5. Draw bounding boxes and labels with custom helpers.
      6. Print per-class object counts to stdout.
      7. Optionally display the result window and/or save to disk.

    Args:
        source:     File path to the input image (e.g. "input/sample.jpg").
        save:       If True, write the annotated image to OUTPUT_DIR.
        show:       If True, display the result in an OpenCV window.
        classes:    List of class names to keep (e.g. ["person", "car"]).
                    None means all detected classes are shown.
        model_size: YOLOv8 variant — "n", "s", "m", or "l" (default "n").
        conf:       Confidence threshold override. Falls back to
                    CONFIDENCE_THRESHOLD from config if None.
        imgsz:      Inference resolution (320 / 640 / 1280). Auto-selected
                    based on input image size when None.

    Returns:
        The path to the saved output image, or None if not saved.

    Raises:
        FileNotFoundError: If the source image does not exist.
        ValueError:        If the source file cannot be read by OpenCV.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source image not found: {source}")

    model_path = _resolve_model_path(model_size)
    model = YOLO(model_path)

    frame = cv2.imread(source)
    if frame is None:
        raise ValueError(f"OpenCV could not read image: {source}")

    effective_conf = conf if conf is not None else CONFIDENCE_THRESHOLD
    effective_imgsz = imgsz if imgsz is not None else _auto_imgsz(frame)

    print(f"[detect_image] model=yolov8{model_size}, conf={effective_conf}, imgsz={effective_imgsz}")

    results = model.predict(source=frame, conf=effective_conf, imgsz=effective_imgsz, verbose=False)
    result = results[0]

    allowed = set(classes) if classes else None
    annotated, counts = draw_detections(frame.copy(), result, model.names, allowed)

    total = sum(info["count"] for info in counts.values())
    if counts:
        summary = ", ".join(
            f"{name}: {info['count']} (avg {info['avg_conf']:.0%})"
            for name, info in sorted(counts.items())
        )
        print(f"[detect_image] Detected {total} object(s) in '{source}' — {summary}.")
    else:
        print(f"[detect_image] No objects detected in '{source}'.")

    output_path = None
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(source))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{basename}_result.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"[detect_image] Result saved to '{output_path}'.")

    if show:
        cv2.imshow("Detection Result", annotated)
        print("[detect_image] Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path
