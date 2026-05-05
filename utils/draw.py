"""Helper functions for drawing bounding boxes and labels on frames/images."""

import cv2
import numpy as np
from .config import FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS


def get_color_for_class(class_id: int) -> tuple:
    """
    Return a deterministic BGR color for a given class ID.

    Uses a fixed palette derived from the class index so the same class
    always gets the same color across images.

    Args:
        class_id: Integer class index from the YOLO model.

    Returns:
        A (B, G, R) tuple with values in [0, 255].
    """
    hue = int((class_id * 37) % 180)  # 37 chosen to spread colors evenly
    hsv_color = np.uint8([[[hue, 220, 220]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])


def draw_detections(
    frame: np.ndarray,
    results,
    class_names: list,
    allowed_classes: set | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Draw bounding boxes and labels for detections on the frame.

    Optionally filters to only the classes listed in `allowed_classes`.
    Returns both the annotated frame and per-class statistics.

    Args:
        frame:           The BGR image (H x W x 3 numpy array) to annotate.
        results:         A single Ultralytics Results object (results[0]).
        class_names:     List of class name strings from the YOLO model.
        allowed_classes: Set of class name strings to keep; None means all.

    Returns:
        (annotated_frame, counts) where counts is
        {class_name: {"count": int, "avg_conf": float}}.
    """
    raw: dict[str, list] = {}  # {name: [count, conf_sum]}

    if results.boxes is None or len(results.boxes) == 0:
        return frame, {}

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        if allowed_classes is not None and label_name not in allowed_classes:
            continue

        if label_name not in raw:
            raw[label_name] = [0, 0.0]
        raw[label_name][0] += 1
        raw[label_name][1] += confidence

        label = f"{label_name}: {confidence:.2f}"
        color = get_color_for_class(class_id)

        # Semi-transparent fill inside the bounding box
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)

        # Solid border with thin white outline for contrast
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), BOX_THICKNESS + 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label badge — above the box when space allows, inside otherwise
        pad = 6
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
        )
        badge_h = label_h + baseline + pad * 2
        badge_y1 = y1 - badge_h if y1 - badge_h >= 0 else y1
        badge_y2 = badge_y1 + badge_h
        badge_x2 = x1 + label_w + pad * 2

        # Badge shadow for readability
        cv2.rectangle(frame, (x1 + 2, badge_y1 + 2), (badge_x2 + 2, badge_y2 + 2),
                      (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (x1 + pad, badge_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    counts = {
        name: {"count": c, "avg_conf": s / c}
        for name, (c, s) in raw.items()
    }
    return frame, counts
