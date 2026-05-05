"""
Gradio web UI for YOLOv8 object detection.

Run:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import time
import cv2
import numpy as np
import torch
import gradio as gr
from ultralytics import YOLO

from utils.config import CONFIDENCE_THRESHOLD
from utils.draw import draw_detections

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Model cache ───────────────────────────────────────────────────────────────
_model_cache: dict[str, YOLO] = {}


def get_model(size: str) -> YOLO:
    """Return cached YOLO model for the given size key, loading on first call."""
    if size not in _model_cache:
        _model_cache[size] = YOLO(f"models/yolov8{size}.pt")
    return _model_cache[size]


get_model("n")  # pre-load nano at startup

# ── Constants ─────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "person",
    "bicycle", "car", "motorcycle", "bus", "truck",
    "dog", "cat", "bird", "horse", "cow", "sheep",
    "chair", "couch", "bed", "dining table",
    "bottle", "cup", "fork", "knife", "spoon", "bowl",
    "laptop", "mouse", "keyboard", "cell phone", "tv",
    "book", "clock", "vase", "backpack", "umbrella",
    "handbag", "suitcase",
]

MODEL_INFO = {
    "n": "Nano  —  6 MB  · fastest",
    "s": "Small — 22 MB  · balanced",
    "m": "Medium — 52 MB · accurate",
    "l": "Large — 87 MB  · most accurate",
}

# ── Inference ─────────────────────────────────────────────────────────────────
def _resolve_imgsz(imgsz: int | str, h: int, w: int) -> int:
    """Return the effective inference resolution. 'auto' picks based on image dimensions."""
    if imgsz == "auto":
        longest = max(h, w)
        if longest > 3000:
            return 1920
        if longest > 1500:
            return 1280
        if longest > 800:
            return 960
        return 640
    return int(imgsz)


def detect(
    image: np.ndarray,
    selected_classes: list,
    confidence: float,
    model_size: str,
    imgsz: int | str,
) -> tuple:
    """
    Run YOLOv8 inference on the uploaded image and return annotated results.

    Args:
        image:            RGB numpy array from Gradio image input.
        selected_classes: Class names to keep; empty list means all.
        confidence:       Minimum confidence threshold (0.1–1.0).
        model_size:       Model variant key — one of 'n', 's', 'm', 'l'.
        imgsz:            Inference resolution (320 / 640 / 1280 / 'auto').

    Returns:
        (annotated_rgb, table_rows, stats_markdown)
    """
    if image is None:
        return None, [], ""

    t0 = time.perf_counter()
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    effective_imgsz = _resolve_imgsz(imgsz, h, w)
    model = get_model(model_size)
    allowed = set(selected_classes) if selected_classes else None

    results = model.predict(
        source=frame_bgr,
        conf=confidence,
        iou=0.45,
        imgsz=effective_imgsz,
        device=DEVICE,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    annotated, counts = draw_detections(frame_bgr.copy(), results[0], model.names, allowed)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    total = sum(info["count"] for info in counts.values())
    table = [
        [name, info["count"], f'{info["avg_conf"]:.1%}']
        for name, info in sorted(counts.items())
    ]

    imgsz_label = f"{effective_imgsz}px (auto)" if imgsz == "auto" else f"{effective_imgsz}px"
    stats = (
        f"**{total} object(s) detected** &nbsp;·&nbsp; "
        f"Inference: **{elapsed_ms:.0f} ms** &nbsp;·&nbsp; "
        f"Resolution: **{w}×{h}** (infer @ {imgsz_label}) &nbsp;·&nbsp; "
        f"Model: **YOLOv8{model_size}** &nbsp;·&nbsp; "
        f"Device: **{DEVICE}**"
    )

    return annotated_rgb, table, stats


# ── UI ────────────────────────────────────────────────────────────────────────
DESCRIPTION = """
# YOLOv8 Object Detection

Upload an image to detect objects using a pre-trained YOLOv8 model.
Recognises **80 COCO classes** — people, vehicles, animals, and everyday objects.
"""

with gr.Blocks(title="YOLOv8 Object Detection — Group-9") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():

        # ── Left: Controls ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Input Image",
                type="numpy",
                sources=["upload"],
            )

            with gr.Accordion("Model Settings", open=True):
                model_radio = gr.Radio(
                    choices=list(MODEL_INFO.keys()),
                    value="n",
                    label="Model Size",
                    info="Larger = more accurate, slower first load",
                )
                gr.Markdown(
                    "\n".join(f"- **{k}:** {v}" for k, v in MODEL_INFO.items())
                )
                imgsz_radio = gr.Dropdown(
                    choices=["auto", 320, 480, 640, 800, 960, 1280, 1920],
                    value="auto",
                    label="Inference Resolution (px)",
                    info="Auto: 1280px for large images (>2000px), 640px otherwise. Must be multiple of 32.",
                )

            with gr.Accordion("Detection Settings", open=True):
                confidence_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=CONFIDENCE_THRESHOLD, step=0.05,
                    label="Confidence Threshold",
                    info="Discard detections below this score",
                )
                class_filter = gr.CheckboxGroup(
                    choices=COCO_CLASSES,
                    value=[],
                    label="Class Filter  (empty = all 80 classes)",
                )

            detect_btn = gr.Button("Detect", variant="primary", size="lg")

        # ── Right: Results ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Detection Result",
            )
            stats_md = gr.Markdown("")
            summary_table = gr.Dataframe(
                headers=["Class", "Count", "Avg Confidence"],
                datatype=["str", "number", "str"],
                label="Detection Summary",
                interactive=False,
                wrap=True,
            )

    detect_btn.click(
        fn=detect,
        inputs=[
            image_input, class_filter, confidence_slider,
            model_radio, imgsz_radio,
        ],
        outputs=[image_output, summary_table, stats_md],
    )

if __name__ == "__main__":
    demo.launch(
        css=".built-with { display: none !important; } footer { display: none !important; }",
        show_error=True,
    )
