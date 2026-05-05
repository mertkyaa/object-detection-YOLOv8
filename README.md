# Static Image Object Detection Using YOLO
Computer Vision Project  
Mert Kaya · Sıla Deniz Esen · Hayrunnisa Hamidi

---

## Overview

This project implements an object detection application using a **pre-trained YOLOv8 model**.  
It detects and labels 80 common object categories (COCO dataset) from static images via CLI or a Gradio web UI.

---

## Project Structure

```
yolo-object-detection/
├── main.py             # CLI entry point
├── detect_image.py     # Static image detection
├── app.py              # Gradio web UI
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── config.py       # Thresholds & paths
│   └── draw.py         # Bounding-box drawing helpers
├── models/             # YOLOv8 weights (auto-downloaded)
├── input/              # Put test images here
└── output/             # Annotated results saved here
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Running

### CLI

```bash
# Basic detection (display result window)
python main.py --source input/sample.jpg

# Save result to output/
python main.py --source input/sample.jpg --save

# Headless (no window, just save)
python main.py --source input/sample.jpg --save --no-show

# Choose model size (n/s/m/l) and set confidence
python main.py --source input/sample.jpg --model s --conf 0.30 --save --no-show

# Set inference resolution (auto-selected if omitted)
python main.py --source input/sample.jpg --imgsz 1280 --save --no-show

# Filter specific classes
python main.py --source input/sample.jpg --classes person car --save --no-show
```

### Gradio Web UI

```bash
python app.py
# Open http://localhost:7860 in your browser
```

---

## Configuration

Edit `utils/config.py` to adjust defaults:

| Parameter              | Default             | Description                          |
|------------------------|---------------------|--------------------------------------|
| `CONFIDENCE_THRESHOLD` | `0.35`              | Minimum confidence to show a box     |
| `MODEL_PATH`           | `models/yolov8n.pt` | Default model weights path           |
| `OUTPUT_DIR`           | `output/`           | Where result images are saved        |

---

## Detection Pipeline

```
Input (image file / Gradio upload)
        ↓
  Preprocessing (resize, normalize)   ← handled by Ultralytics internally
        ↓
  YOLOv8 Inference (model.predict)
        ↓
  Post-processing (NMS + conf filter)
        ↓
  Draw Bounding Boxes + Labels
        ↓
  Output (display window / saved image / Gradio web UI)
```

---

## Test Image Sources

| File | Source | License |
|------|--------|---------|
| `bus.jpg` / `sample.jpg` | [Ultralytics](https://ultralytics.com/images/bus.jpg) | Ultralytics official test image |
| `zidane.jpg` | [Ultralytics](https://ultralytics.com/images/zidane.jpg) | Ultralytics official test image |
| `nyc_street.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:On_The_Streets_of_NYC_(4511061466).jpg) — FaceMePLS, 2010 | CC BY 2.0 |
| `nyc_taxi.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:2010_11_NYC_TAXI,_PIGEONS_5472.jpg) — Paul Harrison, 2010 | CC BY-SA 4.0 |
| `dog_walk.jpg` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Walking_the_dog_Redhead_in_Boots_(49763148076).jpg) — Billie Grace Ward, 2020 | CC BY 2.0 |

---

## Dependencies

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy
- Pillow
- Gradio

