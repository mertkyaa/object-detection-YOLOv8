"""Configuration constants for the YOLO object detection application."""

# Minimum confidence score to accept a detection
CONFIDENCE_THRESHOLD = 0.35

# Path to the pre-trained YOLOv8 nano model
# The model is auto-downloaded by Ultralytics on first use
MODEL_PATH = "models/yolov8n.pt"

# Directory where output images are saved
OUTPUT_DIR = "output/"

# Label font scale and thickness for bounding box annotations
FONT_SCALE = 0.45
FONT_THICKNESS = 1
BOX_THICKNESS = 1
