"""
Entry point for the YOLOv8 Object Detection application.

Detects objects in a static image file using a pre-trained YOLOv8 model.

Usage examples:
  python main.py --source input/sample.jpg
  python main.py --source input/sample.jpg --save
  python main.py --source input/sample.jpg --no-show
  python main.py --source input/sample.jpg --classes person car
  python main.py --source input/sample.jpg --model s --conf 0.30 --imgsz 1280
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the detection application.

    Returns:
        Parsed argument namespace with attributes:
          - source  (str)  : Path to input image
          - save    (bool) : Whether to save the annotated output image
          - show    (bool) : Whether to display the result window
          - classes (list) : Class names to keep (None = all)
    """
    parser = argparse.ArgumentParser(
        description="Object Detection using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input image file (e.g. input/sample.jpg).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save the annotated output image to the output/ directory.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        default=True,
        help="Do not display the result window (useful in headless environments).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLASS",
        help="Only detect these class names (e.g. --classes person car dog).",
    )
    parser.add_argument(
        "--model",
        choices=["n", "s", "m", "l"],
        default="n",
        help="YOLOv8 model size: n=nano, s=small, m=medium, l=large (default: n).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Confidence threshold in [0, 1] (default: value from config.py).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        choices=[320, 640, 1280],
        default=None,
        metavar="SIZE",
        help="Inference resolution in pixels (default: auto — 1280 for large images, 640 otherwise).",
    )

    return parser.parse_args()


def main() -> None:
    """Run YOLOv8 inference on a static image."""
    args = parse_args()

    from detect_image import detect_image
    detect_image(
        source=args.source,
        save=args.save,
        show=args.show,
        classes=args.classes,
        model_size=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
