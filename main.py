"""Command-line entry point for image and camera plant disease detection."""

import argparse
from pathlib import Path

from display import display_prediction
from model_loader import ModelLoader
from preprocessing import preprocess_image


def _capture_camera(camera_index: int):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Camera mode requires OpenCV. Install the opencv-python package."
        ) from error

    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        success, frame = camera.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not capture a frame from camera {camera_index}")
        return frame
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect plant diseases from an image.")
    parser.add_argument("image", nargs="?", help="Path to an image file")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Capture one frame from the camera",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("plant_disease.tflite"),
        help="Path to plant_disease.tflite",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if args.camera == bool(args.image):
        parser.error("provide an image path or use --camera")

    model = ModelLoader(args.model, confidence_threshold=args.threshold)
    if args.camera:
        image = _capture_camera(args.camera_index)
        prepared = preprocess_image(image, color_order="BGR")
    else:
        prepared = preprocess_image(args.image)
    display_prediction(model.predict(prepared))


if __name__ == "__main__":
    main()
