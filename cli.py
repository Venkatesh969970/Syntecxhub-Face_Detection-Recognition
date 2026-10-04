"""Command-line interface for local face registration and recognition."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

from faceid.registry import FaceRegistry
from faceid.vision import FaceEngine, annotate_frame


def _camera_index(source: str) -> int | None:
    try:
        return int(source)
    except ValueError:
        return None


def _largest_face(detections):
    if not detections:
        return None
    return max(detections, key=lambda item: item[0][2] * item[0][3])


def _register_images(args, engine: FaceEngine, registry: FaceRegistry) -> int:
    registered = 0
    for image_path in args.images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        detections = engine.detect(image)
        face = _largest_face(detections)
        if face is None:
            print(f"No face found in {image_path}; skipped.")
            continue
        if len(detections) > 1:
            print(f"Multiple faces in {image_path}; registering the largest one.")
        registry.register(args.name, face[1])
        registered += 1
    if registered == 0:
        raise ValueError("No face samples were registered")
    print(f"Registered {registered} face sample(s) for {args.name}.")
    return 0


def _register_camera(args, engine: FaceEngine, registry: FaceRegistry) -> int:
    camera_index = _camera_index(args.camera)
    if camera_index is None:
        raise ValueError("Camera must be an integer device index, such as 0")
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    samples = []
    try:
        while len(samples) < args.samples:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the camera")
            detections = engine.detect(frame)
            preview = frame.copy()
            for (x, y, width, height), _ in detections:
                cv2.rectangle(preview, (x, y), (x + width, y + height), (70, 190, 90), 2)
            cv2.putText(
                preview,
                f"SPACE capture | Q finish ({len(samples)}/{args.samples})",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Register face", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                face = _largest_face(detections)
                if face is None:
                    print("No face detected; try again.")
                else:
                    samples.append(face[1])
                    print(f"Captured sample {len(samples)}/{args.samples}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not samples:
        raise ValueError("No face samples were captured")
    for embedding in samples:
        registry.register(args.name, embedding)
    print(f"Registered {len(samples)} face sample(s) for {args.name}.")
    return 0


def _recognize_image(args, image, engine: FaceEngine, registry: FaceRegistry) -> int:
    detections = engine.detect(image)
    annotated = annotate_frame(image, detections, registry, args.threshold)
    print(f"Detected {len(detections)} face(s).")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not write output image: {output_path}")
    if not args.no_display:
        cv2.imshow("Face recognition", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def _recognize_stream(args, engine: FaceEngine, registry: FaceRegistry) -> int:
    camera_index = _camera_index(args.source)
    capture = cv2.VideoCapture(camera_index if camera_index is not None else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    try:
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                fps = 20.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

        while True:
            success, frame = capture.read()
            if not success:
                break
            annotated = annotate_frame(frame, engine.detect(frame), registry, args.threshold)
            if writer is not None:
                writer.write(annotated)
            if not args.no_display:
                cv2.imshow("Face recognition", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


def _recognize(args, engine: FaceEngine, registry: FaceRegistry) -> int:
    if _camera_index(args.source) is None:
        image = cv2.imread(args.source)
        if image is not None:
            return _recognize_image(args, image, engine, registry)
    return _recognize_stream(args, engine, registry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and recognize faces locally with OpenCV.")
    parser.add_argument("--registry", default="data/known_faces.json", help="Local embeddings JSON file")
    parser.add_argument("--models-dir", default="models", help="Folder for downloaded ONNX models")
    commands = parser.add_subparsers(dest="command", required=True)

    recognize = commands.add_parser("recognize", help="Process an image, video, or camera")
    recognize.add_argument("--source", required=True, help="Image/video path or camera index (0)")
    recognize.add_argument("--output", help="Save the annotated image or video")
    recognize.add_argument("--threshold", type=float, default=0.363, help="SFace cosine threshold")
    recognize.add_argument("--no-display", action="store_true", help="Do not open a preview window")

    add_images = commands.add_parser("register-images", help="Register one or more face images")
    add_images.add_argument("--name", required=True, help="Name to associate with these face samples")
    add_images.add_argument("images", nargs="+", help="Image paths (use several for better coverage)")

    add_camera = commands.add_parser("register-camera", help="Capture face samples from a webcam")
    add_camera.add_argument("--name", required=True, help="Name to register")
    add_camera.add_argument("--camera", default="0", help="Camera device index (default: 0)")
    add_camera.add_argument("--samples", type=int, default=5, help="Number of samples to capture")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "register-camera" and args.samples < 1:
        parser.error("--samples must be at least 1")
    try:
        registry = FaceRegistry(args.registry)
        engine = FaceEngine(args.models_dir)
        if args.command == "recognize":
            return _recognize(args, engine, registry)
        if args.command == "register-images":
            return _register_images(args, engine, registry)
        return _register_camera(args, engine, registry)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())