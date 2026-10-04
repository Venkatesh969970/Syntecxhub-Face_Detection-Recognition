"""OpenCV YuNet detection and SFace embedding extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2

from faceid.models import ensure_models
from faceid.registry import FaceRegistry


class FaceEngine:
    def __init__(self, models_dir: str | Path = "models") -> None:
        detector_model, recognizer_model = ensure_models(models_dir)
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "This OpenCV build does not include YuNet/SFace. Install opencv-contrib-python."
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(detector_model), "", (320, 320), 0.8, 0.3, 5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")

    def detect(self, frame) -> list[tuple[tuple[int, int, int, int], list[float]]]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []

        detected = []
        for face in faces:
            aligned_face = self.recognizer.alignCrop(frame, face)
            embedding = self.recognizer.feature(aligned_face).reshape(-1).tolist()
            x, y, face_width, face_height = (int(round(value)) for value in face[:4])
            detected.append(((x, y, face_width, face_height), embedding))
        return detected


def annotate_frame(
    frame,
    detections: Sequence[tuple[tuple[int, int, int, int], list[float]]],
    registry: FaceRegistry,
    threshold: float = 0.363,
):
    annotated = frame.copy()
    for (x, y, width, height), embedding in detections:
        match = registry.identify(embedding, threshold)
        if match is None:
            label, color = "Unknown", (40, 90, 235)
        else:
            label, score = match
            label = f"{label} {score:.2f}"
            color = (70, 190, 90)

        top = max(0, y)
        left = max(0, x)
        cv2.rectangle(annotated, (left, top), (left + width, top + height), color, 2)
        text_y = max(20, top - 8)
        cv2.putText(
            annotated,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated