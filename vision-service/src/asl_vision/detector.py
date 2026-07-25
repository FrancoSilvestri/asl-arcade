"""Thin wrapper around the fine-tuned YOLOv8 detector.

The model is loaded once per process and reused for every frame.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = "weights/asl-yolov8m.pt"

model = YOLO(WEIGHTS_PATH)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


def decode_frame(encoded: str) -> np.ndarray:
    """Decode a base64-encoded image into a BGR array.

    The Unity client sends JPEG bytes wrapped in base64 inside a JSON body.
    """
    raw = base64.b64decode(encoded)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def detect(frame: np.ndarray, min_confidence: float = 0.0) -> list[Detection]:
    """Run the detector over one frame and return detections above a threshold."""
    detections: list[Detection] = []
    for result in model(frame, verbose=False):
        for box in result.boxes:
            confidence = float(box.conf)
            if confidence < min_confidence:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    label=model.names[int(box.cls)],
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )
    return detections


def best_match(frame: np.ndarray, target: str, threshold: float) -> Detection | None:
    """Return the highest-confidence detection of `target` in this frame, if the
    detector is confident enough that the player produced that sign."""
    candidates = [
        d for d in detect(frame, min_confidence=threshold)
        if d.label.upper() == target.upper()
    ]
    return max(candidates, key=lambda d: d.confidence, default=None)
