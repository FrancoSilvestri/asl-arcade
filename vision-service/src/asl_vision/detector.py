"""Thin wrapper around the fine-tuned YOLOv8 detector.

The model is loaded once per process and reused for every frame. Loading it per
call, or per stream, costs about a second and a full copy of the weights in
memory, which is what the hackathon version did.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


class FrameDecodeError(ValueError):
    """Raised when the payload is not a decodable JPEG/PNG frame."""


@lru_cache(maxsize=1)
def load_model(weights: Path | None = None):
    """Load and cache the detector. Import is deferred so that the module can be
    imported (and the API unit-tested) without pulling in torch."""
    from ultralytics import YOLO

    path = Path(weights or config.WEIGHTS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. "
            "Run scripts/download_weights.py, or point ASL_WEIGHTS at a .pt file."
        )
    model = YOLO(str(path))
    if config.DEVICE:
        model.to(config.DEVICE)
    return model


def decode_frame(encoded: str) -> np.ndarray:
    """Decode a base64-encoded image into a BGR array.

    The Unity client sends JPEG bytes wrapped in base64 inside a JSON body.
    """
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FrameDecodeError("frame is not valid base64") from exc

    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise FrameDecodeError("frame bytes could not be decoded as an image")
    return frame


def detect(frame: np.ndarray, min_confidence: float = 0.0) -> list[Detection]:
    """Run the detector over one frame and return detections above a threshold."""
    model = load_model()
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
