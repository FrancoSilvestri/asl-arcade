"""HTTP service the Unity client talks to.

The game holds a target letter, streams webcam frames here, and asks a single
question per frame: is the player currently making this sign? The service answers
with a boolean plus the confidence behind it, and the game decides what to do.

Keeping the verdict on the server means the game never has to know anything about
the model, the class list, or the threshold.
"""
from __future__ import annotations

from flask import Flask, jsonify, request

from . import config
from .detector import FrameDecodeError, best_match, decode_frame, load_model

app = Flask(__name__)


@app.get("/health")
def health():
    """Liveness probe. Reports whether the weights are loadable."""
    try:
        load_model()
    except Exception as exc:  # torch and ultralytics raise a wide range of errors
        return jsonify({"status": "degraded", "detail": str(exc)}), 503
    return jsonify({"status": "ok", "threshold": config.CONF_THRESHOLD}), 200


def _score_request():
    """Validate the payload and score one frame.

    Returns either an error tuple to hand straight back to Flask, or the frame
    dimensions plus the best match. Both endpoints share this so the two response
    shapes never drift apart on anything that matters.
    """
    payload = request.get_json(silent=True) or {}
    encoded = payload.get("frame")
    target = payload.get("target")

    if not encoded or not target:
        return (jsonify({"error": "both 'frame' and 'target' are required"}), 400), None

    target = str(target).strip().upper()
    if target not in config.VALID_TARGETS:
        return (jsonify({"error": f"target must be a single letter A-Z, got {target!r}"}), 400), None

    try:
        frame = decode_frame(encoded)
    except FrameDecodeError as exc:
        return (jsonify({"error": str(exc)}), 400), None

    height, width, channels = frame.shape
    return None, (target, width, height, channels,
                  best_match(frame, target, config.CONF_THRESHOLD))


@app.post("/detect")
def detect_frame():
    error, scored = _score_request()
    if error:
        return error
    target, width, height, _channels, match = scored

    return jsonify({
        "target": target,
        "target_detected": match is not None,
        "confidence": round(match.confidence, 3) if match else 0.0,
        "box": list(match.box) if match else None,
        "width": width,
        "height": height,
    }), 200


@app.post("/")
def detect_frame_legacy():
    """The contract the shipped Unity build speaks.

    The Windows build published with this project was compiled in 2024 against an
    earlier version of this service: it posts to the root path and deserialises a
    response with `conf` rather than `confidence`. The client is already out in
    the world, so the server is what adapts.

    New clients should use /detect.
    """
    error, scored = _score_request()
    if error:
        return error
    _target, width, height, channels, match = scored

    return jsonify({
        "message": "Frame received",
        "width": width,
        "height": height,
        "channels": channels,
        "conf": round(match.confidence, 2) if match else 0,
        "target_detected": match is not None,
    }), 200


def main():
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
