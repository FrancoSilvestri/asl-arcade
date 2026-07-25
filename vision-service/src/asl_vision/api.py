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
from .detector import best_match, decode_frame

app = Flask(__name__)


@app.post("/detect")
def detect_frame():
    payload = request.get_json(silent=True) or {}
    encoded = payload.get("frame")
    target = payload.get("target")

    if not encoded or not target:
        return jsonify({"error": "both 'frame' and 'target' are required"}), 400

    target = str(target).strip().upper()
    frame = decode_frame(encoded)
    height, width = frame.shape[:2]
    match = best_match(frame, target, config.CONF_THRESHOLD)

    return jsonify({
        "target": target,
        "target_detected": match is not None,
        "confidence": round(match.confidence, 3) if match else 0.0,
        "box": list(match.box) if match else None,
        "width": width,
        "height": height,
    }), 200


def main():
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)


if __name__ == "__main__":
    main()
