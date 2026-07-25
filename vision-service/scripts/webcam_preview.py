"""Local preview: run the detector on the default webcam and draw the results.

This is the debugging tool, not the game path. It never touches the HTTP service.

    python scripts/webcam_preview.py            # press q to quit
    python scripts/webcam_preview.py --conf 0.3
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asl_vision.detector import detect  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.65)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            for d in detect(frame, min_confidence=args.conf):
                x1, y1, x2, y2 = d.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
                cv2.putText(frame, f"{d.label} {d.confidence * 100:.0f}%",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow("ASL detection preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
