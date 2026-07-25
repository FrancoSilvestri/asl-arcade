"""Stand-in for the Unity client: stream webcam frames to the service and print
whether the requested sign was recognised.

Useful for exercising the service without the game running.

    python scripts/client_demo.py --target A
    python scripts/client_demo.py --target A --url http://192.168.1.20:8000
"""
import argparse
import base64
import sys

import cv2
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="letter the player is asked to sign")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality, 1-100")
    args = parser.parse_args()

    target = (args.target or input("target letter: ")).strip().upper()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    endpoint = args.url.rstrip("/") + "/detect"
    print(f"streaming to {endpoint}, target {target}. ctrl-c to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("dropped frame from the camera", file=sys.stderr)
                break

            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
            )
            if not ok:
                continue

            response = requests.post(endpoint, json={
                "frame": base64.b64encode(buffer.tobytes()).decode("utf-8"),
                "target": target,
            }, timeout=5)

            if response.status_code != 200:
                print(f"error {response.status_code}: {response.text}", file=sys.stderr)
                continue

            body = response.json()
            if body["target_detected"]:
                print(f"detected {target} at {body['confidence']:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    main()
