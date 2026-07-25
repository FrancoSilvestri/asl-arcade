"""Fetch the trained detector from the GitHub release.

The weights are 50 MB, so they live on a release rather than in git history.

    python scripts/download_weights.py
"""
import argparse
import sys
import urllib.request
from pathlib import Path

RELEASE_URL = (
    "https://github.com/FrancoSilvestri/asl-arcade/releases/download/v1.0.0/asl-yolov8m.pt"
)
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "weights" / "asl-yolov8m.pt"


def report(count, block_size, total):
    if total <= 0:
        return
    done = min(count * block_size, total)
    sys.stdout.write(f"\r{done / 1e6:6.1f} / {total / 1e6:.1f} MB")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=RELEASE_URL)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if args.dest.exists():
        print(f"{args.dest} already exists, nothing to do")
        return

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.url}")
    urllib.request.urlretrieve(args.url, args.dest, reporthook=report)
    print(f"\nsaved to {args.dest}")


if __name__ == "__main__":
    main()
