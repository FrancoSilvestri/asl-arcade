"""Runtime configuration, read from the environment with sane defaults.

Everything that was hardcoded during the hackathon lives here now: the weights
path, the bind address, and the confidence threshold used to accept a sign.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Path to the trained detector. See scripts/download_weights.py to fetch it.
WEIGHTS_PATH = Path(os.getenv("ASL_WEIGHTS", REPO_ROOT / "weights" / "asl-yolov8m.pt"))

# Bind address for the inference service. During the hackathon this was the
# laptop's address on the venue LAN so the Unity client could reach it.
HOST = os.getenv("ASL_HOST", "0.0.0.0")
PORT = int(os.getenv("ASL_PORT", "8000"))
DEBUG = os.getenv("ASL_DEBUG", "0") == "1"

# A detection at or above this confidence counts as the player having produced
# the requested sign. 0.65 is what the game ran with.
CONF_THRESHOLD = float(os.getenv("ASL_CONF_THRESHOLD", "0.65"))

# Device passed to Ultralytics: "cuda", "cpu", or a device index.
DEVICE = os.getenv("ASL_DEVICE", "")
