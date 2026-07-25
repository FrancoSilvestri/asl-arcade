"""Fine-tune a YOLOv8 backbone on the ASL letters dataset.

Usage:
    python scripts/train.py --model yolov8m.pt --epochs 100 --patience 15

The dataset is not vendored in this repository. See configs/data.yaml.example
and the dataset section of the README for how to fetch it from Roboflow.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8m.pt",
                        help="pretrained backbone to fine-tune (yolov8s.pt, yolov8m.pt, ...)")
    parser.add_argument("--data", default="configs/data.yaml",
                        help="Ultralytics dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15,
                        help="early-stopping patience in epochs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="asl-yolov8m")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, training on {device}")

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(
            f"{data} not found. Copy configs/data.yaml.example to {data} and point "
            "its train/val/test keys at your local copy of the dataset."
        )

    model = YOLO(args.model)
    model.train(
        data=str(data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        name=args.name,
        device=device,
    )


if __name__ == "__main__":
    main()
