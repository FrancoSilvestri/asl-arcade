# Vision service

Fine-tuned YOLOv8 detector for the 26 American Sign Language letters, served over
HTTP for the Unity client.

## Setup

```bash
cd vision-service
python -m venv .venv && .venv/Scripts/activate    # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
python scripts/download_weights.py                # 50 MB, from the GitHub release
python -m asl_vision.api
```

The service comes up on `http://0.0.0.0:8000`. `asl-vision` works as a console
command too, and `curl localhost:8000/health` tells you whether the weights loaded.

For GPU inference install a CUDA build of torch from
[pytorch.org](https://pytorch.org) before installing the requirements. The service
runs on CPU too, roughly an order of magnitude slower.

## Configuration

Everything is environment driven, with defaults that work out of the box.

| Variable | Default | Meaning |
|---|---|---|
| `ASL_WEIGHTS` | `weights/asl-yolov8m.pt` | path to the detector checkpoint |
| `ASL_HOST` | `0.0.0.0` | bind address |
| `ASL_PORT` | `8000` | bind port |
| `ASL_CONF_THRESHOLD` | `0.65` | confidence at which a sign is accepted |
| `ASL_DEVICE` | unset | `cuda`, `cpu`, or a device index |
| `ASL_DEBUG` | `0` | set to `1` for the Flask debugger |

## API

### `POST /detect`

```json
{ "frame": "<base64-encoded JPEG>", "target": "A" }
```

`target` is the letter the game asked the player to sign, case insensitive, a
single character A-Z.

```json
{
  "target": "A",
  "target_detected": true,
  "confidence": 0.812,
  "box": [412, 233, 618, 470],
  "width": 1280,
  "height": 720
}
```

`target_detected` is true only when the detector saw `target` at or above the
configured threshold. When nothing matched, `confidence` is `0.0` and `box` is
`null`, still with a 200: "no match" is a normal answer, not an error.

`400` is returned for a missing `frame` or `target`, a `target` that is not a
single letter, or bytes that do not decode to an image.

### `GET /health`

Returns `200` with the active threshold once the weights load, `503` with a
`detail` message if they do not. Use it to tell "the model is missing" apart from
"the service is down".

### `POST /` (legacy)

The Windows build published with this project was compiled in 2024 against an
earlier version of this service. It posts to the root path and deserialises a
response shaped like this:

```json
{
  "message": "Frame received",
  "width": 1280,
  "height": 720,
  "channels": 3,
  "conf": 0.81,
  "target_detected": true
}
```

Same request body and same validation rules as `/detect`; only the response
differs. `conf` is `0` rather than `null` when nothing matched, because that is
what the client expects.

That build is already distributed, so the server is what adapts. The two
endpoints share one code path and cannot disagree on a verdict. **New clients
should use `/detect`.**

## Layout

```
src/asl_vision/
  config.py     environment-driven settings
  detector.py   model loading, frame decoding, detection
  api.py        Flask routes
scripts/
  train.py            fine-tune a backbone
  webcam_preview.py   local preview window, no service involved
  client_demo.py      stand-in for the Unity client
  download_weights.py fetch the checkpoint from the release
tests/          API contract tests, detector stubbed out
docs/           training results and the demo clip
```

The model is loaded once per process and cached. Decoding, detection, and the
accept/reject decision are separate functions so the HTTP layer stays thin and the
contract can be tested without torch installed.

## Tests

```bash
pytest tests -v
```

The detector is stubbed, so the suite needs neither the weights nor a GPU. It
covers the contract the Unity client depends on: payload validation, the response
shape, case-insensitive targets, and the failure modes that must return 4xx rather
than crash the service.

## Dataset

Not vendored here. Download
[American Sign Language Letters v1](https://universe.roboflow.com/david-lee-d0rhs/american-sign-language-letters/dataset/1)
from Roboflow Universe in YOLOv8 format, copy `configs/data.yaml.example` to
`configs/data.yaml`, and point it at your copy.

## Training

```bash
python scripts/train.py --model yolov8m.pt --epochs 100 --patience 15
```

Results, curves, and an honest read of how well the model generalises are in
[docs/metrics.md](docs/metrics.md).
