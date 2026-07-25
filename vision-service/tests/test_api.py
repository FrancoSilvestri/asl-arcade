"""Contract tests for the detection endpoint.

The detector itself is stubbed out. These tests are about the contract the Unity
client depends on: which payloads are accepted, what shape the answer has, and
which failures come back as 4xx instead of a 500.
"""
import base64

import cv2
import numpy as np
import pytest

from asl_vision.detector import Detection, FrameDecodeError


def encoded_frame(width=64, height=48):
    frame = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


@pytest.fixture(autouse=True)
def stub_detector(api, monkeypatch):
    """No weights, no torch: every test decides what the detector 'saw'."""
    monkeypatch.setattr(api, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(api, "best_match", lambda *a, **k: None)


def test_health_reports_ok_and_threshold(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert 0 < body["threshold"] <= 1


def test_health_degrades_when_weights_are_missing(api, client, monkeypatch):
    def boom():
        raise FileNotFoundError("Model weights not found at weights/asl-yolov8m.pt")

    monkeypatch.setattr(api, "load_model", boom)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json()["status"] == "degraded"


@pytest.mark.parametrize("payload", [
    {},
    {"frame": encoded_frame()},
    {"target": "A"},
    {"frame": "", "target": "A"},
])
def test_missing_fields_are_rejected(client, payload):
    assert client.post("/detect", json=payload).status_code == 400


@pytest.mark.parametrize("target", ["AB", "1", "", "hello", "ñ"])
def test_target_must_be_a_single_letter(client, target):
    response = client.post("/detect", json={"frame": encoded_frame(), "target": target})
    assert response.status_code == 400


def test_target_is_case_insensitive(api, client, monkeypatch):
    seen = {}

    def fake_best_match(frame, target, threshold):
        seen["target"] = target
        return None

    monkeypatch.setattr(api, "best_match", fake_best_match)
    response = client.post("/detect", json={"frame": encoded_frame(), "target": "a"})
    assert response.status_code == 200
    assert seen["target"] == "A"


def test_undecodable_frame_is_a_client_error_not_a_crash(api, client, monkeypatch):
    def boom(_):
        raise FrameDecodeError("frame bytes could not be decoded as an image")

    monkeypatch.setattr(api, "decode_frame", boom)
    response = client.post("/detect", json={"frame": "!!!not base64!!!", "target": "A"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_reports_a_match_with_confidence_and_box(api, client, monkeypatch):
    monkeypatch.setattr(
        api, "best_match",
        lambda *a, **k: Detection(label="A", confidence=0.8123, box=(10, 20, 30, 40)),
    )
    body = client.post("/detect", json={"frame": encoded_frame(), "target": "A"}).get_json()
    assert body["target_detected"] is True
    assert body["confidence"] == 0.812
    assert body["box"] == [10, 20, 30, 40]
    assert body["target"] == "A"


def test_reports_no_match_without_failing(client):
    body = client.post("/detect", json={"frame": encoded_frame(), "target": "Z"}).get_json()
    assert body["target_detected"] is False
    assert body["confidence"] == 0.0
    assert body["box"] is None


def test_frame_dimensions_are_echoed_back(client):
    body = client.post(
        "/detect", json={"frame": encoded_frame(320, 240), "target": "A"}
    ).get_json()
    assert (body["width"], body["height"]) == (320, 240)
