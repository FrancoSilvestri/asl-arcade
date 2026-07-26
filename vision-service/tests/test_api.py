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


class TestLegacyEndpoint:
    """The root path serves the contract the shipped Unity build was built against.

    Its FlaskResponse deserialises exactly six fields. Renaming or dropping any of
    them silently breaks a client that is already distributed, so they are pinned
    here.
    """

    FIELDS = {"message", "width", "height", "channels", "conf", "target_detected"}

    def test_response_carries_exactly_the_fields_the_build_expects(self, client):
        body = client.post("/", json={"frame": encoded_frame(), "target": "A"}).get_json()
        assert set(body) == self.FIELDS

    def test_confidence_is_named_conf_and_is_zero_without_a_match(self, client):
        body = client.post("/", json={"frame": encoded_frame(), "target": "A"}).get_json()
        assert body["target_detected"] is False
        assert body["conf"] == 0

    def test_conf_carries_the_real_value_on_a_match(self, api, client, monkeypatch):
        monkeypatch.setattr(
            api, "best_match",
            lambda *a, **k: Detection(label="A", confidence=0.8123, box=(1, 2, 3, 4)),
        )
        body = client.post("/", json={"frame": encoded_frame(), "target": "A"}).get_json()
        assert body["target_detected"] is True
        assert body["conf"] == 0.81

    def test_channels_are_reported(self, client):
        body = client.post("/", json={"frame": encoded_frame(), "target": "A"}).get_json()
        assert body["channels"] == 3

    def test_dimensions_are_echoed_back(self, client):
        body = client.post(
            "/", json={"frame": encoded_frame(320, 240), "target": "A"}
        ).get_json()
        assert (body["width"], body["height"]) == (320, 240)

    @pytest.mark.parametrize("payload,reason", [
        ({}, "no fields"),
        ({"frame": encoded_frame()}, "no target"),
        ({"frame": encoded_frame(), "target": "AB"}, "target is not one letter"),
    ])
    def test_shares_the_validation_rules_of_detect(self, client, payload, reason):
        assert client.post("/", json=payload).status_code == 400, reason

    def test_both_endpoints_agree_on_the_verdict(self, api, client, monkeypatch):
        monkeypatch.setattr(
            api, "best_match",
            lambda *a, **k: Detection(label="C", confidence=0.77, box=(0, 0, 1, 1)),
        )
        payload = {"frame": encoded_frame(), "target": "C"}
        legacy = client.post("/", json=payload).get_json()
        modern = client.post("/detect", json=payload).get_json()
        assert legacy["target_detected"] == modern["target_detected"] is True
        assert legacy["conf"] == round(modern["confidence"], 2)
        assert (legacy["width"], legacy["height"]) == (modern["width"], modern["height"])
