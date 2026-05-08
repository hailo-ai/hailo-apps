from unittest.mock import patch, MagicMock

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.rekor import (
    CreditPolicy, parse_rekor_response, should_call_rekor, call_rekor_carcheck,
)


SAMPLE_RESPONSE = {
    "data_type": "alpr_results",
    "epoch_time": 1714998000000,
    "img_width": 1920, "img_height": 1080,
    "error": False, "version": 2, "uuid": "abc",
    "credit_cost": 1, "credits_monthly_used": 100, "credits_monthly_total": 1000,
    "processing_time": {"total": 230.0, "plates": 200.0, "vehicles": 30.0},
    "regions_of_interest": [],
    "results": [
        {
            "plate": "ABC1234",
            "region": "ny",
            "confidence": 92.5,
            "region_confidence": 80.0,
            "matches_template": 1,
            "coordinates": [[0,0],[10,0],[10,5],[0,5]],
            "vehicle_detected": True,
            "candidates": [
                {"plate": "ABC1234", "confidence": 92.5, "matches_template": 1},
                {"plate": "ABC1Z34", "confidence": 80.1, "matches_template": 0},
            ],
            "vehicle": {
                "make":[{"name":"Honda","confidence":80.0}],
                "make_model":[{"name":"Honda Civic","confidence":75.0}],
                "color":[{"name":"red","confidence":90.0}],
                "year":[{"name":"2018-2020","confidence":60.0}],
                "orientation":[{"name":"front","confidence":85.0}],
                "body_type":[{"name":"sedan","confidence":80.0}],
            },
        },
    ],
}


def test_parse_rekor_response_extracts_top_plate():
    parsed = parse_rekor_response(SAMPLE_RESPONSE)
    assert parsed["plate"] == "ABC1234"
    assert parsed["plate_confidence"] == 92.5
    assert parsed["region"] == "ny"
    assert parsed["credit_cost"] == 1
    assert parsed["candidates"][0]["plate"] == "ABC1234"
    assert parsed["make"] == "Honda"


def test_parse_rekor_response_empty_results():
    parsed = parse_rekor_response({"results": [], "credit_cost": 0})
    assert parsed["plate"] is None


def test_should_call_rekor_skip_low_quality():
    p = CreditPolicy(min_quality_score_to_call=80.0)
    ok, recv, reason = should_call_rekor(quality_score=50.0, camera_role="ingress",
                                         credits_used=0, monthly_budget=500,
                                         known_recently=False, policy=p)
    assert not ok
    assert reason == "skip_low_quality"


def test_should_call_rekor_skip_egress():
    p = CreditPolicy(call_rekor_on_egress=False)
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="egress",
                                      credits_used=0, monthly_budget=500,
                                      known_recently=False, policy=p)
    assert not ok
    assert reason == "skip_egress"


def test_should_call_rekor_skip_known_plate():
    p = CreditPolicy()
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                      credits_used=0, monthly_budget=500,
                                      known_recently=True, policy=p)
    assert not ok
    assert reason == "skip_known_plate"


def test_should_call_rekor_emergency_conservation_known():
    p = CreditPolicy(emergency_threshold=0.85)
    ok, _, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                      credits_used=900, monthly_budget=1000,
                                      known_recently=True, policy=p)
    assert not ok
    assert reason == "emergency_credit_conservation"


def test_should_call_rekor_happy_path():
    p = CreditPolicy()
    ok, recv, reason = should_call_rekor(quality_score=90.0, camera_role="ingress",
                                         credits_used=0, monthly_budget=500,
                                         known_recently=False, policy=p)
    assert ok
    assert reason == "plate_only"


def test_call_rekor_carcheck_posts_base64(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = SAMPLE_RESPONSE
    fake_resp.raise_for_status.return_value = None
    with patch("hailo_apps.python.standalone_apps.tunnelvision.rekor.requests.post",
               return_value=fake_resp) as mock_post:
        out = call_rekor_carcheck(str(img), secret_key="SK", recognize_vehicle=False)
    mock_post.assert_called_once()
    assert out["plate"] == "ABC1234"
