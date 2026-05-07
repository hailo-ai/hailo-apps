import base64
from dataclasses import dataclass
from typing import Optional, Tuple

import requests


REKOR_BASE_URL = "https://api.openalpr.com/v3/recognize_bytes"


@dataclass
class CreditPolicy:
    monthly_credit_budget: int = 500
    min_quality_score_to_call: float = 80.0
    min_plate_confidence_to_call: float = 0.70
    emergency_threshold: float = 0.85
    call_rekor_on_ingress: bool = True
    call_rekor_on_egress: bool = False
    default_recognize_vehicle: bool = False


def _normalize_plate(plate: Optional[str]) -> Optional[str]:
    if plate is None:
        return None
    return "".join(c for c in plate.upper() if c.isalnum())


def _top_attr(vehicle: Optional[dict], key: str) -> Tuple[Optional[str], Optional[float]]:
    values = (vehicle or {}).get(key) or []
    if not values:
        return None, None
    return values[0].get("name"), values[0].get("confidence")


def parse_rekor_response(payload: dict) -> dict:
    first_result = (payload.get("results") or [None])[0]
    first_vehicle = first_result.get("vehicle") if first_result else None
    make, make_conf = _top_attr(first_vehicle, "make")
    make_model, make_model_conf = _top_attr(first_vehicle, "make_model")
    color, color_conf = _top_attr(first_vehicle, "color")
    year, year_conf = _top_attr(first_vehicle, "year")
    orientation, orientation_conf = _top_attr(first_vehicle, "orientation")
    body_type, body_type_conf = _top_attr(first_vehicle, "body_type")

    proc = payload.get("processing_time") or {}
    return {
        "data_type": payload.get("data_type"),
        "epoch_time": payload.get("epoch_time"),
        "img_width": payload.get("img_width"),
        "img_height": payload.get("img_height"),
        "error": payload.get("error"),
        "version": payload.get("version"),
        "uuid": payload.get("uuid"),
        "credit_cost": payload.get("credit_cost"),
        "credits_monthly_used": payload.get("credits_monthly_used"),
        "credits_monthly_total": payload.get("credits_monthly_total"),
        "processing_time_total_ms": proc.get("total"),
        "processing_time_plates_ms": proc.get("plates"),
        "processing_time_vehicles_ms": proc.get("vehicles"),
        "regions_of_interest": payload.get("regions_of_interest"),
        "plate": first_result.get("plate") if first_result else None,
        "normalized_plate": _normalize_plate(first_result.get("plate") if first_result else None),
        "region": first_result.get("region") if first_result else None,
        "plate_confidence": first_result.get("confidence") if first_result else None,
        "region_confidence": first_result.get("region_confidence") if first_result else None,
        "matches_template": first_result.get("matches_template") if first_result else None,
        "coordinates": first_result.get("coordinates") if first_result else None,
        "candidates": (first_result.get("candidates") if first_result else []) or [],
        "vehicle_detected": first_result.get("vehicle_detected") if first_result else None,
        "make": make, "make_confidence": make_conf,
        "make_model": make_model, "make_model_confidence": make_model_conf,
        "color": color, "color_confidence": color_conf,
        "year_range": year, "year_confidence": year_conf,
        "orientation": orientation, "orientation_confidence": orientation_conf,
        "body_type": body_type, "body_type_confidence": body_type_conf,
    }


def should_call_rekor(
    *,
    quality_score: float,
    camera_role: str,
    credits_used: int,
    monthly_budget: int,
    known_recently: bool,
    policy: CreditPolicy,
) -> Tuple[bool, bool, str]:
    """Returns (call?, recognize_vehicle?, reason)."""
    if quality_score < policy.min_quality_score_to_call:
        return False, False, "skip_low_quality"
    if camera_role == "egress" and not policy.call_rekor_on_egress:
        return False, False, "skip_egress"
    if camera_role == "ingress" and not policy.call_rekor_on_ingress:
        return False, False, "skip_ingress_disabled"
    monthly_ratio = (credits_used / monthly_budget) if monthly_budget > 0 else 0.0
    if monthly_ratio >= policy.emergency_threshold:
        if known_recently:
            return False, False, "emergency_credit_conservation"
        return True, False, "plate_only"
    if known_recently:
        return False, False, "skip_known_plate"
    return True, policy.default_recognize_vehicle, (
        "vehicle_enrichment" if policy.default_recognize_vehicle else "plate_only"
    )


def call_rekor_carcheck(
    image_path: str,
    *,
    secret_key: str,
    recognize_vehicle: bool = False,
    country: str = "us",
    timeout_seconds: int = 8,
) -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read())
    url = (
        f"{REKOR_BASE_URL}"
        f"?recognize_vehicle={1 if recognize_vehicle else 0}"
        f"&country={country}"
        f"&secret_key={secret_key}"
    )
    response = requests.post(url, data=img_b64, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_rekor_response(response.json())
