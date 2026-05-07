import json
from pathlib import Path

import pytest

from hailo_apps.python.standalone_apps.tunnelvision.zone import (
    Zone, ZoneConfig, load_zones,
)


def test_contains_point_inside_square():
    z = Zone(name="t", zone_type="trigger", action="start_visit", camera_id=0,
             polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert z.contains_point(0.5, 0.5)
    assert not z.contains_point(1.5, 0.5)
    assert not z.contains_point(-0.1, 0.5)


def test_contains_point_outside_pentagon():
    z = Zone(name="t", zone_type="trigger", action="start_visit", camera_id=0,
             polygon=[[0.325, 0.2824], [0.1297, 0.988], [0.9695, 0.9722],
                      [0.7219, 0.2167], [0.3281, 0.2852]])
    assert z.contains_point(0.5, 0.6)
    assert not z.contains_point(0.0, 0.0)


def test_overlaps_bbox():
    z = Zone(name="t", zone_type="alpr", action="capture_plate", camera_id=0,
             polygon=[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
    assert z.overlaps_bbox([0.5, 0.5, 0.6, 0.6])
    assert z.overlaps_bbox([0.1, 0.1, 0.5, 0.5])  # one corner inside
    assert not z.overlaps_bbox([0.0, 0.0, 0.1, 0.1])


def test_load_zones_from_file(tmp_path: Path):
    config = {
        "zones": [
            {
                "name": "ingress_gate", "zone_type": "trigger", "camera_id": 0,
                "action": "start_visit",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            },
            {
                "name": "lpr_zone", "zone_type": "alpr", "camera_id": 0,
                "action": "capture_plate",
                "polygon": [[0.2, 0.2], [0.7, 0.2], [0.7, 0.7], [0.2, 0.7]],
            },
            {
                "name": "egress_gate", "zone_type": "trigger", "camera_id": 1,
                "action": "end_visit",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            },
        ],
    }
    p = tmp_path / "z.json"
    p.write_text(json.dumps(config))

    cfg = load_zones(str(p))
    assert len(cfg.for_camera(0)) == 2
    assert len(cfg.for_camera(1)) == 1
    assert cfg.zones_with_action(0, "capture_plate")[0].name == "lpr_zone"


def test_load_real_zones_file():
    """Smoke test against the lalaland site config shipped with the experiment."""
    cfg = load_zones("experiments/tunnelvision/zones-lalaland.json")
    cam0 = cfg.for_camera(0)
    cam1 = cfg.for_camera(1)
    assert len(cam0) >= 2
    assert any(z.action == "start_visit" for z in cam0)
    assert any(z.action == "capture_plate" for z in cam0)


def test_load_jajamaica_zones_file():
    """jajamaica config exists and has the three expected zones, even with placeholder polygons."""
    cfg = load_zones("experiments/tunnelvision/zones-jajamaica.json")
    cam0 = cfg.for_camera(0)
    cam1 = cfg.for_camera(1)
    assert any(z.action == "start_visit" for z in cam0)
    assert any(z.action == "capture_plate" for z in cam0)
    assert any(z.action == "end_visit" for z in cam1)
    assert any(z.action == "end_visit" for z in cam1)
