import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Zone:
    name: str
    zone_type: str            # "trigger", "alpr", "ignore", ...
    action: Optional[str]     # "start_visit", "capture_plate", "end_visit", "skip"
    camera_id: Optional[int]  # 0 = ingress (cam1), 1 = egress (cam2)
    polygon: list             # list of [x, y] in normalized 0-1 coords

    def contains_point(self, x: float, y: float) -> bool:
        n = len(self.polygon)
        inside = False
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        xinters = p1x
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def overlaps_bbox(self, bbox) -> bool:
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        else:
            if len(bbox) != 4:
                return False
            x1, y1, x2, y2 = bbox
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                   ((x1 + x2) / 2, (y1 + y2) / 2)]
        return any(self.contains_point(cx, cy) for cx, cy in corners)


@dataclass
class ZoneConfig:
    zones: list = field(default_factory=list)

    def for_camera(self, camera_id: int) -> list:
        return [z for z in self.zones
                if z.camera_id is None or z.camera_id == camera_id]

    def zones_with_action(self, camera_id: int, action: str) -> list:
        return [z for z in self.for_camera(camera_id) if z.action == action]

    def has_action(self, camera_id: int, action: str) -> bool:
        return bool(self.zones_with_action(camera_id, action))


def load_zones(path: str) -> ZoneConfig:
    data = json.loads(Path(path).read_text())
    zones = [
        Zone(
            name=z["name"],
            zone_type=z["zone_type"],
            action=z.get("action"),
            camera_id=z.get("camera_id"),
            polygon=z["polygon"],
        )
        for z in data.get("zones", [])
    ]
    return ZoneConfig(zones=zones)
