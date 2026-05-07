from datetime import datetime, timedelta, timezone

from hailo_apps.python.standalone_apps.tunnelvision.correlator import (
    Correlator, ActiveVisit, EgressEvent,
)


def _ts(seconds_ago: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)


def test_match_by_exact_plate_wins():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="ABC1234",
                                  ingress_at=_ts(180), color="red"))
    c.enqueue_ingress(ActiveVisit(visit_id="v2", plate="DEF5678",
                                  ingress_at=_ts(160), color="blue"))
    eg = EgressEvent(plate="ABC1234", egress_at=_ts(0), color="red")
    match = c.match_egress(eg)
    assert match.visit_id == "v1"
    assert match.method == "plate_exact"
    assert match.confidence > 90


def test_match_by_queue_order_when_no_plate():
    c = Correlator(min_match_score=30.0, expected_tunnel_seconds=180)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate=None, ingress_at=_ts(180)))
    c.enqueue_ingress(ActiveVisit(visit_id="v2", plate=None, ingress_at=_ts(120)))
    eg = EgressEvent(plate=None, egress_at=_ts(0))
    match = c.match_egress(eg)
    assert match.visit_id == "v1"   # FIFO head
    assert match.method == "track_sequence"


def test_match_returns_none_below_threshold():
    c = Correlator(min_match_score=80.0, expected_tunnel_seconds=180)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="AAA", ingress_at=_ts(600)))
    eg = EgressEvent(plate="ZZZ", egress_at=_ts(0))
    match = c.match_egress(eg)
    assert match is None or match.confidence < 80


def test_completed_visits_are_removed_from_queue():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="v1", plate="ABC1234", ingress_at=_ts(180)))
    eg = EgressEvent(plate="ABC1234", egress_at=_ts(0))
    c.match_egress(eg)
    assert c.active_visit_count() == 0


def test_handles_three_simultaneous_cars_in_order():
    c = Correlator(min_match_score=60.0, expected_tunnel_seconds=300)
    c.enqueue_ingress(ActiveVisit(visit_id="A", plate="A1", ingress_at=_ts(300)))
    c.enqueue_ingress(ActiveVisit(visit_id="B", plate="B2", ingress_at=_ts(240)))
    c.enqueue_ingress(ActiveVisit(visit_id="C", plate="C3", ingress_at=_ts(180)))
    assert c.match_egress(EgressEvent(plate="A1", egress_at=_ts(0))).visit_id == "A"
    assert c.match_egress(EgressEvent(plate="B2", egress_at=_ts(0))).visit_id == "B"
    assert c.match_egress(EgressEvent(plate="C3", egress_at=_ts(0))).visit_id == "C"
    assert c.active_visit_count() == 0
