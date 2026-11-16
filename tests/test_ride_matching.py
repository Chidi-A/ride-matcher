"""End-to-end checks of the ride-matching graph against all 20 ride requests.

Expected values are computed here straight from the CSV files with the csv
module, not with the agent's own tool functions, so a bug in those tools can't
hide itself. MAX_EXTRA_PICKUP_MINUTES is imported from ride_config, the same
place the agent reads it.

Each graph test runs against several model behaviours:
  - "live": the real OpenRouter model (OPENROUTER_API_KEY + OPENROUTER_MODEL
    must be set; otherwise skipped). One call per request, cached for the session.
  - adversarial stubs that always answer badly, to prove the guardrails hold no
    matter what the model says.
"""
import csv
import json
import os
import re

import pytest

import openrouter_client
import ride_graph
from ride_config import MAX_EXTRA_PICKUP_MINUTES

# --- Ground truth from the raw CSVs ---------------------------------------------

with ride_graph.REQUESTS_CSV.open(newline="") as f:
    REQUESTS = list(csv.DictReader(f))
with ride_graph.DRIVERS_CSV.open(newline="") as f:
    SNAPSHOT = list(csv.DictReader(f))

REQUEST_IDS = [r["request_id"] for r in REQUESTS]


def rows_for(request_id):
    return [r for r in SNAPSHOT if r["request_id"] == request_id]


def available_rows(request_id):
    return [r for r in rows_for(request_id) if r["driver_status"] == "available"]


def fastest_minutes(request_id):
    return min(int(r["road_pickup_minutes"]) for r in available_rows(request_id))


def nearest_by_km(request_id):
    return min(available_rows(request_id), key=lambda r: float(r["straight_line_km"]))


def disagreement_request_ids():
    """Requests whose nearest available driver by straight_line_km is not a fastest one by road."""
    out = []
    for rid in REQUEST_IDS:
        if available_rows(rid):
            nearest = nearest_by_km(rid)
            if int(nearest["road_pickup_minutes"]) > fastest_minutes(rid):
                out.append(rid)
    return out


DISAGREEMENT_IDS = disagreement_request_ids()

# --- Model behaviours -------------------------------------------------------------


def _offered_ids(messages):
    payload = json.loads(re.search(r"Input:\n(\{.*\})\n\nReply", messages[0]["content"], re.S).group(1))
    return payload["request"]["request_id"], [d["driver_id"] for d in payload["drivers"]]


def picks_nearest_by_km(messages, response_format=None):
    """Ignores the offered list and names the nearest available driver by straight line."""
    rid, _ = _offered_ids(messages)
    return json.dumps({"driver_id": nearest_by_km(rid)["driver_id"], "reason": "closest on the map"})


def picks_unavailable(messages, response_format=None):
    """Names an offline/on_trip candidate if there is one, else an unknown id."""
    rid, _ = _offered_ids(messages)
    bad = [r["driver_id"] for r in rows_for(rid) if r["driver_status"] != "available"]
    return json.dumps({"driver_id": bad[0] if bad else "DRV-00000", "reason": "x"})


def picks_last_offered(messages, response_format=None):
    _, ids = _offered_ids(messages)
    return json.dumps({"driver_id": ids[-1], "reason": "slowest offered"})


def returns_garbage(messages, response_format=None):
    return "I think the best driver is probably the second one."


STUBS = {
    "nearest_by_km": picks_nearest_by_km,
    "unavailable": picks_unavailable,
    "last_offered": picks_last_offered,
    "garbage": returns_garbage,
}


def _live_configured():
    try:
        openrouter_client.get_api_key()
        openrouter_client.get_model()
        return True
    except RuntimeError:
        return False


MODELS = [
    pytest.param(
        "live",
        marks=[
            pytest.mark.live,
            pytest.mark.skipif(not _live_configured(), reason="OPENROUTER_API_KEY / OPENROUTER_MODEL not set"),
        ],
    ),
    *STUBS,
]


@pytest.fixture(scope="session")
def results():
    """Run each (model, request) pair once per session; live calls are the expensive part."""
    cache = {}

    def get(model, request_id):
        if (model, request_id) not in cache:
            call = openrouter_client.chat if model == "live" else STUBS[model]
            cache[(model, request_id)] = ride_graph.run(request_id, model_call=call)["result"]
        return cache[(model, request_id)]

    return get


# --- Tests --------------------------------------------------------------------------


def test_all_20_requests_present():
    assert len(REQUEST_IDS) == 20
    assert len(set(REQUEST_IDS)) == 20


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("request_id", REQUEST_IDS)
def test_each_request_gets_one_fast_available_driver_or_explicit_no_match(results, model, request_id):
    result = results(model, request_id)
    assert result["request_id"] == request_id

    if not available_rows(request_id):
        # No available candidate: the agent must say so explicitly, with no driver.
        assert result["status"] == "no_match"
        expected = "no_available_drivers" if rows_for(request_id) else "no_candidate_drivers"
        assert result["no_match_reason"] == expected
        assert "driver_id" not in result
        return

    assert result["status"] == "matched", result
    # Exactly one driver: a single id string that is one of this request's candidate rows.
    driver_id = result["driver_id"]
    assert isinstance(driver_id, str)
    matching_rows = [r for r in rows_for(request_id) if r["driver_id"] == driver_id]
    assert len(matching_rows) == 1, f"{driver_id} is not a single candidate row for {request_id}"
    row = matching_rows[0]

    assert row["driver_status"] == "available"
    minutes = int(row["road_pickup_minutes"])
    assert minutes <= fastest_minutes(request_id) + MAX_EXTRA_PICKUP_MINUTES
    assert result["pickup_minutes"] == minutes


def test_route_note_bridge_request_is_a_distance_time_disagreement():
    """The one-way-bridge route_note marks a request where km and road minutes pick different drivers."""
    flagged = {
        r["request_id"]
        for r in SNAPSHOT
        if r["driver_status"] == "available" and "one-way bridge" in r["route_note"].lower()
    }
    assert flagged == {"RQ-0307-014"}
    (request_id,) = flagged
    assert request_id in DISAGREEMENT_IDS
    assert nearest_by_km(request_id)["driver_id"] == "DRV-48216"  # 0.87 km but 17 road minutes
    assert "one-way bridge" in nearest_by_km(request_id)["route_note"]


def test_disagreement_set_is_found_from_data():
    # Guard against the helper silently finding nothing (which would make the test below vacuous).
    assert "RQ-0307-014" in DISAGREEMENT_IDS
    assert len(DISAGREEMENT_IDS) >= 2


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("request_id", DISAGREEMENT_IDS)
def test_straight_line_nearest_does_not_beat_road_time(results, model, request_id):
    """For every km-vs-road disagreement, the assignment follows road minutes, not straight-line km."""
    result = results(model, request_id)
    assert result["status"] == "matched", result
    nearest = nearest_by_km(request_id)
    assigned = next(r for r in rows_for(request_id) if r["driver_id"] == result["driver_id"])

    assert int(assigned["road_pickup_minutes"]) <= fastest_minutes(request_id) + MAX_EXTRA_PICKUP_MINUTES
    if int(nearest["road_pickup_minutes"]) > fastest_minutes(request_id) + MAX_EXTRA_PICKUP_MINUTES:
        assert result["driver_id"] != nearest["driver_id"]


def test_no_available_candidates_gives_explicit_no_match(monkeypatch):
    """A request whose candidates are all offline/on_trip must end in no_match without calling the model."""
    request_id = "RQ-0306-012"
    patched = tuple(
        {**r, "driver_status": "offline"} if r["request_id"] == request_id else r for r in SNAPSHOT
    )
    monkeypatch.setattr(ride_graph, "_snapshot", lambda: patched)
    monkeypatch.setattr(
        ride_graph, "_rows_by_key", lambda: {(r["driver_id"], r["request_id"]): r for r in patched}
    )

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("model was called with no available drivers")

    result = ride_graph.run(request_id, model_call=must_not_be_called)["result"]
    assert result["status"] == "no_match"
    assert result["no_match_reason"] == "no_available_drivers"
    assert "driver_id" not in result
