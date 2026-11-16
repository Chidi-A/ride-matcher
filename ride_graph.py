"""Two-node LangGraph graph that matches one ride request to a driver.

Node one (rank_drivers) is plain code: it runs the tool functions below
(is_eligible, is_available, pickup_minutes) on every candidate row for the
request, keeps only drivers who pass and whose road pickup minutes are within
MAX_EXTRA_PICKUP_MINUTES (ride_config.py) of the fastest one, and ranks them by
road pickup minutes.
Node two (pick_driver) sends only that checked list to the OpenRouter model named
by OPENROUTER_MODEL, with a JSON schema asking for one driver_id and a one-line
reason. A driver_id not on the checked list (or an unparseable reply) is
rejected and replaced by the checked candidate with the lowest road pickup
minutes, never the lowest straight-line km.
"""
import csv
import json
import re
import sys
from functools import lru_cache
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from openrouter_client import chat
from ride_config import MAX_EXTRA_PICKUP_MINUTES

ROOT = Path(__file__).resolve().parent
REQUESTS_CSV = ROOT / "data" / "Uber Ride Requests Test Set (week of 3-9 March 2025).csv"
DRIVERS_CSV = ROOT / "data" / "Uber Candidate Drivers Snapshot (week of 3-9 March 2025).csv"

REQUEST_FIELDS = ("request_id", "requested_at", "pickup_zone", "pickup_river_side")


class RideState(TypedDict, total=False):
    request: dict[str, str]
    ranked_drivers: list[dict[str, Any]]
    excluded: list[dict[str, str]]
    result: dict[str, Any]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_request(request_id: str) -> dict[str, str]:
    for row in _read_csv(REQUESTS_CSV):
        if row["request_id"] == request_id:
            return row
    raise KeyError(f"Request {request_id} not found in {REQUESTS_CSV.name}")


@lru_cache(maxsize=1)
def _snapshot() -> tuple[dict[str, str], ...]:
    return tuple(_read_csv(DRIVERS_CSV))


@lru_cache(maxsize=1)
def _rows_by_key() -> dict[tuple[str, str], dict[str, str]]:
    return {(r["driver_id"], r["request_id"]): r for r in _snapshot()}


def candidate_rows(request_id: str) -> list[dict[str, str]]:
    return [r for r in _snapshot() if r["request_id"] == request_id]


# --- Tool functions -------------------------------------------------------------
# A driver can appear under several requests with different statuses (e.g.
# DRV-73351 is on_trip for RQ-0304-004 but available for RQ-0307-015), so
# availability is read from the row for the request being matched. Called with
# a driver_id alone, is_available uses that driver's most recent status.

def is_available(driver_id: str, request_id: str | None = None) -> bool:
    """True only when driver_status is exactly "available"."""
    if request_id is not None:
        row = _rows_by_key().get((driver_id, request_id))
    else:
        rows = [r for r in _snapshot() if r["driver_id"] == driver_id]
        row = max(rows, key=lambda r: datetime.fromisoformat(r["status_updated_at"]), default=None)
    return row is not None and row["driver_status"].strip() == "available"


def pickup_minutes(driver_id: str, request_id: str) -> int:
    """road_pickup_minutes for this driver to this request's pickup point."""
    row = _rows_by_key().get((driver_id, request_id))
    if row is None:
        raise KeyError(f"{driver_id} is not a candidate for {request_id}")
    return int(row["road_pickup_minutes"])


def is_eligible(driver_id: str, request_id: str) -> bool:
    """True only when the driver is a candidate row for request_id and is available for it."""
    return (driver_id, request_id) in _rows_by_key() and is_available(driver_id, request_id)


def count_available_rows() -> int:
    return sum(is_available(r["driver_id"], r["request_id"]) for r in _snapshot())


def _no_match(request_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "no_match", "request_id": request_id, "no_match_reason": reason, **extra}


# --- Node one -----------------------------------------------------------------

def rank_drivers(state: RideState) -> RideState:
    request = state["request"]
    request_id = request["request_id"]
    candidates = candidate_rows(request_id)
    eligible, excluded = [], []
    for row in candidates:
        if is_eligible(row["driver_id"], request_id):
            eligible.append(row)
        else:
            excluded.append({"driver_id": row["driver_id"], "reason": row["driver_status"]})

    # Drop anyone too much slower than the fastest eligible driver, so the model
    # can never be offered (or pick) a nearer-looking but slower driver.
    if eligible:
        fastest = min(pickup_minutes(r["driver_id"], request_id) for r in eligible)
        limit = fastest + MAX_EXTRA_PICKUP_MINUTES
        for row in [r for r in eligible if pickup_minutes(r["driver_id"], request_id) > limit]:
            eligible.remove(row)
            excluded.append({"driver_id": row["driver_id"], "reason": "exceeds_max_extra_pickup_minutes"})

    # Road minutes decide the order; straight-line km never does. Ties go to driver_id
    # so the order (and the fallback, rank 1) is deterministic.
    eligible.sort(key=lambda r: (pickup_minutes(r["driver_id"], request_id), r["driver_id"]))
    ranked = [
        {
            "rank": i,
            "driver_id": r["driver_id"],
            "road_pickup_minutes": pickup_minutes(r["driver_id"], request_id),
            "driver_river_side": r["driver_river_side"],
            "route_note": r["route_note"],
            "status_updated_at": r["status_updated_at"],
        }
        for i, r in enumerate(eligible, start=1)
    ]
    update: RideState = {"ranked_drivers": ranked, "excluded": excluded}
    if not ranked:
        reason = "no_available_drivers" if candidates else "no_candidate_drivers"
        update["result"] = _no_match(request_id, reason)
    return update


def route_after_ranking(state: RideState) -> str:
    return "pick_driver" if state["ranked_drivers"] else END


# --- Node two -----------------------------------------------------------------

PROMPT = """You are dispatching one ride request. Pick exactly one driver from the list below.
The list contains only checked, available drivers within {max_extra} minutes of the fastest,
ranked by estimated road pickup minutes.
Use only these drivers. Judge by road pickup minutes, route notes, and status timestamps,
not by straight-line distance.

Input:
{payload}

Reply with JSON only: {{"driver_id": "<one driver_id from the list>", "reason": "<one line>"}}"""


def build_payload(state: RideState) -> dict[str, Any]:
    return {
        "request": {k: state["request"][k] for k in REQUEST_FIELDS},
        "drivers": state["ranked_drivers"],
    }


def choice_schema(driver_ids: list[str]) -> dict[str, Any]:
    """OpenRouter structured-output schema: one driver_id from the checked list plus a reason."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "driver_choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "driver_id": {"type": "string", "enum": driver_ids},
                    "reason": {"type": "string", "description": "One line explaining the choice."},
                },
                "required": ["driver_id", "reason"],
                "additionalProperties": False,
            },
        },
    }


def _parse_choice(text: str) -> dict[str, str] | None:
    """Return {"driver_id", "reason"} or None if the reply isn't that shape."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Some providers wrap JSON in prose or code fences despite the schema.
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict) or not isinstance(data.get("driver_id"), str):
        return None
    reason = data.get("reason")
    reason = " ".join(reason.split()) if isinstance(reason, str) else ""
    return {"driver_id": data["driver_id"].strip(), "reason": reason}


def fallback_driver(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    """Checked candidate with the lowest road_pickup_minutes (ties by driver_id). Never straight-line km."""
    return min(ranked, key=lambda d: (d["road_pickup_minutes"], d["driver_id"]))


def pick_driver(state: RideState, model_call=chat) -> RideState:
    request = state["request"]
    request_id = request["request_id"]
    ranked = state["ranked_drivers"]
    by_id = {d["driver_id"]: d for d in ranked}
    messages = [{"role": "user", "content": PROMPT.format(max_extra=MAX_EXTRA_PICKUP_MINUTES, payload=json.dumps(build_payload(state), indent=2))}]
    response_format = choice_schema(list(by_id))

    choice = None
    for _ in range(2):  # one retry, only for unparseable answers
        try:
            text = model_call(messages, response_format=response_format)
        except Exception as exc:  # network error, rate limit after retries, bad key
            return {"result": _no_match(request_id, "model_error", error=str(exc))}
        choice = _parse_choice(text)
        if choice:
            break

    extra: dict[str, Any] = {}
    if choice and choice["driver_id"] in by_id:
        driver = by_id[choice["driver_id"]]
        source, reason = "model", choice["reason"]
    else:
        driver = fallback_driver(ranked)
        source = "fallback"
        if choice:
            extra = {"fallback_reason": "driver_not_on_list", "model_driver_id": choice["driver_id"]}
        else:
            extra = {"fallback_reason": "invalid_model_response"}
        reason = f"Fallback: lowest road pickup minutes among checked candidates ({extra['fallback_reason']})."

    minutes = driver["road_pickup_minutes"]
    eta = datetime.fromisoformat(request["requested_at"]) + timedelta(minutes=minutes)
    return {
        "result": {
            "status": "matched",
            "request_id": request_id,
            "driver_id": driver["driver_id"],
            "pickup_minutes": minutes,
            "pickup_eta": eta.isoformat(),
            "reason": reason,
            "choice_source": source,
            **extra,
        }
    }


# --- Graph --------------------------------------------------------------------

def build_graph(model_call=chat):
    graph = StateGraph(RideState)
    graph.add_node("rank_drivers", rank_drivers)
    graph.add_node("pick_driver", lambda s: pick_driver(s, model_call))
    graph.add_edge(START, "rank_drivers")
    graph.add_conditional_edges("rank_drivers", route_after_ranking, ["pick_driver", END])
    graph.add_edge("pick_driver", END)
    return graph.compile()


def run(request_id: str, model_call=chat) -> RideState:
    return build_graph(model_call).invoke({"request": load_request(request_id)})


def explain_vs_nearest(request_id: str, result: dict[str, Any]) -> str:
    """One sentence comparing the assigned driver with the nearest available one by straight line."""
    rows = {r["driver_id"]: r for r in candidate_rows(request_id)}
    available = [r for r in rows.values() if is_eligible(r["driver_id"], request_id)]
    nearest = min(available, key=lambda r: float(r["straight_line_km"]))
    chosen = rows[result["driver_id"]]
    if chosen["driver_id"] == nearest["driver_id"]:
        return f"{chosen['driver_id']} is also the nearest available driver in straight-line distance."
    note = f" ({nearest['route_note']})" if nearest["route_note"] else ""
    return (
        f"{chosen['driver_id']} is {chosen['straight_line_km']} km away but {chosen['road_pickup_minutes']} road "
        f"minutes out, beating {nearest['driver_id']}, which is nearer at {nearest['straight_line_km']} km "
        f"but needs {nearest['road_pickup_minutes']} road minutes{note}."
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--count-available"]:
        print(f"{count_available_rows()} of {len(_snapshot())} snapshot rows pass the availability check")
        sys.exit(0)
    request_id = sys.argv[1] if len(sys.argv) > 1 else "RQ-0306-012"
    result = run(request_id)["result"]
    print(f"Request:          {result['request_id']}")
    if result["status"] != "matched":
        print(f"No match:         {result['no_match_reason']}")
        if result.get("error"):
            print(f"Error:            {result['error']}")
        sys.exit(1)
    print(f"Assigned driver:  {result['driver_id']}")
    print(f"Arrival estimate: {result['pickup_minutes']} min (ETA {result['pickup_eta']})")
    print(f"Chosen by:        {result['choice_source']}")
    print(f"Model reason:     {result['reason']}")
    print(f"Why:              {explain_vs_nearest(request_id, result)}")
