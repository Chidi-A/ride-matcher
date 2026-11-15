"""Two-node LangGraph graph that matches one ride request to a driver.

Node one (rank_drivers) is plain code: it keeps only the request's drivers whose
status is exactly "available" and ranks them by road pickup minutes.
Node two (pick_driver) sends only that checked list to the OpenRouter model in
config.json and accepts the answer only if the driver is on the list.
"""
import csv
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from openrouter_client import chat

ROOT = Path(__file__).resolve().parent
REQUESTS_CSV = ROOT / "data" / "Uber Ride Requests Test Set (week of 3-9 March 2025).csv"
DRIVERS_CSV = ROOT / "data" / "Uber Candidate Drivers Snapshot (week of 3-9 March 2025).csv"

AVAILABLE_STATUSES = {"available"}
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


def candidate_rows(request_id: str) -> list[dict[str, str]]:
    return [r for r in _read_csv(DRIVERS_CSV) if r["request_id"] == request_id]


def _no_match(request_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "no_match", "request_id": request_id, "no_match_reason": reason, **extra}


# --- Node one -----------------------------------------------------------------

def rank_drivers(state: RideState) -> RideState:
    request = state["request"]
    available, excluded = [], []
    for row in candidate_rows(request["request_id"]):
        if row["driver_status"].strip() in AVAILABLE_STATUSES:
            available.append(row)
        else:
            excluded.append({"driver_id": row["driver_id"], "reason": row["driver_status"]})

    # Road minutes decide the order; straight-line km only breaks ties.
    available.sort(
        key=lambda r: (int(r["road_pickup_minutes"]), float(r["straight_line_km"]), r["driver_id"])
    )
    ranked = [
        {
            "rank": i,
            "driver_id": r["driver_id"],
            "road_pickup_minutes": int(r["road_pickup_minutes"]),
            "driver_river_side": r["driver_river_side"],
            "route_note": r["route_note"],
            "status_updated_at": r["status_updated_at"],
        }
        for i, r in enumerate(available, start=1)
    ]
    update: RideState = {"ranked_drivers": ranked, "excluded": excluded}
    if not ranked:
        update["result"] = _no_match(request["request_id"], "no_available_drivers")
    return update


def route_after_ranking(state: RideState) -> str:
    return "pick_driver" if state["ranked_drivers"] else END


# --- Node two -----------------------------------------------------------------

PROMPT = """You are dispatching one ride request. Pick exactly one driver from the list below.
The list contains only available drivers, already ranked by estimated road pickup minutes.
Use only these drivers. Consider road pickup minutes, route notes, and status timestamps.

Input:
{payload}

Reply with JSON only, no other text:
{{"driver_id": "<one driver_id from the list>", "pickup_minutes": <integer>, "reason": "<one sentence>"}}"""


def build_payload(state: RideState) -> dict[str, Any]:
    return {
        "request": {k: state["request"][k] for k in REQUEST_FIELDS},
        "drivers": state["ranked_drivers"],
    }


def _parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("driver_id") else None


def pick_driver(state: RideState, model_call=chat) -> RideState:
    request = state["request"]
    request_id = request["request_id"]
    prompt = PROMPT.format(payload=json.dumps(build_payload(state), indent=2))

    answer = None
    for _ in range(2):  # one retry, only for unparseable answers
        try:
            text = model_call([{"role": "user", "content": prompt}])
        except Exception as exc:  # network error, rate limit, bad key
            return {"result": _no_match(request_id, "model_error", error=str(exc))}
        answer = _parse_json(text)
        if answer:
            break
    if not answer:
        return {"result": _no_match(request_id, "invalid_model_response")}

    driver_id = str(answer["driver_id"]).strip()
    by_id = {d["driver_id"]: d for d in state["ranked_drivers"]}
    if driver_id not in by_id:
        return {"result": _no_match(request_id, "driver_not_on_list", model_driver_id=driver_id)}

    minutes = by_id[driver_id]["road_pickup_minutes"]
    eta = datetime.fromisoformat(request["requested_at"]) + timedelta(minutes=minutes)
    result = {
        "status": "matched",
        "request_id": request_id,
        "driver_id": driver_id,
        "pickup_minutes": minutes,
        "pickup_eta": eta.isoformat(),
        "reason": answer.get("reason", ""),
    }
    if answer.get("pickup_minutes") != minutes:
        result["model_minutes_mismatch"] = answer.get("pickup_minutes")
    return {"result": result}


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
    available = [r for r in rows.values() if r["driver_status"].strip() in AVAILABLE_STATUSES]
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
    request_id = sys.argv[1] if len(sys.argv) > 1 else "RQ-0306-012"
    result = run(request_id)["result"]
    print(f"Request:          {result['request_id']}")
    if result["status"] != "matched":
        print(f"No match:         {result['no_match_reason']}")
        sys.exit(1)
    print(f"Assigned driver:  {result['driver_id']}")
    print(f"Arrival estimate: {result['pickup_minutes']} min (ETA {result['pickup_eta']})")
    print(f"Why:              {explain_vs_nearest(request_id, result)}")
