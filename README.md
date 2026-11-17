# ride-matcher

Matches one ride request to a driver with a two-node LangGraph graph:

1. **`rank_drivers`** (plain code) runs three tool functions on every candidate row for the request, keeps the drivers who pass, drops any of them more than `MAX_EXTRA_PICKUP_MINUTES` (in `ride_config.py`, currently 3) slower than the fastest one, and ranks the rest by estimated road pickup minutes (ties broken by driver_id; straight-line km is never used):
   - `is_available(driver_id, request_id=None)`: true only when `driver_status` is `available`. A driver can show up under several requests with different statuses, so the graph always passes `request_id`. Called with a driver_id alone, it uses that driver's most recent status.
   - `pickup_minutes(driver_id, request_id)`: that row's `road_pickup_minutes`.
   - `is_eligible(driver_id, request_id)`: true only when the driver is a candidate row for that request and passes `is_available`.
2. **`pick_driver`** sends only that checked list to the OpenRouter model named by `OPENROUTER_MODEL`. It uses a strict JSON schema (`response_format`, with `driver_id` limited to the checked IDs) asking for `{"driver_id", "reason"}`, where the reason is one line. A `driver_id` not on the checked list, or a reply that can't be parsed after one retry, is rejected. The graph then falls back to the checked candidate with the lowest `road_pickup_minutes`, never the lowest straight-line km. The result's `choice_source` (`model` or `fallback`) and `fallback_reason` show which happened. A failed model call (network, auth, rate limit after retries) returns a `model_error` no-match. With no checked drivers, the graph returns `no_available_drivers` or `no_candidate_drivers`.
3. **Rate limits:** a 429 from OpenRouter is retried up to 3 times. The pause comes from `Retry-After`, then `X-RateLimit-Reset`, then 5/10/20 s backoff, capped at 60 s. A daily free-model quota (`free-models-per-day`) isn't retried.

## Run output: `RQ-0306-012`

`RQ-0306-012` is a bridge-case request (Rose Quarter, east side) where straight-line distance and road time disagree.

```
$ .venv/bin/python ride_graph.py RQ-0306-012
Request:          RQ-0306-012
Assigned driver:  DRV-45581
Arrival estimate: 3 min (ETA 2025-03-06T19:15:40-08:00)
Chosen by:        model
Model reason:     Only available driver, on the east side matching the pickup river side, with a 3-minute road pickup and recent status update.
Why:              DRV-45581 is 0.61 km away but 3 road minutes out, beating DRV-17239, which is nearer at 0.36 km but needs 9 road minutes (Freight hold at SE Division Pl rail crossing; routed via SE 11th Ave).
```

**Why DRV-45581 won:** DRV-17239 looked nearest at 0.36 km, but a freight hold at the SE Division Pl rail crossing makes it 9 road minutes away, while DRV-45581 reaches the pickup in 3.

### OpenRouter Activity

The model call for this run (`nvidia/nemotron-3.5-super-vl:free`):

![OpenRouter Activity showing one Nemotron 3.5 Super VL (free) call](docs/openrouter-activity.png)
Call details: 2026-11-15 07:21 (GMT+1), `nvidia/nemotron-3.5-super-vl:free`, Default key, 488 input / 691 output tokens, finish reason `stop`. This is the only call on the account, so it is the `RQ-0306-012` run above.

## Plan-mode prompt

> Plan a two-node LangGraph graph for one ride request. Node one is plain code: it keeps only that request's available drivers from the drivers snapshot and ranks them by estimated road pickup minutes, not straight-line distance. Node two sends only that checked list to the OpenRouter model named in config and asks it to pick one driver and give an arrival estimate. Do not write any code yet.

## Pushback

The first plan had no safe outcome when matching failed: if the model returned an invalid answer it silently assigned the top-ranked driver anyway, and a request with no available drivers just ended with no defined result. The approved plan closed this by having node two check that the returned driver is on the checked list and return an explicit no-match result (`driver_not_on_list`, `invalid_model_response`, `model_error`, `no_available_drivers`, or `no_candidate_drivers`) instead of a driver.

*Later change:* an off-list or unparseable model answer now falls back to the checked candidate with the lowest road pickup minutes, flagged with `choice_source: "fallback"`, instead of returning no match. That fallback is still limited to the checked list. Model errors and empty candidate lists still return a no-match.

## Tests

`MAX_EXTRA_PICKUP_MINUTES` is defined once in `ride_config.py`, and both `ride_graph.py` and the tests import it from there.

- `.venv/bin/python -m pytest -m "not live"`: offline. Four deliberately bad stand-in models (picks the nearest driver by km, picks an unavailable driver, picks the slowest offered driver, returns text that isn't JSON) run against all 20 requests.
- `.venv/bin/python -m pytest`: adds the `live` tests, one real OpenRouter call per request (19 calls). They're skipped if `OPENROUTER_API_KEY` or `OPENROUTER_MODEL` isn't set.

For every request, the tests check that exactly one available driver is assigned, within `MAX_EXTRA_PICKUP_MINUTES` of the fastest available candidate. A request with no available candidate must return an explicit no-match. Expected values are computed from the CSVs directly. A separate test covers every request where the nearest driver by straight-line km isn't the fastest by road, including RQ-0307-014, whose route_note mentions the one-way bridge.

## Design decisions

### Threshold: `MAX_EXTRA_PICKUP_MINUTES = 3`

It's set in `ride_config.py`. It started at 3, was raised to 5 to see the effect, and is back at 3.

**What the value changes in this dataset:**

- **At 3:** 11 requests offer the model more than one driver. Each extra driver is at most 3 minutes slower than the fastest.
- **At 5:** 14 requests would. The three extra ones are requests where the driver who looks nearest by straight-line km is 4–5 minutes slower by road:
  - RQ-0304-005: DRV-26148 (8 min) would be offered next to DRV-47038 (4 min).
  - RQ-0305-009: DRV-91254 (7 min) would be offered next to DRV-57120 (3 min).
  - RQ-0308-017: DRV-53761 (9 min) would be offered next to DRV-34871 (4 min).
- **Spreading rides across drivers:** at 3, DRV-57120 is the only option for RQ-0305-009, and DRV-34871 is the only option for RQ-0308-017. Each is also the fastest driver for another request, but those requests have alternatives: RQ-0305-008 also offers DRV-68831, and RQ-0306-011 also offers DRV-50316 and DRV-88092. A limit of 5 would give the model an alternative for RQ-0305-009 and RQ-0308-017.
- **Rider wait:** at 3, an assigned driver is never more than 3 minutes slower than the fastest. At 5, a rider could wait 9 minutes when a 4-minute driver was free.
- **Unaffected either way:** the four requests where the nearest-by-km driver is 6–13 minutes slower (RQ-0303-002, RQ-0305-007, RQ-0306-012, RQ-0307-014) never offer that driver.

**Why 3 was chosen:** rider wait comes first. A limit of 5 would add an alternative on only 3 requests, and in all three that alternative is the nearer-looking driver who is 4–5 minutes slower by road: the exact mistake this matcher exists to avoid. At 3, rides can still be spread across drivers, since 11 of the 19 matched requests offer a second driver who is at most 3 minutes slower. The cost is that DRV-57120 is the only option for RQ-0305-009 and DRV-34871 is the only option for RQ-0308-017.

**What would make me change it:**

- **Lower it** if rider complaints or cancellations rise with pickup time.
- **Raise it** if the same drivers keep getting assigned while nearby drivers stay idle, or if one driver is the only option for requests close together in time.
- **Make it relative** if typical pickup times grow, where a fixed number of minutes matters less: for example, a percentage of the fastest pickup time.

### Key prompt excerpt

The model gets this text, with `{max_extra}` filled in from `ride_config.py` and `{payload}` holding the request plus only the checked drivers. The reply is also held to a JSON schema whose `driver_id` must be one of the offered IDs.

```
You are dispatching one ride request. Pick exactly one driver from the list below.
The list contains only checked, available drivers within {max_extra} minutes of the fastest,
ranked by estimated road pickup minutes.
Use only these drivers. Judge by road pickup minutes, route notes, and status timestamps,
not by straight-line distance.

Input:
{payload}

Reply with JSON only: {"driver_id": "<one driver_id from the list>", "reason": "<one line>"}
```

### Requests with no available candidate

**RQ-0308-018** has no candidate rows at all. The snapshot's 4 rows for that slot are under `RQ-0309-018`, an ID that isn't in the requests file, which looks like a date typo. The agent returns this explicit no-match without calling the model:

```
{'status': 'no_match', 'request_id': 'RQ-0308-018', 'no_match_reason': 'no_candidate_drivers'}
```

Every other request has at least one available candidate. The `no_available_drivers` case (candidates exist, none available) is covered by `test_no_available_candidates_gives_explicit_no_match`, which marks every driver for one request offline and checks that the model isn't called.

## Known limitations

**Stale driver status.** `is_available` checks only that `driver_status` is `available`. It never compares `status_updated_at` with the request's `requested_at`. In the snapshot, many available statuses were last updated hours before the request, and many are timestamped after it, so a driver marked available may already be offline or on a trip, or the row may describe a later moment than the request. The prompt asks the model to consider status timestamps, but no code enforces a freshness limit.

## Data

- `data/Uber Ride Requests Test Set (week of 3-9 March 2025).csv`: 20 ride requests (pickup zone, coordinates, river side, bridge/disagreement flags).
- `data/Uber Candidate Drivers Snapshot (week of 3-9 March 2025).csv`: 80 candidate driver rows across the requests (status, position, straight-line km vs. road pickup minutes, route notes).
  - Known gap: the snapshot has 4 rows for `RQ-0309-018`, which isn't in the requests file, and none for `RQ-0308-018`. So `RQ-0308-018` returns `no_candidate_drivers`.

## Setup

1. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (e.g. `nvidia/nemotron-3.5-super-vl:free`). `.env` is gitignored, and both values are read only from those environment variables. An exported shell variable wins over `.env`.
3. `config.json` holds only the OpenRouter base URL.
4. Run: `.venv/bin/python ride_graph.py <request_id>`
5. Count snapshot rows that pass the availability check: `.venv/bin/python ride_graph.py --count-available` (currently 55 of 80).
