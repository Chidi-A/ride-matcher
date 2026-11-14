# ride-matcher

Matching ride requests to candidate drivers.

## Data

- `data/Uber Ride Requests Test Set (week of 3-9 March 2025).csv`: 20 ride requests (pickup zone, coordinates, river side, bridge/disagreement flags).
- `data/Uber Candidate Drivers Snapshot (week of 3-9 March 2025).csv`: 80 candidate driver rows across the requests (status, position, straight-line km vs. road pickup minutes, route notes).

## Setup

1. Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY`. `.env` is gitignored; the key is read only from that environment variable.
2. The OpenRouter model id lives in `config.json` (`openrouter_model`).
3. Test the connection: `python3 openrouter_client.py`
