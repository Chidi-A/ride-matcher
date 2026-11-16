"""Matching rules shared by the agent (ride_graph.py) and the tests.

Change values here only; nothing else should hard-code them.
"""

# A driver is only offered to the model (and only ever assigned) if their road
# pickup time is at most this many minutes slower than the fastest available
# candidate for the same request.
MAX_EXTRA_PICKUP_MINUTES = 5
