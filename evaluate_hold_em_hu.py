#!/usr/bin/env python3
"""
Summarise heads‑up holdem results per model, including
a paired‑data SE that exploits the within‑pair correlation ρ.

Usage:
    python evaluate_hold_em_hu.py
"""

import json
import math
import os
import statistics
from collections import defaultdict
from typing import List, Tuple

# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join("data", "heads_up_texas_hold_em_hands_log.json")
OUT_FILE   = "model_ratings_holdem_hu.json"        # saved next to the script

# ---------------------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    hands_log = json.load(f)

stats: dict[str, dict[str, any]]                   # per‑hand deltas
stats = {}

pairs: defaultdict[str, List[Tuple[int, int]]]     # (r1, r2) for each model
pairs = defaultdict(list)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def update(model: str, delta: int) -> None:
    rec = stats.setdefault(model, {"hands": 0, "deltas": []})
    rec["hands"] += 1
    rec["deltas"].append(delta)


def _corr(xs: List[int], ys: List[int]) -> float:
    """
    Fallback correlation calculation for Python versions < 3.10,
    equivalent to statistics.correlation.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sd_x   = statistics.stdev(xs)
    sd_y   = statistics.stdev(ys)
    if sd_x == 0 or sd_y == 0:
        return 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sd_x * sd_y)

# ----------------------------------------------------------------------------
# pass 1 – collect per‑hand deltas *and* per‑pair tuples
# ----------------------------------------------------------------------------

for pair_entry in hands_log:                       # one paired round
    h1, h2 = pair_entry["sub_hands"]               # exactly two sub‑hands

    c1, c2 = h1["chips_won"], h2["chips_won"]
    w1, l1 = h1["winner"],  h1["losers"][0]
    w2, l2 = h2["winner"],  h2["losers"][0]

    # update per‑hand stats
    update(w1,  c1); update(l1, -c1)
    update(w2,  c2); update(l2, -c2)

    # build (r1, r2) for each of the two models in the pair
    for m in (w1, l1):                            # the same two models
        r1 =  c1 if m == w1 else -c1
        r2 =  c2 if m == w2 else -c2
        pairs[m].append((r1, r2))

# ----------------------------------------------------------------------------
# pass 2 – final summary with paired SE
# ----------------------------------------------------------------------------

summary = {}
for model, rec in stats.items():
    hands   = rec["hands"]                       # total sub‑hands (N)
    deltas  = rec["deltas"]
    net     = sum(deltas)
    mean    = net / hands
    std_dev = statistics.stdev(deltas) if hands > 1 else 0.0

    # paired correlation ρ
    r1r2 = pairs[model]
    if len(r1r2) > 1:
        r1s, r2s = zip(*r1r2)
        rho = _corr(list(r1s), list(r2s))
        rho = max(-1.0, min(1.0, rho))           # numeric safety
    else:
        rho = 0.0

    # SE of mean per hand, adjusted for pairing
    se_paired = std_dev / math.sqrt(hands) * math.sqrt(1 + rho)

    summary[model] = {
        "hands"     : hands,
        "net_chips" : net,
        "mean"      : mean,
        "std_dev"   : std_dev,
        "rho"       : rho,
        "se_paired" : se_paired,
    }

with open(OUT_FILE, "w", encoding="utf-8") as out_f:
    json.dump(summary, out_f, indent=2)

print(f"✅  Wrote summary with paired SE to {OUT_FILE}")