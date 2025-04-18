#!/usr/bin/env python3
"""
Summarise heads‑up holdem results per model.

Usage:
    python summarise_holdem.py

Reads:
    data/heads_up_texas_hold_em_hands_log.json

Writes:
    model_ratings_holdem_hu.json    # saved in the current working directory
"""
import json
import os
import statistics

# ---------------------------------------------------------------------------

INPUT_FILE = os.path.join("data", "heads_up_texas_hold_em_hands_log.json")
OUT_FILE   = "model_ratings_holdem_hu.json"      # ⬅️  now saved alongside the script

# ---------------------------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    hands_log = json.load(f)

stats: dict[str, dict[str, any]] = {}

def update(model: str, delta: int) -> None:
    record = stats.setdefault(model, {"hands": 0, "deltas": []})
    record["hands"] += 1
    record["deltas"].append(delta)

for pair in hands_log:                              # paired round
    for hand in pair["sub_hands"]:                  # each half‑hand
        chips = hand["chips_won"]
        winner = hand["winner"]
        update(winner, chips)                       # winner gains
        for loser in hand["losers"]:
            update(loser, -chips)                  # losers lose

# ---------------------------------------------------------------------------

summary = {}
for model, record in stats.items():
    hands = record["hands"]
    deltas = record["deltas"]
    net = sum(deltas)
    mean = net / hands
    std = statistics.pstdev(deltas) if hands > 1 else 0.0
    summary[model] = {
        "hands": hands,
        "net_chips": net,
        "mean": mean,
        "std_dev": std,
    }

with open(OUT_FILE, "w", encoding="utf-8") as out_f:
    json.dump(summary, out_f, indent=2)

print(f"✅  Wrote summary to {OUT_FILE}")