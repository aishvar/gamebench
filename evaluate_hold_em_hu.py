#!/usr/bin/env python3
"""evaluate_hold_em_hu.py ‑ GameBench Heads‑Up Hold'em evaluator

Tracks cumulative performance for every model/strategy found in
`data/heads_up_texas_hold_em_hands_log.json` and stores results in
`model_ratings_holdem_hu.json` so subsequent runs process only new hands.

As the true unique identifier of a hand is the per‑hand log filename
(`round_log_file`), we now de‑duplicate on that field (not on the
sometimes‑reused `timestamp`).

Output file schema
------------------
{
  "model_ratings": {
      "model‑name": {"hands": int, "net_chips": float},
      ...
  },
  "processed_round_files": ["heads_up_holdem_YYYYMMDD‑HHMMSS‑XXXXXX.log", ...]
}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

LOG_DEFAULT = Path("data/heads_up_texas_hold_em_hands_log.json")
OUT_DEFAULT = Path("model_ratings_holdem_hu.json")

# Models to ignore completely when computing ratings
IGNORE_MODELS: Set[str] = {"openai/o3-2025-04-16"}

Stat = Dict[str, float]        # {"hands": int, "net_chips": float}
Stats = Dict[str, Stat]        # {model: Stat}
KEY_PROCESSED = "processed_round_files"  # JSON key for seen identifiers

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_output(path: Path) -> Tuple[Stats, Set[str]]:
    if not path.exists():
        return defaultdict(lambda: {"hands": 0, "net_chips": 0.0}), set()
    with path.open("r", encoding="utf‑8") as fp:
        data = json.load(fp)
    ratings: Stats = defaultdict(lambda: {"hands": 0, "net_chips": 0.0},
                                 data.get("model_ratings", {}))
    # Gracefully accept the old key name to preserve compatibility
    seen = set(data.get(KEY_PROCESSED, data.get("processed_timestamps", [])))
    return ratings, seen


def save_output(path: Path, ratings: Stats, seen: Set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf‑8") as fp:
        json.dump({
            "model_ratings": ratings,
            KEY_PROCESSED: sorted(seen)
        }, fp, indent=2)
    tmp.replace(path)

# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process(log_path: Path, ratings: Stats, seen: Set[str]) -> None:
    """Update `ratings` and `seen` in place with new hands only."""
    with log_path.open("r", encoding="utf‑8") as fp:
        hands = json.load(fp)

    for hand in hands:
        uid: str = hand["round_log_file"]  # unique per hand
        if uid in seen:
            continue

        chips_won = float(hand["chips_won"])
        winner = hand["winner"]
        losers = hand["losers"] or []

        # Skip this hand entirely if any participant is in the ignore list
        if winner in IGNORE_MODELS or any(l in IGNORE_MODELS for l in losers):
            seen.add(uid)  # mark as processed so we don't revisit
            continue

        n_losers = max(len(losers), 1)

        # Winner stats
        w = ratings[winner]
        w["hands"] += 1
        w["net_chips"] += chips_won

        # Losers stats (equal share)
        loss = chips_won / n_losers
        for loser in losers:
            l = ratings[loser]
            l["hands"] += 1
            l["net_chips"] -= loss

        seen.add(uid)

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def chips_per_hand(stat: Stat) -> float:
    h = stat["hands"]
    return stat["net_chips"] / h if h else float("-inf")


def print_table(ratings: Stats) -> None:
    header = f"{'MODEL':<40}  {'HANDS':>7}  {'NET':>10}  {'CHIPS/HAND':>11}"
    print(header)
    print("‑" * len(header))
    for model, stat in sorted(ratings.items(), key=lambda kv: chips_per_hand(kv[1]), reverse=True):
        hands = int(stat["hands"])
        net = stat["net_chips"]
        cph = net / hands if hands else 0.0
        print(f"{model:<40}  {hands:>7}  {net:>10.1f}  {cph:>11.2f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate GameBench Hold'em HU logs and store cumulative model ratings.")
    ap.add_argument("--log", type=Path, default=LOG_DEFAULT, help="hands_log.json path")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT, help="output ratings JSON path")
    args = ap.parse_args()

    ratings, seen = load_output(args.out)
    process(args.log, ratings, seen)
    save_output(args.out, ratings, seen)
    print_table(ratings)


if __name__ == "__main__":
    main()