#!/usr/bin/env python3
"""evaluate_hold_em_hu.py ‑ GameBench Heads‑Up Hold'em evaluator

Tallies, for every model/strategy appearing in
`data/heads_up_texas_hold_em_hands_log.json`, the cumulative number of
hands played and the net chips won/lost.  Results and the set of already
processed timestamps are stored together in
`model_ratings_holdem_hu.json` so repeat invocations only consider new
hands.

Output file schema
------------------
{
  "model_ratings": {
      "model‑name": {"hands": int, "net_chips": float},
      ...
  },
  "processed_timestamps": ["YYYYMMDD‑HHMMSS", ...]
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

# helpers --------------------------------------------------------------------

Stat = Dict[str, float]                 # {"hands": int, "net_chips": float}
Stats = Dict[str, Stat]                 # {model: Stat}


def load_output(path: Path) -> Tuple[Stats, Set[str]]:
    if not path.exists():
        return defaultdict(lambda: {"hands": 0, "net_chips": 0.0}), set()
    with path.open("r", encoding="utf‑8") as fp:
        data = json.load(fp)
    ratings = defaultdict(lambda: {"hands": 0, "net_chips": 0.0}, data.get("model_ratings", {}))
    seen = set(data.get("processed_timestamps", []))
    return ratings, seen


def save_output(path: Path, ratings: Stats, seen: Set[str]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf‑8") as fp:
        json.dump({
            "model_ratings": ratings,
            "processed_timestamps": sorted(seen)
        }, fp, indent=2)
    tmp.replace(path)


def process(log_path: Path, ratings: Stats, seen: Set[str]) -> None:
    """Update `ratings` and `seen` in place with new hands only."""
    with log_path.open("r", encoding="utf‑8") as fp:
        hands = json.load(fp)

    for hand in hands:
        ts: str = hand["timestamp"]
        if ts in seen:
            continue

        chips_won = float(hand["chips_won"])
        winner = hand["winner"]
        losers = hand["losers"] or []
        n_losers = max(len(losers), 1)

        # Winner update
        w = ratings[winner]
        w["hands"] += 1
        w["net_chips"] += chips_won

        # Losers update (equal share of loss)
        per_loser = chips_won / n_losers
        for loser in losers:
            l = ratings[loser]
            l["hands"] += 1
            l["net_chips"] -= per_loser

        seen.add(ts)


def print_table(ratings: Stats) -> None:
    header = f"{'MODEL':<40}  {'HANDS':>7}  {'NET':>10}"
    print(header)
    print("‑" * len(header))
    for model, stat in sorted(ratings.items(), key=lambda kv: kv[1]["net_chips"], reverse=True):
        print(f"{model:<40}  {int(stat['hands']):>7}  {stat['net_chips']:>10.1f}")


# entry ----------------------------------------------------------------------

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
