#!/usr/bin/env python3
"""GameBench Heads‑Up Poker Hand Log Evaluator

Reads `heads_up_texas_hold_em_hands_log.json`, updates an incremental
state file so it never double‑counts hands it has already processed, and
prints a summary table showing, for each model/strategy:
    • total hands played
    • net chips won (positive) or lost (negative)

Usage
-----
$ python evaluate_hold_em_hu.py                                   # default paths
$ python evaluate_hold_em_hu.py --log my_log.json                 # custom log
$ python evaluate_hold_em_hu.py --state evaluator_state.json      # custom state

Assumptions
-----------
‣ Each JSON object corresponds to one completed hand.
‣ `winner` gained `chips_won` chips from the losers combined.
‣ Losses are split equally among the `losers` list so the evaluator is
  chip‑sum‑zero per hand (important if there is ever >1 loser).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

DEFAULT_LOG = Path("data/heads_up_texas_hold_em_hands_log.json")
DEFAULT_STATE = Path("evaluator_state.json")

StatDict = Dict[str, Dict[str, float]]  # {model: {"hands": int, "chips": float}}


def load_state(path: Path) -> Set[str]:
    """Return the set of timestamps that have already been processed."""
    if path.exists():
        try:
            with path.open("r", encoding="utf‑8") as fp:
                return set(json.load(fp))
        except Exception:
            print(f"[WARN] Corrupt state file '{path}', rebuilding …")
    return set()


def save_state(path: Path, timestamps: Set[str]) -> None:
    """Write the processed‑timestamp set back to disk."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf‑8") as fp:
        json.dump(sorted(timestamps), fp, indent=2)
    tmp.replace(path)


def process_log(log_path: Path, seen: Set[str]) -> StatDict:
    """Parse the hand log and accumulate stats for NEW hands only."""
    stats: StatDict = defaultdict(lambda: {"hands": 0, "chips": 0.0})

    with log_path.open("r", encoding="utf‑8") as fp:
        hands = json.load(fp)

    for hand in hands:
        ts: str = hand["timestamp"]
        if ts in seen:
            continue  # already accounted for

        chips_won = float(hand["chips_won"])
        winner = hand["winner"]
        losers = hand["losers"]
        n_losers = max(len(losers), 1)

        # Update winner
        stats[winner]["hands"] += 1
        stats[winner]["chips"] += chips_won

        # Update losers (equal split)
        per_loser = chips_won / n_losers
        for loser in losers:
            stats[loser]["hands"] += 1
            stats[loser]["chips"] -= per_loser

        seen.add(ts)

    return stats


def merge_stats(a: StatDict, b: StatDict) -> StatDict:
    """Combine two StatDict objects in‑place and return the merged dict."""
    for model, sb in b.items():
        sa = a[model]
        sa["hands"] += sb["hands"]
        sa["chips"] += sb["chips"]
    return a


def format_table(stats: StatDict) -> str:
    """Pretty‑print a summary table sorted by net chips descending."""
    header = f"{'MODEL':<40}  {'HANDS':>8}  {'NET_CHIPS':>10}"
    rows = [header, "‑" * len(header)]
    for model, s in sorted(stats.items(), key=lambda kv: kv[1]["chips"], reverse=True):
        rows.append(f"{model:<40}  {int(s['hands']):>8}  {s['chips']:>10.1f}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GameBench hand logs.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Path to hands_log.json file")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Path to evaluator state file")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"[ERROR] Log file '{args.log}' not found.")

    seen = load_state(args.state)
    stats = process_log(args.log, seen)
    # If nothing new processed, still show cumulative stats (from previous runs)
    cumulative: StatDict = defaultdict(lambda: {"hands": 0, "chips": 0.0})
    # read previous cumulative stats if exist
    prev_stats_path = args.state.with_suffix(".stats.json")
    if prev_stats_path.exists():
        with prev_stats_path.open("r", encoding="utf‑8") as fp:
            prev_stats: StatDict = json.load(fp)
        merge_stats(cumulative, prev_stats)
    merge_stats(cumulative, stats)

    print(format_table(cumulative))

    # Persist state and cumulative stats
    save_state(args.state, seen)
    with prev_stats_path.open("w", encoding="utf‑8") as fp:
        json.dump(cumulative, fp, indent=2)


if __name__ == "__main__":
    main()
