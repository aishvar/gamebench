#!/usr/bin/env python3

#use this for 2-player games

import os
import json

LOG_PATH = os.path.join("data", "hands_log.json")
RATINGS_PATH = "model_ratings.json"

def canonical_name(model_name):
    if model_name is None:
        return None
    return model_name.split(":", 1)[0]

def elo_expected(ratingA, ratingB):
    """Returns expected score for A vs. B using standard 400-point spacing."""
    return 1.0 / (1.0 + 10.0 ** ((ratingB - ratingA) / 400.0))

def elo_update(ratingA, ratingB, scoreA, k=20):
    """
    ratingA, ratingB: current Elo ratings for A and B.
    scoreA: actual score for A (1 if A wins, 0 if A loses).
    Returns the updated (ratingA, ratingB).
    """
    expectedA = elo_expected(ratingA, ratingB)
    expectedB = 1.0 - expectedA
    newA = ratingA + k * (scoreA - expectedA)
    newB = ratingB + k * ((1.0 - scoreA) - expectedB)
    return newA, newB

def main():
    # 1) Load existing ratings (if any)
    try:
        with open(RATINGS_PATH, "r") as f:
            stored = json.load(f)
        processed_timestamps = set(stored.get("processed_timestamps", []))
        model_ratings = stored.get("model_ratings", {})
    except FileNotFoundError:
        model_ratings = {}
        processed_timestamps = set()

    # 2) Read the hands_log
    with open(LOG_PATH, "r") as f:
        rounds = json.load(f)

    new_timestamps = []

    # 3) Process each game
    for entry in rounds:
        ts = entry["timestamp"]
        if ts in processed_timestamps:
            continue

        winner = canonical_name(entry["winner"])
        losers = [canonical_name(l) for l in entry["losers"]]

        # If there's no winner, skip rating updates
        if winner is None:
            new_timestamps.append(ts)
            continue

        # Ensure ratings exist
        if winner not in model_ratings:
            model_ratings[winner] = 1000.0
        for lm in losers:
            if lm not in model_ratings:
                model_ratings[lm] = 1000.0

        # Perform pairwise Elo updates for each loser
        for lm in losers:
            w_rating = model_ratings[winner]
            l_rating = model_ratings[lm]
            # Winner gets 1.0, loser gets 0.0
            new_w, new_l = elo_update(w_rating, l_rating, 1.0, k=20)
            model_ratings[winner] = new_w
            model_ratings[lm] = new_l

        new_timestamps.append(ts)

    # 4) Save updated ratings
    processed_timestamps.update(new_timestamps)
    with open(RATINGS_PATH, "w") as f:
        json.dump({
            "model_ratings": model_ratings,
            "processed_timestamps": sorted(list(processed_timestamps))
        }, f, indent=2)

    # Print models by highest rating
    sorted_models = sorted(model_ratings.items(), key=lambda kv: kv[1], reverse=True)
    for model, r in sorted_models:
        print(f"{model:<50} rating={r:.2f}")

if __name__ == "__main__":
    main()