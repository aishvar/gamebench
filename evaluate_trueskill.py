#!/usr/bin/env python3

import os
import json
from trueskill_minimal import Rating, rate

#use this for multiple players

#LOG_PATH = os.path.join("data", "hands_log.json") #change 
#RATINGS_PATH = "model_ratings.json" #change

def canonical_name(model_name):
    # Handle cases where model_name might be None
    if model_name is None:
        return None
    return model_name.split(":", 1)[0]

def main():
    # 1) Load existing ratings (if any)
    try:
        with open(RATINGS_PATH, "r") as f:
            stored = json.load(f)
        processed_timestamps = set(stored.get("processed_timestamps", []))
        model_ratings_data = stored.get("model_ratings", {})
        model_ratings = {}
        # Convert stored ratings into Rating objects, ensuring canonical naming
        for m, vals in model_ratings_data.items():
            c = canonical_name(m)
            model_ratings[c] = Rating(mu=vals["mu"], sigma=vals["sigma"])
    except FileNotFoundError:
        model_ratings = {}
        processed_timestamps = set()

    # 2) Read the hands_log
    with open(LOG_PATH, "r") as f:
        rounds = json.load(f)

    # 3) Process each game
    new_timestamps = []
    for entry in rounds:
        ts = entry["timestamp"]
        if ts in processed_timestamps:
            continue

        winner = entry["winner"]
        # If there's no winner, skip rating logic but mark as processed
        if winner is None:
            new_timestamps.append(ts)
            continue

        winner_model = canonical_name(winner)
        loser_models = [canonical_name(l) for l in entry["losers"]]

        # Ensure rating objects exist for the involved models
        if winner_model not in model_ratings:
            model_ratings[winner_model] = Rating()
        for lm in loser_models:
            if lm not in model_ratings:
                model_ratings[lm] = Rating()

        # Build input to `rate`
        all_models = [winner_model] + loser_models
        current_ratings = [model_ratings[m] for m in all_models]
        ranks = [1] + [0]*len(loser_models)

        updated = rate(current_ratings, ranks=ranks)

        # Store updated ratings back
        for m, new_r in zip(all_models, updated):
            model_ratings[m] = new_r

        new_timestamps.append(ts)

    # 4) Save updated ratings
    processed_timestamps.update(new_timestamps)
    with open(RATINGS_PATH, "w") as f:
        json.dump({
            "model_ratings": {
                m: {"mu": r.mu, "sigma": r.sigma}
                for m, r in model_ratings.items()
            },
            "processed_timestamps": sorted(list(processed_timestamps))
        }, f, indent=2)

    # Print top models by mu
    sorted_models = sorted(
        model_ratings.items(),
        key=lambda kv: kv[1].mu,
        reverse=True
    )
    for model, r in sorted_models:
        print(f"{model:<50} mu={r.mu:.2f}, sigma={r.sigma:.2f}")

if __name__ == "__main__":
    main()