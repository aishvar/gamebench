#!/usr/bin/env python3

import os
import json
from trueskill_minimal import Rating, rate

LOG_PATH = os.path.join("data", "hands_log.json")
RATINGS_PATH = "model_ratings.json"

def main():
    # 1) Load existing ratings (if any)
    try:
        with open(RATINGS_PATH, "r") as f:
            # { "google/gemini-2.5-pro-exp-03-25:free": {"mu": 25.0, "sigma": 8.3333}, ...}
            stored = json.load(f)
        model_ratings = {
            model: Rating(mu=vals["mu"], sigma=vals["sigma"])
            for model, vals in stored.items()
        }
    except FileNotFoundError:
        model_ratings = {}

    # 2) Read the hands_log
    with open(LOG_PATH, "r") as f:
        rounds = json.load(f)

    # 3) Process each game
    for entry in rounds:
        winner_model = entry["winner"]
        loser_models = entry["losers"]

        # Make sure all involved models have a rating object
        if winner_model not in model_ratings:
            model_ratings[winner_model] = Rating()
        for lm in loser_models:
            if lm not in model_ratings:
                model_ratings[lm] = Rating()

        # Build input to `rate`. 
        # TrueSkill expects lists of Ratings, plus a parallel list of ranks 
        # (rank=0 for the winner, rank=1 for each loser).
        # So the ordering is: [winner, loser1, loser2, ...],
        # with ranks=[0,1,1,...] if they all lost equally.
        all_models = [winner_model] + loser_models
        current_ratings = [model_ratings[m] for m in all_models]
        # Instead of [0, 1, 1, ...], do [1, 0, 0, ...]
        ranks = [1] + [0]*len(loser_models)

        updated = rate(current_ratings, ranks=ranks)
        
        # Store updated ratings back
        for m, new_r in zip(all_models, updated):
            model_ratings[m] = new_r

    # 4) Save updated ratings
    with open(RATINGS_PATH, "w") as f:
        json.dump(
            {m: {"mu": r.mu, "sigma": r.sigma} for m, r in model_ratings.items()},
            f,
            indent=2
        )

    # Print top models by (mu - 3*sigma) if you want
    sorted_models = sorted(
        model_ratings.items(),
        key=lambda kv: (kv[1].mu - 3*kv[1].sigma),
        reverse=True
    )
    for model, r in sorted_models:
        skill = r.mu - 3*r.sigma
        print(f"{model:<50} mu={r.mu:.2f}, sigma={r.sigma:.2f}, skill≈{skill:.2f}")

if __name__ == "__main__":
    main()