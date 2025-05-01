#!/usr/bin/env python3

import os
import json
import trueskill
from collections import defaultdict

LOG_PATH = os.path.join("data", "hands_log_multi_liars_poker.json")
RATINGS_PATH = "model_trueskill_ratings.json"

def canonical_name(model_name):
    if model_name is None:
        return None
    # Handles names like "openai/gpt-4o:floor" or just "gpt-4.1-mini-2025-04-14"
    return model_name.split(":", 1)[0]

def main():
    # Initialize TrueSkill environment
    # Set tau to 0 to prevent ratings from drifting too much over time without new games
    env = trueskill.TrueSkill(tau=0)

    # 1) Load existing ratings (if any)
    try:
        with open(RATINGS_PATH, "r") as f:
            stored = json.load(f)
        processed_timestamps = set(stored.get("processed_timestamps", []))
        # Deserialize Rating objects
        model_ratings = {
            name: env.create_rating(mu=mu, sigma=sigma)
            for name, (mu, sigma) in stored.get("model_ratings", {}).items()
        }
    except FileNotFoundError:
        model_ratings = {}
        processed_timestamps = set()
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {RATINGS_PATH}. Starting fresh.")
        model_ratings = {}
        processed_timestamps = set()


    # 2) Read the hands_log
    try:
        with open(LOG_PATH, "r") as f:
            rounds = json.load(f)
    except FileNotFoundError:
        print(f"Error: Log file not found at {LOG_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {LOG_PATH}")
        return

    new_timestamps = []

    # 3) Process each game
    for entry in rounds:
        ts = entry.get("timestamp")
        if not ts:
            print(f"Warning: Entry missing timestamp: {entry}. Skipping.")
            continue

        if ts in processed_timestamps:
            continue

        # Extract ranked models
        ranked_models_raw = []
        i = 1
        while f"rank{i}" in entry:
            ranked_models_raw.append(entry[f"rank{i}"])
            i += 1

        if not ranked_models_raw:
            print(f"Warning: Entry missing ranks: {entry}. Skipping.")
            new_timestamps.append(ts) # Mark as processed even if skipped
            continue

        ranked_models = [canonical_name(m) for m in ranked_models_raw]

        # Ensure ratings exist for all models in the current game
        current_game_ratings = []
        for model_name in ranked_models:
            if model_name not in model_ratings:
                model_ratings[model_name] = env.create_rating()
            current_game_ratings.append(model_ratings[model_name])

        # Prepare rating groups for TrueSkill (each player is their own group)
        rating_groups = [(r,) for r in current_game_ratings]

        # Update ratings using TrueSkill
        try:
            updated_rating_groups = env.rate(rating_groups)
        except FloatingPointError as e:
             print(f"Warning: FloatingPointError during rating update for timestamp {ts}. Skipping update. Error: {e}")
             new_timestamps.append(ts) # Mark as processed even if skipped
             continue


        # Store updated ratings back into the main dictionary
        for i, model_name in enumerate(ranked_models):
            model_ratings[model_name] = updated_rating_groups[i][0]

        new_timestamps.append(ts)

    # 4) Save updated ratings
    processed_timestamps.update(new_timestamps)
    # Serialize Rating objects for JSON
    serializable_ratings = {
        name: {"mu": r.mu, "sigma": r.sigma} for name, r in model_ratings.items()
    }
    try:
        with open(RATINGS_PATH, "w") as f:
            json.dump({
                "model_ratings": serializable_ratings,
                "processed_timestamps": sorted(list(processed_timestamps))
            }, f, indent=2)
    except IOError as e:
        print(f"Error: Could not write ratings to {RATINGS_PATH}. Error: {e}")


    # 5) Print models by highest exposure (mu - k * sigma)
    # k = env.mu / env.sigma # This is how expose is calculated internally, but we can just use the method
    sorted_models = sorted(
        model_ratings.items(),
        key=lambda item: env.expose(item[1]),
        reverse=True
    )

    print(f"{'Model':<50} {'Mu':<10} {'Sigma':<10} {'Exposure':<10}")
    print("-" * 80)
    for model, r in sorted_models:
        exposure = env.expose(r)
        print(f"{model:<50} {r.mu:<10.2f} {r.sigma:<10.2f} {exposure:<10.2f}")

if __name__ == "__main__":
    main()
