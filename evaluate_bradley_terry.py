#!/usr/bin/env python3

# Bradley-Terry ratings for 2-player games, fit on all hands at once.
# Unlike online Elo, the result does not depend on the order games were played in.
# Ratings are put on the Elo scale (400 points = 10x odds) and centred at 1000.

import os
import json
import numpy as np

LOG_PATH = os.path.join("data", "hands_log.json")
RATINGS_PATH = "model_ratings_bt.json"

ELO_SCALE = 400.0 / np.log(10.0)  # logit units -> Elo points
ELO_CENTER = 1000.0
PRIOR_GAMES = 1.0  # virtual games (half win, half loss) vs an average opponent; keeps unbeaten/winless models finite
BOOTSTRAP_SAMPLES = 500
SEED = 0

def canonical_name(model_name):
    if model_name is None:
        return None
    return model_name.split(":", 1)[0]

def load_games():
    with open(LOG_PATH, "r") as f:
        rounds = json.load(f)

    games = []  # (winner, loser)
    for entry in rounds:
        winner = canonical_name(entry["winner"])
        losers = [canonical_name(l) for l in entry["losers"]]
        # Same rules as the Elo script: skip rounds without a winner and self-matches
        if winner is None or winner in losers:
            continue
        for lm in losers:
            games.append((winner, lm))
    return games, len(rounds)

def fit_bt(winners, losers, n_players, iters=2000, tol=1e-10):
    """MM algorithm (Hunter 2004). Returns log-strengths, mean-centred."""
    wins = np.bincount(winners, minlength=n_players) + PRIOR_GAMES / 2
    p = np.ones(n_players)
    for _ in range(iters):
        inv = 1.0 / (p[winners] + p[losers])
        denom = np.bincount(winners, weights=inv, minlength=n_players)
        denom += np.bincount(losers, weights=inv, minlength=n_players)
        denom += PRIOR_GAMES / (p + 1.0)  # virtual games vs strength-1 opponent
        new_p = wins / denom
        new_p /= np.exp(np.mean(np.log(new_p)))
        if np.max(np.abs(np.log(new_p) - np.log(p))) < tol:
            p = new_p
            break
        p = new_p
    log_p = np.log(p)
    return log_p - log_p.mean()

def main():
    games, num_rounds = load_games()
    names = sorted({m for g in games for m in g})
    idx = {m: i for i, m in enumerate(names)}
    winners = np.array([idx[w] for w, _ in games])
    losers = np.array([idx[l] for _, l in games])

    ratings = ELO_CENTER + ELO_SCALE * fit_bt(winners, losers, len(names))

    # Bootstrap over games for 95% intervals
    rng = np.random.default_rng(SEED)
    samples = np.empty((BOOTSTRAP_SAMPLES, len(names)))
    for b in range(BOOTSTRAP_SAMPLES):
        pick = rng.integers(0, len(games), len(games))
        samples[b] = ELO_CENTER + ELO_SCALE * fit_bt(winners[pick], losers[pick], len(names))
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)

    with open(RATINGS_PATH, "w") as f:
        json.dump({
            "model_ratings": {m: float(ratings[i]) for i, m in enumerate(names)},
            "ci95": {m: [float(lo[i]), float(hi[i])] for i, m in enumerate(names)},
            "num_rounds": num_rounds,
            "num_games": len(games),
        }, f, indent=2)

    order = np.argsort(-ratings)
    for i in order:
        print(f"{names[i]:<50} rating={ratings[i]:.1f}  95% CI [{lo[i]:.0f}, {hi[i]:.0f}]")

if __name__ == "__main__":
    main()
