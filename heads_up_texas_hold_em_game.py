#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
import time
import os
import json
import re
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import defaultdict
from dataclasses import dataclass, field
import fcntl
from datetime import datetime

try:
    from llm_client import LLMClient, parse_response_text
except ImportError:
    print("Error: llm_client.py not found. Cannot run LLM-based strategies.")
    raise SystemExit

# ----------------------------------------------------------------------------
# LOGGING AND STORAGE SETUP
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(DATA_DIR, "heads_up_texas_hold_em_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

HAND_HISTORY_JSON = os.path.join(DATA_DIR, "heads_up_texas_hold_em_hands_log.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeadsUpTexasHoldEm")

# ----------------------------------------------------------------------------
# CARD FUNCTIONS
# ----------------------------------------------------------------------------

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["♣", "♦", "♥", "♠"]

def card_str(card: int) -> str:
    rank = card % 13
    suit = card // 13
    return RANK_NAMES[rank] + SUIT_NAMES[suit]

def shuffle_deck() -> List[int]:
    deck = list(range(52))
    random.shuffle(deck)
    return deck

# ----------------------------------------------------------------------------
# COMMON LLM CONFIGS (from liars_poker_game.py)
# ----------------------------------------------------------------------------

COMMON_CONFIGS = [
    # === OpenAI ===
    {"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-2024-11-20"},
    {"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-2025-04-14"},
    {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
    {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-nano-2025-04-14"},
    {"strategy_type": "llm", "provider": "openai", "model": "o3-mini-2025-01-31"},

    # === Anthropic ===
    {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},

    # === DeepSeek ===
    {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-chat-v3-0324:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-qwen-32b:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-llama-70b:floor"},

    # === LLaMA ===
    {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-maverick:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-scout:floor"},

    # === Google ===
    {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-pro-preview-03-25:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemma-3-27b-it:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.0-flash-001:floor"},

    # === Miscellaneous ===
    {"strategy_type": "llm", "provider": "openrouter", "model": "cohere/command-a:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "x-ai/grok-3-beta:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "mistralai/mistral-small-3.1-24b-instruct:floor"},
    {"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwq-32b:nitro"},
]

# ----------------------------------------------------------------------------
# PLAYER CLASS
# ----------------------------------------------------------------------------

@dataclass
class Player:
    player_id: int
    strategy_type: str  # "naive", "llm", or "random"
    model_config: Optional[Dict[str, str]] = None
    client: Optional[LLMClient] = field(default=None, init=False)
    original_order: int = 0
    effective_strategy: Optional[str] = None
    effective_model_config: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.strategy_type == "llm":
            if not self.model_config:
                raise ValueError("Missing model config for LLM player.")
            self.client = LLMClient(
                provider=self.model_config["provider"],
                model=self.model_config["model"],
                max_tokens=1024,
                temperature=0.5,
                max_retries=2,
                timeout=60
            )
        elif self.strategy_type in ("naive", "random"):
            pass
        else:
            raise ValueError(f"Unrecognized strategy '{self.strategy_type}'.")

    def get_display_name(self) -> str:
        if self.strategy_type == "naive":
            return "Naive All-In"
        elif self.strategy_type == "llm":
            p = self.model_config.get("provider", "?")
            m = self.model_config.get("model", "?")
            return f"{p}/{m}"
        elif self.strategy_type == "random":
            if not self.effective_strategy:
                return "Random(undecided)"
            if self.effective_strategy == "naive":
                return "Random(Naive All-In)"
            if self.effective_strategy == "llm":
                if self.effective_model_config:
                    p = self.effective_model_config.get("provider", "?")
                    m = self.effective_model_config.get("model", "?")
                    return f"Random({p}/{m})"
                return "Random(LLM??)"
            return f"Random({self.effective_strategy})"
        else:
            return "Unknown"

# ----------------------------------------------------------------------------
# HEADS-UP TEXAS HOLD'EM GAME CLASS
# ----------------------------------------------------------------------------

class HeadsUpTexasHoldEmGame:
    STARTING_STACK = 100
    SMALL_BLIND = 1
    BIG_BLIND = 2

    def __init__(self, player_configs: List[Dict[str, Any]]):
        if len(player_configs) != 2:
            raise ValueError("Heads-up game requires exactly 2 players.")
        self.players: List[Player] = []
        for i, cfg in enumerate(player_configs):
            st = cfg.get("strategy_type", "naive")
            if st == "llm":
                pl = Player(
                    player_id=i,
                    strategy_type="llm",
                    model_config={"provider": cfg["provider"], "model": cfg["model"]},
                    original_order=i,
                )
            elif st == "naive":
                pl = Player(player_id=i, strategy_type="naive", original_order=i)
            elif st == "random":
                pl = Player(player_id=i, strategy_type="random", original_order=i)
            else:
                raise ValueError(f"Bad strategy {st} for player {i}")
            self.players.append(pl)
        random.shuffle(self.players)
        self.button_index = 0
        self.round_log: List[str] = []
        self.game_active = False

    # --- Utility Methods ---

    def _log(self, msg: str):
        logger.info(msg)
        self.round_log.append(msg)

    def _opponent_idx(self, idx: int) -> int:
        return 1 - idx

    # --- Card Dealing ---

    def _deal_hole_cards(self, deck: List[int]) -> Tuple[List[int], List[int]]:
        return deck[:2], deck[2:4]

    def _get_community_cards(self, deck: List[int]) -> Tuple[List[int], List[int], List[int]]:
        flop = deck[4:7]
        turn = [deck[7]]
        river = [deck[8]]
        return flop, turn, river

    def _describe_cards(self, cards: List[int]) -> str:
        return " ".join(card_str(c) for c in cards)

    # --- Hand Evaluation (Simplified) ---
    def rank_7card_hand(self, seven: List[int]) -> Tuple[int, List[int]]:
        ranks = [c % 13 for c in seven]
        suits = [c // 13 for c in seven]
        rank_counts = defaultdict(int)
        for r in ranks:
            rank_counts[r] += 1
        sorted_counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        suit_counts = defaultdict(int)
        for s in suits:
            suit_counts[s] += 1
        flush_suit = None
        for s, ct in suit_counts.items():
            if ct >= 5:
                flush_suit = s
                break
        unique_r = sorted(set(ranks))
        if 12 in unique_r:
            unique_r.append(-1)
        longest = 1
        best_high = unique_r[0]
        cur_len = 1
        for i in range(1, len(unique_r)):
            if unique_r[i] == unique_r[i - 1] + 1:
                cur_len += 1
                if cur_len > longest:
                    longest = cur_len
                    best_high = unique_r[i]
            else:
                cur_len = 1
        has_straight = (longest >= 5)
        if flush_suit is not None:
            flush_cards = sorted([r for r, s in zip(ranks, suits) if s == flush_suit])
            if 12 in flush_cards:
                flush_cards.append(-1)
            sf_len = 1
            sf_hi = flush_cards[0]
            best_sf_len = 1
            for i in range(1, len(flush_cards)):
                if flush_cards[i] == flush_cards[i - 1] + 1:
                    sf_len += 1
                    if sf_len > best_sf_len:
                        best_sf_len = sf_len
                        sf_hi = flush_cards[i]
                elif flush_cards[i] == flush_cards[i - 1]:
                    continue
                else:
                    sf_len = 1
            if best_sf_len >= 5:
                return (8, [sf_hi])
        top_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        if top_count == 4:
            qr = sorted_counts[0][0]
            kicker = max(r for r in ranks if r != qr)
            return (7, [qr, kicker])
        if top_count == 3 and second_count >= 2:
            tr = sorted_counts[0][0]
            pr = sorted_counts[1][0]
            return (6, [tr, pr])
        if flush_suit is not None:
            flush_cards = sorted([r for r, s in zip(ranks, suits) if s == flush_suit], reverse=True)
            return (5, flush_cards[:5])
        if has_straight:
            return (4, [best_high])
        if top_count == 3:
            t = sorted_counts[0][0]
            kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
            return (3, [t] + kickers)
        if top_count == 2:
            p = sorted_counts[0][0]
            if second_count == 2:
                p2 = sorted_counts[1][0]
                kicker = max(r for r in ranks if r not in (p, p2))
                pair_sorted = sorted([p, p2], reverse=True)
                return (2, pair_sorted + [kicker])
            else:
                kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
                return (1, [p] + kickers)
        top5 = sorted(ranks, reverse=True)[:5]
        return (0, top5)

    def compare_final_hands(self, cardsA: List[int], cardsB: List[int], board: List[int]) -> int:
        bestA = self.rank_7card_hand(cardsA + board)
        bestB = self.rank_7card_hand(cardsB + board)
        if bestA[0] > bestB[0]:
            return +1
        elif bestA[0] < bestB[0]:
            return -1
        else:
            for a, b in zip(bestA[1], bestB[1]):
                if a > b:
                    return +1
                elif a < b:
                    return -1
            return 0

    # --- Betting Logic (with Standard Min-Raise, and Forced Action Each Round) ---

    def _init_chips(self):
        self.stacks = [self.STARTING_STACK, self.STARTING_STACK]
        self.pot = 0
        self.current_bet = 0
        self.bets = [0, 0]
        self.last_raise_size = self.BIG_BLIND

    def _take_chips(self, pidx: int, amt: int):
        if amt > self.stacks[pidx]:
            amt = self.stacks[pidx]
        self.stacks[pidx] -= amt
        self.pot += amt

    def _post_blinds(self):
        sb = self.button_index
        bb = self._opponent_idx(sb)
        self._take_chips(sb, self.SMALL_BLIND)
        self.bets[sb] = self.SMALL_BLIND
        self._take_chips(bb, self.BIG_BLIND)
        self.bets[bb] = self.BIG_BLIND
        self.current_bet = self.BIG_BLIND

    def _settle_bets(self):
        self.bets = [0, 0]
        self.current_bet = 0
        # last_raise_size carries over

    def _betting_round(self, street_name: str) -> bool:
        """
        Force both players to act in the betting round.

        ‑ On any street *except* Pre‑flop, a voluntary BET / RAISE followed by a
          CALL ends the round immediately.
        ‑ On Pre‑flop we still need the big blind’s option, so the “bet‑was‑called
          → end round” rule triggers only after a voluntary raise.
        """
        self._log(f"--- {street_name} betting round ---")
        next_to_act = self.button_index if street_name == "Preflop" else self._opponent_idx(self.button_index)

        consecutive_checks = 0          # for check‑check detection
        round_has_voluntary_bet = False # becomes True after first BET / RAISE

        while True:
            if self.stacks[next_to_act] == 0:                       # player already all‑in
                if self.bets[0] == self.bets[1]:
                    return True
                next_to_act = self._opponent_idx(next_to_act)
                continue

            to_call = self.current_bet - self.bets[next_to_act]
            action = self._get_player_action(next_to_act, to_call, street_name)
            if not action or not isinstance(action, str):
                self._log(f"Player {next_to_act} => parse fail => fold.")
                return False

            action = action.strip().upper()
            self._log(f"Player {next_to_act} => {action}")

            # -------------------------- FOLD --------------------------
            if action == "FOLD":
                return False

            # -------------------------- CHECK -------------------------
            if action == "CHECK":
                if to_call:              # check when a call is required → treat as CALL
                    action = "CALL"
                else:
                    consecutive_checks += 1
                    if consecutive_checks >= 2 and self.bets[0] == self.bets[1]:
                        return True
                    next_to_act = self._opponent_idx(next_to_act)
                    continue

            # -------------------------- CALL --------------------------
            if action == "CALL":
                call_amt = min(to_call, self.stacks[next_to_act])
                self.stacks[next_to_act] -= call_amt
                self.pot += call_amt
                self.bets[next_to_act] += call_amt

                # if this call closed a voluntary bet/raise, end the street immediately
                if to_call and round_has_voluntary_bet:
                    return True

                consecutive_checks = 1   # one “check” equivalent (the call of zero next)
                next_to_act = self._opponent_idx(next_to_act)
                continue

            # ------------------- BET / RAISE --------------------------
            m = re.match(r"(BET|RAISE)\s*:\s*(\d+)", action)
            if not m:
                self._log("Unrecognized action => fold.")
                return False

            raise_amt = int(m.group(2))
            old_bet    = self.current_bet
            already_in = self.bets[next_to_act]

            new_bet = raise_amt if (old_bet == 0 and m.group(1) == "BET") else old_bet + raise_amt
            if (new_bet - old_bet) < self.last_raise_size:
                new_bet = old_bet + self.last_raise_size
            if new_bet <= already_in:
                self._log("Invalid raise (new bet not higher) => fold.")
                return False

            cost = min(new_bet - already_in, self.stacks[next_to_act])
            new_bet = already_in + cost

            # commit chips
            self.stacks[next_to_act] -= cost
            self.pot                 += cost
            self.bets[next_to_act]    = new_bet
            self.current_bet          = new_bet
            self.last_raise_size      = new_bet - old_bet
            round_has_voluntary_bet   = True
            consecutive_checks        = 0
            next_to_act               = self._opponent_idx(next_to_act)

    # ----------------------- NEW JSON PARSING HELPERS ------------------------

    @staticmethod
    def _extract_action_from_text(txt: str) -> Optional[str]:
        """
        Extract the 'action' value from a variety of possible LLM formats.
        """
        # Fast path – clean fences and try direct load
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.I)
        try:
            return str(json.loads(cleaned).get("action"))
        except Exception:
            pass

        # Fallback – grab the first well‑formed JSON object
        m = re.search(r"\{.*?\}", txt, re.S)
        if m:
            try:
                return str(json.loads(m.group(0)).get("action"))
            except Exception:
                pass
        return None

    # --------------------------- PLAYER ACTION ------------------------------

    def _get_player_action(self, pidx: int, to_call: int, street: str) -> Optional[str]:
        pl = self.players[pidx]
        if pl.strategy_type == "llm":
            return self._get_llm_action(pl, pidx, to_call, street)
        elif pl.strategy_type == "naive":
            return self._get_naive_action(pidx, to_call)
        elif pl.strategy_type == "random":
            if not pl.effective_strategy:
                picks = [("naive", None)] + [
                    ("llm", {"provider": cc["provider"], "model": cc["model"]})
                    for cc in COMMON_CONFIGS
                ]
                pick = random.choice(picks)
                pl.effective_strategy, pl.effective_model_config = pick
                if pl.effective_strategy == "llm":
                    try:
                        pl.client = LLMClient(
                            provider=pick[1]["provider"],
                            model=pick[1]["model"],
                            max_tokens=1024,
                            temperature=0.5,
                            max_retries=2,
                            timeout=60
                        )
                    except Exception as e:
                        self._log(f"LLM init error; falling back to naive. {e}")
                        pl.effective_strategy = "naive"
            if pl.effective_strategy == "naive":
                return self._get_naive_action(pidx, to_call)
            return self._get_llm_action(pl, pidx, to_call, street)
        return None

    def _get_naive_action(self, pidx: int, to_call: int) -> str:
        stack = self.stacks[pidx]
        if to_call >= stack:
            return "CALL"
        if to_call == 0:
            return f"BET: {stack}"
        needed = stack - to_call
        return f"RAISE: {needed}"

    def _get_llm_action(self, pl: Player, pidx: int, to_call: int, street: str) -> Optional[str]:
        if not pl.client:
            self._log("LLM strategy with no client; folding.")
            return None

        user_msg = (
            f"Street: {street}\n"
            f"Pot: {self.pot}\n"
            f"Your stack: {self.stacks[pidx]}\n"
            f"Opponent stack: {self.stacks[self._opponent_idx(pidx)]}\n"
            f"Amount to call: {to_call}\n"
            "Valid actions:\n"
            "  - FOLD\n"
            "  - CHECK (if nothing to call)\n"
            "  - CALL (if required)\n"
            "  - BET: X (if no current bet) or RAISE: X (if there is a bet; X must be at least the min raise)\n"
            "Return your action as JSON with keys 'reasoning' and 'action'."
        )
        sys_msg = "You are playing simplified heads-up Texas Hold'em. Return valid JSON."
        dev_msg = "Heads-up Hold'em: return an action in JSON (FOLD/CALL/CHECK/BET: X/RAISE: X)."

        resp = pl.client.call_llm(
            developer_message=dev_msg,
            user_message=user_msg,
            system_message=sys_msg
        )
        if not resp:
            return None

        txt = parse_response_text(resp) or ""
        return self._extract_action_from_text(txt)

    # --- Round Resolution and Logging ---

    def play_round(self) -> Tuple[Optional[int], List[int], List[str]]:
        self.game_active = True
        self.round_log = []
        deck = shuffle_deck()
        c0, c1 = self._deal_hole_cards(deck)
        flop, turn, river = self._get_community_cards(deck)

        self._log("--- New Heads-Up Hand ---")
        self._init_chips()
        self._post_blinds()
        self._log(f"Button => Player {self.button_index} ({self.players[self.button_index].get_display_name()})")
        self._log(f"P0 hole: {self._describe_cards(c0)} / stack={self.stacks[0]}")
        self._log(f"P1 hole: {self._describe_cards(c1)} / stack={self.stacks[1]}")

        for street, cards in [("Preflop", None), ("Flop", flop), ("Turn", turn), ("River", river)]:
            if street != "Preflop":
                self._log(f"{street.upper()}: {self._describe_cards(cards)}")
            if not self._betting_round(street):
                return self._fold_result()
            self._settle_bets()

        board = flop + turn + river
        cmp_val = self.compare_final_hands(c0, c1, board)
        if cmp_val > 0:
            return self._end_hand_with_winner(0)
        if cmp_val < 0:
            return self._end_hand_with_winner(1)
        self._log("Tie at showdown => no single winner.")
        self._save_log()
        return (None, [0, 1], self.round_log)

    def _fold_result(self) -> Tuple[Optional[int], List[int], List[str]]:
        folder = 0 if self.bets[0] < self.bets[1] else 1
        return self._end_hand_with_winner(self._opponent_idx(folder))

    def _end_hand_with_winner(self, widx: int) -> Tuple[Optional[int], List[int], List[str]]:
        self._log(f"Player {widx} ({self.players[widx].get_display_name()}) WINS the pot of {self.pot} chips!")
        self.game_active = False
        self._save_log()
        ts = time.strftime("%Y%m%d-%H%M%S")
        loser = self._opponent_idx(widx)
        hand_entry = {
            "timestamp": ts,
            "winner": self.players[widx].get_display_name(),
            "losers": [self.players[loser].get_display_name()],
            "round_log_file": self.log_filename,
            "chips_won": self.pot,
        }
        try:
            with open(HAND_HISTORY_JSON, "a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                cont = f.read().strip()
                hist = json.loads(cont) if cont else []
                if not isinstance(hist, list):
                    logger.warning("Corrupt hand history; overwriting.")
                    hist = []
                hist.append(hand_entry)
                f.seek(0)
                f.truncate(0)
                json.dump(hist, f, indent=2)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            self._log(f"Error updating hand history: {e}")
        return (
            self.players[widx].original_order,
            [self.players[self._opponent_idx(widx)].original_order],
            self.round_log
        )

    def _save_log(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.log_filename = f"heads_up_holdem_{ts}.log"
        fullp = os.path.join(LOGS_DIR, self.log_filename)
        try:
            with open(fullp, "w", encoding="utf-8") as f:
                f.write("\n".join(self.round_log))
            logger.info(f"Hand log saved => {fullp}")
        except Exception as e:
            logger.error(f"Failed to save hand log: {e}")

# ----------------------------------------------------------------------------
# INTERACTIVE CONFIGURATION
# ----------------------------------------------------------------------------

def get_player_configurations() -> List[Dict[str, Any]]:
    configs = []
    for i in range(2):
        print(f"\nPlayer {i+1} config:")
        print("  N => Naive All-In")
        print("  R => Random (naive or LLM)")
        print("  Or choose from the following LLM list:")
        for idx, cc in enumerate(COMMON_CONFIGS, start=1):
            print(f"    {idx}: {cc['provider']}/{cc['model']}")
        choice = input("Your choice: ").strip().upper()
        if choice == "N":
            configs.append({"strategy_type": "naive"})
        elif choice == "R":
            configs.append({"strategy_type": "random"})
        elif choice.isdigit() and 1 <= int(choice) <= len(COMMON_CONFIGS):
            c = COMMON_CONFIGS[int(choice) - 1]
            configs.append({
                "strategy_type": "llm",
                "provider": c["provider"],
                "model": c["model"]
            })
        else:
            print("Invalid input. Defaulting to naive.")
            configs.append({"strategy_type": "naive"})
    return configs

# ----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Heads-Up Texas Hold'em: Both players must act each round ===")
    try:
        pconfs = get_player_configurations()
        rounds_str = input("How many hands? [1]: ").strip()
        rounds_num = int(rounds_str) if rounds_str.isdigit() else 1
        game = HeadsUpTexasHoldEmGame(pconfs)
        for i in range(rounds_num):
            print(f"\n--- Hand {i+1}/{rounds_num} ---")
            winner, losers, _ = game.play_round()
            print("Hand ended in a tie." if winner is None else f"Winner: Player {winner}")
            print("Losers:", losers)
            game.button_index = game._opponent_idx(game.button_index)  # rotate button
        print(f"\nAll done. Logs are in {LOGS_DIR}")
        print(f"Results appended to {HAND_HISTORY_JSON}")
    except Exception as e:
        logger.error(f"Fatal Error => {e}", exc_info=True)
        print(f"Fatal Error => {e}")
