#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
import time
import os
import json
import re
import itertools
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

RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
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
    {"strategy_type": "llm", "provider": "openai", "model": "o4-mini-2025-04-16"},
    #{"strategy_type": "llm", "provider": "openai", "model": "o3-2025-04-16"},
    # === Anthropic ===
    # {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    # {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},
    # # === DeepSeek ===
    # {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-chat-v3-0324:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-qwen-32b:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-llama-70b:floor"},
    # # === LLaMA ===
    # {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-maverick:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-scout:floor"},
    # # === Google ===
    # {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-pro-preview-03-25:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemma-3-27b-it:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.0-flash-001:floor"},
    # # === Miscellaneous ===
    # {"strategy_type": "llm", "provider": "openrouter", "model": "cohere/command-a:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "x-ai/grok-3-beta:floor"},
    # {"strategy_type": "llm", "provider": "openrouter", "model": "mistralai/mistral-small-3.1-24b-instruct:floor"},
]

# ----------------------------------------------------------------------------
# PLAYER CLASS
# ----------------------------------------------------------------------------


@dataclass
class Player:
    player_id: int
    strategy_type: str  # "llm", or "random"
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
                max_tokens=8192,
                temperature=0.5,
                max_retries=2,
                timeout=60,
            )
        elif self.strategy_type == "random":
            pass
        else:
            raise ValueError(f"Unrecognized strategy '{self.strategy_type}'.")

    def get_display_name(self) -> str:
        if self.strategy_type == "llm":
            p = self.model_config.get("provider", "?")
            m = self.model_config.get("model", "?")
            return f"{p}/{m}"

        if self.strategy_type == "random":
            if not self.effective_strategy:
                return "Random(undecided)"
            if self.effective_strategy == "llm":
                p = self.effective_model_config.get("provider", "?")
                m = self.effective_model_config.get("model", "?")
                return f"{p}/{m}"
            return self.effective_strategy

        return "Unknown"


# ----------------------------------------------------------------------------
# HEADS-UP TEXAS HOLD'EM GAME CLASS
# ----------------------------------------------------------------------------


class HeadsUpTexasHoldEmGame:
    STARTING_STACK = 200
    SMALL_BLIND = 1
    BIG_BLIND = 2

    def __init__(self, player_configs: List[Dict[str, Any]]):
        if len(player_configs) != 2:
            raise ValueError("Heads-up game requires exactly 2 players.")
        self.players: List[Player] = []
        for i, cfg in enumerate(player_configs):
            st = cfg.get("strategy_type", "random")
            if st == "llm":
                pl = Player(
                    player_id=i,
                    strategy_type="llm",
                    model_config={"provider": cfg["provider"], "model": cfg["model"]},
                    original_order=i,
                )
            elif st == "random":
                pl = Player(player_id=i, strategy_type="random", original_order=i)
            else:
                raise ValueError(f"Bad strategy {st} for player {i}")
            self.players.append(pl)
        random.shuffle(self.players)
        self._assign_random_strategies()
        self.button_index = 0
        self.round_log: List[str] = []
        self.public_action_history: List[str] = []
        self.game_active = False

        # pending sub‑hand records (only flushed after paired round finishes)
        self.pending_hand_entries: List[Dict[str, Any]] = []

    # --- Utility Methods ---

    def _log(self, msg: str):
        logger.info(msg)
        self.round_log.append(msg)

    def _opponent_idx(self, idx: int) -> int:
        return 1 - idx

    # --- Random-strategy assignment (performed once at game start) ---

    def _assign_random_strategies(self):
        reserved_models = {
            (pl.model_config["provider"], pl.model_config["model"])
            for pl in self.players
            if pl.strategy_type == "llm"
        }

        available_picks: List[Tuple[str, Optional[Dict[str, str]]]] = []

        for cc in COMMON_CONFIGS:
            mdl_sig = (cc["provider"], cc["model"])
            if mdl_sig not in reserved_models:
                available_picks.append(
                    ("llm", {"provider": cc["provider"], "model": cc["model"]})
                )

        for pl in self.players:
            if pl.strategy_type != "random":
                continue
            if not available_picks:
                logger.warning("Out of unique strategies; defaulting to llm.")
                pl.effective_strategy = "llm"
                pl.effective_model_config = None
                continue

            idx = random.randrange(len(available_picks))
            pick = available_picks.pop(idx)

            pl.effective_strategy, pl.effective_model_config = pick

            if pl.effective_strategy == "llm":
                try:
                    pl.client = LLMClient(
                        provider=pl.effective_model_config["provider"],
                        model=pl.effective_model_config["model"],
                        max_tokens=8192,
                        temperature=0.5,
                        max_retries=2,
                        timeout=60,
                    )
                except Exception as e:
                    logger.warning(f"LLM init error; falling back to llm. {e}")
                    pl.effective_strategy = "llm"
                    pl.effective_model_config = None

    # --- Card Dealing ---

    def _deal_hole_cards(self, deck: List[int]) -> Tuple[List[int], List[int]]:
        return deck[:2], deck[2:4]

    def _get_community_cards(
        self, deck: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
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
        sorted_counts = sorted(
            rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True
        )
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
            unique_r.sort()
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
        has_straight = longest >= 5
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
            flush_cards = sorted(
                [r for r, s in zip(ranks, suits) if s == flush_suit], reverse=True
            )
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

    def compare_final_hands(
        self, cardsA: List[int], cardsB: List[int], board: List[int]
    ) -> int:
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

    # ----------------------- NEW: ALL-IN EV SPLIT ---------------------------

    def _ev_split_allin(self) -> Tuple[Optional[int], List[int], List[str]]:
        needed = 5 - len(self.board_so_far)
        seen = set(self.hole_cards[0] + self.hole_cards[1] + self.board_so_far)
        remaining_cards = [c for c in range(52) if c not in seen]

        wins0 = wins1 = ties = 0
        for extra in itertools.combinations(remaining_cards, needed):
            board = self.board_so_far + list(extra)
            res = self.compare_final_hands(
                self.hole_cards[0], self.hole_cards[1], board
            )
            if res > 0:
                wins0 += 1
            elif res < 0:
                wins1 += 1
            else:
                ties += 1

        total = wins0 + wins1 + ties
        equity0 = (wins0 + ties * 0.5) / total
        equity1 = 1.0 - equity0

        chips0 = int(round(equity0 * self.pot))
        chips1 = self.pot - chips0

        self.stacks[0] += chips0
        self.stacks[1] += chips1
        self.pot = 0  # pot is fully distributed

        self._log(
            f"All‑in equity split ⇒ P0 gets {chips0} chips ({equity0:.3%}), "
            f"P1 gets {chips1} chips ({equity1:.3%})."
        )

        # award exact 50/50 to the button
        if self.stacks[0] == self.stacks[1]:
            btn = self.button_index
            loser = self._opponent_idx(btn)
            self._log(
                f"Exact 50/50, awarded to button ⇒ Player {btn} (net 0 chips)."
            )
            self.game_active = False
            self._save_log()
            return (btn, [loser], self.round_log)

        # otherwise normal equity winner
        widx = 0 if self.stacks[0] > self.stacks[1] else 1
        loser = self._opponent_idx(widx)
        net_chips_won = self.stacks[widx] - self.STARTING_STACK

        self._log(
            f"Player {widx} ({self.players[widx].get_display_name()}) "
            f"WINS net {net_chips_won} chips after equity split!"
        )

        hand_entry = {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "winner": self.players[widx].get_display_name(),
            "losers": [self.players[loser].get_display_name()],
            "round_log_file": getattr(self, "log_filename", ""),
            "chips_won": net_chips_won,
        }
        self.pending_hand_entries.append(hand_entry)

        self.game_active = False
        self._save_log()
        return (widx, [loser], self.round_log)

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

    def _betting_round(self, street_name: str) -> bool:
        self._log(f"--- {street_name} betting round ---")
        next_to_act = (
            self.button_index
            if street_name == "Preflop"
            else self._opponent_idx(self.button_index)
        )

        consecutive_checks = 0
        round_has_voluntary_bet = False

        while True:
            if self.stacks[next_to_act] == 0:
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
            self.public_action_history.append(
                f"{street_name.lower()}: P{next_to_act} {action}"
            )

            if action == "FOLD":
                return False

            if action == "CHECK":
                if to_call:
                    action = "CALL"
                else:
                    consecutive_checks += 1
                    if consecutive_checks >= 2 and self.bets[0] == self.bets[1]:
                        return True
                    next_to_act = self._opponent_idx(next_to_act)
                    continue

            if action == "CALL":
                if to_call == 0:
                    consecutive_checks += 1
                    if consecutive_checks >= 2 and self.bets[0] == self.bets[1]:
                        return True
                    next_to_act = self._opponent_idx(next_to_act)
                    continue

                call_amt = min(to_call, self.stacks[next_to_act])
                self.stacks[next_to_act] -= call_amt
                self.pot += call_amt
                self.bets[next_to_act] += call_amt

                if to_call and round_has_voluntary_bet:
                    return True

                consecutive_checks = 1
                next_to_act = self._opponent_idx(next_to_act)
                continue

            m = re.match(r"(BET|RAISE)\s*:\s*(\d+)", action)
            if not m:
                self._log("Unrecognized action => fold.")
                return False

            raise_amt = int(m.group(2))
            old_bet = self.current_bet
            already_in = self.bets[next_to_act]

            new_bet = (
                raise_amt
                if (old_bet == 0 and m.group(1) == "BET")
                else old_bet + raise_amt
            )
            if (new_bet - old_bet) < self.last_raise_size:
                new_bet = old_bet + self.last_raise_size
            if new_bet <= already_in:
                self._log("Invalid raise (new bet not higher) => fold.")
                return False

            cost = min(new_bet - already_in, self.stacks[next_to_act])
            new_bet = already_in + cost

            self.stacks[next_to_act] -= cost
            self.pot += cost
            self.bets[next_to_act] = new_bet
            self.current_bet = new_bet
            self.last_raise_size = new_bet - old_bet
            round_has_voluntary_bet = True
            consecutive_checks = 0
            next_to_act = self._opponent_idx(next_to_act)

    # ----------------------- NEW JSON PARSING HELPERS ------------------------

    @staticmethod
    def _extract_action_from_text(txt: str) -> Optional[str]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip(), flags=re.I)
        try:
            return str(json.loads(cleaned).get("action"))
        except Exception:
            pass

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
        elif pl.strategy_type == "random":
            if not pl.effective_strategy:
                picks = [
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
                            max_tokens=8192,
                            temperature=0.5,
                            max_retries=2,
                            timeout=60,
                        )
                    except Exception as e:
                        self._log(
                            f"LLM init error; falling back to llm default. {e}"
                        )
                        pl.effective_strategy = "llm"
                        pl.effective_model_config = {"provider": COMMON_CONFIGS[0]["provider"], "model": COMMON_CONFIGS[0]["model"]}
            return self._get_llm_action(pl, pidx, to_call, street)
        return None

    def _get_llm_action(
        self, pl: Player, pidx: int, to_call: int, street: str
    ) -> Optional[str]:
        if not pl.client:
            self._log("LLM strategy with no client; folding.")
            return None

        sys_msg = (
            "You are playing a simplified heads‑up Texas Hold'em cash game against one opponent.\n"
            "\n"
            "Valid actions (case‑insensitive):\n"
            "  • FOLD\n"
            "  • CHECK          – only when Amount to call is 0\n"
            "  • CALL           – match the current bet when Amount to call > 0\n"
            "  • BET: X         – when no current bet (X = chips you wager)\n"
            "  • RAISE: X       – X is the **additional** chips above the current bet and "
            "must be at least the table's minimum‑raise size\n"
            "\n"
            "Respond with **JSON only**, exactly two keys:\n"
            '  { "reasoning": "<detailed thoughts>", "action": "<one of the actions above>" }\n'
            "\n"
            "Examples:\n"
            '  { "reasoning": "No equity, fold to pressure", "action": "FOLD" }\n'
            '  { "reasoning": "Top pair, value bet",          "action": "BET: 6" }\n'
            "\n"
            "General tips: fold very weak holdings facing large bets; check marginal hands when free; "
            "call with suitable odds; bet strong hands for value; raise with premiums or strong draws."
        )

        dev_msg = sys_msg

        community = (
            self._describe_cards(self.board_so_far) if self.board_so_far else "(none)"
        )
        history_str = (
            "; ".join(self.public_action_history) if self.public_action_history else "(none)"
        )

        user_msg = (
            f"You are Player {pidx}\n"
            f"Street: {street}\n"
            f"Pot: {self.pot}\n"
            f"Current bet: {self.current_bet}\n"
            f"Your stack: {self.stacks[pidx]}\n"
            f"Opponent stack: {self.stacks[self._opponent_idx(pidx)]}\n"
            f"Amount to call: {to_call}\n"
            f"Hole cards: {self._describe_cards(self.hole_cards[pidx])}\n"
            f"Community cards: {community}\n"
            f"History: {history_str}"
        )

        resp = pl.client.call_llm(
            developer_message=dev_msg, user_message=user_msg, system_message=sys_msg
        )
        if not resp:
            return None

        txt = parse_response_text(resp) or ""
        return self._extract_action_from_text(txt)

    # --- Round Resolution and Logging ---

    def _fold_result(self) -> Tuple[Optional[int], List[int], List[str]]:
        folder = 0 if self.bets[0] < self.bets[1] else 1
        return self._end_hand_with_winner(self._opponent_idx(folder))

    def _end_hand_with_winner(
        self, widx: int
    ) -> Tuple[Optional[int], List[int], List[str]]:
        self.stacks[widx] += self.pot
        net_chips_won = self.stacks[widx] - self.STARTING_STACK

        self._log(
            f"Player {widx} ({self.players[widx].get_display_name()}) "
            f"WINS the pot of {self.pot} chips! (Net +{net_chips_won})"
        )
        self.game_active = False
        self._save_log()

        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        loser = self._opponent_idx(widx)
        hand_entry = {
            "timestamp": ts,
            "winner": self.players[widx].get_display_name(),
            "losers": [self.players[loser].get_display_name()],
            "round_log_file": self.log_filename,
            "chips_won": net_chips_won,
        }
        self.pending_hand_entries.append(hand_entry)

        return (
            widx,
            [self._opponent_idx(widx)],
            self.round_log,
        )

    def _save_log(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.log_filename = f"heads_up_holdem_{ts}.log"
        fullp = os.path.join(LOGS_DIR, self.log_filename)
        try:
            with open(fullp, "w", encoding="utf-8") as f:
                f.write("\n".join(self.round_log))
            logger.info(f"Hand log saved => {fullp}")
        except Exception as e:
            logger.error(f"Failed to save hand log: {e}")

    # --- Single Hand (internal) ---

    def _play_single_hand(
        self,
        c0: List[int],
        c1: List[int],
        flop: List[int],
        turn: List[int],
        river: List[int],
    ) -> Tuple[Optional[int], List[int], List[str]]:
        self.game_active = True
        self.round_log = []
        self.public_action_history = []

        self.hole_cards = [c0, c1]
        self.board_so_far: List[int] = []

        self._init_chips()
        self._post_blinds()
        btn = self.button_index
        opp = self._opponent_idx(btn)
        self._log("--- New Heads-Up Hand ---")
        self._log(f"Button   => Player {btn} ({self.players[btn].get_display_name()})")
        self._log(f"Opponent => Player {opp} ({self.players[opp].get_display_name()})")
        self._log(f"P0 hole: {self._describe_cards(c0)} / stack={self.stacks[0]}")
        self._log(f"P1 hole: {self._describe_cards(c1)} / stack={self.stacks[1]}")

        for street, cards in [
            ("Preflop", None),
            ("Flop", flop),
            ("Turn", turn),
            ("River", river),
        ]:
            if street != "Preflop":
                self._log(f"{street.upper()}: {self._describe_cards(cards)}")
                self.board_so_far.extend(cards)

            if not self._betting_round(street):
                return self._fold_result()
            self._settle_bets()

            # --- NEW: Detect mutual all-in and split pot by equity
            if self.stacks[0] == 0 and self.stacks[1] == 0:
                return self._ev_split_allin()

        board = flop + turn + river
        outcome = self.compare_final_hands(c0, c1, board)
        if outcome > 0:
            return self._end_hand_with_winner(0)
        if outcome < 0:
            return self._end_hand_with_winner(1)

        # --- TIE CASE HANDLING ---
        self._log("Tie at showdown => no single winner.")
        self._save_log()

        # Log a neutral entry: randomly assign winner/loser, 0 chips.
        widx = random.randint(0, 1)
        loser = self._opponent_idx(widx)
        hand_entry = {
            "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
            "winner": self.players[widx].get_display_name(),
            "losers": [self.players[loser].get_display_name()],
            "round_log_file": self.log_filename,
            "chips_won": 0,
        }
        self.pending_hand_entries.append(hand_entry)

        return (None, [0, 1], self.round_log)

    # --- Paired Rounds Orchestrator ---

    def play_round(
        self,
    ) -> Tuple[List[Optional[int]], List[List[int]], List[str]]:
        deck = shuffle_deck()
        self._assign_random_strategies()
        c0, c1 = self._deal_hole_cards(deck)
        flop, turn, river = self._get_community_cards(deck)

        # First sub-hand with initial player ordering
        res1 = self._play_single_hand(c0, c1, flop, turn, river)

        # Swap player models/strategies while keeping card positions the same
        self.players[0], self.players[1] = self.players[1], self.players[0]

        # Second sub-hand with models swapped, cards in original positions
        res2 = self._play_single_hand(c0, c1, flop, turn, river)

        combined_log = (
            res1[2]
            + ["--- Paired hand restart (models swapped) ---"]
            + res2[2]
        )
        winners = [res1[0], res2[0]]
        losers_lists = [res1[1], res2[1]]

        # --- NEW: Persist paired‑hand aggregate log + single JSON entry ---
        # 1) write one .log file for the whole paired round
        pair_ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        paired_log_filename = f"heads_up_holdem_pair_{pair_ts}.log"
        with open(os.path.join(LOGS_DIR, paired_log_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(combined_log))
        logger.info(f"Paired hand log saved => {paired_log_filename}")

        # 2) write the combined JSON entry once the paired round is complete
        sub_entries = self.pending_hand_entries[:]
        self.pending_hand_entries = []

        if sub_entries:
            paired_entry = {
                "timestamp": pair_ts[:-7],  # drop micros for consistency
                "sub_hands": sub_entries,
                "round_log_file": paired_log_filename,
            }
            try:
                with open(HAND_HISTORY_JSON, "a+", encoding="utf-8") as fh:
                    fcntl.flock(fh, fcntl.LOCK_EX)
                    fh.seek(0)
                    data_txt = fh.read().strip()
                    hist = json.loads(data_txt) if data_txt else []
                    if not isinstance(hist, list):
                        hist = []
                    hist.append(paired_entry)
                    fh.seek(0)
                    fh.truncate(0)
                    json.dump(hist, fh, indent=2)
                    fcntl.flock(fh, fcntl.LOCK_UN)
            except Exception as e:
                logger.error(f"Failed to write paired‑hand history: {e}")

        # Restore original player order for future rounds
        self.players[0], self.players[1] = self.players[1], self.players[0]

        return winners, losers_lists, combined_log


# ----------------------------------------------------------------------------
# INTERACTIVE CONFIGURATION
# ----------------------------------------------------------------------------


def get_player_configurations() -> List[Dict[str, Any]]:
    configs = []
    for i in range(2):
        print(f"\nPlayer {i+1} config:")
        print("  R => Random")
        print("  Or choose from the following LLM list:")
        for idx, cc in enumerate(COMMON_CONFIGS, start=1):
            print(f"    {idx}: {cc['provider']}/{cc['model']}")
        choice = input("Your choice: ").strip().upper()
        if choice == "R":
            configs.append({"strategy_type": "random"})
        elif choice.isdigit() and 1 <= int(choice) <= len(COMMON_CONFIGS):
            c = COMMON_CONFIGS[int(choice) - 1]
            configs.append(
                {
                    "strategy_type": "llm",
                    "provider": c["provider"],
                    "model": c["model"],
                }
            )
        else:
            print("Invalid input. Defaulting to random.")
            configs.append({"strategy_type": "random"})
    return configs


# ----------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Heads-Up Texas Hold'em: Paired Hands ===")
    try:
        pconfs = get_player_configurations()
        rounds_str = input("How many paired rounds? [1]: ").strip()
        rounds_num = int(rounds_str) if rounds_str.isdigit() else 1
        game = HeadsUpTexasHoldEmGame(pconfs)
        for i in range(rounds_num):
            print(f"\n--- Paired Round {i+1}/{rounds_num} ---")
            winners, losers_lists, _ = game.play_round()
            for idx, w in enumerate(winners, 1):
                if w is None:
                    print(f"  Sub-hand {idx}: tie")
                else:
                    print(f"  Sub-hand {idx}: Winner: Player {w}")
            game.button_index = game._opponent_idx(game.button_index)
        print(f"\nAll done. Logs are in {LOGS_DIR}")
        print(f"Results appended to {HAND_HISTORY_JSON}")
    except Exception as e:
        logger.error(f"Fatal Error => {e}", exc_info=True)
        print(f"Fatal Error => {e}")