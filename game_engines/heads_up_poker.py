# game_engines/heads_up_poker.py

import random
import itertools
import json
import copy
import logging
import time # Import time for log_event
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from game_engines.base_game import BaseGame
from model_orchestrator.utils import (
    log_initial_state, log_hand_result, log_action_result,
    log_event_to_game_file, close_game_log, format_cards
)

logger = logging.getLogger(__name__)

SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_TO_VALUE = {rank: i for i, rank in enumerate(RANKS, start=2)}


class Deck:
    def __init__(self, random_seed=None):
        self.cards = list(itertools.product(RANKS, SUITS))
        self._rng = random.Random(random_seed)
        self._rng.shuffle(self.cards)
        logger.debug(f"Deck initialized with seed {random_seed}. Cards: {len(self.cards)}")

    def draw(self, num=1) -> List[Tuple[str, str]]:
        if num > len(self.cards):
            logger.error(f"Cannot draw {num} cards, only {len(self.cards)} remaining")
            raise ValueError(f"Cannot draw {num} cards, only {len(self.cards)} remaining")
        drawn_cards = [self.cards.pop() for _ in range(num)]
        logger.debug(f"Drew {num} cards: {drawn_cards}. Remaining: {len(self.cards)}")
        return drawn_cards

    def to_dict(self):
        return {'cards': self.cards}

    @classmethod
    def from_dict(cls, data, random_seed=None):
        deck = cls(random_seed=random_seed)
        deck.cards = data.get('cards', [])
        return deck

class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hole_cards: List[Tuple[str, str]] = []
        self.current_bet = 0
        self.total_bet_in_hand = 0
        self.folded = False
        self.all_in = False
        self.last_action = None

    def bet(self, amount) -> int:
        if amount < 0:
            logger.warning(f"Attempted negative bet: {amount} for player {self.name}")
            raise ValueError("Bet amount cannot be negative")
        bet_amount = min(amount, self.stack)
        self.stack -= bet_amount
        self.current_bet += bet_amount
        self.total_bet_in_hand += bet_amount
        if self.stack == 0:
            self.all_in = True
            logger.info(f"Player {self.name} is all-in.")
        logger.debug(f"Player {self.name} bets {bet_amount}. Stack: {self.stack}, CurrentBet: {self.current_bet}, TotalBetHand: {self.total_bet_in_hand}")
        return bet_amount

    def fold(self):
        self.folded = True
        self.last_action = "fold"
        logger.info(f"Player {self.name} folds.")

    def reset_for_new_hand(self):
        logger.debug(f"Resetting player {self.name} for new hand. Stack: {self.stack}")
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet_in_hand = 0
        self.folded = False
        self.all_in = False
        if self.stack == 0:
             self.all_in = True
        self.last_action = None

    def to_dict(self):
        return {
            'name': self.name, 'stack': self.stack, 'hole_cards': self.hole_cards,
            'current_bet': self.current_bet, 'total_bet_in_hand': self.total_bet_in_hand,
            'folded': self.folded, 'all_in': self.all_in, 'last_action': self.last_action,
        }

    @classmethod
    def from_dict(cls, data):
        player = cls(data['name'], data['stack'])
        player.hole_cards = data.get('hole_cards', [])
        player.current_bet = data.get('current_bet', 0)
        player.total_bet_in_hand = data.get('total_bet_in_hand', 0)
        player.folded = data.get('folded', False)
        player.all_in = data.get('all_in', False)
        player.last_action = data.get('last_action')
        return player

class HandEvaluator:
    @staticmethod
    def evaluate_five_card_hand(hand):
        values = sorted([RANK_TO_VALUE[rank] for rank, suit in hand], reverse=True)
        suits = [suit for rank, suit in hand]
        value_counts = Counter(values)
        counts = sorted(value_counts.values(), reverse=True)

        is_flush = len(set(suits)) == 1
        sorted_vals = sorted(list(set(values)))
        is_straight = False
        straight_high = None

        if set([14, 2, 3, 4, 5]).issubset(values):
            is_straight = True
            straight_high = 5
        else:
            if len(sorted_vals) >= 5:
                for i in range(len(sorted_vals) - 4):
                    window = sorted_vals[i:i+5]
                    if all(window[j] == window[0] + j for j in range(5)):
                        is_straight = True
                        straight_high = window[-1]
                        break

        rank_score = -1
        tiebreakers = ()

        if is_straight and is_flush: rank_score = 8; tiebreakers = (straight_high,)
        elif 4 in counts:
            rank_score = 7
            quad = [val for val, cnt in value_counts.items() if cnt == 4][0]
            kicker = max([val for val in values if val != quad])
            tiebreakers = (quad, kicker)
        elif 3 in counts and 2 in counts:
            rank_score = 6
            trip = max([val for val, cnt in value_counts.items() if cnt == 3])
            pair = max([val for val, cnt in value_counts.items() if cnt == 2])
            tiebreakers = (trip, pair)
        elif is_flush: rank_score = 5; tiebreakers = tuple(sorted(values, reverse=True)[:5]) # Only top 5 matter
        elif is_straight: rank_score = 4; tiebreakers = (straight_high,)
        elif 3 in counts:
            rank_score = 3
            trip = [val for val, cnt in value_counts.items() if cnt == 3][0]
            kickers = sorted([val for val in values if val != trip], reverse=True)[:2]
            tiebreakers = (trip,) + tuple(kickers)
        elif counts.count(2) >= 2:
            rank_score = 2
            pairs = sorted([val for val, cnt in value_counts.items() if cnt == 2], reverse=True)
            high_pair, low_pair = pairs[0], pairs[1]
            kicker = max([val for val in values if val not in pairs])
            tiebreakers = (high_pair, low_pair, kicker)
        elif 2 in counts:
            rank_score = 1
            pair = [val for val, cnt in value_counts.items() if cnt == 2][0]
            kickers = sorted([val for val in values if val != pair], reverse=True)[:3]
            tiebreakers = (pair,) + tuple(kickers)
        else: rank_score = 0; tiebreakers = tuple(sorted(values, reverse=True)[:5])

        return (rank_score, tiebreakers)

    @staticmethod
    def best_hand_rank(cards: List[Tuple[str, str]]) -> Tuple[int, tuple]:
        if len(cards) < 5: return (-1, ())
        if len(cards) == 5: return HandEvaluator.evaluate_five_card_hand(cards)
        best_rank: Tuple[int, tuple] = (-1, ())
        for combo in itertools.combinations(cards, 5):
            rank = HandEvaluator.evaluate_five_card_hand(list(combo))
            # Use lexicographical comparison for tiebreakers
            if rank[0] > best_rank[0] or (rank[0] == best_rank[0] and rank[1] > best_rank[1]):
                best_rank = rank
        return best_rank

    @staticmethod
    def hand_description(rank_tuple: Tuple[int, tuple]) -> str:
        rank_names = {8: "Straight Flush", 7: "Four of a Kind", 6: "Full House", 5: "Flush", 4: "Straight", 3: "Three of a Kind", 2: "Two Pair", 1: "One Pair", 0: "High Card"}
        rank_value, tiebreakers = rank_tuple
        desc = rank_names.get(rank_value, "Unknown")
        readable_tiebreakers = [RANK_TO_VALUE.get(tb, str(tb)) for tb in tiebreakers]
        # Corrected: Convert values back to ranks
        value_to_rank = {v: r for r, v in RANK_TO_VALUE.items()}
        readable_tiebreakers = [value_to_rank.get(tb, str(tb)) for tb in tiebreakers]

        return f"{desc} ({', '.join(readable_tiebreakers)})"

    @staticmethod
    def determine_winner(player1: Player, player2: Player, community_cards: List[Tuple[str, str]]) -> Tuple[Optional[Player], Tuple[int, tuple], Tuple[int, tuple]]:
        hand1, hand2 = (-1, ()), (-1, ())
        cards1, cards2 = player1.hole_cards + community_cards, player2.hole_cards + community_cards
        if not player1.folded and len(cards1) >= 5: hand1 = HandEvaluator.best_hand_rank(cards1)
        if not player2.folded and len(cards2) >= 5: hand2 = HandEvaluator.best_hand_rank(cards2)

        logger.info(f"Evaluating hands: {player1.name} ({HandEvaluator.hand_description(hand1) if not player1.folded else 'Folded'}) vs {player2.name} ({HandEvaluator.hand_description(hand2) if not player2.folded else 'Folded'})")

        # Compare ranks using lexicographical comparison for tiebreakers
        if hand1 > hand2: return player1, hand1, hand2
        if hand2 > hand1: return player2, hand1, hand2

        # Handle Ties/Fold scenarios
        if not player1.folded and not player2.folded and hand1 == hand2 and hand1[0] != -1: return None, hand1, hand2 # Tie
        if not player1.folded and player2.folded: return player1, hand1, hand2 # P1 wins
        if player1.folded and not player2.folded: return player2, hand1, hand2 # P2 wins
        logger.error("Showdown reached with both players folded or invalid hands."); return None, hand1, hand2 # Error/Tie

class HeadsUpPoker(BaseGame):
    def __init__(self, game_id=None, players=None, random_seed=None,
                 player1_name="Player1", player2_name="Player2",
                 starting_stack=1000, small_blind=10, big_blind=20):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._rng = random.Random(random_seed)
        self._current_seed = random_seed
        super().__init__(game_id, players=[player1_name, player2_name], random_seed=self._current_seed)
        logger.info(f"HeadsUpPoker game initialized. ID: {self.game_id}, Seed: {self._current_seed}")
        log_event_to_game_file(f"Game Initialized. Seed: {self._current_seed}, Blinds: {small_blind}/{big_blind}, Start Stack: {starting_stack}\n")

    def _initialize_game_state(self):
        deck_seed = hash((self._current_seed, "init")) if self._current_seed is not None else None
        self.deck = Deck(random_seed=deck_seed)
        self.players_obj = [Player(self.player1_name, self.starting_stack), Player(self.player2_name, self.starting_stack)]
        self.community_cards: List[Tuple[str, str]] = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_index = self._rng.choice([0, 1])
        self.active_player_index = -1
        self.last_raiser_index = -1
        self.stage = "pre-deal"
        self.hand_number = 0
        self.hand_complete = False
        self.game_complete = False
        self.aggressor_action_closed = False
        logger.debug("Game state initialized.")

    def reset(self):
        logger.info("Resetting game to initial state.")
        self._initialize_game_state()
        self.history = []
        log_event_to_game_file("--- GAME RESET ---\n")

    def reset_hand(self):
        player1_stack = self.players_obj[0].stack
        player2_stack = self.players_obj[1].stack
        if player1_stack <= 0 or player2_stack <= 0:
            self.game_complete = True
            winner = self.players_obj[0] if player1_stack > player2_stack else self.players_obj[1]
            logger.info(f"Game complete. Player {winner.name} wins.")
            log_event_to_game_file(f"--- GAME OVER: {winner.name} wins ---\n")
            return

        if self.hand_number > 0: # Log completion of previous hand
             self.log_event("hand_complete", {
                 "hand_number": self.hand_number,
                 "player_stacks": {p.name: p.stack for p in self.players_obj},
                 "pot": self.pot # Log final pot before reset
             })

        self.hand_number += 1
        logger.info(f"Resetting for Hand #{self.hand_number}")
        log_event_to_game_file(f"--- Starting Hand #{self.hand_number} ---\n")

        deck_seed = hash((self._current_seed, self.hand_number)) if self._current_seed is not None else None
        self.deck = Deck(random_seed=deck_seed)
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.last_raiser_index = -1
        self.dealer_index = 1 - self.dealer_index
        self.active_player_index = -1
        self.stage = "pre-deal"
        self.hand_complete = False
        self.aggressor_action_closed = False
        for player in self.players_obj: player.reset_for_new_hand()
        logger.debug(f"Hand {self.hand_number} reset complete. Dealer: {self.players_obj[self.dealer_index].name}")

    def start_hand(self):
        if self.stage != "pre-deal" or self.game_complete: return False
        logger.info(f"Starting Hand #{self.hand_number}")

        sb_player_index = self.dealer_index
        bb_player_index = 1 - self.dealer_index
        small_blind_player = self.players_obj[sb_player_index]
        big_blind_player = self.players_obj[bb_player_index]
        logger.info(f"Dealer (SB): {small_blind_player.name}, BB: {big_blind_player.name}")

        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)
        self.pot += sb_amount + bb_amount
        self.current_bet = max(small_blind_player.current_bet, big_blind_player.current_bet)
        self.last_raiser_index = bb_player_index if bb_amount >= sb_amount else sb_player_index

        logger.info(f"Blinds posted. SB: {sb_amount} by {small_blind_player.name} (Stack: {small_blind_player.stack}). BB: {bb_amount} by {big_blind_player.name} (Stack: {big_blind_player.stack}). Pot: {self.pot}. Current Bet: {self.current_bet}")
        self.log_event("blinds_posted", {
            "hand": self.hand_number,
            "small_blind": {"player": small_blind_player.name, "amount": sb_amount, "stack": small_blind_player.stack},
            "big_blind": {"player": big_blind_player.name, "amount": bb_amount, "stack": big_blind_player.stack},
            "pot": self.pot
        })

        try:
            self.players_obj[0].hole_cards = self.deck.draw(2)
            self.players_obj[1].hole_cards = self.deck.draw(2)
        except ValueError as e: logger.error(f"Error dealing cards: {e}"); self.game_complete = True; return False

        log_event_to_game_file(f"Hole cards dealt:\n  {self.players_obj[0].name}: {format_cards(self.players_obj[0].hole_cards)}\n  {self.players_obj[1].name}: {format_cards(self.players_obj[1].hole_cards)}\n")
        self.log_event("hole_cards_dealt", { "hand": self.hand_number, self.players_obj[0].name: self.players_obj[0].hole_cards, self.players_obj[1].name: self.players_obj[1].hole_cards })

        self.stage = "pre-flop"
        self.active_player_index = sb_player_index
        self.aggressor_action_closed = False
        logger.info(f"Pre-flop stage begins. Active player: {self.players_obj[self.active_player_index].name}. Current Bet: {self.current_bet}")
        log_initial_state(self.get_state(), self.hand_number, self.stage)
        return True

    def deal_community_cards(self):
        if self.game_complete: return False
        num_cards_to_deal, next_stage, event_name = 0, None, None
        if self.stage == "pre-flop": num_cards_to_deal, next_stage, event_name = 3, "flop", "flop_dealt"
        elif self.stage == "flop": num_cards_to_deal, next_stage, event_name = 1, "turn", "turn_dealt"
        elif self.stage == "turn": num_cards_to_deal, next_stage, event_name = 1, "river", "river_dealt"
        else: logger.error(f"deal_community_cards called at invalid stage: {self.stage}"); return False

        try:
            if len(self.deck.cards) > num_cards_to_deal: self.deck.draw(1) # Burn
            new_cards = self.deck.draw(num_cards_to_deal)
            self.community_cards.extend(new_cards)
            logger.info(f"{next_stage.capitalize()} dealt: {format_cards(new_cards)}. Community: {format_cards(self.community_cards)}")
            log_event_to_game_file(f"{next_stage.capitalize()} dealt: {format_cards(self.community_cards)}\n")
            self.log_event(event_name, {"hand": self.hand_number, "cards": new_cards, "community_total": self.community_cards})
        except ValueError as e: logger.error(f"Error dealing community cards: {e}"); self.game_complete = True; return False

        self.stage = next_stage
        self.current_bet = 0
        self.last_raiser_index = -1
        self.aggressor_action_closed = False
        for player in self.players_obj: player.current_bet = 0; player.last_action = None

        self.active_player_index = 1 - self.dealer_index
        if self.players_obj[self.active_player_index].all_in or self.players_obj[self.active_player_index].folded:
            self.active_player_index = 1 - self.active_player_index

        logger.info(f"Stage is now {self.stage}. Active player: {self.players_obj[self.active_player_index].name}. Bets reset.")
        log_event_to_game_file(f"--- {self.stage.upper()} --- Active: {self.players_obj[self.active_player_index].name}\n")
        self._check_betting_round_complete()
        return True

    def get_state(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        requesting_player = next((p for p in self.players_obj if p.name == player_id), None) if player_id else None
        active_player_obj = self.players_obj[self.active_player_index] if 0 <= self.active_player_index < len(self.players_obj) else None

        # --- Corrected: Return subset of self.history (dicts) ---
        recent_history = []
        if self.history:
            current_hand_no = self.history[-1].get('hand', self.hand_number)
            # Get last N events from the current hand
            count = 10 # Limit history size passed in state
            for event in reversed(self.history):
                if event.get('hand') == current_hand_no:
                    recent_history.append(event)
                    if len(recent_history) >= count:
                        break
            recent_history.reverse() # Put back in chronological order


        state = {
            "game_id": self.game_id, "hand_number": self.hand_number, "stage": self.stage,
            "dealer": self.players_obj[self.dealer_index].name,
            "active_player": active_player_obj.name if active_player_obj and not self.hand_complete else None,
            "pot": self.pot, "community_cards": self.community_cards, "current_bet": self.current_bet,
            "players": {},
            "history": recent_history # Pass list of dicts
        }

        for i, player in enumerate(self.players_obj):
            player_info = {
                "name": player.name, "stack": player.stack, "current_bet": player.current_bet,
                "total_bet_in_hand": player.total_bet_in_hand, "folded": player.folded,
                "all_in": player.all_in, "is_dealer": (i == self.dealer_index),
                "last_action": player.last_action
            }
            if player_id is None or player.name == player_id or self.stage == "showdown":
                player_info["hole_cards"] = player.hole_cards
            else: player_info["hole_cards"] = "Hidden"
            state["players"][player.name] = player_info

        if requesting_player and active_player_obj and requesting_player.name == active_player_obj.name:
             state["valid_actions"] = self.get_valid_actions(player_id)
        elif player_id is None:
             state["valid_actions"] = self.get_valid_actions(active_player_obj.name if active_player_obj else None)

        return state

    # --- Removed unused method ---
    # def get_recent_action_history_tuples(self, count=5) -> List[Tuple[str, str, Any]]: ...

    def get_valid_actions(self, player_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.hand_complete or self.game_complete or self.stage == "showdown": return []
        target_player_index = next((i for i, p in enumerate(self.players_obj) if p.name == player_id), self.active_player_index) if player_id else self.active_player_index
        if target_player_index != self.active_player_index: return []

        player = self.players_obj[target_player_index]
        opponent = self.players_obj[1 - target_player_index]
        if player.folded or player.all_in: return []

        valid_actions = []
        call_amount = self.current_bet - player.current_bet
        effective_call_amount = min(call_amount, player.stack)

        can_check = call_amount <= 0
        can_call = call_amount > 0

        if can_check: valid_actions.append({"action_type": "check"})
        if can_call:
            valid_actions.append({"action_type": "call", "amount": effective_call_amount})
            valid_actions.append({"action_type": "fold"}) # Fold only possible if call needed

        can_raise_technically = player.stack > effective_call_amount and not opponent.all_in
        if can_raise_technically:
            last_bet_or_raise_size = self.big_blind
            if self.last_raiser_index != -1:
                raiser = self.players_obj[self.last_raiser_index]
                opponent_index = 1 - self.last_raiser_index
                opponent_bet = self.players_obj[opponent_index].current_bet
                # Approximate raise size needed (at least BB)
                last_raise_diff = raiser.current_bet - opponent_bet
                last_bet_or_raise_size = max(self.big_blind, last_raise_diff if last_raise_diff > 0 else self.big_blind )

            min_total_bet_after_raise = self.current_bet + last_bet_or_raise_size
            min_raise_amount_to_add = min_total_bet_after_raise - player.current_bet
            # Ensure raise amount itself is at least BB, unless going all-in for less
            min_raise_amount_to_add = max(min_raise_amount_to_add, self.big_blind if player.stack >= call_amount + self.big_blind else 0)

            if player.stack >= min_raise_amount_to_add:
                min_amount_param = min_raise_amount_to_add
                max_amount_param = player.stack - effective_call_amount # Max *additional* chips beyond call
                max_amount_param = player.stack # Max additional is simply stack remaining

                # Ensure min is feasible
                min_amount_param = min(min_amount_param, max_amount_param)

                if max_amount_param >= min_amount_param:
                     valid_actions.append({
                         "action_type": "raise",
                         "min_amount": min_amount_param,
                         "max_amount": max_amount_param
                     })

        logger.debug(f"Player {player.name} valid actions: {valid_actions}")
        return valid_actions

    def apply_action(self, action: Dict[str, Any], player_id: Optional[str] = None) -> bool:
        if self.hand_complete or self.game_complete: return False
        acting_player_index = self.active_player_index
        if player_id and self.players_obj[acting_player_index].name != player_id: return False
        player = self.players_obj[acting_player_index]
        opponent = self.players_obj[1 - acting_player_index]
        if player.folded or player.all_in: self._check_betting_round_complete(); return False

        action_type = action.get("action_type", "").lower()
        amount = action.get("amount")
        valid_actions = self.get_valid_actions(player.name)
        is_valid_choice = any(a["action_type"] == action_type for a in valid_actions)
        logger.info(f"Player {player.name} attempts action: {action_type} {f'(Amount: {amount})' if amount is not None else ''}")

        processed_action_amount = 0

        if action_type == "fold":
            if not is_valid_choice: return False
            player.fold(); player.last_action = "fold"; processed_action_amount = 0
            if not opponent.folded: self._award_pot(opponent); self.hand_complete = True
        elif action_type == "check":
            if not is_valid_choice or player.current_bet < self.current_bet: return False
            player.last_action = "check"; processed_action_amount = 0
        elif action_type == "call":
            if not is_valid_choice: return False
            call_amount_needed = self.current_bet - player.current_bet
            if call_amount_needed <= 0: return False
            actual_call_amount = min(call_amount_needed, player.stack)
            processed_action_amount = player.bet(actual_call_amount)
            self.pot += processed_action_amount; player.last_action = "call"
            if player.all_in: logger.info(f"{player.name} calls {processed_action_amount} and is all-in.")
        elif action_type == "raise":
            if not is_valid_choice or amount is None: return False
            raise_amount = int(amount)
            raise_validation = next((a for a in valid_actions if a["action_type"] == "raise"), None)
            if not raise_validation: return False
            min_raise_add, max_raise_add = raise_validation["min_amount"], raise_validation["max_amount"]
            if not (min_raise_add <= raise_amount <= max_raise_add): return False

            processed_action_amount = player.bet(raise_amount)
            self.pot += processed_action_amount
            self.current_bet = player.current_bet # New level to match
            self.last_raiser_index = acting_player_index
            player.last_action = "raise"; self.aggressor_action_closed = False
            logger.info(f"{player.name} raises by {processed_action_amount}. Total bet this round: {player.current_bet}. New current bet level: {self.current_bet}")
        else: return False

        log_data = {
            "player": player.name, "action": action_type,
            "amount": processed_action_amount if action_type in ['call', 'raise'] else None,
            "stack": player.stack, "player_bet_round": player.current_bet,
            "pot": self.pot, "current_bet_level": self.current_bet
        }
        self.log_event("action", log_data)
        # Pass amount correctly to log_action_result
        log_action_result(player.name, action_type, processed_action_amount if action_type in ['call', 'raise'] else None, self.get_state())

        if not self.hand_complete:
            self.active_player_index = 1 - acting_player_index
            logger.debug(f"Action applied. Switching active player to {self.players_obj[self.active_player_index].name}")
            self._check_betting_round_complete()
        return True

    def _check_betting_round_complete(self):
        if self.hand_complete or self.game_complete: return True
        player1, player2 = self.players_obj[0], self.players_obj[1]
        active_player = self.players_obj[self.active_player_index]
        inactive_player = self.players_obj[1 - self.active_player_index]

        if player1.all_in and player2.all_in: self._advance_to_showdown(); return True
        if player1.folded or player2.folded: return True # Handled elsewhere

        round_closed = False
        if active_player.all_in:
            if inactive_player.current_bet >= active_player.current_bet: round_closed = True
            else: self.active_player_index = 1 - self.active_player_index; return False # Opponent needs to act
        else: # Standard check
            bets_matched = (player1.current_bet == player2.current_bet)
            # Check if player who just acted (inactive_player) closed the action with check/call
            # And ensure both players have had a chance to act (beyond blinds) if not all-in
            acted_beyond_blinds = (player1.last_action is not None or player1.all_in or (self.stage == "pre-flop" and player1.current_bet==self.big_blind)) and \
                                  (player2.last_action is not None or player2.all_in or (self.stage == "pre-flop" and player2.current_bet==self.big_blind))
            # More precise check: Has the action made a full circle back to the aggressor (or BB preflop)?
            action_closed_round = False
            if self.last_raiser_index != -1: # There was a raise
                 # Action closes if non-raiser calls
                 if inactive_player.last_action == "call" and self.active_player_index == self.last_raiser_index:
                      action_closed_round = True
            else: # No raise yet this round (only blinds or checks/calls)
                 # Action closes if BB checks preflop, or if player checks/calls after opponent checked/called
                 is_preflop_bb_check = self.stage == "pre-flop" and inactive_player.last_action == "check" and inactive_player == self.players_obj[1-self.dealer_index]
                 # Or if player checks back after opponent checked, or calls after opponent checked/called blind
                 action_returned = inactive_player.last_action in ["check", "call"]
                 # Ensure both put in same amount
                 if bets_matched and (is_preflop_bb_check or action_returned):
                      # Need to ensure the betting wasn't just the blinds being posted
                      if player1.total_bet_in_hand > 0 or player2.total_bet_in_hand > 0: # Check total bet in hand
                          # Check if at least one voluntary action happened
                          if player1.last_action or player2.last_action:
                              action_closed_round = True


            if bets_matched and action_closed_round:
                 logger.info(f"Betting round ends: Bets matched ({player1.current_bet}) and action closed.")
                 round_closed = True

        if round_closed:
            logger.info(f"--- End of Betting Round ({self.stage}) --- Pot: {self.pot}")
            log_event_to_game_file(f"--- End of Betting Round ({self.stage}) --- Pot: {self.pot}\n")
            if self.stage == "river": self._advance_to_showdown()
            else: self.deal_community_cards()
            return True

        # Check if next player can act
        current_active_player = self.players_obj[self.active_player_index]
        if current_active_player.folded or current_active_player.all_in:
             other_player = self.players_obj[1 - self.active_player_index]
             if other_player.folded or other_player.all_in or other_player.current_bet == self.current_bet:
                  if self.stage == "river": self._advance_to_showdown()
                  else: self.deal_community_cards()
                  return True
             else: self.active_player_index = 1 - self.active_player_index; return False # Switch back

        logger.debug(f"Betting round continues. Active player: {self.players_obj[self.active_player_index].name}")
        return False

    def _advance_to_showdown(self):
        logger.info("Advancing to showdown.")
        while self.stage != "river" and not self.game_complete and len(self.community_cards) < 5:
            next_stage = "flop" if self.stage == "pre-flop" else ("turn" if self.stage == "flop" else "river")
            num_to_deal = 3 if next_stage=="flop" else 1
            try:
                if len(self.deck.cards) > num_to_deal: self.deck.draw(1) # Burn
                new_cards = self.deck.draw(num_to_deal)
                self.community_cards.extend(new_cards)
                logger.info(f"Auto-dealing {next_stage}: {format_cards(new_cards)}")
                log_event_to_game_file(f"Auto-dealing {next_stage}: {format_cards(self.community_cards)}\n")
                self.log_event(f"{next_stage}_dealt_auto", {"cards": new_cards, "community_total": self.community_cards})
                self.stage = next_stage
            except ValueError as e: logger.error(f"Error auto-dealing cards: {e}"); self.game_complete = True; break
        self.stage = "river" # Ensure stage is river before handling showdown
        self._handle_showdown()

    def _handle_showdown(self):
        if self.hand_complete: return
        logger.info("--- Showdown ---")
        self.stage = "showdown"
        player1, player2 = self.players_obj[0], self.players_obj[1]

        if player1.folded and not player2.folded: self._award_pot(player2); log_data = self._create_hand_result_log(winner=player2, reason="opponent_folded")
        elif not player1.folded and player2.folded: self._award_pot(player1); log_data = self._create_hand_result_log(winner=player1, reason="opponent_folded")
        elif player1.folded and player2.folded: self._split_pot(); log_data = self._create_hand_result_log(winner=None, reason="error_both_folded")
        else:
            winner, hand1_rank, hand2_rank = HandEvaluator.determine_winner(player1, player2, self.community_cards)
            hand1_desc = HandEvaluator.hand_description(hand1_rank)
            hand2_desc = HandEvaluator.hand_description(hand2_rank)
            logger.info(f"{player1.name}: {format_cards(player1.hole_cards)} -> {hand1_desc}")
            logger.info(f"{player2.name}: {format_cards(player2.hole_cards)} -> {hand2_desc}")
            log_event_to_game_file(f"Showdown:\n  {player1.name}: {format_cards(player1.hole_cards)} -> {hand1_desc}\n  {player2.name}: {format_cards(player2.hole_cards)} -> {hand2_desc}\n")
            self.log_event("showdown", { "hand": self.hand_number, "community_cards": self.community_cards, "players": { player1.name: {"hole_cards": player1.hole_cards, "hand_rank": hand1_rank, "hand_description": hand1_desc}, player2.name: {"hole_cards": player2.hole_cards, "hand_rank": hand2_rank, "hand_description": hand2_desc} } })

            if winner is None: self._split_pot(); log_data = self._create_hand_result_log(winner=None, reason="tie_showdown", hand1_desc=hand1_desc, hand2_desc=hand2_desc)
            else: self._award_pot(winner); log_data = self._create_hand_result_log(winner=winner, reason="better_hand", winning_hand_desc=(hand1_desc if winner == player1 else hand2_desc))

        self.log_event("hand_result", log_data)
        self.hand_complete = True
        log_hand_result(self.get_state(), self.hand_number) # Log final state after pot award

    def _award_pot(self, winner: Player):
        logger.debug(f"Awarding pot of {self.pot} to {winner.name}")
        winner.stack += self.pot; self.pot = 0

    def _split_pot(self):
        logger.debug(f"Splitting pot of {self.pot}")
        split_amount, remainder = divmod(self.pot, 2)
        self.players_obj[0].stack += split_amount
        self.players_obj[1].stack += split_amount
        if remainder > 0:
            non_dealer_index = 1 - self.dealer_index
            self.players_obj[non_dealer_index].stack += remainder
            logger.debug(f"Awarding remainder {remainder} to {self.players_obj[non_dealer_index].name}")
        self.pot = 0

    def _create_hand_result_log(self, winner: Optional[Player], reason: str, **kwargs) -> Dict[str, Any]:
         # Get pot size *before* awarding/splitting
         pot_before_award = sum(p.total_bet_in_hand for p in self.players_obj)
         # Stacks *after* awarding pot will be reflected when get_state is called later
         result_data = {
             "hand": self.hand_number, "winner": winner.name if winner else None,
             "reason": reason, "pot_awarded": pot_before_award,
             # Final stacks will be logged by log_hand_result using get_state
         }
         if winner is None and reason == "tie_showdown":
              result_data["hand1_desc"] = kwargs.get("hand1_desc"); result_data["hand2_desc"] = kwargs.get("hand2_desc")
         elif winner and reason == "better_hand":
              result_data["winning_hand_desc"] = kwargs.get("winning_hand_desc")
         return result_data

    def is_terminal(self) -> bool:
        if not self.game_complete and any(p.stack <= 0 for p in self.players_obj):
            self.game_complete = True; logger.info("Game is terminal: A player has zero stack.")
        return self.game_complete

    def get_rewards(self) -> Dict[str, float]:
        if not self.is_terminal(): return {p.name: 0.0 for p in self.players_obj}
        return {p.name: p.stack - self.starting_stack for p in self.players_obj}

    def get_result(self) -> Dict[str, Any]:
        if not self.is_terminal(): return {"status": "in_progress"}
        winner = "Tie"
        if self.players_obj[0].stack > self.players_obj[1].stack: winner = self.players_obj[0].name
        elif self.players_obj[1].stack > self.players_obj[0].stack: winner = self.players_obj[1].name
        return { "status": "complete", "winner": winner, "players": { p.name: { "final_stack": p.stack, "net_winnings": p.stack - self.starting_stack } for p in self.players_obj }, "total_hands": self.hand_number }

    def log_event(self, event_type: str, data: Dict[str, Any]):
        log_entry = { 'timestamp': time.time(), 'hand': self.hand_number, 'stage': self.stage, 'event_type': event_type, 'data': data }
        self.history.append(log_entry)

    # --- Persistence Methods ---
    def _restore_from_state(self, state_data: Dict[str, Any]):
        # Simplified restore - assumes basic attributes are present
        self.game_id = state_data.get('game_id', self.game_id)
        # ... restore other attributes like players, seed, blinds, etc. ...
        self._current_seed = state_data.get('random_seed')
        self._rng = random.Random(self._current_seed)
        self.pot = state_data.get('pot', 0)
        self.community_cards = state_data.get('community_cards', [])
        # ... restore player objects using Player.from_dict ...
        self.players_obj = [Player.from_dict(p_data) for p_data in state_data.get('players_data', [])]
        self.history = state_data.get('history', [])
        # ... restore stage, hand_number, indices, etc. ...
        self.stage = state_data.get('stage', 'pre-deal')
        self.hand_number = state_data.get('hand_number', 0)
        self.dealer_index = state_data.get('dealer_index', 0)
        self.active_player_index = state_data.get('active_player_index', -1)
        self.last_raiser_index = state_data.get('last_raiser_index', -1)
        self.hand_complete = state_data.get('hand_complete', False)
        self.game_complete = state_data.get('game_complete', False)
        logger.info(f"Game state restored for ID {self.game_id}")

    def to_json(self) -> str:
        state = {
            'game_id': self.game_id, 'players': self.players, 'random_seed': self._current_seed,
            'player1_name': self.player1_name, 'player2_name': self.player2_name,
            'starting_stack': self.starting_stack, 'small_blind': self.small_blind, 'big_blind': self.big_blind,
            'pot': self.pot, 'community_cards': self.community_cards, 'current_bet': self.current_bet,
            'dealer_index': self.dealer_index, 'active_player_index': self.active_player_index,
            'last_raiser_index': self.last_raiser_index, 'stage': self.stage, 'hand_number': self.hand_number,
            'hand_complete': self.hand_complete, 'game_complete': self.game_complete, 'aggressor_action_closed': self.aggressor_action_closed,
            'deck': self.deck.to_dict(), 'players_data': [p.to_dict() for p in self.players_obj],
            'history': self.history, 'is_interactive': self.is_interactive
        }
        # Use default=str for non-serializable items if any (like timestamps if not float)
        return json.dumps(state, indent=2, default=str)


def run_non_interactive_game( game: HeadsUpPoker, player1_agent, player2_agent, num_hands: int ) -> Dict[str, Any]:
    if not player1_agent or not player2_agent: raise ValueError("Both player agents must be provided")
    if game.is_interactive: logger.warning("Running non-interactive game loop on a game marked as interactive."); game.set_interactive_mode(False)
    agents = {game.player1_name: player1_agent, game.player2_name: player2_agent}
    logger.info(f"Starting non-interactive game run for {num_hands} hands.")

    while game.hand_number < num_hands and not game.is_terminal():
        game.reset_hand()
        if game.is_terminal(): break
        if not game.start_hand(): logger.error(f"Failed to start hand {game.hand_number}. Aborting game."); break

        while not game.hand_complete and not game.is_terminal():
            active_player_obj = game.players_obj[game.active_player_index]
            active_player_name = active_player_obj.name

            if active_player_obj.folded or active_player_obj.all_in: game._check_betting_round_complete(); continue

            state_for_agent = game.get_state(player_id=active_player_name)
            valid_actions = state_for_agent.get("valid_actions", [])

            if not valid_actions:
                 logger.warning(f"No valid actions for {active_player_name}. State: {game.stage}, Bet: {game.current_bet}, PlayerBet: {active_player_obj.current_bet}")
                 if game._check_betting_round_complete(): continue
                 else: logger.error("Stuck state: No valid actions but round not complete. Forcing fold."); action = {"action_type": "fold"}
            else:
                agent_func = agents[active_player_name]
                try: action = agent_func(state_for_agent, valid_actions); assert action is not None
                except Exception as e:
                    logger.error(f"Agent {active_player_name} failed: {e}. Using fallback.")
                    from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                    action = PokerResponseParser().get_fallback_action(valid_actions)

            success = game.apply_action(action, active_player_name)
            if not success:
                logger.warning(f"Agent {active_player_name} provided invalid action: {action}. Applying fallback.")
                from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                fallback_action = PokerResponseParser().get_fallback_action(valid_actions)
                game.apply_action(fallback_action, active_player_name) # Apply fallback

    logger.info(f"Non-interactive game run finished after {game.hand_number} hands.")
    close_game_log()
    final_result = game.get_result()
    final_result["hands_played"] = game.hand_number
    final_result["players"] = { p.name: { "final_stack": p.stack, "net_winnings": p.stack - game.starting_stack } for p in game.players_obj }
    # final_result["hand_results_summary"] = "See game log for details"
    return final_result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from model_orchestrator.utils import init_game_log
    init_game_log("./logs")
    def random_agent(state, valid_actions):
        chosen = random.choice(valid_actions)
        if chosen["action_type"] == "raise": chosen["amount"] = random.randint(chosen["min_amount"], chosen["max_amount"])
        logger.debug(f"RandomAgent chooses: {chosen}")
        return chosen
    test_game = HeadsUpPoker(random_seed=42, starting_stack=500, small_blind=5, big_blind=10)
    result = run_non_interactive_game(game=test_game, player1_agent=random_agent, player2_agent=random_agent, num_hands=5)
    print("\n--- FINAL GAME RESULT ---"); print(json.dumps(result, indent=2))