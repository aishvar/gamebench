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
        self.aggressor_action_closed = False # Flag to track betting round closure
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
        self.aggressor_action_closed = False # Reset for new hand
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
        # BB is considered the last 'raiser' initially if their bet is >= SB
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
        self.active_player_index = sb_player_index # SB acts first pre-flop
        self.aggressor_action_closed = False # Initialize for pre-flop betting round
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
        self.current_bet = 0 # Reset bet level for the new round
        self.last_raiser_index = -1 # Reset last raiser for the new round
        self.aggressor_action_closed = False # Reset for new betting round
        for player in self.players_obj: player.current_bet = 0; player.last_action = None # Reset player's bet amount this round

        # Action starts with the player out of position (small blind post-flop)
        self.active_player_index = 1 - self.dealer_index
        # Skip player if they are all-in or folded
        if self.players_obj[self.active_player_index].all_in or self.players_obj[self.active_player_index].folded:
            self.active_player_index = 1 - self.active_player_index # Switch to the other player

        logger.info(f"Stage is now {self.stage}. Active player: {self.players_obj[self.active_player_index].name}. Bets reset.")
        log_event_to_game_file(f"--- {self.stage.upper()} --- Active: {self.players_obj[self.active_player_index].name}\n")

        # Important: Check immediately if betting is even possible or if round completes instantly
        self._check_betting_round_complete()
        return True

    def get_state(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        requesting_player = next((p for p in self.players_obj if p.name == player_id), None) if player_id else None
        active_player_obj = self.players_obj[self.active_player_index] if 0 <= self.active_player_index < len(self.players_obj) else None

        # --- Corrected: Return subset of self.history (dicts) ---
        recent_history = []
        if self.history:
            # Determine the hand number of the most recent event
            current_hand_no = self.hand_number
            if self.history[-1].get('hand'):
                 current_hand_no = self.history[-1]['hand']

            # Get last N events from the current hand
            count = 10 # Limit history size passed in state
            for event in reversed(self.history):
                if event.get('hand') == current_hand_no:
                    recent_history.append(event)
                    if len(recent_history) >= count:
                        break
                # Stop if we go back to a previous hand's event
                elif event.get('hand') is not None and event.get('hand') < current_hand_no:
                     break
            recent_history.reverse() # Put back in chronological order


        state = {
            "game_id": self.game_id, "hand_number": self.hand_number, "stage": self.stage,
            "dealer": self.players_obj[self.dealer_index].name,
            "active_player": active_player_obj.name if active_player_obj and not self.hand_complete and not self.game_complete else None,
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
            # Reveal cards only to the owner, or at showdown
            if player_id is None or player.name == player_id or (self.stage == "showdown" and not player.folded):
                player_info["hole_cards"] = player.hole_cards
            else: player_info["hole_cards"] = "Hidden" # Keep hidden otherwise
            state["players"][player.name] = player_info

        # Add valid actions only if it's the requesting player's turn (or if no specific player requested)
        if not self.hand_complete and not self.game_complete:
            if player_id is None: # General state view
                 if active_player_obj:
                      state["valid_actions"] = self.get_valid_actions(active_player_obj.name)
            elif requesting_player and active_player_obj and requesting_player.name == active_player_obj.name:
                 state["valid_actions"] = self.get_valid_actions(player_id)

        return state

    def get_valid_actions(self, player_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.hand_complete or self.game_complete or self.stage == "showdown": return []

        # Determine the target player index
        target_player_index = -1
        if player_id:
            try:
                target_player_index = next(i for i, p in enumerate(self.players_obj) if p.name == player_id)
            except StopIteration:
                logger.warning(f"get_valid_actions called for unknown player_id: {player_id}")
                return []
        elif 0 <= self.active_player_index < len(self.players_obj):
            target_player_index = self.active_player_index
        else: # No valid active player index
             return []

        # Check if it's actually this player's turn
        if target_player_index != self.active_player_index:
             # logger.debug(f"Not player {player_id}'s turn (active: {self.players_obj[self.active_player_index].name})")
             return []

        player = self.players_obj[target_player_index]
        opponent = self.players_obj[1 - target_player_index]

        # No actions if folded or already all-in
        if player.folded or player.all_in: return []

        valid_actions = []
        call_amount = self.current_bet - player.current_bet # Amount needed *additional* to call
        effective_call_amount = min(call_amount, player.stack) # Can't call more than stack

        can_check = call_amount <= 0
        can_call = call_amount > 0

        # --- CHECK ---
        if can_check:
            valid_actions.append({"action_type": "check"})

        # --- CALL / FOLD ---
        if can_call:
            valid_actions.append({"action_type": "call", "amount": effective_call_amount})
            valid_actions.append({"action_type": "fold"}) # Fold is always an option if a bet is faced

        # --- RAISE ---
        # Conditions: Player not all-in after call, opponent not all-in
        can_raise_technically = player.stack > effective_call_amount and not opponent.all_in

        if can_raise_technically:
            # Determine the minimum raise amount allowed
            # In heads-up, the minimum raise is typically the size of the last bet or raise,
            # but at least the big blind.

            # Size of the previous bet/raise in this round
            last_bet_or_raise_size = self.big_blind # Default minimum raise size
            if self.last_raiser_index != -1:
                # Calculate the amount the last raiser *added* to the previous bet level
                raiser = self.players_obj[self.last_raiser_index]
                # Find the bet level *before* the last raise
                bet_before_last_raise = 0
                if self.stage == "pre-flop":
                    # Special handling for blinds
                    sb_idx, bb_idx = self.dealer_index, 1 - self.dealer_index
                    if self.last_raiser_index == bb_idx: # BB made the first 'raise' (the BB itself)
                         bet_before_last_raise = self.players_obj[sb_idx].current_bet # SB's bet
                    else: # SB raised over BB or later raise
                         # Need history or more state to know precisely? Assume opponent's bet
                         bet_before_last_raise = self.players_obj[1-self.last_raiser_index].current_bet

                else: # Post-flop rounds start with 0 bets
                    # Find opponent's bet level *at the time of the raise*
                    # This requires more complex history tracking. Simpler: Find opponent's *current* bet
                    # as that was the level the raise was made over.
                     bet_before_last_raise = self.players_obj[1-self.last_raiser_index].current_bet

                raise_diff = raiser.current_bet - bet_before_last_raise
                last_bet_or_raise_size = max(self.big_blind, raise_diff if raise_diff > 0 else self.big_blind)


            # Min TOTAL bet player must make for a valid raise
            min_total_bet_after_raise = self.current_bet + last_bet_or_raise_size

            # Min amount player needs to ADD to their current bet
            min_raise_amount_to_add = min_total_bet_after_raise - player.current_bet
            # Ensure the added amount is positive (it should be if raising)
            min_raise_amount_to_add = max(1, min_raise_amount_to_add)

            # Can the player actually raise by this minimum amount?
            if player.stack >= min_raise_amount_to_add + effective_call_amount:
                 min_amount_param = min_raise_amount_to_add # The minimum *additional* amount
                 max_amount_param = player.stack - effective_call_amount # The maximum *additional* amount (all-in)

                 # Ensure min is not more than max (e.g., if going all-in is less than standard min raise)
                 if min_amount_param <= max_amount_param:
                      valid_actions.append({
                          "action_type": "raise",
                          "min_amount": min_amount_param,
                          "max_amount": max_amount_param
                      })
            # Special case: Player can go all-in, but it's less than a standard min-raise
            elif player.stack > effective_call_amount: # They have *something* left to raise with
                 all_in_raise_amount = player.stack - effective_call_amount
                 # Check if this all-in amount is legal (must be >= call amount)
                 if all_in_raise_amount > 0:
                     valid_actions.append({
                         "action_type": "raise",
                         "min_amount": all_in_raise_amount,
                         "max_amount": all_in_raise_amount # All-in is the only option
                     })


        logger.debug(f"Player {player.name} valid actions: {valid_actions}")
        return valid_actions

    def apply_action(self, action: Dict[str, Any], player_id: Optional[str] = None) -> bool:
        if self.hand_complete or self.game_complete: return False

        acting_player_index = self.active_player_index
        # Verify player_id matches the active player if provided
        if player_id and (acting_player_index == -1 or self.players_obj[acting_player_index].name != player_id):
            logger.error(f"Action applied by non-active player? Expected {self.players_obj[acting_player_index].name if acting_player_index != -1 else 'N/A'}, got {player_id}")
            return False
        if acting_player_index == -1: logger.error("apply_action called with no active player index."); return False

        player = self.players_obj[acting_player_index]
        opponent = self.players_obj[1 - acting_player_index]

        # Skip action if player can't act
        if player.folded or player.all_in:
            logger.warning(f"Attempted action by player {player.name} who is folded or all-in.")
            # Check completion just in case state is weird
            self._check_betting_round_complete()
            return False

        action_type = action.get("action_type", "").lower()
        amount = action.get("amount")
        valid_actions = self.get_valid_actions(player.name)
        # Find the specific valid action dict matching the type chosen
        chosen_valid_action = next((a for a in valid_actions if a["action_type"] == action_type), None)

        logger.info(f"Player {player.name} attempts action: {action_type} {f'(Amount: {amount})' if amount is not None else ''}")

        if not chosen_valid_action:
             logger.warning(f"Invalid action type '{action_type}' chosen by {player.name}. Valid: {[a['action_type'] for a in valid_actions]}")
             return False # Action type itself is invalid for current state

        processed_action_amount = 0
        action_closes_round = False # Reset flag

        if action_type == "fold":
            player.fold(); player.last_action = "fold"; processed_action_amount = 0
            # Award pot immediately if opponent hasn't folded
            if not opponent.folded:
                self._award_pot(opponent)
                self.hand_complete = True
                # Log result immediately after awarding pot
                log_data = self._create_hand_result_log(winner=opponent, reason="opponent_folded")
                self.log_event("hand_result", log_data)
                log_hand_result(self.get_state(), self.hand_number)


        elif action_type == "check":
            # Validation: Check is only valid if call_amount <= 0
            call_amount_needed = self.current_bet - player.current_bet
            if call_amount_needed > 0:
                 logger.warning(f"Invalid Check by {player.name}: Bet of {call_amount_needed} faced.")
                 return False
            player.last_action = "check"; processed_action_amount = 0
            # Checking doesn't reset aggression flag but might close round (checked later)

        elif action_type == "call":
            call_amount_needed = self.current_bet - player.current_bet
            # Validation: Call is only valid if call_amount > 0
            if call_amount_needed <= 0:
                logger.warning(f"Invalid Call by {player.name}: No bet to call.")
                return False
            # Use the effective call amount from the valid action list
            actual_call_amount = chosen_valid_action.get("amount", 0)
            # Double check against stack
            actual_call_amount = min(actual_call_amount, player.stack)

            processed_action_amount = player.bet(actual_call_amount)
            self.pot += processed_action_amount; player.last_action = "call"
            if player.all_in: logger.info(f"{player.name} calls {processed_action_amount} and is all-in.")

            # Determine if calling closes the round
            if self.last_raiser_index != -1: # If calling a raise
                 action_closes_round = True
            # Special pre-flop case: SB calling BB's initial blind doesn't close action yet
            elif self.stage == "pre-flop" and acting_player_index == self.dealer_index:
                 action_closes_round = False
            else: # Calling a bet that wasn't a raise (e.g., limped pot, BB calls SB's post-flop bet)
                 action_closes_round = True # Assume calling closes it unless it's SB preflop

            self.aggressor_action_closed = action_closes_round


        elif action_type == "raise":
            if amount is None:
                 logger.warning(f"Raise action by {player.name} missing amount.")
                 return False # Amount required for raise action
            raise_amount = int(amount) # The amount to ADD

            # Validate amount against the min/max from valid_actions
            min_raise_add = chosen_valid_action.get("min_amount", 0)
            max_raise_add = chosen_valid_action.get("max_amount", float('inf'))

            if not (min_raise_add <= raise_amount <= max_raise_add):
                logger.warning(f"Invalid raise amount {raise_amount} by {player.name}. Valid: {min_raise_add}-{max_raise_add}")
                return False

            # Calculate total bet needed (call + raise amount)
            call_amount_needed = max(0, self.current_bet - player.current_bet)
            total_bet_this_action = call_amount_needed + raise_amount
            total_bet_this_action = min(total_bet_this_action, player.stack) # Ensure not betting more than stack

            # Process the bet
            processed_action_amount = player.bet(total_bet_this_action)
            self.pot += processed_action_amount
            self.current_bet = player.current_bet # New bet level for opponent to match
            self.last_raiser_index = acting_player_index # This player is now the aggressor
            player.last_action = "raise";
            self.aggressor_action_closed = False # Raising re-opens the action
            logger.info(f"{player.name} raises by {raise_amount} (total bet: {processed_action_amount}). Total bet this round: {player.current_bet}. New current bet level: {self.current_bet}")

        else:
             logger.error(f"Unhandled action type: {action_type}")
             return False


        # --- Logging and State Update ---
        if not self.hand_complete: # Don't switch player or check round if hand ended (e.g., fold)
            # Switch player FIRST
            next_player_index = 1 - acting_player_index
            self.active_player_index = next_player_index
            logger.debug(f"Action applied by {player.name}. Switching active player to {self.players_obj[next_player_index].name}")

            # Log event with context about who acted and who is next
            log_data = {
                "player": player.name, # Player who acted
                "action": action_type,
                "amount": processed_action_amount if action_type in ['call', 'raise'] else None,
                "player_bet_round": player.current_bet,
                "player_stack": player.stack,
                "pot": self.pot,
                "current_bet_level": self.current_bet,
                # Add next player info to the event data itself for clarity
                "next_player": self.players_obj[next_player_index].name
            }
            self.log_event("action", log_data)

            # Log formatted state AFTER switching
            log_action_result(player.name, action_type, processed_action_amount if action_type in ['call', 'raise'] else None, self.get_state())

            # Check if the betting round is now complete AFTER the state update
            self._check_betting_round_complete()

        return True

    def _check_betting_round_complete(self):
        """ Checks if the current betting round should end and advances the game stage if so. """
        if self.hand_complete or self.game_complete: return True # Hand/Game already over
        player1, player2 = self.players_obj[0], self.players_obj[1]

        # --- Trivial End Conditions ---
        if player1.folded or player2.folded:
            # Hand outcome is determined elsewhere (_handle_showdown or _award_pot in apply_action)
            # No more betting is possible.
            if not self.hand_complete:
                 logger.warning("_check_betting_round_complete called when a player folded but hand not marked complete.")
                 # Force showdown/award if needed
                 self._handle_showdown()
            return True

        if player1.all_in and player2.all_in:
            logger.debug("Both players all-in, advancing to showdown.")
            self._advance_to_showdown()
            return True

        # --- Check Standard Betting Completion ---
        bets_matched = (player1.current_bet == player2.current_bet)

        round_closed = False
        if bets_matched:
            # Action completes if the aggressor's action is closed (by a call) OR
            # if the Big Blind checks pre-flop OR if a player checks back post-flop.

            current_player_index = self.active_player_index
            current_player = self.players_obj[current_player_index]
            last_action_player = self.players_obj[1 - current_player_index] # Player who just acted

            # Case 1: Aggression was closed by the last action (typically a call)
            if self.aggressor_action_closed:
                 # Check if the player whose turn it is now *was* the aggressor or if round started with checks
                 # Basically, did action circle back and get closed?
                 if last_action_player.last_action == "call": # Action closed by call
                      round_closed = True
                 # If pre-flop BB checks after SB call/limp
                 elif self.stage == "pre-flop" and current_player_index == (1 - self.dealer_index) and current_player.last_action == "check":
                      round_closed = True
                 # If post-flop player checks back
                 elif self.stage != "pre-flop" and current_player.last_action == "check" and last_action_player.last_action == "check":
                      round_closed = True


            # Case 2: Player whose turn it is is all-in (opponent must have matched)
            elif current_player.all_in:
                 round_closed = True

            # Case 3: Specific Pre-flop BB check scenario (handled slightly differently by aggressor_action_closed logic now)
            # Ensure pre-flop BB check closes action
            if not round_closed and self.stage == "pre-flop" and \
               current_player_index == (1 - self.dealer_index) and \
               current_player.last_action == "check" and \
               last_action_player.last_action in ["call", "check"]: # SB must have called or checked (limped)
                logger.debug("Pre-flop BB check closes action.")
                round_closed = True

             # Case 4: Specific Post-flop check-back scenario
            if not round_closed and self.stage != "pre-flop" and \
               current_player.last_action == "check" and \
               last_action_player.last_action == "check":
                logger.debug("Post-flop check-back closes action.")
                round_closed = True


        # --- Advance Game if Round Closed ---
        if round_closed:
            logger.info(f"--- End of Betting Round ({self.stage}) --- Pot: {self.pot}")
            log_event_to_game_file(f"--- End of Betting Round ({self.stage}) --- Pot: {self.pot}\n")
            if self.stage == "river":
                self._advance_to_showdown()
            else:
                # Check if further actions are possible or if we should skip to showdown
                p1_can_act = not (player1.folded or player1.all_in)
                p2_can_act = not (player2.folded or player2.all_in)
                if not (p1_can_act and p2_can_act): # If at least one player cannot act further
                     logger.debug("One player cannot act further, advancing to showdown after round end.")
                     self._advance_to_showdown()
                else:
                     self.deal_community_cards() # Deal next street
            return True # Round is complete

        # --- If Round Not Closed, Ensure Active Player Can Act ---
        # If the current active player is folded or all-in, they can't act.
        # The round should continue ONLY if the other player still needs to act (e.g. match an all-in)
        current_active_player = self.players_obj[self.active_player_index]
        if current_active_player.folded or current_active_player.all_in:
             other_player = self.players_obj[1 - self.active_player_index]
             # If the other player is also folded/all-in or has matched the bet, the round should have ended above.
             if other_player.folded or other_player.all_in or other_player.current_bet == current_active_player.current_bet:
                 logger.warning("Round check inconsistency: Active player cannot act, but opponent seems matched/done. Forcing round end check again.")
                 # This state indicates a potential logic flaw elsewhere or an edge case.
                 # Re-run the logic that should lead to showdown/dealing.
                 if self.stage == "river": self._advance_to_showdown()
                 else: self._advance_to_showdown() # Safer to go to showdown if state is weird
                 return True
             else:
                 # This means the current player is stuck, but the opponent *still* needs to act.
                 # This shouldn't happen if the active player switching is correct.
                 logger.error("Critical State Error: Active player cannot act, but opponent still needs to. Check player switching logic.")
                 # As a fallback, maybe force switch? But this hides the root cause.
                 # self.active_player_index = 1 - self.active_player_index
                 return False # Indicate round continues, hoping the next call fixes it.

        logger.debug(f"Betting round continues. Active player: {self.players_obj[self.active_player_index].name}")
        return False # Round continues


    def _advance_to_showdown(self):
        if self.hand_complete: return # Avoid advancing if already done

        logger.info("Advancing to showdown (dealing remaining cards if necessary).")
        # Deal remaining community cards automatically if not yet river
        while self.stage != "river" and not self.game_complete and len(self.community_cards) < 5:
            # Determine next stage and cards to deal
            next_stage = ""
            num_to_deal = 0
            if self.stage == "pre-flop": next_stage, num_to_deal = "flop", 3
            elif self.stage == "flop": next_stage, num_to_deal = "turn", 1
            elif self.stage == "turn": next_stage, num_to_deal = "river", 1
            else: break # Should not happen

            try:
                # Burn card if deck allows
                if len(self.deck.cards) > num_to_deal: self.deck.draw(1) # Burn
                new_cards = self.deck.draw(num_to_deal)
                self.community_cards.extend(new_cards)
                logger.info(f"Auto-dealing {next_stage}: {format_cards(new_cards)} -> Community: {format_cards(self.community_cards)}")
                log_event_to_game_file(f"Auto-dealing {next_stage}: {format_cards(self.community_cards)}\n")
                self.log_event(f"{next_stage}_dealt_auto", {"hand": self.hand_number, "cards": new_cards, "community_total": self.community_cards})
                self.stage = next_stage
            except ValueError as e: # Not enough cards
                logger.error(f"Error auto-dealing cards for {next_stage}: {e}. Deck empty?")
                self.game_complete = True # Mark game as potentially broken
                break # Stop dealing

        # Ensure stage is marked as river for showdown logic, even if dealing failed
        self.stage = "river"
        # Proceed to evaluate hands
        self._handle_showdown()

    def _handle_showdown(self):
        if self.hand_complete: return # Avoid double processing
        logger.info("--- Showdown ---")
        self.stage = "showdown" # Explicitly set stage
        player1, player2 = self.players_obj[0], self.players_obj[1]

        # Determine winner based on folds or hand evaluation
        log_data = {}
        winner = None
        reason = ""
        winning_hand_desc = ""

        if player1.folded and not player2.folded:
            winner = player2; reason = "opponent_folded"
            self._award_pot(winner)
        elif not player1.folded and player2.folded:
            winner = player1; reason = "opponent_folded"
            self._award_pot(winner)
        elif player1.folded and player2.folded: # Should not happen
             reason = "error_both_folded"
             self._split_pot() # Safest fallback
        else: # Both players active, evaluate hands
            winner_obj, hand1_rank, hand2_rank = HandEvaluator.determine_winner(player1, player2, self.community_cards)
            hand1_desc = HandEvaluator.hand_description(hand1_rank)
            hand2_desc = HandEvaluator.hand_description(hand2_rank)

            logger.info(f"{player1.name}: {format_cards(player1.hole_cards)} -> {hand1_desc}")
            logger.info(f"{player2.name}: {format_cards(player2.hole_cards)} -> {hand2_desc}")
            log_event_to_game_file(f"Showdown:\n  {player1.name}: {format_cards(player1.hole_cards)} -> {hand1_desc}\n  {player2.name}: {format_cards(player2.hole_cards)} -> {hand2_desc}\n")
            self.log_event("showdown", {
                "hand": self.hand_number,
                "community_cards": self.community_cards,
                "players": {
                     player1.name: {"hole_cards": player1.hole_cards, "hand_rank": hand1_rank, "hand_description": hand1_desc},
                     player2.name: {"hole_cards": player2.hole_cards, "hand_rank": hand2_rank, "hand_description": hand2_desc}
                }
            })

            if winner_obj is None: # Tie
                reason = "tie_showdown"
                self._split_pot()
                log_data = self._create_hand_result_log(winner=None, reason=reason, hand1_desc=hand1_desc, hand2_desc=hand2_desc)
            else: # Clear winner
                winner = winner_obj
                reason = "better_hand"
                winning_hand_desc = hand1_desc if winner == player1 else hand2_desc
                self._award_pot(winner)
                log_data = self._create_hand_result_log(winner=winner, reason=reason, winning_hand_desc=winning_hand_desc)

        # If log_data wasn't set in showdown evaluation (e.g., fold scenario), create it now
        if not log_data:
             log_data = self._create_hand_result_log(winner=winner, reason=reason)

        self.log_event("hand_result", log_data)
        self.hand_complete = True
        log_hand_result(self.get_state(), self.hand_number) # Log final state after pot award

    def _award_pot(self, winner: Player):
        logger.info(f"Awarding pot of {self.pot} to {winner.name}")
        winner.stack += self.pot; self.pot = 0

    def _split_pot(self):
        logger.info(f"Splitting pot of {self.pot}")
        split_amount, remainder = divmod(self.pot, 2)
        self.players_obj[0].stack += split_amount
        self.players_obj[1].stack += split_amount
        if remainder > 0:
            # Award remainder to the player out of position (non-dealer in heads-up post-flop)
            # Pre-flop it's usually the player closest to dealer's left (SB)
            # Using non-dealer index is consistent post-flop.
            non_dealer_index = 1 - self.dealer_index
            self.players_obj[non_dealer_index].stack += remainder
            logger.debug(f"Awarding remainder {remainder} to {self.players_obj[non_dealer_index].name}")
        self.pot = 0

    def _create_hand_result_log(self, winner: Optional[Player], reason: str, **kwargs) -> Dict[str, Any]:
         # Get pot size *before* awarding/splitting
         pot_before_award = sum(p.total_bet_in_hand for p in self.players_obj) # More accurate pot size
         # Stacks *after* awarding pot will be reflected when get_state is called later
         result_data = {
             "hand": self.hand_number,
             "winner": winner.name if winner else None, # Use None for tie
             "reason": reason,
             "pot_awarded": pot_before_award,
             # Final stacks will be logged by log_hand_result using get_state
         }
         # Add hand descriptions if available from kwargs
         if "hand1_desc" in kwargs: result_data["hand1_desc"] = kwargs["hand1_desc"]
         if "hand2_desc" in kwargs: result_data["hand2_desc"] = kwargs["hand2_desc"]
         if "winning_hand_desc" in kwargs: result_data["winning_hand_desc"] = kwargs["winning_hand_desc"]
         return result_data

    def is_terminal(self) -> bool:
        if not self.game_complete and any(p.stack <= 0 for p in self.players_obj):
            self.game_complete = True; logger.info("Game is terminal: A player has zero stack.")
        return self.game_complete

    def get_rewards(self) -> Dict[str, float]:
        # Rewards are typically only meaningful at the end of the game
        if not self.is_terminal(): return {p.name: 0.0 for p in self.players_obj}
        return {p.name: p.stack - self.starting_stack for p in self.players_obj}

    def get_result(self) -> Dict[str, Any]:
        if not self.is_terminal(): return {"status": "in_progress"}
        winner_name = "Tie" # Default to Tie
        p1_stack = self.players_obj[0].stack
        p2_stack = self.players_obj[1].stack
        if p1_stack > p2_stack: winner_name = self.players_obj[0].name
        elif p2_stack > p1_stack: winner_name = self.players_obj[1].name

        return {
            "status": "complete",
            "winner": winner_name,
            "players": { p.name: { "final_stack": p.stack, "net_winnings": p.stack - self.starting_stack } for p in self.players_obj },
            "total_hands": self.hand_number
        }

    def log_event(self, event_type: str, data: Dict[str, Any]):
        # Ensure hand number is consistent, especially for delayed events like hand_result
        current_hand = self.hand_number
        log_entry = {
            'timestamp': time.time(),
            'hand': current_hand,
            'stage': self.stage, # Log stage at time of event
            'event_type': event_type,
            'data': data # Store relevant data
        }
        self.history.append(log_entry)

    # --- Persistence Methods ---
    def _restore_from_state(self, state_data: Dict[str, Any]):
        # Simplified restore - requires careful state management if used
        logger.warning("Restoring from state is experimental and may be incomplete.")
        self.game_id = state_data.get('game_id', self.game_id)
        self.players = state_data.get('players', self.players)
        self._current_seed = state_data.get('random_seed')
        self._rng = random.Random(self._current_seed)

        self.player1_name = state_data.get('player1_name')
        self.player2_name = state_data.get('player2_name')
        self.starting_stack = state_data.get('starting_stack')
        self.small_blind = state_data.get('small_blind')
        self.big_blind = state_data.get('big_blind')

        self.pot = state_data.get('pot', 0)
        self.community_cards = state_data.get('community_cards', [])
        self.current_bet = state_data.get('current_bet', 0)
        self.dealer_index = state_data.get('dealer_index', 0)
        self.active_player_index = state_data.get('active_player_index', -1)
        self.last_raiser_index = state_data.get('last_raiser_index', -1)
        self.stage = state_data.get('stage', 'pre-deal')
        self.hand_number = state_data.get('hand_number', 0)
        self.hand_complete = state_data.get('hand_complete', False)
        self.game_complete = state_data.get('game_complete', False)
        self.aggressor_action_closed = state_data.get('aggressor_action_closed', False)

        # Restore Deck and Players
        deck_data = state_data.get('deck')
        self.deck = Deck.from_dict(deck_data, self._current_seed) if deck_data else Deck(self._current_seed)
        players_data = state_data.get('players_data')
        self.players_obj = [Player.from_dict(p_data) for p_data in players_data] if players_data else []

        self.history = state_data.get('history', [])
        self.is_interactive = state_data.get('is_interactive', True)
        logger.info(f"Game state restored for ID {self.game_id}")

    def to_json(self) -> str:
        # Create a serializable representation of the game state
        state = {
            'game_id': self.game_id, 'players': self.players, 'random_seed': self._current_seed,
            'player1_name': self.player1_name, 'player2_name': self.player2_name,
            'starting_stack': self.starting_stack, 'small_blind': self.small_blind, 'big_blind': self.big_blind,
            'pot': self.pot, 'community_cards': self.community_cards, 'current_bet': self.current_bet,
            'dealer_index': self.dealer_index, 'active_player_index': self.active_player_index,
            'last_raiser_index': self.last_raiser_index, 'stage': self.stage, 'hand_number': self.hand_number,
            'hand_complete': self.hand_complete, 'game_complete': self.game_complete, 'aggressor_action_closed': self.aggressor_action_closed,
            'deck': self.deck.to_dict(), # Serialize deck
            'players_data': [p.to_dict() for p in self.players_obj], # Serialize players
            'history': self.history,
            'is_interactive': self.is_interactive
        }
        # Use default=str for potential non-serializable items (like timestamps if not float)
        try:
             return json.dumps(state, indent=2, default=str)
        except TypeError as e:
             logger.error(f"Failed to serialize game state to JSON: {e}")
             # Attempt serialization without history as it might contain complex objects
             state.pop('history', None)
             return json.dumps(state, indent=2, default=str)


def run_non_interactive_game( game: HeadsUpPoker, player1_agent, player2_agent, num_hands: int ) -> Dict[str, Any]:
    if not player1_agent or not player2_agent: raise ValueError("Both player agents must be provided")
    if game.is_interactive: logger.warning("Running non-interactive game loop on a game marked as interactive."); game.set_interactive_mode(False)
    agents = {game.player1_name: player1_agent, game.player2_name: player2_agent}
    logger.info(f"Starting non-interactive game run for {num_hands} hands.")

    while game.hand_number < num_hands and not game.is_terminal():
        game.reset_hand()
        if game.is_terminal(): break # Check terminal state after reset (e.g., one player broke)
        if not game.start_hand():
             logger.error(f"Failed to start hand {game.hand_number + 1}. Aborting game."); break # Adjust hand number for logging

        # Inner loop for actions within a hand
        while not game.hand_complete and not game.is_terminal():
            if game.active_player_index == -1:
                 logger.error(f"Game loop error: No active player set in stage {game.stage}. Hand {game.hand_number}")
                 # Attempt to recover or break
                 if game._check_betting_round_complete(): continue # Try to advance stage
                 else: break # Break inner loop if cannot recover

            active_player_obj = game.players_obj[game.active_player_index]
            active_player_name = active_player_obj.name

            # If current player is folded or all-in, they can't act. Check round completion.
            if active_player_obj.folded or active_player_obj.all_in:
                logger.debug(f"Skipping turn for {active_player_name} (folded/all-in). Checking round completion.")
                if game._check_betting_round_complete(): continue # Check if round ends, potentially advancing stage
                else:
                     # If round didn't complete, it must be opponent's turn
                     logger.debug(f"Round not complete, switching to opponent.")
                     game.active_player_index = 1 - game.active_player_index
                     continue # Continue loop for the other player's turn

            # Get state and actions for the agent
            state_for_agent = game.get_state(player_id=active_player_name)
            valid_actions = state_for_agent.get("valid_actions", [])

            if not valid_actions:
                 # This can happen if a player is all-in but not marked, or round should have ended.
                 logger.warning(f"No valid actions for active player {active_player_name}. State: Stage={game.stage}, AllIn={active_player_obj.all_in}, Folded={active_player_obj.folded}")
                 # Try to force check round completion again
                 if game._check_betting_round_complete():
                     logger.info("Checking round completion again resolved the no valid actions issue.")
                     continue
                 else:
                     logger.error("Stuck state: No valid actions but round not complete. Forcing fold.")
                     action = {"action_type": "fold"} # Force fold as last resort
            else:
                # Call the agent function
                agent_func = agents[active_player_name]
                try:
                    action = agent_func(state_for_agent, valid_actions)
                    if action is None: # Agent returned None
                         raise ValueError("Agent returned None action")
                    # Minimal validation of returned action structure
                    if not isinstance(action, dict) or "action_type" not in action:
                         raise ValueError(f"Agent returned invalid action format: {action}")

                except Exception as e:
                    logger.exception(f"Agent {active_player_name} failed: {e}. Using fallback.")
                    # Use a robust fallback mechanism
                    from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                    parser = PokerResponseParser() # Create instance if needed
                    action = parser.get_fallback_action(valid_actions) # Get fallback

            # Apply the chosen/fallback action
            success = game.apply_action(action, active_player_name)
            if not success:
                # If apply_action failed (e.g., validation failed internally), use fallback
                logger.warning(f"Agent {active_player_name} provided invalid action: {action}. Applying fallback.")
                from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                parser = PokerResponseParser()
                fallback_action = parser.get_fallback_action(valid_actions)
                # Ensure fallback action has necessary amount if it's a raise/call
                if fallback_action["action_type"] == "raise":
                     # Find raise details in valid_actions to get min amount
                     raise_details = next((va for va in valid_actions if va["action_type"] == "raise"), None)
                     fallback_action["amount"] = raise_details["min_amount"] if raise_details else game.big_blind
                elif fallback_action["action_type"] == "call":
                     call_details = next((va for va in valid_actions if va["action_type"] == "call"), None)
                     fallback_action["amount"] = call_details["amount"] if call_details else 0

                logger.info(f"Applying fallback action: {fallback_action}")
                game.apply_action(fallback_action, active_player_name) # Apply the fallback

            # Small delay for readability / API rate limits if needed
            # time.sleep(0.05)

    logger.info(f"Non-interactive game run finished. Total hands intended: {num_hands}. Hands played: {game.hand_number}.")
    close_game_log() # Ensure log is closed
    final_result = game.get_result()
    # Add more details to the final result if needed
    final_result["hands_played"] = game.hand_number # Ensure hands played is accurate
    final_result["players"] = { p.name: { "final_stack": p.stack, "net_winnings": p.stack - game.starting_stack } for p in game.players_obj }
    # final_result["game_history"] = game.get_history() # Optionally include full history

    return final_result


if __name__ == "__main__":
    # Example run configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from model_orchestrator.utils import init_game_log
    # Make sure logs directory exists
    if not os.path.exists("./logs"): os.makedirs("./logs")
    init_game_log("./logs")

    # Simple agent for testing
    def random_agent(state, valid_actions):
        chosen = random.choice(valid_actions)
        if chosen["action_type"] == "raise":
            # Choose a random valid raise amount
            min_r = chosen.get("min_amount", game.big_blind) # Use BB as min if not specified
            max_r = chosen.get("max_amount", state['players'][state['active_player']]['stack']) # Max is stack
            # Ensure min_r <= max_r
            if min_r > max_r: min_r = max_r
            chosen["amount"] = random.randint(min_r, max_r)
        logger.debug(f"RandomAgent ({state['active_player']}) chooses: {chosen}")
        return chosen

    # Initialize game
    test_game = HeadsUpPoker(random_seed=42, starting_stack=1000, small_blind=10, big_blind=20)

    # Run the game
    result = run_non_interactive_game(game=test_game, player1_agent=random_agent, player2_agent=random_agent, num_hands=10) # Run 10 hands

    # Print final results
    print("\n--- FINAL GAME RESULT ---")
    print(json.dumps(result, indent=2))