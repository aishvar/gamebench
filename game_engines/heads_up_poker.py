# game_engines/heads_up_poker.py

import random
import itertools
import json
import copy
import logging
import time # Import time for log_event
import statistics # For calculating averages in run_non_interactive_game
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
        self.starting_stack_this_hand = stack # Store initial stack for the hand
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
        if self.stack <= 0: # Use <= 0 for safety
            self.all_in = True
            logger.info(f"Player {self.name} is all-in.")
        logger.debug(f"Player {self.name} bets {bet_amount}. Stack: {self.stack}, CurrentBet: {self.current_bet}, TotalBetHand: {self.total_bet_in_hand}")
        return bet_amount

    def fold(self):
        self.folded = True
        self.last_action = "fold"
        logger.info(f"Player {self.name} folds.")

    def reset_for_new_hand(self, starting_stack):
        """Resets player state for a new hand, including resetting stack."""
        logger.debug(f"Resetting player {self.name} for new hand. Previous stack: {self.stack}")
        self.stack = starting_stack # <<< KEY CHANGE: Reset stack
        self.starting_stack_this_hand = starting_stack # Store for calculating net change
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet_in_hand = 0
        self.folded = False
        self.all_in = False
        if self.stack <= 0: # Check after reset (unlikely but possible if starting_stack <= 0)
             self.all_in = True
        self.last_action = None
        logger.debug(f"Player {self.name} stack reset to {self.stack}")


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
        player.starting_stack_this_hand = data.get('starting_stack_this_hand', player.stack) # Restore if available
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

        # Ace-low straight (A, 2, 3, 4, 5)
        if set([14, 2, 3, 4, 5]).issubset(values):
            is_straight = True
            straight_high = 5
        else: # Other straights
            # Use set for faster checking, convert back to sorted list
            unique_sorted_vals = sorted(list(set(values)), reverse=True)
            if len(unique_sorted_vals) >= 5:
                for i in range(len(unique_sorted_vals) - 4):
                    # Check for consecutive sequence from high to low
                    if all(unique_sorted_vals[i+j] == unique_sorted_vals[i] - j for j in range(5)):
                        is_straight = True
                        straight_high = unique_sorted_vals[i] # Highest card in the straight
                        break

        rank_score = -1
        tiebreakers = ()

        if is_straight and is_flush:
            rank_score = 8
            # Handle Ace-low straight flush explicitly for tiebreaker
            if straight_high == 5 and 14 in values: # Steel wheel
                tiebreakers = (5,)
            else:
                tiebreakers = (straight_high,)
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
        elif is_flush:
            rank_score = 5
            tiebreakers = tuple(sorted(values, reverse=True)[:5]) # Only top 5 matter
        elif is_straight:
            rank_score = 4
            tiebreakers = (straight_high,)
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
        else:
            rank_score = 0
            tiebreakers = tuple(sorted(values, reverse=True)[:5])

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
        if rank_value == -1: return "Invalid Hand"
        desc = rank_names.get(rank_value, "Unknown")
        # Corrected: Convert values back to ranks
        value_to_rank = {v: r for r, v in RANK_TO_VALUE.items()}
        readable_tiebreakers = [value_to_rank.get(tb, str(tb)) for tb in tiebreakers]

        return f"{desc} ({', '.join(readable_tiebreakers)})"

    @staticmethod
    def determine_winner(player1: Player, player2: Player, community_cards: List[Tuple[str, str]]) -> Tuple[Optional[Player], Tuple[int, tuple], Tuple[int, tuple]]:
        hand1, hand2 = (-1, ()), (-1, ())
        cards1, cards2 = [], []
        if not player1.folded:
            cards1 = player1.hole_cards + community_cards
            if len(cards1) >= 5:
                hand1 = HandEvaluator.best_hand_rank(cards1)
        if not player2.folded:
            cards2 = player2.hole_cards + community_cards
            if len(cards2) >= 5:
                hand2 = HandEvaluator.best_hand_rank(cards2)

        hand1_desc = HandEvaluator.hand_description(hand1) if not player1.folded else 'Folded'
        hand2_desc = HandEvaluator.hand_description(hand2) if not player2.folded else 'Folded'
        logger.info(f"Evaluating hands: {player1.name} ({hand1_desc}) vs {player2.name} ({hand2_desc})")

        # Handle folds first
        if player1.folded and not player2.folded: return player2, hand1, hand2
        if not player1.folded and player2.folded: return player1, hand1, hand2
        if player1.folded and player2.folded: # Should not happen ideally
            logger.error("Showdown reached with both players folded."); return None, hand1, hand2

        # If neither folded, compare hands
        # Compare ranks using lexicographical comparison for tiebreakers
        if hand1 > hand2: return player1, hand1, hand2
        if hand2 > hand1: return player2, hand1, hand2
        if hand1 == hand2 and hand1[0] != -1: return None, hand1, hand2 # Tie

        # Fallback if hands are invalid (-1) for some reason
        logger.error("Showdown reached with invalid hands and no folds."); return None, hand1, hand2


class HeadsUpPoker(BaseGame):
    def __init__(self, game_id=None, players=None, random_seed=None,
                 player1_name="Player1", player2_name="Player2",
                 starting_stack=1000, small_blind=10, big_blind=20):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.starting_stack = starting_stack # This is the stack players START EACH HAND with
        self.small_blind = small_blind
        self.big_blind = big_blind
        self._rng = random.Random(random_seed)
        self._current_seed = random_seed
        super().__init__(game_id, players=[player1_name, player2_name], random_seed=self._current_seed)
        logger.info(f"HeadsUpPoker game initialized. ID: {self.game_id}, Seed: {self._current_seed}, Start Stack Per Hand: {self.starting_stack}")
        log_event_to_game_file(f"Game Initialized. Seed: {self._current_seed}, Blinds: {small_blind}/{big_blind}, Start Stack Per Hand: {starting_stack}\n")

    def _initialize_game_state(self):
        """Initializes state for the very first hand or after a full game reset."""
        deck_seed = hash((self._current_seed, "init")) if self._current_seed is not None else None
        self.deck = Deck(random_seed=deck_seed)
        # Players start with the defined starting stack for the first hand
        self.players_obj = [Player(self.player1_name, self.starting_stack), Player(self.player2_name, self.starting_stack)]
        self.community_cards: List[Tuple[str, str]] = []
        self.pot = 0
        self.current_bet = 0 # Bet level for the current round
        self.dealer_index = self._rng.choice([0, 1]) # Randomly assign dealer for first hand
        self.active_player_index = -1 # Set during hand start
        self.last_raiser_index = -1 # Tracks last aggressor in betting round
        self.stage = "pre-deal" # Initial stage before a hand starts
        self.hand_number = 0 # Hand counter
        self.hand_complete = False # Flag for current hand status
        self.game_complete = False # Flag for overall game completion (e.g., hand limit reached)
        self.aggressor_action_closed = False # Flag to track betting round closure
        self.last_hand_outcome = None # Store outcome of the most recent hand
        logger.debug("Initial game state created.")

    def reset(self):
        """Resets the entire game back to its initial state (before hand 1)."""
        logger.info("Resetting game to initial state.")
        self._initialize_game_state()
        self.history = []
        log_event_to_game_file("--- GAME RESET ---\n")

    def reset_hand(self):
        """Resets the state for the start of a new hand."""
        # Log outcome of previous hand before resetting stacks
        if self.hand_number > 0 and self.last_hand_outcome:
            # Log the state *after* the pot was awarded but *before* resetting stacks
             log_hand_result(self.get_state(), self.hand_number, self.last_hand_outcome)

        # --- Stack Reset Logic ---
        previous_stacks = {p.name: p.stack for p in self.players_obj}
        logger.debug(f"Hand {self.hand_number} ended. Stacks before reset: {previous_stacks}")
        for player in self.players_obj:
            player.reset_for_new_hand(self.starting_stack) # <<< KEY CHANGE

        self.hand_number += 1
        logger.info(f"--- Starting Hand #{self.hand_number} --- Stacks reset to {self.starting_stack}")
        log_event_to_game_file(f"--- Starting Hand #{self.hand_number} ---\n")

        # --- Reset Hand-Specific State ---
        deck_seed = hash((self._current_seed, self.hand_number)) if self._current_seed is not None else None
        self.deck = Deck(random_seed=deck_seed)
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.last_raiser_index = -1
        self.dealer_index = 1 - self.dealer_index # Rotate dealer
        self.active_player_index = -1 # Will be set in start_hand
        self.stage = "pre-deal"
        self.hand_complete = False
        self.aggressor_action_closed = False
        self.last_hand_outcome = None # Clear last outcome

        # Check for game completion based on hand limit is done in run_non_interactive_game
        # No need for the stack-based game completion check here anymore

        logger.debug(f"Hand {self.hand_number} reset complete. Dealer: {self.players_obj[self.dealer_index].name}")

    def start_hand(self):
        """Posts blinds, deals cards, and sets up the pre-flop betting round."""
        if self.stage != "pre-deal" or self.game_complete:
             logger.warning(f"Attempted to start hand in invalid state (Stage: {self.stage}, GameComplete: {self.game_complete})")
             return False
        # Hand number is already incremented in reset_hand
        logger.info(f"Posting blinds for Hand #{self.hand_number}")

        sb_player_index = self.dealer_index
        bb_player_index = 1 - self.dealer_index
        small_blind_player = self.players_obj[sb_player_index]
        big_blind_player = self.players_obj[bb_player_index]
        logger.info(f"Dealer (SB): {small_blind_player.name}, BB: {big_blind_player.name}")

        # Post blinds
        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)
        self.pot += sb_amount + bb_amount
        self.current_bet = max(small_blind_player.current_bet, big_blind_player.current_bet)
        # BB is considered the last 'raiser' initially if their bet is >= SB
        self.last_raiser_index = bb_player_index # if bb_amount >= sb_amount else sb_player_index # Simplified: BB always sets initial bet level

        logger.info(f"Blinds posted. SB: {sb_amount} by {small_blind_player.name} (Stack: {small_blind_player.stack}). BB: {bb_amount} by {big_blind_player.name} (Stack: {big_blind_player.stack}). Pot: {self.pot}. Current Bet Level: {self.current_bet}")
        self.log_event("blinds_posted", {
            "hand": self.hand_number,
            "small_blind": {"player": small_blind_player.name, "amount": sb_amount, "stack": small_blind_player.stack},
            "big_blind": {"player": big_blind_player.name, "amount": bb_amount, "stack": big_blind_player.stack},
            "pot": self.pot
        })

        # Deal hole cards
        try:
            self.players_obj[0].hole_cards = self.deck.draw(2)
            self.players_obj[1].hole_cards = self.deck.draw(2)
            log_event_to_game_file(f"Hole cards dealt:\n  {self.players_obj[0].name}: {format_cards(self.players_obj[0].hole_cards)}\n  {self.players_obj[1].name}: {format_cards(self.players_obj[1].hole_cards)}\n")
            self.log_event("hole_cards_dealt", { "hand": self.hand_number, self.players_obj[0].name: self.players_obj[0].hole_cards, self.players_obj[1].name: self.players_obj[1].hole_cards })
        except ValueError as e:
            logger.error(f"Error dealing cards: {e}"); self.game_complete = True; return False

        # Set up for pre-flop betting
        self.stage = "pre-flop"
        self.active_player_index = sb_player_index # SB acts first pre-flop
        self.aggressor_action_closed = False # Action is open
        logger.info(f"Pre-flop stage begins. Active player: {self.players_obj[self.active_player_index].name}. Current Bet Level: {self.current_bet}")
        log_initial_state(self.get_state(), self.hand_number, self.stage)
        self._check_betting_round_complete() # Check if blinds put someone all-in immediately
        return True

    def deal_community_cards(self):
        """Deals the Flop, Turn, or River and resets betting state for the new round."""
        if self.game_complete or self.hand_complete: return False
        num_cards_to_deal, next_stage, event_name = 0, None, None
        if self.stage == "pre-flop": num_cards_to_deal, next_stage, event_name = 3, "flop", "flop_dealt"
        elif self.stage == "flop": num_cards_to_deal, next_stage, event_name = 1, "turn", "turn_dealt"
        elif self.stage == "turn": num_cards_to_deal, next_stage, event_name = 1, "river", "river_dealt"
        else:
            logger.error(f"deal_community_cards called at invalid stage: {self.stage}"); return False

        try:
            # Burn card
            if len(self.deck.cards) > num_cards_to_deal:
                 self.deck.draw(1) # Burn card (no need to store it)
                 logger.debug("Burned a card.")
            else:
                 logger.warning(f"Not enough cards left to burn before dealing {next_stage}.")

            # Deal community cards
            if len(self.deck.cards) < num_cards_to_deal:
                raise ValueError(f"Not enough cards left in deck to deal the {next_stage} ({len(self.deck.cards)} < {num_cards_to_deal})")

            new_cards = self.deck.draw(num_cards_to_deal)
            self.community_cards.extend(new_cards)
            logger.info(f"{next_stage.capitalize()} dealt: {format_cards(new_cards)}. Community: {format_cards(self.community_cards)}")
            log_event_to_game_file(f"{next_stage.capitalize()} dealt: {format_cards(self.community_cards)}\n")
            self.log_event(event_name, {"hand": self.hand_number, "cards": new_cards, "community_total": self.community_cards})
        except ValueError as e:
            logger.error(f"Error dealing community cards: {e}");
            # This likely means the deck ran out unexpectedly. Mark hand as incomplete/error?
            # For now, advance to showdown with available cards.
            self.game_complete = True # Mark game as potentially problematic
            self._advance_to_showdown()
            return False

        # --- Reset Betting State for New Round ---
        self.stage = next_stage
        self.current_bet = 0 # Reset bet level for the new round
        self.last_raiser_index = -1 # Reset last raiser
        self.aggressor_action_closed = False # Reset betting closed flag
        for player in self.players_obj:
             player.current_bet = 0 # Reset player's bet amount *this round*
             player.last_action = None # Clear last action

        # Action starts with the player out of position (SB position post-flop)
        self.active_player_index = self.dealer_index # SB/Dealer acts first post-flop
        # Skip player if they are all-in or folded (should already be handled by round end check?)
        if self.players_obj[self.active_player_index].all_in or self.players_obj[self.active_player_index].folded:
            self.active_player_index = 1 - self.active_player_index # Switch to the other player

        logger.info(f"Stage is now {self.stage}. Active player: {self.players_obj[self.active_player_index].name}. Bets reset.")
        log_event_to_game_file(f"--- {self.stage.upper()} --- Active: {self.players_obj[self.active_player_index].name}\n")

        # Check immediately if betting is possible or if round completes instantly (e.g., one player all-in)
        self._check_betting_round_complete()
        return True

    def get_state(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns the game state, optionally from a specific player's perspective."""
        requesting_player = next((p for p in self.players_obj if p.name == player_id), None) if player_id else None
        active_player_obj = self.players_obj[self.active_player_index] if 0 <= self.active_player_index < len(self.players_obj) else None

        # Get recent history for the current hand
        recent_history = []
        if self.history:
            current_hand_no = self.hand_number
            count = 10 # Limit history size passed in state
            for event in reversed(self.history):
                if isinstance(event, dict) and event.get('hand') == current_hand_no:
                    recent_history.append(event)
                    if len(recent_history) >= count: break
                elif isinstance(event, dict) and event.get('hand') is not None and event.get('hand') < current_hand_no: break
            recent_history.reverse() # Put back in chronological order

        state = {
            "game_id": self.game_id, "hand_number": self.hand_number, "stage": self.stage,
            "dealer": self.players_obj[self.dealer_index].name,
            "active_player": active_player_obj.name if active_player_obj and not self.hand_complete and not self.game_complete else None,
            "pot": self.pot, "community_cards": self.community_cards,
            "current_bet_level": self.current_bet, # Renamed for clarity in adapter
            "players": {},
            "history": recent_history # Pass list of dicts
        }

        for i, player in enumerate(self.players_obj):
            player_info = {
                "name": player.name, "stack": player.stack,
                "current_bet": player.current_bet, # Bet amount this round
                "total_bet_in_hand": player.total_bet_in_hand, # Total invested in hand
                "folded": player.folded, "all_in": player.all_in,
                "is_dealer": (i == self.dealer_index),
                "last_action": player.last_action
            }
            # Reveal cards only to the owner, or at showdown stage
            if player_id is None or player.name == player_id or (self.stage == "showdown" and not player.folded):
                player_info["hole_cards"] = player.hole_cards
            else:
                player_info["hole_cards"] = "Hidden" # Keep hidden otherwise
            state["players"][player.name] = player_info

        # Add valid actions only if it's the requesting player's turn
        if not self.hand_complete and not self.game_complete and requesting_player and active_player_obj and requesting_player.name == active_player_obj.name:
            state["valid_actions"] = self.get_valid_actions(player_id)
        elif not self.hand_complete and not self.game_complete and player_id is None and active_player_obj:
            # If no specific player requested, include actions for the active player
            state["valid_actions"] = self.get_valid_actions(active_player_obj.name)


        return state

    def get_valid_actions(self, player_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Calculates the valid actions for the specified or active player."""
        if self.hand_complete or self.game_complete or self.stage == "showdown": return []

        # Determine the target player index
        target_player_index = -1
        active_player_name = self.players_obj[self.active_player_index].name if 0 <= self.active_player_index < len(self.players_obj) else None

        if player_id:
             if player_id != active_player_name:
                  # logger.debug(f"get_valid_actions called for {player_id}, but active player is {active_player_name}")
                  return [] # Not this player's turn
             try:
                 target_player_index = next(i for i, p in enumerate(self.players_obj) if p.name == player_id)
             except StopIteration:
                 logger.warning(f"get_valid_actions called for unknown player_id: {player_id}")
                 return []
        elif active_player_name:
             target_player_index = self.active_player_index
        else: # No valid active player index
              logger.warning("get_valid_actions called with no active player set.")
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

        # --- FOLD --- (Always possible if facing a bet)
        if can_call:
            valid_actions.append({"action_type": "fold"})

        # --- CHECK ---
        if can_check:
            valid_actions.append({"action_type": "check"})

        # --- CALL ---
        if can_call:
            valid_actions.append({"action_type": "call", "amount": effective_call_amount})


        # --- RAISE ---
        # Conditions: Player has chips remaining after calling, and opponent is not already all-in.
        can_afford_call = player.stack >= effective_call_amount
        has_chips_after_call = player.stack > effective_call_amount
        opponent_can_respond = not opponent.all_in

        if can_afford_call and has_chips_after_call and opponent_can_respond:
            # Determine the minimum raise amount allowed
            # Size of the previous bet/raise in this round (relative increase)
            last_bet_or_raise_increase = self.big_blind # Default minimum raise size is BB

            # Find the size of the last increase in betting
            if self.last_raiser_index != -1:
                # Find the bet level *before* the last raise occurred.
                # This requires looking at the history or tracking previous bet levels.
                # Simplified approach: Assume the last raise increased the bet from the opponent's *current* bet level.
                 # This isn't perfect, especially with multiple raises, but is a common rule implementation.
                 bet_level_before_last_raise = opponent.current_bet
                 last_increase = self.current_bet - bet_level_before_last_raise
                 last_bet_or_raise_increase = max(self.big_blind, last_increase)
            elif self.current_bet > 0 : # If there's a bet, but no raise yet (e.g. initial bet post-flop, or BB pre-flop)
                 last_bet_or_raise_increase = max(self.big_blind, self.current_bet)

            # Min TOTAL bet player must make for a valid raise
            min_total_bet_after_raise = self.current_bet + last_bet_or_raise_increase

            # Min amount player needs to ADD to their current bet
            min_raise_amount_to_add = min_total_bet_after_raise - player.current_bet
            min_raise_amount_to_add = max(1, min_raise_amount_to_add) # Ensure positive add amount

            # Calculate the actual minimum amount they can add (limited by stack)
            effective_min_raise_add = min(min_raise_amount_to_add, player.stack - effective_call_amount)

            # Max amount player can ADD (all-in)
            max_raise_amount_to_add = player.stack - effective_call_amount # The max additional is going all-in

            # Add raise option if the minimum possible raise is valid and <= max possible raise
            if effective_min_raise_add > 0 and effective_min_raise_add <= max_raise_amount_to_add:
                 valid_actions.append({
                     "action_type": "raise",
                     "min_amount": effective_min_raise_add, # Min *additional* amount
                     "max_amount": max_raise_amount_to_add # Max *additional* amount
                 })
        # Special case: Player can go all-in, but it's less than a standard min-raise (but more than call)
        elif can_afford_call and has_chips_after_call and not opponent_can_respond: # Opponent is all-in, player covers
             pass # Cannot raise if opponent is all-in
        elif can_afford_call and has_chips_after_call: # Player can go all-in for less than min-raise
             all_in_raise_amount = player.stack - effective_call_amount
             if all_in_raise_amount > 0: # Must be more than just calling
                 valid_actions.append({
                     "action_type": "raise",
                     "min_amount": all_in_raise_amount, # Only option is all-in
                     "max_amount": all_in_raise_amount
                 })


        # logger.debug(f"Player {player.name} valid actions: {valid_actions}")
        return valid_actions

    def apply_action(self, action: Dict[str, Any], player_id: Optional[str] = None) -> bool:
        """Applies a validated action to the game state."""
        if self.hand_complete or self.game_complete:
            logger.warning(f"Attempted action in completed hand/game.")
            return False

        acting_player_index = self.active_player_index
        if player_id and (acting_player_index == -1 or self.players_obj[acting_player_index].name != player_id):
            logger.error(f"Action applied by non-active player? Expected {self.players_obj[acting_player_index].name if acting_player_index != -1 else 'N/A'}, got {player_id}")
            return False
        if acting_player_index == -1:
            logger.error("apply_action called with no active player index."); return False

        player = self.players_obj[acting_player_index]
        opponent = self.players_obj[1 - acting_player_index]

        # Double-check if player can act (should be caught by get_valid_actions, but safety check)
        if player.folded or player.all_in:
            logger.warning(f"Attempted action by player {player.name} who is already folded or all-in.")
            self._check_betting_round_complete() # Check completion, state might be inconsistent
            return False # Action cannot be applied

        action_type = action.get("action_type", "").lower()
        amount_param = action.get("amount") # Amount parameter from action dict (for raise/call)
        # Validate against dynamically generated valid actions
        valid_actions = self.get_valid_actions(player.name)
        chosen_valid_action = next((va for va in valid_actions if va["action_type"] == action_type), None)

        logger.info(f"Player {player.name} attempts action: {action_type} {f'(Amount Param: {amount_param})' if amount_param is not None else ''}")

        if not chosen_valid_action:
             logger.warning(f"Invalid action type '{action_type}' chosen by {player.name}. Valid: {[a['action_type'] for a in valid_actions]}. Raw Action: {action}")
             # Attempt to apply fallback logic directly here? Or let the caller handle retry?
             # For now, return False, indicating failure.
             return False

        processed_bet_amount = 0 # Amount actually removed from stack and added to pot
        action_closes_round = False # Flag to track if this action ends the betting round

        if action_type == "fold":
            player.fold()
            player.last_action = "fold"
            processed_bet_amount = 0
            # Award pot immediately if opponent hasn't folded
            if not opponent.folded:
                self._award_pot(opponent)
                self.hand_complete = True
                self.last_hand_outcome = self._create_hand_result_log(winner=opponent, reason="opponent_folded")
                self.log_event("hand_result", self.last_hand_outcome)
                # Log result message immediately
                log_hand_result(self.get_state(), self.hand_number, self.last_hand_outcome)


        elif action_type == "check":
            # Validation: Check is only valid if call_amount <= 0
            call_amount_needed = self.current_bet - player.current_bet
            if call_amount_needed > 0:
                 logger.warning(f"Invalid Check by {player.name}: Bet of {call_amount_needed} faced.")
                 return False
            player.last_action = "check"
            processed_bet_amount = 0
            # Checking might close the round if the other player also checked or BB checks preflop. Handled in _check_betting_round_complete.

        elif action_type == "call":
            call_amount_needed = self.current_bet - player.current_bet
            # Validation: Call is only valid if call_amount > 0
            if call_amount_needed <= 0:
                logger.warning(f"Invalid Call by {player.name}: No bet to call.")
                return False

            # Use the effective call amount calculated in get_valid_actions
            actual_call_amount = chosen_valid_action.get("amount", 0)
            # Sanity check against stack (should be guaranteed by get_valid_actions)
            actual_call_amount = min(actual_call_amount, player.stack)

            processed_bet_amount = player.bet(actual_call_amount)
            self.pot += processed_bet_amount
            player.last_action = "call"
            if player.all_in: logger.info(f"{player.name} calls {processed_bet_amount} and is all-in.")

            # Calling closes the round if it matches the last aggressor's bet
            if self.last_raiser_index != -1: # Calling a raise always closes action on this player
                 action_closes_round = True
            # Special case: Pre-flop, if SB calls the BB's initial blind, BB still has option to raise/check
            elif self.stage == "pre-flop" and acting_player_index == self.dealer_index: # SB is calling
                 action_closes_round = False # BB still to act
            else: # Calling a bet that wasn't a raise (limped pot, or post-flop initial bet)
                 action_closes_round = True

            self.aggressor_action_closed = action_closes_round


        elif action_type == "raise":
            if amount_param is None:
                 logger.warning(f"Raise action by {player.name} missing amount parameter.")
                 return False
            raise_amount_add = int(amount_param) # The amount to ADD to the current bet level

            # Validate amount against the min/max from valid_actions
            min_raise_add = chosen_valid_action.get("min_amount", 0)
            max_raise_add = chosen_valid_action.get("max_amount", float('inf'))

            if not (min_raise_add <= raise_amount_add <= max_raise_add):
                logger.warning(f"Invalid raise amount {raise_amount_add} by {player.name}. Valid ADD range: {min_raise_add}-{max_raise_add}")
                # Maybe try to clamp to valid range? Or just fail? Failing is safer.
                return False

            # Calculate total bet needed (call existing bet + add raise amount)
            call_amount_needed = max(0, self.current_bet - player.current_bet)
            total_bet_this_action = call_amount_needed + raise_amount_add
            # Ensure not betting more than stack (should be guaranteed by max_raise_add validation)
            total_bet_this_action = min(total_bet_this_action, player.stack)

            # Process the bet
            processed_bet_amount = player.bet(total_bet_this_action)
            self.pot += processed_bet_amount
            # Update game state for the new bet level
            self.current_bet = player.current_bet # New bet level for opponent to match
            self.last_raiser_index = acting_player_index # This player is now the aggressor
            player.last_action = "raise";
            self.aggressor_action_closed = False # Raising always re-opens the action
            logger.info(f"{player.name} raises by {raise_amount_add} (total bet this action: {processed_bet_amount}). New Bet Level: {self.current_bet}. Stack: {player.stack}")

        else:
             logger.error(f"Unhandled action type in apply_action: {action_type}")
             return False


        # --- Post-Action Processing ---
        if not self.hand_complete: # Don't switch player or check round if hand ended (e.g., fold)
            # Switch active player FIRST
            next_player_index = 1 - acting_player_index
            self.active_player_index = next_player_index
            logger.debug(f"Action applied by {player.name}. Switching active player to {self.players_obj[next_player_index].name}")

            # Log event reflecting state *before* checking round completion
            log_data_event = {
                "player": player.name, # Player who acted
                "action": action_type,
                "amount": processed_bet_amount if action_type in ['call', 'raise'] else None,
                "player_bet_round": player.current_bet,
                "player_stack": player.stack,
                "pot": self.pot,
                "current_bet_level": self.current_bet,
                "next_player": self.players_obj[next_player_index].name
            }
            self.log_event("action", log_data_event)

            # Log formatted action result AFTER switching active player
            log_action_result(player.name, action_type, processed_bet_amount if action_type in ['call', 'raise'] else None, self.get_state()) # Pass current state

            # Check if the betting round is now complete AFTER the state update and player switch
            self._check_betting_round_complete()

        return True

    def _check_betting_round_complete(self):
        """ Checks if the current betting round should end and advances the game stage if so. """
        if self.hand_complete or self.game_complete: return True # Hand/Game already over
        player1, player2 = self.players_obj[0], self.players_obj[1]

        # --- Trivial End Conditions ---
        # Fold already handled in apply_action, sets hand_complete
        if player1.all_in and player2.all_in:
            logger.debug("Both players all-in, betting ends. Advancing to showdown.")
            self._advance_to_showdown()
            return True
        # If one player is all-in and the other has called (or folded)
        if (player1.all_in and (player2.folded or player1.current_bet == player2.current_bet)) or \
           (player2.all_in and (player1.folded or player1.current_bet == player2.current_bet)):
             # Ensure the non-all-in player has completed their action if needed
             active_player_obj = self.players_obj[self.active_player_index]
             # If the active player is the one NOT all-in, they might still need to act (call the all-in)
             if not active_player_obj.all_in and active_player_obj.current_bet < self.current_bet and not active_player_obj.folded:
                 logger.debug("Betting continues: Non-all-in player needs to call.")
                 return False # Round continues, active player needs to act
             else:
                 logger.debug("One player all-in, opponent called or folded. Advancing to showdown.")
                 self._advance_to_showdown()
                 return True

        # --- Check Standard Betting Completion ---
        # Bets are considered matched if player bets are equal AND action has completed around
        bets_matched = (player1.current_bet == player2.current_bet)

        round_closed = False
        if bets_matched:
            # Action completes if:
            # 1. Pre-flop: BB checks (after SB call/limp)
            # 2. Post-flop: Player checks back (after opponent check)
            # 3. Any street: A player calls a bet/raise (handled by aggressor_action_closed)

            # Has action closed? Needs considering who acted last and if it was aggressive.
            # player_who_just_acted_idx = 1 - self.active_player_index # Index of player whose turn it just was
            # player_who_just_acted = self.players_obj[player_who_just_acted_idx]

            # Condition 1: Last action closed aggression (e.g., a call)
            if self.aggressor_action_closed:
                 round_closed = True
                 logger.debug("Betting round closed: Aggressor action was closed (e.g., call).")
            # Condition 2: Check completes the round
            elif player1.last_action == "check" and player2.last_action == "check":
                 round_closed = True
                 logger.debug("Betting round closed: Check-check.")
            # Condition 3: Pre-flop BB check
            elif self.stage == "pre-flop" and \
                 self.active_player_index == self.dealer_index and \
                 player2.last_action == "check" and \
                 player1.last_action in ["call", "check"]: # SB called/limped, BB checked
                 round_closed = True
                 logger.debug("Betting round closed: Pre-flop BB check.")


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

        # --- If Round Not Closed ---
        logger.debug(f"Betting round continues. Active player: {self.players_obj[self.active_player_index].name}")
        # Ensure active player *can* act (redundant check, should be handled earlier)
        current_active_player = self.players_obj[self.active_player_index]
        if current_active_player.folded or current_active_player.all_in:
             logger.error("Critical State Error: Betting round not closed, but active player cannot act!")
             # Attempt recovery: Force showdown or switch player? Force showdown is safer.
             self._advance_to_showdown()
             return True

        return False # Round continues

    def _advance_to_showdown(self):
        """Deals remaining community cards automatically and proceeds to hand evaluation."""
        if self.hand_complete: return # Avoid advancing if already done

        logger.info("Advancing to showdown (dealing remaining cards if necessary).")
        # Deal remaining community cards automatically if not yet river and players aren't folded
        player1, player2 = self.players_obj[0], self.players_obj[1]
        needs_dealing = not (player1.folded or player2.folded) # Only deal if cards matter

        while needs_dealing and self.stage != "river" and not self.game_complete and len(self.community_cards) < 5:
            # Determine next stage and cards to deal
            next_stage = ""
            num_to_deal = 0
            if self.stage == "pre-flop": next_stage, num_to_deal = "flop", 3
            elif self.stage == "flop": next_stage, num_to_deal = "turn", 1
            elif self.stage == "turn": next_stage, num_to_deal = "river", 1
            else: break # Should not happen

            try:
                # Burn card if deck allows
                if len(self.deck.cards) > num_to_deal:
                     self.deck.draw(1); logger.debug("Auto-burning card.")
                else: logger.warning("Not enough cards to burn before auto-dealing.")

                if len(self.deck.cards) < num_to_deal:
                    raise ValueError(f"Not enough cards to auto-deal {next_stage}")

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
        self.stage = "showdown" # Change stage *before* handling showdown
        # Proceed to evaluate hands
        self._handle_showdown()

    def _handle_showdown(self):
        """Evaluates hands if necessary and awards the pot."""
        if self.hand_complete: return # Avoid double processing
        logger.info("--- Showdown ---")
        # Stage is already set to showdown in _advance_to_showdown

        player1, player2 = self.players_obj[0], self.players_obj[1]

        # Determine winner based on folds or hand evaluation
        outcome_data = {}
        winner = None
        reason = ""
        hand1_rank, hand2_rank = (-1, ()), (-1, ()) # Initialize ranks

        # Check folds first (should be redundant if apply_action handled it, but safe)
        if player1.folded and not player2.folded:
            winner = player2; reason = "opponent_folded"
            self._award_pot(winner)
        elif not player1.folded and player2.folded:
            winner = player1; reason = "opponent_folded"
            self._award_pot(winner)
        elif player1.folded and player2.folded: # Should not happen
             reason = "error_both_folded"
             self._split_pot() # Safest fallback
        else: # Both players active (or were active until river), evaluate hands
            winner_obj, hand1_rank, hand2_rank = HandEvaluator.determine_winner(player1, player2, self.community_cards)

            # Log the showdown details
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

            # Award pot based on winner
            if winner_obj is None: # Tie
                reason = "tie_showdown"
                self._split_pot()
                outcome_data = self._create_hand_result_log(winner=None, reason=reason, hand1_desc=hand1_desc, hand2_desc=hand2_desc)
            else: # Clear winner
                winner = winner_obj
                reason = "better_hand"
                winning_hand_desc = hand1_desc if winner == player1 else hand2_desc
                self._award_pot(winner)
                outcome_data = self._create_hand_result_log(winner=winner, reason=reason, winning_hand_desc=winning_hand_desc)


        # If outcome_data wasn't set in showdown evaluation (e.g., fold scenario handled earlier), create it now
        if not outcome_data:
             outcome_data = self._create_hand_result_log(winner=winner, reason=reason)

        # Store outcome, log event, mark hand complete
        self.last_hand_outcome = outcome_data
        self.log_event("hand_result", outcome_data)
        self.hand_complete = True
        # Logging of the final state message is now done in reset_hand for the *next* hand,
        # or at the very end of run_non_interactive_game

    def _award_pot(self, winner: Player):
        """Awards the current pot to the winner."""
        logger.info(f"Awarding pot of {self.pot} to {winner.name}")
        winner.stack += self.pot
        self.pot = 0

    def _split_pot(self):
        """Splits the pot between players in case of a tie."""
        logger.info(f"Splitting pot of {self.pot}")
        split_amount, remainder = divmod(self.pot, 2)
        self.players_obj[0].stack += split_amount
        self.players_obj[1].stack += split_amount
        if remainder > 0:
            # Award remainder to the player out of position (non-dealer)
            non_dealer_index = 1 - self.dealer_index
            self.players_obj[non_dealer_index].stack += remainder
            logger.debug(f"Awarding remainder {remainder} chip to {self.players_obj[non_dealer_index].name}")
        self.pot = 0

    def _create_hand_result_log(self, winner: Optional[Player], reason: str, **kwargs) -> Dict[str, Any]:
         """Creates a dictionary summarizing the hand outcome for logging and results."""
         # Calculate pot size *before* awarding/splitting based on total bets
         pot_before_award = sum(p.total_bet_in_hand for p in self.players_obj)
         # Calculate net winnings for this hand
         p1_net = self.players_obj[0].stack - self.players_obj[0].starting_stack_this_hand
         p2_net = self.players_obj[1].stack - self.players_obj[1].starting_stack_this_hand

         result_data = {
             "hand": self.hand_number,
             "winner": winner.name if winner else "Tie", # Use "Tie" string for clarity
             "reason": reason,
             "pot_awarded": pot_before_award,
             "final_stacks": {p.name: p.stack for p in self.players_obj},
             "net_winnings": {self.players_obj[0].name: p1_net, self.players_obj[1].name: p2_net}
         }
         # Add hand descriptions if available from kwargs
         result_data.update(kwargs) # Add hand1_desc, hand2_desc, winning_hand_desc if passed
         return result_data

    def get_last_hand_outcome(self) -> Optional[Dict[str, Any]]:
        """Returns the result dictionary of the most recently completed hand."""
        return self.last_hand_outcome

    def is_terminal(self) -> bool:
        """Checks if the game (experiment run) should terminate. Controlled by hand limit."""
        # The game itself doesn't terminate based on stacks anymore per hand.
        # Termination is handled by the loop in run_non_interactive_game based on num_hands.
        # Return self.game_complete which is set by the runner or if errors occur.
        return self.game_complete

    def get_rewards(self) -> Dict[str, float]:
        """Returns the net winnings for the *last completed hand*."""
        # Since stacks reset, rewards are per-hand. Return the last hand's net winnings.
        if self.last_hand_outcome and "net_winnings" in self.last_hand_outcome:
             return self.last_hand_outcome["net_winnings"]
        else:
             # Return 0 if no hand has completed yet
             return {p.name: 0.0 for p in self.players_obj}

    def get_result(self) -> Dict[str, Any]:
        """Returns the *final* result of the game run (intended for after N hands)."""
        # This should be called by the runner *after* the loop finishes.
        # It should ideally return the aggregated results calculated in run_non_interactive_game.
        # However, the runner stores the result dict directly.
        # This method might return the state of the *last hand* if called mid-run,
        # or be less meaningful if called independently.
        if not self.game_complete:
             return {"status": "in_progress", "hands_played": self.hand_number}

        # If game is marked complete, try to return last hand outcome or a summary
        final_outcome = {
             "status": "complete",
             "total_hands_played": self.hand_number,
             "final_stacks_last_hand": {p.name: p.stack for p in self.players_obj}
             # The true aggregate result is calculated and returned by run_non_interactive_game
        }
        if self.last_hand_outcome:
            final_outcome["last_hand_details"] = self.last_hand_outcome

        return final_outcome

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Logs a game event to the internal history list."""
        current_hand = self.hand_number
        log_entry = {
            'timestamp': time.time(),
            'hand': current_hand,
            'stage': self.stage, # Log stage at time of event
            'event_type': event_type,
            'data': copy.deepcopy(data) # Deep copy data to avoid modification issues
        }
        self.history.append(log_entry)

    # --- Persistence Methods (Less relevant with per-hand resets, but kept for structure) ---
    def _restore_from_state(self, state_data: Dict[str, Any]):
        logger.warning("Restoring from state is experimental and may be inconsistent with per-hand resets.")
        # Basic restoration, might need more work if used seriously
        self.game_id = state_data.get('game_id', self.game_id)
        # ... (restore other attributes)
        self.players_obj = [Player.from_dict(p_data) for p_data in state_data.get('players_data', [])]
        # ...
        logger.info(f"Game state partially restored for ID {self.game_id}")

    def to_json(self) -> str:
        # Create a serializable representation of the *current* game state
        state = self.get_state() # Get current state dict
        state['game_config'] = { # Add config for potential restoration
            'player1_name': self.player1_name, 'player2_name': self.player2_name,
            'starting_stack': self.starting_stack, 'small_blind': self.small_blind, 'big_blind': self.big_blind,
            'random_seed': self._current_seed
        }
        state['full_history'] = self.history # Include full history separate from truncated state history
        state['game_complete'] = self.game_complete
        state['dealer_index'] = self.dealer_index
        state['last_raiser_index'] = self.last_raiser_index
        state['aggressor_action_closed'] = self.aggressor_action_closed
        # Deck state might be needed for perfect restoration
        # state['deck_state'] = self.deck.to_dict()

        try:
             return json.dumps(state, indent=2, default=str)
        except TypeError as e:
             logger.error(f"Failed to serialize game state to JSON: {e}")
             # Attempt serialization without potentially problematic parts
             state.pop('full_history', None)
             state.pop('history', None)
             return json.dumps(state, indent=2, default=str)


# ==============================================================================
# Non-Interactive Game Runner Function
# ==============================================================================

def run_non_interactive_game( game: HeadsUpPoker, player1_agent, player2_agent, num_hands: int ) -> Dict[str, Any]:
    """
    Runs a game simulation for a fixed number of hands with agents making decisions.
    Stacks are reset to the initial starting stack for each hand.

    Args:
        game: An initialized HeadsUpPoker game instance.
        player1_agent: Function representing player 1's decision logic.
                       Signature: agent(state: Dict, valid_actions: List) -> Dict
        player2_agent: Function representing player 2's decision logic.
        num_hands: The total number of hands to play.

    Returns:
        A dictionary containing:
            - hands_played: Actual number of hands completed.
            - per_hand_results: A list of dictionaries, one for each hand's outcome.
            - aggregate_results: Summary statistics over all hands (e.g., avg net winnings).
    """
    if not player1_agent or not player2_agent:
        raise ValueError("Both player agents must be provided")
    if game.is_interactive:
        logger.warning("Running non-interactive game loop on a game marked as interactive.")
        game.set_interactive_mode(False)

    agents = {game.player1_name: player1_agent, game.player2_name: player2_agent}
    logger.info(f"Starting non-interactive game run for {num_hands} hands. Stacks reset each hand.")

    per_hand_results = []
    player_ids = [game.player1_name, game.player2_name]
    total_net_winnings = {name: 0.0 for name in player_ids}
    hands_completed_successfully = 0

    # --- Main Hand Loop ---
    while game.hand_number < num_hands:
        try:
            game.reset_hand() # Resets state AND player stacks for the new hand
            if not game.start_hand():
                logger.error(f"Failed to start hand {game.hand_number}. Aborting game run.")
                game.game_complete = True # Mark as incomplete run
                break

            # --- Inner Action Loop (within a hand) ---
            while not game.hand_complete:
                if game.active_player_index == -1:
                    logger.error(f"Game loop error: No active player set in stage {game.stage}. Hand {game.hand_number}")
                    # Attempt to recover or break hand
                    if game._check_betting_round_complete(): continue # Try to advance stage
                    else:
                        logger.error("Cannot recover from no active player state. Breaking hand.")
                        # Mark hand as errored? For now, just break.
                        game.hand_complete = True # Force hand completion status
                        game.last_hand_outcome = {"error": "No active player state"}
                        continue # Exit inner loop

                active_player_obj = game.players_obj[game.active_player_index]
                active_player_name = active_player_obj.name

                # Skip turn if player cannot act (folded/all-in) - Round completion checked after action
                if active_player_obj.folded or active_player_obj.all_in:
                    logger.debug(f"Skipping turn for {active_player_name} (folded={active_player_obj.folded}, all_in={active_player_obj.all_in}). Checking round completion.")
                    if game._check_betting_round_complete():
                        logger.debug("Round completed after skipping turn.")
                        continue # Continue outer loop (will check hand_complete)
                    else:
                        # If round didn't complete, it must be opponent's turn (should have been switched already?)
                        logger.warning(f"Round not complete after skipping turn for {active_player_name}. Active player should switch.")
                        # Explicitly switch if needed, though apply_action should handle this
                        if game.active_player_index != (1 - game.players_obj.index(active_player_obj)):
                             game.active_player_index = 1 - game.players_obj.index(active_player_obj)
                             logger.info(f"Manually switched active player to {game.players_obj[game.active_player_index].name}")
                        continue # Continue inner loop for the other player's turn

                # Get state and actions for the agent
                state_for_agent = game.get_state(player_id=active_player_name)
                valid_actions = state_for_agent.get("valid_actions", [])

                if not valid_actions and not game.hand_complete:
                    # This implies a state where the active player should act but has no options.
                    # Often happens if round completion logic didn't catch an all-in scenario correctly.
                    logger.warning(f"No valid actions for active player {active_player_name}, but hand not complete. State: Stage={game.stage}, AllIn={active_player_obj.all_in}, Folded={active_player_obj.folded}, BetLevel={game.current_bet}, PlayerBet={active_player_obj.current_bet}")
                    # Try to force check round completion again
                    if game._check_betting_round_complete():
                        logger.info("Checking round completion again resolved the no valid actions issue.")
                        continue
                    else:
                        logger.error("Stuck state: No valid actions but round not complete. Forcing fold as fallback.")
                        action = {"action_type": "fold"} # Force fold as last resort
                        game.apply_action(action, active_player_name) # Apply the forced action
                        # apply_action will set hand_complete if fold occurs
                else:
                    # Call the agent function
                    agent_func = agents[active_player_name]
                    action = None
                    try:
                        action = agent_func(state_for_agent, valid_actions)
                        if action is None: raise ValueError("Agent returned None action")
                        if not isinstance(action, dict) or "action_type" not in action:
                             raise ValueError(f"Agent returned invalid action format: {action}")
                    except Exception as e:
                        logger.exception(f"Agent {active_player_name} failed: {e}. Using fallback.")
                        from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                        parser = PokerResponseParser()
                        action = parser.get_fallback_action(valid_actions) # Get fallback

                    # Apply the chosen/fallback action
                    success = game.apply_action(action, active_player_name)
                    if not success:
                        # If apply_action failed (e.g., LLM hallucinated invalid amount after validation)
                        logger.warning(f"Game engine rejected action from {active_player_name}: {action}. Applying fallback.")
                        from model_orchestrator.response_parsers.poker_parser import PokerResponseParser
                        parser = PokerResponseParser()
                        fallback_action = parser.get_fallback_action(valid_actions)
                        logger.info(f"Applying fallback action: {fallback_action}")
                        if not game.apply_action(fallback_action, active_player_name):
                             logger.error(f"FATAL: Fallback action {fallback_action} also failed for player {active_player_name}. Breaking hand.")
                             game.hand_complete = True # Force hand end
                             game.last_hand_outcome = {"error": "Fallback action failed"}


                # Optional delay can go here: time.sleep(0.01)

            # --- End of Inner Action Loop (Hand Complete) ---
            hand_outcome = game.get_last_hand_outcome()
            if hand_outcome and "error" not in hand_outcome:
                 per_hand_results.append(hand_outcome)
                 # Update total winnings
                 for name, winnings in hand_outcome.get("net_winnings", {}).items():
                     total_net_winnings[name] += winnings
                 hands_completed_successfully += 1
            elif hand_outcome and "error" in hand_outcome:
                 logger.error(f"Hand {game.hand_number} ended with error: {hand_outcome['error']}")
                 # Append error info?
                 per_hand_results.append({"hand": game.hand_number, "error": hand_outcome['error']})
            else:
                 logger.error(f"Hand {game.hand_number} completed but no outcome recorded.")
                 per_hand_results.append({"hand": game.hand_number, "error": "Outcome missing"})


        except Exception as hand_err:
            logger.exception(f"Unexpected error during hand {game.hand_number}: {hand_err}. Stopping game run.")
            game.game_complete = True # Mark as incomplete run due to error
            per_hand_results.append({"hand": game.hand_number, "error": f"Runtime error: {hand_err}"})
            break # Exit the loop

    # --- End of Main Hand Loop ---
    game.game_complete = True # Mark game as finished
    logger.info(f"Non-interactive game run finished. Target hands: {num_hands}. Hands attempted: {game.hand_number}. Hands completed successfully: {hands_completed_successfully}.")

    # --- Calculate Aggregate Results ---
    aggregate_results = {}
    if hands_completed_successfully > 0:
         aggregate_results = {
             f"{name}_total_net_winnings": total_net_winnings[name] for name in player_ids
         }
         aggregate_results.update({
             f"{name}_avg_net_winnings_per_hand": total_net_winnings[name] / hands_completed_successfully for name in player_ids
         })
    else: # Handle case where no hands completed
         aggregate_results = {f"{name}_total_net_winnings": 0.0 for name in player_ids}
         aggregate_results.update({f"{name}_avg_net_winnings_per_hand": 0.0 for name in player_ids})

    logger.info(f"Aggregate Results: {aggregate_results}")

    # Close the game log file
    close_game_log()

    # --- Prepare Final Return Value ---
    final_result = {
        "status": "complete" if hands_completed_successfully == num_hands else "incomplete",
        "hands_played": hands_completed_successfully,
        "target_hands": num_hands,
        "player_models": {}, # Runner should fill this
        "game_config": { # Include game config
             'player1_name': game.player1_name, 'player2_name': game.player2_name,
             'starting_stack': game.starting_stack, 'small_blind': game.small_blind,
             'big_blind': game.big_blind, 'random_seed': game._current_seed
        },
        "per_hand_results": per_hand_results,
        "aggregate_results": aggregate_results
    }

    return final_result


# Example usage block (optional)
if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    from model_orchestrator.utils import init_game_log
    # Make sure logs directory exists
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs") # Relative path
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    init_game_log(log_dir)

    # Simple agent for testing
    def random_agent(state, valid_actions):
        # Ensure valid_actions is not empty
        if not valid_actions:
             logger.error(f"RandomAgent ({state['active_player']}) received empty valid_actions. State: {state}")
             # This shouldn't happen if game logic is correct, return fold as safest fallback
             return {"action_type": "fold"}

        chosen = random.choice(valid_actions)
        if chosen["action_type"] == "raise":
            min_r = chosen.get("min_amount", 1) # Min raise amount
            max_r = chosen.get("max_amount", 1) # Max raise amount
            # Ensure min <= max, else use min
            if min_r > max_r: max_r = min_r
            try:
                chosen["amount"] = random.randint(min_r, max_r)
            except ValueError as e:
                 logger.warning(f"RandomAgent ({state['active_player']}) randInt error: min={min_r}, max={max_r}. Using min. Error: {e}")
                 chosen["amount"] = min_r
        # logger.debug(f"RandomAgent ({state['active_player']}) chooses: {chosen}")
        return chosen

    # Initialize game
    test_game = HeadsUpPoker(random_seed=42, starting_stack=1000, small_blind=10, big_blind=20)

    # Run the game
    num_test_hands = 5
    result = run_non_interactive_game(game=test_game, player1_agent=random_agent, player2_agent=random_agent, num_hands=num_test_hands)

    # Print final results
    print("\n--- FINAL GAME RUN RESULT ---")
    print(json.dumps(result, indent=2))

    # Example: Accessing average winnings
    p1_avg = result.get("aggregate_results", {}).get("Player1_avg_net_winnings_per_hand", "N/A")
    p2_avg = result.get("aggregate_results", {}).get("Player2_avg_net_winnings_per_hand", "N/A")
    print(f"\nPlayer1 Avg Net Winnings/Hand: {p1_avg:.2f}")
    print(f"Player2 Avg Net Winnings/Hand: {p2_avg:.2f}")