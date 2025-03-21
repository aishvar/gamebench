import random
import itertools
import json
import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from game_engines.base_game import BaseGame

# Card and Deck Handling
SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_TO_VALUE = {rank: i for i, rank in enumerate(RANKS, start=2)}


class Deck:
    def __init__(self, random_seed=None):
        self.cards = [(rank, suit) for suit in SUITS for rank in RANKS]
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(self.cards)

    def draw(self, num=1):
        if num > len(self.cards):
            raise ValueError(f"Cannot draw {num} cards, only {len(self.cards)} remaining")
        return [self.cards.pop() for _ in range(num)]
    
    def to_dict(self):
        return {'cards': self.cards}
    
    @classmethod
    def from_dict(cls, data):
        deck = cls(random_seed=None)  # Create with default random seed
        deck.cards = data['cards']
        return deck


# Player Representation
class Player:
    def __init__(self, name, stack):
        self.name = name
        self.stack = stack
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

    def bet(self, amount):
        if amount < 0:
            raise ValueError("Bet amount cannot be negative")
        bet_amount = min(amount, self.stack)
        self.stack -= bet_amount
        self.current_bet += bet_amount
        if self.stack == 0:
            self.all_in = True
        return bet_amount

    def fold(self):
        self.folded = True

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False
        
    def to_dict(self):
        return {
            'name': self.name,
            'stack': self.stack,
            'hole_cards': self.hole_cards,
            'current_bet': self.current_bet,
            'folded': self.folded,
            'all_in': self.all_in
        }
    
    @classmethod
    def from_dict(cls, data):
        player = cls(data['name'], data['stack'])
        player.hole_cards = data['hole_cards']
        player.current_bet = data['current_bet']
        player.folded = data['folded']
        player.all_in = data['all_in']
        return player


# Hand Evaluation
class HandEvaluator:
    @staticmethod
    def evaluate_five_card_hand(hand):
        values = sorted([RANK_TO_VALUE[rank] for rank, suit in hand], reverse=True)
        suits = [suit for rank, suit in hand]
        value_counts = Counter(values)
        counts = sorted(value_counts.values(), reverse=True)
        unique_vals = sorted(set(values), reverse=True)
        
        is_flush = len(set(suits)) == 1
        
        # Check for straight (account for Ace-low)
        sorted_vals = sorted(set(values))
        is_straight = False
        straight_high = None
        if len(sorted_vals) >= 5:
            for i in range(len(sorted_vals) - 4):
                window = sorted_vals[i:i+5]
                if window == list(range(window[0], window[0] + 5)):
                    is_straight = True
                    straight_high = window[-1]
            # Ace-low straight check
            if set([14, 2, 3, 4, 5]).issubset(set(values)):
                is_straight = True
                straight_high = 5
        
        if is_flush and is_straight:
            return (8, (straight_high,))
        if 4 in counts:
            quad = [val for val, cnt in value_counts.items() if cnt == 4][0]
            kicker = max([val for val in values if val != quad])
            return (7, (quad, kicker))
        if 3 in counts and 2 in counts:
            trip = max([val for val, cnt in value_counts.items() if cnt == 3])
            pair = max([val for val, cnt in value_counts.items() if cnt == 2])
            return (6, (trip, pair))
        if is_flush:
            return (5, tuple(sorted(values, reverse=True)))
        if is_straight:
            return (4, (straight_high,))
        if 3 in counts:
            trip = [val for val, cnt in value_counts.items() if cnt == 3][0]
            kickers = sorted([val for val in values if val != trip], reverse=True)
            return (3, (trip,) + tuple(kickers))
        if counts.count(2) >= 2:
            pairs = sorted([val for val, cnt in value_counts.items() if cnt == 2], reverse=True)
            kicker = max([val for val in values if val not in pairs])
            return (2, tuple(pairs) + (kicker,))
        if 2 in counts:
            pair = [val for val, cnt in value_counts.items() if cnt == 2][0]
            kickers = sorted([val for val in values if val != pair], reverse=True)
            return (1, (pair,) + tuple(kickers))
        return (0, tuple(values))

    @staticmethod
    def best_hand_rank(cards):
        if len(cards) <= 5:
            return HandEvaluator.evaluate_five_card_hand(cards)
        best_rank = (-1, ())
        for combo in itertools.combinations(cards, 5):
            rank = HandEvaluator.evaluate_five_card_hand(list(combo))
            if rank > best_rank:
                best_rank = rank
        return best_rank

    @staticmethod
    def hand_description(rank_tuple):
        rank_names = {
            8: "Straight Flush",
            7: "Four of a Kind",
            6: "Full House",
            5: "Flush",
            4: "Straight",
            3: "Three of a Kind",
            2: "Two Pair",
            1: "One Pair",
            0: "High Card"
        }
        rank_value, tiebreakers = rank_tuple
        desc = rank_names.get(rank_value, "Unknown")
        return f"{desc} {tiebreakers}"

    @staticmethod
    def determine_winner(player1, player2, community_cards):
        hand1 = HandEvaluator.best_hand_rank(player1.hole_cards + community_cards)
        hand2 = HandEvaluator.best_hand_rank(player2.hole_cards + community_cards)
        if hand1 > hand2:
            return player1, hand1, hand2
        elif hand2 > hand1:
            return player2, hand1, hand2
        else:
            return None, hand1, hand2


# HeadsUpPoker Game Implementation
class HeadsUpPoker(BaseGame):
    def __init__(self, game_id=None, players=None, random_seed=None, 
                 player1_name="Player1", player2_name="Player2", 
                 starting_stack=1000, small_blind=10, big_blind=20):
        """
        Initialize a heads-up poker game.
        
        Args:
            game_id: Unique game identifier
            players: List of player identifiers
            random_seed: Seed for reproducibility
            player1_name: Name of first player
            player2_name: Name of second player
            starting_stack: Starting chip count for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Call parent initializer
        super().__init__(game_id, players=[player1_name, player2_name], random_seed=random_seed)
    
    def _initialize_game_state(self):
        """Initialize the poker game state"""
        self.deck = Deck(self.random_seed)
        self.players_obj = [
            Player(self.player1_name, self.starting_stack), 
            Player(self.player2_name, self.starting_stack)
        ]
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_index = 0  # Dealer button position, switches each hand
        self.active_player_index = 1  # Non-dealer acts first after pre-flop
        self.last_raiser = None
        self.stage = "pre-deal"  # pre-deal, pre-flop, flop, turn, river, showdown
        self.hand_number = 1
        self.hand_complete = False
        self.game_complete = False
        
    def reset(self):
        """Reset the game to initial state"""
        self._initialize_game_state()
        self.history = []
        
    def reset_hand(self):
        """Reset for a new hand while preserving player stacks"""
        # Log the reset
        self.log_event("hand_complete", {
            "hand_number": self.hand_number,
            "player_stacks": {
                p.name: p.stack for p in self.players_obj
            }
        })
        
        # Check if game is over based on stacks
        if any(p.stack <= 0 for p in self.players_obj):
            self.game_complete = True
            return
            
        # Prepare for new hand
        self.hand_number += 1
        self.deck = Deck(self.random_seed)
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.last_raiser = None
        self.dealer_index = 1 - self.dealer_index  # Move dealer button
        self.active_player_index = 1 - self.dealer_index  # Non-dealer acts first after pre-flop
        self.stage = "pre-deal"
        self.hand_complete = False
        
        # Reset player state for new hand
        for player in self.players_obj:
            player.reset_for_new_hand()
    
    def start_hand(self):
        """Start a new hand, dealing cards and posting blinds"""
        if self.stage != "pre-deal":
            return False
            
        # Post blinds
        small_blind_player = self.players_obj[self.dealer_index]
        big_blind_player = self.players_obj[1 - self.dealer_index]
        
        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)
        self.pot += sb_amount + bb_amount
        self.current_bet = self.big_blind
        
        # Log blind posting
        self.log_event("blinds_posted", {
            "small_blind": {"player": small_blind_player.name, "amount": sb_amount},
            "big_blind": {"player": big_blind_player.name, "amount": bb_amount}
        })
        
        # Deal hole cards
        for player in self.players_obj:
            player.hole_cards = self.deck.draw(2)
        
        # Log dealing
        self.log_event("hole_cards_dealt", {"hand_number": self.hand_number, **{player.name: player.hole_cards for player in self.players_obj}})
        
        # Set stage to pre-flop
        self.stage = "pre-flop"
        self.active_player_index = self.dealer_index  # Small blind acts first pre-flop
        
        return True
    
    def deal_community_cards(self):
        """Deal community cards based on current stage"""
        if self.stage == "pre-flop":
            # Deal flop (3 cards)
            self.community_cards = self.deck.draw(3)
            self.stage = "flop"
            self.log_event("flop_dealt", {"cards": self.community_cards})
            
            # Reset betting for new round
            for player in self.players_obj:
                player.current_bet = 0
            self.current_bet = 0
            self.last_raiser = None
            
            # Set active player (non-dealer acts first post-flop)
            self.active_player_index = 1 - self.dealer_index
            
        elif self.stage == "flop":
            # Deal turn (1 card)
            self.community_cards.append(self.deck.draw(1)[0])
            self.stage = "turn"
            self.log_event("turn_dealt", {"card": self.community_cards[-1]})
            
            # Reset betting for new round
            for player in self.players_obj:
                player.current_bet = 0
            self.current_bet = 0
            self.last_raiser = None
            
            # Non-dealer acts first
            self.active_player_index = 1 - self.dealer_index
            
        elif self.stage == "turn":
            # Deal river (1 card)
            self.community_cards.append(self.deck.draw(1)[0])
            self.stage = "river"
            self.log_event("river_dealt", {"card": self.community_cards[-1]})
            
            # Reset betting for new round
            for player in self.players_obj:
                player.current_bet = 0
            self.current_bet = 0
            self.last_raiser = None
            
            # Non-dealer acts first
            self.active_player_index = 1 - self.dealer_index
            
        return True
    
    def get_state(self, player_id=None):
        """Get the current game state from a specific player's perspective"""
        # Base game state visible to all players
        state = {
            "hand_number": self.hand_number,
            "stage": self.stage,
            "dealer": self.players_obj[self.dealer_index].name,
            "active_player": self.players_obj[self.active_player_index].name if not self.hand_complete else None,
            "pot": self.pot,
            "community_cards": self.community_cards,
            "current_bet": self.current_bet,
            "players": {}
        }
    
        # Add player information
        for player in self.players_obj:
            player_info = {
                "stack": player.stack,
                "current_bet": player.current_bet,
                "folded": player.folded,
                "all_in": player.all_in
            }
            
            # Include hole cards only if:
            # 1. No specific player_id is provided (full state view for non-interactive mode)
            # 2. The player_id matches this player
            # 3. We're in showdown or hand is complete
            if (player_id is None or 
                player.name == player_id or 
                self.stage == "showdown" or 
                self.hand_complete):
                player_info["hole_cards"] = player.hole_cards
            
            state["players"][player.name] = player_info
        
        return state
    
    def get_valid_actions(self, player_id=None):
        """Get valid actions for the current player"""
        # If hand is complete, no actions available
        if self.hand_complete or self.stage == "showdown":
            return []
            
        # If player_id is provided, check if it's this player's turn
        if player_id and self.players_obj[self.active_player_index].name != player_id:
            return []
            
        player = self.players_obj[self.active_player_index]
        if player.folded or player.all_in:
            return []
            
        # Calculate call amount
        call_amount = self.current_bet - player.current_bet
        
        valid_actions = []
        
        # Fold is always an option unless check is free
        if call_amount > 0:
            valid_actions.append({"action_type": "fold"})
        
        # Call/Check
        if call_amount == 0:
            valid_actions.append({"action_type": "check"})
        else:
            valid_actions.append({
                "action_type": "call", 
                "amount": call_amount
            })
        
        # Raise, if player has enough chips
        if player.stack > call_amount:
            # Determine minimum raise
            min_raise = self.big_blind
            if self.current_bet > 0:
                min_raise = self.current_bet * 2 - player.current_bet
                
            # Ensure minimum raise doesn't exceed player's stack
            min_raise = min(min_raise, player.stack)
            
            valid_actions.append({
                "action_type": "raise",
                "min_amount": min_raise,
                "max_amount": player.stack
            })
            
        return valid_actions
    
    def apply_action(self, action, player_id=None):
        """Apply a player action to the game state"""
        # Validate player's turn if player_id provided
        if player_id and self.players_obj[self.active_player_index].name != player_id:
            return False
            
        # Handle completed hand
        if self.hand_complete:
            return False
            
        player = self.players_obj[self.active_player_index]
        if player.folded or player.all_in:
            return False
            
        action_type = action.get("action_type")
        if not action_type:
            return False
            
        # Process action
        if action_type == "fold":
            player.fold()
            self.log_event("action", {
                "player": player.name,
                "action": "fold"
            })
            
            # Handle immediate win when opponent folds
            other_player = self.players_obj[1 - self.active_player_index]
            if not other_player.folded:
                other_player.stack += self.pot
                self.log_event("hand_result", {
                    "winner": other_player.name,
                    "reason": "opponent_folded",
                    "pot": self.pot
                })
                self.hand_complete = True
                return True
                
        elif action_type == "check":
            if self.current_bet != player.current_bet:
                return False  # Cannot check if there's a bet to call
                
            self.log_event("action", {
                "player": player.name,
                "action": "check"
            })
            
        elif action_type == "call":
            call_amount = self.current_bet - player.current_bet
            if call_amount <= 0:
                return False  # Cannot call if there's nothing to call
                
            amount_bet = player.bet(call_amount)
            self.pot += amount_bet
            
            self.log_event("action", {
                "player": player.name,
                "action": "call",
                "amount": amount_bet
            })
            
        elif action_type == "raise":
            amount = action.get("amount")
            if not amount or not isinstance(amount, (int, float)):
                return False
                
            # Validate raise amount
            call_amount = self.current_bet - player.current_bet
            min_raise = self.big_blind
            if self.current_bet > 0:
                min_raise = self.current_bet * 2 - player.current_bet
                
            if amount < min_raise or amount > player.stack:
                return False
                
            # Process the raise
            amount_bet = player.bet(amount)
            self.pot += amount_bet
            self.current_bet = player.current_bet
            self.last_raiser = self.active_player_index
            
            self.log_event("action", {
                "player": player.name,
                "action": "raise",
                "amount": amount_bet,
                "current_bet": self.current_bet
            })
            
        else:
            return False  # Unknown action
            
        # Move to next player
        self.active_player_index = 1 - self.active_player_index
        
        # Check if betting round is complete
        self._check_betting_round_complete()
        
        return True
    
    def _check_betting_round_complete(self):
        """Check if the current betting round is complete"""
        # Count active (not folded or all-in) players
        active_players = [p for p in self.players_obj if not p.folded and not p.all_in]
        
        # If only one active player, end the hand
        if len(active_players) <= 1:
            self._handle_showdown()
            return True
            
        # If all active players have matched the current bet, move to next stage
        if all(p.current_bet == self.current_bet for p in active_players):
            if self.stage == "pre-flop":
                self.deal_community_cards()  # Deal flop
            elif self.stage == "flop":
                self.deal_community_cards()  # Deal turn
            elif self.stage == "turn":
                self.deal_community_cards()  # Deal river
            elif self.stage == "river":
                self._handle_showdown()
                
            return True
            
        return False
    
    def _handle_showdown(self):
        """Handle showdown and determine winner"""
        self.stage = "showdown"
        
        # If only one player isn't folded, they win automatically
        active_players = [p for p in self.players_obj if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            winner.stack += self.pot
            
            self.log_event("hand_result", {
                "winner": winner.name,
                "reason": "opponent_folded",
                "pot": self.pot
            })
            
            self.hand_complete = True
            return
            
        # Compare hands for showdown
        winner, hand1, hand2 = HandEvaluator.determine_winner(
            self.players_obj[0], self.players_obj[1], self.community_cards
        )
        
        # Log hand comparison
        self.log_event("showdown", {
            "community_cards": self.community_cards,
            "players": {
                self.players_obj[0].name: {
                    "hole_cards": self.players_obj[0].hole_cards,
                    "hand_rank": hand1,
                    "hand_description": HandEvaluator.hand_description(hand1)
                },
                self.players_obj[1].name: {
                    "hole_cards": self.players_obj[1].hole_cards,
                    "hand_rank": hand2,
                    "hand_description": HandEvaluator.hand_description(hand2)
                }
            }
        })
        
        # Award pot
        if winner is None:
            # Split pot for tie
            split_amount = self.pot // 2
            self.players_obj[0].stack += split_amount
            self.players_obj[1].stack += self.pot - split_amount  # Account for odd chips
            
            self.log_event("hand_result", {
                "result": "tie",
                "pot": self.pot,
                "split_amounts": {
                    self.players_obj[0].name: split_amount,
                    self.players_obj[1].name: self.pot - split_amount
                }
            })
        else:
            winner.stack += self.pot
            
            self.log_event("hand_result", {
                "winner": winner.name,
                "reason": "better_hand",
                "pot": self.pot,
                "winning_hand": HandEvaluator.hand_description(
                    hand1 if winner == self.players_obj[0] else hand2
                )
            })
        
        self.hand_complete = True
    
    def is_terminal(self):
        """Check if the game is in a terminal state"""
        return self.game_complete
    
    def get_rewards(self):
        """Get rewards for each player"""
        # For poker, rewards are the change in stack size from starting
        rewards = {}
        for player in self.players_obj:
            rewards[player.name] = player.stack - self.starting_stack
        return rewards
    
    def get_result(self):
        """Get final game result"""
        if not self.is_terminal():
            return {"status": "in_progress"}
            
        return {
            "status": "complete",
            "players": {
                player.name: {
                    "final_stack": player.stack,
                    "net_winnings": player.stack - self.starting_stack
                }
                for player in self.players_obj
            },
            "winner": max(self.players_obj, key=lambda p: p.stack).name
        }
    
    def _restore_from_state(self, state_data):
        """Restore game from serialized state"""
        # Restore deck
        self.deck = Deck.from_dict(state_data.get('deck', {'cards': []}))
        
        # Restore game state
        self.pot = state_data.get('pot', 0)
        self.community_cards = state_data.get('community_cards', [])
        self.current_bet = state_data.get('current_bet', 0)
        self.dealer_index = state_data.get('dealer_index', 0)
        self.active_player_index = state_data.get('active_player_index', 0)
        self.last_raiser = state_data.get('last_raiser')
        self.stage = state_data.get('stage', 'pre-deal')
        self.hand_number = state_data.get('hand_number', 1)
        self.hand_complete = state_data.get('hand_complete', False)
        self.game_complete = state_data.get('game_complete', False)
        
        # Restore players
        players_data = state_data.get('players_data', [])
        if players_data:
            self.players_obj = [Player.from_dict(p) for p in players_data]
        
        # Restore history
        self.history = state_data.get('history', [])
    
    def to_json(self):
        """Serialize the game to JSON string"""
        state = {
            'game_id': self.game_id,
            'players': self.players,
            'random_seed': self.random_seed,
            'player1_name': self.player1_name,
            'player2_name': self.player2_name,
            'starting_stack': self.starting_stack,
            'small_blind': self.small_blind,
            'big_blind': self.big_blind,
            'pot': self.pot,
            'community_cards': self.community_cards,
            'current_bet': self.current_bet,
            'dealer_index': self.dealer_index,
            'active_player_index': self.active_player_index,
            'last_raiser': self.last_raiser,
            'stage': self.stage,
            'hand_number': self.hand_number,
            'hand_complete': self.hand_complete,
            'game_complete': self.game_complete,
            'deck': self.deck.to_dict(),
            'players_data': [p.to_dict() for p in self.players_obj],
            'history': self.history,
            'is_interactive': self.is_interactive
        }
        return json.dumps(state, indent=2)


# Non-interactive mode for benchmarking
def run_non_interactive_game(player1_agent=None, player2_agent=None, num_hands=1, random_seed=None):
    """
    Run a complete poker game without user interaction.
    
    Args:
        player1_agent: Function that takes game state and returns action
        player2_agent: Function that takes game state and returns action
        num_hands: Number of hands to play
        random_seed: Random seed for reproducibility
        
    Returns:
        Game result data
    """
    from model_orchestrator.utils import log_initial_state, log_hand_result, close_game_log
    if not player1_agent or not player2_agent:
        raise ValueError("Both player agents must be provided")
        
    game = HeadsUpPoker(random_seed=random_seed)
    game.set_interactive_mode(False)
    
    agents = [player1_agent, player2_agent]
    
    # Track results for each hand
    hand_results = []
    total_hands_played = 0
    
    for i in range(num_hands):
        if game.is_terminal():
            break
            
        # Start a new hand
        game.start_hand()
        log_initial_state(game.get_state(), game.hand_number)
        total_hands_played += 1
        
        # DEBUG: Log hole cards after starting the hand
        logger.info(f"Hand {game.hand_number} started")
        logger.info(f"Player 1 ({game.players_obj[0].name}) cards: {game.players_obj[0].hole_cards}")
        logger.info(f"Player 2 ({game.players_obj[1].name}) cards: {game.players_obj[1].hole_cards}")
        
        while not game.hand_complete:
            # Get current player
            active_idx = game.active_player_index
            active_player = game.players_obj[active_idx]
            agent = agents[active_idx]
            
            # Skip if player can't act
            if active_player.folded or active_player.all_in:
                # This should trigger the betting round complete check
                game._check_betting_round_complete()
                continue
                
            # Get game state from player perspective
            state = game.get_state(active_player.name)
            
            # DEBUG: Verify hole cards in the state
            player_info = state["players"].get(active_player.name, {})
            logger.info(f"Agent {active_player.name} state has hole cards: {player_info.get('hole_cards', None)}")
            
            valid_actions = game.get_valid_actions(active_player.name)
            
            if not valid_actions:
                # No valid actions available, check if round is complete
                game._check_betting_round_complete()
                continue
                
            # Get agent's action
            action = agent(state, valid_actions)
            
            # Apply the action
            success = game.apply_action(action, active_player.name)
            if not success:
                # If invalid action, default to least risky action
                if any(a.get("action_type") == "check" for a in valid_actions):
                    game.apply_action({"action_type": "check"}, active_player.name)
                else:
                    # Find a call action if available
                    call_actions = [a for a in valid_actions if a.get("action_type") == "call"]
                    if call_actions:
                        game.apply_action(call_actions[0], active_player.name)
                    else:
                        # Last resort: fold
                        game.apply_action({"action_type": "fold"}, active_player.name)
        
        # Record hand result before resetting
        winner_name = None
        for event in reversed(game.history):
            if event.get("event_type") == "hand_result":
                data = event.get("data", {})
                winner_name = data.get("winner")
                if winner_name:
                    break
        
        hand_result = {
            "hand_number": game.hand_number,
            "players": {
                player.name: {
                    "stack": player.stack,
                    "net_change": player.stack - game.starting_stack
                }
                for player in game.players_obj
            },
            "winner": winner_name
        }
        hand_results.append(hand_result)
        if hand_result:
            log_hand_result(hand_result, game.get_state())
        
        # Reset for next hand
        game.reset_hand()
    
    # Create final result with detailed information
    final_result = {
        "status": "complete",
        "players": {
            player.name: {
                "final_stack": player.stack,
                "net_winnings": player.stack - game.starting_stack
            }
            for player in game.players_obj
        },
        "hands_played": total_hands_played,
        "hand_results": hand_results
    }
    
    # Determine overall winner
    player1 = game.players_obj[0]
    player2 = game.players_obj[1]
    
    if player1.stack > player2.stack:
        final_result["winner"] = player1.name
    elif player2.stack > player1.stack:
        final_result["winner"] = player2.name
    else:
        final_result["winner"] = None  # Tie
        
    # Calculate win rates
    p1_wins = sum(1 for result in hand_results if result["winner"] == player1.name)
    p2_wins = sum(1 for result in hand_results if result["winner"] == player2.name)
    
    if total_hands_played > 0:
        final_result["win_rates"] = {
            player1.name: p1_wins / total_hands_played,
            player2.name: p2_wins / total_hands_played
        }
    
    close_game_log()
    return final_result

# Example Usage
if __name__ == "__main__":
    # Example of simple agent function that always calls
    def always_call_agent(state, valid_actions):
        for action in valid_actions:
            if action["action_type"] == "call":
                return action
            elif action["action_type"] == "check":
                return action
        # If can't call or check, fold
        return {"action_type": "fold"}
    
    # Example of simple agent function that randomly selects an action
    def random_agent(state, valid_actions):
        return random.choice(valid_actions)
    
    # Run a non-interactive game
    result = run_non_interactive_game(
        player1_agent=always_call_agent,
        player2_agent=random_agent,
        num_hands=10,
        random_seed=42
    )
    
    print("Game result:", json.dumps(result, indent=2))