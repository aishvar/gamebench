#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
import time
import os
import json
import re
from typing import List, Tuple, Dict, Optional, NamedTuple, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import fcntl  # Added for file locking on logging

try:
    from llm_client import LLMClient, parse_response_text, log_llm_call, LOGS_DIR
except ImportError:
    # Provide a dummy implementation if llm_client is missing, for basic structure testing
    print("Warning: llm_client.py not found. Using dummy implementation.")
    raise SystemExit("Aborting: llm_client.py is required to run this program.")



LOGS_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiarsPokerGame")
GAME_LOGS_DIR = os.path.join(LOGS_DIR, "liars_poker_multi")
os.makedirs(GAME_LOGS_DIR, exist_ok=True)

class Bid(NamedTuple):
    """Represents a bid in Liar's Poker."""
    quantity: int
    digit: int

    def __str__(self):
        return f"{self.quantity} {self.digit}s"

    def is_higher_than(self, other: Optional['Bid']) -> bool:
        if other is None:
            return True
        if self.quantity > other.quantity:
            return True
        if self.quantity == other.quantity and self.digit > other.digit:
            return True
        return False

@dataclass
class Player:
    """Represents a player in the game."""
    player_id: int
    strategy_type: str
    model_config: Optional[Dict[str, str]] = None
    client: Optional[LLMClient] = field(default=None, init=False)
    hand: List[int] = field(default_factory=list)
    original_order: int = 0
    effective_strategy: Optional[str] = None
    effective_model_config: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.strategy_type == 'random':
            logger.info(f"Player {self.player_id} strategy is 'random' - actual sub-strategy will be chosen each round.")
            return
        if self.strategy_type == 'llm':
            if not self.model_config:
                raise ValueError(f"Player {self.player_id} has strategy 'llm' but no model_config provided.")
            logger.info(f"Initializing LLM client for Player {self.player_id}: {self.model_config}")
            try:
                self.client = LLMClient(
                    provider=self.model_config["provider"],
                    model=self.model_config["model"],
                    max_tokens=8192,
                    temperature=0.5,
                    max_retries=2,
                    timeout=60
                )
            except ValueError as e:
                logger.error(f"Failed to initialize LLM client for Player {self.player_id}: {e}")
                raise
            except KeyError as e:
                logger.error(f"Missing key in model_config for Player {self.player_id}: {e}")
                raise ValueError(f"Invalid model_config for Player {self.player_id}: {self.model_config}")
        elif self.strategy_type in ('naive_5050', 'intelligent'):
            logger.info(f"Player {self.player_id} initialized with {self.strategy_type} strategy.")
        else:
            raise ValueError(f"Unknown strategy_type '{self.strategy_type}' for Player {self.player_id}")

    def get_display_name(self) -> str:
        if self.strategy_type == 'random':
            if self.effective_strategy == 'llm' and self.effective_model_config:
                provider = self.effective_model_config.get('provider', 'unknown')
                model = self.effective_model_config.get('model', 'unknown')
                if 'quasar-alpha' in model:
                    return 'openrouter/quasar-alpha'
                if 'gemini-2.5-flash-preview:thinking' in model:
                    return 'openrouter/gemini-2.5-flash-preview-thinking'
                if provider == "openrouter":
                    model = model.replace("openrouter/", "", 1)
                return model
            elif self.effective_strategy == 'naive_5050':
                return "Naive 50/50"
            elif self.effective_strategy == 'intelligent':
                return "Intelligent"
            else:
                return "Random(Undecided)"
        elif self.strategy_type == 'llm' and self.model_config:
            provider = self.model_config.get('provider', 'unknown')
            model = self.model_config.get('model', 'unknown')
            if 'quasar-alpha' in model:
                return 'openrouter/quasar-alpha'
            if 'gemini-2.5-flash-preview:thinking' in model:
                    return 'openrouter/gemini-2.5-flash-preview-thinking'
            if provider == "openrouter":
                return model.replace("openrouter/", "", 1)
            if model.startswith(provider + "/"):
                return model
            else:
                return f"{model}"
        elif self.strategy_type == 'naive_5050':
            return "Naive 50/50"
        elif self.strategy_type == 'intelligent':
            return "Intelligent"
        else:
            return f"Unknown Strategy ({self.strategy_type})"

class LiarsPokerGame:
    """Manages the Liar's Poker game simulation."""
    MAX_DIGITS_PER_HAND = 8
    MAX_ACTION_PARSE_ATTEMPTS = 2
    COMMON_CONFIGS = [
        # === OpenAI ===
        #{"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-2024-11-20"},
        #{"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-2025-04-14"},
        {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
        {"strategy_type": "llm", "provider": "openai", "model": "gpt-4.1-nano-2025-04-14"},
        #{"strategy_type": "llm", "provider": "openai", "model": "o3-mini-2025-01-31"},
        {"strategy_type": "llm", "provider": "openai", "model": "o4-mini-2025-04-16"},
        {"strategy_type": "llm", "provider": "openai", "model": "o3-2025-04-16"},

        # === Anthropic ===
        {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},

        # === DeepSeek ===
        {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-chat-v3-0324:floor"},
        #{"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1:floor"},
        #{"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-qwen-32b:floor"},
        #{"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1-distill-llama-70b:floor"},

        # === LLaMA ===
        {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.1-8b-instruct:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-maverick:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "meta-llama/llama-4-scout:floor"},

        # === Google ===
        {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-pro-preview-03-25:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemma-3-27b-it:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.0-flash-001:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-flash-preview:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "google/gemini-2.5-flash-preview:thinking"},

        # === Qwen ===
        #{"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwq-32b:nitro"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwen3-30b-a3b-04-28:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwen3-32b:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwen3-235b-a22b-04-28:floor"},

        # === Miscellaneous ===
        {"strategy_type": "llm", "provider": "openrouter", "model": "x-ai/grok-3-beta:floor"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "mistralai/mistral-small-3.1-24b-instruct:floor"},
    ]

    def __init__(self, player_configs: List[Dict[str, Any]]):
        if len(player_configs) != 6:
            raise ValueError("Progressive Elimination game requires exactly 6 players.")
        self.players: List[Player] = []
        for i, config in enumerate(player_configs):
            strategy = config.get("strategy_type")
            if strategy == 'llm':
                model_cfg = {"provider": config["provider"], "model": config["model"]}
                self.players.append(Player(player_id=i, strategy_type='llm', model_config=model_cfg, original_order=i))
            elif strategy == 'naive_5050':
                self.players.append(Player(player_id=i, strategy_type='naive_5050', original_order=i))
            elif strategy == 'random':
                self.players.append(Player(player_id=i, strategy_type='random', original_order=i))
            elif strategy == 'intelligent':
                self.players.append(Player(player_id=i, strategy_type='intelligent', original_order=i))
            else:
                raise ValueError(f"Invalid player configuration at index {i}: missing or unknown strategy_type. Config: {config}")
        self.current_player_index: int = 0
        self.current_bid: Optional[Bid] = None
        self.bid_history: List[Tuple[int, Bid]] = []
        self.round_log: List[str] = []
        self.total_digits_in_play: int = 0
        self.game_active: bool = False
        self.active_players: List[Player] = []
        self.ranks: Dict[int, int] = {} # Maps original_order -> rank (1st, 2nd, etc.)
        self.next_rank_to_assign: int = 6
        # Stores {'type': 'EVENT_TYPE', 'details': {...}} dicts chronologically for the whole game
        self.event_history: List[Dict[str, Any]] = field(default_factory=list)

    def _log_round_event(self, message: str):
        logger.info(message)
        self.round_log.append(message)

    def _generate_hands(self) -> List[List[int]]:
        num_players = len(self.players)
        possible_digits = list(range(10)) * (num_players * self.MAX_DIGITS_PER_HAND // 10 + num_players)
        random.shuffle(possible_digits)
        hands_int = []
        start_index = 0
        for _ in range(num_players):
            hand = possible_digits[start_index : start_index + self.MAX_DIGITS_PER_HAND]
            if len(hand) != self.MAX_DIGITS_PER_HAND:
                raise RuntimeError(f"Could not generate enough digits for hands.")
            hands_int.append(hand)
            start_index += self.MAX_DIGITS_PER_HAND

        hand_strs = {"".join(map(str, sorted(h))) for h in hands_int}
        if len(hand_strs) != num_players:
            self._log_round_event("Warning: Duplicate hands generated, proceeding anyway.")
        self._log_round_event(f"Generated {num_players} hands.")
        return hands_int

    def _setup_round(self):
        self._log_round_event("--- Starting New Round ---")
        original_players = self.players.copy()
        random.shuffle(original_players)
        players_for_round = []
        used_strategies = set()

        for i, original_player in enumerate(original_players):
            new_player = Player(
                player_id=i,
                strategy_type=original_player.strategy_type,
                model_config=(original_player.model_config.copy() if original_player.model_config else None),
                original_order=original_player.original_order,
            )
            if new_player.strategy_type != 'random':
                if new_player.strategy_type == 'naive_5050':
                    used_strategies.add(('naive_5050', None))
                elif new_player.strategy_type == 'llm':
                    provider = new_player.model_config["provider"]
                    model = new_player.model_config["model"]
                    used_strategies.add(('llm', (provider, model)))
                elif new_player.strategy_type == 'intelligent':
                    used_strategies.add(('intelligent', None))
                players_for_round.append(new_player)
            else:
                possible_choices = []
                if ('naive_5050', None) not in used_strategies:
                    possible_choices.append({"strategy_type": "naive_5050"})
                #if ('intelligent', None) not in used_strategies:
                #    possible_choices.append({"strategy_type": "intelligent"})
                for cfg in self.COMMON_CONFIGS:
                    if cfg['strategy_type'] == 'llm':
                        model_key = (cfg['provider'], cfg['model'])
                        if ('llm', model_key) not in used_strategies:
                            possible_choices.append(cfg)
                if not possible_choices:
                    possible_choices = self.COMMON_CONFIGS.copy()

                selected_cfg = random.choice(possible_choices)
                if selected_cfg['strategy_type'] == 'naive_5050':
                    new_player.effective_strategy = 'naive_5050'
                    new_player.effective_model_config = None
                    used_strategies.add(('naive_5050', None))
                    self._log_round_event(f"Random player {i} chose sub-strategy: Naive 50/50")
                elif selected_cfg['strategy_type'] == 'intelligent':
                    new_player.effective_strategy = 'intelligent'
                    new_player.effective_model_config = None
                    used_strategies.add(('intelligent', None))
                    self._log_round_event(f"Random player {i} chose sub-strategy: Intelligent")
                else:
                    new_player.effective_strategy = 'llm'
                    new_player.effective_model_config = {'provider': selected_cfg['provider'], 'model': selected_cfg['model']}
                    used_strategies.add(('llm', (selected_cfg['provider'], selected_cfg['model'])))
                    self._log_round_event(f"Random player {i} chose LLM: {selected_cfg['provider']}/{selected_cfg['model']}")
                    try:
                        new_player.client = LLMClient(
                            provider=selected_cfg["provider"],
                            model=selected_cfg["model"],
                            max_tokens=8192,
                            temperature=0.5,
                            max_retries=2,
                            timeout=60
                        )
                    except Exception as e:
                        self._log_round_event(f"Error initializing LLM client for random player {i}: {e}")
                players_for_round.append(new_player)

        self.players = players_for_round
        self._log_round_event(f"Player order: {[p.get_display_name() for p in self.players]}")
        hands = self._generate_hands()
        for i, player in enumerate(self.players):
            player.hand = hands[i]
            logger.debug(f"Player {player.player_id} ({player.get_display_name()}) Hand: {''.join(map(str, player.hand))}")
        self._log_round_event(f"Hands dealt to {len(self.players)} players.")
        self.current_player_index = 0
        self.current_bid = None
        self.bid_history = []
        self.round_log = self.round_log[-3:]
        self.total_digits_in_play = len(self.players) * self.MAX_DIGITS_PER_HAND
        self.game_active = True
        self._log_round_event("Round setup complete. Starting bids.")

    def _initial_setup(self):
        """Sets up the initial players, strategies, and hands for the elimination game."""
        self._log_round_event("--- Initial Game Setup ---")
        original_players = self.players.copy() # Use the initially configured players
        random.shuffle(original_players)
        players_for_game = []
        used_strategies = set()

        for i, original_player in enumerate(original_players):
            # Create new Player instances to ensure clean state, but keep original_order
            new_player = Player(
                player_id=i, # Assign temporary round ID
                strategy_type=original_player.strategy_type,
                model_config=(original_player.model_config.copy() if original_player.model_config else None),
                original_order=original_player.original_order, # Keep track of the original player
            )

            # Assign effective strategy for 'random' players
            if new_player.strategy_type == 'random':
                possible_choices = []
                if ('naive_5050', None) not in used_strategies:
                    possible_choices.append({"strategy_type": "naive_5050"})
                #if ('intelligent', None) not in used_strategies:
                #    possible_choices.append({"strategy_type": "intelligent"})
                for cfg in self.COMMON_CONFIGS:
                    if cfg['strategy_type'] == 'llm':
                        model_key = (cfg['provider'], cfg['model'])
                        if ('llm', model_key) not in used_strategies:
                            possible_choices.append(cfg)
                if not possible_choices:
                    possible_choices = self.COMMON_CONFIGS.copy() # Fallback if all unique used

                selected_cfg = random.choice(possible_choices)
                if selected_cfg['strategy_type'] == 'naive_5050':
                    new_player.effective_strategy = 'naive_5050'
                    new_player.effective_model_config = None
                    used_strategies.add(('naive_5050', None))
                    self._log_round_event(f"Initial Player {i} (Original: {new_player.original_order}, Random) chose sub-strategy: Naive 50/50")
                elif selected_cfg['strategy_type'] == 'intelligent':
                    new_player.effective_strategy = 'intelligent'
                    new_player.effective_model_config = None
                    used_strategies.add(('intelligent', None))
                    self._log_round_event(f"Initial Player {i} (Original: {new_player.original_order}, Random) chose sub-strategy: Intelligent")
                else:
                    new_player.effective_strategy = 'llm'
                    new_player.effective_model_config = {'provider': selected_cfg['provider'], 'model': selected_cfg['model']}
                    used_strategies.add(('llm', (selected_cfg['provider'], selected_cfg['model'])))
                    self._log_round_event(f"Initial Player {i} (Original: {new_player.original_order}, Random) chose LLM: {selected_cfg['provider']}/{selected_cfg['model']}")
                    # Initialize LLM client here if needed for 'random' type
                    try:
                        new_player.client = LLMClient(
                            provider=selected_cfg["provider"],
                            model=selected_cfg["model"],
                            max_tokens=8192,
                            temperature=0.5,
                            max_retries=2,
                            timeout=60
                        )
                    except Exception as e:
                        self._log_round_event(f"Error initializing LLM client for random player {i}: {e}")
                        # Handle error appropriately, maybe raise or assign a default non-LLM strategy
                        raise # Re-raise for now
            elif new_player.strategy_type == 'naive_5050':
                used_strategies.add(('naive_5050', None))
            elif new_player.strategy_type == 'llm':
                provider = new_player.model_config["provider"]
                model = new_player.model_config["model"]
                used_strategies.add(('llm', (provider, model)))
            elif new_player.strategy_type == 'intelligent':
                used_strategies.add(('intelligent', None))

            players_for_game.append(new_player)

        self.players = players_for_game # Overwrite self.players with the shuffled/configured list for the game
        self._log_round_event(f"Initial player order (Round IDs): {[f'P{p.player_id}(Orig:{p.original_order} - {p.get_display_name()})' for p in self.players]}")

        # Generate and deal hands
        hands = self._generate_hands()
        for i, player in enumerate(self.players):
            player.hand = hands[i]
            logger.debug(f"Player {player.player_id} (Original: {player.original_order} - {player.get_display_name()}) Hand: {''.join(map(str, player.hand))}")
        self._log_round_event(f"Hands dealt to {len(self.players)} players.")

        # Set initial game state (bids/history reset in play_elimination_game)
        self.current_player_index = 0
        self.game_active = True # Mark game as active
        self._log_round_event("Initial setup complete. Starting elimination game.")

    def _get_prompt_context(self, player: Player) -> Tuple[str, str, str]:
        system_message = (
            "You are a strategic player in a game of Liar's Poker. "
            "Analyze the situation, your hand, and the bidding history to make the best move. "
            "Your goal is to either make a valid higher bid or challenge the last bid if you think it's unlikely. "
            "Provide your response as a JSON object containing two keys: 'reasoning' (explaining your thought process) "
            "and 'action' (containing *only* your chosen action string: 'BID: [quantity] [digit]s' or 'CHALLENGE')."
            " Do not include any explanation or formatting outside the JSON object. Only return the JSON."
        )
        developer_message = (
            f"Liar's Poker Rules:\n"
            f"- Each of the {len(self.active_players)} active players has {self.MAX_DIGITS_PER_HAND} secret digits.\n"
            f"- Players take turns bidding on the total count of a specific digit (0-9) across ALL active players' hands.\n"
            f"- A bid consists of a quantity and a digit (e.g., '3 5s' means at least three 5s exist in total).\n"
            f"- Your bid must be strictly higher than the current bid '{self.current_bid or 'None'}'.\n"
            f"  - 'Higher' means: higher quantity (e.g., 3 9s -> 4 0s) OR same quantity but higher digit (e.g., 3 5s -> 3 6s).\n"
            f"- Instead of bidding, you can challenge the current bid by saying 'CHALLENGE'. You can only challenge the immediately preceding bid.\n"
            f"- If a bid is challenged, the actual count of the digit is revealed. If count >= bid quantity, the bidder wins. If count < bid quantity, the challenger wins.\n"
            f"- The loser of a challenge is eliminated.\n"
            f"- The maximum possible quantity for any digit is {self.total_digits_in_play}.\n"
            f"Output Format:\n"
            f"Respond with a valid JSON object containing 'reasoning' and 'action' keys.\n"
            f"Example 1 (Making a bid):\n"
            f"{{\n"
            f'  "reasoning": "I have two 6s in my hand. Player 3 previously bid 3 5s. Bidding 4 6s seems reasonable, increasing both quantity and digit based on my hand.",\n'
            f'  "action": "BID: 4 6s"\n'
            f"}}\n"
            f"Example 2 (Challenging):\n"
            f"{{\n"
            f'  "reasoning": "The current bid is 11 8s made by Player 2. With only {self.total_digits_in_play} total digits remaining, this seems extremely unlikely, even if I have one 8. I should challenge.",\n'
            f'  "action": "CHALLENGE"\n'
            f"}}\n"
            f"Ensure the 'action' value is *exactly* 'BID: [quantity] [digit]s' or 'CHALLENGE'."
            " IMPORTANT: Do not include any text or Markdown formatting outside the JSON. Only return the JSON object."
        )
        hand_str = "".join(map(str, player.hand))
        # Anonymize player list
        player_list_str = ", ".join([f"Player {p.player_id}" for p in self.active_players])

        # Generate persistent history string (limited to last 30 events)
        game_history_lines = ["Game History:"]
        history_to_show = self.event_history[-30:]
        for event in history_to_show:
            event_type = event.get('type')
            if event_type == 'BID':
                game_history_lines.append(f"- Player {event['player_id']} bid: {event['bid']}")
            elif event_type == 'CHALLENGE':
                game_history_lines.append(f"- Player {event['challenger_id']} challenged Player {event['challenged_id']}'s bid of {event['bid']}.")
            elif event_type == 'RESULT':
                game_history_lines.append(f"  - Outcome: Actual count was {event['actual_count']}. {event['outcome']}.")
            elif event_type == 'ELIMINATION':
                game_history_lines.append(f"  - Player {event['player_id']} was eliminated (Rank {event['rank']}).")
            elif event_type == 'NEXT_ROUND_INFO':
                game_history_lines.append(f"--- Next Round Started ({event['players_left']} players) ---")
                game_history_lines.append(f"  - Total Digits: {event['total_digits']}")
                game_history_lines.append(f"  - Player {event['starting_player_id']} starts bidding.")
                game_history_lines.append(f"-------------------------")

        # Add bids from the *current* round (if any) - distinct from event_history
        if self.bid_history:
            game_history_lines.append("Current Round Bids:")
            for pid, bid in self.bid_history:
                 game_history_lines.append(f"- Player {pid} bid: {bid}")
        elif not self.current_bid: # Only show if no bids made yet in this round
             game_history_lines.append("Current Round Bids: No bids yet.")

        history_str = "\n".join(game_history_lines)

        # Construct user message with anonymized info and new history
        user_message = (
            f"Game State:\n"
            f"- Your Hand: {hand_str}\n"
            f"- Your ID: Player {player.player_id}\n"
            f"- Active Players: {player_list_str}\n" # Anonymized list
            f"- Number of Digits per Player: {self.MAX_DIGITS_PER_HAND}\n"
            f"- Total Digits Remaining: {self.total_digits_in_play}\n"
            f"- Current Bid: {self.current_bid or 'None'}"
        )
        if self.current_bid and self.bid_history: # Check bid_history is not empty
             last_bidder_id = self.bid_history[-1][0]
             user_message += f" (made by Player {last_bidder_id})" # Anonymized ID

        user_message += (
            f"\n{history_str}\n\n"
            "What is your action? Provide your reasoning and action in the specified JSON format."
        )
        return developer_message, user_message, system_message

    def _parse_action_string(self, action_str: Optional[str]) -> Union[Bid, str, None]:
        if not action_str:
            self._log_round_event("Extracted action string was empty.")
            return None
        action_str = action_str.strip().upper()
        self._log_round_event(f"Attempting to parse extracted action string: '{action_str}'")
        if action_str == "CHALLENGE":
            if self.current_bid is None:
                self._log_round_event("Parse Error: CHALLENGE with no existing bid.")
                return None
            return "CHALLENGE"
        match = re.match(r"BID:\s*(\d+)\s+(\d)S?", action_str)
        if match:
            try:
                quantity = int(match.group(1))
                digit = int(match.group(2))
                if not (0 <= digit <= 9):
                    self._log_round_event(f"Parse Error: Invalid digit {digit}.")
                    return None
                if not (1 <= quantity <= self.total_digits_in_play):
                    self._log_round_event(f"Parse Error: Invalid quantity {quantity}. Must be between 1 and {self.total_digits_in_play}.")
                    return None
                bid = Bid(quantity=quantity, digit=digit)
                if not bid.is_higher_than(self.current_bid):
                    self._log_round_event(f"Parse Error: Bid {bid} is not higher than current bid {self.current_bid}.")
                    return None
                return bid
            except Exception as e:
                self._log_round_event(f"Parse Error: {e}")
                return None
        self._log_round_event("Parse Error: Action string did not match 'CHALLENGE' or 'BID: Q D'.")
        return None

    def _get_llm_action(self, player: Player) -> Union[Bid, str, None]:
        if not player.client:
            self._log_round_event(f"Error: Player {player.player_id} is 'llm' type but no client was found.")
            return None
        developer_msg, user_msg, system_msg = self._get_prompt_context(player)

        for attempt in range(self.MAX_ACTION_PARSE_ATTEMPTS + 1):
            self._log_round_event(f"Requesting LLM action from Player {player.player_id} ({player.get_display_name()}), Attempt {attempt + 1}/{self.MAX_ACTION_PARSE_ATTEMPTS + 1}")
            response_payload = player.client.call_llm(developer_message=developer_msg, user_message=user_msg, system_message=system_msg)
            try:
                request_data = {
                    "developer_message": developer_msg,
                    "user_message": user_msg,
                    "system_message": system_msg,
                    "max_tokens": player.client.max_tokens,
                    "temperature": player.client.temperature
                }
                if response_payload is not None and player.model_config:
                    log_llm_call(player.model_config["provider"], player.model_config["model"], request_data, response_payload)
            except NameError:
                logger.warning("log_llm_call function not found, skipping LLM call logging.")
            except Exception as log_err:
                logger.error(f"Error during LLM call logging: {log_err}")

            if response_payload is None:
                self._log_round_event(f"Player {player.player_id} LLM call failed after retries.")
                if attempt == self.MAX_ACTION_PARSE_ATTEMPTS:
                    return None
                continue

            response_text = parse_response_text(response_payload)
            if response_text is None:
                self._log_round_event(f"Player {player.player_id} response content was empty.")
                if attempt == self.MAX_ACTION_PARSE_ATTEMPTS:
                    return None
                continue

            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):].strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[len("```"):].strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            parsed_action = None
            try:
                llm_output = json.loads(cleaned)
                if (isinstance(llm_output, dict) and "action" in llm_output and "reasoning" in llm_output):
                    reasoning_str = llm_output["reasoning"]
                    action_str = llm_output["action"]
                    self._log_round_event(f"Player {player.player_id} Reasoning: {reasoning_str}")
                    parsed_action = self._parse_action_string(action_str)
                else:
                    self._log_round_event(f"Player {player.player_id} JSON missing 'action' or 'reasoning'.")
            except json.JSONDecodeError:
                self._log_round_event(f"Player {player.player_id} response was not valid JSON: '{cleaned}'")

            if parsed_action is not None:
                return parsed_action
            else:
                self._log_round_event(f"Player {player.player_id} provided invalid action or malformed JSON. Re-prompting...")
                user_msg += "\n\n**Please ensure your output is ONLY the valid JSON object with 'reasoning' and 'action' keys, and the 'action' value follows the required format ('BID: ...' or 'CHALLENGE'). Do not include explanations outside the JSON or markdown formatting like ```json.**"

        self._log_round_event(f"Player {player.player_id} failed after all parse attempts.")
        return None

    def _get_naive_action(self, player: Player) -> Union[Bid, str, None]:
        self._log_round_event(f"Calculating action for Player {player.player_id} (Naive 50/50)")
        if self.current_bid is None:
            naive_bid = Bid(quantity=1, digit=0)
            self._log_round_event(f"Player {player.player_id} making initial bid: {naive_bid}")
            return naive_bid
        else:
            if random.random() < 0.5:
                self._log_round_event(f"Player {player.player_id} chose to CHALLENGE.")
                return "CHALLENGE"
            else:
                current_qty = self.current_bid.quantity
                current_digit = self.current_bid.digit
                if current_digit < 9:
                    naive_bid = Bid(quantity=current_qty, digit=current_digit + 1)
                else:
                    naive_bid = Bid(quantity=current_qty + 1, digit=0)
                if naive_bid.quantity > self.total_digits_in_play:
                    self._log_round_event(f"Naive tried an invalid bid {naive_bid}, forcing CHALLENGE.")
                    return "CHALLENGE"
                else:
                    self._log_round_event(f"Naive chooses BID: {naive_bid}")
                    return naive_bid

    # New method for the "intelligent" strategy:
    def _get_intelligent_action(self, player: Player) -> Union[Bid, str, None]:
        hand_freq = Counter(player.hand)

        # Step 1 — If there is no current bid
        if self.current_bid is None:
            # (a) any digit with freq >= 3
            candidates = [(freq, d) for d, freq in hand_freq.items() if freq >= 3]
            if candidates:
                # pick the highest such bid, where 'higher quantity' > 'higher digit'
                best = max(candidates, key=lambda x: (x[0], x[1]))
                return Bid(quantity=best[0], digit=best[1])
            else:
                # (b) highest digit that appears at least once
                max_digit_with_freq = max((d for d in hand_freq if hand_freq[d] >= 1), default=0)
                return Bid(quantity=2, digit=max_digit_with_freq)

        # Step 2 — If there is a current bid
        q, d = self.current_bid.quantity, self.current_bid.digit
        c = player.hand.count(d)

        # Challenge Condition: if q >= 3 and q > c + 1
        if q >= 3 and q > c + 1:
            return "CHALLENGE"

        # Step 3 — Otherwise, raise the bid intelligently
        # find the digit d_max with the highest frequency, tie-break on digit value
        d_max, f_max = max(hand_freq.items(), key=lambda x: (x[1], x[0]))
        # if same q and d_max > d -> Bid(q, d_max)
        if q == self.current_bid.quantity and d_max > d:
            next_bid = Bid(quantity=q, digit=d_max)
        else:
            next_bid = Bid(quantity=q + 1, digit=d_max)

        if not next_bid.is_higher_than(self.current_bid):
            return "CHALLENGE"  # fallback
        if next_bid.quantity > self.total_digits_in_play:
            return "CHALLENGE"  # fallback if invalid
        return next_bid

    def _count_digit_occurrences(self, digit: int) -> int:
        count = 0
        for p in self.active_players:
            count += p.hand.count(digit)
        return count

    def _resolve_challenge(self) -> Tuple[int, List[int]]:
        # This method is DEPRECATED. Use _resolve_elimination_challenge instead.
        raise NotImplementedError("_resolve_challenge is deprecated.")
        if not self.current_bid or not self.bid_history:
            raise RuntimeError("Cannot resolve challenge without a current bid.")
        challenger_id = self.current_player_index
        challenged_bidder_id, challenged_bid = self.bid_history[-1]
        challenger = self.active_players[challenger_id]
        challenged_bidder_player = next((p for p in self.active_players if p.player_id == challenged_bidder_id), None)
        if not challenged_bidder_player:
            raise RuntimeError(f"Could not find challenged bidder ID {challenged_bidder_id}")
        self._log_round_event(f"Player {challenger_id} ({challenger.get_display_name()}) challenges Player {challenged_bidder_id} ({challenged_bidder_player.get_display_name()})'s bid of {challenged_bid}.")

        hands_reveal_log = "Revealed Hands: " + " | ".join([f"P{p.player_id}({p.get_display_name()}):{''.join(map(str,p.hand))}" for p in self.players])
        self._log_round_event(hands_reveal_log)

        actual_count = self._count_digit_occurrences(challenged_bid.digit)
        self._log_round_event(f"Actual count of {challenged_bid.digit}s: {actual_count}")
        winner_id = None
        all_player_ids = [p.player_id for p in self.players]
        if actual_count >= challenged_bid.quantity:
            winner_id = challenged_bidder_id
            self._log_round_event(f"Challenge failed: count({actual_count}) >= {challenged_bid.quantity}. Bidder wins.")
        else:
            winner_id = challenger_id
            self._log_round_event(f"Challenge successful: count({actual_count}) < {challenged_bid.quantity}. Challenger wins.")

        loser_ids = [pid for pid in all_player_ids if pid != winner_id]
        loser_display = []
        for pid in loser_ids:
            pl = next((p for p in self.players if p.player_id == pid), None)
            loser_display.append(f"{pid} ({pl.get_display_name() if pl else 'Unknown'})")
        self._log_round_event(f"Losers: {', '.join(loser_display)}")
        self.game_active = False
        return winner_id, loser_ids

    def _resolve_elimination_challenge(self):
        """Resolves a challenge, determines the loser, eliminates them, and resets for the next round."""
        if not self.current_bid or not self.bid_history:
            self._log_round_event("Error: Cannot resolve challenge without a current bid.")
            # As a fallback, maybe eliminate the challenger? Or handle error state.
            # For now, let's eliminate the challenger if this state is reached.
            loser_player = self.active_players[self.current_player_index]
            winner_player = None # No clear winner
            self._log_round_event(f"Error state: Eliminating challenger Player {loser_player.player_id} due to missing bid.")
            # Need to log elimination event here for forfeit case
            loser_original_order = loser_player.original_order
            assigned_rank = self.next_rank_to_assign
            self.ranks[loser_original_order] = assigned_rank
            self.next_rank_to_assign -= 1
            self.event_history.append({
                'type': 'ELIMINATION',
                'player_id': loser_player.player_id,
                'rank': assigned_rank
            })
            self._log_round_event(f"Player {loser_player.player_id}(Orig:{loser_original_order} - {loser_player.get_display_name()}) is ELIMINATED due to error/invalid challenge. Assigned Rank: {assigned_rank}")

        else:
            challenger_player = self.active_players[self.current_player_index]
            challenged_bidder_id, challenged_bid = self.bid_history[-1]

            # Find the challenged bidder within the *active* players list
            challenged_bidder_player = next((p for p in self.active_players if p.player_id == challenged_bidder_id), None)

            if not challenged_bidder_player:
                # This should ideally not happen if history/state is managed correctly
                self._log_round_event(f"Error: Challenged bidder (Player ID {challenged_bidder_id}) not found among active players. Eliminating challenger.")
                loser_player = challenger_player
                winner_player = None # No clear winner
                # Log challenge attempt
                self.event_history.append({
                    'type': 'CHALLENGE',
                    'challenger_id': challenger_player.player_id,
                    'challenged_id': challenged_bidder_id, # ID is known even if player isn't active
                    'bid': str(challenged_bid)
                })
                # Log elimination event for challenger due to error
                loser_original_order = loser_player.original_order
                assigned_rank = self.next_rank_to_assign
                self.ranks[loser_original_order] = assigned_rank
                self.next_rank_to_assign -= 1
                self.event_history.append({
                    'type': 'ELIMINATION',
                    'player_id': loser_player.player_id,
                    'rank': assigned_rank
                })
                self._log_round_event(f"Player {loser_player.player_id}(Orig:{loser_original_order} - {loser_player.get_display_name()}) is ELIMINATED due to bidder not found error. Assigned Rank: {assigned_rank}")
            else:
                self._log_round_event(f"Player {challenger_player.player_id}(Orig:{challenger_player.original_order} - {challenger_player.get_display_name()}) challenges Player {challenged_bidder_player.player_id}(Orig:{challenged_bidder_player.original_order} - {challenged_bidder_player.get_display_name()})'s bid of {challenged_bid}.")
                # Log challenge event
                self.event_history.append({
                    'type': 'CHALLENGE',
                    'challenger_id': challenger_player.player_id,
                    'challenged_id': challenged_bidder_player.player_id,
                    'bid': str(challenged_bid)
                })

                # Reveal hands of active players
                hands_reveal_log = "Revealed Hands: " + " | ".join([f"P{p.player_id}(Orig:{p.original_order}-{p.get_display_name()}):{''.join(map(str,p.hand))}" for p in self.active_players])
                self._log_round_event(hands_reveal_log)

                actual_count = self._count_digit_occurrences(challenged_bid.digit)
                self._log_round_event(f"Actual count of {challenged_bid.digit}s across {len(self.active_players)} active players: {actual_count}")

                # Determine the loser
                if actual_count >= challenged_bid.quantity:
                    # Bid stands, challenger loses
                    loser_player = challenger_player
                    winner_player = challenged_bidder_player
                    challenge_outcome_str = "Bidder Wins"
                    self._log_round_event(f"Challenge failed: count({actual_count}) >= {challenged_bid.quantity}. Bidder (Player {winner_player.player_id}) wins the challenge.")
                else:
                    # Challenge succeeds, bidder loses
                    loser_player = challenged_bidder_player
                    winner_player = challenger_player
                    challenge_outcome_str = "Challenger Wins"
                    self._log_round_event(f"Challenge successful: count({actual_count}) < {challenged_bid.quantity}. Challenger (Player {winner_player.player_id}) wins the challenge.")

                # Log challenge result
                self.event_history.append({
                    'type': 'RESULT',
                    'bid': str(challenged_bid),
                    'actual_count': actual_count,
                    'outcome': challenge_outcome_str,
                    'winner_id': winner_player.player_id, # ID of player who won the challenge resolution
                    'loser_id': loser_player.player_id    # ID of player who lost the challenge resolution
                })

                # Eliminate the loser (log event before removal)
                loser_original_order = loser_player.original_order
                assigned_rank = self.next_rank_to_assign
                self.ranks[loser_original_order] = assigned_rank
                self.next_rank_to_assign -= 1
                self.event_history.append({
                    'type': 'ELIMINATION',
                    'player_id': loser_player.player_id,
                    'rank': assigned_rank
                })
                self._log_round_event(f"Player {loser_player.player_id}(Orig:{loser_original_order} - {loser_player.get_display_name()}) is ELIMINATED. Assigned Rank: {assigned_rank}")

        # --- Common Logic after loser determined (either by error or challenge result) ---

        # Update total digits in play (only if a player was actually eliminated)
        if loser_player:
             self.total_digits_in_play -= self.MAX_DIGITS_PER_HAND
             self._log_round_event(f"Total digits remaining in play: {self.total_digits_in_play}")

        # Remove loser from active players list
        loser_index = -1
        if loser_player: # Check if a loser was identified
            for idx, p in enumerate(self.active_players):
                if p.player_id == loser_player.player_id:
                    loser_index = idx
                    break
            if loser_index != -1:
                self.active_players.pop(loser_index)
                self._log_round_event(f"Removed player {loser_player.player_id} from active list. {len(self.active_players)} players remain.")
            else:
                self._log_round_event(f"Error: Could not find loser player {loser_player.player_id} in active list for removal.") # Should not happen

        # Reset bid state for the next round of bidding among remaining players
        self.current_bid = None
        self.bid_history = [] # Clear the per-round bid history
        self._log_round_event("Current bid and per-round history cleared.")

        # Log NEXT_ROUND_INFO and set the next player index
        if len(self.active_players) > 0:
            next_player_id_to_start = -1
            if winner_player:
                try:
                    # Find the index of the winner in the *updated* active_players list
                    winner_current_index = self.active_players.index(winner_player)
                    self.current_player_index = winner_current_index
                    next_player_id_to_start = winner_player.player_id
                    self._log_round_event(f"Player {winner_player.player_id}(Orig:{winner_player.original_order} - {winner_player.get_display_name()}) starts the next bidding round.")
                except ValueError:
                    self._log_round_event("Error: Winner not found in updated active list. Resetting index to 0.")
                    self.current_player_index = 0
                    next_player_id_to_start = self.active_players[0].player_id
            else:
                 # If no clear winner from challenge (error state), default to player 0 of remaining
                 self._log_round_event("No clear winner from challenge or error state. Player at index 0 starts next.")
                 self.current_player_index = 0
                 next_player_id_to_start = self.active_players[0].player_id

            # Log the start of the next round info
            self.event_history.append({
                'type': 'NEXT_ROUND_INFO',
                'players_left': len(self.active_players),
                'starting_player_id': next_player_id_to_start,
                'total_digits': self.total_digits_in_play
            })
        else:
             # Game might be over if only 1 player left after elimination
             self._log_round_event("Challenge resolved, less than 2 players remaining.")
             self.current_player_index = 0

    def play_elimination_game(self) -> Dict[int, int]:
        """Plays a full game of Liar's Poker with progressive elimination until one winner remains."""
        self._initial_setup() # Sets up self.players, deals hands, sets initial player index
        # Capture actual player configurations (including random sub-strategies chosen)
        self.initial_players_list = self.players.copy()

        # Initialize game state for elimination
        self.active_players = self.players.copy() # Start with all configured players
        self.ranks = {} # Stores {original_order: rank}
        self.next_rank_to_assign = len(self.active_players)
        self.total_digits_in_play = len(self.active_players) * self.MAX_DIGITS_PER_HAND
        self.current_bid = None
        self.bid_history = [] # Per-round bid history
        self.event_history = [] # Persistent game event history
        self.round_log = self.round_log[-3:] # Keep only recent setup logs
        self.game_active = True # Game is active

        self._log_round_event(f"--- Starting Elimination Game with {len(self.active_players)} players. Total digits: {self.total_digits_in_play} ---")

        # Log initial round info
        initial_player_id = self.active_players[0].player_id if self.active_players else -1
        self.event_history.append({
            'type': 'NEXT_ROUND_INFO',
            'players_left': len(self.active_players),
            'starting_player_id': initial_player_id,
            'total_digits': self.total_digits_in_play
        })

        while len(self.active_players) > 1:
            if not self.game_active:
                self._log_round_event("Game loop detected inactive state unexpectedly. Breaking loop.")
                break

            # Ensure player index is valid for the current number of active players
            if self.current_player_index >= len(self.active_players):
                self._log_round_event(f"Warning: Player index {self.current_player_index} out of bounds for {len(self.active_players)} active players. Resetting to 0.")
                self.current_player_index = 0

            current_player = self.active_players[self.current_player_index]
            self._log_round_event(f"\n--- Turn: Player {current_player.player_id}(Orig:{current_player.original_order} - {current_player.get_display_name()}) [{len(self.active_players)} players left] ---")

            action: Union[Bid, str, None] = None
            player_strategy = current_player.strategy_type
            effective_strategy = current_player.effective_strategy if player_strategy == 'random' else player_strategy

            try:
                if effective_strategy == 'llm':
                    action = self._get_llm_action(current_player)
                elif effective_strategy == 'naive_5050':
                    action = self._get_naive_action(current_player)
                elif effective_strategy == 'intelligent':
                    action = self._get_intelligent_action(current_player)
                else:
                    self._log_round_event(f"Error: Unknown effective strategy '{effective_strategy}' for Player {current_player.player_id}.")
                    action = None # Treat as forfeit
            except Exception as e:
                self._log_round_event(f"Error during action selection for Player {current_player.player_id}: {e}")
                action = None # Treat as forfeit

            # Handle player action
            if action is None:
                self._log_round_event(f"Player {current_player.player_id}(Orig:{current_player.original_order} - {current_player.get_display_name()}) forfeits or encountered an error.")
                assigned_rank = self.next_rank_to_assign
                self.ranks[current_player.original_order] = assigned_rank
                self.next_rank_to_assign -= 1
                # Log Elimination Event
                self.event_history.append({
                    'type': 'ELIMINATION',
                    'player_id': current_player.player_id,
                    'rank': assigned_rank
                })
                self._log_round_event(f"Player {current_player.player_id}(Orig:{current_player.original_order}) is ELIMINATED due to forfeit/error. Assigned Rank: {assigned_rank}")

                self.total_digits_in_play -= self.MAX_DIGITS_PER_HAND
                eliminated_player_index = self.current_player_index
                eliminated_player = self.active_players.pop(eliminated_player_index)
                self._log_round_event(f"Removed player {eliminated_player.player_id}. {len(self.active_players)} players remain. Total digits: {self.total_digits_in_play}")

                # Reset bid state as the round ends here
                self.current_bid = None
                self.bid_history = [] # Clear per-round history
                self._log_round_event("Bid state reset due to elimination.")

                # The next player in the list (at the same index) starts the new round
                self.current_player_index %= len(self.active_players) if len(self.active_players) > 0 else 0
                if len(self.active_players) > 0:
                    next_p = self.active_players[self.current_player_index]
                    self._log_round_event(f"Player {next_p.player_id}(Orig:{next_p.original_order}) starts the next bidding round.")
                    # Log Next Round Info Event
                    self.event_history.append({
                        'type': 'NEXT_ROUND_INFO',
                        'players_left': len(self.active_players),
                        'starting_player_id': next_p.player_id,
                        'total_digits': self.total_digits_in_play
                    })

            elif action == "CHALLENGE":
                if self.current_bid is None:
                    self._log_round_event("Error: Player challenged with no current bid. Treating as forfeit.")
                    # Repeat forfeit logic (including ELIMINATION and NEXT_ROUND_INFO logging)
                    assigned_rank = self.next_rank_to_assign
                    self.ranks[current_player.original_order] = assigned_rank
                    self.next_rank_to_assign -= 1
                    self.event_history.append({
                        'type': 'ELIMINATION',
                        'player_id': current_player.player_id,
                        'rank': assigned_rank
                    })
                    self._log_round_event(f"Player {current_player.player_id}(Orig:{current_player.original_order}) is ELIMINATED due to invalid challenge. Assigned Rank: {assigned_rank}")
                    self.total_digits_in_play -= self.MAX_DIGITS_PER_HAND
                    eliminated_player_index = self.current_player_index
                    eliminated_player = self.active_players.pop(eliminated_player_index)
                    self._log_round_event(f"Removed player {eliminated_player.player_id}. {len(self.active_players)} players remain. Total digits: {self.total_digits_in_play}")
                    self.current_bid = None
                    self.bid_history = [] # Clear per-round history
                    self.current_player_index %= len(self.active_players) if len(self.active_players) > 0 else 0
                    if len(self.active_players) > 0:
                        next_p = self.active_players[self.current_player_index]
                        self._log_round_event(f"Player {next_p.player_id}(Orig:{next_p.original_order}) starts the next bidding round.")
                        self.event_history.append({
                            'type': 'NEXT_ROUND_INFO',
                            'players_left': len(self.active_players),
                            'starting_player_id': next_p.player_id,
                            'total_digits': self.total_digits_in_play
                        })
                else:
                    self._resolve_elimination_challenge() # This handles elimination, rank, reset, event logging, and setting next player
                    # The index is set within _resolve_elimination_challenge

            elif isinstance(action, Bid):
                # Validate bid (already partially done in parse, but double check here)
                if action.is_higher_than(self.current_bid) and action.quantity <= self.total_digits_in_play:
                    self.current_bid = action
                    # Log BID event to persistent history
                    self.event_history.append({
                        'type': 'BID',
                        'player_id': current_player.player_id,
                        'bid': str(action)
                    })
                    # Also append to the current round's self.bid_history
                    self.bid_history.append((current_player.player_id, action))
                    self._log_round_event(f"Player {current_player.player_id} bids: {action}")
                    # Advance to the next player
                    self.current_player_index = (self.current_player_index + 1) % len(self.active_players)
                else:
                    self._log_round_event(f"Error: Player {current_player.player_id} proposed invalid bid {action} (Current: {self.current_bid}, Max Qty: {self.total_digits_in_play}). Treating as forfeit.")
                    # Repeat forfeit logic (including ELIMINATION and NEXT_ROUND_INFO logging)
                    assigned_rank = self.next_rank_to_assign
                    self.ranks[current_player.original_order] = assigned_rank
                    self.next_rank_to_assign -= 1
                    self.event_history.append({
                        'type': 'ELIMINATION',
                        'player_id': current_player.player_id,
                        'rank': assigned_rank
                    })
                    self._log_round_event(f"Player {current_player.player_id}(Orig:{current_player.original_order}) is ELIMINATED due to invalid bid. Assigned Rank: {assigned_rank}")
                    self.total_digits_in_play -= self.MAX_DIGITS_PER_HAND
                    eliminated_player_index = self.current_player_index
                    eliminated_player = self.active_players.pop(eliminated_player_index)
                    self._log_round_event(f"Removed player {eliminated_player.player_id}. {len(self.active_players)} players remain. Total digits: {self.total_digits_in_play}")
                    self.current_bid = None
                    self.bid_history = [] # Clear per-round history
                    self.current_player_index %= len(self.active_players) if len(self.active_players) > 0 else 0
                    if len(self.active_players) > 0:
                        next_p = self.active_players[self.current_player_index]
                        self._log_round_event(f"Player {next_p.player_id}(Orig:{next_p.original_order}) starts the next bidding round.")
                        self.event_history.append({
                            'type': 'NEXT_ROUND_INFO',
                            'players_left': len(self.active_players),
                            'starting_player_id': next_p.player_id,
                            'total_digits': self.total_digits_in_play
                        })

            # Add delay based on player type
            if effective_strategy == 'llm':
                time.sleep(1.0)
            else:
                time.sleep(0.1)

        # Game loop finished (only one player left)
        if len(self.active_players) == 1:
            winner = self.active_players[0]
            self.ranks[winner.original_order] = 1 # Assign 1st place
            self._log_round_event(f"\n--- Elimination Game Over ---")
            self._log_round_event(f"Winner: Player {winner.player_id}(Orig:{winner.original_order} - {winner.get_display_name()}) is Rank 1!")
            self.game_active = False
        elif self.game_active:
            # This case might occur if the loop broke unexpectedly
            self._log_round_event("Game ended with multiple players remaining due to unexpected state.")
            # Assign remaining players ranks based on next_rank_to_assign
            remaining_players_sorted = sorted(self.active_players, key=lambda p: p.original_order)
            for player in remaining_players_sorted:
                if player.original_order not in self.ranks:
                    self.ranks[player.original_order] = self.next_rank_to_assign
                    self._log_round_event(f"Assigning remaining Player {player.player_id}(Orig:{player.original_order}) Rank: {self.next_rank_to_assign}")
                    self.next_rank_to_assign -= 1
            self.game_active = False
        else:
            # Game was already inactive
             self._log_round_event("Game concluded.")

        # Log final rankings
        logger.info("\n--- Final Rankings ---")
        sorted_ranks_list = sorted(self.ranks.items(), key=lambda item: item[1])
        initial_players_dict = {p.original_order: p for p in self.initial_players_list}
        for original_order, rank in sorted_ranks_list:
            player = initial_players_dict.get(original_order)
            if player:
                player_name = player.get_display_name()
                logger.info(f"Rank {rank}: Player {original_order} ({player_name})")
            else:
                logger.warning(f"Rank {rank}: Could not find player details for original_order {original_order}.")

        # Save the consolidated game log
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(GAME_LOGS_DIR, f"liars_poker_elimination_game_{timestamp}.log")
        try:
            with open(log_filename, "w", encoding='utf-8') as f:
                # Apply file lock
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    f.write("\n".join(self.round_log))
                    fcntl.flock(f, fcntl.LOCK_UN)
                except BlockingIOError:
                    logger.warning(f"Could not acquire lock for {log_filename}, skipping write.")
                except Exception as lock_e:
                    logger.error(f"Error locking/unlocking {log_filename}: {lock_e}")
                    # Still attempt to write if locking fails but isn't BlockingIOError
                    try:
                        f.seek(0) # Ensure we are at the beginning if error occurred after partial write attempt
                        f.truncate(0)
                        f.write("\n".join(self.round_log))
                    except Exception as write_e:
                         logger.error(f"Failed to write to {log_filename} after lock error: {write_e}")

            logger.info(f"Full game log saved to {log_filename}")
        except Exception as e:
            logger.error(f"Failed to save game log: {e}")

        # --- Build and Persist Summary Data ---
        summary_timestamp = time.strftime("%Y%m%d-%H%M%S") # Use a consistent timestamp or a new one
        summary_data = {
            "timestamp": summary_timestamp,
            "rank1": None,
            "rank2": None,
            "rank3": None,
            "rank4": None,
            "rank5": None,
            "rank6": None,
        }

        # Fill in each rank from the sorted final ranks
        for pos in range(1, 7):
            for original_order, rank_val in self.ranks.items():
                if rank_val == pos:
                    player_obj = next((p for p in self.initial_players_list if p.original_order == original_order), None)
                    if player_obj:
                        summary_data[f"rank{pos}"] = player_obj.get_display_name()
                    else:
                         logger.warning(f"Could not find player details for rank {pos} (original_order {original_order}) in summary.")
                    break # Found the player for this rank position

        # --- Open and Write to hands_log_multi_liars_poker.json ---
        summary_log_filename = "hands_log_multi_liars_poker.json"
        summary_path = os.path.join(LOGS_DIR, summary_log_filename)
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX) # Acquire exclusive lock
                f.seek(0) # Move to the beginning to read content
                content = f.read().strip()
                log_entries = []
                if content:
                    try:
                        # Try parsing existing content as JSON list
                        data = json.loads(content)
                        if isinstance(data, list):
                            log_entries = data
                        else:
                             # Handle case where file exists but isn't a valid JSON list (e.g., contains single object)
                            logger.warning(f"Existing content in {summary_path} is not a list. Resetting log.")
                            log_entries = [] # Or potentially [data] if you want to wrap it
                    except json.JSONDecodeError:
                        # Handle case where file exists but isn't valid JSON
                        logger.warning(f"Could not decode JSON from {summary_path}. Resetting log.")
                        log_entries = []
                # else: log_entries is already initialized as []

                log_entries.append(summary_data) # Add the new summary

                # Prepare to overwrite the file
                f.seek(0)
                f.truncate(0)
                json.dump(log_entries, f, indent=2) # Write the updated list back
                fcntl.flock(f, fcntl.LOCK_UN) # Release lock
            logger.info(f"Summary log updated: {summary_path}")
        except BlockingIOError:
             logger.error(f"Could not acquire lock for {summary_path}. Summary not saved.")
        except Exception as e:
            logger.error(f"Failed to write summary to {summary_path}: {e}")
            # Attempt to release lock if held, though it might fail if the error was before lock release
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except Exception:
                 pass # Ignore errors during unlock attempt after another error

        # Update misleading print statement
        print(f"\nElimination game finished. Logs saved:\n- Full game log: {log_filename}\n- Summary JSON: {summary_path}")

        return self.ranks

    def play_round(self) -> Tuple[Optional[int], List[int], List[str]]:
        # Dummy implementation or raise error if called
        raise NotImplementedError("play_round is deprecated. Use play_elimination_game.")
        # This method is now DEPRECATED for elimination mode.
        # Consider removing or keeping for potential single-round mode later.

def get_player_configurations() -> List[Dict[str, Any]]:
    configs = []
    num_players = 6 # Hardcoded for elimination mode
    print(f"Configuring {num_players} players for Progressive Elimination game.")

    print("\nEnter configurations for each player.")
    print("Supported LLM providers: openai, anthropic, openrouter, etc.")
    print("Internal Strategies: Naive 50/50 (N), Random (R), Intelligent (I)")
    print("Common Choices:")
    print("  N: Naive 50/50")
    print("  R: Random")
    print("  I: Intelligent")
    for i, cfg in enumerate(LiarsPokerGame.COMMON_CONFIGS, 1):
        print(f"  {i}: {cfg['provider']}/{cfg['model']}")

    for i in range(num_players):
        while True:
            choice = input(f"\nSelect config for Player {i+1} (N, R, I, or a number from 1 to {len(LiarsPokerGame.COMMON_CONFIGS)}) or C: ").strip().upper()
            selected_config = None
            if choice == 'N':
                selected_config = {"strategy_type": "naive_5050"}
            elif choice == 'R':
                selected_config = {"strategy_type": "random"}
            elif choice == 'I':
                selected_config = {"strategy_type": "intelligent"}
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(LiarsPokerGame.COMMON_CONFIGS):
                    llm_cfg = LiarsPokerGame.COMMON_CONFIGS[idx]
                    selected_config = {"strategy_type": "llm","provider": llm_cfg["provider"],"model": llm_cfg["model"]}
                else:
                    print("Invalid number.")
                    continue
            elif choice == 'C':
                provider = input("Enter provider: ").strip()
                model = input("Enter model name: ").strip()
                if not provider or not model:
                    print("Provider/Model cannot be empty.")
                    continue
                selected_config = {"strategy_type": "llm","provider": provider,"model": model}
            else:
                print("Invalid choice.")
                continue
            if selected_config:
                configs.append(selected_config)
                break

    return configs

if __name__ == "__main__":
    print("--- Liar's Poker LLM Benchmark ---")
    try:
        is_dummy = False
        try:
            _ = LLMClient(provider="dummy", model="dummy", max_tokens=1, temperature=1, max_retries=0, timeout=1)
            if 'Dummy LLMClient' in str(LLMClient):
                is_dummy = True
        except Exception:
            if 'Dummy LLMClient' in str(LLMClient):
                is_dummy = True

        if is_dummy:
            print("\nWARNING: Running with DUMMY LLM Client. No real API calls.\n")
            player_configs = [
                {"strategy_type": "naive_5050"},
                {"strategy_type": "llm", "provider": "dummy_openai", "model": "dummy_gpt4"}
            ]
        else:
            player_configs = get_player_configurations()

        print("\nInitializing game...")
        game = LiarsPokerGame(player_configs=player_configs)

        print("\n--- Starting Progressive Elimination Game ---")
        final_ranks = game.play_elimination_game() # Returns dict {original_order: rank}

        print("\n--- Final Rankings --- ")
        # Reconstruct player names from initial configs stored potentially in game.players based on original_order
        # Need a reliable way to map original_order back to display name.
        # Use game.initial_players_list which was stored in __init__
        initial_players_dict = {p.original_order: p for p in game.initial_players_list}

        sorted_ranks_list = sorted(final_ranks.items(), key=lambda item: item[1])
        for original_order, rank in sorted_ranks_list:
            player = initial_players_dict.get(original_order)
            if player:
                player_name = player.get_display_name()
                print(f"Rank {rank}: Player {original_order} ({player_name})")
            else:
                print(f"Rank {rank}: Original Player {original_order} (Details lookup failed)")

        summary_log_filename = "hands_log_multi_liars_poker.json"
        print(f"\nElimination game finished. Full game log in {GAME_LOGS_DIR}, summary in {os.path.join(LOGS_DIR, summary_log_filename)}")

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Game setup or execution failed: {e}", exc_info=True)
        print(f"\nError: {e}")
    except ImportError:
        logger.error("Failed to import llm_client. Please ensure it's available.")
        print("\nError: Could not import llm_client.py.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")