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

# Assuming llm_client.py is in the same parent directory or accessible via PYTHONPATH
# Ensure llm_client.py exists and is correctly implemented
try:
    from llm_client import LLMClient, parse_response_text, log_llm_call, LOGS_DIR
except ImportError:
    # Provide a dummy implementation if llm_client is missing, for basic structure testing
    print("Warning: llm_client.py not found. Using dummy implementation.")
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)
    class LLMClient:
        def __init__(self, provider: str, model: str, max_tokens: int, temperature: float, max_retries: int, timeout: int):
            self.provider = provider
            self.model = model
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.max_retries = max_retries
            self.timeout = timeout
            print(f"Dummy LLMClient initialized for {provider}/{model}")
        def call_llm(self, developer_message: str, user_message: str, system_message: str) -> Optional[Dict[str, Any]]:
            print(f"--- Dummy LLM Call ({self.model}) ---")
            print(f"System: {system_message}")
            print(f"Developer: {developer_message}")
            print(f"User: {user_message}")
            dummy_reasoning = "This is a dummy reasoning based on the prompt."
            if "Current Bid: None" in user_message:
                dummy_action = "BID: 2 5s"
            else:
                bid_match = re.search(r"Current Bid: Bid\(quantity=(\d+), digit=(\d+)\)", user_message)
                if bid_match and random.random() > 0.3:
                    qty = int(bid_match.group(1))
                    digit = int(bid_match.group(2))
                    if digit < 9:
                        dummy_action = f"BID: {qty} {digit+1}s"
                    else:
                        dummy_action = f"BID: {qty+1} 0s"
                else:
                    dummy_action = "CHALLENGE"
            response_content = json.dumps({"reasoning": dummy_reasoning, "action": dummy_action})
            return {"model": self.model,"usage": {"prompt_tokens": 100, "completion_tokens": 50},"choices": [{"message": {"content": response_content}}]}

    def parse_response_text(response_json: Optional[Dict[str, Any]]) -> Optional[str]:
        if response_json and "choices" in response_json and response_json["choices"]:
            message = response_json["choices"][0].get("message", {})
            content = message.get("content")
            if content:
                return str(content).strip()
        return None

    def log_llm_call(provider: str, model: str, request_data: Dict[str, Any], response_data: Dict[str, Any]) -> str:
        print(f"Dummy LLM Call Logged for provider={provider}, model={model}")
        return "dummy_log_file.json"

LOGS_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiarsPokerGame")
GAME_LOGS_DIR = os.path.join(LOGS_DIR, "game_logs")
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
        {"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-2024-11-20"},
        {"strategy_type": "llm", "provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
        {"strategy_type": "llm", "provider": "openai", "model": "o3-mini-2025-01-31"},

        # === Anthropic ===
        {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
        {"strategy_type": "llm", "provider": "anthropic", "model": "claude-3-7-sonnet-20250219"},

        # === DeepSeek ===
        {"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-chat-v3-0324:floor"},
        #SO SLOW{"strategy_type": "llm", "provider": "openrouter", "model": "deepseek/deepseek-r1:floor"},
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
        {"strategy_type": "llm", "provider": "openrouter", "model": "openrouter/optimus-alpha"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "openrouter/quasar-alpha"},
        {"strategy_type": "llm", "provider": "openrouter", "model": "qwen/qwq-32b:nitro"},
    ]

    def __init__(self, player_configs: List[Dict[str, Any]]):
        if not (2 <= len(player_configs) <= 6):
            raise ValueError("Number of players must be between 2 and 6.")
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
                if ('intelligent', None) not in used_strategies:
                    possible_choices.append({"strategy_type": "intelligent"})
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
            f"- Each of the {len(self.players)} players has {self.MAX_DIGITS_PER_HAND} secret digits.\n"
            f"- Players take turns bidding on the total count of a specific digit (0-9) across ALL players' hands.\n"
            f"- A bid consists of a quantity and a digit (e.g., '3 5s' means at least three 5s exist in total).\n"
            f"- Your bid must be strictly higher than the current bid '{self.current_bid or 'None'}'.\n"
            f"  - 'Higher' means: higher quantity (e.g., 3 9s -> 4 0s) OR same quantity but higher digit (e.g., 3 5s -> 3 6s).\n"
            f"- Instead of bidding, you can challenge the current bid by saying 'CHALLENGE'. You can only challenge the immediately preceding bid.\n"
            f"- If a bid is challenged, the actual count of the digit is revealed. If count >= bid quantity, the bidder wins. If count < bid quantity, the challenger wins.\n"
            f"- The maximum possible quantity for any digit is {self.total_digits_in_play}.\n"
            f"Output Format:\n"
            f"Respond with a valid JSON object containing 'reasoning' and 'action' keys.\n"
            f"Example 1 (Making a bid):\n"
            f"{{\n"
            f'  "reasoning": "I have two 6s in my hand. The current bid is only 3 5s. Bidding 4 6s seems reasonable, increasing both quantity and digit based on my hand.",\n'
            f'  "action": "BID: 4 6s"\n'
            f"}}\n"
            f"Example 2 (Challenging):\n"
            f"{{\n"
            f'  "reasoning": "The current bid is 11 8s. With only {self.total_digits_in_play} total digits in play, this seems extremely unlikely, even if I have one 8. I should challenge.",\n'
            f'  "action": "CHALLENGE"\n'
            f"}}\n"
            f"Ensure the 'action' value is *exactly* 'BID: [quantity] [digit]s' or 'CHALLENGE'."
            " IMPORTANT: Do not include any text or Markdown formatting outside the JSON. Only return the JSON object."
        )
        hand_str = "".join(map(str, player.hand))
        player_list_str = ", ".join([f"Player {p.player_id} ({p.get_display_name()})" for p in self.players])
        history_str = ("\n".join([f"  - Player {pid} bid: {bid}" for pid, bid in self.bid_history]) if self.bid_history else "  - No bids yet.")
        user_message = (
            f"Game State:\n"
            f"- Your Hand: {hand_str}\n"
            f"- Players: {player_list_str}\n"
            f"- Your Turn: Player {player.player_id} ({player.get_display_name()})\n"
            f"- Current Bid: {self.current_bid or 'None'}"
        )
        if self.current_bid:
            last_bidder_id = self.bid_history[-1][0]
            last_bidder = next(p for p in self.players if p.player_id == last_bidder_id)
            user_message += f" (made by Player {last_bidder_id} - {last_bidder.get_display_name()})"
        user_message += (
            f"\n- Bid History:\n{history_str}\n\n"
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
                    self._log_round_event(f"Parse Error: Invalid quantity {quantity} > {self.total_digits_in_play}.")
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
        for p in self.players:
            count += p.hand.count(digit)
        return count

    def _resolve_challenge(self) -> Tuple[int, List[int]]:
        if not self.current_bid or not self.bid_history:
            raise RuntimeError("Cannot resolve challenge without a current bid.")
        challenger_id = self.current_player_index
        challenged_bidder_id, challenged_bid = self.bid_history[-1]
        challenger = self.players[challenger_id]
        challenged_bidder_player = next((p for p in self.players if p.player_id == challenged_bidder_id), None)
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

    def play_round(self) -> Tuple[Optional[int], List[int], List[str]]:
        self._setup_round()
        winner_round_id: Optional[int] = None
        loser_round_ids: List[int] = []

        while self.game_active:
            current_player = self.players[self.current_player_index]
            self._log_round_event(f"\n--- Turn: Player {current_player.player_id} ({current_player.get_display_name()}) ---")
            action: Union[Bid, str, None] = None

            if current_player.strategy_type == 'random':
                if current_player.effective_strategy == 'llm':
                    action = self._get_llm_action(current_player)
                elif current_player.effective_strategy == 'naive_5050':
                    action = self._get_naive_action(current_player)
                elif current_player.effective_strategy == 'intelligent':
                    action = self._get_intelligent_action(current_player)
                else:
                    self._log_round_event(f"Error: Random player has no effective sub-strategy.")
                    action = None
            elif current_player.strategy_type == 'llm':
                action = self._get_llm_action(current_player)
            elif current_player.strategy_type == 'naive_5050':
                action = self._get_naive_action(current_player)
            elif current_player.strategy_type == 'intelligent':
                action = self._get_intelligent_action(current_player)
            else:
                self._log_round_event(f"Error: Unknown strategy '{current_player.strategy_type}'.")
                action = None

            if action is None:
                self._log_round_event(f"Player {current_player.player_id} ({current_player.get_display_name()}) forfeits the round.")
                self.game_active = False
                winner_round_id = None
                loser_round_ids = [current_player.player_id]
                break
            elif action == "CHALLENGE":
                try:
                    winner_round_id, loser_round_ids = self._resolve_challenge()
                except RuntimeError as e:
                    self._log_round_event(f"Error resolving challenge: {e}. Aborting round.")
                    winner_round_id, loser_round_ids = None, [p.player_id for p in self.players]
                break
            elif isinstance(action, Bid):
                self.current_bid = action
                self.bid_history.append((current_player.player_id, action))
                self._log_round_event(f"Player {current_player.player_id} bids: {action}")
                self.current_player_index = (self.current_player_index + 1) % len(self.players)

            if current_player.strategy_type == 'random':
                if current_player.effective_strategy == 'llm':
                    time.sleep(1.0)
                else:
                    time.sleep(0.1)
            elif current_player.strategy_type == 'llm':
                time.sleep(1.0)
            elif current_player.strategy_type in ('naive_5050', 'intelligent'):
                time.sleep(0.1)

        self._log_round_event("--- Round Ended ---")
        winner_original_order: Optional[int] = None
        winner_display_name: Optional[str] = None

        if winner_round_id is not None:
            winner_player = next((p for p in self.players if p.player_id == winner_round_id), None)
            if winner_player:
                winner_original_order = winner_player.original_order
                winner_display_name = winner_player.get_display_name()

        loser_original_orders: List[int] = []
        for lid in loser_round_ids:
            lp = next((p for p in self.players if p.player_id == lid), None)
            if lp:
                loser_original_orders.append(lp.original_order)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(GAME_LOGS_DIR, f"liars_poker_round_{timestamp}.log")
        try:
            with open(log_filename, "w", encoding='utf-8') as f:
                f.write("\n".join(self.round_log))
            logger.info(f"Round log saved to {log_filename}")
        except Exception as e:
            logger.error(f"Failed to save round log: {e}")

        hands_log_path = os.path.join(LOGS_DIR, "hands_log.json")
        hand_entry = {"timestamp": timestamp, "winner": winner_display_name, "losers": [], "round_log_file": os.path.basename(log_filename)}
        for lid in loser_round_ids:
            loser_p = next((p for p in self.players if p.player_id == lid), None)
            if loser_p:
                hand_entry["losers"].append(loser_p.get_display_name())

        try:
            with open(hands_log_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                content = f.read().strip()
                if content:
                    try:
                        hands_log = json.loads(content)
                        if not isinstance(hands_log, list):
                            logger.warning(f"hands_log.json wasn't a list. Overwriting.")
                            hands_log = []
                    except Exception:
                        logger.warning(f"hands_log.json invalid. Overwriting.")
                        hands_log = []
                else:
                    hands_log = []
                hands_log.append(hand_entry)
                f.seek(0)
                f.truncate(0)
                json.dump(hands_log, f, indent=2)
                fcntl.flock(f, fcntl.LOCK_UN)
            logger.info(f"Hand log updated in {hands_log_path}")
        except Exception as e:
            logger.error(f"Failed to update hands log: {e}")

        return winner_original_order, loser_original_orders, self.round_log

def get_player_configurations() -> List[Dict[str, Any]]:
    configs = []
    while True:
        try:
            num_players_str = input("Enter the number of players (2-6) [default: 2]: ")
            if not num_players_str:
                num_players = 2
            else:
                num_players = int(num_players_str)
            if 2 <= num_players <= 6:
                break
            else:
                print("Invalid number of players. Must be between 2 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

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

        num_rounds_str = input("\nHow many rounds? [default: 1]: ")
        if not num_rounds_str.strip():
            num_rounds = 1
        else:
            num_rounds = int(num_rounds_str)

        print("\nInitializing game...")
        game = LiarsPokerGame(player_configs=player_configs)

        for round_index in range(num_rounds):
            print(f"\n--- Starting Round {round_index + 1}/{num_rounds} ---")
            winner_original_order, loser_original_orders, round_log = game.play_round()

            current_players = game.players
            if winner_original_order is not None:
                w_player = next((p for p in current_players if p.original_order == winner_original_order), None)
                w_name = w_player.get_display_name() if w_player else "Unknown"
                print(f"\nWinner: Player {winner_original_order} ({w_name})")
            else:
                print("\nNo winner declared (forfeit or error).")

            print("Losers:")
            for lo in loser_original_orders:
                l_player = next((p for p in current_players if p.original_order == lo), None)
                l_name = l_player.get_display_name() if l_player else "Unknown"
                print(f" - Player {lo} ({l_name})")

        print(f"\nAll {num_rounds} rounds finished. Logs in {GAME_LOGS_DIR}, summary in {os.path.join(LOGS_DIR, 'hands_log.json')}")

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Game setup or execution failed: {e}", exc_info=True)
        print(f"\nError: {e}")
    except ImportError:
        logger.error("Failed to import llm_client. Please ensure it's available.")
        print("\nError: Could not import llm_client.py.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")