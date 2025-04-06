import random
import logging
import time
import os
import json
import re
from typing import List, Tuple, Dict, Optional, NamedTuple, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

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
            # Dummy response for testing purposes
            print(f"--- Dummy LLM Call ({self.model}) ---")
            print(f"System: {system_message}")
            print(f"Developer: {developer_message}")
            print(f"User: {user_message}")
            # Simulate a valid response structure
            dummy_reasoning = "This is a dummy reasoning based on the prompt."
            # Simulate either a bid or challenge based on simple logic for testing
            if "Current Bid: None" in user_message:
                dummy_action = "BID: 2 5s"
            else:
                 # Try to make a slightly higher bid or challenge randomly
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


            response_content = json.dumps({
                "reasoning": dummy_reasoning,
                "action": dummy_action
            })
            # Simulate the structure returned by the actual client
            return {
                "model": self.model,
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [{"message": {"content": response_content}}]
            }

    def parse_response_text(response_json: Optional[Dict[str, Any]]) -> Optional[str]:
         if response_json and "choices" in response_json and response_json["choices"]:
              message = response_json["choices"][0].get("message", {})
              content = message.get("content")
              if content:
                  return str(content).strip()
         return None

    def log_llm_call(response_json: Optional[Dict[str, Any]], prompt_messages: Dict[str, str], start_time: float):
         # Dummy logging function
         if response_json:
             print(f"Dummy LLM Call Logged: Model={response_json.get('model', 'N/A')}, Time={time.time() - start_time:.2f}s")
         else:
             print("Dummy LLM Call Logged: Failed call")

# Override LOGS_DIR to save logs in the 'data' directory within the current folder
LOGS_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging for the game
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LiarsPokerGame")

# Directory for game summary logs
GAME_LOGS_DIR = os.path.join(LOGS_DIR, "game_logs")
os.makedirs(GAME_LOGS_DIR, exist_ok=True)

# --- Data Structures ---

class Bid(NamedTuple):
    """Represents a bid in Liar's Poker."""
    quantity: int
    digit: int

    def __str__(self):
        return f"{self.quantity} {self.digit}s"

    def is_higher_than(self, other: Optional['Bid']) -> bool:
        """Checks if this bid is strictly higher than another bid."""
        if other is None:
            return True # Any bid is higher than no bid
        if self.quantity > other.quantity:
            return True
        if self.quantity == other.quantity and self.digit > other.digit:
            return True
        return False

@dataclass
class Player:
    """Represents a player in the game."""
    player_id: int
    model_config: Dict[str, str] # e.g., {"provider": "openai", "model": "gpt-4o-mini"}
    client: LLMClient = field(init=False)
    hand: List[int] = field(default_factory=list)
    original_order: int = 0 # To track initial position if needed

    def __post_init__(self):
        """Initialize the LLM client after dataclass initialization."""
        logger.info(f"Initializing LLM client for Player {self.player_id}: {self.model_config}")
        try:
            self.client = LLMClient(
                provider=self.model_config["provider"],
                model=self.model_config["model"],
                max_tokens=10000, # Increased slightly for JSON + reasoning
                temperature=0.5, # Encourage more deterministic actions
                max_retries=2,
                timeout=60 # Increased slightly for potentially longer reasoning
            )
        except ValueError as e:
            logger.error(f"Failed to initialize LLM client for Player {self.player_id}: {e}")
            # Re-raise or handle appropriately - for now, re-raise to stop setup
            raise
        except KeyError as e:
            logger.error(f"Missing key in model_config for Player {self.player_id}: {e}")
            raise ValueError(f"Invalid model_config for Player {self.player_id}: {self.model_config}")


# --- Game Logic ---

class LiarsPokerGame:
    """Manages the Liar's Poker game simulation."""

    MAX_DIGITS_PER_HAND = 8
    MAX_ACTION_PARSE_ATTEMPTS = 2 # How many times to re-prompt if action is invalid

    def __init__(self, player_configs: List[Dict[str, str]]):
        """
        Initializes the game with player configurations.

        Args:
            player_configs: A list of dictionaries, each specifying
                            {"provider": <provider_name>, "model": <model_name>}
                            for a player.
        """
        if not (2 <= len(player_configs) <= 6):
            raise ValueError("Number of players must be between 2 and 6.")

        self.players: List[Player] = []
        for i, config in enumerate(player_configs):
             # Add original_order before shuffling
            self.players.append(Player(player_id=i, model_config=config, original_order=i))

        self.current_player_index: int = 0
        self.current_bid: Optional[Bid] = None
        self.bid_history: List[Tuple[int, Bid]] = [] # Stores (player_id, Bid)
        self.round_log: List[str] = [] # Log messages for the current round
        self.total_digits_in_play: int = 0 # Calculated during setup_round
        self.game_active: bool = False

    def _log_round_event(self, message: str):
        """Logs an event to both the logger and the round log list."""
        logger.info(message)
        self.round_log.append(message)

    def _generate_hands(self) -> List[List[int]]:
        """Generates unique 8-digit hands for all players."""
        num_players = len(self.players)
        # Ensure enough digits for potential uniqueness issues
        possible_digits = list(range(10)) * (num_players * self.MAX_DIGITS_PER_HAND // 10 + num_players) # Add extra margin
        random.shuffle(possible_digits)

        hands_int = []
        start_index = 0
        for _ in range(num_players):
            # Take 8 digits for the hand
            hand = possible_digits[start_index : start_index + self.MAX_DIGITS_PER_HAND]
            if len(hand) != self.MAX_DIGITS_PER_HAND:
                 # This should ideally not happen with sufficient initial digits
                 raise RuntimeError(f"Could not generate enough unique digits for hands. Need {self.MAX_DIGITS_PER_HAND}, got {len(hand)}")
            hands_int.append(hand)
            start_index += self.MAX_DIGITS_PER_HAND

        # Simple check for duplicate hands (unlikely with random dealing but possible)
        hand_strs = {"".join(map(str, sorted(h))) for h in hands_int}
        if len(hand_strs) != num_players:
            self._log_round_event("Warning: Duplicate hands generated, proceeding anyway.") # Or could regenerate

        self._log_round_event(f"Generated {num_players} hands.")
        return hands_int


    def _setup_round(self):
        """Sets up a new round: shuffles players, deals hands."""
        self._log_round_event("--- Starting New Round ---")
        # 1. Shuffle player order for this round
        random.shuffle(self.players)
        # Update player_id based on shuffled order for round clarity
        for i, player in enumerate(self.players):
            player.player_id = i # This ID is relative to the current round's turn order

        self._log_round_event(f"Player order: {[p.model_config['model'] for p in self.players]}")

        # 2. Generate and assign hands
        hands = self._generate_hands()
        for i, player in enumerate(self.players):
            player.hand = hands[i]
            # Log hands only to file, not console by default unless debugging
            logger.debug(f"Player {player.player_id} ({player.model_config['model']}) Hand: {''.join(map(str, player.hand))}")
        self._log_round_event(f"Hands dealt to {len(self.players)} players.")

        # 3. Reset round state
        self.current_player_index = 0
        self.current_bid = None
        self.bid_history = []
        self.round_log = [ # Keep only recent setup logs
            self.round_log[0], # "--- Starting New Round ---"
            self.round_log[1], # "Player order: ..."
            self.round_log[2]  # "Hands dealt..."
        ]
        self.total_digits_in_play = len(self.players) * self.MAX_DIGITS_PER_HAND
        self.game_active = True
        self._log_round_event("Round setup complete. Starting bids.")

    def _get_prompt_context(self, player: Player) -> Tuple[str, str, str]:
        """Generates the developer, user, and system messages for the LLM."""
        system_message = (
            "You are a strategic player in a game of Liar's Poker. "
            "Analyze the situation, your hand, and the bidding history to make the best move. "
            "Your goal is to either make a valid higher bid or challenge the last bid if you think it's unlikely. "
            "Provide your response as a JSON object containing two keys: 'reasoning' (explaining your thought process) "
            "and 'action' (containing *only* your chosen action string: 'BID: [quantity] [digit]s' or 'CHALLENGE')."
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
        )

        # Construct user message with current game state
        hand_str = "".join(map(str, player.hand))
        player_list_str = ", ".join([f"Player {p.player_id} ({p.model_config['model']})" for p in self.players])
        history_str = "\n".join([f"  - Player {pid} bid: {bid}" for pid, bid in self.bid_history]) if self.bid_history else "  - No bids yet."

        user_message = (
            f"Game State:\n"
            f"- Your Hand: {hand_str}\n"
            f"- Players: {player_list_str}\n"
            f"- Your Turn: Player {player.player_id} ({player.model_config['model']})\n"
            f"- Current Bid: {self.current_bid or 'None'}"
        )
        if self.current_bid:
             last_bidder_id = self.bid_history[-1][0]
             last_bidder_model = next(p.model_config['model'] for p in self.players if p.player_id == last_bidder_id)
             user_message += f" (made by Player {last_bidder_id} - {last_bidder_model})"

        user_message += f"\n- Bid History:\n{history_str}\n\n"
        user_message += "What is your action? Provide your reasoning and action in the specified JSON format."

        return developer_message, user_message, system_message

    def _parse_action_string(self, action_str: Optional[str]) -> Union[Bid, str, None]:
        """Parses the extracted action string into a Bid object or 'CHALLENGE'."""
        if not action_str:
            self._log_round_event("Extracted action string was empty.")
            return None

        action_str = action_str.strip().upper()
        self._log_round_event(f"Attempting to parse extracted action string: '{action_str}'")

        # Check for challenge first
        if action_str == "CHALLENGE":
             # Check if challenging is valid (must be a bid to challenge)
            if self.current_bid is None:
                self._log_round_event("Parse Error: Tried to CHALLENGE when no bid exists.")
                return None # Invalid action
            return "CHALLENGE"

        # Check for bid format "BID: [quantity] [digit]s"
        match = re.match(r"BID:\s*(\d+)\s+(\d)S?", action_str)
        if match:
            try:
                quantity = int(match.group(1))
                digit = int(match.group(2))

                if not (0 <= digit <= 9):
                     self._log_round_event(f"Parse Error: Invalid digit {digit} in action string.")
                     return None
                if not (1 <= quantity <= self.total_digits_in_play):
                     self._log_round_event(f"Parse Error: Invalid quantity {quantity} (max is {self.total_digits_in_play}) in action string.")
                     return None

                bid = Bid(quantity=quantity, digit=digit)

                # Check if bid is validly higher than current bid
                if not bid.is_higher_than(self.current_bid):
                    self._log_round_event(f"Parse Error: Bid {bid} is not higher than current bid {self.current_bid}.")
                    return None

                return bid
            except ValueError:
                self._log_round_event("Parse Error: Could not convert bid parts to integers from action string.")
                return None
            except Exception as e:
                self._log_round_event(f"Parse Error: Unexpected error parsing bid from action string - {e}")
                return None

        self._log_round_event("Parse Error: Action string did not match 'CHALLENGE' or 'BID: [quantity] [digit]s' format.")
        return None

    def _get_player_action(self, player: Player) -> Union[Bid, str, None]:
        """Gets and validates an action from the specified player's LLM (expecting JSON)."""
        developer_msg, user_msg, system_msg = self._get_prompt_context(player)

        for attempt in range(self.MAX_ACTION_PARSE_ATTEMPTS + 1):
            self._log_round_event(f"Requesting action from Player {player.player_id} ({player.model_config['model']}), Attempt {attempt + 1}/{self.MAX_ACTION_PARSE_ATTEMPTS + 1}")

            start_time = time.time()
            response_payload = player.client.call_llm(
                developer_message=developer_msg,
                user_message=user_msg,
                system_message=system_msg
            )
            # Assuming log_llm_call is defined elsewhere and handles the payload
            try:
                 log_llm_call(
                     response_json=response_payload,
                     prompt_messages={"developer": developer_msg, "user": user_msg, "system": system_msg},
                     start_time=start_time
                 )
            except NameError:
                 logger.warning("log_llm_call function not found, skipping LLM call logging.")
            except Exception as log_err:
                 logger.error(f"Error during LLM call logging: {log_err}")


            if response_payload is None:
                self._log_round_event(f"Player {player.player_id} LLM call failed after retries.")
                if attempt == self.MAX_ACTION_PARSE_ATTEMPTS:
                     return None # Forfeit on final attempt failure
                continue # Try prompting again

            response_text = parse_response_text(response_payload) # Get the raw text content from the payload
            if response_text is None:
                 self._log_round_event(f"Player {player.player_id} response content was empty or missing.")
                 if attempt == self.MAX_ACTION_PARSE_ATTEMPTS:
                     return None
                 continue # Try prompting again

            # --- MODIFIED: Clean and Parse JSON ---
            parsed_action = None
            cleaned_response_text = response_text.strip()

            # Remove Markdown code fences (common pattern)
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[len("```json"):].strip()
            if cleaned_response_text.startswith("```"): # Handle case without 'json' tag
                cleaned_response_text = cleaned_response_text[len("```"):].strip()
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-len("```")].strip()

            try:
                # Attempt to parse the cleaned response text as JSON
                llm_output = json.loads(cleaned_response_text)

                if not isinstance(llm_output, dict):
                     # Log both original and cleaned text for easier debugging
                     self._log_round_event(f"Player {player.player_id} response is not a JSON object after cleaning: '{cleaned_response_text}' (Original: '{response_text}')")
                elif "action" not in llm_output or "reasoning" not in llm_output:
                     self._log_round_event(f"Player {player.player_id} JSON response missing 'action' or 'reasoning' key: {llm_output}")
                elif not isinstance(llm_output.get("action"), str):
                     self._log_round_event(f"Player {player.player_id} 'action' value is not a string: {llm_output}")
                else:
                    # Successfully parsed JSON and found action string
                    action_str = llm_output["action"]
                    reasoning_str = llm_output.get("reasoning", "[No reasoning provided]")
                    self._log_round_event(f"Player {player.player_id} Reasoning: {reasoning_str}")

                    # Now parse the extracted action string using the original logic
                    parsed_action = self._parse_action_string(action_str)

            except json.JSONDecodeError:
                # Log both original and cleaned text for easier debugging
                self._log_round_event(f"Player {player.player_id} response was not valid JSON after cleaning: '{cleaned_response_text}' (Original: '{response_text}')")
            except Exception as e:
                 self._log_round_event(f"Player {player.player_id} unexpected error processing JSON response: {e}")
            # --- End JSON Parsing ---

            if parsed_action is not None:
                # Valid action parsed from the 'action' field
                return parsed_action
            else:
                 # Invalid action format/logic OR JSON parsing failed
                 self._log_round_event(f"Player {player.player_id} provided invalid action or malformed/unparsable JSON. Re-prompting...")
                 # Modify user message slightly to emphasize format on retry
                 user_msg += "\n\n**Please ensure your output is ONLY the valid JSON object with 'reasoning' and 'action' keys, and the 'action' value follows the required format ('BID: ...' or 'CHALLENGE'). Do not include explanations outside the JSON or markdown formatting like ```json.**"

        # If loop finishes without returning, all attempts failed
        self._log_round_event(f"Player {player.player_id} failed to provide a valid action after {self.MAX_ACTION_PARSE_ATTEMPTS + 1} attempts.")
        return None # Player forfeits
    
    def _count_digit_occurrences(self, digit: int) -> int:
        """Counts the total occurrences of a digit across all player hands."""
        count = 0
        all_hands_str = ""
        for p in self.players:
            count += p.hand.count(digit)
            all_hands_str += f" P{p.player_id}:{''.join(map(str, p.hand))}" # For logging
        logger.debug(f"Counting digit {digit} across hands:{all_hands_str} -> Count: {count}")
        return count

    def _resolve_challenge(self) -> Tuple[int, List[int]]:
        """
        Resolves a challenge by counting digits and determining the winner/loser.

        Returns:
            Tuple[int, List[int]]: (winner_player_id, loser_player_ids)
        """
        if not self.current_bid or not self.bid_history:
             # This should be caught earlier, but defensive check
             raise RuntimeError("Cannot resolve challenge without a current bid history.")

        challenger_id = self.current_player_index
        challenged_bidder_id, challenged_bid = self.bid_history[-1]

        challenger_model = self.players[challenger_id].model_config['model']
        # Find the player object corresponding to the challenged bidder ID
        challenged_bidder_player = next((p for p in self.players if p.player_id == challenged_bidder_id), None)
        if not challenged_bidder_player:
             raise RuntimeError(f"Could not find player object for challenged bidder ID {challenged_bidder_id}")
        challenged_bidder_model = challenged_bidder_player.model_config['model']


        self._log_round_event(f"Player {challenger_id} ({challenger_model}) challenges Player {challenged_bidder_id}'s ({challenged_bidder_model}) bid of {challenged_bid}.")

        # Reveal hands for verification logging
        hands_reveal_log = "Revealed Hands: " + " | ".join([f"P{p.player_id}({p.model_config['model']}):{''.join(map(str,p.hand))}" for p in self.players])
        self._log_round_event(hands_reveal_log)

        actual_count = self._count_digit_occurrences(challenged_bid.digit)
        self._log_round_event(f"Actual count of {challenged_bid.digit}s across all hands: {actual_count}")

        winner_id = None
        # Get all player IDs to identify losers (everyone except winner)
        all_player_ids = [p.player_id for p in self.players]

        if actual_count >= challenged_bid.quantity:
            # Bidder was correct (or understated), challenger loses
            winner_id = challenged_bidder_id
            self._log_round_event(f"Challenge Failed! Actual count ({actual_count}) >= Bid quantity ({challenged_bid.quantity}).")
            self._log_round_event(f"Winner: Player {winner_id} ({challenged_bidder_model})")
        else:
            # Bidder was wrong (overstated), challenger wins
            winner_id = challenger_id
            self._log_round_event(f"Challenge Successful! Actual count ({actual_count}) < Bid quantity ({challenged_bid.quantity}).")
            self._log_round_event(f"Winner: Player {winner_id} ({challenger_model})")

        # Everyone except the winner is a loser
        loser_ids = [pid for pid in all_player_ids if pid != winner_id]
        loser_models = [next(p.model_config['model'] for p in self.players if p.player_id == pid) for pid in loser_ids]
        self._log_round_event(f"Losers: Players {', '.join([f'{pid} ({model})' for pid, model in zip(loser_ids, loser_models)])}")

        self.game_active = False # Mark game as finished for this round
        return winner_id, loser_ids

    def play_round(self) -> Tuple[Optional[int], List[int], List[str]]:
        """
        Plays a single round of Liar's Poker.

        Returns:
            Tuple[Optional[int], List[int], List[str]]:
                (winner_player_original_order, loser_player_original_orders, round_log)
                Returns (None, list_of_loser_original_orders, log) if round ends with a forfeit.
        """
        self._setup_round()
        winner_id: Optional[int] = None
        loser_ids: List[int] = []

        while self.game_active:
            current_player = self.players[self.current_player_index]
            self._log_round_event(f"\n--- Turn: Player {current_player.player_id} ({current_player.model_config['model']}) ---")

            action = self._get_player_action(current_player)

            if action is None:
                # Player failed to provide a valid action after retries -> Forfeits
                self._log_round_event(f"Player {current_player.player_id} ({current_player.model_config['model']}) forfeits the round due to invalid/failed action.")
                # In a forfeit, the forfeiting player is the sole loser
                self.game_active = False
                winner_id = None
                loser_ids = [current_player.player_id]  # Only the forfeiting player is a loser
                break # End the round

            elif action == "CHALLENGE":
                # Resolve the challenge and end the round
                try:
                    winner_id, loser_ids = self._resolve_challenge()
                except RuntimeError as e:
                    self._log_round_event(f"Error resolving challenge: {e}. Ending round abnormally.")
                    winner_id, loser_ids = None, []
                # self.game_active is set to False inside _resolve_challenge (or above on error)
                break # Exit the loop

            elif isinstance(action, Bid):
                # Player made a valid bid
                self.current_bid = action
                self.bid_history.append((current_player.player_id, action))
                self._log_round_event(f"Player {current_player.player_id} ({current_player.model_config['model']}) bids: {action}")
                # Move to the next player
                self.current_player_index = (self.current_player_index + 1) % len(self.players)

            # Optional: Add a small delay between turns if not using dummy client
            if 'Dummy LLMClient' not in str(self.players[0].client.__class__):
                 time.sleep(1.0) # Slightly longer delay for real LLMs

        # --- Round End ---
        self._log_round_event("--- Round Ended ---")

        # Map round winner/loser IDs back to original player order
        winner_original_order = next((p.original_order for p in self.players if p.player_id == winner_id), None) if winner_id is not None else None
        loser_original_orders = [next((p.original_order for p in self.players if p.player_id == loser_id), None) for loser_id in loser_ids]
        loser_original_orders = [order for order in loser_original_orders if order is not None]  # Filter out any None values

        # Save the round log to a file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_filename = os.path.join(GAME_LOGS_DIR, f"liars_poker_round_{timestamp}.log")
        try:
            with open(log_filename, "w", encoding='utf-8') as f: # Ensure utf-8 encoding
                f.write("\n".join(self.round_log))
            logger.info(f"Round log saved to {log_filename}")
        except Exception as e:
            logger.error(f"Failed to save round log: {e}")

        # Save hand result to hands_log.json
        hands_log_path = os.path.join(LOGS_DIR, "hands_log.json")
        hand_entry = {
            "timestamp": timestamp,
            "winner": winner_original_order,
            "losers": loser_original_orders,
            "round_log_file": os.path.basename(log_filename)
        }

        try:
            # Read existing log entries or create a new list
            if os.path.exists(hands_log_path):
                with open(hands_log_path, "r") as f:
                    hands_log = json.load(f)
            else:
                hands_log = []
            
            # Append new entry
            hands_log.append(hand_entry)
            
            # Write updated log
            with open(hands_log_path, "w") as f:
                json.dump(hands_log, f, indent=2)
            
            logger.info(f"Hand log updated in {hands_log_path}")
        except Exception as e:
            logger.error(f"Failed to update hands log: {e}")

        return winner_original_order, loser_original_orders, self.round_log


# --- Main Execution ---

def get_player_configurations() -> List[Dict[str, str]]:
    """Gets player model configurations from the user."""
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
                print("Invalid number of players. Please enter a number between 2 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\nEnter LLM configurations for each player.")
    # Adjusted default/example providers and models
    print("Supported providers: openai, anthropic, openrouter (or others supported by your llm_client.py)")
    print("Example model for openai: gpt-4o-mini")
    print("Example model for anthropic: claude-3-haiku-20240307")
    print("Example model for openrouter: meta-llama/llama-3-8b-instruct, google/gemini-flash-1.5")
    print("Ensure API keys (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY) are set as environment variables as needed by llm_client.py.")

    # Pre-defined common configs for easier input
    common_configs = {
        "1": ("openai", "gpt-4o-mini"),
        "2": ("anthropic", "claude-3-haiku-20240307"),
        "3": ("openrouter", "google/gemini-2.5-pro-exp-03-25:free"),
        "4": ("openrouter", "deepseek/deepseek-chat-v3-0324:free"),
        "5": ("openai", "gpt-4o"),
        "6": ("anthropic", "claude-3-sonnet-20240229"),
        "7": ("openrouter", "meta-llama/llama-4-maverick:free"),
    }

    for i in range(num_players):
        while True:
            print(f"\n--- Player {i+1} ---")
            print("Common Choices:")
            for key, (prov, mod) in common_configs.items():
                print(f"  {key}: {prov}/{mod}")
            print("  C: Custom input")

            choice = input(f"Select configuration for Player {i+1} (1-{len(common_configs)}) or C: ").strip().upper()

            selected_config = None
            if choice in common_configs:
                selected_config = common_configs[choice]
                provider, model = selected_config
                print(f"Selected: {provider}/{model}")
            elif choice == 'C':
                provider = input(f"Enter provider for Player {i+1}: ").lower().strip()
                # Basic validation - allow any non-empty string if llm_client is flexible
                if not provider:
                    print("Provider cannot be empty.")
                    continue

                model = input(f"Enter model name for Player {i+1}: ").strip()
                if not model:
                    print("Model name cannot be empty.")
                    continue
                selected_config = (provider, model)
            else:
                print("Invalid choice.")
                continue

            if selected_config:
                configs.append({"provider": selected_config[0], "model": selected_config[1]})
                print(f"Player {i+1} configured: Provider={selected_config[0]}, Model={selected_config[1]}")
                break # Exit inner loop once valid config is added
    return configs

if __name__ == "__main__":
    print("--- Liar's Poker LLM Benchmark ---")

    try:
        # Check if running with dummy client
        if 'Dummy LLMClient' in str(LLMClient):
            print("\nWARNING: Running with DUMMY LLM Client. No real API calls will be made.\n")
            # Use default dummy configs if running dummy
            player_configs = [
                {"provider": "dummy_openai", "model": "dummy_gpt4"},
                {"provider": "dummy_anthropic", "model": "dummy_claude3"}
            ]
            print("Using default dummy configurations:")
            for i, cfg in enumerate(player_configs):
                 print(f" Player {i+1}: {cfg['provider']}/{cfg['model']}")
        else:
             player_configs = get_player_configurations()


        print("\nInitializing game...")
        game = LiarsPokerGame(player_configs=player_configs)

        print("\nStarting game round...")
        winner_original_order, loser_original_orders, round_log = game.play_round()

        print("\n--- Game Round Finished ---")
        if winner_original_order is not None:
            # Find the original config using original_order
            winner_config = next(cfg for i, cfg in enumerate(player_configs) if i == winner_original_order)
            print(f"Winner: Player {winner_original_order} ({winner_config['provider']}/{winner_config['model']})")
            print("Losers:")
            for loser_order in loser_original_orders:
                loser_config = next(cfg for i, cfg in enumerate(player_configs) if i == loser_order)
                print(f" - Player {loser_order} ({loser_config['provider']}/{loser_config['model']})")
        else:
            # No winner (forfeit case)
            print("No winner declared.")
            print("Losers (forfeited):")
            for loser_order in loser_original_orders:
                loser_config = next(cfg for i, cfg in enumerate(player_configs) if i == loser_order)
                print(f" - Player {loser_order} ({loser_config['provider']}/{loser_config['model']})")

        print(f"\nFull round log saved in {GAME_LOGS_DIR}")
        print(f"Cumulative game results saved in {os.path.join(LOGS_DIR, 'hands_log.json')}")
        # print("\n".join(round_log)) # Uncomment to print full log to console as well

    except (ValueError, RuntimeError, KeyError) as e:
        logger.error(f"Game setup or execution failed: {e}", exc_info=True)
        print(f"\nError: {e}")
    except ImportError:
         # Already handled by the dummy client check, but good practice
         logger.error("Failed to import llm_client. Please ensure it's available.")
         print("\nError: Could not import llm_client.py.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")