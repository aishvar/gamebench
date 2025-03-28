# model_orchestrator/utils.py

import os
import datetime
import logging
import json
# Corrected import: Added Union
from typing import Dict, Any, Optional, TextIO, List, Tuple, Union

# Configure logging
# logging.basicConfig(level=logging.INFO) # Usually configured by main script
logger = logging.getLogger(__name__)

# Global log file handle
_log_file: Optional[TextIO] = None
_log_filename: Optional[str] = None

def init_game_log(output_dir: str = "./logs") -> str:
    """Initialize a new game log file in the specified directory."""
    global _log_file, _log_filename

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created game log directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create log directory {output_dir}: {e}")
            # Fallback to current dir? Or raise? For now, log error and continue without file logging.
            _log_file = None
            _log_filename = None
            return ""

    # Close any existing log file handle
    if _log_file is not None and not _log_file.closed:
        try:
            _log_file.close()
            logger.debug("Closed existing game log file.")
        except Exception as e:
            logger.warning(f"Could not close previous log file: {e}")
        _log_file = None
        _log_filename = None


    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_filename = os.path.join(output_dir, f"poker_game_log_{timestamp}.txt")

    try:
        # Open the file in UTF-8 encoding
        _log_file = open(_log_filename, 'w', encoding='utf-8')
        header = f"POKER GAME LOG - {timestamp}\n"
        separator = "="*80 + "\n"
        _log_file.write(header)
        _log_file.write(separator + "\n")
        _log_file.flush()
        logger.info(f"Game log initialized: {_log_filename}")
        return _log_filename
    except IOError as e:
        logger.error(f"Failed to open game log file {_log_filename}: {e}")
        _log_file = None
        _log_filename = None
        return ""

def log_event_to_game_file(message: str, flush: bool = True):
    """Write a raw message to the game log file if it's open."""
    global _log_file
    if _log_file and not _log_file.closed:
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            _log_file.write(f"[{timestamp}] {message}")
            if not message.endswith('\n'):
                 _log_file.write('\n') # Ensure newline
            if flush:
                _log_file.flush()
        except Exception as e:
            logger.error(f"Failed to write to game log: {e}")
            # Consider closing the file if writing fails repeatedly
            # close_game_log()


def log_initial_state(state: Dict[str, Any], hand_number: int, stage: str):
    """Log the state at the beginning of a betting round."""
    if not _log_file or _log_file.closed: return

    lines = ["\n" + "="*30 + f" HAND {hand_number} | STAGE: {stage.upper()} " + "="*30]

    # Players
    lines.append("Players:")
    for p_name, p_data in state.get("players", {}).items():
        cards = p_data.get("hole_cards", "N/A")
        cards_str = format_cards(cards) if isinstance(cards, list) else str(cards)
        lines.append(f"  {p_name}: Stack={p_data.get('stack', '?')} | Bet(Round)={p_data.get('current_bet','?')} | Cards={cards_str} {'(Dealer)' if p_data.get('is_dealer') else ''}")

    # Community Cards
    community = state.get("community_cards", [])
    lines.append(f"Community: {format_cards(community) if community else 'None'}")

    # Pot & Betting
    lines.append(f"Pot: {state.get('pot', '?')}")
    # Use consistent naming - look for 'current_bet_level' first from adapter context, fallback to 'current_bet'
    current_bet_level = state.get('current_bet_level', state.get('current_bet', '?'))
    lines.append(f"Current Bet Level: {current_bet_level}")


    # Active Player
    lines.append(f"Active Player: {state.get('active_player', 'N/A')}")

    # Valid Actions (Optional, can be long)
    valid_actions = state.get('valid_actions', [])
    if valid_actions:
         lines.append("Valid Actions:")
         # Format actions slightly better for the log
         for action in valid_actions:
              action_str = f"  - {action['action_type']}"
              if action['action_type'] == 'call':
                   action_str += f" (Amount: {action.get('amount', '?')})"
              elif action['action_type'] == 'raise':
                   action_str += f" (Min: {action.get('min_amount', '?')}, Max: {action.get('max_amount', '?')})"
              lines.append(action_str)
         # lines.append(f"  - {action}") # Old way: Log the action dict representation
    lines.append("="*80 + "\n")

    log_event_to_game_file("\n".join(lines))


def log_llm_call(model_id: str, prompt: str, response_text: Optional[str],
                parsed_action: Optional[Dict[str, Any]] = None, is_retry: bool = False):
    """Log an LLM call, response, and parsed action to the game log."""
    if not _log_file or _log_file.closed: return

    marker = "#"*80
    title = f"LLM CALL{' (Retry)' if is_retry else ''}: {model_id}"
    lines = ["\n" + marker, title, marker + "\n"]

    # Prompt (truncated)
    lines.append("--- PROMPT --->")
    prompt_lines = prompt.splitlines()
    lines.extend(prompt_lines[:50]) # Limit prompt lines in log
    if len(prompt_lines) > 50:
         lines.append("... [prompt truncated]")
    lines.append("<--- END PROMPT ---\n")


    # Response (truncated)
    lines.append("--- RESPONSE --->")
    if response_text:
        # Handle potential multi-line JSON within response text for logging
        try:
            # Attempt to format if it looks like JSON
            if (response_text.strip().startswith('{') and response_text.strip().endswith('}')) or \
               (response_text.strip().startswith('[') and response_text.strip().endswith(']')):
                parsed_json = json.loads(response_text.strip())
                response_log_text = json.dumps(parsed_json, indent=2)
            else:
                 response_log_text = response_text
        except json.JSONDecodeError:
             response_log_text = response_text # Log as is if not valid JSON

        response_lines = response_log_text.splitlines()
        lines.extend(response_lines[:20]) # Limit response lines
        if len(response_lines) > 20:
             lines.append("... [response truncated]")
    else:
        lines.append("!!! No response text received or LLM call failed !!!")
    lines.append("<--- END RESPONSE ---\n")

    # Parsed Action
    if parsed_action:
        # Log parsed action cleanly
        lines.append(f"--- PARSED ACTION: {json.dumps(parsed_action)} ---\n")
    elif not response_text: # Check if response_text was None/empty
        lines.append("--- PARSED ACTION: N/A (LLM Call Failed or No Response Text) ---\n")
    else: # If response_text existed but parsing failed
        lines.append("--- PARSED ACTION: FAILED or NO ACTION FOUND in response ---\n")


    log_event_to_game_file("\n".join(lines))


def log_action_result(player_id: str, action_type: str, amount: Optional[Union[int, float]], updated_state: Dict[str, Any]):
    """Log the result of applying an action, reflecting the state *after* the action."""
    if not _log_file or _log_file.closed: return

    lines = ["\n" + "-"*30 + f" ACTION RESULT: {player_id} " + "-"*30] # Changed title for clarity
    action_desc = action_type.upper()
    if amount is not None and action_type in ['call', 'raise']: # Show amount only for call/raise
         action_desc += f" {amount}"
    lines.append(f"Action By: {player_id} -> {action_desc}") # Clarify who acted

    # Log key state changes affecting the player who acted
    player_state = updated_state.get("players", {}).get(player_id, {})
    lines.append(f"  {player_id} New Stack: {player_state.get('stack', '?')}")
    lines.append(f"  {player_id} Bet This Round: {player_state.get('current_bet', '?')}")

    # Overall state AFTER action and player switch
    lines.append(f"New Pot Size: {updated_state.get('pot', '?')}")
    # Use consistent naming - get 'current_bet' from state as it's the canonical value
    lines.append(f"New Bet Level: {updated_state.get('current_bet', '?')}")
    lines.append(f"Current Stage: {updated_state.get('stage', '?')}") # Show current stage
    # This uses the active_player from the state *after* the switch happened in apply_action
    lines.append(f"Next Active Player: {updated_state.get('active_player', 'N/A')}")
    lines.append("-"*80 + "\n")

    log_event_to_game_file("\n".join(lines))


def log_hand_result(final_state: Dict[str, Any], hand_number: int):
    """Log the result of a completed hand using final state."""
    if not _log_file or _log_file.closed: return

    lines = ["\n" + "="*30 + f" HAND {hand_number} RESULT " + "="*30]

    # Find winner/reason from history if possible (more reliable than inferring from state)
    winner_name = "Unknown"
    reason = "Unknown"
    winning_hand_desc = ""
    hand1_desc = ""
    hand2_desc = ""
    pot_awarded = final_state.get("pot", 0) # Pot should be 0 after award, get from history?

    # Look backwards in history for 'hand_result' event for this hand
    history = final_state.get('history', []) # Get history from state if available
    hand_result_event_data = None
    showdown_event_data = None

    for event in reversed(history):
         # Ensure event is a dict and has the expected keys
         if isinstance(event, dict) and event.get('hand') == hand_number:
              if event.get('event_type') == 'hand_result' and not hand_result_event_data:
                    hand_result_event_data = event.get('data', {})
              elif event.get('event_type') == 'showdown' and not showdown_event_data:
                   showdown_event_data = event.get('data', {})
         # Stop if we have both or go back too far
         if hand_result_event_data and showdown_event_data: break
         if isinstance(event, dict) and event.get('hand') is not None and event.get('hand') < hand_number: break


    if hand_result_event_data:
        winner_from_event = hand_result_event_data.get('winner') # Can be None for tie
        winner_name = 'Tie' if winner_from_event is None else winner_from_event
        reason = hand_result_event_data.get('reason', '?')
        winning_hand_desc = hand_result_event_data.get('winning_hand_desc', '')
        # Get hand descriptions from result event if present (for ties)
        hand1_desc = hand_result_event_data.get('hand1_desc', '')
        hand2_desc = hand_result_event_data.get('hand2_desc', '')
        pot_awarded = hand_result_event_data.get('pot_awarded', pot_awarded)


    # Get player names from state if needed
    player_names = list(final_state.get("players", {}).keys())
    p1_name = player_names[0] if len(player_names) > 0 else "Player1"
    p2_name = player_names[1] if len(player_names) > 1 else "Player2"

    # Try getting hand descriptions from showdown event if not in result event
    if not hand1_desc and showdown_event_data:
         hand1_desc = showdown_event_data.get("players", {}).get(p1_name, {}).get("hand_description", "")
    if not hand2_desc and showdown_event_data:
         hand2_desc = showdown_event_data.get("players", {}).get(p2_name, {}).get("hand_description", "")


    lines.append(f"Winner: {winner_name}") # Display 'Tie' or player name
    lines.append(f"Reason: {reason}")
    if winning_hand_desc:
        lines.append(f"Winning Hand Desc: {winning_hand_desc}")
    elif reason == "tie_showdown":
         lines.append(f"  {p1_name} Hand: {hand1_desc}")
         lines.append(f"  {p2_name} Hand: {hand2_desc}")
    lines.append(f"Pot Awarded: {pot_awarded}")

    # Show final stacks
    lines.append("Final Stacks After Hand:")
    for p_name, p_data in final_state.get("players", {}).items():
        lines.append(f"  {p_name}: {p_data.get('stack', '?')}")

    lines.append("="*80 + "\n")
    log_event_to_game_file("\n".join(lines))

def close_game_log():
    """Close the game log file if it's open."""
    global _log_file, _log_filename
    if _log_file and not _log_file.closed:
        try:
            _log_file.write("\n--- END OF GAME LOG ---\n")
            _log_file.close()
            logger.info(f"Game log closed: {_log_filename}")
            _log_file = None
            _log_filename = None
        except Exception as e:
            logger.error(f"Error closing game log: {e}")

def format_cards(cards: List[Tuple[str, str]]) -> str:
    """Format a list of card tuples into a readable string."""
    if not cards:
        return "None"

    # Unicode symbols (use alternatives if encoding issues persist)
    suit_map = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠"}
    # Fallback map
    # suit_map = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}

    formatted = []
    for card in cards:
        # Add defensive check if card is not a tuple/list of len 2
        if isinstance(card, (list, tuple)) and len(card) == 2:
            rank, suit = card
            suit_symbol = suit_map.get(suit, suit[0]) # Fallback to first letter
            formatted.append(f"{rank}{suit_symbol}")
        elif isinstance(card, str) and card == "Hidden": # Handle the 'Hidden' placeholder
            formatted.append("[Hidden]")
        else:
            logger.warning(f"Unexpected card format in format_cards: {card}")
            formatted.append("?")


    return ", ".join(formatted)

# Add function to parse LLM response text (moved from llm_client for utility access)
def parse_response_text(response_json: Dict[str, Any]) -> Optional[str]:
    """Safely parse the assistant's text content from different response formats."""
    if not response_json: return None
    content = None
    try:
        # OpenAI / OpenRouter (common format) - Check 'message' first for non-streaming
        if "choices" in response_json and isinstance(response_json["choices"], list) and response_json["choices"]:
            choice = response_json["choices"][0]
            # Check message for complete response
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if content: return str(content)
            # Check delta for streaming response chunk
            if "delta" in choice and isinstance(choice["delta"], dict):
                content = choice["delta"].get("content")
                if content: return str(content)


        # Anthropic Claude - Check list structure first
        if "content" in response_json and isinstance(response_json["content"], list) and response_json["content"]:
            first_block = response_json["content"][0]
            if isinstance(first_block, dict) and first_block.get("type") == "text":
                 content = first_block.get("text")
                 if content: return str(content)
            # Check for Pydantic model attribute if not dict (seen with Anthropic client)
            elif hasattr(first_block, 'text'):
                 content = first_block.text
                 if content: return str(content)

        # Anthropic streaming delta
        if "delta" in response_json and isinstance(response_json["delta"], dict):
             if response_json["delta"].get("type") == "text_delta":
                  content = response_json["delta"].get("text")
                  if content: return str(content)
             # Handle Anthropic message_delta (for message completion events)
             elif response_json["delta"].get("type") == "message_delta":
                  # Usage info might be here, but not text content itself
                  pass


        # Older / Other formats (add as needed)
        if "completion" in response_json: # e.g., older OpenAI completion endpoint
             content = response_json["completion"]
             if isinstance(content, str): return content # Ensure it's a string


        if content is None:
             # Check if it's maybe a simple string response directly? Unlikely for structured APIs.
             if isinstance(response_json, str):
                  return response_json # If the whole thing is just a string

             logger.warning(f"Could not extract text content from LLM response structure: Keys={list(response_json.keys())}")
             # Log the structure for debugging
             # logger.debug(f"Full response structure for parsing failure: {json.dumps(response_json, default=str)}")
             return None

    except Exception as e:
        logger.error(f"Error parsing LLM response text: {e}. Response keys: {list(response_json.keys())}")
        # logger.debug(f"Full response structure during parsing error: {json.dumps(response_json, default=str)}")
        return None

    return str(content) if content is not None else None