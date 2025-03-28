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
    if _log_file is not None:
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
         for action in valid_actions:
              lines.append(f"  - {action}") # Log the action dict representation
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
        response_lines = response_text.splitlines()
        lines.extend(response_lines[:20]) # Limit response lines
        if len(response_lines) > 20:
             lines.append("... [response truncated]")
    else:
        lines.append("!!! No response text received or LLM call failed !!!")
    lines.append("<--- END RESPONSE ---\n")

    # Parsed Action
    if parsed_action:
        lines.append(f"--- PARSED ACTION: {parsed_action} ---\n")
    elif not response_text: # Check if response_text was None/empty
        lines.append("--- PARSED ACTION: N/A (LLM Call Failed or No Response Text) ---\n")
    else: # If response_text existed but parsing failed
        lines.append("--- PARSED ACTION: FAILED or NO ACTION FOUND in response ---\n")


    log_event_to_game_file("\n".join(lines))


def log_action_result(player_id: str, action_type: str, amount: Optional[Union[int, float]], updated_state: Dict[str, Any]):
    """Log the result of applying an action."""
    if not _log_file or _log_file.closed: return

    lines = ["\n" + "-"*30 + f" ACTION APPLIED: {player_id} " + "-"*30]
    action_desc = action_type.upper()
    if amount is not None:
         action_desc += f" {amount}"
    lines.append(f"Action: {action_desc}")

    # Log key state changes
    player_state = updated_state.get("players", {}).get(player_id, {})
    lines.append(f"  New Stack: {player_state.get('stack', '?')}")
    lines.append(f"  Bet This Round: {player_state.get('current_bet', '?')}")
    lines.append(f"New Pot Size: {updated_state.get('pot', '?')}")
    lines.append(f"New Bet Level: {updated_state.get('current_bet', '?')}") # Current bet TO CALL
    lines.append(f"Next Stage: {updated_state.get('stage', '?')}")
    lines.append(f"Next Active Player: {updated_state.get('active_player', 'N/A')}")
    lines.append("-"*80 + "\n")

    log_event_to_game_file("\n".join(lines))


def log_hand_result(final_state: Dict[str, Any], hand_number: int):
    """Log the result of a completed hand using final state."""
    if not _log_file or _log_file.closed: return

    lines = ["\n" + "="*30 + f" HAND {hand_number} RESULT " + "="*30]

    # Find winner/reason from history if possible (more reliable than inferring from state)
    winner = "Unknown"
    reason = "Unknown"
    winning_hand_desc = ""
    # Look backwards in history for 'hand_result' event for this hand
    history = final_state.get('history', []) # Get history from state if available
    for event in reversed(history):
         # Ensure event is a dict and has the expected keys
         if isinstance(event, dict) and event.get('event_type') == 'hand_result' and event.get('hand') == hand_number:
              data = event.get('data', {})
              winner = data.get('winner', 'Tie') # Default to Tie if winner key missing
              reason = data.get('reason', '?')
              winning_hand_desc = data.get('winning_hand_desc', '')
              # Ensure winner is None if it's 'Tie' or explicitly None in data
              if winner == 'Tie' or winner is None:
                   winner = None
              break # Found the result event


    lines.append(f"Winner: {'Tie' if winner is None else winner}") # Display 'Tie' if winner is None
    lines.append(f"Reason: {reason}")
    if winning_hand_desc:
        lines.append(f"Winning Hand: {winning_hand_desc}")

    # Show final stacks
    lines.append("Final Stacks:")
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
        # OpenAI / OpenRouter (common format)
        if "choices" in response_json and isinstance(response_json["choices"], list) and response_json["choices"]:
            choice = response_json["choices"][0]
            # Check delta for streaming response chunk
            if "delta" in choice and isinstance(choice["delta"], dict):
                content = choice["delta"].get("content")
                if content: return str(content)
            # Check message for complete response
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content")
                if content: return str(content)

        # Anthropic Claude
        if "content" in response_json and isinstance(response_json["content"], list) and response_json["content"]:
            first_block = response_json["content"][0]
            if isinstance(first_block, dict) and first_block.get("type") == "text":
                 content = first_block.get("text")
                 if content: return str(content)
            # Check for Pydantic model attribute if not dict (less common for final response)
            elif hasattr(first_block, 'text'):
                 content = first_block.text
                 if content: return str(content)
        # Anthropic streaming delta
        if "delta" in response_json and isinstance(response_json["delta"], dict):
             if response_json["delta"].get("type") == "text_delta":
                  content = response_json["delta"].get("text")
                  if content: return str(content)


        # Older / Other formats (add as needed)
        if "completion" in response_json: # e.g., older OpenAI completion endpoint
             content = response_json["completion"]
             if isinstance(content, str): return content # Ensure it's a string


        if content is None:
             # Check if it's maybe a simple string response directly? Unlikely for structured APIs.
             if isinstance(response_json, str):
                  return response_json # If the whole thing is just a string

             logger.warning(f"Could not extract text content from LLM response structure: Keys={list(response_json.keys())}")
             return None

    except Exception as e:
        logger.error(f"Error parsing LLM response text: {e}. Response keys: {list(response_json.keys())}")
        return None

    return str(content) if content is not None else None

