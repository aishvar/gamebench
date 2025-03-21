# model_orchestrator/utils.py

import os
import datetime
import logging
from typing import Dict, Any, Optional, TextIO, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global log file handle
_log_file: Optional[TextIO] = None
_log_filename: Optional[str] = None

def init_game_log(output_dir: str = "./logs") -> str:
    """Initialize a new game log file."""
    global _log_file, _log_filename
    
    # Close any existing log
    if _log_file is not None:
        _log_file.close()
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_filename = os.path.join(output_dir, f"poker_game_log_{timestamp}.txt")
    
    # Open the file
    _log_file = open(_log_filename, 'w')
    _log_file.write(f"POKER GAME LOG - {timestamp}\n")
    _log_file.write("="*80 + "\n\n")
    _log_file.flush()
    
    return _log_filename

def log_to_game_file(message: str, flush: bool = True):
    """Write a message to the game log file."""
    global _log_file
    
    if _log_file is not None:
        _log_file.write(message)
        if flush:
            _log_file.flush()

def log_initial_state(state: Dict[str, Any], hand_number: int):
    """Log the initial state of a hand."""
    message = []
    message.append(f"\n{'='*80}\n")
    message.append(f"HAND {hand_number} - INITIAL STATE\n")
    message.append(f"{'='*80}\n\n")
    
    # Format players
    message.append("Players:\n")
    for player_id, player_data in state.get("players", {}).items():
        cards = player_data.get("hole_cards", [])
        formatted_cards = format_cards(cards)
        message.append(f"  {player_id}: Stack={player_data.get('stack')} Cards={formatted_cards}\n")
    
    # Format community cards
    community_cards = state.get("community_cards", [])
    formatted_community = format_cards(community_cards)
    message.append(f"\nCommunity Cards: {formatted_community}\n")
    
    # Format other state info
    message.append(f"Pot: {state.get('pot', 0)}\n")
    message.append(f"Stage: {state.get('stage', 'unknown')}\n")
    message.append(f"Active Player: {state.get('active_player', 'unknown')}\n")
    message.append(f"Current Bet: {state.get('current_bet', 0)}\n\n")
    
    log_to_game_file("".join(message))

def log_llm_call(model_id: str, prompt: str, response: Optional[str] = None, 
                action: Optional[Dict[str, Any]] = None, is_retry: bool = False):
    """Log an LLM call and its result."""
    message = []
    
    if is_retry:
        message.append(f"\n{'#'*80}\n")
        message.append(f"RETRY LLM CALL TO {model_id}\n")
        message.append(f"{'#'*80}\n\n")
    else:
        message.append(f"\n{'#'*80}\n")
        message.append(f"LLM CALL TO {model_id}\n")
        message.append(f"{'#'*80}\n\n")
    
    # Write prompt (truncated if very long)
    message.append("PROMPT:\n")
    if len(prompt) > 1000:
        message.append(prompt[:1000] + "...\n")
    else:
        message.append(prompt + "\n")
    
    # Write response if available
    if response:
        message.append("\nRESPONSE:\n")
        if len(response) > 1000:
            message.append(response[:1000] + "...\n")
        else:
            message.append(response + "\n")
    
    # Write action if available
    if action:
        message.append("\nPARSED ACTION:\n")
        action_type = action.get("action_type", "unknown")
        if action_type == "fold":
            message.append("FOLD\n")
        elif action_type == "check":
            message.append("CHECK\n")
        elif action_type == "call":
            message.append(f"CALL: {action.get('amount', '')}\n")
        elif action_type == "raise":
            message.append(f"RAISE: {action.get('amount', '')}\n")
        else:
            message.append(f"{action}\n")
    
    log_to_game_file("".join(message))

def log_action_result(player_id: str, action: Dict[str, Any], state: Optional[Dict[str, Any]] = None):
    """Log the result of an action."""
    message = []
    message.append("\nACTION TAKEN:\n")
    
    action_type = action.get("action_type", "unknown")
    if action_type == "fold":
        message.append(f"{player_id} folds\n")
    elif action_type == "check":
        message.append(f"{player_id} checks\n")
    elif action_type == "call":
        amount = action.get("amount", "unknown")
        message.append(f"{player_id} calls {amount}\n")
    elif action_type == "raise":
        amount = action.get("amount", "unknown")
        message.append(f"{player_id} raises to {amount}\n")
    else:
        message.append(f"{player_id} takes action: {action}\n")
    
    # Write updated state if available
    if state:
        message.append("\nUPDATED STATE:\n")
        # Format pot and community cards
        pot = state.get("pot", 0)
        community_cards = state.get("community_cards", [])
        formatted_community = format_cards(community_cards)
        
        message.append(f"Pot: {pot}\n")
        message.append(f"Community Cards: {formatted_community}\n")
        message.append(f"Stage: {state.get('stage', 'unknown')}\n")
        message.append(f"Active Player: {state.get('active_player', 'unknown')}\n")
    
    log_to_game_file("".join(message))

def log_hand_result(result: Dict[str, Any], final_state: Optional[Dict[str, Any]] = None):
    """Log the result of a hand."""
    message = []
    message.append(f"\n{'='*80}\n")
    message.append("HAND RESULT\n")
    message.append(f"{'='*80}\n\n")
    
    winner = result.get("winner")
    if winner:
        message.append(f"Winner: {winner}\n")
    else:
        message.append("Result: Tie\n")
    
    message.append(f"Pot: {result.get('pot', 0)}\n")
    
    if "reason" in result:
        message.append(f"Reason: {result['reason']}\n")
    
    # Write final state if available
    if final_state:
        message.append("\nFinal State:\n")
        # Format player stacks
        for player_id, player_data in final_state.get("players", {}).items():
            message.append(f"  {player_id}: Stack={player_data.get('stack')}\n")
    
    message.append("\n")
    log_to_game_file("".join(message))

def close_game_log():
    """Close the game log file."""
    global _log_file
    
    if _log_file is not None:
        _log_file.close()
        _log_file = None

def format_cards(cards: List[Tuple[str, str]]) -> str:
    """Format cards for logging."""
    if not cards:
        return "None"
    
    formatted = []
    for rank, suit in cards:
        # Use suit symbols if possible
        suit_symbol = {
            "Hearts": "♥",
            "Diamonds": "♦",
            "Clubs": "♣",
            "Spades": "♠"
        }.get(suit, suit[0])
        
        formatted.append(f"{rank}{suit_symbol}")
    
    return ", ".join(formatted)