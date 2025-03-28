# model_orchestrator/game_adapter.py

import logging
import json
import time # Import time if needed for history timestamps etc.
from typing import Dict, Any, List, Optional, Tuple, Union

from .prompt_templates.template_loader import load_template, render_template
from .response_parsers.base_parser import BaseParser
from .response_parsers.poker_parser import PokerResponseParser
from game_engines.base_game import BaseGame
from game_engines.heads_up_poker import HeadsUpPoker # Import specific type if needed
from .utils import format_cards # Use the logging utility for card formatting

# Configure logging
# logging.basicConfig(level=logging.INFO) # Configured by runner usually
logger = logging.getLogger(__name__)

class GameAdapter:
    """
    Adapter between game engines and LLM clients.
    Handles conversion of game states to prompts and LLM responses to game actions.
    Now accepts the game object directly for state preparation.
    """

    def __init__(
        self,
        # game: BaseGame, # No longer store game instance here
        game_type: str = "poker",
        system_template_path: str = None, # Use full path/name
        prompt_template_path: str = None, # Use full path/name
        response_parser: BaseParser = None,
        max_history_items: int = 10
    ):
        """
        Initialize the game adapter.

        Args:
            game_type: Type of game for loading appropriate templates and parsers.
            system_template_path: Optional custom system template name (e.g., 'poker_system').
            prompt_template_path: Optional custom prompt template name (e.g., 'poker_prompt').
            response_parser: Optional custom response parser.
            max_history_items: Max number of recent actions to include in prompt.
        """
        # self.game = game # Don't store game, pass it to methods instead
        self.game_type = game_type.lower()
        self.max_history_items = max_history_items

        # Load templates using names (e.g., 'poker_system')
        sys_template_name = system_template_path or f"{self.game_type}_system"
        prompt_template_name = prompt_template_path or f"{self.game_type}_prompt"
        self.system_template = load_template(sys_template_name)
        self.prompt_template = load_template(prompt_template_name)

        # Initialize response parser
        self.response_parser = response_parser or self._initialize_parser()

        # Action history is now derived from the game object's history


    def _initialize_parser(self) -> BaseParser:
        """Initialize the appropriate response parser for the game type."""
        if self.game_type == "poker":
            return PokerResponseParser()
        # Add more game types as they are implemented
        logger.error(f"Unsupported game type for parser: {self.game_type}")
        raise ValueError(f"Unsupported game type: {self.game_type}")

    def prepare_system_prompt(self) -> str:
        """Prepare the system prompt for the LLM."""
        # System prompt typically doesn't need variable substitution from game state
        return self.system_template

    def prepare_prompt(self, game: BaseGame, player_id: str) -> str:
        """
        Prepare a game state prompt for the specified player using the current game object.

        Args:
            game: The current game instance (e.g., HeadsUpPoker).
            player_id: ID of the player for whom to prepare the prompt.

        Returns:
            Formatted prompt string.
        """
        # Get game state *from the game object* for the player's perspective
        state = game.get_state(player_id)
        # Valid actions are now part of the state dict returned by get_state
        valid_actions = state.get("valid_actions", [])

        # Prepare variables for template substitution based on game type
        if isinstance(game, HeadsUpPoker): # Check if it's a poker game
            variables = self._prepare_poker_variables(state, valid_actions, player_id, game.history)
        else:
            logger.error(f"Cannot prepare prompt variables for unsupported game type: {type(game)}")
            raise ValueError(f"Unsupported game type for prompt preparation: {type(game)}")

        # Render the template with the variables
        try:
             return render_template(self.prompt_template, variables)
        except KeyError as e:
             logger.error(f"Missing variable for prompt template: {e}. State: {state}, Variables: {variables}")
             # Return a fallback prompt or raise
             return f"Error: Missing data for prompt ({e}). State: {json.dumps(state)}"
        except Exception as e:
             logger.error(f"Error rendering prompt template: {e}")
             return f"Error: Could not render prompt. State: {json.dumps(state)}"


    def _prepare_poker_variables(
        self,
        state: Dict[str, Any], # State dict already from player's perspective
        valid_actions: List[Dict[str, Any]],
        player_id: str,
        game_history: List[Dict[str, Any]] # Pass the full history
    ) -> Dict[str, Any]:
        """Prepare variables for the poker prompt template."""
        player_info = state["players"].get(player_id, {})
        opponent_id = next((p_name for p_name in state["players"] if p_name != player_id), None)
        opponent_info = state["players"].get(opponent_id, {}) if opponent_id else {}

        # --- Extract and Format Data ---
        player_stack = player_info.get("stack", 0)
        opponent_stack = opponent_info.get("stack", 0)
        pot = state.get("pot", 0)
        current_bet_level = state.get("current_bet", 0) # Highest bet this round
        player_current_bet = player_info.get("current_bet", 0) # Player's bet this round
        call_amount_needed = max(0, current_bet_level - player_current_bet)

        # Hole cards (already filtered by get_state)
        hole_cards = player_info.get("hole_cards", "Unknown") # Should have cards or 'Hidden'
        player_cards_str = format_cards(hole_cards) if isinstance(hole_cards, list) else str(hole_cards)

        # Community cards (only show if stage is appropriate)
        stage = state.get("stage", "unknown")
        community_cards = state.get("community_cards", [])
        community_cards_str = "None yet"
        if stage in ["flop", "turn", "river", "showdown"] and community_cards:
            community_cards_str = format_cards(community_cards)

        # Format recent actions from game history
        recent_actions_str = self._format_recent_actions(game_history)

        # Format valid actions clearly
        formatted_valid_actions = self._format_valid_actions(valid_actions, player_stack, call_amount_needed)

        return {
            "hand_number": state.get("hand_number", "?"),
            "stage": stage.replace("-", " ").title(), # Format stage nicely
            "player_stack": player_stack,
            "opponent_stack": opponent_stack,
            "pot": pot,
            "current_bet_level": current_bet_level, # Renamed for clarity
            "player_current_bet": player_current_bet, # Added player's current round bet
            "call_amount_needed": call_amount_needed, # Added amount needed to call
            "player_cards": player_cards_str,
            "community_cards": community_cards_str,
            "recent_actions": recent_actions_str,
            "valid_actions": formatted_valid_actions
        }

    def _format_recent_actions(self, game_history: List[Dict[str, Any]]) -> str:
        """Format recent actions from the game history log."""
        action_events = [
            event for event in game_history
            if event.get("event_type") == "action" and event.get("hand") == game_history[-1].get("hand") # Only actions from current hand
        ]

        if not action_events:
            return "No actions yet this hand."

        # Get the most recent actions
        recent = action_events[-self.max_history_items:]
        formatted = []
        for event in recent:
            data = event.get("data", {})
            player = data.get("player", "?")
            action = data.get("action", "?")
            amount = data.get("amount")
            stage = event.get("stage", "?") # Include stage if available in event

            desc = f"({stage}) {player} {action}"
            if amount is not None:
                desc += f" {amount}"
            formatted.append(desc)

        return "\n".join(formatted) if formatted else "No actions yet this hand."

    def _format_valid_actions(self, valid_actions: List[Dict[str, Any]], player_stack: int, call_amount_needed: int) -> str:
        """Format the list of valid actions into readable text for the prompt."""
        if not valid_actions:
            return "No valid actions available (hand may be over)."

        formatted = []
        for action in valid_actions:
            action_type = action.get("action_type", "").lower()

            if action_type == "fold":
                formatted.append("- Fold")
            elif action_type == "check":
                formatted.append("- Check")
            elif action_type == "call":
                amount_to_call = action.get("amount", call_amount_needed) # Use calculated amount
                effective_call = min(amount_to_call, player_stack)
                call_desc = f"- Call {effective_call}"
                if effective_call < amount_to_call:
                     call_desc += " (All-in)"
                formatted.append(call_desc)
            elif action_type == "raise":
                min_amount = action.get("min_amount", 0)
                max_amount = action.get("max_amount", player_stack) # Max is usually player's stack
                # Clarify if max_amount means 'all-in'
                max_desc = f"{max_amount}"
                if max_amount == player_stack:
                     max_desc += " (All-in)"

                # Ensure min raise is possible
                if min_amount <= max_amount:
                     if min_amount == max_amount: # Only one raise amount possible (all-in for less than min raise)
                          formatted.append(f"- Raise {min_amount} (All-in)")
                     else:
                          formatted.append(f"- Raise (Add between {min_amount} and {max_desc})")
                else:
                     # Should not happen with correct get_valid_actions logic
                     logger.warning(f"Invalid raise range generated: min {min_amount}, max {max_amount}")


        return "\n".join(formatted)


    def parse_and_validate_response(self, response_text: str, valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse LLM response, validate against valid actions, and return a valid action (or fallback).

        Args:
            response_text: Raw text response from the LLM.
            valid_actions: List of valid actions for the current game state.

        Returns:
            A validated action dict (could be fallback).
        """
        if not valid_actions:
             logger.warning("parse_and_validate_response called with no valid actions.")
             # Return a dummy action or handle appropriately
             return {"action_type": "fold"} # Default to fold if no actions possible

        try:
            # Try to parse the response
            parsed_action = self.response_parser.parse_response(response_text)

            # Validate the parsed action against the official valid actions list
            if self.response_parser.validate_action(parsed_action, valid_actions):
                logger.info(f"Parsed and validated action: {parsed_action}")
                # Ensure amount is correctly set for raise if parsed from text without it
                if parsed_action["action_type"] == "raise" and "amount" not in parsed_action:
                     # Find the raise details in valid_actions
                     raise_details = next((va for va in valid_actions if va["action_type"] == "raise"), None)
                     if raise_details:
                          # Default to minimum raise if amount missing and validation passed? Risky.
                          # Parser should ideally extract amount. If not, fallback might be safer.
                          logger.warning(f"Raise parsed without amount, validation passed? Using min raise amount {raise_details['min_amount']}")
                          parsed_action["amount"] = raise_details["min_amount"]
                     else:
                          logger.error("Raise validated but details not found in valid_actions. Falling back.")
                          return self.response_parser.get_fallback_action(valid_actions)

                return parsed_action
            else:
                logger.warning(f"Parsed action {parsed_action} failed validation against {valid_actions}. Using fallback.")
                return self.response_parser.get_fallback_action(valid_actions)

        except Exception as e:
            logger.error(f"Error parsing response '{response_text[:100]}...': {e}. Using fallback.")
            return self.response_parser.get_fallback_action(valid_actions)


    # No longer needed - history comes from game object
    # def update_history(self, action_description: str): ...

    # No longer needed - apply action happens in runner/game loop
    # def apply_llm_action(self, player_id: str, llm_response: str) -> bool: ...

    # No longer needed - formatting happens in _format_recent_actions
    # def _format_action_description(self, player_id: str, action: Dict[str, Any]) -> str: ...
