import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import random

from .prompt_templates.template_loader import load_template, render_template
from .response_parsers.base_parser import BaseParser
from .response_parsers.poker_parser import PokerResponseParser
from game_engines.base_game import BaseGame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameAdapter:
    """
    Adapter between game engines and LLM clients.
    Handles conversion of game states to prompts and LLM responses to game actions.
    """
    
    def __init__(
        self, 
        game: BaseGame, 
        game_type: str = "poker",
        system_template: str = None,
        prompt_template: str = None,
        response_parser: BaseParser = None
    ):
        """
        Initialize the game adapter.
        
        Args:
            game: Instance of a game (must inherit from BaseGame)
            game_type: Type of game for loading appropriate templates and parsers
            system_template: Optional custom system template name
            prompt_template: Optional custom prompt template name
            response_parser: Optional custom response parser
        """
        self.game = game
        self.game_type = game_type.lower()
        
        # Load templates
        self.system_template = self._load_system_template(system_template)
        self.prompt_template = self._load_prompt_template(prompt_template)
        
        # Initialize response parser
        self.response_parser = response_parser or self._initialize_parser()
        
        # Action history for context
        self.action_history = []
        self.max_history_items = 10
    
    def _load_system_template(self, template_name: Optional[str] = None) -> str:
        """Load the system template for the game type."""
        if template_name:
            return load_template(template_name)
        return load_template(f"{self.game_type}_system")
    
    def _load_prompt_template(self, template_name: Optional[str] = None) -> str:
        """Load the prompt template for the game type."""
        if template_name:
            return load_template(template_name)
        return load_template(f"{self.game_type}_prompt")
    
    def _initialize_parser(self) -> BaseParser:
        """Initialize the appropriate response parser for the game type."""
        if self.game_type == "poker":
            return PokerResponseParser()
        # Add more game types as they are implemented
        raise ValueError(f"Unsupported game type: {self.game_type}")
    
    def prepare_system_prompt(self) -> str:
        """
        Prepare the system prompt for the LLM.
        
        Returns:
            Formatted system prompt
        """
        # System prompt typically doesn't need variable substitution
        return self.system_template
    
    def prepare_prompt(self, player_id: str) -> str:
        """
        Prepare a game state prompt for the specified player.
        
        Args:
            player_id: ID of the player for whom to prepare the prompt
            
        Returns:
            Formatted prompt string
        """
        # Get game state from the player's perspective
        state = self.game.get_state(player_id)
        valid_actions = self.game.get_valid_actions(player_id)
        
        # Prepare variables for template substitution
        if self.game_type == "poker":
            variables = self._prepare_poker_variables(state, valid_actions, player_id)
        else:
            raise ValueError(f"Unsupported game type: {self.game_type}")
        
        # Render the template with the variables
        return render_template(self.prompt_template, variables)
    
    def _prepare_poker_variables(
    self, 
    state: Dict[str, Any], 
    valid_actions: List[Dict[str, Any]], 
    player_id: str
) -> Dict[str, Any]:
        """
        Prepare variables for poker prompt template.
        
        Args:
            state: Game state dictionary
            valid_actions: List of valid actions
            player_id: ID of the player
            
        Returns:
            Dictionary of variables for template substitution
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Get player-specific information
        player_info = state["players"].get(player_id, {})
        
        # Log player info for debugging
        logger.info(f"Preparing prompt for player {player_id}")
        logger.info(f"Player info: {player_info}")
        
        # Find opponent info
        opponent_id = next((p for p in state["players"].keys() if p != player_id), None)
        opponent_info = state["players"].get(opponent_id, {}) if opponent_id else {}
        
        # Format cards nicely
        hole_cards = player_info.get("hole_cards", [])
        logger.info(f"Player {player_id} hole cards: {hole_cards}")
        
        player_cards = self._format_cards(hole_cards)
        community_cards = self._format_cards(state.get("community_cards", []))
        
        # Format valid actions
        formatted_actions = self._format_valid_actions(valid_actions)
        
        # Format recent actions for context
        recent_actions = self._format_recent_actions()
        
        return {
            "hand_number": state.get("hand_number", 1),
            "stage": state.get("stage", "unknown").replace("pre-", "").capitalize(),
            "player_stack": player_info.get("stack", 0),
            "opponent_stack": opponent_info.get("stack", 0),
            "pot": state.get("pot", 0),
            "current_bet": state.get("current_bet", 0),
            "player_cards": player_cards,
            "community_cards": community_cards if community_cards else "None yet",
            "recent_actions": recent_actions,
            "valid_actions": formatted_actions
        }
    
    def _format_cards(self, cards: List[Tuple[str, str]]) -> str:
        """Format card tuples into readable text."""
        if not cards:
            return "None"
            
        formatted = []
        for rank, suit in cards:
            # Use unicode suit symbols if available
            suit_symbol = {
                "Hearts": "♥",
                "Diamonds": "♦",
                "Clubs": "♣",
                "Spades": "♠"
            }.get(suit, suit[0])
            
            formatted.append(f"{rank}{suit_symbol}")
            
        return ", ".join(formatted)
    
    def _format_valid_actions(self, valid_actions: List[Dict[str, Any]]) -> str:
        """Format the list of valid actions into readable text."""
        if not valid_actions:
            return "No valid actions available"
            
        formatted = []
        for action in valid_actions:
            action_type = action.get("action_type", "").capitalize()
            
            if action_type == "Fold":
                formatted.append("- Fold your hand")
            elif action_type == "Check":
                formatted.append("- Check (pass)")
            elif action_type == "Call":
                amount = action.get("amount", 0)
                formatted.append(f"- Call {amount} chips")
            elif action_type == "Raise":
                min_amount = action.get("min_amount", 0)
                max_amount = action.get("max_amount", "all-in")
                formatted.append(f"- Raise between {min_amount} and {max_amount} chips")
                
        return "\n".join(formatted)
    
    def _format_recent_actions(self) -> str:
        """Format recent actions from history for context."""
        if not self.action_history:
            return "No previous actions"
            
        # Get the most recent actions (limited by max_history_items)
        recent = self.action_history[-self.max_history_items:]
        return "\n".join(recent)
    
    def update_history(self, action_description: str):
        """
        Add an action description to the history.
        
        Args:
            action_description: Description of the action to add
        """
        self.action_history.append(action_description)
        # Trim history if it gets too long
        if len(self.action_history) > self.max_history_items * 2:
            self.action_history = self.action_history[-self.max_history_items:]
    
    def parse_response(self, response_text: str, valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse an LLM response into a game action.
        
        Args:
            response_text: Raw text response from the LLM
            valid_actions: List of valid actions for validation
            
        Returns:
            A validated action dict
        """
        try:
            # Try to parse the response
            action = self.response_parser.parse_response(response_text)
            
            # Validate the action
            if self.response_parser.validate_action(action, valid_actions):
                return action
            else:
                logger.warning(f"Invalid action: {action}")
                # Use fallback if validation fails
                return self.response_parser.get_fallback_action(valid_actions)
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            # Use fallback for parsing errors
            return self.response_parser.get_fallback_action(valid_actions)
    
    def apply_llm_action(self, player_id: str, llm_response: str) -> bool:
        """
        Process an LLM response and apply the resulting action to the game.
        
        Args:
            player_id: ID of the player taking the action
            llm_response: Raw text response from the LLM
            
        Returns:
            True if action was applied successfully, False otherwise
        """
        # Get valid actions for the player
        valid_actions = self.game.get_valid_actions(player_id)
        
        if not valid_actions:
            logger.warning(f"No valid actions for player {player_id}")
            return False
            
        # Parse the response into an action
        action = self.parse_response(llm_response, valid_actions)
        
        # Format and log the action
        action_description = self._format_action_description(player_id, action)
        self.update_history(action_description)
        
        # Apply the action to the game
        success = self.game.apply_action(action, player_id)
        
        return success
    
    def _format_action_description(self, player_id: str, action: Dict[str, Any]) -> str:
        """Format an action into a readable description for history."""
        action_type = action.get("action_type", "unknown").lower()
        
        if action_type == "fold":
            return f"{player_id} folds"
        elif action_type == "check":
            return f"{player_id} checks"
        elif action_type == "call":
            return f"{player_id} calls"
        elif action_type == "raise":
            amount = action.get("amount", "unknown")
            return f"{player_id} raises to {amount}"
        else:
            return f"{player_id} takes action: {json.dumps(action)}"