from abc import ABC, abstractmethod
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

class BaseGame(ABC):
    """
    Abstract base class for all game implementations.
    Defines standard interfaces for game state management, actions, and serialization.
    """
    
    def __init__(self, game_id: str = None, players: List[str] = None, random_seed: int = None):
        """
        Initialize a new game with optional configuration.
        
        Args:
            game_id: Unique identifier for this game session
            players: List of player identifiers
            random_seed: Optional seed for reproducible randomness
        """
        self.game_id = game_id or str(int(time.time()))
        self.players = players or []
        self.random_seed = random_seed
        self.history = []
        self.is_interactive = True
        self._initialize_game_state()
    
    @abstractmethod
    def _initialize_game_state(self) -> None:
        """Initialize the internal game state"""
        pass
    
    @abstractmethod
    def get_state(self, player_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a representation of the current game state.
        
        Args:
            player_id: If provided, return the state from this player's perspective
                      (hiding information not visible to this player)
        
        Returns:
            Dictionary containing game state information
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, player_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a list of valid actions for the given player or current player.
        
        Args:
            player_id: The player to get valid actions for, or current player if None
        
        Returns:
            List of action objects with their parameters
        """
        pass
    
    @abstractmethod
    def apply_action(self, action: Dict[str, Any], player_id: Optional[str] = None) -> bool:
        """
        Apply the given action to the game state.
        
        Args:
            action: Action to apply, typically a dict with action type and parameters
            player_id: The player performing the action, or current player if None
        
        Returns:
            True if the action was applied successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the game is in a terminal state (game over).
        
        Returns:
            True if the game is over, False otherwise
        """
        pass
    
    @abstractmethod
    def get_rewards(self) -> Dict[str, float]:
        """
        Get the rewards for each player at the current state.
        
        Returns:
            Dictionary mapping player IDs to their rewards
        """
        pass
    
    @abstractmethod
    def get_result(self) -> Dict[str, Any]:
        """
        Get the final game result (only valid in terminal states).
        
        Returns:
            Dictionary containing game outcome information
        """
        pass
    
    def set_interactive_mode(self, interactive: bool) -> None:
        """
        Set whether the game should run in interactive or non-interactive mode.
        
        Args:
            interactive: True for human interaction, False for programmatic mode
        """
        self.is_interactive = interactive
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event to the game history.
        
        Args:
            event_type: Type of event (e.g., 'action', 'state_change')
            data: Event data to log
        """
        self.history.append({
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete game history.
        
        Returns:
            List of historical events
        """
        return self.history
    
    def to_json(self) -> str:
        """
        Serialize the game state to JSON.
        
        Returns:
            JSON string representation of the game state
        """
        state = {
            'game_id': self.game_id,
            'players': self.players,
            'random_seed': self.random_seed,
            'state': self.get_state(),
            'is_terminal': self.is_terminal(),
            'history': self.history
        }
        return json.dumps(state, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseGame':
        """
        Create a game instance from a JSON string.
        
        Args:
            json_str: JSON string produced by to_json()
        
        Returns:
            A new game instance with the restored state
        """
        data = json.loads(json_str)
        game = cls(
            game_id=data['game_id'],
            players=data['players'],
            random_seed=data['random_seed']
        )
        game._restore_from_state(data)
        return game
    
    @abstractmethod
    def _restore_from_state(self, state_data: Dict[str, Any]) -> None:
        """
        Restore game state from serialized data.
        
        Args:
            state_data: The state data to restore from
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the game to its initial state"""
        pass