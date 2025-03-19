from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class BaseParser(ABC):
    """
    Abstract base class for parsing LLM responses into game actions.
    """
    
    @abstractmethod
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse raw LLM response text into a structured action object.
        
        Args:
            response_text: Raw text response from an LLM
            
        Returns:
            A structured action object compatible with the game engine
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        pass
    
    @abstractmethod
    def validate_action(self, action: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> bool:
        """
        Validate if the parsed action is valid according to game rules.
        
        Args:
            action: The parsed action to validate
            valid_actions: List of valid actions for the current game state
            
        Returns:
            True if action is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_fallback_action(self, valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide a fallback action when the LLM response is invalid or unclear.
        
        Args:
            valid_actions: List of valid actions for the current game state
            
        Returns:
            A valid action to use as fallback
        """
        pass
    
    def normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the action format to ensure compatibility with the game engine.
        
        Args:
            action: The action to normalize
            
        Returns:
            Normalized action
        """
        # Default implementation returns the action unchanged
        # Override in subclasses for game-specific normalization
        return action