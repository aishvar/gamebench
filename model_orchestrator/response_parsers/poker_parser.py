import json
import re
import logging
from typing import Dict, Any, List, Optional, Union
import random

from .base_parser import BaseParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokerResponseParser(BaseParser):
    """
    Parser for LLM responses in poker games.
    Handles JSON and text-based responses, normalizing them to game actions.
    """
    
    # Regular expression patterns for detecting poker actions
    ACTION_PATTERNS = {
        "fold": r"(?i)(?:I\s+)?(fold|folding|give[s]?\s+up|pass|surrender)",
        "check": r"(?i)(?:I\s+)?(check|checking|pass|stand\s+pat)",
        "call": r"(?i)(?:I\s+)?(call|calling|match|matches|meet|meeting)(?:\s+the\s+bet)?",
        "raise": r"(?i)(?:I\s+)?(raise|raising|bet|betting|increase|re-raise)(?:\s+(?:by|to)\s+)?(?:\$|£|€)?(\d+)?"
    }
    
    def __init__(self):
        """Initialize the poker response parser."""
        self.last_parsed_response = None
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse an LLM response into a structured poker action.
        
        Args:
            response_text: Raw text response from an LLM
            
        Returns:
            A structured action object compatible with the poker game engine
            
        Raises:
            ValueError: If the response cannot be parsed
        """
        self.last_parsed_response = response_text
        
        # First try to parse as JSON
        json_action = self._try_parse_json(response_text)
        if json_action:
            logger.debug(f"Successfully parsed JSON response: {json_action}")
            return self.normalize_action(json_action)
            
        # If not JSON, try to extract using regex patterns
        text_action = self._try_parse_text(response_text)
        if text_action:
            logger.debug(f"Successfully parsed text response: {text_action}")
            return self.normalize_action(text_action)
            
        # If all parsing attempts fail
        logger.warning(f"Failed to parse response: {response_text[:100]}...")
        raise ValueError(f"Could not parse poker action from response")
    
    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract and parse JSON from the response text.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed action dict or None if parsing fails
        """
        # Look for JSON block markers
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find a JSON-like structure with braces
            json_match = re.search(r'({[\s\S]*})', text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                return None
                
        try:
            # Parse the JSON
            action_data = json.loads(json_str)
            
            # Verify it has the expected structure
            if "action" in action_data:
                action_type = action_data["action"].lower()
                
                # Build standardized action
                if action_type in ["fold", "check"]:
                    return {"action_type": action_type}
                elif action_type == "call":
                    return {"action_type": "call"}
                elif action_type == "raise" or action_type == "bet":
                    # Ensure we have an amount for raise
                    if "amount" in action_data and isinstance(action_data["amount"], (int, float)):
                        return {
                            "action_type": "raise",
                            "amount": int(action_data["amount"])
                        }
            
            # If we got here, the JSON didn't have the expected structure
            logger.warning(f"JSON found but does not contain valid poker action: {action_data}")
            return None
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None
    
    def _try_parse_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract poker actions using regular expressions.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed action dict or None if parsing fails
        """
        # Check for fold
        if re.search(self.ACTION_PATTERNS["fold"], text):
            return {"action_type": "fold"}
            
        # Check for check
        if re.search(self.ACTION_PATTERNS["check"], text):
            return {"action_type": "check"}
            
        # Check for call
        if re.search(self.ACTION_PATTERNS["call"], text):
            return {"action_type": "call"}
            
        # Check for raise/bet with amount
        raise_match = re.search(self.ACTION_PATTERNS["raise"], text)
        if raise_match:
            # Try to extract amount
            amount_match = re.search(r'(?:by|to)?\s*[\$£€]?(\d+)', text)
            if amount_match and amount_match.group(1):
                try:
                    amount = int(amount_match.group(1))
                    return {
                        "action_type": "raise",
                        "amount": amount
                    }
                except ValueError:
                    # If amount is mentioned but we can't parse it
                    logger.warning(f"Found raise action but couldn't parse amount: {text}")
            
            # Raise without specific amount, need to be handled elsewhere
            return {"action_type": "raise"}
            
        # No recognized action found
        return None
    
    def validate_action(self, action: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> bool:
        """
        Validate if the parsed action is valid according to game rules.
        
        Args:
            action: The parsed action to validate
            valid_actions: List of valid actions for the current game state
            
        Returns:
            True if action is valid, False otherwise
        """
        if not action or "action_type" not in action:
            return False
            
        action_type = action["action_type"]
        
        # Check if action type is in valid actions
        valid_action_types = [a["action_type"] for a in valid_actions]
        if action_type not in valid_action_types:
            logger.warning(f"Action type '{action_type}' not in valid actions: {valid_action_types}")
            return False
            
        # For raise, check if amount is valid
        if action_type == "raise":
            # If no amount specified, can't validate
            if "amount" not in action:
                logger.warning("Raise action missing amount")
                return False
                
            # Find raise action to get min/max
            raise_action = next((a for a in valid_actions if a["action_type"] == "raise"), None)
            if not raise_action:
                logger.warning("Raise action not found in valid actions")
                return False
                
            # Check amount within range
            if action["amount"] < raise_action.get("min_amount", 0) or action["amount"] > raise_action.get("max_amount", float('inf')):
                logger.warning(f"Raise amount {action['amount']} outside valid range: {raise_action.get('min_amount', 0)} - {raise_action.get('max_amount', 'inf')}")
                return False
                
        return True
    
    def get_fallback_action(self, valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Provide a fallback action when the LLM response is invalid or unclear.
        Uses a conservative strategy (check/call/fold) to avoid risky plays.
        
        Args:
            valid_actions: List of valid actions for the current game state
            
        Returns:
            A valid action to use as fallback
        """
        # Conservative strategy for fallback:
        # 1. Check if possible
        # 2. Call if possible
        # 3. Fold as last resort
        
        # First try to find a check action
        check_action = next((a for a in valid_actions if a["action_type"] == "check"), None)
        if check_action:
            logger.info("Using fallback action: check")
            return {"action_type": "check"}
            
        # Then try to find a call action
        call_action = next((a for a in valid_actions if a["action_type"] == "call"), None)
        if call_action:
            logger.info("Using fallback action: call")
            return call_action
            
        # Finally, fold if nothing else works
        logger.info("Using fallback action: fold")
        return {"action_type": "fold"}
    
    def normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize the action format to ensure compatibility with the poker game engine.
        
        Args:
            action: The action to normalize
            
        Returns:
            Normalized action
        """
        if not action or "action_type" not in action:
            return action
            
        normalized = {"action_type": action["action_type"].lower()}
        
        # For fold and check, just the action type is needed
        if normalized["action_type"] in ["fold", "check"]:
            return normalized
            
        # For call, the amount will be determined by the game engine
        if normalized["action_type"] == "call":
            return normalized
            
        # For raise, ensure we have an amount
        if normalized["action_type"] == "raise":
            # Convert 'bet' to 'raise' for consistency
            if normalized["action_type"] == "bet":
                normalized["action_type"] = "raise"
                
            # Include amount if present
            if "amount" in action and isinstance(action["amount"], (int, float)):
                normalized["amount"] = int(action["amount"])
                
        return normalized