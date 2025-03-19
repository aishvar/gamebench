"""Configuration schemas for experiments."""

import jsonschema
from typing import Dict, Any, List

# Schema for model configurations
MODEL_SCHEMA = {
    "type": "object",
    "required": ["name", "provider"],
    "properties": {
        "name": {"type": "string"},
        "provider": {"type": "string", "enum": ["openai", "anthropic", "openrouter"]},
        "max_tokens": {"type": "integer", "minimum": 1},
        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
        "max_retries": {"type": "integer", "minimum": 0},
        "timeout": {"type": "integer", "minimum": 1}
    }
}

# Schema for game configurations
GAME_SCHEMA = {
    "type": "object",
    "required": ["type"],
    "properties": {
        "type": {"type": "string", "enum": ["poker"]},
        "starting_stack": {"type": "integer", "minimum": 1},
        "small_blind": {"type": "integer", "minimum": 1},
        "big_blind": {"type": "integer", "minimum": 1},
        "random_seed": {"type": "integer"}
    }
}

# Schema for experiment configurations
EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": ["name", "game", "models", "iterations"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "game": GAME_SCHEMA,
        "models": {
            "type": "array",
            "minItems": 1,
            "items": MODEL_SCHEMA
        },
        "iterations": {"type": "integer", "minimum": 1},
        "metrics": {
            "type": "array",
            "items": {"type": "string"}
        },
        "output": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "csv"]},
                "directory": {"type": "string"}
            }
        }
    }
}

# Functions for validating configuration objects
def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """Validate model configuration."""
    try:
        jsonschema.validate(instance=config, schema=MODEL_SCHEMA)
        return []
    except jsonschema.exceptions.ValidationError as e:
        return [f"Model config validation error: {e.message}"]

def validate_game_config(config: Dict[str, Any]) -> List[str]:
    """Validate game configuration."""
    try:
        jsonschema.validate(instance=config, schema=GAME_SCHEMA)
        return []
    except jsonschema.exceptions.ValidationError as e:
        return [f"Game config validation error: {e.message}"]

def validate_experiment_config(config: Dict[str, Any]) -> List[str]:
    """Validate experiment configuration."""
    try:
        jsonschema.validate(instance=config, schema=EXPERIMENT_SCHEMA)
        return []
    except jsonschema.exceptions.ValidationError as e:
        return [f"Experiment config validation error: {e.message}"]
