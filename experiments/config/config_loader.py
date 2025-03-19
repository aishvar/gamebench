"""Functions for loading and validating configuration files."""

import os
import yaml
import json
import logging
from typing import Dict, Any, List, Union, Optional

from .config_schema import (
    validate_model_config, 
    validate_game_config, 
    validate_experiment_config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration directories
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(CONFIG_DIR, "default_configs")

def load_config(
    config_path: str, 
    config_type: str = "experiment"
) -> Dict[str, Any]:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file
        config_type: Type of configuration (experiment, model, or game)
        
    Returns:
        Parsed and validated configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file is invalid
    """
    # Check if file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load file based on extension
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext in ('.yaml', '.yml'):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif file_ext == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.error(f"Unsupported config file format: {file_ext}")
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise ValueError(f"Error loading config file: {e}")
    
    # Validate configuration
    errors = validate_config(config, config_type)
    if errors:
        logger.error(f"Invalid configuration: {', '.join(errors)}")
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")
    
    # Process environment variables
    config = process_env_vars(config)
    
    # For experiment configs, merge with defaults
    if config_type == "experiment":
        config = merge_with_defaults(config)
    
    return config

def validate_config(
    config: Dict[str, Any], 
    config_type: str = "experiment"
) -> List[str]:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration
        
    Returns:
        List of validation errors (empty if valid)
    """
    if config_type == "model":
        return validate_model_config(config)
    elif config_type == "game":
        return validate_game_config(config)
    elif config_type == "experiment":
        return validate_experiment_config(config)
    else:
        return [f"Unknown config type: {config_type}"]

def merge_configs(
    base_config: Dict[str, Any], 
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add the value
            result[key] = value
            
    return result

def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge an experiment configuration with default configs.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Configuration with defaults merged in
    """
    # Load default configs
    try:
        # Default game config
        game_type = config.get("game", {}).get("type", "poker")
        default_game_path = os.path.join(DEFAULT_CONFIG_DIR, f"{game_type}.yaml")
        
        if os.path.exists(default_game_path):
            with open(default_game_path, 'r') as f:
                default_game = yaml.safe_load(f)
            # Merge with user-provided game config
            if "game" in config:
                config["game"] = merge_configs(default_game, config["game"])
            else:
                config["game"] = default_game
        
        # Default model configs
        default_models_path = os.path.join(DEFAULT_CONFIG_DIR, "models.yaml")
        if os.path.exists(default_models_path):
            with open(default_models_path, 'r') as f:
                default_models = yaml.safe_load(f)
            
            # If user provided model references instead of full configs, resolve them
            if "models" in config:
                resolved_models = []
                for model_config in config["models"]:
                    if isinstance(model_config, str):
                        # If it's just a string, look it up in default models
                        if model_config in default_models:
                            resolved_models.append(default_models[model_config])
                        else:
                            logger.warning(f"Unknown model reference: {model_config}")
                    else:
                        # If it's a dict, use it as is
                        resolved_models.append(model_config)
                config["models"] = resolved_models
        
        # Default experiment config
        default_experiment_path = os.path.join(DEFAULT_CONFIG_DIR, "experiment.yaml")
        if os.path.exists(default_experiment_path):
            with open(default_experiment_path, 'r') as f:
                default_experiment = yaml.safe_load(f)
            # Merge with user-provided config, preserving user values
            config = merge_configs(default_experiment, config)
            
    except Exception as e:
        logger.warning(f"Error merging default configs: {e}")
        # Continue with user config if defaults can't be loaded
        
    return config

def process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process environment variable references in a configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        result = {}
        for key, value in config.items():
            result[key] = process_env_vars(value)
        return result
    elif isinstance(config, list):
        return [process_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract environment variable name
        env_var = config[2:-1]
        # Get the value, with optional default after colon
        if ":" in env_var:
            env_name, default = env_var.split(":", 1)
            return os.environ.get(env_name, default)
        else:
            return os.environ.get(env_var, config)
    else:
        return config