import os
import yaml
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get template directory path
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

def load_template(template_name: str) -> str:
    """
    Load a template from the templates directory.
    
    Args:
        template_name: Name of the template file (with or without extension)
        
    Returns:
        The template content as a string
        
    Raises:
        FileNotFoundError: If the template file doesn't exist
        ValueError: If the template is invalid
    """
    # Ensure the template has the correct extension
    if not template_name.endswith((".txt", ".yaml", ".yml")):
        template_name += ".txt"
        
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    
    try:
        with open(template_path, "r") as f:
            content = f.read()
            
        # Basic validation - check that there are format placeholders
        if "{" not in content or "}" not in content:
            logger.warning(f"Template {template_name} may not have format placeholders")
            
        return content
        
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise FileNotFoundError(f"Template file not found: {template_name}")
    except Exception as e:
        logger.error(f"Error loading template {template_name}: {e}")
        raise ValueError(f"Invalid template {template_name}: {e}")

def render_template(template_content: str, variables: Dict[str, Any]) -> str:
    """
    Render a template by substituting variables.
    
    Args:
        template_content: The template content with {placeholders}
        variables: Dictionary of variables to substitute
        
    Returns:
        The rendered template
        
    Raises:
        KeyError: If a required variable is missing
        ValueError: If template rendering fails
    """
    try:
        # Perform basic variable substitution
        return template_content.format(**variables)
    except KeyError as e:
        missing_key = str(e).strip("'")
        logger.error(f"Missing required variable in template: {missing_key}")
        raise KeyError(f"Missing required template variable: {missing_key}")
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise ValueError(f"Template rendering failed: {e}")

def load_yaml_template(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template (useful for complex templates with multiple sections).
    
    Args:
        template_name: Name of the YAML template file
        
    Returns:
        The parsed YAML content
        
    Raises:
        FileNotFoundError: If the template file doesn't exist
        ValueError: If the YAML is invalid
    """
    # Ensure the template has the correct extension
    if not template_name.endswith((".yaml", ".yml")):
        template_name += ".yaml"
        
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    
    try:
        with open(template_path, "r") as f:
            content = yaml.safe_load(f)
            
        return content
        
    except FileNotFoundError:
        logger.error(f"YAML template file not found: {template_path}")
        raise FileNotFoundError(f"YAML template file not found: {template_name}")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in template {template_name}: {e}")
        raise ValueError(f"Invalid YAML in template {template_name}: {e}")
