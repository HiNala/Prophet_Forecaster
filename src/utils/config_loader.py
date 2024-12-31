"""
Configuration loader utility for the Prophet Forecaster application.
Handles loading and parsing of the YAML configuration file.
"""

import os
import yaml
from typing import Dict, Any

def get_config_path() -> str:
    """Get the path to the configuration file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, 'config', 'config.yaml')

def load_config() -> Dict[str, Any]:
    """
    Load the configuration from the YAML file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If the configuration file is invalid
    """
    config_path = get_config_path()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}")

def get_setting(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get a nested configuration setting.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        *keys (str): Sequence of keys to access nested config
        default (Any, optional): Default value if setting not found
    
    Returns:
        Any: Configuration value or default if not found
    """
    try:
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default 