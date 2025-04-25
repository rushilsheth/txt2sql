"""
Configuration Module

This module provides configuration settings for the text-to-SQL application.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

import yaml

from text_to_sql.utils.config_types import (
    SystemConfig, DatabaseConfig, LLMConfig, 
    AppConfig, LoggingConfig, AgentConfig
)

logger = logging.getLogger(__name__)

# Default configuration values are defined in the typed classes

def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SystemConfig instance with configuration values
    """
    # Start with an empty dictionary
    config_dict = {}
    
    # If config path is provided, load from file
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Update config with values from file
            if file_config:
                config_dict = file_config
            
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # Override with environment variables
    config_dict = _override_from_env(config_dict)
    
    # Convert to typed config
    config = SystemConfig.from_dict(config_dict)
    
    return config

def _update_dict(target: Dict[str, Any], source: Dict[str, Any]):
    """
    Recursively update a dictionary with values from another dictionary.
    
    Args:
        target: Dictionary to update
        source: Dictionary with new values
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _update_dict(target[key], value)
        else:
            target[key] = value

def _override_from_env(config: Dict[str, Any], prefix: str = "TEXTTOSQL") -> Dict[str, Any]:
    """
    Override configuration values from environment variables.
    
    Environment variables are expected in the format:
    {PREFIX}_{SECTION}_{KEY}
    
    For example, TEXTTOSQL_DATABASE_HOST would override config["database"]["host"]
    
    Args:
        config: Configuration dictionary to update
        prefix: Prefix for environment variables
        
    Returns:
        Updated configuration dictionary
    """
    # Create a copy of the config to avoid modifying the original
    config = config.copy()
    
    # Look for environment variables that match our pattern
    for env_var, env_value in os.environ.items():
        # Only process variables with the right prefix
        if env_var.startswith(f"{prefix}_"):
            # Remove prefix and split into parts
            parts = env_var[len(prefix) + 1:].lower().split('_')
            
            # Need at least section and key
            if len(parts) >= 2:
                section = parts[0]
                key = '_'.join(parts[1:])  # Join remaining parts as key
                
                # Ensure section exists in config
                if section not in config:
                    config[section] = {}
                
                # Convert value to appropriate type
                if env_value.lower() in ('true', 'yes', '1', 'y'):
                    typed_value = True
                elif env_value.lower() in ('false', 'no', '0', 'n'):
                    typed_value = False
                elif env_value.isdigit():
                    typed_value = int(env_value)
                elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') == 1:
                    typed_value = float(env_value)
                else:
                    typed_value = env_value
                
                # Set the value in the config
                config[section][key] = typed_value
                logger.debug(f"Overrode {section}.{key} from environment variable {env_var}")
    
    return config

def setup_logging(config: Union[Dict[str, Any], LoggingConfig, SystemConfig]):
    """
    Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary or object
    """
    # Extract logging config
    if isinstance(config, SystemConfig):
        logging_config = config.logging.to_dict()
    elif isinstance(config, LoggingConfig):
        logging_config = config.to_dict()
    elif isinstance(config, dict):
        logging_config = config.get("logging", {})
    else:
        logging_config = {}
    
    # Import here to avoid circular imports
    from text_to_sql.utils.logging import configure_logging
    configure_logging(logging_config)