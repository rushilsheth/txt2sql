"""
Environment Validation Module

This module provides utilities for validating environment variables and other
prerequisites for the text-to-SQL system.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Required environment variables for different features
REQUIRED_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "postgres": ["TEXTTOSQL_DATABASE_HOST", "TEXTTOSQL_DATABASE_USER"],
    "dynamic_coordinator": ["OPENAI_API_KEY"]
}

def validate_environment(features: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate that required environment variables are set.
    
    Args:
        features: List of features to validate. If None, validates all features.
        
    Returns:
        Tuple containing:
        - Boolean indicating if all required variables are set
        - List of missing variables
    """
    features = features or list(REQUIRED_ENV_VARS.keys())
    
    all_valid = True
    missing_vars = []
    
    for feature in features:
        if feature in REQUIRED_ENV_VARS:
            for var in REQUIRED_ENV_VARS[feature]:
                if not os.environ.get(var):
                    all_valid = False
                    missing_vars.append((var, feature))
    
    # Log warnings for missing variables
    for var, feature in missing_vars:
        logger.warning(f"Environment variable {var} required for {feature} is not set")
    
    return all_valid, [var for var, _ in missing_vars]

def validate_postgres_connection() -> bool:
    """
    Validate that PostgreSQL connection can be established.
    
    Returns:
        Boolean indicating if connection is valid
    """
    try:
        import psycopg2
        
        # Get connection parameters from environment
        host = os.environ.get("TEXTTOSQL_DATABASE_HOST", "localhost")
        port = os.environ.get("TEXTTOSQL_DATABASE_PORT", "5432")
        dbname = os.environ.get("TEXTTOSQL_DATABASE_DBNAME", "adventureworks")
        user = os.environ.get("TEXTTOSQL_DATABASE_USER", "postgres")
        password = os.environ.get("TEXTTOSQL_DATABASE_PASSWORD", "")
        
        # Try to connect
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        
        # Close connection
        conn.close()
        
        logger.info(f"Successfully connected to PostgreSQL at {host}:{port}/{dbname}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return False

def validate_openai_api_key() -> bool:
    """
    Validate that OpenAI API key is set and valid.
    
    Returns:
        Boolean indicating if API key is valid
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return False
    
    # Validate API key format (basic check)
    if not api_key.startswith("sk-") or len(api_key) < 20:
        logger.warning("OPENAI_API_KEY does not appear to be valid (should start with 'sk-')")
        return False
    
    # For a more thorough check, we would need to make an API call
    try:
        import openai
        
        # Set the API key
        openai.api_key = api_key
        
        # Make a simple call to verify the key
        response = openai.models.list()
        
        logger.info("Successfully validated OpenAI API key")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate OpenAI API key: {e}")
        return False

def validate_requirements_for_mode(mode: str) -> bool:
    """
    Validate requirements for a specific application mode.
    
    Args:
        mode: Application mode ("standard", "simple", "dynamic")
        
    Returns:
        Boolean indicating if requirements are met
    """
    # Database connection is required for all modes
    if not validate_postgres_connection():
        logger.error("Database connection validation failed, required for all modes")
        return False
    
    # OpenAI API key is required for agent-based modes
    if mode in ["simple", "dynamic"]:
        if not validate_environment(["openai"])[0]:
            logger.warning("OpenAI API key validation failed, required for agent-based modes")
            # Continue anyway, as this is just a warning
    
    # Dynamic coordinator has additional requirements
    if mode == "dynamic":
        if not validate_environment(["dynamic_coordinator"])[0]:
            logger.error("Environment validation failed for dynamic coordinator mode")
            return False
        
        if not validate_openai_api_key():
            logger.error("OpenAI API key validation failed, required for dynamic coordinator")
            return False
    
    return True

def check_and_warn_about_missing_features(config: Dict) -> None:
    """
    Check configuration for features that may require environment variables.
    
    Args:
        config: Application configuration
    """
    # Check for dynamic coordinator
    if config.get("agent", {}).get("use_dynamic_coordinator", False):
        if not validate_environment(["dynamic_coordinator"])[0]:
            logger.warning(
                "Dynamic coordinator is enabled in config but environment "
                "variables are not properly set"
            )
    
    # Check for OpenAI API usage
    if config.get("llm", {}).get("model", "").startswith("gpt-"):
        if not validate_environment(["openai"])[0]:
            logger.warning(
                "OpenAI model specified in config but OPENAI_API_KEY "
                "environment variable is not set"
            )