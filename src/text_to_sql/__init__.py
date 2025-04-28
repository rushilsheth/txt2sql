"""
Text-to-SQL Package

This package provides a framework for handling natural language queries to SQL databases,
with visualization capabilities.
"""

# IMPORTANT: Set up logging before any other imports.
from text_to_sql.config import load_config, setup_logging
# Load a minimal configuration (or defaults) and configure logging immediately.
# (You can adjust 'None' to a default config dictionary if available)
_config = load_config(None)
setup_logging(_config)

import logging
import os
import sys
from typing import Any, Dict, Optional

from text_to_sql.db.base import DatabaseManager
from text_to_sql.db.postgres import PostgresDatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.llm.semantic import SemanticEngine
from text_to_sql.utils.config_types import SystemConfig
from text_to_sql.visualization.dashboard import Dashboard
from text_to_sql.agent.main import create_text_to_sql_agent

__version__ = "0.1.0"

def initialize_logging():
    """Initialize logging with basic configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def create_app(config_path: Optional[str] = None, use_agents: bool = True) -> Dashboard:
    """
    Create and initialize the application.
    
    Args:
        config_path: Path to the configuration file
        use_agents: Whether to use the agent-based approach
        
    Returns:
        Initialized Dashboard instance
    """
    # Load configuration
    config = load_config(config_path)
    
    # (Re)configure logging if needed (setup_logging has already been called early).
    setup_logging(config)
    
    # Get logger - now this logger uses our StructuredLogger
    logger = logging.getLogger(__name__)
    

    # Dynamically create the concrete DatabaseManager based on configuration.
    db_config = config.database
    
    # Log startup with structured data
    logger.info(
        f"Creating Text-to-SQL application",
        tags={"component": "main", "action": "create_app", "mode": "agent" if use_agents else "standard"},
        structured_data={"version": __version__}
    )
    # Create dashboard instance with appropriate configuration
    dashboard = Dashboard(
        db_config=db_config,
        llm_config=config.llm,
        app_config=config.app,
        agent_config=config.agent,
        use_agents=use_agents,
        use_semantic_engine=config.app.use_semantic_engine,
        debug_mode=config.app.debug_mode
    )
    
    # Initialize the dashboard
    dashboard.initialize()
    
    return dashboard

def run_app(config_path: Optional[str] = None, use_agents: bool = True):
    """
    Run the application with the given configuration.
    
    Args:
        config_path: Path to the configuration file
        use_agents: Whether to use the agent-based approach
    """
    # Load configuration and reconfigure (if needed)
    config = load_config(config_path)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create and initialize the application
        app = create_app(config_path, use_agents)
        
        # Launch the dashboard
        app.launch(
            server_name=config.app.host,
            server_port=config.app.port,
            share=config.app.share
        )
    except Exception as e:
        logger.exception(
            f"Error running application: {e}",
            tags={"component": "main", "action": "run_app_error", "error_type": type(e).__name__}
        )