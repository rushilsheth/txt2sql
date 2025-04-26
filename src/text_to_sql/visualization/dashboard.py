"""
Dashboard Module

This module provides the main dashboard interface and layout for the text-to-SQL application.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr

from text_to_sql.db.base import DatabaseManager
from text_to_sql.db.postgres import PostgresDatabaseManager
from text_to_sql.llm.engine import LLMEngine
from text_to_sql.llm.semantic import SemanticEngine
from text_to_sql.utils.config_types import DatabaseConfig, LLMConfig, AppConfig, AgentConfig
from text_to_sql.visualization.app import TextToSQLApp
from text_to_sql.visualization.app_with_agents import AgentBasedTextToSQLApp

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Main dashboard for the text-to-SQL application.
    
    This class provides the entry point for launching the application
    and setting up all the necessary components.
    """
    
    def __init__(
        self,
        db_config: Union[Dict[str, Any], DatabaseConfig],
        llm_config: Union[Dict[str, Any], LLMConfig],
        use_agents: bool = False,
        use_semantic_engine: bool = True,
        debug_mode: bool = False,
        app_config: Optional[Union[Dict[str, Any], AppConfig]] = None,
        agent_config: Optional[Union[Dict[str, Any], AgentConfig]] = None
    ):
        """
        Initialize the dashboard.
        
        Args:
            db_config: Database configuration
            llm_config: LLM configuration
            use_agents: Whether to use the agent-based approach
            use_semantic_engine: Whether to use the semantic engine (for non-agent mode)
            debug_mode: Whether to enable debug mode
            app_config: Optional application configuration
            agent_config: Optional agent configuration (for agent mode)
        """
        # Convert dictionaries to typed configs if needed
        self.db_config = db_config if isinstance(db_config, DatabaseConfig) else DatabaseConfig.from_dict(db_config)
        self.llm_config = llm_config if isinstance(llm_config, LLMConfig) else LLMConfig.from_dict(llm_config)
        self.app_config = app_config if isinstance(app_config, AppConfig) else AppConfig() if app_config is None else AppConfig.from_dict(app_config)
        self.agent_config = agent_config if isinstance(agent_config, AgentConfig) else AgentConfig() if agent_config is None else AgentConfig.from_dict(agent_config)
        
        self.use_agents = use_agents
        self.use_semantic_engine = use_semantic_engine
        self.debug_mode = debug_mode
        
        # Initialize components
        self.db_manager = None
        self.llm_engine = None
        self.semantic_engine = None
        self.app = None
    
    def initialize(self):
        """Initialize all components of the dashboard."""
        # Initialize database manager
        db_type = self.db_config.type.lower()
        
        if db_type == "postgres":
            self.db_manager = PostgresDatabaseManager(self.db_config.get_connection_params())
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Connect to the database
        if not self.db_manager.connect():
            raise RuntimeError("Failed to connect to the database")
        
        # Initialize LLM engine
        self.llm_engine = LLMEngine(
            model=self.llm_config.model,
            api_key=self.llm_config.api_key,
            temperature=self.llm_config.temperature,
            timeout=self.llm_config.timeout,
            max_tokens=self.llm_config.max_tokens,
            db_manager=self.db_manager
        )
        
        # Initialize the appropriate app based on use_agents flag
        if self.use_agents:
            logger.info(
                "Initializing agent-based application",
                tags={"component": "dashboard", "action": "initialize", "mode": "agent"}
            )
            
            # Initialize agent-based app
            self.app = AgentBasedTextToSQLApp(
                db_manager=self.db_manager,
                llm_engine=self.llm_engine,
                agent_config=self.agent_config.to_dict(),
                debug_mode=self.debug_mode,
                app_config=self.app_config
            )
        else:
            logger.info(
                "Initializing standard application",
                tags={"component": "dashboard", "action": "initialize", "mode": "standard"}
            )
            
            # Initialize semantic engine if requested
            if self.use_semantic_engine:
                self.semantic_engine = SemanticEngine(
                    llm_engine=self.llm_engine,
                    db_manager=self.db_manager
                )
            
            # Initialize the standard app
            self.app = TextToSQLApp(
                db_manager=self.db_manager,
                llm_engine=self.llm_engine,
                semantic_engine=self.semantic_engine,
                debug_mode=self.debug_mode,
                theme=self.app_config.theme
            )
        
        # Build the app interface
        if hasattr(self.app, 'build_app'):
            self.app.build_app()
        
        return self
    
    def launch(self, **kwargs):
        """
        Launch the dashboard.
        
        Args:
            **kwargs: Keyword arguments to pass to gr.launch()
        """
        if self.app is None:
            self.initialize()
        
        # Merge app config with kwargs
        launch_kwargs = {
            "server_name": self.app_config.host,
            "server_port": self.app_config.port,
            "share": self.app_config.share
        }
        launch_kwargs.update(kwargs)
        
        logger.info(
            f"Launching application on {self.app_config.host}:{self.app_config.port}",
            tags={"component": "dashboard", "action": "launch", "share": self.app_config.share}
        )
        
        self.app.launch(**launch_kwargs)
    
    def __del__(self):
        """Clean up resources on object destruction."""
        if self.db_manager and self.db_manager.connected:
            logger.debug(
                "Disconnecting from database",
                tags={"component": "dashboard", "action": "disconnect"}
            )
            self.db_manager.disconnect()