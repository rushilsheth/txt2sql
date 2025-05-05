#!/usr/bin/env python
"""
Command-line Interface for Text-to-SQL Application

This module provides a command-line interface for the text-to-SQL application.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import text_to_sql
from text_to_sql import run_app
from text_to_sql.config import load_config, setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL: Natural Language Database Interface"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file",
        default=os.environ.get("TEXTTOSQL_CONFIG", None)
    )
    
    parser.add_argument(
        "--standard", 
        action="store_true",
        help="Use standard (non-agent) approach"
    )
    
    parser.add_argument(
        "--dynamic", 
        action="store_true",
        help="Use dynamic agent coordinator (requires OpenAI API key)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--router",
        action="store_true",
        help="Launch the Gradio UI for LLM routing of queries"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version=f"Text-to-SQL v{text_to_sql.__version__}"
    )
    
    return parser.parse_args()

def main():
    """Main function for the command-line interface."""
    # Initialize basic logging
    text_to_sql.initialize_logging()
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Set debug mode in the environment if specified
        if args.debug:
            os.environ["TEXTTOSQL_APP_DEBUG_MODE"] = "true"
        
        if args.router:
            # Launch the router-enabled Gradio app.
            from text_to_sql.visualization.app_router import RouterBasedTextToSQLApp
            from text_to_sql.db.postgres import PostgresDatabaseManager
            from text_to_sql.llm.engine import LLMEngine
            
            config = load_config(args.config)
            setup_logging(config)
            
            # Instantiate your database manager
            db_manager = PostgresDatabaseManager(config.database.get_connection_params())
            llm_config = config.llm
            if not db_manager.connect():
                logger.error("Failed to connect to the database.")
                sys.exit(1)
            llm_engine = LLMEngine(model=llm_config.model,
                        api_key=llm_config.api_key,
                        temperature=llm_config.temperature,
                        timeout=llm_config.timeout,
                        max_tokens=llm_config.max_tokens,
                        db_manager=db_manager)
            
            app_config = config.app
            agent_config = config.agent
            app = RouterBasedTextToSQLApp(db_manager, llm_engine, agent_config, app_config, debug_mode=args.debug)
            app.build_app()
            app.launch(
                server_name=config.app.host,
                server_port=config.app.port,
                share=config.app.share
            )
            return 0

        # Set dynamic coordinator if specified
        if args.dynamic:
            os.environ["TEXTTOSQL_AGENT_USE_DYNAMIC_COORDINATOR"] = "true"
        
        # Run the application
        run_app(
            config_path=args.config,
            use_agents=not args.standard
        )
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        return 0
        
    except Exception as e:
        logger.error(f"Error running application: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())