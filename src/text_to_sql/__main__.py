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
from text_to_sql.config import load_config

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