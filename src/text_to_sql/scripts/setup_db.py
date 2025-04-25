#!/usr/bin/env python
"""
Database Setup Script

This script sets up the database for the text-to-SQL application.
It creates the database if it doesn't exist and restores the AdventureWorks sample data.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from text_to_sql.config import load_config, setup_logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("setup_db")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up the AdventureWorks database")
    parser.add_argument(
        "--config", 
        help="Path to configuration file",
        default=os.environ.get("TEXTTOSQL_CONFIG", None)
    )
    parser.add_argument(
        "--host", 
        help="Database host",
        default=None
    )
    parser.add_argument(
        "--port", 
        help="Database port",
        type=int,
        default=None
    )
    parser.add_argument(
        "--user", 
        help="Database user",
        default=None
    )
    parser.add_argument(
        "--password", 
        help="Database password",
        default=None
    )
    parser.add_argument(
        "--database", 
        help="Database name",
        default=None
    )
    parser.add_argument(
        "--data-path", 
        help="Path to the AdventureWorks SQL file",
        default=None
    )
    parser.add_argument(
        "--skip-download", 
        help="Skip downloading the data",
        action="store_true"
    )
    parser.add_argument(
        "--force", 
        help="Force database recreation",
        action="store_true"
    )
    
    return parser.parse_args()

def create_database(config):
    """
    Create the database if it doesn't exist.
    
    Args:
        config: Database configuration
        
    Returns:
        True if database was created or already exists, False otherwise
    """
    db_host = config.host
    db_port = config.port
    db_user = config.user
    db_password = config.password
    db_name = config.dbname
    
    # Connect to the default database to create a new database
    try:
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if exists:
            logger.info(f"Database '{db_name}' already exists")
            
            # Drop if force flag is set
            if getattr(config, "force", False):
                logger.info(f"Force flag set, dropping database '{db_name}'")
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                logger.info(f"Database '{db_name}' dropped")
                exists = False
        
        if not exists:
            # Create database
            cursor.execute(f"CREATE DATABASE {db_name}")
            logger.info(f"Database '{db_name}' created")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False

def load_adventureworks_data(config):
    """
    Load the AdventureWorks sample data into the database.
    
    Args:
        config: Database configuration
        
    Returns:
        True if data was loaded successfully, False otherwise
    """
    db_host = config.host
    db_port = config.port
    db_user = config.user
    db_password = config.password
    db_name = config.dbname
    data_path = config.data_path
    
    if not data_path or not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        return False
    
    try:
        # Set up environment variables for pg_restore
        env = os.environ.copy()
        if db_password:
            env["PGPASSWORD"] = db_password
        
        # Check if we're dealing with a SQL file or a dump file
        if data_path.endswith(".sql"):
            # Use psql for SQL files
            cmd = [
                "psql",
                f"-h{db_host}",
                f"-p{db_port}",
                f"-U{db_user}",
                f"-d{db_name}",
                "-f", data_path
            ]
            logger.info(f"Loading AdventureWorks data using psql: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        else:
            # Use pg_restore for dump files
            cmd = [
                "pg_restore",
                f"-h{db_host}",
                f"-p{db_port}",
                f"-U{db_user}",
                f"-d{db_name}",
                "-v", data_path
            ]
            logger.info(f"Loading AdventureWorks data using pg_restore: {' '.join(cmd)}")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error loading data: {result.stderr}")
            return False
        
        logger.info("AdventureWorks data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

def main():
    """Main function to set up the database."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    # Override config with command line arguments
    db_config = config.database
    if args.host:
        db_config.host = args.host
    if args.port:
        db_config.port = args.port
    if args.user:
        db_config.user = args.user
    if args.password:
        db_config.password = args.password
    if args.database:
        db_config.dbname = args.database
    
    # Add command line arguments to config
    db_config.force = args.force
    
    # Download data if needed
    if not args.skip_download:
        from download_data import download_adventureworks_data
        data_path = args.data_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "adventureworks.sql"
        )
        if not download_adventureworks_data(data_path):
            logger.error("Failed to download AdventureWorks data")
            return 1
        db_config.data_path = data_path
    elif args.data_path:
        db_config.data_path = args.data_path
    else:
        logger.error("No data path provided and download skipped")
        return 1
    
    # Create database
    if not create_database(db_config):
        logger.error("Failed to create database")
        return 1
    
    # Load data
    if not load_adventureworks_data(db_config):
        logger.error("Failed to load AdventureWorks data")
        return 1
    
    logger.info("Database setup completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())