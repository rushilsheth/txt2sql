#!/usr/bin/env python
"""
Database Setup Script

This script sets up the database for the text-to-SQL application.
It downloads (if needed) and imports the AdventureWorks sample data.
"""

import argparse
import logging
import os
import subprocess
import sys
import yaml
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from download_data import download_file, ADVENTUREWORKS_URL
# Import DatabaseConfig instead of working with raw dictionary config
from text_to_sql.utils.config_types import DatabaseConfig

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
        default="config.yaml"
    )
    parser.add_argument(
        "--skip-download", help="Skip downloading the data", action="store_true"
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as file:
        # Read raw content and expand any environment variables
        config_raw = file.read()
        config_str = os.path.expandvars(config_raw)
    return yaml.safe_load(config_str)


def create_database(db_config: DatabaseConfig):
    try:
        conn = psycopg2.connect(
            dbname="postgres",  # connect to the default postgres database
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{db_config.dbname}'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {db_config.dbname}")
            logger.info(f"Database {db_config.dbname} created.")
        else:
            logger.info(f"Database {db_config.dbname} already exists.")

        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)


def import_data(db_config: DatabaseConfig, data_path):
    try:
        command = [
            "psql",
            "-h", db_config.host,
            "-p", str(db_config.port),
            "-U", db_config.user,
            "-d", db_config.dbname,
            "-f", data_path
        ]
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config.password

        # Set cwd to the directory containing the SQL file so any relative CSV references work
        cwd = os.path.dirname(data_path)

        result = subprocess.run(
            command,
            check=True,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Data imported successfully")
        logger.info(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to import data: {e}")
        logger.error(e.stderr.decode())
        sys.exit(1)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Use DatabaseConfig to parse the database section of the YAML config
    db_config = DatabaseConfig.from_dict(config.get("database", {}))
    # Convert relative data_path to an absolute path
    data_path = os.path.abspath(config.get("data_path", os.path.join("data", "adventureworks.sql")))
    
    if not args.skip_download:
        download_file(ADVENTUREWORKS_URL, data_path)

    if not os.path.exists(data_path):
        logger.error(f"SQL file does not exist: {data_path}")
        sys.exit(1)

    create_database(db_config)
    import_data(db_config, data_path)


if __name__ == "__main__":
    main()