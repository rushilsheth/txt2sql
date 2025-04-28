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
import shutil  # New import for cleanup
from pathlib import Path
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

from download_data import download_file, ADVENTUREWORKS_URL
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


def clone_repository(target_dir):
    repo_url = "https://github.com/morenoh149/postgresDBSamples.git"
    logger.info(f"Cloning repository from {repo_url} into {target_dir}...")
    try:
        subprocess.run(
            ["git", "clone", repo_url, target_dir],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to clone repository.")
        logger.error(e.stderr.decode())
        sys.exit(1)

import psycopg2

def data_already_loaded(db_config) -> bool:
    """
    Return True if AdventureWorks data is already present.

    We treat the presence of at least one row in person.address
    as a proxy for “the import completed successfully.”  Adjust
    the table if you prefer a different sentinel.
    """
    try:
        with psycopg2.connect(
            dbname=db_config.dbname,
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port
        ) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM person.address LIMIT 1;")
            return cur.fetchone() is not None
    except Exception:
        # If the table doesn’t exist yet, the query will error;
        # treat that the same as “not loaded”.
        return False

def cleanup_repository(target_dir):
    """Delete the target directory if it exists."""
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            logger.info(f"Successfully deleted {target_dir}.")
        except Exception as e:
            logger.error(f"Failed to delete {target_dir}: {e}")
    else:
        logger.info(f"{target_dir} does not exist, nothing to delete.")

def import_data(db_config: DatabaseConfig, data_path):
    try:
        command = [
            "psql",
            "-v", "ON_ERROR_STOP=1",
            "-h", db_config.host,
            "-p", str(db_config.port),
            "-U", db_config.user,
            "-d", db_config.dbname,
            "-f", data_path
        ]
        env = os.environ.copy()
        env["PGPASSWORD"] = db_config.password

        # Set cwd to the directory containing the SQL file so that relative CSV paths (e.g., data/...)
        # in install.sql resolve correctly.
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
    db_config = DatabaseConfig.from_dict(config.get("database", {}))
    
    # Point to the desired SQL file (after cloning, the structure is:
    # adventureworks/adventureworks/install.sql and CSV files in adventureworks/adventureworks/data)
    sql_file_path = os.path.abspath(os.path.join("adventureworks", "adventureworks", "install.sql"))
    
    # If the SQL file does not exist, clone the repository.
    if not os.path.exists(sql_file_path):
        logger.error(f"SQL file does not exist: {sql_file_path}.")
        clone_repository("adventureworks")
        if not os.path.exists(sql_file_path):
            logger.error(f"After cloning, SQL file still not found: {sql_file_path}")
            sys.exit(1)
    
    create_database(db_config)
    if data_already_loaded(db_config):
        logger.info("AdventureWorks data already present – skipping import.")
    else:
        import_data(db_config, sql_file_path)

    cleanup_repository("adventureworks")

if __name__ == "__main__":
    main()