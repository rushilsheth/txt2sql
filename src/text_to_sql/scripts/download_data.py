#!/usr/bin/env python
"""
Data Download Script

Downloads AdventureWorks database directly from morenoh149's postgresDBSamples repository.
"""

import argparse
import logging
import os
from urllib.request import urlretrieve

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_data")

# URL for AdventureWorks SQL file ready for PostgreSQL import
ADVENTUREWORKS_URL = "https://raw.githubusercontent.com/lorint/AdventureWorks-for-Postgres/master/install.sql"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download AdventureWorks sample database"
    )
    parser.add_argument(
        "--output",
        help="Output path for the downloaded SQL file",
        default=os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "adventureworks.sql"
        ))
    )
    parser.add_argument(
        "--force",
        help="Force download even if file exists",
        action="store_true"
    )
    return parser.parse_args()


def download_file(url, output_path, force=False):
    """
    Download a file from a URL.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        force: Force re-download even if the file exists

    Returns:
        Path to the downloaded file
    """
    if not force and os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path

    try:
        logger.info(f"Downloading {url} to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        urlretrieve(url, output_path)
        logger.info(f"Download completed successfully")
        return output_path

    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return None


def main():
    args = parse_args()
    download_file(ADVENTUREWORKS_URL, args.output, args.force)


if __name__ == "__main__":
    main()
