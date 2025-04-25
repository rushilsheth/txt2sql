#!/usr/bin/env python
"""
Data Download Script

This script downloads the AdventureWorks sample database for PostgreSQL.
"""

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("download_data")

# URL for AdventureWorks PostgreSQL version
ADVENTUREWORKS_URL = "https://github.com/lorint/AdventureWorks-for-Postgres/archive/refs/heads/master.zip"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download AdventureWorks sample data")
    parser.add_argument(
        "--output", 
        help="Output path for the downloaded file",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "adventureworks.sql"
        )
    )
    parser.add_argument(
        "--force", 
        help="Force download even if file exists",
        action="store_true"
    )
    
    return parser.parse_args()

def download_file(url, output_path):
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        
    Returns:
        Path to the downloaded file
    """
    try:
        logger.info(f"Downloading {url} to {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download file
        urlretrieve(url, output_path)
        
        logger.info(f"Download completed: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return None

def extract_zip(zip_path, extract_dir):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to the zip file
        extract_dir: Directory to extract to
        
    Returns:
        Path to the extracted directory
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Return the path to the extracted directory
        # (assumes zip contains a single top-level directory)
        extracted_dirs = [f.path for f in os.scandir(extract_dir) if f.is_dir()]
        if not extracted_dirs:
            logger.error("No directories found in extracted zip")
            return None
        
        return extracted_dirs[0]
        
    except Exception as e:
        logger.error(f"Error extracting zip: {e}")
        return None

def find_sql_file(directory):
    """
    Find the main SQL file in the AdventureWorks repository.
    
    Args:
        directory: Directory to search in
        
    Returns:
        Path to the SQL file
    """
    try:
        # Look for the main SQL file
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".sql") and ("adventure" in file.lower() or "install" in file.lower()):
                    return os.path.join(root, file)
        
        logger.error("SQL file not found in extracted directory")
        return None
        
    except Exception as e:
        logger.error(f"Error finding SQL file: {e}")
        return None

def download_adventureworks_data(output_path, force=False):
    """
    Download and prepare the AdventureWorks sample data.
    
    Args:
        output_path: Path to save the final SQL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if file exists
        if os.path.exists(output_path) and not force:
            logger.info(f"Output file already exists: {output_path}")
            return True
        
        # Create temporary directory
        temp_dir = os.path.join(os.path.dirname(output_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download zip file
        zip_path = os.path.join(temp_dir, "adventureworks.zip")
        if not download_file(ADVENTUREWORKS_URL, zip_path):
            return False
        
        # Extract zip file
        extracted_dir = extract_zip(zip_path, temp_dir)
        if not extracted_dir:
            return False
        
        # Find SQL file
        sql_file = find_sql_file(extracted_dir)
        if not sql_file:
            return False
        
        # Copy SQL file to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(sql_file, "r") as src, open(output_path, "w") as dst:
            dst.write(src.read())
        
        logger.info(f"AdventureWorks SQL file saved to {output_path}")
        
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            logger.warning(f"Failed to clean up temporary directory: {temp_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading AdventureWorks data: {e}")
        return False

def main():
    """Main function to download the AdventureWorks data."""
    args = parse_args()
    
    if download_adventureworks_data(args.output, args.force):
        logger.info("Download completed successfully")
        return 0
    else:
        logger.error("Download failed")
        return 1

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main())