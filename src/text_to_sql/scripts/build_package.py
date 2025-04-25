#!/usr/bin/env python
"""
Package Build Script

This script builds the Python package for distribution.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("build_package")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build the Python package")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean build directories before building"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests before building"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to PyPI after building"
    )
    parser.add_argument(
        "--repository",
        help="Repository to upload to (default: pypi)",
        default="pypi"
    )
    
    return parser.parse_args()

def run_command(command, cwd=None):
    """
    Run a shell command.
    
    Args:
        command: Command to run
        cwd: Working directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Running command: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd
        )
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {e}")
        return False

def clean_build_dirs():
    """Clean build directories."""
    logger.info("Cleaning build directories")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Remove build directories
    dirs_to_remove = [
        root_dir / "build",
        root_dir / "dist",
        root_dir / "*.egg-info"
    ]
    
    for path in dirs_to_remove:
        # Use glob to handle wildcards
        for p in root_dir.glob(str(path.name)):
            if p.is_dir():
                logger.info(f"Removing directory: {p}")
                shutil.rmtree(p)
    
    # Remove __pycache__ directories
    for path in root_dir.glob("**/__pycache__"):
        if path.is_dir():
            logger.info(f"Removing directory: {path}")
            shutil.rmtree(path)
    
    return True

def run_tests():
    """Run tests."""
    logger.info("Running tests")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Run pytest
    return run_command("poetry run pytest", cwd=root_dir)

def build_package():
    """Build the package."""
    logger.info("Building package")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Build the package
    return run_command("poetry build", cwd=root_dir)

def upload_package(repository):
    """
    Upload the package to PyPI.
    
    Args:
        repository: Repository to upload to
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Uploading package to {repository}")
    
    # Get the project root directory
    root_dir = Path(__file__).resolve().parent.parent
    
    # Upload the package
    return run_command(f"poetry publish --repository {repository}", cwd=root_dir)

def main():
    """Main function."""
    args = parse_args()
    
    # Clean build directories if requested
    if args.clean:
        if not clean_build_dirs():
            return 1
    
    # Run tests if requested
    if args.test:
        if not run_tests():
            logger.error("Tests failed, aborting build")
            return 1
    
    # Build the package
    if not build_package():
        logger.error("Build failed")
        return 1
    
    # Upload the package if requested
    if args.upload:
        if not upload_package(args.repository):
            logger.error("Upload failed")
            return 1
    
    logger.info("Build completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())