#!/usr/bin/env python
"""
Wrapper script to run the Prophet Forecaster interactive CLI.
"""

from src.main_active import main
import logging
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

if __name__ == "__main__":
    logger.info("=== Starting Prophet Forecaster Interactive CLI ===")
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        logger.info("=== Prophet Forecaster Session Ended ===") 