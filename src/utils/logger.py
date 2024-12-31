"""
Logger utility for the Prophet Forecaster application.
Handles all logging operations with proper formatting and file output.
"""

import os
import logging
from datetime import datetime
from typing import Optional
from .config_loader import load_config, get_setting

def setup_logger(name: str = "prophet_forecaster") -> logging.Logger:
    """
    Set up and configure the application logger.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Load logging configuration
    config = load_config()
    log_config = get_setting(config, "logging", default={})
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(get_setting(log_config, "level", default="INFO"))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        get_setting(log_config, "format", 
                   default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = get_setting(log_config, "file", default="logs/prophet_forecaster.log")
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "prophet_forecaster") -> logging.Logger:
    """
    Get or create a logger instance.
    
    Args:
        name (str): Name of the logger
    
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only setup if no handlers exist
        logger = setup_logger(name)
    return logger

# Global logger instance
logger = get_logger()

# Convenience methods
def debug(msg: str, *args, **kwargs):
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    """Log an info message."""
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """Log an error message."""
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """Log a critical message."""
    logger.critical(msg, *args, **kwargs)

def exception(msg: str, *args, **kwargs):
    """Log an exception message."""
    logger.exception(msg, *args, **kwargs) 