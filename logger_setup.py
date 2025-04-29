"""
Logger Setup Module

This module configures the logging for the application.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file (default: True)
    """
    # Create logs directory if it doesn't exist
    log_dir = "bot_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        file_handler = RotatingFileHandler(
            f"{log_dir}/bot.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Set specific loggers to lower levels for debugging
    logging.getLogger('app').setLevel(log_level)
    logging.getLogger('flask.app').setLevel(log_level)
    
    # Suppress chatty third-party logs
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    return root_logger