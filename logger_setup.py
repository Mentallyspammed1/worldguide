"""
Logger Setup Module

This module provides a unified logging configuration for the trading bot.
It includes colorized console output and file logging with rotation.
"""

import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Ensure log directory exists
log_dir = "bot_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Colors for different log levels
LOG_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.RED + Style.BRIGHT
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on level"""
    
    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{Style.RESET_ALL}"
            record.msg = f"{LOG_COLORS[levelname]}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

def setup_logger(name, log_file=None, level=logging.INFO, format_string=None):
    """
    Set up a logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (relative to log_dir)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        logging.Logger: Configured logger
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    colored_formatter = ColoredFormatter(format_string)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file provided
    if log_file:
        file_path = os.path.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            file_path, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_default_loggers():
    """Configure the default loggers used by the trading bot"""
    # App-wide logger
    root_logger = setup_logger("trading_bot", "trading_bot.log")
    
    # Component-specific loggers
    setup_logger("trading_bot.strategies", "strategies.log")
    setup_logger("trading_bot.indicators", "indicators.log")
    setup_logger("trading_bot.orders", "orders.log")
    setup_logger("trading_bot.risk", "risk.log")
    setup_logger("trading_bot.web", "web.log")
    
    # Debug logger for detailed diagnostics
    debug_logger = setup_logger("trading_bot.debug", "debug.log", level=logging.DEBUG)
    
    return root_logger