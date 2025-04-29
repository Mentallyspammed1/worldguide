
"""
Logging configuration module
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Configure logger with file and console handlers with color"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('bot_logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_formatter = logging.Formatter(
        f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - '
        f'{Fore.GREEN}%(name)s{Style.RESET_ALL} - '
        f'{Fore.YELLOW}%(levelname)s{Style.RESET_ALL} - '
        f'{Fore.WHITE}%(message)s{Style.RESET_ALL}'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join('bot_logs', log_file),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_default_loggers():
    """Configure the default loggers used by the trading bot"""
    # App-wide logger
    app_logger = setup_logger('app', 'app.log')
    
    # Trading bot logger
    bot_logger = setup_logger('trading_bot', 'trading_bot.log')
    
    # Views logger 
    views_logger = setup_logger('views', 'app.log')
    
    # Utils logger
    utils_logger = setup_logger('utils', 'trading_bot.log')
    
    return app_logger, bot_logger, views_logger, utils_logger
