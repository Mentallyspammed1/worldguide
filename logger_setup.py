
"""
Enhanced Logger Setup Module
"""
import logging
import os
from colorama import init, Fore, Style

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Add color to level name
        record.levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Setup a logger with both file and console output"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create logs directory if needed
    os.makedirs('bot_logs', exist_ok=True)
    
    # File handler with standard formatting
    fh = logging.FileHandler(os.path.join('bot_logs', log_file))
    fh.setLevel(level)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler with colors
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(ch_formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
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
