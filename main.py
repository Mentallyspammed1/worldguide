
"""
Main entry point for Pyrmethus Trading Bot
"""

import logging
import os
from colorama import init, Fore, Style
from flask_socketio import SocketIO

from app import app
from logger_setup import setup_default_loggers
from trading_bot import TradingBot

# Initialize colorama
init(autoreset=True)

# Set up logging
app_logger, bot_logger, views_logger, utils_logger = setup_default_loggers()
logger = app_logger 

# Log app startup
logger.info(f"{Fore.CYAN}Pyrmethus Trading Bot starting up...{Style.RESET_ALL}")

# Import after logging setup
from app import socketio

# Create database tables
with app.app_context():
    from app import db
    import models
    
    try:
        db.create_all()
        logger.info(f"{Fore.GREEN}Database tables created successfully{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"{Fore.RED}Error creating database tables: {e}{Style.RESET_ALL}")

# Initialize trading bot
try:
    bot = TradingBot(config_file="config.json")
    app.bot = bot
except Exception as e:
    logger.error(f"{Fore.RED}Failed to initialize trading bot: {e}{Style.RESET_ALL}")
    bot = None

if __name__ == "__main__":
    try:
        socketio.run(
            app, 
            host="0.0.0.0",
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
    except Exception as e:
        logger.error(f"{Fore.RED}Failed to start server: {e}{Style.RESET_ALL}")
