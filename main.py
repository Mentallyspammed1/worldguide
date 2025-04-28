#!/usr/bin/env python3
"""
Cryptocurrency Trading Bot Main Entry Point

This script serves as the entry point for the trading bot application,
loading configurations and starting either the bot or web interface based
on command-line arguments.
"""

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from trading_bot import TradingBot
from web_interface import app as flask_app

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            "bot_logs/trading_bot.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("main")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    parser.add_argument(
        "--config", default="config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--web", action="store_true", help="Start the web interface"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run in backtest mode"
    )
    parser.add_argument(
        "--symbol", help="Override trading symbol from config"
    )
    parser.add_argument(
        "--exchange", help="Override exchange from config"
    )
    return parser.parse_args()


def ensure_directories():
    """Ensure required directories exist"""
    directories = ["bot_logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def main():
    """Main entry point for the trading bot application"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Ensure required directories exist
    ensure_directories()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    if args.web:
        # Start web interface
        logger.info("Starting web interface")
        flask_app.run(host="0.0.0.0", port=5000, debug=args.debug)
    else:
        # Start trading bot
        logger.info("Starting trading bot")
        try:
            bot = TradingBot(args.config)
            # Override config with command-line arguments if provided
            if args.symbol:
                bot.config["symbol"] = args.symbol
            if args.exchange:
                bot.config["exchange"] = args.exchange
            
            if args.backtest:
                logger.info("Running in backtest mode")
                bot.run_backtest()
            else:
                bot.run()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.exception(f"Critical error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()