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
import time
import signal
from datetime import datetime
from logging.handlers import RotatingFileHandler
from threading import Thread, Event
from typing import Dict, List, Optional

# Export the Flask application for Gunicorn
# This is the variable that Gunicorn looks for by default
from app import app

# Configure basic logging
os.makedirs("bot_logs", exist_ok=True)

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

# Global stop event for graceful shutdown
stop_event = Event()


def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shutdown bot threads"""
    logger.info("Shutdown signal received, stopping all threads...")
    stop_event.set()
    
    # Wait a bit for threads to clean up
    time.sleep(1)
    logger.info("Exiting main process")
    sys.exit(0)


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Bot")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--web", action="store_true", help="Start web interface")
    mode_group.add_argument("--bot", action="store_true", help="Start trading bot")
    mode_group.add_argument("--backtest", action="store_true", help="Run backtest")
    mode_group.add_argument("--validate", action="store_true", help="Validate configuration only")
    
    # General options
    parser.add_argument("-c", "--config", type=str, default="config.json", 
                        help="Configuration file path")
    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Enable debug logging")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Increase output verbosity")
    
    # Bot options
    parser.add_argument("-s", "--symbol", type=str, 
                        help="Trading symbol (e.g., BTC/USDT:USDT)")
    parser.add_argument("-e", "--exchange", type=str, 
                        help="Exchange name (e.g., bybit)")
    parser.add_argument("-t", "--timeframe", type=str, 
                        help="Trading timeframe (e.g., 15m)")
    parser.add_argument("--strategy", type=str, 
                        help="Strategy name (e.g., ehlers_supertrend)")
    
    # Backtest options
    parser.add_argument("--start-date", type=str, 
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, 
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--initial-balance", type=float, 
                        help="Initial balance for backtest")
    
    return parser.parse_args()


def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        "bot_logs",
        "data",
        "backtest_results",
        "models",
        "strategies"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def run_trading_bot(args):
    """Run the trading bot in a separate thread"""
    from trading_bot import TradingBot
    
    try:
        # Create and configure bot instance
        bot = TradingBot(config_file=args.config)
        
        # Override config with command-line arguments if provided
        if args.symbol:
            bot.set_symbol(args.symbol)
        if args.exchange:
            bot.set_exchange(args.exchange)
        if args.timeframe:
            bot.set_timeframe(args.timeframe)
        if args.strategy:
            bot.set_strategy(args.strategy)
        
        # Start the bot's main loop
        logger.info(f"Starting trading bot with symbol: {bot.symbol}, exchange: {bot.exchange_id}")
        bot.run(stop_event=stop_event)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.exception(f"Critical error in trading bot: {e}")
        # Don't exit the entire application if web is running
        if not args.web:
            sys.exit(1)


def run_backtest(args):
    """Run backtesting with the specified parameters"""
    from trading_bot import TradingBot
    from backtesting import Backtester
    
    try:
        # Create and configure bot instance
        bot = TradingBot(config_file=args.config)
        
        # Override config with command-line arguments if provided
        if args.symbol:
            bot.set_symbol(args.symbol)
        if args.exchange:
            bot.set_exchange(args.exchange)
        if args.timeframe:
            bot.set_timeframe(args.timeframe)
        if args.strategy:
            bot.set_strategy(args.strategy)
        
        # Parse dates
        start_date = args.start_date or bot.config.get("backtest", {}).get("start_date")
        end_date = args.end_date or bot.config.get("backtest", {}).get("end_date")
        initial_balance = args.initial_balance or bot.config.get("backtest", {}).get("initial_balance", 10000.0)
        
        # Run backtest
        backtester = Backtester(bot)
        result = backtester.run(
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance
        )
        
        # Print results summary
        logger.info(f"Backtest completed with {result.total_trades} trades")
        logger.info(f"Final balance: {result.final_balance:.2f} (ROI: {result.roi:.2f}%)")
        logger.info(f"Win rate: {result.win_rate:.2f}% (Wins: {result.profitable_trades}, Losses: {result.losing_trades})")
        logger.info(f"Profit factor: {result.profit_factor:.2f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2f}%")
        
        # Save results
        result_file = f"backtest_results/{bot.symbol.replace('/', '_')}_{bot.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result.save_to_file(result_file)
        logger.info(f"Backtest results saved to {result_file}")
        
        # Generate and save charts
        charts_file = result_file.replace(".json", "_charts.png")
        result.plot_equity_curve(charts_file)
        logger.info(f"Backtest charts saved to {charts_file}")
    
    except Exception as e:
        logger.exception(f"Error in backtest: {e}")
        sys.exit(1)


def validate_config(args):
    """Validate configuration without running the bot"""
    from trading_bot import TradingBot
    
    try:
        # Create and configure bot instance
        bot = TradingBot(config_file=args.config, validate_only=True)
        
        # Perform validation
        validation_result = bot.validate_config()
        
        if validation_result["valid"]:
            logger.info("Configuration validation successful!")
            for key, value in validation_result["details"].items():
                logger.info(f"  - {key}: {value}")
        else:
            logger.error("Configuration validation failed!")
            for error in validation_result["errors"]:
                logger.error(f"  - {error}")
            sys.exit(1)
    
    except Exception as e:
        logger.exception(f"Error validating configuration: {e}")
        sys.exit(1)


def main():
    """Main entry point for the trading bot application"""
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Ensure required directories exist
    ensure_directories()
    
    # Set log level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif args.verbose:
        # Set to INFO level for main components
        for component in ["main", "trading_bot", "strategies", "indicators", "risk_management"]:
            logging.getLogger(component).setLevel(logging.INFO)
    
    # Determine mode of operation
    if args.web:
        # Start web interface and optionally trading bot in the background
        logger.info("Starting web interface")
        
        # Start trading bot in a separate thread if requested
        if args.bot:
            logger.info("Also starting trading bot in background")
            bot_thread = Thread(target=run_trading_bot, args=(args,), daemon=True)
            bot_thread.start()
        
        # Run web interface
        app.run(host="0.0.0.0", port=5000, debug=args.debug)
    
    elif args.backtest:
        # Run backtesting
        logger.info("Running in backtest mode")
        run_backtest(args)
    
    elif args.validate:
        # Validate configuration only
        logger.info("Validating configuration")
        validate_config(args)
    
    else:
        # Default to running the trading bot
        logger.info("Starting trading bot")
        run_trading_bot(args)


if __name__ == "__main__":
    main()