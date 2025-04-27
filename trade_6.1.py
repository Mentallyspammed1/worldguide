


2025-04-25 17:57:47,927 - ScalpingBotV4 - ERROR [MainThread] - Error applying precision or checking limits for order size: bybit amount of DOT/USDT:USDT must be greater than minimum amount precision of 0.1
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/trade_6.1.py", line 1551, in calculate_order_size
    amount_precise = self.exchange.amount_to_precision(self.symbol, order_size_base)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/base/exchange.py", line 5518, in amount_to_precision
    raise InvalidOrder(self.id + ' amount of ' + market['symbol'] + ' must be greater than minimum amount precision of ' + self.number_to_string(market['precision']['amount']))
ccxt.base.errors.InvalidOrder: bybit amount of DOT/USDT:USDT must be greater than minimum amount precision of 0.1
2025-04-25 17:57:47,932 - ScalpingBotV4 - WARNING [MainThread] - Cannot evaluate entry signal: Calculated order size is zero (check balance, order size %, or min limits).
2025-04-25 17:57:56,334 - ScalpingBotV4 - INFO [MainThread] -
===== Iteration 20 ==== Timestamp: 2025-04-25T22:57:56+00:00
2025-04-25 17:57:57,478 - ScalpingBotV4 - INFO [MainThread] - ===== Iteration 20 ==== Current Price: 4.2873 OB Imbalance: 1.172
2025-04-25 17:57:57,507 - ScalpingBotV4 - INFO [MainThread] - Indicators (calc took 0.03s): EMA=4.2879, RSI=60.3, ATR=0.0066, MACD(L/S)=(0.0200/-0.0001), Stoch(K/D)=(44.0/71.9)
2025-04-25 17:57:57,890 - ScalpingBotV4 - INFO [MainThread] - Fetched available balance: 10.2844 USDT
2025-04-25 17:57:57,894 - ScalpingBotV4 - ERROR [MainThread] - Error applying precision or checking limits for order size: bybit amount of DOT/USDT:USDT must be greater than minimum amount precision of 0.1
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/trade_6.1.py", line 1551, in calculate_order_size
    amount_precise = self.exchange.amount_to_precision(self.symbol, order_size_base)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.12/site-packages/ccxt/base/exchange.py", line 5518, in amount_to_precision
    raise InvalidOrder(self.id + ' amount of ' + market['symbol'] + ' must be greater than minimum amount precision of ' + self.number_to_string(market['precision']['amount']))
ccxt.base.errors.InvalidOrder: bybit amount of DOT/USDT:USDT must be greater than minimum amount precision of 0.1
2025-04-25 17:57:57,911 - ScalpingBotV4 - WARNING [MainThread] - Cannot evaluate entry signal: Calculated order size is zero (check balance, order size %, or min limits).
# -*- coding: utf-8 -*-
"""
Scalping Bot v4 - Pyrmethus Enhanced Edition (Bybit V5 Optimized)

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based Stop Loss (SL) and Take Profit (TP)
handling.

Key Enhancements V4 (Compared to a hypothetical V3):
- Fixed critical error in price formatting using market precision (ValueError).
- Calculated and stored integer decimal places for amount/price during init.
- Used pre-calculated decimal places for consistent f-string formatting.
- Solidified Bybit V5 parameter-based SL/TP in create_order ('stopLoss', 'takeProfit').
- Enhanced Trailing Stop Loss (TSL) logic using edit_order (Marked EXPERIMENTAL,
  requires careful verification with specific CCXT version and exchange behavior).
- Added `enable_trailing_stop_loss` configuration flag.
- Added `sl_trigger_by`, `tp_trigger_by` configuration options for Bybit V5.
- Added explicit `testnet_mode` configuration flag for clarity.
- Added `close_positions_on_exit` configuration flag.
- Robust persistent state file handling with automated backup on corruption and atomic writes.
- Improved API retry logic with more specific error handling.
- Improved error handling, logging verbosity, type hinting, and overall code clarity.
- More detailed validation of configuration parameters.
- Refined PnL calculation and logging.
- Enhanced shutdown procedure.
"""

import logging
import os
import sys
import time
import json
import shutil # For state file backup and management
import math   # For potential decimal place calculation
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ccxt
# import ccxt.async_support as ccxt_async # Retained for potential future asynchronous implementation
import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# --- Arcane Constants & Configuration Defaults ---
CONFIG_FILE_NAME: str = "config.yaml"
STATE_FILE_NAME: str = "scalping_bot_state.json"
LOG_FILE_NAME: str = "scalping_bot_v4.log"
DEFAULT_EXCHANGE_ID: str = "bybit"
DEFAULT_TIMEFRAME: str = "1m"
DEFAULT_RETRY_MAX: int = 3
DEFAULT_RETRY_DELAY_SECONDS: int = 3
DEFAULT_SLEEP_INTERVAL_SECONDS: int = 10
STRONG_SIGNAL_THRESHOLD_ABS: int = 3 # Absolute score threshold for strong signal adjustment
ENTRY_SIGNAL_THRESHOLD_ABS: int = 2  # Absolute score threshold to trigger entry
ATR_MULTIPLIER_SL: float = 2.0       # Default ATR multiplier for Stop Loss
ATR_MULTIPLIER_TP: float = 3.0       # Default ATR multiplier for Take Profit
DEFAULT_PRICE_DECIMALS: int = 4      # Fallback price decimal places
DEFAULT_AMOUNT_DECIMALS: int = 8     # Fallback amount decimal places


# Position Status Constants
STATUS_PENDING_ENTRY: str = 'pending_entry' # Order placed but not yet filled
STATUS_ACTIVE: str = 'active'           # Order filled, position is live
STATUS_CLOSING: str = 'closing'         # Manual close initiated (optional use)
STATUS_CANCELED: str = 'canceled'       # Order explicitly cancelled
STATUS_UNKNOWN: str = 'unknown'         # Order status cannot be determined

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Centralized Logger Setup ---
logger = logging.getLogger("ScalpingBotV4")
logger.setLevel(logging.DEBUG) # Set logger to lowest level to allow handlers to control verbosity
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(threadName)s] - %(message)s" # Added brackets for clarity
)

# Console Handler (INFO level by default, configurable)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Default console level
logger.addHandler(console_handler)

# File Handler (DEBUG level for detailed logs)
file_handler: Optional[logging.FileHandler] = None
try:
    # Ensure log directory exists if LOG_FILE_NAME includes a path
    log_dir = os.path.dirname(LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Created log directory: {log_dir}") # Print directly as logger might not be fully set up

    file_handler = logging.FileHandler(LOG_FILE_NAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # Always log debug level to file
    logger.addHandler(file_handler)
except IOError as e:
    # Use print as logger file handler failed
    print(f"{Fore.RED}Fatal: Failed to configure log file {LOG_FILE_NAME}: {e}{Style.RESET_ALL}", file=sys.stderr)
except Exception as e:
    print(f"{Fore.RED}Fatal: Unexpected error setting up file logging: {e}{Style.RESET_ALL}", file=sys.stderr)

# Load environment variables from .env file if present
if load_dotenv():
    logger.info(f"{Fore.CYAN}# Summoning secrets from .env scroll...{Style.RESET_ALL}")
else:
    logger.debug("No .env scroll found or no secrets contained within.")


# --- Robust API Retry Decorator ---
def retry_api_call(max_retries: int = DEFAULT_RETRY_MAX, initial_delay: int = DEFAULT_RETRY_DELAY_SECONDS) -> Callable:
    """
    Decorator for retrying CCXT API calls with exponential backoff.
    Handles common transient network errors, rate limits, and distinguishes
    non-retryable exchange errors like insufficient funds or invalid orders.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds between retries.

    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            """Wrapper function that implements the retry logic."""
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    # Channeling the API call...
                    return func(*args, **kwargs)

                # Specific Retryable CCXT Errors
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit veil encountered ({func.__name__}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}) Error: {e}{Style.RESET_ALL}"
                    )
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    # Log specific exception type for better debugging
                    # Log as WARNING as these are often temporary and retryable
                    logger.warning(
                        f"{Fore.YELLOW}Network ether disturbed ({func.__name__}: {type(e).__name__} - {e}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )

                # Exchange Errors - Some are retryable, others indicate fundamental issues
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    if any(phrase in err_str for phrase in [
                        "order not found", "order does not exist", "unknown order",
                        "order already cancelled", "order was canceled", "cancel order failed", # Added variations
                        "order is finished", "order has been filled" # Variations of closed states
                    ]):
                        # Log as WARNING because it's often expected (e.g., checking a closed order)
                        logger.warning(f"{Fore.YELLOW}Order vanished, unknown, or final state encountered for {func.__name__}: {e}. Returning None (no retry).{Style.RESET_ALL}")
                        return None # OrderNotFound or similar is often final for that specific call
                    elif any(phrase in err_str for phrase in [
                        "insufficient balance", "insufficient margin", "margin is insufficient",
                        "available balance insufficient", "insufficient funds" # Added variations
                    ]):
                         logger.error(f"{Fore.RED}Insufficient essence (funds/margin) for {func.__name__}: {e}. Aborting operation (no retry).{Style.RESET_ALL}")
                         return None # Cannot recover without external action
                    elif any(phrase in err_str for phrase in [
                        "invalid order", "parameter error", "size too small", "price too low",
                        "price too high", "invalid price precision", "invalid amount precision",
                        "api key is invalid", "authentication failed", "invalid signature", # Auth errors
                        "position mode not modified", "risk limit", # Config/Setup errors
                        "reduceonly", # Often indicates trying to close more than open
                        "order cost not meet", # Minimum cost issue
                    ]):
                         logger.error(f"{Fore.RED}Invalid parameters, limits, auth, or config for {func.__name__}: {e}. Aborting operation (no retry). Review logic/config/permissions.{Style.RESET_ALL}")
                         return None # Requires code or config fix
                    # --- Potentially Retryable Exchange Errors ---
                    # These might include temporary glitches, server-side issues, etc.
                    else:
                        logger.warning(
                            f"{Fore.YELLOW}Exchange spirit troubled ({func.__name__}: {type(e).__name__} - {e}). Pausing {delay}s... "
                            f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                        )
                # Catch-all for other unexpected exceptions during the API call
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Unexpected rift during {func.__name__}: {type(e).__name__} - {e}. Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}",
                        exc_info=True # Log traceback for unexpected errors
                    )

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    # Exponential backoff, capped at 60 seconds
                    delay = min(delay * 2, 60)
                    retries += 1
                else:
                    # Max retries reached, operation failed
                    logger.error(
                        f"{Fore.RED}Max retries ({max_retries}) reached for {func.__name__}. The spell falters.{Style.RESET_ALL}"
                    )
                    return None # Indicate failure after exhausting retries

            # Should theoretically not be reached, but acts as a failsafe
            return None
        return wrapper
    return decorator


# --- Core Scalping Bot Class ---
class ScalpingBot:
    """
    Pyrmethus Enhanced Scalping Bot v4. Optimized for Bybit V5 API.

    This bot implements a scalping strategy using technical indicators and
    order book analysis. It features persistent state management, ATR-based or
    fixed percentage SL/TP, parameter-based SL/TP placement via Bybit V5 API,
    and an **experimental** Trailing Stop Loss (TSL) mechanism using `edit_order`.

    **Disclaimer:** Trading cryptocurrencies involves significant risk. This bot
    is provided for educational purposes and experimentation. Use at your own risk.
    Thoroughly test in simulation and testnet modes before considering live trading.
    The TSL feature using `edit_order` requires careful verification and may not
    be supported correctly by the exchange or CCXT version. Consider alternatives
    like cancel/replace for TSL if `edit_order` proves unreliable.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """
        Initializes the bot instance.

        Loads configuration, validates settings, establishes exchange connection,
        loads persistent state, and caches necessary market information including
        price and amount decimal places.

        Args:
            config_file: Path to the YAML configuration file.
            state_file: Path to the JSON file for storing persistent state.
        """
        logger.info(f"{Fore.MAGENTA}--- Pyrmethus Scalping Bot v4 Awakening ---{Style.RESET_ALL}")
        self.config: Dict[str, Any] = {}
        self.state_file: str = state_file
        self.load_config(config_file)
        self.validate_config() # Validate before using config values

        # --- Bind Core Attributes from Config/Environment ---
        # Credentials (prefer environment variables for security)
        self.api_key: Optional[str] = os.getenv("BYBIT_API_KEY")
        self.api_secret: Optional[str] = os.getenv("BYBIT_API_SECRET")

        # Exchange Settings
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.testnet_mode: bool = self.config["exchange"]["testnet_mode"] # Explicit flag for clarity

        # Trading Parameters
        self.symbol: str = self.config["trading"]["symbol"]
        # Internal simulation mode (runs logic without placing real orders)
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]
        self.entry_order_type: str = self.config["trading"]["entry_order_type"]
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]
        self.timeframe: str = self.config["trading"]["timeframe"]
        self.close_positions_on_exit: bool = self.config["trading"]["close_positions_on_exit"]

        # Order Book Parameters
        self.order_book_depth: int = self.config["order_book"]["depth"]
        self.imbalance_threshold: float = self.config["order_book"]["imbalance_threshold"]

        # Indicator Parameters
        self.volatility_window: int = self.config["indicators"]["volatility_window"]
        self.volatility_multiplier: float = self.config["indicators"]["volatility_multiplier"]
        self.ema_period: int = self.config["indicators"]["ema_period"]
        self.rsi_period: int = self.config["indicators"]["rsi_period"]
        self.macd_short_period: int = self.config["indicators"]["macd_short_period"]
        self.macd_long_period: int = self.config["indicators"]["macd_long_period"]
        self.macd_signal_period: int = self.config["indicators"]["macd_signal_period"]
        self.stoch_rsi_period: int = self.config["indicators"]["stoch_rsi_period"]
        self.stoch_rsi_k_period: int = self.config["indicators"]["stoch_rsi_k_period"]
        self.stoch_rsi_d_period: int = self.config["indicators"]["stoch_rsi_d_period"]
        self.atr_period: int = self.config["indicators"]["atr_period"]

        # Risk Management Parameters
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = self.config["risk_management"]["atr_sl_multiplier"]
        self.atr_tp_multiplier: float = self.config["risk_management"]["atr_tp_multiplier"]
        # Base percentages used if use_atr_sl_tp is False
        self.base_stop_loss_pct: Optional[float] = self.config["risk_management"].get("stop_loss_percentage")
        self.base_take_profit_pct: Optional[float] = self.config["risk_management"].get("take_profit_percentage")
        # Bybit V5 trigger price types (e.g., MarkPrice, LastPrice, IndexPrice)
        self.sl_trigger_by: Optional[str] = self.config["risk_management"].get("sl_trigger_by")
        self.tp_trigger_by: Optional[str] = self.config["risk_management"].get("tp_trigger_by")
        # Trailing Stop Loss settings
        self.enable_trailing_stop_loss: bool = self.config["risk_management"]["enable_trailing_stop_loss"]
        self.trailing_stop_loss_percentage: Optional[float] = self.config["risk_management"].get("trailing_stop_loss_percentage")
        # Position and Exit settings
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.time_based_exit_minutes: Optional[int] = self.config["risk_management"].get("time_based_exit_minutes")
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        # Optional signal strength adjustment (kept simple)
        self.strong_signal_adjustment_factor: float = self.config["risk_management"]["strong_signal_adjustment_factor"]
        self.weak_signal_adjustment_factor: float = self.config["risk_management"]["weak_signal_adjustment_factor"]

        # --- Internal Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0 # Simple daily PnL tracker (reset manually or add logic)
        # Stores active and pending positions as a list of dictionaries.
        # Example structure:
        # {
        #     'id': str,                       # Exchange Order ID of the entry order
        #     'symbol': str,                   # Trading symbol
        #     'side': str,                     # 'buy' or 'sell'
        #     'size': float,                   # Current position size (base currency)
        #     'original_size': float,          # Requested size at entry
        #     'entry_price': float,            # Average fill price of the entry order
        #     'entry_time': float,             # Timestamp (seconds) of entry fill
        #     'status': str,                   # e.g., STATUS_PENDING_ENTRY, STATUS_ACTIVE, STATUS_CANCELED
        #     'entry_order_type': str,         # 'market' or 'limit'
        #     'stop_loss_price': Optional[float], # Price level for SL (parameter-based)
        #     'take_profit_price': Optional[float],# Price level for TP (parameter-based)
        #     'confidence': int,               # Signal score at entry
        #     'trailing_stop_price': Optional[float], # Current activated TSL price
        #     'last_update_time': float        # Timestamp of the last state update
        # }
        self.open_positions: List[Dict[str, Any]] = []
        self.market_info: Optional[Dict[str, Any]] = None # Cache for market details (precision, limits)
        self.price_decimals: int = DEFAULT_PRICE_DECIMALS   # Calculated/retrieved price decimal places
        self.amount_decimals: int = DEFAULT_AMOUNT_DECIMALS # Calculated/retrieved amount decimal places

        # --- Setup & Initialization Steps ---
        self._configure_logging_level() # Apply logging level from config
        self.exchange: ccxt.Exchange = self._initialize_exchange() # Connect to exchange
        self._load_market_info() # Load and cache market details for the symbol, calculate decimals
        self._load_state() # Load persistent state from file

        # --- Log Final Operating Modes ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")

        if not self.simulation_mode:
            if not self.testnet_mode:
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- WARNING: LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}")
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- Ensure configuration and risk parameters are correct! ---{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}--- Running in full internal simulation mode. No real orders will be placed. ---{Style.RESET_ALL}")


        logger.info(f"{Fore.CYAN}Scalping Bot V4 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}")

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper() # Nested under 'logging' section now
            log_level = getattr(logging, log_level_str, logging.INFO)

            # Set the specific bot logger level and the console handler level
            # File handler remains at DEBUG regardless of config
            logger.setLevel(log_level) # Set our bot's logger level
            console_handler.setLevel(log_level) # Console mirrors bot logger level

            logger.info(f"Console logging level enchanted to: {log_level_str}")
            logger.debug(f"Detailed logging enabled in file: {LOG_FILE_NAME}")
            # Log effective levels for verification
            logger.debug(f"Effective Bot Logger Level: {logging.getLevelName(logger.level)}")
            logger.debug(f"Effective Console Handler Level: {logging.getLevelName(console_handler.level)}")
            if file_handler:
                logger.debug(f"Effective File Handler Level: {logging.getLevelName(file_handler.level)}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error configuring logging level: {e}. Using default INFO.{Style.RESET_ALL}")
            logger.setLevel(logging.INFO)
            console_handler.setLevel(logging.INFO)

    def _load_state(self) -> None:
        """
        Loads bot state (open positions) from the state file.
        Implements backup and recovery logic in case of file corruption.
        """
        logger.debug(f"Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        try:
            if os.path.exists(self.state_file):
                # Check file size first to avoid loading large corrupted files unnecessarily
                if os.path.getsize(self.state_file) == 0:
                    logger.warning(f"{Fore.YELLOW}State file {self.state_file} is empty. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
                    self._save_state() # Create a valid empty state file
                    return

                with open(self.state_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip(): # Double check content after reading
                        logger.warning(f"{Fore.YELLOW}State file {self.state_file} contains only whitespace. Starting fresh.{Style.RESET_ALL}")
                        self.open_positions = []
                        self._save_state()
                        return

                    saved_state = json.loads(content)
                    # Basic validation: Ensure it's a list (of position dicts)
                    if isinstance(saved_state, list):
                        # Optional: Add more rigorous validation of each position dict structure here
                        valid_positions = []
                        for i, pos in enumerate(saved_state):
                            if isinstance(pos, dict) and 'id' in pos and 'side' in pos:
                                valid_positions.append(pos)
                            else:
                                logger.warning(f"Ignoring invalid entry #{i+1} in state file (not a valid position dict): {pos}")
                        self.open_positions = valid_positions
                        logger.info(f"{Fore.GREEN}Recalled {len(self.open_positions)} valid position(s) from state file.{Style.RESET_ALL}")
                        # Attempt to remove backup if load was successful
                        if os.path.exists(state_backup_file):
                            try:
                                os.remove(state_backup_file)
                                logger.debug(f"Removed old state backup file: {state_backup_file}")
                            except OSError as remove_err:
                                logger.warning(f"Could not remove old state backup file {state_backup_file}: {remove_err}")
                    else:
                        raise ValueError(f"Invalid state format in {self.state_file} - expected a list, got {type(saved_state)}.")
            else:
                logger.info(f"No prior state file found ({self.state_file}). Beginning anew.")
                self.open_positions = []
                self._save_state() # Create initial empty state file

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"{Fore.RED}Error decoding state file {self.state_file}: {e}. Attempting recovery...{Style.RESET_ALL}")
            # Attempt to restore from backup if it exists
            if os.path.exists(state_backup_file):
                logger.warning(f"{Fore.YELLOW}Attempting to restore state from backup: {state_backup_file}{Style.RESET_ALL}")
                try:
                    shutil.copyfile(state_backup_file, self.state_file)
                    logger.info(f"{Fore.GREEN}State successfully restored from backup. Retrying load...{Style.RESET_ALL}")
                    self._load_state() # Retry loading the restored file
                    return # Exit this attempt, the retry will handle it
                except Exception as restore_err:
                    logger.error(f"{Fore.RED}Failed to restore state from backup {state_backup_file}: {restore_err}. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
                    self._save_state() # Create a fresh state file
            else:
                # No backup exists, backup the corrupted file and start fresh
                corrupted_backup_path = f"{self.state_file}.corrupted_{int(time.time())}"
                logger.warning(f"{Fore.YELLOW}No backup found. Backing up corrupted file to {corrupted_backup_path} and starting fresh.{Style.RESET_ALL}")
                try:
                    if os.path.exists(self.state_file): # Ensure file exists before copying
                        shutil.copyfile(self.state_file, corrupted_backup_path)
                except Exception as backup_err:
                     logger.error(f"{Fore.RED}Could not back up corrupted state file {self.state_file}: {backup_err}{Style.RESET_ALL}")
                # Start with empty state and save it
                self.open_positions = []
                self._save_state()

        except Exception as e:
            # Catch any other unexpected errors during state loading
            logger.critical(f"{Fore.RED}Fatal error loading state from {self.state_file}: {e}{Style.RESET_ALL}", exc_info=True)
            logger.warning("Proceeding with an empty state due to critical load failure. Manual review of state file recommended.")
            self.open_positions = []

    def _save_state(self) -> None:
        """
        Saves the current bot state (list of open positions) to the state file.
        Creates a backup before overwriting the main state file using an atomic move.
        """
        if not hasattr(self, 'open_positions'): # Ensure state exists before saving
             logger.error("Cannot save state: 'open_positions' attribute does not exist.")
             return

        logger.debug(f"Recording current state ({len(self.open_positions)} positions) to {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        temp_state_file = f"{self.state_file}.tmp_{os.getpid()}" # Use PID for uniqueness

        try:
            # Write to a temporary file first
            with open(temp_state_file, 'w', encoding='utf-8') as f:
                # Use default=str to handle potential non-serializable types gracefully (e.g., numpy types)
                # Convert numpy types explicitly if needed: json.dump(self.open_positions, f, indent=4, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
                json.dump(self.open_positions, f, indent=4, default=str)

            # If temporary write succeeds, create backup of current state (if exists)
            if os.path.exists(self.state_file):
                try:
                    # copy2 preserves more metadata than copyfile
                    shutil.copy2(self.state_file, state_backup_file)
                except Exception as backup_err:
                    logger.warning(f"Could not create state backup {state_backup_file}: {backup_err}. Proceeding with overwrite.")

            # Atomically replace the old state file with the new one
            # On Windows, os.replace provides atomicity if possible, otherwise falls back.
            # On POSIX, os.rename is generally atomic.
            os.replace(temp_state_file, self.state_file)

            logger.debug("State recorded successfully.")
        except IOError as e:
            logger.error(f"{Fore.RED}Could not scribe state to {self.state_file}: {e}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True)
            # Clean up temporary file if it still exists after an error
            if os.path.exists(temp_state_file):
                try: os.remove(temp_state_file)
                except OSError: pass


    def load_config(self, config_file: str) -> None:
        """Loads configuration from the specified YAML file or creates a default one if not found."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            # Basic structure validation
            if not isinstance(self.config, dict):
                 logger.critical(f"{Fore.RED}Fatal: Config file {config_file} has invalid structure (must be a dictionary). Aborting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.info(f"{Fore.GREEN}Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")

        except FileNotFoundError:
            logger.warning(f"{Fore.YELLOW}Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
                logger.info(f"{Fore.YELLOW}A default spellbook has been crafted: '{config_file}'.{Style.RESET_ALL}")
                logger.info(f"{Fore.YELLOW}IMPORTANT: Please review and tailor its enchantments (API keys in .env, symbol, risk settings), then restart the bot.{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"{Fore.RED}Failed to craft default spellbook: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1) # Exit after creating default config, user needs to review/edit it

        except yaml.YAMLError as e:
            logger.critical(f"{Fore.RED}Fatal: Error parsing spellbook {config_file}: {e}. Check YAML syntax. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Fatal: Unexpected chaos loading configuration: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file with sensible defaults and explanations."""
        default_config = {
            "logging": {
                # Logging level for console output: DEBUG, INFO, WARNING, ERROR, CRITICAL
                "level": "INFO",
            },
            "exchange": {
                "exchange_id": os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID),
                # Explicit flag for using testnet environment. Set to false for mainnet.
                "testnet_mode": os.getenv("TESTNET_MODE", "True").lower() in ("true", "1", "yes"),
                # API Credentials should be set via environment variables:
                # BYBIT_API_KEY=YOUR_KEY
                # BYBIT_API_SECRET=YOUR_SECRET
            },
            "trading": {
                # Symbol format depends on exchange (e.g., BTC/USDT:USDT for Bybit USDT Perpetuals)
                "symbol": os.getenv("TRADING_SYMBOL", "BTC/USDT:USDT"),
                # Timeframe for OHLCV data (e.g., 1m, 5m, 1h) - Must be supported by exchange
                "timeframe": os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME),
                # Internal simulation: if true, bot runs logic but places no real orders.
                "simulation_mode": os.getenv("SIMULATION_MODE", "True").lower() in ("true", "1", "yes"),
                # Entry order type: 'limit' or 'market'
                "entry_order_type": os.getenv("ENTRY_ORDER_TYPE", "limit").lower(),
                # Offset from current price for limit buy orders (e.g., 0.0005 = 0.05% below current)
                "limit_order_offset_buy": float(os.getenv("LIMIT_ORDER_OFFSET_BUY", 0.0005)),
                # Offset from current price for limit sell orders (e.g., 0.0005 = 0.05% above current)
                "limit_order_offset_sell": float(os.getenv("LIMIT_ORDER_OFFSET_SELL", 0.0005)),
                # Set to true to automatically close all open positions via market order on bot shutdown.
                # USE WITH CAUTION! Ensure this is the desired behavior.
                "close_positions_on_exit": False,
            },
            "order_book": {
                # Number of bid/ask levels to fetch
                "depth": 10,
                # Ask volume / Bid volume ratio threshold to indicate sell pressure
                # e.g., 1.5 means asks > 1.5 * bids = sell signal component
                # e.g., < (1 / 1.5) = 0.67 means asks < 0.67 * bids = buy signal component
                "imbalance_threshold": 1.5
            },
            "indicators": {
                # --- Volatility ---
                "volatility_window": 20,
                "volatility_multiplier": 0.0, # Multiplier for vol-based order size adjustment (0 = disabled)
                # --- Moving Average ---
                "ema_period": 10,
                # --- Oscillators ---
                "rsi_period": 14,
                "stoch_rsi_period": 14, # RSI period used within StochRSI
                "stoch_rsi_k_period": 3,  # Stochastic %K smoothing period
                "stoch_rsi_d_period": 3,  # Stochastic %D smoothing period
                # --- Trend/Momentum ---
                "macd_short_period": 12,
                "macd_long_period": 26,
                "macd_signal_period": 9,
                # --- Average True Range (for SL/TP) ---
                "atr_period": 14,
            },
            "risk_management": {
                # Percentage of available quote balance to use for each new order (e.g., 0.01 = 1%)
                "order_size_percentage": 0.01,
                # Maximum number of concurrent open positions allowed
                "max_open_positions": 1,
                # --- Stop Loss & Take Profit Method ---
                # Set to true to use ATR for SL/TP, false to use fixed percentages
                "use_atr_sl_tp": True,
                # --- ATR Settings (if use_atr_sl_tp is true) ---
                "atr_sl_multiplier": ATR_MULTIPLIER_SL, # SL = entry_price +/- (ATR * multiplier)
                "atr_tp_multiplier": ATR_MULTIPLIER_TP, # TP = entry_price +/- (ATR * multiplier)
                # --- Fixed Percentage Settings (if use_atr_sl_tp is false) ---
                "stop_loss_percentage": 0.005, # e.g., 0.005 = 0.5% SL from entry price
                "take_profit_percentage": 0.01, # e.g., 0.01 = 1.0% TP from entry price
                # --- Bybit V5 Trigger Price Types ---
                # Optional: Specify trigger type ('MarkPrice', 'LastPrice', 'IndexPrice'). Check Bybit/CCXT docs.
                # Defaults to exchange default (often MarkPrice) if not specified or None.
                "sl_trigger_by": "MarkPrice", # Or "LastPrice", "IndexPrice", None
                "tp_trigger_by": "MarkPrice", # Or "LastPrice", "IndexPrice", None
                # --- Trailing Stop Loss (EXPERIMENTAL - VERIFY SUPPORT) ---
                "enable_trailing_stop_loss": False, # Enable/disable TSL feature (Default OFF due to experimental nature)
                # Trail distance as a percentage (e.g., 0.003 = 0.3%). Used only if TSL is enabled.
                "trailing_stop_loss_percentage": 0.003,
                # --- Other Exit Conditions ---
                # Time limit in minutes to close a position if still open (0 or null/None to disable)
                "time_based_exit_minutes": 60,
                # --- Signal Adjustment (Optional - currently unused in signal score) ---
                # Multiplier applied to order size based on signal strength (1.0 = no adjustment)
                "strong_signal_adjustment_factor": 1.0, # Applied if abs(score) >= STRONG_SIGNAL_THRESHOLD_ABS
                "weak_signal_adjustment_factor": 1.0,   # Applied if abs(score) < STRONG_SIGNAL_THRESHOLD_ABS
            },
        }
        # --- Environment Variable Overrides (Example) ---
        # Allow overriding specific config values via environment variables if needed
        if os.getenv("ORDER_SIZE_PERCENTAGE"):
             try:
                override_val = float(os.environ["ORDER_SIZE_PERCENTAGE"])
                if 0 < override_val <= 1:
                    default_config["risk_management"]["order_size_percentage"] = override_val
                    logger.info(f"Overrode order_size_percentage from environment variable to {override_val}.")
                else:
                    logger.warning(f"Invalid ORDER_SIZE_PERCENTAGE ({override_val}) in environment, must be > 0 and <= 1. Using default.")
             except ValueError:
                 logger.warning("Invalid ORDER_SIZE_PERCENTAGE in environment (not a number), using default.")
        # ... add more overrides as desired ...

        try:
            with open(config_file, "w", encoding='utf-8') as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False)
            logger.info(f"Default spellbook '{config_file}' successfully crafted.")
        except IOError as e:
            logger.error(f"{Fore.RED}Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}")
            raise # Re-raise the exception to be caught by the calling function

    def validate_config(self) -> None:
        """Performs detailed validation of the loaded configuration parameters."""
        logger.debug("Scrutinizing the configuration spellbook for potency and validity...")
        try:
            # --- Helper Validation Functions ---
            def _get_nested(data: Dict, keys: List[str], default: Any = None) -> Any:
                """Safely get nested dictionary values."""
                value = data
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return default
                return value

            def _check(condition: bool, key_path: str, message: str):
                 if not condition:
                     raise ValueError(f"Config Error [{key_path}]: {message}")

            # --- Section Existence ---
            required_top_level_keys = {"logging", "exchange", "trading", "order_book", "indicators", "risk_management"}
            missing_sections = required_top_level_keys - self.config.keys()
            _check(not missing_sections, ".", f"Missing required configuration sections: {missing_sections}")
            for section in required_top_level_keys:
                 _check(isinstance(self.config[section], dict), section, "Must be a dictionary (mapping).")

            # --- Logging Section ---
            cfg_log = self.config["logging"]
            log_level = _get_nested(cfg_log, ["level"], "INFO").upper()
            _check(hasattr(logging, log_level), "logging.level", f"Invalid level '{log_level}'. Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")

            # --- Exchange Section ---
            cfg_ex = self.config["exchange"]
            ex_id = _get_nested(cfg_ex, ["exchange_id"])
            _check(isinstance(ex_id, str) and ex_id, "exchange.exchange_id", "Must be a non-empty string.")
            _check(ex_id in ccxt.exchanges, "exchange.exchange_id", f"Unsupported exchange_id: '{ex_id}'. Available examples: {ccxt.exchanges[:10]}...")
            _check(isinstance(_get_nested(cfg_ex, ["testnet_mode"]), bool), "exchange.testnet_mode", "Must be true or false.")

            # --- Trading Section ---
            cfg_tr = self.config["trading"]
            _check(isinstance(_get_nested(cfg_tr, ["symbol"]), str) and _get_nested(cfg_tr, ["symbol"]), "trading.symbol", "Must be a non-empty string.")
            _check(isinstance(_get_nested(cfg_tr, ["timeframe"]), str) and _get_nested(cfg_tr, ["timeframe"]), "trading.timeframe", "Must be a non-empty string.")
            # Note: Timeframe/Symbol validity checked later against exchange capabilities after connection
            _check(isinstance(_get_nested(cfg_tr, ["simulation_mode"]), bool), "trading.simulation_mode", "Must be true or false.")
            entry_type = _get_nested(cfg_tr, ["entry_order_type"])
            _check(entry_type in ["market", "limit"], "trading.entry_order_type", f"Must be 'market' or 'limit', got: {entry_type}")
            _check(isinstance(_get_nested(cfg_tr, ["limit_order_offset_buy"]), (int, float)) and _get_nested(cfg_tr, ["limit_order_offset_buy"], -1) >= 0, "trading.limit_order_offset_buy", "Must be a non-negative number.")
            _check(isinstance(_get_nested(cfg_tr, ["limit_order_offset_sell"]), (int, float)) and _get_nested(cfg_tr, ["limit_order_offset_sell"], -1) >= 0, "trading.limit_order_offset_sell", "Must be a non-negative number.")
            _check(isinstance(_get_nested(cfg_tr, ["close_positions_on_exit"]), bool), "trading.close_positions_on_exit", "Must be true or false.")

            # --- Order Book Section ---
            cfg_ob = self.config["order_book"]
            _check(isinstance(_get_nested(cfg_ob, ["depth"]), int) and _get_nested(cfg_ob, ["depth"], 0) > 0, "order_book.depth", "Must be a positive integer.")
            _check(isinstance(_get_nested(cfg_ob, ["imbalance_threshold"]), (int, float)) and _get_nested(cfg_ob, ["imbalance_threshold"], 0) > 0, "order_book.imbalance_threshold", "Must be a positive number.")

            # --- Indicators Section ---
            cfg_ind = self.config["indicators"]
            period_keys = ["volatility_window", "ema_period", "rsi_period", "macd_short_period",
                           "macd_long_period", "macd_signal_period", "stoch_rsi_period",
                           "stoch_rsi_k_period", "stoch_rsi_d_period", "atr_period"]
            for key in period_keys:
                 _check(isinstance(_get_nested(cfg_ind, [key]), int) and _get_nested(cfg_ind, [key], 0) > 0, f"indicators.{key}", "Must be a positive integer.")
            _check(isinstance(_get_nested(cfg_ind, ["volatility_multiplier"]), (int, float)) and _get_nested(cfg_ind, ["volatility_multiplier"], -1) >= 0, "indicators.volatility_multiplier", "Must be a non-negative number.")
            _check(cfg_ind['macd_short_period'] < cfg_ind['macd_long_period'], "indicators", "macd_short_period must be less than macd_long_period.")

            # --- Risk Management Section ---
            cfg_rm = self.config["risk_management"]
            use_atr = _get_nested(cfg_rm, ["use_atr_sl_tp"])
            _check(isinstance(use_atr, bool), "risk_management.use_atr_sl_tp", "Must be true or false.")
            if use_atr:
                 _check(isinstance(_get_nested(cfg_rm, ["atr_sl_multiplier"]), (int, float)) and _get_nested(cfg_rm, ["atr_sl_multiplier"], 0) > 0, "risk_management.atr_sl_multiplier", "Must be a positive number when use_atr_sl_tp is true.")
                 _check(isinstance(_get_nested(cfg_rm, ["atr_tp_multiplier"]), (int, float)) and _get_nested(cfg_rm, ["atr_tp_multiplier"], 0) > 0, "risk_management.atr_tp_multiplier", "Must be a positive number when use_atr_sl_tp is true.")
            else:
                 sl_pct = _get_nested(cfg_rm, ["stop_loss_percentage"])
                 tp_pct = _get_nested(cfg_rm, ["take_profit_percentage"])
                 _check(sl_pct is not None and isinstance(sl_pct, (int, float)) and 0 < sl_pct < 1, "risk_management.stop_loss_percentage", "Must be a number between 0 and 1 (exclusive) when use_atr_sl_tp is false.")
                 _check(tp_pct is not None and isinstance(tp_pct, (int, float)) and 0 < tp_pct < 1, "risk_management.take_profit_percentage", "Must be a number between 0 and 1 (exclusive) when use_atr_sl_tp is false.")

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None]
            sl_trig = _get_nested(cfg_rm, ["sl_trigger_by"])
            tp_trig = _get_nested(cfg_rm, ["tp_trigger_by"])
            _check(sl_trig in valid_triggers, "risk_management.sl_trigger_by", f"Invalid trigger type: '{sl_trig}'. Must be one of: {valid_triggers}")
            _check(tp_trig in valid_triggers, "risk_management.tp_trigger_by", f"Invalid trigger type: '{tp_trig}'. Must be one of: {valid_triggers}")

            enable_tsl = _get_nested(cfg_rm, ["enable_trailing_stop_loss"])
            _check(isinstance(enable_tsl, bool), "risk_management.enable_trailing_stop_loss", "Must be true or false.")
            if enable_tsl:
                 tsl_pct = _get_nested(cfg_rm, ["trailing_stop_loss_percentage"])
                 _check(tsl_pct is not None and isinstance(tsl_pct, (int, float)) and 0 < tsl_pct < 1, "risk_management.trailing_stop_loss_percentage", "Must be a positive number less than 1 (e.g., 0.005 for 0.5%) when TSL is enabled.")

            order_size_pct = _get_nested(cfg_rm, ["order_size_percentage"])
            _check(isinstance(order_size_pct, (int, float)) and 0 < order_size_pct <= 1, "risk_management.order_size_percentage", "Must be a number between 0 (exclusive) and 1 (inclusive).")
            _check(isinstance(_get_nested(cfg_rm, ["max_open_positions"]), int) and _get_nested(cfg_rm, ["max_open_positions"], 0) > 0, "risk_management.max_open_positions", "Must be a positive integer.")
            time_exit = _get_nested(cfg_rm, ["time_based_exit_minutes"])
            _check(time_exit is None or (isinstance(time_exit, int) and time_exit >= 0), "risk_management.time_based_exit_minutes", "Must be a non-negative integer or null/None.")
            _check(isinstance(_get_nested(cfg_rm, ["strong_signal_adjustment_factor"]), (int, float)) and _get_nested(cfg_rm, ["strong_signal_adjustment_factor"], 0) > 0, "risk_management.strong_signal_adjustment_factor", "Must be a positive number.")
            _check(isinstance(_get_nested(cfg_rm, ["weak_signal_adjustment_factor"]), (int, float)) and _get_nested(cfg_rm, ["weak_signal_adjustment_factor"], 0) > 0, "risk_management.weak_signal_adjustment_factor", "Must be a positive number.")

            logger.info(f"{Fore.GREEN}Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except ValueError as e:
            logger.critical(f"{Fore.RED}Configuration flaw detected: {e}. Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Unexpected chaos during configuration validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and configures the CCXT exchange instance."""
        logger.info(f"Opening communication channel with {self.exchange_id.upper()}...")

        # Check credentials early, especially if not in pure simulation mode
        creds_found = self.api_key and self.api_secret
        if not self.simulation_mode and not creds_found:
             logger.critical(f"{Fore.RED}API Key/Secret essence missing in environment variables (BYBIT_API_KEY, BYBIT_API_SECRET). Cannot trade live or on testnet. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        elif creds_found and self.simulation_mode:
             logger.warning(f"{Fore.YELLOW}API Key/Secret found, but internal simulation_mode is True. Credentials will NOT be used for placing orders.{Style.RESET_ALL}")
        elif not creds_found and self.simulation_mode:
             logger.info(f"{Fore.CYAN}Running in internal simulation mode. API credentials not required or found.{Style.RESET_ALL}")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            # Base configuration for the exchange instance
            exchange_config = {
                'enableRateLimit': True, # Essential for respecting API limits
                'options': {
                    # Default market type (adjust based on typical use for the exchange)
                    'defaultType': 'swap' if 'bybit' in self.exchange_id.lower() else 'future',
                    'adjustForTimeDifference': True, # Corrects minor clock drifts
                    # Specific options can be added, e.g., for Bybit V5:
                    # 'recvWindow': 10000, # Example: Increase receive window if timeouts occur
                }
            }
            # Add Bybit V5 specific option if needed (check if CCXT handles this automatically)
            # if self.exchange_id.lower() == 'bybit':
            #     exchange_config['options']['defaultSubType'] = 'linear' # Or 'inverse'

            # Add API credentials only if needed (i.e., not internal simulation)
            if not self.simulation_mode:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                logger.debug("API credentials added to exchange configuration.")

            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT's unified method (if not in internal simulation)
            if not self.simulation_mode and self.testnet_mode:
                try:
                    logger.info("Attempting to switch exchange instance to Testnet mode...")
                    exchange.set_sandbox_mode(True)
                    logger.info(f"{Fore.YELLOW}Exchange sandbox mode explicitly enabled via CCXT.{Style.RESET_ALL}")
                except ccxt.NotSupported:
                     logger.warning(f"{Fore.YELLOW}Exchange {self.exchange_id} does not support unified set_sandbox_mode via CCXT. Testnet functionality depends on API key type and exchange base URL.{Style.RESET_ALL}")
                except Exception as sandbox_err:
                     logger.error(f"{Fore.RED}Error setting sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}")
            elif not self.simulation_mode and not self.testnet_mode:
                 logger.info(f"{Fore.GREEN}Exchange sandbox mode is OFF. Operating on Mainnet.{Style.RESET_ALL}")


            # Load markets to get symbol info, limits, precision, etc.
            logger.debug("Loading market matrix...")
            exchange.load_markets(reload=True) # Force reload to get latest info
            logger.debug("Market matrix loaded.")

            # Validate timeframe against loaded exchange capabilities
            if self.timeframe not in exchange.timeframes:
                available_tf = list(exchange.timeframes.keys())
                logger.critical(f"{Fore.RED}Timeframe '{self.timeframe}' not supported by {self.exchange_id}. Available options start with: {available_tf[:15]}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)

            # Validate symbol against loaded markets
            if self.symbol not in exchange.markets:
                available_sym = list(exchange.markets.keys())
                logger.critical(f"{Fore.RED}Symbol '{self.symbol}' not found on {self.exchange_id}. Available symbols start with: {available_sym[:10]}... Check config. Aborting.{Style.RESET_ALL}")
                sys.exit(1)

            # Log confirmation of symbol and timeframe validity
            logger.info(f"Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed available on {self.exchange_id}.")

            # Optional: Check if the specific market is active/tradeable
            market_details = exchange.market(self.symbol)
            if not market_details.get('active', True): # Default to True if 'active' key is missing
                 logger.warning(f"{Fore.YELLOW}Warning: Market '{self.symbol}' is marked as inactive by the exchange.{Style.RESET_ALL}")

            # Optional: Perform a simple API call to test connectivity/authentication (if not simulating)
            if not self.simulation_mode:
                logger.debug("Performing initial API connectivity test (fetching time)...")
                try:
                    server_time_ms = exchange.fetch_time()
                    server_time_str = pd.to_datetime(server_time_ms, unit='ms', utc=True).isoformat()
                    logger.debug(f"Exchange time crystal synchronized: {server_time_str}")
                    logger.info(f"{Fore.GREEN}API connection and authentication successful.{Style.RESET_ALL}")
                except ccxt.AuthenticationError as auth_err:
                    logger.critical(f"{Fore.RED}Authentication check failed after loading markets: {auth_err}. Check API keys/permissions. Aborting.{Style.RESET_ALL}")
                    sys.exit(1)
                except Exception as time_err:
                    # Warn but don't necessarily abort for time fetch failure
                    logger.warning(f"{Fore.YELLOW}Connectivity test partially failed: Could not fetch server time. Proceeding cautiously. Error: {time_err}{Style.RESET_ALL}")

            logger.info(f"{Fore.GREEN}Connection established and configured for {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Fore.RED}Authentication failed for {self.exchange_id.upper()}: {e}. Check API keys/permissions in .env file or config. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
             logger.critical(f"{Fore.RED}Could not connect to {self.exchange_id.upper()} ({type(e).__name__}): {e}. Check network/exchange status. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except ccxt.NotSupported as e:
             logger.critical(f"{Fore.RED}Exchange {self.exchange_id} reported 'Not Supported' during initialization: {e}. Configuration might be incompatible. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            # Catch-all for unexpected initialization errors
            logger.critical(f"{Fore.RED}Unexpected error initializing exchange {self.exchange_id.upper()}: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _calculate_decimal_places(self, tick_size: Union[float, int]) -> int:
        """
        Calculates the number of decimal places from a tick size (e.g., 0.01 -> 2).
        Handles various formats and uses math.log10 for robustness.

        Args:
            tick_size: The minimum price or amount increment.

        Returns:
            The number of decimal places, or a default if calculation fails.
        """
        if not isinstance(tick_size, (float, int)) or tick_size <= 0:
            logger.warning(f"Invalid tick size for decimal calculation: {tick_size}. Returning default.")
            return DEFAULT_PRICE_DECIMALS # Use price default as fallback

        try:
            # Use logarithm for standard powers of 10
            if math.log10(tick_size) % 1 == 0:
                # Handle cases like 1, 10, 100 (log10 gives 0, 1, 2) -> 0 decimals
                # Handle cases like 0.1, 0.01 (log10 gives -1, -2) -> 1, 2 decimals
                decimals = max(0, int(-math.log10(tick_size)))
                return decimals
            else:
                # If not a power of 10 (e.g., 0.5, 0.0025), convert to string and count
                s = format(tick_size, '.16f').rstrip('0')
                if '.' in s:
                    return len(s.split('.')[-1])
                else: # Should be integer > 1 if not power of 10, e.g., 5, 25
                    return 0
        except Exception as e:
            logger.error(f"Error calculating decimal places for tick size {tick_size}: {e}. Using string fallback.")
            # Fallback using string formatting if log fails
            try:
                s = format(tick_size, '.16f').rstrip('0')
                if '.' in s:
                    return len(s.split('.')[-1])
                else:
                    return 0
            except Exception as e_str:
                 logger.error(f"String fallback for decimal calculation also failed for {tick_size}: {e_str}. Returning default.")
                 return DEFAULT_PRICE_DECIMALS


    def _load_market_info(self) -> None:
        """
        Loads and caches essential market information (precision, limits) for the trading symbol.
        Crucially, calculates and stores the number of decimal places for price and amount.
        """
        logger.debug(f"Loading market details for {self.symbol}...")
        try:
            # Use exchange.market() which gets info from already loaded markets
            self.market_info = self.exchange.market(self.symbol)
            if not self.market_info: # Should not happen if symbol validation passed, but check anyway
                raise ValueError(f"Market info for symbol '{self.symbol}' could not be retrieved after loading markets.")

            # --- Extract Precision and Limits ---
            precision = self.market_info.get('precision', {})
            limits = self.market_info.get('limits', {})
            amount_tick = precision.get('amount') # Minimum amount step
            price_tick = precision.get('price')   # Minimum price step
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            # --- Calculate Decimal Places ---
            # Determine based on tick size
            self.price_decimals = self._calculate_decimal_places(price_tick) if price_tick is not None else DEFAULT_PRICE_DECIMALS
            self.amount_decimals = self._calculate_decimal_places(amount_tick) if amount_tick is not None else DEFAULT_AMOUNT_DECIMALS

            # Log warnings if essential info is missing
            if amount_tick is None or price_tick is None:
                logger.warning(f"{Fore.YELLOW}Precision tick size info incomplete for {self.symbol}. PriceTick={price_tick}, AmountTick={amount_tick}. Calculations might be inaccurate.{Style.RESET_ALL}")
            if min_amount is None:
                 logger.warning(f"{Fore.YELLOW}Minimum order amount limit missing for {self.symbol}. Cannot enforce minimum amount.{Style.RESET_ALL}")
            if min_cost is None:
                 logger.warning(f"{Fore.YELLOW}Minimum order cost limit missing for {self.symbol}. Cannot enforce minimum cost.{Style.RESET_ALL}")

            # Log key details including calculated decimals
            logger.info(f"Market Details for {self.symbol}: "
                        f"Price Tick={price_tick}, Price Decimals={self.price_decimals}, "
                        f"Amount Tick={amount_tick}, Amount Decimals={self.amount_decimals}, "
                        f"Min Amount={min_amount}, Min Cost={min_cost}")
            logger.debug("Market details loaded and cached.")

        except KeyError as e:
             logger.critical(f"{Fore.RED}KeyError accessing market info for {self.symbol}: {e}. Market structure might be unexpected. Market Info: {self.market_info}. Aborting.{Style.RESET_ALL}", exc_info=True)
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}Failed to load or validate crucial market info for {self.symbol}: {e}. Cannot continue safely. Aborting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol."""
        logger.debug(f"Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and ticker.get('last') is not None:
            try:
                price = float(ticker['last'])
                # Log with appropriate precision using stored decimal count
                logger.debug(f"Current market price ({self.symbol}): {price:.{self.price_decimals}f}")
                return price
            except (ValueError, TypeError) as e:
                 logger.error(f"{Fore.RED}Error converting ticker 'last' price to float for {self.symbol}. Value: {ticker.get('last')}. Error: {e}{Style.RESET_ALL}")
                 return None
        else:
            logger.warning(f"{Fore.YELLOW}Could not fetch valid 'last' price for {self.symbol}. Ticker data received: {ticker}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self) -> Optional[Dict[str, Any]]:
        """Fetches order book data (bids, asks) and calculates the volume imbalance."""
        logger.debug(f"Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        try:
            # params = {} # Add if needed for specific exchange behavior
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth) #, params=params)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            result = {'bids': bids, 'asks': asks, 'imbalance': None}

            # Ensure bids/asks are lists of lists/tuples with at least price and volume
            valid_bids = [bid for bid in bids if isinstance(bid, (list, tuple)) and len(bid) >= 2 and isinstance(bid[1], (int, float)) and bid[1] >= 0]
            valid_asks = [ask for ask in asks if isinstance(ask, (list, tuple)) and len(ask) >= 2 and isinstance(ask[1], (int, float)) and ask[1] >= 0]

            if valid_bids and valid_asks:
                # Calculate total volume within the fetched depth
                bid_volume = sum(float(bid[1]) for bid in valid_bids)
                ask_volume = sum(float(ask[1]) for ask in valid_asks)

                # Calculate imbalance ratio (Ask Volume / Bid Volume)
                # Handle potential division by zero
                if bid_volume > 1e-12: # Use a small epsilon to avoid floating point issues
                    imbalance_ratio = ask_volume / bid_volume
                    result['imbalance'] = imbalance_ratio
                    logger.debug(f"Order Book ({self.symbol}) Imbalance (Ask Vol / Bid Vol): {imbalance_ratio:.3f}")
                elif ask_volume > 1e-12: # Bids are zero, asks are not
                    result['imbalance'] = float('inf') # Represent infinite pressure upwards
                    logger.debug(f"Order Book ({self.symbol}) Imbalance: Infinity (Zero Bid Volume)")
                else: # Both bid and ask volumes are effectively zero
                    result['imbalance'] = 1.0 # Define as balanced if both are zero? Or None? Let's use None.
                    result['imbalance'] = None
                    logger.debug(f"Order Book ({self.symbol}) Imbalance: N/A (Near-Zero Bid/Ask Volume)")
            else:
                logger.warning(f"{Fore.YELLOW}Order book data incomplete or invalid for {self.symbol}. Bids found: {len(bids)}, Asks found: {len(asks)}, Valid Bids: {len(valid_bids)}, Valid Asks: {len(valid_asks)}{Style.RESET_ALL}")

            return result
        except Exception as e:
            # Catch potential errors during fetch_order_book call itself
            logger.error(f"{Fore.RED}Error fetching or processing order book for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None # Indicate failure

    @retry_api_call(max_retries=2, initial_delay=1) # Less critical, fewer retries
    def fetch_historical_data(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data and prepares it as a Pandas DataFrame.
        Calculates the required number of candles based on indicator periods.

        Args:
            limit: Optional override for the number of candles to fetch.

        Returns:
            A pandas DataFrame with OHLCV data, or None on failure.
        """
        # Determine the minimum number of candles required for all indicators
        if limit is None:
             # Calculate max lookback needed by any indicator + buffer
             required_limit = max(
                 self.volatility_window + 1 if self.volatility_window else 0,
                 self.ema_period + 1 if self.ema_period else 0,
                 self.rsi_period + 2 if self.rsi_period else 0, # RSI needs one prior diff
                 (self.macd_long_period + self.macd_signal_period) if (self.macd_long_period and self.macd_signal_period) else 0, # Needs ~long+signal for good MACD hist
                 (self.rsi_period + self.stoch_rsi_period + max(self.stoch_rsi_k_period, self.stoch_rsi_d_period) + 1) if self.stoch_rsi_period else 0, # StochRSI needs RSI series first
                 self.atr_period + 1 if self.atr_period else 0
             ) + 25 # Add a larger safety buffer (e.g., 25)

             # Ensure a minimum fetch limit if all periods are very small
             required_limit = max(required_limit, 50)
             # Cap the limit to avoid excessive requests (adjust as needed)
             required_limit = min(required_limit, 1000) # Max limit depends on exchange (e.g., Bybit often 1000 or 1500)
        else:
             required_limit = limit

        logger.debug(f"Fetching approximately {required_limit} historical candles for {self.symbol} ({self.timeframe})...")
        try:
            # Fetch OHLCV data
            # params = {} # Add exchange-specific params if needed
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=required_limit) #, params=params)

            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}No historical OHLCV data returned for {self.symbol} with timeframe {self.timeframe} and limit {required_limit}.{Style.RESET_ALL}")
                return None

            # Convert to DataFrame and process
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Ensure UTC timezone
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_len = len(df)
            df.dropna(inplace=True) # Remove rows with any NaN values (e.g., from conversion errors)
            final_len = len(df)

            if final_len < initial_len:
                logger.debug(f"Dropped {initial_len - final_len} rows containing NaNs from historical data.")

            # Check if enough data remains after cleaning for the longest indicator calculation
            # Use a slightly smaller required length for this check to be lenient
            min_required_len = max(self.volatility_window, self.ema_period, self.rsi_period+1,
                                    self.macd_long_period + self.macd_signal_period,
                                    self.rsi_period + self.stoch_rsi_period + max(self.stoch_rsi_k_period, self.stoch_rsi_d_period),
                                    self.atr_period + 1) + 5 # Smaller buffer for check

            if final_len < min_required_len:
                 logger.warning(f"{Fore.YELLOW}Insufficient valid historical data after cleaning for {self.symbol}. Fetched {initial_len}, valid {final_len}, needed ~{min_required_len} for full indicator calc. Indicators might be inaccurate or fail.{Style.RESET_ALL}")
                 # Return the potentially insufficient DataFrame and let indicator functions handle it
                 if final_len == 0: return None # Definitely return None if empty

            logger.debug(f"Fetched and processed {final_len} valid historical candles for {self.symbol}.")
            return df

        except ccxt.BadSymbol as e:
             logger.critical(f"{Fore.RED}BadSymbol error fetching history for {self.symbol}. Is the symbol correct in config? Error: {e}{Style.RESET_ALL}")
             sys.exit(1) # Exit if symbol is fundamentally wrong
        except Exception as e:
            # Catch other potential errors during fetch or processing
            logger.error(f"{Fore.RED}Error fetching/processing historical data for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: Optional[str] = None) -> float:
        """
        Fetches the 'free' or 'available' balance for a specific currency code (quote currency by default).
        Attempts to handle unified margin account structures (like Bybit V5).

        Args:
            currency_code: The currency code (e.g., 'USDT') to fetch the balance for. Defaults to the quote currency of the market.

        Returns:
            The available balance as a float, or 0.0 if unavailable or error occurs.
        """
        # Determine the quote currency from the market info if not provided
        quote_currency = currency_code or self.market_info.get('quote')
        if not quote_currency:
             logger.error(f"{Fore.RED}Cannot determine quote currency to fetch balance (market_info: {self.market_info}).{Style.RESET_ALL}")
             return 0.0

        logger.debug(f"Fetching available balance for {quote_currency}...")

        # Handle simulation mode: return a dummy balance
        if self.simulation_mode:
            dummy_balance = 10000.0
            logger.warning(f"{Fore.YELLOW}[SIMULATION] Returning dummy balance {dummy_balance:.{self.price_decimals}f} {quote_currency}.{Style.RESET_ALL}")
            return dummy_balance

        try:
            # Add params if needed for specific account types (e.g., Bybit Unified)
            params = {}
            # if self.exchange_id.lower() == 'bybit': params = {'accountType': 'UNIFIED'} # Or CONTRACT / SPOT
            balance_data = self.exchange.fetch_balance(params=params)

            free_balance: Optional[Union[str, float]] = None # Balance might be string or float

            # --- Standard CCXT 'free' Balance ---
            if quote_currency in balance_data:
                 free_balance = balance_data[quote_currency].get('free')
                 if free_balance is not None:
                     logger.debug(f"Found standard 'free' balance for {quote_currency}: {free_balance}")

            # --- Alternative/Unified Account Check (Using 'info' - EXCHANGE SPECIFIC) ---
            # Check if 'free' was missing/zero, or if we suspect a unified account structure
            if free_balance is None or float(free_balance) == 0.0:
                logger.debug(f"'free' balance for {quote_currency} is zero or missing. Checking 'info' for alternatives (e.g., Bybit V5 unified)...")
                info_data = balance_data.get('info', {})

                # --- Bybit V5 Specific Logic (Example) ---
                if self.exchange_id.lower() == 'bybit' and isinstance(info_data, dict):
                    try:
                        result_list = info_data.get('result', {}).get('list', [])
                        if result_list and isinstance(result_list[0], dict):
                            account_info = result_list[0]
                            account_type = account_info.get('accountType')
                            logger.debug(f"Checking Bybit account info: Type={account_type}")

                            # Unified Margin/Trading Account Check
                            if account_type in ['UNIFIED', 'CONTRACT']: # Adjust based on your Bybit account type
                                # Look for total available balance first
                                unified_avail_str = account_info.get('totalAvailableBalance')
                                if unified_avail_str is not None:
                                    logger.debug(f"Found Bybit Unified/Contract 'totalAvailableBalance': {unified_avail_str}")
                                    free_balance = unified_avail_str
                                else:
                                    # Fallback: Check coin-specific available balance within Unified/Contract
                                    coin_list = account_info.get('coin', [])
                                    for coin_entry in coin_list:
                                        if isinstance(coin_entry, dict) and coin_entry.get('coin') == quote_currency:
                                            coin_avail = coin_entry.get('availableToWithdraw') or coin_entry.get('availableBalance') # Check multiple possible fields
                                            if coin_avail is not None:
                                                 logger.debug(f"Using coin-specific balance ('{quote_currency}') within {account_type} account: {coin_avail}")
                                                 free_balance = coin_avail
                                                 break # Found the specific coin balance

                            # SPOT Account Check (if relevant)
                            elif account_type == 'SPOT':
                                coin_list = account_info.get('coin', [])
                                for coin_entry in coin_list:
                                     if isinstance(coin_entry, dict) and coin_entry.get('coin') == quote_currency:
                                         spot_avail = coin_entry.get('free') # Spot usually uses 'free'
                                         if spot_avail is not None:
                                              logger.debug(f"Found Bybit SPOT balance for {quote_currency}: {spot_avail}")
                                              free_balance = spot_avail
                                              break
                    except (AttributeError, IndexError, KeyError, TypeError) as bybit_info_err:
                        logger.warning(f"{Fore.YELLOW}Could not parse expected Bybit structure in balance 'info': {bybit_info_err}. Using standard 'free' balance (or 0.0). Raw info: {info_data}{Style.RESET_ALL}")
                # --- Add logic for other exchanges' 'info' structure if needed ---
                # elif self.exchange_id.lower() == 'some_other_exchange':
                #     # Parse 'info' for that specific exchange
                #     pass

            # --- Final Conversion and Return ---
            final_balance = 0.0
            if free_balance is not None:
                try:
                    final_balance = float(free_balance)
                    if final_balance < 0:
                        logger.warning(f"Fetched balance for {quote_currency} is negative ({final_balance}). Treating as 0.0.")
                        final_balance = 0.0
                except (ValueError, TypeError) as conv_err:
                    logger.error(f"{Fore.RED}Could not convert final balance value '{free_balance}' to float for {quote_currency}: {conv_err}. Returning 0.0{Style.RESET_ALL}")
                    final_balance = 0.0
            else:
                logger.warning(f"{Fore.YELLOW}Could not determine available balance for {quote_currency} after checking standard and info fields. Returning 0.0.{Style.RESET_ALL}")


            logger.info(f"Fetched available balance: {final_balance:.{self.price_decimals}f} {quote_currency}") # Log balance with price precision for readability
            return final_balance

        except ccxt.AuthenticationError as e:
             # Should not happen if simulation mode check is done first, but handle defensively
             logger.error(f"{Fore.RED}Authentication failed fetching balance for {quote_currency}: {e}. Ensure API keys are valid and have permissions.{Style.RESET_ALL}")
             return 0.0
        except Exception as e:
            # Catch-all for unexpected errors during balance fetch
            logger.error(f"{Fore.RED}Unexpected error fetching balance for {quote_currency}: {e}{Style.RESET_ALL}", exc_info=True)
            return 0.0

    @retry_api_call(max_retries=1) # Typically don't retry 'fetch_order' aggressively if it fails
    def fetch_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetches the status and details of a specific order by its ID.

        Args:
            order_id: The exchange's ID for the order.
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            The CCXT order dictionary or None if not found or error occurs.
            The dictionary may contain an 'info' key with raw exchange data.
        """
        if not order_id:
            logger.warning("fetch_order_status called with empty order_id.")
            return None

        target_symbol = symbol or self.symbol
        logger.debug(f"Fetching status for order {order_id} ({target_symbol})...")

        # Handle simulation mode: return a placeholder if needed, or manage simulated orders
        if self.simulation_mode:
            # Find the simulated order in our state
            for pos in self.open_positions:
                 if pos['id'] == order_id:
                    logger.debug(f"[SIMULATION] Returning cached/simulated status for order {order_id}.")
                    # Construct a basic simulated order structure mimicking ccxt
                    sim_status = pos.get('status', STATUS_UNKNOWN)
                    sim_avg = pos['entry_price'] if sim_status == STATUS_ACTIVE else None
                    sim_filled = pos['size'] if sim_status == STATUS_ACTIVE else 0.0
                    sim_amount = pos['original_size']

                    return {
                        'id': order_id, 'symbol': target_symbol, 'status': sim_status,
                        'type': pos.get('entry_order_type', 'limit'), 'side': pos.get('side'),
                        'amount': sim_amount, 'filled': sim_filled, 'remaining': max(0, sim_amount - sim_filled),
                        'average': sim_avg, 'timestamp': pos.get('entry_time', time.time()) * 1000 if sim_status == STATUS_ACTIVE else time.time() * 1000,
                        'datetime': pd.to_datetime(pos.get('entry_time', time.time()), unit='s', utc=True).isoformat() if sim_status == STATUS_ACTIVE else pd.Timestamp.now(tz='UTC').isoformat(),
                        'stopLossPrice': pos.get('stop_loss_price'),
                        'takeProfitPrice': pos.get('take_profit_price'),
                        'info': {'simulated': True, 'orderId': order_id}
                    }
            logger.warning(f"[SIMULATION] Simulated order {order_id} not found in current state for status check.")
            return None # Not found in simulation state

        # --- Live Mode ---
        try:
            # Add params if needed (e.g., {'category': 'linear'} for Bybit V5)
            params = {}
            if self.exchange_id.lower() == 'bybit':
                market_type = self.exchange.market(target_symbol).get('type', 'swap')
                if market_type in ['swap', 'future']:
                    params['category'] = 'linear' if self.exchange.market(target_symbol).get('linear') else 'inverse'
                # elif market_type == 'spot': params['category'] = 'spot' # If needed

            order_info = self.exchange.fetch_order(order_id, target_symbol, params=params)

            # Log key details
            status = order_info.get('status', STATUS_UNKNOWN)
            filled = order_info.get('filled', 0.0)
            avg_price = order_info.get('average')
            remaining = order_info.get('remaining')
            amount = order_info.get('amount')
            order_type = order_info.get('type')
            order_side = order_info.get('side')

            logger.debug(f"Order {order_id} ({order_type} {order_side}): Status={status}, Filled={filled}/{amount}, AvgPrice={avg_price}, Remaining={remaining}")

            return order_info

        except ccxt.OrderNotFound:
            # This is common, treat as non-error, indicates order is final (closed/canceled) or never existed
            logger.warning(f"{Fore.YELLOW}Order {order_id} not found on exchange ({target_symbol}). Assumed closed, cancelled, or invalid ID.{Style.RESET_ALL}")
            return None # Return None, let the calling function decide how to interpret this
        except ccxt.NetworkError as e:
             # Allow retry decorator to handle network issues
             logger.warning(f"Network error fetching status for order {order_id}: {e}. Retrying if possible.")
             raise e # Re-raise to trigger retry logic in decorator
        except ccxt.ExchangeError as e:
             # Log other exchange errors, but usually don't retry fetch_order
             # Check if it's a potentially recoverable error string before logging as ERROR
             err_str = str(e).lower()
             if "order is finished" in err_str: # Bybit sometimes uses this instead of OrderNotFound
                  logger.warning(f"{Fore.YELLOW}Order {order_id} not found (reported as 'finished'). Assuming closed/cancelled. Error: {e}{Style.RESET_ALL}")
                  return None
             else:
                  logger.error(f"{Fore.RED}Exchange error fetching status for order {order_id}: {e}. Returning None.{Style.RESET_ALL}")
                  return None
        except Exception as e:
            # Catch unexpected errors
            logger.error(f"{Fore.RED}Unexpected error fetching status for order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
            return None


    # --- Indicator Calculation Methods ---
    # These methods are static as they operate purely on input data
    # Added basic error handling and checks for sufficient data length

    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> Optional[float]:
        """Calculates the rolling standard deviation of log returns."""
        if close_prices is None or window <= 0: return None
        min_len = window + 1
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for volatility (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            log_returns = np.log(close_prices / close_prices.shift(1))
            # Use pandas std() which calculates sample std dev by default (ddof=1)
            # Use ddof=0 for population std dev if preferred. Sample is more common for finance.
            volatility = log_returns.rolling(window=window, min_periods=window).std(ddof=1).iloc[-1]
            return float(volatility) if pd.notna(volatility) else None
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}", exc_info=False) # Avoid spamming logs
            return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates the Exponential Moving Average (EMA)."""
        if close_prices is None or period <= 0: return None
        # EMA needs 'period' points for decent initialization with adjust=False
        min_len = period
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for EMA (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            # adjust=False uses the recursive definition: EMA_today = alpha * Price_today + (1 - alpha) * EMA_yesterday
            ema = close_prices.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates the Relative Strength Index (RSI)."""
        if close_prices is None or period <= 0: return None
        min_len = period + 1 # Needs at least period+1 for the initial diff
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for RSI (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            # Use Exponential Moving Average for smoothing gains and losses (common method)
            avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
            # Alternative: Simple Moving Average
            # avg_gain = gain.rolling(window=period, min_periods=period).mean()
            # avg_loss = loss.rolling(window=period, min_periods=period).mean()

            # Calculate Relative Strength (RS)
            # Add epsilon to avoid division by zero
            rs = avg_gain / (avg_loss + 1e-12) # Use small epsilon

            # Calculate RSI
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_val = rsi.iloc[-1]
            return float(rsi_val) if pd.notna(rsi_val) else None
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_macd(close_prices: pd.Series, short_period: int, long_period: int, signal_period: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculates the Moving Average Convergence Divergence (MACD), Signal Line, and Histogram."""
        if close_prices is None or not all(p > 0 for p in [short_period, long_period, signal_period]): return None, None, None
        if short_period >= long_period:
             logger.error("MACD short_period must be less than long_period.")
             return None, None, None
        # Rough minimum length for meaningful values (long EMA needs init, signal EMA needs MACD series)
        min_len = long_period + signal_period
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for MACD (need ~{min_len}, got {len(close_prices)}).")
            return None, None, None
        try:
            ema_short = close_prices.ewm(span=short_period, adjust=False, min_periods=short_period).mean()
            ema_long = close_prices.ewm(span=long_period, adjust=False, min_periods=long_period).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
            histogram = macd_line - signal_line

            # Get the latest values
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            # Return None if any value is NaN (e.g., due to insufficient data at the end)
            if pd.isna(macd_val) or pd.isna(signal_val) or pd.isna(hist_val):
                return None, None, None
            return float(macd_val), float(signal_val), float(hist_val)
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=False)
            return None, None, None

    @staticmethod
    def calculate_stoch_rsi(close_prices: pd.Series, rsi_period: int, stoch_period: int, k_period: int, d_period: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculates the Stochastic RSI (%K and %D)."""
        if close_prices is None or not all(p > 0 for p in [rsi_period, stoch_period, k_period, d_period]): return None, None
        # Estimate minimum length: Need RSI series, then rolling Stoch window, then smoothing
        min_len = rsi_period + stoch_period + max(k_period, d_period)
        if len(close_prices) < min_len:
            logger.debug(f"Insufficient data for StochRSI (need ~{min_len}, got {len(close_prices)}).")
            return None, None
        try:
            # --- Calculate RSI Series ---
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean()
            loss = -delta.where(delta < 0, 0.0).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean()
            rs = gain / (loss + 1e-12) # Epsilon added
            rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna() # Drop initial NaNs from RSI calc

            if len(rsi_series) < stoch_period:
                logger.debug(f"Insufficient RSI series length for StochRSI window (need {stoch_period}, got {len(rsi_series)}).")
                return None, None

            # --- Calculate Stochastic of RSI ---
            min_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min()
            max_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max()
            # Add epsilon to denominator to avoid division by zero if max == min
            stoch_rsi_raw = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)).clip(0, 100) # Clip ensures 0-100 range

            # --- Calculate %K and %D ---
            # %K is typically a smoothed version of stoch_rsi_raw (SMA or EMA)
            stoch_k = stoch_rsi_raw.rolling(window=k_period, min_periods=k_period).mean()
            # %D is a smoothing of %K
            stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

            k_val, d_val = stoch_k.iloc[-1], stoch_d.iloc[-1]

            if pd.isna(k_val) or pd.isna(d_val):
                 return None, None
            return float(k_val), float(d_val)
        except Exception as e:
            logger.error(f"Error calculating StochRSI: {e}", exc_info=False)
            return None, None

    @staticmethod
    def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates the Average True Range (ATR)."""
        if high_prices is None or low_prices is None or close_prices is None or period <= 0: return None
        min_len = period + 1 # Needs prior close for TR calculation
        if len(high_prices) < min_len or len(low_prices) < min_len or len(close_prices) < min_len:
            logger.debug(f"Insufficient data for ATR (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))

            # Combine the three components to find the True Range (TR)
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False) # skipna=False propagates NaN if any component is NaN
            tr = tr.fillna(0) # Fill initial NaN from shift(1) with 0 for ATR calculation

            # Calculate ATR using Exponential Moving Average (common method)
            # Can also use Simple Moving Average (SMA) or Wilder's Smoothing
            atr = tr.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            # Alternative using Wilder's Smoothing (com = period - 1):
            # atr = tr.ewm(com=period-1, adjust=False, min_periods=period).mean().iloc[-1]

            return float(atr) if pd.notna(atr) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=False)
            return None


    # --- Trading Logic & Order Management Methods ---

    def calculate_order_size(self, current_price: float) -> float:
        """
        Calculates the order size in the BASE currency based on available balance,
        percentage risk, market limits, and optional volatility adjustment.

        Args:
            current_price: The current market price.

        Returns:
            The calculated order size in BASE currency (e.g., BTC, ETH),
            or 0.0 if checks fail or size is below minimum.
        """
        if self.market_info is None:
            logger.error("Market info not loaded, cannot calculate order size.")
            return 0.0
        if current_price <= 1e-12: # Avoid division by zero / nonsensical prices
             logger.error(f"Current price ({current_price}) is too low or invalid for order size calculation.")
             return 0.0

        quote_currency = self.market_info['quote']
        base_currency = self.market_info['base']

        # Fetch available balance
        balance = self.fetch_balance(currency_code=quote_currency)
        if balance <= 0:
            logger.warning(f"{Fore.YELLOW}Insufficient balance ({balance:.{self.price_decimals}f} {quote_currency}) for order size calculation. Need more {quote_currency}.{Style.RESET_ALL}")
            return 0.0

        # --- Calculate Base Order Size in Quote Currency ---
        base_order_size_quote = balance * self.order_size_percentage
        if base_order_size_quote <= 0:
            logger.warning(f"Calculated base order size is zero or negative ({base_order_size_quote:.{self.price_decimals}f} {quote_currency}). Balance: {balance}, Percentage: {self.order_size_percentage}")
            return 0.0

        final_size_quote = base_order_size_quote

        # --- Optional Volatility Adjustment ---
        if self.volatility_multiplier is not None and self.volatility_multiplier > 0:
            logger.debug("Applying volatility adjustment to order size...")
            # Fetch recent data specifically for volatility calculation
            # Requires fetching history again, consider passing indicators dict if available
            hist_data_vol = self.fetch_historical_data(limit=self.volatility_window + 5) # Fetch enough candles
            volatility = None
            if hist_data_vol is not None and not hist_data_vol.empty:
                 volatility = self.calculate_volatility(hist_data_vol['close'], self.volatility_window)

            if volatility is not None and volatility > 1e-9:
                # Inverse relationship: higher volatility -> smaller size factor
                # Adjust scaling factor as needed based on typical volatility values for the asset/timeframe
                # Example: If volatility is daily % (e.g., 0.02 for 2%), multiplier of 1 means 1 / (1 + 0.02*1) = ~0.98 size
                # Ensure multiplier makes sense for the scale of volatility value
                # Using *100 assumes volatility is like 0.01, adjust if needed.
                size_factor = 1 / (1 + volatility * self.volatility_multiplier * 100)
                # Clamp the factor to avoid extreme adjustments (e.g., 0.25x to 2.0x)
                size_factor = max(0.25, min(2.0, size_factor))
                final_size_quote = base_order_size_quote * size_factor
                logger.info(f"Volatility ({volatility:.5f}) adjustment factor: {size_factor:.3f}. "
                            f"Adjusted size: {final_size_quote:.{self.price_decimals}f} {quote_currency} (Base: {base_order_size_quote:.{self.price_decimals}f})")
            else:
                logger.debug(f"Volatility adjustment skipped: Volatility N/A ({volatility}) or multiplier is zero.")
        else:
            logger.debug("Volatility adjustment disabled.")

        # --- Convert Quote Size to Base Amount ---
        order_size_base = final_size_quote / current_price

        # --- Apply Exchange Precision and Check Limits ---
        try:
            # Apply Amount Precision using CCXT helper
            amount_precise = self.exchange.amount_to_precision(self.symbol, order_size_base)
            amount_precise = float(amount_precise) # Convert back to float

            # Apply Price Precision (for cost calculation estimate)
            price_precise = self.exchange.price_to_precision(self.symbol, current_price)
            price_precise = float(price_precise)

            # Check Minimum Amount Limit
            min_amount = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount is not None and amount_precise < min_amount:
                logger.warning(f"{Fore.YELLOW}Calculated order amount {amount_precise:.{self.amount_decimals}f} {base_currency} is below minimum required {min_amount}. Cannot place order.{Style.RESET_ALL}")
                return 0.0

            # Check Minimum Cost Limit (Estimated Cost)
            estimated_cost = amount_precise * price_precise
            min_cost = self.market_info.get('limits', {}).get('cost', {}).get('min')
            if min_cost is not None and estimated_cost < min_cost:
                logger.warning(f"{Fore.YELLOW}Estimated order cost {estimated_cost:.{self.price_decimals}f} {quote_currency} is below minimum required {min_cost}. Cannot place order. (Amt: {amount_precise}, Price: {price_precise}){Style.RESET_ALL}")
                return 0.0

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}Error applying precision or checking limits for order size: {e}{Style.RESET_ALL}", exc_info=True)
            return 0.0
        except Exception as e: # Catch any other calculation error
             logger.error(f"{Fore.RED}Unexpected error during order size precision/limit checks: {e}{Style.RESET_ALL}", exc_info=True)
             return 0.0

        # Log the final calculated size in BASE currency
        logger.info(f"{Fore.CYAN}Calculated final order size: {amount_precise:.{self.amount_decimals}f} {base_currency} "
                    f"(Value: ~{final_size_quote:.{self.price_decimals}f} {quote_currency}){Style.RESET_ALL}")

        # Return the final size in BASE currency, adjusted for precision and limits
        return amount_precise


    def _calculate_sl_tp_prices(self, entry_price: float, side: str, current_price: float, atr: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates Stop Loss (SL) and Take Profit (TP) prices based on configuration.
        Uses either ATR or fixed percentages from the entry price.
        Includes sanity checks against the current price and applies exchange precision.

        Args:
            entry_price: The estimated or actual entry price of the position.
            side: 'buy' or 'sell'.
            current_price: The current market price, used for sanity checks.
            atr: The current ATR value (required if use_atr_sl_tp is True).

        Returns:
            A tuple containing (stop_loss_price, take_profit_price) as floats,
            correctly formatted to the exchange's price precision.
            Values can be None if calculation is not possible or not configured.
        """
        stop_loss_price: Optional[float] = None
        take_profit_price: Optional[float] = None

        if self.market_info is None: # Should not happen
            logger.error("Market info not loaded, cannot calculate SL/TP.")
            return None, None

        # Helper to format price for logging using stored decimals
        def format_price(p):
            if p is None: return "N/A"
            return f"{p:.{self.price_decimals}f}"

        # --- Calculate based on ATR ---
        if self.use_atr_sl_tp:
            if atr is None or atr <= 1e-12: # Check for valid ATR
                logger.warning(f"{Fore.YELLOW}ATR SL/TP enabled, but ATR is invalid ({atr}). Cannot calculate SL/TP.{Style.RESET_ALL}")
                return None, None
            if not self.atr_sl_multiplier or not self.atr_tp_multiplier or self.atr_sl_multiplier <= 0 or self.atr_tp_multiplier <= 0:
                 logger.warning(f"{Fore.YELLOW}ATR SL/TP enabled, but multipliers (SL={self.atr_sl_multiplier}, TP={self.atr_tp_multiplier}) are invalid/non-positive. Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None

            logger.debug(f"Calculating SL/TP using ATR={atr:.5f}, SL Mult={self.atr_sl_multiplier}, TP Mult={self.atr_tp_multiplier}")
            sl_delta = atr * self.atr_sl_multiplier
            tp_delta = atr * self.atr_tp_multiplier

            if side == "buy":
                stop_loss_price = entry_price - sl_delta
                take_profit_price = entry_price + tp_delta
            elif side == "sell": # Short position
                stop_loss_price = entry_price + sl_delta
                take_profit_price = entry_price - tp_delta
            else:
                 logger.error(f"Invalid side '{side}' provided for SL/TP calculation.")
                 return None, None

        # --- Calculate based on Fixed Percentage ---
        else:
            if self.base_stop_loss_pct is None or self.base_take_profit_pct is None:
                 logger.warning(f"{Fore.YELLOW}Fixed Percentage SL/TP enabled, but 'stop_loss_percentage' or 'take_profit_percentage' not set/valid in config. Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None
            # Validation already happened in validate_config, but double check > 0
            if self.base_stop_loss_pct <= 0 or self.base_take_profit_pct <= 0:
                 logger.warning(f"{Fore.YELLOW}Fixed Percentage SL/TP values (SL={self.base_stop_loss_pct}, TP={self.base_take_profit_pct}) must be positive. Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None

            logger.debug(f"Calculating SL/TP using Fixed %: SL={self.base_stop_loss_pct*100:.2f}%, TP={self.base_take_profit_pct*100:.2f}%")
            if side == "buy":
                stop_loss_price = entry_price * (1 - self.base_stop_loss_pct)
                take_profit_price = entry_price * (1 + self.base_take_profit_pct)
            elif side == "sell": # Short position
                stop_loss_price = entry_price * (1 + self.base_stop_loss_pct)
                take_profit_price = entry_price * (1 - self.base_take_profit_pct)
            else:
                 logger.error(f"Invalid side '{side}' provided for SL/TP calculation.")
                 return None, None

        # --- Sanity Check & Price Precision Application ---
        # Apply precision BEFORE final checks to work with realistic values
        try:
            if stop_loss_price is not None:
                if stop_loss_price <= 1e-12: # Ensure positive price
                    logger.warning(f"Calculated stop loss price ({stop_loss_price}) is zero or negative. Setting SL to None.")
                    stop_loss_price = None
                else:
                    # Use price_to_precision to format correctly
                    stop_loss_price = float(self.exchange.price_to_precision(self.symbol, stop_loss_price))

            if take_profit_price is not None:
                if take_profit_price <= 1e-12: # Ensure positive price
                     logger.warning(f"Calculated take profit price ({take_profit_price}) is zero or negative. Setting TP to None.")
                     take_profit_price = None
                else:
                    take_profit_price = float(self.exchange.price_to_precision(self.symbol, take_profit_price))

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Error applying precision to SL/TP prices: {e}. Original SL={stop_loss_price}, TP={take_profit_price}. Setting both to None.{Style.RESET_ALL}")
            return None, None
        except Exception as e: # Catch other unexpected errors
             logger.error(f"{Fore.RED}Unexpected error applying precision to SL/TP: {e}. Setting both to None.{Style.RESET_ALL}", exc_info=True)
             return None, None


        # --- Final Sanity Check & Adjustment against Current Price ---
        # Prevent placing SL/TP orders that would trigger immediately or are illogical
        if stop_loss_price is not None:
             # Allow a small buffer (e.g., 0.1% of price or 1 tick size)
             buffer = max(current_price * 0.0005, float(self.market_info['precision']['price'] or 0.0001))
             if side == "buy" and stop_loss_price >= current_price - buffer:
                 adj_sl = current_price * (1 - (self.base_stop_loss_pct or 0.001)) # Fallback adjustment
                 adj_sl = min(adj_sl, entry_price - buffer) # Ensure below entry too
                 logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) too close to current price ({format_price(current_price)}) for LONG. Adjusting SL to avoid immediate trigger (new estimate: {format_price(adj_sl)}). Consider wider SL settings.{Style.RESET_ALL}")
                 stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl)) if adj_sl > 0 else None
             elif side == "sell" and stop_loss_price <= current_price + buffer:
                  adj_sl = current_price * (1 + (self.base_stop_loss_pct or 0.001)) # Fallback adjustment
                  adj_sl = max(adj_sl, entry_price + buffer) # Ensure above entry too
                  logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) too close to current price ({format_price(current_price)}) for SHORT. Adjusting SL to avoid immediate trigger (new estimate: {format_price(adj_sl)}). Consider wider SL settings.{Style.RESET_ALL}")
                  stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl)) if adj_sl > 0 else None

        if take_profit_price is not None:
            buffer = max(current_price * 0.0005, float(self.market_info['precision']['price'] or 0.0001))
            if side == "buy" and take_profit_price <= current_price + buffer:
                 adj_tp = current_price * (1 + (self.base_take_profit_pct or 0.001)) # Fallback adjustment
                 adj_tp = max(adj_tp, entry_price + buffer*2) # Ensure above entry+buffer
                 logger.warning(f"{Fore.YELLOW}Calculated TP ({format_price(take_profit_price)}) too close to current price ({format_price(current_price)}) for LONG. Adjusting TP (new estimate: {format_price(adj_tp)}). Consider wider TP settings.{Style.RESET_ALL}")
                 take_profit_price = float(self.exchange.price_to_precision(self.symbol, adj_tp)) if adj_tp > 0 else None
            elif side == "sell" and take_profit_price >= current_price - buffer:
                 adj_tp = current_price * (1 - (self.base_take_profit_pct or 0.001)) # Fallback adjustment
                 adj_tp = min(adj_tp, entry_price - buffer*2) # Ensure below entry-buffer
                 logger.warning(f"{Fore.YELLOW}Calculated TP ({format_price(take_profit_price)}) too close to current price ({format_price(current_price)}) for SHORT. Adjusting TP (new estimate: {format_price(adj_tp)}). Consider wider TP settings.{Style.RESET_ALL}")
                 take_profit_price = float(self.exchange.price_to_precision(self.symbol, adj_tp)) if adj_tp > 0 else None

        # Final check: Ensure SL and TP are logical relative to entry and each other
        if stop_loss_price is not None and take_profit_price is not None:
             if side == "buy" and stop_loss_price >= take_profit_price:
                  logger.warning(f"{Fore.YELLOW}Final check: Calculated SL ({format_price(stop_loss_price)}) >= TP ({format_price(take_profit_price)}) for LONG. Setting TP to None to avoid conflict.{Style.RESET_ALL}")
                  take_profit_price = None
             elif side == "sell" and stop_loss_price <= take_profit_price:
                  logger.warning(f"{Fore.YELLOW}Final check: Calculated SL ({format_price(stop_loss_price)}) <= TP ({format_price(take_profit_price)}) for SHORT. Setting TP to None to avoid conflict.{Style.RESET_ALL}")
                  take_profit_price = None
        # Check relative to entry price
        if stop_loss_price is not None:
            if side == 'buy' and stop_loss_price >= entry_price:
                logger.warning(f"{Fore.YELLOW}Final check: Calculated SL ({format_price(stop_loss_price)}) >= Entry ({format_price(entry_price)}) for LONG. Setting SL to None.{Style.RESET_ALL}")
                stop_loss_price = None
            elif side == 'sell' and stop_loss_price <= entry_price:
                logger.warning(f"{Fore.YELLOW}Final check: Calculated SL ({format_price(stop_loss_price)}) <= Entry ({format_price(entry_price)}) for SHORT. Setting SL to None.{Style.RESET_ALL}")
                stop_loss_price = None
        if take_profit_price is not None:
            if side == 'buy' and take_profit_price <= entry_price:
                logger.warning(f"{Fore.YELLOW}Final check: Calculated TP ({format_price(take_profit_price)}) <= Entry ({format_price(entry_price)}) for LONG. Setting TP to None.{Style.RESET_ALL}")
                take_profit_price = None
            elif side == 'sell' and take_profit_price >= entry_price:
                logger.warning(f"{Fore.YELLOW}Final check: Calculated TP ({format_price(take_profit_price)}) >= Entry ({format_price(entry_price)}) for SHORT. Setting TP to None.{Style.RESET_ALL}")
                take_profit_price = None


        logger.debug(f"Final calculated SL={format_price(stop_loss_price)}, TP={format_price(take_profit_price)} for {side} entry near {format_price(entry_price)}")
        return stop_loss_price, take_profit_price


    def compute_trade_signal_score(self, price: float, indicators: Dict[str, Optional[Union[float, Tuple[Optional[float], ...]]]], orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        """
        Computes a simple trade signal score based on configured indicators and order book imbalance.

        Score Contributions (Example - adjust weights as needed):
        - Order Book Imbalance: +1.0 (Buy Pressure), -1.0 (Sell Pressure)
        - Price vs EMA:        +1.0 (Above), -1.0 (Below)
        - RSI:                 +1.0 (Oversold), -1.0 (Overbought), +0.5 (Rising from OS), -0.5 (Falling from OB)
        - MACD Crossover:      +1.0 (Bullish Cross), -1.0 (Bearish Cross)
        - StochRSI:            +1.0 (Oversold), -1.0 (Overbought), +0.5 (K crossing D upwards), -0.5 (K crossing D downwards)

        Args:
            price: Current market price.
            indicators: Dictionary of calculated indicator values.
            orderbook_imbalance: Calculated order book imbalance ratio (Ask Vol / Bid Vol).

        Returns:
            A tuple containing the final integer score and a list of string reasons for the score components.
        """
        score = 0.0
        reasons = []
        # Define thresholds (consider making these configurable)
        RSI_OVERSOLD, RSI_OVERBOUGHT = 35, 65
        STOCH_OVERSOLD, STOCH_OVERBOUGHT = 25, 75
        EMA_THRESHOLD_MULTIPLIER = 0.0002 # Price must be 0.02% above/below EMA

        # --- 1. Order Book Imbalance ---
        if orderbook_imbalance is not None:
            if self.imbalance_threshold <= 0:
                 reasons.append(f"{Fore.WHITE}[ 0.0] OB Invalid Threshold ({self.imbalance_threshold}){Style.RESET_ALL}")
            else:
                 imb_buy_thresh = 1.0 / self.imbalance_threshold # e.g., if thresh=1.5, buy_thresh=0.67
                 imb = orderbook_imbalance
                 if imb < imb_buy_thresh:
                     score += 1.0
                     reasons.append(f"{Fore.GREEN}[+1.0] OB Buy Pressure (Imb: {imb:.2f} < {imb_buy_thresh:.2f}){Style.RESET_ALL}")
                 elif imb > self.imbalance_threshold:
                     score -= 1.0
                     reasons.append(f"{Fore.RED}[-1.0] OB Sell Pressure (Imb: {imb:.2f} > {self.imbalance_threshold:.2f}){Style.RESET_ALL}")
                 else: # Between thresholds
                     reasons.append(f"{Fore.WHITE}[ 0.0] OB Balanced (Imb: {imb:.2f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] OB Data N/A{Style.RESET_ALL}")

        # --- 2. EMA Trend ---
        ema = indicators.get('ema')
        if ema is not None and ema > 1e-9: # Check if EMA is valid and not zero
            ema_upper_bound = ema * (1 + EMA_THRESHOLD_MULTIPLIER)
            ema_lower_bound = ema * (1 - EMA_THRESHOLD_MULTIPLIER)
            price_f = f"{price:.{self.price_decimals}f}" # Formatted price
            ema_f = f"{ema:.{self.price_decimals}f}"     # Formatted EMA
            if price > ema_upper_bound:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] Price > EMA ({price_f} > {ema_f}){Style.RESET_ALL}")
            elif price < ema_lower_bound:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] Price < EMA ({price_f} < {ema_f}){Style.RESET_ALL}")
            else: # Price is within the threshold range of EMA
                reasons.append(f"{Fore.WHITE}[ 0.0] Price near EMA ({price_f} ~ {ema_f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}")

        # --- 3. RSI Momentum/OB/OS ---
        rsi = indicators.get('rsi')
        if rsi is not None:
            rsi_f = f"{rsi:.1f}"
            if rsi < RSI_OVERSOLD:
                score += 1.0 # Strong buy signal in oversold
                reasons.append(f"{Fore.GREEN}[+1.0] RSI Oversold ({rsi_f} < {RSI_OVERSOLD}){Style.RESET_ALL}")
            elif rsi > RSI_OVERBOUGHT:
                score -= 1.0 # Strong sell signal in overbought
                reasons.append(f"{Fore.RED}[-1.0] RSI Overbought ({rsi_f} > {RSI_OVERBOUGHT}){Style.RESET_ALL}")
            # Optional: Add weaker signals for moving out of zones (needs previous RSI value)
            # elif prev_rsi < RSI_OVERSOLD and rsi >= RSI_OVERSOLD: score += 0.5 ...
            else: # RSI is in the neutral zone
                reasons.append(f"{Fore.WHITE}[ 0.0] RSI Neutral ({RSI_OVERSOLD} <= {rsi_f} <= {RSI_OVERBOUGHT}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}")

        # --- 4. MACD Momentum/Cross ---
        macd_line, macd_signal = indicators.get('macd_line'), indicators.get('macd_signal')
        # Check if both values are valid numbers
        if macd_line is not None and macd_signal is not None:
            macd_f = f"{macd_line:.{self.price_decimals}f}" # Use price decimals for MACD value formatting
            sig_f = f"{macd_signal:.{self.price_decimals}f}"
            # Basic check: MACD line relative to signal line
            if macd_line > macd_signal:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] MACD Line > Signal ({macd_f} > {sig_f}){Style.RESET_ALL}")
            else: # MACD <= Signal
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] MACD Line <= Signal ({macd_f} <= {sig_f}){Style.RESET_ALL}")
            # Optional: Add crossover detection (needs previous values)
            # Optional: Consider MACD histogram value/slope
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}")

        # --- 5. Stochastic RSI OB/OS & Cross ---
        stoch_k, stoch_d = indicators.get('stoch_k'), indicators.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            k_f, d_f = f"{stoch_k:.1f}", f"{stoch_d:.1f}"
            # Condition for Oversold: Both K and D below threshold (strong signal)
            if stoch_k < STOCH_OVERSOLD and stoch_d < STOCH_OVERSOLD:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] StochRSI Oversold (K={k_f}, D={d_f} < {STOCH_OVERSOLD}){Style.RESET_ALL}")
            # Condition for Overbought: Both K and D above threshold (strong signal)
            elif stoch_k > STOCH_OVERBOUGHT and stoch_d > STOCH_OVERBOUGHT:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] StochRSI Overbought (K={k_f}, D={d_f} > {STOCH_OVERBOUGHT}){Style.RESET_ALL}")
            # Optional: Check for crossover signals (weaker signals)
            # elif prev_k < prev_d and k >= d: score += 0.5 ... (needs previous K/D)
            else: # StochRSI is in the neutral zone or K/D are crossing
                reasons.append(f"{Fore.WHITE}[ 0.0] StochRSI Neutral (K={k_f}, D={d_f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}")

        # --- Final Score Calculation ---
        # Round the raw score to the nearest integer
        final_score = int(round(score))
        logger.debug(f"Signal score calculation complete. Raw Score: {score:.2f}, Final Integer Score: {final_score}")

        return final_score, reasons


    @retry_api_call(max_retries=2, initial_delay=2) # Retry order placement a couple of times
    def place_entry_order(
        self, side: str, order_size_base: float, confidence_level: int,
        order_type: str, current_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Places an entry order (market or limit) on the exchange.
        Includes Bybit V5 specific parameters for SL/TP if provided.

        Args:
            side: 'buy' or 'sell'.
            order_size_base: The desired order size in the BASE currency (e.g., BTC amount).
            confidence_level: The signal score associated with this entry.
            order_type: 'market' or 'limit'.
            current_price: The current market price (used for limit offset).
            stop_loss_price: The calculated stop loss price (float) to send with the order.
            take_profit_price: The calculated take profit price (float) to send with the order.

        Returns:
            The CCXT order dictionary if the order was placed successfully (or simulated),
            otherwise None. Includes a 'bot_custom_info' key with confidence level.
        """
        if self.market_info is None:
             logger.error("Market info not available. Cannot place entry order.")
             return None
        if order_size_base <= 0:
             logger.error("Order size (base) is zero or negative. Cannot place entry order.")
             return None
        if current_price <= 1e-12:
             logger.error("Current price is zero or negative. Cannot place entry order.")
             return None

        base_currency = self.market_info['base']
        quote_currency = self.market_info['quote']

        params = {} # Dictionary for extra parameters (like SL/TP)
        limit_price: Optional[float] = None # For limit orders

        try:
            # --- Ensure Amount is Precise (already done in calculate_order_size) ---
            # order_size_base is assumed to be already precise and limit-checked
            amount_precise = order_size_base
            logger.debug(f"Using precise entry amount: {amount_precise:.{self.amount_decimals}f} {base_currency}")

            # --- Calculate Precise Limit Price if Applicable ---
            if order_type == "limit":
                offset = self.limit_order_entry_offset_pct_buy if side == 'buy' else self.limit_order_entry_offset_pct_sell
                # Buy limit below current price, Sell limit above current price
                price_factor = (1 - offset) if side == 'buy' else (1 + offset)
                limit_price_raw = current_price * price_factor
                if limit_price_raw <= 1e-12:
                     logger.error(f"Calculated limit price ({limit_price_raw}) is zero or negative. Cannot place limit order.")
                     return None
                limit_price = float(self.exchange.price_to_precision(self.symbol, limit_price_raw))
                logger.debug(f"Precise limit price: {limit_price:.{self.price_decimals}f} {quote_currency}")

            # --- Add Bybit V5 SL/TP & Trigger Parameters ---
            # Prices should already be precise floats from _calculate_sl_tp_prices
            if stop_loss_price is not None:
                params['stopLoss'] = str(stop_loss_price) # Bybit V5 often expects string format for prices in params
                if self.sl_trigger_by: params['slTriggerBy'] = self.sl_trigger_by
                logger.debug(f"Adding SL param: stopLoss={params['stopLoss']}, slTriggerBy={params.get('slTriggerBy')}")
            if take_profit_price is not None:
                params['takeProfit'] = str(take_profit_price) # Use string format
                if self.tp_trigger_by: params['tpTriggerBy'] = self.tp_trigger_by
                logger.debug(f"Adding TP param: takeProfit={params['takeProfit']}, tpTriggerBy={params.get('tpTriggerBy')}")

            # Bybit V5 might need category for derivatives
            if 'bybit' in self.exchange_id.lower():
                market_type = self.exchange.market(self.symbol).get('type', 'swap')
                if market_type in ['swap', 'future']:
                    params['category'] = 'linear' if self.exchange.market(self.symbol).get('linear') else 'inverse'


            # --- Final Pre-flight Checks (Limits - redundant if checked in calc_size, but safe) ---
            # These were checked during calculate_order_size, but double-check amount here
            min_amount = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount is not None and amount_precise < min_amount:
                 logger.warning(f"{Fore.YELLOW}Final check: Entry amount {amount_precise:.{self.amount_decimals}f} {base_currency} is below minimum {min_amount}. Aborting placement.{Style.RESET_ALL}")
                 return None
            # Cost check is harder without exact fill price, rely on initial check

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}Error preparing entry order values/parameters: {e}{Style.RESET_ALL}", exc_info=True)
            return None
        except Exception as e: # Catch other errors
             logger.error(f"{Fore.RED}Unexpected error preparing entry order: {e}{Style.RESET_ALL}", exc_info=True)
             return None

        # --- Log Order Details ---
        log_color = Fore.GREEN if side == 'buy' else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_info = f"SL={params.get('stopLoss', 'N/A')}" + (f" ({params['slTriggerBy']})" if 'slTriggerBy' in params else "")
        tp_info = f"TP={params.get('takeProfit', 'N/A')}" + (f" ({params['tpTriggerBy']})" if 'tpTriggerBy' in params else "")
        limit_price_info = f"Limit={limit_price:.{self.price_decimals}f}" if limit_price else f"Market (~{current_price:.{self.price_decimals}f})"
        estimated_value_quote = amount_precise * (limit_price or current_price)

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            # Simulate fill price: use limit price if set, otherwise current price
            sim_entry_price = limit_price if order_type == "limit" else current_price
            # Simulate status: 'open' for limit, 'closed' (filled) for market
            sim_status = 'open' if order_type == 'limit' else 'closed'
            sim_filled = amount_precise if sim_status == 'closed' else 0.0
            sim_remaining = 0.0 if sim_status == 'closed' else amount_precise
            sim_cost = sim_filled * sim_entry_price if sim_status == 'closed' else 0.0

            simulated_order = {
                "id": sim_id, "clientOrderId": sim_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.to_datetime('now', utc=True).isoformat(),
                "symbol": self.symbol, "type": order_type, "side": side,
                "amount": amount_precise,
                "price": limit_price, # Price of the limit order itself
                "average": sim_entry_price if sim_status == 'closed' else None, # Fill price
                "cost": sim_cost, # Filled cost
                "status": sim_status,
                "filled": sim_filled,
                "remaining": sim_remaining,
                # Store SL/TP info as they would have been sent
                "stopLossPrice": stop_loss_price,
                "takeProfitPrice": take_profit_price,
                "info": { # Mimic CCXT structure
                    "simulated": True, "orderId": sim_id,
                    "confidence": confidence_level,
                    "initial_base_size": order_size_base,
                    "stopLoss": params.get('stopLoss'), # Store string param if sent
                    "takeProfit": params.get('takeProfit'),
                    "slTriggerBy": params.get('slTriggerBy'),
                    "tpTriggerBy": params.get('tpTriggerBy'),
                    "category": params.get('category'),
                    "reduceOnly": params.get('reduceOnly', False)
                }
            }
            logger.info(
                f"{log_color}[SIMULATION] Placing {action_desc}: "
                f"ID: {sim_id}, Size: {amount_precise:.{self.amount_decimals}f} {base_currency}, "
                f"Price: {limit_price_info}, "
                f"Est. Value: {estimated_value_quote:.{self.price_decimals}f} {quote_currency}, Confidence: {confidence_level}, {sl_info}, {tp_info}{Style.RESET_ALL}"
            )
            # Add custom info directly for state management consistency
            simulated_order['bot_custom_info'] = {"confidence": confidence_level, "initial_base_size": order_size_base}
            return simulated_order

        # --- Live Trading Mode ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order...")
            logger.info(f" -> Size: {amount_precise:.{self.amount_decimals}f} {base_currency}")
            logger.info(f" -> Price: {limit_price_info}")
            logger.info(f" -> Value: ~{estimated_value_quote:.{self.price_decimals}f} {quote_currency}")
            logger.info(f" -> Params: {params}") # Includes SL/TP, category etc.

            order: Optional[Dict[str, Any]] = None
            try:
                if order_type == "market":
                    # Note: Bybit V5 might not accept price for market orders, rely on params
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    if limit_price is None: # Should not happen due to earlier checks
                         logger.error(f"{Fore.RED}Limit price not calculated. Cannot place live limit order.{Style.RESET_ALL}")
                         return None
                    order = self.exchange.create_limit_order(self.symbol, side, amount_precise, limit_price, params=params)

                # Process the order response
                if order:
                    oid = order.get('id', 'N/A')
                    otype = order.get('type', 'N/A')
                    oside = (order.get('side') or 'N/A').upper()
                    oamt = order.get('amount', 0.0)
                    ofilled = order.get('filled', 0.0)
                    oprice = order.get('price') # Original limit price
                    oavg = order.get('average') # Average fill price
                    ocost = order.get('cost', 0.0)
                    ostatus = order.get('status', STATUS_UNKNOWN)

                    # Try to get SL/TP info back from the order response's 'info' field for confirmation
                    # Field names vary by exchange (e.g., Bybit might use 'stopLoss', 'takeProfit')
                    info_sl = order.get('info', {}).get('stopLoss', 'N/A')
                    info_tp = order.get('info', {}).get('takeProfit', 'N/A')
                    info_sl_trig = order.get('info', {}).get('slTriggerBy', 'N/A')
                    info_tp_trig = order.get('info', {}).get('tpTriggerBy', 'N/A')

                    log_price_details = f"Limit: {oprice:.{self.price_decimals}f}" if oprice else "Market"
                    if oavg: log_price_details += f", AvgFill: {oavg:.{self.price_decimals}f}"

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: "
                        f"ID: {oid}, Type: {otype}, Side: {oside}, "
                        f"Amount: {oamt:.{self.amount_decimals}f}, Filled: {ofilled:.{self.amount_decimals}f}, "
                        f"{log_price_details}, Cost: {ocost:.{self.price_decimals}f} {quote_currency}, "
                        f"Status: {ostatus}, "
                        f"SL Confirmed: {info_sl} ({info_sl_trig}), TP Confirmed: {info_tp} ({info_tp_trig}), "
                        f"Confidence: {confidence_level}{Style.RESET_ALL}"
                    )
                    # Store bot-specific context with the order dict before returning
                    order['bot_custom_info'] = {"confidence": confidence_level, "initial_base_size": order_size_base}
                    return order
                else:
                    # This case might occur if the API call succeeds but returns an empty/None response
                    logger.error(f"{Fore.RED}LIVE {action_desc} order placement potentially failed: API call succeeded but returned None or empty response.{Style.RESET_ALL}")
                    return None

            # Handle specific, potentially non-retryable errors based on decorator logic
            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                # These are typically logged as ERROR by the retry decorator already if non-retryable
                logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed ({type(e).__name__}): {e}. Check balance or order parameters.{Style.RESET_ALL}")
                if isinstance(e, ccxt.InvalidOrder): logger.error(f" -> Params Sent: {params}")
                # Re-fetch balance after insufficient funds error for logging
                if isinstance(e, ccxt.InsufficientFunds): self.fetch_balance(quote_currency)
                return None # Explicitly return None for these failures
            except ccxt.ExchangeError as e:
                 # Catch other exchange errors not handled by the specific cases above or in the decorator
                 logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed (ExchangeError): {e}{Style.RESET_ALL}")
                 logger.error(f" -> Params Sent: {params}")
                 return None
            except Exception as e:
                # Catch any unexpected Python errors during the process
                logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed (Unexpected Python Error): {e}{Style.RESET_ALL}", exc_info=True)
                logger.error(f" -> Params Sent: {params}")
                return None


    @retry_api_call(max_retries=1) # Only retry cancellation once if it fails initially
    def cancel_order_by_id(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Cancels a single open order by its ID.

        Args:
            order_id: The exchange's ID for the order to cancel.
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            True if the order was successfully cancelled or already not found/closed,
            False otherwise.
        """
        if not order_id:
            logger.warning("cancel_order_by_id called with empty order_id.")
            return False

        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}Attempting to cancel order {order_id} ({target_symbol})...{Style.RESET_ALL}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            logger.info(f"[SIMULATION] Simulating cancellation of order {order_id}.")
            # Find and update the simulated order status in our state
            found_and_cancelled = False
            for pos in self.open_positions:
                 # Only cancel orders that are actually pending
                 if pos['id'] == order_id and pos['status'] == STATUS_PENDING_ENTRY:
                     pos['status'] = STATUS_CANCELED # Mark as canceled in simulation
                     pos['last_update_time'] = time.time()
                     logger.info(f"[SIMULATION] Marked simulated order {order_id} as canceled.")
                     found_and_cancelled = True
                     break
            if not found_and_cancelled:
                 logger.warning(f"[SIMULATION] Simulated order {order_id} not found or not in cancellable state ({STATUS_PENDING_ENTRY}).")
            # In simulation, return True if we marked it cancelled or if it wasn't pending anyway
            self._save_state() # Save updated state if changed
            return True

        # --- Live Mode ---
        else:
            try:
                # Add params if needed (e.g., Bybit V5 category)
                params = {}
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    if market_type in ['swap', 'future']:
                        params['category'] = 'linear' if self.exchange.market(target_symbol).get('linear') else 'inverse'
                    # elif market_type == 'spot': params['category'] = 'spot'

                response = self.exchange.cancel_order(order_id, target_symbol, params=params)
                logger.info(f"{Fore.YELLOW}---> Successfully initiated cancellation for order {order_id}. Response: {response}{Style.RESET_ALL}")
                return True
            except ccxt.OrderNotFound:
                # If the order is already gone (filled, cancelled, expired), consider it a success.
                logger.warning(f"{Fore.YELLOW}Order {order_id} not found during cancellation attempt (already closed/cancelled?). Treating as success.{Style.RESET_ALL}")
                return True
            except ccxt.NetworkError as e:
                 # Let the retry decorator handle transient network issues
                 logger.warning(f"Network error cancelling order {order_id}: {e}. Retrying if possible.")
                 raise e # Re-raise to trigger retry
            except ccxt.ExchangeError as e:
                 # Log specific exchange errors during cancellation more prominently
                 # These are often non-retryable (e.g., "order already filled")
                 err_str = str(e).lower()
                 if any(phrase in err_str for phrase in ["order has been filled", "order is finished", "already closed", "already cancelled"]):
                      logger.warning(f"{Fore.YELLOW}Cannot cancel order {order_id}: Already filled or finished. Error: {e}{Style.RESET_ALL}")
                      return True # Treat as 'success' in the sense that it doesn't need cancelling
                 else:
                      logger.error(f"{Fore.RED}Exchange error cancelling order {order_id}: {e}{Style.RESET_ALL}")
                      return False # Cancellation failed due to exchange state/rules
            except Exception as e:
                # Catch unexpected Python errors
                logger.error(f"{Fore.RED}Unexpected error cancelling order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                return False

    @retry_api_call() # Retry fetching open orders or cancelAll
    def cancel_all_symbol_orders(self, symbol: Optional[str] = None) -> int:
        """
        Attempts to cancel all open orders for the specified symbol.
        Prefers `fetchOpenOrders` and individual cancellation, falls back to `cancelAllOrders`.

        Args:
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            The number of orders successfully cancelled (or -1 if `cancelAllOrders` was used
            and count is unknown). Returns 0 if no orders were found or cancellation failed.
        """
        target_symbol = symbol or self.symbol
        logger.info(f"Checking for and attempting to cancel all open orders for {target_symbol}...")
        cancelled_count = 0

        # --- Simulation Mode ---
        if self.simulation_mode:
             logger.info(f"[SIMULATION] Simulating cancellation of all PENDING orders for {target_symbol}.")
             sim_cancelled = 0
             for pos in self.open_positions:
                  if pos.get('symbol') == target_symbol and pos.get('status') == STATUS_PENDING_ENTRY:
                      pos['status'] = STATUS_CANCELED
                      pos['last_update_time'] = time.time()
                      sim_cancelled += 1
             if sim_cancelled > 0:
                 logger.info(f"[SIMULATION] Marked {sim_cancelled} pending simulated orders as canceled.")
                 self._save_state()
             else:
                 logger.info(f"[SIMULATION] No pending simulated orders found for {target_symbol} to cancel.")
             return sim_cancelled

        # --- Live Mode ---
        try:
            # Prefer fetching open orders and cancelling individually for better feedback
            if self.exchange.has['fetchOpenOrders']:
                logger.debug(f"Fetching open orders for {target_symbol}...")
                # Add params if needed e.g. {'category':'linear'} for Bybit derivatives
                params = {}
                if self.exchange_id.lower() == 'bybit':
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    if market_type in ['swap', 'future']:
                        params['category'] = 'linear' if self.exchange.market(target_symbol).get('linear') else 'inverse'

                open_orders = self.exchange.fetch_open_orders(target_symbol, params=params)

                if not open_orders:
                    logger.info(f"No open orders found for {target_symbol}.")
                    return 0

                logger.warning(f"{Fore.YELLOW}Found {len(open_orders)} open order(s) for {target_symbol}. Attempting individual cancellation...{Style.RESET_ALL}")
                for order in open_orders:
                    order_id = order.get('id')
                    if not order_id:
                        logger.warning(f"Found open order without ID: {order}. Skipping cancellation.")
                        continue

                    # Use the individual cancel function (which handles simulation/live)
                    if self.cancel_order_by_id(order_id, target_symbol):
                        cancelled_count += 1
                        # Add small delay to avoid rate limits if cancelling many orders
                        time.sleep(max(0.2, self.exchange.rateLimit / 1000))
                    else:
                        logger.error(f"Failed to cancel order {order_id} during bulk cancellation. Continuing...")

            # Fallback to cancelAllOrders if fetchOpenOrders is not supported or preferred
            elif self.exchange.has['cancelAllOrders']:
                 logger.warning(f"{Fore.YELLOW}Exchange lacks 'fetchOpenOrders', attempting unified 'cancelAllOrders' for {target_symbol}...{Style.RESET_ALL}")
                 # Add params if needed
                 params = {}
                 if self.exchange_id.lower() == 'bybit':
                     market_type = self.exchange.market(target_symbol).get('type', 'swap')
                     if market_type in ['swap', 'future']:
                         params['category'] = 'linear' if self.exchange.market(target_symbol).get('linear') else 'inverse'

                 response = self.exchange.cancel_all_orders(target_symbol, params=params)
                 logger.info(f"'cancelAllOrders' response: {response}")
                 # We don't know the exact count from this response typically
                 cancelled_count = -1 # Indicate unknown count, but action was attempted
            else:
                 # If neither method is supported, we cannot proceed
                 logger.error(f"{Fore.RED}Exchange does not support 'fetchOpenOrders' or 'cancelAllOrders' for {target_symbol}. Cannot cancel orders automatically.{Style.RESET_ALL}")
                 return 0

            final_msg = f"Order cancellation process finished for {target_symbol}. "
            if cancelled_count >= 0:
                 final_msg += f"Successfully cancelled: {cancelled_count} order(s)."
            else:
                 final_msg += "'cancelAllOrders' attempted (exact count unknown)."
            logger.info(final_msg)
            return cancelled_count

        except ccxt.AuthenticationError as e:
             logger.error(f"{Fore.RED}Authentication error during bulk order cancellation for {target_symbol}: {e}{Style.RESET_ALL}")
             return cancelled_count # Return count successfully cancelled before error
        except Exception as e:
            # Catch any other errors during the fetch or cancel loop
            logger.error(f"{Fore.RED}Error during bulk order cancellation for {target_symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return cancelled_count # Return count successfully cancelled before error


    # --- Position Management & State Update Logic ---

    def _check_pending_entries(self, indicators: Dict) -> None:
        """
        Checks the status of any pending limit entry orders.
        If an order is filled, updates its status to 'active', records entry price/time,
        and recalculates/stores SL/TP based on the actual fill price.
        Removes failed (canceled/rejected) or vanished pending orders from the state.

        Args:
            indicators: Dictionary containing current indicator values (e.g., ATR for SL/TP recalc).
        """
        pending_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_PENDING_ENTRY]
        if not pending_positions:
            return # No pending orders to check

        logger.debug(f"Checking status of {len(pending_positions)} pending entry order(s)...")
        # Fetch current price once if needed for SL/TP recalculation on fill
        current_price_for_check: Optional[float] = None
        needs_price_check = any(self.use_atr_sl_tp or not self.use_atr_sl_tp for pos in pending_positions) # Check if SL/TP calc needed
        if needs_price_check:
             current_price_for_check = self.fetch_market_price()
             if current_price_for_check is None:
                  logger.warning("Could not fetch current price for pending entry check/SL/TP update. SL/TP might be based on estimate.")

        positions_to_remove_ids = set()
        positions_to_update_data = {} # Dict[order_id, updated_position_dict]

        for position in pending_positions:
            entry_order_id = position.get('id')
            if not entry_order_id: # Safety check
                logger.error(f"Found pending position with no ID: {position}. Marking for removal.")
                # How to remove if ID is missing? Need a robust way or just log. For now, skip.
                continue

            order_info = self.fetch_order_status(entry_order_id, symbol=position.get('symbol'))

            # Case 1: Order Not Found or Finished Externally
            if order_info is None:
                # Could be filled and gone, cancelled externally, or ID issue. Assume it's no longer pending.
                logger.warning(f"{Fore.YELLOW}Pending entry order {entry_order_id} status unknown/not found on exchange. Removing from pending state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)
                continue

            order_status = order_info.get('status')
            filled_amount = order_info.get('filled', 0.0)
            entry_price = order_info.get('average') # Use average fill price

            # Case 2: Order Fully Filled ('closed' status in CCXT, filled > 0)
            # Note: 'filled' can sometimes be None, treat as 0.0
            filled_amount = float(filled_amount) if filled_amount is not None else 0.0
            if order_status == 'closed' and filled_amount > 1e-12: # Use epsilon for float comparison
                if entry_price is None or float(entry_price) <= 1e-12:
                    logger.error(f"{Fore.RED}Entry order {entry_order_id} is 'closed' but has invalid/zero average fill price ({entry_price}). Cannot activate position correctly. Removing.{Style.RESET_ALL}")
                    positions_to_remove_ids.add(entry_order_id)
                    continue

                entry_price = float(entry_price) # Ensure float

                # Compare filled amount to originally requested amount (optional check)
                orig_size = position.get('original_size')
                if orig_size is not None and orig_size > 1e-12:
                    size_diff_ratio = abs(filled_amount - orig_size) / orig_size
                    if size_diff_ratio > 0.05: # e.g., > 5% difference
                         logger.warning(f"{Fore.YELLOW}Entry order {entry_order_id} filled amount ({filled_amount}) differs significantly (>5%) from requested ({orig_size}). Using actual filled amount.{Style.RESET_ALL}")

                logger.info(f"{Fore.GREEN}---> Pending entry order {entry_order_id} FILLED! Amount: {filled_amount:.{self.amount_decimals}f}, Avg Price: {entry_price:.{self.price_decimals}f}{Style.RESET_ALL}")

                # --- Update Position State ---
                updated_pos = position.copy() # Work on a copy
                updated_pos['size'] = filled_amount # Update size to actual filled amount
                updated_pos['entry_price'] = entry_price
                updated_pos['status'] = STATUS_ACTIVE
                # Use fill timestamp from order if available, else use current time
                fill_time_ms = order_info.get('timestamp')
                updated_pos['entry_time'] = fill_time_ms / 1000 if fill_time_ms else time.time()
                updated_pos['last_update_time'] = time.time()

                # --- Recalculate and Store SL/TP based on Actual Fill Price ---
                # This ensures SL/TP is relative to the real entry, not the estimated price used at order placement
                if current_price_for_check:
                    atr_value = indicators.get('atr') # Get current ATR
                    sl_price, tp_price = self._calculate_sl_tp_prices(
                        entry_price=updated_pos['entry_price'], # Use actual fill price
                        side=updated_pos['side'],
                        current_price=current_price_for_check, # Use current price for sanity checks
                        atr=atr_value
                    )
                    # Store these potentially adjusted SL/TP values in the state
                    # Note: These might differ from what the exchange *thinks* the SL/TP is if they were set via params initially.
                    # This bot logic will now use these calculated prices for its internal checks (like TSL).
                    updated_pos['stop_loss_price'] = sl_price
                    updated_pos['take_profit_price'] = tp_price
                    logger.info(f"Stored internal SL={format_price(sl_price)}, TP={format_price(tp_price)} for activated pos {entry_order_id} based on actual entry {entry_price:.{self.price_decimals}f}.")
                    # We assume the exchange's param-based SL/TP handles itself based on the initial request.
                    # If we need to *update* the exchange's SL/TP after fill, that requires edit_order/cancel_replace logic here.
                else:
                     logger.warning(f"{Fore.YELLOW}Could not fetch current price or SL/TP recalculation skipped after entry fill for {entry_order_id}. Stored internal SL/TP might be based on estimate.{Style.RESET_ALL}")
                     # Keep SL/TP prices that were originally stored when position was created (based on estimate)

                positions_to_update_data[entry_order_id] = updated_pos

            # Case 3: Order Failed (canceled, rejected, expired)
            elif order_status in ['canceled', 'rejected', 'expired']:
                reason = order_info.get('info', {}).get('rejectReason', 'No reason provided') # Try to get specific reason
                logger.warning(f"{Fore.YELLOW}Pending entry order {entry_order_id} failed (Status: {order_status}, Reason: {reason}). Removing from state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)

            # Case 4: Order Still Open or Partially Filled (remains pending)
            elif order_status == 'open':
                logger.debug(f"Pending entry order {entry_order_id} is still 'open'. Filled: {filled_amount:.{self.amount_decimals}f}. Waiting...")
                # Update last checked time?
                # positions_to_update_data[entry_order_id] = {'last_update_time': time.time()} # Update timestamp only
            elif order_status == 'closed' and filled_amount <= 1e-12:
                 # Can mean cancelled before any fill occurred
                 logger.warning(f"{Fore.YELLOW}Pending entry order {entry_order_id} has status 'closed' but zero filled amount. Assuming cancelled pre-fill. Removing.{Style.RESET_ALL}")
                 positions_to_remove_ids.add(entry_order_id)
            else:
                 # Unexpected status
                 logger.warning(f"Pending entry order {entry_order_id} has unexpected status: {order_status}. Filled: {filled_amount}. Leaving as pending for now.")


        # --- Apply State Updates ---
        if positions_to_remove_ids or positions_to_update_data:
            original_count = len(self.open_positions)
            new_positions_list = []
            removed_count = 0
            updated_count = 0

            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    continue # Skip removed positions

                if pos_id in positions_to_update_data:
                    # If only timestamp updated, merge; otherwise replace full dict
                    if list(positions_to_update_data[pos_id].keys()) == ['last_update_time']:
                        pos.update(positions_to_update_data[pos_id])
                        new_positions_list.append(pos)
                    else: # Replace with the fully updated dict (e.g., on fill)
                        new_positions_list.append(positions_to_update_data[pos_id])
                    updated_count += 1
                else:
                    new_positions_list.append(pos) # Keep unchanged positions

            if removed_count > 0 or updated_count > 0:
                 self.open_positions = new_positions_list
                 logger.debug(f"Pending checks complete. Updated/Filled: {len(positions_to_update_data)}, Removed/Failed: {removed_count}. Current total: {len(self.open_positions)}")
                 self._save_state() # Save the changes


    def _manage_active_positions(self, current_price: float, indicators: Dict) -> None:
        """
        Manages currently active positions.

        - Checks if the position was closed externally (e.g., SL/TP hit via parameters, manual closure).
        - Implements Time-Based Exit logic by placing a market close order.
        - Implements **EXPERIMENTAL** Trailing Stop Loss (TSL) logic using `edit_order`.
          **VERIFY EXCHANGE/CCXT SUPPORT FOR edit_order with SL/TP modification.**
        - Updates position state (e.g., TSL price) or removes closed positions.

        Args:
            current_price: The current market price.
            indicators: Dictionary containing current indicator values (e.g., ATR).
        """
        active_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_ACTIVE]
        if not active_positions:
            return # No active positions to manage

        logger.debug(f"Managing {len(active_positions)} active position(s) against current price: {current_price:.{self.price_decimals}f}...")

        positions_to_remove_ids = set()
        positions_to_update_state = {} # Store state updates {pos_id: {key: value}}

        for position in active_positions:
            pos_id = position.get('id') # This is the ID of the original entry order
            if not pos_id: continue # Skip if somehow ID is missing

            symbol = position.get('symbol', self.symbol) # Use position symbol if stored
            side = position['side']
            entry_price = position.get('entry_price')
            position_size = position.get('size')
            entry_time = position.get('entry_time', time.time()) # Use current if somehow missing
            # SL/TP prices stored in our state (recalculated on fill if possible)
            stored_sl_price = position.get('stop_loss_price')
            stored_tp_price = position.get('take_profit_price')
            # Current activated trailing stop price (if TSL is active for this pos)
            trailing_sl_price = position.get('trailing_stop_price')
            is_tsl_active = trailing_sl_price is not None

            # Basic validation
            if not all([symbol, side, entry_price, position_size, entry_time]):
                 logger.error(f"Active position {pos_id} has incomplete data: {position}. Skipping management.")
                 continue
            if position_size <= 1e-12:
                 logger.warning(f"Active position {pos_id} has size ~0 ({position_size}). Assuming closed. Removing.")
                 positions_to_remove_ids.add(pos_id)
                 continue

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None # Price at which the position closed

            # --- 1. Check Status of the Original Entry Order (Primary Check) ---
            # With parameter-based SL/TP, the original order's status change often indicates closure.
            # However, Bybit V5 might also create separate SL/TP orders. This check remains crucial.
            order_info = self.fetch_order_status(pos_id, symbol=symbol)

            if order_info is None:
                # Order not found - Could be closed & archived, cancelled, or ID issue.
                # Assume closed externally if the position was marked active.
                logger.warning(f"{Fore.YELLOW}Active position's main order {pos_id} not found on exchange. Assuming closed/cancelled externally. Removing position from state.{Style.RESET_ALL}")
                exit_reason = f"Main Order Vanished ({pos_id})"
                exit_price = current_price # Best guess is current price at time of check

            elif order_info.get('status') == 'closed':
                # Order is closed. Could be SL/TP hit, manual close, or potentially filled earlier if state was lagging.
                # Use average fill price if available (unlikely for closure), fallback to price, current price
                exit_price_raw = order_info.get('average') or order_info.get('price') or current_price
                exit_price = float(exit_price_raw) if exit_price_raw is not None else current_price

                # Try to infer reason based on Bybit V5 info fields if available
                order_info_details = order_info.get('info', {})
                bybit_status = order_info_details.get('orderStatus', '').lower()
                bybit_sl = order_info_details.get('stopLoss')
                bybit_tp = order_info_details.get('takeProfit')
                bybit_trigger_price = order_info_details.get('triggerPrice')

                if 'stop-loss order triggered' in bybit_status or \
                   (bybit_sl and bybit_trigger_price and str(bybit_trigger_price) == str(bybit_sl)):
                    exit_reason = f"Stop Loss Triggered (Exchange Reported - Order {pos_id})"
                    logger.info(f"{Fore.RED}{exit_reason} at ~{exit_price:.{self.price_decimals}f}{Style.RESET_ALL}")
                elif 'take-profit order triggered' in bybit_status or \
                     (bybit_tp and bybit_trigger_price and str(bybit_trigger_price) == str(bybit_tp)):
                     exit_reason = f"Take Profit Triggered (Exchange Reported - Order {pos_id})"
                     logger.info(f"{Fore.GREEN}{exit_reason} at ~{exit_price:.{self.price_decimals}f}{Style.RESET_ALL}")
                else:
                    # Infer based on price comparison if exchange info is unclear
                    sl_triggered = stored_sl_price and (
                        (side == 'buy' and exit_price <= stored_sl_price * 1.001) or
                        (side == 'sell' and exit_price >= stored_sl_price * 0.999)
                    )
                    tp_triggered = stored_tp_price and (
                        (side == 'buy' and exit_price >= stored_tp_price * 0.999) or
                        (side == 'sell' and exit_price <= stored_tp_price * 1.001)
                    )
                    if sl_triggered: exit_reason = f"Stop Loss Hit (Inferred Price - Order {pos_id})"
                    elif tp_triggered: exit_reason = f"Take Profit Hit (Inferred Price - Order {pos_id})"
                    else: exit_reason = f"Order Closed Externally (Reason Unclear - {pos_id}, Status: {bybit_status})"
                    log_color = Fore.RED if sl_triggered else (Fore.GREEN if tp_triggered else Fore.YELLOW)
                    logger.info(f"{log_color}{exit_reason} at ~{exit_price:.{self.price_decimals}f}{Style.RESET_ALL}")

            elif order_info.get('status') in ['canceled', 'rejected', 'expired']:
                 # This shouldn't happen for an 'active' position, indicates state inconsistency or external action.
                 logger.error(f"{Fore.RED}State inconsistency! Active position {pos_id} linked to order with status '{order_info.get('status')}'. Removing position.{Style.RESET_ALL}")
                 exit_reason = f"Order Inconsistency ({order_info.get('status')})"
                 exit_price = current_price

            # --- If order still seems active on exchange, check bot-managed exits ---
            if not exit_reason and order_info and order_info.get('status') == 'open':

                # --- 2. Time-Based Exit Check ---
                if self.time_based_exit_minutes is not None and self.time_based_exit_minutes > 0:
                    time_elapsed_seconds = time.time() - entry_time
                    time_elapsed_minutes = time_elapsed_seconds / 60
                    if time_elapsed_minutes >= self.time_based_exit_minutes:
                        logger.info(f"{Fore.YELLOW}Time limit ({self.time_based_exit_minutes} min) reached for active position {pos_id} (Age: {time_elapsed_minutes:.1f}m). Initiating market close.{Style.RESET_ALL}")
                        # --- Place Market Close Order ---
                        market_close_order = self._place_market_close_order(position, current_price)
                        if market_close_order:
                            # Assume closed, mark for removal. PnL calculated below.
                            exit_reason = "Time Limit Reached (Market Close Attempted)"
                            # Use fill price from close order if available, else current price as estimate
                            exit_price = market_close_order.get('average') or current_price
                            logger.info(f"Market close order for time exit placed successfully for {pos_id}. Exit Price ~{exit_price:.{self.price_decimals}f}")
                        else:
                            # CRITICAL: Failed to close position! Needs manual intervention.
                            logger.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL FAILURE: Failed to place market close order for time-based exit of position {pos_id}. POSITION REMAINS OPEN! Manual intervention required!{Style.RESET_ALL}")
                            # Do NOT set exit_reason, let it try again next loop or require manual action.
                # --- End Time-Based Exit ---


                # --- 3. Trailing Stop Loss (TSL) Logic ---
                # *** WARNING: EXPERIMENTAL and depends heavily on exchange/CCXT support for `edit_order`. ***
                # *** Use `enable_trailing_stop_loss: true` in config with extreme caution. ***
                # *** A safer TSL implementation often involves monitoring price and placing a market/limit close order. ***
                if not exit_reason and self.enable_trailing_stop_loss:
                    if not self.trailing_stop_loss_percentage or not (0 < self.trailing_stop_loss_percentage < 1):
                         logger.error("TSL enabled but trailing_stop_loss_percentage is invalid. Disabling TSL for this iteration.")
                    else:
                        new_tsl_price: Optional[float] = None
                        # Calculate potential new TSL based on current price
                        tsl_factor = (1 - self.trailing_stop_loss_percentage) if side == 'buy' else (1 + self.trailing_stop_loss_percentage)
                        potential_tsl_price_raw = current_price * tsl_factor

                        # --- TSL Activation Check ---
                        if not is_tsl_active:
                            # Activation condition: Price must have moved in favor significantly
                            # Example: Price crosses breakeven + half the initial SL distance, or halfway to TP
                            # Simple condition: Price crosses breakeven + a small buffer
                            activation_buffer = max(entry_price * 0.001, float(self.market_info['precision']['price'] or 0.0001)*5) # 0.1% or 5 ticks
                            breakeven_plus_buffer = entry_price + activation_buffer if side == 'buy' else entry_price - activation_buffer

                            activate_tsl = (side == 'buy' and current_price > breakeven_plus_buffer) or \
                                           (side == 'sell' and current_price < breakeven_plus_buffer)

                            if activate_tsl:
                                # Initial TSL: Trail from current price, but ensure it's better than breakeven (+buffer) and original SL (if set)
                                initial_trail_price = potential_tsl_price_raw
                                if stored_sl_price is not None:
                                     initial_trail_price = max(initial_trail_price, stored_sl_price) if side == 'buy' else min(initial_trail_price, stored_sl_price)
                                # Ensure TSL doesn't activate worse than breakeven buffer
                                initial_trail_price = max(initial_trail_price, breakeven_plus_buffer) if side == 'buy' else min(initial_trail_price, breakeven_plus_buffer)

                                try:
                                    new_tsl_price = float(self.exchange.price_to_precision(self.symbol, initial_trail_price))
                                    if new_tsl_price <= 0: new_tsl_price = None # Safety check
                                    if new_tsl_price: logger.info(f"{Fore.MAGENTA}Trailing Stop ACTIVATING for {side.upper()} {pos_id} at {new_tsl_price:.{self.price_decimals}f} (Price: {current_price:.{self.price_decimals}f}){Style.RESET_ALL}")
                                except Exception as format_err:
                                     logger.error(f"Error formatting initial TSL price {initial_trail_price}: {format_err}")

                        # --- TSL Update Check ---
                        elif is_tsl_active: # TSL already active, check if price moved further favorably
                            update_needed = (side == 'buy' and potential_tsl_price_raw > trailing_sl_price) or \
                                            (side == 'sell' and potential_tsl_price_raw < trailing_sl_price)

                            if update_needed:
                                try:
                                    potential_tsl_price = float(self.exchange.price_to_precision(self.symbol, potential_tsl_price_raw))
                                    if potential_tsl_price > 0 and potential_tsl_price != trailing_sl_price: # Check if update is meaningful
                                        new_tsl_price = potential_tsl_price
                                        logger.info(f"{Fore.MAGENTA}Trailing Stop UPDATING for {side.upper()} {pos_id} from {trailing_sl_price:.{self.price_decimals}f} to {new_tsl_price:.{self.price_decimals}f} (Price: {current_price:.{self.price_decimals}f}){Style.RESET_ALL}")
                                except Exception as format_err:
                                     logger.error(f"Error formatting updated TSL price {potential_tsl_price_raw}: {format_err}")

                        # --- Attempt EXPERIMENTAL Update via edit_order ---
                        if new_tsl_price is not None:
                            logger.warning(f"{Fore.YELLOW}[EXPERIMENTAL] Attempting TSL update via edit_order for {pos_id} to SL={new_tsl_price}. VERIFY EXCHANGE/CCXT SUPPORT!{Style.RESET_ALL}")
                            try:
                                # Parameters for editing SL/TP on Bybit V5 (check CCXT docs for exact names/types)
                                # `edit_order` might require more params than just SL/TP.
                                # Need original order type, side, amount, price (for limits).
                                edit_params = {
                                    'stopLoss': str(new_tsl_price) # Send as string for Bybit V5 params
                                    # To keep existing TP, you might need to resend it (if known and valid)
                                    # 'takeProfit': str(stored_tp_price) if stored_tp_price else None
                                }
                                if self.sl_trigger_by: edit_params['slTriggerBy'] = self.sl_trigger_by
                                # if stored_tp_price and self.tp_trigger_by: edit_params['tpTriggerBy'] = self.tp_trigger_by # Resend TP trigger?
                                # Bybit V5 category
                                if 'bybit' in self.exchange_id.lower():
                                    market_type = self.exchange.market(self.symbol).get('type', 'swap')
                                    if market_type in ['swap', 'future']:
                                        edit_params['category'] = 'linear' if self.exchange.market(self.symbol).get('linear') else 'inverse'

                                # Fetch fresh order info to get required parameters for edit_order
                                fresh_order_info = self.fetch_order_status(pos_id, symbol=symbol)
                                if not fresh_order_info or fresh_order_info.get('status') != 'open':
                                     logger.warning(f"Order {pos_id} status is not 'open' ({fresh_order_info.get('status') if fresh_order_info else 'Not Found'}). Cannot edit for TSL.")
                                     continue # Skip edit attempt for this position

                                edited_order = self.exchange.edit_order(
                                    id=pos_id,
                                    symbol=symbol,
                                    type=fresh_order_info.get('type'), # Use fresh type
                                    side=fresh_order_info.get('side'), # Use fresh side
                                    amount=fresh_order_info.get('amount'), # Use fresh amount
                                    price=fresh_order_info.get('price'), # Use fresh limit price (if limit)
                                    params=edit_params
                                )

                                if edited_order:
                                    confirmed_sl = edited_order.get('info',{}).get('stopLoss') or edited_order.get('stopLossPrice') # Check multiple possible fields
                                    logger.info(f"{Fore.MAGENTA}---> Successfully modified order {pos_id} via edit_order. New Confirmed SL: {confirmed_sl}. Applying TSL price {new_tsl_price} to internal state.{Style.RESET_ALL}")
                                    # Update internal state ONLY if edit seems successful
                                    update_payload = {
                                        'trailing_stop_price': new_tsl_price,
                                        'stop_loss_price': new_tsl_price, # Update the main internal SL price as well
                                        'last_update_time': time.time()
                                    }
                                    # Merge with potential previous updates for this pos_id
                                    positions_to_update_state[pos_id] = {**positions_to_update_state.get(pos_id, {}), **update_payload}

                                else:
                                    logger.error(f"{Fore.RED}TSL update via edit_order for {pos_id} potentially failed (API returned None/empty). TSL state NOT updated.{Style.RESET_ALL}")

                            except ccxt.NotSupported as e:
                                logger.error(f"{Fore.RED}TSL update failed: edit_order for SL/TP modification is NOT SUPPORTED by CCXT for {self.exchange_id} or for this order type ({fresh_order_info.get('type', 'N/A')}). Error: {e}. Consider disabling TSL or using market close logic.{Style.RESET_ALL}")
                                self.enable_trailing_stop_loss = False # Disable TSL permanently if not supported
                            except ccxt.OrderNotFound:
                                 logger.warning(f"{Fore.YELLOW}TSL update failed: Order {pos_id} not found during edit attempt (likely closed just now).{Style.RESET_ALL}")
                            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                                 logger.error(f"{Fore.RED}TSL update via edit_order failed for {pos_id} ({type(e).__name__}): {e}. Params sent: {edit_params}. TSL state NOT updated.{Style.RESET_ALL}")
                            except Exception as e:
                                logger.error(f"{Fore.RED}Unexpected error modifying order {pos_id} for TSL: {e}{Style.RESET_ALL}", exc_info=True)
                # --- End TSL Logic ---


            # --- 4. Process Exit and Log PnL ---
            if exit_reason:
                positions_to_remove_ids.add(pos_id)
                # Calculate and log PnL if entry/exit prices and size are valid
                if exit_price is not None and entry_price is not None and entry_price > 0 and exit_price > 0 and position_size is not None and position_size > 0:
                    # PnL calculation in quote currency
                    pnl_quote = (exit_price - entry_price) * position_size if side == 'buy' else (entry_price - exit_price) * position_size
                    # PnL calculation as percentage
                    try:
                        pnl_pct = ((exit_price / entry_price) - 1) * 100 if side == 'buy' else ((entry_price / exit_price) - 1) * 100
                    except ZeroDivisionError:
                        pnl_pct = 0.0 # Should not happen with price checks, but safety first

                    pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
                    quote_ccy = self.market_info['quote']
                    base_ccy = self.market_info['base']
                    logger.info(
                        f"{pnl_color}---> Position {pos_id} Closed. Reason: {exit_reason}. "
                        f"Entry: {entry_price:.{self.price_decimals}f}, Exit: {exit_price:.{self.price_decimals}f}, Size: {position_size:.{self.amount_decimals}f} {base_ccy}. "
                        f"Est. PnL: {pnl_quote:.{self.price_decimals}f} {quote_ccy} ({pnl_pct:.3f}%){Style.RESET_ALL}"
                    )
                    self.daily_pnl += pnl_quote # Update simple daily PnL tracker
                else:
                    # Log closure without PnL if prices/size are invalid/unknown
                    logger.warning(f"{Fore.YELLOW}---> Position {pos_id} Closed. Reason: {exit_reason}. Exit/entry price/size unknown or invalid, cannot calculate PnL accurately. Entry={entry_price}, Exit={exit_price}, Size={position_size}{Style.RESET_ALL}")


        # --- Apply State Updates and Removals ---
        if positions_to_remove_ids or positions_to_update_state:
            original_count = len(self.open_positions)
            new_positions_list = []
            removed_count = 0
            updated_count = 0

            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    continue # Skip removed positions
                if pos_id in positions_to_update_state:
                    pos.update(positions_to_update_state[pos_id]) # Apply updates
                    updated_count +=1
                new_positions_list.append(pos) # Keep updated or unchanged positions

            if removed_count > 0 or updated_count > 0:
                 self.open_positions = new_positions_list
                 logger.debug(f"Active position management complete. Updated: {updated_count}, Removed: {removed_count}. Current total: {len(self.open_positions)}")
                 self._save_state() # Save changes to state file


    def _place_market_close_order(self, position: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """
        Places a market order to close the given position.
        Uses 'reduce_only' parameter if supported by the exchange.

        Args:
            position: The position dictionary from the bot's state.
            current_price: The current market price (used for simulation & logging).

        Returns:
            The CCXT order dictionary of the close order if successful (or simulated),
            otherwise None.
        """
        pos_id = position['id']
        side = position['side']
        size = position['size']
        symbol = position.get('symbol', self.symbol)
        base_currency = self.market_info['base']
        quote_currency = self.market_info['quote']

        if size is None or size <= 1e-12:
            logger.error(f"Cannot close position {pos_id}: Size is zero, negative, or None ({size}).")
            return None

        # Determine the side of the closing order (opposite of the position side)
        close_side = 'sell' if side == 'buy' else 'buy'
        log_color = Fore.YELLOW # Use yellow for closing actions

        logger.warning(f"{log_color}Initiating MARKET CLOSE for position {pos_id} (Entry Side: {side.upper()}, Close Side: {close_side.upper()}, Size: {size:.{self.amount_decimals}f} {base_currency})...{Style.RESET_ALL}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_close_id = f"sim_close_{int(time.time() * 1000)}_{close_side[:1]}"
            # Simulate fill at the provided current price
            sim_avg_close_price = current_price
            sim_cost = size * sim_avg_close_price
            simulated_close_order = {
                "id": sim_close_id, "clientOrderId": sim_close_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.to_datetime('now', utc=True).isoformat(),
                "symbol": symbol, "type": "market", "side": close_side,
                "amount": size,
                "price": None, # Market orders don't have a specific price
                "average": sim_avg_close_price, # Fill price
                "cost": sim_cost,
                "status": 'closed', # Assume immediate fill in simulation
                "filled": size,
                "remaining": 0.0,
                "reduceOnly": True, # Indicate it was intended as reduceOnly
                "info": {
                    "simulated": True, "orderId": sim_close_id,
                    "reduceOnly": True, # Store flag in info as well
                    "closed_position_id": pos_id
                }
            }
            logger.info(f"{log_color}[SIMULATION] Market Close Order Placed: ID {sim_close_id}, Size {size:.{self.amount_decimals}f}, AvgPrice {sim_avg_close_price:.{self.price_decimals}f}{Style.RESET_ALL}")
            return simulated_close_order

        # --- Live Trading Mode ---
        else:
            order: Optional[Dict[str, Any]] = None
            try:
                # Use 'reduceOnly' parameter to ensure the order only closes/reduces the position
                params = {}
                # Check if exchange explicitly supports reduceOnly in createOrder params
                if self.exchange.safe_value(self.exchange.options, 'createMarketBuyOrderRequiresPrice', False):
                    # Some exchanges require price for market buy orders even with reduceOnly
                    # This is less common now but check if needed
                    # if close_side == 'buy': params['price'] = current_price * 1.1 # Set a high price to ensure fill? Risky.
                    pass

                # Add reduceOnly if supported
                if self.exchange.has.get('reduceOnly'):
                    params['reduceOnly'] = True
                    logger.debug("Using 'reduceOnly=True' parameter for market close order.")
                else:
                     logger.warning(f"{Fore.YELLOW}Exchange {self.exchange_id} might not explicitly support 'reduceOnly' parameter via CCXT `has`. Close order might accidentally open a new position if size is incorrect or position already closed.{Style.RESET_ALL}")
                     # Still attempt it, some exchanges support it even if `has` is False

                # Bybit V5 might need category
                if 'bybit' in self.exchange_id.lower():
                    market_type = self.exchange.market(symbol).get('type', 'swap')
                    if market_type in ['swap', 'future']:
                        params['category'] = 'linear' if self.exchange.market(symbol).get('linear') else 'inverse'

                order = self.exchange.create_market_order(symbol, close_side, size, params=params)

                if order:
                    oid = order.get('id', 'N/A')
                    oavg = order.get('average') # Actual average fill price
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    ofilled = order.get('filled', 0.0)
                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, "
                        f"Status: {ostatus}, Filled: {ofilled:.{self.amount_decimals}f}, AvgFill ~{oavg:.{self.price_decimals}f} (if available){Style.RESET_ALL}"
                    )
                    return order
                else:
                    # API call returned None/empty
                    logger.error(f"{Fore.RED}LIVE Market Close order placement failed: API returned None or empty response for position {pos_id}.{Style.RESET_ALL}")
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                 # Specific errors often indicating state mismatch or parameter issues
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} ({type(e).__name__}): {e}. Position might already be closed or size incorrect.{Style.RESET_ALL}")
                 logger.error(f" -> Params Sent: {params}")
                 return None
            except ccxt.ExchangeError as e:
                 # General exchange errors
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (ExchangeError): {e}{Style.RESET_ALL}")
                 logger.error(f" -> Params Sent: {params}")
                 # Example: Bybit error if position already closed might be caught here
                 if "position size is zero" in str(e).lower():
                      logger.warning(f" -> Exchange indicates position {pos_id} size is already zero.")
                 return None
            except Exception as e:
                 # Unexpected Python errors
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True)
                 logger.error(f" -> Params Sent: {params}")
                 return None


    # --- Main Bot Execution Loop ---

    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches all required market data (price, order book, history) for one iteration."""
        logger.debug("Fetching market data bundle...")
        start_time = time.time()
        try:
            # Fetch concurrently? For simple cases, sequential is fine.
            # Use asyncio or threading for performance optimization if needed.
            current_price = self.fetch_market_price()
            order_book_data = self.fetch_order_book() # Contains 'imbalance' if successful
            historical_data = self.fetch_historical_data() # Fetches required lookback

            # --- Check if all essential data was fetched ---
            if current_price is None:
                logger.warning(f"{Fore.YELLOW}Failed to fetch current market price. Skipping iteration.{Style.RESET_ALL}")
                return None
            if historical_data is None or historical_data.empty:
                logger.warning(f"{Fore.YELLOW}Failed to fetch sufficient historical data. Skipping iteration.{Style.RESET_ALL}")
                return None

            # Order book is optional, proceed without it if failed
            order_book_imbalance = None
            if order_book_data is None:
                 logger.warning(f"{Fore.YELLOW}Failed to fetch order book data. Proceeding without imbalance signal.{Style.RESET_ALL}")
            else:
                 order_book_imbalance = order_book_data.get('imbalance') # Can be None, float, or inf

            fetch_duration = time.time() - start_time
            logger.debug(f"Market data fetched successfully in {fetch_duration:.2f}s.")
            return {
                "price": current_price,
                "order_book_imbalance": order_book_imbalance,
                "historical_data": historical_data
            }
        except Exception as e:
            # Catch unexpected errors during the data fetching process
            logger.error(f"{Fore.RED}Unexpected error during market data fetching phase: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculates all required technical indicators based on the historical data."""
        logger.debug("Calculating technical indicators...")
        start_time = time.time()
        indicators = {}
        # Ensure required columns exist and data is sufficient
        if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             logger.error("Historical data DataFrame is missing required columns (OHLCV). Cannot calculate indicators.")
             return {} # Return empty dict
        if len(historical_data) < 2: # Need at least 2 rows for most indicators
             logger.error(f"Insufficient historical data rows ({len(historical_data)}) for indicator calculation.")
             return {}

        close = historical_data['close']
        high = historical_data['high']
        low = historical_data['low']

        # Calculate each indicator using the static methods
        indicators['volatility'] = self.calculate_volatility(close, self.volatility_window)
        indicators['ema'] = self.calculate_ema(close, self.ema_period)
        indicators['rsi'] = self.calculate_rsi(close, self.rsi_period)
        indicators['macd_line'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd(
            close, self.macd_short_period, self.macd_long_period, self.macd_signal_period
        )
        indicators['stoch_k'], indicators['stoch_d'] = self.calculate_stoch_rsi(
            close, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period
        )
        indicators['atr'] = self.calculate_atr(high, low, close, self.atr_period)

        calc_duration = time.time() - start_time
        # Log calculated indicators (format nicely using stored decimals)
        log_inds = {
            'EMA': f"{indicators['ema']:.{self.price_decimals}f}" if indicators.get('ema') is not None else 'N/A',
            'RSI': f"{indicators['rsi']:.1f}" if indicators.get('rsi') is not None else 'N/A',
            'ATR': f"{indicators['atr']:.{self.price_decimals}f}" if indicators.get('atr') is not None else 'N/A', # ATR uses price scale
            'MACD_L': f"{indicators['macd_line']:.{self.price_decimals}f}" if indicators.get('macd_line') is not None else 'N/A', # MACD uses price scale
            'MACD_S': f"{indicators['macd_signal']:.{self.price_decimals}f}" if indicators.get('macd_signal') is not None else 'N/A',
            'StochK': f"{indicators['stoch_k']:.1f}" if indicators.get('stoch_k') is not None else 'N/A',
            'StochD': f"{indicators['stoch_d']:.1f}" if indicators.get('stoch_d') is not None else 'N/A',
        }
        logger.info(f"Indicators (calc took {calc_duration:.2f}s): "
                    f"EMA={log_inds['EMA']}, RSI={log_inds['RSI']}, ATR={log_inds['ATR']}, "
                    f"MACD(L/S)=({log_inds['MACD_L']}/{log_inds['MACD_S']}), "
                    f"Stoch(K/D)=({log_inds['StochK']}/{log_inds['StochD']})")

        # Check for None values which indicate calculation issues (e.g., insufficient data)
        if any(v is None for v in indicators.values()):
             logger.warning(f"{Fore.YELLOW}One or more indicators calculated as None. Check data sufficiency or calculation logic.{Style.RESET_ALL}")

        return indicators

    def _process_signals_and_entry(self, market_data: Dict, indicators: Dict) -> None:
        """
        Analyzes market data and indicators to compute a trade signal.
        If a signal meets the threshold and conditions allow, attempts to place an entry order.

        Args:
            market_data: Dictionary containing current price, order book imbalance, etc.
            indicators: Dictionary containing calculated indicator values.
        """
        current_price = market_data['price']
        orderbook_imbalance = market_data['order_book_imbalance']
        atr_value = indicators.get('atr') # Needed for potential ATR-based SL/TP and size calc

        # --- 1. Check Entry Conditions ---
        # Can we open a new position?
        active_or_pending = [p for p in self.open_positions if p.get('status') in [STATUS_ACTIVE, STATUS_PENDING_ENTRY]]
        can_open_new = len(active_or_pending) < self.max_open_positions
        if not can_open_new:
             logger.info(f"{Fore.YELLOW}Max open/pending positions ({self.max_open_positions}) reached. Skipping new entry check.{Style.RESET_ALL}")
             return

        # Calculate potential order size - do this early to see if trading is possible
        # Pass current price and ATR if needed for size calculation (e.g., vol-adjusted size)
        order_size_base = self.calculate_order_size(current_price) # Returns size in BASE currency
        can_trade = order_size_base > 0
        if not can_trade:
             logger.warning(f"{Fore.YELLOW}Cannot evaluate entry signal: Calculated order size is zero (check balance, order size %, or min limits).{Style.RESET_ALL}")
             return

        # --- 2. Compute Signal Score ---
        signal_score, reasons = self.compute_trade_signal_score(
            current_price, indicators, orderbook_imbalance
        )
        logger.info(f"Trade Signal Score: {signal_score}")
        # Log reasons only if score is non-zero or debug level is enabled
        if signal_score != 0 or logger.isEnabledFor(logging.DEBUG):
            for reason in reasons:
                 # Indent reasons for better readability
                 logger.info(f"  -> {reason}")

        # --- 3. Determine Entry Action ---
        entry_side: Optional[str] = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'buy'
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'sell'

        # --- 4. Place Entry Order if Signal Strong Enough ---
        if entry_side:
            log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
            logger.info(f"{log_color}Potential {entry_side.upper()} entry signal detected (Score: {signal_score} >= Threshold: {ENTRY_SIGNAL_THRESHOLD_ABS}){Style.RESET_ALL}")

            # --- Calculate SL/TP Prices BEFORE placing order ---
            # Use current_price as the reference for initial SL/TP calculation before entry
            sl_price, tp_price = self._calculate_sl_tp_prices(
                entry_price=current_price, # Use current price as estimate for calculation basis
                side=entry_side,
                current_price=current_price, # Pass current price for sanity checks
                atr=atr_value
            )

            # Check if SL/TP calculation is strictly required and failed
            # If ATR is enabled but ATR is None, abort.
            if self.use_atr_sl_tp and atr_value is None:
                 logger.error(f"{Fore.RED}ATR SL/TP required but ATR is None. Cannot place entry order.{Style.RESET_ALL}")
                 return
            # If Fixed % is enabled but not set, abort (already validated, but belt-and-suspenders).
            if not self.use_atr_sl_tp and (self.base_stop_loss_pct is None or self.base_take_profit_pct is None):
                 logger.error(f"{Fore.RED}Fixed % SL/TP required but percentages not set. Cannot place entry order.{Style.RESET_ALL}")
                 return
            # If SL/TP calculation resulted in None due to other issues (e.g., price checks), log warning.
            # Decide whether to proceed without SL/TP or abort. For now, proceed but log warning.
            if sl_price is None and tp_price is None and (self.use_atr_sl_tp or self.base_stop_loss_pct is not None):
                 logger.warning(f"{Fore.YELLOW}SL/TP calculation failed or resulted in None (check logs). Attempting to place order without SL/TP parameters.{Style.RESET_ALL}")
                 # If SL/TP is critical, uncomment the next line to abort:
                 # return

            # --- Place the Actual Entry Order ---
            entry_order = self.place_entry_order(
                side=entry_side,
                order_size_base=order_size_base, # Pass calculated BASE currency size
                confidence_level=signal_score,
                order_type=self.entry_order_type,
                current_price=current_price, # Pass for limit/market reference
                stop_loss_price=sl_price,    # Pass calculated SL (float or None)
                take_profit_price=tp_price   # Pass calculated TP (float or None)
            )

            # --- Update Bot State if Order Placement Succeeded ---
            if entry_order:
                order_id = entry_order.get('id')
                if not order_id:
                     logger.error(f"{Fore.RED}Entry order placed but received no order ID! Cannot track position. Order response: {entry_order}{Style.RESET_ALL}")
                     # Attempt to cancel if possible? Risky without ID.
                     return # Cannot track without ID

                order_status = entry_order.get('status')
                initial_pos_status = STATUS_UNKNOWN
                is_filled_immediately = False

                if order_status == 'open':
                    initial_pos_status = STATUS_PENDING_ENTRY
                elif order_status == 'closed': # Assumed filled for market orders or immediate limit fills
                    initial_pos_status = STATUS_ACTIVE
                    is_filled_immediately = True
                elif order_status in ['canceled', 'rejected', 'expired']:
                    logger.warning(f"Entry order {order_id} failed immediately ({order_status}). Not adding position.")
                    return # Don't add failed orders
                else: # e.g., 'triggered', 'partially_filled' (less common for entry)
                    logger.warning(f"Entry order {order_id} has potentially intermediate status: {order_status}. Treating as PENDING for now.")
                    initial_pos_status = STATUS_PENDING_ENTRY

                # Use actual fill price/time if available, else estimate/use order details
                filled_amount = float(entry_order.get('filled', 0.0))
                avg_fill_price = float(entry_order.get('average')) if entry_order.get('average') else None
                order_timestamp_ms = entry_order.get('timestamp') # Timestamp of order creation/update
                # Use requested amount for original size, even if partially filled initially
                requested_amount = float(entry_order.get('amount', order_size_base))

                # Use filled price/time if available, else use limit price or current price as estimate
                actual_entry_price = avg_fill_price if is_filled_immediately and avg_fill_price else (entry_order.get('price') or current_price)
                actual_entry_time_sec = order_timestamp_ms / 1000 if is_filled_immediately and order_timestamp_ms else None

                # Create the new position dictionary for state tracking
                new_position = {
                    "id": order_id,
                    "symbol": self.symbol, # Store symbol with position
                    "side": entry_side,
                    # Store both requested and initially filled size
                    "size": filled_amount if is_filled_immediately else requested_amount, # Current size reflects fill status
                    "original_size": requested_amount, # Always store the requested size
                    "entry_price": actual_entry_price, # Best estimate/actual fill price
                    "entry_time": actual_entry_time_sec, # Timestamp of fill (if known), else None
                    "status": initial_pos_status,
                    "entry_order_type": self.entry_order_type,
                    # Store the SL/TP prices that were *sent* with the order request (from calculation)
                    # These are the bot's target SL/TPs. Exchange confirms separately if params worked.
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "confidence": signal_score,
                    "trailing_stop_price": None, # TSL not active initially
                    "last_update_time": time.time()
                }
                self.open_positions.append(new_position)
                entry_price_f = f"{actual_entry_price:.{self.price_decimals}f}" if actual_entry_price else "N/A"
                logger.info(f"{log_color}---> Position {order_id} added to state with status: {initial_pos_status}. Entry Price Est/Actual: ~{entry_price_f}{Style.RESET_ALL}")

                # If filled immediately, ensure size/price/time reflect that accurately
                if is_filled_immediately:
                    if filled_amount > 0: new_position['size'] = filled_amount
                    if avg_fill_price: new_position['entry_price'] = avg_fill_price
                    if order_timestamp_ms: new_position['entry_time'] = order_timestamp_ms / 1000

                self._save_state() # Save state after adding the new position
            else:
                 logger.error(f"{Fore.RED}Entry order placement failed (API call returned None or error). No position added.{Style.RESET_ALL}")
        else:
            # Log only if score was calculated (avoids spamming neutral messages)
            if signal_score is not None:
                 logger.info(f"Neutral signal score ({signal_score}). No entry action taken.")


    def run(self) -> None:
        """Starts the main trading loop of the bot."""
        logger.info(f"{Fore.CYAN}--- Initiating Trading Loop (Symbol: {self.symbol}, Timeframe: {self.timeframe}) ---{Style.RESET_ALL}")

        # --- Optional Initial Cleanup ---
        # Consider cancelling any potentially orphaned orders from previous runs.
        # Be cautious if other processes might be trading the same symbol.
        # logger.warning("Performing initial check for existing open orders...")
        # initial_cancel_count = self.cancel_all_symbol_orders()
        # logger.info(f"Initial cancel check completed. Cancelled {initial_cancel_count} orders.")

        while True:
            self.iteration += 1
            start_time_iter = time.time()
            # Use UTC time for consistency
            timestamp_str = pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')
            loop_prefix = f"{Fore.BLUE}===== Iteration {self.iteration} ===={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Timestamp: {timestamp_str}")

            try:
                # --- 1. Fetch Market Data Bundle ---
                market_data = self._fetch_market_data()
                if market_data is None:
                    logger.warning("Essential market data missing, pausing before next iteration.")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue # Skip rest of loop

                current_price = market_data['price']
                ob_imb = market_data.get('order_book_imbalance')
                ob_imb_str = f"{ob_imb:.3f}" if isinstance(ob_imb, (float, int)) else ('Inf' if ob_imb == float('inf') else 'N/A')

                # Use stored price_decimals for formatting
                logger.info(f"{loop_prefix} Current Price: {current_price:.{self.price_decimals}f} "
                            f"OB Imbalance: {ob_imb_str}")

                # --- 2. Calculate Indicators ---
                indicators = self._calculate_indicators(market_data['historical_data'])
                if not indicators: # Check if indicator calculation failed critically
                     logger.error("Indicator calculation failed. Skipping trading logic for this iteration.")
                     time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue

                # --- 3. Check Pending Entry Orders ---
                # Checks if limit orders were filled and updates state accordingly
                self._check_pending_entries(indicators) # Pass indicators for potential SL/TP recalc on fill

                # --- 4. Manage Active Positions ---
                # Handles TSL updates, time-based exits, and checks for external closures (SL/TP hits)
                # Run this *before* checking for new entries to free up position slots if exits occur
                self._manage_active_positions(current_price, indicators) # Pass current price and indicators

                # --- 5. Process Signals & Potential New Entry ---
                # Evaluates indicators/market data for new trade opportunities
                self._process_signals_and_entry(market_data, indicators)

                # --- 6. Loop Pacing ---
                end_time_iter = time.time()
                execution_time = end_time_iter - start_time_iter
                # Ensure minimum sleep interval even if execution was fast
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - execution_time)
                logger.debug(f"{loop_prefix} Iteration took {execution_time:.2f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                # Allow Ctrl+C to break the loop for graceful shutdown
                logger.warning(f"\n{Fore.YELLOW}Keyboard interrupt received. Exiting main loop...{Style.RESET_ALL}")
                break # Exit the while loop
            except SystemExit as e:
                 logger.warning(f"SystemExit called with code {e.code}. Exiting main loop...")
                 raise e # Re-raise to be caught by main block's finally
            except Exception as e:
                # Catch-all for unexpected errors within the main loop
                # Log traceback for detailed debugging
                logger.critical(f"{Fore.RED}{Style.BRIGHT}{loop_prefix} CRITICAL UNHANDLED ERROR in main loop: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                # Implement a cool-down period before potentially retrying
                error_pause_seconds = 60
                logger.warning(f"{Fore.YELLOW}Pausing for {error_pause_seconds} seconds due to critical loop error before next attempt...{Style.RESET_ALL}")
                time.sleep(error_pause_seconds)

        logger.info("Main trading loop terminated.")


    def shutdown(self):
        """Performs graceful shutdown procedures for the bot."""
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Graceful Shutdown Sequence ---{Style.RESET_ALL}")

        # --- 1. Cancel Open PENDING Orders ---
        # Focus on orders potentially created by this bot instance that haven't filled
        pending_orders_cancelled = 0
        logger.info("Attempting to cancel known PENDING entry orders...")
        # Iterate over a copy as cancellation might affect the list indirectly via state saves
        pending_positions = [p for p in list(self.open_positions) if p.get('status') == STATUS_PENDING_ENTRY]

        if not pending_positions:
             logger.info("No pending entry orders found in state to cancel.")
        else:
            for pos in pending_positions:
                pos_id = pos.get('id')
                pos_symbol = pos.get('symbol', self.symbol)
                if not pos_id: continue # Skip if ID is missing

                logger.info(f"Cancelling pending entry order {pos_id} ({pos_symbol})...")
                if self.cancel_order_by_id(pos_id, symbol=pos_symbol):
                    pending_orders_cancelled += 1
                    # Update state immediately to reflect cancellation during shutdown
                    pos['status'] = STATUS_CANCELED # Mark as canceled in our state
                    pos['last_update_time'] = time.time()
                else:
                    logger.error(f"{Fore.RED}Failed to cancel pending order {pos_id} during shutdown. Manual check recommended.{Style.RESET_ALL}")
            logger.info(f"Attempted cancellation of {len(pending_positions)} pending orders. Success/Already Gone: {pending_orders_cancelled}.")


        # --- 2. Optionally Close ACTIVE Positions ---
        # Use the configured flag `close_positions_on_exit`
        active_positions = [p for p in self.open_positions if p.get('status') == STATUS_ACTIVE]

        if self.close_positions_on_exit and active_positions:
             logger.warning(f"{Fore.YELLOW}Configuration requests closing {len(active_positions)} active position(s) via market order...{Style.RESET_ALL}")
             current_price_for_close = self.fetch_market_price() # Get latest price for closing
             if current_price_for_close:
                 closed_count = 0
                 failed_close_ids = []
                 for position in list(active_positions): # Iterate copy
                    pos_id = position['id']
                    logger.info(f"Attempting market close for active position {pos_id}...")
                    close_order = self._place_market_close_order(position, current_price_for_close)
                    if close_order:
                        closed_count += 1
                        logger.info(f"{Fore.YELLOW}Market close order successfully placed for position {pos_id}. Marking as closed in state.{Style.RESET_ALL}")
                        # Update state immediately after placing close order
                        position['status'] = 'closed_on_exit' # Custom status
                        position['last_update_time'] = time.time()
                    else:
                        failed_close_ids.append(pos_id)
                        # CRITICAL: Failed to close during shutdown!
                        logger.error(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Failed to place market close order for {pos_id} during shutdown. Manual check required!{Style.RESET_ALL}")

                 logger.info(f"Attempted to close {len(active_positions)} active positions. Orders placed for: {closed_count}. Failed for: {len(failed_close_ids)}.")
                 if failed_close_ids:
                      logger.error(f"Failed close order IDs: {failed_close_ids}")

             else:
                 logger.error(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Cannot fetch current price for market close during shutdown. {len(active_positions)} position(s) remain open! Manual check required!{Style.RESET_ALL}")
        elif active_positions:
            # Log remaining active positions if not closing them
            logger.warning(f"{Fore.YELLOW}{len(active_positions)} position(s) remain active (Close on exit disabled or failed). Manual management may be required:{Style.RESET_ALL}")
            for pos in active_positions:
                pos_id = pos.get('id')
                side = pos.get('side')
                size = pos.get('size')
                entry = pos.get('entry_price')
                logger.warning(f" -> ID: {pos_id}, Side: {side}, Size: {size:.{self.amount_decimals}f}, Entry: {entry:.{self.price_decimals}f}")

        # --- 3. Final State Save ---
        # Save the potentially modified state (cancelled orders, positions marked closed)
        logger.info("Saving final bot state...")
        self._save_state()

        logger.info(f"{Fore.CYAN}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")
        # Ensure all log handlers are closed properly
        logging.shutdown()


# --- Main Execution Block ---
if __name__ == "__main__":
    bot_instance: Optional[ScalpingBot] = None
    exit_code: int = 0

    try:
        # --- Pre-computation/Setup ---
        # Ensure state directory exists if STATE_FILE_NAME includes a path
        state_dir = os.path.dirname(STATE_FILE_NAME)
        if state_dir and not os.path.exists(state_dir):
            try:
                # Use print here as logger might not be fully ready
                print(f"Creating state directory: {state_dir}")
                os.makedirs(state_dir)
            except OSError as e:
                 print(f"{Fore.RED}Fatal: Could not create state directory '{state_dir}': {e}. Aborting.{Style.RESET_ALL}", file=sys.stderr)
                 sys.exit(1)

        # --- Initialize and Run Bot ---
        # Initialization handles config loading, validation, exchange connection, market info, state loading
        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)
        bot_instance.run() # Start the main trading loop

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal received (Ctrl+C). Initiating graceful shutdown...{Style.RESET_ALL}")
        exit_code = 130 # Standard exit code for Ctrl+C

    except SystemExit as e:
        # Catch sys.exit() calls (e.g., from config validation failure)
        # Log SystemExit if it wasn't a clean exit (code 0) or standard interrupt code
        exit_code = e.code if e.code is not None else 1 # Default to 1 if code is None
        if exit_code not in [0, 130]:
             logger.error(f"Bot exited via SystemExit with unexpected code: {exit_code}.")
        else:
             logger.info(f"Bot exited via SystemExit with code: {exit_code}.")

    except Exception as e:
        # Catch any critical unhandled exceptions during setup or run that weren't caught internally
        logger.critical(f"{Fore.RED}{Style.BRIGHT}An unhandled critical error occurred outside the main loop: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
        exit_code = 1 # Indicate an error exit

    finally:
        # --- Graceful Shutdown ---
        # This block executes regardless of how the try block was exited
        if bot_instance:
            logger.info("Executing bot shutdown procedures...")
            try:
                bot_instance.shutdown()
            except Exception as shutdown_err:
                 logger.error(f"{Fore.RED}Error during bot shutdown procedure: {shutdown_err}{Style.RESET_ALL}", exc_info=True)
                 # Ensure logging is shut down even if bot shutdown fails
                 logging.shutdown()
        else:
            # If bot instance creation failed, just ensure logging is shut down
            logger.info("Bot instance not fully initialized or shutdown already initiated. Shutting down logging.")
            logging.shutdown()

        print(f"{Fore.MAGENTA}Pyrmethus bids thee farewell. Exit Code: {exit_code}{Style.RESET_ALL}")
        sys.exit(exit_code)
