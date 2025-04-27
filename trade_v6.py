

arameters are correct! ---
2025-04-25 17:35:36,280 - ScalpingBotV4 - INFO [MainThread] - Scalping Bot V4 initialized. Symbol: DOT/USDT:USDT, Timeframe: 3m
2025-04-25 17:35:36,280 - ScalpingBotV4 - INFO [MainThread] - --- Initiating Trading Loop (Symbol: DOT/USDT:USDT, Timeframe: 3m) ---
2025-04-25 17:35:36,281 - ScalpingBotV4 - INFO [MainThread] -
===== Iteration 1 ==== Timestamp: 2025-04-25T22:35:36+00:00
2025-04-25 17:35:37,492 - ScalpingBotV4 - CRITICAL [MainThread] - ===== Iteration 1 ==== CRITICAL UNHANDLED ERROR in main loop: Invalid format specifier '.0.0001f' for object of type 'float'
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/worldguide/trade_v6.py", line 2754, in run
    logger.info(f"{loop_prefix} Current Price: {current_price:.{self.market_info['precision']['price']}f} "
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: Invalid format specifier '.0.0001f' for object of type 'float'
2025-04-25 17:35:37,498 - ScalpingBotV4 - WARNING [MainThread] - Pausing for 60 seconds due to critical loop error before next attempt...

# -*- coding: utf-8 -*-
"""
Scalping Bot v4 - Pyrmethus Enhanced Edition (Bybit V5 Optimized)

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based Stop Loss (SL) and Take Profit (TP)
handling.

Key Enhancements V4 (Compared to a hypothetical V3):
- Solidified Bybit V5 parameter-based SL/TP in create_order.
- Enhanced Trailing Stop Loss (TSL) logic using edit_order (Requires careful
  verification with specific CCXT version and exchange behavior).
- Added `enable_trailing_stop_loss` configuration flag.
- Added `sl_trigger_by`, `tp_trigger_by` configuration options for Bybit V5.
- Added explicit `testnet_mode` configuration flag for clarity.
- Robust persistent state file handling with automated backup on corruption.
- Improved error handling, logging verbosity, type hinting, and overall code clarity.
- More detailed validation of configuration parameters.
"""

import logging
import os
import sys
import time
import json
import shutil # For state file backup and management
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

# Position Status Constants
STATUS_PENDING_ENTRY: str = 'pending_entry' # Order placed but not yet filled
STATUS_ACTIVE: str = 'active'           # Order filled, position is live
STATUS_CLOSING: str = 'closing'         # Manual close initiated (optional use)
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
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
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
                    logger.error(
                        f"{Fore.RED}Network ether disturbed ({func.__name__}: {type(e).__name__}). Pausing {delay}s... "
                        f"(Attempt {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )

                # Exchange Errors - Some are retryable, others indicate fundamental issues
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    if any(phrase in err_str for phrase in [
                        "order not found", "order does not exist", "unknown order",
                        "order already cancelled", "order was canceled", "cancel order failed" # Added variations
                    ]):
                        logger.warning(f"{Fore.YELLOW}Order vanished, unknown, or cancellation failed for {func.__name__}: {e}. Returning None (no retry).{Style.RESET_ALL}")
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
                    ]):
                         logger.error(f"{Fore.RED}Invalid parameters, limits, auth, or config for {func.__name__}: {e}. Aborting operation (no retry). Review logic/config/permissions.{Style.RESET_ALL}")
                         return None # Requires code or config fix
                    # --- Potentially Retryable Exchange Errors ---
                    else:
                        logger.error(
                            f"{Fore.RED}Exchange spirit troubled ({func.__name__}: {type(e).__name__} - {e}). Pausing {delay}s... "
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
    and an experimental Trailing Stop Loss (TSL) mechanism using `edit_order`.

    **Disclaimer:** Trading cryptocurrencies involves significant risk. This bot
    is provided for educational purposes and experimentation. Use at your own risk.
    Thoroughly test in simulation and testnet modes before considering live trading.
    The TSL feature using `edit_order` requires careful verification.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """
        Initializes the bot instance.

        Loads configuration, validates settings, establishes exchange connection,
        loads persistent state, and caches necessary market information.

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
        #     'id': str,                   # Exchange Order ID of the entry order
        #     'side': str,                 # 'buy' or 'sell'
        #     'size': float,               # Current position size (base currency)
        #     'original_size': float,      # Requested size at entry
        #     'entry_price': float,        # Average fill price of the entry order
        #     'entry_time': float,         # Timestamp (seconds) of entry fill
        #     'status': str,               # e.g., STATUS_PENDING_ENTRY, STATUS_ACTIVE
        #     'entry_order_type': str,     # 'market' or 'limit'
        #     'stop_loss_price': Optional[float], # Price level for SL (parameter-based)
        #     'take_profit_price': Optional[float],# Price level for TP (parameter-based)
        #     'confidence': int,           # Signal score at entry
        #     'trailing_stop_price': Optional[float], # Current activated TSL price
        #     'last_update_time': float    # Timestamp of the last state update
        # }
        self.open_positions: List[Dict[str, Any]] = []
        self.market_info: Optional[Dict[str, Any]] = None # Cache for market details (precision, limits)

        # --- Setup & Initialization Steps ---
        self._configure_logging_level() # Apply logging level from config
        self.exchange: ccxt.Exchange = self._initialize_exchange() # Connect to exchange
        self._load_market_info() # Load and cache market details for the symbol
        self._load_state() # Load persistent state from file

        # --- Log Final Operating Modes ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")

        if not self.simulation_mode:
            if not self.testnet_mode:
                logger.warning(f"{Fore.RED}--- WARNING: LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}")
                logger.warning(f"{Fore.RED}--- Ensure configuration and risk parameters are correct! ---{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}--- Running in full internal simulation mode. No real orders will be placed. ---{Style.RESET_ALL}")


        logger.info(f"{Fore.CYAN}Scalping Bot V4 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}")

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        try:
            log_level_str = self.config.get("logging_level", "INFO").upper()
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
                        self.open_positions = saved_state
                        logger.info(f"{Fore.GREEN}Recalled {len(self.open_positions)} position(s) from state file.{Style.RESET_ALL}")
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
        Creates a backup before overwriting the main state file.
        """
        logger.debug(f"Recording current state ({len(self.open_positions)} positions) to {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        temp_state_file = f"{self.state_file}.tmp" # Use a temporary file for atomic write

        try:
            # Write to a temporary file first
            with open(temp_state_file, 'w', encoding='utf-8') as f:
                # Use default=str to handle potential non-serializable types gracefully (e.g., numpy types)
                json.dump(self.open_positions, f, indent=4, default=str)

            # If temporary write succeeds, create backup of current state (if exists)
            if os.path.exists(self.state_file):
                try:
                    shutil.copyfile(self.state_file, state_backup_file)
                except Exception as backup_err:
                    logger.warning(f"Could not create state backup {state_backup_file}: {backup_err}. Proceeding with overwrite.")

            # Atomically replace the old state file with the new one
            shutil.move(temp_state_file, self.state_file)

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
            # Logging level for console output: DEBUG, INFO, WARNING, ERROR, CRITICAL
            "logging_level": "INFO",
            "exchange": {
                "exchange_id": os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID),
                # Explicit flag for using testnet environment. Set to false for mainnet.
                "testnet_mode": os.getenv("TESTNET_MODE", "True").lower() in ("true", "1", "yes"),
            },
            "trading": {
                # Symbol format depends on exchange (e.g., BTC/USDT:USDT for Bybit USDT Perpetuals)
                "symbol": os.getenv("TRADING_SYMBOL", "BTC/USDT:USDT"),
                # Timeframe for OHLCV data (e.g., 1m, 5m, 1h)
                "timeframe": os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME),
                # Internal simulation: if true, bot runs logic but places no real orders.
                "simulation_mode": os.getenv("SIMULATION_MODE", "True").lower() in ("true", "1", "yes"),
                # Entry order type: 'limit' or 'market'
                "entry_order_type": os.getenv("ENTRY_ORDER_TYPE", "limit").lower(),
                # Offset from current price for limit buy orders (e.g., 0.0005 = 0.05%)
                "limit_order_offset_buy": float(os.getenv("LIMIT_ORDER_OFFSET_BUY", 0.0005)),
                # Offset from current price for limit sell orders (e.g., 0.0005 = 0.05%)
                "limit_order_offset_sell": float(os.getenv("LIMIT_ORDER_OFFSET_SELL", 0.0005)),
            },
            "order_book": {
                # Number of bid/ask levels to fetch
                "depth": 10,
                # Ask volume / Bid volume ratio threshold to indicate sell pressure
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
                "stoch_rsi_period": 14,
                "stoch_rsi_k_period": 3,
                "stoch_rsi_d_period": 3,
                # --- Trend/Momentum ---
                "macd_short_period": 12,
                "macd_long_period": 26,
                "macd_signal_period": 9,
                # --- Average True Range (for SL/TP) ---
                "atr_period": 14,
            },
            "risk_management": {
                # Percentage of available quote balance to use for each order (e.g., 0.01 = 1%)
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
                # Defaults to MarkPrice if not specified or None.
                "sl_trigger_by": "MarkPrice",
                "tp_trigger_by": "MarkPrice",
                # --- Trailing Stop Loss (Experimental) ---
                "enable_trailing_stop_loss": True, # Enable/disable TSL feature
                # Trail distance as a percentage (e.g., 0.003 = 0.3%). Used only if TSL is enabled.
                "trailing_stop_loss_percentage": 0.003,
                # --- Other Exit Conditions ---
                # Time limit in minutes to close a position if still open (0 or null to disable)
                "time_based_exit_minutes": 60,
                # --- Signal Adjustment (Optional) ---
                # Multiplier applied to order size based on signal strength (1.0 = no adjustment)
                "strong_signal_adjustment_factor": 1.0, # Applied if abs(score) >= STRONG_SIGNAL_THRESHOLD_ABS
                "weak_signal_adjustment_factor": 1.0,   # Applied if abs(score) < STRONG_SIGNAL_THRESHOLD_ABS
            },
        }
        # --- Environment Variable Overrides (Example) ---
        # Allow overriding specific config values via environment variables if needed
        if os.getenv("ORDER_SIZE_PERCENTAGE"):
             try:
                default_config["risk_management"]["order_size_percentage"] = float(os.environ["ORDER_SIZE_PERCENTAGE"])
                logger.info("Overrode order_size_percentage from environment variable.")
             except ValueError:
                 logger.warning("Invalid ORDER_SIZE_PERCENTAGE in environment, using default.")
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
            def is_positive(key: str, v: Any) -> bool:
                if not isinstance(v, (int, float)) or v <= 0:
                    raise ValueError(f"'{key}' must be a positive number, got: {v}")
                return True
            def is_non_negative(key: str, v: Any) -> bool:
                 if not isinstance(v, (int, float)) or v < 0:
                    raise ValueError(f"'{key}' must be a non-negative number, got: {v}")
                 return True
            def is_percentage(key: str, v: Any) -> bool:
                 if not isinstance(v, (int, float)) or not (0 < v <= 1): # Allow up to 100%
                    raise ValueError(f"'{key}' must be a number between 0 (exclusive) and 1 (inclusive), got: {v}")
                 return True
            def is_string(key: str, v: Any) -> bool:
                 if not isinstance(v, str) or not v:
                    raise ValueError(f"'{key}' must be a non-empty string, got: {v}")
                 return True
            def is_bool(key: str, v: Any) -> bool:
                 if not isinstance(v, bool):
                    raise ValueError(f"'{key}' must be a boolean (true/false), got: {v}")
                 return True
            def is_pos_int(key: str, v: Any) -> bool:
                 if not isinstance(v, int) or v <= 0:
                    raise ValueError(f"'{key}' must be a positive integer, got: {v}")
                 return True
            def is_non_neg_int(key: str, v: Any) -> bool:
                 if not isinstance(v, int) or v < 0:
                    raise ValueError(f"'{key}' must be a non-negative integer (0 or greater), got: {v}")
                 return True

            # --- Section Existence ---
            required_sections = {"exchange", "trading", "order_book", "indicators", "risk_management"}
            missing_sections = required_sections - self.config.keys()
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {missing_sections}")
            for section in required_sections:
                if not isinstance(self.config[section], dict):
                    raise ValueError(f"Configuration section '{section}' must be a dictionary.")

            # --- Exchange Section ---
            cfg_ex = self.config["exchange"]
            is_string("exchange.exchange_id", cfg_ex.get("exchange_id"))
            if cfg_ex["exchange_id"] not in ccxt.exchanges:
                raise ValueError(f"Unsupported exchange_id: '{cfg_ex['exchange_id']}'. Available: {ccxt.exchanges}")
            is_bool("exchange.testnet_mode", cfg_ex.get("testnet_mode"))

            # --- Trading Section ---
            cfg_tr = self.config["trading"]
            is_string("trading.symbol", cfg_tr.get("symbol"))
            is_string("trading.timeframe", cfg_tr.get("timeframe"))
            # Note: Timeframe validity checked later against exchange capabilities
            is_bool("trading.simulation_mode", cfg_tr.get("simulation_mode"))
            entry_type = cfg_tr.get("entry_order_type")
            if entry_type not in ["market", "limit"]:
                raise ValueError(f"trading.entry_order_type must be 'market' or 'limit', got: {entry_type}")
            is_non_negative("trading.limit_order_offset_buy", cfg_tr.get("limit_order_offset_buy"))
            is_non_negative("trading.limit_order_offset_sell", cfg_tr.get("limit_order_offset_sell"))

            # --- Order Book Section ---
            cfg_ob = self.config["order_book"]
            is_pos_int("order_book.depth", cfg_ob.get("depth"))
            is_positive("order_book.imbalance_threshold", cfg_ob.get("imbalance_threshold"))

            # --- Indicators Section ---
            cfg_ind = self.config["indicators"]
            period_keys = ["volatility_window", "ema_period", "rsi_period", "macd_short_period",
                           "macd_long_period", "macd_signal_period", "stoch_rsi_period",
                           "stoch_rsi_k_period", "stoch_rsi_d_period", "atr_period"]
            for key in period_keys:
                is_pos_int(f"indicators.{key}", cfg_ind.get(key))
            is_non_negative("indicators.volatility_multiplier", cfg_ind.get("volatility_multiplier"))
            # Check MACD period relationship
            if cfg_ind['macd_short_period'] >= cfg_ind['macd_long_period']:
                raise ValueError("indicators.macd_short_period must be less than macd_long_period.")

            # --- Risk Management Section ---
            cfg_rm = self.config["risk_management"]
            is_bool("risk_management.use_atr_sl_tp", cfg_rm.get("use_atr_sl_tp"))
            if cfg_rm["use_atr_sl_tp"]:
                is_positive("risk_management.atr_sl_multiplier", cfg_rm.get("atr_sl_multiplier"))
                is_positive("risk_management.atr_tp_multiplier", cfg_rm.get("atr_tp_multiplier"))
            else:
                is_percentage("risk_management.stop_loss_percentage", cfg_rm.get("stop_loss_percentage"))
                is_percentage("risk_management.take_profit_percentage", cfg_rm.get("take_profit_percentage"))

            # Validate trigger price types (allow None or specific strings)
            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None] # Add more if needed based on Bybit docs
            sl_trig = cfg_rm.get("sl_trigger_by")
            tp_trig = cfg_rm.get("tp_trigger_by")
            if sl_trig not in valid_triggers:
                raise ValueError(f"Invalid risk_management.sl_trigger_by: '{sl_trig}'. Must be one of: {valid_triggers}")
            if tp_trig not in valid_triggers:
                raise ValueError(f"Invalid risk_management.tp_trigger_by: '{tp_trig}'. Must be one of: {valid_triggers}")

            is_bool("risk_management.enable_trailing_stop_loss", cfg_rm.get("enable_trailing_stop_loss"))
            if cfg_rm["enable_trailing_stop_loss"]:
                tsl_pct = cfg_rm.get("trailing_stop_loss_percentage")
                # TSL percentage must be positive and typically less than 1 (100%)
                if not (isinstance(tsl_pct, (int, float)) and 0 < tsl_pct < 1):
                     raise ValueError("risk_management.trailing_stop_loss_percentage must be a positive number less than 1 (e.g., 0.005 for 0.5%) when TSL is enabled.")

            is_percentage("risk_management.order_size_percentage", cfg_rm.get("order_size_percentage"))
            is_pos_int("risk_management.max_open_positions", cfg_rm.get("max_open_positions"))
            is_non_neg_int("risk_management.time_based_exit_minutes", cfg_rm.get("time_based_exit_minutes"))
            is_positive("risk_management.strong_signal_adjustment_factor", cfg_rm.get("strong_signal_adjustment_factor"))
            is_positive("risk_management.weak_signal_adjustment_factor", cfg_rm.get("weak_signal_adjustment_factor"))

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
        elif not creds_found and not self.simulation_mode:
             # This case should be caught above, but acts as a safeguard
             logger.warning(f"{Fore.YELLOW}API Key/Secret missing, but simulation_mode is False. Forcing internal simulation mode.{Style.RESET_ALL}")
             self.simulation_mode = True

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            # Base configuration for the exchange instance
            exchange_config = {
                'enableRateLimit': True, # Essential for respecting API limits
                'options': {
                    'defaultType': 'swap' if 'bybit' in self.exchange_id else 'future', # Default to swap for Bybit, future otherwise (adjust if needed)
                    'adjustForTimeDifference': True, # Corrects minor clock drifts
                    # Add other exchange-specific options if necessary
                    # 'warnOnFetchOpenOrdersWithoutSymbol': False, # Example option
                }
            }

            # Add API credentials only if NOT in internal simulation mode
            if not self.simulation_mode:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                logger.debug("API credentials added to exchange configuration.")
            else:
                logger.debug("Running in internal simulation mode; API credentials not used for exchange init.")


            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT's unified method (if not in internal simulation)
            if not self.simulation_mode and self.testnet_mode:
                try:
                    logger.info("Attempting to switch exchange instance to Testnet mode...")
                    exchange.set_sandbox_mode(True)
                    logger.info(f"{Fore.YELLOW}Exchange sandbox mode explicitly enabled.{Style.RESET_ALL}")
                except ccxt.NotSupported:
                     logger.warning(f"{Fore.YELLOW}Exchange {self.exchange_id} does not support unified set_sandbox_mode via CCXT. Testnet functionality depends on API key type.{Style.RESET_ALL}")
                except Exception as sandbox_err:
                     logger.error(f"{Fore.RED}Error setting sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}")

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
                try:
                    server_time_ms = exchange.fetch_time()
                    server_time_str = pd.to_datetime(server_time_ms, unit='ms', utc=True).isoformat()
                    logger.debug(f"Exchange time crystal synchronized: {server_time_str}")
                except ccxt.AuthenticationError as auth_err:
                    logger.critical(f"{Fore.RED}Authentication check failed after loading markets: {auth_err}. Check API keys/permissions. Aborting.{Style.RESET_ALL}")
                    sys.exit(1)
                except Exception as time_err:
                    # Warn but don't necessarily abort for time fetch failure
                    logger.warning(f"{Fore.YELLOW}Optional check failed: Could not fetch server time. Proceeding cautiously. Error: {time_err}{Style.RESET_ALL}")

            logger.info(f"{Fore.GREEN}Connection established and configured for {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Fore.RED}Authentication failed for {self.exchange_id.upper()}: {e}. Check API keys/permissions in .env file. Aborting.{Style.RESET_ALL}")
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

    def _load_market_info(self) -> None:
        """Loads and caches essential market information (precision, limits) for the trading symbol."""
        logger.debug(f"Loading market details for {self.symbol}...")
        try:
            self.market_info = self.exchange.market(self.symbol)
            if not self.market_info: # Should not happen if symbol validation passed, but check anyway
                raise ValueError(f"Market info for symbol '{self.symbol}' could not be retrieved after loading markets.")

            # --- Pre-validate crucial precision and limit information ---
            precision = self.market_info.get('precision', {})
            limits = self.market_info.get('limits', {})
            amount_precision = precision.get('amount')
            price_precision = precision.get('price')
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            if amount_precision is None or price_precision is None:
                logger.warning(f"{Fore.YELLOW}Precision info missing for {self.symbol}. Relying on CCXT defaults, but trades might fail. Market Info: {self.market_info}{Style.RESET_ALL}")
                # Assign defaults if missing to avoid None errors later, though this is risky
                self.market_info['precision']['amount'] = self.market_info['precision'].get('amount', 8) # Default to 8 decimal places for amount
                self.market_info['precision']['price'] = self.market_info['precision'].get('price', 8)   # Default to 8 decimal places for price
            if min_amount is None:
                 logger.warning(f"{Fore.YELLOW}Minimum order amount limit missing for {self.symbol}. Cannot enforce minimum amount.{Style.RESET_ALL}")
            if min_cost is None:
                 logger.warning(f"{Fore.YELLOW}Minimum order cost limit missing for {self.symbol}. Cannot enforce minimum cost.{Style.RESET_ALL}")

            # Log key details for reference
            logger.info(f"Market Details for {self.symbol}: "
                        f"Price Precision={price_precision}, Amount Precision={amount_precision}, "
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
                # Log with appropriate precision based on market info
                price_prec = self.market_info['precision']['price']
                # Use safe formatting in case precision is not an int
                price_format = f".{int(price_prec)}f" if isinstance(price_prec, (int, float)) and price_prec >= 0 else ""
                logger.debug(f"Current market price ({self.symbol}): {price:{price_format}}")
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
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            result = {'bids': bids, 'asks': asks, 'imbalance': None}

            # Ensure bids/asks are lists of lists/tuples with at least price and volume
            valid_bids = [bid for bid in bids if isinstance(bid, (list, tuple)) and len(bid) >= 2 and bid[1] is not None]
            valid_asks = [ask for ask in asks if isinstance(ask, (list, tuple)) and len(ask) >= 2 and ask[1] is not None]

            if valid_bids and valid_asks:
                # Calculate total volume within the fetched depth
                # Use try-except for float conversion for extra safety
                try:
                    bid_volume = sum(float(bid[1]) for bid in valid_bids)
                    ask_volume = sum(float(ask[1]) for ask in valid_asks)
                except (ValueError, TypeError):
                    logger.error(f"{Fore.RED}Error converting order book volumes to float for imbalance calculation.{Style.RESET_ALL}", exc_info=True)
                    return result # Return with imbalance=None

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
                    logger.debug(f"Order Book ({self.symbol}) Imbalance: N/A (Zero Bid/Ask Volume)")
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
        """
        # Determine the minimum number of candles required for all indicators
        if limit is None:
             required_limit = max(
                 self.volatility_window + 1 if self.volatility_window else 0,
                 self.ema_period + 1 if self.ema_period else 0,
                 self.rsi_period + 2 if self.rsi_period else 0, # RSI needs one prior diff
                 (self.macd_long_period + self.macd_signal_period) if (self.macd_long_period and self.macd_signal_period) else 0,
                 (self.stoch_rsi_period + self.stoch_rsi_k_period + self.stoch_rsi_d_period + 1) if self.stoch_rsi_period else 0, # Needs buffer for RSI calc within
                 self.atr_period + 1 if self.atr_period else 0
             ) + 20 # Add a safety buffer

             # Ensure a minimum fetch limit if all periods are very small
             required_limit = max(required_limit, 50)
        else:
             required_limit = limit

        logger.debug(f"Fetching approximately {required_limit} historical candles for {self.symbol} ({self.timeframe})...")
        try:
            # Fetch OHLCV data
            # Add 'since' parameter? Could be useful for very long histories, but limit is usually sufficient.
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
            min_required_len = required_limit - 25 # Allow for some buffer loss
            if final_len < min_required_len :
                 logger.warning(f"{Fore.YELLOW}Insufficient valid historical data after cleaning for {self.symbol}. Fetched {initial_len}, valid {final_len}, needed ~{min_required_len}. Indicators might be inaccurate or fail.{Style.RESET_ALL}")
                 # Decide whether to return None or the potentially insufficient DataFrame
                 # Returning None might be safer if indicators are critical
                 # return None
                 # Or return the df and let indicator functions handle insufficient data:
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
        Returns 0.0 if balance cannot be fetched or is zero.
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
            logger.warning(f"{Fore.YELLOW}Simulation mode: Returning dummy balance {dummy_balance:.4f} {quote_currency}.{Style.RESET_ALL}")
            return dummy_balance

        try:
            balance_data = self.exchange.fetch_balance()
            free_balance: Optional[float] = None

            # --- Try Standard 'free' Balance ---
            if quote_currency in balance_data:
                 free_balance = balance_data[quote_currency].get('free')
                 if free_balance is not None:
                     logger.debug(f"Found standard 'free' balance: {free_balance} {quote_currency}")

            # --- Fallback for Unified/Other Structures (Example: Bybit V5 Unified) ---
            # Check if 'free' was missing or zero, and try alternative fields from 'info'
            if free_balance is None or free_balance == 0.0:
                logger.debug(f"'free' balance for {quote_currency} is zero or missing. Checking 'info' for alternatives (e.g., Bybit V5 unified)...")
                # Access the raw 'info' structure - THIS IS EXCHANGE-SPECIFIC
                info_data = balance_data.get('info', {})
                # Example path for Bybit V5 Unified Margin account balance
                # Path: info -> result -> list -> [0] -> totalAvailableBalance
                try:
                    if isinstance(info_data, dict) and \
                       isinstance(info_data.get('result'), dict) and \
                       isinstance(info_data['result'].get('list'), list) and \
                       len(info_data['result']['list']) > 0 and \
                       isinstance(info_data['result']['list'][0], dict):

                        # Check if it's a Unified account first (avoids using spot balance accidentally)
                        account_type = info_data['result']['list'][0].get('accountType')
                        if account_type == 'UNIFIED': # Or 'CONTRACT' depending on Bybit version/setup
                            usable_balance_str = info_data['result']['list'][0].get('totalAvailableBalance')
                            if usable_balance_str is not None:
                                logger.debug(f"Found Bybit V5 Unified 'totalAvailableBalance': {usable_balance_str}")
                                free_balance = float(usable_balance_str)
                            else:
                                # Also check coin-specific availableToWithdraw if total is missing
                                coin_list = info_data['result']['list'][0].get('coin', [])
                                for coin_entry in coin_list:
                                    if isinstance(coin_entry, dict) and coin_entry.get('coin') == quote_currency:
                                        avail_withdraw = coin_entry.get('availableToWithdraw')
                                        if avail_withdraw is not None:
                                             logger.debug(f"Using coin-specific 'availableToWithdraw' for {quote_currency}: {avail_withdraw}")
                                             free_balance = float(avail_withdraw)
                                             break
                        else:
                             logger.debug(f"Account type is '{account_type}', not Unified. Standard 'free' balance applies (or is zero).")

                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as info_err:
                    logger.warning(f"{Fore.YELLOW}Could not parse expected structure in balance 'info' for alternative balance: {info_err}. Using standard 'free' balance (or 0.0). Raw info: {info_data}{Style.RESET_ALL}")

            # --- Final Conversion and Return ---
            final_balance = 0.0
            if free_balance is not None:
                try:
                    final_balance = float(free_balance)
                    if final_balance < 0: final_balance = 0.0 # Ensure balance is not negative
                except (ValueError, TypeError):
                    logger.error(f"{Fore.RED}Could not convert final balance value '{free_balance}' to float for {quote_currency}. Returning 0.0{Style.RESET_ALL}")
                    final_balance = 0.0

            logger.info(f"Fetched available balance: {final_balance:.4f} {quote_currency}")
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
        Returns the order dictionary or None if not found or error occurs.
        """
        if not order_id:
            logger.warning("fetch_order_status called with empty order_id.")
            return None

        target_symbol = symbol or self.symbol
        logger.debug(f"Fetching status for order {order_id} ({target_symbol})...")

        # Handle simulation mode: return a placeholder if needed, or manage simulated orders
        if self.simulation_mode:
            # Find the simulated order in our state (this assumes state is managed correctly)
            # This part might need refinement based on how simulated orders are stored/tracked
            for pos in self.open_positions:
                 if pos['id'] == order_id:
                    # Return a simulated 'closed' status for market orders after a short delay?
                    # Or always 'open' for limits until explicitly managed?
                    # For now, return a basic simulated structure if found, otherwise None
                    logger.debug(f"[SIMULATION] Returning cached/simulated status for order {order_id}.")
                    # This is a very basic simulation, real fetch is bypassed
                    sim_status = pos.get('status', STATUS_UNKNOWN)
                    sim_avg = pos['entry_price'] if sim_status == STATUS_ACTIVE else None
                    sim_filled = pos['size'] if sim_status == STATUS_ACTIVE else 0.0
                    # Return a structure mimicking ccxt order dict
                    return {
                        'id': order_id, 'symbol': target_symbol, 'status': sim_status,
                        'filled': sim_filled, 'average': sim_avg, 'info': {'simulated': True}
                        # Add other fields as needed for simulation consistency
                    }
            logger.warning(f"[SIMULATION] Simulated order {order_id} not found in current state.")
            return None # Not found in simulation state

        # --- Live Mode ---
        try:
            # params = {} # Add if needed (e.g., {'category': 'linear'} for Bybit V5)
            order_info = self.exchange.fetch_order(order_id, target_symbol) #, params=params)

            # Log key details
            status = order_info.get('status', STATUS_UNKNOWN)
            filled = order_info.get('filled', 0.0)
            avg_price = order_info.get('average')
            remaining = order_info.get('remaining', 0.0)
            amount = order_info.get('amount', 0.0)
            order_type = order_info.get('type')
            order_side = order_info.get('side')

            logger.debug(f"Order {order_id} ({order_type} {order_side}): Status={status}, Filled={filled}/{amount}, AvgPrice={avg_price}, Remaining={remaining}")

            return order_info

        except ccxt.OrderNotFound:
            # This is a common case, treat as non-error, indicates order is final (closed/canceled) or never existed
            logger.warning(f"{Fore.YELLOW}Order {order_id} not found on exchange ({target_symbol}). Assumed closed, cancelled, or invalid ID.{Style.RESET_ALL}")
            return None # Return None, let the calling function decide how to interpret this
        except ccxt.NetworkError as e:
             # Allow retry decorator to handle network issues
             logger.warning(f"Network error fetching status for order {order_id}: {e}. Retrying if possible.")
             raise e
        except ccxt.ExchangeError as e:
            # Log other exchange errors, but usually don't retry fetch_order
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
            # logger.debug(f"Insufficient data for volatility (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            log_returns = np.log(close_prices / close_prices.shift(1))
            volatility = log_returns.rolling(window=window).std(ddof=0).iloc[-1] # ddof=0 for population std dev
            return float(volatility) if pd.notna(volatility) else None
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}", exc_info=False) # Avoid spamming logs
            return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates the Exponential Moving Average (EMA)."""
        if close_prices is None or period <= 0: return None
        min_len = period # EMA technically needs 'period' points for good initialization
        if len(close_prices) < min_len:
            # logger.debug(f"Insufficient data for EMA (need {min_len}, got {len(close_prices)}).")
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
            # logger.debug(f"Insufficient data for RSI (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            # Use Exponential Moving Average for smoothing gains and losses
            avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

            # Calculate Relative Strength (RS)
            # Add epsilon to avoid division by zero
            rs = avg_gain / (avg_loss + 1e-9)

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
        min_len = long_period + signal_period # Rough minimum length for meaningful values
        if len(close_prices) < min_len:
            # logger.debug(f"Insufficient data for MACD (need ~{min_len}, got {len(close_prices)}).")
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

            # Return None if any value is NaN (e.g., due to insufficient data)
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
            # logger.debug(f"Insufficient data for StochRSI (need ~{min_len}, got {len(close_prices)}).")
            return None, None
        try:
            # --- Calculate RSI Series ---
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
            loss = -delta.where(delta < 0, 0.0).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
            rs = gain / (loss + 1e-9)
            rsi_series = (100.0 - (100.0 / (1.0 + rs))).dropna() # Drop initial NaNs from RSI calc

            if len(rsi_series) < stoch_period:
                # logger.debug(f"Insufficient RSI series length for StochRSI window (need {stoch_period}, got {len(rsi_series)}).")
                return None, None

            # --- Calculate Stochastic of RSI ---
            min_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).min()
            max_rsi = rsi_series.rolling(window=stoch_period, min_periods=stoch_period).max()
            # Add epsilon to denominator to avoid division by zero if max == min
            stoch_rsi = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-9)).clip(0, 100) # Clip ensures 0-100 range

            # --- Calculate %K and %D ---
            # %K is typically a smoothed version of stoch_rsi (SMA or EMA)
            stoch_k = stoch_rsi.rolling(window=k_period, min_periods=k_period).mean()
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
            # logger.debug(f"Insufficient data for ATR (need {min_len}, got {len(close_prices)}).")
            return None
        try:
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift(1))
            low_close = np.abs(low_prices - close_prices.shift(1))

            # Combine the three components to find the True Range (TR)
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False) # Use skipna=False? Check implications.

            # Calculate ATR using Exponential Moving Average (common method)
            # Can also use Simple Moving Average (SMA) or Wilder's Smoothing
            atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().iloc[-1]
            # Alternative using Wilder's Smoothing (alpha = 1/period):
            # atr = tr.ewm(com=period-1, adjust=False, min_periods=period).mean().iloc[-1]

            return float(atr) if pd.notna(atr) else None
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}", exc_info=False)
            return None


    # --- Trading Logic & Order Management Methods ---

    def calculate_order_size(self, current_price: float) -> float:
        """
        Calculates the order size in the QUOTE currency based on available balance,
        percentage risk, and optional volatility adjustment.
        Returns the size in quote currency, or 0.0 if checks fail.
        """
        if self.market_info is None:
            logger.error("Market info not loaded, cannot calculate order size.")
            return 0.0
        if current_price <= 1e-9: # Avoid division by zero
             logger.error(f"Current price ({current_price}) is too low for order size calculation.")
             return 0.0

        quote_currency = self.market_info['quote']
        base_currency = self.market_info['base']
        amount_precision_digits = self.market_info['precision'].get('amount') # Number of decimal places for amount
        price_precision_digits = self.market_info['precision'].get('price')   # Number of decimal places for price

        # Fetch available balance
        balance = self.fetch_balance(currency_code=quote_currency)
        if balance <= 0:
            logger.warning(f"{Fore.YELLOW}Insufficient balance ({balance:.4f} {quote_currency}) for order size calculation. Need more {quote_currency}.{Style.RESET_ALL}")
            return 0.0

        # --- Calculate Base Order Size ---
        base_order_size_quote = balance * self.order_size_percentage
        if base_order_size_quote <= 0:
            logger.warning(f"Calculated base order size is zero or negative ({base_order_size_quote:.4f} {quote_currency}). Balance: {balance}, Percentage: {self.order_size_percentage}")
            return 0.0

        final_size_quote = base_order_size_quote

        # --- Optional Volatility Adjustment ---
        # Note: This requires fetching historical data again if not passed in.
        # Consider calculating indicators once per loop and passing them.
        if self.volatility_multiplier is not None and self.volatility_multiplier > 0:
            logger.debug("Applying volatility adjustment to order size...")
            # Fetch recent data specifically for volatility calculation
            hist_data_vol = self.fetch_historical_data(limit=self.volatility_window + 5) # Fetch enough candles
            volatility = None
            if hist_data_vol is not None and not hist_data_vol.empty:
                 volatility = self.calculate_volatility(hist_data_vol['close'], self.volatility_window)

            if volatility is not None and volatility > 1e-9:
                # Inverse relationship: higher volatility -> smaller size factor
                # Multiplier * 100 assumes volatility is ~0.01 for 1% move, adjust scaling if needed
                size_factor = 1 / (1 + volatility * self.volatility_multiplier * 100)
                # Clamp the factor to avoid extreme adjustments (e.g., 0.25x to 1.75x)
                size_factor = max(0.25, min(1.75, size_factor))
                final_size_quote = base_order_size_quote * size_factor
                logger.info(f"Volatility ({volatility:.5f}) adjustment factor: {size_factor:.3f}. "
                            f"Adjusted size: {final_size_quote:.4f} {quote_currency} (Base: {base_order_size_quote:.4f})")
            else:
                logger.debug(f"Volatility adjustment skipped: Volatility N/A ({volatility}) or multiplier is zero.")
        else:
            logger.debug("Volatility adjustment disabled.")

        # --- Convert Quote Size to Base Amount ---
        order_size_base = final_size_quote / current_price

        # --- Apply Exchange Precision and Check Limits ---
        try:
            # Apply Amount Precision
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base)
            amount_precise = float(amount_precise_str)

            # Apply Price Precision (for cost calculation estimate)
            price_precise_str = self.exchange.price_to_precision(self.symbol, current_price)
            price_precise = float(price_precise_str)

            # Check Minimum Amount Limit
            min_amount = self.market_info.get('limits', {}).get('amount', {}).get('min')
            if min_amount is not None and amount_precise < min_amount:
                logger.warning(f"{Fore.YELLOW}Calculated order amount {amount_precise:.{amount_precision_digits}f} {base_currency} is below minimum required {min_amount}. Setting size to 0.{Style.RESET_ALL}")
                return 0.0

            # Check Minimum Cost Limit (Estimated Cost)
            estimated_cost = amount_precise * price_precise
            min_cost = self.market_info.get('limits', {}).get('cost', {}).get('min')
            if min_cost is not None and estimated_cost < min_cost:
                logger.warning(f"{Fore.YELLOW}Estimated order cost {estimated_cost:.{price_precision_digits}f} {quote_currency} is below minimum required {min_cost}. Calculated amount: {amount_precise}, Price: {price_precise}. Setting size to 0.{Style.RESET_ALL}")
                return 0.0

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Error applying precision or checking limits for order size: {e}{Style.RESET_ALL}", exc_info=True)
            return 0.0

        # Log the final calculated size in both quote and base currencies
        logger.info(f"{Fore.CYAN}Calculated final order size: {final_size_quote:.{price_precision_digits or 4}f} {quote_currency} "
                    f"({amount_precise:.{amount_precision_digits}f} {base_currency}){Style.RESET_ALL}")

        # Return the calculated size in QUOTE currency, as entry function might use it
        return final_size_quote


    def _calculate_sl_tp_prices(self, entry_price: float, side: str, current_price: float, atr: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates Stop Loss (SL) and Take Profit (TP) prices based on configuration.
        Uses either ATR or fixed percentages from the entry price.
        Includes sanity checks against the current price.

        Args:
            entry_price: The estimated or actual entry price of the position.
            side: 'buy' or 'sell'.
            current_price: The current market price, used for sanity checks.
            atr: The current ATR value (required if use_atr_sl_tp is True).

        Returns:
            A tuple containing (stop_loss_price, take_profit_price).
            Values can be None if calculation is not possible or not configured.
        """
        stop_loss_price: Optional[float] = None
        take_profit_price: Optional[float] = None

        if self.market_info is None: # Should not happen
            logger.error("Market info not loaded, cannot calculate SL/TP.")
            return None, None

        price_precision_digits = self.market_info['precision']['price']
        # Helper to format price for logging
        def format_price(p):
            if p is None: return "N/A"
            fmt = f".{int(price_precision_digits)}f" if isinstance(price_precision_digits, (int, float)) and price_precision_digits >= 0 else ""
            return f"{p:{fmt}}"

        # --- Calculate based on ATR ---
        if self.use_atr_sl_tp:
            if atr is None or atr <= 1e-12: # Check for valid ATR
                logger.warning(f"{Fore.YELLOW}ATR SL/TP enabled, but ATR is invalid ({atr}). Cannot calculate SL/TP.{Style.RESET_ALL}")
                return None, None
            if not self.atr_sl_multiplier or not self.atr_tp_multiplier:
                 logger.warning(f"{Fore.YELLOW}ATR SL/TP enabled, but multipliers (SL={self.atr_sl_multiplier}, TP={self.atr_tp_multiplier}) are invalid. Cannot calculate SL/TP.{Style.RESET_ALL}")
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
                 logger.warning(f"{Fore.YELLOW}Fixed Percentage SL/TP enabled, but 'stop_loss_percentage' or 'take_profit_percentage' not set in config. Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None
            if not (0 < self.base_stop_loss_pct < 1) or not (0 < self.base_take_profit_pct < 1):
                 logger.warning(f"{Fore.YELLOW}Fixed Percentage SL/TP enabled, but values (SL={self.base_stop_loss_pct}, TP={self.base_take_profit_pct}) are invalid (must be > 0 and < 1). Cannot calculate SL/TP.{Style.RESET_ALL}")
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

        # --- Apply Exchange Price Precision ---
        try:
            if stop_loss_price is not None and stop_loss_price > 0: # Ensure positive price
                stop_loss_price = float(self.exchange.price_to_precision(self.symbol, stop_loss_price))
            else:
                logger.warning(f"Calculated stop loss price ({stop_loss_price}) is zero or negative. Setting SL to None.")
                stop_loss_price = None

            if take_profit_price is not None and take_profit_price > 0: # Ensure positive price
                take_profit_price = float(self.exchange.price_to_precision(self.symbol, take_profit_price))
            else:
                 logger.warning(f"Calculated take profit price ({take_profit_price}) is zero or negative. Setting TP to None.")
                 take_profit_price = None

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}Error applying precision to SL/TP prices: {e}. Original SL={stop_loss_price}, TP={take_profit_price}. Setting both to None.{Style.RESET_ALL}")
            return None, None

        # --- Sanity Check & Adjustment against Current Price ---
        # Prevent placing SL/TP orders that would trigger immediately
        if stop_loss_price is not None:
            if side == "buy" and stop_loss_price >= current_price * 0.999: # Allow tiny margin for fluctuation
                adj_sl = current_price * (1 - (self.base_stop_loss_pct or 0.001)) # Adjust slightly below current
                logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) is at or above current price ({format_price(current_price)}) for LONG. Adjusting SL slightly below current price to ~{format_price(adj_sl)}.{Style.RESET_ALL}")
                stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl))
            elif side == "sell" and stop_loss_price <= current_price * 1.001: # Allow tiny margin
                 adj_sl = current_price * (1 + (self.base_stop_loss_pct or 0.001)) # Adjust slightly above current
                 logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) is at or below current price ({format_price(current_price)}) for SHORT. Adjusting SL slightly above current price to ~{format_price(adj_sl)}.{Style.RESET_ALL}")
                 stop_loss_price = float(self.exchange.price_to_precision(self.symbol, adj_sl))

        if take_profit_price is not None:
            if side == "buy" and take_profit_price <= current_price * 1.001:
                 adj_tp = current_price * (1 + (self.base_take_profit_pct or 0.001)) # Adjust slightly above current
                 logger.warning(f"{Fore.YELLOW}Calculated TP ({format_price(take_profit_price)}) is at or below current price ({format_price(current_price)}) for LONG. Adjusting TP slightly above current price to ~{format_price(adj_tp)}.{Style.RESET_ALL}")
                 take_profit_price = float(self.exchange.price_to_precision(self.symbol, adj_tp))
            elif side == "sell" and take_profit_price >= current_price * 0.999:
                 adj_tp = current_price * (1 - (self.base_take_profit_pct or 0.001)) # Adjust slightly below current
                 logger.warning(f"{Fore.YELLOW}Calculated TP ({format_price(take_profit_price)}) is at or above current price ({format_price(current_price)}) for SHORT. Adjusting TP slightly below current price to ~{format_price(adj_tp)}.{Style.RESET_ALL}")
                 take_profit_price = float(self.exchange.price_to_precision(self.symbol, adj_tp))

        # Final check: Ensure SL and TP are logical relative to each other
        if stop_loss_price is not None and take_profit_price is not None:
             if side == "buy" and stop_loss_price >= take_profit_price:
                  logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) >= TP ({format_price(take_profit_price)}) for LONG. Setting TP to None to avoid conflict.{Style.RESET_ALL}")
                  take_profit_price = None
             elif side == "sell" and stop_loss_price <= take_profit_price:
                  logger.warning(f"{Fore.YELLOW}Calculated SL ({format_price(stop_loss_price)}) <= TP ({format_price(take_profit_price)}) for SHORT. Setting TP to None to avoid conflict.{Style.RESET_ALL}")
                  take_profit_price = None


        logger.debug(f"Final calculated SL={format_price(stop_loss_price)}, TP={format_price(take_profit_price)} for {side} entry near {format_price(entry_price)}")
        return stop_loss_price, take_profit_price


    def compute_trade_signal_score(self, price: float, indicators: Dict[str, Optional[Union[float, Tuple[Optional[float], ...]]]], orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        """
        Computes a simple trade signal score based on configured indicators and order book imbalance.

        Score Contributions (Example):
        - Order Book Imbalance: +1 (Buy Pressure), -1 (Sell Pressure)
        - Price vs EMA: +1 (Above), -1 (Below)
        - RSI: +1 (Oversold), -1 (Overbought)
        - MACD: +1 (Line > Signal), -1 (Line <= Signal)
        - StochRSI: +1 (Oversold), -1 (Overbought)

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
        # Imbalance = Ask Volume / Bid Volume
        # Low imbalance (< 1/thresh) suggests buy pressure (fewer asks relative to bids)
        # High imbalance (> thresh) suggests sell pressure (more asks relative to bids)
        if orderbook_imbalance is not None:
            # Ensure threshold is valid
            if self.imbalance_threshold <= 0:
                 logger.warning("Order book imbalance_threshold must be positive. Skipping OB signal.")
                 reasons.append(f"{Fore.WHITE}[ 0.0] OB Invalid Threshold{Style.RESET_ALL}")
            else:
                 imb_buy_thresh = 1.0 / self.imbalance_threshold # e.g., if thresh=1.5, buy_thresh=0.67
                 if orderbook_imbalance < imb_buy_thresh:
                     score += 1.0
                     reasons.append(f"{Fore.GREEN}[+1.0] OB Buy Pressure (Imb: {orderbook_imbalance:.2f} < {imb_buy_thresh:.2f}){Style.RESET_ALL}")
                 elif orderbook_imbalance > self.imbalance_threshold:
                     score -= 1.0
                     reasons.append(f"{Fore.RED}[-1.0] OB Sell Pressure (Imb: {orderbook_imbalance:.2f} > {self.imbalance_threshold:.2f}){Style.RESET_ALL}")
                 else: # Between thresholds
                     reasons.append(f"{Fore.WHITE}[ 0.0] OB Balanced (Imb: {orderbook_imbalance:.2f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] OB Data N/A{Style.RESET_ALL}")

        # --- 2. EMA Trend ---
        ema = indicators.get('ema')
        if ema is not None and ema > 1e-9: # Check if EMA is valid and not zero
            ema_upper_bound = ema * (1 + EMA_THRESHOLD_MULTIPLIER)
            ema_lower_bound = ema * (1 - EMA_THRESHOLD_MULTIPLIER)
            if price > ema_upper_bound:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] Price > EMA ({price:.{self.market_info['precision']['price']}f} > {ema:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}")
            elif price < ema_lower_bound:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] Price < EMA ({price:.{self.market_info['precision']['price']}f} < {ema:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}")
            else: # Price is within the threshold range of EMA
                reasons.append(f"{Fore.WHITE}[ 0.0] Price near EMA ({price:.{self.market_info['precision']['price']}f} ~ {ema:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}")

        # --- 3. RSI Momentum/OB/OS ---
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] RSI Oversold ({rsi:.1f} < {RSI_OVERSOLD}){Style.RESET_ALL}")
            elif rsi > RSI_OVERBOUGHT:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] RSI Overbought ({rsi:.1f} > {RSI_OVERBOUGHT}){Style.RESET_ALL}")
            else: # RSI is in the neutral zone
                reasons.append(f"{Fore.WHITE}[ 0.0] RSI Neutral ({RSI_OVERSOLD} <= {rsi:.1f} <= {RSI_OVERBOUGHT}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}")

        # --- 4. MACD Momentum/Cross ---
        macd_line, macd_signal = indicators.get('macd_line'), indicators.get('macd_signal')
        # Check if both values are valid numbers
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal: # Bullish cross / MACD above signal
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] MACD Line > Signal ({macd_line:.4f} > {macd_signal:.4f}){Style.RESET_ALL}")
            else: # Bearish cross / MACD below signal
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] MACD Line <= Signal ({macd_line:.4f} <= {macd_signal:.4f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}")

        # --- 5. Stochastic RSI OB/OS ---
        stoch_k, stoch_d = indicators.get('stoch_k'), indicators.get('stoch_d')
        # Check if both %K and %D are valid
        if stoch_k is not None and stoch_d is not None:
            # Condition for Oversold: Both K and D below threshold
            if stoch_k < STOCH_OVERSOLD and stoch_d < STOCH_OVERSOLD:
                score += 1.0
                reasons.append(f"{Fore.GREEN}[+1.0] StochRSI Oversold (K={stoch_k:.1f}, D={stoch_d:.1f} < {STOCH_OVERSOLD}){Style.RESET_ALL}")
            # Condition for Overbought: Both K and D above threshold
            elif stoch_k > STOCH_OVERBOUGHT and stoch_d > STOCH_OVERBOUGHT:
                score -= 1.0
                reasons.append(f"{Fore.RED}[-1.0] StochRSI Overbought (K={stoch_k:.1f}, D={stoch_d:.1f} > {STOCH_OVERBOUGHT}){Style.RESET_ALL}")
            else: # StochRSI is in the neutral zone or K/D are crossing
                reasons.append(f"{Fore.WHITE}[ 0.0] StochRSI Neutral (K={stoch_k:.1f}, D={stoch_d:.1f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}")

        # --- Final Score Calculation ---
        # Round the raw score to the nearest integer
        final_score = int(round(score))
        logger.debug(f"Signal score calculation complete. Raw Score: {score:.2f}, Final Integer Score: {final_score}")

        return final_score, reasons


    @retry_api_call(max_retries=2, initial_delay=2) # Retry order placement a couple of times
    def place_entry_order(
        self, side: str, order_size_quote: float, confidence_level: int,
        order_type: str, current_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Places an entry order (market or limit) on the exchange.
        Includes Bybit V5 specific parameters for SL/TP if provided.

        Args:
            side: 'buy' or 'sell'.
            order_size_quote: The desired order size in the quote currency.
            confidence_level: The signal score associated with this entry.
            order_type: 'market' or 'limit'.
            current_price: The current market price (used for limit offset and base amount calc).
            stop_loss_price: The calculated stop loss price to send with the order.
            take_profit_price: The calculated take profit price to send with the order.

        Returns:
            The CCXT order dictionary if the order was placed successfully (or simulated),
            otherwise None.
        """
        if self.market_info is None:
             logger.error("Market info not available. Cannot place entry order.")
             return None
        if order_size_quote <= 0:
             logger.error("Order size (quote) is zero or negative. Cannot place entry order.")
             return None
        if current_price <= 1e-9:
             logger.error("Current price is zero or negative. Cannot place entry order.")
             return None

        base_currency = self.market_info['base']
        quote_currency = self.market_info['quote']
        amount_precision_digits = self.market_info['precision']['amount']
        price_precision_digits = self.market_info['precision']['price']

        # --- Calculate Base Amount and Apply Precision ---
        order_size_base = order_size_quote / current_price
        params = {} # Dictionary for extra parameters (like SL/TP)
        limit_price_precise: Optional[float] = None # For limit orders

        try:
            # Calculate precise amount
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base)
            amount_precise = float(amount_precise_str)
            logger.debug(f"Precise entry amount: {amount_precise:.{amount_precision_digits}f} {base_currency}")

            # Calculate precise limit price if applicable
            if order_type == "limit":
                offset = self.limit_order_entry_offset_pct_buy if side == 'buy' else self.limit_order_entry_offset_pct_sell
                # Buy limit below current price, Sell limit above current price
                price_factor = (1 - offset) if side == 'buy' else (1 + offset)
                limit_price = current_price * price_factor
                limit_price_precise_str = self.exchange.price_to_precision(self.symbol, limit_price)
                limit_price_precise = float(limit_price_precise_str)
                logger.debug(f"Precise limit price: {limit_price_precise:.{price_precision_digits}f} {quote_currency}")

            # --- Add Bybit V5 SL/TP & Trigger Parameters ---
            # Ensure prices are valid floats before formatting
            if stop_loss_price is not None and isinstance(stop_loss_price, (int, float)) and stop_loss_price > 0:
                params['stopLoss'] = self.exchange.price_to_precision(self.symbol, stop_loss_price) # Bybit V5 uses 'stopLoss', not 'stopLossPrice' in createOrder
                if self.sl_trigger_by: params['slTriggerBy'] = self.sl_trigger_by
            elif stop_loss_price is not None:
                 logger.warning(f"Invalid stop_loss_price ({stop_loss_price}) provided, SL parameter will not be sent.")

            if take_profit_price is not None and isinstance(take_profit_price, (int, float)) and take_profit_price > 0:
                params['takeProfit'] = self.exchange.price_to_precision(self.symbol, take_profit_price) # Bybit V5 uses 'takeProfit'
                if self.tp_trigger_by: params['tpTriggerBy'] = self.tp_trigger_by
            elif take_profit_price is not None:
                 logger.warning(f"Invalid take_profit_price ({take_profit_price}) provided, TP parameter will not be sent.")

            # --- Final Pre-flight Checks (Limits) ---
            # Re-check amount and cost limits with precise values
            limits = self.market_info.get('limits', {})
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            if min_amount is not None and amount_precise < min_amount:
                 logger.warning(f"{Fore.YELLOW}Entry amount {amount_precise:.{amount_precision_digits}f} {base_currency} is below minimum {min_amount}. Cannot place order.{Style.RESET_ALL}")
                 return None

            # Estimate cost using limit price (if available) or current price
            cost_check_price = limit_price_precise if limit_price_precise is not None else current_price
            estimated_cost = amount_precise * cost_check_price
            if min_cost is not None and estimated_cost < min_cost:
                 logger.warning(f"{Fore.YELLOW}Estimated entry cost {estimated_cost:.{price_precision_digits}f} {quote_currency} is below minimum {min_cost}. Cannot place order.{Style.RESET_ALL}")
                 return None

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}Error preparing entry order values/parameters: {e}{Style.RESET_ALL}", exc_info=True)
            return None

        # --- Log Order Details ---
        log_color = Fore.GREEN if side == 'buy' else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_info = f"SL={params.get('stopLoss', 'N/A')}" + (f" ({params['slTriggerBy']})" if 'slTriggerBy' in params else "")
        tp_info = f"TP={params.get('takeProfit', 'N/A')}" + (f" ({params['tpTriggerBy']})" if 'tpTriggerBy' in params else "")
        limit_price_info = f"Limit={limit_price_precise:.{price_precision_digits}f}" if limit_price_precise else f"Market (~{current_price:.{price_precision_digits}f})"

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            # Simulate fill price: use limit price if set, otherwise current price
            sim_entry_price = limit_price_precise if order_type == "limit" else current_price
            sim_cost = amount_precise * sim_entry_price
            # Simulate status: 'open' for limit, 'closed' (filled) for market
            sim_status = 'open' if order_type == 'limit' else 'closed'
            sim_filled = amount_precise if sim_status == 'closed' else 0.0
            sim_remaining = 0.0 if sim_status == 'closed' else amount_precise

            simulated_order = {
                "id": sim_id, "clientOrderId": sim_id,
                "timestamp": int(time.time() * 1000),
                "datetime": pd.to_datetime('now', utc=True).isoformat(),
                "symbol": self.symbol, "type": order_type, "side": side,
                "amount": amount_precise,
                "price": limit_price_precise, # Price of the limit order itself
                "average": sim_entry_price if sim_status == 'closed' else None, # Fill price
                "cost": sim_cost if sim_status == 'closed' else 0.0, # Filled cost
                "status": sim_status,
                "filled": sim_filled,
                "remaining": sim_remaining,
                "stopLossPrice": params.get('stopLoss'), # Store SL/TP info as sent
                "takeProfitPrice": params.get('takeProfit'),
                "info": { # Mimic CCXT structure
                    "simulated": True, "orderId": sim_id,
                    "confidence": confidence_level,
                    "initial_quote_size": order_size_quote,
                    "stopLoss": params.get('stopLoss'),
                    "takeProfit": params.get('takeProfit'),
                    "slTriggerBy": params.get('slTriggerBy'),
                    "tpTriggerBy": params.get('tpTriggerBy')
                    # Add other relevant fields from a real order if needed
                }
            }
            logger.info(
                f"{log_color}[SIMULATION] Placing {action_desc}: "
                f"ID: {sim_id}, Size: {amount_precise:.{amount_precision_digits}f} {base_currency}, "
                f"Price: {limit_price_info}, "
                f"Est. Value: {sim_cost:.2f} {quote_currency}, Confidence: {confidence_level}, {sl_info}, {tp_info}{Style.RESET_ALL}"
            )
            # Add custom info directly for state management consistency
            simulated_order['bot_custom_info'] = {"confidence": confidence_level, "initial_quote_size": order_size_quote}
            return simulated_order

        # --- Live Trading Mode ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order...")
            logger.info(f" -> Size: {amount_precise:.{amount_precision_digits}f} {base_currency}")
            logger.info(f" -> Price: {limit_price_info}")
            logger.info(f" -> Params: {params} (Includes SL/TP: {sl_info}, {tp_info})")

            order: Optional[Dict[str, Any]] = None
            try:
                if order_type == "market":
                    # Note: Bybit V5 might not accept price for market orders
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    if limit_price_precise is None: # Should not happen due to earlier checks
                         logger.error(f"{Fore.RED}Limit price not calculated. Cannot place live limit order.{Style.RESET_ALL}")
                         return None
                    order = self.exchange.create_limit_order(self.symbol, side, amount_precise, limit_price_precise, params=params)

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
                    info_sl = order.get('info', {}).get('stopLoss', 'N/A')
                    info_tp = order.get('info', {}).get('takeProfit', 'N/A')
                    info_sl_trig = order.get('info', {}).get('slTriggerBy', 'N/A')
                    info_tp_trig = order.get('info', {}).get('tpTriggerBy', 'N/A')

                    log_price_details = f"Limit: {oprice:.{price_precision_digits}f}" if oprice else "Market"
                    if oavg: log_price_details += f", AvgFill: {oavg:.{price_precision_digits}f}"

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: "
                        f"ID: {oid}, Type: {otype}, Side: {oside}, "
                        f"Amount: {oamt:.{amount_precision_digits}f}, Filled: {ofilled:.{amount_precision_digits}f}, "
                        f"{log_price_details}, Cost: {ocost:.2f} {quote_currency}, "
                        f"Status: {ostatus}, "
                        f"SL Confirmed: {info_sl} ({info_sl_trig}), TP Confirmed: {info_tp} ({info_tp_trig}), "
                        f"Confidence: {confidence_level}{Style.RESET_ALL}"
                    )
                    # Store bot-specific context with the order dict before returning
                    order['bot_custom_info'] = {"confidence": confidence_level, "initial_quote_size": order_size_quote}
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
                 return None
            except Exception as e:
                # Catch any unexpected Python errors during the process
                logger.error(f"{Fore.RED}LIVE {action_desc} Order Failed (Unexpected Python Error): {e}{Style.RESET_ALL}", exc_info=True)
                return None


    @retry_api_call(max_retries=1) # Only retry cancellation once if it fails initially
    def cancel_order_by_id(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Cancels a single open order by its ID.

        Args:
            order_id: The exchange's ID for the order to cancel.
            symbol: The trading symbol (defaults to the bot's symbol).

        Returns:
            True if the order was successfully cancelled or already not found,
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
            found = False
            for pos in self.open_positions:
                 if pos['id'] == order_id and pos['status'] == STATUS_PENDING_ENTRY:
                     pos['status'] = 'canceled' # Mark as canceled in simulation
                     pos['last_update_time'] = time.time()
                     logger.info(f"[SIMULATION] Marked simulated order {order_id} as canceled.")
                     found = True
                     break
            if not found:
                 logger.warning(f"[SIMULATION] Simulated order {order_id} not found or not in cancellable state.")
            # In simulation, always return True unless order wasn't found in a cancellable state?
            # For simplicity, return True, assuming cancellation would succeed or order is already gone.
            self._save_state() # Save updated state if changed
            return True

        # --- Live Mode ---
        else:
            try:
                # params = {} # Add exchange-specific params if needed (e.g., {'orderType': 'Limit'} for Bybit V5)
                response = self.exchange.cancel_order(order_id, target_symbol) #, params=params)
                logger.info(f"{Fore.YELLOW}---> Successfully initiated cancellation for order {order_id}. Response: {response}{Style.RESET_ALL}")
                return True
            except ccxt.OrderNotFound:
                # If the order is already gone (filled, cancelled, expired), consider it a success.
                logger.warning(f"{Fore.YELLOW}Order {order_id} not found during cancellation attempt (already closed/cancelled?). Treating as success.{Style.RESET_ALL}")
                return True
            except ccxt.NetworkError as e:
                 # Let the retry decorator handle transient network issues
                 logger.error(f"{Fore.RED}Network error cancelling order {order_id}: {e}. Retrying if possible.{Style.RESET_ALL}")
                 raise e # Re-raise to trigger retry
            except ccxt.ExchangeError as e:
                 # Log specific exchange errors during cancellation more prominently
                 # These are often non-retryable (e.g., "order already filled")
                 err_str = str(e).lower()
                 if "order has been filled" in err_str or "order is finished" in err_str:
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
             logger.info(f"[SIMULATION] Simulating cancellation of all orders for {target_symbol}.")
             sim_cancelled = 0
             for pos in self.open_positions:
                  if pos.get('symbol') == target_symbol and pos.get('status') == STATUS_PENDING_ENTRY:
                      pos['status'] = 'canceled'
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
                # params = {} # Add if needed e.g. {'category':'linear'}
                open_orders = self.exchange.fetch_open_orders(target_symbol) #, params=params)

                if not open_orders:
                    logger.info(f"No open orders found for {target_symbol}.")
                    return 0

                logger.warning(f"{Fore.YELLOW}Found {len(open_orders)} open order(s) for {target_symbol}. Attempting individual cancellation...{Style.RESET_ALL}")
                for order in open_orders:
                    order_id = order.get('id')
                    if not order_id:
                        logger.warning(f"Found open order without ID: {order}. Skipping cancellation.")
                        continue

                    if self.cancel_order_by_id(order_id, target_symbol):
                        cancelled_count += 1
                    else:
                        logger.error(f"Failed to cancel order {order_id} during bulk cancellation. Continuing...")
                    # Optional: Small delay between cancellations to avoid rate limits if cancelling many orders
                    # Consider a value slightly larger than exchange rate limit interval
                    time.sleep(max(0.2, self.exchange.rateLimit / 1000))

            # Fallback to cancelAllOrders if fetchOpenOrders is not supported or preferred
            elif self.exchange.has['cancelAllOrders']:
                 logger.warning(f"{Fore.YELLOW}Exchange lacks 'fetchOpenOrders', attempting unified 'cancelAllOrders' for {target_symbol}...{Style.RESET_ALL}")
                 # params = {} # Add if needed
                 response = self.exchange.cancel_all_orders(target_symbol) #, params=params)
                 logger.info(f"'cancelAllOrders' response: {response}")
                 # We don't know the exact count from this response typically
                 cancelled_count = -1 # Indicate unknown count, but action was attempted
            else:
                 # If neither method is supported, we cannot proceed
                 logger.error(f"{Fore.RED}Exchange does not support 'fetchOpenOrders' or 'cancelAllOrders' for {target_symbol}. Cannot cancel orders automatically.{Style.RESET_ALL}")
                 return 0

            final_msg = f"Order cancellation process finished for {target_symbol}. "
            if cancelled_count >= 0:
                 final_msg += f"Attempted to cancel: {cancelled_count} order(s)."
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
        Removes failed (canceled/rejected) pending orders from the state.
        """
        pending_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_PENDING_ENTRY]
        if not pending_positions:
            return # No pending orders to check

        logger.debug(f"Checking status of {len(pending_positions)} pending entry order(s)...")
        # Fetch current price once if needed for SL/TP recalculation on fill
        current_price_for_check: Optional[float] = None
        if any(self.use_atr_sl_tp or not self.use_atr_sl_tp for pos in pending_positions): # Check if SL/TP needs recalculation
             current_price_for_check = self.fetch_market_price()

        positions_to_remove_ids = set()
        positions_to_update_data = {} # Dict[order_id, updated_position_dict]

        for position in pending_positions:
            entry_order_id = position['id']
            if not entry_order_id: # Safety check
                logger.error(f"Found pending position with no ID: {position}. Removing.")
                positions_to_remove_ids.add(entry_order_id) # Use ID if present, else need another key?
                continue

            order_info = self.fetch_order_status(entry_order_id)

            # Case 1: Order Not Found
            if order_info is None:
                # Could be filled and gone, or cancelled externally. Assume it's no longer pending.
                logger.warning(f"{Fore.YELLOW}Pending entry order {entry_order_id} status unknown/not found on exchange. Removing from pending state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)
                continue

            order_status = order_info.get('status')
            filled_amount = order_info.get('filled', 0.0)
            entry_price = order_info.get('average') # Use average fill price

            # Case 2: Order Fully Filled ('closed' status in CCXT)
            if order_status == 'closed' and filled_amount is not None and filled_amount > 0:
                if entry_price is None or entry_price <= 0:
                    logger.error(f"{Fore.RED}Entry order {entry_order_id} is 'closed' but has invalid average fill price ({entry_price}). Cannot activate position correctly. Removing.{Style.RESET_ALL}")
                    positions_to_remove_ids.add(entry_order_id)
                    continue

                # Compare filled amount to originally requested amount (optional check)
                orig_size = position.get('original_size', position['size']) # Use original if stored
                size_diff_ratio = abs(filled_amount - orig_size) / orig_size if orig_size > 1e-9 else 0
                if size_diff_ratio > 0.05: # e.g., > 5% difference
                     logger.warning(f"{Fore.YELLOW}Entry order {entry_order_id} filled amount ({filled_amount}) differs significantly (>5%) from requested ({orig_size}). Using actual filled amount.{Style.RESET_ALL}")

                logger.info(f"{Fore.GREEN}---> Pending entry order {entry_order_id} FILLED! Amount: {filled_amount}, Avg Price: {entry_price}{Style.RESET_ALL}")

                # --- Update Position State ---
                updated_pos = position.copy() # Work on a copy
                updated_pos['size'] = filled_amount # Update size to actual filled amount
                updated_pos['entry_price'] = entry_price
                updated_pos['status'] = STATUS_ACTIVE
                # Use fill timestamp from order if available, else use current time
                updated_pos['entry_time'] = order_info.get('timestamp', time.time() * 1000) / 1000
                updated_pos['last_update_time'] = time.time()

                # --- Recalculate and Store SL/TP based on Actual Fill Price ---
                # This ensures SL/TP is relative to the real entry, not the estimated price
                if current_price_for_check:
                    atr_value = indicators.get('atr') # Get current ATR
                    sl_price, tp_price = self._calculate_sl_tp_prices(
                        entry_price=updated_pos['entry_price'], # Use actual fill price
                        side=updated_pos['side'],
                        current_price=current_price_for_check, # Use current price for sanity checks
                        atr=atr_value
                    )
                    # Store these possibly adjusted SL/TP values in the state
                    updated_pos['stop_loss_price'] = sl_price
                    updated_pos['take_profit_price'] = tp_price
                    logger.info(f"Stored SL={sl_price}, TP={tp_price} for activated position {entry_order_id} based on actual entry {entry_price}.")
                else:
                     # This might happen if SL/TP calc wasn't needed or price fetch failed
                     logger.warning(f"{Fore.YELLOW}Could not fetch current price or SL/TP recalculation skipped after entry fill for {entry_order_id}. Stored SL/TP might be based on estimate.{Style.RESET_ALL}")
                     # Keep SL/TP prices that were originally sent with the order (already in 'position' dict)

                positions_to_update_data[entry_order_id] = updated_pos

            # Case 3: Order Failed (canceled, rejected, expired)
            elif order_status in ['canceled', 'rejected', 'expired']:
                reason = order_info.get('info', {}).get('rejectReason', 'No reason provided') # Try to get Bybit rejection reason
                logger.warning(f"{Fore.YELLOW}Pending entry order {entry_order_id} failed (Status: {order_status}, Reason: {reason}). Removing from state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)

            # Case 4: Order Still Open or Partially Filled (remains pending)
            elif order_status == 'open' or (order_status == 'closed' and filled_amount == 0): # 'closed' with 0 fill can mean cancelled before fill
                logger.debug(f"Pending entry order {entry_order_id} is still '{order_status}'. Filled: {filled_amount}. Waiting...")
                # Optionally handle partial fills here if desired (treat as active? wait for full fill?)
                # For simplicity, we wait for full fill ('closed' status with filled > 0)

        # --- Apply State Updates ---
        if positions_to_remove_ids or positions_to_update_data:
            original_count = len(self.open_positions)
            new_positions_list = []
            for pos in self.open_positions:
                pos_id = pos['id']
                if pos_id in positions_to_remove_ids:
                    continue # Skip removed positions
                elif pos_id in positions_to_update_data:
                    new_positions_list.append(positions_to_update_data[pos_id]) # Replace with updated data
                else:
                    new_positions_list.append(pos) # Keep unchanged positions

            if len(new_positions_list) != original_count or positions_to_update_data:
                 self.open_positions = new_positions_list
                 logger.debug(f"Pending checks complete. Updated positions: {len(positions_to_update_data)}, Removed: {len(positions_to_remove_ids)}. Current total: {len(self.open_positions)}")
                 self._save_state() # Save the changes


    def _manage_active_positions(self, current_price: float, indicators: Dict) -> None:
        """
        Manages currently active positions.

        - Checks if the position was closed externally (SL/TP hit via parameters, manual closure).
        - Implements Time-Based Exit logic.
        - Implements Trailing Stop Loss (TSL) logic using `edit_order` (EXPERIMENTAL).
        - Updates position state (e.g., TSL price) or removes closed positions.

        Args:
            current_price: The current market price.
            indicators: Dictionary containing current indicator values (e.g., ATR).
        """
        active_positions = [pos for pos in self.open_positions if pos.get('status') == STATUS_ACTIVE]
        if not active_positions:
            return # No active positions to manage

        logger.debug(f"Managing {len(active_positions)} active position(s) against current price: {current_price:.{self.market_info['precision']['price']}f}...")

        positions_to_remove_ids = set()
        positions_to_update_state = {} # Store state updates {pos_id: {key: value}}

        for position in active_positions:
            pos_id = position['id'] # This is the ID of the original entry order
            side = position['side']
            entry_price = position['entry_price']
            entry_time = position.get('entry_time', time.time()) # Use current if somehow missing
            # SL/TP prices stored in our state (these might have been adjusted on fill)
            stored_sl_price = position.get('stop_loss_price')
            stored_tp_price = position.get('take_profit_price')
            # Current activated trailing stop price (if TSL is active for this pos)
            trailing_sl_price = position.get('trailing_stop_price')
            is_tsl_active = trailing_sl_price is not None

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None # Price at which the position closed

            # --- 1. Check Status of the Original Entry Order ---
            # With parameter-based SL/TP, the original order itself might close when SL/TP is hit.
            # However, sometimes Bybit creates separate SL/TP orders. Fetching the original
            # order status is still a primary check.
            order_info = self.fetch_order_status(pos_id)

            if order_info is None:
                # Order not found - could be closed, cancelled, or ID issue.
                # Hard to know the exact reason. Assume closed externally.
                logger.warning(f"{Fore.YELLOW}Active position's main order {pos_id} not found on exchange. Assuming closed/cancelled externally. Removing position from state.{Style.RESET_ALL}")
                exit_reason = f"Main Order Vanished ({pos_id})"
                # We don't know the exit price here.

            elif order_info.get('status') == 'closed':
                # Order is closed, likely hit SL/TP or was closed manually via API/UI
                exit_price = order_info.get('average') or order_info.get('price') or current_price # Best guess for exit price
                filled_amount = order_info.get('filled', 0.0) # Should match position size ideally

                # Try to infer reason based on Bybit V5 info fields if available
                # Note: Field names might vary ('stopLoss', 'takeProfit', 'orderStatus' etc. in info)
                order_info_details = order_info.get('info', {})
                sl_triggered_by_exchange = order_info_details.get('stopLoss', '') == str(exit_price) or order_info_details.get('triggerPrice', '') == str(stored_sl_price) # Example checks
                tp_triggered_by_exchange = order_info_details.get('takeProfit', '') == str(exit_price) or order_info_details.get('triggerPrice', '') == str(stored_tp_price) # Example checks
                order_status_info = order_info_details.get('orderStatus', '').lower() # e.g., 'Filled', 'Triggered'

                if 'stop-loss order triggered' in order_status_info or sl_triggered_by_exchange:
                    exit_reason = f"Stop Loss Triggered (Exchange Reported - Order {pos_id})"
                    logger.info(f"{Fore.RED}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}")
                elif 'take-profit order triggered' in order_status_info or tp_triggered_by_exchange:
                     exit_reason = f"Take Profit Triggered (Exchange Reported - Order {pos_id})"
                     logger.info(f"{Fore.GREEN}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}")
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
                    else: exit_reason = f"Order Closed Externally (Reason Unclear - {pos_id})"
                    log_color = Fore.RED if sl_triggered else (Fore.GREEN if tp_triggered else Fore.YELLOW)
                    logger.info(f"{log_color}{exit_reason} at ~{exit_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}")

            elif order_info.get('status') in ['canceled', 'rejected', 'expired']:
                 # This shouldn't happen for an 'active' position, indicates state inconsistency
                 logger.error(f"{Fore.RED}State inconsistency! Active position {pos_id} has order status '{order_info.get('status')}'. Removing position.{Style.RESET_ALL}")
                 exit_reason = f"Order Inconsistency ({order_info.get('status')})"

            # --- If order still seems active on exchange, check bot-managed exits ---
            if not exit_reason and order_info and order_info.get('status') == 'open':

                # --- 2. Time-Based Exit Check ---
                if self.time_based_exit_minutes is not None and self.time_based_exit_minutes > 0:
                    time_elapsed_seconds = time.time() - entry_time
                    time_elapsed_minutes = time_elapsed_seconds / 60
                    if time_elapsed_minutes >= self.time_based_exit_minutes:
                        logger.info(f"{Fore.YELLOW}Time limit ({self.time_based_exit_minutes} min) reached for active position {pos_id}. Initiating market close.{Style.RESET_ALL}")
                        exit_reason = "Time Limit Reached"
                        # --- Place Market Close Order ---
                        market_close_order = self._place_market_close_order(position, current_price)
                        if market_close_order:
                            # Use fill price from close order if available
                            exit_price = market_close_order.get('average', current_price)
                            logger.info(f"Market close order for time exit placed successfully for {pos_id}.")
                            # Mark for removal, PnL will be calculated below
                        else:
                            # CRITICAL: Failed to close position! Needs manual intervention.
                            logger.critical(f"{Fore.RED}CRITICAL FAILURE: Failed to place market close order for time-based exit of position {pos_id}. POSITION REMAINS OPEN! Manual intervention required!{Style.RESET_ALL}")
                            exit_reason = None # Reset reason as close failed

                # --- 3. Trailing Stop Loss (TSL) Logic ---
                # *** WARNING: TSL using edit_order is EXPERIMENTAL and highly dependent ***
                # *** on the specific exchange (Bybit V5) and CCXT version supporting it correctly. ***
                # *** It may NOT work as expected or be supported for modifying SL/TP on existing orders. ***
                # *** A safer alternative is often a cancel/replace mechanism. ***
                # *** VERIFY THIS THOROUGHLY IN TESTNET BEFORE ANY LIVE USE. ***
                if not exit_reason and self.enable_trailing_stop_loss:
                    if not self.trailing_stop_loss_percentage or not (0 < self.trailing_stop_loss_percentage < 1):
                         logger.error("TSL enabled but trailing_stop_loss_percentage is invalid. Disabling TSL for this iteration.")
                         # Optionally disable permanently: self.enable_trailing_stop_loss = False
                    else:
                        new_tsl_price: Optional[float] = None
                        tsl_factor = (1 - self.trailing_stop_loss_percentage) if side == 'buy' else (1 + self.trailing_stop_loss_percentage)

                        # --- TSL Activation ---
                        # Activate TSL only if price moves favorably beyond a certain point (e.g., halfway to original TP or breakeven)
                        if not is_tsl_active:
                            # Define activation condition (e.g., price crosses breakeven + buffer)
                            # Using entry price as simple activation threshold for now
                            breakeven_plus_buffer = entry_price * (1 + 0.001) if side == 'buy' else entry_price * (1 - 0.001)
                            activate = (side == 'buy' and current_price > breakeven_plus_buffer) or \
                                       (side == 'sell' and current_price < breakeven_plus_buffer)

                            if activate:
                                potential_tsl = current_price * tsl_factor
                                # Ensure activation TSL is better than original SL (if exists) and breakeven
                                initial_trail_price = potential_tsl
                                if stored_sl_price is not None:
                                     initial_trail_price = max(initial_trail_price, stored_sl_price) if side == 'buy' else min(initial_trail_price, stored_sl_price)
                                initial_trail_price = max(initial_trail_price, breakeven_plus_buffer) if side == 'buy' else min(initial_trail_price, breakeven_plus_buffer)

                                new_tsl_price = float(self.exchange.price_to_precision(self.symbol, initial_trail_price))
                                logger.info(f"{Fore.MAGENTA}Trailing Stop ACTIVATING for {side.upper()} {pos_id} at {new_tsl_price:.{self.market_info['precision']['price']}f} (Price: {current_price:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}")

                        # --- TSL Update ---
                        elif is_tsl_active: # TSL already active, check if price moved further
                            potential_tsl = current_price * tsl_factor
                            # Only update if the new TSL price is *better* than the current one
                            update_needed = (side == 'buy' and potential_tsl > trailing_sl_price) or \
                                            (side == 'sell' and potential_tsl < trailing_sl_price)

                            if update_needed:
                                new_tsl_price = float(self.exchange.price_to_precision(self.symbol, potential_tsl))
                                logger.info(f"{Fore.MAGENTA}Trailing Stop UPDATING for {side.upper()} {pos_id} from {trailing_sl_price:.{self.market_info['precision']['price']}f} to {new_tsl_price:.{self.market_info['precision']['price']}f} (Price: {current_price:.{self.market_info['precision']['price']}f}){Style.RESET_ALL}")

                        # --- Attempt to Apply TSL Update via edit_order ---
                        if new_tsl_price is not None:
                            logger.warning(f"{Fore.YELLOW}Attempting EXPERIMENTAL TSL update via edit_order for {pos_id} to SL={new_tsl_price}. VERIFY EXCHANGE/CCXT SUPPORT!{Style.RESET_ALL}")
                            try:
                                # Parameters for editing SL/TP on Bybit V5 (check CCXT docs for exact names)
                                # Often requires specifying 'stopLoss', 'takeProfit', 'slTriggerBy', 'tpTriggerBy'
                                edit_params = {
                                    'stopLoss': self.exchange.price_to_precision(self.symbol, new_tsl_price)
                                    # Keep existing TP if set, otherwise it might be removed
                                    # 'takeProfit': self.exchange.price_to_precision(self.symbol, stored_tp_price) if stored_tp_price else None
                                }
                                # Include trigger types if configured
                                if self.sl_trigger_by: edit_params['slTriggerBy'] = self.sl_trigger_by
                                if self.tp_trigger_by and stored_tp_price: edit_params['tpTriggerBy'] = self.tp_trigger_by
                                # Bybit V5 edit might need category
                                # if 'bybit' in self.exchange_id: edit_params['category'] = 'linear' # Or 'inverse'

                                # CCXT's edit_order often requires type, side, amount etc. Pass them from fetched order_info
                                # This is the most brittle part - check CCXT implementation for your exchange!
                                edited_order = self.exchange.edit_order(
                                    id=pos_id,
                                    symbol=self.symbol,
                                    type=order_info.get('type'), # Original type
                                    side=order_info.get('side'), # Original side
                                    amount=order_info.get('amount'), # Original amount
                                    price=order_info.get('price'), # Original limit price (if limit order)
                                    params=edit_params
                                )
                                if edited_order:
                                    confirmed_sl = edited_order.get('info',{}).get('stopLoss', 'N/A')
                                    logger.info(f"{Fore.MAGENTA}---> Successfully modified order {pos_id} via edit_order. New Confirmed SL: {confirmed_sl}. Applying TSL price {new_tsl_price} to state.{Style.RESET_ALL}")
                                    # Update state only if edit seems successful
                                    positions_to_update_state[pos_id] = {
                                        'trailing_stop_price': new_tsl_price,
                                        'stop_loss_price': new_tsl_price, # Update the main SL price as well
                                        'last_update_time': time.time()
                                    }
                                else:
                                    # edit_order succeeded syntactically but returned None/empty
                                    logger.error(f"{Fore.RED}TSL update via edit_order for {pos_id} potentially failed (API returned None/empty). TSL state NOT updated.{Style.RESET_ALL}")

                            except ccxt.NotSupported as e:
                                logger.error(f"{Fore.RED}TSL update failed: edit_order for SL/TP modification is NOT SUPPORTED by CCXT for {self.exchange_id} or for this order type ({order_info.get('type')}). Error: {e}. Disable TSL or implement cancel/replace.{Style.RESET_ALL}")
                                # Consider disabling TSL permanently if not supported
                                # self.enable_trailing_stop_loss = False
                            except ccxt.OrderNotFound:
                                 logger.warning(f"{Fore.YELLOW}TSL update failed: Order {pos_id} not found during edit attempt (likely closed just now).{Style.RESET_ALL}")
                            except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
                                 logger.error(f"{Fore.RED}TSL update via edit_order failed for {pos_id} ({type(e).__name__}): {e}. Params sent: {edit_params}. TSL state NOT updated.{Style.RESET_ALL}")
                            except Exception as e:
                                logger.error(f"{Fore.RED}Unexpected error modifying order {pos_id} for TSL: {e}{Style.RESET_ALL}", exc_info=True)


            # --- 4. Process Exit and Log PnL ---
            if exit_reason:
                positions_to_remove_ids.add(pos_id)
                # Calculate and log PnL if exit price is known
                if exit_price is not None and entry_price is not None and entry_price > 0 and exit_price > 0:
                    position_size = position['size']
                    # PnL calculation in quote currency
                    pnl_quote = (exit_price - entry_price) * position_size if side == 'buy' else (entry_price - exit_price) * position_size
                    # PnL calculation as percentage
                    pnl_pct = ((exit_price / entry_price) - 1) * 100 if side == 'buy' else ((entry_price / exit_price) - 1) * 100

                    pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
                    price_prec = self.market_info['precision']['price']
                    logger.info(
                        f"{pnl_color}---> Position {pos_id} Closed. Reason: {exit_reason}. "
                        f"Entry: {entry_price:.{price_prec}f}, Exit: {exit_price:.{price_prec}f}, Size: {position_size}. "
                        f"Est. PnL: {pnl_quote:.{self.market_info['precision']['price']}f} {self.market_info['quote']} ({pnl_pct:.3f}%){Style.RESET_ALL}"
                    )
                    self.daily_pnl += pnl_quote # Update simple daily PnL tracker
                else:
                    # Log closure without PnL if prices are invalid/unknown
                    logger.warning(f"{Fore.YELLOW}---> Position {pos_id} Closed. Reason: {exit_reason}. Exit or entry price unknown/invalid, cannot calculate PnL accurately. Entry={entry_price}, Exit={exit_price}{Style.RESET_ALL}")


        # --- Apply State Updates and Removals ---
        if positions_to_remove_ids or positions_to_update_state:
            original_count = len(self.open_positions)
            new_positions_list = []
            removed_count = 0
            updated_count = 0

            for pos in self.open_positions:
                pos_id = pos['id']
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
            current_price: The current market price (used for simulation).

        Returns:
            The CCXT order dictionary of the close order if successful (or simulated),
            otherwise None.
        """
        pos_id = position['id']
        side = position['side']
        size = position['size']
        base_currency = self.market_info['base']
        price_prec = self.market_info['precision']['price']
        amount_prec = self.market_info['precision']['amount']

        if size <= 0:
            logger.error(f"Cannot close position {pos_id}: Size is zero or negative ({size}).")
            return None

        # Determine the side of the closing order (opposite of the position side)
        close_side = 'sell' if side == 'buy' else 'buy'
        log_color = Fore.YELLOW # Use yellow for closing actions

        logger.warning(f"{log_color}Initiating MARKET CLOSE for position {pos_id} (Entry Side: {side}, Close Side: {close_side}, Size: {size:.{amount_prec}f} {base_currency})...{Style.RESET_ALL}")

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
                "symbol": self.symbol, "type": "market", "side": close_side,
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
            logger.info(f"{log_color}[SIMULATION] Market Close Order Placed: ID {sim_close_id}, Size {size:.{amount_prec}f}, AvgPrice {sim_avg_close_price:.{price_prec}f}{Style.RESET_ALL}")
            return simulated_close_order

        # --- Live Trading Mode ---
        else:
            order: Optional[Dict[str, Any]] = None
            try:
                # Use 'reduceOnly' parameter to ensure the order only closes/reduces the position
                params = {}
                if self.exchange.has.get('reduceOnly'):
                    params['reduceOnly'] = True
                    logger.debug("Using 'reduceOnly=True' parameter for market close order.")
                else:
                     logger.warning(f"{Fore.YELLOW}Exchange {self.exchange_id} might not support 'reduceOnly' parameter via CCXT. Close order might accidentally open a new position if size is incorrect or position already closed.{Style.RESET_ALL}")

                 # Bybit V5 might need category
                 # if 'bybit' in self.exchange_id: params['category'] = 'linear'

                order = self.exchange.create_market_order(self.symbol, close_side, size, params=params)

                if order:
                    oid = order.get('id', 'N/A')
                    oavg = order.get('average') # Actual average fill price
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, "
                        f"Status: {ostatus}, AvgFill ~{oavg:.{price_prec}f} (if available){Style.RESET_ALL}"
                    )
                    return order
                else:
                    # API call returned None/empty
                    logger.error(f"{Fore.RED}LIVE Market Close order placement failed: API returned None or empty response for position {pos_id}.{Style.RESET_ALL}")
                    return None

            except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
                 # Specific errors often indicating state mismatch or parameter issues
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} ({type(e).__name__}): {e}. Position might already be closed or size incorrect.{Style.RESET_ALL}")
                 return None
            except ccxt.ExchangeError as e:
                 # General exchange errors
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (ExchangeError): {e}{Style.RESET_ALL}")
                 return None
            except Exception as e:
                 # Unexpected Python errors
                 logger.error(f"{Fore.RED}LIVE Market Close Order Failed for {pos_id} (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True)
                 return None


    # --- Main Bot Execution Loop ---

    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches all required market data (price, order book, history) for one iteration."""
        logger.debug("Fetching market data bundle...")
        start_time = time.time()
        try:
            # Use concurrent fetching if performance becomes an issue (requires async/threading)
            current_price = self.fetch_market_price()
            order_book_data = self.fetch_order_book() # Contains 'imbalance' if successful
            historical_data = self.fetch_historical_data() # Fetches required lookback

            # --- Check if all essential data was fetched ---
            if current_price is None:
                logger.warning(f"{Fore.YELLOW}Failed to fetch current market price. Skipping iteration.{Style.RESET_ALL}")
                return None
            if order_book_data is None:
                 # Non-critical, proceed without order book data
                 logger.warning(f"{Fore.YELLOW}Failed to fetch order book data. Proceeding without imbalance signal.{Style.RESET_ALL}")
                 order_book_imbalance = None
            else:
                 order_book_imbalance = order_book_data.get('imbalance')

            if historical_data is None or historical_data.empty:
                logger.warning(f"{Fore.YELLOW}Failed to fetch sufficient historical data. Skipping iteration.{Style.RESET_ALL}")
                return None

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
        # Ensure required columns exist
        if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
             logger.error("Historical data DataFrame is missing required columns (OHLCV). Cannot calculate indicators.")
             return {} # Return empty dict

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
        # Log calculated indicators (format nicely)
        log_inds = {k: f"{v:.4f}" if isinstance(v, float) else ("N/A" if v is None else v) for k, v in indicators.items()}
        price_prec = self.market_info['precision']['price']
        logger.info(f"Indicators (calc took {calc_duration:.2f}s): "
                    f"EMA={log_inds['ema']}, RSI={log_inds['rsi']}, ATR={log_inds['atr']}, "
                    f"MACD_L={log_inds['macd_line']}, MACD_S={log_inds['macd_signal']}, "
                    f"StochK={log_inds['stoch_k']}, StochD={log_inds['stoch_d']}")

        return indicators

    def _process_signals_and_entry(self, market_data: Dict, indicators: Dict) -> None:
        """
        Analyzes market data and indicators to compute a trade signal.
        If a signal meets the threshold and conditions allow, attempts to place an entry order.
        """
        current_price = market_data['price']
        orderbook_imbalance = market_data['order_book_imbalance']
        atr_value = indicators.get('atr') # Needed for potential ATR-based SL/TP

        # --- 1. Check Entry Conditions ---
        # Can we open a new position?
        can_open_new = len(self.open_positions) < self.max_open_positions
        if not can_open_new:
             logger.info(f"{Fore.YELLOW}Max open positions ({self.max_open_positions}) reached. Skipping new entry check.{Style.RESET_ALL}")
             return

        # Calculate potential order size - do this early to see if trading is possible
        order_size_quote = self.calculate_order_size(current_price)
        can_trade = order_size_quote > 0
        if not can_trade:
             logger.warning(f"{Fore.YELLOW}Cannot evaluate entry signal: Calculated order size is zero (insufficient balance or below limits).{Style.RESET_ALL}")
             return

        # --- 2. Compute Signal Score ---
        signal_score, reasons = self.compute_trade_signal_score(
            current_price, indicators, orderbook_imbalance
        )
        logger.info(f"Trade Signal Score: {signal_score}")
        # Log reasons only if score is non-zero or debug level is enabled
        if signal_score != 0 or logger.isEnabledFor(logging.DEBUG):
            for reason in reasons:
                 logger.info(f" -> {reason}") # Use info level for reasons when score is non-zero

        # --- 3. Determine Entry Action ---
        entry_side: Optional[str] = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'buy'
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'sell'

        # --- 4. Place Entry Order if Signal Strong Enough ---
        if entry_side:
            log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
            logger.info(f"{log_color}Potential {entry_side.upper()} entry signal detected (Score: {signal_score} meets threshold {ENTRY_SIGNAL_THRESHOLD_ABS}){Style.RESET_ALL}")

            # --- Calculate SL/TP Prices BEFORE placing order ---
            # Use current_price as the reference for initial SL/TP calculation before entry
            sl_price, tp_price = self._calculate_sl_tp_prices(
                entry_price=current_price, # Use current price as estimate for calculation basis
                side=entry_side,
                current_price=current_price, # Pass current price for sanity checks
                atr=atr_value
            )
            # Proceed only if SL/TP calculation is valid (or not required)
            # If SL/TP is absolutely required, add check here: if sl_price is None: return
            if self.use_atr_sl_tp and atr_value is None:
                 logger.error(f"{Fore.RED}ATR SL/TP required but ATR is None. Cannot place entry order.{Style.RESET_ALL}")
                 return
            if not self.use_atr_sl_tp and (self.base_stop_loss_pct is None or self.base_take_profit_pct is None):
                 logger.error(f"{Fore.RED}Fixed % SL/TP required but percentages not set. Cannot place entry order.{Style.RESET_ALL}")
                 return
            # If SL/TP calculation resulted in None due to other issues (e.g., price checks), log and potentially abort
            if sl_price is None and tp_price is None and (self.use_atr_sl_tp or self.base_stop_loss_pct is not None):
                 logger.warning(f"{Fore.YELLOW}SL/TP calculation failed (check logs). Proceeding without SL/TP parameters if possible, otherwise aborting entry.{Style.RESET_ALL}")
                 # Decide: Abort or place without SL/TP? Aborting is safer if SL/TP is critical.
                 # For now, let's allow placing without SL/TP if calculation failed but wasn't strictly impossible
                 # return # Uncomment to abort if SL/TP calc fails

            # --- Place the Actual Entry Order ---
            entry_order = self.place_entry_order(
                side=entry_side,
                order_size_quote=order_size_quote, # Pass calculated size
                confidence_level=signal_score,
                order_type=self.entry_order_type,
                current_price=current_price, # Pass for market/limit reference
                stop_loss_price=sl_price,    # Pass calculated SL (or None)
                take_profit_price=tp_price   # Pass calculated TP (or None)
            )

            # --- Update Bot State if Order Placement Succeeded ---
            if entry_order:
                order_status = entry_order.get('status')
                order_id = entry_order.get('id')
                if not order_id:
                     logger.error(f"{Fore.RED}Entry order placed but received no order ID! Cannot track position. Order: {entry_order}{Style.RESET_ALL}")
                     return # Cannot track without ID

                # Determine initial position status based on order status
                initial_pos_status = STATUS_UNKNOWN
                if order_status == 'open':
                    initial_pos_status = STATUS_PENDING_ENTRY
                elif order_status == 'closed': # Assumed filled for market orders or immediate limit fills
                    initial_pos_status = STATUS_ACTIVE
                elif order_status in ['canceled', 'rejected', 'expired']:
                    logger.warning(f"Entry order {order_id} failed immediately ({order_status}). Not adding position.")
                    return # Don't add failed orders
                else:
                    logger.warning(f"Entry order {order_id} has unexpected initial status: {order_status}. Treating as PENDING.")
                    initial_pos_status = STATUS_PENDING_ENTRY

                # Use average fill price if available (especially for market/filled), else estimate
                # Use order price for limits, fall back to current price if all else fails
                actual_entry_price = entry_order.get('average') or entry_order.get('price') or current_price
                filled_amount = entry_order.get('filled', 0.0)
                requested_amount = entry_order.get('amount', 0.0)

                # Create the new position dictionary for state tracking
                new_position = {
                    "id": order_id,
                    "symbol": self.symbol, # Store symbol with position
                    "side": entry_side,
                    # Store both requested and initially filled size
                    "size": filled_amount if initial_pos_status == STATUS_ACTIVE else requested_amount,
                    "original_size": requested_amount,
                    "entry_price": actual_entry_price,
                    # Use timestamp from order if available, else current time
                    "entry_time": entry_order.get('timestamp', time.time() * 1000) / 1000 if initial_pos_status == STATUS_ACTIVE else None,
                    "status": initial_pos_status,
                    "entry_order_type": self.entry_order_type,
                    # Store the SL/TP prices that were *sent* with the order request
                    # These might be slightly different from recalculated ones if entry price deviated
                    "stop_loss_price": entry_order.get('stopLossPrice') or sl_price, # Prefer response if available
                    "take_profit_price": entry_order.get('takeProfitPrice') or tp_price,
                    "confidence": signal_score,
                    "trailing_stop_price": None, # TSL not active initially
                    "last_update_time": time.time()
                }
                self.open_positions.append(new_position)
                logger.info(f"{log_color}---> Position {order_id} added to state with status: {initial_pos_status}. Entry Price Est: ~{actual_entry_price:.{self.market_info['precision']['price']}f}{Style.RESET_ALL}")

                # If entry was market and filled immediately, update state more accurately
                if initial_pos_status == STATUS_ACTIVE:
                    if filled_amount > 0:
                        new_position['size'] = filled_amount # Ensure size is filled size
                    if entry_order.get('average'):
                        new_position['entry_price'] = entry_order['average'] # Update with definite fill price
                    if entry_order.get('timestamp'):
                         new_position['entry_time'] = entry_order['timestamp'] / 1000 # Update with fill time

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
        # Consider cancelling any potentially orphaned orders from previous runs,
        # but be cautious not to interfere with other bots or manual trades.
        # logger.warning("Performing initial check for existing open orders...")
        # self.cancel_all_symbol_orders() # Use with caution!

        while True:
            self.iteration += 1
            start_time_iter = time.time()
            loop_prefix = f"{Fore.BLUE}===== Iteration {self.iteration} ===={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Timestamp: {pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}")

            try:
                # --- 1. Fetch Market Data Bundle ---
                market_data = self._fetch_market_data()
                if market_data is None:
                    logger.warning("Essential market data missing, pausing before next iteration.")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue # Skip rest of loop

                current_price = market_data['price']
                logger.info(f"{loop_prefix} Current Price: {current_price:.{self.market_info['precision']['price']}f} "
                            f"OB Imbalance: {f'{market_data['order_book_imbalance']:.3f}' if market_data.get('order_book_imbalance') is not None else 'N/A'}")

                # --- 2. Calculate Indicators ---
                indicators = self._calculate_indicators(market_data['historical_data'])
                if not indicators: # Check if indicator calculation failed critically
                     logger.error("Indicator calculation failed. Skipping trading logic for this iteration.")
                     time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue

                # --- 3. Check Pending Entry Orders ---
                # Checks if limit orders were filled and updates state accordingly
                self._check_pending_entries(indicators) # Pass indicators for potential SL/TP recalc

                # --- 4. Process Signals & Potential New Entry ---
                # Evaluates indicators/market data for new trade opportunities
                self._process_signals_and_entry(market_data, indicators)

                # --- 5. Manage Active Positions ---
                # Handles TSL updates, time-based exits, and checks for external closures (SL/TP hits)
                # Re-fetch current price *just before* managing positions for most up-to-date checks?
                # current_price_for_manage = self.fetch_market_price() or current_price # Use latest or fallback
                self._manage_active_positions(current_price, indicators) # Pass current price and indicators (e.g., for TSL checks)

                # --- 6. Loop Pacing ---
                end_time_iter = time.time()
                execution_time = end_time_iter - start_time_iter
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - execution_time)
                logger.debug(f"{loop_prefix} Iteration took {execution_time:.2f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            except KeyboardInterrupt:
                # Allow Ctrl+C to break the loop for graceful shutdown
                logger.warning("Keyboard interrupt received. Exiting main loop...")
                break # Exit the while loop
            except SystemExit as e:
                 logger.warning(f"SystemExit called with code {e.code}. Exiting main loop...")
                 raise e # Re-raise to be caught by main block
            except Exception as e:
                # Catch-all for unexpected errors within the main loop
                logger.critical(f"{Fore.RED}{loop_prefix} CRITICAL UNHANDLED ERROR in main loop: {e}{Style.RESET_ALL}", exc_info=True)
                # Implement a cool-down period before potentially retrying
                logger.warning(f"{Fore.YELLOW}Pausing for 60 seconds due to critical loop error before next attempt...{Style.RESET_ALL}")
                time.sleep(60)

        logger.info("Main trading loop terminated.")


    def shutdown(self):
        """Performs graceful shutdown procedures for the bot."""
        logger.warning(f"{Fore.YELLOW}--- Initiating Graceful Shutdown Sequence ---{Style.RESET_ALL}")

        # --- 1. Cancel Open Orders ---
        # Focus on orders potentially created by this bot instance (pending entries)
        # Active positions might have associated SL/TP orders managed by the exchange.
        pending_orders_cancelled = 0
        logger.info("Attempting to cancel known PENDING entry orders...")
        # Iterate over a copy as cancellation might affect the list indirectly via state saves
        for pos in list(self.open_positions):
            # Only cancel orders that are still marked as pending entry
            if pos.get('status') == STATUS_PENDING_ENTRY:
                logger.info(f"Cancelling pending entry order {pos.get('id')}...")
                if self.cancel_order_by_id(pos.get('id')):
                    pending_orders_cancelled += 1
                    # Update state immediately to reflect cancellation during shutdown
                    pos['status'] = 'canceled'
                    pos['last_update_time'] = time.time()
                else:
                    logger.error(f"Failed to cancel pending order {pos.get('id')} during shutdown.")
        logger.info(f"Cancelled {pending_orders_cancelled} pending entry orders.")

        # Optional: Broader cancellation (use with caution)
        # cancel_all_on_exit = False # Make this configurable?
        # if cancel_all_on_exit:
        #    logger.warning(f"{Fore.YELLOW}Attempting to cancel ALL open orders for {self.symbol}...{Style.RESET_ALL}")
        #    self.cancel_all_symbol_orders()

        # --- 2. Optionally Close Active Positions ---
        # Make this explicitly configurable and default to False for safety
        CLOSE_POSITIONS_ON_EXIT = self.config.get("trading",{}).get("close_positions_on_exit", False)
        active_positions = [p for p in self.open_positions if p.get('status') == STATUS_ACTIVE]

        if CLOSE_POSITIONS_ON_EXIT and active_positions:
             logger.warning(f"{Fore.YELLOW}Configuration requests closing {len(active_positions)} active position(s) via market order...{Style.RESET_ALL}")
             current_price_for_close = self.fetch_market_price() # Get latest price for closing
             if current_price_for_close:
                 for position in list(active_positions): # Iterate copy
                    close_order = self._place_market_close_order(position, current_price_for_close)
                    if close_order:
                        logger.info(f"{Fore.YELLOW}Market close order successfully placed for position {position['id']}.{Style.RESET_ALL}")
                        # Remove the position from state immediately after successful close order
                        self.open_positions = [p for p in self.open_positions if p.get('id') != position.get('id')]
                    else:
                        # CRITICAL: Failed to close during shutdown!
                        logger.error(f"{Fore.RED}CRITICAL: Failed to place market close order for {position['id']} during shutdown. Manual check required!{Style.RESET_ALL}")
             else:
                 logger.error(f"{Fore.RED}CRITICAL: Cannot fetch current price for market close during shutdown. {len(active_positions)} position(s) remain open! Manual check required!{Style.RESET_ALL}")
        elif active_positions:
            # Log remaining active positions if not closing them
            logger.warning(f"{Fore.YELLOW}{len(active_positions)} position(s) remain active (Close on exit disabled or failed). Manual management required:{Style.RESET_ALL}")
            for pos in active_positions:
                logger.warning(f" -> ID: {pos.get('id')}, Side: {pos.get('side')}, Size: {pos.get('size')}, Entry: {pos.get('entry_price')}")

        # --- 3. Final State Save ---
        # Save the potentially modified state (cancelled orders, closed positions)
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
                logger.info(f"Creating state directory: {state_dir}")
                os.makedirs(state_dir)
            except OSError as e:
                 logger.critical(f"{Fore.RED}Fatal: Could not create state directory '{state_dir}': {e}. Aborting.{Style.RESET_ALL}")
                 sys.exit(1)

        # --- Initialize and Run Bot ---
        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)
        bot_instance.run() # Start the main trading loop

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal received (Ctrl+C). Initiating graceful shutdown...{Style.RESET_ALL}")
        exit_code = 130 # Standard exit code for Ctrl+C

    except SystemExit as e:
        # Log SystemExit if it wasn't a clean exit (code 0) or standard interrupt code
        if e.code not in [0, 130]:
             logger.error(f"Bot exited via SystemExit with unexpected code: {e.code}.")
        else:
             logger.info(f"Bot exited via SystemExit with code: {e.code}.")
        exit_code = e.code if e.code is not None else 1 # Default to 1 if code is None

    except Exception as e:
        # Catch any critical unhandled exceptions during setup or run
        logger.critical(f"{Fore.RED}An unhandled critical error occurred outside the main loop: {e}{Style.RESET_ALL}", exc_info=True)
        exit_code = 1 # Indicate an error exit

    finally:
        # --- Graceful Shutdown ---
        if bot_instance:
            logger.info("Executing bot shutdown procedures...")
            bot_instance.shutdown()
        else:
            # If bot instance creation failed, just ensure logging is shut down
            logger.info("Bot instance not fully initialized or shutdown already initiated. Shutting down logging.")
            logging.shutdown()

        print(f"{Fore.MAGENTA}Pyrmethus bids thee farewell. Exit Code: {exit_code}{Style.RESET_ALL}")
        sys.exit(exit_code)
