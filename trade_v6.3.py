# -*- coding: utf-8 -*-
"""
Scalping Bot v6.2 - Pyrmethus Enhanced Edition (Bybit V5 Optimized)

Implements an enhanced cryptocurrency scalping bot using ccxt, specifically
tailored for Bybit V5 API's parameter-based Stop Loss (SL) and Take Profit (TP)
handling. This version incorporates fixes for precision errors, improved state
management, enhanced logging, robust error handling, refined position sizing,
and clearer code structure based on prior versions and analysis.

Key Enhancements V6.2 (Compared to V6.1 & Previous):
- Integrated enhanced `calculate_order_size` accepting indicators/signal_score.
- Bound config values to dedicated instance attributes in `__init__`.
- Added Bybit V5 'category' parameter more consistently in API calls.
- Refined state management (`_load_state`, `_save_state`) with more validation,
  robust atomic writes using `tempfile`, and better backup handling.
- Strengthened `_manage_active_positions`: Improved external closure detection logic,
  refined TSL update/edit logic with clearer warnings and parameter handling
  (resending TP when editing SL). Made state updates within loops safer.
- Enhanced `validate_config` for more thorough checks.
- Improved logging with more context (function names) and consistency.
- Refined indicator calculations with better minimum length checks.
- Refactored `run` method using internal helper methods for clarity.
- Improved type hinting and docstrings throughout.
- Refined shutdown procedure.
- Enhanced API retry decorator with more specific error handling.
- Added more comprehensive balance fetching logic for Bybit V5.
- Improved PnL calculation and logging.
"""

import logging
import os
import sys
import time
import json
import shutil # For state file backup and management
import math   # For decimal place calculation
import tempfile # For atomic state saving
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
LOG_FILE_NAME: str = "scalping_bot_v6.log" # Updated log file name
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
STATUS_CANCELED: str = 'canceled'       # Order explicitly cancelled by bot or user
STATUS_REJECTED: str = 'rejected'       # Order rejected by exchange
STATUS_EXPIRED: str = 'expired'         # Order expired (e.g., timeInForce)
STATUS_CLOSED_EXT: str = 'closed_externally' # Position closed by SL/TP/Manual action detected
STATUS_CLOSED_ON_EXIT: str = 'closed_on_exit' # Position closed during bot shutdown
STATUS_UNKNOWN: str = 'unknown'         # Order status cannot be determined

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Centralized Logger Setup ---
logger = logging.getLogger("ScalpingBotV6") # Updated logger name
logger.setLevel(logging.DEBUG) # Set logger to lowest level (DEBUG captures everything)
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(funcName)s] - %(message)s" # Added funcName
)

# Console Handler (INFO level by default, configurable)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Default console level, overridden by config
logger.addHandler(console_handler)

# File Handler (DEBUG level for detailed logs)
file_handler: Optional[logging.FileHandler] = None
try:
    log_dir = os.path.dirname(LOG_FILE_NAME)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Use print here as logger might not be fully configured if directory creation fails
        print(f"{Fore.CYAN}Created log directory: {log_dir}{Style.RESET_ALL}")

    file_handler = logging.FileHandler(LOG_FILE_NAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # Log everything to file
    logger.addHandler(file_handler)
except IOError as e:
    print(f"{Fore.RED}Fatal: Failed to configure log file {LOG_FILE_NAME}: {e}{Style.RESET_ALL}", file=sys.stderr)
    # Optionally exit if file logging is critical
    # sys.exit(1)
except Exception as e:
    print(f"{Fore.RED}Fatal: Unexpected error setting up file logging: {e}{Style.RESET_ALL}", file=sys.stderr)
    # sys.exit(1)

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
    non-retryable exchange errors. Enhanced logging.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds between retries.

    Returns:
        A decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
            retries = 0
            delay = initial_delay
            # Get instance and function name for logging
            instance_name = args[0].__class__.__name__ if args and hasattr(args[0], '__class__') else ''
            func_name = f"{instance_name}.{func.__name__}" if instance_name else func.__name__

            while retries <= max_retries:
                try:
                    # Use DEBUG for attempt logging to avoid cluttering INFO
                    logger.debug(f"API Call: {func_name} - Attempt {retries + 1}/{max_retries + 1}")
                    result = func(*args, **kwargs)
                    # Log success at DEBUG level as well, maybe with partial result?
                    # logger.debug(f"API Call Succeeded: {func_name} -> {str(result)[:50]}...")
                    return result

                # --- Specific Retryable CCXT Errors ---
                except ccxt.RateLimitExceeded as e:
                    log_msg = f"Rate limit encountered ({func_name}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1}) Error: {e}"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    log_msg = f"Network/Exchange issue ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Authentication / Permission Errors (Usually Fatal/Non-Retryable) ---
                except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
                    log_msg = f"Authentication/Permission ritual failed ({func_name}: {type(e).__name__} - {e}). Check API keys, permissions, IP whitelist. Aborting call (no retry)."
                    # Use CRITICAL as this often stops the bot
                    logger.critical(f"{Fore.RED}{log_msg}{Style.RESET_ALL}")
                    return None # Indicate non-retryable failure

                # --- Exchange Errors - Distinguish Retryable vs Non-Retryable ---
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    non_retryable_phrases = [
                        "order not found", "order does not exist", "unknown order",
                        "order already cancel", "order was cancel", "cancel order failed", # Typo "canceled"?
                        "order is finished", "order has been filled", "already closed", "order status error",
                        "cannot be modified", "position status", # Bybit: position already closed etc.
                        "insufficient balance", "insufficient margin", "margin is insufficient",
                        "available balance insufficient", "insufficient funds", "risk limit",
                        "position side is not match", # Bybit hedge mode issue?
                        "invalid order", "parameter error", "size too small", "price too low",
                        "price too high", "invalid price precision", "invalid amount precision",
                        "order cost not meet", "qty must be greater than", "must be greater than",
                        "api key is invalid", "invalid api key", # Auth (redundant?)
                        "position mode not modified", "leverage not modified", # Config/Setup issues
                        "reduceonly", "reduce-only", # Position logic errors
                        "bad symbol", "invalid symbol", # Config error
                        "account category not match", # Bybit V5 category issue
                        "order amount lower than minimum", "order price lower than minimum", # Min size/price issues
                        "invalid stop loss price", "invalid take profit price", # SL/TP issues
                    ]
                    if any(phrase in err_str for phrase in non_retryable_phrases):
                        # Use WARNING for common "not found" / "already closed" / "insufficient"
                        # Use ERROR for potentially more serious config/logic errors
                        is_common_final_state = ("not found" in err_str or "already" in err_str or "is finished" in err_str or "insufficient" in err_str)
                        log_level = logging.WARNING if is_common_final_state else logging.ERROR
                        log_color = Fore.YELLOW if log_level == logging.WARNING else Fore.RED
                        log_msg = f"Non-retryable ExchangeError ({func_name}: {type(e).__name__} - {e}). Aborting call."
                        logger.log(log_level, f"{log_color}{log_msg}{Style.RESET_ALL}")
                        # Special case: Return a specific marker for InsufficientFunds? Could be useful.
                        # if "insufficient" in err_str: return "INSUFFICIENT_FUNDS" # Or handle in caller
                        return None # Indicate non-retryable failure

                    # --- Potentially Retryable Exchange Errors ---
                    # Temporary glitches, server issues, nonce problems etc.
                    retryable_phrases = ["nonce", "timeout", "service unavailable", "internal error", "busy", "too many visits", "connection reset by peer"]
                    if any(phrase in err_str for phrase in retryable_phrases):
                        log_msg = f"Potentially transient ExchangeError ({func_name}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                    # --- Unknown/Default Exchange Errors ---
                    else:
                        # Log as WARNING but still retry by default
                        log_msg = f"Unclassified ExchangeError ({func_name}: {type(e).__name__} - {e}). Assuming transient, pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Catch-all for other unexpected exceptions ---
                except Exception as e:
                    # Log these as ERROR with traceback
                    log_msg = f"Unexpected Python exception during {func_name}: {type(e).__name__} - {e}. Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.error(f"{Fore.RED}{log_msg}{Style.RESET_ALL}", exc_info=True)

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60) # Exponential backoff, capped at 60s
                    retries += 1
                else:
                    # Log final failure as ERROR
                    logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name}. Spell falters. Last error: {e}{Style.RESET_ALL}")
                    return None # Indicate failure after exhausting retries

            # This line should technically not be reached if the loop logic is correct
            logger.error(f"Exited retry loop unexpectedly for {func_name}")
            return None
        return wrapper
    return decorator


# --- Core Scalping Bot Class ---
class ScalpingBot:
    """
    Pyrmethus Enhanced Scalping Bot v6.2. Optimized for Bybit V5 API.

    Implements scalping using technical indicators and order book analysis.
    Features robust state management, ATR/percentage SL/TP, parameter-based
    SL/TP (Bybit V5), and **EXPERIMENTAL** Trailing Stop Loss (TSL) via `edit_order`.

    **Disclaimer:** Trading cryptocurrencies involves significant risk. Use at your own risk.
    Thoroughly test in simulation/testnet modes. TSL via `edit_order` is experimental and
    requires verification on your exchange/CCXT version.
    """

    def __init__(self, config_file: str = CONFIG_FILE_NAME, state_file: str = STATE_FILE_NAME) -> None:
        """Initializes the bot: Loads config, validates, connects, loads state & market info."""
        logger.info(f"{Fore.MAGENTA}--- Pyrmethus Scalping Bot v6.2 Awakening ---{Style.RESET_ALL}")
        self.config: Dict[str, Any] = {}
        self.state_file: str = state_file
        self.load_config(config_file)
        self.validate_config() # Validate before using config values

        # --- Bind Core Attributes from Config/Environment ---
        # Credentials (prefer environment variables)
        self.api_key: Optional[str] = os.getenv("BYBIT_API_KEY") or self.config.get("exchange", {}).get("api_key")
        self.api_secret: Optional[str] = os.getenv("BYBIT_API_SECRET") or self.config.get("exchange", {}).get("api_secret")

        # Exchange Settings
        self.exchange_id: str = self.config["exchange"]["exchange_id"]
        self.testnet_mode: bool = self.config["exchange"]["testnet_mode"]

        # Trading Parameters
        self.symbol: str = self.config["trading"]["symbol"]
        self.timeframe: str = self.config["trading"]["timeframe"]
        self.simulation_mode: bool = self.config["trading"]["simulation_mode"]
        self.entry_order_type: str = self.config["trading"]["entry_order_type"].lower()
        self.limit_order_entry_offset_pct_buy: float = self.config["trading"]["limit_order_offset_buy"]
        self.limit_order_entry_offset_pct_sell: float = self.config["trading"]["limit_order_offset_sell"]
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
        self.order_size_percentage: float = self.config["risk_management"]["order_size_percentage"]
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = self.config["risk_management"].get("atr_sl_multiplier", ATR_MULTIPLIER_SL)
        self.atr_tp_multiplier: float = self.config["risk_management"].get("atr_tp_multiplier", ATR_MULTIPLIER_TP)
        self.base_stop_loss_pct: Optional[float] = self.config["risk_management"].get("stop_loss_percentage")
        self.base_take_profit_pct: Optional[float] = self.config["risk_management"].get("take_profit_percentage")
        self.sl_trigger_by: Optional[str] = self.config["risk_management"].get("sl_trigger_by")
        self.tp_trigger_by: Optional[str] = self.config["risk_management"].get("tp_trigger_by")
        self.enable_trailing_stop_loss: bool = self.config["risk_management"]["enable_trailing_stop_loss"]
        self.trailing_stop_loss_percentage: Optional[float] = self.config["risk_management"].get("trailing_stop_loss_percentage")
        self.time_based_exit_minutes: Optional[int] = self.config["risk_management"].get("time_based_exit_minutes")
        self.strong_signal_adjustment_factor: float = self.config["risk_management"]["strong_signal_adjustment_factor"]
        self.weak_signal_adjustment_factor: float = self.config["risk_management"]["weak_signal_adjustment_factor"]

        # --- Internal Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0 # Simple daily PnL tracker, reset manually or via external script
        # Stores active and pending positions as a list of dictionaries
        # See STATUS_* constants for 'status' field values.
        # Key fields: 'id', 'symbol', 'side', 'size', 'original_size', 'entry_price',
        #             'entry_time', 'status', 'stop_loss_price', 'take_profit_price',
        #             'trailing_stop_price', 'last_update_time', 'confidence', etc.
        self.open_positions: List[Dict[str, Any]] = []
        self.market_info: Optional[Dict[str, Any]] = None # Cache for market details
        self.price_decimals: int = DEFAULT_PRICE_DECIMALS
        self.amount_decimals: int = DEFAULT_AMOUNT_DECIMALS

        # --- Setup & Initialization Steps ---
        self._configure_logging_level()
        self.exchange: ccxt.Exchange = self._initialize_exchange()
        self._load_market_info() # Needs exchange object, calculates decimals internally
        self._load_state()       # Load persistent state after market info

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

        logger.info(f"{Fore.CYAN}Scalping Bot V6.2 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}")

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            console_handler.setLevel(log_level)
            logger.info(f"Console logging level enchanted to: {log_level_str}")
            # Log effective levels at DEBUG for troubleshooting
            logger.debug(f"Effective Bot Logger Level: {logging.getLevelName(logger.level)}")
            logger.debug(f"Effective Console Handler Level: {logging.getLevelName(console_handler.level)}")
            if file_handler:
                logger.debug(f"Effective File Handler Level: {logging.getLevelName(file_handler.level)}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error configuring logging level: {e}. Using default INFO.{Style.RESET_ALL}", exc_info=True)
            console_handler.setLevel(logging.INFO)

    def _validate_state_entry(self, entry: Any, index: int) -> bool:
        """Validates a single entry from the loaded state file."""
        func_name = "_validate_state_entry"
        if not isinstance(entry, dict):
            logger.warning(f"[{func_name}] State file entry #{index+1} is not a dictionary, skipping: {str(entry)[:100]}...")
            return False

        # Minimum required keys for any state
        required_keys = {'id', 'symbol', 'side', 'status', 'original_size'}
        # Optional but highly recommended keys: 'size', 'entry_order_type', 'last_update_time'
        # Keys required if status is ACTIVE or CLOSED_*
        active_or_closed_required = {'entry_price', 'entry_time', 'size'}

        missing_keys = required_keys - entry.keys()
        if missing_keys:
            logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {entry.get('id', 'N/A')}) missing required keys: {missing_keys}, skipping.")
            return False

        status = entry.get('status')
        pos_id = entry.get('id', 'N/A')

        if status in [STATUS_ACTIVE, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT]:
            missing_active = active_or_closed_required - entry.keys()
            if missing_active:
                 logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has status '{status}' but missing keys: {missing_active}, skipping.")
                 return False
            # Check for valid numeric values in critical fields for active/closed positions
            for key in ['entry_price', 'size']:
                 val = entry.get(key)
                 if val is None or not isinstance(val, (float, int)) or float(val) <= 0:
                      logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has status '{status}' but invalid {key}: {val}. Skipping.")
                      return False

        # Further type checks and conversions for numeric fields (allow flexibility but log warnings)
        numeric_keys = ['size', 'original_size', 'entry_price', 'stop_loss_price', 'take_profit_price', 'trailing_stop_price']
        for key in numeric_keys:
             if key in entry and entry[key] is not None:
                original_value = entry[key]
                if not isinstance(original_value, (float, int)):
                    try:
                        entry[key] = float(original_value)
                        if entry[key] != original_value: # Log if conversion happened
                            logger.debug(f"[{func_name}] Converted state key '{key}' for ID {pos_id} from {type(original_value)} '{original_value}' to float '{entry[key]}'.")
                    except (ValueError, TypeError):
                        logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has non-numeric value for '{key}': {original_value}. Keeping original, but might cause issues.")
                        # Depending on strictness, could return False here

        # Check status validity
        valid_statuses = [STATUS_PENDING_ENTRY, STATUS_ACTIVE, STATUS_CLOSING, STATUS_CANCELED,
                          STATUS_REJECTED, STATUS_EXPIRED, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT, STATUS_UNKNOWN]
        if status not in valid_statuses:
            logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has unrecognized status '{status}'. Treating as unknown.")
            entry['status'] = STATUS_UNKNOWN # Standardize

        return True

    def _load_state(self) -> None:
        """
        Loads bot state robustly from JSON file with backup and validation.
        """
        func_name = "_load_state"
        logger.info(f"[{func_name}] Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        initial_state_loaded = False
        loaded_from = ""

        # Attempt 1: Load main state file
        if os.path.exists(self.state_file):
            try:
                if os.path.getsize(self.state_file) == 0:
                    logger.warning(f"{Fore.YELLOW}[{func_name}] State file {self.state_file} is empty. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
                    initial_state_loaded = True
                    loaded_from = "empty_main"
                else:
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            logger.warning(f"{Fore.YELLOW}[{func_name}] State file {self.state_file} contains only whitespace. Starting fresh.{Style.RESET_ALL}")
                            self.open_positions = []
                            initial_state_loaded = True
                            loaded_from = "whitespace_main"
                        else:
                            saved_state_raw = json.loads(content)
                            if isinstance(saved_state_raw, list):
                                valid_positions = []
                                invalid_count = 0
                                for i, pos_data in enumerate(saved_state_raw):
                                    if self._validate_state_entry(pos_data, i):
                                        valid_positions.append(pos_data)
                                    else:
                                        invalid_count += 1
                                self.open_positions = valid_positions
                                loaded_from = "main_file"
                                log_msg = f"[{func_name}] Recalled {len(self.open_positions)} valid position(s) from {self.state_file}."
                                if invalid_count > 0:
                                    log_msg += f" Skipped {invalid_count} invalid entries."
                                logger.info(f"{Fore.GREEN}{log_msg}{Style.RESET_ALL}")
                                initial_state_loaded = True
                                # Try to remove old backup if main load was successful
                                if os.path.exists(state_backup_file):
                                    try:
                                        os.remove(state_backup_file)
                                        logger.debug(f"[{func_name}] Removed old state backup: {state_backup_file}")
                                    except OSError as remove_err:
                                        logger.warning(f"[{func_name}] Could not remove old state backup {state_backup_file}: {remove_err}")
                            else:
                                raise ValueError(f"Invalid state format - expected a list, got {type(saved_state_raw)}.")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"{Fore.RED}[{func_name}] Error decoding/validating primary state file {self.state_file}: {e}. Attempting recovery...{Style.RESET_ALL}")
                # Recovery attempt will happen below if initial_state_loaded is False
            except Exception as e:
                 logger.error(f"{Fore.RED}[{func_name}] Unexpected error reading primary state file {self.state_file}: {e}{Style.RESET_ALL}", exc_info=True)

        else: # Main file doesn't exist
            logger.info(f"[{func_name}] No prior state file found ({self.state_file}). Checking for backup...")
            loaded_from = "no_main_file"
            # Will proceed to backup check

        # Attempt 2: Recovery from backup if primary load failed or file was missing
        if not initial_state_loaded:
            if os.path.exists(state_backup_file):
                logger.warning(f"{Fore.YELLOW}[{func_name}] Attempting to restore state from backup: {state_backup_file}{Style.RESET_ALL}")
                try:
                    # Backup the potentially corrupted file before overwriting (if it exists)
                    corrupted_path = ""
                    if os.path.exists(self.state_file) and loaded_from != "no_main_file": # Only backup if it existed but was bad
                         corrupted_path = f"{self.state_file}.corrupted_{int(time.time())}"
                         shutil.copy2(self.state_file, corrupted_path)
                         logger.warning(f"[{func_name}] Backed up corrupted state file to {corrupted_path}")

                    shutil.copy2(state_backup_file, self.state_file)
                    logger.info(f"{Fore.GREEN}[{func_name}] State restored from backup {state_backup_file} to {self.state_file}. Retrying load...{Style.RESET_ALL}")
                    # Recurse ONLY ONCE for recovery to avoid infinite loop on bad backup
                    return self._load_state() # !!! CAUTION: Ensure this doesn't loop infinitely

                except Exception as restore_err:
                    logger.error(f"{Fore.RED}[{func_name}] Failed to restore state from backup {state_backup_file}: {restore_err}. Starting fresh.{Style.RESET_ALL}")
                    self.open_positions = []
                    loaded_from = "backup_restore_failed"
                    initial_state_loaded = True # Mark as loaded (fresh state) to prevent further attempts
            else:
                # No main file AND no backup file, or main file was bad AND no backup
                logger.warning(f"{Fore.YELLOW}[{func_name}] No valid state or backup file found. Starting fresh.{Style.RESET_ALL}")
                if loaded_from not in ["empty_main", "whitespace_main", "no_main_file"] and os.path.exists(self.state_file):
                     # Attempt to backup corrupted file if not already done (and if it existed)
                     try:
                         corrupted_path = f"{self.state_file}.corrupted_{int(time.time())}"
                         shutil.copy2(self.state_file, corrupted_path)
                         logger.warning(f"[{func_name}] Backed up corrupted state file to {corrupted_path}")
                     except Exception as backup_err:
                          logger.error(f"{Fore.RED}[{func_name}] Could not back up corrupted state file {self.state_file}: {backup_err}{Style.RESET_ALL}")

                self.open_positions = []
                loaded_from = "fresh_start"
                initial_state_loaded = True # Mark as loaded (fresh state)

        # Ensure initial state file exists even if starting fresh (or if loaded from backup)
        # Save empty list if starting fresh, or save the potentially restored/validated state
        if not os.path.exists(self.state_file) or os.path.getsize(self.state_file) == 0:
             logger.info(f"[{func_name}] Ensuring state file exists ({loaded_from}). Saving current state (Positions: {len(self.open_positions)}).")
             self._save_state()

    def _save_state(self) -> None:
        """
        Saves the current bot state (list of open positions) atomically.
        Uses tempfile, os.replace, and creates a backup.
        """
        func_name = "_save_state"
        if not hasattr(self, 'open_positions'):
             logger.error(f"[{func_name}] Cannot save state: 'open_positions' attribute missing.")
             return
        if not isinstance(self.open_positions, list):
             logger.error(f"[{func_name}] Cannot save state: 'open_positions' is not a list ({type(self.open_positions)}).")
             return

        state_backup_file = f"{self.state_file}.bak"
        state_dir = os.path.dirname(self.state_file) or '.' # Current dir if no path specified
        os.makedirs(state_dir, exist_ok=True) # Ensure directory exists

        temp_fd, temp_path = None, None
        try:
            # Create temp file securely in the target directory
            temp_fd, temp_path = tempfile.mkstemp(dir=state_dir, prefix=os.path.basename(self.state_file) + '.tmp_')
            logger.debug(f"[{func_name}] Saving state ({len(self.open_positions)} positions) via temp file {temp_path}...")

            # Prepare state data for JSON serialization (ensure basic types, handle numpy if necessary)
            def json_serializer(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)): # Handle arrays if they sneak in
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif pd.isna(obj): # Handle Pandas NaT or NaN
                    return None
                # Consider adding datetime serialization if needed:
                # elif isinstance(obj, datetime.datetime): return obj.isoformat()
                try: # Fallback string conversion, log if used unexpectedly
                    return str(obj)
                except Exception:
                    logger.warning(f"[{func_name}] Could not serialize object of type {type(obj)}: {obj}")
                    return None # Or raise error

            # Dump state to temporary file using the file descriptor
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(self.open_positions, f, indent=4, default=json_serializer)
            temp_fd = None # fd is now closed by the 'with' block

            # Create Backup of Current State File (if it exists) before replacing
            if os.path.exists(self.state_file):
                try:
                    shutil.copy2(self.state_file, state_backup_file) # copy2 preserves metadata
                    logger.debug(f"[{func_name}] Created state backup: {state_backup_file}")
                except Exception as backup_err:
                    logger.warning(f"[{func_name}] Could not create state backup {state_backup_file}: {backup_err}. Proceeding cautiously.")

            # Atomically Replace the Old State File with the New One
            os.replace(temp_path, self.state_file)
            temp_path = None # Reset temp_path as it's been moved/renamed
            logger.debug(f"[{func_name}] State recorded successfully to {self.state_file}")

        except (IOError, OSError, TypeError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Could not scribe state to {self.state_file} (Error Type: {type(e).__name__}): {e}{Style.RESET_ALL}", exc_info=True)
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True)
        finally:
            # Clean up temp file if it still exists (i.e., if error occurred before os.replace)
            if temp_fd is not None: # If fd was opened but not closed by 'with' (e.g., error during dump)
                try: os.close(temp_fd); logger.debug(f"[{func_name}] Closed dangling temp fd {temp_fd}")
                except OSError: pass
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"[{func_name}] Removed temp state file {temp_path} after error.")
                except OSError as rm_err:
                    logger.error(f"[{func_name}] Error removing temp state file {temp_path} after error: {rm_err}")

    def load_config(self, config_file: str) -> None:
        """Loads configuration from YAML or creates a default."""
        func_name = "load_config"
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            if not isinstance(self.config, dict):
                 logger.critical(f"{Fore.RED}[{func_name}] Fatal: Config file {config_file} has invalid structure (must be a dictionary). Aborting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.info(f"{Fore.GREEN}[{func_name}] Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Failed to create default config: {e}{Style.RESET_ALL}")
                # Exit even if default creation fails, as we need a config
            logger.critical(f"{Fore.YELLOW}[{func_name}] Exiting after creating default config. Please review '{config_file}' and restart.{Style.RESET_ALL}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.critical(f"{Fore.RED}[{func_name}] Fatal: Error parsing spellbook {config_file}: {e}. Check YAML syntax. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Fatal: Unexpected chaos loading configuration: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file."""
        func_name = "create_default_config"
        logger.info(f"[{func_name}] Crafting default spellbook: {config_file}...")
        # Default config structure with comments
        default_config = {
            "logging": {
                "level": "INFO", # Logging level for console (DEBUG, INFO, WARNING, ERROR)
            },
            "exchange": {
                "exchange_id": DEFAULT_EXCHANGE_ID, # Exchange name (e.g., 'bybit', 'binance')
                "testnet_mode": True, # Use exchange's testnet/sandbox environment
                # API keys should be set via environment variables (BYBIT_API_KEY, BYBIT_API_SECRET)
                # Or uncomment and add here (less secure):
                # "api_key": "YOUR_API_KEY",
                # "api_secret": "YOUR_API_SECRET",
            },
            "trading": {
                "symbol": "BTC/USDT:USDT", # Trading pair (e.g., Bybit Linear Perpetual format)
                "timeframe": DEFAULT_TIMEFRAME, # Candle timeframe (e.g., '1m', '5m', '1h')
                "simulation_mode": True, # Internal simulation (no real orders) - overrides testnet_mode for placing orders
                "entry_order_type": "limit", # 'market' or 'limit' for entry orders
                "limit_order_offset_buy": 0.0005, # 0.05% below current price for buy limit orders
                "limit_order_offset_sell": 0.0005, # 0.05% above current price for sell limit orders
                "close_positions_on_exit": False, # Attempt market close of active positions on bot shutdown
            },
            "order_book": {
                "depth": 10, # Number of bids/asks levels to fetch
                "imbalance_threshold": 1.5, # Ask/Bid volume ratio threshold (e.g., >1.5 for sell signal)
            },
            "indicators": {
                # Standard Deviation of log returns
                "volatility_window": 20, # Period for volatility calculation
                "volatility_multiplier": 0.0, # Multiplier for order size adjustment (0 to disable)
                # EMA
                "ema_period": 10, # Period for Exponential Moving Average
                # RSI
                "rsi_period": 14, # Period for Relative Strength Index
                # MACD
                "macd_short_period": 12, # Short EMA period for MACD
                "macd_long_period": 26, # Long EMA period for MACD
                "macd_signal_period": 9, # Signal line EMA period for MACD
                # Stochastic RSI
                "stoch_rsi_period": 14, # Period for underlying RSI calculation
                "stoch_rsi_k_period": 3, # %K smoothing period for StochRSI
                "stoch_rsi_d_period": 3, # %D smoothing period for StochRSI
                # ATR
                "atr_period": 14, # Period for Average True Range
            },
            "risk_management": {
                "order_size_percentage": 0.02, # % of available balance to use for order size (e.g., 0.02 = 2%)
                "max_open_positions": 1, # Maximum concurrent active/pending positions
                # Stop Loss / Take Profit Type
                "use_atr_sl_tp": True, # Use ATR for SL/TP calculation
                # ATR SL/TP Settings (used if use_atr_sl_tp is true)
                "atr_sl_multiplier": ATR_MULTIPLIER_SL, # Multiplier for ATR Stop Loss distance
                "atr_tp_multiplier": ATR_MULTIPLIER_TP, # Multiplier for ATR Take Profit distance
                # Fixed Percentage SL/TP Settings (used if use_atr_sl_tp is false)
                "stop_loss_percentage": 0.005, # 0.5% Stop Loss from entry price
                "take_profit_percentage": 0.01, # 1.0% Take Profit from entry price
                # Bybit V5 Trigger Prices for SL/TP
                "sl_trigger_by": "MarkPrice", # Trigger type: MarkPrice, LastPrice, IndexPrice (or null/None)
                "tp_trigger_by": "MarkPrice", # Trigger type: MarkPrice, LastPrice, IndexPrice (or null/None)
                # Experimental Trailing Stop Loss (requires exchange support for modifyOrder SL/TP)
                "enable_trailing_stop_loss": False, # EXPERIMENTAL: Enable Trailing Stop Loss
                "trailing_stop_loss_percentage": 0.003, # 0.3% trailing distance (if enabled)
                # Time-Based Exit
                "time_based_exit_minutes": 60, # Max position duration in minutes (0 or null to disable)
                # Signal Strength Size Adjustment
                "strong_signal_adjustment_factor": 1.0, # Multiplier for order size on strong signals (>= STRONG_SIGNAL_THRESHOLD_ABS)
                "weak_signal_adjustment_factor": 1.0,   # Multiplier for order size on weak signals (unused by default if score < entry threshold)
            },
        }
        # --- Environment Variable Overrides ---
        # Example for order size percentage - add others as needed
        env_order_pct_str = os.getenv("ORDER_SIZE_PERCENTAGE")
        if env_order_pct_str:
            try:
                override_val = float(env_order_pct_str)
                if 0 < override_val <= 1:
                    default_config["risk_management"]["order_size_percentage"] = override_val
                    logger.info(f"[{func_name}] Overrode order_size_percentage from environment to {override_val}.")
                else:
                    logger.warning(f"[{func_name}] Invalid ORDER_SIZE_PERCENTAGE ('{env_order_pct_str}') in env, must be > 0 and <= 1. Using default.")
            except ValueError:
                logger.warning(f"[{func_name}] Invalid ORDER_SIZE_PERCENTAGE ('{env_order_pct_str}') in env, not a number. Using default.")

        try:
            config_dir = os.path.dirname(config_file)
            if config_dir: # Ensure directory exists if specified
                 os.makedirs(config_dir, exist_ok=True)
                 # logger.info(f"[{func_name}] Ensured config directory exists: {config_dir}")

            with open(config_file, "w", encoding='utf-8') as f:
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False)
            logger.info(f"{Fore.YELLOW}[{func_name}] A default spellbook has been crafted: '{config_file}'.{Style.RESET_ALL}")
            # Message is now logged before sys.exit in load_config

        except IOError as e:
            logger.error(f"{Fore.RED}[{func_name}] Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}")
            raise # Re-raise to be caught by calling function

    def validate_config(self) -> None:
        """Performs detailed validation of the loaded configuration parameters."""
        func_name = "validate_config"
        logger.debug(f"[{func_name}] Scrutinizing the configuration spellbook...")
        try:
            # Helper functions for validation within this method's scope
            def _get_nested(data: Dict, keys: List[str], section: str, default: Any = KeyError, req_type: Optional[Union[type, Tuple[type,...]]] = None, allow_none: bool = False):
                """Safely gets nested value, checks type, returns default or raises."""
                value = data
                full_key_path = section
                try:
                    for key in keys:
                        full_key_path += f".{key}"
                        if isinstance(value, dict):
                            value = value[key]
                        else: raise KeyError # Current level is not a dict
                except KeyError:
                    if default is not KeyError: return default
                    raise KeyError(f"Missing required configuration key: '{full_key_path}'")
                except Exception as e: # Catch unexpected errors during access
                     raise ValueError(f"Error accessing config key '{full_key_path}': {e}")

                # Type and None checks
                if value is None:
                    if allow_none: return None
                    else: raise TypeError(f"Configuration key '{full_key_path}' cannot be None.")
                if req_type is not None and not isinstance(value, req_type):
                     # Special case: allow int where float is expected, perform conversion
                     if isinstance(req_type, tuple) and float in req_type and isinstance(value, int):
                          return float(value)
                     if req_type is float and isinstance(value, int):
                          return float(value)
                     raise TypeError(f"Configuration key '{full_key_path}' expects type {req_type} but got {type(value)}.")
                return value

            def _check_range(value: Union[int, float], key_path: str, min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None, min_exclusive: bool = False, max_exclusive: bool = False):
                """Checks if value is within the specified range."""
                if min_val is not None:
                    op = ">" if min_exclusive else ">="
                    if not (value > min_val if min_exclusive else value >= min_val):
                        raise ValueError(f"Configuration key '{key_path}' ({value}) must be {op} {min_val}")
                if max_val is not None:
                    op = "<" if max_exclusive else "<="
                    if not (value < max_val if max_exclusive else value <= max_val):
                        raise ValueError(f"Configuration key '{key_path}' ({value}) must be {op} {max_val}")

            # --- Validate Sections ---
            required_sections = ["logging", "exchange", "trading", "order_book", "indicators", "risk_management"]
            for section in required_sections:
                 if section not in self.config or not isinstance(self.config[section], dict):
                      raise ValueError(f"Config Error: Missing or invalid section '{section}'. Must be a dictionary.")

            # --- Logging ---
            log_level = _get_nested(self.config, ["level"], "logging", default="INFO", req_type=str).upper()
            if not hasattr(logging, log_level): raise ValueError(f"logging.level: Invalid level '{log_level}'.")

            # --- Exchange ---
            ex_id = _get_nested(self.config, ["exchange_id"], "exchange", req_type=str)
            if not ex_id or ex_id not in ccxt.exchanges: raise ValueError(f"exchange.exchange_id: Invalid or unsupported '{ex_id}'.")
            _get_nested(self.config, ["testnet_mode"], "exchange", req_type=bool)
            # API key validation is implicit in _initialize_exchange if not simulation_mode

            # --- Trading ---
            _get_nested(self.config, ["symbol"], "trading", req_type=str) # Symbol validated against exchange later
            _get_nested(self.config, ["timeframe"], "trading", req_type=str) # Timeframe validated against exchange later
            _get_nested(self.config, ["simulation_mode"], "trading", req_type=bool)
            entry_type = _get_nested(self.config, ["entry_order_type"], "trading", req_type=str).lower()
            if entry_type not in ["market", "limit"]: raise ValueError(f"trading.entry_order_type: Must be 'market' or 'limit'.")
            _check_range(_get_nested(self.config, ["limit_order_offset_buy"], "trading", req_type=(float, int)), "trading.limit_order_offset_buy", min_val=0)
            _check_range(_get_nested(self.config, ["limit_order_offset_sell"], "trading", req_type=(float, int)), "trading.limit_order_offset_sell", min_val=0)
            _get_nested(self.config, ["close_positions_on_exit"], "trading", req_type=bool)

            # --- Order Book ---
            _check_range(_get_nested(self.config, ["depth"], "order_book", req_type=int), "order_book.depth", min_val=1)
            _check_range(_get_nested(self.config, ["imbalance_threshold"], "order_book", req_type=(float, int)), "order_book.imbalance_threshold", min_val=0, min_exclusive=True)

            # --- Indicators ---
            periods = ["volatility_window", "ema_period", "rsi_period", "macd_short_period",
                       "macd_long_period", "macd_signal_period", "stoch_rsi_period",
                       "stoch_rsi_k_period", "stoch_rsi_d_period", "atr_period"]
            for p in periods: _check_range(_get_nested(self.config, [p], "indicators", req_type=int), f"indicators.{p}", min_val=1)
            _check_range(_get_nested(self.config, ["volatility_multiplier"], "indicators", req_type=(float, int)), "indicators.volatility_multiplier", min_val=0)
            if self.config["indicators"]["macd_short_period"] >= self.config["indicators"]["macd_long_period"]:
                 raise ValueError("indicators: macd_short_period must be less than macd_long_period.")

            # --- Risk Management ---
            use_atr = _get_nested(self.config, ["use_atr_sl_tp"], "risk_management", req_type=bool)
            if use_atr:
                _check_range(_get_nested(self.config, ["atr_sl_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_sl_multiplier", min_val=0, min_exclusive=True)
                _check_range(_get_nested(self.config, ["atr_tp_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_tp_multiplier", min_val=0, min_exclusive=True)
            else:
                sl_pct = _get_nested(self.config, ["stop_loss_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                tp_pct = _get_nested(self.config, ["take_profit_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                if sl_pct is None and tp_pct is None:
                     logger.warning(f"{Fore.YELLOW}[{func_name}] risk_management: use_atr_sl_tp is false, but both stop_loss_percentage and take_profit_percentage are missing/null. No SL/TP will be used.{Style.RESET_ALL}")
                if sl_pct is not None: _check_range(sl_pct, "risk_management.stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)
                if tp_pct is not None: _check_range(tp_pct, "risk_management.take_profit_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None] # None is valid (exchange default)
            sl_trig = _get_nested(self.config, ["sl_trigger_by"], "risk_management", default=None, req_type=str, allow_none=True)
            tp_trig = _get_nested(self.config, ["tp_trigger_by"], "risk_management", default=None, req_type=str, allow_none=True)
            if sl_trig not in valid_triggers: raise ValueError(f"risk_management.sl_trigger_by: Invalid trigger '{sl_trig}'. Valid: {valid_triggers}")
            if tp_trig not in valid_triggers: raise ValueError(f"risk_management.tp_trigger_by: Invalid trigger '{tp_trig}'. Valid: {valid_triggers}")

            enable_tsl = _get_nested(self.config, ["enable_trailing_stop_loss"], "risk_management", req_type=bool)
            if enable_tsl:
                tsl_pct = _get_nested(self.config, ["trailing_stop_loss_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                if tsl_pct is None: raise ValueError("risk_management.trailing_stop_loss_percentage must be set if enable_trailing_stop_loss is true.")
                _check_range(tsl_pct, "risk_management.trailing_stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            order_size_pct = _get_nested(self.config, ["order_size_percentage"], "risk_management", req_type=(float, int))
            _check_range(order_size_pct, "risk_management.order_size_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=False) # Allow 100%? Maybe <= 1 is better.
            _check_range(_get_nested(self.config, ["max_open_positions"], "risk_management", req_type=int), "risk_management.max_open_positions", min_val=1)
            time_exit = _get_nested(self.config, ["time_based_exit_minutes"], "risk_management", req_type=int, allow_none=True)
            if time_exit is not None: _check_range(time_exit, "risk_management.time_based_exit_minutes", min_val=0) # Allow 0 to disable
            _check_range(_get_nested(self.config, ["strong_signal_adjustment_factor"], "risk_management", req_type=(float, int)), "risk_management.strong_signal_adjustment_factor", min_val=0, min_exclusive=True)
            _check_range(_get_nested(self.config, ["weak_signal_adjustment_factor"], "risk_management", req_type=(float, int)), "risk_management.weak_signal_adjustment_factor", min_val=0, min_exclusive=True)

            logger.info(f"{Fore.GREEN}[{func_name}] Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except (ValueError, TypeError, KeyError) as e:
            logger.critical(f"{Fore.RED}[{func_name}] Configuration flaw detected: {e}. Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Unexpected chaos during configuration validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and configures the CCXT exchange instance."""
        func_name = "_initialize_exchange"
        logger.info(f"[{func_name}] Summoning exchange spirits for {self.exchange_id.upper()}...")

        creds_found = self.api_key and self.api_secret
        if not self.simulation_mode and not creds_found:
             logger.critical(f"{Fore.RED}[{func_name}] API Key/Secret essence missing (check .env or config). Cannot trade live/testnet. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        elif creds_found and self.simulation_mode:
             logger.warning(f"{Fore.YELLOW}[{func_name}] API Key/Secret found, but internal simulation_mode is True. Credentials will NOT be used for placing orders.{Style.RESET_ALL}")
        elif not creds_found and self.simulation_mode:
             logger.info(f"{Fore.CYAN}[{func_name}] Running in internal simulation mode. API credentials not required/found.{Style.RESET_ALL}")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    # Try to set default market type based on symbol if possible, else guess
                    'defaultType': 'swap' if ':usdt' in self.symbol.lower() or ':usd' in self.symbol.lower() else 'spot',
                    'adjustForTimeDifference': True,
                    # Bybit V5 options:
                    # 'recvWindow': 10000, # Increase if timeouts occur
                    # 'brokerId': 'YOUR_BROKER_ID', # If applicable
                },
                # Set newUpdates to False if state tracking relies solely on fetch_order/fetch_position calls
                # 'newUpdates': False,
            }
            # Add credentials ONLY if NOT in internal simulation mode
            if not self.simulation_mode:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                logger.debug(f"[{func_name}] API credentials loaded into exchange config.")
            else: # Add dummy keys if simulating, CCXT sometimes requires them
                exchange_config['apiKey'] = 'SIMULATED_KEY'
                exchange_config['secret'] = 'SIMULATED_SECRET'
                logger.debug(f"[{func_name}] Using dummy API credentials for simulation mode.")


            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT unified method (if not simulating internally)
            if not self.simulation_mode and self.testnet_mode:
                logger.info(f"[{func_name}] Attempting to enable exchange sandbox mode...")
                try:
                    exchange.set_sandbox_mode(True)
                    # Verify if it actually worked (some exchanges might silently fail)
                    # This often involves checking the API endpoint URL used
                    urls = exchange.urls.get('api', {})
                    api_url = urls.get('public') or urls.get('private') or str(urls)
                    if 'test' in str(api_url).lower() or 'sandbox' in str(api_url).lower():
                        logger.info(f"{Fore.YELLOW}[{func_name}] Exchange sandbox mode explicitly enabled via CCXT. API URL: {api_url}{Style.RESET_ALL}")
                    else:
                         logger.warning(f"{Fore.YELLOW}[{func_name}] CCXT set_sandbox_mode(True) called, but API URL ({api_url}) doesn't indicate testnet. Testnet may depend on API key type.{Style.RESET_ALL}")

                except ccxt.NotSupported:
                    logger.warning(f"{Fore.YELLOW}[{func_name}] {self.exchange_id} does not support unified set_sandbox_mode via CCXT. Testnet depends on API key type/URL.{Style.RESET_ALL}")
                except Exception as sandbox_err:
                    logger.error(f"{Fore.RED}[{func_name}] Error setting sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}")
            elif not self.simulation_mode and not self.testnet_mode:
                 logger.info(f"{Fore.GREEN}[{func_name}] Exchange sandbox mode is OFF. Operating on Mainnet.{Style.RESET_ALL}")

            # Load markets (needed for symbol validation, precision, limits, etc.)
            logger.debug(f"[{func_name}] Loading market matrix...")
            exchange.load_markets(reload=True) # Force reload to get latest info
            logger.debug(f"[{func_name}] Market matrix loaded.")

            # --- Validate Symbol and Timeframe against Exchange Data ---
            if self.symbol not in exchange.markets:
                available_sym = list(exchange.markets.keys())[:15] # Show some examples
                logger.critical(f"{Fore.RED}[{func_name}] Symbol '{self.symbol}' not found on {self.exchange_id}. Available examples: {available_sym}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)

            market_details = exchange.market(self.symbol)
            if 'timeframes' in exchange.has and self.timeframe not in exchange.timeframes:
                available_tf = list(exchange.timeframes.keys())[:15]
                logger.critical(f"{Fore.RED}[{func_name}] Timeframe '{self.timeframe}' unsupported by {self.exchange_id}. Available examples: {available_tf}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)
            elif 'timeframes' not in exchange.has:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Exchange does not list supported timeframes via CCXT. Assuming '{self.timeframe}' is valid.{Style.RESET_ALL}")

            if not market_details.get('active', True):
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Warning: Market '{self.symbol}' is marked as inactive by the exchange.{Style.RESET_ALL}")

            logger.info(f"[{func_name}] Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed available (or assumed).")

            # Perform initial API connectivity test (if not simulating)
            if not self.simulation_mode:
                logger.debug(f"[{func_name}] Performing initial API connectivity test (fetch balance)...")
                quote_currency = market_details.get('quote')
                if not quote_currency:
                     logger.warning(f"{Fore.YELLOW}[{func_name}] Could not determine quote currency from market info for balance check. Skipping check.{Style.RESET_ALL}")
                else:
                    # Use fetch_balance as a more comprehensive initial test
                    initial_balance_check = self.fetch_balance(currency_code=quote_currency)
                    if initial_balance_check is not None: # fetch_balance returns float or None on error
                         logger.info(f"{Fore.GREEN}[{func_name}] API connection and authentication successful (Balance Check OK).{Style.RESET_ALL}")
                    else:
                         # Fetch_balance logs specific errors, just add a critical failure message here
                         logger.critical(f"{Fore.RED}[{func_name}] Initial API connectivity/auth test (fetch_balance) failed. Check logs (esp. API key, permissions, IP whitelist). Aborting.{Style.RESET_ALL}")
                         sys.exit(1)

            logger.info(f"{Fore.GREEN}[{func_name}] Exchange spirits aligned for {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Fore.RED}[{func_name}] Authentication failed for {self.exchange_id}: {e}. Check API keys/permissions/IP whitelist. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
             logger.critical(f"{Fore.RED}[{func_name}] Connection failed to {self.exchange_id} ({type(e).__name__}): {e}. Check network/exchange status. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Unexpected error initializing exchange {self.exchange_id}: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _calculate_decimal_places(self, tick_size: Optional[Union[float, int, str]]) -> int:
        """Calculates decimal places from tick size (e.g., 0.01 -> 2). Handles various formats."""
        func_name = "_calculate_decimal_places"
        # Determine appropriate fallback based on context (price or amount) - requires context or use a generic default
        fallback_decimals = DEFAULT_PRICE_DECIMALS # Assuming price context if called standalone
        if tick_size is None:
            logger.debug(f"[{func_name}] Tick size is None. Using fallback decimals: {fallback_decimals}.")
            return fallback_decimals
        try:
            # Convert string tick size to float if necessary
            if isinstance(tick_size, str):
                try: tick_size = float(tick_size)
                except ValueError:
                    logger.warning(f"[{func_name}] Could not convert tick size string '{tick_size}' to float. Using fallback {fallback_decimals}.")
                    return fallback_decimals

            if not isinstance(tick_size, (float, int)):
                logger.warning(f"[{func_name}] Tick size '{tick_size}' is not float or int ({type(tick_size)}). Using fallback {fallback_decimals}.")
                return fallback_decimals

            if tick_size <= 0:
                logger.warning(f"[{func_name}] Tick size '{tick_size}' is non-positive. Using fallback {fallback_decimals}.")
                return fallback_decimals

            # Handle integer tick size (e.g., 1 means 0 decimal places)
            if isinstance(tick_size, int) or (isinstance(tick_size, float) and tick_size.is_integer()):
                 # logger.debug(f"[{func_name}] Integer tick size {tick_size} -> 0 decimals.")
                 return 0

            # Use string formatting method for robustness with scientific notation and precision issues
            # Format with high precision, remove trailing zeros, then count after decimal point
            s = format(tick_size, '.16f').rstrip('0')
            if '.' in s:
                decimals = len(s.split('.')[-1])
                # logger.debug(f"[{func_name}] Calculated {decimals} decimals for tick size {tick_size} from string '{s}'")
                return decimals
            else:
                # This case (float without '.' after formatting and rstrip) shouldn't happen if it's not an integer
                logger.warning(f"[{func_name}] Unexpected format for tick size {tick_size} ('{s}'). Assuming 0 decimals.")
                return 0
        except Exception as e:
            logger.error(f"[{func_name}] Unexpected error calculating decimals for tick size {tick_size}: {e}. Using fallback {fallback_decimals}.")
            return fallback_decimals

    def _load_market_info(self) -> None:
        """Loads and caches market info (precision, limits), calculating decimal places."""
        func_name = "_load_market_info"
        logger.debug(f"[{func_name}] Loading market details for {self.symbol}...")
        if not self.exchange or not self.exchange.markets:
            logger.critical(f"{Fore.RED}[{func_name}] Exchange not initialized or markets not loaded. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        try:
            self.market_info = self.exchange.market(self.symbol)
            if not self.market_info:
                # This should have been caught in _initialize_exchange, but double-check
                raise ValueError(f"Market info for '{self.symbol}' not found even after loading markets.")

            precision = self.market_info.get('precision', {})
            limits = self.market_info.get('limits', {})
            amount_tick = precision.get('amount')
            price_tick = precision.get('price')
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            # Use _calculate_decimal_places for robustness
            self.price_decimals = self._calculate_decimal_places(price_tick)
            self.amount_decimals = self._calculate_decimal_places(amount_tick)

            # Log warnings if essential info missing or calculation defaulted
            if amount_tick is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Amount tick size (precision.amount) missing for {self.symbol}. Using calculated/default amount decimals: {self.amount_decimals}. Order rounding might be inaccurate.{Style.RESET_ALL}")
            if price_tick is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Price tick size (precision.price) missing for {self.symbol}. Using calculated/default price decimals: {self.price_decimals}. Order/SL/TP rounding might be inaccurate.{Style.RESET_ALL}")
            if min_amount is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Min order amount limit (limits.amount.min) missing for {self.symbol}. Size checks might be incomplete.{Style.RESET_ALL}")
            if min_cost is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Min order cost limit (limits.cost.min) missing for {self.symbol}. Cost checks might be incomplete.{Style.RESET_ALL}")

            logger.info(f"[{func_name}] Market Details for {self.symbol}: Price Decimals={self.price_decimals} (Tick: {price_tick}), Amount Decimals={self.amount_decimals} (Tick: {amount_tick})")
            logger.debug(f"[{func_name}] Limits: Min Amount={min_amount}, Min Cost={min_cost}")
            # Log market type for context
            market_type = self.market_info.get('type', 'unknown')
            linear_inverse = "linear" if self.market_info.get('linear') else ("inverse" if self.market_info.get('inverse') else "N/A")
            logger.debug(f"[{func_name}] Market Type: {market_type}, Contract Type: {linear_inverse}")
            logger.debug(f"[{func_name}] Market details loaded and cached.")

        except (KeyError, ValueError, TypeError) as e:
             # Log the problematic market_info structure for debugging
             logger.critical(f"{Fore.RED}[{func_name}] Error loading/parsing market info for {self.symbol}: {e}. Market Info structure: {str(self.market_info)[:500]}... Aborting.{Style.RESET_ALL}", exc_info=False)
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Failed to load crucial market info for {self.symbol}: {e}. Aborting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    # --- Data Fetching Methods ---

    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol."""
        func_name = "fetch_market_price"
        # logger.debug(f"[{func_name}] Fetching ticker for {self.symbol}...")
        ticker = self.exchange.fetch_ticker(self.symbol)
        last_price = ticker.get('last') if ticker else None
        if last_price is not None:
            try:
                price = float(last_price)
                if price <= 0:
                     logger.warning(f"{Fore.YELLOW}[{func_name}] Fetched 'last' price ({last_price}) is non-positive for {self.symbol}.{Style.RESET_ALL}")
                     return None
                # logger.debug(f"[{func_name}] Current market price ({self.symbol}): {self.format_price(price)}")
                return price
            except (ValueError, TypeError) as e:
                 logger.error(f"{Fore.RED}[{func_name}] Error converting ticker 'last' price ({last_price}) to float for {self.symbol}: {e}{Style.RESET_ALL}")
                 return None
        else:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Could not fetch valid 'last' price for {self.symbol}. Ticker: {str(ticker)[:200]}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self) -> Optional[Dict[str, Any]]:
        """Fetches order book and calculates volume imbalance."""
        func_name = "fetch_order_book"
        # logger.debug(f"[{func_name}] Fetching order book for {self.symbol} (depth: {self.order_book_depth})...")
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            result = {'bids': bids, 'asks': asks, 'imbalance': None}

            # Validate structure and filter non-numeric/negative volumes
            valid_bids = [float(bid[1]) for bid in bids if isinstance(bid, (list, tuple)) and len(bid) >= 2 and isinstance(bid[1], (int, float)) and bid[1] >= 0]
            valid_asks = [float(ask[1]) for ask in asks if isinstance(ask, (list, tuple)) and len(ask) >= 2 and isinstance(ask[1], (int, float)) and ask[1] >= 0]

            if not valid_bids or not valid_asks:
                 logger.debug(f"[{func_name}] Order book data invalid/incomplete for {self.symbol}. Valid Bids: {len(valid_bids)}/{len(bids)}, Valid Asks: {len(valid_asks)}/{len(asks)}")
                 return result # Return structure but imbalance will be None

            bid_volume = sum(valid_bids)
            ask_volume = sum(valid_asks)
            epsilon = 1e-12 # Small number to avoid division by zero

            if bid_volume > epsilon:
                imbalance_ratio = ask_volume / bid_volume
                result['imbalance'] = imbalance_ratio
                # logger.debug(f"[{func_name}] Order Book ({self.symbol}) Imbalance (Ask/Bid): {imbalance_ratio:.3f}")
            elif ask_volume > epsilon: # Bids near zero, asks exist
                result['imbalance'] = float('inf') # Represent infinite imbalance
                logger.debug(f"[{func_name}] Order Book ({self.symbol}) Imbalance: Inf (Near-Zero Bid Vol)")
            else: # Both volumes near zero
                result['imbalance'] = None # Cannot calculate meaningful imbalance
                logger.debug(f"[{func_name}] Order Book ({self.symbol}) Imbalance: N/A (Near-Zero Bid & Ask Volumes)")
            return result
        except Exception as e:
            # Log with less severity unless it's critical? Maybe warning is enough.
            logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/processing order book for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=False) # exc_info=False to reduce noise
            return None

    @retry_api_call(max_retries=2, initial_delay=1)
    def fetch_historical_data(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data. Calculates required candle count based
        on the longest indicator period if limit is not specified.
        """
        func_name = "fetch_historical_data"
        min_required_len_for_calc: int = 50 # Default absolute minimum
        fetch_limit: int = 100 # Default fetch limit

        if limit is None:
             # Calculate the maximum lookback needed by any indicator
             required_periods = [
                 self.volatility_window + 1, # Std needs N+1 points for N window
                 self.ema_period,
                 self.rsi_period + 1, # Diff needs +1
                 self.macd_long_period + self.macd_signal_period, # Needs long period, then signal period on MACD line
                 # StochRSI needs RSI series (rsi_p+1), then Stoch window (stoch_p), then smoothing (max(k,d))
                 self.rsi_period + 1 + self.stoch_rsi_period + max(self.stoch_rsi_k_period, self.stoch_rsi_d_period),
                 self.atr_period + 1 # Needs shift
             ]
             valid_periods = [p for p in required_periods if isinstance(p, int) and p > 0]
             max_lookback = max(valid_periods) if valid_periods else 100 # Use 100 if no indicators set
             min_required_len_for_calc = max_lookback # Min valid rows needed AFTER cleaning

             # Add buffer for stability and ensure a reasonable minimum fetch
             buffer = 50
             required_fetch = max(min_required_len_for_calc + buffer, 150)
             # Cap request size to avoid overwhelming the API (common limit is 1000 or 1500)
             fetch_limit = min(required_fetch, 1000)
             logger.debug(f"[{func_name}] Calculated required history: Lookback={max_lookback}, Fetch Limit={fetch_limit}")
        else:
             fetch_limit = limit
             # Estimate minimum needed if limit is manually overridden
             min_required_len_for_calc = max(fetch_limit - 50, 50) # Assume some buffer was included
             logger.debug(f"[{func_name}] Using provided history limit: {fetch_limit}")

        logger.debug(f"[{func_name}] Fetching ~{fetch_limit} historical candles for {self.symbol} ({self.timeframe})...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=fetch_limit)

            if not ohlcv:
                logger.warning(f"{Fore.YELLOW}[{func_name}] No historical OHLCV data returned for {self.symbol} ({self.timeframe}, limit {fetch_limit}).{Style.RESET_ALL}")
                return None
            if len(ohlcv) < 5: # Need at least a few candles for basic checks
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Very few historical OHLCV data points returned ({len(ohlcv)}). Insufficient.{Style.RESET_ALL}")
                 return None

            # Convert to DataFrame and process
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            # Convert OHLCV columns to numeric, coercing errors to NaN
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            initial_len = len(df)
            # Drop rows where essential price data is missing (NaN)
            df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            # Optional: Fill NaN volumes with 0 if volume indicator is used downstream
            # df['volume'].fillna(0, inplace=True)
            final_len = len(df)

            if final_len < initial_len:
                logger.debug(f"[{func_name}] Dropped {initial_len - final_len} rows with NaNs in OHLC from history.")

            # Check if enough data remains *after* cleaning for calculations
            if final_len < min_required_len_for_calc:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Insufficient valid historical data after cleaning for {self.symbol}. Got {final_len} rows, need ~{min_required_len_for_calc} for full indicator calculation. Indicators might be inaccurate or None.{Style.RESET_ALL}")
                 if final_len == 0: return None # Cannot proceed if DataFrame becomes empty

            logger.debug(f"[{func_name}] Fetched and processed {final_len} valid historical candles for {self.symbol}.")
            return df

        except ccxt.BadSymbol as e:
             # This is likely a config error, treat as critical
             logger.critical(f"{Fore.RED}[{func_name}] BadSymbol fetching history for {self.symbol}: {e}. Check symbol configuration. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] Error fetching/processing historical data for {self.symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call()
    def fetch_balance(self, currency_code: Optional[str] = None) -> Optional[float]:
        """
        Fetches the available balance for a specific currency (quote currency by default).
        Handles Bybit V5 Unified/Contract/Spot accounts intelligently. Returns None on failure.
        """
        func_name = "fetch_balance"
        # Determine target currency (usually quote currency of the trading pair)
        quote_currency = currency_code or (self.market_info.get('quote') if self.market_info else None)
        if not quote_currency:
             logger.error(f"{Fore.RED}[{func_name}] Cannot determine currency code. Market Info: {self.market_info}, Code passed: {currency_code}.{Style.RESET_ALL}")
             return None

        logger.debug(f"[{func_name}] Fetching available balance for {quote_currency}...")

        # --- Simulation Mode ---
        if self.simulation_mode:
            # Return a large dummy balance for simulation
            dummy_balance = 100000.0
            logger.warning(f"{Fore.YELLOW}[SIMULATION][{func_name}] Returning dummy balance: {self.format_price(dummy_balance)} {quote_currency}{Style.RESET_ALL}")
            return dummy_balance

        # --- Live/Testnet Mode ---
        try:
            params = {}
            # Bybit V5 specific: Determine accountType based on market info if possible
            # This helps ccxt target the right endpoint/account category
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                market_type = self.market_info.get('type', 'swap')
                if market_type in ['swap', 'future']:
                    # For derivatives, unified is common, but contract also exists.
                    # Let's try UNIFIED first, but be aware CONTRACT might be needed for some older accounts.
                    params = {'accountType': 'UNIFIED'}
                    logger.debug(f"[{func_name}] Using Bybit accountType hint: UNIFIED (for {market_type})")
                elif market_type == 'spot':
                    params = {'accountType': 'SPOT'}
                    logger.debug(f"[{func_name}] Using Bybit accountType hint: SPOT")
                # If type is option, add 'OPTION'
                elif market_type == 'option':
                     params = {'accountType': 'OPTION'}
                     logger.debug(f"[{func_name}] Using Bybit accountType hint: OPTION")

            # Fetch balance data from the exchange
            balance_data = self.exchange.fetch_balance(params=params)
            available_balance_str: Optional[str] = None # Store as string initially for flexibility

            # --- Strategy: Standard 'free' -> Bybit 'info' (more detailed) -> Fallback 'total' ---

            # 1. Standard CCXT 'free' balance
            if quote_currency in balance_data:
                free_balance = balance_data[quote_currency].get('free')
                if free_balance is not None:
                    available_balance_str = str(free_balance)
                    logger.debug(f"[{func_name}] Found standard CCXT 'free' balance: {available_balance_str} {quote_currency}")

            # 2. Parse 'info' (Exchange-Specific - Bybit V5 Example)
            # Check if standard 'free' was zero, None, or potentially invalid
            is_free_usable = False
            if available_balance_str is not None:
                try: is_free_usable = (float(available_balance_str) > 1e-12)
                except (ValueError, TypeError): pass

            # If standard 'free' is not usable, try parsing the raw 'info' field (especially for Bybit)
            if not is_free_usable and self.exchange_id.lower() == 'bybit':
                logger.debug(f"[{func_name}] Standard 'free' zero/missing/invalid. Checking Bybit 'info' structure...")
                info_data = balance_data.get('info', {})
                try:
                    # Bybit V5 /v5/account/wallet-balance structure
                    # Expects info['result']['list'][0]['coin'] which is a list of coin balances
                    result_list = info_data.get('result', {}).get('list', [])
                    if result_list and isinstance(result_list, list) and len(result_list) > 0 and isinstance(result_list[0], dict):
                        account_info = result_list[0] # Usually the first item contains the relevant account type balances
                        account_type = account_info.get('accountType') # e.g., UNIFIED, CONTRACT, SPOT
                        logger.debug(f"[{func_name}] Parsing Bybit 'info': Account Type = {account_type}")

                        coin_list = account_info.get('coin', [])
                        found_coin = False
                        for coin_data in coin_list:
                            if isinstance(coin_data, dict) and coin_data.get('coin') == quote_currency:
                                found_coin = True
                                # Prioritize fields likely representing available margin/cash:
                                # 1. availableToWithdraw (Most reliable 'free' cash)
                                # 2. availableBalance (Unified Margin specific, might include borrow?)
                                # 3. walletBalance (Total, includes unrealized PnL, less ideal for placing new orders)
                                avail_str = coin_data.get('availableToWithdraw')
                                source = "availableToWithdraw"
                                if avail_str is None or float(avail_str) < 1e-12:
                                     avail_str = coin_data.get('availableBalance') # Relevant for UNIFIED
                                     source = "availableBalance"
                                if avail_str is None or float(avail_str) < 1e-12:
                                     avail_str = coin_data.get('walletBalance') # Fallback
                                     source = "walletBalance"

                                if avail_str is not None:
                                    available_balance_str = str(avail_str)
                                    logger.debug(f"[{func_name}] Using Bybit '{source}' from 'info' for {quote_currency}: {available_balance_str}")
                                    break # Found the coin, stop searching
                        if not found_coin:
                             logger.debug(f"[{func_name}] Currency {quote_currency} not found within Bybit 'info' coin list for account type {account_type}.")

                    else: logger.debug(f"[{func_name}] Bybit 'info.result.list' structure not as expected or empty.")
                except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
                    # Log as warning, as we might fallback to 'total'
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Could not parse Bybit balance 'info' field: {e}. Raw info snippet: {str(info_data)[:200]}...{Style.RESET_ALL}")

            # 3. Fallback to 'total' balance (Less Accurate, includes used margin)
            # Check again if we have a usable balance string
            is_available_usable = False
            if available_balance_str is not None:
                try: is_available_usable = (float(available_balance_str) > 1e-12)
                except (ValueError, TypeError): pass

            if not is_available_usable and quote_currency in balance_data:
                 total_balance = balance_data[quote_currency].get('total')
                 if total_balance is not None:
                     total_balance_str = str(total_balance)
                     logger.warning(f"{Fore.YELLOW}[{func_name}] Available balance zero/missing/invalid. Using 'total' balance ({total_balance_str}) as fallback for {quote_currency}. (NOTE: Includes used margin/collateral){Style.RESET_ALL}")
                     available_balance_str = total_balance_str

            # --- Final Conversion and Return ---
            if available_balance_str is not None:
                try:
                    final_balance = float(available_balance_str)
                    # Handle potential negative balance (e.g., due to fees or liquidation debt)
                    if final_balance < 0:
                        logger.warning(f"{Fore.YELLOW}[{func_name}] Fetched balance for {quote_currency} is negative ({final_balance}). Treating as 0.0 available.{Style.RESET_ALL}")
                        final_balance = 0.0
                    # Use INFO level for the final successful balance fetch
                    logger.info(f"[{func_name}] Available balance for {quote_currency}: {self.format_price(final_balance)}")
                    return final_balance
                except (ValueError, TypeError):
                    logger.error(f"{Fore.RED}[{func_name}] Could not convert final balance string '{available_balance_str}' to float for {quote_currency}. Returning None.{Style.RESET_ALL}")
                    return None
            else:
                # If we reach here, no usable balance ('free', 'info', or 'total') was found
                logger.error(f"{Fore.RED}[{func_name}] Failed to determine available balance for {quote_currency} after checking all sources. Returning None.{Style.RESET_ALL}")
                # Log raw balance data at DEBUG level for inspection
                logger.debug(f"[{func_name}] Raw balance data: {str(balance_data)[:500]}...")
                return None # Explicitly return None on failure

        except ccxt.AuthenticationError as e:
             # This error should ideally be caught by the retry decorator, but handle here too
             logger.error(f"{Fore.RED}[{func_name}] Authentication failed fetching balance: {e}. Check API credentials.{Style.RESET_ALL}")
             return None
        except Exception as e:
            # Catch any other unexpected errors during the process
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching balance for {quote_currency}: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    @retry_api_call(max_retries=1) # Usually don't want to retry fetching status aggressively
    def fetch_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetches status of a specific order using ccxt unified fetch_order.
        Returns the order dictionary or None if not found, error, or invalid ID.
        Handles simulation mode by checking internal state.
        """
        func_name = "fetch_order_status"
        if not order_id or not isinstance(order_id, str):
            logger.warning(f"[{func_name}] Called with invalid order_id: {order_id}. Returning None.")
            return None

        target_symbol = symbol or self.symbol
        logger.debug(f"[{func_name}] Fetching status for order {order_id} ({target_symbol})...")

        # --- Simulation Mode ---
        if self.simulation_mode:
            # Find the order/position in our internal state
            pos = next((p for p in self.open_positions if p.get('id') == order_id), None)
            if pos:
                logger.debug(f"[SIMULATION][{func_name}] Returning cached status for simulated order {order_id}.")
                # Construct a simplified order dict mimicking ccxt structure based on state
                sim_status = pos.get('status', STATUS_UNKNOWN)
                # Map internal status to rough ccxt equivalents
                ccxt_status = 'open' if sim_status == STATUS_PENDING_ENTRY else \
                              'closed' if sim_status == STATUS_ACTIVE else \
                              'canceled' if sim_status == STATUS_CANCELED else \
                              sim_status # Pass others like 'rejected', 'expired', 'closed_externally'

                sim_filled = pos.get('size', 0.0) if sim_status == STATUS_ACTIVE else 0.0
                sim_avg = pos.get('entry_price') if sim_status == STATUS_ACTIVE else None
                sim_amount = pos.get('original_size', 0.0)
                sim_remaining = max(0.0, sim_amount - sim_filled) if sim_amount else 0.0
                # Use last_update_time or entry_time for timestamp
                sim_timestamp_s = pos.get('last_update_time') or pos.get('entry_time') or time.time()
                sim_timestamp_ms = int(sim_timestamp_s * 1000)

                return {
                    'id': order_id,
                    'symbol': target_symbol,
                    'status': ccxt_status,
                    'type': pos.get('entry_order_type', 'limit'),
                    'side': pos.get('side'),
                    'price': pos.get('limit_price'), # Price of the limit order if applicable
                    'average': sim_avg, # Fill price
                    'amount': sim_amount, # Original requested amount
                    'filled': sim_filled, # Filled amount
                    'remaining': sim_remaining,
                    'timestamp': sim_timestamp_ms,
                    'datetime': pd.to_datetime(sim_timestamp_ms, unit='ms', utc=True).isoformat(),
                    'stopLossPrice': pos.get('stop_loss_price'),
                    'takeProfitPrice': pos.get('take_profit_price'),
                    'info': {'simulated': True, 'orderId': order_id, 'internalStatus': sim_status} # Add internal status for clarity
                }
            else:
                # If not found in open_positions, maybe it was an old closed one? For simulation, assume not found.
                logger.warning(f"[SIMULATION][{func_name}] Simulated order {order_id} not found in current open_positions state.")
                return None # Not found in active/pending state

        # --- Live/Testnet Mode ---
        else:
            try:
                params = {}
                # Add Bybit V5 category param hint
                if self.exchange_id.lower() == 'bybit' and self.market_info:
                    market_type = self.market_info.get('type', 'swap')
                    is_linear = self.market_info.get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'
                    elif market_type == 'option': params['category'] = 'option'
                    # logger.debug(f"[{func_name}] Using Bybit category hint: {params.get('category')}")

                # Use the unified fetch_order method
                order_info = self.exchange.fetch_order(order_id, target_symbol, params=params)

                # Log key details from the fetched order
                status = order_info.get('status', STATUS_UNKNOWN)
                filled = order_info.get('filled', 0.0)
                avg_price = order_info.get('average')
                logger.debug(f"[{func_name}] Order {order_id} Status={status}, Filled={self.format_amount(filled)}, AvgPrice={self.format_price(avg_price)}")
                return order_info

            except ccxt.OrderNotFound as e:
                # This is common for orders that were filled/cancelled long ago or never existed
                logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} ({target_symbol}) not found on exchange. Assumed closed/cancelled/invalid. Error: {e}{Style.RESET_ALL}")
                return None # Return None to indicate it's not available
            except ccxt.ExchangeError as e:
                 # Check if the error message indicates a final state (already closed/filled)
                 err_str = str(e).lower()
                 if any(phrase in err_str for phrase in ["order is finished", "order has been filled", "already closed", "order status error", "order was canceled"]):
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} reported as finished/filled/closed via ExchangeError. Treating as 'not found'. Error: {e}{Style.RESET_ALL}")
                      return None # Treat as effectively not found/final state
                 else:
                      # Log other exchange errors more seriously
                      logger.error(f"{Fore.RED}[{func_name}] Exchange error fetching order {order_id}: {e}. Returning None.{Style.RESET_ALL}")
                      return None
            except Exception as e:
                # Catch any unexpected Python errors during the fetch
                logger.error(f"{Fore.RED}[{func_name}] Unexpected Python error fetching order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                return None

    # --- Indicator Calculation Methods (Static) ---
    # Added minimum length checks and basic error handling

    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> Optional[float]:
        """Calculates rolling log return standard deviation (volatility)."""
        func_name = "calculate_volatility"
        # Need window + 1 prices for 'window' returns
        min_len = window + 1
        if close_prices is None or not isinstance(close_prices, pd.Series) or window <= 0:
             # logger.error(f"[{func_name}] Invalid input: close_prices={type(close_prices)}, window={window}")
             return None
        if len(close_prices.dropna()) < min_len:
            # logger.debug(f"[{func_name}] Insufficient data length ({len(close_prices.dropna())}) for window {window}. Need {min_len}.")
            return None
        try:
             # Ensure prices are positive for log returns
            if (close_prices <= 0).any():
                # logger.warning(f"[{func_name}] Non-positive prices found, cannot calculate log returns.")
                return None
            log_returns = np.log(close_prices / close_prices.shift(1))
            # Ensure enough non-NaN returns for the window
            if log_returns.dropna().shape[0] < window:
                 # logger.debug(f"[{func_name}] Insufficient non-NaN log returns ({log_returns.dropna().shape[0]}) for window {window}.")
                 return None
            volatility = log_returns.rolling(window=window, min_periods=window).std(ddof=1).iloc[-1]
            return float(volatility) if pd.notna(volatility) else None
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating volatility (window {window}): {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates Exponential Moving Average (EMA)."""
        func_name = "calculate_ema"
        min_len = period # Need at least 'period' points for EMA calculation with min_periods
        if close_prices is None or not isinstance(close_prices, pd.Series) or period <= 0: return None
        if len(close_prices.dropna()) < min_len:
            # logger.debug(f"[{func_name}] Insufficient data length ({len(close_prices.dropna())}) for EMA period {period}.")
            return None
        try:
            ema = close_prices.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating EMA (period {period}): {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates Relative Strength Index (RSI)."""
        func_name = "calculate_rsi"
        # Need period+1 prices for 'period' deltas
        min_len = period + 1
        if close_prices is None or not isinstance(close_prices, pd.Series) or period <= 0: return None
        if len(close_prices.dropna()) < min_len:
            # logger.debug(f"[{func_name}] Insufficient data length ({len(close_prices.dropna())}) for RSI period {period}. Need {min_len}.")
            return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).fillna(0) # Fill first NaN gain after diff
            loss = -delta.where(delta < 0, 0.0).fillna(0) # Fill first NaN loss after diff

            # Use EWM directly for Wilder's smoothing (common for RSI)
            avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

            # Calculate RSI
            rs = avg_gain / (avg_loss + 1e-12) # Add epsilon for stability if avg_loss is zero
            rsi = 100.0 - (100.0 / (1.0 + rs))
            rsi_val = rsi.iloc[-1]

            # Ensure RSI is within 0-100 range
            return float(max(0.0, min(100.0, rsi_val))) if pd.notna(rsi_val) else None
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating RSI (period {period}): {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_macd(close_prices: pd.Series, short_p: int, long_p: int, signal_p: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculates MACD line, Signal line, and Histogram."""
        func_name = "calculate_macd"
        # Minimum length roughly requires long_p for EMAs, then signal_p on MACD line
        min_len = long_p + signal_p
        if close_prices is None or not isinstance(close_prices, pd.Series) or not all(p > 0 for p in [short_p, long_p, signal_p]): return None, None, None
        if short_p >= long_p: logger.error(f"[{func_name}] MACD short period ({short_p}) must be less than long period ({long_p})."); return None, None, None
        if len(close_prices.dropna()) < min_len:
            # logger.debug(f"[{func_name}] Insufficient data length ({len(close_prices.dropna())}) for MACD ({short_p},{long_p},{signal_p}). Need ~{min_len}.")
            return None, None, None
        try:
            ema_short = close_prices.ewm(span=short_p, adjust=False, min_periods=short_p).mean()
            ema_long = close_prices.ewm(span=long_p, adjust=False, min_periods=long_p).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal_p, adjust=False, min_periods=signal_p).mean()
            histogram = macd_line - signal_line

            # Get the last values
            macd_val = macd_line.iloc[-1]
            signal_val = signal_line.iloc[-1]
            hist_val = histogram.iloc[-1]

            # Check if any result is NaN (can happen if min_periods aren't met despite length check)
            if any(pd.isna(v) for v in [macd_val, signal_val, hist_val]):
                # logger.debug(f"[{func_name}] MACD calculation resulted in NaN values.")
                return None, None, None
            return float(macd_val), float(signal_val), float(hist_val)
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating MACD ({short_p},{long_p},{signal_p}): {e}", exc_info=False)
            return None, None, None

    @staticmethod
    def calculate_rsi_series(close_prices: pd.Series, period: int) -> Optional[pd.Series]:
        """Helper to calculate the full RSI series needed for StochRSI."""
        func_name = "calculate_rsi_series"
        min_len = period + 1
        if close_prices is None or not isinstance(close_prices, pd.Series) or period <= 0: return None
        if len(close_prices.dropna()) < min_len: return None
        try:
            delta = close_prices.diff(1)
            gain = delta.where(delta > 0, 0.0).fillna(0)
            loss = -delta.where(delta < 0, 0.0).fillna(0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
            rs = avg_gain / (avg_loss + 1e-12)
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi.clip(0, 100) # Clip entire series to 0-100
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating RSI Series (period {period}): {e}", exc_info=False)
            return None

    @staticmethod
    def calculate_stoch_rsi(close_prices: pd.Series, rsi_p: int, stoch_p: int, k_p: int, d_p: int) -> Tuple[Optional[float], Optional[float]]:
        """Calculates Stochastic RSI (%K and %D)."""
        func_name = "calculate_stoch_rsi"
        # Estimate minimum length needed
        min_len_est = rsi_p + 1 + stoch_p + max(k_p, d_p)
        if close_prices is None or not isinstance(close_prices, pd.Series) or not all(p > 0 for p in [rsi_p, stoch_p, k_p, d_p]): return None, None
        if len(close_prices.dropna()) < min_len_est:
            # logger.debug(f"[{func_name}] Insufficient data length ({len(close_prices.dropna())}) for StochRSI ({rsi_p},{stoch_p},{k_p},{d_p}). Need ~{min_len_est}.")
            return None, None
        try:
            # 1. Calculate RSI Series
            rsi_series = ScalpingBot.calculate_rsi_series(close_prices, rsi_p)
            if rsi_series is None or rsi_series.isna().all():
                # logger.debug(f"[{func_name}] Underlying RSI series calculation failed.")
                return None, None

            # Drop initial NaNs from RSI series
            rsi_series = rsi_series.dropna()
            # Check length *after* dropping NaNs from RSI calculation
            if len(rsi_series) < stoch_p + max(k_p, d_p):
                 # logger.debug(f"[{func_name}] Insufficient non-NaN RSI values ({len(rsi_series)}) for Stoch window({stoch_p}) + smoothing({max(k_p,d_p)}).")
                 return None, None

            # 2. Calculate Stochastics on RSI Series
            min_rsi = rsi_series.rolling(window=stoch_p, min_periods=stoch_p).min()
            max_rsi = rsi_series.rolling(window=stoch_p, min_periods=stoch_p).max()
            # Add epsilon to denominator to prevent division by zero if max == min
            stoch_rsi_raw = (100.0 * (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)).clip(0, 100)

            # 3. Calculate %K and %D (smoothing)
            stoch_k = stoch_rsi_raw.rolling(window=k_p, min_periods=k_p).mean()
            stoch_d = stoch_k.rolling(window=d_p, min_periods=d_p).mean()

            # Get last values
            k_val = stoch_k.iloc[-1]
            d_val = stoch_d.iloc[-1]

            if pd.isna(k_val) or pd.isna(d_val):
                # logger.debug(f"[{func_name}] StochRSI K or D calculation resulted in NaN.")
                return None, None

            # Clip final values just in case (should be handled by earlier clips)
            return float(max(0.0, min(100.0, k_val))), float(max(0.0, min(100.0, d_val)))
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating StochRSI ({rsi_p},{stoch_p},{k_p},{d_p}): {e}", exc_info=False)
            return None, None

    @staticmethod
    def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> Optional[float]:
        """Calculates Average True Range (ATR) using Wilder's smoothing."""
        func_name = "calculate_atr"
        # Need period+1 points for TR calculation involving previous close
        min_len = period + 1
        if high_prices is None or low_prices is None or close_prices is None or period <= 0: return None
        if not isinstance(high_prices, pd.Series) or not isinstance(low_prices, pd.Series) or not isinstance(close_prices, pd.Series): return None
        # Check length of all inputs after dropping potential NaNs individually first? Or assume aligned index. Let's check combined length.
        df_temp = pd.DataFrame({'h': high_prices, 'l': low_prices, 'c': close_prices}).dropna()
        if len(df_temp) < min_len:
            # logger.debug(f"[{func_name}] Insufficient aligned data length ({len(df_temp)}) for ATR period {period}. Need {min_len}.")
            return None

        high, low, close = df_temp['h'], df_temp['l'], df_temp['c'] # Use cleaned series
        try:
            high_low = high - low
            high_close_prev = np.abs(high - close.shift(1))
            low_close_prev = np.abs(low - close.shift(1))

            # Create DataFrame for TR calculation, ensuring alignment
            tr_df = pd.DataFrame({'hl': high_low, 'hcp': high_close_prev, 'lcp': low_close_prev})
            # Calculate True Range (maximum of the three components)
            true_range = tr_df.max(axis=1, skipna=False) # skipna=False ensures NaN propagates if any component is NaN

            # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha = 1/period)
            # com = period - 1 gives alpha = 1 / (1 + com) = 1 / period
            atr = true_range.ewm(com=period - 1, adjust=False, min_periods=period).mean().iloc[-1]

            # Alternative: Standard EMA smoothing:
            # atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean().iloc[-1]

            # Check for NaN result and non-negative value
            return float(atr) if pd.notna(atr) and atr >= 0 else None
        except Exception as e:
            logger.error(f"[{func_name}] Error calculating ATR (period {period}): {e}", exc_info=False)
            return None

    # --- Trading Logic & Order Management Methods ---

    def calculate_order_size(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        signal_score: Optional[int] = None
    ) -> float:
        """
        Calculates the order size in the BASE currency.

        Factors considered:
        - Available quote currency balance.
        - Configured risk percentage (`order_size_percentage`).
        - Optional volatility adjustment (`volatility_multiplier`).
        - Optional signal strength adjustment (`strong_signal_adjustment_factor`).
        - Exchange minimum order amount and cost limits.
        - Exchange amount precision.

        Args:
            current_price: The current market price (used for conversion and cost estimation).
            indicators: Dictionary containing calculated indicators (e.g., 'volatility', 'atr').
            signal_score: The computed signal score for potential size adjustment.

        Returns:
            The calculated order size in BASE currency, rounded to exchange precision,
            or 0.0 if any checks fail (insufficient balance, below minimums, etc.).
        """
        func_name = "calculate_order_size"

        # --- 0. Pre-computation Checks & Setup ---
        if self.market_info is None:
            logger.error(f"[{func_name}] Market info not loaded. Cannot calculate order size.")
            return 0.0
        if current_price <= 1e-12: # Use a small epsilon
            logger.error(f"[{func_name}] Current price ({current_price}) is zero or negative. Cannot calculate order size.")
            return 0.0
        if not (0 < self.order_size_percentage <= 1.0): # Ensure percentage is valid
             logger.error(f"[{func_name}] Invalid order_size_percentage ({self.order_size_percentage}) in config. Must be > 0 and <= 1.")
             return 0.0

        try:
            quote_currency = self.market_info['quote']
            base_currency = self.market_info['base']
            limits = self.market_info.get('limits', {})
            precision = self.market_info.get('precision', {})
            # Get limits safely, use None if not present
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')
            # Get precision safely, use defaults if not present (already loaded in __init__)
            amount_decimals = self.amount_decimals
            price_decimals = self.price_decimals
            amount_tick_size = precision.get('amount') # Used for logging/debugging
        except KeyError as e:
             logger.error(f"[{func_name}] Missing critical market info key: {e} (Quote: {self.market_info.get('quote')}, Base: {self.market_info.get('base')}). Cannot calculate size.")
             return 0.0
        except Exception as e:
             logger.error(f"[{func_name}] Unexpected error accessing market/precision details: {e}. Cannot calculate size.", exc_info=True)
             return 0.0

        logger.debug(f"[{func_name}] Starting size calculation for {base_currency}/{quote_currency} at price {self.format_price(current_price)}.")

        # --- 1. Get Available Balance ---
        # fetch_balance returns float or None on error, and handles simulation internally
        available_balance = self.fetch_balance(currency_code=quote_currency)
        if available_balance is None:
            # fetch_balance already logs the reason for failure
            logger.error(f"[{func_name}] Failed to fetch available balance for {quote_currency}. Cannot calculate order size.")
            return 0.0
        if available_balance <= 1e-9: # Check against small epsilon
            logger.info(f"[{func_name}] Available balance ({self.format_price(available_balance)} {quote_currency}) is effectively zero. Cannot place order.")
            return 0.0
        logger.debug(f"[{func_name}] Available balance: {self.format_price(available_balance)} {quote_currency}")

        # --- 2. Calculate Base Order Size (Value in Quote Currency) ---
        target_quote_value = available_balance * self.order_size_percentage
        logger.debug(f"[{func_name}] Target Quote Value (Balance * Risk %): {self.format_price(target_quote_value)} {quote_currency} (Risk: {self.order_size_percentage*100:.2f}%)")

        # --- 3. Apply Adjustments (Volatility & Signal Strength) ---
        adjustment_factor = 1.0

        # 3a. Volatility Adjustment (Inverse relationship: higher vol -> smaller size)
        if self.volatility_multiplier > 0:
            volatility = indicators.get('volatility')
            if volatility is not None and volatility > 1e-9: # Ensure volatility is positive and non-zero
                # Simple inverse scaling, clamped to avoid extreme sizes
                # Example: multiplier=1, vol=0.01 -> factor = 1/(1+0.01) = ~0.99
                # Example: multiplier=1, vol=0.05 -> factor = 1/(1+0.05) = ~0.95
                vol_adj_factor = 1.0 / (1.0 + volatility * self.volatility_multiplier)
                # Clamp the factor to prevent excessively small or large adjustments
                vol_adj_factor = max(0.1, min(2.0, vol_adj_factor)) # Clamp between 0.1x and 2.0x
                adjustment_factor *= vol_adj_factor
                logger.info(f"[{func_name}] Applied Volatility ({volatility:.5f}) Adjustment Factor: {vol_adj_factor:.3f}. New Total Factor: {adjustment_factor:.3f}")
            elif volatility is None:
                 logger.debug(f"[{func_name}] Volatility adjustment skipped: Volatility indicator is None.")
            else: # Volatility is zero or negative
                 logger.debug(f"[{func_name}] Volatility adjustment skipped: Volatility is zero or negative ({volatility}).")
        else:
            logger.debug(f"[{func_name}] Volatility adjustment disabled (volatility_multiplier = {self.volatility_multiplier}).")

        # 3b. Signal Strength Adjustment (Direct relationship: stronger signal -> potentially larger size)
        if signal_score is not None and (self.strong_signal_adjustment_factor != 1.0 or self.weak_signal_adjustment_factor != 1.0):
             abs_score = abs(signal_score)
             sig_adj_factor = 1.0
             adj_type = "Normal"
             # Apply strong factor if score meets or exceeds the strong threshold
             if abs_score >= STRONG_SIGNAL_THRESHOLD_ABS:
                 sig_adj_factor = self.strong_signal_adjustment_factor
                 adj_type = "Strong"
             # Apply weak factor ONLY if enabled AND score is below entry but non-zero? (This logic seems unlikely based on typical use)
             # Generally, this function is called *after* deciding to enter (score >= ENTRY_SIGNAL_THRESHOLD_ABS)
             # Let's assume normal entry threshold uses 1.0x unless weak_signal_adj is meant for that? Clarify config intent if needed.
             # If weak_signal_adjustment_factor is intended for scores >= ENTRY_THRESHOLD but < STRONG_THRESHOLD, adjust logic here.
             # Current assumption: Strong uses strong factor, others (meeting entry threshold) use 1.0.
             # elif abs_score >= ENTRY_SIGNAL_THRESHOLD_ABS: # Score meets entry but not strong
             #     sig_adj_factor = self.weak_signal_adjustment_factor # Use weak factor for 'normal' signals? Check config intent.
             #     adj_type = "Weak/Normal"

             # Only apply if the factor is actually different from 1.0
             if abs(sig_adj_factor - 1.0) > 1e-9:
                 adjustment_factor *= sig_adj_factor
                 logger.info(f"[{func_name}] Applied Signal Score ({signal_score}, Type: {adj_type}) Adjustment Factor: {sig_adj_factor:.3f}. New Total Factor: {adjustment_factor:.3f}")
             else:
                  logger.debug(f"[{func_name}] Signal score ({signal_score}, Type: {adj_type}) factor is {sig_adj_factor:.3f}, no size adjustment applied.")
        else:
            logger.debug(f"[{func_name}] Signal score adjustment skipped (Score: {signal_score}, Factors: S={self.strong_signal_adjustment_factor}, W={self.weak_signal_adjustment_factor}).")

        # Clamp the final combined adjustment factor to reasonable bounds
        # e.g., minimum 0.05x, maximum 2.5x of original risk percentage
        final_adjustment_factor = max(0.05, min(2.5, adjustment_factor))
        if abs(final_adjustment_factor - adjustment_factor) > 1e-9:
            logger.debug(f"[{func_name}] Clamped total adjustment factor from {adjustment_factor:.3f} to {final_adjustment_factor:.3f}.")
        elif abs(final_adjustment_factor - 1.0) > 1e-9: # Log if adjustment happened
             logger.debug(f"[{func_name}] Final combined adjustment factor: {final_adjustment_factor:.3f}")

        final_quote_value = target_quote_value * final_adjustment_factor
        if final_quote_value <= 1e-9: # Check against small epsilon
             logger.warning(f"{Fore.YELLOW}[{func_name}] Calculated quote value is zero or negative ({final_quote_value}) after adjustments. Cannot place order.{Style.RESET_ALL}")
             return 0.0
        logger.debug(f"[{func_name}] Final adjusted quote value for order: {self.format_price(final_quote_value)} {quote_currency}")

        # --- 4. Convert Quote Value to Base Amount ---
        order_size_base_raw = final_quote_value / current_price
        logger.debug(f"[{func_name}] Raw base amount calculated: {self.format_amount(order_size_base_raw, self.amount_decimals + 4)} {base_currency}") # Log with extra precision

        # --- 5. Apply Exchange Precision AND Check Limits ---
        try:
            # Use ccxt's built-in amount_to_precision
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base_raw)
            if amount_precise_str is None:
                 logger.error(f"[{func_name}] Failed to apply amount precision (amount_to_precision returned None). Raw: {order_size_base_raw}")
                 return 0.0
            amount_precise = float(amount_precise_str)
            logger.debug(f"[{func_name}] Amount after precision ({amount_tick_size}): {self.format_amount(amount_precise)} {base_currency}")

            # Check 1: Amount after precision must be > 0
            if amount_precise <= 1e-12: # Use small epsilon
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Order size is zero or negative ({amount_precise}) after applying exchange amount precision ({amount_tick_size}). Raw: {order_size_base_raw}. Cannot place order.{Style.RESET_ALL}")
                 return 0.0

            # Check 2: Amount must be >= minimum amount limit (if specified)
            if min_amount is not None and amount_precise < float(min_amount):
                logger.warning(f"{Fore.YELLOW}[{func_name}] Calculated order size {self.format_amount(amount_precise)} {base_currency} is below exchange minimum amount {self.format_amount(min_amount)}. Cannot place order.{Style.RESET_ALL}")
                return 0.0

            # Check 3: Estimated cost must be >= minimum cost limit (if specified)
            if min_cost is not None:
                # Estimate cost using precise amount and precise price
                price_precise_str = self.exchange.price_to_precision(self.symbol, current_price)
                if price_precise_str is None:
                     logger.warning(f"[{func_name}] Failed to get precise price for cost check. Using raw price {current_price}.")
                     price_for_cost_check = current_price
                else: price_for_cost_check = float(price_precise_str)

                estimated_cost = amount_precise * price_for_cost_check
                if estimated_cost < float(min_cost):
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Estimated order cost {self.format_price(estimated_cost)} {quote_currency} is below exchange minimum cost {self.format_price(min_cost)}. Cannot place order.{Style.RESET_ALL}")
                    return 0.0
                else:
                     logger.debug(f"[{func_name}] Estimated cost {self.format_price(estimated_cost)} {quote_currency} meets minimum {self.format_price(min_cost)}.")
            else:
                 logger.debug(f"[{func_name}] Minimum cost limit not specified by exchange. Skipping cost check.")


            # --- Success ---
            logger.info(f"{Fore.CYAN}[{func_name}] Calculated final valid order size: {self.format_amount(amount_precise)} {base_currency}{Style.RESET_ALL}")
            return amount_precise

        except (ccxt.InvalidOrder, ccxt.ExchangeError) as e:
            # Errors specifically from precision/limit functions
            logger.error(f"{Fore.RED}[{func_name}] Error applying exchange precision/limits: {type(e).__name__} - {e}. Raw amount: {order_size_base_raw}.{Style.RESET_ALL}")
            return 0.0
        except (ValueError, TypeError) as e:
            # Errors from float conversion or comparison with limits
            logger.error(f"{Fore.RED}[{func_name}] Error processing limits/precision values: {type(e).__name__} - {e}. MinAmount={min_amount}, MinCost={min_cost}, AmountPrecise={amount_precise_str}.{Style.RESET_ALL}")
            return 0.0
        except Exception as e:
             # Catch-all for unexpected issues in this block
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error during size precision/limit checks: {e}{Style.RESET_ALL}", exc_info=True)
             return 0.0

    def _calculate_sl_tp_prices(
        self, entry_price: float, side: str,
        current_price: float, # Used for logging/context, maybe future relative calcs
        atr: Optional[float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculates Stop Loss (SL) and Take Profit (TP) prices based on configuration.

        Uses either ATR (`use_atr_sl_tp` = true) or fixed percentages.
        Applies exchange price precision and performs sanity checks (e.g., SL below entry for buy).

        Args:
            entry_price: The entry price of the position.
            side: The position side ('buy' or 'sell').
            current_price: The current market price (for context).
            atr: The current ATR value (used if `use_atr_sl_tp` is true).

        Returns:
            A tuple containing (stop_loss_price, take_profit_price).
            Values can be None if calculation fails, parameters are invalid,
            or sanity checks fail. Prices are rounded to exchange precision.
        """
        func_name = "_calculate_sl_tp_prices"
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None

        if self.market_info is None:
            logger.error(f"[{func_name}] Market info not available. Cannot calculate SL/TP.")
            return None, None
        if entry_price <= 1e-12:
            logger.error(f"[{func_name}] Invalid entry price ({entry_price}) for SL/TP calculation.")
            return None, None
        if side not in ['buy', 'sell']:
             logger.error(f"[{func_name}] Invalid side '{side}' for SL/TP calculation.")
             return None, None

        # --- Calculate Raw SL/TP Prices based on Config ---
        if self.use_atr_sl_tp:
            # --- ATR Based SL/TP ---
            if atr is None or atr <= 1e-12: # Use epsilon
                logger.warning(f"{Fore.YELLOW}[{func_name}] ATR SL/TP enabled, but ATR is invalid ({atr}). Cannot calculate SL/TP.{Style.RESET_ALL}")
                return None, None
            if not (self.atr_sl_multiplier > 0 and self.atr_tp_multiplier > 0):
                 logger.warning(f"{Fore.YELLOW}[{func_name}] ATR SL/TP enabled, but multipliers invalid (SL={self.atr_sl_multiplier}, TP={self.atr_tp_multiplier}). Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None

            logger.debug(f"[{func_name}] Calculating SL/TP using ATR={self.format_price(atr)}, SL Mult={self.atr_sl_multiplier}, TP Mult={self.atr_tp_multiplier}")
            sl_delta = atr * self.atr_sl_multiplier
            tp_delta = atr * self.atr_tp_multiplier
            stop_loss_price_raw = entry_price - sl_delta if side == "buy" else entry_price + sl_delta
            take_profit_price_raw = entry_price + tp_delta if side == "buy" else entry_price - tp_delta
        else:
            # --- Fixed Percentage Based SL/TP ---
            sl_pct = self.base_stop_loss_pct
            tp_pct = self.base_take_profit_pct
            # Check if at least one is configured and valid
            sl_valid = sl_pct is not None and 0 < sl_pct < 1
            tp_valid = tp_pct is not None and 0 < tp_pct < 1

            if not sl_valid and not tp_valid:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Fixed % SL/TP selected, but percentages invalid or missing (SL={sl_pct}, TP={tp_pct}). Cannot calculate SL/TP.{Style.RESET_ALL}")
                 return None, None # Neither SL nor TP can be calculated

            logger.debug(f"[{func_name}] Calculating SL/TP using Fixed %: SL={f'{sl_pct*100:.2f}%' if sl_valid else 'N/A'}, TP={f'{tp_pct*100:.2f}%' if tp_valid else 'N/A'}")
            if sl_valid:
                stop_loss_price_raw = entry_price * (1 - sl_pct) if side == "buy" else entry_price * (1 + sl_pct)
            if tp_valid:
                take_profit_price_raw = entry_price * (1 + tp_pct) if side == "buy" else entry_price * (1 - tp_pct)

        # --- Apply Precision and Perform Sanity Checks ---
        stop_loss_price_final: Optional[float] = None
        take_profit_price_final: Optional[float] = None
        price_tick = float(self.market_info['precision'].get('price', 1 / (10**self.price_decimals))) # Estimate tick if missing

        try:
            # Process SL
            if stop_loss_price_raw is not None:
                 if stop_loss_price_raw <= 1e-12: # Check before precision
                      logger.warning(f"[{func_name}] Raw SL price ({stop_loss_price_raw}) is zero/negative. SL set to None.")
                 else:
                    sl_precise_str = self.exchange.price_to_precision(self.symbol, stop_loss_price_raw)
                    if sl_precise_str is None: raise ValueError("price_to_precision returned None for SL")
                    sl_precise = float(sl_precise_str)

                    # Sanity Check: SL must be 'worse' than entry price
                    sl_valid_side = (side == 'buy' and sl_precise < entry_price - price_tick * 0.5) or \
                                    (side == 'sell' and sl_precise > entry_price + price_tick * 0.5)
                    if sl_valid_side:
                        stop_loss_price_final = sl_precise
                    else:
                        logger.warning(f"{Fore.YELLOW}[{func_name}] Calculated SL price {self.format_price(sl_precise)} is not valid relative to entry price {self.format_price(entry_price)} for {side} side (or too close). SL set to None.{Style.RESET_ALL}")

            # Process TP
            if take_profit_price_raw is not None:
                 if take_profit_price_raw <= 1e-12: # Check before precision
                      logger.warning(f"[{func_name}] Raw TP price ({take_profit_price_raw}) is zero/negative. TP set to None.")
                 else:
                    tp_precise_str = self.exchange.price_to_precision(self.symbol, take_profit_price_raw)
                    if tp_precise_str is None: raise ValueError("price_to_precision returned None for TP")
                    tp_precise = float(tp_precise_str)

                    # Sanity Check: TP must be 'better' than entry price
                    tp_valid_side = (side == 'buy' and tp_precise > entry_price + price_tick * 0.5) or \
                                    (side == 'sell' and tp_precise < entry_price - price_tick * 0.5)
                    if tp_valid_side:
                        take_profit_price_final = tp_precise
                    else:
                        logger.warning(f"{Fore.YELLOW}[{func_name}] Calculated TP price {self.format_price(tp_precise)} is not valid relative to entry price {self.format_price(entry_price)} for {side} side (or too close). TP set to None.{Style.RESET_ALL}")

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Error applying precision to SL/TP: {e}. Raw SL={stop_loss_price_raw}, TP={take_profit_price_raw}. Setting respective price to None.{Style.RESET_ALL}")
            # Decide if failure on one should nullify the other? Currently handled individually.
            # Example: if sl_precise failed, stop_loss_price_final remains None, tp might still be set.
            # Re-evaluating which failed:
            if 'SL' in str(e): stop_loss_price_final = None
            if 'TP' in str(e): take_profit_price_final = None
            # If error is generic, potentially nullify both? Safer but less flexible.
            # stop_loss_price_final, take_profit_price_final = None, None
        except Exception as e:
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error during SL/TP precision/sanity checks: {e}. Setting both SL/TP to None.{Style.RESET_ALL}", exc_info=True)
             return None, None

        # Final Cross-Check: SL vs TP (if both were successfully calculated)
        if stop_loss_price_final is not None and take_profit_price_final is not None:
             sl_tp_conflict = (side == "buy" and stop_loss_price_final >= take_profit_price_final) or \
                              (side == "sell" and stop_loss_price_final <= take_profit_price_final)
             if sl_tp_conflict:
                  logger.warning(f"{Fore.YELLOW}[{func_name}] Final SL {self.format_price(stop_loss_price_final)} conflicts with TP {self.format_price(take_profit_price_final)} for {side} side. Setting TP to None to prioritize SL.{Style.RESET_ALL}")
                  # Prioritize SL by default in case of conflict, nullify TP
                  take_profit_price_final = None

        logger.debug(f"[{func_name}] Final calculated SL={self.format_price(stop_loss_price_final)}, TP={self.format_price(take_profit_price_final)}")
        return stop_loss_price_final, take_profit_price_final

    def compute_trade_signal_score(self, price: float, indicators: Dict[str, Any], orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        """
        Computes a trade signal score based on multiple factors:
        - Order Book Imbalance
        - EMA Trend (Price vs EMA)
        - RSI Overbought/Oversold
        - MACD Cross / Position
        - Stochastic RSI Overbought/Oversold

        Returns:
            Tuple[int, List[str]]: The final integer score and a list of strings detailing the reasons.
            Positive score suggests BUY, Negative score suggests SELL.
        """
        func_name = "compute_trade_signal_score"
        score = 0.0
        reasons = []
        # --- Configurable Thresholds (could be moved to config.yaml) ---
        RSI_OS, RSI_OB = 35, 65 # RSI Oversold/Overbought levels
        STOCH_OS, STOCH_OB = 25, 75 # StochRSI Oversold/Overbought levels
        EMA_THRESH_MULT = 0.0002 # Price must be this % away from EMA to trigger signal (0.02%)
        # ---

        # 1. Order Book Imbalance Signal (Ask Volume / Bid Volume)
        imb_str = "N/A"
        imb_reason = f"{Fore.WHITE}[ 0.0] OB N/A{Style.RESET_ALL}"
        if orderbook_imbalance is not None:
            imb = orderbook_imbalance
            imb_str = f"{imb:.3f}" if imb != float('inf') else "Inf"
            if self.imbalance_threshold <= 0: # Config validation should catch this, but double check
                imb_reason = f"{Fore.YELLOW}[ 0.0] OB Invalid Threshold ({self.imbalance_threshold}){Style.RESET_ALL}"
            elif imb == float('inf'): # Infinite imbalance (zero bids) -> Strong sell signal
                score -= 1.0
                imb_reason = f"{Fore.RED}[-1.0] OB Sell (Imb: Inf){Style.RESET_ALL}"
            else:
                 # Buy signal if Ask/Bid ratio is significantly LOW (more bids)
                 imb_buy_thresh = 1.0 / self.imbalance_threshold # e.g., if threshold is 1.5, buy below 1/1.5 = 0.667
                 if imb < imb_buy_thresh:
                     score += 1.0
                     imb_reason = f"{Fore.GREEN}[+1.0] OB Buy (Imb < {imb_buy_thresh:.3f}){Style.RESET_ALL}"
                 # Sell signal if Ask/Bid ratio is significantly HIGH (more asks)
                 elif imb > self.imbalance_threshold:
                     score -= 1.0
                     imb_reason = f"{Fore.RED}[-1.0] OB Sell (Imb > {self.imbalance_threshold:.3f}){Style.RESET_ALL}"
                 else: # Imbalance is within neutral range
                     imb_reason = f"{Fore.WHITE}[ 0.0] OB Neutral{Style.RESET_ALL}"
        reasons.append(f"{imb_reason} (Val: {imb_str})")

        # 2. EMA Trend Signal
        ema = indicators.get('ema')
        ema_str = self.format_price(ema)
        ema_reason = f"{Fore.WHITE}[ 0.0] EMA N/A{Style.RESET_ALL}"
        if ema is not None and ema > 1e-9: # Ensure EMA is valid
            price_ema_upper_band = ema * (1 + EMA_THRESH_MULT)
            price_ema_lower_band = ema * (1 - EMA_THRESH_MULT)
            if price > price_ema_upper_band: # Price significantly above EMA -> Bullish
                score += 1.0
                ema_reason = f"{Fore.GREEN}[+1.0] Price > EMA{Style.RESET_ALL}"
            elif price < price_ema_lower_band: # Price significantly below EMA -> Bearish
                score -= 1.0
                ema_reason = f"{Fore.RED}[-1.0] Price < EMA{Style.RESET_ALL}"
            else: # Price close to EMA -> Neutral
                ema_reason = f"{Fore.WHITE}[ 0.0] Price ~ EMA{Style.RESET_ALL}"
        reasons.append(f"{ema_reason} (EMA: {ema_str}, Price: {self.format_price(price)})")

        # 3. RSI Momentum & Overbought/Oversold Signal
        rsi = indicators.get('rsi')
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        rsi_reason = f"{Fore.WHITE}[ 0.0] RSI N/A{Style.RESET_ALL}"
        if rsi is not None:
            if rsi < RSI_OS: # Oversold -> Potential Buy Signal
                score += 1.0
                rsi_reason = f"{Fore.GREEN}[+1.0] RSI Oversold (<{RSI_OS}){Style.RESET_ALL}"
            elif rsi > RSI_OB: # Overbought -> Potential Sell Signal
                score -= 1.0
                rsi_reason = f"{Fore.RED}[-1.0] RSI Overbought (>{RSI_OB}){Style.RESET_ALL}"
            else: # Neutral RSI zone
                rsi_reason = f"{Fore.WHITE}[ 0.0] RSI Neutral{Style.RESET_ALL}"
        reasons.append(f"{rsi_reason} (Val: {rsi_str})")

        # 4. MACD Momentum/Cross Signal
        macd_line, macd_signal = indicators.get('macd_line'), indicators.get('macd_signal')
        # Format MACD values with more precision if needed
        macd_str = f"L:{self.format_price(macd_line, self.price_decimals+2)}/S:{self.format_price(macd_signal, self.price_decimals+2)}" if macd_line is not None and macd_signal is not None else "N/A"
        macd_reason = f"{Fore.WHITE}[ 0.0] MACD N/A{Style.RESET_ALL}"
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal: # MACD line above signal -> Bullish momentum
                score += 1.0
                macd_reason = f"{Fore.GREEN}[+1.0] MACD Line > Signal{Style.RESET_ALL}"
            else: # MACD line below or equal to signal -> Bearish momentum
                score -= 1.0
                macd_reason = f"{Fore.RED}[-1.0] MACD Line <= Signal{Style.RESET_ALL}"
        reasons.append(f"{macd_reason} ({macd_str})")

        # 5. Stochastic RSI Overbought/Oversold Signal
        stoch_k, stoch_d = indicators.get('stoch_k'), indicators.get('stoch_d')
        stoch_str = f"K:{stoch_k:.1f}/D:{stoch_d:.1f}" if stoch_k is not None and stoch_d is not None else "N/A"
        stoch_reason = f"{Fore.WHITE}[ 0.0] StochRSI N/A{Style.RESET_ALL}"
        if stoch_k is not None and stoch_d is not None:
            # Require both K and D to be in OB/OS zones for stronger signal
            if stoch_k < STOCH_OS and stoch_d < STOCH_OS: # Both lines below OS level -> Buy Signal
                score += 1.0
                stoch_reason = f"{Fore.GREEN}[+1.0] StochRSI Oversold (<{STOCH_OS}){Style.RESET_ALL}"
            elif stoch_k > STOCH_OB and stoch_d > STOCH_OB: # Both lines above OB level -> Sell Signal
                score -= 1.0
                stoch_reason = f"{Fore.RED}[-1.0] StochRSI Overbought (>{STOCH_OB}){Style.RESET_ALL}"
            # Optional: Could add weaker signals if only K crosses, or if K crosses D within zones
            else: # StochRSI is in the neutral zone
                stoch_reason = f"{Fore.WHITE}[ 0.0] StochRSI Neutral{Style.RESET_ALL}"
        reasons.append(f"{stoch_reason} ({stoch_str})")

        # --- Final Score Calculation ---
        # Round the accumulated score to the nearest integer
        final_score = int(round(score))
        logger.debug(f"[{func_name}] Raw Signal Score: {score:.2f}, Final Integer Score: {final_score}")
        return final_score, reasons

    @retry_api_call(max_retries=2, initial_delay=2) # Allow retries for order placement
    def place_entry_order(
        self, side: str, order_size_base: float, confidence_level: int,
        order_type: str, current_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Places the entry order (market or limit) on the exchange.
        Includes Bybit V5 specific parameters for SL/TP if provided.
        Handles simulation mode by creating a simulated order dictionary.

        Args:
            side: 'buy' or 'sell'.
            order_size_base: The calculated and validated order size in base currency.
            confidence_level: The signal score associated with this entry.
            order_type: 'market' or 'limit'.
            current_price: The current market price (used for limit offset and logging).
            stop_loss_price: The calculated stop loss price (optional).
            take_profit_price: The calculated take profit price (optional).

        Returns:
            The order dictionary returned by ccxt (or a simulated one),
            or None if the order placement fails. Includes 'bot_custom_info'.
        """
        func_name = "place_entry_order"
        if self.market_info is None: logger.error(f"[{func_name}] Market info missing."); return None
        if order_size_base <= 1e-12: logger.error(f"[{func_name}] Invalid order size {order_size_base}."); return None
        if current_price <= 1e-12: logger.error(f"[{func_name}] Invalid current price {current_price}."); return None
        if side not in ['buy', 'sell']: logger.error(f"[{func_name}] Invalid side '{side}'."); return None
        if order_type not in ['market', 'limit']: logger.error(f"[{func_name}] Invalid order_type '{order_type}'."); return None


        base_currency = self.market_info['base']
        quote_currency = self.market_info['quote']
        params: Dict[str, Any] = {}
        limit_price: Optional[float] = None # Only set for limit orders

        try:
            # Amount should already be precise from calculate_order_size
            amount_precise = order_size_base
            logger.debug(f"[{func_name}] Using pre-validated precise amount: {self.format_amount(amount_precise)} {base_currency}")

            # Calculate Limit Price if order_type is 'limit'
            if order_type == "limit":
                offset = self.limit_order_entry_offset_pct_buy if side == 'buy' else self.limit_order_entry_offset_pct_sell
                if offset < 0: logger.warning(f"[{func_name}] Limit order offset is negative ({offset}), using 0."); offset = 0.0

                # Buy limit below current price, Sell limit above current price
                price_factor = (1.0 - offset) if side == 'buy' else (1.0 + offset)
                limit_price_raw = current_price * price_factor

                if limit_price_raw <= 1e-12: # Check for non-positive price
                    raise ValueError(f"Calculated limit price is zero or negative ({limit_price_raw})")

                # Apply exchange price precision to the calculated limit price
                limit_price_str = self.exchange.price_to_precision(self.symbol, limit_price_raw)
                if limit_price_str is None: raise ValueError("price_to_precision returned None for limit price")
                limit_price = float(limit_price_str)
                logger.debug(f"[{func_name}] Calculated precise limit price: {self.format_price(limit_price)}")

                # Sanity check limit price placement relative to current price
                price_tick = float(self.market_info['precision'].get('price', 1e-8))
                if (side == 'buy' and limit_price >= current_price - price_tick * 0.5) or \
                   (side == 'sell' and limit_price <= current_price + price_tick * 0.5):
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Limit price {self.format_price(limit_price)} is not favorable vs current {self.format_price(current_price)} for {side} order (or too close). Check offset/slippage. Proceeding anyway...{Style.RESET_ALL}")

            # --- Prepare Common Order Parameters ---
            # Add Bybit V5 SL/TP Params if they exist and are valid
            # CCXT generally expects these as strings for Bybit V5
            if stop_loss_price is not None and stop_loss_price > 0:
                params['stopLoss'] = self.format_price(stop_loss_price) # Format to string with correct decimals
                if self.sl_trigger_by: params['slTriggerBy'] = self.sl_trigger_by
            if take_profit_price is not None and take_profit_price > 0:
                params['takeProfit'] = self.format_price(take_profit_price)
                if self.tp_trigger_by: params['tpTriggerBy'] = self.tp_trigger_by

            if params.get('stopLoss') or params.get('takeProfit'):
                 logger.debug(f"[{func_name}] Adding SL/TP parameters to order: {params}")

            # Add Bybit V5 category parameter hint
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                market_type = self.market_info.get('type', 'swap')
                is_linear = self.market_info.get('linear', True) # Default to linear if unsure
                if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                elif market_type == 'spot': params['category'] = 'spot'
                elif market_type == 'option': params['category'] = 'option'
                if 'category' in params: logger.debug(f"[{func_name}] Added Bybit category hint: {params['category']}")

        except (ccxt.ExchangeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Error preparing order values/parameters: {e}{Style.RESET_ALL}", exc_info=True)
            return None
        except Exception as e:
             logger.error(f"{Fore.RED}[{func_name}] Unexpected error preparing order: {e}{Style.RESET_ALL}", exc_info=True)
             return None

        # --- Log Order Details Before Placing ---
        log_color = Fore.GREEN if side == 'buy' else Fore.RED
        action_desc = f"{order_type.upper()} {side.upper()} ENTRY"
        sl_info = f"SL={params.get('stopLoss', 'N/A')}" + (f" (Trig: {params['slTriggerBy']})" if 'slTriggerBy' in params else "")
        tp_info = f"TP={params.get('takeProfit', 'N/A')}" + (f" (Trig: {params['tpTriggerBy']})" if 'tpTriggerBy' in params else "")
        limit_info = f"at Limit {self.format_price(limit_price)}" if limit_price else f"at Market (~{self.format_price(current_price)})"
        # Estimate value using limit price if available, else current price
        est_value_price = limit_price if limit_price else current_price
        estimated_value = amount_precise * est_value_price

        log_entry_details = (
            f"ID: ???, Size: {self.format_amount(amount_precise)} {base_currency}, Price: {limit_info}, "
            f"EstVal: {self.format_price(estimated_value)} {quote_currency}, Conf: {confidence_level}, {sl_info}, {tp_info}"
        )

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_id = f"sim_entry_{int(time.time() * 1000)}_{side[:1]}"
            # Assume limit orders stay 'open', market orders fill 'closed' instantly
            sim_status = 'open' if order_type == 'limit' else 'closed'
            sim_entry_price = limit_price if order_type == "limit" else current_price # Simulate fill at market/limit
            sim_filled = amount_precise if sim_status == 'closed' else 0.0
            sim_avg = sim_entry_price if sim_status == 'closed' else None
            sim_remaining = amount_precise - sim_filled
            sim_ts = int(time.time() * 1000)

            simulated_order = {
                "id": sim_id, "timestamp": sim_ts, "datetime": pd.to_datetime(sim_ts, unit='ms', utc=True).isoformat(),
                "symbol": self.symbol, "type": order_type, "side": side, "status": sim_status,
                "price": limit_price, # The limit price set for the order
                "amount": amount_precise, # Requested amount
                "filled": sim_filled, # Amount filled
                "remaining": sim_remaining, # Amount remaining
                "average": sim_avg, # Average fill price
                "cost": sim_filled * sim_avg if sim_avg else 0.0, # Cost of filled portion
                # Include SL/TP prices as they were calculated/intended
                "stopLossPrice": stop_loss_price,
                "takeProfitPrice": take_profit_price,
                # Mimic info structure loosely
                "info": {"simulated": True, "orderId": sim_id, **params}, # Include params that would have been sent
                # Add custom info for bot's internal use
                "bot_custom_info": {"confidence": confidence_level, "initial_base_size": order_size_base}
            }
            logger.info(f"{log_color}[SIMULATION] Placing {action_desc}: {log_entry_details.replace('ID: ???', f'ID: {sim_id}')}{Style.RESET_ALL}")
            return simulated_order

        # --- Live Trading Mode ---
        else:
            logger.info(f"{log_color}Attempting to place LIVE {action_desc} order...")
            # Log details again, maybe slightly different format for live
            log_live_details = (f" -> Symbol: {self.symbol}, Side: {side}, Type: {order_type}\n"
                                f" -> Size: {self.format_amount(amount_precise)} {base_currency}\n"
                                f" -> Price: {limit_info}\n"
                                f" -> Est. Value: ~{self.format_price(estimated_value)} {quote_currency}\n"
                                f" -> Confidence: {confidence_level}\n"
                                f" -> Parameters: {params}")
            logger.info(log_live_details)

            order: Optional[Dict[str, Any]] = None
            try:
                # Use the appropriate CCXT method based on order type
                if order_type == "market":
                    order = self.exchange.create_market_order(self.symbol, side, amount_precise, params=params)
                elif order_type == "limit":
                    if limit_price is None: # Should have been caught earlier, but safety check
                         raise ValueError("Limit price is None for create_limit_order call.")
                    order = self.exchange.create_limit_order(self.symbol, side, amount_precise, limit_price, params=params)
                # else: # Should be caught by initial validation
                #     raise ValueError(f"Unsupported live order type '{order_type}' encountered.")

                # Process the response from the exchange
                if order:
                    oid = order.get('id', 'N/A')
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    ofilled = order.get('filled', 0.0)
                    oavg = order.get('average')
                    # Check if SL/TP was acknowledged in the response 'info' (exchange specific)
                    info_sl = order.get('info', {}).get('stopLoss', 'N/A') # Bybit V5 field name
                    info_tp = order.get('info', {}).get('takeProfit', 'N/A') # Bybit V5 field name

                    logger.info(
                        f"{log_color}---> LIVE {action_desc} Order Placed: ID: {oid}, Status: {ostatus}, "
                        f"Filled: {self.format_amount(ofilled)}, AvgPrice: {self.format_price(oavg)}, "
                        f"SL Sent/Confirmed: {params.get('stopLoss', 'N/A')}/{info_sl}, "
                        f"TP Sent/Confirmed: {params.get('takeProfit', 'N/A')}/{info_tp}{Style.RESET_ALL}"
                    )
                    # Add custom info to the returned order dict
                    order['bot_custom_info'] = {"confidence": confidence_level, "initial_base_size": order_size_base}
                    return order
                else:
                    # This case might happen if the API call returns None without raising an exception (e.g., due to retry decorator failure)
                    logger.error(f"{Fore.RED}LIVE {action_desc} order placement API call returned None unexpectedly. Check exchange status/logs.{Style.RESET_ALL}")
                    return None

            # Handle specific, potentially informative exceptions first
            except ccxt.InsufficientFunds as e:
                logger.error(f"{Fore.RED}LIVE {action_desc} Failed: Insufficient Funds. Error: {e}{Style.RESET_ALL}")
                # Maybe trigger a balance fetch here to log current state?
                self.fetch_balance(quote_currency)
                return None
            except ccxt.InvalidOrder as e:
                 logger.error(f"{Fore.RED}LIVE {action_desc} Failed: Invalid Order (check parameters, size, price, limits). Error: {e}{Style.RESET_ALL}")
                 # Log parameters sent for debugging
                 logger.error(f"Order Details: side={side}, amount={amount_precise}, price={limit_price}, params={params}")
                 return None
            # Handle general exchange errors
            except ccxt.ExchangeError as e:
                 logger.error(f"{Fore.RED}LIVE {action_desc} Failed (ExchangeError): {e}{Style.RESET_ALL}")
                 return None
            # Handle unexpected Python errors
            except Exception as e:
                logger.error(f"{Fore.RED}LIVE {action_desc} Failed (Unexpected Python Error): {e}{Style.RESET_ALL}", exc_info=True)
                return None

    @retry_api_call(max_retries=1) # Retry once if cancel fails initially
    def cancel_order_by_id(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Cancels a specific order by its ID.
        Returns True if cancellation was successful OR if the order was already
        gone (not found, closed, cancelled). Returns False on persistent failure.
        Handles simulation mode.
        """
        func_name = "cancel_order_by_id"
        if not order_id or not isinstance(order_id, str):
            logger.warning(f"[{func_name}] Invalid order_id provided: {order_id}. Cannot cancel.")
            return False

        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}[{func_name}] Attempting to cancel order {order_id} ({target_symbol})...{Style.RESET_ALL}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            # In simulation, assume cancellation always works if the order exists in our state
            # Find the position/order
            pos_index = next((i for i, p in enumerate(self.open_positions) if p.get('id') == order_id), -1)
            if pos_index != -1:
                 original_status = self.open_positions[pos_index].get('status')
                 # Mark as cancelled in state (caller should save state if needed)
                 self.open_positions[pos_index]['status'] = STATUS_CANCELED
                 self.open_positions[pos_index]['last_update_time'] = time.time()
                 logger.info(f"[SIMULATION][{func_name}] Marked order {order_id} as CANCELLED in state (was {original_status}).")
                 return True
            else:
                 logger.warning(f"[SIMULATION][{func_name}] Order {order_id} not found in open_positions state. Assuming already gone.")
                 return True # Treat as success if not found

        # --- Live/Testnet Mode ---
        else:
            try:
                params = {}
                # Add Bybit V5 category param hint
                if self.exchange_id.lower() == 'bybit' and self.market_info:
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    is_linear = self.exchange.market(target_symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'
                    elif market_type == 'option': params['category'] = 'option'

                # Use the unified cancel_order method
                response = self.exchange.cancel_order(order_id, target_symbol, params=params)
                # Response format varies; log snippet for confirmation
                logger.info(f"{Fore.GREEN}---> Cancellation successful for order {order_id}. Response: {str(response)[:150]}...{Style.RESET_ALL}")
                return True

            except ccxt.OrderNotFound as e:
                # This is expected if the order was filled, cancelled manually, or never existed
                logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} not found during cancellation (already closed/cancelled/invalid?). Treating as success. Error: {e}{Style.RESET_ALL}")
                return True # Return True because the desired state (order gone) is achieved
            except ccxt.NetworkError as e:
                # Let the retry decorator handle transient network issues
                logger.warning(f"{Fore.YELLOW}[{func_name}] Network error cancelling {order_id}: {e}. Retrying if possible...{Style.RESET_ALL}")
                raise e # Re-raise to trigger retry
            except ccxt.ExchangeError as e:
                 # Check if the error message indicates the order is already in a final state
                 err_str = str(e).lower()
                 final_state_phrases = ["order has been filled", "order is finished", "already closed", "already cancel", "order status error", "cannot be cancel"]
                 if any(p in err_str for p in final_state_phrases):
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Cannot cancel order {order_id}: Already in a final state. Treating as success. Error: {e}{Style.RESET_ALL}")
                      return True # Return True as the order is effectively gone/finalized
                 else:
                      # Log other exchange errors as failures
                      logger.error(f"{Fore.RED}[{func_name}] Exchange error cancelling order {order_id}: {e}{Style.RESET_ALL}")
                      return False # Return False indicating cancellation failed
            except Exception as e:
                # Catch unexpected Python errors
                logger.error(f"{Fore.RED}[{func_name}] Unexpected error cancelling order {order_id}: {e}{Style.RESET_ALL}", exc_info=True)
                return False # Return False for unexpected failures

    @retry_api_call() # Retry the entire operation if needed
    def cancel_all_symbol_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancels all open orders for the specified symbol.
        Uses `cancelAllOrders` if available, otherwise falls back to fetching
        open orders and cancelling them individually. Handles simulation.

        Args:
            symbol: The market symbol (e.g., 'BTC/USDT:USDT'). Uses self.symbol if None.

        Returns:
            Number of orders successfully cancelled (or marked cancelled in sim),
            or -1 if `cancelAllOrders` was used (count often unreliable),
            or 0 if no orders were found or cancellation failed.
        """
        func_name = "cancel_all_symbol_orders"
        target_symbol = symbol or self.symbol
        logger.info(f"{Fore.YELLOW}[{func_name}] Attempting to cancel all OPEN orders for {target_symbol}...{Style.RESET_ALL}")
        cancelled_count = 0

        # --- Simulation Mode ---
        if self.simulation_mode:
             logger.info(f"[SIMULATION][{func_name}] Simulating cancel all open orders for {target_symbol}.")
             initial_open_count = 0
             indices_to_cancel = []
             for i, pos in enumerate(self.open_positions):
                 # Cancel orders that are PENDING for the target symbol
                 if pos.get('symbol') == target_symbol and pos.get('status') == STATUS_PENDING_ENTRY:
                     initial_open_count += 1
                     indices_to_cancel.append(i)

             if not indices_to_cancel:
                 logger.info(f"[SIMULATION][{func_name}] No pending orders found in state for {target_symbol} to cancel.")
                 return 0

             # Mark them as cancelled in state
             for i in indices_to_cancel:
                  self.open_positions[i]['status'] = STATUS_CANCELED
                  self.open_positions[i]['last_update_time'] = time.time()
                  cancelled_count += 1

             logger.info(f"[SIMULATION][{func_name}] Marked {cancelled_count}/{initial_open_count} pending orders as CANCELLED in state for {target_symbol}.")
             # Caller should save state after this if needed
             return cancelled_count

        # --- Live/Testnet Mode ---
        else:
            try:
                params = {}
                # Add Bybit V5 category param hint
                if self.exchange_id.lower() == 'bybit' and self.market_info:
                    market_type = self.exchange.market(target_symbol).get('type', 'swap')
                    is_linear = self.exchange.market(target_symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'
                    elif market_type == 'option': params['category'] = 'option'
                    # Bybit specific filter (check API docs if needed)
                    # params['orderFilter'] = 'Order' # To cancel only regular orders?
                    # params['stopOrderType'] = 'Stop' # Or 'StopLoss', 'TakeProfit'? Not usually needed for cancelAll

                # Strategy 1: Use unified cancelAllOrders if supported (often faster)
                if self.exchange.has.get('cancelAllOrders'):
                    logger.debug(f"[{func_name}] Using unified 'cancelAllOrders' capability for {target_symbol}...")
                    response = self.exchange.cancel_all_orders(target_symbol, params=params)
                    # Response format varies greatly, may not indicate count reliably
                    logger.info(f"{Fore.GREEN}[{func_name}] 'cancelAllOrders' executed for {target_symbol}. Response snippet: {str(response)[:150]}...")
                    # Return -1 to indicate the action was attempted but count is unknown/unreliable
                    return -1

                # Strategy 2: Fallback to fetchOpenOrders + individual cancel (more reliable count)
                elif self.exchange.has.get('fetchOpenOrders'):
                    logger.warning(f"[{func_name}] 'cancelAllOrders' unavailable/unsupported. Falling back to fetchOpenOrders + individual cancel for {target_symbol}...")
                    # Fetch only open orders for the specific symbol
                    open_orders = self.exchange.fetch_open_orders(target_symbol, params=params) # Use same params

                    if not open_orders:
                        logger.info(f"[{func_name}] No open orders found via fetchOpenOrders for {target_symbol}.")
                        return 0

                    logger.warning(f"{Fore.YELLOW}[{func_name}] Found {len(open_orders)} open order(s) via fetch. Attempting to cancel individually...{Style.RESET_ALL}")
                    success_count = 0
                    failure_count = 0
                    for order in open_orders:
                        order_id = order.get('id')
                        if not order_id:
                            logger.warning(f"[{func_name}] Skipping order with missing ID in fetchOpenOrders result: {order}")
                            continue
                        # Use the robust cancel_order_by_id which handles "not found" as success
                        if self.cancel_order_by_id(order_id, target_symbol):
                            success_count += 1
                            # Add a small delay to avoid potential rate limits if cancelling many orders
                            time.sleep(max(0.1, self.exchange.rateLimit / 2000)) # Half rate limit delay? Be conservative.
                        else:
                            failure_count += 1
                            logger.error(f"[{func_name}] Failed to cancel order {order_id} during bulk cancellation attempt.")
                            # Optionally add a slightly longer delay after a failure
                            time.sleep(0.5)

                    final_msg = f"[{func_name}] Individual cancellation process finished for {target_symbol}. "
                    final_msg += f"Succeeded/Gone: {success_count}, Failed: {failure_count} (Total Found: {len(open_orders)})"
                    log_level = logging.INFO if failure_count == 0 else logging.WARNING
                    logger.log(log_level, final_msg)
                    return success_count # Return number successfully cancelled/gone

                else:
                     # This should ideally not happen if exchange has basic trading capabilities
                     logger.error(f"{Fore.RED}[{func_name}] Exchange {self.exchange_id} supports neither 'cancelAllOrders' nor 'fetchOpenOrders'. Cannot cancel all orders automatically.{Style.RESET_ALL}")
                     return 0 # Indicate no action could be taken

            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Error during bulk order cancellation for {target_symbol}: {e}{Style.RESET_ALL}", exc_info=True)
                # Return the count successful before the error, or 0 if error was early
                return cancelled_count if 'success_count' not in locals() else success_count

    # --- Position Management & State Update Logic ---

    def _check_pending_entries(self, indicators: Dict) -> None:
        """
        Checks the status of orders marked as STATUS_PENDING_ENTRY.
        Updates the position state to STATUS_ACTIVE if filled, or removes
        if cancelled/rejected/expired. Recalculates SL/TP based on fill price.
        """
        func_name = "_check_pending_entries"
        # Create a list of IDs to check to avoid modifying list while iterating
        pending_ids = [p['id'] for p in self.open_positions if p.get('status') == STATUS_PENDING_ENTRY]

        if not pending_ids: return # Nothing to check

        logger.debug(f"[{func_name}] Checking status of {len(pending_ids)} pending entry order(s): {pending_ids}")
        needs_state_save = False
        positions_to_remove_ids = set() # IDs of positions confirmed closed/failed
        positions_to_update_data = {} # {order_id: updated_position_dict} for activated orders

        # Use current price for SL/TP recalc if needed, fetch only once
        current_price_for_check: Optional[float] = None

        for entry_order_id in pending_ids:
            # Find the corresponding position dictionary in the main list
            position = next((p for p in self.open_positions if p.get('id') == entry_order_id), None)
            if position is None:
                logger.warning(f"[{func_name}] Could not find position data for pending ID {entry_order_id} in state list. Inconsistency?")
                continue # Skip to next ID

            pos_symbol = position.get('symbol', self.symbol)

            # Fetch the latest status from the exchange
            order_info = self.fetch_order_status(entry_order_id, symbol=pos_symbol)

            # --- Process Order Status ---

            # Case 1: Order Not Found / Vanished (Treat as failed/cancelled externally)
            if order_info is None:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} not found via fetch. Assuming externally cancelled or invalid. Removing from state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)
                continue

            order_status = order_info.get('status') # e.g., 'open', 'closed', 'canceled', 'rejected', 'expired'
            filled_amount = float(order_info.get('filled', 0.0))
            avg_fill_price = order_info.get('average') # Average fill price from ccxt

            # Case 2: Order Fully Filled (ccxt status 'closed')
            if order_status == 'closed' and filled_amount > 1e-12: # Use epsilon
                # Validate fill price
                if avg_fill_price is None or float(avg_fill_price) <= 1e-12:
                    logger.error(f"{Fore.RED}[{func_name}] Pending order {entry_order_id} reported 'closed' but has invalid fill price ({avg_fill_price}). Cannot activate. Removing from state.{Style.RESET_ALL}")
                    positions_to_remove_ids.add(entry_order_id)
                    continue

                entry_price = float(avg_fill_price)
                orig_size = position.get('original_size') # Requested size
                if orig_size is not None and abs(filled_amount - orig_size) > max(orig_size * 0.01, 1e-9): # Allow 1% tolerance or epsilon
                     logger.warning(f"{Fore.YELLOW}[{func_name}] Filled amount {self.format_amount(filled_amount)} differs significantly from requested {self.format_amount(orig_size)} for order {entry_order_id}. Using actual filled amount.{Style.RESET_ALL}")

                logger.info(f"{Fore.GREEN}---> Pending entry order {entry_order_id} FILLED! Side: {position['side']}, Amount: {self.format_amount(filled_amount)}, Avg Price: {self.format_price(entry_price)}{Style.RESET_ALL}")

                # Prepare updated position data
                updated_pos = position.copy()
                updated_pos['status'] = STATUS_ACTIVE
                updated_pos['size'] = filled_amount # Use the actual filled amount
                updated_pos['entry_price'] = entry_price
                # Try to get accurate fill timestamp, fallback to current time
                fill_time_ms = order_info.get('lastTradeTimestamp') or order_info.get('timestamp') # Bybit might have lastTradeTimestamp
                updated_pos['entry_time'] = fill_time_ms / 1000.0 if fill_time_ms else time.time()
                updated_pos['last_update_time'] = time.time()

                # Recalculate internal SL/TP based on actual fill price (important!)
                if current_price_for_check is None:
                    current_price_for_check = self.fetch_market_price() # Fetch only if needed

                if current_price_for_check:
                    atr_value = indicators.get('atr') # Get current ATR
                    sl_price, tp_price = self._calculate_sl_tp_prices(
                        entry_price=updated_pos['entry_price'], # Use actual fill price
                        side=updated_pos['side'],
                        current_price=current_price_for_check, # Pass current for context
                        atr=atr_value
                    )
                    # Update the SL/TP in our state even if they were sent with the order
                    # This ensures our internal state reflects calculated values based on actual fill
                    updated_pos['stop_loss_price'] = sl_price
                    updated_pos['take_profit_price'] = tp_price
                    logger.info(f"[{func_name}] Stored internal SL={self.format_price(sl_price)}, TP={self.format_price(tp_price)} for activated pos {entry_order_id} based on fill price {self.format_price(entry_price)}.")
                else:
                    logger.warning(f"[{func_name}] Could not fetch current price for SL/TP recalculation after fill for {entry_order_id}. SL/TP in state might be based on pre-entry price.")

                # Store the fully updated dictionary for this position
                positions_to_update_data[entry_order_id] = updated_pos

            # Case 3: Order Failed (canceled, rejected, expired)
            elif order_status in ['canceled', 'rejected', 'expired']:
                # Try to get a reason from the 'info' field (exchange specific)
                reject_reason = order_info.get('info', {}).get('rejectReason', 'N/A') # Bybit example
                cancel_type = order_info.get('info', {}).get('cancelType', 'N/A') # Bybit example
                reason_detail = f"RejectReason: {reject_reason}" if reject_reason != 'N/A' else f"CancelType: {cancel_type}" if cancel_type != 'N/A' else f"Status: {order_status}"
                logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} failed ({reason_detail}). Removing from state.{Style.RESET_ALL}")
                positions_to_remove_ids.add(entry_order_id)

            # Case 4: Order Still Open or Partially Filled (ccxt status 'open')
            elif order_status == 'open':
                 remaining = order_info.get('remaining', 'N/A')
                 logger.debug(f"[{func_name}] Pending order {entry_order_id} still 'open'. Filled: {self.format_amount(filled_amount)}, Remaining: {self.format_amount(remaining)}. Waiting...")
                 # Optional: Update last_update_time?
                 # position['last_update_time'] = time.time() # Update timestamp directly? Risky if loop fails before save
                 # needs_state_save = True # Need to save if timestamp updated

            # Case 5: Closed but zero filled (likely cancelled just before any fill)
            elif order_status == 'closed' and filled_amount <= 1e-12:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Pending order {entry_order_id} reported 'closed' but has zero filled amount. Assuming cancelled pre-fill. Removing from state.{Style.RESET_ALL}")
                 positions_to_remove_ids.add(entry_order_id)

            # Case 6: Unexpected Status
            else:
                 logger.warning(f"[{func_name}] Pending order {entry_order_id} has unexpected status: '{order_status}'. Filled: {self.format_amount(filled_amount)}. Leaving as pending for now.")
                 # Consider adding logic to handle specific unexpected statuses if they occur

        # --- Apply State Updates Atomically After Checking All Pending Orders ---
        if positions_to_remove_ids or positions_to_update_data:
            new_positions_list = []
            removed_count = 0
            updated_count = 0
            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    continue # Don't include removed positions in the new list
                if pos_id in positions_to_update_data:
                    # Replace the old position dict with the updated one
                    new_positions_list.append(positions_to_update_data[pos_id])
                    updated_count += 1
                else:
                    # Keep positions that were not modified or removed
                    new_positions_list.append(pos)

            if removed_count > 0 or updated_count > 0:
                self.open_positions = new_positions_list
                logger.debug(f"[{func_name}] Pending checks complete. Activated: {updated_count}, Removed: {removed_count}. New total positions: {len(self.open_positions)}")
                self._save_state() # Save the updated state

    def _manage_active_positions(self, current_price: float, indicators: Dict) -> None:
        """
        Manages positions marked as STATUS_ACTIVE.
        - Checks for external closure (SL/TP hit, manual close).
        - Implements Time-Based Exit.
        - Updates Trailing Stop Loss (TSL) if enabled [EXPERIMENTAL].
        - Logs PnL for closed positions.
        - Updates position state.
        """
        func_name = "_manage_active_positions"
        # Get a snapshot of active position IDs to iterate over
        active_ids = [p['id'] for p in self.open_positions if p.get('status') == STATUS_ACTIVE]

        if not active_ids: return # No active positions to manage

        logger.debug(f"[{func_name}] Managing {len(active_ids)} active position(s) against current price: {self.format_price(current_price)}...")

        positions_to_remove_ids = set() # IDs of positions confirmed closed
        positions_to_update_state = {} # {pos_id: {key: value}} for TSL or other updates

        for pos_id in active_ids:
            # Find the current position data from the main list
            position = next((p for p in self.open_positions if p.get('id') == pos_id and p.get('status') == STATUS_ACTIVE), None)
            # Check if position still exists and is active (could have been closed in a previous iteration step)
            if position is None:
                logger.debug(f"[{func_name}] Position {pos_id} no longer active or found. Skipping management.")
                continue

            # --- Get Essential Position Details ---
            symbol = position.get('symbol', self.symbol)
            side = position.get('side')
            entry_price = position.get('entry_price')
            position_size = position.get('size') # Current size (should be > 0 for active)
            entry_time = position.get('entry_time')

            # Basic validation for critical data
            if not all([symbol, side, entry_price, position_size, entry_time]):
                 logger.error(f"[{func_name}] Active position {pos_id} is missing essential data (Symbol, Side, EntryPrice, Size, EntryTime). Skipping management. Data: {position}")
                 positions_to_remove_ids.add(pos_id) # Mark for removal due to bad state
                 positions_to_update_state[pos_id] = {'status': STATUS_UNKNOWN, 'last_update_time': time.time()} # Mark as unknown
                 continue
            if position_size <= 1e-12: # Check size again
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Active position {pos_id} has size ~0 ({position_size}). Assuming closed. Removing from active state.{Style.RESET_ALL}")
                 positions_to_remove_ids.add(pos_id)
                 positions_to_update_state[pos_id] = {'status': STATUS_CLOSED_EXT, 'exit_reason': 'Zero Size Detected', 'last_update_time': time.time()}
                 continue

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None # Price at which closure occurred/detected

            # --- 1. Check for External Closure (SL/TP/Manual/Liquidation/ADL) ---
            # Strategy: Use fetch_positions (more reliable for Bybit V5) if available, fallback to fetch_order_status.
            position_closed_externally = False
            if self.exchange.has.get('fetchPositions'):
                try:
                    # Fetch positions for the specific symbol
                    # Note: fetchPositions might return multiple positions if hedging is enabled or for different contract types
                    fetched_positions = self.exchange.fetch_positions([symbol])
                    # Find the position matching our side (assuming no hedging or only one position per side)
                    matching_position = next((p for p in fetched_positions if p.get('side') == side and float(p.get('contracts', 0.0)) > 1e-12), None)

                    if matching_position:
                        # Position exists on the exchange, check if size matches our state
                        exchange_size = float(matching_position.get('contracts', 0.0))
                        # Use a small tolerance for size comparison
                        size_tolerance = max(position_size * 0.001, 1e-9)
                        if abs(exchange_size - position_size) > size_tolerance:
                             logger.warning(f"{Fore.YELLOW}[{func_name}] Size mismatch for {pos_id}. State: {self.format_amount(position_size)}, Exchange: {self.format_amount(exchange_size)}. External change likely occurred.{Style.RESET_ALL}")
                             # Partial closure? Could update size, but safer to mark as potentially closed/external change.
                             position_closed_externally = True
                             exit_reason = f"External Size Change (State: {position_size}, Exch: {exchange_size})"
                             exit_price = current_price # Best guess exit price
                        else:
                             # Position found on exchange with matching size, likely still active
                             logger.debug(f"[{func_name}] Position {pos_id} confirmed active via fetchPositions (Size: {exchange_size}).")
                    else:
                        # Position matching our symbol/side NOT found in fetchPositions result
                        logger.info(f"{Fore.YELLOW}[{func_name}] Position {pos_id} ({symbol}/{side}) not found via fetchPositions. Assuming closed externally.{Style.RESET_ALL}")
                        position_closed_externally = True
                        exit_reason = "Not Found via fetchPositions"
                        exit_price = current_price # Best guess exit price

                except Exception as e:
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Error using fetchPositions to check {pos_id}: {e}. Falling back to fetch_order_status.{Style.RESET_ALL}")
                    # Fallback to fetch_order_status check below

            # Fallback/Alternative Check: Use fetch_order_status of the *entry* order ID
            # Less reliable for SL/TP parameter closures but better than nothing.
            if not position_closed_externally and not self.exchange.has.get('fetchPositions'):
                order_info = self.fetch_order_status(pos_id, symbol=symbol)
                if order_info is None:
                    # Order not found could mean closed, but also just purged from history. Less reliable.
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Entry order {pos_id} not found via fetch_order_status. Cannot confirm active status this way. Assuming potentially closed.{Style.RESET_ALL}")
                    # Let's not immediately close based on this, but proceed to other checks.
                    # Could add logic: if fetchPositions unavailable AND fetch_order fails, maybe close after N attempts?
                elif order_info.get('status') != 'open':
                    # If the original limit order is now 'closed', it means it filled (handled by _check_pending).
                    # If it's 'canceled' or other non-open status AFTER being active, it's unusual.
                    # Market entry orders are usually 'closed' immediately.
                    # This check is less useful for confirming *ongoing* active status for parameter-based SL/TP.
                    logger.debug(f"[{func_name}] Entry order {pos_id} status is '{order_info.get('status')}' (not 'open'). May not reflect current position status accurately.")
                    # We don't infer closure from this check alone if the position was already active.


            # --- If External Closure Detected by fetchPositions ---
            if position_closed_externally:
                # Infer reason based on price movement relative to stored SL/TP (best effort)
                if exit_reason == "Not Found via fetchPositions" or "External Size Change" in exit_reason:
                    inferred_reason = self._infer_exit_reason_price_based(position, current_price) # Use current price for inference
                    if inferred_reason:
                         exit_reason += f" ({inferred_reason})"

                log_color = Fore.RED if "SL" in exit_reason else (Fore.GREEN if "TP" in exit_reason else Fore.YELLOW)
                logger.info(f"{log_color}[{func_name}] External closure detected for {pos_id}. Reason: {exit_reason}. Exit Price ~ {self.format_price(exit_price)}{Style.RESET_ALL}")
                # Mark for removal and PnL logging
                positions_to_remove_ids.add(pos_id)
                positions_to_update_state[pos_id] = {'status': STATUS_CLOSED_EXT, 'exit_reason': exit_reason, 'exit_price': exit_price, 'last_update_time': time.time()}
                self._log_position_pnl(position, exit_price, exit_reason) # Log PnL now
                continue # Move to the next active position

            # --- If Position Appears Active, Check Bot-Managed Exits ---

            # --- 2. Time-Based Exit Check ---
            if self.time_based_exit_minutes is not None and self.time_based_exit_minutes > 0:
                time_elapsed_min = (time.time() - entry_time) / 60.0
                if time_elapsed_min >= self.time_based_exit_minutes:
                    logger.info(f"{Fore.YELLOW}[{func_name}] Time limit ({self.time_based_exit_minutes} min) reached for {pos_id} (Age: {time_elapsed_min:.1f}m). Initiating market close.{Style.RESET_ALL}")
                    # Attempt to close the position via market order
                    market_close_order = self._place_market_close_order(position, current_price)
                    if market_close_order:
                        exit_reason = f"Time Limit ({self.time_based_exit_minutes}min)"
                        # Use actual fill price from close order if available, else current price
                        exit_price_raw = market_close_order.get('average') or current_price
                        try: exit_price = float(exit_price_raw)
                        except (TypeError, ValueError): exit_price = current_price
                        logger.info(f"[{func_name}] Market close for time exit placed for {pos_id}. Exit confirmed/estimated at ~{self.format_price(exit_price)}")
                        # Mark for removal and PnL logging
                        positions_to_remove_ids.add(pos_id)
                        positions_to_update_state[pos_id] = {'status': STATUS_CLOSED_ON_EXIT, 'exit_reason': exit_reason, 'exit_price': exit_price, 'last_update_time': time.time()}
                        self._log_position_pnl(position, exit_price, exit_reason)
                        continue # Position closed, move to next
                    else:
                        # CRITICAL FAILURE: Failed to close the position!
                        logger.critical(f"{Fore.RED}{Style.BRIGHT}[{func_name}] CRITICAL FAILURE closing timed-out position {pos_id}. REMAINS OPEN! Manual action required!{Style.RESET_ALL}")
                        # Do NOT mark for removal, let it retry or require manual intervention. Maybe add a 'failed_close_attempt' flag?

            # --- 3. Trailing Stop Loss (TSL) Logic [EXPERIMENTAL] ---
            # Execute only if TSL is enabled AND position wasn't closed by time limit
            if self.enable_trailing_stop_loss:
                if not self.trailing_stop_loss_percentage or not (0 < self.trailing_stop_loss_percentage < 1):
                     logger.error(f"[{func_name}] TSL enabled but percentage invalid ({self.trailing_stop_loss_percentage}). Disabling TSL this cycle.")
                else:
                    # Calculate if TSL needs to be activated or updated
                    new_tsl_price = self._update_trailing_stop_price(position, current_price)
                    if new_tsl_price is not None:
                        # If a new TSL price is calculated, attempt to modify the order's SL
                        logger.warning(f"{Fore.YELLOW}[EXPERIMENTAL][{func_name}] TSL update triggered for {pos_id}. New target SL: {self.format_price(new_tsl_price)}. Attempting edit_order... VERIFY EXCHANGE SUPPORT!{Style.RESET_ALL}")
                        # Attempt to edit the order (handles retries internally)
                        edit_success = self._attempt_edit_order_for_tsl(position, new_tsl_price)

                        if edit_success:
                            # If edit attempt was successful (or at least didn't hard fail), update internal state
                            update_payload = {
                                'trailing_stop_price': new_tsl_price, # Store the activated/updated TSL price
                                'stop_loss_price': new_tsl_price, # Update main SL to follow the TSL
                                'last_update_time': time.time()
                            }
                            # Merge updates carefully (if other updates for this pos_id exist)
                            current_updates = positions_to_update_state.get(pos_id, {})
                            positions_to_update_state[pos_id] = {**current_updates, **update_payload}
                            logger.info(f"{Fore.MAGENTA}---> Internal TSL state for {pos_id} updated to {self.format_price(new_tsl_price)} after edit attempt.{Style.RESET_ALL}")
                        else:
                             logger.error(f"{Fore.RED}[{func_name}] TSL update via edit_order for {pos_id} failed or skipped. TSL state NOT updated. Check logs for edit attempt details.{Style.RESET_ALL}")


        # --- Apply State Updates and Removals After Iterating All Active Positions ---
        if positions_to_remove_ids or positions_to_update_state:
            new_positions_list = []
            removed_count = 0
            updated_count = 0
            original_count = len(self.open_positions)

            for pos in self.open_positions:
                pos_id = pos.get('id')
                if pos_id in positions_to_remove_ids:
                    removed_count += 1
                    # Optionally log final state of removed position
                    logger.debug(f"[{func_name}] Removing position {pos_id} from state. Final stored state: {positions_to_update_state.get(pos_id)}")
                    continue # Skip adding to new list

                # Apply updates if any exist for this position ID
                if pos_id in positions_to_update_state:
                    update_data = positions_to_update_state[pos_id]
                    pos.update(update_data) # Update the dictionary in place
                    updated_count += 1
                    logger.debug(f"[{func_name}] Applied state updates to {pos_id}: {update_data}")

                # Add the (potentially updated) position to the new list
                new_positions_list.append(pos)

            if removed_count > 0 or updated_count > 0:
                self.open_positions = new_positions_list
                logger.debug(f"[{func_name}] Position management state update complete. Original Count: {original_count}, Updated: {updated_count}, Removed: {removed_count}. New Count: {len(self.open_positions)}")
                self._save_state() # Save the modified state

    def _infer_exit_reason_price_based(self, position: Dict, exit_price: float) -> Optional[str]:
        """
        Tries to infer the reason for an external position closure based ONLY
        on exit price relative to stored SL/TP. Used as a fallback.
        """
        func_name = "_infer_exit_reason_price_based"
        pos_id = position.get('id', 'N/A')
        side = position.get('side')
        entry_price = position.get('entry_price')
        stored_sl = position.get('stop_loss_price')
        stored_tp = position.get('take_profit_price')

        if not all([side, entry_price]): return None # Not enough info

        try:
            price_tick = float(self.market_info['precision']['price']) if self.market_info else 1e-8
            # Use a tolerance based on ticks or a small percentage of entry price
            tolerance = max(entry_price * 0.0005, price_tick * 5) # 0.05% or 5 ticks, adjust as needed

            sl_hit, tp_hit = False, False

            if stored_sl is not None:
                  sl_hit = (side == 'buy' and exit_price <= stored_sl + tolerance) or \
                           (side == 'sell' and exit_price >= stored_sl - tolerance)

            if stored_tp is not None:
                  tp_hit = (side == 'buy' and exit_price >= stored_tp - tolerance) or \
                           (side == 'sell' and exit_price <= stored_tp + tolerance)

            if sl_hit and tp_hit:
                # Price landed near both? Ambiguous.
                logger.debug(f"[{func_name}] Exit price {exit_price} near both SL ({stored_sl}) and TP ({stored_tp}) for {pos_id}.")
                return "SL/TP Hit (Ambiguous)"
            elif sl_hit:
                logger.debug(f"[{func_name}] Inferred SL hit for {pos_id} (Exit: {exit_price}, SL: {stored_sl})")
                return "SL Hit (Price Inference)"
            elif tp_hit:
                logger.debug(f"[{func_name}] Inferred TP hit for {pos_id} (Exit: {exit_price}, TP: {stored_tp})")
                return "TP Hit (Price Inference)"
            else:
                # Price doesn't match SL or TP closely
                logger.debug(f"[{func_name}] Exit price {exit_price} doesn't match SL ({stored_sl}) or TP ({stored_tp}) for {pos_id}.")
                return "Manual/Other (Price Inference)"
        except Exception as e:
            logger.warning(f"[{func_name}] Error during price-based inference for {pos_id}: {e}")
            return None


    def _update_trailing_stop_price(self, position: Dict, current_price: float) -> Optional[float]:
        """
        Calculates the new Trailing Stop Loss (TSL) price if the conditions are met.
        - Activates TSL if price moves favorably beyond a buffer from entry.
        - Updates active TSL if price moves further favorably.
        - Ensures the TSL price is valid and adheres to price precision.

        Args:
            position: The position dictionary (must contain 'side', 'entry_price',
                      'stop_loss_price', optionally 'trailing_stop_price').
            current_price: The current market price.

        Returns:
            The new TSL price if it should be updated/activated, otherwise None.
            Returns None if TSL percentage is invalid or calculation fails.
        """
        func_name = "_update_trailing_stop_price"
        side = position.get('side')
        entry_price = position.get('entry_price')
        current_tsl = position.get('trailing_stop_price') # The currently active TSL price, if any
        base_sl = position.get('stop_loss_price') # Original SL or last manually set SL
        tsl_percentage = self.trailing_stop_loss_percentage

        # Validate necessary inputs
        if not side or not entry_price:
            logger.error(f"[{func_name}] Missing side or entry_price for TSL calculation on pos {position.get('id')}")
            return None
        if not tsl_percentage or not (0 < tsl_percentage < 1):
            # Log this only once maybe? Or let caller handle repeated logs.
            # logger.error(f"[{func_name}] Invalid TSL percentage ({tsl_percentage}). Cannot calculate TSL.")
            return None
        if current_price <= 0:
            logger.warning(f"[{func_name}] Invalid current_price ({current_price}) for TSL calculation.")
            return None

        # --- Calculate Potential New TSL based on Current Price ---
        new_tsl_price: Optional[float] = None
        # For buy: TSL = current_price * (1 - percentage)
        # For sell: TSL = current_price * (1 + percentage)
        tsl_factor = (1.0 - tsl_percentage) if side == 'buy' else (1.0 + tsl_percentage)
        potential_tsl_raw = current_price * tsl_factor

        try:
            # Apply price precision
            potential_tsl_str = self.exchange.price_to_precision(self.symbol, potential_tsl_raw)
            if potential_tsl_str is None: raise ValueError("price_to_precision returned None")
            potential_tsl = float(potential_tsl_str)

            # Ensure potential TSL is positive after precision
            if potential_tsl <= 1e-12:
                 logger.debug(f"[{func_name}] Potential TSL ({potential_tsl}) is zero/negative after precision. Skipping.")
                 potential_tsl = None

        except (ccxt.ExchangeError, ValueError, TypeError) as e:
            logger.error(f"[{func_name}] Error applying precision to potential TSL price {potential_tsl_raw}: {e}. Cannot calculate TSL.")
            potential_tsl = None
        except Exception as e:
             logger.error(f"[{func_name}] Unexpected error formatting potential TSL {potential_tsl_raw}: {e}. Cannot calculate TSL.", exc_info=True)
             potential_tsl = None

        if potential_tsl is None: return None # Stop if potential TSL is invalid

        # --- TSL Activation Check (if TSL is not currently active) ---
        is_tsl_active = current_tsl is not None
        if not is_tsl_active:
            # Condition 1: Price must have moved favorably beyond entry by a small buffer
            # This prevents activating TSL immediately on tiny fluctuations
            price_tick = float(self.market_info['precision'].get('price', 1e-8))
            # Buffer: e.g., 0.1% of entry price or 5 ticks, whichever is larger
            activation_buffer = max(entry_price * 0.001, price_tick * 5)
            activation_price_buy = entry_price + activation_buffer
            activation_price_sell = entry_price - activation_buffer

            price_moved_enough = (side == 'buy' and current_price > activation_price_buy) or \
                                 (side == 'sell' and current_price < activation_price_sell)

            if price_moved_enough:
                # Condition 2: Potential TSL must be better (more protective) than the base SL (if base SL exists)
                is_better_than_base = True # Assume true if no base SL
                if base_sl is not None:
                    is_better_than_base = (side == 'buy' and potential_tsl > base_sl) or \
                                          (side == 'sell' and potential_tsl < base_sl)

                if is_better_than_base:
                    # Conditions met, activate TSL
                    new_tsl_price = potential_tsl
                    logger.info(f"{Fore.MAGENTA}[{func_name}] TSL ACTIVATING for {side} {position['id']} at {self.format_price(new_tsl_price)} (Current Price: {self.format_price(current_price)}, Entry: {self.format_price(entry_price)}, BaseSL: {self.format_price(base_sl)}){Style.RESET_ALL}")
                else:
                    logger.debug(f"[{func_name}] TSL activation condition met for {position['id']}, but potential TSL {self.format_price(potential_tsl)} not better than base SL {self.format_price(base_sl)}. Holding base SL.")
            # else: # Price hasn't moved enough from entry yet
                 # logger.debug(f"[{func_name}] Price ({self.format_price(current_price)}) not beyond activation threshold ({self.format_price(activation_price_buy if side=='buy' else activation_price_sell)}) for {position['id']}. TSL not activated.")

        # --- TSL Update Check (if TSL is already active) ---
        elif is_tsl_active:
            # Update TSL only if the potential new TSL is better (more protective) than the current active TSL
            update_needed = (side == 'buy' and potential_tsl > current_tsl) or \
                            (side == 'sell' and potential_tsl < current_tsl)

            if update_needed:
                 # Ensure the new TSL isn't trying to cross the current price (can happen with volatile wicks)
                 is_valid_update = (side == 'buy' and potential_tsl < current_price) or \
                                   (side == 'sell' and potential_tsl > current_price)
                 if is_valid_update:
                     new_tsl_price = potential_tsl
                     logger.info(f"{Fore.MAGENTA}[{func_name}] TSL UPDATING for {side} {position['id']} from {self.format_price(current_tsl)} to {self.format_price(new_tsl_price)} (Current Price: {self.format_price(current_price)}){Style.RESET_ALL}")
                 else:
                      logger.warning(f"[{func_name}] Potential TSL update for {position['id']} to {self.format_price(potential_tsl)} skipped because it crossed current price {self.format_price(current_price)}.")


        # Return the newly calculated TSL price if activation/update occurred
        return new_tsl_price

    def _attempt_edit_order_for_tsl(self, position: Dict, new_tsl_price: float) -> bool:
        """
        EXPERIMENTAL: Attempts to modify the order's Stop Loss using `edit_order`.
        This is highly exchange-specific and may not be supported or reliable.
        Handles Bybit V5 requirement of resending TP when modifying SL.

        Args:
            position: The position dictionary.
            new_tsl_price: The new stop loss price to set.

        Returns:
            bool: True if the edit API call was attempted and seemed successful
                  (or returned without immediate failure/NotSupported).
                  False if the call failed, was not supported, or order couldn't be edited.
        """
        func_name = "_attempt_edit_order_for_tsl"
        pos_id = position.get('id') # This should be the *original entry order ID* if that's what needs editing
        symbol = position.get('symbol', self.symbol)
        # Retrieve current TP from state to resend it (required by Bybit V5 editOrder)
        stored_tp_price = position.get('take_profit_price')

        # --- Log Warnings ---
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}[EXPERIMENTAL][{func_name}] Attempting TSL update via edit_order for Order ID {pos_id}. THIS IS EXPERIMENTAL AND MAY NOT WORK RELIABLY ON {self.exchange_id.upper()}.{Style.RESET_ALL}")
        if self.exchange_id.lower() == 'bybit':
             logger.warning(f"{Fore.YELLOW}[Bybit V5 Note] Resending TP={self.format_price(stored_tp_price)} along with SL update for {pos_id}. If TP is omitted, it might be removed by the edit.{Style.RESET_ALL}")

        # --- Check Exchange Capability ---
        if not self.exchange.has.get('editOrder'):
            logger.error(f"{Fore.RED}[{func_name}] TSL Update FAILED: Exchange {self.exchange_id} does not report 'editOrder' capability via CCXT. Cannot modify order SL/TP. Disable TSL.{Style.RESET_ALL}")
            # Permanently disable TSL for this run if editOrder isn't supported
            self.enable_trailing_stop_loss = False
            logger.warning(f"{Fore.YELLOW}---> Disabling 'enable_trailing_stop_loss' due to lack of exchange support.{Style.RESET_ALL}")
            return False

        # --- Prepare Edit Parameters ---
        try:
            edit_params = {
                'stopLoss': self.format_price(new_tsl_price) # Send SL as formatted string
            }
            # Add trigger price type if configured
            if self.sl_trigger_by: edit_params['slTriggerBy'] = self.sl_trigger_by

            # Resend Take Profit if it exists (CRITICAL for Bybit V5)
            if stored_tp_price is not None and stored_tp_price > 0:
                edit_params['takeProfit'] = self.format_price(stored_tp_price)
                if self.tp_trigger_by: edit_params['tpTriggerBy'] = self.tp_trigger_by
                logger.debug(f"[{func_name}] Resending TP={edit_params['takeProfit']} (Trig: {edit_params.get('tpTriggerBy', 'N/A')}) with SL update.")
            else:
                logger.debug(f"[{func_name}] No existing TP found or TP is zero for {pos_id}. Not resending TP with SL edit.")

            # Add Bybit V5 category parameter hint
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                market_type = self.market_info.get('type', 'swap')
                is_linear = self.market_info.get('linear', True)
                if market_type in ['swap', 'future']: edit_params['category'] = 'linear' if is_linear else 'inverse'
                elif market_type == 'spot': edit_params['category'] = 'spot' # SL/TP might not apply to spot edit? Check docs
                elif market_type == 'option': edit_params['category'] = 'option'
                if 'category' in edit_params: logger.debug(f"[{func_name}] Added Bybit category hint for edit: {edit_params['category']}")

            # --- Fetch Fresh Order Info (Potentially Optional, depends on CCXT/Exchange) ---
            # Some exchanges might not require all original details for SL/TP modify
            # However, safer to assume they might be needed by ccxt's edit_order implementation
            # Let's skip fetching fresh info for now and rely on ccxt's handling, unless errors occur.
            # If errors like "missing parameter" occur, uncomment and adapt this section:
            # logger.debug(f"[{func_name}] Fetching fresh order info for {pos_id} before editing...")
            # fresh_order_info = self.fetch_order_status(pos_id, symbol=symbol)
            # if not fresh_order_info: #  or fresh_order_info.get('status') != 'open': # Status check might be too strict if order filled but SL/TP modifiable
            #      logger.warning(f"[{func_name}] Cannot edit TSL: Failed to fetch fresh info for order {pos_id} or order not found/active.")
            #      return False
            # # Extract necessary fields if required by edit_order (consult ccxt docs)
            # order_type = fresh_order_info.get('type')
            # order_side = fresh_order_info.get('side')
            # order_amount = fresh_order_info.get('amount') # Original amount? Filled amount? Check docs.
            # order_price = fresh_order_info.get('price') # Price for limit orders
            # if not all([order_type, order_side, order_amount]): raise ValueError("Missing essential fields from fresh order info needed for edit.")
            # if order_type == 'limit' and order_price is None: raise ValueError("Missing price from fresh limit order info needed for edit.")

            logger.debug(f"[{func_name}] Calling edit_order for {pos_id}. New SL: {self.format_price(new_tsl_price)}. Params: {edit_params}")

            # --- Perform the Edit API Call (Using retry decorator internally) ---
            @retry_api_call(max_retries=1, initial_delay=1) # Retry edit call once
            def _edit_call_wrapper(order_id, symbol, params):
                # edit_order signature simplified here: CCXT often only needs ID, Symbol, Params for SL/TP mod.
                # If full signature is needed, pass type, side, amount, price from fresh_order_info above.
                # return self.exchange.edit_order(order_id, symbol, order_type, order_side, order_amount, order_price, params)
                return self.exchange.edit_order(order_id, symbol, params=params) # Try simplified call first

            edited_order_response = _edit_call_wrapper(pos_id, symbol, edit_params)

            # --- Process Edit Response ---
            if edited_order_response:
                # Check response for confirmation (highly exchange specific)
                confirmed_sl = edited_order_response.get('info',{}).get('stopLoss') or edited_order_response.get('stopLossPrice')
                confirmed_tp = edited_order_response.get('info',{}).get('takeProfit') or edited_order_response.get('takeProfitPrice')
                response_id = edited_order_response.get('id', 'N/A')
                logger.info(f"{Fore.MAGENTA}---> edit_order attempt for {pos_id} completed. Response ID: {response_id}. Exchange Response SL: {confirmed_sl}, TP: {confirmed_tp}{Style.RESET_ALL}")

                # Basic Success Check: Did the API call return without error?
                # More robust check: Compare confirmed_sl with new_tsl_price (requires parsing float)
                edit_seems_ok = False
                if confirmed_sl:
                    try:
                        # Compare floats with tolerance
                        if abs(float(confirmed_sl) - new_tsl_price) < 1e-9:
                             logger.info(f"{Fore.GREEN}---> TSL edit CONFIRMED via matching response SL for {pos_id}.{Style.RESET_ALL}")
                             edit_seems_ok = True
                        else:
                             logger.warning(f"{Fore.YELLOW}---> TSL edit response SL ({confirmed_sl}) mismatch target ({new_tsl_price}) for {pos_id}. Edit *might* have failed or has lag.{Style.RESET_ALL}")
                             # Still return True, assuming API call itself didn't fail? Or False? Let's be optimistic but log warning.
                             edit_seems_ok = True # Assume non-failure is potential success
                    except (ValueError, TypeError):
                         logger.warning(f"{Fore.YELLOW}---> Could not parse confirmed SL '{confirmed_sl}' to float for comparison. Assuming edit attempt was okay.{Style.RESET_ALL}")
                         edit_seems_ok = True # Assume okay if response exists but can't parse SL
                else:
                    logger.warning(f"{Fore.YELLOW}---> TSL edit response did not contain confirmation SL field for {pos_id}. Assuming attempt was okay (but unverified).{Style.RESET_ALL}")
                    edit_seems_ok = True # Assume okay if response exists but no SL field

                return edit_seems_ok

            else: # edit_order call itself (or retries) returned None
                logger.error(f"{Fore.RED}[{func_name}] TSL update via edit_order failed: API call returned None after retries for {pos_id}.{Style.RESET_ALL}")
                return False

        except ccxt.NotSupported as e:
            # Catch NotSupported specifically if 'has' check missed it or for specific params
            logger.error(f"{Fore.RED}[{func_name}] TSL Update FAILED: edit_order with SL/TP params reported as NOT SUPPORTED by exchange/CCXT. Error: {e}. Disable TSL.{Style.RESET_ALL}")
            self.enable_trailing_stop_loss = False # Disable TSL
            logger.warning(f"{Fore.YELLOW}---> Disabling 'enable_trailing_stop_loss' due to NotSupported error during edit.{Style.RESET_ALL}")
            return False
        except (ccxt.OrderNotFound, ccxt.InvalidOrder) as e:
             # Order might have closed between TSL calculation and edit attempt, or parameters invalid
             logger.warning(f"{Fore.YELLOW}[{func_name}] TSL edit_order failed for {pos_id}: Order likely closed or parameters invalid ({type(e).__name__}): {e}.{Style.RESET_ALL}")
             return False # Don't treat as success
        except ccxt.ExchangeError as e:
             # Handle other exchange errors during edit attempt
             logger.error(f"{Fore.RED}[{func_name}] TSL edit_order failed for {pos_id} (ExchangeError): {e}.{Style.RESET_ALL}")
             return False
        except Exception as e:
            # Catch unexpected Python errors during preparation or call
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error attempting TSL edit for {pos_id}: {e}{Style.RESET_ALL}", exc_info=True)
            return False

    def _log_position_pnl(self, position: Dict, exit_price: Optional[float], reason: str) -> None:
        """
        Calculates and logs the Profit and Loss (PnL) for a closed position.
        Updates the simple daily PnL tracker.

        Args:
            position: The position dictionary (containing entry details and size).
            exit_price: The price at which the position was closed.
            reason: A string describing why the position was closed.
        """
        func_name = "_log_position_pnl"
        pos_id = position.get('id', 'N/A')
        side = position.get('side')
        entry_price = position.get('entry_price')
        # Use 'size' which should reflect the actual filled size upon activation
        position_size = position.get('size')
        symbol = position.get('symbol', self.symbol)

        # Validate necessary data for PnL calculation
        if not all([pos_id, side, entry_price, position_size, exit_price, symbol]):
            logger.warning(f"{Fore.YELLOW}---> Pos {pos_id} Closed ({reason}). PnL calculation skipped: Missing data (E={entry_price}, X={exit_price}, Size={position_size}).{Style.RESET_ALL}")
            return
        if entry_price <= 0 or exit_price <= 0 or position_size <= 0:
            logger.warning(f"{Fore.YELLOW}---> Pos {pos_id} Closed ({reason}). PnL calculation skipped: Invalid data (E={entry_price}, X={exit_price}, Size={position_size}).{Style.RESET_ALL}")
            return

        try:
            # Calculate PnL in Quote Currency
            pnl_quote = 0.0
            if side == 'buy':
                pnl_quote = (exit_price - entry_price) * position_size
            elif side == 'sell':
                pnl_quote = (entry_price - exit_price) * position_size

            # Calculate PnL Percentage relative to entry price
            pnl_pct = 0.0
            if entry_price > 0: # Avoid division by zero
                 pnl_pct = (exit_price / entry_price - 1.0) * 100.0 if side == 'buy' else \
                           (entry_price / exit_price - 1.0) * 100.0

            pnl_color = Fore.GREEN if pnl_quote >= 0 else Fore.RED
            quote_ccy = self.market_info['quote'] if self.market_info else 'Quote'
            base_ccy = self.market_info['base'] if self.market_info else 'Base'

            # Log the closure details and PnL
            log_msg = (f"{pnl_color}---> Position Closed: ID={pos_id} Symbol={symbol} Side={side.upper()} Reason='{reason}'.\n"
                       f"     Entry: {self.format_price(entry_price)}, Exit: {self.format_price(exit_price)}, Size: {self.format_amount(position_size)} {base_ccy}.\n"
                       f"     Est. PnL: {self.format_price(pnl_quote)} {quote_ccy} ({pnl_pct:+.3f}%){Style.RESET_ALL}") # Added '+' sign for percentage
            logger.info(log_msg)

            # Update simple daily PnL tracker
            self.daily_pnl += pnl_quote
            logger.info(f"Daily PnL Updated: {self.format_price(self.daily_pnl)} {quote_ccy}")

        except ZeroDivisionError:
            logger.error(f"[{func_name}] PnL calculation failed for {pos_id}: Division by zero (Entry Price: {entry_price}).")
        except Exception as e:
            logger.error(f"[{func_name}] Unexpected error calculating PnL for {pos_id}: {e}", exc_info=True)

    @retry_api_call(max_retries=1) # Retry market close once on failure
    def _place_market_close_order(self, position: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        """
        Places a market order specifically designed to close the given position.
        Uses 'reduceOnly' parameter if supported by the exchange/ccxt.

        Args:
            position: The position dictionary (must contain 'id', 'side', 'size', 'symbol').
            current_price: Current market price (used for simulation fill).

        Returns:
            The order dictionary returned by ccxt (or simulated), or None on failure.
        """
        func_name = "_place_market_close_order"
        pos_id = position.get('id', 'N/A')
        side = position.get('side')
        size = position.get('size') # Current size of the position to close
        symbol = position.get('symbol', self.symbol)

        if self.market_info is None: logger.error(f"[{func_name}] Market info missing."); return None
        base_ccy = self.market_info['base']
        quote_ccy = self.market_info['quote']

        if size is None or size <= 1e-12: # Use epsilon
            logger.error(f"[{func_name}] Cannot place market close for {pos_id}: Invalid or zero size ({size}).")
            return None
        if side not in ['buy', 'sell']:
             logger.error(f"[{func_name}] Cannot place market close for {pos_id}: Invalid side '{side}'.")
             return None

        # Determine the side needed to close the position
        close_side = 'sell' if side == 'buy' else 'buy'
        log_color = Fore.YELLOW # Use yellow for closure actions

        logger.warning(f"{log_color}[{func_name}] Initiating MARKET CLOSE for Position ID {pos_id} ({symbol}). "
                       f"Entry Side: {side.upper()}, Close Order: {close_side.upper()}, Size: {self.format_amount(size)} {base_ccy}...{Style.RESET_ALL}")

        # --- Simulation Mode ---
        if self.simulation_mode:
            sim_close_id = f"sim_close_{int(time.time() * 1000)}_{close_side[:1]}"
            sim_avg_close = current_price # Simulate fill at current price
            sim_ts = int(time.time() * 1000)
            simulated_close_order = {
                "id": sim_close_id, "timestamp": sim_ts, "datetime": pd.to_datetime(sim_ts, unit='ms', utc=True).isoformat(),
                "symbol": symbol, "type": "market", "side": close_side, "status": 'closed',
                "amount": size, # Amount requested to close
                "filled": size, # Assume fully filled in sim
                "remaining": 0.0,
                "average": sim_avg_close, # Fill price
                "cost": size * sim_avg_close,
                "reduceOnly": True, # Mark as intended reduceOnly
                "info": {"simulated": True, "orderId": sim_close_id, "reduceOnly": True, "closed_position_id": pos_id}
            }
            logger.info(f"{log_color}[SIMULATION] Market Close Order Placed: ID {sim_close_id}, AvgPrice {self.format_price(sim_avg_close)}{Style.RESET_ALL}")
            return simulated_close_order

        # --- Live/Testnet Mode ---
        else:
            try:
                params = {}
                # Use reduceOnly parameter - critical for ensuring it only closes position
                # Check how the specific exchange expects it via ccxt
                # Bybit V5 uses params['reduceOnly'] = True
                if self.exchange_id.lower() == 'bybit':
                     params['reduceOnly'] = True
                     logger.debug(f"[{func_name}] Using Bybit V5 'reduceOnly=True' parameter.")
                # Other exchanges might use different keys in params or require it in create_order
                # Consult ccxt unified API docs or exchange-specific overrides if needed
                # Example: some might use {'reduce_only': True} or similar

                # Add Bybit V5 category parameter hint
                if self.exchange_id.lower() == 'bybit' and self.market_info:
                    market_type = self.exchange.market(symbol).get('type', 'swap')
                    is_linear = self.exchange.market(symbol).get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot' # reduceOnly might not apply to spot?
                    elif market_type == 'option': params['category'] = 'option'
                    if 'category' in params: logger.debug(f"[{func_name}] Added Bybit category hint for close: {params['category']}")

                logger.debug(f"[{func_name}] Placing live market close order for {pos_id}. Side: {close_side}, Size: {size}, Params: {params}")
                # Use create_market_order (or create_order with 'market' type)
                order = self.exchange.create_market_order(symbol, close_side, size, params=params)

                # Process response
                if order:
                    oid = order.get('id', 'N/A')
                    oavg = order.get('average') # Actual fill price
                    ostatus = order.get('status', STATUS_UNKNOWN)
                    ofilled = order.get('filled', 0.0)
                    logger.info(
                        f"{log_color}---> LIVE Market Close Order Placed: ID {oid}, "
                        f"Status: {ostatus}, Filled: {self.format_amount(ofilled)}, AvgFill: {self.format_price(oavg)}{Style.RESET_ALL}"
                    )
                    return order
                else:
                    # Should be caught by retry decorator if API call failed, but handle defensively
                    logger.error(f"{Fore.RED}[{func_name}] LIVE Market Close failed: API call returned None for {pos_id}.{Style.RESET_ALL}")
                    return None

            # Handle specific errors related to closing orders
            except ccxt.InsufficientFunds as e:
                 # This might happen if position already closed or margin issues
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Market Close for {pos_id} failed (InsufficientFunds): {e}. Position might already be closed or margin changed.{Style.RESET_ALL}")
                 return None # Treat as potentially closed? Or failure? Let's return None for now.
            except ccxt.InvalidOrder as e:
                 # Often indicates position already closed, reduceOnly conflict, or parameter issue
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Market Close for {pos_id} failed (InvalidOrder): {e}. Likely already closed or reduceOnly/parameter issue.{Style.RESET_ALL}")
                 return None # Treat as potentially closed
            except ccxt.ExchangeError as e:
                 # Check for specific messages indicating closure state
                 err_str = str(e).lower()
                 # Bybit examples: "order cost not meet", "position size is zero", "position idx not match position mode" (hedge mode issues?)
                 # General examples: "reduce-only", "position does not exist"
                 close_related_errors = ["order cost not meet", "position size is zero", "reduce-only", "position does not exist", "position is closed"]
                 if any(phrase in err_str for phrase in close_related_errors):
                      logger.warning(f"{Fore.YELLOW}[{func_name}] Market close for {pos_id} failed with ExchangeError, likely already closed or reduce-only issue: {e}{Style.RESET_ALL}")
                      return None # Treat as potentially closed
                 else:
                      logger.error(f"{Fore.RED}[{func_name}] Market Close for {pos_id} failed (ExchangeError): {e}{Style.RESET_ALL}")
                      return None # Treat as failed
            except Exception as e:
                 logger.error(f"{Fore.RED}[{func_name}] Market Close for {pos_id} failed (Unexpected Error): {e}{Style.RESET_ALL}", exc_info=True)
                 return None # Treat as failed

    # --- Main Bot Execution Loop ---

    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches the bundle of market data needed for one iteration."""
        func_name = "_fetch_market_data"
        logger.debug(f"[{func_name}] Fetching market data bundle...")
        start_time = time.time()
        try:
            # Fetch data concurrently? For now, sequential is simpler.
            current_price = self.fetch_market_price()
            order_book_data = self.fetch_order_book()
            historical_data = self.fetch_historical_data() # Auto-calculates lookback

            # --- Validate Fetched Data ---
            if current_price is None:
                logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch current market price. Skipping iteration.{Style.RESET_ALL}")
                return None
            if historical_data is None or historical_data.empty:
                # fetch_historical_data logs reasons for None/empty
                logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch sufficient historical data. Skipping iteration.{Style.RESET_ALL}")
                return None
            # Order book is optional, proceed even if None
            if order_book_data is None:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Failed to fetch order book data. Proceeding without imbalance signal.{Style.RESET_ALL}")

            imbalance = order_book_data.get('imbalance') if order_book_data else None
            fetch_duration = time.time() - start_time
            logger.debug(f"[{func_name}] Market data fetched in {fetch_duration:.3f}s.")

            return {
                "price": current_price,
                "order_book_imbalance": imbalance,
                "historical_data": historical_data
            }
        except Exception as e:
            # Catch unexpected errors during the data fetching sequence
            logger.error(f"{Fore.RED}[{func_name}] Unexpected error during market data fetch sequence: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculates all technical indicators based on historical data."""
        func_name = "_calculate_indicators"
        logger.debug(f"[{func_name}] Calculating indicators...")
        start_time = time.time()
        indicators = {} # Dictionary to store results

        # --- Input Validation ---
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
             logger.error(f"[{func_name}] Invalid or empty historical data provided for indicator calculation.")
             return {} # Return empty dict
        required_cols = ['open', 'high', 'low', 'close', 'volume'] # Volume might be optional depending on indicators used
        if not all(col in historical_data.columns for col in required_cols):
             logger.error(f"[{func_name}] Historical data is missing required columns ({required_cols}). Cannot calculate indicators.")
             return {}
        if len(historical_data) < 2: # Most indicators need at least 2 points
             logger.warning(f"[{func_name}] Historical data has very few rows ({len(historical_data)}). Indicators will likely be None.")
             # Proceed, but expect None results

        # --- Extract Series ---
        # Ensure correct types, handle potential NaNs if not already cleaned in fetch
        close = historical_data['close'].astype(float)
        high = historical_data['high'].astype(float)
        low = historical_data['low'].astype(float)
        # volume = historical_data['volume'].astype(float) # If needed

        # --- Calculate Indicators ---
        # Static methods handle internal checks for length and errors, return None on failure
        indicators['volatility'] = self.calculate_volatility(close, self.volatility_window)
        indicators['ema'] = self.calculate_ema(close, self.ema_period)
        indicators['rsi'] = self.calculate_rsi(close, self.rsi_period)
        macd_line, macd_signal, macd_hist = self.calculate_macd(close, self.macd_short_period, self.macd_long_period, self.macd_signal_period)
        indicators['macd_line'], indicators['macd_signal'], indicators['macd_hist'] = macd_line, macd_signal, macd_hist
        stoch_k, stoch_d = self.calculate_stoch_rsi(close, self.rsi_period, self.stoch_rsi_period, self.stoch_rsi_k_period, self.stoch_rsi_d_period)
        indicators['stoch_k'], indicators['stoch_d'] = stoch_k, stoch_d
        indicators['atr'] = self.calculate_atr(high, low, close, self.atr_period)

        calc_duration = time.time() - start_time
        # --- Log Calculated Indicator Values ---
        # Format values nicely for logging
        log_msg = (
            f"Indicators (calc {calc_duration:.3f}s): "
            f"EMA({self.ema_period})={self.format_price(indicators.get('ema'))} | "
            f"RSI({self.rsi_period})={self.format_indicator(indicators.get('rsi'), 1)} | "
            f"ATR({self.atr_period})={self.format_price(indicators.get('atr'))} | "
            f"MACD({self.macd_short_period},{self.macd_long_period},{self.macd_signal_period})="
            f"{self.format_price(indicators.get('macd_line'), self.price_decimals+2)}/" # More precision for MACD
            f"{self.format_price(indicators.get('macd_signal'), self.price_decimals+2)}/"
            f"{self.format_price(indicators.get('macd_hist'), self.price_decimals+2)} | "
            f"StochRSI({self.stoch_rsi_period},{self.stoch_rsi_k_period},{self.stoch_rsi_d_period})="
            f"{self.format_indicator(indicators.get('stoch_k'), 1)}/"
            f"{self.format_indicator(indicators.get('stoch_d'), 1)} | "
            f"Vol({self.volatility_window})={self.format_indicator(indicators.get('volatility'), 5)}"
        )
        logger.info(log_msg) # Use INFO level for the summary line

        # Check for None values which indicate calculation issues
        nones = [k for k, v in indicators.items() if v is None]
        if nones:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Some indicators returned None: {nones}. This might be due to insufficient data length or errors during calculation. Check DEBUG logs.{Style.RESET_ALL}")

        return indicators

    def _process_signals_and_entry(self, market_data: Dict, indicators: Dict) -> None:
        """
        Analyzes signals based on indicators and market data, checks entry conditions,
        calculates order size and SL/TP, and places an entry order if warranted.
        Updates the bot state accordingly.
        """
        func_name = "_process_signals_and_entry"
        current_price = market_data.get('price')
        imbalance = market_data.get('order_book_imbalance')
        atr_value = indicators.get('atr') # Needed for SL/TP and potentially size

        # --- 1. Pre-computation & Validation ---
        if current_price is None: # Should have been caught earlier
             logger.error(f"[{func_name}] Cannot process signals: Current price is missing.")
             return
        # Indicators dict might be empty or contain Nones if calculation failed
        if not indicators:
            logger.warning(f"[{func_name}] Cannot process signals: Indicators dictionary is empty.")
            return

        # --- 2. Check Max Positions Condition ---
        # Count positions that are actively managed or waiting to become active
        active_or_pending_count = sum(1 for p in self.open_positions if p.get('status') in [STATUS_ACTIVE, STATUS_PENDING_ENTRY])
        if active_or_pending_count >= self.max_open_positions:
             logger.info(f"{Fore.CYAN}[{func_name}] Max positions limit ({self.max_open_positions}) reached (Currently {active_or_pending_count}). Skipping new entry evaluation.{Style.RESET_ALL}")
             return

        # --- 3. Compute Trade Signal Score ---
        signal_score, reasons = self.compute_trade_signal_score(current_price, indicators, imbalance)
        score_color = Fore.GREEN if signal_score > 0 else (Fore.RED if signal_score < 0 else Fore.WHITE)
        # Log score summary at INFO level
        logger.info(f"[{func_name}] Trade Signal Score: {score_color}{signal_score}{Style.RESET_ALL}")
        # Log detailed reasons only at DEBUG level or if score is significant
        if logger.isEnabledFor(logging.DEBUG) or abs(signal_score) >= ENTRY_SIGNAL_THRESHOLD_ABS:
            logger.debug(f"[{func_name}] Signal Breakdown:")
            for reason in reasons: logger.debug(f"  -> {reason}")

        # --- 4. Determine Entry Action based on Score ---
        entry_side: Optional[str] = None
        if signal_score >= ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'buy'
        elif signal_score <= -ENTRY_SIGNAL_THRESHOLD_ABS:
            entry_side = 'sell'

        # If no strong signal, log and exit this function
        if not entry_side:
            if signal_score is not None: # Avoid logging if score calculation failed entirely
                logger.info(f"[{func_name}] Neutral signal ({signal_score}). Entry threshold |{ENTRY_SIGNAL_THRESHOLD_ABS}| not met.")
            return

        # --- 5. Proceed with Entry Logic ---
        log_color = Fore.GREEN if entry_side == 'buy' else Fore.RED
        logger.info(f"{log_color}[{func_name}] {entry_side.upper()} signal score ({signal_score}) meets threshold. Preparing potential entry...{Style.RESET_ALL}")

        # --- 6. Calculate Order Size ---
        # Pass indicators and score for potential adjustments in calculate_order_size
        order_size_base = self.calculate_order_size(current_price, indicators, signal_score)
        if order_size_base <= 0:
             # calculate_order_size logs the specific reason (balance, limits, etc.)
             logger.warning(f"{Fore.YELLOW}[{func_name}] Entry aborted: Calculated order size is zero or invalid. Check preceding logs.{Style.RESET_ALL}")
             return

        # --- 7. Calculate SL/TP Prices ---
        # Use current_price as a proxy for entry price for this pre-order calculation
        sl_price, tp_price = self._calculate_sl_tp_prices(
            entry_price=current_price,
            side=entry_side,
            current_price=current_price, # Pass current price for context
            atr=atr_value # Pass current ATR
        )
        # Check if calculation failed AND if SL/TP are mandatory based on config
        # Example: If using ATR SL/TP, both must be calculable unless ATR is None
        sl_tp_required_atr = self.use_atr_sl_tp and atr_value is not None
        sl_tp_required_pct = not self.use_atr_sl_tp and (self.base_stop_loss_pct is not None or self.base_take_profit_pct is not None)
        sl_tp_required = sl_tp_required_atr or sl_tp_required_pct

        # If required, at least one of SL or TP must be valid
        sl_tp_valid = sl_price is not None or tp_price is not None
        if sl_tp_required and not sl_tp_valid:
             logger.error(f"{Fore.RED}[{func_name}] Entry aborted: Required SL/TP calculation failed (Result: SL={sl_price}, TP={tp_price}). Check ATR/percentage settings and data.{Style.RESET_ALL}")
             return
        elif not sl_tp_valid: # Calculation failed but maybe not strictly required
             logger.warning(f"{Fore.YELLOW}[{func_name}] SL/TP calculation failed or returned None (SL={sl_price}, TP={tp_price}). Proceeding without attaching SL/TP params to order.{Style.RESET_ALL}")
        else:
             logger.info(f"[{func_name}] Calculated SL={self.format_price(sl_price)}, TP={self.format_price(tp_price)} for potential entry.")


        # --- 8. Place Entry Order ---
        entry_order = self.place_entry_order(
            side=entry_side,
            order_size_base=order_size_base,
            confidence_level=signal_score,
            order_type=self.entry_order_type, # Use bound attribute
            current_price=current_price,
            stop_loss_price=sl_price, # Pass calculated SL (can be None)
            take_profit_price=tp_price # Pass calculated TP (can be None)
        )

        # --- 9. Update Bot State if Order Placed Successfully ---
        if entry_order:
            order_id = entry_order.get('id')
            if not order_id:
                 # This is critical - we placed an order but don't know its ID
                 logger.critical(f"{Fore.RED}{Style.BRIGHT}[{func_name}] FATAL: Entry order placed but exchange response missing order ID! Cannot track position. Manual check REQUIRED! Response snippet: {str(entry_order)[:250]}{Style.RESET_ALL}")
                 # Consider stopping the bot here? Or just log critical and continue?
                 return # Cannot add position to state without ID

            # Determine initial status based on order response
            order_status = entry_order.get('status') # 'open', 'closed', 'canceled', etc.
            initial_pos_status = STATUS_UNKNOWN
            is_filled_immediately = False

            if order_status == 'open': # Limit order placed but not filled
                initial_pos_status = STATUS_PENDING_ENTRY
            elif order_status == 'closed': # Market order likely filled, or limit filled instantly
                initial_pos_status = STATUS_ACTIVE
                is_filled_immediately = True
            elif order_status in ['canceled', 'rejected', 'expired']:
                # Order failed immediately
                reason = entry_order.get('info', {}).get('rejectReason', order_status)
                logger.warning(f"{Fore.YELLOW}[{func_name}] Entry order {order_id} failed immediately (Status: {order_status}, Reason: {reason}). No position added to state.{Style.RESET_ALL}")
                return # Do not add to state
            else: # Unusual initial status
                 initial_pos_status = STATUS_PENDING_ENTRY # Treat as pending by default
                 logger.warning(f"[{func_name}] Entry order {order_id} has unusual initial status '{order_status}'. Treating as {initial_pos_status}.")

            # Extract details for state entry
            filled_amount = float(entry_order.get('filled', 0.0))
            avg_fill_price = float(entry_order.get('average')) if entry_order.get('average') is not None else None
            timestamp_ms = entry_order.get('timestamp')
            # Use the 'amount' from the order response as 'original_size' requested
            requested_amount = float(entry_order.get('amount', order_size_base)) # Fallback to calculated size

            # Set state fields based on immediate fill status
            entry_price_state = avg_fill_price if is_filled_immediately and avg_fill_price else None
            entry_time_state = timestamp_ms / 1000.0 if is_filled_immediately and timestamp_ms else None
            # Store current filled amount, will be updated if pending fills later
            size_state = filled_amount if is_filled_immediately else 0.0

            # Create the new position dictionary
            new_position = {
                "id": order_id, # Use the actual order ID from exchange
                "symbol": self.symbol,
                "side": entry_side,
                "size": size_state, # Current filled size (0 if pending)
                "original_size": requested_amount, # The size requested in the order
                "entry_price": entry_price_state, # Avg fill price (None if pending)
                "entry_time": entry_time_state, # Fill time (None if pending)
                "status": initial_pos_status, # Initial status (pending or active)
                "entry_order_type": self.entry_order_type,
                # Store the SL/TP prices *intended* for this position based on pre-entry calculation
                # These might be updated later based on actual fill price in _check_pending_entries
                "stop_loss_price": sl_price,
                "take_profit_price": tp_price,
                "confidence": signal_score, # Store signal score for analysis
                "trailing_stop_price": None, # Initialize TSL price as None
                "last_update_time": time.time() # Time this state entry was created/updated
            }
            # Add the new position to our list
            self.open_positions.append(new_position)
            logger.info(f"{Fore.CYAN}---> Position {order_id} added to internal state. Status: {initial_pos_status}, Initial Entry Price: {self.format_price(entry_price_state)}, Original Size: {self.format_amount(requested_amount)}{Style.RESET_ALL}")
            # Save the state immediately after adding a position
            self._save_state()
        else:
             # place_entry_order returned None, indicating failure
             logger.error(f"{Fore.RED}[{func_name}] Entry order placement failed (API call error or validation). No position added to state. Check preceding logs.{Style.RESET_ALL}")

    def run(self) -> None:
        """Starts the main trading loop."""
        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Initiating Trading Loop (Symbol: {self.symbol}, TF: {self.timeframe}) ---{Style.RESET_ALL}")
        while True:
            self.iteration += 1
            start_time_iter = time.time()
            ts_now = pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')
            loop_prefix = f"{Fore.BLUE}===== Iteration: {self.iteration} ====={Style.RESET_ALL}"
            logger.info(f"\n{loop_prefix} Timestamp: {ts_now}")

            try:
                # --- Core Loop Steps ---
                # 1. Fetch Fresh Market Data (Price, Order Book, History)
                market_data = self._fetch_market_data()
                if market_data is None:
                    logger.warning(f"{loop_prefix} Failed to fetch market data bundle. Pausing...")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue # Skip rest of iteration
                current_price = market_data['price'] # Guaranteed non-None if market_data is not None
                # Log key fetched data early
                logger.info(f"{loop_prefix} Current Price: {self.format_price(current_price)} | "
                            f"OB Imbalance: {self.format_indicator(market_data.get('order_book_imbalance'), 3)}")

                # 2. Calculate Technical Indicators
                indicators = self._calculate_indicators(market_data['historical_data'])
                if not indicators: # Check if indicator calculation failed critically
                    logger.warning(f"{loop_prefix} Failed to calculate indicators. Pausing...")
                    time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS)
                    continue # Skip rest of iteration

                # 3. Check Status of Pending Entry Orders (Limit Orders)
                # This updates state if a pending order fills -> becomes active
                self._check_pending_entries(indicators)

                # 4. Manage Active Positions
                # - Checks for external closure (SL/TP hit, manual)
                # - Manages time-based exits
                # - Updates experimental Trailing Stop Loss
                # - Updates state for closed/modified positions
                self._manage_active_positions(current_price, indicators)

                # 5. Process Signals & Evaluate Potential New Entry
                # - Calculates signal score
                # - Checks max positions limit
                # - Calculates order size, SL/TP
                # - Places new entry order if conditions met
                # - Updates state if new position added
                self._process_signals_and_entry(market_data, indicators)

                # --- Loop Pacing ---
                exec_time = time.time() - start_time_iter
                wait_time = max(0.1, DEFAULT_SLEEP_INTERVAL_SECONDS - exec_time) # Ensure minimum wait
                logger.debug(f"{loop_prefix} Iteration completed in {exec_time:.3f}s. Waiting {wait_time:.2f}s...")
                time.sleep(wait_time)

            # --- Exception Handling ---
            except KeyboardInterrupt:
                logger.warning(f"\n{Fore.YELLOW}{Style.BRIGHT}>>> Keyboard interrupt detected. Initiating graceful shutdown... <<< {Style.RESET_ALL}")
                break # Exit the while loop
            except SystemExit as e:
                 logger.warning(f"{Fore.YELLOW}>>> SystemExit called with code {e.code}. Exiting trading loop... <<<")
                 raise e # Re-raise to be caught by main block
            except Exception as e:
                # Catch any other unexpected errors within the main loop
                logger.critical(f"{Fore.RED}{Style.BRIGHT}{loop_prefix} CRITICAL UNHANDLED ERROR in main loop: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
                # Pause for a longer duration after a critical error to prevent rapid error loops
                error_pause_seconds = 60
                logger.warning(f"{Fore.YELLOW}Pausing for {error_pause_seconds} seconds due to critical error before next attempt...{Style.RESET_ALL}")
                try:
                    time.sleep(error_pause_seconds)
                except KeyboardInterrupt: # Allow interruption during pause
                    logger.warning("\nInterrupt during error pause. Exiting loop...")
                    break

        logger.info(f"{Fore.CYAN}--- Main trading loop terminated ---{Style.RESET_ALL}")

    def shutdown(self):
        """
        Performs graceful shutdown procedures:
        - Cancels pending entry orders.
        - Optionally closes active positions based on config.
        - Saves the final state.
        - Shuts down logging.
        """
        func_name = "shutdown"
        logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Graceful Shutdown Sequence ---{Style
