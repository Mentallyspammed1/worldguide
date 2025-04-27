21s ❯ python trade_v6.4.py
  File "/data/data/com.termux/files/home/worldguide/trade_v6.4.py", line 1641
    else: buy_t=1.0/imb_t; imb_c=f"Neutral ({buy_t:.3f}-{imb_t:.3f})"; if imb<buy_t: imb_p=1.0; imb_c=f"Buy (Imb<{buy_t:.3f})" ; elif imb>imb_t: imb_p=-1.0; imb_c=f"Sell (Imb>{imb_t:.3f})"
                                                                       ^^                                                                     SyntaxError: invalid syntax

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
- Added helper methods for formatting numbers in logs.
"""

import logging
import os
import sys
import time
import json
import shutil # For state file backup and management
import math   # For decimal place calculation, isnan
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
MIN_DATAFRAME_LENGTH_WARN: int = 50  # Warn if historical data is less than this

# Position Status Constants
STATUS_PENDING_ENTRY: str = 'pending_entry' # Order placed but not yet filled
STATUS_ACTIVE: str = 'active'           # Order filled, position is live
STATUS_CLOSING: str = 'closing'         # Manual close initiated (optional use)
STATUS_CANCELED: str = 'canceled'       # Order explicitly cancelled by bot or user
STATUS_REJECTED: str = 'rejected'       # Order rejected by exchange
STATUS_EXPIRED: str = 'expired'         # Order expired (e.g., timeInForce)
STATUS_CLOSED_EXT: str = 'closed_externally' # Position closed by SL/TP/Manual/Exchange action detected
STATUS_CLOSED_ON_EXIT: str = 'closed_on_exit' # Position closed during bot shutdown
STATUS_UNKNOWN: str = 'unknown'         # Order/Position status cannot be determined

# Initialize colorama for cross-platform colored terminal output
colorama_init(autoreset=True)

# --- Centralized Logger Setup ---
logger = logging.getLogger("ScalpingBotV6") # Updated logger name
logger.setLevel(logging.DEBUG) # Set logger to lowest level (DEBUG captures everything)
# Prevent duplicate logs if root logger already has handlers (e.g., in some environments)
logger.propagate = False
# Add function name to formatter for better context
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s [%(funcName)s:%(lineno)d] - %(message)s" # Added funcName and lineno
)

# Console Handler (INFO level by default, configurable)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Default console level, overridden by config
logger.addHandler(console_handler)

# File Handler (DEBUG level for detailed logs)
file_handler: Optional[logging.FileHandler] = None
try:
    # Ensure log directory exists
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
    non-retryable exchange errors. Enhanced logging and error classification.

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
            # Check if the first argument is an instance of ScalpingBot
            instance = args[0] if args and isinstance(args[0], ScalpingBot) else None
            instance_name = instance.__class__.__name__ if instance else ''
            # Attempt to get the actual function name even if it's a method
            func_name_actual = getattr(func, '__name__', repr(func))
            func_name_log = f"{instance_name}.{func_name_actual}" if instance_name else func_name_actual


            while retries <= max_retries:
                try:
                    # Use DEBUG for attempt logging to avoid cluttering INFO
                    logger.debug(f"API Call: {func_name_log} - Attempt {retries + 1}/{max_retries + 1}")
                    result = func(*args, **kwargs)
                    # Log success at DEBUG level as well, maybe with partial result?
                    # logger.debug(f"API Call Succeeded: {func_name_log} -> {str(result)[:50]}...")
                    return result

                # --- Specific Retryable CCXT Errors ---
                except ccxt.RateLimitExceeded as e:
                    log_msg = f"Rate limit encountered ({func_name_log}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1}) Error: {e}"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                    log_msg = f"Network/Exchange issue ({func_name_log}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Authentication / Permission Errors (Usually Fatal/Non-Retryable) ---
                except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
                    log_msg = f"Authentication/Permission ritual failed ({func_name_log}: {type(e).__name__} - {e}). Check API keys, permissions, IP whitelist. Aborting call (no retry)."
                    # Use CRITICAL as this often stops the bot
                    logger.critical(f"{Fore.RED}{log_msg}{Style.RESET_ALL}")
                    return None # Indicate non-retryable failure

                # --- Exchange Errors - Distinguish Retryable vs Non-Retryable ---
                except ccxt.ExchangeError as e:
                    err_str = str(e).lower()
                    # --- Non-Retryable / Fatal Conditions ---
                    # Expanded list based on common exchange responses and logic errors
                    non_retryable_phrases = [
                        "order not found", "order does not exist", "unknown order", "no such order",
                        "order already cancel", "order was cancel", "cancel order failed", # Typo "canceled"?
                        "order is finished", "order has been filled", "already closed", "order status error",
                        "cannot be modified", "position status", "position already closed", "no position found",
                        "insufficient balance", "insufficient margin", "margin is insufficient",
                        "available balance insufficient", "insufficient funds", "account has insufficient balance",
                        "risk limit", "exceed risk limit",
                        "position side is not match", # Bybit hedge mode issue?
                        "invalid order", "parameter error", "invalid parameter", "params error",
                        "size too small", "size must be", "price too low", "invalid tif",
                        "price too high", "invalid price precision", "invalid amount precision",
                        "order cost not meet", "qty must be greater than", "must be greater than",
                        "api key is invalid", "invalid api key", "invalid sign", # Auth (redundant?)
                        "position mode not modified", "leverage not modified", # Config/Setup issues
                        "reduceonly", "reduce-only", "size would exceed", # Position logic errors
                        "bad symbol", "invalid symbol", "contract not found", # Config error
                        "account category not match", # Bybit V5 category issue
                        "order amount lower than minimum", "order price lower than minimum", # Min size/price issues
                        "invalid stop loss price", "invalid take profit price", # SL/TP issues
                        "market is closed", "instrument is closed", # Exchange state
                        "ip mismatch", "ip address not", # IP Whitelist issue
                        "system maintenance", "server maintenance", # Exchange state (might be retryable short term? treat as non-retryable for now)
                        "account not unified", "account not spot", # Bybit V5 account issues
                        "uid mismatch", # Credentials issue
                    ]
                    if any(phrase in err_str for phrase in non_retryable_phrases):
                        # Use WARNING for common "not found" / "already closed" / "insufficient" / "reduce-only"
                        # Use ERROR for potentially more serious config/logic/auth errors
                        is_common_final_state = any(p in err_str for p in ["not found", "already", "is finished", "insufficient", "reduce-only"])
                        log_level = logging.WARNING if is_common_final_state else logging.ERROR
                        log_color = Fore.YELLOW if log_level == logging.WARNING else Fore.RED
                        log_msg = f"Non-retryable ExchangeError ({func_name_log}: {type(e).__name__} - {e}). Aborting call."
                        logger.log(log_level, f"{log_color}{log_msg}{Style.RESET_ALL}")
                        # Special case: Could return a specific marker for InsufficientFunds? Could be useful.
                        # if "insufficient" in err_str: return "INSUFFICIENT_FUNDS" # Or handle in caller
                        return None # Indicate non-retryable failure

                    # --- Potentially Retryable Exchange Errors ---
                    # Temporary glitches, server issues, nonce problems etc.
                    retryable_phrases = ["nonce", "timeout", "service unavailable", "internal error", "server error", "busy", "too many visits", "connection reset by peer"]
                    if any(phrase in err_str for phrase in retryable_phrases):
                        log_msg = f"Potentially transient ExchangeError ({func_name_log}: {type(e).__name__} - {e}). Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")
                    # --- Unknown/Default Exchange Errors ---
                    else:
                        # Log as WARNING but still retry by default, could be a new transient error message
                        log_msg = f"Unclassified ExchangeError ({func_name_log}: {type(e).__name__} - {e}). Assuming transient, pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                        logger.warning(f"{Fore.YELLOW}{log_msg}{Style.RESET_ALL}")

                # --- Catch-all for other unexpected exceptions ---
                except Exception as e:
                    # Log these as ERROR with traceback for debugging
                    log_msg = f"Unexpected Python exception during API call {func_name_log}: {type(e).__name__} - {e}. Pausing {delay}s... (Attempt {retries + 1}/{max_retries + 1})"
                    logger.error(f"{Fore.RED}{log_msg}{Style.RESET_ALL}", exc_info=True)

                # --- Retry Logic ---
                if retries < max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 60) # Exponential backoff, capped at 60s
                    retries += 1
                else:
                    # Log final failure as ERROR after exhausting retries
                    logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name_log}. Spell falters. Last error: {e}{Style.RESET_ALL}")
                    return None # Indicate failure after exhausting retries

            # This line should technically not be reached if the loop logic is correct
            logger.error(f"{Fore.RED}Exited API retry loop unexpectedly for {func_name_log}{Style.RESET_ALL}")
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
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Pyrmethus Scalping Bot v6.2 Awakening ---{Style.RESET_ALL}")
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
        # Use get with default 0.0 to avoid KeyError if validation missed it
        self.limit_order_entry_offset_pct_buy: float = float(self.config["trading"].get("limit_order_offset_buy", 0.0))
        self.limit_order_entry_offset_pct_sell: float = float(self.config["trading"].get("limit_order_offset_sell", 0.0))
        self.close_positions_on_exit: bool = self.config["trading"]["close_positions_on_exit"]

        # Order Book Parameters
        self.order_book_depth: int = self.config["order_book"]["depth"]
        self.imbalance_threshold: float = float(self.config["order_book"]["imbalance_threshold"])

        # Indicator Parameters (ensure defaults are used if missing, though validate_config should catch)
        self.volatility_window: int = self.config["indicators"].get("volatility_window", 20)
        self.volatility_multiplier: float = float(self.config["indicators"].get("volatility_multiplier", 0.0))
        self.ema_period: int = self.config["indicators"].get("ema_period", 10)
        self.rsi_period: int = self.config["indicators"].get("rsi_period", 14)
        self.macd_short_period: int = self.config["indicators"].get("macd_short_period", 12)
        self.macd_long_period: int = self.config["indicators"].get("macd_long_period", 26)
        self.macd_signal_period: int = self.config["indicators"].get("macd_signal_period", 9)
        self.stoch_rsi_period: int = self.config["indicators"].get("stoch_rsi_period", 14)
        self.stoch_rsi_k_period: int = self.config["indicators"].get("stoch_rsi_k_period", 3)
        self.stoch_rsi_d_period: int = self.config["indicators"].get("stoch_rsi_d_period", 3)
        self.atr_period: int = self.config["indicators"].get("atr_period", 14)

        # Risk Management Parameters
        self.order_size_percentage: float = float(self.config["risk_management"]["order_size_percentage"])
        self.max_open_positions: int = self.config["risk_management"]["max_open_positions"]
        self.use_atr_sl_tp: bool = self.config["risk_management"]["use_atr_sl_tp"]
        self.atr_sl_multiplier: float = float(self.config["risk_management"].get("atr_sl_multiplier", ATR_MULTIPLIER_SL))
        self.atr_tp_multiplier: float = float(self.config["risk_management"].get("atr_tp_multiplier", ATR_MULTIPLIER_TP))
        self.base_stop_loss_pct: Optional[float] = float(slp) if (slp := self.config["risk_management"].get("stop_loss_percentage")) is not None else None
        self.base_take_profit_pct: Optional[float] = float(tpp) if (tpp := self.config["risk_management"].get("take_profit_percentage")) is not None else None
        # Ensure triggers are valid strings or None
        sl_trig = self.config["risk_management"].get("sl_trigger_by")
        tp_trig = self.config["risk_management"].get("tp_trigger_by")
        self.sl_trigger_by: Optional[str] = str(sl_trig) if sl_trig else None
        self.tp_trigger_by: Optional[str] = str(tp_trig) if tp_trig else None
        self.enable_trailing_stop_loss: bool = self.config["risk_management"]["enable_trailing_stop_loss"]
        self.trailing_stop_loss_percentage: Optional[float] = float(tslp) if (tslp := self.config["risk_management"].get("trailing_stop_loss_percentage")) is not None else None
        self.time_based_exit_minutes: Optional[int] = int(tbe) if (tbe := self.config["risk_management"].get("time_based_exit_minutes")) is not None else None
        self.strong_signal_adjustment_factor: float = float(self.config["risk_management"].get("strong_signal_adjustment_factor", 1.0))
        self.weak_signal_adjustment_factor: float = float(self.config["risk_management"].get("weak_signal_adjustment_factor", 1.0))

        # --- Internal Bot State ---
        self.iteration: int = 0
        self.daily_pnl: float = 0.0 # Simple daily PnL tracker, reset manually or via external script
        # Stores active and pending positions as a list of dictionaries
        # See STATUS_* constants for 'status' field values.
        # Key fields defined in `_validate_state_entry` and used throughout.
        self.open_positions: List[Dict[str, Any]] = []
        self.market_info: Optional[Dict[str, Any]] = None # Cache for market details
        self.price_decimals: int = DEFAULT_PRICE_DECIMALS
        self.amount_decimals: int = DEFAULT_AMOUNT_DECIMALS

        # --- Setup & Initialization Steps ---
        self._configure_logging_level()
        self.exchange: ccxt.Exchange = self._initialize_exchange() # Handles market loading and validation
        self._load_market_info() # Needs exchange object, calculates decimals internally
        self._load_state()       # Load persistent state after market info and validation

        # --- Log Final Operating Modes ---
        sim_color = Fore.YELLOW if self.simulation_mode else Fore.CYAN
        test_color = Fore.YELLOW if self.testnet_mode else Fore.GREEN
        logger.warning(f"{sim_color}{Style.BRIGHT}--- INTERNAL SIMULATION MODE: {self.simulation_mode} ---{Style.RESET_ALL}")
        logger.warning(f"{test_color}{Style.BRIGHT}--- EXCHANGE TESTNET MODE: {self.testnet_mode} ---{Style.RESET_ALL}")

        if not self.simulation_mode:
            if not self.testnet_mode:
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- WARNING: LIVE TRADING ON MAINNET ACTIVE ---{Style.RESET_ALL}")
                logger.warning(f"{Fore.RED}{Style.BRIGHT}--- Ensure configuration and risk parameters are correct! ---{Style.RESET_ALL}")
            else:
                logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- LIVE TRADING ON TESTNET ACTIVE ---{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}--- Running in full internal simulation mode. No real orders will be placed. ---{Style.RESET_ALL}")

        logger.info(f"{Fore.GREEN}{Style.BRIGHT}Scalping Bot V6.2 initialized. Symbol: {self.symbol}, Timeframe: {self.timeframe}{Style.RESET_ALL}")

    def _configure_logging_level(self) -> None:
        """Sets the console logging level based on the configuration file."""
        func_name = "_configure_logging_level"
        try:
            log_level_str = self.config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            # Set level on the handler, not the logger itself initially
            console_handler.setLevel(log_level)
            logger.info(f"Console logging level enchanted to: {log_level_str}")
            # Log effective levels at DEBUG for troubleshooting
            logger.debug(f"Effective Bot Logger Level: {logging.getLevelName(logger.level)}") # Will be DEBUG
            logger.debug(f"Effective Console Handler Level: {logging.getLevelName(console_handler.level)}")
            if file_handler:
                logger.debug(f"Effective File Handler Level: {logging.getLevelName(file_handler.level)}") # Will be DEBUG
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] Error configuring logging level: {e}. Using default INFO.{Style.RESET_ALL}", exc_info=True)
            console_handler.setLevel(logging.INFO)

    def _validate_state_entry(self, entry: Any, index: int) -> bool:
        """Validates a single entry from the loaded state file."""
        func_name = "_validate_state_entry"
        if not isinstance(entry, dict):
            logger.warning(f"[{func_name}] State file entry #{index+1} is not a dictionary, skipping: {str(entry)[:100]}...")
            return False

        # Minimum required keys for any state entry to be useful
        required_keys = {'id', 'symbol', 'side', 'status', 'original_size'}
        # Optional but highly recommended keys: 'size', 'entry_order_type', 'last_update_time', 'confidence'
        # Keys required if status implies an active or previously active position
        active_or_closed_required = {'entry_price', 'entry_time', 'size'}

        missing_keys = required_keys - entry.keys()
        if missing_keys:
            logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {entry.get('id', 'N/A')}) missing required keys: {missing_keys}, skipping.")
            return False

        pos_id = entry.get('id', 'N/A') # Get ID early for logging
        status = entry.get('status')

        # Check required fields for active/closed states
        if status in [STATUS_ACTIVE, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT]:
            missing_active = active_or_closed_required - entry.keys()
            if missing_active:
                 logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has status '{status}' but missing keys: {missing_active}, skipping.")
                 return False
            # Check for valid numeric values in critical fields for active/closed positions
            for key in ['entry_price', 'size']:
                 val = entry.get(key)
                 # Allow 0 size if status is closed, but not active
                 is_invalid_numeric = val is None or not isinstance(val, (float, int)) or \
                                      (status == STATUS_ACTIVE and float(val) <= 1e-12) or \
                                      (status != STATUS_ACTIVE and float(val) < 0) # Allow 0 size for closed
                 if is_invalid_numeric:
                      logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has status '{status}' but invalid {key}: {val}. Skipping.")
                      return False

        # Further type checks and conversions for numeric fields (allow flexibility but log warnings)
        # Ensure critical sizes/prices are floats
        numeric_keys_float = ['size', 'original_size', 'entry_price', 'stop_loss_price', 'take_profit_price', 'trailing_stop_price', 'exit_price']
        for key in numeric_keys_float:
             if key in entry and entry[key] is not None:
                original_value = entry[key]
                try:
                    float_val = float(original_value)
                    # Replace only if conversion changes the value significantly or type changes
                    if float_val != original_value or not isinstance(original_value, float):
                        entry[key] = float_val
                        # Log only if conversion actually happened (avoid log spam)
                        if entry[key] != original_value:
                           logger.debug(f"[{func_name}] Converted state key '{key}' for ID {pos_id} from {type(original_value)} '{original_value}' to float '{entry[key]}'.")
                except (ValueError, TypeError):
                    logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has non-convertible value for float key '{key}': {original_value}. Setting to None.")
                    entry[key] = None # Set to None if conversion fails

        # Check status validity
        valid_statuses = [STATUS_PENDING_ENTRY, STATUS_ACTIVE, STATUS_CLOSING, STATUS_CANCELED,
                          STATUS_REJECTED, STATUS_EXPIRED, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT, STATUS_UNKNOWN]
        if status not in valid_statuses:
            logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has unrecognized status '{status}'. Treating as unknown.")
            entry['status'] = STATUS_UNKNOWN # Standardize

        # Ensure side is lowercase 'buy' or 'sell'
        side = entry.get('side')
        if side and isinstance(side, str):
            entry['side'] = side.lower()
            if entry['side'] not in ['buy', 'sell']:
                 logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) has invalid side '{side}'. Skipping.")
                 return False
        elif not side:
             logger.warning(f"[{func_name}] State file entry #{index+1} (ID: {pos_id}) missing side. Skipping.")
             return False


        return True

    def _load_state(self) -> None:
        """
        Loads bot state robustly from JSON file with backup and validation.
        Retries loading from backup if primary file is corrupted.
        """
        func_name = "_load_state"
        logger.info(f"[{func_name}] Attempting to recall state from {self.state_file}...")
        state_backup_file = f"{self.state_file}.bak"
        loaded_state = None
        loaded_from = "N/A"

        # --- Attempt 1: Load main state file ---
        if os.path.exists(self.state_file):
            try:
                file_size = os.path.getsize(self.state_file)
                if file_size == 0:
                    logger.warning(f"{Fore.YELLOW}[{func_name}] State file {self.state_file} is empty. Will check backup or start fresh.{Style.RESET_ALL}")
                    loaded_from = "empty_main"
                else:
                    with open(self.state_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            logger.warning(f"{Fore.YELLOW}[{func_name}] State file {self.state_file} contains only whitespace. Will check backup or start fresh.{Style.RESET_ALL}")
                            loaded_from = "whitespace_main"
                        else:
                            # Attempt to load and validate
                            saved_state_raw = json.loads(content)
                            if isinstance(saved_state_raw, list):
                                valid_positions = []
                                invalid_count = 0
                                for i, pos_data in enumerate(saved_state_raw):
                                    if self._validate_state_entry(pos_data, i): # Validation happens here
                                        valid_positions.append(pos_data)
                                    else:
                                        invalid_count += 1
                                loaded_state = valid_positions # Store valid positions
                                loaded_from = "main_file"
                                log_msg = f"[{func_name}] Recalled {len(loaded_state)} valid position(s) from {self.state_file}."
                                if invalid_count > 0:
                                    log_msg += f" Skipped {invalid_count} invalid entries."
                                logger.info(f"{Fore.GREEN}{log_msg}{Style.RESET_ALL}")
                                # Try to remove old backup if main load was successful
                                if os.path.exists(state_backup_file):
                                     try:
                                         os.remove(state_backup_file)
                                         logger.debug(f"[{func_name}] Removed old state backup file: {state_backup_file}")
                                     except OSError as remove_err:
                                         logger.warning(f"[{func_name}] Could not remove old state backup {state_backup_file}: {remove_err}")
                            else:
                                # Raise error if format is wrong (not a list)
                                raise ValueError(f"Invalid state format - expected a list, got {type(saved_state_raw)}.")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"{Fore.RED}[{func_name}] Error decoding/validating primary state file {self.state_file}: {e}. Attempting recovery from backup...{Style.RESET_ALL}")
                loaded_from = "corrupt_main"
                # Backup the corrupted file before trying backup
                if os.path.exists(self.state_file):
                     try:
                         corrupted_path = f"{self.state_file}.corrupted_{int(time.time())}"
                         shutil.copy2(self.state_file, corrupted_path)
                         logger.warning(f"[{func_name}] Backed up corrupted state file to {corrupted_path}")
                     except Exception as backup_err:
                          logger.error(f"[{func_name}] Could not back up corrupted state file {self.state_file}: {backup_err}")
            except Exception as e:
                 logger.error(f"{Fore.RED}[{func_name}] Unexpected error reading primary state file {self.state_file}: {e}{Style.RESET_ALL}", exc_info=True)
                 loaded_from = "read_error_main"

        else: # Main file doesn't exist
            logger.info(f"[{func_name}] No prior state file found ({self.state_file}). Checking for backup...")
            loaded_from = "no_main_file"

        # --- Attempt 2: Recovery from backup if primary load failed or file was missing/empty ---
        if loaded_state is None: # Only try backup if primary load didn't succeed
            if os.path.exists(state_backup_file):
                logger.warning(f"{Fore.YELLOW}[{func_name}] Attempting to restore state from backup: {state_backup_file}{Style.RESET_ALL}")
                try:
                    # Validate backup file content before attempting to use it
                    file_size_bak = os.path.getsize(state_backup_file)
                    if file_size_bak == 0:
                         logger.warning(f"{Fore.YELLOW}[{func_name}] Backup state file {state_backup_file} is empty. Starting fresh.{Style.RESET_ALL}")
                         loaded_from = "empty_backup"
                    else:
                        with open(state_backup_file, 'r', encoding='utf-8') as f_bak:
                             content_bak = f_bak.read()
                             if not content_bak.strip():
                                 logger.warning(f"{Fore.YELLOW}[{func_name}] Backup state file {state_backup_file} contains only whitespace. Starting fresh.{Style.RESET_ALL}")
                                 loaded_from = "whitespace_backup"
                             else:
                                saved_state_raw_bak = json.loads(content_bak)
                                if isinstance(saved_state_raw_bak, list):
                                    valid_positions_bak = []
                                    invalid_count_bak = 0
                                    for i, pos_data_bak in enumerate(saved_state_raw_bak):
                                        if self._validate_state_entry(pos_data_bak, i):
                                            valid_positions_bak.append(pos_data_bak)
                                        else:
                                            invalid_count_bak += 1
                                    loaded_state = valid_positions_bak # Use validated backup state
                                    loaded_from = "backup_file"
                                    log_msg = f"[{func_name}] Successfully restored and validated {len(loaded_state)} position(s) from backup {state_backup_file}."
                                    if invalid_count_bak > 0: log_msg += f" Skipped {invalid_count_bak} invalid entries from backup."
                                    logger.info(f"{Fore.GREEN}{log_msg}{Style.RESET_ALL}")
                                    # Restore backup to main file location for consistency
                                    try:
                                        shutil.copy2(state_backup_file, self.state_file)
                                        logger.info(f"[{func_name}] Restored validated backup content to main state file: {self.state_file}")
                                    except Exception as copy_err:
                                        logger.error(f"[{func_name}] Failed to copy validated backup to main file: {copy_err}. Using state loaded from backup directly.")
                                else:
                                    raise ValueError(f"Invalid backup state format - expected list, got {type(saved_state_raw_bak)}.")
                except (json.JSONDecodeError, ValueError, TypeError) as e_bak:
                    logger.error(f"{Fore.RED}[{func_name}] Error decoding/validating backup state file {state_backup_file}: {e_bak}. Starting fresh.{Style.RESET_ALL}")
                    loaded_from = "corrupt_backup"
                except Exception as e_bak:
                    logger.error(f"{Fore.RED}[{func_name}] Failed to read/process state backup {state_backup_file}: {e_bak}. Starting fresh.{Style.RESET_ALL}", exc_info=True)
                    loaded_from = "read_error_backup"
            else:
                # No main file (or bad main) AND no backup file exists
                logger.warning(f"{Fore.YELLOW}[{func_name}] No valid primary state or backup file found ({loaded_from}). Starting with a fresh state.{Style.RESET_ALL}")
                loaded_from = "fresh_start"

        # --- Final State Assignment ---
        if loaded_state is None:
            # If state is still None after all attempts, initialize empty list
            self.open_positions = []
            logger.info(f"[{func_name}] Initialized with empty position state (Reason: {loaded_from}).")
        else:
            # Assign the successfully loaded (and validated) state
            self.open_positions = loaded_state

        # --- Ensure State File Exists ---
        # Save the current state (empty or loaded) to ensure the file exists for the next run.
        # This also handles the case where the initial file was missing or empty.
        logger.debug(f"[{func_name}] Ensuring state file exists and reflects current state ({len(self.open_positions)} positions). Saving...")
        self._save_state()

    def _save_state(self) -> None:
        """
        Saves the current bot state (list of open positions) atomically.
        Uses tempfile, os.replace, and creates a backup. Includes serialization fixes.
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
            # Create temp file securely in the target directory to ensure atomic replace works
            temp_fd, temp_path = tempfile.mkstemp(dir=state_dir, prefix=os.path.basename(self.state_file) + '.tmp_')
            logger.debug(f"[{func_name}] Saving state ({len(self.open_positions)} positions) via temp file {temp_path}...")

            # Prepare state data for JSON serialization (handle numpy types, Pandas NaT/NaN, etc.)
            def json_serializer(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    # Handle potential numpy NaN/Inf explicitly
                    if np.isnan(obj): return None # Serialize NaN as null
                    if np.isinf(obj): return str(obj) # Serialize Inf as "Infinity" or "-Infinity"
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)): # Handle arrays if they sneak in
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif pd.isna(obj): # Handle Pandas NaT or NaN -> convert to None for JSON
                    return None
                # Consider adding datetime serialization if needed:
                # elif isinstance(obj, datetime.datetime): return obj.isoformat()
                # Fallback: try converting to string, log warning if this happens unexpectedly
                try:
                    # Don't convert basic types already handled by json
                    if isinstance(obj, (str, int, float, bool, type(None))):
                        raise TypeError(f"Object of type {type(obj)} should be handled by default JSON encoder.")
                    obj_str = str(obj)
                    logger.warning(f"[{func_name}] Used fallback string serialization for object of type {type(obj)}: {obj_str[:100]}...")
                    return obj_str
                except Exception as serialize_err:
                    logger.error(f"[{func_name}] Could not serialize object of type {type(obj)}: {serialize_err}. Value: {str(obj)[:100]}...")
                    return None # Represent un-serializable objects as null

            # Dump state to temporary file using the file descriptor
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                # Use allow_nan=False to force our custom serializer for NaN/Inf
                json.dump(self.open_positions, f, indent=4, default=json_serializer, allow_nan=False)
            temp_fd = None # fd is now closed by the 'with' block

            # Create Backup of Current State File (if it exists) before replacing
            if os.path.exists(self.state_file):
                try:
                    shutil.copy2(self.state_file, state_backup_file) # copy2 preserves metadata
                    logger.debug(f"[{func_name}] Created state backup: {state_backup_file}")
                except Exception as backup_err:
                    logger.warning(f"[{func_name}] Could not create state backup {state_backup_file}: {backup_err}. Proceeding cautiously.")

            # Atomically Replace the Old State File with the New One (should work if temp is on same filesystem)
            os.replace(temp_path, self.state_file)
            temp_path = None # Reset temp_path as it's been moved/renamed
            logger.debug(f"[{func_name}] State recorded successfully to {self.state_file}")

        except (IOError, OSError, TypeError) as e:
            logger.error(f"{Fore.RED}[{func_name}] Could not scribe state to {self.state_file} (Error Type: {type(e).__name__}): {e}{Style.RESET_ALL}", exc_info=True)
        except Exception as e:
            logger.error(f"{Fore.RED}[{func_name}] An unexpected error occurred while recording state: {e}{Style.RESET_ALL}", exc_info=True)
        finally:
            # Clean up temp file if it still exists (i.e., if error occurred before os.replace or during close)
            if temp_fd is not None: # If fd was opened but not closed by 'with' (e.g., error during dump)
                try: os.close(temp_fd); logger.debug(f"[{func_name}] Closed dangling temp fd {temp_fd}")
                except OSError: pass # Ignore errors closing already closed fd
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
            # --- DEBUG: Print path before loading ---
            logger.debug(f"--- [{func_name}] Attempting to load config from: {os.path.abspath(config_file)} ---")
            if not os.path.exists(config_file):
                 logger.critical(f"--- [{func_name}] FATAL: Config file DOES NOT EXIST at {os.path.abspath(config_file)} ---")
                 # Try creating default, then exit
                 try: self.create_default_config(config_file)
                 except Exception as create_e: logger.error(f"Failed creating default: {create_e}")
                 sys.exit(1)
            # --- END DEBUG ---

            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            # --- DEBUG: Print loaded config ---
            # logger.debug("--- DEBUG: Config loaded by PyYAML ---")
            # import pprint
            # pprint.pprint(self.config)
            # logger.debug("------------------------------------")
            # --- END DEBUG ---

            if not isinstance(self.config, dict):
                 logger.critical(f"{Fore.RED}[{func_name}] Fatal: Config file {config_file} has invalid structure (must be a dictionary, not {type(self.config)}). Aborting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.info(f"{Fore.GREEN}[{func_name}] Configuration spellbook loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError: # Should be caught by exists check now, but keep for safety
            logger.warning(f"{Fore.YELLOW}[{func_name}] Configuration spellbook '{config_file}' not found.{Style.RESET_ALL}")
            try:
                self.create_default_config(config_file)
            except Exception as e:
                logger.error(f"{Fore.RED}[{func_name}] Failed to create default config: {e}{Style.RESET_ALL}", exc_info=True)
                logger.critical(f"{Fore.RED}[{func_name}] Exiting due to failure creating default config.{Style.RESET_ALL}")
                sys.exit(1)
            logger.critical(f"{Fore.YELLOW}[{func_name}] Exiting after creating default config. Please review '{config_file}' (especially API keys if not using .env) and restart.{Style.RESET_ALL}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.critical(f"{Fore.RED}[{func_name}] Fatal: Error parsing spellbook {config_file}: {e}. Check YAML syntax (indentation!). Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Fatal: Unexpected chaos loading configuration: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def create_default_config(self, config_file: str) -> None:
        """Creates a default configuration file with explanations."""
        func_name = "create_default_config"
        logger.info(f"[{func_name}] Crafting default spellbook: {config_file}...")
        # Default config structure with comments (as used in v6.2)
        default_config = {
            "# === Logging Configuration ===" : None,
            "logging": {
                "level": "INFO", # Logging level for console (DEBUG, INFO, WARNING, ERROR, CRITICAL). File logs are always DEBUG.
            },
            "# === Exchange Configuration ===" : None,
            "exchange": {
                "exchange_id": DEFAULT_EXCHANGE_ID, # Exchange name (lowercase, e.g., 'bybit', 'binance', 'okx')
                "testnet_mode": True, # Use exchange's testnet/sandbox environment? (Requires Testnet API Keys)
                # --- API Keys ---
                # BEST PRACTICE: Set API keys via environment variables:
                # BYBIT_API_KEY=your_key
                # BYBIT_API_SECRET=your_secret
                # Or, uncomment and add here (LESS SECURE):
                # "api_key": "YOUR_API_KEY_HERE",
                # "api_secret": "YOUR_API_SECRET_HERE",
            },
            "# === Trading Parameters ===" : None,
            "trading": {
                "symbol": "BTC/USDT:USDT", # Trading pair (check CCXT format for your exchange, e.g., Bybit Linear Perpetual)
                "timeframe": DEFAULT_TIMEFRAME, # Candle timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
                "simulation_mode": True, # Internal simulation only (no real orders placed, ignores testnet_mode for orders)
                "entry_order_type": "limit", # 'market' or 'limit' for entry orders.
                "limit_order_offset_buy": 0.0005, # Place buy limit 0.05% BELOW current market price. Set 0 for exact price.
                "limit_order_offset_sell": 0.0005, # Place sell limit 0.05% ABOVE current market price. Set 0 for exact price.
                "close_positions_on_exit": False, # Attempt to MARKET CLOSE active positions on bot shutdown (Ctrl+C)?
            },
            "# === Order Book Analysis ===" : None,
            "order_book": {
                "depth": 10, # Number of bids/asks levels to fetch from order book.
                "imbalance_threshold": 1.5, # Ask/Bid volume ratio threshold for signal generation.
            },
            "# === Technical Indicator Parameters ===" : None,
            "indicators": {
                "volatility_window": 20,
                "volatility_multiplier": 0.0,
                "ema_period": 10,
                "rsi_period": 14,
                "macd_short_period": 12,
                "macd_long_period": 26,
                "macd_signal_period": 9,
                "stoch_rsi_period": 14,
                "stoch_rsi_k_period": 3,
                "stoch_rsi_d_period": 3,
                "atr_period": 14,
            },
            "# === Risk Management ===" : None,
            "risk_management": {
                "order_size_percentage": 0.02,
                "max_open_positions": 1,
                "use_atr_sl_tp": True,
                "atr_sl_multiplier": ATR_MULTIPLIER_SL,
                "atr_tp_multiplier": ATR_MULTIPLIER_TP,
                "stop_loss_percentage": 0.005,
                "take_profit_percentage": 0.01,
                "sl_trigger_by": "MarkPrice",
                "tp_trigger_by": "MarkPrice",
                "enable_trailing_stop_loss": False,
                "trailing_stop_loss_percentage": 0.003,
                "time_based_exit_minutes": 60,
                "strong_signal_adjustment_factor": 1.0,
                "weak_signal_adjustment_factor": 1.0,
            },
        }

        # --- Environment Variable Overrides --- (Example for order size)
        env_order_pct_str = os.getenv("ORDER_SIZE_PERCENTAGE")
        if env_order_pct_str:
            try:
                override_val = float(env_order_pct_str)
                if 0 < override_val <= 1.0:
                    default_config["risk_management"]["order_size_percentage"] = override_val
                    logger.info(f"[{func_name}] Overriding risk_management.order_size_percentage from environment variable to {override_val:.4f}.")
                else: logger.warning(f"[{func_name}] Invalid ORDER_SIZE_PERCENTAGE ('{env_order_pct_str}') in environment. Must be > 0 and <= 1.")
            except ValueError: logger.warning(f"[{func_name}] Invalid ORDER_SIZE_PERCENTAGE ('{env_order_pct_str}') in environment. Not a valid number.")

        try:
            config_dir = os.path.dirname(config_file)
            if config_dir: os.makedirs(config_dir, exist_ok=True)
            # Write default config, including comments if possible (simple write for now)
            with open(config_file, "w", encoding='utf-8') as f:
                # Manual comments (better than nothing)
                f.write("# Configuration Spellbook for Pyrmethus Scalping Bot v6.2\n\n")
                # Dump section by section to control order slightly? No, let dump handle it.
                yaml.dump(default_config, f, indent=4, sort_keys=False, default_flow_style=False, allow_unicode=True)

            logger.info(f"{Fore.YELLOW}[{func_name}] A default spellbook has been crafted: '{config_file}'. Please review and configure.{Style.RESET_ALL}")

        except IOError as e:
            logger.error(f"{Fore.RED}[{func_name}] Could not scribe default spellbook {config_file}: {e}{Style.RESET_ALL}", exc_info=True)
            raise

    def validate_config(self) -> None:
        """Performs detailed validation of the loaded configuration parameters."""
        # (Implementation is the same as provided in v6.2 - extensive checks)
        func_name = "validate_config"
        logger.debug(f"[{func_name}] Scrutinizing the configuration spellbook...")
        errors = [] # Collect all errors before exiting

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
                        else: raise KeyError # Current level is not a dict, but key was expected
                except KeyError:
                    if default is not KeyError: return default
                    errors.append(f"Missing required configuration key: '{full_key_path}'")
                    return None # Return None on missing key to allow collection of multiple errors
                except Exception as e: # Catch unexpected errors during access
                     errors.append(f"Error accessing config key '{full_key_path}': {e}")
                     return None

                # Type and None checks
                if value is None:
                    if allow_none: return None
                    else: errors.append(f"Configuration key '{full_key_path}' cannot be null/None."); return None
                if req_type is not None and not isinstance(value, req_type):
                     # Special case: allow int where float is expected, perform conversion
                     if isinstance(req_type, tuple) and float in req_type and isinstance(value, int):
                          return float(value)
                     if req_type is float and isinstance(value, int):
                          return float(value)
                     errors.append(f"Configuration key '{full_key_path}' expects type {req_type} but got {type(value)}.")
                     return None
                return value

            def _check_range(value: Optional[Union[int, float]], key_path: str, min_val: Optional[Union[int, float]] = None, max_val: Optional[Union[int, float]] = None, min_exclusive: bool = False, max_exclusive: bool = False):
                """Checks if value is within the specified range. Appends to errors list."""
                if value is None: return # Cannot check range if value is None (handled by _get_nested)
                try:
                    val_f = float(value) # Convert to float for comparison
                    if min_val is not None:
                        op_str = ">" if min_exclusive else ">="
                        if not (val_f > float(min_val) if min_exclusive else val_f >= float(min_val)):
                            errors.append(f"Configuration key '{key_path}' ({value}) must be {op_str} {min_val}")
                    if max_val is not None:
                        op_str = "<" if max_exclusive else "<="
                        if not (val_f < float(max_val) if max_exclusive else val_f <= float(max_val)):
                            errors.append(f"Configuration key '{key_path}' ({value}) must be {op_str} {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"Cannot perform range check: Configuration key '{key_path}' has non-numeric value '{value}'")


            # --- Validate Sections ---
            required_sections = ["logging", "exchange", "trading", "order_book", "indicators", "risk_management"]
            for section in required_sections:
                 if section not in self.config or not isinstance(self.config.get(section), dict):
                      errors.append(f"Config Error: Missing or invalid section '{section}'. Must be a dictionary.")
                      self.config[section] = {} # Add empty dict to prevent downstream KeyErrors

            # --- Logging ---
            log_level = _get_nested(self.config, ["level"], "logging", default="INFO", req_type=str).upper()
            if log_level and not hasattr(logging, log_level): errors.append(f"logging.level: Invalid level '{log_level}'. Use DEBUG, INFO, WARNING, ERROR, CRITICAL.")

            # --- Exchange ---
            ex_id = _get_nested(self.config, ["exchange_id"], "exchange", req_type=str)
            if ex_id and ex_id not in ccxt.exchanges: errors.append(f"exchange.exchange_id: Invalid or unsupported '{ex_id}'. See `ccxt.exchanges`.")
            _get_nested(self.config, ["testnet_mode"], "exchange", req_type=bool)

            # --- Trading ---
            _get_nested(self.config, ["symbol"], "trading", req_type=str)
            _get_nested(self.config, ["timeframe"], "trading", req_type=str)
            _get_nested(self.config, ["simulation_mode"], "trading", req_type=bool)
            entry_type = _get_nested(self.config, ["entry_order_type"], "trading", req_type=str)
            if entry_type and entry_type.lower() not in ["market", "limit"]: errors.append(f"trading.entry_order_type: Must be 'market' or 'limit'.")
            _check_range(_get_nested(self.config, ["limit_order_offset_buy"], "trading", req_type=(float, int)), "trading.limit_order_offset_buy", min_val=0)
            _check_range(_get_nested(self.config, ["limit_order_offset_sell"], "trading", req_type=(float, int)), "trading.limit_order_offset_sell", min_val=0)
            _get_nested(self.config, ["close_positions_on_exit"], "trading", req_type=bool)

            # --- Order Book ---
            _check_range(_get_nested(self.config, ["depth"], "order_book", req_type=int), "order_book.depth", min_val=1, max_val=1000)
            _check_range(_get_nested(self.config, ["imbalance_threshold"], "order_book", req_type=(float, int)), "order_book.imbalance_threshold", min_val=0, min_exclusive=True)

            # --- Indicators ---
            periods = ["volatility_window", "ema_period", "rsi_period", "macd_short_period",
                       "macd_long_period", "macd_signal_period", "stoch_rsi_period",
                       "stoch_rsi_k_period", "stoch_rsi_d_period", "atr_period"]
            for p in periods: _check_range(_get_nested(self.config, [p], "indicators", req_type=int), f"indicators.{p}", min_val=1)
            _check_range(_get_nested(self.config, ["volatility_multiplier"], "indicators", req_type=(float, int)), "indicators.volatility_multiplier", min_val=0)
            macd_s = _get_nested(self.config, ["macd_short_period"], "indicators", req_type=int)
            macd_l = _get_nested(self.config, ["macd_long_period"], "indicators", req_type=int)
            if macd_s is not None and macd_l is not None and macd_s >= macd_l:
                 errors.append("indicators: macd_short_period must be strictly less than macd_long_period.")

            # --- Risk Management ---
            use_atr = _get_nested(self.config, ["use_atr_sl_tp"], "risk_management", req_type=bool)
            if use_atr is not None:
                 if use_atr:
                    _check_range(_get_nested(self.config, ["atr_sl_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_sl_multiplier", min_val=0, min_exclusive=True)
                    _check_range(_get_nested(self.config, ["atr_tp_multiplier"], "risk_management", req_type=(float, int)), "risk_management.atr_tp_multiplier", min_val=0, min_exclusive=True)
                 else:
                    sl_pct = _get_nested(self.config, ["stop_loss_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                    tp_pct = _get_nested(self.config, ["take_profit_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                    if sl_pct is None and tp_pct is None:
                         logger.warning(f"{Fore.YELLOW}[{func_name}] risk_management: use_atr_sl_tp is false, and both stop_loss_percentage and take_profit_percentage are null/missing. No SL/TP will be used.{Style.RESET_ALL}")
                    if sl_pct is not None: _check_range(sl_pct, "risk_management.stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)
                    if tp_pct is not None: _check_range(tp_pct, "risk_management.take_profit_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            valid_triggers = ["MarkPrice", "LastPrice", "IndexPrice", None]
            sl_trig = _get_nested(self.config, ["sl_trigger_by"], "risk_management", default=None, req_type=str, allow_none=True)
            tp_trig = _get_nested(self.config, ["tp_trigger_by"], "risk_management", default=None, req_type=str, allow_none=True)
            if sl_trig is not None and sl_trig not in valid_triggers: errors.append(f"risk_management.sl_trigger_by: Invalid trigger '{sl_trig}'. Valid: {valid_triggers}")
            if tp_trig is not None and tp_trig not in valid_triggers: errors.append(f"risk_management.tp_trigger_by: Invalid trigger '{tp_trig}'. Valid: {valid_triggers}")

            enable_tsl = _get_nested(self.config, ["enable_trailing_stop_loss"], "risk_management", req_type=bool)
            if enable_tsl is not None and enable_tsl:
                tsl_pct = _get_nested(self.config, ["trailing_stop_loss_percentage"], "risk_management", req_type=(float, int), allow_none=True)
                if tsl_pct is None: errors.append("risk_management.trailing_stop_loss_percentage must be set if enable_trailing_stop_loss is true.")
                else: _check_range(tsl_pct, "risk_management.trailing_stop_loss_percentage", min_val=0, max_val=1, min_exclusive=True, max_exclusive=True)

            order_size_pct = _get_nested(self.config, ["order_size_percentage"], "risk_management", req_type=(float, int))
            _check_range(order_size_pct, "risk_management.order_size_percentage", min_val=0, max_val=1.0, min_exclusive=True, max_exclusive=False) # Allow up to 100%
            _check_range(_get_nested(self.config, ["max_open_positions"], "risk_management", req_type=int), "risk_management.max_open_positions", min_val=1)
            time_exit = _get_nested(self.config, ["time_based_exit_minutes"], "risk_management", req_type=int, allow_none=True)
            if time_exit is not None: _check_range(time_exit, "risk_management.time_based_exit_minutes", min_val=0) # Allow 0 to disable cleanly
            _check_range(_get_nested(self.config, ["strong_signal_adjustment_factor"], "risk_management", default=1.0, req_type=(float, int)), "risk_management.strong_signal_adjustment_factor", min_val=0, min_exclusive=True)
            _check_range(_get_nested(self.config, ["weak_signal_adjustment_factor"], "risk_management", default=1.0, req_type=(float, int)), "risk_management.weak_signal_adjustment_factor", min_val=0, min_exclusive=True)


            # --- Final Check ---
            if errors:
                logger.critical(f"{Fore.RED}{Style.BRIGHT}[{func_name}] Configuration flaws detected ({len(errors)}):{Style.RESET_ALL}")
                for i, error in enumerate(errors):
                    logger.critical(f"{Fore.RED}  {i+1}. {error}{Style.RESET_ALL}")
                logger.critical(f"{Fore.RED}Mend the '{CONFIG_FILE_NAME}' scroll. Aborting.{Style.RESET_ALL}")
                sys.exit(1)
            else:
                logger.info(f"{Fore.GREEN}[{func_name}] Configuration spellbook deemed valid and potent.{Style.RESET_ALL}")

        except Exception as e: # Catch errors during the validation process itself
            logger.critical(f"{Fore.RED}[{func_name}] Unexpected chaos during configuration validation: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _initialize_exchange(self) -> ccxt.Exchange:
        """Initializes and configures the CCXT exchange instance."""
        # (Implementation is the same as provided in v6.2 - includes market loading and basic validation)
        func_name = "_initialize_exchange"
        logger.info(f"[{func_name}] Summoning exchange spirits for {self.exchange_id.upper()}...")

        creds_found = self.api_key and self.api_secret
        if not self.simulation_mode and not creds_found:
             logger.critical(f"{Fore.RED}[{func_name}] API Key/Secret essence missing (check .env or config). Cannot trade live/testnet without credentials. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        elif creds_found and self.simulation_mode:
             logger.warning(f"{Fore.YELLOW}[{func_name}] API Key/Secret found, but internal simulation_mode is True. Credentials will NOT be used for placing orders.{Style.RESET_ALL}")
        elif not creds_found and self.simulation_mode:
             logger.info(f"{Fore.CYAN}[{func_name}] Running in internal simulation mode. API credentials not required/found.{Style.RESET_ALL}")

        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_config = {
                'enableRateLimit': True, # Enable built-in rate limiting
                'options': {
                    # Try to set default market type based on symbol format (e.g., ':USDT' implies linear swap)
                    'defaultType': 'swap' if ':' in self.symbol else 'spot',
                    'adjustForTimeDifference': True, # Auto-adjust for time skew
                    # Bybit V5 options: Set default category based on symbol type?
                    # 'defaultSubType': 'linear' if ':USDT' in self.symbol else ('inverse' if ':USD' in self.symbol else None),
                },
                # Set newUpdates to False if state tracking relies solely on fetch_order/fetch_position calls
                # 'newUpdates': False, # Default is True, might be useful for websocket updates later
            }
            # Add credentials ONLY if NOT in internal simulation mode
            if not self.simulation_mode:
                exchange_config['apiKey'] = self.api_key
                exchange_config['secret'] = self.api_secret
                logger.debug(f"[{func_name}] API credentials loaded into exchange config.")
            else: # Add dummy keys if simulating, CCXT sometimes requires them, avoids certain errors
                exchange_config['apiKey'] = 'SIMULATED_KEY_' + str(int(time.time())) # Make slightly unique
                exchange_config['secret'] = 'SIMULATED_SECRET'
                logger.debug(f"[{func_name}] Using dummy API credentials for simulation mode.")

            exchange = exchange_class(exchange_config)

            # Set testnet mode using CCXT unified method (only if NOT simulating internally)
            if not self.simulation_mode and self.testnet_mode:
                logger.info(f"[{func_name}] Attempting to enable exchange sandbox mode (testnet)...")
                try:
                    # Check if the method exists before calling
                    if hasattr(exchange, 'set_sandbox_mode') and callable(exchange.set_sandbox_mode):
                        exchange.set_sandbox_mode(True)
                        # Verify if it likely worked by checking the API URL used by the instance
                        urls = exchange.urls.get('api', {})
                        # Get any relevant URL (public, private, specific market type)
                        api_url = urls.get('public') or urls.get('private') or urls.get('swap') or urls.get('linear') or str(urls)
                        testnet_indicators = ['test', 'sandbox', 'demo', 'uat']
                        if any(indicator in str(api_url).lower() for indicator in testnet_indicators):
                            logger.info(f"{Fore.YELLOW}[{func_name}] Exchange sandbox mode explicitly enabled via CCXT. API URL seems correct: {api_url}{Style.RESET_ALL}")
                        else:
                             logger.warning(f"{Fore.YELLOW}[{func_name}] CCXT set_sandbox_mode(True) called, but API URL ({api_url}) doesn't clearly indicate testnet. Testnet activation might depend solely on API key type for {self.exchange_id}.{Style.RESET_ALL}")
                    else:
                        logger.warning(f"{Fore.YELLOW}[{func_name}] Exchange instance for {self.exchange_id} does not have 'set_sandbox_mode' method. Testnet depends on API key type/config URL.{Style.RESET_ALL}")

                except ccxt.NotSupported: # Should be caught by hasattr check, but belt-and-suspenders
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Unified set_sandbox_mode is NotSupported by {self.exchange_id} via CCXT. Testnet depends on API key type/URL.{Style.RESET_ALL}")
                except Exception as sandbox_err:
                    logger.error(f"{Fore.RED}[{func_name}] Error trying to set sandbox mode: {sandbox_err}. Continuing, but testnet may not be active.{Style.RESET_ALL}")
            elif not self.simulation_mode and not self.testnet_mode:
                 logger.info(f"{Fore.GREEN}[{func_name}] Exchange sandbox mode is OFF. Operating on Mainnet.{Style.RESET_ALL}")

            # Load markets (needed for symbol validation, precision, limits, etc.)
            # Use retry decorator for loading markets as it's a critical API call
            @retry_api_call(max_retries=2, initial_delay=5)
            def _load_markets_with_retry(exch_instance):
                 logger.debug(f"[{func_name}] Loading market matrix...")
                 return exch_instance.load_markets(reload=True) # Force reload

            markets = _load_markets_with_retry(exchange)
            if markets is None:
                 logger.critical(f"{Fore.RED}[{func_name}] Failed to load markets from {self.exchange_id} after retries. Cannot proceed. Aborting.{Style.RESET_ALL}")
                 sys.exit(1)
            logger.debug(f"[{func_name}] Market matrix loaded ({len(markets)} markets).")

            # --- Validate Symbol and Timeframe against Exchange Data ---
            if self.symbol not in exchange.markets:
                available_sym = list(exchange.markets.keys())[:15] # Show some examples
                logger.critical(f"{Fore.RED}[{func_name}] Symbol '{self.symbol}' not found on {self.exchange_id}. Available examples: {available_sym}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)

            market_details = exchange.market(self.symbol) # Store for use after validation
            self.market_info = market_details # Cache the basic market info immediately

            if 'timeframes' in exchange.has and exchange.has['timeframes'] and self.timeframe not in exchange.timeframes:
                available_tf = list(exchange.timeframes.keys())[:15]
                logger.critical(f"{Fore.RED}[{func_name}] Timeframe '{self.timeframe}' unsupported by {self.exchange_id}. Available examples: {available_tf}... Aborting.{Style.RESET_ALL}")
                sys.exit(1)
            elif 'timeframes' not in exchange.has or not exchange.has['timeframes']:
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Exchange does not list supported timeframes via CCXT. Assuming '{self.timeframe}' is valid.{Style.RESET_ALL}")

            if not market_details.get('active', True):
                 logger.warning(f"{Fore.YELLOW}[{func_name}] Warning: Market '{self.symbol}' is marked as inactive by the exchange.{Style.RESET_ALL}")

            logger.info(f"[{func_name}] Symbol '{self.symbol}' and timeframe '{self.timeframe}' confirmed available (or assumed).")

            # Perform initial API connectivity test (if not simulating)
            if not self.simulation_mode:
                logger.debug(f"[{func_name}] Performing initial API connectivity test (fetch balance)...")
                quote_currency = market_details.get('quote')
                if not quote_currency:
                     logger.critical(f"{Fore.RED}[{func_name}] Could not determine quote currency from market info for balance check. Aborting.{Style.RESET_ALL}")
                     sys.exit(1) # Need quote currency for balance check and sizing

                # Use fetch_balance as a more comprehensive initial test (includes auth)
                initial_balance_check = self.fetch_balance(currency_code=quote_currency)
                if initial_balance_check is not None: # fetch_balance returns float or None on error
                     logger.info(f"{Fore.GREEN}[{func_name}] API connection and authentication successful (Balance Check OK).{Style.RESET_ALL}")
                else:
                     # Fetch_balance logs specific errors, just add a critical failure message here
                     logger.critical(f"{Fore.RED}[{func_name}] Initial API connectivity/auth test (fetch_balance) failed. Check logs (esp. API key validity, permissions, IP whitelist). Aborting.{Style.RESET_ALL}")
                     sys.exit(1)

            logger.info(f"{Fore.GREEN}{Style.BRIGHT}[{func_name}] Exchange spirits aligned for {self.exchange_id.upper()}.{Style.RESET_ALL}")
            return exchange

        except ccxt.AuthenticationError as e:
             logger.critical(f"{Fore.RED}[{func_name}] Authentication failed for {self.exchange_id}: {e}. Check API keys/permissions/IP whitelist. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except (ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.NetworkError) as e:
             logger.critical(f"{Fore.RED}[{func_name}] Connection failed to {self.exchange_id} ({type(e).__name__}): {e}. Check network/exchange status. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except ccxt.BadSymbol as e: # Catch BadSymbol specifically if it occurs during init/market loading
             logger.critical(f"{Fore.RED}[{func_name}] BadSymbol error during exchange initialization or market loading for '{self.symbol}': {e}. Check symbol format. Aborting.{Style.RESET_ALL}")
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Unexpected error initializing exchange {self.exchange_id}: {e}{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    def _calculate_decimal_places(self, tick_size: Optional[Union[float, int, str]]) -> int:
        """
        Calculates decimal places from tick size (e.g., 0.01 -> 2). Handles various formats.
        Returns a default value if calculation fails or tick_size is invalid.
        """
        # (Implementation is the same as provided in v6.2)
        func_name = "_calculate_decimal_places"
        if tick_size is None: return -1

        try:
            if isinstance(tick_size, str):
                try: tick_size = float(tick_size)
                except ValueError: logger.warning(f"[{func_name}] Invalid tick size string '{tick_size}'."); return -1
            if not isinstance(tick_size, (float, int)): logger.warning(f"[{func_name}] Invalid tick size type {type(tick_size)}."); return -1
            if tick_size <= 0: logger.warning(f"[{func_name}] Non-positive tick size {tick_size}."); return -1

            if isinstance(tick_size, int) or tick_size.is_integer(): return 0

            s = format(tick_size, '.16f').rstrip('0')
            if '.' in s: return len(s.split('.')[-1])
            else: return 0
        except Exception as e:
            logger.error(f"[{func_name}] Unexpected error calculating decimals for tick size {tick_size}: {e}.")
            return -1

    def _load_market_info(self) -> None:
        """Loads and caches market info (precision, limits), calculating decimal places."""
        # (Implementation is the same as provided in v6.2 - uses cached market_info)
        func_name = "_load_market_info"
        logger.debug(f"[{func_name}] Loading market details for {self.symbol}...")
        if not self.exchange or not self.exchange.markets:
            logger.critical(f"{Fore.RED}[{func_name}] Exchange not initialized or markets not loaded. Aborting.{Style.RESET_ALL}")
            sys.exit(1)
        if self.market_info is None or self.market_info.get('symbol') != self.symbol:
             logger.critical(f"{Fore.RED}[{func_name}] Market info for '{self.symbol}' not found in cached data. Aborting.{Style.RESET_ALL}")
             sys.exit(1)

        try:
            market = self.market_info
            precision = market.get('precision', {})
            limits = market.get('limits', {})
            amount_tick = precision.get('amount')
            price_tick = precision.get('price')
            min_amount = limits.get('amount', {}).get('min')
            min_cost = limits.get('cost', {}).get('min')

            price_decimals_calc = self._calculate_decimal_places(price_tick)
            amount_decimals_calc = self._calculate_decimal_places(amount_tick)

            self.price_decimals = price_decimals_calc if price_decimals_calc != -1 else DEFAULT_PRICE_DECIMALS
            self.amount_decimals = amount_decimals_calc if amount_decimals_calc != -1 else DEFAULT_AMOUNT_DECIMALS

            log_details = f"Price Decimals={self.price_decimals}"
            if price_decimals_calc == -1:
                log_details += f" (Used Default, Tick: {price_tick})"
                logger.warning(f"{Fore.YELLOW}[{func_name}] Price tick size missing/invalid for {self.symbol}. Using default: {self.price_decimals}.{Style.RESET_ALL}")
            else: log_details += f" (Tick: {price_tick})"

            log_details += f", Amount Decimals={self.amount_decimals}"
            if amount_decimals_calc == -1:
                log_details += f" (Used Default, Tick: {amount_tick})"
                logger.warning(f"{Fore.YELLOW}[{func_name}] Amount tick size missing/invalid for {self.symbol}. Using default: {self.amount_decimals}.{Style.RESET_ALL}")
            else: log_details += f" (Tick: {amount_tick})"

            logger.info(f"[{func_name}] Market Details for {self.symbol}: {log_details}")
            logger.debug(f"[{func_name}] Limits: Min Amount={min_amount}, Min Cost={min_cost}")

            if min_amount is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Min order amount limit missing for {self.symbol}.{Style.RESET_ALL}")
            if min_cost is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Min order cost limit missing for {self.symbol}.{Style.RESET_ALL}")

            market_type = market.get('type', 'unknown'); sub_type = market.get('subType', 'N/A')
            settle_ccy = market.get('settle', 'N/A'); is_linear = market.get('linear', False)
            is_inverse = market.get('inverse', False); contract_type = "Linear" if is_linear else ("Inverse" if is_inverse else "N/A")
            logger.debug(f"[{func_name}] Market Type: {market_type}, SubType: {sub_type}, Settle: {settle_ccy}, Contract Logic: {contract_type}")
            logger.debug(f"[{func_name}] Market details processed.")

        except (KeyError, ValueError, TypeError) as e:
             logger.critical(f"{Fore.RED}[{func_name}] Error parsing market info for {self.symbol}: {e}. Market Info: {str(self.market_info)[:500]}... Aborting.{Style.RESET_ALL}", exc_info=False)
             sys.exit(1)
        except Exception as e:
            logger.critical(f"{Fore.RED}[{func_name}] Failed loading market info for {self.symbol}: {e}. Aborting.{Style.RESET_ALL}", exc_info=True)
            sys.exit(1)

    # --- Formatting Helpers ---
    def format_price(self, price: Optional[Union[float, int, str]], default: str = "N/A") -> str:
        """Formats a price value using the market's price precision."""
        if price is None: return default
        try:
            price_float = float(price)
            if math.isnan(price_float): return default
            return f"{price_float:.{self.price_decimals}f}"
        except (ValueError, TypeError):
            return str(price) # Return original string if conversion fails

    def format_amount(self, amount: Optional[Union[float, int, str]], default: str = "N/A") -> str:
        """Formats an amount value using the market's amount precision."""
        if amount is None: return default
        try:
            amount_float = float(amount)
            if math.isnan(amount_float): return default
            return f"{amount_float:.{self.amount_decimals}f}"
        except (ValueError, TypeError):
            return str(amount) # Return original string if conversion fails

    def format_indicator(self, value: Optional[Union[float, int, str]], decimals: int = 2, default: str = "N/A") -> str:
        """Formats an indicator value to a specified number of decimals."""
        if value is None: return default
        try:
            value_float = float(value)
            if math.isnan(value_float): return default
            return f"{value_float:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value) # Return original string if conversion fails

    # --- Data Fetching Methods ---
    # fetch_market_price, fetch_order_book, fetch_historical_data, fetch_balance, fetch_order_status
    # (Implementations are the same as provided in v6.2)

    @retry_api_call()
    def fetch_market_price(self) -> Optional[float]:
        """Fetches the last traded price for the symbol."""
        func_name = "fetch_market_price"
        params = {}
        if self.exchange_id.lower() == 'bybit' and self.market_info:
             mkt_type = self.market_info.get('type', 'swap')
             is_linear = self.market_info.get('linear', True)
             if mkt_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
             elif mkt_type == 'spot': params['category'] = 'spot'
             elif mkt_type == 'option': params['category'] = 'option'
        ticker = self.exchange.fetch_ticker(self.symbol, params=params)
        last_price = ticker.get('last') if ticker else None
        if last_price is not None:
            try:
                price = float(last_price)
                if price <= 1e-12: logger.warning(f"{Fore.YELLOW}[{func_name}] Fetched zero/negative price ({last_price}).{Style.RESET_ALL}"); return None
                return price
            except (ValueError, TypeError) as e: logger.error(f"{Fore.RED}[{func_name}] Error converting price '{last_price}': {e}{Style.RESET_ALL}"); return None
        else: logger.warning(f"{Fore.YELLOW}[{func_name}] Could not fetch 'last' price. Ticker: {str(ticker)[:200]}{Style.RESET_ALL}"); return None

    @retry_api_call()
    def fetch_order_book(self) -> Optional[Dict[str, Any]]:
        """Fetches order book and calculates volume imbalance."""
        func_name = "fetch_order_book"
        try:
            params = {}
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                 mkt_type = self.market_info.get('type', 'swap')
                 is_linear = self.market_info.get('linear', True)
                 if mkt_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                 elif mkt_type == 'spot': params['category'] = 'spot'
                 elif mkt_type == 'option': params['category'] = 'option'
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth, params=params)
            bids = orderbook.get('bids', []); asks = orderbook.get('asks', [])
            result = {'bids': bids, 'asks': asks, 'imbalance': None, 'best_bid': None, 'best_ask': None}
            if bids and isinstance(bids[0], (list, tuple)) and len(bids[0]) >= 2: result['best_bid'] = float(bids[0][0])
            if asks and isinstance(asks[0], (list, tuple)) and len(asks[0]) >= 2: result['best_ask'] = float(asks[0][0])
            valid_bid_vols = [float(b[1]) for b in bids if isinstance(b, (list, tuple)) and len(b) >= 2 and isinstance(b[1], (int, float)) and b[1] >= 0]
            valid_ask_vols = [float(a[1]) for a in asks if isinstance(a, (list, tuple)) and len(a) >= 2 and isinstance(a[1], (int, float)) and a[1] >= 0]
            if not valid_bid_vols or not valid_ask_vols: logger.debug(f"[{func_name}] Incomplete OB data."); return result
            bid_vol_sum = sum(valid_bid_vols); ask_vol_sum = sum(valid_ask_vols); epsilon = 1e-12
            if bid_vol_sum > epsilon: result['imbalance'] = ask_vol_sum / bid_vol_sum
            elif ask_vol_sum > epsilon: result['imbalance'] = float('inf')
            else: result['imbalance'] = None
            return result
        except Exception as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Error fetching/processing OB: {e}{Style.RESET_ALL}", exc_info=False); return None

    @retry_api_call(max_retries=2, initial_delay=1)
    def fetch_historical_data(self, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Fetches historical OHLCV data, calculating required limit if needed."""
        func_name = "fetch_historical_data"
        required_periods = [self.volatility_window+1, self.ema_period, self.rsi_period+1, self.macd_long_period+self.macd_signal_period, (self.rsi_period+1)+self.stoch_rsi_period+max(self.stoch_rsi_k_period, self.stoch_rsi_d_period), self.atr_period+1]
        valid_periods = [p for p in required_periods if isinstance(p, int) and p > 0]; max_lookback = max(valid_periods) if valid_periods else 100
        min_req_len = max(max_lookback, MIN_DATAFRAME_LENGTH_WARN)
        fetch_limit: int
        if limit is None: fetch_limit = min(max(min_req_len + 50, 150), 1000)
        else: fetch_limit = limit; logger.debug(f"[{func_name}] Using provided limit: {limit}")
        logger.debug(f"[{func_name}] Fetching ~{fetch_limit} candles (Min valid needed: {min_req_len})...")
        try:
            params = {}
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                 mkt_type = self.market_info.get('type', 'swap'); is_linear = self.market_info.get('linear', True)
                 if mkt_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                 elif mkt_type == 'spot': params['category'] = 'spot'
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe, limit=fetch_limit, params=params)
            if not ohlcv: logger.warning(f"{Fore.YELLOW}[{func_name}] No OHLCV data returned.{Style.RESET_ALL}"); return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce'); df.set_index('timestamp', inplace=True)
            init_len = len(df); df.dropna(subset=[df.index.name], inplace=True); dropped_ts = init_len - len(df)
            if dropped_ts > 0: logger.debug(f"[{func_name}] Dropped {dropped_ts} rows with invalid timestamps.")
            init_len = len(df); ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in ohlcv_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
            essential_cols = ['open', 'high', 'low', 'close']; df.dropna(subset=essential_cols, inplace=True)
            final_len = len(df); dropped_ohlc = init_len - final_len
            if dropped_ohlc > 0: logger.debug(f"[{func_name}] Dropped {dropped_ohlc} rows with NaNs in OHLC.")
            if final_len < min_req_len: logger.warning(f"{Fore.YELLOW}[{func_name}] Insufficient valid history ({final_len}) after cleaning. Need ~{min_req_len}. Indicators may fail.{Style.RESET_ALL}");
            if final_len == 0: return None
            logger.debug(f"[{func_name}] Fetched/processed {final_len} valid candles."); return df
        except ccxt.BadSymbol as e: logger.critical(f"{Fore.RED}[{func_name}] BadSymbol {self.symbol}: {e}. Aborting.{Style.RESET_ALL}"); sys.exit(1)
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error fetching/processing history: {e}{Style.RESET_ALL}", exc_info=True); return None

    @retry_api_call()
    def fetch_balance(self, currency_code: Optional[str] = None) -> Optional[float]:
        """Fetches available balance. Returns None on failure. Handles simulation."""
        # (Implementation is the same as provided in v6.2 - detailed Bybit V5 parsing)
        func_name = "fetch_balance"
        quote_currency = currency_code or (self.market_info.get('quote') if self.market_info else None)
        if not quote_currency: logger.error(f"{Fore.RED}[{func_name}] Cannot determine quote currency.{Style.RESET_ALL}"); return None
        logger.debug(f"[{func_name}] Fetching available balance for {quote_currency}...")
        if self.simulation_mode: dummy_bal = 100000.0; logger.warning(f"[SIM] [{func_name}] Dummy balance: {self.format_price(dummy_bal)} {quote_currency}"); return dummy_bal

        try:
            params = {}; account_type_hint = None
            if self.exchange_id.lower() == 'bybit' and self.market_info:
                market_type = self.market_info.get('type', 'swap'); is_spot = self.market_info.get('spot', False)
                is_swap = self.market_info.get('swap', False); is_future = self.market_info.get('future', False); is_option = self.market_info.get('option', False)
                if is_spot: account_type_hint = 'SPOT'
                elif is_swap or is_future: account_type_hint = 'UNIFIED' # Or 'CONTRACT'
                elif is_option: account_type_hint = 'OPTION'
                if account_type_hint: params['accountType'] = account_type_hint; logger.debug(f"[{func_name}] Using Bybit accountType hint: {account_type_hint}")

            balance_data = self.exchange.fetch_balance(params=params)
            available_balance_str: Optional[str] = None; balance_source: str = "N/A"

            # 1. Standard CCXT 'free'
            if quote_currency in balance_data and isinstance(balance_data[quote_currency], dict):
                free_bal = balance_data[quote_currency].get('free')
                if free_bal is not None:
                    try:
                        if float(free_bal) > 1e-12: available_balance_str = str(free_bal); balance_source = "ccxt 'free'"; logger.debug(f"[{func_name}] Found usable ccxt 'free': {available_balance_str}")
                        else: logger.debug(f"[{func_name}] ccxt 'free' zero/negligible ({free_bal}).")
                    except: logger.debug(f"[{func_name}] ccxt 'free' not valid number ({free_bal}).")

            # 2. Bybit 'info' parsing
            if available_balance_str is None and self.exchange_id.lower() == 'bybit':
                logger.debug(f"[{func_name}] Checking Bybit 'info' structure..."); info_data = balance_data.get('info', {})
                try:
                    result_list = info_data.get('result', {}).get('list', [])
                    if result_list and isinstance(result_list, list) and len(result_list) > 0:
                        target_account_info = next((acc for acc in result_list if acc.get('accountType') == account_type_hint), None) if account_type_hint else result_list[0]
                        if target_account_info:
                           account_type_found = target_account_info.get('accountType'); logger.debug(f"[{func_name}] Parsing Bybit 'info' account: {account_type_found}")
                           coin_list = target_account_info.get('coin', []); found_coin = False
                           for coin_data in coin_list:
                               if isinstance(coin_data, dict) and coin_data.get('coin') == quote_currency:
                                   found_coin = True; temp_bal_str: Optional[str] = None; temp_src: Optional[str] = None
                                   priority = ['availableToWithdraw', 'availableBalance', 'equity', 'walletBalance']
                                   for field in priority:
                                       val = coin_data.get(field)
                                       if val is not None:
                                           try:
                                               if float(val) > 1e-12: temp_bal_str = str(val); temp_src = f"info.{account_type_found}.{field}"; break
                                           except: continue
                                   if temp_bal_str: available_balance_str = temp_bal_str; balance_source = temp_src; logger.debug(f"[{func_name}] Using Bybit '{balance_source}': {available_balance_str}"); break
                           if not found_coin: logger.debug(f"[{func_name}] {quote_currency} not in Bybit 'info' coin list for account {account_type_found}.")
                        else: logger.debug(f"[{func_name}] Target account type ({account_type_hint}) not found in Bybit 'info'.")
                    else: logger.debug(f"[{func_name}] Bybit 'info.result.list' empty/invalid.")
                except Exception as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Error parsing Bybit 'info': {e}. Snippet: {str(info_data)[:200]}...{Style.RESET_ALL}")

            # 3. Fallback to 'total'
            if available_balance_str is None and quote_currency in balance_data and isinstance(balance_data[quote_currency], dict):
                 total_bal = balance_data[quote_currency].get('total')
                 if total_bal is not None:
                     try:
                        if float(total_bal) > 1e-12: available_balance_str = str(total_bal); balance_source = "ccxt 'total'"; logger.warning(f"{Fore.YELLOW}[{func_name}] Using 'total' balance ({available_balance_str}) as fallback (includes used margin).{Style.RESET_ALL}")
                        else: logger.debug(f"[{func_name}] Fallback 'total' zero/negligible ({total_bal}).")
                     except: logger.debug(f"[{func_name}] Fallback 'total' not valid number ({total_bal}).")

            # 4. Final Conversion
            if available_balance_str is not None:
                try:
                    final_bal = float(available_balance_str)
                    if final_bal < 0: logger.warning(f"{Fore.YELLOW}[{func_name}] Balance negative ({final_bal}), treating as 0.0.{Style.RESET_ALL}"); final_bal = 0.0
                    logger.info(f"[{func_name}] Available balance {quote_currency}: {self.format_price(final_bal)} (Source: {balance_source})")
                    return final_bal
                except (ValueError, TypeError): logger.error(f"{Fore.RED}[{func_name}] Failed converting final balance '{available_balance_str}' to float.{Style.RESET_ALL}"); return None
            else: logger.error(f"{Fore.RED}[{func_name}] Failed to determine balance for {quote_currency} from any source.{Style.RESET_ALL}"); return None
        except ccxt.AuthenticationError as e: logger.error(f"{Fore.RED}[{func_name}] Auth error fetching balance: {e}.{Style.RESET_ALL}"); return None
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching balance: {e}{Style.RESET_ALL}", exc_info=True); return None

    @retry_api_call(max_retries=1)
    def fetch_order_status(self, order_id: str, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches status of specific order. Returns order dict or None if not found/error. Handles simulation."""
        # (Implementation is the same as provided in v6.2)
        func_name = "fetch_order_status"
        if not order_id or not isinstance(order_id, str): logger.warning(f"[{func_name}] Invalid order_id: {order_id}."); return None
        target_symbol = symbol or self.symbol
        logger.debug(f"[{func_name}] Fetching status for order {order_id} ({target_symbol})...")
        if self.simulation_mode: # Simulation Logic from v6.2
            pos = next((p for p in self.open_positions if p.get('id') == order_id), None)
            if pos:
                logger.debug(f"[SIM][{func_name}] Returning cached status for {order_id}.")
                sim_status = pos.get('status', STATUS_UNKNOWN)
                ccxt_status = 'open' if sim_status == STATUS_PENDING_ENTRY else ('closed' if sim_status in [STATUS_ACTIVE, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT] else ('canceled' if sim_status == STATUS_CANCELED else sim_status))
                is_active_or_closed = sim_status in [STATUS_ACTIVE, STATUS_CLOSED_EXT, STATUS_CLOSED_ON_EXIT]
                sim_filled = pos.get('size', 0.0) if is_active_or_closed else 0.0
                sim_avg = pos.get('entry_price') if is_active_or_closed else None
                sim_amount = pos.get('original_size', 0.0); sim_remaining = 0.0 if sim_status != STATUS_PENDING_ENTRY else max(0.0, sim_amount - pos.get('size', 0.0))
                sim_ts_s = pos.get('last_update_time') or pos.get('entry_time') or time.time(); sim_ts_ms = int(sim_ts_s * 1000)
                sim_order = {'id': order_id, 'timestamp': sim_ts_ms, 'datetime': pd.to_datetime(sim_ts_ms, unit='ms', utc=True).isoformat(timespec='milliseconds') + 'Z', 'symbol': target_symbol, 'type': pos.get('entry_order_type', 'limit'), 'side': pos.get('side'), 'status': ccxt_status, 'price': pos.get('limit_price'), 'amount': sim_amount, 'filled': sim_filled, 'remaining': sim_remaining, 'average': sim_avg, 'cost': sim_filled * sim_avg if sim_avg else 0.0, 'stopLossPrice': pos.get('stop_loss_price'), 'takeProfitPrice': pos.get('take_profit_price'), 'info': {'simulated': True, 'orderId': order_id, 'internalStatus': sim_status}}
                return sim_order
            else: logger.warning(f"[SIM][{func_name}] Simulated order {order_id} not found in state."); return None
        else: # Live Mode
            try:
                params = {}; target_market_info = self.exchange.market(target_symbol) if target_symbol in self.exchange.markets else self.market_info
                if self.exchange_id.lower() == 'bybit' and target_market_info:
                    market_type = target_market_info.get('type', 'swap'); is_linear = target_market_info.get('linear', True)
                    if market_type in ['swap', 'future']: params['category'] = 'linear' if is_linear else 'inverse'
                    elif market_type == 'spot': params['category'] = 'spot'
                    elif market_type == 'option': params['category'] = 'option'
                order_info = self.exchange.fetch_order(order_id, target_symbol, params=params)
                status = order_info.get('status', STATUS_UNKNOWN); filled = order_info.get('filled', 0.0); avg_price = order_info.get('average'); remaining = order_info.get('remaining', 0.0)
                logger.debug(f"[{func_name}] Order {order_id}: Status={status}, Filled={self.format_amount(filled)}, Avg={self.format_price(avg_price)}, Rem={self.format_amount(remaining)}")
                return order_info
            except ccxt.OrderNotFound as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} not found. Assumed final. Error: {e}{Style.RESET_ALL}"); return None
            except ccxt.ExchangeError as e:
                 err_str = str(e).lower(); final_phrases = ["order is finished", "has been filled", "already closed", "status error", "was cancel", "already cancel"]
                 if any(phrase in err_str for phrase in final_phrases): logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} reported as final via ExchangeError. Error: {e}{Style.RESET_ALL}"); return None
                 else: logger.error(f"{Fore.RED}[{func_name}] Exchange error fetching order {order_id}: {e}.{Style.RESET_ALL}"); return None
            except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error fetching order {order_id}: {e}{Style.RESET_ALL}", exc_info=True); return None

    # --- Indicator Calculation Methods (Static) ---
    # calculate_volatility, calculate_ema, calculate_rsi, calculate_macd, calculate_stoch_rsi, calculate_atr
    # (Implementations are the same as provided in v6.2)
    @staticmethod
    def calculate_volatility(close_prices: pd.Series, window: int) -> Optional[float]:
        func_name="calculate_volatility"; min_len=window+1
        if close_prices is None or not isinstance(close_prices, pd.Series) or window<=0: return None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len: return None
        try:
            valid_prices=close_prices[close_prices>1e-12];
            if len(valid_prices)<min_len: return None
            log_returns=np.log(valid_prices/valid_prices.shift(1))
            if log_returns.dropna().shape[0]<window: return None
            vol=log_returns.rolling(window=window,min_periods=window).std(ddof=1).iloc[-1]
            return float(vol) if pd.notna(vol) and vol>=0 else None
        except Exception as e: logger.error(f"[{func_name}] Win={window}: {e}",exc_info=False); return None
    @staticmethod
    def calculate_ema(close_prices: pd.Series, period: int) -> Optional[float]:
        func_name="calculate_ema"; min_len=period
        if close_prices is None or not isinstance(close_prices, pd.Series) or period<=0: return None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len: return None
        try:
            ema=close_prices.ewm(span=period,adjust=False,min_periods=period).mean().iloc[-1]
            return float(ema) if pd.notna(ema) else None
        except Exception as e: logger.error(f"[{func_name}] P={period}: {e}",exc_info=False); return None
    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int) -> Optional[float]:
        func_name="calculate_rsi"; min_len=period+1
        if close_prices is None or not isinstance(close_prices, pd.Series) or period<=0: return None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len: return None
        try:
            delta=close_prices.diff(1); gain=delta.where(delta>0,0.0).fillna(0); loss=-delta.where(delta<0,0.0).fillna(0)
            avg_gain=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean(); avg_loss=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
            last_avg_gain=avg_gain.iloc[-1]; last_avg_loss=avg_loss.iloc[-1]
            if pd.isna(last_avg_gain) or pd.isna(last_avg_loss): return None
            if last_avg_loss==0: rsi=100.0 if last_avg_gain>0 else 50.0
            else: rs=last_avg_gain/last_avg_loss; rsi=100.0-(100.0/(1.0+rs))
            return float(max(0.0,min(100.0,rsi))) if pd.notna(rsi) else None
        except Exception as e: logger.error(f"[{func_name}] P={period}: {e}",exc_info=False); return None
    @staticmethod
    def calculate_macd(close_prices: pd.Series, short_p: int, long_p: int, signal_p: int) -> Tuple[Optional[float],Optional[float],Optional[float]]:
        func_name="calculate_macd"; min_len=long_p+signal_p
        if close_prices is None or not isinstance(close_prices, pd.Series) or not all(isinstance(p,int) and p>0 for p in [short_p,long_p,signal_p]): return None,None,None
        if short_p>=long_p: return None,None,None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len: return None,None,None
        try:
            ema_s=close_prices.ewm(span=short_p,adjust=False,min_periods=short_p).mean(); ema_l=close_prices.ewm(span=long_p,adjust=False,min_periods=long_p).mean()
            macd_l=ema_s-ema_l; signal_l=macd_l.ewm(span=signal_p,adjust=False,min_periods=signal_p).mean(); hist=macd_l-signal_l
            macd_v,signal_v,hist_v=macd_l.iloc[-1],signal_l.iloc[-1],hist.iloc[-1]
            if any(pd.isna(v) for v in [macd_v,signal_v,hist_v]): return None,None,None
            return float(macd_v),float(signal_v),float(hist_v)
        except Exception as e: logger.error(f"[{func_name}] Ps={short_p},{long_p},{signal_p}: {e}",exc_info=False); return None,None,None
    @staticmethod
    def calculate_rsi_series(close_prices: pd.Series, period: int) -> Optional[pd.Series]:
        func_name="calculate_rsi_series"; min_len=period+1
        if close_prices is None or not isinstance(close_prices,pd.Series) or period<=0: return None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len: return None
        try:
            delta=close_prices.diff(1); gain=delta.where(delta>0,0.0).fillna(0); loss=-delta.where(delta<0,0.0).fillna(0)
            avg_gain=gain.ewm(alpha=1/period,adjust=False,min_periods=period).mean(); avg_loss=loss.ewm(alpha=1/period,adjust=False,min_periods=period).mean()
            rs=avg_gain/(avg_loss+1e-12); rsi=100.0-(100.0/(1.0+rs)); return rsi.clip(0,100)
        except Exception as e: logger.error(f"[{func_name}] P={period}: {e}",exc_info=False); return None
    @staticmethod
    def calculate_stoch_rsi(close_prices: pd.Series, rsi_p: int, stoch_p: int, k_p: int, d_p: int) -> Tuple[Optional[float],Optional[float]]:
        func_name="calculate_stoch_rsi"; min_len_est=(rsi_p+1)+stoch_p+max(k_p,d_p)-2
        if close_prices is None or not isinstance(close_prices,pd.Series) or not all(isinstance(p,int) and p>0 for p in [rsi_p,stoch_p,k_p,d_p]): return None,None
        close_prices=close_prices.dropna()
        if len(close_prices)<min_len_est: return None,None
        try:
            rsi_s=ScalpingBot.calculate_rsi_series(close_prices,rsi_p)
            if rsi_s is None or rsi_s.isna().all(): return None,None
            rsi_s_valid=rsi_s.dropna()
            if len(rsi_s_valid)<stoch_p+max(k_p,d_p)-1: return None,None
            min_r=rsi_s_valid.rolling(window=stoch_p,min_periods=stoch_p).min(); max_r=rsi_s_valid.rolling(window=stoch_p,min_periods=stoch_p).max()
            stoch_raw=(100.0*(rsi_s_valid-min_r)/(max_r-min_r+1e-12))
            stoch_k=stoch_raw.rolling(window=k_p,min_periods=1).mean(); stoch_d=stoch_k.rolling(window=d_p,min_periods=1).mean()
            k_v,d_v=stoch_k.iloc[-1],stoch_d.iloc[-1]
            if pd.isna(k_v) or pd.isna(d_v): return None,None
            return float(max(0.0,min(100.0,k_v))),float(max(0.0,min(100.0,d_v)))
        except Exception as e: logger.error(f"[{func_name}] Ps={rsi_p},{stoch_p},{k_p},{d_p}: {e}",exc_info=False); return None,None
    @staticmethod
    def calculate_atr(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, period: int) -> Optional[float]:
        func_name="calculate_atr"; min_len=period+1
        if high_prices is None or low_prices is None or close_prices is None or period<=0: return None
        if not isinstance(high_prices,pd.Series) or not isinstance(low_prices,pd.Series) or not isinstance(close_prices,pd.Series): return None
        df=pd.DataFrame({'high':high_prices,'low':low_prices,'close':close_prices}).dropna()
        if len(df)<min_len: return None
        h,l,c=df['high'],df['low'],df['close']
        try:
            h_l=h-l; h_cp=np.abs(h-c.shift(1)); l_cp=np.abs(l-c.shift(1))
            tr_df=pd.DataFrame({'hl':h_l,'hcp':h_cp,'lcp':l_cp}); true_range=tr_df.max(axis=1,skipna=True)
            atr=true_range.ewm(com=period-1,adjust=False,min_periods=period).mean().iloc[-1]
            return float(atr) if pd.notna(atr) and atr>=0 else None
        except Exception as e: logger.error(f"[{func_name}] P={period}: {e}",exc_info=False); return None

    # --- Trading Logic & Order Management ---
    # calculate_order_size, _calculate_sl_tp_prices, compute_trade_signal_score,
    # place_entry_order, cancel_order_by_id, cancel_all_symbol_orders,
    # _check_pending_entries, _manage_active_positions, _infer_exit_reason_price_based,
    # _update_trailing_stop_price, _attempt_edit_order_for_tsl, _log_position_pnl,
    # _place_market_close_order
    # (Implementations are the same as provided in v6.2)

    def calculate_order_size(self, current_price: float, indicators: Dict[str, Any], signal_score: Optional[int] = None) -> float:
        # (Implementation from v6.2)
        func_name = "calculate_order_size"
        if self.market_info is None: logger.error(f"[{func_name}] Market info missing."); return 0.0
        if current_price <= 1e-12: logger.error(f"[{func_name}] Invalid price {current_price}."); return 0.0
        if not (0 < self.order_size_percentage <= 1.0): logger.error(f"[{func_name}] Invalid risk % {self.order_size_percentage}."); return 0.0
        try:
            quote_currency = self.market_info['quote']; base_currency = self.market_info['base']
            limits = self.market_info.get('limits', {}); precision = self.market_info.get('precision', {})
            min_amount_str = limits.get('amount', {}).get('min'); min_cost_str = limits.get('cost', {}).get('min')
            min_amount = float(min_amount_str) if min_amount_str is not None else None
            min_cost = float(min_cost_str) if min_cost_str is not None else None
            amount_tick_size = precision.get('amount')
        except Exception as e: logger.error(f"[{func_name}] Error accessing market details: {e}."); return 0.0
        logger.debug(f"[{func_name}] Starting size calc for {base_currency}/{quote_currency} @ {self.format_price(current_price)}. Limits: MinAmt={min_amount}, MinCost={min_cost}")
        available_balance = self.fetch_balance(currency_code=quote_currency)
        if available_balance is None or available_balance <= 1e-9: logger.info(f"[{func_name}] Insufficient/failed balance fetch."); return 0.0
        logger.debug(f"[{func_name}] Available balance: {self.format_price(available_balance)} {quote_currency}")
        target_quote_value = available_balance * self.order_size_percentage
        logger.debug(f"[{func_name}] Target Quote Val: {self.format_price(target_quote_value)} (Risk: {self.order_size_percentage*100:.2f}%)")
        adjustment_factor = 1.0
        if self.volatility_multiplier > 1e-9:
            vol = indicators.get('volatility')
            if vol is not None and vol > 1e-9:
                try: vol_adj = max(0.1, min(2.0, 1.0 / (1.0 + vol * self.volatility_multiplier))); adjustment_factor *= vol_adj; logger.info(f"[{func_name}] Vol ({self.format_indicator(vol, 5)}) Factor: {vol_adj:.3f}. Total Factor: {adjustment_factor:.3f}")
                except Exception as e: logger.warning(f"[{func_name}] Vol adj error: {e}. Skipping.")
            else: logger.debug(f"[{func_name}] Vol adj skipped: Vol={vol}.")
        apply_sig_adj = signal_score is not None and (abs(self.strong_signal_adjustment_factor - 1.0) > 1e-9 or abs(self.weak_signal_adjustment_factor - 1.0) > 1e-9)
        if apply_sig_adj:
            abs_score = abs(signal_score); sig_adj = 1.0; adj_type = "Normal"
            if abs_score >= STRONG_SIGNAL_THRESHOLD_ABS: sig_adj = self.strong_signal_adjustment_factor; adj_type = "Strong"
            elif abs_score >= ENTRY_SIGNAL_THRESHOLD_ABS: pass # Normal = 1.0x
            if abs(sig_adj - 1.0) > 1e-9: adjustment_factor *= sig_adj; logger.info(f"[{func_name}] Signal ({signal_score}, {adj_type}) Factor: {sig_adj:.3f}. Total Factor: {adjustment_factor:.3f}")
        final_adjustment_factor = max(0.05, min(2.5, adjustment_factor))
        if abs(final_adjustment_factor - 1.0) > 1e-9: logger.debug(f"[{func_name}] Final Adj Factor: {final_adjustment_factor:.3f}")
        final_quote_value = target_quote_value * final_adjustment_factor
        if final_quote_value <= 1e-9: logger.warning(f"{Fore.YELLOW}[{func_name}] Quote value zero after adjustments.{Style.RESET_ALL}"); return 0.0
        logger.debug(f"[{func_name}] Final Quote Value: {self.format_price(final_quote_value)} {quote_currency}")
        order_size_base_raw = final_quote_value / current_price
        logger.debug(f"[{func_name}] Raw Base Amount: {self.format_amount(order_size_base_raw, self.amount_decimals + 4)} {base_currency}")
        try:
            amount_precise_str = self.exchange.amount_to_precision(self.symbol, order_size_base_raw)
            if amount_precise_str is None: logger.error(f"[{func_name}] amount_to_precision returned None."); return 0.0
            amount_precise = float(amount_precise_str)
            logger.debug(f"[{func_name}] Amount rounded: {self.format_amount(amount_precise)} (Tick: {amount_tick_size})")
            if amount_precise <= 1e-12: logger.warning(f"{Fore.YELLOW}[{func_name}] Amount zero after precision.{Style.RESET_ALL}"); return 0.0
            if min_amount is not None and amount_precise < min_amount: logger.warning(f"{Fore.YELLOW}[{func_name}] Amount {self.format_amount(amount_precise)} < MinAmount {self.format_amount(min_amount)}.{Style.RESET_ALL}"); return 0.0
            if min_cost is not None:
                price_precise_str = self.exchange.price_to_precision(self.symbol, current_price)
                price_for_cost = float(price_precise_str) if price_precise_str else current_price
                est_cost = amount_precise * price_for_cost
                if est_cost < min_cost: logger.warning(f"{Fore.YELLOW}[{func_name}] EstCost {self.format_price(est_cost)} < MinCost {self.format_price(min_cost)}.{Style.RESET_ALL}"); return 0.0
                logger.debug(f"[{func_name}] EstCost {self.format_price(est_cost)} >= MinCost {self.format_price(min_cost)}.")
            logger.info(f"{Fore.CYAN}[{func_name}] Calculated final size: {self.format_amount(amount_precise)} {base_currency}{Style.RESET_ALL}")
            return amount_precise
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error during precision/limit checks: {e}{Style.RESET_ALL}", exc_info=True); return 0.0

    def _calculate_sl_tp_prices(self, entry_price: float, side: str, current_price: float, atr: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        # (Implementation from v6.2)
        func_name="_calculate_sl_tp_prices"
        sl_raw, tp_raw = None, None
        if self.market_info is None or entry_price <= 1e-12 or side not in ['buy','sell']: return None, None
        if self.use_atr_sl_tp:
            if atr is None or atr <= 1e-12 or not (self.atr_sl_multiplier > 0 and self.atr_tp_multiplier > 0): logger.warning(f"{Fore.YELLOW}[{func_name}] Invalid ATR/Multipliers for ATR SL/TP.{Style.RESET_ALL}"); return None,None
            sl_delta = atr*self.atr_sl_multiplier; tp_delta = atr*self.atr_tp_multiplier
            sl_raw = entry_price - sl_delta if side == "buy" else entry_price + sl_delta
            tp_raw = entry_price + tp_delta if side == "buy" else entry_price - tp_delta
        else:
            sl_pct, tp_pct = self.base_stop_loss_pct, self.base_take_profit_pct; sl_valid=sl_pct is not None and 0<sl_pct<1; tp_valid=tp_pct is not None and 0<tp_pct<1
            if not sl_valid and not tp_valid: return None,None
            if sl_valid: sl_raw = entry_price*(1.0 - sl_pct) if side == "buy" else entry_price*(1.0 + sl_pct)
            if tp_valid: tp_raw = entry_price*(1.0 + tp_pct) if side == "buy" else entry_price*(1.0 - tp_pct)
        sl_final, tp_final = None, None
        price_tick = float(self.market_info['precision'].get('price', 1e-8)); tolerance = max(price_tick*0.5, entry_price*0.0001, 1e-9)
        try:
            if sl_raw is not None:
                if sl_raw > 1e-12:
                    sl_prec_str = self.exchange.price_to_precision(self.symbol, sl_raw); sl_prec = float(sl_prec_str) if sl_prec_str else None
                    if sl_prec and ((side=='buy' and sl_prec<entry_price-tolerance) or (side=='sell' and sl_prec>entry_price+tolerance)): sl_final = sl_prec
                    else: logger.warning(f"{Fore.YELLOW}[{func_name}] SL {self.format_price(sl_prec)} invalid vs entry {self.format_price(entry_price)}.{Style.RESET_ALL}")
                else: logger.warning(f"[{func_name}] Raw SL zero/negative ({sl_raw}).")
            if tp_raw is not None:
                if tp_raw > 1e-12:
                    tp_prec_str = self.exchange.price_to_precision(self.symbol, tp_raw); tp_prec = float(tp_prec_str) if tp_prec_str else None
                    if tp_prec and ((side=='buy' and tp_prec>entry_price+tolerance) or (side=='sell' and tp_prec<entry_price-tolerance)): tp_final = tp_prec
                    else: logger.warning(f"{Fore.YELLOW}[{func_name}] TP {self.format_price(tp_prec)} invalid vs entry {self.format_price(entry_price)}.{Style.RESET_ALL}")
                else: logger.warning(f"[{func_name}] Raw TP zero/negative ({tp_raw}).")
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error applying precision to SL/TP: {e}{Style.RESET_ALL}"); return None, None
        if sl_final is not None and tp_final is not None:
            if (side=="buy" and sl_final>=tp_final-tolerance) or (side=="sell" and sl_final<=tp_final+tolerance):
                logger.warning(f"{Fore.YELLOW}[{func_name}] Final SL {self.format_price(sl_final)} conflicts TP {self.format_price(tp_final)}. Nullifying TP.{Style.RESET_ALL}"); tp_final=None
        logger.debug(f"[{func_name}] Final SL={self.format_price(sl_final)}, TP={self.format_price(tp_final)}")
        return sl_final, tp_final

    def compute_trade_signal_score(self, price: float, indicators: Dict[str, Any], orderbook_imbalance: Optional[float]) -> Tuple[int, List[str]]:
        # (Implementation from v6.2)
        func_name = "compute_trade_signal_score"; score = 0.0; reasons = []
        RSI_OS, RSI_OB = 35, 65; STOCH_OS, STOCH_OB = 25, 75; EMA_THRESH_MULT = 0.0002; MACD_HIST_ZERO_THRESH = 1e-9
        def fmt_reason(t, p, c, v, d=2): c=Fore.GREEN if p>0 else(Fore.RED if p<0 else Fore.WHITE); vs=self.format_indicator(v,d) if v is not None else "N/A"; return f"{c}[{p:+.1f}] {t:<5} {c:<25} (Val: {vs}){Style.RESET_ALL}"
        imb_p=0.0; imb_c="N/A"; imb_v=orderbook_imbalance
        if imb_v is not None:
            imb=imb_v; imb_t=self.imbalance_threshold
            if imb_t<=0: imb_c=f"Invalid Thresh ({imb_t})"
            elif imb==float('inf'): imb_p=-1.0; imb_c="Sell (Imb=Inf)"
            else: buy_t=1.0/imb_t; imb_c=f"Neutral ({buy_t:.3f}-{imb_t:.3f})"; if imb<buy_t: imb_p=1.0; imb_c=f"Buy (Imb<{buy_t:.3f})" ; elif imb>imb_t: imb_p=-1.0; imb_c=f"Sell (Imb>{imb_t:.3f})"
        else: imb_c="OB N/A"
        score+=imb_p; reasons.append(fmt_reason("OB", imb_p, imb_c, imb_v, 3))
        ema=indicators.get('ema'); ema_p=0.0; ema_c="N/A"
        if ema is not None and ema>1e-9: upper=ema*(1.0+EMA_THRESH_MULT); lower=ema*(1.0-EMA_THRESH_MULT); ema_c="Price ~ EMA"; if price>upper: ema_p=1.0; ema_c=f"Price>EMA+{EMA_THRESH_MULT*100:.2f}%"; elif price<lower: ema_p=-1.0; ema_c=f"Price<EMA-{EMA_THRESH_MULT*100:.2f}%"
        else: ema_c="EMA N/A"
        score+=ema_p; reasons.append(fmt_reason("EMA", ema_p, ema_c, ema, self.price_decimals))
        rsi=indicators.get('rsi'); rsi_p=0.0; rsi_c="N/A"
        if rsi is not None: rsi_c="Neutral"; if rsi<RSI_OS: rsi_p=1.0; rsi_c=f"Oversold (<{RSI_OS})"; elif rsi>RSI_OB: rsi_p=-1.0; rsi_c=f"Overbought (>{RSI_OB})"
        else: rsi_c="RSI N/A"
        score+=rsi_p; reasons.append(fmt_reason("RSI", rsi_p, rsi_c, rsi, 1))
        macd_h=indicators.get('macd_hist'); macd_p=0.0; macd_c="N/A"
        if macd_h is not None: macd_c="Hist ~ 0 (Neutral)"; if macd_h>MACD_HIST_ZERO_THRESH: macd_p=1.0; macd_c="Hist > 0 (Bull Cross)"; elif macd_h<-MACD_HIST_ZERO_THRESH: macd_p=-1.0; macd_c="Hist < 0 (Bear Cross)"
        else: macd_c="MACD N/A"
        score+=macd_p; reasons.append(fmt_reason("MACD", macd_p, macd_c, macd_h, self.price_decimals + 2))
        k,d=indicators.get('stoch_k'),indicators.get('stoch_d'); stoch_p=0.0; stoch_c="N/A"
        if k is not None and d is not None: stoch_c="Neutral"; if k<STOCH_OS and d<STOCH_OS: stoch_p=1.0; stoch_c=f"Oversold (K&D<{STOCH_OS})"; elif k>STOCH_OB and d>STOCH_OB: stoch_p=-1.0; stoch_c=f"Overbought (K&D>{STOCH_OB})"
        else: stoch_c="Stoch N/A"
        score+=stoch_p; stoch_vs=f"K:{self.format_indicator(k,1)}/D:{self.format_indicator(d,1)}"; reasons.append(f"{fmt_reason('Stoch', stoch_p, stoch_c, None)} -> {stoch_vs}")
        final_score=int(round(score)); logger.debug(f"[{func_name}] Raw Score: {score:.2f}, Final: {final_score}"); return final_score, reasons

    def place_entry_order(self, side: str, order_size_base: float, confidence_level: int, order_type: str, current_price: float, stop_loss_price: Optional[float]=None, take_profit_price: Optional[float]=None) -> Optional[Dict[str,Any]]:
        # (Implementation from v6.2)
        func_name="place_entry_order";
        if self.market_info is None or order_size_base<=1e-12 or current_price<=1e-12 or side not in ['buy','sell'] or order_type not in ['market','limit']: return None
        base_ccy=self.market_info['base']; quote_ccy=self.market_info['quote']; params: Dict[str,Any]={}; limit_price: Optional[float]=None
        try:
            amount_p=order_size_base; logger.debug(f"[{func_name}] Using precise amount: {self.format_amount(amount_p)} {base_ccy}")
            if order_type=="limit":
                offset=self.limit_order_entry_offset_pct_buy if side=='buy' else self.limit_order_entry_offset_pct_sell
                if offset<0: offset=0.0
                factor=(1.0-offset) if side=='buy' else (1.0+offset); limit_raw=current_price*factor
                if limit_raw<=1e-12: raise ValueError("Limit price zero/negative")
                limit_str=self.exchange.price_to_precision(self.symbol,limit_raw)
                if limit_str is None: raise ValueError("Precision failed for limit price")
                limit_price=float(limit_str); logger.debug(f"[{func_name}] Limit price: {self.format_price(limit_price)}")
            if stop_loss_price is not None and stop_loss_price>1e-12: params['stopLoss']=self.format_price(stop_loss_price); if self.sl_trigger_by: params['slTriggerBy']=self.sl_trigger_by
            if take_profit_price is not None and take_profit_price>1e-12: params['takeProfit']=self.format_price(take_profit_price); if self.tp_trigger_by: params['tpTriggerBy']=self.tp_trigger_by
            if self.exchange_id.lower()=='bybit' and self.market_info:
                mkt_t=self.market_info.get('type','swap'); is_l=self.market_info.get('linear',True)
                if mkt_t in ['swap','future']: params['category']='linear' if is_l else 'inverse'
                elif mkt_t=='spot': params['category']='spot'
                elif mkt_t=='option': params['category']='option'
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error preparing order: {e}{Style.RESET_ALL}",exc_info=True); return None
        log_color=Fore.GREEN if side=='buy' else Fore.RED; action=f"{order_type.upper()} {side.upper()} ENTRY"; sl_inf=f"SL={params.get('stopLoss','N/A')}"+ (f" ({params.get('slTriggerBy')})" if 'slTriggerBy' in params else ""); tp_inf=f"TP={params.get('takeProfit','N/A')}"+ (f" ({params.get('tpTriggerBy')})" if 'tpTriggerBy' in params else ""); limit_inf=f"at Limit {self.format_price(limit_price)}" if limit_price else f"at Market (~{self.format_price(current_price)})"; est_val=amount_p*(limit_price or current_price); log_entry=f"ID: ???, Size: {self.format_amount(amount_p)} {base_ccy}, Price: {limit_inf}, EstVal: {self.format_price(est_val)} {quote_ccy}, Conf: {confidence_level}, {sl_inf}, {tp_inf}"
        if self.simulation_mode:
            sim_id=f"sim_entry_{int(time.time()*1000)}_{side[:1]}{confidence_level}"; sim_stat='open' if order_type=='limit' else 'closed'; sim_ep=limit_price if order_type=="limit" else current_price; sim_fill=amount_p if sim_stat=='closed' else 0.0; sim_avg=sim_ep if sim_stat=='closed' else None; sim_rem=amount_p-sim_fill; sim_ts=int(time.time()*1000)
            sim_order={"id":sim_id, "timestamp":sim_ts, "datetime":pd.to_datetime(sim_ts,unit='ms',utc=True).isoformat(timespec='milliseconds')+'Z', "symbol":self.symbol, "type":order_type, "side":side, "status":sim_stat, "price":limit_price, "amount":amount_p, "filled":sim_fill, "remaining":sim_rem, "average":sim_avg, "cost":sim_fill*sim_avg if sim_avg else 0.0, "stopLossPrice":stop_loss_price, "takeProfitPrice":take_profit_price, "info":{"simulated":True, "orderId":sim_id, **params}, "bot_custom_info":{"confidence":confidence_level, "initial_base_size":order_size_base}}
            logger.info(f"{log_color}[SIM] Placing {action}: {log_entry.replace('ID: ???', f'ID: {sim_id}')}{Style.RESET_ALL}"); return sim_order
        else:
            logger.info(f"{log_color}{Style.BRIGHT}Attempting LIVE {action}...{Style.RESET_ALL}"); log_live=f"{log_color} -> Sym:{self.symbol}, Side:{side.upper()}, Type:{order_type.upper()}\n -> Size:{self.format_amount(amount_p)} {base_ccy}\n -> Price:{limit_inf}\n -> EstVal:~{self.format_price(est_val)} {quote_ccy}\n -> Conf:{confidence_level}\n -> Params:{params}{Style.RESET_ALL}"; logger.info(log_live)
            order:Optional[Dict[str,Any]]=None
            try:
                if order_type=="market": order=self.exchange.create_market_order(self.symbol,side,amount_p,params=params)
                elif order_type=="limit":
                    if limit_price is None: raise ValueError("Limit price is None")
                    order=self.exchange.create_limit_order(self.symbol,side,amount_p,limit_price,params=params)
                if order:
                    oid=order.get('id','N/A'); ostatus=order.get('status',STATUS_UNKNOWN); ofilled=order.get('filled',0.0); oavg=order.get('average'); info_sl=order.get('info',{}).get('stopLoss') or order.get('stopLossPrice'); info_tp=order.get('info',{}).get('takeProfit') or order.get('takeProfitPrice')
                    logger.info(f"{log_color}---> LIVE {action} Placed: ID:{oid}, Status:{ostatus}, Filled:{self.format_amount(ofilled)}, Avg:{self.format_price(oavg)}, SL Sent/Conf:{params.get('stopLoss','N/A')}/{self.format_price(info_sl)}, TP Sent/Conf:{params.get('takeProfit','N/A')}/{self.format_price(info_tp)}{Style.RESET_ALL}")
                    order['bot_custom_info']={"confidence":confidence_level, "initial_base_size":order_size_base}; return order
                else: logger.error(f"{Fore.RED}LIVE {action} API call returned None.{Style.RESET_ALL}"); return None
            except(ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e: logger.error(f"{Fore.RED}LIVE {action} Failed ({type(e).__name__}): {e}{Style.RESET_ALL}"); if isinstance(e,ccxt.InsufficientFunds): self.fetch_balance(quote_ccy); return None
            except Exception as e: logger.error(f"{Fore.RED}LIVE {action} Failed (Unexpected): {e}{Style.RESET_ALL}",exc_info=True); return None

    def cancel_order_by_id(self, order_id: str, symbol: Optional[str] = None) -> bool:
        # (Implementation from v6.2)
        func_name="cancel_order_by_id"
        if not order_id or not isinstance(order_id,str): return False
        target_symbol=symbol or self.symbol; logger.info(f"{Fore.YELLOW}[{func_name}] Cancelling {order_id} ({target_symbol})...{Style.RESET_ALL}")
        if self.simulation_mode: # Sim Logic from v6.2
            idx=next((i for i,p in enumerate(self.open_positions) if p.get('id')==order_id),-1)
            if idx!=-1:
                if self.open_positions[idx].get('status')==STATUS_PENDING_ENTRY:
                    self.open_positions[idx]['status']=STATUS_CANCELED; self.open_positions[idx]['last_update_time']=time.time(); self.open_positions[idx]['exit_reason']="Cancelled by Bot"; logger.info(f"[SIM][{func_name}] Marked {order_id} as CANCELLED."); return True
                else: logger.warning(f"[SIM][{func_name}] Order {order_id} not pending ('{self.open_positions[idx].get('status')}')."); return True
            else: logger.warning(f"[SIM][{func_name}] Order {order_id} not found."); return True
        else: # Live Mode
            try:
                params={}; target_market=self.exchange.market(target_symbol) if target_symbol in self.exchange.markets else self.market_info
                if self.exchange_id.lower()=='bybit' and target_market: mkt_t=target_market.get('type','swap'); is_l=target_market.get('linear',True); if mkt_t in ['swap','future']: params['category']='linear' if is_l else 'inverse'; elif mkt_t=='spot': params['category']='spot'; elif mkt_t=='option': params['category']='option'
                resp=self.exchange.cancel_order(order_id,target_symbol,params=params); logger.info(f"{Fore.GREEN}---> Cancel success for {order_id}. Resp: {str(resp)[:150]}...{Style.RESET_ALL}"); return True
            except ccxt.OrderNotFound: logger.warning(f"{Fore.YELLOW}[{func_name}] Order {order_id} not found (already gone?). Success.{Style.RESET_ALL}"); return True
            except ccxt.NetworkError as e: raise e
            except ccxt.ExchangeError as e: err=str(e).lower(); final=["filled","finished","already closed","already cancel","status error","cannot be cancel"]; if any(p in err for p in final): logger.warning(f"{Fore.YELLOW}[{func_name}] Cannot cancel {order_id}: Final state. Success. Err: {e}{Style.RESET_ALL}"); return True; else: logger.error(f"{Fore.RED}[{func_name}] Exchange error cancel {order_id}: {e}{Style.RESET_ALL}"); return False
            except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error cancel {order_id}: {e}{Style.RESET_ALL}",exc_info=True); return False

    def cancel_all_symbol_orders(self, symbol: Optional[str] = None) -> int:
        # (Implementation from v6.2)
        func_name="cancel_all_symbol_orders"; target_symbol=symbol or self.symbol; logger.info(f"{Fore.YELLOW}[{func_name}] Cancelling all OPEN orders for {target_symbol}...{Style.RESET_ALL}"); cancelled_count=0
        if self.simulation_mode: # Sim Logic from v6.2
             logger.info(f"[SIM][{func_name}] Sim cancel all for {target_symbol}.")
             indices=[i for i,p in enumerate(self.open_positions) if p.get('symbol')==target_symbol and p.get('status')==STATUS_PENDING_ENTRY]; initial_open=len(indices)
             if not indices: logger.info(f"[SIM][{func_name}] No pending orders found."); return 0
             for idx in indices: self.open_positions[idx]['status']=STATUS_CANCELED; self.open_positions[idx]['last_update_time']=time.time(); self.open_positions[idx]['exit_reason']="Cancelled All"; cancelled_count+=1
             logger.info(f"[SIM][{func_name}] Marked {cancelled_count}/{initial_open} pending orders CANCELLED."); if cancelled_count > 0: self._save_state(); return cancelled_count
        else: # Live Mode
            try:
                params={}; target_market=self.exchange.market(target_symbol) if target_symbol in self.exchange.markets else self.market_info
                if self.exchange_id.lower()=='bybit' and target_market: mkt_t=target_market.get('type','swap'); is_l=target_market.get('linear',True); if mkt_t in ['swap','future']: params['category']='linear' if is_l else 'inverse'; elif mkt_t=='spot': params['category']='spot'; elif mkt_t=='option': params['category']='option'
                if self.exchange.has.get('cancelAllOrders'):
                    logger.debug(f"[{func_name}] Using cancelAllOrders for {target_symbol}..."); resp=self.exchange.cancel_all_orders(target_symbol,params=params); success=False
                    if isinstance(resp, list): success=True; cancelled_count=len(resp); logger.info(f"{Fore.GREEN}[{func_name}] cancelAllOrders OK. Reported {cancelled_count} cancellations.")
                    elif isinstance(resp, dict) and resp.get('retCode')==0: success=True # Check specific exchange success codes?
                    if success: logger.info(f"{Fore.GREEN}[{func_name}] cancelAllOrders OK. Count={cancelled_count if cancelled_count>0 else 'Unknown'}. Resp: {str(resp)[:150]}..."); return cancelled_count if cancelled_count>0 else -1
                    else: logger.warning(f"{Fore.YELLOW}[{func_name}] cancelAllOrders resp unknown/failed? {str(resp)[:150]}...{Style.RESET_ALL}"); return -1
                elif self.exchange.has.get('fetchOpenOrders'):
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Using fetchOpenOrders + individual cancel...{Style.RESET_ALL}"); open_orders=self.exchange.fetch_open_orders(target_symbol,params=params)
                    if not open_orders: logger.info(f"[{func_name}] No open orders found."); return 0
                    logger.warning(f"{Fore.YELLOW}[{func_name}] Found {len(open_orders)} orders. Cancelling individually...{Style.RESET_ALL}"); success_c=0; fail_c=0; delay=max(0.2,self.exchange.rateLimit/1000.0)
                    for o in open_orders: oid=o.get('id'); if not oid: logger.warning(f"[{func_name}] Skipping order with no ID: {o}"); continue; if self.cancel_order_by_id(oid, target_symbol): success_c+=1; else: fail_c+=1; time.sleep(delay)
                    log_level=logging.INFO if fail_c==0 else logging.WARNING; logger.log(log_level,f"[{func_name}] Individual cancel done. Success/Gone:{success_c}, Failed:{fail_c}."); return success_c
                else: logger.error(f"{Fore.RED}[{func_name}] Exchange lacks cancelAllOrders & fetchOpenOrders! Cannot cancel.{Style.RESET_ALL}"); return 0
            except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Error during bulk cancel: {e}{Style.RESET_ALL}",exc_info=True); return locals().get('success_c', 0)

    def _check_pending_entries(self, indicators: Dict) -> None:
        # (Implementation from v6.2)
        func_name="_check_pending_entries"; pending_ids=[p['id'] for p in self.open_positions if p.get('status')==STATUS_PENDING_ENTRY]
        if not pending_ids: return
        logger.debug(f"[{func_name}] Checking {len(pending_ids)} pending orders...")
        to_remove=set(); to_update={}; current_price_recalc=None
        for oid in pending_ids:
            idx=next((i for i,p in enumerate(self.open_positions) if p.get('id')==oid and p.get('status')==STATUS_PENDING_ENTRY),-1)
            if idx==-1: continue
            pos=self.open_positions[idx]; sym=pos.get('symbol',self.symbol)
            order_info=self.fetch_order_status(oid,symbol=sym)
            if order_info is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Pending {oid} not found. Removing.{Style.RESET_ALL}"); to_remove.add(oid); continue
            stat=order_info.get('status'); filled=float(order_info.get('filled',0.0)); avg_p=order_info.get('average'); remain=float(order_info.get('remaining',0.0))
            is_filled=(stat=='closed' and filled>1e-12 and remain<1e-9)
            if is_filled:
                if avg_p is None or float(avg_p)<=1e-12: logger.error(f"{Fore.RED}[{func_name}] Order {oid} filled but invalid avg price {avg_p}. Removing.{Style.RESET_ALL}"); to_remove.add(oid); continue
                ep=float(avg_p); orig_s=pos.get('original_size'); logger.info(f"{Fore.GREEN}{Style.BRIGHT}---> Pending {oid} FILLED! Side:{pos['side']}, Amt:{self.format_amount(filled)}, Avg:{self.format_price(ep)}{Style.RESET_ALL}")
                upd={**pos,'status':STATUS_ACTIVE,'size':filled,'entry_price':ep,'entry_time':(order_info.get('lastTradeTimestamp') or order_info.get('timestamp') or time.time()*1000)/1000.0,'last_update_time':time.time()}
                if current_price_recalc is None: current_price_recalc=self.fetch_market_price()
                if current_price_recalc: atr=indicators.get('atr'); sl,tp=self._calculate_sl_tp_prices(ep,upd['side'],current_price_recalc,atr); upd['stop_loss_price']=sl; upd['take_profit_price']=tp; logger.info(f"[{func_name}] Internal SL/TP updated for {oid}: SL={self.format_price(sl)}, TP={self.format_price(tp)}")
                else: logger.warning(f"[{func_name}] Failed price fetch for SL/TP recalc on fill {oid}.")
                to_update[oid]=upd
            elif stat in ['canceled','rejected','expired']: reason=order_info.get('info',{}).get('rejectReason','N/A'); logger.warning(f"{Fore.YELLOW}[{func_name}] Pending {oid} failed ({stat}, Reason:{reason}). Removing.{Style.RESET_ALL}"); to_remove.add(oid)
            elif stat=='open': logger.debug(f"[{func_name}] Pending {oid} still open. Filled:{self.format_amount(filled)}, Rem:{self.format_amount(remain)}.")
            elif stat=='closed' and filled<=1e-12: logger.warning(f"{Fore.YELLOW}[{func_name}] Pending {oid} closed with zero fill. Assuming cancelled. Removing.{Style.RESET_ALL}"); to_remove.add(oid)
            else: logger.warning(f"[{func_name}] Pending {oid} unexpected status '{stat}'. Leaving pending.")
        if to_remove or to_update: # Apply changes
            new_list=[]; removed_c=0; updated_c=0; needs_save=False
            for p in self.open_positions: pid=p.get('id')
            if pid in to_remove: removed_c+=1; needs_save=True; continue
            if pid in to_update: new_list.append(to_update[pid]); updated_c+=1; needs_save=True; continue
            new_list.append(p) # Keep unchanged
            if needs_save: self.open_positions=new_list; logger.debug(f"[{func_name}] State update. Activated:{updated_c}, Removed:{removed_c}, Total:{len(new_list)}."); self._save_state()

    def _manage_active_positions(self, current_price: float, indicators: Dict) -> None:
        # (Implementation from v6.2 - includes fetchPositions logic)
        func_name="_manage_active_positions"; active_tuples=[(p['id'],p) for p in self.open_positions if p.get('status')==STATUS_ACTIVE]
        if not active_tuples: return
        logger.debug(f"[{func_name}] Managing {len(active_tuples)} active positions @ {self.format_price(current_price)}...")
        to_remove=set(); to_update={}; active_syms=list(set(p['symbol'] for _,p in active_tuples if p.get('symbol')))
        exch_pos_map={}; can_fetch_pos=self.exchange.has.get('fetchPositions') and active_syms
        if can_fetch_pos:
            try:
                logger.debug(f"[{func_name}] Batch fetching positions: {active_syms}"); params={}
                if self.exchange_id.lower()=='bybit': # Add category hint if possible
                    mkt=self.exchange.market(active_syms[0]) if active_syms[0] in self.exchange.markets else None
                    if mkt and mkt.get('type') in ['swap','future']: params['category']='linear' if mkt.get('linear',True) else 'inverse'
                fetched_pos=self.exchange.fetch_positions(symbols=active_syms,params=params)
                for p in fetched_pos: sym=p.get('symbol'); side=p.get('side'); int_side='buy' if side=='long' else ('sell' if side=='short' else None)
                if sym and int_side: exch_pos_map.setdefault(sym,{})[int_side]=p
                logger.debug(f"[{func_name}] Fetched {len(fetched_pos)} positions. Map created.")
            except Exception as e: logger.warning(f"{Fore.YELLOW}[{func_name}] Error batch fetching positions: {e}. Relying on order checks.{Style.RESET_ALL}"); can_fetch_pos=False
        for pid,pos in active_tuples:
            if pid in to_remove: continue
            sym=pos.get('symbol',self.symbol); side=pos.get('side'); entry_p=pos.get('entry_price'); size=pos.get('size'); entry_t=pos.get('entry_time')
            if not all([sym,side,entry_p,size,entry_t]): logger.error(f"[{func_name}] Pos {pid} missing data."); to_remove.add(pid); to_update[pid]={'status':STATUS_UNKNOWN,'exit_reason':'Missing Data'}; continue
            if size<=1e-12: logger.warning(f"{Fore.YELLOW}[{func_name}] Pos {pid} zero size. Removing.{Style.RESET_ALL}"); to_remove.add(pid); to_update[pid]={'status':STATUS_CLOSED_EXT,'exit_reason':'Zero Size'}; self._log_position_pnl(pos,current_price,"Zero Size"); continue
            exit_r: Optional[str]=None; exit_p: Optional[float]=current_price; closed_ext=False
            if can_fetch_pos: # Check via fetchPositions
                exch_p=exch_pos_map.get(sym,{}).get(side)
                if exch_p:
                    exch_s_val=exch_p.get('contracts') or exch_p.get('size');
                    if exch_s_val is not None:
                         exch_s=float(exch_s_val); tick=float(self.market_info['precision'].get('amount',1e-8)); tol=max(size*0.001,tick)
                         if abs(exch_s-size)>tol: closed_ext=True; exit_r=f"External Size Change (State:{size:.8f},Exch:{exch_s:.8f})"; liq_p=exch_p.get('liquidationPrice'); if liq_p: exit_p=float(liq_p); exit_r+="(Liq?)"
                         #else: logger.debug(f"[{func_name}] Pos {pid} active via fetchPositions (Size:{exch_s}).") # Position exists, size matches
                    #else: logger.warning(f"[{func_name}] Could not get size from fetched pos {pid}.") # Cannot confirm size
                else: closed_ext=True; exit_r="Not Found via fetchPositions"; exit_r+=f" ({self._infer_exit_reason_price_based(pos,current_price)})" # Infer based on price
            # Fallback check (less reliable)
            # if not closed_ext and not can_fetch_pos: ... (v6.2 fallback order check logic was here, removed for brevity/reliability focus on fetchPositions)
            if closed_ext: # Process external closure
                log_clr=Fore.RED if "SL" in exit_r or "Liq" in exit_r else (Fore.GREEN if "TP" in exit_r else Fore.YELLOW); logger.info(f"{log_clr}[{func_name}] External closure {pid}. Reason:{exit_r}. Exit~{self.format_price(exit_p)}{Style.RESET_ALL}")
                to_remove.add(pid); to_update[pid]={'status':STATUS_CLOSED_EXT,'exit_reason':exit_r,'exit_price':exit_p,'last_update_time':time.time()}; self._log_position_pnl(pos,exit_p,exit_r); continue
            # Check Time Exit
            if self.time_based_exit_minutes and self.time_based_exit_minutes>0:
                elapsed=(time.time()-entry_t)/60.0
                if elapsed>=self.time_based_exit_minutes:
                    logger.info(f"{Fore.YELLOW}[{func_name}] Time limit ({self.time_based_exit_minutes}m) reached for {pid} (Age:{elapsed:.1f}m). Closing...{Style.RESET_ALL}")
                    close_order=self._place_market_close_order(pos,current_price)
                    if close_order:
                        exit_r=f"Time Limit ({self.time_based_exit_minutes}m)"; close_avg=close_order.get('average'); exit_p_raw=close_avg if close_avg else current_price; try: exit_p=float(exit_p_raw)
                        except: pass; logger.info(f"[{func_name}] Market close for time OK {pid}. Exit~{self.format_price(exit_p)}")
                        to_remove.add(pid); to_update[pid]={'status':STATUS_CLOSED_ON_EXIT,'exit_reason':exit_r,'exit_price':exit_p,'exit_order_id':close_order.get('id'),'last_update_time':time.time()}; self._log_position_pnl(pos,exit_p,exit_r); continue
                    else: logger.critical(f"{Fore.RED}{Style.BRIGHT}[{func_name}] CRITICAL FAIL closing timed-out {pid}! REMAINS OPEN!{Style.RESET_ALL}"); upd=to_update.get(pid,{}); upd['failed_close_attempts']=pos.get('failed_close_attempts',0)+1; to_update[pid]=upd
            # Check TSL [EXPERIMENTAL]
            if self.enable_trailing_stop_loss:
                new_tsl=self._update_trailing_stop_price(pos,current_price)
                if new_tsl is not None:
                    logger.warning(f"{Fore.YELLOW}[EXP][{func_name}] TSL update {pid} -> {self.format_price(new_tsl)}. Attempting edit...{Style.RESET_ALL}")
                    edit_ok=self._attempt_edit_order_for_tsl(pos,new_tsl)
                    if edit_ok: upd=to_update.get(pid,{}); upd.update({'trailing_stop_price':new_tsl,'stop_loss_price':new_tsl,'last_update_time':time.time()}); to_update[pid]=upd; logger.info(f"{Fore.MAGENTA}---> Internal TSL state {pid} updated to {self.format_price(new_tsl)}.{Style.RESET_ALL}")
                    else: logger.error(f"{Fore.RED}[{func_name}] TSL edit_order {pid} failed/skipped. State NOT updated.{Style.RESET_ALL}")
        if to_remove or to_update: # Apply state changes
            new_list=[]; removed_c=0; updated_c=0; needs_save=False; orig_c=len(self.open_positions)
            for p in self.open_positions: pid=p.get('id')
            if pid in to_remove: removed_c+=1; needs_save=True; logger.debug(f"[{func_name}] Removing {pid} from state. Final update: {to_update.get(pid)}"); continue
            if pid in to_update: upd_d=to_update[pid]; if 'status' in upd_d and p['status']!=STATUS_ACTIVE: del upd_d['status']; # Avoid status overwrite
            if upd_d: p.update(upd_d); updated_c+=1; needs_save=True; logger.debug(f"[{func_name}] Applied updates to {pid}: {upd_d}")
            new_list.append(p)
            if needs_save: self.open_positions=new_list; logger.debug(f"[{func_name}] Manage active state update: Orig={orig_c}, Upd={updated_c}, Rem={removed_c}, New={len(new_list)}."); self._save_state()

    def _infer_exit_reason_price_based(self, position: Dict, exit_price: float) -> Optional[str]:
        # (Implementation from v6.2)
        func_name="_infer_exit_reason_price_based"; pid=position.get('id','N/A'); side=position.get('side'); entry_p=position.get('entry_price'); sl=position.get('stop_loss_price'); tp=position.get('take_profit_price');
        if not all([side, entry_p, self.market_info]): return "Inference Failed (Data)"
        try: tick_str=self.market_info['precision'].get('price'); tick=float(tick_str) if tick_str else 10.0**(-self.price_decimals); tol=max(tick*2,entry_p*0.0005); sl_hit,tp_hit=False,False
        if sl is not None: sl_hit=abs(exit_price-sl)<=tol
        if tp is not None: tp_hit=abs(exit_price-tp)<=tol
        if sl_hit and tp_hit: return "SL/TP Hit (Ambiguous)"
        elif sl_hit: return "SL Hit?"
        elif tp_hit: return "TP Hit?"
        else: return "Manual/Other?"
        except Exception as e: logger.warning(f"[{func_name}] Price inference error {pid}: {e}"); return "Inference Error"

    def _update_trailing_stop_price(self, position: Dict, current_price: float) -> Optional[float]:
        # (Implementation from v6.2)
        func_name="_update_trailing_stop_price"; pid=position.get('id','N/A'); side=position.get('side'); entry_p=position.get('entry_price'); current_tsl=position.get('trailing_stop_price'); base_sl=position.get('stop_loss_price') if current_tsl is None else current_tsl; tsl_pct=self.trailing_stop_loss_percentage
        if not side or not entry_p or not tsl_pct or not(0<tsl_pct<1) or current_price<=0 or self.market_info is None: return None
        potential_tsl_raw=current_price*(1.0-tsl_pct) if side=='buy' else current_price*(1.0+tsl_pct); potential_tsl:Optional[float]=None
        try: tsl_str=self.exchange.price_to_precision(self.symbol,potential_tsl_raw); if tsl_str is None: raise ValueError("Precision failed"); potential_tsl=float(tsl_str); if potential_tsl<=1e-12: potential_tsl=None
        except Exception as e: logger.error(f"[{func_name}] Error formatting potential TSL {potential_tsl_raw} for {pid}: {e}"); return None
        if potential_tsl is None: return None
        new_tsl:Optional[float]=None; is_better=(side=='buy' and potential_tsl>(base_sl or 0.0)) or (side=='sell' and potential_tsl<(base_sl or float('inf')))
        if is_better:
            is_valid=(side=='buy' and potential_tsl<current_price) or (side=='sell' and potential_tsl>current_price)
            if is_valid: new_tsl=potential_tsl; action="ACTIVATING" if current_tsl is None else "UPDATING"; logger.info(f"{Fore.MAGENTA}[{func_name}] TSL {action} {side} {pid}: {self.format_price(base_sl)}->{self.format_price(new_tsl)} (Price:{self.format_price(current_price)}){Style.RESET_ALL}")
            else: logger.warning(f"[{func_name}] Potential TSL {self.format_price(potential_tsl)} skipped: crosses current price {self.format_price(current_price)}.")
        return new_tsl

    def _attempt_edit_order_for_tsl(self, position: Dict, new_tsl_price: float) -> bool:
        # (Implementation from v6.2)
        func_name="_attempt_edit_order_for_tsl"; oid=position.get('id'); sym=position.get('symbol'); tp_stored=position.get('take_profit_price')
        if not oid: return False; logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}[EXP]{Style.RESET_ALL} [{func_name}] Attempting TSL edit for {oid} -> {self.format_price(new_tsl_price)}...");
        if not self.exchange.has.get('editOrder'): logger.error(f"{Fore.RED}[{func_name}] FAILED: Exchange {self.exchange_id} no 'editOrder' support.{Style.RESET_ALL}"); if self.enable_trailing_stop_loss: self.enable_trailing_stop_loss=False; logger.warning("--> TSL Disabled."); return False
        try:
            params={'stopLoss':self.format_price(new_tsl_price)}; if self.sl_trigger_by: params['slTriggerBy']=self.sl_trigger_by
            if tp_stored is not None and tp_stored>1e-12: params['takeProfit']=self.format_price(tp_stored); if self.tp_trigger_by: params['tpTriggerBy']=self.tp_trigger_by; logger.debug(f"[{func_name}] Resending TP={params['takeProfit']}")
            if self.exchange_id.lower()=='bybit': # Add category hint
                 mkt=self.exchange.market(sym) if sym in self.exchange.markets else self.market_info
                 if mkt: mkt_t=mkt.get('type','swap'); is_l=mkt.get('linear',True); if mkt_t in ['swap','future']: params['category']='linear' if is_l else 'inverse'; elif mkt_t=='spot': params['category']='spot'; elif mkt_t=='option': params['category']='option'
            logger.debug(f"[{func_name}] Calling edit_order for {oid}. Params: {params}");
            @retry_api_call(max_retries=1, initial_delay=1)
            def _edit_call(oid_arg,sym_arg,params_arg): return self.exchange.edit_order(oid_arg,sym_arg,params=params_arg) # Simplified call using params for Bybit V5
            resp=_edit_call(oid,sym,params);
            if resp:
                resp_id=resp.get('id','N/A'); info=resp.get('info',{}); sl_conf=info.get('stopLoss'); tp_conf=info.get('takeProfit'); logger.info(f"{Fore.MAGENTA}---> edit_order attempt OK {oid}. RespID:{resp_id}. SLConf:{self.format_price(sl_conf)}, TPConf:{self.format_price(tp_conf)}{Style.RESET_ALL}");
                edit_ok=False; if sl_conf is not None: try: fmt_target=self.format_price(new_tsl_price); if sl_conf==fmt_target: logger.info(f"{Fore.GREEN}---> TSL edit CONFIRMED via matching response SL.{Style.RESET_ALL}"); edit_ok=True; else: logger.warning(f"{Fore.YELLOW}---> TSL edit response SL ({sl_conf}) mismatch target ({fmt_target}). Assume OK cautiously.{Style.RESET_ALL}"); edit_ok=True; except Exception: logger.warning(f"{Fore.YELLOW}---> Error comparing SL resp. Assume OK cautiously.{Style.RESET_ALL}"); edit_ok=True; else: logger.warning(f"{Fore.YELLOW}---> TSL edit resp no SL confirmation. Assume OK cautiously.{Style.RESET_ALL}"); edit_ok=True; return edit_ok
            else: logger.error(f"{Fore.RED}[{func_name}] TSL edit_order {oid} API returned None.{Style.RESET_ALL}"); return False
        except ccxt.NotSupported as e: logger.error(f"{Fore.RED}[{func_name}] TSL FAILED: editOrder NOT SUPPORTED. Disable TSL. Err: {e}{Style.RESET_ALL}"); if self.enable_trailing_stop_loss: self.enable_trailing_stop_loss=False; logger.warning("--> TSL Disabled."); return False
        except(ccxt.OrderNotFound,ccxt.InvalidOrder) as e: logger.warning(f"{Fore.YELLOW}[{func_name}] TSL edit failed {oid}: Order gone/invalid? ({type(e).__name__}): {e}{Style.RESET_ALL}"); return False
        except ccxt.ExchangeError as e: err=str(e).lower(); if any(p in err for p in ["does not exist","not allow mod","already closed"]): logger.warning(f"{Fore.YELLOW}[{func_name}] TSL edit failed {oid}: Order closed/cannot modify? ({type(e).__name__}): {e}{Style.RESET_ALL}"); else: logger.error(f"{Fore.RED}[{func_name}] TSL edit ExchangeError {oid}: {e}{Style.RESET_ALL}"); return False
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected TSL edit error {oid}: {e}{Style.RESET_ALL}",exc_info=True); return False

    def _log_position_pnl(self, position: Dict, exit_price: Optional[float], reason: str) -> None:
        # (Implementation from v6.2)
        func_name="_log_position_pnl"; pid=position.get('id','N/A'); side=position.get('side'); entry_p=position.get('entry_price'); size=position.get('size'); sym=position.get('symbol',self.symbol)
        if not all([pid!='N/A', side, entry_p, size, exit_price, sym]): logger.warning(f"{Fore.YELLOW}---> Pos {pid} Closed ({reason}). PnL calc skipped: Missing data.{Style.RESET_ALL}"); return
        if entry_p<=0 or exit_price<=0 or size<=0: logger.warning(f"{Fore.YELLOW}---> Pos {pid} Closed ({reason}). PnL calc skipped: Invalid data.{Style.RESET_ALL}"); return
        try:
            pnl_q=0.0; if side=='buy': pnl_q=(exit_price-entry_p)*size; elif side=='sell': pnl_q=(entry_p-exit_price)*size
            pnl_pct=0.0; entry_val=entry_p*size; if entry_val!=0: pnl_pct=(pnl_q/abs(entry_val))*100.0
            pnl_clr=Fore.GREEN if pnl_q>=0 else Fore.RED; quote_ccy=self.market_info['quote'] if self.market_info else 'Quote'; base_ccy=self.market_info['base'] if self.market_info else 'Base'
            log_msg=(f"{pnl_clr}{Style.BRIGHT}---> Position Closed: ID={pid} Sym={sym} Side={side.upper()} Reason='{reason}'\n"
                    f"     Entry: {self.format_price(entry_p)}, Exit: {self.format_price(exit_price)}, Size: {self.format_amount(size)} {base_ccy}\n"
                    f"     Est. PnL: {self.format_price(pnl_q)} {quote_ccy} ({pnl_pct:.3f}%){Style.RESET_ALL}")
            logger.info(log_msg); self.daily_pnl+=pnl_q; logger.info(f"Daily PnL Updated: {self.format_price(self.daily_pnl)} {quote_ccy}")
        except Exception as e: logger.error(f"[{func_name}] Error calculating PnL {pid}: {e}",exc_info=True)

    @retry_api_call(max_retries=1)
    def _place_market_close_order(self, position: Dict[str, Any], current_price: float) -> Optional[Dict[str, Any]]:
        # (Implementation from v6.2)
        func_name="_place_market_close_order"; pid=position.get('id'); side=position.get('side'); size=position.get('size'); sym=position.get('symbol',self.symbol); base_ccy=self.market_info['base']; quote_ccy=self.market_info['quote']
        if size is None or size<=1e-12: logger.error(f"[{func_name}] Cannot close {pid}: Invalid size {size}."); return None
        close_side='sell' if side=='buy' else 'buy'; log_clr=Fore.YELLOW; logger.warning(f"{log_clr}[{func_name}] Initiating MARKET CLOSE for {pid} ({sym}). Side:{close_side.upper()}, Size:{self.format_amount(size)}...{Style.RESET_ALL}")
        if self.simulation_mode: # Sim Logic from v6.2
            sim_cid=f"sim_close_{int(time.time()*1000)}_{close_side[:1]}"; sim_avg=current_price; sim_ts=int(time.time()*1000)
            sim_order={"id":sim_cid, "timestamp":sim_ts, "datetime":pd.to_datetime(sim_ts,unit='ms',utc=True).isoformat(timespec='milliseconds')+'Z', "symbol":sym, "type":"market", "side":close_side, "status":'closed', "amount":size, "filled":size, "remaining":0.0, "average":sim_avg, "cost":size*sim_avg, "reduceOnly":True, "info":{"simulated":True, "orderId":sim_cid, "reduceOnly":True, "closed_position_id":pid}}
            logger.info(f"{log_clr}[SIM] Market Close: ID {sim_cid}, Avg {self.format_price(sim_avg)}{Style.RESET_ALL}"); return sim_order
        else: # Live Mode
            try:
                params={'reduceOnly':True}; logger.debug(f"[{func_name}] Using reduceOnly=True.")
                if self.exchange_id.lower()=='bybit': # Add category hint
                    mkt=self.exchange.market(sym) if sym in self.exchange.markets else self.market_info
                    if mkt: mkt_t=mkt.get('type','swap'); is_l=mkt.get('linear',True); if mkt_t in ['swap','future']: params['category']='linear' if is_l else 'inverse'; elif mkt_t=='spot': params['category']='spot'
                logger.debug(f"[{func_name}] Placing live market close {pid}. Side:{close_side}, Size:{size}, Params:{params}")
                order=self.exchange.create_market_order(sym,close_side,size,params=params)
                if order: oid=order.get('id','N/A'); oavg=order.get('average'); ostat=order.get('status',STATUS_UNKNOWN); ofilled=order.get('filled',0.0); logger.info(f"{log_clr}---> LIVE Market Close Placed: ID:{oid}, Status:{ostat}, Filled:{self.format_amount(ofilled)}, Avg:{self.format_price(oavg)}{Style.RESET_ALL}"); return order
                else: logger.error(f"{Fore.RED}[{func_name}] LIVE Market Close API returned None for {pid}.{Style.RESET_ALL}"); return None
            except(ccxt.InsufficientFunds,ccxt.InvalidOrder) as e: logger.error(f"{Fore.RED}[{func_name}] LIVE Close {pid} Failed ({type(e).__name__}): {e}. Already closed?{Style.RESET_ALL}"); return None
            except ccxt.ExchangeError as e: err=str(e).lower(); if any(p in err for p in ["cost not meet","size is zero","reduce-only","position idx not match"]): logger.warning(f"{Fore.YELLOW}[{func_name}] Market close {pid} failed, likely already closed/reduce conflict. Err:{e}{Style.RESET_ALL}"); return None; else: logger.error(f"{Fore.RED}[{func_name}] LIVE Close ExchangeError {pid}: {e}{Style.RESET_ALL}"); return None
            except Exception as e: logger.error(f"{Fore.RED}[{func_name}] LIVE Close Unexpected {pid}: {e}{Style.RESET_ALL}",exc_info=True); return None

    # --- Main Bot Loop ---

    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetches price, order book, history."""
        # (Implementation from v6.2)
        func_name="_fetch_market_data"; logger.debug(f"[{func_name}] Fetching market data bundle..."); start_t=time.time()
        try:
            price=self.fetch_market_price(); ob_data=self.fetch_order_book(); hist_data=self.fetch_historical_data()
            if price is None: logger.warning(f"{Fore.YELLOW}[{func_name}] Failed price fetch.{Style.RESET_ALL}"); return None
            if hist_data is None or hist_data.empty: logger.warning(f"{Fore.YELLOW}[{func_name}] Failed history fetch.{Style.RESET_ALL}"); return None
            imb=ob_data.get('imbalance') if ob_data else None; dur=time.time()-start_t; logger.debug(f"[{func_name}] Data fetched in {dur:.2f}s.")
            return {"price":price, "order_book_imbalance":imb, "historical_data":hist_data}
        except Exception as e: logger.error(f"{Fore.RED}[{func_name}] Unexpected error: {e}{Style.RESET_ALL}", exc_info=True); return None

    def _calculate_indicators(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculates all technical indicators."""
        # (Implementation from v6.2)
        func_name="_calculate_indicators"; logger.debug(f"[{func_name}] Calculating indicators..."); start_t=time.time(); indicators={}
        if not isinstance(historical_data,pd.DataFrame) or historical_data.empty: return {}
        req_cols=['open','high','low','close','volume'];
        if not all(c in historical_data.columns for c in req_cols): return {}
        if len(historical_data)<2: logger.warning(f"[{func_name}] Short history ({len(historical_data)}).")
        c,h,l=historical_data['close'],historical_data['high'],historical_data['low']
        indicators['volatility']=self.calculate_volatility(c,self.volatility_window); indicators['ema']=self.calculate_ema(c,self.ema_period); indicators['rsi']=self.calculate_rsi(c,self.rsi_period)
        m_l,m_s,m_h=self.calculate_macd(c,self.macd_short_period,self.macd_long_period,self.macd_signal_period); indicators['macd_line'],indicators['macd_signal'],indicators['macd_hist']=m_l,m_s,m_h
        s_k,s_d=self.calculate_stoch_rsi(c,self.rsi_period,self.stoch_rsi_period,self.stoch_rsi_k_period,self.stoch_rsi_d_period); indicators['stoch_k'],indicators['stoch_d']=s_k,s_d
        indicators['atr']=self.calculate_atr(h,l,c,self.atr_period)
        dur=time.time()-start_t; log_msg=(f"Indicators (calc {dur:.2f}s): EMA={self.format_price(indicators.get('ema'))} RSI={self.format_indicator(indicators.get('rsi'),1)} ATR={self.format_price(indicators.get('atr'))} MACD={self.format_price(indicators.get('macd_line'),self.price_decimals+1)}/{self.format_price(indicators.get('macd_signal'),self.price_decimals+1)} Stoch={self.format_indicator(indicators.get('stoch_k'),1)}/{self.format_indicator(indicators.get('stoch_d'),1)} Vol={self.format_indicator(indicators.get('volatility'),5)}"); logger.info(log_msg)
        nones=[k for k,v in indicators.items() if v is None]; if nones: logger.warning(f"{Fore.YELLOW}[{func_name}] Indicators returning None: {nones}.{Style.RESET_ALL}"); return indicators

    def _process_signals_and_entry(self, market_data: Dict, indicators: Dict) -> None:
        """Analyzes signals, checks conditions, and potentially places entry order."""
        # (Implementation from v6.2)
        func_name="_process_signals_and_entry"; price=market_data['price']; imb=market_data['order_book_imbalance']; atr=indicators.get('atr')
        active_or_pending=sum(1 for p in self.open_positions if p.get('status') in [STATUS_ACTIVE,STATUS_PENDING_ENTRY])
        if active_or_pending>=self.max_open_positions: logger.info(f"{Fore.CYAN}[{func_name}] Max positions ({self.max_open_positions}) reached. Skipping entry.{Style.RESET_ALL}"); return
        score,reasons=self.compute_trade_signal_score(price,indicators,imb); score_clr=Fore.GREEN if score>0 else(Fore.RED if score<0 else Fore.WHITE); logger.info(f"Trade Signal Score: {score_clr}{score}{Style.RESET_ALL}")
        if logger.isEnabledFor(logging.DEBUG) or abs(score)>=ENTRY_SIGNAL_THRESHOLD_ABS:
            for r in reasons: logger.debug(f"  -> {r}")
        entry_side:Optional[str]=None; if score>=ENTRY_SIGNAL_THRESHOLD_ABS: entry_side='buy'; elif score<=-ENTRY_SIGNAL_THRESHOLD_ABS: entry_side='sell'
        if not entry_side: logger.info(f"[{func_name}] Neutral signal ({score}). Threshold ({ENTRY_SIGNAL_THRESHOLD_ABS}) not met."); return
        log_clr=Fore.GREEN if entry_side=='buy' else Fore.RED; logger.info(f"{log_clr}[{func_name}] {entry_side.upper()} signal score ({score}) meets threshold. Preparing entry...{Style.RESET_ALL}")
        size=self.calculate_order_size(price,indicators,score)
        if size<=0: logger.warning(f"{Fore.YELLOW}[{func_name}] Entry aborted: Order size zero/invalid.{Style.RESET_ALL}"); return
        sl,tp=self._calculate_sl_tp_prices(price,entry_side,price,atr)
        sl_tp_req=self.use_atr_sl_tp or (self.base_stop_loss_pct is not None and self.base_take_profit_pct is not None); sl_tp_fail=(sl is None and tp is None)
        if sl_tp_req and sl_tp_fail: logger.error(f"{Fore.RED}[{func_name}] Entry aborted: Required SL/TP calculation failed.{Style.RESET_ALL}"); return
        elif sl_tp_fail: logger.warning(f"{Fore.YELLOW}[{func_name}] SL/TP calculation failed. Proceeding without SL/TP params.{Style.RESET_ALL}")
        entry_order=self.place_entry_order(side=entry_side,order_size_base=size,confidence_level=score,order_type=self.entry_order_type,current_price=price,stop_loss_price=sl,take_profit_price=tp)
        if entry_order:
            oid=entry_order.get('id'); if not oid: logger.critical(f"{Fore.RED}[{func_name}] Entry order placed but NO ID received! Cannot track! Resp:{str(entry_order)[:200]}{Style.RESET_ALL}"); return
            stat=entry_order.get('status'); init_stat=STATUS_UNKNOWN; filled_imm=False
            if stat=='open': init_stat=STATUS_PENDING_ENTRY
            elif stat=='closed': init_stat=STATUS_ACTIVE; filled_imm=True
            elif stat in ['canceled','rejected','expired']: logger.warning(f"{Fore.YELLOW}[{func_name}] Entry order {oid} failed immediately ({stat}). Not adding position.{Style.RESET_ALL}"); return
            else: init_stat=STATUS_PENDING_ENTRY; logger.warning(f"[{func_name}] Entry {oid} unusual initial status '{stat}'. Treating as PENDING.")
            fill_amt=float(entry_order.get('filled',0.0)); avg_fill=float(entry_order.get('average')) if entry_order.get('average') else None; ts_ms=entry_order.get('timestamp'); req_amt=float(entry_order.get('amount',size))
            entry_p_state=avg_fill if filled_imm and avg_fill else None; entry_t_state=ts_ms/1000.0 if filled_imm and ts_ms else None; size_state=fill_amt if filled_imm else 0.0
            new_pos={"id":oid, "symbol":self.symbol, "side":entry_side, "size":size_state, "original_size":req_amt, "entry_price":entry_p_state, "entry_time":entry_t_state, "status":init_stat, "entry_order_type":self.entry_order_type, "stop_loss_price":sl, "take_profit_price":tp, "confidence":score, "trailing_stop_price":None, "last_update_time":time.time()}
            self.open_positions.append(new_pos); logger.info(f"{Fore.CYAN}---> Position {oid} added. Status:{init_stat}, Entry:{self.format_price(entry_p_state)}{Style.RESET_ALL}"); self._save_state()
        else: logger.error(f"{Fore.RED}[{func_name}] Entry order placement failed (API call error).{Style.RESET_ALL}")

    def run(self) -> None:
        """Starts the main trading loop."""
        # (Implementation from v6.2)
        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Initiating Trading Loop (Symbol: {self.symbol}, TF: {self.timeframe}) ---{Style.RESET_ALL}")
        while True:
            self.iteration+=1; start_t=time.time(); ts_now=pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds'); loop_pfx=f"{Fore.BLUE}===== Iter {self.iteration} ====={Style.RESET_ALL}"; logger.info(f"\n{loop_pfx} Timestamp: {ts_now}")
            try:
                market_data=self._fetch_market_data()
                if market_data is None: time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue
                price=market_data['price']; logger.info(f"{loop_pfx} Price: {self.format_price(price)} OB Imb: {self.format_indicator(market_data.get('order_book_imbalance'),3)}")
                indicators=self._calculate_indicators(market_data['historical_data'])
                if not indicators: time.sleep(DEFAULT_SLEEP_INTERVAL_SECONDS); continue
                self._check_pending_entries(indicators) # Check fills first
                self._manage_active_positions(price,indicators) # Manage exits/TSL
                self._process_signals_and_entry(market_data,indicators) # Check for new entry
                exec_t=time.time()-start_t; wait_t=max(0.1,DEFAULT_SLEEP_INTERVAL_SECONDS-exec_t); logger.debug(f"{loop_pfx} Loop took {exec_t:.2f}s. Waiting {wait_t:.2f}s...")
                time.sleep(wait_t)
            except KeyboardInterrupt: logger.warning(f"\n{Fore.YELLOW}Keyboard interrupt. Initiating shutdown...{Style.RESET_ALL}"); break
            except SystemExit as e: logger.warning(f"SystemExit code {e.code}. Exiting loop..."); raise e
            except Exception as e: logger.critical(f"{Fore.RED}{Style.BRIGHT}{loop_pfx} CRITICAL UNHANDLED ERROR: {type(e).__name__} - {e}{Style.RESET_ALL}",exc_info=True); pause=60; logger.warning(f"{Fore.YELLOW}Pausing {pause}s...{Style.RESET_ALL}"); try: time.sleep(pause); except KeyboardInterrupt: break
        logger.info("Main trading loop terminated.")

    def shutdown(self):
        """Graceful shutdown: cancel pending, optionally close active, save state."""
        # (Implementation from v6.2)
        func_name="shutdown"; logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}--- Initiating Graceful {func_name} ---{Style.RESET_ALL}"); needs_save=False
        pending=[p for p in list(self.open_positions) if p.get('status')==STATUS_PENDING_ENTRY]; cancelled_p=0
        if pending:
            logger.info(f"[{func_name}] Cancelling {len(pending)} PENDING orders...")
            for p in pending: pid=p.get('id'); psym=p.get('symbol',self.symbol); if not pid: continue; logger.info(f"Cancelling pending {pid} ({psym})...");
            if self.cancel_order_by_id(pid,symbol=psym): # Uses sim logic internally too
                cancelled_p+=1; idx=next((i for i,x in enumerate(self.open_positions) if x.get('id')==pid),-1); # Find original to update
                if idx!=-1: self.open_positions[idx]['status']=STATUS_CANCELED; self.open_positions[idx]['last_update_time']=time.time(); self.open_positions[idx]['exit_reason']="Cancelled on Shutdown"; needs_save=True; else: logger.warning(f"Cannot find {pid} in state after cancel success?")
            else: logger.error(f"{Fore.RED}Failed cancel pending {pid}. Manual check needed.{Style.RESET_ALL}")
            logger.info(f"[{func_name}] Pending cancel done. Cancelled/Gone: {cancelled_p}/{len(pending)}.")
        active=[p for p in self.open_positions if p.get('status')==STATUS_ACTIVE]
        if self.close_positions_on_exit and active:
            logger.warning(f"{Fore.YELLOW}[{func_name}] Closing {len(active)} ACTIVE positions (close_positions_on_exit=True)...{Style.RESET_ALL}"); price=self.fetch_market_price()
            if price is None: logger.critical(f"{Fore.RED}{Style.BRIGHT}CRITICAL: Cannot fetch price for market close! {len(active)} positions remain OPEN! Manual action required!{Style.RESET_ALL}")
            else: closed_c=0; failed_ids=[]
            for pos in list(active): # Iterate copy
                pid=pos.get('id'); psym=pos.get('symbol',self.symbol); logger.info(f"Attempting market close active {pid} ({psym})..."); close_order=self._place_market_close_order(pos,price)
                if close_order: closed_c+=1; logger.info(f"{Fore.YELLOW}---> Market close OK {pid}. Marking '{STATUS_CLOSED_ON_EXIT}'.{Style.RESET_ALL}"); pos['status']=STATUS_CLOSED_ON_EXIT; pos['last_update_time']=time.time(); needs_save=True; close_fill=close_order.get('average') or price; self._log_position_pnl(pos,close_fill,f"Closed on Exit (Order {close_order.get('id')})")
                else: failed_ids.append(pid); logger.error(f"{Fore.RED}{Style.BRIGHT}CRITICAL FAIL close active {pid}. REMAINS OPEN! Manual action!{Style.RESET_ALL}")
            logger.info(f"[{func_name}] Active close attempt: Closed {closed_c}/{len(active)}. Failed: {len(failed_ids)}."); if failed_ids: logger.error(f"Failed close IDs: {failed_ids}. MANUAL ACTION!")
        elif active:
            logger.warning(f"{Fore.YELLOW}[{func_name}] {len(active)} position(s) remain active ('close_positions_on_exit'=false):{Style.RESET_ALL}")
            for p in active: logger.warning(f" -> ID:{p.get('id')}, Sym:{p.get('symbol')}, Side:{p.get('side')}, Size:{self.format_amount(p.get('size'))}, Entry:{self.format_price(p.get('entry_price'))}")
        if needs_save: logger.info(f"[{func_name}] Saving final state..."); self._save_state()
        else: logger.info(f"[{func_name}] No state changes during shutdown.")
        logger.info(f"{Fore.CYAN}{Style.BRIGHT}--- Scalping Bot Shutdown Complete ---{Style.RESET_ALL}")
        logging.shutdown()

# --- Main Execution Block ---
if __name__ == "__main__":
    bot_instance: Optional[ScalpingBot] = None
    exit_code: int = 0

    try:
        # Pre-run setup: Ensure state directory exists
        state_dir = os.path.dirname(STATE_FILE_NAME)
        if state_dir and not os.path.exists(state_dir):
            try:
                print(f"{Fore.CYAN}Creating state directory: {state_dir}{Style.RESET_ALL}")
                os.makedirs(state_dir)
            except OSError as e:
                 print(f"{Fore.RED}Fatal: Could not create state directory '{state_dir}': {e}. Aborting.{Style.RESET_ALL}", file=sys.stderr)
                 sys.exit(1)

        # Initialize Bot (handles config, validation, exchange, state, market info)
        bot_instance = ScalpingBot(config_file=CONFIG_FILE_NAME, state_file=STATE_FILE_NAME)
        bot_instance.run() # Start main loop

    except KeyboardInterrupt:
        logger.warning(f"\n{Fore.YELLOW}Shutdown signal detected (KeyboardInterrupt in main).{Style.RESET_ALL}")
        exit_code = 130
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if exit_code not in [0, 130]: logger.error(f"Bot exited via SystemExit with unexpected code: {exit_code}.")
        else: logger.info(f"Bot exited via SystemExit with code: {exit_code}.")
    except Exception as e:
        logger.critical(f"{Fore.RED}{Style.BRIGHT}Unhandled critical error occurred outside main loop: {type(e).__name__} - {e}{Style.RESET_ALL}", exc_info=True)
        exit_code = 1
    finally:
        logger.info("Initiating final shutdown procedures...")
        if bot_instance:
            try: bot_instance.shutdown()
            except Exception as shutdown_err:
                 # Use print as logger might be closed by shutdown()
                 print(f"{Fore.RED}Error during final bot shutdown procedure: {shutdown_err}{Style.RESET_ALL}", file=sys.stderr)
                 logging.shutdown() # Attempt final log shutdown
        else:
            # If bot instance creation failed, just ensure logging is shut down
            print(f"{Fore.YELLOW}Bot instance not fully initialized. Shutting down logging.{Style.RESET_ALL}")
            logging.shutdown()

        print(f"\n{Fore.MAGENTA}Pyrmethus Bot v6.2 has concluded its watch. Exit Code: {exit_code}{Style.RESET_ALL}")
        sys.exit(exit_code)
