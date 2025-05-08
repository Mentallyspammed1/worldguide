Okay, let's implement the enhancements based on the detailed plan, addressing the `SyntaxError`, improving precision with `Decimal`, enhancing robustness, integrating Bybit V5 specifics, refining configuration, logging, status display, journaling, notifications, and code clarity.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ██████╗ ██╗   ██╗██████╗ ███╗   ███╗███████╗ ██████╗ █████╗ ██╗     ██████╗
# ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗
# ██████╔╝ ╚████╔╝ ██████╔╝██╔████╔██║███████╗██║     ███████║██║     ██████╔╝
# ██╔═══╝   ╚██╔╝  ██╔══██╗██║╚██╔╝██║╚════██║██║     ██╔══██║██║     ██╔═══╝
# ██║        ██║   ██║  ██║██║ ╚═╝ ██║███████║╚██████╗██║  ██║███████╗██║
# ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝
# Pyrmethus - Termux Trading Spell (v2.2.1 - Syntax Fix & Enhanced Precision)
# Conjures market insights and executes trades on Bybit Futures (V5 API)
# Incorporates robust configuration, multi-condition signals, position-based stops,
# enhanced error handling, Decimal precision, and basic entry journaling.

import os
import time
import logging
import sys
import subprocess
import csv # For Journaling
from datetime import datetime # For Journaling timestamp
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero
import copy
import textwrap # For wrapping signal reason

# Attempt to import necessary enchantments
try:
    import ccxt
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import init, Fore, Style, Back
    import requests
    # Explicitly list common required packages for the error message
    COMMON_PACKAGES = ['ccxt', 'python-dotenv', 'pandas', 'numpy', 'tabulate', 'colorama', 'requests']
except ImportError as e:
    # Provide specific guidance for Termux users
    init(autoreset=True)
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell in your Termux terminal:")
    print(f"{Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}")
    # Offer to install all common dependencies
    print(f"\n{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
    print(f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
    sys.exit(1)

# Weave the Colorama magic into the terminal
init(autoreset=True)

# Set Decimal precision (adjust if needed, higher precision means more memory/CPU)
# The default precision (usually 28) is often sufficient when combined with quantize.
# Increase only if necessary for extremely small values or high precision instruments.
# getcontext().prec = 50 # Example: Increase precision if needed. Default is usually fine.

# --- Arcane Configuration & Logging Setup ---
logger = logging.getLogger(__name__) # Define logger early for config class

# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

def trade_log(self, message, *args, **kws):
    """Custom logging method for trade-related events."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)

# Add the custom method to the Logger class if it doesn't exist
if not hasattr(logging.Logger, 'trade'):
    logging.Logger.trade = trade_log

# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)-8s] " # Padded levelname
    + Fore.WHITE + "(%(filename)s:%(lineno)d) " # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s"
)
# Set level via environment variable or default to INFO
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Ensure handlers are not duplicated if script is reloaded (e.g., in some interactive environments)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
else:
    # Check if a handler already exists to avoid duplicates
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
         stream_handler = logging.StreamHandler(sys.stdout)
         stream_handler.setFormatter(log_formatter)
         logger.addHandler(stream_handler)


# Prevent duplicate messages if the root logger is also configured (common issue)
logger.propagate = False


class TradingConfig:
    """Holds the sacred parameters of our spell, enhanced with precision awareness and validation."""
    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        # Default symbol format for Bybit V5 Unified is BASE/QUOTE:SETTLE, e.g., BTC/USDT:USDT
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=['linear', 'inverse', 'swap']).lower() # 'linear', 'inverse' or 'swap'
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        # Risk as a percentage of total equity (e.g., 0.01 for 1%, 0.001 for 0.1%)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5")) # 0.001% to 50% risk
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")) # Reasonable bounds
        # TSL activation threshold in ATR units above entry price
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")) # Reasonable bounds
        # Bybit V5 TSL distance is a percentage (e.g., 0.5 for 0.5%). Ensure value is suitable.
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0")) # 0.001% to 10% trail
        # Trigger type for SL/TSL orders. Bybit V5 allows LastPrice, MarkPrice, IndexPrice.
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]) # Usually same as SL

        # --- Optimized Indicator Periods (Read from .env with new recommended defaults) ---
        self.trend_ema_period = self._get_env("TREND_EMA_PERIOD", "12", Fore.YELLOW, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period = self._get_env("FAST_EMA_PERIOD", "9", Fore.YELLOW, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period = self._get_env("SLOW_EMA_PERIOD", "21", Fore.YELLOW, cast_type=int, min_val=2, max_val=500)
        self.stoch_period = self._get_env("STOCH_PERIOD", "7", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k = self._get_env("STOCH_SMOOTH_K", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d = self._get_env("STOCH_SMOOTH_D", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.atr_period = self._get_env("ATR_PERIOD", "5", Fore.YELLOW, cast_type=int, min_val=1, max_val=100)

        # --- Signal Logic Thresholds (Configurable) ---
        self.stoch_oversold_threshold = self._get_env("STOCH_OVERSOLD_THRESHOLD", "30", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", "70", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        # Loosened Trend Filter Threshold (price within X% of Trend EMA)
        self.trend_filter_buffer_percent = self._get_env("TREND_FILTER_BUFFER_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        # ATR Filter Threshold (price move must be > X * ATR)
        self.atr_move_filter_multiplier = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))

        # Epsilon: Small fixed value for comparing quantities to zero.
        # Derived dynamically from market step size can be complex and error-prone.
        # A tiny fixed value is generally safe for typical crypto precision.
        self.position_qty_epsilon = Decimal("1E-12")
        logger.debug(f"Using fixed position_qty_epsilon: {self.position_qty_epsilon:.1E}")


        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)
        self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
        self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)
        # --- Journaling Configuration ---
        self.journal_file_path = self._get_env("JOURNAL_FILE_PATH", "bybit_trading_journal.csv", Fore.YELLOW)
        self.enable_journaling = self._get_env("ENABLE_JOURNALING", "True", Fore.YELLOW, cast_type=bool)


        if not self.api_key or not self.api_secret:
            logger.critical(Fore.RED + Style.BRIGHT + "BYBIT_API_KEY or BYBIT_API_SECRET not found in .env scroll! Halting.")
            sys.exit(1)

        # --- Post-Load Validations ---
        # Validate EMA periods relative to each other
        if self.fast_ema_period >= self.slow_ema_period:
             logger.critical(f"{Fore.RED+Style.BRIGHT}FAST_EMA_PERIOD ({self.fast_ema_period}) must be less than SLOW_EMA_PERIOD ({self.slow_ema_period}). Halting.")
             sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
             logger.warning(f"{Fore.YELLOW}TREND_EMA_PERIOD ({self.trend_ema_period}) is not significantly longer than SLOW_EMA_PERIOD ({self.slow_ema_period}). Consider increasing TREND_EMA_PERIOD for a smoother trend filter.")

        # Validate Stochastic thresholds relative to each other
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
             logger.critical(f"{Fore.RED+Style.BRIGHT}STOCH_OVERSOLD_THRESHOLD ({self.stoch_oversold_threshold.normalize()}) must be less than STOCH_OVERBOUGHT_THRESHOLD ({self.stoch_overbought_threshold.normalize()}). Halting.")
             sys.exit(1)

        # Validate TSL activation relative to SL multiplier
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
             logger.warning(f"{Fore.YELLOW}TSL_ACTIVATION_ATR_MULTIPLIER ({self.tsl_activation_atr_multiplier.normalize()}) is less than SL_ATR_MULTIPLIER ({self.sl_atr_multiplier.normalize()}). TSL may activate before the fixed SL is surpassed if market moves slowly.")

        logger.debug("Configuration loaded successfully.")

    def _get_env(self, key: str, default: Any, color: str, cast_type: type = str,
                 min_val: Optional[Union[int, Decimal]] = None,
                 max_val: Optional[Union[int, Decimal]] = None,
                 allowed_values: Optional[List[str]] = None) -> Any:
        """Gets value from environment, casts, validates, and logs."""
        value_str = os.getenv(key)
        is_default = False
        # Mask secrets in logs
        log_value = "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str

        if value_str is None or value_str.strip() == "": # Treat empty string as not set
            value = default
            is_default = True
            if default is not None:
                # Log default only if it's actually used because the key wasn't set
                logger.warning(f"{color}Using default value for {key}: {default}")
            # Use default value string for casting below if needed
            value_str = str(default) if default is not None else None
        else:
             # Log the fetched value (masked if needed)
             logger.info(f"{color}Summoned {key}: {log_value}")

        # Handle case where default is None and no value is set
        if value_str is None:
            if default is None:
                # If no value is set and default is None, it means the config is missing a required value.
                # This should ideally be caught by explicit checks after config loading (like API keys).
                # But for robustness, return None and let caller handle.
                return None
            else:
                # This case should be covered by the is_default logic above, but double check
                logger.warning(f"{color}Value for {key} not found, using default: {default}")
                value = default # Keep original default value/type
                value_str = str(default) # Use string representation for casting attempt below

        # --- Casting ---
        casted_value = None
        try:
            if cast_type == bool:
                # Robust boolean check
                casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on'] if value_str is not None else bool(default) # Handle default boolean
            elif cast_type == Decimal:
                casted_value = Decimal(value_str) if value_str is not None else Decimal(str(default)) # Handle default Decimal
            elif cast_type == int:
                casted_value = int(value_str) if value_str is not None else int(default) # Handle default int
            elif cast_type == float:
                # Warn against using float for critical financial values
                if key in ["RISK_PERCENTAGE", "SL_ATR_MULTIPLIER", "TSL_ACTIVATION_ATR_MULTIPLIER", "TRAILING_STOP_PERCENT", "TREND_FILTER_BUFFER_PERCENT", "ATR_MOVE_FILTER_MULTIPLIER"]:
                     logger.warning(f"{Fore.YELLOW}Using float for critical config '{key}'. Consider using Decimal for better precision.")
                casted_value = float(value_str) if value_str is not None else float(default) # Handle default float
            else: # Default is str
                casted_value = str(value_str) if value_str is not None else str(default) # Handle default str
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Attempting to use default value '{default}'.")
            # Attempt to cast the default value itself
            try:
                if default is None: return None # If default was None, casting failed, return None
                # Recast default carefully to the target type
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}: {cast_default_err}. Halting.")
                sys.exit(1)

        # --- Validation ---
        if casted_value is None: # Should not happen if casting succeeded or defaulted
             logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to obtain a valid value for {key} after casting attempts. Halting.")
             sys.exit(1)

        # Allowed values check (case-insensitive for strings)
        if allowed_values:
            # Convert casted_value to lower for comparison if it's a string
            comp_value = casted_value.lower() if isinstance(casted_value, str) else casted_value
            lower_allowed = [v.lower() for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}")
                # Return default after logging error
                try:
                    if default is None: return None
                    if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                    if cast_type == Decimal: return Decimal(str(default))
                    return cast_type(default)
                except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                    logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__} for validation fallback: {cast_default_err}. Halting.")
                    sys.exit(1)


        # Min/Max checks (for numeric types - Decimal, int, float)
        validation_failed = False
        if isinstance(casted_value, (Decimal, int, float)):
            try:
                # Ensure min_val/max_val are comparable types (Decimal if casted is Decimal, else same type)
                # Convert min_val/max_val to string first to handle potential float inputs before Decimal conversion
                min_val_comp = Decimal(str(min_val)) if isinstance(casted_value, Decimal) and min_val is not None else min_val
                max_val_comp = Decimal(str(max_val)) if isinstance(casted_value, Decimal) and max_val is not None else max_val

                if min_val_comp is not None and casted_value < min_val_comp:
                    logger.error(f"{Fore.RED}{key} value {casted_value.normalize() if isinstance(casted_value, Decimal) else casted_value} is below minimum {min_val.normalize() if isinstance(min_val, Decimal) else min_val}. Using default: {default}")
                    validation_failed = True
                if max_val_comp is not None and casted_value > max_val_comp:
                     logger.error(f"{Fore.RED}{key} value {casted_value.normalize() if isinstance(casted_value, Decimal) else casted_value} is above maximum {max_val.normalize() if isinstance(max_val, Decimal) else max_val}. Using default: {default}")
                     validation_failed = True
            except InvalidOperation as e:
                 logger.error(f"{Fore.RED}Error during min/max validation for {key} with value {casted_value} and limits ({min_val}, {max_val}): {e}. Using default: {default}")
                 validation_failed = True
            except TypeError as e:
                 logger.error(f"{Fore.RED}TypeError during min/max validation for {key}: {e}. Value type: {type(casted_value).__name__}, Limit types: {type(min_val).__name__}/{type(max_val).__name__}. Using default: {default}")
                 validation_failed = True


        if validation_failed:
            # Re-cast default to ensure correct type is returned
            try:
                if default is None: return None
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if cast_type == Decimal: return Decimal(str(default))
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}: {cast_default_err}. Halting.")
                sys.exit(1)

        return casted_value

# --- Instantiate Configuration ---
logger.info(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.2.1...")
# Summon secrets from the .env scroll
load_dotenv()
# Instantiate CONFIG *after* logging is set up and dotenv is loaded
CONFIG = TradingConfig()

# --- Global Variables ---
# Declared at module level, but need `global` inside functions if assigned to.
# Accessing only doesn't strictly require `global` in Python 3, but adding it can sometimes improve clarity
# or avoid the specific SyntaxError encountered if the interpreter's analysis is complex.
# Let's be explicit and add global where modification happens.
MARKET_INFO: Optional[Dict] = None # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None # Global for the exchange instance
# Tracks active SL/TSL *presence* using markers for position-based stops (no order IDs for these in Bybit V5)
# This tracker is crucial for knowing if a position *should* have a stop attached.
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None}, # sl_id/tsl_id stores marker (e.g., "POS_SL_LONG") or None
    "short": {"sl_id": None, "tsl_id": None}
}

# --- Core Spell Functions ---

def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    """Generic wrapper to fetch data with retries and exponential backoff."""
    # global EXCHANGE, CONFIG # Only need global if ASSIGNING to EXCHANGE or CONFIG
    # Accessing EXCHANGE and CONFIG is fine without global here.

    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    # Check if 'params' is a dict before attempting to add category
    if 'params' not in kwargs or not isinstance(kwargs['params'], dict):
        kwargs['params'] = {}
    # Ensure category is set if it's a V5 exchange and category is configured
    # Check for EXCHANGE.options and its structure defensively
    # Also ensure it's a method that actually *uses* category (most V5 unified do)
    method_name = getattr(fetch_function, '__name__', str(fetch_function))
    methods_using_category = [
        'fetch_ohlcv', 'fetch_ticker', 'fetch_order_book', 'fetch_trades',
        'fetch_balance', 'fetch_positions', 'fetch_order', 'fetch_open_orders',
        'create_market_order', 'create_limit_order', 'cancel_order', 'cancel_orders',
        'private_post_order_cancel_all', 'private_post_position_set_trading_stop'
    ] # Add other V5 methods requiring category as needed

    if hasattr(EXCHANGE, 'options') and isinstance(EXCHANGE.options, dict) and 'v5' in EXCHANGE.options and isinstance(EXCHANGE.options['v5'], dict) and 'category' in EXCHANGE.options['v5'] and method_name in methods_using_category:
         if 'category' not in kwargs['params']: # Only add if not explicitly provided
             kwargs['params']['category'] = EXCHANGE.options['v5']['category']
             # logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {method_name}")

    for attempt in range(CONFIG.max_fetch_retries + 1): # +1 to allow logging final failure
        try:
            # Log the attempt number and function being called at DEBUG level
            # Simple masking for common sensitive keys in dict params
            log_kwargs = {}
            for k, v in kwargs.items():
                 if isinstance(v, dict):
                      log_kwargs[k] = {vk: ('****' if isinstance(vk, str) and ('secret' in vk.lower() or 'key' in vk.lower() or 'password' in vk.lower()) else vv) for vk, vv in v.items()}
                 else:
                      log_kwargs[k] = '****' if isinstance(k, str) and ('secret' in k.lower() or 'key' in k.lower() or 'password' in k.lower() or (isinstance(v, str) and len(v) > 30 and ('sig' in k.lower() or 'sign' in k.lower()))) else v
            # Mask args if they look like keys/secrets (less common, but defensive)
            log_args = ['****' if isinstance(a, str) and ('secret' in a.lower() or 'key' in a.lower() or 'password' in a.lower() or len(a) > 30) else a for a in args]
            logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {method_name} with args={log_args}, kwargs={log_kwargs}")

            result = fetch_function(*args, **kwargs)
            return result # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2 ** attempt # Exponential backoff (1, 2, 4, 8...)
            logger.warning(Fore.YELLOW + f"{method_name}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}")
            if attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            else:
                logger.error(Fore.RED + f"{method_name}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues.")
                # On final failure, return None. Caller handles the consequence.
                return None
        except ccxt.ExchangeNotAvailable as e:
             last_exception = e
             logger.error(Fore.RED + f"{method_name}: Exchange not available: {e}. Stopping retries.")
             # This is usually a hard stop, no point retrying
             return None # Indicate failure
        except ccxt.AuthenticationError as e:
             last_exception = e
             logger.critical(Fore.RED + Style.BRIGHT + f"{method_name}: Authentication error: {e}. Halting script.")
             sys.exit(1) # Exit immediately on auth failure
        except (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied, ccxt.NotSupported) as e:
            # These are typically non-retryable errors related to the request parameters or exchange state, or unsupported features.
            last_exception = e
            error_type = type(e).__name__
            logger.error(Fore.RED + f"{method_name}: Non-retryable error ({error_type}): {e}. Stopping retries for this call.")
            # Re-raise these specific errors so the caller can handle them appropriately
            raise e
        except ccxt.ExchangeError as e:
            # Includes rate limit errors, potentially invalid requests etc.
            last_exception = e
            # Attempt to extract Bybit V5 error code and message from the exception's info dictionary
            error_code = None
            error_message = str(e)
            if hasattr(e, 'info') and isinstance(e.info, dict):
                 info_data = e.info
                 # Check for Bybit V5 structure: info -> retCode, retMsg
                 if 'retCode' in info_data:
                      error_code = info_data.get('retCode')
                      error_message = info_data.get('retMsg', error_message)
                 # Fallback to checking general info keys if V5 structure isn't present
                 elif 'code' in info_data:
                      error_code = info_data.get('code')
                      error_message = info_data.get('msg', error_message) # Common msg key

            should_retry = True
            wait_time = 2 * (attempt + 1) # Default backoff

            # Check for common rate limit codes/messages
            # Bybit V5: 10009 (Freq), 10017 (Key limit), 10018 (IP limit), 10020 (too many requests)
            if "Rate limit exceeded" in error_message or error_code in [10017, 10018, 10009, 10020]:
                 wait_time = 5 * (attempt + 1) # Longer wait for rate limits
                 logger.warning(f"{Fore.YELLOW}{method_name}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s... Error: {error_message}")
            # Check for specific non-retryable errors (e.g., invalid parameter codes, state issues)
            # Bybit V5 Invalid Parameter codes often start with 11xxxx or others indicating bad input/state
            # 30034: Position status not normal, 110025: SL/TP order not found (might be benign for TSL activation attempt)
            # 110001: Parameter error, 110006: Invalid price precision, 110007: Invalid order quantity
            # 110041: Order quantity exceeds limit, 110042: Order price exceeds limit
            # 110013: Insufficient balance, 110017: Order amount lower than min notional
            # 30042: Risk limit error (e.g., trying to open too large position)
            elif error_code is not None and (
                (110000 <= error_code <= 110100 and error_code not in [110025]) or # Exclude 110025 from automatic non-retry
                error_code in [30034, 30042]
            ): # Add other known non-retryable codes
                 logger.error(Fore.RED + f"{method_name}: Non-retryable parameter/logic exchange error (Code: {error_code}): {error_message}. Stopping retries.")
                 should_retry = False
                 # Re-raise these specific errors so the caller can handle them appropriately
                 raise e # Re-raise immediately
            # Handle the 110025 case specifically - it might be retryable if it's a timing issue, or not if the SL/TP genuinely doesn't exist.
            # For now, treat 110025 as potentially retryable by default ExchangeError logic unless specified otherwise by the caller context (e.g., TSL logic).
            # If a caller needs to handle 110025 specially (like in TSL management), they should catch ExchangeError and check the code.
            else:
                 # General exchange error, apply default backoff
                 logger.warning(f"{Fore.YELLOW}{method_name}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {error_message}")

            if should_retry and attempt < CONFIG.max_fetch_retries:
                time.sleep(wait_time)
            elif should_retry: # Final attempt failed
                 logger.error(Fore.RED + f"{method_name}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to exchange errors.")
                 break # Exit retry loop, will return None
            else: # Non-retryable error encountered (and not re-raised above)
                 break # Exit retry loop, will return None

        except Exception as e:
            # Catch-all for unexpected errors
            last_exception = e
            logger.error(Fore.RED + f"{method_name}: Unexpected shadow encountered: {e}", exc_info=True)
            break # Stop on unexpected errors

    # If loop finished without returning, it means all retries failed or a break occurred
    # Re-raise the last specific non-retryable exception if it wasn't already
    if isinstance(last_exception, (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.BadRequest, ccxt.PermissionDenied, ccxt.NotSupported)):
         # These should have been re-raised inside the loop, but defensive check
         raise last_exception # Propagate specific non-retryable errors

    # For other failures (Network, ExchangeNotAvailable, general ExchangeError after retries), return None
    if last_exception:
         logger.error(f"{method_name} ultimately failed after retries or encountered a non-retryable error type not explicitly re-raised.")
         return None # Indicate failure

    # Should not reach here if successful, but defensive return None
    return None

# --- Exchange Nexus Initialization ---
logger.info(Fore.MAGENTA + Style.BRIGHT + "\nEstablishing Nexus with the Exchange v2.2.1...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True, # CCXT built-in rate limiter
        "options": {
            'defaultType': 'swap', # More specific for futures/swaps than 'future'
            'defaultSubType': CONFIG.market_type, # 'linear' or 'inverse' or 'swap'
            'adjustForTimeDifference': True, # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            'brokerId': 'PyrmethusV221', # Custom identifier for Bybit API tracking
            'v5': {'category': CONFIG.market_type} # Explicitly set category for V5 requests where applicable
        }
    }
    # Log options excluding secrets for debugging
    log_options = exchange_options.copy()
    log_options['apiKey'] = '****'
    log_options['secret'] = '****'
    # Ensure nested options are also masked if they contain sensitive info
    log_options['options'] = copy.deepcopy(exchange_options['options'])
    if 'v5' in log_options['options']: # Defensive check
         if 'apiKey' in log_options['options']['v5']: log_options['options']['v5']['apiKey'] = '****'
         if 'secret' in log_options['options']['v5']: log_options['options']['v5']['secret'] = '****'

    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials (important!)
    logger.info("Verifying credentials and connection...")
    EXCHANGE.check_required_credentials() # Checks if keys are present/formatted ok
    logger.info("Credentials format check passed.")
    # Fetch time to verify connectivity, API key validity, and clock sync
    # Use fetch_time with retries
    server_time_ms = fetch_with_retries(EXCHANGE.fetch_time)
    if server_time_ms is None:
         logger.critical(Fore.RED + Style.BRIGHT + "Failed to fetch server time after retries. Cannot verify connection/credentials. Halting.")
         sys.exit(1)

    local_time_ms = EXCHANGE.milliseconds()
    time_diff_ms = abs(server_time_ms - local_time_ms)
    logger.info(f"Exchange time synchronized: {EXCHANGE.iso8601(server_time_ms)} (Difference: {time_diff_ms} ms)")
    if time_diff_ms > 5000: # Warn if clock skew is significant (e.g., > 5 seconds)
        logger.warning(Fore.YELLOW + f"Significant time difference ({time_diff_ms} ms) between system and exchange. Check system clock synchronization.")

    # Load markets (force reload to ensure fresh data)
    logger.info("Loading market spirits (market data)...")
    # load_markets can sometimes fail transiently, wrap in retry logic manually if needed, or handle error
    try:
        EXCHANGE.load_markets(True) # Force reload
        logger.info(Fore.GREEN + Style.BRIGHT + f"Successfully connected to Bybit Nexus ({CONFIG.market_type.capitalize()} Markets).")
    except Exception as e:
         logger.critical(Fore.RED + Style.BRIGHT + f"Failed to load markets: {e}", exc_info=True)
         sys.exit(1)


    # Verify symbol exists and get market details
    if CONFIG.symbol not in EXCHANGE.markets:
         logger.error(Fore.RED + Style.BRIGHT + f"Symbol {CONFIG.symbol} not found in Bybit {CONFIG.market_type} market spirits.")
         # Suggest available symbols more effectively
         available_symbols = []
         try:
             # Extract settle currency robustly (handles SYMBOL/QUOTE:SETTLE format)
             settle_currency_candidates = CONFIG.symbol.split(':') # e.g., ['BTC/USDT', 'USDT']
             settle_currency = settle_currency_candidates[-1].strip() if len(settle_currency_candidates) > 1 else None
             logger.info(f"Attempting to find active symbols settling in {settle_currency or 'any currency'} for type {CONFIG.market_type}...")

             for s, m in EXCHANGE.markets.items():
                  # Check if market matches the configured type (linear/inverse/swap) and is active
                  is_correct_type = False
                  # Check 'linear', 'inverse', 'swap' boolean flags first, then 'type'
                  if CONFIG.market_type == 'linear' and (m.get('linear', False) or m.get('type') == 'linear'): is_correct_type = True
                  if CONFIG.market_type == 'inverse' and (m.get('inverse', False) or m.get('type') == 'inverse'): is_correct_type = True
                  if CONFIG.market_type == 'swap' and (m.get('swap', False) or m.get('type') == 'swap'): is_correct_type = True

                  # Filter by settle currency if known and check if active
                  if m.get('active') and is_correct_type:
                      # Only include if settle currency matches, or if no settle currency was parsed from input symbol
                      # Note: m.get('settle') might be None for inverse markets, check quote instead
                      if settle_currency is None: # If no settle currency was specified in config symbol
                           # Add any active market of the correct type
                           available_symbols.append(s)
                      elif m.get('settle') == settle_currency:
                          # Add if settle currency matches
                          available_symbols.append(s)
                      elif CONFIG.market_type == 'inverse' and m.get('quote') == settle_currency:
                           # For inverse, check if quote matches settle currency (e.g., BTC/USD settles in USD)
                           available_symbols.append(s)


         except Exception as parse_err:
             logger.error(f"Could not parse symbol or filter suggestions: {parse_err}")
             # Fallback: List all active symbols of the correct type if filtering fails
             available_symbols = [
                 s for s, m in EXCHANGE.markets.items()
                 if m.get('active') and ((CONFIG.market_type == 'linear' and (m.get('linear', False) or m.get('type') == 'linear')) or (CONFIG.market_type == 'inverse' and (m.get('inverse', False) or m.get('type') == 'inverse')) or (CONFIG.market_type == 'swap' and (m.get('swap', False) or m.get('type') == 'swap')))
             ]


         suggestion_limit = 50 # Increased suggestion limit
         if available_symbols:
             suggestions = ", ".join(sorted(available_symbols)[:suggestion_limit])
             if len(available_symbols) > suggestion_limit:
                 suggestions += ", ..."
             logger.info(Fore.CYAN + f"Available active {CONFIG.market_type} symbols (sample): " + suggestions)
         else:
             logger.info(Fore.CYAN + f"Could not find any active {CONFIG.market_type} symbols to suggest matching criteria.")
         sys.exit(1)
    else:
        MARKET_INFO = EXCHANGE.market(CONFIG.symbol)
        logger.info(Fore.CYAN + f"Market spirit for {CONFIG.symbol} acknowledged (ID: {MARKET_INFO.get('id')}).")

        # --- Log key precision and limits using Decimal ---
        # Extract values safely, providing defaults or logging errors
        try:
            # precision['price'] might be a tick size (Decimal/string) or number of decimal places (int)
            price_precision_raw = MARKET_INFO['precision'].get('price')
            # precision['amount'] might be a step size (Decimal/string) or number of decimal places (int)
            amount_precision_raw = MARKET_INFO['precision'].get('amount')
            min_amount_raw = MARKET_INFO['limits']['amount'].get('min')
            max_amount_raw = MARKET_INFO['limits']['amount'].get('max') # Max might be None
            contract_size_raw = MARKET_INFO.get('contractSize') # Can be None
            min_cost_raw = MARKET_INFO['limits'].get('cost', {}).get('min') # Min cost might not exist
            min_notional_raw = MARKET_INFO['limits'].get('notional', {}).get('min') # V5 Unified uses Notional

            # Convert to Decimal for logging and potential use, handle None/N/A
            # Use Decimal(str(value)) to preserve precision if value is float/int
            price_prec_dec = Decimal(str(price_precision_raw)) if price_precision_raw is not None else Decimal("NaN")
            amount_prec_dec = Decimal(str(amount_precision_raw)) if amount_precision_raw is not None else Decimal("NaN")
            min_amount_dec = Decimal(str(min_amount_raw)) if min_amount_raw is not None else Decimal("NaN")
            max_amount_dec = Decimal(str(max_amount_raw)) if max_amount_raw is not None else Decimal('Infinity') # Use Infinity for no max
            contract_size_dec = Decimal(str(contract_size_raw)) if contract_size_raw is not None else Decimal("1") # Default to '1' if not present/None
            min_cost_dec = Decimal(str(min_cost_raw)) if min_cost_raw is not None else Decimal("NaN")
            min_notional_dec = Decimal(str(min_notional_raw)) if min_notional_raw is not None else Decimal("NaN")


            logger.debug(f"Market Precision: Price Tick/Decimals={price_prec_dec.normalize() if not price_prec_dec.is_nan() else 'N/A'}, Amount Step/Decimals={amount_prec_dec.normalize() if not amount_prec_dec.is_nan() else 'N/A'}")
            logger.debug(f"Market Limits: Min Amount={min_amount_dec.normalize() if not min_amount_dec.is_nan() else 'N/A'}, Max Amount={max_amount_dec.normalize() if max_amount_dec != Decimal('Infinity') else 'Infinity'}")
            logger.debug(f"Market Limits: Min Cost={min_cost_dec.normalize() if not min_cost_dec.is_nan() else 'N/A'}, Min Notional={min_notional_dec.normalize() if not min_notional_dec.is_nan() else 'N/A'}")
            logger.debug(f"Contract Size: {contract_size_dec.normalize()}")

        except (KeyError, TypeError, InvalidOperation) as e:
             logger.critical(f"{Fore.RED+Style.BRIGHT}Failed to parse critical market info (precision/limits/size) from MARKET_INFO: {e}. Halting.", exc_info=True)
             logger.debug(f"Problematic MARKET_INFO: {MARKET_INFO}")
             sys.exit(1)

except ccxt.AuthenticationError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Authentication failed! Check API Key/Secret validity and permissions. Error: {e}")
    sys.exit(1)
except ccxt.NetworkError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Network error connecting to Bybit: {e}. Check internet connection and Bybit status.")
    sys.exit(1)
except ccxt.ExchangeNotAvailable as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Bybit exchange is currently unavailable: {e}. Check Bybit status.")
    sys.exit(1)
except ccxt.ExchangeError as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Exchange Nexus Error during initialization: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error during Nexus initialization: {e}", exc_info=True)
    sys.exit(1)

# --- Termux Utility Spell ---
def termux_notify(title: str, content: str) -> None:
    """Sends a notification using Termux API (if available) via termux-toast."""
    if not os.getenv("TERMUX_VERSION"):
        logger.debug("Not running in Termux environment. Skipping notification.")
        return

    try:
        # Check if command exists using which (more portable than 'command -v')
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False)
        if check_cmd.returncode != 0:
            logger.debug("termux-toast command not found. Skipping notification.")
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        # Keep it simple for toast notifications
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '')
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '')

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 200 # Keep toast messages concise
        full_message = f"{safe_title}: {safe_content}"[:max_len]

        # Use list format for subprocess.run for security (prevents shell injection)
        # Example styling: gravity middle, black text on green background, short duration
        # Ensure command and args are passed as separate list items
        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', 'green', '-s', full_message]
        # shell=False is default for list format, but good to be explicit
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5, shell=False) # Add timeout

        if result.returncode != 0:
            # Log stderr if available
            stderr_msg = result.stderr.strip()
            logger.warning(f"termux-toast command failed with code {result.returncode}" + (f": {stderr_msg}" if stderr_msg else ""))
        # No else needed, success is silent

    except FileNotFoundError:
         logger.debug("termux-toast command not found (FileNotFoundError). Skipping notification.")
    except subprocess.TimeoutExpired:
         logger.warning("termux-toast command timed out. Skipping notification.")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True)

# --- Precision Casting Spells ---

def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    """Formats price according to market precision rules using exchange's method."""
    # global MARKET_INFO, EXCHANGE # No assignment, no global needed
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            price_dec = Decimal(str(price))
            # Quantize to a sensible default precision (e.g., 8 decimals)
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            return str(price) # Last resort

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Ensure input is float as expected by CCXT methods.
        # Need to handle potential NaN Decimal input
        if isinstance(price, Decimal) and price.is_nan():
             logger.warning(f"Attempted to format NaN price for {symbol}. Returning 'NaN'.")
             return "NaN"
        # Ensure input is float for CCXT method
        price_float = float(price)
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError, InvalidOperation, ValueError, TypeError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid price format '{price}': {e}. Using fallback formatting.")
         try:
             price_dec = Decimal(str(price))
             return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
         except Exception:
              return str(price)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback.")
        try:
             price_dec = Decimal(str(price))
             return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_DOWN))
        except Exception:
            return str(price)

def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    """Formats amount according to market precision rules (step size) using exchange's method."""
    # global MARKET_INFO, EXCHANGE # No assignment, no global needed
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            amount_dec = Decimal(str(amount))
            # Quantize to a sensible default precision (e.g., 8 decimals)
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            return str(amount) # Last resort

    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes.
        # Bybit usually requires truncation (ROUND_DOWN) for quantity.
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND # Basic mapping
        # Ensure input is float as expected by CCXT methods.
        # Need to handle potential NaN Decimal input
        if isinstance(amount, Decimal) and amount.is_nan():
             logger.warning(f"Attempted to format NaN amount for {symbol}. Returning 'NaN'.")
             return "NaN"
        # Ensure input is float for CCXT method
        amount_float = float(amount)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError, InvalidOperation, ValueError, TypeError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid amount format '{amount}': {e}. Using fallback formatting.")
         try:
             amount_dec = Decimal(str(amount))
             return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
         except Exception:
              return str(amount)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback.")
        try:
             amount_dec = Decimal(str(amount))
             return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode))
        except Exception:
            return str(amount)

# --- Data Fetching and Processing ---

def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using the retry wrapper and perform validation."""
    # global EXCHANGE # No assignment, no global needed
    logger.info(Fore.CYAN + f"# Channeling market whispers for {symbol} ({timeframe})...")

    if EXCHANGE is None or not hasattr(EXCHANGE, 'fetch_ohlcv'):
         logger.error(Fore.RED + "Exchange object not properly initialized or missing fetch_ohlcv.")
         return None

    # Ensure limit is positive (already validated in config, but double check)
    if limit <= 0:
         logger.error(f"Invalid OHLCV limit requested: {limit}. Using default 100.")
         limit = 100

    ohlcv_data = None
    try:
        # fetch_with_retries handles category param automatically
        ohlcv_data = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
    except Exception as e:
        # fetch_with_retries should handle most errors, but catch any unexpected ones here
        logger.error(Fore.RED + f"Unhandled exception during fetch_ohlcv call via fetch_with_retries: {e}", exc_info=True)
        return None

    if ohlcv_data is None:
        # fetch_with_retries already logged the failure reason
        logger.error(Fore.RED + f"Failed to fetch OHLCV data for {symbol}.")
        return None
    # Check if ohlcv_data is a list and not empty
    if not isinstance(ohlcv_data, list) or not ohlcv_data:
        logger.error(Fore.RED + f"Received empty or invalid OHLCV data type: {type(ohlcv_data)}. Content: {str(ohlcv_data)[:100]}")
        return None

    # Check if each item in the list is a list of expected length
    expected_ohlcv_len = 6 # timestamp, open, high, low, close, volume
    if not all(isinstance(item, list) and len(item) >= expected_ohlcv_len for item in ohlcv_data):
        logger.error(Fore.RED + f"Received OHLCV data with unexpected item format. Expected list of lists with >= {expected_ohlcv_len} items.")
        # Log first few problematic items for debugging
        for i, item in enumerate(ohlcv_data[:5]):
             if not (isinstance(item, list) and len(item) >= expected_ohlcv_len):
                  logger.debug(f"Problematic OHLCV item {i}: {item}")
        return None

    try:
        # Convert list of lists to DataFrame, ensuring columns have appropriate names.
        column_names = ["timestamp", "open", "high", "low", "close"]
        if len(ohlcv_data[0]) >= 6:
             column_names.append("volume")
        df = pd.DataFrame(ohlcv_data, columns=column_names)

        # Convert timestamp immediately to UTC datetime objects
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        initial_len = len(df)
        df.dropna(subset=["timestamp"], inplace=True) # Drop rows where timestamp conversion failed
        if len(df) < initial_len:
             dropped_count = initial_len - len(df)
             logger.warning(f"Dropped {dropped_count} rows due to invalid timestamp.")
             initial_len = len(df) # Update initial_len for next check

        # Convert numeric columns to float first for pandas/numpy compatibility
        # Use Decimal for final calculations, but pandas/numpy work better with floats for intermediate steps
        numeric_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns: numeric_cols.append("volume")
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaNs in critical price columns *after* conversion
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len:
            dropped_count = initial_len - len(df)
            logger.warning(f"Dropped {dropped_count} rows with missing essential price data from OHLCV.")

        if df.empty:
            logger.error(Fore.RED + "DataFrame is empty after processing and cleaning OHLCV data (all rows dropped?).")
            return None

        df = df.set_index("timestamp")
        # Ensure data is sorted chronologically (fetch_ohlcv usually guarantees this, but verify)
        if not df.index.is_monotonic_increasing:
             logger.warning("OHLCV data was not sorted chronologically. Sorting now.")
             df.sort_index(inplace=True)

        # Check for duplicate timestamps (can indicate data issues)
        if df.index.duplicated().any():
             duplicates = df.index[df.index.duplicated()].unique()
             logger.warning(Fore.YELLOW + f"Duplicate timestamps found in OHLCV data ({len(duplicates)} unique duplicates). Keeping last entry for each.")
             df = df[~df.index.duplicated(keep='last')]

        # Check time difference between last two candles vs expected interval
        if len(df) > 1:
             time_diff = df.index[-1] - df.index[-2]
             try:
                 # Use pandas to parse timeframe string robustly
                 interval_seconds = EXCHANGE.parse_timeframe(timeframe)
                 expected_interval_td = pd.Timedelta(interval_seconds, unit='s')
                 # Allow some tolerance (e.g., 20% of interval + 10s buffer) for minor timing differences/API lag
                 tolerance_seconds = interval_seconds * 0.2 + 10
                 tolerance = pd.Timedelta(seconds=tolerance_seconds)
                 if abs(time_diff.total_seconds()) > expected_interval_td.total_seconds() + tolerance.total_seconds():
                      logger.warning(f"Unexpected large time gap between last two candles: {time_diff} (expected ~{expected_interval_td}, allowed lag ~{tolerance}).")
             except ValueError:
                 logger.warning(f"Could not parse timeframe '{timeframe}' to calculate expected interval for time gap check.")
             except Exception as time_check_e:
                 logger.warning(f"Error during time difference check: {time_check_e}")

        # Check if enough candles are returned compared to the requested limit
        if len(df) < limit:
             logger.warning(f"{Fore.YELLOW}Fetched fewer candles ({len(df)}) than requested limit ({limit}). Data might be incomplete or market history short.")

        logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return df
    except Exception as e:
        logger.error(Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None

def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Decimal]]:
    """Calculate technical indicators using CONFIG periods, returning results as Decimals for precision."""
    # global CONFIG # Accessing CONFIG is fine without global here.
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        # Defensive: Make a copy to avoid modifying the original DataFrame if it's used elsewhere
        df_calc = df.copy()
        # Convert relevant columns to float, coercing errors
        # Use .values to get numpy arrays for potentially better performance with pandas operations
        close = pd.to_numeric(df_calc["close"], errors='coerce').values
        high = pd.to_numeric(df_calc["high"], errors='coerce').values
        low = pd.to_numeric(df_calc["low"], errors='coerce').values

        # Drop any rows where these critical columns are NaN *after* conversion
        # Need to do this on the DataFrame before extracting arrays if dropping is desired
        df_calc.dropna(subset=["close", "high", "low"], inplace=True)
        if df_calc.empty:
             logger.error(Fore.RED + "DataFrame is empty after dropping rows with NaN price data. Cannot calculate indicators.")
             return None

        # Re-extract arrays from the cleaned DataFrame
        close = df_calc["close"].astype(float).values
        high = df_calc["high"].astype(float).values
        low = df_calc["low"].astype(float).values
        index = df_calc.index # Keep track of the index for re-aligning series


        # --- Check Data Length Requirements ---
        # Ensure enough data for EWMA initial states to stabilize somewhat
        required_len_ema_stable = max(CONFIG.fast_ema_period, CONFIG.slow_ema_period, CONFIG.trend_ema_period)
        # Stoch requires period + smoothing - 2 for the first valid %D (approx). ATR requires period + 1.
        required_len_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2
        required_len_atr = CONFIG.atr_period + 1

        min_required_len = max(required_len_ema_stable, required_len_stoch, required_len_atr)
        # Add a buffer to ensure the latest indicator values are not the very first calculated ones
        # Use smoothing period or a fixed buffer (e.g., 5) for safety margins on .iloc[-1]/-2
        min_safe_len = min_required_len + max(CONFIG.stoch_smooth_d, CONFIG.atr_period, 5)


        if len(df_calc) < min_safe_len:
             logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for stable indicators (minimum safe: {min_safe_len}). Indicator values might be less reliable. Increase OHLCV_LIMIT or wait for more data.")
             # Proceed anyway, but warn

        if len(df_calc) < min_required_len:
             logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)}) for core indicator calculations (minimum required: {min_required_len}). Cannot calculate indicators.")
             return None # Critical failure if even minimum isn't met


        # --- Calculations using Pandas (via numpy arrays) and CONFIG periods ---
        # Create Pandas Series from numpy arrays for easier calculations and index alignment
        close_series = pd.Series(close, index=index)
        high_series = pd.Series(high, index=index)
        low_series = pd.Series(low, index=index)

        fast_ema_series = close_series.ewm(span=CONFIG.fast_ema_period, adjust=False).mean()
        slow_ema_series = close_series.ewm(span=CONFIG.slow_ema_period, adjust=False).mean()
        trend_ema_series = close_series.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()

        # Stochastic Oscillator %K and %D
        # Calculate lowest low and highest high over the lookback period
        low_min_series = low_series.rolling(window=CONFIG.stoch_period).min()
        high_max_series = high_series.rolling(window=CONFIG.stoch_period).max()

        # Calculate raw %K
        # Add epsilon to prevent division by zero if high_max == low_min over the period
        # Use numpy's epsilon for float calculations
        # Handle cases where high_max_series - low_min_series is zero or NaN by returning NaN for k
        stoch_k_raw_series = pd.Series(np.nan, index=index) # Initialize with NaN
        valid_range_mask = (high_max_series - low_min_series) > np.finfo(float).eps # Mask where range is valid and > epsilon
        stoch_k_raw_series[valid_range_mask] = 100 * (close_series[valid_range_mask] - low_min_series[valid_range_mask]) / (high_max_series[valid_range_mask] - low_min_series[valid_range_mask])


        # Smooth %K to get final %K and then smooth %K to get %D
        stoch_k_series = stoch_k_raw_series.rolling(window=CONFIG.stoch_smooth_k).mean()
        stoch_d_series = stoch_k_series.rolling(window=CONFIG.stoch_smooth_d).mean()

        # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
        # Calculate True Range first
        prev_close_series = close_series.shift(1)
        tr_series = pd.concat([
            high_series - low_series,
            (high_series - prev_close_series).abs(),
            (low_series - prev_close_series).abs()
        ], axis=1).max(axis=1) # Max of (H-L), |H-C(prev)|, |L-C(prev)|

        # Use ewm with alpha = 1/period for Wilder's smoothing, skip initial NaN TR value (first TR is NaN due to shift)
        # .ewm(alpha=...) is equivalent to Wilder's if adjust=False
        # ATR is calculated on the TR series, skipping the first NaN value
        atr_series = tr_series.iloc[1:].ewm(alpha=1/CONFIG.atr_period, adjust=False).mean()
        # Prepend NaN back to align index with original DataFrame/series length
        atr_series = pd.concat([pd.Series([np.nan], index=[tr_series.index[0]]), atr_series])


        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places
        # Use .normalize() for display to remove excess zeros, but keep full precision internally
        price_quantizer = Decimal("1E-8") # Example: 8 decimal places
        percent_quantizer = Decimal("1E-2") # Example: 2 decimal places for percentages
        atr_quantizer = Decimal("1E-8") # Example: 8 decimal places for ATR (same as price)


        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(series: pd.Series, quantizer: Decimal, name: str, default_val: Decimal = Decimal("NaN")) -> Decimal:
            if series.empty or series.isna().all():
                logger.warning(f"Indicator series '{name}' is empty or all NaN.")
                return default_val
            # Get the last valid (non-NaN) value
            latest_valid_val = series.dropna().iloc[-1] if not series.dropna().empty else None

            if latest_valid_val is None:
                 # This case should be covered by the initial min_required_len check, but defensive
                 logger.warning(f"Indicator calculation for '{name}' resulted in NaN or only NaNs.")
                 return default_val
            try:
                # Convert via string for precision, then quantize
                # Use str() conversion to avoid potential float precision issues when creating Decimal
                return Decimal(str(latest_valid_val)).quantize(quantizer)
            except (InvalidOperation, TypeError) as e:
                logger.error(f"Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning default.", exc_info=True)
                return default_val

        # Get latest values
        latest_fast_ema = get_latest_decimal(fast_ema_series, price_quantizer, "fast_ema")
        latest_slow_ema = get_latest_decimal(slow_ema_series, price_quantizer, "slow_ema")
        latest_trend_ema = get_latest_decimal(trend_ema_series, price_quantizer, "trend_ema")
        latest_stoch_k = get_latest_decimal(stoch_k_series, percent_quantizer, "stoch_k", default_val=Decimal("NaN")) # Default NaN, don't default to 50
        latest_stoch_d = get_latest_decimal(stoch_d_series, percent_quantizer, "stoch_d", default_val=Decimal("NaN")) # Default NaN
        latest_atr = get_latest_decimal(atr_series, atr_quantizer, "atr", default_val=Decimal("NaN")) # Default NaN if calc failed

        # --- Calculate Stochastic Cross Signals (Boolean) ---
        # Requires at least 2 data points for the shift (previous vs current)
        stoch_kd_bullish = False
        stoch_kd_bearish = False
        # Ensure series have enough data and last two values are not NaN
        # Use .dropna() to get series with only valid values
        k_series_valid = stoch_k_series.dropna()
        d_series_valid = stoch_d_series.dropna()

        if len(k_series_valid) >= 2 and len(d_series_valid) >= 2:
             try:
                  # Get the last two valid (non-nan) values
                  stoch_k_last = Decimal(str(k_series_valid.iloc[-1]))
                  stoch_d_last = Decimal(str(d_series_valid.iloc[-1]))
                  stoch_k_prev = Decimal(str(k_series_valid.iloc[-2]))
                  stoch_d_prev = Decimal(str(d_series_valid.iloc[-2]))

                  # Check for crossover using previous vs current values (Decimal comparison)
                  # Bullish cross: K crosses above D
                  stoch_kd_bullish = (stoch_k_last > stoch_d_last) and (stoch_k_prev <= stoch_d_prev)
                  # Bearish cross: K crosses below D
                  stoch_kd_bearish = (stoch_k_last < stoch_d_last) and (stoch_k_prev >= stoch_d_prev)

                  # Optional: Ensure crossover happens within or from the relevant zones for signalling
                  # Only signal bullish cross if it happens *in* or *from* the oversold zone (K or D was <= threshold)
                  # Check if *either* K or D was below/at the threshold in the previous candle
                  if stoch_kd_bullish and stoch_k_prev > CONFIG.stoch_oversold_threshold and stoch_d_prev > CONFIG.stoch_oversold_threshold:
                      logger.debug(f"Stoch K/D Bullish cross ({stoch_k_prev:.2f}/{stoch_d_prev:.2f} -> {stoch_k_last:.2f}/{stoch_d_last:.2f}) happened strictly above oversold zone ({CONFIG.stoch_oversold_threshold.normalize()}). Not using for signal.")
                      stoch_kd_bullish = False
                  # Only signal bearish cross if it happens *in* or *from* the overbought zone (K or D was >= threshold)
                  # Check if *either* K or D was above/at the threshold in the previous candle
                  if stoch_kd_bearish and stoch_k_prev < CONFIG.stoch_overbought_threshold and stoch_d_prev < CONFIG.stoch_overbought_threshold:
                       logger.debug(f"Stoch K/D Bearish cross ({stoch_k_prev:.2f}/{stoch_d_prev:.2f} -> {stoch_k_last:.2f}/{stoch_d_last:.2f}) happened strictly below overbought zone ({CONFIG.stoch_overbought_threshold.normalize()}). Not using for signal.")
                       stoch_kd_bearish = False

             except (InvalidOperation, TypeError, IndexError) as e: # Add IndexError for iloc[-2]
                  logger.warning(f"Error calculating Stoch K/D cross: {e}. Cross signals will be False.", exc_info=True)
                  stoch_kd_bullish = False
                  stoch_kd_bearish = False
        else:
             logger.debug(f"Not enough valid data points ({len(k_series_valid)} K, {len(d_series_valid)} D) for Stoch K/D cross calculation.")


        indicators_out = {
            "fast_ema": latest_fast_ema,
            "slow_ema": latest_slow_ema,
            "trend_ema": latest_trend_ema,
            "stoch_k": latest_stoch_k,
            "stoch_d": latest_stoch_d,
            "atr": latest_atr,
            "atr_period": CONFIG.atr_period, # Store period for display
            "stoch_kd_bullish": stoch_kd_bullish, # Add cross signals
            "stoch_kd_bearish": stoch_kd_bearish # Add cross signals
        }

        # Check if any crucial indicator calculation failed (returned NaN default)
        critical_indicators = ['fast_ema', 'slow_ema', 'trend_ema', 'stoch_k', 'stoch_d', 'atr']
        # Filter out keys from critical_indicators that are not in indicators_out (defensive)
        actual_critical_indicators = [key for key in critical_indicators if key in indicators_out]

        failed_indicators = [key for key in actual_critical_indicators if isinstance(indicators_out.get(key), Decimal) and indicators_out[key].is_nan()]

        if failed_indicators:
             logger.error(f"{Fore.RED}One or more critical indicators failed to calculate (NaN): {', '.join(failed_indicators)}")
             # Return partial results if possible, or None if critical for signal generation
             # For this strategy, most are critical. Let's return None if *any* critical indicator is NaN.
             # If we wanted to allow trading based on *available* indicators, we would return indicators_out here.
             # For robustness, let's fail the whole indicator calculation if core indicators are NaN.
             return None # Signal failure

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        return indicators_out

    except Exception as e:
        logger.error(Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True)
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Fetch current positions using retry wrapper, returning quantities and prices as Decimals."""
    # global EXCHANGE, CONFIG # No assignment, no global needed
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")

    if EXCHANGE is None:
         logger.error("Exchange object not available for fetching positions.")
         return None

    # Initialize with Decimal zero/NaN for clarity
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "pnl": Decimal("NaN")}
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        # fetch_positions for V5 Unified requires the symbol and category
        # Pass the market ID, not the symbol string, as per CCXT unified format
        market_id = MARKET_INFO.get('id') if MARKET_INFO else symbol # Fallback to symbol string if market_info not loaded
        if market_id is None:
             logger.error(Fore.RED + f"Cannot fetch positions, market ID for {symbol} is not available.")
             return None
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[market_id], params={'category': CONFIG.market_type})

    except Exception as e:
        # Handle potential exceptions raised by fetch_with_retries itself (e.g., AuthenticationError, Non-retryable ExchangeError)
        logger.error(Fore.RED + f"Unhandled exception during fetch_positions call via fetch_with_retries: {e}", exc_info=True)
        return None # Indicate failure

    if positions_data is None:
         # fetch_with_retries already logged the failure reason
         logger.error(Fore.RED + f"Failed to fetch positions for {symbol}.")
         return None # Indicate failure

    if not isinstance(positions_data, list):
         logger.error(f"Unexpected data type received from fetch_positions: {type(positions_data)}. Expected list. Data: {str(positions_data)[:200]}")
         return None

    # Process the fetched positions - find the primary long/short position for the symbol
    # Handle potential duplicate entries for the same side (e.g. from hedge mode or API quirks)
    # Prioritize non-zero quantity if multiple entries exist for the same side.
    found_non_zero_long = False
    found_non_zero_short = False

    for pos in positions_data:
        # Ensure pos is a dictionary and has 'symbol' and 'info' keys
        if not isinstance(pos, dict) or 'symbol' not in pos or 'info' not in pos or not isinstance(pos['info'], dict):
            logger.warning(f"Skipping invalid position data format: {pos}")
            continue

        pos_symbol = pos['symbol']
        # Compare against market ID, not the CCXT symbol string
        if MARKET_INFO and pos_symbol != MARKET_INFO.get('id'):
            logger.debug(f"Ignoring position data for different symbol ID: {pos_symbol}")
            continue
        # If market_info wasn't loaded, fallback to symbol string comparison
        elif not MARKET_INFO and pos_symbol != symbol:
             logger.debug(f"Ignoring position data for different symbol: {pos_symbol}")
             continue


        pos_info = pos['info'] # Use the extracted info dict

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        side = pos.get("side") # Unified field ('long'/'short')
        if side not in ["long", "short"]:
            # Fallback for Bybit V5 'info' field if unified 'side' is missing/invalid
            # Bybit V5 info['side'] uses 'Buy'/'Sell'
            side_raw = pos_info.get("side", "").lower() # e.g., "Buy" or "Sell"
            if side_raw == "buy": side = "long"
            elif side_raw == "sell": side = "short"
            else:
                 logger.warning(f"Could not determine side for position: Info={str(pos_info)[:100]}. Skipping.")
                 continue

        # Get quantity ('contracts' or 'size') - Use unified field first, fallback to info
        contracts_raw = pos.get("contracts") # Unified field ('contracts' seems standard)
        if contracts_raw is None:
            contracts_raw = pos_info.get("size") # Common Bybit V5 field in 'info'

        # Get entry price - Use unified field first, fallback to info
        entry_price_raw = pos.get("entryPrice") # Unified field
        if entry_price_raw is None:
            # Check 'avgPrice' (common in V5) or 'entryPrice' in info
            entry_price_raw = pos_info.get("avgPrice", pos_info.get("entryPrice"))

        # Get Liq Price and PnL (these are less standardized, rely more on unified fields if available)
        liq_price_raw = pos.get("liquidationPrice") # Unified field
        if liq_price_raw is None:
            liq_price_raw = pos_info.get("liqPrice") # Bybit V5 info field

        pnl_raw = pos.get("unrealizedPnl") # Unified field
        if pnl_raw is None:
             # Check Bybit specific info fields (note potential spelling 'unrealisedPnl')
             pnl_raw = pos_info.get("unrealisedPnl", pos_info.get("unrealizedPnl"))


        # --- Convert to Decimal and Store ---
        if side in pos_dict and contracts_raw is not None:
            try:
                # Convert via string for precision
                contracts = Decimal(str(contracts_raw))

                # Use epsilon check for effectively zero positions
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(f"Ignoring effectively zero size {side} position for {symbol} (Qty: {contracts.normalize()}).")
                    continue # Skip processing this entry

                # If we already processed a non-zero quantity for this side, skip this entry
                # This handles cases where API might return multiple entries for the same position side (e.g., different TP/SL attached)
                if (side == "long" and found_non_zero_long) or \
                   (side == "short" and found_non_zero_short):
                     logger.debug(f"Already processed a non-zero {side} position for {symbol}. Skipping subsequent entries for this side.")
                     continue # Skip processing this entry

                # Convert other fields, handling potential None or invalid values
                # Use try-except around individual conversions for better resilience
                try: entry_price = Decimal(str(entry_price_raw)) if entry_price_raw is not None and str(entry_price_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: entry_price = Decimal("NaN"); logger.warning(f"Could not parse {side} entry price: '{entry_price_raw}'")

                try: liq_price = Decimal(str(liq_price_raw)) if liq_price_raw is not None and str(liq_price_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: liq_price = Decimal("NaN"); logger.warning(f"Could not parse {side} liq price: '{liq_price_raw}'")

                try: pnl = Decimal(str(pnl_raw)) if pnl_raw is not None and str(pnl_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: pnl = Decimal("NaN"); logger.warning(f"Could not parse {side} pnl: '{pnl_raw}'")


                # Assign to the dictionary
                pos_dict[side]["qty"] = contracts
                pos_dict[side]["entry_price"] = entry_price
                pos_dict[side]["liq_price"] = liq_price
                pos_dict[side]["pnl"] = pnl

                # Mark side as found (only the first non-zero entry per side)
                if side == "long": found_non_zero_long = True
                else: found_non_zero_short = True

                # Log with formatted decimals (for display)
                entry_log = f"{entry_price.normalize()}" if not entry_price.is_nan() else "N/A"
                liq_log = f"{liq_price.normalize()}" if not liq_price.is_nan() else "N/A"
                pnl_log = f"{pnl:+.4f}" if not pnl.is_nan() else "N/A"
                logger.info(Fore.YELLOW + f"Found active {side.upper()} position: Qty={contracts.normalize()}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log}")


            except (InvalidOperation, TypeError) as e:
                 logger.error(f"Could not parse position data for {side} side: Qty='{contracts_raw}'. Error: {e}", exc_info=True)
                 # This specific position entry is problematic, but we don't exit.
                 # The pos_dict[side] will retain its default NaN/0 values unless overwritten by a valid entry.
                 continue # Continue to the next position entry

        elif side not in pos_dict:
            logger.warning(f"Position data found for unknown side '{side}'. Skipping.")

    if not found_non_zero_long and not found_non_zero_short:
         logger.info(Fore.BLUE + f"No active non-zero positions reported by exchange for {symbol}.")
    elif found_non_zero_long and found_non_zero_short:
         # This indicates hedge mode or an issue. Bot assumes one-way.
         logger.warning(Fore.YELLOW + f"Both LONG and SHORT positions found for {symbol}. Pyrmethus assumes one-way mode and will manage the first non-zero position found for each side. Please ensure your exchange account is configured for one-way trading.")
         # The pos_dict will contain the first non-zero quantity found for each side.

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return pos_dict

def get_balance(currency: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    """Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals."""
    # global EXCHANGE # No assignment, no global needed
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching balance.")
        return None, None

    balance_data = None
    try:
        # Bybit V5 fetch_balance might need accountType (UNIFIED/CONTRACT) or coin.
        # CCXT's defaultType/SubType and category *should* handle this, but params might be needed.
        # Let's rely on fetch_with_retries to add category if configured.
        # fetch_with_retries returns None on failure
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance)
    except Exception as e:
        logger.error(Fore.RED + f"Unhandled exception during fetch_balance call via fetch_with_retries: {e}", exc_info=True)
        return None, None

    if balance_data is None:
        # fetch_with_retries already logged the failure
        logger.error(Fore.RED + f"Failed to fetch balance after retries. Cannot assess risk capital.")
        return None, None

    # --- Parse Balance Data ---
    # Initialize with NaN Decimals to indicate failure to find/parse
    free_balance = Decimal("NaN")
    total_balance = Decimal("NaN") # Represents Equity for futures/swaps

    try:
        # Check 'info' first for exchange-specific structure (most reliable for Bybit V5)
        if 'info' in balance_data and isinstance(balance_data['info'], dict):
            info_data = balance_data['info']
            # V5 structure: result -> list -> account objects
            if 'result' in info_data and isinstance(info_data['result'], dict) and \
               'list' in info_data['result'] and isinstance(info_data['result']['list'], list):
                for account in info_data['result']['list']:
                    # Find the account object for the target currency or the UNIFIED account
                    account_type = account.get('accountType') # UNIFIED or CONTRACT
                    if account_type == 'UNIFIED':
                         # V5 Unified structure often has equity and available balance at the account level
                         # These are the key metrics for risk calculation
                         equity_raw = account.get('totalEquity')
                         free_raw = account.get('totalAvailableBalance') # Balance available for trading/withdrawal

                         # Attempt to convert and assign if found
                         if equity_raw is not None:
                             try: total_balance = Decimal(str(equity_raw))
                             except InvalidOperation: logger.warning(f"Could not parse UNIFIED totalEquity: '{equity_raw}'")
                         if free_raw is not None:
                              try: free_balance = Decimal(str(free_raw))
                              except InvalidOperation: logger.warning(f"Could not parse UNIFIED totalAvailableBalance: '{free_raw}'")

                         # If we found the key metrics for UNIFIED account, we can stop looking
                         if not total_balance.is_nan() and not free_balance.is_nan():
                              logger.debug(f"Parsed Bybit V5 UNIFIED info structure: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Equity={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")
                              break # Found relevant account info

                    elif account_type == 'CONTRACT' and account.get('coin') == currency:
                         # For CONTRACT accounts, find the specific currency
                         equity_raw = account.get('equity')
                         free_raw = account.get('availableToWithdraw', account.get('availableBalance')) # Check alternative keys

                         # Attempt to convert and assign if found
                         if equity_raw is not None:
                             try: total_balance = Decimal(str(equity_raw))
                             except InvalidOperation: logger.warning(f"Could not parse CONTRACT equity for {currency}: '{equity_raw}'")
                         if free_raw is not None:
                              try: free_balance = Decimal(str(free_raw))
                              except InvalidOperation: logger.warning(f"Could not parse CONTRACT available for {currency}: '{free_raw}'")

                         if not total_balance.is_nan() and not free_balance.is_nan():
                              logger.debug(f"Parsed Bybit V5 CONTRACT info structure for {currency}: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Equity={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")
                              break # Found the currency account

            else:
                 logger.warning("Bybit V5 info structure not as expected ('result'/'list' missing).")

        # Fallback to CCXT unified structure if info parsing failed or not V5 structure
        # Only attempt fallback if total_balance is still NaN after V5 parsing attempt
        if total_balance.is_nan() and currency in balance_data and isinstance(balance_data[currency], dict):
            currency_balance = balance_data[currency]
            free_raw = currency_balance.get('free')
            total_raw = currency_balance.get('total') # 'total' usually represents equity in futures

            if free_raw is not None:
                 try: free_balance = Decimal(str(free_raw))
                 except InvalidOperation: logger.warning(f"Could not parse unified free balance for {currency}: '{free_raw}'")
            if total_raw is not None:
                 try: total_balance = Decimal(str(total_raw))
                 except InvalidOperation: logger.warning(f"Could not parse unified total balance for {currency}: '{total_raw}'")

            logger.debug(f"Parsed CCXT unified structure for {currency}: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Total={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")
        elif total_balance.is_nan(): # If still NaN after both attempts
            logger.warning(f"Balance data for {currency} not found in unified structure either.")


        # If parsing failed completely, balances will remain NaN
        if free_balance.is_nan():
             logger.warning(f"Could not find or parse free balance for {currency} in balance data.")
        if total_balance.is_nan():
             logger.warning(f"Could not find or parse total/equity balance for {currency} in balance data.")
             # Critical if equity is needed for risk calc
             logger.error(Fore.RED + "Failed to determine account equity. Cannot proceed safely.")
             return free_balance, None # Indicate equity failure specifically

        # Use 'total' balance (Equity) as the primary value for risk calculation
        equity = total_balance

        # Log final parsed values
        logger.info(Fore.GREEN + f"Vault contains {free_balance.normalize() if not free_balance.is_nan() else 'N/A'} free {currency} (Equity/Total: {equity.normalize() if not equity.is_nan() else 'N/A'}).")
        return free_balance, equity # Return free and total (equity)

    except (InvalidOperation, TypeError, KeyError) as e:
         logger.error(Fore.RED + f"Error parsing balance data for {currency}: {e}. Raw balance data sample: {str(balance_data)[:500]}", exc_info=True)
         return None, None # Indicate parsing failure
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None

# --- Order and State Management ---

def check_order_status(order_id: str, symbol: str, timeout: int) -> Optional[Dict]:
    """Checks order status with retries and timeout. Returns the final order dict or None."""
    # global EXCHANGE, CONFIG # No assignment, no global needed
    logger.info(Fore.CYAN + f"Verifying final status of order {order_id} for {symbol} (Timeout: {timeout}s)...")
    if EXCHANGE is None or MARKET_INFO is None: # Also need market info for ID
        logger.error("Exchange object or Market Info not loaded for checking order status.")
        return None

    start_time = time.time()
    last_status = 'unknown'
    attempt = 0
    check_interval = 1.5 # seconds between checks

    while time.time() - start_time < timeout:
        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        order_status_data = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Bybit V5 fetch_order requires category AND orderId/clientOrderId.
            # Ensure category is in params (handled by wrapper)
            # Use market ID for symbol parameter
            fetch_order_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
            # fetch_order expects the CCXT unified symbol string, not market ID, for the first argument
            order_status_data = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_order_params)

            if order_status_data and isinstance(order_status_data, dict):
                last_status = order_status_data.get('status', 'unknown')
                # CCXT represents filled quantity as Decimal or float. Convert to Decimal.
                filled_qty_raw = order_status_data.get('filled', 0.0)
                filled_qty = Decimal(str(filled_qty_raw)) if filled_qty_raw is not None else Decimal('0.0') # Convert to Decimal for accurate comparison

                # Bybit V5 statuses: New, PartiallyFilled, Filled, Cancelled, Rejected, PartiallyFilledCanceled, Deactivated, Triggered, Untriggered
                # CCXT status: open, closed, canceled, rejected, expired
                # CCXT 'closed' maps to Bybit 'Filled'. 'canceled' maps to 'Cancelled'. 'rejected' to 'Rejected'.
                # Market orders should ideally be 'Filled' ('closed' in CCXT) or 'Rejected'.
                bybit_v5_status = order_status_data.get('info', {}).get('orderStatus', 'N/A')
                logger.debug(f"Order {order_id} status check: CCXT '{last_status}' (Bybit V5 '{bybit_v5_status}'), Filled: {filled_qty.normalize()}")

                # Check for terminal states (fully filled, canceled, rejected, expired)
                # 'closed' usually means fully filled for market/limit orders on Bybit.
                # For market orders, 'closed' with non-zero fill is success.
                if last_status == 'closed' and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                     logger.info(Fore.GREEN + f"Order {order_id} confirmed FILLED (status 'closed').")
                     return order_status_data # Return the final order dict (success)
                elif last_status in ['canceled', 'rejected', 'expired']: # CCXT terminal failure states
                     logger.warning(f"Order {order_id} reached terminal failure state: {last_status}. Filled: {filled_qty.normalize()}.")
                     return order_status_data # Return the final order dict (failure to fill)
                # If 'open' but fully filled (can happen briefly), treat as terminal 'closed'
                # Check remaining amount using epsilon
                # CCXT represents remaining quantity as Decimal or float. Convert to Decimal.
                remaining_qty_raw = order_status_data.get('remaining', 0.0)
                remaining_qty = Decimal(str(remaining_qty_raw)) if remaining_qty_raw is not None else Decimal('0.0')
                if last_status == 'open' and remaining_qty.copy_abs() < CONFIG.position_qty_epsilon and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                    logger.info(f"Order {order_id} is 'open' but fully filled ({filled_qty.normalize()}). Treating as 'closed'.")
                    order_status_data['status'] = 'closed' # Update status locally for clarity
                    return order_status_data

            else:
                # fetch_with_retries failed or returned unexpected data
                # Error logged within fetch_with_retries, just note it here
                logger.warning(f"fetch_order call failed or returned invalid data for {order_id}. Continuing check loop.")
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. This is a terminal state indicating it never existed or was fully purged.
            # For market orders, this likely means it was rejected or expired immediately and purged.
            logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange. Likely rejected/expired.")
            # Synthesize a 'rejected' status for caller handling
            return {'status': 'rejected', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Order not found (likely rejected/expired)', 'retCode': 40001}} # Use a common Bybit rejection code if known
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            # Critical non-retryable errors - re-raise immediately
            logger.critical(Fore.RED + Style.BRIGHT + f"Authentication/Permission error during order status check for {order_id}: {e}. Halting.")
            sys.exit(1)
        except ccxt.ExchangeError as e:
             # Specific ExchangeErrors raised by fetch_with_retries.
             # If it's a non-retryable one re-raised by fetch_with_retries, it's caught here.
             logger.error(Fore.RED + f"Exchange error during order status check for {order_id}: {e}. Stopping checks.")
             # Re-raise to caller for explicit handling if needed, otherwise it returns None below.
             raise e
        except Exception as e:
            # Catch any other unexpected error during the check itself
            logger.error(f"Unexpected error during order status check loop for {order_id}: {e}", exc_info=True)
            # Decide whether to retry or fail; retrying is part of the loop.

        # Wait before the next check_order_status attempt
        time_elapsed = time.time() - start_time
        if time.time() - start_time < timeout: # Check timeout again before sleeping
             sleep_duration = min(check_interval, timeout - (time.time() - start_time)) # Don't sleep past timeout
             if sleep_duration > 0:
                 logger.debug(f"Order {order_id} status ({last_status}) not terminal/filled. Sleeping {sleep_duration:.1f}s...")
                 time.sleep(sleep_duration)
                 check_interval = min(check_interval * 1.2, 5) # Slightly increase interval up to 5s
             else:
                 break # Time is up
        else:
            break # Exit loop if timeout reached

    # --- Timeout Reached ---
    logger.error(Fore.RED + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: {last_status}.")
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"Performing final status check for order {order_id} after timeout...")
        # Use market ID for symbol parameter
        final_fetch_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
        # fetch_order expects the CCXT unified symbol string, not market ID, for the first argument
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=final_fetch_params)

        if final_check_status and isinstance(final_check_status, dict):
             final_status = final_check_status.get('status', 'unknown')
             # CCXT represents filled quantity as Decimal or float. Convert to Decimal.
             final_filled_raw = final_check_status.get('filled', 0.0)
             final_filled = Decimal(str(final_filled_raw)) if final_filled_raw is not None else Decimal('0.0')
             bybit_v5_status = final_check_status.get('info', {}).get('orderStatus', 'N/A')
             logger.info(f"Final status after timeout: CCXT '{final_status}' (Bybit V5 '{bybit_v5_status}'), Filled: {final_filled.normalize()}")
             # Return this final status even if timed out earlier
             return final_check_status
        else:
             logger.error(f"Final status check for order {order_id} also failed or returned invalid data.")
             # Synthesize a failure status if check failed
             return {'status': 'check_failed', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Final status check failed after timeout', 'retCode': -1}} # Use a custom code
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND on final check.")
        # Synthesize a 'rejected' status
        return {'status': 'rejected', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Order not found on final check', 'retCode': 40001}}
    except Exception as e:
        logger.error(f"Error during final status check for order {order_id}: {e}", exc_info=True)
        # Synthesize a failure status
        return {'status': 'check_failed', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': f'Exception during final status check: {e}', 'retCode': -2}}

    # If reached here, timed out and final check also failed or returned invalid data.
    logger.error(Fore.RED + f"Order {order_id} check ultimately failed after timeout and final check.")
    # Fallback return for complete failure
    return None


def log_trade_entry_to_journal(
    symbol: str, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]
) -> None:
    """Appends a trade entry record to the CSV journal file."""
    # global CONFIG # Accessing CONFIG is fine without global here.
    if not CONFIG.enable_journaling:
        return

    filepath = CONFIG.journal_file_path
    now = datetime.utcnow()
    # Basic strategy identifier based on configuration
    strategy_id = (
        f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})"
        f"_STOCH({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d})"
        f"_ATR({CONFIG.atr_period})"
        f"_TREND({CONFIG.trend_ema_period},{'ON' if CONFIG.trade_only_with_trend else 'OFF'},{CONFIG.trend_filter_buffer_percent.normalize()}%)"
        f"_ATRFILTER({CONFIG.atr_move_filter_multiplier.normalize()}x)"
        f"_RISK({(CONFIG.risk_percentage*100).normalize()}%)"
        f"_SL({CONFIG.sl_atr_multiplier.normalize()}xATR)"
        f"_TSL({CONFIG.tsl_activation_atr_multiplier.normalize()}xATR,{CONFIG.trailing_stop_percent.normalize()}%)"
    )

    # Define headers - Use names matching the provided CSV sample
    # Match headers from the provided bybit_trading_journal.csv exactly
    fieldnames = [
        'createdTime', 'updatedTime', 'symbol', 'category', 'Position_Direction',
        'avgEntryPrice', 'avgExitPrice', 'closedSize', 'closedPnl', 'leverage',
        'orderId', 'execType', 'orderType', 'side', 'qty', 'cumEntryValue',
        'orderPrice', 'fillCount', 'cumExitValue', 'Strategy', 'Emotions', 'Lessons_Learned'
    ]

    # Prepare data for the entry record
    # Note: We only have entry data here. Exit/PnL/etc. are marked as placeholders.
    entry_data = {
        'createdTime': now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], # Millisecond precision
        'updatedTime': now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'symbol': symbol.replace('/', '').replace(':', ''), # Match journal format (e.g., BTCUSDT not BTC/USDT:USDT)
        'category': CONFIG.market_type,
        'Position_Direction': 'Long' if side.lower() == 'buy' else 'Short', # Use 'Buy'/'Sell' for API side, 'Long'/'Short' for journal direction
        'avgEntryPrice': f"{avg_price.normalize()}", # Format entry price as string, use normalize
        'avgExitPrice': '', # Placeholder
        'closedSize': '', # Placeholder
        'closedPnl': '', # Placeholder
        'leverage': '', # Placeholder (difficult to get reliably without fetching position again)
        'orderId': order_id or 'N/A', # Use actual order ID if available
        'execType': 'Trade', # Assuming market order fill is a trade execution
        'orderType': 'Market', # Assuming market order entry
        'side': side.capitalize(), # 'Buy' or 'Sell' as used in CCXT/API
        'qty': f"{qty.normalize()}", # Format quantity as string, normalize for cleaner look
        'cumEntryValue': f"{(qty * avg_price).normalize()}", # Estimated entry value, normalize
        'orderPrice': f"{avg_price.normalize()}", # Use average fill price for market order 'price' field in journal, normalize
        'fillCount': '1', # Placeholder, might be multiple fills for large market orders
        'cumExitValue': '', # Placeholder
        'Strategy': strategy_id[:250], # Log strategy string (limit length if needed)
        'Emotions': '', # Manual field
        'Lessons_Learned': '' # Manual field
    }

    try:
        file_exists = os.path.isfile(filepath)
        # Use 'a+' mode to append if file exists, create if not
        with open(filepath, 'a+', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Check if file is empty or newly created to write header
            csvfile.seek(0) # Go to the beginning of the file
            is_empty = not csvfile.read(1) # Read one character, if empty, file is empty
            if not file_exists or is_empty:
                writer.writeheader() # Write header only if file is new or empty
            writer.writerow(entry_data)
        logger.info(f"Logged trade entry to journal: {filepath}")
    except IOError as e:
        logger.error(f"{Fore.RED}Error writing entry to journal file '{filepath}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error writing entry to journal: {e}", exc_info=True)


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    """Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision."""
    # global MARKET_INFO, EXCHANGE, CONFIG, order_tracker # Need global for ASSIGNING to order_tracker
    global order_tracker # Only order_tracker is assigned to in this function

    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"Attempting {trade_action} for {symbol}...")

    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(Fore.RED + f"{trade_action} failed: Market info or Exchange not available.")
        return False

    # --- Pre-computation & Validation ---
    quote_currency = MARKET_INFO.get('settle', 'USDT') # Use settle currency (e.g., USDT)
    # Need to use the dedicated function to get balance with retries
    free_balance, total_equity = get_balance(quote_currency)
    if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid, NaN, or zero account equity ({total_equity}). Cannot calculate risk capital.")
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: No valid equity.")
        return False

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid ATR value ({atr.normalize() if atr is not None else 'None'}). Check indicator calculation.")
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid ATR.")
        return False

    # Fetch current ticker price using fetch_ticker with retries
    ticker_data = None
    try:
        # fetch_with_retries handles category param automatically
        # Use market ID for fetch_ticker
        ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, MARKET_INFO['id'], params={'category': CONFIG.market_type})
    except Exception as e: # Catch exceptions from fetch_with_retries itself
         logger.error(Fore.RED + f"{trade_action} failed: Unhandled exception fetching ticker: {e}", exc_info=True)
         termux_notify("Entry Failed", f"{symbol} {side.upper()}: Ticker fetch error.")
         return False

    if not ticker_data or ticker_data.get("last") is None:
         # fetch_with_retries already logged details if it failed and returned None
         logger.error(Fore.RED + f"{trade_action} failed: Cannot fetch current ticker price for sizing/SL calculation. Ticker data: {ticker_data}")
         termux_notify("Entry Failed", f"{symbol} {side.upper()}: No valid ticker price.")
         return False

    try:
        # Use 'last' price as current price estimate, convert to Decimal
        price = Decimal(str(ticker_data["last"]))
        if price <= Decimal(0):
             logger.error(Fore.RED + f"{trade_action} failed: Fetched current price ({price.normalize()}) is zero or negative. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Price is zero/negative.")
             return False
        logger.debug(f"Current ticker price: {price:.8f} {quote_currency}") # Log with high precision for debug

        # --- Calculate Stop Loss Price ---
        sl_distance_points_raw = CONFIG.sl_atr_multiplier * atr
        # Ensure stop distance is positive after calculation
        sl_distance_points = sl_distance_points_raw.copy_abs()

        if sl_distance_points <= Decimal("0"): # Use Decimal zero
             logger.error(f"{Fore.RED}{trade_action} failed: Stop distance calculation resulted in zero or negative value ({sl_distance_points.normalize()}). Check ATR ({atr:.6f}) and multiplier ({CONFIG.sl_atr_multiplier.normalize()}).")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero SL distance.")
             return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else: # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations/API call
        # Bybit V5 set-trading-stop requires the trigger price as a string.
        sl_price_str_for_api = format_price(symbol, sl_price_raw)
        # Convert back to Decimal *after* formatting for consistent internal representation if needed,
        # but for the API call we use the string. Let's keep the Decimal for validation.
        try:
            sl_price_decimal_for_validation = Decimal(sl_price_str_for_api)
        except InvalidOperation:
             logger.error(Fore.RED + f"{trade_action} failed: Formatted SL price '{sl_price_str_for_api}' is not a valid Decimal. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid formatted SL.")
             return False

        logger.debug(f"ATR: {atr:.6f}, SL Multiplier: {CONFIG.sl_atr_multiplier.normalize()}, Raw SL Distance Points: {sl_distance_points_raw:.6f}, Absolute SL Distance Points: {sl_distance_points:.6f}")
        logger.debug(f"Raw SL Price: {sl_price_raw:.8f}, Formatted SL Price for API: {sl_price_str_for_api} (Decimal: {sl_price_decimal_for_validation})")


        # Sanity check SL placement relative to current price
        # Use a small multiple of price tick size for tolerance if available, else a tiny Decimal
        try:
            price_precision_info = MARKET_INFO['precision'].get('price')
            # If precision is number of decimals (int), calculate tick size 1 / (10^decimals)
            if isinstance(price_precision_info, int):
                price_tick_size = Decimal(1) / (Decimal(10) ** price_precision_info)
            # If precision is tick size (string or Decimal)
            elif isinstance(price_precision_info, (str, Decimal)):
                 price_tick_size = Decimal(str(price_precision_info))
            else:
                 price_tick_size = Decimal("1E-8") # Fallback tiny Decimal
        except Exception:
            price_tick_size = Decimal("1E-8") # Fallback tiny Decimal
            logger.warning(f"Could not determine price tick size from market info. Using default: {price_tick_size:.1E}")


        tolerance_ticks = price_tick_size * Decimal('5') # Allow a few ticks tolerance

        # Ensure SL price is valid (not NaN, > 0)
        if sl_price_decimal_for_validation.is_nan() or sl_price_decimal_for_validation <= Decimal("0"):
             logger.error(Fore.RED + f"{trade_action} failed: Calculated/Formatted SL price ({sl_price_decimal_for_validation.normalize()}) is invalid. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid SL Price.")
             return False

        # Use price_decimal_for_validation for comparison to price
        if side == "buy" and sl_price_decimal_for_validation >= price - tolerance_ticks:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price_decimal_for_validation.normalize()}) is too close to or above current price ({price.normalize()}) [Tolerance: {tolerance_ticks.normalize()}]. Check ATR/multiplier or market precision. Aborting.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: SL too close/wrong side.")
            return False
        if side == "sell" and sl_price_decimal_for_validation <= price + tolerance_ticks:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated SL price ({sl_price_decimal_for_validation.normalize()}) is too close to or below current price ({price.normalize()}) [Tolerance: {tolerance_ticks.normalize()}]. Check ATR/multiplier or market precision. Aborting.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: SL too close/wrong side.")
            return False


        # --- Calculate Position Size ---
        risk_amount_quote = total_equity * risk_percentage
        # Stop distance in quote currency (use absolute difference between current price and SL price, ensure Decimals)
        # Use the *formatted* SL price decimal for this calculation as it's what will be sent to the exchange.
        stop_distance_quote = (price - sl_price_decimal_for_validation).copy_abs()

        if stop_distance_quote <= Decimal("0"):
             logger.error(Fore.RED + f"{trade_action} failed: Stop distance in quote currency is zero or negative ({stop_distance_quote.normalize()}). Check ATR, multiplier, or market precision. Cannot calculate size.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero stop distance.")
             return False

        # Calculate quantity based on contract size and linear/inverse type
        contract_size = Decimal(str(MARKET_INFO.get('contractSize', '1'))) # Ensure Decimal, default to '1' if missing
        qty_raw = Decimal('0')

        # --- Sizing Logic ---
        # (Logic explanation retained from previous thought process - confirmed formulas)
        if CONFIG.market_type in ['linear', 'swap']: # Linear and Swap use similar quantity calculation
            # Qty (Base) = Risk Amount (Quote) / Stop Distance (Quote)
            if stop_distance_quote <= Decimal('0'): # Double check division by zero possibility
                 logger.error(Fore.RED + f"{trade_action} failed: Stop distance is zero for linear/swap sizing. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero stop dist linear/swap.")
                 return False
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(f"Linear/Swap Sizing: Qty (Base) = {risk_amount_quote.normalize()} {quote_currency} / {stop_distance_quote.normalize()} {quote_currency} = {qty_raw.normalize()}")

        elif CONFIG.market_type == 'inverse':
            # Qty (Contracts) = Risk Amount (Quote) * Price (Quote/Base) / Stop Distance (Quote/Base)
            if price <= Decimal("0"):
                logger.error(Fore.RED + f"{trade_action} failed: Cannot calculate inverse size with zero or negative price.")
                termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero/negative price for sizing.")
                return False
            if stop_distance_quote <= Decimal('0'): # Double check division by zero possibility
                 logger.error(Fore.RED + f"{trade_action} failed: Stop distance is zero for inverse sizing. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero stop dist inverse.")
                 return False
            qty_raw = (risk_amount_quote * price) / stop_distance_quote
            logger.debug(f"Inverse Sizing (Contract Size = {contract_size.normalize()} {quote_currency}): Qty (Contracts) = ({risk_amount_quote.normalize()} * {price.normalize()}) / {stop_distance_quote.normalize()} = {qty_raw.normalize()}")

        else:
            logger.error(f"{trade_action} failed: Unsupported market type for sizing: {CONFIG.market_type}")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: Unsupported market type.")
            return False

        # --- Format and Validate Quantity ---
        # Format quantity according to market precision (ROUND_DOWN to be conservative)
        qty_formatted_str = format_amount(symbol, qty_raw, ROUND_DOWN)
        try:
             qty = Decimal(qty_formatted_str) # Convert back to Decimal after formatting
        except InvalidOperation:
             logger.error(Fore.RED + f"{trade_action} failed: Formatted quantity '{qty_formatted_str}' is not a valid Decimal. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid formatted qty.")
             return False

        logger.debug(f"Risk Amount: {risk_amount_quote.normalize()} {quote_currency}, Stop Distance: {stop_distance_quote.normalize()} {quote_currency}")
        logger.debug(f"Raw Qty: {qty_raw:.12f}, Formatted Qty (Rounded Down): {qty.normalize()}") # Use normalize for log

        # Validate Quantity Against Market Limits
        min_qty_str = str(MARKET_INFO['limits']['amount'].get('min')) if MARKET_INFO['limits']['amount'].get('min') is not None else "0"
        max_qty_str = str(MARKET_INFO['limits']['amount'].get('max')) if MARKET_INFO['limits']['amount'].get('max') is not None else None
        try:
            min_qty = Decimal(min_qty_str)
        except InvalidOperation:
             logger.error(Fore.RED + f"{trade_action} failed: Could not parse min amount limit '{min_qty_str}'. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid min qty limit.")
             return False

        # max_qty is infinity if None, otherwise convert to Decimal
        try:
            max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else Decimal('Infinity')
        except InvalidOperation:
             logger.error(Fore.RED + f"{trade_action} failed: Could not parse max amount limit '{max_qty_str}'. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid max qty limit.")
             return False


        # Use epsilon for zero check
        if qty < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated quantity ({qty.normalize()}) is zero or below minimum ({min_qty.normalize()}). Risk amount ({risk_amount_quote:.4f}), stop distance ({stop_distance_quote:.4f}), or equity might be too small. Cannot place order.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: Qty < Min Qty.")
            return False
        if max_qty != Decimal('Infinity') and qty > max_qty:
            logger.warning(Fore.YELLOW + f"Calculated quantity {qty.normalize()} exceeds maximum {max_qty.normalize()}. Capping order size to {max_qty.normalize()}.")
            qty = max_qty # Use the Decimal max_qty
            # Re-format capped amount - crucial! Use ROUND_DOWN again.
            qty_formatted_str = format_amount(symbol, qty, ROUND_DOWN)
            try:
                qty = Decimal(qty_formatted_str)
            except InvalidOperation:
                logger.error(Fore.RED + f"{trade_action} failed: Re-formatted capped quantity '{qty_formatted_str}' is not a valid Decimal. Aborting.")
                termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid capped qty format.")
                return False

            logger.info(f"Re-formatted capped Qty: {qty.normalize()}")
            # Double check if capped value is now below min (unlikely but possible with large steps)
            if qty < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
                 logger.error(Fore.RED + f"{trade_action} failed: Capped quantity ({qty.normalize()}) is now below minimum ({min_qty.normalize()}) or zero. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Capped Qty < Min Qty.")
                 return False

        # Validate minimum cost/notional value (Bybit V5 often uses notional)
        min_notional_str = str(MARKET_INFO['limits'].get('notional', {}).get('min')) if MARKET_INFO['limits'].get('notional', {}).get('min') is not None else None
        min_cost_str = str(MARKET_INFO['limits'].get('cost', {}).get('min')) if MARKET_INFO['limits'].get('cost', {}).get('min') is not None else None

        estimated_notional_or_cost = qty * price # Estimate notional/cost as Qty * Current Price
        if min_notional_str is not None:
            try:
                min_notional = Decimal(min_notional_str)
                if estimated_notional_or_cost.copy_abs() < min_notional: # Use abs in comparison
                    logger.error(Fore.RED + f"{trade_action} failed: Estimated order notional value ({estimated_notional_or_cost.normalize()} {quote_currency}) is below minimum required ({min_notional.normalize()} {quote_currency}). Increase risk or equity. Cannot place order.")
                    termux_notify("Entry Failed", f"{symbol} {side.upper()}: Value < Min Notional.")
                    return False
                logger.debug(f"Passed Min Notional check ({estimated_notional_or_cost.normalize()} >= {min_notional.normalize()})")
            except Exception as notional_err:
                 logger.warning(f"Could not validate against minimum notional limit: {notional_err}. Skipping check.", exc_info=True)
        elif min_cost_str is not None: # Fallback to min cost if notional not present
             try:
                 min_cost = Decimal(min_cost_str)
                 if estimated_notional_or_cost.copy_abs() < min_cost: # Use abs in comparison
                      logger.error(Fore.RED + f"{trade_action} failed: Estimated order cost ({estimated_notional_or_cost.normalize()} {quote_currency}) is below minimum required ({min_cost.normalize()} {quote_currency}). Increase risk or equity. Cannot place order.")
                      termux_notify("Entry Failed", f"{symbol} {side.upper()}: Cost < Min Cost.")
                      return False
                 logger.debug(f"Passed Min Cost check ({estimated_notional_or_cost.normalize()} >= {min_cost.normalize()})")
             except Exception as cost_err:
                  logger.warning(f"Could not validate against minimum cost limit: {cost_err}. Skipping check.", exc_info=True)
        else:
             logger.debug("No minimum notional or cost limit found in market info to validate against.")


        logger.info(Fore.YELLOW + f"Calculated Order: Side={side.upper()}, Qty={qty.normalize()}, Entry≈{price:.4f}, SL={sl_price_str_for_api} (ATR={atr:.4f})")

    except (InvalidOperation, TypeError, DivisionByZero, KeyError) as e:
         logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Error during pre-calculation/validation: {e}", exc_info=True)
         termux_notify("Entry Failed", f"{symbol} {side.upper()}: Calc error.")
         return False
    except Exception as e: # Catch any other unexpected errors
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during pre-calculation: {e}", exc_info=True)
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Unexpected calc error.")
        return False

    # --- Cast the Market Order Spell ---
    order = None
    order_id = None
    filled_qty = Decimal("0.0") # Initialize filled_qty for later use
    average_price = price # Initialize average_price for later use, fallback to entry estimate

    try:
        logger.trade(f"Submitting {side.upper()} market order for {qty.normalize()} {symbol}...")
        # Bybit V5 create_market_order requires category and symbol (handled by wrapper & MARKET_INFO['id'])
        # Set positionIdx=0 for one-way mode.
        create_order_params = {'positionIdx': 0, 'category': CONFIG.market_type} # Ensure category is in params

        # Use market ID for symbol parameter
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=MARKET_INFO['id'], # Use market ID
            side=side,
            amount=float(qty), # Explicitly cast Decimal qty to float for CCXT API
            params=create_order_params
        )

        if order is None:
            # fetch_with_retries logged the error
            logger.error(Fore.RED + f"{trade_action} failed: Market order placement failed after retries.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: Order place fail.")
            return False

        logger.debug(f"Market order raw response: {order}")
        # Try to get the order ID from the unified field first
        order_id = order.get('id')

        # --- Verify Order Fill (Crucial Step) ---
        # Market orders should fill immediately. Check the response for confirmation.
        # Bybit V5 market order response structure is different from limit/stop orders.
        # It usually includes filled quantity and average price directly in the 'info' or 'trades' section.
        fill_verified = False
        if isinstance(order.get('info'), dict):
            info_data = order['info']
            ret_code = info_data.get('retCode')
            ret_msg = info_data.get('retMsg')

            if ret_code == 0: # Bybit V5 success code
                 logger.debug(f"{trade_action}: Market order submission retCode 0 ({ret_msg}). Attempting to parse fill details from response.")
                 # V5 market order response structure: info -> result -> list -> [order_details]
                 # or sometimes directly in info -> result
                 # Let's check for cumExecQty and avgPrice in info['result'] or info['list'][0]
                 result_data = info_data.get('result', {})
                 order_id_from_response = result_data.get('orderId')
                 cum_exec_qty_raw = result_data.get('cumExecQty')
                 avg_price_raw = result_data.get('avgPrice')

                 if order_id_from_response:
                     order_id = order_id_from_response # Prefer ID from result if available
                     logger.debug(f"Order ID from V5 result: {order_id}")

                 # Try parsing fill details from result, fallback to list if present
                 if cum_exec_qty_raw is None and isinstance(result_data.get('list'), list) and result_data['list']:
                     list_item = result_data['list'][0] # Assume first item is the relevant order
                     cum_exec_qty_raw = list_item.get('cumExecQty')
                     avg_price_raw = list_item.get('avgPrice')
                     if order_id is None: order_id = list_item.get('orderId', order_id) # Use orderId from list if no result ID

                 if cum_exec_qty_raw is not None and avg_price_raw is not None:
                     try:
                         filled_qty_from_response = Decimal(str(cum_exec_qty_raw))
                         avg_price_from_response = Decimal(str(avg_price_raw))

                         # Check if the filled quantity is significant
                         if filled_qty_from_response.copy_abs() >= CONFIG.position_qty_epsilon:
                             filled_qty = filled_qty_from_response
                             average_price = avg_price_from_response
                             fill_verified = True
                             logger.trade(Fore.GREEN + Style.BRIGHT + f"Market order filled immediately (ID: {order_id}). Filled: {filled_qty.normalize()} @ {average_price.normalize()}") # Use normalize for display
                         else:
                             logger.warning(f"Market order submitted (ID: {order_id}), but filled quantity from response ({filled_qty_from_response.normalize()}) is zero or negligible. Falling back to status check.")
                             # Need to check status to be sure
                             pass # Fall through to status check below

                     except InvalidOperation as e:
                         logger.error(f"Failed to parse fill details (cumExecQty/avgPrice) from V5 response for order {order_id}: {e}. Raw: cumExecQty={cum_exec_qty_raw}, avgPrice={avg_price_raw}. Falling back to status check.", exc_info=True)
                         # Fall through to status check below
                 elif order_id is None:
                     # retCode 0 but no orderId found in result or list - very unusual
                     logger.error(Fore.RED + f"{trade_action} failed: Market order submitted (retCode 0) but no Order ID or fill details found in V5 response. Cannot track or confirm fill. Aborting.")
                     termux_notify("Entry Failed", f"{symbol} {side.upper()}: No order ID/fill in V5 response.")
                     return False
                 else:
                      # retCode 0, orderId found, but no fill details in response. Likely still processing or needs status check.
                      logger.warning(f"Market order submitted (ID: {order_id}, retCode 0), but immediate fill details (cumExecQty/avgPrice) missing in response. Falling back to status check.")
                      # Fall through to status check below
            else:
                 # Non-zero retCode means submission failed
                 error_msg = info_data.get('retMsg', 'Unknown error') if isinstance(info_data, dict) else 'N/A'
                 error_code = info_data.get('retCode', 'N/A') if isinstance(info_data, dict) else 'N/A'
                 logger.error(Fore.RED + f"{trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Submit failed ({ret_code}).")
                 return False
        else: # Order response does not contain V5 info dict or retCode
             # Rely on standard CCXT 'id' if present
             logger.warning("Market order response does not contain expected V5 'info' structure. Relying on standard 'id' field and check_order_status.")
             if not order_id:
                 logger.error(Fore.RED + f"{trade_action} failed: Market order response missing both standard 'id' and V5 'info'. Cannot track order. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: No order ID in response.")
                 return False
             # Order ID is available, need to check status
             logger.info(f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}...")
             time.sleep(CONFIG.order_check_delay_seconds)


        # --- If fill wasn't confirmed immediately, check status ---
        if not fill_verified:
             if order_id is None:
                  logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Cannot check order status, order ID is missing. Aborting.")
                  termux_notify("Entry Failed", f"{symbol} {side.upper()}: Order ID missing for check.")
                  return False

             logger.info(f"Verifying fill status via check_order_status for order {order_id}...")
             # check_order_status expects CCXT unified symbol string
             order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

             if order_status_data is None or order_status_data.get('status') != 'closed': # check_order_status failed/timed out or returned non-closed
                 order_final_status = order_status_data.get('status', 'unknown') if order_status_data else 'check_failed'
                 filled_qty_check = Decimal(str(order_status_data.get('filled', '0.0'))) if order_status_data else Decimal('0.0')

                 logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} did not fill successfully via status check: Status '{order_final_status}', Filled Qty: {filled_qty_check.normalize()}. Aborting SL placement.")
                 # Attempt to cancel if not already in a terminal state (defensive)
                 terminal_ccxt_statuses = ['canceled', 'rejected', 'expired']
                 if order_final_status not in terminal_ccxt_statuses and order_id:
                      try:
                           logger.info(f"Attempting cancellation of failed/stuck order {order_id}.")
                           cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                           # cancel_order expects CCXT unified symbol string
                           fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)
                           logger.warning(f"Cancellation request sent for order {order_id}.")
                      except ccxt.OrderNotFound:
                           logger.warning(f"Order {order_id} already not found when attempting cancellation.")
                      except Exception as cancel_err:
                           logger.warning(f"Failed to send cancellation for stuck order {order_id}: {cancel_err}")

                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Order not filled ({order_final_status}).")
                 return False
             else: # check_order_status returned 'closed'
                 filled_qty_status = Decimal(str(order_status_data.get('filled', '0.0')))
                 # Use average from status check if available, fallback to initial estimate
                 average_price_status = Decimal(str(order_status_data.get('average', str(price))))
                 logger.trade(Fore.GREEN + Style.BRIGHT + f"Market order confirmed filled via status check (ID: {order_id}). Filled: {filled_qty_status.normalize()} @ {average_price_status.normalize()}") # Use normalize
                 filled_qty = filled_qty_status
                 average_price = average_price_status
                 fill_verified = True # Confirmed via status check


        # --- If fill is verified, proceed to SL placement ---
        if not fill_verified:
            # This case should ideally not be reached if the logic is correct, but defensive check.
            logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Fill verification failed after immediate check and status check fallback for order {order_id}. Aborting.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: Fill verify failed.")
            return False

        # --- Place Initial Stop-Loss Order (Set on Position for Bybit V5) ---
        # Use the actual filled quantity and average price for logging/journaling,
        # but the SL is set on the *position*, not the order, based on the *entry* logic price.
        # Bybit V5 set-trading-stop doesn't take quantity or a specific fill ID. It applies to the position.
        position_side = "long" if side == "buy" else "short"
        logger.trade(f"Setting initial SL for new {position_side.upper()} position (filled qty: {filled_qty.normalize()}, entry price: {average_price.normalize()})...")

        # Use the SL price calculated earlier, already formatted string
        sl_price_str_for_api_for_position = sl_price_str_for_api

        # Define parameters for setting the stop-loss on the position
        set_sl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': MARKET_INFO['id'], # Use exchange-specific market ID
            'stopLoss': sl_price_str_for_api_for_position, # Trigger price string
            'slTriggerBy': CONFIG.sl_trigger_by,
            'tpslMode': 'Full', # Apply to the whole position
            'side': 'Buy' if position_side == 'long' else 'Sell', # Add side parameter as required by V5 docs ('Buy' for Long position, 'Sell' for Short position)
            'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position SL: Trigger={sl_price_str_for_api_for_position}, TriggerBy={CONFIG.sl_trigger_by}, Side={set_sl_params['side']}")
        logger.debug(f"Set SL Params (for setTradingStop): {set_sl_params}")

        sl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            # Bybit V5 uses POST /v5/position/set-trading-stop
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                # fetch_with_retries handles the call and retries
                sl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_sl_params)
            else:
                logger.error(Fore.RED + "Cannot set SL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # If the method isn't found, it's a configuration/library issue, not retryable via fetch_with_retries.
                # Re-raise as a critical failure to trigger emergency close
                raise ccxt.NotSupported("SL setting method not available via CCXT.")

            logger.debug(f"Set SL raw response: {sl_set_response}")

            # Handle potential failure from fetch_with_retries (returns None on failure)
            if sl_set_response is None:
                 # fetch_with_retries already logged the failure
                 # Re-raise as a critical failure to trigger emergency close
                 raise ccxt.ExchangeError("Set SL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(sl_set_response.get('info'), dict) and sl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Stop Loss successfully set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_api_for_position}).")
                # --- Update Global State ---
                # Use a placeholder to indicate SL is active on the position
                sl_marker_id = f"POS_SL_{position_side.upper()}"
                # Use global keyword as we are ASSIGNING to order_tracker
                global order_tracker
                order_tracker[position_side] = {"sl_id": sl_marker_id, "tsl_id": None}
                logger.info(f"Updated order tracker: {order_tracker}")

                # --- Log Entry to Journal ---
                log_trade_entry_to_journal(
                    symbol=CONFIG.symbol, side=side, qty=filled_qty, avg_price=average_price, order_id=order_id
                )

                # Use actual average fill price in notification
                entry_msg = (
                    f"ENTERED {side.upper()} {filled_qty.normalize()} {symbol.split('/')[0]} @ {average_price.normalize()}. "
                    f"Initial SL @ {sl_price_str_for_api_for_position}. TSL pending profit threshold."
                )
                logger.trade(Back.GREEN + Fore.BLACK + Style.BRIGHT + entry_msg + Style.RESET_ALL) # Ensure reset
                termux_notify("Trade Entry", f"{symbol} {side.upper()} @ {average_price.normalize()}, SL: {sl_price_str_for_api_for_position}")
                return True # SUCCESS!

            else:
                 # Extract error message if possible
                 error_msg = "Unknown reason."
                 error_code = None
                 if isinstance(sl_set_response.get('info'), dict):
                      error_msg = sl_set_response['info'].get('retMsg', error_msg)
                      error_code = sl_set_response['info'].get('retCode')
                      error_msg += f" (Code: {error_code})"
                 # Re-raise as a critical failure to trigger emergency close
                 raise ccxt.ExchangeError(f"Stop loss setting failed. Exchange message: {error_msg}")

        # --- Handle SL Setting Failures ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
             # This is critical - position opened but SL setting failed. Emergency close needed.
             logger.critical(Fore.RED + Style.BRIGHT + f"CRITICAL: Failed to set stop-loss on position after entry: {e}. Position is UNPROTECTED.")
             logger.warning(Fore.YELLOW + "Attempting emergency market closure of unprotected position...")
             termux_notify("CRITICAL ERROR!", f"{symbol} SL failed. Attempting emergency close.")
             try:
                 # Determine which position side was attempted based on the entry side
                 pos_side_to_check = "long" if side == "buy" else "short"

                 # Re-fetch position quantity just in case (belt and suspenders)
                 # Use the snapshot logic from trading_spell_cycle to get the latest quantity
                 current_positions_after_sl_attempt = get_current_position(symbol)
                 emergency_close_qty = Decimal("0.0")

                 if current_positions_after_sl_attempt and pos_side_to_check in current_positions_after_sl_attempt:
                     pos_qty = current_positions_after_sl_attempt[pos_side_to_check].get('qty', Decimal("0.0"))
                     if pos_qty is not None: # Check if qty is not None before taking abs
                         emergency_close_qty = pos_qty.copy_abs()

                 # Fallback to filled_qty if current_positions fetch failed or returned zero/None
                 if emergency_close_qty.copy_abs() < CONFIG.position_qty_epsilon:
                     emergency_close_qty = filled_qty.copy_abs() # Use filled qty from successful order
                     logger.warning(f"Could not fetch current position qty for emergency close, using filled qty ({emergency_close_qty.normalize()}).")

                 # If quantity is still negligible, maybe position wasn't actually opened significantly?
                 if emergency_close_qty.copy_abs() < CONFIG.position_qty_epsilon:
                      logger.warning(f"Emergency closure quantity ({emergency_close_qty.normalize()}) is negligible. Position may not have been significantly opened. Skipping emergency closure.")
                      termux_notify("Emergency Skipped", f"{symbol} Qty negligible. Check exchange.")
                      # Reset tracker state as position is likely flat/negligible
                      order_tracker[pos_side_to_check] = {"sl_id": None, "tsl_id": None}
                      logger.info(f"Updated order tracker after skipping emergency close: {order_tracker}")
                      return False # Indicate failure of the entry process

                 emergency_close_side = "sell" if pos_side_to_check == "long" else "buy"
                 # Format emergency close quantity precisely
                 close_qty_str = format_amount(symbol, emergency_close_qty, ROUND_DOWN)
                 try:
                     close_qty_decimal = Decimal(close_qty_str)
                 except InvalidOperation:
                      logger.critical(Fore.RED + f"Emergency closure failed: Invalid Decimal after formatting close quantity '{close_qty_str}'. MANUAL CLOSURE REQUIRED for {pos_side_to_check.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {pos_side_to_check.upper()} POS UNPROTECTED & BAD QTY! Close manually!")
                      # Do NOT reset tracker state here, as we don't know the position status for sure.
                      return False # Indicate failure of the entire entry process


                 # Check against minimum quantity again before closing
                 try:
                      min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
                 except (KeyError, InvalidOperation, TypeError):
                      logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                      min_qty_close = Decimal("0") # Assume zero if unavailable


                 if close_qty_decimal < min_qty_close or close_qty_decimal.copy_abs() < CONFIG.position_qty_epsilon:
                      logger.critical(f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} for {pos_side_to_check} position is below minimum {min_qty_close.normalize()} or zero. MANUAL CLOSURE REQUIRED for {pos_side_to_check.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {pos_side_to_check.upper()} POS UNPROTECTED & < MIN QTY! Close manually!")
                      # Do NOT reset tracker state here, as we don't know the position status for sure.
                      return False # Indicate failure of the entire entry process

                 # Place the emergency closure order
                 emergency_close_params = {'reduceOnly': True, 'positionIdx': 0, 'category': CONFIG.market_type} # Ensure it only closes + one-way mode + category
                 # Use market ID for symbol parameter
                 emergency_close_order = fetch_with_retries(
                     EXCHANGE.create_market_order,
                     symbol=MARKET_INFO['id'], # Use market ID
                     side=emergency_close_side,
                     amount=float(close_qty_decimal), # CCXT needs float
                     params=emergency_close_params
                 )

                 if emergency_close_order and (emergency_close_order.get('id') or (isinstance(emergency_close_order.get('info'), dict) and emergency_close_order['info'].get('retCode') == 0)):
                     close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
                     logger.trade(Fore.GREEN + f"Emergency closure order placed successfully: ID {close_id}")
                     termux_notify("Closure Attempted", f"{symbol} emergency closure sent.")
                     # Reset tracker state as position *should* be closing (best effort)
                     order_tracker[pos_side_to_check] = {"sl_id": None, "tsl_id": None}
                     logger.info(f"Updated order tracker after emergency close attempt: {order_tracker}")
                 else:
                      error_msg = emergency_close_order.get('info', {}).get('retMsg', 'Unknown error') if isinstance(emergency_close_order, dict) else str(emergency_close_order)
                      logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed): {error_msg}. MANUAL INTERVENTION REQUIRED for {pos_side_to_check.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {pos_side_to_check.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                      # Do NOT reset tracker state here.

             except Exception as close_err:
                 logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure): {close_err}. MANUAL INTERVENTION REQUIRED for {pos_side_to_check.upper()} position!", exc_info=True)
                 termux_notify("EMERGENCY!", f"{symbol} {pos_side_to_check.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
                 # Do NOT reset tracker state here.

             return False # Signal overall failure of the entry process

        except Exception as e:
            logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error setting SL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position may be open without Stop Loss due to unexpected SL setting error. MANUAL INTERVENTION ADVISED.")
            termux_notify("CRITICAL ERROR!", f"{symbol} Unexpected SL error. Attempting emergency close.")
            # Consider emergency closure here too? Yes, safer. Re-use the emergency closure logic.
            try:
                 position_side = "long" if side == "buy" else "short"
                 # Re-fetch position quantity just in case
                 current_positions = get_current_position(symbol)
                 emergency_close_qty = Decimal("0.0")
                 if current_positions and position_side in current_positions:
                     pos_qty = current_positions[position_side].get('qty', Decimal("0.0"))
                     if pos_qty is not None: emergency_close_qty = pos_qty.copy_abs()

                 if emergency_close_qty.copy_abs() < CONFIG.position_qty_epsilon:
                      logger.warning(f"Emergency closure quantity ({emergency_close_qty.normalize()}) is negligible after unexpected SL error. Position may not have been significantly opened. Skipping emergency closure.")
                      termux_notify("Emergency Skipped", f"{symbol} Qty negligible. Check exchange.")
                      # Reset tracker state as position is likely flat/negligible
                      order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                      logger.info(f"Updated order tracker after skipping emergency close: {order_tracker}")
                      return False # Indicate failure

                 emergency_close_side = "sell" if position_side == "long" else "buy"
                 close_qty_str = format_amount(symbol, emergency_close_qty, ROUND_DOWN)
                 try:
                     close_qty_decimal = Decimal(close_qty_str)
                 except InvalidOperation:
                      logger.critical(Fore.RED + f"Emergency closure failed: Invalid Decimal after formatting close quantity '{close_qty_str}'. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & BAD QTY! Close manually!")
                      return False

                 try:
                      min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
                 except (KeyError, InvalidOperation, TypeError):
                      logger.warning("Could not determine minimum order quantity for emergency closure validation.")
                      min_qty_close = Decimal("0") # Assume zero if unavailable

                 if close_qty_decimal < min_qty_close or close_qty_decimal.copy_abs() < CONFIG.position_qty_epsilon:
                      logger.critical(f"{Fore.RED}Emergency closure quantity {close_qty_decimal.normalize()} for {position_side} position is below minimum {min_qty_close.normalize()} or zero. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!")
                      return False

                 emergency_close_params = {'reduceOnly': True, 'positionIdx': 0, 'category': CONFIG.market_type} # Ensure it only closes + one-way mode + category
                 emergency_close_order = fetch_with_retries(
                     EXCHANGE.create_market_order,
                     symbol=MARKET_INFO['id'], # Use market ID
                     side=emergency_close_side,
                     amount=float(close_qty_decimal),
                     params=emergency_close_params
                 )
                 if emergency_close_order and (emergency_close_order.get('id') or (isinstance(emergency_close_order.get('info'), dict) and emergency_close_order['info'].get('retCode') == 0)):
                     close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
                     logger.trade(Fore.GREEN + f"Emergency closure order placed successfully after unexpected SL error: ID {close_id}")
                     termux_notify("Closure Attempted", f"{symbol} emergency closure sent after SL error.")
                     # Use global keyword as we are ASSIGNING to order_tracker
                     global order_tracker
                     order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
                     logger.info(f"Updated order tracker after emergency close attempt: {order_tracker}")
                 else:
                      error_msg = emergency_close_order.get('info', {}).get('retMsg', 'Unknown error') if isinstance(emergency_close_order, dict) else str(emergency_close_order)
                      logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed) after unexpected SL error: {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
                      termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
            except Exception as close_err:
                 logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure) after unexpected SL error: {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
                 termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")

            return False # Signal overall failure

    # --- Handle Initial Market Order Failures ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
        # Error placing the initial market order itself (handled by fetch_with_retries re-raising)
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Exchange error placing market order: {e}")
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Exchange error ({type(e).__name__}).")
        return False
    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during market order placement: {e}", exc_info=True)
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Unexpected order error.")
        return False

def manage_trailing_stop(
    symbol: str,
    position_side: str, # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal
) -> None:
    """Manages the activation and setting of a trailing stop loss on the position, using Decimal."""
    # global order_tracker, EXCHANGE, MARKET_INFO, CONFIG # Need global for ASSIGNING to order_tracker
    global order_tracker # Only order_tracker is assigned to in this function

    logger.debug(f"Checking TSL status for {position_side.upper()} position...")

    if EXCHANGE is None or MARKET_INFO is None:
         logger.error("Exchange or Market Info not available, cannot manage TSL.")
         return

    # --- Initial Checks ---
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        # Use global keyword as we are ASSIGNING to order_tracker
        global order_tracker
        if order_tracker[position_side]["sl_id"] or order_tracker[position_side]["tsl_id"]:
             logger.info(f"Position {position_side} appears closed or invalid (Qty: {position_qty.normalize()}, Entry: {entry_price.normalize()}). Clearing stale order trackers.")
             order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return # No significant position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + f"Cannot evaluate TSL activation: Invalid ATR value ({atr.normalize() if atr is not None else 'None'}).")
        return

    if current_price is None or current_price.is_nan() or current_price <= Decimal("0"):
         logger.warning(Fore.YELLOW + "Cannot evaluate TSL activation: Invalid current price.")
         return


    # --- Get Current Tracker State ---
    initial_sl_marker = order_tracker[position_side]["sl_id"] # Could be ID or placeholder "POS_SL_..."
    active_tsl_marker = order_tracker[position_side]["tsl_id"] # Could be ID or placeholder "POS_TSL_..."

    # If TSL is already active (has a marker), assume exchange handles the trail.
    if active_tsl_marker:
        log_msg = f"{position_side.upper()} TSL ({active_tsl_marker}) is already active. Exchange is managing the trail."
        logger.debug(log_msg)
        # Sanity check: Ensure initial SL marker is None if TSL is active
        if initial_sl_marker:
             logger.warning(f"Inconsistent state: TSL active ({active_tsl_marker}) but initial SL marker ({initial_sl_marker}) is also present. Clearing initial SL marker.")
             order_tracker[position_side]["sl_id"] = None
        return # TSL is already active, nothing more to do here

    # If TSL is not active, check if we *should* activate it.
    # Requires an initial SL marker to be present to indicate the position is at least protected by a fixed SL.
    if not initial_sl_marker:
        # This can happen if the initial SL setting failed, or if state got corrupted.
        logger.warning(f"Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker. Position might be unprotected or already managed externally.")
        # Consider adding logic here to try and set a regular SL if missing? Or just warn.
        return # Cannot activate TSL if initial SL state is unknown/missing

    # --- Check TSL Activation Condition ---
    profit = Decimal("NaN")
    try:
        if position_side == "long":
            profit = current_price - entry_price
        else: # short
            profit = entry_price - current_price
    except (TypeError, InvalidOperation): # Handle potential NaN in prices
        logger.warning("Cannot calculate profit for TSL check due to NaN price(s).", exc_info=True)
        return

    # Activation threshold in price points
    activation_threshold_points = CONFIG.tsl_activation_atr_multiplier * atr
    logger.debug(f"{position_side.upper()} Profit: {profit.normalize() if not profit.is_nan() else 'N/A'}, TSL Activation Threshold (Points): {activation_threshold_points.normalize()} ({CONFIG.tsl_activation_atr_multiplier.normalize()} * ATR)")

    # Activate TSL only if profit exceeds the threshold (use Decimal comparison)
    if not profit.is_nan() and profit > activation_threshold_points:
        logger.trade(Fore.GREEN + Style.BRIGHT + f"Profit threshold reached for {position_side.upper()} position (Profit {profit:.4f} > Threshold {activation_threshold_points:.4f}). Activating TSL.")

        # --- Set Trailing Stop Loss on Position ---
        # Bybit V5 sets TSL directly on the position using specific parameters.
        # We use the same `set_trading_stop` endpoint as the initial SL, but provide TSL params.

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        # Ensure correct formatting for the API (string representation with sufficient precision)
        # Quantize to a reasonable number of decimal places for percentage (e.g., 3-4)
        # Bybit V5 expects a string representing the percentage value directly.
        # Use .quantize for consistent precision string
        # Ensure the percentage value is positive for the API
        trail_percent_str = str(CONFIG.trailing_stop_percent.copy_abs().quantize(Decimal("0.001"))) # Format to 3 decimal places, use absolute value

        # Bybit V5 Parameters for setting TSL on position:
        # Endpoint: POST /v5/position/set-trading-stop
        set_tsl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': MARKET_INFO['id'], # Use exchange-specific market ID
            'trailingStop': trail_percent_str, # Trailing distance percentage (as string)
            'tpslMode': 'Full', # Apply to the whole position
            'slTriggerBy': CONFIG.tsl_trigger_by, # Trigger type for the trail (LastPrice, MarkPrice, IndexPrice)
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate the trail immediately. If omitted, Bybit activates when price moves favorably by trail %. Check docs.
            # Recommended: Don't set activePrice here. Let Bybit handle the initial activation based on the trail distance from the best price.
            # To remove the fixed SL when activating TSL, Bybit V5 documentation indicates setting 'stopLoss' to "" (empty string) or '0'.
            # Setting to "" is often safer to explicitly indicate removal.
            'stopLoss': '', # Remove the fixed SL when activating TSL by setting it to an empty string
            'side': 'Buy' if position_side == 'long' else 'Sell', # Add side parameter as required by V5 docs ('Buy' for Long position, 'Sell' for Short position)
            'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Side={set_tsl_params['side']}, Removing Fixed SL")
        logger.debug(f"Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            if hasattr(EXCHANGE, 'private_post_position_set_trading_stop'):
                 # fetch_with_retries handles the call and retries
                tsl_set_response = fetch_with_retries(EXCHANGE.private_post_position_set_trading_stop, params=set_tsl_params)
            else:
                logger.error(Fore.RED + "Cannot set TSL: CCXT method 'private_post_position_set_trading_stop' not found for Bybit.")
                # If the method isn't found, it's a configuration/library issue.
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries (returns None on failure)
            if tsl_set_response is None:
                 # fetch_with_retries already logged the failure
                 raise ccxt.ExchangeError("Set TSL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response.get('info'), dict) and tsl_set_response['info'].get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%")
                # --- Update Global State ---
                # Set TSL active marker and clear the initial SL marker
                tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                # Use global keyword as we are ASSIGNING to order_tracker
                global order_tracker
                order_tracker[position_side]["tsl_id"] = tsl_marker_id
                order_tracker[position_side]["sl_id"] = None # Remove initial SL marker marker from tracker
                logger.info(f"Updated order tracker: {order_tracker}")
                termux_notify("TSL Activated", f"{symbol} {position_side.upper()} TSL active ({trail_percent_str}%).")
                return # Success

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(tsl_set_response.get('info'), dict):
                     error_msg = tsl_set_response['info'].get('retMsg', error_msg)
                     error_code = tsl_set_response['info'].get('retCode')
                     error_msg += f" (Code: {error_code})"
                # Check if error was due to trying to remove non-existent SL (might be benign, e.g., SL already hit)
                # Example Bybit code: 110025 = SL/TP order not found or completed
                if error_code == 110025:
                     logger.warning(f"TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL. Assuming TSL is active and fixed SL was already gone.")
                     # Proceed as if successful, update tracker
                     tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                     # Use global keyword as we are ASSIGNING to order_tracker
                     global order_tracker
                     order_tracker[position_side]["tsl_id"] = tsl_marker_id
                     order_tracker[position_side]["sl_id"] = None
                     logger.info(f"Updated order tracker (assuming TSL active despite code 110025): {order_tracker}")
                     termux_notify("TSL Activated*", f"{symbol} {position_side.upper()} TSL active (check exchange).")
                     return # Treat as success for now
                else:
                    raise ccxt.ExchangeError(f"Failed to activate trailing stop loss. Exchange message: {error_msg}")

        # --- Handle TSL Setting Failures ---
        except (ccxt.ExchangeError, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # TSL setting failed. Initial SL marker *should* still be in the tracker if it was set initially.
            # Position might be protected by the initial SL, or might be unprotected if initial SL failed.
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to activate TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + "Position continues with initial SL (if successfully set) or may be UNPROTECTED if initial SL state is uncertain. MANUAL INTERVENTION ADVISED if initial SL state is uncertain.")
            # Do NOT clear the initial SL marker here. Do not set TSL marker.
            termux_notify("TSL Failed!", f"{symbol} TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position continues with initial SL (if successfully set) or may be UNPROTECTED. MANUAL INTERVENTION ADVISED if initial SL state is uncertain.")
            termux_notify("TSL Failed!", f"{symbol} TSL activation failed (unexpected). Check logs/position.")

    else:
        # Profit threshold not met
        sl_status_log = f"({initial_sl_marker})" if initial_sl_marker else "(None!)"
        logger.debug(f"{position_side.upper()} profit ({profit:.4f if not profit.is_nan() else 'N/A'}) has not crossed TSL activation threshold ({activation_threshold_points:.4f}). Keeping initial SL {sl_status_log}.")

# --- Signal Generation ---

def generate_signals(df_last_candles: pd.DataFrame, indicators: Optional[Dict[str, Decimal]], equity: Optional[Decimal]) -> Dict[str, Union[bool, str]]:
    """Generates trading signals based on indicator conditions, using Decimal."""
    # global CONFIG # Accessing CONFIG is fine without global here.
    long_signal = False
    short_signal = False
    signal_reason = "No signal - Initial State / Data Pending"

    if not indicators:
        logger.warning("Cannot generate signals: indicators dictionary is missing or calculation failed.")
        return {"long": False, "short": False, "reason": "Indicators missing/failed"}
    if df_last_candles is None or df_last_candles.empty:
         logger.warning("Cannot generate signals: insufficient candle data.")
         return {"long": False, "short": False, "reason": "Insufficient candle data"}
    # Equity is used for logging context in the signal reason, not for signal generation logic itself, so it's not strictly required for signals, but useful info.

    try:
        # Get latest candle data (iloc[-1])
        latest = df_last_candles.iloc[-1]
        current_price_float = latest["close"]
        if pd.isna(current_price_float):
             logger.warning("Cannot generate signals: latest close price is NaN.")
             return {"long": False, "short": False, "reason": "Latest price is NaN"}
        current_price = Decimal(str(current_price_float))

        # Get previous candle close for ATR move check (iloc[-2])
        previous_close = Decimal("NaN")
        if len(df_last_candles) >= 2:
             prev_candle = df_last_candles.iloc[-2]
             previous_close_float = prev_candle["close"]
             if not pd.isna(previous_close_float):
                  previous_close = Decimal(str(previous_close_float))
             else:
                  logger.debug("Previous close price is NaN in second-to-last candle, ATR move filter will be skipped.")
        else:
             logger.debug("Not enough candles (<2) for previous close, ATR move filter will be skipped.")


        if current_price <= Decimal(0):
             logger.warning("Cannot generate signals: current price is zero or negative.")
             return {"long": False, "short": False, "reason": "Invalid price (<= 0)"}

        # Use .get with default Decimal('NaN') or False to handle missing/failed indicators gracefully
        # Check if indicators is not None first
        k = indicators.get('stoch_k', Decimal('NaN')) if indicators else Decimal('NaN')
        d = indicators.get('stoch_d', Decimal('NaN')) if indicators else Decimal('NaN') # Keep d for context in reason string
        fast_ema = indicators.get('fast_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        slow_ema = indicators.get('slow_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        trend_ema = indicators.get('trend_ema', Decimal('NaN')) if indicators else Decimal('NaN')
        atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')
        stoch_kd_bullish = indicators.get('stoch_kd_bullish', False) if indicators else False # Already calculated
        stoch_kd_bearish = indicators.get('stoch_kd_bearish', False) if indicators else False # Already calculated


        # Check if any *required* indicator is NaN (should be caught in calculate_indicators, but defensive)
        # Only check if indicators is not None, as that's already a failure case
        if indicators:
             required_indicators_vals = {'fast_ema': fast_ema, 'slow_ema': slow_ema, 'trend_ema': trend_ema, 'atr': atr, 'stoch_k': k}
             nan_indicators = [name for name, val in required_indicators_vals.items() if isinstance(val, Decimal) and val.is_nan()]
             if nan_indicators:
                  # This should ideally be caught by calculate_indicators returning None, but defensive
                  logger.warning(f"Cannot generate signals: Required indicator(s) are NaN after calculation: {', '.join(nan_indicators)}")
                  return {"long": False, "short": False, "reason": f"NaN indicator(s): {', '.join(nan_indicators)}"}


        # Define conditions using Decimal comparisons for precision and CONFIG thresholds
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema
        ema_aligned = fast_ema == slow_ema # For logging

        # Loosened Trend Filter (price within +/- CONFIG.trend_filter_buffer_percent% of Trend EMA)
        # Calculate buffer points precisely
        # Avoid division by zero if trend_ema is zero (unlikely but possible)
        trend_buffer_points = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal('100')) if trend_ema.copy_abs() > Decimal('0') else Decimal('0')

        price_above_trend_loosened = current_price > trend_ema - trend_buffer_points
        price_below_trend_loosened = current_price < trend_ema + trend_buffer_points
        price_strictly_above_trend = current_price > trend_ema # For logging reason
        price_strictly_below_trend = current_price < trend_ema # For logging reason
        price_at_trend = current_price >= trend_ema - trend_buffer_points and current_price <= trend_ema + trend_buffer_points


        # Combined Stochastic Condition: Oversold OR bullish K/D cross (in/from oversold zone - checked in calculate_indicators)
        stoch_long_condition = (not k.is_nan() and k < CONFIG.stoch_oversold_threshold) or stoch_kd_bullish # Check k is not NaN
        # Combined Stochastic Condition: Overbought OR bearish K/D cross (in/from overbought zone - checked in calculate_indicators)
        stoch_short_condition = (not k.is_nan() and k > CONFIG.stoch_overbought_threshold) or stoch_kd_bearish # Check k is not NaN
        stoch_neutral = not stoch_long_condition and not stoch_short_condition # For logging


        # ATR Filter: Check if the price move from the previous close is significant
        is_significant_move = True # Assume true if filter multiplier is 0 or filter cannot be calculated
        atr_move_check_reason_part = ""

        if CONFIG.atr_move_filter_multiplier > Decimal('0'):
             if not previous_close.is_nan() and atr > Decimal('0'):
                  price_move_points = (current_price - previous_close).copy_abs()
                  atr_move_threshold_points = atr * CONFIG.atr_move_filter_multiplier
                  is_significant_move = price_move_points > atr_move_threshold_points
                  atr_move_check_reason_part = f"Move({price_move_points.normalize()}) {' > ' if is_significant_move else '<= '} {CONFIG.atr_move_filter_multiplier.normalize()}xATR({atr_move_threshold_points.normalize()})"
                  logger.debug(f"ATR Move Filter: {atr_move_check_reason_part}. Significant: {is_significant_move}")
             elif previous_close.is_nan():
                  is_significant_move = False # Cannot apply filter if prev close unknown
                  atr_move_check_reason_part = "Move Filter Skipped (Need >=2 candles)"
                  logger.debug(atr_move_check_reason_part)
             else: # atr is 0 or NaN
                  is_significant_move = False # Cannot apply filter if ATR is bad
                  atr_move_check_reason_part = f"Move Filter Skipped (Invalid ATR: {atr.normalize()})"
                  logger.debug(atr_move_check_reason_part)
        else:
             atr_move_check_reason_part = "Move Filter OFF"
             logger.debug(atr_move_check_reason_part)


        # --- Signal Logic ---
        # Combine all conditions
        potential_long = ema_bullish_cross and stoch_long_condition and is_significant_move
        potential_short = ema_bearish_cross and stoch_short_condition and is_significant_move

        # Apply trend filter if enabled
        if potential_long:
            if CONFIG.trade_only_with_trend:
                if price_above_trend_loosened:
                    long_signal = True
                    # Detailed reason string
                    reason_parts = [
                        f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                        f"Stoch({CONFIG.stoch_period}) Long ({k.normalize()}{'<' if not k.is_nan() and k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' or KD Cross' if stoch_kd_bullish else ''})", # Use normalize for K
                        f"Price({current_price.normalize()}) > Trend({trend_ema.normalize()})-{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer", # Use normalize for prices/buffer
                        atr_move_check_reason_part
                    ]
                    signal_reason = "Long Signal: " + " | ".join(reason_parts)
                else:
                    # Log detailed rejection reason if trend filter is ON
                    trend_reason_part = f"Price({current_price.normalize()}) !> Trend({trend_ema.normalize()})-{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer" # Use normalize
                    reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                         f"Stoch({CONFIG.stoch_period}) Long ({k.normalize()}{'<' if not k.is_nan() and k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' or KD Cross' if stoch_kd_bullish else ''})", # Use normalize for K
                         trend_reason_part,
                         atr_move_check_reason_part # Still include ATR reason
                    ]
                    signal_reason = "Long Blocked (Trend Filter ON): " + " | ".join(reason_parts)


            else: # Trend filter off
                long_signal = True
                reason_parts = [
                    f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                    f"Stoch({CONFIG.stoch_period}) Long ({k.normalize()}{'<' if not k.is_nan() and k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' or KD Cross' if stoch_kd_bullish else ''})", # Use normalize for K
                    atr_move_check_reason_part
                ]
                signal_reason = "Long Signal (Trend Filter OFF): " + " | ".join(reason_parts)

        elif potential_short:
             if CONFIG.trade_only_with_trend:
                 if price_below_trend_loosened:
                     short_signal = True
                     # Detailed reason string
                     reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                         f"Stoch({CONFIG.stoch_period}) Short ({k.normalize()}{'>' if not k.is_nan() and k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' or KD Cross' if stoch_kd_bearish else ''})", # Use normalize for K
                         f"Price({current_price.normalize()}) < Trend({trend_ema.normalize()})+{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer", # Use normalize for prices/buffer
                         atr_move_check_reason_part
                     ]
                     signal_reason = "Short Signal: " + " | ".join(reason_parts)
                 else:
                     # Log detailed rejection reason if trend filter is ON
                     trend_reason_part = f"Price({current_price.normalize()}) !< Trend({trend_ema.normalize()})+{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer" # Use normalize
                     reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                         f"Stoch({CONFIG.stoch_period}) Short ({k.normalize()}{'>' if not k.is_nan() and k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' or KD Cross' if stoch_kd_bearish else ''})", # Use normalize for K
                         trend_reason_part,
                         atr_move_check_reason_part # Still include ATR reason
                     ]
                     signal_reason = "Short Blocked (Trend Filter ON): " + " | ".join(reason_parts)
             else: # Trend filter off
                 short_signal = True
                 reason_parts = [
                    f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                    f"Stoch({CONFIG.stoch_period}) Short ({k.normalize()}{'>' if not k.is_nan() and k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' or KD Cross' if stoch_kd_bearish else ''})", # Use normalize for K
                    atr_move_check_reason_part
                 ]
                 signal_reason = "Short Signal (Trend Filter OFF): " + " | ".join(reason_parts)

        else:
             # No signal - build detailed reason why
             reason_parts = []
             # Check each major condition group
             if ema_bullish_cross: reason_parts.append(f"EMA Bullish")
             elif ema_bearish_cross: reason_parts.append(f"EMA Bearish")
             else: reason_parts.append(f"EMA Neutral") # Changed from Aligned to Neutral

             if stoch_long_condition: reason_parts.append(f"Stoch Long Met ({k.normalize()})") # Use normalize
             elif stoch_short_condition: reason_parts.append(f"Stoch Short Met ({k.normalize()})") # Use normalize
             elif not k.is_nan(): reason_parts.append(f"Stoch Neutral ({k.normalize()})") # Use normalize
             else: reason_parts.append(f"Stoch N/A ({k.normalize()})") # Handle NaN Stoch K

             if not is_significant_move: reason_parts.append(atr_move_check_reason_part) # Add ATR reason if it failed

             # Add trend filter status if enabled
             if CONFIG.trade_only_with_trend:
                 if price_strictly_above_trend: trend_status_part = "Price Above Trend"
                 elif price_strictly_below_trend: trend_status_part = "Price Below Trend"
                 elif price_at_trend: trend_status_part = "Price At Trend"
                 else: trend_status_part = "Price vs Trend N/A"
                 reason_parts.append(f"Trend Filter ON ({trend_status_part})")
             else:
                 reason_parts.append("Trend Filter OFF")


             signal_reason = "No Signal: " + " | ".join(reason_parts)


        # Log the outcome
        if long_signal or short_signal:
             logger.info(Fore.GREEN + Style.BRIGHT + f"Signal Check: {signal_reason}")
        else:
             # Log reason for no signal at debug level unless blocked by trend filter
             if "Blocked" in signal_reason:
                 logger.info(Fore.YELLOW + f"Signal Check: {signal_reason}")
             else:
                 logger.debug(f"Signal Check: {signal_reason}")

    except Exception as e:
        logger.error(f"{Fore.RED}Error generating signals: {e}", exc_info=True)
        return {"long": False, "short": False, "reason": f"Exception generating signals: {e}"}

    return {"long": long_signal, "short": short_signal, "reason": signal_reason}

# --- Status Display ---

def print_status_panel(
    cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Decimal]],
    positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, Union[bool, str]],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]] # Pass tracker state snapshot explicitly
) -> None:
    """Displays the current state using a mystical status panel with Decimal precision."""
    # global CONFIG, MARKET_INFO # Accessing CONFIG and MARKET_INFO is fine without global here.

    header_color = Fore.MAGENTA + Style.BRIGHT
    section_color = Fore.CYAN
    value_color = Fore.WHITE
    reset_all = Style.RESET_ALL

    print(header_color + "\n" + "=" * 80)
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else f"{Fore.YELLOW}N/A"
    print(f" Cycle: {value_color}{cycle}{header_color} | Timestamp: {value_color}{ts_str}")
    settle_curr = MARKET_INFO.get('settle', 'Quote') if MARKET_INFO else 'Quote'
    equity_str = f"{equity.normalize()} {settle_curr}" if equity is not None and not equity.is_nan() else f"{Fore.YELLOW}N/A"
    print(f" Equity: {Fore.GREEN}{equity_str}" + reset_all)
    print(header_color + "-" * 80)

    # --- Market & Indicators ---
    # Use .get(..., Decimal('NaN')) for safe access to indicator values, default to None/NaN if indicators is None
    price_str = f"{price.normalize()}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')
    atr_str = f"{atr.normalize()}" if not atr.is_nan() else f"{Fore.YELLOW}N/A"
    trend_ema = indicators.get('trend_ema', Decimal('NaN')) if indicators else Decimal('NaN')
    trend_ema_str = f"{trend_ema.normalize()}" if not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and not trend_ema.is_nan():
        # Calculate buffer points precisely
        trend_buffer_points = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal('100')) if trend_ema.copy_abs() > Decimal('0') else Decimal('0')
        if price > trend_ema + trend_buffer_points: price_color = Fore.GREEN; trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema - trend_buffer_points: price_color = Fore.RED; trend_desc = f"{price_color}(Below Trend)"
        else: price_color = Fore.YELLOW; trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get('stoch_k', Decimal('NaN')) if indicators else Decimal('NaN')
    stoch_d = indicators.get('stoch_d', Decimal('NaN')) if indicators else Decimal('NaN')
    stoch_k_str = f"{stoch_k.normalize()}" if not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_d_str = f"{stoch_d.normalize()}" if not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_color = Fore.YELLOW
    stoch_desc = f"{Fore.YELLOW}Stoch N/A"
    if not stoch_k.is_nan():
         if stoch_k < CONFIG.stoch_oversold_threshold: stoch_color = Fore.GREEN; stoch_desc = f"{stoch_color}Oversold (<{CONFIG.stoch_oversold_threshold.normalize()})"
         elif stoch_k > CONFIG.stoch_overbought_threshold: stoch_color = Fore.RED; stoch_desc = f"{stoch_color}Overbought (>{CONFIG.stoch_overbought_threshold.normalize()})"
         else: stoch_color = Fore.YELLOW; stoch_desc = f"{stoch_color}Neutral ({CONFIG.stoch_oversold_threshold.normalize()}-{CONFIG.stoch_overbought_threshold.normalize()})"

    fast_ema = indicators.get('fast_ema', Decimal('NaN')) if indicators else Decimal('NaN')
    slow_ema = indicators.get('slow_ema', Decimal('NaN')) if indicators else Decimal('NaN')
    fast_ema_str = f"{fast_ema.normalize()}" if not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
    slow_ema_str = f"{slow_ema.normalize()}" if not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
    ema_cross_color = Fore.WHITE
    ema_desc = f"{Fore.YELLOW}EMA N/A"
    if not fast_ema.is_nan() and not slow_ema.is_nan():
        if fast_ema > slow_ema: ema_cross_color = Fore.GREEN; ema_desc = f"{ema_cross_color}Bullish"
        elif fast_ema < slow_ema: ema_cross_color = Fore.RED; ema_desc = f"{ema_cross_color}Bearish"
        else: ema_cross_color = Fore.YELLOW; ema_desc = f"{Fore.YELLOW}Neutral" # Changed from Aligned to Neutral

    status_data = [
        [section_color + "Market", value_color + CONFIG.symbol, f"{price_color}{price_str}{value_color}"], # Ensure value_color resets after price
        [section_color + f"Trend EMA ({CONFIG.trend_ema_period})", f"{value_color}{trend_ema_str}{value_color}", trend_desc + reset_all], # Ensure reset_all after colored desc
        [section_color + f"ATR ({CONFIG.atr_period})", f"{value_color}{atr_str}{value_color}", ""], # Display ATR period from CONFIG
        [section_color + f"EMA Fast/Slow ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})", f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}{value_color}", ema_desc + reset_all], # Display EMA periods from CONFIG
        [section_color + f"Stoch %K/%D ({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d})", f"{stoch_color}{stoch_k_str} / {stoch_d_str}{value_color}", stoch_desc + reset_all], # Display Stoch periods from CONFIG
    ]
    print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "left", "left")))

    # --- Positions & Orders ---
    pos_avail = positions is not None
    long_pos = positions.get('long', {}) if pos_avail else {}
    short_pos = positions.get('short', {}) if pos_avail else {}

    # Safely get values, handling None or NaN Decimals
    long_qty = long_pos.get('qty', Decimal("0.0"))
    short_qty = short_pos.get('qty', Decimal("0.0"))
    long_entry = long_pos.get('entry_price', Decimal("NaN"))
    short_entry = short_pos.get('entry_price', Decimal("NaN"))
    long_pnl = long_pos.get('pnl', Decimal("NaN"))
    short_pnl = short_pos.get('pnl', Decimal("NaN"))
    long_liq = long_pos.get('liq_price', Decimal("NaN"))
    short_liq = short_pos.get('liq_price', Decimal("NaN"))

    # Use the passed tracker state snapshot
    long_sl_marker = order_tracker_state.get('long', {}).get('sl_id')
    long_tsl_marker = order_tracker_state.get('long', {}).get('tsl_id')
    short_sl_marker = order_tracker_state.get('short', {}).get('sl_id')
    short_tsl_marker = order_tracker_state.get('short', {}).get('tsl_id')


    # Determine SL/TSL status strings based on tracker and position existence
    def get_stop_status(sl_marker, tsl_marker, has_pos):
        if not has_pos:
             return f"{value_color}-" # No position, display dash
        if tsl_marker:
            # Show last few chars if it's a long ID string, otherwise show marker
            display_marker = tsl_marker if len(str(tsl_marker)) < 10 else f"...{str(tsl_marker)[-6:]}"
            return f"{Fore.GREEN}TSL Active ({display_marker})"
        elif sl_marker:
            # Show last few chars if it's a long ID string, otherwise show marker
            display_marker = sl_marker if len(str(sl_marker)) < 10 else f"...{str(sl_marker)[-6:]}"
            return f"{Fore.YELLOW}SL Active ({display_marker})"
        else:
            # No marker found in tracker, but position exists
            return f"{Fore.RED}{Style.BRIGHT}NONE (!)" # Highlight missing stop

    has_long_pos_panel = long_qty.copy_abs() >= CONFIG.position_qty_epsilon
    has_short_pos_panel = short_qty.copy_abs() >= CONFIG.position_qty_epsilon
    long_stop_status = get_stop_status(long_sl_marker, long_tsl_marker, has_long_pos_panel)
    short_stop_status = get_stop_status(short_sl_marker, short_tsl_marker, has_short_pos_panel)

    # Format position details, handle potential None or NaN from failed fetch/parsing
    if not pos_avail:
        long_qty_str, short_qty_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_entry_str, short_entry_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_pnl_str, short_pnl_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
        long_liq_str, short_liq_str = f"{Fore.YELLOW}N/A", f"{Fore.YELLOW}N/A"
    else:
        # Format Decimals nicely, remove trailing zeros for quantity (more readable)
        long_qty_str = f"{long_qty.normalize()}" if has_long_pos_panel else "0"
        short_qty_str = f"{short_qty.normalize()}" if has_short_pos_panel else "0"

        long_entry_str = f"{long_entry.normalize()}" if has_long_pos_panel and not long_entry.is_nan() else "-" # Use normalize for cleaner price display
        short_entry_str = f"{short_entry.normalize()}" if has_short_pos_panel and not short_entry.is_nan() else "-" # Use normalize

        # PnL color based on value, only display if position exists and PnL is not NaN
        long_pnl_color = Fore.GREEN if not long_pnl.is_nan() and long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if not short_pnl.is_nan() and short_pnl >= 0 else Fore.RED
        long_pnl_str = f"{long_pnl_color}{long_pnl:+.4f}{value_color}" if has_long_pos_panel and not long_pnl.is_nan() else "-"
        short_pnl_str = f"{short_pnl_color}{short_pnl:+.4f}{value_color}" if has_short_pos_panel and not short_pnl.is_nan() else "-"

        # Liq price color (usually red), only display if position exists and liq is not NaN and > 0
        long_liq_str = f"{Fore.RED}{long_liq.normalize()}{value_color}" if has_long_pos_panel and not long_liq.is_nan() and long_liq > Decimal('0') else "-" # Use normalize, check > 0
        short_liq_str = f"{Fore.RED}{short_liq.normalize()}{value_color}" if has_short_pos_panel and not short_liq.is_nan() and short_liq > Decimal('0') else "-" # Use normalize, check > 0


    position_data = [
        [section_color + "Status", Fore.GREEN + "LONG" + reset_all, Fore.RED + "SHORT" + reset_all], # Ensure headers are reset
        [section_color + "Quantity", f"{value_color}{long_qty_str}{value_color}", f"{value_color}{short_qty_str}{value_color}"],
        [section_color + "Entry Price", f"{value_color}{long_entry_str}{value_color}", f"{value_color}{short_entry_str}{value_color}"],
        [section_color + "Unrealized PnL", long_pnl_str + reset_all, short_pnl_str + reset_all], # Ensure reset after colored PnL
        [section_color + "Liq. Price (Est.)", long_liq_str + reset_all, short_liq_str + reset_all], # Ensure reset after colored Liq
        [section_color + "Active Stop", long_stop_status + reset_all, short_stop_status + reset_all], # Ensure reset after colored stop status
    ]
    print(tabulate(position_data, headers="firstrow", tablefmt="fancy_grid", colalign=("left", "left", "left")))

    # --- Signals ---
    long_signal_status = signals.get('long', False)
    short_signal_status = signals.get('short', False)
    long_signal_color = Fore.GREEN + Style.BRIGHT if long_signal_status else Fore.WHITE
    short_signal_color = Fore.RED + Style.BRIGHT if short_signal_status else Fore.WHITE
    trend_status = f"(Trend Filter: {value_color}{'ON' if CONFIG.trade_only_with_trend else 'OFF'}{header_color})"
    signal_reason_text = signals.get('reason', 'N/A')
    print(f" Signals {trend_status}: Long [{long_signal_color}{str(long_signal_status).upper():<5}{header_color}] | Short [{short_signal_color}{str(short_signal_status).upper():<5}{header_color}]") # Use .upper() for bool string

    # Display the signal reason below, wrapping if necessary
    reason_prefix = " Reason: "
    max_line_width = 80 # Approximate terminal width
    # Use textwrap to handle wrapping properly
    # textwrap.fill adds the initial prefix and subsequent indent
    wrapped_reason = textwrap.fill(
        signal_reason_text,
        width=max_line_width, # Wrap to total line width
        initial_indent=f"{Fore.YELLOW}{reason_prefix}", # Add color and prefix to first line
        subsequent_indent=" " * len(reason_prefix), # Indent subsequent lines
        replace_whitespace=False # Don't replace whitespace within the text
    )
    print(wrapped_reason + reset_all) # Ensure color is reset after the wrapped text

    print(header_color + "=" * 80 + reset_all)

# --- Main Trading Cycle & Loop ---

def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell with enhanced precision and logic."""
    # global CONFIG, EXCHANGE, MARKET_INFO, order_tracker # Need global for ASSIGNING to order_tracker in manage_trailing_stop (indirectly via function call) and potentially in place_risked_market_order
    # Let's be explicit where assignment happens or where state is critical.
    global order_tracker # order_tracker is modified in this function scope via function calls

    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.time()
    cycle_success = True # Track if cycle completes without critical errors that prevent status display

    # 1. Fetch Market Data
    df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
    if df is None or df.empty:
        logger.error(Fore.RED + "Halting cycle: Market data fetch failed or returned empty.")
        cycle_success = False
        # No status panel if no data to derive price/timestamp from
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} ABORTED due to Data Fetch Failure (Duration: {end_time - start_time:.2f}s) ---")
        return # Skip cycle

    # 2. Get Current Price & Timestamp from Data (and previous close for ATR filter)
    current_price: Optional[Decimal] = None
    last_timestamp: Optional[pd.Timestamp] = None
    try:
        if df.empty:
             raise ValueError("DataFrame is empty after fetch/processing.")

        last_candle = df.iloc[-1]
        current_price_float = last_candle["close"]
        if pd.isna(current_price_float):
             raise ValueError("Latest close price is NaN")
        current_price = Decimal(str(current_price_float))
        last_timestamp = df.index[-1] # Already UTC from fetch_market_data
        logger.debug(f"Latest candle: Time={last_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}, Close={current_price.normalize()}") # Use normalize

        # Check for stale data
        now_utc = pd.Timestamp.utcnow().tz_localize('UTC') # Ensure now_utc is timezone-aware UTC
        time_diff = now_utc - last_timestamp
        try:
             interval_seconds = EXCHANGE.parse_timeframe(CONFIG.interval)
             allowed_lag = pd.Timedelta(seconds=interval_seconds * 1.5 + 60) # Allow 1.5 intervals + 60s buffer
             if time_diff > allowed_lag:
                  logger.warning(Fore.YELLOW + f"Market data may be stale. Last candle: {last_timestamp.strftime('%H:%M:%S')} UTC ({time_diff} ago). Allowed lag: ~{allowed_lag}")
        except ValueError:
            logger.warning("Could not parse interval to check data staleness.")

    except (IndexError, KeyError, ValueError, InvalidOperation, TypeError) as e:
        logger.error(Fore.RED + f"Halting cycle: Failed to get/process current price/timestamp from DataFrame: {e}", exc_info=True)
        cycle_success = False
        # No status panel if price invalid
        end_time = time.time()
        logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} ABORTED due to Price Processing Error (Duration: {end_time - start_time:.2f}s) ---")
        return # Skip cycle

    # 3. Calculate Indicators (returns Decimals)
    indicators = calculate_indicators(df)
    if indicators is None:
        logger.error(Fore.RED + "Indicator calculation failed. Continuing cycle but skipping trade actions.")
        cycle_success = False # Mark as failed for logging, but continue to fetch state and show panel

    current_atr = indicators.get('atr', Decimal('NaN')) if indicators else Decimal('NaN')

    # 4. Get Current State (Balance & Positions as Decimals)
    quote_currency = MARKET_INFO.get('settle', 'USDT')
    free_balance, current_equity = get_balance(quote_currency)
    if current_equity is None or current_equity.is_nan() or current_equity <= Decimal("0"):
        logger.error(Fore.RED + "Failed to fetch valid current balance/equity. Cannot perform risk calculation or trading actions.")
        cycle_success = False
        # Allow falling through to display panel

    # Fetch positions (crucial state)
    positions = get_current_position(CONFIG.symbol)
    if positions is None:
        logger.error(Fore.RED + "Failed to fetch current positions. Cannot manage state or trade.")
        cycle_success = False
        # Allow falling through to display panel

    # --- Capture State Snapshot for Status Panel & Logic ---
    # Do this *before* potentially modifying state (like TSL management or entry)
    # Need to use global keyword as we are accessing the global order_tracker
    global order_tracker
    order_tracker_snapshot = copy.deepcopy(order_tracker)
    positions_snapshot = positions if positions is not None else {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "pnl": Decimal("NaN"), "liq_price": Decimal("NaN")}
    }

    # --- Initialize signals dictionary ---
    signals: Dict[str, Union[bool, str]] = {"long": False, "short": False, "reason": "Skipped: Initial State"}

    # --- Logic continues only if critical data is available (positions and equity) ---
    # Re-evaluate can_trade_logic after fetching current_equity and positions
    can_trade_logic = (
        positions is not None and
        current_equity is not None and
        not current_equity.is_nan() and
        current_equity > Decimal('0')
    )


    if not can_trade_logic:
         signals["reason"] = "Skipped: Critical data missing (Equity/Position fetch failed)"
         logger.warning(signals["reason"])
    else:
        # Use the *current* state from `positions` dict (not snapshot) for logic decisions
        active_long_pos = positions.get('long', {})
        active_short_pos = positions.get('short', {})
        active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
        active_short_qty = active_short_pos.get('qty', Decimal('0.0'))

        # Check if already have a significant position in either direction
        has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
        has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
        is_flat = not has_long_pos and not has_short_pos

        # 5. Manage Trailing Stops
        # Only attempt TSL management if indicators and current price are available, AND we have a position
        if (has_long_pos or has_short_pos) and indicators is not None and not current_price.is_nan() and not current_atr.is_nan():
             if has_long_pos:
                 logger.debug("Managing TSL for existing LONG position...")
                 # manage_trailing_stop modifies order_tracker, needs global
                 manage_trailing_stop(CONFIG.symbol, "long", active_long_qty, active_long_pos.get('entry_price', Decimal('NaN')), current_price, current_atr)
             elif has_short_pos:
                 logger.debug("Managing TSL for existing SHORT position...")
                 # manage_trailing_stop modifies order_tracker, needs global
                 manage_trailing_stop(CONFIG.symbol, "short", active_short_qty, active_short_pos.get('entry_price', Decimal('NaN')), current_price, current_atr)
        else:
             # If flat, ensure trackers are clear (belt-and-suspenders check)
             # Need to use global keyword as we are accessing the global order_tracker
             global order_tracker
             if is_flat and (order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]):
                 logger.info("Position is flat, ensuring order trackers are cleared.")
                 order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                 order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                 # Update the snapshot to reflect the clearing for the panel display
                 order_tracker_snapshot["long"] = {"sl_id": None, "tsl_id": None}
                 order_tracker_snapshot["short"] = {"sl_id": None, "tsl_id": None}

             if indicators is None or current_price.is_nan() or current_atr.is_nan():
                  if (has_long_pos or has_short_pos):
                      logger.warning("Skipping TSL management for open position due to missing indicators, invalid price, or invalid ATR.")


        # --- IMPORTANT: Re-fetch position state AFTER TSL management ---
        # This checks if the position was closed by the TSL hitting during the previous steps.
        logger.debug("Re-fetching position state after TSL management...")
        positions_after_tsl = get_current_position(CONFIG.symbol)
        if positions_after_tsl is None:
             logger.error(Fore.RED + "Failed to re-fetch positions after TSL check. Cannot safely determine position state for entry.")
             cycle_success = False
             signals["reason"] = "Skipped: Position re-fetch failed"
        else:
             # Update active quantities and flat status based on the re-fetched state
             active_long_pos = positions_after_tsl.get('long', {})
             active_short_pos = positions_after_tsl.get('short', {})
             active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
             active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
             has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
             has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
             is_flat = not has_long_pos and not has_short_pos
             logger.debug(f"Position Status After TSL Check: Flat = {is_flat} (Long Qty: {active_long_qty.normalize()}, Short Qty: {active_short_qty.normalize()})")

             # Clear trackers if now flat (e.g., TSL hit)
             # Need to use global keyword as we are accessing the global order_tracker
             global order_tracker
             if is_flat and (order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]):
                  logger.info("Position became flat (likely TSL hit), clearing order trackers.")
                  order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                  order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                  # Update snapshot too
                  order_tracker_snapshot["long"] = {"sl_id": None, "tsl_id": None}
                  order_tracker_snapshot["short"] = {"sl_id": None, "tsl_id": None}


             # 6. Generate Trading Signals
             # Signals only generated if indicators and current price are available, AND we have enough data for the ATR filter (unless filter is off)
             enough_candles_for_atr_filter = len(df) >= 2
             atr_filter_enabled = CONFIG.atr_move_filter_multiplier > Decimal('0')
             can_generate_signals = indicators is not None and not current_price.is_nan() and (enough_candles_for_atr_filter or not atr_filter_enabled)

             if can_generate_signals:
                 # Pass last 2 candles for ATR filter calculation if enough data, otherwise pass all available (1 or more)
                 df_for_signals = df.iloc[-2:] if enough_candles_for_atr_filter else df.iloc[-1:]
                 signals_data = generate_signals(df_for_signals, indicators, current_equity)
                 signals = {"long": signals_data["long"], "short": signals_data["short"], "reason": signals_data["reason"]} # Keep reason
             else:
                 reason_str = "Skipped: "
                 if indicators is None: reason_str += "Indicators missing. "
                 if current_price is None or current_price.is_nan(): reason_str += "Current price NaN. "
                 if atr_filter_enabled and not enough_candles_for_atr_filter: reason_str += f"Need >=2 candles for ATR filter ({len(df)} found). "
                 signals = {"long": False, "short": False, "reason": reason_str.strip()}
                 logger.warning(f"Skipping signal generation: {signals['reason']}")


             # 7. Execute Trades based on Signals
             # Only attempt entry if now flat, indicators/ATR are available, equity is sufficient, AND signals were successfully generated.
             if is_flat and can_generate_signals and "Skipped" not in signals.get("reason","") and indicators is not None and not current_atr.is_nan(): # Equity checked in can_trade_logic
                 trade_attempted = False
                 if signals.get("long"):
                     logger.info(Fore.GREEN + Style.BRIGHT + f"Long signal detected! {signals.get('reason', '')}. Attempting entry...")
                     trade_attempted = True
                     # place_risked_market_order modifies order_tracker, needs global
                     if place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr):
                          logger.info(Fore.GREEN + f"Long entry process completed successfully for cycle {cycle_count}.")
                     else:
                          logger.error(Fore.RED + f"Long entry process failed for cycle {cycle_count}.")
                          # Optional: Implement cooldown logic here if needed

                 elif signals.get("short"):
                     logger.info(Fore.RED + Style.BRIGHT + f"Short signal detected! {signals.get('reason', '')}. Attempting entry.")
                     trade_attempted = True
                     # place_risked_market_order modifies order_tracker, needs global
                     if place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr):
                          logger.info(Fore.GREEN + f"Short entry process completed successfully for cycle {cycle_count}.")
                     else:
                          logger.error(Fore.RED + f"Short entry process failed for cycle {cycle_count}.")
                          # Optional: Implement cooldown logic here if needed

                 # If a trade was attempted, main loop sleep handles the pause.

             elif not is_flat:
                  pos_side = "LONG" if has_long_pos else "SHORT"
                  logger.info(f"Position ({pos_side}) already open, skipping new entry signals.")
                  # Future: Add exit logic based on counter-signals or other conditions if desired.
                  # Example: if pos_side == "LONG" and signals.get("short"): close_position("long")
                  # Example: if pos_side == "SHORT" and signals.get("long"): close_position("short")
             # ELSE: is_flat but conditions for entry not met (e.g., signal generation skipped)


    # 8. Display Status Panel (Always display if data allows)
    # Use the state captured *before* TSL management and potential trade execution for consistency
    # Ensure positions_after_tsl is used for the panel if available, as it's the most up-to-date state before panel display
    final_positions_for_panel = positions_after_tsl if 'positions_after_tsl' in locals() and positions_after_tsl is not None else positions_snapshot

    print_status_panel(
        cycle_count, last_timestamp, current_price, indicators,
        final_positions_for_panel, current_equity, signals, order_tracker_snapshot # Use the snapshots/final state
    )

    end_time = time.time()
    status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"
    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")

def graceful_shutdown() -> None:
    """Dispels active orders and closes open positions gracefully with precision."""
    logger.warning(Fore.YELLOW + Style.BRIGHT + "\nInitiating Graceful Shutdown Sequence...")
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    # global EXCHANGE, MARKET_INFO, order_tracker, CONFIG # Need global for ASSIGNING to order_tracker
    global order_tracker # order_tracker is modified here

    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(Fore.RED + "Exchange object or Market Info not available. Cannot perform clean shutdown.")
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Cannot perform clean shutdown - Exchange not ready.")
        return

    symbol = CONFIG.symbol
    market_id = MARKET_INFO.get('id') # Exchange specific ID
    quote_currency = MARKET_INFO.get('settle', 'USDT') # Use settle currency

    # 1. Cancel All Open Orders for the Symbol
    # This primarily targets limit orders if any were used. Position-based SL/TP are not separate orders.
    try:
        logger.info(Fore.CYAN + f"Dispelling all cancellable open orders for {symbol}...")
        # fetch_with_retries handles category param
        # Fetch open orders first to log IDs (best effort)
        open_orders_list = []
        try:
            # Bybit V5 fetch_open_orders requires category and symbol (market ID)
            fetch_params = {'category': CONFIG.market_type, 'symbol': market_id}
            # fetch_open_orders expects the CCXT unified symbol string, not market ID, for the first argument
            open_orders_list = fetch_with_retries(EXCHANGE.fetch_open_orders, symbol, params=fetch_params)
            if open_orders_list is None:
                 logger.warning(Fore.YELLOW + "Fetching open orders failed, cannot list orders to be cancelled. Proceeding with cancel all if available.")
                 open_orders_list = [] # Treat as empty list to proceed
            elif open_orders_list:
                 order_ids = [o.get('id', 'N/A') for o in open_orders_list]
                 logger.info(f"Found {len(open_orders_list)} open orders to attempt cancellation: {', '.join(order_ids)}")
            else:
                 logger.info("No cancellable open orders found via fetch_open_orders.")
        except Exception as fetch_err:
             logger.warning(Fore.YELLOW + f"Could not fetch open orders before cancelling: {fetch_err}. Proceeding with cancel all if available.", exc_info=True)
             open_orders_list = [] # Ensure it's a list


        # Attempt to cancel using Bybit V5 cancel-all endpoint first
        cancel_all_successful = False
        if hasattr(EXCHANGE, 'private_post_order_cancel_all'):
             try:
                 logger.info(f"Attempting cancel_all_orders for {symbol} via private_post_order_cancel_all...")
                 # Bybit V5 cancel_all requires category and symbol (market ID)
                 cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                 response = fetch_with_retries(EXCHANGE.private_post_order_cancel_all, params=cancel_params)

                 if response is None:
                      logger.warning(Fore.YELLOW + "Cancel all orders request failed after retries.")
                 elif isinstance(response, dict) and response.get('info', {}).get('retCode') == 0:
                      logger.info(Fore.GREEN + "Cancel all command successful (retCode 0).")
                      cancel_all_successful = True
                 else:
                      error_msg = response.get('info', {}).get('retMsg', 'Unknown error') if isinstance(response, dict) else str(response)
                      error_code = response.get('info', {}).get('retCode') if isinstance(response, dict) else 'N/A'
                      logger.warning(Fore.YELLOW + f"Cancel all orders command sent, success confirmation unclear or failed: {error_msg} (Code: {error_code}). MANUAL CHECK REQUIRED.")
             except Exception as cancel_all_err:
                  logger.warning(Fore.YELLOW + f"Exception during cancel_all_orders attempt: {cancel_all_err}. Falling back to individual cancels.", exc_info=True)

        if not cancel_all_successful and open_orders_list:
             # Fallback to individual cancellation if cancel_all failed or isn't used
             logger.info("Attempting to cancel orders individually...")
             cancelled_count = 0
             for order in open_orders_list:
                  try:
                       order_id = order.get('id')
                       if not order_id: continue # Skip if no ID
                       logger.debug(f"Cancelling order {order_id}...")
                       # Bybit V5 cancel_order requires category and symbol (market ID)
                       individual_cancel_params = {'category': CONFIG.market_type, 'symbol': MARKET_INFO['id']}
                       # cancel_order expects CCXT unified symbol string
                       fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=individual_cancel_params)
                       logger.info(f"Cancel request sent for order {order_id}.")
                       cancelled_count += 1
                       time.sleep(0.2) # Small delay between cancels to respect rate limits
                  except ccxt.OrderNotFound:
                       logger.warning(f"Order {order_id} already gone when attempting cancellation.")
                  except Exception as ind_cancel_err:
                       logger.error(f"Failed to cancel order {order_id}: {ind_cancel_err}", exc_info=True)
             logger.info(f"Attempted to cancel {cancelled_count}/{len(open_orders_list)} orders individually.")
        elif not cancel_all_successful and not open_orders_list:
             logger.info("Skipping individual order cancellation as no open orders were found.")


        # Clear local tracker regardless, as intent is to have no active tracked orders
        logger.info("Clearing local order tracker state.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        # Catch unexpected errors during the *entire cancellation block*
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during order cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.", exc_info=True)
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Order cancel error. Check logs.")


    # Add a small delay after cancelling orders before checking/closing positions
    logger.info("Waiting briefly after order cancellation before checking positions...")
    time.sleep(max(CONFIG.order_check_delay_seconds, 2)) # Wait at least 2 seconds

    # 2. Close Any Open Positions
    try:
        logger.info(Fore.CYAN + "Checking for lingering positions to close...")
        # Fetch final position state using the dedicated function with retries
        positions = get_current_position(symbol)

        closed_count = 0
        total_positions_to_close = 0

        if positions:
            try:
                 # Get minimum quantity for validation using Decimal
                 min_qty_dec = Decimal(str(MARKET_INFO['limits']['amount']['min']))
            except (KeyError, InvalidOperation, TypeError):
                 logger.warning("Could not determine minimum order quantity for closure validation.")
                 min_qty_dec = Decimal("0") # Assume zero if unavailable


            # Filter for positions with significant quantity
            fetched_positions_to_process = {}
            if positions is not None: # Check if positions fetch itself succeeded
                for side, pos_data in positions.items():
                     qty = pos_data.get('qty', Decimal("0.0"))
                     if qty is not None and qty.copy_abs() >= CONFIG.position_qty_epsilon: # Ensure qty is not None before abs
                          fetched_positions_to_process[side] = pos_data
                total_positions_to_close = len(fetched_positions_to_process)


            if not fetched_positions_to_process:
                 logger.info(Fore.GREEN + "No significant open positions found requiring closure.")
            else:
                logger.warning(Fore.YELLOW + f"Found {total_positions_to_close} positions requiring closure.")

                for side, pos_data in fetched_positions_to_process.items():
                     qty = pos_data.get('qty', Decimal("0.0"))
                     entry_price = pos_data.get('entry_price', Decimal("NaN"))
                     close_side = "sell" if side == "long" else "buy"
                     logger.warning(Fore.YELLOW + f"Closing {side.upper()} position (Qty: {qty.normalize()}, Entry: {entry_price.normalize() if not entry_price.is_nan() else 'N/A'}) with market order...") # Use normalize
                     try:
                         # Format quantity precisely for closure order (use absolute value and round down)
                         close_qty_str = format_amount(symbol, qty.copy_abs(), ROUND_DOWN)
                         try:
                             close_qty_decimal = Decimal(close_qty_str)
                         except InvalidOperation:
                              logger.critical(Fore.RED + f"Closure failed: Invalid Decimal after formatting close quantity '{close_qty_str}'. MANUAL CLOSURE REQUIRED for {side.upper()} position!")
                              termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS BAD QTY! Close manually!")
                              continue # Skip trying to close this position


                         # Validate against minimum quantity before attempting closure
                         if close_qty_decimal < min_qty_dec or close_qty_decimal.copy_abs() < CONFIG.position_qty_epsilon:
                              logger.critical(f"{Fore.RED}Closure quantity {close_qty_decimal.normalize()} for {side} position is below minimum {min_qty_dec.normalize()} or zero. MANUAL CLOSURE REQUIRED!")
                              termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS < MIN QTY! Close manually!")
                              continue # Skip trying to close this position

                         # Place the closure market order
                         close_params = {'reduceOnly': True, 'positionIdx': 0, 'category': CONFIG.market_type} # Crucial: Only close, don't open new position + one-way mode + category
                         # Use market ID for symbol parameter
                         close_order = fetch_with_retries(
                             EXCHANGE.create_market_order,
                             symbol=MARKET_INFO['id'], # Use market ID
                             side=close_side,
                             amount=float(close_qty_decimal), # CCXT needs float
                             params=close_params
                         )

                         # Check response for success (Bybit V5 retCode 0)
                         if close_order and (close_order.get('id') or (isinstance(close_order.get('info'), dict) and close_order['info'].get('retCode') == 0)):
                            close_id = close_order.get('id', 'N/A (retCode 0)')
                            logger.trade(Fore.GREEN + f"Position closure order placed successfully: ID {close_id}")
                            closed_count += 1
                            # Wait briefly to allow fill confirmation before checking next position (if any)
                            time.sleep(max(CONFIG.order_check_delay_seconds, 2))
                            # Optional: Verify closure order status? Might slow shutdown significantly.
                         else:
                            # Log critical error if closure order placement fails
                            error_msg = close_order.get('info', {}).get('retMsg', 'No ID and no success code.') if isinstance(close_order, dict) else str(close_order)
                            error_code = close_order.get('info', {}).get('retCode', 'N/A') if isinstance(close_order, dict) else 'N/A'
                            logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO PLACE closure order for {side} position ({qty.normalize()}): {error_msg} (Code: {error_code}). MANUAL INTERVENTION REQUIRED!")
                            termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")

                     except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"FAILED TO CLOSE {side} position ({qty.normalize()}): {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
                         termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")
                     except Exception as e:
                         logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error closing {side} position: {e}. MANUAL INTERVENTION REQUIRED!", exc_info=True)
                         termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Manual action!")


            # Final summary message
            if total_positions_to_close == 0: # Case where fetch found no significant positions
                 logger.info(Fore.GREEN + "No significant open positions found requiring closure.")
            elif closed_count == total_positions_to_close: # Case where all found positions were attempted to be closed
                 logger.info(Fore.GREEN + f"Successfully placed closure orders for all {closed_count} detected positions.")
            elif closed_count > 0: # Case where some were closed, but not all found
                 logger.warning(Fore.YELLOW + f"Placed closure orders for {closed_count} positions, but {total_positions_to_close - closed_count} positions may remain. MANUAL CHECK REQUIRED.")
                 termux_notify("Shutdown Warning!", f"{symbol} Manual check needed - {total_positions_to_close - closed_count} positions might remain.")
            else: # Case where positions were found but none were successfully attempted to be closed
                logger.warning(Fore.YELLOW + "Attempted shutdown but closure orders failed or were not possible for all open positions. MANUAL CHECK REQUIRED.")
                termux_notify("Shutdown Warning!", f"{symbol} Manual check needed - positions might remain.")


        elif positions is None:
             # Failure to fetch positions during shutdown is critical
             logger.critical(Fore.RED + Style.BRIGHT + "Could not fetch final positions during shutdown. MANUAL CHECK REQUIRED on exchange!")
             termux_notify("Shutdown Warning!", f"{symbol} Cannot confirm position status. Check exchange!")

    except Exception as e:
        logger.error(Fore.RED + Style.BRIGHT + f"Error during position closure phase: {e}. Manual check advised.", exc_info=True)
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Error during position closure. Check logs.")


    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful Shutdown Sequence Complete.")
    termux_notify("Shutdown Complete", f"{CONFIG.symbol} bot stopped.")

# --- Main Spell Invocation ---
if __name__ == "__main__":
    # Display Banner first
    print(Fore.CYAN + Style.BRIGHT + "# Summoning the Pyrmethus Banner..." + Style.RESET_ALL) # Corrected bot name in banner log
    print(Fore.RED + Style.BRIGHT + r"""
#  ██████╗ ██╗   ██╗██████╗ ███╗   ███╗███████╗ ██████╗ █████╗ ██╗     ██████╗
#  ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗
#  ██████╔╝ ╚████╔╝ ██████╔╝██╔████╔██║███████╗██║     ███████║██║     ██████╔╝
#  ██╔═══╝   ╚██╔╝  ██╔══██╗██║╚██╔╝██║╚════██║██║     ██╔══██║██║     ██╔═══╝
#  ██║        ██║   ██║  ██║██║ ╚═╝ ██║███████║╚██████╗██║  ██║███████╗██║
#  ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝
    """ + Style.RESET_ALL)
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Termux Trading Spell Activated (v2.2.1 Enhanced Precision & V5) ***" + Style.RESET_ALL) # Updated version
    print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL) # Separator with same background

    logger.info(f"Initializing Pyrmethus v2.2.1...") # Updated version log
    # Log level is set during initial logging configuration before config class
    logger.info(f"Log Level configured to: {log_level_str}")

    # CONFIG is already instantiated before this block

    # Log key configuration parameters for verification
    logger.info(f"--- Trading Configuration ---")
    logger.info(f"Symbol: {CONFIG.symbol} ({CONFIG.market_type.capitalize()})")
    logger.info(f"Timeframe: {CONFIG.interval}")
    # Use .normalize() to remove unnecessary trailing zeros for display
    logger.info(f"Risk per trade: {(CONFIG.risk_percentage * 100).normalize()}%")
    logger.info(f"SL Multiplier: {CONFIG.sl_atr_multiplier.normalize()}")
    logger.info(f"TSL Activation: {CONFIG.tsl_activation_atr_multiplier.normalize()} * ATR Profit")
    logger.info(f"TSL Trail Percent: {CONFIG.trailing_stop_percent.normalize()}%")
    logger.info(f"Trigger Prices: SL={CONFIG.sl_trigger_by}, TSL={CONFIG.tsl_trigger_by}")
    logger.info(f"Trend Filter EMA({CONFIG.trend_ema_period}): {CONFIG.trade_only_with_trend} (Buffer: {CONFIG.trend_filter_buffer_percent.normalize()}%)") # Added buffer to log
    logger.info(f"ATR Move Filter Multiplier: {CONFIG.atr_move_filter_multiplier.normalize()}x") # Added ATR filter multiplier to log
    logger.info(f"Indicator Periods: Fast EMA({CONFIG.fast_ema_period}), Slow EMA({CONFIG.slow_ema_period}), Trend EMA({CONFIG.trend_ema_period}), Stoch({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d}), ATR({CONFIG.atr_period})")
    logger.info(f"Signal Thresholds: Stoch OS(<{CONFIG.stoch_oversold_threshold.normalize()}), Stoch OB(>{CONFIG.stoch_overbought_threshold.normalize()})")
    logger.info(f"Position Quantity Epsilon: {CONFIG.position_qty_epsilon:.2E}") # Scientific notation
    logger.info(f"Loop Interval: {CONFIG.loop_sleep_seconds}s")
    logger.info(f"OHLCV Limit: {CONFIG.ohlcv_limit}")
    logger.info(f"Fetch Retries: {CONFIG.max_fetch_retries}")
    logger.info(f"Order Check Timeout: {CONFIG.order_check_timeout_seconds}s")
    logger.info(f"Journaling Enabled: {CONFIG.enable_journaling} (File: {CONFIG.journal_file_path})")
    logger.info(f"-----------------------------")


    # Final check if exchange connection and market info loading succeeded
    if MARKET_INFO and EXCHANGE:
         termux_notify("Bot Started", f"Monitoring {CONFIG.symbol} (v2.2.1)") # Updated version notif
         logger.info(Fore.GREEN + Style.BRIGHT + f"Initialization complete. Awaiting market whispers...")
         print(Fore.MAGENTA + "=" * 80 + Style.RESET_ALL) # Separator before first cycle log
    else:
         # Error should have been logged during init, exit was likely called, but double-check.
         logger.critical(Fore.RED + Style.BRIGHT + "Exchange or Market info failed to load during initialization. Cannot start trading loop.")
         termux_notify("Bot Init Failed!", f"{CONFIG.symbol} Exchange/Market load error.")
         sys.exit(1) # Exit if init failed

    cycle = 0
    try:
        while True:
            cycle += 1
            try:
                trading_spell_cycle(cycle)
            except Exception as cycle_error:
                 # Catch errors *within* a cycle to prevent the whole script from crashing
                 logger.error(Fore.RED + Style.BRIGHT + f"Unhandled Error during trading cycle {cycle}: {cycle_error}", exc_info=True)
                 termux_notify("Cycle Error!", f"{CONFIG.symbol} Cycle {cycle} failed. Check logs.")
                 # Decide if a single cycle failure is fatal. For now, log and continue to the next cycle after sleep.
                 # If errors are persistent, fetch_with_retries/other checks should eventually halt.

            logger.info(Fore.BLUE + f"Cycle {cycle} finished. Resting for {CONFIG.loop_sleep_seconds} seconds...")
            time.sleep(CONFIG.loop_sleep_seconds)

    except KeyboardInterrupt:
        logger.warning(Fore.YELLOW + "\nCtrl+C detected! Initiating graceful shutdown...")
        graceful_shutdown()
    except Exception as e:
        # Catch unexpected errors in the main loop *outside* of the trading_spell_cycle call
        logger.critical(Fore.RED + Style.BRIGHT + f"\nFATAL RUNTIME ERROR in Main Loop (Cycle {cycle}): {e}", exc_info=True)
        termux_notify("Bot CRASHED!", f"{CONFIG.symbol} FATAL ERROR! Check logs!")
        logger.warning(Fore.YELLOW + "Attempting graceful shutdown after crash...")
        try:
            graceful_shutdown() # Attempt cleanup even on unexpected crash
        except Exception as shutdown_err:
            logger.error(f"Error during crash shutdown: {shutdown_err}", exc_info=True)
        sys.exit(1) # Exit with error code
    finally:
        # Ensure logs are flushed before exit, regardless of how loop ended
        logger.info("Flushing logs...")
        logging.shutdown()
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80)
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + "*** Pyrmethus Trading Spell Deactivated ***")
        print(Back.MAGENTA + Fore.WHITE + Style.BRIGHT + " " * 80 + Style.RESET_ALL)

```
