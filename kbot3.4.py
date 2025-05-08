import os
import time
import logging
import sys
import subprocess
import csv # For Journaling
from datetime import datetime # For Journaling timestamp
from typing import Dict, Optional, Any, Tuple, Union, List
from decimal import Decimal, getcontext, ROUND_DOWN, InvalidOperation, DivisionByZero, ROUND_HALF_EVEN
import copy
import textwrap # For wrapping signal reason
import platform # To detect OS for subprocess compatibility
import signal # For graceful shutdown on SIGINT, SIGTERM

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
    # Provide specific guidance for Termux users or general pip install
    init(autoreset=True) # Initialize colorama for the error message itself
    missing_pkg = e.name
    print(f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {Style.BRIGHT}{missing_pkg}{Style.NORMAL}")
    print(f"{Fore.YELLOW}To conjure it, cast the following spell:")
    if os.getenv("TERMUX_VERSION"):
  

# Set Decimal precision context for the entire application
# Set precision high enough to handle small values in prices, quantities, calculations.
# The default precision (usually 28) is often sufficient when combined with quantize.
# Increase only if necessary for extremely small values or instruments requiring ultra-high precision.
# Using Decimal prevents floating-point errors in financial calculations.
# A precision of 50 is generous and should cover most crypto use cases.
    getcontext().prec = 50
# Default rounding mode is ROUND_HALF_EVEN, which is generally suitable for financial calculations (rounds .5 to nearest even digit).
# Use ROUND_DOWN specifically for quantizing order quantities to meet exchange requirements.
# Use ROUND_HALF_EVEN for quantizing prices or percentages for display or API calls where applicable.

# --- Arcane Configuration & Logging Setup ---
# Define logger early for config class and other modules
logger = logging.getLogger(__name__)

# Define custom log levels for trade actions
TRADE_LEVEL_NUM = logging.INFO + 5  # Between INFO and WARNING
# Add the custom method to the Logger class if it doesn't exist
if not hasattr(logging.Logger, 'trade'):
    logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")
    def trade_log(self, message, *args, **kws):
        \"\"\"Custom logging method for trade-related events.\"\"\"
        if self.isEnabledFor(TRADE_LEVEL_NUM):
            # pylint: disable=protected-access
            self._log(TRADE_LEVEL_NUM, message, args, **kws)
    logging.Logger.trade = trade_log


# More detailed log format, includes module and line number for easier debugging
log_formatter = logging.Formatter(
    Fore.CYAN + "%(asctime)s "
    + Style.BRIGHT + "[%(levelname)-8s] " # Padded levelname
    + Fore.WHITE + "(%(filename)s:%(lineno)d) " # Added file/line info
    + Style.RESET_ALL
    + Fore.WHITE + "%(message)s" + Style.RESET_ALL # Ensure reset at the end
)
# Set level via environment variable or default to INFO
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
# Validate log level string
valid_log_levels = ["DEBUG", "INFO", "TRADE", "WARNING", "ERROR", "CRITICAL"]
if log_level_str not in valid_log_levels:
     print(f"{Fore.YELLOW}Warning: Invalid LOG_LEVEL '{log_level_str}' in environment. Defaulting to INFO.{Style.RESET_ALL}")
     log_level_str = "INFO"

# Use the custom TRADE level number if specified
log_level = TRADE_LEVEL_NUM if log_level_str == "TRADE" else getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Ensure handlers are not duplicated if script is reloaded (e.g., in some interactive environments)
# Check if a handler already exists writing to stdout
if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
     stream_handler = logging.StreamHandler(sys.stdout)
     stream_handler.setFormatter(log_formatter)
     logger.addHandler(stream_handler)

# Prevent duplicate messages if the root logger is also configured (common issue)
# This logger will handle its own messages and not pass them up.
logger.propagate = False


class TradingConfig:
    \"\"\"Holds the sacred parameters of our spell, enhanced with precision awareness and validation.\"\"\"
    def __init__(self):
        logger.debug("Loading configuration from environment variables...")
        # Default symbol format for Bybit V5 Unified is BASE/QUOTE:SETTLE, e.g., BTC/USDT:USDT
        # Using USDT as default settle currency as it's common for linear/swap
        # Bybit V5 supports different account types (UNIFIED, CONTRACT). Unified is recommended.
        # CCXT handles mapping, but ensure your API keys are for the correct account type.
        self.symbol = self._get_env("SYMBOL", "BTC/USDT:USDT", Fore.YELLOW)
        # 'linear', 'inverse', 'swap'. 'swap' is common for perpetuals on Unified.
        # This primarily affects how CCXT interacts and how quantity is calculated.
        # Ensure market type matches the symbol's settle currency (e.g., USDT for linear/swap, base for inverse)
        self.market_type = self._get_env("MARKET_TYPE", "linear", Fore.YELLOW, allowed_values=['linear', 'inverse', 'swap']).lower()
        self.interval = self._get_env("INTERVAL", "1m", Fore.YELLOW)
        # Risk as a percentage of total equity (e.g., 0.01 for 1%, 0.001 for 0.1%)
        self.risk_percentage = self._get_env("RISK_PERCENTAGE", "0.01", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.00001"), max_val=Decimal("0.5")) # 0.001% to 50% risk
        self.sl_atr_multiplier = self._get_env("SL_ATR_MULTIPLIER", "1.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")) # Reasonable bounds
        # TSL activation threshold in ATR units above entry price
        self.tsl_activation_atr_multiplier = self._get_env("TSL_ACTIVATION_ATR_MULTIPLIER", "1.0", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.1"), max_val=Decimal("20.0")) # Reasonable bounds
        # Bybit V5 TSL distance is a percentage (e.g., 0.5 for 0.5%). Ensure value is suitable.
        # Note: Bybit expects a *string* for this param, but we store as Decimal internally.
        self.trailing_stop_percent = self._get_env("TRAILING_STOP_PERCENT", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0.001"), max_val=Decimal("10.0")) # 0.001% to 10% trail
        # Trigger type for SL/TSL orders. Bybit V5 allows LastPrice, MarkPrice, IndexPrice.
        self.sl_trigger_by = self._get_env("SL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"])
        self.tsl_trigger_by = self._get_env("TSL_TRIGGER_BY", "LastPrice", Fore.YELLOW, allowed_values=["LastPrice", "MarkPrice", "IndexPrice"]) # Usually same as SL

        # --- Indicator Periods (Read from .env with new recommended defaults) ---
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
        # ATR Filter Threshold (price move from previous close must be > X * ATR)
        self.atr_move_filter_multiplier = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", "0.5", Fore.YELLOW, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))

        # Epsilon: Small fixed value for comparing quantities to zero.
        # Derived dynamically from market step size can be complex and error-prone.
        # A tiny fixed value is generally safe for typical crypto precision for zero checks.
        # For formatting *actual* trade quantities, always use market precision.
        self.position_qty_epsilon = Decimal("1E-12")
        logger.debug(f"Using fixed position_qty_epsilon for zero checks: {self.position_qty_epsilon:.1E}")

        # API Keys
        self.api_key = self._get_env("BYBIT_API_KEY", None, Fore.RED)
        self.api_secret = self._get_env("BYBIT_API_SECRET", None, Fore.RED)

        # Operational Parameters
        self.ohlcv_limit = self._get_env("OHLCV_LIMIT", "200", Fore.YELLOW, cast_type=int, min_val=50, max_val=1000)
        # Fixed typo: LOOP_SLEP_SECONDS -> LOOP_SLEEP_SECONDS
        self.loop_sleep_seconds = self._get_env("LOOP_SLEEP_SECONDS", "15", Fore.YELLOW, cast_type=int, min_val=5)
        self.order_check_delay_seconds = self._get_env("ORDER_CHECK_DELAY_SECONDS", "2", Fore.YELLOW, cast_type=int, min_val=1)
        self.order_check_timeout_seconds = self._get_env("ORDER_CHECK_TIMEOUT_SECONDS", "12", Fore.YELLOW, cast_type=int, min_val=5)
        self.max_fetch_retries = self._get_env("MAX_FETCH_RETRIES", "3", Fore.YELLOW, cast_type=int, min_val=1, max_val=10)
        self.trade_only_with_trend = self._get_env("TRADE_ONLY_WITH_TREND", "True", Fore.YELLOW, cast_type=bool)

        # Journaling Configuration
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
                 min_val: Optional[Union[int, float, Decimal]] = None, # Allow float for min/max input
                 max_val: Optional[Union[int, float, Decimal]] = None, # Allow float for min/max input
                 allowed_values: Optional[List[str]] = None) -> Any:
        \"\"\"Gets value from environment, casts, validates, and logs.\"\"\"
        value_str = os.getenv(key)
        is_default = False
        # Mask secrets in logs
        log_value = "****" if "SECRET" in key.upper() or "KEY" in key.upper() else value_str

        if value_str is None or value_str.strip() == "": # Treat empty string as not set
            value = default
            is_default = True
            if default is not None:
                # Log default only if it's actually used because the key wasn't set or was empty
                logger.warning(f"{color}Using default value for {key}: {default}")
            # Use string representation of default for casting below if needed, *unless* default is None
            value_str = str(default) if default is not None else None
        else:
             # Log the fetched value (masked if needed)
             logger.info(f"{color}Summoned {key}: {log_value}")

        # Handle case where default is None and no value is set - this is a required config missing
        if value_str is None and default is None:
            logger.critical(f"{Fore.RED+Style.BRIGHT}Required configuration '{key}' not found in environment variables and no default provided. Halting.")
            sys.exit(1)

        # --- Casting ---
        casted_value = None
        try:
            if cast_type == bool:
                # Robust boolean check for string inputs
                # If value_str is None (meaning default was used), convert default (which must be boolean) to string for comparison
                value_to_cast = value_str if value_str is not None else str(default)
                casted_value = value_to_cast.lower() in ['true', '1', 'yes', 'y', 'on']
            elif cast_type == Decimal:
                # Cast from string representation to preserve precision
                # Handle potential None for value_str if default was used
                value_to_cast_dec = value_str if value_str is not None else str(default)
                casted_value = Decimal(value_to_cast_dec)
            elif cast_type == int:
                value_to_cast_int = value_str if value_str is not None else str(default)
                casted_value = int(float(value_to_cast_int)) # Cast via float first to handle "1.0" etc.
            elif cast_type == float:
                # Warn against using float for critical financial values
                if key in ["RISK_PERCENTAGE", "SL_ATR_MULTIPLIER", "TSL_ACTIVATION_ATR_MULTIPLIER", "TRAILING_STOP_PERCENT", "TREND_FILTER_BUFFER_PERCENT", "ATR_MOVE_FILTER_MULTIPLIER"]:
                     logger.warning(f"{Fore.YELLOW}Using float for critical config '{key}'. Consider using Decimal for better precision.")
                value_to_cast_float = value_str if value_str is not None else str(default)
                casted_value = float(value_to_cast_float)
            else: # Default is str
                value_to_cast_str = value_str if value_str is not None else str(default)
                casted_value = str(value_to_cast_str)

        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"{Fore.RED}Could not cast {key} ('{value_str}') to {cast_type.__name__}: {e}. Attempting to use default value '{default}'.")
            # Attempt to cast the default value itself
            try:
                if default is None: # If default is None and casting failed, return None
                    return None
                if cast_type == bool: casted_value = str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                elif cast_type == Decimal: casted_value = Decimal(str(default))
                elif cast_type == int: casted_value = int(float(default)) # Use float intermediary
                else: casted_value = cast_type(default)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__}: {cast_default_err}. Halting.")
                sys.exit(1)

        # --- Validation ---
        # Allowed values check (case-insensitive for strings)
        if allowed_values:
            # Convert casted_value to lower for comparison if it's a string
            comp_value = str(casted_value).lower() if isinstance(casted_value, str) else casted_value
            lower_allowed = [str(v).lower() for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Invalid value '{casted_value}' for {key}. Allowed values: {allowed_values}. Using default: {default}")
                # Return default after logging error, re-cast default carefully
                try:
                    if default is None: return None
                    if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                    elif cast_type == Decimal: return Decimal(str(default))
                    elif cast_type == int: return int(float(default)) # Use float intermediary
                    return cast_type(default)
                except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                    logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__} for validation fallback: {cast_default_err}. Halting.")
                    sys.exit(1)


        # Min/Max checks (for numeric types - Decimal, int, float)
        validation_failed = False
        if isinstance(casted_value, (Decimal, int, float)):
            try:
                # --- Type Coercion for Comparison ---
                # Convert casted_value to Decimal for consistent comparison if it's int or float
                compare_value = Decimal(str(casted_value)) if not isinstance(casted_value, Decimal) else casted_value

                # Convert min/max limits to Decimal for comparison, handling None
                min_val_comp = Decimal(str(min_val)) if min_val is not None else None
                max_val_comp = Decimal(str(max_val)) if max_val is not None else None

                # Perform comparison using Decimal
                if min_val_comp is not None and compare_value < min_val_comp:
                    logger.error(f"{Fore.RED}{key} value {casted_value} is below minimum {min_val}. Using default: {default}")
                    validation_failed = True
                if max_val_comp is not None and compare_value > max_val_comp:
                     logger.error(f"{Fore.RED}{key} value {casted_value} is above maximum {max_val}. Using default: {default}")
                     validation_failed = True

            except InvalidOperation as e:
                 logger.error(f"{Fore.RED}Error during Decimal conversion for min/max validation for {key} with value {casted_value} and limits ({min_val}, {max_val}): {e}. Using default: {default}")
                 validation_failed = True
            except TypeError as e:
                 # Should be less likely with Decimal conversion, but catch anyway
                 logger.error(f"{Fore.RED}TypeError during min/max validation for {key}: {e}. Value type: {type(casted_value).__name__}, Limit types: {type(min_val).__name__}/{type(max_val).__name__}. Using default: {default}")
                 validation_failed = True


        if validation_failed:
            # Re-cast default to ensure correct type is returned
            try:
                if default is None: return None # Should be caught earlier, but defensive
                if cast_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                elif cast_type == Decimal: return Decimal(str(default))
                elif cast_type == int: return int(float(default)) # Use float intermediary
                return cast_type(default)
            except (ValueError, TypeError, InvalidOperation) as cast_default_err:
                logger.critical(f"{Fore.RED+Style.BRIGHT}Default value '{default}' for {key} also cannot be cast to {cast_type.__name__} for validation fallback: {cast_default_err}. Halting.")
                sys.exit(1)

        # Return the original casted_value (not the potentially Decimal-coerced compare_value)
        return casted_value

# --- Instantiate Configuration ---
logger.info(Fore.MAGENTA + Style.BRIGHT + "Initializing Arcane Configuration v2.3.1...")
# Summon secrets from the .env scroll
load_dotenv()
# Instantiate CONFIG *after* logging is set up and dotenv is loaded
CONFIG = TradingConfig()

# --- Global Variables ---
# Declared at module level, needs `global` inside functions only if assigned to.
MARKET_INFO: Optional[Dict] = None # Global to store market details after connection
EXCHANGE: Optional[ccxt.Exchange] = None # Global for the exchange instance
# Tracks active SL/TSL *presence* using markers for position-based stops (no order IDs for these in Bybit V5 set-trading-stop).
# This tracker is crucial for knowing if a position *should* have a stop attached and which type.
# The values are placeholders like "POS_SL_LONG" or "POS_TSL_SHORT" to indicate status, not actual order IDs.
# Order IDs for SL/TSL set via set-trading-stop on Bybit V5 are internal to the exchange and not returned by the call.
# We assume one-way mode, so only one active position (long or short) at a time.
# Initialize with None to represent no active stop markers
order_tracker: Dict[str, Dict[str, Optional[str]]] = {
    "long": {"sl_id": None, "tsl_id": None}, # sl_id/tsl_id stores marker (e.g., "POS_SL_LONG") or None
    "short": {"sl_id": None, "tsl_id": None}
}

# Flag to signal graceful shutdown
shutdown_requested = False

# --- Core Spell Functions ---

def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    \"\"\"Generic wrapper to fetch data with retries and exponential backoff.\"\"\"
    # No need for global EXCHANGE, CONFIG here as we are only accessing them.

    if EXCHANGE is None:
        logger.critical("Exchange object is None, cannot fetch data.")
        return None # Indicate critical failure

    last_exception = None
    # Add category param automatically for V5 if not already present in kwargs['params']
    # Check if 'params' is a dict before attempting to add category
    if 'params' not in kwargs or not isinstance(kwargs['params'], dict):
        kwargs['params'] = {}
    # Ensure category is set if it's a Bybit V5 exchange and category is configured in options
    # Check for EXCHANGE.options and its structure defensively
    # Bybit V5 Unified endpoints often require the category param (linear, inverse, spot, option, etc.)
    # CCXT handles mapping, but explicitly setting in params can ensure it's used.
    # Ensure MARKET_TYPE from config is used as the category.
    if hasattr(EXCHANGE, 'id') and EXCHANGE.id == 'bybit':
        # Explicitly add category based on config if not already provided in params
        if 'category' not in kwargs['params'] and CONFIG.market_type:
            kwargs['params']['category'] = CONFIG.market_type
            logger.debug(f"Auto-added category '{kwargs['params']['category']}' to params for {fetch_function.__name__}")

    # Log the full kwargs with category added before the loop starts
    log_kwargs_initial = {}
    for k, v in kwargs.items():
         if isinstance(v, dict):
              log_kwargs_initial[k] = {vk: ('****' if isinstance(vk, str) and ('secret' in vk.lower() or 'key' in vk.lower() or 'password' in vk.lower() or 'sign' in vk.lower() or 'sig' in vk.lower()) else vv) for vk, vv in v.items()}
         else:
              log_kwargs_initial[k] = '****' if isinstance(k, str) and ('secret' in k.lower() or 'key' in k.lower() or 'password' in k.lower()) else v
    log_args_initial = ['****' if isinstance(a, str) and ('secret' in a.lower() or 'key' in a.lower() or 'password' in a.lower() or len(a) > 30) else a for a in args]
    method_name_initial = getattr(fetch_function, '__name__', str(fetch_function))
    logger.debug(f"Preparing to call {method_name_initial} with args={log_args_initial}, kwargs={log_kwargs_initial}")


    for attempt in range(CONFIG.max_fetch_retries + 1): # +1 to allow logging final failure
        if shutdown_requested:
             logger.warning("Shutdown requested, aborting fetch_with_retries.")
             return None # Abort if shutting down

        try:
            # Log the attempt number and function being called at DEBUG level
            # Use the already prepared masked args/kwargs
            method_name = getattr(fetch_function, '__name__', str(fetch_function))
            logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {method_name}...")
            # logger.debug(f"Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}: Calling {method_name} with args={log_args_initial}, kwargs={log_kwargs_initial}") # Redundant if logged above

            result = fetch_function(*args, **kwargs)
            return result # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection) as e:
            last_exception = e
            wait_time = 2 ** attempt # Exponential backoff (1, 2, 4, 8...)
            logger.warning(Fore.YELLOW + f"{method_name}: Network issue (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}). Retrying in {wait_time}s... Error: {e}")
            if attempt < CONFIG.max_fetch_retries:
                if not shutdown_requested: time.sleep(wait_time) # Check shutdown before sleep
            else:
                logger.error(Fore.RED + f"{method_name}: Failed after {CONFIG.max_fetch_retries + 1} attempts due to network issues.")
                # On final network failure, return None. Caller handles the consequence.
                return None
        except ccxt.ExchangeNotAvailable as e:
             last_exception = e
             logger.error(Fore.RED + f"{method_name}: Exchange not available: {e}. Stopping retries.")
             # This is usually a hard stop, no point retrying transiently
             return None # Indicate failure
        except ccxt.AuthenticationError as e:
             last_exception = e
             logger.critical(Fore.RED + Style.BRIGHT + f"{method_name}: Authentication error: {e}. Halting script.")
             # Request shutdown to prevent further actions
             global shutdown_requested
             shutdown_requested = True
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

            # Check for common rate limit codes/messages (Bybit V5 examples)
            # Bybit V5: 10009 (Freq), 10017 (Key limit), 10018 (IP limit), 10020 (too many requests)
            if "Rate limit exceeded" in error_message or "too many visits" in error_message.lower() or error_code in [10017, 10018, 10009, 10020]:
                 wait_time = 5 * (attempt + 1) # Longer wait for rate limits
                 logger.warning(f"{Fore.YELLOW}{method_name}: Rate limit hit (Code: {error_code}). Retrying in {wait_time}s... Error: {error_message}")
            # Check for specific non-retryable errors (e.g., invalid parameter codes, state issues)
            # Bybit V5 Invalid Parameter codes often start with 11xxxx or others indicating bad input/state
            # 30034: Position status not normal, 110025: SL/TP order not found (might be benign for TSL activation attempt)
            # 110001: Parameter error, 110006: Invalid price precision, 110007: Invalid order quantity
            # 110041: Order quantity exceeds limit, 110042: Order price exceeds limit
            # 110013: Insufficient balance, 110017: Order amount lower than min notional
            # 30042: Risk limit error (e.g., trying to open too large position)
            # 340003: Duplicate order (for clientOrderId, not applicable for market order without clientOrderId)
            elif error_code is not None and (
                (110000 <= error_code <= 110100 and error_code not in [110025]) or # Exclude 110025 from automatic non-retry unless caller handles
                error_code in [30034, 30042, 110013, 110017, 110041, 110042, 340003] # Add specific non-retryable codes
            ):
                 logger.error(Fore.RED + f"{method_name}: Non-retryable parameter/logic exchange error (Code: {error_code}): {error_message}. Stopping retries.")
                 should_retry = False
                 # Re-raise these specific errors so the caller can handle them appropriately
                 # Note: ccxt.ExchangeError is already the base class, re-raising it here
                 # allows the caller to catch ExchangeError or a more specific subclass.
                 raise e
            # Handle the 110025 case specifically - it might be retryable if it's a timing issue, or not if the SL/TP genuinely doesn't exist.
            # For now, treat 110025 as potentially retryable by default ExchangeError logic unless specified otherwise by the caller context (e.g., TSL management).
            # If a caller needs to handle 110025 specially (like in TSL management), they should catch ExchangeError and check the code.
            else:
                 # General exchange error, apply default backoff
                 logger.warning(f"{Fore.YELLOW}{method_name}: Exchange error (Attempt {attempt + 1}/{CONFIG.max_fetch_retries + 1}, Code: {error_code}). Retrying in {wait_time}s... Error: {error_message}")

            if should_retry and attempt < CONFIG.max_fetch_retries:
                 if not shutdown_requested: time.sleep(wait_time) # Check shutdown before sleep
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

    # Should not reach here if successful, but defensive return None if somehow loop finishes without success/failure
    return None


# --- Exchange Nexus Initialization ---
logger.info(Fore.MAGENTA + Style.BRIGHT + "\\nEstablishing Nexus with the Exchange v2.3.1...")
try:
    exchange_options = {
        "apiKey": CONFIG.api_key,
        "secret": CONFIG.api_secret,
        "enableRateLimit": True, # CCXT built-in rate limiter
        "options": {
            'defaultType': CONFIG.market_type, # Use configured market_type (linear/inverse/swap)
            'adjustForTimeDifference': True, # Auto-sync clock with server
            # Bybit V5 API often requires 'category' for unified endpoints
            'brokerId': 'PyrmethusV231', # Custom identifier for Bybit API tracking (optional but good practice)
            'v5': {'category': CONFIG.market_type} # Explicitly set category for V5 requests based on config
        }
    }
    # Log options excluding secrets for debugging
    log_options = copy.deepcopy(exchange_options) # Use deepcopy to avoid modifying original
    log_options['apiKey'] = '****'
    log_options['secret'] = '****'
    # Ensure nested options are also masked if they contain sensitive info (although none here)

    logger.debug(f"Initializing CCXT Bybit with options: {log_options}")

    EXCHANGE = ccxt.bybit(exchange_options)

    # Test connectivity and credentials (important!)
    logger.info("Verifying credentials and connection...")
    try:
        EXCHANGE.check_required_credentials() # Checks if keys are present/formatted ok
        logger.info("Credentials format check passed.")
    except Exception as e:
         logger.critical(Fore.RED + Style.BRIGHT + f"Credential format check failed: {e}. Halting.")
         sys.exit(1)

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
        logger.warning(Fore.YELLOW + f"Significant time difference ({time_diff_ms} ms) between system and exchange. Check system clock synchronization (e.g., using 'ntpdate pool.ntp.org' or systemd-timesyncd).")

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
             # Also get base and quote for filtering
             parts = CONFIG.symbol.replace(':','/').split('/') # e.g., ['BTC', 'USDT', 'USDT'] or ['BTC', 'USD']
             base_currency = parts[0].strip().upper() if parts else None
             quote_currency_candidate = parts[1].strip().upper() if len(parts) > 1 else None
             settle_currency_candidate = parts[-1].strip().upper() if len(parts) > 1 else None # Assume last part after : or / is settle

             # Decide on the target settle currency for suggestions based on market type and symbol format
             target_settle = None
             if CONFIG.market_type == 'inverse':
                  target_settle = base_currency # Inverse contracts settle in the base currency (e.g., BTC/USD settles in BTC)
                  logger.debug(f"Derived target settle currency for INVERSE market: {target_settle}")
             elif CONFIG.market_type in ['linear', 'swap']:
                  # For linear/swap, settle is usually the quote currency or explicitly defined after ':'
                  if settle_currency_candidate and settle_currency_candidate != quote_currency_candidate: # Format BASE/QUOTE:SETTLE
                      target_settle = settle_currency_candidate
                      logger.debug(f"Derived target settle currency from symbol suffix: {target_settle}")
                  else: # Format BASE/QUOTE or BASE/QUOTE:QUOTE
                      target_settle = quote_currency_candidate
                      logger.debug(f"Derived target settle currency from quote currency: {target_settle}")
             else:
                  logger.debug("Could not derive specific target settle currency for market type.")


             logger.info(f"Attempting to find active symbols settling in {target_settle or 'any currency'} for type {CONFIG.market_type}...")

             for s, m in EXCHANGE.markets.items():
                  # Check if market matches the configured type (linear/inverse/swap) and is active
                  is_correct_type = (
                      (CONFIG.market_type == 'linear' and m.get('linear', False)) or
                      (CONFIG.market_type == 'inverse' and m.get('inverse', False)) or
                      (CONFIG.market_type == 'swap' and m.get('swap', False))
                  )
                  # Fallback check using 'type' field if boolean flags are missing
                  if not is_correct_type:
                       is_correct_type = m.get('type') == CONFIG.market_type

                  # Filter by settle currency if known and check if active
                  if m.get('active') and is_correct_type:
                      market_settle = m.get('settle') # CCXT standard field

                      if target_settle is None: # If no specific target settle was derived from config
                           available_symbols.append(s)
                      elif market_settle and market_settle.upper() == target_settle:
                          available_symbols.append(s)

         except Exception as parse_err:
             logger.error(f"Could not parse symbol or filter suggestions: {parse_err}")
             # Fallback: List all active symbols of the correct type if filtering fails
             available_symbols = [
                 s for s, m in EXCHANGE.markets.items()
                 if m.get('active') and (
                    (CONFIG.market_type == 'linear' and m.get('linear', False)) or
                    (CONFIG.market_type == 'inverse' and m.get('inverse', False)) or
                    (CONFIG.market_type == 'swap' and m.get('swap', False)) or
                    (m.get('type') == CONFIG.market_type) # Fallback type check
                 )
             ]


         suggestion_limit = 50 # Increased suggestion limit
         if available_symbols:
             # Sort for readability, limit suggestions
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
            # precision['price'] is the tick size (Decimal/string)
            price_tick_size_raw = MARKET_INFO['precision'].get('price')
            # precision['amount'] is the step size (Decimal/string)
            amount_step_size_raw = MARKET_INFO['precision'].get('amount')
            min_amount_raw = MARKET_INFO['limits']['amount'].get('min')
            max_amount_raw = MARKET_INFO['limits']['amount'].get('max') # Max might be None
            contract_size_raw = MARKET_INFO.get('contractSize') # Can be None or 1 for linear/swap
            min_cost_raw = MARKET_INFO['limits'].get('cost', {}).get('min') # Min cost might not exist
            min_notional_raw = MARKET_INFO['limits'].get('notional', {}).get('min') # V5 Unified uses Notional

            # Convert to Decimal for logging and potential use, handle None/N/A
            # Use Decimal(str(value)) to preserve precision if value is float/int
            price_tick_dec = Decimal(str(price_tick_size_raw)) if price_tick_size_raw is not None else Decimal("NaN")
            amount_step_dec = Decimal(str(amount_step_size_raw)) if amount_step_size_raw is not None else Decimal("NaN")
            min_amount_dec = Decimal(str(min_amount_raw)) if min_amount_raw is not None else Decimal("NaN")
            max_amount_dec = Decimal(str(max_amount_raw)) if max_amount_raw is not None else Decimal('Infinity') # Use Infinity for no max
            contract_size_dec = Decimal(str(contract_size_raw)) if contract_size_raw is not None else Decimal("1") # Default to '1' if not present/None
            min_cost_dec = Decimal(str(min_cost_raw)) if min_cost_raw is not None else Decimal("NaN")
            min_notional_dec = Decimal(str(min_notional_raw)) if min_notional_raw is not None else Decimal("NaN")


            logger.debug(f"Market Precision: Price Tick Size={price_tick_dec.normalize() if not price_tick_dec.is_nan() else 'N/A'}, Amount Step Size={amount_step_dec.normalize() if not amount_step_dec.is_nan() else 'N/A'}")
            logger.debug(f"Market Limits: Min Amount={min_amount_dec.normalize() if not min_amount_dec.is_nan() else 'N/A'}, Max Amount={max_amount_dec.normalize() if max_amount_dec != Decimal('Infinity') else 'Infinity'}")
            logger.debug(f"Market Limits: Min Cost={min_cost_dec.normalize() if not min_cost_dec.is_nan() else 'N/A'}, Min Notional={min_notional_dec.normalize() if not min_notional_dec.is_nan() else 'N/A'}")
            logger.debug(f"Contract Size: {contract_size_dec.normalize()}")

            # --- Validate Market Type vs Settle Currency ---
            market_settle_currency = MARKET_INFO.get('settle')
            market_base_currency = MARKET_INFO.get('base')
            market_quote_currency = MARKET_INFO.get('quote')
            logger.debug(f"Market Currencies: Base={market_base_currency}, Quote={market_quote_currency}, Settle={market_settle_currency}")

            if CONFIG.market_type == 'inverse' and market_settle_currency != market_base_currency:
                 logger.critical(f"{Fore.RED+Style.BRIGHT}Configuration Mismatch: MARKET_TYPE is 'inverse' but symbol {CONFIG.symbol} settles in {market_settle_currency} (expected {market_base_currency}). Halting.")
                 sys.exit(1)
            if CONFIG.market_type in ['linear', 'swap'] and market_settle_currency != market_quote_currency:
                 # Allow settle=base for inverse, but warn if linear/swap doesn't settle in quote
                 logger.warning(f"{Fore.YELLOW}Configuration Check: MARKET_TYPE is '{CONFIG.market_type}' but symbol {CONFIG.symbol} settles in {market_settle_currency} (usually settles in quote: {market_quote_currency}). Ensure this is intended.")

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
    \"\"\"Sends a notification using Termux API (if available) via termux-toast.\"\"\"
    # Check if running in Termux specifically
    if platform.system() != "Linux" or not os.getenv("TERMUX_VERSION"):
        # logger.debug("Not running in Termux environment. Skipping notification.") # Too noisy
        return

    try:
        # Check if command exists using which (more portable than 'command -v')
        check_cmd = subprocess.run(['which', 'termux-toast'], capture_output=True, text=True, check=False, encoding='utf-8')
        if check_cmd.returncode != 0:
            # logger.debug("termux-toast command not found. Skipping notification.") # Too noisy
            return

        # Basic sanitization - focus on preventing shell interpretation issues
        # Replace potentially problematic characters with spaces or remove them
        # Keep it simple for toast notifications
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\\\', '').replace('\\n', ' ')
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\\\', '').replace('\\n', ' ')

        # Limit length to avoid potential buffer issues or overly long toasts
        max_len = 200 # Keep toast messages concise
        full_message = f"{safe_title}: {safe_content}"[:max_len]

        # Use list format for subprocess.run for security (prevents shell injection)
        # Example styling: gravity middle, black text on green background, short duration
        # Ensure command and args are passed as separate list items
        # Use a slightly longer duration for important messages
        duration_flag = '-d'
        duration_value = 'long' if 'CRITICAL' in title.upper() or 'EMERGENCY' in title.upper() else 'short'
        # Use different background colors based on message type
        bg_color = 'red' if 'FAIL' in title.upper() or 'ERROR' in title.upper() or 'EMERGENCY' in title.upper() else 'yellow' if 'WARN' in title.upper() else 'green'

        cmd_list = ['termux-toast', '-g', 'middle', '-c', 'black', '-b', bg_color, duration_flag, duration_value, full_message]
        # shell=False is default for list format, but good to be explicit
        result = subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5, shell=False, encoding='utf-8') # Add timeout and encoding

        if result.returncode != 0:
            # Log stderr if available
            stderr_msg = result.stderr.strip()
            logger.warning(f"termux-toast command failed with code {result.returncode}" + (f": {stderr_msg}" if stderr_msg else ""))
        # No else needed, success is silent

    except FileNotFoundError:
         # logger.debug("termux-toast command not found (FileNotFoundError). Skipping notification.") # Too noisy
         pass
    except subprocess.TimeoutExpired:
         logger.warning("termux-toast command timed out. Skipping notification.")
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logger.warning(Fore.YELLOW + f"Could not conjure Termux notification: {e}", exc_info=True)

# --- Precision Casting Spells ---

def format_price(symbol: str, price: Union[Decimal, str, float, int]) -> str:
    \"\"\"Formats price according to market precision rules (tick size) using exchange's method.\"\"\"
    # No assignment, no global needed
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format price.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            price_dec = Decimal(str(price))
            # Quantize to a sensible default precision (e.g., 8 decimals) if market info fails
            return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_HALF_EVEN).normalize()) # Use ROUND_HALF_EVEN for price, normalize for clean string
        except Exception:
            return str(price) # Last resort

    try:
        # CCXT's price_to_precision handles rounding/truncation based on market rules (tick size).
        # Need to handle potential NaN Decimal input
        if isinstance(price, Decimal) and price.is_nan():
             logger.warning(f"Attempted to format NaN price for {symbol}. Returning 'NaN'.")
             return "NaN"
        # Ensure input is float as expected by CCXT methods for price formatting
        price_float = float(price)
        # Note: price_to_precision typically truncates towards zero by default for Bybit.
        # If specific rounding (like ROUND_HALF_EVEN) is desired *before* tick size application,
        # it needs to be done manually *before* calling price_to_precision.
        # However, for API calls, matching the exchange's exact tick requirement is paramount.
        return EXCHANGE.price_to_precision(symbol, price_float)
    except (AttributeError, KeyError, InvalidOperation, ValueError, TypeError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid price format '{price}': {e}. Using fallback formatting.")
         try:
             price_dec = Decimal(str(price))
             # Use ROUND_HALF_EVEN for fallback display/internal use if exchange method fails
             return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_HALF_EVEN).normalize())
         except Exception:
              return str(price)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting price {price} for {symbol}: {e}. Using fallback.")
        try:
             price_dec = Decimal(str(price))
             return str(price_dec.quantize(Decimal("1E-8"), rounding=ROUND_HALF_EVEN).normalize())
        except Exception:
            return str(price)

def format_amount(symbol: str, amount: Union[Decimal, str, float, int], rounding_mode=ROUND_DOWN) -> str:
    \"\"\"Formats amount according to market precision rules (step size) using exchange's method.\"\"\"
    # No assignment, no global needed
    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(f"{Fore.RED}Market info or Exchange not loaded for {symbol}, cannot format amount.")
        # Fallback with a reasonable number of decimal places using Decimal
        try:
            amount_dec = Decimal(str(amount))
            # Quantize to a sensible default precision (e.g., 8 decimals) using specified rounding
            return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode).normalize()) # Normalize for clean string
        except Exception:
            return str(amount) # Last resort

    try:
        # CCXT's amount_to_precision handles step size and rounding.
        # Map Python Decimal rounding modes to CCXT rounding modes if needed.
        # Bybit usually requires truncation (ROUND_DOWN) for quantity.
        # CCXT TRUNCATE (== 1) corresponds to ROUND_DOWN. CCXT ROUND (== 0) corresponds to ROUND_HALF_EVEN.
        ccxt_rounding_mode = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND

        # Need to handle potential NaN Decimal input
        if isinstance(amount, Decimal) and amount.is_nan():
             logger.warning(f"Attempted to format NaN amount for {symbol}. Returning 'NaN'.")
             return "NaN"
        # Ensure input is float as expected by CCXT methods for amount formatting
        amount_float = float(amount)
        return EXCHANGE.amount_to_precision(symbol, amount_float, rounding_mode=ccxt_rounding_mode)
    except (AttributeError, KeyError, InvalidOperation, ValueError, TypeError) as e:
         logger.error(f"{Fore.RED}Market info for {symbol} missing precision data or invalid amount format '{amount}': {e}. Using fallback formatting.")
         try:
             amount_dec = Decimal(str(amount))
             return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode).normalize()) # Normalize
         except Exception:
              return str(amount)
    except Exception as e:
        logger.error(f"{Fore.RED}Error formatting amount {amount} for {symbol}: {e}. Using fallback.")
        try:
             amount_dec = Decimal(str(amount))
             return str(amount_dec.quantize(Decimal("1E-8"), rounding=rounding_mode).normalize()) # Normalize
        except Exception:
            return str(amount)

# --- Data Fetching and Processing ---

def fetch_market_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    \"\"\"Fetch OHLCV data using the retry wrapper and perform validation.\"\"\"
    # No assignment, no global needed
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
        # fetch_with_retries handles category param automatically based on config market_type
        # fetch_ohlcv expects the CCXT unified symbol string (e.g., BTC/USDT:USDT)
        # Limit parameter might be adjusted by CCXT based on exchange capabilities.
        fetch_params = {'limit': limit} # Pass limit in params as well for some exchanges
        ohlcv_data = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, params=fetch_params) # Pass limit=None here if passing in params

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
        column_names = ["timestamp", "open", "high", "low", "close", "volume"]
        # Adjust column names if fewer than 6 columns are present (unlikely but defensive)
        if len(ohlcv_data[0]) < 6:
             column_names = ["timestamp", "open", "high", "low", "close"][:len(ohlcv_data[0])]

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
            # Use errors='coerce' to turn invalid parsing into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for NaNs in critical price columns *after* conversion
        initial_len_before_price_dropna = len(df)
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        if len(df) < initial_len_before_price_dropna:
            dropped_count = initial_len_before_price_dropna - len(df)
            logger.warning(f"Dropped {dropped_count} rows with missing essential price data (NaN) from OHLCV.")

        if df.empty:
            logger.error(Fore.RED + "DataFrame is empty after processing and cleaning OHLCV data (all rows dropped?).")
            return None

        # --- Convert Price/Volume columns to Decimal AFTER cleaning NaNs ---
        # Keep as float for indicator calcs, but this is where you *could* convert if needed elsewhere
        # try:
        #     for col in ["open", "high", "low", "close"]:
        #         df[col] = df[col].apply(lambda x: Decimal(str(x)))
        #     if "volume" in df.columns:
        #         df["volume"] = df["volume"].apply(lambda x: Decimal(str(x)))
        # except (InvalidOperation, TypeError) as dec_err:
        #      logger.error(f"Error converting DataFrame columns to Decimal: {dec_err}. Keeping as float.", exc_info=True)
        #      # Fallback: keep as float
        # logger.debug("DataFrame numeric columns kept as float for indicator calculations.")


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
                 # Use pandas to parse timeframe string robustly if possible, fallback to CCXT
                 try:
                     expected_interval_td = pd.Timedelta(EXCHANGE.parse_timeframe(timeframe), unit='s')
                 except ValueError:
                     # Fallback parsing if pandas fails (e.g., custom timeframe string)
                     interval_seconds = EXCHANGE.parse_timeframe(timeframe) # CCXT method
                     expected_interval_td = pd.Timedelta(interval_seconds, unit='s')

                 # Allow some tolerance (e.g., 20% of interval + 10s buffer) for minor timing differences/API lag
                 tolerance_seconds = expected_interval_td.total_seconds() * 0.2 + 10
                 tolerance = pd.Timedelta(seconds=tolerance_seconds)

                 # Compare absolute difference to expected + tolerance
                 if abs(time_diff.total_seconds() - expected_interval_td.total_seconds()) > tolerance.total_seconds():
                      logger.warning(f"Unexpected time gap between last two candles: {time_diff} (expected ~{expected_interval_td}, allowed lag ~{tolerance}).")
             except (ValueError, AttributeError) as parse_err:
                 logger.warning(f"Could not parse timeframe '{timeframe}' to calculate expected interval for time gap check: {parse_err}")
             except Exception as time_check_e:
                 logger.warning(f"Error during time difference check: {time_check_e}")

        # Check if enough candles are returned compared to the requested limit
        # CCXT might return fewer than requested if history is short or based on exchange limits
        fetched_count = len(df)
        if fetched_count < limit:
             # Only warn if significantly fewer are returned (e.g., less than 90% requested)
             if fetched_count < limit * 0.9 and fetched_count < CONFIG.ohlcv_limit * 0.9: # Check against config limit too
                 logger.warning(f"{Fore.YELLOW}Fetched significantly fewer candles ({fetched_count}) than requested limit ({limit}). Data might be incomplete or market history short.")
             else:
                 logger.debug(f"Fetched {fetched_count} candles (requested limit: {limit}).")

        logger.info(Fore.GREEN + f"Market whispers received ({len(df)} candles). Latest: {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        return df
    except Exception as e:
        logger.error(Fore.RED + f"Error processing OHLCV data into DataFrame: {e}", exc_info=True)
        return None

def calculate_indicators(df: pd.DataFrame) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
    \"\"\"Calculate technical indicators using CONFIG periods, returning results as Decimals for precision.\"\"\"
    # Accessing CONFIG is fine without global here.
    logger.info(Fore.CYAN + "# Weaving indicator patterns...")
    if df is None or df.empty:
        logger.error(Fore.RED + "Cannot calculate indicators on missing or empty DataFrame.")
        return None
    try:
        # Ensure data is float for TA-Lib / Pandas calculations, convert to Decimal at the end
        # Defensive: Make a copy to avoid modifying the original DataFrame if it's used elsewhere
        df_calc = df.copy()
        # Use float type directly for calculations
        numeric_cols_for_dropna = ["open", "high", "low", "close"]
        df_calc.dropna(subset=numeric_cols_for_dropna, inplace=True)

        if df_calc.empty:
             logger.error(Fore.RED + "DataFrame is empty after dropping rows with NaN price data. Cannot calculate indicators.")
             return None

        # Use pandas Series directly which often simplifies calculations
        close_series = df_calc["close"].astype(float)
        high_series = df_calc["high"].astype(float)
        low_series = df_calc["low"].astype(float)


        # --- Check Data Length Requirements ---
        # Ensure enough data for EWMA initial states to stabilize somewhat
        # Generally, need at least span periods, but more is better for stability.
        required_len_ema_stable = max(CONFIG.fast_ema_period, CONFIG.slow_ema_period, CONFIG.trend_ema_period) * 2 # Aim for 2x span
        # Stoch requires period + smoothing K + smoothing D for first stable D value.
        required_len_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 1 # More accurate minimum
        # ATR requires period + 1 for first value, more for stability.
        required_len_atr = CONFIG.atr_period * 2 # Aim for 2x period for stability

        min_required_len = max(required_len_ema_stable, required_len_stoch, required_len_atr)
        # Add a buffer to ensure the latest indicator values (iloc[-1]) are meaningful
        # Use a buffer of at least smoothing periods or a fixed value
        min_safe_len = min_required_len + max(CONFIG.stoch_smooth_d, CONFIG.atr_period, 5) # Buffer ensures last value isn't the very first calculated one


        if len(df_calc) < min_safe_len:
             logger.warning(f"{Fore.YELLOW}Not enough data ({len(df_calc)}) for fully stable indicators (minimum safe estimate: {min_safe_len}). Indicator values might be less reliable. Increase OHLCV_LIMIT or wait for more data.")
             # Proceed anyway, but warn

        # Check absolute minimum needed to calculate *any* value (even if unstable)
        abs_min_len_ema = max(CONFIG.fast_ema_period, CONFIG.slow_ema_period, CONFIG.trend_ema_period)
        abs_min_len_stoch = CONFIG.stoch_period + CONFIG.stoch_smooth_k + CONFIG.stoch_smooth_d - 2 # Min for first D
        abs_min_len_atr = CONFIG.atr_period + 1 # Min for first ATR
        abs_min_required = max(abs_min_len_ema, abs_min_len_stoch, abs_min_len_atr)

        if len(df_calc) < abs_min_required:
             logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)}) for core indicator calculations (absolute minimum required: {abs_min_required}). Cannot calculate indicators.")
             return None # Critical failure if even minimum isn't met


        # --- Calculations using Pandas ---
        # Use float values for calculations, convert to Decimal at the end
        # adjust=False matches standard EMA behavior (gives more weight to recent data)
        fast_ema_series = close_series.ewm(span=CONFIG.fast_ema_period, adjust=False).mean()
        slow_ema_series = close_series.ewm(span=CONFIG.slow_ema_period, adjust=False).mean()
        trend_ema_series = close_series.ewm(span=CONFIG.trend_ema_period, adjust=False).mean()

        # Stochastic Oscillator %K and %D
        low_min_series = low_series.rolling(window=CONFIG.stoch_period).min()
        high_max_series = high_series.rolling(window=CONFIG.stoch_period).max()
        range_series = high_max_series - low_min_series
        # Use np.where for safe division, handling zero range and NaN correctly
        stoch_k_raw_series = pd.Series(np.where(range_series > np.finfo(float).eps, 100 * (close_series - low_min_series) / range_series, np.nan), index=close_series.index)

        # Smooth %K to get final %K and then smooth %K to get %D
        stoch_k_series = stoch_k_raw_series.rolling(window=CONFIG.stoch_smooth_k).mean()
        stoch_d_series = stoch_k_series.rolling(window=CONFIG.stoch_smooth_d).mean()

        # Average True Range (ATR) - Wilder's smoothing matches TradingView standard
        prev_close_series = close_series.shift(1)
        tr_series = pd.concat([
            high_series - low_series,
            (high_series - prev_close_series).abs(),
            (low_series - prev_close_series).abs()
        ], axis=1).max(axis=1, skipna=False) # skipna=False to ensure first TR is NaN

        # Use ewm with alpha = 1/period for Wilder's smoothing (adjust=False)
        # ATR is calculated on the TR series. Initial NaNs in TR will lead to initial NaNs in ATR.
        if len(tr_series.dropna()) >= CONFIG.atr_period: # Check if enough non-NaN TR values exist
             # adjust=False is crucial for Wilder's smoothing
             atr_series = tr_series.ewm(alpha=1/CONFIG.atr_period, adjust=False).mean()
        else:
             logger.warning(f"Not enough valid data points ({len(tr_series.dropna())}) for ATR calculation (need at least {CONFIG.atr_period} non-NaN TR values). ATR will be NaN.")
             atr_series = pd.Series(np.nan, index=close_series.index)


        # --- Extract Latest Values & Convert to Decimal ---
        # Define quantizers for consistent decimal places for intermediate Decimal objects if needed
        # Note: Final values are returned without quantization here; formatting happens elsewhere.
        # price_quantizer = Decimal("1E-8")
        # percent_quantizer = Decimal("1E-2")
        # atr_quantizer = Decimal("1E-8")

        # Helper to safely get latest non-NaN value, convert to Decimal, and handle errors
        def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
            if series.empty or series.isna().all():
                logger.debug(f"Indicator series '{name}' is empty or all NaN.")
                return Decimal("NaN")
            # Get the last valid (non-NaN) value using .iloc[-1] after dropna()
            latest_valid_series = series.dropna()
            if latest_valid_series.empty:
                logger.debug(f"Indicator series '{name}' has no valid (non-NaN) values.")
                return Decimal("NaN")

            latest_valid_val = latest_valid_series.iloc[-1]

            try:
                # Convert via string for precision
                return Decimal(str(latest_valid_val))
            except (InvalidOperation, TypeError) as e:
                logger.error(f"Could not convert indicator '{name}' value {latest_valid_val} to Decimal: {e}. Returning NaN.", exc_info=True)
                return Decimal("NaN")

        # Get latest values as Decimals
        latest_fast_ema = get_latest_decimal(fast_ema_series, "fast_ema")
        latest_slow_ema = get_latest_decimal(slow_ema_series, "slow_ema")
        latest_trend_ema = get_latest_decimal(trend_ema_series, "trend_ema")
        latest_stoch_k = get_latest_decimal(stoch_k_series, "stoch_k")
        latest_stoch_d = get_latest_decimal(stoch_d_series, "stoch_d")
        latest_atr = get_latest_decimal(atr_series, "atr")

        # --- Calculate Stochastic Cross Signals (Boolean) ---
        stoch_kd_bullish = False
        stoch_kd_bearish = False
        # Need at least 2 valid points in both K and D series *after* dropping NaNs
        k_series_valid = stoch_k_series.dropna()
        d_series_valid = stoch_d_series.dropna()

        if len(k_series_valid) >= 2 and len(d_series_valid) >= 2:
             try:
                  # Get the last two valid (non-nan) values as Decimals
                  stoch_k_last = Decimal(str(k_series_valid.iloc[-1]))
                  stoch_d_last = Decimal(str(d_series_valid.iloc[-1]))
                  stoch_k_prev = Decimal(str(k_series_valid.iloc[-2]))
                  stoch_d_prev = Decimal(str(d_series_valid.iloc[-2]))

                  # Check for crossover using previous vs current values (Decimal comparison)
                  # Bullish cross: K crosses above D
                  stoch_kd_bullish_raw = (stoch_k_last > stoch_d_last) and (stoch_k_prev <= stoch_d_prev)
                  # Bearish cross: K crosses below D
                  stoch_kd_bearish_raw = (stoch_k_last < stoch_d_last) and (stoch_k_prev >= stoch_d_prev)

                  # --- Refined Cross Logic: Filter by Zone ---
                  # Only signal bullish cross if it happens *in* or *from* the oversold zone
                  # (i.e., previous K or D was below the threshold)
                  if stoch_kd_bullish_raw and (stoch_k_prev <= CONFIG.stoch_oversold_threshold or stoch_d_prev <= CONFIG.stoch_oversold_threshold):
                      stoch_kd_bullish = True
                  elif stoch_kd_bullish_raw:
                       logger.debug(f"Stoch K/D Bullish cross ({stoch_k_prev:.2f}/{stoch_d_prev:.2f} -> {stoch_k_last:.2f}/{stoch_d_last:.2f}) ignored: Occurred strictly above oversold zone ({CONFIG.stoch_oversold_threshold.normalize()}).")

                  # Only signal bearish cross if it happens *in* or *from* the overbought zone
                  # (i.e., previous K or D was above the threshold)
                  if stoch_kd_bearish_raw and (stoch_k_prev >= CONFIG.stoch_overbought_threshold or stoch_d_prev >= CONFIG.stoch_overbought_threshold):
                       stoch_kd_bearish = True
                  elif stoch_kd_bearish_raw:
                       logger.debug(f"Stoch K/D Bearish cross ({stoch_k_prev:.2f}/{stoch_d_prev:.2f} -> {stoch_k_last:.2f}/{stoch_d_last:.2f}) ignored: Occurred strictly below overbought zone ({CONFIG.stoch_overbought_threshold.normalize()}).")

             except (InvalidOperation, TypeError, IndexError) as e: # Add IndexError for iloc[-2]
                  logger.warning(f"Error calculating Stoch K/D cross: {e}. Cross signals will be False.", exc_info=False) # Don't need full traceback usually
                  stoch_kd_bullish = False
                  stoch_kd_bearish = False
        else:
             logger.debug(f"Not enough valid data points ({len(k_series_valid)} K, {len(d_series_valid)} D available after NaN drop) for Stoch K/D cross calculation (need >= 2).")


        indicators_out = {
            "fast_ema": latest_fast_ema,
            "slow_ema": latest_slow_ema,
            "trend_ema": latest_trend_ema,
            "stoch_k": latest_stoch_k,
            "stoch_d": latest_stoch_d,
            "atr": latest_atr,
            "atr_period": CONFIG.atr_period, # Store period for display/context
            "stoch_kd_bullish": stoch_kd_bullish, # Add cross signals
            "stoch_kd_bearish": stoch_kd_bearish # Add cross signals
        }

        # Check if any crucial indicator calculation failed (returned NaN default)
        critical_indicators = ['fast_ema', 'slow_ema', 'trend_ema', 'atr', 'stoch_k'] # Stoch D is less critical for primary signal
        # Check if the *values* in the output dict are NaN Decimals
        failed_indicators = [key for key in critical_indicators if isinstance(indicators_out.get(key), Decimal) and indicators_out[key].is_nan()]

        if failed_indicators:
             logger.error(f"{Fore.RED}One or more critical indicators failed to calculate (NaN): {', '.join(failed_indicators)}. Cannot proceed with trading logic based on these indicators.")
             # Return None to indicate critical failure in indicator calculation
             return None

        logger.info(Fore.GREEN + "Indicator patterns woven successfully.")
        return indicators_out

    except Exception as e:
        logger.error(Fore.RED + f"Failed to weave indicator patterns: {e}", exc_info=True)
        return None

def get_current_position(symbol: str) -> Optional[Dict[str, Dict[str, Any]]]:
    \"\"\"Fetch current positions using retry wrapper, returning quantities and prices as Decimals.\"\"\"
    # No assignment, no global needed
    logger.info(Fore.CYAN + f"# Consulting position spirits for {symbol}...")

    if EXCHANGE is None or MARKET_INFO is None:
         logger.error("Exchange object or Market Info not available for fetching positions.")
         return None

    # Initialize with Decimal zero/NaN for clarity
    # Ensure keys match expected Bybit V5 position structure for consistency
    pos_dict = {
        "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "unrealizedPnl": Decimal("NaN"), "positionIdx": None},
        "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "unrealizedPnl": Decimal("NaN"), "positionIdx": None}
    }

    positions_data = None
    try:
        # fetch_with_retries handles category param automatically
        # fetch_positions for V5 Unified requires category.
        # Pass the market ID (exchange-specific) in params for Bybit V5 specifics.
        # CCXT fetch_positions expects unified symbol string list as first arg
        fetch_pos_params = {'symbol': MARKET_INFO['id']} # Category added by wrapper
        positions_data = fetch_with_retries(EXCHANGE.fetch_positions, symbols=[symbol], params=fetch_pos_params)

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
        # Ensure pos is a dictionary and has basic structure
        if not isinstance(pos, dict) or 'symbol' not in pos or 'info' not in pos or not isinstance(pos['info'], dict):
            logger.warning(f"Skipping invalid position data format: {pos}")
            continue

        # Check if the position belongs to the configured symbol (using unified symbol string)
        pos_symbol_unified = pos.get('symbol')
        if pos_symbol_unified != symbol:
            logger.debug(f"Ignoring position data for different symbol: {pos_symbol_unified}")
            continue

        pos_info = pos['info'] # Use the extracted info dict

        # Determine side ('long' or 'short') - check unified field first, then 'info'
        # Bybit V5 'side' field in the list response might be 'Buy'/'Sell' or 'None' (for one-way mode).
        # CCXT unified 'side' field should normalize this to 'long'/'short' or None.
        side = pos.get("side") # CCXT unified field ('long'/'short' or None)
        position_idx_raw = pos_info.get("positionIdx") # V5 one-way = 0, hedge long = 1, hedge short = 2

        # If CCXT unified 'side' is None (common in one-way mode), determine from info or quantity sign
        if side is None:
            side_raw_info = pos_info.get("side", "") # Bybit V5 info['side'] uses 'Buy'/'Sell' or 'None'
            if side_raw_info == "Buy": side = "long"
            elif side_raw_info == "Sell": side = "short"
            # If side is still None (e.g., info['side'] was 'None'), rely on quantity sign below

        # Get quantity ('contracts' or 'size') - Use unified field first, fallback to info
        contracts_raw = pos.get("contracts") # CCXT unified field
        if contracts_raw is None:
            contracts_raw = pos_info.get("size") # Common Bybit V5 field in 'info'

        # --- Convert Quantity to Decimal and Determine Side (if needed) ---
        if contracts_raw is not None:
            try:
                contracts = Decimal(str(contracts_raw))

                # If side is still None (e.g., from one-way mode response), determine from quantity sign
                if side is None:
                    if contracts > CONFIG.position_qty_epsilon: side = "long"
                    elif contracts < -CONFIG.position_qty_epsilon: side = "short" # Check negative for short
                    else:
                         logger.debug(f"Position entry has zero/negligible quantity ({contracts.normalize()}) and ambiguous side. Skipping.")
                         continue # Effectively flat, skip

                if side not in ["long", "short"]:
                     # This case should be rare after the above checks, but defensive
                     logger.debug(f"Could not definitively determine long/short side from quantity ({contracts.normalize()}) or info fields after all checks. Skipping position.")
                     continue # Skip processing this entry

                # Use epsilon check for effectively zero positions *after* determining side
                if contracts.copy_abs() < CONFIG.position_qty_epsilon:
                    logger.debug(f"Ignoring effectively zero size {side} position for {symbol} (Qty: {contracts.normalize()}).")
                    continue # Skip processing this entry

                # If we already processed a non-zero quantity for this side, skip this entry
                # This handles cases where API might return multiple entries for the same position side
                if (side == "long" and found_non_zero_long) or \\
                   (side == "short" and found_non_zero_short):
                     logger.debug(f"Already processed a non-zero {side} position for {symbol}. Skipping subsequent entries for this side.")
                     continue # Skip processing this entry

                # --- Parse Other Fields (Entry Price, Liq, PnL) ---
                # Get entry price - Use unified field first, fallback to info
                entry_price_raw = pos.get("entryPrice") # CCXT unified field
                if entry_price_raw is None:
                    entry_price_raw = pos_info.get("avgPrice", pos_info.get("entryPrice"))

                # Get Liq Price and PnL (rely on unified fields if available, fallback to info)
                liq_price_raw = pos.get("liquidationPrice") # CCXT unified field
                if liq_price_raw is None:
                    liq_price_raw = pos_info.get("liqPrice")

                pnl_raw = pos.get("unrealizedPnl") # CCXT unified field
                if pnl_raw is None:
                     pnl_raw = pos_info.get("unrealisedPnl", pos_info.get("unrealizedPnl"))

                # --- Convert Other Fields to Decimal ---
                try: entry_price = Decimal(str(entry_price_raw)) if entry_price_raw is not None and str(entry_price_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: entry_price = Decimal("NaN"); logger.warning(f"Could not parse {side} entry price: '{entry_price_raw}'")

                try: liq_price = Decimal(str(liq_price_raw)) if liq_price_raw is not None and str(liq_price_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: liq_price = Decimal("NaN"); logger.warning(f"Could not parse {side} liq price: '{liq_price_raw}'")

                try: pnl = Decimal(str(pnl_raw)) if pnl_raw is not None and str(pnl_raw).strip() != '' else Decimal("NaN")
                except InvalidOperation: pnl = Decimal("NaN"); logger.warning(f"Could not parse {side} pnl: '{pnl_raw}'")


                # --- Assign to the Dictionary ---
                # Store the absolute quantity, the side is already determined
                pos_dict[side]["qty"] = contracts.copy_abs()
                pos_dict[side]["entry_price"] = entry_price
                pos_dict[side]["liq_price"] = liq_price
                pos_dict[side]["unrealizedPnl"] = pnl # Use the standardized key name
                pos_dict[side]["positionIdx"] = position_idx_raw # Store positionIdx for debugging/context

                # Mark side as found (only the first non-zero entry per side)
                if side == "long": found_non_zero_long = True
                else: found_non_zero_short = True

                # Log with formatted decimals (for display)
                entry_log = f"{entry_price.normalize()}" if not entry_price.is_nan() else "N/A"
                liq_log = f"{liq_price.normalize()}" if not liq_price.is_nan() else "N/A"
                pnl_log = f"{pnl:+.4f}" if not pnl.is_nan() else "N/A"
                logger.info(Fore.YELLOW + f"Found active {side.upper()} position: Qty={contracts.copy_abs().normalize()}, Entry={entry_log}, Liq≈{liq_log}, PnL≈{pnl_log} (posIdx: {position_idx_raw})")


            except (InvalidOperation, TypeError) as e:
                 logger.error(f"Could not parse position data for {side} side: Qty='{contracts_raw}'. Error: {e}", exc_info=True)
                 continue # Continue to the next position entry

        elif contracts_raw is None:
             logger.debug(f"Position entry for {pos_symbol_unified} has no contracts/size field. Skipping.")


    if not found_non_zero_long and not found_non_zero_short:
         logger.info(Fore.BLUE + f"No active non-zero positions reported by exchange for {symbol}.")
         # If no positions found, ensure tracker is cleared as a fallback
         # Need global as we are assigning to order_tracker (in case it held stale state)
         global order_tracker
         if order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]:
             logger.info("Position fetch returned flat state, clearing local order tracker.")
             order_tracker["long"] = {"sl_id": None, "tsl_id": None}
             order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    elif found_non_zero_long and found_non_zero_short:
         # This indicates hedge mode or an issue. Bot assumes one-way.
         logger.warning(Fore.YELLOW + f"Both LONG and SHORT positions found for {symbol}. Pyrmethus assumes one-way mode and will manage based on the first non-zero entry found for each side. Ensure your exchange account is configured for one-way trading.")
         # The pos_dict will contain the first non-zero quantity found for each side.

    logger.info(Fore.GREEN + "Position spirits consulted.")
    return pos_dict

def get_balance(currency: str) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    \"\"\"Fetches the free and total balance for a specific currency using retry wrapper, returns Decimals.\"\"\"
    # No assignment, no global needed
    logger.info(Fore.CYAN + f"# Querying the Vault of {currency}...")

    if EXCHANGE is None:
        logger.error("Exchange object not available for fetching balance.")
        return None, None

    balance_data = None
    try:
        # Bybit V5 fetch_balance requires category for UNIFIED/CONTRACT accounts.
        # fetch_with_retries adds category automatically from CONFIG.market_type.
        # For UNIFIED, balance is account-wide (usually in USDT/USDC).
        # For CONTRACT, balance is per-coin. Passing 'coin' in params might be needed.
        # Let's try the generic fetch first, which should work for UNIFIED.
        fetch_params = {} # Category added by fetch_with_retries
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance, params=fetch_params)

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
        # --- Bybit V5 Specific Parsing (Prioritized) ---
        # V5 structure: info -> result -> list -> [account objects]
        account_type_found = None
        if 'info' in balance_data and isinstance(balance_data['info'], dict):
            info_data = balance_data['info']
            if 'result' in info_data and isinstance(info_data['result'], dict) and \\
               'list' in info_data['result'] and isinstance(info_data['result']['list'], list):
                account_list = info_data['result']['list']

                # --- Look for UNIFIED Account First ---
                unified_account = next((acc for acc in account_list if acc.get('accountType') == 'UNIFIED'), None)
                if unified_account:
                     account_type_found = 'UNIFIED'
                     equity_raw = unified_account.get('totalEquity')
                     free_raw = unified_account.get('totalAvailableBalance')
                     margin_balance_raw = unified_account.get('totalMarginBalance') # Another potential indicator

                     if equity_raw is not None:
                         try: total_balance = Decimal(str(equity_raw))
                         except InvalidOperation: logger.warning(f"Could not parse UNIFIED totalEquity: '{equity_raw}'")
                     if free_raw is not None:
                          try: free_balance = Decimal(str(free_raw))
                          except InvalidOperation: logger.warning(f"Could not parse UNIFIED totalAvailableBalance: '{free_raw}'")
                     # If equity is missing, try margin balance as fallback
                     if total_balance.is_nan() and margin_balance_raw is not None:
                          try:
                               total_balance = Decimal(str(margin_balance_raw))
                               logger.debug("Using totalMarginBalance as fallback for UNIFIED equity.")
                          except InvalidOperation: logger.warning(f"Could not parse UNIFIED totalMarginBalance: '{margin_balance_raw}'")

                     logger.debug(f"Parsed Bybit V5 UNIFIED info: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Equity={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")

                # --- If No UNIFIED, Look for CONTRACT Account (Matching Configured Currency) ---
                elif not account_type_found: # Only check if UNIFIED wasn't found
                    contract_account = next((acc for acc in account_list if acc.get('accountType') == 'CONTRACT'), None)
                    if contract_account:
                         account_type_found = 'CONTRACT'
                         logger.debug(f"Found CONTRACT account. Looking for currency {currency} within its 'coin' list...")
                         # CONTRACT account balance is nested within 'coin' list
                         coin_list = contract_account.get('coin', [])
                         target_coin_info = next((coin_data for coin_data in coin_list if isinstance(coin_data, dict) and coin_data.get('coin') == currency), None)

                         if target_coin_info:
                              equity_raw = target_coin_info.get('equity')
                              free_raw = target_coin_info.get('availableToWithdraw', target_coin_info.get('availableBalance')) # Check alternative keys

                              if equity_raw is not None:
                                   try: total_balance = Decimal(str(equity_raw))
                                   except InvalidOperation: logger.warning(f"Could not parse CONTRACT equity for {currency}: '{equity_raw}'")
                              if free_raw is not None:
                                   try: free_balance = Decimal(str(free_raw))
                                   except InvalidOperation: logger.warning(f"Could not parse CONTRACT available for {currency}: '{free_raw}'")

                              logger.debug(f"Parsed Bybit V5 CONTRACT info for {currency}: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Equity={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")
                         else:
                              logger.warning(f"CONTRACT account found, but currency {currency} details not present in 'coin' list.")
                    else:
                         logger.warning("Neither UNIFIED nor CONTRACT account type found in V5 response list.")
            else:
                 logger.debug("Bybit V5 info structure not as expected ('result'/'list' missing). Falling back to unified structure.")
        else:
             logger.debug("No 'info' dictionary in balance data. Falling back to unified structure.")


        # --- Fallback to CCXT Unified Structure ---
        # Only use fallback if V5 parsing didn't yield a valid total_balance (equity)
        if total_balance.is_nan():
            logger.debug(f"Attempting fallback parsing using CCXT unified balance structure for currency {currency}...")
            if currency in balance_data and isinstance(balance_data[currency], dict):
                currency_balance = balance_data[currency]
                free_raw = currency_balance.get('free')
                total_raw = currency_balance.get('total') # 'total' usually represents equity in futures

                # Only overwrite if the value was not found via V5 parsing
                if free_balance.is_nan() and free_raw is not None:
                     try: free_balance = Decimal(str(free_raw))
                     except InvalidOperation: logger.warning(f"Could not parse unified free balance for {currency}: '{free_raw}'")
                if total_balance.is_nan() and total_raw is not None:
                     try: total_balance = Decimal(str(total_raw))
                     except InvalidOperation: logger.warning(f"Could not parse unified total balance for {currency}: '{total_raw}'")

                logger.debug(f"Parsed CCXT unified structure for {currency}: Free={free_balance.normalize() if not free_balance.is_nan() else 'N/A'}, Total={total_balance.normalize() if not total_balance.is_nan() else 'N/A'}")
            elif not account_type_found: # Only log this if V5 parsing also failed to find an account
                logger.warning(f"Balance data for {currency} not found in unified structure either.")


        # --- Final Check and Return ---
        if free_balance.is_nan():
             logger.warning(f"Could not find or parse FREE balance for {currency} after all parsing attempts.")
        if total_balance.is_nan():
             logger.warning(f"Could not find or parse TOTAL/EQUITY balance for {currency} after all parsing attempts.")
             # Critical if equity is needed for risk calc
             logger.error(Fore.RED + "Failed to determine account equity. Cannot proceed safely.")
             # Return NaN free balance if available, but None for equity to signal critical failure
             return free_balance if not free_balance.is_nan() else None, None

        # Use 'total' balance (Equity) as the primary value for risk calculation
        equity = total_balance

        logger.info(Fore.GREEN + f"Vault contains {free_balance.normalize() if not free_balance.is_nan() else 'N/A'} free {currency} (Equity/Total: {equity.normalize()}).")
        return free_balance, equity # Return free and total (equity)

    except (InvalidOperation, TypeError, KeyError) as e:
         logger.error(Fore.RED + f"Error parsing balance data for {currency}: {e}. Raw balance data sample: {str(balance_data)[:500]}", exc_info=True)
         return None, None # Indicate parsing failure
    except Exception as e:
        logger.error(Fore.RED + f"Unexpected shadow encountered querying vault: {e}", exc_info=True)
        return None, None

# --- Order and State Management ---

def check_order_status(order_id: str, symbol: str, timeout: int) -> Optional[Dict]:
    \"\"\"Checks order status with retries and timeout. Returns the final order dict or None.\"\"\"
    # No assignment, no global needed
    logger.info(Fore.CYAN + f"Verifying final status of order {order_id} for {symbol} (Timeout: {timeout}s)...")
    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(Fore.RED + "Exchange object or Market Info not loaded for checking order status.")
        return None

    start_time = time.time()
    last_status = 'unknown'
    last_bybit_status = 'unknown'
    attempt = 0
    check_interval = 1.5 # seconds between checks

    while time.time() - start_time < timeout: # Loop while time remains
        if shutdown_requested:
             logger.warning("Shutdown requested, aborting status check for order {order_id}.")
             return None # Abort status check on shutdown

        attempt += 1
        logger.debug(f"Checking order {order_id}, attempt {attempt}...")
        order_status_data = None
        try:
            # Use fetch_with_retries for the underlying fetch_order call
            # Bybit V5 fetch_order requires category AND orderId/clientOrderId.
            # Ensure category is in params (handled by wrapper)
            # Use market ID for symbol parameter in params.
            # CCXT fetch_order expects unified symbol string as the second argument.
            fetch_order_params = {'symbol': MARKET_INFO['id']} # Category added by wrapper
            order_status_data = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=fetch_order_params)

            if order_status_data and isinstance(order_status_data, dict):
                last_status = order_status_data.get('status', 'unknown') # CCXT status
                last_bybit_status = order_status_data.get('info', {}).get('orderStatus', 'N/A') # Bybit specific status

                filled_qty_raw = order_status_data.get('filled', 0.0)
                filled_qty = Decimal(str(filled_qty_raw)) if filled_qty_raw is not None else Decimal('0.0') # Convert to Decimal

                logger.debug(f"Order {order_id} status check: CCXT '{last_status}' (Bybit V5 '{last_bybit_status}'), Filled: {filled_qty.normalize()}")

                # --- Check for Terminal States ---
                # CCXT states: 'closed' (filled), 'canceled', 'rejected', 'expired'
                # Bybit V5 states: 'Filled', 'Cancelled', 'Rejected', 'PartiallyFilledCanceled', 'Deactivated'
                is_terminal_ccxt = last_status in ['closed', 'canceled', 'rejected', 'expired']
                is_terminal_bybit = last_bybit_status in ['Filled', 'Cancelled', 'Rejected', 'PartiallyFilledCanceled', 'Deactivated']

                if is_terminal_ccxt or is_terminal_bybit:
                    # If 'closed' or 'Filled', ensure filled quantity is significant (for market orders)
                    if (last_status == 'closed' or last_bybit_status == 'Filled') and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                         logger.info(Fore.GREEN + f"Order {order_id} confirmed FILLED (Status: CCXT '{last_status}', Bybit '{last_bybit_status}').")
                         return order_status_data # Return the final order dict (success)
                    # Handle cases where it's 'closed' but zero fill (e.g., rejected IOC)
                    elif (last_status == 'closed' or last_bybit_status == 'Filled') and filled_qty.copy_abs() < CONFIG.position_qty_epsilon:
                         logger.warning(f"Order {order_id} has terminal status '{last_status}'/'{last_bybit_status}' but negligible fill ({filled_qty.normalize()}). Treating as failed.")
                         # Modify status locally if needed for consistency? Maybe return as is.
                         return order_status_data # Return the final order dict (failure to fill)
                    else: # Other terminal failure states
                         logger.warning(f"Order {order_id} reached terminal failure state: CCXT '{last_status}', Bybit '{last_bybit_status}'. Filled: {filled_qty.normalize()}.")
                         return order_status_data # Return the final order dict (failure)

                # Check for 'open'/'PartiallyFilled' but fully filled (can happen briefly)
                remaining_qty_raw = order_status_data.get('remaining', 0.0)
                remaining_qty = Decimal(str(remaining_qty_raw)) if remaining_qty_raw is not None else Decimal('0.0')
                if (last_status == 'open' or last_bybit_status == 'PartiallyFilled') and remaining_qty.copy_abs() < CONFIG.position_qty_epsilon and filled_qty.copy_abs() >= CONFIG.position_qty_epsilon:
                    logger.info(f"Order {order_id} is '{last_status}'/'{last_bybit_status}' but fully filled ({filled_qty.normalize()}). Treating as 'closed'/'Filled'.")
                    order_status_data['status'] = 'closed' # Update status locally for clarity
                    order_status_data['info']['orderStatus'] = 'Filled' # Update Bybit status too
                    return order_status_data

            else:
                # fetch_with_retries failed or returned unexpected data
                logger.warning(f"fetch_order call failed or returned invalid data for {order_id}. Continuing check loop.")
                # Continue the loop to retry check_order_status itself

        except ccxt.OrderNotFound:
            # Order is definitively not found. Likely rejected/expired immediately and purged.
            logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND by exchange. Likely rejected/expired.")
            # Synthesize a 'rejected' status for caller handling
            # Use a common Bybit rejection code if known (e.g., 110043 Order does not exist)
            return {'status': 'rejected', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Order not found (likely rejected/expired)', 'retCode': 110043, 'orderStatus': 'Rejected'}}
        except (ccxt.AuthenticationError, ccxt.PermissionDenied) as e:
            # Critical non-retryable errors - re-raise immediately
            logger.critical(Fore.RED + Style.BRIGHT + f"Authentication/Permission error during order status check for {order_id}: {e}. Halting.")
            # Request shutdown
            global shutdown_requested
            shutdown_requested = True
            sys.exit(1)
        except ccxt.ExchangeError as e:
             # Specific ExchangeErrors raised by fetch_with_retries.
             logger.error(Fore.RED + f"Exchange error during order status check for {order_id}: {e}. Stopping checks for this order.")
             # Synthesize a failure status if check failed due to exchange error
             return {'status': 'check_failed', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': f'Exchange error during status check: {e}', 'retCode': -1, 'orderStatus': 'check_failed'}}
        except Exception as e:
            # Catch-all for unexpected errors during the check itself
            logger.error(f"Unexpected error during order status check loop for {order_id}: {e}", exc_info=True)
            # Let the loop continue or timeout

        # Wait before the next check_order_status attempt if not terminal
        time_elapsed = time.time() - start_time
        if time.time() - start_time < timeout: # Check timeout again before sleeping
             # Calculate remaining time and sleep duration
             remaining_time = timeout - time_elapsed
             sleep_duration = min(check_interval, remaining_time)
             if sleep_duration > 0 and not shutdown_requested: # Only sleep if time remains and not shutting down
                 logger.debug(f"Order {order_id} status ({last_status}/{last_bybit_status}) not terminal/filled. Sleeping {sleep_duration:.1f}s...")
                 time.sleep(sleep_duration)
                 check_interval = min(check_interval * 1.2, 5) # Slightly increase interval up to 5s
             elif shutdown_requested:
                 break # Exit loop if shutdown requested during sleep calculation
             else:
                 break # Time is up
        else:
            break # Exit loop if timeout reached

    # --- Timeout Reached or Shutdown Requested ---
    if shutdown_requested:
         logger.warning(f"Shutdown requested, status check aborted for order {order_id}.")
         return None

    # --- Timeout Reached ---
    logger.error(Fore.RED + f"Timed out checking status for order {order_id} after {timeout} seconds. Last known status: CCXT '{last_status}', Bybit '{last_bybit_status}'.")
    # Attempt one final fetch outside the loop to get the very last state if possible
    final_check_status = None
    try:
        logger.info(f"Performing final status check for order {order_id} after timeout...")
        final_fetch_params = {'symbol': MARKET_INFO['id']} # Category added by wrapper
        final_check_status = fetch_with_retries(EXCHANGE.fetch_order, order_id, symbol, params=final_fetch_params)

        if final_check_status and isinstance(final_check_status, dict):
             final_status_ccxt = final_check_status.get('status', 'unknown')
             final_status_bybit = final_check_status.get('info', {}).get('orderStatus', 'N/A')
             final_filled_raw = final_check_status.get('filled', 0.0)
             final_filled = Decimal(str(final_filled_raw)) if final_filled_raw is not None else Decimal('0.0')
             logger.info(f"Final status after timeout: CCXT '{final_status_ccxt}' (Bybit V5 '{final_status_bybit}'), Filled: {final_filled.normalize()}")
             # Return this final status even if timed out earlier
             return final_check_status
        else:
             logger.error(f"Final status check for order {order_id} also failed or returned invalid data.")
             # Synthesize a failure status if check failed
             return {'status': 'check_failed', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Final status check failed after timeout', 'retCode': -1, 'orderStatus': 'check_failed'}}
    except ccxt.OrderNotFound:
        logger.error(Fore.RED + f"Order {order_id} confirmed NOT FOUND on final check.")
        # Synthesize a 'rejected' status
        return {'status': 'rejected', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': 'Order not found on final check', 'retCode': 110043, 'orderStatus': 'Rejected'}}
    except Exception as e:
        logger.error(f"Error during final status check for order {order_id}: {e}", exc_info=True)
        # Synthesize a failure status
        return {'status': 'check_failed', 'filled': Decimal('0.0'), 'remaining': Decimal('0.0'), 'id': order_id, 'info': {'retMsg': f'Exception during final status check: {e}', 'retCode': -2, 'orderStatus': 'check_failed'}}

    # If reached here, timed out and final check also failed or returned invalid data.
    logger.error(Fore.RED + f"Order {order_id} check ultimately failed after timeout and final check.")
    # Fallback return for complete failure
    return None


def log_trade_entry_to_journal(
    symbol: str, side: str, qty: Decimal, avg_price: Decimal, order_id: Optional[str]
) -> None:
    \"\"\"Appends a trade entry record to the CSV journal file.\"\"\"
    # Accessing CONFIG is fine without global here.
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
    # Use the CCXT unified symbol string for the journal
    journal_symbol = symbol # Keep the unified format for better tracking

    entry_data = {
        'createdTime': now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], # Millisecond precision UTC
        'updatedTime': now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], # Use entry time initially
        'symbol': journal_symbol,
        'category': CONFIG.market_type,
        # Determine Position_Direction based on entry side ('buy' -> 'Long', 'sell' -> 'Short')
        'Position_Direction': 'Long' if side.lower() == 'buy' else 'Short',
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
        # Specify utf-8 encoding for broader compatibility
        with open(filepath, 'a+', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Check if file is empty or newly created to write header
            csvfile.seek(0) # Go to the beginning of the file
            first_char = csvfile.read(1) # Read one character
            csvfile.seek(0, os.SEEK_END) # Go back to the end to append
            if not file_exists or not first_char: # Check if file didn't exist OR was empty
                writer.writeheader() # Write header only if file is new or empty
            writer.writerow(entry_data)
        logger.info(f"Logged trade entry to journal: {filepath}")
    except IOError as e:
        logger.error(f"{Fore.RED}Error writing entry to journal file '{filepath}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"{Fore.RED}Unexpected error writing entry to journal: {e}", exc_info=True)


def place_risked_market_order(symbol: str, side: str, risk_percentage: Decimal, atr: Decimal) -> bool:
    \"\"\"Places a market order with calculated size and initial ATR-based stop-loss, using Decimal precision.\"\"\"
    # Need global for ASSIGNING to order_tracker
    global order_tracker

    trade_action = f"{side.upper()} Market Entry"
    logger.trade(Style.BRIGHT + f"Attempting {trade_action} for {symbol}...")

    if MARKET_INFO is None or EXCHANGE is None:
        logger.error(Fore.RED + f"{trade_action} failed: Market info or Exchange not available.")
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Market info/Exchange error.")
        return False

    # --- Pre-computation & Validation ---
    # Use the settle currency from MARKET_INFO for balance checks
    quote_currency = MARKET_INFO.get('settle', 'USDT') # Fallback to USDT if settle is missing
    market_id = MARKET_INFO.get('id') # Exchange specific ID

    # Need to use the dedicated function to get balance with retries
    free_balance, total_equity = get_balance(quote_currency)
    if total_equity is None or total_equity.is_nan() or total_equity <= Decimal("0"):
        logger.error(Fore.RED + f"{trade_action} failed: Invalid, NaN, or zero account equity ({total_equity.normalize() if total_equity is not None and not total_equity.is_nan() else 'N/A'}). Cannot calculate risk capital.")
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
        # fetch_ticker expects the unified symbol string
        ticker_data = fetch_with_retries(EXCHANGE.fetch_ticker, symbol)
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
        logger.debug(f"Current ticker price: {price.normalize()} {quote_currency}") # Use normalize for debug log

        # --- Calculate Stop Loss Price ---
        sl_distance_points_raw = CONFIG.sl_atr_multiplier * atr
        # Ensure stop distance is positive after calculation
        sl_distance_points = sl_distance_points_raw.copy_abs()

        if sl_distance_points <= Decimal("0"):
             logger.error(f"{Fore.RED}{trade_action} failed: Stop distance calculation resulted in zero or negative value ({sl_distance_points.normalize()}). Check ATR ({atr.normalize()}) and multiplier ({CONFIG.sl_atr_multiplier.normalize()}).")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero SL distance.")
             return False

        if side == "buy":
            sl_price_raw = price - sl_distance_points
        else: # side == "sell"
            sl_price_raw = price + sl_distance_points

        # Format SL price according to market precision *before* using it in calculations/API call
        # Bybit V5 set-trading-stop requires the trigger price as a string.
        sl_price_str_for_api = format_price(symbol, sl_price_raw)
        # Convert back to Decimal *after* formatting for consistent internal representation and validation
        try:
            sl_price_decimal_for_validation = Decimal(sl_price_str_for_api)
        except InvalidOperation:
             logger.error(Fore.RED + f"{trade_action} failed: Formatted SL price '{sl_price_str_for_api}' is not a valid Decimal. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid formatted SL.")
             return False

        logger.debug(f"ATR: {atr.normalize()}, SL Multiplier: {CONFIG.sl_atr_multiplier.normalize()}, Raw SL Distance Points: {sl_distance_points_raw.normalize()}, Absolute SL Distance Points: {sl_distance_points.normalize()}")
        logger.debug(f"Raw SL Price: {sl_price_raw.normalize()}, Formatted SL Price for API: {sl_price_str_for_api} (Decimal: {sl_price_decimal_for_validation.normalize()})")

        # --- Sanity check SL placement relative to current price ---
        # Use a small multiple of price tick size for tolerance
        try:
            price_tick_size_raw = MARKET_INFO['precision'].get('price')
            price_tick_size = Decimal(str(price_tick_size_raw)) if price_tick_size_raw is not None else Decimal("1E-8") # Fallback tiny Decimal
        except Exception as tick_err:
            logger.warning(f"Could not determine price tick size from market info: {tick_err}. Using default: 1E-8")
            price_tick_size = Decimal("1E-8")

        # Ensure tick size is positive
        if price_tick_size <= Decimal(0):
            logger.warning(f"Price tick size ({price_tick_size.normalize()}) is zero or negative. Using default: 1E-8 for SL validation.")
            price_tick_size = Decimal("1E-8")

        tolerance_ticks = price_tick_size * Decimal('5') # Allow a few ticks tolerance

        # Ensure SL price is valid (not NaN, > 0)
        if sl_price_decimal_for_validation.is_nan() or sl_price_decimal_for_validation <= Decimal("0"):
             logger.error(Fore.RED + f"{trade_action} failed: Calculated/Formatted SL price ({sl_price_decimal_for_validation.normalize()}) is invalid. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid SL Price.")
             return False

        # Use sl_price_decimal_for_validation for comparison to price
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
        # Linear/Swap (e.g., BTC/USDT): Qty in Base = Risk (Quote) / Stop Distance (Quote / Base)
        if CONFIG.market_type in ['linear', 'swap']:
            if stop_distance_quote <= Decimal('0'):
                 logger.error(Fore.RED + f"{trade_action} failed: Stop distance is zero for linear/swap sizing. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero stop dist linear/swap.")
                 return False
            qty_raw = risk_amount_quote / stop_distance_quote
            logger.debug(f"Linear/Swap Sizing: Qty (Base) = {risk_amount_quote.normalize()} {quote_currency} / {stop_distance_quote.normalize()} {quote_currency}/Base = {qty_raw.normalize()}")

        # Inverse (e.g., BTC/USD): Qty in Contracts = Risk (Quote) / (Contract Size (Base/Contract) * Stop Distance (Quote/Base))
        elif CONFIG.market_type == 'inverse':
            if contract_size <= Decimal("0"):
                 logger.error(Fore.RED + f"{trade_action} failed: Invalid Contract Size ({contract_size.normalize()}). Cannot calculate inverse size.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid contract size.")
                 return False
            if stop_distance_quote <= Decimal("0"):
                 logger.error(Fore.RED + f"{trade_action} failed: Stop distance is zero for inverse sizing. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero stop dist inverse.")
                 return False

            denominator = contract_size * stop_distance_quote
            if denominator <= Decimal("0"):
                 logger.error(Fore.RED + f"{trade_action} failed: Inverse sizing denominator ({denominator.normalize()}) is zero or negative. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Zero denom inverse.")
                 return False

            qty_raw = risk_amount_quote / denominator
            logger.debug(f"Inverse Sizing (Contract Size = {contract_size.normalize()} Base/Contract): Qty (Contracts) = {risk_amount_quote.normalize()} {quote_currency} / ({contract_size.normalize()} Base/Contract * {stop_distance_quote.normalize()} {quote_currency}/Base) = {qty_raw.normalize()}")

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
        logger.debug(f"Raw Qty: {qty_raw.normalize()}, Formatted Qty (Rounded Down): {qty.normalize()}")

        # --- Validate Quantity Against Market Limits ---
        min_qty_str = str(MARKET_INFO['limits']['amount'].get('min')) if MARKET_INFO and 'limits' in MARKET_INFO and 'amount' in MARKET_INFO['limits'] and MARKET_INFO['limits']['amount'].get('min') is not None else "0"
        max_qty_str = str(MARKET_INFO['limits']['amount'].get('max')) if MARKET_INFO and 'limits' in MARKET_INFO and 'amount' in MARKET_INFO['limits'] and MARKET_INFO['limits']['amount'].get('max') is not None else None

        try:
            min_qty = Decimal(min_qty_str)
            max_qty = Decimal(str(max_qty_str)) if max_qty_str is not None else Decimal('Infinity')
        except InvalidOperation as lim_err:
             logger.error(Fore.RED + f"{trade_action} failed: Could not parse amount limits (Min: '{min_qty_str}', Max: '{max_qty_str}'): {lim_err}. Aborting.")
             termux_notify("Entry Failed", f"{symbol} {side.upper()}: Invalid qty limits.")
             return False

        # Use epsilon for zero check alongside minimum quantity check
        if qty < min_qty or qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.error(Fore.RED + f"{trade_action} failed: Calculated quantity ({qty.normalize()}) is zero or below minimum ({min_qty.normalize()}). Risk amount ({risk_amount_quote.normalize()}), stop distance ({stop_distance_quote.normalize()}), or equity might be too small. Cannot place order.")
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

        # --- Validate minimum cost/notional value (Bybit V5 often uses notional) ---
        min_notional_str = str(MARKET_INFO['limits'].get('notional', {}).get('min')) if MARKET_INFO and 'limits' in MARKET_INFO and MARKET_INFO['limits'].get('notional', {}).get('min') is not None else None
        min_cost_str = str(MARKET_INFO['limits'].get('cost', {}).get('min')) if MARKET_INFO and 'limits' in MARKET_INFO and MARKET_INFO['limits'].get('cost', {}).get('min') is not None else None

        estimated_notional_or_cost = qty * price # Estimate notional/cost as Qty * Current Price
        validation_passed = False
        if min_notional_str is not None:
            try:
                min_notional = Decimal(min_notional_str)
                if estimated_notional_or_cost.copy_abs() >= min_notional:
                    logger.debug(f"Passed Min Notional check ({estimated_notional_or_cost.normalize()} >= {min_notional.normalize()})")
                    validation_passed = True
                else:
                    logger.error(Fore.RED + f"{trade_action} failed: Estimated order notional value ({estimated_notional_or_cost.normalize()} {quote_currency}) is below minimum required ({min_notional.normalize()} {quote_currency}). Increase risk or equity. Cannot place order.")
                    termux_notify("Entry Failed", f"{symbol} {side.upper()}: Value < Min Notional.")
                    return False # Fail early
            except Exception as notional_err:
                 logger.warning(f"Could not validate against minimum notional limit: {notional_err}. Skipping check.", exc_info=True)
                 validation_passed = True # Skip check if validation fails
        elif min_cost_str is not None: # Fallback to min cost if notional not present
             try:
                 min_cost = Decimal(min_cost_str)
                 if estimated_notional_or_cost.copy_abs() >= min_cost:
                      logger.debug(f"Passed Min Cost check ({estimated_notional_or_cost.normalize()} >= {min_cost.normalize()})")
                      validation_passed = True
                 else:
                      logger.error(Fore.RED + f"{trade_action} failed: Estimated order cost ({estimated_notional_or_cost.normalize()} {quote_currency}) is below minimum required ({min_cost.normalize()} {quote_currency}). Increase risk or equity. Cannot place order.")
                      termux_notify("Entry Failed", f"{symbol} {side.upper()}: Cost < Min Cost.")
                      return False # Fail early
             except Exception as cost_err:
                  logger.warning(f"Could not validate against minimum cost limit: {cost_err}. Skipping check.", exc_info=True)
                  validation_passed = True # Skip check if validation fails
        else:
             logger.debug("No minimum notional or cost limit found in market info to validate against.")
             validation_passed = True # No limit to check

        if not validation_passed:
             # This should only be reached if validation was skipped due to error, but defensive check
             logger.error(Fore.RED + f"{trade_action} failed: Could not validate minimum notional/cost. Aborting.")
             return False


        logger.info(Fore.YELLOW + f"Calculated Order: Side={side.upper()}, Qty={qty.normalize()}, Entry≈{price.normalize()}, SL={sl_price_str_for_api} (ATR={atr.normalize()})")

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
    average_price = price # Initialize average_price, fallback to entry estimate

    try:
        logger.trade(f"Submitting {side.upper()} market order for {qty.normalize()} {symbol}...")
        # Bybit V5 create_market_order requires category and symbol (market ID in params).
        # Set positionIdx=0 for one-way mode.
        # CCXT create_market_order expects unified symbol string as arg.
        # CCXT handles mapping 'buy'/'sell' to 'Buy'/'Sell'.
        create_order_params = {
            'positionIdx': 0,
            'symbol': market_id # Pass market ID in params for Bybit V5
            # Category added by fetch_with_retries wrapper
            }

        # Use unified symbol string for create_market_order main argument
        order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol, # Use unified symbol string
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
        # Try to get the order ID from the unified field first, fallback to info
        order_id = order.get('id')
        if not order_id and isinstance(order.get('info'), dict):
             order_id = order['info'].get('orderId', order_id) # Check V5 specific field

        # --- Verify Order Fill (Crucial Step) ---
        fill_verified = False
        if isinstance(order.get('info'), dict):
            info_data = order['info']
            ret_code = info_data.get('retCode')
            ret_msg = info_data.get('retMsg')

            # Bybit V5 Market Order Success (retCode 0) often returns fill details immediately
            if ret_code == 0:
                 logger.debug(f"{trade_action}: Market order submission retCode 0 ({ret_msg}). Attempting to parse fill details from response.")
                 # V5 structure might vary slightly, check common fields
                 cum_exec_qty_raw = order.get('filled', info_data.get('cumExecQty')) # Check unified 'filled' first
                 avg_price_raw = order.get('average', info_data.get('avgPrice')) # Check unified 'average' first

                 if cum_exec_qty_raw is not None and avg_price_raw is not None:
                     try:
                         filled_qty_from_response = Decimal(str(cum_exec_qty_raw))
                         avg_price_from_response = Decimal(str(avg_price_raw))

                         # Check if the filled quantity is significant and close to the requested quantity
                         min_qty_str_for_fill = str(MARKET_INFO['limits']['amount'].get('min', '0'))
                         min_qty_for_fill = Decimal(min_qty_str_for_fill)
                         # Tolerance: 0.1% of requested qty + 10% of min step size (or min_qty)
                         amount_step_size = Decimal(str(MARKET_INFO['precision'].get('amount', '1E-8')))
                         fill_tolerance = (qty.copy_abs() * Decimal("0.001")) + (min(amount_step_size, min_qty_for_fill) * Decimal("0.1"))

                         if filled_qty_from_response.copy_abs() >= qty.copy_abs() - fill_tolerance and filled_qty_from_response.copy_abs() >= CONFIG.position_qty_epsilon:
                             filled_qty = filled_qty_from_response
                             average_price = avg_price_from_response
                             fill_verified = True
                             logger.trade(Fore.GREEN + Style.BRIGHT + f"Market order filled immediately (ID: {order_id}). Filled: {filled_qty.normalize()} @ {average_price.normalize()}")
                         else:
                             logger.warning(f"Market order submitted (ID: {order_id}), but filled quantity from response ({filled_qty_from_response.normalize()}) is less than expected ({qty.normalize()}). Falling back to status check.")
                             # Need to check status to be sure
                     except InvalidOperation as e:
                         logger.error(f"Failed to parse fill details (Qty/AvgPrice) from V5 response for order {order_id}: {e}. Raw: Qty={cum_exec_qty_raw}, AvgPrice={avg_price_raw}. Falling back to status check.", exc_info=True)
                 elif order_id is None:
                     # retCode 0 but no orderId found - critical issue
                     logger.error(Fore.RED + f"{trade_action} failed: Market order submitted (retCode 0) but no Order ID found in response. Cannot track or confirm fill. Aborting.")
                     termux_notify("Entry Failed", f"{symbol} {side.upper()}: No order ID in V5 response.")
                     return False
                 else:
                      # retCode 0, orderId found, but no fill details in immediate response. Needs status check.
                      logger.warning(f"Market order submitted (ID: {order_id}, retCode 0), but immediate fill details missing in response. Falling back to status check.")
            else:
                 # Non-zero retCode means submission failed
                 logger.error(Fore.RED + f"{trade_action} failed: Market order submission failed. Exchange message: {ret_msg} (Code: {ret_code})")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: Submit failed ({ret_code}).")
                 return False
        else: # Order response does not contain V5 info dict or structure differs
             logger.warning("Market order response structure differs from expected V5 'info'. Relying on standard 'id' field and check_order_status.")
             if not order_id:
                 logger.error(Fore.RED + f"{trade_action} failed: Market order response missing standard 'id' and cannot parse V5 info. Cannot track order. Aborting.")
                 termux_notify("Entry Failed", f"{symbol} {side.upper()}: No order ID in response.")
                 return False
             # Order ID is available, need to check status
             logger.info(f"Waiting {CONFIG.order_check_delay_seconds}s before checking fill status for order {order_id}...")
             if not shutdown_requested: time.sleep(CONFIG.order_check_delay_seconds)


        # --- If fill wasn't confirmed immediately, check status ---
        if not fill_verified:
             if order_id is None:
                  logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Cannot check order status, order ID is missing. Aborting.")
                  termux_notify("Entry Failed", f"{symbol} {side.upper()}: Order ID missing for check.")
                  return False

             logger.info(f"Verifying fill status via check_order_status for order {order_id}...")
             # check_order_status expects CCXT unified symbol string
             order_status_data = check_order_status(order_id, symbol, timeout=CONFIG.order_check_timeout_seconds)

             # Re-evaluate fill_verified based on status check result
             if order_status_data is not None:
                  status_ccxt = order_status_data.get('status')
                  status_bybit = order_status_data.get('info', {}).get('orderStatus')
                  is_filled = status_ccxt == 'closed' or status_bybit == 'Filled'

                  if is_filled:
                       filled_qty_status = Decimal(str(order_status_data.get('filled', '0.0')))
                       # Use average from status check if available, fallback to initial estimate
                       average_price_status_raw = order_status_data.get('average')
                       average_price_status = Decimal(str(average_price_status_raw)) if average_price_status_raw is not None else average_price # Fallback

                       # Check if the filled quantity is significant and close to the requested quantity
                       min_qty_str_for_fill = str(MARKET_INFO['limits']['amount'].get('min', '0'))
                       min_qty_for_fill = Decimal(min_qty_str_for_fill)
                       amount_step_size = Decimal(str(MARKET_INFO['precision'].get('amount', '1E-8')))
                       fill_tolerance = (qty.copy_abs() * Decimal("0.001")) + (min(amount_step_size, min_qty_for_fill) * Decimal("0.1"))

                       if filled_qty_status.copy_abs() >= qty.copy_abs() - fill_tolerance and filled_qty_status.copy_abs() >= CONFIG.position_qty_epsilon:
                            logger.trade(Fore.GREEN + Style.BRIGHT + f"Market order confirmed filled via status check (ID: {order_id}). Filled: {filled_qty_status.normalize()} @ {average_price_status.normalize()}")
                            filled_qty = filled_qty_status
                            average_price = average_price_status
                            fill_verified = True # Confirmed via status check
                       else:
                            logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} status '{status_ccxt}'/'{status_bybit}' indicates filled, but filled quantity ({filled_qty_status.normalize()}) is less than requested ({qty.normalize()}) or negligible. Aborting SL placement.")
                            fill_verified = False # Treat as failure
                  else:
                       # Order status check returned a non-filled terminal state or check failed
                       filled_qty_check = Decimal(str(order_status_data.get('filled', '0.0')))
                       logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} did not fill successfully. Final status: CCXT '{status_ccxt}', Bybit '{status_bybit}'. Filled Qty: {filled_qty_check.normalize()}. Aborting SL placement.")
                       fill_verified = False # Mark as failed

             else: # check_order_status returned None (e.g., timeout, check error)
                  logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Order {order_id} status check failed or timed out. Cannot confirm fill. Aborting SL placement.")
                  fill_verified = False # Mark as failed

             # --- Attempt Cancellation if Fill Failed and Order Might Be Stuck ---
             if not fill_verified and order_status_data is not None:
                 status_ccxt = order_status_data.get('status')
                 status_bybit = order_status_data.get('info', {}).get('orderStatus')
                 # Define non-terminal states where cancellation might be useful
                 cancellable_ccxt = ['open', 'partial'] # CCXT statuses
                 cancellable_bybit = ['New', 'PartiallyFilled', 'Untriggered'] # Bybit statuses (Untriggered for stop/limit, less likely for market)

                 if status_ccxt in cancellable_ccxt or status_bybit in cancellable_bybit:
                      try:
                           logger.warning(f"Attempting cancellation of failed/stuck order {order_id} (Status: {status_ccxt}/{status_bybit}).")
                           cancel_params = {'symbol': MARKET_INFO['id']} # Category added by wrapper
                           # cancel_order expects CCXT unified symbol string
                           fetch_with_retries(EXCHANGE.cancel_order, order_id, symbol, params=cancel_params)
                           logger.warning(f"Cancellation request sent for order {order_id}.")
                      except ccxt.OrderNotFound:
                           logger.warning(f"Order {order_id} already not found when attempting cancellation.")
                      except Exception as cancel_err:
                           logger.warning(f"Failed to send cancellation for stuck order {order_id}: {cancel_err}")

             if not fill_verified:
                 termux_notify("Entry Failed", f"{symbol} {side.upper()} Order not filled/verified.")
                 return False # Indicate failure of the entry process


        # --- If fill is verified, proceed to SL placement ---
        if not fill_verified:
            # This case should ideally not be reached if the logic above is correct, but defensive check.
            logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Fill verification failed after all checks for order {order_id}. Aborting.")
            termux_notify("Entry Failed", f"{symbol} {side.upper()}: Fill verify failed.")
            return False

        # --- Place Initial Stop-Loss Order (Set on Position for Bybit V5) ---
        # Use the actual filled quantity and average price for logging/journaling.
        # The SL is set on the *position* based on the *entry* logic price calculation.
        position_side = "long" if side == "buy" else "short"
        logger.trade(f"Setting initial SL for new {position_side.upper()} position (filled qty: {filled_qty.normalize()}, avg entry price: {average_price.normalize()})...")

        # Use the SL price calculated and formatted earlier
        sl_price_str_for_api_for_position = sl_price_str_for_api

        # Define parameters for setting the stop-loss on the position
        # Bybit V5 API uses "Buy" for long positions, "Sell" for short positions in set-trading-stop's 'side' parameter
        bybit_pos_side = "Buy" if position_side == "long" else "Sell"
        set_sl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': market_id, # Use exchange-specific market ID
            'stopLoss': sl_price_str_for_api_for_position, # Trigger price string
            'slTriggerBy': CONFIG.sl_trigger_by,
            'tpslMode': 'Full', # Apply to the whole position (alternative: Partial)
            'side': bybit_pos_side, # Required side param ('Buy' for Long pos, 'Sell' for Short pos)
            'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position SL: Trigger={sl_price_str_for_api_for_position}, TriggerBy={CONFIG.sl_trigger_by}, Side={set_sl_params['side']}")
        logger.debug(f"Set SL Params (for setTradingStop): {set_sl_params}")

        sl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            # Bybit V5 uses POST /v5/position/set-trading-stop
            # Check if method exists dynamically (more robust than hardcoding)
            set_trading_stop_method = getattr(EXCHANGE, 'private_post_position_set_trading_stop', None)
            if callable(set_trading_stop_method):
                # fetch_with_retries handles the call and retries
                sl_set_response = fetch_with_retries(set_trading_stop_method, params=set_sl_params)
            else:
                logger.error(Fore.RED + "Cannot set SL: CCXT method for 'set_trading_stop' not found for Bybit.")
                # Re-raise as a critical failure to trigger emergency close
                raise ccxt.NotSupported("SL setting method not available via CCXT.")

            logger.debug(f"Set SL raw response: {sl_set_response}")

            # Handle potential failure from fetch_with_retries (returns None on failure)
            if sl_set_response is None:
                 # fetch_with_retries already logged the failure
                 # Re-raise as a critical failure to trigger emergency close
                 raise ccxt.ExchangeError("Set SL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(sl_set_response, dict) and sl_set_response.get('retCode') == 0: # V5 uses retCode directly in response usually
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Stop Loss successfully set directly on the {position_side.upper()} position (Trigger: {sl_price_str_for_api_for_position}).")
                # --- Update Global State ---
                # Use a placeholder to indicate SL is active on the position
                sl_marker_id = f"POS_SL_{position_side.upper()}"
                # Use global keyword as we are ASSIGNING to order_tracker
                global order_tracker
                order_tracker[position_side] = {"sl_id": sl_marker_id, "tsl_id": None}
                logger.info(f"Updated order tracker: {order_tracker}")

                # --- Log Entry to Journal ---
                # Pass the original CCXT symbol string to the journal function
                log_trade_entry_to_journal(
                    symbol=symbol, side=side, qty=filled_qty, avg_price=average_price, order_id=order_id
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
                 if isinstance(sl_set_response, dict):
                      error_msg = sl_set_response.get('retMsg', error_msg)
                      error_code = sl_set_response.get('retCode')
                      error_msg += f" (Code: {error_code})"
                 # Re-raise as a critical failure to trigger emergency close
                 raise ccxt.ExchangeError(f"Stop loss setting failed. Exchange message: {error_msg}")

        # --- Handle SL Setting Failures (CRITICAL - Emergency Close) ---
        except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
             # This is critical - position opened but SL setting failed.
             logger.critical(Fore.RED + Style.BRIGHT + f"CRITICAL: Failed to set stop-loss on position after entry: {e}. Position is UNPROTECTED.")
             # Trigger emergency close
             close_unprotected_position(symbol, position_side, filled_qty) # Pass filled qty as fallback
             return False # Signal overall failure of the entry process

        except Exception as e:
            # Catch any other unexpected errors during SL setting
            logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error setting SL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position may be open without Stop Loss due to unexpected SL setting error.")
            # Trigger emergency close
            close_unprotected_position(symbol, position_side, filled_qty) # Pass filled qty as fallback
            return False # Signal overall failure

    # --- Handle Initial Market Order Failures (Caught by fetch_with_retries re-raising) ---
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError, ccxt.BadRequest, ccxt.PermissionDenied) as e:
        # Error placing the initial market order itself
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Exchange error placing market order: {e}")
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Exchange error ({type(e).__name__}).")
        return False
    except Exception as e:
        # Catch unexpected errors during the *entire* order placement block
        logger.error(Fore.RED + Style.BRIGHT + f"{trade_action} failed: Unexpected error during market order placement process: {e}", exc_info=True)
        termux_notify("Entry Failed", f"{symbol} {side.upper()}: Unexpected order error.")
        return False

def close_unprotected_position(symbol: str, position_side: str, fallback_qty: Decimal) -> None:
    \"\"\"Attempts to close an unprotected position via market order as an emergency measure.\"\"\"
    # Needs global for ASSIGNING to order_tracker
    global order_tracker

    logger.warning(Fore.YELLOW + Style.BRIGHT + f"Attempting emergency market closure of unprotected {position_side.upper()} position...")
    termux_notify("CRITICAL!", f"{symbol} SL failed. Attempting emergency close.")

    if EXCHANGE is None or MARKET_INFO is None:
        logger.critical(Fore.RED + Style.BRIGHT + "EMERGENCY CLOSURE FAILED: Exchange/Market Info not available. MANUAL INTERVENTION REQUIRED!")
        termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} CLOSURE FAILED! Exchange N/A! Close manually!")
        return

    market_id = MARKET_INFO['id']
    emergency_close_qty = Decimal("0.0")

    try:
        # 1. Re-fetch position quantity to be certain
        logger.debug("Re-fetching current position quantity for emergency closure...")
        current_positions_emergency = get_current_position(symbol)

        if current_positions_emergency and position_side in current_positions_emergency:
             pos_qty = current_positions_emergency[position_side].get('qty', Decimal("0.0"))
             if pos_qty is not None: # Check if qty is not None before taking abs
                 emergency_close_qty = pos_qty.copy_abs()

        # 2. Fallback to originally filled quantity if fetch fails or returns zero
        if emergency_close_qty.copy_abs() < CONFIG.position_qty_epsilon:
            logger.warning(f"Could not fetch current position qty for emergency close, or qty is negligible. Falling back to originally filled/intended qty ({fallback_qty.normalize()}).")
            emergency_close_qty = fallback_qty.copy_abs()

        # 3. Validate quantity (check against zero/epsilon and min order size)
        if emergency_close_qty.copy_abs() < CONFIG.position_qty_epsilon:
              logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED: Closure quantity ({emergency_close_qty.normalize()}) is negligible. Position may already be closed or initial order failed. MANUAL CHECK REQUIRED.")
              termux_notify("EMERGENCY Check!", f"{symbol} {position_side.upper()} closure skipped (negligible qty). Verify position!")
              # Reset tracker state as position is likely flat/negligible
              order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
              logger.info(f"Updated order tracker after skipping emergency close: {order_tracker}")
              return # Exit the closure function

        # Format emergency close quantity precisely using ROUND_DOWN
        close_qty_str = format_amount(symbol, emergency_close_qty, ROUND_DOWN)
        try:
            close_qty_decimal = Decimal(close_qty_str)
        except InvalidOperation:
             logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED: Invalid Decimal after formatting close quantity '{close_qty_str}'. MANUAL CLOSURE REQUIRED for {position_side.upper()} position!")
             termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & BAD QTY! Close manually!")
             # Do NOT reset tracker state here, as we don't know the position status for sure.
             return # Exit the closure function

        # Check against minimum quantity again before closing
        try:
             min_qty_close = Decimal(str(MARKET_INFO['limits']['amount']['min']))
        except (KeyError, InvalidOperation, TypeError):
             logger.warning("Could not determine minimum order quantity for emergency closure validation.")
             min_qty_close = Decimal("0") # Assume zero if unavailable

        if close_qty_decimal < min_qty_close or close_qty_decimal.copy_abs() < CONFIG.position_qty_epsilon:
             logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED: Closure quantity {close_qty_decimal.normalize()} for {position_side} position is below minimum {min_qty_close.normalize()} or zero. MANUAL CLOSURE REQUIRED!")
             termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & < MIN QTY! Close manually!")
             # Do NOT reset tracker state here.
             return # Exit the closure function

        # 4. Place the emergency closure market order
        emergency_close_side = "sell" if position_side == "long" else "buy"
        logger.trade(f"Submitting emergency {emergency_close_side.upper()} market order for {close_qty_decimal.normalize()} {symbol}...")
        emergency_close_params = {
            'reduceOnly': True,
            'positionIdx': 0, # Ensure one-way mode
            'symbol': market_id # Pass market ID for Bybit V5
            # Category added by fetch_with_retries wrapper
            }
        # Use unified symbol string for create_market_order main argument
        emergency_close_order = fetch_with_retries(
            EXCHANGE.create_market_order,
            symbol=symbol,
            side=emergency_close_side,
            amount=float(close_qty_decimal), # CCXT needs float
            params=emergency_close_params
        )

        # 5. Check response (basic check for submission success)
        if emergency_close_order and (emergency_close_order.get('id') or (isinstance(emergency_close_order.get('info'), dict) and emergency_close_order['info'].get('retCode') == 0)):
            close_id = emergency_close_order.get('id', 'N/A (retCode 0)')
            logger.trade(Fore.GREEN + Style.BRIGHT + f"Emergency closure order placed successfully: ID {close_id}. Position should be closed.")
            termux_notify("Closure Attempted", f"{symbol} emergency closure sent.")
            # Reset tracker state as position *should* be closing (best effort)
            order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
            logger.info(f"Updated order tracker after emergency close attempt: {order_tracker}")
        else:
             # Extract error message if possible
             error_msg = "Unknown error."
             error_code = None
             if isinstance(emergency_close_order, dict) and 'info' in emergency_close_order:
                 error_msg = emergency_close_order['info'].get('retMsg', error_msg)
                 error_code = emergency_close_order['info'].get('retCode')
                 error_msg += f" (Code: {error_code})"
             elif isinstance(emergency_close_order, dict): # Handle case where 'info' might be missing
                 error_msg = str(emergency_close_order)

             logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Order placement failed): {error_msg}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!")
             termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
             # Do NOT reset tracker state here.

    except Exception as close_err:
        logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED (Exception during closure attempt): {close_err}. MANUAL INTERVENTION REQUIRED for {position_side.upper()} position!", exc_info=True)
        termux_notify("EMERGENCY!", f"{symbol} {position_side.upper()} POS UNPROTECTED & CLOSURE FAILED! Manual action needed!")
        # Do NOT reset tracker state here.


def manage_trailing_stop(
    symbol: str,
    position_side: str, # 'long' or 'short'
    position_qty: Decimal,
    entry_price: Decimal,
    current_price: Decimal,
    atr: Decimal
) -> None:
    \"\"\"Manages the activation and setting of a trailing stop loss on the position, using Decimal.\"\"\"
    # Needs global for ASSIGNING to order_tracker
    global order_tracker

    logger.debug(f"Checking TSL status for {position_side.upper()} position...")

    if EXCHANGE is None or MARKET_INFO is None:
         logger.error("Exchange or Market Info not available, cannot manage TSL.")
         return

    market_id = MARKET_INFO['id']

    # --- Initial Checks ---
    if position_qty.copy_abs() < CONFIG.position_qty_epsilon or entry_price.is_nan() or entry_price <= Decimal("0"):
        # If position seems closed or invalid, ensure tracker is clear.
        # This check might be redundant if called after position re-fetch, but safe to keep.
        if order_tracker[position_side]["sl_id"] or order_tracker[position_side]["tsl_id"]:
             logger.info(f"Position {position_side} seems closed or invalid in TSL check (Qty: {position_qty.normalize()}, Entry: {entry_price.normalize() if not entry_price.is_nan() else 'N/A'}). Clearing stale order trackers.")
             order_tracker[position_side] = {"sl_id": None, "tsl_id": None}
        return # No significant position to manage TSL for

    if atr is None or atr.is_nan() or atr <= Decimal("0"):
        logger.warning(Fore.YELLOW + f"Cannot evaluate TSL activation: Invalid ATR value ({atr.normalize() if atr is not None else 'None'}).")
        return

    if current_price is None or current_price.is_nan() or current_price <= Decimal("0"):
         logger.warning(Fore.YELLOW + "Cannot evaluate TSL activation: Invalid current price.")
         return


    # --- Get Current Tracker State ---
    # Access global tracker state directly
    initial_sl_marker = order_tracker[position_side]["sl_id"]
    active_tsl_marker = order_tracker[position_side]["tsl_id"]

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
    # Requires an initial SL marker to be present to indicate the position is protected by a fixed SL.
    if not initial_sl_marker:
        # This can happen if the initial SL setting failed, or if state got corrupted.
        logger.warning(f"Cannot activate TSL for {position_side.upper()}: Initial SL protection marker is missing from tracker. Position might be unprotected or already managed externally. MANUAL CHECK ADVISED.")
        # Do NOT attempt to set a regular SL here automatically, as the state is uncertain.
        termux_notify("TSL Skipped", f"{symbol} {position_side.upper()} TSL skipped - No SL marker.")
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
        logger.trade(Fore.GREEN + Style.BRIGHT + f"Profit threshold reached for {position_side.upper()} position (Profit {profit.normalize()} > Threshold {activation_threshold_points.normalize()}). Activating TSL.")

        # --- Set Trailing Stop Loss on Position ---
        # Bybit V5 sets TSL directly on the position using specific parameters.

        # TSL distance as percentage string (e.g., "0.5" for 0.5%)
        # Quantize to a reasonable number of decimal places for percentage (e.g., 3-4)
        # Use ROUND_HALF_EVEN for percentages. Ensure positive value.
        # Bybit V5 API expects a string representing the percentage value.
        trail_percent_str = str(CONFIG.trailing_stop_percent.copy_abs().quantize(Decimal("0.001"), rounding=ROUND_HALF_EVEN).normalize())

        # Bybit V5 API uses "Buy" for long positions, "Sell" for short positions in set-trading-stop's 'side' param
        bybit_pos_side = "Buy" if position_side == "long" else "Sell"

        # Define parameters for setting the trailing stop on the position
        set_tsl_params = {
            'category': CONFIG.market_type, # Required
            'symbol': market_id, # Use exchange-specific market ID
            'trailingStop': trail_percent_str, # Trailing distance percentage (as string)
            'tpslMode': 'Full', # Apply to the whole position
            'slTriggerBy': CONFIG.tsl_trigger_by, # Trigger type for the trail
            # 'activePrice': format_price(symbol, current_price), # Optional: Price to activate trail immediately. Bybit usually activates automatically when price moves trail% favorably. Omit for default behavior.
            # Remove the fixed SL when activating TSL by setting 'stopLoss' to "" (empty string) or '0'. "" is safer.
            'stopLoss': '',
            'side': bybit_pos_side, # Required side param
            'positionIdx': 0 # Assuming one-way mode
        }
        logger.trade(f"Setting Position TSL: Trail={trail_percent_str}%, TriggerBy={CONFIG.tsl_trigger_by}, Side={set_tsl_params['side']}, Removing Fixed SL")
        logger.debug(f"Set TSL Params (for setTradingStop): {set_tsl_params}")

        tsl_set_response = None
        try:
            # Use the specific endpoint via CCXT's implicit methods
            set_trading_stop_method = getattr(EXCHANGE, 'private_post_position_set_trading_stop', None)
            if callable(set_trading_stop_method):
                # fetch_with_retries handles the call and retries
                tsl_set_response = fetch_with_retries(set_trading_stop_method, params=set_tsl_params)
            else:
                logger.error(Fore.RED + "Cannot set TSL: CCXT method for 'set_trading_stop' not found for Bybit.")
                raise ccxt.NotSupported("TSL setting method not available.")

            logger.debug(f"Set TSL raw response: {tsl_set_response}")

            # Handle potential failure from fetch_with_retries (returns None on failure)
            if tsl_set_response is None:
                 raise ccxt.ExchangeError("Set TSL request failed after retries.")

            # Check Bybit V5 response structure for success (retCode == 0)
            if isinstance(tsl_set_response, dict) and tsl_set_response.get('retCode') == 0:
                logger.trade(Fore.GREEN + Style.BRIGHT + f"Trailing Stop Loss successfully activated for {position_side.upper()} position. Trail: {trail_percent_str}%")
                # --- Update Global State ---
                tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                # Assign to global order_tracker directly
                order_tracker[position_side]["tsl_id"] = tsl_marker_id
                order_tracker[position_side]["sl_id"] = None # Remove initial SL marker
                logger.info(f"Updated order tracker: {order_tracker}")
                termux_notify("TSL Activated", f"{symbol} {position_side.upper()} TSL active ({trail_percent_str}%).")
                return # Success

            else:
                # Extract error message if possible
                error_msg = "Unknown reason."
                error_code = None
                if isinstance(tsl_set_response, dict):
                     error_msg = tsl_set_response.get('retMsg', error_msg)
                     error_code = tsl_set_response.get('retCode')
                     error_msg += f" (Code: {error_code})"

                # Check if error was due to trying to remove non-existent SL (benign, e.g., SL already hit)
                # Bybit code: 110025 = SL/TP order not found or completed
                if error_code == 110025:
                     logger.warning(f"TSL activation may have succeeded, but received code 110025 (SL/TP not found/completed) when trying to clear fixed SL. Assuming TSL is active and fixed SL was already gone.")
                     # Proceed as if successful, update tracker
                     tsl_marker_id = f"POS_TSL_{position_side.upper()}"
                     # Assign to global order_tracker directly
                     order_tracker[position_side]["tsl_id"] = tsl_marker_id
                     order_tracker[position_side]["sl_id"] = None
                     logger.info(f"Updated order tracker (assuming TSL active despite code 110025): {order_tracker}")
                     termux_notify("TSL Activated*", f"{symbol} {position_side.upper()} TSL active (check exchange).")
                     return # Treat as success for now
                else:
                    # Raise other errors
                    raise ccxt.ExchangeError(f"Failed to activate trailing stop loss. Exchange message: {error_msg}")

        # --- Handle TSL Setting Failures ---
        except (ccxt.ExchangeError, ccxt.InvalidOrder, ccxt.NotSupported, ccxt.BadRequest, ccxt.PermissionDenied) as e:
            # TSL setting failed. Initial SL marker *should* still be in the tracker if it was set initially.
            logger.error(Fore.RED + Style.BRIGHT + f"Failed to activate TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + "Position continues with initial SL (if successfully set). TSL activation failed.")
            # Do NOT clear the initial SL marker here. Do not set TSL marker.
            termux_notify("TSL Failed!", f"{symbol} TSL activation failed. Check logs/position.")
        except Exception as e:
            logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error activating TSL: {e}", exc_info=True)
            logger.warning(Fore.YELLOW + Style.BRIGHT + "Position continues with initial SL (if successfully set). Unexpected TSL error.")
            termux_notify("TSL Failed!", f"{symbol} TSL activation failed (unexpected). Check logs/position.")

    else:
        # Profit threshold not met
        sl_status_log = f"({initial_sl_marker})" if initial_sl_marker else "(None!)"
        logger.debug(f"{position_side.upper()} profit ({profit.normalize() if not profit.is_nan() else 'N/A'}) has not crossed TSL activation threshold ({activation_threshold_points.normalize()}). Keeping initial SL {sl_status_log}.")

# --- Signal Generation ---

def generate_signals(df_last_candles: pd.DataFrame, indicators: Optional[Dict[str, Union[Decimal, bool, int]]], equity: Optional[Decimal]) -> Dict[str, Union[bool, str]]:
    \"\"\"Generates trading signals based on indicator conditions, using Decimal.\"\"\"
    # Accessing CONFIG is fine without global here.
    long_signal = False
    short_signal = False
    signal_reason = "No signal - Initial State / Data Pending"

    # --- Input Validation ---
    if indicators is None:
        logger.warning("Cannot generate signals: indicators dictionary is missing or calculation failed.")
        return {"long": False, "short": False, "reason": "Indicators missing/failed"}
    if df_last_candles is None or df_last_candles.empty:
         logger.warning("Cannot generate signals: insufficient candle data.")
         return {"long": False, "short": False, "reason": "Insufficient candle data"}
    if equity is None or equity.is_nan() or equity <= Decimal(0):
        # Equity isn't strictly needed for signals, but log if invalid for context
        logger.debug("Equity is missing or invalid, signal reason context may be limited.")

    try:
        # --- Get Latest Data & Indicators ---
        # Use iloc[-1] for the most recent complete candle
        if len(df_last_candles) < 1:
             logger.warning("Cannot generate signals: DataFrame has no rows.")
             return {"long": False, "short": False, "reason": "No candle data rows"}
        latest = df_last_candles.iloc[-1]

        # Safely get current price
        current_price_float = latest.get("close") # Use .get for safety
        if current_price_float is None or pd.isna(current_price_float):
             logger.warning("Cannot generate signals: latest close price is missing or NaN.")
             return {"long": False, "short": False, "reason": "Latest price is NaN/missing"}
        current_price = Decimal(str(current_price_float))
        if current_price <= Decimal(0):
             logger.warning("Cannot generate signals: current price is zero or negative.")
             return {"long": False, "short": False, "reason": "Invalid price (<= 0)"}

        # Safely get previous candle close for ATR move check (iloc[-2])
        previous_close = Decimal("NaN")
        if len(df_last_candles) >= 2:
             prev_candle = df_last_candles.iloc[-2]
             previous_close_float = prev_candle.get("close")
             if previous_close_float is not None and not pd.isna(previous_close_float):
                  previous_close = Decimal(str(previous_close_float))
             else:
                  logger.debug("Previous close price is NaN/missing in second-to-last candle, ATR move filter might be skipped.")
        else:
             logger.debug("Not enough candles (<2) for previous close, ATR move filter might be skipped.")

        # Safely get indicator values using .get with default Decimal('NaN') or False
        k = indicators.get('stoch_k', Decimal('NaN'))
        d = indicators.get('stoch_d', Decimal('NaN')) # Keep d for context in reason string
        fast_ema = indicators.get('fast_ema', Decimal('NaN'))
        slow_ema = indicators.get('slow_ema', Decimal('NaN'))
        trend_ema = indicators.get('trend_ema', Decimal('NaN'))
        atr = indicators.get('atr', Decimal('NaN'))
        stoch_kd_bullish = indicators.get('stoch_kd_bullish', False)
        stoch_kd_bearish = indicators.get('stoch_kd_bearish', False)


        # Check if any *required* indicator is NaN (should be caught earlier, but defensive)
        required_indicators_vals = {'fast_ema': fast_ema, 'slow_ema': slow_ema, 'trend_ema': trend_ema, 'atr': atr, 'stoch_k': k}
        nan_indicators = [name for name, val in required_indicators_vals.items() if isinstance(val, Decimal) and val.is_nan()]
        if nan_indicators:
             logger.warning(f"Cannot generate signals: Required indicator(s) are NaN: {', '.join(nan_indicators)}")
             return {"long": False, "short": False, "reason": f"NaN indicator(s): {', '.join(nan_indicators)}"}


        # --- Define Conditions ---
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema

        # Loosened Trend Filter (price within +/- CONFIG.trend_filter_buffer_percent% of Trend EMA)
        trend_buffer_points = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal('100'))
        price_above_trend_loosened = current_price > trend_ema - trend_buffer_points
        price_below_trend_loosened = current_price < trend_ema + trend_buffer_points

        # Combined Stochastic Condition: Oversold OR bullish K/D cross (filtered in calculate_indicators)
        stoch_long_condition = k < CONFIG.stoch_oversold_threshold or stoch_kd_bullish
        # Combined Stochastic Condition: Overbought OR bearish K/D cross (filtered in calculate_indicators)
        stoch_short_condition = k > CONFIG.stoch_overbought_threshold or stoch_kd_bearish

        # ATR Filter: Check if the price move from the previous close is significant
        is_significant_move = True # Default to True (filter passed or disabled)
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
                  atr_move_check_reason_part = f"Move Filter Skipped (Need >=2 candles)"
                  logger.debug(atr_move_check_reason_part)
             else: # atr is 0 or NaN
                  is_significant_move = False # Cannot apply filter if ATR is bad
                  atr_move_check_reason_part = f"Move Filter Skipped (Invalid ATR: {atr.normalize()})"
                  logger.debug(atr_move_check_reason_part)
        else:
             atr_move_check_reason_part = "Move Filter OFF"
             logger.debug(atr_move_check_reason_part)


        # --- Signal Logic ---
        potential_long = ema_bullish_cross and stoch_long_condition and is_significant_move
        potential_short = ema_bearish_cross and stoch_short_condition and is_significant_move

        # Apply trend filter if enabled
        if potential_long:
            if CONFIG.trade_only_with_trend:
                if price_above_trend_loosened:
                    long_signal = True
                    reason_parts = [
                        f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                        f"Stoch({CONFIG.stoch_period}) Long (K:{k.normalize()}{'<' if k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' | KD Cross' if stoch_kd_bullish else ''})",
                        f"Price({current_price.normalize()}) > Trend({trend_ema.normalize()})-{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer",
                        atr_move_check_reason_part
                    ]
                    signal_reason = "Long Signal: " + " | ".join(filter(None, reason_parts)) # Filter out empty strings
                else:
                    trend_reason_part = f"Price({current_price.normalize()}) !> Trend({trend_ema.normalize()})-{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer"
                    reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                         f"Stoch({CONFIG.stoch_period}) Long (K:{k.normalize()}{'<' if k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' | KD Cross' if stoch_kd_bullish else ''})",
                         trend_reason_part,
                         atr_move_check_reason_part
                    ]
                    signal_reason = "Long Blocked (Trend Filter ON): " + " | ".join(filter(None, reason_parts))
            else: # Trend filter off
                long_signal = True
                reason_parts = [
                    f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bullish",
                    f"Stoch({CONFIG.stoch_period}) Long (K:{k.normalize()}{'<' if k < CONFIG.stoch_oversold_threshold else ''}{CONFIG.stoch_oversold_threshold.normalize()}{' | KD Cross' if stoch_kd_bullish else ''})",
                    atr_move_check_reason_part
                ]
                signal_reason = "Long Signal (Trend Filter OFF): " + " | ".join(filter(None, reason_parts))

        elif potential_short:
             if CONFIG.trade_only_with_trend:
                 if price_below_trend_loosened:
                     short_signal = True
                     reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                         f"Stoch({CONFIG.stoch_period}) Short (K:{k.normalize()}{'>' if k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' | KD Cross' if stoch_kd_bearish else ''})",
                         f"Price({current_price.normalize()}) < Trend({trend_ema.normalize()})+{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer",
                         atr_move_check_reason_part
                     ]
                     signal_reason = "Short Signal: " + " | ".join(filter(None, reason_parts))
                 else:
                     trend_reason_part = f"Price({current_price.normalize()}) !< Trend({trend_ema.normalize()})+{CONFIG.trend_filter_buffer_percent.normalize()}% Buffer"
                     reason_parts = [
                         f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                         f"Stoch({CONFIG.stoch_period}) Short (K:{k.normalize()}{'>' if k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' | KD Cross' if stoch_kd_bearish else ''})",
                         trend_reason_part,
                         atr_move_check_reason_part
                     ]
                     signal_reason = "Short Blocked (Trend Filter ON): " + " | ".join(filter(None, reason_parts))
             else: # Trend filter off
                 short_signal = True
                 reason_parts = [
                    f"EMA({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period}) Bearish",
                    f"Stoch({CONFIG.stoch_period}) Short (K:{k.normalize()}{'>' if k > CONFIG.stoch_overbought_threshold else ''}{CONFIG.stoch_overbought_threshold.normalize()}{' | KD Cross' if stoch_kd_bearish else ''})",
                    atr_move_check_reason_part
                 ]
                 signal_reason = "Short Signal (Trend Filter OFF): " + " | ".join(filter(None, reason_parts))

        else:
             # No signal - build detailed reason why
             reason_parts = []
             # Check each major condition group
             if ema_bullish_cross: reason_parts.append(f"EMA Bullish ({fast_ema.normalize()}/{slow_ema.normalize()})")
             elif ema_bearish_cross: reason_parts.append(f"EMA Bearish ({fast_ema.normalize()}/{slow_ema.normalize()})")
             else: reason_parts.append(f"EMA Neutral ({fast_ema.normalize()}/{slow_ema.normalize()})")

             if stoch_long_condition: reason_parts.append(f"Stoch Long Met (K:{k.normalize()})")
             elif stoch_short_condition: reason_parts.append(f"Stoch Short Met (K:{k.normalize()})")
             else: reason_parts.append(f"Stoch Neutral (K:{k.normalize()})")

             # Add trend filter status if enabled and relevant
             if CONFIG.trade_only_with_trend:
                  if not price_above_trend_loosened and ema_bullish_cross: reason_parts.append("Trend Filter Block Long")
                  elif not price_below_trend_loosened and ema_bearish_cross: reason_parts.append("Trend Filter Block Short")
                  else: reason_parts.append("Trend Filter ON")
             else:
                 reason_parts.append("Trend Filter OFF")

             if not is_significant_move: reason_parts.append(atr_move_check_reason_part) # Add ATR reason if it failed

             signal_reason = "No Signal: " + " | ".join(filter(None, reason_parts))


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

    # Ensure boolean return values
    return {"long": bool(long_signal), "short": bool(short_signal), "reason": signal_reason}

# --- Status Display ---

def print_status_panel(
    cycle: int, timestamp: Optional[pd.Timestamp], price: Optional[Decimal], indicators: Optional[Dict[str, Union[Decimal, bool, int]]],
    positions: Optional[Dict[str, Dict[str, Any]]], equity: Optional[Decimal], signals: Dict[str, Union[bool, str]],
    order_tracker_state: Dict[str, Dict[str, Optional[str]]] # Pass tracker state snapshot explicitly
) -> None:
    \"\"\"Displays the current state using a mystical status panel with Decimal precision.\"\"\"
    # Accessing CONFIG and MARKET_INFO is fine without global here.

    header_color = Fore.MAGENTA + Style.BRIGHT
    section_color = Fore.CYAN + Style.BRIGHT
    value_color = Fore.WHITE
    reset_all = Style.RESET_ALL

    print(header_color + "\\n" + "=" * 80)
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else f"{Fore.YELLOW}N/A"
    print(f" Cycle: {value_color}{cycle}{header_color} | Timestamp: {value_color}{ts_str}")
    settle_curr = MARKET_INFO.get('settle', 'Quote') if MARKET_INFO else 'Quote'
    equity_str = f"{equity.normalize()} {settle_curr}" if equity is not None and not equity.is_nan() else f"{Fore.YELLOW}N/A"
    print(f" Equity: {Fore.GREEN}{equity_str}" + reset_all)
    print(header_color + "-" * 80)

    # --- Market & Indicators ---
    # Use .get with default Decimal('NaN') for safe access to indicator values
    indicators = indicators or {} # Ensure indicators is a dict even if None was passed
    price_str = f"{price.normalize()}" if price is not None and not price.is_nan() else f"{Fore.YELLOW}N/A"
    atr = indicators.get('atr', Decimal('NaN'))
    atr_str = f"{atr.normalize()}" if not atr.is_nan() else f"{Fore.YELLOW}N/A"
    trend_ema = indicators.get('trend_ema', Decimal('NaN'))
    trend_ema_str = f"{trend_ema.normalize()}" if not trend_ema.is_nan() else f"{Fore.YELLOW}N/A"

    price_color = Fore.WHITE
    trend_desc = f"{Fore.YELLOW}Trend N/A"
    if price is not None and not price.is_nan() and not trend_ema.is_nan():
        # Calculate buffer points precisely
        trend_buffer_points = trend_ema.copy_abs() * (CONFIG.trend_filter_buffer_percent / Decimal('100'))
        if price > trend_ema + trend_buffer_points: price_color = Fore.GREEN; trend_desc = f"{price_color}(Above Trend)"
        elif price < trend_ema - trend_buffer_points: price_color = Fore.RED; trend_desc = f"{price_color}(Below Trend)"
        else: price_color = Fore.YELLOW; trend_desc = f"{price_color}(At Trend)"

    stoch_k = indicators.get('stoch_k', Decimal('NaN'))
    stoch_d = indicators.get('stoch_d', Decimal('NaN'))
    stoch_k_str = f"{stoch_k.normalize()}" if not stoch_k.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_d_str = f"{stoch_d.normalize()}" if not stoch_d.is_nan() else f"{Fore.YELLOW}N/A"
    stoch_color = Fore.YELLOW
    stoch_desc = f"{Fore.YELLOW}Stoch N/A"
    if not stoch_k.is_nan():
         if stoch_k < CONFIG.stoch_oversold_threshold: stoch_color = Fore.GREEN; stoch_desc = f"{stoch_color}Oversold (<{CONFIG.stoch_oversold_threshold.normalize()})"
         elif stoch_k > CONFIG.stoch_overbought_threshold: stoch_color = Fore.RED; stoch_desc = f"{stoch_color}Overbought (>{CONFIG.stoch_overbought_threshold.normalize()})"
         else: stoch_color = Fore.YELLOW; stoch_desc = f"{stoch_color}Neutral ({CONFIG.stoch_oversold_threshold.normalize()}-{CONFIG.stoch_overbought_threshold.normalize()})"
         # Add KD cross info if available
         if indicators.get('stoch_kd_bullish'): stoch_desc += f"{Fore.GREEN} K>D Cross"
         if indicators.get('stoch_kd_bearish'): stoch_desc += f"{Fore.RED} K<D Cross"


    fast_ema = indicators.get('fast_ema', Decimal('NaN'))
    slow_ema = indicators.get('slow_ema', Decimal('NaN'))
    fast_ema_str = f"{fast_ema.normalize()}" if not fast_ema.is_nan() else f"{Fore.YELLOW}N/A"
    slow_ema_str = f"{slow_ema.normalize()}" if not slow_ema.is_nan() else f"{Fore.YELLOW}N/A"
    ema_cross_color = Fore.WHITE
    ema_desc = f"{Fore.YELLOW}EMA N/A"
    if not fast_ema.is_nan() and not slow_ema.is_nan():
        if fast_ema > slow_ema: ema_cross_color = Fore.GREEN; ema_desc = f"{ema_cross_color}Bullish"
        elif fast_ema < slow_ema: ema_cross_color = Fore.RED; ema_desc = f"{ema_cross_color}Bearish"
        else: ema_cross_color = Fore.YELLOW; ema_desc = f"{Fore.YELLOW}Neutral"

    status_data = [
        [section_color + "Market", value_color + CONFIG.symbol, f"{price_color}{price_str}{value_color}"],
        [section_color + f"Trend EMA ({CONFIG.trend_ema_period})", f"{value_color}{trend_ema_str}{value_color}", trend_desc + reset_all],
        [section_color + f"ATR ({indicators.get('atr_period', 'N/A')})", f"{value_color}{atr_str}{value_color}", ""], # Get ATR period from indicators dict
        [section_color + f"EMA ({CONFIG.fast_ema_period}/{CONFIG.slow_ema_period})", f"{ema_cross_color}{fast_ema_str} / {slow_ema_str}{value_color}", ema_desc + reset_all],
        [section_color + f"Stoch ({CONFIG.stoch_period},{CONFIG.stoch_smooth_k},{CONFIG.stoch_smooth_d})", f"{stoch_color}{stoch_k_str} / {stoch_d_str}{value_color}", stoch_desc + reset_all],
    ]
    print(tabulate(status_data, tablefmt="fancy_grid", colalign=("left", "left", "left")))

    # --- Positions & Orders ---
    pos_avail = positions is not None
    positions = positions or {} # Ensure positions is a dict even if None
    long_pos = positions.get('long', {})
    short_pos = positions.get('short', {})

    # Safely get values, handling None or NaN Decimals
    long_qty = long_pos.get('qty', Decimal("0.0"))
    short_qty = short_pos.get('qty', Decimal("0.0"))
    long_entry = long_pos.get('entry_price', Decimal("NaN"))
    short_entry = short_pos.get('entry_price', Decimal("NaN"))
    long_pnl = long_pos.get('unrealizedPnl', Decimal("NaN"))
    short_pnl = short_pos.get('unrealizedPnl', Decimal("NaN"))
    long_liq = long_pos.get('liq_price', Decimal("NaN"))
    short_liq = short_pos.get('liq_price', Decimal("NaN"))

    # Use the passed tracker state snapshot
    order_tracker_state = order_tracker_state or {"long": {}, "short": {}} # Ensure dict
    long_sl_marker = order_tracker_state.get('long', {}).get('sl_id')
    long_tsl_marker = order_tracker_state.get('long', {}).get('tsl_id')
    short_sl_marker = order_tracker_state.get('short', {}).get('sl_id')
    short_tsl_marker = order_tracker_state.get('short', {}).get('tsl_id')


    # Determine SL/TSL status strings based on tracker and position existence
    def get_stop_status(sl_marker, tsl_marker, has_pos):
        if not has_pos:
             return f"{value_color}-" # No position, display dash
        if tsl_marker:
            # Show marker directly (placeholders are short)
            return f"{Fore.GREEN}TSL Active ({tsl_marker})"
        elif sl_marker:
            # Show marker directly
            return f"{Fore.YELLOW}SL Active ({sl_marker})"
        else:
            # No marker found in tracker, but position exists - CRITICAL
            return f"{Back.RED}{Fore.WHITE}{Style.BRIGHT} NONE (!) {reset_all}" # Highlight missing stop more strongly

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

        long_entry_str = f"{long_entry.normalize()}" if has_long_pos_panel and not long_entry.is_nan() else "-"
        short_entry_str = f"{short_entry.normalize()}" if has_short_pos_panel and not short_entry.is_nan() else "-"

        # PnL color based on value
        long_pnl_color = Fore.GREEN if not long_pnl.is_nan() and long_pnl >= 0 else Fore.RED
        short_pnl_color = Fore.GREEN if not short_pnl.is_nan() and short_pnl >= 0 else Fore.RED
        long_pnl_str = f"{long_pnl_color}{long_pnl:+.4f}{value_color}" if has_long_pos_panel and not long_pnl.is_nan() else "-"
        short_pnl_str = f"{short_pnl_color}{short_pnl:+.4f}{value_color}" if has_short_pos_panel and not short_pnl.is_nan() else "-"


        # Liq price color (usually red)
        long_liq_str = f"{Fore.RED}{long_liq.normalize()}{value_color}" if has_long_pos_panel and not long_liq.is_nan() and long_liq > Decimal('0') else "-"
        short_liq_str = f"{Fore.RED}{short_liq.normalize()}{value_color}" if has_short_pos_panel and not short_liq.is_nan() and short_liq > Decimal('0') else "-"


    position_data = [
        [section_color + "Status", Fore.GREEN + "LONG" + reset_all, Fore.RED + "SHORT" + reset_all],
        [section_color + "Quantity", f"{value_color}{long_qty_str}{value_color}", f"{value_color}{short_qty_str}{value_color}"],
        [section_color + "Entry Price", f"{value_color}{long_entry_str}{value_color}", f"{value_color}{short_entry_str}{value_color}"],
        [section_color + "Unrealized PnL", long_pnl_str + reset_all, short_pnl_str + reset_all],
        [section_color + "Liq. Price (Est.)", long_liq_str + reset_all, short_liq_str + reset_all],
        [section_color + "Active Stop", long_stop_status + reset_all, short_stop_status + reset_all],
    ]
    print(tabulate(position_data, headers="firstrow", tablefmt="fancy_grid", colalign=("left", "left", "left")))

    # --- Signals ---
    signals = signals or {"long": False, "short": False, "reason": "N/A"} # Ensure dict
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
    wrapped_reason = textwrap.fill(
        signal_reason_text,
        width=max_line_width,
        initial_indent=f"{Fore.YELLOW}{reason_prefix}",
        subsequent_indent=" " * len(reason_prefix),
        replace_whitespace=False, # Keep existing whitespace
        drop_whitespace=False # Keep leading/trailing whitespace on lines if needed
    )
    print(wrapped_reason + reset_all) # Ensure color is reset after the wrapped text

    print(header_color + "=" * 80 + reset_all)

# --- Main Trading Cycle & Loop ---

def trading_spell_cycle(cycle_count: int) -> None:
    \"\"\"Executes one cycle of the trading spell with enhanced precision and logic.\"\"\"
    # Accesses global order_tracker implicitly via called functions (manage_tsl, place_order)
    # No direct assignment here, so 'global' keyword is not strictly needed in this function itself.

    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\\n--- Starting Cycle {cycle_count} ---")
    start_time = time.time()
    # Initialize cycle status flags and data containers
    cycle_success = True # Assume success unless critical failure occurs
    df = None
    current_price = None
    last_timestamp = None
    indicators = None
    current_equity = None
    positions = None
    signals = {"long": False, "short": False, "reason": "Cycle Start"}
    # Take initial snapshot of tracker state for the panel (will be updated later)
    order_tracker_snapshot = copy.deepcopy(order_tracker)
    final_positions_for_panel = None # Will hold the most recent position state for the panel

    try:
        # 1. Fetch Market Data
        df = fetch_market_data(CONFIG.symbol, CONFIG.interval, CONFIG.ohlcv_limit)
        if df is None or df.empty:
            logger.error(Fore.RED + "Halting cycle: Market data fetch failed or returned empty.")
            raise RuntimeError("Market data fetch failed") # Use exception to break cycle flow

        # 2. Get Current Price & Timestamp from Data
        if not df.empty:
            last_candle = df.iloc[-1]
            current_price_float = last_candle.get("close")
            if current_price_float is not None and not pd.isna(current_price_float):
                current_price = Decimal(str(current_price_float))
                if current_price <= Decimal(0):
                     logger.warning(f"Latest price ({current_price.normalize()}) is zero or negative. Treating as invalid.")
                     current_price = None # Mark as invalid
            last_timestamp = df.index[-1] # Already UTC

            # Check for stale data
            if last_timestamp:
                 now_utc = pd.Timestamp.utcnow().tz_localize('UTC')
                 time_diff = now_utc - last_timestamp
                 try:
                      interval_seconds = EXCHANGE.parse_timeframe(CONFIG.interval)
                      allowed_lag = pd.Timedelta(seconds=interval_seconds * 1.5 + 60)
                      if time_diff > allowed_lag and time_diff.total_seconds() > CONFIG.loop_sleep_seconds + 5:
                           logger.warning(Fore.YELLOW + f"Market data may be stale. Last candle: {last_timestamp.strftime('%H:%M:%S')} UTC ({time_diff} ago). Allowed lag: ~{allowed_lag}")
                 except (ValueError, AttributeError):
                     logger.warning("Could not parse interval to check data staleness.")
            else:
                 logger.warning("Could not determine last timestamp from data.")
        else:
             logger.error("DataFrame became empty unexpectedly after fetch.")
             raise RuntimeError("Market data processing failed")

        if current_price is None:
             logger.error(Fore.RED + "Halting cycle: Failed to get valid current price from DataFrame.")
             raise RuntimeError("Price processing failed")

        # 3. Calculate Indicators (returns Decimals)
        indicators = calculate_indicators(df)
        if indicators is None:
            logger.error(Fore.RED + "Indicator calculation failed. Cannot proceed with signal generation or TSL management based on ATR.")
            cycle_success = False # Mark cycle as having issues, but continue to fetch state
            current_atr = Decimal('NaN') # Ensure ATR is NaN if indicators failed
        else:
            current_atr = indicators.get('atr', Decimal('NaN'))


        # 4. Get Current State (Balance & Positions as Decimals)
        quote_currency = MARKET_INFO.get('settle', 'USDT')
        free_balance, current_equity = get_balance(quote_currency)
        if current_equity is None or current_equity.is_nan() or current_equity <= Decimal("0"):
            logger.error(Fore.RED + "Failed to fetch valid current balance/equity. Cannot perform risk calculation or trading actions.")
            cycle_success = False # Critical failure for trading
            # Allow falling through to display panel if possible

        positions = get_current_position(CONFIG.symbol)
        if positions is None:
            logger.error(Fore.RED + "Failed to fetch current positions. Cannot manage state or trade.")
            cycle_success = False # Critical failure for trading
            # Initialize positions to avoid errors later in the panel display
            positions = {
                "long": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "unrealizedPnl": Decimal("NaN"), "positionIdx": None},
                "short": {"qty": Decimal("0.0"), "entry_price": Decimal("NaN"), "liq_price": Decimal("NaN"), "unrealizedPnl": Decimal("NaN"), "positionIdx": None}
            }
        final_positions_for_panel = copy.deepcopy(positions) # Initial snapshot for panel


        # --- Core Logic (Only if critical data available) ---
        can_trade_logic = cycle_success # Check if previous steps succeeded for trading logic
        if not can_trade_logic:
            signals["reason"] = "Skipped: Critical data missing (Equity/Position/Indicators)"
            logger.warning(signals["reason"])
        else:
            # Use the *current* state from `positions` dict
            active_long_pos = positions.get('long', {})
            active_short_pos = positions.get('short', {})
            active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
            active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
            active_long_entry = active_long_pos.get('entry_price', Decimal('NaN'))
            active_short_entry = active_short_pos.get('entry_price', Decimal('NaN'))

            has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
            has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
            is_flat = not has_long_pos and not has_short_pos

            # 5. Manage Trailing Stops (if position exists and indicators are valid)
            if (has_long_pos or has_short_pos) and indicators is not None and not current_price.is_nan() and not current_atr.is_nan():
                 if has_long_pos and not active_long_entry.is_nan() and active_long_entry > Decimal(0):
                     logger.debug("Managing TSL for existing LONG position...")
                     manage_trailing_stop(CONFIG.symbol, "long", active_long_qty, active_long_entry, current_price, current_atr)
                 elif has_short_pos and not active_short_entry.is_nan() and active_short_entry > Decimal(0):
                     logger.debug("Managing TSL for existing SHORT position...")
                     manage_trailing_stop(CONFIG.symbol, "short", active_short_qty, active_short_entry, current_price, current_atr)
                 else:
                     pos_side_invalid_entry = "LONG" if has_long_pos else "SHORT"
                     logger.warning(f"Cannot manage TSL for {pos_side_invalid_entry} position: Invalid entry price ({active_long_entry.normalize() if has_long_pos else active_short_entry.normalize() if has_short_pos else 'N/A'}).")
                     termux_notify("TSL Skipped", f"{symbol} {pos_side_invalid_entry} TSL skipped - Invalid entry price.")
            elif is_flat:
                 # Ensure trackers are clear if flat (belt-and-suspenders)
                 global order_tracker # Needed for assignment
                 if order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]:
                     logger.info("Position is flat, ensuring order trackers are cleared.")
                     order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                     order_tracker["short"] = {"sl_id": None, "tsl_id": None}
            # Update tracker snapshot *after* TSL management attempt
            order_tracker_snapshot = copy.deepcopy(order_tracker)


            # --- IMPORTANT: Re-fetch position state AFTER TSL management ---
            # Checks if the position was closed by TSL/SL hitting during TSL management or previous steps.
            logger.debug("Re-fetching position state after TSL management...")
            positions_after_tsl = get_current_position(CONFIG.symbol)
            if positions_after_tsl is None:
                 logger.error(Fore.RED + "Failed to re-fetch positions after TSL check. Cannot safely determine position state for entry. Cycle marked as failed.")
                 cycle_success = False
                 signals["reason"] = "Skipped: Position re-fetch failed after TSL check"
                 # Use the last known good position state for the panel
                 final_positions_for_panel = copy.deepcopy(positions) # Use state before re-fetch attempt
                 logger.warning("Using potentially stale position data for status panel due to re-fetch failure.")
            else:
                 # Update active quantities and flat status based on the re-fetched state
                 final_positions_for_panel = copy.deepcopy(positions_after_tsl) # Update panel snapshot
                 active_long_pos = positions_after_tsl.get('long', {})
                 active_short_pos = positions_after_tsl.get('short', {})
                 active_long_qty = active_long_pos.get('qty', Decimal('0.0'))
                 active_short_qty = active_short_pos.get('qty', Decimal('0.0'))
                 has_long_pos = active_long_qty.copy_abs() >= CONFIG.position_qty_epsilon
                 has_short_pos = active_short_qty.copy_abs() >= CONFIG.position_qty_epsilon
                 is_flat = not has_long_pos and not has_short_pos
                 logger.debug(f"Position Status After TSL Check: Flat = {is_flat} (Long Qty: {active_long_qty.normalize()}, Short Qty: {active_short_qty.normalize()})")

                 # Clear trackers if now flat (e.g., TSL/SL hit)
                 global order_tracker # Needed for assignment
                 if is_flat and (order_tracker["long"]["sl_id"] or order_tracker["long"]["tsl_id"] or order_tracker["short"]["sl_id"] or order_tracker["short"]["tsl_id"]):
                      logger.info("Position became flat (likely TSL/SL hit), clearing order trackers.")
                      order_tracker["long"] = {"sl_id": None, "tsl_id": None}
                      order_tracker["short"] = {"sl_id": None, "tsl_id": None}
                      order_tracker_snapshot = copy.deepcopy(order_tracker) # Update panel snapshot too


                 # 6. Generate Trading Signals (only if indicators are valid)
                 if indicators is not None:
                     # Pass last 2 candles for ATR filter calculation if enough data
                     df_for_signals = df.iloc[-2:] if len(df) >= 2 else df
                     signals_data = generate_signals(df_for_signals, indicators, current_equity)
                     signals = {"long": signals_data["long"], "short": signals_data["short"], "reason": signals_data["reason"]}
                 else:
                     signals = {"long": False, "short": False, "reason": "Skipped Signal Gen: Indicators failed"}
                     logger.warning(signals["reason"])


                 # 7. Execute Trades based on Signals (only if flat and signal exists and cycle is currently successful)
                 if is_flat and cycle_success and (signals.get("long") or signals.get("short")):
                     # Ensure ATR is valid before attempting entry
                     if current_atr.is_nan():
                          logger.error(Fore.RED + f"Cannot attempt entry: ATR is NaN. Signal reason: {signals.get('reason', '')}")
                     elif signals.get("long"):
                          logger.info(Fore.GREEN + Style.BRIGHT + f"Long signal detected! {signals.get('reason', '')}. Attempting entry...")
                          if place_risked_market_order(CONFIG.symbol, "buy", CONFIG.risk_percentage, current_atr):
                               logger.info(Fore.GREEN + f"Long entry process completed successfully for cycle {cycle_count}.")
                               # After successful entry, re-fetch state for panel accuracy
                               positions_after_entry = get_current_position(CONFIG.symbol)
                               if positions_after_entry is not None:
                                   final_positions_for_panel = copy.deepcopy(positions_after_entry)
                                   order_tracker_snapshot = copy.deepcopy(order_tracker) # Capture global state after entry
                               else: logger.warning(Fore.YELLOW + "Failed to re-fetch state after successful long entry. Panel might be slightly stale.")
                          else:
                               logger.error(Fore.RED + f"Long entry process failed for cycle {cycle_count}.")
                               cycle_success = False # Mark cycle as failed if entry fails
                               # Re-fetch state after failed entry attempt too
                               positions_after_failed_entry = get_current_position(CONFIG.symbol)
                               if positions_after_failed_entry is not None:
                                   final_positions_for_panel = copy.deepcopy(positions_after_failed_entry)
                                   order_tracker_snapshot = copy.deepcopy(order_tracker) # Capture potentially updated tracker state
                               else: logger.warning(Fore.YELLOW + "Failed to re-fetch state after failed long entry. Panel might be slightly stale.")

                     elif signals.get("short"):
                          logger.info(Fore.RED + Style.BRIGHT + f"Short signal detected! {signals.get('reason', '')}. Attempting entry.")
                          if place_risked_market_order(CONFIG.symbol, "sell", CONFIG.risk_percentage, current_atr):
                               logger.info(Fore.GREEN + f"Short entry process completed successfully for cycle {cycle_count}.")
                               # Re-fetch state for panel accuracy
                               positions_after_entry = get_current_position(CONFIG.symbol)
                               if positions_after_entry is not None:
                                   final_positions_for_panel = copy.deepcopy(positions_after_entry)
                                   order_tracker_snapshot = copy.deepcopy(order_tracker)
                               else: logger.warning(Fore.YELLOW + "Failed to re-fetch state after successful short entry. Panel might be slightly stale.")
                          else:
                               logger.error(Fore.RED + f"Short entry process failed for cycle {cycle_count}.")
                               cycle_success = False # Mark cycle as failed
                               # Re-fetch state after failed entry attempt
                               positions_after_failed_entry = get_current_position(CONFIG.symbol)
                               if positions_after_failed_entry is not None:
                                   final_positions_for_panel = copy.deepcopy(positions_after_failed_entry)
                                   order_tracker_snapshot = copy.deepcopy(order_tracker)
                               else: logger.warning(Fore.YELLOW + "Failed to re-fetch state after failed short entry. Panel might be slightly stale.")

                 elif is_flat:
                      logger.debug("Position is flat, but no trade signal generated or cycle failed previously.")
                 elif not is_flat:
                      pos_side = "LONG" if has_long_pos else "SHORT"
                      logger.debug(f"Position ({pos_side}) already open, skipping new entry signals.")
                      # Future: Add exit logic here if desired

    except RuntimeError as e:
        # Catch critical errors that halt the cycle early
        logger.critical(f"Cycle {cycle_count} halted due to critical error: {e}")
        cycle_success = False
    except Exception as e:
        # Catch any other unexpected errors during the cycle
        logger.error(f"Unexpected error during cycle {cycle_count}: {e}", exc_info=True)
        cycle_success = False


    # 8. Display Status Panel (Always attempt to display with available data)
    print_status_panel(
        cycle_count, last_timestamp, current_price, indicators,
        final_positions_for_panel, # Use the most up-to-date position state obtained
        current_equity, signals,
        order_tracker_snapshot # Use the most up-to-date tracker state obtained
    )

    end_time = time.time()
    status_log = "Complete" if cycle_success else "Completed with WARNINGS/ERRORS"
    logger.info(Fore.MAGENTA + f"--- Cycle {cycle_count} {status_log} (Duration: {end_time - start_time:.2f}s) ---")


def graceful_shutdown(signum=None, frame=None) -> None:
    \"\"\"Dispels active orders and closes open positions gracefully with precision.\"\"\"
    # Use signum and frame parameters for signal handler compatibility
    signal_name = signal.Signals(signum).name if signum else "Manual"
    logger.warning(Fore.YELLOW + Style.BRIGHT + f"\\nInitiating Graceful Shutdown Sequence (Signal: {signal_name})...")
    termux_notify("Shutdown", f"Closing orders/positions for {CONFIG.symbol}.")

    # Needs global for ASSIGNING to order_tracker and shutdown_requested
    global order_tracker
    global shutdown_requested

    if shutdown_requested: # Avoid running multiple times if signal caught repeatedly
        logger.warning("Shutdown already in progress.")
        return
    shutdown_requested = True # Set the flag

    if EXCHANGE is None or MARKET_INFO is None:
        logger.error(Fore.RED + "Exchange object or Market Info not available. Cannot perform clean shutdown.")
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Cannot perform clean shutdown - Exchange not ready.")
        return

    symbol = CONFIG.symbol
    market_id = MARKET_INFO.get('id') # Exchange specific ID

    # 1. Cancel All Cancellable Open Orders for the Symbol
    # This targets limit/stop orders if any were used (not position-based SL/TP).
    try:
        logger.info(Fore.CYAN + f"Dispelling all cancellable open orders for {symbol}...")
        # Bybit V5 cancel_all requires category and symbol (market ID) in params
        # Use fetch_with_retries for robustness
        cancel_params = {'category': CONFIG.market_type, 'symbol': market_id}
        cancel_all_method = getattr(EXCHANGE, 'private_post_order_cancel_all', None)

        if callable(cancel_all_method):
             logger.info(f"Attempting cancel_all_orders for {symbol} via specific endpoint...")
             response = fetch_with_retries(cancel_all_method, params=cancel_params)

             if response is None:
                  logger.warning(Fore.YELLOW + "Cancel all orders request failed after retries. Manual check advised.")
             elif isinstance(response, dict) and response.get('retCode') == 0:
                  logger.info(Fore.GREEN + "Cancel all command successful (retCode 0).")
             else:
                  error_msg = response.get('retMsg', 'Unknown error') if isinstance(response, dict) else str(response)
                  error_code = response.get('retCode') if isinstance(response, dict) else 'N/A'
                  logger.warning(Fore.YELLOW + f"Cancel all orders command sent, but confirmation unclear or failed: {error_msg} (Code: {error_code}). Manual check advised.")
        else:
             logger.warning("Exchange object does not have 'private_post_order_cancel_all'. Attempting generic cancel_all_orders (may not work for V5).")
             # Fallback to generic CCXT method (might not be implemented or work correctly for V5)
             try:
                 # Generic cancel_all_orders might need symbol or market_id depending on CCXT implementation
                 # Let's try passing unified symbol and market_id in params
                 cancel_all_fallback_params = {'symbol': market_id} # Category added by wrapper
                 response = fetch_with_retries(EXCHANGE.cancel_all_orders, symbol, params=cancel_all_fallback_params)
                 logger.info(f"Generic cancel_all_orders response: {response}") # Response format varies
             except ccxt.NotSupported:
                 logger.error(Fore.RED + "Generic cancel_all_orders not supported by CCXT for Bybit. Cannot automatically cancel orders this way.")
             except Exception as generic_cancel_err:
                 logger.error(f"Error attempting generic cancel_all_orders: {generic_cancel_err}")


        # 2. Clear Position-Based SL/TP (Set them to empty strings)
        # This is crucial as cancel_all does NOT affect position-based stops in V5.
        # Need to check current position state first.
        logger.info("Attempting to clear position-based SL/TP settings...")
        current_positions_shutdown = get_current_position(symbol)
        sides_to_clear = []
        if current_positions_shutdown:
            if current_positions_shutdown.get('long', {}).get('qty', Decimal(0)).copy_abs() >= CONFIG.position_qty_epsilon:
                sides_to_clear.append('long')
            if current_positions_shutdown.get('short', {}).get('qty', Decimal(0)).copy_abs() >= CONFIG.position_qty_epsilon:
                sides_to_clear.append('short')

        if not sides_to_clear:
             logger.info("No active positions found to clear SL/TP settings for.")
        else:
            for side in sides_to_clear:
                logger.info(f"Clearing SL/TP for {side.upper()} position...")
                try:
                    bybit_pos_side = "Buy" if side == "long" else "Sell"
                    clear_sl_tp_params = {
                        'category': CONFIG.market_type,
                        'symbol': market_id,
                        'stopLoss': '', # Clear SL
                        'takeProfit': '', # Clear TP (even if not used by bot, clear it)
                        'tpslMode': 'Full',
                        'side': bybit_pos_side,
                        'positionIdx': 0
                    }
                    set_trading_stop_method = getattr(EXCHANGE, 'private_post_position_set_trading_stop', None)
                    if callable(set_trading_stop_method):
                        clear_response = fetch_with_retries(set_trading_stop_method, params=clear_sl_tp_params)
                        if clear_response and clear_response.get('retCode') == 0:
                             logger.info(f"Successfully cleared SL/TP settings for {side.upper()} position.")
                        elif clear_response and clear_response.get('retCode') == 110025: # SL/TP not found (already gone)
                             logger.info(f"SL/TP already cleared or not found for {side.upper()} position (Code 110025).")
                        else:
                             error_msg = clear_response.get('retMsg', 'Unknown error') if isinstance(clear_response, dict) else str(clear_response)
                             error_code = clear_response.get('retCode') if isinstance(clear_response, dict) else 'N/A'
                             logger.warning(f"Failed to clear SL/TP for {side.upper()} position: {error_msg} (Code: {error_code})")
                    else:
                         logger.error("Cannot clear SL/TP: CCXT method for 'set_trading_stop' not found.")

                except Exception as clear_err:
                    logger.error(f"Error clearing SL/TP for {side.upper()} position: {clear_err}", exc_info=True)


        # 3. Clear Local Tracker Regardless
        logger.info("Clearing local order tracker state.")
        order_tracker["long"] = {"sl_id": None, "tsl_id": None}
        order_tracker["short"] = {"sl_id": None, "tsl_id": None}

    except Exception as e:
        # Catch unexpected errors during the cancellation/clearing phase
        logger.error(Fore.RED + Style.BRIGHT + f"Unexpected error during order/stop cancellation phase: {e}. MANUAL CHECK REQUIRED on exchange.", exc_info=True)
        termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Order/Stop cancel error. Check logs.")

    # 4. Close Open Positions
    try:
        logger.info(Fore.CYAN + f"Checking for open positions in {symbol} to close...")
        # Re-fetch positions *after* attempting cancellations and SL/TP clearing
        final_positions = get_current_position(symbol)
        if final_positions is None:
             logger.error(Fore.RED + "Failed to fetch final positions. Cannot confirm closure. MANUAL CHECK REQUIRED.")
             termux_notify("Shutdown Warning!", f"{CONFIG.symbol} Pos fetch fail. Check manually!")
             return

        closed_count = 0
        total_positions_to_close = 0
        sides_to_close = []

        if final_positions.get('long', {}).get('qty', Decimal(0)).copy_abs() >= CONFIG.position_qty_epsilon:
            total_positions_to_close += 1
            sides_to_close.append('long')
        if final_positions.get('short', {}).get('qty', Decimal(0)).copy_abs() >= CONFIG.position_qty_epsilon:
            total_positions_to_close += 1
            sides_to_close.append('short')

        if not sides_to_close:
            logger.info(Fore.GREEN + "No open positions found to close.")
        else:
            logger.info(f"Found {total_positions_to_close} open position(s) to close: {', '.join(s.upper() for s in sides_to_close)}")

            for side in sides_to_close:
                pos_data = final_positions[side]
                pos_qty = pos_data.get('qty', Decimal('0.0')).copy_abs()

                if pos_qty < CONFIG.position_qty_epsilon:
                    logger.info(f"Skipping closure for {side.upper()} position - negligible quantity ({pos_qty.normalize()}).")
                    continue

                close_side = "sell" if side == "long" else "buy"
                logger.trade(f"Attempting to close {side.upper()} position (Qty: {pos_qty.normalize()}) with market order...")

                # Format close quantity precisely using ROUND_DOWN
                close_qty_str = format_amount(symbol, pos_qty, ROUND_DOWN)
                try:
                    close_qty_decimal = Decimal(close_qty_str)
                    if close_qty_decimal.copy_abs() < CONFIG.position_qty_epsilon:
                         logger.warning(f"Formatted closure quantity for {side.upper()} is negligible ({close_qty_decimal.normalize()}). Skipping closure.")
                         continue
                    # Validate against min quantity
                    min_qty_close = Decimal(str(MARKET_INFO['limits']['amount'].get('min','0')))
                    if close_qty_decimal < min_qty_close:
                         logger.error(Fore.RED + f"Closure quantity {close_qty_decimal.normalize()} for {side.upper()} is below minimum {min_qty_close.normalize()}. MANUAL CLOSURE REQUIRED!")
                         termux_notify("EMERGENCY!", f"{symbol} {side.upper()} CLOSURE FAILED (< Min Qty)! Close manually!")
                         continue # Try closing other side if applicable
                except (InvalidOperation, KeyError, TypeError) as fmt_err:
                     logger.critical(Fore.RED + Style.BRIGHT + f"EMERGENCY CLOSURE FAILED: Error formatting/validating close quantity '{close_qty_str}' for {side.upper()}: {fmt_err}. MANUAL CLOSURE REQUIRED!")
                     termux_notify("EMERGENCY!", f"{symbol} {side.upper()} CLOSURE FAILED (Bad Qty)! Close manually!")
                     continue # Try closing other side

                try:
                    close_params = {
                        'reduceOnly': True,
                        'positionIdx': 0,
                        'symbol': market_id # Pass market ID for Bybit V5
                        # Category added by wrapper
                    }
                    # Use unified symbol string for create_market_order main arg
                    close_order = fetch_with_retries(
                        EXCHANGE.create_market_order,
                        symbol=symbol,
                        side=close_side,
                        amount=float(close_qty_decimal),
                        params=close_params
                    )

                    # Basic check for close order submission success
                    if close_order and (close_order.get('id') or (isinstance(close_order.get('info'), dict) and close_order['info'].get('retCode') == 0)):
                        close_id = close_order.get('id', 'N/A (retCode 0)')
                        logger.trade(Fore.GREEN + Style.BRIGHT + f"Position closure order for {side.upper()} placed successfully: ID {close_id}.")
                        closed_count += 1
                        # Optional: Verify closure with check_order_status or re-fetching positions again after a delay
                        if not shutdown_requested: time.sleep(CONFIG.order_check_delay_seconds) # Wait briefly
                    else:
                        # Extract error message if possible
                        error_msg = "Unknown error."
                        error_code = None
                        if isinstance(close_order, dict) and 'info' in close_order:
                            error_msg = close_order['info'].get('retMsg', error_msg)
                            error_code = close_order['info'].get('retCode')
                            error_msg += f" (Code: {error_code})"
                        elif isinstance(close_order, dict):
                             error_msg = str(close_order)

                        logger.critical(Fore.RED + Style.BRIGHT + f"POSITION CLOSURE ORDER FAILED for {side.upper()} position: {error_msg}. MANUAL CLOSURE REQUIRED!")
                        termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Close manually!")

                except Exception as close_exec_err:
                     logger.critical(Fore.RED + Style.BRIGHT + f"Exception during {side.upper()} position closure attempt: {close_exec_err}. MANUAL CLOSURE REQUIRED!", exc_info=True)
                     termux_notify("EMERGENCY!", f"{symbol} {side.upper()} POS CLOSURE FAILED! Close manually!")

            # Final status message
            if closed_count == total_positions_to_close and total_positions_to_close > 0:
                logger.info(Fore.GREEN + Style.BRIGHT + f"All {closed_count} identified position(s) successfully closed.")
                termux_notify("Shutdown Complete", f"{symbol} Positions closed.")
            elif total_positions_to_close > 0:
                logger.warning(Fore.YELLOW + Style.BRIGHT + f"Attempted to close {total_positions_to_close} position(s), but only {closed_count} closure orders were placed successfully. MANUAL CHECK REQUIRED!")
                termux_notify("Shutdown Warning!", f"{symbol} Some pos closures failed! Check manually!")

    except Exception as e:
        logger.critical(Fore.RED + Style.BRIGHT + f"Unexpected error during position closure phase: {e}. MANUAL CHECK REQUIRED!", exc_info=True)
        termux_notify("EMERGENCY!", f"{symbol} UNEXPECTED SHUTDOWN ERROR! Check position manually!")

    logger.warning(Fore.YELLOW + Style.BRIGHT + "Graceful shutdown sequence finished. Pyrmethus is fading...")
    # sys.exit(0) # Exit cleanly after attempting shutdown

def main_loop() -> None:
    """The main loop orchestrating the trading spell."""
    # Needs global for shutdown_requested check
    global shutdown_requested
    cycle_count = 0

    # --- Initial Check: Clear Stale Trackers ---
    # Before the loop starts, check initial position state and clear trackers if flat.
    logger.info("Performing initial position check to clear stale trackers...")
    initial_positions = get_current_position(CONFIG.symbol)
    if initial_positions:
        initial_long_qty = initial_positions.get('long', {}).get('qty', Decimal('0.0'))
        initial_short_qty = initial_positions.get('short', {}).get('qty', Decimal('0.0'))
        is_initially_flat = (initial_long_qty.copy_abs() < CONFIG.position_qty_epsilon and
                             initial_short_qty.copy_abs() < CONFIG.position_qty_epsilon)
        if is_initially_flat:
            logger.info("Initial position is flat. Clearing any potentially stale order trackers.")
            # Need global for assignment
            global order_tracker
            order_tracker["long"] = {"sl_id": None, "tsl_id": None}
            order_tracker["short"] = {"sl_id": None, "tsl_id": None}
        else:
             # If not flat initially, warn user to check stops as bot doesn't know initial SL state
             logger.warning(Fore.YELLOW + "Initial position found. Pyrmethus does not know the state of existing stops for this position. It will manage TSL activation based on configured rules if profit threshold is met, potentially overwriting existing stops. MANUAL CHECK ADVISED.")
             # Don't clear trackers if not flat.
    else:
        logger.warning("Failed to fetch initial position state. Trackers remain as initialized.")


    while not shutdown_requested:
        cycle_count += 1
        try:
            trading_spell_cycle(cycle_count)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Initiating graceful shutdown...")
            graceful_shutdown(signal.SIGINT) # Call shutdown with signal number
            break # Exit loop after initiating shutdown
        except Exception as e:
            # Catch unexpected errors in the main loop itself
            logger.critical(f"CRITICAL UNHANDLED ERROR in main loop: {e}", exc_info=True)
            termux_notify("Pyrmethus CRITICAL", f"Unhandled loop error: {e}")
            # Attempt graceful shutdown before potentially dying
            logger.warning("Attempting graceful shutdown due to critical loop error...")
            graceful_shutdown(None) # Call shutdown manually
            # Optionally re-raise or exit after attempting shutdown
            sys.exit(1) # Exit with error status after critical failure

        # Check shutdown flag again before sleeping
        if shutdown_requested:
            logger.info("Shutdown requested, exiting main loop.")
            break

        # Wait before the next cycle
        try:
            logger.debug(f"Sleeping for {CONFIG.loop_sleep_seconds} seconds...")
            time.sleep(CONFIG.loop_sleep_seconds)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt during sleep. Initiating graceful shutdown...")
            graceful_shutdown(signal.SIGINT) # Call shutdown with signal number
            break # Exit loop immediately

    logger.info("Main loop finished.")


if __name__ == "__main__":
    print(Fore.GREEN + Style.BRIGHT + """
 ██████╗ ██╗   ██╗██████╗ ███╗   ███╗███████╗ ██████╗ █████╗ ██╗     ██████╗
 ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗ ████║██╔════╝██╔════╝██╔══██╗██║     ██╔══██╗
 ██████╔╝ ╚████╔╝ ██████╔╝██╔████╔██║███████╗██║     ███████║██║     ██████╔╝
 ██╔═══╝   ╚██╔╝  ██╔══██╗██║╚██╔╝██║╚════██║██║     ██╔══██║██║     ██╔═══╝
 ██║        ██║   ██║  ██║██║ ╚═╝ ██║███████║╚██████╗██║  ██║███████╗██║
 ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝
    """ + Fore.YELLOW + "Pyrmethus - Termux Trading Spell v2.3.1" + Style.RESET_ALL)
    print(Fore.CYAN + "Enhanced Precision, V5 API & Robustness")
    print(Fore.CYAN + "Conjuring market insights on Bybit Futures...")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown) # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown) # kill command
    logger.info("Registered signal handlers for SIGINT and SIGTERM.")

    # Start the main trading loop
    main_loop()

    logger.info("Pyrmethus spell has concluded.")
