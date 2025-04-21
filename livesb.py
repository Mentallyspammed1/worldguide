```python
# sxs.py
# Enhanced and Upgraded Scalping Bot Framework
# Derived from xrscalper.py, focusing on robust execution, error handling,
# advanced position management (BE, TSL), and Bybit V5 compatibility.

# Standard Library Imports
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo # Use zoneinfo for modern timezone handling

# Third-Party Imports
import ccxt # Exchange interaction library
import numpy as np # Numerical operations, used for NaN and jitter
import pandas as pd # Data manipulation and analysis
import pandas_ta as ta # Technical analysis library built on pandas
import requests # Used by ccxt for HTTP requests
from colorama import Fore, Style, init # Colored terminal output
from dotenv import load_dotenv # Loading environment variables

# --- Initialization ---
init(autoreset=True) # Ensure colorama resets styles automatically
load_dotenv() # Load environment variables from .env file (e.g., API keys)

# Set Decimal precision (high precision for financial calculations)
# Trade-off: Higher precision reduces potential rounding errors in complex calculations
# but might slightly impact performance. 36 is generally very high for crypto.
getcontext().prec = 36

# --- Neon Color Scheme (for console logging enhancement) ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
# API Keys loaded from environment variables
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
# Critical check: Ensure API keys are present before proceeding
if not API_KEY or not API_SECRET:
    # Use a basic logger setup for this critical startup error as full logging isn't ready yet
    logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
    logging.critical(f"{NEON_RED}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are not set.")

CONFIG_FILE = "config.json" # Name of the configuration file
LOG_DIRECTORY = "bot_logs" # Directory to store log files
# Ensure the log directory exists early in the script execution
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Default Timezone (will be overridden by config if specified)
# Using IANA timezone database names (e.g., "America/Chicago", "Europe/London", "UTC")
# https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
DEFAULT_TIMEZONE = ZoneInfo("America/Chicago")
TIMEZONE = DEFAULT_TIMEZONE # Global variable updated by load_config

# API Retry Settings (defaults, can be overridden by config)
MAX_API_RETRIES = 5 # Default maximum number of retries for failed API calls
RETRY_DELAY_SECONDS = 7 # Default base delay between retries (increases exponentially)
# HTTP status codes considered generally retryable for network/server issues
RETRYABLE_HTTP_CODES = [429, 500, 502, 503, 504]

# Valid Timeframes for Data Fetching
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
# Mapping between internal interval notation and CCXT's timeframe notation
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Default Indicator Periods (can be overridden by config.json) - Using standardized names
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_PERIOD = 20
DEFAULT_WILLIAMS_R_PERIOD = 14
DEFAULT_MFI_PERIOD = 14
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_RSI_PERIOD = 14 # Underlying RSI period for StochRSI
DEFAULT_STOCH_RSI_K_PERIOD = 3
DEFAULT_STOCH_RSI_D_PERIOD = 3
DEFAULT_RSI_PERIOD = 14
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2.0
DEFAULT_SMA10_PERIOD = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_PERIOD = 50 # Lookback period for Fibonacci High/Low
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci retracement levels
LOOP_DELAY_SECONDS = 10 # Default minimum time between main loop cycles (end of one to start of next)
POSITION_CONFIRM_DELAY_SECONDS = 10 # Default wait time after placing an order before confirming position status

# Global config dictionary, loaded by load_config
config: Dict[str, Any] = {}

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter to redact sensitive information (like API keys/secrets)
    from log messages before they are written.
    """
    # Cache patterns for a minor performance gain, avoids recompiling regex implicitly
    _patterns = {}

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record and redacts sensitive data."""
        # Format the original message using the parent class formatter
        msg = super().format(record)

        # Redact API Key if present and pattern cached/created
        if API_KEY:
            if API_KEY not in self._patterns:
                # Simple replacement, not regex needed here
                self._patterns[API_KEY] = "***API_KEY***"
            if API_KEY in msg:
                msg = msg.replace(API_KEY, self._patterns[API_KEY])

        # Redact API Secret if present and pattern cached/created
        if API_SECRET:
            if API_SECRET not in self._patterns:
                self._patterns[API_SECRET] = "***API_SECRET***"
            if API_SECRET in msg:
                msg = msg.replace(API_SECRET, self._patterns[API_SECRET])

        return msg

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.
    - Creates a default configuration file if it doesn't exist.
    - Merges loaded configuration with defaults to ensure all keys are present.
    - Validates configuration parameters against types, ranges, and allowed values.
    - Resets invalid parameters to their defaults and logs warnings.
    - Saves the updated (merged, validated) configuration back to the file if changes occurred.
    - Updates the global `TIMEZONE` variable based on the validated config.

    Args:
        filepath (str): The path to the configuration JSON file.

    Returns:
        Dict[str, Any]: The validated configuration dictionary.
    """
    global TIMEZONE # Allow modification of the global timezone variable

    # Define the default configuration structure and values
    default_config = {
        # --- Trading Pair & Timeframe ---
        "symbol": "BTC/USDT:USDT", # Example for Bybit linear perpetual
        "interval": "5", # Default timeframe (e.g., "5" for 5 minutes)

        # --- API & Bot Behavior ---
        "retry_delay": RETRY_DELAY_SECONDS, # Base delay between API retries (seconds)
        "max_api_retries": MAX_API_RETRIES, # Max retries for API calls
        "enable_trading": False, # CRITICAL SAFETY FEATURE: Must be explicitly True to execute live trades.
        "use_sandbox": True, # CRITICAL SAFETY FEATURE: Connects to exchange testnet by default.
        "max_concurrent_positions": 1, # Max open positions allowed simultaneously for this specific symbol instance.
        "quote_currency": "USDT", # Quote currency (used for balance checks, sizing). Ensure it matches the symbol's quote asset.
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait time after order placement before checking position status.
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Minimum delay between main loop cycles.
        "timezone": "America/Chicago", # IANA timezone name for local time display in console logs.

        # --- Risk Management ---
        "risk_per_trade": 0.01, # Fraction of available balance to risk per trade (e.g., 0.01 = 1%).
        "leverage": 20, # Desired leverage multiplier. Ensure it's supported by the exchange/market and within safe limits.
        "stop_loss_multiple": 1.8, # ATR multiple for calculating the initial Stop Loss distance.
        "take_profit_multiple": 0.7, # ATR multiple for calculating the initial Take Profit distance.

        # --- Order Execution ---
        "entry_order_type": "market", # Type of order for entry: "market" or "limit".
        "limit_order_offset_buy": 0.0005, # For limit entries: Percentage offset from the target price for BUY orders (0.0005 = 0.05%). Price = Target * (1 - Offset)
        "limit_order_offset_sell": 0.0005, # For limit entries: Percentage offset from the target price for SELL orders (0.0005 = 0.05%). Price = Target * (1 + Offset)

        # --- Advanced Position Management ---
        "enable_trailing_stop": True, # Enable exchange-native Trailing Stop Loss (Requires exchange support, e.g., Bybit V5).
        # IMPORTANT (Bybit V5 TSL): TSL uses PRICE DISTANCE, not percentage.
        # 'callback_rate' is used here to *calculate* that price distance based on the activation price.
        "trailing_stop_callback_rate": 0.005, # Percentage (0.005 = 0.5%) used to calculate the trail distance from the *activation price*. Distance = ActivationPrice * CallbackRate.
        "trailing_stop_activation_percentage": 0.003, # Profit percentage (0.003 = 0.3%) move from entry price needed to trigger calculation of TSL activation price. ActivationPrice = Entry +/- (Entry * Activation%)
        "enable_break_even": True, # Enable moving Stop Loss to break-even point once profit target is hit.
        "break_even_trigger_atr_multiple": 1.0, # Move SL to break-even when profit reaches (ATR * this multiple).
        "break_even_offset_ticks": 2, # Place the break-even SL slightly beyond the entry price (in number of ticks) to cover potential fees/slippage.

        "time_based_exit_minutes": None, # Optional: Exit position automatically after specified minutes (e.g., 120 for 2 hours). Set to null or None to disable.

        # --- Indicator Periods & Parameters (Defaults defined above) ---
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_PERIOD,
        "bollinger_bands_period": DEFAULT_BBANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BBANDS_STDDEV,
        "cci_period": DEFAULT_CCI_PERIOD,
        "williams_r_period": DEFAULT_WILLIAMS_R_PERIOD,
        "mfi_period": DEFAULT_MFI_PERIOD,
        "stoch_rsi_period": DEFAULT_STOCH_RSI_PERIOD,
        "stoch_rsi_rsi_period": DEFAULT_STOCH_RSI_RSI_PERIOD,
        "stoch_rsi_k_period": DEFAULT_STOCH_RSI_K_PERIOD,
        "stoch_rsi_d_period": DEFAULT_STOCH_RSI_D_PERIOD,
        "psar_step": DEFAULT_PSAR_STEP,
        "psar_max_step": DEFAULT_PSAR_MAX_STEP,
        "sma_10_period": DEFAULT_SMA10_PERIOD,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "fibonacci_period": DEFAULT_FIB_PERIOD, # Lookback for Fib High/Low

        # --- Indicator Calculation & Scoring Control ---
        "orderbook_limit": 25, # Number of order book levels to fetch and analyze. Check exchange limits (Bybit V5 linear: up to 200).
        "signal_score_threshold": 1.5, # Minimum absolute weighted score required to trigger a BUY or SELL signal.
        "stoch_rsi_oversold_threshold": 25, # StochRSI level below which it's considered oversold (influences score).
        "stoch_rsi_overbought_threshold": 75, # StochRSI level above which it's considered overbought (influences score).
        "volume_confirmation_multiplier": 1.5, # Minimum ratio of current volume to Volume MA required for positive volume confirmation score.
        "indicators": { # Toggle calculation and scoring contribution for each indicator
            # Key names MUST match _check_<key> methods and weight_sets keys
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
            # Add new indicators here and ensure corresponding _check_ method and weight entries exist
        },
        "weight_sets": { # Define different scoring weights for various strategies
            "scalping": { # Example: Faster signals, momentum-focused
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: Balanced approach
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Ensure keys here match keys in "indicators" and correspond to _check_ methods
        },
        "active_weight_set": "default" # Selects which weight set from "weight_sets" to use for scoring.
    }

    current_config = default_config.copy() # Start with default values
    config_loaded_successfully = False
    needs_saving = False # Flag to track if the config file needs to be updated/saved

    # --- Load Existing Config (if exists) ---
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            # Merge loaded config over defaults, ensuring all default keys exist in the final config
            merged_config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
            config_loaded_successfully = True
            # Check if merge resulted in changes (e.g., new defaults added)
            if merged_config != loaded_config: # Compare merged vs originally loaded
                needs_saving = True
                print(f"{NEON_YELLOW}Configuration merged with new defaults/structure.{RESET}")
            current_config = merged_config # Use the merged config for validation

        except (json.JSONDecodeError, IOError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            current_config = default_config # Fallback to default
            needs_saving = True # Need to save the default config
        except Exception as e:
             print(f"{NEON_RED}Unexpected error loading config {filepath}: {e}. Using default config.{RESET}")
             current_config = default_config
             needs_saving = True
    else:
        # Config file doesn't exist, create it using defaults
        print(f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}")
        current_config = default_config
        needs_saving = True

    # --- Validation Section ---
    # Work with the 'current_config' (which is either loaded+merged or default)
    original_config_before_validation = current_config.copy() # Keep a copy to check if validation modified anything

    # Helper function for validating parameters, logging errors, and resetting to default
    def validate_param(key, default_value, validation_func, error_msg_format):
        """Validates a config key, resets to default if invalid, and returns original validity."""
        is_valid = False
        current_value = current_config.get(key)
        try:
            if key in current_config and validation_func(current_value):
                is_valid = True # Value exists and passes validation
            else:
                # Reset to default if key is missing or validation fails
                current_config[key] = default_value
                # Format error message safely, using repr() for potentially complex values
                value_repr = repr(current_value) if current_value is not None else 'None'
                print(f"{NEON_RED}{error_msg_format.format(key=key, value=value_repr, default=default_value)}{RESET}")
                # Needs saving because we changed the value
                nonlocal needs_saving
                needs_saving = True
        except Exception as validation_err:
            # Catch errors *within* the validation function itself
            print(f"{NEON_RED}Error validating config key '{key}' (Value: {repr(current_value)}): {validation_err}. Resetting to default '{default_value}'.{RESET}")
            current_config[key] = default_value
            needs_saving = True
        return is_valid

    # --- Validate Core Parameters ---
    validate_param("symbol", default_config["symbol"],
                   lambda v: isinstance(v, str) and v.strip(),
                   "CRITICAL: Config key '{key}' is missing, empty, or invalid ({value}). Resetting to default: '{default}'. Check symbol format.")

    validate_param("interval", default_config["interval"],
                   lambda v: v in VALID_INTERVALS,
                   "Invalid interval '{value}' in config for '{key}'. Resetting to default '{default}'. Valid: " + ", ".join(VALID_INTERVALS) + ".")

    validate_param("entry_order_type", default_config["entry_order_type"],
                   lambda v: v in ["market", "limit"],
                   "Invalid entry_order_type '{value}' for '{key}'. Must be 'market' or 'limit'. Resetting to default '{default}'.")

    validate_param("quote_currency", default_config["quote_currency"],
                   lambda v: isinstance(v, str) and len(v) >= 3 and v.isupper(), # Basic sanity check
                   "Invalid quote_currency '{value}' for '{key}'. Should be uppercase currency code (e.g., USDT). Resetting to '{default}'.")

    # Validate timezone and update global TIMEZONE constant
    try:
        tz_str = current_config.get("timezone", default_config["timezone"])
        if not isinstance(tz_str, str): raise TypeError("Timezone must be a string")
        tz_info = ZoneInfo(tz_str) # This raises ZoneInfoNotFoundError if invalid
        current_config["timezone"] = tz_str # Store the validated string
        TIMEZONE = tz_info # Update the global constant for use elsewhere
    except Exception as tz_err:
        print(f"{NEON_RED}Invalid timezone '{tz_str}' in config: {tz_err}. Resetting to default '{default_config['timezone']}'.{RESET}")
        current_config["timezone"] = default_config["timezone"]
        TIMEZONE = ZoneInfo(default_config["timezone"]) # Reset global to default
        needs_saving = True

    # Validate active weight set exists in the 'weight_sets' dictionary
    validate_param("active_weight_set", default_config["active_weight_set"],
                   lambda v: isinstance(v, str) and v in current_config.get("weight_sets", {}),
                   "Active weight set '{value}' for '{key}' not found in 'weight_sets' section. Resetting to '{default}'.")

    # --- Validate Numeric Parameters (using Decimal for range checks) ---
    # Format: key: (min_val, max_val, allow_min_equal, allow_max_equal, requires_integer, default_val)
    numeric_params = {
        "risk_per_trade": (0, 1, False, False, False, default_config["risk_per_trade"]), # Exclusive bounds 0-1
        "leverage": (1, 1000, True, True, True, default_config["leverage"]), # Realistic max leverage, must be int
        "stop_loss_multiple": (0, float('inf'), False, True, False, default_config["stop_loss_multiple"]), # Must be > 0
        "take_profit_multiple": (0, float('inf'), False, True, False, default_config["take_profit_multiple"]), # Must be > 0
        "trailing_stop_callback_rate": (0, 1, False, False, False, default_config["trailing_stop_callback_rate"]), # Exclusive bounds 0-1
        "trailing_stop_activation_percentage": (0, 1, True, False, False, default_config["trailing_stop_activation_percentage"]), # Allow 0%, must be < 1
        "break_even_trigger_atr_multiple": (0, float('inf'), False, True, False, default_config["break_even_trigger_atr_multiple"]), # Must be > 0
        "break_even_offset_ticks": (0, 1000, True, True, True, default_config["break_even_offset_ticks"]), # Must be int >= 0
        "signal_score_threshold": (0, float('inf'), False, True, False, default_config["signal_score_threshold"]), # Must be > 0
        "atr_period": (2, 1000, True, True, True, default_config["atr_period"]), # Min 2 for ATR calc
        "ema_short_period": (1, 1000, True, True, True, default_config["ema_short_period"]),
        "ema_long_period": (1, 1000, True, True, True, default_config["ema_long_period"]),
        "rsi_period": (2, 1000, True, True, True, default_config["rsi_period"]), # Min 2 for RSI calc
        "bollinger_bands_period": (2, 1000, True, True, True, default_config["bollinger_bands_period"]),
        "bollinger_bands_std_dev": (0, 10, False, True, False, default_config["bollinger_bands_std_dev"]), # Must be > 0
        "cci_period": (2, 1000, True, True, True, default_config["cci_period"]), # Typical min period
        "williams_r_period": (2, 1000, True, True, True, default_config["williams_r_period"]),
        "mfi_period": (2, 1000, True, True, True, default_config["mfi_period"]),
        "stoch_rsi_period": (2, 1000, True, True, True, default_config["stoch_rsi_period"]),
        "stoch_rsi_rsi_period": (2, 1000, True, True, True, default_config["stoch_rsi_rsi_period"]),
        "stoch_rsi_k_period": (1, 1000, True, True, True, default_config["stoch_rsi_k_period"]),
        "stoch_rsi_d_period": (1, 1000, True, True, True, default_config["stoch_rsi_d_period"]),
        "psar_step": (0, 1, False, True, False, default_config["psar_step"]), # Must be > 0
        "psar_max_step": (0, 1, False, True, False, default_config["psar_max_step"]), # Must be > 0
        "sma_10_period": (1, 1000, True, True, True, default_config["sma_10_period"]),
        "momentum_period": (1, 1000, True, True, True, default_config["momentum_period"]),
        "volume_ma_period": (1, 1000, True, True, True, default_config["volume_ma_period"]),
        "fibonacci_period": (2, 1000, True, True, True, default_config["fibonacci_period"]), # Needs at least 2 bars
        "orderbook_limit": (1, 200, True, True, True, default_config["orderbook_limit"]), # Bybit V5 linear max is 200
        "position_confirm_delay_seconds": (0, 120, True, True, False, default_config["position_confirm_delay_seconds"]), # Allow 0 delay
        "loop_delay_seconds": (1, 300, True, True, False, default_config["loop_delay_seconds"]),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False, default_config["stoch_rsi_oversold_threshold"]), # Must be < 100
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False, default_config["stoch_rsi_overbought_threshold"]), # Must be > 0
        "volume_confirmation_multiplier": (0, float('inf'), False, True, False, default_config["volume_confirmation_multiplier"]), # Must be > 0
        "limit_order_offset_buy": (0, 0.1, True, False, False, default_config["limit_order_offset_buy"]), # 0% to 10% offset seems reasonable
        "limit_order_offset_sell": (0, 0.1, True, False, False, default_config["limit_order_offset_sell"]),
        "retry_delay": (1, 120, True, True, False, default_config["retry_delay"]),
        "max_api_retries": (0, 10, True, True, True, default_config["max_api_retries"]), # Must be int >= 0
        "max_concurrent_positions": (1, 10, True, True, True, default_config["max_concurrent_positions"]), # Must be int >= 1
    }
    for key, (min_val, max_val, allow_min, allow_max, is_integer, default_val) in numeric_params.items():
        value = current_config.get(key)
        param_is_valid = False
        if value is not None:
            try:
                val_dec = Decimal(str(value)) # Convert to Decimal for reliable checks
                if not val_dec.is_finite(): raise ValueError("Value not finite")

                # Check bounds using Decimal comparison
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))
                lower_bound_ok = (val_dec >= min_dec) if allow_min else (val_dec > min_dec)
                upper_bound_ok = (val_dec <= max_dec) if allow_max else (val_dec < max_dec)

                if lower_bound_ok and upper_bound_ok:
                    # Convert to final type (int or float) after successful validation
                    if is_integer:
                        # Check if it's actually an integer before converting
                        if val_dec == val_dec.to_integral_value():
                            final_value = int(val_dec)
                            current_config[key] = final_value # Store validated integer
                            param_is_valid = True
                        else:
                            raise ValueError("Non-integer value provided for integer parameter")
                    else:
                        final_value = float(val_dec) # Store as float if not integer required
                        current_config[key] = final_value
                        param_is_valid = True
                # else: Bounds check failed, param_is_valid remains False
            except (ValueError, TypeError, InvalidOperation):
                 pass # Invalid format or failed conversion/checks, param_is_valid remains False

        if not param_is_valid:
            # Use validate_param to log error and reset to default
            bound_str = f"{'>' if not allow_min else '>='} {min_val} and {'<' if not allow_max else '<='} {max_val}"
            type_str = 'integer' if is_integer else 'number'
            err_msg = f"Invalid value for '{{key}}' ({{value}}). Must be a {type_str} ({bound_str}). Resetting to default '{{default}}'."
            validate_param(key, default_val, lambda v: False, err_msg) # Force reset and log

    # Specific validation for time_based_exit_minutes (allows None or positive number)
    time_exit_key = "time_based_exit_minutes"
    time_exit_value = current_config.get(time_exit_key)
    time_exit_valid = False
    if time_exit_value is None:
        time_exit_valid = True # None is valid
    else:
        try:
            time_exit_float = float(time_exit_value)
            if time_exit_float > 0 and np.isfinite(time_exit_float):
                 current_config[time_exit_key] = time_exit_float # Store validated float
                 time_exit_valid = True
            else: raise ValueError("Must be positive and finite if set")
        except (ValueError, TypeError):
            pass # Invalid format or non-positive/non-finite

    if not time_exit_valid:
         validate_param(time_exit_key, default_config[time_exit_key], lambda v: False, # Force reset
                        "Invalid value for '{{key}}' ({{value}}). Must be 'None' or a positive number. Resetting to default ('{{default}}').")

    # --- Validate Boolean Parameters ---
    bool_params = ["enable_trading", "use_sandbox", "enable_trailing_stop", "enable_break_even"]
    for key in bool_params:
         validate_param(key, default_config[key], lambda v: isinstance(v, bool),
                        "Invalid value for '{{key}}' ({{value}}). Must be boolean (true/false). Resetting to default '{{default}}'.")

    # --- Validate Indicator Enable Flags (must exist in defaults and be boolean) ---
    indicators_key = 'indicators'
    if indicators_key in current_config and isinstance(current_config[indicators_key], dict):
        indicators_dict = current_config[indicators_key]
        default_indicators = default_config[indicators_key]
        keys_to_remove = [] # Collect unknown keys to remove later
        for ind_key, ind_val in indicators_dict.items():
            # Check if key exists in default config (prevents unknown indicators)
            if ind_key not in default_indicators:
                print(f"{NEON_YELLOW}Warning: Unknown key '{ind_key}' found in '{indicators_key}' section. It will be removed.{RESET}")
                keys_to_remove.append(ind_key)
                needs_saving = True
                continue
            # Validate value is boolean
            if not isinstance(ind_val, bool):
                default_ind_val = default_indicators.get(ind_key, False) # Should exist, but safety fallback
                print(f"{NEON_RED}Invalid value for '{indicators_key}.{ind_key}' ({repr(ind_val)}). Must be boolean (true/false). Resetting to default '{default_ind_val}'.{RESET}")
                indicators_dict[ind_key] = default_ind_val
                needs_saving = True
        # Remove unknown keys outside the loop
        for r_key in keys_to_remove: del indicators_dict[r_key]
    else:
        # If 'indicators' key is missing or not a dict, reset to default
        print(f"{NEON_RED}Invalid or missing '{indicators_key}' section in config. Resetting to default.{RESET}")
        current_config[indicators_key] = default_config[indicators_key].copy() # Use copy of default
        needs_saving = True

    # --- Validate Weight Sets Structure and Values ---
    ws_key = "weight_sets"
    if ws_key in current_config and isinstance(current_config[ws_key], dict):
        weight_sets = current_config[ws_key]
        default_indicators_keys = default_config['indicators'].keys() # Get valid indicator keys
        sets_to_remove = []
        for set_name, weights in weight_sets.items():
            if not isinstance(weights, dict):
                 print(f"{NEON_RED}Invalid structure for weight set '{set_name}' (must be a dictionary of indicator:weight pairs). Removing this set.{RESET}")
                 sets_to_remove.append(set_name)
                 needs_saving = True
                 continue

            weights_to_remove = []
            for ind_key, weight_val in weights.items():
                # Ensure weight key matches a known indicator key
                if ind_key not in default_indicators_keys:
                    print(f"{NEON_YELLOW}Warning: Weight defined for unknown/invalid indicator '{ind_key}' in weight set '{set_name}'. Removing this weight entry.{RESET}")
                    weights_to_remove.append(ind_key)
                    needs_saving = True
                    continue

                # Validate weight value is numeric and non-negative
                try:
                    weight_dec = Decimal(str(weight_val))
                    if not weight_dec.is_finite() or weight_dec < 0:
                        raise ValueError("Weight must be non-negative and finite")
                    # Store validated weight as float (common usage in scoring)
                    weights[ind_key] = float(weight_dec)
                except (ValueError, TypeError, InvalidOperation):
                     # Attempt to get default weight, fallback to 0.0
                     default_weight_set = default_config[ws_key].get(set_name, {})
                     default_weight = default_weight_set.get(ind_key, 0.0)
                     print(f"{NEON_RED}Invalid weight value '{repr(weight_val)}' for indicator '{ind_key}' in weight set '{set_name}'. Must be a non-negative number. Resetting to default '{default_weight}'.{RESET}")
                     weights[ind_key] = float(default_weight) # Reset to float
                     needs_saving = True
            # Remove invalid weights from the current set
            for r_key in weights_to_remove: del weights[r_key]
        # Remove invalid sets
        for r_key in sets_to_remove: del weight_sets[r_key]
    else:
         print(f"{NEON_RED}Invalid or missing '{ws_key}' section in config. Resetting to default.{RESET}")
         current_config[ws_key] = default_config[ws_key].copy() # Use copy of default
         needs_saving = True

    # --- Post-validation Check: Ensure active_weight_set still exists after potential removals ---
    active_set = current_config.get("active_weight_set")
    if active_set not in current_config.get("weight_sets", {}):
         default_active_set = default_config["active_weight_set"]
         print(f"{NEON_RED}Previously selected active_weight_set '{active_set}' is no longer valid (possibly removed during validation). Resetting to default '{default_active_set}'.{RESET}")
         current_config["active_weight_set"] = default_active_set
         needs_saving = True

    # --- Save Updated Config if Necessary ---
    # Needs saving if file was created, merged, or validation caused changes
    if needs_saving or current_config != original_config_before_validation:
        try:
            with open(filepath, "w", encoding="utf-8") as f_write:
                # Dump the validated and potentially corrected config
                json.dump(current_config, f_write, indent=4, ensure_ascii=False, sort_keys=True)
            print(f"{NEON_YELLOW}Saved updated configuration to {filepath}{RESET}")
        except IOError as e:
            print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
        except Exception as e:
             print(f"{NEON_RED}Unexpected error saving config file {filepath}: {e}{RESET}")

    return current_config

def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """
    Recursively merges the loaded configuration dictionary onto the default dictionary.
    - Ensures all keys from the default config exist in the final merged config.
    - Prioritizes values from the loaded config if a key exists in both.
    - Handles nested dictionaries recursively.
    - Adds keys present only in the loaded_config (allows user extensions).

    Args:
        loaded_config (Dict): The configuration loaded from the file.
        default_config (Dict): The default configuration structure and values.

    Returns:
        Dict: The merged configuration dictionary.
    """
    merged = default_config.copy() # Start with the default structure

    for key, loaded_value in loaded_config.items():
        if key in merged:
            default_value = merged[key]
            # If both loaded and default values for the key are dictionaries, recurse
            if isinstance(loaded_value, dict) and isinstance(default_value, dict):
                merged[key] = _merge_configs(loaded_value, default_value)
            else:
                # Otherwise, overwrite default with loaded value (validation happens later)
                merged[key] = loaded_value
        else:
            # If key from loaded config doesn't exist in default, add it.
            # This allows users to add custom keys to their config if needed.
            merged[key] = loaded_value

    # Ensure all keys from the default config are present in the merged result.
    # This handles cases where a default key was missing entirely in the loaded config.
    for key, default_value in default_config.items():
        if key not in merged:
            merged[key] = default_value
            # print(f"Debug: Added missing key '{key}' with default value during merge.") # Optional debug log

    return merged

# --- Logging Setup ---
def setup_logger(name: str, config: Dict[str, Any], level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger instance with specified name, configuration, and level.
    - Configures a rotating file handler (logs in UTC).
    - Configures a colored console handler (logs in local timezone specified in config).
    - Uses SensitiveFormatter to redact API keys/secrets.
    - Prevents duplicate log messages if called multiple times for the same logger name.

    Args:
        name (str): The name for the logger instance.
        config (Dict[str, Any]): The bot's configuration dictionary (used for timezone).
        level (int): The logging level for the console handler (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if this function is called again for the same logger
    if logger.hasHandlers():
        logger.handlers.clear()

    # Set the logger's base level to DEBUG to capture all messages.
    # Handlers will filter based on their individual levels.
    logger.setLevel(logging.DEBUG)

    # --- File Handler (Rotating, UTC Timestamps) ---
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        # Directory should exist from earlier check, but ensure again just in case.
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        # Rotate logs: 10MB per file, keep last 5 files
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use UTC timestamps in file logs for consistency across different systems/locations
        # ISO 8601 format with milliseconds and UTC 'Z' indicator
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S' # ISO 8601 standard date format
        )
        file_formatter.converter = time.gmtime # Force formatter to use UTC time
        file_handler.setFormatter(file_formatter)
        # Log DEBUG level and above to the file
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to basic console logging if file handler setup fails
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}. File logging disabled.{RESET}")
        # Ensure there's at least one handler if file logging failed
        if not logger.hasHandlers():
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(basic_handler)

    # --- Console Handler (Colored, Local Timestamps) ---
    # Add console handler only if no StreamHandler exists yet (prevents duplicates if file handler failed)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        # Determine local timezone for console display from validated config
        try:
            # Use the globally updated TIMEZONE variable from load_config
            console_tz = TIMEZONE
        except Exception as tz_err: # Should not happen if load_config worked, but safety check
            print(f"{NEON_RED}Error getting configured timezone for console logs: {tz_err}. Using UTC.{RESET}")
            console_tz = ZoneInfo("UTC")

        # Formatter with colors and local time (including timezone abbreviation %Z)
        console_formatter = SensitiveFormatter(
            f"{NEON_BLUE}%(asctime)s{RESET} {NEON_YELLOW}%(levelname)-8s{RESET} {NEON_PURPLE}[%(name)s]{RESET} %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S %Z' # Example: 2023-10-27 15:30:00 CDT
        )

        # Custom time converter function for the console formatter
        def local_time_converter(*args):
            """Returns the current time as a timetuple in the configured local timezone."""
            return datetime.now(console_tz).timetuple()

        console_formatter.converter = local_time_converter # Assign the converter
        stream_handler.setFormatter(console_formatter)
        # Set console log level based on the function argument (e.g., INFO)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger (avoids potential duplicate outputs)
    logger.propagate = False
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and configures the CCXT Bybit exchange object.
    - Sets API keys, rate limiting, timeouts, and V5-specific options.
    - Handles enabling Sandbox (Testnet) mode with URL verification and fallbacks.
    - Loads exchange markets and validates the configured trading symbol.
    - Performs an initial connection test by fetching balance.
    - Includes robust error handling for common initialization failures.

    Args:
        config (Dict[str, Any]): The bot's configuration dictionary.
        logger (logging.Logger): The logger instance.

    Returns:
        Optional[ccxt.Exchange]: The configured CCXT exchange object, or None if initialization fails.
    """
    lg = logger # Alias for convenience
    try:
        # CCXT Exchange configuration options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable ccxt's built-in rate limiter
            'rateLimit': 150, # Milliseconds between requests (approx 6.6 req/s). Adjust based on Bybit V5 limits and VIP level.
            'options': {
                'defaultType': 'linear', # CRUCIAL for Bybit V5 USDT/USDC perpetuals/futures. Use 'inverse' for inverse contracts.
                'adjustForTimeDifference': True, # Helps mitigate timestamp synchronization errors.
                # Set reasonable network timeouts (in milliseconds) for various API calls
                'fetchTickerTimeout': 15000,      # 15 seconds
                'fetchBalanceTimeout': 20000,     # 20 seconds
                'createOrderTimeout': 25000,      # 25 seconds
                'cancelOrderTimeout': 20000,      # 20 seconds
                'fetchPositionsTimeout': 25000,   # 25 seconds
                'fetchOHLCVTimeout': 20000,       # 20 seconds
                'fetchOrderBookTimeout': 15000,   # 15 seconds
                'setLeverageTimeout': 20000,      # 20 seconds
                'fetchMyTradesTimeout': 20000,    # 20 seconds (Added)
                'fetchClosedOrdersTimeout': 25000,# 25 seconds (Added)
                # Custom User-Agent can help identify your bot's traffic to the exchange (Optional)
                'user-agent': 'sxsBot/1.2 (+https://github.com/your_repo)', # Optional: Update URL if applicable
                # Bybit V5 specific settings (Consult Bybit/CCXT docs if needed)
                # 'recvWindow': 10000, # Optional: Increase if timestamp errors persist despite adjustForTimeDifference
                # 'brokerId': 'YOUR_BROKER_ID', # Optional: If participating in Bybit broker program
                # 'enableUnifiedMargin': False, # Set to True if using Bybit's Unified Trading Account (UTA)
                # 'enableUnifiedAccount': False, # May be an alias for above; check CCXT/Bybit documentation
            }
        }

        # Instantiate the Bybit exchange class from ccxt
        exchange_id = "bybit" # Explicitly target Bybit
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # --- Sandbox (Testnet) Mode Setup ---
        if config.get('use_sandbox', True): # Default to sandbox if key is missing
            lg.warning(f"{NEON_YELLOW}INITIALIZING IN SANDBOX MODE (Testnet){RESET}")
            try:
                # Attempt to use CCXT's standard method for enabling sandbox mode
                exchange.set_sandbox_mode(True)
                lg.info(f"Sandbox mode enabled via exchange.set_sandbox_mode(True) for {exchange.id}.")

                # --- CRITICAL VERIFICATION: Check if the API URL actually changed ---
                current_api_url = exchange.urls.get('api', '')
                if 'testnet' not in current_api_url.lower():
                    lg.warning(f"set_sandbox_mode did not change API URL to testnet! Current URL: {current_api_url}")
                    # Attempt manual override using URLs defined in exchange description
                    test_url_info = exchange.describe().get('urls', {}).get('test')
                    api_test_url = None
                    if isinstance(test_url_info, str):
                        api_test_url = test_url_info
                    elif isinstance(test_url_info, dict): # Sometimes 'test' contains a dict of URLs
                        # Prioritize public/private keys if they exist
                        api_test_url = test_url_info.get('public') or test_url_info.get('private') or next((url for url in test_url_info.values() if isinstance(url, str)), None)

                    if api_test_url:
                        exchange.urls['api'] = api_test_url
                        lg.info(f"Manually set API URL to Testnet based on exchange.describe(): {exchange.urls['api']}")
                    else:
                        # Final hardcoded fallback if describe() didn't provide a usable URL
                        fallback_test_url = 'https://api-testnet.bybit.com'
                        lg.warning(f"Could not reliably determine testnet URL from describe(). Hardcoding fallback: {fallback_test_url}")
                        exchange.urls['api'] = fallback_test_url
                else:
                     lg.info(f"Confirmed API URL is set to Testnet: {current_api_url}")

            except AttributeError:
                # Fallback if the ccxt version doesn't support set_sandbox_mode
                lg.warning(f"{exchange.id} ccxt version might lack set_sandbox_mode. Manually setting Testnet API URL.")
                exchange.urls['api'] = 'https://api-testnet.bybit.com' # Ensure this is the correct V5 testnet URL
                lg.info(f"Manually set Bybit API URL to Testnet: {exchange.urls['api']}")
            except Exception as e_sandbox:
                lg.error(f"Error encountered trying to enable sandbox mode: {e_sandbox}. Ensure API keys are for Testnet. Proceeding with potentially incorrect URL.", exc_info=True)
        else:
            # --- Live (Real Money) Environment Setup ---
            lg.info(f"{NEON_GREEN}INITIALIZING IN LIVE (Real Money) Environment.{RESET}")
            # Ensure API URL is set to production if sandbox was somehow enabled previously
            current_api_url = exchange.urls.get('api', '')
            if 'testnet' in current_api_url.lower():
                lg.warning("Detected testnet URL while in live mode configuration. Attempting to reset to production URL.")
                # Find the production URL from exchange description
                prod_url_info = exchange.describe().get('urls', {}).get('api')
                api_prod_url = None
                if isinstance(prod_url_info, str):
                     api_prod_url = prod_url_info
                elif isinstance(prod_url_info, dict): # API URL might be nested
                     api_prod_url = prod_url_info.get('public') or prod_url_info.get('private') or next((url for url in prod_url_info.values() if isinstance(url, str)), None)

                if api_prod_url:
                    exchange.urls['api'] = api_prod_url
                    lg.info(f"Reset API URL to Production based on exchange.describe(): {exchange.urls['api']}")
                else:
                    # Fallback to 'www' URL or hardcoded default if 'api' is not found
                    www_url = exchange.describe().get('urls',{}).get('www')
                    if www_url and isinstance(www_url, str):
                         exchange.urls['api'] = www_url # Less ideal, but better than testnet
                         lg.info(f"Reset API URL to Production using 'www' fallback: {exchange.urls['api']}")
                    else:
                         fallback_prod_url = 'https://api.bybit.com' # Hardcoded V5 production URL
                         lg.error(f"Could not determine production API URL automatically. Hardcoding fallback: {fallback_prod_url}")
                         exchange.urls['api'] = fallback_prod_url

        lg.info(f"Initializing {exchange.id} (API Endpoint: {exchange.urls.get('api', 'URL Not Set')})...")

        # --- Load Markets (Essential for precision, limits, IDs, fees) ---
        lg.info(f"Loading markets for {exchange.id} (this may take a moment)...")
        try:
             # Use safe_api_call for robustness, force reload to ensure freshness
             safe_api_call(exchange.load_markets, lg, reload=True)
             lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.symbols)} symbols.")

             # --- Validate Target Symbol Existence & Compatibility ---
             target_symbol = config.get("symbol")
             if not target_symbol:
                  lg.critical(f"{NEON_RED}FATAL: 'symbol' not defined in configuration!{RESET}")
                  return None
             if target_symbol not in exchange.markets:
                  lg.critical(f"{NEON_RED}FATAL: Target symbol '{target_symbol}' not found in loaded markets for {exchange.id}!{RESET}")
                  lg.critical(f"{NEON_RED}>> Check symbol spelling, format, and availability on the exchange ({'Testnet' if config.get('use_sandbox') else 'Live'}).{RESET}")
                  # Provide hint for common Bybit V5 linear format
                  if '/' in target_symbol and ':' not in target_symbol and exchange.id == 'bybit':
                       base, quote = target_symbol.split('/')[:2]
                       suggested_symbol = f"{base}/{quote}:{quote}"
                       lg.warning(f"{NEON_YELLOW}Hint: For Bybit V5 linear perpetuals, the format is often like '{suggested_symbol}'.{RESET}")
                  # List available markets if the list is small and potentially helpful
                  if 0 < len(exchange.symbols) < 50:
                       lg.debug(f"Available symbols sample: {sorted(list(exchange.symbols))[:10]}...")
                  elif len(exchange.symbols) == 0:
                       lg.error("No symbols were loaded from the exchange at all.")
                  return None # Fatal error if configured symbol doesn't exist
             else:
                  lg.info(f"Target symbol '{target_symbol}' validated against loaded markets.")
                  # Optional: Add checks here for market type compatibility (e.g., ensure it's linear if expected)
                  market_info_check = exchange.market(target_symbol)
                  if market_info_check.get('linear') is not True and exchange.options.get('defaultType') == 'linear':
                       lg.warning(f"Target symbol '{target_symbol}' might not be a linear contract, but defaultType is linear. Verify settings.")


        except Exception as market_err:
             lg.critical(f"{NEON_RED}CRITICAL: Failed to load markets after retries: {market_err}. Bot cannot operate without market data. Exiting.{RESET}", exc_info=True)
             return None # Fatal error

        # --- Initial Connection & Permissions Test (Fetch Balance) ---
        # This also helps confirm the correct account type (CONTRACT/UNIFIED) is accessible.
        account_type_hint = exchange.options.get('defaultType', 'linear').upper() # e.g., LINEAR -> check CONTRACT
        account_type_to_test = 'CONTRACT' if account_type_hint != 'INVERSE' else 'CONTRACT' # Or potentially 'UNIFIED'
        lg.info(f"Performing initial connection test by fetching balance (Account Type Hint based on defaultType: {account_type_to_test})...")
        quote_curr = config.get("quote_currency", "USDT") # Use configured quote currency
        balance_decimal = fetch_balance(exchange, quote_curr, lg) # Use the dedicated robust function

        if balance_decimal is not None:
             # Balance fetch succeeded, log the result
             lg.info(f"{NEON_GREEN}Successfully connected and fetched initial {quote_curr} balance: {balance_decimal:.4f}{RESET}")
             if balance_decimal == Decimal('0'):
                  lg.warning(f"{NEON_YELLOW}Initial available {quote_curr} balance is zero. Ensure funds are in the correct account type (e.g., CONTRACT, UNIFIED) and wallet.{RESET}")
        else:
             # fetch_balance logs detailed errors, add a critical warning here as failure is significant.
             lg.critical(f"{NEON_RED}CRITICAL: Initial balance fetch for {quote_curr} failed.{RESET}")
             lg.critical(f"{NEON_RED}>> Check API key validity, permissions (read access needed), IP whitelisting, account type (CONTRACT/UNIFIED?), and network connectivity.{RESET}")
             # Decide if this is fatal. For a trading bot, inability to fetch balance usually is.
             # return None # Uncomment to make initial balance fetch failure fatal

        lg.info(f"CCXT exchange '{exchange.id}' initialized. Sandbox: {config.get('use_sandbox')}, Default Type: {exchange.options.get('defaultType')}")
        return exchange

    # --- Specific Exception Handling for Initialization ---
    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Please check: API Key/Secret correctness, validity (not expired), enabled permissions (read, trade), and IP whitelisting configuration on the Bybit website.{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> This could be a temporary issue with the exchange API, incorrect exchange settings (e.g., defaultType), or network problems. Check Bybit status pages.{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check your internet connection, DNS resolution, and firewall settings. Ensure api.bybit.com (or testnet) is reachable.{RESET}")
    except Exception as e:
        # Catch any other unexpected errors during the setup process
        lg.critical(f"{NEON_RED}Unexpected critical error initializing CCXT exchange: {e}{RESET}", exc_info=True)

    return None # Return None if any critical error occurred during initialization

# --- API Call Wrapper with Enhanced Retries ---
def safe_api_call(func, logger: logging.Logger, *args, **kwargs):
    """
    Wraps a CCXT API call with robust retry logic for network issues, rate limits,
    and specific transient exchange errors. Uses exponential backoff with jitter.

    Args:
        func: The CCXT exchange method to call (e.g., exchange.fetch_ticker).
        logger: The logger instance for logging retry attempts and errors.
        *args: Positional arguments for the CCXT function.
        **kwargs: Keyword arguments for the CCXT function.

    Returns:
        The result of the API call if successful after retries.

    Raises:
        The original exception if the call fails after all retries or encounters
        a non-retryable error (e.g., AuthenticationError, InvalidOrder).
    """
    lg = logger
    # Get retry parameters from global config if available, else use constants
    # Using globals() is generally okay for script-level config, but passing config explicitly would be cleaner
    global config
    max_retries = config.get("max_api_retries", MAX_API_RETRIES) if config else MAX_API_RETRIES
    base_retry_delay = config.get("retry_delay", RETRY_DELAY_SECONDS) if config else RETRY_DELAY_SECONDS
    attempts = 0
    last_exception = None

    while attempts <= max_retries:
        try:
            # Attempt the API call
            result = func(*args, **kwargs)
            # Optional: Log successful calls at DEBUG level (can be verbose)
            # lg.debug(f"API call '{func.__name__}' successful (Attempt {attempts+1}).")
            return result # Success, return the result

        # --- Retryable Network/Server Availability Errors ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
            last_exception = e
            # Exponential backoff: delay = base * (1.5 ^ attempts)
            wait_time = base_retry_delay * (1.5 ** attempts)
            # Add random jitter (+/- 10% of wait_time) to prevent simultaneous retries (thundering herd)
            wait_time *= (1 + (np.random.rand() - 0.5) * 0.2)
            wait_time = min(wait_time, 60) # Cap maximum wait time (e.g., 60 seconds)
            lg.warning(f"{NEON_YELLOW}Retryable Network/Availability Error in '{func.__name__}': {type(e).__name__}. "
                       f"Retrying in {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1}). Details: {e}{RESET}")

        # --- Rate Limit Errors ---
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            retry_after_header = None
            header_wait_seconds = 0
            # Try to extract 'Retry-After' value from headers (can be seconds or ms)
            # Check ccxt standard attribute first
            if hasattr(e, 'http_headers') and isinstance(e.http_headers, dict):
                 retry_after_header = e.http_headers.get('Retry-After') or e.http_headers.get('retry-after')
            # Fallback check on the underlying requests response if available
            elif hasattr(e, 'response') and hasattr(e.response, 'headers'):
                 retry_after_header = e.response.headers.get('Retry-After') or e.response.headers.get('retry-after')

            # Use a stronger exponential backoff for rate limits: base * (2.0 ^ attempts)
            backoff_wait = base_retry_delay * (2.0 ** attempts)
            backoff_wait *= (1 + (np.random.rand() - 0.5) * 0.2) # Add jitter

            # Parse Retry-After header value if found
            if retry_after_header:
                try:
                    header_wait_seconds = float(retry_after_header)
                    # Assume seconds unless value looks like milliseconds (e.g., > 10000)
                    if header_wait_seconds > 1000: header_wait_seconds /= 1000.0
                    header_wait_seconds += 0.5 # Add a small buffer (0.5 seconds)
                    lg.debug(f"Rate limit 'Retry-After' header: {retry_after_header} -> Parsed wait: {header_wait_seconds:.1f}s")
                except (ValueError, TypeError):
                    lg.warning(f"Could not parse Retry-After header value: {retry_after_header}")
                    header_wait_seconds = 0 # Ignore invalid header

            # Use the longer of calculated backoff or header-suggested wait time
            final_wait_time = max(backoff_wait, header_wait_seconds)
            final_wait_time = min(final_wait_time, 90) # Cap rate limit wait time (e.g., 90 seconds)

            lg.warning(f"{NEON_YELLOW}Rate Limit Exceeded in '{func.__name__}'. "
                       f"Retrying in {final_wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1}). Error: {e}{RESET}")
            wait_time = final_wait_time # Use the determined wait time for the sleep

        # --- Authentication Errors (FATAL - Non-Retryable) ---
        except ccxt.AuthenticationError as e:
             lg.error(f"{NEON_RED}Authentication Error in '{func.__name__}': {e}. This is likely NOT retryable.{RESET}")
             lg.error(f"{NEON_RED}>> Check API Key/Secret validity, permissions, IP whitelist, and environment (Live/Testnet). Aborting call.{RESET}")
             raise e # Re-raise immediately, do not retry

        # --- Invalid Order / Input Errors (FATAL - Non-Retryable) ---
        # These usually indicate a problem with parameters (size, price, etc.)
        except (ccxt.InvalidOrder, ccxt.BadSymbol, ccxt.ArgumentsRequired, ccxt.BadRequest) as e:
            lg.error(f"{NEON_RED}Invalid Request Error in '{func.__name__}': {e}. This indicates an issue with order parameters or symbol. NOT retryable.{RESET}")
            lg.error(f"{NEON_RED}>> Check parameters: {args}, {kwargs}. Aborting call.{RESET}")
            raise e # Re-raise immediately

        # --- Potentially Retryable Exchange-Specific Errors ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str = str(e).lower()
            # Try to get HTTP status code and exchange-specific error code
            http_status_code = getattr(e, 'http_status_code', None)
            exchange_code = getattr(e, 'code', None) # CCXT often parses this

            # Try extracting Bybit's 'retCode' from the message string if ccxt didn't parse it
            if exchange_code is None and 'bybit' in str(type(e)).lower() and 'retcode' in err_str:
                try:
                     # Example: "bybit {"retCode":10006,"retMsg":"Too many visits!"...}"
                     start_index = err_str.find('"retcode":')
                     if start_index != -1:
                          code_part = err_str[start_index + len('"retcode":'):]
                          end_index = code_part.find(',')
                          code_str = code_part[:end_index].strip()
                          if code_str.isdigit(): exchange_code = int(code_str)
                except Exception: pass # Ignore parsing errors

            # List of known Bybit V5 transient/retryable error codes (expand as needed)
            # Ref: https://bybit-exchange.github.io/docs/v5/error_code
            bybit_retry_codes = [
                10001, # Internal server error (param error or server issue)
                10002, # Service unavailable / Server error
                10004, # Sign check error (can be transient timing)
                10006, # Too many requests (might also be caught by RateLimitExceeded)
                10010, # Request expired (timestamp issue, maybe retryable with adjustForTimeDifference)
                10016, # Service temporarily unavailable (maintenance/upgrade)
                # 10018, # Request validation failed (sometimes transient) - Risky to retry blindly
                110001, # Internal error, try again
                130021, # Order qty error (sometimes transient precision/price issue)
                130150, # System busy
                131204, # Cannot connect to matching engine
                170131, # Too many requests (contract specific?)
                170146, # Risk limit cannot be adjusted when the symbol has active order
                # Add other codes based on observation
            ]
            # General retryable error messages (case-insensitive)
            retryable_messages = [
                 "internal server error", "service unavailable", "system busy",
                 "matching engine busy", "please try again", "request timeout",
                 "nonce is too small", # Can happen with clock drift, retry might help
                 "order placement optimization", # Occasional Bybit transient message
            ]

            is_retryable = False
            # Check retryable conditions: Bybit code OR HTTP code OR message content
            if exchange_code in bybit_retry_codes: is_retryable = True
            if not is_retryable and http_status_code in RETRYABLE_HTTP_CODES: is_retryable = True
            if not is_retryable and any(msg in err_str for msg in retryable_messages): is_retryable = True

            if is_retryable:
                 # Use standard backoff for these transient errors
                 wait_time = base_retry_delay * (1.5 ** attempts)
                 wait_time *= (1 + (np.random.rand() - 0.5) * 0.2) # Add jitter
                 wait_time = min(wait_time, 60) # Cap wait time
                 lg.warning(f"{NEON_YELLOW}Potentially Retryable Exchange Error in '{func.__name__}': {e} (Code: {exchange_code}, HTTP: {http_status_code}). "
                            f"Retrying in {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1})...{RESET}")
                 time.sleep(wait_time) # Sleep before the next attempt
                 attempts += 1
                 continue # Go to the next iteration of the while loop directly
            else:
                 # Non-retryable ExchangeError (e.g., insufficient balance, invalid parameters not caught earlier)
                 lg.error(f"{NEON_RED}Non-Retryable Exchange Error in '{func.__name__}': {e} (Code: {exchange_code}, HTTP: {http_status_code}){RESET}")
                 raise e # Re-raise immediately

        # --- Catch any other unexpected error ---
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected Error during API call '{func.__name__}': {e}{RESET}", exc_info=True)
            raise e # Re-raise unexpected errors immediately, do not retry

        # --- Increment Attempt Counter (if a retryable exception occurred and wasn't 'continue'd) ---
        # This block is reached after the sleep for retryable network/rate limit errors
        if attempts <= max_retries:
             # Wait time should have been calculated in the corresponding except block
             calculated_wait_time = locals().get('wait_time', 0)
             if calculated_wait_time > 0:
                 time.sleep(calculated_wait_time)
             else:
                 # Fallback sleep if wait_time wasn't set (shouldn't happen with current logic)
                 lg.warning(f"Wait time not calculated for retry attempt {attempts+1}. Using base delay.")
                 time.sleep(base_retry_delay)
             attempts += 1
        else:
            # Should be handled by the loop condition, but as a safety break
            break

    # --- Max Retries Exceeded ---
    # If the loop completes without returning, it means max retries were hit
    lg.error(f"{NEON_RED}Max retries ({max_retries}) exceeded for API call '{func.__name__}'. Aborting call.{RESET}")
    if last_exception:
        lg.error(f"Last encountered error: {type(last_exception).__name__}: {last_exception}")
        raise last_exception # Raise the last captured exception
    else:
        # Fallback if no exception was captured (shouldn't normally happen)
        raise ccxt.RequestTimeout(f"Max retries exceeded for {func.__name__} but no specific exception was captured during retry loop.")


# --- CCXT Data Fetching Functions (using safe_api_call) ---

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using CCXT's fetch_ticker.
    - Uses safe_api_call for robust fetching with retries.
    - Implements a priority order for selecting the price (e.g., last, mark, close).
    - Converts the selected price to Decimal for precision.
    - Performs validation to ensure the price is positive and finite.

    Args:
        exchange (ccxt.Exchange): The initialized CCXT exchange object.
        symbol (str): The trading symbol (e.g., 'BTC/USDT:USDT').
        logger (logging.Logger): The logger instance.

    Returns:
        Optional[Decimal]: The current price as a Decimal, or None if fetching/parsing fails.
    """
    lg = logger
    try:
        # Fetch ticker data using the robust wrapper
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)

        if not ticker or not isinstance(ticker, dict):
            lg.error(f"Failed to fetch valid ticker data for {symbol} after retries.")
            # safe_api_call should have logged the root cause if retries failed
            return None

        lg.debug(f"Raw Ticker data for {symbol}: {json.dumps(ticker, indent=2, default=str)}") # Log full ticker for debug

        # --- Helper for Safe Decimal Conversion ---
        def to_decimal(value, context_str: str = "price") -> Optional[Decimal]:
            """Safely converts a value to a positive, finite Decimal. Returns None on failure."""
            if value is None: return None
            try:
                # Convert via string to handle floats/ints consistently
                d = Decimal(str(value))
                # Ensure the price is valid (finite number greater than zero)
                if d.is_finite() and d > 0:
                    return d
                else:
                    lg.debug(f"Invalid {context_str} value from ticker (non-finite or non-positive): {value}. Discarding.")
                    return None
            except (InvalidOperation, ValueError, TypeError):
                lg.debug(f"Invalid {context_str} format, cannot convert to Decimal: {value}. Discarding.")
                return None

        # --- Extract Potential Price Candidates ---
        p_last = to_decimal(ticker.get('last'), 'last price')
        p_mark = to_decimal(ticker.get('mark'), 'mark price') # Important for futures/swaps
        p_close = to_decimal(ticker.get('close', ticker.get('last')), 'close price') # Use 'close', fallback to 'last' if close is missing
        p_bid = to_decimal(ticker.get('bid'), 'bid price')
        p_ask = to_decimal(ticker.get('ask'), 'ask price')
        p_avg = to_decimal(ticker.get('average'), 'average price') # Often (High+Low+Close)/3 or VWAP-like

        # Calculate Mid Price if Bid and Ask are valid and spread is reasonable
        p_mid = None
        if p_bid is not None and p_ask is not None:
             if p_ask > p_bid: # Ensure ask is higher than bid
                 spread = p_ask - p_bid
                 spread_pct = (spread / p_ask) * 100 if p_ask > 0 else Decimal('0')
                 # Warn if spread is excessive (e.g., > 1%) - might indicate illiquid market or bad data
                 if spread_pct > Decimal('1.0'):
                      lg.debug(f"Ticker spread for {symbol} is > 1% ({spread_pct:.2f}%). Mid price calculation may be less reliable.")
                 p_mid = (p_bid + p_ask) / Decimal('2')
                 if not p_mid.is_finite(): p_mid = None # Ensure calculation didn't result in non-finite
             else:
                 lg.debug(f"Bid ({p_bid}) is not less than Ask ({p_ask}) for {symbol}. Cannot calculate Mid price.")

        # --- Determine Market Type for Price Priority ---
        market_info = exchange.market(symbol) if symbol in exchange.markets else {}
        is_contract = market_info.get('contract', False) or market_info.get('type') in ['swap', 'future']

        # --- Select Price Based on Priority ---
        # Priority Order: mark (for contracts) > last > close > average > mid > ask > bid
        selected_price: Optional[Decimal] = None
        price_source: str = "N/A"

        if is_contract and p_mark:
            selected_price, price_source = p_mark, "Mark Price"
        elif p_last:
            selected_price, price_source = p_last, "Last Price"
        elif p_close: # Close is often same as Last, but good fallback
            selected_price, price_source = p_close, "Close Price"
        elif p_avg:
            selected_price, price_source = p_avg, "Average Price"
        elif p_mid:
            selected_price, price_source = p_mid, "Mid Price (Bid/Ask)"
        elif p_ask:
            # Using only Ask or Bid is less ideal due to spread
            if p_bid: # Log spread if using Ask as fallback
                 spread_pct = ((p_ask - p_bid) / p_ask) * 100 if p_ask > 0 else Decimal('0')
                 if spread_pct > Decimal('2.0'): # Higher warning threshold if using Ask directly
                      lg.warning(f"Using 'ask' price ({p_ask}) as fallback for {symbol}, but spread seems large ({spread_pct:.2f}%, Bid: {p_bid}).")
            selected_price, price_source = p_ask, "Ask Price (Fallback)"
        elif p_bid:
            # Last resort
            selected_price, price_source = p_bid, "Bid Price (Last Resort Fallback)"

        # --- Final Validation and Return ---
        if selected_price is not None and selected_price.is_finite() and selected_price > 0:
            lg.info(f"Current price ({symbol}): {selected_price} (Source: {price_source})")
            return selected_price
        else:
            lg.error(f"{NEON_RED}Failed to extract a valid positive price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call or during price processing
        lg.error(f"{NEON_RED}Error fetching/processing current price for {symbol}: {e}{RESET}", exc_info=False) # Keep log concise on error
        return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries and robust processing.
    - Uses safe_api_call for fetching.
    - Converts internal timeframe format (e.g., "5") to CCXT format (e.g., "5m").
    - Creates a pandas DataFrame.
    - Performs data cleaning:
        - Converts timestamps to UTC datetime objects.
        - Converts OHLCV columns to Decimal, handling NaNs and invalid values robustly.
        - Drops rows with invalid timestamps or NaN in essential price columns.
        - Checks for and drops rows with inconsistent OHLC values (e.g., High < Low).
        - Sorts by timestamp and removes duplicates.
    - Returns an empty DataFrame on failure or if no valid data remains after cleaning.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Trading symbol.
        timeframe (str): Internal timeframe string (e.g., "1", "5", "60", "D").
        limit (int): Maximum number of klines to fetch.
        logger (Optional[logging.Logger]): Logger instance. Uses default if None.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data (Decimal type) indexed by UTC timestamp,
                      or an empty DataFrame if fetching/processing fails.
    """
    lg = logger or logging.getLogger(__name__) # Use provided logger or get a default one
    empty_df = pd.DataFrame() # Standard empty DataFrame to return on failure

    # Check if exchange supports fetching OHLCV data
    if not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV according to ccxt 'has' attribute.")
        return empty_df

    try:
        # Convert internal timeframe to CCXT's format
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            lg.error(f"Invalid internal timeframe '{timeframe}' provided. Valid map keys: {list(CCXT_INTERVAL_MAP.keys())}. Cannot fetch klines.")
            return empty_df

        lg.debug(f"Fetching up to {limit} klines for {symbol} with timeframe {ccxt_timeframe} (Internal: {timeframe})...")
        # Fetch data using the safe API call wrapper
        ohlcv_data = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe=ccxt_timeframe, limit=limit)

        # Validate the raw data structure
        if ohlcv_data is None or not isinstance(ohlcv_data, list) or len(ohlcv_data) == 0:
            # safe_api_call logs error if it failed after retries. Log warning if it just returned empty.
            if ohlcv_data is not None:
                lg.warning(f"{NEON_YELLOW}No kline data returned by {exchange.id}.fetch_ohlcv for {symbol} {ccxt_timeframe}. Check symbol/interval/exchange status.{RESET}")
            return empty_df

        # Convert raw list of lists into a pandas DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            lg.warning(f"Kline data DataFrame is empty after initial creation for {symbol} {ccxt_timeframe}.")
            return empty_df

        # --- Data Cleaning and Type Conversion ---
        # 1. Convert timestamp to datetime (UTC) and set as index
        try:
            # pd.to_datetime handles various timestamp formats; 'ms' unit is standard for ccxt
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            initial_len_ts = len(df)
            # Drop rows where timestamp conversion failed (resulted in NaT)
            df.dropna(subset=['timestamp'], inplace=True)
            if len(df) < initial_len_ts:
                lg.debug(f"Dropped {initial_len_ts - len(df)} rows with invalid timestamps for {symbol}.")
            if df.empty:
                lg.warning(f"DataFrame became empty after dropping invalid timestamps for {symbol}.")
                return empty_df
            # Set the valid timestamp column as the DataFrame index
            df.set_index('timestamp', inplace=True)
        except Exception as ts_err:
             lg.error(f"Critical error processing timestamps for {symbol}: {ts_err}. Returning empty DataFrame.", exc_info=True)
             return empty_df

        # 2. Convert OHLCV columns to Decimal with robust error handling
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            if col not in df.columns:
                lg.warning(f"Required column '{col}' missing in fetched kline data for {symbol}. Skipping conversion.")
                continue
            try:
                # Helper function for safe conversion to Decimal, returns Decimal('NaN') on failure
                def safe_to_decimal(x, col_name) -> Decimal:
                    """Converts input to Decimal, returns Decimal('NaN') on failure, NaN, non-finite, or non-positive (for price)."""
                    if pd.isna(x) or str(x).strip() == '': return Decimal('NaN')
                    try:
                        d = Decimal(str(x)) # Convert via string representation
                        # Check conditions: must be finite. Prices must be > 0, Volume >= 0.
                        is_price = col_name in ['open', 'high', 'low', 'close']
                        is_valid = d.is_finite() and (d > 0 if is_price else d >= 0)
                        if is_valid:
                            return d
                        else:
                            # lg.debug(f"Invalid Decimal value in '{col_name}': {x} (Non-finite or non-positive price/negative volume)")
                            return Decimal('NaN')
                    except (InvalidOperation, TypeError, ValueError):
                        # lg.debug(f"Conversion error to Decimal for '{x}' in column '{col_name}', returning NaN.")
                        return Decimal('NaN')

                # Apply the safe conversion function to the column
                df[col] = df[col].apply(lambda x: safe_to_decimal(x, col))

            except Exception as conv_err: # Catch unexpected errors during the apply step
                 lg.error(f"Unexpected error converting column '{col}' to Decimal for {symbol}: {conv_err}. Attempting float conversion as fallback.", exc_info=True)
                 # Fallback: try converting to float, coercing errors to standard float NaN
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 # Ensure any remaining non-finite floats (inf, -inf) are also treated as NaN
                 if pd.api.types.is_float_dtype(df[col]):
                     df[col] = df[col].apply(lambda x: x if np.isfinite(x) else np.nan)


        # 3. Drop rows with NaN in essential price columns (Open, High, Low, Close)
        initial_len_nan = len(df)
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, how='any', inplace=True)
        rows_dropped_nan = initial_len_nan - len(df)
        if rows_dropped_nan > 0:
            lg.debug(f"Dropped {rows_dropped_nan} rows with NaN price data for {symbol}.")

        # 4. Additional Sanity Check: Ensure OHLC consistency (e.g., High >= Low, High >= Open, etc.)
        # Convert columns to numeric temporarily if they aren't Decimal (due to fallback)
        for col in essential_price_cols:
            if col in df.columns and not isinstance(df[col].iloc[0], Decimal) and not pd.api.types.is_numeric_dtype(df[col]):
                 df[col] = pd.to_numeric(df[col], errors='coerce')
        # Re-check for NaNs introduced by coercion before comparison
        df.dropna(subset=essential_price_cols, how='any', inplace=True)

        if not df.empty: # Proceed only if DF still has data
            try:
                # Create boolean mask for rows where OHLC logic is violated
                # Uses Decimal comparison if types are correct, otherwise numeric comparison
                invalid_ohlc_mask = (df['high'] < df['low']) | \
                                    (df['high'] < df['open']) | \
                                    (df['high'] < df['close']) | \
                                    (df['low'] > df['open']) | \
                                    (df['low'] > df['close'])
                invalid_count = invalid_ohlc_mask.sum()
                if invalid_count > 0:
                    lg.warning(f"{NEON_YELLOW}Found {invalid_count} klines with inconsistent OHLC data (e.g., High < Low) for {symbol}. Dropping these rows.{RESET}")
                    df = df[~invalid_ohlc_mask] # Keep only rows where the mask is False
            except TypeError as cmp_err:
                # This might happen if data conversion failed unexpectedly, leading to mixed types
                lg.warning(f"Could not perform OHLC consistency check for {symbol} due to type error: {cmp_err}. Skipping check.")
            except Exception as cmp_err:
                 lg.warning(f"Unexpected error during OHLC consistency check for {symbol}: {cmp_err}. Skipping check.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} {ccxt_timeframe} became empty after cleaning (NaN drop or OHLC check).")
            return empty_df

        # 5. Sort by timestamp index (should already be sorted, but ensures) and remove duplicates
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            num_duplicates = df.index.duplicated().sum()
            lg.debug(f"Found {num_duplicates} duplicate timestamps in kline data for {symbol}. Keeping the last entry for each duplicate.")
            # Keep the last occurrence of each duplicated timestamp
            df = df[~df.index.duplicated(keep='last')]

        # --- Final Log and Return ---
        lg.info(f"Successfully fetched and processed {len(df)} valid klines for {symbol} {ccxt_timeframe} (requested limit: {limit})")
        # Log head/tail only if DEBUG level is enabled and DataFrame is not empty
        if lg.isEnabledFor(logging.DEBUG) and not df.empty:
             lg.debug(f"Kline check ({symbol}): First row:\n{df.head(1)}\nLast row:\n{df.tail(1)}")
        return df

    except ValueError as ve: # Catch specific validation errors raised within the function
        lg.error(f"{NEON_RED}Kline fetch/processing error for {symbol}: {ve}{RESET}")
        return empty_df
    except Exception as e:
        # Catch errors from safe_api_call or unexpected errors during processing
        lg.error(f"{NEON_RED}Unexpected error fetching or processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return empty_df


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """
    Fetches order book data using CCXT with retries, validation, and Decimal conversion.
    - Uses safe_api_call for fetching.
    - Validates the structure of the returned order book data.
    - Converts bid/ask prices and amounts to Decimal, ensuring they are finite and positive.
    - Logs warnings for invalid entries but attempts to return partial data if possible.
    - Returns a dictionary containing 'bids' and 'asks' lists (each entry is [Decimal(price), Decimal(amount)]),
      along with original metadata, or None on failure.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): Trading symbol.
        limit (int): Number of order book levels to fetch (depth).
        logger (logging.Logger): Logger instance.

    Returns:
        Optional[Dict]: Order book dictionary with Decimal values, or None if fetching/processing fails.
                        Structure: {'bids': [[price, amount], ...], 'asks': [[price, amount], ...], 'timestamp': ..., ...}
    """
    lg = logger
    # Check if exchange supports fetching the order book
    if not exchange.has.get('fetchOrderBook'):
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook according to ccxt 'has' attribute.")
        return None

    try:
        lg.debug(f"Fetching order book for {symbol} with limit {limit}...")
        # Fetch order book using the safe API call wrapper
        orderbook_raw = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        # Validate the raw response
        if not orderbook_raw: # safe_api_call logs error if retries failed
            lg.warning(f"fetch_order_book for {symbol} returned None or empty after retries.")
            return None
        if not isinstance(orderbook_raw, dict) or \
           'bids' not in orderbook_raw or 'asks' not in orderbook_raw or \
           not isinstance(orderbook_raw['bids'], list) or not isinstance(orderbook_raw['asks'], list):
            lg.warning(f"Invalid order book structure received for {symbol}. Data: {str(orderbook_raw)[:200]}...") # Log snippet
            return None

        # --- Process bids and asks, converting to Decimal ---
        cleaned_book = {
            'bids': [],
            'asks': [],
            # Preserve original metadata if available
            'timestamp': orderbook_raw.get('timestamp'),
            'datetime': orderbook_raw.get('datetime'),
            'nonce': orderbook_raw.get('nonce')
        }
        conversion_errors = 0
        invalid_format_count = 0

        for side in ['bids', 'asks']:
            for entry in orderbook_raw[side]:
                # Ensure entry is a list/tuple with exactly two elements (price, amount)
                if isinstance(entry, (list, tuple)) and len(entry) == 2:
                    try:
                        # Convert price and amount to Decimal via string representation
                        price = Decimal(str(entry[0]))
                        amount = Decimal(str(entry[1]))

                        # Validate: Price must be > 0, Amount must be >= 0, both must be finite
                        if price.is_finite() and price > 0 and amount.is_finite() and amount >= 0:
                            cleaned_book[side].append([price, amount])
                        else:
                            # lg.debug(f"Invalid Decimal price/amount in {side} entry for {symbol}: P={price}, A={amount}") # Can be verbose
                            conversion_errors += 1
                    except (InvalidOperation, ValueError, TypeError):
                        # lg.debug(f"Conversion error for {side} entry: {entry} in {symbol}") # Can be verbose
                        conversion_errors += 1
                else:
                    # Log entries with unexpected format
                    # lg.warning(f"Invalid {side[:-1]} entry format in orderbook for {symbol}: {entry}")
                    invalid_format_count += 1

        # Log summary of cleaning issues if any occurred
        if conversion_errors > 0:
            lg.debug(f"Orderbook ({symbol}): Encountered {conversion_errors} entries with invalid/non-finite/non-positive values during Decimal conversion.")
        if invalid_format_count > 0:
            lg.warning(f"{NEON_YELLOW}Orderbook ({symbol}): Encountered {invalid_format_count} entries with unexpected format (expected [price, amount]).{RESET}")

        # Ensure bids are sorted descending by price and asks ascending by price
        # CCXT usually provides sorted data, but this ensures consistency. Use Decimal comparison.
        try:
            cleaned_book['bids'].sort(key=lambda x: x[0], reverse=True)
            cleaned_book['asks'].sort(key=lambda x: x[0])
        except Exception as sort_err:
             lg.warning(f"Could not sort order book entries for {symbol}: {sort_err}. Data might be unsorted.")

        # Check if the cleaned book is empty
        if not cleaned_book['bids'] and not cleaned_book['asks']:
            lg.warning(f"Orderbook for {symbol} is empty after cleaning/conversion (originally had {len(orderbook_raw['bids'])} bids, {len(orderbook_raw['asks'])} asks).")
            # Still return the structure, as the fetch itself might have succeeded
            return cleaned_book
        elif not cleaned_book['bids']:
             lg.warning(f"Orderbook ({symbol}) contains no valid bids after cleaning.")
        elif not cleaned_book['asks']:
             lg.warning(f"Orderbook ({symbol}) contains no valid asks after cleaning.")

        lg.debug(f"Successfully fetched and processed order book for {symbol} ({len(cleaned_book['bids'])} valid bids, {len(cleaned_book['asks'])} valid asks).")
        return cleaned_book

    except Exception as e:
        # Catch errors raised by safe_api_call or during other processing steps
        lg.error(f"{NEON_RED}Error fetching or processing order book for {symbol}: {e}{RESET}", exc_info=False)
        return None

# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes market data (OHLCV, order book) to generate trading signals.
    - Calculates technical indicators using pandas_ta.
    - Handles Decimal/float type conversions between raw data, TA library, and internal use.
    - Provides helper methods for market precision, limits, and SL/TP calculation.
    - Generates weighted trading signals based on configured indicators and weights.
    """

    def __init__(
        self,
        df: pd.DataFrame, # Expects OHLCV columns with Decimal type from fetch_klines
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        """
        Initializes the TradingAnalyzer.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing OHLCV data (expects Decimal values)
                               indexed by timestamp (UTC).
            logger (logging.Logger): Logger instance for logging messages.
            config (Dict[str, Any]): The bot's configuration dictionary.
            market_info (Dict[str, Any]): Dictionary containing details for the specific market
                                          (precision, limits, symbol, etc.) obtained from CCXT.
        """
        if not isinstance(df, pd.DataFrame):
             raise ValueError("TradingAnalyzer requires a pandas DataFrame.")
        self.df = df.copy() # Work on a copy to avoid modifying the original DataFrame passed in
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "N/A") # Internal interval string
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval) # CCXT timeframe string

        # Dictionary to store the latest calculated indicator values.
        # Uses Decimal for price-based values (OHLC, ATR, Bands, EMAs, PSAR, VWAP) and float for others.
        self.indicator_values: Dict[str, Union[Decimal, float, datetime, None]] = {}
        # Dictionary to map internal indicator keys (e.g., "EMA_Short") to the actual
        # column names generated by pandas_ta in the DataFrame (e.g., "EMA_9").
        self.ta_column_names: Dict[str, Optional[str]] = {}
        # Dictionary to store calculated Fibonacci retracement levels (Decimal prices).
        self.fib_levels_data: Dict[str, Decimal] = {}

        # Load the active weight set configuration for signal scoring
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Active weight set '{self.active_weight_set_name}' not found or is empty in config for {self.symbol}. Signal scores may be zero.{RESET}")
            self.weights = {} # Use empty dict to prevent errors if weights are missing

        # --- Caches for Precision/Limits (populated by getter methods) ---
        # Avoid recalculating these frequently within a single analysis cycle
        self._cached_price_precision: Optional[int] = None
        self._cached_min_tick_size: Optional[Decimal] = None
        self._cached_amount_precision: Optional[int] = None
        self._cached_min_amount_step: Optional[Decimal] = None

        # Perform initial validation and calculations upon instantiation
        self._initialize_analysis()

    def _initialize_analysis(self) -> None:
        """Performs initial checks and calculations when the analyzer is created."""
        if self.df.empty:
             self.logger.warning(f"TradingAnalyzer initialized with an empty DataFrame for {self.symbol}. No calculations will be performed.")
             return

        # Verify required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.logger.error(f"DataFrame for {self.symbol} is missing required columns: {', '.join(missing_cols)}. Cannot perform analysis.")
            self.df = pd.DataFrame() # Clear DF to prevent errors in subsequent methods
            return

        # Verify data types (expecting Decimal from fetch_klines)
        first_valid_idx = self.df['close'].first_valid_index()
        if first_valid_idx is None or not isinstance(self.df.loc[first_valid_idx, 'close'], Decimal):
             self.logger.warning(f"DataFrame 'close' column for {self.symbol} does not appear to contain Decimal values as expected. Calculations might proceed but precision could be affected.")

        # Verify DataFrame contains some non-NaN data in essential columns
        if self.df[essential_price_cols].isnull().all().all():
             self.logger.error(f"DataFrame for {self.symbol} contains all NaN values in required price columns. Cannot perform analysis.")
             self.df = pd.DataFrame() # Clear DF
             return

        # --- Proceed with Initial Calculations ---
        try:
            self._calculate_all_indicators()
            # Update latest values *after* indicators are calculated and potentially back-converted
            self._update_latest_indicator_values()
            # Calculate initial Fibonacci levels
            self.calculate_fibonacci_levels()
        except Exception as init_calc_err:
             # Catch any unexpected errors during the initial calculation phase
             self.logger.error(f"Error during TradingAnalyzer initial indicator/Fibonacci calculation for {self.symbol}: {init_calc_err}", exc_info=True)
             # Consider clearing the DataFrame or setting a 'failed' state depending on severity
             # self.df = pd.DataFrame()

    def _get_ta_col_name(self, base_name: str, result_df_columns: List[str]) -> Optional[str]:
        """
        Helper method to robustly find the actual column name generated by pandas_ta
        for a given internal indicator base name.

        Args:
            base_name (str): Internal identifier for the indicator (e.g., "ATR", "EMA_Short").
            result_df_columns (List[str]): List of column names present in the DataFrame after
                                           TA calculations have been run.

        Returns:
            Optional[str]: The matched column name string, or None if no unambiguous match is found.
        """
        if not result_df_columns: return None # Cannot search if no columns exist

        # --- Dynamically Define Expected Patterns Based on Config ---
        # Use float representation matching pandas_ta common output (e.g., '2.0' for std dev)
        bb_std_dev_str = f"{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BBANDS_STDDEV)):.1f}"
        psar_step_str = f"{float(self.config.get('psar_step', DEFAULT_PSAR_STEP)):g}" # Use 'g' for general format (avoids trailing zeros like 0.020)
        psar_max_str = f"{float(self.config.get('psar_max_step', DEFAULT_PSAR_MAX_STEP)):g}"

        # Get relevant period parameters from config or defaults
        param_keys = [
            ('atr_period', DEFAULT_ATR_PERIOD), ('ema_short_period', DEFAULT_EMA_SHORT_PERIOD),
            ('ema_long_period', DEFAULT_EMA_LONG_PERIOD), ('momentum_period', DEFAULT_MOMENTUM_PERIOD),
            ('cci_period', DEFAULT_CCI_PERIOD), ('williams_r_period', DEFAULT_WILLIAMS_R_PERIOD),
            ('mfi_period', DEFAULT_MFI_PERIOD), ('rsi_period', DEFAULT_RSI_PERIOD),
            ('bollinger_bands_period', DEFAULT_BBANDS_PERIOD), ('sma_10_period', DEFAULT_SMA10_PERIOD),
            ('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD),
            ('stoch_rsi_period', DEFAULT_STOCH_RSI_PERIOD), ('stoch_rsi_rsi_period', DEFAULT_STOCH_RSI_RSI_PERIOD),
            ('stoch_rsi_k_period', DEFAULT_STOCH_RSI_K_PERIOD), ('stoch_rsi_d_period', DEFAULT_STOCH_RSI_D_PERIOD)
        ]
        params = {key: int(self.config.get(key, default)) for key, default in param_keys} # Ensure integers

        # Map internal base names to lists of potential pandas_ta column name patterns
        # These patterns should match the typical output of pandas_ta functions
        expected_patterns = {
            "ATR": [f"ATRr_{params['atr_period']}"], # 'r' suffix often means 'real' or result
            "EMA_Short": [f"EMA_{params['ema_short_period']}"],
            "EMA_Long": [f"EMA_{params['ema_long_period']}"],
            "Momentum": [f"MOM_{params['momentum_period']}"],
            "CCI": [f"CCI_{params['cci_period']}", f"CCI_{params['cci_period']}_0.015"], # Common suffix for constant
            "Williams_R": [f"WILLR_{params['williams_r_period']}"],
            "MFI": [f"MFI_{params['mfi_period']}"],
            "VWAP": ["VWAP", "VWAP_D"], # pandas_ta default VWAP often daily anchored ('_D')
            "PSAR_long": [f"PSARl_{psar_step_str}_{psar_max_str}"], # 'l' for long trend signal
            "PSAR_short": [f"PSARs_{psar_step_str}_{psar_max_str}"], # 's' for short trend signal
            "PSAR_af": [f"PSARaf_{psar_step_str}_{psar_max_str}"], # Acceleration factor
            "PSAR_rev": [f"PSARr_{psar_step_str}_{psar_max_str}"], # Reversal points
            "SMA_10": [f"SMA_{params['sma_10_period']}"],
            "StochRSI_K": [f"STOCHRSIk_{params['stoch_rsi_period']}_{params['stoch_rsi_rsi_period']}_{params['stoch_rsi_k_period']}"],
            "StochRSI_D": [f"STOCHRSId_{params['stoch_rsi_period']}_{params['stoch_rsi_rsi_period']}_{params['stoch_rsi_k_period']}_{params['stoch_rsi_d_period']}"],
            "RSI": [f"RSI_{params['rsi_period']}"],
            "BB_Lower": [f"BBL_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Middle": [f"BBM_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Upper": [f"BBU_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Bandwidth": [f"BBB_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Percent": [f"BBP_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            # Custom name used for Volume MA calculation
            "Volume_MA": [f"VOL_SMA_{params['volume_ma_period']}"]
        }

        patterns_to_check = expected_patterns.get(base_name)
        if not patterns_to_check:
            # self.logger.debug(f"No expected column pattern defined for indicator base name: '{base_name}'")
            return None

        # --- Search Strategy (from most specific to least specific) ---
        # 1. Exact Match (Case-Sensitive) - Most reliable
        for pattern in patterns_to_check:
            if pattern in result_df_columns:
                # self.logger.debug(f"Mapped '{base_name}' to column '{pattern}' (Exact Match)")
                return pattern

        # 2. Case-Insensitive Exact Match
        patterns_lower = [p.lower() for p in patterns_to_check]
        # Create mapping from lower-case column name to original case for efficient lookup
        cols_lower_map = {col.lower(): col for col in result_df_columns}
        for i, pattern_lower in enumerate(patterns_lower):
             if pattern_lower in cols_lower_map:
                  original_col_name = cols_lower_map[pattern_lower]
                  # self.logger.debug(f"Mapped '{base_name}' to column '{original_col_name}' (Case-Insensitive Exact Match)")
                  return original_col_name

        # 3. Starts With Match (Case-Insensitive) - Handles potential suffixes added by TA lib
        for pattern in patterns_to_check:
            pattern_lower = pattern.lower()
            for col in result_df_columns:
                col_lower = col.lower()
                # Check if column starts with the pattern (case-insensitive)
                if col_lower.startswith(pattern_lower):
                     # Sanity check: ensure the remaining suffix doesn't look like a different indicator name
                     suffix = col[len(pattern):]
                     if not any(c.isalpha() for c in suffix): # Allow numbers, underscores, periods in suffix
                          # self.logger.debug(f"Mapped '{base_name}' to column '{col}' (StartsWith Match: '{pattern}')")
                          return col
                     # else: suffix contains letters, might be a different indicator, continue search

        # 4. Fallback: Simple base name substring check (use with caution)
        # Example: base_name 'StochRSI_K' -> simple_base 'stochrsi'
        simple_base = base_name.split('_')[0].lower()
        potential_matches = [col for col in result_df_columns if simple_base in col.lower()]

        if len(potential_matches) == 1:
             match = potential_matches[0]
             # self.logger.debug(f"Mapped '{base_name}' to '{match}' via unique simple substring search ('{simple_base}').")
             return match
        elif len(potential_matches) > 1:
              # Ambiguous: Multiple columns contain the simple base name.
              # Try to resolve by checking if one of the full expected patterns is among the matches.
              for pattern in patterns_to_check:
                   if pattern in potential_matches: # Check exact expected pattern
                        # self.logger.debug(f"Resolved ambiguous substring match for '{base_name}' to expected pattern '{pattern}'.")
                        return pattern
                   if pattern.lower() in [p.lower() for p in potential_matches]: # Check case-insensitive expected pattern
                        # Find the original case match
                        original_case_match = next((p for p in potential_matches if p.lower() == pattern.lower()), None)
                        if original_case_match:
                             # self.logger.debug(f"Resolved ambiguous substring match for '{base_name}' to expected pattern '{original_case_match}' (case-insensitive).")
                             return original_case_match

              # If still ambiguous after checking expected patterns, it's safer to return None
              self.logger.warning(f"{NEON_YELLOW}Ambiguous substring match for '{base_name}' ('{simple_base}'): Found {potential_matches}. Could not resolve clearly based on expected patterns: {patterns_to_check}. No column mapped.{RESET}")
              return None

        # If no match found by any method
        self.logger.debug(f"Could not find matching column name for indicator '{base_name}' (Expected patterns: {patterns_to_check}) in DataFrame columns: {result_df_columns}")
        return None


    def _calculate_all_indicators(self):
        """
        Calculates all technical indicators specified and enabled in the configuration
        using the pandas_ta library. Handles data type conversions for compatibility.
        Updates the internal DataFrame with the calculated indicator columns.
        """
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty for {self.symbol}, cannot calculate indicators.")
            return

        # --- Determine Minimum Required Data Length ---
        # Estimate required length based on the longest period among enabled & weighted indicators
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        active_weights = self.weights # Use weights loaded during init

        # Helper to add period requirement if indicator is active
        def add_req_if_active(indicator_key, config_period_key, default_period):
            """Adds period requirement if indicator is enabled and has non-zero weight."""
            is_enabled = indicators_config.get(indicator_key, False)
            try: weight = float(active_weights.get(indicator_key, 0.0))
            except (ValueError, TypeError): weight = 0.0

            if is_enabled and weight > 1e-9: # Check weight > small tolerance
                try:
                    period = int(self.config.get(config_period_key, default_period))
                    if period > 0: required_periods.append(period)
                    else: self.logger.warning(f"Invalid zero/negative period configured for {config_period_key} ({period}). Ignoring for length check.")
                except (ValueError, TypeError):
                     self.logger.warning(f"Invalid period format for {config_period_key} ('{self.config.get(config_period_key)}'). Ignoring for length check.")

        # Add requirements for indicators with standard periods
        add_req_if_active("atr", "atr_period", DEFAULT_ATR_PERIOD)
        add_req_if_active("momentum", "momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req_if_active("cci", "cci_period", DEFAULT_CCI_PERIOD)
        add_req_if_active("wr", "williams_r_period", DEFAULT_WILLIAMS_R_PERIOD)
        add_req_if_active("mfi", "mfi_period", DEFAULT_MFI_PERIOD)
        add_req_if_active("sma_10", "sma_10_period", DEFAULT_SMA10_PERIOD)
        add_req_if_active("rsi", "rsi_period", DEFAULT_RSI_PERIOD)
        add_req_if_active("bollinger_bands", "bollinger_bands_period", DEFAULT_BBANDS_PERIOD)
        add_req_if_active("volume_confirmation", "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Fibonacci period is a lookback window, include it
        try: fib_period = int(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))
        except (ValueError, TypeError): fib_period = DEFAULT_FIB_PERIOD
        if fib_period > 0: required_periods.append(fib_period)

        # Compound indicators: EMA Alignment requires both short and long EMAs
        if indicators_config.get("ema_alignment", False) and float(active_weights.get("ema_alignment", 0.0)) > 1e-9:
             add_req_if_active("ema_alignment", "ema_short_period", DEFAULT_EMA_SHORT_PERIOD) # Use proxy key
             add_req_if_active("ema_alignment", "ema_long_period", DEFAULT_EMA_LONG_PERIOD)

        # StochRSI requires its main period and the underlying RSI period
        if indicators_config.get("stoch_rsi", False) and float(active_weights.get("stoch_rsi", 0.0)) > 1e-9:
             add_req_if_active("stoch_rsi", "stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD)
             add_req_if_active("stoch_rsi", "stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD)

        # Calculate minimum required length, add a buffer (e.g., 30 bars) for indicator stabilization
        min_required_data = max(required_periods) + 30 if required_periods else 50 # Default buffer if no periods found

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient kline data ({len(self.df)} points) for {self.symbol} {self.ccxt_interval} "
                                f"to reliably calculate all active indicators (min recommended: {min_required_data} based on max period: {max(required_periods) if required_periods else 'N/A'}). "
                                f"Results may contain NaNs or be inaccurate.{RESET}")
             # Proceed with calculation, but be aware of potential issues in results

        try:
            # --- Prepare DataFrame for pandas_ta ---
            # pandas_ta generally works best with float types for OHLCV.
            # Convert Decimal columns to float temporarily, storing original types.
            df_calc = self.df # Use alias for clarity, modification happens on the instance's copy
            original_types = {}
            cols_to_float = ['open', 'high', 'low', 'close', 'volume']

            for col in cols_to_float:
                 if col in df_calc.columns:
                     # Check type of first non-NaN value to determine original type
                     first_valid_idx = df_calc[col].first_valid_index()
                     if first_valid_idx is not None:
                          col_type = type(df_calc.loc[first_valid_idx, col])
                          original_types[col] = col_type
                          # Only convert if the original type was Decimal
                          if col_type == Decimal:
                               # self.logger.debug(f"Converting Decimal column '{col}' to float for TA calculation.")
                               # Apply conversion robustly: finite Decimals -> float, non-finite -> np.nan
                               df_calc[col] = df_calc[col].apply(
                                   lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan
                               )
                          elif not pd.api.types.is_numeric_dtype(df_calc[col]):
                              # If original type wasn't Decimal and isn't numeric, attempt conversion
                              self.logger.debug(f"Column '{col}' is not Decimal or numeric ({col_type}), attempting conversion to float.")
                              df_calc[col] = pd.to_numeric(df_calc[col], errors='coerce')
                     else: # Column is all NaN
                          original_types[col] = None # Mark as unknown original type
                          # self.logger.debug(f"Column '{col}' contains only NaN values, skipping conversion.")

            # --- Dynamically Create pandas_ta Strategy ---
            ta_strategy = ta.Strategy(
                name="SXS_Dynamic_Strategy",
                description="Calculates indicators based on sxsBot config file",
                ta=[] # Initialize empty list of indicators to add
            )

            # --- Map internal keys to pandas_ta function names and parameters ---
            # Use lambda functions to fetch parameters from self.config at the time of calculation
            # Ensure parameters are cast to the types expected by pandas_ta (int for lengths, float for std/step)
            ta_map = {
                 # Indicator Key : { pandas_ta_function_name, parameter_name: lambda: type_cast(self.config.get(...)), ... }
                 "atr": {"kind": "atr", "length": lambda: int(self.config.get("atr_period", DEFAULT_ATR_PERIOD))},
                 "ema_short": {"kind": "ema", "length": lambda: int(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))},
                 "ema_long": {"kind": "ema", "length": lambda: int(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))},
                 "momentum": {"kind": "mom", "length": lambda: int(self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))},
                 "cci": {"kind": "cci", "length": lambda: int(self.config.get("cci_period", DEFAULT_CCI_PERIOD))},
                 "wr": {"kind": "willr", "length": lambda: int(self.config.get("williams_r_period", DEFAULT_WILLIAMS_R_PERIOD))},
                 "mfi": {"kind": "mfi", "length": lambda: int(self.config.get("mfi_period", DEFAULT_MFI_PERIOD))},
                 "sma_10": {"kind": "sma", "length": lambda: int(self.config.get("sma_10_period", DEFAULT_SMA10_PERIOD))},
                 "rsi": {"kind": "rsi", "length": lambda: int(self.config.get("rsi_period", DEFAULT_RSI_PERIOD))},
                 "vwap": {"kind": "vwap"}, # VWAP typically uses default daily anchoring in pandas_ta
                 "psar": {"kind": "psar",
                          "step": lambda: float(self.config.get("psar_step", DEFAULT_PSAR_STEP)),
                          "max_step": lambda: float(self.config.get("psar_max_step", DEFAULT_PSAR_MAX_STEP))},
                 "stoch_rsi": {"kind": "stochrsi",
                               "length": lambda: int(self.config.get("stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD)),
                               "rsi_length": lambda: int(self.config.get("stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD)),
                               "k": lambda: int(self.config.get("stoch_rsi_k_period", DEFAULT_STOCH_RSI_K_PERIOD)),
                               "d": lambda: int(self.config.get("stoch_rsi_d_period", DEFAULT_STOCH_RSI_D_PERIOD))},
                 "bollinger_bands": {"kind": "bbands",
                                     "length": lambda: int(self.config.get("bollinger_bands_period", DEFAULT_BBANDS_PERIOD)),
                                     "std": lambda: float(self.config.get("bollinger_bands_std_dev", DEFAULT_BBANDS_STDDEV))},
                 # Note: Volume MA is calculated separately below, not added to ta_map
            }

            # --- Add Indicators to Strategy Based on Config/Weights ---
            calculated_indicator_keys = set() # Track which base indicators are added to avoid duplicates

            # Always calculate ATR if possible, as it's needed for SL/TP/BE sizing, even if not weighted for signals
            if "atr" in ta_map:
                 try:
                     params = {k: v() for k, v in ta_map["atr"].items() if k != 'kind'} # Extract params using lambdas
                     ta_strategy.ta.append(ta.Indicator(ta_map["atr"]["kind"], **params))
                     calculated_indicator_keys.add("atr")
                     # self.logger.debug(f"Adding ATR to TA strategy with params: {params}")
                 except Exception as e: self.logger.error(f"Error preparing ATR indicator for strategy: {e}")

            # Add other indicators only if they are enabled AND have a non-zero weight in the active set
            for key, is_enabled in indicators_config.items():
                 if key == "atr": continue # Already handled
                 try: weight = float(active_weights.get(key, 0.0))
                 except (ValueError, TypeError): weight = 0.0

                 # Skip if disabled in config OR has zero weight in the active set
                 if not is_enabled or weight < 1e-9: continue

                 # Handle compound indicators or indicators needing special logic
                 if key == "ema_alignment":
                      # Requires both short and long EMAs to be calculated
                      for ema_key in ["ema_short", "ema_long"]:
                          if ema_key not in calculated_indicator_keys and ema_key in ta_map:
                               try:
                                   params = {k: v() for k, v in ta_map[ema_key].items() if k != 'kind'}
                                   ta_strategy.ta.append(ta.Indicator(ta_map[ema_key]["kind"], **params))
                                   calculated_indicator_keys.add(ema_key)
                                   # self.logger.debug(f"Adding {ema_key} (for ema_alignment) to TA strategy with params: {params}")
                               except Exception as e: self.logger.error(f"Error preparing {ema_key} indicator for strategy: {e}")
                 elif key == "volume_confirmation":
                      # Volume MA is calculated separately after the main strategy run
                      pass
                 elif key == "orderbook":
                      # Order book analysis doesn't use pandas_ta
                      pass
                 elif key in ta_map:
                      # Check if this base indicator was already added (e.g., if EMA was added via ema_alignment)
                      if key not in calculated_indicator_keys:
                           try:
                               indicator_def = ta_map[key]
                               params = {k: v() for k, v in indicator_def.items() if k != 'kind'}
                               ta_strategy.ta.append(ta.Indicator(indicator_def["kind"], **params))
                               calculated_indicator_keys.add(key) # Mark base key as calculated
                               # self.logger.debug(f"Adding {key} to TA strategy with params: {params}")
                           except Exception as e: self.logger.error(f"Error preparing {key} indicator for strategy: {e}")
                 else:
                      # This indicates a mismatch between config['indicators'], config['weights'], and ta_map/special handling
                      self.logger.warning(f"Indicator '{key}' is enabled and weighted but has no calculation definition in ta_map or special handling. It will be ignored.")

            # --- Execute the TA Strategy ---
            if ta_strategy.ta: # Only run if indicators were actually added
                 self.logger.info(f"Running pandas_ta strategy '{ta_strategy.name}' with {len(ta_strategy.ta)} indicators for {self.symbol}...")
                 try:
                     # df.ta.strategy() applies the indicators and appends columns to df_calc inplace
                     df_calc.ta.strategy(ta_strategy, append=True)
                     self.logger.info(f"Pandas_ta strategy calculation complete for {self.symbol}.")
                 except Exception as ta_err:
                      self.logger.error(f"{NEON_RED}Error running pandas_ta strategy for {self.symbol}: {ta_err}{RESET}", exc_info=True)
                      # Allow continuation, but subsequent steps might fail if indicators are missing
            else:
                 self.logger.info(f"No pandas_ta indicators added to the strategy based on config and weights for {self.symbol}.")

            # --- Calculate Volume Moving Average Separately ---
            vol_key = "volume_confirmation"
            vol_ma_p = 0
            try: vol_ma_p = int(self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
            except (ValueError, TypeError): pass

            # Calculate only if enabled, weighted, and period is valid
            if indicators_config.get(vol_key, False) and float(active_weights.get(vol_key, 0.0)) > 1e-9 and vol_ma_p > 0:
                 try:
                     vol_ma_col = f"VOL_SMA_{vol_ma_p}" # Define a unique column name
                     # Ensure 'volume' column exists and is numeric (should be float after conversion)
                     if 'volume' in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc['volume']):
                          # Use pandas_ta directly for SMA on the volume column
                          # Fill potential NaNs in volume with 0 before calculating SMA to avoid propagation
                          df_calc[vol_ma_col] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_p)
                          # self.logger.debug(f"Calculated Volume MA ({vol_ma_col}) for {self.symbol}.")
                          calculated_indicator_keys.add("volume_ma") # Mark for column mapping
                     else:
                          self.logger.warning(f"Volume column missing or not numeric in DataFrame for {self.symbol}, cannot calculate Volume MA.")
                 except Exception as vol_ma_err:
                      self.logger.error(f"Error calculating Volume MA for {self.symbol}: {vol_ma_err}")

            # --- Map Internal Names to Actual DataFrame Column Names ---
            # Get all column names present after calculations
            final_df_columns = df_calc.columns.tolist()
            # Define the mapping from internal keys (used in checks/scoring) to the base names used in _get_ta_col_name
            indicator_mapping = {
                # Internal Name : TA Indicator Base Name (used in _get_ta_col_name patterns)
                "ATR": "ATR", "EMA_Short": "EMA_Short", "EMA_Long": "EMA_Long",
                "Momentum": "Momentum", "CCI": "CCI", "Williams_R": "Williams_R", "MFI": "MFI",
                "SMA_10": "SMA_10", "RSI": "RSI", "VWAP": "VWAP",
                # PSAR generates multiple columns; map the specific ones needed by checks
                "PSAR_long": "PSAR_long", "PSAR_short": "PSAR_short",
                # StochRSI generates K and D; map both if used
                "StochRSI_K": "StochRSI_K", "StochRSI_D": "StochRSI_D",
                # BBands generates multiple; map components needed by checks
                "BB_Lower": "BB_Lower", "BB_Middle": "BB_Middle", "BB_Upper": "BB_Upper",
                "Volume_MA": "Volume_MA" # Use the custom name defined above
            }
            self.ta_column_names = {} # Clear previous mapping
            for internal_name, ta_base_name in indicator_mapping.items():
                 # Find the actual column name using the robust helper method
                 mapped_col = self._get_ta_col_name(ta_base_name, final_df_columns)
                 if mapped_col:
                     self.ta_column_names[internal_name] = mapped_col
                 # else: Warning logged by _get_ta_col_name if not found

            # --- Convert Selected Columns Back to Decimal (if original was Decimal) ---
            # Prioritize converting indicators used in precise calculations (ATR, price-based levels)
            # Check if original 'close' price was Decimal as a proxy for whether conversion is needed/meaningful
            if original_types.get('close') == Decimal:
                cols_to_decimalize = ["ATR", "BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long",
                                      "PSAR_short", "VWAP", "SMA_10", "EMA_Short", "EMA_Long"]

                for key in cols_to_decimalize:
                    col_name = self.ta_column_names.get(key)
                    # Check if indicator was calculated, column name found, and column exists
                    if col_name and col_name in df_calc.columns:
                         # Check if the column actually contains float data (might be object/int if errors occurred)
                         if pd.api.types.is_float_dtype(df_calc[col_name]):
                             try:
                                 # self.logger.debug(f"Converting calculated column '{col_name}' (for '{key}') back to Decimal.")
                                 # Convert float column back to Decimal, handling potential NaNs/infs robustly
                                 df_calc[col_name] = df_calc[col_name].apply(
                                     lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                                 )
                             except (ValueError, TypeError, InvalidOperation) as conv_err:
                                  self.logger.error(f"Failed to convert TA column '{col_name}' back to Decimal: {conv_err}. Leaving as float.")
                         # else: Column is not float, skip conversion attempt

            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Final DF columns sample: {self.df.columns.tolist()[:15]}...")
            self.logger.debug(f"Mapped TA column names for {self.symbol}: {self.ta_column_names}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Critical error during indicator calculation setup or execution for {self.symbol}: {e}{RESET}", exc_info=True)
            # Consider clearing the DataFrame or setting flags to prevent further use
            # self.df = pd.DataFrame()

    def _update_latest_indicator_values(self):
        """
        Extracts the latest values for OHLCV and all calculated indicators from the
        DataFrame and stores them in the `self.indicator_values` dictionary.
        Handles type consistency (Decimal for price/ATR, float for others) and NaNs.
        """
        self.indicator_values = {} # Reset dictionary

        if self.df.empty:
            self.logger.warning(f"Cannot update latest indicator values: DataFrame empty for {self.symbol}.")
            return
        try:
            # Ensure index is sorted chronologically to get the truly latest row
            if not self.df.index.is_monotonic_increasing:
                 self.logger.warning(f"DataFrame index for {self.symbol} is not sorted. Sorting before extracting latest values.")
                 self.df.sort_index(inplace=True)

            # Get the last row of the DataFrame
            latest_row = self.df.iloc[-1]
            latest_timestamp = latest_row.name # Get timestamp from the index
            self.indicator_values["Timestamp"] = latest_timestamp # Store timestamp as datetime object

        except IndexError:
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be empty or calculations failed.")
            return
        except Exception as e:
             self.logger.error(f"Unexpected error getting latest row for {self.symbol}: {e}", exc_info=True)
             return

        # --- Process Base OHLCV Columns ---
        # These should ideally be Decimal from fetch_klines or after reconversion
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            key_name = base_col.capitalize() # e.g., 'Open'
            value = Decimal('NaN') # Default to Decimal NaN
            if base_col in latest_row.index:
                 raw_value = latest_row[base_col]
                 if isinstance(raw_value, Decimal):
                      value = raw_value if raw_value.is_finite() else Decimal('NaN')
                 elif pd.notna(raw_value): # Handle case where it might be float/int after fallback
                      try:
                           dec_val = Decimal(str(raw_value))
                           value = dec_val if dec_val.is_finite() else Decimal('NaN')
                      except (InvalidOperation, ValueError, TypeError):
                           value = Decimal('NaN') # Failed conversion
                 # else: raw_value was None or NaN, value remains Decimal('NaN')
            self.indicator_values[key_name] = value

        # --- Process TA Indicators Using Mapped Column Names ---
        for key, col_name in self.ta_column_names.items():
            # Determine expected type (Decimal for price-based, float otherwise)
            is_price_based = key in ["ATR", "BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long",
                                     "PSAR_short", "VWAP", "SMA_10", "EMA_Short", "EMA_Long"]
            target_type = Decimal if is_price_based else float
            nan_value = Decimal('NaN') if target_type == Decimal else np.nan
            value = nan_value # Default to appropriate NaN

            if col_name and col_name in latest_row.index:
                raw_value = latest_row[col_name]
                # Check if the value is valid (not None, not pd.NA, etc.)
                if pd.notna(raw_value):
                    try:
                        # Attempt conversion to the target type
                        if target_type == Decimal:
                             # Convert via string, check finiteness
                             converted_value = Decimal(str(raw_value))
                             value = converted_value if converted_value.is_finite() else nan_value
                        else: # Target is float
                             converted_value = float(raw_value)
                             value = converted_value if np.isfinite(converted_value) else nan_value
                    except (ValueError, TypeError, InvalidOperation):
                        # self.logger.debug(f"Could not convert TA value {key} ('{col_name}': {raw_value}) to {target_type}. Storing NaN.")
                        value = nan_value # Use appropriate NaN on conversion failure
                # else: raw_value is already NaN/None, value remains default NaN

            # Store the processed value or the appropriate NaN type
            self.indicator_values[key] = value

        # --- Log Summary of Latest Values (formatted, DEBUG level) ---
        if self.logger.isEnabledFor(logging.DEBUG):
            log_vals = {}
            price_prec = self.get_price_precision_places() # Use helper for precision
            amount_prec = self.get_amount_precision_places() # Use helper

            # Define key categories for formatting consistency
            price_keys = ['Open','High','Low','Close','ATR','BB_Lower','BB_Middle','BB_Upper',
                          'PSAR_long','PSAR_short','VWAP','SMA_10','EMA_Short','EMA_Long']
            amount_keys = ['Volume', 'Volume_MA'] # Treat Volume MA like an amount for formatting
            other_float_keys = ['Momentum', 'StochRSI_K', 'StochRSI_D', 'RSI', 'CCI', 'Williams_R', 'MFI']

            for k, v in self.indicator_values.items():
                if k == "Timestamp":
                    log_vals[k] = v.strftime('%Y-%m-%d %H:%M:%S %Z') if isinstance(v, datetime) else str(v)
                    continue

                formatted_val = "NaN" # Default for None or NaN types
                try:
                    if isinstance(v, Decimal) and v.is_finite():
                         # Use price or amount precision based on key category
                         prec = price_prec if k in price_keys else amount_prec if k in amount_keys else 8 # Default precision for other Decimals
                         formatted_val = f"{v:.{prec}f}"
                    elif isinstance(v, float) and np.isfinite(v):
                         # Use amount precision for volume-like floats, otherwise default float precision
                         prec = amount_prec if k in amount_keys else 5 # Default precision for other floats
                         formatted_val = f"{v:.{prec}f}"
                    elif isinstance(v, int): # Handle integers directly
                         formatted_val = str(v)
                except ValueError: # Handle invalid precision value if getters failed
                    formatted_val = str(v)

                # Only include non-NaN finite values in the log summary
                if formatted_val != "NaN":
                     log_vals[k] = formatted_val

            if log_vals:
                 # Sort keys for consistent log output: Prices, Amounts, Others alphabetically
                 sort_order = {k: 0 for k in price_keys}
                 sort_order.update({k: 1 for k in amount_keys})
                 sorted_keys = sorted(log_vals.keys(), key=lambda x: (sort_order.get(x, 2), x)) # Prices first, then amounts, then others
                 sorted_log_vals = {k: log_vals[k] for k in sorted_keys}
                 self.logger.debug(f"Latest indicator values updated ({self.symbol}): {json.dumps(sorted_log_vals)}")
            else:
                 self.logger.warning(f"No valid latest indicator values could be determined for {self.symbol} after processing.")

    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """
        Calculates Fibonacci retracement levels based on the High/Low range over a specified window.
        - Uses Decimal precision for calculations.
        - Quantizes results based on market's minimum tick size.
        - Stores results in `self.fib_levels_data`.

        Args:
            window (Optional[int]): The lookback period (number of bars) for finding High/Low.
                                    Uses 'fibonacci_period' from config if None.

        Returns:
            Dict[str, Decimal]: Dictionary of Fibonacci levels, e.g., {"Fib_38.2%": Decimal("...")}.
                                Returns empty dict if calculation fails.
        """
        # Use window from config if not provided, ensure it's an integer > 0
        if window is None:
            try: window = int(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))
            except (ValueError, TypeError): window = DEFAULT_FIB_PERIOD
        if not isinstance(window, int) or window <= 1:
            self.logger.warning(f"Invalid window ({window}) for Fibonacci calculation on {self.symbol}. Needs integer > 1. Using default {DEFAULT_FIB_PERIOD}.")
            window = DEFAULT_FIB_PERIOD

        self.fib_levels_data = {} # Clear previous calculation results

        # Basic validation checks
        if self.df.empty or 'high' not in self.df.columns or 'low' not in self.df.columns:
             self.logger.debug(f"Fibonacci calc skipped for {self.symbol}: DataFrame empty or missing high/low columns.")
             return {}
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)} bars) for Fibonacci calculation (requires {window} bars) on {self.symbol}.")
            return {}

        # Get the relevant slice of the DataFrame
        df_slice = self.df.tail(window)

        try:
            # Extract high/low series, drop NaNs, find max/min (should be Decimal type)
            high_series = df_slice["high"].dropna()
            low_series = df_slice["low"].dropna()

            if high_series.empty or low_series.empty:
                 self.logger.warning(f"No valid high/low data points found in the last {window} bars for Fibonacci calculation on {self.symbol}.")
                 return {}

            # Find the highest high and lowest low within the window
            high_price = high_series.max()
            low_price = low_series.min()

            # Ensure we obtained valid, finite Decimal prices
            if not isinstance(high_price, Decimal) or not high_price.is_finite() or \
               not isinstance(low_price, Decimal) or not low_price.is_finite():
                self.logger.warning(f"Could not find valid finite high/low Decimal prices for Fibonacci (Window: {window}) on {self.symbol}. High: {high_price}, Low: {low_price}")
                return {}

            # --- Calculate Fibonacci Levels using Decimal Arithmetic ---
            price_range = high_price - low_price

            # Get market tick size for quantization
            min_tick = self.get_min_tick_size()
            if not min_tick.is_finite() or min_tick <= 0:
                # Fallback if tick size is invalid (shouldn't happen with robust getter)
                price_precision = self.get_price_precision_places()
                quantizer = Decimal('1e-' + str(price_precision))
                self.logger.warning(f"Invalid min_tick_size ({min_tick}), using precision-based quantizer ({quantizer}) for Fibonacci.")
            else:
                quantizer = min_tick # Use the market's minimum price increment

            calculated_levels = {}
            if price_range < quantizer: # Handle very small or zero range
                # If range is smaller than tick size, all levels effectively collapse to the high/low
                if price_range <= 0:
                    self.logger.debug(f"Fibonacci range is zero or negative (High={high_price}, Low={low_price}) for {self.symbol}. Setting all levels to High price.")
                    level_price_quantized = high_price.quantize(quantizer, rounding=ROUND_DOWN)
                else: # Range is positive but smaller than one tick
                     self.logger.debug(f"Fibonacci range ({price_range}) is smaller than tick size ({quantizer}) for {self.symbol}. Levels will likely collapse.")
                     # Treat high/low as the effective levels
                     level_price_quantized = high_price.quantize(quantizer, rounding=ROUND_DOWN) # Use high for consistency

                # Assign the single price to all standard levels
                calculated_levels = {f"Fib_{level_pct * 100:.1f}%": level_price_quantized for level_pct in FIB_LEVELS}
            else:
                # Calculate normal levels based on the range
                for level_pct_str in map(str, FIB_LEVELS): # Iterate through standard levels
                    level_pct = Decimal(level_pct_str)
                    level_name = f"Fib_{level_pct * 100:.1f}%" # e.g., "Fib_38.2%"

                    # Standard Retracement Level price = High - (Range * Percentage)
                    level_price_raw = high_price - (price_range * level_pct)

                    # Quantize the calculated level price based on market tick size
                    # Round DOWN for levels calculated from High (ensures level <= raw calculation)
                    calculated_levels[level_name] = level_price_raw.quantize(quantizer, rounding=ROUND_DOWN)

            # Store the calculated levels and log them
            self.fib_levels_data = calculated_levels
            price_prec = self.get_price_precision_places()
            # Format levels for logging
            log_levels = {k: f"{v:.{price_prec}f}" for k, v in calculated_levels.items()}
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol} (Window: {window}, High: {high_price:.{price_prec}f}, Low: {low_price:.{price_prec}f}, Tick: {min_tick}): {log_levels}")
            return calculated_levels

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Fibonacci calc error for {self.symbol}: DataFrame missing column '{e}'. Ensure OHLCV data is present.{RESET}")
            return {}
        except (ValueError, TypeError, InvalidOperation) as e:
             self.logger.error(f"{NEON_RED}Fibonacci calc error for {self.symbol}: Invalid data type or operation during calculation. {e}{RESET}")
             return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            return {}

    # --- Precision and Limit Helper Methods ---
    # These methods extract precision and limit information from the market_info dictionary,
    # providing fallbacks and caching results for efficiency within an analysis cycle.

    def get_price_precision_places(self) -> int:
        """Determines price precision (number of decimal places) from market info."""
        # Return cached value if already calculated for this instance
        if self._cached_price_precision is not None:
            return self._cached_price_precision

        precision = None
        source = "Unknown"
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price') # This can be int (places) or float/str (tick size)

            if price_precision_val is not None:
                # Case 1: Integer value represents decimal places directly
                if isinstance(price_precision_val, int) and price_precision_val >= 0:
                    precision, source = price_precision_val, "market.precision.price (int)"
                # Case 2: Float/String value represents tick size; calculate places from it
                else:
                    try:
                        tick_size = Decimal(str(price_precision_val))
                        if tick_size.is_finite() and tick_size > 0:
                            # Number of decimal places is the absolute value of the exponent
                            precision = abs(tick_size.normalize().as_tuple().exponent)
                            source = f"market.precision.price (tick: {tick_size})"
                        else: pass # Invalid tick size value, proceed to fallbacks
                    except (TypeError, ValueError, InvalidOperation): pass # Error converting tick size

            # Fallback 1: Infer from limits.price.min (if it resembles a tick size)
            if precision is None:
                min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
                if min_price_val is not None:
                    try:
                        min_price_tick = Decimal(str(min_price_val))
                        # Heuristic: if min price is between 0 and 1, treat it as tick size
                        if min_price_tick.is_finite() and 0 < min_price_tick < Decimal('1'):
                            precision = abs(min_price_tick.normalize().as_tuple().exponent)
                            source = f"market.limits.price.min (tick: {min_price_tick})"
                    except (TypeError, ValueError, InvalidOperation): pass

            # Fallback 2: Infer from last close price's decimal places (least reliable)
            if precision is None:
                last_close = self.indicator_values.get("Close") # Assumes latest values updated
                if isinstance(last_close, Decimal) and last_close.is_finite() and last_close > 0:
                    try:
                        # Get decimal places from the normalized exponent
                        p = abs(last_close.normalize().as_tuple().exponent)
                        # Set a reasonable range for crypto price decimal places (e.g., 0 to 12)
                        if 0 <= p <= 12:
                            precision = p
                            source = f"Inferred from Last Close Price ({last_close})"
                    except Exception: pass # Ignore potential errors during inference

        except Exception as e:
            self.logger.warning(f"Error determining price precision places for {self.symbol}: {e}. Using default.")

        # --- Final Default Fallback ---
        if precision is None:
            default_precision = 4 # A common default for many USDT pairs
            precision = default_precision
            source = f"Default ({default_precision})"
            self.logger.warning(f"{NEON_YELLOW}Could not determine price precision places for {self.symbol}. Using default: {precision}. Verify market info.{RESET}")

        # Cache and return the determined precision
        self._cached_price_precision = precision
        # self.logger.debug(f"Price precision places for {self.symbol}: {precision} (Source: {source})")
        return precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        if self._cached_min_tick_size is not None:
            return self._cached_min_tick_size

        tick_size = None
        source = "Unknown"
        try:
            # 1. Try precision.price directly if it's not an integer (interpreted as tick size)
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None and not isinstance(price_precision_val, int):
                try:
                    tick = Decimal(str(price_precision_val))
                    if tick.is_finite() and tick > 0:
                        tick_size, source = tick, "market.precision.price (value)"
                    else: raise ValueError # Invalid tick
                except (TypeError, ValueError, InvalidOperation): pass

            # 2. Fallback: Try limits.price.min (often represents tick size)
            if tick_size is None:
                min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
                if min_price_val is not None:
                    try:
                        min_tick = Decimal(str(min_price_val))
                        if min_tick.is_finite() and min_tick > 0:
                            tick_size, source = min_tick, "market.limits.price.min"
                        else: raise ValueError # Invalid min price
                    except (TypeError, ValueError, InvalidOperation): pass

            # 3. Fallback: Calculate from integer precision.price (number of decimal places)
            if tick_size is None and price_precision_val is not None and isinstance(price_precision_val, int) and price_precision_val >= 0:
                 tick_size = Decimal('1e-' + str(price_precision_val))
                 source = f"Calculated from market.precision.price (int: {price_precision_val})"

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}. Using fallback.")

        # --- Final Fallback: Calculate from derived decimal places ---
        if tick_size is None:
            price_precision_places = self.get_price_precision_places() # Call the robust getter
            tick_size = Decimal('1e-' + str(price_precision_places))
            source = f"Calculated from Derived Precision ({price_precision_places})"
            self.logger.warning(f"{NEON_YELLOW}Using fallback tick size based on derived precision for {self.symbol}: {tick_size}. Verify market info.{RESET}")

        # Emergency fallback if all methods failed to produce a valid positive Decimal
        if not isinstance(tick_size, Decimal) or not tick_size.is_finite() or tick_size <= 0:
             fallback_tick = Decimal('0.00000001') # Arbitrary small positive value
             self.logger.error(f"{NEON_RED}Failed to determine a valid tick size for {self.symbol}! Using emergency fallback: {fallback_tick}. Orders may fail.{RESET}")
             tick_size = fallback_tick
             source = "Emergency Fallback"

        self._cached_min_tick_size = tick_size
        # self.logger.debug(f"Min Tick Size for {self.symbol}: {tick_size} (Source: {source})")
        return tick_size

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (number of decimal places) from market info."""
        if self._cached_amount_precision is not None:
            return self._cached_amount_precision

        precision = None
        source = "Unknown"
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount') # Can be int (places) or float/str (step size)

            if amount_precision_val is not None:
                # Case 1: Integer value represents decimal places
                if isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                    precision, source = amount_precision_val, "market.precision.amount (int)"
                # Case 2: Float/String value represents step size; infer places
                else:
                     try:
                          step_size = Decimal(str(amount_precision_val))
                          if step_size.is_finite() and step_size > 0:
                               precision = abs(step_size.normalize().as_tuple().exponent)
                               source = f"market.precision.amount (step: {step_size})"
                          else: pass
                     except (TypeError, ValueError, InvalidOperation): pass

            # Fallback 1: Infer from limits.amount.min (if it looks like a step size)
            if precision is None:
                min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
                if min_amount_val is not None:
                    try:
                        min_amount_step = Decimal(str(min_amount_val))
                        if min_amount_step.is_finite() and min_amount_step > 0:
                            # If step size is fractional or has decimals, infer places
                            if min_amount_step < 1 or '.' in str(min_amount_val):
                               precision = abs(min_amount_step.normalize().as_tuple().exponent)
                               source = f"market.limits.amount.min (step: {min_amount_step})"
                            # If step size is a whole number (e.g., 1), precision is 0
                            elif min_amount_step >= 1 and min_amount_step == min_amount_step.to_integral_value():
                               precision = 0
                               source = f"market.limits.amount.min (int step: {min_amount_step})"
                        else: pass
                    except (TypeError, ValueError, InvalidOperation): pass

        except Exception as e:
            self.logger.warning(f"Error determining amount precision places for {self.symbol}: {e}. Using default.")

        # --- Final Default Fallback ---
        if precision is None:
            default_precision = 8 # Common default for crypto base amounts (e.g., BTC, ETH)
            precision = default_precision
            source = f"Default ({default_precision})"
            self.logger.warning(f"{NEON_YELLOW}Could not determine amount precision places for {self.symbol}. Using default: {precision}. Verify market info.{RESET}")

        self._cached_amount_precision = precision
        # self.logger.debug(f"Amount precision places for {self.symbol}: {precision} (Source: {source})")
        return precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) from market info as Decimal."""
        if self._cached_min_amount_step is not None:
            return self._cached_min_amount_step

        step_size = None
        source = "Unknown"
        try:
            # 1. Try precision.amount directly if not integer (interpreted as step size)
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None and not isinstance(amount_precision_val, int):
                try:
                    step = Decimal(str(amount_precision_val))
                    if step.is_finite() and step > 0:
                        step_size, source = step, "market.precision.amount (value)"
                    else: raise ValueError
                except (TypeError, ValueError, InvalidOperation): pass

            # 2. Fallback: Try limits.amount.min (often is the step size)
            if step_size is None:
                min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
                if min_amount_val is not None:
                    try:
                        min_step = Decimal(str(min_amount_val))
                        if min_step.is_finite() and min_step > 0:
                            step_size, source = min_step, "market.limits.amount.min"
                        else: raise ValueError
                    except (TypeError, ValueError, InvalidOperation): pass

            # 3. Fallback: Calculate from integer precision.amount (decimal places)
            if step_size is None and amount_precision_val is not None and isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                 step_size = Decimal('1e-' + str(amount_precision_val))
                 source = f"Calculated from market.precision.amount (int: {amount_precision_val})"

        except Exception as e:
            self.logger.warning(f"Could not determine min amount step for {self.symbol} from market info: {e}. Using fallback.")

        # --- Final Fallback: Calculate from derived decimal places ---
        if step_size is None:
            amount_precision_places = self.get_amount_precision_places() # Use robust getter
            step_size = Decimal('1e-' + str(amount_precision_places))
            source = f"Calculated from Derived Precision ({amount_precision_places})"
            self.logger.warning(f"{NEON_YELLOW}Using fallback amount step based on derived precision for {self.symbol}: {step_size}. Verify market info.{RESET}")

        # Emergency fallback
        if not isinstance(step_size, Decimal) or not step_size.is_finite() or step_size <= 0:
             fallback_step = Decimal('0.00000001') # Arbitrary small positive value
             self.logger.error(f"{NEON_RED}Failed to determine a valid amount step size for {self.symbol}! Using emergency fallback: {fallback_step}. Orders may fail.{RESET}")
             step_size = fallback_step
             source = "Emergency Fallback"

        self._cached_min_amount_step = step_size
        # self.logger.debug(f"Min Amount Step for {self.symbol}: {step_size} (Source: {source})")
        return step_size

    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """
        Finds the N nearest calculated Fibonacci levels to the given current price.

        Args:
            current_price (Decimal): The current market price to compare against.
            num_levels (int): The number of nearest levels to return.

        Returns:
            List[Tuple[str, Decimal]]: A list of tuples, where each tuple contains the
                                       Fibonacci level name (str) and its price (Decimal),
                                       sorted by distance to the current price (nearest first).
                                       Returns an empty list if no valid levels or price provided.
        """
        if not self.fib_levels_data:
            # self.logger.debug(f"Fibonacci levels not calculated or empty for {self.symbol}. Cannot find nearest.")
            return []
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) provided for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            # Calculate distance from current price to each valid Fibonacci level
            for name, level_price in self.fib_levels_data.items():
                # Ensure the stored level price is a valid Decimal before calculating distance
                if isinstance(level_price, Decimal) and level_price.is_finite() and level_price > 0:
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    # self.logger.debug(f"Skipping invalid or non-finite Fib level during distance calculation: {name}={level_price}.")
                    pass # Skip invalid levels

            if not level_distances:
                self.logger.debug(f"No valid Fibonacci levels found to compare distance against for {self.symbol}.")
                return []

            # Sort the levels based on their distance to the current price (ascending)
            level_distances.sort(key=lambda item: item['distance'])

            # Return the name and level price for the nearest N levels requested
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- Signal Generation and Indicator Check Methods ---

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates the final trading signal ('BUY', 'SELL', or 'HOLD') based on a
        weighted score derived from enabled indicator check methods.
        - Uses Decimal for score aggregation to maintain precision.
        - Compares the final score against the configured threshold.
        - Logs the signal decision and contributing factors.

        Args:
            current_price (Decimal): The current market price (used for logging and potentially some checks).
            orderbook_data (Optional[Dict]): Processed order book data (needed for '_check_orderbook').

        Returns:
            str: The final trading signal: 'BUY', 'SELL', or 'HOLD'.
        """
        final_score = Decimal("0.0") # Initialize score as Decimal
        total_weight = Decimal("0.0") # Sum of weights for indicators that provided a valid score
        active_indicator_count = 0 # Count of indicators contributing to the score
        contributing_indicators = {} # Dictionary to store scores of contributing indicators {indicator_key: score_str}

        # --- Basic Input Validation ---
        if not self.indicator_values:
            self.logger.warning(f"Signal Generation Skipped for {self.symbol}: Indicator values dictionary is empty (calculation likely failed).")
            return "HOLD"
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Signal Generation Skipped for {self.symbol}: Invalid current price ({current_price}).")
            return "HOLD"
        if not self.weights:
            self.logger.warning(f"Signal Generation Warning for {self.symbol}: Active weight set ('{self.active_weight_set_name}') is missing or empty. Score will be zero.")
            # No weights means no score can be generated, default to HOLD
            return "HOLD"

        # --- Iterate Through Configured Indicators ---
        # Find available check methods in this class (methods starting with _check_)
        available_check_methods = {m.replace('_check_', '') for m in dir(self) if m.startswith('_check_') and callable(getattr(self, m))}

        for indicator_key, is_enabled in self.config.get("indicators", {}).items():
            # Skip if indicator is explicitly disabled in the config
            if not is_enabled: continue

            # Check if a corresponding check method exists for this enabled indicator
            if indicator_key not in available_check_methods:
                 # Warn only if it's enabled AND has a non-zero weight defined (otherwise it's harmless)
                 if float(self.weights.get(indicator_key, 0.0)) > 1e-9:
                     self.logger.warning(f"No check method '_check_{indicator_key}' found for enabled and weighted indicator '{indicator_key}' in TradingAnalyzer. Skipping.")
                 continue # Skip processing this indicator

            # Get the weight for this indicator from the active weight set
            weight_val = self.weights.get(indicator_key)
            # Skip if no weight is defined for this indicator in the active set
            if weight_val is None: continue

            # Validate and convert weight to Decimal, ensuring it's non-negative
            try:
                weight = Decimal(str(weight_val))
                if not weight.is_finite() or weight < 0:
                    raise ValueError("Weight must be non-negative and finite")
                # Skip efficiently if weight is zero (or extremely close to it)
                if weight < Decimal('1e-9'): continue
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid weight value '{weight_val}' configured for indicator '{indicator_key}' in weight set '{self.active_weight_set_name}'. Skipping this indicator.")
                continue

            # --- Execute the Indicator Check Method ---
            check_method_name = f"_check_{indicator_key}"
            score_float = np.nan # Default score is NaN (indicating no signal or error)

            try:
                method = getattr(self, check_method_name)
                # Special handling for orderbook check which requires extra data arguments
                if indicator_key == "orderbook":
                    if orderbook_data:
                        # Pass the orderbook dictionary and current price (Decimal)
                        score_float = method(orderbook_data=orderbook_data, current_price=current_price)
                    else:
                        # self.logger.debug("Orderbook check skipped: No orderbook data provided.")
                        score_float = np.nan # Cannot score without data
                else:
                    # Call the standard check method without extra arguments
                    score_float = method()

            except Exception as e:
                self.logger.error(f"Error executing indicator check '{check_method_name}' for {self.symbol}: {e}", exc_info=True)
                score_float = np.nan # Ensure score is NaN if the check method itself fails

            # --- Aggregate Score (using Decimal) ---
            # Process the score only if it's a valid, finite float number
            if pd.notna(score_float) and np.isfinite(score_float):
                try:
                    # Convert the float score [-1.0, 1.0] returned by check method to Decimal
                    score_dec = Decimal(str(score_float))
                    # Clamp score to the expected range [-1, 1] just in case a check method returned slightly out of bounds
                    clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_dec))

                    # Add the weighted score to the final aggregate score
                    final_score += clamped_score * weight
                    # Add the weight of this contributing indicator to the total weight sum
                    total_weight += weight
                    active_indicator_count += 1
                    # Store the clamped score (as string for JSON logging) for debugging
                    contributing_indicators[indicator_key] = f"{clamped_score:.3f}"

                except (ValueError, TypeError, InvalidOperation) as calc_err:
                    self.logger.error(f"Error processing score for indicator '{indicator_key}' (Raw Score: {score_float}, Weight: {weight}): {calc_err}")
            # else: Score was NaN or infinite from the check method; it does not contribute to the final score or total weight.

        # --- Determine Final Signal Based on Aggregated Score ---
        final_signal = "HOLD" # Default signal
        # Only generate BUY/SELL if there was meaningful contribution (total weight > 0)
        if total_weight > Decimal('1e-9'): # Use a small tolerance
            # Get the signal threshold from config, validate, use Decimal
            try:
                threshold_str = self.config.get("signal_score_threshold", "1.5")
                threshold = Decimal(str(threshold_str))
                # Ensure threshold is positive and finite
                if not threshold.is_finite() or threshold <= 0:
                    raise ValueError("Signal score threshold must be positive and finite")
            except (ValueError, TypeError, InvalidOperation):
                # Fallback to default threshold if config value is invalid
                default_threshold_str = str(default_config["signal_score_threshold"])
                threshold = Decimal(default_threshold_str)
                self.logger.warning(f"{NEON_YELLOW}Invalid 'signal_score_threshold' ('{threshold_str}') in config. Using default: {threshold}.{RESET}")

            # Compare the final aggregated score against the positive/negative threshold
            if final_score >= threshold:
                final_signal = "BUY"
            elif final_score <= -threshold:
                final_signal = "SELL"
            # else: Score is within the neutral zone (-threshold < score < threshold), signal remains "HOLD"
        else:
            # Log if no indicators contributed significantly
            self.logger.debug(f"No indicators provided valid scores or had non-zero weights for {self.symbol} (Total Weight: {total_weight:.4f}). Defaulting signal to HOLD.")

        # --- Log the Signal Generation Summary ---
        price_prec = self.get_price_precision_places()
        # Choose color based on the final signal
        sig_color = NEON_GREEN if final_signal == "BUY" else NEON_RED if final_signal == "SELL" else NEON_YELLOW
        log_msg = (
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Strategy='{self.active_weight_set_name}', ActiveInd={active_indicator_count}, "
            f"TotalWeight={total_weight:.2f}, FinalScore={final_score:.4f} (Threshold: +/-{threshold:.2f}) "
            f"==> {sig_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)

        # Log the scores of contributing indicators only if logger level is DEBUG
        if self.logger.isEnabledFor(logging.DEBUG) and contributing_indicators:
             # Sort scores by indicator key for consistent log output
             sorted_scores = dict(sorted(contributing_indicators.items()))
             self.logger.debug(f"  Contributing Scores ({self.symbol}): {json.dumps(sorted_scores)}")

        return final_signal

    # --- Indicator Check Methods ---
    # Each method should:
    # 1. Fetch required latest values from `self.indicator_values`.
    # 2. Validate that the fetched values are usable (not NaN, correct type expected).
    # 3. Perform the specific indicator logic.
    # 4. Return a float score between -1.0 (strong sell) and 1.0 (strong buy),
    #    or np.nan if the check cannot be performed (e.g., due to missing data).

    def _check_ema_alignment(self) -> float:
        """Checks EMA alignment (Short vs Long) and price position relative to EMAs.
           Returns float score [-1.0, 1.0] or np.nan."""
        ema_s = self.indicator_values.get("EMA_Short") # Expect Decimal or NaN
        ema_l = self.indicator_values.get("EMA_Long")  # Expect Decimal or NaN
        close = self.indicator_values.get("Close")     # Expect Decimal or NaN

        # Validate inputs: ensure all are finite Decimals
        if not all(isinstance(v, Decimal) and v.is_finite() for v in [ema_s, ema_l, close]):
            return np.nan # Cannot perform check if any value is missing or invalid

        try:
            # Determine relative positions
            price_above_short = close > ema_s
            short_above_long = ema_s > ema_l # Bullish EMA cross/alignment

            # --- Scoring Logic ---
            if short_above_long: # EMAs indicate uptrend bias
                if price_above_short: return 1.0   # Strongest Bullish: Close > Short > Long
                else: return -0.2 # Weaker Bullish: Short > Long, but Close < Short (potential pullback/weakness)
            else: # EMAs indicate downtrend bias (Long >= Short)
                if not price_above_short: return -1.0 # Strongest Bearish: Close < Short <= Long
                else: return 0.2 # Weaker Bearish: Long >= Short, but Close > Short (potential pullback/weakness)

        except TypeError: # Should not happen if validation passes, but safety check
            self.logger.warning(f"Type error during EMA alignment check for {self.symbol}.", exc_info=False)
            return np.nan

    def _check_momentum(self) -> float:
        """Scores based on the Momentum indicator value relative to zero.
           Returns float score [-1.0, 1.0] or np.nan."""
        momentum = self.indicator_values.get("Momentum") # Expect float or np.nan

        # Validate input: ensure it's a finite float/int
        if not isinstance(momentum, (float, int)) or not np.isfinite(momentum):
            return np.nan

        # Simple threshold-based scoring (could be refined with normalization)
        # These thresholds might need tuning based on asset volatility and timeframe.
        strong_pos_thresh = 0.5 # Example threshold for strong positive momentum relative to price scale (needs context)
        weak_pos_thresh = 0.1
        strong_neg_thresh = -0.5
        weak_neg_thresh = -0.1

        # Apply scaling based on thresholds
        if momentum >= strong_pos_thresh * 2: return 1.0  # Very strong positive momentum
        if momentum >= strong_pos_thresh: return 0.7    # Strong positive momentum
        if momentum >= weak_pos_thresh: return 0.3      # Weak positive momentum
        if momentum <= strong_neg_thresh * 2: return -1.0 # Very strong negative momentum
        if momentum <= strong_neg_thresh: return -0.7   # Strong negative momentum
        if momentum <= weak_neg_thresh: return -0.3     # Weak negative momentum

        # If momentum is between weak negative and weak positive thresholds (close to zero)
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Scores based on current volume relative to its moving average. High volume confirms trend strength.
           Returns float score [-1.0, 1.0] or np.nan."""
        current_volume = self.indicator_values.get("Volume") # Expect Decimal or NaN
        volume_ma = self.indicator_values.get("Volume_MA") # Expect float or np.nan

        # Validate inputs
        if not isinstance(current_volume, Decimal) or not current_volume.is_finite() or current_volume < 0:
            return np.nan # Invalid current volume
        if not isinstance(volume_ma, (float, int)) or not np.isfinite(volume_ma) or volume_ma <= 0:
            # If MA is zero or negative, comparison is meaningless or impossible
            # Could happen with very short periods or insufficient data
            return np.nan

        try:
            # Convert MA float to Decimal for accurate comparison
            volume_ma_dec = Decimal(str(volume_ma))
            # Get multiplier from config, ensure it's a valid Decimal > 0
            try:
                multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))
                if not multiplier.is_finite() or multiplier <= 0: raise ValueError
            except (ValueError, TypeError, InvalidOperation):
                 multiplier = Decimal('1.5') # Fallback to default if invalid

            # Avoid division by zero if MA is extremely small
            if volume_ma_dec < Decimal('1e-12'): return 0.0 # Treat as neutral

            # Calculate ratio of current volume to its moving average
            ratio = current_volume / volume_ma_dec

            # --- Scoring Logic ---
            # Score positive if volume is significantly above average (confirms move)
            if ratio >= multiplier * Decimal('1.5'): return 1.0 # Very high volume confirmation (strong signal)
            if ratio >= multiplier: return 0.6                 # High volume confirmation (moderate signal)

            # Score slightly negative if volume is significantly below average (lack of interest/conviction)
            if ratio <= (Decimal('1') / (multiplier * Decimal('1.5'))): return -0.4 # Unusually low volume
            if ratio <= (Decimal('1') / multiplier): return -0.2                 # Low volume

            # Volume is within the 'normal' range (between low and high thresholds)
            return 0.0

        except (InvalidOperation, ZeroDivisionError, TypeError) as e:
             self.logger.warning(f"Error during volume confirmation calculation for {self.symbol}: {e}")
             return np.nan

    def _check_stoch_rsi(self) -> float:
        """Scores based on Stochastic RSI K and D values relative to overbought/oversold thresholds and their crossover.
           Returns float score [-1.0, 1.0] or np.nan."""
        k_val = self.indicator_values.get("StochRSI_K") # Expect float or np.nan
        d_val = self.indicator_values.get("StochRSI_D") # Expect float or np.nan

        # Validate inputs: ensure K and D are finite floats/ints
        if not isinstance(k_val, (float, int)) or not np.isfinite(k_val) or \
           not isinstance(d_val, (float, int)) or not np.isfinite(d_val):
            return np.nan

        # Get thresholds from config, ensuring they are valid floats within 0-100
        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
            # Basic validation: thresholds must be within range and oversold < overbought
            if not (0 < oversold < 100 and 0 < overbought < 100 and oversold < overbought):
                raise ValueError("Invalid StochRSI thresholds")
        except (ValueError, TypeError):
             oversold, overbought = 25.0, 75.0 # Fallback to safe defaults
             self.logger.warning(f"{NEON_YELLOW}Invalid StochRSI thresholds in config, using defaults ({oversold}/{overbought}) for {self.symbol}.{RESET}")

        score = 0.0 # Initialize score to neutral

        # --- Scoring Logic ---
        # 1. Extreme Conditions (Strongest Signal): Both K and D in overbought/oversold zones
        if k_val < oversold and d_val < oversold:
            score = 1.0 # Strong Buy Signal (Deeply Oversold)
        elif k_val > overbought and d_val > overbought:
            score = -1.0 # Strong Sell Signal (Deeply Overbought)

        # 2. Crossover Signals (Moderate Signal): K crossing D adds confirmation
        # Use a small tolerance for crossover to avoid noise around exact equality
        cross_tolerance = 1.0 # Adjust if needed
        is_bullish_cross = k_val > d_val + cross_tolerance and k_val > oversold # K crosses above D, ensure not deep in oversold
        is_bearish_cross = d_val > k_val + cross_tolerance and k_val < overbought # D crosses above K (or K crosses below D), ensure not deep in overbought

        if is_bullish_cross:
             # Give bullish cross positive score, potentially overriding weak OB signal
             score = max(score, 0.7) # Max ensures we don't overwrite a stronger signal (e.g., deep OS)
        elif is_bearish_cross:
             # Give bearish cross negative score, potentially overriding weak OS signal
             score = min(score, -0.7) # Min ensures we don't overwrite a stronger signal (e.g., deep OB)

        # 3. General Position Bias (Weakest Signal): Position relative to 50 midpoint
        mid_point = 50.0
        # If currently scoring positive or neutral, reinforce slightly if both > 50
        if score >= 0 and k_val > mid_point and d_val > mid_point:
             score = max(score, 0.1) # Minor bullish bias reinforcement
        # If currently scoring negative or neutral, reinforce slightly if both < 50
        elif score <= 0 and k_val < mid_point and d_val < mid_point:
             score = min(score, -0.1) # Minor bearish bias reinforcement

        # Could add divergence checks here for more advanced signals (more complex)

        # Final clamp score to ensure it's strictly within [-1.0, 1.0]
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        """Scores based on RSI value relative to standard overbought (70) / oversold (30) levels, with extremes (80/20).
           Returns float score [-1.0, 1.0] or np.nan."""
        rsi = self.indicator_values.get("RSI") # Expect float or np.nan

        # Validate input: ensure RSI is a finite float/int (typically 0-100)
        if not isinstance(rsi, (float, int)) or not np.isfinite(rsi):
            return np.nan

        # --- Graded Scoring based on RSI Level ---
        if rsi >= 80: return -1.0 # Extreme Overbought (Strong Sell Signal)
        if rsi >= 70: return -0.7 # Standard Overbought (Moderate Sell Signal)
        if rsi > 60: return -0.3  # Approaching Overbought (Weak Sell Signal)

        if rsi <= 20: return 1.0 # Extreme Oversold (Strong Buy Signal)
        if rsi <= 30: return 0.7 # Standard Oversold (Moderate Buy Signal)
        if rsi < 40: return 0.3  # Approaching Oversold (Weak Buy Signal)

        # Neutral zone (typically 40-60)
        # if 40 <= rsi <= 60: return 0.0
        # Implicitly returns 0.0 if none of the above conditions match
        return 0.0

    def _check_cci(self) -> float:
        """Scores based on CCI value relative to standard levels (+/-100) and extremes (+/-200).
           Returns float score [-1.0, 1.0] or np.nan."""
        cci = self.indicator_values.get("CCI") # Expect float or np.nan

        # Validate input: ensure CCI is a finite float/int
        if not isinstance(cci, (float, int)) or not np.isfinite(cci):
            return np.nan

        # --- Scoring based on CCI levels ---
        # CCI > +100 suggests overbought (potential sell)
        # CCI < -100 suggests oversold (potential buy)
        if cci >= 200: return -1.0 # Extreme Overbought/Sell Signal
        if cci >= 100: return -0.7 # Standard Overbought/Sell Signal
        if cci > 0: return -0.1   # Mild bearish momentum bias (above zero line)

        if cci <= -200: return 1.0 # Extreme Oversold/Buy Signal
        if cci <= -100: return 0.7 # Standard Oversold/Buy Signal
        if cci < 0: return 0.1   # Mild bullish momentum bias (below zero line)

        # Exactly zero
        return 0.0

    def _check_wr(self) -> float:
        """Scores based on Williams %R value. Note W%R ranges from -100 (least oversold) to 0 (most overbought).
           Standard levels are -20 (Overbought threshold) and -80 (Oversold threshold).
           Returns float score [-1.0, 1.0] or np.nan."""
        wr = self.indicator_values.get("Williams_R") # Expect float (range -100 to 0) or np.nan

        # Validate input: ensure W%R is a finite float/int within expected range
        if not isinstance(wr, (float, int)) or not np.isfinite(wr) or not (-100 <= wr <= 0):
             # Log if value is outside expected range, might indicate calculation issue
             if isinstance(wr, (float, int)) and np.isfinite(wr):
                  self.logger.warning(f"Williams %R value ({wr}) is outside expected range [-100, 0] for {self.symbol}.")
             return np.nan

        # --- Scoring based on W%R levels (remembering inverse relationship) ---
        # W%R near 0 = Overbought (Sell signal)
        # W%R near -100 = Oversold (Buy signal)
        if wr >= -10: return -1.0 # Extreme Overbought (Strong Sell)
        if wr >= -20: return -0.7 # Standard Overbought (Moderate Sell)
        if wr > -50: return -0.2  # In upper half (closer to OB, slight sell bias)

        if wr <= -90: return 1.0 # Extreme Oversold (Strong Buy)
        if wr <= -80: return 0.7 # Standard Oversold (Moderate Buy)
        if wr < -50: return 0.2  # In lower half (closer to OS, slight buy bias)

        # Exactly -50 (midpoint)
        return 0.0

    def _check_psar(self) -> float:
        """Scores based on Parabolic SAR position relative to price (indicated by which PSAR value is active).
           PSAR below price = Uptrend (+1.0). PSAR above price = Downtrend (-1.0).
           Returns float score [-1.0, 1.0] or np.nan."""
        # pandas_ta PSAR calculation typically returns NaN for the non-active direction
        psar_l = self.indicator_values.get("PSAR_long")  # Value below price (uptrend) if active (Decimal or NaN)
        psar_s = self.indicator_values.get("PSAR_short") # Value above price (downtrend) if active (Decimal or NaN)

        # Check which PSAR value is finite (and implicitly non-NaN)
        # Assumes Decimal('NaN') or np.nan is used for invalid/inactive states
        l_active = isinstance(psar_l, Decimal) and psar_l.is_finite()
        s_active = isinstance(psar_s, Decimal) and psar_s.is_finite()

        if l_active and not s_active:
            return 1.0  # Uptrend Signal: PSAR Long is active (plotting below price)
        elif s_active and not l_active:
            return -1.0 # Downtrend Signal: PSAR Short is active (plotting above price)
        elif not l_active and not s_active:
            # self.logger.debug(f"PSAR check ({self.symbol}): Neither long nor short PSAR value is active/valid.")
            return np.nan # Indeterminate state or insufficient data for PSAR calculation
        else:
             # This state (both L and S are finite Decimals) shouldn't normally happen with standard PSAR.
             # It might indicate an issue with the TA calculation or data interpretation.
             self.logger.warning(f"PSAR check ({self.symbol}) encountered unusual state: Both Long ({psar_l}) and Short ({psar_s}) seem active/valid. Returning neutral (0.0).")
             return 0.0

    def _check_sma_10(self) -> float:
        """Scores based on current price position relative to the 10-period Simple Moving Average.
           Returns float score [-1.0, 1.0] or np.nan."""
        sma = self.indicator_values.get("SMA_10")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close") # Expect Decimal or NaN

        # Validate inputs: ensure both are finite Decimals
        if not isinstance(sma, Decimal) or not sma.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        try:
            # Basic check: Price above SMA suggests bullish bias, below suggests bearish.
            if close > sma: return 0.6  # Moderate Buy Signal (Price > SMA)
            if close < sma: return -0.6 # Moderate Sell Signal (Price < SMA)
            # else: Price is exactly on SMA
            return 0.0
        except TypeError: # Safety net for comparison errors
             self.logger.warning(f"Type error during SMA_10 check for {self.symbol}.", exc_info=False)
             return np.nan

    def _check_vwap(self) -> float:
        """Scores based on current price position relative to the Volume Weighted Average Price (VWAP).
           VWAP often acts as a session mean or support/resistance.
           Returns float score [-1.0, 1.0] or np.nan."""
        vwap = self.indicator_values.get("VWAP")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close") # Expect Decimal or NaN

        # Validate inputs: ensure both are finite Decimals
        if not isinstance(vwap, Decimal) or not vwap.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        # Scoring Logic: Price relative to VWAP indicates intraday trend/strength
        try:
            if close > vwap: return 0.7  # Moderate Buy Signal (Price trading above VWAP)
            if close < vwap: return -0.7 # Moderate Sell Signal (Price trading below VWAP)
            # else: Price is exactly on VWAP
            return 0.0
        except TypeError:
             self.logger.warning(f"Type error during VWAP check for {self.symbol}.", exc_info=False)
             return np.nan

    def _check_mfi(self) -> float:
        """Scores based on Money Flow Index (MFI) relative to standard overbought (80) / oversold (20) levels, with extremes (90/10).
           Returns float score [-1.0, 1.0] or np.nan."""
        mfi = self.indicator_values.get("MFI") # Expect float or np.nan

        # Validate input: ensure MFI is a finite float/int (typically 0-100)
        if not isinstance(mfi, (float, int)) or not np.isfinite(mfi):
            return np.nan

        # --- Graded Scoring based on MFI Level ---
        if mfi >= 90: return -1.0 # Extreme Overbought (Strong Sell Signal - potential exhaustion)
        if mfi >= 80: return -0.7 # Standard Overbought (Moderate Sell Signal)
        if mfi > 70: return -0.3  # Approaching Overbought (Weak Sell Signal)

        if mfi <= 10: return 1.0 # Extreme Oversold (Strong Buy Signal - potential exhaustion)
        if mfi <= 20: return 0.7 # Standard Oversold (Moderate Buy Signal)
        if mfi < 30: return 0.3  # Approaching Oversold (Weak Buy Signal)

        # Neutral zone (e.g., 30-70)
        return 0.0

    def _check_bollinger_bands(self) -> float:
        """Scores based on price position relative to Bollinger Bands (Lower, Middle, Upper).
           Touching bands suggests reversal potential. Position vs middle band suggests trend continuation/mean reversion bias.
           Returns float score [-1.0, 1.0] or np.nan."""
        bbl = self.indicator_values.get("BB_Lower")   # Expect Decimal or NaN
        bbm = self.indicator_values.get("BB_Middle")  # Expect Decimal or NaN
        bbu = self.indicator_values.get("BB_Upper")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close")    # Expect Decimal or NaN

        # Validate inputs: ensure all are finite Decimals
        if not all(isinstance(v, Decimal) and v.is_finite() for v in [bbl, bbm, bbu, close]):
            return np.nan

        # Validate band structure: Upper band must be greater than Lower band
        band_width = bbu - bbl
        if band_width <= 0:
            # self.logger.debug(f"BBands check skipped for {self.symbol}: Upper band ({bbu}) <= Lower band ({bbl}).")
            return np.nan # Invalid bands

        try:
            # --- Scoring Logic ---
            # 1. Price Touching or Exceeding Bands (Strong Reversal/Fade Signal)
            # Use a small tolerance (e.g., 0.1% of band width) for "touching"
            tolerance = band_width * Decimal('0.001')
            if close <= bbl + tolerance: return 1.0 # Strong Buy Signal (at/below lower band)
            if close >= bbu - tolerance: return -1.0 # Strong Sell Signal (at/above upper band)

            # 2. Price Between Bands: Position relative to Middle Band (Mean Reversion / Trend Bias)
            # If price is above middle band, suggests potential reversion lower (slight sell bias)
            if close > bbm:
                 # Scale score from 0 (at BBM) towards -0.5 (approaching BBU)
                 # Normalize position within the upper half of the band: (Close - Mid) / (Upper - Mid)
                 position_in_upper_band = (close - bbm) / (bbu - bbm) # Range approx 0 to 1
                 score = float(position_in_upper_band) * -0.5 # Scale to 0 to -0.5
                 return max(-0.5, score) # Limit score (shouldn't exceed -0.5 here)
            # If price is below middle band, suggests potential reversion higher (slight buy bias)
            else: # close < bbm (since close == bbm handled implicitly)
                 # Scale score from 0 (at BBM) towards +0.5 (approaching BBL)
                 # Normalize position within the lower half: (Mid - Close) / (Mid - Lower)
                 position_in_lower_band = (bbm - close) / (bbm - bbl) # Range approx 0 to 1
                 score = float(position_in_lower_band) * 0.5 # Scale to 0 to +0.5
                 return min(0.5, score) # Limit score

        except (TypeError, ZeroDivisionError, InvalidOperation) as e: # Handle comparison errors or division by zero if bands collapse
             self.logger.warning(f"Error during Bollinger Bands check for {self.symbol}: {e}")
             return np.nan

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """
        Analyzes Order Book Imbalance based on the volume within configured levels.
        Compares cumulative bid volume vs. cumulative ask volume.
        Returns float score [-1.0 (ask heavy), 1.0 (bid heavy)] or np.nan.
        """
        # Validate input data
        if not orderbook_data or not isinstance(orderbook_data.get('bids'), list) or not isinstance(orderbook_data.get('asks'), list):
            # self.logger.debug(f"Orderbook check skipped for {self.symbol}: Invalid or missing orderbook data.")
            return np.nan

        bids = orderbook_data['bids'] # List of [Decimal(price), Decimal(amount)], sorted high to low
        asks = orderbook_data['asks'] # List of [Decimal(price), Decimal(amount)], sorted low to high

        # Ensure bids and asks lists are not empty
        if not bids or not asks:
            # self.logger.debug(f"Orderbook check skipped for {self.symbol}: Bids or asks list is empty.")
            return np.nan

        try:
            # Use the number of levels specified in the config
            levels_to_analyze = int(self.config.get("orderbook_limit", 10))
            # Clamp levels to the actual available data depth
            levels_to_analyze = min(len(bids), len(asks), levels_to_analyze)
            if levels_to_analyze <= 0:
                self.logger.debug(f"Orderbook check ({self.symbol}): No levels to analyze ({levels_to_analyze}). Returning neutral.")
                return 0.0

            # --- Calculate Cumulative Volume within the specified levels ---
            # Ensure amounts are Decimal before summing
            total_bid_volume = sum(b[1] for b in bids[:levels_to_analyze] if isinstance(b[1], Decimal))
            total_ask_volume = sum(a[1] for a in asks[:levels_to_analyze] if isinstance(a[1], Decimal))

            # Calculate total volume in the analyzed range
            total_volume = total_bid_volume + total_ask_volume

            # Avoid division by zero if total volume is negligible
            if total_volume < Decimal('1e-12'):
                # self.logger.debug(f"Orderbook check ({self.symbol}): Zero total volume in analyzed {levels_to_analyze} levels.")
                return 0.0 # Return neutral if no significant volume

            # --- Calculate Order Book Imbalance (OBI) ---
            # Simple difference ratio: (Bids - Asks) / Total
            # Ranges from -1.0 (all asks) to +1.0 (all bids). 0.0 indicates perfect balance.
            obi_diff_ratio = (total_bid_volume - total_ask_volume) / total_volume

            # Convert the Decimal ratio to float for the score, clamping just in case
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi_diff_ratio)))

            # self.logger.debug(f"OB Check ({self.symbol}, {levels_to_analyze} levels): BidVol={total_bid_volume:.4f}, AskVol={total_ask_volume:.4f}, OBI_Diff={obi_diff_ratio:.4f} -> Score={score:.4f}")
            return score

        except (IndexError, ValueError, TypeError, InvalidOperation, ZeroDivisionError) as e:
             self.logger.warning(f"Orderbook analysis calculation failed for {self.symbol}: {e}", exc_info=False)
             return np.nan
        except Exception as e:
             self.logger.error(f"Unexpected error during order book analysis for {self.symbol}: {e}", exc_info=True)
             return np.nan

    # --- TP/SL Calculation Method ---

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential initial Take Profit (TP) and Stop Loss (SL) prices
        based on an estimated entry price, the trading signal ('BUY'/'SELL'), and ATR.
        - Uses Decimal precision throughout.
        - Applies market tick size for quantization.
        - Ensures TP/SL are valid prices and appropriately positioned relative to entry.

        Args:
            entry_price_estimate (Decimal): The estimated or target entry price.
            signal (str): The trading signal ('BUY' or 'SELL').

        Returns:
            Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
                - Validated Entry Price Estimate (currently just passed through, could be refined)
                - Calculated Take Profit Price (Decimal), or None if invalid/not calculable.
                - Calculated Stop Loss Price (Decimal), or None if invalid/not calculable.
        """
        final_tp: Optional[Decimal] = None
        final_sl: Optional[Decimal] = None

        # --- Validate Inputs ---
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"TP/SL Calc skipped for {self.symbol}: Invalid signal '{signal}'.")
            return entry_price_estimate, None, None

        # Fetch latest ATR value (should be Decimal or NaN)
        atr_val = self.indicator_values.get("ATR")
        if not isinstance(atr_val, Decimal) or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}TP/SL Calc Fail ({self.symbol} {signal}): Invalid or non-positive ATR value ({atr_val}). Cannot calculate TP/SL.{RESET}")
            return entry_price_estimate, None, None

        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
            self.logger.warning(f"{NEON_YELLOW}TP/SL Calc Fail ({self.symbol} {signal}): Invalid entry price estimate ({entry_price_estimate}).{RESET}")
            return entry_price_estimate, None, None

        try:
            # --- Get Parameters as Decimals ---
            try:
                tp_mult = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
                sl_mult = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))
                if tp_mult <= 0 or sl_mult <= 0:
                     raise ValueError("Multipliers must be positive")
            except (ValueError, TypeError, InvalidOperation):
                 self.logger.warning(f"Invalid TP/SL multipliers in config. Using defaults ({default_config['take_profit_multiple']}/{default_config['stop_loss_multiple']}).")
                 tp_mult = Decimal(str(default_config["take_profit_multiple"]))
                 sl_mult = Decimal(str(default_config["stop_loss_multiple"]))

            # Get market tick size for quantization
            min_tick = self.get_min_tick_size() # Should be a valid positive Decimal
            quantizer = min_tick

            # --- Calculate Raw Price Offsets based on ATR ---
            tp_offset = atr_val * tp_mult
            sl_offset = atr_val * sl_mult

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                tp_raw = entry_price_estimate + tp_offset
                sl_raw = entry_price_estimate - sl_offset
            else: # SELL signal
                tp_raw = entry_price_estimate - tp_offset
                sl_raw = entry_price_estimate + sl_offset

            # --- Quantize TP/SL Prices using Market Tick Size ---
            # Apply rounding logic:
            # - TP: Round towards neutral (less profit) to increase chance of execution.
            #       BUY TP -> Round Down; SELL TP -> Round Up.
            # - SL: Round away from entry (more loss potential) to avoid premature stops due to rounding.
            #       BUY SL -> Round Down; SELL SL -> Round Up.
            if tp_raw.is_finite():
                rounding_mode_tp = ROUND_DOWN if signal == "BUY" else ROUND_UP
                final_tp = tp_raw.quantize(quantizer, rounding=rounding_mode_tp)
            else: final_tp = None

            if sl_raw.is_finite():
                rounding_mode_sl = ROUND_DOWN if signal == "BUY" else ROUND_UP
                final_sl = sl_raw.quantize(quantizer, rounding=rounding_mode_sl)
            else: final_sl = None

            # --- Validation and Refinement ---
            # 1. Ensure SL is strictly further from entry than the tick size allows after rounding
            if final_sl is not None:
                if signal == "BUY" and final_sl >= entry_price_estimate:
                    # If BUY SL ended up >= entry, move it one tick below entry
                    corrected_sl = (entry_price_estimate - min_tick).quantize(quantizer, rounding=ROUND_DOWN)
                    self.logger.debug(f"Adjusted BUY SL ({final_sl}) to be below entry: {corrected_sl}")
                    final_sl = corrected_sl
                elif signal == "SELL" and final_sl <= entry_price_estimate:
                    # If SELL SL ended up <= entry, move it one tick above entry
                    corrected_sl = (entry_price_estimate + min_tick).quantize(quantizer, rounding=ROUND_UP)
                    self.logger.debug(f"Adjusted SELL SL ({final_sl}) to be above entry: {corrected_sl}")
                    final_sl = corrected_sl
                # Ensure SL didn't become invalid (e.g., negative) after adjustment
                if final_sl <= 0:
                     self.logger.error(f"{NEON_RED}Calculated {signal} SL became zero or negative ({final_sl}) after adjustments. Nullifying SL.{RESET}")
                     final_sl = None

            # 2. Ensure TP offers potential profit (strictly beyond entry after rounding)
            if final_tp is not None:
                 if signal == "BUY" and final_tp <= entry_price_estimate:
                     self.logger.warning(f"{NEON_YELLOW}Calculated BUY TP ({final_tp}) is not above entry price ({entry_price_estimate}) after rounding. Nullifying TP.{RESET}")
                     final_tp = None
                 elif signal == "SELL" and final_tp >= entry_price_estimate:
                     self.logger.warning(f"{NEON_YELLOW}Calculated SELL TP ({final_tp}) is not below entry price ({entry_price_estimate}) after rounding. Nullifying TP.{RESET}")
                     final_tp = None
                 # Ensure TP didn't become invalid (e.g., negative)
                 if final_tp is not None and final_tp <= 0:
                      self.logger.warning(f"{NEON_YELLOW}Calculated {signal} TP is zero or negative ({final_tp}). Nullifying TP.{RESET}")
                      final_tp = None

            # 3. Final check: If SL or TP calculation failed, ensure they are None
            if final_sl is not None and (not final_sl.is_finite() or final_sl <= 0): final_sl = None
            if final_tp is not None and (not final_tp.is_finite() or final_tp <= 0): final_tp = None


            # --- Log Results ---
            price_prec = self.get_price_precision_places()
            tp_str = f"{final_tp:.{price_prec}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_prec}f}" if final_sl else "None"
            self.logger.info(f"Calculated TP/SL for {signal} {self.symbol}: "
                             f"EntryEst={entry_price_estimate:.{price_prec}f}, ATR={atr_val:.{price_prec+2}f}, "
                             f"Tick={min_tick}, TP={tp_str}, SL={sl_str}")

            return entry_price_estimate, final_tp, final_sl

        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected error calculating TP/SL for {signal} {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price_estimate, None, None

# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the *available* balance for a specific currency using CCXT.
    - Handles Bybit V5 account types (prioritizes 'CONTRACT', could be adapted for 'UNIFIED').
    - Parses various possible balance response structures from CCXT.
    - Falls back to using 'total' balance if 'free'/'available' cannot be found (with warning).
    - Converts the balance to Decimal, ensuring it's non-negative and finite.
    - Returns the available balance as Decimal, or None if fetching/parsing fails.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        currency (str): The currency code (e.g., 'USDT', 'BTC').
        logger (logging.Logger): Logger instance.

    Returns:
        Optional[Decimal]: Available balance as Decimal, or None on failure.
    """
    lg = logger
    balance_info = None
    account_type_tried = "N/A" # Track which account type was queried

    # --- Attempt 1: Fetch with Specific Account Type (Bybit V5 Optimization) ---
    if exchange.id == 'bybit':
        # Determine preferred account type based on exchange's defaultType setting
        # Default 'linear' usually corresponds to 'CONTRACT' or 'UNIFIED' (if enabled)
        # Default 'inverse' usually corresponds to 'CONTRACT'
        # TODO: Add logic to check config for unified account preference if needed
        preferred_account_type = 'CONTRACT' # Common default for derivatives
        lg.debug(f"Attempting Bybit V5 balance fetch for {currency} (Account Type: {preferred_account_type})...")
        try:
            params = {'accountType': preferred_account_type}
            balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)
            account_type_tried = preferred_account_type
            # Optional: Log raw response for debugging structure issues
            # lg.debug(f"Raw balance response (Type: {account_type_tried}): {json.dumps(balance_info, default=str)}")
        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            # Handle specific errors indicating the account type might be wrong/unused
            # Bybit code 10001 can sometimes mean wrong account type in this context
            if "account type does not exist" in err_str or "unified account" in err_str or getattr(e, 'code', None) == 10001:
                lg.info(f"Account type '{preferred_account_type}' may not be applicable or used. Falling back to default balance fetch for {currency}.")
            else:
                # Log other exchange errors but still try fallback
                lg.warning(f"Exchange error fetching balance with type '{preferred_account_type}' for {currency}: {e}. Falling back.")
            balance_info = None # Ensure fallback is triggered
        except Exception as e: # Catch other potential errors from safe_api_call
             lg.warning(f"Failed fetching balance with type '{preferred_account_type}' for {currency}: {e}. Falling back.")
             balance_info = None

    # --- Attempt 2: Fallback to Default Fetch (if specific type failed or not Bybit) ---
    if balance_info is None:
        lg.debug(f"Fetching balance for {currency} using default parameters...")
        try:
            balance_info = safe_api_call(exchange.fetch_balance, lg)
            account_type_tried = "Default" # Indicate default fetch was used
            # lg.debug(f"Raw balance response (Type: {account_type_tried}): {json.dumps(balance_info, default=str)}")
        except Exception as e:
            lg.error(f"{NEON_RED}Failed to fetch balance info for {currency} even with default parameters: {e}{RESET}")
            return None # Both attempts failed

    # --- Parse the Balance Information ---
    if not balance_info:
         lg.error(f"Balance fetch (Type: {account_type_tried}) returned empty or None response for {currency}.")
         return None

    free_balance_str = None
    parse_source = "Unknown"

    # --- Parsing Logic (tries various common structures) ---
    # Structure 1: Standard CCXT `balance[currency]['free']` or `balance[currency]['available']`
    if currency in balance_info and isinstance(balance_info.get(currency), dict):
        currency_data = balance_info[currency]
        if currency_data.get('free') is not None:
            free_balance_str = str(currency_data['free'])
            parse_source = f"Standard ['{currency}']['free']"
        elif currency_data.get('available') is not None: # Some exchanges use 'available'
            free_balance_str = str(currency_data['available'])
            parse_source = f"Standard ['{currency}']['available']"

    # Structure 2: Top-level `balance['free'][currency]` or `balance['available'][currency]`
    elif free_balance_str is None and isinstance(balance_info.get('free'), dict) and balance_info['free'].get(currency) is not None:
         free_balance_str = str(balance_info['free'][currency])
         parse_source = f"Top-level ['free']['{currency}']"
    elif free_balance_str is None and isinstance(balance_info.get('available'), dict) and balance_info['available'].get(currency) is not None:
         free_balance_str = str(balance_info['available'][currency])
         parse_source = f"Top-level ['available']['{currency}']"

    # Structure 3: Bybit V5 Specific Parsing (from `info` field) - Often more reliable for V5 details
    elif free_balance_str is None and exchange.id == 'bybit' and isinstance(balance_info.get('info'), dict):
         info_data = balance_info['info']
         # V5 responses often nest results under 'result'
         result_data = info_data.get('result', info_data) # Use result if present, else info itself

         # Check for V5 list structure (common for balance endpoints)
         if isinstance(result_data.get('list'), list) and result_data['list']:
             account_list = result_data['list']
             found_in_list = False
             for account_details in account_list:
                 if not isinstance(account_details, dict): continue # Skip invalid entries

                 # --- Unified Account Parsing ---
                 # Unified accounts contain a 'coin' list
                 if isinstance(account_details.get('coin'), list):
                     for coin_data in account_details['coin']:
                         if isinstance(coin_data, dict) and coin_data.get('coin') == currency:
                              # Prefer availableToWithdraw > availableBalance > walletBalance for usable funds
                              free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                              if free is not None:
                                  free_balance_str = str(free)
                                  parse_source = f"Bybit V5 info.result.list[coin='{currency}'] (Type: Unified)"
                                  found_in_list = True; break # Found currency in unified list
                     if found_in_list: break # Exit outer loop if found

                 # --- Contract Account Parsing (Derivatives) ---
                 # Contract accounts have balance directly under the account entry
                 elif account_details.get('accountType') == 'CONTRACT':
                     # Check if the account entry itself has the target currency (e.g., USDT balance for linear)
                     # Prefer availableBalance > walletBalance
                     if account_details.get('coin') == currency: # Ensure currency matches if specified directly
                          free = account_details.get('availableBalance') or account_details.get('walletBalance')
                          if free is not None:
                                free_balance_str = str(free)
                                parse_source = f"Bybit V5 info.result.list[CONTRACT] (Coin: {currency})"
                                found_in_list = True; break
                     # Sometimes CONTRACT balance is listed without explicit coin key if it's the main margin currency
                     elif currency == account_details.get('marginCoin', currency): # Check against marginCoin or assume target currency
                          free = account_details.get('availableBalance') or account_details.get('walletBalance')
                          if free is not None:
                               free_balance_str = str(free)
                               parse_source = f"Bybit V5 info.result.list[CONTRACT] (Margin Coin: {currency})"
                               found_in_list = True; break

             # Fallback: If not found via specific type checks, look in first entry generically (less reliable)
             if not found_in_list and account_list and isinstance(account_list[0], dict):
                  first_acc = account_list[0]
                  free = first_acc.get('availableBalance') or first_acc.get('availableToWithdraw') or first_acc.get('walletBalance')
                  if free is not None:
                       free_balance_str = str(free)
                       parse_source = f"Bybit V5 info.result.list[0] (Fallback Guess)"

         # Alternative V5 Structure: Check if currency is a direct key under 'info' or 'info.result'
         elif free_balance_str is None and isinstance(result_data.get(currency), dict):
              currency_data = result_data[currency]
              # Look for common available balance keys
              free = currency_data.get('available') or currency_data.get('free') or currency_data.get('availableBalance') or currency_data.get('walletBalance')
              if free is not None:
                  free_balance_str = str(free)
                  parse_source = f"Bybit V5 info[.result]['{currency}']"


    # --- Fallback: Use 'total' balance if 'free'/'available' couldn't be found ---
    if free_balance_str is None:
         total_balance_str = None
         parse_source_total = "Unknown Total"
         # Check standard structures for 'total'
         if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('total') is not None:
             total_balance_str = str(balance_info[currency]['total'])
             parse_source_total = f"Standard ['{currency}']['total']"
         elif isinstance(balance_info.get('total'), dict) and balance_info['total'].get(currency) is not None:
             total_balance_str = str(balance_info['total'][currency])
             parse_source_total = f"Top-level ['total']['{currency}']"
         # TODO: Add V5 'total' parsing from info if needed

         if total_balance_str is not None:
              lg.warning(f"{NEON_YELLOW}Could not find 'free' or 'available' balance for {currency}. Using 'total' balance ({total_balance_str}) as fallback ({parse_source_total})."
                         f" Note: 'total' may include collateral or unrealized PNL and might not be fully available for new trades.{RESET}")
              free_balance_str = total_balance_str
              parse_source = parse_source_total + " (Fallback)"
         else:
              # If even 'total' couldn't be found, log error and return None
              lg.error(f"{NEON_RED}Could not determine any balance ('free', 'available', or 'total') for {currency} after checking known structures (Account Type Searched: {account_type_tried}).{RESET}")
              lg.debug(f"Full balance_info structure: {json.dumps(balance_info, default=str)}")
              return None

    # --- Convert the Found Balance String to Decimal ---
    try:
        final_balance = Decimal(free_balance_str)
        # Validate the converted Decimal value
        if not final_balance.is_finite():
             lg.warning(f"Parsed balance for {currency} ('{free_balance_str}' from {parse_source}) resulted in a non-finite Decimal. Treating as zero.")
             final_balance = Decimal('0')
        # Treat negative available balance as zero for trading decisions
        if final_balance < 0:
             lg.warning(f"Parsed available balance for {currency} ('{free_balance_str}' from {parse_source}) is negative. Treating as zero available.")
             final_balance = Decimal('0')

        lg.info(f"Available {currency} balance determined (Source: {parse_source}, AccType Searched: {account_type_tried}): {final_balance:.4f}") # Adjust precision as needed
        return final_balance
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"{NEON_RED}Failed to convert final balance string '{free_balance_str}' (from {parse_source}) to Decimal for {currency}: {e}{RESET}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Retrieves the market information dictionary for a specific symbol from CCXT.
    - Ensures exchange markets are loaded first.
    - Handles cases where the symbol is not found.
    - Adds convenient boolean flags (is_contract, is_linear, is_inverse, is_spot)
      to the returned dictionary for easier logic elsewhere.
    - Infers linear/inverse status if not explicitly provided, based on defaultType and quote currency.
    - Logs key market details and checks if the market is active.

    Args:
        exchange (ccxt.Exchange): Initialized CCXT exchange object.
        symbol (str): The trading symbol (e.g., 'BTC/USDT:USDT').
        logger (logging.Logger): Logger instance.

    Returns:
        Optional[Dict]: The CCXT market dictionary augmented with convenience flags,
                        or None if market info cannot be retrieved or is invalid.
    """
    lg = logger
    try:
        # --- Ensure Markets Are Loaded ---
        # Crucial step, as market info is needed for almost all operations
        if not exchange.markets or not exchange.markets_by_id: # Check both for robustness
             lg.info(f"Markets not loaded for {exchange.id}. Attempting explicit load...")
             try:
                 # Use safe_api_call to load markets robustly
                 safe_api_call(exchange.load_markets, lg, reload=True) # Force reload
                 lg.info(f"Markets reloaded successfully ({len(exchange.symbols)} symbols found).")
                 # Double-check if markets are populated now
                 if not exchange.markets:
                      lg.error(f"{NEON_RED}Market loading appeared successful but exchange.markets is still empty! Cannot proceed.{RESET}")
                      return None
             except Exception as load_err:
                  lg.error(f"{NEON_RED}Failed to load markets after retries: {load_err}. Cannot get market info for {symbol}.{RESET}")
                  return None # Market loading failed, cannot proceed

        # --- Retrieve Market Dictionary ---
        # Use exchange.market() which handles lookup by symbol, id, etc.
        market = exchange.market(symbol)
        if not market or not isinstance(market, dict):
             lg.error(f"{NEON_RED}Market '{symbol}' not found in CCXT's loaded markets for {exchange.id}.{RESET}")
             # Provide hint for common Bybit V5 linear format if applicable
             if '/' in symbol and ':' not in symbol and exchange.id == 'bybit':
                  base, quote = symbol.split('/')[:2]
                  suggested_symbol = f"{base}/{quote}:{quote}"
                  lg.warning(f"{NEON_YELLOW}Hint: For Bybit V5 linear perpetuals, check format like '{suggested_symbol}'.{RESET}")
             return None

        # --- Add Convenience Flags for Easier Logic ---
        # Use .get() with defaults for safety in case keys are missing
        market_type = market.get('type', '').lower() # e.g., 'spot', 'swap', 'future'
        is_spot = (market_type == 'spot')
        is_swap = (market_type == 'swap') # Typically perpetual swaps
        is_future = (market_type == 'future') # Typically dated futures
        # General contract flag (covers swap, future, or explicit 'contract' flag)
        is_contract = is_swap or is_future or market.get('contract', False)

        # Determine Linear vs. Inverse (Crucial for contract sizing and PNL)
        is_linear = market.get('linear', False)
        is_inverse = market.get('inverse', False)

        # --- Infer Linear/Inverse if not explicitly set (common for V5) ---
        if is_contract and not is_linear and not is_inverse:
            lg.debug(f"Market {symbol} is contract but linear/inverse flag not explicit. Inferring...")
            # 1. Check defaultType set during exchange initialization
            default_type = exchange.options.get('defaultType', '').lower()
            if default_type == 'linear':
                 is_linear = True
                 lg.debug(f" > Inferred Linear based on exchange defaultType.")
            elif default_type == 'inverse':
                 is_inverse = True
                 lg.debug(f" > Inferred Inverse based on exchange defaultType.")
            else:
                 # 2. Fallback: Infer based on quote currency (less reliable but common pattern)
                 quote_id = market.get('quoteId', '').upper()
                 if quote_id in ['USD']: # Typically USD-margined are inverse
                     is_inverse = True
                     lg.debug(f" > Inferred Inverse based on quote currency '{quote_id}'.")
                 elif quote_id in ['USDT', 'USDC', 'BUSD', 'DAI']: # Stablecoin-margined are usually linear
                     is_linear = True
                     lg.debug(f" > Inferred Linear based on quote currency '{quote_id}'.")
                 else:
                     # If quote is crypto (e.g., BTC), it's likely inverse
                     # If quote is fiat other than USD, could be either, default guess? Risky.
                     lg.warning(f" > Could not reliably infer linear/inverse for {symbol} (Quote: {quote_id}). Assuming Linear (default guess). Verify!")
                     is_linear = True # Default assumption if unclear

        # Add the flags to the market dictionary (modifies the copy)
        market['is_spot'] = is_spot
        market['is_contract'] = is_contract
        market['is_linear'] = is_linear
        market['is_inverse'] = is_inverse

        # --- Log Key Market Details ---
        lg.debug(f"Market Info Retrieved ({symbol}): ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                 f"Type={market_type}, IsContract={is_contract}, IsLinear={is_linear}, IsInverse={is_inverse}, "
                 f"IsActive={market.get('active', True)}, ContractSize={market.get('contractSize', 'N/A')}")
        # Log precision and limits at debug level - crucial for order placement issues
        lg.debug(f"  Precision Info: {market.get('precision')}")
        lg.debug(f"  Limit Info: {market.get('limits')}")

        # --- Check if Market is Active ---
        if not market.get('active', True):
             # Market is marked as inactive by the exchange (delisted, suspended, etc.)
             lg.warning(f"{NEON_YELLOW}Market {symbol} is marked as inactive by the exchange. Trading may not be possible.{RESET}")
             # Depending on strategy requirements, you might want to return None here to prevent trading attempts.
             # return None

        return market

    except ccxt.BadSymbol as e:
        # Specific error if the symbol format is invalid or not supported
        lg.error(f"{NEON_RED}Invalid symbol format or symbol not supported by {exchange.id}: '{symbol}'. Error: {e}{RESET}")
        return None
    except Exception as e:
        # Catch any other unexpected errors during market info retrieval
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # Configuration: fraction (0 to 1)
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting final amount
    logger: Optional[logging.Logger] = None,
    config_override: Optional[Dict] = None # Optional: Allow passing config if needed globally
) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on account balance, risk percentage,
    stop loss distance, and market constraints (precision, limits, contract type).

    - Uses Decimal for all financial calculations to maintain precision.
    - Correctly applies amount step size and min/max amount/cost limits.
    - Currently supports **Linear Contracts** and **Spot** markets.
    - **Explicitly blocks Inverse Contracts** due to different sizing logic.
    - Returns the calculated position size (in base currency for Linear/Spot) as Decimal,
      or None if calculation fails or constraints cannot be met.

    Args:
        balance (Decimal): Available balance in the quote currency (e.g., USDT).
        risk_per_trade (float): Fraction of balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price (Decimal): The calculated initial stop loss price.
        entry_price (Decimal): The estimated or actual entry price.
        market_info (Dict): Market details dictionary from get_market_info().
        exchange (ccxt.Exchange): Initialized CCXT exchange object (for formatting).
        logger (Optional[logging.Logger]): Logger instance. Uses default if None.
        config_
