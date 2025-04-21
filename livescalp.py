```python
# sxs.py
# Enhanced and Upgraded Scalping Bot Framework
# Derived from xrscalper.py, focusing on robust execution, error handling,
# advanced position management (BE, TSL), and Bybit V5 compatibility.

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

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta  # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
init(autoreset=True) # Ensure colorama resets styles automatically
load_dotenv() # Load environment variables from .env file

# Set Decimal precision (high precision for financial calculations)
# Consider adjusting based on performance vs. required precision needs.
getcontext().prec = 36

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Log error and raise immediately if keys are missing
    # Use a basic handler for this critical startup error before full logging is set up
    logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
    logging.critical(f"{NEON_RED}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file.{RESET}")
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET environment variables are not set.")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Ensure the log directory exists early
os.makedirs(LOG_DIRECTORY, exist_ok=True)

TIMEZONE = ZoneInfo("America/Chicago")  # Use IANA timezone database names (configurable later)
MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 7  # Increased default delay
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
# HTTP status codes considered retryable for generic network issues
RETRYABLE_HTTP_CODES = [429, 500, 502, 503, 504]

# Default indicator periods (can be overridden by config.json) - Standardized Names
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_PERIOD = 20
DEFAULT_WILLIAMS_R_PERIOD = 14
DEFAULT_MFI_PERIOD = 14
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_RSI_PERIOD = 14
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
DEFAULT_FIB_PERIOD = 50
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] # Standard Fibonacci levels
LOOP_DELAY_SECONDS = 10 # Default time between the end of one cycle and the start of the next
POSITION_CONFIRM_DELAY_SECONDS = 10 # Default wait time after placing order before confirming position
# QUOTE_CURRENCY is loaded dynamically from config

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    _patterns = {} # Cache patterns for slight performance gain

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Redact API Key if present
        if API_KEY and API_KEY not in self._patterns:
            self._patterns[API_KEY] = "***API_KEY***"
        if API_KEY and API_KEY in msg:
            msg = msg.replace(API_KEY, self._patterns[API_KEY])
        # Redact API Secret if present
        if API_SECRET and API_SECRET not in self._patterns:
            self._patterns[API_SECRET] = "***API_SECRET***"
        if API_SECRET and API_SECRET in msg:
            msg = msg.replace(API_SECRET, self._patterns[API_SECRET])
        return msg

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file, creating default if not found,
    ensuring all default keys are present with validation, and saving updates.
    Returns the validated configuration dictionary.
    """
    # Define the default configuration structure and values
    default_config = {
        # Trading pair and timeframe
        "symbol": "BTC/USDT:USDT", # Bybit linear perpetual example
        "interval": "5", # Default timeframe (e.g., "5" for 5 minutes)

        # API and Bot Behavior
        "retry_delay": RETRY_DELAY_SECONDS, # Delay between API retries (can be overridden)
        "max_api_retries": MAX_API_RETRIES, # Max retries for API calls
        "enable_trading": False, # Safety Feature: Must be explicitly set to true to trade
        "use_sandbox": True, # Safety Feature: Use testnet by default
        "max_concurrent_positions": 1, # Max open positions for this symbol instance (Note: current logic handles 1)
        "quote_currency": "USDT", # Quote currency for balance checks and sizing (ensure matches symbol)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Delay after order before confirming position
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between main loop cycles
        "timezone": "America/Chicago", # IANA timezone name for local time display in logs

        # Risk Management
        "risk_per_trade": 0.01, # Fraction of balance to risk (e.g., 0.01 = 1%)
        "leverage": 20, # Desired leverage (Ensure supported by exchange/market)
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing/initial fixed SL)
        "take_profit_multiple": 0.7, # ATR multiple for initial TP

        # Order Execution
        "entry_order_type": "market", # "market" or "limit"
        "limit_order_offset_buy": 0.0005, # % offset from price for BUY limit (0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005, # % offset from price for SELL limit

        # Advanced Position Management
        "enable_trailing_stop": True, # Use exchange-native Trailing Stop Loss (Bybit V5)
        # IMPORTANT: Bybit V5 TSL uses PRICE DISTANCE. 'callback_rate' is used to *calculate* this distance.
        "trailing_stop_callback_rate": 0.005, # Used to calculate trail distance % from *activation price* (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003, # % profit move from entry to calculate TSL activation price
        "enable_break_even": True, # Enable moving SL to break-even point
        "break_even_trigger_atr_multiple": 1.0, # Move SL when profit >= X * ATR
        "break_even_offset_ticks": 2, # Place BE SL X ticks beyond entry price
        "time_based_exit_minutes": None, # Optional: Exit after X minutes (e.g., 120)

        # Indicator Periods & Parameters (Using renamed defaults)
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
        "fibonacci_period": DEFAULT_FIB_PERIOD,

        # Indicator Calculation & Scoring Control
        "orderbook_limit": 25, # Depth of order book levels to fetch/analyze
        "signal_score_threshold": 1.5, # Weighted score needed to trigger BUY/SELL signal
        "stoch_rsi_oversold_threshold": 25, # Threshold for StochRSI oversold score
        "stoch_rsi_overbought_threshold": 75, # Threshold for StochRSI overbought score
        "volume_confirmation_multiplier": 1.5, # Volume > Multiplier * VolMA for confirmation
        "indicators": { # Toggle calculation and scoring contribution
            # Key names MUST match _check_<key> methods and weight_sets keys
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
        },
        "weight_sets": { # Define scoring weights for different strategies
            "scalping": { # Example: Faster, momentum-focused
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Example: Balanced
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
            # Ensure keys here match keys in "indicators" and _check_ methods
        },
        "active_weight_set": "default" # Select the active weight set
    }

    config = default_config.copy() # Start with defaults
    config_loaded_successfully = False
    needs_saving = False # Flag to track if the file needs updating

    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            # Merge loaded config over defaults, ensuring all default keys exist
            config = _merge_configs(loaded_config, default_config)
            print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")
            config_loaded_successfully = True
            # Check if merge introduced changes needing save (e.g., added new default keys)
            if config != loaded_config: # Simple check, might not catch subtle type changes fixed later
                needs_saving = True
                print(f"{NEON_YELLOW}Configuration merged with new defaults/structure.{RESET}")

        except (json.JSONDecodeError, IOError) as e:
            print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
            config = default_config # Use in-memory default
            needs_saving = True # Need to save the default config
        except Exception as e:
             print(f"{NEON_RED}Unexpected error loading config {filepath}: {e}. Using default config.{RESET}")
             config = default_config
             needs_saving = True
    else:
        # Config file doesn't exist, create it with defaults
        print(f"{NEON_YELLOW}Config file not found. Creating default config at {filepath}{RESET}")
        config = default_config
        needs_saving = True

    # --- Validation Section ---
    current_config = config # Work with the potentially merged/default config
    original_config_before_validation = config.copy() # For checking if validation changed anything

    # Helper for validation logging and default setting
    def validate_param(key, default_value, validation_func, error_msg_format):
        is_valid = False
        current_value = current_config.get(key)
        try:
            if key in current_config and validation_func(current_value):
                is_valid = True
            else:
                # If key missing or validation fails
                current_config[key] = default_value
                # Format error message safely, handling cases where value might be complex type
                value_repr = repr(current_value) if current_value is not None else 'None'
                print(f"{NEON_RED}{error_msg_format.format(key=key, value=value_repr, default=default_value)}{RESET}")
        except Exception as validation_err:
            # Catch errors within the validation function itself
            print(f"{NEON_RED}Error validating '{key}' ({repr(current_value)}): {validation_err}. Resetting to default '{default_value}'.{RESET}")
            current_config[key] = default_value
        return is_valid # Return whether the original value was valid

    # Validate symbol (must be non-empty string)
    validate_param("symbol", default_config["symbol"],
                   lambda v: isinstance(v, str) and v.strip(),
                   "CRITICAL: Config key '{key}' is missing, empty, or invalid ({value}). Resetting to default: '{default}'.")

    # Validate interval
    validate_param("interval", default_config["interval"],
                   lambda v: v in VALID_INTERVALS,
                   "Invalid interval '{value}' in config for '{key}'. Resetting to default '{default}'. Valid: " + str(VALID_INTERVALS) + ".")

    # Validate entry order type
    validate_param("entry_order_type", default_config["entry_order_type"],
                   lambda v: v in ["market", "limit"],
                   "Invalid entry_order_type '{value}' for '{key}'. Must be 'market' or 'limit'. Resetting to '{default}'.")

    # Validate timezone
    try:
        tz_str = current_config.get("timezone", default_config["timezone"])
        tz_info = ZoneInfo(tz_str)
        current_config["timezone"] = tz_str # Store the valid string
        global TIMEZONE # Update the global constant
        TIMEZONE = tz_info
    except Exception as tz_err:
        print(f"{NEON_RED}Invalid timezone '{tz_str}' in config: {tz_err}. Resetting to default '{default_config['timezone']}'.{RESET}")
        current_config["timezone"] = default_config["timezone"]
        TIMEZONE = ZoneInfo(default_config["timezone"])

    # Validate active weight set exists
    validate_param("active_weight_set", default_config["active_weight_set"],
                   lambda v: isinstance(v, str) and v in current_config.get("weight_sets", {}),
                   "Active weight set '{value}' for '{key}' not found in 'weight_sets'. Resetting to '{default}'.")

    # Validate numeric parameters (ranges and types) using Decimal for checks
    numeric_params = {
        # key: (min_val, max_val, allow_min_equal, allow_max_equal, is_integer, default_val)
        "risk_per_trade": (0, 1, False, False, False, default_config["risk_per_trade"]),
        "leverage": (1, 1000, True, True, True, default_config["leverage"]), # Realistic max leverage
        "stop_loss_multiple": (0, float('inf'), False, True, False, default_config["stop_loss_multiple"]),
        "take_profit_multiple": (0, float('inf'), False, True, False, default_config["take_profit_multiple"]),
        "trailing_stop_callback_rate": (0, 1, False, False, False, default_config["trailing_stop_callback_rate"]),
        "trailing_stop_activation_percentage": (0, 1, True, False, False, default_config["trailing_stop_activation_percentage"]), # Allow 0%
        "break_even_trigger_atr_multiple": (0, float('inf'), False, True, False, default_config["break_even_trigger_atr_multiple"]),
        "break_even_offset_ticks": (0, 1000, True, True, True, default_config["break_even_offset_ticks"]), # Increased max ticks
        "signal_score_threshold": (0, float('inf'), False, True, False, default_config["signal_score_threshold"]),
        "atr_period": (2, 1000, True, True, True, default_config["atr_period"]), # Min 2 for ATR
        "ema_short_period": (1, 1000, True, True, True, default_config["ema_short_period"]),
        "ema_long_period": (1, 1000, True, True, True, default_config["ema_long_period"]),
        "rsi_period": (2, 1000, True, True, True, default_config["rsi_period"]), # Min 2 for RSI
        "bollinger_bands_period": (2, 1000, True, True, True, default_config["bollinger_bands_period"]),
        "bollinger_bands_std_dev": (0, 10, False, True, False, default_config["bollinger_bands_std_dev"]),
        "cci_period": (2, 1000, True, True, True, default_config["cci_period"]), # Check min req for CCI
        "williams_r_period": (2, 1000, True, True, True, default_config["williams_r_period"]),
        "mfi_period": (2, 1000, True, True, True, default_config["mfi_period"]),
        "stoch_rsi_period": (2, 1000, True, True, True, default_config["stoch_rsi_period"]),
        "stoch_rsi_rsi_period": (2, 1000, True, True, True, default_config["stoch_rsi_rsi_period"]),
        "stoch_rsi_k_period": (1, 1000, True, True, True, default_config["stoch_rsi_k_period"]),
        "stoch_rsi_d_period": (1, 1000, True, True, True, default_config["stoch_rsi_d_period"]),
        "psar_step": (0, 1, False, True, False, default_config["psar_step"]), # Allow 1? Check ta lib
        "psar_max_step": (0, 1, False, True, False, default_config["psar_max_step"]),
        "sma_10_period": (1, 1000, True, True, True, default_config["sma_10_period"]),
        "momentum_period": (1, 1000, True, True, True, default_config["momentum_period"]),
        "volume_ma_period": (1, 1000, True, True, True, default_config["volume_ma_period"]),
        "fibonacci_period": (2, 1000, True, True, True, default_config["fibonacci_period"]), # Need at least 2 points
        "orderbook_limit": (1, 200, True, True, True, default_config["orderbook_limit"]), # Bybit V5 supports up to 200 for linear
        "position_confirm_delay_seconds": (0, 120, True, True, False, default_config["position_confirm_delay_seconds"]),
        "loop_delay_seconds": (1, 300, True, True, False, default_config["loop_delay_seconds"]),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False, default_config["stoch_rsi_oversold_threshold"]),
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False, default_config["stoch_rsi_overbought_threshold"]),
        "volume_confirmation_multiplier": (0, float('inf'), False, True, False, default_config["volume_confirmation_multiplier"]),
        "limit_order_offset_buy": (0, 0.1, True, False, False, default_config["limit_order_offset_buy"]), # 10% offset max? Reasonable.
        "limit_order_offset_sell": (0, 0.1, True, False, False, default_config["limit_order_offset_sell"]),
        "retry_delay": (1, 120, True, True, False, default_config["retry_delay"]),
        "max_api_retries": (0, 10, True, True, True, default_config["max_api_retries"]),
        "max_concurrent_positions": (1, 10, True, True, True, default_config["max_concurrent_positions"]), # Limit realistically
    }
    for key, (min_val, max_val, allow_min, allow_max, is_integer, default_val) in numeric_params.items():
        value = current_config.get(key)
        is_valid = False
        if value is not None:
            try:
                val_dec = Decimal(str(value)) # Convert to Decimal first for reliable checks
                if not val_dec.is_finite(): raise ValueError("Value not finite")

                # Check bounds using Decimal comparison
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))
                lower_bound_ok = (val_dec >= min_dec) if allow_min else (val_dec > min_dec)
                upper_bound_ok = (val_dec <= max_dec) if allow_max else (val_dec < max_dec)

                if lower_bound_ok and upper_bound_ok:
                    # Convert to final type (int or float) after validation
                    if is_integer:
                        # Check if it's actually an integer before converting
                        if val_dec % 1 == 0:
                            final_value = int(val_dec)
                            current_config[key] = final_value # Store validated integer
                            is_valid = True
                        else:
                            raise ValueError("Non-integer value provided for integer parameter")
                    else:
                        final_value = float(val_dec) # Store as float if not integer
                        current_config[key] = final_value
                        is_valid = True
                # else: Bounds check failed, is_valid remains False
            except (ValueError, TypeError, InvalidOperation):
                 pass # Invalid format or failed checks, is_valid remains False

        if not is_valid:
            # Use validate_param to log error and reset to default
            err_msg = (f"Invalid value for '{{key}}' ({{value}}). Must be {'integer' if is_integer else 'number'} "
                       f"between {min_val} ({'inclusive' if allow_min else 'exclusive'}) and "
                       f"{max_val} ({'inclusive' if allow_max else 'exclusive'}). Resetting to default '{{default}}'.")
            validate_param(key, default_val, lambda v: False, err_msg) # Force reset

    # Specific validation for time_based_exit_minutes (allow None or positive number)
    time_exit_key = "time_based_exit_minutes"
    time_exit_value = current_config.get(time_exit_key)
    time_exit_valid = False
    if time_exit_value is None:
        time_exit_valid = True
    else:
        try:
            time_exit_float = float(time_exit_value)
            if time_exit_float > 0:
                 current_config[time_exit_key] = time_exit_float # Store validated float
                 time_exit_valid = True
            else: raise ValueError("Must be positive if set")
        except (ValueError, TypeError):
            pass # Invalid format or non-positive

    if not time_exit_valid:
         validate_param(time_exit_key, default_config[time_exit_key], lambda v: False, # Force reset
                        "Invalid value for '{{key}}' ({{value}}). Must be 'None' or a positive number. Resetting to default ('{{default}}').")

    # Validate boolean parameters
    bool_params = ["enable_trading", "use_sandbox", "enable_trailing_stop", "enable_break_even"]
    for key in bool_params:
         validate_param(key, default_config[key], lambda v: isinstance(v, bool),
                        "Invalid value for '{{key}}' ({{value}}). Must be boolean (true/false). Resetting to default '{{default}}'.")

    # Validate indicator enable flags (must be boolean)
    indicators_key = 'indicators'
    if indicators_key in current_config and isinstance(current_config[indicators_key], dict):
        indicators_dict = current_config[indicators_key]
        default_indicators = default_config[indicators_key]
        for ind_key, ind_val in indicators_dict.items():
            # Check if key exists in default (ensures consistency) and if value is boolean
            if ind_key not in default_indicators:
                print(f"{NEON_YELLOW}Warning: Unknown key '{ind_key}' found in '{indicators_key}'. Ignoring.")
                # Optionally remove unknown keys? del current_config[indicators_key][ind_key]
                continue
            if not isinstance(ind_val, bool):
                default_ind_val = default_indicators.get(ind_key, False) # Default to False if somehow missing
                print(f"{NEON_RED}Invalid value for '{indicators_key}.{ind_key}' ({repr(ind_val)}). Must be boolean. Resetting to default '{default_ind_val}'.{RESET}")
                indicators_dict[ind_key] = default_ind_val
    else:
        # If 'indicators' key is missing or not a dict, reset to default
        print(f"{NEON_RED}Invalid or missing '{indicators_key}' section. Resetting to default.{RESET}")
        current_config[indicators_key] = default_config[indicators_key]

    # Validate weight sets structure and values
    ws_key = "weight_sets"
    if ws_key in current_config and isinstance(current_config[ws_key], dict):
        weight_sets = current_config[ws_key]
        default_indicators_keys = default_config['indicators'].keys()
        for set_name, weights in weight_sets.items():
            if not isinstance(weights, dict):
                 print(f"{NEON_RED}Invalid structure for weight set '{set_name}' (must be a dictionary). Skipping validation for this set.{RESET}")
                 continue # Or reset this specific weight set to default?
            for ind_key, weight_val in weights.items():
                # Ensure weight key matches an enabled indicator key
                if ind_key not in default_indicators_keys:
                    print(f"{NEON_YELLOW}Warning: Weight defined for unknown/disabled indicator '{ind_key}' in weight set '{set_name}'. Ignoring.{RESET}")
                    continue
                # Validate weight value is numeric (convertible to Decimal, non-negative)
                try:
                    weight_dec = Decimal(str(weight_val))
                    if not weight_dec.is_finite() or weight_dec < 0:
                        raise ValueError("Weight must be non-negative and finite")
                    # Store validated weight as float (common usage)
                    weights[ind_key] = float(weight_dec)
                except (ValueError, TypeError, InvalidOperation):
                     default_weight = default_config[ws_key].get(set_name, {}).get(ind_key, 0.0) # Get default weight or 0
                     print(f"{NEON_RED}Invalid weight value '{repr(weight_val)}' for indicator '{ind_key}' in weight set '{set_name}'. Must be a non-negative number. Resetting to default '{default_weight}'.{RESET}")
                     weights[ind_key] = float(default_weight) # Reset to float
    else:
         print(f"{NEON_RED}Invalid or missing '{ws_key}' section. Resetting to default.{RESET}")
         current_config[ws_key] = default_config[ws_key]


    # If config was updated during merge/validation or file creation, save it back
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
    Recursively merges the loaded configuration with default values.
    Ensures all keys from the default config exist in the final config.
    Prioritizes values from the loaded config if key exists in both. Handles nested dictionaries.
    Adds keys present only in loaded_config (allows user additions).
    """
    merged = default_config.copy() # Start with default structure

    for key, value in loaded_config.items():
        if key in merged:
            # If key exists in both and both values are dicts, recurse
            if isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = _merge_configs(value, merged[key])
            else:
                # Overwrite default with loaded value. Type validation happens later.
                merged[key] = value
        else:
            # Key from loaded config doesn't exist in default, add it (allows user extensions)
            merged[key] = value

    # Ensure all keys from default_config are present in merged (handles cases where a key was missing in loaded)
    for key, default_value in default_config.items():
        if key not in merged:
            merged[key] = default_value
            # print(f"Debug: Added missing key '{key}' from defaults during merge.") # Debug log

    return merged

# --- Logging Setup ---
def setup_logger(name: str, config: Dict[str, Any], level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with rotating file and colored console handlers based on config."""
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if logger is somehow reused (e.g., during reloads)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Capture all levels at the logger level

    # --- File Handler (Rotating) ---
    log_filename = os.path.join(LOG_DIRECTORY, f"{name}.log")
    try:
        # Ensure directory exists (should be done earlier, but double-check)
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use UTC timestamps in file logs for consistency
        file_formatter = SensitiveFormatter(
            # Consistent ISO 8601 format with milliseconds and UTC 'Z' indicator
            "%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S' # Use ISO 8601 standard date format
        )
        file_formatter.converter = time.gmtime # Ensure logs use UTC time
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to console logging if file setup fails
        print(f"{NEON_RED}Error setting up file logger {log_filename}: {e}. File logging disabled.{RESET}")
        if not logger.hasHandlers(): # Add basic stream handler if no handlers exist at all
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(basic_handler)

    # --- Console Handler (Colored) ---
    # Check if at least one handler exists before adding console handler
    # (Avoid duplicate console logs if file handler failed and added basic stream handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler()
        # Use local timezone specified in config for console display
        try:
            console_tz_str = config.get("timezone", "UTC") # Fallback to UTC if missing
            console_tz = ZoneInfo(console_tz_str)
        except Exception:
            print(f"{NEON_RED}Invalid timezone '{console_tz_str}' for console logs. Using UTC.{RESET}")
            console_tz = ZoneInfo("UTC")

        console_formatter = SensitiveFormatter(
            f"{NEON_BLUE}%(asctime)s{RESET} {NEON_YELLOW}%(levelname)-8s{RESET} {NEON_PURPLE}[%(name)s]{RESET} %(message)s",
            # Format string includes timezone name (%Z)
            datefmt='%Y-%m-%d %H:%M:%S %Z'
        )

        # Custom converter to generate local time tuples for the console formatter
        def local_time_converter(*args):
            # Capture current time in the configured local timezone
            return datetime.now(console_tz).timetuple()

        console_formatter.converter = local_time_converter # Use local time for display
        stream_handler.setFormatter(console_formatter)
        stream_handler.setLevel(level) # Set console level (e.g., INFO from function arg)
        logger.addHandler(stream_handler)

    logger.propagate = False # Prevent duplicate logs in root logger
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(config: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with V5 defaults and enhanced error handling."""
    lg = logger # Use alias for brevity
    try:
        # Use Decimal for number representations in options where applicable
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 150, # Default: ~6.6 req/s. Adjust based on specific V5 endpoint limits & VIP levels.
            'options': {
                'defaultType': 'linear', # Crucial for Bybit V5 USDT/USDC perpetuals/futures
                'adjustForTimeDifference': True, # Helps with timestamp issues
                # Set reasonable timeouts (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 25000,
                'fetchOHLCVTimeout': 20000,
                'fetchOrderBookTimeout': 15000,
                'setLeverageTimeout': 20000,
                'fetchMyTradesTimeout': 20000, # Added timeout
                'fetchClosedOrdersTimeout': 25000, # Added timeout
                # Custom User-Agent can help identify your bot traffic
                'user-agent': 'sxsBot/1.1 (+https://github.com/your_repo)', # Optional: Update URL if applicable
                # Bybit V5 specific settings (check if needed based on issues)
                # 'recvWindow': 10000, # Increase if timestamp errors persist despite adjustForTimeDifference
                # 'brokerId': 'YOUR_BROKER_ID', # If affiliated with Bybit broker program
                # 'enableUnifiedMargin': False, # Set True if using Unified Trading Account (UTA)
                # 'enableUnifiedAccount': False, # Alias for above? Check CCXT/Bybit docs
            }
        }

        # Ensure exchange is 'bybit' for this specialized setup
        exchange_id = "bybit"
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        # --- Sandbox Mode Setup ---
        if config.get('use_sandbox', True): # Default to sandbox if key is missing
            lg.warning(f"{NEON_YELLOW}INITIALIZING IN SANDBOX MODE (Testnet){RESET}")
            try:
                # CCXT's generic method
                exchange.set_sandbox_mode(True)
                lg.info(f"Sandbox mode enabled via exchange.set_sandbox_mode(True) for {exchange.id}.")
                # Explicitly check the API URL after setting sandbox mode
                if 'testnet' not in exchange.urls.get('api', ''):
                    lg.warning(f"set_sandbox_mode did not change API URL to testnet. Current API URL: {exchange.urls.get('api')}")
                    # Attempt manual override using known testnet URL from describe()
                    test_url = exchange.describe().get('urls', {}).get('test')
                    if isinstance(test_url, str):
                        exchange.urls['api'] = test_url
                        lg.info(f"Manually set API URL to Testnet: {exchange.urls['api']}")
                    elif isinstance(test_url, dict): # Sometimes 'test' key holds a dict
                        # Try to find a public/private key within the test dict
                        api_test_url = test_url.get('public') or test_url.get('private') or list(test_url.values())[0]
                        if api_test_url and isinstance(api_test_url, str):
                             exchange.urls['api'] = api_test_url
                             lg.info(f"Manually set API URL to Testnet (from dict): {exchange.urls['api']}")
                        else:
                              # Final hardcoded fallback if describe() structure is unexpected
                              fallback_test_url = 'https://api-testnet.bybit.com'
                              lg.warning(f"Could not reliably determine testnet URL. Hardcoding fallback: {fallback_test_url}")
                              exchange.urls['api'] = fallback_test_url
                    else:
                         # Final hardcoded fallback
                         fallback_test_url = 'https://api-testnet.bybit.com'
                         lg.warning(f"Could not reliably determine testnet URL. Hardcoding fallback: {fallback_test_url}")
                         exchange.urls['api'] = fallback_test_url
                else:
                     lg.info(f"Confirmed API URL is set to Testnet: {exchange.urls['api']}")

            except AttributeError:
                lg.warning(f"{exchange.id} ccxt version might not support set_sandbox_mode. Manually setting Testnet API URL.")
                # Manually set Bybit testnet URL (ensure this is correct for V5)
                exchange.urls['api'] = 'https://api-testnet.bybit.com'
                lg.info(f"Manually set Bybit API URL to Testnet: {exchange.urls['api']}")
            except Exception as e_sandbox:
                lg.error(f"Error enabling sandbox mode: {e_sandbox}. Ensure API keys are for Testnet. Proceeding with potentially incorrect URL.", exc_info=True)
        else:
            lg.info(f"{NEON_GREEN}INITIALIZING IN LIVE (Real Money) Environment.{RESET}")
            # Ensure API URL is production URL if sandbox was previously set somehow
            if 'testnet' in exchange.urls.get('api',''):
                lg.warning("Detected testnet URL while in live mode. Resetting to production URL.")
                # Find the production URL (might be under 'api' or 'www')
                prod_url = exchange.describe().get('urls', {}).get('api')
                if isinstance(prod_url, str):
                    exchange.urls['api'] = prod_url
                elif isinstance(prod_url, dict): # API URL might be nested
                    api_prod_url = prod_url.get('public') or prod_url.get('private') or list(prod_url.values())[0]
                    if api_prod_url and isinstance(api_prod_url, str):
                         exchange.urls['api'] = api_prod_url
                    else: # Try 'www' as fallback
                         www_url = exchange.describe().get('urls',{}).get('www')
                         if www_url and isinstance(www_url, str):
                              exchange.urls['api'] = www_url # Less ideal, but better than testnet
                         else:
                              lg.error("Could not determine production API URL automatically. Using potentially incorrect URL.")
                else: # Try 'www' if 'api' wasn't found or invalid
                     www_url = exchange.describe().get('urls',{}).get('www')
                     if www_url and isinstance(www_url, str):
                          exchange.urls['api'] = www_url
                     else:
                          lg.error("Could not determine production API URL automatically. Using potentially incorrect URL.")

                lg.info(f"Reset API URL to Production (best guess): {exchange.urls.get('api')}")


        lg.info(f"Initializing {exchange.id} (API: {exchange.urls.get('api', 'URL Not Set')})...")

        # --- Load Markets (Essential for precision, limits, IDs) ---
        lg.info(f"Loading markets for {exchange.id} (may take a moment)...")
        try:
             # Use safe_api_call for robustness during market loading
             safe_api_call(exchange.load_markets, lg, reload=True) # Force reload
             lg.info(f"Markets loaded successfully for {exchange.id}. Found {len(exchange.symbols)} symbols.")

             # Validate target symbol existence and compatibility after loading
             target_symbol = config.get("symbol")
             if target_symbol and target_symbol not in exchange.markets:
                  lg.error(f"{NEON_RED}FATAL: Target symbol '{target_symbol}' not found in loaded markets! Check symbol format/availability on the exchange.{RESET}")
                  # Suggest common Bybit format if applicable
                  if '/' in target_symbol and ':' not in target_symbol:
                       base, quote = target_symbol.split('/')
                       suggested_symbol = f"{base}/{quote}:{quote}"
                       lg.warning(f"{NEON_YELLOW}Hint: For Bybit V5 linear perpetuals, the format is usually like '{suggested_symbol}'.{RESET}")
                  # List available markets if feasible
                  if 0 < len(exchange.symbols) < 50:
                       lg.debug(f"Available symbols: {sorted(exchange.symbols)}")
                  elif len(exchange.symbols) == 0:
                       lg.error("No symbols were loaded from the exchange.")
                  else:
                       lg.warning(f"{NEON_YELLOW}Check the symbol '{target_symbol}' on the Bybit website/API documentation.{RESET}")
                  return None # Fatal error if configured symbol doesn't exist
             else:
                  lg.info(f"Target symbol '{target_symbol}' found in loaded markets.")
                  # Optional: Add check for linear/contract type compatibility here if needed

        except Exception as market_err:
             lg.critical(f"{NEON_RED}CRITICAL: Failed to load markets after retries: {market_err}. Cannot operate without market data. Exiting.{RESET}", exc_info=True)
             return None # Fatal error

        # --- Initial Connection & Permissions Test (Fetch Balance - Crucial for V5 Account Type) ---
        # For V5, fetch balance for the relevant account type (e.g., CONTRACT or UNIFIED)
        account_type_to_test = 'CONTRACT' # Common for derivatives, might need to check for UNIFIED later if CONTRACT fails
        lg.info(f"Performing initial connection test by fetching balance (Account Type Hint: {account_type_to_test})...")
        quote_curr = config.get("quote_currency", "USDT")
        balance_decimal = fetch_balance(exchange, quote_curr, lg) # Use the dedicated function

        if balance_decimal is not None:
             lg.info(f"{NEON_GREEN}Successfully connected and fetched initial {quote_curr} balance: {balance_decimal:.4f}{RESET}")
             if balance_decimal == 0:
                  lg.warning(f"{NEON_YELLOW}Initial {quote_curr} balance is zero. Ensure funds are available in the correct account type (CONTRACT/UNIFIED).{RESET}")
        else:
             # fetch_balance logs errors, add a critical warning here as it indicates potential issues
             lg.critical(f"{NEON_RED}CRITICAL: Initial balance fetch for {quote_curr} failed. Check API permissions (validity, IP whitelist, read access), account type (CONTRACT/UNIFIED?), and network connection.{RESET}")
             # Consider if this should be fatal. For trading, it likely is.
             # return None # Make it fatal if balance fetch fails

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {config.get('use_sandbox')}, Default Type: {exchange.options.get('defaultType')}")
        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check API Key/Secret format, validity, permissions, and IP whitelisting in your .env file and on the exchange website.{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> This could be temporary, or indicate issues with exchange settings or API endpoints.{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error during initialization: {e}{RESET}")
        lg.critical(f"{NEON_RED}>> Check your internet connection and firewall settings.{RESET}")
    except Exception as e:
        lg.critical(f"{NEON_RED}Unexpected error initializing CCXT exchange: {e}{RESET}", exc_info=True)

    return None

# --- API Call Wrapper with Retries ---
def safe_api_call(func, logger: logging.Logger, *args, **kwargs):
    """
    Wraps an API call with retry logic for network/rate limit/specific exchange errors.
    Uses exponential backoff and handles common retryable scenarios.
    """
    lg = logger
    # Use configuration for retries and delay, fallback to constants
    max_retries = config.get("max_api_retries", MAX_API_RETRIES) if 'config' in globals() else MAX_API_RETRIES
    base_retry_delay = config.get("retry_delay", RETRY_DELAY_SECONDS) if 'config' in globals() else RETRY_DELAY_SECONDS
    attempts = 0
    last_exception = None

    while attempts <= max_retries:
        try:
            result = func(*args, **kwargs)
            # Log successful call completion at DEBUG level (can be verbose)
            lg.debug(f"API call '{func.__name__}' successful (Attempt {attempts+1}).")
            return result # Success

        # --- Retryable Network/Availability Errors ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
            last_exception = e
            # Exponential backoff for network issues
            wait_time = base_retry_delay * (1.5 ** attempts)
            # Add random jitter (e.g., +/- 10%) to prevent thundering herd
            wait_time *= (1 + (np.random.rand() - 0.5) * 0.2)
            wait_time = min(wait_time, 60) # Cap wait time to avoid excessive delays
            lg.warning(f"{NEON_YELLOW}Retryable network/availability error in '{func.__name__}': {type(e).__name__}. "
                       f"Waiting {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1}). Error: {e}{RESET}")

        # --- Rate Limit Errors ---
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            retry_after_header = None
            header_wait = 0
            # Try to get retry-after header info (might be in seconds or ms)
            if hasattr(e, 'http_headers') and e.http_headers: # Check ccxt standard attribute first
                 retry_after_header = e.http_headers.get('Retry-After') or e.http_headers.get('retry-after')
            elif hasattr(e, 'response') and hasattr(e.response, 'headers'): # Fallback check
                 retry_after_header = e.response.headers.get('Retry-After') or e.response.headers.get('retry-after')

            # Stronger exponential backoff for rate limits
            wait_time = base_retry_delay * (2.0 ** attempts)
            wait_time *= (1 + (np.random.rand() - 0.5) * 0.2) # Add jitter

            if retry_after_header:
                try:
                    header_wait = float(retry_after_header) + 0.5 # Assume seconds, add buffer
                    # Check if it looks like milliseconds (e.g., > 10000) and convert
                    if header_wait > 1000: header_wait = (header_wait / 1000.0) + 0.5
                    lg.debug(f"Rate limit Retry-After header detected: {retry_after_header} -> Parsed wait: {header_wait:.1f}s")
                except (ValueError, TypeError):
                    lg.warning(f"Could not parse Retry-After header value: {retry_after_header}")
                    header_wait = 0 # Reset if parsing fails

            # Use the longer of calculated backoff or header wait time
            final_wait_time = max(wait_time, header_wait)
            final_wait_time = min(final_wait_time, 90) # Cap rate limit wait time

            lg.warning(f"{NEON_YELLOW}Rate limit exceeded in '{func.__name__}'. Waiting {final_wait_time:.1f}s "
                       f"(Attempt {attempts+1}/{max_retries+1}). Error: {e}{RESET}")
            wait_time = final_wait_time # Use the determined wait time for the sleep

        # --- Authentication Errors (Non-Retryable) ---
        except ccxt.AuthenticationError as e:
             lg.error(f"{NEON_RED}Authentication Error in '{func.__name__}': {e}. Aborting call.{RESET}")
             lg.error(f"{NEON_RED}>> Check API Key/Secret validity, permissions, IP whitelist, and environment (Live/Testnet).{RESET}")
             raise e # Don't retry, re-raise immediately

        # --- Exchange Specific Errors (Potentially Retryable) ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_str = str(e).lower()
            http_status_code = getattr(e, 'http_status_code', None)
            exchange_code = None

            # Try extracting Bybit's retCode from the message (common pattern)
            try:
                 # Example: "bybit {"retCode":10006,"retMsg":"Too many visits!"...}"
                 # Example: "[10006] request frequent"
                 start_index = err_str.find('"retcode":')
                 if start_index != -1:
                      code_part = err_str[start_index + len('"retcode":'):]
                      end_index = code_part.find(',')
                      code_str = code_part[:end_index].strip()
                      if code_str.isdigit(): exchange_code = int(code_str)
                 elif err_str.startswith('[') and ']' in err_str: # Alternative format
                      code_str = err_str[1:err_str.find(']')]
                      if code_str.isdigit(): exchange_code = int(code_str)
            except Exception: pass # Ignore parsing errors

            # Bybit V5 specific transient error codes (add more as identified)
            # Reference: https://bybit-exchange.github.io/docs/v5/error_code
            bybit_retry_codes = [
                10001, # Internal server error
                10002, # Service unavailable / Server error
                10006, # Too many requests (might be caught by RateLimitExceeded too)
                10016, # Service temporarily unavailable due to maintenance or upgrade
                10018, # Request validation failed (sometimes transient)
                130021, # Order quantity is invalid (sometimes due to temporary price/precision issue)
                130150, # System busy, please try again later
                131204, # Cannot connect to matching engine (transient)
                170131, # Too many requests (contract specific?)
                # Add others based on experience
            ]
            # General retryable messages (case-insensitive)
            retryable_messages = [
                 "internal server error", "service unavailable", "system busy",
                 "request validation failed", "matching engine busy", "please try again later",
                 "nonce is too small", # Can happen with adjustForTimeDifference off or clock drift
                 "order placement optimization" # Occasional Bybit transient message
            ]

            is_retryable = False
            if exchange_code in bybit_retry_codes: is_retryable = True
            if not is_retryable and http_status_code in RETRYABLE_HTTP_CODES: is_retryable = True
            if not is_retryable and any(msg in err_str for msg in retryable_messages): is_retryable = True

            if is_retryable:
                 # Use standard backoff for these transient errors
                 wait_time = base_retry_delay * (1.5 ** attempts)
                 wait_time *= (1 + (np.random.rand() - 0.5) * 0.2) # Add jitter
                 wait_time = min(wait_time, 60) # Cap wait time
                 lg.warning(f"{NEON_YELLOW}Potentially retryable exchange error in '{func.__name__}': {e} (Code: {exchange_code}, HTTP: {http_status_code}). "
                            f"Waiting {wait_time:.1f}s (Attempt {attempts+1}/{max_retries+1})...{RESET}")
                 time.sleep(wait_time) # Sleep here inside the except block
                 attempts += 1
                 continue # Skip incrementing attempt at the end and retry directly
            else:
                 # Non-retryable exchange error
                 lg.error(f"{NEON_RED}Non-retryable Exchange Error in '{func.__name__}': {e} (Code: {exchange_code}, HTTP: {http_status_code}){RESET}")
                 raise e # Re-raise immediately

        # --- Catch any other unexpected error ---
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error during API call '{func.__name__}': {e}{RESET}", exc_info=True)
            raise e # Re-raise unexpected errors immediately

        # --- Sleep and Increment Attempt ---
        # Ensure wait_time is defined even if no exception matched above (shouldn't happen with current structure)
        if attempts <= max_retries:
            # Ensure sleep happens only if a retry is needed and wait_time was calculated
            calculated_wait_time = locals().get('wait_time', 0) # Get wait_time if set in except blocks
            if calculated_wait_time > 0:
                time.sleep(calculated_wait_time)
            else: # Should not happen, but prevents infinite loop if wait_time is missed
                 lg.warning(f"Wait time not calculated for retry attempt {attempts+1}. Using base delay.")
                 time.sleep(base_retry_delay)

            attempts += 1
        else: # Should be handled by the loop condition, but as safety break
            break


    # If loop completes, max retries exceeded
    lg.error(f"{NEON_RED}Max retries ({max_retries+1}) exceeded for API call '{func.__name__}'. Last Error: {type(last_exception).__name__}{RESET}")
    if last_exception:
        raise last_exception # Raise the last known exception
    else:
        # Fallback if no exception was captured (shouldn't normally happen)
        raise ccxt.RequestTimeout(f"Max retries exceeded for {func.__name__} (no specific exception captured during retry loop)")


# --- CCXT Data Fetching (Using safe_api_call) ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks, retries, and Decimal conversion."""
    lg = logger
    try:
        # Use safe_api_call for robustness
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol)
        if not ticker:
            lg.error(f"Failed to fetch ticker for {symbol} after retries (returned None or empty).")
            return None

        lg.debug(f"Raw Ticker data for {symbol}: {json.dumps(ticker, indent=2)}") # Log full ticker for debug

        # --- Robust Decimal conversion helper ---
        def to_decimal(value, context_str: str = "price") -> Optional[Decimal]:
            """Safely converts a value to a positive, finite Decimal."""
            if value is None: return None
            try:
                d = Decimal(str(value))
                # Ensure the price is finite and positive
                if d.is_finite() and d > 0:
                    return d
                else:
                    lg.debug(f"Invalid {context_str} value (non-finite or non-positive): {value}. Discarding.")
                    return None
            except (InvalidOperation, ValueError, TypeError):
                lg.debug(f"Invalid {context_str} format, cannot convert to Decimal: {value}. Discarding.")
                return None

        # --- Price Extraction Logic with Priority ---
        # Priorities: last > mark (if contract) > close > average > mid(bid/ask) > ask > bid
        p_last = to_decimal(ticker.get('last'), 'last price')
        p_mark = to_decimal(ticker.get('mark'), 'mark price') # Relevant for contracts
        p_close = to_decimal(ticker.get('close', ticker.get('last')), 'close price') # Use 'close', fallback to 'last'
        p_bid = to_decimal(ticker.get('bid'), 'bid price')
        p_ask = to_decimal(ticker.get('ask'), 'ask price')

        # Calculate average/mid if needed and possible
        p_avg = to_decimal(ticker.get('average'), 'average price')
        p_mid = None
        if p_bid is not None and p_ask is not None:
             # Ensure bid < ask before calculating mid
             if p_bid < p_ask:
                 p_mid = (p_bid + p_ask) / Decimal('2')
                 # Validate mid calculation didn't result in non-finite
                 if not p_mid.is_finite(): p_mid = None
             else: lg.debug(f"Bid ({p_bid}) is not less than Ask ({p_ask}). Cannot calculate Mid price.")

        # Determine market type for priority setting
        market_info = exchange.market(symbol) if symbol in exchange.markets else {}
        is_contract = market_info.get('contract', False) or market_info.get('type') in ['swap', 'future']

        # Select price based on priority
        price = None
        source = "N/A"

        if is_contract and p_mark:
            price, source = p_mark, "Mark Price (Contract)"
        elif p_last:
            price, source = p_last, "Last Price"
        elif p_close:
            price, source = p_close, "Close Price"
        elif p_avg:
            price, source = p_avg, "Average Price"
        elif p_mid:
            price, source = p_mid, "Mid Price (Bid/Ask)"
        elif p_ask:
            # Check spread before using only Ask
            if p_bid:
                 spread_pct = ((p_ask - p_bid) / p_ask) * 100 if p_ask > 0 else Decimal('0')
                 if spread_pct > Decimal('2.0'): # Warn if spread > 2%
                      lg.warning(f"Using 'ask' price ({p_ask}) as fallback, but spread seems large ({spread_pct:.2f}%, Bid: {p_bid}).")
            price, source = p_ask, "Ask Price (Fallback)"
        elif p_bid:
            price, source = p_bid, "Bid Price (Last Resort Fallback)"

        # --- Final Validation ---
        if price is not None and price.is_finite() and price > 0:
            lg.info(f"Current price ({symbol}): {price} (Source: {source})")
            return price
        else:
            lg.error(f"{NEON_RED}Failed to extract a valid positive price from ticker data for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except Exception as e:
        # Catch errors raised by safe_api_call or during parsing
        lg.error(f"{NEON_RED}Error fetching/processing current price for {symbol}: {e}{RESET}", exc_info=False) # Keep log concise
        return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries, robust validation, and Decimal conversion."""
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    empty_df = pd.DataFrame() # Return this on failure

    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return empty_df

    try:
        # Convert our interval format to CCXT's expected format
        ccxt_timeframe = CCXT_INTERVAL_MAP.get(timeframe)
        if not ccxt_timeframe:
            lg.error(f"Invalid timeframe '{timeframe}' provided. Valid intervals: {list(VALID_INTERVALS)}. Cannot fetch klines.")
            return empty_df

        lg.debug(f"Fetching {limit} klines for {symbol} with timeframe {ccxt_timeframe} (Config: {timeframe})...")
        # Use safe_api_call to handle retries
        ohlcv_data = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe=ccxt_timeframe, limit=limit)

        if ohlcv_data is None or not isinstance(ohlcv_data, list) or len(ohlcv_data) == 0:
            # Error logged by safe_api_call if failed after retries
            if ohlcv_data is not None: # Log only if it returned empty list/None without raising error
                lg.warning(f"{NEON_YELLOW}No valid kline data returned by fetch_ohlcv for {symbol} {ccxt_timeframe}. Check symbol/interval/exchange status.{RESET}")
            return empty_df

        # Process the data into a pandas DataFrame
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            lg.warning(f"Kline data DataFrame is empty after initial creation for {symbol} {ccxt_timeframe}.")
            return empty_df

        # --- Data Cleaning and Type Conversion ---
        # 1. Convert timestamp to datetime objects (UTC), coerce errors, set as index
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
            initial_len_ts = len(df)
            df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
            if len(df) < initial_len_ts:
                lg.debug(f"Dropped {initial_len_ts - len(df)} rows with invalid timestamps.")
            if df.empty:
                lg.warning("DataFrame empty after timestamp conversion/dropna.")
                return empty_df
            df.set_index('timestamp', inplace=True)
        except Exception as ts_err:
             lg.error(f"Error processing timestamps: {ts_err}. Returning empty DataFrame.", exc_info=True)
             return empty_df

        # 2. Convert price/volume columns to Decimal, handling errors robustly
        cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
        for col in cols_to_convert:
            if col not in df.columns:
                lg.warning(f"Required column '{col}' missing in fetched kline data.")
                continue # Skip missing column
            try:
                # Use a helper for safe conversion to Decimal
                def safe_to_decimal(x, col_name) -> Decimal:
                    """Converts input to Decimal, returning Decimal('NaN') on failure or non-finite."""
                    if pd.isna(x) or str(x).strip() == '': return Decimal('NaN')
                    try:
                        d = Decimal(str(x))
                        # Check for finite and positive (for price/volume)
                        # Allow zero volume, but prices should be positive
                        is_price = col_name in ['open', 'high', 'low', 'close']
                        if d.is_finite() and (d > 0 if is_price else d >= 0):
                            return d
                        else:
                            # lg.debug(f"Non-finite or non-positive value in '{col_name}': {x}")
                            return Decimal('NaN')
                    except (InvalidOperation, TypeError, ValueError):
                        # lg.debug(f"Could not convert '{x}' to Decimal in column '{col_name}', returning NaN.")
                        return Decimal('NaN')

                df[col] = df[col].apply(lambda x: safe_to_decimal(x, col))

            except Exception as conv_err: # Catch unexpected errors during apply
                 lg.error(f"Unexpected error converting column '{col}' to Decimal: {conv_err}. Trying float conversion.", exc_info=True)
                 # Fallback: try converting to float, coerce errors to NaN
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 # After float conversion, check for non-finite floats and replace with NaN
                 if pd.api.types.is_float_dtype(df[col]):
                     df[col] = df[col].apply(lambda x: x if np.isfinite(x) else np.nan)

        # 3. Drop rows with NaN in essential price columns (O, H, L, C)
        initial_len_nan = len(df)
        essential_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_cols, how='any', inplace=True)
        rows_dropped_nan = initial_len_nan - len(df)
        if rows_dropped_nan > 0:
            lg.debug(f"Dropped {rows_dropped_nan} rows with NaN price data for {symbol}.")

        # 4. Additional check: Ensure OHLC consistency (e.g., H >= O, H >= L, H >= C, L <= O, L <= C)
        # Convert to numeric for comparison if not Decimal after fallbacks
        for col in essential_cols:
            if col in df.columns and not isinstance(df[col].iloc[0], Decimal):
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        try: # Wrap comparison in try-except for safety
            invalid_ohlc_mask = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                                (df['low'] > df['open']) | (df['low'] > df['close'])
            invalid_count = invalid_ohlc_mask.sum()
            if invalid_count > 0:
                lg.warning(f"Found {invalid_count} rows with inconsistent OHLC data (e.g., High < Low). Dropping these rows.")
                df = df[~invalid_ohlc_mask]
        except TypeError as cmp_err:
            # Might happen if conversion to numeric failed or mix of types
            lg.warning(f"Could not perform OHLC consistency check due to type error: {cmp_err}. Skipping.")
        except Exception as cmp_err:
             lg.warning(f"Error during OHLC consistency check: {cmp_err}. Skipping.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} {ccxt_timeframe} became empty after cleaning (NaN drop or OHLC check).")
            return empty_df

        # 5. Sort by timestamp index and remove duplicates (keeping the last occurrence)
        df.sort_index(inplace=True)
        if df.index.has_duplicates:
            num_duplicates = df.index.duplicated().sum()
            lg.debug(f"Found {num_duplicates} duplicate timestamps. Keeping last entry for each.")
            df = df[~df.index.duplicated(keep='last')]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {ccxt_timeframe} (requested {limit})")
        if lg.isEnabledFor(logging.DEBUG) and not df.empty:
             # Log head/tail only if DEBUG is enabled and df is not empty
             lg.debug(f"Kline check: First row:\n{df.head(1)}\nLast row:\n{df.tail(1)}")
        return df

    except ValueError as ve: # Catch validation errors raised within the function
        lg.error(f"{NEON_RED}Kline fetch/processing error for {symbol}: {ve}{RESET}")
        return empty_df
    except Exception as e:
        # Catch errors from safe_api_call or during processing
        lg.error(f"{NEON_RED}Unexpected error fetching/processing klines for {symbol}: {e}{RESET}", exc_info=True)
        return empty_df


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries, validation, and Decimal conversion."""
    lg = logger
    if not exchange.has['fetchOrderBook']:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    try:
        lg.debug(f"Fetching order book for {symbol} with limit {limit}...")
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit)

        if not orderbook: # Error already logged by safe_api_call if it failed after retries
            lg.warning(f"fetch_order_book for {symbol} returned None after retries.")
            return None

        # --- Validate Structure and Content ---
        if not isinstance(orderbook, dict) or \
           'bids' not in orderbook or 'asks' not in orderbook or \
           not isinstance(orderbook['bids'], list) or not isinstance(orderbook['asks'], list):
            lg.warning(f"Invalid orderbook structure received for {symbol}. Data: {orderbook}")
            return None

        # --- Convert prices and amounts to Decimal ---
        cleaned_book = {
            'bids': [], 'asks': [],
            'timestamp': orderbook.get('timestamp'), # Keep original metadata
            'datetime': orderbook.get('datetime'),
            'nonce': orderbook.get('nonce')
            }
        conversion_errors = 0
        invalid_format_count = 0

        for side in ['bids', 'asks']:
            for entry in orderbook[side]:
                if isinstance(entry, list) and len(entry) == 2:
                    try:
                        # Convert price and amount to Decimal
                        price = Decimal(str(entry[0]))
                        amount = Decimal(str(entry[1]))
                        # Validate: finite, price > 0, amount >= 0
                        if price.is_finite() and price > 0 and amount.is_finite() and amount >= 0:
                            cleaned_book[side].append([price, amount])
                        else:
                            # lg.debug(f"Invalid price/amount in {side} entry: P={price}, A={amount}") # Verbose
                            conversion_errors += 1
                    except (InvalidOperation, ValueError, TypeError):
                        # lg.debug(f"Conversion error for {side} entry: {entry}") # Verbose
                        conversion_errors += 1
                else:
                    # lg.warning(f"Invalid {side[:-1]} entry format in orderbook: {entry}")
                    invalid_format_count += 1

        if conversion_errors > 0:
            lg.debug(f"Orderbook ({symbol}): Encountered {conversion_errors} entries with invalid/non-finite/non-positive values.")
        if invalid_format_count > 0:
            lg.warning(f"Orderbook ({symbol}): Encountered {invalid_format_count} entries with invalid format (expected [price, amount]).")

        # Ensure bids are sorted descending and asks ascending (ccxt usually guarantees this, but verify)
        # Use Decimal comparison for sorting
        cleaned_book['bids'].sort(key=lambda x: x[0], reverse=True)
        cleaned_book['asks'].sort(key=lambda x: x[0])

        # Proceed even if some entries failed, but log if book becomes empty
        if not cleaned_book['bids'] and not cleaned_book['asks']:
            lg.warning(f"Orderbook for {symbol} is empty after cleaning/conversion.")
            # Return the empty book structure if fetch itself succeeded
            return cleaned_book
        elif not cleaned_book['bids']:
             lg.warning(f"Orderbook ({symbol}) has no valid bids after cleaning.")
        elif not cleaned_book['asks']:
             lg.warning(f"Orderbook ({symbol}) has no valid asks after cleaning.")


        lg.debug(f"Successfully fetched and processed orderbook for {symbol} ({len(cleaned_book['bids'])} bids, {len(cleaned_book['asks'])} asks).")
        return cleaned_book

    except Exception as e:
        # Catch errors raised by safe_api_call or other validation issues
        lg.error(f"{NEON_RED}Error fetching or processing order book for {symbol}: {e}{RESET}", exc_info=False)
        return None

# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """
    Analyzes trading data using pandas_ta and generates weighted signals.
    Handles Decimal/float conversions for TA calculations and results.
    Provides methods for accessing market precision and calculating TP/SL.
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
            df: Pandas DataFrame with OHLCV data (expects Decimal values), indexed by timestamp.
            logger: Logger instance for logging messages.
            config: Dictionary containing bot configuration.
            market_info: Dictionary containing market details (precision, limits, etc.).
        """
        self.df = df.copy() # Work on a copy to avoid modifying original DF
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "N/A")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)

        # Stores latest indicator values (Decimal for prices/ATR, float for others)
        self.indicator_values: Dict[str, Union[Decimal, float, None]] = {}
        # Stores generated pandas_ta column names mapped to internal keys
        self.ta_column_names: Dict[str, Optional[str]] = {}
        # Stores calculated Fibonacci levels (Decimal)
        self.fib_levels_data: Dict[str, Decimal] = {}
        # Active weight set and weights
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})

        if not self.weights:
            logger.warning(f"{NEON_YELLOW}Active weight set '{self.active_weight_set_name}' not found or empty for {self.symbol}. Scoring may be zero.{RESET}")
            self.weights = {} # Use empty dict to prevent errors

        # Validate DataFrame and perform initial calculations
        self._initialize_analysis()

    def _initialize_analysis(self) -> None:
        """Checks DataFrame validity and runs initial calculations."""
        if self.df.empty:
             self.logger.warning(f"TradingAnalyzer initialized with empty DataFrame for {self.symbol}. No calculations performed.")
             return

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in required_cols):
            self.logger.error(f"DataFrame missing required columns: {required_cols}. Found: {self.df.columns.tolist()}. Cannot perform analysis.")
            self.df = pd.DataFrame() # Clear DF to prevent further errors
            return
        if self.df[required_cols].isnull().all().all():
             self.logger.error("DataFrame contains all NaN values in required OHLCV columns. Cannot perform analysis.")
             self.df = pd.DataFrame() # Clear DF
             return

        # Proceed with calculations
        try:
            self._calculate_all_indicators()
            self._update_latest_indicator_values() # Run AFTER indicator calculation
            self.calculate_fibonacci_levels() # Calculate initial Fib levels
        except Exception as init_calc_err:
             self.logger.error(f"Error during initial indicator/Fibonacci calculation for {self.symbol}: {init_calc_err}", exc_info=True)
             # Depending on severity, could clear self.df here too


    def _get_ta_col_name(self, base_name: str, result_df_columns: List[str]) -> Optional[str]:
        """
        Helper to find the actual column name generated by pandas_ta.
        Searches for expected patterns based on config parameters. More robust matching.

        Args:
            base_name: Internal identifier for the indicator (e.g., "ATR", "EMA_Short").
            result_df_columns: List of column names present in the DataFrame after TA calculation.

        Returns:
            The matched column name string, or None if not found.
        """
        if not result_df_columns: return None

        # --- Define expected patterns dynamically based on current config ---
        # Use float representation matching pandas_ta common output (e.g., '2.0' for std dev)
        bb_std_dev_str = f"{float(self.config.get('bollinger_bands_std_dev', DEFAULT_BBANDS_STDDEV)):.1f}"
        psar_step_str = str(float(self.config.get('psar_step', DEFAULT_PSAR_STEP)))
        psar_max_str = str(float(self.config.get('psar_max_step', DEFAULT_PSAR_MAX_STEP)))
        # Handle potential integer vs float representation in pandas_ta column names
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
        params = {key: self.config.get(key, default) for key, default in param_keys}

        # Map internal base names to lists of potential pandas_ta column name patterns
        expected_patterns = {
            "ATR": [f"ATRr_{params['atr_period']}"],
            "EMA_Short": [f"EMA_{params['ema_short_period']}"],
            "EMA_Long": [f"EMA_{params['ema_long_period']}"],
            "Momentum": [f"MOM_{params['momentum_period']}"],
            "CCI": [f"CCI_{params['cci_period']}", f"CCI_{params['cci_period']}_0.015"], # Common CCI suffix
            "Williams_R": [f"WILLR_{params['williams_r_period']}"],
            "MFI": [f"MFI_{params['mfi_period']}"],
            "VWAP": ["VWAP_D"], # Default pandas_ta VWAP is often daily anchored
            "PSAR_long": [f"PSARl_{psar_step_str}_{psar_max_str}"],
            "PSAR_short": [f"PSARs_{psar_step_str}_{psar_max_str}"],
            "PSAR_af": [f"PSARaf_{psar_step_str}_{psar_max_str}"],
            "PSAR_rev": [f"PSARr_{psar_step_str}_{psar_max_str}"],
            "SMA_10": [f"SMA_{params['sma_10_period']}"],
            "StochRSI_K": [f"STOCHRSIk_{params['stoch_rsi_period']}_{params['stoch_rsi_rsi_period']}_{params['stoch_rsi_k_period']}"],
            "StochRSI_D": [f"STOCHRSId_{params['stoch_rsi_period']}_{params['stoch_rsi_rsi_period']}_{params['stoch_rsi_k_period']}_{params['stoch_rsi_d_period']}"],
            "RSI": [f"RSI_{params['rsi_period']}"],
            "BB_Lower": [f"BBL_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Middle": [f"BBM_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Upper": [f"BBU_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Bandwidth": [f"BBB_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            "BB_Percent": [f"BBP_{params['bollinger_bands_period']}_{bb_std_dev_str}"],
            # Custom name used for Volume MA in _calculate_all_indicators
            "Volume_MA": [f"VOL_SMA_{params['volume_ma_period']}"]
        }

        patterns_to_check = expected_patterns.get(base_name, [])
        if not patterns_to_check:
            # self.logger.debug(f"No expected column pattern defined for indicator base name: '{base_name}'")
            return None

        # --- Search Strategy ---
        # 1. Exact Match First (most reliable)
        for pattern in patterns_to_check:
            if pattern in result_df_columns:
                # self.logger.debug(f"Mapped '{base_name}' to column '{pattern}' (Exact Match)")
                return pattern

        # 2. Case-Insensitive Exact Match
        patterns_lower = [p.lower() for p in patterns_to_check]
        cols_lower_map = {col.lower(): col for col in result_df_columns}
        for i, pattern_lower in enumerate(patterns_lower):
             if pattern_lower in cols_lower_map:
                  original_col_name = cols_lower_map[pattern_lower]
                  # self.logger.debug(f"Mapped '{base_name}' to column '{original_col_name}' (Case-Insensitive Exact Match)")
                  return original_col_name

        # 3. Starts With Match (Handles potential suffixes like CCI_20_0.015)
        for pattern in patterns_to_check:
            # Check both exact case and lower case startswith
            pattern_lower = pattern.lower()
            for col in result_df_columns:
                col_lower = col.lower()
                # Check if column starts with the pattern (case-sensitive or insensitive)
                if col.startswith(pattern) or col_lower.startswith(pattern_lower):
                     # Add check: ensure suffix doesn't indicate a different indicator type
                     suffix = col[len(pattern):]
                     if not any(c.isalpha() for c in suffix): # Allow numbers, underscores, periods in suffix
                          # self.logger.debug(f"Mapped '{base_name}' to column '{col}' (StartsWith Match: '{pattern}')")
                          return col

        # 4. Fallback: Simple base name substring check (more risky)
        # Example: base_name 'StochRSI_K' -> simple_base 'stochrsi'
        simple_base = base_name.split('_')[0].lower()
        potential_matches = [col for col in result_df_columns if simple_base in col.lower()]

        if len(potential_matches) == 1:
             match = potential_matches[0]
             # self.logger.debug(f"Mapped '{base_name}' to '{match}' via unique simple substring search ('{simple_base}').")
             return match
        elif len(potential_matches) > 1:
              # If multiple substring matches, try to find one closer to the expected full patterns
              for pattern in patterns_to_check:
                   if pattern in potential_matches: # Check if full expected pattern is among matches
                        # self.logger.debug(f"Mapped '{base_name}' to '{pattern}' resolving ambiguous substring search.")
                        return pattern
              # If still ambiguous, log warning and return None (safer)
              self.logger.warning(f"Ambiguous substring match for '{base_name}' ('{simple_base}'): Found {potential_matches}. Could not resolve clearly based on expected patterns: {patterns_to_check}.")
              return None

        # If no match found by any method
        self.logger.debug(f"Could not find matching column name for indicator '{base_name}' (Expected patterns: {patterns_to_check}) in DataFrame columns.")
        return None


    def _calculate_all_indicators(self):
        """Calculates all enabled indicators using pandas_ta, handling types and column names."""
        if self.df.empty:
            self.logger.warning(f"DataFrame is empty, cannot calculate indicators for {self.symbol}.")
            return

        # --- Determine minimum required data length based on enabled & weighted indicators ---
        required_periods = []
        indicators_config = self.config.get("indicators", {})
        active_weights = self.weights # Use stored weights from init

        def add_req_if_active(indicator_key, config_period_key, default_period):
            """Adds period requirement if indicator is enabled and has non-zero weight."""
            is_enabled = indicators_config.get(indicator_key, False)
            # Check weight using float conversion for safety
            try: weight = float(active_weights.get(indicator_key, 0.0))
            except (ValueError, TypeError): weight = 0.0

            if is_enabled and weight > 0:
                # Get period from config or default, ensure it's a positive integer
                try:
                    period = int(self.config.get(config_period_key, default_period))
                    if period > 0: required_periods.append(period)
                    else: self.logger.warning(f"Invalid zero/negative period for {config_period_key}. Ignoring for length check.")
                except (ValueError, TypeError):
                     self.logger.warning(f"Invalid period format for {config_period_key}. Ignoring for length check.")

        # Add requirements for indicators with periods
        add_req_if_active("atr", "atr_period", DEFAULT_ATR_PERIOD)
        add_req_if_active("momentum", "momentum_period", DEFAULT_MOMENTUM_PERIOD)
        add_req_if_active("cci", "cci_period", DEFAULT_CCI_PERIOD)
        add_req_if_active("wr", "williams_r_period", DEFAULT_WILLIAMS_R_PERIOD)
        add_req_if_active("mfi", "mfi_period", DEFAULT_MFI_PERIOD)
        add_req_if_active("sma_10", "sma_10_period", DEFAULT_SMA10_PERIOD)
        add_req_if_active("rsi", "rsi_period", DEFAULT_RSI_PERIOD)
        add_req_if_active("bollinger_bands", "bollinger_bands_period", DEFAULT_BBANDS_PERIOD)
        add_req_if_active("volume_confirmation", "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)

        # Fibonacci period (used for high/low range, not TA lib directly but needs data)
        try: fib_period = int(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))
        except (ValueError, TypeError): fib_period = DEFAULT_FIB_PERIOD
        if fib_period > 0: required_periods.append(fib_period)

        # Compound indicators: EMA Alignment requires short and long EMAs
        if indicators_config.get("ema_alignment", False) and float(active_weights.get("ema_alignment", 0.0)) > 0:
             add_req_if_active("ema_alignment", "ema_short_period", DEFAULT_EMA_SHORT_PERIOD) # Use proxy key
             add_req_if_active("ema_alignment", "ema_long_period", DEFAULT_EMA_LONG_PERIOD)

        # StochRSI needs its main period and the underlying RSI period
        if indicators_config.get("stoch_rsi", False) and float(active_weights.get("stoch_rsi", 0.0)) > 0:
             add_req_if_active("stoch_rsi", "stoch_rsi_period", DEFAULT_STOCH_RSI_PERIOD)
             add_req_if_active("stoch_rsi", "stoch_rsi_rsi_period", DEFAULT_STOCH_RSI_RSI_PERIOD)

        # Calculate minimum required length with buffer
        min_required_data = max(required_periods) + 30 if required_periods else 50 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all active indicators reliably "
                                f"(min recommended: {min_required_data} based on periods: {sorted(required_periods)}). Results may contain NaNs.{RESET}")
             # Proceed anyway, but be aware of potential NaNs

        try:
            # --- Prepare DataFrame for pandas_ta (convert Decimals to float) ---
            # Operate on self.df directly as it's already a copy
            original_types = {}
            df_calc = self.df # Use alias for clarity

            cols_to_float = ['open', 'high', 'low', 'close', 'volume']
            for col in cols_to_float:
                 if col in df_calc.columns:
                     # Check first non-NaN value's type
                     first_valid_idx = df_calc[col].first_valid_index()
                     if first_valid_idx is not None:
                          col_type = type(df_calc.loc[first_valid_idx, col])
                          original_types[col] = col_type
                          if col_type == Decimal:
                               # self.logger.debug(f"Converting Decimal column '{col}' to float for TA calculation.")
                               # Apply conversion robustly, handle non-finite Decimals -> NaN
                               df_calc[col] = df_calc[col].apply(lambda x: float(x) if isinstance(x, Decimal) and x.is_finite() else np.nan)
                     else: # Column is all NaN
                          original_types[col] = None
                          # self.logger.debug(f"Column '{col}' is all NaN, skipping conversion.")

            # --- Create pandas_ta Strategy ---
            ta_strategy = ta.Strategy(
                name="SXS_Strategy",
                description="Dynamic TA indicators based on sxsBot config",
                ta=[] # Initialize empty list, append based on config
            )

            # --- Map internal keys to pandas_ta function names and parameters ---
            # Use lambda functions to dynamically get parameters from self.config at runtime
            # Ensure parameters are cast to expected types (int for length, float for std/step)
            ta_map = {
                 "atr": {"kind": "atr", "length": lambda: int(self.config.get("atr_period", DEFAULT_ATR_PERIOD))},
                 "ema_short": {"kind": "ema", "length": lambda: int(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))},
                 "ema_long": {"kind": "ema", "length": lambda: int(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))},
                 "momentum": {"kind": "mom", "length": lambda: int(self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))},
                 "cci": {"kind": "cci", "length": lambda: int(self.config.get("cci_period", DEFAULT_CCI_PERIOD))},
                 "wr": {"kind": "willr", "length": lambda: int(self.config.get("williams_r_period", DEFAULT_WILLIAMS_R_PERIOD))},
                 "mfi": {"kind": "mfi", "length": lambda: int(self.config.get("mfi_period", DEFAULT_MFI_PERIOD))},
                 "sma_10": {"kind": "sma", "length": lambda: int(self.config.get("sma_10_period", DEFAULT_SMA10_PERIOD))},
                 "rsi": {"kind": "rsi", "length": lambda: int(self.config.get("rsi_period", DEFAULT_RSI_PERIOD))},
                 "vwap": {"kind": "vwap"}, # VWAP defaults usually anchor to day
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
                 # Volume MA handled separately below
            }

            # --- Add Indicators to Strategy based on config/weights ---
            calculated_indicator_keys = set() # Track which base indicators are added

            # Always calculate ATR if possible (needed for SL/TP/BE, sizing)
            if "atr" in ta_map:
                 try:
                     params = {k: v() for k, v in ta_map["atr"].items() if k != 'kind'}
                     ta_strategy.ta.append(ta.Indicator(ta_map["atr"]["kind"], **params))
                     calculated_indicator_keys.add("atr")
                     # self.logger.debug(f"Adding ATR to TA strategy with params: {params}")
                 except Exception as e: self.logger.error(f"Error preparing ATR indicator: {e}")


            # Add other indicators based on config enable/weight
            for key, is_enabled in indicators_config.items():
                 if key == "atr": continue # Already handled
                 try: weight = float(active_weights.get(key, 0.0))
                 except (ValueError, TypeError): weight = 0.0

                 if not is_enabled or weight == 0.0: continue # Skip disabled or zero-weight

                 # Handle compound indicators or specific logic
                 if key == "ema_alignment":
                      for ema_key in ["ema_short", "ema_long"]:
                          if ema_key not in calculated_indicator_keys and ema_key in ta_map:
                               try:
                                   params = {k: v() for k, v in ta_map[ema_key].items() if k != 'kind'}
                                   ta_strategy.ta.append(ta.Indicator(ta_map[ema_key]["kind"], **params))
                                   calculated_indicator_keys.add(ema_key)
                                   # self.logger.debug(f"Adding {ema_key} to TA strategy with params: {params}")
                               except Exception as e: self.logger.error(f"Error preparing {ema_key} indicator: {e}")
                 elif key == "volume_confirmation":
                      # Handled separately after main TA run
                      pass
                 elif key in ta_map:
                      # Check if already added (e.g., multiple weights map to same TA indicator like bbands)
                      if key not in calculated_indicator_keys:
                           try:
                               indicator_def = ta_map[key]
                               params = {k: v() for k, v in indicator_def.items() if k != 'kind'}
                               ta_strategy.ta.append(ta.Indicator(indicator_def["kind"], **params))
                               calculated_indicator_keys.add(key) # Mark base key as calculated
                               # self.logger.debug(f"Adding {key} to TA strategy with params: {params}")
                           except Exception as e: self.logger.error(f"Error preparing {key} indicator: {e}")
                 elif key == "orderbook":
                       pass # Not calculated via pandas_ta
                 else:
                      self.logger.warning(f"Indicator '{key}' is enabled and weighted but has no definition in ta_map or special handling.")

            # --- Run the TA Strategy ---
            if ta_strategy.ta: # Only run if indicators were added
                 self.logger.info(f"Running pandas_ta strategy '{ta_strategy.name}' with {len(ta_strategy.ta)} indicators...")
                 try:
                     # Use df.ta.strategy() - appends columns inplace
                     df_calc.ta.strategy(ta_strategy, append=True)
                     self.logger.info("Pandas_ta strategy calculation complete.")
                 except Exception as ta_err:
                      self.logger.error(f"{NEON_RED}Error running pandas_ta strategy: {ta_err}{RESET}", exc_info=True)
                      # Continue without these indicators if strategy fails, NaNs will be handled later
            else:
                 self.logger.info("No pandas_ta indicators added to the strategy based on config.")

            # --- Calculate Volume MA Separately ---
            vol_key = "volume_confirmation"
            vol_ma_p = 0
            try: vol_ma_p = int(self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
            except (ValueError, TypeError): pass

            if indicators_config.get(vol_key, False) and float(active_weights.get(vol_key, 0.0)) > 0 and vol_ma_p > 0:
                 try:
                     vol_ma_col = f"VOL_SMA_{vol_ma_p}" # Custom column name
                     # Ensure volume column exists and is numeric (should be float now)
                     if 'volume' in df_calc.columns and pd.api.types.is_numeric_dtype(df_calc['volume']):
                          # Use fillna(0) before calculating SMA on volume
                          df_calc[vol_ma_col] = ta.sma(df_calc['volume'].fillna(0), length=vol_ma_p)
                          # self.logger.debug(f"Calculated Volume MA ({vol_ma_col}).")
                          calculated_indicator_keys.add("volume_ma") # Mark for mapping
                     else:
                          self.logger.warning(f"Volume column missing or not numeric, cannot calculate Volume MA.")
                 except Exception as vol_ma_err:
                      self.logger.error(f"Error calculating Volume MA: {vol_ma_err}")

            # --- Map Calculated Column Names ---
            # Get all columns after TA calculations
            final_df_columns = df_calc.columns.tolist()
            # Map internal names used in scoring/checks to actual DataFrame column names
            indicator_mapping = {
                # Internal Name : TA Indicator Base (used in _get_ta_col_name patterns)
                "ATR": "ATR", "EMA_Short": "EMA_Short", "EMA_Long": "EMA_Long",
                "Momentum": "Momentum", "CCI": "CCI", "Williams_R": "Williams_R", "MFI": "MFI",
                "SMA_10": "SMA_10", "RSI": "RSI", "VWAP": "VWAP",
                # PSAR generates multiple columns, map specific ones if needed by checks
                "PSAR_long": "PSAR_long", "PSAR_short": "PSAR_short",
                # StochRSI generates K and D, map both
                "StochRSI_K": "StochRSI_K", "StochRSI_D": "StochRSI_D",
                # BBands generates multiple, map components if needed by checks
                "BB_Lower": "BB_Lower", "BB_Middle": "BB_Middle", "BB_Upper": "BB_Upper",
                "Volume_MA": "Volume_MA" # Custom name used
            }
            for internal_name, ta_base_name in indicator_mapping.items():
                 # Find the column name using the helper
                 mapped_col = self._get_ta_col_name(ta_base_name, final_df_columns)
                 if mapped_col:
                     self.ta_column_names[internal_name] = mapped_col
                 # else: Warning logged by _get_ta_col_name if not found

            # --- Convert selected columns back to Decimal if original was Decimal ---
            # ATR is often critical for calculations needing precision
            cols_to_decimalize = ["ATR"]
            # Also convert price-based indicators back (BBands, PSAR, VWAP, EMAs, SMA)
            price_indicators = ["BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long", "PSAR_short",
                                "VWAP", "SMA_10", "EMA_Short", "EMA_Long"]
            cols_to_decimalize.extend(price_indicators)

            for key in cols_to_decimalize:
                col_name = self.ta_column_names.get(key)
                # Check if calculation was done, column exists, and original 'close' was Decimal
                if col_name and col_name in df_calc.columns and original_types.get('close') == Decimal:
                     try:
                         # self.logger.debug(f"Converting calculated column '{col_name}' (for '{key}') back to Decimal.")
                         # Convert float column back to Decimal, handling potential NaNs/infs
                         df_calc[col_name] = df_calc[col_name].apply(
                             lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                         )
                     except (ValueError, TypeError, InvalidOperation) as conv_err:
                          self.logger.error(f"Failed to convert TA column '{col_name}' back to Decimal: {conv_err}. Leaving as float.")

            # self.logger.debug(f"Finished indicator calculations. Final DF columns: {self.df.columns.tolist()}")
            # self.logger.debug(f"Mapped TA columns: {self.ta_column_names}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error during indicator calculation setup or execution: {e}{RESET}", exc_info=True)


    def _update_latest_indicator_values(self):
        """Updates the indicator_values dict with the latest values from the DataFrame, handling types and NaNs."""
        # Initialize with None or Decimal('NaN')
        self.indicator_values = {}

        if self.df.empty:
            self.logger.warning(f"Cannot update latest values: DataFrame empty for {self.symbol}.")
            return
        try:
            # Ensure index is sorted (should be from fetch_klines)
            if not self.df.index.is_monotonic_increasing:
                 self.logger.warning("DataFrame index not sorted, sorting before getting latest values.")
                 self.df.sort_index(inplace=True)

            # Use the last row
            latest = self.df.iloc[-1]
            latest_timestamp = self.df.index[-1]
            self.indicator_values["Timestamp"] = latest_timestamp # Store timestamp as well

        except IndexError:
            self.logger.error(f"Error accessing latest row for {self.symbol}. DataFrame might be empty after calculations.")
            return
        except Exception as e:
             self.logger.error(f"Unexpected error getting latest row for {self.symbol}: {e}")
             return

        # --- Process Base OHLCV (should be Decimal from fetch_klines) ---
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            key_name = base_col.capitalize()
            value = None # Default to None
            if base_col in latest.index:
                 raw_value = latest[base_col]
                 if isinstance(raw_value, Decimal):
                      value = raw_value if raw_value.is_finite() else Decimal('NaN')
                 elif pd.notna(raw_value): # Handle case where it might be float after fallback
                      try:
                           dec_val = Decimal(str(raw_value))
                           value = dec_val if dec_val.is_finite() else Decimal('NaN')
                      except: value = Decimal('NaN')
                 # else: value remains None if raw_value was NaN/None
            self.indicator_values[key_name] = value if value is not None else Decimal('NaN')


        # --- Process TA indicators using mapped column names ---
        for key, col_name in self.ta_column_names.items():
            value = None # Default to None
            raw_value = None
            target_type = Decimal if key in ["ATR", "BB_Lower", "BB_Middle", "BB_Upper", "PSAR_long", "PSAR_short", "VWAP", "SMA_10", "EMA_Short", "EMA_Long"] else float

            if col_name and col_name in latest.index:
                raw_value = latest[col_name]
                # Check if the value is valid (not None, not pd.NA)
                if pd.notna(raw_value):
                    try:
                        # Attempt conversion to target type
                        if target_type == Decimal:
                             converted_value = Decimal(str(raw_value))
                             value = converted_value if converted_value.is_finite() else (Decimal('NaN') if target_type == Decimal else np.nan)
                        else: # Target is float
                             converted_value = float(raw_value)
                             value = converted_value if np.isfinite(converted_value) else np.nan

                    except (ValueError, TypeError, InvalidOperation):
                        # self.logger.debug(f"Could not convert TA value {key} ('{col_name}': {raw_value}) to {target_type}. Storing NaN/None.")
                        value = Decimal('NaN') if target_type == Decimal else np.nan
                # else: raw_value is NaN/None, value remains None

            # Store the processed value or appropriate NaN type
            self.indicator_values[key] = value if value is not None else (Decimal('NaN') if target_type == Decimal else np.nan)

        # --- Log Summary (formatted) ---
        log_vals = {}
        price_prec = self.get_price_precision()
        amount_prec = self.get_amount_precision_places()

        # Define which keys represent prices vs other values for formatting
        price_keys = ['Open','High','Low','Close','ATR','BB_Lower','BB_Middle','BB_Upper',
                      'PSAR_long','PSAR_short','VWAP','SMA_10','EMA_Short','EMA_Long']
        amount_keys = ['Volume', 'Volume_MA'] # Volume MA is float, format like amount

        for k, v in self.indicator_values.items():
            if k == "Timestamp": log_vals[k] = str(v); continue # Handle timestamp separately

            formatted_val = "NaN" # Default for None or NaN
            if isinstance(v, Decimal) and v.is_finite():
                 prec = price_prec if k in price_keys else amount_prec if k in amount_keys else 8 # Default precision for other Decimals
                 try: formatted_val = f"{v:.{prec}f}"
                 except ValueError: formatted_val = str(v) # Fallback if precision invalid
            elif isinstance(v, float) and np.isfinite(v):
                 prec = amount_prec if k in amount_keys else 5 # Default precision for floats
                 try: formatted_val = f"{v:.{prec}f}"
                 except ValueError: formatted_val = str(v)
            elif isinstance(v, int): # Handle integers directly
                 formatted_val = str(v)

            if formatted_val != "NaN": # Only include non-NaN values in log
                 log_vals[k] = formatted_val

        if log_vals:
             # Sort keys for consistent log output
             sorted_keys = sorted(log_vals.keys(), key=lambda x: (x not in price_keys and x not in amount_keys, x)) # Prices first, then amounts, then others alphabetically
             sorted_log_vals = {k: log_vals[k] for k in sorted_keys}
             self.logger.debug(f"Latest values updated ({self.symbol}): {json.dumps(sorted_log_vals)}")
        else:
             self.logger.warning(f"No valid latest indicator values could be determined for {self.symbol}.")


    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculates Fibonacci retracement levels using Decimal precision based on high/low over a window."""
        window = window or int(self.config.get("fibonacci_period", DEFAULT_FIB_PERIOD))
        self.fib_levels_data = {} # Clear previous levels

        if self.df.empty or 'high' not in self.df.columns or 'low' not in self.df.columns:
             self.logger.debug(f"Fibonacci calc skipped: DataFrame empty or missing high/low columns.")
             return {}
        if len(self.df) < window:
            self.logger.debug(f"Not enough data ({len(self.df)}) for Fibonacci ({window} bars) on {self.symbol}.")
            return {}

        df_slice = self.df.tail(window)
        try:
            # Extract high/low series (should be Decimal)
            high_series = df_slice["high"].dropna()
            low_series = df_slice["low"].dropna()

            if high_series.empty or low_series.empty:
                 self.logger.warning(f"No valid high/low data points found in the last {window} bars for Fibonacci.")
                 return {}

            high_price = high_series.max()
            low_price = low_series.min()

            # Ensure we got valid Decimal prices
            if not isinstance(high_price, Decimal) or not high_price.is_finite() or \
               not isinstance(low_price, Decimal) or not low_price.is_finite():
                self.logger.warning(f"Could not find valid finite high/low Decimal prices for Fibonacci (Window: {window}). High: {high_price}, Low: {low_price}")
                return {}

            # --- Calculate Levels using Decimal ---
            diff = high_price - low_price
            levels = {}
            price_precision = self.get_price_precision()
            # Use min_tick for quantization if available and valid, else use precision
            min_tick = self.get_min_tick_size()
            quantizer = min_tick if min_tick.is_finite() and min_tick > 0 else Decimal('1e-' + str(price_precision))

            if diff <= 0: # Handle zero or negative range (high <= low)
                if diff == 0:
                    self.logger.debug(f"Fibonacci range is zero (High=Low={high_price}). Setting all levels to this price.")
                    level_price_quantized = high_price.quantize(quantizer, rounding=ROUND_DOWN)
                    levels = {f"Fib_{level_pct * 100:.1f}%": level_price_quantized for level_pct in FIB_LEVELS}
                else: # Should not happen if max/min logic is correct
                    self.logger.error(f"Fibonacci Calc Error: Max high ({high_price}) < Min low ({low_price}). Check data.")
                    return {}
            else:
                # Calculate normal levels
                for level_pct_str in map(str, FIB_LEVELS):
                    level_pct = Decimal(level_pct_str)
                    level_name = f"Fib_{level_pct * 100:.1f}%"
                    # Retracement Level price = High - (Range * Percentage)
                    level_price_raw = high_price - (diff * level_pct)
                    # Quantize level price based on market precision/tick (round down for levels derived from high)
                    levels[level_name] = level_price_raw.quantize(quantizer, rounding=ROUND_DOWN)

            self.fib_levels_data = levels
            log_levels = {k: f"{v:.{price_precision}f}" for k, v in levels.items()} # Format for logging
            self.logger.debug(f"Calculated Fibonacci levels (Window: {window}, High: {high_price:.{price_precision}f}, Low: {low_price:.{price_precision}f}): {log_levels}")
            return levels

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Fibonacci error: Missing column '{e}'. Ensure OHLCV data is present.{RESET}")
            return {}
        except (ValueError, TypeError, InvalidOperation) as e:
             self.logger.error(f"{NEON_RED}Fibonacci error: Invalid data type or operation during calculation. {e}{RESET}")
             return {}
        except Exception as e:
            self.logger.error(f"{NEON_RED}Unexpected Fibonacci calculation error: {e}{RESET}", exc_info=True)
            return {}

    # --- Precision and Limit Helpers ---

    def get_price_precision(self) -> int:
        """Determines price precision (decimal places) from market info. More robust checks."""
        # Cache the result for efficiency within the cycle
        if hasattr(self, '_cached_price_precision'):
            return self._cached_price_precision

        precision = None
        source = "Unknown"
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')

            if price_precision_val is not None:
                # Case 1: Integer precision (decimal places)
                if isinstance(price_precision_val, int) and price_precision_val >= 0:
                    precision, source = price_precision_val, "market.precision.price (int)"
                # Case 2: Float/String precision (tick size)
                else:
                    try:
                        tick_size = Decimal(str(price_precision_val))
                        if tick_size.is_finite() and tick_size > 0:
                            precision = abs(tick_size.normalize().as_tuple().exponent)
                            source = f"market.precision.price (tick: {tick_size})"
                        # else: Invalid tick size, continue to fallbacks
                    except (TypeError, ValueError, InvalidOperation): pass # Ignore parsing error

            # Fallback 1: Infer from limits.price.min (if it looks like a tick size)
            if precision is None:
                min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
                if min_price_val is not None:
                    try:
                        min_price_tick = Decimal(str(min_price_val))
                        if min_price_tick.is_finite() and 0 < min_price_tick < Decimal('1'): # Heuristic
                            precision = abs(min_price_tick.normalize().as_tuple().exponent)
                            source = f"market.limits.price.min ({min_price_tick})"
                    except (TypeError, ValueError, InvalidOperation): pass

            # Fallback 2: Infer from last close price (least reliable)
            if precision is None:
                last_close = self.indicator_values.get("Close") # Assumes updated
                if isinstance(last_close, Decimal) and last_close.is_finite() and last_close > 0:
                    try:
                        p = abs(last_close.normalize().as_tuple().exponent)
                        if 0 <= p <= 12: # Reasonable range for crypto price decimal places
                            precision = p
                            source = f"Last Close Price ({last_close})"
                    except Exception: pass

        except Exception as e:
            self.logger.warning(f"Error determining price precision for {self.symbol}: {e}. Falling back.")

        # --- Final Default Fallback ---
        if precision is None:
            default_precision = 4 # Common default
            precision = default_precision
            source = f"Default ({default_precision})"
            self.logger.warning(f"Could not determine price precision for {self.symbol}. Using default: {precision}.")

        # Cache and return
        self._cached_price_precision = precision
        # self.logger.debug(f"Price precision for {self.symbol}: {precision} (Source: {source})")
        return precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        if hasattr(self, '_cached_min_tick_size'):
            return self._cached_min_tick_size

        tick_size = None
        source = "Unknown"
        try:
            # 1. Try precision.price directly as tick size
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None and not isinstance(price_precision_val, int):
                try:
                    tick = Decimal(str(price_precision_val))
                    if tick.is_finite() and tick > 0:
                        tick_size, source = tick, "market.precision.price (value)"
                except (TypeError, ValueError, InvalidOperation): pass

            # 2. Fallback: Try limits.price.min
            if tick_size is None:
                min_price_val = self.market_info.get('limits', {}).get('price', {}).get('min')
                if min_price_val is not None:
                    try:
                        min_tick = Decimal(str(min_price_val))
                        if min_tick.is_finite() and min_tick > 0:
                            tick_size, source = min_tick, "market.limits.price.min"
                    except (TypeError, ValueError, InvalidOperation): pass

            # 3. Fallback: Calculate from integer precision.price
            if tick_size is None and price_precision_val is not None and isinstance(price_precision_val, int) and price_precision_val >= 0:
                 tick_size = Decimal('1e-' + str(price_precision_val))
                 source = f"market.precision.price (int: {price_precision_val})"

        except Exception as e:
            self.logger.warning(f"Could not determine min tick size for {self.symbol} from market info: {e}.")

        # --- Final Fallback: Calculate from derived decimal places ---
        if tick_size is None:
            price_precision_places = self.get_price_precision() # Call the robust getter
            tick_size = Decimal('1e-' + str(price_precision_places))
            source = f"Derived Precision ({price_precision_places})"
            self.logger.warning(f"Using fallback tick size based on derived precision for {self.symbol}: {tick_size}")

        if not isinstance(tick_size, Decimal) or not tick_size.is_finite() or tick_size <= 0:
             fallback_tick = Decimal('0.00000001') # Smallest possible reasonable fallback
             self.logger.error(f"Failed to determine a valid tick size! Using emergency fallback: {fallback_tick}")
             tick_size = fallback_tick
             source = "Emergency Fallback"

        self._cached_min_tick_size = tick_size
        # self.logger.debug(f"Min Tick Size for {self.symbol}: {tick_size} (Source: {source})")
        return tick_size

    def get_amount_precision_places(self) -> int:
        """Determines amount precision (decimal places) from market info."""
        if hasattr(self, '_cached_amount_precision'):
            return self._cached_amount_precision

        precision = None
        source = "Unknown"
        try:
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')

            if amount_precision_val is not None:
                # Case 1: Integer precision
                if isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                    precision, source = amount_precision_val, "market.precision.amount (int)"
                # Case 2: Float/String (step size) -> infer places
                else:
                     try:
                          step_size = Decimal(str(amount_precision_val))
                          if step_size.is_finite() and step_size > 0:
                               precision = abs(step_size.normalize().as_tuple().exponent)
                               source = f"market.precision.amount (step: {step_size})"
                     except (TypeError, ValueError, InvalidOperation): pass

            # Fallback 1: Infer from limits.amount.min (if step size)
            if precision is None:
                min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
                if min_amount_val is not None:
                    try:
                        min_amount_step = Decimal(str(min_amount_val))
                        # Check if it looks like a step size (fractional or < 1)
                        if min_amount_step.is_finite() and 0 < min_amount_step <= Decimal('1'):
                           if min_amount_step < 1 or '.' in str(min_amount_val):
                               precision = abs(min_amount_step.normalize().as_tuple().exponent)
                               source = f"market.limits.amount.min (step: {min_amount_step})"
                        # Handle integer min amount (e.g., 1 means 0 decimal places)
                        elif min_amount_step.is_finite() and min_amount_step >= 1 and min_amount_step % 1 == 0:
                           precision = 0
                           source = f"market.limits.amount.min (int: {min_amount_step})"
                    except (TypeError, ValueError, InvalidOperation): pass

        except Exception as e:
            self.logger.warning(f"Error determining amount precision for {self.symbol}: {e}.")

        # --- Final Default Fallback ---
        if precision is None:
            default_precision = 8 # Common for crypto base amounts
            precision = default_precision
            source = f"Default ({default_precision})"
            self.logger.warning(f"Could not determine amount precision for {self.symbol}. Using default: {precision}.")

        self._cached_amount_precision = precision
        # self.logger.debug(f"Amount precision for {self.symbol}: {precision} (Source: {source})")
        return precision

    def get_min_amount_step(self) -> Decimal:
        """Gets the minimum amount increment (step size) from market info as Decimal."""
        if hasattr(self, '_cached_min_amount_step'):
            return self._cached_min_amount_step

        step_size = None
        source = "Unknown"
        try:
            # 1. Try precision.amount directly as step size
            precision_info = self.market_info.get('precision', {})
            amount_precision_val = precision_info.get('amount')
            if amount_precision_val is not None and not isinstance(amount_precision_val, int):
                try:
                    step = Decimal(str(amount_precision_val))
                    if step.is_finite() and step > 0:
                        step_size, source = step, "market.precision.amount (value)"
                except (TypeError, ValueError, InvalidOperation): pass

            # 2. Fallback: Try limits.amount.min
            if step_size is None:
                min_amount_val = self.market_info.get('limits', {}).get('amount', {}).get('min')
                if min_amount_val is not None:
                    try:
                        min_step = Decimal(str(min_amount_val))
                        if min_step.is_finite() and min_step > 0:
                            step_size, source = min_step, "market.limits.amount.min"
                    except (TypeError, ValueError, InvalidOperation): pass

            # 3. Fallback: Calculate from integer precision.amount
            if step_size is None and amount_precision_val is not None and isinstance(amount_precision_val, int) and amount_precision_val >= 0:
                 step_size = Decimal('1e-' + str(amount_precision_val))
                 source = f"market.precision.amount (int: {amount_precision_val})"

        except Exception as e:
            self.logger.warning(f"Could not determine min amount step for {self.symbol} from market info: {e}.")

        # --- Final Fallback: Calculate from derived decimal places ---
        if step_size is None:
            amount_precision_places = self.get_amount_precision_places()
            step_size = Decimal('1e-' + str(amount_precision_places))
            source = f"Derived Precision ({amount_precision_places})"
            self.logger.warning(f"Using fallback amount step based on derived precision for {self.symbol}: {step_size}")

        if not isinstance(step_size, Decimal) or not step_size.is_finite() or step_size <= 0:
             fallback_step = Decimal('0.00000001') # Smallest possible reasonable fallback
             self.logger.error(f"Failed to determine a valid amount step size! Using emergency fallback: {fallback_step}")
             step_size = fallback_step
             source = "Emergency Fallback"

        self._cached_min_amount_step = step_size
        # self.logger.debug(f"Min Amount Step for {self.symbol}: {step_size} (Source: {source})")
        return step_size


    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            # self.logger.debug(f"Fibonacci levels not calculated or empty for {self.symbol}.")
            return []
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level_price in self.fib_levels_data.items():
                # Ensure level_price is a valid Decimal before calculating distance
                if isinstance(level_price, Decimal) and level_price.is_finite() and level_price > 0:
                    distance = abs(current_price - level_price)
                    level_distances.append({'name': name, 'level': level_price, 'distance': distance})
                else:
                    # self.logger.debug(f"Invalid or non-finite Fib level skipped: {name}={level_price}.")
                    pass

            if not level_distances:
                self.logger.debug("No valid Fibonacci levels found to compare distance.")
                return []

            # Sort by distance (Decimal comparison works correctly)
            level_distances.sort(key=lambda x: x['distance'])

            # Return the name and level price for the nearest N levels
            return [(item['name'], item['level']) for item in level_distances[:num_levels]]

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- Signal Generation and Indicator Checks ---

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates final trading signal (BUY/SELL/HOLD) based on weighted score of enabled indicators.
        Uses Decimal for score aggregation for precision. Returns 'BUY', 'SELL', or 'HOLD'.
        """
        final_score = Decimal("0.0")
        total_weight = Decimal("0.0")
        active_indicator_count = 0
        contributing_indicators = {} # Store scores of indicators that contributed

        # --- Basic Validation ---
        if not self.indicator_values:
            self.logger.warning("Signal Generation Skipped: Indicator values dictionary is empty."); return "HOLD"
        if not isinstance(current_price, Decimal) or not current_price.is_finite() or current_price <= 0:
            self.logger.warning(f"Signal Generation Skipped: Invalid current price ({current_price})."); return "HOLD"
        if not self.weights:
            self.logger.warning("Signal Generation Warning: Active weight set is missing or empty. Score will be zero."); return "HOLD"

        # --- Iterate through indicators listed in the config ---
        available_check_methods = {m.replace('_check_', '') for m in dir(self) if m.startswith('_check_')}

        for indicator_key, is_enabled in self.config.get("indicators", {}).items():
            if not is_enabled: continue # Skip disabled indicators

            # Check if a check method exists for this indicator
            if indicator_key not in available_check_methods:
                 if float(self.weights.get(indicator_key, 0.0)) > 0: # Warn only if weighted
                     self.logger.warning(f"No check method '_check_{indicator_key}' found for enabled and weighted indicator.")
                 continue

            # Get weight, ensuring it's a valid Decimal >= 0
            weight_val = self.weights.get(indicator_key)
            if weight_val is None: continue # Skip if no weight defined

            try:
                weight = Decimal(str(weight_val))
                if not weight.is_finite() or weight < 0: raise ValueError("Weight not finite or negative")
                if weight == 0: continue # Skip zero weight efficiently
            except (ValueError, TypeError, InvalidOperation):
                self.logger.warning(f"Invalid weight '{weight_val}' for indicator '{indicator_key}'. Skipping."); continue

            # Find and execute the corresponding check method
            check_method_name = f"_check_{indicator_key}"
            score_float = np.nan # Default score

            try:
                method = getattr(self, check_method_name)
                # Special handling for orderbook which needs extra data
                if indicator_key == "orderbook":
                    if orderbook_data:
                        # Pass Decimal price and orderbook dict
                        score_float = method(orderbook_data, current_price)
                    # else: No orderbook data, score remains NaN
                else:
                    score_float = method() # Call the check method

            except Exception as e:
                self.logger.error(f"Error executing indicator check '{check_method_name}' for {self.symbol}: {e}", exc_info=True)
                score_float = np.nan # Ensure score is NaN on error

            # --- Aggregate Score (using Decimal) ---
            if pd.notna(score_float) and np.isfinite(score_float):
                try:
                    # Convert float score [-1, 1] to Decimal
                    score_dec = Decimal(str(score_float))
                    # Clamp score just in case a check method returns out of bounds
                    clamped_score = max(Decimal("-1.0"), min(Decimal("1.0"), score_dec))

                    final_score += clamped_score * weight
                    total_weight += weight # Sum weights of indicators that provided a valid score
                    active_indicator_count += 1
                    contributing_indicators[indicator_key] = f"{clamped_score:.3f}" # Store clamped score string

                except (ValueError, TypeError, InvalidOperation) as calc_err:
                    self.logger.error(f"Error processing score for {indicator_key} ({score_float}): {calc_err}")
            # else: Score was NaN or infinite from the check method, not counted

        # --- Determine Final Signal ---
        final_signal = "HOLD"
        # Check if any indicators contributed meaningfully
        if total_weight <= Decimal('1e-9'): # Use tolerance for float/Decimal comparison
            # self.logger.debug(f"No indicators contributed valid scores or weights for {self.symbol} (Total Weight: {total_weight:.4f}). Defaulting to HOLD.")
            pass # Signal remains HOLD
        else:
            # Get threshold from config, validate, use Decimal
            try:
                threshold_str = self.config.get("signal_score_threshold", "1.5")
                threshold = Decimal(str(threshold_str))
                if not threshold.is_finite() or threshold <= 0: raise ValueError("Threshold must be positive and finite")
            except (ValueError, TypeError, InvalidOperation):
                default_threshold = Decimal(str(default_config["signal_score_threshold"])) # Use default
                self.logger.warning(f"Invalid signal_score_threshold '{threshold_str}'. Using default {default_threshold}.")
                threshold = default_threshold

            # Compare final score (raw weighted score) to threshold
            if final_score >= threshold:
                final_signal = "BUY"
            elif final_score <= -threshold:
                final_signal = "SELL"
            # else: final_signal remains "HOLD"

        # --- Log Summary ---
        price_prec = self.get_price_precision()
        sig_color = NEON_GREEN if final_signal == "BUY" else NEON_RED if final_signal == "SELL" else NEON_YELLOW
        log_msg = (
            f"Signal ({self.symbol} @ {current_price:.{price_prec}f}): "
            f"Set='{self.active_weight_set_name}', ActInd={active_indicator_count}, "
            f"TotW={total_weight:.2f}, Score={final_score:.4f} (Thr: +/-{threshold:.2f}) "
            f"==> {sig_color}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        # Log contributing indicator scores only if logger level is DEBUG
        if self.logger.isEnabledFor(logging.DEBUG) and contributing_indicators:
             # Sort scores by indicator key for consistent logging
             sorted_scores = dict(sorted(contributing_indicators.items()))
             self.logger.debug(f"  Contributing Scores ({self.symbol}): {json.dumps(sorted_scores)}")

        return final_signal

    # --- Indicator Check Methods (return float score -1.0 to 1.0 or np.nan) ---
    # Ensure methods fetch values from self.indicator_values, handle potential NaN/None/Decimal/float types

    def _check_ema_alignment(self) -> float:
        """Checks if EMAs are aligned and price confirms trend. Returns float score [-1.0, 1.0] or np.nan."""
        ema_s = self.indicator_values.get("EMA_Short") # Expect Decimal or NaN
        ema_l = self.indicator_values.get("EMA_Long")  # Expect Decimal or NaN
        close = self.indicator_values.get("Close")     # Expect Decimal or NaN

        # Use helper to check if all are valid finite Decimals
        def is_valid_decimal(val):
            return isinstance(val, Decimal) and val.is_finite()

        if not all(map(is_valid_decimal, [ema_s, ema_l, close])):
            return np.nan

        # Check relative positions
        try:
            price_above_short = close > ema_s
            price_above_long = close > ema_l
            short_above_long = ema_s > ema_l

            if price_above_short and short_above_long: return 1.0   # Strong Bullish: Close > Short > Long
            if not price_above_long and not short_above_long: return -1.0 # Strong Bearish: Close < Short < Long (assuming short < long if not short > long)

            # Weaker signals or disagreement
            if short_above_long: # EMAs bullish
                return 0.3 if price_above_short else -0.2 # Price confirms / Price disagrees
            else: # EMAs bearish (short_below_long)
                return -0.3 if not price_above_long else 0.2 # Price confirms / Price disagrees

        except TypeError: # Handles potential comparison errors if Decimal conversion failed upstream
            self.logger.warning("Type error during EMA alignment check.", exc_info=True)
            return np.nan

        return 0.0 # Should be unreachable if logic is sound, maybe if EMAs are exactly equal

    def _check_momentum(self) -> float:
        """Scores based on Momentum indicator value. Returns float score [-1.0, 1.0] or np.nan."""
        momentum = self.indicator_values.get("Momentum") # Expect float or np.nan
        if not isinstance(momentum, (float, int)) or not np.isfinite(momentum): return np.nan

        # Normalize or use thresholds? Thresholds are simpler but less adaptive.
        # Using simple thresholds based on typical MOM behavior around zero.
        # These might need tuning based on asset volatility and timeframe.
        strong_pos = 0.5  # Example threshold for strong positive momentum
        weak_pos = 0.1
        strong_neg = -0.5 # Example threshold for strong negative momentum
        weak_neg = -0.1

        if momentum >= strong_pos * 2: return 1.0
        if momentum >= strong_pos: return 0.7
        if momentum >= weak_pos: return 0.3
        if momentum <= strong_neg * 2: return -1.0
        if momentum <= strong_neg: return -0.7
        if momentum <= weak_neg: return -0.3

        return 0.0 # Between weak thresholds

    def _check_volume_confirmation(self) -> float:
        """Scores based on current volume relative to its moving average. Returns float score [-1.0, 1.0] or np.nan."""
        current_volume = self.indicator_values.get("Volume") # Expect Decimal or NaN
        volume_ma = self.indicator_values.get("Volume_MA") # Expect float or np.nan

        # Validate inputs
        if not isinstance(current_volume, Decimal) or not current_volume.is_finite() or current_volume < 0: return np.nan
        if not isinstance(volume_ma, (float, int)) or not np.isfinite(volume_ma) or volume_ma <= 0: return np.nan # MA should be positive

        try:
            volume_ma_dec = Decimal(str(volume_ma)) # Convert MA float to Decimal for comparison
            multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))
            if multiplier <= 0: multiplier = Decimal('1.5') # Ensure positive

            # Avoid division by zero if MA is effectively zero
            if volume_ma_dec < Decimal('1e-12'): return 0.0 # Treat as neutral if MA is tiny

            ratio = current_volume / volume_ma_dec

            # Score based on ratio
            if ratio >= multiplier * Decimal('1.5'): return 1.0 # Very high volume confirmation
            if ratio >= multiplier: return 0.7                 # High volume confirmation
            if ratio <= (Decimal('1') / multiplier): return -0.4 # Unusually low volume (potential lack of interest/trap)

            # Scale between low volume and high volume confirmation? Maybe too complex.
            # Let's return neutral if within normal range.
            return 0.0

        except (InvalidOperation, ZeroDivisionError) as e:
             self.logger.warning(f"Error during volume confirmation calculation: {e}")
             return np.nan

    def _check_stoch_rsi(self) -> float:
        """Scores based on Stochastic RSI K and D values and thresholds. Returns float score [-1.0, 1.0] or np.nan."""
        k = self.indicator_values.get("StochRSI_K") # Expect float or np.nan
        d = self.indicator_values.get("StochRSI_D") # Expect float or np.nan

        if not isinstance(k, (float, int)) or not np.isfinite(k) or \
           not isinstance(d, (float, int)) or not np.isfinite(d):
            return np.nan

        # Get thresholds from config, ensure they are floats
        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75))
            if not (0 < oversold < 100 and 0 < overbought < 100 and oversold < overbought):
                raise ValueError("Invalid StochRSI thresholds")
        except (ValueError, TypeError):
             oversold, overbought = 25.0, 75.0 # Fallback to defaults
             self.logger.warning("Invalid StochRSI thresholds in config, using defaults (25/75).")

        score = 0.0
        # 1. Extreme conditions (strongest signal)
        if k < oversold and d < oversold: score = 1.0 # Strong Buy (Oversold)
        elif k > overbought and d > overbought: score = -1.0 # Strong Sell (Overbought)

        # 2. Consider crosses (K crossing D) - adds confirmation or reversal indication
        # Use a small tolerance for crossing to avoid noise
        cross_tolerance = 1.0
        if k > d + cross_tolerance and d < overbought: # Bullish cross (K crosses above D), ensure not already deep in overbought
             score = max(score, 0.7) # Give bullish cross a strong positive score
        elif d > k + cross_tolerance and k > oversold: # Bearish cross (D crosses above K), ensure not already deep in oversold
             score = min(score, -0.7) # Give bearish cross a strong negative score

        # 3. General position relative to 50 (weaker bias)
        mid_point = 50.0
        if k > mid_point and d > mid_point and score >= 0: # Both bullish, reinforce positive score slightly
             score = max(score, 0.2)
        elif k < mid_point and d < mid_point and score <= 0: # Both bearish, reinforce negative score slightly
             score = min(score, -0.2)
        # Could add divergence checks here (more complex)

        # Clamp final score just in case intermediate logic pushed it over
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        """Scores based on RSI value relative to standard overbought/oversold levels. Returns float score [-1.0, 1.0] or np.nan."""
        rsi = self.indicator_values.get("RSI") # Expect float or np.nan
        if not isinstance(rsi, (float, int)) or not np.isfinite(rsi): return np.nan

        # Use graded score based on standard levels (30/70) and extremes (20/80)
        if rsi >= 80: return -1.0 # Extreme Overbought
        if rsi >= 70: return -0.7 # Overbought
        if rsi > 60: return -0.3  # Approaching Overbought
        if rsi <= 20: return 1.0 # Extreme Oversold
        if rsi <= 30: return 0.7 # Oversold
        if rsi < 40: return 0.3  # Approaching Oversold

        # Neutral zone (40-60)
        if 40 <= rsi <= 60: return 0.0

        # Should cover all valid RSI ranges (0-100), default just in case
        return 0.0

    def _check_cci(self) -> float:
        """Scores based on CCI value relative to standard +/-100 levels. Returns float score [-1.0, 1.0] or np.nan."""
        cci = self.indicator_values.get("CCI") # Expect float or np.nan
        if not isinstance(cci, (float, int)) or not np.isfinite(cci): return np.nan

        # Standard levels: +100 Overbought (Sell signal), -100 Oversold (Buy signal)
        # Use extremes (+/-200) for stronger signals
        if cci >= 200: return -1.0 # Extreme Sell signal
        if cci >= 100: return -0.7 # Standard Sell signal
        if cci > 0: return -0.2   # Mild bearish momentum (above zero line)
        if cci <= -200: return 1.0 # Extreme Buy signal
        if cci <= -100: return 0.7 # Standard Buy signal
        if cci < 0: return 0.2   # Mild bullish momentum (below zero line)

        return 0.0 # Exactly zero

    def _check_wr(self) -> float:
        """Scores based on Williams %R value relative to standard -20/-80 levels. Returns float score [-1.0, 1.0] or np.nan."""
        wr = self.indicator_values.get("Williams_R") # Expect float (range -100 to 0) or np.nan
        if not isinstance(wr, (float, int)) or not np.isfinite(wr): return np.nan

        # Standard levels: -20 Overbought (Sell signal), -80 Oversold (Buy signal)
        # Use extremes (-10 / -90)
        if wr >= -10: return -1.0 # Extreme Overbought
        if wr >= -20: return -0.7 # Standard Overbought
        if wr > -50: return -0.2  # In upper half (more bearish bias)
        if wr <= -90: return 1.0 # Extreme Oversold
        if wr <= -80: return 0.7 # Standard Oversold
        if wr < -50: return 0.2  # In lower half (more bullish bias)

        return 0.0 # Exactly -50

    def _check_psar(self) -> float:
        """Scores based on Parabolic SAR position relative to price. Returns float score [-1.0, 1.0] or np.nan."""
        psar_l = self.indicator_values.get("PSAR_long")  # Expect Decimal or NaN
        psar_s = self.indicator_values.get("PSAR_short") # Expect Decimal or NaN
        # Close price not strictly needed here, PSARl/s indicates trend direction

        l_active = isinstance(psar_l, Decimal) and psar_l.is_finite()
        s_active = isinstance(psar_s, Decimal) and psar_s.is_finite()

        if l_active and not s_active:
            return 1.0  # Uptrend signaled: PSAR Long is active (plots below price)
        elif s_active and not l_active:
            return -1.0 # Downtrend signaled: PSAR Short is active (plots above price)
        elif not l_active and not s_active:
            # self.logger.debug("PSAR check: Neither long nor short PSAR value is active/valid.")
            return np.nan # Indeterminate or insufficient data
        else:
             # Both active/valid shouldn't happen with standard PSAR calculation
             self.logger.warning(f"PSAR check encountered unusual state: Both Long ({psar_l}) and Short ({psar_s}) seem active/valid. Returning neutral.")
             return 0.0

    def _check_sma_10(self) -> float:
        """Scores based on price position relative to the 10-period SMA. Returns float score [-1.0, 1.0] or np.nan."""
        sma = self.indicator_values.get("SMA_10")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close") # Expect Decimal or NaN

        if not isinstance(sma, Decimal) or not sma.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        try:
            if close > sma: return 0.6  # Price above SMA (Bullish bias)
            if close < sma: return -0.6 # Price below SMA (Bearish bias)
        except TypeError: return np.nan # Comparison failed

        return 0.0 # Price exactly on SMA

    def _check_vwap(self) -> float:
        """Scores based on price position relative to VWAP. Returns float score [-1.0, 1.0] or np.nan."""
        vwap = self.indicator_values.get("VWAP")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close") # Expect Decimal or NaN

        if not isinstance(vwap, Decimal) or not vwap.is_finite() or \
           not isinstance(close, Decimal) or not close.is_finite():
           return np.nan

        # VWAP acts as a dynamic support/resistance or mean for the session (often daily)
        try:
            if close > vwap: return 0.7  # Price above VWAP (Bullish bias for session)
            if close < vwap: return -0.7 # Price below VWAP (Bearish bias for session)
        except TypeError: return np.nan

        return 0.0 # Price exactly on VWAP

    def _check_mfi(self) -> float:
        """Scores based on Money Flow Index relative to standard 20/80 levels. Returns float score [-1.0, 1.0] or np.nan."""
        mfi = self.indicator_values.get("MFI") # Expect float or np.nan
        if not isinstance(mfi, (float, int)) or not np.isfinite(mfi): return np.nan

        # Standard levels: 80 Overbought, 20 Oversold. Use extremes 90/10.
        if mfi >= 90: return -1.0 # Extreme Overbought
        if mfi >= 80: return -0.7 # Overbought
        if mfi > 65: return -0.3  # Approaching Overbought / Upper range
        if mfi <= 10: return 1.0 # Extreme Oversold
        if mfi <= 20: return 0.7 # Oversold
        if mfi < 35: return 0.3  # Approaching Oversold / Lower range

        # Neutral zone (35-65)
        if 35 <= mfi <= 65: return 0.0

        return 0.0 # Default fallback

    def _check_bollinger_bands(self) -> float:
        """Scores based on price position relative to Bollinger Bands. Returns float score [-1.0, 1.0] or np.nan."""
        bbl = self.indicator_values.get("BB_Lower")   # Expect Decimal or NaN
        bbm = self.indicator_values.get("BB_Middle")  # Expect Decimal or NaN
        bbu = self.indicator_values.get("BB_Upper")   # Expect Decimal or NaN
        close = self.indicator_values.get("Close")    # Expect Decimal or NaN

        # Validate inputs are finite Decimals
        def is_valid_decimal(val): return isinstance(val, Decimal) and val.is_finite()
        if not all(map(is_valid_decimal, [bbl, bbm, bbu, close])):
            return np.nan

        # Check for valid band range (Upper > Lower)
        band_width = bbu - bbl
        if band_width <= 0:
            # self.logger.debug("BBands check skipped: Upper band <= lower band.")
            return np.nan # Invalid bands

        try:
            # --- Scoring Logic ---
            # 1. Touching or exceeding bands (strong reversal/fade signal)
            if close <= bbl: return 1.0 # Strong Buy signal (touch/below lower band)
            if close >= bbu: return -1.0 # Strong Sell signal (touch/above upper band)

            # 2. Position relative to middle band (weaker trend/mean reversion bias)
            if close > bbm: # Above middle band -> Bearish bias (expect reversion to mean)
                 # Scale score from 0 (at BBM) to -0.7 (near BBU)
                 position_in_upper_band = (close - bbm) / (bbu - bbm) # Range 0 to 1
                 score = float(position_in_upper_band) * -0.7
                 return max(-0.7, score) # Limit score
            else: # Below middle band -> Bullish bias (expect reversion to mean)
                 # Scale score from 0 (at BBM) to +0.7 (near BBL)
                 position_in_lower_band = (bbm - close) / (bbm - bbl) # Range 0 to 1
                 score = float(position_in_lower_band) * 0.7
                 return min(0.7, score) # Limit score

        except (TypeError, ZeroDivisionError): # Handle comparison errors or zero band width
             return np.nan

        return 0.0 # Should only be reached if close == bbm

    def _check_orderbook(self, orderbook_data: Optional[Dict], current_price: Decimal) -> float:
        """Analyzes Order Book Imbalance (OBI). Returns float score [-1.0, 1.0] or np.nan."""
        if not orderbook_data or not isinstance(orderbook_data.get('bids'), list) or not isinstance(orderbook_data.get('asks'), list):
            # self.logger.debug("Orderbook check skipped: Invalid or missing data.")
            return np.nan

        bids = orderbook_data['bids'] # List of [Decimal(price), Decimal(amount)], sorted high to low
        asks = orderbook_data['asks'] # List of [Decimal(price), Decimal(amount)], sorted low to high

        if not bids or not asks:
            # self.logger.debug("Orderbook check skipped: Bids or asks list is empty.")
            return np.nan

        try:
            # Use configured number of levels
            levels_to_analyze = min(len(bids), len(asks), int(self.config.get("orderbook_limit", 10)))
            if levels_to_analyze <= 0: return 0.0

            # --- Calculate Weighted Volume Imbalance (more sophisticated) ---
            # Calculate mid-price as reference
            mid_price = (bids[0][0] + asks[0][0]) / Decimal('2') if bids and asks else current_price

            bid_pressure, ask_pressure = Decimal('0'), Decimal('0')
            # Weight levels closer to mid-price more heavily? Or just sum volume?
            # Let's try summing volume within the levels first (simpler OBI)
            total_bid_volume = sum(b[1] for b in bids[:levels_to_analyze])
            total_ask_volume = sum(a[1] for a in asks[:levels_to_analyze])

            total_volume = total_bid_volume + total_ask_volume
            if total_volume <= Decimal('1e-12'): # Avoid division by zero
                # self.logger.debug("Orderbook check: Zero total volume in analyzed levels.")
                return 0.0

            # Calculate Order Book Imbalance (OBI) - Ratio of bid volume to total volume
            # obi_ratio = total_bid_volume / total_volume
            # Scale OBI ratio [0, 1] to score [-1, 1] where 0.5 maps to 0
            # Score = (OBI Ratio - 0.5) * 2
            # score = float((obi_ratio - Decimal('0.5')) * Decimal('2'))

            # Alternative: Simple difference ratio
            obi_diff = (total_bid_volume - total_ask_volume) / total_volume
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi_diff))) # Already in range [-1, 1]

            # self.logger.debug(f"OB Check ({levels_to_analyze} levels): BidVol={total_bid_volume:.4f}, AskVol={total_ask_volume:.4f}, OBI_Diff={obi_diff:.4f} -> Score={score:.4f}")
            return score

        except (IndexError, ValueError, TypeError, InvalidOperation, ZeroDivisionError) as e:
             self.logger.warning(f"Orderbook analysis failed during calculation: {e}", exc_info=False)
             return np.nan

    # --- TP/SL Calculation ---

    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential Take Profit and initial Stop Loss based on entry estimate, signal, and ATR.
        Returns (Validated Entry Estimate, TP Price, SL Price) using Decimal precision and market constraints.
        Returns None for TP/SL if calculation is not possible or invalid.
        """
        # Initialize to None
        final_tp: Optional[Decimal] = None
        final_sl: Optional[Decimal] = None

        # --- Validate Inputs ---
        if signal not in ["BUY", "SELL"]:
            self.logger.debug(f"TP/SL Calc skipped: Invalid signal '{signal}'.")
            return entry_price_estimate, None, None

        atr_val = self.indicator_values.get("ATR") # Expect Decimal or NaN
        if not isinstance(atr_val, Decimal) or not atr_val.is_finite() or atr_val <= 0:
            self.logger.warning(f"TP/SL Calc Fail ({signal}): Invalid or non-positive ATR ({atr_val}). Cannot calculate TP/SL.")
            return entry_price_estimate, None, None

        if not isinstance(entry_price_estimate, Decimal) or not entry_price_estimate.is_finite() or entry_price_estimate <= 0:
            self.logger.warning(f"TP/SL Calc Fail ({signal}): Invalid entry price estimate ({entry_price_estimate}).")
            return entry_price_estimate, None, None

        try:
            # --- Get Parameters as Decimals ---
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", "1.0")))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", "1.5")))
            if tp_mult <= 0 or sl_mult <= 0:
                 self.logger.warning("TP/SL multipliers must be positive. Using defaults.")
                 tp_mult = Decimal(str(default_config["take_profit_multiple"]))
                 sl_mult = Decimal(str(default_config["stop_loss_multiple"]))


            # Fetch precision details using helper methods
            min_tick = self.get_min_tick_size() # Decimal
            # Define the quantization factor based on tick size
            quantizer = min_tick # Should be a valid positive Decimal from getter

            # --- Calculate Raw Offsets ---
            tp_offset = atr_val * tp_mult
            sl_offset = atr_val * sl_mult

            # --- Calculate Raw TP/SL Prices ---
            if signal == "BUY":
                tp_raw = entry_price_estimate + tp_offset
                sl_raw = entry_price_estimate - sl_offset
            else: # SELL
                tp_raw = entry_price_estimate - tp_offset
                sl_raw = entry_price_estimate + sl_offset

            # --- Quantize TP/SL using Market Tick Size ---
            # Round TP towards neutral (less profit -> floor for BUY, ceil for SELL)
            # Round SL away from entry (more room -> floor for BUY, ceil for SELL)
            if tp_raw.is_finite():
                rounding_mode_tp = ROUND_DOWN if signal == "BUY" else ROUND_UP
                final_tp = tp_raw.quantize(quantizer, rounding=rounding_mode_tp)

            if sl_raw.is_finite():
                # Note: SL rounding is AWAY from entry
                rounding_mode_sl = ROUND_DOWN if signal == "BUY" else ROUND_UP
                final_sl = sl_raw.quantize(quantizer, rounding=rounding_mode_sl)

            # --- Validation and Refinement ---
            # 1. Ensure SL is strictly beyond entry price by at least one tick
            if final_sl is not None:
                sl_entry_diff = abs(final_sl - entry_price_estimate)
                if sl_entry_diff < min_tick:
                    # Move SL one tick further away from entry
                    if signal == "BUY":
                        corrected_sl = (entry_price_estimate - min_tick).quantize(quantizer, rounding=ROUND_DOWN)
                    else: # SELL
                        corrected_sl = (entry_price_estimate + min_tick).quantize(quantizer, rounding=ROUND_UP)
                    # Only update if the correction is valid and actually different
                    if corrected_sl.is_finite() and corrected_sl != final_sl:
                         self.logger.debug(f"Adjusted {signal} SL {final_sl} to be >= 1 tick away: {corrected_sl}")
                         final_sl = corrected_sl
                    else:
                         self.logger.warning(f"{signal} SL {final_sl} is too close to Entry {entry_price_estimate} (Diff: {sl_entry_diff}, Tick: {min_tick}), correction failed or ineffective. Nullifying SL.")
                         final_sl = None # Nullify if adjustment failed


            # 2. Ensure TP offers potential profit (strictly beyond entry by at least one tick)
            if final_tp is not None:
                 tp_entry_diff = abs(final_tp - entry_price_estimate)
                 if tp_entry_diff < min_tick:
                     self.logger.warning(f"{signal} TP {final_tp} is too close to Entry {entry_price_estimate} (Diff: {tp_entry_diff}, Tick: {min_tick}). Nullifying TP.")
                     final_tp = None

            # 3. Ensure SL/TP are positive numbers
            if final_sl is not None and final_sl <= 0:
                 self.logger.error(f"Calculated SL is zero or negative ({final_sl}). Nullifying SL.")
                 final_sl = None
            if final_tp is not None and final_tp <= 0:
                 self.logger.warning(f"Calculated TP is zero or negative ({final_tp}). Nullifying TP.")
                 final_tp = None

            # --- Log Results ---
            price_prec = self.get_price_precision()
            tp_str = f"{final_tp:.{price_prec}f}" if final_tp else "None"
            sl_str = f"{final_sl:.{price_prec}f}" if final_sl else "None"
            # self.logger.debug(f"Calc TP/SL ({signal}): EntryEst={entry_price_estimate:.{price_prec}f}, ATR={atr_val:.{price_prec+2}f}, "
            #                   f"Tick={min_tick}, TP={tp_str}, SL={sl_str}")

            return entry_price_estimate, final_tp, final_sl

        except Exception as e:
            self.logger.error(f"Unexpected error calculating TP/SL for {signal}: {e}", exc_info=True)
            return entry_price_estimate, None, None


# --- Trading Logic Helper Functions ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches available balance for a specific currency using CCXT, handling V5 specifics.
    Prioritizes 'CONTRACT' account type for Bybit, falls back to default fetch.
    Returns available balance as Decimal, or None if fetch fails or balance not found.
    """
    lg = logger
    balance_info = None
    account_type_tried = "N/A" # Keep track of which account type succeeded or was tried last

    # --- Try Fetching with Specific Account Type (Bybit V5) ---
    if exchange.id == 'bybit':
        # Prioritize CONTRACT for derivatives. Could check for UNIFIED based on config/market if needed.
        account_type_to_try = 'CONTRACT'
        lg.debug(f"Attempting Bybit V5 balance fetch for {currency} (Account Type: {account_type_to_try})...")
        try:
            params = {'accountType': account_type_to_try}
            balance_info = safe_api_call(exchange.fetch_balance, lg, params=params)
            account_type_tried = account_type_to_try
            # lg.debug(f"Raw balance response (Type: {account_type_tried}): {json.dumps(balance_info, default=str)}") # Can be very verbose
        except ccxt.ExchangeError as e:
            # Handle specific errors like "account type does not exist" gracefully
            if "account type does not exist" in str(e).lower() or getattr(e, 'code', None) == 10001: # 10001 can sometimes mean wrong account type
                lg.info(f"Account type {account_type_to_try} likely not used. Falling back to default balance fetch.")
            else:
                lg.warning(f"Exchange error fetching balance with type {account_type_to_try}: {e}. Falling back to default fetch.")
            balance_info = None # Ensure fallback is triggered
        except Exception as e: # Catch errors from safe_api_call too
             lg.warning(f"Failed fetching balance with type {account_type_to_try}: {e}. Falling back to default fetch.")
             balance_info = None

    # --- Fallback to Default Fetch (or if not Bybit, or if specific type failed) ---
    if balance_info is None:
        lg.debug(f"Fetching balance for {currency} using default parameters...")
        try:
            balance_info = safe_api_call(exchange.fetch_balance, lg)
            account_type_tried = "Default" # Mark that default was used
            # lg.debug(f"Raw balance response (Type: {account_type_tried}): {json.dumps(balance_info, default=str)}")
        except Exception as e:
            lg.error(f"Failed to fetch balance info for {currency} even with default params: {e}")
            return None # Both attempts failed

    # --- Parse the balance_info (handle various structures) ---
    if not balance_info:
         lg.error(f"Balance fetch (Type: {account_type_tried}) returned empty or None.")
         return None

    free_balance_str = None
    parse_source = "Unknown"

    # Structure 1: Standard CCXT `balance[currency]['free']`
    if currency in balance_info and isinstance(balance_info.get(currency), dict):
        if balance_info[currency].get('free') is not None:
            free_balance_str = str(balance_info[currency]['free'])
            parse_source = f"Standard ['{currency}']['free']"
        elif balance_info[currency].get('available') is not None: # Alternative key some exchanges use
            free_balance_str = str(balance_info[currency]['available'])
            parse_source = f"Standard ['{currency}']['available']"

    # Structure 2: Top-level `balance['free'][currency]`
    elif isinstance(balance_info.get('free'), dict) and balance_info['free'].get(currency) is not None:
         free_balance_str = str(balance_info['free'][currency])
         parse_source = f"Top-level ['free']['{currency}']"
    elif isinstance(balance_info.get('available'), dict) and balance_info['available'].get(currency) is not None:
         free_balance_str = str(balance_info['available'][currency])
         parse_source = f"Top-level ['available']['{currency}']"

    # Structure 3: Bybit V5 Specific Parsing (from `info` field - more reliable for V5)
    # This needs careful handling as structure varies between CONTRACT, UNIFIED, SPOT etc.
    elif exchange.id == 'bybit' and isinstance(balance_info.get('info'), dict):
         info_data = balance_info['info']
         result_data = info_data.get('result', info_data) # Use result if present, else info itself

         # Check for list structure (common for V5)
         if isinstance(result_data.get('list'), list) and result_data['list']:
             account_list = result_data['list']
             # Try to find the specific account type if possible
             found_in_list = False
             for account_details in account_list:
                 acc_type = account_details.get('accountType')
                 # Check if this account matches the type we tried, or if we used Default
                 if account_type_tried == 'Default' or acc_type == account_type_tried:
                     # Check for Unified account structure (list of coins)
                     if isinstance(account_details.get('coin'), list):
                         for coin_data in account_details['coin']:
                             if coin_data.get('coin') == currency:
                                  # Prefer availableToWithdraw > availableBalance > walletBalance
                                  free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                  if free is not None:
                                      free_balance_str = str(free)
                                      parse_source = f"Bybit V5 info.result.list[coin='{currency}'] (AccType: {acc_type or 'N/A'})"
                                      found_in_list = True; break
                         if found_in_list: break
                     # Check for Contract account structure (direct balance fields)
                     elif acc_type == 'CONTRACT':
                           # Prefer availableBalance > walletBalance
                           free = account_details.get('availableBalance') or account_details.get('walletBalance')
                           if free is not None:
                                free_balance_str = str(free)
                                parse_source = f"Bybit V5 info.result.list[CONTRACT] (AccType: {acc_type})"
                                found_in_list = True; break
                     # Add SPOT account check if needed
                     # elif acc_type == 'SPOT': ...

             # If not found via specific type, maybe it's in the first entry generically? Less reliable.
             if not found_in_list and account_list:
                  # Fallback to first entry, look for common keys
                  first_acc = account_list[0]
                  free = first_acc.get('availableBalance') or first_acc.get('availableToWithdraw') or first_acc.get('walletBalance')
                  if free is not None:
                       free_balance_str = str(free)
                       parse_source = f"Bybit V5 info.result.list[0] (Fallback)"

         # Alternative: Look for the currency directly under 'info' or 'info.result' (less common for V5 balances)
         elif isinstance(result_data.get(currency), dict):
              currency_data = result_data[currency]
              free = currency_data.get('available') or currency_data.get('free') or currency_data.get('availableBalance') or currency_data.get('walletBalance')
              if free is not None:
                  free_balance_str = str(free)
                  parse_source = f"Bybit V5 info[.result]['{currency}']"


    # --- Fallback: Use 'total' if 'free'/'available' is completely unavailable ---
    if free_balance_str is None:
         total_balance_str = None
         parse_source_total = "Unknown Total"
         if currency in balance_info and isinstance(balance_info.get(currency), dict) and balance_info[currency].get('total') is not None:
             total_balance_str = str(balance_info[currency]['total'])
             parse_source_total = f"Standard ['{currency}']['total']"
         elif isinstance(balance_info.get('total'), dict) and balance_info['total'].get(currency) is not None:
             total_balance_str = str(balance_info['total'][currency])
             parse_source_total = f"Top-level ['total']['{currency}']"
         # Add V5 total check from info if needed

         if total_balance_str is not None:
              lg.warning(f"{NEON_YELLOW}Could not find 'free' or 'available' balance for {currency}. Using 'total' balance ({total_balance_str}) as fallback ({parse_source_total}). This may include collateral/unrealized PNL.{RESET}")
              free_balance_str = total_balance_str
              parse_source = parse_source_total + " (Fallback)"
         else:
              lg.error(f"{NEON_RED}Could not determine any balance ('free', 'available', or 'total') for {currency} after checking known structures (Account Type Searched: {account_type_tried}).{RESET}")
              lg.debug(f"Full balance_info structure: {json.dumps(balance_info, default=str)}")
              return None # No balance found

    # --- Convert the found balance string to Decimal ---
    try:
        final_balance = Decimal(free_balance_str)
        if not final_balance.is_finite():
             lg.warning(f"Parsed balance for {currency} ('{free_balance_str}' from {parse_source}) is not finite. Treating as zero.")
             final_balance = Decimal('0')
        # Allow negative balance? Some margin accounts might show negative available if overdrawn.
        # Let's treat negative available as zero for trading decisions.
        if final_balance < 0:
             lg.warning(f"Parsed available balance for {currency} ('{free_balance_str}' from {parse_source}) is negative. Treating as zero available.")
             final_balance = Decimal('0')

        lg.info(f"Available {currency} balance (Source: {parse_source}, AccType: {account_type_tried}): {final_balance:.4f}") # Adjust precision as needed
        return final_balance
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Failed to convert final balance string '{free_balance_str}' (from {parse_source}) to Decimal for {currency}: {e}")
        return None


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Gets market information dictionary from CCXT, ensuring markets are loaded.
    Adds convenience flags (is_contract, is_linear, etc.) for easier use.
    Returns the market dictionary or None on failure.
    """
    lg = logger
    try:
        # Ensure markets are loaded (critical step)
        if not exchange.markets or not exchange.markets_by_id: # Check both for robustness
             lg.info(f"Markets not loaded for {exchange.id}. Attempting to load...")
             try:
                 safe_api_call(exchange.load_markets, lg, reload=True)
                 lg.info(f"Markets loaded successfully ({len(exchange.symbols)} symbols).")
             except Exception as load_err:
                  lg.error(f"{NEON_RED}Failed to load markets after retries: {load_err}. Cannot get market info.{RESET}")
                  return None # Cannot proceed

        # Retrieve the market dictionary using the provided symbol
        market = exchange.market(symbol) # This handles lookup by symbol, id, base/quote etc.
        if not market or not isinstance(market, dict):
             lg.error(f"{NEON_RED}Market '{symbol}' not found in CCXT markets after loading.{RESET}")
             # Provide hint for common Bybit V5 format
             if '/' in symbol and ':' not in symbol and exchange.id == 'bybit':
                  base, quote = symbol.split('/')[:2] # Handle potential extra parts
                  suggested_symbol = f"{base}/{quote}:{quote}"
                  lg.warning(f"{NEON_YELLOW}Hint: For Bybit V5 linear perpetuals, try format like '{suggested_symbol}'.{RESET}")
             return None

        # --- Add Convenience Flags ---
        # Use .get() with defaults for safety
        market_type = market.get('type', '').lower()
        is_spot = market_type == 'spot'
        is_swap = market_type == 'swap' # Perpetual swaps
        is_future = market_type == 'future' # Dated futures
        # General contract flag
        is_contract = is_swap or is_future or market.get('contract', False)

        # Linear/Inverse (Crucial for contracts)
        is_linear = market.get('linear', False)
        is_inverse = market.get('inverse', False)
        # Infer if not explicitly set (common for V5 via defaultType)
        if is_contract and not is_linear and not is_inverse:
            # Check defaultType set during initialization
            default_type = exchange.options.get('defaultType', '').lower()
            if default_type == 'linear':
                 is_linear = True
            elif default_type == 'inverse':
                 is_inverse = True
            # Fallback inference based on quote currency (less reliable but common)
            elif market.get('quoteId', '').upper() == 'USD':
                 is_inverse = True
            else: # Default to linear for USDT, USDC etc. quoted contracts
                 is_linear = True
            # lg.debug(f"Inferred linear/inverse for {symbol}: Linear={is_linear}, Inverse={is_inverse}")

        market['is_spot'] = is_spot
        market['is_contract'] = is_contract
        market['is_linear'] = is_linear
        market['is_inverse'] = is_inverse

        # Log key details for verification
        lg.debug(f"Market Info ({symbol}): ID={market.get('id')}, Base={market.get('base')}, Quote={market.get('quote')}, "
                 f"Type={market_type}, Contract={is_contract}, Linear={is_linear}, Inverse={is_inverse}, "
                 f"Active={market.get('active', True)}, ContractSize={market.get('contractSize', 'N/A')}")
        # Log precision/limits for debugging sizing/order issues
        lg.debug(f"  Precision: {market.get('precision')}")
        lg.debug(f"  Limits: {market.get('limits')}")

        # Check if market is active
        if not market.get('active', True):
             lg.warning(f"{NEON_YELLOW}Market {symbol} is marked as inactive by the exchange.{RESET}")
             # Depending on strictness, could return None here

        return market

    except ccxt.BadSymbol as e:
        lg.error(f"{NEON_RED}Invalid symbol format or symbol not supported by {exchange.id}: '{symbol}'. Error: {e}{RESET}")
        return None
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float, # From config (0 to 1)
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange, # Needed for formatting amount/price
    logger: Optional[logging.Logger] = None,
    config_override: Optional[Dict] = None # Allow passing config for overrides if needed
) -> Optional[Decimal]:
    """
    Calculates position size based on risk, SL distance, balance, and market constraints.
    Uses Decimal for calculations and applies precision/step limits correctly.
    Handles Linear contracts. Inverse contracts explicitly not supported yet.
    Returns the calculated size as a Decimal, or None if calculation fails.
    """
    lg = logger or logging.getLogger(__name__)
    # Use global config if no override is provided
    cfg = config_override if config_override is not None else config

    # --- Extract Market Info & Validate Inputs ---
    symbol = market_info.get('symbol', 'UNKNOWN')
    quote_currency = market_info.get('quote', cfg.get("quote_currency", "USDT"))
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_linear = market_info.get('is_linear', False)
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit of the 'amount' field for orders/positions
    size_unit = base_currency if (is_linear or not is_contract) else "Contracts" # Inverse amount is in contracts

    # --- Input Validation ---
    if not isinstance(balance, Decimal) or not balance.is_finite() or balance <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive balance ({balance}).")
        return None
    if not isinstance(risk_per_trade, (float, int)) or not (0 < risk_per_trade < 1):
        lg.error(f"Size Calc Fail ({symbol}): Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
        return None
    if not isinstance(initial_stop_loss_price, Decimal) or not initial_stop_loss_price.is_finite() or initial_stop_loss_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if not isinstance(entry_price, Decimal) or not entry_price.is_finite() or entry_price <= 0:
        lg.error(f"Size Calc Fail ({symbol}): Invalid or non-positive entry_price ({entry_price}).")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Size Calc Fail ({symbol}): Stop loss price cannot be equal to entry price.")
        return None
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Size Calc Fail ({symbol}): Market info missing 'limits' or 'precision' dictionary.")
        return None

    # --- Explicitly block Inverse Contracts (requires different logic) ---
    if is_inverse:
        lg.error(f"{NEON_RED}Inverse contract sizing is not implemented. Aborting sizing for {symbol}.{RESET}")
        return None
    # Assume Linear if it's a contract but not explicitly Inverse (and warn if not explicitly Linear)
    if is_contract and not is_linear:
         lg.warning(f"{NEON_YELLOW}Market {symbol} is a contract but not marked as Linear. Assuming Linear sizing logic. Verify market info.{RESET}")

    try:
        # --- Initialize Analyzer for Precision/Step (lightweight) ---
        # Pass empty DF, only needs config and market_info here
        analyzer = TradingAnalyzer(pd.DataFrame(), lg, cfg, market_info)
        min_amount_step = analyzer.get_min_amount_step()
        amount_precision_places = analyzer.get_amount_precision_places()

        # --- Calculate Risk Amount in Quote Currency ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade))

        # --- Calculate Stop Loss Distance Per Unit ---
        # For Linear/Spot: Distance is in Quote Currency per unit of Base Currency (e.g., USDT per BTC)
        sl_distance_per_unit_quote = abs(entry_price - initial_stop_loss_price)
        if not sl_distance_per_unit_quote.is_finite() or sl_distance_per_unit_quote <= 0:
            lg.error(f"Size Calc Fail ({symbol}): Stop loss distance is zero, negative or NaN ({sl_distance_per_unit_quote}). Check SL/Entry prices.")
            return None

        # --- Get Contract Size (Value Multiplier) ---
        # For Linear/Spot: contractSize is 1 (amount is in base currency)
        # For Inverse: contractSize is value of 1 contract in USD (e.g., 100 for BTCUSD) - Not handled here.
        contract_size_val = Decimal('1') # Default for linear/spot
        if is_contract: # Should only be linear if we got here
            contract_size_str = market_info.get('contractSize')
            if contract_size_str is not None:
                try:
                    cs = Decimal(str(contract_size_str))
                    if cs.is_finite() and cs > 0:
                        contract_size_val = cs
                    else: raise ValueError("Invalid contract size value")
                except (ValueError, TypeError, InvalidOperation):
                    lg.warning(f"Invalid contract size '{contract_size_str}' for {symbol}. Using default {contract_size_val}.")
            # else: If contractSize missing for linear, 1 is usually correct assumption.

        # --- Calculate Initial Position Size (in Base Currency for Linear/Spot) ---
        # Size (Base) = Risk Amount (Quote) / (SL Distance (Quote/Base) * Contract Size (multiplier=1 for linear))
        if contract_size_val <= 0 or sl_distance_per_unit_quote <= 0:
             lg.error(f"Size Calc Fail ({symbol}): Invalid contract size ({contract_size_val}) or SL distance ({sl_distance_per_unit_quote}).")
             return None

        calculated_size = risk_amount_quote / (sl_distance_per_unit_quote * contract_size_val)

        if not calculated_size.is_finite() or calculated_size <= 0:
            lg.error(f"Initial size calculation resulted in zero/negative/NaN: {calculated_size}. Check inputs (Balance={balance}, Risk={risk_per_trade}, SLDist={sl_distance_per_unit_quote}).")
            return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f} {quote_currency}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote_currency}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit_quote}")
        lg.info(f"  ContractSize(Multiplier)={contract_size_val}, Size Unit={size_unit}")
        lg.info(f"  Initial Calculated Size = {calculated_size:.{amount_precision_places+4}f} {size_unit}") # Log with extra precision initially

        # --- Apply Market Limits and Precision/Step Size ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})

        min_amount_limit = Decimal(str(amount_limits.get('min', '0')))
        max_amount_limit = Decimal(str(amount_limits.get('max', 'inf')))
        min_cost_limit = Decimal(str(cost_limits.get('min', '0')))
        max_cost_limit = Decimal(str(cost_limits.get('max', 'inf')))

        adjusted_size = calculated_size

        # 1. Apply Amount Step Size (Round DOWN to nearest valid step)
        if min_amount_step.is_finite() and min_amount_step > 0:
            original_size_before_step = adjusted_size
            # Floor division rounds down to the nearest multiple of step size
            adjusted_size = (adjusted_size // min_amount_step) * min_amount_step
            if adjusted_size != original_size_before_step:
                 lg.info(f"  Size adjusted by Amount Step Size ({min_amount_step}): {original_size_before_step:.{amount_precision_places+2}f} -> {adjusted_size:.{amount_precision_places}f} {size_unit}")
        else:
             lg.warning(f"Amount step size is invalid ({min_amount_step}). Skipping step adjustment. Final size might be rejected.")
             # Fallback: Round to precision places (less accurate than step)
             adjusted_size = adjusted_size.quantize(Decimal('1e-' + str(amount_precision_places)), rounding=ROUND_DOWN)
             lg.info(f"  Size adjusted by Amount Precision ({amount_precision_places} places): {calculated_size:.{amount_precision_places+2}f} -> {adjusted_size:.{amount_precision_places}f} {size_unit}")


        # 2. Check Min Amount Limit AFTER step adjustment
        if min_amount_limit.is_finite() and min_amount_limit > 0 and adjusted_size < min_amount_limit:
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size after step/precision adjustment ({adjusted_size:.{amount_precision_places}f}) is below minimum limit ({min_amount_limit:.{amount_precision_places}f}). Increase risk or adjust SL.{RESET}")
             return None # Cannot place order below min amount

        # 3. Clamp by Max Amount Limit
        original_size_before_max_clamp = adjusted_size
        if max_amount_limit.is_finite() and adjusted_size > max_amount_limit:
             adjusted_size = max_amount_limit
             # Re-apply step rounding DOWN after clamping to max
             if min_amount_step.is_finite() and min_amount_step > 0:
                  adjusted_size = (adjusted_size // min_amount_step) * min_amount_step
             else: # Fallback precision rounding
                  adjusted_size = adjusted_size.quantize(Decimal('1e-' + str(amount_precision_places)), rounding=ROUND_DOWN)

             lg.warning(f"{NEON_YELLOW}Size capped by Max Amount Limit: {original_size_before_max_clamp:.{amount_precision_places}f} -> {adjusted_size:.{amount_precision_places}f} {size_unit}{RESET}")
             # Re-check min amount after capping
             if min_amount_limit.is_finite() and min_amount_limit > 0 and adjusted_size < min_amount_limit:
                  lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size after capping by max limit and rounding ({adjusted_size:.{amount_precision_places}f}) falls below minimum limit ({min_amount_limit:.{amount_precision_places}f}).{RESET}")
                  return None


        # 4. Check Cost Limits (Min/Max) with the final adjusted size
        # Estimated Cost (Quote) = Size (Base) * Entry Price (Quote/Base) * Contract Size (Multiplier)
        estimated_cost = adjusted_size * entry_price * contract_size_val
        lg.debug(f"  Cost Check: Final Size={adjusted_size:.{amount_precision_places}f} {size_unit}, Est. Cost={estimated_cost:.4f} {quote_currency} "
                 f"(Min Limit:{min_cost_limit}, Max Limit:{max_cost_limit})")

        # Check Min Cost (Notional Value)
        if min_cost_limit.is_finite() and min_cost_limit > 0 and estimated_cost < min_cost_limit:
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Estimated cost {estimated_cost:.4f} {quote_currency} is below minimum cost limit {min_cost_limit}. Increase risk/balance or adjust SL.{RESET}")
             return None

        # Check Max Cost (Notional Value)
        if max_cost_limit.is_finite() and max_cost_limit > 0 and estimated_cost > max_cost_limit:
             # This implies risk % calculation led to cost > max allowed. Cap size based on max cost.
             if entry_price > 0 and contract_size_val > 0:
                  # Calculate max size based on max cost: max_size = max_cost / (entry_price * contract_size_val)
                  size_based_on_max_cost = max_cost_limit / (entry_price * contract_size_val)
                  original_size_before_cost_cap = adjusted_size

                  # Re-apply step rounding DOWN
                  if min_amount_step.is_finite() and min_amount_step > 0:
                       adjusted_size_capped = (size_based_on_max_cost // min_amount_step) * min_amount_step
                  else: # Fallback rounding
                       adjusted_size_capped = size_based_on_max_cost.quantize(Decimal('1e-' + str(amount_precision_places)), rounding=ROUND_DOWN)

                  # Ensure we are reducing the size and it's still valid
                  if adjusted_size_capped < adjusted_size and adjusted_size_capped > 0:
                      lg.warning(f"{NEON_YELLOW}Size capped by Max Cost Limit: {original_size_before_cost_cap:.{amount_precision_places}f} -> {adjusted_size_capped:.{amount_precision_places}f} {size_unit}{RESET}")
                      adjusted_size = adjusted_size_capped
                      # Re-check min amount limit after cost capping
                      if min_amount_limit.is_finite() and min_amount_limit > 0 and adjusted_size < min_amount_limit:
                           lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Size after max cost capping ({adjusted_size:.{amount_precision_places}f}) falls below min amount limit ({min_amount_limit:.{amount_precision_places}f}).{RESET}")
                           return None
                  else:
                       lg.warning(f"Size capping based on Max Cost Limit ({max_cost_limit}) resulted in no change or invalid size ({adjusted_size_capped}). Original Size: {adjusted_size}")
                       # Stick with previous adjusted_size, but it might still fail on order placement

             else:
                 lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Estimated cost {estimated_cost:.4f} exceeds maximum cost limit {max_cost_limit}, but cannot recalculate size due to zero price/contract size.")
                 return None

        # --- Final Validation of the Calculated Size ---
        final_size = adjusted_size
        if not final_size.is_finite() or final_size <= 0:
             lg.error(f"{NEON_RED}Final calculated size is zero, negative, or NaN ({final_size}) after all adjustments. Aborting trade.{RESET}")
             return None
        # Final check against min amount (important after potential reductions)
        if min_amount_limit.is_finite() and min_amount_limit > 0 and final_size < min_amount_limit:
             lg.error(f"{NEON_RED}Size Calc Fail ({symbol}): Final size ({final_size:.{amount_precision_places}f}) is below minimum amount limit ({min_amount_limit:.{amount_precision_places}f}).{RESET}")
             return None

        # Use CCXT's amount_to_precision for final formatting (optional, step size should handle it)
        try:
             final_size_str = exchange.amount_to_precision(symbol, float(final_size))
             final_size_decimal = Decimal(final_size_str)
             lg.info(f"{NEON_GREEN}Final Position Size ({symbol}): {final_size_decimal} {size_unit} (Formatted: {final_size_str}){RESET}")
             return final_size_decimal
        except Exception as fmt_err:
             lg.warning(f"Could not format final size using exchange.amount_to_precision: {fmt_err}. Returning unformatted Decimal.")
             lg.info(f"{NEON_GREEN}Final Position Size ({symbol}): {final_size} {size_unit} (Unformatted){RESET}")
             return final_size


    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position using fetch_positions or fetch_position with robust parsing for Bybit V5.
    Returns a standardized dictionary for the open position, or None if no position exists.
    Handles potential errors and uses safe_api_call.
    """
    lg = logger
    # Check capability, prefer fetchPosition if available (more targeted for V5)
    has_fetch_position = exchange.has.get('fetchPosition', False)
    has_fetch_positions = exchange.has.get('fetchPositions', False)

    if not has_fetch_position and not has_fetch_positions:
        lg.warning(f"Exchange {exchange.id} supports neither fetchPosition nor fetchPositions. Cannot check position status.")
        return None

    # Get market info (needed for ID and parsing context)
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"Cannot get open position: Failed to load market info for {symbol}.")
        return None
    market_id = market_info.get('id')
    if not market_id:
         lg.error(f"Cannot get open position: Market ID missing for symbol {symbol}.")
         return None

    positions_data: List[Dict] = []
    fetch_method_used = "None"
    params = {}
    # Bybit V5 requires market ID for filtering
    if exchange.id == 'bybit':
        params = {'symbol': market_id, 'category': 'linear'} # Assume linear based on init

    try:
        # --- Attempt 1: fetchPosition (Preferred for V5 and specific symbol check) ---
        if has_fetch_position:
            fetch_method_used = f"fetchPosition(symbol='{symbol}')"
            lg.debug(f"Attempting {fetch_method_used}...")
            try:
                # fetchPosition returns a single position dict or raises error if not found/supported
                # Note: Some exchanges might return empty dict/list even for fetchPosition
                position_data = safe_api_call(exchange.fetch_position, lg, symbol, params=params)
                if isinstance(position_data, dict) and position_data: # Ensure it's a non-empty dict
                    positions_data = [position_data] # Wrap in list for consistent processing
                    lg.debug(f"{fetch_method_used} successful.")
                # Handle case where fetchPosition returns empty list/dict without error
                elif position_data is not None:
                     lg.info(f"No active position found for {symbol} via {fetch_method_used} (returned empty).")
                     return None
                # else: safe_api_call failed, error logged within

            except ccxt.NotSupported as e:
                lg.debug(f"{fetch_method_used} not supported ({e}). Falling back.")
                fetch_method_used = "None" # Reset for next attempt
            except ccxt.PositionNotFound as e: # Specific exception if implemented by ccxt wrapper
                 lg.info(f"No active position found for {symbol} via {fetch_method_used} ({type(e).__name__}).")
                 return None
            except ccxt.ExchangeError as e:
                 # Check for common "position not found" errors from exchanges (esp. Bybit V5)
                 err_str = str(e).lower()
                 # Bybit V5 Codes: 110025 (Position not found), can also appear for other issues.
                 # Sometimes returns generic errors for no position. Check messages.
                 bybit_no_pos_codes = [110025]
                 no_pos_messages = ["position not found", "position does not exist", "no position"]
                 is_no_pos_error = (getattr(e, 'code', None) in bybit_no_pos_codes) or \
                                   any(msg in err_str for msg in no_pos_messages)

                 if is_no_pos_error:
                     lg.info(f"No active position found for {symbol} via {fetch_method_used} (Code: {getattr(e, 'code', 'N/A')}, Msg: {e}).")
                     return None # No position exists
                 else:
                     # Log other exchange errors but attempt fallback if possible
                     lg.warning(f"Exchange error during {fetch_method_used}: {e}. Falling back.")
                     fetch_method_used = "None" # Reset for next attempt
            except Exception as e: # Catch other errors from safe_api_call
                 lg.warning(f"Error during {fetch_method_used}: {e}. Falling back.")
                 fetch_method_used = "None"

        # --- Attempt 2: fetchPositions filtered by symbol(s) (if fetchPosition failed/unavailable) ---
        if not positions_data and has_fetch_positions:
            # Some exchanges support filtering fetchPositions by symbol ID
            fetch_method_used = f"fetchPositions(symbols=['{symbol}'])"
            lg.debug(f"Attempting {fetch_method_used}...")
            try:
                 # Pass symbol in a list
                 fetched_data = safe_api_call(exchange.fetch_positions, lg, symbols=[symbol], params=params)
                 if fetched_data is not None: # Can return empty list if no position
                     positions_data = fetched_data
                     lg.debug(f"{fetch_method_used} successful, found {len(positions_data)} entries.")
                     # If empty list, means no position for this symbol
                     if not positions_data:
                          lg.info(f"No active position found for {symbol} via {fetch_method_used}.")
                          return None
                 # else: safe_api_call failed, error logged within

            except ccxt.NotSupported as e: # Filtering by symbol might not be supported
                  lg.debug(f"{fetch_method_used} filtering not supported ({e}). Falling back to fetch all.")
                  fetch_method_used = "None" # Reset for next attempt
            except ccxt.ExchangeError as e: # Catch other exchange errors during filtered fetch
                 lg.warning(f"Exchange error during {fetch_method_used}: {e}. Falling back to fetch all.")
                 fetch_method_used = "None"
            except Exception as e: # Catch other errors from safe_api_call
                 lg.warning(f"Error during {fetch_method_used}: {e}. Falling back to fetch all.")
                 fetch_method_used = "None"

        # --- Attempt 3: Fetch ALL positions (final fallback) ---
        if not positions_data and fetch_method_used == "None" and has_fetch_positions:
            fetch_method_used = "fetchPositions (all symbols)"
            lg.debug(f"Attempting {fetch_method_used} as fallback...")
            try:
                # Fetch all positions, remove params specific to filtering
                all_positions_data = safe_api_call(exchange.fetch_positions, lg, params={'category': 'linear'} if exchange.id == 'bybit' else {})
                if all_positions_data:
                     # Filter the results for the target symbol (using market_id for accuracy)
                     positions_data = [p for p in all_positions_data if p.get('info', {}).get('symbol') == market_id or p.get('symbol') == symbol]
                     lg.debug(f"Fetched {len(all_positions_data)} total positions, found {len(positions_data)} matching {symbol} (ID: {market_id}).")
                     if not positions_data:
                          lg.info(f"No active position found for {symbol} after fetching all positions.")
                          return None
                else:
                    lg.info(f"Fallback fetch of all positions returned no data or failed ({fetch_method_used}).")
                    return None
            except Exception as e:
                 lg.error(f"Error during fallback fetch of all positions: {e}")
                 return None # Final attempt failed

    except Exception as fetch_err:
         # Catch unexpected errors during the fetch logic itself
         lg.error(f"{NEON_RED}Unexpected error during position fetching process for {symbol}: {fetch_err}{RESET}", exc_info=True)
         return None

    # --- Process the Fetched Position Data ---
    active_position = None
    # Define a sensible threshold for non-zero size using Decimal
    # Use min amount step size if available, otherwise a small default
    analyzer = TradingAnalyzer(pd.DataFrame(), lg, config, market_info) # Temp instance for limits
    min_size_threshold = analyzer.get_min_amount_step() / Decimal('2') # Half the min step is a good threshold
    if min_size_threshold <= 0: min_size_threshold = Decimal('1e-9') # Fallback threshold

    if not positions_data:
        lg.info(f"No position data found/matched for {symbol} after checking via {fetch_method_used}.")
        return None

    # Iterate through the list (usually 0 or 1 entry after filtering)
    for pos_data in positions_data:
        if not isinstance(pos_data, dict): continue # Skip invalid entries

        # Ensure it matches the symbol (redundant check after filtering, but safe)
        pos_symbol = pos_data.get('symbol')
        info_symbol = pos_data.get('info', {}).get('symbol')
        if pos_symbol != symbol and info_symbol != market_id:
            continue

        # --- Extract Position Size (Crucial for V5, handle string/float/int) ---
        pos_size_val = None
        size_source = "N/A"
        # Try standard 'contracts' field (float/int)
        if pos_data.get('contracts') is not None:
            pos_size_val = pos_data['contracts']
            size_source = "'contracts' field"
        # Try Bybit V5 'info.size' field (string)
        elif isinstance(pos_data.get('info'), dict) and pos_data['info'].get('size') is not None:
             pos_size_val = pos_data['info']['size']
             size_source = "'info.size' field"
        # Try Bybit V5 'info.contracts' (sometimes present)
        elif isinstance(pos_data.get('info'), dict) and pos_data['info'].get('contracts') is not None:
             pos_size_val = pos_data['info']['contracts']
             size_source = "'info.contracts' field"

        if pos_size_val is None:
             lg.warning(f"Could not determine position size for an entry of {symbol} ({pos_data}). Skipping entry.")
             continue

        # --- Convert Size to Decimal and Check Threshold ---
        try:
            # Handle potential leading '+' sign from some exchanges
            pos_size_str = str(pos_size_val).lstrip('+')
            position_size_decimal = Decimal(pos_size_str)

            # Check if size magnitude exceeds threshold
            if abs(position_size_decimal) > min_size_threshold:
                lg.debug(f"Found potential active position entry for {symbol} (Size: {position_size_decimal} from {size_source}, Threshold: {min_size_threshold})")
                active_position = pos_data.copy() # Work on a copy
                # Standardize and store the Decimal size directly
                active_position['contractsDecimal'] = position_size_decimal
                break # Assume only one relevant position per symbol/side/mode
            else:
                 lg.debug(f"Ignoring position entry for {symbol} with size {position_size_decimal} (from {size_source}) below threshold {min_size_threshold}.")
        except (ValueError, TypeError, InvalidOperation) as parse_err:
            lg.warning(f"Could not parse position size '{pos_size_val}' (from {size_source}) for {symbol}: {parse_err}. Skipping entry.")
            continue

    # --- Post-Process the Found Active Position ---
    if active_position:
        try:
            # Ensure basic fields exist or initialize
            if 'info' not in active_position or not isinstance(active_position['info'], dict): active_position['info'] = {}
            info_dict = active_position['info']

            # --- Standardize Side ---
            pos_side = active_position.get('side')
            if pos_side not in ['long', 'short']:
                size_dec = active_position['contractsDecimal'] # Use Decimal size
                if size_dec > min_size_threshold: pos_side = 'long'
                elif size_dec < -min_size_threshold: pos_side = 'short'
                else:
                     lg.warning(f"Position size {size_dec} is below threshold {min_size_threshold}, cannot determine side reliably. Discarding position.")
                     return None
                active_position['side'] = pos_side
                # lg.debug(f"Inferred position side as '{pos_side}'.")

            # --- Standardize Entry Price (Decimal) ---
            entry_price_val = active_position.get('entryPrice') or info_dict.get('entryPrice') or info_dict.get('avgPrice') # V5 uses avgPrice in info
            active_position['entryPriceDecimal'] = Decimal(str(entry_price_val)) if entry_price_val is not None else None

            # --- Standardize Liquidation Price (Decimal) ---
            liq_price_val = active_position.get('liquidationPrice') or info_dict.get('liqPrice') # V5 uses liqPrice
            # Ignore '0' or invalid liq price strings
            active_position['liquidationPriceDecimal'] = Decimal(str(liq_price_val)) if liq_price_val is not None and str(liq_price_val) not in ['0', '0.0', ''] else None

            # --- Standardize Unrealized PNL (Decimal) ---
            pnl_val = active_position.get('unrealizedPnl') or info_dict.get('unrealisedPnl') # V5 uses unrealisedPnl
            active_position['unrealizedPnlDecimal'] = Decimal(str(pnl_val)) if pnl_val is not None else None

            # --- Standardize Leverage ---
            leverage_val = active_position.get('leverage') or info_dict.get('leverage')
            active_position['leverageDecimal'] = Decimal(str(leverage_val)) if leverage_val is not None else None

            # --- Extract SL/TP/TSL from 'info' (Bybit V5 is primary source) ---
            sl_str = info_dict.get('stopLoss')
            tp_str = info_dict.get('takeProfit')
            tsl_dist_str = info_dict.get('trailingStop') # Price distance value (string)
            tsl_act_str = info_dict.get('activePrice')   # Activation price for TSL (string)

            def safe_decimal_or_none(value_str):
                """Helper: Converts string to Decimal, returns None if invalid or zero/empty."""
                if value_str is None or str(value_str).strip() in ['', '0', '0.0']: return None
                try:
                    d = Decimal(str(value_str))
                    return d if d.is_finite() and d > 0 else None # Ensure positive
                except (InvalidOperation, ValueError, TypeError): return None

            active_position['stopLossPriceDecimal'] = safe_decimal_or_none(sl_str)
            active_position['takeProfitPriceDecimal'] = safe_decimal_or_none(tp_str)
            # Trailing stop distance is price distance from V5
            active_position['trailingStopLossValueDecimal'] = safe_decimal_or_none(tsl_dist_str)
            active_position['trailingStopActivationPriceDecimal'] = safe_decimal_or_none(tsl_act_str)

            # --- Get Timestamp (Prefer Bybit V5 updatedTime, convert to ms) ---
            ts_val = info_dict.get('updatedTime') or active_position.get('timestamp') # V5 updatedTime is ms string
            timestamp_ms = None
            if ts_val is not None:
                 try: timestamp_ms = int(ts_val)
                 except (ValueError, TypeError): pass
            active_position['timestamp_ms'] = timestamp_ms

            # --- Log Formatted Position Info ---
            price_prec = analyzer.get_price_precision()
            amt_prec = analyzer.get_amount_precision_places()

            # Helper for safe formatting of Decimals or showing 'N/A'
            def fmt(val, prec): return f"{val:.{prec}f}" if isinstance(val, Decimal) and val.is_finite() else 'N/A'

            entry_fmt = fmt(active_position['entryPriceDecimal'], price_prec)
            # Display absolute size with correct precision, include sign in log text maybe
            size_fmt = fmt(abs(active_position['contractsDecimal']), amt_prec)
            liq_fmt = fmt(active_position['liquidationPriceDecimal'], price_prec)
            lev_fmt = f"{fmt(active_position['leverageDecimal'], 1)}x" if active_position['leverageDecimal'] is not None else 'N/A'
            pnl_fmt = fmt(active_position['unrealizedPnlDecimal'], price_prec) # PNL uses price precision
            sl_fmt = fmt(active_position['stopLossPriceDecimal'], price_prec) if active_position['stopLossPriceDecimal'] else 'None'
            tp_fmt = fmt(active_position['takeProfitPriceDecimal'], price_prec) if active_position['takeProfitPriceDecimal'] else 'None'
            tsl_d_fmt = fmt(active_position['trailingStopLossValueDecimal'], price_prec) if active_position['trailingStopLossValueDecimal'] else 'None' # Distance
            tsl_a_fmt = fmt(active_position['trailingStopActivationPriceDecimal'], price_prec) if active_position['trailingStopActivationPriceDecimal'] else 'None' # Activation Px
            ts_dt_str = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp_ms else "N/A"

            logger.info(f"{NEON_GREEN}Active {pos_side.upper()} position found ({symbol}):{RESET} "
                        f"Size={size_fmt}, Entry={entry_fmt}, Liq={liq_fmt}, Lev={lev_fmt}, PnL={pnl_fmt}, "
                        f"SL={sl_fmt}, TP={tp_fmt}, TSL(Dist/Act): {tsl_d_fmt}/{tsl_a_fmt} (Updated: {ts_dt_str})")
            # Log the full processed dict at DEBUG level
            lg.debug(f"Full processed position details: {json.dumps(active_position, default=str, indent=2)}")
            return active_position

        except (ValueError, TypeError, InvalidOperation, KeyError) as proc_err:
             lg.error(f"Error processing active position details for {symbol}: {proc_err}", exc_info=True)
             lg.debug(f"Problematic raw position data: {active_position}")
             return None # Failed processing

    else:
        # No entries in the fetched list had a non-zero size
        logger.info(f"No active open position found for {symbol} (checked via {fetch_method_used}).")
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets leverage using CCXT's set_leverage method, handling Bybit V5 specifics.
    Returns True on success or if leverage is already set, False on failure.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.debug(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # No action needed, considered success

    if not isinstance(leverage, int) or leverage <= 0:
        lg.error(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be a positive integer.")
        return False

    # Check exchange capability (optional, as we might try anyway)
    if not exchange.has.get('setLeverage'):
        lg.warning(f"Exchange {exchange.id} may not support setLeverage via ccxt standard call. Attempting anyway...")

    market_id = market_info.get('id')
    if not market_id:
         lg.error(f"Cannot set leverage: Market ID missing for symbol {symbol}.")
         return False

    try:
        # Check current leverage first? Less efficient, but avoids unnecessary calls/errors
        # current_pos = get_open_position(exchange, symbol, lg) # Could reuse this, but adds latency
        # current_lev = current_pos.get('leverageDecimal') if current_pos else None
        # if current_lev is not None and current_lev == Decimal(str(leverage)):
        #      lg.info(f"Leverage for {symbol} already set to {leverage}x. Skipping.")
        #      return True

        lg.info(f"Attempting to set leverage for {symbol} (ID: {market_id}) to {leverage}x...")
        params = {}
        # --- Bybit V5 Specific Parameters ---
        if exchange.id == 'bybit':
            # V5 requires buyLeverage and sellLeverage as strings, category
