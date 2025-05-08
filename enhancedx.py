Okay, here is the enhanced version of your `sxs.py` script.

**Key Enhancements Implemented:**

1.  **Improved Docstrings and Comments:** Added comprehensive docstrings to functions and classes, explaining their purpose, parameters, and returns. Added comments for complex logic sections.
2.  **Constants:** Introduced constants for common strings like 'BUY', 'SELL', 'HOLD', 'long', 'short', 'market', 'limit' to improve readability and reduce typo risks. Defined constants for colors.
3.  **Configuration Validation:** Enhanced `load_config` with more specific error messages and added validation for nested structures (`indicators`, `weight_sets`) ensuring keys exist and values have the correct type.
4.  **Robust API Calls (`safe_api_call`):** Added logging for the *type* of error being retried. Ensured consistent `None` return on final failure. Added specific checks for `AuthenticationError` (non-retryable).
5.  **Data Fetching:** Ensured consistent use of `Decimal` in fetch functions. Added basic validation after fetching (e.g., checking klines aren't empty). Improved logging within fetch functions.
6.  **`TradingAnalyzer` Enhancements:**
    *   Added methods for `get_price_precision_places` and `get_amount_precision_places`.
    *   Ensured `Decimal` usage throughout calculations.
    *   Added logging for calculated indicator values (optional, can be enabled if needed).
    *   Refined `calculate_entry_tp_sl` for robust `Decimal` math and rounding.
    *   Improved logging in `generate_trading_signal`.
7.  **Trading Logic Functions:**
    *   **`calculate_position_size`:** Strict `Decimal` usage, better validation, clearer logging of calculation steps. Uses market precision step for final size adjustment.
    *   **`get_open_position`:** More robust parsing of position details using `Decimal`. Clearer logging. Specific parsing of V5 fields like SL/TP/TSL.
    *   **`place_trade`:** Uses constants, converts to `float` only at the CCXT call boundary. Includes `positionIdx` for Bybit.
    *   **`_set_position_protection`:** Carefully implemented based on Bybit V5 API (`/v5/position/set-trading-stop`) structure and parameters. Uses `Decimal` and converts to string for the API. Checks `retCode`.
    *   **`set_trailing_stop_loss`:** Calculates parameters using `Decimal` and market precision, then calls `_set_position_protection`.
    *   **Break-Even Logic:** Integrated BE logic directly into the main trading loop, checking profit against ATR multiple and using `_set_position_protection` to update the stop loss. Added check to prevent resetting TSL if BE triggers.
8.  **Main Loop (`analyze_and_trade_symbol`)**:
    *   Improved structure and flow for clarity.
    *   More detailed logging of decisions and actions.
    *   Wrapped core logic in `try...except` to prevent loop crashes on single cycle errors.
    *   Refined position management logic (entry, exit, BE updates).
9.  **Error Handling:** More specific `try...except` blocks in various places.
10. **Readability:** Improved variable names and code formatting.
11. **Type Hinting:** Reviewed and ensured comprehensive type hinting.

```python
# sxs.py
# Enhanced and Upgraded Scalping Bot Framework for Bybit V5
# Derived from xrscalper.py, focusing on robust execution, granular error handling,
# advanced position management (Break-Even, Trailing Stop Loss), and Bybit V5 API compatibility.
# Version: 1.1.0

# Standard Library Imports
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Third-Party Imports
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv

# --- Initialization ---
init(autoreset=True)  # Initialize Colorama
load_dotenv()         # Load environment variables from .env file

# Set Decimal precision for high-precision financial calculations
getcontext().prec = 36

# --- Color Scheme ---
COLOR_SUCCESS = Fore.LIGHTGREEN_EX
COLOR_INFO = Fore.CYAN
COLOR_WARNING = Fore.YELLOW
COLOR_ERROR = Fore.LIGHTRED_EX
COLOR_DEBUG = Fore.MAGENTA
COLOR_RESET = Style.RESET_ALL

# --- Constants ---
# Credentials (Ensure these are set in your .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True) # Ensure log directory exists

# Timezone
DEFAULT_TIMEZONE_STR = "America/Chicago"
TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE_STR) # Default, will be updated from config

# API Call Settings
MAX_API_RETRIES = 5
RETRY_DELAY_SECONDS = 7
RETRYABLE_HTTP_CODES = [429, 500, 502, 503, 504] # Common retryable server/rate limit errors

# Trading Parameters
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = { # Maps config intervals to CCXT timeframe strings
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
BUY_SIGNAL = "BUY"
SELL_SIGNAL = "SELL"
HOLD_SIGNAL = "HOLD"
LONG_SIDE = "long"
SHORT_SIDE = "short"
MARKET_ORDER = "market"
LIMIT_ORDER = "limit"

# Default Indicator Periods (used if not specified in config)
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
DEFAULT_FIB_PERIOD = 50 # Note: Fibonacci calculation isn't explicitly used in signals
DEFAULT_PSAR_STEP = 0.02
DEFAULT_PSAR_MAX_STEP = 0.2

# Misc Defaults
DEFAULT_LOOP_DELAY_SECONDS = 10
DEFAULT_POSITION_CONFIRM_DELAY_SECONDS = 10
MIN_DECIMAL_VALUE = Decimal("1e-18") # Small value for comparisons

# Global config dictionary (populated by load_config)
config: Dict[str, Any] = {}
default_config: Dict[str, Any] = {} # Stores the structure and defaults

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive information (API keys)."""
    _patterns = {}
    # Class variable to store sensitive keys and their values
    _sensitive_keys = {"API_KEY": API_KEY, "API_SECRET": API_SECRET}

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing sensitive values."""
        msg = super().format(record)
        # Update patterns if keys change (though unlikely in this setup)
        for key_name, key_value in self._sensitive_keys.items():
            if key_value: # Only process if the key has a value
                if key_value not in self._patterns:
                    # Create a redaction pattern like "***API_KEY***"
                    self._patterns[key_value] = f"***{key_name}***"
                # Replace occurrences of the sensitive value in the log message
                if key_value in msg:
                    msg = msg.replace(key_value, self._patterns[key_value])
        return msg

def _merge_configs(loaded_config: Dict, default_config: Dict) -> Dict:
    """
    Recursively merges the loaded configuration with the default configuration.
    Ensures that all keys from the default config are present and handles nested dictionaries.

    Args:
        loaded_config: The configuration dictionary loaded from the file.
        default_config: The default configuration dictionary.

    Returns:
        A new dictionary representing the merged configuration.
    """
    merged = default_config.copy()
    for key, value in loaded_config.items():
        if key in merged:
            if isinstance(value, dict) and isinstance(merged[key], dict):
                # Recursively merge nested dictionaries
                merged[key] = _merge_configs(value, merged[key])
            else:
                # Overwrite default value with loaded value (if types match or loaded is acceptable)
                # Validation happens later, here we just merge structure
                merged[key] = value
        else:
            # Keep keys from loaded config even if not in default (might be user additions)
            # Consider logging a warning here if strict adherence to defaults is required
            merged[key] = value
    return merged

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file, validates it against defaults and constraints,
    and saves an updated version if defaults were applied or the structure changed.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        The validated configuration dictionary.

    Raises:
        ValueError: If critical configuration like API keys are missing.
    """
    global TIMEZONE, default_config # Allow modification of global TIMEZONE and default_config storage

    # Define the default configuration structure and values
    default_config = {
        # Core Settings
        "symbol": "BTC/USDT:USDT",
        "interval": "5", # See VALID_INTERVALS
        "enable_trading": False, # Master switch for live trading
        "use_sandbox": True, # Use Bybit testnet environment
        "quote_currency": "USDT", # Currency for balance/PNL calculations
        "timezone": DEFAULT_TIMEZONE_STR, # For console logging timestamps

        # API & Loop Timing
        "retry_delay": RETRY_DELAY_SECONDS, # Base delay for API retries (seconds)
        "max_api_retries": MAX_API_RETRIES, # Max retries for failed API calls
        "position_confirm_delay_seconds": DEFAULT_POSITION_CONFIRM_DELAY_SECONDS, # Wait time after placing order (seconds)
        "loop_delay_seconds": DEFAULT_LOOP_DELAY_SECONDS, # Pause between cycles (seconds)

        # Position & Risk Management
        "max_concurrent_positions": 1, # Max open positions for this symbol (currently supports 1)
        "risk_per_trade": 0.01, # Fraction of balance to risk (e.g., 0.01 = 1%)
        "leverage": 20, # Desired leverage
        "stop_loss_multiple": 1.8, # Stop loss distance in ATR multiples
        "take_profit_multiple": 0.7, # Take profit distance in ATR multiples

        # Order Execution
        "entry_order_type": MARKET_ORDER, # 'market' or 'limit'
        "limit_order_offset_buy": 0.0005, # % offset below current price for limit buy (e.g., 0.0005 = 0.05%)
        "limit_order_offset_sell": 0.0005, # % offset above current price for limit sell

        # Advanced Position Management
        "enable_trailing_stop": True,
        "trailing_stop_callback_rate": 0.005, # Trail distance as % of activation price (e.g., 0.005 = 0.5%)
        "trailing_stop_activation_percentage": 0.003, # Profit % to trigger TSL activation (e.g., 0.003 = 0.3%)
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0, # Profit in ATR multiples to trigger BE
        "break_even_offset_ticks": 2, # Number of price ticks above/below entry for BE stop loss

        # Exit Conditions
        "time_based_exit_minutes": None, # Optional: Close position after X minutes (not implemented in core loop yet)

        # Indicator Periods & Settings
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
        "fibonacci_period": DEFAULT_FIB_PERIOD, # Used for potential future analysis

        # Data Fetching
        "orderbook_limit": 25, # Number of bids/asks to fetch

        # Signal Generation
        "signal_score_threshold": 1.5, # Minimum absolute weighted score to generate BUY/SELL
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "volume_confirmation_multiplier": 1.5, # Current volume must be > X * MA volume

        # Indicator Activation (Which indicators contribute to the score)
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True,
            # ATR is always calculated for risk management, not directly for signal score
        },
        # Weight Sets (Define different weighting strategies)
        "weight_sets": {
            "scalping": { # Example aggressive scalping weights
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # Default balanced weights
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default" # Which weight set to use for scoring
    }

    current_config = default_config.copy() # Start with defaults
    needs_saving = False # Flag to indicate if the config file should be updated

    # --- Load existing config file ---
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_config_raw = json.load(f)
            # Merge loaded config onto defaults to ensure all keys are present
            current_config = _merge_configs(loaded_config_raw, default_config)
            print(f"{COLOR_SUCCESS}Successfully loaded and merged configuration from {filepath}{COLOR_RESET}")
            # Check if the loaded config was missing keys or had different structure
            if current_config != loaded_config_raw:
                print(f"{COLOR_WARNING}Configuration was updated with default values or structure. Review changes.{COLOR_RESET}")
                needs_saving = True
        except json.JSONDecodeError as e:
            print(f"{COLOR_ERROR}Error decoding JSON from config file {filepath}: {e}. Using default configuration.{COLOR_RESET}")
            current_config = default_config.copy() # Reset to defaults on decode error
            needs_saving = True
        except Exception as e:
            print(f"{COLOR_ERROR}Unexpected error loading config file {filepath}: {e}. Using default configuration.{COLOR_RESET}")
            current_config = default_config.copy()
            needs_saving = True
    else:
        print(f"{COLOR_WARNING}Config file not found at {filepath}. Creating a default configuration file.{COLOR_RESET}")
        current_config = default_config.copy()
        needs_saving = True

    # --- Validation ---
    validation_errors = []

    def _log_validation_error(message: str, key: str, default_value: Any):
        nonlocal needs_saving
        validation_errors.append(message)
        current_config[key] = default_value # Reset to default
        needs_saving = True

    # Validate string parameters
    validate_param("symbol", lambda v: isinstance(v, str) and v.strip(), "Symbol must be a non-empty string.", current_config, default_config, _log_validation_error)
    validate_param("interval", lambda v: v in VALID_INTERVALS, f"Interval must be one of {VALID_INTERVALS}.", current_config, default_config, _log_validation_error)
    validate_param("entry_order_type", lambda v: v in [MARKET_ORDER, LIMIT_ORDER], f"Entry order type must be '{MARKET_ORDER}' or '{LIMIT_ORDER}'.", current_config, default_config, _log_validation_error)
    validate_param("quote_currency", lambda v: isinstance(v, str) and len(v) >= 3 and v.isupper(), "Quote currency must be a valid uppercase currency code (e.g., USDT).", current_config, default_config, _log_validation_error)
    validate_param("active_weight_set", lambda v: isinstance(v, str) and v in current_config.get("weight_sets", {}), f"Active weight set must be a valid key from 'weight_sets'.", current_config, default_config, _log_validation_error)

    # Validate boolean parameters
    validate_param("enable_trading", lambda v: isinstance(v, bool), "Enable trading must be true or false.", current_config, default_config, _log_validation_error)
    validate_param("use_sandbox", lambda v: isinstance(v, bool), "Use sandbox must be true or false.", current_config, default_config, _log_validation_error)
    validate_param("enable_trailing_stop", lambda v: isinstance(v, bool), "Enable trailing stop must be true or false.", current_config, default_config, _log_validation_error)
    validate_param("enable_break_even", lambda v: isinstance(v, bool), "Enable break even must be true or false.", current_config, default_config, _log_validation_error)

    # Validate Timezone
    try:
        tz_str = current_config.get("timezone", DEFAULT_TIMEZONE_STR)
        TIMEZONE = ZoneInfo(tz_str)
    except ZoneInfoNotFoundError:
        _log_validation_error(f"Invalid timezone string '{tz_str}'. Must be a valid IANA time zone name.", "timezone", default_config["timezone"])
        TIMEZONE = ZoneInfo(default_config["timezone"]) # Ensure TIMEZONE is set
    except Exception as e:
        _log_validation_error(f"Error processing timezone '{tz_str}': {e}", "timezone", default_config["timezone"])
        TIMEZONE = ZoneInfo(default_config["timezone"])

    # Validate numeric parameters
    numeric_params: Dict[str, Tuple[Union[int, float], Union[int, float], bool, bool, bool]] = {
        # key: (min_val, max_val, allow_min_inclusive, allow_max_inclusive, is_integer)
        "risk_per_trade": (0, 1, False, False, False), # Must be > 0 and < 1
        "leverage": (1, 150, True, True, True), # Bybit max leverage varies, 150 is common cap
        "stop_loss_multiple": (0, 100, False, True, False), # Must be > 0
        "take_profit_multiple": (0, 100, False, True, False), # Must be > 0
        "trailing_stop_callback_rate": (0, 0.5, False, False, False), # Must be > 0, reasonable upper limit
        "trailing_stop_activation_percentage": (0, 0.5, False, False, False), # Must be > 0
        "break_even_trigger_atr_multiple": (0, 100, False, True, False), # Must be > 0
        "break_even_offset_ticks": (0, 1000, True, True, True), # Can be 0 or positive integer
        "signal_score_threshold": (0, 100, False, True, False), # Must be > 0
        "atr_period": (2, 1000, True, True, True),
        "ema_short_period": (1, 1000, True, True, True),
        "ema_long_period": (2, 1000, True, True, True), # Long EMA > Short EMA validated elsewhere if needed
        "rsi_period": (2, 1000, True, True, True),
        "bollinger_bands_period": (2, 1000, True, True, True),
        "bollinger_bands_std_dev": (0, 10, False, True, False), # Must be > 0
        "cci_period": (2, 1000, True, True, True),
        "williams_r_period": (2, 1000, True, True, True),
        "mfi_period": (2, 1000, True, True, True),
        "stoch_rsi_period": (2, 1000, True, True, True),
        "stoch_rsi_rsi_period": (2, 1000, True, True, True),
        "stoch_rsi_k_period": (1, 1000, True, True, True),
        "stoch_rsi_d_period": (1, 1000, True, True, True),
        "psar_step": (0, 1, False, True, False), # Must be > 0
        "psar_max_step": (0, 1, False, True, False), # Must be > 0
        "sma_10_period": (1, 1000, True, True, True),
        "momentum_period": (1, 1000, True, True, True),
        "volume_ma_period": (1, 1000, True, True, True),
        "fibonacci_period": (2, 1000, True, True, True),
        "orderbook_limit": (1, 200, True, True, True), # Bybit V5 limit is 200 for level 2
        "position_confirm_delay_seconds": (0, 120, True, True, False),
        "loop_delay_seconds": (1, 300, True, True, False),
        "stoch_rsi_oversold_threshold": (0, 100, True, False, False), # >= 0 and < 100
        "stoch_rsi_overbought_threshold": (0, 100, False, True, False), # > 0 and <= 100
        "volume_confirmation_multiplier": (0, 100, False, True, False), # Must be > 0
        "limit_order_offset_buy": (0, 0.1, True, True, False), # 0% to 10%
        "limit_order_offset_sell": (0, 0.1, True, True, False), # 0% to 10%
        "retry_delay": (1, 120, True, True, False),
        "max_api_retries": (0, 10, True, True, True),
        "max_concurrent_positions": (1, 10, True, True, True), # Limited practicality > 1 for now
    }

    for key, (min_val, max_val, allow_min, allow_max, is_int) in numeric_params.items():
        value = current_config.get(key)
        default_val = default_config.get(key)
        try:
            # Attempt conversion to Decimal first for robust validation
            val_dec = Decimal(str(value))
            if not val_dec.is_finite():
                raise ValueError("Value is not finite (NaN or Infinity).")

            # Check bounds
            lower_bound_ok = (val_dec >= min_val) if allow_min else (val_dec > min_val)
            upper_bound_ok = (val_dec <= max_val) if allow_max else (val_dec < max_val)

            if not (lower_bound_ok and upper_bound_ok):
                bound_str = f"{'>=' if allow_min else '>'}{min_val} and {'<=' if allow_max else '<'}{max_val}"
                raise ValueError(f"Value must be within the range: {bound_str}.")

            # Assign validated and correctly typed value
            current_config[key] = int(val_dec) if is_int else float(val_dec)

        except (ValueError, TypeError, InvalidOperation) as e:
            type_str = 'an integer' if is_int else 'a number'
            bound_str = f"{'>=' if allow_min else '>'}{min_val} and {'<=' if allow_max else '<'}{max_val}"
            _log_validation_error(f"Invalid value '{value}' for '{key}'. Must be {type_str} where {bound_str}. Error: {e}", key, default_val)

    # Validate nested structures: indicators and weight_sets
    if not isinstance(current_config.get("indicators"), dict):
         _log_validation_error("Config 'indicators' must be a dictionary.", "indicators", default_config["indicators"])
    else:
        # Ensure all default indicator keys exist and have boolean values
        default_indicators = default_config["indicators"]
        for ind_key, default_val in default_indicators.items():
            if ind_key not in current_config["indicators"] or not isinstance(current_config["indicators"].get(ind_key), bool):
                 if ind_key not in current_config["indicators"]:
                      msg = f"Missing key '{ind_key}' in 'indicators'."
                 else:
                     msg = f"Invalid value type for '{ind_key}' in 'indicators'. Must be boolean (true/false)."
                 _log_validation_error(msg + " Resetting to default.", f"indicators.{ind_key}", default_val)
                 current_config["indicators"][ind_key] = default_val # Ensure key exists after error

    if not isinstance(current_config.get("weight_sets"), dict):
        _log_validation_error("Config 'weight_sets' must be a dictionary.", "weight_sets", default_config["weight_sets"])
    else:
        default_weight_sets = default_config["weight_sets"]
        # Ensure default weight sets exist
        for set_name, default_weights in default_weight_sets.items():
            if set_name not in current_config["weight_sets"] or not isinstance(current_config["weight_sets"].get(set_name), dict):
                _log_validation_error(f"Weight set '{set_name}' is missing or not a dictionary.", f"weight_sets.{set_name}", default_weights)
                current_config["weight_sets"][set_name] = default_weights # Reset the whole set
            else:
                # Validate individual weights within the set
                current_weights = current_config["weight_sets"][set_name]
                for ind_key, default_weight in default_weights.items():
                    weight_val = current_weights.get(ind_key)
                    is_valid_num = isinstance(weight_val, (int, float)) and not isinstance(weight_val, bool)
                    if ind_key not in current_weights or not is_valid_num:
                         if ind_key not in current_weights:
                             msg = f"Missing weight key '{ind_key}' in weight set '{set_name}'."
                         else:
                             msg = f"Invalid weight value for '{ind_key}' in set '{set_name}'. Must be a number."
                         _log_validation_error(msg + " Resetting to default.", f"weight_sets.{set_name}.{ind_key}", default_weight)
                         current_weights[ind_key] = default_weight # Reset specific weight

    # Print all validation errors at the end
    if validation_errors:
        print(f"{COLOR_ERROR}--- Configuration Validation Errors ---{COLOR_RESET}")
        for error in validation_errors:
            print(f"{COLOR_ERROR}- {error}{COLOR_RESET}")
        print(f"{COLOR_ERROR}--- Configuration has been reset to defaults where errors occurred. ---{COLOR_RESET}")

    # Save the potentially modified configuration
    if needs_saving:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                # Dump the validated (and potentially corrected) config
                json.dump(current_config, f, indent=4, sort_keys=True)
            print(f"{COLOR_WARNING}Saved updated configuration file to {filepath}{COLOR_RESET}")
        except Exception as e:
            print(f"{COLOR_ERROR}Critical Error: Could not save configuration file {filepath}: {e}{COLOR_RESET}")
            # Depending on severity, might want to raise an exception here

    # Final check for essential API keys AFTER potential load/save
    if not API_KEY or not API_SECRET:
        # Use basicConfig because logger setup might depend on config
        logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s: %(message)s')
        logging.critical(f"{COLOR_ERROR}CRITICAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env file or environment variables.{COLOR_RESET}")
        raise ValueError("API keys not configured. Bot cannot start.")

    return current_config

def validate_param(key: str, validation_func: callable, error_msg: str, current_config: Dict, default_config: Dict, error_callback: callable):
    """Helper function to validate a single parameter."""
    value = current_config.get(key)
    default = default_config.get(key)
    try:
        if key not in current_config or not validation_func(value):
             error_callback(f"{error_msg} Invalid value: '{repr(value)}'. Resetting to default '{default}'.", key, default)
    except Exception as e:
        # Catch errors within the validation function itself
        error_callback(f"Error during validation for '{key}' (value: '{repr(value)}'): {e}. Resetting to default '{default}'.", key, default)


# --- Logging Setup ---
def setup_logger(name: str, config_dict: Optional[Dict[str, Any]] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with both file (UTC) and console (local time) handlers.

    Args:
        name: The name for the logger.
        config_dict: The configuration dictionary, used to get the timezone for console logs.
        level: The minimum logging level for the console handler (e.g., logging.INFO).

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger(name)
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG) # Set base level to lowest, handlers will filter
    config_dict = config_dict or config # Use global config if specific one not provided

    # --- File Handler (UTC time) ---
    log_filename = os.path.join(LOG_DIRECTORY, f"{name.replace(':', '_').replace('/', '_')}.log") # Sanitize name
    try:
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=10 * 1024 * 1024, # 10 MB per file
            backupCount=5,             # Keep 5 backup logs
            encoding='utf-8'
        )
        # Use the custom formatter to redact sensitive data
        file_formatter = SensitiveFormatter(
            "%(asctime)s.%(msecs)03dZ %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%dT%H:%M:%S'
        )
        file_formatter.converter = time.gmtime # Use UTC for file logs
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG) # Log everything to file
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to basic console logging if file handler fails
        logging.basicConfig(level=logging.ERROR)
        logger.error(f"Failed to configure file logger for {name}: {e}", exc_info=True)
        logger.error("Logging to file disabled for this logger.")


    # --- Console Handler (Local time based on config) ---
    try:
        console_tz_str = config_dict.get("timezone", DEFAULT_TIMEZONE_STR)
        console_tz = ZoneInfo(console_tz_str)
    except Exception:
        logger.warning(f"Could not load timezone '{config_dict.get('timezone')}'. Using default {DEFAULT_TIMEZONE_STR} for console logs.")
        console_tz = ZoneInfo(DEFAULT_TIMEZONE_STR)

    stream_handler = logging.StreamHandler()
    # Use the custom formatter here as well if console output might contain keys
    console_formatter = SensitiveFormatter(
        f"{COLOR_INFO}%(asctime)s{COLOR_RESET} {COLOR_WARNING}%(levelname)-8s{COLOR_RESET} {COLOR_DEBUG}[%(name)s]{COLOR_RESET} %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S %Z' # Include timezone name
    )
    # Custom time converter for local timezone display
    console_formatter.converter = lambda *args: datetime.now(console_tz).timetuple()
    stream_handler.setFormatter(console_formatter)
    stream_handler.setLevel(level) # Set console level (e.g., INFO, WARNING)
    logger.addHandler(stream_handler)

    # Prevent logs from propagating to the root logger if handlers are set
    logger.propagate = False
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(config_dict: Dict[str, Any], logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object with API keys, options, and sandbox mode.

    Args:
        config_dict: The configuration dictionary.
        logger: The logger instance for logging messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    lg.info(f"Initializing Bybit V5 exchange (ccxt version: {ccxt.__version__})")
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'rateLimit': 150, # milliseconds between requests (adjust as needed)
            'options': {
                'defaultType': 'linear',       # Or 'inverse' if needed
                'adjustForTimeDifference': True, # Auto-sync client/server time
                # Timeouts in milliseconds for various API calls
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 25000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 25000,
                'fetchOHLCVTimeout': 20000,
                'fetchOrderBookTimeout': 15000,
                'setLeverageTimeout': 20000,
                'fetchMyTradesTimeout': 20000,
                'fetchClosedOrdersTimeout': 25000,
                 # Bybit V5 specific: enable Unified Margin Account if needed
                # 'enableUnifiedAccount': True,
                # 'enableUnifiedMargin': True,
            }
        })

        # Configure Sandbox (Testnet) or Live mode
        if config_dict.get('use_sandbox', True):
            lg.warning(f"{COLOR_WARNING}--- SANDBOX MODE ENABLED ---{COLOR_RESET}")
            try:
                exchange.set_sandbox_mode(True)
                # Double-check the API URL, sometimes set_sandbox_mode needs help
                if 'testnet' not in str(exchange.urls.get('api', '')).lower():
                    lg.warning("Sandbox URL might not be set correctly by set_sandbox_mode, attempting manual override.")
                    # Explicitly set testnet URL for V5
                    exchange.urls['api'] = 'https://api-testnet.bybit.com'
                lg.info(f"Using Testnet API Endpoint: {exchange.urls['api']}")
            except Exception as e:
                 lg.error(f"Failed to set sandbox mode: {e}. Check CCXT version and Bybit implementation.", exc_info=True)
                 return None
        else:
            lg.info(f"{COLOR_SUCCESS}--- LIVE TRADING MODE ENABLED ---{COLOR_RESET}")
            # Ensure the production URL is used if previously sandboxed
            if 'testnet' in str(exchange.urls.get('api', '')).lower():
                 lg.info("Resetting API URL to production.")
                 exchange.urls['api'] = 'https://api.bybit.com'
            lg.info(f"Using Production API Endpoint: {exchange.urls['api']}")

        # Load markets to verify connection and get symbol details
        lg.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets(reload=True) # Force reload to get latest info
        lg.info("Markets loaded successfully.")

        # Validate the configured symbol exists on the exchange
        symbol = config_dict["symbol"]
        if symbol not in exchange.markets:
            lg.critical(f"{COLOR_ERROR}Symbol '{symbol}' not found on {exchange.id}. Available markets: {list(exchange.markets.keys())[:10]}...{COLOR_RESET}")
            return None
        lg.info(f"Symbol '{symbol}' found and validated.")

        # Perform an initial balance fetch to confirm API key validity
        quote_currency = config_dict["quote_currency"]
        lg.info(f"Fetching initial balance for {quote_currency}...")
        initial_balance = fetch_balance(exchange, quote_currency, lg)
        if initial_balance is None:
            lg.critical(f"{COLOR_ERROR}Failed to fetch initial balance. Check API key permissions and connection.{COLOR_RESET}")
            # Don't return None here if trading is disabled, but log critical error.
            # If trading enabled, failing balance check is critical.
            if config_dict.get("enable_trading"):
                 return None
            lg.warning("Proceeding without initial balance check as trading is disabled.")
        else:
            lg.info(f"{COLOR_SUCCESS}Initial balance fetch successful: {initial_balance:.4f} {quote_currency}{COLOR_RESET}")

        lg.info(f"{COLOR_SUCCESS}Exchange '{exchange.id}' initialized successfully.{COLOR_RESET}")
        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{COLOR_ERROR}Authentication Error: Failed to authenticate with Bybit. Check API Key and Secret. Details: {e}{COLOR_RESET}", exc_info=True)
        return None
    except ccxt.ExchangeError as e:
        lg.critical(f"{COLOR_ERROR}Exchange Error: An error occurred during exchange initialization. Details: {e}{COLOR_RESET}", exc_info=True)
        return None
    except Exception as e:
        lg.critical(f"{COLOR_ERROR}Unexpected Error: Failed to initialize exchange: {e}{COLOR_RESET}", exc_info=True)
        return None

# --- API Call Helper ---
def safe_api_call(func: callable, logger: logging.Logger, *args, **kwargs) -> Any:
    """
    Wraps API calls with retry logic for network errors and rate limits.

    Args:
        func: The CCXT exchange method to call (e.g., exchange.fetch_balance).
        logger: The logger instance.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the API call, or None if it fails after all retries.
    """
    lg = logger
    max_retries = config.get("max_api_retries", MAX_API_RETRIES)
    base_delay = config.get("retry_delay", RETRY_DELAY_SECONDS)

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            # lg.debug(f"API call {func.__name__} successful (Attempt {attempt + 1})")
            return result
        except ccxt.AuthenticationError as e:
            lg.error(f"{COLOR_ERROR}API Authentication Error on {func.__name__}: {e}. Check keys/permissions. Not retrying.{COLOR_RESET}")
            return None # Authentication errors are not retryable
        except ccxt.RateLimitExceeded as e:
            delay = base_delay * (2 ** attempt) # Exponential backoff
            if attempt == max_retries:
                lg.error(f"{COLOR_ERROR}API Rate Limit Exceeded on {func.__name__} after {max_retries} retries: {e}{COLOR_RESET}")
                return None
            lg.warning(f"{COLOR_WARNING}Rate Limit Exceeded on {func.__name__} (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...{COLOR_RESET}")
            time.sleep(delay)
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
            delay = base_delay * (2 ** attempt) * (1 + np.random.rand() * 0.5) # Add jitter
            if attempt == max_retries:
                lg.error(f"{COLOR_ERROR}API call {func.__name__} failed after {max_retries} retries due to {type(e).__name__}: {e}{COLOR_RESET}")
                return None
            lg.warning(f"{COLOR_WARNING}Retryable {type(e).__name__} on {func.__name__} (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...{COLOR_RESET}")
            time.sleep(delay)
        except ccxt.ExchangeError as e:
             # Catch other specific exchange errors that might be retryable or need specific handling
             # Example: Bybit maintenance errors often have specific codes
             http_status = getattr(e, 'http_status', None)
             if http_status in RETRYABLE_HTTP_CODES:
                 delay = base_delay * (2 ** attempt) * (1 + np.random.rand() * 0.5)
                 if attempt == max_retries:
                     lg.error(f"{COLOR_ERROR}API call {func.__name__} failed with status {http_status} after {max_retries} retries: {e}{COLOR_RESET}")
                     return None
                 lg.warning(f"{COLOR_WARNING}Retryable HTTP status {http_status} on {func.__name__} (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f}s...{COLOR_RESET}")
                 time.sleep(delay)
             else:
                 # Non-retryable exchange error (e.g., Insufficient Funds, Invalid Order)
                 lg.error(f"{COLOR_ERROR}Non-retryable Exchange Error on {func.__name__}: {e}{COLOR_RESET}", exc_info=False) # Set exc_info=False for cleaner logs usually
                 return None
        except Exception as e:
            # Catch any other unexpected error during the API call
            lg.error(f"{COLOR_ERROR}Unexpected Error during API call {func.__name__}: {e}{COLOR_RESET}", exc_info=True)
            return None
    return None # Should not be reached, but ensures a return path


# --- Data Fetching Functions ---
def fetch_balance(exchange: ccxt.Exchange, quote_currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for the specified quote currency using safe_api_call.

    Args:
        exchange: Initialized CCXT exchange object.
        quote_currency: The currency symbol (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        Available balance as a Decimal, or None if fetch fails.
    """
    lg = logger
    try:
        # Bybit V5 might require specifying account type if not using Unified
        # params = {'accountType': 'CONTRACT'} # Or 'UNIFIED' or 'SPOT'
        balance_data = safe_api_call(exchange.fetch_balance, lg) # Add params=params if needed
        if balance_data:
            # Accessing balance might differ slightly based on account type (Contract/Unified)
            # Check 'info' for more detailed structure if needed
            # For Linear Perpetual (often under CONTRACT or UNIFIED account)
            if quote_currency in balance_data:
                 # Try top level first
                 free_balance_str = str(balance_data[quote_currency].get('free', 0))
            elif 'info' in balance_data and isinstance(balance_data['info'].get('result', {}).get('list'), list):
                 # Check V5 structure result -> list -> coin -> walletBalance/availableToWithdraw
                 acc_list = balance_data['info']['result']['list']
                 found_balance = '0'
                 for item in acc_list:
                      if item.get('coin') == quote_currency:
                           # 'availableToWithdraw' is generally safer for trading capital
                           found_balance = item.get('availableToWithdraw', '0')
                           if Decimal(found_balance) > MIN_DECIMAL_VALUE: break
                           # Fallback to walletBalance if availableToWithdraw is 0
                           found_balance = item.get('walletBalance', '0')
                           break
                 free_balance_str = found_balance
            else:
                 # Fallback or older structures
                 free_balance_str = str(balance_data.get('free', {}).get(quote_currency, 0))

            free_balance = Decimal(free_balance_str)
            if free_balance.is_finite():
                # lg.debug(f"Fetched balance: {free_balance} {quote_currency}")
                return free_balance
            else:
                lg.error(f"Invalid balance value received: {free_balance_str}")
                return None
        else:
            lg.warning("fetch_balance call returned no data.")
            return None
    except (InvalidOperation, TypeError) as e:
        lg.error(f"Error converting balance data to Decimal: {e}", exc_info=True)
    except Exception as e:
        lg.error(f"Unexpected error fetching balance: {e}", exc_info=True)
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (Kline) data using safe_api_call and returns it as a Pandas DataFrame.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h').
        limit: Number of candles to fetch.
        logger: Logger instance.

    Returns:
        A Pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        and Decimal values, or an empty DataFrame on failure.
    """
    lg = logger
    # lg.debug(f"Fetching {limit} klines for {symbol} ({timeframe})...")
    try:
        # Bybit V5 uses 'category': 'linear' or 'inverse' or 'spot'
        params = {'category': 'linear'} if 'bybit' in exchange.id.lower() else {}
        klines = safe_api_call(exchange.fetch_ohlcv, lg, symbol, timeframe, limit=limit, params=params)

        if klines and isinstance(klines, list) and len(klines) > 0:
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp to datetime objects (optional, depends on usage)
            # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            # Convert financial columns to Decimal for precision
            for col in ['open', 'high', 'low', 'close', 'volume']:
                # Handle potential None values or invalid strings before conversion
                 df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))
            df = df.dropna() # Remove rows with NaN if any conversion failed
            lg.debug(f"Successfully fetched {len(df)} klines for {symbol}.")
            return df
        elif klines is None:
             lg.warning(f"Kline fetch for {symbol} returned None after retries.")
             return pd.DataFrame() # Return empty df if API call failed
        else:
            lg.warning(f"Kline fetch for {symbol} returned empty or invalid data: {klines}")
            return pd.DataFrame()
    except (ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Error converting kline data to Decimal/DataFrame for {symbol}: {e}", exc_info=True)
    except Exception as e:
        lg.error(f"Unexpected error fetching klines for {symbol}: {e}", exc_info=True)
    return pd.DataFrame() # Return empty df on any exception

def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current ticker price (last trade price) using safe_api_call.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol.
        logger: Logger instance.

    Returns:
        The current price as a Decimal, or None if fetch fails.
    """
    lg = logger
    # lg.debug(f"Fetching current price for {symbol}...")
    try:
        # Bybit V5 uses 'category' param
        params = {'category': 'linear'} if 'bybit' in exchange.id.lower() else {}
        ticker = safe_api_call(exchange.fetch_ticker, lg, symbol, params=params)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            price = Decimal(str(ticker['last']))
            if price.is_finite() and price > 0:
                # lg.debug(f"Fetched current price for {symbol}: {price}")
                return price
            else:
                lg.warning(f"Invalid 'last' price received for {symbol}: {ticker['last']}")
                return None
        else:
            lg.warning(f"Could not fetch valid ticker or 'last' price for {symbol}. Ticker: {ticker}")
            return None
    except (InvalidOperation, TypeError) as e:
        lg.error(f"Error converting ticker price to Decimal for {symbol}: {e}", exc_info=True)
    except Exception as e:
        lg.error(f"Unexpected error fetching current price for {symbol}: {e}", exc_info=True)
    return None

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict[str, List[Tuple[Decimal, Decimal]]]]:
    """
    Fetches the order book (bids and asks) using safe_api_call.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol.
        limit: Number of price levels to fetch (max depends on exchange, e.g., Bybit V5 L2 is 200).
        logger: Logger instance.

    Returns:
        A dictionary with 'bids' and 'asks' lists, containing [price, volume] tuples
        as Decimals, or None if fetch fails.
    """
    lg = logger
    # lg.debug(f"Fetching order book for {symbol} (limit: {limit})...")
    try:
        # Bybit V5 uses 'category' param
        params = {'category': 'linear'} if 'bybit' in exchange.id.lower() else {}
        orderbook = safe_api_call(exchange.fetch_order_book, lg, symbol, limit=limit, params=params)

        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            # Convert prices and volumes to Decimal
            bids = [(Decimal(str(price)), Decimal(str(volume))) for price, volume in orderbook['bids']]
            asks = [(Decimal(str(price)), Decimal(str(volume))) for price, volume in orderbook['asks']]
            # lg.debug(f"Successfully fetched order book for {symbol} with {len(bids)} bids and {len(asks)} asks.")
            return {'bids': bids, 'asks': asks}
        else:
            lg.warning(f"Order book fetch for {symbol} returned invalid data: {orderbook}")
            return None
    except (InvalidOperation, TypeError) as e:
        lg.error(f"Error converting order book data to Decimal for {symbol}: {e}", exc_info=True)
    except Exception as e:
        lg.error(f"Unexpected error fetching order book for {symbol}: {e}", exc_info=True)
    return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Retrieves detailed market information (precision, limits, contract size) for a symbol.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol.
        logger: Logger instance.

    Returns:
        A dictionary containing market details, or None if the symbol is not found
        or essential information is missing.
    """
    lg = logger
    try:
        # Ensure markets are loaded (should be done during init, but double-check)
        if not exchange.markets:
            lg.warning("Markets not loaded, attempting to load now...")
            exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
            lg.error(f"Symbol '{symbol}' not found in loaded markets.")
            return None

        market = exchange.market(symbol)

        # --- Essential Information Validation ---
        if not market:
            lg.error(f"Could not retrieve market data for symbol '{symbol}' from ccxt.")
            return None

        precision = market.get('precision')
        limits = market.get('limits')

        if not precision or not isinstance(precision, dict) or 'price' not in precision or 'amount' not in precision:
             lg.error(f"Market precision information missing or invalid for {symbol}. Precision: {precision}")
             return None
        if not limits or not isinstance(limits, dict) or 'amount' not in limits or 'price' not in limits or 'cost' not in limits:
             lg.error(f"Market limits information missing or invalid for {symbol}. Limits: {limits}")
             return None
        if limits['amount'].get('min') is None or limits['price'].get('min') is None:
             lg.error(f"Minimum amount/price limits missing for {symbol}. Limits: {limits}")
             return None

        # --- Extract Information ---
        market_info = {
            'id': market.get('id'), # Exchange-specific ID (e.g., BTCUSDT for Bybit)
            'symbol': market.get('symbol'), # Standardized symbol (e.g., BTC/USDT:USDT)
            'base': market.get('base'), # Base currency (e.g., BTC)
            'quote': market.get('quote'), # Quote currency (e.g., USDT)
            'settle': market.get('settle'), # Settle currency (e.g., USDT for linear)
            'precision': { # Store precision values as strings initially
                'price': str(precision.get('price')),   # Price tick size
                'amount': str(precision.get('amount')), # Amount step size
            },
            'limits': { # Store limits as strings initially
                'amount': {
                    'min': str(limits['amount'].get('min')),
                    'max': str(limits['amount'].get('max')),
                },
                'price': {
                    'min': str(limits['price'].get('min')),
                    'max': str(limits['price'].get('max')),
                },
                'cost': { # Minimum and maximum order value (price * amount)
                    'min': str(limits['cost'].get('min')),
                    'max': str(limits['cost'].get('max')),
                },
            },
            # Contract specific details
            'contract': market.get('contract', False), # Is it a futures/swap contract?
            'linear': market.get('linear', False), # Is it linear (vs inverse)?
            'contractSize': str(market.get('contractSize', '1')), # Value of 1 contract
        }

        # --- Log Retrieved Info ---
        lg.debug(f"Market Info for {symbol}:")
        lg.debug(f"  ID: {market_info['id']}, Base: {market_info['base']}, Quote: {market_info['quote']}, Settle: {market_info['settle']}")
        lg.debug(f"  Type: {'Linear Contract' if market_info['linear'] else 'Inverse Contract' if market_info['contract'] else 'Spot'}")
        lg.debug(f"  Contract Size: {market_info['contractSize']}")
        lg.debug(f"  Precision: Price Tick={market_info['precision']['price']}, Amount Step={market_info['precision']['amount']}")
        lg.debug(f"  Limits: Amount Min={market_info['limits']['amount']['min']}, Price Min={market_info['limits']['price']['min']}, Cost Min={market_info['limits']['cost']['min']}")

        return market_info

    except Exception as e:
        lg.error(f"Unexpected error fetching market info for {symbol}: {e}", exc_info=True)
        return None


# --- Trading Analysis ---
class TradingAnalyzer:
    """
    Analyzes market data (OHLCV) using technical indicators and generates trading signals.
    Provides helper methods for precision and formatting based on market info.
    """
    def __init__(self, klines_df: pd.DataFrame, logger: logging.Logger, config_dict: Dict[str, Any], market_info: Dict[str, Any]):
        """
        Initializes the analyzer with kline data, config, and market information.

        Args:
            klines_df: Pandas DataFrame containing OHLCV data.
            logger: Logger instance.
            config_dict: Configuration dictionary.
            market_info: Market information dictionary from get_market_info.
        """
        if klines_df.empty:
            logger.warning("TradingAnalyzer initialized with empty klines DataFrame.")
        self.df = klines_df.copy() # Work on a copy to avoid modifying original data
        self.lg = logger
        self.config = config_dict
        self.market_info = market_info
        self.indicator_values: Dict[str, Optional[Decimal]] = {} # Store latest indicator values as Decimals

        # Pre-calculate precision values for frequent use
        self.min_tick_size = self._calculate_min_tick_size()
        self.min_amount_step = self._calculate_min_amount_step()
        self.price_precision_places = self._calculate_price_precision_places()
        self.amount_precision_places = self._calculate_amount_precision_places()

        if not self.df.empty:
            self.calculate_indicators()
        else:
            self.lg.warning("Skipping indicator calculation due to empty DataFrame.")


    # --- Precision and Formatting Helpers ---

    def _calculate_min_tick_size(self) -> Decimal:
        """Calculates the minimum price tick size from market info."""
        try:
            tick_str = self.market_info['precision']['price']
            tick = Decimal(tick_str)
            if tick.is_finite() and tick > 0:
                return tick
            else:
                self.lg.warning(f"Invalid price tick size '{tick_str}', using default 0.0001.")
                return Decimal('0.0001')
        except (KeyError, InvalidOperation, TypeError):
            self.lg.warning("Could not determine price tick size, using default 0.0001.", exc_info=True)
            return Decimal('0.0001')

    def _calculate_min_amount_step(self) -> Decimal:
        """Calculates the minimum amount/quantity step size from market info."""
        try:
            step_str = self.market_info['precision']['amount']
            step = Decimal(step_str)
            if step.is_finite() and step > 0:
                return step
            else:
                self.lg.warning(f"Invalid amount step size '{step_str}', using default 0.001.")
                return Decimal('0.001')
        except (KeyError, InvalidOperation, TypeError):
            self.lg.warning("Could not determine amount step size, using default 0.001.", exc_info=True)
            return Decimal('0.001')

    def _calculate_price_precision_places(self) -> int:
        """Calculates the number of decimal places for price formatting."""
        tick = self.min_tick_size
        # The exponent of a Decimal gives the number of digits after the decimal point
        return abs(tick.as_tuple().exponent) if tick.is_finite() else 4 # Default precision

    def _calculate_amount_precision_places(self) -> int:
        """Calculates the number of decimal places for amount/quantity formatting."""
        step = self.min_amount_step
        return abs(step.as_tuple().exponent) if step.is_finite() else 3 # Default precision

    def format_price(self, price: Union[Decimal, float, str]) -> Decimal:
        """Formats a price to the market's required tick size (precision)."""
        try:
            price_dec = Decimal(str(price))
            # Quantize rounds to the nearest multiple of the tick size
            return price_dec.quantize(self.min_tick_size, rounding=ROUND_DOWN) # Use ROUND_DOWN or ROUND_HALF_UP depending on need
        except (InvalidOperation, TypeError):
            self.lg.error(f"Could not format invalid price value: {price}", exc_info=True)
            return Decimal('NaN')

    def format_amount(self, amount: Union[Decimal, float, str]) -> Decimal:
        """Formats an amount/quantity to the market's required step size (precision)."""
        try:
            amount_dec = Decimal(str(amount))
            # Quantize rounds *down* to the nearest multiple of the step size
            return (amount_dec // self.min_amount_step) * self.min_amount_step
        except (InvalidOperation, TypeError):
            self.lg.error(f"Could not format invalid amount value: {amount}", exc_info=True)
            return Decimal('NaN')


    # --- Indicator Calculation ---

    def calculate_indicators(self) -> None:
        """Calculates all enabled technical indicators and stores the latest value."""
        if self.df.empty:
            self.lg.warning("Cannot calculate indicators: DataFrame is empty.")
            return

        indicators_to_run = self.config.get("indicators", {})
        self.lg.debug("Calculating indicators...")
        # Make sure essential columns are numeric Decimals
        for col in ['open', 'high', 'low', 'close', 'volume']:
             if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                  self.lg.error(f"Column '{col}' is missing or not numeric in DataFrame. Cannot calculate indicators.")
                  return # Stop calculation if data is bad

        # Always calculate ATR as it's needed for risk management
        self._calculate_atr()

        # Calculate indicators based on config flags
        # Wrap individual calculations in try-except for resilience
        if indicators_to_run.get("ema_alignment"): self._safe_indicator_calc(self._calculate_ema)
        if indicators_to_run.get("momentum"): self._safe_indicator_calc(self._calculate_momentum)
        if indicators_to_run.get("volume_confirmation"): self._safe_indicator_calc(self._calculate_volume_ma)
        if indicators_to_run.get("stoch_rsi"): self._safe_indicator_calc(self._calculate_stoch_rsi)
        if indicators_to_run.get("rsi"): self._safe_indicator_calc(self._calculate_rsi)
        if indicators_to_run.get("bollinger_bands"): self._safe_indicator_calc(self._calculate_bbands)
        if indicators_to_run.get("vwap"): self._safe_indicator_calc(self._calculate_vwap)
        if indicators_to_run.get("cci"): self._safe_indicator_calc(self._calculate_cci)
        if indicators_to_run.get("wr"): self._safe_indicator_calc(self._calculate_williams_r)
        if indicators_to_run.get("psar"): self._safe_indicator_calc(self._calculate_psar)
        if indicators_to_run.get("sma_10"): self._safe_indicator_calc(self._calculate_sma)
        if indicators_to_run.get("mfi"): self._safe_indicator_calc(self._calculate_mfi)

        # Log calculated values for debugging (optional)
        # self.lg.debug(f"Calculated Indicators: {self.indicator_values}")

    def _safe_indicator_calc(self, calculation_func: callable):
        """Safely executes an indicator calculation function."""
        try:
            calculation_func()
        except Exception as e:
            self.lg.error(f"Error calculating indicator {calculation_func.__name__}: {e}", exc_info=True)

    # --- Individual Indicator Calculation Methods ---
    # These methods calculate the indicator using pandas_ta and store the *latest* value
    # in self.indicator_values as a Decimal.

    def _calculate_ema(self) -> None:
        short_period = self.config["ema_short_period"]
        long_period = self.config["ema_long_period"]
        short_ema = ta.ema(self.df['close'], length=short_period)
        long_ema = ta.ema(self.df['close'], length=long_period)
        if short_ema is not None and not short_ema.empty and long_ema is not None and not long_ema.empty:
            self.indicator_values["EMA_short"] = Decimal(str(short_ema.iloc[-1])).normalize() if pd.notna(short_ema.iloc[-1]) else None
            self.indicator_values["EMA_long"] = Decimal(str(long_ema.iloc[-1])).normalize() if pd.notna(long_ema.iloc[-1]) else None

    def _calculate_momentum(self) -> None:
        mom = ta.momentum(self.df['close'], length=self.config["momentum_period"])
        if mom is not None and not mom.empty:
            self.indicator_values["Momentum"] = Decimal(str(mom.iloc[-1])).normalize() if pd.notna(mom.iloc[-1]) else None

    def _calculate_volume_ma(self) -> None:
        vol_ma = ta.sma(self.df['volume'], length=self.config["volume_ma_period"])
        if vol_ma is not None and not vol_ma.empty:
            self.indicator_values["Volume_MA"] = Decimal(str(vol_ma.iloc[-1])).normalize() if pd.notna(vol_ma.iloc[-1]) else None
        # Store current volume as well
        self.indicator_values["Volume"] = self.df['volume'].iloc[-1].normalize() if pd.notna(self.df['volume'].iloc[-1]) else None


    def _calculate_stoch_rsi(self) -> None:
        stochrsi_df = ta.stochrsi(
            self.df['close'],
            length=self.config["stoch_rsi_period"],
            rsi_length=self.config["stoch_rsi_rsi_period"],
            k=self.config["stoch_rsi_k_period"],
            d=self.config["stoch_rsi_d_period"]
        )
        # pandas-ta returns a DataFrame, we usually want the %K or %D line
        # Using STOCHRSIk_% as the faster line
        if stochrsi_df is not None and not stochrsi_df.empty and f'STOCHRSIk_{self.config["stoch_rsi_period"]}_{self.config["stoch_rsi_rsi_period"]}_{self.config["stoch_rsi_k_period"]}' in stochrsi_df.columns:
             k_col = f'STOCHRSIk_{self.config["stoch_rsi_period"]}_{self.config["stoch_rsi_rsi_period"]}_{self.config["stoch_rsi_k_period"]}'
             self.indicator_values["StochRSI_K"] = Decimal(str(stochrsi_df[k_col].iloc[-1])).normalize() if pd.notna(stochrsi_df[k_col].iloc[-1]) else None
             # Optionally store D line as well if needed for crossover signals
             # d_col = f'STOCHRSId_{self.config["stoch_rsi_period"]}_{self.config["stoch_rsi_rsi_period"]}_{self.config["stoch_rsi_d_period"]}'
             # self.indicator_values["StochRSI_D"] = Decimal(str(stochrsi_df[d_col].iloc[-1])) if pd.notna(stochrsi_df[d_col].iloc[-1]) else None


    def _calculate_rsi(self) -> None:
        rsi = ta.rsi(self.df['close'], length=self.config["rsi_period"])
        if rsi is not None and not rsi.empty:
            self.indicator_values["RSI"] = Decimal(str(rsi.iloc[-1])).normalize() if pd.notna(rsi.iloc[-1]) else None

    def _calculate_bbands(self) -> None:
        bbands_df = ta.bbands(
            self.df['close'],
            length=self.config["bollinger_bands_period"],
            std=self.config["bollinger_bands_std_dev"]
        )
        if bbands_df is not None and not bbands_df.empty:
            # Column names might vary slightly with pandas_ta versions, adjust if needed
            upper_col, middle_col, lower_col = f'BBU_{self.config["bollinger_bands_period"]}_{float(self.config["bollinger_bands_std_dev"])}', f'BBM_{self.config["bollinger_bands_period"]}_{float(self.config["bollinger_bands_std_dev"])}', f'BBL_{self.config["bollinger_bands_period"]}_{float(self.config["bollinger_bands_std_dev"])}'
            if upper_col in bbands_df: self.indicator_values["BB_upper"] = Decimal(str(bbands_df[upper_col].iloc[-1])).normalize() if pd.notna(bbands_df[upper_col].iloc[-1]) else None
            if lower_col in bbands_df: self.indicator_values["BB_lower"] = Decimal(str(bbands_df[lower_col].iloc[-1])).normalize() if pd.notna(bbands_df[lower_col].iloc[-1]) else None
            if middle_col in bbands_df: self.indicator_values["BB_middle"] = Decimal(str(bbands_df[middle_col].iloc[-1])).normalize() if pd.notna(bbands_df[middle_col].iloc[-1]) else None


    def _calculate_vwap(self) -> None:
        vwap = ta.vwap(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
        if vwap is not None and not vwap.empty:
            self.indicator_values["VWAP"] = Decimal(str(vwap.iloc[-1])).normalize() if pd.notna(vwap.iloc[-1]) else None

    def _calculate_cci(self) -> None:
        cci = ta.cci(self.df['high'], self.df['low'], self.df['close'], length=self.config["cci_period"])
        if cci is not None and not cci.empty:
            self.indicator_values["CCI"] = Decimal(str(cci.iloc[-1])).normalize() if pd.notna(cci.iloc[-1]) else None

    def _calculate_williams_r(self) -> None:
        wr = ta.willr(self.df['high'], self.df['low'], self.df['close'], length=self.config["williams_r_period"])
        if wr is not None and not wr.empty:
            self.indicator_values["Williams_R"] = Decimal(str(wr.iloc[-1])).normalize() if pd.notna(wr.iloc[-1]) else None

    def _calculate_psar(self) -> None:
        psar_df = ta.psar(
            self.df['high'],
            self.df['low'],
            af=self.config["psar_step"],
            max_af=self.config["psar_max_step"]
        )
        if psar_df is not None and not psar_df.empty:
             # PSAR returns multiple columns, we need the latest non-NaN value which indicates the current trend's stop level
             # Check 'PSARl' (long stop) and 'PSARs' (short stop) columns
             last_psar_l = psar_df['PSARl'].iloc[-1]
             last_psar_s = psar_df['PSARs'].iloc[-1]
             latest_psar_val = None
             if pd.notna(last_psar_l): latest_psar_val = last_psar_l
             elif pd.notna(last_psar_s): latest_psar_val = last_psar_s

             if latest_psar_val is not None:
                 self.indicator_values["PSAR"] = Decimal(str(latest_psar_val)).normalize()
             else:
                 self.indicator_values["PSAR"] = None # If both are NaN


    def _calculate_sma(self) -> None:
        sma = ta.sma(self.df['close'], length=self.config["sma_10_period"])
        if sma is not None and not sma.empty:
            self.indicator_values["SMA_10"] = Decimal(str(sma.iloc[-1])).normalize() if pd.notna(sma.iloc[-1]) else None

    def _calculate_mfi(self) -> None:
        mfi = ta.mfi(self.df['high'], self.df['low'], self.df['close'], self.df['volume'], length=self.config["mfi_period"])
        if mfi is not None and not mfi.empty:
            self.indicator_values["MFI"] = Decimal(str(mfi.iloc[-1])).normalize() if pd.notna(mfi.iloc[-1]) else None

    def _calculate_atr(self) -> None:
        """Calculates Average True Range (ATR)."""
        atr = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=self.config["atr_period"])
        if atr is not None and not atr.empty:
            value = Decimal(str(atr.iloc[-1])).normalize() if pd.notna(atr.iloc[-1]) else None
            # Ensure ATR is positive
            if value is not None and value > 0:
                self.indicator_values["ATR"] = value
            elif value is not None:
                 self.lg.warning(f"Calculated ATR is not positive ({value}), setting to None.")
                 self.indicator_values["ATR"] = None
            else:
                 self.indicator_values["ATR"] = None
        else:
             self.indicator_values["ATR"] = None


    # --- Signal Component Checkers ---
    # These methods evaluate the condition of a single indicator or factor
    # and return a score: 1.0 for bullish, -1.0 for bearish, 0.0 for neutral.

    def _check_ema_alignment(self) -> float:
        """Checks if short EMA is above/below long EMA."""
        short_ema = self.indicator_values.get("EMA_short")
        long_ema = self.indicator_values.get("EMA_long")
        if short_ema is not None and long_ema is not None:
            if short_ema > long_ema: return 1.0
            if short_ema < long_ema: return -1.0
        return 0.0 # Neutral if EMAs are equal or missing

    def _check_momentum(self) -> float:
        """Checks if momentum is positive or negative."""
        mom = self.indicator_values.get("Momentum")
        if mom is not None:
            if mom > 0: return 1.0
            if mom < 0: return -1.0
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Checks if current volume is significantly above its moving average."""
        vol = self.indicator_values.get("Volume")
        vol_ma = self.indicator_values.get("Volume_MA")
        if vol is not None and vol_ma is not None and vol_ma > 0: # Avoid division by zero
            multiplier = Decimal(str(self.config["volume_confirmation_multiplier"]))
            if vol > vol_ma * multiplier: return 1.0 # Strong volume confirmation
            # Optional: Could add a negative score if volume is very low, but currently only checks for high volume
            # if vol < vol_ma * Decimal('0.5'): return -0.5 # Example: weak confirmation
        return 0.0 # Neutral otherwise

    def _check_stoch_rsi(self) -> float:
        """Checks if StochRSI (%K line) is oversold or overbought."""
        stoch_rsi_k = self.indicator_values.get("StochRSI_K")
        if stoch_rsi_k is not None:
            oversold = Decimal(str(self.config["stoch_rsi_oversold_threshold"]))
            overbought = Decimal(str(self.config["stoch_rsi_overbought_threshold"]))
            if stoch_rsi_k < oversold: return 1.0 # Oversold suggests potential upward reversal (buy signal)
            if stoch_rsi_k > overbought: return -1.0 # Overbought suggests potential downward reversal (sell signal)
        return 0.0

    def _check_rsi(self) -> float:
        """Checks if RSI is oversold (<30) or overbought (>70)."""
        rsi = self.indicator_values.get("RSI")
        if rsi is not None:
            if rsi < 30: return 1.0 # Oversold (buy signal)
            if rsi > 70: return -1.0 # Overbought (sell signal)
        return 0.0

    def _check_bollinger_bands(self, current_price: Decimal) -> float:
        """Checks if price is near the upper/lower Bollinger Band."""
        upper = self.indicator_values.get("BB_upper")
        lower = self.indicator_values.get("BB_lower")
        if upper is not None and lower is not None and current_price.is_finite():
            # Touching or below lower band is a buy signal (mean reversion)
            if current_price <= lower: return 1.0
            # Touching or above upper band is a sell signal (mean reversion)
            if current_price >= upper: return -1.0
        return 0.0

    def _check_vwap(self, current_price: Decimal) -> float:
        """Checks if price is above or below VWAP."""
        vwap = self.indicator_values.get("VWAP")
        if vwap is not None and current_price.is_finite():
            if current_price > vwap: return 1.0 # Above VWAP is generally bullish intraday
            if current_price < vwap: return -1.0 # Below VWAP is generally bearish intraday
        return 0.0

    def _check_cci(self) -> float:
        """Checks if CCI is below -100 (oversold) or above +100 (overbought)."""
        cci = self.indicator_values.get("CCI")
        if cci is not None:
            if cci < -100: return 1.0 # Oversold condition (buy signal)
            if cci > 100: return -1.0 # Overbought condition (sell signal)
        return 0.0

    def _check_williams_r(self) -> float:
        """Checks if Williams %R is below -80 (oversold) or above -20 (overbought)."""
        wr = self.indicator_values.get("Williams_R")
        if wr is not None:
            if wr <= -80: return 1.0 # Oversold (buy signal)
            if wr >= -20: return -1.0 # Overbought (sell signal)
        return 0.0

    def _check_psar(self, current_price: Decimal) -> float:
        """Checks if price is above or below the Parabolic SAR."""
        psar = self.indicator_values.get("PSAR")
        if psar is not None and current_price.is_finite():
            if current_price > psar: return 1.0 # Price above SAR is bullish trend
            if current_price < psar: return -1.0 # Price below SAR is bearish trend
        return 0.0

    def _check_sma(self, current_price: Decimal) -> float:
        """Checks if price is above or below the short-term SMA (e.g., SMA 10)."""
        sma = self.indicator_values.get("SMA_10")
        if sma is not None and current_price.is_finite():
            if current_price > sma: return 1.0 # Price above short-term MA is bullish
            if current_price < sma: return -1.0 # Price below short-term MA is bearish
        return 0.0

    def _check_mfi(self) -> float:
        """Checks if Money Flow Index is below 20 (oversold) or above 80 (overbought)."""
        mfi = self.indicator_values.get("MFI")
        if mfi is not None:
            if mfi < 20: return 1.0 # Oversold (buy signal)
            if mfi > 80: return -1.0 # Overbought (sell signal)
        return 0.0

    def _check_orderbook(self, orderbook_data: Optional[Dict[str, List[Tuple[Decimal, Decimal]]]], current_price: Decimal) -> float:
        """Analyzes immediate order book pressure (imbalance near current price)."""
        if orderbook_data and current_price.is_finite() and current_price > 0:
            try:
                # Define a small price range around the current price for analysis
                price_range_factor = Decimal('0.001') # e.g., 0.1% around the price
                bid_range_min = current_price * (Decimal('1') - price_range_factor)
                ask_range_max = current_price * (Decimal('1') + price_range_factor)

                # Sum volume within the defined range
                bid_vol_near = sum(vol for price, vol in orderbook_data['bids'] if price >= bid_range_min)
                ask_vol_near = sum(vol for price, vol in orderbook_data['asks'] if price <= ask_range_max)

                if bid_vol_near > 0 and ask_vol_near > 0:
                    imbalance_ratio = bid_vol_near / ask_vol_near
                    if imbalance_ratio > Decimal('1.5'): return 1.0 # Significantly more bid volume suggests buying pressure
                    if imbalance_ratio < Decimal('0.67'): return -1.0 # Significantly more ask volume suggests selling pressure
                elif bid_vol_near > 0 and ask_vol_near == 0:
                    return 1.0 # Only bids in range, strong buy pressure
                elif ask_vol_near > 0 and bid_vol_near == 0:
                    return -1.0 # Only asks in range, strong sell pressure

            except (ZeroDivisionError, InvalidOperation, TypeError) as e:
                self.lg.warning(f"Could not calculate order book imbalance: {e}")
        return 0.0 # Neutral if data missing or calculation fails

    # --- Signal Generation ---

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        """
        Generates a trading signal (BUY, SELL, HOLD) based on a weighted score of enabled indicators.

        Args:
            current_price: The current market price (used by some checks).
            orderbook_data: Fetched order book data (used by order book check).

        Returns:
            Signal string: "BUY", "SELL", or "HOLD".
        """
        # Check if essential data is available
        if self.df.empty or not current_price.is_finite() or current_price <= 0:
             self.lg.warning("Cannot generate signal: Missing klines or valid current price.")
             return HOLD_SIGNAL

        active_weight_set_name = self.config.get("active_weight_set", "default")
        weights = self.config.get("weight_sets", {}).get(active_weight_set_name)
        enabled_indicators = self.config.get("indicators", {})

        if not weights:
            self.lg.error(f"Active weight set '{active_weight_set_name}' not found in configuration. Using 0 weights.")
            weights = {} # Prevent crash, but signal will likely be HOLD

        scores: Dict[str, float] = {}
        total_score: float = 0.0
        active_factors: int = 0

        # Calculate score for each enabled indicator/factor
        if enabled_indicators.get("ema_alignment"): scores["ema_alignment"] = self._check_ema_alignment()
        if enabled_indicators.get("momentum"): scores["momentum"] = self._check_momentum()
        if enabled_indicators.get("volume_confirmation"): scores["volume_confirmation"] = self._check_volume_confirmation()
        if enabled_indicators.get("stoch_rsi"): scores["stoch_rsi"] = self._check_stoch_rsi()
        if enabled_indicators.get("rsi"): scores["rsi"] = self._check_rsi()
        if enabled_indicators.get("bollinger_bands"): scores["bollinger_bands"] = self._check_bollinger_bands(current
