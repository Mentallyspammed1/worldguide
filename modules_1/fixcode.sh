#!/bin/bash

# Script to update Python files with corrected versions by overwriting them.
# Run this script from the directory containing the .py files (e.g., modules_1).

# Define Colors for output (optional)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting file update process...${NC}"
echo "This script will overwrite existing files. Make sure you have backups!"
read -p "Press Enter to continue, or Ctrl+C to cancel."

# === Backup Function ===
backup_file() {
  local filename="$1"
  local backup_filename="${filename}.$(date +%Y%m%d_%H%M%S).bak"
  if [ -f "$filename" ]; then
    echo -e "${YELLOW}Backing up '$filename' to '$backup_filename'...${NC}"
    cp "$filename" "$backup_filename"
    if [ $? -ne 0 ]; then
      echo -e "${RED}ERROR: Failed to create backup for '$filename'. Aborting.${NC}"
      exit 1
    fi
  else
    echo -e "${YELLOW}File '$filename' not found, skipping backup (will create new file).${NC}"
  fi
}

# === File Content Definitions (Heredocs) ===

# --- utils.py Content ---
read -r -d '' UTILS_CONTENT <<'EOF_UTILS'
# File: utils.py
"""
Utility functions, constants, and shared classes for the trading bot.

Includes:
- Configuration constants (file paths, default values).
- API behavior constants (retries, delays).
- Trading constants (intervals, Fibonacci levels).
- Default indicator periods.
- Timezone handling (using zoneinfo or fallback).
- Custom logging formatter for redacting sensitive data.
- Helper functions for market data (precision, tick size).
- Color constants for terminal output.
- Formatting utilities (signals).
- Exponential backoff utility.
"""

import logging
import os
import datetime as dt
import time # Retained for potential use, though monotonic is often preferred for intervals
import random
from decimal import Decimal, getcontext, InvalidOperation
from typing import Any, Dict, Optional, Type, Union, List # Added List, Union

# Attempt to use zoneinfo (Python 3.9+), fallback to UTC if unavailable or error occurs
_ZoneInfo: Optional[Type[dt.tzinfo]] = None
_ZoneInfoNotFoundError: Optional[Type[Exception]] = None
try:
    from zoneinfo import ZoneInfo as _ZI, ZoneInfoNotFoundError as _ZINF
    _ZoneInfo = _ZI
    _ZoneInfoNotFoundError = _ZINF
    _zoneinfo_available = True
except ImportError:
    _zoneinfo_available = False
    # Fallback: Try importing pytz if zoneinfo is not available
    try:
        import pytz # type: ignore
        # Create a wrapper that mimics ZoneInfo constructor
        class PytzZoneInfoWrapper:
            def __init__(self, key: str):
                try: self._tz = pytz.timezone(key)
                except pytz.UnknownTimeZoneError: raise ZoneInfoNotFoundError(f"pytz: Unknown timezone '{key}'") from None # Mimic ZoneInfo error
                except Exception as e: raise ZoneInfoNotFoundError(f"pytz error for '{key}': {e}") from e
            def __getattr__(self, name): return getattr(self._tz, name) # Delegate methods
            def __str__(self): return self._tz.zone
        _ZoneInfo = PytzZoneInfoWrapper
        # Define a compatible error type for the except block in set_timezone
        class ZoneInfoNotFoundError(Exception): pass # Define if pytz is used
        _ZoneInfoNotFoundError = ZoneInfoNotFoundError
        print("Warning: 'zoneinfo' not found. Using 'pytz' for timezone support.", file=sys.stderr)
    except ImportError:
        print("Warning: Neither 'zoneinfo' nor 'pytz' found. Using basic UTC fallback only.", file=sys.stderr)
        # Define a basic UTC tzinfo class if both fail
        class _UTCFallback(dt.tzinfo):
            def utcoffset(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]: return dt.timedelta(0)
            def dst(self, d: Optional[dt.datetime]) -> Optional[dt.timedelta]: return dt.timedelta(0)
            def tzname(self, d: Optional[dt.datetime]) -> Optional[str]: return "UTC"
            def __repr__(self) -> str: return "<UTCFallback tzinfo>"
        _ZoneInfo = _UTCFallback
        # Define a dummy error class for the except block consistency
        class ZoneInfoNotFoundError(Exception): pass
        _ZoneInfoNotFoundError = ZoneInfoNotFoundError


# Try initializing Colorama safely
try:
    from colorama import Fore, Style, init
    init(autoreset=True) # Initialize Colorama
    # Color constants
    NEON_GREEN = Fore.LIGHTGREEN_EX; NEON_BLUE = Fore.LIGHTBLUE_EX; NEON_PURPLE = Fore.LIGHTMAGENTA_EX
    NEON_YELLOW = Fore.LIGHTYELLOW_EX; NEON_RED = Fore.LIGHTRED_EX; NEON_CYAN = Fore.LIGHTCYAN_EX
    RESET_ALL_STYLE = Style.RESET_ALL
except ImportError:
    print("Warning: 'colorama' not installed. Colored output will be disabled.", file=sys.stderr)
    # Define fallback empty strings if colorama is not available
    NEON_GREEN = NEON_BLUE = NEON_PURPLE = NEON_YELLOW = NEON_RED = NEON_CYAN = RESET_ALL_STYLE = ""

# --- Module-level logger ---
_module_logger = logging.getLogger(__name__) # Use dunder name for logger

# --- Decimal Context ---
try: getcontext().prec = 38
except Exception as e: _module_logger.error(f"Failed to set Decimal precision: {e}")

# --- Configuration Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE = "America/Chicago"

# --- API and Bot Behavior Constants ---
MAX_API_RETRIES = 3 # Default max retries (can be overridden by config)
RETRY_DELAY_SECONDS = 5.0 # Default base delay for non-rate-limit retries (use float)
MAX_RETRY_DELAY_SECONDS = 60.0 # Default max delay cap for exponential backoff (use float)
POSITION_CONFIRM_DELAY_SECONDS = 8.0 # Moved default here from config_loader for easier access

# --- Trading Constants ---
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
CCXT_INTERVAL_MAP = { "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "120": "2h", "240": "4h", "360": "6h", "720": "12h", "D": "1d", "W": "1w", "M": "1M" }
FIB_LEVELS = [ Decimal(str(f)) for f in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0] ]

# --- Indicator Default Periods (Centralized Source of Truth) ---
DEFAULT_INDICATOR_PERIODS = {
    "atr_period": 14, "cci_window": 20, "cci_constant": Decimal("0.015"),
    "williams_r_window": 14, "mfi_window": 14, "stoch_rsi_window": 14,
    "stoch_rsi_rsi_window": 14, "stoch_rsi_k": 3, "stoch_rsi_d": 3,
    "rsi_period": 14, "bollinger_bands_period": 20, "bollinger_bands_std_dev": Decimal("2.0"),
    "sma_10_window": 10, "ema_short_period": 9, "ema_long_period": 21,
    "momentum_period": 10, "volume_ma_period": 20, "fibonacci_window": 50,
    "psar_initial_af": Decimal("0.02"), "psar_af_step": Decimal("0.02"), "psar_max_af": Decimal("0.2"),
    # Thresholds/multipliers are now primarily defined in config_loader defaults for user tuning
    "default_atr_percentage_of_price": Decimal("0.01") # Fallback ATR % if calculation fails
}

# Global timezone object, lazily initialized by get_timezone()
_TIMEZONE: Optional[dt.tzinfo] = None

def _exponential_backoff(
    attempt: int,
    base_delay: float = RETRY_DELAY_SECONDS, # Use constant
    max_delay: float = MAX_RETRY_DELAY_SECONDS, # Use constant
    jitter: bool = True
) -> float:
    """Calculates exponential backoff delay with optional jitter and cap."""
    if attempt < 0: _module_logger.error("Attempt number must be non-negative."); return base_delay
    try:
        delay = base_delay * (2 ** attempt)
        if jitter: delay = random.uniform(delay * 0.7, delay * 1.3) # +/- 30% jitter
        return min(delay, max_delay)
    except OverflowError: _module_logger.warning(f"Exp backoff overflow attempt {attempt}. Using max_delay."); return max_delay

def set_timezone(tz_str: str) -> None:
    """Sets the global timezone object used by the application."""
    global _TIMEZONE
    if _ZoneInfo is None: _module_logger.error("No timezone implementation available."); _TIMEZONE = None; return
    try:
        _TIMEZONE = _ZoneInfo(tz_str)
        _module_logger.info(f"Timezone set using '{tz_str}'. Effective: {str(_TIMEZONE)}")
    except (_ZoneInfoNotFoundError, Exception) as tz_err: # Catch specific error + general fallback
        _module_logger.error(f"Error loading timezone '{tz_str}'. Using UTC. Error: {tz_err}", exc_info=(_ZoneInfoNotFoundError is None or not isinstance(tz_err, _ZoneInfoNotFoundError))) # Show traceback for unexpected errors
        try: _TIMEZONE = _ZoneInfo("UTC")
        except: _TIMEZONE = dt.timezone.utc # Absolute fallback

def get_timezone() -> dt.tzinfo:
    """Retrieves or initializes the global timezone object."""
    global _TIMEZONE
    if _TIMEZONE is None:
        chosen_tz = os.getenv("TIMEZONE", DEFAULT_TIMEZONE)
        _module_logger.info(f"Initializing timezone to '{chosen_tz}'...")
        set_timezone(chosen_tz)
        if _TIMEZONE is None: # If set_timezone failed critically
             _TIMEZONE = dt.timezone.utc; _module_logger.critical("Forced basic UTC timezone.")
    return _TIMEZONE

class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API key/secret strings."""
    _secrets_to_redact: List[Tuple[str, str]] = []

    @classmethod
    def set_sensitive_data(cls, *args: Optional[str]) -> None:
        """Registers sensitive strings for redaction."""
        cls._secrets_to_redact = []
        for s in args:
            if s and isinstance(s, str):
                placeholder = f"***{s.__class__.__name__.upper()[:3]}...***" # Generic placeholder
                cls._secrets_to_redact.append((s, placeholder))
        if cls._secrets_to_redact: _module_logger.debug(f"Sensitive data registered for redaction: {len(cls._secrets_to_redact)} items.")

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record, applying redaction."""
        formatted_message = super().format(record)
        temp_message = formatted_message
        try:
            secrets = getattr(SensitiveFormatter, '_secrets_to_redact', [])
            for original, placeholder in secrets:
                if original in temp_message: temp_message = temp_message.replace(original, placeholder)
            return temp_message
        except Exception as e: _module_logger.error(f"Error during log redaction: {e}"); return formatted_message

def get_price_precision(market_info: Dict[str, Any], logger: Optional[logging.Logger] = None) -> int:
    """Determines price precision (decimal places) from market info."""
    lg = logger or _module_logger; symbol = market_info.get('symbol', '?'); default_prec = 8
    try:
        prec_dict = market_info.get('precision', {}); price_prec = prec_dict.get('price')
        if isinstance(price_prec, int) and price_prec >= 0: return price_prec
        if price_prec is not None:
            tick_size = Decimal(str(price_prec))
            if tick_size > 0: return abs(tick_size.normalize().as_tuple().exponent)
        min_p_str = market_info.get('limits', {}).get('price', {}).get('min')
        if min_p_str:
            min_tick = Decimal(str(min_p_str))
            if min_tick > 0: return abs(min_tick.normalize().as_tuple().exponent)
        places = market_info.get('decimal_places') or market_info.get('price_decimals')
        if isinstance(places, int) and places >= 0: return places
    except Exception as e: lg.warning(f"Err get price prec {symbol}: {e}. Default {default_prec}.")
    return default_prec

def get_min_tick_size(market_info: Dict[str, Any], logger: Optional[logging.Logger] = None) -> Decimal:
    """Determines minimum price increment (tick size) as Decimal."""
    lg = logger or _module_logger; symbol = market_info.get('symbol', '?'); default_tick = Decimal('1e-8')
    try:
        prec_dict = market_info.get('precision', {}); price_prec = prec_dict.get('price')
        if isinstance(price_prec, (str, float)):
            tick_size = Decimal(str(price_prec));
            if tick_size > 0: return tick_size
        elif isinstance(price_prec, int) and price_prec >= 0:
            return Decimal('1e-' + str(price_prec))
        min_p_str = market_info.get('limits', {}).get('price', {}).get('min')
        if min_p_str:
            min_tick = Decimal(str(min_p_str));
            if min_tick > 0: return min_tick
        price_prec_places = get_price_precision(market_info, lg)
        return Decimal('1e-' + str(price_prec_places))
    except Exception as e: lg.warning(f"Err get tick size {symbol}: {e}. Default {default_tick}.")
    return default_tick

def format_signal(signal_text: Any, success: bool = True) -> str:
    """Formats trading signals or statuses with color."""
    signal_str = str(signal_text).upper(); color = RESET_ALL_STYLE
    if success:
        if signal_str == "BUY": color = NEON_GREEN
        elif signal_str == "SELL": color = NEON_RED
        elif signal_str == "HOLD": color = NEON_YELLOW
        elif signal_str in ["ACTIVE", "CONFIRMED", "OK"]: color = NEON_GREEN
        elif signal_str in ["PENDING", "WAITING"]: color = NEON_YELLOW
        else: color = NEON_CYAN
    else: color = NEON_RED
    return f"{color}{signal_text}{RESET_ALL_STYLE}"

# --- End of utils.py ---
EOF_UTILS

# --- config_loader.py Content ---
read -r -d '' CONFIG_LOADER_CONTENT <<'EOF_CONFIG_LOADER'
# File: config_loader.py
import json
import os
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Union

# Import constants and color codes from utils
try:
    from utils import (
        CONFIG_FILE, DEFAULT_INDICATOR_PERIODS,
        NEON_RED, NEON_YELLOW, RESET_ALL_STYLE,
        RETRY_DELAY_SECONDS, VALID_INTERVALS,
        POSITION_CONFIRM_DELAY_SECONDS # Import default value
    )
except ImportError:
    print("Warning: Failed import constants from utils.py.", file=sys.stderr)
    CONFIG_FILE = "config.json"; DEFAULT_INDICATOR_PERIODS = {}
    NEON_RED = NEON_YELLOW = RESET_ALL_STYLE = ""
    RETRY_DELAY_SECONDS = 5.0; VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
    POSITION_CONFIRM_DELAY_SECONDS = 8.0


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures default keys are present."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config

def _validate_numeric(key, cfg, default, min_v=None, max_v=None, is_int=False, allow_none=False):
    """Internal helper to validate numeric config values."""
    value = cfg.get(key); is_valid = False; corrected_value = default
    save_needed = False # Flag specific to this function

    if allow_none and value is None: is_valid = True; corrected_value = None
    elif isinstance(value, bool): pass # Invalid type
    elif isinstance(value, (int, float, Decimal, str)):
        try:
            num_val = Decimal(str(value))
            if is_int and num_val != num_val.to_integral_value(): raise ValueError("Must be int")
            if min_v is not None and num_val < Decimal(str(min_v)): raise ValueError(f"< min {min_v}")
            if max_v is not None and num_val > Decimal(str(max_v)): raise ValueError(f"> max {max_v}")
            # Format back to original type or keep Decimal
            if is_int: corrected_value = int(num_val)
            elif isinstance(default, float): corrected_value = float(num_val)
            else: corrected_value = num_val # Keep Decimal or original if default was None
            is_valid = True
        except (InvalidOperation, ValueError, TypeError): pass # Invalid format/range

    if not is_valid:
        print(f"{NEON_YELLOW}Config WARN: Invalid value '{value}' for '{key}'. Using default: '{default}'.{RESET_ALL_STYLE}")
        cfg[key] = default; save_needed = True
    elif cfg[key] != corrected_value: # Apply type correction if needed (e.g., "10.0" -> 10.0)
         # Only report if value changes significantly, not just type for same value like 10 vs 10.0
         if type(cfg[key]) != type(corrected_value) or cfg[key] != corrected_value :
             # print(f"{NEON_YELLOW}Config NOTE: Corrected type/value for '{key}': {cfg[key]} -> {corrected_value}{RESET_ALL_STYLE}") # Optional: Log type corrections
             cfg[key] = corrected_value; save_needed = True

    return save_needed # Return if save is needed due to this validation

def load_config(filepath: str = CONFIG_FILE) -> Dict[str, Any]:
    """Loads, validates, and ensures defaults in the configuration file."""
    default_config = {
        "exchange_id": "bybit", "default_market_type": "unified", "symbols_to_trade": ["BTC/USDT:USDT"],
        "interval": "5", "log_level": "INFO", "api_key": None, "api_secret": None, "use_sandbox": False,
        "max_api_retries": 3, "retry_delay": RETRY_DELAY_SECONDS, "api_timeout_ms": 15000,
        "market_cache_duration_seconds": 3600, "circuit_breaker_cooldown_seconds": 300,
        "order_rate_limit_per_second": 10.0, "enable_trading": False, "risk_per_trade": 0.01,
        "leverage": 10.0, "max_concurrent_positions": 1, "quote_currency": "USDT",
        "entry_order_type": "market", "limit_order_offset_buy": 0.0005, "limit_order_offset_sell": 0.0005,
        "order_confirmation_delay_seconds": 0.75, "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "position_confirm_retries": 3, "close_confirm_delay_seconds": 2.0,
        "protection_setup_timeout_seconds": 30.0, "limit_order_timeout_seconds": 300.0,
        "limit_order_poll_interval_seconds": 5.0, "limit_order_stale_timeout_seconds": 600.0,
        "adjust_limit_orders": False, "post_only": False, "time_in_force": "GTC",
        "enable_trailing_stop": True, "trailing_stop_distance_percent": 0.01,
        "trailing_stop_activation_offset_percent": 0.005, "tsl_activate_immediately_if_profitable": True,
        "enable_break_even": True, "break_even_trigger_atr_multiple": 1.0, "break_even_offset_ticks": 2,
        "time_based_exit_minutes": None, "stop_loss_multiple": 1.5, "take_profit_multiple": 2.0,
        "signal_score_threshold": 0.7, "kline_limit": 500, "min_kline_length": 100, "orderbook_limit": 25,
        "min_active_indicators_for_signal": 7, "indicator_buffer_candles": 20,
        "indicators": {k: True for k in DEFAULT_INDICATOR_PERIODS}, # Dynamic based on defaults
        "weight_sets": { "default": { "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1, "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3, "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1 } },
        "active_weight_set": "default",
        "exchange_options": {"options": {}},
        "market_load_params": {}, "balance_fetch_params": {}, "fetch_positions_params": {},
        "create_order_params": {}, "edit_order_params": {}, "cancel_order_params": {},
        "cancel_all_orders_params": {}, "fetch_order_params": {}, "fetch_open_orders_params": {},
        "fetch_closed_orders_params": {}, "fetch_my_trades_params": {}, "set_leverage_params": {},
        "set_trading_stop_params": {}, "set_position_mode_params": {},
        "library_log_levels": {"ccxt": "INFO"}, # Example: quiet ccxt debug by default
        **DEFAULT_INDICATOR_PERIODS # Add defaults directly
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath,"w",encoding="utf-8") as f: json.dump(default_config,f,indent=4,ensure_ascii=False,default=str) # Handle Decimal default
            print(f"{NEON_YELLOW}Created default config: {filepath}{RESET_ALL_STYLE}")
            return default_config
        except IOError as e: print(f"{NEON_RED}Error creating default config: {e}{RESET_ALL_STYLE}"); return default_config

    try:
        with open(filepath, encoding="utf-8") as f: config_from_file = json.load(f)
        updated_config = _ensure_config_keys(config_from_file, default_config)
        save_needed = (updated_config != config_from_file)

        # --- Validation ---
        save_needed |= _validate_numeric("max_api_retries", updated_config, default_config["max_api_retries"], min_v=0, is_int=True)
        save_needed |= _validate_numeric("retry_delay", updated_config, default_config["retry_delay"], min_v=0)
        save_needed |= _validate_numeric("api_timeout_ms", updated_config, default_config["api_timeout_ms"], min_v=1000, is_int=True)
        save_needed |= _validate_numeric("risk_per_trade", updated_config, default_config["risk_per_trade"], min_v=0, max_v=1)
        save_needed |= _validate_numeric("leverage", updated_config, default_config["leverage"], min_v=0) # Allow 0 leverage (e.g., for spot or cross)
        save_needed |= _validate_numeric("max_concurrent_positions", updated_config, default_config["max_concurrent_positions"], min_v=1, is_int=True)
        save_needed |= _validate_numeric("signal_score_threshold", updated_config, default_config["signal_score_threshold"], min_v=0)
        save_needed |= _validate_numeric("orderbook_limit", updated_config, default_config["orderbook_limit"], min_v=1, is_int=True)
        save_needed |= _validate_numeric("kline_limit", updated_config, default_config["kline_limit"], min_v=10, is_int=True)
        save_needed |= _validate_numeric("min_kline_length", updated_config, default_config["min_kline_length"], min_v=10, is_int=True) # Reasonable min
        save_needed |= _validate_numeric("position_confirm_delay_seconds", updated_config, default_config["position_confirm_delay_seconds"], min_v=0)
        save_needed |= _validate_numeric("time_based_exit_minutes", updated_config, default_config["time_based_exit_minutes"], min_v=1, allow_none=True)
        save_needed |= _validate_numeric("trailing_stop_distance_percent", updated_config, default_config["trailing_stop_distance_percent"], min_v=1e-9)
        save_needed |= _validate_numeric("trailing_stop_activation_offset_percent", updated_config, default_config["trailing_stop_activation_offset_percent"], min_v=0)
        save_needed |= _validate_numeric("break_even_trigger_atr_multiple", updated_config, default_config["break_even_trigger_atr_multiple"], min_v=0)
        save_needed |= _validate_numeric("break_even_offset_ticks", updated_config, default_config["break_even_offset_ticks"], min_v=0, is_int=True)

        # Validate Indicator Periods
        for key in DEFAULT_INDICATOR_PERIODS:
             default_val = default_config.get(key) # Use .get for safety
             if default_val is not None: # Check if key exists in defaults
                 is_int_param = isinstance(default_val, int)
                 min_value = 1 if is_int_param else Decimal('1e-9')
                 save_needed |= _validate_numeric(key, updated_config, default_val, min_v=min_value, is_int=is_int_param)

        # Other basic validations
        if updated_config.get("interval") not in VALID_INTERVALS: updated_config["interval"] = default_config["interval"]; save_needed = True
        if not isinstance(updated_config.get("exchange_id"), str) or not updated_config.get("exchange_id"): updated_config["exchange_id"] = default_config["exchange_id"]; save_needed = True
        if updated_config.get("entry_order_type") not in ["market", "limit", "conditional"]: updated_config["entry_order_type"] = "market"; save_needed = True
        symbols = updated_config.get("symbols_to_trade");
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s,str) and s for s in symbols): updated_config["symbols_to_trade"] = default_config["symbols_to_trade"]; save_needed = True

        if save_needed:
            try:
                with open(filepath,"w",encoding="utf-8") as f: json.dump(updated_config,f,indent=4,ensure_ascii=False,default=str)
                print(f"{NEON_YELLOW}Corrected/updated config saved: {filepath}{RESET_ALL_STYLE}")
            except IOError as e: print(f"{NEON_RED}Error writing corrected config: {e}{RESET_ALL_STYLE}")
        return updated_config
    except Exception as e: print(f"{NEON_RED}Error loading config {filepath}: {e}. Using default.{RESET_ALL_STYLE}"); return default_config.copy()

EOF_CONFIG_LOADER

# --- logger_setup.py Content ---
read -r -d '' LOGGER_SETUP_CONTENT <<'EOF_LOGGER_SETUP'
# File: logger_setup.py
"""
Configures the application's logging system.
Sets up handlers for console (stdout/stderr) and rotating files,
using a custom formatter for timezone-aware timestamps and sensitive data masking.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone as dt_timezone, tzinfo # Use built-in timezone, import tzinfo for type hint
from typing import Optional, Type # For type hinting

# Import utility functions and classes
try:
    from utils import LOG_DIRECTORY, SensitiveFormatter, get_timezone
except ImportError:
    print("ERROR: Failed to import required components from 'utils'. Ensure 'utils.py' exists.", file=sys.stderr)
    LOG_DIRECTORY = os.path.join(os.getcwd(), 'logs')
    try: from zoneinfo import ZoneInfo; get_timezone = lambda: ZoneInfo("UTC")
    except ImportError: from datetime import timezone as dt_tz; get_timezone = lambda: dt_tz.utc
    try: SensitiveFormatter
    except NameError: class SensitiveFormatter(logging.Formatter): pass; print("Warning: Using fallback basic Formatter.", file=sys.stderr)

# --- Custom Formatter to Handle Timezone ---
class TimezoneAwareFormatter(SensitiveFormatter):
    """
    A logging formatter that includes timezone information in timestamps.
    Uses a specified timezone object for localization. Inherits from SensitiveFormatter.
    """
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = '%', timezone: Optional[tzinfo] = None):
        """Initializes the formatter with timezone support."""
        super().__init__(fmt, datefmt, style)
        try:
            self.timezone = timezone or get_timezone()
            if not isinstance(self.timezone, tzinfo): raise TypeError("Invalid timezone object")
        except Exception as e:
            print(f"Warning: Error getting timezone: {e}. Defaulting to UTC.", file=sys.stderr)
            self.timezone = dt_timezone.utc

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """Formats the log record's creation time using the configured timezone."""
        utc_dt = datetime.fromtimestamp(record.created, tz=dt_timezone.utc)
        local_dt = utc_dt.astimezone(self.timezone)
        effective_datefmt = datefmt or self.datefmt
        if effective_datefmt:
            s = local_dt.strftime(effective_datefmt)
        else: # Default format if no datefmt provided
            s = local_dt.strftime('%Y-%m-%d %H:%M:%S')
            ms = int(local_dt.microsecond / 1000)
            s = f"{s},{ms:03d} {local_dt.strftime('%Z%z')}" # Add TZ name and offset
        return s

# --- Global Logging Configuration Function ---
_logging_configured = False # Prevent reconfiguration

def configure_logging(config: dict):
    """
    Configures the root logger based on application configuration.
    Sets up console (stdout INFO+, stderr WARNING+) and file handlers (DEBUG+).
    """
    global _logging_configured
    if _logging_configured:
        logging.getLogger("App.Init").warning("Logging already configured. Skipping.")
        return

    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    try: tz = get_timezone()
    except Exception as e: print(f"CRITICAL: Failed get timezone: {e}. Using UTC.", file=sys.stderr); tz = dt_timezone.utc

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            try: handler.close(); root_logger.removeHandler(handler)
            except Exception as e: print(f"Warning: Error removing handler {handler}: {e}", file=sys.stderr)

    log_dir_ok = False
    try: os.makedirs(LOG_DIRECTORY, exist_ok=True); log_dir_ok = True
    except OSError as e: print(f"CRITICAL: Cannot create log dir '{LOG_DIRECTORY}': {e}.", file=sys.stderr)

    console_log_format="%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s"; console_date_format='%Y-%m-%d %H:%M:%S %Z'
    file_log_format="%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(threadName)s - %(message)s"; file_date_format='%Y-%m-%d %H:%M:%S'
    console_formatter=TimezoneAwareFormatter(fmt=console_log_format,datefmt=console_date_format,timezone=tz)
    file_formatter=TimezoneAwareFormatter(fmt=file_log_format,datefmt=file_date_format,timezone=tz)

    handlers_added = []
    try: # STDOUT
        console_stdout = logging.StreamHandler(sys.stdout); console_stdout.setFormatter(console_formatter); console_stdout.setLevel(log_level)
        class InfoFilter(logging.Filter):
            def filter(self, record): return log_level <= record.levelno < logging.WARNING
        console_stdout.addFilter(InfoFilter()); root_logger.addHandler(console_stdout); handlers_added.append("stdout")
    except Exception as e: print(f"CRITICAL: Failed setup stdout handler: {e}", file=sys.stderr)
    try: # STDERR
        console_stderr = logging.StreamHandler(sys.stderr); console_stderr.setFormatter(console_formatter); console_stderr.setLevel(logging.WARNING)
        root_logger.addHandler(console_stderr); handlers_added.append("stderr")
    except Exception as e: print(f"CRITICAL: Failed setup stderr handler: {e}", file=sys.stderr)
    if log_dir_ok: # File Handler
        log_filepath = os.path.join(LOG_DIRECTORY, "xrscalper_bot_main.log")
        try:
            file_handler = RotatingFileHandler(log_filepath, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8', delay=True)
            file_handler.setFormatter(file_formatter); file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler); handlers_added.append("file")
        except Exception as e: print(f"CRITICAL: Failed setup file logger {log_filepath}: {e}", file=sys.stderr)
    else: print("Warning: File logging disabled.", file=sys.stderr)

    library_log_levels = config.get('library_log_levels', {})
    default_lib_levels = {'ccxt': logging.INFO, 'urllib3': logging.WARNING, 'asyncio': logging.WARNING}
    configured_libs = set()
    for lib, lvl_str in library_log_levels.items():
        lvl_int = getattr(logging, lvl_str.upper(), None)
        if lvl_int: logging.getLogger(lib).setLevel(lvl_int); configured_libs.add(lib)
        else: logging.warning(f"Invalid level '{lvl_str}' for lib '{lib}'.")
    for lib, lvl_int in default_lib_levels.items():
        if lib not in configured_libs: logging.getLogger(lib).setLevel(lvl_int)

    init_logger = logging.getLogger("App.Init")
    if handlers_added: init_logger.info(f"Logging configured. Console: {log_level_str}, File: DEBUG. Handlers: {handlers_added}")
    else: init_logger.error("Logging config FAILED. No handlers added.")
    _logging_configured = True
EOF_LOGGER_SETUP

# --- analysis.py Content ---
read -r -d '' ANALYSIS_CONTENT <<'EOF_ANALYSIS'
# File: analysis.py
"""
Module for analyzing trading data, calculating technical indicators, and generating trading signals.
"""

import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext
from typing import Any, Dict, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
try: import pandas_ta as ta
except ImportError: print("ERROR: pandas_ta missing.", file=sys.stderr); ta = None

try: getcontext().prec = 38
except Exception as e: logging.getLogger(__name__).error(f"Failed set Decimal precision: {e}")

try: # Import utils safely
    from utils import ( CCXT_INTERVAL_MAP, DEFAULT_INDICATOR_PERIODS, FIB_LEVELS,
        get_min_tick_size, get_price_precision, format_signal,
        NEON_RED, NEON_YELLOW, NEON_GREEN, RESET_ALL_STYLE, NEON_PURPLE, NEON_BLUE, NEON_CYAN)
except ImportError: # Fallbacks
    print("Warning: Error importing from utils in analysis.py", file=sys.stderr)
    NEON_RED=NEON_YELLOW=NEON_GREEN=RESET_ALL_STYLE=NEON_PURPLE=NEON_BLUE=NEON_CYAN=""
    DEFAULT_INDICATOR_PERIODS={}; CCXT_INTERVAL_MAP={}; FIB_LEVELS=[]
    def get_price_precision(m,l): return 4
    def get_min_tick_size(m,l): return Decimal('0.0001')
    def format_signal(s, **kwargs): return str(s)

ATR_KEY="ATR"; EMA_SHORT_KEY="EMA_Short"; EMA_LONG_KEY="EMA_Long"; MOMENTUM_KEY="Momentum"
CCI_KEY="CCI"; WILLIAMS_R_KEY="Williams_R"; MFI_KEY="MFI"; VWAP_KEY="VWAP"
PSAR_LONG_KEY="PSAR_long"; PSAR_SHORT_KEY="PSAR_short"; SMA10_KEY="SMA10"
STOCHRSI_K_KEY="StochRSI_K"; STOCHRSI_D_KEY="StochRSI_D"; RSI_KEY="RSI"
BB_LOWER_KEY="BB_Lower"; BB_MIDDLE_KEY="BB_Middle"; BB_UPPER_KEY="BB_Upper"
VOLUME_MA_KEY="Volume_MA"; OPEN_KEY="Open"; HIGH_KEY="High"; LOW_KEY="Low"; CLOSE_KEY="Close"; VOLUME_KEY="Volume"

DECIMAL_INDICATOR_KEYS = { ATR_KEY, OPEN_KEY, HIGH_KEY, LOW_KEY, CLOSE_KEY, VOLUME_KEY, BB_LOWER_KEY,
    BB_MIDDLE_KEY, BB_UPPER_KEY, PSAR_LONG_KEY, PSAR_SHORT_KEY, EMA_SHORT_KEY, EMA_LONG_KEY,
    SMA10_KEY, VWAP_KEY, VOLUME_MA_KEY }

class TradingAnalyzer:
    """Analyzes trading data using technical indicators and generates signals."""

    INDICATOR_CONFIG: Dict[str, Dict[str, Any]] = {
        "ATR": {"func_name": "atr", "params_map": {"length": "atr_period"}, "main_col_pattern": "ATRr_{length}", "type": "decimal", "min_data_param_key": "length", "concat": False},
        "EMA_Short": {"func_name": "ema", "params_map": {"length": "ema_short_period"}, "main_col_pattern": "EMA_{length}", "type": "decimal", "pass_close_only": True, "min_data_param_key": "length", "concat": False},
        "EMA_Long": {"func_name": "ema", "params_map": {"length": "ema_long_period"}, "main_col_pattern": "EMA_{length}", "type": "decimal", "pass_close_only": True, "min_data_param_key": "length", "concat": False},
        "Momentum": {"func_name": "mom", "params_map": {"length": "momentum_period"}, "main_col_pattern": "MOM_{length}", "type": "float", "pass_close_only": True, "min_data_param_key": "length", "concat": False},
        "CCI": {"func_name": "cci", "params_map": {"length": "cci_window", "c": "cci_constant"}, "main_col_pattern": "CCI_{length}_{c:.3f}", "type": "float", "min_data_param_key": "length", "concat": False},
        "Williams_R": {"func_name": "willr", "params_map": {"length": "williams_r_window"}, "main_col_pattern": "WILLR_{length}", "type": "float", "min_data_param_key": "length", "concat": False},
        "MFI": {"func_name": "mfi", "params_map": {"length": "mfi_window"}, "main_col_pattern": "MFI_{length}", "type": "float", "concat": True, "min_data_param_key": "length"},
        "VWAP": {"func_name": "vwap", "params_map": {}, "main_col_pattern": "VWAP_D", "type": "decimal", "concat": True, "min_data": 1},
        "PSAR": {"func_name": "psar", "params_map": {"initial": "psar_initial_af", "step": "psar_af_step", "max": "psar_max_af"}, "multi_cols": {"PSAR_long": "PSARl_{initial}_{max}", "PSAR_short": "PSARs_{initial}_{max}"}, "type": "decimal", "concat": True, "min_data": 2}, # Adjusted pattern
        "StochRSI": {"func_name": "stochrsi", "params_map": {"length": "stoch_rsi_window", "rsi_length": "stoch_rsi_rsi_window", "k": "stoch_rsi_k", "d": "stoch_rsi_d"}, "multi_cols": {"StochRSI_K": "STOCHRSIk_{length}_{rsi_length}_{k}_{d}", "StochRSI_D": "STOCHRSId_{length}_{rsi_length}_{k}_{d}"}, "type": "float", "concat": True, "min_data_param_key": "length"},
        "Bollinger_Bands": {"func_name": "bbands", "params_map": {"length": "bollinger_bands_period", "std": "bollinger_bands_std_dev"}, "multi_cols": {"BB_Lower": "BBL_{length}_{std:.1f}", "BB_Middle": "BBM_{length}_{std:.1f}", "BB_Upper": "BBU_{length}_{std:.1f}"}, "type": "decimal", "concat": True, "min_data_param_key": "length"},
        "Volume_MA": {"func_name": "_calculate_volume_ma", "params_map": {"length": "volume_ma_period"}, "main_col_pattern": "VOL_SMA_{length}", "type": "decimal", "min_data_param_key": "length", "concat": False},
        "SMA10": {"func_name": "sma", "params_map": {"length": "sma_10_window"}, "main_col_pattern": "SMA_{length}", "type": "decimal", "pass_close_only": True, "min_data_param_key": "length", "concat": False},
        "RSI": {"func_name": "rsi", "params_map": {"length": "rsi_period"}, "main_col_pattern": "RSI_{length}", "type": "float", "pass_close_only": True, "min_data_param_key": "length", "concat": False},
    }

    def __init__(
        self, df: pd.DataFrame, logger: logging.Logger,
        config: Dict[str, Any], market_info: Dict[str, Any]
    ) -> None:
        """Initializes the TradingAnalyzer."""
        self.logger = logger; self.config = config; self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN'); self.interval = str(config.get("interval", "5"))
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval)
        self.indicator_values: Dict[str, Union[Decimal, float, None]] = {}; self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default"); self.weights = config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {}; self.ta_column_names: Dict[str, str] = {}; self.df_calculated: pd.DataFrame = pd.DataFrame()
        self.indicator_type_map: Dict[str, str] = { k: d.get("type", "float") for k, d in self.INDICATOR_CONFIG.items() }
        for d in self.INDICATOR_CONFIG.values():
            if "multi_cols" in d: self.indicator_type_map.update({sk: d.get("type", "float") for sk in d["multi_cols"]})

        if not isinstance(df, pd.DataFrame) or df.empty: raise ValueError(f"Input df invalid {self.symbol}.")
        if not self.ccxt_interval: raise ValueError(f"Interval '{self.interval}' invalid.")
        if not self.weights: self.logger.warning(f"Weight set '{self.active_weight_set_name}' empty {self.symbol}.")
        req=['open','high','low','close','volume']; missing=[c for c in req if c not in df.columns]
        if missing: raise ValueError(f"df missing columns: {missing}")

        self.df_original_ohlcv = df.copy()
        if self.df_original_ohlcv.index.tz is not None: self.df_original_ohlcv.index = self.df_original_ohlcv.index.tz_localize(None)
        self._validate_and_prepare_df_calculated()
        if not self.df_calculated.empty and ta is not None:
            self._calculate_all_indicators(); self._update_latest_indicator_values(); self.calculate_fibonacci_levels()
        elif ta is None: self.logger.error("pandas_ta missing."); self.indicator_values = {}
        else: self.logger.error(f"DataFrame prep failed {self.symbol}."); self.indicator_values = {}

    def _validate_and_prepare_df_calculated(self) -> None:
        """Validates, converts OHLCV to float64, checks data length."""
        # Logic unchanged
        self.df_calculated=self.df_original_ohlcv.copy(); req=['open','high','low','close','volume']; min_rows=1
        for c in req:
            try: num_c=pd.to_numeric(self.df_calculated[c],errors='coerce'); self.df_calculated[c]=num_c.astype('float64')
            except Exception as e: self.logger.critical(f"Failed convert '{c}' {self.symbol}: {e}"); self.df_calculated=pd.DataFrame(); return
            if self.df_calculated[c].isna().any(): self.logger.warning(f"{self.df_calculated[c].isna().sum()} NaNs in '{c}' {self.symbol}.")
        self.df_calculated.dropna(subset=['open','high','low','close'],inplace=True)
        if self.df_calculated.empty: self.logger.error(f"DataFrame empty after dropna {self.symbol}."); return
        enabled=self.config.get("indicators",{}); max_lb=1
        for k,d in self.INDICATOR_CONFIG.items():
            if enabled.get(k.lower(),False):
                pk=d.get("min_data_param_key"); ck=d.get("params_map",{}).get(pk) if pk else None
                if ck: v=self.get_period(ck);
                if isinstance(v,(int,float,Decimal)) and Decimal(str(v))>0: max_lb=max(max_lb,int(Decimal(str(v))))
                elif isinstance(d.get("min_data"),int): max_lb=max(max_lb,d["min_data"])
        buf=self.config.get("indicator_buffer_candles",20); min_rows=max_lb+buf
        if len(self.df_calculated)<min_rows: self.logger.warning(f"Insuff rows ({len(self.df_calculated)}<{min_rows}) {self.symbol}.")

    def get_period(self, key: str) -> Any: # Logic unchanged
        val=self.config.get(key); return val if val is not None else DEFAULT_INDICATOR_PERIODS.get(key)

    def _format_ta_column_name(self, pattern: str, params: Dict[str, Any]) -> str: # Logic unchanged
        fmt_p={};
        for k,v in params.items():
            if v is None: fmt_p[k]="DEF"
            elif isinstance(v,(float,Decimal)): fmt_p[k]=float(v) if f"{{{k}:." in pattern else str(v).replace('.','_')
            else: fmt_p[k]=v
        try: return pattern.format(**fmt_p)
        except Exception as e: self.logger.error(f"Err fmt TA col '{pattern}': {e}"); base=pattern.split("{")[0].rstrip('_'); keys="_".join(map(str,params.values())); return f"{base}_{keys}_ERR"

    def _calculate_volume_ma(self, df: pd.DataFrame, length: int) -> Optional[pd.Series]: # Logic unchanged
        if 'volume' not in df.columns or not(isinstance(length,int) and length>0): return None
        vol=df['volume'].astype(float).fillna(0);
        if len(vol)<length: return pd.Series(np.nan,index=df.index)
        try: return ta.sma(vol,length=length) if ta else None
        except Exception as e: self.logger.error(f"Err Vol SMA {self.symbol}: {e}"); return None

    def _calculate_all_indicators(self) -> None: # Logic unchanged (uses partial match from previous fix)
        if self.df_calculated.empty or ta is None: return
        df_work=self.df_calculated; enabled=self.config.get("indicators",{}); self.ta_column_names={}
        for key,details in self.INDICATOR_CONFIG.items():
            if not enabled.get(key.lower(),False): continue
            params={}; valid=True
            for p_name,c_key in details.get("params_map",{}).items():
                val=self.get_period(c_key);
                if val is None: valid=False; break
                try: params[p_name]=float(val) if isinstance(val,(Decimal,str)) else int(val) if isinstance(val,int) else val
                except: valid=False; break
            if not valid: continue
            try:
                func_name=details["func_name"]; func=getattr(ta,func_name,None) if hasattr(ta,func_name) else getattr(self,func_name,None)
                if not func: continue
                min_len=int(params.get(details.get("min_data_param_key","length"),details.get("min_data",1)))
                if len(df_work)<min_len: continue
                inputs={}; req_cols=['open','high','low','close','volume']; close_s=df_work['close'].astype(float)
                if func_name!="_calculate_volume_ma": import inspect; sig=inspect.signature(func).parameters; inputs={c:df_work[c].astype(float) for c in req_cols if c in sig and c in df_work.columns}
                result=None
                if func_name=="_calculate_volume_ma": result=func(df_work,**params)
                elif details.get("pass_close_only",False): result=func(close=close_s,**params)
                elif inputs: result=func(**inputs,**params)
                else: result=func(close=close_s,**params)
                if result is None: continue
                concat=details.get("concat",False); col_name=self._format_ta_column_name(details.get("main_col_pattern",""),params) if "main_col_pattern" in details else None
                if isinstance(result,pd.Series):
                    if col_name:
                        if col_name in df_work.columns: df_work.drop(columns=[col_name],inplace=True,errors='ignore')
                        if concat: df_work=pd.concat([df_work,result.to_frame(name=col_name).astype('float64')],axis=1,copy=False)
                        else: df_work[col_name]=result.astype('float64')
                        self.ta_column_names[key]=col_name
                elif isinstance(result,pd.DataFrame):
                    if concat:
                        try: piece=result.astype('float64')
                        except: piece=pd.DataFrame({c:pd.to_numeric(result[c],errors='coerce').astype('float64') for c in result.columns},index=result.index)
                        cols_drop=[c for c in piece.columns if c in df_work.columns];
                        if cols_drop: df_work.drop(columns=cols_drop,inplace=True,errors='ignore')
                        df_work=pd.concat([df_work,piece],axis=1,copy=False)
                        if "multi_cols" in details:
                            for ik,pat in details["multi_cols"].items():
                                acn=self._format_ta_column_name(pat,params)
                                if acn in df_work.columns: self.ta_column_names[ik]=acn
                                else:
                                     partial=next((c for c in df_work.columns if c.startswith(acn.split('_')[0]) and all(p in c for p in map(str,params.values()))),None)
                                     if partial: self.logger.warning(f"Partial match {acn} -> {partial}"); self.ta_column_names[ik]=partial
                                     else: self.logger.warning(f"Mapped col '{acn}'({ik}) not found.")
                    else: self.logger.error(f"{key} (concat=False) returned DataFrame.")
            except Exception as e: self.logger.error(f"Error calc ind {key}: {e}",exc_info=True)
        self.df_calculated=df_work

    def _update_latest_indicator_values(self) -> None: # Logic unchanged
        # ... (Same as previous version) ...
        if self.df_calculated.empty: self.indicator_values={}; return
        try:
            latest_ind=self.df_calculated.iloc[-1]; latest_ohlcv=self.df_original_ohlcv.iloc[-1]
            ohlcv_map={OPEN_KEY:"open",HIGH_KEY:"high",LOW_KEY:"low",CLOSE_KEY:"close",VOLUME_KEY:"volume"}
            latest_vals:Dict[str,Union[Decimal,float,None]]={}; all_keys=set(self.indicator_type_map.keys())|set(ohlcv_map.keys())|set(self.ta_column_names.keys())
            for k in all_keys: latest_vals[k]=None
            for ik,ac in self.ta_column_names.items():
                if ac in latest_ind.index:
                    v=latest_ind[ac]; tt=self.indicator_type_map.get(ik,"float")
                    if pd.notna(v):
                        try: latest_vals[ik]=Decimal(str(v)) if tt=="decimal" else float(v)
                        except: pass
            for dk,sc in ohlcv_map.items():
                 v=latest_ohlcv.get(sc);
                 if pd.notna(v):
                      try: latest_vals[dk]=Decimal(str(v))
                      except: pass
            self.indicator_values=latest_vals
        except Exception as e: self.logger.error(f"Err update latest {self.symbol}: {e}"); self.indicator_values={}

    # --- Fibonacci Calculations --- (Logic unchanged)
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]: # Logic unchanged
        window=window or self.get_period("fibonacci_window");
        if not(isinstance(window,int)and window>0 and len(self.df_original_ohlcv)>=window): return {}
        df_s=self.df_original_ohlcv.tail(window)
        try:
            h=Decimal(str(pd.to_numeric(df_s["high"],errors='coerce').max())); l=Decimal(str(pd.to_numeric(df_s["low"],errors='coerce').min()))
            diff=h-l; levels={}; prec=get_price_precision(self.market_info,self.logger); tick=get_min_tick_size(self.market_info,self.logger); quant=tick if tick>0 else Decimal(f'1e-{prec}')
            if diff>0:
                for p in FIB_LEVELS: price=h-(diff*p); levels[f"Fib_{p*100:.1f}%"]=(price/quant).quantize(Decimal('1'),ROUND_DOWN)*quant
            else: lvl=(h/quant).quantize(Decimal('1'),ROUND_DOWN)*quant; levels={f"Fib_{p*100:.1f}%":lvl for p in FIB_LEVELS}
            self.fib_levels_data=levels; return levels
        except Exception as e: self.logger.error(f"Fib err {self.symbol}: {e}"); return {}
    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]: # Logic unchanged
        if not self.fib_levels_data or not(isinstance(current_price,Decimal)and current_price>0) or num_levels<=0: return []
        try:
            dists=[{'name':n,'level':p,'distance':abs(current_price-p)} for n,p in self.fib_levels_data.items() if isinstance(p,Decimal)and p>0]
            dists.sort(key=lambda x:x['d']); return [(i['name'],i['level']) for i in dists[:num_levels]]
        except Exception as e: self.logger.error(f"Nearest Fib err {self.symbol}: {e}"); return []

    # --- EMA Score --- (Logic unchanged)
    def calculate_ema_alignment_score(self) -> float: # Logic unchanged
        ema_s,ema_l,close=self.indicator_values.get(EMA_SHORT_KEY),self.indicator_values.get(EMA_LONG_KEY),self.indicator_values.get(CLOSE_KEY)
        if not all(isinstance(v,Decimal) for v in [ema_s,ema_l,close]): return np.nan
        if close>ema_s>ema_l: return 1.0;
        if close<ema_s<ema_l: return -1.0;
        return 0.0

    # --- Signal Generation --- (Logic unchanged)
    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[Dict]) -> str:
        # ... (Same as previous version) ...
        self.signals={"BUY":0,"SELL":0,"HOLD":1}; score,w_sum=Decimal("0"),Decimal("0"); act_c,nan_c=0,0; dbg_s={}
        if not self.indicator_values: return "HOLD"
        min_req=self.config.get("min_active_indicators_for_signal",7); valid_c=sum(1 for k in self.INDICATOR_CONFIG if pd.notna(self.indicator_values.get(k)))
        if valid_c<min_req: self.logger.warning(f"Signal {self.symbol}: {valid_c} valid < req {min_req}. HOLD."); return "HOLD"
        if not(isinstance(current_price,Decimal) and current_price>0): return "HOLD"
        weights=self.weights;
        if not weights: return "HOLD"
        for k_l,w_v in weights.items():
            if not self.config.get("indicators",{}).get(k_l,False): continue
            try: w=Decimal(str(w_v))
            except: continue
            if w==0: continue
            m_name=f"_check_{k_l}"; m_obj=getattr(self,m_name,None)
            if not m_obj or not callable(m_obj): continue
            s_f=np.nan
            try: s_f=m_obj(orderbook_data,current_price) if k_l=="orderbook" else m_obj()
            except Exception as e: self.logger.error(f"Err check {m_name}: {e}")
            dbg_s[k_l]=f"{s_f:.3f}" if pd.notna(s_f) else "NaN"
            if pd.notna(s_f):
                try: s_d=Decimal(str(s_f)); clamp=max(Decimal("-1"),min(Decimal("1"),s_d)); score+=clamp*w; w_sum+=abs(w); act_c+=1
                except: nan_c+=1
            else: nan_c+=1
        final_sig="HOLD"; thresh=Decimal(str(self.get_period("signal_score_threshold") or "0.7"))
        if w_sum>0:
            if score>=thresh: final_sig="BUY"
            elif score<=-thresh: final_sig="SELL"
        self.logger.info(f"Signal ({self.symbol} @ {_format_price_or_na(current_price,get_price_precision(self.market_info,self.logger))}): Set='{self.active_weight_set_name}', Checks[A:{act_c},N:{nan_c}], WgtSum={w_sum:.2f}, Score={score:.4f} (Th:{threshold:.2f}) ==> {format_signal(final_sig)}")
        self.logger.debug(f"Scores: {dbg_s}")
        self.signals={"BUY":int(final_sig=="BUY"),"SELL":int(final_sig=="SELL"),"HOLD":int(final_sig=="HOLD")}
        return final_sig

    # --- Individual Indicator Check Methods (_check_*) ---
    # (Implementations remain the same as previous version)
    def _check_ema_alignment(self) -> float: return self.calculate_ema_alignment_score()
    def _check_momentum(self) -> float:
        mom=self.indicator_values.get(MOMENTUM_KEY); close=self.indicator_values.get(CLOSE_KEY);
        if pd.isna(mom) or not isinstance(close,Decimal) or close<=0: return np.nan
        try: score=Decimal(str(mom))/close; return float(max(Decimal("-1"),min(Decimal("1"),score*10)))
        except: return 0.0
    def _check_volume_confirmation(self) -> float:
        vol=self.indicator_values.get(VOLUME_KEY); vol_ma=self.indicator_values.get(VOLUME_MA_KEY)
        mult=Decimal(str(self.get_period("volume_confirmation_multiplier") or "1.5"))
        if not all(isinstance(v,Decimal)and pd.notna(v)and v>=0 for v in [vol,vol_ma])or mult<=0: return np.nan
        if vol_ma==0: return 0.0; ratio=vol/vol_ma; return 1.0 if ratio>mult else -0.4 if ratio<(1/mult if mult>0 else 0) else 0.0
    def _check_stoch_rsi(self) -> float:
        k,d=self.indicator_values.get(STOCHRSI_K_KEY),self.indicator_values.get(STOCHRSI_D_KEY)
        if pd.isna(k) or pd.isna(d): return np.nan; k,d=float(k),float(d); os=float(self.get_period("stoch_rsi_oversold_threshold")or 20); ob=float(self.get_period("stoch_rsi_overbought_threshold")or 80)
        score=0.0;
        if k<os and d<os: score=1.0
        elif k>ob and d>ob: score=-1.0
        if k>d and score>=0: score=max(score,0.4)
        if k<d and score<=0: score=min(score,-0.4)
        if 40<k<60 and 40<d<60: score*=0.5
        return score
    def _check_rsi(self) -> float:
        rsi=self.indicator_values.get(RSI_KEY);
        if pd.isna(rsi): return np.nan; rsi=float(rsi)
        os,ob=float(self.get_period("rsi_oversold_threshold")or 30),float(self.get_period("rsi_overbought_threshold")or 70)
        if rsi<=os: return 1.0
        if rsi>=ob: return -1.0
        if os<rsi<50: return(50.0-rsi)/(50.0-os)*0.8
        if 50<rsi<ob: return(50.0-rsi)/(ob-50.0)*0.8
        return 0.0
    def _check_cci(self) -> float:
        cci=self.indicator_values.get(CCI_KEY);
        if pd.isna(cci): return np.nan; cci=float(cci)
        sos,mob=float(self.get_period("cci_strong_oversold")or -150),float(self.get_period("cci_moderate_overbought")or 100)
        sob,mos=float(self.get_period("cci_strong_overbought")or 150),float(self.get_period("cci_moderate_oversold")or -100)
        if cci<=sos: return 1.0
        if cci>=sob: return -1.0
        if cci<mos: return 0.6
        if cci>mob: return -0.6
        return 0.0
    def _check_wr(self) -> float:
        wr=self.indicator_values.get(WILLIAMS_R_KEY);
        if pd.isna(wr): return np.nan; wr=float(wr)
        os,ob=float(self.get_period("wr_oversold_threshold")or -80),float(self.get_period("wr_overbought_threshold")or -20)
        if wr<=os: return 1.0
        if wr>=ob: return -1.0
        mid=(os+ob)/2.0;
        if os<wr<mid: return(wr-mid)/(os-mid)*0.7
        if mid<=wr<ob: return(wr-mid)/(ob-mid)*-0.7
        return 0.0
    def _check_psar(self) -> float:
        psar_l,psar_s,close=self.indicator_values.get(PSAR_LONG_KEY),self.indicator_values.get(PSAR_SHORT_KEY),self.indicator_values.get(CLOSE_KEY)
        if not isinstance(close,Decimal): return np.nan
        is_long=isinstance(psar_l,Decimal) and close>psar_l; is_short=isinstance(psar_s,Decimal) and close<psar_s
        if is_long and not is_short: return 1.0
        if is_short and not is_long: return -1.0
        return 0.0
    def _check_sma_10(self) -> float:
        sma,close=self.indicator_values.get(SMA10_KEY),self.indicator_values.get(CLOSE_KEY)
        if not all(isinstance(v,Decimal) for v in [sma,close]): return np.nan
        diff=(close-sma)/sma if sma>0 else Decimal(0); return float(max(Decimal("-1"),min(Decimal("1"),diff*10)))
    def _check_vwap(self) -> float:
        vwap,close=self.indicator_values.get(VWAP_KEY),self.indicator_values.get(CLOSE_KEY)
        if not all(isinstance(v,Decimal) for v in [vwap,close]): return np.nan
        diff=(close-vwap)/vwap if vwap>0 else Decimal(0); return float(max(Decimal("-1"),min(Decimal("1"),diff*15)))
    def _check_mfi(self) -> float:
        mfi=self.indicator_values.get(MFI_KEY);
        if pd.isna(mfi): return np.nan; mfi=float(mfi)
        os,ob=float(self.get_period("mfi_oversold_threshold")or 20),float(self.get_period("mfi_overbought_threshold")or 80)
        if mfi<=os: return 1.0
        if mfi>=ob: return -1.0
        return 0.0
    def _check_bollinger_bands(self) -> float:
        bb_l,bb_m,bb_u,close=self.indicator_values.get(BB_LOWER_KEY),self.indicator_values.get(BB_MIDDLE_KEY),self.indicator_values.get(BB_UPPER_KEY),self.indicator_values.get(CLOSE_KEY)
        if not all(isinstance(v,Decimal) for v in [bb_l,bb_m,bb_u,close]): return np.nan
        if close<=bb_l: return 1.0
        if close>=bb_u: return -1.0
        width=bb_u-bb_l;
        if width>0: pos=(close-bb_m)/(width/2); return float(max(Decimal("-0.7"),min(Decimal("0.7"),pos*Decimal("0.7"))))
        return 0.0
    def _check_orderbook(self, orderbook_data: Optional[Dict[str, Any]], current_price: Decimal) -> float:
        if not orderbook_data: return np.nan
        try:
            bids=orderbook_data.get("bids",[]); asks=orderbook_data.get("asks",[])
            if not bids or not asks: return np.nan
            levels=self.config.get("orderbook_check_levels",10)
            bid_q=sum(Decimal(str(b[1])) for b in bids[:levels] if len(b)==2 and b[1] is not None)
            ask_q=sum(Decimal(str(a[1])) for a in asks[:levels] if len(a)==2 and a[1] is not None)
            total_q=bid_q+ask_q;
            if total_q==0: return 0.0
            obi=(bid_q-ask_q)/total_q; return float(max(Decimal("-1"),min(Decimal("1"),obi)))
        except Exception as e: self.logger.warning(f"OB analysis error: {e}"); return np.nan

    # --- TP/SL Calculation ---
    def calculate_entry_tp_sl(
        self, entry_price_estimate: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculates potential TP/SL based on ATR and config multipliers."""
        # Logic unchanged from previous version
        if signal not in ["BUY", "SELL"] or not(isinstance(entry_price_estimate,Decimal) and entry_price_estimate>0): return entry_price_estimate,None,None
        atr=self.indicator_values.get(ATR_KEY)
        if not(isinstance(atr,Decimal) and atr>0):
            atr_pct_str=str(self.get_period("default_atr_percentage_of_price") or "0.0")
            try: atr_pct=Decimal(atr_pct_str);
            if atr_pct>0: atr=entry_price_estimate*atr_pct
            else: self.logger.error(f"No ATR/default {self.symbol}"); return entry_price_estimate,None,None
            except: self.logger.error(f"Invalid default ATR% {self.symbol}"); return entry_price_estimate,None,None
        try:
            tp_mult=Decimal(str(self.get_period("take_profit_multiple")or"2.0")); sl_mult=Decimal(str(self.get_period("stop_loss_multiple")or"1.5"))
            if tp_mult<=0 or sl_mult<=0: raise ValueError("Mults must be >0")
            prec=get_price_precision(self.market_info,self.logger); tick=get_min_tick_size(self.market_info,self.logger); quant=tick if tick>0 else Decimal(f'1e-{prec}')
            if quant<=0: quant=Decimal('1e-8')
            tp_off=atr*tp_mult; sl_off=atr*sl_mult
            if signal=="BUY": tp=((entry_price_estimate+tp_off)/quant).quantize(Decimal('1'),ROUND_UP)*quant; sl=((entry_price_estimate-sl_off)/quant).quantize(Decimal('1'),ROUND_DOWN)*quant
            else: tp=((entry_price_estimate-tp_off)/quant).quantize(Decimal('1'),ROUND_DOWN)*quant; sl=((entry_price_estimate+sl_off)/quant).quantize(Decimal('1'),ROUND_UP)*quant
            # Final validation
            if sl is not None and sl<=0: sl=None
            if tp is not None and tp<=0: tp=None
            if sl and tp and((signal=="BUY" and sl>=tp)or(signal=="SELL" and sl<=tp)): sl=tp=None
            if sl and((signal=="BUY" and sl>=entry_price_estimate)or(signal=="SELL" and sl<=entry_price_estimate)): sl=None
            if tp and((signal=="BUY" and tp<=entry_price_estimate)or(signal=="SELL" and tp>=entry_price_estimate)): tp=None
            return entry_price_estimate,tp,sl
        except Exception as e: self.logger.error(f"Err calc TP/SL {self.symbol}: {e}"); return entry_price_estimate,None,None
