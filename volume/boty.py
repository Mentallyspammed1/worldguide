Okay, here is the enhanced version of the Python script.

**Key Enhancements:**

1.  **Improved Readability:** Added vertical whitespace (blank lines) to separate logical blocks of code within functions and globally. Broke down long lines for better clarity.
2.  **Code Structure & Imports:** Organized imports according to standard practice (standard library, third-party, local - though no local here). Alphabetized imports within sections.
3.  **Docstrings:** Added docstrings to functions and classes explaining their purpose, parameters, return values, and potential exceptions, especially for complex functions like `load_config`, `fetch_klines_ccxt`, `calculate_position_size`, etc.
4.  **Comments:** Added inline comments to clarify non-obvious logic, specific choices (like Bybit API parameters), error handling conditions, and the purpose of certain variables or blocks.
5.  **Constants:** Grouped constants logically and ensured consistent uppercase naming.
6.  **Error Handling:** Made exception handling slightly more specific where appropriate and improved logging messages within `except` blocks for better debugging. Clarified retry logic comments.
7.  **Type Hinting:** Reviewed and maintained type hints for consistency.
8.  **Logging:** Enhanced log messages for clarity, especially during configuration validation and API interactions. Ensured appropriate log levels are used. Added a docstring to `SensitiveFormatter`.
9.  **Configuration:** Improved comments within `load_config` to explain the validation process and the handling of the global `QUOTE_CURRENCY`.
10. **Main Function:** Clarified the setup and main loop logic with comments. Corrected the logger retrieval for symbols within the loop. Added `logging.shutdown()`.
11. **Placeholders:** Added explicit comments and basic docstrings to the placeholder functions/classes (`VolumaticOBStrategy`, `SignalGenerator`, `analyze_and_trade_symbol`, etc.) to indicate they need full implementation.
12. **Minor Fixes:** Corrected the logger call in the main loop (`setup_logger` instead of undefined `get_logger_for_symbol`).

```python
# pyrmethus_volumatic_bot.py - Merged & Minified v1.4.1 (Fixed global scope error)
# Enhanced Version: Improved readability, comments, docstrings, and structure.

# --- Standard Library Imports ---
import hashlib
import hmac
import json
import logging
import math
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

# Attempt to import zoneinfo, provide fallback for older Python versions
try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    print("Warning: 'zoneinfo' module not found. Falling back to UTC. Ensure Python 3.9+ and install 'tzdata'.")
    # Basic fallback ZoneInfo class mimicking essential behavior with UTC
    class ZoneInfo: # type: ignore [no-redef]
        def __init__(self, key: str):
            self._key = "UTC" # Ignore the key, always use UTC
        def __call__(self, dt=None):
            return dt.replace(tzinfo=timezone.utc) if dt else None
        def fromutc(self, dt):
            return dt.replace(tzinfo=timezone.utc)
        def utcoffset(self, dt):
            return timedelta(0)
        def dst(self, dt):
            return timedelta(0)
        def tzname(self, dt):
            return "UTC"
    class ZoneInfoNotFoundError(Exception): pass # type: ignore [no-redef]

# --- Third-Party Library Imports ---
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# --- Global Settings & Initializations ---
getcontext().prec = 28  # Set Decimal precision
colorama_init(autoreset=True) # Initialize Colorama
load_dotenv() # Load environment variables from .env file

# --- Constants ---
BOT_VERSION = "1.4.1"

# API Credentials (Loaded from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    print(f"{Fore.RED}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET missing in .env. Exiting.{Style.RESET_ALL}")
    sys.exit(1)

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
DEFAULT_TIMEZONE_STR = "America/Chicago" # Default if not set in .env
TIMEZONE_STR = os.getenv("TIMEZONE", DEFAULT_TIMEZONE_STR)

# Timezone Initialization
try:
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except ZoneInfoNotFoundError:
    print(f"{Fore.RED}Timezone '{TIMEZONE_STR}' not found. Using UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"
except Exception as tz_err:
    print(f"{Fore.RED}Timezone initialization error for '{TIMEZONE_STR}': {tz_err}. Using UTC.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC"

# API Interaction Parameters
MAX_API_RETRIES = 3             # Max attempts for API calls
RETRY_DELAY_SECONDS = 5         # Initial delay between retries
POSITION_CONFIRM_DELAY_SECONDS = 8 # Delay after placing order before confirming position
LOOP_DELAY_SECONDS = 15         # Base delay between main loop cycles
BYBIT_API_KLINE_LIMIT = 1000    # Max klines per Bybit API request

# Kline/Data Parameters
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Configurable intervals
CCXT_INTERVAL_MAP = { # Map config intervals to CCXT timeframe strings
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
DEFAULT_FETCH_LIMIT = 750       # Default number of klines to fetch
MAX_DF_LEN = 2000               # Maximum length of DataFrame to keep in memory

# Default Strategy Parameters (Volumatic Trend)
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0

# Default Strategy Parameters (Order Blocks)
DEFAULT_OB_SOURCE = "Wicks"     # "Wicks" or "Body"
DEFAULT_PH_LEFT = 10            # Pivot High lookback
DEFAULT_PH_RIGHT = 10           # Pivot High lookforward
DEFAULT_PL_LEFT = 10            # Pivot Low lookback
DEFAULT_PL_RIGHT = 10           # Pivot Low lookforward
DEFAULT_OB_EXTEND = True        # Extend OBs until violated
DEFAULT_OB_MAX_BOXES = 50       # Max active OBs to track

# Trading Parameters
QUOTE_CURRENCY = "USDT"         # Default, will be updated from config

# Colorama Colors
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

# --- Global State ---
_shutdown_requested = False     # Flag for graceful shutdown

# --- Setup Log Directory ---
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
except OSError as e:
    print(f"{NEON_RED}{BRIGHT}FATAL: Could not create log directory '{LOG_DIRECTORY}': {e}. Exiting.{RESET}")
    sys.exit(1)

# --- TypedDict Definitions for Data Structures ---
class OrderBlock(TypedDict):
    id: str
    type: str           # 'bull' or 'bear'
    timestamp: pd.Timestamp
    top: Decimal
    bottom: Decimal
    active: bool
    violated: bool
    violation_ts: Optional[pd.Timestamp]
    extended_to_ts: Optional[pd.Timestamp]

class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame
    last_close: Decimal
    current_trend_up: Optional[bool]
    trend_just_changed: bool
    active_bull_boxes: List[OrderBlock]
    active_bear_boxes: List[OrderBlock]
    vol_norm_int: Optional[int]
    atr: Optional[Decimal]
    upper_band: Optional[Decimal]
    lower_band: Optional[Decimal]

class MarketInfo(TypedDict): # Enhanced from ccxt market structure
    id: str
    symbol: str
    base: str
    quote: str
    settle: Optional[str]
    baseId: str
    quoteId: str
    settleId: Optional[str]
    type: str
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool
    contract: bool
    linear: Optional[bool]
    inverse: Optional[bool]
    quanto: Optional[bool]
    taker: float
    maker: float
    contractSize: Optional[Any] # Can be int or float string
    expiry: Optional[int]
    expiryDatetime: Optional[str]
    strike: Optional[float]
    optionType: Optional[str]
    precision: Dict[str, Any]
    limits: Dict[str, Any]
    info: Dict[str, Any] # Original exchange-specific info
    # --- Added fields ---
    is_contract: bool           # True if swap or future
    is_linear: bool             # True if linear contract
    is_inverse: bool            # True if inverse contract
    contract_type_str: str      # "Linear", "Inverse", "Spot", "Unknown"
    min_amount_decimal: Optional[Decimal]
    max_amount_decimal: Optional[Decimal]
    min_cost_decimal: Optional[Decimal]
    max_cost_decimal: Optional[Decimal]
    amount_precision_step_decimal: Optional[Decimal] # Smallest increment for amount
    price_precision_step_decimal: Optional[Decimal]  # Smallest increment for price
    contract_size_decimal: Decimal # Parsed contract size as Decimal

class PositionInfo(TypedDict): # Enhanced from ccxt position structure
    id: Optional[str]
    symbol: str
    timestamp: Optional[int]
    datetime: Optional[str]
    contracts: Optional[float]  # Original contracts field (may differ from 'size')
    contractSize: Optional[Any] # From market info, added for convenience
    side: Optional[str]         # 'long' or 'short'
    notional: Optional[Any]
    leverage: Optional[Any]     # Use leverage_decimal
    unrealizedPnl: Optional[Any] # Use pnl_decimal
    realizedPnl: Optional[Any]
    collateral: Optional[Any]
    entryPrice: Optional[Any]   # Use entry_price_decimal
    markPrice: Optional[Any]
    liquidationPrice: Optional[Any] # Use liq_price_decimal
    marginMode: Optional[str]
    hedged: Optional[bool]
    maintenanceMargin: Optional[Any]
    maintenanceMarginPercentage: Optional[float]
    initialMargin: Optional[Any]
    initialMarginPercentage: Optional[float]
    marginRatio: Optional[float]
    lastUpdateTimestamp: Optional[int]
    info: Dict[str, Any] # Original exchange-specific info
    # --- Added/Processed fields ---
    size_decimal: Decimal       # Position size as Decimal (positive for long, negative for short)
    stopLossPrice: Optional[str] # From info if available
    takeProfitPrice: Optional[str] # From info if available
    trailingStopLoss: Optional[str] # TSL distance/percentage (from info)
    tslActivationPrice: Optional[str] # Price at which TSL activates (from info)
    # --- Bot State Tracking ---
    be_activated: bool          # Has break-even been activated for this position?
    tsl_activated: bool         # Is a trailing stop loss currently active on the exchange?

class SignalResult(TypedDict):
    signal: str # "BUY", "SELL", "HOLD", "EXIT_LONG", "EXIT_SHORT"
    reason: str # Explanation for the signal
    initial_sl: Optional[Decimal] # Calculated SL for a new entry
    initial_tp: Optional[Decimal] # Calculated TP for a new entry

# --- Logging Configuration ---

class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive API keys and secrets.
    """
    _api_key_placeholder = "***BYBIT_API_KEY***"
    _api_secret_placeholder = "***BYBIT_API_SECRET***"

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, redacting sensitive information."""
        msg = super().format(record)
        key = API_KEY
        secret = API_SECRET
        try:
            # Only redact if key/secret are non-empty strings
            if key and isinstance(key, str) and key in msg:
                msg = msg.replace(key, self._api_key_placeholder)
            if secret and isinstance(secret, str) and secret in msg:
                msg = msg.replace(secret, self._api_secret_placeholder)
        except Exception as e:
            # Avoid crashing the application if logging fails
            print(f"WARNING: Error during log message redaction: {e}", file=sys.stderr)
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger instance with both file and console handlers.

    Args:
        name: The name for the logger (often the module or symbol name).

    Returns:
        The configured Logger instance.
    """
    # Sanitize name for filename compatibility
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): # Avoid adding handlers multiple times
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels, handlers control output level

    # --- File Handler ---
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use UTC time for file logs for consistency
        ff = SensitiveFormatter(
            "%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        ff.converter = time.gmtime # type: ignore [assignment] # Use UTC timestamps in log files
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(fh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler ---
    try:
        sh = logging.StreamHandler(sys.stdout)
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,
            logging.INFO: NEON_BLUE,
            logging.WARNING: NEON_YELLOW,
            logging.ERROR: NEON_RED,
            logging.CRITICAL: NEON_RED + BRIGHT,
        }

        class NeonConsoleFormatter(SensitiveFormatter):
            """Custom formatter for colorful console output using local time."""
            _level_colors = level_colors
            _tz = TIMEZONE # Use the configured local timezone

            def format(self, record: logging.LogRecord) -> str:
                level_color = self._level_colors.get(record.levelno, NEON_BLUE)
                # Format using local time from the TIMEZONE object
                log_fmt = (f"{NEON_BLUE}%(asctime)s{RESET} - "
                           f"{level_color}%(levelname)-8s{RESET} - "
                           f"{NEON_PURPLE}[%(name)s]{RESET} - %(message)s")
                # Create a formatter specific to this record to set the local time converter
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
                # Lambda function to get current time in the configured timezone
                formatter.converter = lambda *args: datetime.now(self._tz).timetuple() # type: ignore [assignment]
                # Apply the base SensitiveFormatter's redaction AFTER color formatting
                return super(NeonConsoleFormatter, formatter).format(record)

        sh.setFormatter(NeonConsoleFormatter("%(message)s")) # Base format string, color handles details

        # Set console log level from environment variable or default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent messages from reaching the root logger
    return logger

# --- Initial Logger Setup ---
init_logger = setup_logger("init")
init_logger.info(f"{Fore.MAGENTA}{BRIGHT}Pyrmethus Volumatic Bot v{BOT_VERSION} Initializing...{Style.RESET_ALL}")
init_logger.info(f"Using Timezone: {TIMEZONE_STR}")

# --- Configuration Loading & Validation ---

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures all keys from default_config exist in config. Adds missing keys.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default values and structure.
        parent_key: The key path used for logging messages (internal use).

    Returns:
        A tuple containing:
            - The updated configuration dictionary.
            - A boolean indicating if any changes were made.
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it from defaults
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' = {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially updates the configuration file.

    Handles file not found, JSON errors, missing keys, and basic type/range validation.
    Updates the global QUOTE_CURRENCY variable based on the loaded config.

    Args:
        filepath: Path to the configuration JSON file.

    Returns:
        The loaded (and potentially updated) configuration dictionary.
    """
    global QUOTE_CURRENCY # Declare intent to modify the global variable
    init_logger.info(f"{Fore.CYAN}# Loading configuration from '{filepath}'...{Style.RESET_ALL}")

    # Define the default configuration structure and values
    default_config = {
        "trading_pairs": ["BTC/USDT"],
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25, # Note: Not currently used in provided code snippet
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01, # 1% risk per trade
        "leverage": 20,
        "max_concurrent_positions": 1,
        "quote_currency": "USDT", # Default quote currency
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER),
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER),
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT,
            "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT,
            "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, # e.g., entry within 0.5% of OB edge
            "ob_exit_proximity_factor": 1.001  # e.g., SL moved if price gets within 0.1%
        },
        "protection": {
            "enable_trailing_stop": True,
            "trailing_stop_callback_rate": 0.005, # 0.5% TSL distance
            "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves 0.3% past entry
            "enable_break_even": True,
            "break_even_trigger_atr_multiple": 1.0, # Move SL to BE when price moves 1 * ATR
            "break_even_offset_ticks": 2, # Move SL BE_Ticks * price_step above/below entry
            "initial_stop_loss_atr_multiple": 1.8, # Initial SL placed 1.8 * ATR away
            "initial_take_profit_atr_multiple": 0.7 # Initial TP placed 0.7 * ATR away (can be 0 to disable)
        }
    }

    config_needs_saving: bool = False
    loaded_config: Dict[str, Any] = {}

    # 1. Handle File Existence
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Set global from default
            return default_config
        except IOError as e:
            init_logger.critical(f"{NEON_RED}FATAL: Error creating config file '{filepath}': {e}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Set global from default
            return default_config # Return default config in case of write error

    # 2. Load Existing File
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
            raise TypeError("Configuration file does not contain a valid JSON object.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Set global from default
            return default_config
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}FATAL: Error recreating config file: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Set global from default
            return default_config
    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Set global from default
        return default_config

    # 3. Ensure All Keys Exist (Add missing ones from defaults)
    try:
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True # Mark for saving if keys were added

        # 4. Validation Logic
        def validate_numeric(cfg: Dict, key_path: str, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False) -> bool:
            """
            Validates a numeric value within the config. Corrects if invalid, using default.

            Args:
                cfg: The config dictionary (potentially nested).
                key_path: Dot-separated path to the key (e.g., "protection.leverage").
                min_val: Minimum allowed value (inclusive unless is_strict_min).
                max_val: Maximum allowed value (inclusive).
                is_strict_min: If True, value must be strictly greater than min_val.
                is_int: If True, value must be an integer.
                allow_zero: If True, zero is allowed even if outside min/max range.

            Returns:
                True if the value was corrected, False otherwise.
            """
            nonlocal config_needs_saving # Allow modification of the outer scope variable
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse dictionaries to reach the target key
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key) # Get default for fallback
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid key path '{key_path}'. Cannot validate.")
                return False # Cannot validate if path is wrong

            if original_val is None:
                init_logger.warning(f"Config validation: Value missing at '{key_path}'. Using default: {repr(default_val)}")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val # Start with the current value

            try:
                # Convert to Decimal for robust comparison
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))

                # Check range
                min_check_passed = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check_passed = min_check_passed and num_val <= max_dec
                zero_allowed = allow_zero and num_val.is_zero()

                if not range_check_passed and not zero_allowed:
                    raise ValueError("Value outside allowed range.")

                # Check type (int/float) and potentially correct minor type mismatches (e.g., int where float expected)
                target_type = int if is_int else float
                converted_val = target_type(num_val)
                needs_type_correction = False

                if isinstance(original_val, bool): # Explicitly disallow bools for numeric fields
                     raise TypeError("Boolean value found where numeric value was expected.")
                elif is_int and not isinstance(original_val, int):
                    needs_type_correction = True # e.g., 10.0 should be 10
                elif not is_int and not isinstance(original_val, float):
                    # Allow ints where floats are expected, convert them
                    if isinstance(original_val, int):
                        converted_val = float(original_val)
                        needs_type_correction = True
                    else: # Neither float nor int, needs correction
                         needs_type_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9: # Check float precision issues
                    needs_type_correction = True
                elif isinstance(original_val, int) and original_val != converted_val: # Check int->float conversion differences
                     needs_type_correction = True


                if needs_type_correction:
                    init_logger.info(f"{NEON_YELLOW}Config Update: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val
                    corrected = True

            except (ValueError, InvalidOperation, TypeError) as e:
                # Handle conversion errors, range errors, type errors
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(
                    f"{NEON_YELLOW}Config Validation: Invalid value for '{key_path}'='{repr(original_val)}'. "
                    f"Using default: {repr(default_val)}. Error: {e}. "
                    f"Expected type: {'int' if is_int else 'float'}, Range: {range_str}{RESET}"
                )
                final_val = default_val # Use default value
                corrected = True

            if corrected:
                current_level[leaf_key] = final_val # Update the config dictionary
                config_needs_saving = True # Mark for saving

            return corrected # Return whether a correction was made

        # --- Perform Validations ---
        init_logger.debug("# Validating configuration parameters...")

        # Validate top-level parameters
        if not isinstance(updated_config.get("trading_pairs"), list) or not all(isinstance(s, str) and s and '/' in s for s in updated_config.get("trading_pairs", [])):
            init_logger.warning(f"Invalid 'trading_pairs' format. Using default: {default_config['trading_pairs']}.")
            updated_config["trading_pairs"] = default_config["trading_pairs"]
            config_needs_saving = True
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"Invalid 'interval' value '{updated_config.get('interval')}'. Using default: '{default_config['interval']}'.")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('0.5'), is_strict_min=True) # Risk > 0% and <= 50%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Allow 0 for spot? (Maybe needs adjustment)
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)

        # Validate boolean flags
        if not isinstance(updated_config.get("enable_trading"), bool):
            init_logger.warning(f"Invalid 'enable_trading' value. Using default: '{default_config['enable_trading']}'.")
            updated_config["enable_trading"] = default_config["enable_trading"]
            config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
            init_logger.warning(f"Invalid 'use_sandbox' value. Using default: '{default_config['use_sandbox']}'.")
            updated_config["use_sandbox"] = default_config["use_sandbox"]
            config_needs_saving = True

        # Validate quote currency (must be non-empty string)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
            init_logger.warning(f"Invalid 'quote_currency'. Using default: '{default_config['quote_currency']}'.")
            updated_config["quote_currency"] = default_config["quote_currency"]
            config_needs_saving = True

        # Validate strategy parameters
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # 1.0 to 1.1 (e.g., 0% to 10% range)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # 1.0 to 1.1

        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
            init_logger.warning(f"Invalid strategy_params.ob_source. Using default: '{DEFAULT_OB_SOURCE}'.")
            updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
            config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
            init_logger.warning(f"Invalid strategy_params.ob_extend. Using default: '{DEFAULT_OB_EXTEND}'.")
            updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
            config_needs_saving = True

        # Validate protection parameters
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
            init_logger.warning(f"Invalid protection.enable_trailing_stop. Using default.")
            updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]
            config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
            init_logger.warning(f"Invalid protection.enable_break_even. Using default.")
            updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]
            config_needs_saving = True

        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.1'), is_strict_min=True) # 0.01% to 10%
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.1'), allow_zero=True) # 0% to 10%
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('20.0'), is_strict_min=True) # SL > 0.1 ATR
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('20.0'), allow_zero=True) # TP can be 0 (disabled)

        # 5. Save Updated Config if Necessary
        if config_needs_saving:
             try:
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Config file '{filepath}' updated with missing/corrected values.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated config to '{filepath}': {save_err}{RESET}", exc_info=True)

        # Set the global quote currency AFTER loading and validation
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.info(f"Global quote currency set to: {NEON_YELLOW}{QUOTE_CURRENCY}{RESET}")
        init_logger.info(f"{Fore.CYAN}# Configuration loading/validation complete.{Style.RESET_ALL}")
        return updated_config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}FATAL: Unhandled error during config processing: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT") # Ensure global is set even on critical failure
        return default_config

# --- Load Configuration ---
CONFIG = load_config(CONFIG_FILE)
# Note: QUOTE_CURRENCY global variable is now set based on the loaded config.

# --- Exchange Interaction Functions ---

def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and configures the CCXT Bybit exchange object.

    Handles sandbox mode, loads markets with retries, and performs an
    initial balance check.

    Args:
        logger: The logger instance to use for logging.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    lg.info(f"{Fore.CYAN}# Initializing Bybit exchange connection...{Style.RESET_ALL}")
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable built-in rate limiting
            'options': {
                'defaultType': 'linear',    # Prefer linear contracts
                'adjustForTimeDifference': True, # Adjust for clock skew
                # Timeouts for various operations (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000, # Longer timeout for potentially large kline fetches
            }
        }
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode based on config
        is_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< SANDBOX MODE ACTIVE >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< LIVE TRADING ACTIVE >>> !!!{RESET}")

        # Load market data with retries
        lg.info(f"Loading market data for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Market load attempt {attempt + 1}...")
                # Force reload on retries to ensure fresh data
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Market data loaded successfully ({len(exchange.markets)} symbols found).{RESET}")
                    markets_loaded = True
                    break
                else:
                    # Handle case where load_markets succeeds but returns empty data
                    last_market_error = ValueError("Market data returned empty from exchange.")
                    lg.warning(f"Market data appears empty (Attempt {attempt + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}): {e}.")
            except ccxt.AuthenticationError as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API keys. Exiting.{RESET}")
                return None # Fatal error
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None # Fatal error

            if not markets_loaded and attempt < MAX_API_RETRIES:
                delay = RETRY_DELAY_SECONDS * (attempt + 1)
                lg.warning(f"Retrying market load in {delay}s...")
                time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load market data after multiple attempts. Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"Exchange initialized: {exchange.id} | Sandbox Mode: {is_sandbox}")

        # Perform initial balance check
        lg.info(f"Checking initial balance for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err:
            # This shouldn't happen if market loading succeeded, but check anyway
            lg.critical(f"{NEON_RED}Authentication error during initial balance check: {auth_err}. Exiting.{RESET}")
            return None
        except Exception as balance_err:
            # Log non-fatal balance check errors as warnings
            lg.warning(f"{NEON_YELLOW}Initial balance check failed: {balance_err}.{RESET}", exc_info=False) # Don't need full traceback usually

        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial balance confirmed: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            lg.info(f"{Fore.CYAN}# Exchange initialization complete.{Style.RESET_ALL}")
            return exchange
        else:
            lg.error(f"{NEON_RED}Could not confirm initial balance for {QUOTE_CURRENCY}.{RESET}")
            # Decide whether to proceed based on trading flag
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but initial balance check failed. Cannot proceed safely. Exiting.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without initial balance confirmation.{RESET}")
                lg.info(f"{Fore.CYAN}# Exchange initialization complete (no balance confirmation).{Style.RESET_ALL}")
                return exchange

    except Exception as e:
        lg.critical(f"{NEON_RED}Critical failure during exchange initialization: {e}{RESET}", exc_info=True)
        return None

def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True, allow_negative: bool = False) -> Optional[Decimal]:
    """
    Safely converts a value (often from market data) to a Decimal.

    Handles None, empty strings, and invalid number formats.

    Args:
        value: The value to convert.
        field_name: Name of the field being converted (for logging).
        allow_zero: If False, returns None if the value is zero.
        allow_negative: If False, returns None if the value is negative.

    Returns:
        The converted Decimal, or None if conversion fails or constraints are not met.
    """
    if value is None:
        return None
    try:
        s_val = str(value).strip()
        if not s_val: # Handle empty strings
             return None
        d_val = Decimal(s_val)
        if not allow_zero and d_val.is_zero():
            # init_logger.debug(f"Zero value disallowed for {field_name}: {value}") # Optional debug log
            return None
        if not allow_negative and d_val < Decimal('0'):
            # init_logger.debug(f"Negative value disallowed for {field_name}: {value}") # Optional debug log
            return None
        return d_val
    except (InvalidOperation, TypeError, ValueError):
        # init_logger.warning(f"Could not convert market field '{field_name}' to Decimal: {value}") # Optional warning
        return None

def _format_price(exchange: ccxt.Exchange, symbol: str, price: Union[Decimal, float, str]) -> Optional[str]:
    """
    Formats a price value according to the market's precision rules using ccxt.

    Args:
        exchange: The ccxt exchange instance.
        symbol: The market symbol (e.g., 'BTC/USDT').
        price: The price to format (as Decimal, float, or string).

    Returns:
        The formatted price string, or None if formatting fails or price is invalid.
    """
    try:
        price_decimal = Decimal(str(price))
        # Ensure price is positive before formatting
        if price_decimal <= Decimal('0'):
            init_logger.warning(f"Attempted to format non-positive price '{price}' for {symbol}.")
            return None
        # Use ccxt's price_to_precision for correct formatting
        formatted_str = exchange.price_to_precision(symbol, float(price_decimal))
        # Double-check formatted string represents a positive value (handles potential edge cases)
        return formatted_str if Decimal(formatted_str) > Decimal('0') else None
    except (InvalidOperation, ValueError, TypeError, cc
