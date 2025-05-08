# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.7: Comprehensive enhancements based on v1.1.6 review.
#               Improved docstrings, type hinting (TypedDict), error handling,
#               Decimal usage, API interaction robustness, config validation,
#               logging clarity, and overall code structure.

"""
Pyrmethus Volumatic Bot: A Python Trading Bot for Bybit V5

This bot implements a trading strategy based on the combination of:
1.  **Volumatic Trend:** An EMA/SWMA crossover system with ATR-based bands,
    incorporating normalized volume analysis.
2.  **Pivot Order Blocks (OBs):** Identifying potential support/resistance zones
    based on pivot highs and lows derived from candle wicks or bodies.

Key Features:
-   Connects to Bybit V5 API (Linear/Inverse contracts, Spot potentially adaptable).
-   Supports Sandbox (testnet) and Live trading environments.
-   Fetches OHLCV data and calculates strategy indicators using pandas and pandas-ta.
-   Identifies Volumatic Trend direction and changes.
-   Detects Pivot Highs/Lows and creates/manages Order Blocks.
-   Generates BUY/SELL/EXIT signals based on trend alignment and OB proximity.
-   Calculates position size based on risk percentage and stop-loss distance.
-   Sets leverage for contract markets.
-   Places market orders to enter and exit positions.
-   Advanced Position Management:
    -   Sets initial Stop Loss (SL) and Take Profit (TP) based on ATR multiples.
    -   Implements Trailing Stop Loss (TSL) with configurable activation and callback.
    -   Implements Break-Even (BE) stop adjustment based on profit targets.
-   Robust API interaction with retries and error handling (Network, Rate Limit, Auth).
-   Secure handling of API credentials via `.env` file.
-   Flexible configuration via `config.json` with validation and defaults.
-   Detailed logging with Neon color scheme for console output and rotating file logs.
-   Sensitive data redaction in logs.
-   Graceful shutdown handling.
"""

# --- Core Libraries ---
import hashlib
import hmac
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from zoneinfo import ZoneInfo # Requires tzdata package (pip install tzdata)

# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy (pip install numpy)
import pandas as pd # Requires pandas (pip install pandas)
import pandas_ta as ta # Requires pandas_ta (pip install pandas_ta)
import requests # Requires requests (pip install requests)
import ccxt # Requires ccxt (pip install ccxt)
from colorama import Fore, Style, init as colorama_init # Requires colorama (pip install colorama)
from dotenv import load_dotenv # Requires python-dotenv (pip install python-dotenv)

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision globally for high-accuracy calculations
colorama_init(autoreset=True) # Initialize Colorama for console colors, resetting after each print
load_dotenv() # Load environment variables from a .env file in the project root

# --- Constants ---
# API Credentials (Loaded securely from .env file)
API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
API_SECRET: Optional[str] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Critical error if API keys are missing
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file")

# Configuration and Logging Files/Directories
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
TIMEZONE_STR: str = os.getenv("TIMEZONE", "America/Chicago") # Default timezone if not set in .env
try:
    # Attempt to load user-specified timezone, fallback to UTC if tzdata is not installed or invalid
    # Example: "America/Chicago", "Europe/London", "Asia/Tokyo", "UTC"
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except Exception as tz_err:
    print(f"{Fore.RED}Failed to initialize timezone '{TIMEZONE_STR}'. Error: {tz_err}. "
          f"Install 'tzdata' package (`pip install tzdata`) or check timezone name. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")
    TIMEZONE_STR = "UTC" # Update string to reflect fallback

# API Interaction Settings
MAX_API_RETRIES: int = 3        # Maximum number of consecutive retries for failed API calls
RETRY_DELAY_SECONDS: int = 5    # Base delay (in seconds) between API retries (may increase exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Wait time after placing an order before fetching position details
LOOP_DELAY_SECONDS: int = 15    # Default delay between trading cycles (can be overridden in config.json)
BYBIT_API_KLINE_LIMIT: int = 1000 # Maximum number of Klines Bybit V5 API returns per request

# Timeframes Mapping
# Bybit API expects specific strings for intervals
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750 # Default number of klines to fetch if not specified or less than strategy needs
MAX_DF_LEN: int = 2000         # Internal limit to prevent excessive memory usage by the Pandas DataFrame

# Strategy Defaults (Used if values are missing, invalid, or out of range in config.json)
DEFAULT_VT_LENGTH: int = 40             # Volumatic Trend EMA/SWMA length
DEFAULT_VT_ATR_PERIOD: int = 200        # ATR period for Volumatic Trend bands
DEFAULT_VT_VOL_EMA_LENGTH: int = 950    # Volume Normalization EMA length (Adjusted: 1000 often > API limit)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0  # ATR multiplier for Volumatic Trend bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Currently unused step ATR multiplier
DEFAULT_OB_SOURCE: str = "Wicks"        # Order Block source ("Wicks" or "Body")
DEFAULT_PH_LEFT: int = 10               # Pivot High lookback periods (left)
DEFAULT_PH_RIGHT: int = 10              # Pivot High lookback periods (right)
DEFAULT_PL_LEFT: int = 10               # Pivot Low lookback periods (left)
DEFAULT_PL_RIGHT: int = 10              # Pivot Low lookback periods (right)
DEFAULT_OB_EXTEND: bool = True          # Extend Order Block visuals to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50          # Max number of active Order Blocks to track/display

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")
QUOTE_CURRENCY: str = "USDT" # Placeholder, will be updated by load_config()

# Logging Colors (Neon Theme for Console Output)
NEON_GREEN: str = Fore.LIGHTGREEN_EX
NEON_BLUE: str = Fore.CYAN
NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW
NEON_RED: str = Fore.LIGHTRED_EX
NEON_CYAN: str = Fore.CYAN
RESET: str = Style.RESET_ALL
BRIGHT: str = Style.BRIGHT
DIM: str = Style.DIM

# Ensure log directory exists before setting up loggers
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Type Definitions for Structured Data ---
class OrderBlock(TypedDict):
    """Represents a bullish or bearish Order Block identified on the chart."""
    id: str                 # Unique identifier (e.g., "B_231026143000")
    type: str               # 'bull' or 'bear'
    left_idx: pd.Timestamp  # Timestamp of the candle that formed the OB
    right_idx: pd.Timestamp # Timestamp of the last candle the OB extends to (or violation candle)
    top: Decimal            # Top price level of the OB
    bottom: Decimal         # Bottom price level of the OB
    active: bool            # True if the OB is currently considered valid
    violated: bool          # True if the price has closed beyond the OB boundary

class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis process."""
    dataframe: pd.DataFrame         # The DataFrame with all calculated indicators (Decimal values)
    last_close: Decimal             # The closing price of the most recent candle
    current_trend_up: Optional[bool]# True if Volumatic Trend is up, False if down, None if undetermined
    trend_just_changed: bool        # True if the trend flipped on the last candle
    active_bull_boxes: List[OrderBlock] # List of currently active bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active bearish OBs
    vol_norm_int: Optional[int]     # Normalized volume indicator (0-100+, integer) for the last candle
    atr: Optional[Decimal]          # ATR value for the last candle (must be positive)
    upper_band: Optional[Decimal]   # Volumatic Trend upper band value for the last candle
    lower_band: Optional[Decimal]   # Volumatic Trend lower band value for the last candle

class MarketInfo(TypedDict):
    """Standardized market information dictionary."""
    # Standard CCXT fields (may vary slightly by exchange)
    id: str
    symbol: str
    base: str
    quote: str
    settle: Optional[str]
    baseId: str
    quoteId: str
    settleId: Optional[str]
    type: str # 'spot', 'swap', 'future', etc.
    spot: bool
    margin: bool
    swap: bool
    future: bool
    option: bool
    active: bool
    contract: bool
    linear: bool
    inverse: bool
    quanto: bool
    taker: float
    maker: float
    contractSize: Optional[Any] # Often float or int, convert to Decimal when used
    expiry: Optional[int]
    expiryDatetime: Optional[str]
    strike: Optional[float]
    optionType: Optional[str]
    precision: Dict[str, Any] # {'amount': float/str, 'price': float/str, 'cost': float/str, 'base': float, 'quote': float} - Convert to Decimal
    limits: Dict[str, Any] # {'leverage': {'min': float, 'max': float}, 'amount': {'min': float/str, 'max': float/str}, 'price': {'min': float, 'max': float}, 'cost': {'min': float/str, 'max': float/str}} - Convert to Decimal
    info: Dict[str, Any] # Exchange-specific raw info
    # Custom added fields for convenience
    is_contract: bool
    is_linear: bool
    is_inverse: bool
    contract_type_str: str
    min_amount_decimal: Optional[Decimal]
    max_amount_decimal: Optional[Decimal]
    min_cost_decimal: Optional[Decimal]
    max_cost_decimal: Optional[Decimal]
    amount_precision_step_decimal: Optional[Decimal]
    price_precision_step_decimal: Optional[Decimal]
    contract_size_decimal: Decimal

class PositionInfo(TypedDict):
    """Standardized position information dictionary."""
    id: Optional[str]
    symbol: str
    timestamp: Optional[int]
    datetime: Optional[str]
    contracts: Optional[float] # Use size_decimal instead
    contractSize: Optional[Any] # Convert to Decimal when used
    side: Optional[str] # 'long' or 'short'
    notional: Optional[Any] # Convert to Decimal when used
    leverage: Optional[Any] # Convert to Decimal when used
    unrealizedPnl: Optional[Any] # Convert to Decimal when used
    realizedPnl: Optional[Any]
    collateral: Optional[Any]
    entryPrice: Optional[Any] # Convert to Decimal when used
    markPrice: Optional[Any]
    liquidationPrice: Optional[Any] # Convert to Decimal when used
    marginMode: Optional[str] # 'cross' or 'isolated'
    hedged: Optional[bool]
    maintenanceMargin: Optional[Any]
    maintenanceMarginPercentage: Optional[float]
    initialMargin: Optional[Any]
    initialMarginPercentage: Optional[float]
    marginRatio: Optional[float]
    lastUpdateTimestamp: Optional[int]
    info: Dict[str, Any] # Exchange-specific raw info
    # Custom added/parsed fields
    size_decimal: Decimal # Parsed position size as Decimal (positive for long, negative for short)
    stopLossPrice: Optional[str] # Parsed from info if available
    takeProfitPrice: Optional[str] # Parsed from info if available
    trailingStopLoss: Optional[str] # Parsed from info if available (Bybit TSL distance)
    tslActivationPrice: Optional[str] # Parsed from info if available (Bybit TSL activation price)


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """
    Custom logging formatter that redacts sensitive API keys/secrets
    from log messages to prevent accidental exposure in log files or console.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing API keys/secrets with placeholders."""
        msg = super().format(record)
        # Ensure API keys exist and are strings before attempting replacement
        if API_KEY and isinstance(API_KEY, str):
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET and isinstance(API_SECRET, str):
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a dedicated logger instance for a specific context (e.g., 'init', 'BTC/USDT').

    Configures both a console handler (with Neon colors, level filtering based on
    CONSOLE_LOG_LEVEL environment variable, and timezone-aware timestamps)
    and a rotating file handler (capturing DEBUG level and above, with sensitive
    data redaction).

    Args:
        name: The name for the logger (e.g., "init", "BTC/USDT"). Used for filtering
              and naming the log file.

    Returns:
        The configured logging.Logger instance. Returns existing instance if already configured.
    """
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize name for filenames/logger keys
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger instance already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels; handlers below will filter output

    # --- File Handler (DEBUG level, Rotating, Redaction) ---
    try:
        # Rotate log file when it reaches 10MB, keep 5 backup files
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Use SensitiveFormatter for detailed file log, redacting API keys/secrets
        ff = SensitiveFormatter(
            "%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S' # Standard date format for files
        )
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG) # Log everything from DEBUG upwards to the file
        logger.addHandler(fh)
    except Exception as e:
        # Use print for errors during logger setup itself, as logger might not be functional
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # --- Console Handler (Configurable Level, Neon Colors, Timezone) ---
    try:
        sh = logging.StreamHandler(sys.stdout) # Explicitly use stdout
        # Define color mapping for different log levels
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,      # Dim Cyan for Debug
            logging.INFO: NEON_BLUE,             # Bright Cyan for Info
            logging.WARNING: NEON_YELLOW,        # Bright Yellow for Warning
            logging.ERROR: NEON_RED,             # Bright Red for Error
            logging.CRITICAL: NEON_RED + BRIGHT, # Bright Red + Bold for Critical
        }

        # Custom formatter for console output with colors and timezone-aware timestamps
        class NeonConsoleFormatter(SensitiveFormatter):
            """Applies Neon color scheme and configured timezone to console log messages."""
            def format(self, record: logging.LogRecord) -> str:
                level_color = level_colors.get(record.levelno, NEON_BLUE) # Default to Info color
                # Format: Time(TZ) - Level - [LoggerName] - Message
                log_fmt = (
                    f"{NEON_BLUE}%(asctime)s{RESET} - "
                    f"{level_color}%(levelname)-8s{RESET} - "
                    f"{NEON_PURPLE}[%(name)s]{RESET} - "
                    f"%(message)s"
                )
                # Create a formatter instance with the defined format and date style
                formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S') # Use Time only for console clarity
                # Ensure timestamps reflect the configured TIMEZONE
                formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
                # Apply sensitive data redaction before returning the final message
                # We inherit from SensitiveFormatter, so super().format handles redaction
                return super().format(record)

        sh.setFormatter(NeonConsoleFormatter())
        # Get desired console log level from environment variable (e.g., DEBUG, INFO, WARNING), default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Fallback to INFO if invalid level provided
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent log messages from bubbling up to the root logger
    return logger

# Initialize the 'init' logger early for messages during startup and configuration loading
init_logger = setup_logger("init")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively checks if all keys from `default_config` exist in `config`.
    If a key is missing, it's added to `config` with the default value from `default_config`.
    Logs any keys that were added using the `init_logger`.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default structure and values.
        parent_key: Used internally for tracking nested key paths for logging (e.g., "strategy_params.vt_length").

    Returns:
        A tuple containing:
            - The potentially updated configuration dictionary.
            - A boolean indicating whether any changes were made (True if keys were added).
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            # Key is missing, add it with the default value
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default value: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dicts, recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If nested dict was changed, update the parent dict and mark as changed
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add type mismatch check here, but validation below handles it more robustly.
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads, validates, and potentially updates configuration from a JSON file.

    Steps:
    1. Checks if the config file exists. If not, creates a default one.
    2. Loads the JSON data from the file. Handles decoding errors by recreating the default file.
    3. Ensures all keys from the default structure exist in the loaded config, adding missing ones.
    4. Performs detailed type and range validation on critical numeric and string parameters.
       - Uses default values and logs warnings/corrections if validation fails.
       - Leverages Decimal for robust numeric comparisons.
    5. If any keys were added or values corrected, saves the updated config back to the file.
    6. Updates the global `QUOTE_CURRENCY` based on the validated config.
    7. Returns the validated (and potentially updated) configuration dictionary.

    Args:
        filepath: The path to the configuration JSON file (e.g., "config.json").

    Returns:
        The loaded and validated configuration dictionary. Returns default configuration
        if the file cannot be read, created, or parsed, or if validation encounters
        unexpected errors.
    """
    # Define the default configuration structure and values
    default_config = {
        # General Settings
        "interval": "5",                # Default timeframe (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # Base delay for API retries
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Default klines to fetch per cycle
        "orderbook_limit": 25,          # (Currently Unused) Limit for order book fetching if implemented
        "enable_trading": False,        # Master switch for placing actual trades
        "use_sandbox": True,            # Use Bybit's testnet environment
        "risk_per_trade": 0.01,         # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                 # Default leverage for contract trading (0 to disable setting)
        "max_concurrent_positions": 1,  # (Currently Unused) Max open positions allowed simultaneously
        "quote_currency": "USDT",       # The currency to calculate balance and risk against
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between trading cycles
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait after order before checking position

        # Strategy Parameters (Volumatic Trend + OB)
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": float(DEFAULT_VT_ATR_MULTIPLIER), # Store as float in JSON
            "vt_step_atr_multiplier": float(DEFAULT_VT_STEP_ATR_MULTIPLIER), # Unused, store as float
            "ob_source": DEFAULT_OB_SOURCE, # "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, # Pivot High lookbacks
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT, # Pivot Low lookbacks
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, # Price must be <= OB top * factor (long) or >= OB bottom / factor (short)
            "ob_exit_proximity_factor": 1.001   # Exit if price >= Bear OB top * factor or <= Bull OB bottom / factor
        },
        # Position Protection Parameters
        "protection": {
             "enable_trailing_stop": True,      # Use trailing stop loss
             "trailing_stop_callback_rate": 0.005, # TSL distance as % of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Activate TSL when price moves this % from entry (e.g., 0.003 = 0.3%)
             "enable_break_even": True,         # Move SL to entry + offset when profit target hit
             "break_even_trigger_atr_multiple": 1.0, # Profit needed (in ATR multiples) to trigger BE
             "break_even_offset_ticks": 2,       # Move SL this many ticks beyond entry for BE
             "initial_stop_loss_atr_multiple": 1.8, # Initial SL distance in ATR multiples
             "initial_take_profit_atr_multiple": 0.7 # Initial TP distance in ATR multiples (0 to disable)
        }
    }
    config_needs_saving: bool = False
    loaded_config: Dict[str, Any] = {}

    # --- File Existence Check & Default Creation ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            # Update global QUOTE_CURRENCY immediately after creating default
            global QUOTE_CURRENCY
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Return defaults immediately
        except IOError as e:
            init_logger.error(f"{NEON_RED}FATAL: Error creating default config file '{filepath}': {e}{RESET}")
            init_logger.warning("Using internal default configuration values. Bot may not function correctly.")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config # Use internal defaults if file creation fails

    # --- File Loading ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        if not isinstance(loaded_config, dict):
             raise TypeError("Configuration file does not contain a valid JSON object.")
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating default file.{RESET}")
        try: # Attempt to recreate the file with defaults if corrupted
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4, ensure_ascii=False)
            init_logger.info(f"{NEON_GREEN}Recreated default config file due to decode error: {filepath}{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
        except IOError as e_create:
            init_logger.error(f"{NEON_RED}FATAL: Error recreating default config file after decode error: {e_create}. Using internal defaults.{RESET}")
            QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
            return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}FATAL: Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.warning("Using internal default configuration values. Bot may not function correctly.")
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config

    # --- Validation and Merging ---
    try:
        # Ensure all keys from default_config exist in loaded_config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True # Mark for saving if keys were added

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """
            Validates a numeric config value at `key_path` (e.g., "protection.leverage").

            Checks type (int/float), range [min_val, max_val] or (min_val, max_val] if strict.
            Uses the default value from `default_config` and logs a warning/info if correction needed.
            Updates the `cfg` dictionary in place if correction is made.
            Uses Decimal for robust comparisons.

            Returns:
                True if a correction was made, False otherwise.
            """
            nonlocal config_needs_saving # Allow modification of the outer scope variable
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse nested dictionaries to reach the target key
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid path '{key_path}'. Cannot validate.")
                return False # Path itself is wrong

            if original_val is None:
                # This case should be rare due to _ensure_config_keys, but handle defensively
                init_logger.warning(f"Config validation: Key missing at '{key_path}' during numeric check. Using default: {repr(default_val)}")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val # Start with the original value

            try:
                # Convert to Decimal for robust comparison, handle potential strings
                num_val = Decimal(str(original_val))
                min_dec = Decimal(str(min_val))
                max_dec = Decimal(str(max_val))

                # Check range
                min_check = num_val > min_dec if is_strict_min else num_val >= min_dec
                range_check = min_check and num_val <= max_dec
                # Check if zero is allowed and value is zero, bypassing range check if so
                zero_ok = allow_zero and num_val == Decimal(0)

                if not range_check and not zero_ok:
                    raise ValueError("Value out of allowed range.")

                # Check type and convert if necessary
                target_type = int if is_int else float
                # Attempt conversion to target type
                converted_val = target_type(num_val)

                # Check if type or value changed significantly after conversion
                # This ensures int remains int, float remains float (within tolerance)
                needs_correction = False
                if isinstance(original_val, bool): # Don't try to convert bools here
                     raise TypeError("Boolean value found where numeric expected.")
                elif is_int and not isinstance(original_val, int):
                    needs_correction = True
                elif not is_int and not isinstance(original_val, float):
                     # If float expected, allow int input but convert it
                    if isinstance(original_val, int):
                        needs_correction = True
                    else: # Input is neither float nor int (e.g., string)
                        needs_correction = True
                elif isinstance(original_val, float) and abs(original_val - converted_val) > 1e-9:
                    # Check if float value changed significantly after potential Decimal conversion
                    needs_correction = True
                elif isinstance(original_val, int) and original_val != converted_val:
                     # Should not happen if is_int=True, but check defensively
                     needs_correction = True

                if needs_correction:
                    init_logger.info(f"{NEON_YELLOW}Config: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val
                    corrected = True

            except (ValueError, InvalidOperation, TypeError) as e:
                # Handle cases where value is non-numeric, out of range, or conversion fails
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{repr(original_val)}'. Using default: {repr(default_val)}. Error: {e}. Expected range: {range_str}{RESET}")
                final_val = default_val # Use the default value
                corrected = True

            # If a correction occurred, update the config dictionary and mark for saving
            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        # --- Apply Validations to Specific Config Keys ---
        # General
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.warning(f"{NEON_YELLOW}Invalid config 'interval' '{updated_config.get('interval')}'. Valid: {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 50, MAX_DF_LEN, is_int=True) # Ensure minimum useful fetch limit
        validate_numeric(updated_config, "risk_per_trade", Decimal('0'), Decimal('1'), is_strict_min=True) # Risk must be > 0 and <= 1
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True, allow_zero=True) # Leverage 0 means no setting attempt
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        if not isinstance(updated_config.get("quote_currency"), str) or not updated_config.get("quote_currency"):
             init_logger.warning(f"Invalid 'quote_currency' '{updated_config.get('quote_currency')}'. Must be a non-empty string. Using default '{default_config['quote_currency']}'.")
             updated_config["quote_currency"] = default_config["quote_currency"]
             config_needs_saving = True
        if not isinstance(updated_config.get("enable_trading"), bool):
             init_logger.warning(f"Invalid 'enable_trading' value '{updated_config.get('enable_trading')}'. Must be true or false. Using default '{default_config['enable_trading']}'.")
             updated_config["enable_trading"] = default_config["enable_trading"]
             config_needs_saving = True
        if not isinstance(updated_config.get("use_sandbox"), bool):
             init_logger.warning(f"Invalid 'use_sandbox' value '{updated_config.get('use_sandbox')}'. Must be true or false. Using default '{default_config['use_sandbox']}'.")
             updated_config["use_sandbox"] = default_config["use_sandbox"]
             config_needs_saving = True

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True) # Allow long ATR period
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True) # Allow long Vol EMA
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # e.g., 1.005 = 0.5% proximity
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1) # e.g., 1.001 = 0.1% proximity
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source '{updated_config['strategy_params']['ob_source']}'. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
             config_needs_saving = True
        if not isinstance(updated_config["strategy_params"].get("ob_extend"), bool):
             init_logger.warning(f"Invalid strategy_params.ob_extend value '{updated_config['strategy_params']['ob_extend']}'. Must be true or false. Using default '{DEFAULT_OB_EXTEND}'.")
             updated_config["strategy_params"]["ob_extend"] = DEFAULT_OB_EXTEND
             config_needs_saving = True

        # Protection Params
        if not isinstance(updated_config["protection"].get("enable_trailing_stop"), bool):
             init_logger.warning(f"Invalid protection.enable_trailing_stop value '{updated_config['protection']['enable_trailing_stop']}'. Must be true or false. Using default '{default_config['protection']['enable_trailing_stop']}'.")
             updated_config["protection"]["enable_trailing_stop"] = default_config["protection"]["enable_trailing_stop"]
             config_needs_saving = True
        if not isinstance(updated_config["protection"].get("enable_break_even"), bool):
             init_logger.warning(f"Invalid protection.enable_break_even value '{updated_config['protection']['enable_break_even']}'. Must be true or false. Using default '{default_config['protection']['enable_break_even']}'.")
             updated_config["protection"]["enable_break_even"] = default_config["protection"]["enable_break_even"]
             config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", Decimal('0.0001'), Decimal('0.5'), is_strict_min=True) # Must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", Decimal('0'), Decimal('0.5'), allow_zero=True) # 0 means activate immediately
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", Decimal('0.1'), Decimal('10.0'))
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # 0 means move SL exactly to entry
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", Decimal('0.1'), Decimal('100.0'))
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", Decimal('0'), Decimal('100.0'), allow_zero=True) # 0 disables initial TP

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             try:
                 # Convert potentially corrected values back to standard types for JSON
                 # (e.g., if defaults were used which might be Decimals internally)
                 # json.dumps handles basic types (int, float, str, bool, list, dict)
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(updated_config, f_write, indent=4, ensure_ascii=False)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration with defaults/corrections to: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)
                 init_logger.warning("Proceeding with corrected config in memory, but file update failed.")

        # Update the global QUOTE_CURRENCY from the validated config
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.debug(f"Quote currency set to: {QUOTE_CURRENCY}")

        return updated_config # Return the validated and potentially corrected config

    except Exception as e:
        init_logger.error(f"{NEON_RED}FATAL: Unexpected error processing configuration: {e}. Using internal defaults.{RESET}", exc_info=True)
        QUOTE_CURRENCY = default_config.get("quote_currency", "USDT")
        return default_config # Fallback to defaults on unexpected error

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
# QUOTE_CURRENCY is updated inside load_config()

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes and validates the CCXT Bybit exchange object.

    Steps:
    1. Sets API keys, rate limiting, default type (linear), timeouts.
    2. Configures sandbox mode based on `config.json`.
    3. Loads exchange markets with retries, ensuring markets are actually populated.
    4. Performs an initial balance check for the configured `QUOTE_CURRENCY`.
       - If trading is enabled, a failed balance check is treated as a fatal error.
       - If trading is disabled, logs a warning but allows proceeding.

    Args:
        logger: The logger instance to use for status messages.

    Returns:
        The initialized ccxt.Exchange object if successful, otherwise None.
    """
    lg = logger # Alias for convenience
    try:
        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',         # Assume linear contracts by default
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                # Timeouts for various operations (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000,      # Longer timeout for potentially large kline fetches
            }
        }
        # Instantiate the Bybit exchange object
        exchange = ccxt.bybit(exchange_options)

        # Configure Sandbox Mode
        is_sandbox = CONFIG.get('use_sandbox', True)
        exchange.set_sandbox_mode(is_sandbox)
        if is_sandbox:
            lg.warning(f"{NEON_YELLOW}<<< USING SANDBOX MODE (Testnet Environment) >>>{RESET}")
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! <<< USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK >>> !!!{RESET}")

        # Load Markets with Retries
        lg.info(f"Attempting to load markets for {exchange.id}...")
        markets_loaded = False
        last_market_error = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # Force reload on retries to ensure fresh market data
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"{NEON_GREEN}Markets loaded successfully ({len(exchange.markets)} symbols found).{RESET}")
                    markets_loaded = True
                    break # Exit retry loop on success
                else:
                    last_market_error = "Market loading returned empty result"
                    lg.warning(f"Market loading returned empty result (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                last_market_error = e
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}.")
                if attempt >= MAX_API_RETRIES:
                    lg.critical(f"{NEON_RED}Maximum retries exceeded while loading markets due to network errors. Last error: {last_market_error}. Exiting.{RESET}")
                    return None
            except ccxt.AuthenticationError as e:
                 last_market_error = e
                 lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API Key/Secret/Permissions. Exiting.{RESET}")
                 return None
            except Exception as e:
                last_market_error = e
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

            # Apply delay before retrying
            if not markets_loaded and attempt < MAX_API_RETRIES:
                 delay = RETRY_DELAY_SECONDS * (attempt + 1) # Exponential backoff
                 lg.warning(f"Retrying market load in {delay} seconds...")
                 time.sleep(delay)

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load markets for {exchange.id} after all retries. Last error: {last_market_error}. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} | Sandbox: {is_sandbox}")

        # Initial Balance Check
        lg.info(f"Attempting initial balance fetch for quote currency ({QUOTE_CURRENCY})...")
        initial_balance: Optional[Decimal] = None
        try:
            initial_balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        except ccxt.AuthenticationError as auth_err:
            # Handle auth errors specifically here as they are critical during balance check
            lg.critical(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
            return None
        except Exception as balance_err:
             # Catch other potential errors during the initial balance check
             lg.warning(f"{NEON_YELLOW}Initial balance fetch encountered an error: {balance_err}.{RESET}", exc_info=True)
             # Let the logic below decide based on trading enabled status

        # Evaluate balance check result based on trading mode
        if initial_balance is not None:
            lg.info(f"{NEON_GREEN}Initial available balance: {initial_balance.normalize()} {QUOTE_CURRENCY}{RESET}")
            return exchange # Success!
        else:
            # Balance fetch failed (fetch_balance logs the failure reason)
            lg.error(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}.{RESET}")
            if CONFIG.get('enable_trading', False):
                lg.critical(f"{NEON_RED}Trading is enabled, but initial balance check failed. Cannot proceed safely. Exiting.{RESET}")
                return None
            else:
                lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without confirmed initial balance, but errors might occur later.{RESET}")
                return exchange # Allow proceeding in non-trading mode

    except Exception as e:
        # Catch-all for errors during the initialization process itself
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using `fetch_ticker`.

    Prioritizes 'last' price. Falls back progressively:
    1. Mid-price ((bid + ask) / 2) if both bid and ask are valid.
    2. 'ask' price if only ask is valid.
    3. 'bid' price if only bid is valid.

    Includes retry logic for network errors and rate limits.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        The current price as a Decimal, or None if fetching fails after retries
        or a non-retryable error occurs.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker data for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None

            # Helper to safely convert ticker values to Decimal
            def safe_decimal_from_ticker(value: Optional[Any], field_name: str) -> Optional[Decimal]:
                """Safely converts ticker field to positive Decimal."""
                if value is None: return None
                try:
                    s_val = str(value).strip()
                    if not s_val: return None
                    dec_val = Decimal(s_val)
                    return dec_val if dec_val > Decimal('0') else None
                except (ValueError, InvalidOperation, TypeError):
                    lg.debug(f"Could not parse ticker field '{field_name}' value '{value}' to Decimal.")
                    return None

            # 1. Try 'last' price
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')

            # 2. Fallback to mid-price if 'last' is invalid
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()}) -> {price.normalize()}")
                # 3. Fallback to 'ask' if only ask is valid
                elif ask:
                    price = ask
                    lg.debug(f"Using 'ask' price as fallback: {price.normalize()}")
                # 4. Fallback to 'bid' if only bid is valid
                elif bid:
                    price = bid
                    lg.debug(f"Using 'bid' price as fallback: {price.normalize()}")

            # Check if a valid price was obtained
            if price:
                lg.debug(f"Current price successfully fetched for {symbol}: {price.normalize()}")
                return price.normalize() # Ensure normalization
            else:
                last_exception = ValueError(f"No valid price ('last', 'mid', 'ask', 'bid') found in ticker data. Ticker: {ticker}")
                lg.warning(f"No valid price found in ticker (Attempt {attempts + 1}). Retrying...")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempts, just retry after waiting
            continue
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching price: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if needed
            # For now, assume potentially retryable unless it's an auth error
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Exit on unexpected errors

        # Increment attempt counter and apply delay (only if not a rate limit wait)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV (kline) data using CCXT's `fetch_ohlcv` method with enhancements.

    - Handles Bybit V5 'category' parameter automatically based on market info.
    - Implements robust retry logic for network errors and rate limits.
    - Validates fetched data timestamp lag to detect potential staleness.
    - Processes data into a Pandas DataFrame with Decimal types for precision.
    - Cleans data (drops rows with NaNs in key columns, zero prices/volumes).
    - Trims DataFrame to `MAX_DF_LEN` to manage memory usage.
    - Ensures DataFrame is sorted by timestamp.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        timeframe: The CCXT timeframe string (e.g., "5m", "1h", "1d").
        limit: The desired number of klines. Will be capped by `BYBIT_API_KLINE_LIMIT`
               per API request, but the function aims to fetch the `limit` specified
               (may require multiple requests in future versions, currently single request).
        logger: The logger instance for status messages.

    Returns:
        A Pandas DataFrame containing the OHLCV data, indexed by timestamp (UTC),
        with columns ['open', 'high', 'low', 'close', 'volume'] as Decimals.
        Returns an empty DataFrame if fetching or processing fails critically.
    """
    lg = logger
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV. Cannot fetch klines.")
        return pd.DataFrame()

    # Determine the actual number of klines to request in this single API call
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
        lg.warning(f"Requested limit ({limit}) exceeds API limit ({BYBIT_API_KLINE_LIMIT}). Will request {BYBIT_API_KLINE_LIMIT}. "
                   f"Strategy might require more data than fetched in this single request.")
    elif limit < 50: # Warn if requesting very few candles, might impact indicators
        lg.warning(f"Requesting a small number of klines ({limit}). Ensure this is sufficient for all indicator calculations.")


    ohlcv_data: Optional[List[List[Union[int, float, str]]]] = None
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching klines for {symbol} | TF: {timeframe} | Limit: {actual_request_limit} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            params = {}
            # Add Bybit V5 specific parameters
            if 'bybit' in exchange.id.lower():
                 try:
                     # Determine category (linear/inverse/spot) for Bybit V5 API
                     market = exchange.market(symbol) # Assumes markets are loaded
                     category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                     params['category'] = category
                     lg.debug(f"Using Bybit category: {category} for kline fetch.")
                 except KeyError:
                     lg.warning(f"Market '{symbol}' not found in loaded markets during kline fetch. Cannot set Bybit category.")
                 except Exception as e:
                     lg.warning(f"Could not automatically determine market category for {symbol} kline fetch: {e}. Proceeding without category param.")

            # Fetch OHLCV data
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit, params=params)
            fetched_count = len(ohlcv_data) if ohlcv_data else 0
            lg.debug(f"API returned {fetched_count} candles (requested {actual_request_limit}).")

            # --- Basic Validation on Fetched Data ---
            if ohlcv_data and fetched_count > 0:
                # Validate timestamp lag of the last candle
                try:
                    last_candle_timestamp_ms = ohlcv_data[-1][0]
                    last_ts = pd.to_datetime(last_candle_timestamp_ms, unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    # Get timeframe duration in seconds
                    interval_seconds = exchange.parse_timeframe(timeframe) # CCXT provides this helper
                    # Allow a lag of up to 2 intervals (generous)
                    max_allowed_lag_seconds = interval_seconds * 2.5 # Allow 2.5 intervals lag
                    actual_lag_seconds = (now_utc - last_ts).total_seconds()

                    if actual_lag_seconds < 0: # Clock skew issue?
                         lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_ts} is in the future? (Lag: {actual_lag_seconds:.1f}s). Check system clock/timezone. Proceeding cautiously.{RESET}")
                         break # Proceed but log warning
                    elif actual_lag_seconds <= max_allowed_lag_seconds:
                        lg.debug(f"Last kline timestamp {last_ts} seems recent (Lag: {actual_lag_seconds:.1f}s <= Max Allowed: {max_allowed_lag_seconds:.1f}s). Data OK.")
                        break # Successful fetch and basic validation passed, exit retry loop
                    else:
                        last_exception = ValueError(f"Kline data potentially stale. Last candle lag ({actual_lag_seconds:.1f}s) > max allowed ({max_allowed_lag_seconds:.1f}s).")
                        lg.warning(f"{NEON_YELLOW}Timestamp lag detected: {last_exception}. Retrying fetch...{RESET}")
                        ohlcv_data = None # Discard potentially stale data and retry

                except IndexError:
                    last_exception = IndexError("Received empty OHLCV data list, cannot validate timestamp.")
                    lg.warning(f"{last_exception}. Retrying fetch...")
                    ohlcv_data = None
                except Exception as ts_err:
                    last_exception = ts_err
                    lg.warning(f"Could not validate kline timestamp lag: {ts_err}. Proceeding with fetched data, but be cautious.")
                    break # Proceed even if validation fails, but log warning
            else:
                last_exception = ValueError(f"API returned no kline data (fetched_count={fetched_count}).")
                lg.warning(f"{last_exception} (Attempt {attempts + 1}). Retrying...")
                ohlcv_data = None # Ensure ohlcv_data is None if fetch failed

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3 # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts for rate limit waits
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching klines: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return pd.DataFrame() # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
            # Check for specific non-retryable errors
            if "invalid timeframe" in str(e).lower() or "Interval is not supported" in str(e):
                lg.critical(f"{NEON_RED}Invalid timeframe '{timeframe}' specified for {exchange.id}. Exiting fetch.{RESET}")
                return pd.DataFrame()
            # Otherwise, assume retryable for now
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
            return pd.DataFrame() # Stop on unexpected errors

        # Increment attempt counter and apply delay (only if fetch failed and retries remain)
        attempts += 1
        if attempts <= MAX_API_RETRIES and ohlcv_data is None:
             time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # After retry loop, check if data was successfully fetched
    if not ohlcv_data:
        lg.error(f"{NEON_RED}Failed to fetch kline data for {symbol} {timeframe} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
        return pd.DataFrame()

    # --- Process Fetched Data into DataFrame ---
    try:
        lg.debug(f"Processing {len(ohlcv_data)} fetched candles into DataFrame...")
        # Define standard OHLCV column names
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Create DataFrame, ensure columns match the fetched data structure
        df = pd.DataFrame(ohlcv_data, columns=cols[:len(ohlcv_data[0])])

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        if df.empty:
            lg.error("DataFrame became empty after timestamp conversion/dropna.")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential errors robustly
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Apply pd.to_numeric first, coercing errors to NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # Convert valid finite numbers to Decimal, others become Decimal('NaN')
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                )
            else:
                 lg.warning(f"Expected column '{col}' not found in fetched kline data.")

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential price columns or non-positive close price
        essential_price_cols = ['open', 'high', 'low', 'close']
        df.dropna(subset=essential_price_cols, inplace=True)
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN volume or negative volume (if volume column exists)
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            df = df[df['volume'] >= Decimal('0')] # Allow zero volume

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows during cleaning (NaNs, zero/neg prices, neg volume).")

        if df.empty:
            lg.warning(f"Kline DataFrame is empty after cleaning for {symbol} {timeframe}.")
            return pd.DataFrame()

        # Ensure DataFrame is sorted by timestamp index (should be from fetch_ohlcv, but verify)
        if not df.index.is_monotonic_increasing:
            lg.warning("Kline DataFrame index was not monotonic, sorting...")
            df.sort_index(inplace=True)

        # --- Memory Management ---
        # Trim DataFrame if it exceeds the maximum allowed length
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds maximum ({MAX_DF_LEN}). Trimming to most recent {MAX_DF_LEN} candles.")
            df = df.iloc[-MAX_DF_LEN:].copy() # Keep the latest N rows

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()

# Helper function to safely convert market info values to Decimal
def _safe_market_decimal(value: Optional[Any], field_name: str, allow_zero: bool = True) -> Optional[Decimal]:
    """Converts a market info value to Decimal, handling None and invalid formats."""
    if value is None:
        return None
    try:
        d_val = Decimal(str(value))
        if not allow_zero and d_val <= Decimal('0'):
             #init_logger.debug(f"Market info field '{field_name}' has non-positive value '{d_val}' (zero not allowed), returning None.")
             return None
        if allow_zero and d_val < Decimal('0'):
             #init_logger.debug(f"Market info field '{field_name}' has negative value '{d_val}' (zero allowed), returning None.")
             return None
        return d_val
    except (InvalidOperation, TypeError, ValueError):
        #init_logger.debug(f"Could not convert market info field '{field_name}' value '{value}' to Decimal.")
        return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[MarketInfo]:
    """
    Retrieves, validates, and standardizes market information for a symbol.

    - Reloads markets if the symbol is initially not found.
    - Extracts precision (price, amount), limits (min/max amount, cost),
      contract type (linear/inverse/spot), and contract size.
    - Adds convenience flags (`is_contract`, `is_linear`, etc.) and parsed
      Decimal values for precision/limits to the returned dictionary.
    - Includes retry logic for network errors.
    - Logs critical warnings and returns None if essential precision data is missing.

    Args:
        exchange: The initialized ccxt.Exchange object (must have markets loaded).
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        A MarketInfo TypedDict containing standardized market details, including
        parsed Decimal values for limits/precision, or None if the market is not found,
        essential data is missing, or a critical error occurs.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            market: Optional[Dict] = None
            # Check if markets are loaded and the symbol exists
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market info for '{symbol}' not found in loaded markets. Attempting to reload markets...")
                try:
                    exchange.load_markets(reload=True) # Force reload
                    lg.info("Markets reloaded.")
                except Exception as reload_err:
                     last_exception = reload_err
                     lg.error(f"Failed to reload markets while searching for {symbol}: {reload_err}")
                     # Continue to the retry logic below

            # Try fetching the market dictionary again after potential reload
            try:
                market = exchange.market(symbol)
            except ccxt.BadSymbol:
                 market = None # Handled below
            except Exception as fetch_err:
                 last_exception = fetch_err
                 lg.warning(f"Error fetching market dict for '{symbol}' after reload: {fetch_err}. Retrying...")
                 market = None

            if market is None:
                # Symbol not found or error fetching market dict
                if attempts < MAX_API_RETRIES:
                    lg.warning(f"Symbol '{symbol}' not found or market fetch failed (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}). Retrying check...")
                    # Fall through to retry delay
                else:
                    lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after reload and retries. Last error: {last_exception}{RESET}")
                    return None # Symbol definitively not found or fetch failed
            else:
                # --- Market Found - Extract and Standardize Details ---
                lg.debug(f"Market found for '{symbol}'. Standardizing details...")
                std_market = market.copy() # Work on a copy

                # Add custom flags for easier logic later
                std_market['is_contract'] = std_market.get('contract', False) or std_market.get('type') in ['swap', 'future']
                std_market['is_linear'] = std_market.get('linear', False) and std_market['is_contract']
                std_market['is_inverse'] = std_market.get('inverse', False) and std_market['is_contract']
                std_market['contract_type_str'] = "Linear" if std_market['is_linear'] else \
                                                  "Inverse" if std_market['is_inverse'] else \
                                                  "Spot" if std_market.get('spot') else "Unknown"

                # Safely parse precision and limits into Decimal
                precision = std_market.get('precision', {})
                limits = std_market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})
                price_limits = limits.get('price', {})

                # Parse precision steps (must be positive)
                std_market['amount_precision_step_decimal'] = _safe_market_decimal(precision.get('amount'), 'precision.amount', allow_zero=False)
                std_market['price_precision_step_decimal'] = _safe_market_decimal(precision.get('price'), 'precision.price', allow_zero=False)

                # Parse limits (allow zero for min, must be positive if set)
                std_market['min_amount_decimal'] = _safe_market_decimal(amount_limits.get('min'), 'limits.amount.min')
                std_market['max_amount_decimal'] = _safe_market_decimal(amount_limits.get('max'), 'limits.amount.max', allow_zero=False)
                std_market['min_cost_decimal'] = _safe_market_decimal(cost_limits.get('min'), 'limits.cost.min')
                std_market['max_cost_decimal'] = _safe_market_decimal(cost_limits.get('max'), 'limits.cost.max', allow_zero=False)

                # Parse contract size (must be positive, default to 1)
                contract_size_val = std_market.get('contractSize', '1')
                std_market['contract_size_decimal'] = _safe_market_decimal(contract_size_val, 'contractSize', allow_zero=False) or Decimal('1')

                # --- Critical Validation: Essential Precision ---
                if std_market['amount_precision_step_decimal'] is None or std_market['price_precision_step_decimal'] is None:
                    lg.error(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' is missing essential positive precision data.")
                    lg.error(f"  Parsed Amount Precision: {std_market['amount_precision_step_decimal']}")
                    lg.error(f"  Parsed Price Precision: {std_market['price_precision_step_decimal']}")
                    lg.error(f"  Raw Precision Dict: {precision}")
                    lg.error("Trading calculations require valid positive amount and price precision steps. Cannot proceed safely.")
                    return None # Returning None is safer for the bot flow

                # Log extracted details for verification
                log_msg = (
                    f"Market Info ({symbol}): ID={std_market.get('id', 'N/A')}, Type={std_market.get('type', 'N/A')}, "
                    f"Contract Type={std_market['contract_type_str']}, Active={std_market.get('active', 'N/A')}\n"
                    f"  Precision Steps (Amount/Price): {std_market['amount_precision_step_decimal'].normalize()} / {std_market['price_precision_step_decimal'].normalize()}\n"
                    f"  Limits (Amount Min/Max): {std_market['min_amount_decimal'].normalize() if std_market['min_amount_decimal'] is not None else 'N/A'} / {std_market['max_amount_decimal'].normalize() if std_market['max_amount_decimal'] is not None else 'N/A'}\n"
                    f"  Limits (Cost Min/Max): {std_market['min_cost_decimal'].normalize() if std_market['min_cost_decimal'] is not None else 'N/A'} / {std_market['max_cost_decimal'].normalize() if std_market['max_cost_decimal'] is not None else 'N/A'}\n"
                    f"  Contract Size: {std_market['contract_size_decimal'].normalize()}"
                )
                lg.debug(log_msg)

                # Cast to MarketInfo TypedDict before returning
                try:
                    # Select only keys defined in MarketInfo for type safety
                    # Note: This requires manually listing keys or more complex introspection
                    # Simpler approach: Trust the structure matches and cast
                    final_market_info: MarketInfo = std_market # type: ignore
                    return final_market_info
                except Exception as cast_err:
                     lg.error(f"Error casting market dictionary to MarketInfo TypedDict: {cast_err}")
                     # Fallback: Return the dictionary anyway if casting fails but data seems okay
                     return std_market # type: ignore

        # --- Error Handling with Retries ---
        except ccxt.BadSymbol as e:
            # Symbol is definitively invalid according to the exchange
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error retrieving market info for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded retrieving market info for {symbol} due to network errors. Last error: {last_exception}{RESET}")
                return None
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error getting market info: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
             return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error retrieving market info for {symbol}: {e}{RESET}")
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded retrieving market info for {symbol} due to exchange errors. Last error: {last_exception}{RESET}")
                 return None
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error retrieving market info for {symbol}: {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # Should only be reached if all retries fail due to network/exchange errors
    lg.error(f"{NEON_RED}Failed to retrieve market info for {symbol} after all attempts. Last error: {last_exception}{RESET}")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available trading balance for a specific currency (e.g., USDT).

    - Handles Bybit V5 account types (UNIFIED, CONTRACT) to find the correct balance.
    - Parses various potential balance fields ('free', 'availableToWithdraw', 'availableBalance').
    - Includes retry logic for network errors and rate limits.
    - Handles authentication errors critically.

    Args:
        exchange: The initialized ccxt.Exchange object.
        currency: The currency code to fetch the balance for (e.g., "USDT").
        logger: The logger instance for status messages.

    Returns:
        The available balance as a Decimal (non-negative), or None if fetching fails
        after retries or a critical error occurs.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: Optional[str] = None
            found: bool = False
            balance_info: Optional[Dict] = None

            # Bybit V5 often requires specifying account type (Unified or Contract)
            # Check specific types first, then fallback to default request
            account_types_to_check = []
            if 'bybit' in exchange.id.lower():
                 account_types_to_check = ['UNIFIED', 'CONTRACT']
            account_types_to_check.append('') # Always check default/unspecified type

            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    type_desc = f"Account Type: {acc_type}" if acc_type else "Default Account Type"
                    lg.debug(f"Fetching balance for {currency} ({type_desc}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try different ways to extract balance ---
                    # 1. Standard CCXT structure ('free' field)
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        lg.debug(f"Found balance for {currency} in standard 'free' field: {balance_str}")
                        found = True; break

                    # 2. Bybit V5 structure (often nested in 'info') - Check specific fields
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        for account_details in balance_info['info']['result']['list']:
                             # Check if accountType matches the one requested (or if no type was requested)
                            if (not acc_type or account_details.get('accountType') == acc_type) and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Prioritize 'availableToWithdraw', then 'availableBalance', then 'walletBalance'
                                        balance_val = coin_data.get('availableToWithdraw')
                                        source_field = 'availableToWithdraw'
                                        if balance_val is None:
                                             balance_val = coin_data.get('availableBalance')
                                             source_field = 'availableBalance'
                                        if balance_val is None:
                                             balance_val = coin_data.get('walletBalance') # Last resort
                                             source_field = 'walletBalance'

                                        if balance_val is not None:
                                            balance_str = str(balance_val)
                                            lg.debug(f"Found balance for {currency} in Bybit V5 structure (Account: {account_details.get('accountType')}, Field: {source_field}): {balance_str}")
                                            found = True; break # Found in coin list
                                if found: break # Found in account details list
                        if found: break # Found across account types

                except ccxt.ExchangeError as e:
                    # Errors like "account type does not exist" are expected when checking multiple types
                    if acc_type: # Only log as debug if checking a specific type
                        lg.debug(f"Minor exchange error fetching balance for type '{acc_type}': {e}. Trying next type...")
                    else: # Raise error if default fetch fails
                        raise e
                    continue # Try the next account type
                except Exception as e:
                    # Catch other unexpected errors during a specific account type check
                    lg.warning(f"Unexpected error fetching balance for type '{acc_type or 'Default'}': {e}. Trying next step...")
                    last_exception = e # Store for potential final error message
                    continue # Try the next account type

            # --- Process the result ---
            if found and balance_str is not None:
                try:
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative
                    final_balance = max(balance_decimal, Decimal('0'))
                    lg.debug(f"Successfully parsed balance for {currency}: {final_balance.normalize()}")
                    return final_balance # Success
                except (ValueError, InvalidOperation, TypeError) as e:
                    # Raise an error if the found balance string cannot be converted
                    raise ccxt.ExchangeError(f"Failed to convert fetched balance string '{balance_str}' for {currency} to Decimal: {e}")
            elif not found:
                # If not found after checking all types
                raise ccxt.ExchangeError(f"Could not find balance information for currency '{currency}' in any checked account type or response structure. Last response: {balance_info}")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance for {currency}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
            # Raise the exception to be caught by the caller (initialize_exchange) for critical handling
            raise e
        except ccxt.ExchangeError as e:
            last_exception = e
            # Log exchange errors (like currency not found, conversion errors) and retry
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for {currency}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance for {currency}: {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[PositionInfo]:
    """
    Checks for an existing open position for the given symbol using `fetch_positions`.

    - Handles Bybit V5 specifics (category, symbol filtering, parsing `info` field).
    - Determines position side ('long'/'short') and size accurately using Decimal.
    - Parses key position details (entry price, leverage, SL/TP, TSL) into a standardized format.
    - Includes retry logic for network errors and rate limits.
    - Returns a standardized `PositionInfo` dictionary if an active position is found,
      otherwise returns None. Returns None on failure.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        A PositionInfo TypedDict containing details of the open position if found,
        otherwise None.
    """
    lg = logger
    attempts = 0
    last_exception = None
    market_id: Optional[str] = None
    category: Optional[str] = None

    # --- Determine Market ID and Category ---
    try:
        market = exchange.market(symbol)
        market_id = market['id']
        category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
        lg.debug(f"Using Market ID: {market_id}, Category: {category} for position check.")
    except KeyError:
        lg.error(f"Market '{symbol}' not found in loaded markets. Cannot check position.")
        return None
    except Exception as e:
        lg.error(f"Error determining market details for position check: {e}")
        return None

    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            positions: List[Dict] = []

            # --- Fetch Positions (Handling Bybit V5 Specifics) ---
            try:
                # Attempt to fetch positions for the specific symbol and category
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with specific params: {params}")
                # Request only specific symbol if possible, otherwise fetch all and filter
                if exchange.has.get('fetchPositionsForSymbol'):
                     positions = exchange.fetch_positions([symbol], params=params)
                elif exchange.has.get('fetchPositions'):
                     all_positions = exchange.fetch_positions(params=params)
                     # Filter the results manually by symbol or market ID
                     positions = [
                         p for p in all_positions
                         if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id
                     ]
                     lg.debug(f"Fetched {len(all_positions)} total positions, filtered down to {len(positions)} for {symbol}.")
                else:
                     raise ccxt.NotSupported("Exchange does not support fetchPositions or fetchPositionsForSymbol.")

            except ccxt.ExchangeError as e:
                 # Bybit often returns specific codes for "no position" or related issues
                 no_pos_codes = [110025] # e.g., "position idx not match position mode" can indicate no pos in one-way
                 no_pos_messages = ["position not found", "no position", "position does not exist"]
                 err_str = str(e).lower()
                 code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', '')) # Check different code attributes
                 code_match = any(str(c) in code_str for c in no_pos_codes) if code_str else False

                 if code_match or any(msg in err_str for msg in no_pos_messages):
                     lg.info(f"No open position found for {symbol} (Exchange message: {e}).")
                     return None # No position exists
                 else:
                     # Re-raise other exchange errors to be handled by the main retry loop
                     raise e

            # --- Process Fetched Positions ---
            active_position_raw: Optional[Dict] = None
            # Define a small threshold based on amount precision to consider a position "open"
            size_threshold = Decimal('1e-9') # Default small value
            try:
                amount_precision_str = exchange.market(symbol)['precision']['amount']
                if amount_precision_str:
                    # Use a fraction of the minimum amount step as the threshold
                    size_threshold = Decimal(str(amount_precision_str)) * Decimal('0.01') # 1% of step size
            except Exception as prec_err:
                lg.warning(f"Could not get amount precision for {symbol} to set size threshold: {prec_err}. Using default: {size_threshold}")
            lg.debug(f"Using position size threshold (absolute): {size_threshold.normalize()}")

            # Iterate through the filtered positions to find an active one
            for pos in positions:
                # Extract size from 'info' (exchange-specific) or standard 'contracts' field
                # Bybit V5 uses 'size' in info, CCXT might map to 'contracts'
                size_str_info = str(pos.get('info', {}).get('size', '')).strip()
                size_str_std = str(pos.get('contracts', '')).strip()
                size_str = size_str_info if size_str_info else size_str_std # Prioritize info['size']

                if not size_str:
                    lg.debug(f"Skipping position entry with missing size data: {pos.get('info', {})}")
                    continue

                try:
                    # Convert size to Decimal and check against threshold (absolute value)
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold:
                        # Found an active position with significant size
                        active_position_raw = pos
                        active_position_raw['size_decimal'] = size_decimal # Store the parsed Decimal size
                        lg.debug(f"Found active position entry: Size={size_decimal.normalize()}")
                        break # Stop searching once an active position is found
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     # Log error if size string cannot be parsed, skip this entry
                     lg.warning(f"Could not parse/check position size string '{size_str}': {parse_err}. Skipping this position entry.")
                     continue # Move to the next position entry

            # --- Format and Return Active Position ---
            if active_position_raw:
                # Standardize the position dictionary using PositionInfo structure
                std_pos = active_position_raw.copy() # Work on a copy
                info = std_pos.get('info', {}) # Exchange-specific details

                # Determine Side (long/short) reliably
                side = std_pos.get('side') # Standard CCXT field
                size = std_pos['size_decimal'] # Use the parsed Decimal size

                if side not in ['long', 'short']:
                    # Fallback using Bybit V5 'side' field ('Buy'/'Sell') or inferred from size
                    side_v5 = str(info.get('side', '')).lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif size > size_threshold: side = 'long' # Infer from positive size
                    elif size < -size_threshold: side = 'short' # Infer from negative size
                    else: side = None # Cannot determine side

                if not side:
                    lg.error(f"Could not determine side for active position {symbol}. Size: {size}. Data: {info}")
                    return None # Cannot proceed without side
                std_pos['side'] = side # Update the standardized dict

                # Standardize other key fields (prefer standard CCXT, fallback to info)
                std_pos['entryPrice'] = std_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice')
                std_pos['leverage'] = std_pos.get('leverage') or info.get('leverage')
                std_pos['liquidationPrice'] = std_pos.get('liquidationPrice') or info.get('liqPrice')
                std_pos['unrealizedPnl'] = std_pos.get('unrealizedPnl') or info.get('unrealisedPnl') or info.get('unrealizedPnl')

                # Parse protection levels from 'info' (Bybit V5 specific fields)
                # Ensure they are non-empty strings and represent non-zero values before storing
                def get_protection_field(field_name: str) -> Optional[str]:
                    value = info.get(field_name)
                    if value is None: return None
                    s_value = str(value).strip()
                    try:
                         # Check if it's a valid number and not zero
                         if s_value and Decimal(s_value) != Decimal('0'):
                             return s_value
                    except (InvalidOperation, ValueError, TypeError):
                         return None # Ignore if not a valid non-zero number string
                    return None

                std_pos['stopLossPrice'] = get_protection_field('stopLoss')
                std_pos['takeProfitPrice'] = get_protection_field('takeProfit')
                std_pos['trailingStopLoss'] = get_protection_field('trailingStop') # TSL distance
                std_pos['tslActivationPrice'] = get_protection_field('activePrice') # TSL activation price

                # Helper for formatting Decimal values for logging
                def format_decimal(value: Optional[Any]) -> str:
                    if value is None: return 'N/A'
                    try: return str(Decimal(str(value)).normalize())
                    except (InvalidOperation, TypeError, ValueError): return 'Invalid'

                # Log summary of the found position
                ep_str = format_decimal(std_pos.get('entryPrice'))
                size_str = std_pos['size_decimal'].normalize()
                sl_str = format_decimal(std_pos.get('stopLossPrice'))
                tp_str = format_decimal(std_pos.get('takeProfitPrice'))
                tsl_dist_str = format_decimal(std_pos.get('trailingStopLoss'))
                tsl_act_str = format_decimal(std_pos.get('tslActivationPrice'))
                tsl_log = f"({tsl_dist_str}/{tsl_act_str})" if tsl_dist_str != 'N/A' or tsl_act_str != 'N/A' else "(N/A)"
                pnl_str = format_decimal(std_pos.get('unrealizedPnl'))
                liq_str = format_decimal(std_pos.get('liquidationPrice'))

                lg.info(f"{NEON_GREEN}{BRIGHT}Active {side.upper()} Position Found ({symbol}):{RESET} "
                        f"Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, "
                        f"SL={sl_str}, TP={tp_str}, TSL={tsl_log}")

                # Cast to PositionInfo TypedDict before returning
                try:
                    final_position_info: PositionInfo = std_pos # type: ignore
                    return final_position_info
                except Exception as cast_err:
                     lg.error(f"Error casting position dictionary to PositionInfo TypedDict: {cast_err}")
                     return std_pos # type: ignore # Return raw dict if cast fails
            else:
                # No position with size > threshold was found after filtering
                lg.info(f"No active position found for {symbol} (checked {len(positions)} filtered entries).")
                return None

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching positions for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error fetching positions: {e}. Check API Key/Secret/Permissions. Stopping fetch.{RESET}")
            return None # Fatal error for this operation
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Could add checks for specific non-retryable errors here
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: MarketInfo, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a derivatives symbol using `set_leverage`.

    - Skips if the market is not a contract (spot) or leverage is invalid (<= 0).
    - Handles Bybit V5 specific parameters (category, buy/sell leverage).
    - Includes retry logic for network/exchange errors.
    - Checks for specific Bybit codes indicating success (`0`) or leverage already set (`110045`).
    - Identifies and handles known non-retryable leverage errors.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        leverage: The desired integer leverage level.
        market_info: The standardized MarketInfo dictionary.
        logger: The logger instance for status messages.

    Returns:
        True if leverage was set successfully or was already set correctly, False otherwise.
    """
    lg = logger
    # Validate input and market type
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True # Consider success as no action needed for spot
    if not isinstance(leverage, int) or leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be a positive integer.")
        return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage method.")
        return False

    market_id = market_info['id'] # Use the exchange-specific market ID

    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {market_id} to {leverage}x (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            params = {}
            # --- Bybit V5 Specific Parameters ---
            if 'bybit' in exchange.id.lower():
                 # Determine category and set buy/sell leverage explicitly for Bybit V5
                 category = market_info.get('contract_type_str', 'Linear').lower() # 'linear' or 'inverse'
                 params = {
                     'category': category,
                     #'symbol': market_id, # Symbol passed separately to set_leverage
                     'buyLeverage': str(leverage), # Bybit expects strings
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 setLeverage params: {params}")

            # --- Execute set_leverage call ---
            # Pass market_id as the symbol argument
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"set_leverage raw response: {response}")

            # --- Check Response (Bybit V5 specific codes) ---
            # CCXT might parse some responses, but Bybit often returns structured info
            ret_code_str = None
            ret_msg = "N/A"
            if isinstance(response, dict):
                 ret_code_str = str(response.get('retCode', response.get('code'))) # Check both common keys
                 ret_msg = response.get('retMsg', response.get('message', 'Unknown Bybit API message'))

            if ret_code_str == '0':
                 lg.info(f"{NEON_GREEN}Leverage successfully set for {market_id} to {leverage}x (Bybit Code: 0).{RESET}")
                 return True
            elif ret_code_str == '110045': # "Leverage not modified"
                 lg.info(f"{NEON_YELLOW}Leverage for {market_id} is already {leverage}x (Bybit Code: 110045).{RESET}")
                 return True
            elif ret_code_str is not None and ret_code_str != 'None': # Check if a non-zero code was returned
                 # Raise an error for other non-zero Bybit return codes
                 raise ccxt.ExchangeError(f"Bybit API error setting leverage: {ret_msg} (Code: {ret_code_str})")
            else:
                # Assume success if no specific error code structure is found and no exception was raised by CCXT
                lg.info(f"{NEON_GREEN}Leverage set/confirmed for {market_id} to {leverage}x (No specific Bybit error code found in response).{RESET}")
                return True

        # --- Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', '')) # Check different code attributes
            err_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage for {market_id}: {e} (Code: {err_code_str}){RESET}")

            # Check for non-retryable conditions based on code or message
            if err_code_str == '110045' or "leverage not modified" in err_str:
                lg.info(f"{NEON_YELLOW}Leverage already set (detected via error). Treating as success.{RESET}")
                return True # Already set is considered success

            # List of known fatal Bybit error codes for leverage setting
            # Based on Bybit API docs and common issues
            fatal_codes = [
                '10001', # Parameter error
                '10004', # Sign check error
                '110009',# Margin mode cannot be modified when position exists
                '110013',# Parameter error (generic)
                '110028',# Leverage cannot be greater/less than risk limit leverage
                '110043',# Position status is not normal (e.g., closing)
                '110044',# Leverage cannot be modified when position exists (cross margin)
                '110055',# Risk limit cannot be modified when position exists
                '3400045',# Set margin mode failed
            ]
            # Common fatal message fragments
            fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient balance", "invalid leverage"]

            if err_code_str in fatal_codes or any(msg in err_str for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE leverage error. Aborting leverage set.{RESET}")
                return False # Fatal error

            # If error is potentially retryable and retries remain
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError setting leverage. Last error: {last_exception}{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage for {market_id}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError setting leverage. Last error: {last_exception}{RESET}")
                return False

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error setting leverage: {e}. Check API Key/Secret/Permissions. Stopping leverage set.{RESET}")
             return False # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage for {market_id}: {e}{RESET}", exc_info=True)
            return False # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set leverage for {market_id} to {leverage}x after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: MarketInfo, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on risk parameters and market constraints.

    Uses Decimal for all financial calculations to ensure precision. Handles both
    linear and inverse contracts. Applies market precision and limits (amount, cost)
    to the calculated size.

    Args:
        balance: Available trading balance (in quote currency, e.g., USDT). Must be positive.
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%). Must be > 0 and <= 1.
        initial_stop_loss_price: The calculated initial stop loss price. Must be positive and different from entry.
        entry_price: The intended entry price (or current price). Must be positive.
        market_info: The standardized MarketInfo dictionary containing precision, limits, contract type, etc.
        exchange: The initialized ccxt.Exchange object (used for precision formatting).
        logger: The logger instance for status messages.

    Returns:
        The calculated position size as a Decimal, adjusted for market rules (positive value),
        or None if sizing is not possible (e.g., invalid inputs, insufficient balance for min size,
        cannot meet limits, calculation errors).
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE')
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base_currency # Unit of the position size

    lg.info(f"{BRIGHT}--- Position Sizing Calculation ({symbol}) ---{RESET}")

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Sizing failed: Invalid balance ({balance.normalize()} {quote_currency}). Must be positive.")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
            raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed: Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed: Entry price ({entry_price.normalize()}) and Stop Loss price ({initial_stop_loss_price.normalize()}) must be positive.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed: Stop Loss price ({initial_stop_loss_price.normalize()}) cannot be equal to Entry price ({entry_price.normalize()}).")
        return None

    # --- Extract Market Constraints (using pre-parsed Decimal values from MarketInfo) ---
    try:
        amount_precision_step = market_info['amount_precision_step_decimal']
        price_precision_step = market_info['price_precision_step_decimal']
        min_amount = market_info['min_amount_decimal']
        max_amount = market_info['max_amount_decimal']
        min_cost = market_info['min_cost_decimal']
        max_cost = market_info['max_cost_decimal']
        contract_size = market_info['contract_size_decimal']

        # Check if essential constraints are valid
        if amount_precision_step is None or amount_precision_step <= 0:
             raise ValueError(f"Invalid or missing amount precision step: {amount_precision_step}")
        if price_precision_step is None or price_precision_step <= 0:
             raise ValueError(f"Invalid or missing price precision step: {price_precision_step}")
        if contract_size <= Decimal('0'):
             raise ValueError(f"Invalid contract size: {contract_size}")

        # Set defaults for optional limits if they are None
        min_amount = min_amount if min_amount is not None else Decimal('0')
        max_amount = max_amount if max_amount is not None else Decimal('inf')
        min_cost = min_cost if min_cost is not None else Decimal('0')
        max_cost = max_cost if max_cost is not None else Decimal('inf')

        lg.debug(f"  Market Constraints: Amount Step={amount_precision_step.normalize()}, Min/Max Amount={min_amount.normalize()}/{max_amount.normalize()}, "
                 f"Min/Max Cost={min_cost.normalize()}/{max_cost.normalize()}, Contract Size={contract_size.normalize()}")

    except (KeyError, ValueError, TypeError) as e:
        lg.error(f"Sizing failed: Error extracting or validating market details for {symbol}: {e}")
        lg.debug(f"  MarketInfo received: {market_info}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    risk_amount_quote = (balance * risk_decimal).quantize(Decimal('1e-8'), ROUND_DOWN) # Quantize risk amount
    stop_loss_distance_price = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance_price <= Decimal('0'):
        # This should be caught earlier, but double-check
        lg.error(f"Sizing failed: Stop Loss distance is zero or negative ({stop_loss_distance_price.normalize()}).")
        return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"  Risk Per Trade: {risk_decimal:.2%}")
    lg.info(f"  Risk Amount: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"  Entry Price: {entry_price.normalize()}")
    lg.info(f"  Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"  Stop Loss Distance (Price): {stop_loss_distance_price.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}")

    # --- Calculate Initial Position Size (based on risk) ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear Contracts or Spot
            # Risk per unit = Price distance * Contract Size (value change per contract/base unit per $1 price move)
            value_change_per_unit = stop_loss_distance_price * contract_size
            if value_change_per_unit <= Decimal('1e-18'): # Use tolerance for zero check
                 lg.error("Sizing failed (Linear/Spot): Calculated value change per unit is effectively zero.")
                 return None
            # Size = Total Risk Amount / Value Change Per Unit
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calc: RiskAmt={risk_amount_quote} / ValueChangePerUnit={value_change_per_unit} = {calculated_size}")

        else: # Inverse Contracts
            # Risk per contract = Contract Size * |(1 / Entry) - (1 / SL)| (change in value of 1 contract in quote currency)
            if entry_price <= 0 or initial_stop_loss_price <= 0:
                 lg.error("Sizing failed (Inverse): Entry or SL price is zero or negative.")
                 return None
            # Use Decimal for inverse calculation
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'): # Tolerance for zero check
                 lg.error("Sizing failed (Inverse): Calculated inverse factor is effectively zero.")
                 return None
            risk_per_contract_quote = contract_size * inverse_factor
            if risk_per_contract_quote <= Decimal('1e-18'): # Tolerance for zero check
                 lg.error("Sizing failed (Inverse): Calculated risk per contract is effectively zero.")
                 return None
            # Size = Total Risk Amount / Risk per Contract
            calculated_size = risk_amount_quote / risk_per_contract_quote
            lg.debug(f"  Inverse Calc: RiskAmt={risk_amount_quote} / RiskPerContract={risk_per_contract_quote} = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err:
        lg.error(f"Sizing failed: Error during initial size calculation: {calc_err}.")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed: Initial calculated size is zero or negative ({calculated_size.normalize()}). Check inputs/risk/market data.")
        return None

    lg.info(f"  Initial Calculated Size = {calculated_size.normalize()} {size_unit}")

    # --- Apply Market Limits and Precision ---
    adjusted_size = calculated_size

    # Helper function to estimate cost
    def estimate_cost(size: Decimal, price: Decimal) -> Optional[Decimal]:
        if price <= 0 or size <= 0: return None
        try:
             # Cost = Size * Price * ContractSize (Linear) or Size * ContractSize / Price (Inverse)
             if not is_inverse:
                 return (size * price * contract_size)
             else:
                 return (size * contract_size) / price
        except (InvalidOperation, OverflowError, ZeroDivisionError):
            return None

    # 1. Apply Amount Limits (Min/Max Size)
    if min_amount > 0 and adjusted_size < min_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} is below minimum amount {min_amount.normalize()}. Adjusting UP to minimum.{RESET}")
        adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} exceeds maximum amount {max_amount.normalize()}. Adjusting DOWN to maximum.{RESET}")
        adjusted_size = max_amount
    lg.debug(f"  Size after Amount Limits: {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Cost Limits (Min/Max Order Value)
    estimated_cost_after_amount_limits = estimate_cost(adjusted_size, entry_price)
    cost_adjustment_applied = False

    if estimated_cost_after_amount_limits is not None:
        lg.debug(f"  Estimated Cost (after amount limits): {estimated_cost_after_amount_limits.normalize()} {quote_currency}")

        # Check Minimum Cost
        if min_cost > 0 and estimated_cost_after_amount_limits < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost_after_amount_limits.normalize()} is below minimum cost {min_cost.normalize()}. Attempting to increase size.{RESET}")
            required_size_for_min_cost = None
            try:
                # Calculate size needed to meet min cost: MinCost / (Price * ContrSize) [Lin] or MinCost * Price / ContrSize [Inv]
                if not is_inverse:
                    denominator = entry_price * contract_size
                    if denominator <= 0: raise ZeroDivisionError("Linear cost calc denominator zero/negative")
                    required_size_for_min_cost = min_cost / denominator
                else:
                    if entry_price <= 0 or contract_size <= 0: raise ValueError("Invalid price/contract size for inverse cost calc")
                    required_size_for_min_cost = (min_cost * entry_price) / contract_size

                if required_size_for_min_cost is None or required_size_for_min_cost <= 0: raise ValueError("Invalid required size calculation for min cost")
                lg.info(f"  Size required to meet min cost: {required_size_for_min_cost.normalize()} {size_unit}")

                # Check if this required size exceeds max amount limit
                if max_amount < Decimal('inf') and required_size_for_min_cost > max_amount:
                    lg.error(f"{NEON_RED}Sizing failed: Cannot meet minimum cost ({min_cost.normalize()}) without exceeding maximum amount limit ({max_amount.normalize()}).{RESET}")
                    return None

                # Adjust size up, ensuring it's still meets the original min_amount if that was larger
                adjusted_size = max(min_amount, required_size_for_min_cost)
                cost_adjustment_applied = True

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed: Failed to calculate size needed for minimum cost: {cost_calc_err}.{RESET}")
                return None

        # Check Maximum Cost
        elif max_cost < Decimal('inf') and estimated_cost_after_amount_limits > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost_after_amount_limits.normalize()} exceeds maximum cost {max_cost.normalize()}. Attempting to reduce size.{RESET}")
            max_size_for_max_cost = None
            try:
                # Calculate max size allowed by max cost: MaxCost / (Price * ContrSize) [Lin] or MaxCost * Price / ContrSize [Inv]
                if not is_inverse:
                    denominator = entry_price * contract_size
                    if denominator <= 0: raise ZeroDivisionError("Linear cost calc denominator zero/negative")
                    max_size_for_max_cost = max_cost / denominator
                else:
                    if entry_price <= 0 or contract_size <= 0: raise ValueError("Invalid price/contract size for inverse cost calc")
                    max_size_for_max_cost = (max_cost * entry_price) / contract_size

                if max_size_for_max_cost is None or max_size_for_max_cost <= 0: raise ValueError("Invalid max size calculation for max cost")
                lg.info(f"  Maximum size allowed by max cost: {max_size_for_max_cost.normalize()} {size_unit}")

                # Adjust size down, ensuring it's still at least the minimum amount
                adjusted_size = max(min_amount, min(adjusted_size, max_size_for_max_cost)) # Take the smaller of current adjusted or max allowed by cost
                cost_adjustment_applied = True

            except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
                lg.error(f"{NEON_RED}Sizing failed: Failed to calculate maximum size allowed by maximum cost: {cost_calc_err}.{RESET}")
                return None

    elif min_cost > 0 or max_cost < Decimal('inf'):
        lg.warning("Could not estimate cost for limit check. Proceeding without cost limit adjustments.")

    if cost_adjustment_applied:
        lg.info(f"  Size after Cost Limits: {adjusted_size.normalize()} {size_unit}")


    # 3. Apply Amount Precision (Rounding DOWN to step size for safety)
    final_size = adjusted_size
    try:
        # Use CCXT's built-in precision formatting (usually truncates/floors)
        # Convert to float for CCXT function, then back to Decimal
        amount_str_formatted = exchange.amount_to_precision(symbol, float(adjusted_size))
        final_size_ccxt = Decimal(amount_str_formatted)

        # Manual Rounding Down as a more conservative approach / fallback
        # size = floor(value / step) * step
        final_size_manual_floor = (adjusted_size // amount_precision_step) * amount_precision_step

        # Choose the manually floored value as it's generally safer (avoids rounding up)
        final_size = final_size_manual_floor

        if final_size != adjusted_size:
            lg.info(f"Applied amount precision (Floor): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
            # Log the CCXT version for comparison if different
            if abs(final_size_ccxt - final_size) > Decimal('1e-18'):
                 lg.debug(f"  (CCXT amount_to_precision suggested: {final_size_ccxt.normalize()})")

    except Exception as fmt_err:
        lg.error(f"{NEON_RED}Error applying amount precision: {fmt_err}. Using unrounded size: {final_size.normalize()}{RESET}")
        # Continue with the unrounded size, but subsequent checks might fail

    # --- Final Validation after Precision ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Sizing failed: Final size after precision is zero or negative ({final_size.normalize()}).{RESET}")
        return None
    # Check Min Amount again after rounding down
    if min_amount > 0 and final_size < min_amount:
        lg.error(f"{NEON_RED}Sizing failed: Final size {final_size.normalize()} is below minimum amount {min_amount.normalize()} after precision adjustment.{RESET}")
        # Option: Could bump up to min_amount here, but that slightly violates risk. Aborting is safer.
        return None
    # Check Max Amount again (should be fine if rounded down, but check)
    if max_amount < Decimal('inf') and final_size > max_amount:
         lg.error(f"{NEON_RED}Sizing failed: Final size {final_size.normalize()} exceeds maximum amount {max_amount.normalize()} after precision.{RESET}")
         return None

    # Final check on cost after precision applied (especially if rounded down)
    final_cost = estimate_cost(final_size, entry_price)
    if final_cost is not None:
        lg.debug(f"  Final Estimated Cost: {final_cost.normalize()} {quote_currency}")
        if min_cost > 0 and final_cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Final cost {final_cost.normalize()} is slightly below min cost {min_cost.normalize()} after precision rounding.{RESET}")
             # Option: Try bumping size up by one step if possible within other limits
             try:
                 next_step_size = final_size + amount_precision_step
                 next_step_cost = estimate_cost(next_step_size, entry_price)

                 if next_step_cost is not None:
                     # Check if bumping up is valid (meets min cost, doesn't exceed max amount/cost)
                     can_bump_up = (next_step_cost >= min_cost) and \
                                   (max_amount == Decimal('inf') or next_step_size <= max_amount) and \
                                   (max_cost == Decimal('inf') or next_step_cost <= max_cost)

                     if can_bump_up:
                         lg.info(f"{NEON_YELLOW}Bumping final size up by one step to {next_step_size.normalize()} to meet minimum cost.{RESET}")
                         final_size = next_step_size
                     else:
                         lg.error(f"{NEON_RED}Sizing failed: Cannot meet minimum cost even by bumping size one step due to other limits (Max Amount/Cost).{RESET}")
                         return None
                 else:
                      lg.error(f"{NEON_RED}Sizing failed: Could not estimate cost for bumped size {next_step_size.normalize()}. Cannot verify min cost.{RESET}")
                      return None

             except Exception as bump_err:
                 lg.error(f"{NEON_RED}Sizing failed: Error trying to bump size for min cost: {bump_err}.{RESET}")
                 return None
        elif max_cost < Decimal('inf') and final_cost > max_cost:
            # This shouldn't happen if rounding down, but check defensively
             lg.error(f"{NEON_RED}Sizing failed: Final cost {final_cost.normalize()} exceeds maximum cost {max_cost.normalize()} after precision.{RESET}")
             return None
    elif min_cost > 0:
         lg.warning("Could not perform final cost check after precision adjustment.")


    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size: {final_size.normalize()} {size_unit} <<< {RESET}")
    lg.info(f"{BRIGHT}--- End Position Sizing ---{RESET}")
    return final_size

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: MarketInfo,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Places a market order (buy or sell) using `create_order`.

    - Maps trade signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT") to order sides ("buy", "sell").
    - Handles Bybit V5 specific parameters (category, positionIdx, reduceOnly, timeInForce).
    - Includes retry logic for network/exchange errors and rate limits.
    - Identifies and handles non-retryable order errors (e.g., insufficient funds, invalid parameters).
    - Logs order details clearly.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        trade_signal: The signal driving the trade ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated position size (must be positive Decimal).
        market_info: The standardized MarketInfo dictionary.
        logger: The logger instance for status messages.
        reduce_only: Set to True for closing orders to ensure they only reduce/close a position.
        params: Optional additional parameters to pass to create_order's `params` argument.

    Returns:
        The order dictionary returned by CCXT upon successful placement, or None if the
        order fails after retries or due to fatal errors.
    """
    lg = logger
    # Map signal to CCXT side ('buy' or 'sell')
    side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}
    side = side_map.get(trade_signal.upper())

    # --- Input Validation ---
    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided to place_trade. Must be BUY, SELL, EXIT_LONG, or EXIT_SHORT.")
        return None
    if not isinstance(position_size, Decimal) or position_size <= Decimal('0'):
        lg.error(f"Invalid position size provided to place_trade: {position_size}. Must be a positive Decimal.")
        return None

    order_type = 'market' # This bot currently only uses market orders
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    # Use settle currency for contracts, base for spot, fallback if missing
    settle_currency = market_info.get('settle') if is_contract else base_currency
    size_unit = settle_currency if is_contract and settle_currency else base_currency

    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info['id'] # Use the exchange-specific market ID

    # --- Prepare Order Arguments ---
    # CCXT typically expects float amount, convert Decimal carefully
    try:
         amount_float = float(position_size)
    except Exception as float_err:
         lg.error(f"Failed to convert position size {position_size.normalize()} to float for API call: {float_err}")
         return None

    order_args = {
        'symbol': market_id,
        'type': order_type,
        'side': side,
        'amount': amount_float,
    }
    order_params = {} # For exchange-specific parameters

    # --- Bybit V5 Specific Parameters ---
    if 'bybit' in exchange.id.lower() and is_contract:
        try:
            category = market_info.get('contract_type_str', 'Linear').lower()
            order_params = {
                'category': category,
                'positionIdx': 0  # Use 0 for one-way mode (required by Bybit V5 for non-hedge mode)
            }
            if reduce_only:
                order_params['reduceOnly'] = True
                # Use IOC for reduceOnly market orders to prevent resting if market moves away quickly
                # (FOK might also work, but IOC is safer if partial fill is acceptable for closing)
                order_params['timeInForce'] = 'IOC' # Immediate Or Cancel
            lg.debug(f"Using Bybit V5 order params: {order_params}")
        except Exception as e:
            lg.error(f"Failed to set Bybit V5 parameters for order: {e}. Order might fail.")
            # Proceed cautiously without params if setting failed

    # Merge any additional custom parameters provided by the caller
    if params:
        order_params.update(params)

    if order_params:
        order_args['params'] = order_params # Add exchange-specific params to the main args

    # Log the trade attempt
    lg.info(f"{BRIGHT}===> Attempting {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <==={RESET}")
    if order_params: lg.debug(f"  with Params: {order_params}")

    # --- Execute Order with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            order_result = exchange.create_order(**order_args)

            # Log Success
            order_id = order_result.get('id', 'N/A')
            status = order_result.get('status', 'N/A')
            # Safely format potential Decimal/float/str values from result
            avg_price_str = _safe_market_decimal(order_result.get('average'), 'average')
            filled_amount_str = _safe_market_decimal(order_result.get('filled'), 'filled')
            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET} "
                f"ID: {order_id}, Status: {status}"
            )
            if avg_price_str: log_msg += f", Avg Fill Price: ~{avg_price_str.normalize()}"
            if filled_amount_str: log_msg += f", Filled Amount: {filled_amount_str.normalize()}"
            lg.info(log_msg)
            lg.debug(f"Full order result: {order_result}")
            return order_result # Return the successful order details

        # --- Error Handling with Retries ---
        except ccxt.InsufficientFunds as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({action_desc}): Insufficient funds for {symbol} {side} {position_size}. Error: {e}{RESET}")
            return None # Non-retryable
        except ccxt.InvalidOrder as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed ({action_desc}): Invalid order parameters for {symbol}. Error: {e}{RESET}")
            lg.error(f"  Order arguments used: {order_args}")
            # Add hints based on common causes
            if "minimum" in str(e).lower() or "too small" in str(e).lower():
                 lg.error(f"  >> Hint: Check position size ({position_size.normalize()}) against market minimum amount/cost limits.")
            elif "precision" in str(e).lower():
                 lg.error(f"  >> Hint: Check position size ({position_size.normalize()}) against amount precision step ({market_info.get('amount_precision_step_decimal')}).")
            elif "exceed" in str(e).lower() or "too large" in str(e).lower():
                 lg.error(f"  >> Hint: Check position size ({position_size.normalize()}) against market maximum amount/cost limits.")
            return None # Non-retryable
        except ccxt.ExchangeError as e:
            last_exception = e
            err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            lg.error(f"{NEON_RED}Order Failed ({action_desc}): Exchange error placing order for {symbol}. Error: {e} (Code: {err_code_str}){RESET}")

            # Check for known fatal Bybit error codes related to orders
            # Based on Bybit V5 API docs and common issues
            fatal_order_codes = [
                '10001', # Parameter error
                '10004', # Sign check error
                '110007',# Qty exceed max limit per order
                '110013',# Parameter error (generic)
                '110014',# Qty must be greater than 0
                '110017',# Order price/qty violates contract rules (precision?)
                '110025',# Position idx not match position mode (can happen if mode changed externally)
                '110040',# Order cost exceed limit
                '30086', # reduceOnly check failed - TIF is not IOC/FOK
                '3303001',# Invalid symbol
                '3303005',# Price is too high/low compared to mark price
                '3400060',# Order qty exceeds risk limit level
                '3400088',# The order quantity is invalid (lot size?)
            ]
            # Common fatal message fragments
            fatal_messages = ["invalid parameter", "precision", "exceed limit", "risk limit", "invalid symbol", "reduce only check failed"]

            if err_code_str in fatal_order_codes or any(msg in str(e).lower() for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE order error.{RESET}")
                return None # Non-retryable

            # Assume other exchange errors might be temporary and retry if attempts remain
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError placing order. Last error: {last_exception}{RESET}")
                 return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError placing order. Last error: {last_exception}{RESET}")
                return None

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error placing order: {e}. Check API Key/Secret/Permissions. Stopping order placement.{RESET}")
             return None # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)
            return None # Stop on unexpected errors

        # Increment attempt counter (only if not a rate limit wait) and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """
    Internal helper: Sets Stop Loss (SL), Take Profit (TP), and/or Trailing Stop Loss (TSL)
    for an existing position using Bybit's V5 private API endpoint `/v5/position/set-trading-stop`.

    **Important:** This uses a direct API call (`private_post`) and relies on Bybit's specific
    V5 endpoint and parameters. It might break if Bybit changes its API structure. CCXT's
    standard methods might not fully support Bybit's V5 combined TSL/SL/TP parameters.

    Handles parameter validation (e.g., SL/TP/Activation relative to entry), price formatting
    to market precision, and API response checking. TSL settings take precedence over fixed SL.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        market_info: The standardized MarketInfo dictionary.
        position_info: The standardized PositionInfo dictionary for the open position.
        logger: The logger instance for status messages.
        stop_loss_price: Desired fixed SL price (Decimal). Set to 0 or None to clear/ignore.
        take_profit_price: Desired fixed TP price (Decimal). Set to 0 or None to clear/ignore.
        trailing_stop_distance: Desired TSL distance (in price units, Decimal). Set to 0 or None to clear/ignore. Must be positive if setting TSL.
        tsl_activation_price: Price at which TSL should activate (Decimal). Required if distance > 0.

    Returns:
        True if the protection was set/updated successfully via API or if no change was needed.
        False if validation fails, API call fails after retries, or a critical error occurs.
    """
    lg = logger
    endpoint = '/v5/position/set-trading-stop' # Bybit V5 endpoint

    # --- Input and State Validation ---
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False # Cannot set SL/TP/TSL on spot
    if not position_info:
        lg.error(f"Protection setting failed for {symbol}: Missing position information.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short']:
        lg.error(f"Protection setting failed for {symbol}: Invalid position side ('{pos_side}').")
        return False
    try:
        if entry_price_str is None: raise ValueError("Missing entry price")
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed for {symbol}: Invalid or missing entry price ('{entry_price_str}'): {e}")
        return False
    try:
        price_tick = market_info['price_precision_step_decimal']
        if price_tick is None or price_tick <= 0: raise ValueError("Invalid price tick size")
    except (KeyError, ValueError, TypeError) as e:
         lg.error(f"Protection setting failed for {symbol}: Could not get valid price precision from market info: {e}")
         return False

    params_to_set: Dict[str, Any] = {} # Parameters to send in the API request
    log_parts: List[str] = [f"{BRIGHT}Attempting to set protection for {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):{RESET}"]
    any_protection_requested = False # Flag to check if any valid protection was requested
    set_tsl_active = False # Flag if TSL distance > 0 is being set

    # --- Format and Validate Protection Parameters ---
    try:
        # Helper to format price to exchange precision string
        def format_price_param(price_decimal: Optional[Decimal], param_name: str) -> Optional[str]:
            """Formats price to string, respecting exchange precision. Returns None if invalid."""
            if price_decimal is None: return None
            # Allow "0" to clear protection
            if price_decimal == 0: return "0"
            # Reject negative prices immediately
            if price_decimal < 0:
                lg.warning(f"Invalid negative price {price_decimal.normalize()} provided for {param_name}. Ignoring.")
                return None
            try:
                # Use CCXT's price_to_precision for correct rounding/truncating
                formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal))
                # Final check: ensure formatted price is still positive after formatting
                if Decimal(formatted_str) > 0:
                     return formatted_str
                else:
                     lg.warning(f"Formatted {param_name} '{formatted_str}' resulted in zero or negative value. Ignoring.")
                     return None
            except Exception as e:
                lg.error(f"Failed to format {param_name} value {price_decimal.normalize()} to exchange precision: {e}.")
                return None

        # --- Trailing Stop Loss (TSL) ---
        # Bybit requires TSL distance (trailingStop) and activation price (activePrice)
        if isinstance(trailing_stop_distance, Decimal):
            any_protection_requested = True
            if trailing_stop_distance > 0: # Setting an active TSL
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0:
                    lg.error(f"TSL request failed: Valid positive activation price ({tsl_activation_price}) is required when TSL distance ({trailing_stop_distance.normalize()}) > 0.")
                else:
                    # Validate activation price makes sense relative to entry (must be beyond entry)
                    is_valid_activation = (pos_side == 'long' and tsl_activation_price > entry_price) or \
                                          (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_activation:
                        lg.error(f"TSL request failed: Activation price {tsl_activation_price.normalize()} must be beyond entry price {entry_price.normalize()} for a {pos_side} position.")
                    else:
                        # Ensure distance is at least one tick
                        min_valid_distance = max(trailing_stop_distance, price_tick)
                        fmt_distance = format_price_param(min_valid_distance, "TSL Distance")
                        fmt_activation = format_price_param(tsl_activation_price, "TSL Activation Price")

                        if fmt_distance and fmt_activation:
                            params_to_set['trailingStop'] = fmt_distance
                            params_to_set['activePrice'] = fmt_activation
                            log_parts.append(f"  - Setting TSL: Distance={fmt_distance}, Activation={fmt_activation}")
                            set_tsl_active = True # Mark TSL as being actively set
                        else:
                            lg.error(f"TSL request failed: Could not format TSL parameters (Distance: {fmt_distance}, Activation: {fmt_activation}). Check precision/values.")

            elif trailing_stop_distance == 0: # Clearing TSL
                params_to_set['trailingStop'] = "0"
                # Also clear activation price when clearing TSL distance for Bybit
                params_to_set['activePrice'] = "0"
                log_parts.append("  - Clearing TSL (Distance and Activation Price set to 0)")
                # set_tsl_active remains False

        # --- Fixed Stop Loss (SL) ---
        # Can only set fixed SL if TSL is *not* being actively set (Bybit limitation)
        if not set_tsl_active:
            if isinstance(stop_loss_price, Decimal):
                any_protection_requested = True
                if stop_loss_price > 0: # Setting an active SL
                    # Validate SL price makes sense relative to entry (must be beyond entry)
                    is_valid_sl = (pos_side == 'long' and stop_loss_price < entry_price) or \
                                  (pos_side == 'short' and stop_loss_price > entry_price)
                    if not is_valid_sl:
                        lg.error(f"Fixed SL request failed: SL price {stop_loss_price.normalize()} must be beyond entry price {entry_price.normalize()} for a {pos_side} position.")
                    else:
                        fmt_sl = format_price_param(stop_loss_price, "Stop Loss")
                        if fmt_sl:
                            params_to_set['stopLoss'] = fmt_sl
                            log_parts.append(f"  - Setting Fixed SL: {fmt_sl}")
                        else:
                            lg.error(f"Fixed SL request failed: Could not format SL price {stop_loss_price.normalize()}.")
                elif stop_loss_price == 0: # Clearing SL
                    # Only send "0" if SL field wasn't already populated by TSL logic (which it shouldn't be if set_tsl_active is False)
                    if 'stopLoss' not in params_to_set:
                         params_to_set['stopLoss'] = "0"
                         log_parts.append("  - Clearing Fixed SL (set to 0)")
        elif isinstance(stop_loss_price, Decimal) and stop_loss_price > 0:
             # TSL is active, cannot set fixed SL
             lg.warning(f"Ignoring fixed SL request ({stop_loss_price.normalize()}) because active TSL is being set.")


        # --- Fixed Take Profit (TP) ---
        # TP can usually be set alongside SL or TSL
        if isinstance(take_profit_price, Decimal):
            any_protection_requested = True
            if take_profit_price > 0: # Setting an active TP
                # Validate TP price makes sense relative to entry (must be beyond entry)
                is_valid_tp = (pos_side == 'long' and take_profit_price > entry_price) or \
                              (pos_side == 'short' and take_profit_price < entry_price)
                if not is_valid_tp:
                    lg.error(f"Fixed TP request failed: TP price {take_profit_price.normalize()} must be beyond entry price {entry_price.normalize()} for a {pos_side} position.")
                else:
                    fmt_tp = format_price_param(take_profit_price, "Take Profit")
                    if fmt_tp:
                        params_to_set['takeProfit'] = fmt_tp
                        log_parts.append(f"  - Setting Fixed TP: {fmt_tp}")
                    else:
                        lg.error(f"Fixed TP request failed: Could not format TP price {take_profit_price.normalize()}.")
            elif take_profit_price == 0: # Clearing TP
                 if 'takeProfit' not in params_to_set:
                     params_to_set['takeProfit'] = "0"
                     log_parts.append("  - Clearing Fixed TP (set to 0)")

    except Exception as fmt_err:
        lg.error(f"Error during protection parameter formatting/validation: {fmt_err}", exc_info=True)
        return False

    # --- Check if any valid parameters were actually prepared for the API call ---
    if not params_to_set:
        if any_protection_requested:
            lg.warning(f"{NEON_YELLOW}No valid protection parameters to set for {symbol} after formatting/validation. No API call made.{RESET}")
            # Return False because the requested action couldn't be fulfilled due to validation errors
            return False
        else:
            lg.debug(f"No protection changes requested or needed for {symbol}. No API call required.")
            return True # Success, as no action was needed

    # --- Prepare Final API Parameters ---
    category = market_info.get('contract_type_str', 'Linear').lower()
    market_id = market_info['id']
    # Get position index (should be 0 for one-way mode)
    position_idx = 0 # Default to one-way mode
    try:
        # Attempt to get positionIdx from position info if available (might indicate hedge mode)
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
            if position_idx != 0:
                 lg.warning(f"Detected positionIdx={position_idx}. Ensure this matches your Bybit account mode (One-Way vs Hedge). Using detected index.")
    except (ValueError, TypeError):
        lg.debug("Could not parse positionIdx from position info, defaulting to 0 (one-way mode).")
        position_idx = 0

    # Construct the final parameters dictionary for the API call
    final_api_params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full', # Use 'Full' for setting SL/TP on the entire position
        # Common trigger prices: LastPrice, MarkPrice, IndexPrice
        'slTriggerBy': 'LastPrice', # Trigger SL based on Last Price (configurable maybe later)
        'tpTriggerBy': 'LastPrice', # Trigger TP based on Last Price
        # Order types when triggered: Market, Limit (Limit requires slLimitPrice/tpLimitPrice)
        'slOrderType': 'Market',    # Use Market order when SL is triggered
        'tpOrderType': 'Market',    # Use Market order when TP is triggered
        'positionIdx': position_idx # Specify position index (0 for one-way)
    }
    final_api_params.update(params_to_set) # Add the specific SL/TP/TSL values

    # Log the attempt only if there are parameters to set
    lg.info("\n".join(log_parts)) # Log what is being attempted
    lg.debug(f"  Final API parameters for {endpoint}: {final_api_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post to {endpoint} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            # Use exchange.private_post for endpoints not directly mapped by CCXT standard methods
            response = exchange.private_post(endpoint, params=final_api_params)
            lg.debug(f"Set protection raw API response: {response}")

            # --- Check Bybit V5 Response ---
            ret_code_str = None
            ret_msg = "Unknown message"
            if isinstance(response, dict):
                 ret_code_str = str(response.get('retCode', response.get('code'))) # Check both keys
                 ret_msg = response.get('retMsg', response.get('message', 'Unknown Bybit API message'))

            if ret_code_str == '0':
                 # Check message for "not modified" cases which are still success
                 # Bybit messages can vary, check common fragments
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "order is not modified"]
                 if any(m in ret_msg.lower() for m in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Protection parameters already set or no change needed. (Message: {ret_msg}){RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection parameters successfully set/updated for {symbol}.{RESET}")
                 return True # Success

            else:
                 # Log the specific Bybit error and raise ExchangeError for retry or handling
                 error_message = f"Bybit API error setting protection: {ret_msg} (Code: {ret_code_str})"
                 lg.error(f"{NEON_RED}{error_message}{RESET}")
                 raise ccxt.ExchangeError(error_message) # Let retry logic handle it

        # --- Standard CCXT Error Handling with Retries ---
        except ccxt.ExchangeError as e: # Catches re-raised Bybit errors or other CCXT exchange errors
            last_exception = e
            err_code_str = str(getattr(e, 'code', '') or getattr(e, 'retCode', ''))
            err_str = str(e).lower()
            lg.warning(f"{NEON_YELLOW}Exchange error setting protection for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")

            # Check for known fatal/non-retryable error codes/messages
            # Based on Bybit V5 API docs and common issues
            fatal_protect_codes = [
                '10001', # Parameter error
                '10002', # API key invalid
                '110013',# Parameter error (generic)
                '110036',# SL price cannot be higher/lower than entry/mark price (depending on trigger)
                '110043',# Position status is not normal (e.g., closing)
                '110084',# SL price invalid / out of range
                '110085',# TP price invalid / out of range
                '110086',# TP price cannot be lower/higher than entry/mark price
            ]
            fatal_messages = ["invalid parameter", "invalid price", "cannot be higher", "cannot be lower", "position status"]

            if err_code_str in fatal_protect_codes or any(msg in err_str for msg in fatal_messages):
                 lg.error(f"{NEON_RED} >> Hint: This appears to be a NON-RETRYABLE protection setting error. Aborting protection set.{RESET}")
                 return False # Fatal error for this operation

            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError setting protection. Last error: {last_exception}{RESET}")
                 return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting protection for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError setting protection. Last error: {last_exception}{RESET}")
                return False

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded setting protection for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Check API Key/Secret/Permissions. Stopping protection set.{RESET}")
            return False # Fatal error for this operation

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol} (Attempt {attempts + 1}): {e}{RESET}", exc_info=True)
            # Unexpected errors are likely fatal for this operation
            return False

        # Increment attempt counter and delay before retrying (only for retryable errors)
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False

def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: MarketInfo, position_info: PositionInfo, config: Dict[str, Any],
                             logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool:
    """
    Calculates Trailing Stop Loss (TSL) parameters based on configuration settings
    (callback rate, activation percentage) and the current position's entry price.
    It then calls the internal `_set_position_protection` function to apply the TSL
    and optionally a fixed Take Profit.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol.
        market_info: The standardized MarketInfo dictionary.
        position_info: The standardized PositionInfo dictionary for the open position.
        config: The main configuration dictionary containing protection settings.
        logger: The logger instance.
        take_profit_price: An optional fixed Take Profit price (Decimal) to set simultaneously.
                           Set to 0 or None to clear/ignore existing TP.

    Returns:
        True if the TSL (and optional TP) was successfully calculated and the API call
        to set protection was initiated successfully (returned True), False otherwise.
    """
    lg = logger
    prot_cfg = config.get("protection", {}) # Get protection sub-dictionary

    # --- Input Validation ---
    if not market_info or not position_info:
        lg.error(f"TSL calculation failed for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short']:
        lg.error(f"TSL calculation failed for {symbol}: Invalid position side ('{pos_side}').")
        return False

    try:
        # Extract parameters and convert to Decimal
        if entry_price_str is None: raise ValueError("Missing entry price")
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(prot_cfg["trailing_stop_callback_rate"])) # e.g., 0.005 for 0.5%
        activation_percentage = Decimal(str(prot_cfg["trailing_stop_activation_percentage"])) # e.g., 0.003 for 0.3%
        price_tick = market_info['price_precision_step_decimal']

        # Validate parameters
        if not (entry_price > 0 and callback_rate > 0 and activation_percentage >= 0 and price_tick and price_tick > 0):
             raise ValueError("Invalid input values (entry, callback rate > 0, activation % >= 0, or positive tick size).")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"TSL calculation failed for {symbol}: Invalid configuration or market/position info: {e}")
        lg.debug(f"  Problematic Inputs: Entry='{entry_price_str}', Config={prot_cfg}, Tick={market_info.get('price_precision_step_decimal')}")
        return False

    # --- Calculate TSL Activation Price and Distance ---
    try:
        lg.info(f"Calculating TSL for {symbol} ({pos_side.upper()}): Entry={entry_price.normalize()}, Act%={activation_percentage:.3%}, CB%={callback_rate:.3%}")

        # 1. Calculate Activation Price
        activation_offset = entry_price * activation_percentage
        raw_activation_price = (entry_price + activation_offset) if pos_side == 'long' else (entry_price - activation_offset)

        # Quantize activation price to the nearest tick (away from entry)
        if pos_side == 'long':
            # Round up for long activation to ensure it's strictly beyond entry
            quantized_activation_price = (raw_activation_price / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick
            # Ensure activation is strictly greater than entry by at least one tick
            tsl_activation_price = max(quantized_activation_price, entry_price + price_tick)
        else: # Short position
            # Round down for short activation to ensure it's strictly beyond entry
            quantized_activation_price = (raw_activation_price / price_tick).quantize(Decimal('1'), ROUND_DOWN) * price_tick
            # Ensure activation is strictly less than entry by at least one tick
            tsl_activation_price = min(quantized_activation_price, entry_price - price_tick)

        # Final check on activation price validity
        if tsl_activation_price <= 0 or \
           (pos_side == 'long' and tsl_activation_price <= entry_price) or \
           (pos_side == 'short' and tsl_activation_price >= entry_price):
            lg.error(f"TSL calculation failed: Calculated Activation Price ({tsl_activation_price.normalize()}) is invalid relative to entry ({entry_price.normalize()}).")
            return False

        # 2. Calculate Trailing Distance (based on activation price and callback rate)
        # Distance = Activation Price * Callback Rate
        # Use abs() in case activation price somehow became negative (shouldn't happen)
        raw_trailing_distance = abs(tsl_activation_price) * callback_rate

        # Quantize distance UP to the nearest tick and ensure it's at least one tick
        tsl_trailing_distance = max((raw_trailing_distance / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick, price_tick)

        if tsl_trailing_distance <= 0:
            lg.error(f"TSL calculation failed: Calculated Trailing Distance ({tsl_trailing_distance.normalize()}) is zero or negative.")
            return False

        lg.info(f"  => Calculated TSL Activation Price: {tsl_activation_price.normalize()}")
        lg.info(f"  => Calculated TSL Trailing Distance: {tsl_trailing_distance.normalize()}")
        tp_action = "None"
        if isinstance(take_profit_price, Decimal):
             tp_action = f"{take_profit_price.normalize()}" if take_profit_price != 0 else "Clear TP"
        lg.info(f"  => Setting alongside Fixed TP: {tp_action}")

        # --- Call Internal Function to Set Protection ---
        # Pass None for fixed stop_loss_price as TSL takes precedence when distance > 0
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL when active
            take_profit_price=take_profit_price, # Pass through TP setting/clearing
            trailing_stop_distance=tsl_trailing_distance,
            tsl_activation_price=tsl_activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during TSL calculation or setting: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy Implementation ---
class VolumaticOBStrategy:
    """
    Implements the core logic for the Volumatic Trend and Pivot Order Block strategy.

    Responsibilities:
    - Calculates Volumatic Trend indicators (EMAs, ATR Bands, Volume Normalization).
    - Identifies Pivot Highs/Lows based on configuration.
    - Creates Order Blocks (OBs) from pivots using configured source (Wicks/Body).
    - Manages the state of OBs (active, violated, extends).
    - Prunes the list of active OBs to a maximum number per type.
    - Returns structured analysis results including the processed DataFrame.
    """
    def __init__(self, config: Dict[str, Any], market_info: MarketInfo, logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the config.

        Args:
            config: The main configuration dictionary.
            market_info: The standardized MarketInfo dictionary.
            logger: The logger instance for this strategy instance.

        Raises:
            ValueError: If critical configuration parameters are invalid.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger # Alias for convenience

        strategy_cfg = config.get("strategy_params", {})
        # Protection config not directly used here but could be for future features

        # Load strategy parameters (already validated by load_config)
        try:
            self.vt_length = int(strategy_cfg["vt_length"])
            self.vt_atr_period = int(strategy_cfg["vt_atr_period"])
            self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"])
            self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))

            self.ob_source = str(strategy_cfg["ob_source"]) # "Wicks" or "Body"
            self.ph_left = int(strategy_cfg["ph_left"])
            self.ph_right = int(strategy_cfg["ph_right"])
            self.pl_left = int(strategy_cfg["pl_left"])
            self.pl_right = int(strategy_cfg["pl_right"])
            self.ob_extend = bool(strategy_cfg["ob_extend"])
            self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])

            # Basic sanity checks on loaded values
            if not (self.vt_length > 0 and self.vt_atr_period > 0 and self.vt_vol_ema_length > 0 and \
                    self.vt_atr_multiplier > 0 and self.ph_left > 0 and self.ph_right > 0 and \
                    self.pl_left > 0 and self.pl_right > 0 and self.ob_max_boxes > 0):
                raise ValueError("One or more strategy parameters are invalid (must be positive).")
            if self.ob_source not in ["Wicks", "Body"]:
                 raise ValueError(f"Invalid ob_source '{self.ob_source}'. Must be 'Wicks' or 'Body'.")

        except (KeyError, ValueError, TypeError) as e:
            self.lg.error(f"FATAL: Failed to initialize VolumaticOBStrategy due to invalid config: {e}")
            self.lg.debug(f"Strategy Config received: {strategy_cfg}")
            raise ValueError(f"Strategy initialization failed: {e}") from e


        # Initialize Order Block storage
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum data length required based on longest lookback period
        # EMA needs length, SWMA needs 4, ATR needs period, Vol Norm needs length
        # Pivots need left + right + 1
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length) # *2 for EMA stability
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1)
        # Add buffer for indicator stabilization (e.g., 50 candles)
        self.min_data_len = max(required_for_vt, required_for_pivots) + 50

        # Log initialized parameters
        self.lg.info(f"{NEON_CYAN}--- Initializing VolumaticOB Strategy Engine ---{RESET}")
        self.lg.info(f"  VT Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Vol EMA Length={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  OB Params: Source='{self.ob_source}', PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend OBs={self.ob_extend}, Max Active OBs={self.ob_max_boxes}")
        self.lg.info(f"  Minimum Historical Data Required: ~{self.min_data_len} candles")

        # Warning if required data exceeds typical API limits significantly
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 10: # Add small buffer to limit check
            self.lg.warning(f"{NEON_YELLOW}CONFIGURATION NOTE:{RESET} Strategy requires {self.min_data_len} candles, which might exceed the API fetch limit ({BYBIT_API_KLINE_LIMIT}) in a single request. "
                          f"Ensure 'fetch_limit' in config.json is sufficient or consider reducing long lookback periods (vt_atr_period, vt_vol_ema_length).")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates EMA(SWMA(series, 4), length).

        SWMA(4) uses weights [1, 2, 2, 1] / 6. This combination provides smoothing.
        Handles NaN inputs gracefully.

        Args:
            series: The input Pandas Series (typically 'close' prices as float).
            length: The length for the final EMA calculation.

        Returns:
            A Pandas Series containing the calculated indicator (as float).
        """
        if not isinstance(series, pd.Series) or len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index, dtype=float) # Return NaNs if input is invalid

        # Ensure numeric type (already done in update, but double check)
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all(): # Check if all values are NaN after conversion
            return pd.Series(np.nan, index=series.index, dtype=float)

        # Calculate SWMA(4) using rolling apply with dot product
        # Weights: [1, 2, 2, 1] / 6
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        # min_periods=4 ensures we only calculate where 4 data points are available
        swma = numeric_series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate EMA of the SWMA result using pandas_ta
        # fillna=np.nan prevents forward-filling issues if EMA starts later
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan)
        return ema_of_swma # Returns float Series

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, is_high: bool) -> pd.Series:
        """
        Identifies Pivot Highs or Pivot Lows in a series.

        A Pivot High is a value strictly higher than `left_bars` values to its left
        and `right_bars` values to its right.
        A Pivot Low is a value strictly lower than `left_bars` values to its left
        and `right_bars` values to its right.
        Uses strict inequality ('>' or '<'). Handles NaNs.

        Args:
            series: The Pandas Series (float) to search for pivots (e.g., 'high' or 'low').
            left_bars: The number of bars to look back to the left (must be >= 1).
            right_bars: The number of bars to look forward to the right (must be >= 1).
            is_high: True to find Pivot Highs, False to find Pivot Lows.

        Returns:
            A boolean Pandas Series, True where a pivot point is identified.
        """
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1:
            return pd.Series(False, index=series.index, dtype=bool) # Return all False if input invalid

        # Ensure numeric type
        num_series = pd.to_numeric(series, errors='coerce')
        if num_series.isnull().all():
            return pd.Series(False, index=series.index, dtype=bool)

        # Initialize all points as potential pivots (where value is not NaN)
        pivot_conditions = num_series.notna()

        # Check left bars: current >(or <) previous bars
        for i in range(1, left_bars + 1):
            shifted = num_series.shift(i)
            if is_high:
                pivot_conditions &= (num_series > shifted)
            else:
                pivot_conditions &= (num_series < shifted)
            # Handle NaNs introduced by shift or in original data
            pivot_conditions = pivot_conditions.fillna(False)


        # Check right bars: current >(or <) future bars
        for i in range(1, right_bars + 1):
            shifted = num_series.shift(-i)
            if is_high:
                pivot_conditions &= (num_series > shifted)
            else:
                pivot_conditions &= (num_series < shifted)
            # Handle NaNs introduced by shift or in original data
            pivot_conditions = pivot_conditions.fillna(False)

        # Final fillna for any remaining edge cases
        return pivot_conditions.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes historical OHLCV data to calculate indicators and manage Order Blocks.

        Steps:
        1. Validates input DataFrame.
        2. Converts relevant columns to float for TA library compatibility.
        3. Calculates indicators: ATR, Volumatic EMAs, Trend, Bands, Volume Normalization, Pivots.
        4. Converts calculated float indicators back to Decimal format in the main DataFrame.
        5. Cleans the final DataFrame (drops rows with essential NaNs).
        6. Identifies new Order Blocks based on pivots and configuration.
        7. Updates existing Order Blocks (checks for violations, extends if configured).
        8. Prunes active Order Blocks to `ob_max_boxes`.
        9. Compiles and returns the `StrategyAnalysisResults`.

        Args:
            df_input: The input Pandas DataFrame containing OHLCV data (index=Timestamp,
                      columns including 'open', 'high', 'low', 'close', 'volume' as Decimals).

        Returns:
            A StrategyAnalysisResults TypedDict containing the processed DataFrame,
            current trend, active OBs, and other key indicator values for the latest candle.
            Returns default/empty results if processing fails critically.
        """
        # Prepare a default/empty result structure for failure cases
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.lg.error("Strategy update failed: Input DataFrame is empty.")
            return empty_results

        # Work on a copy to avoid modifying the original DataFrame passed in
        df = df_input.copy()

        # --- Input Data Validation ---
        if not isinstance(df.index, pd.DatetimeIndex):
            self.lg.error("Strategy update failed: DataFrame index is not a DatetimeIndex.")
            return empty_results
        if not df.index.is_monotonic_increasing:
             self.lg.warning("Input DataFrame index is not monotonic. Sorting index...")
             df.sort_index(inplace=True)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            self.lg.error(f"Strategy update failed: Missing one or more required columns: {required_cols}.")
            return empty_results
        if len(df) < self.min_data_len:
            self.lg.warning(f"Strategy update: Insufficient data ({len(df)} candles < ~{self.min_data_len} required). Results may be inaccurate or incomplete, especially for initial candles.")
            # Proceed but warn

        self.lg.debug(f"Starting strategy analysis on {len(df)} candles (minimum required: {self.min_data_len}).")

        # --- Convert to Float for TA Libraries ---
        # pandas_ta and numpy work best with floats.
        try:
            df_float = pd.DataFrame(index=df.index)
            for col in required_cols:
                # Convert Decimal/Object to float, coercing errors to NaN
                df_float[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where essential float columns became NaN during conversion
            initial_float_len = len(df_float)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if len(df_float) < initial_float_len:
                 self.lg.debug(f"Dropped {initial_float_len - len(df_float)} rows due to NaN in OHLC after float conversion.")
            if df_float.empty:
                self.lg.error("Strategy update failed: DataFrame became empty after float conversion and NaN drop.")
                return empty_results
        except Exception as e:
            self.lg.error(f"Strategy update failed: Error during conversion to float: {e}", exc_info=True)
            return empty_results

        # --- Indicator Calculations (using df_float) ---
        try:
            self.lg.debug("Calculating indicators (ATR, EMAs, Bands, Volume Norm, Pivots)...")
            # Average True Range (ATR)
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # Volumatic Trend EMAs
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length) # EMA(SWMA(close))
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan) # Standard EMA(close)

            # Determine Trend Direction (ema2 crosses ema1 of the *previous* bar)
            # Ensure comparison happens only when both values are valid
            valid_comparison = df_float['ema2'].notna() & df_float['ema1'].shift(1).notna()
            trend_up_series = pd.Series(np.nan, index=df_float.index, dtype=object) # Use object to store bool/NaN
            trend_up_series[valid_comparison] = df_float['ema2'] > df_float['ema1'].shift(1)
            # Fill initial NaNs with False (assuming no trend initially) before comparison needed
            # trend_up_series.fillna(False, inplace=True) # Or fill later after change calc

            # Identify Trend Changes (where trend differs from previous bar's trend)
            # Compare current boolean trend to previous boolean trend
            trend_changed_series = (trend_up_series != trend_up_series.shift(1)) & \
                                   trend_up_series.notna() & trend_up_series.shift(1).notna()
            df_float['trend_changed'] = trend_changed_series.fillna(False).astype(bool) # Store as pure boolean

            # Store the boolean trend (after potentially filling NaNs if desired, but NaN handling below is safer)
            df_float['trend_up'] = trend_up_series # Keep potential NaNs for band calculation logic

            # Capture EMA1 and ATR values at the exact point the trend changed
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these values to get the relevant EMA/ATR for the current trend segment
            # This ensures bands are based on the values when the current trend started
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Volumatic Trend Bands only where underlying values are valid
            atr_multiplier_float = float(self.vt_atr_multiplier)
            valid_band_calc = df_float['ema1_for_bands'].notna() & df_float['atr_for_bands'].notna()
            df_float['upper_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)
            df_float['lower_band'] = np.where(valid_band_calc, df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float), np.nan)


            # Volume Normalization
            volume_numeric = df_float['volume'].fillna(0.0) # Use float volume, handle NaNs
            # Use a reasonable min_periods for the rolling max to avoid NaN at the start
            min_periods_vol = max(1, min(self.vt_vol_ema_length // 2, len(df_float))) # Dynamic min periods
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=min_periods_vol).max().fillna(method='bfill') # Backfill early NaNs
            # Calculate normalized volume (0-100 range, potentially > 100)
            # Use small epsilon to avoid division by zero/near-zero
            epsilon = 1e-18
            df_float['vol_norm'] = np.where(df_float['vol_max'] > epsilon,
                                            (volume_numeric / df_float['vol_max'] * 100.0),
                                            0.0) # Assign 0 if max volume is zero or negative
            # Handle potential NaNs and clip unreasonable values (e.g., > 200%)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)

            # Pivot High/Low Calculation
            # Select source series based on config ('Wicks' or 'Body')
            if self.ob_source == "Wicks":
                high_series = df_float['high']
                low_series = df_float['low']
            else: # Use candle body ("Body")
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)
            # Find pivots using the helper function
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, is_high=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, is_high=False)

            self.lg.debug("Indicator calculations complete (float).")

        except Exception as e:
            self.lg.error(f"Strategy update failed: Error during indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy Calculated Float Results back to Original Decimal DataFrame ---
        try:
            self.lg.debug("Converting calculated indicators back to Decimal format...")
            indicator_cols_numeric = ['atr', 'ema1', 'ema2', 'upper_band', 'lower_band', 'vol_norm']
            indicator_cols_bool = ['trend_up', 'trend_changed', 'is_ph', 'is_pl']

            for col in indicator_cols_numeric:
                if col in df_float.columns:
                    # Reindex to align with original Decimal DataFrame's index (handles NaNs introduced)
                    source_series = df_float[col].reindex(df.index)
                    # Convert numeric (float) back to Decimal, preserving NaNs
                    df[col] = source_series.apply(
                        lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                    )
            for col in indicator_cols_bool:
                 if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    # Convert potential object/NaN bool series to proper bool, filling NaNs as False
                    if col == 'trend_up':
                         # Special handling for trend_up: keep NaN if calculation failed, else bool
                         df[col] = source_series.apply(lambda x: bool(x) if pd.notna(x) else None)
                    else:
                         df[col] = source_series.fillna(False).astype(bool)

        except Exception as e:
            self.lg.error(f"Strategy update failed: Error converting calculated indicators back to Decimal: {e}", exc_info=True)
            # Continue if possible, OB management might still work partially

        # --- Clean Final Decimal DataFrame ---
        # Drop rows where essential indicators might be NaN (e.g., at the start of the series)
        # Trend_up can be None, so don't drop based on it here. ATR, Bands are crucial.
        initial_len_final = len(df)
        required_indicator_cols = ['close', 'atr', 'upper_band', 'lower_band', 'is_ph', 'is_pl']
        # Also drop if trend_up could not be determined (NaN) for a valid candle (should be rare)
        df.dropna(subset=required_indicator_cols, inplace=True)
        # df = df[df['trend_up'].notna()] # Optional: Be strict about having a trend determination

        rows_dropped_final = initial_len_final - len(df)
        if rows_dropped_final > 0:
            self.lg.debug(f"Dropped {rows_dropped_final} rows from final DataFrame due to missing essential indicators (likely at start).")

        if df.empty:
            self.lg.warning("Strategy update: DataFrame became empty after final indicator cleaning. Cannot process Order Blocks.")
            empty_results['dataframe'] = df # Return the empty df
            return empty_results

        self.lg.debug("Indicators finalized in Decimal DataFrame. Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_ob_count = 0
            violated_ob_count = 0
            last_candle_idx = df.index[-1]
            active_bull_boxes_before = [b for b in self.bull_boxes if b['active']]
            active_bear_boxes_before = [b for b in self.bear_boxes if b['active']]

            # 1. Identify New Order Blocks from Pivots on *all* historical candles in the df
            # This rebuilds OBs based on the current lookback settings each time
            # Consider optimizing later if performance is an issue (only check recent pivots)
            new_bull_candidates = []
            new_bear_candidates = []
            for timestamp, candle in df.iterrows():
                try:
                    # Create Bearish OB from Pivot High
                    if candle.get('is_ph'):
                        # Check if an OB from this exact candle already exists in the *current* list (prevent duplicates per run)
                        # Or if it existed before but was violated (don't recreate violated)
                        if not any(ob['left_idx'] == timestamp and ob['type'] == 'bear' for ob in self.bear_boxes + new_bear_candidates):
                            ob_top = candle['high'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                            ob_bottom = candle['open'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                            # Ensure valid Decimal values and top > bottom, check for NaN
                            if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and \
                               pd.notna(ob_top) and pd.notna(ob_bottom) and ob_top > ob_bottom:
                                ob_id = f"B_{timestamp.strftime('%y%m%d%H%M%S')}" # Unique ID
                                new_bear_candidates.append(OrderBlock(
                                    id=ob_id, type='bear', left_idx=timestamp, right_idx=timestamp, # Initial right_idx
                                    top=ob_top, bottom=ob_bottom, active=True, violated=False
                                ))
                                new_ob_count += 1
                            #else: self.lg.debug(f"Skipped Bear OB at {timestamp}: Invalid top/bottom ({ob_top}/{ob_bottom}).")

                    # Create Bullish OB from Pivot Low
                    if candle.get('is_pl'):
                        if not any(ob['left_idx'] == timestamp and ob['type'] == 'bull' for ob in self.bull_boxes + new_bull_candidates):
                            ob_top = candle['open'] if self.ob_source == "Wicks" else max(candle['open'], candle['close'])
                            ob_bottom = candle['low'] if self.ob_source == "Wicks" else min(candle['open'], candle['close'])
                            if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and \
                               pd.notna(ob_top) and pd.notna(ob_bottom) and ob_top > ob_bottom:
                                ob_id = f"L_{timestamp.strftime('%y%m%d%H%M%S')}"
                                new_bull_candidates.append(OrderBlock(
                                    id=ob_id, type='bull', left_idx=timestamp, right_idx=timestamp,
                                    top=ob_top, bottom=ob_bottom, active=True, violated=False
                                ))
                                new_ob_count += 1
                            #else: self.lg.debug(f"Skipped Bull OB at {timestamp}: Invalid top/bottom ({ob_top}/{ob_bottom}).")

                except Exception as pivot_proc_err:
                    self.lg.warning(f"Error processing potential pivot at {timestamp}: {pivot_proc_err}", exc_info=False) # Less verbose exc_info

            # Add new candidates to the main lists
            self.bull_boxes.extend(new_bull_candidates)
            self.bear_boxes.extend(new_bear_candidates)
            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new potential Order Blocks.")

            # 2. Manage Existing and New Order Blocks (Check Violations, Extend) across relevant history
            # Iterate through candles *after* the OB was formed up to the latest
            active_bull_boxes_after_add = [b for b in self.bull_boxes if b['active']]
            active_bear_boxes_after_add = [b for b in self.bear_boxes if b['active']]

            for box in active_bull_boxes_after_add:
                 # Find candles after the box was formed
                 relevant_candles = df[df.index > box['left_idx']]
                 for ts, candle in relevant_candles.iterrows():
                      close_price = candle.get('close')
                      if isinstance(close_price, Decimal) and pd.notna(close_price):
                           # Violation check: Close below the bottom of a bull OB
                           if close_price < box['bottom']:
                                box['active'] = False
                                box['violated'] = True
                                box['right_idx'] = ts # Mark violation time
                                violated_ob_count +=1
                                self.lg.debug(f"Bullish OB {box['id']} VIOLATED at {ts.strftime('%H:%M')} by close {close_price.normalize()} < {box['bottom'].normalize()}")
                                break # Stop checking this box once violated
                           else:
                                # Update right_idx if still active
                                box['right_idx'] = ts
                      # else: lg.warning(f"Invalid close price at {ts} for OB check.")

            for box in active_bear_boxes_after_add:
                 relevant_candles = df[df.index > box['left_idx']]
                 for ts, candle in relevant_candles.iterrows():
                      close_price = candle.get('close')
                      if isinstance(close_price, Decimal) and pd.notna(close_price):
                           # Violation check: Close above the top of a bear OB
                           if close_price > box['top']:
                                box['active'] = False
                                box['violated'] = True
                                box['right_idx'] = ts
                                violated_ob_count += 1
                                self.lg.debug(f"Bearish OB {box['id']} VIOLATED at {ts.strftime('%H:%M')} by close {close_price.normalize()} > {box['top'].normalize()}")
                                break
                           else:
                                box['right_idx'] = ts
                      # else: lg.warning(f"Invalid close price at {ts} for OB check.")

            # Extend active, non-violated boxes to the last candle if enabled
            if self.ob_extend:
                for box in self.bull_boxes + self.bear_boxes:
                     if box['active'] and not box['violated']:
                          box['right_idx'] = last_candle_idx

            if violated_ob_count > 0:
                 self.lg.debug(f"Processed violations for {violated_ob_count} Order Blocks.")

            # 3. Prune Order Blocks (Keep only the most recent 'ob_max_boxes' *active* ones)
            # Filter only active boxes, sort by creation time (descending), take the top N
            self.bull_boxes = sorted([b for b in self.bull_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted([b for b in self.bear_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.lg.debug(f"Pruned Order Blocks. Kept Active: Bulls={len(self.bull_boxes)}, Bears={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"Strategy update failed: Error during Order Block processing: {e}", exc_info=True)
            # Continue to return results, but OBs might be inaccurate/incomplete

        # --- Prepare Final StrategyAnalysisResults ---
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper to safely extract Decimal values from the last candle
        def safe_decimal_from_candle(candle_data: Optional[pd.Series], col_name: str, positive_only: bool = False) -> Optional[Decimal]:
            """Safely extracts a Decimal value from a Series, returns None if invalid/NaN."""
            if candle_data is None: return None
            value = candle_data.get(col_name)
            if isinstance(value, Decimal) and pd.notna(value): # Check isnan before potential comparison
                 # Check positivity constraint if required
                 if not positive_only or value > Decimal('0'):
                     return value
            # lg.debug(f"safe_decimal: Value for '{col_name}' is invalid or failed check: {value}")
            return None

        # Helper to safely extract Boolean values (handling None/NaN)
        def safe_bool_from_candle(candle_data: Optional[pd.Series], col_name: str) -> Optional[bool]:
            """Safely extracts a boolean, returns None if NaN/missing."""
            if candle_data is None: return None
            value = candle_data.get(col_name)
            if pd.isna(value) or value is None:
                return None
            return bool(value)


        # Construct the results dictionary
        final_dataframe = df # Return the fully processed DataFrame with Decimals
        last_close_val = safe_decimal_from_candle(last_candle_final, 'close') or Decimal('0')
        current_trend_val = safe_bool_from_candle(last_candle_final, 'trend_up') # Can be None if undetermined
        trend_changed_val = bool(safe_bool_from_candle(last_candle_final, 'trend_changed')) # Default False if None
        vol_norm_dec = safe_decimal_from_candle(last_candle_final, 'vol_norm')
        vol_norm_int_val = int(vol_norm_dec) if vol_norm_dec is not None else None
        atr_val = safe_decimal_from_candle(last_candle_final, 'atr', positive_only=True) # ATR must be positive
        upper_band_val = safe_decimal_from_candle(last_candle_final, 'upper_band')
        lower_band_val = safe_decimal_from_candle(last_candle_final, 'lower_band')

        analysis_results = StrategyAnalysisResults(
            dataframe=final_dataframe,
            last_close=last_close_val,
            current_trend_up=current_trend_val,
            trend_just_changed=trend_changed_val,
            active_bull_boxes=self.bull_boxes, # Return the pruned list of active OBs
            active_bear_boxes=self.bear_boxes,
            vol_norm_int=vol_norm_int_val,
            atr=atr_val,
            upper_band=upper_band_val,
            lower_band=lower_band_val
        )

        # Log summary of the final results for the *last* candle
        trend_str = f"{NEON_GREEN}UP{RESET}" if analysis_results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if analysis_results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}Undetermined{RESET}"
        atr_str = f"{analysis_results['atr'].normalize()}" if analysis_results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"

        self.lg.debug(f"--- Strategy Analysis Results ({time_str}) ---")
        self.lg.debug(f"  Last Close: {analysis_results['last_close'].normalize()}")
        self.lg.debug(f"  Trend: {trend_str} (Changed on this candle: {analysis_results['trend_just_changed']})")
        self.lg.debug(f"  ATR: {atr_str}")
        self.lg.debug(f"  Volume Norm (%): {analysis_results['vol_norm_int']}")
        self.lg.debug(f"  Bands (Upper/Lower): {analysis_results['upper_band'].normalize() if analysis_results['upper_band'] else 'N/A'} / {analysis_results['lower_band'].normalize() if analysis_results['lower_band'] else 'N/A'}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(analysis_results['active_bull_boxes'])} / {len(analysis_results['active_bear_boxes'])}")
        # Optionally log the details of the active OBs at DEBUG level
        # for ob in analysis_results['active_bull_boxes']: self.lg.debug(f"    Bull OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
        # for ob in analysis_results['active_bear_boxes']: self.lg.debug(f"    Bear OB: {ob['id']} [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
        self.lg.debug(f"---------------------------------------------")

        return analysis_results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """
    Generates trading signals based on strategy analysis and position state.

    Responsibilities:
    - Evaluates `StrategyAnalysisResults` against entry/exit rules.
    - Considers the current open position (if any).
    - Generates signals: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD".
    - Calculates initial Stop Loss (SL) and Take Profit (TP) levels for new entries.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Signal Generator with parameters from the config.

        Args:
            config: The main configuration dictionary.
            logger: The logger instance.

        Raises:
            ValueError: If critical configuration parameters are invalid.
        """
        self.config = config
        self.logger = logger
        self.lg = logger # Alias
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        try:
            # Load parameters used for signal generation and SL/TP calculation
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))

            # Basic validation
            if not (self.ob_entry_proximity_factor >= 1): raise ValueError("ob_entry_proximity_factor must be >= 1.0")
            if not (self.ob_exit_proximity_factor >= 1): raise ValueError("ob_exit_proximity_factor must be >= 1.0")
            if not (self.initial_tp_atr_multiple >= 0): raise ValueError("initial_take_profit_atr_multiple must be >= 0")
            if not (self.initial_sl_atr_multiple > 0): raise ValueError("initial_stop_loss_atr_multiple must be > 0")

            self.lg.info(f"{NEON_CYAN}--- Initializing Signal Generator ---{RESET}")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor.normalize()}")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor.normalize()}")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")
            self.lg.info(f"-----------------------------------")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.error(f"{NEON_RED}FATAL: Error initializing SignalGenerator parameters from config: {e}.{RESET}", exc_info=True)
             raise ValueError(f"SignalGenerator initialization failed: {e}") from e

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[PositionInfo]) -> str:
        """
        Determines the trading signal based on strategy analysis and current position.

        Logic Flow:
        1. Validate inputs.
        2. If position exists, check for Exit conditions (trend flip, OB proximity).
        3. If no position exists, check for Entry conditions (trend alignment, OB proximity).
        4. Default to HOLD if no entry/exit conditions met.

        Args:
            analysis_results: The results from `VolumaticOBStrategy.update()`.
            open_position: Standardized `PositionInfo` dict, or None if no position.

        Returns:
            The generated signal string: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "HOLD".
        """
        lg = self.logger

        # --- Validate Input ---
        # Check for essential valid results from analysis
        if not analysis_results or \
           analysis_results['current_trend_up'] is None or \
           analysis_results['last_close'] <= 0 or \
           analysis_results['atr'] is None or analysis_results['atr'] <= 0:
            lg.warning(f"{NEON_YELLOW}Signal Generation: Invalid or incomplete strategy analysis results. Defaulting to HOLD.{RESET}")
            lg.debug(f"  Problematic Analysis: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
            return "HOLD"

        # Extract key values for easier access
        last_close = analysis_results['last_close']
        trend_is_up = analysis_results['current_trend_up'] # This is boolean or None
        trend_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        position_side = open_position['side'] if open_position else None

        signal: str = "HOLD" # Default signal

        lg.debug(f"--- Signal Generation Check ---")
        trend_log = 'UP' if trend_is_up else 'DOWN' if trend_is_up is False else 'None'
        lg.debug(f"  Input: Close={last_close.normalize()}, Trend={trend_log}, TrendChanged={trend_changed}, Position={position_side or 'None'}")
        lg.debug(f"  Active OBs: Bull={len(active_bull_obs)}, Bear={len(active_bear_obs)}")

        # --- 1. Check Exit Conditions (if position exists) ---
        if position_side == 'long':
            # Exit Long if trend flips down *on the last candle*
            if trend_is_up is False and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped to DOWN on last candle.{RESET}")
            # Exit Long if price nears a bearish OB (use exit proximity factor)
            elif active_bear_obs: # Check only if not already exiting
                try:
                    # Find the closest bear OB (based on top edge proximity)
                    closest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - last_close))
                    # Exit threshold: Price >= OB Top / Exit Proximity Factor (safer division)
                    # Note: Original logic used multiplication, division seems more intuitive for exit threshold near resistance
                    exit_threshold = closest_bear_ob['top'] / self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else closest_bear_ob['top']
                    if last_close >= exit_threshold:
                        signal = "EXIT_LONG"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price {last_close.normalize()} >= Bear OB exit threshold {exit_threshold.normalize()} (OB ID: {closest_bear_ob['id']}, Top: {closest_bear_ob['top'].normalize()}){RESET}")
                except (ZeroDivisionError, InvalidOperation, Exception) as e:
                    lg.warning(f"Error during Bearish OB exit check for long position: {e}")

        elif position_side == 'short':
            # Exit Short if trend flips up *on the last candle*
            if trend_is_up is True and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped to UP on last candle.{RESET}")
            # Exit Short if price nears a bullish OB (use exit proximity factor)
            elif active_bull_obs: # Check only if not already exiting
                try:
                    # Find the closest bull OB (based on bottom edge proximity)
                    closest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - last_close))
                    # Exit threshold: Price <= OB Bottom * Exit Proximity Factor
                    # Note: Original logic used division, multiplication seems more intuitive for exit threshold near support
                    exit_threshold = closest_bull_ob['bottom'] * self.ob_exit_proximity_factor
                    if last_close <= exit_threshold:
                        signal = "EXIT_SHORT"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price {last_close.normalize()} <= Bull OB exit threshold {exit_threshold.normalize()} (OB ID: {closest_bull_ob['id']}, Bottom: {closest_bull_ob['bottom'].normalize()}){RESET}")
                except Exception as e:
                    lg.warning(f"Error during Bullish OB exit check for short position: {e}")

        # If an exit signal was generated, return it immediately
        if signal != "HOLD":
            lg.debug(f"--- Signal Result: {signal} (Exit Condition Met) ---")
            return signal

        # --- 2. Check Entry Conditions (if NO position exists) ---
        if position_side is None:
            # Check for BUY signal: Trend is UP and price is within a Bullish OB's proximity
            if trend_is_up is True and active_bull_obs:
                for ob in active_bull_obs:
                    try:
                        # Entry zone: OB Bottom <= Price <= OB Top * Entry Proximity Factor
                        entry_zone_bottom = ob['bottom']
                        entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                        if entry_zone_bottom <= last_close <= entry_zone_top:
                            signal = "BUY"
                            lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price {last_close.normalize()} within Bull OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                            break # Take the first valid entry signal found
                    except (InvalidOperation, Exception) as e:
                         lg.warning(f"Error checking Bull OB {ob.get('id')} for entry: {e}")


            # Check for SELL signal: Trend is DOWN and price is within a Bearish OB's proximity
            elif trend_is_up is False and active_bear_obs:
                for ob in active_bear_obs:
                    try:
                        # Entry zone: OB Bottom / Entry Proximity Factor <= Price <= OB Top
                        # Safe division check
                        entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']
                        entry_zone_top = ob['top']
                        if entry_zone_bottom <= last_close <= entry_zone_top:
                            signal = "SELL"
                            lg.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price {last_close.normalize()} within Bear OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                            break # Take the first valid entry signal found
                    except (ZeroDivisionError, InvalidOperation, Exception) as e:
                         lg.warning(f"Error checking Bear OB {ob.get('id')} for entry: {e}")


        # --- 3. Default to HOLD ---
        if signal == "HOLD":
            lg.debug(f"Signal: HOLD - No valid entry or exit conditions met.")

        lg.debug(f"--- Signal Result: {signal} ---")
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: MarketInfo, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels for a new entry.

        Uses entry price, current ATR, configured multipliers, and market price precision.
        Ensures SL/TP levels are strictly beyond the entry price after formatting.

        Args:
            entry_price: The estimated or actual entry price (positive Decimal).
            signal: The entry signal ("BUY" or "SELL").
            atr: The current Average True Range value (positive Decimal).
            market_info: The standardized MarketInfo dictionary.
            exchange: The initialized ccxt.Exchange object (for price formatting).

        Returns:
            A tuple containing:
                - Calculated Take Profit price (Decimal), or None if disabled or calculation fails.
                - Calculated Stop Loss price (Decimal), or None if calculation fails critically.
            Returns (None, None) if inputs are invalid or critical errors occur.
        """
        lg = self.logger
        lg.debug(f"Calculating Initial TP/SL for {signal} signal at entry {entry_price.normalize()} with ATR {atr.normalize()}")

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            lg.error(f"TP/SL Calc Failed: Invalid signal '{signal}'.")
            return None, None
        if not isinstance(entry_price, Decimal) or entry_price <= 0:
             lg.error(f"TP/SL Calc Failed: Entry price ({entry_price}) must be a positive Decimal.")
             return None, None
        if not isinstance(atr, Decimal) or atr <= 0:
            lg.error(f"TP/SL Calc Failed: ATR ({atr}) must be a positive Decimal.")
            return None, None
        try:
            price_tick = market_info['price_precision_step_decimal']
            if price_tick is None or price_tick <= 0: raise ValueError("Invalid price tick size")
        except (KeyError, ValueError, TypeError) as e:
            lg.error(f"TP/SL Calc Failed: Could not get valid price precision from market info: {e}")
            return None, None

        # --- Calculate Raw TP/SL ---
        try:
            tp_atr_multiple = self.initial_tp_atr_multiple # Already Decimal from init
            sl_atr_multiple = self.initial_sl_atr_multiple # Already Decimal from init

            # Calculate offsets
            tp_offset = atr * tp_atr_multiple
            sl_offset = atr * sl_atr_multiple

            # Calculate raw levels
            take_profit_raw: Optional[Decimal] = None
            if tp_atr_multiple > 0: # Only calculate TP if multiplier is positive
                 take_profit_raw = (entry_price + tp_offset) if signal == "BUY" else (entry_price - tp_offset)

            stop_loss_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)

            lg.debug(f"  Raw Levels: TP={take_profit_raw.normalize() if take_profit_raw else 'N/A'}, SL={stop_loss_raw.normalize()}")

            # --- Format Levels to Market Precision ---
            # Helper function to format and validate price level
            def format_level(price_decimal: Optional[Decimal], level_name: str) -> Optional[Decimal]:
                """Formats price to exchange precision, returns positive Decimal or None."""
                if price_decimal is None or price_decimal <= 0:
                    lg.debug(f"Calculated {level_name} is invalid or zero/negative ({price_decimal}).")
                    return None
                try:
                    # Use CCXT's price_to_precision for correct rounding/truncating
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal))
                    formatted_decimal = Decimal(formatted_str)
                    # Final check: ensure formatted price is still positive
                    if formatted_decimal > 0:
                         return formatted_decimal
                    else:
                         lg.warning(f"Formatted {level_name} ({formatted_str}) resulted in zero or negative value. Ignoring {level_name}.")
                         return None
                except Exception as e:
                    lg.error(f"Error formatting {level_name} value {price_decimal.normalize()} to exchange precision: {e}.")
                    return None

            # Format TP and SL
            take_profit_final = format_level(take_profit_raw, "Take Profit")
            stop_loss_final = format_level(stop_loss_raw, "Stop Loss")

            # --- Final Adjustments and Validation ---
            # Ensure SL is strictly beyond entry after formatting
            if stop_loss_final is not None:
                sl_invalid = (signal == "BUY" and stop_loss_final >= entry_price) or \
                             (signal == "SELL" and stop_loss_final <= entry_price)
                if sl_invalid:
                    lg.warning(f"Formatted {signal} Stop Loss {stop_loss_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Adjusting by one tick.")
                    # Adjust SL by one tick further away from entry
                    adjusted_sl_raw = (stop_loss_final - price_tick) if signal == "BUY" else (stop_loss_final + price_tick)
                    stop_loss_final = format_level(adjusted_sl_raw, "Adjusted Stop Loss") # Reformat after adjustment
                    if stop_loss_final is None or stop_loss_final <= 0:
                         lg.error(f"{NEON_RED}CRITICAL: Failed to calculate valid adjusted SL after initial SL was invalid.{RESET}")
                         return take_profit_final, None # Return None for SL is critical failure

            # Ensure TP is strictly beyond entry (if TP is enabled and calculated)
            if take_profit_final is not None:
                tp_invalid = (signal == "BUY" and take_profit_final <= entry_price) or \
                             (signal == "SELL" and take_profit_final >= entry_price)
                if tp_invalid:
                    lg.warning(f"Formatted {signal} Take Profit {take_profit_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Disabling TP for this entry.")
                    take_profit_final = None # Disable TP if it ends up on the wrong side

            # Log final calculated levels
            tp_log = take_profit_final.normalize() if take_profit_final else "None (Disabled or Calc Failed)"
            sl_log = stop_loss_final.normalize() if stop_loss_final else "None (Calc Failed!)"
            lg.info(f"  >>> Calculated Initial Levels: TP={tp_log}, SL={sl_log}")

            # Critical check: Ensure SL calculation was successful
            if stop_loss_final is None:
                lg.error(f"{NEON_RED}Stop Loss calculation failed critically. Cannot determine position size or place trade safely.{RESET}")
                return take_profit_final, None # Return None for SL

            return take_profit_final, stop_loss_final

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
            return None, None

# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: MarketInfo) -> None:
    """
    Performs one full cycle of analysis and trading logic for a single symbol.

    Steps:
    1. Logs cycle start and key configurations.
    2. Fetches and validates kline data, ensuring sufficient length.
    3. Runs the strategy analysis using `strategy_engine`.
    4. Validates analysis results (trend, close, ATR).
    5. Gets current market state (live price, open position).
    6. Generates a trading signal using `signal_generator`.
    7. Executes trading actions based on the signal and `enable_trading` flag:
        - **Entry:** If no position and BUY/SELL signal:
            - Checks balance.
            - Calculates initial SL/TP.
            - Sets leverage (if applicable).
            - Calculates position size.
            - Places market order.
            - Confirms position opens.
            - Sets initial protection (TSL or Fixed SL/TP).
        - **Exit:** If position exists and EXIT signal:
            - Places reduce-only market order to close.
        - **Management:** If position exists and HOLD signal:
            - Checks and applies Break-Even adjustment if conditions met.
            - Checks and attempts to set TSL if enabled but not active.
    8. Logs cycle end and duration.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol to analyze and trade.
        config: The main configuration dictionary.
        logger: The logger instance for this symbol's trading activity.
        strategy_engine: The initialized VolumaticOBStrategy instance.
        signal_generator: The initialized SignalGenerator instance.
        market_info: The standardized MarketInfo dictionary for the symbol.
    """
    lg = logger
    lg.info(f"\n{BRIGHT}---=== Cycle Start: Analyzing {symbol} ({config['interval']} TF) ===---{RESET}")
    cycle_start_time = time.monotonic()

    # Log key config settings for this cycle for easier debugging
    prot_cfg = config.get("protection", {})
    strat_cfg = config.get("strategy_params", {})
    lg.debug(f"Cycle Config: Trading={'ENABLED' if config.get('enable_trading') else 'DISABLED'}, Sandbox={config.get('use_sandbox')}, "
             f"Risk={config.get('risk_per_trade'):.2%}, Lev={config.get('leverage')}x, "
             f"TSL={'ON' if prot_cfg.get('enable_trailing_stop') else 'OFF'} (Act%={prot_cfg.get('trailing_stop_activation_percentage'):.3%}, CB%={prot_cfg.get('trailing_stop_callback_rate'):.3%}), "
             f"BE={'ON' if prot_cfg.get('enable_break_even') else 'OFF'} (TrigATR={prot_cfg.get('break_even_trigger_atr_multiple')}, Offset={prot_cfg.get('break_even_offset_ticks')} ticks), "
             f"InitSL Mult={prot_cfg.get('initial_stop_loss_atr_multiple')}, InitTP Mult={prot_cfg.get('initial_take_profit_atr_multiple')}, "
             f"OB Source={strat_cfg.get('ob_source')}")

    # --- 1. Fetch Kline Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        # This should not happen if config validation worked, but check defensively
        lg.critical(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe. Skipping cycle.")
        return

    # Determine how many klines to fetch
    min_required_data = strategy_engine.min_data_len
    fetch_limit_from_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    # Need at least what the strategy requires, but fetch user preference if it's more
    fetch_limit_needed = max(min_required_data, fetch_limit_from_config)
    # Actual request limit is capped by the API
    fetch_limit_request = min(fetch_limit_needed, BYBIT_API_KLINE_LIMIT)

    lg.info(f"Requesting {fetch_limit_request} klines for {symbol} ({ccxt_interval}). "
            f"(Strategy requires min: {min_required_data}, Config requests: {fetch_limit_from_config}, API limit: {BYBIT_API_KLINE_LIMIT})")
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit_request, logger=lg)
    fetched_count = len(klines_df)

    # --- 2. Validate Fetched Data ---
    if klines_df.empty or fetched_count < min_required_data:
        # Check if failure was due to hitting API limit but still not getting enough data
        hit_api_limit_but_insufficient = (fetch_limit_request == BYBIT_API_KLINE_LIMIT and
                                          fetched_count == BYBIT_API_KLINE_LIMIT and
                                          fetched_count < min_required_data)
        if hit_api_limit_but_insufficient:
            lg.error(f"{NEON_RED}CRITICAL DATA ISSUE:{RESET} Fetched maximum {fetched_count} klines allowed by API, "
                     f"but strategy requires {min_required_data}. Analysis will be inaccurate or fail. "
                     f"{NEON_YELLOW}ACTION REQUIRED: Reduce lookback periods in strategy config (e.g., vt_atr_period, vt_vol_ema_length)! Skipping cycle.{RESET}")
        elif klines_df.empty:
            lg.error(f"Failed to fetch any kline data for {symbol} {ccxt_interval}. Cannot proceed. Skipping cycle.")
        else: # Fetched some data, but not enough
            lg.error(f"Fetched only {fetched_count} klines, but strategy requires {min_required_data}. "
                     f"Analysis may be inaccurate or fail. Skipping cycle.")
        return # Cannot proceed without sufficient data

    # --- 3. Run Strategy Analysis ---
    lg.debug("Running strategy analysis engine...")
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis update failed unexpectedly: {analysis_err}{RESET}", exc_info=True)
        return # Stop cycle if analysis fails

    # Validate essential analysis results needed for signal generation and trading
    if not analysis_results or \
       analysis_results['current_trend_up'] is None or \
       analysis_results['last_close'] <= 0 or \
       analysis_results['atr'] is None or analysis_results['atr'] <= 0:
        lg.error(f"{NEON_RED}Strategy analysis did not produce valid essential results (missing/invalid trend, close price, or ATR). Skipping cycle.{RESET}")
        lg.debug(f"Problematic Analysis Results: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
        return
    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr'] # Guaranteed positive Decimal here
    lg.info(f"Strategy Analysis Complete: Trend={'UP' if analysis_results['current_trend_up'] else 'DOWN'}, "
            f"Last Close={latest_close.normalize()}, ATR={current_atr.normalize()}")

    # --- 4. Get Current Market State (Price & Position) ---
    lg.debug("Fetching current market price and checking for open positions...")
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    open_position: Optional[PositionInfo] = get_open_position(exchange, symbol, lg) # Returns standardized dict or None

    # Determine price to use for real-time checks (prefer live price, fallback to last close)
    # Ensure the price used is valid and positive
    price_for_checks: Optional[Decimal] = None
    if current_market_price and current_market_price > 0:
         price_for_checks = current_market_price
    elif latest_close > 0:
         price_for_checks = latest_close
         lg.debug(f"Using last kline close price ({latest_close.normalize()}) for checks as live price is unavailable/invalid.")
    else:
         lg.error(f"{NEON_RED}Cannot determine a valid current price (Live={current_market_price}, LastClose={latest_close}). Skipping cycle.{RESET}")
         return

    # --- 5. Generate Trading Signal ---
    lg.debug("Generating trading signal based on analysis and position...")
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation failed unexpectedly: {signal_err}{RESET}", exc_info=True)
        return # Stop cycle if signal generation fails

    lg.info(f"Generated Signal: {BRIGHT}{signal}{RESET}")

    # --- 6. Trading Logic Execution ---
    trading_enabled = config.get("enable_trading", False)

    # --- Scenario: Trading Disabled (Analysis/Logging Only) ---
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading is DISABLED.{RESET} Analysis complete. Signal was: {signal}")
        # Log potential action if trading were enabled
        if open_position is None and signal in ["BUY", "SELL"]:
            lg.info(f"  (Action if enabled: Would attempt to {signal} {symbol})")
        elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]:
            lg.info(f"  (Action if enabled: Would attempt to {signal} current {open_position['side']} position on {symbol})")
        elif open_position:
             lg.info(f"  (Action if enabled: Would manage existing {open_position['side']} position)")
        else: # HOLD signal, no position
            lg.info("  (Action if enabled: No entry/exit action indicated)")
        # End cycle here if trading disabled
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Analysis-Only Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return # Stop further processing

    # ======================================
    # --- Trading IS Enabled Below Here ---
    # ======================================
    lg.info(f"{BRIGHT}Trading is ENABLED. Processing signal '{signal}'...{RESET}")

    # --- Scenario 1: No Position -> Consider Entry ---
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** {signal} Signal & No Position: Initiating Entry Sequence... ***{RESET}")

        # Fetch current balance
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot fetch valid positive balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
            return

        # Calculate initial SL/TP based on latest close and current ATR
        # Use latest_close for initial calculation as it corresponds to the signal generation candle
        initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)

        if initial_sl_calc is None:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to calculate a valid initial Stop Loss. Cannot size position.{RESET}")
            return
        if initial_tp_calc is None:
            lg.info(f"{NEON_YELLOW}Initial Take Profit calculation failed or is disabled (TP Mult = 0). Proceeding without initial TP.{RESET}")

        # Set Leverage (if contract market and leverage > 0 in config)
        if market_info['is_contract']:
            leverage_to_set = int(config.get('leverage', 0))
            if leverage_to_set > 0:
                lg.info(f"Setting leverage to {leverage_to_set}x for {symbol}...")
                leverage_ok = set_leverage_ccxt(exchange, symbol, leverage_to_set, market_info, lg)
                if not leverage_ok:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set leverage to {leverage_to_set}x.{RESET}")
                    return
            else:
                lg.info("Leverage setting skipped (config leverage is 0). Using exchange default or previously set leverage.")

        # Calculate Position Size
        # Use latest_close for sizing calculation as it corresponds to the SL calculation price
        position_size = calculate_position_size(balance, config["risk_per_trade"], initial_sl_calc, latest_close, market_info, exchange, lg)
        if position_size is None or position_size <= 0:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Position sizing failed or resulted in zero/negative size ({position_size}).{RESET}")
            return

        # Place Market Order
        lg.warning(f"{BRIGHT}===> PLACING {signal} MARKET ORDER | Size: {position_size.normalize()} <==={RESET}")
        trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

        # --- Post-Trade Actions (Confirmation and Protection) ---
        if trade_order and trade_order.get('id'):
            confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
            lg.info(f"Order {trade_order['id']} placed. Waiting {confirm_delay}s for position confirmation...")
            time.sleep(confirm_delay)

            # Confirm position opened
            # Retry fetching position confirmation slightly
            confirmed_position: Optional[PositionInfo] = None
            for confirm_attempt in range(2): # Try twice with short delay
                 confirmed_position = get_open_position(exchange, symbol, lg)
                 if confirmed_position: break
                 if confirm_attempt == 0:
                      lg.warning("Position confirmation attempt 1 failed, retrying in 3s...")
                      time.sleep(3)

            if confirmed_position:
                try:
                    # Get actual entry price if available, fallback to latest close used for sizing
                    entry_price_actual: Optional[Decimal] = None
                    entry_price_actual_str = confirmed_position.get('entryPrice')
                    if entry_price_actual_str:
                         entry_price_actual = _safe_market_decimal(entry_price_actual_str, 'confirmed_position.entryPrice', allow_zero=False)

                    if entry_price_actual is None:
                        lg.warning(f"{NEON_YELLOW}Could not get actual entry price from confirmed position. Using original price ({latest_close.normalize()}) for protection calculation.{RESET}")
                        entry_price_actual = latest_close # Fallback

                    lg.info(f"{NEON_GREEN}Position Confirmed! Actual/Estimated Entry: ~{entry_price_actual.normalize()}{RESET}")

                    # Recalculate SL/TP based on actual/fallback entry price and current ATR for setting protection
                    # Use current_atr from the analysis results of the signal candle
                    prot_tp_calc, prot_sl_calc = signal_generator.calculate_initial_tp_sl(entry_price_actual, signal, current_atr, market_info, exchange)

                    if prot_sl_calc is None:
                        # This is critical - position is open but SL cannot be set
                        lg.error(f"{NEON_RED}{BRIGHT}CRITICAL ERROR: Position entered, but failed to recalculate SL based on entry price {entry_price_actual.normalize()}! POSITION IS UNPROTECTED! Manual SL required!{RESET}")
                    else:
                        # Set protection (TSL or Fixed SL/TP)
                        protection_set_success = False
                        if prot_cfg.get("enable_trailing_stop", True):
                            lg.info(f"Setting Trailing Stop Loss (TSL) based on entry {entry_price_actual.normalize()}...")
                            protection_set_success = set_trailing_stop_loss(
                                exchange, symbol, market_info, confirmed_position, config, lg,
                                take_profit_price=prot_tp_calc # Set TP alongside TSL if calculated
                            )
                        # Only set fixed SL/TP if TSL is disabled AND either SL or TP is valid
                        elif not prot_cfg.get("enable_trailing_stop", True) and (prot_sl_calc or prot_tp_calc):
                            lg.info(f"Setting Fixed Stop Loss / Take Profit based on entry {entry_price_actual.normalize()}...")
                            protection_set_success = _set_position_protection(
                                exchange, symbol, market_info, confirmed_position, lg,
                                stop_loss_price=prot_sl_calc,
                                take_profit_price=prot_tp_calc
                            )
                        else:
                            lg.info("No protection (TSL or Fixed SL/TP) enabled or calculated. Position entered without API protection.")
                            protection_set_success = True # Considered success as no action was needed

                        # Log final status
                        if protection_set_success:
                            lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY & INITIAL PROTECTION SETUP COMPLETE ({symbol} {signal}) ==={RESET}")
                        else:
                            lg.error(f"{NEON_RED}{BRIGHT}=== TRADE PLACED, BUT FAILED TO SET PROTECTION ({symbol} {signal}). MANUAL MONITORING REQUIRED! ==={RESET}")

                except Exception as post_trade_err:
                    lg.error(f"{NEON_RED}Error during post-trade setup (protection setting): {post_trade_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Position is confirmed open for {symbol}, but protection setup failed! Manual check recommended!{RESET}")
            else:
                # Order placed but position not found after delay and retries
                lg.error(f"{NEON_RED}Order {trade_order['id']} was placed, but FAILED TO CONFIRM open position for {symbol} after {confirm_delay + 3}s delay! Manual check required! Order might have failed to fill or API issue.{RESET}")
        else:
            # Order placement itself failed
            lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). No order was placed. ===")

    # --- Scenario 2: Existing Position -> Consider Exit or Manage ---
    elif open_position:
        pos_side = open_position['side'] # 'long' or 'short'
        pos_size_decimal = open_position.get('size_decimal', Decimal('0')) # Get parsed Decimal size
        lg.info(f"Existing {pos_side.upper()} position found (Size: {pos_size_decimal.normalize()}). Signal: {signal}")

        # Check if the signal triggers an exit
        exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                         (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_triggered:
            # --- Handle Exit Signal ---
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Received! Closing {pos_side} position... ***{RESET}")
            try:
                # Ensure size is positive for the closing order amount
                size_to_close = abs(pos_size_decimal)
                if size_to_close <= 0:
                    lg.warning(f"Attempting to close {pos_side} position, but parsed size is {pos_size_decimal}. Position might already be closed or data is stale. No close order placed.")
                    return # No action needed if size is zero/negative

                lg.info(f"===> Placing {signal} MARKET Order (Reduce Only) | Size: {size_to_close.normalize()} <===")
                # Pass the original EXIT signal to place_trade for logging clarity
                close_order = place_trade(exchange, symbol, signal, size_to_close, market_info, lg, reduce_only=True)

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) placed successfully for {symbol}.{RESET}")
                    # Optional: Wait and confirm position is actually closed here later
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual intervention may be required!{RESET}")
            except Exception as close_err:
                lg.error(f"{NEON_RED}Error encountered while trying to close {pos_side} position for {symbol}: {close_err}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual closing of position {symbol} may be needed!{RESET}")

        else: # Signal allows holding position
            # --- Handle Position Management (Break-Even, TSL Recovery) ---
            lg.debug(f"Signal ({signal}) allows holding {pos_side} position. Performing position management checks...")

            # Extract current protection levels and entry price safely
            try:
                tsl_dist_str = open_position.get('trailingStopLoss')
                tsl_active = tsl_dist_str and Decimal(tsl_dist_str) > 0
            except (KeyError, ValueError, InvalidOperation, TypeError): tsl_active = False

            current_sl: Optional[Decimal] = _safe_market_decimal(open_position.get('stopLossPrice'), 'pos.stopLossPrice', allow_zero=False)
            current_tp: Optional[Decimal] = _safe_market_decimal(open_position.get('takeProfitPrice'), 'pos.takeProfitPrice', allow_zero=False)
            entry_price: Optional[Decimal] = _safe_market_decimal(open_position.get('entryPrice'), 'pos.entryPrice', allow_zero=False)

            # --- Break-Even (BE) Logic ---
            be_enabled = prot_cfg.get("enable_break_even", True)
            # Check BE only if enabled, TSL is not active, and we have needed data (entry, atr, current price)
            if be_enabled and not tsl_active and entry_price and price_for_checks:
                lg.debug(f"Checking Break-Even condition for {pos_side} position...")
                lg.debug(f"  BE Check Inputs: Entry={entry_price.normalize()}, CurrentPrice={price_for_checks.normalize()}, ATR={current_atr.normalize()}, CurrentSL={current_sl.normalize() if current_sl else 'None'}")
                try:
                    be_trigger_atr_multiple = Decimal(str(prot_cfg["break_even_trigger_atr_multiple"]))
                    be_offset_ticks = int(prot_cfg["break_even_offset_ticks"])
                    price_tick = market_info['price_precision_step_decimal']
                    if price_tick is None: raise ValueError("Missing price tick for BE calc")

                    if not (be_trigger_atr_multiple > 0 and be_offset_ticks >= 0):
                         raise ValueError("Invalid BE config parameters (trigger multiple > 0, offset ticks >= 0).")

                    # Calculate profit in terms of ATR multiples
                    profit_in_price = (price_for_checks - entry_price) if pos_side == 'long' else (entry_price - price_for_checks)
                    # Avoid division by zero if ATR is somehow zero (should be caught earlier)
                    profit_in_atr = profit_in_price / current_atr if current_atr > 0 else Decimal(0)
                    lg.debug(f"  Profit (Price): {profit_in_price.normalize()}, Profit (ATR multiples): {profit_in_atr:.3f}")
                    lg.debug(f"  BE Trigger ATR Multiple: {be_trigger_atr_multiple.normalize()}")

                    # Check if profit target is reached
                    if profit_in_atr >= be_trigger_atr_multiple:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}Break-Even profit target REACHED! (Profit ATRs >= Trigger ATRs){RESET}")

                        # Calculate the BE Stop Loss level (Entry +/- Offset)
                        be_offset_value = price_tick * Decimal(str(be_offset_ticks))
                        be_stop_loss_raw = (entry_price + be_offset_value) if pos_side == 'long' else (entry_price - be_offset_value)

                        # Quantize BE SL to nearest tick (away from current price / towards profit)
                        if pos_side == 'long':
                            # Round up for long BE SL to lock in more profit/reduce risk further
                            be_stop_loss = (be_stop_loss_raw / price_tick).quantize(Decimal('1'), ROUND_UP) * price_tick
                        else: # Short
                            # Round down for short BE SL
                            be_stop_loss = (be_stop_loss_raw / price_tick).quantize(Decimal('1'), ROUND_DOWN) * price_tick

                        # Validate calculated BE SL
                        if be_stop_loss and be_stop_loss > 0 and \
                           ((pos_side == 'long' and be_stop_loss < price_for_checks) or \
                            (pos_side == 'short' and be_stop_loss > price_for_checks)): # Ensure SL is beyond current price
                            lg.debug(f"  Calculated BE Stop Loss level: {be_stop_loss.normalize()}")

                            # Check if the new BE SL is an improvement over the current SL
                            update_sl_needed = False
                            if current_sl is None:
                                update_sl_needed = True
                                lg.info("  Current SL is not set. Applying BE SL.")
                            elif (pos_side == 'long' and be_stop_loss > current_sl) or \
                                 (pos_side == 'short' and be_stop_loss < current_sl):
                                update_sl_needed = True
                                lg.info(f"  New BE SL ({be_stop_loss.normalize()}) is improvement over current SL ({current_sl.normalize()}). Applying update.")
                            else:
                                lg.debug(f"  Current SL ({current_sl.normalize()}) is already at or better than calculated BE SL ({be_stop_loss.normalize()}). No BE update needed.")

                            # If update is needed, call the protection function
                            if update_sl_needed:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even at {be_stop_loss.normalize()} ***{RESET}")
                                # Pass current TP to avoid accidentally clearing it
                                # Pass None for TSL params as we are setting a fixed SL
                                if _set_position_protection(exchange, symbol, market_info, open_position, lg,
                                                            stop_loss_price=be_stop_loss,
                                                            take_profit_price=current_tp, # Preserve existing TP
                                                            trailing_stop_distance=None, # Ensure TSL is not set/cleared unintentionally
                                                            tsl_activation_price=None):
                                    lg.info(f"{NEON_GREEN}Break-Even Stop Loss set successfully.{RESET}")
                                else:
                                    lg.error(f"{NEON_RED}Failed to set Break-Even Stop Loss via API.{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Break-Even triggered, but calculated BE Stop Loss level ({be_stop_loss}) is invalid or not beyond current price ({price_for_checks}).{RESET}")
                    else:
                        lg.debug(f"Break-Even profit target not yet reached.")

                except (KeyError, ValueError, InvalidOperation, TypeError, ZeroDivisionError) as be_err:
                    lg.error(f"{NEON_RED}Error during Break-Even check: {be_err}{RESET}", exc_info=True)
            elif be_enabled:
                lg.debug(f"Break-Even check skipped: {'TSL is active' if tsl_active else 'Missing required data (entry price or current price)'}.")
            # else: lg.debug("Break-Even check skipped: Disabled in configuration.")


            # --- TSL Setup/Recovery Logic ---
            # If TSL is enabled in config but not detected as active on the position
            tsl_enabled = prot_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not tsl_active and entry_price:
                 lg.warning(f"{NEON_YELLOW}Trailing Stop Loss (TSL) is enabled in config, but not detected as active on the current {pos_side} position. Attempting TSL setup/recovery...{RESET}")
                 # Recalculate initial TP based on entry price and current ATR (in case it was missed or cleared)
                 # We don't need the SL from this calculation for TSL setup.
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(entry_price, pos_side.upper(), current_atr, market_info, exchange)

                 # Attempt to set the TSL (and potentially the recalculated TP)
                 if set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc):
                     lg.info(f"{NEON_GREEN}TSL setup/recovery successful for {symbol}.{RESET}")
                 else:
                     lg.error(f"{NEON_RED}TSL setup/recovery FAILED for {symbol}. Fixed SL might still be active if previously set.{RESET}")
            elif tsl_enabled:
                 lg.debug(f"TSL setup/recovery check skipped: {'TSL already active' if tsl_active else 'Missing required data (entry price)'}.")
            # else: lg.debug("TSL setup/recovery check skipped: Disabled in configuration.")

            # --- No Action Needed ---
            if not exit_triggered: # Only log if not exiting
                 lg.debug("Position management checks complete. Holding position.")


    # --- Scenario 3: No Position and HOLD Signal ---
    elif open_position is None and signal == "HOLD":
         lg.info("Signal is HOLD and no position is open. No trading action taken.")

    # --- Scenario 4: Unhandled State (Should not normally happen) ---
    else:
        lg.error(f"Unhandled state encountered: Signal='{signal}', Position Side='{open_position['side'] if open_position else None}'. No action defined.")


    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"{BRIGHT}---=== Cycle End: {symbol} (Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---{RESET}\n")

# --- Main Function ---
def main() 
