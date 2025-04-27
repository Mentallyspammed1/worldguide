# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.6: Applied Neon color scheme, enhanced type hinting, logging, docs, robustness.

# --- Core Libraries ---
import json
import logging
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict
from zoneinfo import ZoneInfo  # Requires tzdata package (pip install tzdata)

import ccxt  # Requires ccxt (pip install ccxt)

# --- Dependencies (Install via pip) ---
import numpy as np  # Requires numpy (pip install numpy)
import pandas as pd  # Requires pandas (pip install pandas)
import pandas_ta as ta  # Requires pandas_ta (pip install pandas_ta)
import requests  # Requires requests (pip install requests)
from colorama import Fore, Style, init  # Requires colorama (pip install colorama)
from dotenv import load_dotenv  # Requires python-dotenv (pip install python-dotenv)

# --- Initialize Environment and Settings ---
getcontext().prec = 28  # Set Decimal precision globally for high-accuracy calculations
init(autoreset=True)  # Initialize Colorama for console colors, resetting after each print
load_dotenv()  # Load environment variables from a .env file in the project root

# --- Constants ---
# API Credentials (Loaded securely from .env file)
API_KEY: str | None = os.getenv("BYBIT_API_KEY")
API_SECRET: str | None = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Critical error if API keys are missing
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file")

# Configuration and Logging Files/Directories
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
try:
    # Attempt to load user-specified timezone, fallback to UTC if tzdata is not installed or invalid
    # Example: "America/Chicago", "Europe/London", "Asia/Tokyo", "UTC"
    TIMEZONE_STR = os.getenv("TIMEZONE", "America/Chicago")  # Default timezone if not set in .env
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# API Interaction Settings
MAX_API_RETRIES: int = 3  # Maximum number of consecutive retries for failed API calls
RETRY_DELAY_SECONDS: int = 5  # Base delay (in seconds) between API retries (may increase exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8  # Wait time after placing an order before fetching position details
LOOP_DELAY_SECONDS: int = 15  # Default delay between trading cycles (can be overridden in config.json)
BYBIT_API_KLINE_LIMIT: int = 1000  # Maximum number of Klines Bybit V5 API returns per request

# Timeframes Mapping
# Bybit API expects specific strings for intervals
VALID_INTERVALS: list[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750  # Default number of klines to fetch if not specified or less than strategy needs
MAX_DF_LEN: int = 2000  # Internal limit to prevent excessive memory usage by the Pandas DataFrame

# Strategy Defaults (Used if values are missing, invalid, or out of range in config.json)
DEFAULT_VT_LENGTH: int = 40             # Volumatic Trend EMA/SWMA length
DEFAULT_VT_ATR_PERIOD: int = 200        # ATR period for Volumatic Trend bands
DEFAULT_VT_VOL_EMA_LENGTH: int = 950    # Volume Normalization EMA length (Adjusted: 1000 often > API limit)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0  # ATR multiplier for Volumatic Trend bands
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0  # Currently unused step ATR multiplier
DEFAULT_OB_SOURCE: str = "Wicks"        # Order Block source ("Wicks" or "Body")
DEFAULT_PH_LEFT: int = 10               # Pivot High lookback periods (left)
DEFAULT_PH_RIGHT: int = 10              # Pivot High lookback periods (right)
DEFAULT_PL_LEFT: int = 10               # Pivot Low lookback periods (left)
DEFAULT_PL_RIGHT: int = 10              # Pivot Low lookback periods (right)
DEFAULT_OB_EXTEND: bool = True          # Extend Order Block visuals to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50          # Max number of active Order Blocks to track/display

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")
QUOTE_CURRENCY: str = "USDT"  # Placeholder, will be updated by load_config()

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


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter that redacts sensitive API keys/secrets
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
    """Sets up a dedicated logger instance for a specific context (e.g., 'init', symbol).
    Configures both a console handler (with Neon colors and level filtering)
    and a rotating file handler (capturing DEBUG level and above).

    Args:
        name: The name for the logger (e.g., "init", "BTC/USDT"). Used for filtering
              and naming the log file.

    Returns:
        The configured logging.Logger instance.
    """
    safe_name = name.replace('/', '_').replace(':', '-')  # Sanitize name for filenames/logger keys
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger instance already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels; handlers below will filter output

    # --- File Handler ---
    # Logs DEBUG level and above to a rotating file (e.g., pyrmethus_bot_BTC_USDT.log)
    try:
        # Rotate log file when it reaches 10MB, keep 5 backup files
        fh = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        # Use a standard, detailed formatter for the file log (includes timestamp, level, name, line number)
        ff = SensitiveFormatter(
            "%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)  # Log everything from DEBUG upwards to the file
        logger.addHandler(fh)
    except Exception:
        # Use print for errors during logger setup itself, as logger might not be functional
        pass

    # --- Console Handler ---
    # Logs INFO level (or level from ENV) and above to the console with Neon colors
    try:
        sh = logging.StreamHandler()
        # Define color mapping for different log levels
        level_colors = {
            logging.DEBUG: NEON_CYAN + DIM,      # Dim Cyan for Debug
            logging.INFO: NEON_BLUE,             # Bright Cyan for Info
            logging.WARNING: NEON_YELLOW,        # Bright Yellow for Warning
            logging.ERROR: NEON_RED,             # Bright Red for Error
            logging.CRITICAL: NEON_RED + BRIGHT,  # Bright Red + Bold for Critical
        }

        # Custom formatter for console output with colors and timezone-aware timestamps
        class NeonConsoleFormatter(SensitiveFormatter):
            """Applies Neon color scheme and timezone to console log messages."""
            def format(self, record: logging.LogRecord) -> str:
                level_color = level_colors.get(record.levelno, NEON_BLUE)  # Default to Info color
                # Format: Time - Level - [LoggerName] - Message
                log_fmt = (
                    f"{NEON_BLUE}%(asctime)s{RESET} - "
                    f"{level_color}%(levelname)-8s{RESET} - "
                    f"{NEON_PURPLE}[%(name)s]{RESET} - "
                    f"%(message)s"
                )
                # Create a formatter instance with the defined format and date style
                formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
                # Ensure timestamps reflect the configured TIMEZONE
                formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
                # Apply sensitive data redaction before returning the final message
                return super().format(record)

        sh.setFormatter(NeonConsoleFormatter())
        # Get desired console log level from environment variable (e.g., DEBUG, INFO, WARNING), default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)  # Fallback to INFO if invalid level provided
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception:
        pass

    logger.propagate = False  # Prevent log messages from bubbling up to the root logger
    return logger


# Initialize the 'init' logger early for messages during startup and configuration loading
init_logger = setup_logger("init")


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively checks if all keys from `default_config` exist in `config`.
    If a key is missing, it's added to `config` with the default value.
    Logs any keys that were added.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: Used internally for tracking nested key paths for logging.

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
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default: {repr(default_value)}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # If both default and loaded values are dicts, recurse into nested dict
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                # If nested dict was changed, update the parent dict and mark as changed
                updated_config[key] = nested_config
                changed = True
    return updated_config, changed


def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from a JSON file.
    - Creates a default config file if it doesn't exist.
    - Validates loaded config against defaults, adding missing keys.
    - Performs type and range validation on critical numeric parameters.
    - Saves the updated config back to the file if changes were made (missing keys or corrections).
    - Returns the validated (and potentially updated) configuration dictionary.
    - Falls back to default config if loading or validation fails critically.

    Args:
        filepath: The path to the configuration JSON file (e.g., "config.json").

    Returns:
        The loaded and validated configuration dictionary. Returns defaults if loading fails.
    """
    # Define the default configuration structure and values
    default_config = {
        # General Settings
        "interval": "5",                # Default timeframe (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS,  # Base delay for API retries
        "fetch_limit": DEFAULT_FETCH_LIMIT,  # Default klines to fetch per cycle
        "orderbook_limit": 25,          # (Currently Unused) Limit for order book fetching if implemented
        "enable_trading": False,        # Master switch for placing actual trades
        "use_sandbox": True,            # Use Bybit's testnet environment
        "risk_per_trade": 0.01,         # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                 # Default leverage for contract trading
        "max_concurrent_positions": 1,  # (Currently Unused) Max open positions allowed simultaneously
        "quote_currency": "USDT",       # The currency to calculate balance and risk against
        "loop_delay_seconds": LOOP_DELAY_SECONDS,  # Delay between trading cycles
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,  # Wait after order before checking position

        # Strategy Parameters (Volumatic Trend + OB)
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,  # Unused
            "ob_source": DEFAULT_OB_SOURCE,  # "Wicks" or "Body"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,  # Pivot High lookbacks
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,  # Pivot Low lookbacks
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005,  # Price must be <= OB top * factor (long) or >= OB bottom / factor (short)
            "ob_exit_proximity_factor": 1.001   # Exit if price >= Bear OB top * factor or <= Bull OB bottom / factor
        },
        # Position Protection Parameters
        "protection": {
             "enable_trailing_stop": True,      # Use trailing stop loss
             "trailing_stop_callback_rate": 0.005,  # TSL distance as % of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003,  # Activate TSL when price moves this % from entry (e.g., 0.003 = 0.3%)
             "enable_break_even": True,         # Move SL to entry + offset when profit target hit
             "break_even_trigger_atr_multiple": 1.0,  # Profit needed (in ATR multiples) to trigger BE
             "break_even_offset_ticks": 2,       # Move SL this many ticks beyond entry for BE
             "initial_stop_loss_atr_multiple": 1.8,  # Initial SL distance in ATR multiples
             "initial_take_profit_atr_multiple": 0.7  # Initial TP distance in ATR multiples (0 to disable)
        }
    }
    config_needs_saving: bool = False
    loaded_config: dict[str, Any] = {}

    # --- File Existence Check ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}")
            return default_config  # Return defaults immediately after creation
        except OSError as e:
            init_logger.error(f"{NEON_RED}Error creating default config file '{filepath}': {e}{RESET}")
            init_logger.warning("Using internal default configuration values.")
            return default_config  # Use internal defaults if file creation fails

    # --- File Loading ---
    try:
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating default file.{RESET}")
        try:  # Attempt to recreate the file with defaults if corrupted
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            return default_config
        except OSError as e_create:
            init_logger.error(f"{NEON_RED}Error recreating default config file after decode error: {e_create}. Using internal defaults.{RESET}")
            return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error loading config file '{filepath}': {e}{RESET}", exc_info=True)
        init_logger.warning("Using internal default configuration values.")
        return default_config

    # --- Validation and Merging ---
    try:
        # Ensure all keys from default_config exist in loaded_config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True  # Mark for saving if keys were added

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: dict, key_path: str, min_val: int | float | Decimal, max_val: int | float | Decimal,
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """Validates a numeric configuration value at a given path (e.g., "protection.leverage").
            Checks type (int/float), range [min_val, max_val] or (min_val, max_val] if strict.
            Uses the default value and logs a warning if validation fails.
            Updates the config dictionary in place if correction is needed.

            Returns:
                True if a correction was made, False otherwise.
            """
            nonlocal config_needs_saving  # Allow modification of the outer scope variable
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
                return False  # Path itself is wrong

            if original_val is None:
                init_logger.warning(f"Config validation error: Key missing at '{key_path}'. Should have been added by _ensure_config_keys.")
                # Attempt to add it again just in case, though this shouldn't normally happen
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val  # Start with the original value

            try:
                # Convert to Decimal for robust comparison
                num_val = Decimal(str(original_val))
                # Check range
                min_check = num_val > Decimal(str(min_val)) if is_strict_min else num_val >= Decimal(str(min_val))
                range_check = min_check and num_val <= Decimal(str(max_val))
                # Check if zero is allowed and value is zero, bypassing range check if so
                zero_ok = allow_zero and num_val == Decimal(0)

                if not range_check and not zero_ok:
                    raise ValueError("Value out of allowed range.")

                # Check type and convert if necessary
                target_type = int if is_int else float
                converted_val = target_type(num_val)

                # Check if type or value changed significantly after conversion (handle potential float inaccuracies)
                needs_correction = False
                if type(converted_val) is not type(original_val):
                    needs_correction = True
                elif isinstance(original_val, (float, int)):
                     # Only compare floats if they are significantly different
                     if abs(float(original_val) - float(converted_val)) > 1e-9:
                          needs_correction = True
                elif converted_val != original_val:  # Compare other types directly
                    needs_correction = True

                if needs_correction:
                    init_logger.info(f"{NEON_YELLOW}Config: Corrected type/value for '{key_path}' from {repr(original_val)} to {repr(converted_val)}.{RESET}")
                    final_val = converted_val
                    corrected = True

            except (ValueError, InvalidOperation, TypeError):
                # Handle cases where value is non-numeric, out of range, or conversion fails
                range_str = f"{'(' if is_strict_min else '['}{min_val}, {max_val}{']'}"
                if allow_zero: range_str += " or 0"
                init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{repr(original_val)}'. Using default: {repr(default_val)}. Expected range: {range_str}{RESET}")
                final_val = default_val  # Use the default value
                corrected = True

            # If a correction occurred, update the config dictionary and mark for saving
            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        # --- Apply Validations to Specific Config Keys ---
        # General
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid config interval '{updated_config.get('interval')}'. Valid: {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        validate_numeric(updated_config, "fetch_limit", 100, MAX_DF_LEN, is_int=True)  # Allow high fetch limit request, capped internally later
        validate_numeric(updated_config, "risk_per_trade", 0, 1, is_strict_min=True)  # Risk must be > 0 and <= 1
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True)  # Leverage 0 means no leverage setting attempt
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, MAX_DF_LEN, is_int=True)  # Allow long ATR period
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)  # Allow long Vol EMA
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)  # e.g., 1.005 = 0.5% proximity
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)  # e.g., 1.001 = 0.1% proximity
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source '{updated_config['strategy_params']['ob_source']}'. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
             config_needs_saving = True

        # Protection Params
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0001, 0.5, is_strict_min=True)  # Must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0, 0.5, allow_zero=True)  # 0 means activate immediately
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.1, 10.0)
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)  # 0 means move SL exactly to entry
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.1, 100.0)
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0, 100.0, allow_zero=True)  # 0 disables initial TP

        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             try:
                 # Convert config back to standard Python types (e.g., float) for JSON serialization
                 config_to_save = json.loads(json.dumps(updated_config))
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated configuration to '{filepath}': {save_err}{RESET}", exc_info=True)

        # Update the global QUOTE_CURRENCY from the validated config
        global QUOTE_CURRENCY
        QUOTE_CURRENCY = updated_config.get("quote_currency", "USDT")
        init_logger.debug(f"Quote currency set to: {QUOTE_CURRENCY}")

        return updated_config  # Return the validated and potentially corrected config

    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error processing configuration: {e}. Using internal defaults.{RESET}", exc_info=True)
        return default_config  # Fallback to defaults on unexpected error


# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)


# QUOTE_CURRENCY is updated inside load_config

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object.
    - Sets API keys, rate limiting, default type (linear).
    - Configures sandbox mode based on config.json.
    - Loads exchange markets with retries.
    - Performs an initial balance check.

    Args:
        logger: The logger instance to use for status messages.

    Returns:
        The initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger  # Alias for convenience
    try:
        # Common CCXT exchange options
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',         # Assume linear contracts by default
                'adjustForTimeDifference': True,  # Auto-adjust for clock skew
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
        if CONFIG.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet Environment){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}!!! USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK !!!{RESET}")

        # Load Markets with Retries
        lg.info(f"Attempting to load markets for {exchange.id}...")
        markets_loaded = False
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # Force reload on retries to ensure fresh market data
                exchange.load_markets(reload=(attempt > 0))
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols found).")
                    markets_loaded = True
                    break  # Exit retry loop on success
                else:
                    lg.warning(f"Market loading returned empty result (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                lg.warning(f"Network error loading markets (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}.")
                if attempt >= MAX_API_RETRIES:
                    lg.critical(f"{NEON_RED}Maximum retries exceeded while loading markets due to network errors. Exiting.{RESET}")
                    return None
                lg.warning(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            except ccxt.AuthenticationError as e:
                 lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API Key/Secret/Permissions. Exiting.{RESET}")
                 return None
            except Exception as e:
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load markets for {exchange.id} after all retries. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} | Sandbox: {CONFIG.get('use_sandbox')}")

        # Initial Balance Check (Optional but recommended)
        lg.info(f"Attempting initial balance fetch for quote currency ({QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                lg.info(f"{NEON_GREEN}Initial available balance: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                # Balance fetch failed after retries within fetch_balance function
                lg.critical(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}. Cannot proceed safely.{RESET}")
                if CONFIG.get('enable_trading', False):
                    lg.critical(f"{NEON_RED}Trading is enabled, but balance check failed. Exiting to prevent errors.{RESET}")
                    return None
                else:
                    lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding without confirmed initial balance, but errors might occur later.{RESET}")
        except ccxt.AuthenticationError as auth_err:
            # Handle auth errors specifically here as they are critical
            lg.critical(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
            return None
        except Exception as balance_err:
             # Catch other potential errors during the initial balance check
             lg.warning(f"{NEON_YELLOW}Initial balance fetch encountered an error: {balance_err}.{RESET}", exc_info=True)
             if CONFIG.get('enable_trading', False):
                 lg.critical(f"{NEON_RED}Trading is enabled, but balance check failed. Exiting to prevent errors.{RESET}")
                 return None
             else:
                 lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding despite balance check error.{RESET}")

        return exchange  # Return the initialized exchange object

    except Exception as e:
        # Catch-all for errors during the initialization process itself
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a given symbol using CCXT's fetch_ticker.
    - Prioritizes 'last' price.
    - Falls back to mid-price (bid+ask)/2 if 'last' is unavailable.
    - Falls back to 'ask' or 'bid' if only one is available.
    - Includes retry logic for network errors and rate limits.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        The current price as a Decimal, or None if fetching fails after retries.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker data for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Decimal | None = None

            # Helper to safely convert ticker values to Decimal
            def safe_decimal_from_ticker(value_str: Any | None, field_name: str) -> Decimal | None:
                if value_str is None: return None
                try:
                    # Check for non-empty string and convert
                    s_val = str(value_str).strip()
                    if not s_val: return None
                    dec_val = Decimal(s_val)
                    # Ensure price is positive
                    return dec_val if dec_val > Decimal('0') else None
                except (ValueError, InvalidOperation, TypeError):
                    lg.warning(f"Could not parse ticker field '{field_name}' value '{value_str}' to Decimal.")
                    return None

            # Try fetching 'last' price first
            price = safe_decimal_from_ticker(ticker.get('last'), 'last')

            # Fallback logic if 'last' price is unavailable or invalid
            if price is None:
                bid = safe_decimal_from_ticker(ticker.get('bid'), 'bid')
                ask = safe_decimal_from_ticker(ticker.get('ask'), 'ask')
                if bid and ask:
                    # Use mid-price if both bid and ask are valid
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price (Bid: {bid.normalize()}, Ask: {ask.normalize()}) -> {price.normalize()}")
                elif ask:
                    # Use ask price if only ask is valid
                    price = ask
                    lg.warning(f"{NEON_YELLOW}Using 'ask' price as fallback: {price.normalize()}{RESET}")
                elif bid:
                    # Use bid price if only bid is valid
                    price = bid
                    lg.warning(f"{NEON_YELLOW}Using 'bid' price as fallback: {price.normalize()}{RESET}")

            # Check if a valid price was obtained
            if price:
                lg.debug(f"Current price successfully fetched for {symbol}: {price.normalize()}")
                return price
            else:
                lg.warning(f"No valid price ('last', 'mid', 'ask', 'bid') found in ticker data (Attempt {attempts + 1}). Ticker: {ticker}")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            # Apply a longer delay for rate limit errors
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue to next attempt without incrementing 'attempts' here, as we waited
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching price: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
             return None  # Fatal error
        except ccxt.ExchangeError as e:
            # General exchange errors might be retryable or not
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Could add checks for specific non-retryable error codes here if needed
            # For now, assume potentially retryable unless it's an auth error
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            # Unexpected errors are less likely to be resolved by retrying
            return None  # Exit on unexpected errors

        # Increment attempt counter and apply delay
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV (kline) data using CCXT's fetch_ohlcv method.
    - Handles Bybit V5 'category' parameter automatically.
    - Implements retry logic for network errors and rate limits.
    - Validates fetched data (e.g., timestamp lag).
    - Processes data into a Pandas DataFrame with Decimal types.
    - Cleans data (drops NaNs, zero prices/volumes).
    - Trims DataFrame to MAX_DF_LEN to manage memory.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        timeframe: The CCXT timeframe string (e.g., "5m", "1h", "1d").
        limit: The desired number of klines. Will be capped by BYBIT_API_KLINE_LIMIT per request.
        logger: The logger instance for status messages.

    Returns:
        A Pandas DataFrame containing the OHLCV data, indexed by timestamp (UTC),
        with columns ['open', 'high', 'low', 'close', 'volume'] as Decimals.
        Returns an empty DataFrame if fetching or processing fails.
    """
    lg = logger
    if not hasattr(exchange, 'fetch_ohlcv') or not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV. Cannot fetch klines.")
        return pd.DataFrame()

    ohlcv: list[list[int | float | str]] | None = None
    # Determine the actual number of klines to request, respecting the API limit
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
        lg.debug(f"Requested limit ({limit}) exceeds API limit ({BYBIT_API_KLINE_LIMIT}). Will request {BYBIT_API_KLINE_LIMIT}.")

    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching klines for {symbol} | Timeframe: {timeframe} | Limit: {actual_request_limit} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            params = {}
            # Add Bybit V5 specific parameters if needed
            if 'bybit' in exchange.id.lower():
                 try:
                     # Determine category (linear/inverse/spot) for Bybit V5 API
                     market = exchange.market(symbol)
                     category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                     params['category'] = category
                     lg.debug(f"Using Bybit category: {category}")
                 except Exception as e:
                     lg.warning(f"Could not automatically determine market category for {symbol} kline fetch: {e}. Proceeding without category param.")

            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit, params=params)
            fetched_count = len(ohlcv) if ohlcv else 0
            lg.debug(f"API returned {fetched_count} candles (requested {actual_request_limit}).")

            # --- Basic Validation on Fetched Data ---
            if ohlcv and fetched_count > 0:
                # Warn if API limit was hit and more data was potentially needed
                if fetched_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT:
                    lg.warning(f"{NEON_YELLOW}Hit API kline limit ({BYBIT_API_KLINE_LIMIT}). Strategy might require more data than fetched in a single request.{RESET}")

                # Validate timestamp lag of the last candle
                try:
                    last_candle_timestamp_ms = ohlcv[-1][0]
                    last_ts = pd.to_datetime(last_candle_timestamp_ms, unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    # Get timeframe duration in seconds
                    interval_seconds = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') else 300  # Default 5 mins
                    # Allow a lag of up to 5 intervals or 5 minutes, whichever is larger
                    max_allowed_lag_seconds = max((interval_seconds * 5), 300)
                    actual_lag_seconds = (now_utc - last_ts).total_seconds()

                    if actual_lag_seconds < max_allowed_lag_seconds:
                        lg.debug(f"Last kline timestamp {last_ts} seems recent (Lag: {actual_lag_seconds:.1f}s <= Max: {max_allowed_lag_seconds}s). Data OK.")
                        break  # Successful fetch and basic validation passed, exit retry loop
                    else:
                        lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_ts} is too old (Lag: {actual_lag_seconds:.1f}s > Max: {max_allowed_lag_seconds}s). Data might be stale. Retrying...{RESET}")
                        ohlcv = None  # Discard potentially stale data and retry
                except Exception as ts_err:
                    lg.warning(f"Could not validate kline timestamp lag: {ts_err}. Proceeding with fetched data.")
                    break  # Proceed even if validation fails, but log warning
            else:
                lg.warning(f"API returned no kline data (Attempt {attempts + 1}). Retrying...")
                ohlcv = None  # Ensure ohlcv is None if fetch failed

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3  # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue loop without incrementing attempts for rate limit waits
        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error fetching klines: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
             return pd.DataFrame()  # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
            # Consider if specific exchange errors are non-retryable
            if "invalid timeframe" in str(e).lower():
                lg.critical(f"{NEON_RED}Invalid timeframe '{timeframe}' specified for {exchange.id}. Exiting.{RESET}")
                return pd.DataFrame()
            # Otherwise, assume retryable for now
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
            return pd.DataFrame()  # Stop on unexpected errors

        # Increment attempt counter (only if not a rate limit wait) and apply delay
        attempts += 1
        if attempts <= MAX_API_RETRIES and ohlcv is None:  # Only sleep if fetch failed and retries remain
             time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # After retry loop, check if data was successfully fetched
    if not ohlcv:
        lg.error(f"{NEON_RED}Failed to fetch kline data for {symbol} {timeframe} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
        return pd.DataFrame()

    # --- Process Fetched Data into DataFrame ---
    try:
        lg.debug(f"Processing {len(ohlcv)} fetched candles into DataFrame...")
        # Define standard OHLCV column names
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Create DataFrame, ensuring columns match the fetched data structure
        df = pd.DataFrame(ohlcv, columns=cols[:len(ohlcv[0])])

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows with invalid timestamps
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential errors
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric first, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Convert valid numbers to Decimal, invalid ones remain NaN (or become Decimal('NaN'))
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
            else:
                 lg.warning(f"Expected column '{col}' not found in fetched kline data.")

        # --- Data Cleaning ---
        initial_len = len(df)
        # Drop rows with NaN in essential price columns or non-positive close price
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN volume or negative volume (if volume column exists)
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            df = df[df['volume'] >= Decimal('0')]  # Allow zero volume

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows during cleaning (NaNs, zero/neg prices, neg volume).")

        if df.empty:
            lg.warning(f"Kline DataFrame is empty after cleaning for {symbol} {timeframe}.")
            return pd.DataFrame()

        # Ensure DataFrame is sorted by timestamp index
        df.sort_index(inplace=True)

        # --- Memory Management ---
        # Trim DataFrame if it exceeds the maximum allowed length
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds maximum ({MAX_DF_LEN}). Trimming to most recent {MAX_DF_LEN} candles.")
            df = df.iloc[-MAX_DF_LEN:].copy()  # Keep the latest N rows

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves and validates market information for a specific symbol from the exchange.
    - Reloads markets if the symbol is initially not found.
    - Extracts precision (price, amount), limits (min/max amount, cost), contract type (linear/inverse/spot), and contract size.
    - Includes retry logic for network errors.
    - Logs critical warnings if essential precision data is missing.

    Args:
        exchange: The initialized ccxt.Exchange object (must have markets loaded).
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        A dictionary containing market details, or None if the market is not found
        or a critical error occurs. The dictionary includes added keys like
        'is_contract', 'is_linear', 'is_inverse', 'contract_type_str'.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            # Check if markets are loaded and the symbol exists
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market info for '{symbol}' not found in loaded markets. Attempting to reload markets...")
                try:
                    exchange.load_markets(reload=True)  # Force reload
                except Exception as reload_err:
                     lg.error(f"Failed to reload markets while searching for {symbol}: {reload_err}")
                     # Don't immediately return, try checking again in case it was temporary

            # Check again after potential reload
            if symbol not in exchange.markets:
                if attempts == 0:
                    # First attempt failed even after reload, log warning and retry
                    lg.warning(f"Symbol '{symbol}' still not found after market reload (Attempt {attempts + 1}). Retrying check...")
                    # No need to sleep here yet, retry immediately
                else:
                    # Subsequent attempts failed, assume symbol is invalid
                    lg.error(f"{NEON_RED}Market '{symbol}' not found on {exchange.id} after reload and retries.{RESET}")
                    return None  # Symbol definitively not found
                attempts += 1
                time.sleep(RETRY_DELAY_SECONDS)  # Wait before next check attempt
                continue  # Go to next attempt

            # --- Market Found - Extract Details ---
            market = exchange.market(symbol)
            if market:
                # Add custom flags for easier logic later
                market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
                market['is_linear'] = market.get('linear', False) and market['is_contract']
                market['is_inverse'] = market.get('inverse', False) and market['is_contract']
                market['contract_type_str'] = "Linear" if market['is_linear'] else "Inverse" if market['is_inverse'] else "Spot" if market.get('spot') else "Unknown"

                # Helper for formatting Decimal values for logging
                def format_decimal(value: Any | None) -> str:
                    if value is None: return 'N/A'
                    try: return str(Decimal(str(value)).normalize())
                    except (InvalidOperation, TypeError): return 'Invalid'

                # Extract precision and limits safely
                precision = market.get('precision', {})
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                # Log extracted details for verification
                log_msg = (
                    f"Market Info ({symbol}): ID={market.get('id', 'N/A')}, Type={market.get('type', 'N/A')}, "
                    f"Contract Type={market['contract_type_str']}, "
                    f"Precision(Price/Amount): {format_decimal(precision.get('price'))} / {format_decimal(precision.get('amount'))}, "
                    f"Limits(Amount Min/Max): {format_decimal(amount_limits.get('min'))} / {format_decimal(amount_limits.get('max'))}, "
                    f"Limits(Cost Min/Max): {format_decimal(cost_limits.get('min'))} / {format_decimal(cost_limits.get('max'))}, "
                    f"Contract Size: {format_decimal(market.get('contractSize', '1'))}"  # Default to 1 if not specified
                )
                lg.debug(log_msg)

                # Critical check: Ensure essential precision info exists
                if precision.get('price') is None or precision.get('amount') is None:
                    lg.error(f"{NEON_RED}CRITICAL VALIDATION FAILED:{RESET} Market '{symbol}' is missing essential precision data (price or amount). Trading calculations will likely fail. Cannot proceed safely.")
                    # Decide whether to return None or raise an error depending on desired strictness
                    return None  # Returning None is safer for the bot flow

                return market  # Return the enriched market dictionary

            else:
                # This case should theoretically not be reached if symbol is in exchange.markets
                lg.error(f"Market dictionary is unexpectedly None for '{symbol}' even though it was found in exchange.markets.")
                return None

        # --- Error Handling with Retries ---
        except ccxt.BadSymbol as e:
            # Symbol is definitively invalid according to the exchange
            lg.error(f"Symbol '{symbol}' is invalid on {exchange.id}: {e}")
            return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error retrieving market info for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded retrieving market info for {symbol} due to network errors.{RESET}")
                return None
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error getting market info: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
             return None  # Fatal error
        except ccxt.ExchangeError as e:
            # General exchange error, potentially temporary
            lg.error(f"{NEON_RED}Exchange error retrieving market info for {symbol}: {e}{RESET}")
            # Could add checks for specific non-retryable errors if known
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded retrieving market info for {symbol} due to exchange errors.{RESET}")
                 return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error retrieving market info for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    # Should only be reached if all retries fail due to network/exchange errors
    lg.error(f"{NEON_RED}Failed to retrieve market info for {symbol} after all attempts.{RESET}")
    return None


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available trading balance for a specific currency (e.g., USDT).
    - Handles Bybit V5 account types (UNIFIED, CONTRACT) to find the correct balance.
    - Includes retry logic for network errors and rate limits.
    - Handles authentication errors critically.

    Args:
        exchange: The initialized ccxt.Exchange object.
        currency: The currency code to fetch the balance for (e.g., "USDT").
        logger: The logger instance for status messages.

    Returns:
        The available balance as a Decimal, or None if fetching fails after retries.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            balance_str: str | None = None
            found: bool = False

            # Bybit V5 often requires specifying account type (Unified or Contract)
            account_types_to_check = ['UNIFIED', 'CONTRACT'] if 'bybit' in exchange.id.lower() else ['']  # Check default if not Bybit

            for acc_type in account_types_to_check:
                try:
                    params = {'accountType': acc_type} if acc_type else {}
                    lg.debug(f"Fetching balance for {currency} (Account Type: {acc_type or 'Default'}, Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
                    balance_info = exchange.fetch_balance(params=params)

                    # --- Try different ways to extract balance from CCXT/Bybit V5 response ---
                    # 1. Standard CCXT structure
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        found = True; break

                    # 2. Bybit V5 structure (often nested in 'info') - Unified/Contract specific
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        for account_details in balance_info['info']['result']['list']:
                             # Ensure we are checking the correct account type if specified
                            if (account_details.get('accountType') == acc_type or not acc_type) and isinstance(account_details.get('coin'), list):
                                for coin_data in account_details['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Try various fields Bybit uses for available balance
                                        free_balance = coin_data.get('availableToWithdraw') or \
                                                       coin_data.get('availableBalance') or \
                                                       coin_data.get('walletBalance')  # Last resort
                                        if free_balance is not None:
                                            balance_str = str(free_balance)
                                            found = True; break  # Found in coin list
                                if found: break  # Found in account details
                        if found: break  # Found across account types

                except ccxt.ExchangeError as e:
                    # Errors like "account type does not exist" are expected when checking multiple types
                    lg.debug(f"Minor exchange error fetching balance for type '{acc_type}': {e}. Trying next type...")
                    continue  # Try the next account type
                except Exception as e:
                    # Catch other unexpected errors during a specific account type check
                    lg.warning(f"Unexpected error fetching balance for type '{acc_type}': {e}. Trying next type...")
                    continue  # Try the next account type

            # --- Fallback: Fetch balance without specifying account type ---
            # This might work for some exchanges or older Bybit accounts
            if not found and '' not in account_types_to_check:  # Only run if default wasn't checked
                 try:
                    lg.debug(f"Fetching balance for {currency} (Default/Fallback, Attempt {attempts + 1})...")
                    balance_info = exchange.fetch_balance()
                    # Repeat extraction logic for the default response structure
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        balance_str = str(balance_info[currency]['free'])
                        found = True
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account_details in balance_info['info']['result']['list']:
                             if isinstance(account_details.get('coin'), list):
                                 for coin_data in account_details['coin']:
                                     if coin_data.get('coin') == currency:
                                         free_balance = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free_balance is not None:
                                             balance_str = str(free_balance)
                                             found = True; break
                                 if found: break
                         if found: break
                 except Exception as e:
                     lg.error(f"Error during fallback balance fetch: {e}", exc_info=True)

            # --- Process the result ---
            if found and balance_str is not None:
                try:
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative
                    final_balance = balance_decimal if balance_decimal >= Decimal('0') else Decimal('0')
                    lg.debug(f"Successfully parsed balance for {currency}: {final_balance.normalize()}")
                    return final_balance  # Success
                except (ValueError, InvalidOperation, TypeError) as e:
                    # Raise an error if the found balance string cannot be converted
                    raise ccxt.ExchangeError(f"Failed to convert fetched balance string '{balance_str}' for {currency} to Decimal: {e}")
            else:
                # If not found after checking all types and fallback
                raise ccxt.ExchangeError(f"Could not find balance information for currency '{currency}' in the response.")

        # --- Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance for {currency}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
            return None  # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            # Log exchange errors (like currency not found, conversion errors) and retry
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for {currency}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching balance for {currency}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None


# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an existing open position for the given symbol using CCXT's fetch_positions.
    - Handles Bybit V5 specifics (category, symbol filtering).
    - Determines position side (long/short) and size accurately.
    - Parses key position details (entry price, leverage, SL/TP, TSL).
    - Includes retry logic for network errors and rate limits.
    - Returns None if no position exists or if fetching fails.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT").
        logger: The logger instance for status messages.

    Returns:
        A dictionary containing details of the open position if found, otherwise None.
        The dictionary is a standardized format including 'size_decimal', 'side',
        'entryPrice', 'leverage', 'stopLossPrice', 'takeProfitPrice', etc.
    """
    lg = logger
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            positions: list[dict] = []
            market_id: str | None = None
            category: str | None = None

            # --- Fetch Positions (Handling Bybit V5 Specifics) ---
            try:
                # Get market details to determine category and market ID
                market = exchange.market(symbol)
                market_id = market['id']
                category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'

                # Attempt to fetch positions for the specific symbol and category
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching positions with specific params: {params}")
                positions = exchange.fetch_positions([symbol], params=params)  # Request only specific symbol

            except ccxt.ArgumentsRequired as e:
                # Some exchanges/versions might require fetching all positions if filtering isn't supported well
                lg.warning(f"Exchange requires fetching all positions ({e}). Filtering locally (potentially slower).")
                params = {'category': category or 'linear'}  # Default to linear if category unknown
                all_positions = exchange.fetch_positions(params=params)
                # Filter the results manually
                positions = [
                    p for p in all_positions
                    if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id
                ]
                lg.debug(f"Fetched {len(all_positions)} total positions, filtered down to {len(positions)} for {symbol}.")

            except ccxt.ExchangeError as e:
                 # Bybit often returns specific codes for "no position"
                 no_pos_codes = [110025]  # Add more codes if discovered
                 no_pos_messages = ["position not found", "position idx not match"]
                 err_str = str(e).lower()
                 code = getattr(e, 'code', None)
                 if code in no_pos_codes or any(msg in err_str for msg in no_pos_messages):
                     lg.info(f"No open position found for {symbol} (Exchange message: {e}).")
                     return None  # No position exists
                 else:
                     # Re-raise other exchange errors to be handled by the main retry loop
                     raise e

            # --- Process Fetched Positions ---
            active_position: dict | None = None
            # Define a small threshold based on amount precision to consider a position "open"
            size_threshold = Decimal('1e-9')  # Default small value
            try:
                amount_precision_str = exchange.market(symbol)['precision']['amount']
                if amount_precision_str:
                    # Use a fraction of the minimum amount step as the threshold
                    size_threshold = Decimal(str(amount_precision_str)) * Decimal('0.1')
            except Exception as prec_err:
                lg.warning(f"Could not get amount precision for {symbol} to set size threshold: {prec_err}. Using default: {size_threshold}")
            lg.debug(f"Using position size threshold: {size_threshold}")

            # Iterate through the filtered positions to find an active one
            for pos in positions:
                # Extract size from 'info' (exchange-specific) or standard 'contracts' field
                size_str = str(pos.get('info', {}).get('size', pos.get('contracts', ''))).strip()
                if not size_str:
                    lg.debug(f"Skipping position entry with missing size data: {pos.get('info', {})}")
                    continue

                try:
                    # Convert size to Decimal and check against threshold
                    size_decimal = Decimal(size_str)
                    if abs(size_decimal) > size_threshold:
                        # Found an active position with significant size
                        active_position = pos
                        active_position['size_decimal'] = size_decimal  # Store the parsed Decimal size
                        lg.debug(f"Found active position entry: Size={size_decimal.normalize()}")
                        break  # Stop searching once an active position is found
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     # Log error if size string cannot be parsed, skip this entry
                     lg.warning(f"Could not parse/check position size string '{size_str}': {parse_err}. Skipping this position entry.")
                     continue  # Move to the next position entry

            # --- Format and Return Active Position ---
            if active_position:
                std_pos = active_position.copy()  # Work on a copy
                info = std_pos.get('info', {})  # Exchange-specific details

                # Determine Side (long/short) reliably
                side = std_pos.get('side')  # Standard CCXT field
                size = std_pos['size_decimal']  # Use the parsed Decimal size

                if side not in ['long', 'short']:
                    # Fallback using Bybit V5 'side' field ('Buy'/'Sell') or inferred from size
                    side_v5 = str(info.get('side', '')).lower()
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif size > size_threshold: side = 'long'  # Infer from positive size
                    elif size < -size_threshold: side = 'short'  # Infer from negative size
                    else: side = None  # Cannot determine side

                if not side:
                    lg.error(f"Could not determine side for active position {symbol}. Data: {info}")
                    return None  # Cannot proceed without side
                std_pos['side'] = side

                # Standardize other key fields, preferring standard CCXT names, falling back to 'info'
                std_pos['entryPrice'] = std_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice')
                std_pos['leverage'] = std_pos.get('leverage') or info.get('leverage')
                std_pos['liquidationPrice'] = std_pos.get('liquidationPrice') or info.get('liqPrice')
                std_pos['unrealizedPnl'] = std_pos.get('unrealizedPnl') or info.get('unrealisedPnl')

                # Parse protection levels (SL/TP/TSL) - ensure they are strings if present
                sl = info.get('stopLoss') or std_pos.get('stopLossPrice')
                tp = info.get('takeProfit') or std_pos.get('takeProfitPrice')
                tsl_dist = info.get('trailingStop')  # Bybit V5 TSL distance
                tsl_act = info.get('activePrice')  # Bybit V5 TSL activation price

                if sl is not None and str(sl).strip() and Decimal(str(sl)) != 0: std_pos['stopLossPrice'] = str(sl)
                if tp is not None and str(tp).strip() and Decimal(str(tp)) != 0: std_pos['takeProfitPrice'] = str(tp)
                if tsl_dist is not None and str(tsl_dist).strip() and Decimal(str(tsl_dist)) != 0: std_pos['trailingStopLoss'] = str(tsl_dist)
                if tsl_act is not None and str(tsl_act).strip() and Decimal(str(tsl_act)) != 0: std_pos['tslActivationPrice'] = str(tsl_act)

                # Log summary of the found position
                ep_str = format_decimal(std_pos.get('entryPrice'))
                size_str = std_pos['size_decimal'].normalize()
                sl_str = format_decimal(std_pos.get('stopLossPrice', None))
                tp_str = format_decimal(std_pos.get('takeProfitPrice', None))
                tsl_str = f"Dist:{format_decimal(std_pos.get('trailingStopLoss', None))}/Act:{format_decimal(std_pos.get('tslActivationPrice', None))}"
                pnl_str = format_decimal(std_pos.get('unrealizedPnl'))
                liq_str = format_decimal(std_pos.get('liquidationPrice'))

                lg.info(f"{NEON_GREEN}Active {side.upper()} Position Found ({symbol}): "
                        f"Size={size_str}, Entry={ep_str}, Liq={liq_str}, PnL={pnl_str}, "
                        f"SL={sl_str}, TP={tp_str}, TSL=({tsl_str}){RESET}")
                return std_pos  # Return the standardized position dictionary
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
            continue  # Continue loop without incrementing attempts
        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error fetching positions: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
            return None  # Fatal error
        except ccxt.ExchangeError as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Could add checks for specific non-retryable errors here
        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets the leverage for a derivatives symbol using CCXT's set_leverage method.
    - Skips if the market is not a contract (spot).
    - Skips if leverage is invalid (<= 0).
    - Handles Bybit V5 specific parameters (category, buy/sell leverage).
    - Includes retry logic for network/exchange errors.
    - Checks for specific Bybit codes indicating success or leverage already set.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT:USDT").
        leverage: The desired integer leverage level.
        market_info: The market information dictionary from get_market_info.
        logger: The logger instance for status messages.

    Returns:
        True if leverage was set successfully or was already set, False otherwise.
    """
    lg = logger
    # Validate input and market type
    if not market_info.get('is_contract', False):
        lg.info(f"Leverage setting skipped for {symbol}: Not a contract market.")
        return True  # Consider success as no action needed for spot
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}). Must be > 0.")
        return False
    if not hasattr(exchange, 'set_leverage') or not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage method.")
        return False

    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {symbol} to {leverage}x (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            params = {}
            market_id = market_info['id']
            # --- Bybit V5 Specific Parameters ---
            if 'bybit' in exchange.id.lower():
                 # Determine category and set buy/sell leverage explicitly for Bybit V5
                 category = 'linear' if market_info.get('linear', True) else 'inverse'  # Default linear if unsure
                 params = {
                     'category': category,
                     'symbol': market_id,  # Use market ID for Bybit API
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 setLeverage params: {params}")

            # --- Execute set_leverage call ---
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"set_leverage raw response: {response}")

            # --- Check Response (Bybit V5 specific codes) ---
            # CCXT might parse some responses, but Bybit often returns structured info
            ret_code = response.get('retCode') if isinstance(response, dict) else None

            if ret_code is not None:  # Bybit V5 response structure detected
                 if ret_code == 0:
                     lg.info(f"{NEON_GREEN}Leverage successfully set for {symbol} to {leverage}x (Bybit Code: 0).{RESET}")
                     return True
                 elif ret_code == 110045:  # "Leverage not modified"
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} is already {leverage}x (Bybit Code: 110045).{RESET}")
                     return True
                 else:
                     # Raise an error for other non-zero Bybit return codes
                     error_message = response.get('retMsg', 'Unknown Bybit API error')
                     raise ccxt.ExchangeError(f"Bybit API error setting leverage: {error_message} (Code: {ret_code})")
            else:
                # Assume success if no specific error code structure is found and no exception was raised
                lg.info(f"{NEON_GREEN}Leverage set/confirmed for {symbol} to {leverage}x (No specific Bybit code in response).{RESET}")
                return True

        # --- Error Handling with Retries ---
        except ccxt.ExchangeError as e:
            last_exception = e
            err_code = getattr(e, 'code', None)  # Check if CCXT parsed a code
            err_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {err_code}){RESET}")

            # Check for non-retryable conditions based on code or message
            if err_code == 110045 or "not modified" in err_str:
                lg.info(f"{NEON_YELLOW}Leverage already set (detected via error). Treating as success.{RESET}")
                return True  # Already set is considered success
            # List of known fatal Bybit error codes for leverage setting
            fatal_codes = [110028, 110009, 110055, 110043, 110044, 110013, 10001, 10004, 3400045]
            fatal_messages = ["margin mode", "position exists", "risk limit", "parameter error", "insufficient balance"]
            if err_code in fatal_codes or any(msg in err_str for msg in fatal_messages):
                lg.error(f"{NEON_RED} >> Hint: This appears to be a non-retryable leverage error. Aborting leverage set.{RESET}")
                return False  # Fatal error

            # If error is potentially retryable and retries remain
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError setting leverage.{RESET}")
                return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting leverage for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError setting leverage.{RESET}")
                return False

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error setting leverage: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
             return False  # Fatal error

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)
            return False  # Stop on unexpected errors

        # Increment attempt counter and delay before retrying
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False


def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Decimal | None:
    """Calculates the appropriate position size based on account balance, risk percentage,
    entry price, stop loss price, and market constraints (precision, limits).
    Handles both linear and inverse contracts.

    Args:
        balance: Available trading balance (in quote currency, e.g., USDT).
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: The calculated initial stop loss price.
        entry_price: The intended entry price (or current price for market orders).
        market_info: The market information dictionary from get_market_info.
        exchange: The initialized ccxt.Exchange object (used for precision formatting).
        logger: The logger instance for status messages.

    Returns:
        The calculated position size as a Decimal, adjusted for market rules,
        or None if sizing is not possible (e.g., invalid inputs, cannot meet limits).
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE')  # e.g., USDT
    base_currency = market_info.get('base', 'BASE')   # e.g., BTC
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base_currency  # Unit of the position size

    lg.info(f"--- Position Sizing Calculation ({symbol}) ---")

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Sizing failed: Invalid balance ({balance} {quote_currency}). Must be positive.")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
            raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive).")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed: Invalid risk_per_trade value '{risk_per_trade}': {e}")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Sizing failed: Entry price ({entry_price}) and Stop Loss price ({initial_stop_loss_price}) must be positive.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Sizing failed: Stop Loss price ({initial_stop_loss_price}) cannot be equal to Entry price ({entry_price}).")
        return None

    # --- Extract Market Constraints ---
    try:
        precision = market_info['precision']
        limits = market_info['limits']
        amount_precision_str = precision.get('amount')  # Minimum step for position size
        price_precision_str = precision.get('price')   # Minimum step for price
        if amount_precision_str is None or price_precision_str is None:
             raise ValueError("Market precision data (amount or price) is missing.")

        amount_precision_step = Decimal(str(amount_precision_str))
        if amount_precision_step <= Decimal('0'):
             raise ValueError(f"Invalid amount precision step size: {amount_precision_step}")

        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})

        min_amount = Decimal(str(amount_limits.get('min', '0')))  # Minimum allowed position size
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')  # Max allowed size

        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')  # Minimum order value (size * price)
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')  # Max order value

        # Contract size (e.g., 1 for BTC/USDT linear, 100 for BTC/USD inverse)
        contract_size_str = market_info.get('contractSize', '1')
        contract_size = Decimal(str(contract_size_str)) if contract_size_str else Decimal('1')
        if contract_size <= Decimal('0'):
             raise ValueError(f"Invalid contract size: {contract_size}")

        lg.debug(f"Market Constraints: Amount Step={amount_precision_step.normalize()}, Min/Max Amount={min_amount.normalize()}/{max_amount.normalize()}, "
                 f"Min/Max Cost={min_cost.normalize()}/{max_cost.normalize()}, Contract Size={contract_size.normalize()}")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Sizing failed: Error extracting or validating market details for {symbol}: {e}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    risk_amount_quote = balance * risk_decimal
    stop_loss_distance_price = abs(entry_price - initial_stop_loss_price)

    if stop_loss_distance_price <= Decimal('0'):
        # This should be caught earlier, but double-check
        lg.error(f"Sizing failed: Stop Loss distance is zero or negative ({stop_loss_distance_price}).")
        return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}")
    lg.info(f"  Risk Per Trade: {risk_decimal:.2%}")
    lg.info(f"  Risk Amount: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"  Entry Price: {entry_price.normalize()}")
    lg.info(f"  Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"  Stop Loss Distance (Price): {stop_loss_distance_price.normalize()}")
    lg.info(f"  Contract Type: {'Inverse' if is_inverse else 'Linear/Spot'}")

    # --- Calculate Initial Position Size (based on risk) ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse:  # Linear Contracts or Spot
            # Risk per unit = Price distance * Contract Size (value change per contract per $1 price move)
            # For spot, contract size is effectively 1 base unit.
            value_change_per_unit = stop_loss_distance_price * contract_size
            if value_change_per_unit <= Decimal('0'):
                 lg.error("Sizing failed (Linear/Spot): Calculated value change per unit is zero or negative.")
                 return None
            # Size = Total Risk Amount / Risk per Unit
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Calc: RiskAmt={risk_amount_quote} / (SLDist={stop_loss_distance_price} * ContrSize={contract_size}) = {calculated_size}")

        else:  # Inverse Contracts
            # Risk per contract = Contract Size * |(1 / Entry) - (1 / SL)| (change in value of 1 contract in quote currency)
            if entry_price <= 0 or initial_stop_loss_price <= 0:
                 lg.error("Sizing failed (Inverse): Entry or SL price is zero or negative.")
                 return None
            inverse_factor = abs((Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price))
            # Use a small tolerance for zero check due to potential floating point issues if not using Decimal strictly
            if inverse_factor <= Decimal('1e-18'):
                 lg.error("Sizing failed (Inverse): Calculated inverse factor is effectively zero.")
                 return None
            risk_per_contract_quote = contract_size * inverse_factor
            if risk_per_contract_quote <= Decimal('0'):
                 lg.error("Sizing failed (Inverse): Calculated risk per contract is zero or negative.")
                 return None
            # Size = Total Risk Amount / Risk per Contract
            calculated_size = risk_amount_quote / risk_per_contract_quote
            lg.debug(f"  Inverse Calc: RiskAmt={risk_amount_quote} / (ContrSize={contract_size} * InvFactor={inverse_factor}) = {calculated_size}")

    except (InvalidOperation, OverflowError, ZeroDivisionError) as calc_err:
        lg.error(f"Sizing failed: Error during initial size calculation: {calc_err}.")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Sizing failed: Initial calculated size is zero or negative ({calculated_size.normalize()}). Check inputs/risk.")
        return None

    lg.info(f"  Initial Calculated Size = {calculated_size.normalize()} {size_unit}")

    # --- Apply Market Limits and Precision ---
    adjusted_size = calculated_size

    # 1. Apply Amount Limits (Min/Max Size)
    if min_amount > 0 and adjusted_size < min_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} is below minimum amount {min_amount.normalize()}. Adjusting UP to minimum.{RESET}")
        adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} exceeds maximum amount {max_amount.normalize()}. Adjusting DOWN to maximum.{RESET}")
        adjusted_size = max_amount
    lg.debug(f"  Size after Amount Limits: {adjusted_size.normalize()} {size_unit}")

    # 2. Apply Cost Limits (Min/Max Order Value) - Requires recalculating size if limits are hit
    estimated_cost = Decimal('0')
    cost_adjustment_applied = False
    try:
        if entry_price > 0:
            # Cost = Size * Price * ContractSize (Linear) or Size * ContractSize / Price (Inverse)
            estimated_cost = (adjusted_size * entry_price * contract_size) if not is_inverse else ((adjusted_size * contract_size) / entry_price)
            lg.debug(f"  Estimated Cost for adjusted size: {estimated_cost.normalize()} {quote_currency}")
    except (InvalidOperation, OverflowError, ZeroDivisionError):
        lg.warning("Could not estimate cost for limit check.")

    if min_cost > 0 and estimated_cost > 0 and estimated_cost < min_cost:
        lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost.normalize()} is below minimum cost {min_cost.normalize()}. Attempting to increase size.{RESET}")
        required_size_for_min_cost = None
        try:
            # Calculate size needed to meet min cost: MinCost / (Price * ContrSize) [Lin] or MinCost * Price / ContrSize [Inv]
            if not is_inverse: required_size_for_min_cost = min_cost / (entry_price * contract_size)
            else: required_size_for_min_cost = (min_cost * entry_price) / contract_size

            if required_size_for_min_cost is None or required_size_for_min_cost <= 0: raise ValueError("Invalid required size calc")
            lg.info(f"  Size required to meet min cost: {required_size_for_min_cost.normalize()} {size_unit}")

            # Check if this required size exceeds max amount limit
            if max_amount < Decimal('inf') and required_size_for_min_cost > max_amount:
                lg.error(f"{NEON_RED}Cannot meet minimum cost ({min_cost.normalize()}) without exceeding maximum amount limit ({max_amount.normalize()}). Aborting sizing.{RESET}")
                return None

            # Adjust size up, ensuring it's still at least the minimum amount
            adjusted_size = max(min_amount, required_size_for_min_cost)
            cost_adjustment_applied = True

        except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
            lg.error(f"{NEON_RED}Failed to calculate size needed for minimum cost: {cost_calc_err}. Aborting sizing.{RESET}")
            return None

    elif max_cost < Decimal('inf') and estimated_cost > 0 and estimated_cost > max_cost:
        lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost.normalize()} exceeds maximum cost {max_cost.normalize()}. Attempting to reduce size.{RESET}")
        max_size_for_max_cost = None
        try:
            # Calculate max size allowed by max cost: MaxCost / (Price * ContrSize) [Lin] or MaxCost * Price / ContrSize [Inv]
            if not is_inverse: max_size_for_max_cost = max_cost / (entry_price * contract_size)
            else: max_size_for_max_cost = (max_cost * entry_price) / contract_size

            if max_size_for_max_cost is None or max_size_for_max_cost <= 0: raise ValueError("Invalid max size calc")
            lg.info(f"  Maximum size allowed by max cost: {max_size_for_max_cost.normalize()} {size_unit}")

            # Adjust size down, ensuring it's still at least the minimum amount
            adjusted_size = max(min_amount, min(adjusted_size, max_size_for_max_cost))  # Take the smaller of current adjusted or max allowed by cost
            cost_adjustment_applied = True

        except (InvalidOperation, OverflowError, ZeroDivisionError, ValueError) as cost_calc_err:
            lg.error(f"{NEON_RED}Failed to calculate maximum size allowed by maximum cost: {cost_calc_err}. Aborting sizing.{RESET}")
            return None

    if cost_adjustment_applied:
        lg.info(f"  Size after Cost Limits: {adjusted_size.normalize()} {size_unit}")

    # 3. Apply Amount Precision (Rounding to step size)
    final_size = adjusted_size
    try:
        # Use CCXT's built-in precision formatting
        amount_str_formatted = exchange.amount_to_precision(symbol, float(adjusted_size))
        final_size = Decimal(amount_str_formatted)
        if final_size != adjusted_size:
            lg.info(f"Applied amount precision (CCXT): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
    except Exception as fmt_err:
        lg.warning(f"{NEON_YELLOW}CCXT amount_to_precision failed: {fmt_err}. Attempting manual rounding down to step size.{RESET}")
        try:
            # Manual fallback: round down to the nearest multiple of the step size
            if amount_precision_step > 0:
                final_size = (adjusted_size // amount_precision_step) * amount_precision_step
                if final_size != adjusted_size:
                     lg.info(f"Applied manual amount precision (floor): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
            else:
                # Should not happen based on earlier check, but handle defensively
                lg.error("Amount precision step is zero or negative, cannot apply manual precision.")
                final_size = adjusted_size  # Use unrounded
        except Exception as manual_err:
            lg.error(f"{NEON_RED}Manual precision application failed: {manual_err}. Using unrounded size.{RESET}")
            final_size = adjusted_size  # Use unrounded as last resort

    # --- Final Validation after Precision ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Final size after precision is zero or negative ({final_size.normalize()}). Aborting sizing.{RESET}")
        return None
    if min_amount > 0 and final_size < min_amount:
        lg.error(f"{NEON_RED}Final size {final_size.normalize()} is below minimum amount {min_amount.normalize()} after precision adjustment. Aborting sizing.{RESET}")
        # Note: Could potentially bump up to min_amount here, but that might violate risk parameters. Aborting is safer.
        return None
    if max_amount < Decimal('inf') and final_size > max_amount:
         lg.error(f"{NEON_RED}Final size {final_size.normalize()} exceeds maximum amount {max_amount.normalize()} after precision. Aborting sizing.{RESET}")
         # This shouldn't happen if limits were applied correctly before precision, but check defensively.
         return None

    # Final check on cost after precision applied (especially if rounded down)
    final_cost = Decimal('0')
    try:
        if entry_price > 0:
            final_cost = (final_size * entry_price * contract_size) if not is_inverse else ((final_size * contract_size) / entry_price)
            lg.debug(f"  Final Estimated Cost: {final_cost.normalize()} {quote_currency}")
    except Exception: pass

    if min_cost > 0 and final_cost > 0 and final_cost < min_cost:
         lg.warning(f"{NEON_YELLOW}Final cost {final_cost.normalize()} is slightly below min cost {min_cost.normalize()} after precision rounding.{RESET}")
         # Option: Try bumping size up by one step if possible within other limits
         try:
             next_step_size = final_size + amount_precision_step
             next_step_cost = Decimal('0')
             if entry_price > 0:
                 next_step_cost = (next_step_size * entry_price * contract_size) if not is_inverse else ((next_step_size * contract_size) / entry_price)

             # Check if bumping up is valid (meets min cost, doesn't exceed max amount/cost)
             can_bump_up = (next_step_cost >= min_cost) and \
                           (max_amount == Decimal('inf') or next_step_size <= max_amount) and \
                           (max_cost == Decimal('inf') or next_step_cost <= max_cost)

             if can_bump_up:
                 lg.info(f"{NEON_YELLOW}Bumping final size up by one step to {next_step_size.normalize()} to meet minimum cost.{RESET}")
                 final_size = next_step_size
             else:
                 lg.error(f"{NEON_RED}Cannot meet minimum cost even by bumping size one step due to other limits. Aborting sizing.{RESET}")
                 return None
         except Exception as bump_err:
             lg.error(f"{NEON_RED}Error trying to bump size for min cost: {bump_err}. Aborting sizing.{RESET}")
             return None

    # --- Success ---
    lg.info(f"{NEON_GREEN}{BRIGHT}Final Calculated Position Size: {final_size.normalize()} {size_unit}{RESET}")
    lg.info("--- End Position Sizing ---")
    return final_size


def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: dict,
                logger: logging.Logger, reduce_only: bool = False, params: dict | None = None) -> dict | None:
    """Places a market order (buy or sell) using CCXT's create_order method.
    - Maps trade signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT") to order sides ("buy", "sell").
    - Handles Bybit V5 specific parameters (category, positionIdx, reduceOnly, timeInForce).
    - Includes retry logic for network/exchange errors and rate limits.
    - Logs order details clearly.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT:USDT").
        trade_signal: The signal driving the trade ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The calculated position size (must be positive).
        market_info: The market information dictionary.
        logger: The logger instance for status messages.
        reduce_only: Set to True for closing orders to ensure they only reduce/close a position.
        params: Optional additional parameters to pass to create_order.

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
    if position_size <= Decimal('0'):
        lg.error(f"Invalid position size provided to place_trade: {position_size}. Must be positive.")
        return None

    order_type = 'market'  # This bot currently only uses market orders for simplicity
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', 'BASE')
    settle_currency = market_info.get('settle', quote_currency) if is_contract else base_currency
    size_unit = settle_currency if is_contract else base_currency  # Display unit

    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info['id']  # Use the exchange-specific market ID

    # --- Prepare Order Arguments ---
    order_args = {
        'symbol': market_id,
        'type': order_type,
        'side': side,
        'amount': float(position_size),  # CCXT typically expects float amount
    }
    order_params = {}  # For exchange-specific parameters

    # --- Bybit V5 Specific Parameters ---
    if 'bybit' in exchange.id.lower():
        try:
            category = 'linear' if market_info.get('linear', True) else 'inverse'
            order_params = {
                'category': category,
                'positionIdx': 0  # Use 0 for one-way mode (required by Bybit V5 for non-hedge mode)
            }
            if reduce_only:
                order_params['reduceOnly'] = True
                # Use IOC for reduceOnly orders to prevent them resting if not immediately fillable
                order_params['timeInForce'] = 'IOC'  # Immediate Or Cancel
            lg.debug(f"Using Bybit V5 order params: {order_params}")
        except Exception as e:
            lg.error(f"Failed to set Bybit V5 parameters: {e}. Order might fail.")
            # Proceed cautiously without params if setting failed

    # Merge any additional custom parameters provided
    if params:
        order_params.update(params)

    if order_params:
        order_args['params'] = order_params  # Add exchange-specific params to the main args

    # Log the trade attempt
    lg.info(f"===> Attempting {action_desc} | {side.upper()} {order_type.upper()} Order | {symbol} | Size: {position_size.normalize()} {size_unit} <===")
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
            avg_price_str = format_decimal(order_result.get('average'))
            filled_amount_str = format_decimal(order_result.get('filled'))
            log_msg = (
                f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET} "
                f"ID: {order_id}, Status: {status}"
            )
            if avg_price_str != 'N/A': log_msg += f", Avg Fill Price: ~{avg_price_str}"
            if filled_amount_str != 'N/A': log_msg += f", Filled Amount: {filled_amount_str}"
            lg.info(log_msg)
            lg.debug(f"Full order result: {order_result}")
            return order_result  # Return the successful order details

        # --- Error Handling with Retries ---
        except ccxt.InsufficientFunds as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed: Insufficient funds for {symbol} {side} {position_size}. Error: {e}{RESET}")
            return None  # Non-retryable
        except ccxt.InvalidOrder as e:
            last_exception = e
            lg.error(f"{NEON_RED}Order Failed: Invalid order parameters for {symbol}. Error: {e}{RESET}")
            lg.error(f"  Order arguments used: {order_args}")
            # Could add hints based on error message (e.g., min size, precision)
            return None  # Non-retryable
        except ccxt.ExchangeError as e:
            last_exception = e
            err_code = getattr(e, 'code', None)
            lg.error(f"{NEON_RED}Order Failed: Exchange error placing order for {symbol}. Error: {e} (Code: {err_code}){RESET}")
            # Check for known fatal Bybit error codes related to orders
            # Example codes (add more as discovered):
            # 110014: Qty is too small; 110007: Qty exceed max limit; 110040: Order cost exceed limit
            # 110013: Parameter error; 110025: Position idx not match position mode; 30086: Reduce only check failed
            # 10001: Parameter error; 10004: Sign check error; 3303001: Invalid symbol; 3303005: Price out of range
            fatal_order_codes = [110014, 110007, 110040, 110013, 110025, 30086, 10001, 10004, 3303001, 3303005]
            if err_code in fatal_order_codes or "invalid parameter" in str(e).lower():
                lg.error(f"{NEON_RED} >> Hint: This appears to be a non-retryable order error.{RESET}")
                return None  # Non-retryable
            # Assume other exchange errors might be temporary and retry
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError placing order.{RESET}")
                 return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error placing order for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError placing order.{RESET}")
                return None

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
             last_exception = e
             lg.critical(f"{NEON_RED}Authentication Error placing order: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
             return None  # Fatal error

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        # Increment attempt counter (only if not a rate limit wait) and delay before retrying
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return None


def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict, logger: logging.Logger,
                             stop_loss_price: Decimal | None = None, take_profit_price: Decimal | None = None,
                             trailing_stop_distance: Decimal | None = None, tsl_activation_price: Decimal | None = None) -> bool:
    """Internal helper function to set Stop Loss (SL), Take Profit (TP), and/or
    Trailing Stop Loss (TSL) for an existing position using Bybit's V5 private API endpoint.
    This is necessary because CCXT's standard methods might not fully support Bybit's V5 TSL/SL/TP parameters together.

    **Important:** This uses a direct API call (`private_post`) and relies on Bybit's specific V5 endpoint and parameters.
                   It might break if Bybit changes its API structure.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol (e.g., "BTC/USDT:USDT").
        market_info: The market information dictionary.
        position_info: The dictionary containing details of the open position (from get_open_position).
        logger: The logger instance for status messages.
        stop_loss_price: The desired fixed SL price (Decimal). Set to 0 to clear existing SL.
        take_profit_price: The desired fixed TP price (Decimal). Set to 0 to clear existing TP.
        trailing_stop_distance: The desired TSL distance (in price units, Decimal). Set to 0 to clear TSL.
        tsl_activation_price: The price at which the TSL should activate (Decimal). Required if distance > 0.

    Returns:
        True if the protection was set/updated successfully or no change was needed, False otherwise.
    """
    lg = logger
    endpoint = '/v5/position/set-trading-stop'  # Bybit V5 endpoint for SL/TP/TSL

    # --- Input and State Validation ---
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False  # Cannot set SL/TP/TSL on spot
    if not position_info:
        lg.error(f"Protection setting failed for {symbol}: Missing position information.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"Protection setting failed for {symbol}: Invalid position side ('{pos_side}') or entry price ('{entry_price_str}').")
        return False
    try:
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed for {symbol}: Invalid entry price format or value '{entry_price_str}': {e}")
        return False

    params_to_set: dict[str, Any] = {}  # Parameters to send in the API request
    log_parts: list[str] = [f"Attempting to set protection for {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):"]
    any_protection_requested = False  # Flag to check if any valid protection was requested

    # --- Format and Validate Protection Parameters ---
    try:
        price_precision_str = market_info['precision']['price']
        min_tick = Decimal(str(price_precision_str))
        if min_tick <= 0: raise ValueError("Invalid price precision (tick size)")

        # Helper to format price to exchange precision
        def format_price_param(price_decimal: Decimal | None, param_name: str) -> str | None:
            """Formats price to string, respecting exchange precision. Returns None if invalid."""
            if price_decimal is None: return None
            if price_decimal == 0: return "0"  # Allow "0" to clear protection
            if price_decimal < 0:
                lg.warning(f"Invalid negative price {price_decimal} provided for {param_name}. Ignoring.")
                return None
            try:
                # Use CCXT's price_to_precision for correct rounding/truncating
                formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal))
                # Final check: ensure formatted price is still positive
                return formatted_str if Decimal(formatted_str) > 0 else None
            except Exception as e:
                lg.error(f"Failed to format {param_name} value {price_decimal} to exchange precision: {e}.")
                return None

        # --- Trailing Stop Loss (TSL) ---
        # Bybit requires TSL distance (trailingStop) and activation price (activePrice)
        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal):
            any_protection_requested = True
            if trailing_stop_distance > 0:  # Setting an active TSL
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0:
                    lg.error(f"TSL request failed: Valid activation price ({tsl_activation_price}) is required when TSL distance ({trailing_stop_distance}) > 0.")
                else:
                    # Validate activation price makes sense relative to entry
                    is_valid_activation = (pos_side == 'long' and tsl_activation_price > entry_price) or \
                                          (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_activation:
                        lg.error(f"TSL request failed: Activation price {tsl_activation_price.normalize()} must be beyond entry price {entry_price.normalize()} for a {pos_side} position.")
                    else:
                        # Ensure distance is at least one tick
                        min_valid_distance = max(trailing_stop_distance, min_tick)
                        fmt_distance = format_price_param(min_valid_distance, "TSL Distance")
                        fmt_activation = format_price_param(tsl_activation_price, "TSL Activation Price")

                        if fmt_distance and fmt_activation:
                            params_to_set['trailingStop'] = fmt_distance
                            params_to_set['activePrice'] = fmt_activation
                            log_parts.append(f"  - Setting TSL: Distance={fmt_distance}, Activation={fmt_activation}")
                            set_tsl = True  # Mark TSL as being set
                        else:
                            lg.error(f"TSL request failed: Could not format TSL parameters (Distance: {fmt_distance}, Activation: {fmt_activation}).")

            elif trailing_stop_distance == 0:  # Clearing TSL
                params_to_set['trailingStop'] = "0"
                # Also clear activation price when clearing TSL distance
                params_to_set['activePrice'] = "0"
                log_parts.append("  - Clearing TSL (Distance and Activation Price set to 0)")
                set_tsl = True  # Mark TSL action (clearing) as requested

        # --- Fixed Stop Loss (SL) ---
        # Can only set fixed SL if TSL is *not* being actively set (Bybit limitation)
        if not (set_tsl and trailing_stop_distance > 0):  # Allow SL clear even if TSL clear requested
            if isinstance(stop_loss_price, Decimal):
                any_protection_requested = True
                if stop_loss_price > 0:  # Setting an active SL
                    # Validate SL price makes sense relative to entry
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
                elif stop_loss_price == 0:  # Clearing SL
                    params_to_set['stopLoss'] = "0"
                    log_parts.append("  - Clearing Fixed SL (set to 0)")

        # --- Fixed Take Profit (TP) ---
        # TP can usually be set alongside SL or TSL
        if isinstance(take_profit_price, Decimal):
            any_protection_requested = True
            if take_profit_price > 0:  # Setting an active TP
                # Validate TP price makes sense relative to entry
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
            elif take_profit_price == 0:  # Clearing TP
                params_to_set['takeProfit'] = "0"
                log_parts.append("  - Clearing Fixed TP (set to 0)")

    except Exception as fmt_err:
        lg.error(f"Error during protection parameter formatting/validation: {fmt_err}", exc_info=True)
        return False

    # Check if any valid parameters were actually prepared for the API call
    if not params_to_set:
        if any_protection_requested:
            lg.warning(f"No valid protection parameters to set for {symbol} after formatting/validation. No API call made.")
            # Return False because the requested action couldn't be fulfilled
            return False
        else:
            lg.debug(f"No protection changes requested for {symbol}. No API call needed.")
            return True  # Success, as no action was needed

    # --- Prepare Final API Parameters ---
    # Determine category and market ID
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    market_id = market_info['id']
    # Get position index (should be 0 for one-way mode)
    position_idx = 0
    try:
        # Attempt to get positionIdx from position info if available (hedge mode?)
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
    except (ValueError, TypeError):
        lg.warning("Could not parse positionIdx from position info, defaulting to 0 (one-way mode).")
        position_idx = 0

    # Construct the final parameters dictionary for the API call
    final_api_params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full',  # Use 'Full' for setting SL/TP on the entire position ('Partial' for partial TP/SL not used here)
        'slTriggerBy': 'LastPrice',  # Trigger SL based on Last Price (options: MarkPrice, IndexPrice)
        'tpTriggerBy': 'LastPrice',  # Trigger TP based on Last Price
        'slOrderType': 'Market',    # Use Market order when SL is triggered
        'tpOrderType': 'Market',    # Use Market order when TP is triggered
        'positionIdx': position_idx  # Specify position index (0 for one-way)
    }
    final_api_params.update(params_to_set)  # Add the specific SL/TP/TSL values

    lg.info("\n".join(log_parts))  # Log what is being attempted
    lg.debug(f"  Final API parameters for {endpoint}: {final_api_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    last_exception = None
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing private_post to {endpoint} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            # Use exchange.private_post for endpoints not directly mapped by CCXT
            response = exchange.private_post(endpoint, params=final_api_params)
            lg.debug(f"Set protection raw API response: {response}")

            # --- Check Bybit V5 Response ---
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown message')

            if ret_code == 0:
                 # Check message for "not modified" cases which are still success
                 if any(m in ret_msg.lower() for m in ["not modified", "no need to modify", "parameter not change"]):
                     lg.info(f"{NEON_YELLOW}Protection parameters already set or no change needed. (Message: {ret_msg}){RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection parameters successfully set/updated for {symbol}.{RESET}")
                 return True  # Success

            else:
                 # Log the specific Bybit error
                 lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}){RESET}")
                 # Check for known fatal/non-retryable error codes
                 # Example codes (add more as discovered):
                 # 110013: Parameter error; 110036: SL price cannot be higher/lower than entry
                 # 110086: TP price cannot be lower/higher than entry; 110084: SL price invalid
                 # 110085: TP price invalid; 10001: Parameter error; 10002: API key invalid
                 fatal_protect_codes = [110013, 110036, 110086, 110084, 110085, 10001, 10002, 110043]  # 110043 = Position closed
                 is_fatal = ret_code in fatal_protect_codes or \
                            "invalid" in ret_msg.lower() or \
                            "parameter" in ret_msg.lower() or \
                            "cannot be" in ret_msg.lower()  # General phrases indicating invalid input

                 if is_fatal:
                     lg.error(f"{NEON_RED} >> Hint: This appears to be a non-retryable protection setting error.{RESET}")
                     return False  # Fatal error, do not retry
                 else:
                     # Raise an ExchangeError for other codes to trigger retry logic
                     raise ccxt.ExchangeError(f"Bybit API error setting protection: {ret_msg} (Code: {ret_code})")

        # --- Standard CCXT Error Handling with Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            last_exception = e
            lg.warning(f"{NEON_YELLOW}Network error setting protection for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Maximum retries exceeded for NetworkError setting protection.{RESET}")
                return False

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = RETRY_DELAY_SECONDS * 3
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded setting protection for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Continue loop without incrementing attempts

        except ccxt.AuthenticationError as e:
            last_exception = e
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Check API Key/Secret/Permissions. Stopping.{RESET}")
            return False  # Fatal error

        except ccxt.ExchangeError as e:  # Catch re-raised Bybit errors or other CCXT exchange errors
             last_exception = e
             lg.warning(f"{NEON_YELLOW}Exchange error setting protection for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
             if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Maximum retries exceeded for ExchangeError setting protection.{RESET}")
                 return False

        except Exception as e:
            last_exception = e
            lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol} (Attempt {attempts + 1}): {e}{RESET}", exc_info=True)
            # Unexpected errors are likely fatal for this operation
            return False

        # Increment attempt counter and delay before retrying (only for retryable errors)
        attempts += 1
        time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts. Last error: {last_exception}{RESET}")
    return False


def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict, config: dict[str, Any],
                             logger: logging.Logger, take_profit_price: Decimal | None = None) -> bool:
    """Calculates Trailing Stop Loss (TSL) parameters based on configuration settings
    (callback rate, activation percentage) and the current position's entry price.
    It then calls the internal `_set_position_protection` function to apply the TSL.

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol.
        market_info: The market information dictionary.
        position_info: The dictionary containing details of the open position.
        config: The main configuration dictionary containing protection settings.
        logger: The logger instance.
        take_profit_price: An optional fixed Take Profit price to set simultaneously (Decimal).
                           Set to 0 to clear existing TP.

    Returns:
        True if the TSL (and optional TP) was successfully calculated and the API call
        to set protection was initiated successfully, False otherwise.
    """
    lg = logger
    prot_cfg = config.get("protection", {})  # Get protection sub-dictionary

    # --- Input Validation ---
    if not market_info or not position_info:
        lg.error(f"TSL calculation failed for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"TSL calculation failed for {symbol}: Invalid position side ('{pos_side}') or entry price ('{entry_price_str}').")
        return False

    try:
        # Extract parameters and convert to Decimal
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(prot_cfg["trailing_stop_callback_rate"]))  # e.g., 0.005 for 0.5%
        activation_percentage = Decimal(str(prot_cfg["trailing_stop_activation_percentage"]))  # e.g., 0.003 for 0.3%
        price_tick_str = market_info['precision']['price']
        price_tick = Decimal(str(price_tick_str))

        # Validate parameters
        if not (entry_price > 0 and callback_rate > 0 and activation_percentage >= 0 and price_tick > 0):
             raise ValueError("Invalid input values (entry, callback rate, activation %, or tick size).")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"TSL calculation failed for {symbol}: Invalid configuration or market info: {e}")
        return False

    # --- Calculate TSL Activation Price and Distance ---
    try:
        lg.info(f"Calculating TSL for {symbol} ({pos_side.upper()}): Entry={entry_price.normalize()}, Act%={activation_percentage:.3%}, CB%={callback_rate:.3%}")

        # 1. Calculate Activation Price
        activation_offset = entry_price * activation_percentage
        raw_activation_price = (entry_price + activation_offset) if pos_side == 'long' else (entry_price - activation_offset)

        # Quantize activation price to the nearest tick (away from entry)
        if pos_side == 'long':
            # Round up for long activation
            quantized_activation_price = raw_activation_price.quantize(price_tick, ROUND_UP)
            # Ensure activation is strictly greater than entry by at least one tick
            tsl_activation_price = max(quantized_activation_price, entry_price + price_tick)
        else:  # Short position
            # Round down for short activation
            quantized_activation_price = raw_activation_price.quantize(price_tick, ROUND_DOWN)
            # Ensure activation is strictly less than entry by at least one tick
            tsl_activation_price = min(quantized_activation_price, entry_price - price_tick)

        if tsl_activation_price <= 0:
            lg.error(f"TSL calculation failed: Calculated Activation Price ({tsl_activation_price.normalize()}) is zero or negative.")
            return False

        # 2. Calculate Trailing Distance (based on activation price and callback rate)
        # Distance = Activation Price * Callback Rate
        raw_trailing_distance = tsl_activation_price * callback_rate

        # Quantize distance UP to the nearest tick and ensure it's at least one tick
        tsl_trailing_distance = max(raw_trailing_distance.quantize(price_tick, ROUND_UP), price_tick)

        if tsl_trailing_distance <= 0:
            lg.error(f"TSL calculation failed: Calculated Trailing Distance ({tsl_trailing_distance.normalize()}) is zero or negative.")
            return False

        lg.info(f"  => Calculated TSL Activation Price: {tsl_activation_price.normalize()}")
        lg.info(f"  => Calculated TSL Trailing Distance: {tsl_trailing_distance.normalize()}")
        if isinstance(take_profit_price, Decimal):
             tp_action = f"{take_profit_price.normalize()}" if take_profit_price != 0 else "Clear TP"
             lg.info(f"  => Also setting/clearing Fixed TP: {tp_action}")

        # --- Call Internal Function to Set Protection ---
        # Pass None for fixed stop_loss_price as TSL takes precedence
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None,  # TSL overrides fixed SL when active
            take_profit_price=take_profit_price,  # Pass through TP setting/clearing
            trailing_stop_distance=tsl_trailing_distance,
            tsl_activation_price=tsl_activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during TSL calculation or setting: {e}{RESET}", exc_info=True)
        return False


# --- Volumatic Trend + OB Strategy Implementation ---
class OrderBlock(TypedDict):
    """Represents a bullish or bearish Order Block identified on the chart."""
    id: str                 # Unique identifier (e.g., "B_231026143000")
    type: str               # 'bull' or 'bear'
    left_idx: pd.Timestamp  # Timestamp of the candle that formed the OB
    right_idx: pd.Timestamp  # Timestamp of the last candle the OB extends to (or violation candle)
    top: Decimal            # Top price level of the OB
    bottom: Decimal         # Bottom price level of the OB
    active: bool            # True if the OB is currently considered valid
    violated: bool          # True if the price has closed beyond the OB boundary


class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis process."""
    dataframe: pd.DataFrame         # The DataFrame with all calculated indicators
    last_close: Decimal             # The closing price of the most recent candle
    current_trend_up: bool | None  # True if Volumatic Trend is up, False if down, None if undetermined
    trend_just_changed: bool        # True if the trend flipped on the last candle
    active_bull_boxes: list[OrderBlock]  # List of currently active bullish OBs
    active_bear_boxes: list[OrderBlock]  # List of currently active bearish OBs
    vol_norm_int: int | None     # Normalized volume indicator (0-100+, integer) for the last candle
    atr: Decimal | None          # ATR value for the last candle
    upper_band: Decimal | None   # Volumatic Trend upper band value for the last candle
    lower_band: Decimal | None   # Volumatic Trend lower band value for the last candle


class VolumaticOBStrategy:
    """Implements the core logic for the Volumatic Trend and Pivot Order Block strategy.
    - Calculates Volumatic Trend indicators (EMAs, ATR Bands, Volume Normalization).
    - Identifies Pivot Highs/Lows based on configuration.
    - Creates Order Blocks (OBs) from pivots.
    - Manages the state of OBs (active, violated, extends).
    - Prunes the list of active OBs to a maximum number.
    """
    def __init__(self, config: dict[str, Any], market_info: dict[str, Any], logger: logging.Logger) -> None:
        """Initializes the strategy engine with parameters from the config.

        Args:
            config: The main configuration dictionary.
            market_info: The market information dictionary.
            logger: The logger instance for this strategy instance.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger  # Alias for convenience

        strategy_cfg = config.get("strategy_params", {})
        config.get("protection", {})  # Needed for context but not directly used here

        # Load strategy parameters from config, using defaults if missing/invalid
        # (Validation should have happened in load_config, but use .get for safety)
        self.vt_length = int(strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH))
        self.vt_atr_period = int(strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD))
        self.vt_vol_ema_length = int(strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH))
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))

        self.ob_source = str(strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE))
        self.ph_left = int(strategy_cfg.get("ph_left", DEFAULT_PH_LEFT))
        self.ph_right = int(strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT))
        self.pl_left = int(strategy_cfg.get("pl_left", DEFAULT_PL_LEFT))
        self.pl_right = int(strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT))
        self.ob_extend = bool(strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND))
        self.ob_max_boxes = int(strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES))

        # Initialize Order Block storage
        self.bull_boxes: list[OrderBlock] = []
        self.bear_boxes: list[OrderBlock] = []

        # Calculate minimum data length required based on longest lookback period
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length)  # Need enough data for longest VT calc
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1)  # Pivot needs left+right+current candle
        self.min_data_len = max(required_for_vt, required_for_pivots) + 50  # Add a buffer

        # Log initialized parameters
        self.lg.info(f"{NEON_CYAN}--- Initializing VolumaticOB Strategy Engine ---{RESET}")
        self.lg.info(f"  VT Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Vol EMA Length={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  OB Params: Source='{self.ob_source}', PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend OBs={self.ob_extend}, Max Active OBs={self.ob_max_boxes}")
        self.lg.info(f"  Minimum Historical Data Required: {self.min_data_len} candles")

        # Warning if required data exceeds typical API limits
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 10:  # Add small buffer to limit check
            self.lg.error(f"{NEON_RED}{BRIGHT}CONFIGURATION WARNING:{RESET} Strategy requires {self.min_data_len} candles, which may exceed the API fetch limit ({BYBIT_API_KLINE_LIMIT}). "
                          f"Consider reducing lookback periods (vt_atr_period, vt_vol_ema_length, ph/pl_left/right) in config.json.{RESET}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates the Smoothed Weighted Moving Average (SWMA) of length 4,
        followed by an Exponential Moving Average (EMA) of the SWMA result.
        SWMA(4) weights: [1, 2, 2, 1] / 6.

        Args:
            series: The input Pandas Series (typically 'close' prices).
            length: The length for the final EMA calculation.

        Returns:
            A Pandas Series containing the EMA(SWMA(series, 4), length).
        """
        if not isinstance(series, pd.Series) or len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index)  # Return NaNs if input is invalid

        # Define weights for SWMA(4)
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        # Convert series to numeric, coercing errors
        numeric_series = pd.to_numeric(series, errors='coerce')

        if numeric_series.isnull().all():  # Check if all values are NaN after conversion
            return pd.Series(np.nan, index=series.index)

        # Calculate SWMA using rolling apply with dot product
        # min_periods=4 ensures we only calculate where 4 data points are available
        swma = numeric_series.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate EMA of the SWMA result using pandas_ta
        # fillna=np.nan prevents forward-filling issues if EMA starts later
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan)
        return ema_of_swma

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, is_high: bool) -> pd.Series:
        """Identifies Pivot Highs or Pivot Lows in a series using a rolling window comparison.
        A Pivot High is a value higher than `left_bars` values to its left and `right_bars` values to its right.
        A Pivot Low is a value lower than `left_bars` values to its left and `right_bars` values to its right.

        Args:
            series: The Pandas Series to search for pivots (e.g., 'high' or 'low' prices).
            left_bars: The number of bars to look back to the left.
            right_bars: The number of bars to look forward to the right.
            is_high: True to find Pivot Highs, False to find Pivot Lows.

        Returns:
            A boolean Pandas Series, True where a pivot point is identified.
        """
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1:
            return pd.Series(False, index=series.index)  # Return all False if input invalid

        # Convert to numeric, coercing errors
        num_series = pd.to_numeric(series, errors='coerce')
        if num_series.isnull().all():
            return pd.Series(False, index=series.index)

        # Initialize all points as potential pivots
        pivot_conditions = pd.Series(True, index=series.index)

        # Check left bars: current >(or <) previous bars
        for i in range(1, left_bars + 1):
            shifted = num_series.shift(i)
            if is_high:
                pivot_conditions &= (num_series > shifted)
            else:
                pivot_conditions &= (num_series < shifted)

        # Check right bars: current >(or <) future bars
        for i in range(1, right_bars + 1):
            shifted = num_series.shift(-i)
            if is_high:
                pivot_conditions &= (num_series > shifted)
            else:
                pivot_conditions &= (num_series < shifted)

        # Fill NaN results (from edges where shifts produce NaN) with False
        return pivot_conditions.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """Processes the historical OHLCV data to calculate all strategy indicators,
        identify pivots, create/update Order Blocks, and return the analysis results.

        Args:
            df_input: The input Pandas DataFrame containing OHLCV data (index=Timestamp, columns=Decimal).

        Returns:
            A StrategyAnalysisResults TypedDict containing the processed DataFrame,
            current trend, active OBs, and other key indicator values for the latest candle.
            Returns empty results if processing fails.
        """
        # Prepare an empty result structure for failure cases
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
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
            self.lg.error("Strategy update failed: DataFrame index is not a monotonic DatetimeIndex.")
            return empty_results
        if len(df) < self.min_data_len:
            # Proceed but warn if data is insufficient
            self.lg.warning(f"Strategy update: Insufficient data ({len(df)} candles < {self.min_data_len} required). Results may be inaccurate or incomplete.")
        self.lg.debug(f"Starting strategy analysis on {len(df)} candles (minimum required: {self.min_data_len}).")

        # --- Convert to Float for TA Libraries ---
        # pandas_ta and numpy work best with floats. We'll convert back to Decimal later.
        try:
            df_float = pd.DataFrame(index=df.index)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.lg.error(f"Strategy update failed: Missing required column '{col}' in input DataFrame.")
                    return empty_results
                df_float[col] = pd.to_numeric(df[col], errors='coerce')  # Convert Decimal/Object to float

            # Drop rows where essential float columns became NaN (shouldn't happen if input is clean)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
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
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)  # EMA(SWMA(close))
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)  # Standard EMA(close)

            # Determine Trend Direction (ema2 crosses ema1.shift(1))
            # Use boolean directly, fillna(False) assumes no trend at start if NaNs exist
            trend_up_series = (df_float['ema2'] > df_float['ema1'].shift(1)).fillna(False)
            df_float['trend_up'] = trend_up_series

            # Identify Trend Changes
            trend_changed_series = (df_float['trend_up'].shift(1) != df_float['trend_up']) & \
                                   df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'] = trend_changed_series.fillna(False)

            # Capture EMA1 and ATR values at the exact point the trend changed
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these values to get the relevant EMA/ATR for the current trend segment
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Volumatic Trend Bands
            atr_multiplier_float = float(self.vt_atr_multiplier)
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float)

            # Volume Normalization
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            # Use a reasonable min_periods for the rolling max to avoid NaN at the start
            min_periods_vol = max(1, self.vt_vol_ema_length // 10)
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=min_periods_vol).max().fillna(0.0)
            # Calculate normalized volume (0-100 range, potentially > 100 if current vol > max in period)
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9,  # Avoid division by zero
                                            (volume_numeric / df_float['vol_max'] * 100.0),
                                            0.0)  # Assign 0 if max volume is zero
            # Handle potential NaNs and clip unreasonable values (e.g., > 200%)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)

            # Pivot High/Low Calculation
            # Select source series based on config ('Wicks' or 'Body')
            if self.ob_source.lower() == "wicks":
                high_series = df_float['high']
                low_series = df_float['low']
            else:  # Use candle body
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)
            # Find pivots using the helper function
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, is_high=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, is_high=False)

            self.lg.debug("Indicator calculations complete.")

        except Exception as e:
            self.lg.error(f"Strategy update failed: Error during indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy Calculated Float Results back to Original Decimal DataFrame ---
        try:
            self.lg.debug("Converting calculated indicators back to Decimal format...")
            indicator_cols = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                              'upper_band', 'lower_band', 'vol_norm', 'is_ph', 'is_pl']
            for col in indicator_cols:
                if col in df_float.columns:
                    # Reindex to align with original Decimal DataFrame's index (handles NaNs introduced)
                    source_series = df_float[col].reindex(df.index)
                    if source_series.dtype == 'bool':
                        df[col] = source_series.astype(bool)  # Keep booleans as bool
                    elif pd.api.types.is_object_dtype(source_series):
                         # Should not happen for these indicators, but handle defensively
                         df[col] = source_series
                    else:
                        # Convert numeric (float) back to Decimal, preserving NaNs
                        df[col] = source_series.apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                        )
        except Exception as e:
            self.lg.error(f"Strategy update failed: Error converting calculated indicators back to Decimal: {e}", exc_info=True)
            # Don't return yet, try to proceed with OB management if possible

        # --- Clean Final Decimal DataFrame ---
        # Drop rows where essential indicators might be NaN (e.g., at the start of the series)
        initial_len_final = len(df)
        required_indicator_cols = ['close', 'atr', 'trend_up', 'upper_band', 'lower_band', 'is_ph', 'is_pl']
        df.dropna(subset=required_indicator_cols, inplace=True)
        rows_dropped_final = initial_len_final - len(df)
        if rows_dropped_final > 0:
            self.lg.debug(f"Dropped {rows_dropped_final} rows from final DataFrame due to missing essential indicators (likely at start).")

        if df.empty:
            self.lg.warning("Strategy update: DataFrame became empty after final indicator cleaning. Cannot process Order Blocks.")
            # Return empty results, but use the (empty) df
            empty_results['dataframe'] = df
            return empty_results

        self.lg.debug("Indicators finalized in Decimal DataFrame. Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_ob_count = 0
            last_candle_idx = df.index[-1]

            # 1. Identify New Order Blocks from Pivots
            for timestamp, candle in df.iterrows():
                try:
                    is_pivot_high = candle.get('is_ph', False)
                    is_pivot_low = candle.get('is_pl', False)

                    # Create Bearish OB from Pivot High
                    if is_pivot_high:
                        # Check if an OB from this exact candle already exists
                        if not any(ob['left_idx'] == timestamp and ob['type'] == 'bear' for ob in self.bear_boxes):
                            # Determine top/bottom based on OB source config
                            if self.ob_source.lower() == 'wicks':
                                ob_top = candle.get('high')
                                ob_bottom = candle.get('open')  # Bearish OB uses high and open of PH candle wick
                            else:  # Body
                                ob_top = max(candle.get('open'), candle.get('close'))
                                ob_bottom = min(candle.get('open'), candle.get('close'))

                            # Ensure valid Decimal values and top > bottom
                            if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom and pd.notna(ob_top) and pd.notna(ob_bottom):
                                ob_id = f"B_{timestamp.strftime('%y%m%d%H%M%S')}"  # Unique ID
                                new_bear_ob = OrderBlock(
                                    id=ob_id, type='bear', left_idx=timestamp, right_idx=last_candle_idx,
                                    top=ob_top, bottom=ob_bottom, active=True, violated=False
                                )
                                self.bear_boxes.append(new_bear_ob)
                                new_ob_count += 1
                                self.lg.debug(f"  + New Bearish OB identified: {ob_id} @ {timestamp.strftime('%H:%M')} [Top:{ob_top.normalize()}, Bot:{ob_bottom.normalize()}]")
                            else:
                                self.lg.warning(f"Could not create Bear OB at {timestamp}: Invalid top/bottom values (Top: {ob_top}, Bottom: {ob_bottom}).")

                    # Create Bullish OB from Pivot Low
                    if is_pivot_low:
                        if not any(ob['left_idx'] == timestamp and ob['type'] == 'bull' for ob in self.bull_boxes):
                            if self.ob_source.lower() == 'wicks':
                                ob_top = candle.get('open')  # Bullish OB uses open and low of PL candle wick
                                ob_bottom = candle.get('low')
                            else:  # Body
                                ob_top = max(candle.get('open'), candle.get('close'))
                                ob_bottom = min(candle.get('open'), candle.get('close'))

                            if isinstance(ob_top, Decimal) and isinstance(ob_bottom, Decimal) and ob_top > ob_bottom and pd.notna(ob_top) and pd.notna(ob_bottom):
                                ob_id = f"L_{timestamp.strftime('%y%m%d%H%M%S')}"
                                new_bull_ob = OrderBlock(
                                    id=ob_id, type='bull', left_idx=timestamp, right_idx=last_candle_idx,
                                    top=ob_top, bottom=ob_bottom, active=True, violated=False
                                )
                                self.bull_boxes.append(new_bull_ob)
                                new_ob_count += 1
                                self.lg.debug(f"  + New Bullish OB identified: {ob_id} @ {timestamp.strftime('%H:%M')} [Top:{ob_top.normalize()}, Bot:{ob_bottom.normalize()}]")
                            else:
                                self.lg.warning(f"Could not create Bull OB at {timestamp}: Invalid top/bottom values (Top: {ob_top}, Bottom: {ob_bottom}).")

                except Exception as pivot_proc_err:
                    self.lg.warning(f"Error processing potential pivot at {timestamp}: {pivot_proc_err}", exc_info=True)

            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new Order Blocks in this update.")

            # 2. Manage Existing Order Blocks (Check Violations, Extend)
            last_candle = df.iloc[-1]
            last_close = last_candle.get('close')

            if isinstance(last_close, Decimal) and pd.notna(last_close):
                for box in self.bull_boxes:
                    if box['active']:
                        # Violation check: Close below the bottom of a bull OB
                        if last_close < box['bottom']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_candle_idx  # Mark violation time
                            self.lg.debug(f"Bullish OB {box['id']} VIOLATED by close {last_close.normalize()} < {box['bottom'].normalize()}")
                        # Extend active OB to current candle if enabled
                        elif self.ob_extend:
                            box['right_idx'] = last_candle_idx

                for box in self.bear_boxes:
                    if box['active']:
                        # Violation check: Close above the top of a bear OB
                        if last_close > box['top']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_candle_idx
                            self.lg.debug(f"Bearish OB {box['id']} VIOLATED by close {last_close.normalize()} > {box['top'].normalize()}")
                        # Extend active OB
                        elif self.ob_extend:
                            box['right_idx'] = last_candle_idx
            else:
                self.lg.warning("Cannot check Order Block violations: Invalid last close price.")

            # 3. Prune Order Blocks (Keep only the most recent 'ob_max_boxes' active ones)
            # Filter only active boxes, sort by creation time (descending), take the top N
            self.bull_boxes = sorted([b for b in self.bull_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted([b for b in self.bear_boxes if b['active']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.lg.debug(f"Pruned Order Blocks. Kept Active: Bulls={len(self.bull_boxes)}, Bears={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"Strategy update failed: Error during Order Block processing: {e}", exc_info=True)
            # Continue to return results, but OBs might be inaccurate

        # --- Prepare Final StrategyAnalysisResults ---
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper to safely extract Decimal values from the last candle
        def safe_decimal_from_candle(candle_data: pd.Series | None, col_name: str, positive_only: bool = False) -> Decimal | None:
            if candle_data is None: return None
            value = candle_data.get(col_name)
            if isinstance(value, Decimal) and pd.notna(value) and np.isfinite(float(value)):
                 if not positive_only or value > Decimal('0'):
                     return value
            return None

        # Construct the results dictionary
        analysis_results = StrategyAnalysisResults(
            dataframe=df,  # Return the fully processed DataFrame
            last_close=safe_decimal_from_candle(last_candle_final, 'close') or Decimal('0'),
            current_trend_up=bool(last_candle_final['trend_up']) if last_candle_final is not None and isinstance(last_candle_final.get('trend_up'), (bool, np.bool_)) else None,
            trend_just_changed=bool(last_candle_final['trend_changed']) if last_candle_final is not None and isinstance(last_candle_final.get('trend_changed'), (bool, np.bool_)) else False,
            active_bull_boxes=self.bull_boxes,  # Return the pruned list of active OBs
            active_bear_boxes=self.bear_boxes,
            vol_norm_int=int(v) if (v := safe_decimal_from_candle(last_candle_final, 'vol_norm')) is not None else None,
            atr=safe_decimal_from_candle(last_candle_final, 'atr', positive_only=True),
            upper_band=safe_decimal_from_candle(last_candle_final, 'upper_band'),
            lower_band=safe_decimal_from_candle(last_candle_final, 'lower_band')
        )

        # Log summary of the final results
        trend_str = f"{NEON_GREEN}UP{RESET}" if analysis_results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if analysis_results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{analysis_results['atr'].normalize()}" if analysis_results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"

        self.lg.debug(f"--- Strategy Analysis Results ({time_str}) ---")
        self.lg.debug(f"  Last Close: {analysis_results['last_close'].normalize()}")
        self.lg.debug(f"  Trend: {trend_str} (Changed: {analysis_results['trend_just_changed']})")
        self.lg.debug(f"  ATR: {atr_str}")
        self.lg.debug(f"  Volume Norm: {analysis_results['vol_norm_int']}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(analysis_results['active_bull_boxes'])} / {len(analysis_results['active_bear_boxes'])}")
        self.lg.debug("---------------------------------------------")

        return analysis_results


# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    based on the results from the VolumaticOBStrategy and the current position state.
    Also calculates initial Stop Loss (SL) and Take Profit (TP) levels for new entries.
    """
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        """Initializes the Signal Generator with parameters from the config.

        Args:
            config: The main configuration dictionary.
            logger: The logger instance.
        """
        self.config = config
        self.logger = logger
        self.lg = logger  # Alias
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        try:
            # Load parameters used for signal generation and SL/TP calculation
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get("ob_entry_proximity_factor", "1.005")))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get("ob_exit_proximity_factor", "1.001")))
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg.get("initial_take_profit_atr_multiple", "0.7")))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg.get("initial_stop_loss_atr_multiple", "1.8")))

            # Basic validation
            if not (self.ob_entry_proximity_factor >= 1): raise ValueError("ob_entry_proximity_factor must be >= 1.0")
            if not (self.ob_exit_proximity_factor >= 1): raise ValueError("ob_exit_proximity_factor must be >= 1.0")
            if not (self.initial_tp_atr_multiple >= 0): raise ValueError("initial_take_profit_atr_multiple must be >= 0")
            if not (self.initial_sl_atr_multiple > 0): raise ValueError("initial_stop_loss_atr_multiple must be > 0")

            self.lg.info("--- Initializing Signal Generator ---")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor:.3f}")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor:.3f}")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")
            self.lg.info("-----------------------------------")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.error(f"{NEON_RED}Error initializing SignalGenerator parameters from config: {e}. Using hardcoded defaults.{RESET}", exc_info=True)
             # Hardcoded defaults as fallback
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: dict | None) -> str:
        """Determines the trading signal based on strategy analysis and current position.
        Logic:
        1. Check for Exit conditions if a position is open.
        2. Check for Entry conditions if no position is open.
        3. Default to HOLD if no entry/exit conditions met.

        Args:
            analysis_results: The results from VolumaticOBStrategy.update().
            open_position: The current open position dictionary (from get_open_position), or None.

        Returns:
            The generated signal string: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "HOLD".
        """
        lg = self.logger

        # --- Validate Input ---
        if not analysis_results or analysis_results['current_trend_up'] is None or \
           analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
            lg.warning(f"{NEON_YELLOW}Signal Generation: Invalid or incomplete strategy analysis results provided. Defaulting to HOLD.{RESET}")
            return "HOLD"

        # Extract key values for easier access
        last_close = analysis_results['last_close']
        trend_is_up = analysis_results['current_trend_up']
        trend_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        position_side = open_position.get('side') if open_position else None

        signal: str = "HOLD"  # Default signal

        lg.debug("--- Signal Generation Check ---")
        lg.debug(f"  Input: Close={last_close.normalize()}, TrendUp={trend_is_up}, TrendChanged={trend_changed}, Position={position_side or 'None'}")
        lg.debug(f"  Active OBs: Bull={len(active_bull_obs)}, Bear={len(active_bear_obs)}")

        # --- 1. Check Exit Conditions (if position exists) ---
        if position_side == 'long':
            # Exit Long if trend flips down
            if trend_is_up is False and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped to DOWN on last candle.{RESET}")
            # Exit Long if price nears a bearish OB
            elif signal == "HOLD" and active_bear_obs:  # Check only if not already exiting
                try:
                    # Find the closest bear OB (based on top edge)
                    closest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - last_close))
                    # Exit threshold: Price >= OB Top * Exit Proximity Factor
                    exit_threshold = closest_bear_ob['top'] * self.ob_exit_proximity_factor
                    if last_close >= exit_threshold:
                        signal = "EXIT_LONG"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price {last_close.normalize()} >= Bear OB exit threshold {exit_threshold.normalize()} (OB ID: {closest_bear_ob['id']}, Top: {closest_bear_ob['top'].normalize()}){RESET}")
                except Exception as e:
                    lg.warning(f"Error during Bearish OB exit check for long position: {e}")

        elif position_side == 'short':
            # Exit Short if trend flips up
            if trend_is_up is True and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped to UP on last candle.{RESET}")
            # Exit Short if price nears a bullish OB
            elif signal == "HOLD" and active_bull_obs:
                try:
                    # Find the closest bull OB (based on bottom edge)
                    closest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - last_close))
                    # Exit threshold: Price <= OB Bottom / Exit Proximity Factor
                    # Avoid division by zero if factor is 1.0 (shouldn't happen based on validation)
                    exit_threshold = closest_bull_ob['bottom'] / self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else closest_bull_ob['bottom']
                    if last_close <= exit_threshold:
                        signal = "EXIT_SHORT"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price {last_close.normalize()} <= Bull OB exit threshold {exit_threshold.normalize()} (OB ID: {closest_bull_ob['id']}, Bottom: {closest_bull_ob['bottom'].normalize()}){RESET}")
                except (ZeroDivisionError, InvalidOperation, Exception) as e:
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
                    # Entry zone: OB Bottom <= Price <= OB Top * Entry Proximity Factor
                    entry_zone_bottom = ob['bottom']
                    entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "BUY"
                        lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price {last_close.normalize()} within Bull OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                        break  # Take the first valid entry signal

            # Check for SELL signal: Trend is DOWN and price is within a Bearish OB's proximity
            elif trend_is_up is False and active_bear_obs:
                for ob in active_bear_obs:
                    # Entry zone: OB Bottom / Entry Proximity Factor <= Price <= OB Top
                    entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']
                    entry_zone_top = ob['top']
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "SELL"
                        lg.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price {last_close.normalize()} within Bear OB entry zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}] (OB ID: {ob['id']}){RESET}")
                        break  # Take the first valid entry signal

        # --- 3. Default to HOLD ---
        if signal == "HOLD":
            lg.debug("Signal: HOLD - No valid entry or exit conditions met.")

        lg.debug(f"--- Signal Result: {signal} ---")
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: dict, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
        """Calculates initial Take Profit (TP) and Stop Loss (SL) levels for a new entry,
        based on the entry price, current ATR, configured multipliers, and market precision.

        Args:
            entry_price: The estimated or actual entry price of the trade.
            signal: The entry signal ("BUY" or "SELL").
            atr: The current Average True Range value (Decimal).
            market_info: The market information dictionary.
            exchange: The initialized ccxt.Exchange object (for price formatting).

        Returns:
            A tuple containing:
                - The calculated Take Profit price (Decimal), or None if disabled or calculation fails.
                - The calculated Stop Loss price (Decimal), or None if calculation fails critically.
            Returns (None, None) if inputs are invalid.
        """
        lg = self.logger
        lg.debug(f"Calculating Initial TP/SL for {signal} signal at entry {entry_price.normalize()} with ATR {atr.normalize()}")

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            lg.error(f"TP/SL Calc Failed: Invalid signal '{signal}'.")
            return None, None
        if entry_price <= 0 or atr <= 0:
            lg.error(f"TP/SL Calc Failed: Entry price ({entry_price}) and ATR ({atr}) must be positive.")
            return None, None
        try:
            price_tick_str = market_info['precision']['price']
            min_tick = Decimal(str(price_tick_str))
            if min_tick <= 0: raise ValueError("Invalid price tick size")
        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
            lg.error(f"TP/SL Calc Failed: Could not get valid price precision from market info: {e}")
            return None, None

        # --- Calculate Raw TP/SL ---
        try:
            tp_atr_multiple = self.initial_tp_atr_multiple
            sl_atr_multiple = self.initial_sl_atr_multiple

            # Calculate offsets
            tp_offset = atr * tp_atr_multiple
            sl_offset = atr * sl_atr_multiple

            # Calculate raw levels
            take_profit_raw: Decimal | None = None
            if tp_atr_multiple > 0:  # Only calculate TP if multiplier is positive
                 take_profit_raw = (entry_price + tp_offset) if signal == "BUY" else (entry_price - tp_offset)

            stop_loss_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)

            lg.debug(f"  Raw Levels: TP={take_profit_raw.normalize() if take_profit_raw else 'N/A'}, SL={stop_loss_raw.normalize()}")

            # --- Format Levels to Market Precision ---
            # Helper function to format and validate
            def format_level(price_decimal: Decimal | None, level_name: str) -> Decimal | None:
                if price_decimal is None or price_decimal <= 0:
                    lg.debug(f"Calculated {level_name} is invalid ({price_decimal}).")
                    return None
                try:
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal))
                    formatted_decimal = Decimal(formatted_str)
                    # Final check: ensure formatted price is still positive
                    if formatted_decimal > 0:
                         return formatted_decimal
                    else:
                         lg.warning(f"Formatted {level_name} ({formatted_str}) is zero or negative. Ignoring.")
                         return None
                except Exception as e:
                    lg.error(f"Error formatting {level_name} value {price_decimal} to exchange precision: {e}.")
                    return None

            # Format TP and SL
            take_profit_final = format_level(take_profit_raw, "Take Profit")
            stop_loss_final = format_level(stop_loss_raw, "Stop Loss")

            # --- Final Adjustments and Validation ---
            # Ensure SL is strictly beyond entry
            if stop_loss_final is not None:
                sl_invalid = (signal == "BUY" and stop_loss_final >= entry_price) or \
                             (signal == "SELL" and stop_loss_final <= entry_price)
                if sl_invalid:
                    lg.warning(f"Formatted {signal} Stop Loss {stop_loss_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Adjusting by one tick.")
                    adjusted_sl_raw = (entry_price - min_tick) if signal == "BUY" else (entry_price + min_tick)
                    stop_loss_final = format_level(adjusted_sl_raw, "Adjusted Stop Loss")
                    if stop_loss_final is None:
                         lg.error(f"{NEON_RED}CRITICAL: Failed to calculate valid adjusted SL after initial SL was invalid.{RESET}")
                         # Returning None for SL is critical failure
                         return take_profit_final, None

            # Ensure TP is strictly beyond entry (if TP is enabled)
            if take_profit_final is not None:
                tp_invalid = (signal == "BUY" and take_profit_final <= entry_price) or \
                             (signal == "SELL" and take_profit_final >= entry_price)
                if tp_invalid:
                    lg.warning(f"Formatted {signal} Take Profit {take_profit_final.normalize()} is not strictly beyond entry {entry_price.normalize()}. Disabling TP for this entry.")
                    take_profit_final = None  # Disable TP if it ends up on the wrong side

            # Log final calculated levels
            tp_log = take_profit_final.normalize() if take_profit_final else "None (Disabled or Calc Failed)"
            sl_log = stop_loss_final.normalize() if stop_loss_final else "None (Calc Failed!)"
            lg.info(f"  Calculated Initial Levels: TP={tp_log}, SL={sl_log}")

            # Critical check: Ensure SL calculation was successful
            if stop_loss_final is None:
                lg.error(f"{NEON_RED}Stop Loss calculation failed critically. Cannot determine position size or place trade safely.{RESET}")
                return take_profit_final, None  # Return None for SL

            return take_profit_final, stop_loss_final

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
            return None, None


# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: dict) -> None:
    """Performs one full cycle of the trading logic for a single symbol:
    1. Fetches and validates kline data.
    2. Runs the strategy analysis.
    3. Fetches current market price and position status.
    4. Generates a trading signal (BUY, SELL, EXIT_LONG, EXIT_SHORT, HOLD).
    5. Executes trading actions based on the signal and `enable_trading` config flag:
        - Enters new positions (calculates size, sets leverage, places order, sets initial protection).
        - Exits existing positions based on exit signals.
        - Manages existing positions (checks Break-Even, ensures TSL is active if enabled).

    Args:
        exchange: The initialized ccxt.Exchange object.
        symbol: The trading symbol to analyze and trade.
        config: The main configuration dictionary.
        logger: The logger instance for this symbol's trading activity.
        strategy_engine: The initialized VolumaticOBStrategy instance.
        signal_generator: The initialized SignalGenerator instance.
        market_info: The market information dictionary for the symbol.
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
        lg.critical(f"Invalid interval '{config['interval']}' configured. Cannot map to CCXT timeframe. Skipping cycle.")
        return

    # Determine how many klines to fetch
    min_required_data = strategy_engine.min_data_len
    fetch_limit_from_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    # Need at least what the strategy requires, but fetch user preference if it's more (up to API limit)
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
                     f"but strategy requires {min_required_data}. Analysis will be inaccurate. "
                     f"{NEON_YELLOW}ACTION REQUIRED: Reduce lookback periods in strategy config! Skipping cycle.{RESET}")
        elif klines_df.empty:
            lg.error(f"Failed to fetch any kline data for {symbol} {ccxt_interval}. Cannot proceed. Skipping cycle.")
        else:  # Fetched some data, but not enough
            lg.error(f"Fetched only {fetched_count} klines, but strategy requires {min_required_data}. "
                     f"Analysis may be inaccurate. Skipping cycle.")
        return  # Cannot proceed without sufficient data

    # --- 3. Run Strategy Analysis ---
    lg.debug("Running strategy analysis engine...")
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis update failed unexpectedly: {analysis_err}{RESET}", exc_info=True)
        return  # Stop cycle if analysis fails

    # Validate analysis results
    if not analysis_results or analysis_results['current_trend_up'] is None or \
       analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
        lg.error(f"{NEON_RED}Strategy analysis did not produce valid results (missing trend, close price, or ATR). Skipping cycle.{RESET}")
        lg.debug(f"Problematic Analysis Results: {analysis_results}")
        return
    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr']
    lg.info(f"Strategy Analysis Complete: Trend={'UP' if analysis_results['current_trend_up'] else 'DOWN'}, "
            f"Last Close={latest_close.normalize()}, ATR={current_atr.normalize()}")

    # --- 4. Get Current Market State (Price & Position) ---
    lg.debug("Fetching current market price and checking for open positions...")
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    open_position = get_open_position(exchange, symbol, lg)  # Returns dict or None

    # Determine price to use for checks (prefer live price, fallback to last close)
    price_for_checks = current_market_price if current_market_price and current_market_price > 0 else latest_close
    if price_for_checks <= 0:
        lg.error(f"{NEON_RED}Cannot determine a valid current price (Live={current_market_price}, LastClose={latest_close}). Skipping cycle.{RESET}")
        return
    if current_market_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch live market price. Using last kline close price ({latest_close.normalize()}) for protection checks.{RESET}")

    # --- 5. Generate Trading Signal ---
    lg.debug("Generating trading signal based on analysis and position...")
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation failed unexpectedly: {signal_err}{RESET}", exc_info=True)
        return  # Stop cycle if signal generation fails

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
        else:  # HOLD signal, no position
            lg.info("  (Action if enabled: No entry/exit action indicated)")
        # End cycle here if trading disabled
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Analysis-Only Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return  # Stop further processing

    # --- Trading IS Enabled ---
    lg.info(f"{BRIGHT}Trading is ENABLED. Processing signal '{signal}'...{RESET}")

    # --- Scenario 1: No Position -> Consider Entry ---
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** {signal} Signal & No Position: Initiating Entry Sequence... ***{RESET}")

        # Fetch current balance
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot fetch valid balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
            return

        # Calculate initial SL/TP based on latest close and ATR
        # Use latest_close for initial calculation as live price might fluctuate too much before entry
        initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)

        if initial_sl_calc is None:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to calculate a valid initial Stop Loss. Cannot size position.{RESET}")
            return
        if initial_tp_calc is None:
            lg.warning(f"{NEON_YELLOW}Initial Take Profit calculation failed or is disabled (TP Mult = 0). Proceeding without initial TP.{RESET}")

        # Set Leverage (if contract market)
        if market_info['is_contract']:
            leverage_to_set = int(config.get('leverage', 0))
            if leverage_to_set > 0:
                lg.info(f"Setting leverage to {leverage_to_set}x for {symbol}...")
                leverage_ok = set_leverage_ccxt(exchange, symbol, leverage_to_set, market_info, lg)
                if not leverage_ok:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set leverage to {leverage_to_set}x.{RESET}")
                    return
            else:
                lg.info("Leverage setting skipped (config leverage is 0 or invalid). Using exchange default.")

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
            confirmed_position = get_open_position(exchange, symbol, lg)
            if confirmed_position:
                try:
                    # Get actual entry price if available, fallback to latest close used for sizing
                    entry_price_actual_str = confirmed_position.get('entryPrice')
                    entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close
                    if entry_price_actual <= 0: entry_price_actual = latest_close  # Ensure positive
                    lg.info(f"{NEON_GREEN}Position Confirmed! Actual/Estimated Entry: ~{entry_price_actual.normalize()}{RESET}")

                    # Recalculate SL/TP based on actual entry price and current ATR for setting protection
                    prot_tp_calc, prot_sl_calc = signal_generator.calculate_initial_tp_sl(entry_price_actual, signal, current_atr, market_info, exchange)

                    if prot_sl_calc is None:
                        # This is critical - position is open but SL cannot be set
                        lg.error(f"{NEON_RED}{BRIGHT}CRITICAL ERROR: Position entered, but failed to recalculate SL based on entry price {entry_price_actual.normalize()}! POSITION IS UNPROTECTED!{RESET}")
                    else:
                        # Set protection (TSL or Fixed SL/TP)
                        protection_set_success = False
                        if prot_cfg.get("enable_trailing_stop", True):
                            lg.info(f"Setting Trailing Stop Loss (TSL) based on entry {entry_price_actual.normalize()}...")
                            protection_set_success = set_trailing_stop_loss(
                                exchange, symbol, market_info, confirmed_position, config, lg,
                                take_profit_price=prot_tp_calc  # Set TP alongside TSL if calculated
                            )
                        elif not prot_cfg.get("enable_trailing_stop", True) and (prot_sl_calc or prot_tp_calc):
                             # Only set fixed SL/TP if TSL is disabled and either SL or TP is valid
                            lg.info(f"Setting Fixed Stop Loss / Take Profit based on entry {entry_price_actual.normalize()}...")
                            protection_set_success = _set_position_protection(
                                exchange, symbol, market_info, confirmed_position, lg,
                                stop_loss_price=prot_sl_calc,
                                take_profit_price=prot_tp_calc
                            )
                        else:
                            lg.info("No protection (TSL or Fixed SL/TP) enabled in config. Position entered without protection.")
                            protection_set_success = True  # Considered success as no action was needed

                        # Log final status
                        if protection_set_success:
                            lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY & INITIAL PROTECTION SETUP COMPLETE ({symbol} {signal}) ==={RESET}")
                        else:
                            lg.error(f"{NEON_RED}{BRIGHT}=== TRADE PLACED, BUT FAILED TO SET PROTECTION ({symbol} {signal}). MANUAL MONITORING REQUIRED! ==={RESET}")

                except Exception as post_trade_err:
                    lg.error(f"{NEON_RED}Error during post-trade setup (protection setting): {post_trade_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Position is confirmed open for {symbol}, but may lack protection! Manual check recommended!{RESET}")
            else:
                # Order placed but position not found after delay
                lg.error(f"{NEON_RED}Order {trade_order['id']} was placed, but FAILED TO CONFIRM open position for {symbol} after {confirm_delay}s delay! Manual check required!{RESET}")
        else:
            # Order placement itself failed
            lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). No order was placed. ===")

    # --- Scenario 2: Existing Position -> Consider Exit or Manage ---
    elif open_position:
        pos_side = open_position['side']
        pos_size_decimal = open_position.get('size_decimal', Decimal('0'))  # Get parsed Decimal size
        lg.info(f"Existing {pos_side.upper()} position found (Size: {pos_size_decimal.normalize()}). Signal: {signal}")

        # Check if the signal triggers an exit
        exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                         (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_triggered:
            # --- Handle Exit Signal ---
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Received! Closing {pos_side} position... ***{RESET}")
            try:
                # Determine closing side and ensure size is positive
                # size_to_close = abs(pos_size_decimal) # Already positive from parsing? Double check.
                # Use abs() for safety
                size_to_close = abs(pos_size_decimal)
                if size_to_close <= 0:
                    lg.warning(f"Attempting to close {pos_side} position, but parsed size is {pos_size_decimal}. Position might already be closed or data is stale.")
                    return  # No action needed if size is zero/negative

                lg.info(f"===> Placing {signal} MARKET Order (Reduce Only) | Size: {size_to_close.normalize()} <===")
                # Pass the original EXIT signal to place_trade for logging clarity
                close_order = place_trade(exchange, symbol, signal, size_to_close, market_info, lg, reduce_only=True)

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) placed successfully for {symbol}.{RESET}")
                    # Optionally, wait and confirm position is actually closed here
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual intervention may be required!{RESET}")
            except Exception as close_err:
                lg.error(f"{NEON_RED}Error encountered while trying to close {pos_side} position for {symbol}: {close_err}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual closing of position {symbol} may be needed!{RESET}")

        else:
            # --- Handle Position Management (if not exiting) ---
            lg.debug(f"Signal ({signal}) allows holding {pos_side} position. Performing position management checks...")

            # Extract current protection levels and entry price safely
            try: tsl_active = open_position.get('trailingStopLoss') and Decimal(str(open_position['trailingStopLoss'])) > 0
            except (KeyError, ValueError, InvalidOperation, TypeError): tsl_active = False

            try: current_sl_str = open_position.get('stopLossPrice'); current_sl = Decimal(str(current_sl_str)) if current_sl_str and str(current_sl_str) != '0' else None
            except (KeyError, ValueError, InvalidOperation, TypeError): current_sl = None

            try: current_tp_str = open_position.get('takeProfitPrice'); current_tp = Decimal(str(current_tp_str)) if current_tp_str and str(current_tp_str) != '0' else None
            except (KeyError, ValueError, InvalidOperation, TypeError): current_tp = None

            try: entry_price_str = open_position.get('entryPrice'); entry_price = Decimal(str(entry_price_str)) if entry_price_str else None
            except (KeyError, ValueError, InvalidOperation, TypeError): entry_price = None

            # --- Break-Even (BE) Logic ---
            be_enabled = prot_cfg.get("enable_break_even", True)
            # Check BE only if enabled, TSL is not active, and we have needed data
            if be_enabled and not tsl_active and entry_price and current_atr and price_for_checks > 0:
                lg.debug(f"Checking Break-Even condition for {pos_side} position...")
                lg.debug(f"  BE Check Inputs: Entry={entry_price.normalize()}, CurrentPrice={price_for_checks.normalize()}, ATR={current_atr.normalize()}, CurrentSL={current_sl}")
                try:
                    be_trigger_atr_multiple = Decimal(str(prot_cfg["break_even_trigger_atr_multiple"]))
                    be_offset_ticks = int(prot_cfg["break_even_offset_ticks"])
                    price_tick = Decimal(str(market_info['precision']['price']))

                    if not (be_trigger_atr_multiple > 0 and be_offset_ticks >= 0 and price_tick > 0):
                         raise ValueError("Invalid BE config parameters (trigger multiple, offset ticks, or price tick).")

                    # Calculate profit in terms of ATR multiples
                    profit_in_price = (price_for_checks - entry_price) if pos_side == 'long' else (entry_price - price_for_checks)
                    profit_in_atr = profit_in_price / current_atr if current_atr > 0 else Decimal(0)
                    lg.debug(f"  Profit (Price): {profit_in_price.normalize()}, Profit (ATR multiples): {profit_in_atr:.3f}")
                    lg.debug(f"  BE Trigger ATR Multiple: {be_trigger_atr_multiple.normalize()}")

                    # Check if profit target is reached
                    if profit_in_atr >= be_trigger_atr_multiple:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}Break-Even profit target REACHED! (Profit ATRs >= Trigger ATRs){RESET}")

                        # Calculate the BE Stop Loss level (Entry + Offset)
                        be_offset_value = price_tick * Decimal(str(be_offset_ticks))
                        be_stop_loss_raw = (entry_price + be_offset_value) if pos_side == 'long' else (entry_price - be_offset_value)

                        # Quantize BE SL to nearest tick (away from current price / towards profit)
                        if pos_side == 'long':
                            be_stop_loss = be_stop_loss_raw.quantize(price_tick, ROUND_UP)
                        else:  # Short
                            be_stop_loss = be_stop_loss_raw.quantize(price_tick, ROUND_DOWN)

                        if be_stop_loss and be_stop_loss > 0:
                            lg.debug(f"  Calculated BE Stop Loss level: {be_stop_loss.normalize()}")
                            # Check if the new BE SL is better than the current SL
                            update_sl_needed = False
                            if current_sl is None:
                                update_sl_needed = True
                                lg.info("  Current SL is not set. Applying BE SL.")
                            elif (pos_side == 'long' and be_stop_loss > current_sl) or \
                                 (pos_side == 'short' and be_stop_loss < current_sl):
                                update_sl_needed = True
                                lg.info(f"  New BE SL ({be_stop_loss.normalize()}) is improvement over current SL ({current_sl.normalize()}). Applying update.")
                            else:
                                lg.debug(f"  Current SL ({current_sl.normalize()}) is already at or better than calculated BE SL ({be_stop_loss.normalize()}). No update needed.")

                            # If update is needed, call the protection function
                            if update_sl_needed:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even at {be_stop_loss.normalize()} ***{RESET}")
                                # Pass current TP to avoid accidentally clearing it
                                if _set_position_protection(exchange, symbol, market_info, open_position, lg,
                                                            stop_loss_price=be_stop_loss, take_profit_price=current_tp):
                                    lg.info(f"{NEON_GREEN}Break-Even Stop Loss set successfully.{RESET}")
                                else:
                                    lg.error(f"{NEON_RED}Failed to set Break-Even Stop Loss via API.{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Break-Even triggered, but calculated BE Stop Loss level ({be_stop_loss}) is invalid.{RESET}")
                    else:
                        lg.debug("Break-Even profit target not yet reached.")

                except (KeyError, ValueError, InvalidOperation, TypeError, ZeroDivisionError) as be_err:
                    lg.error(f"{NEON_RED}Error during Break-Even check: {be_err}{RESET}", exc_info=True)
            elif be_enabled:
                lg.debug(f"Break-Even check skipped: {'TSL is active' if tsl_active else 'Missing required data (entry price, ATR, or current price)'}.")
            else:  # BE disabled
                lg.debug("Break-Even check skipped: Disabled in configuration.")

            # --- TSL Setup/Recovery Logic ---
            # If TSL is enabled in config but not detected as active on the position
            tsl_enabled = prot_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not tsl_active and entry_price and current_atr:
                 lg.warning(f"{NEON_YELLOW}Trailing Stop Loss (TSL) is enabled in config, but not detected as active on the current {pos_side} position. Attempting TSL setup/recovery...{RESET}")
                 # Recalculate initial TP based on current state (in case it was missed or cleared)
                 # We don't need the SL from this calculation for TSL setup.
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(entry_price, pos_side.upper(), current_atr, market_info, exchange)

                 # Attempt to set the TSL (and potentially TP)
                 if set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc):
                     lg.info(f"{NEON_GREEN}TSL setup/recovery successful for {symbol}.{RESET}")
                 else:
                     lg.error(f"{NEON_RED}TSL setup/recovery FAILED for {symbol}. Fixed SL might still be active if previously set.{RESET}")
            elif tsl_enabled:
                 lg.debug(f"TSL setup/recovery check skipped: {'TSL already active' if tsl_active else 'Missing required data (entry price or ATR)'}.")
            else:  # TSL disabled
                 lg.debug("TSL setup/recovery check skipped: Disabled in configuration.")

    # --- Scenario 3: No Position and HOLD Signal ---
    elif open_position is None and signal == "HOLD":
         lg.info("Signal is HOLD and no position is open. No trading action taken.")

    # --- Scenario 4: Unhandled State (Should not normally happen) ---
    else:
        lg.error(f"Unhandled state encountered: Signal='{signal}', Position Side='{open_position.get('side') if open_position else None}'. No action defined.")

    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"{BRIGHT}---=== Cycle End: {symbol} (Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---{RESET}\n")


# --- Main Function ---
def main() -> None:
    """Main execution function:
    - Initializes logging and loads configuration.
    - Prompts user for confirmation if live trading is enabled.
    - Initializes the CCXT exchange object.
    - Prompts user for the trading symbol and timeframe.
    - Sets up symbol-specific logger and strategy instances.
    - Enters the main trading loop, calling `analyze_and_trade_symbol` repeatedly.
    - Handles graceful shutdown on KeyboardInterrupt (Ctrl+C) or critical errors.
    """
    global CONFIG, QUOTE_CURRENCY  # Allow main to update config (e.g., timeframe)

    # Use init_logger for startup messages before symbol is chosen
    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Volumatic OB Bot v1.1.6 ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    init_logger.info(f"Loaded Configuration: Quote Currency={QUOTE_CURRENCY}, Trading Enabled={CONFIG['enable_trading']}, Sandbox Mode={CONFIG['use_sandbox']}")
    try:
        # Log versions of key libraries
        py_version = os.sys.version.split()[0]
        ccxt_version = getattr(ccxt, '__version__', 'N/A')
        pd_version = getattr(pd, '__version__', 'N/A')
        ta_version = getattr(ta, 'version', 'N/A')  # pandas_ta version attribute might vary
        np_version = getattr(np, '__version__', 'N/A')
        init_logger.info(f"Versions: Python={py_version}, CCXT={ccxt_version}, Pandas={pd_version}, Numpy={np_version}, Pandas-TA={ta_version}")
    except Exception as e:
        init_logger.warning(f"Could not retrieve library versions: {e}")

    # --- User Confirmation for Live Trading ---
    if CONFIG.get("enable_trading", False):
        init_logger.warning(f"\n{NEON_YELLOW}{BRIGHT}╔═════════════════════════════════════════════════════════════╗")
        init_logger.warning(f"║ {Fore.RED}{BRIGHT}!!!           LIVE TRADING IS ENABLED           !!!{RESET}{NEON_YELLOW}{BRIGHT} ║")
        init_logger.warning(f"╚═════════════════════════════════════════════════════════════╝{RESET}")
        mode_str = f"{NEON_RED}!!! LIVE (REAL FUNDS AT RISK) !!!" if not CONFIG.get('use_sandbox') else f"{NEON_GREEN}SANDBOX (Testnet Funds)"
        init_logger.warning(f"Trading Mode: {mode_str}{RESET}")

        # Display key risk and protection settings for review
        prot_cfg = CONFIG.get("protection", {})
        init_logger.warning(f"{BRIGHT}--- Please Review Key Trading Settings ---{RESET}")
        init_logger.warning(f"  Risk Per Trade: {CONFIG.get('risk_per_trade', 0.01):.2%}")
        init_logger.warning(f"  Leverage: {CONFIG.get('leverage', 0)}x (0 uses exchange default)")
        init_logger.warning(f"  Trailing Stop (TSL): {'ENABLED' if prot_cfg.get('enable_trailing_stop', False) else 'DISABLED'}")
        if prot_cfg.get('enable_trailing_stop', False):
            init_logger.warning(f"    - Callback Rate: {prot_cfg.get('trailing_stop_callback_rate', 0):.3%}")
            init_logger.warning(f"    - Activation Pct: {prot_cfg.get('trailing_stop_activation_percentage', 0):.3%}")
        init_logger.warning(f"  Break Even (BE): {'ENABLED' if prot_cfg.get('enable_break_even', False) else 'DISABLED'}")
        if prot_cfg.get('enable_break_even', False):
            init_logger.warning(f"    - Trigger ATRs: {prot_cfg.get('break_even_trigger_atr_multiple', 0)}")
            init_logger.warning(f"    - Offset Ticks: {prot_cfg.get('break_even_offset_ticks', 0)}")
        init_logger.warning(f"  Initial SL ATR Multiple: {prot_cfg.get('initial_stop_loss_atr_multiple', 0)}")
        tp_mult = prot_cfg.get('initial_take_profit_atr_multiple', 0)
        init_logger.warning(f"  Initial TP ATR Multiple: {tp_mult} {'(Disabled)' if tp_mult == 0 else ''}")

        # Confirmation prompt
        try:
            input(f"\n{BRIGHT}>>> Press {NEON_GREEN}Enter{RESET}{BRIGHT} to confirm these settings and START TRADING, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to ABORT... {RESET}")
            init_logger.info("User confirmed settings. Proceeding with trading.")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup via Ctrl+C.")
            logging.shutdown()
            return  # Exit the program
    else:
        init_logger.info(f"{NEON_YELLOW}Trading is DISABLED in config.json. Running in analysis-only mode.{RESET}")
        # Add a small delay so user sees the message
        time.sleep(2)

    # --- Initialize Exchange ---
    init_logger.info("Initializing CCXT exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical("Exchange initialization failed. Cannot continue. Exiting.")
        logging.shutdown()
        return  # Exit if exchange setup fails

    init_logger.info(f"Exchange '{exchange.id}' initialized successfully.")

    # --- Get Trading Symbol from User ---
    target_symbol: str | None = None
    market_info: dict | None = None
    while target_symbol is None:
        try:
            # Prompt user for the symbol
            symbol_input = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT, ETH/USD:ETH): {RESET}").strip().upper()
            if not symbol_input:
                continue  # Ask again if input is empty

            init_logger.info(f"Validating symbol '{symbol_input}' on {exchange.id}...")
            # Use get_market_info to validate and fetch details
            m_info = get_market_info(exchange, symbol_input, init_logger)

            if m_info:
                 target_symbol = m_info['symbol']  # Use the standardized symbol from CCXT
                 market_info = m_info
                 init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} | Type: {market_info.get('contract_type_str', 'Unknown')}")
                 # Critical check: Ensure precision is available (re-check after get_market_info)
                 if market_info.get('precision', {}).get('price') is None or market_info.get('precision', {}).get('amount') is None:
                      init_logger.critical(f"{NEON_RED}CRITICAL ERROR:{RESET} Market '{target_symbol}' is missing essential price or amount precision data after validation! Cannot trade safely. Exiting.")
                      logging.shutdown()
                      return
                 break  # Exit loop once valid symbol is found
            else:
                 # get_market_info already logged the error
                 init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' could not be validated or market info retrieval failed. Please try again.{RESET}")
                 init_logger.info("Common formats: BASE/QUOTE (e.g., BTC/USDT), BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT for linear, ETH/USD:ETH for inverse)")

        except KeyboardInterrupt:
            init_logger.info("User aborted during symbol selection.")
            logging.shutdown()
            return
        except Exception as e:
            init_logger.error(f"Unexpected error during symbol validation: {e}", exc_info=True)
            # Loop continues to ask again

    # Ensure market_info is not None (should be set if target_symbol is set)
    if market_info is None:
         init_logger.critical("Market info is unexpectedly None after symbol selection. Exiting.")
         logging.shutdown()
         return

    # --- Get Timeframe from User ---
    selected_interval: str | None = None
    while selected_interval is None:
        default_tf = CONFIG.get('interval', '5')  # Get default from config
        interval_input = input(f"{NEON_YELLOW}Enter timeframe {VALID_INTERVALS} (default: {default_tf}): {RESET}").strip()

        if not interval_input:
            interval_input = default_tf
            init_logger.info(f"No timeframe entered, using default: {interval_input}")

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             # Update the interval in the global CONFIG dictionary for the current session
             CONFIG["interval"] = selected_interval
             ccxt_tf = CCXT_INTERVAL_MAP.get(selected_interval, "N/A")
             init_logger.info(f"Using timeframe: {selected_interval} (CCXT mapping: {ccxt_tf})")
             break
        else:
             init_logger.error(f"{NEON_RED}Invalid timeframe '{interval_input}'. Please choose from: {VALID_INTERVALS}{RESET}")

    # --- Setup Symbol-Specific Logger & Strategy Instances ---
    # Use the validated target_symbol for the logger name
    symbol_logger = setup_logger(target_symbol)
    symbol_logger.info(f"{BRIGHT}---=== Starting Trading Loop for: {target_symbol} | Timeframe: {CONFIG['interval']} ===---{RESET}")
    symbol_logger.info(f"Trading Enabled: {CONFIG['enable_trading']} | Sandbox Mode: {CONFIG['use_sandbox']}")
    prot_cfg_log = CONFIG.get("protection", {})  # Re-fetch for logging
    symbol_logger.info(f"Key Protections: TSL={'ON' if prot_cfg_log.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if prot_cfg_log.get('enable_break_even') else 'OFF'}")

    try:
        # Initialize strategy engine and signal generator instances
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_err:
        symbol_logger.critical(f"Failed to initialize strategy engine or signal generator: {engine_err}. Exiting.", exc_info=True)
        logging.shutdown()
        return

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}Entering main trading loop... Press Ctrl+C to stop gracefully.{RESET}")
    loop_count = 0
    try:
        while True:
            loop_start_time = time.time()
            loop_count += 1
            symbol_logger.debug(f">>> Loop #{loop_count} Start: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # --- Core Logic Execution ---
            try:
                analyze_and_trade_symbol(
                    exchange=exchange,
                    symbol=target_symbol,
                    config=CONFIG,
                    logger=symbol_logger,
                    strategy_engine=strategy_engine,
                    signal_generator=signal_generator,
                    market_info=market_info
                )
            # --- Handle Specific Exceptions within the Loop ---
            except ccxt.RateLimitExceeded as e:
                symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded during main loop: {e}. Waiting 60 seconds...{RESET}")
                time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e:
                delay = RETRY_DELAY_SECONDS * 3
                symbol_logger.error(f"{NEON_RED}Network error encountered in main loop: {e}. Waiting {delay} seconds...{RESET}")
                time.sleep(delay)
            except ccxt.AuthenticationError as e:
                # Critical auth errors should stop the bot
                symbol_logger.critical(f"{NEON_RED}CRITICAL AUTHENTICATION ERROR in main loop: {e}. Stopping bot.{RESET}")
                break  # Exit the while loop
            except ccxt.ExchangeNotAvailable as e:
                symbol_logger.error(f"{NEON_RED}Exchange not available (e.g., maintenance): {e}. Waiting 60 seconds...{RESET}")
                time.sleep(60)
            except ccxt.OnMaintenance as e:
                symbol_logger.error(f"{NEON_RED}Exchange is on maintenance: {e}. Waiting 5 minutes...{RESET}")
                time.sleep(300)
            except ccxt.ExchangeError as e:
                # Catch other potentially recoverable exchange errors
                symbol_logger.error(f"{NEON_RED}Unhandled Exchange Error in main loop: {e}{RESET}", exc_info=True)
                time.sleep(10)  # Wait before potentially retrying cycle
            except Exception as loop_err:
                # Catch any other unexpected errors to prevent crash
                symbol_logger.error(f"{NEON_RED}Critical unexpected error in main trading loop: {loop_err}{RESET}", exc_info=True)
                symbol_logger.warning("Waiting 15 seconds before next cycle attempt...")
                time.sleep(15)  # Pause before potentially retrying the cycle

            # --- Loop Delay ---
            elapsed_time = time.time() - loop_start_time
            # Use the loop delay from the (potentially updated) global CONFIG
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_duration = max(0, loop_delay - elapsed_time)

            symbol_logger.debug(f"<<< Loop #{loop_count} completed in {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s...")
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt detected (Ctrl+C). Initiating graceful shutdown...")
    except Exception as critical_err:
        # Catch errors outside the inner try/except (e.g., during sleep?)
        init_logger.critical(f"CRITICAL UNHANDLED ERROR outside main loop: {critical_err}", exc_info=True)
        if 'symbol_logger' in locals():  # Log to symbol logger if available
             symbol_logger.critical(f"CRITICAL UNHANDLED ERROR outside main loop: {critical_err}", exc_info=True)

    # --- Shutdown Sequence ---
    finally:
        shutdown_msg = f"--- Pyrmethus Bot ({target_symbol or 'N/A'}) Shutting Down ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals():  # Log shutdown message to symbol logger too
            symbol_logger.info(shutdown_msg)

        # Attempt to close exchange connection (optional for sync CCXT, but good practice)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Closing CCXT exchange connection...")
                # exchange.close() # Often not strictly needed for synchronous client, can cause issues
                init_logger.info("Exchange connection closed (or cleanup skipped).")
            except Exception as close_err:
                init_logger.error(f"Error during exchange.close(): {close_err}")

        # Ensure log handlers are closed properly
        logging.shutdown()  # Standard way to flush and close handlers

        # Explicitly close handlers as a fallback (sometimes helps in complex scenarios)
        try:
            loggers_to_close = [logging.getLogger(name) for name in logging.root.manager.loggerDict if isinstance(logging.getLogger(name), logging.Logger)]
            loggers_to_close.append(logging.getLogger())  # Add root logger
            loggers_to_close.append(init_logger)
            if 'symbol_logger' in locals(): loggers_to_close.append(symbol_logger)

            unique_handlers = set()
            for logger_instance in loggers_to_close:
                 if hasattr(logger_instance, 'handlers'):
                     for handler in logger_instance.handlers:
                         unique_handlers.add(handler)

            for handler in unique_handlers:
                 try:
                     handler.close()
                 except Exception:
                     # Non-critical error during handler closing
                     pass
        except Exception:
            pass


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
