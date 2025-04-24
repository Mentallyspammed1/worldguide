import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
from zoneinfo import ZoneInfo # Requires tzdata package (pip install tzdata)

# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy
import pandas as pd # Requires pandas
import pandas_ta as ta # Requires pandas_ta
import requests # Requires requests
# import websocket # Requires websocket-client (Removed - Unused in v1.1.5)
import ccxt # Requires ccxt
from colorama import Fore, Style, init as colorama_init # Requires colorama
from dotenv import load_dotenv # Requires python-dotenv

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision for financial calculations
colorama_init(autoreset=True) # Initialize Colorama for colored console output
load_dotenv() # Load environment variables from .env file

# --- Constants ---
# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in the .env file")

# Configuration and Logging
CONFIG_FILE: str = "config.json"
LOG_DIRECTORY: str = "bot_logs"
try:
    # Attempt to load user-defined timezone, fallback to UTC if tzdata is not installed or timezone invalid
    user_timezone_str = os.getenv("TIMEZONE", "America/Chicago") # Default to Chicago if not set
    TIMEZONE = ZoneInfo(user_timezone_str)
except Exception:
    print(f"{Fore.RED}Failed to initialize timezone '{user_timezone_str}'. Install 'tzdata' package (`pip install tzdata`) or set a valid IANA timezone in .env. Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")

# API Interaction Settings
MAX_API_RETRIES: int = 3 # Maximum number of retries for non-critical API calls
RETRY_DELAY_SECONDS: int = 5 # Base delay between API retries (can increase exponentially)
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Wait time after placing order before confirming position state
LOOP_DELAY_SECONDS: int = 15 # Default delay between trading cycles (can be overridden in config)
BYBIT_API_KLINE_LIMIT: int = 1000 # Bybit V5 Kline limit per single API request

# Timeframes
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750 # Default kline fetch limit if not set in config (used if less than strategy min_data_len)
MAX_DF_LEN: int = 2000 # Internal limit to prevent excessive memory usage by Pandas DataFrame

# Strategy Defaults (Used if values are missing or invalid in config.json)
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950 # Adjusted default (Original 1000 often > API Limit if many candles needed)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Note: Step ATR Multiplier currently unused in core logic
DEFAULT_OB_SOURCE: str = "Wicks" # Alternative: "Body" - Determines if Order Blocks use candle wicks or bodies
DEFAULT_PH_LEFT: int = 10
DEFAULT_PH_RIGHT: int = 10 # Pivot High lookback periods
DEFAULT_PL_LEFT: int = 10
DEFAULT_PL_RIGHT: int = 10 # Pivot Low lookback periods
DEFAULT_OB_EXTEND: bool = True # Whether to visually extend OBs to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50 # Maximum number of active Order Blocks to track per side

# QUOTE_CURRENCY (e.g., "USDT") is dynamically loaded from the config file globally later.

# Logging Colors
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
    """Custom logging formatter that redacts sensitive API keys from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing API keys/secrets with placeholders."""
        msg = super().format(record)
        if API_KEY: msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET: msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a dedicated logger instance with both console and rotating file handlers.
    Ensures handlers are not added multiple times if the logger already exists.

    Args:
        name: The name for the logger (often the trading symbol or 'init').

    Returns:
        The configured logging.Logger instance.
    """
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize name for use in filenames
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists (e.g., during re-init)
    if logger.hasHandlers():
        # Clear existing handlers to potentially update settings (like level)
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        # return logger # Option 1: Return existing logger as is
        # Option 2: Re-add handlers below to apply potential config changes

    logger.setLevel(logging.DEBUG) # Capture all levels; handlers control output level

    # File Handler (Level: DEBUG, Rotating, UTF-8)
    try:
        # Rotate logs at 10MB, keep 5 backups
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        # Detailed format for file logs
        ff = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG) # Log everything to the file
        logger.addHandler(fh)
    except Exception as e:
        # Print error directly as logger might not be fully functional
        print(f"{NEON_RED}Error setting up file logger '{log_filename}': {e}{RESET}")

    # Console Handler (Level from ENV or INFO default, Timezone-aware)
    try:
        sh = logging.StreamHandler()
        # Use timezone-aware timestamps for console output
        logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
        # Colored and formatted console output
        sf = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        sh.setFormatter(sf)
        # Get desired console log level from environment variable, default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid level name
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent messages propagating to the root logger
    return logger

# Initialize the 'init' logger early for messages during startup and config loading
init_logger = setup_logger("init")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """
    Recursively ensures all keys from the default configuration exist in the loaded configuration.
    Adds missing keys with their default values and logs the additions.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: Used internally for logging the path of nested keys.

    Returns:
        A tuple containing:
        - The updated configuration dictionary.
        - A boolean indicating if any keys were added or modified.
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config Update: Added missing key '{full_key_path}' with default value: {default_value}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
        # Optional: Could add type checking here as well, but validation handles it later
        # elif type(updated_config.get(key)) != type(default_value) and default_value is not None:
        #     init_logger.warning(f"Config Type Mismatch: Key '{full_key_path}' has type {type(updated_config.get(key))} but default is {type(default_value)}. Validation will attempt correction.")

    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file. If the file doesn't exist, creates a default one.
    Validates the loaded configuration against defaults, adds missing keys, validates types/ranges,
    and saves the updated configuration back to the file if any changes were made.

    Args:
        filepath: The path to the configuration JSON file (e.g., "config.json").

    Returns:
        The loaded and validated configuration dictionary. Returns default values
        if the file cannot be loaded, parsed, or validated correctly.
    """
    # Define the default configuration structure and values
    default_config = {
        "interval": "5",                    # Default timeframe (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # API retry delay base (seconds)
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Preferred number of klines to fetch (strategy may override)
        "orderbook_limit": 25,              # Max order book depth (currently unused in logic)
        "enable_trading": False,            # Master switch for placing real trades (IMPORTANT!)
        "use_sandbox": True,                # Use Bybit's testnet environment (True) or live (False)
        "risk_per_trade": 0.01,             # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                     # Desired leverage for contract trading (e.g., 20x)
        "max_concurrent_positions": 1,      # Max open positions per symbol (current implementation supports 1)
        "quote_currency": "USDT",           # The quote currency for balance checks and PnL (e.g., USDT, USDC)
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between trading cycles (seconds)
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait after order placement to check position status (seconds)
        "strategy_params": {                # Parameters for the Volumatic OB Strategy
            "vt_length": DEFAULT_VT_LENGTH,             # Volumatic Trend EMA length
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,       # Volumatic Trend ATR period
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, # Volumatic Trend Volume EMA length
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER, # Volumatic Trend ATR multiplier for bands
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER, # Step ATR Multiplier (currently unused)
            "ob_source": DEFAULT_OB_SOURCE,           # Order Block source ("Wicks" or "Body")
            "ph_left": DEFAULT_PH_LEFT,               # Pivot High left lookback
            "ph_right": DEFAULT_PH_RIGHT,             # Pivot High right lookback
            "pl_left": DEFAULT_PL_LEFT,               # Pivot Low left lookback
            "pl_right": DEFAULT_PL_RIGHT,             # Pivot Low right lookback
            "ob_extend": DEFAULT_OB_EXTEND,           # Extend OBs visually to current bar
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,     # Max active OBs to track per side
            "ob_entry_proximity_factor": 1.005,       # Factor > 1 to slightly widen OB for entry check (e.g., 1.005 = 0.5% wider range)
            "ob_exit_proximity_factor": 1.001         # Factor > 1 to slightly widen opposing OB for exit check (e.g., 1.001 = 0.1% closer threshold)
        },
        "protection": {                     # Position protection settings
             "enable_trailing_stop": True,             # Use Trailing Stop Loss (TSL)?
             "trailing_stop_callback_rate": 0.005,     # TSL trail distance as a percentage of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Profit percentage from entry to activate TSL (e.g., 0.003 = 0.3%)
             "enable_break_even": True,                # Move SL to break-even after profit target hit?
             "break_even_trigger_atr_multiple": 1.0,   # ATR multiples in profit required to trigger BE
             "break_even_offset_ticks": 2,             # Ticks above/below entry for BE SL placement (uses price precision)
             "initial_stop_loss_atr_multiple": 1.8,    # Initial SL distance in ATR multiples from entry
             "initial_take_profit_atr_multiple": 0.7   # Initial TP distance in ATR multiples from entry (set to 0 to disable initial TP)
        }
    }
    config_needs_saving = False
    loaded_config = {}

    # --- Create Default Config File if Missing ---
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Configuration file '{filepath}' not found. Creating a default config file.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            return default_config # Return the defaults as the current config
        except IOError as e:
            init_logger.critical(f"{NEON_RED}Fatal Error: Could not create default config file '{filepath}': {e}. Using hardcoded defaults, but cannot save changes.{RESET}")
            return default_config # Return defaults, but saving won't work

    # --- Load Existing Config File ---
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}. Attempting to recreate with defaults.{RESET}")
        try:
            # Backup corrupted file before overwriting
            backup_path = f"{filepath}.corrupted_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            os.rename(filepath, backup_path)
            init_logger.warning(f"Backed up corrupted config to: {backup_path}")
            # Try to recreate the file with defaults
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            return default_config
        except IOError as e_create:
            init_logger.critical(f"{NEON_RED}Fatal Error: Could not recreate default config file after JSON error: {e_create}. Using hardcoded defaults.{RESET}")
            return default_config
    except Exception as e:
        init_logger.critical(f"{NEON_RED}Fatal Error: Unexpected error loading config file '{filepath}': {e}. Using hardcoded defaults.{RESET}", exc_info=True)
        return default_config

    # --- Ensure All Keys Exist and Perform Validation ---
    try:
        # Ensure all default keys exist in the loaded config, add if missing
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """
            Validates a numeric config value within nested dictionaries.
            Logs a warning and uses the default value if the current value is invalid.

            Args:
                cfg: The config dict (potentially nested) to validate.
                key_path: Dot-separated path to the key (e.g., "strategy_params.vt_length").
                min_val: Minimum allowed value (inclusive unless is_strict_min).
                max_val: Maximum allowed value (inclusive).
                is_strict_min: If True, value must be strictly greater than min_val.
                is_int: If True, value must be an integer.
                allow_zero: If True, zero is allowed even if outside min/max range (useful for disabling features like TP).

            Returns:
                True if the value was corrected to the default, False otherwise.
            """
            nonlocal config_needs_saving # Allow modification of the outer flag
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse dictionaries to find the value and its default
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key) # Use .get() for safety
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid key path '{key_path}'. Cannot validate.")
                return False # Path is invalid, cannot validate or correct

            if original_val is None:
                # This case should ideally be handled by _ensure_config_keys, but check defensively
                init_logger.warning(f"Config '{key_path}': Key missing during validation, but should have been added. Using default: {default_val}.")
                current_level[leaf_key] = default_val
                config_needs_saving = True
                return True

            corrected = False
            final_val = original_val
            try:
                # Use Decimal for accurate range checking
                num_val = Decimal(str(original_val))
                min_check = num_val > Decimal(str(min_val)) if is_strict_min else num_val >= Decimal(str(min_val))
                max_check = num_val <= Decimal(str(max_val))

                # Check range, allowing zero if specified
                if not (min_check and max_check) and not (allow_zero and num_val == 0):
                    raise ValueError(f"Value {num_val} out of allowed range ({min_val} to {max_val}, strict_min={is_strict_min}, allow_zero={allow_zero})")

                # Check integer requirement
                if is_int and num_val != num_val.to_integral_value(rounding=ROUND_DOWN):
                    raise ValueError(f"Value {num_val} must be an integer.")

                # Convert to the target type (int or float)
                target_type = int if is_int else float
                final_val = target_type(num_val)

                # Check if type conversion or value changed significantly (handling float precision)
                if type(final_val) is not type(original_val) or abs(float(original_val) - float(final_val)) > 1e-9:
                   corrected = True # Mark as corrected if type changed or value changed non-trivially
            except (ValueError, InvalidOperation, TypeError) as e:
                init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{original_val}' (Error: {e}). Using default: {default_val}.{RESET}")
                final_val = default_val # Use the default value
                corrected = True

            if corrected:
                current_level[leaf_key] = final_val # Update the config dictionary
                config_needs_saving = True
            return corrected

        # --- Apply Specific Validations ---
        # Validate interval
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid config interval '{updated_config.get('interval')}'. Must be one of {VALID_INTERVALS}. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True

        # Validate numeric parameters using the helper
        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        # Allow fetch_limit > API limit internally; kline fetch handles actual request cap. Use MAX_DF_LEN as upper bound.
        validate_numeric(updated_config, "fetch_limit", 100, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", 0, 1, is_strict_min=True) # Risk must be > 0% and <= 100%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True) # Leverage 0 often means cross/spot, allow it
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, 1000, is_int=True)
        # Validate Vol EMA length, allow user setting > API limit (handled later in strategy init if needed)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1) # Must be >= 1
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)  # Must be >= 1
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid strategy_params.ob_source '{updated_config['strategy_params']['ob_source']}'. Must be 'Wicks' or 'Body'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE
             config_needs_saving = True

        # Protection Params
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0001, 0.5, is_strict_min=True) # TSL rate must be > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0, 0.5, allow_zero=True) # TSL Activation can be 0 (activate immediately)
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.1, 10.0)
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # BE offset can be 0
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.1, 100.0)
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0, 100.0, allow_zero=True) # TP can be 0 (disabled)

        # Validate boolean flags
        bool_keys = [
            ("enable_trading", False), ("use_sandbox", True),
            ("strategy_params", "ob_extend", True),
            ("protection", "enable_trailing_stop", True),
            ("protection", "enable_break_even", True)
        ]
        for path_tuple in bool_keys:
            current_level = updated_config
            default_level = default_config
            key_path = ""
            try:
                for key in path_tuple[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                    key_path += f"{key}."
                leaf_key = path_tuple[-1]
                key_path += leaf_key
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)

                if not isinstance(original_val, bool):
                    init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{original_val}' (expected boolean True/False). Using default: {default_val}.{RESET}")
                    current_level[leaf_key] = default_val
                    config_needs_saving = True
            except (KeyError, TypeError):
                 init_logger.error(f"Config validation error: Invalid path checking boolean key '{key_path}'.")


        # --- Save Updated Config if Necessary ---
        if config_needs_saving:
             init_logger.info(f"Configuration changes detected (missing keys added or values corrected). Saving updated config to: {filepath}")
             try:
                 # Convert any internal Decimal objects back to standard types (float/int) for JSON serialization
                 # A simple way is to re-load from a json dump/load cycle
                 config_to_save = json.loads(json.dumps(updated_config, default=str)) # Use default=str for Decimals if needed
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Successfully saved updated configuration.{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated config file '{filepath}': {save_err}. The bot will use the corrected values in memory, but they are not saved.{RESET}", exc_info=True)

        return updated_config

    except Exception as e:
        init_logger.critical(f"{NEON_RED}Fatal Error: Unexpected error processing configuration: {e}. Using hardcoded defaults.{RESET}", exc_info=True)
        # In case of unexpected error during validation, return defaults to allow bot to potentially run
        return default_config

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Ensure QUOTE_CURRENCY is set globally after loading config

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object.
    Handles API key setup, sandbox mode, rate limiting, market loading with retries,
    and initial balance check.

    Args:
        logger: The logger instance for logging initialization messages.

    Returns:
        An initialized ccxt.Exchange object if successful, otherwise None.
    """
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear', # Default contract type (can be overridden by market info or params)
                'adjustForTimeDifference': True, # Auto-adjust for clock skew between client and server
                # Set longer timeouts for potentially slow API operations
                'fetchTickerTimeout': 15000,    # 15 seconds
                'fetchBalanceTimeout': 20000,   # 20 seconds
                'createOrderTimeout': 30000,    # 30 seconds
                'cancelOrderTimeout': 20000,    # 20 seconds
                'fetchPositionsTimeout': 25000, # 25 seconds
                'fetchOHLCVTimeout': 60000,     # 60 seconds for potentially large kline fetches
            }
        }
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** USING SANDBOX MODE (Testnet) ***{RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}*** USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK ***{RESET}")

        # Load markets with retries
        lg.info(f"Attempting to load markets for {exchange.id}...")
        markets_loaded = False
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # Force reload on retries to get fresh data
                exchange.load_markets(reload=True if attempt > 0 else False)
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols found).")
                    markets_loaded = True
                    break
                else:
                    lg.warning(f"Market loading returned empty result (Attempt {attempt+1}/{MAX_API_RETRIES+1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    retry_wait = RETRY_DELAY_SECONDS * (attempt + 1)
                    lg.warning(f"Network error loading markets (Attempt {attempt+1}/{MAX_API_RETRIES+1}): {e}. Retrying in {retry_wait}s...")
                    time.sleep(retry_wait)
                else:
                    lg.critical(f"{NEON_RED}Fatal Error: Failed to load markets after {MAX_API_RETRIES+1} attempts due to network errors: {e}. Cannot continue. Exiting.{RESET}")
                    return None
            except ccxt.RateLimitExceeded as e:
                 # Wait longer for rate limits and don't count as a standard retry
                 wait = RETRY_DELAY_SECONDS * 5
                 lg.warning(f"Rate limit hit loading markets: {e}. Waiting {wait}s before retrying...")
                 time.sleep(wait)
                 # Continue loop without incrementing 'attempt' for rate limit
                 continue
            except ccxt.AuthenticationError as e:
                lg.critical(f"{NEON_RED}Fatal Error: Authentication failed while loading markets: {e}. Check API Key/Secret/Permissions. Exiting.{RESET}")
                return None
            except Exception as e:
                lg.critical(f"{NEON_RED}Fatal Error: An unexpected error occurred loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Fatal Error: Failed to load markets after all attempts. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} (Sandbox: {CONFIG.get('use_sandbox', True)})")

        # Attempt initial balance fetch to verify API key permissions
        lg.info(f"Attempting initial balance fetch for quote currency '{QUOTE_CURRENCY}'...")
        try:
            # Use the dedicated fetch_balance helper
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                lg.info(f"{NEON_GREEN}Initial balance check OK: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                # Balance fetch failed (might be permissions or currency not held)
                lg.critical(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}. Could not retrieve balance.{RESET}")
                if CONFIG.get('enable_trading', False):
                    lg.critical(f"{NEON_RED}Trading is ENABLED, but balance check failed. This may indicate API key permission issues for trading/balance endpoints. Cannot proceed safely. Exiting.{RESET}")
                    return None
                else:
                    lg.warning(f"{NEON_YELLOW}Trading is DISABLED. Proceeding cautiously despite balance check failure. Position sizing may fail later.{RESET}")
        except ccxt.AuthenticationError as auth_err:
             # Catch auth errors specifically from fetch_balance too
             lg.critical(f"{NEON_RED}Fatal Error: Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
             return None
        except Exception as balance_err:
             # Catch other unexpected errors during balance fetch
             lg.warning(f"{NEON_YELLOW}Unexpected error during initial balance fetch: {balance_err}.{RESET}", exc_info=True)
             if CONFIG.get('enable_trading', False):
                 lg.critical(f"{NEON_RED}Trading is ENABLED, and a critical error occurred during balance check. Exiting.{RESET}")
                 return None
             else:
                 lg.warning(f"{NEON_YELLOW}Trading is DISABLED. Proceeding despite unexpected balance check error.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}Fatal Error: Failed to initialize exchange due to Authentication Error: {e}. Check API Key/Secret.{RESET}")
        return None
    except Exception as e:
        lg.critical(f"{NEON_RED}Fatal Error: Failed to initialize exchange: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using CCXT's fetch_ticker with retries.
    Prioritizes 'last' price, falls back to mid-price (bid+ask)/2, then ask, then bid.

    Args:
        exchange: The initialized CCXT exchange object.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        logger: The logger instance for logging messages.

    Returns:
        The current market price as a Decimal, or None if fetching fails or no valid price found.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price: Optional[Decimal] = None

            # Helper to safely convert ticker price values to Decimal
            def safe_decimal_price(value: Optional[Union[str, float, int]], name: str) -> Optional[Decimal]:
                """Converts price value to Decimal, returns None if invalid or non-positive."""
                try:
                    if value is not None and str(value).strip() != '':
                        dec_val = Decimal(str(value))
                        # Ensure price is positive
                        if dec_val > Decimal('0'):
                            return dec_val
                        else:
                           lg.debug(f"Ticker field '{name}' is zero or negative ('{value}'). Discarding.")
                           return None
                except (InvalidOperation, ValueError, TypeError):
                    lg.debug(f"Could not convert ticker field '{name}' value '{value}' to Decimal.")
                    return None
                return None # If value is None or empty string

            # 1. Try 'last' price
            price = safe_decimal_price(ticker.get('last'), 'last')

            # 2. Fallback to mid-price if 'last' is invalid
            if price is None:
                bid = safe_decimal_price(ticker.get('bid'), 'bid')
                ask = safe_decimal_price(ticker.get('ask'), 'ask')
                if bid and ask and ask >= bid: # Ensure ask is not lower than bid
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price fallback for {symbol}: ({bid} + {ask}) / 2 = {price.normalize()}")
                # 3. Fallback to 'ask' price if mid-price calculation failed
                elif ask:
                    price = ask
                    lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback for {symbol}: {price.normalize()}{RESET}")
                # 4. Fallback to 'bid' price if only bid is available
                elif bid:
                    price = bid
                    lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback for {symbol}: {price.normalize()}{RESET}")

            # Check if a valid price was determined
            if price:
                lg.debug(f"Current price successfully fetched for {symbol}: {price.normalize()}")
                return price
            else:
                lg.warning(f"No valid price (last, mid, ask, bid) found in ticker (Attempt {attempts + 1}). Ticker data: {ticker}")
                # Continue to retry if attempts remain

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5 # Wait longer for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count for rate limit, just wait and retry
        except ccxt.ExchangeError as e:
            # Exchange errors fetching tickers are often not retryable (e.g., invalid symbol)
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            return None # Assume fatal for this operation
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Unexpected errors are likely fatal for this operation

        # Increment attempt count and implement exponential backoff for retries
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries and robust processing.
    Handles Bybit V5 'category' parameter, validates fetched data, converts to Decimal,
    checks for timestamp lag, cleans data, and caps DataFrame length.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '5m', '1h').
        limit: Desired number of klines. The actual request will be capped at BYBIT_API_KLINE_LIMIT.
        logger: Logger instance for logging messages.

    Returns:
        A pandas DataFrame containing the OHLCV data with a DatetimeIndex (UTC) and Decimal values,
        sorted chronologically. Returns an empty DataFrame if fetching or processing fails.
    """
    lg = logger
    if not exchange.has.get('fetchOHLCV'):
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV method.")
        return pd.DataFrame()

    ohlcv_data: Optional[List[List[Union[int, float, str]]]] = None
    # Determine the actual limit for the API request, capped by the exchange's limit
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
        lg.debug(f"Requested kline limit {limit} exceeds API limit {BYBIT_API_KLINE_LIMIT}. Requesting {actual_request_limit}.")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching klines for {symbol} ({timeframe}), Request Limit={actual_request_limit} (Attempt {attempts+1}/{MAX_API_RETRIES+1})")
            params = {}
            # Add 'category' parameter for Bybit V5 unified/contract accounts
            if 'bybit' in exchange.id.lower():
                 try:
                     # Use cached market info if possible, else fetch
                     market = exchange.market(symbol)
                     # Determine category based on market type (linear, inverse, spot)
                     # Default to linear if type is ambiguous
                     category = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse') else 'spot' if market.get('spot') else 'linear'
                     params['category'] = category
                     lg.debug(f"Using category '{category}' for Bybit kline fetch based on market type.")
                 except Exception as e:
                     lg.warning(f"Could not automatically determine market category for {symbol} kline fetch: {e}. Letting CCXT handle default.")
                     # Let ccxt attempt the call without the category parameter if lookup fails

            # Fetch OHLCV data using CCXT
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit, params=params)
            received_count = len(ohlcv_data) if ohlcv_data else 0
            lg.debug(f"Received {received_count} raw candle entries (requested {actual_request_limit}).")

            # Check if API limit was hit when more data might have been needed by the strategy
            if received_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT:
                lg.warning(f"{NEON_YELLOW}Hit API kline limit ({BYBIT_API_KLINE_LIMIT}). Strategy requested {limit} candles, but only received {received_count}. Results might be affected if strategy requires more lookback.{RESET}")

            if ohlcv_data and received_count > 0:
                # --- Validate timestamp lag of the last candle ---
                try:
                    last_timestamp_ms = ohlcv_data[-1][0]
                    # Ensure timestamp is a valid number before conversion
                    if not isinstance(last_timestamp_ms, (int, float)) or last_timestamp_ms <= 0:
                         raise ValueError(f"Invalid last timestamp format: {last_timestamp_ms}")

                    last_dt_utc = pd.to_datetime(last_timestamp_ms, unit='ms', utc=True, errors='raise')
                    now_utc = pd.Timestamp.utcnow()

                    # Estimate interval duration in seconds for lag check
                    interval_seconds = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') and exchange.parse_timeframe(timeframe) else 300 # Default 5min if parse fails
                    # Allow lag up to 5 intervals or 5 minutes, whichever is greater, plus a small buffer
                    max_allowed_lag_seconds = max((interval_seconds * 5), 300) + 60 # Add 1 min buffer

                    lag_seconds = (now_utc - last_dt_utc).total_seconds()

                    if lag_seconds >= 0 and lag_seconds < max_allowed_lag_seconds:
                        lg.debug(f"Last kline timestamp {last_dt_utc} is recent (Lag: {lag_seconds:.1f}s <= Max Allowed: {max_allowed_lag_seconds}s). Data looks current.")
                        break # Data seems valid, exit retry loop
                    elif lag_seconds < 0:
                        lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_dt_utc} appears to be in the future? (Lag: {lag_seconds:.1f}s). Proceeding, but check system clock.{RESET}")
                        break # Proceed but warn
                    else: # lag_seconds >= max_allowed_lag_seconds
                        lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_dt_utc} seems too old (Lag: {lag_seconds:.1f}s > Max Allowed: {max_allowed_lag_seconds}s). Retrying fetch...{RESET}")
                        ohlcv_data = None # Discard potentially stale data and force retry
                except (ValueError, TypeError, IndexError, pd.errors.OutOfBoundsDatetime) as ts_err:
                    lg.warning(f"Could not validate timestamp lag due to error: {ts_err}. Proceeding with fetched data, but it might be stale.")
                    break # Proceed even if validation fails, but log warning
                except Exception as ts_err:
                     lg.warning(f"Unexpected error during timestamp lag validation: {ts_err}. Proceeding.")
                     break # Proceed on unexpected errors
            else:
                lg.warning(f"No kline data received from API (Attempt {attempts+1}/{MAX_API_RETRIES+1}). Retrying...")
                # ohlcv_data is already None or empty list, loop will continue

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES:
                retry_wait = RETRY_DELAY_SECONDS * (attempts + 1) # Exponential backoff
                lg.warning(f"Network error fetching klines for {symbol} ({timeframe}): {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1} in {retry_wait}s...")
                time.sleep(retry_wait)
            else:
                lg.error(f"{NEON_RED}Max retries exceeded for network errors fetching klines: {e}. Cannot fetch data.{RESET}")
                return pd.DataFrame() # Return empty DF on persistent network failure
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count for rate limit
        except ccxt.ExchangeError as e:
            # Some exchange errors might be retryable, but many (like invalid symbol) are not.
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol} ({timeframe}): {e}{RESET}")
            # Consider adding checks for specific retryable error codes if known.
            # For now, assume most exchange errors fetching klines are fatal for this cycle.
            return pd.DataFrame()
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol} ({timeframe}): {e}{RESET}", exc_info=True)
            return pd.DataFrame() # Return empty DF on unexpected errors

        attempts += 1
        # Only sleep if we need to retry (i.e., ohlcv_data is still None)
        if attempts <= MAX_API_RETRIES and ohlcv_data is None:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    # --- Process Fetched Data into DataFrame ---
    if not ohlcv_data:
        lg.warning(f"Failed to fetch valid kline data for {symbol} ({timeframe}) after all retries.")
        return pd.DataFrame()

    try:
        lg.debug("Processing fetched kline data into pandas DataFrame...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Use standard column names, ensure we don't assume more columns than provided
        num_cols = len(ohlcv_data[0]) if ohlcv_data else 0
        df = pd.DataFrame(ohlcv_data, columns=cols[:num_cols])

        # Convert timestamp to DatetimeIndex (UTC), coercing errors
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp conversion failed
        if df.empty:
            lg.warning("Kline data became empty after timestamp conversion/dropna.")
            return pd.DataFrame()
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential non-numeric values robustly
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # 1. Convert to numeric first, making non-numerics NaN
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                # 2. Convert valid numbers (non-NaN, finite) to Decimal, keep others as NaN
                #    Using str() ensures exact representation for Decimal
                df[col] = numeric_series.apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                )
            elif col in ['open', 'high', 'low', 'close']: # Ensure essential OHLC columns exist
                 lg.error(f"Essential column '{col}' missing after DataFrame creation. Cannot proceed.")
                 return pd.DataFrame()

        # --- Clean Data ---
        initial_len = len(df)
        # Drop rows with NaN in essential OHLC columns
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive
        df = df[df['close'] > Decimal('0')]
        # Handle volume column if present
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            # Ensure volume is non-negative
            df = df[df['volume'] >= Decimal('0')]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid/NaN OHLCV data during cleaning.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} ({timeframe}) is empty after cleaning invalid rows.")
            return pd.DataFrame()

        # Sort by timestamp (should be sorted, but ensure) and remove duplicates
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        # Trim DataFrame length if it exceeds the internal maximum
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds internal max ({MAX_DF_LEN}). Trimming to keep most recent data.")
            df = df.iloc[-MAX_DF_LEN:]

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} ({timeframe})")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame: {e}{RESET}", exc_info=True)
        return pd.DataFrame() # Return empty DF on processing errors

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Retrieves and validates market information (precision, limits, contract type, etc.)
    from the CCXT exchange object. Includes retries for market loading if needed.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance for logging messages.

    Returns:
        A dictionary containing detailed market information if found and valid, otherwise None.
        The dictionary is augmented with 'is_contract', 'is_linear', 'is_inverse',
        and 'contract_type_str' keys for convenience.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            # Check if markets are loaded and contain the symbol. Reload if necessary.
            # Use `markets_by_id` as well for robustness, some exchanges might use ID lookup more reliably internally
            markets_available = exchange.markets and (symbol in exchange.markets or symbol in exchange.markets_by_id)

            if not markets_available:
                lg.info(f"Market info for '{symbol}' not found in currently loaded markets. Attempting to reload markets (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
                try:
                    exchange.load_markets(reload=True)
                    # Re-check after reload
                    markets_available = exchange.markets and (symbol in exchange.markets or symbol in exchange.markets_by_id)
                    if markets_available:
                         lg.info("Markets reloaded successfully.")
                    else:
                         # Only log error if reload *still* doesn't find the symbol
                         lg.error(f"Symbol '{symbol}' still not found after reloading markets.")
                except Exception as load_err:
                    lg.error(f"Failed to reload markets: {load_err}")
                    # If reload fails, it's unlikely we can get the market info later
                    return None

            # If symbol not found after potential reload, fail definitively
            if not markets_available:
                 # No need for more retries if symbol not found after reload attempt
                 lg.error(f"{NEON_RED}Market '{symbol}' not found on exchange '{exchange.id}' even after reloading.{RESET}")
                 return None

            # Retrieve market data using ccxt's safe method
            market = exchange.market(symbol)
            if market:
                # --- Augment market dictionary with derived fields ---
                market['is_contract'] = market.get('contract', False) or market.get('swap', False) or market.get('future', False)
                market['is_linear'] = market.get('linear', False) and market['is_contract']
                market['is_inverse'] = market.get('inverse', False) and market['is_contract']
                market['is_spot'] = market.get('spot', False) and not market['is_contract'] # Explicitly check spot and not contract

                # Determine contract type string
                if market['is_linear']: market['contract_type_str'] = "Linear Contract"
                elif market['is_inverse']: market['contract_type_str'] = "Inverse Contract"
                elif market['is_spot']: market['contract_type_str'] = "Spot"
                else: market['contract_type_str'] = "Unknown" # Should cover options or other types if exchange adds them

                # Log key market details for verification
                def format_precision_or_limit(value: Any) -> str:
                    """Safely formats market precision/limit values to string."""
                    try:
                        if value is None: return 'N/A'
                        # Handle potential scientific notation from JSON/API
                        dec_val = Decimal(str(value))
                        return f"{dec_val:.{abs(dec_val.normalize().as_tuple().exponent)}f}" if dec_val.normalize().as_tuple().exponent < 0 else str(dec_val.normalize())
                    except (InvalidOperation, ValueError, TypeError):
                        return f"Error({value})"

                precision = market.get('precision', {})
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})
                price_limits = limits.get('price', {}) # Added price limits

                lg.debug(f"Market Info Retrieved for: {symbol}")
                lg.debug(f"  ID: {market.get('id', 'N/A')}, Type: {market.get('type', 'N/A')}, Active: {market.get('active', 'N/A')}")
                lg.debug(f"  Market Type: {market['contract_type_str']}")
                lg.debug(f"  Base: {market.get('base', 'N/A')}, Quote: {market.get('quote', 'N/A')}, Settle: {market.get('settle', 'N/A')}")
                lg.debug(f"  Precision -> Price: {format_precision_or_limit(precision.get('price'))}, Amount: {format_precision_or_limit(precision.get('amount'))}")
                lg.debug(f"  Limits -> Amount (Min/Max): {format_precision_or_limit(amount_limits.get('min'))} / {format_precision_or_limit(amount_limits.get('max'))}")
                lg.debug(f"  Limits -> Cost (Min/Max): {format_precision_or_limit(cost_limits.get('min'))} / {format_precision_or_limit(cost_limits.get('max'))}")
                lg.debug(f"  Limits -> Price (Min/Max): {format_precision_or_limit(price_limits.get('min'))} / {format_precision_or_limit(price_limits.get('max'))}")
                # Contract size defaults to 1 for spot/linear, important for inverse
                contract_size_val = market.get('contractSize', 1 if not market['is_inverse'] else None)
                lg.debug(f"  Contract Size: {format_precision_or_limit(contract_size_val)}")

                # --- Critical Validation: Precision and Limits ---
                price_prec = precision.get('price')
                amount_prec = precision.get('amount')
                min_amount = amount_limits.get('min')
                # Contract size is crucial for contracts, especially inverse
                contract_size = contract_size_val if market['is_contract'] else 1

                missing_critical_info = False
                if price_prec is None:
                    lg.error(f"{NEON_RED}CRITICAL: Market {symbol} is missing required PRICE precision information! Trading will likely fail.{RESET}")
                    missing_critical_info = True
                if amount_prec is None:
                    lg.error(f"{NEON_RED}CRITICAL: Market {symbol} is missing required AMOUNT precision information! Trading will likely fail.{RESET}")
                    missing_critical_info = True
                if min_amount is None:
                    lg.warning(f"{NEON_YELLOW}Market {symbol} is missing MINIMUM AMOUNT limit information. Using 0, but sizing might be inaccurate.{RESET}")
                    # Allow proceeding but with caution
                if market['is_contract'] and contract_size is None:
                     lg.error(f"{NEON_RED}CRITICAL: Market {symbol} is a contract but missing CONTRACT SIZE information! Trading will likely fail.{RESET}")
                     missing_critical_info = True

                if missing_critical_info:
                    # If critical info is missing, do not return the market object
                    return None

                # If all checks pass, return the augmented market dictionary
                return market
            else:
                # This case should not be reachable if markets_available check passed, but handle defensively
                lg.error(f"Market dictionary is None for {symbol} even though it was found in exchange.markets. Potential CCXT issue.")
                return None

        except ccxt.BadSymbol as e:
            # Symbol format is wrong or not supported by the exchange
            lg.error(f"Invalid symbol format or symbol not supported by {exchange.id}: {e}")
            return None # Not retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES:
                retry_wait = RETRY_DELAY_SECONDS * (attempts + 1)
                lg.warning(f"Network error getting market info for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1} in {retry_wait}s...")
                time.sleep(retry_wait)
            else:
                lg.error(f"Max retries exceeded for network errors getting market info for {symbol}: {e}")
                return None
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"Rate limit exceeded getting market info: {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.ExchangeError as e:
            # General exchange errors might indicate temporary issues or config problems
            lg.error(f"Exchange error getting market info for {symbol}: {e}")
            # Could retry specific exchange errors if known to be transient, but assume fatal for now
            return None
        except Exception as e:
            lg.error(f"Unexpected error getting market info for {symbol}: {e}", exc_info=True)
            return None

        attempts += 1

    lg.error(f"Failed to retrieve market info for {symbol} after all attempts.")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency using CCXT.
    Handles Bybit V5 account types (UNIFIED, CONTRACT) and includes retries.

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency code (e.g., 'USDT', 'BTC'). Case-sensitive, should match exchange's representation.
        logger: Logger instance for logging messages.

    Returns:
        The available balance as a Decimal if found and valid, otherwise None.
        Checks 'free' balance, which typically represents available funds.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info: Optional[Dict] = None
            balance_str: Optional[str] = None
            found: bool = False
            account_types_to_check = ['UNIFIED', 'CONTRACT'] # Bybit V5 primary account types for derivatives/unified

            # --- Attempt 1: Fetch balance using Bybit V5 specific account types ---
            if 'bybit' in exchange.id.lower():
                for acc_type in account_types_to_check:
                    try:
                        lg.debug(f"Fetching balance for {currency} (Account Type: {acc_type}, Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
                        # Use fetch_balance with params for Bybit V5
                        params = {'accountType': acc_type}
                        info = exchange.fetch_balance(params=params)

                        # Check standard CCXT structure first ('free' balance)
                        if currency in info and info[currency].get('free') is not None:
                            balance_str = str(info[currency]['free'])
                            lg.debug(f"Found balance in standard structure for {acc_type}: {balance_str}")
                            found = True
                            break # Found balance, no need to check raw 'info' or other types

                        # Check Bybit's raw 'info' structure as a fallback (less standardized)
                        # Structure: info['info']['result']['list'][0]['coin'] is a list of coins
                        elif 'info' in info and 'result' in info['info'] and isinstance(info['info']['result'].get('list'), list):
                            for account_details in info['info']['result']['list']:
                                # Check if account type matches or if type is not specified in response item
                                # Also ensure 'coin' key exists and is a list
                                if (account_details.get('accountType') == acc_type or acc_type == 'UNIFIED') \
                                   and isinstance(account_details.get('coin'), list):
                                    for coin_data in account_details['coin']:
                                        if coin_data.get('coin') == currency:
                                            # Prioritize keys likely representing available balance
                                            free_balance = coin_data.get('availableToWithdraw') or \
                                                           coin_data.get('availableBalance') or \
                                                           coin_data.get('walletBalance') # Less ideal fallback
                                            if free_balance is not None:
                                                balance_str = str(free_balance)
                                                lg.debug(f"Found balance in raw 'info' structure for {acc_type} / {currency}: {balance_str}")
                                                found = True
                                                break # Found coin balance
                                    if found: break # Found in this account type's list
                            if found: break # Found balance, exit account type loop
                    except ccxt.ExchangeError as e:
                         # Ignore errors like "account type not supported" and try next type or default fetch
                         lg.debug(f"Exchange error fetching balance for type {acc_type} (may be expected): {e}. Trying next...")
                    except Exception as e:
                         lg.warning(f"Unexpected error fetching balance for type {acc_type}: {e}. Trying next...")

            # --- Attempt 2: If not found in specific types or not Bybit, try default fetch_balance ---
            if not found:
                try:
                    lg.debug(f"Fetching default balance for {currency} (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
                    info = exchange.fetch_balance() # No params
                    # Check standard structure
                    if currency in info and info[currency].get('free') is not None:
                        balance_str = str(info[currency]['free'])
                        lg.debug(f"Found balance in default fetch (standard structure): {balance_str}")
                        found = True
                    # Check raw structure as fallback (less common for default fetch)
                    elif 'info' in info and 'result' in info['info'] and isinstance(info['info']['result'].get('list'), list):
                         for account_details in info['info']['result']['list']:
                             if isinstance(account_details.get('coin'), list):
                                 for coin_data in account_details['coin']:
                                     if coin_data.get('coin') == currency:
                                         free_balance = coin_data.get('availableToWithdraw') or \
                                                        coin_data.get('availableBalance') or \
                                                        coin_data.get('walletBalance')
                                         if free_balance is not None:
                                             balance_str = str(free_balance)
                                             lg.debug(f"Found balance in default fetch (raw structure fallback): {balance_str}")
                                             found = True
                                             break
                                 if found: break
                             if found: break
                except Exception as e:
                    lg.error(f"Failed default balance fetch attempt: {e}", exc_info=False) # Don't need full trace usually

            # --- Process the found balance string ---
            if found and balance_str is not None:
                try:
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative (can happen with rounding or reporting issues)
                    return balance_decimal if balance_decimal >= Decimal('0') else Decimal('0')
                except (InvalidOperation, ValueError, TypeError) as conv_err:
                    # If conversion fails, raise as ExchangeError to trigger retry potentially
                    lg.error(f"Failed to convert balance string '{balance_str}' for {currency} to Decimal: {conv_err}")
                    # Raise error to potentially retry fetching
                    raise ccxt.ExchangeError(f"Balance conversion error for {currency}") from conv_err
            else:
                # If currency wasn't found after all checks, raise ExchangeError to trigger retry
                lg.warning(f"Balance for currency '{currency}' not found in fetch_balance response (Attempt {attempts+1}).")
                raise ccxt.ExchangeError(f"Balance for currency '{currency}' not found.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.AuthenticationError as e:
            # Authentication errors are critical and not retryable
            lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions. Stopping balance fetch.{RESET}")
            return None # Return None immediately
        except ccxt.ExchangeError as e:
            # Log specific exchange errors and decide whether to retry
            lg.warning(f"{NEON_YELLOW}Exchange error during balance fetch: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Could add checks here for specific non-retryable error codes from the exchange if known
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            # Unexpected errors might be retryable, but log as error

        # Increment attempt count and wait before next retry
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using CCXT's fetch_positions.
    Handles Bybit V5 parameters (category, symbol), parses position details robustly,
    and determines if a position is effectively open based on size threshold.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance for logging messages.

    Returns:
        A dictionary containing details of the open position if found (size > threshold),
        otherwise None. The dictionary includes standardized fields and a 'size_decimal' key.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching position status for {symbol} (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            positions: List[Dict] = []
            market_id: Optional[str] = None
            category: Optional[str] = None

            # --- Get Market Details for Bybit V5 API Call ---
            try:
                # Use cached market info if possible
                market = exchange.market(symbol)
                market_id = market['id']
                # Determine category, default to linear
                category = 'linear' if market.get('linear', True) else 'inverse' if market.get('inverse') else 'spot' if market.get('spot') else 'linear'
            except Exception as market_err:
                 lg.warning(f"Could not get market details ({market_err}) before fetching position. Proceeding without category/symbol filter if possible.")
                 # Allow fallback to fetch_positions without symbol filtering if market lookup fails

            # --- Fetch Positions using Bybit V5 Parameters ---
            try:
                # Bybit V5 often requires category and symbol for specific position fetch
                if market_id and category and 'bybit' in exchange.id.lower():
                    params = {'category': category, 'symbol': market_id}
                    lg.debug(f"Fetching specific position using params: {params}")
                    # Use fetch_position (singular) if available and preferred for single symbol
                    if exchange.has.get('fetchPosition'):
                         single_pos = exchange.fetch_position(symbol, params=params)
                         # fetch_position might return the position dict directly or raise exception if none
                         # Need to wrap in a list for consistent processing below
                         positions = [single_pos] if single_pos else []
                    else:
                         # Fallback to fetch_positions with a symbol list
                         positions = exchange.fetch_positions([symbol], params=params)
                else:
                    # Fallback if not Bybit or market info unavailable: Fetch all positions (less efficient)
                    lg.debug("Fetching all positions (no specific symbol/category filter).")
                    all_positions = exchange.fetch_positions() # May need params={} for some exchanges
                    # Filter locally for the desired symbol using standard 'symbol' and raw 'info.symbol' ID
                    positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]

            except ccxt.ArgumentsRequired as e:
                 # Handle cases where exchange *requires* symbol/category but we couldn't provide it
                 lg.warning(f"ArgumentsRequired fetching position ({e}). Fetching all positions for category '{category}' (if known) as fallback.")
                 params = {'category': category} if category else {}
                 all_positions = exchange.fetch_positions(params=params)
                 positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
            except (ccxt.ExchangeError, ccxt.NetworkError) as e:
                 # Handle specific Bybit error code for "position not found" gracefully
                 # Also check common strings indicating no position
                 no_pos_indicators = ["position not found", "no position found", "position idx not match position"]
                 if (hasattr(e, 'code') and e.code == 110025) or any(indicator in str(e).lower() for indicator in no_pos_indicators):
                     lg.info(f"No open position found for {symbol} based on API response ({e}).")
                     return None # Explicitly no position found
                 else:
                     # Re-raise other exchange/network errors to be caught by the outer retry loop
                     raise e

            # --- Process Fetched Positions to Find an Active One ---
            active_position: Optional[Dict] = None
            # Define a small threshold to consider a position 'open' (ignore dust)
            # Use amount precision if available, otherwise a very small default Decimal
            size_threshold = Decimal('1e-9') # Default tiny threshold
            try:
                market = exchange.market(symbol) # Re-fetch if needed
                amount_precision_str = market['precision']['amount']
                if amount_precision_str:
                    # Use a fraction (e.g., 10%) of the smallest amount step as the threshold
                    size_threshold = Decimal(str(amount_precision_str)) * Decimal('0.1')
                    # Ensure threshold itself is positive
                    if size_threshold <= 0: size_threshold = Decimal('1e-9')
            except Exception as prec_err:
                lg.warning(f"Could not get amount precision for {symbol} to set size threshold ({prec_err}). Using default threshold: {size_threshold}.")
            lg.debug(f"Using position size threshold: {size_threshold} (Size must be > this value)")

            # Iterate through the list of position entries returned by the API
            for pos_entry in positions:
                if not isinstance(pos_entry, dict):
                    lg.debug(f"Skipping non-dictionary item in position list: {pos_entry}")
                    continue

                # Get size preferentially from 'info' (often more accurate/direct from exchange)
                # Fallback to standard CCXT fields like 'contracts' or 'contractSize' (less common for size)
                size_str = str(pos_entry.get('info', {}).get('size', pos_entry.get('contracts', '')))

                if not size_str or size_str.lower() == 'nan':
                    lg.debug(f"Skipping position entry with missing or invalid size string: '{size_str}'. Entry: {pos_entry.get('info', {})}")
                    continue

                # --- Safely parse and check position size ---
                try:
                    size = Decimal(size_str)
                    # Check if the *absolute* size is greater than the threshold
                    if abs(size) > size_threshold:
                         lg.debug(f"Found potentially active position entry: Size={size.normalize()}, Threshold={size_threshold}")
                         active_position = pos_entry
                         break # Found an active position matching the criteria, stop searching
                    else:
                         lg.debug(f"Position entry size {size.normalize()} is <= threshold {size_threshold}. Ignoring as dust/closed.")
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     lg.warning(f"Could not parse position size string '{size_str}' to Decimal: {parse_err}. Skipping this position entry.")
                     continue # Skip to the next position entry if size cannot be parsed

            # --- Format and Return the Active Position ---
            if active_position:
                # Create a copy to avoid modifying the cached CCXT response
                standardized_pos = active_position.copy()
                info = standardized_pos.get('info', {}) # Raw exchange response for fallbacks

                # Add parsed Decimal size (already validated > threshold)
                standardized_pos['size_decimal'] = size # Use the successfully parsed size from the loop

                # Standardize side ('long' or 'short')
                pos_side = standardized_pos.get('side') # CCXT standard field
                if pos_side not in ['long', 'short']:
                    # Infer side from Bybit V5 'info' ('Buy'/'Sell') or the sign of the size
                    side_v5 = info.get('side', '').lower()
                    if side_v5 == 'buy': pos_side = 'long'
                    elif side_v5 == 'sell': pos_side = 'short'
                    elif size > size_threshold: pos_side = 'long' # Positive size implies long
                    elif size < -size_threshold: pos_side = 'short' # Negative size implies short
                    else: pos_side = None # Should not happen if size > threshold, but check

                if not pos_side:
                    lg.error(f"Could not determine position side for {symbol} despite size {size}. Position info: {info}")
                    return None # Cannot proceed without a clear side

                standardized_pos['side'] = pos_side

                # Standardize other common fields, preferring CCXT standard fields but falling back to 'info'
                # Use helper for safe Decimal conversion
                def safe_decimal_from_pos(key_standard: str, key_info: Optional[str] = None) -> Optional[str]:
                     val = standardized_pos.get(key_standard)
                     if val is None and key_info: val = info.get(key_info)
                     try:
                         return str(Decimal(str(val))) if val is not None and str(val).strip() != '' else None
                     except (ValueError, InvalidOperation, TypeError): return None

                standardized_pos['entryPrice'] = safe_decimal_from_pos('entryPrice', 'avgPrice')
                standardized_pos['leverage'] = safe_decimal_from_pos('leverage') # Leverage often string/int
                standardized_pos['liquidationPrice'] = safe_decimal_from_pos('liquidationPrice', 'liqPrice')
                standardized_pos['unrealizedPnl'] = safe_decimal_from_pos('unrealizedPnl', 'unrealisedPnl') # Note Bybit spelling

                # Standardize SL/TP/TSL fields from 'info' (as these are often exchange-specific)
                # Ensure they are returned as strings if present and non-zero
                def format_protection_price(value: Any) -> Optional[str]:
                     s_val = str(value).strip()
                     if s_val and s_val != '0' and s_val != '0.0':
                         try: return str(Decimal(s_val)) # Validate and standardize format
                         except: return None
                     return None

                standardized_pos['stopLossPrice'] = format_protection_price(info.get('stopLoss'))
                standardized_pos['takeProfitPrice'] = format_protection_price(info.get('takeProfit'))
                standardized_pos['trailingStopLoss'] = format_protection_price(info.get('trailingStop')) # Bybit V5 TSL distance/offset
                standardized_pos['tslActivationPrice'] = format_protection_price(info.get('activePrice')) # Bybit V5 TSL activation price

                # Log the details of the found active position
                ep_str = standardized_pos.get('entryPrice', 'N/A')
                size_str_log = standardized_pos['size_decimal'].normalize()
                sl_str_log = standardized_pos.get('stopLossPrice', 'N/A')
                tp_str_log = standardized_pos.get('takeProfitPrice', 'N/A')
                tsl_dist_log = standardized_pos.get('trailingStopLoss', 'N/A')
                tsl_act_log = standardized_pos.get('tslActivationPrice', 'N/A')
                tsl_str_log = f"Active (Dist:{tsl_dist_log}, Act:{tsl_act_log})" if tsl_dist_log and tsl_dist_log != 'N/A' else "Inactive"

                lg.info(f"{NEON_GREEN}Active {pos_side.upper()} Position Found ({symbol}): Size={size_str_log}, Entry={ep_str}, SL={sl_str_log}, TP={tp_str_log}, TSL={tsl_str_log}{RESET}")
                return standardized_pos
            else:
                # No position entry met the size threshold check
                lg.info(f"No active position found for {symbol} (checked {len(positions)} entries, none exceeded size threshold {size_threshold}).")
                return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching positions: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.AuthenticationError as e:
            lg.critical(f"{NEON_RED}Authentication Error fetching positions: {e}. Check API permissions for position endpoints. Stopping position check.{RESET}")
            return None # Fatal for this operation
        except ccxt.ExchangeError as e:
            # Log other exchange errors and retry (unless identified as non-retryable above)
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Add checks for specific fatal error codes if necessary
        except Exception as e:
            # Catch unexpected errors during parsing or processing
            lg.error(f"{NEON_RED}Unexpected error processing position data: {e}{RESET}", exc_info=True)
            # Treat unexpected errors as potentially fatal for this check in the cycle
            return None

        # Increment attempt count and wait before retry
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets the leverage for a derivatives symbol using CCXT's set_leverage method.
    Includes Bybit V5 specific parameters and handles common success/failure responses.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        leverage: Desired leverage (integer, should be > 0 for setting).
        market_info: Dictionary containing market details (used for ID and type).
        logger: Logger instance for logging messages.

    Returns:
        True if leverage was set successfully OR was already set correctly, False otherwise.
    """
    lg = logger
    is_contract = market_info.get('is_contract', False)

    # Skip if not a contract market where leverage is applicable
    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # Consider success as no action was needed
    # Validate leverage input (must be positive to set)
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}) provided. Must be > 0.")
        return False
    # Check if the exchange supports setting leverage via CCXT standard method
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not report support for setLeverage via CCXT standard method.")
        # Could potentially add exchange-specific private calls here if needed
        return False

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {symbol} to {leverage}x (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            params = {}
            market_id = market_info['id'] # Use the exchange-specific market ID

            # --- Add Bybit V5 specific parameters ---
            if 'bybit' in exchange.id.lower():
                 # Determine category, default linear
                 category = 'linear' if market_info.get('linear', True) else 'inverse'
                 # Bybit V5 unified margin often requires setting both buy and sell leverage
                 # Note: Some accounts might use set_leverage on symbol only, others need buy/sell.
                 # Providing both is generally safer for V5 unified.
                 params = {
                     'category': category,
                     'symbol': market_id, # Explicitly pass symbol in params too for some versions/endpoints
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 params for setLeverage: {params}")

            # Call CCXT's set_leverage method (handles underlying API call)
            # Pass leverage as float/int, symbol as market_id, and specific params
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"Set leverage raw response: {response}") # Log raw response for debugging

            # --- Check Bybit V5 response code if available ---
            # CCXT often returns the parsed response which might include retCode for Bybit
            ret_code = None
            if isinstance(response, dict):
                 # Check standard 'info' field or direct key
                 ret_code = response.get('info', {}).get('retCode', response.get('retCode'))

            if ret_code is not None:
                 if ret_code == 0:
                     lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (Bybit Code 0).{RESET}")
                     return True
                 elif ret_code == 110045: # Bybit V5 code for "Leverage not modified"
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Bybit Code 110045 - Not Modified).{RESET}")
                     return True
                 else:
                     # Raise an error for other non-zero Bybit codes to potentially retry or fail
                     error_message = response.get('info', {}).get('retMsg', response.get('retMsg', 'Unknown Bybit API error'))
                     raise ccxt.ExchangeError(f"Bybit API error setting leverage: {error_message} (Code: {ret_code})")
            else:
                # If no retCode (e.g., non-Bybit exchange or older CCXT version),
                # assume success if no exception was raised (standard CCXT behavior)
                lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (No specific code in response, assumed OK).{RESET}")
                return True

        except ccxt.ExchangeError as e:
            # Handle specific exchange errors, especially those indicating success or non-retryable issues
            error_code_attr = getattr(e, 'code', None) # Bybit error code might be attached to the exception
            error_str = str(e).lower()
            # Extract Bybit code from message string if not directly available
            bybit_code_in_msg = None
            if "code=" in error_str:
                 try: bybit_code_in_msg = int(error_str.split("code=")[1].split(" ")[0].strip())
                 except: pass # Ignore parsing errors

            effective_code = error_code_attr or bybit_code_in_msg
            lg.error(f"{NEON_RED}Exchange error setting leverage: {e} (Code: {effective_code or 'N/A'}){RESET}")

            # Check for "Leverage not modified" or similar messages/codes indicating it's already set
            if effective_code == 110045 or "not modified" in error_str or "leverage is same" in error_str:
                lg.info(f"{NEON_YELLOW}Leverage already set correctly (inferred from error message/code).{RESET}")
                return True # Treat as success

            # List known fatal/non-retryable error codes or strings for leverage setting
            # (These might need adjustment based on exchange specifics and account type)
            fatal_codes = [110043, 110013, 10001, 10004, 30086, 30087, 3400041] # Example Bybit codes (margin mode, param error, risk limit)
            fatal_strings = ["margin mode cannot be switched", "position exists", "risk limit", "parameter error", "invalid leverage"]
            if effective_code in fatal_codes or any(s in error_str for s in fatal_strings):
                lg.error(" >> Hint: This leverage error seems non-retryable. Check position status, margin mode, risk limits, or leverage value.")
                return False # Non-retryable error

            # If error is potentially retryable and attempts remain
            elif attempts >= MAX_API_RETRIES:
                 lg.error("Max retries exceeded for ExchangeError setting leverage.")
                 return False
            # Otherwise, assume retryable (will loop)

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"Max retries exceeded for NetworkError setting leverage: {e}")
                 return False
            lg.warning(f"{NEON_YELLOW}Network error setting leverage (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
             wait_time = RETRY_DELAY_SECONDS * 5
             lg.warning(f"{NEON_YELLOW}Rate limit hit setting leverage: {e}. Waiting {wait_time}s...{RESET}")
             time.sleep(wait_time)
             continue # Don't increment attempt count
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting leverage: {e}{RESET}", exc_info=True)
            return False # Unexpected errors likely fatal for this operation

        # Increment attempt count and wait before retry (exponential backoff)
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: Dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on risk percentage, stop loss distance,
    account balance, and market constraints (precision, limits, contract type).
    Uses Decimal for all financial calculations.

    Args:
        balance: Available balance in the quote currency (Decimal).
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: Calculated stop loss price (Decimal).
        entry_price: Estimated entry price (Decimal).
        market_info: Dictionary containing market details from get_market_info.
        exchange: Initialized CCXT exchange object (used for formatting amount).
        logger: Logger instance for logging messages.

    Returns:
        The calculated position size as a Decimal, adjusted for market rules (precision, limits),
        or None if calculation is not possible, inputs are invalid, or size violates critical limits.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info.get('quote', 'QUOTE') # Fallback for logging if missing
    base_currency = market_info.get('base', 'BASE')   # Fallback for logging
    is_contract = market_info['is_contract']
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit for size (Contracts for derivatives, Base currency units for spot)
    size_unit = "Contracts" if is_contract else base_currency

    lg.info(f"Calculating Position Size for {symbol}:")
    lg.debug(f"  Inputs: Balance={balance.normalize()} {quote_currency}, Risk={risk_per_trade:.2%}, Entry={entry_price.normalize()}, SL={initial_stop_loss_price.normalize()}")

    # --- Initial Validations ---
    if balance <= Decimal('0'):
        lg.error(f"{NEON_RED}Position sizing failed: Balance is zero or negative ({balance} {quote_currency}).{RESET}")
        return None
    try:
        risk_decimal = Decimal(str(risk_per_trade))
        if not (Decimal('0') < risk_decimal <= Decimal('1')):
            raise ValueError("Risk per trade must be between 0 (exclusive) and 1 (inclusive)")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"{NEON_RED}Position sizing failed: Invalid risk_per_trade value '{risk_per_trade}': {e}.{RESET}")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"{NEON_RED}Position sizing failed: Entry price ({entry_price}) or SL price ({initial_stop_loss_price}) is zero or negative.{RESET}")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"{NEON_RED}Position sizing failed: Entry price and Stop Loss price cannot be the same ({entry_price}). Division by zero risk.{RESET}")
        return None

    # --- Extract Market Details Safely ---
    try:
        precision = market_info['precision']
        limits = market_info['limits']

        amount_precision_str = precision.get('amount')
        price_precision_str = precision.get('price') # Used for logging/context
        if amount_precision_str is None: raise ValueError("Amount precision ('precision.amount') missing in market info")
        if price_precision_str is None: raise ValueError("Price precision ('precision.price') missing in market info")

        # Smallest amount step (tick size for amount)
        amount_tick_size = Decimal(str(amount_precision_str))
        if amount_tick_size <= 0: raise ValueError(f"Invalid amount precision step size: {amount_tick_size}")

        # Amount Limits
        amount_limits = limits.get('amount', {})
        min_amount_str = amount_limits.get('min')
        max_amount_str = amount_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')
        if min_amount < 0: raise ValueError(f"Invalid minimum amount limit: {min_amount}")

        # Cost Limits
        cost_limits = limits.get('cost', {})
        min_cost_str = cost_limits.get('min')
        max_cost_str = cost_limits.get('max')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')
        if min_cost < 0: raise ValueError(f"Invalid minimum cost limit: {min_cost}")

        # Contract size (crucial for contracts, defaults to 1 for spot)
        contract_size_str = market_info.get('contractSize', '1') # Default to '1' if missing (common for spot/linear)
        contract_size = Decimal(str(contract_size_str)) if contract_size_str else Decimal('1')
        if contract_size <= 0: raise ValueError(f"Invalid contract size: {contract_size}")

        lg.debug(f"  Market Details: Amount Tick={amount_tick_size}, Min/Max Amount={min_amount}/{max_amount}")
        lg.debug(f"  Market Details: Min/Max Cost={min_cost}/{max_cost}, Contract Size={contract_size}")

    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"{NEON_RED}Position sizing failed: Error extracting required market details (precision/limits/contractSize): {e}.{RESET}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    risk_amount_quote = balance * risk_decimal
    stop_loss_distance_price = abs(entry_price - initial_stop_loss_price)
    if stop_loss_distance_price <= Decimal('0'):
        # Should be caught by initial validation, but double-check
        lg.error(f"Position sizing failed: Stop loss distance is zero or negative ({stop_loss_distance_price}).")
        return None

    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%}, Max Risk Amount: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"  SL Price Distance: {stop_loss_distance_price.normalize()}")

    # --- Calculate Initial Position Size Based on Risk ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear Contracts or Spot
            # Value change per unit (contract or base currency unit) = SL distance * contract size
            # For spot/linear, contractSize is typically 1 base unit value.
            value_change_per_unit = stop_loss_distance_price * contract_size
            # Avoid division by zero or extremely small values
            if value_change_per_unit.copy_abs() < Decimal('1e-18'):
                 lg.error(f"Position sizing failed (Linear/Spot): Calculated value change per unit is near zero ({value_change_per_unit}). Cannot determine size.")
                 return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Risk Calc: Size = Risk Amount / (SL Price Dist * Contract Size)")
            lg.debug(f"  -> Size = {risk_amount_quote} / ({stop_loss_distance_price} * {contract_size}) = {calculated_size}")
        else: # Inverse Contracts
            # Value change per contract = contract_size * abs(1/entry_price - 1/sl_price)
            if entry_price <= 0 or initial_stop_loss_price <= 0: # Should be caught earlier
                 lg.error("Position sizing failed (Inverse): Entry or SL price is zero or negative.")
                 return None
            # Use Decimal for precision in inverse calculation
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor.copy_abs() < Decimal('1e-18'):
                 lg.error(f"Position sizing failed (Inverse): Calculated inverse factor is near zero ({inverse_factor}). Cannot determine size.")
                 return None
            value_change_per_contract = contract_size * inverse_factor
            if value_change_per_contract.copy_abs() < Decimal('1e-18'):
                 lg.error(f"Position sizing failed (Inverse): Calculated value change per contract is near zero ({value_change_per_contract}).")
                 return None
            calculated_size = risk_amount_quote / value_change_per_contract
            lg.debug(f"  Inverse Risk Calc: Size = Risk Amount / (Contract Size * abs(1/Entry - 1/SL))")
            lg.debug(f"  -> Size = {risk_amount_quote} / ({contract_size} * {inverse_factor}) = {calculated_size}")

    except (OverflowError, InvalidOperation, ZeroDivisionError) as calc_err:
        lg.error(f"{NEON_RED}Position sizing failed during initial calculation: {calc_err}.{RESET}")
        return None

    # Ensure calculated size is positive
    if calculated_size <= 0:
        lg.error(f"{NEON_RED}Position sizing failed: Initial calculated size is zero or negative ({calculated_size}). Check inputs and risk settings.{RESET}")
        return None

    lg.info(f"  Initial Calculated Size (Pre-Limits): {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size for Market Limits (Amount and Cost) ---
    adjusted_size = calculated_size
    limit_adjustment_applied = False

    # 1. Apply Amount Limits (Min/Max Amount)
    if min_amount > 0 and adjusted_size < min_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size} is below minimum amount {min_amount}. Adjusting UP to minimum.{RESET}")
        adjusted_size = min_amount
        limit_adjustment_applied = True
    if max_amount.is_finite() and adjusted_size > max_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size} exceeds maximum amount {max_amount}. Adjusting DOWN to maximum.{RESET}")
        adjusted_size = max_amount
        limit_adjustment_applied = True

    # 2. Apply Cost Limits (Min/Max Cost) - Calculate estimated cost based on *adjusted* size
    estimated_cost = Decimal('0')
    try:
        if entry_price > 0:
            # Cost formula depends on contract type
            if not is_inverse: # Linear/Spot Cost = Size * Price * ContractSize
                estimated_cost = adjusted_size * entry_price * contract_size
            else: # Inverse Cost = Size * ContractSize / Price
                 estimated_cost = (adjusted_size * contract_size) / entry_price
        lg.debug(f"  Estimated Cost for size {adjusted_size.normalize()}: {estimated_cost.normalize()} {quote_currency}")

        # Check Minimum Cost
        if min_cost > 0 and estimated_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost} is below minimum cost {min_cost}. Attempting to increase size to meet min cost.{RESET}")
            required_size_for_min_cost = None
            try:
                # Calculate the size needed to meet the minimum cost
                if not is_inverse:
                    if entry_price > 0 and contract_size > 0:
                         required_size_for_min_cost = min_cost / (entry_price * contract_size)
                    else: raise ValueError("Cannot calculate required size for min cost (Linear/Spot) due to zero entry/contract size")
                else: # Inverse
                    if contract_size > 0:
                         required_size_for_min_cost = (min_cost * entry_price) / contract_size
                    else: raise ValueError("Cannot calculate required size for min cost (Inverse) due to zero contract size")

                lg.info(f"  Required size to meet min cost: {required_size_for_min_cost.normalize()} {size_unit}")
                # Ensure the required size doesn't violate max amount limit
                if max_amount.is_finite() and required_size_for_min_cost > max_amount:
                    lg.error(f"{NEON_RED}Cannot meet minimum cost ({min_cost}) without exceeding maximum amount limit ({max_amount}). Aborting trade.{RESET}")
                    return None
                # Adjust size up. Must also respect the minimum amount limit if it's larger.
                adjusted_size = max(min_amount, required_size_for_min_cost)
                limit_adjustment_applied = True
                lg.info(f"  Adjusted size to meet minimum cost: {adjusted_size.normalize()} {size_unit}")
            except (ValueError, OverflowError, InvalidOperation, ZeroDivisionError) as min_cost_err:
                lg.error(f"{NEON_RED}Failed to calculate required size for minimum cost ({min_cost}): {min_cost_err}. Aborting trade.{RESET}")
                return None

        # Check Maximum Cost
        elif max_cost.is_finite() and estimated_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost} exceeds maximum cost {max_cost}. Attempting to decrease size.{RESET}")
            max_size_for_max_cost = None
            try:
                # Calculate the maximum size allowed by the maximum cost
                if not is_inverse:
                    if entry_price > 0 and contract_size > 0:
                         max_size_for_max_cost = max_cost / (entry_price * contract_size)
                    else: raise ValueError("Cannot calculate max size for max cost (Linear/Spot) due to zero entry/contract size")
                else: # Inverse
                     if contract_size > 0:
                         max_size_for_max_cost = (max_cost * entry_price) / contract_size
                     else: raise ValueError("Cannot calculate max size for max cost (Inverse) due to zero contract size")

                lg.info(f"  Maximum size allowed by max cost: {max_size_for_max_cost.normalize()} {size_unit}")
                # Adjust size down. Must still be >= minimum amount limit.
                # Take the minimum of the current adjusted size and the max size allowed by cost.
                adjusted_size = max(min_amount, min(adjusted_size, max_size_for_max_cost))
                limit_adjustment_applied = True
                lg.info(f"  Adjusted size to meet maximum cost: {adjusted_size.normalize()} {size_unit}")
            except (ValueError, OverflowError, InvalidOperation, ZeroDivisionError) as max_cost_err:
                 lg.error(f"{NEON_RED}Failed to calculate maximum size for maximum cost ({max_cost}): {max_cost_err}. Aborting trade.{RESET}")
                 return None

    except (OverflowError, InvalidOperation, ZeroDivisionError) as cost_calc_err:
         lg.error(f"Error during cost calculation/adjustment: {cost_calc_err}. Proceeding without cost limit adjustment, but limits might be violated.")

    if limit_adjustment_applied:
         lg.debug(f"  Size after Amount/Cost Limit Adjustments: {adjusted_size.normalize()} {size_unit}")

    # --- Apply Amount Precision (Tick Size) ---
    # This should be the *last* step before final validation
    final_size = adjusted_size
    try:
        # Use CCXT's amount_to_precision for reliable formatting according to exchange rules
        # It typically handles rounding/truncating based on market['precision']['mode']
        formatted_amount_str = exchange.amount_to_precision(symbol, float(adjusted_size))
        final_size = Decimal(formatted_amount_str)

        # Log if precision adjustment changed the value
        if final_size != adjusted_size:
             lg.info(f"Applied amount precision (Tick Size: {amount_tick_size}): {adjusted_size} -> {final_size}")

    except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as fmt_err:
        lg.warning(f"CCXT amount_to_precision failed: {fmt_err}. Attempting manual rounding DOWN as fallback.")
        # Manual rounding down using the amount tick size as a fallback
        try:
            if amount_tick_size > 0:
                # Floor division to get the number of ticks, then multiply back
                final_size = (adjusted_size // amount_tick_size) * amount_tick_size
                if final_size != adjusted_size:
                    lg.info(f"Applied manual amount precision (Floor): {adjusted_size} -> {final_size}")
            else:
                # Should not happen due to earlier checks, but handle defensively
                lg.error("Amount tick size is zero during manual precision step. Using unrounded size.")
                final_size = adjusted_size # Keep unrounded
        except (InvalidOperation, TypeError) as manual_err:
            lg.error(f"Manual precision rounding failed: {manual_err}. Using unrounded size: {adjusted_size}")
            final_size = adjusted_size # Keep unrounded

    # --- Final Validation Checks ---
    # 1. Check if size became zero or negative after precision
    if final_size <= 0:
        lg.error(f"{NEON_RED}Position sizing failed: Final size is zero or negative ({final_size}) after applying precision. Aborting trade.{RESET}")
        return None
    # 2. Check if final size is still above minimum amount limit
    if min_amount > 0 and final_size < min_amount:
        lg.error(f"{NEON_RED}Position sizing failed: Final size {final_size} is below minimum amount {min_amount} after precision adjustments. Aborting trade.{RESET}")
        # Consider attempting to bump up to min_amount if feasible, but safer to abort here.
        # Could add: final_size = min_amount, then re-check cost, but increases complexity.
        return None
    # 3. Re-check cost limits with the *final precise size*
    final_cost = Decimal('0')
    try:
        if entry_price > 0:
            final_cost = (final_size * entry_price * contract_size) if not is_inverse else ((final_size * contract_size) / entry_price)
        lg.debug(f"  Final Cost Check for size {final_size.normalize()}: {final_cost.normalize()} {quote_currency}")

        # Check Min Cost again (precision might have pushed it slightly below)
        if min_cost > 0 and final_cost < min_cost:
            # Check if the difference is very small (e.g., less than 1% of min_cost) - might be rounding artifact
            cost_diff_ratio = (min_cost - final_cost) / min_cost if min_cost > 0 else Decimal(1)
            if cost_diff_ratio < Decimal("0.01"): # Allow small discrepancy
                 lg.debug(f"Final cost {final_cost} is slightly below min cost {min_cost} after precision, but within tolerance. Proceeding.")
            else:
                 # Try bumping size by one tick to meet min cost if possible
                 lg.warning(f"{NEON_YELLOW}Final cost {final_cost} is below minimum {min_cost} after precision. Attempting to bump size by one step.{RESET}")
                 try:
                     next_step_size = final_size + amount_tick_size
                     next_step_cost = Decimal('0')
                     if entry_price > 0:
                         next_step_cost = (next_step_size * entry_price * contract_size) if not is_inverse else ((next_step_size * contract_size) / entry_price)

                     # Check if the next step size is valid (meets min cost, doesn't exceed max amount/cost)
                     is_valid_next_step = (next_step_cost >= min_cost) and \
                                          (not max_amount.is_finite() or next_step_size <= max_amount) and \
                                          (not max_cost.is_finite() or next_step_cost <= max_cost)

                     if is_valid_next_step:
                         lg.warning(f"{NEON_YELLOW}Bumping size by one step to {next_step_size.normalize()} to meet min cost {min_cost} (New cost: {next_step_cost.normalize()}).{RESET}")
                         final_size = next_step_size
                     else:
                         lg.error(f"{NEON_RED}Cannot meet minimum cost ({min_cost}). Increasing size by one step to {next_step_size} would violate other limits (Cost: {next_step_cost}, MaxAmt: {max_amount}, MaxCost: {max_cost}). Aborting trade.{RESET}")
                         return None
                 except Exception as bump_err:
                     lg.error(f"{NEON_RED}Error attempting to bump size for minimum cost: {bump_err}. Aborting trade.{RESET}")
                     return None

        # Check Max Cost again
        elif max_cost.is_finite() and final_cost > max_cost:
             # This shouldn't happen if previous checks were correct, but double-check
             lg.error(f"{NEON_RED}Position sizing failed: Final cost {final_cost} exceeds maximum cost {max_cost} after precision adjustments. Aborting trade.{RESET}")
             return None

    except (OverflowError, InvalidOperation, ZeroDivisionError) as final_cost_err:
         lg.error(f"Error during final cost check: {final_cost_err}. Proceeding with calculated size, but cost limits might be violated.")


    lg.info(f"{NEON_GREEN}{BRIGHT}>>> Final Calculated Position Size: {final_size.normalize()} {size_unit} <<< {RESET}")
    return final_size

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: Dict,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Places a market order using CCXT's create_order method with retries.
    Handles Bybit V5 specific parameters ('category', 'positionIdx', 'reduceOnly', 'timeInForce').

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        trade_signal: The signal determining the side ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The size of the order (Decimal, always positive).
        market_info: Dictionary containing market details.
        logger: Logger instance for logging messages.
        reduce_only: If True, set the reduceOnly flag (primarily for closing/reducing positions).
        params: Optional dictionary of additional parameters to pass to create_order (will be merged with defaults).

    Returns:
        The order dictionary returned by CCXT upon successful placement, otherwise None.
    """
    lg = logger
    # Map signals to CCXT sides ('buy' or 'sell')
    # EXIT_SHORT means buying back, EXIT_LONG means selling
    side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}
    side = side_map.get(trade_signal.upper())

    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided to place_trade function.")
        return None
    if position_size <= 0:
        lg.error(f"Invalid position size ({position_size}) provided to place_trade. Size must be positive.")
        return None

    order_type = 'market'
    is_contract = market_info['is_contract']
    base_currency = market_info.get('base', 'BASE')
    # Determine unit for logging/context (settle currency for contracts, base for spot)
    size_unit = market_info.get('settle', base_currency) if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info['id'] # Use exchange-specific ID for the order

    # --- Prepare Base Order Arguments for CCXT ---
    order_args = {
        'symbol': market_id,
        'type': order_type,
        'side': side,
        'amount': float(position_size) # CCXT generally expects float for amount in create_order
    }

    # --- Prepare Exchange-Specific Parameters (Especially for Bybit V5) ---
    order_params = {}
    if 'bybit' in exchange.id.lower():
        # Determine category, default linear
        category = 'linear' if market_info.get('linear', True) else 'inverse'
        # Common Bybit V5 parameters for market orders
        order_params = {
            'category': category,
            'positionIdx': 0  # Assume one-way mode (0 for unified/classic hedge mode needs 1 or 2)
            # 'qty': str(position_size) # Some Bybit endpoints might prefer 'qty' as string
        }
        if reduce_only:
            order_params['reduceOnly'] = True
            # Use IOC for reduceOnly market orders to prevent resting orders if partially filled or price moves
            # FOK (FillOrKill) is another option, but IOC is generally safer for market closes.
            order_params['timeInForce'] = 'IOC' # ImmediateOrCancel
        lg.debug(f"Using Bybit V5 params for order: {order_params}")

    # Merge any additional custom params provided by the caller
    if params:
        # Ensure custom params don't overwrite essential ones unless intended
        for key, value in params.items():
             if key in order_params:
                 lg.debug(f"Custom param '{key}' overwrites default param in place_trade.")
             order_params[key] = value
        # order_params.update(params) # Simpler merge if overwriting is acceptable

    # Add the combined parameters dictionary to the main order arguments if not empty
    if order_params:
        order_args['params'] = order_params

    lg.info(f"Attempting to place {BRIGHT}{action_desc} {side.upper()} {order_type}{RESET} order:")
    lg.info(f"  Symbol: {symbol} ({market_id})")
    lg.info(f"  Size: {position_size.normalize()} {size_unit}")
    if reduce_only: lg.info(f"  Reduce Only: {True}")
    if order_args.get('params'): lg.debug(f"  Full Order Params Sent: {order_args['params']}")

    # --- Execute Order Placement with Retries ---
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            # Place the order using CCXT's method
            order = exchange.create_order(**order_args)
            lg.debug(f"Create order raw response: {order}") # Log the full response

            # --- Log Success with Key Order Details ---
            order_id = order.get('id', 'N/A')
            status = order.get('status', 'N/A') # e.g., 'open', 'closed', 'canceled'
            avg_price_str = str(order.get('average', 'N/A')) # Average fill price
            filled_amount_str = str(order.get('filled', 'N/A')) # Amount filled

            # Format numbers safely for logging
            try: avg_price_log = f", AvgPrice: ~{Decimal(avg_price_str).normalize()}" if avg_price_str != 'N/A' else ""
            except: avg_price_log = f", AvgPrice: {avg_price_str}"
            try: filled_amount_log = f", Filled: {Decimal(filled_amount_str).normalize()}" if filled_amount_str != 'N/A' else ""
            except: filled_amount_log = f", Filled: {filled_amount_str}"

            # Log based on status (market orders might close immediately)
            if status == 'closed':
                 log_msg = f"{NEON_GREEN}{action_desc} Order Placed & Filled!{RESET} ID: {order_id}, Status: {status}{avg_price_log}{filled_amount_log}"
                 lg.info(log_msg)
            else: # 'open' or other status
                 log_msg = f"{NEON_GREEN}{action_desc} Order Submitted!{RESET} ID: {order_id}, Status: {status}. Check fill status."
                 lg.info(log_msg)

            return order # Return the successful order dictionary

        # --- Specific CCXT Exception Handling ---
        except ccxt.InsufficientFunds as e:
            # This is usually not retryable without a change in balance.
            lg.error(f"{NEON_RED}Order Placement Failed ({action_desc} {symbol}): Insufficient funds. {e}{RESET}")
            return None # Fail immediately
        except ccxt.InvalidOrder as e:
            # Often indicates issues with parameters, size, price, or market state. Usually not retryable.
            lg.error(f"{NEON_RED}Order Placement Failed ({action_desc} {symbol}): Invalid order parameters or violation. {e}{RESET}")
            lg.error(f"  Order Arguments Used: {order_args}")
            # Add hints based on common InvalidOrder reasons
            err_str = str(e).lower()
            if "size" in err_str or "qty" in err_str or "quantity" in err_str: lg.error("  >> Hint: Check position size against market limits (min/max amount), precision, or step size.")
            if "cost" in err_str or "value" in err_str or "margin" in err_str: lg.error("  >> Hint: Check order cost against market limits (min/max cost) or available margin.")
            if "price" in err_str and order_type != 'market': lg.error("  >> Hint: Check price limits or precision (less likely for market orders).")
            if "reduceonly" in err_str: lg.error("  >> Hint: Check if reduceOnly order contradicts current position or size.")
            return None # Assume fatal for invalid order
        except ccxt.ExchangeError as e:
            # Handle general exchange errors, check for known fatal codes/messages
            error_code = getattr(e, 'code', None) # Bybit specific code might be here
            err_str = str(e).lower()
            lg.error(f"{NEON_RED}Order Placement Failed ({action_desc} {symbol}): Exchange error. {e} (Code: {error_code or 'N/A'}){RESET}")
            # List known fatal error codes/strings for order placement (e.g., account issues, market closed, position mode mismatch)
            fatal_codes = [110014, 110007, 110040, 110013, 110025, 30086, 10001, 30032, 30037] # Example Bybit codes (risk limit, qty error, margin error, position mode)
            fatal_strings = ["position side does not match", "risk limit", "account not unified", "order quantity exceeds limit", "order cost exceeds limit"]
            if error_code in fatal_codes or any(s in err_str for s in fatal_strings):
                lg.error(" >> Hint: This exchange error seems non-retryable for order placement.")
                return None # Non-retryable
            # Otherwise, assume potentially retryable if attempts remain

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"Max retries exceeded for network errors placing order: {e}")
                 return None
            lg.warning(f"{NEON_YELLOW}Network error placing order (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count

        # --- Catch-all for Unexpected Errors ---
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error placing order: {e}{RESET}", exc_info=True)
            return None # Assume fatal for unexpected issues

        # Increment attempt count only if it wasn't a rate limit pause and error was potentially retryable
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """
    Internal helper to set Stop Loss, Take Profit, and/or Trailing Stop Loss for an existing position
    using Bybit V5's specific private API endpoint `/v5/position/set-trading-stop`.

    *** IMPORTANT ***: This function bypasses standard CCXT methods for SL/TP/TSL because
    unified handling across exchanges and account types (especially Bybit V5 Unified/Contract)
    via standard CCXT `edit_order` or `create_order` with SL/TP params can be complex or unreliable.
    Directly using the exchange's specific endpoint provides more control for Bybit V5.

    Args:
        exchange: Initialized CCXT exchange object (must be Bybit).
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        market_info: Dictionary containing market details (ID, precision, type).
        position_info: Dictionary containing details of the current open position (from get_open_position).
        logger: Logger instance for logging messages.
        stop_loss_price: Desired fixed stop loss price (Decimal). Set to 0 or None to remove/not set.
        take_profit_price: Desired fixed take profit price (Decimal). Set to 0 or None to remove/not set.
        trailing_stop_distance: Desired trailing stop distance (Decimal, positive value, in price units). Requires tsl_activation_price. Set to 0 or None to remove/not set TSL.
        tsl_activation_price: Price at which the trailing stop should activate (Decimal). Required if trailing_stop_distance > 0. Set to 0 or None if removing TSL.

    Returns:
        True if the protection was set successfully via the API call or if no changes were needed.
        False if the API call failed, validation failed, or an error occurred.
    """
    lg = logger

    # --- Pre-checks ---
    if 'bybit' not in exchange.id.lower():
        lg.error(f"Protection setting via _set_position_protection is currently only implemented for Bybit. Exchange: {exchange.id}")
        return False
    if not market_info.get('is_contract'):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        # Return True because no action needed, or False if strict? Let's be strict.
        return False
    if not position_info or not isinstance(position_info.get('size_decimal'), Decimal) or position_info['size_decimal'] == 0:
        lg.error(f"Protection setting failed for {symbol}: Missing or invalid current position information (requires valid 'size_decimal').")
        return False

    position_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if position_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"Protection setting failed for {symbol}: Invalid position side ('{position_side}') or missing entry price ('{entry_price_str}') in position info.")
        return False

    try:
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed for {symbol}: Invalid entry price '{entry_price_str}': {e}")
        return False

    # --- Parameter Preparation and Validation ---
    params_to_set: Dict[str, str] = {} # Parameters for the '/v5/position/set-trading-stop' API call
    log_parts: List[str] = [f"Preparing protection request for {symbol} ({position_side.upper()} @ {entry_price.normalize()}):"]
    any_protection_requested: bool = False # Track if any SL/TP/TSL value was explicitly provided

    try:
        price_precision_str = market_info['precision']['price']
        min_price_tick = Decimal(str(price_precision_str))
        if min_price_tick <= 0: raise ValueError("Invalid price precision (tick size must be positive)")

        # Helper to format price values according to market precision using CCXT
        def format_price_param(price_decimal: Optional[Decimal], param_name: str) -> Optional[str]:
            """Formats price to string using exchange precision rules, returns None on error or invalid input."""
            if price_decimal is None: return None
            # Allow explicit clearing of SL/TP/TSL by passing 0
            if price_decimal == Decimal('0'): return "0"
            # Ensure price is positive for setting protection
            if price_decimal < 0:
                lg.warning(f"Attempted to set negative price ({price_decimal}) for {param_name}. Ignoring.")
                return None
            try:
                # Use CCXT's price_to_precision for correct rounding/formatting.
                # Use ROUND_HALF_UP or similar appropriate rounding for protection levels.
                # Let's use exchange default rounding via price_to_precision without explicit mode.
                formatted_price = exchange.price_to_precision(symbol, float(price_decimal))
                # Ensure formatted price is still positive after rounding/formatting
                if Decimal(formatted_price) > 0:
                    return formatted_price
                else:
                    lg.warning(f"Formatted {param_name} price '{formatted_price}' became zero or negative after precision. Ignoring.")
                    return None
            except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
                lg.error(f"Failed to format {param_name} price {price_decimal} using exchange precision: {e}. Ignoring parameter.")
                return None

        # --- Process Trailing Stop ---
        # TSL settings override fixed SL if both are provided (as per Bybit behavior)
        set_tsl = False
        # Check if TSL parameters were provided (not None)
        tsl_distance_provided = isinstance(trailing_stop_distance, Decimal)
        tsl_activation_provided = isinstance(tsl_activation_price, Decimal)

        if tsl_distance_provided or tsl_activation_provided: # If either TSL param is given
            any_protection_requested = True
            # Check if setting TSL (distance > 0)
            if trailing_stop_distance is not None and trailing_stop_distance > 0:
                # Setting TSL requires a positive distance AND a valid activation price
                if not tsl_activation_provided or tsl_activation_price <= 0:
                    lg.error(f"{NEON_RED}Trailing Stop ignored: Invalid or missing activation price ({tsl_activation_price}) required for TSL distance {trailing_stop_distance}.{RESET}")
                else:
                    # Validate activation price is beyond entry price in the profit direction
                    is_valid_activation = (position_side == 'long' and tsl_activation_price > entry_price) or \
                                          (position_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_activation:
                        lg.error(f"{NEON_RED}Trailing Stop ignored: Activation price {tsl_activation_price} is not beyond entry price {entry_price} for {position_side} position.{RESET}")
                    else:
                        # Ensure distance is at least one price tick
                        min_tsl_distance = max(trailing_stop_distance, min_price_tick)
                        fmt_tsl_distance = format_price_param(min_tsl_distance, "Trailing Stop Distance")
                        fmt_tsl_activation = format_price_param(tsl_activation_price, "TSL Activation Price")

                        if fmt_tsl_distance and fmt_tsl_activation:
                            params_to_set['trailingStop'] = fmt_tsl_distance
                            params_to_set['activePrice'] = fmt_tsl_activation
                            log_parts.append(f"  - Set Trailing Stop: Distance={fmt_tsl_distance}, Activation={fmt_tsl_activation}")
                            set_tsl = True # Flag that TSL is being actively set
                        else:
                            lg.error(f"{NEON_RED}Failed to format Trailing Stop parameters (Distance: {fmt_tsl_distance}, Activation: {fmt_tsl_activation}). TSL not set.{RESET}")
            # Check if clearing TSL (distance is explicitly 0)
            elif trailing_stop_distance is not None and trailing_stop_distance == Decimal('0'):
                # Explicitly clear TSL by setting distance to "0"
                params_to_set['trailingStop'] = "0"
                # Bybit docs often suggest setting activePrice to 0 as well when clearing TSL
                params_to_set['activePrice'] = "0"
                log_parts.append("  - Clear Trailing Stop")
                set_tsl = True # Counts as a TSL action (clearing)
            # Handle case where only activation price was provided without distance (invalid state)
            elif tsl_activation_provided and not tsl_distance_provided:
                 lg.warning("TSL Activation price provided without a TSL distance. Ignoring activation price.")


        # --- Process Fixed Stop Loss (Only if TSL is NOT being set or cleared) ---
        sl_provided = isinstance(stop_loss_price, Decimal)
        if not set_tsl and sl_provided:
            any_protection_requested = True
            # Check if setting fixed SL (price > 0)
            if stop_loss_price > 0:
                # Validate SL price is on the correct side (loss side) of entry price
                is_valid_sl = (position_side == 'long' and stop_loss_price < entry_price) or \
                              (position_side == 'short' and stop_loss_price > entry_price)
                if not is_valid_sl:
                    lg.error(f"{NEON_RED}Fixed Stop Loss ignored: SL price {stop_loss_price} is not on the loss side of entry price {entry_price} for {position_side} position.{RESET}")
                else:
                    fmt_sl = format_price_param(stop_loss_price, "Stop Loss")
                    if fmt_sl:
                        params_to_set['stopLoss'] = fmt_sl
                        log_parts.append(f"  - Set Fixed Stop Loss: {fmt_sl}")
                    else:
                        lg.error(f"{NEON_RED}Failed to format Fixed Stop Loss price {stop_loss_price}. SL not set.{RESET}")
            # Check if clearing fixed SL (price is 0)
            elif stop_loss_price == Decimal('0'):
                params_to_set['stopLoss'] = "0"
                log_parts.append("  - Clear Fixed Stop Loss")

        # --- Process Fixed Take Profit ---
        tp_provided = isinstance(take_profit_price, Decimal)
        if tp_provided:
            any_protection_requested = True
            # Check if setting TP (price > 0)
            if take_profit_price > 0:
                # Validate TP price is on the correct side (profit side) of entry price
                is_valid_tp = (position_side == 'long' and take_profit_price > entry_price) or \
                              (position_side == 'short' and take_profit_price < entry_price)
                if not is_valid_tp:
                    lg.error(f"{NEON_RED}Take Profit ignored: TP price {take_profit_price} is not on the profit side of entry price {entry_price} for {position_side} position.{RESET}")
                else:
                    fmt_tp = format_price_param(take_profit_price, "Take Profit")
                    if fmt_tp:
                        params_to_set['takeProfit'] = fmt_tp
                        log_parts.append(f"  - Set Take Profit: {fmt_tp}")
                    else:
                        lg.error(f"{NEON_RED}Failed to format Take Profit price {take_profit_price}. TP not set.{RESET}")
            # Check if clearing TP (price is 0)
            elif take_profit_price == Decimal('0'):
                params_to_set['takeProfit'] = "0"
                log_parts.append("  - Clear Take Profit")

    except ValueError as val_err: # Catch validation errors like invalid precision
         lg.error(f"{NEON_RED}Protection parameter validation failed: {val_err}. Cannot proceed.{RESET}")
         return False
    except Exception as format_err:
        lg.error(f"{NEON_RED}Unexpected error during protection parameter formatting: {format_err}{RESET}", exc_info=True)
        return False

    # --- Check if any parameters were actually prepared for the API call ---
    if not params_to_set:
        if any_protection_requested:
            lg.warning(f"{NEON_YELLOW}No valid protection parameters generated for {symbol} after validation/formatting, despite request. No API call made.{RESET}")
            # Return False because the intent to set protection failed due to invalid inputs/formatting
            return False
        else:
            lg.debug(f"No protection changes requested or parameters provided for {symbol}. No API call needed.")
            return True # No changes needed, so considered successful

    # --- Prepare and Execute Bybit V5 API Call ---
    # Determine Bybit category and market ID
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    market_id = market_info['id']
    # Get position index (usually 0 for one-way mode, might be 1 or 2 for hedge mode)
    position_idx = 0 # Default to one-way mode
    try:
        # Try to get positionIdx from the raw position info if available (more reliable for hedge mode)
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
    except (ValueError, TypeError):
        lg.warning(f"Could not parse 'positionIdx' from position_info. Using default {position_idx} (assuming one-way mode).")

    # Construct the final parameters dictionary for the Bybit API endpoint
    # Note: tpslMode='Full' applies SL/TP to the entire position size.
    #       'Partial' allows setting SL/TP for a portion (requires size param, more complex).
    # Trigger prices (slTriggerBy, tpTriggerBy) default to 'LastPrice', can be 'MarkPrice' or 'IndexPrice'.
    # Order types (slOrderType, tpOrderType) default to 'Market', can be 'Limit' (requires price param).
    final_api_params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full',          # Apply to whole position
        # Optional: Specify trigger price types if needed, default is LastPrice
        # 'slTriggerBy': 'LastPrice',
        # 'tpTriggerBy': 'LastPrice',
        # Optional: Specify order types if needed, default is Market
        # 'slOrderType': 'Market',
        # 'tpOrderType': 'Market',
        'positionIdx': position_idx  # Crucial for hedge mode, 0 for one-way
    }
    final_api_params.update(params_to_set) # Add the specific SL/TP/TSL values prepared earlier

    # Log the intended changes and the API call being made
    lg.info("\n".join(log_parts)) # Log the multi-line summary of actions
    lg.info(f"  Executing Bybit API call: private POST /v5/position/set-trading-stop")
    lg.debug(f"  API Parameters: {final_api_params}")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing protection API call (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            # Use the appropriate CCXT method for making authenticated POST requests to private endpoints
            # This relies on CCXT's internal handling of authentication and request signing.
            # Method name might vary slightly based on CCXT version, but `private_post` is common.
            response = exchange.private_post('/v5/position/set-trading-stop', params=final_api_params)
            lg.debug(f"Set protection raw API response: {response}")

            # --- Check Bybit V5 response code ---
            response_code = response.get('retCode')
            response_msg = response.get('retMsg', 'Unknown response message')

            if response_code == 0:
                 # Check if the message indicates no actual change was needed (already set)
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "same tpsl"]
                 if any(msg_part in response_msg.lower() for msg_part in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Protection already set as requested or no change needed for {symbol}. (API Msg: {response_msg}){RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection successfully set/updated for {symbol}.{RESET}")
                 return True # Success
            else:
                # Log the specific error message from Bybit API
                lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {response_msg} (Code: {response_code}){RESET}")
                # Identify potentially non-retryable error codes based on Bybit documentation
                # Examples: Parameter errors, position not found, invalid mode, risk limits
                fatal_codes = [10001, 10002, 110013, 110025, 110036, 110043, 110084, 110085, 110086, 3400041]
                fatal_strings = ["invalid parameter", "position not found", "tpsl mode error", "risk limit"]
                is_fatal = response_code in fatal_codes or any(fs in response_msg.lower() for fs in fatal_strings)

                if is_fatal:
                     lg.error(" >> Hint: This protection setting error seems non-retryable. Check parameters, position status, or account mode.")
                     return False # Non-retryable error, return False immediately
                else:
                     # If error might be transient, raise exception to trigger retry
                     raise ccxt.ExchangeError(f"Bybit API error setting protection: {response_msg} (Code: {response_code})")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                lg.error(f"Max retries exceeded for network errors setting protection: {e}")
                return False
            lg.warning(f"{NEON_YELLOW}Network error setting protection (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded setting protection: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.AuthenticationError as e:
            # Critical authentication error
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Check API permissions for position modification endpoints. Stopping protection attempt.{RESET}")
            return False # Fatal
        except ccxt.ExchangeError as e:
            # Catch re-raised errors or other general exchange errors during the API call
             if attempts >= MAX_API_RETRIES:
                 lg.error(f"Max retries exceeded for ExchangeError setting protection: {e}")
                 return False
             lg.warning(f"{NEON_YELLOW}Exchange error setting protection (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}. Retrying...{RESET}")
        except Exception as e:
            # Catch unexpected errors during the API call itself
            lg.error(f"{NEON_RED}Unexpected error during protection API call (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}{RESET}", exc_info=True)
            # Unexpected errors are likely fatal for this specific operation
            return False

        # Increment attempt count and wait before retry
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False

def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, config: Dict[str, Any],
                             logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool:
    """
    Calculates Trailing Stop Loss parameters (distance, activation price) based on configuration
    (callback rate, activation percentage) and the current position's entry price.
    Then calls the internal `_set_position_protection` helper to make the API request.

    Args:
        exchange: Initialized CCXT exchange object (must be Bybit for this implementation).
        symbol: Market symbol.
        market_info: Dictionary of market details (precision is crucial).
        position_info: Dictionary of the current open position (entry price is crucial).
        config: The main configuration dictionary (specifically the "protection" section).
        logger: Logger instance for logging messages.
        take_profit_price: Optional fixed take profit price (Decimal) to set simultaneously with the TSL.
                           Set to 0 to clear any existing TP, None to leave TP unchanged (if API supports).
                           Note: Bybit's set-trading-stop endpoint might overwrite TP when setting TSL.
                           Passing the desired TP value here ensures it's included in the API call.

    Returns:
        True if the TSL (and optional TP) was calculated and the API call to set it was successful (or no change needed),
        False otherwise (calculation error, validation error, API error).
    """
    lg = logger
    protection_config = config.get("protection", {}) # Get protection settings safely

    # Validate essential inputs
    if not market_info or not position_info:
        lg.error(f"TSL calculation failed for {symbol}: Missing market or position info.")
        return False
    position_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if position_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"TSL calculation failed for {symbol}: Invalid position side ('{position_side}') or missing entry price ('{entry_price_str}').")
        return False

    try:
        # Parse necessary values from config and market/position info using Decimal
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(protection_config["trailing_stop_callback_rate"]))
        activation_percentage = Decimal(str(protection_config["trailing_stop_activation_percentage"]))
        price_tick_size = Decimal(str(market_info['precision']['price']))

        # Validate parsed values
        if entry_price <= 0: raise ValueError("Entry price must be positive")
        if callback_rate <= 0: raise ValueError("Trailing stop callback rate must be positive")
        if activation_percentage < 0: raise ValueError("Trailing stop activation percentage cannot be negative")
        if price_tick_size <= 0: raise ValueError("Price tick size from market info must be positive")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"{NEON_RED}TSL calculation failed for {symbol}: Invalid parameters in config or market/position info: {e}.{RESET}")
        return False

    lg.info(f"Calculating Trailing Stop for {symbol} ({position_side.upper()}):")
    lg.debug(f"  Inputs: Entry={entry_price.normalize()}, Activation%={activation_percentage:.3%}, CallbackRate={callback_rate:.3%}, TickSize={price_tick_size}")

    try:
        # --- Calculate Activation Price ---
        # Activation offset from entry price
        activation_offset = entry_price * activation_percentage
        raw_activation_price = (entry_price + activation_offset) if position_side == 'long' else (entry_price - activation_offset)

        # Round activation price *away* from entry price to the nearest tick to ensure activation occurs after entry
        # ROUND_UP for long, ROUND_DOWN for short
        rounding_mode = ROUND_UP if position_side == 'long' else ROUND_DOWN
        activation_price = raw_activation_price.quantize(price_tick_size, rounding=rounding_mode)

        # Ensure activation price is strictly beyond entry price by at least one tick after rounding
        if position_side == 'long':
            activation_price = max(activation_price, entry_price + price_tick_size)
        else: # Short
            activation_price = min(activation_price, entry_price - price_tick_size)

        # Check if activation price calculation resulted in a valid (positive) price
        if activation_price <= 0:
            lg.error(f"{NEON_RED}TSL calculation failed: Calculated activation price ({activation_price}) is zero or negative.{RESET}")
            return False
        lg.debug(f"  Calculated Raw Activation Price: {raw_activation_price.normalize()}")
        lg.debug(f"  Calculated Rounded Activation Price: {activation_price.normalize()}")


        # --- Calculate Trailing Distance ---
        # Bybit's `trailingStop` parameter is the distance/offset value in price units.
        # It's often calculated based on the activation price and the callback rate.
        distance_raw = activation_price * callback_rate
        # Round the distance UP to the nearest tick size to ensure it's a valid step.
        # Also ensure the distance is at least one full tick size.
        trailing_distance = max(distance_raw.quantize(price_tick_size, rounding=ROUND_UP), price_tick_size)

        if trailing_distance <= 0:
            lg.error(f"{NEON_RED}TSL calculation failed: Calculated trailing distance ({trailing_distance}) is zero or negative.{RESET}")
            return False
        lg.debug(f"  Calculated Raw Trail Distance: {distance_raw.normalize()}")
        lg.debug(f"  Calculated Rounded Trail Distance: {trailing_distance.normalize()}")


        # --- Log Summary and Call Helper ---
        lg.info(f"  => Activation Price: {activation_price.normalize()}")
        lg.info(f"  => Trail Distance: {trailing_distance.normalize()}")
        # Log the TP that will be sent along with the TSL request
        tp_log = 'None (Leave Unchanged)'
        if isinstance(take_profit_price, Decimal):
             tp_log = take_profit_price.normalize() if take_profit_price != 0 else 'Clear'
        lg.info(f"  Take Profit to Set: {tp_log}")

        # Call the internal helper function to make the API request
        # Pass None for stop_loss_price as TSL parameters will take precedence in the Bybit API call logic
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL in the _set_position_protection logic
            take_profit_price=take_profit_price, # Pass the desired TP along
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during TSL calculation or formatting: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy Implementation ---
class OrderBlock(TypedDict):
    """Represents a detected Order Block with its properties."""
    id: str             # Unique identifier (e.g., "B_230101120000" for Bearish, "L_..." for Bullish)
    type: str           # 'bull' or 'bear'
    left_idx: pd.Timestamp # Timestamp of the pivot candle that formed the OB
    right_idx: pd.Timestamp # Timestamp of the last candle the OB is valid for (or the violation candle)
    top: Decimal        # Top price boundary of the OB
    bottom: Decimal     # Bottom price boundary of the OB
    active: bool        # Is the OB currently considered valid (not violated by close price)?
    violated: bool      # Has the price closed beyond this OB (making it inactive)?

class StrategyAnalysisResults(TypedDict):
    """Structure for returning the results from the strategy analysis engine."""
    dataframe: pd.DataFrame          # The DataFrame with all calculated indicators and OHLCV data.
    last_close: Decimal              # Last closing price from the DataFrame.
    current_trend_up: Optional[bool] # True if the Volumatic trend is currently up, False if down, None if undetermined.
    trend_just_changed: bool         # True if the trend direction changed on the very last candle.
    active_bull_boxes: List[OrderBlock] # List of currently active Bullish Order Blocks.
    active_bear_boxes: List[OrderBlock] # List of currently active Bearish Order Blocks.
    vol_norm_int: Optional[int]      # Normalized volume (0-100+) as an integer, None if unavailable.
    atr: Optional[Decimal]           # Last calculated ATR value, None if unavailable.
    upper_band: Optional[Decimal]    # Last Volumatic upper band value, None if unavailable.
    lower_band: Optional[Decimal]    # Last Volumatic lower band value, None if unavailable.

class VolumaticOBStrategy:
    """
    Implements the Volumatic Trend indicator combined with Pivot-based Order Block detection.
    Calculates indicators, identifies pivots, creates, updates, and manages the state
    of active bullish and bearish order blocks based on price action.
    """
    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the configuration.

        Args:
            config: The main configuration dictionary (expects "strategy_params" key).
            market_info: Dictionary of market details for the symbol being analyzed.
            logger: Logger instance for strategy-specific logging.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger # Alias for convenience in methods

        strategy_cfg = config["strategy_params"]
        # Load Volumatic Trend parameters
        self.vt_length: int = int(strategy_cfg["vt_length"])
        self.vt_atr_period: int = int(strategy_cfg["vt_atr_period"])
        self.vt_vol_ema_length: int = int(strategy_cfg["vt_vol_ema_length"])
        self.vt_atr_multiplier: Decimal = Decimal(str(strategy_cfg["vt_atr_multiplier"]))

        # Load Order Block parameters
        self.ob_source: str = strategy_cfg["ob_source"] # "Wicks" or "Body"
        self.ph_left: int = int(strategy_cfg["ph_left"])
        self.ph_right: int = int(strategy_cfg["ph_right"])
        self.pl_left: int = int(strategy_cfg["pl_left"])
        self.pl_right: int = int(strategy_cfg["pl_right"])
        self.ob_extend: bool = bool(strategy_cfg["ob_extend"])
        self.ob_max_boxes: int = int(strategy_cfg["ob_max_boxes"])

        # Internal state for tracking Order Blocks across updates
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum data length required based on the largest lookback period
        # Need enough data for EMAs, ATR, Volume EMA, and pivot detection windows
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length) # Rough estimate for EMA/ATR convergence
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1) # Data window needed for pivot calc
        self.min_data_len: int = max(required_for_vt, required_for_pivots) + 50 # Add a buffer for stability

        # Sanity check: Ensure Vol EMA length isn't excessively large compared to internal DF limit
        if self.vt_vol_ema_length > MAX_DF_LEN - 50:
            original_len = self.vt_vol_ema_length
            self.vt_vol_ema_length = MAX_DF_LEN - 50
            self.lg.warning(f"{NEON_YELLOW}Volume EMA length ({original_len}) exceeds internal limit ({MAX_DF_LEN}). Capped to {self.vt_vol_ema_length}.{RESET}")

        self.lg.info(f"{NEON_CYAN}--- VolumaticOB Strategy Engine Initialized ---{RESET}")
        self.lg.info(f"  Volumatic Trend Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Volume EMA={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  Order Block Params: Source={self.ob_source}, PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")
        self.lg.info(f"  Calculated Minimum Data Required: {self.min_data_len} candles")

        # Warn if the required data length significantly exceeds the typical API limit
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 50: # Check if significantly more than one request needed
            self.lg.error(f"{NEON_RED}{BRIGHT}CONFIGURATION WARNING:{RESET}{NEON_YELLOW} Strategy requires {self.min_data_len} candles, which likely exceeds the API limit per request ({BYBIT_API_KLINE_LIMIT}).")
            self.lg.error(f"{NEON_YELLOW}  This may lead to insufficient data errors or inaccurate calculations.")
            self.lg.error(f"{NEON_YELLOW}  RECOMMENDATION: Reduce lookback periods (vt_length, vt_atr_period, vt_vol_ema_length, ph_left/right, pl_left/right) in config.json.{RESET}")
        elif self.vt_vol_ema_length > 1000: # Warn if volume EMA length is unusually large
             self.lg.warning(f"{NEON_YELLOW}Volume EMA length ({self.vt_vol_ema_length}) is large. Ensure sufficient historical data is available and fetched.{RESET}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates the Exponential Moving Average (EMA) of the Symmetrically
        Weighted Moving Average (SWMA) of a given series.
        The SWMA(close, 4) is defined as: (close[3]*1 + close[2]*2 + close[1]*2 + close[0]*1) / 6.

        Args:
            series: The input pandas Series (e.g., close prices, should be numeric).
            length: The length (period) for the final EMA calculation.

        Returns:
            A pandas Series containing the calculated EMA(SWMA), with NaNs where calculation wasn't possible.
        """
        # Input validation
        if not isinstance(series, pd.Series) or series.empty:
            return pd.Series(dtype=np.float64) # Return empty series matching input type
        if length <= 0:
             self.lg.warning("EMA SWMA length must be positive.")
             return pd.Series(np.nan, index=series.index) # Return NaNs for invalid length

        # Ensure series is numeric, coercing errors to NaN
        series_numeric = pd.to_numeric(series, errors='coerce')

        # Need at least 4 periods for SWMA calculation
        if len(series_numeric) < 4:
            return pd.Series(np.nan, index=series.index) # Not enough data for SWMA

        # Define SWMA weights for window size 4
        weights = np.array([1., 2., 2., 1.]) / 6.0

        # Calculate SWMA using rolling apply with the defined weights
        # Apply requires numeric input, NaNs will propagate
        # Using raw=True can offer performance benefits if the underlying data is numpy array
        swma = series_numeric.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights) if pd.notna(x).all() else np.nan, raw=True)

        # Calculate EMA of the SWMA result using pandas_ta
        # fillna=np.nan ensures that EMA doesn't fill forward over gaps in SWMA
        ema_swma_result = ta.ema(swma, length=length, fillna=np.nan)

        return ema_swma_result

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, find_highs: bool) -> pd.Series:
        """
        Finds pivot points (highs or lows) in a series based on a simple lookback/lookforward comparison.
        A pivot high requires the candle's value to be strictly greater than all candles
        within the left_bars and right_bars window.
        A pivot low requires the candle's value to be strictly lower.

        Args:
            series: The pandas Series to find pivots in (e.g., high prices for PH, low prices for PL). Should be numeric.
            left_bars: Number of bars to look back to the left (must be >= 1).
            right_bars: Number of bars to look forward to the right (must be >= 1).
            find_highs: If True, finds pivot highs. If False, finds pivot lows.

        Returns:
            A pandas Series of booleans, True where a pivot is detected according to the criteria.
        """
        # Input validation
        if not isinstance(series, pd.Series) or series.empty:
            self.lg.debug("_find_pivots received empty series.")
            return pd.Series(False, index=series.index)
        if left_bars < 1 or right_bars < 1:
            self.lg.warning(f"_find_pivots requires left_bars ({left_bars}) and right_bars ({right_bars}) >= 1. Returning no pivots.")
            return pd.Series(False, index=series.index)

        # Ensure the series contains numeric data, coercing non-numeric to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all():
            self.lg.debug("_find_pivots series contains only NaNs after coercion.")
            return pd.Series(False, index=series.index)

        # --- Efficient Pivot Calculation using rolling window ---
        # Total window size needed for comparison
        window_size = left_bars + right_bars + 1

        # Apply a rolling window and check if the center element (at index `left_bars`) is the extremum
        if find_highs:
            # Find the index of the max value within the rolling window
            # shift(-right_bars) aligns the window result to the potential pivot candle
            pivot_check = numeric_series.rolling(window=window_size, center=False).apply(lambda x: np.argmax(x) == left_bars if pd.notna(x).all() else False, raw=True).shift(-right_bars)
        else: # Find lows
            # Find the index of the min value within the rolling window
            pivot_check = numeric_series.rolling(window=window_size, center=False).apply(lambda x: np.argmin(x) == left_bars if pd.notna(x).all() else False, raw=True).shift(-right_bars)

        # Convert the result (0.0 or 1.0 from apply, or NaN) to boolean, filling NaNs as False
        is_pivot = pivot_check.fillna(0.0).astype(bool)

        return is_pivot


    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes the historical kline DataFrame to calculate all strategy indicators
        (Volumatic Trend, Bands, ATR, Volume Normalization), detect pivots, create/update
        Order Blocks based on pivots and price action, and return the analysis results.

        Args:
            df_input: The input pandas DataFrame with OHLCV data. Must have a DatetimeIndex
                      and columns 'open', 'high', 'low', 'close', 'volume' containing Decimal objects.

        Returns:
            A StrategyAnalysisResults dictionary containing the processed DataFrame,
            the latest state (trend, close, ATR, etc.), and the lists of currently active order blocks.
            Returns a default empty results structure on critical failure.
        """
        # Define default empty results structure for failure cases
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.lg.error("Strategy update received an empty DataFrame. Cannot proceed.")
            return empty_results

        # Work on a copy to avoid modifying the original DataFrame passed to the function
        df = df_input.copy()

        # --- Pre-computation Checks ---
        if not isinstance(df.index, pd.DatetimeIndex):
            self.lg.error("Strategy update DataFrame index is not a DatetimeIndex.")
            return empty_results
        if not df.index.is_monotonic_increasing:
             self.lg.warning("Strategy update DataFrame index is not monotonically increasing. Sorting index...")
             df.sort_index(inplace=True)
             # Check again after sorting
             if not df.index.is_monotonic_increasing:
                  self.lg.error("DataFrame index still not monotonic after sorting. Aborting analysis.")
                  return empty_results
        if len(df) < self.min_data_len:
            self.lg.warning(f"Insufficient data for strategy calculation ({len(df)} candles < required {self.min_data_len}). Results may be inaccurate or contain NaNs.")
            # Proceed, but be aware results might have leading NaNs or be less reliable

        self.lg.debug(f"Starting strategy analysis on {len(df)} candles (minimum required: {self.min_data_len}).")

        # --- Convert to Float for TA-Lib/Pandas-TA Performance ---
        # Most TA libraries work more efficiently with floats. Store original Decimals if needed.
        try:
            self.lg.debug("Converting OHLCV columns to float for TA calculations...")
            df_float = pd.DataFrame(index=df.index)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.lg.error(f"Strategy update failed: Missing required column '{col}' in input DataFrame.")
                    return empty_results
                # Convert Decimal/object column to numeric (float), coercing errors
                df_float[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where essential float conversions failed (e.g., if input was non-numeric string)
            initial_float_len = len(df_float)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if len(df_float) < initial_float_len:
                 self.lg.warning(f"Dropped {initial_float_len - len(df_float)} rows due to NaN in essential OHLC columns after float conversion.")
            if df_float.empty:
                self.lg.error("DataFrame became empty after converting essential OHLC columns to float.")
                return empty_results
            self.lg.debug("Successfully converted relevant columns to float for TA calculations.")
        except Exception as e:
            self.lg.error(f"{NEON_RED}Error converting DataFrame columns to float for TA: {e}{RESET}", exc_info=True)
            return empty_results

        # --- Indicator Calculations (using the float DataFrame) ---
        try:
            self.lg.debug("Calculating indicators: ATR, EMAs, Trend, Bands, Volume Normalization...")
            # ATR (Average True Range)
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # Volumatic Trend EMAs
            # EMA1: EMA of SWMA(close, 4)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)
            # EMA2: Standard EMA of close
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)

            # Determine Trend Direction and Changes
            # Trend is up if EMA2 > EMA1 of the *previous* bar (to avoid lookahead bias)
            # Shift EMA1 by 1 bar for comparison. ffill handles initial NaNs, fillna(False) ensures boolean.
            trend_up_series = (df_float['ema2'] > df_float['ema1'].shift(1)).ffill().fillna(False)
            df_float['trend_up'] = trend_up_series

            # Trend change occurs if trend_up differs from the previous bar's trend_up
            # Ensure both current and previous trend values are not NaN before comparing
            trend_changed_series = (df_float['trend_up'].shift(1) != df_float['trend_up']) & \
                                   df_float['trend_up'].notna() & \
                                   df_float['trend_up'].shift(1).notna()
            # Fill NaNs at the start of the series with False (no change possible on first valid bar)
            df_float['trend_changed'] = trend_changed_series.fillna(False)

            # Calculate Volumatic Bands based on values at trend change points
            # Capture EMA1 and ATR values only on bars where the trend *just* changed
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these captured values to create stable levels for the bands until the next trend change
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Upper and Lower Bands using the forward-filled values
            atr_multiplier_float = float(self.vt_atr_multiplier) # Convert Decimal multiplier for float calculation
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float)

            # Volume Normalization Calculation
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            # Ensure min_periods is reasonable for rolling max calculation, avoid issues with short data
            min_periods_vol = max(1, min(self.vt_vol_ema_length, len(volume_numeric) // 10)) # Heuristic
            # Rolling max volume over the specified period
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=min_periods_vol).max().fillna(0.0)
            # Calculate normalized volume (current volume / max volume in period * 100)
            # Avoid division by zero if max volume is zero or near zero
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9, # Use small threshold instead of == 0
                                           (volume_numeric / df_float['vol_max'] * 100.0),
                                           0.0) # Set to 0 if max volume is effectively zero
            # Fill any remaining NaNs (e.g., at the start) and clip result (e.g., 0-200 range common)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0) # Clip max to 200% arbitrarily

            # --- Pivot Detection ---
            self.lg.debug("Detecting Pivot Highs and Lows using specified source...")
            # Select the source series for pivot detection based on configuration
            if self.ob_source.lower() == "wicks":
                high_series = df_float['high']
                low_series = df_float['low']
                self.lg.debug("Using candle Highs/Lows for pivot detection.")
            else: # Assume "Body"
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)
                self.lg.debug("Using candle Body Max/Min for pivot detection.")

            # Find Pivot Highs (PH) and Pivot Lows (PL)
            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, find_highs=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, find_highs=False)

            self.lg.debug("Indicator calculations and pivot detection complete.")

        except Exception as e:
            self.lg.error(f"{NEON_RED}Error during indicator calculation or pivot detection: {e}{RESET}", exc_info=True)
            return empty_results # Return empty results if indicators fail

        # --- Copy Calculated Indicator Results Back to Original Decimal DataFrame ---
        # This ensures the returned DataFrame primarily uses Decimal, maintaining precision.
        try:
            self.lg.debug("Copying calculated indicators back to main Decimal DataFrame...")
            indicator_cols = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                              'upper_band', 'lower_band', 'vol_norm', 'is_ph', 'is_pl']
            for col in indicator_cols:
                if col in df_float.columns:
                    # Reindex float series to match original Decimal DF index (in case rows were dropped)
                    source_series = df_float[col].reindex(df.index)
                    # Assign based on type
                    if source_series.dtype == 'bool':
                        df[col] = source_series.astype(bool)
                    elif pd.api.types.is_object_dtype(source_series): # Keep objects as is (unlikely here)
                         df[col] = source_series
                    else: # Convert numeric types (float) back to Decimal
                        df[col] = source_series.apply(
                            # Only convert finite numbers, keep NaNs as Decimal('NaN')
                            lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                        )
        except Exception as e:
            self.lg.error(f"{NEON_RED}Error converting calculated indicators back to Decimal format: {e}. Proceeding with potentially mixed types.{RESET}", exc_info=True)
            # If conversion fails, the 'df' might have float columns mixed with Decimal.

        # --- Clean Final DataFrame (Remove rows with NaN in essential indicators) ---
        # Ensure core indicators needed for signals/OBs are present
        initial_len = len(df)
        # Define columns absolutely required for subsequent logic
        required_indicator_cols = ['open', 'high', 'low', 'close', 'atr', 'trend_up', 'upper_band', 'lower_band', 'is_ph', 'is_pl']
        # Check which required columns actually exist in df before trying to dropna
        cols_to_check = [col for col in required_indicator_cols if col in df.columns]
        df.dropna(subset=cols_to_check, inplace=True) # Drop rows missing any of these indicators

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            self.lg.debug(f"Dropped {rows_dropped} rows from final DataFrame due to missing essential indicator values (likely at start of series).")
        if df.empty:
            self.lg.warning("DataFrame became empty after final indicator cleaning. Cannot process Order Blocks or generate results.")
            return empty_results # Cannot proceed if DataFrame is empty

        self.lg.debug("Final DataFrame cleaned. Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_ob_count = 0
            processed_pivot_indices = set() # Track pivots already used to form an OB in this run

            # Iterate backwards through the DataFrame to find recent pivots first
            # Limit iteration depth for performance? Maybe last N candles where N is related to max boxes?
            # Iterate through all candles for now, but check if pivot is recent enough?
            # For simplicity, check all candles where a pivot is marked.
            pivot_indices = df.index[df['is_ph'] | df['is_pl']]

            for pivot_idx in pivot_indices:
                 # Skip if we already processed this exact timestamp in this update cycle
                 if pivot_idx in processed_pivot_indices:
                     continue

                 try:
                     pivot_candle = df.loc[pivot_idx]
                     is_pivot_high = pivot_candle['is_ph']
                     is_pivot_low = pivot_candle['is_pl']

                     # --- Create Bearish OB from Pivot High ---
                     if is_pivot_high:
                         # Check if an OB from this exact pivot candle already exists in our state
                         # Prevents creating duplicates if update is run multiple times on same data
                         if not any(b['left_idx'] == pivot_idx and b['type'] == 'bear' for b in self.bear_boxes):
                             # Determine OB price boundaries based on source config
                             if self.ob_source.lower() == "wicks":
                                 # Bearish OB: Top = High of pivot, Bottom = Open of pivot (common definition)
                                 top = pivot_candle['high']
                                 bottom = pivot_candle['open']
                             else: # Body
                                 # Bearish OB: Top = Max(Open, Close), Bottom = Min(Open, Close)
                                 top = max(pivot_candle['open'], pivot_candle['close'])
                                 bottom = min(pivot_candle['open'], pivot_candle['close'])

                             # Ensure valid Decimal boundaries (Top > Bottom)
                             if pd.notna(top) and pd.notna(bottom) and isinstance(top, Decimal) and isinstance(bottom, Decimal) and top > bottom:
                                 box_id = f"B_{pivot_idx.strftime('%y%m%d%H%M%S')}" # Unique ID: Bearish + Timestamp
                                 new_box = OrderBlock(
                                     id=box_id,
                                     type='bear',
                                     left_idx=pivot_idx, # Timestamp of the pivot candle
                                     right_idx=df.index[-1], # Initially extends to latest candle
                                     top=top,
                                     bottom=bottom,
                                     active=True, # New OBs start active
                                     violated=False
                                 )
                                 self.bear_boxes.append(new_box)
                                 new_ob_count += 1
                                 processed_pivot_indices.add(pivot_idx) # Mark as processed
                                 self.lg.debug(f"  + New Bearish OB created: {box_id} @ {pivot_idx.strftime('%Y-%m-%d %H:%M')} [{bottom.normalize()}-{top.normalize()}]")
                             else:
                                 self.lg.warning(f"Could not create Bearish OB at {pivot_idx}: Invalid boundaries (Top: {top}, Bottom: {bottom}). Check source data.")

                     # --- Create Bullish OB from Pivot Low ---
                     elif is_pivot_low: # Use elif because a candle is typically PH or PL, not both
                          # Check if an OB from this exact pivot candle already exists
                          if not any(b['left_idx'] == pivot_idx and b['type'] == 'bull' for b in self.bull_boxes):
                             if self.ob_source.lower() == "wicks":
                                 # Bullish OB: Top = Open of pivot, Bottom = Low of pivot (common definition)
                                 top = pivot_candle['open']
                                 bottom = pivot_candle['low']
                             else: # Body
                                 # Bullish OB: Top = Max(Open, Close), Bottom = Min(Open, Close)
                                 top = max(pivot_candle['open'], pivot_candle['close'])
                                 bottom = min(pivot_candle['open'], pivot_candle['close'])

                             # Ensure valid Decimal boundaries (Top > Bottom)
                             if pd.notna(top) and pd.notna(bottom) and isinstance(top, Decimal) and isinstance(bottom, Decimal) and top > bottom:
                                 box_id = f"L_{pivot_idx.strftime('%y%m%d%H%M%S')}" # Unique ID: Bullish + Timestamp
                                 new_box = OrderBlock(
                                     id=box_id,
                                     type='bull',
                                     left_idx=pivot_idx,
                                     right_idx=df.index[-1], # Initially extends to latest
                                     top=top,
                                     bottom=bottom,
                                     active=True,
                                     violated=False
                                 )
                                 self.bull_boxes.append(new_box)
                                 new_ob_count += 1
                                 processed_pivot_indices.add(pivot_idx) # Mark as processed
                                 self.lg.debug(f"  + New Bullish OB created: {box_id} @ {pivot_idx.strftime('%Y-%m-%d %H:%M')} [{bottom.normalize()}-{top.normalize()}]")
                             else:
                                 self.lg.warning(f"Could not create Bullish OB at {pivot_idx}: Invalid boundaries (Top: {top}, Bottom: {bottom}). Check source data.")
                 except KeyError as ke:
                     self.lg.warning(f"Missing expected column ({ke}) when processing pivot at index {pivot_idx}. Skipping OB creation.")
                 except Exception as e:
                     # Log error for specific pivot but continue processing others
                     self.lg.warning(f"Error processing pivot/creating OB at index {pivot_idx}: {e}", exc_info=True)

            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new potential Order Blocks in this update.")

            # --- Update Existing Order Blocks (Violation Check & Extension) ---
            last_candle = df.iloc[-1] if not df.empty else None
            last_close_price: Optional[Decimal] = None
            last_idx: Optional[pd.Timestamp] = None

            if last_candle is not None and pd.notna(last_candle.get('close')) and isinstance(last_candle['close'], Decimal):
                last_close_price = last_candle['close']
                last_idx = last_candle.name # Timestamp of the last candle
                violation_check_possible = True
            else:
                self.lg.warning("Cannot check Order Block violations: Invalid or missing last close price in DataFrame.")
                violation_check_possible = False

            if violation_check_possible and last_close_price is not None and last_idx is not None:
                # Check Bullish OBs for violation by the last close price
                for box in self.bull_boxes:
                    if box['active']: # Only check active boxes
                        # Violation Condition: Close price moves below the bottom of the Bullish OB
                        if last_close_price < box['bottom']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx # Mark the timestamp of violation
                            self.lg.debug(f"  - Bullish OB {box['id']} VIOLATED by close {last_close_price.normalize()} < bottom {box['bottom'].normalize()}")
                        # Extend active OB to the current candle's timestamp if configured
                        elif self.ob_extend:
                            box['right_idx'] = last_idx

                # Check Bearish OBs for violation by the last close price
                for box in self.bear_boxes:
                    if box['active']:
                        # Violation Condition: Close price moves above the top of the Bearish OB
                        if last_close_price > box['top']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx
                            self.lg.debug(f"  - Bearish OB {box['id']} VIOLATED by close {last_close_price.normalize()} > top {box['top'].normalize()}")
                        elif self.ob_extend:
                            box['right_idx'] = last_idx

            # --- Prune Order Blocks List ---
            # 1. Keep only active (non-violated) OBs
            active_bull_obs = [b for b in self.bull_boxes if b['active']]
            active_bear_obs = [b for b in self.bear_boxes if b['active']]

            # 2. Sort active OBs by creation time (left_idx) descending (most recent first)
            active_bull_obs.sort(key=lambda b: b['left_idx'], reverse=True)
            active_bear_obs.sort(key=lambda b: b['left_idx'], reverse=True)

            # 3. Keep only the N most recent active OBs as defined by ob_max_boxes
            self.bull_boxes = active_bull_obs[:self.ob_max_boxes]
            self.bear_boxes = active_bear_obs[:self.ob_max_boxes]

            self.lg.debug(f"Pruned Order Blocks list. Kept Active: Bullish={len(self.bull_boxes)}, Bearish={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"{NEON_RED}Error during Order Block processing (creation/update/pruning): {e}{RESET}", exc_info=True)
            # Continue to return results, but OBs might be inaccurate or empty

        # --- Prepare Final Results Dictionary ---
        # Get the very last row from the processed DataFrame
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper to safely get Decimal values from the last candle row
        def safe_decimal_from_candle(candle_row: Optional[pd.Series], col_name: str, positive_only: bool = False) -> Optional[Decimal]:
            """Safely extracts a Decimal value from a Series, returns None if invalid."""
            if candle_row is None or col_name not in candle_row: return None
            value = candle_row[col_name]
            if pd.notna(value) and isinstance(value, Decimal) and value.is_finite():
                 if not positive_only or value > Decimal('0'):
                     return value
            return None

        # Helper to safely get Boolean values
        def safe_bool_from_candle(candle_row: Optional[pd.Series], col_name: str) -> Optional[bool]:
             if candle_row is None or col_name not in candle_row: return None
             value = candle_row[col_name]
             # Check for actual boolean True/False, handle numpy bools too
             if isinstance(value, (bool, np.bool_)):
                 return bool(value)
             return None # Return None if not a valid boolean

        results = StrategyAnalysisResults(
            dataframe=df, # Return the DataFrame with all calculated indicators
            last_close=safe_decimal_from_candle(last_candle_final, 'close') or Decimal('0'), # Default to 0 if extraction fails
            current_trend_up=safe_bool_from_candle(last_candle_final, 'trend_up'), # Can be None if last row had NaN trend
            trend_just_changed=safe_bool_from_candle(last_candle_final, 'trend_changed') or False, # Default to False if extraction fails
            # Return the pruned lists of active OBs managed by the class state
            active_bull_boxes=self.bull_boxes,
            active_bear_boxes=self.bear_boxes,
            # Safely get volume normalization and ATR
            vol_norm_int=int(v.to_integral_value(rounding=ROUND_DOWN)) if (v := safe_decimal_from_candle(last_candle_final, 'vol_norm')) is not None else None,
            atr=safe_decimal_from_candle(last_candle_final, 'atr', positive_only=True),
            upper_band=safe_decimal_from_candle(last_candle_final, 'upper_band'),
            lower_band=safe_decimal_from_candle(last_candle_final, 'lower_band')
        )

        # Log summary of the final analysis results for this update cycle
        trend_status_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else \
                           f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                           f"{NEON_YELLOW}Undetermined{RESET}"
        atr_str = f"{results['atr'].normalize()}" if results['atr'] else "N/A"
        time_str = last_candle_final.name.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_final is not None else "N/A"

        self.lg.debug(f"--- Strategy Analysis Complete ({time_str}) ---")
        self.lg.debug(f"  Last Close: {results['last_close'].normalize()}")
        self.lg.debug(f"  Trend: {trend_status_str}, Trend Changed on Last: {results['trend_just_changed']}")
        self.lg.debug(f"  ATR: {atr_str}, Volume Norm (%): {results['vol_norm_int'] if results['vol_norm_int'] is not None else 'N/A'}")
        self.lg.debug(f"  Volumatic Bands (Lower/Upper): {results['lower_band'].normalize() if results['lower_band'] else 'N/A'} / {results['upper_band'].normalize() if results['upper_band'] else 'N/A'}")
        self.lg.debug(f"  Active Order Blocks (Bull/Bear): {len(results['active_bull_boxes'])} / {len(results['active_bear_boxes'])}")

        return results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """
    Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    based on the strategy analysis results (trend, Order Blocks) and the current position state.
    Also calculates initial Stop Loss and Take Profit levels for potential entries.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Signal Generator with necessary configuration parameters.

        Args:
            config: The main configuration dictionary (expects "strategy_params" and "protection" keys).
            logger: Logger instance for signal generation logging.
        """
        self.config = config
        self.logger = logger
        self.lg = logger # Alias for convenience

        # Load and validate parameters used for signal generation and initial SL/TP calculation
        try:
            strategy_cfg = config["strategy_params"]
            protection_cfg = config["protection"]

            # OB Proximity Factors (used for entry/exit thresholds relative to OB edges)
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            if self.ob_entry_proximity_factor < 1: raise ValueError("ob_entry_proximity_factor must be >= 1.0")

            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            if self.ob_exit_proximity_factor < 1: raise ValueError("ob_exit_proximity_factor must be >= 1.0")

            # Initial TP/SL ATR Multipliers
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            if self.initial_tp_atr_multiple < 0: raise ValueError("initial_take_profit_atr_multiple cannot be negative (0 disables TP)")

            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))
            if self.initial_sl_atr_multiple <= 0: raise ValueError("initial_stop_loss_atr_multiple must be positive")

            self.lg.info(f"{NEON_CYAN}--- Signal Generator Initialized ---{RESET}")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor:.4f} (Widens OB for entry check)")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor:.4f} (Shrinks opposing OB for exit check)")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()} {'(TP Disabled)' if self.initial_tp_atr_multiple == 0 else ''}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.critical(f"{NEON_RED}Fatal Error initializing SignalGenerator with config values: {e}. Using hardcoded defaults as emergency fallback.{RESET}", exc_info=True)
             # Hardcoded defaults as a safety measure if config loading/validation failed critically
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")
             # Raise the error again to potentially stop the bot if init fails
             raise ValueError("Failed to initialize SignalGenerator from config") from e

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """
        Determines the trading signal based on the strategy rules (trend, OB proximity)
        and the current position state (or lack thereof).

        Args:
            analysis_results: The dictionary containing results from the VolumaticOBStrategy update.
            open_position: The dictionary representing the current open position (from get_open_position),
                           or None if no position is currently open.

        Returns:
            A string signal: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "HOLD".
        """
        lg = self.lg

        # --- Input Validation: Ensure essential analysis results are available ---
        if not analysis_results or \
           analysis_results['current_trend_up'] is None or \
           analysis_results['last_close'] <= 0 or \
           analysis_results['atr'] is None or analysis_results['atr'] <= 0:
            lg.warning(f"{NEON_YELLOW}Signal Generation Skipped: Invalid or incomplete strategy analysis results provided (missing trend, close, or ATR). Holding.{RESET}")
            lg.debug(f"  Problematic Analysis Results: {analysis_results}") # Log the faulty results
            return "HOLD"

        # Extract key values from analysis results for easier access
        last_close: Decimal = analysis_results['last_close']
        is_trend_up: bool = analysis_results['current_trend_up'] # Already checked for None
        trend_changed: bool = analysis_results['trend_just_changed']
        active_bull_obs: List[OrderBlock] = analysis_results['active_bull_boxes']
        active_bear_obs: List[OrderBlock] = analysis_results['active_bear_boxes']
        # Determine current position side ('long', 'short', or None)
        position_side: Optional[str] = open_position.get('side') if open_position else None

        signal: str = "HOLD" # Default signal is to do nothing

        lg.debug(f"--- Signal Generation Check ---")
        lg.debug(f"  Last Close: {last_close.normalize()}")
        lg.debug(f"  Trend Up: {is_trend_up}, Trend Changed on Last: {trend_changed}")
        lg.debug(f"  Active OBs (Bull/Bear): {len(active_bull_obs)} / {len(active_bear_obs)}")
        lg.debug(f"  Current Position: {position_side or 'None'}")

        # --- Logic Branch 1: Check for Exit Conditions (only if a position is open) ---
        if position_side == 'long':
            # Exit Long Condition 1: Trend flips from Up to Down on the last candle
            if not is_trend_up and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}>>> EXIT LONG Signal (Reason: Trend Flipped Down) <<< {RESET}")
            # Exit Long Condition 2: Price approaches a relevant Bearish OB
            elif active_bear_obs: # Check only if trend hasn't flipped and bear OBs exist
                try:
                    # Find the *nearest* active Bearish OB (closest top edge to current price)
                    # Filter for OBs whose top is above the current price (relevant resistance)
                    relevant_bear_obs = [ob for ob in active_bear_obs if ob['top'] > last_close]
                    if relevant_bear_obs:
                         nearest_bear_ob = min(relevant_bear_obs, key=lambda ob: ob['top']) # Closest top edge above price
                         # Define exit threshold slightly *below* the OB top using the exit factor
                         # e.g., if factor is 1.001, exit threshold = top / 1.001
                         exit_threshold = nearest_bear_ob['top'] / self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else nearest_bear_ob['top']

                         lg.debug(f"  Long Exit Check: Nearest Bear OB {nearest_bear_ob['id']} Top={nearest_bear_ob['top'].normalize()}, Exit Threshold={exit_threshold.normalize()}")
                         if last_close >= exit_threshold:
                             signal = "EXIT_LONG"
                             lg.warning(f"{NEON_YELLOW}{BRIGHT}>>> EXIT LONG Signal (Reason: Price >= Bearish OB Exit Threshold) <<< {RESET}")
                             lg.warning(f"  Price: {last_close.normalize()}, Threshold: {exit_threshold.normalize()} (OB ID: {nearest_bear_ob['id']})")
                    else:
                         lg.debug("  Long Exit Check: No relevant Bearish OBs found above current price.")
                except Exception as e:
                    lg.warning(f"Error during Bearish OB exit check for long position: {e}")

        elif position_side == 'short':
            # Exit Short Condition 1: Trend flips from Down to Up on the last candle
            if is_trend_up and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}>>> EXIT SHORT Signal (Reason: Trend Flipped Up) <<< {RESET}")
            # Exit Short Condition 2: Price approaches a relevant Bullish OB
            elif active_bull_obs: # Check only if trend hasn't flipped and bull OBs exist
                try:
                    # Find the *nearest* active Bullish OB (closest bottom edge to current price)
                    # Filter for OBs whose bottom is below the current price (relevant support)
                    relevant_bull_obs = [ob for ob in active_bull_obs if ob['bottom'] < last_close]
                    if relevant_bull_obs:
                         nearest_bull_ob = min(relevant_bull_obs, key=lambda ob: ob['bottom'], reverse=True) # Closest bottom edge below price (max of bottoms)
                         # Define exit threshold slightly *above* the OB bottom using the exit factor
                         # e.g., if factor is 1.001, exit threshold = bottom * 1.001
                         exit_threshold = nearest_bull_ob['bottom'] * self.ob_exit_proximity_factor

                         lg.debug(f"  Short Exit Check: Nearest Bull OB {nearest_bull_ob['id']} Bottom={nearest_bull_ob['bottom'].normalize()}, Exit Threshold={exit_threshold.normalize()}")
                         if last_close <= exit_threshold:
                             signal = "EXIT_SHORT"
                             lg.warning(f"{NEON_YELLOW}{BRIGHT}>>> EXIT SHORT Signal (Reason: Price <= Bullish OB Exit Threshold) <<< {RESET}")
                             lg.warning(f"  Price: {last_close.normalize()}, Threshold: {exit_threshold.normalize()} (OB ID: {nearest_bull_ob['id']})")
                    else:
                         lg.debug("  Short Exit Check: No relevant Bullish OBs found below current price.")
                except Exception as e:
                    lg.warning(f"Error during Bullish OB exit check for short position: {e}")

        # If an exit signal was generated, return it immediately
        if signal != "HOLD":
            return signal

        # --- Logic Branch 2: Check for Entry Conditions (only if NO position is open) ---
        if position_side is None:
            # Entry Long Condition: Trend is UP *and* Price enters a relevant Bullish OB zone
            if is_trend_up and active_bull_obs:
                lg.debug("Checking for LONG entry condition (Trend UP, Price in Bullish OB)...")
                for ob in active_bull_obs:
                    # Define entry zone using proximity factor: OB bottom up to OB top * factor
                    # Ensures price has reacted within or slightly above the OB
                    entry_zone_bottom = ob['bottom']
                    entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                    lg.debug(f"  Checking Bullish OB {ob['id']}: Range=[{ob['bottom'].normalize()}-{ob['top'].normalize()}], Entry Zone=[{entry_zone_bottom.normalize()}-{entry_zone_top.normalize()}]")

                    # Check if last close price is within this calculated entry zone
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "BUY"
                        lg.info(f"{NEON_GREEN}{BRIGHT}>>> BUY Signal Triggered <<< {RESET}")
                        lg.info(f"  Reason: Trend is UP and Price ({last_close.normalize()}) entered Bullish OB zone.")
                        lg.info(f"  Triggering OB ID: {ob['id']}, Zone: [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                        break # Found a valid entry OB, stop checking others for this signal
                    else:
                        lg.debug(f"  Price {last_close.normalize()} outside entry zone for OB {ob['id']}.")

            # Entry Short Condition: Trend is DOWN *and* Price enters a relevant Bearish OB zone
            elif not is_trend_up and active_bear_obs:
                lg.debug("Checking for SHORT entry condition (Trend DOWN, Price in Bearish OB)...")
                for ob in active_bear_obs:
                    # Define entry zone using proximity factor: OB bottom / factor up to OB top
                    # Ensures price has reacted within or slightly below the OB
                    entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']
                    entry_zone_top = ob['top']
                    lg.debug(f"  Checking Bearish OB {ob['id']}: Range=[{ob['bottom'].normalize()}-{ob['top'].normalize()}], Entry Zone=[{entry_zone_bottom.normalize()}-{entry_zone_top.normalize()}]")

                    # Check if last close price is within this calculated entry zone
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "SELL"
                        lg.info(f"{NEON_RED}{BRIGHT}>>> SELL Signal Triggered <<< {RESET}")
                        lg.info(f"  Reason: Trend is DOWN and Price ({last_close.normalize()}) entered Bearish OB zone.")
                        lg.info(f"  Triggering OB ID: {ob['id']}, Zone: [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                        break # Found valid entry OB
                    else:
                         lg.debug(f"  Price {last_close.normalize()} outside entry zone for OB {ob['id']}.")

        # --- Logic Branch 3: Default to HOLD ---
        if signal == "HOLD":
            lg.debug(f"Signal Result: HOLD - No valid entry or exit condition met.")

        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: Dict, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels based on
        the entry price, current ATR, configured multipliers, and market price precision.

        Args:
            entry_price: The estimated or actual entry price (Decimal).
            signal: The entry signal that triggered this calculation ("BUY" or "SELL").
            atr: The current ATR value (Decimal).
            market_info: Dictionary of market details (must contain precision info).
            exchange: Initialized CCXT exchange object (used for formatting prices).

        Returns:
            A tuple containing (take_profit_price, stop_loss_price).
            - take_profit_price (Decimal): Calculated TP, or None if disabled (multiplier=0) or calculation failed.
            - stop_loss_price (Decimal): Calculated SL, or None if calculation failed (critical).
            Prices are formatted according to market precision rules.
        """
        lg = self.logger

        # --- Input Validation ---
        if signal.upper() not in ["BUY", "SELL"]:
            lg.error(f"Invalid signal '{signal}' provided for TP/SL calculation.")
            return None, None
        if entry_price <= 0:
            lg.error(f"Invalid entry price ({entry_price}) provided for TP/SL calculation.")
            return None, None
        if atr <= 0:
            lg.error(f"Invalid ATR value ({atr}) provided for TP/SL calculation.")
            return None, None
        if market_info['precision'].get('price') is None:
             lg.error(f"Missing price precision ('precision.price') in market info for {market_info['symbol']}. Cannot calculate precise TP/SL.")
             return None, None

        lg.debug(f"Calculating Initial TP/SL for {signal} entry at {entry_price.normalize()} with ATR {atr.normalize()}:")

        try:
            # Get minimum price tick size for formatting and validation
            price_tick_size = Decimal(str(market_info['precision']['price']))
            if price_tick_size <= 0: raise ValueError("Invalid price tick size (must be positive)")

            # Get multipliers from instance variables (already validated in __init__)
            tp_multiplier = self.initial_tp_atr_multiple
            sl_multiplier = self.initial_sl_atr_multiple

            # Calculate raw price offsets based on ATR and multipliers
            tp_offset = atr * tp_multiplier
            sl_offset = atr * sl_multiplier
            lg.debug(f"  TP Mult: {tp_multiplier}, Offset: {tp_offset.normalize()}")
            lg.debug(f"  SL Mult: {sl_multiplier}, Offset: {sl_offset.normalize()}")

            # Calculate raw TP/SL prices
            raw_tp: Optional[Decimal] = None
            # Only calculate TP if multiplier is positive (TP enabled)
            if tp_multiplier > 0:
                 raw_tp = (entry_price + tp_offset) if signal.upper() == "BUY" else (entry_price - tp_offset)

            # SL is always calculated as it's mandatory for risk management
            raw_sl: Decimal = (entry_price - sl_offset) if signal.upper() == "BUY" else (entry_price + sl_offset)

            lg.debug(f"  Raw Levels: TP={raw_tp.normalize() if raw_tp else 'N/A (Disabled)'}, SL={raw_sl.normalize()}")

            # Helper function to format and validate the calculated price level
            def format_and_validate_level(raw_level: Optional[Decimal], level_name: str) -> Optional[Decimal]:
                """Formats price using CCXT, ensures it's positive."""
                if raw_level is None: return None # Skip if TP was disabled
                # Ensure raw level is positive before formatting
                if raw_level <= 0:
                    lg.warning(f"Calculated raw {level_name} price ({raw_level}) is zero or negative. Invalid.")
                    return None
                try:
                    # Format using CCXT's price_to_precision (handles rounding based on exchange rules)
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(raw_level))
                    formatted_decimal = Decimal(formatted_str)
                    # Final check: Ensure formatted price is still positive
                    if formatted_decimal <= 0:
                        lg.warning(f"Formatted {level_name} price ({formatted_decimal}) became zero or negative after precision. Invalid.")
                        return None
                    lg.debug(f"  Formatted {level_name}: {raw_level.normalize()} -> {formatted_decimal.normalize()}")
                    return formatted_decimal
                except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
                    lg.error(f"{NEON_RED}Error formatting {level_name} level {raw_level} using exchange precision: {e}. Cannot set level.{RESET}")
                    return None

            # Format calculated TP and SL prices
            take_profit = format_and_validate_level(raw_tp, "Take Profit")
            stop_loss = format_and_validate_level(raw_sl, "Stop Loss")

            # --- Post-formatting Validation ---
            # Ensure SL is strictly on the loss side of the entry price after formatting
            if stop_loss is not None:
                sl_valid = (signal.upper() == "BUY" and stop_loss < entry_price) or \
                           (signal.upper() == "SELL" and stop_loss > entry_price)
                if not sl_valid:
                    lg.warning(f"{NEON_YELLOW}Formatted SL {stop_loss.normalize()} is not strictly beyond entry price {entry_price.normalize()} for {signal} signal (possibly due to rounding or small ATR).{RESET}")
                    # Attempt to adjust SL by one tick further away from entry as a fallback
                    adjusted_sl_raw = (entry_price - price_tick_size) if signal.upper() == "BUY" else (entry_price + price_tick_size)
                    stop_loss = format_and_validate_level(adjusted_sl_raw, "Adjusted Stop Loss")
                    if stop_loss:
                         lg.warning(f"  Adjusted SL by one tick to: {stop_loss.normalize()}")
                    else:
                         lg.error(f"{NEON_RED}  Failed to calculate a valid adjusted SL after initial SL was invalid. Critical SL failure.{RESET}")
                         # Critical failure if SL cannot be set correctly
                         return take_profit, None # Return potentially valid TP, but None SL indicates failure

            # Ensure TP (if enabled and calculated) is strictly on the profit side after formatting
            if take_profit is not None:
                tp_valid = (signal.upper() == "BUY" and take_profit > entry_price) or \
                           (signal.upper() == "SELL" and take_profit < entry_price)
                if not tp_valid:
                    lg.warning(f"{NEON_YELLOW}Formatted TP {take_profit.normalize()} is not strictly beyond entry price {entry_price.normalize()} for {signal} signal. Disabling TP.{RESET}")
                    take_profit = None # Disable TP if formatting made it invalid

            # Final log of calculated levels
            tp_status = f"{take_profit.normalize()}" if take_profit else "None (Disabled or Invalid)"
            sl_status = f"{stop_loss.normalize()}" if stop_loss else f"{NEON_RED}FAIL{RESET}"
            lg.info(f"Calculated Initial Protection Levels:")
            lg.info(f"  Take Profit: {tp_status}")
            lg.info(f"  Stop Loss: {sl_status}")

            # Stop Loss is mandatory for risk calculation and safe trading. Return None if it failed.
            if stop_loss is None:
                lg.error(f"{NEON_RED}Stop Loss calculation failed validation. Cannot proceed with trade sizing or protection.{RESET}")
                return take_profit, None # Return TP (if valid) but signal SL failure

            return take_profit, stop_loss

        except ValueError as ve: # Catch validation errors like invalid tick size
             lg.error(f"{NEON_RED}Validation error during TP/SL calculation: {ve}.{RESET}")
             return None, None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
            return None, None

# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: Dict) -> None:
    """
    Performs one full cycle of analysis and trading logic for a single specified symbol.
    This includes: fetching kline data, running the strategy analysis, generating signals,
    checking the current position, and executing/managing trades based on the signals and configuration.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol to analyze and trade (e.g., 'BTC/USDT:USDT').
        config: The main configuration dictionary.
        logger: The logger instance dedicated to this symbol.
        strategy_engine: Initialized VolumaticOBStrategy instance for this symbol.
        signal_generator: Initialized SignalGenerator instance.
        market_info: Dictionary of validated market details for the symbol.
    """
    lg = logger
    lg.info(f"\n{BRIGHT}---=== Cycle Start: Analyzing {symbol} ({config['interval']} TF) ===---{RESET}")
    cycle_start_time = time.monotonic() # Use monotonic clock for duration measurement

    # Map the config interval (e.g., "5") to CCXT's required format (e.g., "5m")
    try:
        ccxt_interval = CCXT_INTERVAL_MAP[config["interval"]]
    except KeyError:
        lg.error(f"Invalid interval '{config['interval']}' found in config during trading cycle. Using default '5m'.")
        ccxt_interval = "5m" # Fallback to a common default

    # --- 1. Determine Kline Fetch Limit ---
    # We need enough data for the strategy's lookback requirements.
    min_required_data = strategy_engine.min_data_len
    # User can specify a preferred fetch limit in config (e.g., for longer initial analysis)
    fetch_limit_from_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    # Use the larger of the strategy minimum or the user preference
    effective_fetch_limit = max(min_required_data, fetch_limit_from_config)
    # However, the actual request limit is capped by the API's maximum klines per request
    request_limit = min(effective_fetch_limit, BYBIT_API_KLINE_LIMIT)
    lg.info(f"Data Requirements: Strategy Min={min_required_data}, Config Preferred={fetch_limit_from_config}, API Max={BYBIT_API_KLINE_LIMIT}")
    lg.info(f"Attempting to fetch {request_limit} klines for {symbol} ({ccxt_interval})...")

    # --- 2. Fetch Kline Data ---
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=request_limit, logger=lg)
    fetched_count = len(klines_df)

    # --- 3. Validate Fetched Data Sufficiency ---
    if klines_df.empty or fetched_count < min_required_data:
        # Check if the reason for insufficient data was hitting the API limit
        api_limit_was_hit = (request_limit == BYBIT_API_KLINE_LIMIT and fetched_count == BYBIT_API_KLINE_LIMIT)

        if klines_df.empty:
            lg.error(f"Failed to fetch any valid kline data for {symbol}. Skipping analysis this cycle.")
        elif api_limit_was_hit and fetched_count < min_required_data:
            # Critical situation: API limit prevents getting enough data for the configured strategy
            lg.error(f"{NEON_RED}{BRIGHT}CRITICAL DATA SHORTFALL:{RESET} Fetched max {fetched_count} klines (API limit), but strategy requires {min_required_data}.")
            lg.error(f"{NEON_YELLOW}  This configuration is likely unusable. ACTION REQUIRED: Reduce lookback periods (e.g., vt_length, vt_atr_period, vt_vol_ema_length, pivot lookbacks) in config.json.{RESET}")
        else: # Got some data, but not enough for the strategy's lookback
            lg.error(f"Fetched only {fetched_count} klines, but strategy requires {min_required_data}. Insufficient data for analysis. Skipping cycle.")
        # Skip the rest of the cycle if data is insufficient
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Cycle End ({symbol}, Insufficient Data, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # --- 4. Run Strategy Analysis Engine ---
    lg.debug("Running strategy analysis engine on fetched data...")
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis engine encountered an error: {analysis_err}{RESET}", exc_info=True)
        # Stop cycle if analysis fails critically
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Cycle End ({symbol}, Analysis Error, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # --- 5. Validate Essential Analysis Results ---
    # Ensure key metrics needed for signal generation and protection are valid
    if not analysis_results or \
       analysis_results['current_trend_up'] is None or \
       analysis_results['last_close'] <= 0 or \
       analysis_results['atr'] is None or analysis_results['atr'] <= 0:
        lg.error(f"{NEON_RED}Strategy analysis did not produce valid results (Trend, Last Close, or ATR missing/invalid). Skipping signal generation and trading actions.{RESET}")
        lg.debug(f"Problematic Analysis Results: {analysis_results}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Cycle End ({symbol}, Invalid Analysis Results, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # Extract key results for use
    latest_close_price: Decimal = analysis_results['last_close']
    current_atr_value: Decimal = analysis_results['atr']
    lg.info(f"Strategy Analysis OK: Trend={analysis_results['current_trend_up']}, Last Close={latest_close_price.normalize()}, ATR={current_atr_value.normalize()}")

    # --- 6. Get Current Market State (Live Price & Position Status) ---
    lg.debug("Fetching current market price and position status...")
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    open_position = get_open_position(exchange, symbol, lg) # Returns dict or None

    # Use live market price if available and valid, otherwise fallback to last kline close for checks
    price_for_checks = current_market_price if current_market_price and current_market_price > 0 else latest_close_price
    if price_for_checks <= 0:
        # This should be rare if analysis results were valid, but check defensively
        lg.error(f"{NEON_RED}Cannot determine a valid current price for checks (Live: {current_market_price}, Last Close: {latest_close_price}). Skipping protection/signal checks.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Cycle End ({symbol}, Invalid Price, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    if current_market_price is None or current_market_price <= 0:
        lg.warning(f"{NEON_YELLOW}Failed to fetch valid live market price. Using last kline close ({latest_close_price.normalize()}) for subsequent checks.{RESET}")
    else:
        lg.debug(f"Using live market price ({current_market_price.normalize()}) for subsequent checks.")


    # --- 7. Generate Trading Signal ---
    lg.debug("Generating trading signal based on analysis and position state...")
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
        lg.info(f"Generated Signal: {BRIGHT}{signal}{RESET}")
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation encountered an error: {signal_err}{RESET}", exc_info=True)
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Cycle End ({symbol}, Signal Error, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return # Cannot proceed without a signal

    # --- 8. Trading Logic Execution ---
    trading_enabled = config.get("enable_trading", False)

    # --- Scenario: Trading Disabled (Log potential actions) ---
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading is DISABLED in config.json.{RESET} Analysis complete.")
        if open_position is None and signal in ["BUY", "SELL"]:
            lg.info(f"  Signal: {signal}. Would attempt to {signal} if trading were enabled.")
        elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]:
            lg.info(f"  Signal: {signal}. Would attempt to close current {open_position['side']} position if trading were enabled.")
        elif signal == "HOLD":
             lg.info(f"  Signal: HOLD. No entry or exit action indicated.")
        else: # Should not happen with current signals
             lg.warning(f"  Signal: {signal}. No specific action defined for this signal in analysis-only mode.")
        # End cycle here if trading is disabled
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Analysis-Only Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # --- Scenario: Trading IS Enabled ---
    lg.debug(f"Trading is ENABLED. Processing signal '{signal}'...")

    # === Action Branch 1: No Position -> Consider Entry ===
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** Signal: {signal} | Current Position: None | Attempting Trade Entry... ***{RESET}")

        # i. Fetch Current Balance for Sizing
        lg.debug("Fetching available balance for position sizing...")
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= Decimal('0'):
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot size order. Failed to fetch valid balance or balance is zero/negative ({balance}).{RESET}")
            return # Stop entry sequence if balance invalid

        # ii. Calculate Initial TP/SL (using latest close as estimated entry price for now)
        #     SL is critical for position sizing.
        lg.debug(f"Calculating initial TP/SL based on last close ({latest_close_price.normalize()}) and current ATR ({current_atr_value.normalize()})...")
        initial_tp, initial_sl = signal_generator.calculate_initial_tp_sl(
            entry_price=latest_close_price, # Use last close as estimate before entry
            signal=signal,
            atr=current_atr_value,
            market_info=market_info,
            exchange=exchange
        )
        if initial_sl is None:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to calculate a valid initial Stop Loss based on current ATR and estimated entry. Cannot proceed without SL.{RESET}")
            return # Stop entry sequence if SL calculation fails

        # iii. Set Leverage (if applicable, for contract markets)
        leverage_set_ok = True # Assume OK for spot or if leverage setting fails but isn't critical
        if market_info['is_contract']:
            leverage_val = int(config.get('leverage', 0))
            if leverage_val > 0:
                lg.debug(f"Setting leverage to {leverage_val}x for {symbol} before entry...")
                leverage_set_ok = set_leverage_ccxt(exchange, symbol, leverage_val, market_info, lg)
                if not leverage_set_ok:
                    lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set leverage for {symbol}. Check permissions and settings.{RESET}")
                    return # Stop entry if leverage setting fails for contracts
            else:
                lg.info(f"Leverage setting skipped (config leverage is {leverage_val}). Using exchange default.")

        # iv. Calculate Position Size based on risk, balance, and SL distance
        lg.debug("Calculating position size based on risk, balance, and initial SL...")
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=config["risk_per_trade"],
            initial_stop_loss_price=initial_sl, # Use the calculated initial SL
            entry_price=latest_close_price, # Use last close as estimate for sizing
            market_info=market_info,
            exchange=exchange,
            logger=lg
        )
        if position_size is None or position_size <= 0:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Position size calculation failed or resulted in zero/negative size ({position_size}). Check risk settings, balance, and market limits.{RESET}")
            return # Stop entry sequence if sizing fails

        # v. Place Market Order to Enter Position
        lg.info(f"{BRIGHT}===> Placing {signal} Market Order | Size: {position_size.normalize()} {market_info.get('base', '')} <==={RESET}")
        trade_order = place_trade(
            exchange=exchange,
            symbol=symbol,
            trade_signal=signal,
            position_size=position_size,
            market_info=market_info,
            logger=lg,
            reduce_only=False # This is an entry order, not reduce only
        )

        # vi. Post-Trade Actions (Confirm Position & Set Protection)
        if trade_order and trade_order.get('id'):
            order_id = trade_order['id']
            # Wait briefly for the position update on the exchange side
            confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
            lg.info(f"Order {order_id} submitted. Waiting {confirm_delay}s for position confirmation and fill details...")
            time.sleep(confirm_delay)

            # Attempt to fetch the newly opened position to confirm and get actual entry price
            lg.debug("Attempting to confirm open position after order placement...")
            confirmed_position = get_open_position(exchange, symbol, lg)

            if confirmed_position and confirmed_position.get('size_decimal') and abs(confirmed_position['size_decimal']) > 0:
                lg.info(f"{NEON_GREEN}Position Confirmed after trade!{RESET}")
                try:
                    # Get actual entry price if available, fallback to estimate if needed
                    entry_price_actual_str = confirmed_position.get('entryPrice')
                    entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close_price
                    if entry_price_actual <= 0:
                         lg.warning(f"Confirmed position entry price is invalid ({entry_price_actual_str}). Using last close ({latest_close_price}) for protection calculation.")
                         entry_price_actual = latest_close_price # Fallback again

                    lg.info(f"  Actual/Estimated Entry Price for Protection: {entry_price_actual.normalize()}")

                    # Re-calculate TP/SL based on the actual/confirmed entry price for potentially better accuracy
                    # Use the *same* ATR value from the analysis that triggered the entry
                    lg.debug("Recalculating TP/SL based on confirmed entry price for protection setting...")
                    protection_tp, protection_sl = signal_generator.calculate_initial_tp_sl(
                        entry_price=entry_price_actual, # Use confirmed entry
                        signal=signal,
                        atr=current_atr_value, # Use ATR from entry signal candle
                        market_info=market_info,
                        exchange=exchange
                    )

                    if protection_sl is None:
                        # This is critical - position is open but SL calculation failed again
                        lg.error(f"{NEON_RED}{BRIGHT}CRITICAL ERROR: Position opened, but failed to recalculate SL ({protection_sl}) based on confirmed entry price! Position is currently UNPROTECTED! Manual intervention required!{RESET}")
                        # Do NOT proceed with protection setting if SL is invalid
                    else:
                        # --- Set Position Protection (TSL or Fixed SL/TP) ---
                        protection_config = config["protection"]
                        protection_set_success = False
                        if protection_config.get("enable_trailing_stop", True):
                            lg.info(f"Setting Initial Trailing Stop Loss (based on entry {entry_price_actual.normalize()})...")
                            # Pass the recalculated TP to set along with TSL
                            protection_set_success = set_trailing_stop_loss(
                                exchange=exchange, symbol=symbol, market_info=market_info,
                                position_info=confirmed_position, config=config, logger=lg,
                                take_profit_price=protection_tp # Use recalculated TP
                            )
                        elif protection_sl or protection_tp: # Only set if fixed SL or TP is enabled/calculated
                            lg.info(f"Setting Initial Fixed Stop Loss / Take Profit (based on entry {entry_price_actual.normalize()})...")
                            # Use the internal helper to set fixed SL and/or TP
                            protection_set_success = _set_position_protection(
                                exchange=exchange, symbol=symbol, market_info=market_info,
                                position_info=confirmed_position, logger=lg,
                                stop_loss_price=protection_sl,    # Use recalculated SL
                                take_profit_price=protection_tp     # Use recalculated TP
                                # Ensure TSL params are None/default when setting fixed SL/TP
                            )
                        else:
                            lg.info("No protection (TSL or Fixed SL/TP) enabled or calculated. Position opened without automated protection.")
                            protection_set_success = True # No protection needed, so considered "successful" in context

                        # Log final outcome of the entry sequence
                        if protection_set_success:
                            lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY SEQUENCE COMPLETE ({symbol} {signal}) - Position Opened & Protection Set/Managed ==={RESET}")
                        else:
                            lg.error(f"{NEON_RED}{BRIGHT}=== ENTRY SEQUENCE FAILED ({symbol} {signal}) - POSITION OPENED, BUT FAILED TO SET PROTECTION via API! MANUAL INTERVENTION REQUIRED! ==={RESET}")

                except Exception as post_trade_err:
                    lg.error(f"{NEON_RED}Error during post-trade setup (protection calculation/setting): {post_trade_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Position likely opened for {symbol}, but setting protection failed. Manual check and intervention required!{RESET}")
            else:
                # This is problematic - order was submitted but position not found after delay
                lg.error(f"{NEON_RED}Order {order_id} submitted, but FAILED TO CONFIRM active open position after {confirm_delay}s delay! Manual check required! Possible fill issue, API delay, or order rejection.{RESET}")
                # Could try fetching order status here for more info: exchange.fetch_order(order_id, symbol)
        else:
            # Order placement itself failed
            lg.error(f"{NEON_RED}=== ENTRY SEQUENCE FAILED ({symbol} {signal}) - Order placement failed. No position opened. ===")

    # === Action Branch 2: Existing Position -> Consider Exit or Manage ===
    elif open_position:
        position_side = open_position['side']
        position_size_dec = open_position.get('size_decimal') # Should be present from get_open_position

        # Validate position data needed for management
        if position_size_dec is None or position_size_dec == 0:
             lg.error(f"Cannot manage position for {symbol}: Position data is missing 'size_decimal' or size is zero. Skipping management. Position Info: {open_position}")
             return # Cannot proceed without valid size

        lg.info(f"Signal: {signal} | Current Position: {position_side.upper()} (Size: {position_size_dec.normalize()})")

        # --- Check for Exit Signal ---
        # Does the generated signal match the condition to close the current position?
        exit_signal_triggered = (signal == "EXIT_LONG" and position_side == 'long') or \
                                (signal == "EXIT_SHORT" and position_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** Signal: {signal}! Initiating CLOSE Sequence for {position_side} position... ***{RESET}")
            try:
                # Size to close is the absolute value of the current position size
                size_to_close = abs(position_size_dec)

                if size_to_close <= 0: # Should be caught above, but double-check
                    lg.warning(f"Attempting to close {symbol} position, but size is zero or negative ({position_size_dec}). Assuming already closed or error in position data.")
                    return

                lg.info(f"{BRIGHT}===> Placing {signal} Market Order (Reduce Only) | Size: {size_to_close.normalize()} <==={RESET}")
                # Use place_trade with reduce_only=True
                close_order = place_trade(
                    exchange=exchange,
                    symbol=symbol,
                    trade_signal=signal, # Pass the exit signal ("EXIT_LONG" or "EXIT_SHORT")
                    position_size=size_to_close,
                    market_info=market_info,
                    logger=lg,
                    reduce_only=True # CRITICAL: Ensure this is a closing order
                )

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) submitted successfully for {symbol}. Position should be closing/closed.{RESET}")
                    # Optional: Could add a short wait and confirmation that position is gone
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check and intervention required!{RESET}")

            except Exception as close_err:
                lg.error(f"{NEON_RED}Error occurred during position closing attempt: {close_err}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual position close may be required for {symbol}!{RESET}")

        # --- No Exit Signal -> Perform Position Management (BE, TSL checks) ---
        else: # Signal is HOLD or an entry signal for the *opposite* direction (ignore entry if already in position)
            lg.debug(f"Signal ({signal}) allows holding {position_side} position. Performing position management checks...")
            protection_config = config["protection"]

            # Extract current protection levels and entry price from position info safely
            try:
                # Check if TSL is active by looking for a positive trailingStop distance in 'info'
                is_tsl_active_on_exchange = open_position.get('trailingStopLoss') is not None and Decimal(str(open_position['trailingStopLoss'])) > 0
            except (ValueError, InvalidOperation, TypeError):
                is_tsl_active_on_exchange = False # Assume not active if conversion fails
            try:
                current_sl_price = Decimal(str(open_position['stopLossPrice'])) if open_position.get('stopLossPrice') else None
            except (ValueError, InvalidOperation, TypeError):
                current_sl_price = None
            try:
                current_tp_price = Decimal(str(open_position['takeProfitPrice'])) if open_position.get('takeProfitPrice') else None
            except (ValueError, InvalidOperation, TypeError):
                current_tp_price = None
            try:
                entry_price = Decimal(str(open_position['entryPrice'])) if open_position.get('entryPrice') else None
            except (ValueError, InvalidOperation, TypeError):
                entry_price = None

            lg.debug(f"  Current Protection State from API: TSL Active={is_tsl_active_on_exchange}, SL={current_sl_price}, TP={current_tp_price}")

            # --- Break-Even Logic ---
            be_enabled = protection_config.get("enable_break_even", True)
            # Check BE only if: BE is enabled, TSL is *not* currently active on the exchange,
            # and we have the necessary data (entry price, ATR, current price)
            if be_enabled and not is_tsl_active_on_exchange and entry_price and current_atr_value > 0 and price_for_checks > 0:
                lg.debug(f"Checking Break-Even condition (BE Enabled, TSL Inactive)...")
                lg.debug(f"  Entry={entry_price.normalize()}, Current Price={price_for_checks.normalize()}, ATR={current_atr_value.normalize()}")
                try:
                    be_trigger_atr_multiple = Decimal(str(protection_config["break_even_trigger_atr_multiple"]))
                    be_offset_ticks = int(protection_config["break_even_offset_ticks"])
                    price_tick_size = Decimal(str(market_info['precision']['price']))

                    # Validate BE parameters
                    if be_trigger_atr_multiple <= 0 or be_offset_ticks < 0 or price_tick_size <= 0:
                        raise ValueError("Invalid Break-Even parameters in config (trigger_atr > 0, offset_ticks >= 0) or market info (tick_size > 0)")

                    # Calculate profit in terms of ATR multiples from entry
                    profit_pips = (price_for_checks - entry_price) if position_side == 'long' else (entry_price - price_for_checks)
                    # Avoid division by zero if ATR is somehow zero
                    profit_in_atr = profit_pips / current_atr_value if current_atr_value > 0 else Decimal(0)

                    lg.debug(f"  Profit Distance: {profit_pips.normalize()}, Profit in ATRs: {profit_in_atr:.3f}")
                    lg.debug(f"  BE Trigger Requirement (ATRs): {be_trigger_atr_multiple.normalize()}")

                    # --- Check if profit target for BE is reached ---
                    if profit_in_atr >= be_trigger_atr_multiple:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}*** Break-Even Profit Target REACHED! (Profit ATRs {profit_in_atr:.3f} >= {be_trigger_atr_multiple}) ***{RESET}")

                        # Calculate the target break-even SL price (entry +/- offset)
                        be_offset_value = price_tick_size * Decimal(str(be_offset_ticks))
                        be_sl_price_raw = (entry_price + be_offset_value) if position_side == 'long' else (entry_price - be_offset_value)

                        # Format the BE SL price using exchange precision rules
                        be_sl_price_formatted: Optional[Decimal] = None
                        try:
                            fmt_str = exchange.price_to_precision(symbol, float(be_sl_price_raw))
                            be_sl_price_formatted = Decimal(fmt_str)
                            if be_sl_price_formatted <= 0: raise ValueError("Formatted BE SL price is zero or negative")
                        except Exception as fmt_err:
                             lg.error(f"{NEON_RED}Failed to format calculated BE SL price {be_sl_price_raw}: {fmt_err}. Cannot move SL to BE.{RESET}")

                        if be_sl_price_formatted:
                            lg.debug(f"  Calculated BE Stop Loss Price: {be_sl_price_formatted.normalize()} (Entry {entry_price.normalize()} {'+' if position_side == 'long' else '-'} {be_offset_ticks} ticks)")

                            # --- Check if the new BE SL is actually better than the current SL ---
                            # We only want to move SL *towards* profit, not further away.
                            should_update_sl = False
                            if current_sl_price is None:
                                should_update_sl = True # Always set SL if none exists
                                lg.info("  Current SL is not set. Setting SL to Break-Even level.")
                            elif (position_side == 'long' and be_sl_price_formatted > current_sl_price) or \
                                 (position_side == 'short' and be_sl_price_formatted < current_sl_price):
                                should_update_sl = True # New BE SL is closer to entry (better)
                                lg.info(f"  New BE SL {be_sl_price_formatted.normalize()} is better than current SL {current_sl_price.normalize()}. Updating.")
                            else:
                                lg.info(f"  Current SL {current_sl_price.normalize()} is already at or better than calculated BE SL {be_sl_price_formatted.normalize()}. No BE update needed.")

                            # If update is needed, call the protection function to set the new SL
                            if should_update_sl:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even at {be_sl_price_formatted.normalize()} ***{RESET}")
                                # Use _set_position_protection to set *only* the SL.
                                # Pass the current TP price (if any) to attempt to preserve it.
                                # Set TSL params to None to ensure fixed SL is set.
                                be_set_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be
