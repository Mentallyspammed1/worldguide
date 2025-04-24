# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.5: Fixed SyntaxError in get_open_position try/except block.
# Enhancements: Improved docstrings, type hinting, comments, logging clarity,
#               fixed loop delay config usage, removed unused import.

# --- Core Libraries ---
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
from zoneinfo import ZoneInfo # Requires tzdata package

# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy
import pandas as pd # Requires pandas
import pandas_ta as ta # Requires pandas_ta
import requests # Requires requests
# import websocket # Requires websocket-client (Removed - Unused)
import ccxt # Requires ccxt
from colorama import Fore, Style, init # Requires colorama
from dotenv import load_dotenv # Requires python-dotenv

# --- Initialize Environment and Settings ---
getcontext().prec = 28 # Set Decimal precision
init(autoreset=True) # Initialize Colorama
load_dotenv() # Load environment variables from .env file

# --- Constants ---
# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
try:
    # Attempt to load timezone, fallback to UTC if tzdata is not installed
    TIMEZONE = ZoneInfo("America/Chicago") # Example: Use 'UTC' or your preferred IANA timezone
except Exception:
    print(f"{Fore.RED}Failed to initialize timezone. Install 'tzdata' package (`pip install tzdata`). Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")

# API Interaction Settings
MAX_API_RETRIES: int = 3 # Maximum number of retries for API calls
RETRY_DELAY_SECONDS: int = 5 # Base delay between API retries
POSITION_CONFIRM_DELAY_SECONDS: int = 8 # Wait time after placing order before confirming position
LOOP_DELAY_SECONDS: int = 15 # Default delay between trading cycles (can be overridden in config)
BYBIT_API_KLINE_LIMIT: int = 1000 # Bybit V5 Kline limit per API request

# Timeframes
VALID_INTERVALS: List[str] = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP: Dict[str, str] = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling Limits
DEFAULT_FETCH_LIMIT: int = 750 # Default kline fetch limit if not set in config (used if less than min_data_len)
MAX_DF_LEN: int = 2000 # Internal limit to prevent excessive memory usage by Pandas DataFrame

# Strategy Defaults (Used if values are missing or invalid in config.json)
DEFAULT_VT_LENGTH: int = 40
DEFAULT_VT_ATR_PERIOD: int = 200
DEFAULT_VT_VOL_EMA_LENGTH: int = 950 # Adjusted default (Original 1000 often > API Limit)
DEFAULT_VT_ATR_MULTIPLIER: float = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER: float = 4.0 # Note: Step ATR Multiplier currently unused in core logic
DEFAULT_OB_SOURCE: str = "Wicks" # Or "Body" - Determines if Order Blocks use candle wicks or bodies
DEFAULT_PH_LEFT: int = 10; DEFAULT_PH_RIGHT: int = 10 # Pivot High lookback periods
DEFAULT_PL_LEFT: int = 10; DEFAULT_PL_RIGHT: int = 10 # Pivot Low lookback periods
DEFAULT_OB_EXTEND: bool = True # Whether to visually extend OBs to the latest candle
DEFAULT_OB_MAX_BOXES: int = 50 # Maximum number of active Order Blocks to track

# Dynamically loaded from config: QUOTE_CURRENCY (e.g., "USDT")

# Logging Colors
NEON_GREEN: str = Fore.LIGHTGREEN_EX; NEON_BLUE: str = Fore.CYAN; NEON_PURPLE: str = Fore.MAGENTA
NEON_YELLOW: str = Fore.YELLOW; NEON_RED: str = Fore.LIGHTRED_EX; NEON_CYAN: str = Fore.CYAN
RESET: str = Style.RESET_ALL; BRIGHT: str = Style.BRIGHT; DIM: str = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter that redacts sensitive API keys from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record, replacing API keys with placeholders."""
        msg = super().format(record)
        if API_KEY: msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET: msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a logger instance with both console and rotating file handlers.

    Args:
        name: The name for the logger (often the trading symbol or 'init').

    Returns:
        The configured logging.Logger instance.
    """
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize for filename
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Capture all levels; handlers control output level

    # File Handler (DEBUG level, rotating)
    try:
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger {log_filename}: {e}{RESET}")

    # Console Handler (Level from ENV or INFO default, timezone-aware)
    try:
        sh = logging.StreamHandler()
        # Use timezone-aware timestamps for console output
        logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
        sf = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        sh.setFormatter(sf)
        # Get desired console log level from environment variable, default to INFO
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid
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
    Recursively ensures all keys from default_config exist in config.
    Adds missing keys with default values and logs the additions.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The dictionary containing default keys and values.
        parent_key: Used for logging nested key paths.

    Returns:
        A tuple containing the updated configuration dictionary and a boolean
        indicating if any changes were made.
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default value: {default_value}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed:
                updated_config[key] = nested_config
                changed = True
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file, validates it against defaults,
    adds missing keys, validates types/ranges, and saves updates if needed.

    Args:
        filepath: The path to the configuration JSON file.

    Returns:
        The loaded and validated configuration dictionary. Returns defaults
        if the file cannot be loaded or parsed correctly.
    """
    # Define the default configuration structure and values
    default_config = {
        "interval": "5",                    # Default timeframe (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS, # API retry delay
        "fetch_limit": DEFAULT_FETCH_LIMIT, # Preferred number of klines to fetch
        "orderbook_limit": 25,              # Max order book depth (currently unused)
        "enable_trading": False,            # Master switch for placing real trades
        "use_sandbox": True,                # Use Bybit's testnet environment
        "risk_per_trade": 0.01,             # Fraction of balance to risk per trade (e.g., 0.01 = 1%)
        "leverage": 20,                     # Desired leverage for contract trading
        "max_concurrent_positions": 1,      # Max open positions per symbol (currently supports 1)
        "quote_currency": "USDT",           # The quote currency for balance checks and sizing
        "loop_delay_seconds": LOOP_DELAY_SECONDS, # Delay between trading cycles
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS, # Wait after order to check position
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
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,     # Max active OBs to track
            "ob_entry_proximity_factor": 1.005,       # Factor to slightly widen OB for entry check (e.g., 1.005 = 0.5% wider)
            "ob_exit_proximity_factor": 1.001         # Factor to slightly widen opposing OB for exit check (e.g., 1.001 = 0.1% wider)
        },
        "protection": {                     # Position protection settings
             "enable_trailing_stop": True,             # Use Trailing Stop Loss (TSL)
             "trailing_stop_callback_rate": 0.005,     # TSL distance as a percentage of activation price (e.g., 0.005 = 0.5%)
             "trailing_stop_activation_percentage": 0.003, # Profit percentage to activate TSL (e.g., 0.003 = 0.3%)
             "enable_break_even": True,                # Move SL to break-even after profit target hit
             "break_even_trigger_atr_multiple": 1.0,   # ATR multiples in profit to trigger BE
             "break_even_offset_ticks": 2,             # Ticks above/below entry for BE SL placement
             "initial_stop_loss_atr_multiple": 1.8,    # Initial SL distance in ATR multiples
             "initial_take_profit_atr_multiple": 0.7   # Initial TP distance in ATR multiples (0 to disable)
        }
    }
    config_needs_saving = False
    loaded_config = {}

    # Create default config file if it doesn't exist
    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}")
            return default_config
        except IOError as e:
            init_logger.error(f"{NEON_RED}Error creating default config file '{filepath}': {e}. Using default values.{RESET}")
            return default_config

    # Load existing config file
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}. Attempting to recreate.{RESET}")
        try:
            # Try to recreate the file with defaults if JSON is invalid
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            return default_config
        except IOError as e_create:
            init_logger.error(f"{NEON_RED}Error recreating default config file: {e_create}. Using default values.{RESET}")
            return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error loading config file '{filepath}': {e}. Using default values.{RESET}", exc_info=True)
        return default_config

    try:
        # Ensure all default keys exist in the loaded config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg: Dict, key_path: str, min_val: Union[int, float, Decimal], max_val: Union[int, float, Decimal],
                             is_strict_min: bool = False, is_int: bool = False, allow_zero: bool = False) -> bool:
            """
            Validates a numeric config value, logs warning & uses default if invalid.

            Args:
                cfg: The config dict (potentially nested).
                key_path: Dot-separated path to the key (e.g., "strategy_params.vt_length").
                min_val: Minimum allowed value.
                max_val: Maximum allowed value.
                is_strict_min: If True, value must be strictly greater than min_val.
                is_int: If True, value must be an integer.
                allow_zero: If True, zero is allowed even if outside min/max range.

            Returns:
                True if the value was corrected, False otherwise.
            """
            nonlocal config_needs_saving
            keys = key_path.split('.')
            current_level = cfg
            default_level = default_config
            try:
                # Traverse dictionaries to find the value and its default
                for key in keys[:-1]:
                    current_level = current_level[key]
                    default_level = default_level[key]
                leaf_key = keys[-1]
                original_val = current_level.get(leaf_key)
                default_val = default_level.get(leaf_key)
            except (KeyError, TypeError):
                init_logger.error(f"Config validation error: Invalid path '{key_path}'.")
                return False # Path is invalid, cannot validate

            if original_val is None:
                # This case should ideally be handled by _ensure_config_keys
                init_logger.warning(f"Config '{key_path}': Key missing during validation. Should have been added.")
                return False

            corrected = False
            final_val = original_val
            try:
                num_val = Decimal(str(original_val))
                min_check = num_val > Decimal(str(min_val)) if is_strict_min else num_val >= Decimal(str(min_val))
                max_check = num_val <= Decimal(str(max_val))

                if not (min_check and max_check) and not (allow_zero and num_val == 0):
                    raise ValueError("Value out of allowed range")

                target_type = int if is_int else float
                final_val = target_type(num_val)

                # Check if type conversion or value changed significantly
                if type(final_val) is not type(original_val) or final_val != original_val:
                    # Allow for float representation differences
                    if not isinstance(original_val, (float, int)) or abs(float(original_val) - float(final_val)) > 1e-9:
                       corrected = True
            except (ValueError, InvalidOperation, TypeError):
                init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{original_val}'. Using default: {default_val}.{RESET}")
                final_val = default_val
                corrected = True

            if corrected:
                current_level[leaf_key] = final_val
                config_needs_saving = True
            return corrected

        # --- Apply Specific Validations ---
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid config interval '{updated_config.get('interval')}'. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True

        validate_numeric(updated_config, "retry_delay", 1, 60, is_int=True)
        # Allow fetch_limit > API limit internally, kline fetch handles actual request cap
        validate_numeric(updated_config, "fetch_limit", 100, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", 0, 1, is_strict_min=True) # Risk must be > 0% and <= 100%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True) # Leverage 0 often means cross/spot
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600, is_int=True)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, 1000, is_int=True)
        # Validate Vol EMA length, but allow user setting > API limit (handled later in strategy init)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, MAX_DF_LEN, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20.0)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)
        if updated_config["strategy_params"].get("ob_source") not in ["Wicks", "Body"]:
             init_logger.warning(f"Invalid ob_source '{updated_config['strategy_params']['ob_source']}'. Using default '{DEFAULT_OB_SOURCE}'.")
             updated_config["strategy_params"]["ob_source"] = DEFAULT_OB_SOURCE; config_needs_saving = True
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0001, 0.5, is_strict_min=True) # TSL > 0
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0, 0.5, allow_zero=True) # TSL Act can be 0
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.1, 10.0)
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True) # BE offset can be 0
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.1, 100.0)
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0, 100.0, allow_zero=True) # TP can be 0 (disabled)

        # Save the configuration file if any keys were added or values corrected
        if config_needs_saving:
             try:
                 # Convert any internal Decimal objects back to standard types for JSON
                 config_to_save = json.loads(json.dumps(updated_config))
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated config file '{filepath}': {save_err}{RESET}", exc_info=True)

        return updated_config

    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error processing config: {e}. Using default values.{RESET}", exc_info=True)
        return default_config

# --- Load Global Configuration ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Ensure QUOTE_CURRENCY is set globally

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """
    Initializes the CCXT Bybit exchange object.
    Handles API key setup, sandbox mode, rate limiting, and market loading with retries.

    Args:
        logger: The logger instance for logging messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True, # Enable built-in rate limiter
            'options': {
                'defaultType': 'linear', # Default contract type (can be overridden by market info)
                'adjustForTimeDifference': True, # Auto-adjust for clock skew
                # Set longer timeouts for potentially slow operations
                'fetchTickerTimeout': 15000,    # 15 seconds
                'fetchBalanceTimeout': 20000,   # 20 seconds
                'createOrderTimeout': 30000,    # 30 seconds
                'cancelOrderTimeout': 20000,    # 20 seconds
                'fetchPositionsTimeout': 20000, # 20 seconds
                'fetchOHLCVTimeout': 60000,     # 60 seconds for potentially large kline fetches
            }
        }
        exchange = ccxt.bybit(exchange_options)

        # Set sandbox mode if configured
        if CONFIG.get('use_sandbox', True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}{BRIGHT}USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK{RESET}")

        # Load markets with retries
        lg.info(f"Loading markets for {exchange.id}...")
        markets_loaded = False
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # Force reload on retries
                exchange.load_markets(reload=True if attempt > 0 else False)
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols).")
                    markets_loaded = True
                    break
                else:
                    lg.warning(f"Market loading returned empty result (Attempt {attempt+1}/{MAX_API_RETRIES+1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error loading markets (Attempt {attempt+1}/{MAX_API_RETRIES+1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.critical(f"{NEON_RED}Failed to load markets after {MAX_API_RETRIES+1} attempts due to network errors: {e}. Exiting.{RESET}")
                    return None
            except ccxt.RateLimitExceeded as e:
                 wait = RETRY_DELAY_SECONDS * 3
                 lg.warning(f"Rate limit hit loading markets: {e}. Waiting {wait}s...")
                 time.sleep(wait) # Wait longer for rate limits
                 # Don't increment attempt count for rate limit, just retry waiting
            except Exception as e:
                lg.critical(f"{NEON_RED}An unexpected error occurred loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None

        if not markets_loaded:
            lg.critical(f"{NEON_RED}Failed to load markets after all attempts. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized: {exchange.id} (Sandbox: {CONFIG.get('use_sandbox')})")

        # Attempt initial balance fetch
        lg.info(f"Attempting initial balance fetch for {QUOTE_CURRENCY}...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                lg.info(f"{NEON_GREEN}Initial balance check OK: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                # Allow proceeding if trading is disabled, but warn heavily. Exit if trading enabled.
                lg.critical(f"{NEON_RED}Initial balance fetch FAILED for {QUOTE_CURRENCY}.{RESET}")
                if CONFIG.get('enable_trading', False):
                    lg.critical(f"{NEON_RED}Trading is enabled, but balance check failed. Cannot proceed safely. Exiting.{RESET}")
                    return None
                else:
                    lg.warning(f"{NEON_YELLOW}Trading is disabled. Proceeding cautiously despite balance check failure.{RESET}")
        except ccxt.AuthenticationError as auth_err:
             lg.critical(f"{NEON_RED}Authentication Error during initial balance fetch: {auth_err}. Check API Key/Secret/Permissions. Exiting.{RESET}")
             return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Unexpected error during initial balance fetch: {balance_err}.{RESET}", exc_info=True)
             if CONFIG.get('enable_trading', False):
                 lg.critical(f"{NEON_RED}Trading is enabled, critical error during balance check. Exiting.{RESET}")
                 return None

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}Failed to initialize exchange due to Authentication Error: {e}. Check API Key/Secret.{RESET}")
        return None
    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize exchange: {e}{RESET}", exc_info=True)
        return None

# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the current market price for a symbol using fetch_ticker with retries.
    Uses 'last' price primarily, falls back to mid-price (bid+ask)/2 if available.

    Args:
        exchange: The initialized CCXT exchange object.
        symbol: The market symbol (e.g., 'BTC/USDT:USDT').
        logger: The logger instance.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price = None

            # Helper to safely convert ticker values to Decimal
            def safe_decimal(value_str: Optional[str], name: str) -> Optional[Decimal]:
                try:
                    if value_str is not None and str(value_str).strip() != '':
                        dec_val = Decimal(str(value_str))
                        if dec_val > 0:
                            return dec_val
                        else:
                           lg.debug(f"Ticker field '{name}' is zero or negative ('{value_str}').")
                           return None
                except (InvalidOperation, ValueError, TypeError):
                    lg.debug(f"Could not convert ticker field '{name}' ('{value_str}') to Decimal.")
                    return None
                return None # If value_str is None or empty

            # Try 'last' price first
            price = safe_decimal(ticker.get('last'), 'last')

            # Fallback to mid-price if 'last' is unavailable or invalid
            if price is None:
                bid = safe_decimal(ticker.get('bid'), 'bid')
                ask = safe_decimal(ticker.get('ask'), 'ask')
                if bid and ask and ask >= bid:
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price fallback: ({bid} + {ask}) / 2 = {price.normalize()}")
                elif ask:
                    price = ask
                    lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback: {price.normalize()}{RESET}")
                elif bid:
                    price = bid
                    lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback: {price.normalize()}{RESET}")

            if price:
                lg.debug(f"Current price for {symbol}: {price.normalize()}")
                return price
            else:
                lg.warning(f"No valid price found in ticker (Attempt {attempts + 1}). Ticker data: {ticker}")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5 # Wait longer for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count for rate limit
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Exchange errors are often not retryable for tickers (e.g., invalid symbol)
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None # Unexpected errors are likely fatal for this operation

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Fetches OHLCV kline data using CCXT with retries and robust processing.
    Handles Bybit V5 category parameter, validates data, converts to Decimal,
    and checks for timestamp lag. Caps request limit at API max.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        timeframe: CCXT timeframe string (e.g., '5m', '1h').
        limit: Desired number of klines. Request will be capped at BYBIT_API_KLINE_LIMIT.
        logger: Logger instance.

    Returns:
        A pandas DataFrame containing the OHLCV data with a DatetimeIndex (UTC),
        or an empty DataFrame if fetching or processing fails.
    """
    lg = logger
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    ohlcv_data: Optional[List[List[Union[int, float, str]]]] = None
    # Determine the actual limit for the API request, capped by the exchange's limit
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
        lg.debug(f"Requested limit {limit} exceeds API limit {BYBIT_API_KLINE_LIMIT}. Requesting {actual_request_limit}.")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching klines for {symbol} ({timeframe}), limit={actual_request_limit} (Attempt {attempts+1}/{MAX_API_RETRIES+1})")
            params = {}
            # Add category parameter for Bybit V5 unified/contract accounts
            if 'bybit' in exchange.id.lower():
                 try:
                     market = exchange.market(symbol)
                     # Determine category based on market type (linear, inverse, spot)
                     category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                     params['category'] = category
                     lg.debug(f"Using category '{category}' for Bybit kline fetch.")
                 except Exception as e:
                     lg.warning(f"Could not automatically determine market category for {symbol} kline fetch: {e}. Using default.")
                     # Let ccxt handle default if market lookup fails

            # Fetch OHLCV data
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit, params=params)
            received_count = len(ohlcv_data) if ohlcv_data else 0
            lg.debug(f"Received {received_count} candles (requested {actual_request_limit}).")

            # Check if API limit was hit when more data was potentially needed
            if received_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT:
                lg.warning(f"{NEON_YELLOW}Hit API kline limit ({BYBIT_API_KLINE_LIMIT}). Strategy might need more data than available in one request.{RESET}")

            if ohlcv_data and received_count > 0:
                # Validate timestamp lag of the last candle
                try:
                    last_timestamp_ms = ohlcv_data[-1][0]
                    last_dt_utc = pd.to_datetime(last_timestamp_ms, unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    # Estimate interval duration in seconds
                    interval_seconds = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') and exchange.parse_timeframe(timeframe) else 300 # Default 5min
                    # Allow lag up to 5 intervals or 5 minutes, whichever is greater
                    max_allowed_lag_seconds = max((interval_seconds * 5), 300)
                    lag_seconds = (now_utc - last_dt_utc).total_seconds()

                    if lag_seconds < max_allowed_lag_seconds:
                        lg.debug(f"Last kline timestamp {last_dt_utc} is recent (Lag: {lag_seconds:.1f}s <= {max_allowed_lag_seconds}s).")
                        break # Data looks good, exit retry loop
                    else:
                        lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_dt_utc} seems too old (Lag: {lag_seconds:.1f}s > {max_allowed_lag_seconds}s). Retrying fetch...{RESET}")
                        ohlcv_data = None # Discard stale data and retry
                except Exception as ts_err:
                    lg.warning(f"Could not validate timestamp lag: {ts_err}. Proceeding with fetched data.")
                    break # Proceed even if validation fails
            else:
                lg.warning(f"No kline data received (Attempt {attempts+1}/{MAX_API_RETRIES+1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES:
                lg.warning(f"Network error fetching klines for {symbol} ({timeframe}): {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...")
                time.sleep(RETRY_DELAY_SECONDS * (attempts + 1)) # Exponential backoff
            else:
                lg.error(f"{NEON_RED}Max retries exceeded for network errors fetching klines: {e}{RESET}")
                return pd.DataFrame()
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count for rate limit
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol} ({timeframe}): {e}{RESET}")
            # Some exchange errors might be retryable, but many (like invalid symbol) are not.
            # Consider adding checks for specific retryable error codes if needed.
            return pd.DataFrame() # Assume fatal for now
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol} ({timeframe}): {e}{RESET}", exc_info=True)
            return pd.DataFrame()

        attempts += 1
        if attempts <= MAX_API_RETRIES and ohlcv_data is None: # Only sleep if we need to retry
            time.sleep(RETRY_DELAY_SECONDS * attempts)

    if not ohlcv_data:
        lg.warning(f"Failed to fetch kline data for {symbol} ({timeframe}) after all retries.")
        return pd.DataFrame()

    # Process fetched klines into a DataFrame
    try:
        lg.debug("Processing fetched kline data into DataFrame...")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Use only the columns available in the fetched data
        df = pd.DataFrame(ohlcv_data, columns=cols[:len(ohlcv_data[0])])

        # Convert timestamp to DatetimeIndex (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True) # Drop rows with invalid timestamps
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential non-numeric values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # First convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Then convert valid numbers to Decimal, invalid stay NaN
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

        # Clean data: Drop rows with NaN in essential columns or invalid values
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive
        df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            # Ensure volume is non-negative
            df = df[df['volume'] >= Decimal('0')]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid OHLCV data.")

        if df.empty:
            lg.warning(f"Kline data for {symbol} ({timeframe}) is empty after cleaning.")
            return pd.DataFrame()

        # Sort by timestamp and trim DataFrame length if necessary
        df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN:
            lg.debug(f"DataFrame length ({len(df)}) exceeds max ({MAX_DF_LEN}). Trimming...")
            df = df.iloc[-MAX_DF_LEN:].copy() # Keep the most recent data

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} ({timeframe})")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data into DataFrame: {e}{RESET}", exc_info=True)
        return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Retrieves and validates market information (precision, limits, contract type)
    from the CCXT exchange object with retries for market loading.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing market details, or None if validation fails.
        Adds 'is_contract', 'is_linear', 'is_inverse', 'contract_type_str' keys.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            # Check if markets are loaded and contain the symbol, reload if necessary
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market info for {symbol} not found in loaded markets. Reloading markets...")
                try:
                    exchange.load_markets(reload=True)
                except Exception as load_err:
                    lg.error(f"Failed to reload markets: {load_err}")
                    # If reload fails, unlikely to find the symbol later
                    return None

            # Final check after potential reload
            if symbol not in exchange.markets:
                if attempts == 0: # Only retry once if symbol not found after reload
                    lg.warning(f"Symbol {symbol} still not found after reloading markets. Retrying check once more...")
                    attempts += 1 # Use up the retry attempt
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    lg.error(f"{NEON_RED}Market {symbol} not found on {exchange.id} even after reloading.{RESET}")
                    return None

            market = exchange.market(symbol)
            if market:
                # Add custom fields for easier contract type identification
                market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
                market['is_linear'] = market.get('linear', False) and market['is_contract']
                market['is_inverse'] = market.get('inverse', False) and market['is_contract']
                market['contract_type_str'] = "Linear" if market['is_linear'] else "Inverse" if market['is_inverse'] else "Spot" if market.get('spot') else "Unknown"

                # Log key market details for verification
                def format_decimal(value: Any) -> str:
                    try: return str(Decimal(str(value)).normalize()) if value is not None else 'N/A'
                    except: return 'Error'

                precision = market.get('precision', {})
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                lg.debug(f"Market Info Retrieved: {symbol}")
                lg.debug(f"  ID: {market.get('id', 'N/A')}, Type: {market.get('type', 'N/A')}, Contract Type: {market['contract_type_str']}")
                lg.debug(f"  Precision (Price/Amount): {format_decimal(precision.get('price'))} / {format_decimal(precision.get('amount'))}")
                lg.debug(f"  Limits Amount (Min/Max): {format_decimal(amount_limits.get('min'))} / {format_decimal(amount_limits.get('max'))}")
                lg.debug(f"  Limits Cost (Min/Max): {format_decimal(cost_limits.get('min'))} / {format_decimal(cost_limits.get('max'))}")
                lg.debug(f"  Contract Size: {format_decimal(market.get('contractSize', '1'))}") # Default to 1 for spot/non-standard

                # Critical check: Ensure necessary precision info exists for trading
                if precision.get('price') is None or precision.get('amount') is None:
                    lg.error(f"{NEON_RED}CRITICAL: Market {symbol} is missing required Price or Amount precision information! Trading may fail.{RESET}")
                    # Depending on strictness, could return None here
                return market
            else:
                # Should not happen if symbol is in exchange.markets, but check anyway
                lg.error(f"Market dictionary is None for {symbol} despite being listed.")
                return None

        except ccxt.BadSymbol as e:
            lg.error(f"Invalid symbol format or symbol not supported by {exchange.id}: {e}")
            return None # Not retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES:
                lg.warning(f"Network error getting market info for {symbol}: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...")
                time.sleep(RETRY_DELAY_SECONDS * (attempts + 1))
            else:
                lg.error(f"Max retries exceeded for network errors getting market info: {e}")
                return None
        except ccxt.ExchangeError as e:
            # General exchange errors might indicate temporary issues or config problems
            lg.error(f"Exchange error getting market info for {symbol}: {e}")
            # Could retry specific exchange errors if known to be transient
            return None # Assume fatal for now
        except Exception as e:
            lg.error(f"Unexpected error getting market info for {symbol}: {e}", exc_info=True)
            return None

        attempts += 1

    lg.error(f"Failed to retrieve market info for {symbol} after all attempts.")
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """
    Fetches the available balance for a specific currency.
    Handles Bybit V5 account types (UNIFIED, CONTRACT) and retries.

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency code (e.g., 'USDT', 'BTC').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails or currency not found.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info: Optional[Dict] = None
            balance_str: Optional[str] = None
            found: bool = False
            account_types_to_check = ['UNIFIED', 'CONTRACT'] # Bybit V5 account types

            # Try fetching balance for specific account types first (Bybit V5)
            if 'bybit' in exchange.id.lower():
                for acc_type in account_types_to_check:
                    try:
                        lg.debug(f"Fetching balance for {currency} (Account Type: {acc_type}, Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
                        params = {'accountType': acc_type}
                        info = exchange.fetch_balance(params=params)

                        # Check standard CCXT structure first
                        if currency in info and info[currency].get('free') is not None:
                            balance_str = str(info[currency]['free'])
                            lg.debug(f"Found balance in standard structure for {acc_type}: {balance_str}")
                            found = True
                            break # Found balance, no need to check raw 'info' or other types

                        # Check Bybit's raw 'info' structure as a fallback
                        elif 'info' in info and 'result' in info['info'] and isinstance(info['info']['result'].get('list'), list):
                            for account_details in info['info']['result']['list']:
                                # Check if account type matches or if type is not specified in response item
                                if (account_details.get('accountType') == acc_type or account_details.get('accountType') is None) \
                                   and isinstance(account_details.get('coin'), list):
                                    for coin_data in account_details['coin']:
                                        if coin_data.get('coin') == currency:
                                            # Try different keys for available balance
                                            free_balance = coin_data.get('availableToWithdraw') or \
                                                           coin_data.get('availableBalance') or \
                                                           coin_data.get('walletBalance') # Less ideal fallback
                                            if free_balance is not None:
                                                balance_str = str(free_balance)
                                                lg.debug(f"Found balance in raw 'info' structure for {acc_type}: {balance_str}")
                                                found = True
                                                break # Found coin balance
                                    if found: break # Found in this account type's list
                            if found: break # Found balance, exit account type loop
                    except ccxt.ExchangeError as e:
                         # Ignore errors like "account type not supported" and try next type
                         lg.debug(f"Exchange error fetching balance for type {acc_type}: {e}. Trying next type...")
                    except Exception as e:
                         lg.warning(f"Unexpected error fetching balance for type {acc_type}: {e}. Trying next type...")

            # If not found in specific types or not Bybit, try default fetch_balance
            if not found:
                try:
                    lg.debug(f"Fetching default balance for {currency} (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
                    info = exchange.fetch_balance()
                    # Check standard structure
                    if currency in info and info[currency].get('free') is not None:
                        balance_str = str(info[currency]['free'])
                        lg.debug(f"Found balance in default fetch (standard structure): {balance_str}")
                        found = True
                    # Check raw structure as fallback (e.g., for exchanges not fully conforming)
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
                                             lg.debug(f"Found balance in default fetch (raw structure): {balance_str}")
                                             found = True
                                             break
                                 if found: break
                             if found: break
                except Exception as e:
                    lg.error(f"Failed default balance fetch: {e}", exc_info=True)

            # Process the found balance string
            if found and balance_str is not None:
                try:
                    balance_decimal = Decimal(balance_str)
                    # Ensure balance is not negative
                    return balance_decimal if balance_decimal >= Decimal('0') else Decimal('0')
                except (InvalidOperation, ValueError, TypeError) as conv_err:
                    # Raise as ExchangeError for retry logic
                    raise ccxt.ExchangeError(f"Failed to convert balance string '{balance_str}' for {currency} to Decimal: {conv_err}")
            else:
                # Raise as ExchangeError if currency wasn't found after checks
                raise ccxt.ExchangeError(f"Balance for currency '{currency}' not found in fetch_balance response.")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.AuthenticationError as e:
            # Authentication errors are critical and not retryable
            lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API Key/Secret/Permissions.{RESET}")
            return None
        except ccxt.ExchangeError as e:
            # Log specific exchange errors and decide whether to retry
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Could add checks here for specific non-retryable error codes from the exchange
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            # Unexpected errors might be retryable, but log as error

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            # Use exponential backoff for retries
            time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

# --- Position & Order Management ---
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Handles Bybit V5 parameters (category, symbol) and parses position details.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        logger: Logger instance.

    Returns:
        A dictionary containing details of the open position if found, otherwise None.
        Adds 'size_decimal' (Decimal) and potentially standardizes other fields.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for {symbol} (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            positions: List[Dict] = []
            market_id: Optional[str] = None
            category: Optional[str] = None

            try:
                # Get market details to determine category and ID for Bybit V5
                market = exchange.market(symbol)
                market_id = market['id']
                category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                # Bybit V5 requires category and symbol for specific position fetch
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetching specific position using params: {params}")
                positions = exchange.fetch_positions([symbol], params=params)
            except ccxt.ArgumentsRequired as e:
                 # Fallback if specific symbol fetch isn't supported or fails
                 lg.warning(f"Specific symbol fetch failed ({e}). Fetching all positions for category '{category}' (slower).")
                 params = {'category': category or 'linear'} # Default to linear if category unknown
                 all_positions = exchange.fetch_positions(params=params)
                 # Filter for the desired symbol using both standard 'symbol' and raw 'info' symbol ID
                 positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
            except ccxt.ExchangeError as e:
                 # Handle specific Bybit error code for "position not found" gracefully
                 if hasattr(e, 'code') and e.code == 110025 or "position not found" in str(e).lower():
                     lg.info(f"No open position found for {symbol} via API ({e}).")
                     return None
                 else:
                     # Re-raise other exchange errors to be caught by the outer retry loop
                     raise e

            active_position: Optional[Dict] = None
            # Define a small threshold to consider a position 'open' (based on amount precision if possible)
            size_threshold = Decimal('1e-9') # Default tiny threshold
            try:
                amount_precision = exchange.market(symbol)['precision']['amount']
                if amount_precision:
                    # Use a fraction of the smallest possible amount step as threshold
                    size_threshold = Decimal(str(amount_precision)) * Decimal('0.1')
            except Exception:
                lg.warning(f"Could not get amount precision for {symbol}. Using default size threshold {size_threshold}.")
            lg.debug(f"Using position size threshold: {size_threshold}")

            # Iterate through fetched positions to find an active one
            for pos in positions:
                # Get size from 'info' (more reliable for Bybit) or standard 'contracts' field
                size_str = str(pos.get('info', {}).get('size', pos.get('contracts', '')))
                if not size_str:
                    lg.debug(f"Skipping position entry with missing size: {pos.get('info', {})}")
                    continue

                # --- Safely parse and check position size ---
                try:
                    size = Decimal(size_str)
                    # Check if the absolute size is greater than the threshold
                    if abs(size) > size_threshold:
                         lg.debug(f"Found potentially active position entry: Size={size}, Threshold={size_threshold}")
                         active_position = pos
                         break # Found an active position, stop searching
                    else:
                         lg.debug(f"Position entry size {size} <= threshold {size_threshold}. Ignoring.")
                except (ValueError, InvalidOperation, TypeError) as parse_err:
                     # Log specific error if size string cannot be parsed
                     lg.warning(f"Could not parse/check position size string '{size_str}': {parse_err}. Skipping this position entry.")
                     continue # Skip to the next position entry
                # --- End safe parsing ---

            # If an active position was found, format and return it
            if active_position:
                # Create a copy to avoid modifying the original CCXT response
                standardized_pos = active_position.copy()
                info = standardized_pos.get('info', {}) # Raw exchange response

                # Add parsed Decimal size
                standardized_pos['size_decimal'] = size # Use the successfully parsed size from the loop

                # Standardize side ('long' or 'short')
                side = standardized_pos.get('side')
                if side not in ['long', 'short']:
                    # Infer side from Bybit V5 'info' or size sign
                    side_v5 = info.get('side', '').lower() # 'Buy' or 'Sell'
                    if side_v5 == 'buy': side = 'long'
                    elif side_v5 == 'sell': side = 'short'
                    elif size > size_threshold: side = 'long'
                    elif size < -size_threshold: side = 'short'
                    else: side = None # Cannot determine side
                if not side:
                    lg.warning(f"Could not determine position side for {symbol}. Position info: {info}")
                    return None # Cannot proceed without side
                standardized_pos['side'] = side

                # Standardize other common fields, preferring CCXT standard fields but falling back to 'info'
                standardized_pos['entryPrice'] = standardized_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice')
                standardized_pos['leverage'] = standardized_pos.get('leverage') or info.get('leverage')
                standardized_pos['liquidationPrice'] = standardized_pos.get('liquidationPrice') or info.get('liqPrice')
                standardized_pos['unrealizedPnl'] = standardized_pos.get('unrealizedPnl') or info.get('unrealisedPnl') # Note Bybit spelling

                # Standardize SL/TP/TSL fields (ensure they are strings if present)
                sl = info.get('stopLoss') or standardized_pos.get('stopLossPrice')
                tp = info.get('takeProfit') or standardized_pos.get('takeProfitPrice')
                tsl_dist = info.get('trailingStop') # Bybit V5 TSL distance
                tsl_act = info.get('activePrice') # Bybit V5 TSL activation price
                if sl is not None and str(sl) != '0': standardized_pos['stopLossPrice'] = str(sl)
                if tp is not None and str(tp) != '0': standardized_pos['takeProfitPrice'] = str(tp)
                if tsl_dist is not None and str(tsl_dist) != '0': standardized_pos['trailingStopLoss'] = str(tsl_dist)
                if tsl_act is not None and str(tsl_act) != '0': standardized_pos['tslActivationPrice'] = str(tsl_act)

                # Log the found active position details
                ep_str = str(Decimal(str(standardized_pos['entryPrice'])).normalize()) if standardized_pos.get('entryPrice') else 'N/A'
                size_str_log = standardized_pos['size_decimal'].normalize()
                sl_str_log = str(standardized_pos.get('stopLossPrice', 'N/A'))
                tp_str_log = str(standardized_pos.get('takeProfitPrice', 'N/A'))
                tsl_str_log = f"D:{standardized_pos.get('trailingStopLoss','N/A')}/A:{standardized_pos.get('tslActivationPrice','N/A')}" if standardized_pos.get('trailingStopLoss') else "N/A"

                lg.info(f"{NEON_GREEN}Active {side.upper()} Position Found ({symbol}): Size={size_str_log}, Entry={ep_str}, SL={sl_str_log}, TP={tp_str_log}, TSL={tsl_str_log}{RESET}")
                return standardized_pos
            else:
                # No position with size > threshold found
                lg.info(f"No active position found for {symbol} (checked {len(positions)} entries).")
                return None

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching positions: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching positions: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue # Don't increment attempt count
        except ccxt.AuthenticationError as e:
            lg.critical(f"{NEON_RED}Authentication Error fetching positions: {e}. Check API permissions. Stopping.{RESET}")
            return None # Fatal
        except ccxt.ExchangeError as e:
            # Log other exchange errors and retry
            lg.warning(f"{NEON_YELLOW}Exchange error fetching positions: {e}. Retry {attempts + 1}/{MAX_API_RETRIES + 1}...{RESET}")
            # Add checks for specific fatal error codes if necessary
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching positions: {e}{RESET}", exc_info=True)
            # Unexpected errors might be retryable, but log as error

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """
    Sets leverage for a derivatives symbol using CCXT's set_leverage method.
    Includes Bybit V5 specific parameters and handles common success/failure responses.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        leverage: Desired leverage (integer > 0).
        market_info: Dictionary containing market details (used for ID and type).
        logger: Logger instance.

    Returns:
        True if leverage was set successfully or was already set correctly, False otherwise.
    """
    lg = logger
    is_contract = market_info.get('is_contract', False)

    # Skip if not a contract market
    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True
    # Validate leverage input
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol} (Invalid leverage value: {leverage}).")
        return False
    # Check if the exchange supports setting leverage
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage via CCXT.")
        return False

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {symbol} to {leverage}x (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            params = {}
            market_id = market_info['id'] # Use the exchange-specific market ID

            # Add Bybit V5 specific parameters (category, buy/sell leverage)
            if 'bybit' in exchange.id.lower():
                 # Assume linear preference if both are somehow true, else inverse
                 category = 'linear' if market_info.get('linear', True) else 'inverse'
                 # Bybit requires buyLeverage and sellLeverage for unified margin
                 params = {
                     'category': category,
                     'symbol': market_id,
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 params for setLeverage: {params}")

            # Call CCXT's set_leverage method
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"Set leverage raw response: {response}")

            # Check Bybit V5 response code if available
            ret_code = response.get('retCode') if isinstance(response, dict) else None
            if ret_code is not None:
                 if ret_code == 0:
                     lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (Bybit Code 0).{RESET}")
                     return True
                 elif ret_code == 110045: # Bybit: "Leverage not modified"
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Bybit Code 110045).{RESET}")
                     return True
                 else:
                     # Raise an error for other non-zero Bybit codes
                     error_message = response.get('retMsg', 'Unknown Bybit API error')
                     raise ccxt.ExchangeError(f"Bybit API error setting leverage: {error_message} (Code: {ret_code})")
            else:
                # If no retCode, assume success (standard CCXT behavior)
                lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (No specific code in response).{RESET}")
                return True

        except ccxt.ExchangeError as e:
            # Handle specific exchange errors, especially those indicating success or non-retryable issues
            error_code = getattr(e, 'code', None) # Bybit error code might be in the exception
            error_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage: {e} (Code: {error_code}){RESET}")
            # Check for "Leverage not modified" or similar messages/codes
            if error_code == 110045 or "not modified" in error_str or "leverage is same" in error_str:
                lg.info(f"{NEON_YELLOW}Leverage already set correctly (inferred from error).{RESET}")
                return True
            # List known fatal/non-retryable error codes or strings for leverage setting
            fatal_codes = [110028, 110009, 110055, 110043, 110044, 110013, 10001, 10004] # Example Bybit codes
            fatal_strings = ["margin mode", "position exists", "risk limit", "parameter error", "invalid leverage"]
            if error_code in fatal_codes or any(s in error_str for s in fatal_strings):
                lg.error(" >> Hint: This leverage error seems non-retryable. Check position status, margin mode, or risk limits.")
                return False
            elif attempts >= MAX_API_RETRIES:
                 lg.error("Max retries exceeded for ExchangeError setting leverage.")
                 return False
            # Otherwise, assume retryable
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
            return False # Unexpected errors likely fatal

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False

def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: Dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """
    Calculates the appropriate position size based on risk percentage, stop loss distance,
    account balance, and market constraints (precision, limits, contract type).

    Args:
        balance: Available balance in the quote currency (Decimal).
        risk_per_trade: Fraction of balance to risk (e.g., 0.01 for 1%).
        initial_stop_loss_price: Calculated stop loss price (Decimal).
        entry_price: Estimated entry price (Decimal).
        market_info: Dictionary containing market details from get_market_info.
        exchange: Initialized CCXT exchange object (for formatting).
        logger: Logger instance.

    Returns:
        The calculated position size as a Decimal, adjusted for market rules,
        or None if calculation is not possible or results in an invalid size.
    """
    lg = logger
    symbol = market_info['symbol']
    quote_currency = market_info['quote']
    base_currency = market_info['base']
    is_contract = market_info['is_contract']
    is_inverse = market_info.get('is_inverse', False)
    # Determine the unit for size (Contracts for derivatives, Base currency for spot)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Initial Validations ---
    if balance <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Invalid balance ({balance} {quote_currency}).")
        return None
    risk_decimal = Decimal(str(risk_per_trade))
    if not (Decimal('0') < risk_decimal <= Decimal('1')):
        lg.error(f"Position sizing failed for {symbol}: Invalid risk_per_trade ({risk_per_trade}). Must be > 0 and <= 1.")
        return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Entry price ({entry_price}) or SL price ({initial_stop_loss_price}) is zero or negative.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed for {symbol}: Entry price and Stop Loss price cannot be the same ({entry_price}).")
        return None

    # --- Extract Market Details ---
    try:
        precision = market_info['precision']
        limits = market_info['limits']
        amount_precision_str = precision.get('amount')
        price_precision_str = precision.get('price') # Used for logging clarity
        if amount_precision_str is None: raise ValueError("Amount precision missing")
        if price_precision_str is None: raise ValueError("Price precision missing")

        # Smallest amount step (tick size for amount)
        amount_tick_size = Decimal(str(amount_precision_str))
        if amount_tick_size <= 0: raise ValueError(f"Invalid amount precision step: {amount_tick_size}")

        # Amount and Cost Limits
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        min_amount = Decimal(str(amount_limits.get('min', '0')))
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')

        # Contract size (usually 1 for linear/spot, can be different for inverse)
        contract_size_str = market_info.get('contractSize', '1')
        contract_size = Decimal(str(contract_size_str)) if contract_size_str else Decimal('1')
        if contract_size <= 0: raise ValueError(f"Invalid contract size: {contract_size}")

    except (KeyError, ValueError, TypeError, InvalidOperation) as e:
        lg.error(f"Position sizing failed for {symbol}: Error extracting market details: {e}")
        return None

    # --- Calculate Risk Amount and Stop Loss Distance ---
    risk_amount_quote = balance * risk_decimal
    stop_loss_distance_price = abs(entry_price - initial_stop_loss_price)
    if stop_loss_distance_price <= Decimal('0'):
        # Should be caught earlier, but double-check
        lg.error(f"Position sizing failed for {symbol}: Stop loss distance is zero or negative.")
        return None

    lg.info(f"Calculating Position Size for {symbol}:")
    lg.info(f"  Balance: {balance.normalize()} {quote_currency}, Risk: {risk_decimal:.2%}, Max Risk Amount: {risk_amount_quote.normalize()} {quote_currency}")
    lg.info(f"  Entry: {entry_price.normalize()}, SL: {initial_stop_loss_price.normalize()}, SL Price Distance: {stop_loss_distance_price.normalize()}")
    lg.info(f"  Contract Type: {market_info['contract_type_str']}, Contract Size: {contract_size.normalize()}")
    lg.info(f"  Amount Precision: {amount_tick_size}, Min/Max Amount: {min_amount}/{max_amount}, Min/Max Cost: {min_cost}/{max_cost}")

    # --- Calculate Initial Position Size Based on Risk ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse: # Linear Contracts or Spot
            # Risk per unit = price change per contract/coin
            # For standard linear contracts/spot, contractSize is often 1 (base currency unit)
            # Risk per unit = stop_loss_distance_price * contract_size (in quote currency)
            value_change_per_unit = stop_loss_distance_price * contract_size
            if value_change_per_unit <= Decimal('1e-18'): # Use small threshold instead of zero
                 lg.error(f"Position sizing failed (Linear/Spot): Calculated value change per unit is too small ({value_change_per_unit}).")
                 return None
            calculated_size = risk_amount_quote / value_change_per_unit
            lg.debug(f"  Linear/Spot Risk Calc: Size = {risk_amount_quote} / ({stop_loss_distance_price} * {contract_size}) = {calculated_size}")
        else: # Inverse Contracts
            # Risk per contract = contract_size * abs(1/entry - 1/sl) (in quote currency)
            if entry_price <= 0 or initial_stop_loss_price <= 0:
                 lg.error("Position sizing failed (Inverse): Entry or SL price is zero or negative.")
                 return None
            inverse_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
            if inverse_factor <= Decimal('1e-18'):
                 lg.error(f"Position sizing failed (Inverse): Calculated inverse factor is too small ({inverse_factor}).")
                 return None
            risk_per_contract = contract_size * inverse_factor
            if risk_per_contract <= Decimal('1e-18'):
                 lg.error(f"Position sizing failed (Inverse): Calculated risk per contract is too small ({risk_per_contract}).")
                 return None
            calculated_size = risk_amount_quote / risk_per_contract
            lg.debug(f"  Inverse Risk Calc: Size = {risk_amount_quote} / ({contract_size} * {inverse_factor}) = {calculated_size}")

    except (OverflowError, InvalidOperation, ZeroDivisionError) as calc_err:
        lg.error(f"Position sizing failed during initial calculation: {calc_err}.")
        return None

    if calculated_size <= 0:
        lg.error(f"Position sizing failed: Initial calculated size is zero or negative ({calculated_size}). Check inputs and risk settings.")
        return None

    lg.info(f"  Initial Calculated Size = {calculated_size.normalize()} {size_unit}")

    # --- Adjust Size for Market Limits (Amount) ---
    adjusted_size = calculated_size
    if min_amount > 0 and adjusted_size < min_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size} is below minimum amount {min_amount}. Adjusting UP to minimum.{RESET}")
        adjusted_size = min_amount
    if max_amount < Decimal('inf') and adjusted_size > max_amount:
        lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size} exceeds maximum amount {max_amount}. Adjusting DOWN to maximum.{RESET}")
        adjusted_size = max_amount
    if adjusted_size != calculated_size:
        lg.debug(f"  Size after Amount Limits: {adjusted_size.normalize()} {size_unit}")

    # --- Adjust Size for Market Limits (Cost) ---
    estimated_cost = Decimal('0')
    cost_adjustment_applied = False
    try:
        if entry_price > 0:
            # Cost = Size * Price * ContractSize (Linear/Spot)
            # Cost = Size * ContractSize / Price (Inverse)
            estimated_cost = (adjusted_size * entry_price * contract_size) if not is_inverse else ((adjusted_size * contract_size) / entry_price)
        lg.debug(f"  Estimated Cost for size {adjusted_size.normalize()}: {estimated_cost.normalize()} {quote_currency}")

        # Check Minimum Cost
        if min_cost > 0 and estimated_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost} is below minimum cost {min_cost}. Attempting to increase size.{RESET}")
            required_size_for_min_cost = None
            try:
                # Calculate the size needed to meet the minimum cost
                if not is_inverse:
                    if entry_price > 0 and contract_size > 0:
                         required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else: # Inverse
                    if contract_size > 0:
                         required_size_for_min_cost = (min_cost * entry_price) / contract_size

                if required_size_for_min_cost is None or required_size_for_min_cost <= 0:
                    raise ValueError("Could not calculate required size for min cost.")

                lg.info(f"  Required size to meet min cost: {required_size_for_min_cost.normalize()} {size_unit}")
                # Ensure the required size doesn't violate max amount limit
                if max_amount < Decimal('inf') and required_size_for_min_cost > max_amount:
                    lg.error(f"{NEON_RED}Cannot meet minimum cost ({min_cost}) without exceeding maximum amount limit ({max_amount}). Aborting trade.{RESET}")
                    return None
                # Adjust size up, but ensure it's at least the minimum amount limit
                adjusted_size = max(min_amount, required_size_for_min_cost)
                cost_adjustment_applied = True
            except (ValueError, OverflowError, InvalidOperation, ZeroDivisionError) as min_cost_err:
                lg.error(f"{NEON_RED}Failed to calculate required size for minimum cost ({min_cost}): {min_cost_err}. Aborting trade.{RESET}")
                return None

        # Check Maximum Cost
        elif max_cost < Decimal('inf') and estimated_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost} exceeds maximum cost {max_cost}. Attempting to decrease size.{RESET}")
            max_size_for_max_cost = None
            try:
                # Calculate the maximum size allowed by the maximum cost
                if not is_inverse:
                    if entry_price > 0 and contract_size > 0:
                         max_size_for_max_cost = max_cost / (entry_price * contract_size)
                else: # Inverse
                     if contract_size > 0:
                         max_size_for_max_cost = (max_cost * entry_price) / contract_size

                if max_size_for_max_cost is None or max_size_for_max_cost <= 0:
                     raise ValueError("Could not calculate max size for max cost.")

                lg.info(f"  Maximum size allowed by max cost: {max_size_for_max_cost.normalize()} {size_unit}")
                # Adjust size down, ensuring it meets min amount limit and doesn't exceed original adjusted size
                adjusted_size = max(min_amount, min(adjusted_size, max_size_for_max_cost))
                cost_adjustment_applied = True
            except (ValueError, OverflowError, InvalidOperation, ZeroDivisionError) as max_cost_err:
                 lg.error(f"{NEON_RED}Failed to calculate maximum size for maximum cost ({max_cost}): {max_cost_err}. Aborting trade.{RESET}")
                 return None

        if cost_adjustment_applied:
            lg.info(f"  Size after Cost Limits: {adjusted_size.normalize()} {size_unit}")

    except (OverflowError, InvalidOperation, ZeroDivisionError) as cost_calc_err:
         lg.error(f"Error during cost calculation/adjustment: {cost_calc_err}. Proceeding without cost adjustment.")

    # --- Apply Amount Precision (Tick Size) ---
    final_size = adjusted_size
    try:
        # Use CCXT's amount_to_precision for reliable formatting
        formatted_amount_str = exchange.amount_to_precision(symbol, float(adjusted_size))
        final_size = Decimal(formatted_amount_str)
        if final_size != adjusted_size:
             lg.info(f"Applied amount precision (CCXT): {adjusted_size} -> {final_size}")
    except (ccxt.ExchangeError, ValueError, TypeError) as fmt_err:
        lg.warning(f"CCXT amount_to_precision failed: {fmt_err}. Attempting manual rounding.")
        # Manual rounding down using the amount tick size as a fallback
        try:
            if amount_tick_size > 0:
                # Floor division to get the number of ticks, then multiply back
                final_size = (adjusted_size // amount_tick_size) * amount_tick_size
                if final_size != adjusted_size:
                    lg.info(f"Applied manual amount precision (floor): {adjusted_size} -> {final_size}")
            else:
                # Should not happen due to earlier checks, but handle defensively
                lg.error("Amount tick size is zero during manual precision step. Using unrounded size.")
                final_size = adjusted_size
        except (InvalidOperation, TypeError) as manual_err:
            lg.error(f"Manual precision rounding failed: {manual_err}. Using unrounded size: {adjusted_size}")
            final_size = adjusted_size

    # --- Final Validation Checks ---
    if final_size <= 0:
        lg.error(f"{NEON_RED}Position sizing failed: Final size is zero or negative ({final_size}) after adjustments and precision. Aborting trade.{RESET}")
        return None
    if min_amount > 0 and final_size < min_amount:
        lg.error(f"{NEON_RED}Position sizing failed: Final size {final_size} is below minimum amount {min_amount} after precision adjustments. Aborting trade.{RESET}")
        # Consider attempting to bump up to min_amount if feasible, but safer to abort.
        return None

    # Re-check cost limits with the final precise size
    final_cost = Decimal('0')
    try:
        if entry_price > 0:
            final_cost = (final_size * entry_price * contract_size) if not is_inverse else ((final_size * contract_size) / entry_price)
        lg.debug(f"  Final Cost for size {final_size.normalize()}: {final_cost.normalize()} {quote_currency}")

        if min_cost > 0 and final_cost < min_cost:
            lg.debug(f"Final cost {final_cost} is slightly below min cost {min_cost} after precision.")
            # Try bumping size by one tick to meet min cost
            try:
                next_step_size = final_size + amount_tick_size
                next_step_cost = Decimal('0')
                if entry_price > 0:
                    next_step_cost = (next_step_size * entry_price * contract_size) if not is_inverse else ((next_step_size * contract_size) / entry_price)

                # Check if the next step size is valid
                is_valid_next_step = (next_step_cost >= min_cost) and \
                                     (max_amount == Decimal('inf') or next_step_size <= max_amount) and \
                                     (max_cost == Decimal('inf') or next_step_cost <= max_cost)

                if is_valid_next_step:
                    lg.warning(f"{NEON_YELLOW}Final cost was below minimum. Bumping size by one step to {next_step_size.normalize()} to meet min cost.{RESET}")
                    final_size = next_step_size
                else:
                    lg.error(f"{NEON_RED}Final cost {final_cost} is below minimum {min_cost}, but increasing size by one step to {next_step_size} would violate other limits (Cost: {next_step_cost}, MaxAmt: {max_amount}, MaxCost: {max_cost}). Aborting trade.{RESET}")
                    return None
            except Exception as bump_err:
                lg.error(f"{NEON_RED}Error attempting to bump size for minimum cost: {bump_err}. Aborting trade.{RESET}")
                return None
        elif max_cost < Decimal('inf') and final_cost > max_cost:
             # This shouldn't happen if previous checks were correct, but double-check
             lg.error(f"{NEON_RED}Position sizing failed: Final cost {final_cost} exceeds maximum cost {max_cost} after precision adjustments. Aborting trade.{RESET}")
             return None

    except (OverflowError, InvalidOperation, ZeroDivisionError) as final_cost_err:
         lg.error(f"Error during final cost check: {final_cost_err}. Proceeding with calculated size, but cost limits might be violated.")


    lg.info(f"{NEON_GREEN}{BRIGHT}Final Calculated Position Size: {final_size.normalize()} {size_unit}{RESET}")
    return final_size

def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: Dict,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """
    Places a market order using CCXT with retries and Bybit V5 specific parameters.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        trade_signal: The signal ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT").
        position_size: The size of the order (Decimal, always positive).
        market_info: Dictionary containing market details.
        logger: Logger instance.
        reduce_only: If True, set the reduceOnly flag (for closing positions).
        params: Additional parameters to pass to create_order (optional).

    Returns:
        The order dictionary returned by CCXT if successful, otherwise None.
    """
    lg = logger
    # Map signals to CCXT sides ('buy' or 'sell')
    side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}
    side = side_map.get(trade_signal.upper())

    if side is None:
        lg.error(f"Invalid trade signal '{trade_signal}' provided to place_trade.")
        return None
    if position_size <= 0:
        lg.error(f"Invalid position size ({position_size}) provided to place_trade. Size must be positive.")
        return None

    order_type = 'market'
    is_contract = market_info['is_contract']
    base_currency = market_info['base']
    # Determine unit for logging/context
    size_unit = market_info.get('settle', base_currency) if is_contract else base_currency
    action_desc = "Close/Reduce" if reduce_only else "Open/Increase"
    market_id = market_info['id'] # Use exchange-specific ID

    # Basic order arguments for CCXT create_order
    order_args = {
        'symbol': market_id,
        'type': order_type,
        'side': side,
        'amount': float(position_size) # CCXT generally expects float amount
    }

    # Add exchange-specific parameters, especially for Bybit V5
    order_params = {}
    if 'bybit' in exchange.id.lower():
        category = 'linear' if market_info.get('linear', True) else 'inverse'
        # Common Bybit V5 parameters
        order_params = {
            'category': category,
            'positionIdx': 0  # Assume one-way mode (0 for unified/classic)
        }
        if reduce_only:
            order_params['reduceOnly'] = True
            # Use IOC for reduceOnly market orders to avoid resting orders if partially filled
            order_params['timeInForce'] = 'IOC' # ImmediateOrCancel
        lg.debug(f"Using Bybit V5 params for order: {order_params}")

    # Merge any additional custom params provided
    if params:
        order_params.update(params)

    # Add the combined parameters dict to the main order arguments
    if order_params:
        order_args['params'] = order_params

    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type} order:")
    lg.info(f"  Symbol: {symbol} ({market_id})")
    lg.info(f"  Size: {position_size.normalize()} {size_unit}")
    if order_args.get('params'): lg.debug(f"  Full Order Params: {order_args['params']}")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            order = exchange.create_order(**order_args)
            lg.debug(f"Create order raw response: {order}")

            # Log success with key order details
            order_id = order.get('id', 'N/A')
            status = order.get('status', 'N/A')
            avg_price_str = str(order.get('average', ''))
            filled_amount_str = str(order.get('filled', ''))

            avg_price_log = f", AvgPrice: ~{Decimal(avg_price_str).normalize()}" if avg_price_str else ""
            filled_amount_log = f", Filled: {Decimal(filled_amount_str).normalize()}" if filled_amount_str else ""

            log_msg = f"{NEON_GREEN}{action_desc} Order Placed!{RESET} ID: {order_id}, Status: {status}{avg_price_log}{filled_amount_log}"
            lg.info(log_msg)
            return order # Success

        # --- Specific CCXT Exception Handling ---
        except ccxt.InsufficientFunds as e:
            # Typically not retryable without balance change
            lg.error(f"{NEON_RED}Insufficient funds to place {side} order for {position_size} {symbol}: {e}{RESET}")
            return None
        except ccxt.InvalidOrder as e:
            # Often indicates issues with parameters, size, price, or market state. Usually not retryable.
            lg.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
            lg.error(f"  Order Args Used: {order_args}")
            # Add hints based on common InvalidOrder reasons if possible
            if "size" in str(e).lower(): lg.error("  >> Hint: Check position size against market limits (min/max amount) and precision.")
            if "cost" in str(e).lower(): lg.error("  >> Hint: Check order cost against market limits (min/max cost).")
            if "price" in str(e).lower(): lg.error("  >> Hint: Check price limits or precision (less likely for market orders).")
            return None
        except ccxt.ExchangeError as e:
            # Handle general exchange errors, check for fatal codes
            error_code = getattr(e, 'code', None) # Bybit specific code might be here
            lg.error(f"{NEON_RED}Exchange error placing order: {e} (Code: {error_code}){RESET}")
            # List known fatal error codes for order placement (e.g., account issues, market closed)
            fatal_codes = [110014, 110007, 110040, 110013, 110025, 30086, 10001] # Example Bybit codes
            if error_code in fatal_codes or "position side does not match" in str(e).lower():
                lg.error(" >> Hint: This exchange error seems non-retryable for order placement.")
                return None
            # Otherwise, assume potentially retryable
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
            return None # Assume fatal

        # Increment attempt count only if it wasn't a rate limit pause
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
    using Bybit V5's specific private endpoint (`/v5/position/set-trading-stop`).

    Note: This function directly uses a private API call, bypassing standard CCXT methods
          for SL/TP/TSL due to limitations or complexities in unified CCXT handling for Bybit V5 protections.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol (e.g., 'BTC/USDT:USDT').
        market_info: Dictionary containing market details.
        position_info: Dictionary containing details of the current open position (from get_open_position).
        logger: Logger instance.
        stop_loss_price: Desired fixed stop loss price (Decimal). Set to 0 to remove.
        take_profit_price: Desired fixed take profit price (Decimal). Set to 0 to remove.
        trailing_stop_distance: Desired trailing stop distance (Decimal, positive value). Requires tsl_activation_price. Set to 0 to remove TSL.
        tsl_activation_price: Price at which the trailing stop should activate (Decimal). Required if trailing_stop_distance > 0.

    Returns:
        True if the protection was set successfully or no changes were needed, False otherwise.
    """
    lg = logger

    # --- Input Validation ---
    if not market_info.get('is_contract'):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract market).")
        return False
    if not position_info:
        lg.error(f"Protection setting failed for {symbol}: Missing current position information.")
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
    params_to_set: Dict[str, str] = {} # Parameters for the API call
    log_parts: List[str] = [f"Preparing protection request for {symbol} ({position_side.upper()} @ {entry_price.normalize()}):"]
    any_protection_requested: bool = False # Track if any SL/TP/TSL value was provided

    try:
        price_precision_str = market_info['precision']['price']
        min_price_tick = Decimal(str(price_precision_str))
        if min_price_tick <= 0: raise ValueError("Invalid price precision")

        # Helper to format price values according to market precision
        def format_price_param(price_decimal: Optional[Decimal], param_name: str) -> Optional[str]:
            """Formats price to string using exchange precision rules."""
            if price_decimal is None: return None
            # Allow explicit clearing of SL/TP/TSL by passing 0
            if price_decimal == Decimal('0'): return "0"
            if price_decimal < 0:
                lg.warning(f"Attempted to set negative price ({price_decimal}) for {param_name}. Ignoring.")
                return None
            try:
                # Use CCXT's price_to_precision for correct rounding/formatting
                # Use ROUND for SL/TP/TSL to avoid overly conservative placement
                formatted_price = exchange.price_to_precision(symbol, float(price_decimal), exchange.ROUND)
                # Ensure formatted price is still positive after rounding
                if Decimal(formatted_price) > 0:
                    return formatted_price
                else:
                    lg.warning(f"Formatted {param_name} price '{formatted_price}' became non-positive. Ignoring.")
                    return None
            except (ccxt.ExchangeError, ValueError, TypeError) as e:
                lg.error(f"Failed to format {param_name} price {price_decimal} using exchange precision: {e}. Ignoring parameter.")
                return None

        # --- Trailing Stop Loss ---
        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal):
            any_protection_requested = True
            if trailing_stop_distance > 0:
                # TSL requires a positive distance and a valid activation price
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0:
                    lg.error(f"Trailing Stop ignored: Invalid or missing activation price ({tsl_activation_price}) for distance {trailing_stop_distance}.")
                else:
                    # Validate activation price is beyond entry price
                    is_valid_activation = (position_side == 'long' and tsl_activation_price > entry_price) or \
                                          (position_side == 'short' and tsl_activation_price < entry_price)
                    if not is_valid_activation:
                        lg.error(f"Trailing Stop ignored: Activation price {tsl_activation_price} is not beyond entry price {entry_price} for {position_side} position.")
                    else:
                        # Ensure distance is at least one price tick
                        min_tsl_distance = max(trailing_stop_distance, min_price_tick)
                        fmt_tsl_distance = format_price_param(min_tsl_distance, "Trailing Stop Distance")
                        fmt_tsl_activation = format_price_param(tsl_activation_price, "TSL Activation Price")

                        if fmt_tsl_distance and fmt_tsl_activation:
                            params_to_set['trailingStop'] = fmt_tsl_distance
                            params_to_set['activePrice'] = fmt_tsl_activation
                            log_parts.append(f"  - Set Trailing Stop: Distance={fmt_tsl_distance}, Activation={fmt_tsl_activation}")
                            set_tsl = True # Flag that TSL is being set (overrides fixed SL)
                        else:
                            lg.error(f"Failed to format Trailing Stop parameters (Distance: {fmt_tsl_distance}, Activation: {fmt_tsl_activation}). TSL not set.")
            elif trailing_stop_distance == Decimal('0'):
                # Explicitly clear TSL
                params_to_set['trailingStop'] = "0"
                # Bybit docs suggest setting activePrice to 0 as well when clearing TSL
                params_to_set['activePrice'] = "0"
                log_parts.append("  - Clear Trailing Stop")
                set_tsl = True # Still counts as a TSL action (clearing)

        # --- Fixed Stop Loss (Only if TSL is not being set/cleared) ---
        if not set_tsl and isinstance(stop_loss_price, Decimal):
            any_protection_requested = True
            if stop_loss_price > 0:
                # Validate SL price is on the correct side of entry price
                is_valid_sl = (position_side == 'long' and stop_loss_price < entry_price) or \
                              (position_side == 'short' and stop_loss_price > entry_price)
                if not is_valid_sl:
                    lg.error(f"Fixed Stop Loss ignored: SL price {stop_loss_price} is not on the correct side of entry price {entry_price} for {position_side} position.")
                else:
                    fmt_sl = format_price_param(stop_loss_price, "Stop Loss")
                    if fmt_sl:
                        params_to_set['stopLoss'] = fmt_sl
                        log_parts.append(f"  - Set Fixed Stop Loss: {fmt_sl}")
                    else:
                        lg.error(f"Failed to format Fixed Stop Loss price {stop_loss_price}. SL not set.")
            elif stop_loss_price == Decimal('0'):
                # Explicitly clear fixed SL
                params_to_set['stopLoss'] = "0"
                log_parts.append("  - Clear Fixed Stop Loss")

        # --- Fixed Take Profit ---
        if isinstance(take_profit_price, Decimal):
            any_protection_requested = True
            if take_profit_price > 0:
                # Validate TP price is on the correct side of entry price
                is_valid_tp = (position_side == 'long' and take_profit_price > entry_price) or \
                              (position_side == 'short' and take_profit_price < entry_price)
                if not is_valid_tp:
                    lg.error(f"Take Profit ignored: TP price {take_profit_price} is not on the correct side of entry price {entry_price} for {position_side} position.")
                else:
                    fmt_tp = format_price_param(take_profit_price, "Take Profit")
                    if fmt_tp:
                        params_to_set['takeProfit'] = fmt_tp
                        log_parts.append(f"  - Set Take Profit: {fmt_tp}")
                    else:
                        lg.error(f"Failed to format Take Profit price {take_profit_price}. TP not set.")
            elif take_profit_price == Decimal('0'):
                # Explicitly clear fixed TP
                params_to_set['takeProfit'] = "0"
                log_parts.append("  - Clear Take Profit")

    except Exception as format_err:
        lg.error(f"Error during protection parameter formatting: {format_err}", exc_info=True)
        return False

    # If no valid parameters were generated after formatting, exit
    if not params_to_set:
        if any_protection_requested:
            lg.warning(f"No valid protection parameters to set for {symbol} after validation/formatting. No API call made.")
            # Return False because the intent to set protection failed
            return False
        else:
            lg.debug(f"No protection changes requested for {symbol}. No API call needed.")
            return True # No changes requested, so يعتبر successful

    # --- Prepare and Execute API Call ---
    # Determine Bybit category and market ID
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    market_id = market_info['id']
    # Get position index (usually 0 for one-way mode)
    position_idx = 0
    try:
        # Try to get positionIdx from the raw info if available
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None:
            position_idx = int(pos_idx_val)
    except (ValueError, TypeError):
        lg.warning("Could not parse positionIdx from position_info. Using default 0.")

    # Construct the final parameters for the Bybit API endpoint
    # Note: tpslMode='Full' means SL/TP apply to the entire position.
    # Trigger prices are set to 'LastPrice' by default. Order types are Market.
    final_api_params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full',          # Apply to whole position ('Partial' also possible)
        'slTriggerBy': 'LastPrice',  # Or MarkPrice, IndexPrice
        'tpTriggerBy': 'LastPrice',  # Or MarkPrice, IndexPrice
        'slOrderType': 'Market',     # Stop loss triggers a market order
        'tpOrderType': 'Market',     # Take profit triggers a market order
        'positionIdx': position_idx
    }
    final_api_params.update(params_to_set) # Add the specific SL/TP/TSL values

    lg.info("\n".join(log_parts)) # Log the intended changes
    lg.debug(f"  Executing Bybit API call: private_post /v5/position/set-trading-stop")
    lg.debug(f"  API Parameters: {final_api_params}")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing protection API call (Attempt {attempts+1}/{MAX_API_RETRIES+1})...")
            # Use the appropriate CCXT method for making authenticated POST requests
            # This might vary slightly depending on CCXT version and internal structure access
            # Assuming `exchange.private_post` is the correct way based on original code
            response = exchange.private_post('/v5/position/set-trading-stop', params=final_api_params)
            lg.debug(f"Set protection raw response: {response}")

            # Check Bybit V5 response code
            response_code = response.get('retCode')
            response_msg = response.get('retMsg', 'Unknown response message')

            if response_code == 0:
                 # Check if the message indicates no actual change was made
                 no_change_msgs = ["not modified", "no need to modify", "parameter not change", "same tpsl"]
                 if any(msg_part in response_msg.lower() for msg_part in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Protection already set or no change needed for {symbol}. (API Msg: {response_msg}){RESET}")
                 else:
                     lg.info(f"{NEON_GREEN}Protection successfully set/updated for {symbol}.{RESET}")
                 return True # Success
            else:
                # Log the error message from Bybit
                lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {response_msg} (Code: {response_code}){RESET}")
                # Identify potentially non-retryable error codes
                fatal_codes = [110013, 110036, 110086, 110084, 110085, 10001, 10002, 110025] # Examples
                fatal_strings = ["invalid", "parameter", "position not found"]
                is_fatal = response_code in fatal_codes or any(fs in response_msg.lower() for fs in fatal_strings)

                if is_fatal:
                     lg.error(" >> Hint: This protection error seems non-retryable. Check parameters or position status.")
                     return False # Non-retryable error
                else:
                     # Raise exception to trigger retry for potentially transient errors
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
            # Critical error
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Check API permissions. Stopping protection attempt.{RESET}")
            return False
        except ccxt.ExchangeError as e:
            # Catch re-raised errors or other general exchange errors
             if attempts >= MAX_API_RETRIES:
                 lg.error(f"Max retries exceeded for ExchangeError setting protection: {e}")
                 return False
             lg.warning(f"{NEON_YELLOW}Exchange error setting protection (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}. Retrying...{RESET}")
        except Exception as e:
            # Catch unexpected errors
            lg.error(f"{NEON_RED}Unexpected error setting protection (Attempt {attempts+1}/{MAX_API_RETRIES+1}): {e}{RESET}", exc_info=True)
            # Unexpected errors are likely fatal for this specific operation
            return False

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts) # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False

def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, config: Dict[str, Any],
                             logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool:
    """
    Calculates Trailing Stop Loss parameters based on configuration (callback rate, activation percentage)
    and the current position's entry price. Then calls the internal `_set_position_protection` helper.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Market symbol.
        market_info: Dictionary of market details.
        position_info: Dictionary of the current open position.
        config: The main configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional fixed take profit price to set simultaneously (Decimal). Set to 0 to clear.

    Returns:
        True if the TSL (and optional TP) was calculated and the API call to set it was successful, False otherwise.
    """
    lg = logger
    protection_config = config["protection"]

    # Validate inputs
    if not market_info or not position_info:
        lg.error(f"TSL calculation failed for {symbol}: Missing market or position info.")
        return False
    position_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if position_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"TSL calculation failed for {symbol}: Invalid position side ('{position_side}') or missing entry price ('{entry_price_str}').")
        return False

    try:
        # Parse necessary values from config and position info
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(protection_config["trailing_stop_callback_rate"]))
        activation_percentage = Decimal(str(protection_config["trailing_stop_activation_percentage"]))
        price_tick_size = Decimal(str(market_info['precision']['price']))

        # Validate parsed values
        if entry_price <= 0: raise ValueError("Entry price must be positive")
        if callback_rate <= 0: raise ValueError("Callback rate must be positive")
        if activation_percentage < 0: raise ValueError("Activation percentage cannot be negative")
        if price_tick_size <= 0: raise ValueError("Price tick size must be positive")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"TSL calculation failed for {symbol}: Invalid parameters in config or market/position info: {e}.")
        return False

    try:
        # Calculate Activation Price
        activation_offset = entry_price * activation_percentage
        raw_activation_price = (entry_price + activation_offset) if position_side == 'long' else (entry_price - activation_offset)

        # Round activation price *away* from entry price to ensure it triggers after entry
        if position_side == 'long':
            activation_price = raw_activation_price.quantize(price_tick_size, ROUND_UP)
            # Ensure activation is strictly greater than entry by at least one tick
            activation_price = max(activation_price, entry_price + price_tick_size)
        else: # Short
            activation_price = raw_activation_price.quantize(price_tick_size, ROUND_DOWN)
            # Ensure activation is strictly less than entry by at least one tick
            activation_price = min(activation_price, entry_price - price_tick_size)

        # Check if activation price calculation resulted in a valid price
        if activation_price <= 0:
            lg.error(f"TSL calculation failed: Calculated activation price ({activation_price}) is zero or negative.")
            return False

        # Calculate Trailing Distance based on the *activation* price and callback rate
        # Note: Bybit uses distance, not percentage, for the trailingStop parameter.
        # The distance is often calculated relative to the activation price.
        distance_raw = activation_price * callback_rate
        # Round distance UP to the nearest tick size, ensuring it's at least one tick
        trailing_distance = max(distance_raw.quantize(price_tick_size, ROUND_UP), price_tick_size)

        if trailing_distance <= 0:
            lg.error(f"TSL calculation failed: Calculated trailing distance ({trailing_distance}) is zero or negative.")
            return False

        lg.info(f"Calculated Trailing Stop for {symbol} ({position_side.upper()}):")
        lg.info(f"  Entry Price: {entry_price.normalize()}")
        lg.info(f"  Activation %: {activation_percentage:.3%}, Callback Rate: {callback_rate:.3%}")
        lg.info(f"  => Calculated Activation Price: {activation_price.normalize()}")
        lg.info(f"  => Calculated Trail Distance: {trailing_distance.normalize()}")
        if isinstance(take_profit_price, Decimal):
             tp_log = take_profit_price.normalize() if take_profit_price != 0 else 'Clear'
             lg.info(f"  Requested Take Profit: {tp_log}")

        # Call the internal helper function to make the API request
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # TSL overrides fixed SL in the API call logic
            take_profit_price=take_profit_price,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error during TSL calculation or setting: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy Implementation ---
class OrderBlock(TypedDict):
    """Represents a detected Order Block."""
    id: str             # Unique identifier (e.g., "B_230101120000")
    type: str           # 'bull' or 'bear'
    left_idx: pd.Timestamp # Timestamp of the pivot candle forming the OB
    right_idx: pd.Timestamp # Timestamp of the last candle the OB is valid for (or violation candle)
    top: Decimal        # Top price of the OB
    bottom: Decimal     # Bottom price of the OB
    active: bool        # Is the OB currently considered valid (not violated)?
    violated: bool      # Has the price closed beyond the OB?

class StrategyAnalysisResults(TypedDict):
    """Structure for returning results from the strategy analysis."""
    dataframe: pd.DataFrame          # DataFrame with all indicators
    last_close: Decimal              # Last closing price
    current_trend_up: Optional[bool] # True if trend is up, False if down, None if undetermined
    trend_just_changed: bool         # True if the trend changed on the last candle
    active_bull_boxes: List[OrderBlock] # List of currently active Bullish OBs
    active_bear_boxes: List[OrderBlock] # List of currently active Bearish OBs
    vol_norm_int: Optional[int]      # Normalized volume (0-100+) as integer, None if invalid
    atr: Optional[Decimal]           # Last ATR value, None if invalid
    upper_band: Optional[Decimal]    # Last Volumatic upper band value, None if invalid
    lower_band: Optional[Decimal]    # Last Volumatic lower band value, None if invalid

class VolumaticOBStrategy:
    """
    Implements the Volumatic Trend indicator combined with Pivot-based Order Block detection.
    Manages the state of active order blocks.
    """
    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the strategy engine with parameters from the config.

        Args:
            config: The main configuration dictionary.
            market_info: Dictionary of market details for the symbol.
            logger: Logger instance.
        """
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self.lg = logger # Alias for convenience

        strategy_cfg = config["strategy_params"]
        # Load Volumatic Trend parameters
        self.vt_length = int(strategy_cfg["vt_length"])
        self.vt_atr_period = int(strategy_cfg["vt_atr_period"])
        self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"])
        self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))

        # Load Order Block parameters
        self.ob_source = strategy_cfg["ob_source"] # "Wicks" or "Body"
        self.ph_left = int(strategy_cfg["ph_left"])
        self.ph_right = int(strategy_cfg["ph_right"])
        self.pl_left = int(strategy_cfg["pl_left"])
        self.pl_right = int(strategy_cfg["pl_right"])
        self.ob_extend = bool(strategy_cfg["ob_extend"])
        self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])

        # Internal state for tracking Order Blocks
        self.bull_boxes: List[OrderBlock] = []
        self.bear_boxes: List[OrderBlock] = []

        # Calculate minimum data length required based on lookback periods
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length) # Need enough data for EMAs/ATR/Vol EMA
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1) # Data needed for pivot calculation window
        self.min_data_len = max(required_for_vt, required_for_pivots) + 50 # Add buffer for stability
        # Ensure Vol EMA length isn't excessively large compared to internal DF limit
        self.vt_vol_ema_length = min(self.vt_vol_ema_length, MAX_DF_LEN - 50)

        self.lg.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.lg.info(f"  Volumatic Trend Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Volume EMA={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.lg.info(f"  Order Block Params: Source={self.ob_source}, PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")
        self.lg.info(f"  Calculated Minimum Data Required: {self.min_data_len} candles")

        # Warn if the required data length exceeds the API limit significantly
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 10:
            self.lg.error(f"{NEON_RED}{BRIGHT}CONFIGURATION WARNING:{RESET} Strategy requires {self.min_data_len} candles, which may exceed the API limit per request ({BYBIT_API_KLINE_LIMIT}).")
            self.lg.error(f"{NEON_YELLOW}  Consider reducing lookback periods (vt_length, vt_atr_period, vt_vol_ema_length, pivot lookbacks) in config.json.{RESET}")
        # Warn if Vol EMA length seems too large for typical usage
        if self.vt_vol_ema_length > 1000:
             self.lg.warning(f"{NEON_YELLOW}Volume EMA length ({self.vt_vol_ema_length}) is large. Ensure sufficient historical data is available.{RESET}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """
        Calculates the EMA of the Symmetrically Weighted Moving Average (SWMA) of a series.
        SWMA(close, 4) = (close[3]*1 + close[2]*2 + close[1]*2 + close[0]*1) / 6

        Args:
            series: The input pandas Series (e.g., close prices).
            length: The length for the final EMA calculation.

        Returns:
            A pandas Series containing the calculated EMA(SWMA).
        """
        if not isinstance(series, pd.Series) or series.empty or length <= 0:
            return pd.Series(np.nan, index=series.index)
        if len(series) < 4: # Need at least 4 periods for SWMA
            return pd.Series(np.nan, index=series.index)

        weights = np.array([1., 2., 2., 1.]) / 6.0
        # Ensure series is numeric, converting non-numeric to NaN
        series_numeric = pd.to_numeric(series, errors='coerce')
        if series_numeric.isnull().all(): # Return NaNs if all values are non-numeric
            return pd.Series(np.nan, index=series.index)

        # Calculate SWMA using rolling apply with the defined weights
        # raw=True might improve performance if series_numeric is already a NumPy array internally
        swma = series_numeric.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate EMA of the SWMA result
        # fillna=np.nan prevents forward filling during EMA calculation if SWMA has leading NaNs
        return ta.ema(swma, length=length, fillna=np.nan)

    def _find_pivots(self, series: pd.Series, left_bars: int, right_bars: int, find_highs: bool) -> pd.Series:
        """
        Finds pivot highs (find_highs=True) or pivot lows (find_highs=False)
        based on a simple lookback comparison. A pivot requires the candle's
        value to be strictly higher (for highs) or lower (for lows) than all
        candles within the left_bars and right_bars window.

        Args:
            series: The pandas Series to find pivots in (e.g., high or low prices).
            left_bars: Number of bars to look back to the left.
            right_bars: Number of bars to look forward to the right.
            find_highs: True to find pivot highs, False to find pivot lows.

        Returns:
            A pandas Series of booleans, True where a pivot is detected.
        """
        if not isinstance(series, pd.Series) or series.empty or left_bars < 1 or right_bars < 1:
            self.lg.warning("Invalid input for _find_pivots. Returning empty Series.")
            return pd.Series(False, index=series.index)

        # Ensure the series contains numeric data
        numeric_series = pd.to_numeric(series, errors='coerce')
        if numeric_series.isnull().all():
            self.lg.debug("Pivot series contains only NaNs.")
            return pd.Series(False, index=series.index)

        # Initialize result series assuming all are pivots initially
        is_pivot = pd.Series(True, index=series.index)

        # Check left bars
        for i in range(1, left_bars + 1):
            shifted_left = numeric_series.shift(i)
            if find_highs:
                # For high pivot, current must be > shifted left
                is_pivot &= (numeric_series > shifted_left)
            else:
                # For low pivot, current must be < shifted left
                is_pivot &= (numeric_series < shifted_left)

        # Check right bars
        for i in range(1, right_bars + 1):
            shifted_right = numeric_series.shift(-i)
            if find_highs:
                # For high pivot, current must be > shifted right
                is_pivot &= (numeric_series > shifted_right)
            else:
                # For low pivot, current must be < shifted right
                is_pivot &= (numeric_series < shifted_right)

        # NaNs introduced by shifting should not be marked as pivots
        return is_pivot.fillna(False)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """
        Processes the historical kline DataFrame to calculate all strategy indicators,
        detect pivots, create/update Order Blocks, and return the analysis results.

        Args:
            df_input: The input pandas DataFrame with OHLCV data (index=Timestamp, columns=Decimal).

        Returns:
            A StrategyAnalysisResults dictionary containing the processed DataFrame,
            latest state, and active order blocks. Returns empty results on failure.
        """
        # Define default empty results structure
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.lg.error("Strategy update received an empty DataFrame.")
            return empty_results

        # Work on a copy to avoid modifying the original DataFrame
        df = df_input.copy()

        # --- Pre-computation Checks ---
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
            self.lg.error("Strategy update DataFrame index is not a monotonic DatetimeIndex.")
            return empty_results
        if len(df) < self.min_data_len:
            self.lg.warning(f"Insufficient data for strategy calculation ({len(df)} candles < required {self.min_data_len}). Results may be inaccurate.")
            # Proceed, but be aware results might have NaNs or be less reliable

        self.lg.debug(f"Starting strategy analysis on {len(df)} candles (minimum required: {self.min_data_len}).")

        # --- Convert to Float for TA-Lib/Pandas-TA ---
        # Most TA libraries work more efficiently with floats. We'll convert back later if needed.
        try:
            df_float = pd.DataFrame(index=df.index)
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.lg.error(f"Strategy update failed: Missing required column '{col}' in input DataFrame.")
                    return empty_results
                df_float[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows where essential float conversions failed
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_float.empty:
                self.lg.error("DataFrame became empty after converting essential columns to float.")
                return empty_results
            self.lg.debug("Successfully converted relevant columns to float for TA calculations.")
        except Exception as e:
            self.lg.error(f"Error converting DataFrame columns to float: {e}", exc_info=True)
            return empty_results

        # --- Indicator Calculations (using float DataFrame) ---
        try:
            self.lg.debug("Calculating indicators: ATR, EMAs, Trend, Bands, Volume Normalization...")
            # ATR
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # Volumatic Trend EMAs
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length) # EMA(SWMA(close,4), length)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan) # Regular EMA(close, length)

            # Determine Trend Direction and Changes
            # Trend is up if EMA2 > EMA1 of the *previous* bar
            # ffill() handles initial NaNs, fillna(False) ensures a boolean result
            trend_up_series = (df_float['ema2'] > df_float['ema1'].shift(1)).ffill().fillna(False)
            df_float['trend_up'] = trend_up_series

            # Trend change occurs if trend_up differs from the previous bar's trend_up
            trend_changed_series = (df_float['trend_up'].shift(1) != df_float['trend_up']) & \
                                   df_float['trend_up'].notna() & \
                                   df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'] = trend_changed_series.fillna(False) # Handle NaNs at the start

            # Calculate Volumatic Bands
            # Capture EMA1 and ATR values only on bars where the trend *just* changed
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these values to create the stable levels for the bands until the next trend change
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Upper and Lower Bands
            atr_multiplier_float = float(self.vt_atr_multiplier)
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_multiplier_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_multiplier_float)

            # Volume Normalization
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            # Ensure min_periods is reasonable for rolling max calculation
            min_periods_vol = max(1, self.vt_vol_ema_length // 10) # Avoid issues with very short data
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=min_periods_vol).max().fillna(0.0)
            # Calculate normalized volume (avoid division by zero)
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9, # Use small threshold
                                           (volume_numeric / df_float['vol_max'] * 100.0),
                                           0.0) # Set to 0 if max volume is near zero
            # Fill any remaining NaNs and clip result (e.g., 0-200 range)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)

            # --- Pivot Detection ---
            self.lg.debug("Detecting Pivot Highs and Lows...")
            # Select the source series for pivots based on config
            if self.ob_source.lower() == "wicks":
                high_series = df_float['high']
                low_series = df_float['low']
                self.lg.debug("Using candle Highs/Lows for pivot detection.")
            else: # "Body"
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)
                self.lg.debug("Using candle Body Max/Min for pivot detection.")

            df_float['is_ph'] = self._find_pivots(high_series, self.ph_left, self.ph_right, find_highs=True)
            df_float['is_pl'] = self._find_pivots(low_series, self.pl_left, self.pl_right, find_highs=False)

        except Exception as e:
            self.lg.error(f"Error during indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy Results Back to Original Decimal DataFrame (Optional but good practice) ---
        # This keeps the primary DataFrame in Decimal format if needed downstream
        try:
            self.lg.debug("Copying calculated indicators back to Decimal DataFrame...")
            indicator_cols = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                              'upper_band', 'lower_band', 'vol_norm', 'is_ph', 'is_pl']
            for col in indicator_cols:
                if col in df_float.columns:
                    # Reindex to align with original Decimal DF index, just in case rows were dropped
                    source_series = df_float[col].reindex(df.index)
                    if source_series.dtype == 'bool':
                        df[col] = source_series.astype(bool)
                    elif pd.api.types.is_object_dtype(source_series): # Keep objects as is (shouldn't happen here)
                         df[col] = source_series
                    else: # Convert numeric types back to Decimal
                        df[col] = source_series.apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                        )
        except Exception as e:
            self.lg.error(f"Error converting calculated indicators back to Decimal: {e}", exc_info=True)
            # Continue with the float results if conversion fails, but log it

        # --- Clean Final DataFrame ---
        # Drop rows where essential indicators might still be NaN (e.g., at the start of the series)
        initial_len = len(df)
        required_indicator_cols = ['close', 'atr', 'trend_up', 'upper_band', 'lower_band', 'is_ph', 'is_pl']
        df.dropna(subset=required_indicator_cols, inplace=True)
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            self.lg.debug(f"Dropped {rows_dropped} rows from final DataFrame due to missing required indicator values.")
        if df.empty:
            self.lg.warning("DataFrame became empty after final indicator cleaning.")
            return empty_results # Cannot proceed if DF is empty

        self.lg.debug("Indicators calculated and DataFrame cleaned. Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_ob_count = 0
            last_candle_index = df.index[-1] if not df.empty else None

            # Iterate through candles where pivots might have formed
            # Focus on candles where 'is_ph' or 'is_pl' is True
            pivot_indices = df.index[df['is_ph'] | df['is_pl']]

            for pivot_idx in pivot_indices:
                 try:
                     pivot_candle = df.loc[pivot_idx]
                     is_pivot_high = pivot_candle['is_ph']
                     is_pivot_low = pivot_candle['is_pl']

                     # --- Create Bearish OB from Pivot High ---
                     if is_pivot_high:
                         # Check if an OB from this exact pivot candle already exists
                         if not any(b['left_idx'] == pivot_idx and b['type'] == 'bear' for b in self.bear_boxes):
                             # Determine OB boundaries based on source config
                             if self.ob_source.lower() == "wicks":
                                 top = pivot_candle['high']
                                 # Bearish OB body is often the Open price of the pivot candle
                                 bottom = pivot_candle['open']
                             else: # Body
                                 top = max(pivot_candle['open'], pivot_candle['close'])
                                 bottom = min(pivot_candle['open'], pivot_candle['close'])

                             # Ensure valid Decimal boundaries
                             if pd.notna(top) and pd.notna(bottom) and isinstance(top, Decimal) and isinstance(bottom, Decimal) and top > bottom:
                                 new_box = OrderBlock(
                                     id=f"B_{pivot_idx.strftime('%y%m%d%H%M%S')}", # Unique ID based on timestamp
                                     type='bear',
                                     left_idx=pivot_idx,
                                     right_idx=last_candle_index, # Initially extends to latest candle
                                     top=top,
                                     bottom=bottom,
                                     active=True,
                                     violated=False
                                 )
                                 self.bear_boxes.append(new_box)
                                 new_ob_count += 1
                                 self.lg.debug(f"  + New Bearish OB created: {new_box['id']} @ {pivot_idx.strftime('%Y-%m-%d %H:%M')} [{bottom.normalize()}-{top.normalize()}]")
                             else:
                                 self.lg.warning(f"Could not create Bearish OB at {pivot_idx}: Invalid boundaries (Top: {top}, Bottom: {bottom})")

                     # --- Create Bullish OB from Pivot Low ---
                     if is_pivot_low:
                          if not any(b['left_idx'] == pivot_idx and b['type'] == 'bull' for b in self.bull_boxes):
                             if self.ob_source.lower() == "wicks":
                                 # Bullish OB body is often the Open price
                                 top = pivot_candle['open']
                                 bottom = pivot_candle['low']
                             else: # Body
                                 top = max(pivot_candle['open'], pivot_candle['close'])
                                 bottom = min(pivot_candle['open'], pivot_candle['close'])

                             if pd.notna(top) and pd.notna(bottom) and isinstance(top, Decimal) and isinstance(bottom, Decimal) and top > bottom:
                                 new_box = OrderBlock(
                                     id=f"L_{pivot_idx.strftime('%y%m%d%H%M%S')}",
                                     type='bull',
                                     left_idx=pivot_idx,
                                     right_idx=last_candle_index,
                                     top=top,
                                     bottom=bottom,
                                     active=True,
                                     violated=False
                                 )
                                 self.bull_boxes.append(new_box)
                                 new_ob_count += 1
                                 self.lg.debug(f"  + New Bullish OB created: {new_box['id']} @ {pivot_idx.strftime('%Y-%m-%d %H:%M')} [{bottom.normalize()}-{top.normalize()}]")
                             else:
                                 self.lg.warning(f"Could not create Bullish OB at {pivot_idx}: Invalid boundaries (Top: {top}, Bottom: {bottom})")
                 except Exception as e:
                     # Log error for specific pivot but continue processing others
                     self.lg.warning(f"Error processing pivot at index {pivot_idx}: {e}", exc_info=True)

            if new_ob_count > 0:
                self.lg.debug(f"Identified {new_ob_count} new Order Blocks.")

            # --- Update Existing Order Blocks (Violation Check & Extension) ---
            last_candle = df.iloc[-1] if not df.empty else None
            if last_candle is not None and pd.notna(last_candle.get('close')) and isinstance(last_candle['close'], Decimal):
                last_close_price = last_candle['close']
                last_idx = last_candle.name # Timestamp of the last candle

                # Check Bullish OBs for violation
                for box in self.bull_boxes:
                    if box['active']:
                        # Violation: Close price goes below the bottom of the Bullish OB
                        if last_close_price < box['bottom']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx # Mark violation time
                            self.lg.debug(f"  - Bullish OB {box['id']} VIOLATED by close {last_close_price} < {box['bottom']}")
                        # Extend active OB to the current candle if configured
                        elif self.ob_extend:
                            box['right_idx'] = last_idx

                # Check Bearish OBs for violation
                for box in self.bear_boxes:
                    if box['active']:
                        # Violation: Close price goes above the top of the Bearish OB
                        if last_close_price > box['top']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx
                            self.lg.debug(f"  - Bearish OB {box['id']} VIOLATED by close {last_close_price} > {box['top']}")
                        elif self.ob_extend:
                            box['right_idx'] = last_idx
            else:
                self.lg.warning("Cannot check Order Block violations: Invalid or missing last close price.")

            # --- Prune Order Blocks ---
            # Keep only active (non-violated) OBs
            active_bull_obs = [b for b in self.bull_boxes if b['active']]
            active_bear_obs = [b for b in self.bear_boxes if b['active']]

            # Sort by creation time (left_idx) descending (most recent first)
            active_bull_obs.sort(key=lambda b: b['left_idx'], reverse=True)
            active_bear_obs.sort(key=lambda b: b['left_idx'], reverse=True)

            # Keep only the N most recent active OBs defined by ob_max_boxes
            self.bull_boxes = active_bull_obs[:self.ob_max_boxes]
            self.bear_boxes = active_bear_obs[:self.ob_max_boxes]

            self.lg.debug(f"Pruned Order Blocks. Kept Active: Bullish={len(self.bull_boxes)}, Bearish={len(self.bear_boxes)} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.lg.error(f"Error during Order Block processing: {e}", exc_info=True)
            # Continue to return results, but OBs might be inaccurate

        # --- Prepare Final Results ---
        last_candle_final = df.iloc[-1] if not df.empty else None

        # Helper to safely get Decimal values from the last candle
        def safe_decimal_from_candle(value: Any, positive_only: bool = False) -> Optional[Decimal]:
            if pd.notna(value) and isinstance(value, Decimal) and np.isfinite(float(value)):
                 if not positive_only or value > 0:
                     return value
            return None

        results = StrategyAnalysisResults(
            dataframe=df, # Return the DataFrame with all calculated indicators
            last_close=safe_decimal_from_candle(last_candle_final.get('close')) or Decimal('0'),
            # Safely get boolean trend status
            current_trend_up=bool(last_candle_final['trend_up']) if last_candle_final is not None and isinstance(last_candle_final.get('trend_up'), (bool, np.bool_)) else None,
            trend_just_changed=bool(last_candle_final['trend_changed']) if last_candle_final is not None and isinstance(last_candle_final.get('trend_changed'), (bool, np.bool_)) else False,
            # Return the pruned lists of active OBs
            active_bull_boxes=self.bull_boxes,
            active_bear_boxes=self.bear_boxes,
            # Safely get volume normalization and ATR
            vol_norm_int=int(v) if (v := safe_decimal_from_candle(last_candle_final.get('vol_norm'))) is not None else None,
            atr=safe_decimal_from_candle(last_candle_final.get('atr'), positive_only=True),
            upper_band=safe_decimal_from_candle(last_candle_final.get('upper_band')),
            lower_band=safe_decimal_from_candle(last_candle_final.get('lower_band'))
        )

        # Log summary of the final results
        trend_status_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else \
                           f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                           f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr'].normalize()}" if results['atr'] else "N/A"
        time_str = last_candle_index.strftime('%Y-%m-%d %H:%M:%S %Z') if last_candle_index else "N/A"

        self.lg.debug(f"Strategy Analysis Complete ({time_str}):")
        self.lg.debug(f"  Last Close: {results['last_close'].normalize()}")
        self.lg.debug(f"  Trend: {trend_status_str}, Trend Changed: {results['trend_just_changed']}")
        self.lg.debug(f"  ATR: {atr_str}, Vol Norm: {results['vol_norm_int']}")
        self.lg.debug(f"  Bands (L/U): {results['lower_band'].normalize() if results['lower_band'] else 'N/A'} / {results['upper_band'].normalize() if results['upper_band'] else 'N/A'}")
        self.lg.debug(f"  Active OBs (Bull/Bear): {len(results['active_bull_boxes'])} / {len(results['active_bear_boxes'])}")

        return results

# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """
    Generates trading signals ("BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", "HOLD")
    based on the strategy analysis results and the current position state.
    Also calculates initial Stop Loss and Take Profit levels.
    """
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initializes the Signal Generator with configuration parameters.

        Args:
            config: The main configuration dictionary.
            logger: Logger instance.
        """
        self.config = config
        self.logger = logger
        self.lg = logger # Alias
        strategy_cfg = config["strategy_params"]
        protection_cfg = config["protection"]

        # Load and validate necessary parameters
        try:
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            if self.ob_entry_proximity_factor < 1: raise ValueError("ob_entry_proximity_factor must be >= 1")

            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            if self.ob_exit_proximity_factor < 1: raise ValueError("ob_exit_proximity_factor must be >= 1")

            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            if self.initial_tp_atr_multiple < 0: raise ValueError("initial_take_profit_atr_multiple cannot be negative")

            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))
            if self.initial_sl_atr_multiple <= 0: raise ValueError("initial_stop_loss_atr_multiple must be positive")

            self.lg.info("Signal Generator Initialized:")
            self.lg.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor:.4f}")
            self.lg.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor:.4f}")
            self.lg.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()} {'(Disabled)' if self.initial_tp_atr_multiple == 0 else ''}")
            self.lg.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.lg.error(f"{NEON_RED}Error initializing SignalGenerator with config values: {e}. Using hardcoded defaults as fallback.{RESET}", exc_info=True)
             # Hardcoded defaults as a safety measure
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """
        Determines the trading signal based on strategy rules and current position.

        Args:
            analysis_results: The results from the VolumaticOBStrategy update.
            open_position: The current open position dictionary, or None if no position.

        Returns:
            A string signal: "BUY", "SELL", "EXIT_LONG", "EXIT_SHORT", or "HOLD".
        """
        lg = self.lg

        # --- Input Validation ---
        if not analysis_results or \
           analysis_results['current_trend_up'] is None or \
           analysis_results['last_close'] <= 0 or \
           analysis_results['atr'] is None or analysis_results['atr'] <= 0:
            lg.warning(f"{NEON_YELLOW}Signal Generation: Invalid or incomplete strategy analysis results. Holding.{RESET}")
            lg.debug(f"  Analysis Results: {analysis_results}") # Log why it failed
            return "HOLD"

        # Extract key values for easier access
        last_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up']
        trend_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']
        position_side = open_position.get('side') if open_position else None # 'long', 'short', or None

        signal = "HOLD" # Default signal

        lg.debug(f"Generating Signal:")
        lg.debug(f"  Last Close: {last_close.normalize()}")
        lg.debug(f"  Trend Up: {is_trend_up}, Trend Changed: {trend_changed}")
        lg.debug(f"  Active OBs (Bull/Bear): {len(active_bull_obs)} / {len(active_bear_obs)}")
        lg.debug(f"  Current Position: {position_side or 'None'}")

        # --- 1. Check for Exit Conditions (if position exists) ---
        if position_side == 'long':
            # Exit Long Condition 1: Trend flips to Down
            if is_trend_up is False and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped to DOWN.{RESET}")
            # Exit Long Condition 2: Price approaches a Bearish OB
            elif signal == "HOLD" and active_bear_obs:
                try:
                    # Find the nearest Bearish OB (based on top edge)
                    nearest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - last_close))
                    # Define exit threshold slightly *above* the OB top using the exit factor
                    exit_threshold = nearest_bear_ob['top'] * self.ob_exit_proximity_factor
                    if last_close >= exit_threshold:
                        signal = "EXIT_LONG"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price {last_close.normalize()} hit Bearish OB exit threshold {exit_threshold.normalize()} (OB ID: {nearest_bear_ob['id']}){RESET}")
                except Exception as e:
                    lg.warning(f"Error during Bearish OB exit check for long position: {e}")

        elif position_side == 'short':
            # Exit Short Condition 1: Trend flips to Up
            if is_trend_up is True and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped to UP.{RESET}")
            # Exit Short Condition 2: Price approaches a Bullish OB
            elif signal == "HOLD" and active_bull_obs:
                try:
                    # Find the nearest Bullish OB (based on bottom edge)
                    nearest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - last_close))
                    # Define exit threshold slightly *below* the OB bottom using the exit factor
                    # Use division for the factor here (e.g., bottom / 1.001)
                    exit_threshold = nearest_bull_ob['bottom'] / self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else nearest_bull_ob['bottom']
                    if last_close <= exit_threshold:
                        signal = "EXIT_SHORT"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price {last_close.normalize()} hit Bullish OB exit threshold {exit_threshold.normalize()} (OB ID: {nearest_bull_ob['id']}){RESET}")
                except Exception as e:
                    lg.warning(f"Error during Bullish OB exit check for short position: {e}")

        # If an exit signal was generated, return it immediately
        if signal != "HOLD":
            return signal

        # --- 2. Check for Entry Conditions (if no position exists) ---
        if position_side is None:
            # Entry Long Condition: Trend is Up AND Price is within a Bullish OB (with proximity factor)
            if is_trend_up is True and active_bull_obs:
                for ob in active_bull_obs:
                    # Define entry zone: bottom of OB up to top * factor
                    entry_zone_bottom = ob['bottom']
                    entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "BUY"
                        lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal Triggered:{RESET}")
                        lg.info(f"  Trend is UP.")
                        lg.info(f"  Price {last_close.normalize()} is within Bullish OB {ob['id']} (ID: {ob['id']})")
                        lg.info(f"  OB Range: [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
                        lg.info(f"  Entry Zone (with prox factor): [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                        break # Found a valid entry OB, no need to check others

            # Entry Short Condition: Trend is Down AND Price is within a Bearish OB (with proximity factor)
            elif is_trend_up is False and active_bear_obs:
                for ob in active_bear_obs:
                    # Define entry zone: bottom / factor up to top of OB
                    entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']
                    entry_zone_top = ob['top']
                    if entry_zone_bottom <= last_close <= entry_zone_top:
                        signal = "SELL"
                        lg.info(f"{NEON_RED}{BRIGHT}SELL Signal Triggered:{RESET}")
                        lg.info(f"  Trend is DOWN.")
                        lg.info(f"  Price {last_close.normalize()} is within Bearish OB {ob['id']} (ID: {ob['id']})")
                        lg.info(f"  OB Range: [{ob['bottom'].normalize()} - {ob['top'].normalize()}]")
                        lg.info(f"  Entry Zone (with prox factor): [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                        break # Found valid entry

        # --- 3. Default to HOLD ---
        if signal == "HOLD":
            lg.debug(f"Signal Result: HOLD - No valid entry or exit condition met.")
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: Dict, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Calculates initial Take Profit (TP) and Stop Loss (SL) levels based on
        entry price, ATR, configured multipliers, and market precision.

        Args:
            entry_price: The estimated or actual entry price (Decimal).
            signal: The entry signal ("BUY" or "SELL").
            atr: The current ATR value (Decimal).
            market_info: Dictionary of market details.
            exchange: Initialized CCXT exchange object (for formatting).

        Returns:
            A tuple containing (take_profit_price, stop_loss_price).
            TP can be None if disabled or calculation fails.
            SL can be None if calculation fails (critical). Prices are Decimals.
        """
        lg = self.lg

        # Validate inputs
        if signal.upper() not in ["BUY", "SELL"]:
            lg.error(f"Invalid signal '{signal}' for TP/SL calculation.")
            return None, None
        if entry_price <= 0:
            lg.error(f"Invalid entry price ({entry_price}) for TP/SL calculation.")
            return None, None
        if atr <= 0:
            lg.error(f"Invalid ATR value ({atr}) for TP/SL calculation.")
            return None, None
        if market_info['precision'].get('price') is None:
             lg.error(f"Missing price precision in market info for {market_info['symbol']}. Cannot calculate TP/SL.")
             return None, None

        try:
            # Get minimum price tick size for formatting
            price_tick_size = Decimal(str(market_info['precision']['price']))
            if price_tick_size <= 0: raise ValueError("Invalid price tick size")

            # Get multipliers from instance variables
            tp_multiplier = self.initial_tp_atr_multiple
            sl_multiplier = self.initial_sl_atr_multiple

            # Calculate raw offsets
            tp_offset = atr * tp_multiplier
            sl_offset = atr * sl_multiplier
            lg.debug(f"Calculating TP/SL for {signal} at {entry_price.normalize()} with ATR {atr.normalize()}:")
            lg.debug(f"  TP Mult: {tp_multiplier}, Offset: {tp_offset.normalize()}")
            lg.debug(f"  SL Mult: {sl_multiplier}, Offset: {sl_offset.normalize()}")

            # Calculate raw TP/SL prices
            raw_tp = None
            if tp_multiplier > 0: # Only calculate TP if multiplier is positive
                 raw_tp = (entry_price + tp_offset) if signal.upper() == "BUY" else (entry_price - tp_offset)

            raw_sl = (entry_price - sl_offset) if signal.upper() == "BUY" else (entry_price + sl_offset)

            lg.debug(f"  Raw Levels: TP={raw_tp.normalize() if raw_tp else 'N/A'}, SL={raw_sl.normalize()}")

            # Helper to format and validate the calculated level
            def format_and_validate_level(raw_level: Optional[Decimal], level_name: str) -> Optional[Decimal]:
                if raw_level is None or raw_level <= 0:
                    lg.debug(f"Calculated raw {level_name} is invalid ({raw_level}).")
                    return None
                try:
                    # Format using CCXT's price_to_precision
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(raw_level))
                    formatted_decimal = Decimal(formatted_str)
                    if formatted_decimal <= 0:
                        lg.warning(f"Formatted {level_name} ({formatted_decimal}) is zero or negative. Invalid.")
                        return None
                    return formatted_decimal
                except (ccxt.ExchangeError, ValueError, TypeError, InvalidOperation) as e:
                    lg.error(f"Error formatting {level_name} level {raw_level}: {e}. Cannot set level.")
                    return None

            # Format TP and SL
            take_profit = format_and_validate_level(raw_tp, "Take Profit")
            stop_loss = format_and_validate_level(raw_sl, "Stop Loss")

            # --- Post-formatting Validation ---
            # Ensure SL is strictly on the loss side of the entry price
            if stop_loss is not None:
                sl_valid = (signal.upper() == "BUY" and stop_loss < entry_price) or \
                           (signal.upper() == "SELL" and stop_loss > entry_price)
                if not sl_valid:
                    lg.warning(f"{NEON_YELLOW}Formatted SL {stop_loss} is not strictly beyond entry price {entry_price} for {signal} signal.{RESET}")
                    # Attempt to adjust SL by one tick away from entry
                    adjusted_sl_raw = (entry_price - price_tick_size) if signal.upper() == "BUY" else (entry_price + price_tick_size)
                    stop_loss = format_and_validate_level(adjusted_sl_raw, "Adjusted Stop Loss")
                    if stop_loss:
                         lg.warning(f"  Adjusted SL to {stop_loss.normalize()} (one tick away).")
                    else:
                         lg.error(f"{NEON_RED}  Failed to calculate valid adjusted SL. Critical SL failure.{RESET}")
                         # Critical failure if SL cannot be set correctly
                         return take_profit, None # Return potentially valid TP, but None SL

            # Ensure TP (if enabled and calculated) is strictly on the profit side
            if take_profit is not None:
                tp_valid = (signal.upper() == "BUY" and take_profit > entry_price) or \
                           (signal.upper() == "SELL" and take_profit < entry_price)
                if not tp_valid:
                    lg.warning(f"{NEON_YELLOW}Formatted TP {take_profit} is not strictly beyond entry price {entry_price} for {signal} signal. Disabling TP.{RESET}")
                    take_profit = None # Disable TP if it's invalid

            lg.info(f"Calculated Initial Protection Levels:")
            lg.info(f"  Take Profit: {take_profit.normalize() if take_profit else 'None (Disabled or Invalid)'}")
            lg.info(f"  Stop Loss: {stop_loss.normalize() if stop_loss else f'{NEON_RED}FAIL{RESET}'}")

            # SL is mandatory for risk calculation, return None if it failed
            if stop_loss is None:
                lg.error(f"{NEON_RED}Stop Loss calculation failed. Cannot proceed with trade sizing.{RESET}")
                return take_profit, None

            return take_profit, stop_loss

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
            return None, None

# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: Dict) -> None:
    """
    Performs one full cycle of analysis and trading logic for a single symbol.
    Fetches data, runs strategy, gets signals, checks position, manages trades.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The market symbol to analyze and trade.
        config: The main configuration dictionary.
        logger: The logger instance for this symbol.
        strategy_engine: Initialized VolumaticOBStrategy instance.
        signal_generator: Initialized SignalGenerator instance.
        market_info: Dictionary of market details for the symbol.
    """
    lg = logger
    lg.info(f"\n{BRIGHT}---=== Cycle Start: Analyzing {symbol} ({config['interval']} TF) ===---{RESET}")
    cycle_start_time = time.monotonic()

    # Map the config interval (e.g., "5") to CCXT's format (e.g., "5m")
    try:
        ccxt_interval = CCXT_INTERVAL_MAP[config["interval"]]
    except KeyError:
        lg.error(f"Invalid interval '{config['interval']}' found in config during cycle. Using default '5m'.")
        ccxt_interval = "5m"

    # --- 1. Determine Kline Fetch Limit ---
    min_required_data = strategy_engine.min_data_len
    fetch_limit_from_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    # Need at least the strategy minimum, but respect user config if it asks for more (up to API limit)
    effective_fetch_limit = max(min_required_data, fetch_limit_from_config)
    # Actual request limit is capped by the API's max klines per request
    request_limit = min(effective_fetch_limit, BYBIT_API_KLINE_LIMIT)
    lg.info(f"Data Requirements: Strategy Min={min_required_data}, Config Pref={fetch_limit_from_config}")
    lg.info(f"Fetching {request_limit} klines for {symbol} ({ccxt_interval})...")

    # --- 2. Fetch Kline Data ---
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=request_limit, logger=lg)
    fetched_count = len(klines_df)

    # --- 3. Validate Fetched Data ---
    if klines_df.empty or fetched_count < min_required_data:
        # Check if the reason for insufficient data was hitting the API limit
        api_limit_was_hit = (request_limit == BYBIT_API_KLINE_LIMIT and fetched_count == BYBIT_API_KLINE_LIMIT)

        if api_limit_was_hit and fetched_count < min_required_data:
            # Critical situation: API limit prevents getting enough data for the strategy config
            lg.error(f"{NEON_RED}{BRIGHT}CRITICAL DATA SHORTFALL:{RESET} Fetched max {fetched_count} klines (API limit), but strategy requires {min_required_data}.")
            lg.error(f"{NEON_YELLOW}  ACTION REQUIRED: Reduce lookback periods (e.g., vt_length, vt_atr_period, vt_vol_ema_length, pivot lookbacks) in config.json.{RESET}")
        elif klines_df.empty:
            lg.error(f"Failed to fetch any valid kline data for {symbol}. Skipping cycle.")
        else: # Got some data, but not enough
            lg.error(f"Fetched only {fetched_count} klines, but strategy requires {min_required_data}. Insufficient data. Skipping cycle.")
        # Skip the rest of the cycle if data is insufficient
        return

    # --- 4. Run Strategy Analysis ---
    lg.debug("Running strategy analysis on fetched data...")
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis engine failed: {analysis_err}{RESET}", exc_info=True)
        return # Cannot proceed without analysis

    # Validate essential results from analysis
    if not analysis_results or \
       analysis_results['current_trend_up'] is None or \
       analysis_results['last_close'] <= 0 or \
       analysis_results['atr'] is None or analysis_results['atr'] <= 0:
        lg.error(f"{NEON_RED}Strategy analysis did not produce valid results (trend, close, or ATR missing/invalid). Skipping cycle.{RESET}")
        lg.debug(f"Problematic Analysis Results: {analysis_results}")
        return

    latest_close_price = analysis_results['last_close']
    current_atr_value = analysis_results['atr']
    lg.info(f"Strategy Analysis Results: Trend={analysis_results['current_trend_up']}, Last Close={latest_close_price.normalize()}, ATR={current_atr_value.normalize()}")

    # --- 5. Get Current Market State (Price & Position) ---
    lg.debug("Fetching current market price and position status...")
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    open_position = get_open_position(exchange, symbol, lg)

    # Use live price if available, otherwise fallback to last close for checks
    price_for_checks = current_market_price if current_market_price and current_market_price > 0 else latest_close_price
    if price_for_checks <= 0:
        lg.error(f"{NEON_RED}Cannot determine a valid current price (Live: {current_market_price}, Last Close: {latest_close_price}). Skipping protection/signal checks.{RESET}")
        return
    if current_market_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch live market price. Using last kline close ({latest_close_price.normalize()}) for protection/signal checks.{RESET}")
    else:
        lg.debug(f"Using live market price ({current_market_price.normalize()}) for checks.")


    # --- 6. Generate Trading Signal ---
    lg.debug("Generating trading signal based on analysis and position...")
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
        lg.info(f"Generated Signal: {BRIGHT}{signal}{RESET}")
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation failed: {signal_err}{RESET}", exc_info=True)
        return # Cannot proceed without signal

    # --- 7. Trading Logic ---
    trading_enabled = config.get("enable_trading", False)

    # --- Scenario: Trading Disabled (Analysis/Logging Only) ---
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading is DISABLED.{RESET} Analysis complete. Signal was: {signal}")
        if open_position is None and signal in ["BUY", "SELL"]:
            lg.info(f"  (Action: Would attempt to {signal} if trading were enabled)")
        elif open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]:
            lg.info(f"  (Action: Would attempt to {signal} current {open_position['side']} position if trading were enabled)")
        else:
            lg.info("  (Action: No entry or exit action indicated by signal)")
        # End cycle here if trading disabled
        cycle_end_time = time.monotonic()
        lg.debug(f"---=== Analysis-Only Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---\n")
        return

    # --- Trading IS Enabled ---
    lg.debug(f"Trading is ENABLED. Processing signal '{signal}'...")

    # --- Scenario 1: No Position -> Consider Entry ---
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** Signal: {signal} | Current Position: None | Initiating Entry Sequence... ***{RESET}")

        # i. Check Balance
        lg.debug("Fetching balance for sizing...")
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= Decimal('0'):
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot size order. Failed to fetch balance or balance is zero/negative ({balance}).{RESET}")
            return

        # ii. Calculate Initial TP/SL using latest close as estimated entry
        lg.debug(f"Calculating initial TP/SL based on last close ({latest_close_price.normalize()}) and ATR ({current_atr_value.normalize()})...")
        initial_tp, initial_sl = signal_generator.calculate_initial_tp_sl(latest_close_price, signal, current_atr_value, market_info, exchange)
        if initial_sl is None:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to calculate a valid initial Stop Loss.{RESET}")
            return
        if initial_tp is None:
            lg.warning(f"Initial Take Profit calculation failed or is disabled. Proceeding without initial TP.")

        # iii. Set Leverage (if applicable)
        leverage_set_ok = True
        if market_info['is_contract']:
            leverage_val = int(config.get('leverage', 0))
            if leverage_val > 0:
                lg.debug(f"Setting leverage to {leverage_val}x for {symbol}...")
                leverage_set_ok = set_leverage_ccxt(exchange, symbol, leverage_val, market_info, lg)
            else:
                lg.info(f"Leverage setting skipped (config leverage is {leverage_val}).")
        if not leverage_set_ok:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set leverage for {symbol}.{RESET}")
            return

        # iv. Calculate Position Size
        lg.debug("Calculating position size...")
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=config["risk_per_trade"],
            initial_stop_loss_price=initial_sl,
            entry_price=latest_close_price, # Use last close as estimate for sizing
            market_info=market_info,
            exchange=exchange,
            logger=lg
        )
        if position_size is None or position_size <= 0:
            lg.error(f"{NEON_RED}Trade Aborted ({signal}): Position size calculation failed or resulted in zero/negative size ({position_size}).{RESET}")
            return

        # v. Place Market Order
        lg.info(f"{BRIGHT}===> Placing {signal} Market Order | Size: {position_size.normalize()} <==={RESET}")
        trade_order = place_trade(
            exchange=exchange,
            symbol=symbol,
            trade_signal=signal,
            position_size=position_size,
            market_info=market_info,
            logger=lg,
            reduce_only=False # This is an entry order
        )

        # vi. Post-Trade Actions (Confirmation & Protection)
        if trade_order and trade_order.get('id'):
            order_id = trade_order['id']
            # Wait briefly for the position update on the exchange
            confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
            lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for position confirmation...")
            time.sleep(confirm_delay)

            # Confirm the position exists now
            lg.debug("Attempting to confirm open position after order placement...")
            confirmed_position = get_open_position(exchange, symbol, lg)

            if confirmed_position:
                lg.info(f"{NEON_GREEN}Position Confirmed after trade!{RESET}")
                try:
                    # Get actual entry price if available, fallback to estimate
                    entry_price_actual_str = confirmed_position.get('entryPrice')
                    entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close_price
                    if entry_price_actual <= 0: entry_price_actual = latest_close_price # Further fallback
                    lg.info(f"  Actual/Estimated Entry Price: {entry_price_actual.normalize()}")

                    # Re-calculate TP/SL based on actual/confirmed entry price for potentially better accuracy
                    lg.debug("Recalculating TP/SL based on confirmed entry for protection setting...")
                    protection_tp, protection_sl = signal_generator.calculate_initial_tp_sl(
                        entry_price_actual, signal, current_atr_value, market_info, exchange
                    )

                    if protection_sl is None:
                        # This is critical - position is open but SL failed
                        lg.error(f"{NEON_RED}{BRIGHT}CRITICAL: Position opened, but failed to recalculate SL ({protection_sl}) for protection! Position is unprotected! Manual intervention required!{RESET}")
                    else:
                        # Set Protection (TSL or Fixed SL/TP)
                        protection_config = config["protection"]
                        protection_set_success = False
                        if protection_config.get("enable_trailing_stop", True):
                            lg.info(f"Setting Initial Trailing Stop Loss (based on entry {entry_price_actual.normalize()})...")
                            protection_set_success = set_trailing_stop_loss(
                                exchange=exchange, symbol=symbol, market_info=market_info,
                                position_info=confirmed_position, config=config, logger=lg,
                                take_profit_price=protection_tp # Pass recalculated TP
                            )
                        elif not protection_config.get("enable_trailing_stop", True) and (protection_sl or protection_tp):
                            lg.info(f"Setting Initial Fixed Stop Loss / Take Profit (based on entry {entry_price_actual.normalize()})...")
                            protection_set_success = _set_position_protection(
                                exchange=exchange, symbol=symbol, market_info=market_info,
                                position_info=confirmed_position, logger=lg,
                                stop_loss_price=protection_sl,
                                take_profit_price=protection_tp
                            )
                        else:
                            lg.info("No protection (TSL or Fixed SL/TP) enabled in config.")
                            protection_set_success = True # No protection needed, so considered "successful"

                        # Log final outcome
                        if protection_set_success:
                            lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY SEQUENCE COMPLETE ({symbol} {signal}) - Position Opened & Protection Set ==={RESET}")
                        else:
                            lg.error(f"{NEON_RED}{BRIGHT}=== ENTRY SEQUENCE FAILED ({symbol} {signal}) - POSITION OPENED, BUT FAILED TO SET PROTECTION! MANUAL INTERVENTION REQUIRED! ==={RESET}")

                except Exception as post_trade_err:
                    lg.error(f"{NEON_RED}Error during post-trade setup (protection setting): {post_trade_err}{RESET}", exc_info=True)
                    lg.warning(f"{NEON_YELLOW}Position likely opened for {symbol}, but protection setting failed. Manual check required!{RESET}")
            else:
                # This is problematic - order was placed but position not found after delay
                lg.error(f"{NEON_RED}Order {order_id} placed, but FAILED TO CONFIRM open position after {confirm_delay}s delay! Manual check required! Possible fill issue or API delay.{RESET}")
        else:
            # Order placement failed
            lg.error(f"{NEON_RED}=== ENTRY SEQUENCE FAILED ({symbol} {signal}) - Order placement failed. No position opened. ===")

    # --- Scenario 2: Existing Position -> Consider Exit or Manage ---
    elif open_position:
        position_side = open_position['side']
        position_size_dec = open_position.get('size_decimal') # Should be present from get_open_position
        if position_size_dec is None:
             lg.error(f"Cannot manage position for {symbol}: Missing 'size_decimal'. Skipping management.")
             return # Cannot proceed without size

        lg.info(f"Signal: {signal} | Current Position: {position_side.upper()} (Size: {position_size_dec.normalize()})")

        # Check if signal indicates closing the current position
        exit_signal_triggered = (signal == "EXIT_LONG" and position_side == 'long') or \
                                (signal == "EXIT_SHORT" and position_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** Signal: {signal}! Initiating Close Sequence for {position_side} position... ***{RESET}")
            try:
                # Determine side needed to close (opposite of current position)
                close_order_side_signal = "EXIT_LONG" if position_side == 'long' else "EXIT_SHORT" # Map back to signal type
                # Size to close is the absolute value of the current position size
                size_to_close = abs(position_size_dec)

                if size_to_close <= 0:
                    lg.warning(f"Attempting to close {symbol} position, but size is zero or negative ({position_size_dec}). Assuming already closed or error in position data.")
                    return

                lg.info(f"{BRIGHT}===> Placing {signal} Market Order (Reduce Only) | Size: {size_to_close.normalize()} <==={RESET}")
                close_order = place_trade(
                    exchange=exchange,
                    symbol=symbol,
                    trade_signal=close_order_side_signal, # Use the signal type for place_trade
                    position_size=size_to_close,
                    market_info=market_info,
                    logger=lg,
                    reduce_only=True # Ensure this is a closing order
                )

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) placed successfully for {symbol}.{RESET}")
                    # Position should be closed or reducing. No further management needed this cycle.
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol}. Manual check required!{RESET}")

            except Exception as close_err:
                lg.error(f"{NEON_RED}Error occurred during position closing attempt: {close_err}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual position close may be required for {symbol}!{RESET}")

        else:
            # Signal allows holding the position. Perform management tasks (BE, TSL checks).
            lg.debug(f"Signal ({signal}) allows holding {position_side} position. Performing position management checks...")
            protection_config = config["protection"]

            # Extract current protection levels from position info safely
            try:
                is_tsl_active_on_exchange = open_position.get('trailingStopLoss') is not None and Decimal(str(open_position['trailingStopLoss'])) > 0
            except (ValueError, InvalidOperation, TypeError):
                is_tsl_active_on_exchange = False
            try:
                current_sl_price = Decimal(str(open_position['stopLossPrice'])) if open_position.get('stopLossPrice') and str(open_position['stopLossPrice']) != '0' else None
            except (ValueError, InvalidOperation, TypeError):
                current_sl_price = None
            try:
                current_tp_price = Decimal(str(open_position['takeProfitPrice'])) if open_position.get('takeProfitPrice') and str(open_position['takeProfitPrice']) != '0' else None
            except (ValueError, InvalidOperation, TypeError):
                current_tp_price = None
            try:
                entry_price = Decimal(str(open_position['entryPrice'])) if open_position.get('entryPrice') else None
            except (ValueError, InvalidOperation, TypeError):
                entry_price = None

            lg.debug(f"  Current Protection State: TSL Active={is_tsl_active_on_exchange}, SL={current_sl_price}, TP={current_tp_price}")

            # --- Break-Even Logic ---
            be_enabled = protection_config.get("enable_break_even", True)
            # Check BE only if enabled, TSL is *not* active, and we have necessary data
            if be_enabled and not is_tsl_active_on_exchange and entry_price and current_atr_value > 0 and price_for_checks > 0:
                lg.debug(f"Checking Break-Even condition...")
                lg.debug(f"  Entry={entry_price.normalize()}, Current Price={price_for_checks.normalize()}, ATR={current_atr_value.normalize()}")
                try:
                    be_trigger_atr_multiple = Decimal(str(protection_config["break_even_trigger_atr_multiple"]))
                    be_offset_ticks = int(protection_config["break_even_offset_ticks"])
                    price_tick_size = Decimal(str(market_info['precision']['price']))

                    if be_trigger_atr_multiple <= 0 or be_offset_ticks < 0 or price_tick_size <= 0:
                        raise ValueError("Invalid Break-Even parameters in config or market info")

                    # Calculate profit in terms of ATR multiples
                    profit_pips = (price_for_checks - entry_price) if position_side == 'long' else (entry_price - price_for_checks)
                    profit_in_atr = profit_pips / current_atr_value if current_atr_value > 0 else Decimal(0)

                    lg.debug(f"  Profit: {profit_pips.normalize()}, Profit in ATRs: {profit_in_atr:.3f}")
                    lg.debug(f"  BE Trigger ATR Multiple: {be_trigger_atr_multiple.normalize()}")

                    # Check if profit target for BE is reached
                    if profit_in_atr >= be_trigger_atr_multiple:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}Break-Even profit target REACHED! (Profit ATRs {profit_in_atr:.3f} >= {be_trigger_atr_multiple}){RESET}")

                        # Calculate the break-even SL price (entry + offset)
                        be_offset_value = price_tick_size * Decimal(str(be_offset_ticks))
                        be_sl_price_raw = (entry_price + be_offset_value) if position_side == 'long' else (entry_price - be_offset_value)

                        # Format the BE SL price
                        be_sl_price_formatted = None
                        try:
                            fmt_str = exchange.price_to_precision(symbol, float(be_sl_price_raw))
                            be_sl_price_formatted = Decimal(fmt_str)
                            if be_sl_price_formatted <= 0: raise ValueError("Formatted BE SL non-positive")
                        except Exception as fmt_err:
                             lg.error(f"Failed to format BE SL price {be_sl_price_raw}: {fmt_err}")

                        if be_sl_price_formatted:
                            lg.debug(f"  Calculated BE Stop Loss Price: {be_sl_price_formatted.normalize()} (Offset: {be_offset_ticks} ticks)")
                            # Check if the new BE SL is better than the current SL (if any)
                            should_update_sl = False
                            if current_sl_price is None:
                                should_update_sl = True
                                lg.info("  Current SL is not set. Setting SL to Break-Even.")
                            elif (position_side == 'long' and be_sl_price_formatted > current_sl_price) or \
                                 (position_side == 'short' and be_sl_price_formatted < current_sl_price):
                                should_update_sl = True
                                lg.info(f"  New BE SL {be_sl_price_formatted} is better than current SL {current_sl_price}. Updating.")
                            else:
                                lg.debug(f"  Current SL {current_sl_price} is already at or better than calculated BE SL {be_sl_price_formatted}. No update needed.")

                            # If update is needed, call the protection function
                            if should_update_sl:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even at {be_sl_price_formatted.normalize()} ***{RESET}")
                                # Use _set_position_protection to set only the SL, keeping existing TP if any
                                be_set_success = _set_position_protection(
                                    exchange=exchange, symbol=symbol, market_info=market_info,
                                    position_info=open_position, logger=lg,
                                    stop_loss_price=be_sl_price_formatted,
                                    take_profit_price=current_tp_price # Keep existing TP
                                    # Ensure TSL params are None
                                )
                                if be_set_success:
                                    lg.info(f"{NEON_GREEN}Break-Even Stop Loss set successfully.{RESET}")
                                else:
                                    lg.error(f"{NEON_RED}Failed to set Break-Even Stop Loss via API.{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Break-Even triggered, but calculated BE SL price ({be_sl_price_raw} -> {be_sl_price_formatted}) is invalid.{RESET}")
                    else:
                        lg.debug("Break-Even profit target not yet reached.")

                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during Break-Even check logic: {be_err}{RESET}", exc_info=True)
            elif be_enabled:
                lg.debug(f"Break-Even check skipped: {'TSL is active on exchange' if is_tsl_active_on_exchange else 'Missing required data (entry, ATR, price)'}.")
            else:
                lg.debug("Break-Even check skipped: Disabled in configuration.")

            # --- TSL Setup/Recovery (if BE didn't run or wasn't enabled/triggered) ---
            # Check if TSL is enabled in config but not active on the exchange
            tsl_enabled_in_config = protection_config.get("enable_trailing_stop", True)
            if tsl_enabled_in_config and not is_tsl_active_on_exchange and entry_price and current_atr_value > 0:
                 lg.warning(f"{NEON_YELLOW}Trailing Stop is enabled in config but not detected as active on the exchange. Attempting TSL setup/recovery...{RESET}")
                 # Re-calculate TP based on current state (might be None)
                 # Use entry_price and current ATR for TSL calculation
                 tp_for_tsl_setup, _ = signal_generator.calculate_initial_tp_sl(
                     entry_price, position_side.upper(), current_atr_value, market_info, exchange
                 )
                 # Attempt to set the TSL
                 tsl_setup_success = set_trailing_stop_loss(
                     exchange=exchange, symbol=symbol, market_info=market_info,
                     position_info=open_position, config=config, logger=lg,
                     take_profit_price=tp_for_tsl_setup # Use recalculated TP
                 )
                 if tsl_setup_success:
                     lg.info(f"{NEON_GREEN}Trailing Stop Loss setup/recovery successful.{RESET}")
                 else:
                     lg.error(f"{NEON_RED}Trailing Stop Loss setup/recovery failed.{RESET}")
            elif tsl_enabled_in_config and is_tsl_active_on_exchange:
                 lg.debug("Trailing Stop is enabled and already active on the exchange. No action needed.")
            elif not tsl_enabled_in_config:
                 lg.debug("Trailing Stop setup/recovery skipped: Disabled in configuration.")
            else: # TSL enabled, but missing data
                 lg.debug("Trailing Stop setup/recovery skipped: Missing required data (entry or ATR).")

    # --- Scenario 3: No Position and HOLD Signal ---
    elif open_position is None and signal == "HOLD":
         lg.info("Signal is HOLD, no existing position. No trading action taken.")

    # --- Scenario 4: Unhandled Signal/Position Combination (Should not happen) ---
    else:
        lg.error(f"Unhandled combination: Signal='{signal}', Position='{open_position.get('side') if open_position else None}'. No action taken.")


    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"{BRIGHT}---=== Cycle End: {symbol} (Duration: {cycle_end_time - cycle_start_time:.2f}s) ===---{RESET}\n")

# --- Main Execution Function ---
def main() -> None:
    """
    Main function to initialize the bot, prompt user for symbol/timeframe,
    handle user confirmation, and run the main trading loop.
    """
    global CONFIG, QUOTE_CURRENCY # Allow main to update global config if needed (e.g., interval)

    # Initial log messages
    init_logger.info(f"{BRIGHT}--- Pyrmethus Volumatic OB Bot v1.1.5 Starting ---{RESET}")
    init_logger.info(f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    init_logger.info(f"Loaded Configuration:")
    init_logger.info(f"  Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"  Trading Enabled: {CONFIG.get('enable_trading', False)}")
    init_logger.info(f"  Sandbox Mode: {CONFIG.get('use_sandbox', True)}")
    try:
        # Log versions of key libraries
        init_logger.info(f"Python Version: {os.sys.version.split()[0]}")
        init_logger.info(f"CCXT Version: {ccxt.__version__}")
        init_logger.info(f"Pandas Version: {pd.__version__}")
        init_logger.info(f"Pandas-TA Version: {getattr(ta, 'version', 'N/A')}") # Handle if version attribute changes
        init_logger.info(f"NumPy Version: {np.__version__}")
        init_logger.info(f"Requests Version: {requests.__version__}")
    except Exception as e:
        init_logger.warning(f"Could not retrieve library versions: {e}")

    # --- User Confirmation for Trading ---
    if CONFIG.get("enable_trading", False):
        init_logger.warning(f"\n{NEON_YELLOW}{BRIGHT}╔════════════════════════════════════════════╗")
        init_logger.warning(f"║ {Fore.RED}{BRIGHT}!!! LIVE TRADING IS ENABLED !!!{RESET}{NEON_YELLOW}{BRIGHT}          ║")
        init_logger.warning(f"╚════════════════════════════════════════════╝{RESET}")
        mode_str = f"{NEON_RED}LIVE (REAL FUNDS AT RISK)" if not CONFIG.get('use_sandbox') else f"{NEON_GREEN}SANDBOX (Testnet)"
        init_logger.warning(f"Trading Mode: {mode_str}{RESET}")

        # Display key protection settings for review
        prot_cfg = CONFIG.get("protection", {})
        risk_pct = CONFIG.get('risk_per_trade', 0) * 100
        leverage = CONFIG.get('leverage', 0)
        init_logger.warning(f"{BRIGHT}--- Review Key Trading Settings ---{RESET}")
        init_logger.warning(f"  Risk Per Trade: {risk_pct:.2f}%")
        init_logger.warning(f"  Leverage: {leverage}x")
        init_logger.warning(f"  Trailing Stop (TSL): {'ENABLED' if prot_cfg.get('enable_trailing_stop') else 'DISABLED'}")
        if prot_cfg.get('enable_trailing_stop'):
             init_logger.warning(f"    Callback Rate: {prot_cfg.get('trailing_stop_callback_rate', 0):.3%}")
             init_logger.warning(f"    Activation Pct: {prot_cfg.get('trailing_stop_activation_percentage', 0):.3%}")
        init_logger.warning(f"  Break Even (BE): {'ENABLED' if prot_cfg.get('enable_break_even') else 'DISABLED'}")
        if prot_cfg.get('enable_break_even'):
             init_logger.warning(f"    Trigger ATRs: {prot_cfg.get('break_even_trigger_atr_multiple', 0)}")
             init_logger.warning(f"    Offset Ticks: {prot_cfg.get('break_even_offset_ticks', 0)}")
        init_logger.warning(f"  Initial SL ATR Mult: {prot_cfg.get('initial_stop_loss_atr_multiple', 0)}")
        tp_mult = prot_cfg.get('initial_take_profit_atr_multiple', 0)
        init_logger.warning(f"  Initial TP ATR Mult: {tp_mult} {'(DISABLED)' if tp_mult == 0 else ''}")

        # Confirmation prompt
        try:
            input(f"\n{BRIGHT}>>> Press {NEON_GREEN}Enter{RESET}{BRIGHT} to confirm these settings and START TRADING, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to ABORT... {RESET}")
            init_logger.info("User confirmed settings. Proceeding with trading enabled.")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup via Ctrl+C during confirmation.")
            print(f"\n{NEON_YELLOW}Bot startup aborted by user.{RESET}")
            logging.shutdown()
            return # Exit gracefully
    else:
        init_logger.info(f"{NEON_YELLOW}Trading is disabled in config.json. Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange ---
    init_logger.info("Initializing CCXT exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"Exchange initialization failed. Cannot continue. Exiting.")
        logging.shutdown()
        return
    init_logger.info(f"Exchange '{exchange.id}' initialized successfully.")

    # --- Get Target Symbol and Market Info ---
    target_symbol: Optional[str] = None
    market_info: Optional[Dict] = None
    while target_symbol is None:
        try:
            # Prompt user for the trading symbol
            symbol_input = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT:USDT, ETH/USD:ETH): {RESET}").strip().upper()
            if not symbol_input:
                continue # Ask again if input is empty

            init_logger.info(f"Validating symbol '{symbol_input}' and fetching market info...")
            m_info = get_market_info(exchange, symbol_input, init_logger)

            if m_info:
                 target_symbol = m_info['symbol'] # Use the standardized symbol from CCXT
                 market_info = m_info
                 init_logger.info(f"Symbol Validated: {NEON_GREEN}{target_symbol}{RESET}")
                 init_logger.info(f"  Market Type: {market_info.get('contract_type_str', 'Unknown')}")
                 # Critical check for precision before proceeding
                 if market_info.get('precision', {}).get('price') is None or market_info.get('precision', {}).get('amount') is None:
                      init_logger.critical(f"{NEON_RED}CRITICAL: Market '{target_symbol}' is missing essential price or amount precision information in its market data. Cannot trade safely. Exiting.{RESET}")
                      logging.shutdown()
                      return
                 break # Exit loop once valid symbol and info are obtained
            else:
                 init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' could not be validated or market info not found on {exchange.id}. Please try again.{RESET}")
                 init_logger.info("Common formats: BASE/QUOTE (spot), BASE/QUOTE:SETTLE (contracts)")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup via Ctrl+C during symbol input.")
            print(f"\n{NEON_YELLOW}Bot startup aborted by user.{RESET}")
            logging.shutdown()
            return
        except Exception as e:
            # Catch unexpected errors during input/validation
            init_logger.error(f"An error occurred during symbol input/validation: {e}", exc_info=True)
            # Allow user to retry

    # --- Get Timeframe ---
    selected_interval: Optional[str] = None
    while selected_interval is None:
        default_interval = CONFIG.get('interval', '5') # Get default from loaded config
        interval_input = input(f"{NEON_YELLOW}Enter timeframe {VALID_INTERVALS} (default: {default_interval}): {RESET}").strip()

        if not interval_input:
            interval_input = default_interval
            init_logger.info(f"No input provided. Using default timeframe: {interval_input}")

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             # Update the interval in the global CONFIG dictionary for the current run
             CONFIG["interval"] = selected_interval
             init_logger.info(f"Using timeframe: {selected_interval} (CCXT mapping: {CCXT_INTERVAL_MAP[selected_interval]})")
             break
        else:
             init_logger.error(f"{NEON_RED}Invalid timeframe '{interval_input}'. Please choose from: {VALID_INTERVALS}{RESET}")

    # --- Setup Symbol-Specific Logger & Strategy Instances ---
    # Use the validated target_symbol for the logger name
    symbol_logger = setup_logger(target_symbol)
    symbol_logger.info(f"{BRIGHT}==========================================================={RESET}")
    symbol_logger.info(f"{BRIGHT}   Starting Trading Loop for: {target_symbol} (TF: {CONFIG['interval']})")
    symbol_logger.info(f"{BRIGHT}==========================================================={RESET}")
    symbol_logger.info(f"Trading Enabled: {CONFIG['enable_trading']}, Sandbox Mode: {CONFIG['use_sandbox']}")
    # Log key settings again in the symbol-specific logger
    prot_cfg = CONFIG.get("protection", {})
    symbol_logger.info(f"Key Settings: Risk={CONFIG.get('risk_per_trade', 0):.2%}, Leverage={CONFIG.get('leverage', 0)}x, TSL={'ON' if prot_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if prot_cfg.get('enable_break_even') else 'OFF'}")

    try:
        # Initialize the strategy engine and signal generator for this symbol
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_err:
        symbol_logger.critical(f"Failed to initialize Strategy Engine or Signal Generator: {engine_err}. Exiting.", exc_info=True)
        logging.shutdown()
        return

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}Entering main trading loop... Press Ctrl+C to stop gracefully.{RESET}")
    loop_count = 0
    try:
        while True:
            loop_start_time = time.time()
            loop_count += 1
            symbol_logger.debug(f">>> Loop #{loop_count} Start | Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")

            try:
                # --- Execute Core Analysis and Trading Logic ---
                analyze_and_trade_symbol(
                    exchange,
                    target_symbol,
                    CONFIG,
                    symbol_logger,
                    strategy_engine,
                    signal_generator,
                    market_info
                )
                # ---------------------------------------------

            # --- Handle Specific Loop-Level Exceptions ---
            except ccxt.RateLimitExceeded as e:
                symbol_logger.warning(f"Rate limit hit during main loop: {e}. Waiting 60 seconds...")
                time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e:
                symbol_logger.error(f"Network error during main loop: {e}. Waiting {RETRY_DELAY_SECONDS * 3} seconds...")
                time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                # Critical authentication errors should stop the bot
                symbol_logger.critical(f"{NEON_RED}CRITICAL Authentication Error encountered during main loop: {e}. Stopping bot.{RESET}")
                break # Exit the main loop
            except ccxt.ExchangeNotAvailable as e:
                symbol_logger.error(f"Exchange currently unavailable: {e}. Waiting 60 seconds...")
                time.sleep(60)
            except ccxt.OnMaintenance as e:
                symbol_logger.error(f"Exchange is under maintenance: {e}. Waiting 5 minutes...")
                time.sleep(300)
            except ccxt.ExchangeError as e:
                # Catch other potentially recoverable exchange errors
                symbol_logger.error(f"Unhandled Exchange Error during main loop: {e}", exc_info=True)
                symbol_logger.warning("Waiting 10 seconds before next cycle...")
                time.sleep(10)
            except Exception as loop_err:
                # Catch any other unexpected errors in the main loop
                symbol_logger.error(f"{NEON_RED}Critical unexpected error in main loop: {loop_err}{RESET}", exc_info=True)
                symbol_logger.warning("Waiting 15 seconds before next cycle...")
                time.sleep(15)

            # --- Loop Delay ---
            loop_end_time = time.time()
            elapsed_seconds = loop_end_time - loop_start_time
            # Use loop delay from CONFIG, fallback to default constant
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_duration = max(0, loop_delay - elapsed_seconds)

            symbol_logger.debug(f"<<< Loop #{loop_count} End | Duration: {elapsed_seconds:.2f}s | Sleeping for: {sleep_duration:.2f}s...")
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt detected (Ctrl+C). Shutting down gracefully...")
    except Exception as critical_err:
        # Catch errors outside the inner try/except (e.g., during loop setup or shutdown)
        init_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR in outer loop/shutdown: {critical_err}{RESET}", exc_info=True)
        # Log to symbol logger too if it exists
        if 'symbol_logger' in locals() and symbol_logger:
            symbol_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR: {critical_err}{RESET}", exc_info=True)

    finally:
        # --- Shutdown Sequence ---
        shutdown_msg = f"--- Pyrmethus Bot ({target_symbol or 'N/A'}) Stopping ---"
        print(f"\n{NEON_YELLOW}{BRIGHT}{shutdown_msg}{RESET}")
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and symbol_logger:
            symbol_logger.info(shutdown_msg)

        # Close exchange connection (optional for sync, but good practice)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Closing CCXT exchange connection (if applicable)...")
                # exchange.close() # Note: Often not strictly necessary for synchronous CCXT usage
                init_logger.info("Exchange connection closed indication.")
            except Exception as close_err:
                init_logger.error(f"Error during exchange.close(): {close_err}")

        # Shutdown logging system
        print("Flushing and closing log handlers...")
        logging.shutdown()
        # Attempt manual handler closing as a safeguard (rarely needed)
        try:
            for logger_instance in logging.Logger.manager.loggerDict.values():
                if isinstance(logger_instance, logging.Logger):
                    for handler in logger_instance.handlers[:]:
                         try: handler.close(); logger_instance.removeHandler(handler)
                         except: pass # Ignore errors during handler close
            for handler in logging.getLogger().handlers[:]: # Root logger
                 try: handler.close(); logging.getLogger().removeHandler(handler)
                 except: pass
        except Exception as log_close_err:
            print(f"Error during manual log handler closing: {log_close_err}")

        print(f"{NEON_YELLOW}{BRIGHT}Bot stopped.{RESET}")

# --- Entry Point ---
if __name__ == "__main__":
    main()
