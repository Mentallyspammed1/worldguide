# --- START OF FILE volumatictrend1.0.4.py ---

# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.0.4: Fixes indentation, refines OB exit logic, improves logging, adds retries to protection setting.

# --- Core Libraries ---
import contextlib
import json
import logging
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, TypedDict
from zoneinfo import ZoneInfo  # Requires tzdata package

# import websocket # Requires websocket-client (Imported but unused, placeholder for potential future WS integration)
import ccxt  # Requires ccxt

# --- Dependencies (Install via pip) ---
import numpy as np  # Requires numpy
import pandas as pd  # Requires pandas
import pandas_ta as ta  # Requires pandas_ta
import requests  # Requires requests
from colorama import Fore, Style, init  # Requires colorama
from dotenv import load_dotenv  # Requires python-dotenv

# Note: requests automatically uses urllib3, no need for separate import unless customizing adapters/retries outside ccxt

# --- Initialize Environment and Settings ---
getcontext().prec = 28  # Set Decimal precision for calculations
init(autoreset=True)  # Initialize Colorama for colored console output
load_dotenv()  # Load environment variables from a .env file

# --- Constants ---
# API Credentials (Loaded from .env file)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# Configuration and Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# << IMPORTANT: Set TIMEZONE to your local timezone or preferred timezone for logs >>
# List of timezone names can be found here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
try:
    TIMEZONE = ZoneInfo("America/Chicago")  # Example: Use 'UTC' for Coordinated Universal Time
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# API Interaction Settings
MAX_API_RETRIES = 3  # Max retries for recoverable API errors (Network, 429, 5xx)
RETRY_DELAY_SECONDS = 5  # Base delay between retries (may increase for rate limits)
POSITION_CONFIRM_DELAY_SECONDS = 8  # Wait time after placing order before confirming position state
LOOP_DELAY_SECONDS = 15  # Min time between the end of one cycle and the start of the next

# Timeframe Settings
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Intervals supported by the bot logic
CCXT_INTERVAL_MAP = {  # Map bot intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling
# Ensure enough data for indicator lookbacks (adjust based on strategy needs)
# Strategy requires roughly max(VT_Length*2, VT_ATR_Period, VT_Vol_EMA_Length, PH_left+PH_right, PL_left+PL_right) + buffer
DEFAULT_FETCH_LIMIT = 750
MAX_DF_LEN = 2000  # Max rows to keep in DataFrame to manage memory

# Default Strategy/Indicator Parameters (can be overridden by config.json)
# Volumatic Trend Params
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 1000
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0  # Note: This is used for internal 'step' calculation in Pine, not core logic here.

# Order Block Params
DEFAULT_OB_SOURCE = "Wicks"  # "Wicks" or "Bodys"
DEFAULT_PH_LEFT = 10
DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10
DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True  # Should OBs extend horizontally until violated?
DEFAULT_OB_MAX_BOXES = 50  # Maximum number of *active* boxes to track

# Dynamically loaded from config: QUOTE_CURRENCY

# Neon Color Scheme for Logging Output
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)


# --- Configuration Loading ---
class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to redact sensitive API keys from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record and redacts API keys."""
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger instance for a given name (e.g., 'init' or symbol).
    Configures file logging (DEBUG level) and console logging (INFO level by default).
    Uses SensitiveFormatter to prevent API keys leaking into logs.

    Args:
        name: The name for the logger (used in log messages and filename).

    Returns:
        Configured logging.Logger instance.
    """
    # Sanitize name for filename and logger name
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Prevent adding handlers multiple times if logger already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)  # Set base level to capture all messages

    # File Handler - Logs everything (DEBUG level and above)
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        # Use standard formatter for file logs
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
             datefmt='%Y-%m-%d %H:%M:%S'  # Standard date format
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception:
        # Fallback to print error if file logging fails
        pass

    # Console Handler - Logs INFO level and above by default (less verbose)
    stream_handler = logging.StreamHandler()
    # Use timezone-aware datetime formatting for console output - NOTE: Formatter needs TZ info if %Z used
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'  # Standard date format for simplicity
    )
    # Set the formatter's default time zone for asctime
    logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()

    stream_handler.setFormatter(stream_formatter)
    # Set desired console log level (INFO for normal operation, DEBUG for detailed tracing)
    console_log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
    console_log_level = getattr(logging, console_log_level_str, logging.INFO)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False  # Prevent messages propagating to the root logger
    return logger


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively ensures that all keys from the default configuration exist
    in the loaded configuration, adding missing keys with default values.

    Args:
        config: The configuration dictionary loaded from the file.
        default_config: The default configuration dictionary.
        parent_key: String representing the path to the current key (for logging).

    Returns:
        A tuple containing:
          - An updated configuration dictionary with all default keys present.
          - A boolean indicating if any changes were made.
    """
    updated_config = config.copy()
    changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value
            changed = True
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default value: {default_value}{RESET}")
        # Recurse if both values are dictionaries
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            nested_updated_config, nested_changed = _ensure_config_keys(
                updated_config[key], default_value, full_key_path
            )
            if nested_changed:
                 updated_config[key] = nested_updated_config
                 changed = True
        # Optional: Add type validation if needed
        # elif not isinstance(updated_config.get(key), type(default_value)):
        #     init_logger.warning(f"Config: Type mismatch for key '{full_key_path}'. Expected {type(default_value)}, got {type(updated_config.get(key))}. Using default.")
        #     updated_config[key] = default_value
        #     changed = True
    return updated_config, changed


def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from a JSON file. Creates a default config file if
    it doesn't exist. Ensures all default keys are present in the loaded config
    and validates the 'interval' setting. Updates the file if keys were added
    or interval was corrected.

    Args:
        filepath: Path to the configuration JSON file.

    Returns:
        The loaded (and potentially updated) configuration dictionary.
    """
    default_config = {
        "interval": "5",  # Default analysis interval (e.g., "5" for 5 minutes)
        "retry_delay": RETRY_DELAY_SECONDS,  # Use constant defined above
        "fetch_limit": DEFAULT_FETCH_LIMIT,  # Use constant
        "orderbook_limit": 25,  # Depth for order book fetching (if used) - Currently unused in logic
        "enable_trading": False,  # << SAFETY: Default to False (dry run mode) >>
        "use_sandbox": True,     # << SAFETY: Default to True (use testnet) >>
        "risk_per_trade": 0.01,  # Risk percentage (e.g., 0.01 = 1% of balance)
        "leverage": 20,          # Desired leverage (must be supported by exchange/symbol)
        "max_concurrent_positions": 1,  # Max open positions for this script instance/symbol - Currently unused in logic
        "quote_currency": "USDT",  # Currency for balance/sizing (e.g., USDT, USDC)
        "loop_delay_seconds": LOOP_DELAY_SECONDS,  # Use constant
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,  # Use constant

        # --- Strategy Parameters (Volumatic Trend & OB) ---
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,  # Not directly used for signals/sizing, keep for completeness
            "ob_source": DEFAULT_OB_SOURCE,  # "Wicks" or "Bodys"
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            # Price must be within Factor *beyond* OB edge for signal
            "ob_entry_proximity_factor": 1.005,  # E.g., 1.005 means price can go 0.5% beyond edge for entry
            "ob_exit_proximity_factor": 1.001  # E.g., 1.001 means price can go 0.1% beyond edge for exit
        },

        # --- Position Protection Settings ---
        "protection": {
             "enable_trailing_stop": True,
             "trailing_stop_callback_rate": 0.005,  # Trailing distance as % of Activation Price (e.g., 0.5%)
             "trailing_stop_activation_percentage": 0.003,  # Activate TSL when profit is >= X * Entry Price (e.g., 0.3%)
             "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0,  # Move SL to BE when profit >= X * ATR
             "break_even_offset_ticks": 2,  # Place BE SL X ticks beyond entry (uses market tick size)
             "initial_stop_loss_atr_multiple": 1.8,  # ATR multiple for initial SL (used for sizing)
             "initial_take_profit_atr_multiple": 0.7  # ATR multiple for initial TP (optional fixed target)
        }
    }

    config_needs_saving = False
    loaded_config = {}

    if not os.path.exists(filepath):
        init_logger.warning(f"{NEON_YELLOW}Config file not found: {filepath}. Creating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            return default_config
        except OSError as e:
            init_logger.error(f"{NEON_RED}Error creating default config file {filepath}: {e}. Using default values.{RESET}")
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)

        # Ensure all default keys exist, preserving existing values
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True
            # Message logged inside _ensure_config_keys

        # Validate interval value after merging defaults
        interval_from_config = updated_config.get("interval")
        if interval_from_config not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid interval '{interval_from_config}' in config. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]
            config_needs_saving = True  # Mark for saving if interval was corrected

        # --- Basic Validation for Numeric Values ---
        # Ensure key numeric values are valid numbers. Convert strings if possible, else use default.
        numeric_keys_to_validate = {
            "risk_per_trade": (0, 1),  # Range: > 0 and <= 1 (allow up to 100% risk, though high)
            "leverage": (0, 200),      # Range: >= 0 (0 means no leverage set)
            "protection.initial_stop_loss_atr_multiple": (0, 100),  # Range: > 0
            "protection.initial_take_profit_atr_multiple": (0, 100),  # Range: >= 0
            "protection.trailing_stop_callback_rate": (0, 1),      # Range: > 0
            "protection.trailing_stop_activation_percentage": (0, 1),  # Range: >= 0
            "protection.break_even_trigger_atr_multiple": (0, 100),  # Range: > 0
            "protection.break_even_offset_ticks": (0, 1000),         # Range: >= 0
        }

        def validate_and_convert(cfg, key_path, valid_range):
            keys = key_path.split('.')
            value = cfg
            default_val = default_config
            try:
                for key in keys:
                    value = value.get(key)
                    default_val = default_val.get(key)
                    if value is None: return False  # Key not found, handled by _ensure_config_keys
            except (KeyError, AttributeError):
                 return False  # Path invalid

            is_changed = False
            original_value = value
            min_val, max_val = valid_range

            try:
                # Attempt conversion to Decimal first for precision
                dec_value = Decimal(str(value))
                # Check range (inclusive lower bound for some, exclusive for others)
                if key_path in ["risk_per_trade", "protection.initial_stop_loss_atr_multiple",
                                "protection.trailing_stop_callback_rate", "protection.break_even_trigger_atr_multiple"]:
                    if not (min_val < dec_value <= max_val):
                        raise ValueError(f"Value {dec_value} out of range ({min_val}, {max_val}] for {key_path}")
                else:  # >= min_val
                    if not (min_val <= dec_value <= max_val):
                        raise ValueError(f"Value {dec_value} out of range [{min_val}, {max_val}] for {key_path}")

                # If value was string, update config with Decimal/numeric type
                # Note: JSON doesn't support Decimal, so saving will convert back to string/float.
                # Keep the validated numeric type in memory.
                if isinstance(original_value, str):
                    # Decide whether to store as float or int based on default type
                    if isinstance(default_val, int):
                         cfg_to_update = cfg
                         for key in keys[:-1]: cfg_to_update = cfg_to_update[key]
                         cfg_to_update[keys[-1]] = int(dec_value)
                         is_changed = True
                    elif isinstance(default_val, float):
                         cfg_to_update = cfg
                         for key in keys[:-1]: cfg_to_update = cfg_to_update[key]
                         cfg_to_update[keys[-1]] = float(dec_value)  # Store as float for JSON compatibility if needed later
                         is_changed = True
            except (ValueError, InvalidOperation, TypeError):
                # Conversion failed or out of range, use default
                init_logger.warning(f"{NEON_YELLOW}Config: Invalid numeric value '{original_value}' for '{key_path}'. Using default: {default_val}{RESET}")
                cfg_to_update = cfg
                for key in keys[:-1]: cfg_to_update = cfg_to_update[key]
                cfg_to_update[keys[-1]] = default_val
                is_changed = True

            return is_changed

        any_numeric_corrected = False
        for key, val_range in numeric_keys_to_validate.items():
             if validate_and_convert(updated_config, key, val_range):
                 any_numeric_corrected = True

        if any_numeric_corrected:
            config_needs_saving = True
            init_logger.warning(f"{NEON_YELLOW}Config file values corrected based on expected numeric types/ranges.{RESET}")

        # Save the updated config back to the file if changes were made
        if config_needs_saving:
             try:
                 # Prepare config for saving (convert Decimals back to strings/floats for JSON)
                 config_to_save = json.loads(json.dumps(updated_config, default=str))  # Simple conversion back
                 with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except OSError as e:
                 init_logger.error(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error preparing/saving updated config: {save_err}{RESET}", exc_info=True)

        return updated_config

    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding config file {filepath}: {e}. Attempting to recreate default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            return default_config
        except OSError as e_create:
             init_logger.error(f"{NEON_RED}Error creating default config file after load error: {e_create}. Using default values.{RESET}")
             return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error loading config: {e}. Using default values.{RESET}", exc_info=True)
        return default_config


# --- Logger Setup (Instantiate the 'init' logger) ---
# Note: This needs to happen *after* constants are defined but *before* load_config is called if load_config uses logging
init_logger = setup_logger("init")  # Logger for initial setup phases


# --- Load Configuration Globally (Can be updated later if dynamic reloading is needed) ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object.
    Handles API keys, sandbox mode, rate limits, timeouts, market loading,
    and performs an initial connection test by fetching balance.

    Args:
        logger: The logger instance to use for messages.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',  # Assume linear contracts (USDT/USDC perpetuals) unless overridden
                'adjustForTimeDifference': True,  # Auto-adjust for clock skew
                # Increased timeouts for potentially slow API calls (in milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000,  # Allow longer for fetching candles
                # Bybit V5 Specific Options (if needed, check CCXT docs)
                # 'recvWindow': 10000, # Example if needed for timing issues
            }
        }
        # Explicitly use the Bybit class
        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        # Set sandbox mode based on config
        if CONFIG.get('use_sandbox', True):  # Default to sandbox if key is missing
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
             lg.warning(f"{NEON_RED}USING LIVE TRADING ENVIRONMENT{RESET}")

        # Load markets with retries
        lg.info(f"Loading markets for {exchange.id}...")
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                # reload=True ensures we get fresh market data on retries
                exchange.load_markets(reload=attempt > 0)
                if exchange.markets:
                    lg.info(f"Markets loaded successfully for {exchange.id} ({len(exchange.markets)} symbols).")
                    break  # Success
                else:
                    lg.warning(f"load_markets returned empty markets list (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error loading markets (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.critical(f"{NEON_RED}Max retries reached loading markets for {exchange.id}. Error: {e}. Exiting.{RESET}")
                    return None  # Critical failure
            except ccxt.AuthenticationError as e:
                 lg.critical(f"{NEON_RED}Authentication error loading markets: {e}. Check API keys.{RESET}")
                 return None
            except ccxt.ExchangeError as e:
                 lg.critical(f"{NEON_RED}Exchange error loading markets: {e}. Exiting.{RESET}")
                 return None
            except Exception as e:
                 lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                 return None
        if not exchange.markets:  # Double check if markets actually loaded after loop
             lg.critical(f"{NEON_RED}Market loading failed, exchange.markets is empty after retries. Exiting.{RESET}")
             return None

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Test connection and API keys by fetching balance
        lg.info(f"Attempting initial balance fetch (Quote Currency: {QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                 lg.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({QUOTE_CURRENCY} available: {balance_val.normalize()})")
            else:
                 # fetch_balance returning None after retries usually indicates a persistent issue
                 lg.critical(f"{NEON_RED}Initial balance fetch failed after retries. Check API key permissions, network, or account type.{RESET}")
                 # Decide if this is fatal. If trading is enabled, it likely is.
                 if CONFIG.get('enable_trading', False):
                      lg.critical(f"{NEON_RED}Trading enabled, but initial balance check failed. This is critical. Exiting.{RESET}")
                      return None
                 else:
                      lg.warning(f"{NEON_YELLOW}Trading disabled, but initial balance check failed. Proceeding, but trading functions will likely fail if enabled later.{RESET}")

        except ccxt.AuthenticationError as auth_err:
             lg.critical(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             lg.critical(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
             return None  # Authentication errors are typically fatal
        except Exception as balance_err:
             # Non-auth errors during initial fetch are warnings, but problematic if persistent
             lg.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Check API permissions/account type if trading fails.{RESET}", exc_info=True)
             # Proceed cautiously if trading is disabled, otherwise consider fatal?
             if CONFIG.get('enable_trading', False):
                  lg.critical(f"{NEON_RED}Trading enabled, initial balance fetch failed unexpectedly. Exiting.{RESET}")
                  return None

        return exchange

    except ccxt.AuthenticationError as e:
        lg.critical(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.critical(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        lg.critical(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)

    return None


# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a symbol using the exchange's ticker.
    Includes retries for network issues and rate limits, and fallbacks for price sources.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol (e.g., 'BTC/USDT').
        logger: Logger instance for messages.

    Returns:
        The current price as a Decimal, or None if fetching fails.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol}... (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            # lg.debug(f"Ticker data for {symbol}: {ticker}") # Uncomment for verbose debugging

            price = None
            # Prioritize 'last', then mid-price ('bid'+'ask')/2, then 'ask', then 'bid'
            # Check for None or empty string before attempting conversion
            last_price_str = ticker.get('last')
            bid_price_str = ticker.get('bid')
            ask_price_str = ticker.get('ask')

            # Helper to safely convert string to positive Decimal
            def safe_decimal(val_str, name):
                if val_str is not None and str(val_str).strip() != '':
                    try:
                        p = Decimal(str(val_str))
                        if p > Decimal('0'): return p
                        else: lg.debug(f"Ticker {name} price '{val_str}' is not positive."); return None
                    except (InvalidOperation, TypeError) as conv_err:
                        lg.warning(f"Invalid ticker {name} price format '{val_str}': {conv_err}"); return None
                return None

            # Try 'last' price
            price = safe_decimal(last_price_str, 'last')
            if price: lg.debug(f"Using 'last' price: {price.normalize()}")

            # Try bid/ask midpoint if 'last' failed or was invalid
            if price is None:
                bid = safe_decimal(bid_price_str, 'bid')
                ask = safe_decimal(ask_price_str, 'ask')
                if bid and ask:
                    if ask >= bid:  # Sanity check: ask >= bid
                        price = (bid + ask) / Decimal('2')
                        lg.debug(f"Using bid/ask midpoint: {price.normalize()} (Bid: {bid.normalize()}, Ask: {ask.normalize()})")
                    else: lg.warning(f"Invalid bid/ask values: Bid={bid.normalize()}, Ask={ask.normalize()}")
                else:
                    # Fallback to 'ask' price if midpoint failed
                    if ask:
                        price = ask; lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback: {price.normalize()}{RESET}")
                    # Fallback to 'bid' price if ask also failed
                    elif bid:
                        price = bid; lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback: {price.normalize()}{RESET}")

            # Check if a valid price was found
            if price is not None:
                return price
            else:
                lg.warning(f"Failed to get a valid positive price from ticker (Attempt {attempts + 1}). Ticker data: {ticker}")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5  # Wait longer for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS)  # Account for standard delay below
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            # Don't retry most exchange errors (e.g., bad symbol) immediately
            return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Don't retry unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch a valid current price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries, validation, and processing.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        timeframe: CCXT timeframe string (e.g., '1m', '5m', '1h').
        limit: Number of candles to fetch.
        logger: Logger instance.

    Returns:
        A pandas DataFrame with Decimal columns ['open', 'high', 'low', 'close', 'volume']
        and a DatetimeIndex (UTC), or an empty DataFrame on failure.
    """
    lg = logger
    if not exchange.has['fetchOHLCV']:
         lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
         return pd.DataFrame()

    ohlcv = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
            # Fetch OHLCV data: [timestamp, open, high, low, close, volume]
            # Bybit V5 params: {'category': 'linear', 'symbol': market_id}
            # CCXT's fetch_ohlcv often handles this automatically based on symbol/market info
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

            if ohlcv is not None and len(ohlcv) > 0:
                # Basic validation: Check if the last timestamp seems reasonably recent
                try:
                    last_ts_ms = ohlcv[-1][0]
                    last_ts = pd.to_datetime(last_ts_ms, unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    # Estimate interval duration (rough estimate if exchange doesn't provide parse_timeframe)
                    interval_seconds = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') and exchange.parse_timeframe(timeframe) else None
                    # Allow a lag, e.g., 5 intervals, or 300 seconds if interval duration unknown
                    max_allowed_lag_seconds = (interval_seconds * 5) if interval_seconds else 300
                    # Allow extra buffer for lower frequency timeframes
                    if timeframe in ['1d', '1w', '1M']: max_allowed_lag_seconds = max(max_allowed_lag_seconds, 3600)  # Allow 1 hour lag for daily+

                    lag_seconds = (now_utc - last_ts).total_seconds()
                    if lag_seconds < max_allowed_lag_seconds:
                         lg.debug(f"Received {len(ohlcv)} klines. Last timestamp: {last_ts} (Lag: {lag_seconds:.1f}s)")
                         break  # Success
                    else:
                         lg.warning(f"{NEON_YELLOW}Received {len(ohlcv)} klines, but last timestamp {last_ts} seems too old (Lag: {lag_seconds:.1f}s > {max_allowed_lag_seconds}s). Retrying...{RESET}")
                         # Potentially stale data, force retry
                         ohlcv = None  # Clear ohlcv to force retry
                except Exception as ts_err:
                    lg.warning(f"Error validating timestamp: {ts_err}. Proceeding cautiously.")
                    break  # Proceed even if timestamp check fails
            else:
                lg.warning(f"fetch_ohlcv returned None or empty list for {symbol} (Attempt {attempt + 1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempt < MAX_API_RETRIES:
                lg.warning(f"{NEON_YELLOW}Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors: {e}{RESET}")
                return pd.DataFrame()  # Return empty DF on persistent network failure
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines for {symbol}: {e}. Retrying in {wait_time}s... (Attempt {attempt + 1}){RESET}")
            time.sleep(wait_time)
        except ccxt.ExchangeError as e:
            lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
            # Depending on the error, might not be retryable (e.g., bad symbol)
            return pd.DataFrame()  # Return empty DF on unrecoverable exchange error
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines: {e}{RESET}", exc_info=True)
            return pd.DataFrame()  # Return empty DF on unexpected error

    if not ohlcv:
        lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
        return pd.DataFrame()

    # --- Data Processing ---
    try:
        # Ensure standard column names, handle potential missing 'turnover'
        # CCXT standard is [timestamp, open, high, low, close, volume]
        # Bybit can return [timestamp, open, high, low, close, volume, turnover]
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if len(ohlcv[0]) > 6: columns.append('turnover')

        # Create DataFrame using only the available columns
        df = pd.DataFrame(ohlcv, columns=columns[:len(ohlcv[0])])

        # Convert timestamp to datetime objects (UTC) and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric (using Decimal for price/volume precision)
        decimal_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in decimal_cols:
            if col in df.columns:
                # Convert to float first to handle various incoming types, then to Decimal
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Convert finite floats to Decimal, NaN stays NaN (will be handled by dropna)
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

        # Validate data: drop rows with NaNs in key price columns or non-positive close
        initial_len = len(df)
        # Drop rows where any required price column is NaN
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Ensure close price is positive after Decimal conversion
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN or non-positive volume if volume is used in strategy
        if 'volume' in df.columns:
             df.dropna(subset=['volume'], inplace=True)
             df = df[df['volume'] >= Decimal('0')]  # Allow zero volume, but not negative

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price/volume data for {symbol}.")

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} empty after cleaning.{RESET}")
             return pd.DataFrame()

        # Ensure data is sorted chronologically (fetch_ohlcv usually returns oldest first)
        df.sort_index(inplace=True)

        # Limit DataFrame size to prevent memory issues
        if len(df) > MAX_DF_LEN:
             lg.debug(f"Trimming DataFrame from {len(df)} to {MAX_DF_LEN} rows.")
             df = df.iloc[-MAX_DF_LEN:].copy()  # Use .copy() to avoid SettingWithCopyWarning

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df  # Return the processed DataFrame

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing kline data for {symbol}: {e}{RESET}", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on processing failure


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves market information for a symbol, including precision, limits,
    and contract type, with retries for network issues.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        logger: Logger instance.

    Returns:
        A dictionary containing market details, or None if the symbol is invalid
        or fetching fails after retries.
    """
    lg = logger
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            # Check if markets are loaded; reload if necessary or if symbol not found
            if not exchange.markets or symbol not in exchange.markets:
                 lg.info(f"Market info for {symbol} not loaded or missing. Reloading markets (Attempt {attempt + 1})...")
                 exchange.load_markets(reload=True)

            # Check again after potential reload
            if symbol not in exchange.markets:
                 # If still not found after reload, it's likely an invalid symbol
                 if attempt == 0: continue  # Allow one reload attempt before failing
                 lg.error(f"{NEON_RED}Market '{symbol}' still not found after reloading markets.{RESET}")
                 return None

            market = exchange.market(symbol)
            if market:
                # Enhance market info with derived flags for convenience
                market_type = market.get('type', 'unknown')  # e.g., spot, swap, future
                is_linear = market.get('linear', False)
                is_inverse = market.get('inverse', False)
                # Consider swap/future as contracts, check 'contract' flag as fallback
                is_contract = market.get('contract', False) or market_type in ['swap', 'future']
                contract_type = "Linear" if is_linear else "Inverse" if is_inverse else "Spot/Other"

                # Add derived flags to the market dictionary
                market['is_contract'] = is_contract
                market['contract_type_str'] = contract_type

                # Log key details
                # Use Decimal for displaying precision/limits for accuracy
                def format_market_val(val):
                    if val is None: return 'N/A'
                    try: return str(Decimal(str(val)).normalize())  # Normalize to remove trailing zeros
                    except Exception: return str(val)

                lg.debug(
                    f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract Type={contract_type}, "
                    f"Precision(Price/Amount): {format_market_val(market.get('precision', {}).get('price'))}/{format_market_val(market.get('precision', {}).get('amount'))}, "
                    f"Limits(Amount Min/Max): {format_market_val(market.get('limits', {}).get('amount', {}).get('min'))}/{format_market_val(market.get('limits', {}).get('amount', {}).get('max'))}, "
                    f"Limits(Cost Min/Max): {format_market_val(market.get('limits', {}).get('cost', {}).get('min'))}/{format_market_val(market.get('limits', {}).get('cost', {}).get('max'))}, "
                    f"Contract Size: {format_market_val(market.get('contractSize', 'N/A'))}"
                )
                # Perform basic validation of essential market info needed later
                if market.get('precision', {}).get('price') is None or \
                   market.get('precision', {}).get('amount') is None:
                    lg.error(f"{NEON_RED}Market info for {symbol} is missing essential precision data (price or amount). Trading may fail.{RESET}")
                    # Decide if this is fatal. For trading, it likely is.
                    # return None # Or allow proceeding but log critical warning
                if market.get('limits', {}).get('amount', {}).get('min') is None:
                    lg.warning(f"{NEON_YELLOW}Market info for {symbol} is missing minimum amount limit. Assuming 0, but orders might fail.{RESET}")

                return market  # Success
            else:
                 # Should not happen if symbol is in exchange.markets, but handle defensively
                 lg.error(f"{NEON_RED}Market dictionary unexpectedly None for '{symbol}' even though key exists in markets.{RESET}")
                 return None  # Treat as failure

        except ccxt.BadSymbol as e:
             lg.error(f"{NEON_RED}Symbol '{symbol}' not supported or invalid on {exchange.id}: {e}{RESET}")
             return None  # Bad symbol is not retryable
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
             if attempt < MAX_API_RETRIES:
                  lg.warning(f"{NEON_YELLOW}Network error getting market info for {symbol} (Attempt {attempt + 1}): {e}. Retrying...{RESET}")
                  time.sleep(RETRY_DELAY_SECONDS)
             else:
                  lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} after network errors: {e}{RESET}")
                  return None
        except ccxt.ExchangeError as e:
            lg.warning(f"{NEON_YELLOW}Exchange error getting market info for {symbol}: {e}. Retrying...{RESET}")
            # Some exchange errors might be temporary, allow retry
            if attempt < MAX_API_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} after exchange errors: {e}{RESET}")
                return None
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Unexpected errors usually not retryable

    return None  # Should only be reached if all retries fail


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the available balance for a specific currency, handling various Bybit V5
    API response structures and including retries.

    Args:
        exchange: Initialized CCXT exchange object.
        currency: The currency code to fetch the balance for (e.g., 'USDT').
        logger: Logger instance.

    Returns:
        The available balance as a Decimal, or None if fetching fails after retries.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None
            available_balance_str = None
            found_structure = False

            # --- Try fetching with specific Bybit V5 account types ---
            # UNIFIED often holds USDT for linear contracts, CONTRACT is for derivatives.
            # Add 'SPOT' if needed, but QUOTE_CURRENCY is usually for margin/contracts.
            account_types_to_try = ['UNIFIED', 'CONTRACT']
            for acc_type in account_types_to_try:
                 try:
                     lg.debug(f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency}...")
                     # Bybit V5 uses 'accountType' in params for fetch_balance
                     balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                     # lg.debug(f"Raw balance response (type {acc_type}): {balance_info}") # Verbose

                     # Structure 1: Standard CCXT (less common for Bybit V5 derivatives)
                     if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] structure: {available_balance_str} (Type: {acc_type})")
                         found_structure = True; break

                     # Structure 2: Bybit V5 Unified/Contract ('info' field)
                     elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             # Check accountType *if* returned, otherwise assume it's the requested type
                             account_type_in_response = account.get('accountType')
                             if (account_type_in_response is None or account_type_in_response == acc_type) and isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                     if coin_data.get('coin') == currency:
                                         # Prefer 'availableToWithdraw' or 'availableBalance' or 'walletBalance'
                                         free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free is not None:
                                             available_balance_str = str(free)
                                             lg.debug(f"Found balance via V5 info.result.list[].coin[] ({acc_type}): {available_balance_str}")
                                             found_structure = True; break
                                 if found_structure: break  # Exit inner loop if found
                         if found_structure: break  # Exit account type loop if found
                         lg.debug(f"Currency '{currency}' not found within V5 info.result.list structure for type '{acc_type}'.")
                     else:
                          lg.debug(f"V5 structure not found for type '{acc_type}'. Response structure: {balance_info.keys()}")  # Log keys found

                 except (ccxt.ExchangeError, ccxt.AuthenticationError) as e:
                     # Specific exchange errors for an account type might just mean it's not applicable
                     lg.debug(f"API error fetching balance for type '{acc_type}': {e}. Trying next type.")
                     continue  # Try next account type
                 except Exception as e:
                     lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}. Trying next type.", exc_info=True)
                     continue

            # --- Fallback: Try default fetch_balance (no params) ---
            # This might return a mix of account types, so need to parse carefully
            if not found_structure:
                 lg.debug(f"Fetching balance using default parameters for {currency} (fallback)...")
                 try:
                      balance_info = exchange.fetch_balance()
                      # lg.debug(f"Raw balance response (default): {balance_info}") # Verbose
                      # Check standard structure again
                      if currency in balance_info and balance_info[currency].get('free') is not None:
                         available_balance_str = str(balance_info[currency]['free'])
                         lg.debug(f"Found balance via standard ['{currency}']['free'] (default fetch): {available_balance_str}")
                         found_structure = True
                      # Check top-level 'free' dict (older CCXT versions?)
                      elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
                         available_balance_str = str(balance_info['free'][currency])
                         lg.debug(f"Found balance via top-level 'free' dict (default fetch): {available_balance_str}")
                         found_structure = True
                      # Check V5 structure again in default response - search all account types returned
                      elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                           for account in balance_info['info']['result']['list']:
                              if isinstance(account.get('coin'), list):
                                  for coin_data in account['coin']:
                                      if coin_data.get('coin') == currency:
                                          free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                          if free is not None:
                                              available_balance_str = str(free)
                                              acc_type_str = account.get('accountType', 'N/A')
                                              lg.debug(f"Found balance via V5 nested structure (default fetch, type {acc_type_str}): {available_balance_str}")
                                              found_structure = True; break  # Found balance for currency in *any* account type list
                                  if found_structure: break  # Exit inner account loop
                              if found_structure: break  # Exit list loop
                           if not found_structure: lg.debug(f"Currency '{currency}' not found within V5 nested structure from default fetch.")

                 except Exception as e:
                      lg.error(f"{NEON_RED}Failed to fetch balance using default parameters: {e}{RESET}", exc_info=True)
                      # Allow retry loop to handle this

            # --- Process the extracted balance string ---
            if found_structure and available_balance_str is not None:
                try:
                    final_balance = Decimal(available_balance_str)
                    if final_balance >= Decimal('0'):
                         lg.info(f"Available {currency} balance: {final_balance.normalize()}")
                         return final_balance  # Success
                    else:
                         lg.error(f"Parsed balance for {currency} is negative ({final_balance.normalize()}). Check account.")
                         # Treat negative balance as an issue, maybe retry? For now, raise error.
                         raise ccxt.ExchangeError(f"Negative balance detected for {currency}")
                except (InvalidOperation, TypeError) as e:
                    lg.error(f"Failed to convert balance string '{available_balance_str}' to Decimal for {currency}: {e}")
                    # Treat conversion failure as an issue, raise error to trigger retry
                    raise ccxt.ExchangeError(f"Balance conversion failed for {currency}")
            else:
                # If still no balance found after all attempts
                lg.error(f"{NEON_RED}Could not determine available balance for {currency} after checking known structures.{RESET}")
                # lg.debug(f"Last balance_info structure checked: {balance_info}") # Too verbose possibly
                # Raise error to trigger retry
                raise ccxt.ExchangeError(f"Balance not found for {currency} in response")

        # --- Retry Logic for Handled Exceptions ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance for {currency}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS)  # Account for standard delay below
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API keys/permissions.{RESET}")
             return None  # Auth errors are fatal, don't retry
        except ccxt.ExchangeError as e:
            # Includes errors raised internally above (negative balance, conversion fail, not found)
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance for {currency}: {e}. Retrying...{RESET}")
            bybit_code = getattr(e, 'code', None)
            # Bybit V5 specific balance errors? (e.g., Invalid accountType, Insufficient permissions)
            if bybit_code in [10001, 10002]:  # API key invalid/expired/no permission or IP restricted
                 lg.critical(f"{NEON_RED}Bybit API Key/Auth Error (Code {bybit_code}) during balance fetch: {e}. Stopping.{RESET}")
                 return None  # Treat as fatal
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error during balance fetch for {currency}: {e}{RESET}", exc_info=True)
            # Allow retry for unexpected errors, but log stack trace

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            # Increase delay slightly on subsequent retries
            time.sleep(RETRY_DELAY_SECONDS * (attempts + 1))
        else:
            lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None  # Return None after all retries fail

    return None  # Should not be reached, but satisfies static analysis


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an open position for the given symbol using fetch_positions.
    Handles Bybit V5 API specifics, parses position details robustly, and includes retries.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        logger: Logger instance.

    Returns:
        A dictionary containing details of the open position if found (standardized format),
        otherwise None.
    """
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts + 1})")
            positions: list[dict] = []

            # Bybit V5 requires category ('linear', 'inverse', 'spot') and often symbol ID
            market_id = None
            category = None
            try:
                market = exchange.market(symbol)
                market_id = market['id']
                # Determine category based on market info
                category = 'linear' if market.get('linear', False) else \
                           'inverse' if market.get('inverse', False) else \
                           'spot' if market.get('spot', False) else 'linear'  # Default to linear if unsure
                # Bybit v5 fetch_positions can filter by symbol and category
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Using fetch_positions with params: {params}")
                positions = exchange.fetch_positions([symbol], params=params)  # Pass symbol list and params
                # lg.debug(f"Fetched positions using fetch_positions([symbol], params): {positions}") # Verbose

            except ccxt.ArgumentsRequired:
                 # Fallback if exchange requires fetching all positions (unlikely for modern Bybit)
                 lg.warning("fetch_positions requires fetching all positions (exchange doesn't support single symbol filter). This may be slower.")
                 params = {'category': category or 'linear'}  # Use determined category if possible
                 all_positions = exchange.fetch_positions(params=params)
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching {symbol}.")
            except ccxt.ExchangeError as e:
                 # Handle specific Bybit "position not found" errors gracefully
                 # Bybit V5 code for position not found: 110025 ("Position not found")
                 no_pos_codes_v5 = [110025]
                 err_str = str(e).lower()
                 if (hasattr(e, 'code') and e.code in no_pos_codes_v5) or \
                    "position not found" in err_str or "no position found" in err_str or \
                    "invalid symbol" in err_str:  # Could be an error if symbol/category combination is wrong
                      lg.info(f"No position or invalid symbol/category for {symbol} (Exchange confirmation: {e}).")
                      return None  # Confirmed no position or invalid query
                 else:
                      # Re-raise other exchange errors to trigger retry logic
                      raise e
            # Let other exceptions (NetworkError, AuthError, etc.) propagate to outer handler

            active_position = None
            # Define a small threshold to consider a position size non-zero
            # Use market info for amount precision if possible, else a small default
            size_threshold = Decimal('1e-9')  # Very small default threshold
            try:
                market = exchange.market(symbol)  # Re-get market if not fetched above
                amount_precision_step_str = market.get('precision', {}).get('amount')
                if amount_precision_step_str:
                    # Threshold slightly smaller than the smallest step size
                    size_threshold = Decimal(str(amount_precision_step_str)) * Decimal('0.1')
            except Exception as market_err:
                lg.debug(f"Could not get market precision for position size threshold ({market_err}), using default {size_threshold}.")

            lg.debug(f"Using position size threshold: {size_threshold}")

            # Iterate through potentially multiple position entries (e.g., hedge mode)
            # In one-way mode (default for this bot), there should be at most one entry per symbol.
            for pos in positions:
                # --- Determine Position Size (robustly checking multiple fields) ---
                pos_size_str = None
                # Prefer 'info.size' from Bybit V5 response as it's the primary source
                if pos.get('info', {}).get('size') is not None:
                    pos_size_str = str(pos['info']['size'])
                elif pos.get('contracts') is not None:  # Standard CCXT field as fallback
                    pos_size_str = str(pos['contracts'])
                # contractSize is size *per contract*, not total position size, so not used here.

                if pos_size_str is None:
                    lg.debug(f"Skipping position entry, could not determine size: {pos.get('info', {})}")
                    continue

                try:
                    position_size = Decimal(pos_size_str)
                    # Check if absolute size exceeds the threshold
                    if abs(position_size) > size_threshold:
                        active_position = pos  # Found a potentially active position
                        lg.debug(f"Found potential active position entry for {symbol} with size {position_size.normalize()}. Details: {pos.get('info', {})}")
                        break  # Assume first non-zero position is the one we manage (for non-hedge mode)
                except (InvalidOperation, TypeError) as parse_err:
                     lg.warning(f"Could not parse position size '{pos_size_str}' to Decimal: {parse_err}. Skipping entry.")
                     continue

            if active_position:
                # --- Standardize Key Position Details ---
                # Make a copy to avoid modifying the original list item
                std_pos = active_position.copy()
                info_dict = std_pos.get('info', {})  # Convenience accessor for 'info' field

                # Size (ensure Decimal, prefer 'contracts' or 'size' from info)
                # Size was already parsed and validated above
                std_pos['size_decimal'] = position_size  # Store standardized Decimal size

                # Side ('long' or 'short') - Derive robustly
                side = std_pos.get('side')  # Standard CCXT field
                if side not in ['long', 'short']:
                    pos_side_v5 = info_dict.get('side', '').lower()  # Bybit V5 'Buy'/'Sell'
                    if pos_side_v5 == 'buy': side = 'long'
                    elif pos_side_v5 == 'sell': side = 'short'
                    # Fallback: derive from size if side is missing/ambiguous
                    elif std_pos['size_decimal'] > size_threshold: side = 'long'
                    elif std_pos['size_decimal'] < -size_threshold: side = 'short'
                    else:
                        lg.warning(f"Position size {std_pos['size_decimal'].normalize()} near zero or side ambiguous. Cannot determine side.")
                        return None  # Cannot reliably determine side
                std_pos['side'] = side  # Store standardized side

                # Entry Price
                # Try 'entryPrice' (standard CCXT) or 'avgPrice'/'entryPrice' from info
                entry_price_str = std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice')
                std_pos['entryPrice'] = entry_price_str  # Store standardized field (string)

                # Leverage
                leverage_str = std_pos.get('leverage') or info_dict.get('leverage')
                std_pos['leverage'] = leverage_str  # Store standardized field (string)

                # Liquidation Price
                liq_price_str = std_pos.get('liquidationPrice') or info_dict.get('liqPrice')
                std_pos['liquidationPrice'] = liq_price_str  # Store standardized field (string)

                # Unrealized PnL
                pnl_str = std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl')
                std_pos['unrealizedPnl'] = pnl_str  # Store standardized field (string)

                # --- Extract Protection Info (SL/TP/TSL from Bybit V5 info dict or standard fields) ---
                # We need to look at the raw 'info' dict for reliable V5 protection details
                sl_price_str = info_dict.get('stopLoss')  # Prefer 'info' field for V5 accuracy
                if not sl_price_str: sl_price_str = std_pos.get('stopLossPrice')  # Fallback to standard

                tp_price_str = info_dict.get('takeProfit')  # Prefer 'info' field
                if not tp_price_str: tp_price_str = std_pos.get('takeProfitPrice')

                tsl_distance_str = info_dict.get('trailingStop')  # Bybit V5: Distance/Value
                tsl_activation_str = info_dict.get('activePrice')  # Bybit V5: Activation price for TSL

                # Store extracted protection info back into the main dict using consistent keys
                # Store as strings for now, parsing happens when needed
                if sl_price_str is not None: std_pos['stopLossPrice'] = str(sl_price_str)
                if tp_price_str is not None: std_pos['takeProfitPrice'] = str(tp_price_str)
                if tsl_distance_str is not None: std_pos['trailingStopLoss'] = str(tsl_distance_str)
                if tsl_activation_str is not None: std_pos['tslActivationPrice'] = str(tsl_activation_str)

                # --- Log Formatted Position Info ---
                # Helper to format values for logging, using market precision if available
                def format_log_val(val_str, precision_type='price', default_prec=4):
                    # Handle None, empty strings, and explicit "0" which can be meaningful for SL/TP clearing
                    if val_str is None or str(val_str).strip() == '': return 'N/A'
                    str_val = str(val_str).strip()
                    if str_val == '0': return '0'  # Show explicit 0
                    try:
                        d_val = Decimal(str_val)
                        # Attempt to get market precision
                        prec = default_prec
                        market = None
                        with contextlib.suppress(Exception): market = exchange.market(symbol)

                        if market:
                            prec_val = market.get('precision', {}).get(precision_type)
                            if prec_val is not None:
                                try:
                                     prec_step_dec = Decimal(str(prec_val))
                                     if prec_step_dec == prec_step_dec.to_integral():
                                         prec = 0  # Integer step size means 0 decimal places
                                     else:
                                         prec = abs(prec_step_dec.normalize().as_tuple().exponent)
                                except Exception: pass  # Stick with default precision if parsing fails

                        # Format using determined precision and normalize
                        # Use quantize with ROUND_DOWN for display consistency
                        exponent = Decimal('1e-' + str(prec))
                        return str(d_val.quantize(exponent, rounding=ROUND_DOWN).normalize())

                    except (InvalidOperation, TypeError):
                        return str_val  # Fallback to original string representation

                entry_price_fmt = format_log_val(std_pos.get('entryPrice'), 'price')
                contracts_fmt = format_log_val(abs(std_pos['size_decimal']), 'amount')  # Show absolute size
                liq_price_fmt = format_log_val(std_pos.get('liquidationPrice'), 'price')
                leverage_fmt = format_log_val(std_pos.get('leverage'), 'price', 1) + 'x' if std_pos.get('leverage') else 'N/A'
                pnl_fmt = format_log_val(std_pos.get('unrealizedPnl'), 'price', 4)  # PnL often uses quote precision
                sl_price_fmt = format_log_val(std_pos.get('stopLossPrice'), 'price')
                tp_price_fmt = format_log_val(std_pos.get('takeProfitPrice'), 'price')
                tsl_dist_fmt = format_log_val(std_pos.get('trailingStopLoss'), 'price')  # TSL distance often uses price precision
                tsl_act_fmt = format_log_val(std_pos.get('tslActivationPrice'), 'price')

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={contracts_fmt}, Entry={entry_price_fmt}, Liq={liq_price_fmt}, "
                            f"Lev={leverage_fmt}, PnL={pnl_fmt}, SL={sl_price_fmt}, TP={tp_price_fmt}, "
                            f"TSL(Dist/Act): {tsl_dist_fmt}/{tsl_act_fmt}")
                # lg.debug(f"Full standardized position details for {symbol}: {std_pos}") # Can be very verbose
                return std_pos  # Success, return the standardized dictionary
            else:
                logger.info(f"No active open position found for {symbol} (checked {len(positions)} entries).")
                return None  # No non-zero position found

        # --- Retry Logic for Handled Exceptions ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching position for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time - RETRY_DELAY_SECONDS)  # Account for standard delay below
        except ccxt.AuthenticationError as e:
             lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Stopping checks.{RESET}")
             return None  # Fatal
        except ccxt.ExchangeError as e:
            # Specific errors like 'position not found' are handled inside the try block.
            # Retry other exchange errors.
            lg.warning(f"{NEON_YELLOW}Exchange error fetching position for {symbol}: {e}. Retrying...{RESET}")
            bybit_code = getattr(e, 'code', None)
            # Bybit V5 Specific Position Errors:
            # 110001: Unknown error (retry)
            # 110004: Account not unified/contract (check API key link) -> Fatal
            # 110013: Parameter error (wrong symbol ID or category?) -> Fatal if params are consistent
            if bybit_code in [110004]:
                 lg.critical(f"{NEON_RED}Bybit Account Error (Code {bybit_code}) during position fetch: {e}. Is API key linked to Unified or Derivatives account? Stopping checks.{RESET}")
                 return None  # Fatal
            elif bybit_code in [110013]:
                 lg.error(f"{NEON_RED}Bybit Parameter Error (Code {bybit_code}) fetching position: {e}. Check symbol/category. Stopping checks.{RESET}")
                 return None  # Likely Fatal
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching/processing positions for {symbol}: {e}{RESET}", exc_info=True)
            # Don't retry unexpected errors immediately, let outer loop handle maybe

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Increase delay on retries
        else:
            lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
            return None

    return None  # Should not be reached


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets leverage for a derivatives symbol using CCXT, handling Bybit V5 specifics.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        leverage: The desired integer leverage value.
        market_info: Market dictionary containing contract details.
        logger: Logger instance.

    Returns:
        True if leverage was set successfully or already correct, False otherwise.
    """
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract).")
        return True  # No action needed for non-contracts, considered success

    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol}: Invalid leverage ({leverage}). Must be > 0.")
        return False

    if not exchange.has.get('setLeverage'):
         lg.error(f"{NEON_RED}Exchange {exchange.id} does not support setLeverage via CCXT.{RESET}")
         return False  # Cannot proceed if function isn't available

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Attempting to set leverage for {symbol} to {leverage}x (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            params = {}
            # Bybit V5 requires setting buy and sell leverage separately for the symbol
            # It also requires category and market ID in the params
            if 'bybit' in exchange.id.lower():
                 market_id = market_info.get('id', symbol)
                 category = 'linear' if market_info.get('linear', True) else 'inverse'
                 params = {
                     'category': category,
                     'symbol': market_id,  # Bybit expects market ID here
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage),
                     'marginMode': 'ISOLATED'  # Assuming Isolated, add check/config if needed
                 }
                 lg.debug(f"Using Bybit V5 params for set_leverage: {params}")
                 # Note: For Cross margin, leverage might be set per-currency or ignored.
                 # Bybit V5: set_leverage works for Isolated margin.

            # CCXT's set_leverage signature: setLeverage(leverage, symbol=None, params={})
            # Use the market ID for the symbol param for clarity with V5
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"Set leverage raw response for {symbol}: {response}")  # Contains exchange-specific response

            # Bybit V5 specific success check or common indications
            ret_code = response.get('retCode')  # Bybit specific
            if ret_code is not None:
                 if ret_code == 0:
                     lg.info(f"{NEON_GREEN}Leverage set/requested successfully for {symbol} to {leverage}x (Bybit retCode 0).{RESET}")
                     return True  # Success based on Bybit code
                 elif ret_code == 110045:  # Bybit: Leverage not modified (already set)
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Bybit retCode 110045).{RESET}")
                     return True  # Already set is success
                 else:
                     # Log other Bybit errors via the exception handler below
                     # Raise specific exception to be caught below
                     raise ccxt.ExchangeError(f"Bybit API error setting leverage: {response.get('retMsg', 'Unknown error')} (Code: {ret_code})")
            else:
                 # If no specific retCode, assume success if no exception raised and response isn't obviously an error
                 lg.info(f"{NEON_GREEN}Leverage set/requested successfully for {symbol} to {leverage}x.{RESET}")
                 return True

        except ccxt.ExchangeError as e:
            err_str = str(e).lower()
            bybit_code = getattr(e, 'code', None)  # Try to extract Bybit code if available
            lg.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")

            # Bybit V5 Specific Error Codes for Leverage:
            # 110045: Leverage not modified (handled above)
            # 110028 / 110009 / 110055: Position/order exists, margin mode conflict (Isolated vs Cross)
            # 110044: Exceed risk limit (leverage too high for position size tier)
            # 110013: Parameter error (leverage value invalid/out of range)
            # 10001: API key issue
            # 10004: Sign check fail
            # 110043: Cannot set margin mode (e.g., if position exists)

            if bybit_code == 110045 or "leverage not modified" in err_str:
                 lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
                 return True  # Already set is considered success
            elif bybit_code in [110028, 110009, 110055, 110043] or "margin mode" in err_str or "position exists" in err_str:
                 lg.error(f"{NEON_YELLOW} >> Hint: Cannot change leverage/margin mode. Check Margin Mode (Isolated/Cross), open orders, or existing position for {symbol}.{RESET}")
                 return False  # Unrecoverable state for automated setting
            elif bybit_code == 110044 or "risk limit" in err_str:
                 lg.error(f"{NEON_YELLOW} >> Hint: Leverage {leverage}x may exceed risk limit tier for current/potential position size. Check Bybit Risk Limits documentation.{RESET}")
                 return False  # Configuration issue, not temporary
            elif bybit_code == 110013 or "parameter error" in err_str or "leverage value invalid" in err_str:
                 lg.error(f"{NEON_YELLOW} >> Hint: Leverage value {leverage}x might be invalid for {symbol} or other params incorrect. Check allowed range on Bybit.{RESET}")
                 return False  # Configuration issue
            # Allow retry for other potentially temporary exchange errors
            elif attempts >= MAX_API_RETRIES:
                 lg.error("Max retries reached for ExchangeError setting leverage.")
                 return False

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                 lg.error("Max retries reached for NetworkError setting leverage.")
                 return False
            lg.warning(f"{NEON_YELLOW}Network error setting leverage for {symbol} (Attempt {attempts + 1}): {e}. Retrying...")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)
            # Decide if unexpected errors should retry. For leverage, probably safer not to retry indefinitely.
            return False  # Treat unexpected errors as non-retryable for this function

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False  # Return False if all retries fail


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,  # e.g., 0.01 for 1%
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: dict,
    exchange: ccxt.Exchange,  # Needed for formatting amount
    logger: logging.Logger
) -> Decimal | None:
    """Calculates the position size based on account balance, risk percentage,
    stop-loss distance, contract specifications, and market limits/precision.

    Args:
        balance: Available account balance (in quote currency) as Decimal.
        risk_per_trade: Risk percentage per trade (e.g., 0.01 = 1%).
        initial_stop_loss_price: Calculated initial stop-loss price as Decimal.
        entry_price: Estimated or actual entry price as Decimal.
        market_info: Market dictionary from get_market_info.
        exchange: Initialized CCXT exchange object (for formatting).
        logger: Logger instance.

    Returns:
        The calculated and adjusted position size as a Decimal, or None if calculation fails.
    """
    lg = logger
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)  # e.g., USDT
    base_currency = market_info.get('base', 'BASE')  # e.g., BTC
    is_contract = market_info.get('is_contract', False)
    is_inverse = market_info.get('inverse', False)
    # Determine the unit for size (Contracts for derivatives, Base currency for Spot)
    size_unit = "Contracts" if is_contract else base_currency

    # --- Input Validation ---
    if balance is None or balance <= Decimal('0'):
        lg.error(f"Position sizing failed ({symbol}): Invalid balance ({balance}).")
        return None
    # risk_per_trade validated during config load/conversion
    risk_per_trade_dec = Decimal(str(risk_per_trade))

    if initial_stop_loss_price is None or initial_stop_loss_price <= Decimal('0') or \
       entry_price is None or entry_price <= Decimal('0'):
        lg.error(f"Position sizing failed ({symbol}): Invalid entry ({entry_price.normalize() if entry_price else 'N/A'}) or SL ({initial_stop_loss_price.normalize() if initial_stop_loss_price else 'N/A'}).")
        return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed ({symbol}): SL price cannot equal entry price.")
         return None
    # Market info essential keys checked in get_market_info
    amount_precision_step_str = market_info.get('precision', {}).get('amount')
    price_precision_step_str = market_info.get('precision', {}).get('price')
    if amount_precision_step_str is None or price_precision_step_str is None:
         lg.error(f"Position sizing failed ({symbol}): Missing amount or price precision in market info.")
         return None

    try:
        # --- Risk Amount ---
        risk_amount_quote = balance * risk_per_trade_dec
        lg.info(f"Position Sizing ({symbol}): Balance={balance.normalize()} {quote_currency}, Risk={risk_per_trade_dec:.2%}, RiskAmt={risk_amount_quote.normalize()} {quote_currency}")

        # --- Stop-Loss Distance ---
        sl_distance_price = abs(entry_price - initial_stop_loss_price)
        if sl_distance_price <= Decimal('0'):
             lg.error(f"Position sizing failed ({symbol}): SL distance is zero/negative ({sl_distance_price.normalize()}).")
             return None
        lg.info(f"  Entry={entry_price.normalize()}, SL={initial_stop_loss_price.normalize()}, SL Dist={sl_distance_price.normalize()}")

        # --- Contract Size ---
        # Value of 1 unit of 'amount' (1 contract or 1 base unit for spot)
        # For Bybit linear USDT/USDC contracts, contractSize is typically 1 (meaning 1 contract = 1 base unit value)
        # For Bybit inverse contracts (e.g., BTCUSD), contractSize is typically 1 (meaning 1 contract = 1 USD value of base) ??? No, it's usually base units (1 BTC).
        # Check CCXT market info: `market['contractSize']`
        contract_size_str = market_info.get('contractSize')
        if contract_size_str is None:
             contract_size = Decimal('1')  # Assume 1 if not specified (common for spot/basic futures)
             lg.debug(f"Contract size missing for {symbol}, assuming 1.")
        else:
            try:
                 contract_size = Decimal(str(contract_size_str))
                 if contract_size <= Decimal('0'): raise ValueError("Contract size must be positive")
            except Exception as e:
                 lg.warning(f"{NEON_YELLOW}Invalid contract size '{contract_size_str}' for {symbol}, using 1. Error: {e}{RESET}")
                 contract_size = Decimal('1')  # Fallback

        lg.info(f"  ContractSize={contract_size.normalize()}, Type={'Linear/Spot' if not is_inverse else 'Inverse'}")

        # --- Calculate Initial Size based on Risk and SL Distance ---
        calculated_size = Decimal('0')
        if not is_inverse:  # Linear Contract or Spot
             # Risk (Quote) = Size (Amount units) * SL_Distance_Price * ContractSize (Quote per Amount unit if not 1)
             # For Linear USDT Perps: Risk = Size_contracts * SL_Distance_Price * 1 (ContractSize)
             # Size (Contracts) = Risk_Amount_Quote / SL_Distance_Price
             if sl_distance_price > Decimal('0'):
                 # The risk per contract is the stop loss distance in price.
                 # If contractSize represents something else (e.g., base units per contract for a non-standard linear),
                 # the formula might need adjustment. Assuming standard linear/spot here.
                 calculated_size = risk_amount_quote / sl_distance_price
             else:
                 lg.error(f"Sizing failed ({symbol}): Linear/Spot SL distance is zero/negative."); return None

        else:  # Inverse Contract (e.g., BTC/USD)
             # Risk (Quote) = Size (Contracts) * ContractSize (Base units per contract) * |(1/Entry_Price) - (1/SL_Price)|
             # Size (Contracts) = Risk_Amount_Quote / (ContractSize * abs(1/EntryPrice - 1/SL_Price))
             lg.info(f"Inverse contract detected ({symbol}). Using specific inverse sizing formula.")
             if entry_price > Decimal('0') and initial_stop_loss_price > Decimal('0'):
                  try:
                     inverse_factor = abs(Decimal('1') / entry_price - Decimal('1') / initial_stop_loss_price)
                     if inverse_factor <= Decimal('0'):
                         lg.error(f"Sizing failed ({symbol}): Inverse factor calculation resulted in zero/negative value ({inverse_factor.normalize()}). Check entry/SL prices.")
                         return None

                     risk_per_contract_quote = contract_size * inverse_factor
                     if risk_per_contract_quote <= Decimal('0'):
                          lg.error(f"Sizing failed ({symbol}): Inverse contract risk per contract is zero/negative ({risk_per_contract_quote.normalize()})."); return None

                     calculated_size = risk_amount_quote / risk_per_contract_quote
                  except (ZeroDivisionError, InvalidOperation) as calc_err:
                     lg.error(f"Sizing failed ({symbol}): Error during inverse calculation ({calc_err}). Check entry/SL prices."); return None
             else:
                 lg.error(f"Sizing failed ({symbol}): Invalid entry/SL price for inverse contract calculation."); return None

        lg.info(f"  Initial Calculated Size = {calculated_size.normalize()} {size_unit}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})  # Limits on position size (in contracts or base currency)
        cost_limits = limits.get('cost', {})     # Limits on position value (in quote currency)
        market_info.get('precision', {})

        # Convert limits/precision strings to Decimal
        try:
            amount_precision_step = Decimal(str(amount_precision_step_str))
            Decimal(str(price_precision_step_str))
            min_amount = Decimal(str(amount_limits.get('min', '0')))  # Default min to 0 if missing
            max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')
            min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
            max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')
        except (InvalidOperation, TypeError) as conv_err:
            lg.error(f"Sizing failed ({symbol}): Error converting market limits/precision to Decimal: {conv_err}")
            return None

        adjusted_size = calculated_size
        # Apply Min/Max Amount Limits first
        if min_amount > Decimal('0') and adjusted_size < min_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} < min amount {min_amount.normalize()}. Adjusting to min amount.{RESET}")
             adjusted_size = min_amount
        if max_amount < Decimal('inf') and adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size.normalize()} > max amount {max_amount.normalize()}. Adjusting to max amount.{RESET}")
             adjusted_size = max_amount

        # Calculate Estimated Cost (in Quote Currency) for Cost Limit Check
        estimated_cost = Decimal('0')
        try:
            if entry_price > Decimal('0'):
                 if not is_inverse:  # Linear / Spot
                      # Cost = Size * EntryPrice * ContractSize (ContractSize is 1 for Linear USDT/USDC)
                      estimated_cost = adjusted_size * entry_price * contract_size
                 else:  # Inverse
                      # Cost (in Quote) = Size * ContractSize / EntryPrice (value in base * quote/base rate)
                      estimated_cost = (adjusted_size * contract_size) / entry_price
            else: lg.warning(f"Cannot estimate cost for {symbol} due to zero entry price.")  # Skip cost check?
        except Exception as cost_est_err:
             lg.error(f"Error estimating cost for size {adjusted_size.normalize()}: {cost_est_err}")
             # Cannot check cost limits if cost estimation fails
             min_cost, max_cost = Decimal('0'), Decimal('inf')  # Effectively disable cost limits

        lg.debug(f"  Size after Amount Limits: {adjusted_size.normalize()} {size_unit}")
        lg.debug(f"  Estimated Cost ({quote_currency}): {estimated_cost.normalize()}")

        # Apply Min/Max Cost Limits (adjusting size if needed)
        cost_adjusted = False
        if min_cost > Decimal('0') and estimated_cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost.normalize()} < min cost {min_cost.normalize()}. Attempting to increase size.{RESET}")
             required_size_for_min_cost = None
             try:
                 if entry_price > Decimal('0') and contract_size > Decimal('0'):
                     if not is_inverse:  # Linear / Spot
                         required_size_for_min_cost = min_cost / (entry_price * contract_size)
                     else:  # Inverse
                          required_size_for_min_cost = (min_cost * entry_price) / contract_size
             except Exception as cost_calc_err: lg.error(f"Error calculating required size for min cost: {cost_calc_err}")

             if required_size_for_min_cost is None or required_size_for_min_cost <= Decimal('0'):
                 lg.error(f"{NEON_RED}Cannot meet min cost {min_cost.normalize()}. Calculation failed or resulted in zero/negative size. Aborted.{RESET}"); return None
             lg.info(f"  Required size for min cost: {required_size_for_min_cost.normalize()}")

             if max_amount < Decimal('inf') and required_size_for_min_cost > max_amount:
                 lg.error(f"{NEON_RED}Cannot meet min cost {min_cost.normalize()} without exceeding max amount {max_amount.normalize()}. Aborted.{RESET}"); return None

             # Adjust size up to meet min cost, ensuring it doesn't drop below min amount itself
             adjusted_size = max(min_amount, required_size_for_min_cost)
             cost_adjusted = True

        elif max_cost < Decimal('inf') and estimated_cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Estimated cost {estimated_cost.normalize()} > max cost {max_cost.normalize()}. Reducing size.{RESET}")
             adjusted_size_for_max_cost = None
             try:
                 if entry_price > Decimal('0') and contract_size > Decimal('0'):
                     if not is_inverse:  # Linear / Spot
                         adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                     else:  # Inverse
                          adjusted_size_for_max_cost = (max_cost * entry_price) / contract_size
             except Exception as cost_calc_err: lg.error(f"Error calculating max size for max cost: {cost_calc_err}")

             if adjusted_size_for_max_cost is None or adjusted_size_for_max_cost <= Decimal('0'):
                  lg.error(f"{NEON_RED}Cannot reduce size to meet max cost {max_cost.normalize()}. Calculation failed or resulted in zero/negative size. Aborted.{RESET}"); return None
             lg.info(f"  Max size allowed by max cost: {adjusted_size_for_max_cost.normalize()}")

             # Reduce size to meet max cost, ensuring it doesn't fall below min amount
             adjusted_size = max(min_amount, min(adjusted_size, adjusted_size_for_max_cost))
             cost_adjusted = True

        if cost_adjusted:
             lg.info(f"  Size after Cost Limits: {adjusted_size.normalize()} {size_unit}")

        # --- Apply Amount Precision (Step Size) ---
        final_size = adjusted_size
        try:
            # Use ccxt's amount_to_precision which correctly handles step sizes.
            # Use TRUNCATE (equivalent to ROUND_DOWN) to ensure we don't exceed risk/cost limits due to rounding up.
            # amount_to_precision takes float, we pass the Decimal converted to float
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size))
            # Convert the formatted string back to Decimal
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision/step (via ccxt, typically truncated): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
        except Exception as fmt_err:
            # Fallback or error if ccxt's method fails - attempt manual rounding down
            lg.warning(f"{NEON_YELLOW}Error applying ccxt amount precision for {symbol}: {fmt_err}. Attempting manual rounding down.{RESET}", exc_info=True)
            try:
                 if amount_precision_step > Decimal('0'):
                      # Floor division to round down to the nearest step size multiple
                      final_size = (adjusted_size // amount_precision_step) * amount_precision_step
                      lg.info(f"Applied manual amount step size ({amount_precision_step.normalize()}): {adjusted_size.normalize()} -> {final_size.normalize()} {size_unit}")
                 else: raise ValueError("Manual step size must be positive")
            except Exception as manual_err:
                 lg.error(f"{NEON_RED}Invalid amount precision value '{amount_precision_step_str}' ({manual_err}). Using size adjusted for limits only.{RESET}", exc_info=True)
                 final_size = adjusted_size  # Use limit-adjusted size without precision formatting

        # --- Final Validation ---
        # Ensure final size is positive
        if final_size <= Decimal('0'):
            lg.error(f"{NEON_RED}Position size became zero/negative ({final_size.normalize()}) after adjustments. Aborted.{RESET}")
            return None
        # Ensure final size still meets minimum amount limit after truncation
        if min_amount > Decimal('0') and final_size < min_amount:
            # This can happen if min_amount itself is not a multiple of step size
            lg.error(f"{NEON_RED}Final size {final_size.normalize()} is below minimum amount {min_amount.normalize()} after precision formatting. Aborted.{RESET}")
            return None

        # Recalculate final cost and check against min_cost again (important after rounding down)
        final_cost = Decimal('0')
        try:
            if entry_price > Decimal('0'):
                 if not is_inverse: final_cost = final_size * entry_price * contract_size
                 else: final_cost = (final_size * contract_size) / entry_price
        except Exception: lg.warning("Could not recalculate final cost.")

        if min_cost > Decimal('0') and final_cost < min_cost:
             lg.debug(f"Final size {final_size.normalize()} results in cost {final_cost.normalize()} which is below min cost {min_cost.normalize()}.")
             # Attempt to bump size up by one step if it meets min cost and doesn't exceed max amount/cost
             try:
                 step_size = amount_precision_step
                 if step_size <= Decimal('0'): raise ValueError("Step size must be positive to bump.")

                 next_step_size = final_size + step_size

                 # Calculate cost of next step size
                 next_step_cost = Decimal('0')
                 if entry_price > Decimal('0'):
                     if not is_inverse: next_step_cost = next_step_size * entry_price * contract_size
                     else: next_step_cost = (next_step_size * contract_size) / entry_price

                 # Check if next step is valid
                 valid_next_step = True
                 if next_step_cost < min_cost: valid_next_step = False  # Still doesn't meet min cost
                 if max_amount < Decimal('inf') and next_step_size > max_amount: valid_next_step = False
                 if max_cost < Decimal('inf') and next_step_cost > max_cost: valid_next_step = False

                 if valid_next_step:
                      lg.warning(f"{NEON_YELLOW}Final size cost {final_cost.normalize()} < min cost {min_cost.normalize()}. Bumping size to next step {next_step_size.normalize()} ({size_unit}) to meet minimums.{RESET}")
                      final_size = next_step_size
                 else:
                      lg.error(f"{NEON_RED}Final size {final_size.normalize()} cost is below minimum, and next step size {next_step_size.normalize()} is invalid (violates min cost or max limits). Aborted.{RESET}")
                      return None
             except Exception as bump_err:
                  lg.error(f"{NEON_RED}Final size cost is below minimum. Error trying to bump size: {bump_err}. Aborted.{RESET}")
                  return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size.normalize()} {size_unit}{RESET}")
        return final_size

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,  # "BUY" or "SELL" (direction of the trade)
    position_size: Decimal,  # Absolute positive size to trade
    market_info: dict,
    logger: logging.Logger,
    reduce_only: bool = False,
    params: dict | None = None  # Allow passing extra params if needed
) -> dict | None:
    """Places a market order using CCXT. Includes basic retry logic for network errors.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: The trading symbol.
        trade_signal: "BUY" or "SELL" indicating the order direction.
        position_size: The absolute size of the order (must be positive).
        market_info: Market dictionary for context.
        logger: Logger instance.
        reduce_only: If True, set the reduceOnly flag (for closing positions).
        params: Optional dictionary of extra parameters for create_order.

    Returns:
        The order dictionary returned by CCXT on success, None on failure.
    """
    lg = logger
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    base_currency = market_info.get('base', '')
    size_unit = market_info.get('settle', base_currency) if is_contract else base_currency  # Settled in quote/settle for contracts

    action_desc = "Close" if reduce_only else "Open/Increase"

    # Use Bybit V5 parameters
    market_id = market_info.get('id', symbol)
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    # Default positionIdx=0 for one-way mode
    position_idx = 0

    order_params = {
        'category': category,
        'positionIdx': position_idx,
        'reduceOnly': reduce_only,
    }
    if reduce_only:
        order_params['timeInForce'] = 'IOC'  # Immediate Or Cancel for reduceOnly

    lg.info(f"Attempting to place {action_desc} {side.upper()} {order_type} order for {symbol}:")
    lg.info(f"  Side: {side.upper()}, Size: {position_size.normalize()} {size_unit}")
    lg.debug(f"  Category: {category}, PositionIdx: {position_idx}, ReduceOnly: {reduce_only}")

    # Merge any additional custom params passed in
    if params:
        order_params.update(params)
    lg.debug(f"  Full Order Params: {order_params}")

    # Convert Decimal amount to float for create_order API call
    try:
        amount_float = float(position_size)
        if amount_float <= 0:
            lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Invalid position size ({position_size.normalize()}). Size must be positive.")
            return None
    except ValueError:
         lg.error(f"Trade aborted ({symbol} {side} {action_desc}): Could not convert size {position_size.normalize()} to float.")
         return None

    # --- Execute Order with Retries for Network Errors ---
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing create_order API call (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            order = exchange.create_order(
                symbol=market_id,  # Use market ID if available
                type=order_type,
                side=side,
                amount=amount_float,
                price=None,  # Market order doesn't need a specific price
                params=order_params
            )
            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'N/A')
            avg_fill_price = order.get('average')
            filled_amount = order.get('filled')

            lg.info(f"{NEON_GREEN}{action_desc} Trade Placed Successfully!{RESET}")
            lg.info(f"  Order ID: {order_id}, Initial Status: {order_status}")
            if avg_fill_price is not None: lg.info(f"  Avg Fill Price: ~{Decimal(str(avg_fill_price)).normalize() if avg_fill_price else 'N/A'}")
            if filled_amount is not None: lg.info(f"  Filled Amount: {Decimal(str(filled_amount)).normalize() if filled_amount else '0'}")
            # lg.debug(f"Raw order response ({symbol} {side} {action_desc}): {order}") # Verbose
            return order  # Success

        # --- Specific Error Handling (Non-retryable errors) ---
        except ccxt.InsufficientFunds as e:
             lg.error(f"{NEON_RED}Insufficient funds to place {action_desc} {side} order ({symbol}): {e}{RESET}")
             return None  # Non-retryable
        except ccxt.InvalidOrder as e:
            lg.error(f"{NEON_RED}Invalid order parameters placing {action_desc} {side} order ({symbol}): {e}{RESET}")
            bybit_code = getattr(e, 'code', None)
            if reduce_only and (bybit_code == 110014 or "reduce-only" in str(e).lower() or "no position" in str(e).lower()):
                 lg.error(f"{NEON_YELLOW} >> Hint (Reduce-Only Fail): Position might be closed, size incorrect, or wrong side specified? Check active position.{RESET}")
            elif bybit_code == 110007 or "order quantity" in str(e).lower() or "amount is invalid" in str(e).lower():
                 lg.error(f"{NEON_YELLOW} >> Hint (Quantity Error): Check order size ({amount_float}) against market precision (step size) and amount limits (min/max).{RESET}")
            elif bybit_code == 110040 or "order cost" in str(e).lower():
                 lg.error(f"{NEON_YELLOW} >> Hint (Cost Error): Check estimated order value ({position_size.normalize()} * price) against market cost limits (min/max).{RESET}")
            elif bybit_code == 110013:
                 lg.error(f"{NEON_YELLOW} >> Hint (Parameter Error): Review order parameters: {order_params}. Ensure they are valid for Bybit V5.{RESET}")
            return None  # Invalid order is generally not retryable
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)
            lg.error(f"{NEON_RED}Exchange error placing {action_desc} order ({symbol}): {e} (Code: {bybit_code}){RESET}")
            if reduce_only and bybit_code == 110025:
                 lg.warning(f"{NEON_YELLOW} >> Hint (Position Not Found): Position might have been closed already when trying to place reduce-only order.{RESET}")
                 return None  # Don't retry if position is gone
            elif bybit_code == 30086 or "risk limit" in str(e).lower():
                 lg.error(f"{NEON_YELLOW} >> Hint (Risk Limit): Order size + existing position size may exceed the risk limit tier for the current leverage. Reduce size/leverage or check Bybit docs.{RESET}")
                 return None  # Risk limit exceeded is not temporary
            # Otherwise, allow retry for other exchange errors

        # --- Retryable Error Handling ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries reached placing order after network errors: {e}{RESET}")
                 return None
            lg.warning(f"{NEON_YELLOW}Network error placing order (Attempt {attempts + 1}): {e}. Retrying...")
        except ccxt.RateLimitExceeded as e:
            # Don't count rate limit waits against retries, just wait and try again
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order: {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue  # Skip incrementing attempts for rate limit
        except Exception as e:
            # Catch unexpected errors
            lg.error(f"{NEON_RED}Unexpected error placing {action_desc} order ({symbol}) (Attempt {attempts + 1}): {e}{RESET}", exc_info=True)
            # Allow retry for unexpected errors

        # --- Increment attempts and wait before next retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None  # Return None if all retries fail


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,
    logger: logging.Logger,
    stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None,
    trailing_stop_distance: Decimal | None = None,  # Value/Distance, not price
    tsl_activation_price: Decimal | None = None,  # Price at which TSL becomes active
) -> bool:
    """Internal helper to set Stop Loss, Take Profit, or Trailing Stop Loss for an
    existing position using Bybit's V5 API endpoint (/v5/position/set-trading-stop).

    Handles parameter validation, formatting according to market precision,
    and making the API call with retries. TSL settings override fixed SL on Bybit V5.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol.
        market_info: Market dictionary containing precision, limits, etc.
        position_info: Position dictionary (standardized format from get_open_position).
        logger: Logger instance.
        stop_loss_price: Target fixed SL price (Decimal). Pass Decimal('0') or None to clear.
        take_profit_price: Target fixed TP price (Decimal). Pass Decimal('0') or None to clear.
        trailing_stop_distance: Target TSL distance (Decimal, positive value). Pass Decimal('0') or None to clear.
        tsl_activation_price: Target TSL activation price (Decimal). Required if setting TSL > 0.

    Returns:
        True if the protection was successfully set or updated, False otherwise.
    """
    lg = logger
    if not market_info.get('is_contract', False):
        lg.warning(f"Protection setting skipped for {symbol} (Not a contract).")
        return False  # Cannot set SL/TP/TSL on non-contracts
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    # --- Extract necessary info from position ---
    pos_side = position_info.get('side')  # 'long' or 'short'
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
         lg.error(f"Cannot set protection for {symbol}: Invalid position side ('{pos_side}') or missing entry price.")
         return False
    try:
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= Decimal('0'): raise ValueError("Entry price must be positive")
    except (InvalidOperation, TypeError, ValueError) as e:
        lg.error(f"Invalid entry price format ('{entry_price_str}') for protection check: {e}"); return False

    # --- Validate and Format Parameters ---
    params_to_set = {}  # Dictionary to hold parameters for the API call
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):"]
    any_protection_requested = False  # Track if any valid protection level was provided (non-zero Decimal)

    try:
        # Get price precision (tick size) for formatting prices accurately
        price_prec_str = market_info.get('precision', {}).get('price')
        if price_prec_str is None: raise ValueError("Missing price precision in market info")
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= Decimal('0'): raise ValueError(f"Invalid tick size: {min_tick_size}")

        # Helper to format price according to market precision using ccxt
        def format_price(price_decimal: Decimal | None) -> str | None:
            """Formats price using exchange.price_to_precision. Returns '0' for Decimal(0)."""
            if price_decimal is None: return None
            if price_decimal < Decimal('0'):
                 lg.warning(f"Cannot format negative price ({price_decimal.normalize()}). Returning None.")
                 return None
            if price_decimal == Decimal('0'):
                 return "0"  # Explicitly format 0 to indicate clearing the protection

            try:
                # Use ccxt's formatter with ROUND mode for setting levels
                formatted = exchange.price_to_precision(symbol=symbol, price=float(price_decimal), rounding_mode=exchange.ROUND)
                if Decimal(formatted) <= Decimal('0'):  # Double-check after formatting
                     lg.warning(f"Formatted price {formatted} became non-positive for input {price_decimal.normalize()}. Returning None.")
                     return None
                return formatted
            except Exception as e:
                 lg.error(f"Failed to format price {price_decimal.normalize()} using exchange precision: {e}. Returning None.")
                 return None

        # --- Trailing Stop ---
        set_tsl = False  # Flag if active TSL (>0) is being set
        if isinstance(trailing_stop_distance, Decimal):
            if trailing_stop_distance > Decimal('0'):
                 any_protection_requested = True
                 if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= Decimal('0'):
                     lg.error(f"{NEON_RED}TSL requested ({trailing_stop_distance.normalize()}), but activation price is missing or invalid ({tsl_activation_price}). Cannot set TSL.{RESET}")
                 else:
                     # Validate TSL Activation Price relative to entry (must be beyond entry)
                     activation_valid = False
                     if pos_side == 'long' and tsl_activation_price > entry_price or pos_side == 'short' and tsl_activation_price < entry_price: activation_valid = True

                     if not activation_valid:
                         lg.error(f"{NEON_RED}TSL Activation Price ({tsl_activation_price.normalize()}) is not strictly beyond entry price ({entry_price.normalize()}) for {pos_side} position. Cannot set TSL.{RESET}")
                     else:
                         # Format TSL distance (must be positive, >= 1 tick)
                         min_dist = max(trailing_stop_distance, min_tick_size)  # Enforce at least one tick
                         formatted_tsl_distance = format_price(min_dist)
                         # Format TSL activation price
                         formatted_activation_price = format_price(tsl_activation_price)

                         if formatted_tsl_distance and formatted_activation_price:
                             params_to_set['trailingStop'] = formatted_tsl_distance
                             params_to_set['activePrice'] = formatted_activation_price
                             log_parts.append(f"  Trailing SL: Dist={formatted_tsl_distance}, Act={formatted_activation_price}")
                             set_tsl = True  # Mark that TSL is being set
                             log_parts.append("  (Fixed SL will be ignored/removed as TSL is active)")
                         else:
                              lg.error(f"Failed to format valid TSL parameters (Dist: {formatted_tsl_distance}, Act: {formatted_activation_price}). TSL not set.")
            elif trailing_stop_distance == Decimal('0'):
                 # Explicitly requested to clear TSL
                 params_to_set['trailingStop'] = "0"  # Bybit V5 uses "0" to clear
                 log_parts.append("  Trailing SL: Clear (Set to 0)")
                 any_protection_requested = True
            # else: trailing_stop_distance is None, do nothing for TSL

        # --- Fixed Stop Loss (Set only if TSL is NOT being set or is being cleared) ---
        if not set_tsl and isinstance(stop_loss_price, Decimal):
            if stop_loss_price > Decimal('0'):
                any_protection_requested = True
                # Validate SL price relative to entry
                sl_valid = False
                if pos_side == 'long' and stop_loss_price < entry_price or pos_side == 'short' and stop_loss_price > entry_price: sl_valid = True

                if not sl_valid:
                    lg.error(f"{NEON_RED}Stop Loss Price ({stop_loss_price.normalize()}) is not strictly beyond entry price ({entry_price.normalize()}) for {pos_side} position. Cannot set SL.{RESET}")
                else:
                    formatted_sl = format_price(stop_loss_price)
                    if formatted_sl:
                        params_to_set['stopLoss'] = formatted_sl
                        log_parts.append(f"  Fixed SL: {formatted_sl}")
                    else:
                        lg.error(f"Failed to format valid SL price: {stop_loss_price.normalize()}. Fixed SL not set.")
            elif stop_loss_price == Decimal('0'):
                 # Explicitly requested to clear fixed SL
                 params_to_set['stopLoss'] = "0"  # Bybit V5 uses "0" to clear
                 log_parts.append("  Fixed SL: Clear (Set to 0)")
                 any_protection_requested = True
            # else: stop_loss_price is None, do nothing for fixed SL

        # --- Fixed Take Profit ---
        if isinstance(take_profit_price, Decimal):
            if take_profit_price > Decimal('0'):
                 any_protection_requested = True
                 # Validate TP price relative to entry
                 tp_valid = False
                 if pos_side == 'long' and take_profit_price > entry_price or pos_side == 'short' and take_profit_price < entry_price: tp_valid = True

                 if not tp_valid:
                    lg.error(f"{NEON_RED}Take Profit Price ({take_profit_price.normalize()}) is not strictly beyond entry price ({entry_price.normalize()}) for {pos_side} position. Cannot set TP.{RESET}")
                 else:
                    formatted_tp = format_price(take_profit_price)
                    if formatted_tp:
                        params_to_set['takeProfit'] = formatted_tp
                        log_parts.append(f"  Fixed TP: {formatted_tp}")
                    else:
                        lg.error(f"Failed to format valid TP price: {take_profit_price.normalize()}. Fixed TP not set.")
            elif take_profit_price == Decimal('0'):
                 # Explicitly requested to clear fixed TP
                 params_to_set['takeProfit'] = "0"  # Bybit V5 uses "0" to clear
                 log_parts.append("  Fixed TP: Clear (Set to 0)")
                 any_protection_requested = True
            # else: take_profit_price is None, do nothing for fixed TP

    except ValueError as ve:
         lg.error(f"Validation error processing protection parameters for {symbol}: {ve}", exc_info=False)
         return False
    except (InvalidOperation, TypeError) as conv_err:
         lg.error(f"Conversion error processing protection parameters for {symbol}: {conv_err}", exc_info=True)
         return False
    except Exception as fmt_err:
         lg.error(f"Error processing/formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # --- Check if any valid parameters were actually set or explicitly cleared ---
    if not params_to_set:
         if any_protection_requested:
              lg.warning(f"No valid protection parameters to set for {symbol} after validation/formatting. No API call made.")
              return False  # Requested action failed validation/formatting
         else:
              lg.info(f"No protection parameters provided or cleared for {symbol}. No API call needed.")
              return True  # Considered success as no action was requested/needed

    # --- Prepare API Call ---
    category = 'linear' if market_info.get('linear', True) else 'inverse'
    market_id = market_info.get('id', symbol)
    position_idx = 0  # Default for one-way mode
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: position_idx = int(pos_idx_val)
    except Exception:
        lg.warning(f"Could not parse positionIdx from position info, using default {position_idx}.")

    final_api_params = {
        'category': category,
        'symbol': market_id,  # Use exchange-specific symbol ID
        'tpslMode': 'Full',  # Affects entire position ('Partial' requires size param)
        'slTriggerBy': 'LastPrice',
        'tpTriggerBy': 'LastPrice',
        'slOrderType': 'Market',
        'tpOrderType': 'Market',
        'positionIdx': position_idx,  # 0 for one-way, 1/2 for hedge
    }
    final_api_params.update(params_to_set)

    lg.info("\n".join(log_parts))  # Log the intended action
    lg.debug(f"  API Call Params: {final_api_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing set-trading-stop API call for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            response = exchange.private_post('/v5/position/set-trading-stop', params=final_api_params)
            lg.debug(f"Set protection raw response for {symbol}: {response}")

            # --- Process Response ---
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown Error')
            response.get('retExtInfo', {})

            if ret_code == 0:
                no_change_msgs = ["not modified", "no need to modify"]  # Simplified check
                if any(msg in ret_msg.lower() for msg in no_change_msgs):
                     lg.info(f"{NEON_YELLOW}Position protection already set to target values or no change needed for {symbol}. Response: {ret_msg}{RESET}")
                else:
                     lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set/updated successfully for {symbol}.{RESET}")
                return True  # Success

            else:
                lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}){RESET}")  # Ext: {ret_ext}
                # Decide if specific errors are retryable or fatal
                non_retryable_codes = [110013, 110036, 110086, 110084, 110085, 10001, 10002]  # Parameter, Logic, Auth errors
                if bybit_code in non_retryable_codes:
                    lg.error(f" >> Hint: Error code {bybit_code} indicates a non-retryable issue (parameter/logic/auth). Check parameters and configuration.")
                    return False  # Fatal error
                # If not explicitly fatal, raise to trigger retry
                raise ccxt.ExchangeError(f"Bybit Exchange Error setting protection: {ret_msg} (Code: {ret_code})")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries reached setting protection after network errors: {e}{RESET}")
                 return False
            lg.warning(f"{NEON_YELLOW}Network error setting protection (Attempt {attempts + 1}): {e}. Retrying...")

        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded setting protection: {e}. Retrying in {wait_time}s... (Attempt {attempts + 1}){RESET}")
            time.sleep(wait_time)
            continue  # Don't count rate limit against attempts

        except ccxt.AuthenticationError as e:
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Stopping bot.{RESET}")
            # This should probably stop the whole bot, but function returns False
            return False  # Fatal

        except ccxt.ExchangeError as e:
             # Catches errors raised above or other exchange errors
             if attempts >= MAX_API_RETRIES:
                  lg.error(f"{NEON_RED}Max retries reached setting protection after exchange errors: {e}{RESET}")
                  return False
             lg.warning(f"{NEON_YELLOW}Exchange error setting protection (Attempt {attempts + 1}): {e}. Retrying...")

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error during protection API call (Attempt {attempts + 1}): {e}{RESET}", exc_info=True)
            if attempts >= MAX_API_RETRIES:
                 lg.error(f"{NEON_RED}Max retries reached after unexpected error setting protection: {e}{RESET}")
                 return False
            # Allow retry for unexpected errors

        # --- Increment attempts and wait before next retry ---
        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Exponential backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False  # Return False if all retries fail


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,
    config: dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Decimal | None = None  # Optional fixed TP to set alongside TSL
) -> bool:
    """Calculates Trailing Stop Loss parameters based on configuration and current position,
    then calls the internal helper `_set_position_protection` to set the TSL
    (and optionally a fixed Take Profit) via the Bybit V5 API.

    Args:
        exchange: Initialized CCXT exchange object.
        symbol: Trading symbol string.
        market_info: Market dictionary from `get_market_info`.
        position_info: Position dictionary from `get_open_position`.
        config: Bot configuration dictionary.
        logger: Logger instance.
        take_profit_price: Optional fixed TP target (Decimal) to set simultaneously. Pass Decimal('0') to clear TP.

    Returns:
        True if TSL (and optional TP) were successfully requested/set, False otherwise.
    """
    lg = logger
    protection_cfg = config.get("protection", {})
    # Note: Check for TSL enabled is usually done by the caller.

    # --- Validate Inputs ---
    if not market_info or not position_info:
        lg.error(f"Cannot calculate TSL for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')  # 'long' or 'short'
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"{NEON_RED}Missing required position info (side, entryPrice) for TSL calculation ({symbol}).{RESET}")
        return False

    try:
        # Extract parameters and convert to Decimal
        entry_price = Decimal(str(entry_price_str))
        # Use validated numeric types from config
        callback_rate = Decimal(str(protection_cfg.get("trailing_stop_callback_rate")))
        activation_percentage = Decimal(str(protection_cfg.get("trailing_stop_activation_percentage")))

        # Basic validation of parameters (already done during config load, but double check)
        if entry_price <= Decimal('0'): raise ValueError("Entry price must be positive")
        if callback_rate <= Decimal('0'): raise ValueError("Callback rate must be positive")
        if activation_percentage < Decimal('0'): raise ValueError("Activation percentage cannot be negative")

        # Get price precision (tick size) for rounding calculations
        price_prec_str = market_info.get('precision', {}).get('price')
        if price_prec_str is None: raise ValueError("Missing price precision in market info")
        min_tick_size = Decimal(str(price_prec_str))
        if min_tick_size <= Decimal('0'): raise ValueError(f"Invalid tick size: {min_tick_size}")

    except (ValueError, InvalidOperation, TypeError, KeyError) as ve:
        lg.error(f"{NEON_RED}Invalid TSL parameter format or position/market info ({symbol}): {ve}. Cannot calculate TSL.{RESET}")
        return False
    except Exception as e:
        lg.error(f"{NEON_RED}Error parsing TSL parameters ({symbol}): {e}. Cannot calculate TSL.{RESET}", exc_info=True)
        return False

    try:
        # --- Calculate Activation Price ---
        activation_price = None
        activation_offset_amount = entry_price * activation_percentage

        if pos_side == 'long':
            raw_activation = entry_price + activation_offset_amount
            activation_price = raw_activation.quantize(min_tick_size, rounding=ROUND_UP)
            if activation_price <= entry_price:
                 activation_price = entry_price + min_tick_size
        else:  # short
            raw_activation = entry_price - activation_offset_amount
            activation_price = raw_activation.quantize(min_tick_size, rounding=ROUND_DOWN)
            if activation_price >= entry_price:
                 activation_price = entry_price - min_tick_size

        # Validate calculated activation price
        if activation_price is None or activation_price <= Decimal('0'):
             lg.error(f"{NEON_RED}Calculated TSL activation price ({activation_price}) is zero/negative for {symbol}. Cannot set TSL.{RESET}")
             return False
        if (pos_side == 'long' and activation_price <= entry_price) or (pos_side == 'short' and activation_price >= entry_price):
             lg.error(f"{NEON_RED}Calculated and rounded TSL Activation Price ({activation_price.normalize()}) is not strictly beyond entry price ({entry_price.normalize()}) for {pos_side} position. Cannot set TSL.{RESET}")
             return False

        # --- Calculate Trailing Stop Distance ---
        # Distance is based on the activation price * callback rate.
        trailing_distance_raw = activation_price * callback_rate
        # Distance must be positive and rounded UP to the nearest tick size increment.
        trailing_distance = trailing_distance_raw.quantize(min_tick_size, rounding=ROUND_UP)
        if trailing_distance < min_tick_size:
             lg.debug(f"Calculated TSL distance {trailing_distance_raw.normalize()} rounded to {trailing_distance.normalize()}, which is less than min tick {min_tick_size.normalize()}. Setting distance to one tick.")
             trailing_distance = min_tick_size

        if trailing_distance <= Decimal('0'):
             lg.error(f"{NEON_RED}Calculated TSL distance zero/negative ({trailing_distance.normalize()}) for {symbol}. Cannot set TSL.{RESET}")
             return False

        # --- Log Calculated Parameters ---
        lg.info(f"Calculated TSL Params for {symbol} ({pos_side.upper()}):")
        lg.info(f"  Entry={entry_price.normalize()}, Act%={activation_percentage:.3%}, Callback%={callback_rate:.3%}")
        lg.info(f"  Min Tick Size: {min_tick_size.normalize()}")
        lg.info(f"  => Activation Price: {activation_price.normalize()}")
        lg.info(f"  => Trailing Distance: {trailing_distance.normalize()}")

        # Format optional TP for logging if provided
        if isinstance(take_profit_price, Decimal):
             tp_log_str = f"{take_profit_price.normalize()}"
             if take_profit_price == Decimal('0'): tp_log_str = "Clear (Set to 0)"
             lg.info(f"  Take Profit Price: {tp_log_str} (Will be set alongside TSL)")

        # --- Call Helper to Set Protection via API ---
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None,  # TSL overrides fixed SL
            take_profit_price=take_profit_price,  # Pass optional fixed TP (Decimal or None or 0)
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False

# --- Volumatic Trend + OB Strategy Implementation ---


class OrderBlock(TypedDict):
    """Represents a detected Order Block with its properties."""
    id: str           # Unique identifier (e.g., type_timestamp)
    type: str         # 'bull' or 'bear'
    left_idx: pd.Timestamp  # Timestamp of the bar where OB formed (pivot bar)
    right_idx: pd.Timestamp  # Timestamp of the last bar OB is considered valid for (or violated)
    top: Decimal      # Top price level of the OB
    bottom: Decimal   # Bottom price level of the OB
    active: bool      # Is the OB still considered valid (not violated)?
    violated: bool    # Has price closed beyond the OB boundaries?


class StrategyAnalysisResults(TypedDict):
    """Structured container for results from the strategy analysis."""
    dataframe: pd.DataFrame        # DataFrame with all indicator calculations (Decimal columns)
    last_close: Decimal            # Latest close price as Decimal
    current_trend_up: bool | None  # True=UP, False=DOWN, None=Undetermined
    trend_just_changed: bool       # True if trend changed on the last completed bar
    active_bull_boxes: list[OrderBlock]  # List of currently active bullish OBs
    active_bear_boxes: list[OrderBlock]  # List of currently active bearish OBs
    vol_norm_int: int | None    # Latest Volume Norm (0-200 integer), or None
    atr: Decimal | None         # Latest ATR as Decimal, or None
    upper_band: Decimal | None  # Latest Volumatic upper band as Decimal, or None
    lower_band: Decimal | None  # Latest Volumatic lower band as Decimal, or None


class VolumaticOBStrategy:
    """Implements the Volumatic Trend and Pivot Order Block strategy.
    Calculates indicators based on Pine Script logic interpretation, manages
    Order Block state (creation, violation, pruning), and returns analysis results.
    """

    def __init__(self, config: dict[str, Any], market_info: dict[str, Any], logger: logging.Logger) -> None:
        """Initializes the strategy engine with configuration parameters.

        Args:
            config: The main bot configuration dictionary.
            market_info: Market details dictionary (used for context).
            logger: Logger instance.
        """
        self.config = config
        self.market_info = market_info  # Store for reference (e.g., interval)
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})

        # --- Store Configurable Strategy Parameters ---
        self.vt_length = int(strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH))
        self.vt_atr_period = int(strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD))
        self.vt_vol_ema_length = int(strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH))
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        self.vt_step_atr_multiplier = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER)))

        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)
        self.ph_left = int(strategy_cfg.get("ph_left", DEFAULT_PH_LEFT))
        self.ph_right = int(strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT))
        self.pl_left = int(strategy_cfg.get("pl_left", DEFAULT_PL_LEFT))
        self.pl_right = int(strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT))
        self.ob_extend = bool(strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND))
        self.ob_max_boxes = int(strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES))

        # --- Internal State Variables ---
        self.bull_boxes: list[OrderBlock] = []
        self.bear_boxes: list[OrderBlock] = []

        # Calculate minimum data length required based on the longest lookback period
        self.min_data_len = max(
             self.vt_length * 2,  # EMA/SWMA need buffer to stabilize
             self.vt_atr_period,
             self.vt_vol_ema_length,
             self.ph_left + self.ph_right + 1,  # Pivot needs left+pivot+right bars (total points)
             self.pl_left + self.pl_right + 1
         ) + 10  # Add a general safety buffer

        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.logger.info(f"  VT Params: Len={self.vt_length}, ATRLen={self.vt_atr_period}, VolLen={self.vt_vol_ema_length}, ATRMult={self.vt_atr_multiplier.normalize()}, StepMult={self.vt_step_atr_multiplier.normalize()}")
        self.logger.info(f"  OB Params: Src={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"  Minimum historical data points recommended: {self.min_data_len}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates a custom EMA based on a 4-period Symmetrically Weighted Moving Average (SWMA).
        Replicates Pine Script `ema(swma(source), length)`. SWMA weights [1, 2, 2, 1] / 6.

        Args:
            series: Input pandas Series (float/numeric).
            length: The length parameter for the final EMA.

        Returns:
            A pandas Series containing the calculated EMA of SWMA (float). Returns NaN for initial periods.
        """
        if len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index, dtype=float)

        weights = np.array([1., 2., 2., 1.]) / 6.0  # Ensure weights are float

        # Ensure input is numeric, coerce errors
        series_numeric = pd.to_numeric(series, errors='coerce')
        # Calculate SWMA using rolling apply with weights
        swma = series_numeric.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        # Calculate EMA of the resulting SWMA series
        ema_of_swma = ta.ema(swma, length=length, fillna=np.nan)

        return ema_of_swma

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """Processes historical data to calculate Volumatic Trend and manage Pivot Order Blocks.

        Args:
            df_input: pandas DataFrame with Decimal OHLCV columns and DatetimeIndex (UTC).

        Returns:
            StrategyAnalysisResults dictionary or default empty structure on failure.
        """
        # --- Default empty result structure ---
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        # --- Pre-computation Checks ---
        if df_input.empty:
             self.logger.error("Strategy update received an empty DataFrame.")
             return empty_results

        # Work on a copy to avoid modifying the original DataFrame passed in
        df = df_input.copy()

        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
             self.logger.error("DataFrame index is invalid (not DatetimeIndex or not sorted). Analysis aborted.")
             return empty_results

        if len(df) < self.min_data_len:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(df)} rows, need >= {self.min_data_len}) for full strategy analysis. Results may be incomplete or inaccurate.{RESET}")

        self.logger.debug(f"Starting strategy analysis on {len(df)} candles.")

        # --- Data Preparation for TA calculations ---
        try:
            df_float = pd.DataFrame(index=df.index)
            float_cols_to_copy = ['open', 'high', 'low', 'close', 'volume']
            for col in float_cols_to_copy:
                if col in df.columns:
                     df_float[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    self.logger.error(f"Required column '{col}' not found in DataFrame. Analysis aborted.")
                    return empty_results
            # Drop rows in float DF if coercion created NaNs in essential price columns
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_float.empty:
                self.logger.error("DataFrame became empty after converting to float and dropping NaNs.")
                return empty_results
        except Exception as e:
            self.logger.error(f"Error converting DataFrame columns to float for TA calculations: {e}", exc_info=True)
            return empty_results

        # --- Volumatic Trend Calculations (using df_float) ---
        # Pine Script Logic:
        # UpTrend = ema(swma(close, 4), len)[1] < ema(close, len)
        # Bands based on EMA1 and ATR at the time of the last trend change.
        try:
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)  # Custom EMA(SWMA)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)  # Standard EMA

            # Trend Detection: Compare previous ema1 to current ema2
            df_float['trend_up'] = (df_float['ema1'].shift(1) < df_float['ema2'])
            df_float['trend_up'] = df_float['trend_up'].ffill()  # Fill initial NaNs

            # Trend Change Detection
            df_float['trend_changed'] = (df_float['trend_up'].shift(1) != df_float['trend_up']) & \
                                        df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'].fillna(False, inplace=True)

            # Stateful Band Calculation: Capture EMA1 and ATR on trend change
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            # Forward fill these reference values
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Bands using reference values and multiplier
            atr_mult_float = float(self.vt_atr_multiplier)
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult_float)

            # Volume Normalization
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            df_float['vol_max'] = volume_numeric.rolling(
                window=self.vt_vol_ema_length,
                min_periods=max(1, self.vt_vol_ema_length // 10)
            ).max().fillna(0.0)  # Fill NaN max with 0

            df_float['vol_norm'] = np.where(
                df_float['vol_max'] > 1e-9,  # Avoid division by zero
                (volume_numeric / df_float['vol_max'] * 100.0),
                0.0
            )
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)  # Clip and fill NaNs

        except Exception as e:
            self.logger.error(f"Error during Volumatic Trend indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy Calculated Float Columns back to the main Decimal DataFrame ---
        # Ensure index alignment and handle potential NaNs/Infs during conversion
        cols_to_copy = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed',
                        'upper_band', 'lower_band', 'vol_norm']
        try:
            for col in cols_to_copy:
                if col in df_float.columns:
                    # Reindex the float series to match the main Decimal DF index
                    source_series = df_float[col].reindex(df.index)
                    if source_series.dtype == 'bool' or pd.api.types.is_object_dtype(source_series):
                        df[col] = source_series
                    else:  # Convert numeric float back to Decimal
                        df[col] = source_series.apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                        )
        except Exception as e:
            self.logger.error(f"Error converting calculated float columns back to Decimal: {e}", exc_info=True)
            return empty_results

        # --- Data Cleaning After Calculations ---
        initial_len_before_drop = len(df)
        required_cols_for_signal = ['upper_band', 'lower_band', 'atr', 'trend_up', 'close']
        df.dropna(subset=required_cols_for_signal, inplace=True)
        rows_dropped = initial_len_before_drop - len(df)
        if rows_dropped > 0:
             self.logger.debug(f"Dropped {rows_dropped} initial rows lacking essential indicator values after calculation.")

        if df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame empty after calculating indicators and dropping NaNs. Insufficient data or no stable trend/ATR established yet.{RESET}")
            return empty_results

        self.logger.debug("Volumatic Trend calculations complete. Proceeding to Order Blocks.")

        # --- Pivot Order Block Calculations & Management ---
        try:
            # Determine high/low series from float DF for pivot detection
            if self.ob_source == "Wicks":
                high_series = df_float['high']
                low_series = df_float['low']
            else:  # "Bodys"
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)

            # Calculate pivot signals on float DF
            ph_signals_float = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high').fillna(0).astype(bool)
            pl_signals_float = ta.pivot(low_series, left=self.pl_left, right=self.pl_right, high_low='low').fillna(0).astype(bool)

            # Align pivot signals back to the main (cleaned) Decimal DataFrame index
            df['ph_signal'] = ph_signals_float.reindex(df.index, fill_value=False)
            df['pl_signal'] = pl_signals_float.reindex(df.index, fill_value=False)

            # --- Identify NEW Pivots Confirmed in the Latest Data ---
            # Iterate through the main DF where signals are now aligned
            new_boxes_found_count = 0
            if not df.empty:
                 for confirmation_idx in df.index:  # Check all rows in the cleaned DF
                     try:
                         ph_confirmed_here = df.loc[confirmation_idx, 'ph_signal']
                         pl_confirmed_here = df.loc[confirmation_idx, 'pl_signal']
                         confirmation_loc_in_df = df.index.get_loc(confirmation_idx)

                         # --- Check for New Bearish OB (from Pivot High Confirmation) ---
                         if ph_confirmed_here:
                             pivot_bar_loc_in_df = confirmation_loc_in_df - self.ph_right
                             if pivot_bar_loc_in_df >= 0:
                                 pivot_bar_idx = df.index[pivot_bar_loc_in_df]
                                 if not any(b['left_idx'] == pivot_bar_idx and b['type'] == 'bear' for b in self.bear_boxes):
                                     ob_candle = df.loc[pivot_bar_idx]
                                     box_top, box_bottom = Decimal('NaN'), Decimal('NaN')
                                     if self.ob_source == "Wicks":
                                         box_top = ob_candle['high']
                                         box_bottom = ob_candle['open']
                                     else:  # "Bodys"
                                         box_top = ob_candle['close']
                                         box_bottom = ob_candle['open']
                                     if pd.notna(box_top) and pd.notna(box_bottom):
                                         if box_bottom > box_top: box_top, box_bottom = box_bottom, box_top  # Ensure top > bottom
                                         if box_top > box_bottom:  # Create only if valid range
                                             self.bear_boxes.append(OrderBlock(
                                                 id=f"bear_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}", type='bear',
                                                 left_idx=pivot_bar_idx, right_idx=df.index[-1],
                                                 top=box_top, bottom=box_bottom, active=True, violated=False))
                                             self.logger.debug(f"{NEON_RED}New Bearish OB created at {pivot_bar_idx} [{box_bottom.normalize()} - {box_top.normalize()}]{RESET}")
                                             new_boxes_found_count += 1

                         # --- Check for New Bullish OB (from Pivot Low Confirmation) ---
                         if pl_confirmed_here:
                             pivot_bar_loc_in_df = confirmation_loc_in_df - self.pl_right
                             if pivot_bar_loc_in_df >= 0:
                                 pivot_bar_idx = df.index[pivot_bar_loc_in_df]
                                 if not any(b['left_idx'] == pivot_bar_idx and b['type'] == 'bull' for b in self.bull_boxes):
                                     ob_candle = df.loc[pivot_bar_idx]
                                     box_top, box_bottom = Decimal('NaN'), Decimal('NaN')
                                     if self.ob_source == "Wicks":
                                         box_top = ob_candle['open']
                                         box_bottom = ob_candle['low']
                                     else:  # "Bodys"
                                         box_top = ob_candle['open']
                                         box_bottom = ob_candle['close']
                                     if pd.notna(box_top) and pd.notna(box_bottom):
                                         if box_bottom > box_top: box_top, box_bottom = box_bottom, box_top  # Ensure top > bottom
                                         if box_top > box_bottom:  # Create only if valid range
                                             self.bull_boxes.append(OrderBlock(
                                                 id=f"bull_{pivot_bar_idx.strftime('%Y%m%d%H%M%S')}", type='bull',
                                                 left_idx=pivot_bar_idx, right_idx=df.index[-1],
                                                 top=box_top, bottom=box_bottom, active=True, violated=False))
                                             self.logger.debug(f"{NEON_GREEN}New Bullish OB created at {pivot_bar_idx} [{box_bottom.normalize()} - {box_top.normalize()}]{RESET}")
                                             new_boxes_found_count += 1

                     except KeyError: continue  # Should not happen with aligned index
                     except Exception as e: self.logger.warning(f"Error processing pivot signal at {confirmation_idx}: {e}", exc_info=True)

            if new_boxes_found_count > 0:
                self.logger.debug(f"Found {new_boxes_found_count} new OB(s). Total counts before pruning: {len(self.bull_boxes)} Bull, {len(self.bear_boxes)} Bear.")

            # --- Manage Existing Order Blocks (Violation Check & Extension) ---
            if not df.empty and 'close' in df.columns and pd.notna(df['close'].iloc[-1]):
                last_bar_idx = df.index[-1]
                last_close = df['close'].iloc[-1]  # Use Decimal close price

                for box in self.bull_boxes:
                    if box['active']:
                        if last_close < box['bottom']:
                            box['active'] = False; box['violated'] = True; box['right_idx'] = last_bar_idx
                            self.logger.debug(f"Bull Box {box['id']} ({box['bottom'].normalize()}-{box['top'].normalize()}) VIOLATED by close {last_close.normalize()} at {last_bar_idx}.")
                        elif self.ob_extend: box['right_idx'] = last_bar_idx

                for box in self.bear_boxes:
                    if box['active']:
                        if last_close > box['top']:
                            box['active'] = False; box['violated'] = True; box['right_idx'] = last_bar_idx
                            self.logger.debug(f"Bear Box {box['id']} ({box['bottom'].normalize()}-{box['top'].normalize()}) VIOLATED by close {last_close.normalize()} at {last_bar_idx}.")
                        elif self.ob_extend: box['right_idx'] = last_bar_idx
            else:
                self.logger.warning("Cannot check OB violations: Last close price or DataFrame is invalid.")

            # --- Prune Older Order Blocks ---
            # Keep the N most recent boxes overall (active or inactive)
            all_bull = sorted(self.bull_boxes, key=lambda b: b['left_idx'], reverse=True)
            all_bear = sorted(self.bear_boxes, key=lambda b: b['left_idx'], reverse=True)
            self.bull_boxes = all_bull[:self.ob_max_boxes]
            self.bear_boxes = all_bear[:self.ob_max_boxes]

            active_bull_count_after_prune = sum(1 for b in self.bull_boxes if b['active'])
            active_bear_count_after_prune = sum(1 for b in self.bear_boxes if b['active'])
            self.logger.debug(f"Pruned OBs. Total Kept: {len(self.bull_boxes)} Bull ({active_bull_count_after_prune} active), {len(self.bear_boxes)} Bear ({active_bear_count_after_prune} active). (Max {self.ob_max_boxes} total of each type)")

        except Exception as e:
             self.logger.error(f"Error during Pivot Order Block processing: {e}", exc_info=True)
             pass  # Allow returning partial results from VT section

        # --- Prepare Final Results ---
        last_row = df.iloc[-1] if not df.empty else None
        final_close = last_row.get('close', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_atr = last_row.get('atr', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_upper_band = last_row.get('upper_band', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_lower_band = last_row.get('lower_band', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        final_vol_norm = last_row.get('vol_norm', Decimal('NaN')) if last_row is not None else Decimal('NaN')
        latest_trend_up = last_row.get('trend_up') if last_row is not None else None
        latest_trend_changed = last_row.get('trend_changed', False) if last_row is not None else False

        # --- Sanitize final values before returning ---
        def sanitize_decimal(val, positive_only=False):
            if pd.notna(val) and isinstance(val, Decimal) and np.isfinite(float(val)):
                if positive_only and val <= Decimal('0'): return None
                return val
            return None

        final_trend_up_bool = bool(latest_trend_up) if isinstance(latest_trend_up, (bool, np.bool_)) else None
        final_vol_norm_int = int(final_vol_norm) if sanitize_decimal(final_vol_norm) is not None else None
        final_atr_dec = sanitize_decimal(final_atr, positive_only=True)
        final_upper_band_dec = sanitize_decimal(final_upper_band)
        final_lower_band_dec = sanitize_decimal(final_lower_band)
        final_close_dec = sanitize_decimal(final_close) or Decimal('0')  # Default to 0 if close is invalid

        active_bull_boxes_final = [b for b in self.bull_boxes if b['active']]
        active_bear_boxes_final = [b for b in self.bear_boxes if b['active']]

        results = StrategyAnalysisResults(
            dataframe=df,  # Return the main Decimal DataFrame with indicators
            last_close=final_close_dec,
            current_trend_up=final_trend_up_bool,
            trend_just_changed=bool(latest_trend_changed),
            active_bull_boxes=active_bull_boxes_final,
            active_bear_boxes=active_bear_boxes_final,
            vol_norm_int=final_vol_norm_int,
            atr=final_atr_dec,
            upper_band=final_upper_band_dec,
            lower_band=final_lower_band_dec
        )

        # Log key summary results
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr'].normalize()}" if results['atr'] else "N/A"
        last_idx_time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
        self.logger.debug(f"Strategy Results ({last_idx_time_str}): "
                          f"Close={results['last_close'].normalize()}, Trend={trend_str}, TrendChg={results['trend_just_changed']}, "
                          f"ATR={atr_str}, VolNorm={results['vol_norm_int']}, "
                          f"Active OBs (Bull/Bear): {len(results['active_bull_boxes'])}/{len(results['active_bear_boxes'])}")

        return results


# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """Generates trading signals (BUY, SELL, HOLD, EXIT_LONG, EXIT_SHORT)
    based on the analysis results from VolumaticOBStrategy and the current
    position state. Also calculates initial TP/SL levels for potential trades.
    """

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        """Initializes the SignalGenerator with configuration parameters."""
        self.config = config
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})
        try:
            # Proximity factors for OB interaction (use Decimal)
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg.get("ob_entry_proximity_factor", 1.005)))
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg.get("ob_exit_proximity_factor", 1.001)))
            if self.ob_entry_proximity_factor < Decimal('1'): self.ob_entry_proximity_factor = Decimal("1.0")
            if self.ob_exit_proximity_factor < Decimal('1'): self.ob_exit_proximity_factor = Decimal("1.0")

            # ATR Multipliers for initial TP/SL (validated during config load)
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg.get("initial_take_profit_atr_multiple")))
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg.get("initial_stop_loss_atr_multiple")))
            # Ensure SL multiple is strictly positive
            if self.initial_sl_atr_multiple <= Decimal('0'):
                 self.logger.error(f"{NEON_RED}Config Error: initial_stop_loss_atr_multiple must be positive. Using default 1.8.{RESET}")
                 self.initial_sl_atr_multiple = Decimal('1.8')

        except (ValueError, TypeError, KeyError, InvalidOperation) as e:
             self.logger.error(f"{NEON_RED}Error initializing SignalGenerator with config values: {e}. Using defaults.{RESET}", exc_info=True)
             self.ob_entry_proximity_factor = Decimal("1.005")
             self.ob_exit_proximity_factor = Decimal("1.001")
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

        self.logger.info("Signal Generator Initialized.")
        self.logger.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor.normalize()}, OB Exit Proximity Factor: {self.ob_exit_proximity_factor.normalize()}")
        self.logger.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}, Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: dict | None) -> str:
        """Determines the appropriate trading signal based on strategy analysis
        (trend, OBs, price action) and the current open position status.
        """
        # --- Input Validation ---
        if not isinstance(analysis_results, dict) or analysis_results.get('dataframe') is None or \
           analysis_results['dataframe'].empty or \
           analysis_results.get('current_trend_up') is None or \
           not isinstance(analysis_results.get('last_close'), Decimal) or analysis_results['last_close'] <= Decimal('0') or \
           analysis_results.get('atr') is None or analysis_results['atr'] <= Decimal('0'):  # atr checked positive by sanitize
            self.logger.warning(f"{NEON_YELLOW}Incomplete or invalid strategy analysis results received. Cannot generate signal. Holding.{RESET}")
            self.logger.debug(f"  Problematic Analysis Results: Trend={analysis_results.get('current_trend_up')}, Close={analysis_results.get('last_close')}, ATR={analysis_results.get('atr')}")
            return "HOLD"

        # Extract key values from results
        latest_close = analysis_results['last_close']
        is_trend_up = analysis_results['current_trend_up']  # True or False
        trend_just_changed = analysis_results['trend_just_changed']
        active_bull_obs = analysis_results['active_bull_boxes']
        active_bear_obs = analysis_results['active_bear_boxes']

        current_pos_side = open_position.get('side') if open_position else None  # 'long' or 'short'
        signal = "HOLD"  # Default signal

        self.logger.debug(f"Signal Check: Close={latest_close.normalize()}, Trend Up={is_trend_up}, Trend Changed={trend_just_changed}, Position={current_pos_side or 'None'}")
        self.logger.debug(f"Active OBs: Bull={len(active_bull_obs)}, Bear={len(active_bear_obs)}")

        # --- Signal Logic ---
        # === 1. Check for EXIT Signal (only if a position is currently open) ===
        if current_pos_side == 'long':
            self.logger.debug("Checking EXIT LONG conditions...")
            # Exit Condition 1: Trend flips DOWN
            if is_trend_up is False:
                 if trend_just_changed:
                      signal = "EXIT_LONG"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped DOWN.{RESET}")
                 # else: pass # Optional: Exit if trend is persistently down

            # Exit Condition 2: Price reaches/exceeds nearest active Bearish OB (Resistance)
            if signal == "HOLD" and active_bear_obs:
                try:
                    closest_bear_ob = min(active_bear_obs, key=lambda ob: abs(ob['top'] - latest_close))
                    # Exit threshold is *beyond* the OB top (price >= threshold)
                    exit_threshold = closest_bear_ob['top'] * self.ob_exit_proximity_factor
                    self.logger.debug(f"  Bear OB Exit Check: Closest Bear OB Top={closest_bear_ob['top'].normalize()}, Exit Threshold={exit_threshold.normalize()}")
                    if latest_close >= exit_threshold:
                         signal = "EXIT_LONG"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price ({latest_close.normalize()}) >= Bear OB Exit Threshold ({exit_threshold.normalize()}) [OB ID: {closest_bear_ob['id']}]{RESET}")
                except Exception as e: self.logger.warning(f"Error checking Bear OB exit condition: {e}")

        elif current_pos_side == 'short':
             self.logger.debug("Checking EXIT SHORT conditions...")
             # Exit Condition 1: Trend flips UP
             if is_trend_up is True:
                 if trend_just_changed:
                     signal = "EXIT_SHORT"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped UP.{RESET}")
                 # else: pass # Optional: Exit if trend is persistently up

             # Exit Condition 2: Price reaches/falls below nearest active Bullish OB (Support)
             if signal == "HOLD" and active_bull_obs:
                 try:
                    closest_bull_ob = min(active_bull_obs, key=lambda ob: abs(ob['bottom'] - latest_close))
                    # Exit threshold is *beyond* the OB bottom (price <= threshold)
                    if self.ob_exit_proximity_factor > Decimal('0'):
                         exit_threshold = closest_bull_ob['bottom'] / self.ob_exit_proximity_factor
                    else: exit_threshold = closest_bull_ob['bottom']  # Fallback
                    self.logger.debug(f"  Bull OB Exit Check: Closest Bull OB Bottom={closest_bull_ob['bottom'].normalize()}, Exit Threshold={exit_threshold.normalize()}")
                    if latest_close <= exit_threshold:
                          signal = "EXIT_SHORT"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price ({latest_close.normalize()}) <= Bull OB Exit Threshold ({exit_threshold.normalize()}) [OB ID: {closest_bull_ob['id']}]{RESET}")
                 except Exception as e: self.logger.warning(f"Error checking Bull OB exit condition: {e}")

        # If an exit signal was generated, return it immediately.
        if signal in ["EXIT_LONG", "EXIT_SHORT"]: return signal

        # === 2. Check for ENTRY Signal (Only if NO position is currently open) ===
        if current_pos_side is None:
            self.logger.debug("Checking ENTRY conditions...")
            # Check for BUY (Long Entry): Trend UP and Price interacts with Bull OB
            if is_trend_up is True and active_bull_obs:
                entry_ob = None
                for ob in active_bull_obs:
                    lower_bound = ob['bottom']
                    upper_bound = ob['top'] * self.ob_entry_proximity_factor
                    if lower_bound <= latest_close <= upper_bound:
                        entry_ob = ob; break
                if entry_ob:
                    signal = "BUY"; self.logger.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price ({latest_close.normalize()}) in/near Bull OB {entry_ob['id']} [{entry_ob['bottom'].normalize()}-{entry_ob['top'].normalize()}]. Entry Zone: [{lower_bound.normalize()}, {upper_bound.normalize()}]{RESET}")
                # else: self.logger.debug(f"  Price ({latest_close.normalize()}) not within entry zone of any active Bull OB.")

            # Check for SELL (Short Entry): Trend DOWN and Price interacts with Bear OB
            elif is_trend_up is False and active_bear_obs:
                 entry_ob = None
                 for ob in active_bear_obs:
                     if self.ob_entry_proximity_factor > Decimal('0'):
                          lower_bound = ob['bottom'] / self.ob_entry_proximity_factor
                     else: lower_bound = ob['bottom']  # Fallback
                     upper_bound = ob['top']
                     if lower_bound <= latest_close <= upper_bound:
                         entry_ob = ob; break
                 if entry_ob:
                     signal = "SELL"; self.logger.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price ({latest_close.normalize()}) in/near Bear OB {entry_ob['id']} [{entry_ob['bottom'].normalize()}-{entry_ob['top'].normalize()}]. Entry Zone: [{lower_bound.normalize()}, {upper_bound.normalize()}]{RESET}")
                 # else: self.logger.debug(f"  Price ({latest_close.normalize()}) not within entry zone of any active Bear OB.")
            # else: self.logger.debug(f"  No entry signal: Trend {'DOWN' if is_trend_up is False else 'UP'}. No relevant active OBs found or no trend.")

        # === 3. Log HOLD Reason ===
        if signal == "HOLD":
             trend_status = 'UP' if is_trend_up is True else 'DOWN' if is_trend_up is False else 'N/A'
             if current_pos_side:
                 self.logger.debug(f"HOLD Signal: Trend ({trend_status}) compatible with existing {current_pos_side} position. No exit condition met.")
             else:
                 ob_status = "No relevant active OBs found for current trend or no trend."
                 if is_trend_up is True and active_bull_obs: ob_status = f"Price ({latest_close.normalize()}) not within entry zone of active Bull OBs."
                 elif is_trend_up is False and active_bear_obs: ob_status = f"Price ({latest_close.normalize()}) not within entry zone of active Bear OBs."
                 self.logger.debug(f"HOLD Signal: No position open. Trend is {trend_status}. {ob_status}")

        return signal

    def calculate_initial_tp_sl(
        self,
        entry_price: Decimal,
        signal: str,  # "BUY" or "SELL"
        atr: Decimal,
        market_info: dict,
        exchange: ccxt.Exchange  # Needed for formatting
    ) -> tuple[Decimal | None, Decimal | None]:
        """Calculates initial Take Profit (TP) and Stop Loss (SL) levels based on
        entry price, current ATR, configured multipliers, and market precision.
        """
        lg = self.logger

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]: lg.error(f"Invalid signal '{signal}'."); return None, None
        if entry_price <= Decimal('0'): lg.error(f"Invalid entry price ({entry_price})."); return None, None
        if atr <= Decimal('0'): lg.error(f"Invalid ATR value ({atr})."); return None, None
        if market_info.get('precision', {}).get('price') is None:
             lg.error("Missing market price precision info."); return None, None

        try:
            price_prec_str = market_info['precision']['price']
            min_tick_size = Decimal(str(price_prec_str))
            if min_tick_size <= Decimal('0'): raise ValueError(f"Invalid tick size: {min_tick_size}")

            tp_multiple = self.initial_tp_atr_multiple
            sl_multiple = self.initial_sl_atr_multiple  # Guaranteed > 0

            tp_offset = atr * tp_multiple
            sl_offset = atr * sl_multiple

            take_profit_raw = None
            stop_loss_raw = None

            if signal == "BUY":
                if tp_multiple > Decimal('0'): take_profit_raw = entry_price + tp_offset
                stop_loss_raw = entry_price - sl_offset
            elif signal == "SELL":
                if tp_multiple > Decimal('0'): take_profit_raw = entry_price - tp_offset
                stop_loss_raw = entry_price + sl_offset

            # --- Format Prices Using Exchange Precision ---
            def format_level(price_decimal: Decimal | None, level_type: str) -> Decimal | None:
                if price_decimal is None: return None
                if price_decimal <= Decimal('0'):
                    lg.warning(f"Calculated {level_type} is zero/negative ({price_decimal.normalize()}). Setting to None.")
                    return None
                try:
                    formatted_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_decimal), rounding_mode=exchange.ROUND)
                    formatted_decimal = Decimal(formatted_str)
                    if formatted_decimal <= Decimal('0'):
                        lg.warning(f"Formatted {level_type} became zero/negative ({formatted_decimal.normalize()}). Setting to None.")
                        return None
                    return formatted_decimal
                except Exception as e:
                     lg.error(f"Error formatting {level_type} price {price_decimal.normalize()}: {e}. Setting to None.")
                     return None

            take_profit = format_level(take_profit_raw, "TP")
            stop_loss = format_level(stop_loss_raw, "SL")

            # --- Final Validation (Ensure SL/TP are strictly beyond entry after rounding) ---
            if stop_loss is not None:
                if (signal == "BUY" and stop_loss >= entry_price) or \
                   (signal == "SELL" and stop_loss <= entry_price):
                    lg.warning(f"Formatted {signal} SL ({stop_loss.normalize()}) not strictly beyond entry ({entry_price.normalize()}). Adjusting by one tick.")
                    stop_loss = (entry_price - min_tick_size) if signal == "BUY" else (entry_price + min_tick_size)
                    stop_loss = format_level(stop_loss, "SL")  # Reformat after adjustment

            if take_profit is not None:
                 if (signal == "BUY" and take_profit <= entry_price) or \
                    (signal == "SELL" and take_profit >= entry_price):
                     lg.warning(f"Formatted {signal} TP ({take_profit.normalize()}) not strictly beyond entry ({entry_price.normalize()}). Setting TP to None.")
                     take_profit = None

            # --- Log & Return ---
            tp_str = f"{take_profit.normalize()}" if take_profit else "None"
            sl_str = f"{stop_loss.normalize()}" if stop_loss else "None (FAILED)"
            lg.debug(f"Calculated Initial Protection Levels: TP={tp_str}, SL={sl_str}")

            if stop_loss is None:
                 lg.error(f"{NEON_RED}Stop Loss calculation failed or resulted in an invalid level. Cannot proceed with sizing.{RESET}")
                 return take_profit, None

            return take_profit, stop_loss

        except (ValueError, InvalidOperation, TypeError) as ve:
             lg.error(f"{NEON_RED}Error calculating initial TP/SL: {ve}{RESET}", exc_info=True)
             return None, None
        except Exception as e:
             lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
             return None, None


# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange,
    symbol: str,
    config: dict[str, Any],
    logger: logging.Logger,
    strategy_engine: VolumaticOBStrategy,  # Passed instance
    signal_generator: SignalGenerator,   # Passed instance
    market_info: dict                    # Passed validated market info
) -> None:
    """Performs one cycle of the trading logic for a single symbol."""
    lg = logger
    lg.info(f"\n---== Analyzing {symbol} ({config.get('interval', 'N/A')}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    # Validation already done in main()

    # Determine fetch limit
    fetch_limit = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    min_strategy_len = strategy_engine.min_data_len
    required_fetch_limit = min_strategy_len + 50  # Add buffer
    if fetch_limit < required_fetch_limit:
         lg.warning(f"{NEON_YELLOW}Configured fetch_limit ({fetch_limit}) < recommended ({required_fetch_limit}). Increasing to {required_fetch_limit}.{RESET}")
         fetch_limit = required_fetch_limit

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit, logger=lg)
    if klines_df.empty or len(klines_df) < min_strategy_len:
        lg.error(f"Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}, need >= {min_strategy_len}). Skipping cycle.")
        return

    # --- 2. Run Strategy Analysis ---
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
         lg.error(f"{NEON_RED}Error during strategy analysis update for {symbol}: {analysis_err}{RESET}", exc_info=True)
         return  # Skip cycle

    # Validate essential results
    if not isinstance(analysis_results, dict) or \
       analysis_results.get('current_trend_up') is None or \
       analysis_results['last_close'] <= Decimal('0') or \
       analysis_results['atr'] is None or analysis_results['atr'] <= Decimal('0'):
         lg.error(f"{NEON_RED}Strategy analysis produced incomplete/invalid results. Cannot proceed. Skipping cycle.{RESET}")
         lg.debug(f"  Analysis Results Dump: {analysis_results}")
         return

    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr']

    # --- Get Current Price (for more accurate BE/TSL checks) ---
    current_market_price = fetch_current_price_ccxt(exchange, symbol, lg)
    price_for_checks = current_market_price if current_market_price else latest_close
    if price_for_checks <= Decimal('0'):
        lg.error(f"{NEON_RED}Cannot get valid price (current or last close) for {symbol}. Skipping cycle.{RESET}")
        return
    if current_market_price is None:
        lg.warning(f"{NEON_YELLOW}Using last close price {latest_close.normalize()} for protection checks as current price fetch failed.{RESET}")

    # --- 3. Check Position & Generate Signal ---
    open_position = get_open_position(exchange, symbol, lg)

    try:
        # Use last_close from analysis for signal generation
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
         lg.error(f"{NEON_RED}Error during signal generation for {symbol}: {signal_err}{RESET}", exc_info=True)
         return  # Skip cycle

    # --- 4. Trading Logic ---
    trading_enabled = config.get("enable_trading", False)
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading disabled. Generated Signal: {signal}. Analysis complete for {symbol}.{RESET}")
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis-Only Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")
        return  # End cycle

    # ===========================================
    # --- EXECUTION LOGIC (Trading Enabled) ---
    # ===========================================
    lg.debug(f"Trading enabled. Signal: {signal}. Position: {'Yes' if open_position else 'No'}")

    # === Scenario 1: No Open Position ===
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {NEON_GREEN if signal == 'BUY' else NEON_RED}{BRIGHT}{signal} Signal & No Position: Initiating Trade Entry Sequence for {symbol}{RESET} ***")

            # --- Pre-Trade Checks & Setup ---
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= Decimal('0'):
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot fetch balance or balance is zero/negative ({balance}).{RESET}")
                return

            lg.debug(f"Calculating initial TP/SL for sizing (Est. Entry={latest_close.normalize()}, ATR={current_atr.normalize()})")
            initial_tp_calc, initial_sl_calc = signal_generator.calculate_initial_tp_sl(
                 entry_price=latest_close, signal=signal, atr=current_atr,
                 market_info=market_info, exchange=exchange)
            if initial_sl_calc is None:
                 lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Initial SL calculation failed. Cannot size.{RESET}")
                 return
            if initial_tp_calc is None: lg.warning(f"{NEON_YELLOW}Initial TP calculation failed/disabled. Proceeding without initial fixed TP.{RESET}")

            leverage_set_success = True
            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 0))
                if leverage > 0:
                    leverage_set_success = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg)
                    if not leverage_set_success:
                         lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Failed to set leverage {leverage}x.{RESET}")
                         return
            position_size = calculate_position_size(
                balance=balance, risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=initial_sl_calc, entry_price=latest_close,
                market_info=market_info, exchange=exchange, logger=lg)
            if position_size is None or position_size <= Decimal('0'):
                lg.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size ({position_size}). Check balance, risk, SL, limits.{RESET}")
                return

            # --- Place Entry Trade ---
            lg.info(f"==> Placing {signal} market order | Size: {position_size.normalize()} {market_info.get('base', '')} <==")
            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg, reduce_only=False)

            # --- Post-Trade: Verify Position & Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order {order_id} placed. Waiting {confirm_delay}s for confirmation...")
                time.sleep(confirm_delay)
                lg.info(f"Confirming position status for {symbol}...")
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        entry_price_actual_str = confirmed_position.get('entryPrice')
                        entry_price_actual = Decimal(str(entry_price_actual_str)) if entry_price_actual_str else latest_close  # Fallback
                        if entry_price_actual <= Decimal('0'): entry_price_actual = latest_close  # Ensure positive
                        lg.info(f"{NEON_GREEN}Position Confirmed! Side: {confirmed_position.get('side')}, Actual Entry: ~{entry_price_actual.normalize()}{RESET}")

                        # --- Set Protection based on Actual Entry ---
                        protection_cfg = config.get("protection", {})
                        tp_for_protection, sl_for_protection = signal_generator.calculate_initial_tp_sl(
                             entry_price=entry_price_actual, signal=signal, atr=current_atr,
                             market_info=market_info, exchange=exchange)
                        if sl_for_protection is None:
                             lg.error(f"{NEON_RED}Failed to recalculate SL for setting protection. Position vulnerable!{RESET}")
                             sl_for_protection = None  # Ensure None if failed

                        protection_set_success = False
                        if protection_cfg.get("enable_trailing_stop", True):
                             lg.info(f"Setting Trailing Stop Loss (using Entry={entry_price_actual.normalize()})...")
                             protection_set_success = set_trailing_stop_loss(
                                 exchange, symbol, market_info, confirmed_position, config, lg,
                                 take_profit_price=tp_for_protection)  # Pass recalculated TP
                        elif not protection_cfg.get("enable_trailing_stop", True) and \
                             (protection_cfg.get("initial_stop_loss_atr_multiple", 0) > 0 or
                              protection_cfg.get("initial_take_profit_atr_multiple", 0) > 0):
                             lg.info(f"Setting Fixed SL/TP (using Entry={entry_price_actual.normalize()})...")
                             if sl_for_protection or tp_for_protection:
                                 protection_set_success = _set_position_protection(
                                     exchange, symbol, market_info, confirmed_position, lg,
                                     stop_loss_price=sl_for_protection, take_profit_price=tp_for_protection)
                             else:
                                 lg.warning(f"{NEON_YELLOW}No valid fixed SL/TP levels calculated based on actual entry. No fixed protection set.{RESET}")
                                 protection_set_success = True  # No action needed
                        else:
                              lg.info("Neither TSL nor Fixed SL/TP enabled. No protection set.")
                              protection_set_success = True

                        if protection_set_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({symbol} {signal}) ===")
                        else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}). MANUAL MONITORING REQUIRED! ===")

                    except Exception as post_trade_err:
                         lg.error(f"{NEON_RED}Critical error during post-trade setup ({symbol}): {post_trade_err}{RESET}", exc_info=True)
                         lg.warning(f"{NEON_YELLOW}Position may be open without protection. Manual check needed!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM position after {confirm_delay}s delay! Manual investigation required!{RESET}")
            else:
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). See previous logs. ===")
        else:  # signal == HOLD and no position
            lg.info(f"Signal is HOLD and no open position for {symbol}. No entry action.")

    # === Scenario 2: Existing Open Position ===
    else:  # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size_decimal = open_position.get('size_decimal', Decimal('0'))
        pos_size_fmt = pos_size_decimal.normalize() if pos_size_decimal else 'N/A'
        lg.info(f"Existing {pos_side.upper()} position found for {symbol} (Size: {pos_size_fmt}). Signal: {signal}")

        # --- Check for Exit Signal ---
        exit_signal_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                                (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal Triggered! Initiating Close Sequence for {pos_side} position on {symbol}... ***{RESET}")
            try:
                close_side_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close = abs(pos_size_decimal)
                if size_to_close <= Decimal('0'):
                    lg.warning(f"Position size to close is zero/negative ({size_to_close.normalize()}). Already closed? Skipping.")
                else:
                    lg.info(f"==> Placing {close_side_signal} MARKET order (reduceOnly=True) | Size: {size_to_close.normalize()} <==")
                    close_order = place_trade(exchange, symbol, close_side_signal, size_to_close, market_info, lg, reduce_only=True)
                    if close_order and close_order.get('id'): lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully. Order ID: {close_order.get('id', 'N/A')}{RESET}")
                    else: lg.error(f"{NEON_RED}Failed to place CLOSE order. Manual check required!{RESET}")
            except Exception as close_err:
                 lg.error(f"{NEON_RED}Unexpected error closing position {symbol}: {close_err}{RESET}", exc_info=True)
                 lg.warning(f"{NEON_YELLOW}Manual intervention may be needed!{RESET}")

        # --- Manage Existing Position (HOLD signal or compatible signal) ---
        else:
            lg.debug(f"Signal ({signal}) allows holding. Managing protections for {pos_side} position...")
            protection_cfg = config.get("protection", {})

            # Get current state needed for management
            tsl_dist_on_pos_str = open_position.get('trailingStopLoss')
            is_tsl_active_on_pos = False
            try:
                if tsl_dist_on_pos_str and Decimal(str(tsl_dist_on_pos_str)) > Decimal('0'):
                     is_tsl_active_on_pos = True
                     lg.debug("Trailing Stop Loss is active on position.")
            except Exception: pass  # Ignore parsing errors here

            current_sl_price = None; current_tp_price = None
            with contextlib.suppress(Exception): current_sl_price = Decimal(str(open_position.get('stopLossPrice'))) if open_position.get('stopLossPrice') and str(open_position.get('stopLossPrice')) != '0' else None
            with contextlib.suppress(Exception): current_tp_price = Decimal(str(open_position.get('takeProfitPrice'))) if open_position.get('takeProfitPrice') and str(open_position.get('takeProfitPrice')) != '0' else None

            entry_price_actual = None
            with contextlib.suppress(Exception): entry_price_actual = Decimal(str(open_position.get('entryPrice'))) if open_position.get('entryPrice') else None

            # --- Break-Even Logic ---
            be_enabled = protection_cfg.get("enable_break_even", True)
            if be_enabled and not is_tsl_active_on_pos and \
               entry_price_actual and current_atr and price_for_checks > Decimal('0'):
                lg.debug(f"Checking Break-Even (Entry: {entry_price_actual.normalize()}, Price: {price_for_checks.normalize()}, ATR: {current_atr.normalize()})...")
                try:
                    be_trigger_atr_mult = Decimal(str(protection_cfg.get("break_even_trigger_atr_multiple", 1.0)))
                    be_offset_ticks = int(protection_cfg.get("break_even_offset_ticks", 2))
                    price_diff = (price_for_checks - entry_price_actual) if pos_side == 'long' else (entry_price_actual - price_for_checks)
                    profit_in_atr = price_diff / current_atr
                    lg.debug(f"  BE Check: Profit ATRs = {profit_in_atr:.2f}, Trigger = {be_trigger_atr_mult.normalize()}")

                    if profit_in_atr >= be_trigger_atr_mult:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}BE Profit target ({be_trigger_atr_mult.normalize()} ATRs) REACHED!{RESET}")
                        min_tick_size = Decimal(str(market_info['precision']['price']))
                        tick_offset_value = min_tick_size * Decimal(str(be_offset_ticks))
                        be_stop_price = (entry_price_actual + tick_offset_value).quantize(min_tick_size, rounding=ROUND_UP) if pos_side == 'long' \
                                   else (entry_price_actual - tick_offset_value).quantize(min_tick_size, rounding=ROUND_DOWN)

                        if be_stop_price and be_stop_price > Decimal('0'):
                             update_be = False
                             if current_sl_price is None: update_be = True; lg.info("  No current SL, setting BE.")
                             elif pos_side == 'long' and be_stop_price > current_sl_price: update_be = True; lg.info(f"  BE SL {be_stop_price.normalize()} > Current SL {current_sl_price.normalize()}.")
                             elif pos_side == 'short' and be_stop_price < current_sl_price: update_be = True; lg.info(f"  BE SL {be_stop_price.normalize()} < Current SL {current_sl_price.normalize()}.")
                             else: lg.debug(f"  Current SL ({current_sl_price.normalize()}) already >= BE target ({be_stop_price.normalize()}).")

                             if update_be:
                                 lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving SL to Break-Even for {symbol} at {be_stop_price.normalize()} ***{RESET}")
                                 success = _set_position_protection(exchange, symbol, market_info, open_position, lg,
                                                                     stop_loss_price=be_stop_price, take_profit_price=current_tp_price)
                                 if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                                 else: lg.error(f"{NEON_RED}Failed to set/update Break-Even SL via API.{RESET}")
                        else: lg.error(f"{NEON_RED}BE triggered, but calculated BE stop price ({be_stop_price}) invalid.{RESET}")
                    else: lg.debug("BE Profit target not reached.")
                except Exception as be_err: lg.error(f"{NEON_RED}Error during BE check: {be_err}{RESET}", exc_info=True)
            elif be_enabled and is_tsl_active_on_pos: lg.debug("BE check skipped: TSL is active.")
            elif not be_enabled: lg.debug("BE check skipped: Disabled in config.")
            else: lg.debug("BE check skipped: Missing data (entry/price/ATR).")

            # --- Trailing Stop Management (Setup/Recovery) ---
            tsl_enabled = protection_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not is_tsl_active_on_pos and entry_price_actual and current_atr:
                 lg.warning(f"{NEON_YELLOW}TSL enabled but not active. Attempting TSL setup/recovery...{RESET}")
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(
                      entry_price=entry_price_actual, signal=pos_side.upper(), atr=current_atr,
                      market_info=market_info, exchange=exchange)
                 tsl_set_success = set_trailing_stop_loss(
                      exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc)
                 if tsl_set_success: lg.info("TSL setup/recovery attempt successful.")
                 else: lg.error("TSL setup/recovery attempt failed.")
            # else: lg.debug("TSL setup/recovery skipped.")

    # --- Cycle End Logging ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")  # Add newline for separation


def main() -> None:
    """Main function to initialize the bot, set up the exchange connection,
    get user input for symbol and interval, and run the main trading loop.
    """
    global CONFIG, QUOTE_CURRENCY  # Allow access to global config state

    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Volumatic Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{RESET}")

    # Config loaded globally, validation and defaults applied

    init_logger.info(f"Config Loaded: Quote={QUOTE_CURRENCY}, Trading={CONFIG.get('enable_trading')}, Sandbox={CONFIG.get('use_sandbox')}")
    try:
        init_logger.info(f"Versions: Python={os.sys.version.split()[0]}, CCXT={ccxt.__version__}, Pandas={pd.__version__}, NumPy={np.__version__}, PandasTA={getattr(ta, 'version', 'N/A')}")
    except Exception as e: init_logger.warning(f"Could not get library versions: {e}")

    # --- User Confirmation for Live Trading ---
    trading_enabled_cfg = CONFIG.get("enable_trading", False)
    sandbox_mode_cfg = CONFIG.get("use_sandbox", True)
    if trading_enabled_cfg:
         init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if sandbox_mode_cfg: init_logger.warning(f"{NEON_YELLOW}Mode: SANDBOX (Testnet) Environment{RESET}")
         else: init_logger.warning(f"{NEON_RED}{BRIGHT}Mode: LIVE (Real Money) Environment{RESET}")

         protection_cfg = CONFIG.get("protection", {})
         init_logger.warning(f"{BRIGHT}--- Key Settings Review ---{RESET}")
         init_logger.warning(f"  Risk Per Trade: {CONFIG.get('risk_per_trade', 0) * 100:.2f}%")
         init_logger.warning(f"  Leverage: {CONFIG.get('leverage', 0)}x")
         init_logger.warning(f"  Trailing SL: {'ENABLED' if protection_cfg.get('enable_trailing_stop') else 'DISABLED'} "
                             f"(CB: {protection_cfg.get('trailing_stop_callback_rate', 0) * 100:.2f}%, "
                             f"Act: {protection_cfg.get('trailing_stop_activation_percentage', 0) * 100:.2f}%)")
         init_logger.warning(f"  Break Even: {'ENABLED' if protection_cfg.get('enable_break_even') else 'DISABLED'} "
                             f"(Trig: {protection_cfg.get('break_even_trigger_atr_multiple', 0)} ATR, "
                             f"Offset: {protection_cfg.get('break_even_offset_ticks', 0)} ticks)")
         init_logger.warning("---------------------------")
         try:
             input(f"{BRIGHT}>>> Review settings CAREFULLY. Press {NEON_GREEN}Enter{RESET}{BRIGHT} to continue, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
             init_logger.info("User confirmed live trading settings. Proceeding...")
         except KeyboardInterrupt:
             init_logger.info("User aborted startup during confirmation.")
             return
    else:
         init_logger.info(f"{NEON_YELLOW}Trading is disabled in config. Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange Connection ---
    init_logger.info("Initializing exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange connection. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange {exchange.id} initialized successfully.")

    # --- Get and Validate Trading Symbol ---
    target_symbol = None
    market_info = None
    while target_symbol is None:  # Loop until a valid symbol is confirmed
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").strip().upper()
            if not symbol_input_raw: continue  # Ask again if input is empty

            # Try direct input first, then common forms like BASE/QUOTE:QUOTE
            symbols_to_try = [symbol_input_raw]
            if '/' in symbol_input_raw and ':' not in symbol_input_raw:
                 symbols_to_try.append(f"{symbol_input_raw}:{QUOTE_CURRENCY}")  # e.g., BTC/USDT -> BTC/USDT:USDT
            elif ':' in symbol_input_raw and '/' not in symbol_input_raw and symbol_input_raw[-1] != ':':
                 # Convert BASE:QUOTE to BASE/QUOTE only if : isn't the last char (avoids BTC/USDT:)
                 symbols_to_try.append(symbol_input_raw.replace(':', '/'))

            symbols_to_try = list(dict.fromkeys(symbols_to_try))  # Remove duplicates

            for symbol_attempt in symbols_to_try:
                init_logger.info(f"Validating symbol '{symbol_attempt}'...")
                market_info_attempt = get_market_info(exchange, symbol_attempt, init_logger)

                if market_info_attempt:
                    target_symbol = market_info_attempt['symbol']  # Use the validated symbol from CCXT
                    market_info = market_info_attempt  # Store the validated market info
                    market_type_desc = market_info.get('contract_type_str', "Unknown Type")
                    init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type_desc})")
                    # Perform essential validation checks on market_info
                    if market_info.get('precision', {}).get('price') is None or \
                       market_info.get('precision', {}).get('amount') is None:
                         init_logger.error(f"{NEON_RED}CRITICAL: Validated market '{target_symbol}' is missing required price/amount precision! Cannot continue.{RESET}")
                         return  # Fatal error if precision is missing
                    break  # Exit the inner loop once a valid symbol is found
            else:
                 if market_info is None:  # If loop finished without finding a valid symbol
                      init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' (and variations {symbols_to_try}) not found or invalid on {exchange.id}. Please check the symbol and try again.{RESET}")

        except KeyboardInterrupt:
            init_logger.info("User aborted startup during symbol input.")
            return
        except Exception as e:
            init_logger.error(f"Error during symbol validation: {e}", exc_info=True)
            # Loop will continue to ask for symbol

    # --- Get and Validate Analysis Interval ---
    selected_interval = None
    while selected_interval is None:  # Loop until valid interval
        default_interval = CONFIG.get('interval', '5')  # Get default from current config
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval {VALID_INTERVALS} (default: {default_interval}): {RESET}").strip()
        if not interval_input: interval_input = default_interval

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             # Update config dictionary in memory ONLY if it changed from default
             if CONFIG.get('interval') != selected_interval:
                 CONFIG["interval"] = selected_interval
                 init_logger.info(f"Interval updated to {selected_interval} in memory.")
                 # Optionally save the updated config back to file here
                 # try:
                 #     config_to_save = json.loads(json.dumps(CONFIG, default=str))
                 #     with open(CONFIG_FILE, "w", encoding="utf-8") as f_write:
                 #         json.dump(config_to_save, f_write, indent=4)
                 #     init_logger.info(f"Saved updated interval to config file.")
                 # except Exception as save_err: init_logger.error(f"Error saving interval to config: {save_err}")

             ccxt_tf = CCXT_INTERVAL_MAP.get(selected_interval)
             init_logger.info(f"Using interval: {selected_interval} (CCXT Timeframe: {ccxt_tf})")
             break
        else:
             init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")

    # --- Setup Symbol-Specific Logger ---
    symbol_logger = setup_logger(target_symbol)  # Use the validated symbol for logger name
    symbol_logger.info(f"---=== {BRIGHT}Starting Trading Loop for {target_symbol} ({CONFIG['interval']}){RESET} ===---")
    symbol_logger.info(f"Trading Enabled: {trading_enabled_cfg}, Sandbox Mode: {sandbox_mode_cfg}")
    protection_cfg = CONFIG.get("protection", {})  # Re-get for symbol logger
    symbol_logger.info(f"Settings: Risk={CONFIG['risk_per_trade']:.2%}, Leverage={CONFIG['leverage']}x, TSL={'ON' if protection_cfg.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if protection_cfg.get('enable_break_even') else 'OFF'}")
    symbol_logger.debug(f"Strategy Params: {json.dumps(CONFIG.get('strategy_params', {}))}")
    symbol_logger.debug(f"Protection Params: {json.dumps(protection_cfg)}")

    # --- Instantiate Strategy Engine and Signal Generator ---
    try:
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_init_err:
        symbol_logger.critical(f"{NEON_RED}Failed to initialize strategy engine or signal generator: {engine_init_err}. Exiting.{RESET}", exc_info=True)
        return

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}Entering main trading loop... Press Ctrl+C to stop.{RESET}")
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")

            # --- Core Logic Execution ---
            try:
                # --- Optional: Dynamic Config Reload ---
                # Uncomment carefully if needed. Requires re-initializing components.
                # current_config = load_config(CONFIG_FILE)
                # if current_config != CONFIG: ... update CONFIG, strategy_engine, signal_generator ...

                analyze_and_trade_symbol(
                    exchange=exchange,
                    symbol=target_symbol,
                    config=CONFIG,  # Pass current config state
                    logger=symbol_logger,
                    strategy_engine=strategy_engine,
                    signal_generator=signal_generator,
                    market_info=market_info  # Pass validated market info
                )
            # --- Robust Error Handling for Common Issues within the Loop ---
            except ccxt.RateLimitExceeded as e:
                 symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting 60 seconds...{RESET}")
                 time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e:
                 symbol_logger.error(f"{NEON_RED}Network error encountered: {e}. Waiting {RETRY_DELAY_SECONDS * 3}s before next cycle...{RESET}")
                 time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                 symbol_logger.critical(f"{NEON_RED}CRITICAL Authentication Error: {e}. API keys may be invalid/expired/permissions revoked. Stopping bot.{RESET}")
                 break  # Exit the main trading loop
            except ccxt.ExchangeNotAvailable as e:
                 symbol_logger.error(f"{NEON_RED}Exchange not available: {e}. Waiting 60 seconds...{RESET}")
                 time.sleep(60)
            except ccxt.OnMaintenance as e:
                 symbol_logger.error(f"{NEON_RED}Exchange under maintenance: {e}. Waiting 5 minutes...{RESET}")
                 time.sleep(300)
            except ccxt.ExchangeError as e:
                 symbol_logger.error(f"{NEON_RED}Unhandled Exchange Error in main loop: {e}{RESET}", exc_info=True)
                 symbol_logger.warning(f"{NEON_YELLOW}Pausing for 10 seconds before next cycle.{RESET}")
                 time.sleep(10)  # Short pause
            except Exception as loop_error:
                 symbol_logger.error(f"{NEON_RED}Critical unexpected error in main trading loop: {loop_error}{RESET}", exc_info=True)
                 symbol_logger.warning(f"{NEON_YELLOW}Pausing for 15 seconds due to unexpected error.{RESET}")
                 time.sleep(15)

            # --- Loop Delay Calculation ---
            elapsed_time = time.time() - loop_start_time
            loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle processed in {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Initiating graceful shutdown...")
    except Exception as critical_error:
        init_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR outside main loop: {critical_error}{RESET}", exc_info=True)
    finally:
        # --- Shutdown Procedures ---
        shutdown_msg = f"--- Pyrmethus Volumatic Bot for {target_symbol or 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
             symbol_logger.info(shutdown_msg)

        # Close exchange connection (optional, good practice)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Attempting to close CCXT exchange connection...")
                # await exchange.close() # Use await if using async version of ccxt
                # For synchronous ccxt, close might not be needed or might be implicit
                init_logger.info("Exchange connection closed (or no action needed).")
            except Exception as close_err:
                init_logger.error(f"Error closing exchange connection: {close_err}")

        logging.shutdown()  # Flush and close all logging handlers properly


if __name__ == "__main__":
    # Ensure all dependencies listed at the top are installed:
    # pip install ccxt pandas numpy pandas_ta requests python-dotenv colorama tzdata
    main()

# --- END OF FILE volumatictrend1.0.4.py ---
