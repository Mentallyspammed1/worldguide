# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.3: Improved config validation, dynamic fetch limit, API limit logging, general refinements.

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

# import websocket # Requires websocket-client (Imported but unused)
import ccxt  # Requires ccxt

# --- Dependencies (Install via pip) ---
import numpy as np  # Requires numpy
import pandas as pd  # Requires pandas
import pandas_ta as ta  # Requires pandas_ta
import requests  # Requires requests
from colorama import Fore, Style, init  # Requires colorama
from dotenv import load_dotenv  # Requires python-dotenv

# --- Initialize Environment and Settings ---
getcontext().prec = 28
init(autoreset=True)
load_dotenv()

# --- Constants ---
# API Credentials
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# Config/Logging
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
try:
    TIMEZONE = ZoneInfo("America/Chicago")  # Example: Use 'UTC' or your local timezone
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# API Interaction
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
POSITION_CONFIRM_DELAY_SECONDS = 8
LOOP_DELAY_SECONDS = 15
BYBIT_API_KLINE_LIMIT = 1000  # Bybit V5 Kline limit per request

# Timeframes
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling
DEFAULT_FETCH_LIMIT = 750  # Default config value if user doesn't set it (used if less than min_data_len)
MAX_DF_LEN = 2000  # Internal limit to prevent excessive memory usage

# Strategy Defaults (Used if missing/invalid in config)
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950  # <-- ADJUSTED DEFAULT (Original 1000 often > API Limit)
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0  # Note: Step ATR Multiplier currently unused in logic
DEFAULT_OB_SOURCE = "Wicks"  # Or "Body"
DEFAULT_PH_LEFT = 10; DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10; DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50

# Dynamically loaded from config: QUOTE_CURRENCY

# Logging Colors
NEON_GREEN = Fore.LIGHTGREEN_EX; NEON_BLUE = Fore.CYAN; NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW; NEON_RED = Fore.LIGHTRED_EX; NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL; BRIGHT = Style.BRIGHT; DIM = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)


# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    """Redacts sensitive API keys from log messages."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY: msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET: msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def setup_logger(name: str) -> logging.Logger:
    """Sets up a logger with console and rotating file handlers."""
    safe_name = name.replace('/', '_').replace(':', '-')  # Sanitize for filename
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): return logger  # Avoid duplicate handlers if called again

    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers control output level

    try:  # File Handler (DEBUG level)
        fh = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except Exception:
        pass

    try:  # Console Handler (Level from ENV or INFO default)
        sh = logging.StreamHandler()
        # Use timezone-aware timestamps for console output
        logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
        sf = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        sh.setFormatter(sf)
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)  # Default to INFO if invalid
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception:
        pass

    logger.propagate = False  # Prevent messages going to root logger
    return logger


def _ensure_config_keys(config: dict[str, Any], default_config: dict[str, Any], parent_key: str = "") -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from default_config exist in config, logs additions."""
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


def load_config(filepath: str) -> dict[str, Any]:
    """Loads, validates, and potentially updates configuration from JSON file."""
    # Define default config structure and values
    default_config = {
        "interval": "5",
        "retry_delay": RETRY_DELAY_SECONDS,
        "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25,  # Note: Orderbook not currently used in this strategy
        "enable_trading": False,
        "use_sandbox": True,
        "risk_per_trade": 0.01,  # 1% risk
        "leverage": 20,  # Default leverage if applicable
        "max_concurrent_positions": 1,  # Note: Currently only supports 1 symbol/position
        "quote_currency": "USDT",
        "loop_delay_seconds": LOOP_DELAY_SECONDS,
        "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH,
            "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH,
            "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER,  # Unused
            "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT,
            "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND,
            "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005,  # e.g., Price must be within 0.5% above Bull OB top
            "ob_exit_proximity_factor": 1.001   # e.g., Price must be within 0.1% above Bear OB top to exit Long
        },
        "protection": {
             "enable_trailing_stop": True,
             "trailing_stop_callback_rate": 0.005,  # 0.5% callback
             "trailing_stop_activation_percentage": 0.003,  # Activate TSL after 0.3% profit
             "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0,  # Move SL to BE after 1.0 ATR profit
             "break_even_offset_ticks": 2,  # Move SL N ticks beyond entry for BE
             "initial_stop_loss_atr_multiple": 1.8,  # Initial SL placement
             "initial_take_profit_atr_multiple": 0.7  # Initial TP placement (0 means no TP)
        }
    }
    config_needs_saving = False
    loaded_config = {}

    if not os.path.exists(filepath):  # Create default if not found
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating a default configuration file.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config file: {filepath}{RESET}")
            return default_config
        except OSError as e:
            init_logger.error(f"{NEON_RED}Error creating default config file: {e}. Using internal defaults.{RESET}")
            return default_config

    try:  # Load existing config
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from config file '{filepath}': {e}. Attempting to recreate with defaults.{RESET}")
        try:  # Try to recreate on decode error
            with open(filepath, "w", encoding="utf-8") as f_create:
                json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config file: {filepath}{RESET}")
            return default_config
        except OSError as e_create:
            init_logger.error(f"{NEON_RED}Error recreating default config file after decode error: {e_create}. Using internal defaults.{RESET}")
            return default_config
    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error loading config file: {e}. Using internal defaults.{RESET}", exc_info=True)
        return default_config

    try:  # Validate and merge loaded config with defaults
        # Ensure all default keys exist
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys:
            config_needs_saving = True

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg, key_path, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            """Validates numeric config values, logs warning & uses default if invalid."""
            nonlocal config_needs_saving
            keys = key_path.split('.')
            current_cfg_level = cfg
            default_cfg_level = default_config
            try:
                # Traverse dictionaries to get the value and its default
                for key in keys[:-1]:
                    current_cfg_level = current_cfg_level[key]
                    default_cfg_level = default_cfg_level[key]
                key_leaf = keys[-1]
                original_value = current_cfg_level.get(key_leaf)
                default_value = default_cfg_level.get(key_leaf)
            except (KeyError, TypeError):
                 # This case should theoretically be handled by _ensure_config_keys adding defaults
                 init_logger.warning(f"Config validation internal error: Key path '{key_path}' invalid after ensuring keys.")
                 return False  # Indicate potential issue, but don't modify further

            if original_value is None:
                 # Also should be handled by _ensure_config_keys
                 return False  # Key missing

            corrected = False
            final_value = original_value
            try:
                # Attempt conversion to Decimal for robust checks
                num_value = Decimal(str(original_value))

                # Range check
                min_check = num_value > min_val if is_strict_min else num_value >= min_val
                if not (min_check and num_value <= max_val):
                    if not (allow_zero and num_value == 0):  # Special case for allowing zero (e.g., TP multiple)
                        raise ValueError("Value out of allowed range")

                # Type check and conversion back to desired type (float/int)
                if is_int:
                    if num_value != num_value.to_integral_value(rounding=ROUND_DOWN):
                        raise ValueError("Value must be an integer")
                    final_value = int(num_value)
                else:
                    final_value = float(num_value)  # Store as float for JSON compatibility

                # Check if the type or numeric value actually changed
                if type(final_value) is not type(original_value) or final_value != original_value:
                     # Log correction only if type changed, even if numerically equivalent (e.g., "10" -> 10)
                     if type(final_value) is not type(original_value):
                         init_logger.info(f"{NEON_YELLOW}Config: Corrected type for '{key_path}' from '{type(original_value).__name__}' to '{type(final_value).__name__}'. Value: {final_value}{RESET}")
                     corrected = True  # Mark as corrected if type or value changed

            except (ValueError, InvalidOperation, TypeError):
                init_logger.warning(f"{NEON_YELLOW}Config: Invalid value '{original_value}' for '{key_path}'. Expected {'integer' if is_int else 'number'} in range {'(' if is_strict_min else '['}{min_val}, {max_val}{']'}{' (or 0)' if allow_zero else ''}. Using default: {default_value}{RESET}")
                final_value = default_value
                corrected = True

            if corrected:
                current_cfg_level[key_leaf] = final_value
                config_needs_saving = True  # Mark that the file needs saving
            return corrected  # Return whether a correction occurred

        # --- Apply Validations ---
        # Top Level
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid interval '{updated_config.get('interval')}' in config. Using default '{default_config['interval']}'. Valid: {VALID_INTERVALS}{RESET}")
            updated_config["interval"] = default_config["interval"]; config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60)
        validate_numeric(updated_config, "fetch_limit", 100, BYBIT_API_KLINE_LIMIT, is_int=True)
        validate_numeric(updated_config, "orderbook_limit", 1, 200, is_int=True)
        validate_numeric(updated_config, "risk_per_trade", 0, 1, is_strict_min=True)  # Risk > 0%
        validate_numeric(updated_config, "leverage", 0, 200, is_int=True)  # Leverage >= 0 (0 likely means no leverage setting)
        validate_numeric(updated_config, "max_concurrent_positions", 1, 10, is_int=True)
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600)
        validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60)

        # Strategy Params
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, 1000, is_int=True)
        # Crucial: Vol EMA length must allow fetching enough data within API limit
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, BYBIT_API_KLINE_LIMIT - 50, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20)
        validate_numeric(updated_config, "strategy_params.vt_step_atr_multiplier", 0.1, 20)
        validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ph_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.pl_right", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_max_boxes", 1, 200, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1)  # Must be >= 1
        validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)  # Must be >= 1

        # Protection Params
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0001, 0.5, is_strict_min=True)  # > 0%
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0, 0.5, allow_zero=True)  # >= 0%
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.1, 10)
        validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)  # >= 0 ticks
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.1, 100)  # > 0
        validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0, 100, allow_zero=True)  # >= 0 (0 means disabled)

        # Save if needed
        if config_needs_saving:
             try:
                 # Ensure the config object uses standard Python types (float, int) before saving
                 config_to_save = json.loads(json.dumps(updated_config))  # Simple way to convert Decimals if any remained
                 with open(filepath, "w", encoding="utf-8") as f_write:
                     json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration with corrections/additions to: {filepath}{RESET}")
             except Exception as save_err:
                 init_logger.error(f"{NEON_RED}Error saving updated config file: {save_err}{RESET}", exc_info=True)

        return updated_config

    except Exception as e:
        init_logger.error(f"{NEON_RED}Unexpected error processing config file: {e}. Using internal defaults.{RESET}", exc_info=True)
        return default_config


# --- Logger & Config Setup ---
init_logger = setup_logger("init")  # Logger for initial setup
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")  # Get quote currency after loading


# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with retries for market loading."""
    lg = logger
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Enable CCXT's built-in rate limiter
            'options': {
                'defaultType': 'linear',  # Default to linear contracts (can be overridden by market info)
                'adjustForTimeDifference': True,  # Adjust for clock skew
                # Timeouts for various operations (milliseconds)
                'fetchTickerTimeout': 15000,
                'fetchBalanceTimeout': 20000,
                'createOrderTimeout': 30000,
                'cancelOrderTimeout': 20000,
                'fetchPositionsTimeout': 20000,
                'fetchOHLCVTimeout': 60000,  # Longer timeout for fetching klines
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
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets(reload=attempt > 0)  # Force reload on retry
                if exchange.markets and len(exchange.markets) > 0:
                    lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols).")
                    break
                else:
                    lg.warning(f"load_markets returned empty or null (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error loading markets (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    lg.critical(f"{NEON_RED}Max retries reached while loading markets due to network errors: {e}. Exiting.{RESET}")
                    return None
            except Exception as e:
                lg.critical(f"{NEON_RED}Unexpected error loading markets: {e}. Exiting.{RESET}", exc_info=True)
                return None
        # Final check if markets loaded after loop
        if not exchange.markets or len(exchange.markets) == 0:
            lg.critical(f"{NEON_RED}Failed to load markets after all retries. Exiting.{RESET}")
            return None

        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Attempt initial balance fetch (optional but good for diagnostics)
        lg.info(f"Attempting initial balance fetch (Quote Currency: {QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None:
                lg.info(f"{NEON_GREEN}Initial balance fetched: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                # This might be okay if trading is disabled, but critical if enabled
                lg.critical(f"{NEON_RED}Initial balance fetch failed (returned None).{RESET}")
                if CONFIG.get('enable_trading', False):
                    lg.critical(f"{NEON_RED} Trading is enabled, but balance fetch failed. Critical error. Exiting.{RESET}")
                    return None
                else:
                    lg.warning(f"{NEON_YELLOW} Trading is disabled, proceeding cautiously despite balance fetch failure.{RESET}")
        except ccxt.AuthenticationError as auth_err:
             lg.critical(f"{NEON_RED}Authentication Error fetching initial balance: {auth_err}. Check API keys/permissions.{RESET}")
             return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Non-critical error during initial balance fetch: {balance_err}.{RESET}", exc_info=True)
             # Proceed only if trading is disabled
             if CONFIG.get('enable_trading', False):
                 lg.critical(f"{NEON_RED} Trading is enabled, critical error during balance fetch. Exiting.{RESET}")
                 return None

        return exchange

    except Exception as e:
        lg.critical(f"{NEON_RED}Failed to initialize CCXT exchange object: {e}{RESET}", exc_info=True)
        return None


# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Decimal | None:
    """Fetches the current market price for a symbol using fetch_ticker with retries."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price = None

            # Helper to safely convert ticker values to Decimal
            def safe_decimal(val_str, name) -> Decimal | None:
                if val_str is not None and str(val_str).strip() != '':
                    try:
                        p = Decimal(str(val_str))
                        return p if p > Decimal('0') else None  # Ensure price is positive
                    except (InvalidOperation, ValueError, TypeError):
                         lg.debug(f"Could not parse ticker field '{name}' value '{val_str}' to Decimal.")
                         return None
                return None

            # Try 'last' price first
            price = safe_decimal(ticker.get('last'), 'last')

            # If 'last' is invalid, try mid-price from bid/ask
            if price is None:
                bid = safe_decimal(ticker.get('bid'), 'bid')
                ask = safe_decimal(ticker.get('ask'), 'ask')
                if bid and ask and ask >= bid:
                    price = (bid + ask) / Decimal('2')
                    lg.debug(f"Using mid-price fallback: ({bid} + {ask}) / 2 = {price.normalize()}")
                elif ask:  # Fallback to ask if only ask is valid
                    price = ask
                    lg.warning(f"{NEON_YELLOW}Using 'ask' price fallback: {price.normalize()}{RESET}")
                elif bid:  # Fallback to bid if only bid is valid
                    price = bid
                    lg.warning(f"{NEON_YELLOW}Using 'bid' price fallback: {price.normalize()}{RESET}")

            if price:
                lg.debug(f"Current price for {symbol}: {price.normalize()}")
                return price
            else:
                lg.warning(f"Failed to get a valid price from ticker data (Attempt {attempts + 1}). Ticker: {ticker}")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching price for {symbol}: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            # Use a longer delay for rate limit errors
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching price for {symbol}: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Don't increment attempts immediately after rate limit sleep
        except ccxt.ExchangeError as e:
            # Exchange errors are often non-recoverable for a specific request
            lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
            return None  # Stop retrying on exchange errors
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop retrying on unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS)

    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries and robust processing."""
    lg = logger
    if not exchange.has['fetchOHLCV']:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV method.")
        return pd.DataFrame()

    ohlcv = None
    # Ensure requested limit doesn't exceed the absolute API limit
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
         lg.debug(f"Requested kline limit {limit} exceeds API limit {BYBIT_API_KLINE_LIMIT}. Requesting {BYBIT_API_KLINE_LIMIT}.")

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={actual_request_limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})")
            # Fetch OHLCV data
            # The 'params' argument could be used for exchange-specific options if needed, e.g., {'category': 'linear'} for Bybit v5
            params = {}
            if 'bybit' in exchange.id.lower():
                 try:  # Attempt to determine category for Bybit
                     market = exchange.market(symbol)
                     category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                     if category in ['linear', 'inverse', 'spot']: params['category'] = category
                     lg.debug(f"Using Bybit category '{category}' for kline fetch.")
                 except Exception as market_err:
                     lg.warning(f"Could not determine Bybit category for kline fetch ({market_err}). Using default.")

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit, params=params)

            returned_count = len(ohlcv) if ohlcv else 0
            lg.debug(f"Exchange returned {returned_count} candles (requested {actual_request_limit}).")

            # *** Specific log if API limit was hit and potentially insufficient ***
            if returned_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT:
                lg.warning(f"{NEON_YELLOW}Fetched {returned_count} candles, which is the API limit ({BYBIT_API_KLINE_LIMIT}). "
                           f"Strategy might require more data ({limit}) than available in one request. "
                           f"Consider reducing lookback periods in config.{RESET}")
            elif returned_count < actual_request_limit and returned_count > 0:
                 lg.debug(f"Exchange returned fewer candles ({returned_count}) than requested ({actual_request_limit}). This might be expected near the start of market history.")

            if ohlcv and returned_count > 0:
                # Basic validation: Check timestamp of the last candle (optional but useful)
                try:
                    last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    # Estimate reasonable lag based on timeframe
                    try: interval_seconds = exchange.parse_timeframe(timeframe)
                    except Exception: interval_seconds = 300  # Default to 5 mins if parsing fails
                    max_lag = max((interval_seconds * 5), 300)  # Allow up to 5 intervals lag, min 5 mins
                    if timeframe in ['1d', '1w', '1M']: max_lag = max(max_lag, 3600 * 6)  # Allow more lag for daily+ TFs

                    lag_seconds = (now_utc - last_ts).total_seconds()
                    if lag_seconds < max_lag:
                        lg.debug(f"Last kline timestamp {last_ts} seems current (Lag: {lag_seconds:.1f}s).")
                        break  # Data looks okay, exit retry loop
                    else:
                        lg.warning(f"{NEON_YELLOW}Last kline timestamp {last_ts} seems old (Lag: {lag_seconds:.1f}s > Max allowed: {max_lag}s). Retrying fetch...{RESET}")
                        ohlcv = None  # Discard potentially stale data and retry
                except Exception as ts_err:
                    lg.warning(f"Could not validate timestamp of last kline: {ts_err}. Proceeding with fetched data.")
                    break  # Proceed even if timestamp validation fails

            else:  # No data returned or empty list
                lg.warning(f"fetch_ohlcv returned no data or an empty list (Attempt {attempt + 1}). Retrying...")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempt < MAX_API_RETRIES:
                lg.warning(f"{NEON_YELLOW}Network error fetching klines (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached fetching klines due to network errors: {e}{RESET}")
                return pd.DataFrame()  # Return empty DataFrame on persistent failure
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5  # Longer wait for rate limits
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching klines: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            # Don't increment attempt count immediately after rate limit sleep, just retry the API call
        except ccxt.ExchangeError as e:
            # Includes BadSymbol, AuthenticationError, etc. Usually not retryable.
            lg.error(f"{NEON_RED}Exchange error fetching klines: {e}{RESET}")
            # Check for specific Bybit "invalid parameter" errors which might indicate symbol/category mismatch
            bybit_code = getattr(e, 'code', None)
            if bybit_code in [10001, 110013]:  # Example Bybit parameter error codes
                lg.error(f" >> Hint: Check if symbol '{symbol}' exists and is valid for the specified category (Linear/Inverse/Spot).")
            return pd.DataFrame()
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching klines: {e}{RESET}", exc_info=True)
            return pd.DataFrame()

        # Increment attempt counter only if it wasn't a rate limit retry
        if not isinstance(e, ccxt.RateLimitExceeded):
            attempts += 1
            if attempts <= MAX_API_RETRIES: time.sleep(RETRY_DELAY_SECONDS)

    if not ohlcv:
        lg.warning(f"{NEON_YELLOW}No kline data obtained for {symbol} {timeframe} after all retries.{RESET}")
        return pd.DataFrame()

    # --- Data Processing ---
    try:
        # Define standard columns
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        # Create DataFrame, adapting columns if exchange returns fewer (e.g., no volume)
        df = pd.DataFrame(ohlcv, columns=columns[:len(ohlcv[0])])

        # Convert timestamp to datetime index (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # Drop rows where timestamp conversion failed
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to Decimal, handling potential NaNs or Infs
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # Convert to numeric first, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Convert valid numerics to Decimal, leave NaNs as is for now
                df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))

        # Data Cleaning
        initial_len = len(df)
        # Drop rows with NaN in essential price columns or non-positive close
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > Decimal('0')]
        # Drop rows with NaN volume or negative volume (if volume exists)
        if 'volume' in df.columns:
            df.dropna(subset=['volume'], inplace=True)
            df = df[df['volume'] >= Decimal('0')]

        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
            lg.debug(f"Dropped {rows_dropped} rows with invalid OHLCV data.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data is empty after cleaning for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        # Ensure data is sorted by time
        df.sort_index(inplace=True)

        # Trim DataFrame if it exceeds the maximum internal length
        if len(df) > MAX_DF_LEN:
            lg.debug(f"Trimming DataFrame from {len(df)} to {MAX_DF_LEN} rows.")
            df = df.iloc[-MAX_DF_LEN:].copy()  # Keep the most recent data

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except Exception as e:
        lg.error(f"{NEON_RED}Error processing fetched kline data: {e}{RESET}", exc_info=True)
        return pd.DataFrame()


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Retrieves and validates market information (precision, limits, type) with retries."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            # Check if market exists, reload markets if necessary
            if not exchange.markets or symbol not in exchange.markets:
                lg.info(f"Market info for '{symbol}' not found in cached markets. Reloading markets...")
                exchange.load_markets(reload=True)
                # Check again after reload
                if symbol not in exchange.markets:
                    # If it's still not found after reload, give up for this attempt
                    if attempts == 0:
                        lg.warning(f"Symbol '{symbol}' not found even after market reload. Will retry.")
                        # Continue to the retry mechanism below
                    else:
                         lg.error(f"{NEON_RED}Market '{symbol}' not found after reload and retries.{RESET}")
                         return None  # Stop retrying if not found after reload

            market = exchange.market(symbol)
            if market:
                # Enhance market dict with useful flags
                market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
                market['is_linear'] = market.get('linear', False) and market['is_contract']
                market['is_inverse'] = market.get('inverse', False) and market['is_contract']
                market['contract_type_str'] = "Linear" if market['is_linear'] else \
                                              "Inverse" if market['is_inverse'] else \
                                              "Spot" if market.get('spot', False) else "Unknown"

                # Log key market details for debugging
                def fmt_val(v): return str(Decimal(str(v)).normalize()) if v is not None else 'N/A'
                precision = market.get('precision', {})
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                cost_limits = limits.get('cost', {})

                lg.debug(f"Market Info Retrieved: {symbol} (ID={market.get('id')}, Type={market.get('type')}, Contract Type={market['contract_type_str']})")
                lg.debug(f"  Precision: Price={fmt_val(precision.get('price'))}, Amount={fmt_val(precision.get('amount'))}")
                lg.debug(f"  Limits (Amount): Min={fmt_val(amount_limits.get('min'))}, Max={fmt_val(amount_limits.get('max'))}")
                lg.debug(f"  Limits (Cost): Min={fmt_val(cost_limits.get('min'))}, Max={fmt_val(cost_limits.get('max'))}")
                lg.debug(f"  Contract Size: {fmt_val(market.get('contractSize', '1'))}")  # Default to 1 if not present

                # Critical check: Ensure necessary precision is available
                if precision.get('price') is None or precision.get('amount') is None:
                    lg.error(f"{NEON_RED}CRITICAL: Market '{symbol}' is missing required price or amount precision information! Trading may fail or be inaccurate.{RESET}")
                    # Depending on strictness, you might want to return None here
                    # return None

                return market
            else:
                # This case should be rare if symbol is in exchange.markets
                lg.error(f"{NEON_RED}Market dictionary is None for symbol '{symbol}' despite being listed.{RESET}")
                return None

        except ccxt.BadSymbol as e:
            # Symbol is definitively invalid according to the exchange
            lg.error(f"{NEON_RED}Symbol '{symbol}' is invalid or not supported by {exchange.id}: {e}{RESET}")
            return None  # No point retrying
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES:
                lg.warning(f"{NEON_YELLOW}Network error getting market info for {symbol} (Attempt {attempts + 1}): {e}. Retrying...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                lg.error(f"{NEON_RED}Max retries reached getting market info for {symbol} due to network errors: {e}{RESET}")
                return None
        except ccxt.ExchangeError as e:
             lg.error(f"{NEON_RED}Exchange error getting market info for {symbol}: {e}{RESET}")
             # Potentially add checks for specific error codes if needed
             return None  # Usually not retryable
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True)
            return None  # Stop on unexpected errors

        attempts += 1

    return None  # Should only be reached if all retries failed


def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Decimal | None:
    """Fetches available balance for a specific currency, handling Bybit V5 account types."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None
            available_balance_str = None
            found_structure = False

            # Bybit V5 uses account types (UNIFIED, CONTRACT, SPOT, etc.)
            # Try common account types relevant for derivatives/unified trading first
            account_types_to_try = ['UNIFIED', 'CONTRACT']  # Prioritize these

            for acc_type in account_types_to_try:
                try:
                    lg.debug(f"Fetching balance (Attempt {attempts + 1}, Account Type: {acc_type}) for {currency}...")
                    # Pass accountType in params for Bybit V5
                    params = {'accountType': acc_type} if 'bybit' in exchange.id.lower() else {}
                    balance_info = exchange.fetch_balance(params=params)

                    # Check standard CCXT structure first
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        available_balance_str = str(balance_info[currency]['free'])
                        lg.debug(f"Found balance in standard structure for {acc_type}: {available_balance_str}")
                        found_structure = True
                        break  # Found it, stop checking account types

                    # Check Bybit V5 specific 'info' structure (list of accounts)
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        lg.debug(f"Checking Bybit V5 'info.result.list' structure for {acc_type}...")
                        for account in balance_info['info']['result']['list']:
                             # Ensure we're looking at the correct account type if specified
                             if account.get('accountType') == acc_type and isinstance(account.get('coin'), list):
                                for coin_data in account['coin']:
                                    if coin_data.get('coin') == currency:
                                        # Try different fields for available balance in V5
                                        free = coin_data.get('availableToWithdraw') or \
                                               coin_data.get('availableBalance') or \
                                               coin_data.get('walletBalance')  # Last resort
                                        if free is not None:
                                            available_balance_str = str(free)
                                            lg.debug(f"Found balance in V5 list structure for {acc_type}: {available_balance_str} (field: {'availableToWithdraw' if coin_data.get('availableToWithdraw') else 'availableBalance' if coin_data.get('availableBalance') else 'walletBalance'})")
                                            found_structure = True
                                            break  # Found coin data
                                if found_structure: break  # Found in this account
                        if found_structure: break  # Found in the list for this acc_type

                except (ccxt.ExchangeError) as e:
                    # Log specific errors like "account type not supported" as debug/warning
                    lg.debug(f"API error fetching balance for account type '{acc_type}': {e}. Trying next type.")
                except Exception as e:
                    # Catch other unexpected errors during fetch for a specific type
                    lg.warning(f"Unexpected error fetching balance for account type '{acc_type}': {e}.", exc_info=True)

            # If not found in prioritized types, try default fetch (might work for SPOT or older API versions)
            if not found_structure:
                try:
                    lg.debug(f"Balance not found in {account_types_to_try}. Trying default fetch_balance...")
                    balance_info = exchange.fetch_balance()  # No params

                    # Re-check standard and V5 structures in the default response
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        available_balance_str = str(balance_info[currency]['free'])
                        found_structure = True
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             if isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                     if coin_data.get('coin') == currency:
                                         free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free is not None:
                                             available_balance_str = str(free)
                                             found_structure = True; break
                                 if found_structure: break
                             if found_structure: break

                except Exception as e:
                    lg.error(f"{NEON_RED}Failed during default balance fetch attempt: {e}{RESET}", exc_info=True)

            # Process the result if found
            if found_structure and available_balance_str is not None:
                try:
                    final_balance = Decimal(available_balance_str)
                    lg.debug(f"Successfully parsed balance for {currency}: {final_balance.normalize()}")
                    # Return 0 if negative balance reported (shouldn't happen for 'free')
                    return final_balance if final_balance >= Decimal('0') else Decimal('0')
                except (InvalidOperation, ValueError, TypeError) as conv_err:
                    # Raise an error that will be caught by the retry mechanism
                    raise ccxt.ExchangeError(f"Failed to convert fetched balance string '{available_balance_str}' to Decimal for {currency}: {conv_err}")
            else:
                # Raise an error if balance wasn't found after trying all methods
                 raise ccxt.ExchangeError(f"Balance information not found for currency '{currency}' in any expected structure.")

        # --- Exception Handling for the outer loop ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching balance: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Retry without incrementing attempts immediately
        except ccxt.AuthenticationError as e:
            # Critical: API keys likely invalid or insufficient permissions
            lg.critical(f"{NEON_RED}Authentication Error fetching balance: {e}. Check API key permissions.{RESET}")
            return None  # Stop retrying
        except ccxt.ExchangeError as e:
            # Catch ExchangeErrors raised internally (e.g., conversion fail, not found) or by CCXT
            lg.warning(f"{NEON_YELLOW}Exchange error fetching balance: {e}. Retrying...{RESET}")
            # Potentially check for specific non-retryable codes here
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
            # Decide if unexpected errors should be retried or not; here we retry.

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             # Exponential backoff might be better, but simple delay for now
             time.sleep(RETRY_DELAY_SECONDS * attempts)

    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict | None:
    """Checks for an open position for the given symbol using fetch_positions with Bybit V5 handling."""
    lg = logger
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})")
            positions: list[dict] = []
            market_id = None
            category = None  # For Bybit V5 params

            # --- Determine Market ID and Category (Best Effort) ---
            try:
                market = exchange.market(symbol)
                market_id = market['id']
                # Determine category based on market type for Bybit V5 API call
                if 'bybit' in exchange.id.lower():
                    category = 'linear' if market.get('linear') else \
                               'inverse' if market.get('inverse') else \
                               'spot' if market.get('spot') else 'linear'  # Default to linear if unsure
                    lg.debug(f"Determined category for position fetch: {category}")
            except Exception as market_err:
                 lg.warning(f"Could not get market info to determine category for position fetch ({market_err}). Assuming 'linear'.")
                 category = 'linear'  # Fallback category
                 market_id = symbol  # Use the input symbol if market lookup failed

            # --- Fetch Positions ---
            try:
                # Prefer fetching specific symbol if supported and category is known
                params = {}
                if category and market_id:
                     params = {'category': category, 'symbol': market_id}
                     lg.debug(f"Attempting fetch_positions with specific symbol params: {params}")
                     # Note: CCXT might internally fetch all and filter if the exchange doesn't support symbol filtering
                     positions = exchange.fetch_positions([symbol], params=params)
                else:  # Fallback if category/market_id determination failed
                     lg.debug("Falling back to fetching all positions (no symbol filter).")
                     all_positions = exchange.fetch_positions(params={'category': category} if category else {})  # Fetch all for the category or default
                     # Filter manually
                     positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
                     lg.debug(f"Fetched {len(all_positions)} total positions, found {len(positions)} matching symbol/market_id.")

            except ccxt.ArgumentsRequired as e:
                 # Some exchanges *require* fetching all positions
                 lg.warning(f"Exchange requires fetching all positions ({e}). This might be slower.")
                 params = {'category': category} if category else {}  # Fetch all for the determined/fallback category
                 all_positions = exchange.fetch_positions(params=params)
                 # Filter manually
                 positions = [p for p in all_positions if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
                 lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} for symbol/market_id.")

            except ccxt.ExchangeError as e:
                 # Handle specific "position not found" errors gracefully
                 no_pos_codes_v5 = [110025]  # Bybit V5 code for "position not found"
                 err_str = str(e).lower()
                 # Check Bybit code or common error substrings
                 if (hasattr(e, 'code') and e.code in no_pos_codes_v5) or \
                    "position not found" in err_str or \
                    "no position found" in err_str:
                      lg.info(f"No open position found for {symbol} based on exchange response ({e}).")
                      return None  # Explicitly no position found
                 # Re-raise other exchange errors to be handled by the retry mechanism
                 else:
                      raise e

            # --- Process Fetched Positions ---
            active_position = None
            # Determine a sensible threshold to consider a position 'open' (e.g., 1/10th of min amount precision)
            size_threshold = Decimal('1e-9')  # Default tiny threshold
            try:
                market = exchange.market(symbol)  # Re-fetch market if needed
                amount_prec_str = market.get('precision', {}).get('amount')
                if amount_prec_str:
                    # Set threshold slightly above zero based on precision
                    size_threshold = Decimal(str(amount_prec_str)) * Decimal('0.1')
            except Exception as market_err:
                lg.debug(f"Could not get market precision for position size threshold ({market_err}), using default: {size_threshold}")
            lg.debug(f"Using position size threshold: {size_threshold}")

            for pos in positions:
                # Bybit V5 often has size in 'info'.'size', CCXT standard is 'contracts' or derived
                pos_size_str = str(pos.get('info', {}).get('size', pos.get('contracts', '')))
                if not pos_size_str:
                    lg.debug(f"Skipping position entry, no size found: Info={pos.get('info', {})}, Contracts={pos.get('contracts')}")
                    continue
                try:
                    position_size = Decimal(pos_size_str)
                    # Check if the absolute size exceeds the threshold
                    if abs(position_size) > size_threshold:
                        lg.debug(f"Found potential active position with size {position_size} (Threshold: {size_threshold}).")
                        active_position = pos
                        break  # Assume only one position per symbol/category
                except (InvalidOperation, ValueError, TypeError) as parse_err:
                    lg.warning(f"Could not parse position size '{pos_size_str}' to Decimal: {parse_err}. Skipping.")
                    continue

            # --- Format and Return Active Position ---
            if active_position:
                # Standardize the position dictionary using info from CCXT and exchange-specific 'info'
                std_pos = active_position.copy()
                info_dict = std_pos.get('info', {})  # Bybit V5 raw info

                # Ensure size is Decimal
                std_pos['size_decimal'] = position_size  # Use the already parsed Decimal size

                # Determine Side (handle inconsistencies)
                side = std_pos.get('side')  # Standard CCXT side ('long'/'short')
                if side not in ['long', 'short']:
                    pos_side_v5 = info_dict.get('side', '').lower()  # Bybit V5 side ('Buy'/'Sell')
                    if pos_side_v5 == 'buy': side = 'long'
                    elif pos_side_v5 == 'sell': side = 'short'
                    # Fallback: Infer side from size if standard/V5 side is missing/invalid
                    elif std_pos['size_decimal'] > size_threshold: side = 'long'
                    elif std_pos['size_decimal'] < -size_threshold: side = 'short'
                    else:
                         lg.warning(f"Position size {std_pos['size_decimal']} is near zero or side is ambiguous. Treating as no position.")
                         return None
                std_pos['side'] = side

                # Standardize other key fields, preferring V5 'info' if available
                std_pos['entryPrice'] = std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice')  # V5 uses avgPrice
                std_pos['leverage'] = std_pos.get('leverage') or info_dict.get('leverage')
                std_pos['liquidationPrice'] = std_pos.get('liquidationPrice') or info_dict.get('liqPrice')  # V5 uses liqPrice
                std_pos['unrealizedPnl'] = std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl')  # V5 spelling

                # Extract protection levels (SL, TP, TSL) - these are often in 'info'
                sl_price_str = info_dict.get('stopLoss') or std_pos.get('stopLossPrice')
                tp_price_str = info_dict.get('takeProfit') or std_pos.get('takeProfitPrice')
                tsl_distance_str = info_dict.get('trailingStop')  # V5 TSL distance
                tsl_activation_str = info_dict.get('activePrice')  # V5 TSL activation price

                # Store as strings, let consuming functions parse to Decimal if needed
                if sl_price_str is not None and str(sl_price_str) != '0': std_pos['stopLossPrice'] = str(sl_price_str)
                if tp_price_str is not None and str(tp_price_str) != '0': std_pos['takeProfitPrice'] = str(tp_price_str)
                if tsl_distance_str is not None and str(tsl_distance_str) != '0': std_pos['trailingStopLoss'] = str(tsl_distance_str)
                if tsl_activation_str is not None and str(tsl_activation_str) != '0': std_pos['tslActivationPrice'] = str(tsl_activation_str)

                # Log the found active position details
                def fmt_log(val_str, p_type='price', p_def=4):
                    """Helper to format Decimal strings for logging, using market precision."""
                    if val_str is None or str(val_str).strip() == '': return 'N/A'
                    s_val = str(val_str).strip()
                    if s_val == '0': return '0'  # Avoid formatting "0" strangely
                    try:
                        d = Decimal(s_val)
                        prec = p_def  # Default precision
                        market = None
                        with contextlib.suppress(Exception): market = exchange.market(symbol)
                        if market:
                            prec_val = market.get('precision', {}).get(p_type)  # 'price' or 'amount'
                            if prec_val is not None:
                                try:
                                    # Calculate decimal places from precision step
                                    step = Decimal(str(prec_val))
                                    prec = 0 if step == step.to_integral_value() else abs(step.normalize().as_tuple().exponent)
                                except Exception: pass  # Use default prec if calculation fails
                        # Quantize and normalize for clean output
                        exp = Decimal('1e-' + str(prec))
                        return str(d.quantize(exp, rounding=ROUND_DOWN).normalize())  # Use normalize() to remove trailing zeros
                    except Exception: return s_val  # Return original string if formatting fails

                ep = fmt_log(std_pos.get('entryPrice'))
                size = fmt_log(abs(std_pos['size_decimal']), 'amount')
                liq = fmt_log(std_pos.get('liquidationPrice'))
                lev = fmt_log(std_pos.get('leverage'), 'price', 1) + 'x' if std_pos.get('leverage') else 'N/A'
                pnl = fmt_log(std_pos.get('unrealizedPnl'), 'price', 4)  # PNL often needs more precision
                sl = fmt_log(std_pos.get('stopLossPrice'))
                tp = fmt_log(std_pos.get('takeProfitPrice'))
                tsl_d = fmt_log(std_pos.get('trailingStopLoss'))  # TSL distance is a price difference
                tsl_a = fmt_log(std_pos.get('tslActivationPrice'))

                logger.info(f"{NEON_GREEN}Active {side.upper()} position found ({symbol}):{RESET} "
                            f"Size={size}, Entry={ep}, Liq={liq}, Lev={lev}, PnL={pnl}, "
                            f"SL={sl}, TP={tp}, TSL(Dist/Act): {tsl_d}/{tsl_a}")

                return std_pos  # Return the standardized position dictionary
            else:
                lg.info(f"No active position found for {symbol} after checking {len(positions)} fetched entries.")
                return None  # No position found matching criteria

        # --- Exception Handling for the outer loop ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            lg.warning(f"{NEON_YELLOW}Network error fetching position: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded fetching position: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Retry without incrementing attempts
        except ccxt.AuthenticationError as e:
            lg.critical(f"{NEON_RED}Authentication Error fetching position: {e}. Stopping.{RESET}")
            return None  # Fatal, stop retrying
        except ccxt.ExchangeError as e:
            # Log other exchange errors and decide whether to retry
            lg.warning(f"{NEON_YELLOW}Exchange error fetching position: {e}. Retrying...{RESET}")
            # Check for specific Bybit errors that might indicate configuration issues
            bybit_code = getattr(e, 'code', None)
            if bybit_code in [110004]:  # Account not found / key linking issue?
                 lg.critical(f"{NEON_RED}Possible Bybit Account Error ({bybit_code}): {e}. Check API key link/permissions. Stopping.{RESET}")
                 return None
            if bybit_code in [110013]:  # Parameter error
                 lg.error(f"{NEON_RED}Bybit Parameter Error ({bybit_code}) fetching position: {e}. Check symbol/category validity. Stopping.{RESET}")
                 return None
            # Add other potentially non-retryable codes here
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error fetching position: {e}{RESET}", exc_info=True)
            # Decide whether to retry unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Simple backoff

    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets leverage for a derivatives symbol using CCXT, with Bybit V5 specifics and retries."""
    lg = logger
    is_contract = market_info.get('is_contract', False)

    if not is_contract:
        lg.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True  # Not applicable, consider it successful
    if leverage <= 0:
        lg.warning(f"Leverage setting skipped for {symbol} (Invalid leverage value: {leverage}).")
        return False  # Invalid input
    if not exchange.has.get('setLeverage'):
        lg.error(f"Exchange {exchange.id} does not support setLeverage method.")
        return False

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Setting leverage for {symbol} to {leverage}x (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")

            params = {}
            market_id = market_info.get('id', symbol)  # Use specific market ID if available

            # --- Bybit V5 Specific Parameters ---
            # Bybit V5 requires category and separate buy/sell leverage for setLeverage endpoint
            if 'bybit' in exchange.id.lower():
                 category = 'linear' if market_info.get('is_linear', True) else 'inverse'  # Determine category from market info
                 # V5 endpoint uses 'buyLeverage' and 'sellLeverage'
                 # Assuming ISOLATED margin mode, set both to the same value.
                 # For CROSS margin, only one needs to be set, but setting both is usually safe.
                 params = {
                     'category': category,
                     'symbol': market_id,  # Required for V5 endpoint via private_post
                     'buyLeverage': str(leverage),
                     'sellLeverage': str(leverage)
                 }
                 lg.debug(f"Using Bybit V5 leverage parameters: {params}")
                 # Note: CCXT's set_leverage might handle some of this automatically,
                 # but providing explicit params ensures compatibility with direct V5 endpoint usage if CCXT wrapper changes.

            # --- Execute set_leverage ---
            # CCXT's `set_leverage` might internally call a different endpoint than the one requiring the params above.
            # We provide `params` for exchanges like Bybit that might need them via the generic call.
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)

            # --- Response Handling (Bybit V5 Specific) ---
            # Bybit V5 responses often have 'retCode' and 'retMsg'
            lg.debug(f"Set leverage raw response: {response}")  # Log raw response for debugging
            ret_code = response.get('retCode') if isinstance(response, dict) else None

            if ret_code is not None:  # Check if it looks like a Bybit V5 response
                 if ret_code == 0:
                     lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (Bybit Code: 0).{RESET}")
                     return True
                 elif ret_code == 110045:  # Bybit: Leverage not modified
                     lg.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Bybit Code: 110045 - Not modified).{RESET}")
                     return True
                 else:
                     # Raise an ExchangeError with Bybit's message for other codes
                     error_message = response.get('retMsg', 'Unknown Bybit API error')
                     raise ccxt.ExchangeError(f"Bybit API error setting leverage: {error_message} (Code: {ret_code})")
            else:
                 # Assume success if no error and no specific Bybit code indicates failure
                 lg.info(f"{NEON_GREEN}Leverage set successfully for {symbol} to {leverage}x (Standard CCXT success).{RESET}")
                 return True

        # --- Exception Handling ---
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None)  # Attempt to get Bybit code if available
            err_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage: {e} (Code: {bybit_code}){RESET}")

            # Check for "Leverage not modified" variations
            if bybit_code == 110045 or "not modified" in err_str or "leverage same" in err_str:
                lg.info(f"{NEON_YELLOW}Leverage already set, considering successful.{RESET}")
                return True  # Treat as success

            # Define non-retryable errors (logic errors, risk limits, parameter issues, auth)
            # These codes are examples and might need adjustment based on Bybit documentation/experience
            non_retry_codes = [
                110028,  # Cross/Isolated margin mode cannot be modified
                110009,  # Position exists, cannot modify leverage (depending on mode/exchange rules)
                110055,  # Risk limit exceeded
                110043,  # Leverage exceeds risk limit
                110044,  # Invalid leverage value (out of range)
                110013,  # Parameter error (e.g., invalid symbol/category)
                10001,  # Generic parameter error
                10004,  # Authentication error / Sign error
                # Add other known fatal error codes here
            ]
            # Check for codes or common substrings indicating non-retryable issues
            if bybit_code in non_retry_codes or any(s in err_str for s in ["margin mode", "position exists", "risk limit", "parameter error", "invalid leverage"]):
                lg.error(" >> Hint: Non-retryable leverage error detected. Check margin mode (Isolated/Cross), existing positions, risk limits, or leverage value.")
                return False  # Stop retrying

            # If potentially retryable and haven't exceeded retries
            elif attempts >= MAX_API_RETRIES:
                lg.error(f"Max retries reached for ExchangeError setting leverage: {e}")
                return False
            # Otherwise, the loop will continue and retry after delay

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                lg.error(f"Max retries reached for network error setting leverage: {e}")
                return False
            lg.warning(f"{NEON_YELLOW}Network error setting leverage (Attempt {attempts + 1}): {e}. Retrying...{RESET}")

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting leverage: {e}{RESET}", exc_info=True)
            # Decide if unexpected errors are retryable; stopping here for safety
            return False

        attempts += 1
        if attempts <= MAX_API_RETRIES:
            time.sleep(RETRY_DELAY_SECONDS * attempts)  # Simple backoff

    lg.error(f"{NEON_RED}Failed to set leverage for {symbol} to {leverage}x after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False


def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Decimal | None:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger
    symbol = market_info['symbol']
    quote = market_info['quote']  # e.g., USDT
    base = market_info['base']   # e.g., BTC
    is_contract = market_info['is_contract']
    is_inverse = market_info.get('is_inverse', False)  # Check if it's an inverse contract

    # Units for size calculation
    size_unit = "Contracts" if is_contract else base  # Base currency units for Spot

    # --- Input Validation ---
    if balance <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Invalid or zero balance provided ({balance}).")
        return None
    risk_dec = Decimal(str(risk_per_trade))
    if not (Decimal('0') < risk_dec <= Decimal('1')):
         lg.error(f"Position sizing failed for {symbol}: Invalid risk_per_trade ({risk_per_trade}). Must be > 0 and <= 1.")
         return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Entry price ({entry_price}) or Stop Loss price ({initial_stop_loss_price}) must be positive.")
        return None
    if initial_stop_loss_price == entry_price:
        lg.error(f"Position sizing failed for {symbol}: Stop Loss price cannot be equal to Entry price.")
        return None

    # --- Get Market Precision & Limits ---
    try:
        precision = market_info['precision']
        limits = market_info['limits']
        amt_prec_str = precision['amount']  # Step size for amount (e.g., '0.001')
        price_prec_str = precision['price']  # Step size for price (e.g., '0.01')
        if amt_prec_str is None or price_prec_str is None:
            raise ValueError("Market is missing amount or price precision.")

        amt_prec_step = Decimal(str(amt_prec_str))
        # price_prec_step = Decimal(str(price_prec_str)) # Not directly used in size calc, but good to have

        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        min_amt = Decimal(str(amount_limits.get('min', '0')))
        max_amt = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')

        # Contract size (value of 1 contract in quote currency for linear, or base for inverse)
        # Default to '1' if not specified (common for spot or if info is missing)
        contract_size_str = market_info.get('contractSize', '1')
        contract_size = Decimal('1')
        try:
             cs_val = Decimal(str(contract_size_str))
             if cs_val > 0: contract_size = cs_val
             else: raise ValueError("Contract size must be positive")
        except (ValueError, InvalidOperation, TypeError) as e:
             lg.warning(f"Invalid contract size '{contract_size_str}' found in market info for {symbol}, using default '1': {e}")
             contract_size = Decimal('1')

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Position sizing failed for {symbol}: Error accessing/converting market precision/limits/contractSize: {e}")
        return None

    # --- Risk Calculation ---
    risk_amt_quote = balance * risk_dec  # Amount of quote currency to risk
    sl_dist_price = abs(entry_price - initial_stop_loss_price)  # Stop distance in price terms
    if sl_dist_price <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Stop Loss distance is zero or negative.")  # Should be caught earlier, but double-check
        return None

    lg.info(f"Position Sizing Calculation ({symbol}):")
    lg.info(f"  Balance: {balance.normalize()} {quote}")
    lg.info(f"  Risk %: {risk_dec:.2%}")
    lg.info(f"  Risk Amount: {risk_amt_quote.normalize()} {quote}")
    lg.info(f"  Entry Price: {entry_price.normalize()}")
    lg.info(f"  Stop Loss Price: {initial_stop_loss_price.normalize()}")
    lg.info(f"  Stop Distance (Price): {sl_dist_price.normalize()}")
    lg.info(f"  Contract Type: {'Inverse' if is_inverse else 'Linear/Spot'}")
    lg.info(f"  Contract Size: {contract_size.normalize()}")
    lg.info(f"  Amount Precision Step: {amt_prec_step.normalize()}")

    # --- Calculate Raw Size based on Risk ---
    calculated_size = Decimal('0')
    try:
        if not is_inverse:
            # --- Linear Contract / Spot Calculation ---
            # Formula: Size (in base/contracts) = Risk Amount (Quote) / Stop Distance (Quote per base/contract)
            # For linear contracts, stop distance in price directly represents quote currency per contract.
            # For spot, stop distance is quote currency per base currency.
            # Need to consider contract size if it's not 1 (though often 1 for linear futures)
            # Value change per contract/base unit = sl_dist_price * contract_size (if contract size is in quote) - simplified here assuming contractSize=1 or spot
            # If contract size represents base units (e.g. 1 contract = 0.001 BTC), the logic is different.
            # Assuming standard linear perpetuals/futures or spot:
            value_change_per_unit = sl_dist_price * contract_size  # How much quote value changes for 1 unit change in price
            if value_change_per_unit <= Decimal('0'):
                lg.error("Sizing failed (Linear/Spot): Calculated value change per unit is zero or negative.")
                return None
            calculated_size = risk_amt_quote / value_change_per_unit
            lg.debug(f"  Raw Linear/Spot Size Calc: {risk_amt_quote.normalize()} / ({sl_dist_price.normalize()} * {contract_size.normalize()}) = {calculated_size.normalize()}")

        else:
            # --- Inverse Contract Calculation ---
            # Formula: Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = Contract Size (Base) * | Price Change in Quote per Base |
            # Price Change (Quote/Base) = | (Quote/Base at Entry) - (Quote/Base at SL) | = | 1/Entry - 1/SL | * QuoteValuePerContractInQuote -> This is complex.
            # Simpler: Value of 1 Contract = Contract Size (Base) / Price (Quote/Base)
            # Change in Value per Contract = | Value at Entry - Value at SL |
            # Change in Value per Contract = | (Contract Size / Entry Price) - (Contract Size / SL Price) |
            # Change in Value per Contract = Contract Size * | 1/Entry Price - 1/SL Price | (Value is in Quote currency)
            lg.debug("Inverse contract sizing calculation.")
            if entry_price > 0 and initial_stop_loss_price > 0:
                inv_factor = abs((Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price))
                if inv_factor <= Decimal('1e-18'):  # Use a small threshold instead of zero check
                    lg.error("Sizing failed (Inverse): Calculated inverse factor is zero or extremely small.")
                    return None
                risk_per_contract_quote = contract_size * inv_factor
                if risk_per_contract_quote <= Decimal('0'):
                     lg.error("Sizing failed (Inverse): Calculated risk per contract is zero or negative.")
                     return None
                calculated_size = risk_amt_quote / risk_per_contract_quote
                lg.debug(f"  Raw Inverse Size Calc: RiskPerContract = {contract_size.normalize()} * |1/{entry_price.normalize()} - 1/{initial_stop_loss_price.normalize()}| = {risk_per_contract_quote.normalize()}")
                lg.debug(f"  Raw Inverse Size = {risk_amt_quote.normalize()} / {risk_per_contract_quote.normalize()} = {calculated_size.normalize()}")
            else:
                lg.error("Sizing failed (Inverse): Entry or SL price is zero or negative.")
                return None
    except (OverflowError, InvalidOperation) as calc_err:
        lg.error(f"Position sizing failed for {symbol}: Arithmetic error during raw size calculation: {calc_err}")
        return None

    if calculated_size <= Decimal('0'):
        lg.error(f"Position sizing failed for {symbol}: Initial calculated size is zero or negative ({calculated_size.normalize()}). Check inputs and risk settings.")
        return None

    lg.info(f"  Initial Calculated Size = {calculated_size.normalize()} {size_unit}")

    # --- Apply Market Limits and Precision ---
    adj_size = calculated_size

    # 1. Apply Amount Limits (Min/Max)
    if min_amt > 0 and adj_size < min_amt:
        lg.warning(f"{NEON_YELLOW}Calculated size {adj_size.normalize()} is below minimum amount {min_amt.normalize()}. Adjusting size UP to minimum.{RESET}")
        adj_size = min_amt
    if max_amt < Decimal('inf') and adj_size > max_amt:
        lg.warning(f"{NEON_YELLOW}Calculated size {adj_size.normalize()} exceeds maximum amount {max_amt.normalize()}. Adjusting size DOWN to maximum.{RESET}")
        adj_size = max_amt
    lg.debug(f"  Size after Amount Limits: {adj_size.normalize()} {size_unit}")

    # 2. Estimate Cost and Apply Cost Limits (Min/Max)
    est_cost = Decimal('0')
    cost_adjusted = False
    try:
        if entry_price > 0:
            # Cost for Linear/Spot: Size * Price * ContractSize (if ContractSize is in base units, adjust formula)
            # Cost for Inverse: Size * ContractSize / Price (Result is in Quote currency)
            # Assuming standard contractSize definitions (1 for spot/linear, base units for inverse)
            if not is_inverse:
                est_cost = adj_size * entry_price  # For spot
                if is_contract: est_cost = adj_size * entry_price * contract_size  # Need confirmation on linear contractSize definition
            else:  # Inverse
                est_cost = (adj_size * contract_size) / entry_price
        lg.debug(f"  Estimated Cost (@ Entry Price): {est_cost.normalize()} {quote}")

        # Check Min Cost
        if min_cost > 0 and est_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {est_cost.normalize()} is below minimum cost {min_cost.normalize()}. Attempting to increase size.{RESET}")
            required_size_for_min_cost = None
            # Recalculate required size: Size = Cost / (Price * ContractSize) [Linear] or Size = Cost * Price / ContractSize [Inverse]
            try:
                if entry_price > 0 and contract_size > 0:
                    if not is_inverse:
                         required_size_for_min_cost = min_cost / (entry_price * contract_size)  # Adjust if linear contractSize != 1
                    else:
                         required_size_for_min_cost = (min_cost * entry_price) / contract_size
            except (OverflowError, InvalidOperation): pass

            if required_size_for_min_cost is None or required_size_for_min_cost <= 0:
                lg.error(f"{NEON_RED}Cannot meet minimum cost {min_cost.normalize()} {quote}. Calculation failed or resulted in non-positive size. Aborted.{RESET}")
                return None
            lg.info(f"  Required size to meet min cost: {required_size_for_min_cost.normalize()} {size_unit}")

            # Ensure the required size doesn't violate max amount limit
            if max_amt < Decimal('inf') and required_size_for_min_cost > max_amt:
                lg.error(f"{NEON_RED}Cannot meet minimum cost {min_cost.normalize()} without exceeding maximum amount limit {max_amt.normalize()}. Aborted.{RESET}")
                return None

            # Adjust size up to the required size (or min_amt if that's larger)
            adj_size = max(min_amt, required_size_for_min_cost)
            cost_adjusted = True

        # Check Max Cost
        elif max_cost < Decimal('inf') and est_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Estimated cost {est_cost.normalize()} exceeds maximum cost {max_cost.normalize()}. Attempting to reduce size.{RESET}")
            max_size_for_max_cost = None
            # Recalculate max size: Size = Cost / (Price * ContractSize) [Linear] or Size = Cost * Price / ContractSize [Inverse]
            try:
                 if entry_price > 0 and contract_size > 0:
                    if not is_inverse:
                        max_size_for_max_cost = max_cost / (entry_price * contract_size)  # Adjust if linear contractSize != 1
                    else:
                        max_size_for_max_cost = (max_cost * entry_price) / contract_size
            except (OverflowError, InvalidOperation): pass

            if max_size_for_max_cost is None or max_size_for_max_cost <= 0:
                 lg.error(f"{NEON_RED}Cannot reduce size to meet maximum cost {max_cost.normalize()} {quote}. Calculation failed or resulted in non-positive size. Aborted.{RESET}")
                 return None
            lg.info(f"  Maximum size allowed by max cost: {max_size_for_max_cost.normalize()} {size_unit}")

            # Adjust size down to the calculated max size (but not below min_amt)
            adj_size = max(min_amt, min(adj_size, max_size_for_max_cost))  # Take the smaller of current adj_size and max_size_for_cost, ensuring it's >= min_amt
            cost_adjusted = True

        if cost_adjusted:
            lg.info(f"  Size after Cost Limits: {adj_size.normalize()} {size_unit}")

    except (OverflowError, InvalidOperation) as cost_err:
        lg.error(f"Position sizing failed for {symbol}: Arithmetic error during cost estimation/adjustment: {cost_err}")
        return None

    # 3. Apply Amount Precision (Rounding Down to nearest step)
    final_size = adj_size
    try:
        # Use ccxt's amount_to_precision for robustness if available and working
        # It should handle rounding correctly based on exchange rules (usually round down)
        fmt_size_str = exchange.amount_to_precision(symbol, float(adj_size))
        final_size = Decimal(fmt_size_str)
        lg.debug(f"Applied amount precision using ccxt.amount_to_precision: {adj_size.normalize()} -> {final_size.normalize()}")
        # Double check if rounding up occurred unexpectedly (some exchanges might?)
        if final_size > adj_size:
             lg.warning(f"ccxt.amount_to_precision rounded UP? {adj_size} -> {final_size}. Attempting manual round down.")
             # Fallback to manual round down if ccxt method seems wrong
             if amt_prec_step > 0:
                 final_size = (adj_size // amt_prec_step) * amt_prec_step
                 lg.info(f"Applied manual amount precision step ({amt_prec_step}): Rounded down to {final_size.normalize()}")
             else:
                 lg.error(f"{NEON_RED}Amount precision step is zero. Cannot apply manual rounding.{RESET}")
                 # Keep the possibly rounded-up value from ccxt? Or fail? Failing seems safer.
                 return None
    except ccxt.BaseError as fmt_err:
        lg.warning(f"{NEON_YELLOW}ccxt.amount_to_precision failed: {fmt_err}. Using manual rounding.{RESET}")
        # Manual rounding down based on precision step
        try:
            if amt_prec_step > 0:
                final_size = (adj_size // amt_prec_step) * amt_prec_step
                lg.info(f"Applied manual amount precision step ({amt_prec_step}): Rounded down to {final_size.normalize()}")
            else:
                 raise ValueError("Amount precision step is zero.")
        except (ValueError, InvalidOperation, TypeError) as manual_err:
            lg.error(f"{NEON_RED}Manual amount precision failed: {manual_err}. Using limit-adjusted size without precision: {adj_size.normalize()}{RESET}")
            final_size = adj_size  # Fallback to size before precision step
    except Exception as generic_fmt_err:
         lg.error(f"{NEON_RED}Unexpected error applying amount precision: {generic_fmt_err}. Using unrounded size.{RESET}", exc_info=True)
         final_size = adj_size

    # --- Final Validation ---
    if final_size <= Decimal('0'):
        lg.error(f"{NEON_RED}Final position size is zero or negative ({final_size.normalize()}) after applying limits and precision. Aborted.{RESET}")
        return None
    # Re-check against minimum amount limit after precision rounding
    if min_amt > 0 and final_size < min_amt:
        lg.error(f"{NEON_RED}Final size {final_size.normalize()} is less than minimum amount {min_amt.normalize()} after precision rounding. Aborted.{RESET}")
        # Option: Could try bumping up to min_amt here, but risk might increase slightly. Aborting is safer.
        # bump_size = min_amt
        # lg.warning(f"Final size below min amount. Bumping to min_amt: {bump_size}. Risk profile might change slightly.")
        # final_size = bump_size
        return None

    # Re-check minimum cost with the final size (important if rounding down occurred)
    final_cost = Decimal('0')
    try:
        if entry_price > 0:
            if not is_inverse: final_cost = final_size * entry_price * contract_size  # Adjust if linear contractSize != 1
            else: final_cost = (final_size * contract_size) / entry_price
        lg.debug(f"  Final Estimated Cost: {final_cost.normalize()} {quote}")

        if min_cost > 0 and final_cost < min_cost:
             lg.debug(f"Final size {final_size.normalize()} results in cost {final_cost.normalize()} < min cost {min_cost.normalize()}.")
             # Try adding one precision step to see if it meets min cost
             try:
                 next_step_size = final_size + amt_prec_step
                 next_step_cost = Decimal('0')
                 if entry_price > 0:
                     if not is_inverse: next_step_cost = next_step_size * entry_price * contract_size
                     else: next_step_cost = (next_step_size * contract_size) / entry_price

                 # Check if this next step is valid (meets min cost, doesn't exceed max amount/cost)
                 valid_next_step = (next_step_cost >= min_cost) and \
                                   (max_amt == Decimal('inf') or next_step_size <= max_amt) and \
                                   (max_cost == Decimal('inf') or next_step_cost <= max_cost)

                 if valid_next_step:
                     lg.warning(f"{NEON_YELLOW}Final size cost is below minimum. Bumping size by one step to {next_step_size.normalize()} to meet minimum cost.{RESET}")
                     final_size = next_step_size
                 else:
                     lg.error(f"{NEON_RED}Final size cost is below minimum ({min_cost.normalize()}), and increasing size by one step ({next_step_size.normalize()}) is invalid (violates limits or still too low). Aborted.{RESET}")
                     return None
             except Exception as bump_err:
                 lg.error(f"{NEON_RED}Error attempting to bump size by one step for min cost: {bump_err}. Aborted.{RESET}")
                 return None

    except (OverflowError, InvalidOperation) as final_cost_err:
         lg.warning(f"Could not reliably verify final cost against minimum: {final_cost_err}. Proceeding with calculated size.")

    lg.info(f"{NEON_GREEN}{BRIGHT}Final Calculated Position Size: {final_size.normalize()} {size_unit}{RESET}")
    return final_size


def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: dict,
                logger: logging.Logger, reduce_only: bool = False, params: dict | None = None) -> dict | None:
    """Places a market order using CCXT with retries, Bybit V5 params, and clear logging."""
    lg = logger
    side = 'buy' if trade_signal in ["BUY", "EXIT_SHORT"] else 'sell' if trade_signal in ["SELL", "EXIT_LONG"] else None
    order_type = 'market'

    # --- Input Validation ---
    if side is None:
        lg.error(f"Trade placement failed: Invalid trade_signal '{trade_signal}'. Must be BUY, SELL, EXIT_LONG, or EXIT_SHORT.")
        return None
    if position_size <= Decimal('0'):
         lg.error(f"Trade placement failed: Position size must be positive ({position_size}).")
         return None

    # --- Determine Context ---
    is_contract = market_info['is_contract']
    base = market_info['base']
    market_info['quote']
    # Use settle currency for contracts if available, otherwise base
    size_unit = market_info.get('settle', base) if is_contract else base
    action_desc = "Close Position" if reduce_only else "Open/Increase Position"
    market_id = market_info['id']  # Use the exchange-specific market ID

    # --- Prepare Order Parameters ---
    # Basic parameters required by CCXT create_order
    order_args = {
        'symbol': market_id,
        'type': order_type,
        'side': side,
        'amount': float(position_size),  # CCXT usually expects float amount
    }

    # Additional parameters, especially for Bybit V5
    order_params = {}
    if 'bybit' in exchange.id.lower():
        category = 'linear' if market_info.get('is_linear', True) else 'inverse'  # Determine category
        order_params['category'] = category
        # Position index (0 for one-way mode, 1/2 for hedge mode buy/sell - assuming one-way)
        order_params['positionIdx'] = 0
        # Set reduceOnly flag if specified
        if reduce_only:
            order_params['reduceOnly'] = True
            # Use IOC for reduceOnly market orders to avoid leaving passive orders if execution fails partially
            order_params['timeInForce'] = 'IOC'  # ImmediateOrCancel
            # Some exchanges might require specifying close strategy for reduceOnly
            # order_params['closeOnTrigger'] = True # Example, check Bybit docs if needed
        # Allow overriding/adding params
        if params:
            order_params.update(params)
        # Add these to the main create_order call's params argument
        order_args['params'] = order_params

    lg.info(f"Attempting to place {action_desc} order:")
    lg.info(f"  Symbol: {symbol} ({market_id})")
    lg.info(f"  Type: {order_type.upper()}")
    lg.info(f"  Side: {side.upper()}")
    lg.info(f"  Amount: {position_size.normalize()} {size_unit} ({order_args['amount']})")
    if reduce_only: lg.info("  Reduce Only: True")
    if order_args.get('params'): lg.debug(f"  Full Order Params: {order_args['params']}")

    # --- Execute Order with Retries ---
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing exchange.create_order (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            order = exchange.create_order(**order_args)

            # --- Process Successful Order ---
            order_id = order.get('id', 'N/A')
            order_status = order.get('status', 'N/A')  # e.g., 'open', 'closed', 'canceled'
            avg_price = order.get('average')  # Average fill price
            filled_amount = order.get('filled')  # Amount filled

            log_msg = f"{NEON_GREEN}{action_desc} Order Placed Successfully!{RESET} " \
                      f"ID: {order_id}, Status: {order_status}"
            if avg_price:
                log_msg += f", Avg Fill Price: ~{Decimal(str(avg_price)).normalize()}"
            if filled_amount:
                log_msg += f", Filled Amount: {Decimal(str(filled_amount)).normalize()}"
            lg.info(log_msg)
            lg.debug(f"Raw order response: {order}")
            return order  # Return the order details

        # --- Exception Handling ---
        except ccxt.InsufficientFunds as e:
            # Typically non-retryable
            lg.error(f"{NEON_RED}Insufficient Funds placing {action_desc} {side} order: {e}{RESET}")
            # Hint: Check available balance, margin requirements, and potential open orders locking funds.
            lg.error(" >> Hint: Check available balance, margin mode, and other open orders.")
            return None
        except ccxt.InvalidOrder as e:
            # Often due to parameter issues (size precision, limits, etc.) - non-retryable
            lg.error(f"{NEON_RED}Invalid Order placing {action_desc} {side} order: {e}{RESET}")
            # Hint: Check order size against market limits (min/max amount/cost) and precision rules.
            # Also check leverage, position mode compatibility.
            lg.error(" >> Hint: Check size/price precision, limits (min/max amount/cost), leverage, position mode.")
            bybit_code = getattr(e, 'code', None)
            if bybit_code == 30086:  # Bybit: order cost not match filter
                 lg.error(" >> Specific Hint (Bybit 30086): Order cost likely below minimum cost limit.")
            return None
        except ccxt.ExchangeError as e:
            # Catch other specific exchange errors
            bybit_code = getattr(e, 'code', None)
            lg.error(f"{NEON_RED}Exchange error placing {action_desc} {side} order: {e} (Code: {bybit_code}){RESET}")
            # Define known fatal/non-retryable error codes from Bybit docs
            non_retry_codes = [
                110014,  # Reduce-only order会使仓位反向 (Reduce-only order would reverse position)
                110007,  # Order quantity exceeded lower limit.
                110040,  # Order quantity exceeded upper limit.
                110013,  # Parameter error
                110025,  # Position status is not normal (e.g., during liquidation)
                30086,  # order cost not match cost filter (already handled by InvalidOrder but good to have)
                10001,  # Generic Parameter error
                # Add more based on experience/docs (e.g., risk limit errors, auth errors if not caught separately)
            ]
            if bybit_code in non_retry_codes:
                lg.error(" >> Hint: Non-retryable exchange error detected.")
                return None
            # Otherwise, assume potentially retryable (e.g., temporary matching engine issue)

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                lg.error(f"{NEON_RED}Max retries reached for network error placing order: {e}{RESET}")
                return None
            lg.warning(f"{NEON_YELLOW}Network error placing order (Attempt {attempts + 1}): {e}. Retrying...{RESET}")

        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded placing order: {e}. Waiting {wait_time}s...{RESET}")
            time.sleep(wait_time)
            continue  # Retry without incrementing attempts

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error placing {action_desc} {side} order: {e}{RESET}", exc_info=True)
            # Decide if unexpected errors should be retried; stopping here for safety
            return None

        # Increment attempt counter only if it wasn't a rate limit retry
        if not isinstance(e, ccxt.RateLimitExceeded):
             attempts += 1
             if attempts <= MAX_API_RETRIES:
                 time.sleep(RETRY_DELAY_SECONDS * attempts)  # Simple backoff

    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return None


def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict, logger: logging.Logger,
                             stop_loss_price: Decimal | None = None, take_profit_price: Decimal | None = None,
                             trailing_stop_distance: Decimal | None = None, tsl_activation_price: Decimal | None = None) -> bool:
    """Internal helper to set SL/TP/TSL for a position via Bybit V5 API (private_post)."""
    lg = logger

    # --- Input Validation ---
    if not market_info.get('is_contract'):
        lg.warning(f"Protection setting skipped for {symbol}: Not a contract market.")
        return False  # Not applicable
    if not position_info:
        lg.error(f"Protection setting failed for {symbol}: Missing current position information.")
        return False
    pos_side = position_info.get('side')  # 'long' or 'short'
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"Protection setting failed for {symbol}: Invalid position side ('{pos_side}') or missing entry price ('{entry_price_str}').")
        return False
    try:
        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0: raise ValueError("Entry price must be positive")
    except (ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"Protection setting failed for {symbol}: Invalid entry price format or value '{entry_price_str}': {e}")
        return False

    # --- Prepare API Parameters ---
    params_to_set = {}
    log_parts = [f"Preparing protection update for {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):"]
    any_change_requested = False  # Track if any valid parameter is being set

    try:
        # Get price precision for formatting
        price_prec_str = market_info['precision']['price']
        min_tick = Decimal(str(price_prec_str))
        if min_tick <= 0: raise ValueError("Invalid price precision step")

        def fmt_price(price_dec: Decimal | None, param_name: str) -> str | None:
            """Formats price to string based on market precision using ccxt."""
            if price_dec is None: return None
            # Allow "0" to clear existing SL/TP/TSL
            if price_dec == 0: return "0"
            if price_dec < 0:
                lg.warning(f"Cannot format negative price {price_dec} for {param_name}.")
                return None
            try:
                # Use ccxt's price_to_precision for correct rounding/formatting
                formatted_str = exchange.price_to_precision(symbol, float(price_dec))
                # Basic validation on the formatted string
                if Decimal(formatted_str) > 0: return formatted_str
                else: lg.warning(f"Formatted price {formatted_str} for {param_name} is not positive."); return None
            except Exception as e:
                lg.error(f"Failed to format price {price_dec} for {param_name} using exchange precision: {e}.")
                return None

        # 1. Trailing Stop Loss (Bybit V5: 'trailingStop' distance, 'activePrice' activation)
        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal):
            any_change_requested = True
            if trailing_stop_distance > 0:
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0:
                    lg.error(f"TSL requested with distance {trailing_stop_distance}, but activation price is invalid ({tsl_activation_price}). Cannot set TSL.")
                else:
                    # Validate activation price relative to entry
                    valid_act_price = (pos_side == 'long' and tsl_activation_price > entry_price) or \
                                      (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not valid_act_price:
                         lg.error(f"TSL Activation Price {tsl_activation_price.normalize()} is not beyond entry price {entry_price.normalize()} for {pos_side} position. Cannot set TSL.")
                    else:
                        # Ensure distance is at least one tick
                        min_dist = max(trailing_stop_distance, min_tick)
                        fmt_tsl_dist = fmt_price(min_dist, "TSL Distance")
                        fmt_act_price = fmt_price(tsl_activation_price, "TSL Activation")

                        if fmt_tsl_dist and fmt_act_price:
                            params_to_set['trailingStop'] = fmt_tsl_dist
                            params_to_set['activePrice'] = fmt_act_price
                            log_parts.append(f"  - Set TSL: Distance={fmt_tsl_dist}, Activation={fmt_act_price}")
                            set_tsl = True  # TSL parameters are being set
                        else:
                            lg.error(f"Failed to format valid TSL parameters (Distance: {fmt_tsl_dist}, Activation: {fmt_act_price}). Cannot set TSL.")
            elif trailing_stop_distance == 0:  # Explicitly clear TSL
                params_to_set['trailingStop'] = "0"
                # Bybit might require clearing activePrice too, or setting it to 0
                params_to_set['activePrice'] = "0"  # Safer to clear both
                log_parts.append("  - Clear TSL (Distance & Activation set to 0)")
                set_tsl = True  # A TSL-related change is being made
            else:  # Negative distance is invalid
                 lg.error(f"Invalid TSL distance requested: {trailing_stop_distance}. Cannot set TSL.")

        # 2. Fixed Stop Loss (Bybit V5: 'stopLoss') - Only set if TSL is NOT being set
        if not set_tsl and isinstance(stop_loss_price, Decimal):
            any_change_requested = True
            if stop_loss_price > 0:
                # Validate SL relative to entry
                valid_sl = (pos_side == 'long' and stop_loss_price < entry_price) or \
                           (pos_side == 'short' and stop_loss_price > entry_price)
                if not valid_sl:
                    lg.error(f"Stop Loss Price {stop_loss_price.normalize()} is not beyond entry price {entry_price.normalize()} for {pos_side} position. Cannot set SL.")
                else:
                    fmt_sl = fmt_price(stop_loss_price, "Stop Loss")
                    if fmt_sl:
                        params_to_set['stopLoss'] = fmt_sl
                        log_parts.append(f"  - Set Fixed SL: {fmt_sl}")
                    else:
                        lg.error(f"Failed to format valid Stop Loss price {stop_loss_price}. Cannot set SL.")
            elif stop_loss_price == 0:  # Explicitly clear SL
                params_to_set['stopLoss'] = "0"
                log_parts.append("  - Clear Fixed SL (Set to 0)")
            else:  # Negative SL price is invalid
                 lg.error(f"Invalid SL price requested: {stop_loss_price}. Cannot set SL.")

        # 3. Fixed Take Profit (Bybit V5: 'takeProfit') - Can be set alongside SL or TSL
        if isinstance(take_profit_price, Decimal):
             any_change_requested = True
             if take_profit_price > 0:
                 # Validate TP relative to entry
                 valid_tp = (pos_side == 'long' and take_profit_price > entry_price) or \
                            (pos_side == 'short' and take_profit_price < entry_price)
                 if not valid_tp:
                      lg.error(f"Take Profit Price {take_profit_price.normalize()} is not beyond entry price {entry_price.normalize()} for {pos_side} position. Cannot set TP.")
                 else:
                     fmt_tp = fmt_price(take_profit_price, "Take Profit")
                     if fmt_tp:
                         params_to_set['takeProfit'] = fmt_tp
                         log_parts.append(f"  - Set Fixed TP: {fmt_tp}")
                     else:
                          lg.error(f"Failed to format valid Take Profit price {take_profit_price}. Cannot set TP.")
             elif take_profit_price == 0:  # Explicitly clear TP
                 params_to_set['takeProfit'] = "0"
                 log_parts.append("  - Clear Fixed TP (Set to 0)")
             else:  # Negative TP price is invalid
                  lg.error(f"Invalid TP price requested: {take_profit_price}. Cannot set TP.")

    except Exception as fmt_err:
        lg.error(f"Error preparing/formatting protection parameters: {fmt_err}", exc_info=True)
        return False

    # --- Check if any valid parameters were prepared ---
    if not params_to_set:
        if any_change_requested:
            lg.warning("Protection change requested, but no valid parameters could be formatted/validated. No API call made.")
            return False  # Indicate failure as the intent was to change but couldn't
        else:
            lg.debug("No protection changes requested or parameters provided. No API call needed.")
            return True  # Indicate success as no action was required

    # --- Construct Full API Request ---
    # Base parameters for Bybit V5 set-trading-stop endpoint
    category = 'linear' if market_info.get('is_linear', True) else 'inverse'
    market_id = market_info['id']
    # Get position index (usually 0 for one-way mode)
    pos_idx = 0
    try:
        pos_idx_val = position_info.get('info', {}).get('positionIdx')
        if pos_idx_val is not None: pos_idx = int(pos_idx_val)
    except (ValueError, TypeError):
        lg.warning(f"Could not parse positionIdx from position info ({pos_idx_val}), using default {pos_idx}.")

    final_params = {
        'category': category,
        'symbol': market_id,
        'tpslMode': 'Full',         # Or 'Partial' if supported/needed - 'Full' applies to entire position
        'tpTriggerBy': 'LastPrice',  # Or 'MarkPrice', 'IndexPrice'
        'slTriggerBy': 'LastPrice',  # Or 'MarkPrice', 'IndexPrice'
        'tpOrderType': 'Market',    # Or 'Limit'
        'slOrderType': 'Market',    # Or 'Limit'
        'positionIdx': pos_idx      # 0 for one-way, 1/2 for hedge buy/sell
    }
    # Add the specific SL/TP/TSL values prepared earlier
    final_params.update(params_to_set)

    lg.info("\n".join(log_parts))  # Log the summary of changes
    lg.debug("  API Call: private_post /v5/position/set-trading-stop")
    lg.debug(f"  API Params: {final_params}")

    # --- Execute API Call with Retries ---
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing set protection API call (Attempt {attempts + 1}/{MAX_API_RETRIES + 1})...")
            # Use private_post for endpoints not directly mapped by CCXT's unified methods
            response = exchange.private_post('/v5/position/set-trading-stop', params=final_params)
            lg.debug(f"Set protection raw response: {response}")

            # --- Process API Response ---
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown response message')

            if ret_code == 0:
                 # Check if message indicates "not modified" - treat as success
                 if any(msg in ret_msg.lower() for msg in ["not modified", "no need to modify", "parameter not change"]):
                      lg.info(f"{NEON_YELLOW}Protection parameters already set as requested (or no change needed). Response: {ret_msg}{RESET}")
                 else:
                      lg.info(f"{NEON_GREEN}Protection parameters set/updated successfully.{RESET}")
                 return True  # Success
            else:
                # Log the specific error from Bybit
                lg.error(f"{NEON_RED}Failed to set position protection: {ret_msg} (Code: {ret_code}){RESET}")
                # Check for known non-retryable error codes
                fatal_codes = [
                    110013,  # Parameter error (e.g., invalid price/symbol/category)
                    110036,  # TP price cannot be lower than entry price (for long position) / higher for short etc. (Logic errors)
                    110086,  # SL price cannot be higher than entry price (for long position) / lower for short etc.
                    110084,  # TP/SL price invalid
                    110085,  # TP/SL order price deviates too much from market price (risk control)
                    10001,  # Generic Parameter Error
                    10002,  # Request parameter error (missing/invalid format)
                    # Add others related to permissions, position status etc.
                ]
                is_fatal = ret_code in fatal_codes or "invalid price" in ret_msg.lower() or "parameter" in ret_msg.lower()

                if is_fatal:
                    lg.error(" >> Hint: Non-retryable error code received. Check parameters, prices relative to entry, and permissions.")
                    return False  # Stop retrying
                else:
                    # Raise an error to trigger the retry mechanism for potentially transient issues
                    raise ccxt.ExchangeError(f"Bybit API error setting protection: {ret_msg} (Code: {ret_code})")

        # --- Exception Handling for Retries ---
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts >= MAX_API_RETRIES:
                lg.error(f"Max retries reached for network error setting protection: {e}")
                return False
            lg.warning(f"{NEON_YELLOW}Network error setting protection (Attempt {attempts + 1}): {e}. Retrying...")
        except ccxt.RateLimitExceeded as e:
            wait_time = RETRY_DELAY_SECONDS * 5
            lg.warning(f"{NEON_YELLOW}Rate limit exceeded setting protection: {e}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            continue  # Retry without incrementing attempts
        except ccxt.AuthenticationError as e:
            lg.critical(f"{NEON_RED}Authentication Error setting protection: {e}. Stopping.")
            return False  # Fatal
        except ccxt.ExchangeError as e:
             # Catch re-raised errors or other exchange errors not handled above
             if attempts >= MAX_API_RETRIES:
                 lg.error(f"Max retries reached for exchange error setting protection: {e}")
                 return False
             lg.warning(f"{NEON_YELLOW}Exchange error setting protection (Attempt {attempts + 1}): {e}. Retrying...")
        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error setting protection (Attempt {attempts + 1}): {e}", exc_info=True)
            # Decide if unexpected errors should be retried; stopping here for safety
            return False

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS * attempts)  # Simple backoff

    lg.error(f"{NEON_RED}Failed to set protection for {symbol} after {MAX_API_RETRIES + 1} attempts.{RESET}")
    return False


def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict, config: dict[str, Any],
                             logger: logging.Logger, take_profit_price: Decimal | None = None) -> bool:
    """Calculates TSL parameters based on config and current position, then calls _set_position_protection."""
    lg = logger
    protection_cfg = config.get("protection", {})

    # --- Input Validation ---
    if not market_info or not position_info:
        lg.error(f"Cannot calculate TSL for {symbol}: Missing market or position info.")
        return False
    pos_side = position_info.get('side')
    entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str:
        lg.error(f"TSL calculation failed for {symbol}: Invalid position side ('{pos_side}') or missing entry price ('{entry_price_str}').")
        return False

    try:
        entry_price = Decimal(str(entry_price_str))
        # Load TSL config parameters, converting to Decimal
        callback_rate = Decimal(str(protection_cfg["trailing_stop_callback_rate"]))
        activation_percentage = Decimal(str(protection_cfg["trailing_stop_activation_percentage"]))

        # Validate config values
        if not (callback_rate > 0): raise ValueError("Trailing stop callback rate must be positive.")
        if not (activation_percentage >= 0): raise ValueError("Trailing stop activation percentage must be non-negative.")
        if entry_price <= 0: raise ValueError("Entry price must be positive.")

        # Get price precision (tick size)
        price_prec_str = market_info['precision']['price']
        min_tick = Decimal(str(price_prec_str))
        if min_tick <= 0: raise ValueError("Invalid price precision step")

    except (KeyError, ValueError, InvalidOperation, TypeError) as e:
        lg.error(f"{NEON_RED}Invalid TSL configuration or market/position info for {symbol}: {e}. Cannot calculate TSL.{RESET}")
        return False

    # --- Calculate TSL Parameters ---
    try:
        # Calculate activation price offset and raw activation price
        activation_offset = entry_price * activation_percentage
        raw_activation_price = (entry_price + activation_offset) if pos_side == 'long' else (entry_price - activation_offset)

        # Quantize activation price to the nearest tick (away from entry price)
        if pos_side == 'long':
            # Round up, ensuring it's at least one tick above entry
            activation_price = raw_activation_price.quantize(min_tick, rounding=ROUND_UP)
            activation_price = max(activation_price, entry_price + min_tick)
        else:  # Short position
            # Round down, ensuring it's at least one tick below entry
            activation_price = raw_activation_price.quantize(min_tick, rounding=ROUND_DOWN)
            activation_price = min(activation_price, entry_price - min_tick)

        if activation_price <= 0:
             lg.error(f"Calculated TSL Activation Price is zero or negative ({activation_price}). Cannot set TSL.")
             return False

        # Calculate trailing distance based on activation price and callback rate
        # Note: Bybit defines trailingStop as a price distance, not a percentage callback from the peak.
        # We calculate the initial distance based on the activation price.
        # Example: If ActPrice is 105 and CB Rate is 1%, initial trail distance is 105 * 0.01 = 1.05
        raw_trail_distance = activation_price * callback_rate

        # Quantize trail distance (round up to nearest tick, ensure minimum of one tick)
        trail_distance = raw_trail_distance.quantize(min_tick, rounding=ROUND_UP)
        trail_distance = max(trail_distance, min_tick)  # Ensure distance is at least one tick

        if trail_distance <= 0:
            lg.error(f"Calculated TSL Distance is zero or negative ({trail_distance}). Cannot set TSL.")
            return False

        lg.info(f"Calculated TSL Parameters ({symbol}, {pos_side.upper()}):")
        lg.info(f"  Entry Price: {entry_price.normalize()}")
        lg.info(f"  Activation % Config: {activation_percentage:.3%}")
        lg.info(f"  Callback Rate Config: {callback_rate:.3%}")
        lg.info(f"  => Calculated Activation Price: {activation_price.normalize()}")
        lg.info(f"  => Calculated Trailing Distance: {trail_distance.normalize()}")
        if isinstance(take_profit_price, Decimal):
             tp_log = take_profit_price.normalize() if take_profit_price != 0 else 'Cleared (0)'
             lg.info(f"  Take Profit (if provided): {tp_log}")

        # --- Call the internal function to set the protection ---
        # Pass None for fixed stop loss, as we are setting TSL
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None,  # TSL overrides fixed SL on Bybit V5
            take_profit_price=take_profit_price,  # TP can coexist with TSL
            trailing_stop_distance=trail_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Volumatic Trend + OB Strategy Implementation ---
class OrderBlock(TypedDict):
    """Represents a detected Pivot Order Block."""
    id: str             # Unique identifier (e.g., B_2301011200 for Bearish)
    type: str           # 'bull' or 'bear'
    left_idx: pd.Timestamp  # Timestamp of the pivot candle forming the block
    right_idx: pd.Timestamp  # Timestamp of the last candle block extends to (or violation candle)
    top: Decimal        # Top price of the order block zone
    bottom: Decimal     # Bottom price of the order block zone
    active: bool        # Is the block currently considered valid (not violated)?
    violated: bool      # Has the block been invalidated by price action?


class StrategyAnalysisResults(TypedDict):
    """Structured results from the strategy analysis."""
    dataframe: pd.DataFrame      # DataFrame with all indicators
    last_close: Decimal          # Last closing price
    current_trend_up: bool | None  # True if bullish trend, False if bearish, None if undetermined
    trend_just_changed: bool     # True if the trend flipped on the last candle
    active_bull_boxes: list[OrderBlock]  # List of currently active bullish OBs
    active_bear_boxes: list[OrderBlock]  # List of currently active bearish OBs
    vol_norm_int: int | None  # Normalized volume (0-100+, as int)
    atr: Decimal | None       # Last calculated ATR value
    upper_band: Decimal | None  # Last calculated Volumatic Upper Band value
    lower_band: Decimal | None  # Last calculated Volumatic Lower Band value


class VolumaticOBStrategy:
    """Implements the Volumatic Trend indicator combined with Pivot Order Blocks.
    Calculates trend, bands, identifies OBs, and manages their state.
    """
    def __init__(self, config: dict[str, Any], market_info: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.market_info = market_info
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})

        # Load parameters from config, using defaults if missing/invalid (validation done in load_config)
        self.vt_length = int(strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH))
        self.vt_atr_period = int(strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD))
        self.vt_vol_ema_length = int(strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH))
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        # self.vt_step_atr_multiplier = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER))) # Currently unused step line
        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)  # "Wicks" or "Body"
        self.ph_left = int(strategy_cfg.get("ph_left", DEFAULT_PH_LEFT))
        self.ph_right = int(strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT))
        self.pl_left = int(strategy_cfg.get("pl_left", DEFAULT_PL_LEFT))
        self.pl_right = int(strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT))
        self.ob_extend = bool(strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND))
        self.ob_max_boxes = int(strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES))

        # Internal state for order blocks (persists between updates)
        self.bull_boxes: list[OrderBlock] = []
        self.bear_boxes: list[OrderBlock] = []

        # --- Calculate minimum data length required by indicators ---
        # Needs enough data for the longest EMA/ATR/Rolling calculation, plus pivot lookbacks.
        # Add a buffer for stability and potential indicator warm-up periods.
        required_for_vt = max(self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length)  # Need more for EMAs to stabilize
        required_for_pivots = max(self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1)
        self.min_data_len = max(required_for_vt, required_for_pivots) + 50  # Add a generous buffer (e.g., 50 candles)

        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy Engine...{RESET}")
        self.logger.info(f"  VT Params: Length={self.vt_length}, ATR Period={self.vt_atr_period}, Vol EMA Length={self.vt_vol_ema_length}, ATR Multiplier={self.vt_atr_multiplier.normalize()}")
        self.logger.info(f"  OB Params: Source={self.ob_source}, PH Lookback={self.ph_left}/{self.ph_right}, PL Lookback={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, Max Boxes={self.ob_max_boxes}")
        self.logger.info(f"  Calculated Minimum Historical Data Needed: {self.min_data_len} candles")

        # --- Crucial Check: Compare required data with API limit ---
        if self.min_data_len > BYBIT_API_KLINE_LIMIT:
             self.logger.error(f"{NEON_RED}{BRIGHT}CONFIGURATION ERROR:{RESET} Strategy requires {self.min_data_len} candles, "
                               f"but the API limit per fetch is {BYBIT_API_KLINE_LIMIT}.")
             self.logger.error(f"{NEON_YELLOW} >> ACTION REQUIRED: Reduce lookback periods in 'config.json' (e.g., 'vt_vol_ema_length', 'vt_atr_period') "
                               f"so the minimum data needed is less than {BYBIT_API_KLINE_LIMIT}.{RESET}")
             # Bot might still run but analysis will be based on incomplete data, leading to incorrect signals.

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates Smoothed Weighted Moving Average (SWMA) then EMA of the result."""
        # SWMA(4) calculation: Equivalent to LWMA with weights [1, 2, 2, 1]
        if len(series) < 4 or length <= 0:
            return pd.Series(np.nan, index=series.index, dtype=float)  # Return NaNs if not enough data

        # Ensure numeric type for calculations
        series_numeric = pd.to_numeric(series, errors='coerce')
        if series_numeric.isnull().all():  # Handle case where all values are NaN
             return pd.Series(np.nan, index=series.index, dtype=float)

        # Define SWMA weights
        weights = np.array([1., 2., 2., 1.]) / 6.0  # Normalized weights

        # Calculate SWMA using rolling apply with dot product
        # min_periods=4 ensures we have a full window before calculating
        swma = series_numeric.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)

        # Calculate EMA of the SWMA result
        return ta.ema(swma, length=length, fillna=np.nan)  # Use pandas_ta for EMA

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """Processes historical data to calculate indicators and manage Order Blocks."""
        # Define empty result structure for early exits
        empty_results = StrategyAnalysisResults(
            dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None,
            trend_just_changed=False, active_bull_boxes=[], active_bear_boxes=[],
            vol_norm_int=None, atr=None, upper_band=None, lower_band=None
        )

        if df_input.empty:
            self.logger.error("Strategy update received an empty DataFrame.")
            return empty_results
        if not isinstance(df_input.index, pd.DatetimeIndex) or not df_input.index.is_monotonic_increasing:
            self.logger.error("DataFrame index is not a monotonic DatetimeIndex.")
            return empty_results

        df = df_input.copy()  # Work on a copy to avoid modifying the original

        # Check if sufficient data length is available AFTER receiving data
        if len(df) < self.min_data_len:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data length provided ({len(df)} candles) for full analysis "
                                f"(requires {self.min_data_len}). Results may be inaccurate.{RESET}")
            # Proceed with calculation, but be aware results might be unreliable

        self.logger.debug(f"Starting strategy analysis on {len(df)} candles (min required: {self.min_data_len}).")

        # --- Indicator Calculation ---
        try:
            # Convert necessary columns to float for TA-Lib/Pandas-TA compatibility
            df_float = pd.DataFrame(index=df.index)
            cols_to_convert = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in cols_to_convert if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Input DataFrame is missing required columns: {', '.join(missing_cols)}. Aborting analysis.")
                return empty_results

            for col in cols_to_convert:
                df_float[col] = pd.to_numeric(df[col], errors='coerce')

            # Drop rows if essential float columns became NaN (shouldn't happen with prior cleaning)
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_float.empty:
                self.logger.error("DataFrame became empty after converting essential columns to float.")
                return empty_results

            # 1. Volumatic Trend Calculations
            # ATR for bands and potentially step line
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)

            # Core EMAs for trend direction
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)  # EMA(SWMA(close, 4), length)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)  # Standard EMA(close, length)

            # Determine Trend Direction (ema1 crossing ema2)
            # Compare current ema2 with *previous* ema1 to detect cross on the current bar
            df_float['trend_up'] = (df_float['ema2'] > df_float['ema1'].shift(1)).ffill()  # Forward fill initial NaNs
            df_float['trend_up'].fillna(False, inplace=True)  # Assume down trend initially if still NaN

            # Detect Trend Change
            df_float['trend_changed'] = (df_float['trend_up'].shift(1) != df_float['trend_up']) & \
                                        df_float['trend_up'].notna() & \
                                        df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'].fillna(False, inplace=True)  # No change on first valid row

            # Capture EMA1 and ATR values *at the time of the trend change* for band calculation
            # This makes the bands "stateful" - they only update their level on a trend change.
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)

            # Forward fill these captured values to keep the bands constant between trend changes
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill()
            df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()

            # Calculate Upper and Lower Bands
            atr_mult_float = float(self.vt_atr_multiplier)  # Convert Decimal multiplier to float for calculation
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult_float)

            # 2. Volume Normalization (Optional but used in some Volumatic versions)
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            # Rolling max volume over the specified period
            vol_max_period = max(1, self.vt_vol_ema_length // 10)  # Use a reasonable min period for rolling max
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=vol_max_period).max().fillna(0.0)
            # Normalize volume: (Current Volume / Max Volume over Period) * 100
            # Avoid division by zero if vol_max is 0
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9, (volume_numeric / df_float['vol_max'] * 100.0), 0.0)
            # Fill NaNs and clip (optional, prevents extreme values)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)  # Clip at 200% as an example

            # 3. Pivot High/Low for Order Blocks
            # Determine source series for pivots
            if self.ob_source.lower() == "wicks":
                high_series = df_float['high']
                low_series = df_float['low']
            else:  # "body"
                high_series = df_float[['open', 'close']].max(axis=1)
                low_series = df_float[['open', 'close']].min(axis=1)

            # Use pandas_ta.pivot for detection
            # Note: ta.pivot returns 1 at the *confirmation* bar (N bars after the pivot)
            ph_signals = ta.pivot(high_series, left=self.ph_left, right=self.ph_right, high_low='high').fillna(0).astype(bool)
            pl_signals = ta.pivot(low_series, left=self.pl_left, right=self.pl_right, high_low='low').fillna(0).astype(bool)

            # Add pivot signals back to the main Decimal DataFrame
            df['ph_signal'] = ph_signals.reindex(df.index, fill_value=False)
            df['pl_signal'] = pl_signals.reindex(df.index, fill_value=False)

        except Exception as e:
            self.logger.error(f"Error during indicator calculation: {e}", exc_info=True)
            return empty_results

        # --- Copy calculated float results back to the main Decimal DataFrame ---
        try:
            cols_to_copy = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed', 'upper_band', 'lower_band', 'vol_norm']
            for col in cols_to_copy:
                if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)  # Align index first
                    if source_series.dtype == 'bool' or pd.api.types.is_bool_dtype(source_series):
                        df[col] = source_series.astype(bool)  # Copy booleans directly
                    elif pd.api.types.is_object_dtype(source_series):  # Handle potential non-numeric objects if any
                         df[col] = source_series
                    else:
                        # Convert finite floats back to Decimal, keep NaNs as Decimal('NaN')
                        df[col] = source_series.apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN')
                        )
        except Exception as e:
            self.logger.error(f"Error converting calculated indicators back to Decimal: {e}", exc_info=True)
            # Continue, but results might be missing some indicators

        # --- Clean Final DataFrame ---
        initial_len = len(df)
        # Drop rows where essential calculated indicators might be NaN (e.g., during warm-up)
        required_cols = ['close', 'atr', 'trend_up', 'upper_band', 'lower_band']  # Ensure these are present
        df.dropna(subset=required_cols, inplace=True)
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             self.logger.debug(f"Dropped {rows_dropped} rows missing essential indicator values after calculation.")
        if df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty after indicator calculations & dropna. Cannot process Order Blocks.{RESET}")
            # Return results with what we have (likely empty/None values)
            last = None
        else:
            last = df.iloc[-1]  # Get last row for final results

        self.logger.debug("Indicator calculations complete. Processing Order Blocks...")

        # --- Order Block Management ---
        try:
            new_boxes_count = 0
            if not df.empty:
                 # Iterate through candles where a pivot was *confirmed*
                 pivot_confirm_indices = df.index[df['ph_signal'] | df['pl_signal']]
                 for conf_idx in pivot_confirm_indices:
                     try:
                         ph_conf = df.loc[conf_idx, 'ph_signal']
                         pl_conf = df.loc[conf_idx, 'pl_signal']
                         conf_loc = df.index.get_loc(conf_idx)  # Location index in the current DF

                         # --- Bearish OB Detection ---
                         if ph_conf:
                             # Pivot High occurred 'ph_right' bars *before* the confirmation bar
                             piv_loc = conf_loc - self.ph_right
                             if piv_loc >= 0:  # Ensure pivot index is within the DF bounds
                                 piv_idx = df.index[piv_loc]  # Timestamp of the actual pivot high candle
                                 # Check if this pivot already generated a bear box
                                 if not any(b['left_idx'] == piv_idx and b['type'] == 'bear' for b in self.bear_boxes):
                                     pivot_candle = df.loc[piv_idx]
                                     # Define OB bounds based on source
                                     top, bot = (pivot_candle['high'], pivot_candle['open']) if self.ob_source.lower() == "wicks" else \
                                                (max(pivot_candle['open'], pivot_candle['close']), min(pivot_candle['open'], pivot_candle['close']))
                                     # Ensure valid Decimal numbers and top > bottom
                                     if pd.notna(top) and pd.notna(bot) and isinstance(top, Decimal) and isinstance(bot, Decimal) and top > bot:
                                         new_box = OrderBlock(
                                             id=f"B_{piv_idx.strftime('%y%m%d%H%M%S')}",  # More unique ID
                                             type='bear', left_idx=piv_idx, right_idx=df.index[-1],  # Initially extends to end
                                             top=top, bottom=bot, active=True, violated=False
                                         )
                                         self.bear_boxes.append(new_box)
                                         new_boxes_count += 1
                                         self.logger.debug(f"  New Bear OB: {new_box['id']} @ {piv_idx} [{new_box['bottom'].normalize()}-{new_box['top'].normalize()}]")
                                     else: self.logger.debug(f"Skipping Bear OB @ {piv_idx}: Invalid candle data (Top={top}, Bot={bot}).")

                         # --- Bullish OB Detection ---
                         if pl_conf:
                             # Pivot Low occurred 'pl_right' bars *before* the confirmation bar
                             piv_loc = conf_loc - self.pl_right
                             if piv_loc >= 0:
                                 piv_idx = df.index[piv_loc]
                                 if not any(b['left_idx'] == piv_idx and b['type'] == 'bull' for b in self.bull_boxes):
                                     pivot_candle = df.loc[piv_idx]
                                     top, bot = (pivot_candle['open'], pivot_candle['low']) if self.ob_source.lower() == "wicks" else \
                                                (max(pivot_candle['open'], pivot_candle['close']), min(pivot_candle['open'], pivot_candle['close']))
                                     if pd.notna(top) and pd.notna(bot) and isinstance(top, Decimal) and isinstance(bot, Decimal) and top > bot:
                                          new_box = OrderBlock(
                                              id=f"L_{piv_idx.strftime('%y%m%d%H%M%S')}", type='bull', left_idx=piv_idx,
                                              right_idx=df.index[-1], top=top, bottom=bot, active=True, violated=False
                                          )
                                          self.bull_boxes.append(new_box)
                                          new_boxes_count += 1
                                          self.logger.debug(f"  New Bull OB: {new_box['id']} @ {piv_idx} [{new_box['bottom'].normalize()}-{new_box['top'].normalize()}]")
                                     else: self.logger.debug(f"Skipping Bull OB @ {piv_idx}: Invalid candle data (Top={top}, Bot={bot}).")
                     except Exception as e:
                         # Log error processing a specific pivot but continue with others
                         self.logger.warning(f"Error processing pivot signal at index {conf_idx}: {e}", exc_info=True)

            if new_boxes_count > 0:
                 self.logger.debug(f"Found {new_boxes_count} new Order Block(s). Total counts before prune: Bull={len(self.bull_boxes)}, Bear={len(self.bear_boxes)}.")

            # --- Manage Existing Order Blocks (Violation Check & Extension) ---
            if last is not None and pd.notna(last.get('close')) and isinstance(last['close'], Decimal):
                last_idx = last.name  # Timestamp index of the last candle
                last_close = last['close']

                for box in self.bull_boxes:
                    if box['active']:
                        # Violation Check: Close below the bottom of the Bull OB
                        if last_close < box['bottom']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx  # Mark violation time
                            self.logger.debug(f"Bull OB {box['id']} VIOLATED by close {last_close.normalize()} < {box['bottom'].normalize()} at {last_idx}.")
                        # Extend active box to the current candle if extension is enabled
                        elif self.ob_extend:
                            box['right_idx'] = last_idx

                for box in self.bear_boxes:
                    if box['active']:
                        # Violation Check: Close above the top of the Bear OB
                        if last_close > box['top']:
                            box['active'] = False
                            box['violated'] = True
                            box['right_idx'] = last_idx
                            self.logger.debug(f"Bear OB {box['id']} VIOLATED by close {last_close.normalize()} > {box['top'].normalize()} at {last_idx}.")
                        elif self.ob_extend:
                            box['right_idx'] = last_idx
            else:
                self.logger.warning("Cannot check OB violations: Last close price is invalid or missing.")

            # --- Prune Order Blocks ---
            # Sort by pivot time (descending) and keep only the most recent N boxes
            self.bull_boxes = sorted([b for b in self.bull_boxes if not b['violated']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted([b for b in self.bear_boxes if not b['violated']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            # Also prune violated boxes if list gets too long? For now, just keep non-violated ones up to max.

            active_bull_count = len(self.bull_boxes)  # Already filtered for active=True essentially
            active_bear_count = len(self.bear_boxes)
            self.logger.debug(f"Pruned inactive/old OBs. Kept Active: Bull={active_bull_count}, Bear={active_bear_count} (Max per type: {self.ob_max_boxes}).")

        except Exception as e:
            self.logger.error(f"Error during Order Block processing: {e}", exc_info=True)
            # Continue to return results based on indicators even if OBs fail

        # --- Prepare Final Results ---
        # Helper to safely get Decimal or None
        def sanitize_dec(value, must_be_positive=False) -> Decimal | None:
             if pd.notna(value) and isinstance(value, Decimal) and np.isfinite(float(value)):
                 if not must_be_positive or value > 0:
                     return value
             return None

        last_close_dec = sanitize_dec(last.get('close')) if last is not None else Decimal('0')
        current_trend = None
        if last is not None and isinstance(last.get('trend_up'), (bool, np.bool_)):
             current_trend = bool(last['trend_up'])
        trend_changed = False
        if last is not None and isinstance(last.get('trend_changed'), (bool, np.bool_)):
             trend_changed = bool(last['trend_changed'])
        last_atr = sanitize_dec(last.get('atr'), must_be_positive=True) if last is not None else None
        last_vol_norm = sanitize_dec(last.get('vol_norm')) if last is not None else None
        vol_norm_int = int(last_vol_norm) if last_vol_norm is not None else None
        last_upper_band = sanitize_dec(last.get('upper_band')) if last is not None else None
        last_lower_band = sanitize_dec(last.get('lower_band')) if last is not None else None

        results = StrategyAnalysisResults(
            dataframe=df,  # Return the DataFrame with all calculations
            last_close=last_close_dec,
            current_trend_up=current_trend,
            trend_just_changed=trend_changed,
            active_bull_boxes=[b for b in self.bull_boxes if b['active']],  # Should be all remaining after prune
            active_bear_boxes=[b for b in self.bear_boxes if b['active']],  # Should be all remaining after prune
            vol_norm_int=vol_norm_int,
            atr=last_atr,
            upper_band=last_upper_band,
            lower_band=last_lower_band
        )

        # Log summary of the final state
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else \
                    f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else \
                    f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr'].normalize()}" if results['atr'] else "N/A"
        time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
        self.logger.debug(f"Strategy Analysis Results ({time_str}):")
        self.logger.debug(f"  Last Close: {results['last_close'].normalize()}")
        self.logger.debug(f"  Trend: {trend_str} (Changed: {results['trend_just_changed']})")
        self.logger.debug(f"  ATR: {atr_str}")
        self.logger.debug(f"  VolNorm: {results['vol_norm_int']}")
        self.logger.debug(f"  Bands: Lower={results['lower_band'].normalize() if results['lower_band'] else 'N/A'}, Upper={results['upper_band'].normalize() if results['upper_band'] else 'N/A'}")
        self.logger.debug(f"  Active OBs: Bull={len(results['active_bull_boxes'])}, Bear={len(results['active_bear_boxes'])}")

        return results


# --- Signal Generation based on Strategy Results ---
class SignalGenerator:
    """Generates trading signals (BUY, SELL, EXIT_LONG, EXIT_SHORT, HOLD) based on strategy analysis and position state."""
    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        protection_cfg = config.get("protection", {})

        # Load parameters used for signal generation
        try:
            # Proximity factor for entering near an OB
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"]))
            if not self.ob_entry_proximity_factor >= 1: raise ValueError("ob_entry_proximity_factor must be >= 1")
            # Proximity factor for exiting if price hits opposing OB
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"]))
            if not self.ob_exit_proximity_factor >= 1: raise ValueError("ob_exit_proximity_factor must be >= 1")
            # ATR multiples for initial TP/SL calculation
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"]))
            if not self.initial_tp_atr_multiple >= 0: raise ValueError("initial_take_profit_atr_multiple must be >= 0")
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"]))
            if not self.initial_sl_atr_multiple > 0: raise ValueError("initial_stop_loss_atr_multiple must be > 0")

        except (KeyError, ValueError, InvalidOperation, TypeError) as e:
             self.logger.error(f"{NEON_RED}Error initializing SignalGenerator with config values: {e}. Using hardcoded defaults.{RESET}", exc_info=True)
             # Fallback to safe defaults
             self.ob_entry_proximity_factor = Decimal("1.005")  # 0.5% proximity
             self.ob_exit_proximity_factor = Decimal("1.001")  # 0.1% proximity
             self.initial_tp_atr_multiple = Decimal("0.7")
             self.initial_sl_atr_multiple = Decimal("1.8")

        self.logger.info("Signal Generator Initialized:")
        self.logger.info(f"  OB Entry Proximity Factor: {self.ob_entry_proximity_factor.normalize()}")
        self.logger.info(f"  OB Exit Proximity Factor: {self.ob_exit_proximity_factor.normalize()}")
        self.logger.info(f"  Initial TP ATR Multiple: {self.initial_tp_atr_multiple.normalize()}")
        self.logger.info(f"  Initial SL ATR Multiple: {self.initial_sl_atr_multiple.normalize()}")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: dict | None) -> str:
        """Determines the trading signal based on strategy state and current position."""
        lg = self.logger

        # --- Validate Inputs ---
        if not isinstance(analysis_results, dict) or \
           analysis_results.get('dataframe') is None or analysis_results['dataframe'].empty or \
           analysis_results.get('current_trend_up') is None or \
           analysis_results.get('last_close') is None or analysis_results['last_close'] <= 0 or \
           analysis_results.get('atr') is None or analysis_results['atr'] <= 0:
            lg.warning(f"{NEON_YELLOW}Invalid or incomplete strategy analysis results provided to signal generator. Holding.{RESET}")
            lg.debug(f"Received analysis results: {analysis_results}")
            return "HOLD"

        # Extract key data for easier access
        close = analysis_results['last_close']
        trend_up = analysis_results['current_trend_up']
        trend_changed = analysis_results['trend_just_changed']
        bull_obs = analysis_results['active_bull_boxes']
        bear_obs = analysis_results['active_bear_boxes']
        pos_side = open_position.get('side') if open_position else None  # 'long', 'short', or None

        signal = "HOLD"  # Default signal

        lg.debug("Signal Generation Check:")
        lg.debug(f"  Close: {close.normalize()}, TrendUp: {trend_up}, TrendChanged: {trend_changed}")
        lg.debug(f"  Position: {pos_side or 'None'}, Active Bull OBs: {len(bull_obs)}, Active Bear OBs: {len(bear_obs)}")

        # --- 1. Exit Signal Checks (Only if a position exists) ---
        if pos_side == 'long':
            # Exit Condition 1: Trend flips to DOWN
            if trend_up is False and trend_changed:
                signal = "EXIT_LONG"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped to DOWN.{RESET}")
            # Exit Condition 2: Price hits a Bearish OB (using exit proximity)
            elif signal == "HOLD" and bear_obs:
                try:
                    # Find the closest Bear OB (based on top edge)
                    closest_bear_ob = min(bear_obs, key=lambda b: abs(b['top'] - close))
                    # Exit if close price >= OB top * proximity factor
                    exit_threshold = closest_bear_ob['top'] * self.ob_exit_proximity_factor
                    if close >= exit_threshold:
                        signal = "EXIT_LONG"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price {close.normalize()} >= Bear OB Exit Threshold {exit_threshold.normalize()} "
                                   f"(OB ID: {closest_bear_ob['id']}, Top: {closest_bear_ob['top'].normalize()}){RESET}")
                except Exception as e:
                    lg.warning(f"Error during Bear OB exit check for long position: {e}")  # Log error but don't crash

        elif pos_side == 'short':
            # Exit Condition 1: Trend flips to UP
            if trend_up is True and trend_changed:
                signal = "EXIT_SHORT"
                lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped to UP.{RESET}")
            # Exit Condition 2: Price hits a Bullish OB (using exit proximity)
            elif signal == "HOLD" and bull_obs:
                 try:
                    # Find the closest Bull OB (based on bottom edge)
                    closest_bull_ob = min(bull_obs, key=lambda b: abs(b['bottom'] - close))
                    # Exit if close price <= OB bottom / proximity factor
                    # (Divide because factor >= 1, need smaller threshold)
                    exit_threshold = closest_bull_ob['bottom'] / self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else closest_bull_ob['bottom']
                    if close <= exit_threshold:
                        signal = "EXIT_SHORT"
                        lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price {close.normalize()} <= Bull OB Exit Threshold {exit_threshold.normalize()} "
                                   f"(OB ID: {closest_bull_ob['id']}, Bottom: {closest_bull_ob['bottom'].normalize()}){RESET}")
                 except Exception as e:
                    lg.warning(f"Error during Bull OB exit check for short position: {e}")

        # If an exit signal was generated, return it immediately
        if signal != "HOLD":
            return signal

        # --- 2. Entry Signal Checks (Only if NO position exists) ---
        if pos_side is None:
            # Entry Condition 1: Trend is UP and price is inside/near an active Bullish OB
            if trend_up is True and bull_obs:
                for ob in bull_obs:
                    # Define entry zone: From OB bottom up to OB top * proximity factor
                    entry_zone_bottom = ob['bottom']
                    entry_zone_top = ob['top'] * self.ob_entry_proximity_factor
                    if entry_zone_bottom <= close <= entry_zone_top:
                        signal = "BUY"
                        lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal Triggered:{RESET}")
                        lg.info("  Trend is UP.")
                        lg.info(f"  Price {close.normalize()} is within Bull OB Entry Zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                        lg.info(f"  (OB ID: {ob['id']}, Base Range: [{ob['bottom'].normalize()}-{ob['top'].normalize()}])")
                        break  # Take the first matching OB

            # Entry Condition 2: Trend is DOWN and price is inside/near an active Bearish OB
            elif trend_up is False and bear_obs:
                 for ob in bear_obs:
                     # Define entry zone: From OB bottom / proximity factor up to OB top
                     entry_zone_bottom = ob['bottom'] / self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']
                     entry_zone_top = ob['top']
                     if entry_zone_bottom <= close <= entry_zone_top:
                         signal = "SELL"
                         lg.info(f"{NEON_RED}{BRIGHT}SELL Signal Triggered:{RESET}")
                         lg.info("  Trend is DOWN.")
                         lg.info(f"  Price {close.normalize()} is within Bear OB Entry Zone [{entry_zone_bottom.normalize()} - {entry_zone_top.normalize()}]")
                         lg.info(f"  (OB ID: {ob['id']}, Base Range: [{ob['bottom'].normalize()}-{ob['top'].normalize()}])")
                         break  # Take the first matching OB

        # --- 3. Hold Signal ---
        if signal == "HOLD":
            lg.debug("Signal: HOLD - No valid entry or exit conditions met.")

        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: dict, exchange: ccxt.Exchange) -> tuple[Decimal | None, Decimal | None]:
        """Calculates initial Take Profit and Stop Loss levels based on ATR and market precision."""
        lg = self.logger

        # --- Input Validation ---
        if signal not in ["BUY", "SELL"]:
            lg.error(f"Invalid signal '{signal}' for TP/SL calculation.")
            return None, None
        if entry_price <= 0:
            lg.error(f"Invalid entry price ({entry_price}) for TP/SL calculation.")
            return None, None
        if atr <= 0:
            lg.error(f"Invalid ATR value ({atr}) for TP/SL calculation.")
            return None, None
        price_prec_str = market_info.get('precision', {}).get('price')
        if price_prec_str is None:
             lg.error(f"Missing price precision in market info for symbol {market_info['symbol']}. Cannot calculate TP/SL.")
             return None, None

        try:
            min_tick = Decimal(str(price_prec_str))
            if min_tick <= 0: raise ValueError("Invalid price precision step")

            # Get ATR multiples from instance variables (already validated in __init__)
            tp_mult = self.initial_tp_atr_multiple
            sl_mult = self.initial_sl_atr_multiple  # Guaranteed > 0

            # Calculate raw offsets
            tp_offset = atr * tp_mult
            sl_offset = atr * sl_mult

            # Calculate raw TP/SL levels
            tp_raw = None
            if tp_mult > 0:  # Only calculate TP if multiple is positive
                tp_raw = (entry_price + tp_offset) if signal == "BUY" else (entry_price - tp_offset)

            sl_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)

            lg.debug(f"Raw TP/SL Calculation (Signal: {signal}, Entry: {entry_price.normalize()}, ATR: {atr.normalize()}):")
            lg.debug(f"  SL Offset = {atr.normalize()} * {sl_mult.normalize()} = {sl_offset.normalize()}")
            lg.debug(f"  Raw SL = {sl_raw.normalize()}")
            if tp_raw:
                lg.debug(f"  TP Offset = {atr.normalize()} * {tp_mult.normalize()} = {tp_offset.normalize()}")
                lg.debug(f"  Raw TP = {tp_raw.normalize()}")
            else:
                 lg.debug("  TP Multiple is 0, skipping TP calculation.")

            # --- Format Levels to Market Precision ---
            symbol = market_info['symbol']

            def format_level(price_dec: Decimal | None, level_name: str) -> Decimal | None:
                """Formats price using exchange precision, returns Decimal or None."""
                if price_dec is None: return None
                if price_dec <= 0:
                    lg.warning(f"Calculated raw {level_name} level is zero or negative ({price_dec}). Cannot format.")
                    return None
                try:
                    # Use ccxt.price_to_precision for correct formatting
                    formatted_str = exchange.price_to_precision(symbol=symbol, price=float(price_dec))
                    formatted_dec = Decimal(formatted_str)
                    if formatted_dec > 0:
                        lg.debug(f"Formatted {level_name}: {price_dec.normalize()} -> {formatted_dec.normalize()}")
                        return formatted_dec
                    else:
                        lg.warning(f"Formatted {level_name} level is zero or negative ({formatted_dec}).")
                        return None
                except Exception as e:
                    lg.error(f"Error formatting {level_name} level {price_dec} for symbol {symbol}: {e}.")
                    return None

            tp_formatted = format_level(tp_raw, "Take Profit")
            sl_formatted = format_level(sl_raw, "Stop Loss")

            # --- Final Validation and Adjustment ---
            # Ensure SL is strictly beyond entry price
            if sl_formatted is not None:
                if (signal == "BUY" and sl_formatted >= entry_price) or \
                   (signal == "SELL" and sl_formatted <= entry_price):
                    lg.warning(f"Calculated SL {sl_formatted.normalize()} is not strictly beyond entry price {entry_price.normalize()}. Adjusting by one tick.")
                    # Adjust SL by one tick away from entry
                    sl_adjusted = (entry_price - min_tick) if signal == "BUY" else (entry_price + min_tick)
                    sl_formatted = format_level(sl_adjusted, "Adjusted SL")  # Re-format adjusted SL

            # Ensure TP is strictly beyond entry price (if calculated)
            if tp_formatted is not None:
                 if (signal == "BUY" and tp_formatted <= entry_price) or \
                    (signal == "SELL" and tp_formatted >= entry_price):
                     lg.warning(f"Calculated TP {tp_formatted.normalize()} is not strictly beyond entry price {entry_price.normalize()}. Disabling initial TP.")
                     tp_formatted = None  # Set TP to None if it's invalid

            # Final check: SL must be valid for trade to proceed
            if sl_formatted is None:
                 lg.error(f"{NEON_RED}Stop Loss calculation failed or resulted in an invalid level after formatting/adjustment. Cannot proceed with trade.{RESET}")
                 return tp_formatted, None  # Return TP (if any) but None for SL

            lg.info("Calculated Initial Protection Levels:")
            lg.info(f"  Stop Loss: {sl_formatted.normalize() if sl_formatted else 'FAILED'}")
            lg.info(f"  Take Profit: {tp_formatted.normalize() if tp_formatted else 'None (Disabled or Failed Calc)'}")

            return tp_formatted, sl_formatted

        except Exception as e:
            lg.error(f"{NEON_RED}Unexpected error calculating initial TP/SL: {e}{RESET}", exc_info=True)
            return None, None


# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: dict) -> None:
    """Performs one cycle of data fetching, analysis, signal generation, and trading/position management for a symbol."""
    lg = logger
    lg.info(f"\n{BRIGHT}---== Analyzing {symbol} ({config['interval']} TF) Cycle Start ==---{RESET}")
    cycle_start_time = time.monotonic()
    ccxt_interval = CCXT_INTERVAL_MAP[config["interval"]]  # Map '5' -> '5m' etc.

    # --- 1. Determine Kline Fetch Limit ---
    min_req_data = strategy_engine.min_data_len  # Minimum candles needed by strategy
    fetch_limit_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)  # User preference from config

    # We need *at least* min_req_data. Request this amount, or the user's preference if it's higher,
    # but cap it at the absolute API limit.
    fetch_limit_needed = max(min_req_data, fetch_limit_config)
    fetch_limit_request = min(fetch_limit_needed, BYBIT_API_KLINE_LIMIT)

    lg.debug(f"Strategy requires minimum {min_req_data} candles.")
    lg.debug(f"Config fetch_limit preference: {fetch_limit_config}.")
    lg.debug(f"Effective fetch limit needed: {fetch_limit_needed}.")
    lg.info(f"Requesting {fetch_limit_request} klines (API limit: {BYBIT_API_KLINE_LIMIT})...")

    # --- 2. Fetch Kline Data ---
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit_request, logger=lg)
    fetched_count = len(klines_df)

    # --- 3. Validate Fetched Data ---
    if klines_df.empty or fetched_count < min_req_data:
        # Check if the failure was due to hitting the API limit but still not getting enough data
        api_limit_hit_but_insufficient = (
            fetch_limit_request == BYBIT_API_KLINE_LIMIT and
            fetched_count == BYBIT_API_KLINE_LIMIT and  # Check if we actually received the limit
            fetched_count < min_req_data  # Verify it's still less than required
        )

        if api_limit_hit_but_insufficient:
             lg.error(f"{NEON_RED}CRITICAL DATA SHORTFALL:{RESET} Fetched {fetched_count} candles (hit API limit {BYBIT_API_KLINE_LIMIT}), "
                      f"but strategy requires {min_req_data}.")
             lg.error(f"{NEON_YELLOW} >> ACTION REQUIRED: Reduce strategy lookback periods (e.g., 'vt_vol_ema_length') in config.json "
                      f"so the minimum data needed is less than {BYBIT_API_KLINE_LIMIT}. Skipping cycle.{RESET}")
        elif klines_df.empty:
             lg.error(f"Failed to fetch any kline data for {symbol}. Check connection or symbol validity. Skipping cycle.")
        else:  # Got some data, but less than required (and didn't hit API limit)
             lg.error(f"Failed to fetch sufficient kline data ({fetched_count}/{min_req_data}). "
                      f"This might occur with new symbols or network issues. Skipping cycle.")
        return  # Cannot proceed without sufficient data

    # --- 4. Run Strategy Analysis ---
    try:
        analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err:
        lg.error(f"{NEON_RED}Strategy analysis failed with error: {analysis_err}{RESET}", exc_info=True)
        return  # Stop cycle if analysis crashes

    # Validate essential results from analysis
    if not analysis_results or \
       analysis_results.get('current_trend_up') is None or \
       analysis_results.get('last_close') is None or analysis_results['last_close'] <= 0 or \
       analysis_results.get('atr') is None or analysis_results['atr'] <= 0:
        lg.error(f"{NEON_RED}Strategy analysis completed but produced invalid/incomplete results. Skipping cycle.{RESET}")
        lg.debug(f"Analysis Results: {analysis_results}")
        return

    latest_close = analysis_results['last_close']
    current_atr = analysis_results['atr']  # Needed for TP/SL, BE checks

    # --- 5. Get Current Market State ---
    # Fetch current price (more real-time than last close)
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    # Use current price if available, otherwise fall back to last close for checks
    price_for_checks = current_price if current_price and current_price > 0 else latest_close
    if price_for_checks <= 0:
        lg.error(f"{NEON_RED}Cannot determine a valid current price ({current_price}) or last close ({latest_close}). Skipping cycle.{RESET}")
        return
    if current_price is None:
        lg.warning(f"{NEON_YELLOW}Failed to fetch real-time price. Using last kline close ({latest_close.normalize()}) for position management checks.{RESET}")

    # Check for existing open position
    open_position = get_open_position(exchange, symbol, lg)  # Returns dict if position exists, else None

    # --- 6. Generate Trading Signal ---
    try:
        signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err:
        lg.error(f"{NEON_RED}Signal generation failed with error: {signal_err}{RESET}", exc_info=True)
        return  # Stop cycle if signal generation crashes

    # --- 7. Trading Logic ---
    trading_enabled = config.get("enable_trading", False)
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading is disabled.{RESET} Generated Signal: {signal}. Analysis complete.")
        # Log potential actions if trading were enabled
        if open_position is None and signal in ["BUY", "SELL"] or open_position and signal in ["EXIT_LONG", "EXIT_SHORT"]: lg.info(f"  (Would attempt to {signal} if trading was enabled)")
        else: lg.info("  (No entry/exit action indicated)")
        # End cycle here if trading disabled
        cycle_end_time = time.monotonic()
        lg.debug(f"---== Analysis-Only Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")
        return

    # --- Trading Enabled ---
    lg.debug(f"Trading is enabled. Signal: {signal}. Current Position: {'Yes (' + open_position['side'] + ')' if open_position else 'No'}")

    # === Scenario 1: No Open Position ===
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"{BRIGHT}*** {signal} Signal received and NO current position. Initiating Entry Sequence... ***{RESET}")

            # Fetch current balance
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= Decimal('0'):
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Cannot fetch valid balance ({balance}).{RESET}")
                return

            # Calculate initial TP/SL based on latest close and ATR
            # Use latest_close for initial calculation as entry price isn't known yet
            tp_initial_calc, sl_initial_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)
            if sl_initial_calc is None:
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Initial Stop Loss calculation failed.{RESET}")
                return
            if tp_initial_calc is None:
                lg.warning("Initial Take Profit calculation failed or TP is disabled (multiple=0).")

            # Set leverage if required (for contracts)
            leverage_ok = True
            if market_info['is_contract']:
                leverage = int(config['leverage'])
                if leverage > 0:
                    leverage_ok = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg)
                # If leverage is 0 in config, assume user doesn't want bot to set it (use exchange default/manual setting)
            if not leverage_ok:
                 lg.error(f"{NEON_RED}Trade Aborted ({signal}): Failed to set leverage.{RESET}")
                 return

            # Calculate position size based on risk and calculated SL
            pos_size = calculate_position_size(balance, config["risk_per_trade"], sl_initial_calc, latest_close, market_info, exchange, lg)
            if pos_size is None or pos_size <= Decimal('0'):
                lg.error(f"{NEON_RED}Trade Aborted ({signal}): Position size calculation failed or resulted in zero/negative size ({pos_size}).{RESET}")
                return

            # Place the Market Order to Enter Position
            lg.info(f"===> Placing {signal} Market Order | Size: {pos_size.normalize()} <===")
            trade_order = place_trade(exchange, symbol, signal, pos_size, market_info, lg, reduce_only=False)

            # Post-Trade Actions (Confirmation & Protection Setting)
            if trade_order and trade_order.get('id'):
                confirm_delay = config.get("position_confirm_delay_seconds", POSITION_CONFIRM_DELAY_SECONDS)
                lg.info(f"Order placed (ID: {trade_order['id']}). Waiting {confirm_delay}s for position confirmation...")
                time.sleep(confirm_delay)

                # Confirm the position was actually opened
                confirmed_pos = get_open_position(exchange, symbol, lg)
                if confirmed_pos:
                    try:
                        entry_actual_str = confirmed_pos.get('entryPrice')
                        entry_actual = Decimal(str(entry_actual_str)) if entry_actual_str else latest_close  # Fallback if avg price not available
                        if entry_actual <= 0: entry_actual = latest_close  # Ensure positive price
                        lg.info(f"{NEON_GREEN}Position Confirmed! Actual Avg Entry Price: ~{entry_actual.normalize()}{RESET}")

                        # --- Set Protection (SL/TP/TSL) using actual entry price ---
                        prot_cfg = config["protection"]
                        # Recalculate TP/SL based on actual entry price
                        tp_prot_calc, sl_prot_calc = signal_generator.calculate_initial_tp_sl(entry_actual, signal, current_atr, market_info, exchange)
                        if sl_prot_calc is None:
                             lg.error(f"{NEON_RED}CRITICAL: Failed to recalculate Stop Loss based on actual entry price! Position is unprotected!{RESET}")
                             # Decide how to handle: exit immediately? Let it run? Log critical error.
                             # For now, just log critical error.
                        else:
                            lg.info("Recalculated protection levels based on actual entry:")
                            lg.info(f"  SL: {sl_prot_calc.normalize() if sl_prot_calc else 'N/A'}")
                            lg.info(f"  TP: {tp_prot_calc.normalize() if tp_prot_calc else 'N/A'}")

                            protection_set_successfully = False
                            # Option 1: Use Trailing Stop Loss if enabled
                            if prot_cfg.get("enable_trailing_stop", True):
                                 lg.info(f"Setting Initial Trailing Stop Loss (based on Entry: {entry_actual.normalize()})...")
                                 # Pass the recalculated TP to TSL function if needed
                                 protection_set_successfully = set_trailing_stop_loss(
                                     exchange, symbol, market_info, confirmed_pos, config, lg,
                                     take_profit_price=tp_prot_calc  # Pass TP to potentially set it alongside TSL
                                 )
                            # Option 2: Use Fixed SL/TP if TSL disabled but SL/TP multiples are set
                            elif not prot_cfg.get("enable_trailing_stop", True) and (sl_prot_calc or tp_prot_calc):
                                 lg.info(f"Setting Initial Fixed Stop Loss / Take Profit (based on Entry: {entry_actual.normalize()})...")
                                 protection_set_successfully = _set_position_protection(
                                     exchange, symbol, market_info, confirmed_pos, lg,
                                     stop_loss_price=sl_prot_calc,
                                     take_profit_price=tp_prot_calc
                                 )
                            # Option 3: No protection enabled
                            else:
                                 lg.info("Neither Trailing Stop nor Fixed SL/TP enabled in config. No protection set.")
                                 protection_set_successfully = True  # Considered success as no action was needed

                            if protection_set_successfully:
                                lg.info(f"{NEON_GREEN}{BRIGHT}=== TRADE ENTRY & INITIAL PROTECTION SETUP COMPLETE ({symbol} {signal}) ==={RESET}")
                            else:
                                lg.error(f"{NEON_RED}{BRIGHT}=== TRADE PLACED, BUT FAILED TO SET INITIAL PROTECTION ({symbol} {signal}). MANUAL MONITORING REQUIRED! ==={RESET}")

                    except Exception as post_trade_err:
                        lg.error(f"{NEON_RED}Error during post-trade setup (protection): {post_trade_err}{RESET}", exc_info=True)
                        lg.warning(f"{NEON_YELLOW}Position confirmed, but may lack protection! Manual check required!{RESET}")
                else:
                    # Order placed, but position not found after delay
                    lg.error(f"{NEON_RED}Order (ID: {trade_order['id']}) was placed, but FAILED TO CONFIRM open position after {confirm_delay}s! "
                             f"Check exchange manually. Order might have failed, been rejected, or filled unexpectedly.{RESET}")
            else:
                # Order placement failed entirely
                lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). No order placed. ===")
        else:
            # No position and signal is HOLD
            lg.info("Signal is HOLD, no existing position. No trading action taken.")

    # === Scenario 2: Existing Position ===
    else:  # Position exists
        pos_side = open_position['side']  # 'long' or 'short'
        try: pos_size = open_position['size_decimal']  # Use pre-parsed Decimal size
        except KeyError: lg.error("Position data missing 'size_decimal'. Cannot manage position."); return
        lg.info(f"Existing {pos_side.upper()} position found (Size: {pos_size.normalize()}). Signal: {signal}")

        # --- Check for Exit Signal ---
        exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or \
                         (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_triggered:
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal received! Initiating Position Close Sequence... ***{RESET}")
            try:
                close_signal = "SELL" if pos_side == 'long' else "BUY"
                size_to_close = abs(pos_size)  # Close the entire position size

                if size_to_close <= Decimal('0'):
                    lg.warning(f"Attempting to close position, but recorded size is zero or negative ({pos_size}). Position might already be closed or data is inconsistent.")
                    return  # Avoid placing zero size order

                lg.info(f"===> Placing {close_signal} MARKET Order (Reduce Only) | Size: {size_to_close.normalize()} <===")
                # Place closing order using reduce_only flag
                close_order = place_trade(exchange, symbol, close_signal, size_to_close, market_info, lg, reduce_only=True)

                if close_order and close_order.get('id'):
                    lg.info(f"{NEON_GREEN}Position CLOSE order (ID: {close_order['id']}) placed successfully.{RESET}")
                    # Consider adding a short delay and confirmation check here too if needed
                else:
                    lg.error(f"{NEON_RED}Failed to place CLOSE order for {symbol} position. Manual check required!{RESET}")

            except Exception as close_err:
                lg.error(f"{NEON_RED}Error occurred while trying to close {pos_side} position: {close_err}{RESET}", exc_info=True)
                lg.warning(f"{NEON_YELLOW}Manual intervention may be needed to close the position!{RESET}")

        # --- No Exit Signal: Perform Position Management (BE, TSL checks) ---
        else:
            lg.debug(f"Signal ({signal}) allows holding position. Performing position management checks...")
            prot_cfg = config["protection"]

            # Extract current protection status from position info
            tsl_dist_str = open_position.get('trailingStopLoss')
            tsl_active = False
            with contextlib.suppress(Exception): tsl_active = tsl_dist_str and Decimal(str(tsl_dist_str)) > 0

            sl_curr_str = open_position.get('stopLossPrice')
            sl_curr = None
            with contextlib.suppress(Exception): sl_curr = Decimal(str(sl_curr_str)) if sl_curr_str and str(sl_curr_str) != '0' else None

            tp_curr_str = open_position.get('takeProfitPrice')
            tp_curr = None
            with contextlib.suppress(Exception): tp_curr = Decimal(str(tp_curr_str)) if tp_curr_str and str(tp_curr_str) != '0' else None

            entry_price_str = open_position.get('entryPrice')
            entry_price = None
            with contextlib.suppress(Exception): entry_price = Decimal(str(entry_price_str)) if entry_price_str else None

            lg.debug(f"  Current Protections: TSL Active={tsl_active} (Dist='{tsl_dist_str}'), Fixed SL={sl_curr.normalize() if sl_curr else 'None'}, Fixed TP={tp_curr.normalize() if tp_curr else 'None'}")

            # --- Break-Even Logic ---
            be_enabled = prot_cfg.get("enable_break_even", True)
            # Check BE only if enabled, TSL is *not* active, and we have necessary data
            if be_enabled and not tsl_active and entry_price and current_atr and price_for_checks > 0:
                lg.debug(f"Checking Break-Even possibility (Entry:{entry_price.normalize()}, Price:{price_for_checks.normalize()}, ATR:{current_atr.normalize()})...")
                try:
                    be_trig_atr_mult = Decimal(str(prot_cfg["break_even_trigger_atr_multiple"]))
                    be_offset_ticks = int(prot_cfg["break_even_offset_ticks"])
                    if be_trig_atr_mult <= 0: raise ValueError("BE trigger multiple must be positive")
                    if be_offset_ticks < 0: raise ValueError("BE offset ticks cannot be negative")

                    # Calculate profit in terms of ATRs
                    profit_in_price = (price_for_checks - entry_price) if pos_side == 'long' else (entry_price - price_for_checks)
                    profit_in_atr = profit_in_price / current_atr if current_atr > 0 else Decimal('0')

                    lg.debug(f"  BE Check: Profit = {profit_in_price.normalize()}, Profit ATRs = {profit_in_atr:.3f}, Trigger = {be_trig_atr_mult.normalize()} ATRs")

                    # Check if profit target is reached
                    if profit_in_atr >= be_trig_atr_mult:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}Break-Even profit target REACHED! (Profit {profit_in_atr:.3f} >= {be_trig_atr_mult.normalize()} ATRs){RESET}")

                        # Calculate Break-Even SL price (Entry + Offset)
                        tick_size = Decimal(str(market_info['precision']['price']))
                        offset_value = tick_size * Decimal(str(be_offset_ticks))
                        # Quantize BE SL to nearest tick away from entry
                        if pos_side == 'long':
                             be_sl_price = (entry_price + offset_value).quantize(tick_size, rounding=ROUND_UP)
                        else:  # Short
                             be_sl_price = (entry_price - offset_value).quantize(tick_size, rounding=ROUND_DOWN)

                        if be_sl_price and be_sl_price > 0:
                            lg.debug(f"  Calculated BE SL Price: {be_sl_price.normalize()} (Entry {entry_price.normalize()} +/- {offset_value.normalize()})")
                            # Check if this BE SL is better than the current SL (if any)
                            update_sl = False
                            if sl_curr is None:
                                update_sl = True
                                lg.info("  Current SL is not set. Setting SL to BE.")
                            elif pos_side == 'long' and be_sl_price > sl_curr or pos_side == 'short' and be_sl_price < sl_curr:
                                update_sl = True
                                lg.info(f"  New BE SL {be_sl_price.normalize()} is better than current SL {sl_curr.normalize()}.")
                            else:
                                lg.debug(f"  Current SL {sl_curr.normalize()} is already at or better than the calculated BE SL {be_sl_price.normalize()}. No SL update needed.")

                            # If update is needed, call the protection function
                            if update_sl:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving Stop Loss to Break-Even at {be_sl_price.normalize()} ***{RESET}")
                                # Keep existing TP if any
                                if _set_position_protection(exchange, symbol, market_info, open_position, lg, stop_loss_price=be_sl_price, take_profit_price=tp_curr):
                                    lg.info(f"{NEON_GREEN}Break-Even SL set/updated successfully.{RESET}")
                                else:
                                    lg.error(f"{NEON_RED}Failed to set/update Break-Even SL via API.{RESET}")
                        else:
                            lg.error(f"{NEON_RED}Break-Even triggered, but calculated BE SL price is invalid ({be_sl_price}). Cannot move SL.{RESET}")
                    else:
                        lg.debug("BE Profit target not yet reached.")
                except Exception as be_err:
                    lg.error(f"{NEON_RED}Error during Break-Even check: {be_err}{RESET}", exc_info=True)
            elif be_enabled:
                lg.debug(f"BE check skipped: {'TSL is already active' if tsl_active else 'Missing required data (entry/ATR/price)'}.")
            else:  # BE disabled
                lg.debug("BE check skipped: Break-Even is disabled in config.")

            # --- TSL Setup/Recovery Logic (If TSL enabled but not detected as active) ---
            # This handles cases where initial TSL setting failed or was cleared externally.
            tsl_enabled = prot_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not tsl_active and entry_price and current_atr:
                 lg.warning(f"{NEON_YELLOW}Trailing Stop Loss is enabled but not currently active on the position. Attempting to set/recover TSL...{RESET}")
                 # Recalculate initial TP based on current ATR (in case it changed) to potentially set alongside TSL
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(entry_price, pos_side.upper(), current_atr, market_info, exchange)
                 if set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc):
                     lg.info(f"{NEON_GREEN}TSL setup/recovery attempt successful.{RESET}")
                 else:
                     lg.error(f"{NEON_RED}TSL setup/recovery attempt failed.{RESET}")
            elif tsl_enabled and tsl_active:
                 lg.debug("TSL is enabled and already active. No recovery needed.")
            elif not tsl_enabled:
                 lg.debug("TSL is disabled. Skipping TSL setup/recovery check.")
            # Note: This logic doesn't *adjust* an active TSL based on market changes, only sets it if missing.
            # Adjusting active TSL would require more complex logic based on current price vs activation price.

    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"{BRIGHT}---== Analysis Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ==---{RESET}\n")


# --- Main Function ---
def main() -> None:
    """Main function to initialize the bot, run user prompts, and start the trading loop."""
    global CONFIG, QUOTE_CURRENCY  # Allow main to potentially update config in memory

    # Use init_logger for setup messages before symbol-specific logger is created
    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Volumatic Bot v1.1.3 ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{RESET}")
    init_logger.info(f"Loaded Configuration: Quote={QUOTE_CURRENCY}, Trading Enabled={CONFIG['enable_trading']}, Sandbox Mode={CONFIG['use_sandbox']}")
    try:
        # Log dependency versions for easier debugging
        init_logger.info(f"Using versions: Python={os.sys.version.split()[0]}, CCXT={ccxt.__version__}, Pandas={pd.__version__}, Pandas-TA={getattr(ta, 'version', 'N/A')}, Numpy={np.__version__}, Requests={requests.__version__}")
    except Exception as e:
        init_logger.warning(f"Could not retrieve all dependency versions: {e}")

    # --- User Confirmation for Live Trading ---
    if CONFIG.get("enable_trading", False):
        init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! TRADING IS ENABLED !!!{RESET}")
        if CONFIG.get('use_sandbox', True):
            init_logger.warning("Mode: SANDBOX (Testnet) - No real funds at risk.")
        else:
            init_logger.warning(f"{NEON_RED}{BRIGHT}Mode: LIVE (Real Funds) - Confirm settings carefully!{RESET}")

        # Display critical settings for review
        prot_cfg = CONFIG["protection"]
        init_logger.warning(f"{BRIGHT}--- Critical Settings Review ---{RESET}")
        init_logger.warning(f"  Risk Per Trade: {CONFIG['risk_per_trade']:.2%}")
        init_logger.warning(f"  Leverage: {CONFIG['leverage']}x (Ensure matches exchange setting if not using bot leverage control)")
        init_logger.warning(f"  Trailing Stop: {'ENABLED' if prot_cfg['enable_trailing_stop'] else 'DISABLED'}")
        if prot_cfg['enable_trailing_stop']:
            init_logger.warning(f"    - Callback Rate: {prot_cfg['trailing_stop_callback_rate']:.3%}")
            init_logger.warning(f"    - Activation Profit %: {prot_cfg['trailing_stop_activation_percentage']:.3%}")
        init_logger.warning(f"  Break Even: {'ENABLED' if prot_cfg['enable_break_even'] else 'DISABLED'}")
        if prot_cfg['enable_break_even']:
            init_logger.warning(f"    - Trigger Profit: {prot_cfg['break_even_trigger_atr_multiple']} ATR")
            init_logger.warning(f"    - Offset: {prot_cfg['break_even_offset_ticks']} ticks")
        init_logger.warning(f"  Initial SL Multiple: {prot_cfg['initial_stop_loss_atr_multiple']} ATR")
        init_logger.warning(f"  Initial TP Multiple: {prot_cfg['initial_take_profit_atr_multiple']} ATR {'(Disabled)' if prot_cfg['initial_take_profit_atr_multiple'] == 0 else ''}")

        try:
            # Prompt user to confirm before proceeding with live trading
            input(f"\n{BRIGHT}>>> Press {NEON_GREEN}Enter{RESET}{BRIGHT} to confirm settings and start trading, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
            init_logger.info("User confirmed settings. Proceeding with trading.")
        except KeyboardInterrupt:
            init_logger.info("User aborted startup via Ctrl+C.")
            # Close log handlers cleanly before exiting
            logging.shutdown()
            for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
            return
    else:
        init_logger.info(f"{NEON_YELLOW}Trading is disabled. Running in analysis-only mode.{RESET}")

    # --- Initialize Exchange ---
    init_logger.info("Initializing CCXT exchange...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Exchange initialization failed. Cannot continue. Exiting.{RESET}")
        # Close log handlers
        logging.shutdown()
        for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
        return
    init_logger.info(f"Exchange '{exchange.id}' initialized successfully.")

    # --- Get Target Symbol ---
    target_symbol = None
    market_info = None
    while target_symbol is None:
        try:
            symbol_input = input(f"{NEON_YELLOW}Enter the trading symbol (e.g., BTC/USDT:USDT for Bybit Linear): {RESET}").strip().upper()
            if not symbol_input: continue  # Ask again if empty input

            # Attempt to validate the symbol and get market info
            init_logger.info(f"Validating symbol '{symbol_input}'...")
            m_info = get_market_info(exchange, symbol_input, init_logger)

            if m_info:
                 target_symbol = m_info['symbol']  # Use the symbol confirmed by CCXT/exchange
                 market_info = m_info
                 init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_info.get('contract_type_str', 'Unknown')})")

                 # CRITICAL Check: Ensure precision info is present before proceeding
                 if market_info.get('precision', {}).get('price') is None or \
                    market_info.get('precision', {}).get('amount') is None:
                      init_logger.critical(f"{NEON_RED}CRITICAL ERROR: Market '{target_symbol}' is missing price or amount precision information in its market data. Cannot trade safely. Exiting.{RESET}")
                      # Close log handlers
                      logging.shutdown()
                      for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
                      return
                 break  # Exit loop once valid symbol is found
            else:
                 # get_market_info logs the specific error (not found, invalid, etc.)
                 init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' could not be validated. Please check the format and ensure it exists on {exchange.id}. Try again.{RESET}")
                 # Suggest common formats
                 init_logger.info("Common formats: BASE/QUOTE (e.g., ETH/USD), BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT)")

        except KeyboardInterrupt:
            init_logger.info("User aborted during symbol selection.")
            # Close log handlers
            logging.shutdown()
            for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
            return
        except Exception as e:
            init_logger.error(f"An unexpected error occurred during symbol input/validation: {e}", exc_info=True)
            # Allow user to try again

    # --- Get Timeframe ---
    selected_interval = None
    while selected_interval is None:
        default_int = CONFIG['interval']  # Get default from loaded config
        interval_input = input(f"{NEON_YELLOW}Enter timeframe {VALID_INTERVALS} (default: {default_int}): {RESET}").strip()

        if not interval_input:  # User pressed Enter, use default
             interval_input = default_int
             init_logger.info(f"Using default timeframe: {interval_input}")

        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             # Update config in memory if user chose a different interval than default
             if CONFIG['interval'] != selected_interval:
                 CONFIG["interval"] = selected_interval
                 init_logger.info(f"Timeframe set to '{selected_interval}' for this session (overriding config default '{default_int}').")
             ccxt_tf = CCXT_INTERVAL_MAP[selected_interval]
             init_logger.info(f"Using timeframe: {selected_interval} (CCXT mapping: {ccxt_tf})")
             break  # Exit loop
        else:
             init_logger.error(f"{NEON_RED}Invalid timeframe '{interval_input}'. Please choose from: {VALID_INTERVALS}{RESET}")

    # --- Setup Symbol-Specific Logger & Strategy Instances ---
    symbol_logger = setup_logger(target_symbol)  # Create logger named after the symbol
    symbol_logger.info(f"---=== {BRIGHT}Starting Trading Loop for: {target_symbol} (Timeframe: {CONFIG['interval']}){RESET} ===---")
    symbol_logger.info(f"Trading Enabled: {CONFIG['enable_trading']}, Sandbox Mode: {CONFIG['use_sandbox']}")
    # Log final settings being used for this symbol
    prot_cfg = CONFIG["protection"]
    symbol_logger.info(f"Key Settings: Risk={CONFIG['risk_per_trade']:.2%}, Leverage={CONFIG['leverage']}x, "
                       f"TSL={'ON' if prot_cfg['enable_trailing_stop'] else 'OFF'}, BE={'ON' if prot_cfg['enable_break_even'] else 'OFF'}")

    try:
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_err:
        symbol_logger.critical(f"{NEON_RED}Failed to initialize strategy engine or signal generator: {engine_err}. Exiting.{RESET}", exc_info=True)
        # Close log handlers
        logging.shutdown()
        for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
        for handler in symbol_logger.handlers[:]: symbol_logger.removeHandler(handler); handler.close()
        return

    # --- Main Trading Loop ---
    symbol_logger.info(f"{BRIGHT}Entering main trading loop... Press Ctrl+C to stop gracefully.{RESET}")
    loop_count = 0
    try:
        while True:
            loop_start_time = time.time()
            loop_count += 1
            symbol_logger.debug(f">>> Starting Loop Cycle #{loop_count} at {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")

            try:  # --- Core Analysis and Trading Call ---
                analyze_and_trade_symbol(
                    exchange, target_symbol, CONFIG, symbol_logger,
                    strategy_engine, signal_generator, market_info
                )
            # --- Robust Exception Handling for the Loop ---
            except ccxt.RateLimitExceeded as e:
                symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded in main loop: {e}. Waiting 60s...{RESET}")
                time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e:
                symbol_logger.error(f"{NEON_RED}Network error encountered in main loop: {e}. Waiting {RETRY_DELAY_SECONDS * 3}s...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e:
                # Critical error, likely requires user intervention (API keys)
                symbol_logger.critical(f"{NEON_RED}CRITICAL AUTHENTICATION ERROR: {e}. Stopping bot.{RESET}")
                break  # Exit the main loop
            except ccxt.ExchangeNotAvailable as e:
                symbol_logger.error(f"{NEON_RED}Exchange is temporarily unavailable: {e}. Waiting 60s...{RESET}")
                time.sleep(60)
            except ccxt.OnMaintenance as e:
                symbol_logger.error(f"{NEON_RED}Exchange is under maintenance: {e}. Waiting 5 minutes...{RESET}")
                time.sleep(300)
            except ccxt.ExchangeError as e:
                # Catch-all for other CCXT exchange-related errors
                symbol_logger.error(f"{NEON_RED}Unhandled CCXT Exchange Error in main loop: {e}{RESET}", exc_info=True)
                # Short delay before next cycle for potentially transient issues
                time.sleep(10)
            except Exception as loop_err:
                # Catch any other unexpected errors within the loop
                symbol_logger.error(f"{NEON_RED}Critical unexpected error in main trading loop: {loop_err}{RESET}", exc_info=True)
                # Wait a bit longer before retrying after an unknown error
                time.sleep(15)

            # --- Loop Delay Calculation ---
            elapsed_time = time.time() - loop_start_time
            loop_delay = config.get("loop_delay_seconds", LOOP_DELAY_SECONDS)
            sleep_time = max(0, loop_delay - elapsed_time)  # Ensure non-negative sleep time
            symbol_logger.debug(f"<<< Loop Cycle #{loop_count} took {elapsed_time:.2f}s. Sleeping for {sleep_time:.2f}s...")
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt detected. Shutting down gracefully...")
    except Exception as critical_err:
        # Catch errors outside the inner try/except (e.g., during sleep or loop control)
        init_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR outside main loop: {critical_err}{RESET}", exc_info=True)
        if 'symbol_logger' in locals():  # Log to symbol logger too if available
             symbol_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR outside main loop: {critical_err}{RESET}", exc_info=True)

    finally:
        # --- Shutdown Procedures ---
        shutdown_msg = f"--- Pyrmethus Bot ({target_symbol or 'N/A'}) Stopping ---"
        init_logger.info(shutdown_msg)
        # Log shutdown message to symbol logger if it was initialized
        if 'symbol_logger' in locals() and isinstance(symbol_logger, logging.Logger):
            symbol_logger.info(shutdown_msg)

        # Close exchange connection (optional for synchronous CCXT, but good practice)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                init_logger.info("Attempting to close exchange connection (if applicable)...")
                # exchange.close() # Uncomment if using async or specific exchanges require it
                init_logger.info("Exchange connection closed or cleanup skipped.")
            except Exception as close_err:
                init_logger.error(f"Error during exchange close: {close_err}")

        # Explicitly close all logging handlers to flush buffers
        try:
            loggers = [l for l in logging.Logger.manager.loggerDict.values() if isinstance(l, logging.Logger)]
            loggers.append(logging.getLogger())  # Include root logger
            if 'init_logger' in locals(): loggers.append(init_logger)
            if 'symbol_logger' in locals(): loggers.append(symbol_logger)

            unique_handlers = set()
            for logger_instance in loggers:
                 if hasattr(logger_instance, 'handlers'):
                     for handler in logger_instance.handlers:
                         unique_handlers.add(handler)

            for handler in unique_handlers:
                try:
                    handler.flush()
                    handler.close()
                except Exception:
                    pass

            logging.shutdown()  # Final standard library cleanup
        except Exception:
             pass


if __name__ == "__main__":
    # REMINDER: Ensure 'vt_vol_ema_length' and other lookbacks in config.json
    # result in a `min_data_len` less than the API limit (1000 for Bybit).
    main()
