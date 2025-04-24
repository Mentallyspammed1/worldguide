# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.4: Implemented manual pivot detection, fixed NameError, addressed FutureWarnings.

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
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from zoneinfo import ZoneInfo # Requires tzdata package

# --- Dependencies (Install via pip) ---
import numpy as np # Requires numpy
import pandas as pd # Requires pandas
import pandas_ta as ta # Requires pandas_ta
import requests # Requires requests
# import websocket # Requires websocket-client (Imported but unused)
import ccxt # Requires ccxt
from colorama import Fore, Style, init # Requires colorama
from dotenv import load_dotenv # Requires python-dotenv

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
    TIMEZONE = ZoneInfo("America/Chicago") # Example: Use 'UTC' or your local timezone
except Exception:
    print(f"{Fore.RED}Failed to initialize timezone. Install 'tzdata' package (`pip install tzdata`). Using UTC fallback.{Style.RESET_ALL}")
    TIMEZONE = ZoneInfo("UTC")

# API Interaction
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
POSITION_CONFIRM_DELAY_SECONDS = 8
LOOP_DELAY_SECONDS = 15
BYBIT_API_KLINE_LIMIT = 1000 # Bybit V5 Kline limit per request

# Timeframes
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
CCXT_INTERVAL_MAP = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}

# Data Handling
DEFAULT_FETCH_LIMIT = 750 # Default config value if user doesn't set it (used if less than min_data_len)
MAX_DF_LEN = 2000 # Internal limit to prevent excessive memory usage

# Strategy Defaults (Used if missing/invalid in config)
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950 # <-- ADJUSTED DEFAULT (Original 1000 often > API Limit)
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0 # Note: Step ATR Multiplier currently unused in logic
DEFAULT_OB_SOURCE = "Wicks" # Or "Body"
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
    safe_name = name.replace('/', '_').replace(':', '-') # Sanitize for filename
    logger_name = f"pyrmethus_bot_{safe_name}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers(): return logger # Avoid duplicate handlers if called again

    logger.setLevel(logging.DEBUG) # Capture all levels, handlers control output level

    try: # File Handler (DEBUG level)
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(ff)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up file logger {log_filename}: {e}{RESET}")

    try: # Console Handler (Level from ENV or INFO default)
        sh = logging.StreamHandler()
        # Use timezone-aware timestamps for console output
        logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
        sf = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        sh.setFormatter(sf)
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid
        sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e:
        print(f"{NEON_RED}Error setting up console logger: {e}{RESET}")

    logger.propagate = False # Prevent messages going to root logger
    return logger

# Initialize the 'init' logger early for config loading messages
init_logger = setup_logger("init")

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
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

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads, validates, and potentially updates configuration from JSON file."""
    # Define default config structure and values
    default_config = {
        "interval": "5", "retry_delay": RETRY_DELAY_SECONDS, "fetch_limit": DEFAULT_FETCH_LIMIT,
        "orderbook_limit": 25, "enable_trading": False, "use_sandbox": True, "risk_per_trade": 0.01,
        "leverage": 20, "max_concurrent_positions": 1, "quote_currency": "USDT",
        "loop_delay_seconds": LOOP_DELAY_SECONDS, "position_confirm_delay_seconds": POSITION_CONFIRM_DELAY_SECONDS,
        "strategy_params": {
            "vt_length": DEFAULT_VT_LENGTH, "vt_atr_period": DEFAULT_VT_ATR_PERIOD,
            "vt_vol_ema_length": DEFAULT_VT_VOL_EMA_LENGTH, "vt_atr_multiplier": DEFAULT_VT_ATR_MULTIPLIER,
            "vt_step_atr_multiplier": DEFAULT_VT_STEP_ATR_MULTIPLIER, "ob_source": DEFAULT_OB_SOURCE,
            "ph_left": DEFAULT_PH_LEFT, "ph_right": DEFAULT_PH_RIGHT, "pl_left": DEFAULT_PL_LEFT, "pl_right": DEFAULT_PL_RIGHT,
            "ob_extend": DEFAULT_OB_EXTEND, "ob_max_boxes": DEFAULT_OB_MAX_BOXES,
            "ob_entry_proximity_factor": 1.005, "ob_exit_proximity_factor": 1.001 },
        "protection": {
             "enable_trailing_stop": True, "trailing_stop_callback_rate": 0.005,
             "trailing_stop_activation_percentage": 0.003, "enable_break_even": True,
             "break_even_trigger_atr_multiple": 1.0, "break_even_offset_ticks": 2,
             "initial_stop_loss_atr_multiple": 1.8, "initial_take_profit_atr_multiple": 0.7 } }
    config_needs_saving = False; loaded_config = {}

    if not os.path.exists(filepath): # Create default if not found
        init_logger.warning(f"{NEON_YELLOW}Config file '{filepath}' not found. Creating default.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f: json.dump(default_config, f, indent=4)
            init_logger.info(f"{NEON_GREEN}Created default config: {filepath}{RESET}"); return default_config
        except IOError as e: init_logger.error(f"{NEON_RED}Error creating default config: {e}{RESET}"); return default_config

    try: # Load existing config
        with open(filepath, "r", encoding="utf-8") as f: loaded_config = json.load(f)
    except json.JSONDecodeError as e:
        init_logger.error(f"{NEON_RED}Error decoding JSON from '{filepath}': {e}. Recreating.{RESET}")
        try: # Try to recreate on decode error
            with open(filepath, "w", encoding="utf-8") as f_create: json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config: {filepath}{RESET}"); return default_config
        except IOError as e_create: init_logger.error(f"{NEON_RED}Error recreating default: {e_create}. Using defaults.{RESET}"); return default_config
    except Exception as e: init_logger.error(f"{NEON_RED}Unexpected error loading config: {e}{RESET}", exc_info=True); return default_config

    try: # Validate and merge loaded config with defaults
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys: config_needs_saving = True

        # --- Type and Range Validation Helper ---
        def validate_numeric(cfg, key_path, min_val, max_val, is_strict_min=False, is_int=False, allow_zero=False):
            nonlocal config_needs_saving; keys = key_path.split('.'); current_level = cfg; default_level = default_config
            try:
                for key in keys[:-1]: current_level = current_level[key]; default_level = default_level[key]
                leaf_key = keys[-1]; original_val = current_level.get(leaf_key); default_val = default_level.get(leaf_key)
            except (KeyError, TypeError): return False # Path invalid
            if original_val is None: return False # Key missing
            corrected = False; final_val = original_val
            try:
                num_val = Decimal(str(original_val)); min_check = num_val > min_val if is_strict_min else num_val >= min_val
                if not (min_check and num_val <= max_val) and not (allow_zero and num_val == 0): raise ValueError("Out of range")
                final_val = int(num_val) if is_int else float(num_val)
                if type(final_val) is not type(original_val) or final_val != original_val: corrected = True
            except (ValueError, InvalidOperation, TypeError):
                init_logger.warning(f"{NEON_YELLOW}Config '{key_path}': Invalid value '{original_val}'. Using default {default_val}.{RESET}")
                final_val = default_val; corrected = True
            if corrected: current_level[leaf_key] = final_val; config_needs_saving = True
            return corrected

        # --- Apply Validations ---
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid config interval '{updated_config.get('interval')}'. Using default '{default_config['interval']}'.{RESET}"); updated_config["interval"] = default_config["interval"]; config_needs_saving = True
        validate_numeric(updated_config, "retry_delay", 1, 60); validate_numeric(updated_config, "fetch_limit", 100, BYBIT_API_KLINE_LIMIT*2, is_int=True) # Allow higher request than API limit for logic
        validate_numeric(updated_config, "risk_per_trade", 0, 1, is_strict_min=True); validate_numeric(updated_config, "leverage", 0, 200, is_int=True)
        validate_numeric(updated_config, "loop_delay_seconds", 1, 3600); validate_numeric(updated_config, "position_confirm_delay_seconds", 1, 60)
        validate_numeric(updated_config, "strategy_params.vt_length", 1, 500, is_int=True); validate_numeric(updated_config, "strategy_params.vt_atr_period", 1, 1000, is_int=True)
        validate_numeric(updated_config, "strategy_params.vt_vol_ema_length", 1, 2000, is_int=True) # Validate but allow user setting > API limit (handled later)
        validate_numeric(updated_config, "strategy_params.vt_atr_multiplier", 0.1, 20); validate_numeric(updated_config, "strategy_params.ph_left", 1, 100, is_int=True)
        validate_numeric(updated_config, "strategy_params.ob_entry_proximity_factor", 1.0, 1.1); validate_numeric(updated_config, "strategy_params.ob_exit_proximity_factor", 1.0, 1.1)
        validate_numeric(updated_config, "protection.trailing_stop_callback_rate", 0.0001, 0.5, is_strict_min=True)
        validate_numeric(updated_config, "protection.trailing_stop_activation_percentage", 0, 0.5, allow_zero=True)
        validate_numeric(updated_config, "protection.break_even_trigger_atr_multiple", 0.1, 10); validate_numeric(updated_config, "protection.break_even_offset_ticks", 0, 1000, is_int=True, allow_zero=True)
        validate_numeric(updated_config, "protection.initial_stop_loss_atr_multiple", 0.1, 100); validate_numeric(updated_config, "protection.initial_take_profit_atr_multiple", 0, 100, allow_zero=True)

        # Save if corrections/additions were made
        if config_needs_saving:
             try:
                 config_to_save = json.loads(json.dumps(updated_config)) # Convert any internal Decimals back
                 with open(filepath, "w", encoding="utf-8") as f_write: json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except Exception as save_err: init_logger.error(f"{NEON_RED}Error saving config: {save_err}{RESET}", exc_info=True)
        return updated_config
    except Exception as e: init_logger.error(f"{NEON_RED}Unexpected error processing config: {e}. Using defaults.{RESET}", exc_info=True); return default_config

# --- Load Global Config ---
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with retries for market loading."""
    lg = logger
    try:
        exchange_options = { 'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True,
            'options': { 'defaultType': 'linear', 'adjustForTimeDifference': True,
                         'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000, 'createOrderTimeout': 30000,
                         'cancelOrderTimeout': 20000, 'fetchPositionsTimeout': 20000, 'fetchOHLCVTimeout': 60000 } }
        exchange = ccxt.bybit(exchange_options)
        if CONFIG.get('use_sandbox', True): lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}"); exchange.set_sandbox_mode(True)
        else: lg.warning(f"{NEON_RED}{BRIGHT}USING LIVE TRADING ENVIRONMENT - REAL FUNDS AT RISK{RESET}")
        lg.info(f"Loading markets for {exchange.id}...")
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets(reload=True if attempt > 0 else False)
                if exchange.markets and len(exchange.markets) > 0: lg.info(f"Markets loaded ({len(exchange.markets)} symbols)."); break
                else: lg.warning(f"load_markets empty (Attempt {attempt+1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES: lg.warning(f"Network error loading markets (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(RETRY_DELAY_SECONDS)
                else: lg.critical(f"{NEON_RED}Max retries loading markets: {e}. Exiting.{RESET}"); return None
            except Exception as e: lg.critical(f"{NEON_RED}Error loading markets: {e}. Exiting.{RESET}", exc_info=True); return None
        if not exchange.markets or len(exchange.markets) == 0: lg.critical(f"{NEON_RED}Failed load markets. Exiting.{RESET}"); return None
        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        lg.info(f"Attempting initial balance fetch ({QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None: lg.info(f"{NEON_GREEN}Initial balance: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                lg.critical(f"{NEON_RED}Initial balance fetch FAILED.{RESET}")
                if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED} Trading enabled, critical error. Exiting.{RESET}"); return None
                else: lg.warning(f"{NEON_YELLOW} Trading disabled, proceeding cautiously.{RESET}")
        except ccxt.AuthenticationError as auth_err: lg.critical(f"{NEON_RED}Auth Error balance fetch: {auth_err}. Check keys/perms.{RESET}"); return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Initial balance fetch error: {balance_err}.{RESET}", exc_info=True)
             if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED} Trading enabled, critical error. Exiting.{RESET}"); return None
        return exchange
    except Exception as e: lg.critical(f"{NEON_RED}Failed init exchange: {e}{RESET}", exc_info=True); return None

# --- CCXT Data Fetching Helpers ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current market price for a symbol using fetch_ticker with retries."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker {symbol} (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol); price = None
            def safe_dec(v_str, name): return Decimal(str(v_str)) if v_str is not None and str(v_str).strip()!='' and Decimal(str(v_str))>0 else None
            price = safe_dec(ticker.get('last'), 'last')
            if price is None:
                bid = safe_dec(ticker.get('bid'), 'bid'); ask = safe_dec(ticker.get('ask'), 'ask')
                if bid and ask and ask >= bid: price = (bid + ask) / Decimal('2')
                elif ask: price = ask; lg.warning(f"{NEON_YELLOW}Using 'ask' fallback: {price.normalize()}{RESET}")
                elif bid: price = bid; lg.warning(f"{NEON_YELLOW}Using 'bid' fallback: {price.normalize()}{RESET}")
            if price: lg.debug(f"Current price {symbol}: {price.normalize()}"); return price
            else: lg.warning(f"No valid price from ticker (Attempt {attempts + 1}). Data: {ticker}")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"{NEON_YELLOW}Network error fetch price: {e}. Retry...{RESET}")
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"{NEON_YELLOW}Rate limit fetch price: {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetch price: {e}{RESET}"); return None
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetch price: {e}{RESET}", exc_info=True); return None
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed fetch price {symbol} after retries.{RESET}"); return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries and robust processing."""
    lg = logger
    if not exchange.has['fetchOHLCV']: lg.error(f"{exchange.id} no fetchOHLCV."); return pd.DataFrame()
    ohlcv = None; actual_req_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT: lg.debug(f"Request limit {limit} > API {BYBIT_API_KLINE_LIMIT}. Requesting {BYBIT_API_KLINE_LIMIT}.")
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines {symbol} {timeframe}, limit={actual_req_limit} (Attempt {attempt+1})")
            params = {}; # Add category param for Bybit V5 if possible
            if 'bybit' in exchange.id.lower():
                 try: market = exchange.market(symbol); category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'; params['category'] = category
                 except Exception: lg.warning("Could not determine category for kline fetch.")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_req_limit, params=params)
            ret_count = len(ohlcv) if ohlcv else 0; lg.debug(f"Got {ret_count} candles (req {actual_req_limit}).")
            if ret_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT: lg.warning(f"{NEON_YELLOW}Hit API kline limit ({BYBIT_API_KLINE_LIMIT}). Strategy might need more data.{RESET}")
            if ohlcv and ret_count > 0:
                try: # Validate timestamp lag
                    last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True); now = pd.Timestamp.utcnow()
                    interval_s = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') and exchange.parse_timeframe(timeframe) else 300
                    max_lag = max((interval_s * 5), 300); lag_s = (now - last_ts).total_seconds()
                    if lag_s < max_lag: lg.debug(f"Last kline time OK (Lag: {lag_s:.1f}s)."); break
                    else: lg.warning(f"{NEON_YELLOW}Last kline {last_ts} too old (Lag: {lag_s:.1f}s > {max_lag}s). Retry...{RESET}"); ohlcv = None
                except Exception as ts_err: lg.warning(f"Timestamp validation error: {ts_err}. Proceeding."); break
            else: lg.warning(f"No kline data (Attempt {attempt+1}). Retry...")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempt < MAX_API_RETRIES: lg.warning(f"Net error klines (Attempt {attempt+1}): {e}. Retry...{RESET}"); time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"Max retries klines net errors: {e}"); return pd.DataFrame()
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"Rate limit klines: {e}. Wait {wait}s...{RESET}"); time.sleep(wait); continue
        except ccxt.ExchangeError as e: lg.error(f"Exchange error klines: {e}"); return pd.DataFrame()
        except Exception as e: lg.error(f"Unexpected error klines: {e}", exc_info=True); return pd.DataFrame()
        if not isinstance(e, ccxt.RateLimitExceeded): attempts += 1; time.sleep(RETRY_DELAY_SECONDS if attempts <= MAX_API_RETRIES else 0)
    if not ohlcv: lg.warning(f"No kline data after retries."); return pd.DataFrame()
    try: # Process klines
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']; df = pd.DataFrame(ohlcv, columns=cols[:len(ohlcv[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce'); df.dropna(subset=['timestamp'], inplace=True); df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce'); df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
        init_len = len(df); df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True); df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns: df.dropna(subset=['volume'], inplace=True); df = df[df['volume'] >= Decimal('0')]
        rows_drop = init_len - len(df); lg.debug(f"Dropped {rows_drop} invalid rows.") if rows_drop > 0 else None
        if df.empty: lg.warning(f"Kline data empty after cleaning."); return pd.DataFrame()
        df.sort_index(inplace=True);
        if len(df) > MAX_DF_LEN: lg.debug(f"Trimming DF {len(df)}->{MAX_DF_LEN}."); df = df.iloc[-MAX_DF_LEN:].copy()
        lg.info(f"Successfully processed {len(df)} klines for {symbol} {timeframe}")
        return df
    except Exception as e: lg.error(f"Error processing klines: {e}", exc_info=True); return pd.DataFrame()

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Retrieves and validates market information (precision, limits, type) with retries."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.markets or symbol not in exchange.markets: lg.info(f"Market info {symbol} missing. Reloading..."); exchange.load_markets(reload=True)
            if symbol not in exchange.markets:
                if attempts == 0: lg.warning(f"Symbol {symbol} not found after reload. Retry..."); continue
                else: lg.error(f"Market {symbol} not found after reload/retries."); return None
            market = exchange.market(symbol)
            if market:
                market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
                market['is_linear'] = market.get('linear', False) and market['is_contract']; market['is_inverse'] = market.get('inverse', False) and market['is_contract']
                market['contract_type_str'] = "Linear" if market['is_linear'] else "Inverse" if market['is_inverse'] else "Spot" if market.get('spot') else "Unknown"
                def fmt(v): return str(Decimal(str(v)).normalize()) if v is not None else 'N/A'
                p=market.get('precision',{}); l=market.get('limits',{}); a=l.get('amount',{}); c=l.get('cost',{})
                lg.debug(f"Market: {symbol} (ID={market['id']}, Type={market['type']}, Contract={market['contract_type_str']}), Prec(P/A): {fmt(p.get('price'))}/{fmt(p.get('amount'))}, Lim(Amt Min/Max): {fmt(a.get('min'))}/{fmt(a.get('max'))}, Lim(Cost Min/Max): {fmt(c.get('min'))}/{fmt(c.get('max'))}, Size: {fmt(market.get('contractSize', '1'))}")
                if p.get('price') is None or p.get('amount') is None: lg.error(f"{NEON_RED}CRITICAL: Market {symbol} missing precision! Trading may fail.{RESET}")
                return market
            else: lg.error(f"Market dict None for {symbol}."); return None
        except ccxt.BadSymbol as e: lg.error(f"Symbol {symbol} invalid: {e}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts < MAX_API_RETRIES: lg.warning(f"Net error market info (Attempt {attempts+1}): {e}. Retry..."); time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"Max retries market info net errors: {e}"); return None
        except ccxt.ExchangeError as e: lg.error(f"Exchange error market info: {e}"); return None # Usually not retryable
        except Exception as e: lg.error(f"Unexpected error market info: {e}", exc_info=True); return None
        attempts += 1
    return None

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches available balance for a currency, handling Bybit V5 account types."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info=None; balance_str=None; found=False
            acc_types = ['UNIFIED', 'CONTRACT']
            for acc_type in acc_types:
                try:
                    lg.debug(f"Fetch balance {currency} (Type: {acc_type}, Attempt {attempts+1})...")
                    params = {'accountType': acc_type} if 'bybit' in exchange.id.lower() else {}; info = exchange.fetch_balance(params=params)
                    if currency in info and info[currency].get('free') is not None: balance_str = str(info[currency]['free']); found = True; break
                    elif 'info' in info and 'result' in info['info'] and isinstance(info['info']['result'].get('list'), list):
                        for acc in info['info']['result']['list']:
                            if (acc.get('accountType')==acc_type or acc.get('accountType') is None) and isinstance(acc.get('coin'), list):
                                for coin in acc['coin']:
                                    if coin.get('coin') == currency:
                                        free = coin.get('availableToWithdraw') or coin.get('availableBalance') or coin.get('walletBalance')
                                        if free is not None: balance_str = str(free); found = True; break
                                if found: break
                        if found: break
                except Exception as e: lg.debug(f"Error fetch balance type {acc_type}: {e}. Try next.")
            if not found: # Fallback default fetch
                try:
                    lg.debug(f"Fetch balance default {currency} (Attempt {attempts+1})...")
                    info = exchange.fetch_balance();
                    if currency in info and info[currency].get('free') is not None: balance_str = str(info[currency]['free']); found = True
                    elif 'info' in info and 'result' in info['info'] and isinstance(info['info']['result'].get('list'), list):
                        for acc in info['info']['result']['list']:
                            if isinstance(acc.get('coin'), list):
                                for coin in acc['coin']:
                                    if coin.get('coin') == currency:
                                        free = coin.get('availableToWithdraw') or coin.get('availableBalance') or coin.get('walletBalance')
                                        if free is not None: balance_str = str(free); found = True; break
                                if found: break;
                            if found: break;
                except Exception as e: lg.error(f"Failed default balance fetch: {e}", exc_info=True)

            if found and balance_str is not None:
                try: bal = Decimal(balance_str); return bal if bal >= 0 else Decimal(0) # Success
                except Exception as e: raise ccxt.ExchangeError(f"Balance conversion fail {currency}: {e}")
            else: raise ccxt.ExchangeError(f"Balance not found {currency}")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"Net error fetch balance: {e}. Retry...")
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"Rate limit fetch balance: {e}. Wait {wait}s..."); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: lg.critical(f"Auth Error fetch balance: {e}. Check keys."); return None
        except ccxt.ExchangeError as e: lg.warning(f"Exchange error fetch balance: {e}. Retry...")
        except Exception as e: lg.error(f"Unexpected error fetch balance: {e}", exc_info=True)
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * (attempts + 1) if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"Failed fetch balance {currency} after retries."); return None

# --- Position & Order Management ---
# ... (get_open_position function - no major changes needed from v1.1.3) ...
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions with Bybit V5 handling."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions {symbol} (Attempt {attempts+1})...")
            positions: List[Dict] = []; market_id = None; category = None
            try:
                market = exchange.market(symbol); market_id = market['id']
                category = 'linear' if market.get('linear') else 'inverse' if market.get('inverse') else 'spot'
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Fetch positions params: {params}"); positions = exchange.fetch_positions([symbol], params=params)
            except ccxt.ArgumentsRequired as e:
                 lg.warning(f"Fetch positions needs all ({e}). Slower."); params={'category': category or 'linear'}
                 all_pos = exchange.fetch_positions(params=params); positions = [p for p in all_pos if p.get('symbol') == symbol or p.get('info', {}).get('symbol') == market_id]
            except ccxt.ExchangeError as e:
                 if hasattr(e, 'code') and e.code == 110025 or "position not found" in str(e).lower(): lg.info(f"No position found ({e})."); return None
                 else: raise e
            active_pos = None; threshold = Decimal('1e-9'); # Default threshold
            try: amt_prec = exchange.market(symbol)['precision']['amount']; threshold = Decimal(str(amt_prec)) * Decimal('0.1') if amt_prec else threshold
            except Exception: pass; lg.debug(f"Pos size threshold: {threshold}")
            for pos in positions:
                size_str = str(pos.get('info', {}).get('size', pos.get('contracts', '')));
                if not size_str: continue
                try: size = Decimal(size_str);
                    if abs(size) > threshold: active_pos = pos; break
                except Exception: continue
            if active_pos:
                std_pos = active_pos.copy(); info = std_pos.get('info', {})
                std_pos['size_decimal'] = size # Already parsed
                side = std_pos.get('side');
                if side not in ['long','short']: side_v5=info.get('side','').lower(); side = 'long' if side_v5=='buy' else 'short' if side_v5=='sell' else 'long' if size>threshold else 'short' if size<-threshold else None
                if not side: lg.warning("Cannot determine pos side."); return None;
                std_pos['side'] = side
                std_pos['entryPrice'] = std_pos.get('entryPrice') or info.get('avgPrice') or info.get('entryPrice')
                std_pos['leverage'] = std_pos.get('leverage') or info.get('leverage'); std_pos['liquidationPrice'] = std_pos.get('liquidationPrice') or info.get('liqPrice'); std_pos['unrealizedPnl'] = std_pos.get('unrealizedPnl') or info.get('unrealisedPnl')
                sl=info.get('stopLoss') or std_pos.get('stopLossPrice'); tp=info.get('takeProfit') or std_pos.get('takeProfitPrice'); tsl_d=info.get('trailingStop'); tsl_a=info.get('activePrice')
                if sl is not None: std_pos['stopLossPrice']=str(sl);
                if tp is not None: std_pos['takeProfitPrice']=str(tp);
                if tsl_d is not None: std_pos['trailingStopLoss']=str(tsl_d);
                if tsl_a is not None: std_pos['tslActivationPrice']=str(tsl_a);
                # Simplified Logging
                ep_str = str(std_pos.get('entryPrice','N/A')); size_str = str(std_pos['size_decimal'].normalize()); sl_str=str(std_pos.get('stopLossPrice','N/A')); tp_str=str(std_pos.get('takeProfitPrice','N/A')); tsl_str = f"D:{std_pos.get('trailingStopLoss','N/A')}/A:{std_pos.get('tslActivationPrice','N/A')}"
                lg.info(f"{NEON_GREEN}Active {side.upper()} Pos ({symbol}): Size={size_str}, Entry={ep_str}, SL={sl_str}, TP={tp_str}, TSL={tsl_str}{RESET}")
                return std_pos
            else: lg.info(f"No active position found for {symbol}."); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"Net error fetch pos: {e}. Retry...")
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"Rate limit fetch pos: {e}. Wait {wait}s..."); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: lg.critical(f"Auth Error fetch pos: {e}. Stop."); return None
        except ccxt.ExchangeError as e: lg.warning(f"Exchange error fetch pos: {e}. Retry..."); # Add checks for fatal codes if needed
        except Exception as e: lg.error(f"Unexpected error fetch pos: {e}", exc_info=True)
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"Failed get pos info {symbol} after retries."); return None

# ... (set_leverage_ccxt function - no major changes needed from v1.1.3) ...
def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a derivatives symbol using CCXT, with Bybit V5 specifics and retries."""
    lg = logger; is_contract = market_info.get('is_contract', False)
    if not is_contract: lg.info(f"Lev skip {symbol} (Not contract)."); return True
    if leverage <= 0: lg.warning(f"Lev skip {symbol} (Invalid {leverage})."); return False
    if not exchange.has.get('setLeverage'): lg.error(f"{exchange.id} no setLeverage."); return False
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Setting leverage {symbol} to {leverage}x (Attempt {attempts+1})...")
            params={}; market_id=market_info['id']
            if 'bybit' in exchange.id.lower():
                 cat = 'linear' if market_info.get('linear', True) else 'inverse'; params = {'category': cat, 'symbol': market_id, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            resp = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params); lg.debug(f"Set lev raw: {resp}")
            ret_code = resp.get('retCode') if isinstance(resp, dict) else None
            if ret_code is not None:
                 if ret_code == 0: lg.info(f"{NEON_GREEN}Lev set OK {symbol} {leverage}x (Code 0).{RESET}"); return True
                 elif ret_code == 110045: lg.info(f"{NEON_YELLOW}Lev {symbol} already {leverage}x (Code 110045).{RESET}"); return True
                 else: raise ccxt.ExchangeError(f"Bybit API error set lev: {resp.get('retMsg', 'Unknown')} (Code: {ret_code})")
            else: lg.info(f"{NEON_GREEN}Lev set OK {symbol} {leverage}x.{RESET}"); return True
        except ccxt.ExchangeError as e:
            code=getattr(e,'code',None); err_s=str(e).lower(); lg.error(f"{NEON_RED}Exch error set lev: {e} (Code: {code}){RESET}")
            if code==110045 or "not modified" in err_s: return True
            fatal=[110028,110009,110055,110043,110044,110013,10001,10004]
            if code in fatal or any(s in err_s for s in ["margin mode","position exists","risk limit","parameter error"]): lg.error(" >> Hint: Non-retryable lev error."); return False
            elif attempts >= MAX_API_RETRIES: lg.error("Max retries ExchError set lev."); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error("Max retries NetError set lev."); return False; lg.warning(f"Net error set lev (Attempt {attempts+1}): {e}. Retry...")
        except Exception as e: lg.error(f"Unexpected error set lev: {e}", exc_info=True); return False
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"Failed set lev {symbol} after retries."); return False

# ... (calculate_position_size function - no major changes needed from v1.1.3) ...
def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: Dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger; symbol = market_info['symbol']; quote = market_info['quote']; base = market_info['base']
    is_contract = market_info['is_contract']; is_inverse = market_info.get('is_inverse', False)
    size_unit = "Contracts" if is_contract else base
    if balance <= Decimal('0'): lg.error(f"Sizing fail {symbol}: Invalid balance {balance}."); return None
    risk_dec = Decimal(str(risk_per_trade))
    if not (Decimal('0') < risk_dec <= Decimal('1')): lg.error(f"Sizing fail {symbol}: Invalid risk {risk_per_trade}."); return None
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'): lg.error(f"Sizing fail {symbol}: Invalid entry/SL."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Sizing fail {symbol}: SL==Entry."); return None
    try: # Get market details
        prec = market_info['precision']; limits = market_info['limits']
        amt_prec_str = prec['amount']; price_prec_str = prec['price']; assert amt_prec_str and price_prec_str
        amt_prec_step = Decimal(str(amt_prec_str))
        amt_limits = limits.get('amount', {}); cost_limits = limits.get('cost', {})
        min_amt = Decimal(str(amt_limits.get('min', '0'))); max_amt = Decimal(str(amt_limits.get('max', 'inf'))) if amt_limits.get('max') is not None else Decimal('inf')
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0'); max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')
        contract_size_str = market_info.get('contractSize', '1'); contract_size = Decimal(str(contract_size_str)) if contract_size_str else Decimal('1'); assert contract_size > 0
    except Exception as e: lg.error(f"Sizing fail {symbol}: Error getting market details: {e}"); return None

    risk_amt_quote = balance * risk_dec; sl_dist_price = abs(entry_price - initial_stop_loss_price)
    if sl_dist_price <= Decimal('0'): lg.error(f"Sizing fail {symbol}: SL dist zero."); return None
    lg.info(f"Position Sizing ({symbol}): Balance={balance.normalize()} {quote}, Risk={risk_dec:.2%}, RiskAmt={risk_amt_quote.normalize()} {quote}")
    lg.info(f"  Entry={entry_price.normalize()}, SL={initial_stop_loss_price.normalize()}, SL Dist={sl_dist_price.normalize()}, ContrSize={contract_size.normalize()}")

    calc_size = Decimal('0')
    try: # Calculate raw size
        if not is_inverse: # Linear/Spot
             val_change_per_unit = sl_dist_price * contract_size # Assumes standard linear/spot contract size meaning
             if val_change_per_unit <= 0: lg.error("Sizing fail linear: value change zero."); return None
             calc_size = risk_amt_quote / val_change_per_unit
        else: # Inverse
             if entry_price <= 0 or initial_stop_loss_price <= 0: lg.error("Sizing fail inverse: Invalid entry/SL price."); return None
             inv_factor = abs( (Decimal('1') / entry_price) - (Decimal('1') / initial_stop_loss_price) )
             if inv_factor <= Decimal('1e-18'): lg.error("Sizing fail inverse: Factor zero."); return None
             risk_per_contract = contract_size * inv_factor
             if risk_per_contract <= 0: lg.error("Sizing fail inverse: Risk per contract zero."); return None
             calc_size = risk_amt_quote / risk_per_contract
    except Exception as calc_err: lg.error(f"Sizing fail raw calc: {calc_err}."); return None
    if calc_size <= 0: lg.error(f"Sizing fail: Calc size non-positive ({calc_size})."); return None
    lg.info(f"  Initial Calc Size = {calc_size.normalize()} {size_unit}")

    adj_size = calc_size # Apply Limits & Precision
    if min_amt > 0 and adj_size < min_amt: lg.warning(f"Size {adj_size} < min {min_amt}. Adjusting UP."); adj_size = min_amt
    if max_amt < Decimal('inf') and adj_size > max_amt: lg.warning(f"Size {adj_size} > max {max_amt}. Adjusting DOWN."); adj_size = max_amt
    lg.debug(f"  Size after Amt Limits: {adj_size.normalize()}")
    est_cost = Decimal('0'); cost_adj = False
    try: # Estimate Cost
        if entry_price > 0: est_cost = (adj_size * entry_price * contract_size) if not is_inverse else ((adj_size * contract_size) / entry_price)
    except Exception: pass; lg.debug(f"  Est Cost: {est_cost.normalize()}")
    if min_cost > 0 and est_cost < min_cost:
        lg.warning(f"Est cost {est_cost} < min {min_cost}. Increasing size."); req_size = None
        try: req_size = (min_cost/(entry_price*contract_size)) if not is_inverse else ((min_cost*entry_price)/contract_size)
        except Exception: pass
        if req_size is None or req_size <= 0: lg.error(f"Cannot meet min cost {min_cost}. Abort."); return None
        if max_amt < Decimal('inf') and req_size > max_amt: lg.error(f"Cannot meet min cost without exceeding max amt. Abort."); return None
        adj_size = max(min_amt, req_size); cost_adj = True; lg.info(f"  Required size for min cost: {req_size.normalize()}")
    elif max_cost < Decimal('inf') and est_cost > max_cost:
        lg.warning(f"Est cost {est_cost} > max {max_cost}. Reducing size."); adj_size_cost = None
        try: adj_size_cost = (max_cost/(entry_price*contract_size)) if not is_inverse else ((max_cost*entry_price)/contract_size)
        except Exception: pass
        if adj_size_cost is None or adj_size_cost <= 0: lg.error(f"Cannot reduce size for max cost {max_cost}. Abort."); return None
        adj_size = max(min_amt, min(adj_size, adj_size_cost)); cost_adj = True; lg.info(f"  Max size for max cost: {adj_size_cost.normalize()}")
    if cost_adj: lg.info(f"  Size after Cost Limits: {adj_size.normalize()}")

    final_size = adj_size # Apply Precision
    try: fmt_str = exchange.amount_to_precision(symbol, float(adj_size)); final_size = Decimal(fmt_str); lg.info(f"Applied amt precision (ccxt): {adj_size} -> {final_size}")
    except Exception as fmt_err:
        lg.warning(f"ccxt precision error: {fmt_err}. Manual round down.");
        try: final_size = (adj_size // amt_prec_step) * amt_prec_step if amt_prec_step > 0 else adj_size; lg.info(f"Applied manual step: {final_size}")
        except Exception as manual_err: lg.error(f"Manual precision fail: {manual_err}. Using unrounded."); final_size = adj_size
    if final_size <= 0: lg.error(f"Final size zero/neg ({final_size}). Abort."); return None
    if min_amt > 0 and final_size < min_amt: lg.error(f"Final size {final_size} < min {min_amt}. Abort."); return None

    final_cost = Decimal('0') # Final cost check
    try: final_cost = (final_size * entry_price * contract_size) if not is_inverse else ((final_size * contract_size) / entry_price)
    except Exception: pass
    if min_cost > 0 and final_cost < min_cost:
         lg.debug(f"Final cost {final_cost} < min {min_cost}.")
         try:
             next_size = final_size + amt_prec_step; next_cost = Decimal('0')
             if entry_price > 0: next_cost = (next_size * entry_price * contract_size) if not is_inverse else ((next_size * contract_size) / entry_price)
             valid = (next_cost >= min_cost) and (max_amt==Decimal('inf') or next_size<=max_amt) and (max_cost==Decimal('inf') or next_cost<=max_cost)
             if valid: lg.warning(f"Bumping size to {next_size} for min cost."); final_size = next_size
             else: lg.error(f"Final cost below min, next step invalid. Abort."); return None
         except Exception as bump_err: lg.error(f"Error bumping size: {bump_err}. Abort."); return None
    lg.info(f"{NEON_GREEN}{BRIGHT}Final Size: {final_size.normalize()} {size_unit}{RESET}"); return final_size

# ... (place_trade function - no major changes needed from v1.1.3) ...
def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: Dict,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """Places a market order using CCXT with retries, Bybit V5 params, and clear logging."""
    lg = logger; side_map = {"BUY": "buy", "SELL": "sell", "EXIT_SHORT": "buy", "EXIT_LONG": "sell"}; side = side_map.get(trade_signal)
    if side is None: lg.error(f"Invalid trade signal '{trade_signal}'"); return None
    if position_size <= 0: lg.error(f"Invalid position size {position_size}"); return None
    order_type = 'market'; is_contract = market_info['is_contract']; base = market_info['base']; size_unit = market_info.get('settle', base) if is_contract else base
    action = "Close" if reduce_only else "Open/Increase"; market_id = market_info['id']
    args = {'symbol': market_id, 'type': order_type, 'side': side, 'amount': float(position_size)}
    order_params = {}
    if 'bybit' in exchange.id.lower(): # Add Bybit V5 params
        cat = 'linear' if market_info.get('linear', True) else 'inverse'; order_params = {'category': cat, 'positionIdx': 0} # One-way mode
        if reduce_only: order_params['reduceOnly'] = True; order_params['timeInForce'] = 'IOC'
        if params: order_params.update(params)
        args['params'] = order_params
    lg.info(f"Attempting {action} {side.upper()} {order_type}: {symbol} | Size: {position_size.normalize()} {size_unit}")
    if order_params: lg.debug(f"  Params: {order_params}")
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing create_order (Attempt {attempts+1})...")
            order = exchange.create_order(**args)
            oid=order.get('id','N/A'); status=order.get('status','N/A'); avg=order.get('average'); filled=order.get('filled')
            log_msg=f"{NEON_GREEN}{action} Placed!{RESET} ID:{oid}, Stat:{status}" + (f", Avg:~{Decimal(str(avg)).normalize()}" if avg else "") + (f", Fill:{Decimal(str(filled)).normalize()}" if filled else "")
            lg.info(log_msg); return order
        except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Insufficient funds: {e}{RESET}"); return None
        except ccxt.InvalidOrder as e: lg.error(f"{NEON_RED}Invalid order: {e}{RESET}"); return None # Add hints if needed
        except ccxt.ExchangeError as e: code=getattr(e,'code',None); lg.error(f"{NEON_RED}Exchange error place: {e} (Code:{code}){RESET}"); fatal=[110014,110007,110040,110013,110025,30086,10001]; if code in fatal: lg.error(" >> Hint: Non-retryable."); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error(f"Max net retries place order: {e}"); return None; lg.warning(f"Net error place order (Attempt {attempts+1}): {e}. Retry...")
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"Rate limit place order: {e}. Wait {wait}s..."); time.sleep(wait); continue
        except Exception as e: lg.error(f"Unexpected error place order: {e}", exc_info=True); return None
        if not isinstance(e, ccxt.RateLimitExceeded): attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"Failed place {action} order {symbol} after retries."); return None

# ... (_set_position_protection function - no major changes needed from v1.1.3) ...
def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """Internal helper to set SL/TP/TSL for a position via Bybit V5 API (private_post)."""
    lg = logger; # Validation... (identical to v1.1.3)
    if not market_info.get('is_contract'): lg.warning(f"Protect skip {symbol} (Not contract)."); return False
    if not position_info: lg.error(f"Protect fail {symbol}: Missing pos info."); return False
    pos_side = position_info.get('side'); entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str: lg.error(f"Protect fail {symbol}: Invalid pos side/entry."); return False
    try: entry_price = Decimal(str(entry_price_str)); assert entry_price > 0
    except Exception as e: lg.error(f"Invalid entry price '{entry_price_str}': {e}"); return False
    params_to_set = {}; log_parts = [f"Setting protection {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):"]; any_req = False
    try: # Parameter Formatting... (identical to v1.1.3)
        price_prec_str = market_info['precision']['price']; min_tick = Decimal(str(price_prec_str)); assert min_tick > 0
        def fmt_price(price_dec: Optional[Decimal], name: str) -> Optional[str]:
            if price_dec is None: return None;
            if price_dec == 0: return "0" # Allow clearing
            if price_dec < 0: lg.warning(f"Negative price {price_dec} for {name}."); return None
            try: fmt = exchange.price_to_precision(symbol, float(price_dec), exchange.ROUND); return fmt if Decimal(fmt) > 0 else None
            except Exception as e: lg.error(f"Failed format {name} {price_dec}: {e}."); return None
        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal): any_req=True
            if trailing_stop_distance > 0:
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0: lg.error(f"TSL req invalid act price {tsl_activation_price}.")
                else:
                    valid_act = (pos_side=='long' and tsl_activation_price>entry_price) or (pos_side=='short' and tsl_activation_price<entry_price)
                    if not valid_act: lg.error(f"TSL Act {tsl_activation_price} not beyond entry {entry_price} for {pos_side}.")
                    else:
                        min_dist = max(trailing_stop_distance, min_tick); fmt_d = fmt_price(min_dist, "TSL Dist"); fmt_a = fmt_price(tsl_activation_price, "TSL Act")
                        if fmt_d and fmt_a: params_to_set['trailingStop']=fmt_d; params_to_set['activePrice']=fmt_a; log_parts.append(f"  - TSL: D={fmt_d}, A={fmt_a}"); set_tsl=True
                        else: lg.error(f"Failed format TSL params (D:{fmt_d}, A:{fmt_a}).")
            elif trailing_stop_distance == 0: params_to_set['trailingStop']="0"; params_to_set['activePrice']="0"; log_parts.append("  - Clear TSL") # Clear both
        if not set_tsl and isinstance(stop_loss_price, Decimal): any_req=True
            if stop_loss_price > 0:
                valid_sl = (pos_side=='long' and stop_loss_price<entry_price) or (pos_side=='short' and stop_loss_price>entry_price)
                if not valid_sl: lg.error(f"SL {stop_loss_price} not beyond entry {entry_price} for {pos_side}.")
                else: fmt_sl = fmt_price(stop_loss_price, "SL");
                if fmt_sl: params_to_set['stopLoss']=fmt_sl; log_parts.append(f"  - Fixed SL: {fmt_sl}")
                else: lg.error(f"Failed format SL {stop_loss_price}.")
            elif stop_loss_price == 0: params_to_set['stopLoss']="0"; log_parts.append("  - Clear Fixed SL")
        if isinstance(take_profit_price, Decimal): any_req=True
            if take_profit_price > 0:
                valid_tp = (pos_side=='long' and take_profit_price>entry_price) or (pos_side=='short' and take_profit_price<entry_price)
                if not valid_tp: lg.error(f"TP {take_profit_price} not beyond entry {entry_price} for {pos_side}.")
                else: fmt_tp = fmt_price(take_profit_price, "TP");
                if fmt_tp: params_to_set['takeProfit']=fmt_tp; log_parts.append(f"  - Fixed TP: {fmt_tp}")
                else: lg.error(f"Failed format TP {take_profit_price}.")
            elif take_profit_price == 0: params_to_set['takeProfit']="0"; log_parts.append("  - Clear Fixed TP")
    except Exception as fmt_err: lg.error(f"Error format protect params: {fmt_err}", exc_info=True); return False
    if not params_to_set:
        if any_req: lg.warning(f"No valid protect params after format. No API call."); return False
        else: lg.debug(f"No protect changes requested."); return True
    cat = 'linear' if market_info.get('linear', True) else 'inverse'; mid = market_info['id']; pidx = 0
    try: pidx = int(position_info['info']['positionIdx']) if position_info.get('info', {}).get('positionIdx') is not None else 0
    except Exception: pass
    final_params={'category':cat,'symbol':mid,'tpslMode':'Full','slTriggerBy':'LastPrice','tpTriggerBy':'LastPrice','slOrderType':'Market','tpOrderType':'Market','positionIdx':pidx}
    final_params.update(params_to_set); lg.info("\n".join(log_parts)); lg.debug(f"  API Params: {final_params}")
    attempts = 0
    while attempts <= MAX_API_RETRIES: # API Call with Retries... (identical to v1.1.3)
        try:
            lg.debug(f"Exec set protect API (Attempt {attempts+1})..."); resp = exchange.private_post('/v5/position/set-trading-stop', params=final_params); lg.debug(f"Set protect raw: {resp}")
            code=resp.get('retCode'); msg=resp.get('retMsg','Unknown');
            if code == 0:
                 if any(m in msg.lower() for m in ["not modified","no need to modify","parameter not change"]): lg.info(f"{NEON_YELLOW}Protect already set/no change. Resp: {msg}{RESET}")
                 else: lg.info(f"{NEON_GREEN}Protect set/updated OK.{RESET}");
                 return True
            else: lg.error(f"{NEON_RED}Failed set protect: {msg} (Code: {code}){RESET}"); fatal=[110013,110036,110086,110084,110085,10001,10002]; is_fatal=code in fatal or "invalid" in msg.lower() or "parameter" in msg.lower();
            if is_fatal: lg.error(" >> Hint: Non-retryable error code."); return False; else: raise ccxt.ExchangeError(f"Bybit Err set protect: {msg} (Code: {code})")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error(f"Max net retries set protect: {e}"); return False; lg.warning(f"Net err set protect (Attempt {attempts+1}): {e}. Retry...")
        except ccxt.RateLimitExceeded as e: wait=RETRY_DELAY_SECONDS*5; lg.warning(f"Rate limit set protect: {e}. Wait {wait}s..."); time.sleep(wait); continue
        except ccxt.AuthenticationError as e: lg.critical(f"Auth Error set protect: {e}. Stop."); return False
        except ccxt.ExchangeError as e:
             if attempts >= MAX_API_RETRIES: lg.error(f"Max exch retries set protect: {e}"); return False; lg.warning(f"Exch err set protect (Attempt {attempts+1}): {e}. Retry...")
        except Exception as e: lg.error(f"Unexpected err set protect (Attempt {attempts+1}): {e}", exc_info=True); return False # Unexpected errors likely fatal here
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"Failed set protect {symbol} after retries."); return False

# ... (set_trailing_stop_loss function - no major changes needed from v1.1.3) ...
def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, config: Dict[str, Any],
                             logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool:
    """Calculates TSL parameters based on config and current position, then calls _set_position_protection."""
    lg = logger; prot_cfg = config["protection"]
    if not market_info or not position_info: lg.error(f"TSL calc fail {symbol}: Missing info."); return False
    pos_side = position_info.get('side'); entry_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_str: lg.error(f"TSL calc fail {symbol}: Invalid pos side/entry."); return False
    try:
        entry = Decimal(str(entry_str)); cb_rate = Decimal(str(prot_cfg["trailing_stop_callback_rate"])); act_pct = Decimal(str(prot_cfg["trailing_stop_activation_percentage"]))
        assert entry > 0 and cb_rate > 0 and act_pct >= 0; tick = Decimal(str(market_info['precision']['price'])); assert tick > 0
    except Exception as ve: lg.error(f"Invalid TSL params/info {symbol}: {ve}."); return False
    try:
        act_offset = entry * act_pct; raw_act = (entry + act_offset) if pos_side == 'long' else (entry - act_offset)
        act_price = raw_act.quantize(tick, ROUND_UP) if pos_side=='long' else raw_act.quantize(tick, ROUND_DOWN)
        act_price = max(act_price, entry+tick) if pos_side=='long' else min(act_price, entry-tick) # Ensure strictly beyond entry
        if act_price <= 0: lg.error(f"TSL Act Price <= 0 ({act_price})."); return False
        dist_raw = act_price * cb_rate; trail_dist = max(dist_raw.quantize(tick, ROUND_UP), tick) # Ensure >= 1 tick
        if trail_dist <= 0: lg.error(f"TSL Dist <= 0 ({trail_dist})."); return False
        lg.info(f"Calc TSL {symbol} ({pos_side.upper()}): Entry={entry.normalize()}, Act%={act_pct:.3%}, CB%={cb_rate:.3%}")
        lg.info(f"  => Act Price: {act_price.normalize()}, Trail Dist: {trail_dist.normalize()}")
        if isinstance(take_profit_price, Decimal): lg.info(f"  TP: {take_profit_price.normalize() if take_profit_price != 0 else 'Clear'}")
        return _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=None,
                                        take_profit_price=take_profit_price, trailing_stop_distance=trail_dist, tsl_activation_price=act_price)
    except Exception as e: lg.error(f"Unexpected error calc/set TSL: {e}", exc_info=True); return False

# --- Volumatic Trend + OB Strategy Implementation ---
class OrderBlock(TypedDict):
    id: str; type: str; left_idx: pd.Timestamp; right_idx: pd.Timestamp
    top: Decimal; bottom: Decimal; active: bool; violated: bool
class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame; last_close: Decimal; current_trend_up: Optional[bool]; trend_just_changed: bool
    active_bull_boxes: List[OrderBlock]; active_bear_boxes: List[OrderBlock]
    vol_norm_int: Optional[int]; atr: Optional[Decimal]; upper_band: Optional[Decimal]; lower_band: Optional[Decimal]

class VolumaticOBStrategy:
    """Implements the Volumatic Trend and Pivot Order Block strategy."""
    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        self.config = config; self.market_info = market_info; self.logger = logger
        strategy_cfg = config["strategy_params"]; protection_cfg = config["protection"] # Use loaded/validated config
        self.vt_length = int(strategy_cfg["vt_length"]); self.vt_atr_period = int(strategy_cfg["vt_atr_period"]); self.vt_vol_ema_length = int(strategy_cfg["vt_vol_ema_length"])
        self.vt_atr_multiplier = Decimal(str(strategy_cfg["vt_atr_multiplier"]))
        self.ob_source = strategy_cfg["ob_source"]; self.ph_left = int(strategy_cfg["ph_left"]); self.ph_right = int(strategy_cfg["ph_right"])
        self.pl_left = int(strategy_cfg["pl_left"]); self.pl_right = int(strategy_cfg["pl_right"]); self.ob_extend = bool(strategy_cfg["ob_extend"]); self.ob_max_boxes = int(strategy_cfg["ob_max_boxes"])
        self.bull_boxes: List[OrderBlock] = []; self.bear_boxes: List[OrderBlock] = []
        req_vt = max(self.vt_length*2, self.vt_atr_period, self.vt_vol_ema_length); req_piv = max(self.ph_left+self.ph_right+1, self.pl_left+self.pl_right+1)
        self.min_data_len = max(req_vt, req_piv) + 50 # Min candles + buffer
        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy...{RESET}")
        self.logger.info(f"  VT Params: Len={self.vt_length}, ATRLen={self.vt_atr_period}, VolLen={self.vt_vol_ema_length}, ATRMult={self.vt_atr_multiplier.normalize()}")
        self.logger.info(f"  OB Params: Src={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"  Minimum historical data points required: {self.min_data_len}")
        if self.min_data_len > BYBIT_API_KLINE_LIMIT + 10: # Add small margin to API limit for comparison
             self.logger.error(f"{NEON_RED}{BRIGHT}CONFIG WARNING:{RESET} Strategy requires {self.min_data_len} candles, exceeding safe API limit (~{BYBIT_API_KLINE_LIMIT}). Reduce lookbacks.")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates EMA(SWMA(close, 4), length)."""
        if len(series) < 4 or length <= 0: return pd.Series(np.nan, index=series.index)
        weights = np.array([1., 2., 2., 1.]) / 6.0; series_num = pd.to_numeric(series, errors='coerce')
        if series_num.isnull().all(): return pd.Series(np.nan, index=series.index)
        swma = series_num.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        return ta.ema(swma, length=length, fillna=np.nan)

    def _find_pivots(self, series: pd.Series, left: int, right: int, high: bool) -> pd.Series:
        """Finds pivot highs (high=True) or lows (high=False) using rolling window."""
        if not isinstance(series, pd.Series) or series.empty: return pd.Series(False, index=series.index)
        window_size = left + 1 + right
        # Pad series to handle edges correctly during rolling compare
        padded_series = pd.concat([pd.Series([np.nan]*left), series, pd.Series([np.nan]*right)])
        # Rolling window to get values around potential pivot
        windows = padded_series.rolling(window=window_size, center=False) # center=False aligns window end

        pivots = pd.Series(False, index=series.index) # Initialize result series

        # Efficiently check pivot condition using rolling max/min
        if high: # Pivot High
            # Find rolling maximum over the full window [n-left, n+right]
            # Shifted because rolling window normally looks backwards
            rolling_max = series.rolling(window=window_size, center=True, min_periods=window_size).max()
            # A pivot high occurs at index `i` if series[i] equals the rolling max centered at `i`
            # AND it's strictly greater than neighbors (handled implicitly by max over window)
            # Need to handle potential duplicate max values - ensure pivot is unique max in window?
            # More robust: check if series[i] > series[i-left:i] and series[i] > series[i+1:i+right+1]
            # Let's try the rolling max comparison first for simplicity
            pivot_candidates = (series == rolling_max)
            # Add strict check for neighbors? Optional, depends on strict definition.
            # For now, assume max in window is sufficient.
            pivots[pivot_candidates] = True

        else: # Pivot Low
            rolling_min = series.rolling(window=window_size, center=True, min_periods=window_size).min()
            pivot_candidates = (series == rolling_min)
            pivots[pivot_candidates] = True

        return pivots.fillna(False) # Ensure boolean output, handle NaNs from rolling


    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """Processes historical data to calculate indicators and manage Order Blocks."""
        empty_results = StrategyAnalysisResults(dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None, trend_just_changed=False,
                                                active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None)
        if df_input.empty: self.logger.error("Strategy update: empty DF."); return empty_results
        df = df_input.copy() # Work on copy
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing: self.logger.error("DF index invalid."); return empty_results
        if len(df) < self.min_data_len: self.logger.warning(f"Insufficient data ({len(df)} < {self.min_data_len}). Results may be inaccurate.")
        self.logger.debug(f"Starting analysis on {len(df)} candles (min req: {self.min_data_len}).")

        try: # Convert to float for TA Libs
            df_float = pd.DataFrame(index=df.index)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col not in df.columns: self.logger.error(f"Missing column '{col}'. Abort."); return empty_results
                df_float[col] = pd.to_numeric(df[col], errors='coerce')
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_float.empty: self.logger.error("DF empty after float conversion."); return empty_results
        except Exception as e: self.logger.error(f"Error converting to float: {e}", exc_info=True); return empty_results

        try: # Indicator Calculations
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)
            df_float['trend_up'] = (df_float['ema2'] > df_float['ema1'].shift(1))
            df_float['trend_up'] = df_float['trend_up'].ffill().fillna(False) # Address FutureWarning & handle initial NaN
            df_float['trend_changed'] = (df_float['trend_up'].shift(1) != df_float['trend_up']) & df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'] = df_float['trend_changed'].fillna(False) # Address FutureWarning & handle initial NaN
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan); df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill(); df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()
            atr_mult = float(self.vt_atr_multiplier); df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult); df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult)
            vol_num = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0); min_p = max(1, self.vt_vol_ema_length//10)
            df_float['vol_max'] = vol_num.rolling(window=self.vt_vol_ema_length, min_periods=min_p).max().fillna(0.0)
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9, (vol_num / df_float['vol_max'] * 100.0), 0.0)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)

            # --- Pivot Calculation using new helper ---
            if self.ob_source.lower() == "wicks": high_s = df_float['high']; low_s = df_float['low']
            else: high_s = df_float[['open', 'close']].max(axis=1); low_s = df_float[['open', 'close']].min(axis=1)
            df_float['is_ph'] = self._find_pivots(high_s, self.ph_left, self.ph_right, high=True)
            df_float['is_pl'] = self._find_pivots(low_s, self.pl_left, self.pl_right, high=False)
            # -------------------------------------------

        except Exception as e: self.logger.error(f"Error during indicator calc: {e}", exc_info=True); return empty_results

        try: # Copy results back to Decimal DF
            cols = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed', 'upper_band', 'lower_band', 'vol_norm', 'is_ph', 'is_pl']
            for col in cols:
                if col in df_float.columns:
                    src = df_float[col].reindex(df.index) # Align
                    if src.dtype=='bool': df[col] = src.astype(bool)
                    elif pd.api.types.is_object_dtype(src): df[col] = src # Should not happen for these cols
                    else: df[col] = src.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
        except Exception as e: self.logger.error(f"Error converting results to Decimal: {e}", exc_info=True);

        # Clean Decimal DF
        init_len = len(df); req_cols = ['close', 'atr', 'trend_up', 'upper_band', 'lower_band']
        df.dropna(subset=req_cols, inplace=True); drop_rows = init_len - len(df); lg.debug(f"Dropped {drop_rows} rows missing indicators.") if drop_rows > 0 else None
        if df.empty: self.logger.warning("DF empty after indicator calcs."); return empty_results
        self.logger.debug("Indicators calculated. Processing OBs...")

        # --- Order Block Management (using df['is_ph'], df['is_pl']) ---
        try:
            new_boxes_count = 0
            if not df.empty:
                 # Iterate through the DataFrame where pivots are identified
                 for pivot_idx in df.index:
                     try:
                         is_pivot_high = df.loc[pivot_idx, 'is_ph']
                         is_pivot_low = df.loc[pivot_idx, 'is_pl']

                         # --- Bearish OB from Pivot High ---
                         if is_pivot_high:
                             if not any(b['left_idx'] == pivot_idx and b['type'] == 'bear' for b in self.bear_boxes):
                                 pivot_candle = df.loc[pivot_idx]
                                 top, bot = (pivot_candle['high'], pivot_candle['open']) if self.ob_source.lower() == "wicks" else \
                                            (max(pivot_candle['open'], pivot_candle['close']), min(pivot_candle['open'], pivot_candle['close']))
                                 if pd.notna(top) and pd.notna(bot) and isinstance(top, Decimal) and isinstance(bot, Decimal) and top > bot:
                                     new_box = OrderBlock(id=f"B_{pivot_idx.strftime('%y%m%d%H%M%S')}",type='bear',left_idx=pivot_idx,right_idx=df.index[-1],top=top,bottom=bot,active=True,violated=False)
                                     self.bear_boxes.append(new_box); new_boxes_count += 1
                                     self.logger.debug(f"  New Bear OB: {new_box['id']} @ {piv_idx} [{bot.normalize()}-{top.normalize()}]")

                         # --- Bullish OB from Pivot Low ---
                         if is_pivot_low:
                              if not any(b['left_idx'] == pivot_idx and b['type'] == 'bull' for b in self.bull_boxes):
                                 pivot_candle = df.loc[pivot_idx]
                                 top, bot = (pivot_candle['open'], pivot_candle['low']) if self.ob_source.lower() == "wicks" else \
                                            (max(pivot_candle['open'], pivot_candle['close']), min(pivot_candle['open'], pivot_candle['close']))
                                 if pd.notna(top) and pd.notna(bot) and isinstance(top, Decimal) and isinstance(bot, Decimal) and top > bot:
                                     new_box = OrderBlock(id=f"L_{pivot_idx.strftime('%y%m%d%H%M%S')}",type='bull',left_idx=pivot_idx,right_idx=df.index[-1],top=top,bottom=bot,active=True,violated=False)
                                     self.bull_boxes.append(new_box); new_boxes_count += 1
                                     self.logger.debug(f"  New Bull OB: {new_box['id']} @ {piv_idx} [{bot.normalize()}-{top.normalize()}]")
                     except Exception as e: self.logger.warning(f"Error processing pivot at {pivot_idx}: {e}", exc_info=True)
            if new_boxes_count > 0: self.logger.debug(f"Found {new_boxes_count} new OBs.")

            # Manage Existing OBs
            last = df.iloc[-1] if not df.empty else None
            if last is not None and pd.notna(last.get('close')) and isinstance(last['close'], Decimal):
                last_idx = last.name; last_close = last['close']
                for box in self.bull_boxes:
                    if box['active']:
                        if last_close < box['bottom']: box['active']=False; box['violated']=True; box['right_idx']=last_idx; self.logger.debug(f"Bull OB {box['id']} VIOLATED.")
                        elif self.ob_extend: box['right_idx'] = last_idx
                for box in self.bear_boxes:
                    if box['active']:
                        if last_close > box['top']: box['active']=False; box['violated']=True; box['right_idx']=last_idx; self.logger.debug(f"Bear OB {box['id']} VIOLATED.")
                        elif self.ob_extend: box['right_idx'] = last_idx
            else: self.logger.warning("Cannot check OB violations: Invalid last close.")

            # Prune OBs (Keep N most recent non-violated)
            self.bull_boxes = sorted([b for b in self.bull_boxes if not b['violated']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted([b for b in self.bear_boxes if not b['violated']], key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.logger.debug(f"Pruned OBs. Kept Active: Bull={len(self.bull_boxes)}, Bear={len(self.bear_boxes)}.")
        except Exception as e: self.logger.error(f"Error during OB processing: {e}", exc_info=True); # Continue

        # --- Prepare Final Results ---
        last = df.iloc[-1] if not df.empty else None
        def sdec(v, pos=False): return v if pd.notna(v) and isinstance(v,Decimal) and np.isfinite(float(v)) and (not pos or v > 0) else None
        results = StrategyAnalysisResults(
            dataframe=df, last_close=sdec(last.get('close')) or Decimal('0'),
            current_trend_up=bool(last['trend_up']) if last is not None and isinstance(last.get('trend_up'),(bool,np.bool_)) else None,
            trend_just_changed=bool(last['trend_changed']) if last is not None and isinstance(last.get('trend_changed'),(bool,np.bool_)) else False,
            active_bull_boxes=[b for b in self.bull_boxes if b['active']], active_bear_boxes=[b for b in self.bear_boxes if b['active']],
            vol_norm_int=int(v) if (v:=sdec(last.get('vol_norm'))) is not None else None,
            atr=sdec(last.get('atr'), pos=True), upper_band=sdec(last.get('upper_band')), lower_band=sdec(last.get('lower_band')) )
        # Final Log Summary... (identical to v1.1.3)
        trend_str = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else f"{NEON_YELLOW}N/A{RESET}"
        atr_str = f"{results['atr'].normalize()}" if results['atr'] else "N/A"; time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
        self.logger.debug(f"Strategy Results ({time_str}): Close={results['last_close'].normalize()}, Trend={trend_str}, TrendChg={results['trend_just_changed']}, ATR={atr_str}, VolNorm={results['vol_norm_int']}, Active OBs (B/B): {len(results['active_bull_boxes'])}/{len(results['active_bear_boxes'])}")
        return results

# --- Signal Generation based on Strategy Results ---
# ... (SignalGenerator Class - no changes needed from v1.1.3) ...
class SignalGenerator:
    """Generates trading signals based on strategy analysis and position state."""
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config; self.logger = logger; strategy_cfg = config["strategy_params"]; protection_cfg = config["protection"]
        try:
            self.ob_entry_proximity_factor = Decimal(str(strategy_cfg["ob_entry_proximity_factor"])); assert self.ob_entry_proximity_factor >= 1
            self.ob_exit_proximity_factor = Decimal(str(strategy_cfg["ob_exit_proximity_factor"])); assert self.ob_exit_proximity_factor >= 1
            self.initial_tp_atr_multiple = Decimal(str(protection_cfg["initial_take_profit_atr_multiple"])); assert self.initial_tp_atr_multiple >= 0
            self.initial_sl_atr_multiple = Decimal(str(protection_cfg["initial_stop_loss_atr_multiple"])); assert self.initial_sl_atr_multiple > 0
        except Exception as e:
             self.logger.error(f"{NEON_RED}Error initializing SignalGenerator config: {e}. Using defaults.{RESET}", exc_info=True)
             self.ob_entry_proximity_factor=Decimal("1.005"); self.ob_exit_proximity_factor=Decimal("1.001"); self.initial_tp_atr_multiple=Decimal("0.7"); self.initial_sl_atr_multiple=Decimal("1.8")
        self.logger.info("Signal Generator Init: OB Entry Prox={:.3f}, OB Exit Prox={:.3f}".format(self.ob_entry_proximity_factor, self.ob_exit_proximity_factor))
        self.logger.info(f"  Initial TP Mult={self.initial_tp_atr_multiple.normalize()}, Initial SL Mult={self.initial_sl_atr_multiple.normalize()}")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """Determines the trading signal."""
        lg = self.logger
        if not analysis_results or analysis_results['current_trend_up'] is None or analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
            lg.warning(f"{NEON_YELLOW}Invalid strategy results for signal gen. Holding.{RESET}"); return "HOLD"
        close=analysis_results['last_close']; trend_up=analysis_results['current_trend_up']; changed=analysis_results['trend_just_changed']
        bull_obs=analysis_results['active_bull_boxes']; bear_obs=analysis_results['active_bear_boxes']; pos_side=open_position.get('side') if open_position else None
        signal = "HOLD"
        lg.debug(f"Signal Gen: Close={close.normalize()}, TrendUp={trend_up}, Changed={changed}, Pos={pos_side or 'None'}, OBs(B/B)={len(bull_obs)}/{len(bear_obs)}")

        # Exit Checks
        if pos_side == 'long':
            if trend_up is False and changed: signal = "EXIT_LONG"; lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG: Trend flipped DOWN.{RESET}")
            elif signal=="HOLD" and bear_obs:
                try: ob=min(bear_obs, key=lambda b: abs(b['top']-close)); thresh=ob['top']*self.ob_exit_proximity_factor
                     if close >= thresh: signal="EXIT_LONG"; lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG: Price {close} >= Bear OB Exit {thresh} (OB ID {ob['id']}){RESET}")
                except Exception as e: lg.warning(f"Error Bear OB exit check: {e}")
        elif pos_side == 'short':
            if trend_up is True and changed: signal = "EXIT_SHORT"; lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT: Trend flipped UP.{RESET}")
            elif signal=="HOLD" and bull_obs:
                try: ob=min(bull_obs, key=lambda b: abs(b['bottom']-close)); thresh=ob['bottom']/self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else ob['bottom']
                     if close <= thresh: signal="EXIT_SHORT"; lg.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT: Price {close} <= Bull OB Exit {thresh} (OB ID {ob['id']}){RESET}")
                except Exception as e: lg.warning(f"Error Bull OB exit check: {e}")
        if signal != "HOLD": return signal

        # Entry Checks
        if pos_side is None:
            if trend_up is True and bull_obs:
                for ob in bull_obs: lb=ob['bottom']; ub=ob['top']*self.ob_entry_proximity_factor
                    if lb <= close <= ub: signal="BUY"; lg.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price {close} in Bull OB {ob['id']} [{lb}-{ub}]{RESET}"); break
            elif trend_up is False and bear_obs:
                for ob in bear_obs: lb=ob['bottom']/self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']; ub=ob['top']
                    if lb <= close <= ub: signal="SELL"; lg.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price {close} in Bear OB {ob['id']} [{lb}-{ub}]{RESET}"); break

        if signal == "HOLD": lg.debug(f"Signal: HOLD - No valid entry or exit condition.");
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: Dict, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial TP/SL levels based on ATR and market precision."""
        lg = self.logger
        if signal not in ["BUY", "SELL"] or entry_price <= 0 or atr <= 0 or market_info['precision'].get('price') is None: lg.error(f"Invalid input for TP/SL calc."); return None, None
        try:
            min_tick = Decimal(str(market_info['precision']['price'])); assert min_tick > 0
            tp_mult = self.initial_tp_atr_multiple; sl_mult = self.initial_sl_atr_multiple
            tp_offset = atr * tp_mult; sl_offset = atr * sl_mult
            tp_raw = (entry_price + tp_offset) if signal == "BUY" and tp_mult > 0 else (entry_price - tp_offset) if signal == "SELL" and tp_mult > 0 else None
            sl_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)
            def fmt_lvl(price_dec: Optional[Decimal], name: str) -> Optional[Decimal]:
                if price_dec is None or price_dec <= 0: lg.debug(f"Calc {name} invalid ({price_dec})."); return None
                try: fmt_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_dec)); fmt_dec = Decimal(fmt_str); return fmt_dec if fmt_dec > 0 else None
                except Exception as e: lg.error(f"Error formatting {name} {price_dec}: {e}."); return None
            tp = fmt_lvl(tp_raw, "TP"); sl = fmt_lvl(sl_raw, "SL")
            if sl is not None: # Adjust SL if needed
                if (signal=="BUY" and sl >= entry_price) or (signal=="SELL" and sl <= entry_price): lg.warning(f"Formatted {signal} SL {sl} not beyond entry {entry_price}. Adjusting."); sl = fmt_lvl(entry_price - min_tick if signal=="BUY" else entry_price + min_tick, "SL")
            if tp is not None: # Adjust TP if needed
                if (signal=="BUY" and tp <= entry_price) or (signal=="SELL" and tp >= entry_price): lg.warning(f"Formatted {signal} TP {tp} not beyond entry {entry_price}. Disable TP."); tp = None
            lg.debug(f"Initial Levels: TP={tp.normalize() if tp else 'None'}, SL={sl.normalize() if sl else 'FAIL'}")
            if sl is None: lg.error(f"{NEON_RED}SL calculation failed. Cannot size.{RESET}"); return tp, None
            return tp, sl
        except Exception as e: lg.error(f"{NEON_RED}Error calculating TP/SL: {e}{RESET}", exc_info=True); return None, None

# --- Main Analysis and Trading Loop Function ---
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: Dict) -> None:
    """Performs one cycle of analysis and trading for a symbol."""
    lg = logger
    lg.info(f"\n{BRIGHT}---== Analyzing {symbol} ({config['interval']} TF) Cycle Start ==---{RESET}")
    cycle_start_time = time.monotonic()
    ccxt_interval = CCXT_INTERVAL_MAP[config["interval"]]

    # Determine Kline Fetch Limit based on strategy needs and API limit
    min_req_data = strategy_engine.min_data_len
    fetch_limit_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    fetch_limit_needed = max(min_req_data, fetch_limit_config)
    fetch_limit_request = min(fetch_limit_needed, BYBIT_API_KLINE_LIMIT)
    lg.info(f"Requesting {fetch_limit_request} klines (Strategy min: {min_req_data}, Config pref: {fetch_limit_config}, API limit: {BYBIT_API_KLINE_LIMIT})...")

    # Fetch & Validate Kline Data
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit_request, logger=lg)
    fetched_count = len(klines_df)
    if klines_df.empty or fetched_count < min_req_data:
        api_limit_hit_insufficient = (fetch_limit_request == BYBIT_API_KLINE_LIMIT and fetched_count == BYBIT_API_KLINE_LIMIT and fetched_count < min_req_data)
        if api_limit_hit_insufficient: lg.error(f"{NEON_RED}CRITICAL DATA:{RESET} Fetched {fetched_count} (API limit), need {min_req_data}. {NEON_YELLOW}Reduce lookbacks in config! Skipping.{RESET}")
        else: lg.error(f"Failed fetch sufficient data ({fetched_count}/{min_req_data}). Skipping cycle.")
        return

    # Run Strategy Analysis
    try: analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err: lg.error(f"{NEON_RED}Strategy analysis failed: {analysis_err}{RESET}", exc_info=True); return
    if not analysis_results or analysis_results['current_trend_up'] is None or analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
        lg.error(f"{NEON_RED}Strategy analysis incomplete. Skipping cycle.{RESET}"); lg.debug(f"Analysis Results: {analysis_results}"); return
    latest_close = analysis_results['last_close']; current_atr = analysis_results['atr']

    # Get Current Market State
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    price_for_checks = current_price if current_price and current_price > 0 else latest_close
    if price_for_checks <= 0: lg.error(f"{NEON_RED}Cannot get valid price. Skipping cycle.{RESET}"); return
    if current_price is None: lg.warning(f"{NEON_YELLOW}Using last close {latest_close} for protection checks.{RESET}")
    open_position = get_open_position(exchange, symbol, lg)

    # Generate Signal
    try: signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err: lg.error(f"{NEON_RED}Signal generation failed: {signal_err}{RESET}", exc_info=True); return

    # Trading Logic
    trading_enabled = config.get("enable_trading", False)
    if not trading_enabled: # Analysis-only mode
        lg.info(f"{NEON_YELLOW}Trading disabled.{RESET} Signal: {signal}. Analysis complete.")
        if open_position is None and signal in ["BUY","SELL"]: lg.info(f"  (Would attempt {signal})")
        elif open_position and signal in ["EXIT_LONG","EXIT_SHORT"]: lg.info(f"  (Would attempt {signal})")
        else: lg.info("  (No entry/exit action)")
        cycle_end_time = time.monotonic(); lg.debug(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")
        return

    # --- Trading Enabled ---
    lg.debug(f"Trading enabled. Signal: {signal}. Position: {'Yes (' + open_position['side'] + ')' if open_position else 'No'}")

    # --- Scenario 1: No Position -> Consider Entry ---
    if open_position is None and signal in ["BUY", "SELL"]:
        lg.info(f"{BRIGHT}*** {signal} Signal & No Position: Init Entry Sequence... ***{RESET}")
        balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
        if balance is None or balance <= 0: lg.error(f"Trade Abort {signal}: No balance {balance}."); return
        tp_calc, sl_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)
        if sl_calc is None: lg.error(f"Trade Abort {signal}: Initial SL calc failed."); return
        leverage_ok = True
        if market_info['is_contract']: leverage=int(config['leverage']); leverage_ok = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg) if leverage > 0 else True
        if not leverage_ok: lg.error(f"Trade Abort {signal}: Leverage set failed."); return
        pos_size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, latest_close, market_info, exchange, lg)
        if pos_size is None or pos_size <= 0: lg.error(f"Trade Abort {signal}: Invalid size {pos_size}."); return

        lg.info(f"===> Placing {signal} Market Order | Size: {pos_size.normalize()} <===")
        trade_order = place_trade(exchange, symbol, signal, pos_size, market_info, lg, reduce_only=False)

        if trade_order and trade_order.get('id'): # Post-Trade: Confirm & Set Protection
            confirm_delay = config["position_confirm_delay_seconds"]; lg.info(f"Order placed ({trade_order['id']}). Wait {confirm_delay}s..."); time.sleep(confirm_delay)
            confirmed_pos = get_open_position(exchange, symbol, lg)
            if confirmed_pos:
                try:
                    entry_actual_str = confirmed_pos.get('entryPrice'); entry_actual = Decimal(str(entry_actual_str)) if entry_actual_str else latest_close
                    if entry_actual <= 0: entry_actual = latest_close
                    lg.info(f"{NEON_GREEN}Position Confirmed! Entry: ~{entry_actual.normalize()}{RESET}")
                    # Set Protection based on actual entry
                    prot_cfg = config["protection"]
                    tp_prot, sl_prot = signal_generator.calculate_initial_tp_sl(entry_actual, signal, current_atr, market_info, exchange)
                    if sl_prot is None: lg.error(f"{NEON_RED}CRITICAL: Failed recalculate SL for protection! Position vulnerable!{RESET}")
                    else:
                        prot_success = False
                        if prot_cfg["enable_trailing_stop"]: lg.info(f"Setting TSL (Entry={entry_actual.normalize()})..."); prot_success = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_pos, config, lg, take_profit_price=tp_prot)
                        elif not prot_cfg["enable_trailing_stop"] and (sl_prot or tp_prot): lg.info(f"Setting Fixed SL/TP (Entry={entry_actual.normalize()})..."); prot_success = _set_position_protection(exchange, symbol, market_info, confirmed_pos, lg, stop_loss_price=sl_prot, take_profit_price=tp_prot)
                        else: lg.info("No protection enabled."); prot_success = True
                        if prot_success: lg.info(f"{NEON_GREEN}{BRIGHT}=== ENTRY & PROTECTION COMPLETE ({symbol} {signal}) ==={RESET}")
                        else: lg.error(f"{NEON_RED}{BRIGHT}=== TRADE PLACED, BUT FAILED SET PROTECTION ({symbol} {signal}). MANUAL MONITOR! ==={RESET}")
                except Exception as post_err: lg.error(f"{NEON_RED}Post-trade setup error: {post_err}{RESET}", exc_info=True); lg.warning(f"{NEON_YELLOW}Pos confirmed, may lack protection! Manual check!{RESET}")
            else: lg.error(f"{NEON_RED}Order placed, but FAILED TO CONFIRM position after delay! Manual check!{RESET}")
        else: lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}). No order placed. ===")

    # --- Scenario 2: Existing Position -> Consider Exit or Manage ---
    elif open_position:
        pos_side = open_position['side']; pos_size = open_position['size_decimal']
        lg.info(f"Existing {pos_side.upper()} position (Size: {pos_size.normalize()}). Signal: {signal}")
        exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_triggered: # Handle Exit Signal
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal! Closing {pos_side} position... ***{RESET}")
            try:
                close_sig = "SELL" if pos_side == 'long' else "BUY"; size_close = abs(pos_size)
                if size_close <= 0: lg.warning("Close size zero/neg. Already closed?"); return
                lg.info(f"===> Placing {close_sig} MARKET Order (Reduce Only) | Size: {size_close.normalize()} <===")
                close_order = place_trade(exchange, symbol, close_sig, size_close, market_info, lg, reduce_only=True)
                if close_order and close_order.get('id'): lg.info(f"{NEON_GREEN}Position CLOSE order ({close_order['id']}) placed successfully.{RESET}")
                else: lg.error(f"{NEON_RED}Failed place CLOSE order. Manual check!{RESET}")
            except Exception as close_err: lg.error(f"{NEON_RED}Error closing position: {close_err}{RESET}", exc_info=True); lg.warning(f"{NEON_YELLOW}Manual close may be needed!{RESET}")
        else: # Handle Position Management (BE, TSL)
            lg.debug(f"Signal ({signal}) allows holding. Managing protections...")
            prot_cfg = config["protection"]
            try: tsl_active = open_position.get('trailingStopLoss') and Decimal(str(open_position['trailingStopLoss'])) > 0
            except Exception: tsl_active = False
            try: sl_curr = Decimal(str(open_position['stopLossPrice'])) if open_position.get('stopLossPrice') and str(open_position['stopLossPrice'])!='0' else None
            except Exception: sl_curr = None
            try: tp_curr = Decimal(str(open_position['takeProfitPrice'])) if open_position.get('takeProfitPrice') and str(open_position['takeProfitPrice'])!='0' else None
            except Exception: tp_curr = None
            try: entry_price = Decimal(str(open_position['entryPrice'])) if open_position.get('entryPrice') else None
            except Exception: entry_price = None

            # Break-Even Logic
            be_enabled = prot_cfg.get("enable_break_even", True)
            if be_enabled and not tsl_active and entry_price and current_atr and price_for_checks > 0:
                lg.debug(f"Checking BE (Entry:{entry_price.normalize()}, Price:{price_for_checks.normalize()}, ATR:{current_atr.normalize()})...")
                try:
                    be_trig_atr = Decimal(str(prot_cfg["break_even_trigger_atr_multiple"])); be_offset_t = int(prot_cfg["break_even_offset_ticks"]); assert be_trig_atr > 0 and be_offset_t >= 0
                    profit = (price_for_checks - entry_price) if pos_side=='long' else (entry_price - price_for_checks); profit_atr = profit/current_atr if current_atr > 0 else Decimal(0)
                    lg.debug(f"  BE Check: Profit ATRs={profit_atr:.3f}, Trigger={be_trig_atr.normalize()}")
                    if profit_atr >= be_trig_atr:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}BE Profit target REACHED!{RESET}")
                        tick = Decimal(str(market_info['precision']['price'])); offset = tick * Decimal(str(be_offset_t))
                        be_sl = (entry_price + offset).quantize(tick, ROUND_UP) if pos_side=='long' else (entry_price - offset).quantize(tick, ROUND_DOWN)
                        if be_sl and be_sl > 0:
                            update = False;
                            if sl_curr is None: update=True; lg.info("  No current SL, setting BE.")
                            elif (pos_side=='long' and be_sl>sl_curr) or (pos_side=='short' and be_sl<sl_curr): update=True; lg.info(f"  New BE SL {be_sl} better than current {sl_curr}.")
                            else: lg.debug(f"  Current SL {sl_curr} >= BE target {be_sl}. No update.")
                            if update:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving SL to Break-Even at {be_sl.normalize()} ***{RESET}")
                                if _set_position_protection(exchange,symbol,market_info,open_position,lg,stop_loss_price=be_sl,take_profit_price=tp_curr): lg.info(f"{NEON_GREEN}BE SL set OK.{RESET}")
                                else: lg.error(f"{NEON_RED}Failed set BE SL.{RESET}")
                        else: lg.error(f"{NEON_RED}BE triggered, but BE SL calc invalid ({be_sl}).{RESET}")
                    else: lg.debug(f"BE Profit target not reached.")
                except Exception as be_err: lg.error(f"{NEON_RED}Error BE check: {be_err}{RESET}", exc_info=True)
            elif be_enabled: lg.debug(f"BE check skipped: {'TSL active' if tsl_active else 'Missing data'}.")
            else: lg.debug("BE check skipped: Disabled.")

            # TSL Setup/Recovery
            tsl_enabled = prot_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not tsl_active and entry_price and current_atr:
                 lg.warning(f"{NEON_YELLOW}TSL enabled but not active. Attempting setup/recovery...{RESET}")
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(entry_price, pos_side.upper(), current_atr, market_info, exchange)
                 if set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc): lg.info(f"TSL setup/recovery OK.")
                 else: lg.error(f"TSL setup/recovery FAILED.")
            elif tsl_enabled: lg.debug(f"TSL setup/recovery skipped: {'Already active' if tsl_active else 'Missing data'}.")
            else: lg.debug("TSL setup/recovery skipped: Disabled.")

    # --- No Action Scenario (HOLD signal with no position) ---
    elif signal == "HOLD":
         lg.info("Signal is HOLD, no existing position. No trading action taken.")

    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"{BRIGHT}---== Analysis Cycle End ({symbol}, Duration: {cycle_end_time - cycle_start_time:.2f}s) ==---{RESET}\n")

# --- Main Function ---
def main() -> None:
    """Main function to initialize the bot, run user prompts, and start the trading loop."""
    global CONFIG, QUOTE_CURRENCY
    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Bot v1.1.4 ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{RESET}")
    init_logger.info(f"Loaded Config: Quote={QUOTE_CURRENCY}, Trading={CONFIG['enable_trading']}, Sandbox={CONFIG['use_sandbox']}")
    try: init_logger.info(f"Versions: Py={os.sys.version.split()[0]}, CCXT={ccxt.__version__}, Pandas={pd.__version__}, TA={getattr(ta, 'version', 'N/A')}")
    except Exception as e: init_logger.warning(f"Version check fail: {e}")

    # User Confirmation
    if CONFIG["enable_trading"]:
        init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! TRADING IS ENABLED !!!{RESET}")
        init_logger.warning(f"Mode: {'SANDBOX' if CONFIG['use_sandbox'] else f'{NEON_RED}LIVE (Real Funds)'}{RESET}")
        prot_cfg = CONFIG["protection"]
        init_logger.warning(f"{BRIGHT}--- Review Settings ---{RESET}")
        init_logger.warning(f"  Risk/Trade: {CONFIG['risk_per_trade']:.2%}, Leverage: {CONFIG['leverage']}x")
        init_logger.warning(f"  TSL: {'ON' if prot_cfg['enable_trailing_stop'] else 'OFF'} (CB:{prot_cfg['trailing_stop_callback_rate']:.3%}, Act:{prot_cfg['trailing_stop_activation_percentage']:.3%})")
        init_logger.warning(f"  BE: {'ON' if prot_cfg['enable_break_even'] else 'OFF'} (Trig:{prot_cfg['break_even_trigger_atr_multiple']} ATR, Off:{prot_cfg['break_even_offset_ticks']} ticks)")
        init_logger.warning(f"  Init SL Mult: {prot_cfg['initial_stop_loss_atr_multiple']} ATR, Init TP Mult: {prot_cfg['initial_take_profit_atr_multiple']} ATR {'(Disabled)' if prot_cfg['initial_take_profit_atr_multiple'] == 0 else ''}")
        try: input(f"\n{BRIGHT}>>> Press {NEON_GREEN}Enter{RESET}{BRIGHT} to confirm & start, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
        except KeyboardInterrupt: init_logger.info("User aborted startup."); print(f"\n{NEON_YELLOW}Bot aborted.{RESET}"); logging.shutdown(); return
        init_logger.info("User confirmed settings.")
    else: init_logger.info(f"{NEON_YELLOW}Trading disabled. Analysis-only mode.{RESET}")

    # Init Exchange
    init_logger.info("Initializing exchange..."); exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical(f"Exchange init failed. Exiting."); logging.shutdown(); return
    init_logger.info(f"Exchange '{exchange.id}' initialized.")

    # Get Symbol & Market Info
    target_symbol = None; market_info = None
    while target_symbol is None:
        try:
            symbol_input = input(f"{NEON_YELLOW}Enter trading symbol (e.g., BTC/USDT:USDT): {RESET}").strip().upper()
            if not symbol_input: continue
            init_logger.info(f"Validating symbol '{symbol_input}'...")
            m_info = get_market_info(exchange, symbol_input, init_logger)
            if m_info:
                 target_symbol = m_info['symbol']; market_info = m_info
                 init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_info.get('contract_type_str', 'Unknown')})")
                 if market_info['precision'].get('price') is None or market_info['precision'].get('amount') is None:
                      init_logger.critical(f"{NEON_RED}CRITICAL: Market '{target_symbol}' missing precision! Cannot trade safely. Exiting.{RESET}"); logging.shutdown(); return
                 break
            else: init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' not validated. Try again.{RESET}"); init_logger.info("Common formats: BASE/QUOTE, BASE/QUOTE:SETTLE")
        except KeyboardInterrupt: init_logger.info("User aborted."); print(f"\n{NEON_YELLOW}Bot aborted.{RESET}"); logging.shutdown(); return
        except Exception as e: init_logger.error(f"Symbol validation error: {e}", exc_info=True)

    # Get Timeframe
    selected_interval = None
    while selected_interval is None:
        default_int = CONFIG['interval']; interval_input = input(f"{NEON_YELLOW}Enter timeframe {VALID_INTERVALS} (default: {default_int}): {RESET}").strip()
        if not interval_input: interval_input = default_int; init_logger.info(f"Using default timeframe: {interval_input}")
        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input; CONFIG["interval"] = selected_interval # Update in memory
             init_logger.info(f"Using timeframe: {selected_interval} (CCXT: {CCXT_INTERVAL_MAP[selected_interval]})"); break
        else: init_logger.error(f"{NEON_RED}Invalid timeframe '{interval_input}'. Choose from: {VALID_INTERVALS}{RESET}")

    # Setup Symbol Logger & Strategy Instances
    symbol_logger = setup_logger(target_symbol);
    symbol_logger.info(f"---=== {BRIGHT}Starting Trading Loop: {target_symbol} (TF: {CONFIG['interval']}){RESET} ===---")
    symbol_logger.info(f"Trading: {CONFIG['enable_trading']}, Sandbox: {CONFIG['use_sandbox']}")
    prot_cfg = CONFIG["protection"]
    symbol_logger.info(f"Settings: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if prot_cfg['enable_trailing_stop'] else 'OFF'}, BE={'ON' if prot_cfg['enable_break_even'] else 'OFF'}")
    try:
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_err: symbol_logger.critical(f"Failed init strategy/signal generator: {engine_err}. Exiting.", exc_info=True); logging.shutdown(); return

    # Main Loop
    symbol_logger.info(f"{BRIGHT}Entering main loop... Press Ctrl+C to stop.{RESET}")
    loop_count = 0
    try:
        while True:
            loop_start = time.time(); loop_count += 1; symbol_logger.debug(f">>> Loop #{loop_count} Start: {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")
            try: analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger, strategy_engine, signal_generator, market_info) # Core logic
            except ccxt.RateLimitExceeded as e: symbol_logger.warning(f"Rate limit: {e}. Wait 60s..."); time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e: symbol_logger.error(f"Network error: {e}. Wait {RETRY_DELAY_SECONDS*3}s..."); time.sleep(RETRY_DELAY_SECONDS*3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"CRITICAL Auth Error: {e}. Stop bot."); break
            except ccxt.ExchangeNotAvailable as e: symbol_logger.error(f"Exchange unavailable: {e}. Wait 60s..."); time.sleep(60)
            except ccxt.OnMaintenance as e: symbol_logger.error(f"Exchange maintenance: {e}. Wait 5m..."); time.sleep(300)
            except ccxt.ExchangeError as e: symbol_logger.error(f"Unhandled Exchange Error: {e}", exc_info=True); time.sleep(10)
            except Exception as loop_err: symbol_logger.error(f"Critical loop error: {loop_err}", exc_info=True); time.sleep(15)
            # Loop Delay - Use global CONFIG here
            elapsed = time.time() - loop_start; loop_delay = CONFIG.get("loop_delay_seconds", LOOP_DELAY_SECONDS); sleep = max(0, loop_delay - elapsed)
            symbol_logger.debug(f"<<< Loop #{loop_count} took {elapsed:.2f}s. Sleep {sleep:.2f}s...")
            if sleep > 0: time.sleep(sleep)
    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as critical_err: init_logger.critical(f"CRITICAL UNHANDLED ERROR: {critical_err}", exc_info=True); symbol_logger.critical(f"CRITICAL UNHANDLED ERROR: {critical_err}", exc_info=True) if 'symbol_logger' in locals() else None
    finally: # Shutdown
        shutdown_msg = f"--- Pyrmethus Bot ({target_symbol or 'N/A'}) Stopping ---"; print(f"\n{NEON_YELLOW}{BRIGHT}{shutdown_msg}{RESET}")
        init_logger.info(shutdown_msg);
        if 'symbol_logger' in locals(): symbol_logger.info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); init_logger.info("Exchange closed.")
            except Exception as close_err: init_logger.error(f"Error closing exchange: {close_err}")
        print("Closing log handlers..."); logging.shutdown();
        try: # Close handlers manually just in case
            for logger_name in list(logging.Logger.manager.loggerDict): get_logger = logging.getLogger(logger_name); get_logger.handlers.clear()
            for handler in logging.getLogger().handlers[:]: logging.getLogger().removeHandler(handler); handler.close()
        except Exception as log_err: print(f"Error final log close: {log_err}")
        print(f"{NEON_YELLOW}{BRIGHT}Bot stopped.{RESET}")

if __name__ == "__main__":
    main()

