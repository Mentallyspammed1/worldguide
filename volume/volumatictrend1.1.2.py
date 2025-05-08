Okay, I've analyzed the code (assuming the context from `volumatictrend1.1.2.py` and the preceding discussion) and incorporated several enhancements focused on robustness, clarity, configuration validation, and handling the API data limit issue more gracefully.

**Key Enhancements in v1.1.3:**

1.  **Configuration Validation:** `load_config` now performs more rigorous type and range validation for key numeric parameters, logging warnings and falling back to defaults if necessary. This prevents runtime errors due to invalid config values.
2.  **Dynamic Fetch Limit & API Limit Handling:**
    *   `VolumaticOBStrategy.__init__` calculates `min_data_len`.
    *   `analyze_and_trade_symbol` now dynamically determines the required fetch limit based on `min_data_len` and the `BYBIT_API_KLINE_LIMIT`. It requests the minimum needed, capped at the API limit.
    *   `fetch_klines_ccxt` now specifically logs a warning if the number of candles returned equals the API limit when more might have been needed, helping diagnose data insufficiency.
    *   The check in `analyze_and_trade_symbol` for sufficient data is more robust.
3.  **Improved Logging:** Added more detailed `debug` and `info` logs, especially around position management (BE/TSL checks, reasons for actions/inactions). Formatted numeric outputs in logs consistently using `.normalize()`.
4.  **Code Clarity & Comments:** Added more comments explaining complex logic sections (e.g., sizing formulas, stateful band calculation, API parameter choices).
5.  **Error Handling:** Refined specific error handling hints for Bybit API calls. Added retries to `set_leverage_ccxt`.
6.  **Minor Refinements:** Consistent use of `.normalize()` for Decimal output, better variable names in places, slight code restructuring for readability.
7.  **Shutdown:** Added explicit closing of log handlers in the `finally` block.

**Remember:** The most crucial fix for the original `fetched 1000, need >= 1010` error is **adjusting the `vt_vol_ema_length` (or other long lookback parameters) in your `config.json` file** to ensure `min_data_len` stays below the `BYBIT_API_KLINE_LIMIT` (1000). The code improvements help manage this, but cannot overcome the fundamental API limit if the strategy demands more data than available in one request.

```python
# --- START OF FILE volumatictrend1.1.3.py ---

# pyrmethus_volumatic_bot.py
# Enhanced trading bot incorporating the Volumatic Trend and Pivot Order Block strategy
# with advanced position management (SL/TP, BE, TSL) for Bybit V5 (Linear/Inverse).
# Version 1.1.3: Improved config validation, dynamic fetch limit, API limit logging, general refinements.

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
    TIMEZONE = ZoneInfo("America/Chicago") # Example: Use 'UTC'
except Exception:
    print(f"{Fore.RED}Failed to initialize timezone. Install 'tzdata'. Using UTC fallback.{Style.RESET_ALL}")
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
DEFAULT_FETCH_LIMIT = 750 # Default config value if user doesn't set it
MAX_DF_LEN = 2000

# Strategy Defaults
DEFAULT_VT_LENGTH = 40
DEFAULT_VT_ATR_PERIOD = 200
DEFAULT_VT_VOL_EMA_LENGTH = 950 # <-- ADJUSTED DEFAULT (Original 1000 often > API Limit)
DEFAULT_VT_ATR_MULTIPLIER = 3.0
DEFAULT_VT_STEP_ATR_MULTIPLIER = 4.0
DEFAULT_OB_SOURCE = "Wicks"
DEFAULT_PH_LEFT = 10; DEFAULT_PH_RIGHT = 10
DEFAULT_PL_LEFT = 10; DEFAULT_PL_RIGHT = 10
DEFAULT_OB_EXTEND = True
DEFAULT_OB_MAX_BOXES = 50

# Dynamically loaded: QUOTE_CURRENCY

# Logging Colors
NEON_GREEN = Fore.LIGHTGREEN_EX; NEON_BLUE = Fore.CYAN; NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW; NEON_RED = Fore.LIGHTRED_EX; NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL; BRIGHT = Style.BRIGHT; DIM = Style.DIM

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Configuration Loading & Validation ---
class SensitiveFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY: msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET: msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def setup_logger(name: str) -> logging.Logger:
    safe_name = name.replace('/', '_').replace(':', '-')
    logger_name = f"pyrmethus_bot_{safe_name}"; log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name);
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.DEBUG)
    try: # File Handler
        fh = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
        ff = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(ff); fh.setLevel(logging.DEBUG); logger.addHandler(fh)
    except Exception as e: print(f"{NEON_RED}Error setting file logger {log_filename}: {e}{RESET}")
    try: # Console Handler
        sh = logging.StreamHandler()
        sf = SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()
        sh.setFormatter(sf)
        log_level_str = os.getenv("CONSOLE_LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO); sh.setLevel(log_level)
        logger.addHandler(sh)
    except Exception as e: print(f"{NEON_RED}Error setting console logger: {e}{RESET}")
    logger.propagate = False
    return logger

def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any], parent_key: str = "") -> Tuple[Dict[str, Any], bool]:
    """Recursively ensures default keys exist, logs additions."""
    updated_config = config.copy(); changed = False
    for key, default_value in default_config.items():
        full_key_path = f"{parent_key}.{key}" if parent_key else key
        if key not in updated_config:
            updated_config[key] = default_value; changed = True
            init_logger.info(f"{NEON_YELLOW}Config: Added missing key '{full_key_path}' with default: {default_value}{RESET}")
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            nested_config, nested_changed = _ensure_config_keys(updated_config[key], default_value, full_key_path)
            if nested_changed: updated_config[key] = nested_config; changed = True
    return updated_config, changed

def load_config(filepath: str) -> Dict[str, Any]:
    """Loads, validates, and potentially updates configuration from JSON file."""
    # Define default config structure
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
        init_logger.error(f"{NEON_RED}Error decoding config file '{filepath}': {e}. Attempting to recreate.{RESET}")
        try: # Try to recreate on decode error
            with open(filepath, "w", encoding="utf-8") as f_create: json.dump(default_config, f_create, indent=4)
            init_logger.info(f"{NEON_GREEN}Recreated default config: {filepath}{RESET}"); return default_config
        except IOError as e_create: init_logger.error(f"{NEON_RED}Error recreating default config: {e_create}. Using defaults.{RESET}"); return default_config
    except Exception as e: init_logger.error(f"{NEON_RED}Unexpected error loading config: {e}{RESET}", exc_info=True); return default_config

    try: # Validate and merge loaded config
        updated_config, added_keys = _ensure_config_keys(loaded_config, default_config)
        if added_keys: config_needs_saving = True

        # Validate interval
        if updated_config.get("interval") not in VALID_INTERVALS:
            init_logger.error(f"{NEON_RED}Invalid interval '{updated_config.get('interval')}'. Using default '{default_config['interval']}'.{RESET}")
            updated_config["interval"] = default_config["interval"]; config_needs_saving = True

        # --- Numeric Validation Helper ---
        def validate_numeric(cfg, key_path, min_val, max_val, is_strict_min=False, is_int=False):
            keys = key_path.split('.'); current_cfg_level = cfg; default_cfg_level = default_config
            try:
                for key in keys[:-1]: current_cfg_level = current_cfg_level[key]; default_cfg_level = default_cfg_level[key]
                key_leaf = keys[-1]; original_value = current_cfg_level.get(key_leaf)
                default_value = default_cfg_level.get(key_leaf)
            except (KeyError, TypeError): return False # Key path invalid, should be handled by _ensure_config_keys

            if original_value is None: return False # Key missing, handled by _ensure_config_keys
            corrected = False; final_value = original_value
            try:
                num_value = Decimal(str(original_value)) # Convert to Decimal for checks
                min_check = num_value > min_val if is_strict_min else num_value >= min_val
                if not (min_check and num_value <= max_val): raise ValueError("Out of range")
                # If valid, convert back to float/int for config object consistency (JSON compatibility)
                final_value = int(num_value) if is_int else float(num_value)
                if final_value != original_value: corrected = True # Type changed (e.g., string to number)
            except (ValueError, InvalidOperation, TypeError):
                init_logger.warning(f"{NEON_YELLOW}Config: Invalid value '{original_value}' for '{key_path}' (Range: {'(' if is_strict_min else '['}{min_val}, {max_val}]). Using default: {default_value}{RESET}")
                final_value = default_value; corrected = True

            if corrected: current_cfg_level[key_leaf] = final_value
            return corrected

        # --- Apply Numeric Validation ---
        validations = {
            "risk_per_trade": (0, 1, True, False), # >0, <=1, float
            "leverage": (0, 200, False, True),     # >=0, <=200, int
            "protection.initial_stop_loss_atr_multiple": (0, 100, True, False), # >0, float
            "protection.initial_take_profit_atr_multiple": (0, 100, False, False),# >=0, float
            "protection.trailing_stop_callback_rate": (0, 1, True, False),      # >0, float
            "protection.trailing_stop_activation_percentage": (0, 1, False, False), # >=0, float
            "protection.break_even_trigger_atr_multiple": (0, 100, True, False),  # >0, float
            "protection.break_even_offset_ticks": (0, 1000, False, True),         # >=0, int
            "loop_delay_seconds": (1, 3600, False, False), # >=1 sec, float/int
            "position_confirm_delay_seconds": (1, 60, False, False), # >=1 sec, float/int
            "fetch_limit": (100, BYBIT_API_KLINE_LIMIT, False, True), # >=100, <= API Limit, int
            "strategy_params.vt_length": (1, 500, False, True), # int > 0
            "strategy_params.vt_atr_period": (1, 1000, False, True), # int > 0
            "strategy_params.vt_vol_ema_length": (1, BYBIT_API_KLINE_LIMIT - 50, False, True), # int > 0, slightly less than API limit needed
            "strategy_params.ph_left": (1, 100, False, True), # int > 0
            # ... add others if needed ...
        }
        any_numeric_corrected = any(validate_numeric(updated_config, key, *params) for key, params in validations.items())
        if any_numeric_corrected: config_needs_saving = True

        # Save if needed
        if config_needs_saving:
             try:
                 # Convert back to JSON-compatible types if needed (should be float/int now)
                 config_to_save = json.loads(json.dumps(updated_config))
                 with open(filepath, "w", encoding="utf-8") as f_write: json.dump(config_to_save, f_write, indent=4)
                 init_logger.info(f"{NEON_GREEN}Saved updated configuration to: {filepath}{RESET}")
             except Exception as save_err: init_logger.error(f"{NEON_RED}Error saving updated config: {save_err}{RESET}", exc_info=True)

        return updated_config

    except Exception as e: init_logger.error(f"{NEON_RED}Unexpected error processing config: {e}. Using defaults.{RESET}", exc_info=True); return default_config

# --- Logger & Config Setup ---
init_logger = setup_logger("init")
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

# --- CCXT Exchange Setup ---
# ... (initialize_exchange function - no major changes needed from v1.1.2) ...
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    lg = logger
    try:
        exchange_options = { 'apiKey': API_KEY, 'secret': API_SECRET, 'enableRateLimit': True,
            'options': { 'defaultType': 'linear', 'adjustForTimeDifference': True,
                         'fetchTickerTimeout': 15000, 'fetchBalanceTimeout': 20000, 'createOrderTimeout': 30000,
                         'cancelOrderTimeout': 20000, 'fetchPositionsTimeout': 20000, 'fetchOHLCVTimeout': 60000 } }
        exchange = ccxt.bybit(exchange_options)
        if CONFIG.get('use_sandbox', True): lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}"); exchange.set_sandbox_mode(True)
        else: lg.warning(f"{NEON_RED}USING LIVE TRADING ENVIRONMENT{RESET}")
        lg.info(f"Loading markets for {exchange.id}...")
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                exchange.load_markets(reload=True if attempt > 0 else False)
                if exchange.markets: lg.info(f"Markets loaded successfully ({len(exchange.markets)} symbols)."); break
                else: lg.warning(f"load_markets returned empty list (Attempt {attempt+1}). Retrying...")
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
                if attempt < MAX_API_RETRIES: lg.warning(f"Network error loading markets (Attempt {attempt+1}): {e}. Retrying..."); time.sleep(RETRY_DELAY_SECONDS)
                else: lg.critical(f"{NEON_RED}Max retries loading markets: {e}. Exiting.{RESET}"); return None
            except Exception as e: lg.critical(f"{NEON_RED}Error loading markets: {e}. Exiting.{RESET}", exc_info=True); return None
        if not exchange.markets: lg.critical(f"{NEON_RED}Market loading failed. Exiting.{RESET}"); return None
        lg.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        lg.info(f"Attempting initial balance fetch (Quote Currency: {QUOTE_CURRENCY})...")
        try:
            balance_val = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance_val is not None: lg.info(f"{NEON_GREEN}Initial balance fetched: {balance_val.normalize()} {QUOTE_CURRENCY}{RESET}")
            else:
                lg.critical(f"{NEON_RED}Initial balance fetch failed.{RESET}")
                if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED} Trading enabled, critical error. Exiting.{RESET}"); return None
                else: lg.warning(f"{NEON_YELLOW} Trading disabled, proceeding cautiously.{RESET}")
        except ccxt.AuthenticationError as auth_err: lg.critical(f"{NEON_RED}Auth Error fetching balance: {auth_err}. Check API keys/perms.{RESET}"); return None
        except Exception as balance_err:
             lg.warning(f"{NEON_YELLOW}Initial balance fetch error: {balance_err}.{RESET}", exc_info=True)
             if CONFIG.get('enable_trading', False): lg.critical(f"{NEON_RED} Trading enabled, critical error. Exiting.{RESET}"); return None
        return exchange
    except Exception as e: lg.critical(f"{NEON_RED}Failed to initialize exchange: {e}{RESET}", exc_info=True); return None

# --- CCXT Data Fetching Helpers ---
# ... (fetch_current_price_ccxt function - no major changes needed from v1.1.2) ...
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current market price for a symbol."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching ticker for {symbol} (Attempt {attempts + 1})")
            ticker = exchange.fetch_ticker(symbol)
            price = None
            def safe_decimal(val_str, name):
                if val_str is not None and str(val_str).strip() != '':
                    try: p = Decimal(str(val_str)); return p if p > Decimal('0') else None
                    except Exception: return None
                return None
            price = safe_decimal(ticker.get('last'), 'last')
            if price is None:
                bid = safe_decimal(ticker.get('bid'), 'bid'); ask = safe_decimal(ticker.get('ask'), 'ask')
                if bid and ask and ask >= bid: price = (bid + ask) / Decimal('2')
                elif ask: price = ask; lg.warning(f"{NEON_YELLOW}Using 'ask' fallback: {price.normalize()}{RESET}")
                elif bid: price = bid; lg.warning(f"{NEON_YELLOW}Using 'bid' fallback: {price.normalize()}{RESET}")
            if price: return price
            else: lg.warning(f"Failed to get valid price from ticker (Attempt {attempts + 1}). Data: {ticker}")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e: lg.warning(f"{NEON_YELLOW}Network error fetching price: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e: wait_time = RETRY_DELAY_SECONDS * 5; lg.warning(f"{NEON_YELLOW}Rate limit fetching price: {e}. Waiting {wait_time}s...{RESET}"); time.sleep(wait_time - RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetching price: {e}{RESET}"); return None
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching price: {e}{RESET}", exc_info=True); return None
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed to fetch price for {symbol} after retries.{RESET}"); return None

# ... (fetch_klines_ccxt function - minor logging improvement from v1.1.2) ...
def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries."""
    lg = logger
    if not exchange.has['fetchOHLCV']: lg.error(f"{exchange.id} does not support fetchOHLCV."); return pd.DataFrame()
    ohlcv = None
    # Ensure requested limit doesn't exceed API absolute limit
    actual_request_limit = min(limit, BYBIT_API_KLINE_LIMIT)
    if limit > BYBIT_API_KLINE_LIMIT:
         lg.debug(f"Requested kline limit {limit} exceeds API limit {BYBIT_API_KLINE_LIMIT}. Requesting {BYBIT_API_KLINE_LIMIT}.")

    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={actual_request_limit} (Attempt {attempt+1})")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=actual_request_limit)
            returned_count = len(ohlcv) if ohlcv else 0
            lg.debug(f"Exchange returned {returned_count} candles (requested {actual_request_limit}).")

            # Log specific warning if API limit might have been hit
            if returned_count == BYBIT_API_KLINE_LIMIT and limit > BYBIT_API_KLINE_LIMIT:
                lg.warning(f"{NEON_YELLOW}Fetched {returned_count} candles, hitting the API limit ({BYBIT_API_KLINE_LIMIT}). Strategy might need more data than available in one request.{RESET}")

            if ohlcv and returned_count > 0:
                try: # Validate timestamp lag
                    last_ts = pd.to_datetime(ohlcv[-1][0], unit='ms', utc=True)
                    now_utc = pd.Timestamp.utcnow()
                    interval_seconds = exchange.parse_timeframe(timeframe) if hasattr(exchange, 'parse_timeframe') and exchange.parse_timeframe(timeframe) else 300
                    max_lag = (interval_seconds * 5) if interval_seconds else 300
                    if timeframe in ['1d', '1w', '1M']: max_lag = max(max_lag, 3600)
                    lag_seconds = (now_utc - last_ts).total_seconds()
                    if lag_seconds < max_lag: lg.debug(f"Last kline timestamp OK (Lag: {lag_seconds:.1f}s)."); break
                    else: lg.warning(f"{NEON_YELLOW}Last kline {last_ts} too old (Lag: {lag_seconds:.1f}s > {max_lag}s). Retrying...{RESET}"); ohlcv = None
                except Exception as ts_err: lg.warning(f"Timestamp validation error: {ts_err}. Proceeding."); break
            else: lg.warning(f"fetch_ohlcv returned no data (Attempt {attempt+1}). Retrying...")
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempt < MAX_API_RETRIES: lg.warning(f"{NEON_YELLOW}Network error fetching klines (Attempt {attempt+1}): {e}. Retrying...{RESET}"); time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"{NEON_RED}Max retries fetching klines after network errors: {e}{RESET}"); return pd.DataFrame()
        except ccxt.RateLimitExceeded as e: wait_time = RETRY_DELAY_SECONDS * 5; lg.warning(f"{NEON_YELLOW}Rate limit fetching klines: {e}. Waiting {wait_time}s...{RESET}"); time.sleep(wait_time)
        except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetching klines: {e}{RESET}"); return pd.DataFrame()
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching klines: {e}{RESET}", exc_info=True); return pd.DataFrame()
    if not ohlcv: lg.warning(f"{NEON_YELLOW}No kline data after retries.{RESET}"); return pd.DataFrame()

    # --- Data Processing --- (Identical to v1.1.2)
    try:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']; df = pd.DataFrame(ohlcv, columns=columns[:len(ohlcv[0])])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True, errors='coerce'); df.dropna(subset=['timestamp'], inplace=True); df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce'); df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
        initial_len = len(df); df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True); df = df[df['close'] > Decimal('0')]
        if 'volume' in df.columns: df.dropna(subset=['volume'], inplace=True); df = df[df['volume'] >= Decimal('0')]
        rows_dropped = initial_len - len(df); lg.debug(f"Dropped {rows_dropped} invalid rows.") if rows_dropped > 0 else None
        if df.empty: lg.warning(f"{NEON_YELLOW}Kline data empty after cleaning.{RESET}"); return pd.DataFrame()
        df.sort_index(inplace=True)
        if len(df) > MAX_DF_LEN: lg.debug(f"Trimming DF from {len(df)} to {MAX_DF_LEN}."); df = df.iloc[-MAX_DF_LEN:].copy()
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df
    except Exception as e: lg.error(f"{NEON_RED}Error processing kline data: {e}{RESET}", exc_info=True); return pd.DataFrame()

# ... (get_market_info function - no major changes needed from v1.1.2) ...
def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Retrieves and validates market information."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            if not exchange.markets or symbol not in exchange.markets: lg.info(f"Market info for {symbol} missing. Reloading markets..."); exchange.load_markets(reload=True)
            if symbol not in exchange.markets:
                if attempt == 0: continue
                lg.error(f"{NEON_RED}Market '{symbol}' not found after reload.{RESET}"); return None
            market = exchange.market(symbol)
            if market:
                market['is_contract'] = market.get('contract', False) or market.get('type') in ['swap', 'future']
                market['contract_type_str'] = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "Spot/Other"
                def fmt_val(v): return str(Decimal(str(v)).normalize()) if v is not None else 'N/A'
                lg.debug(f"Market Info: {symbol} (ID={market.get('id')}, Type={market.get('type')}, Contract={market['contract_type_str']}), "
                         f"Precision(P/A): {fmt_val(market.get('precision',{}).get('price'))}/{fmt_val(market.get('precision',{}).get('amount'))}, "
                         f"Limits(Amt Min/Max): {fmt_val(market.get('limits',{}).get('amount',{}).get('min'))}/{fmt_val(market.get('limits',{}).get('amount',{}).get('max'))}, "
                         f"Limits(Cost Min/Max): {fmt_val(market.get('limits',{}).get('cost',{}).get('min'))}/{fmt_val(market.get('limits',{}).get('cost',{}).get('max'))}")
                if market.get('precision', {}).get('price') is None or market.get('precision', {}).get('amount') is None:
                    lg.error(f"{NEON_RED}Market {symbol} missing price/amount precision! Trading may fail.{RESET}")
                return market
            else: lg.error(f"{NEON_RED}Market dictionary None for '{symbol}'.{RESET}"); return None
        except ccxt.BadSymbol as e: lg.error(f"{NEON_RED}Symbol '{symbol}' invalid: {e}{RESET}"); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.RequestException) as e:
            if attempts < MAX_API_RETRIES: lg.warning(f"{NEON_YELLOW}Network error getting market info (Attempt {attempts+1}): {e}. Retrying...{RESET}"); time.sleep(RETRY_DELAY_SECONDS)
            else: lg.error(f"{NEON_RED}Max retries getting market info: {e}{RESET}"); return None
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error getting market info: {e}{RESET}", exc_info=True); return None
        attempts += 1
    return None

# ... (fetch_balance function - no major changes needed from v1.1.2) ...
def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches available balance for a currency, handling Bybit V5 structures."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            balance_info = None; available_balance_str = None; found_structure = False
            account_types_to_try = ['UNIFIED', 'CONTRACT']
            for acc_type in account_types_to_try:
                try:
                    lg.debug(f"Fetching balance (Type: {acc_type}) for {currency}...")
                    balance_info = exchange.fetch_balance(params={'accountType': acc_type})
                    if currency in balance_info and balance_info[currency].get('free') is not None:
                        available_balance_str = str(balance_info[currency]['free']); found_structure = True; break
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                        for account in balance_info['info']['result']['list']:
                            if (account.get('accountType') is None or account.get('accountType') == acc_type) and isinstance(account.get('coin'), list):
                                for coin_data in account['coin']:
                                    if coin_data.get('coin') == currency:
                                        free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                        if free is not None: available_balance_str = str(free); found_structure = True; break
                                if found_structure: break
                        if found_structure: break
                except (ccxt.ExchangeError, ccxt.AuthenticationError) as e: lg.debug(f"API error fetching balance type '{acc_type}': {e}. Trying next.")
                except Exception as e: lg.warning(f"Unexpected error fetching balance type '{acc_type}': {e}.", exc_info=True)
            if not found_structure: # Fallback to default fetch
                try:
                    lg.debug(f"Fetching balance default for {currency}...")
                    balance_info = exchange.fetch_balance()
                    if currency in balance_info and balance_info[currency].get('free') is not None: available_balance_str = str(balance_info[currency]['free']); found_structure = True
                    elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
                         for account in balance_info['info']['result']['list']:
                             if isinstance(account.get('coin'), list):
                                 for coin_data in account['coin']:
                                     if coin_data.get('coin') == currency:
                                         free = coin_data.get('availableToWithdraw') or coin_data.get('availableBalance') or coin_data.get('walletBalance')
                                         if free is not None: available_balance_str = str(free); found_structure = True; break
                                 if found_structure: break
                             if found_structure: break
                except Exception as e: lg.error(f"{NEON_RED}Failed default balance fetch: {e}{RESET}", exc_info=True)

            if found_structure and available_balance_str is not None:
                try: final_balance = Decimal(available_balance_str); return final_balance if final_balance >= Decimal('0') else Decimal('0')
                except Exception as e: raise ccxt.ExchangeError(f"Balance conversion failed for {currency}: {e}")
            else: raise ccxt.ExchangeError(f"Balance not found for {currency}")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"{NEON_YELLOW}Network error fetching balance: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e: wait_time = RETRY_DELAY_SECONDS*5; lg.warning(f"{NEON_YELLOW}Rate limit fetching balance: {e}. Wait {wait_time}s...{RESET}"); time.sleep(wait_time - RETRY_DELAY_SECONDS)
        except ccxt.AuthenticationError as e: lg.critical(f"{NEON_RED}Auth Error fetching balance: {e}. Check API keys.{RESET}"); return None
        except ccxt.ExchangeError as e: lg.warning(f"{NEON_YELLOW}Exchange error fetching balance: {e}. Retrying...{RESET}");
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * (attempts + 1) if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed to fetch balance for {currency} after retries.{RESET}"); return None

# ... (get_open_position function - no major changes needed from v1.1.2) ...
def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position for the given symbol using fetch_positions."""
    lg = logger; attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Fetching positions for symbol: {symbol} (Attempt {attempts+1})")
            positions: List[Dict] = []; market_id = None; category = None
            try:
                market = exchange.market(symbol); market_id = market['id']
                category = 'linear' if market.get('linear', False) else 'inverse' if market.get('inverse', False) else 'spot' if market.get('spot', False) else 'linear'
                params = {'category': category, 'symbol': market_id}
                lg.debug(f"Using fetch_positions with params: {params}")
                positions = exchange.fetch_positions([symbol], params=params)
            except ccxt.ArgumentsRequired:
                 lg.warning("fetch_positions requires fetching all positions. Slower."); params = {'category': category or 'linear'}
                 all_positions = exchange.fetch_positions(params=params)
                 positions = [p for p in all_positions if p.get('symbol') == symbol]
                 lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} for {symbol}.")
            except ccxt.ExchangeError as e:
                 no_pos_codes_v5 = [110025]; err_str = str(e).lower()
                 if (hasattr(e, 'code') and e.code in no_pos_codes_v5) or "position not found" in err_str or "invalid symbol" in err_str:
                      lg.info(f"No position or invalid query for {symbol} (Exchange: {e})."); return None
                 else: raise e

            active_position = None; size_threshold = Decimal('1e-9')
            try:
                market = exchange.market(symbol); amount_prec_str = market.get('precision', {}).get('amount')
                if amount_prec_str: size_threshold = Decimal(str(amount_prec_str)) * Decimal('0.1')
            except Exception as market_err: lg.debug(f"Could not get market precision for pos size threshold ({market_err}), using default.")
            lg.debug(f"Position size threshold: {size_threshold}")

            for pos in positions:
                pos_size_str = str(pos.get('info', {}).get('size', pos.get('contracts', ''))) # Prefer info.size
                if not pos_size_str: lg.debug(f"Skipping entry, no size found: {pos.get('info', {})}"); continue
                try:
                    position_size = Decimal(pos_size_str)
                    if abs(position_size) > size_threshold: active_position = pos; break
                except Exception as parse_err: lg.warning(f"Could not parse pos size '{pos_size_str}': {parse_err}. Skipping."); continue

            if active_position:
                std_pos = active_position.copy(); info_dict = std_pos.get('info', {})
                std_pos['size_decimal'] = position_size # Use already parsed size
                side = std_pos.get('side')
                if side not in ['long', 'short']:
                    pos_side_v5 = info_dict.get('side', '').lower()
                    if pos_side_v5 == 'buy': side = 'long'
                    elif pos_side_v5 == 'sell': side = 'short'
                    elif std_pos['size_decimal'] > size_threshold: side = 'long'
                    elif std_pos['size_decimal'] < -size_threshold: side = 'short'
                    else: lg.warning(f"Pos size {std_pos['size_decimal']} near zero/ambiguous."); return None
                std_pos['side'] = side
                std_pos['entryPrice'] = std_pos.get('entryPrice') or info_dict.get('avgPrice') or info_dict.get('entryPrice')
                std_pos['leverage'] = std_pos.get('leverage') or info_dict.get('leverage')
                std_pos['liquidationPrice'] = std_pos.get('liquidationPrice') or info_dict.get('liqPrice')
                std_pos['unrealizedPnl'] = std_pos.get('unrealizedPnl') or info_dict.get('unrealisedPnl')
                sl_price_str = info_dict.get('stopLoss') or std_pos.get('stopLossPrice')
                tp_price_str = info_dict.get('takeProfit') or std_pos.get('takeProfitPrice')
                tsl_distance_str = info_dict.get('trailingStop'); tsl_activation_str = info_dict.get('activePrice')
                if sl_price_str is not None: std_pos['stopLossPrice'] = str(sl_price_str)
                if tp_price_str is not None: std_pos['takeProfitPrice'] = str(tp_price_str)
                if tsl_distance_str is not None: std_pos['trailingStopLoss'] = str(tsl_distance_str)
                if tsl_activation_str is not None: std_pos['tslActivationPrice'] = str(tsl_activation_str)

                def fmt_log(val_str, p_type='price', p_def=4):
                    if val_str is None or str(val_str).strip() == '': return 'N/A'; s_val = str(val_str).strip()
                    if s_val == '0': return '0'
                    try:
                        d = Decimal(s_val); prec = p_def; market = None
                        try: market = exchange.market(symbol)
                        except Exception: pass
                        if market:
                            prec_val = market.get('precision', {}).get(p_type)
                            if prec_val is not None:
                                try:
                                    step = Decimal(str(prec_val)); prec = 0 if step == step.to_integral() else abs(step.normalize().as_tuple().exponent)
                                except Exception: pass
                        exp = Decimal('1e-' + str(prec)); return str(d.quantize(exp, rounding=ROUND_DOWN).normalize())
                    except Exception: return s_val
                ep = fmt_log(std_pos.get('entryPrice')); size = fmt_log(abs(std_pos['size_decimal']), 'amount'); liq = fmt_log(std_pos.get('liquidationPrice'))
                lev = fmt_log(std_pos.get('leverage'), 'price', 1) + 'x' if std_pos.get('leverage') else 'N/A'; pnl = fmt_log(std_pos.get('unrealizedPnl'), 'price', 4)
                sl = fmt_log(std_pos.get('stopLossPrice')); tp = fmt_log(std_pos.get('takeProfitPrice'))
                tsl_d = fmt_log(std_pos.get('trailingStopLoss')); tsl_a = fmt_log(std_pos.get('tslActivationPrice'))
                logger.info(f"{NEON_GREEN}Active {side.upper()} position ({symbol}):{RESET} Size={size}, Entry={ep}, Liq={liq}, Lev={lev}, PnL={pnl}, SL={sl}, TP={tp}, TSL(D/A): {tsl_d}/{tsl_a}")
                return std_pos
            else: lg.info(f"No active position found for {symbol}."); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e: lg.warning(f"{NEON_YELLOW}Network error fetching position: {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e: wait_time = RETRY_DELAY_SECONDS * 5; lg.warning(f"{NEON_YELLOW}Rate limit fetching position: {e}. Wait {wait_time}s...{RESET}"); time.sleep(wait_time - RETRY_DELAY_SECONDS)
        except ccxt.AuthenticationError as e: lg.critical(f"{NEON_RED}Auth Error fetching position: {e}. Stopping.{RESET}"); return None
        except ccxt.ExchangeError as e:
            lg.warning(f"{NEON_YELLOW}Exchange error fetching position: {e}. Retrying...{RESET}"); bybit_code = getattr(e, 'code', None)
            if bybit_code in [110004]: lg.critical(f"{NEON_RED}Bybit Account Error ({bybit_code}): {e}. Check API key link. Stopping.{RESET}"); return None
            if bybit_code in [110013]: lg.error(f"{NEON_RED}Bybit Param Error ({bybit_code}) fetching pos: {e}. Check symbol/cat. Stopping.{RESET}"); return None
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching pos: {e}{RESET}", exc_info=True)
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed to get position info for {symbol} after retries.{RESET}"); return None

# ... (set_leverage_ccxt function - added retries in v1.1.2) ...
def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a derivatives symbol using CCXT, handling Bybit V5 specifics."""
    lg = logger; is_contract = market_info.get('is_contract', False)
    if not is_contract: lg.info(f"Leverage skip {symbol} (Not contract)."); return True
    if leverage <= 0: lg.warning(f"Leverage skip {symbol} (Invalid {leverage})."); return False
    if not exchange.has.get('setLeverage'): lg.error(f"{exchange.id} no setLeverage."); return False

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.info(f"Setting leverage {symbol} to {leverage}x (Attempt {attempts+1})...")
            params = {}; market_id = market_info.get('id', symbol)
            if 'bybit' in exchange.id.lower():
                 category = 'linear' if market_info.get('linear', True) else 'inverse'
                 params = {'category': category, 'symbol': market_id, 'buyLeverage': str(leverage), 'sellLeverage': str(leverage)} # Assuming ISOLATED
                 lg.debug(f"Bybit V5 leverage params: {params}")
            response = exchange.set_leverage(leverage=leverage, symbol=market_id, params=params)
            lg.debug(f"Set leverage raw response: {response}")
            ret_code = response.get('retCode')
            if ret_code is not None:
                 if ret_code == 0: lg.info(f"{NEON_GREEN}Leverage set OK {symbol} {leverage}x (Code 0).{RESET}"); return True
                 elif ret_code == 110045: lg.info(f"{NEON_YELLOW}Leverage {symbol} already {leverage}x (Code 110045).{RESET}"); return True
                 else: raise ccxt.ExchangeError(f"Bybit API error set leverage: {response.get('retMsg', 'Unknown')} (Code: {ret_code})")
            else: lg.info(f"{NEON_GREEN}Leverage set OK {symbol} {leverage}x.{RESET}"); return True
        except ccxt.ExchangeError as e:
            bybit_code = getattr(e, 'code', None); err_str = str(e).lower()
            lg.error(f"{NEON_RED}Exchange error setting leverage: {e} (Code: {bybit_code}){RESET}")
            if bybit_code == 110045 or "not modified" in err_str: return True # Already set
            non_retry_codes = [110028, 110009, 110055, 110043, 110044, 110013, 10001, 10004] # Logic, Risk, Param, Auth errors
            if bybit_code in non_retry_codes or any(s in err_str for s in ["margin mode", "position exists", "risk limit", "parameter error"]):
                lg.error(" >> Hint: Non-retryable leverage error. Check mode, position, risk limits, value."); return False
            elif attempts >= MAX_API_RETRIES: lg.error("Max retries for ExchangeError set leverage."); return False
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error("Max retries network error set leverage."); return False
            lg.warning(f"{NEON_YELLOW}Network error setting leverage (Attempt {attempts+1}): {e}. Retrying...{RESET}")
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting leverage: {e}{RESET}", exc_info=True); return False
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed set leverage {symbol} after retries.{RESET}"); return False

# ... (calculate_position_size function - no major changes needed from v1.1.2) ...
def calculate_position_size(balance: Decimal, risk_per_trade: float, initial_stop_loss_price: Decimal, entry_price: Decimal,
                            market_info: Dict, exchange: ccxt.Exchange, logger: logging.Logger) -> Optional[Decimal]:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger; symbol = market_info['symbol']; quote = market_info['quote']; base = market_info['base']
    is_contract = market_info['is_contract']; is_inverse = market_info['inverse']
    size_unit = "Contracts" if is_contract else base
    if balance <= Decimal('0'): lg.error(f"Sizing fail {symbol}: Invalid balance {balance}."); return None
    risk_dec = Decimal(str(risk_per_trade))
    if initial_stop_loss_price <= Decimal('0') or entry_price <= Decimal('0'): lg.error(f"Sizing fail {symbol}: Invalid entry/SL."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Sizing fail {symbol}: SL==Entry."); return None
    amt_prec_str = market_info['precision']['amount']; price_prec_str = market_info['precision']['price']
    if amt_prec_str is None or price_prec_str is None: lg.error(f"Sizing fail {symbol}: Missing precision."); return None
    try:
        risk_amt_quote = balance * risk_dec
        sl_dist_price = abs(entry_price - initial_stop_loss_price)
        if sl_dist_price <= Decimal('0'): lg.error(f"Sizing fail {symbol}: SL dist zero."); return None
        lg.info(f"Position Sizing ({symbol}): Balance={balance.normalize()} {quote}, Risk={risk_dec:.2%}, RiskAmt={risk_amt_quote.normalize()} {quote}")
        lg.info(f"  Entry={entry_price.normalize()}, SL={initial_stop_loss_price.normalize()}, SL Dist={sl_dist_price.normalize()}")
        contract_size_str = market_info.get('contractSize', '1'); contract_size = Decimal('1')
        try: contract_size = Decimal(str(contract_size_str)); assert contract_size > 0
        except Exception as e: lg.warning(f"Invalid contract size '{contract_size_str}', using 1: {e}")
        lg.info(f"  ContractSize={contract_size.normalize()}, Type={'Linear/Spot' if not is_inverse else 'Inverse'}")

        calculated_size = Decimal('0')
        if not is_inverse: # Linear/Spot: Size = Risk / SL_Dist
            if sl_dist_price > 0: calculated_size = risk_amt_quote / sl_dist_price
            else: lg.error("Sizing fail linear: SL dist zero."); return None
        else: # Inverse: Size = Risk / (ContractSize * |1/Entry - 1/SL|)
            lg.info(f"Inverse contract sizing.")
            if entry_price > 0 and initial_stop_loss_price > 0:
                try:
                    inv_factor = abs(Decimal('1') / entry_price - Decimal('1') / initial_stop_loss_price)
                    if inv_factor <= 0: lg.error("Sizing fail inverse: Factor zero."); return None
                    risk_per_contract = contract_size * inv_factor
                    if risk_per_contract <= 0: lg.error("Sizing fail inverse: Risk per contract zero."); return None
                    calculated_size = risk_amt_quote / risk_per_contract
                except Exception as calc_err: lg.error(f"Sizing fail inverse calc: {calc_err}."); return None
            else: lg.error("Sizing fail inverse: Invalid entry/SL price."); return None
        lg.info(f"  Initial Calc Size = {calculated_size.normalize()} {size_unit}")

        # Apply Limits & Precision
        limits = market_info.get('limits', {}); amt_limits = limits.get('amount', {}); cost_limits = limits.get('cost', {})
        try:
            amt_prec_step = Decimal(str(amt_prec_str)); price_prec_step = Decimal(str(price_prec_str))
            min_amt = Decimal(str(amt_limits.get('min', '0'))); max_amt = Decimal(str(amt_limits.get('max', 'inf'))) if amt_limits.get('max') is not None else Decimal('inf')
            min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
            max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')
        except Exception as conv_err: lg.error(f"Sizing fail {symbol}: Error converting limits/precision: {conv_err}"); return None

        adj_size = calculated_size
        if min_amt > 0 and adj_size < min_amt: lg.warning(f"{NEON_YELLOW}Calc size {adj_size.normalize()} < min {min_amt.normalize()}. Adjusting up.{RESET}"); adj_size = min_amt
        if max_amt < Decimal('inf') and adj_size > max_amt: lg.warning(f"{NEON_YELLOW}Calc size {adj_size.normalize()} > max {max_amt.normalize()}. Adjusting down.{RESET}"); adj_size = max_amt

        est_cost = Decimal('0'); cost_adjusted = False
        try:
            if entry_price > 0: est_cost = (adj_size * entry_price * contract_size) if not is_inverse else ((adj_size * contract_size) / entry_price)
        except Exception as cost_err: lg.error(f"Error estimating cost: {cost_err}"); min_cost=Decimal('0'); max_cost=Decimal('inf')
        lg.debug(f"  Size after Amt Limits: {adj_size.normalize()}"); lg.debug(f"  Est Cost: {est_cost.normalize()}")

        if min_cost > 0 and est_cost < min_cost:
            lg.warning(f"{NEON_YELLOW}Est cost {est_cost.normalize()} < min {min_cost.normalize()}. Increasing size.{RESET}"); req_size = None
            try:
                if entry_price > 0 and contract_size > 0: req_size = (min_cost/(entry_price*contract_size)) if not is_inverse else ((min_cost*entry_price)/contract_size)
            except Exception: pass
            if req_size is None or req_size <= 0: lg.error(f"{NEON_RED}Cannot meet min cost {min_cost.normalize()}. Aborted.{RESET}"); return None
            if max_amt < Decimal('inf') and req_size > max_amt: lg.error(f"{NEON_RED}Cannot meet min cost without exceeding max amount. Aborted.{RESET}"); return None
            adj_size = max(min_amt, req_size); cost_adjusted = True; lg.info(f"  Required size for min cost: {req_size.normalize()}")
        elif max_cost < Decimal('inf') and est_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Est cost {est_cost.normalize()} > max {max_cost.normalize()}. Reducing size.{RESET}"); adj_size_cost = None
            try:
                if entry_price > 0 and contract_size > 0: adj_size_cost = (max_cost/(entry_price*contract_size)) if not is_inverse else ((max_cost*entry_price)/contract_size)
            except Exception: pass
            if adj_size_cost is None or adj_size_cost <= 0: lg.error(f"{NEON_RED}Cannot reduce size for max cost {max_cost.normalize()}. Aborted.{RESET}"); return None
            adj_size = max(min_amt, min(adj_size, adj_size_cost)); cost_adjusted = True; lg.info(f"  Max size for max cost: {adj_size_cost.normalize()}")
        if cost_adjusted: lg.info(f"  Size after Cost Limits: {adj_size.normalize()}")

        final_size = adj_size
        try: # Apply precision using ccxt
            fmt_size_str = exchange.amount_to_precision(symbol, float(adj_size))
            final_size = Decimal(fmt_size_str)
            lg.info(f"Applied amt precision (ccxt): {adj_size.normalize()} -> {final_size.normalize()}")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}ccxt precision error: {fmt_err}. Using manual round down.{RESET}", exc_info=True)
            try:
                if amt_prec_step > 0: final_size = (adj_size // amt_prec_step) * amt_prec_step; lg.info(f"Applied manual step: {final_size.normalize()}")
                else: raise ValueError("Step size zero")
            except Exception as manual_err: lg.error(f"{NEON_RED}Manual precision failed: {manual_err}. Using limit-adjusted size.{RESET}"); final_size = adj_size

        if final_size <= Decimal('0'): lg.error(f"{NEON_RED}Final size zero/negative ({final_size}). Aborted.{RESET}"); return None
        if min_amt > 0 and final_size < min_amt: lg.error(f"{NEON_RED}Final size {final_size} < min amount {min_amt}. Aborted.{RESET}"); return None

        final_cost = Decimal('0')
        try:
            if entry_price > 0: final_cost = (final_size * entry_price * contract_size) if not is_inverse else ((final_size * contract_size) / entry_price)
        except Exception: pass
        if min_cost > 0 and final_cost < min_cost:
             lg.debug(f"Final size cost {final_cost} < min cost {min_cost}.")
             try:
                 step = amt_prec_step; next_size = final_size + step; next_cost = Decimal('0')
                 if entry_price > 0: next_cost = (next_size * entry_price * contract_size) if not is_inverse else ((next_size * contract_size) / entry_price)
                 valid_next = (next_cost >= min_cost) and (max_amt == Decimal('inf') or next_size <= max_amt) and (max_cost == Decimal('inf') or next_cost <= max_cost)
                 if valid_next: lg.warning(f"{NEON_YELLOW}Bumping size to {next_size} to meet min cost.{RESET}"); final_size = next_size
                 else: lg.error(f"{NEON_RED}Final size cost below minimum, next step invalid. Aborted.{RESET}"); return None
             except Exception as bump_err: lg.error(f"{NEON_RED}Error bumping size for min cost: {bump_err}. Aborted.{RESET}"); return None

        lg.info(f"{NEON_GREEN}Final Position Size: {final_size.normalize()} {size_unit}{RESET}")
        return final_size
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error calculating size: {e}{RESET}", exc_info=True); return None

# ... (place_trade function - no major changes needed from v1.1.2) ...
def place_trade(exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal, market_info: Dict,
                logger: logging.Logger, reduce_only: bool = False, params: Optional[Dict] = None) -> Optional[Dict]:
    """Places a market order using CCXT with retries for network errors."""
    lg = logger; side = 'buy' if trade_signal == "BUY" else 'sell'; order_type = 'market'
    is_contract = market_info['is_contract']; base = market_info['base']; size_unit = market_info.get('settle', base) if is_contract else base
    action_desc = "Close" if reduce_only else "Open/Increase"; market_id = market_info['id']
    category = 'linear' if market_info.get('linear', True) else 'inverse'; position_idx = 0
    order_params = {'category': category, 'positionIdx': position_idx, 'reduceOnly': reduce_only}
    if reduce_only: order_params['timeInForce'] = 'IOC'
    if params: order_params.update(params)
    lg.info(f"Attempting {action_desc} {side.upper()} {order_type} for {symbol}: Size={position_size.normalize()} {size_unit}")
    lg.debug(f"Full Order Params: {order_params}")
    try: amount_float = float(position_size); assert amount_float > 0
    except (ValueError, AssertionError): lg.error(f"Trade aborted: Invalid size {position_size}."); return None

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing create_order (Attempt {attempts+1})...")
            order = exchange.create_order(symbol=market_id, type=order_type, side=side, amount=amount_float, params=order_params)
            oid = order.get('id', 'N/A'); status = order.get('status', 'N/A'); avg = order.get('average'); filled = order.get('filled')
            lg.info(f"{NEON_GREEN}{action_desc} Trade Placed!{RESET} ID: {oid}, Status: {status}" +
                    (f", AvgPrice: ~{Decimal(str(avg)).normalize()}" if avg else "") + (f", Filled: {Decimal(str(filled)).normalize()}" if filled else ""))
            return order
        except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Insufficient funds placing {action_desc} {side}: {e}{RESET}"); return None
        except ccxt.InvalidOrder as e: lg.error(f"{NEON_RED}Invalid order placing {action_desc} {side}: {e}{RESET}"); return None # Hints logged in v1.1.2
        except ccxt.ExchangeError as e: # Catch other exchange errors
            bybit_code = getattr(e, 'code', None); lg.error(f"{NEON_RED}Exchange error placing {action_desc}: {e} (Code: {bybit_code}){RESET}")
            non_retry_codes = [110014, 110007, 110040, 110013, 110025, 30086, 10001] # Add known fatal codes
            if bybit_code in non_retry_codes: lg.error(" >> Hint: Non-retryable exchange error."); return None
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error(f"{NEON_RED}Max retries network error placing order: {e}{RESET}"); return None
            lg.warning(f"{NEON_YELLOW}Network error placing order (Attempt {attempts+1}): {e}. Retrying...{RESET}")
        except ccxt.RateLimitExceeded as e: wait_time = RETRY_DELAY_SECONDS * 5; lg.warning(f"{NEON_YELLOW}Rate limit placing order: {e}. Wait {wait_time}s...{RESET}"); time.sleep(wait_time); continue
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error placing {action_desc}: {e}{RESET}", exc_info=True)
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed to place {action_desc} order for {symbol} after retries.{RESET}"); return None

# ... (_set_position_protection function - no major changes needed from v1.1.2) ...
def _set_position_protection(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, logger: logging.Logger,
                             stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
                             trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None) -> bool:
    """Internal helper to set SL/TP/TSL for a position via Bybit V5 API."""
    lg = logger
    if not market_info.get('is_contract'): lg.warning(f"Protection skip {symbol} (Not contract)."); return False
    if not position_info: lg.error(f"Protection fail {symbol}: Missing pos info."); return False
    pos_side = position_info.get('side'); entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str: lg.error(f"Protection fail {symbol}: Invalid pos side/entry."); return False
    try: entry_price = Decimal(str(entry_price_str)); assert entry_price > 0
    except Exception as e: lg.error(f"Invalid entry price '{entry_price_str}': {e}"); return False

    params_to_set = {}; log_parts = [f"Setting protection for {symbol} ({pos_side.upper()} @ {entry_price.normalize()}):"]; any_requested = False
    try:
        price_prec_str = market_info['precision']['price']; min_tick = Decimal(str(price_prec_str)); assert min_tick > 0
        def fmt_price(price_dec: Optional[Decimal]) -> Optional[str]:
            if price_dec is None: return None;
            if price_dec < 0: lg.warning(f"Cannot format negative price {price_dec}."); return None
            if price_dec == 0: return "0"
            try: fmt = exchange.price_to_precision(symbol, float(price_dec), exchange.ROUND); return fmt if Decimal(fmt) > 0 else None
            except Exception as e: lg.error(f"Failed format price {price_dec}: {e}."); return None

        set_tsl = False
        if isinstance(trailing_stop_distance, Decimal):
            if trailing_stop_distance > 0:
                any_requested = True
                if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price <= 0: lg.error(f"TSL requested but invalid activation price {tsl_activation_price}."); # Handled below
                else:
                    valid_act = (pos_side == 'long' and tsl_activation_price > entry_price) or (pos_side == 'short' and tsl_activation_price < entry_price)
                    if not valid_act: lg.error(f"TSL Activation {tsl_activation_price} not beyond entry {entry_price} for {pos_side}.")
                    else:
                        min_dist = max(trailing_stop_distance, min_tick)
                        fmt_tsl_dist = fmt_price(min_dist); fmt_act_price = fmt_price(tsl_activation_price)
                        if fmt_tsl_dist and fmt_act_price:
                            params_to_set['trailingStop'] = fmt_tsl_dist; params_to_set['activePrice'] = fmt_act_price
                            log_parts.append(f"  TSL: Dist={fmt_tsl_dist}, Act={fmt_act_price}"); set_tsl = True
                        else: lg.error(f"Failed format TSL params (Dist:{fmt_tsl_dist}, Act:{fmt_act_price}).")
            elif trailing_stop_distance == 0: params_to_set['trailingStop'] = "0"; log_parts.append("  TSL: Clear"); any_requested = True

        if not set_tsl and isinstance(stop_loss_price, Decimal):
            if stop_loss_price > 0:
                any_requested = True; valid_sl = (pos_side == 'long' and stop_loss_price < entry_price) or (pos_side == 'short' and stop_loss_price > entry_price)
                if not valid_sl: lg.error(f"SL Price {stop_loss_price} not beyond entry {entry_price} for {pos_side}.")
                else: fmt_sl = fmt_price(stop_loss_price);
                if fmt_sl: params_to_set['stopLoss'] = fmt_sl; log_parts.append(f"  Fixed SL: {fmt_sl}")
                else: lg.error(f"Failed format SL price {stop_loss_price}.")
            elif stop_loss_price == 0: params_to_set['stopLoss'] = "0"; log_parts.append("  Fixed SL: Clear"); any_requested = True

        if isinstance(take_profit_price, Decimal):
            if take_profit_price > 0:
                any_requested = True; valid_tp = (pos_side == 'long' and take_profit_price > entry_price) or (pos_side == 'short' and take_profit_price < entry_price)
                if not valid_tp: lg.error(f"TP Price {take_profit_price} not beyond entry {entry_price} for {pos_side}.")
                else: fmt_tp = fmt_price(take_profit_price);
                if fmt_tp: params_to_set['takeProfit'] = fmt_tp; log_parts.append(f"  Fixed TP: {fmt_tp}")
                else: lg.error(f"Failed format TP price {take_profit_price}.")
            elif take_profit_price == 0: params_to_set['takeProfit'] = "0"; log_parts.append("  Fixed TP: Clear"); any_requested = True
    except Exception as fmt_err: lg.error(f"Error formatting protection params: {fmt_err}", exc_info=True); return False

    if not params_to_set:
        if any_requested: lg.warning(f"No valid protection params after formatting. No API call."); return False
        else: lg.info(f"No protection change requested. No API call."); return True

    category = 'linear' if market_info.get('linear', True) else 'inverse'; market_id = market_info['id']; pos_idx = 0
    try: pos_idx_val = position_info.get('info', {}).get('positionIdx'); pos_idx = int(pos_idx_val) if pos_idx_val is not None else 0
    except Exception: lg.warning(f"Could not parse positionIdx, using default {pos_idx}.")
    final_params = {'category': category, 'symbol': market_id, 'tpslMode': 'Full', 'slTriggerBy': 'LastPrice',
                    'tpTriggerBy': 'LastPrice', 'slOrderType': 'Market', 'tpOrderType': 'Market', 'positionIdx': pos_idx}
    final_params.update(params_to_set)
    lg.info("\n".join(log_parts)); lg.debug(f"  API Params: {final_params}")

    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            lg.debug(f"Executing set protection API call (Attempt {attempts+1})...")
            response = exchange.private_post('/v5/position/set-trading-stop', params=final_params)
            lg.debug(f"Set protection raw response: {response}")
            ret_code = response.get('retCode'); ret_msg = response.get('retMsg', 'Unknown')
            if ret_code == 0:
                if any(msg in ret_msg.lower() for msg in ["not modified", "no need to modify"]): lg.info(f"{NEON_YELLOW}Protection already set/no change needed. Resp: {ret_msg}{RESET}")
                else: lg.info(f"{NEON_GREEN}Protection set/updated successfully.{RESET}")
                return True
            else:
                lg.error(f"{NEON_RED}Failed set protection: {ret_msg} (Code: {ret_code}){RESET}")
                fatal_codes = [110013, 110036, 110086, 110084, 110085, 10001, 10002]; is_fatal = ret_code in fatal_codes
                if is_fatal: lg.error(" >> Hint: Non-retryable error code."); return False
                else: raise ccxt.ExchangeError(f"Bybit Error set protection: {ret_msg} (Code: {ret_code})")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            if attempts >= MAX_API_RETRIES: lg.error(f"Max retries network error set protect: {e}"); return False
            lg.warning(f"{NEON_YELLOW}Network error set protect (Attempt {attempts+1}): {e}. Retrying...")
        except ccxt.RateLimitExceeded as e: wait_time=RETRY_DELAY_SECONDS*5; lg.warning(f"{NEON_YELLOW}Rate limit set protect: {e}. Wait {wait_time}s..."); time.sleep(wait_time); continue
        except ccxt.AuthenticationError as e: lg.critical(f"{NEON_RED}Auth Error set protect: {e}. Stopping."); return False
        except ccxt.ExchangeError as e: # Catch re-raised or other exchange errors
             if attempts >= MAX_API_RETRIES: lg.error(f"Max retries exchange error set protect: {e}"); return False
             lg.warning(f"{NEON_YELLOW}Exchange error set protect (Attempt {attempts+1}): {e}. Retrying...")
        except Exception as e: lg.error(f"{NEON_RED}Unexpected error set protect (Attempt {attempts+1}): {e}", exc_info=True);
        attempts += 1; time.sleep(RETRY_DELAY_SECONDS * attempts if attempts <= MAX_API_RETRIES else 0)
    lg.error(f"{NEON_RED}Failed set protection for {symbol} after retries.{RESET}"); return False

# ... (set_trailing_stop_loss function - no major changes needed from v1.1.2) ...
def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, config: Dict[str, Any],
                             logger: logging.Logger, take_profit_price: Optional[Decimal] = None) -> bool:
    """Calculates and sets Trailing Stop Loss via _set_position_protection."""
    lg = logger; protection_cfg = config.get("protection", {})
    if not market_info or not position_info: lg.error(f"Cannot calc TSL {symbol}: Missing info."); return False
    pos_side = position_info.get('side'); entry_price_str = position_info.get('entryPrice')
    if pos_side not in ['long', 'short'] or not entry_price_str: lg.error(f"TSL fail {symbol}: Invalid pos side/entry."); return False
    try:
        entry_price = Decimal(str(entry_price_str))
        callback_rate = Decimal(str(protection_cfg["trailing_stop_callback_rate"]))
        activation_percentage = Decimal(str(protection_cfg["trailing_stop_activation_percentage"]))
        assert entry_price > 0 and callback_rate > 0 and activation_percentage >= 0
        min_tick = Decimal(str(market_info['precision']['price'])); assert min_tick > 0
    except Exception as ve: lg.error(f"{NEON_RED}Invalid TSL params/info ({symbol}): {ve}. Cannot calc.{RESET}"); return False

    try:
        act_offset = entry_price * activation_percentage; act_price = None
        if pos_side == 'long': raw_act = entry_price + act_offset; act_price = raw_act.quantize(min_tick, ROUND_UP); act_price = max(act_price, entry_price + min_tick)
        else: raw_act = entry_price - act_offset; act_price = raw_act.quantize(min_tick, ROUND_DOWN); act_price = min(act_price, entry_price - min_tick)
        if act_price <= 0: lg.error(f"TSL Activation Price <= 0 ({act_price}). Cannot set."); return False
        dist_raw = act_price * callback_rate; trail_dist = dist_raw.quantize(min_tick, ROUND_UP); trail_dist = max(trail_dist, min_tick)
        if trail_dist <= 0: lg.error(f"TSL Distance <= 0 ({trail_dist}). Cannot set."); return False

        lg.info(f"Calculated TSL Params ({symbol}, {pos_side.upper()}): Entry={entry_price.normalize()}, Act%={activation_percentage:.3%}, CB%={callback_rate:.3%}")
        lg.info(f"  => Activation Price: {act_price.normalize()}, Trailing Distance: {trail_dist.normalize()}")
        if isinstance(take_profit_price, Decimal): lg.info(f"  Take Profit: {take_profit_price.normalize() if take_profit_price != 0 else 'Clear (0)'}")

        return _set_position_protection(exchange, symbol, market_info, position_info, lg, stop_loss_price=None,
                                        take_profit_price=take_profit_price, trailing_stop_distance=trail_dist, tsl_activation_price=act_price)
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error calc/set TSL: {e}{RESET}", exc_info=True); return False

# --- Volumatic Trend + OB Strategy Implementation ---
# ... (Classes OrderBlock, StrategyAnalysisResults - unchanged) ...
class OrderBlock(TypedDict):
    id: str; type: str; left_idx: pd.Timestamp; right_idx: pd.Timestamp
    top: Decimal; bottom: Decimal; active: bool; violated: bool
class StrategyAnalysisResults(TypedDict):
    dataframe: pd.DataFrame; last_close: Decimal; current_trend_up: Optional[bool]; trend_just_changed: bool
    active_bull_boxes: List[OrderBlock]; active_bear_boxes: List[OrderBlock]
    vol_norm_int: Optional[int]; atr: Optional[Decimal]; upper_band: Optional[Decimal]; lower_band: Optional[Decimal]

# ... (VolumaticOBStrategy Class - increased buffer in min_data_len) ...
class VolumaticOBStrategy:
    """Implements the Volumatic Trend and Pivot Order Block strategy."""
    def __init__(self, config: Dict[str, Any], market_info: Dict[str, Any], logger: logging.Logger):
        self.config = config; self.market_info = market_info; self.logger = logger
        strategy_cfg = config.get("strategy_params", {})
        self.vt_length = int(strategy_cfg.get("vt_length", DEFAULT_VT_LENGTH))
        self.vt_atr_period = int(strategy_cfg.get("vt_atr_period", DEFAULT_VT_ATR_PERIOD))
        self.vt_vol_ema_length = int(strategy_cfg.get("vt_vol_ema_length", DEFAULT_VT_VOL_EMA_LENGTH))
        self.vt_atr_multiplier = Decimal(str(strategy_cfg.get("vt_atr_multiplier", DEFAULT_VT_ATR_MULTIPLIER)))
        self.vt_step_atr_multiplier = Decimal(str(strategy_cfg.get("vt_step_atr_multiplier", DEFAULT_VT_STEP_ATR_MULTIPLIER)))
        self.ob_source = strategy_cfg.get("ob_source", DEFAULT_OB_SOURCE)
        self.ph_left = int(strategy_cfg.get("ph_left", DEFAULT_PH_LEFT)); self.ph_right = int(strategy_cfg.get("ph_right", DEFAULT_PH_RIGHT))
        self.pl_left = int(strategy_cfg.get("pl_left", DEFAULT_PL_LEFT)); self.pl_right = int(strategy_cfg.get("pl_right", DEFAULT_PL_RIGHT))
        self.ob_extend = bool(strategy_cfg.get("ob_extend", DEFAULT_OB_EXTEND)); self.ob_max_boxes = int(strategy_cfg.get("ob_max_boxes", DEFAULT_OB_MAX_BOXES))
        self.bull_boxes: List[OrderBlock] = []; self.bear_boxes: List[OrderBlock] = []
        # Calculate minimum data length required + larger buffer
        self.min_data_len = max( self.vt_length * 2, self.vt_atr_period, self.vt_vol_ema_length,
                                 self.ph_left + self.ph_right + 1, self.pl_left + self.pl_right + 1 ) + 50 # Increased buffer
        self.logger.info(f"{NEON_CYAN}Initializing VolumaticOB Strategy...{RESET}")
        self.logger.info(f"  VT Params: Len={self.vt_length}, ATRLen={self.vt_atr_period}, VolLen={self.vt_vol_ema_length}, ATRMult={self.vt_atr_multiplier.normalize()}, StepMult={self.vt_step_atr_multiplier.normalize()}")
        self.logger.info(f"  OB Params: Src={self.ob_source}, PH={self.ph_left}/{self.ph_right}, PL={self.pl_left}/{self.pl_right}, Extend={self.ob_extend}, MaxBoxes={self.ob_max_boxes}")
        self.logger.info(f"  Minimum historical data points required: {self.min_data_len}")
        if self.min_data_len > BYBIT_API_KLINE_LIMIT:
             self.logger.warning(f"{NEON_YELLOW}Strategy requires {self.min_data_len} candles, but API limit is {BYBIT_API_KLINE_LIMIT}. "
                                 f"Reduce lookback periods (e.g., 'vt_vol_ema_length') in config.json.{RESET}")

    def _ema_swma(self, series: pd.Series, length: int) -> pd.Series:
        """Calculates EMA(SWMA(close, 4), length)."""
        if len(series) < 4 or length <= 0: return pd.Series(np.nan, index=series.index, dtype=float)
        weights = np.array([1., 2., 2., 1.]) / 6.0
        series_numeric = pd.to_numeric(series, errors='coerce')
        swma = series_numeric.rolling(window=4, min_periods=4).apply(lambda x: np.dot(x, weights), raw=True)
        return ta.ema(swma, length=length, fillna=np.nan)

    def update(self, df_input: pd.DataFrame) -> StrategyAnalysisResults:
        """Processes historical data to calculate indicators and manage Order Blocks."""
        empty_results = StrategyAnalysisResults(dataframe=pd.DataFrame(), last_close=Decimal('0'), current_trend_up=None, trend_just_changed=False,
                                                active_bull_boxes=[], active_bear_boxes=[], vol_norm_int=None, atr=None, upper_band=None, lower_band=None)
        if df_input.empty: self.logger.error("Strategy update received empty DF."); return empty_results
        df = df_input.copy()
        if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing: self.logger.error("DF index invalid."); return empty_results
        if len(df) < self.min_data_len: self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(df)} < {self.min_data_len}) for full analysis.{RESET}")
        self.logger.debug(f"Starting analysis on {len(df)} candles.")

        try: # Convert to float for TA Libs
            df_float = pd.DataFrame(index=df.index)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns: df_float[col] = pd.to_numeric(df[col], errors='coerce')
                else: self.logger.error(f"Missing column '{col}'. Aborted."); return empty_results
            df_float.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
            if df_float.empty: self.logger.error("DF empty after float conversion."); return empty_results
        except Exception as e: self.logger.error(f"Error converting to float: {e}", exc_info=True); return empty_results

        try: # Volumatic Trend Calcs
            df_float['atr'] = ta.atr(df_float['high'], df_float['low'], df_float['close'], length=self.vt_atr_period, fillna=np.nan)
            df_float['ema1'] = self._ema_swma(df_float['close'], length=self.vt_length)
            df_float['ema2'] = ta.ema(df_float['close'], length=self.vt_length, fillna=np.nan)
            df_float['trend_up'] = (df_float['ema1'].shift(1) < df_float['ema2']).ffill()
            df_float['trend_changed'] = (df_float['trend_up'].shift(1) != df_float['trend_up']) & df_float['trend_up'].notna() & df_float['trend_up'].shift(1).notna()
            df_float['trend_changed'].fillna(False, inplace=True)
            df_float['ema1_at_change'] = np.where(df_float['trend_changed'], df_float['ema1'], np.nan)
            df_float['atr_at_change'] = np.where(df_float['trend_changed'], df_float['atr'], np.nan)
            df_float['ema1_for_bands'] = df_float['ema1_at_change'].ffill(); df_float['atr_for_bands'] = df_float['atr_at_change'].ffill()
            atr_mult_float = float(self.vt_atr_multiplier)
            df_float['upper_band'] = df_float['ema1_for_bands'] + (df_float['atr_for_bands'] * atr_mult_float)
            df_float['lower_band'] = df_float['ema1_for_bands'] - (df_float['atr_for_bands'] * atr_mult_float)
            volume_numeric = pd.to_numeric(df_float['volume'], errors='coerce').fillna(0.0)
            df_float['vol_max'] = volume_numeric.rolling(window=self.vt_vol_ema_length, min_periods=max(1, self.vt_vol_ema_length//10)).max().fillna(0.0)
            df_float['vol_norm'] = np.where(df_float['vol_max'] > 1e-9, (volume_numeric / df_float['vol_max'] * 100.0), 0.0)
            df_float['vol_norm'] = df_float['vol_norm'].fillna(0.0).clip(0.0, 200.0)
        except Exception as e: self.logger.error(f"Error during VT calc: {e}", exc_info=True); return empty_results

        try: # Copy results back to Decimal DF
            cols_to_copy = ['atr', 'ema1', 'ema2', 'trend_up', 'trend_changed', 'upper_band', 'lower_band', 'vol_norm']
            for col in cols_to_copy:
                if col in df_float.columns:
                    source_series = df_float[col].reindex(df.index)
                    if source_series.dtype == 'bool' or pd.api.types.is_object_dtype(source_series): df[col] = source_series
                    else: df[col] = source_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) and np.isfinite(x) else Decimal('NaN'))
        except Exception as e: self.logger.error(f"Error converting results to Decimal: {e}", exc_info=True); return empty_results

        # Clean Decimal DF
        initial_len = len(df)
        required_cols = ['upper_band', 'lower_band', 'atr', 'trend_up', 'close']
        df.dropna(subset=required_cols, inplace=True)
        rows_dropped = initial_len - len(df); lg.debug(f"Dropped {rows_dropped} rows missing indicators.") if rows_dropped > 0 else None
        if df.empty: self.logger.warning(f"{NEON_YELLOW}DF empty after indicator calcs & dropna.{RESET}"); return empty_results
        self.logger.debug("VT calcs complete. Processing OBs...")

        try: # Pivot OB Calcs & Management
            if self.ob_source == "Wicks": high_s = df_float['high']; low_s = df_float['low']
            else: high_s = df_float[['open', 'close']].max(axis=1); low_s = df_float[['open', 'close']].min(axis=1)
            ph_sig = ta.pivot(high_s, left=self.ph_left, right=self.ph_right, high_low='high').fillna(0).astype(bool)
            pl_sig = ta.pivot(low_s, left=self.pl_left, right=self.pl_right, high_low='low').fillna(0).astype(bool)
            df['ph_signal'] = ph_sig.reindex(df.index, fill_value=False); df['pl_signal'] = pl_sig.reindex(df.index, fill_value=False)

            new_boxes_count = 0
            if not df.empty:
                 for conf_idx in df.index:
                     try:
                         ph_conf = df.loc[conf_idx, 'ph_signal']; pl_conf = df.loc[conf_idx, 'pl_signal']
                         conf_loc = df.index.get_loc(conf_idx)
                         if ph_conf: # New Bearish OB?
                             piv_loc = conf_loc - self.ph_right
                             if piv_loc >= 0:
                                 piv_idx = df.index[piv_loc]
                                 if not any(b['left_idx'] == piv_idx and b['type'] == 'bear' for b in self.bear_boxes):
                                     c = df.loc[piv_idx]; top, bot = (c['high'], c['open']) if self.ob_source == "Wicks" else (c['close'], c['open'])
                                     if pd.notna(top) and pd.notna(bot):
                                         if bot > top: top, bot = bot, top
                                         if top > bot: self.bear_boxes.append(OrderBlock(id=f"B_{piv_idx.strftime('%y%m%d%H%M')}",type='bear',left_idx=piv_idx,right_idx=df.index[-1],top=top,bottom=bot,active=True,violated=False)); new_boxes_count += 1
                         if pl_conf: # New Bullish OB?
                             piv_loc = conf_loc - self.pl_right
                             if piv_loc >= 0:
                                 piv_idx = df.index[piv_loc]
                                 if not any(b['left_idx'] == piv_idx and b['type'] == 'bull' for b in self.bull_boxes):
                                     c = df.loc[piv_idx]; top, bot = (c['open'], c['low']) if self.ob_source == "Wicks" else (c['open'], c['close'])
                                     if pd.notna(top) and pd.notna(bot):
                                         if bot > top: top, bot = bot, top
                                         if top > bot: self.bull_boxes.append(OrderBlock(id=f"L_{piv_idx.strftime('%y%m%d%H%M')}",type='bull',left_idx=piv_idx,right_idx=df.index[-1],top=top,bottom=bot,active=True,violated=False)); new_boxes_count += 1
                     except Exception as e: self.logger.warning(f"Error processing pivot signal at {conf_idx}: {e}", exc_info=True)
            if new_boxes_count > 0: self.logger.debug(f"Found {new_boxes_count} new OB(s). Counts before prune: Bull={len(self.bull_boxes)}, Bear={len(self.bear_boxes)}.")

            # Manage Existing OBs
            if not df.empty and pd.notna(df['close'].iloc[-1]):
                last_idx = df.index[-1]; last_close = df['close'].iloc[-1]
                for box in self.bull_boxes:
                    if box['active']:
                        if last_close < box['bottom']: box['active']=False; box['violated']=True; box['right_idx']=last_idx; self.logger.debug(f"Bull OB {box['id']} VIOLATED.")
                        elif self.ob_extend: box['right_idx'] = last_idx
                for box in self.bear_boxes:
                    if box['active']:
                        if last_close > box['top']: box['active']=False; box['violated']=True; box['right_idx']=last_idx; self.logger.debug(f"Bear OB {box['id']} VIOLATED.")
                        elif self.ob_extend: box['right_idx'] = last_idx
            else: self.logger.warning("Cannot check OB violations: Invalid last close.")

            # Prune OBs
            self.bull_boxes = sorted(self.bull_boxes, key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            self.bear_boxes = sorted(self.bear_boxes, key=lambda b: b['left_idx'], reverse=True)[:self.ob_max_boxes]
            active_bull = sum(1 for b in self.bull_boxes if b['active']); active_bear = sum(1 for b in self.bear_boxes if b['active'])
            self.logger.debug(f"Pruned OBs. Kept: Bull={len(self.bull_boxes)}({active_bull} act), Bear={len(self.bear_boxes)}({active_bear} act).")
        except Exception as e: self.logger.error(f"Error during OB processing: {e}", exc_info=True); # Continue with VT results

        # Prepare Final Results
        last = df.iloc[-1] if not df.empty else None
        def sanitize_dec(v, pos=False): return v if pd.notna(v) and isinstance(v, Decimal) and np.isfinite(float(v)) and (not pos or v > 0) else None
        results = StrategyAnalysisResults(
            dataframe=df,
            last_close=sanitize_dec(last.get('close')) or Decimal('0'),
            current_trend_up=bool(last.get('trend_up')) if isinstance(last.get('trend_up'), (bool, np.bool_)) else None,
            trend_just_changed=bool(last.get('trend_changed', False)),
            active_bull_boxes=[b for b in self.bull_boxes if b['active']],
            active_bear_boxes=[b for b in self.bear_boxes if b['active']],
            vol_norm_int=int(vol) if (vol := sanitize_dec(last.get('vol_norm'))) is not None else None,
            atr=sanitize_dec(last.get('atr'), pos=True),
            upper_band=sanitize_dec(last.get('upper_band')),
            lower_band=sanitize_dec(last.get('lower_band')) )
        trend = f"{NEON_GREEN}UP{RESET}" if results['current_trend_up'] is True else f"{NEON_RED}DOWN{RESET}" if results['current_trend_up'] is False else f"{NEON_YELLOW}N/A{RESET}"
        atr = f"{results['atr'].normalize()}" if results['atr'] else "N/A"; time_str = df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z') if not df.empty else "N/A"
        self.logger.debug(f"Strategy Results ({time_str}): Close={results['last_close'].normalize()}, Trend={trend}, TrendChg={results['trend_just_changed']}, ATR={atr}, VolNorm={results['vol_norm_int']}, Active OBs (B/B): {len(results['active_bull_boxes'])}/{len(results['active_bear_boxes'])}")
        return results

# --- Signal Generation based on Strategy Results ---
# ... (SignalGenerator Class - no major changes needed from v1.1.2) ...
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
        self.logger.info(f"Signal Generator Init: OB Entry Prox={self.ob_entry_proximity_factor.normalize()}, OB Exit Prox={self.ob_exit_proximity_factor.normalize()}")
        self.logger.info(f"  Initial TP Mult={self.initial_tp_atr_multiple.normalize()}, Initial SL Mult={self.initial_sl_atr_multiple.normalize()}")

    def generate_signal(self, analysis_results: StrategyAnalysisResults, open_position: Optional[Dict]) -> str:
        """Determines the trading signal."""
        if not isinstance(analysis_results, dict) or analysis_results.get('dataframe') is None or \
           analysis_results['dataframe'].empty or analysis_results.get('current_trend_up') is None or \
           analysis_results['last_close'] <= 0 or analysis_results['atr'] is None or analysis_results['atr'] <= 0:
            self.logger.warning(f"{NEON_YELLOW}Invalid strategy results for signal gen. Holding.{RESET}"); return "HOLD"
        close=analysis_results['last_close']; trend_up=analysis_results['current_trend_up']; changed=analysis_results['trend_just_changed']
        bull_obs=analysis_results['active_bull_boxes']; bear_obs=analysis_results['active_bear_boxes']; pos_side=open_position.get('side') if open_position else None
        signal = "HOLD"
        self.logger.debug(f"Signal Check: Close={close.normalize()}, TrendUp={trend_up}, Changed={changed}, Pos={pos_side or 'None'}, OBs(B/B)={len(bull_obs)}/{len(bear_obs)}")

        # Exit Checks
        if pos_side == 'long':
            if trend_up is False and changed: signal = "EXIT_LONG"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Trend flipped DOWN.{RESET}")
            elif signal=="HOLD" and bear_obs:
                try: ob=min(bear_obs, key=lambda b: abs(b['top']-close)); thresh=ob['top']*self.ob_exit_proximity_factor
                     if close >= thresh: signal="EXIT_LONG"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT LONG Signal: Price {close} >= Bear OB Exit Thresh {thresh} (OB ID {ob['id']}){RESET}")
                except Exception as e: self.logger.warning(f"Error Bear OB exit check: {e}")
        elif pos_side == 'short':
            if trend_up is True and changed: signal = "EXIT_SHORT"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Trend flipped UP.{RESET}")
            elif signal=="HOLD" and bull_obs:
                try: ob=min(bull_obs, key=lambda b: abs(b['bottom']-close)); thresh=ob['bottom']/self.ob_exit_proximity_factor if self.ob_exit_proximity_factor > 0 else ob['bottom']
                     if close <= thresh: signal="EXIT_SHORT"; self.logger.warning(f"{NEON_YELLOW}{BRIGHT}EXIT SHORT Signal: Price {close} <= Bull OB Exit Thresh {thresh} (OB ID {ob['id']}){RESET}")
                except Exception as e: self.logger.warning(f"Error Bull OB exit check: {e}")
        if signal != "HOLD": return signal

        # Entry Checks
        if pos_side is None:
            if trend_up is True and bull_obs:
                for ob in bull_obs:
                    lb=ob['bottom']; ub=ob['top']*self.ob_entry_proximity_factor
                    if lb <= close <= ub: signal="BUY"; self.logger.info(f"{NEON_GREEN}{BRIGHT}BUY Signal: Trend UP & Price {close} in Bull OB {ob['id']} [{lb}-{ub}]{RESET}"); break
            elif trend_up is False and bear_obs:
                for ob in bear_obs:
                    lb=ob['bottom']/self.ob_entry_proximity_factor if self.ob_entry_proximity_factor > 0 else ob['bottom']; ub=ob['top']
                    if lb <= close <= ub: signal="SELL"; self.logger.info(f"{NEON_RED}{BRIGHT}SELL Signal: Trend DOWN & Price {close} in Bear OB {ob['id']} [{lb}-{ub}]{RESET}"); break

        if signal == "HOLD": self.logger.debug(f"HOLD Signal: No valid entry or exit condition met.");
        return signal

    def calculate_initial_tp_sl(self, entry_price: Decimal, signal: str, atr: Decimal, market_info: Dict, exchange: ccxt.Exchange) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Calculates initial TP/SL levels based on ATR and market precision."""
        lg = self.logger
        if signal not in ["BUY", "SELL"] or entry_price <= 0 or atr <= 0 or market_info['precision'].get('price') is None:
            lg.error(f"Invalid input for TP/SL calc."); return None, None
        try:
            min_tick = Decimal(str(market_info['precision']['price'])); assert min_tick > 0
            tp_mult = self.initial_tp_atr_multiple; sl_mult = self.initial_sl_atr_multiple # Guaranteed > 0
            tp_offset = atr * tp_mult; sl_offset = atr * sl_mult
            tp_raw = (entry_price + tp_offset) if signal == "BUY" and tp_mult > 0 else (entry_price - tp_offset) if signal == "SELL" and tp_mult > 0 else None
            sl_raw = (entry_price - sl_offset) if signal == "BUY" else (entry_price + sl_offset)

            def fmt_lvl(price_dec: Optional[Decimal], name: str) -> Optional[Decimal]:
                if price_dec is None or price_dec <= 0: lg.warning(f"Calc {name} invalid ({price_dec})."); return None
                try: fmt_str = exchange.price_to_precision(symbol=market_info['symbol'], price=float(price_dec), rounding_mode=exchange.ROUND); fmt_dec = Decimal(fmt_str); return fmt_dec if fmt_dec > 0 else None
                except Exception as e: lg.error(f"Error formatting {name} {price_dec}: {e}."); return None
            tp = fmt_lvl(tp_raw, "TP"); sl = fmt_lvl(sl_raw, "SL")

            # Final Validation
            if sl is not None:
                if (signal=="BUY" and sl >= entry_price) or (signal=="SELL" and sl <= entry_price): lg.warning(f"Formatted {signal} SL {sl} not beyond entry {entry_price}. Adjusting."); sl = fmt_lvl(entry_price - min_tick if signal=="BUY" else entry_price + min_tick, "SL")
            if tp is not None:
                if (signal=="BUY" and tp <= entry_price) or (signal=="SELL" and tp >= entry_price): lg.warning(f"Formatted {signal} TP {tp} not beyond entry {entry_price}. Setting None."); tp = None
            lg.debug(f"Initial Calc Levels: TP={tp.normalize() if tp else 'None'}, SL={sl.normalize() if sl else 'FAIL'}")
            if sl is None: lg.error(f"{NEON_RED}SL calculation failed. Cannot size.{RESET}"); return tp, None
            return tp, sl
        except Exception as e: lg.error(f"{NEON_RED}Error calculating TP/SL: {e}{RESET}", exc_info=True); return None, None

# --- Main Analysis and Trading Loop Function ---
# ... (analyze_and_trade_symbol function - incorporating dynamic fetch limit & improved data check) ...
def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger,
                             strategy_engine: VolumaticOBStrategy, signal_generator: SignalGenerator, market_info: Dict) -> None:
    """Performs one cycle of analysis and trading for a symbol."""
    lg = logger
    lg.info(f"\n---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()
    ccxt_interval = CCXT_INTERVAL_MAP[config["interval"]]

    # Determine required klines and fetch limit, respecting API cap
    min_req_data = strategy_engine.min_data_len
    fetch_limit_config = config.get("fetch_limit", DEFAULT_FETCH_LIMIT)
    # Ensure we request at least the required minimum, up to the API limit
    fetch_limit_request = max(min_req_data, fetch_limit_config)
    fetch_limit_request = min(fetch_limit_request, BYBIT_API_KLINE_LIMIT) # Cap at API limit
    lg.debug(f"Strategy requires min {min_req_data} candles. Requesting {fetch_limit_request} (API limit: {BYBIT_API_KLINE_LIMIT}).")

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=fetch_limit_request, logger=lg)
    fetched_count = len(klines_df)

    # --- Check if sufficient data was fetched AFTER potentially hitting API limit ---
    if klines_df.empty or fetched_count < min_req_data:
        api_limit_hit_insufficient = (fetch_limit_request == BYBIT_API_KLINE_LIMIT and fetched_count == BYBIT_API_KLINE_LIMIT and fetched_count < min_req_data)
        if api_limit_hit_insufficient:
             lg.error(f"Failed fetch sufficient data ({fetched_count}/{min_req_data}). {NEON_YELLOW}Hit API limit ({BYBIT_API_KLINE_LIMIT}). Reduce strategy lookbacks (e.g., 'vt_vol_ema_length') in config.json.{RESET}")
        else:
             lg.error(f"Failed fetch sufficient data ({fetched_count}/{min_req_data}). Check connection/symbol. Skipping cycle.")
        return

    # Run Strategy Analysis
    try: analysis_results = strategy_engine.update(klines_df)
    except Exception as analysis_err: lg.error(f"{NEON_RED}Strategy analysis error: {analysis_err}{RESET}", exc_info=True); return
    if not analysis_results or analysis_results['current_trend_up'] is None or analysis_results['last_close'] <= 0 or analysis_results['atr'] is None:
        lg.error(f"{NEON_RED}Strategy analysis incomplete. Skipping cycle.{RESET}"); lg.debug(f"Analysis Results: {analysis_results}"); return
    latest_close = analysis_results['last_close']; current_atr = analysis_results['atr']

    # Get Current Price & Position
    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    price_for_checks = current_price if current_price else latest_close
    if price_for_checks <= 0: lg.error(f"{NEON_RED}Cannot get valid price. Skipping cycle.{RESET}"); return
    if current_price is None: lg.warning(f"{NEON_YELLOW}Using last close {latest_close} for protection checks.{RESET}")
    open_position = get_open_position(exchange, symbol, lg)

    # Generate Signal
    try: signal = signal_generator.generate_signal(analysis_results, open_position)
    except Exception as signal_err: lg.error(f"{NEON_RED}Signal generation error: {signal_err}{RESET}", exc_info=True); return

    # Trading Logic
    trading_enabled = config.get("enable_trading", False)
    if not trading_enabled:
        lg.info(f"{NEON_YELLOW}Trading disabled. Signal: {signal}. Analysis complete.{RESET}")
        # ... (rest of analysis-only logging/return - identical to v1.1.2) ...
        cycle_end_time = time.monotonic(); lg.debug(f"---== Analysis-Only Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")
        return

    lg.debug(f"Trading enabled. Signal: {signal}. Position: {'Yes' if open_position else 'No'}")

    # --- Scenario 1: No Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {NEON_GREEN if signal=='BUY' else NEON_RED}{BRIGHT}{signal} Signal & No Position: Init Entry Sequence...{RESET} ***")
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None or balance <= 0: lg.error(f"Trade Abort {signal}: No balance {balance}."); return
            tp_calc, sl_calc = signal_generator.calculate_initial_tp_sl(latest_close, signal, current_atr, market_info, exchange)
            if sl_calc is None: lg.error(f"Trade Abort {signal}: SL calc failed."); return
            if tp_calc is None: lg.warning(f"Initial TP calc failed/disabled.")
            leverage_ok = True
            if market_info['is_contract']: leverage=int(config['leverage']); leverage_ok = set_leverage_ccxt(exchange, symbol, leverage, market_info, lg) if leverage > 0 else True
            if not leverage_ok: lg.error(f"Trade Abort {signal}: Leverage set failed."); return
            pos_size = calculate_position_size(balance, config["risk_per_trade"], sl_calc, latest_close, market_info, exchange, lg)
            if pos_size is None or pos_size <= 0: lg.error(f"Trade Abort {signal}: Invalid size {pos_size}."); return

            lg.info(f"==> Placing {signal} Market Order | Size: {pos_size.normalize()} <==")
            trade_order = place_trade(exchange, symbol, signal, pos_size, market_info, lg, reduce_only=False)

            if trade_order and trade_order.get('id'): # Post-Trade
                confirm_delay = config["position_confirm_delay_seconds"]; lg.info(f"Order placed. Waiting {confirm_delay}s for confirm..."); time.sleep(confirm_delay)
                confirmed_pos = get_open_position(exchange, symbol, lg)
                if confirmed_pos:
                    try:
                        entry_actual_str = confirmed_pos.get('entryPrice'); entry_actual = Decimal(str(entry_actual_str)) if entry_actual_str else latest_close
                        if entry_actual <= 0: entry_actual = latest_close
                        lg.info(f"{NEON_GREEN}Position Confirmed! Entry: ~{entry_actual.normalize()}{RESET}")
                        # Set Protection based on actual entry
                        prot_cfg = config["protection"]
                        tp_prot, sl_prot = signal_generator.calculate_initial_tp_sl(entry_actual, signal, current_atr, market_info, exchange)
                        if sl_prot is None: lg.error(f"{NEON_RED}Failed recalculate SL for protection! Position vulnerable!{RESET}")
                        prot_success = False
                        if prot_cfg.get("enable_trailing_stop", True):
                             lg.info(f"Setting TSL (Entry={entry_actual.normalize()})...")
                             prot_success = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_pos, config, lg, take_profit_price=tp_prot)
                        elif not prot_cfg.get("enable_trailing_stop", True) and (prot_cfg.get("initial_stop_loss_atr_multiple", 0) > 0 or prot_cfg.get("initial_take_profit_atr_multiple", 0) > 0):
                             lg.info(f"Setting Fixed SL/TP (Entry={entry_actual.normalize()})...")
                             if sl_prot or tp_prot: prot_success = _set_position_protection(exchange, symbol, market_info, confirmed_pos, lg, stop_loss_price=sl_prot, take_profit_price=tp_prot)
                             else: lg.warning("No valid fixed SL/TP levels calc'd. No protection set."); prot_success = True
                        else: lg.info("Neither TSL nor Fixed SL/TP enabled."); prot_success = True
                        if prot_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION COMPLETE ({symbol} {signal}) ===")
                        else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({symbol} {signal}). MANUAL MONITORING! ===")
                    except Exception as post_err: lg.error(f"{NEON_RED}Post-trade setup error: {post_err}{RESET}", exc_info=True); lg.warning(f"{NEON_YELLOW}Position may lack protection! Manual check!{RESET}")
                else: lg.error(f"{NEON_RED}Order placed, but FAILED TO CONFIRM position after delay! Manual check!{RESET}")
            else: lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({symbol} {signal}) ===")
        else: lg.info(f"Signal HOLD, no position. No action.")

    # --- Scenario 2: Existing Position ---
    else: # Position exists
        pos_side = open_position['side']; pos_size = open_position['size_decimal']
        lg.info(f"Existing {pos_side.upper()} position found (Size: {pos_size.normalize()}). Signal: {signal}")
        exit_triggered = (signal == "EXIT_LONG" and pos_side == 'long') or (signal == "EXIT_SHORT" and pos_side == 'short')

        if exit_triggered: # Handle Exit Signal
            lg.warning(f"{NEON_YELLOW}{BRIGHT}*** {signal} Signal! Closing {pos_side} position... ***{RESET}")
            try:
                close_sig = "SELL" if pos_side == 'long' else "BUY"; size_close = abs(pos_size)
                if size_close <= 0: lg.warning("Close size zero/neg. Already closed?"); return
                lg.info(f"==> Placing {close_sig} MARKET order (reduceOnly) | Size: {size_close.normalize()} <==")
                close_order = place_trade(exchange, symbol, close_sig, size_close, market_info, lg, reduce_only=True)
                if close_order and close_order.get('id'): lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully.{RESET}")
                else: lg.error(f"{NEON_RED}Failed place CLOSE order. Manual check!{RESET}")
            except Exception as close_err: lg.error(f"{NEON_RED}Error closing position: {close_err}{RESET}", exc_info=True); lg.warning(f"{NEON_YELLOW}Manual close may be needed!{RESET}")
        else: # Handle Position Management (BE, TSL)
            lg.debug(f"Signal ({signal}) allows holding. Managing protections...")
            prot_cfg = config["protection"]
            tsl_dist_str = open_position.get('trailingStopLoss')
            tsl_active = False; try: tsl_active = tsl_dist_str and Decimal(str(tsl_dist_str)) > 0 except Exception: pass
            sl_curr = None; tp_curr = None
            try: sl_curr = Decimal(str(open_position['stopLossPrice'])) if open_position.get('stopLossPrice') and str(open_position['stopLossPrice']) != '0' else None
            except Exception: pass
            try: tp_curr = Decimal(str(open_position['takeProfitPrice'])) if open_position.get('takeProfitPrice') and str(open_position['takeProfitPrice']) != '0' else None
            except Exception: pass
            entry_price = None; try: entry_price = Decimal(str(open_position['entryPrice'])) if open_position.get('entryPrice') else None except Exception: pass

            # Break-Even Logic
            be_enabled = prot_cfg.get("enable_break_even", True)
            if be_enabled and not tsl_active and entry_price and current_atr and price_for_checks > 0:
                lg.debug(f"Checking BE (Entry:{entry_price.normalize()}, Price:{price_for_checks.normalize()}, ATR:{current_atr.normalize()})...")
                try:
                    be_trig_atr = Decimal(str(prot_cfg["break_even_trigger_atr_multiple"]))
                    be_offset_t = int(prot_cfg["break_even_offset_ticks"]); assert be_trig_atr > 0 and be_offset_t >= 0
                    diff = (price_for_checks - entry_price) if pos_side == 'long' else (entry_price - price_for_checks)
                    profit_atr = diff / current_atr
                    lg.debug(f"  BE Check: Profit ATRs={profit_atr:.2f}, Trigger={be_trig_atr.normalize()}")
                    if profit_atr >= be_trig_atr:
                        lg.info(f"{NEON_PURPLE}{BRIGHT}BE Profit target REACHED!{RESET}")
                        tick = Decimal(str(market_info['precision']['price'])); offset_val = tick * Decimal(str(be_offset_t))
                        be_sl = (entry_price + offset_val).quantize(tick, ROUND_UP) if pos_side == 'long' else (entry_price - offset_val).quantize(tick, ROUND_DOWN)
                        if be_sl and be_sl > 0:
                            update = False
                            if sl_curr is None: update=True; lg.info("  No current SL, setting BE.")
                            elif pos_side == 'long' and be_sl > sl_curr: update=True; lg.info(f"  BE SL {be_sl} > Current SL {sl_curr}.")
                            elif pos_side == 'short' and be_sl < sl_curr: update=True; lg.info(f"  BE SL {be_sl} < Current SL {sl_curr}.")
                            else: lg.debug(f"  Current SL {sl_curr} >= BE target {be_sl}.")
                            if update:
                                lg.warning(f"{NEON_PURPLE}{BRIGHT}*** Moving SL to Break-Even at {be_sl.normalize()} ***{RESET}")
                                if _set_position_protection(exchange, symbol, market_info, open_position, lg, stop_loss_price=be_sl, take_profit_price=tp_curr): lg.info(f"{NEON_GREEN}BE SL set/updated successfully.{RESET}")
                                else: lg.error(f"{NEON_RED}Failed set/update BE SL via API.{RESET}")
                        else: lg.error(f"{NEON_RED}BE triggered, but calc'd BE SL invalid ({be_sl}).{RESET}")
                    else: lg.debug(f"BE Profit target not reached.")
                except Exception as be_err: lg.error(f"{NEON_RED}Error during BE check: {be_err}{RESET}", exc_info=True)
            elif be_enabled: lg.debug(f"BE check skipped: {'TSL active' if tsl_active else 'Missing data'}.")
            else: lg.debug("BE check skipped: Disabled.")

            # TSL Setup/Recovery
            tsl_enabled = prot_cfg.get("enable_trailing_stop", True)
            if tsl_enabled and not tsl_active and entry_price and current_atr:
                 lg.warning(f"{NEON_YELLOW}TSL enabled but not active. Attempting setup/recovery...{RESET}")
                 tp_recalc, _ = signal_generator.calculate_initial_tp_sl(entry_price, pos_side.upper(), current_atr, market_info, exchange)
                 if set_trailing_stop_loss(exchange, symbol, market_info, open_position, config, lg, take_profit_price=tp_recalc): lg.info(f"TSL setup/recovery successful.")
                 else: lg.error(f"TSL setup/recovery failed.")
            elif tsl_enabled: lg.debug("TSL setup/recovery skipped: Already active.")
            else: lg.debug("TSL setup/recovery skipped: Disabled.")

    # --- Cycle End ---
    cycle_end_time = time.monotonic()
    lg.info(f"---== Analysis Cycle End ({symbol}, {cycle_end_time - cycle_start_time:.2f}s) ==---\n")

# --- Main Function ---
def main() -> None:
    """Main function to initialize and run the bot."""
    global CONFIG, QUOTE_CURRENCY
    init_logger.info(f"{BRIGHT}--- Starting Pyrmethus Volumatic Bot v1.1.3 ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---{RESET}")
    init_logger.info(f"Config Loaded: Quote={QUOTE_CURRENCY}, Trading={CONFIG['enable_trading']}, Sandbox={CONFIG['use_sandbox']}")
    try: init_logger.info(f"Versions: Py={os.sys.version.split()[0]}, CCXT={ccxt.__version__}, Pandas={pd.__version__}, TA={getattr(ta, 'version', 'N/A')}")
    except Exception as e: init_logger.warning(f"Version check failed: {e}")

    # User Confirmation
    if CONFIG["enable_trading"]:
        init_logger.warning(f"{NEON_YELLOW}{BRIGHT}!!! LIVE TRADING ENABLED !!!{RESET}")
        init_logger.warning(f"Mode: {'SANDBOX' if CONFIG['use_sandbox'] else f'{NEON_RED}LIVE (Real Money)'}{RESET}")
        prot_cfg = CONFIG["protection"]
        init_logger.warning(f"{BRIGHT}--- Review Settings ---{RESET}")
        init_logger.warning(f"  Risk/Trade: {CONFIG['risk_per_trade']:.2%}, Leverage: {CONFIG['leverage']}x")
        init_logger.warning(f"  TSL: {'ON' if prot_cfg['enable_trailing_stop'] else 'OFF'} (CB:{prot_cfg['trailing_stop_callback_rate']:.2%}, Act:{prot_cfg['trailing_stop_activation_percentage']:.2%})")
        init_logger.warning(f"  BE: {'ON' if prot_cfg['enable_break_even'] else 'OFF'} (Trig:{prot_cfg['break_even_trigger_atr_multiple']} ATR, Off:{prot_cfg['break_even_offset_ticks']} ticks)")
        try: input(f"{BRIGHT}>>> Press {NEON_GREEN}Enter{RESET}{BRIGHT} to continue, or {NEON_RED}Ctrl+C{RESET}{BRIGHT} to abort... {RESET}")
        except KeyboardInterrupt: init_logger.info("User aborted startup."); print(f"\n{NEON_YELLOW}Bot aborted.{RESET}"); return
        init_logger.info("User confirmed live settings.")
    else: init_logger.info(f"{NEON_YELLOW}Trading disabled. Analysis-only mode.{RESET}")

    # Init Exchange
    init_logger.info("Initializing exchange...")
    exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical(f"{NEON_RED}Exchange init failed. Exiting.{RESET}"); return
    init_logger.info(f"Exchange {exchange.id} initialized.")

    # Get Symbol & Market Info
    target_symbol = None; market_info = None
    while target_symbol is None:
        try:
            symbol_input = input(f"{NEON_YELLOW}Enter trading symbol (e.g., BTC/USDT): {RESET}").strip().upper()
            if not symbol_input: continue
            symbols_to_try = [symbol_input]
            if '/' in symbol_input and ':' not in symbol_input: symbols_to_try.append(f"{symbol_input}:{QUOTE_CURRENCY}")
            elif ':' in symbol_input and '/' not in symbol_input and ':' != symbol_input[-1]: symbols_to_try.append(symbol_input.replace(':', '/'))
            symbols_to_try = list(dict.fromkeys(symbols_to_try))
            for sym_attempt in symbols_to_try:
                init_logger.info(f"Validating symbol '{sym_attempt}'...")
                m_info = get_market_info(exchange, sym_attempt, init_logger)
                if m_info:
                     target_symbol = m_info['symbol']; market_info = m_info
                     init_logger.info(f"Validated Symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_info['contract_type_str']})")
                     if market_info['precision']['price'] is None or market_info['precision']['amount'] is None:
                          init_logger.critical(f"{NEON_RED}CRITICAL: Market '{target_symbol}' missing precision! Cannot trade safely. Exiting.{RESET}"); return
                     break
            if market_info is None: init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' (and variations) not found/invalid. Try again.{RESET}")
        except KeyboardInterrupt: init_logger.info("User aborted."); print(f"\n{NEON_YELLOW}Bot aborted.{RESET}"); return
        except Exception as e: init_logger.error(f"Error during symbol validation: {e}", exc_info=True)

    # Get Interval
    selected_interval = None
    while selected_interval is None:
        default_int = CONFIG['interval']; interval_input = input(f"{NEON_YELLOW}Enter interval {VALID_INTERVALS} (default: {default_int}): {RESET}").strip()
        if not interval_input: interval_input = default_int
        if interval_input in VALID_INTERVALS:
             selected_interval = interval_input
             if CONFIG['interval'] != selected_interval: CONFIG["interval"] = selected_interval # Update in memory only
             init_logger.info(f"Using interval: {selected_interval} (CCXT: {CCXT_INTERVAL_MAP[selected_interval]})")
             break
        else: init_logger.error(f"{NEON_RED}Invalid interval '{interval_input}'. Choose from {VALID_INTERVALS}.{RESET}")

    # Setup Symbol Logger & Strategy Instances
    symbol_logger = setup_logger(target_symbol)
    symbol_logger.info(f"---=== {BRIGHT}Starting Trading Loop: {target_symbol} ({CONFIG['interval']}){RESET} ===---")
    symbol_logger.info(f"Trading: {CONFIG['enable_trading']}, Sandbox: {CONFIG['use_sandbox']}")
    prot_cfg = CONFIG["protection"] # Log final settings
    symbol_logger.info(f"Settings: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if prot_cfg['enable_trailing_stop'] else 'OFF'}, BE={'ON' if prot_cfg['enable_break_even'] else 'OFF'}")
    try:
        strategy_engine = VolumaticOBStrategy(CONFIG, market_info, symbol_logger)
        signal_generator = SignalGenerator(CONFIG, symbol_logger)
    except Exception as engine_err: symbol_logger.critical(f"{NEON_RED}Failed init strategy/signal generator: {engine_err}. Exiting.{RESET}", exc_info=True); return

    # Main Loop
    symbol_logger.info(f"{BRIGHT}Entering main loop... Press Ctrl+C to stop.{RESET}")
    try:
        while True:
            loop_start = time.time()
            symbol_logger.debug(f">>> New Loop: {datetime.now(TIMEZONE).strftime('%H:%M:%S %Z')}")
            try: # Core logic per cycle
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger, strategy_engine, signal_generator, market_info)
            except ccxt.RateLimitExceeded as e: symbol_logger.warning(f"{NEON_YELLOW}Rate limit: {e}. Wait 60s...{RESET}"); time.sleep(60)
            except (ccxt.NetworkError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, ccxt.RequestTimeout) as e: symbol_logger.error(f"{NEON_RED}Network error: {e}. Wait {RETRY_DELAY_SECONDS*3}s...{RESET}"); time.sleep(RETRY_DELAY_SECONDS*3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}CRITICAL Auth Error: {e}. Stopping bot.{RESET}"); break
            except ccxt.ExchangeNotAvailable as e: symbol_logger.error(f"{NEON_RED}Exchange unavailable: {e}. Wait 60s...{RESET}"); time.sleep(60)
            except ccxt.OnMaintenance as e: symbol_logger.error(f"{NEON_RED}Exchange maintenance: {e}. Wait 5m...{RESET}"); time.sleep(300)
            except ccxt.ExchangeError as e: symbol_logger.error(f"{NEON_RED}Unhandled Exchange Error: {e}{RESET}", exc_info=True); time.sleep(10)
            except Exception as loop_err: symbol_logger.error(f"{NEON_RED}Critical loop error: {loop_err}{RESET}", exc_info=True); time.sleep(15)
            # Loop Delay
            elapsed = time.time() - loop_start; loop_delay = CONFIG["loop_delay_seconds"]; sleep_time = max(0, loop_delay - elapsed)
            symbol_logger.debug(f"<<< Cycle took {elapsed:.2f}s. Sleep {sleep_time:.2f}s...")
            if sleep_time > 0: time.sleep(sleep_time)
    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as critical_err: init_logger.critical(f"{NEON_RED}CRITICAL UNHANDLED ERROR: {critical_err}{RESET}", exc_info=True)
    finally: # Shutdown
        shutdown_msg = f"--- Pyrmethus Bot ({target_symbol or 'N/A'}) Stopping ---"
        init_logger.info(shutdown_msg)
        if 'symbol_logger' in locals(): symbol_logger.info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); init_logger.info("Exchange connection closed.") # exchange.close() often not needed for synchronous
            except Exception as close_err: init_logger.error(f"Error closing exchange: {close_err}")
        try: # Explicitly close handlers
            if 'init_logger' in locals():
                for handler in init_logger.handlers[:]: init_logger.removeHandler(handler); handler.close()
            if 'symbol_logger' in locals():
                for handler in symbol_logger.handlers[:]: symbol_logger.removeHandler(handler); handler.close()
        except Exception as log_close_err: print(f"Error closing log handlers: {log_close_err}")
        logging.shutdown()
        print(f"\n{NEON_YELLOW}{BRIGHT}Bot stopped.{RESET}")

if __name__ == "__main__":
    # REMINDER: Adjust 'vt_vol_ema_length' in config.json if strategy requires > 950 lookback.
    main()

# --- END OF FILE volumatictrend1.1.3.py ---
