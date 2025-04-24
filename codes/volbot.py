# volbot.py
# Incorporates Volumatic Trend + Order Block strategy into the livexy trading framework.

import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple, List

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta # Import pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

# Initialize colorama and set precision
getcontext().prec = 18  # Increased precision for calculations
init(autoreset=True)
load_dotenv()

# Neon Color Scheme (from livexy) + Strategy Colors (from volbot)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Volbot Colors
COLOR_UP = Fore.CYAN + Style.BRIGHT
COLOR_DN = Fore.YELLOW + Style.BRIGHT
COLOR_BULL_BOX = Fore.GREEN
COLOR_BEAR_BOX = Fore.RED
COLOR_CLOSED_BOX = Fore.LIGHTBLACK_EX
COLOR_INFO = Fore.MAGENTA # Reuse Neon Purple/Magenta
COLOR_HEADER = Fore.BLUE + Style.BRIGHT # Reuse Neon Blue/Cyan

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Timezone for logging and display (adjust as needed)
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3 # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5 # Delay between retries
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # Intervals supported by the bot's logic
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP status codes considered retryable

# --- Default Volbot Strategy Parameters (will be overridden by config.json if present) ---
DEFAULT_VOLBOT_LENGTH = 40
DEFAULT_VOLBOT_ATR_LENGTH = 200 # ATR period used for Volbot levels
DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK = 1000
DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE = 100 # 100th percentile (max)
DEFAULT_VOLBOT_OB_SOURCE = "Wicks" # "Wicks" or "Bodys"
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H = 25
DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L = 25
DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L = 25
DEFAULT_VOLBOT_MAX_BOXES = 50 # Max number of active Bull/Bear boxes

# Default Risk Management Parameters (from livexy)
DEFAULT_ATR_PERIOD = 14 # Default ATR period for SL/TP/BE calculations (Risk Management)

LOOP_DELAY_SECONDS = 15 # Time between the end of one cycle and the start of the next
# QUOTE_CURRENCY dynamically loaded from config

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found,
       and ensuring all default keys are present."""
    default_config = {
        # --- General Bot Settings ---
        "interval": "5", # Default to 5 minute interval (string format for our logic)
        "retry_delay": 5,
        "enable_trading": False, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True (testnet), disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 10,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1, # Limit open positions for this symbol (common strategy)
        "quote_currency": "USDT", # Currency for balance check and sizing

        # --- Volbot Strategy Settings ---
        "volbot_enabled": True, # Flag to easily enable/disable this strategy logic
        "volbot_length": DEFAULT_VOLBOT_LENGTH,
        "volbot_atr_length": DEFAULT_VOLBOT_ATR_LENGTH, # ATR for strategy levels
        "volbot_volume_percentile_lookback": DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK,
        "volbot_volume_normalization_percentile": DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE,
        "volbot_ob_source": DEFAULT_VOLBOT_OB_SOURCE,
        "volbot_pivot_left_len_h": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H,
        "volbot_pivot_right_len_h": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H,
        "volbot_pivot_left_len_l": DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L,
        "volbot_pivot_right_len_l": DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L,
        "volbot_max_boxes": DEFAULT_VOLBOT_MAX_BOXES,
        "volbot_signal_on_trend_flip": True, # Generate signal immediately on trend flip
        "volbot_signal_on_ob_entry": True, # Generate signal when price enters OB in trend direction

        # --- Risk Management Settings ---
        "atr_period": DEFAULT_ATR_PERIOD, # ATR period for SL/TP/BE calculations
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP

        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL (exchange TSL)
        "trailing_stop_callback_rate": 0.005, # Example: 0.5% trail distance relative to entry/activation
        "trailing_stop_activation_percentage": 0.003, # Example: Activate when 0.3% in profit

        # --- Break-Even Stop Config ---
        "enable_break_even": True,              # Enable moving SL to break-even
        "break_even_trigger_atr_multiple": 1.0, # Example: Trigger BE when profit = 1x ATR
        "break_even_offset_ticks": 2,

        # --- Livexy Legacy/Optional Settings (can be removed if not used by Volbot strategy) ---
        # "signal_score_threshold": 1.5, # Not used by Volbot's rule-based signals
        # "indicators": { ... }, # Not directly used by Volbot strategy logic
        # "weight_sets": { ... }, # Not used by Volbot's rule-based signals
        # "active_weight_set": "default", # Not used by Volbot
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file {filepath}: {e}{RESET}")
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            updated_config = _ensure_config_keys(config_from_file, default_config)
            if updated_config != config_from_file:
                try:
                    with open(filepath, "w", encoding="utf-8") as f_write:
                        json.dump(updated_config, f_write, indent=4)
                    print(f"{NEON_YELLOW}Updated config file with missing default keys: {filepath}{RESET}")
                except IOError as e:
                    print(f"{NEON_RED}Error writing updated config file {filepath}: {e}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config file {filepath}: {e}. Using default config.{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        except IOError as e_create:
            print(f"{NEON_RED}Error creating default config file after load error: {e_create}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], default_value)
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config
console_log_level = logging.INFO # Default console level, can be changed in main()

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for the given symbol with file and console handlers."""
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"volbot_{safe_symbol}" # Use safe symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        for handler in logger.handlers:
             if isinstance(handler, logging.StreamHandler):
                  handler.setLevel(console_log_level)
        return logger

    logger.setLevel(logging.DEBUG)

    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
        )
        file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger for {log_filename}: {e}")

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object with error handling."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'fetchTickerTimeout': 10000,
                'fetchBalanceTimeout': 15000,
                'createOrderTimeout': 20000,
            }
        }

        exchange_class = ccxt.bybit
        exchange = exchange_class(exchange_options)

        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)

        logger.info(f"Loading markets for {exchange.id}...")
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        account_type_to_test = 'CONTRACT'
        logger.info(f"Attempting initial balance fetch (Account Type: {account_type_to_test})...")
        try:
            balance = exchange.fetch_balance(params={'type': account_type_to_test})
            available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
            logger.info(f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
        except ccxt.AuthenticationError as auth_err:
            logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
            logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}")
            return None
        except ccxt.ExchangeError as balance_err:
            logger.warning(f"{NEON_YELLOW}Exchange error during initial balance fetch ({account_type_to_test}): {balance_err}. Trying default fetch...{RESET}")
            try:
                balance = exchange.fetch_balance()
                available_quote = balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')
                logger.info(f"{NEON_GREEN}Successfully fetched balance using default parameters.{RESET} (Example: {QUOTE_CURRENCY} available: {available_quote})")
            except Exception as fallback_err:
                logger.warning(f"{NEON_YELLOW}Default balance fetch also failed: {fallback_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")
        except Exception as balance_err:
            logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}")

        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error during initialization: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
    return None


# --- CCXT Data Fetching ---
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT ticker with fallbacks."""
    lg = logger
    try:
        lg.debug(f"Fetching ticker for {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        lg.debug(f"Ticker data for {symbol}: {ticker}")

        price = None
        last_price = ticker.get('last')
        bid_price = ticker.get('bid')
        ask_price = ticker.get('ask')

        if last_price is not None:
            try:
                last_decimal = Decimal(str(last_price))
                if last_decimal > 0: price = last_decimal
            except Exception: pass

        if price is None and bid_price is not None and ask_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                ask_decimal = Decimal(str(ask_price))
                if bid_decimal > 0 and ask_decimal > 0 and bid_decimal <= ask_decimal:
                    price = (bid_decimal + ask_decimal) / 2
                elif ask_decimal > 0: price = ask_decimal # Fallback to ask
            except Exception: pass

        if price is None and ask_price is not None:
            try:
                ask_decimal = Decimal(str(ask_price))
                if ask_decimal > 0: price = ask_decimal
            except Exception: pass

        if price is None and bid_price is not None:
            try:
                bid_decimal = Decimal(str(bid_price))
                if bid_decimal > 0: price = bid_decimal
            except Exception: pass

        if price is not None and price > 0:
            return price
        else:
            lg.error(f"{NEON_RED}Failed to fetch a valid positive current price for {symbol} from ticker.{RESET}")
            return None

    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__)
    try:
        if not exchange.has['fetchOHLCV']:
            lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
            return pd.DataFrame()

        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
            try:
                lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1}/{MAX_API_RETRIES + 1})")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if ohlcv is not None and len(ohlcv) > 0: break
                else: lg.warning(f"fetch_ohlcv returned empty list for {symbol} (Attempt {attempt+1}). Retrying...")
                time.sleep(1)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_API_RETRIES:
                    lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else: raise e
            except ccxt.RateLimitExceeded as e:
                wait_time = int(e.args[0].split(' ')[-2]) if 'try again in' in e.args[0] else RETRY_DELAY_SECONDS * 5
                lg.warning(f"Rate limit exceeded fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                time.sleep(wait_time)
            except ccxt.ExchangeError as e: raise e

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe} after retries.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty: return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df = df[df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")

        if df.empty:
            lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/cleaning.{RESET}")
            return pd.DataFrame()

        df.sort_index(inplace=True)
        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after retries: {e}{RESET}")
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error processing klines for {symbol}: {e}{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error processing klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()

# --- Volbot Strategy Calculation Functions (Adapted from original volbot.py) ---

def ema_swma(series: pd.Series, length: int, logger: logging.Logger) -> pd.Series:
    """Calculates a smoothed EMA based on weighted average of last 4 values."""
    lg = logger
    lg.debug(f"# Calculating Smoothed Weighted EMA (Length: {length})...")
    if len(series) < 4:
        lg.warning("Series too short for SWMA calculation, returning standard EMA.")
        return ta.ema(series, length=length)

    weighted_series = (series.shift(3) / 6 +
                       series.shift(2) * 2 / 6 +
                       series.shift(1) * 2 / 6 +
                       series * 1 / 6)
    # Use adjust=False for behavior closer to TradingView's EMA calculation
    smoothed_ema = ta.ema(weighted_series.dropna(), length=length)
    return smoothed_ema.reindex(series.index)


def calculate_volatility_levels(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Calculates Volbot trend, EMAs, ATR and dynamic levels."""
    lg = logger
    lg.info("Calculating Volumatic Trend Levels...")
    length = config.get("volbot_length", DEFAULT_VOLBOT_LENGTH)
    atr_length = config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH)
    volume_percentile_lookback = config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK)
    # volume_normalization_percentile = config.get("volbot_volume_normalization_percentile", DEFAULT_VOLBOT_VOLUME_NORMALIZATION_PERCENTILE) # Currently hardcoded to 100 (max)

    # Calculate EMAs and ATR for the strategy levels
    df['ema1_strat'] = ema_swma(df['close'], length, lg) # Use strategy length
    df['ema2_strat'] = ta.ema(df['close'], length=length) # Use strategy length
    df['atr_strat'] = ta.atr(df['high'], df['low'], df['close'], length=atr_length) # Use strategy ATR length

    df['trend_up_strat'] = df['ema1_strat'].shift(1) < df['ema2_strat']
    df['trend_changed_strat'] = df['trend_up_strat'] != df['trend_up_strat'].shift(1)

    # Initialize level columns
    for col in ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'last_trend_change_index_strat']:
        df[col] = np.nan

    last_trend_change_idx = 0
    for i in range(1, len(df)):
        if df.loc[df.index[i], 'trend_changed_strat']:
            prev_idx = df.index[i-1]
            ema1_val = df.loc[prev_idx, 'ema1_strat']
            atr_val = df.loc[prev_idx, 'atr_strat']

            if pd.notna(ema1_val) and pd.notna(atr_val):
                upper = ema1_val + atr_val * 3
                lower = ema1_val - atr_val * 3
                lower_vol = lower + atr_val * 4
                upper_vol = upper - atr_val * 4
                step_up = (lower_vol - lower) / 100 if lower_vol > lower else 0
                step_dn = (upper - upper_vol) / 100 if upper > upper_vol else 0

                df.loc[df.index[i]:, 'upper_strat'] = upper
                df.loc[df.index[i]:, 'lower_strat'] = lower
                df.loc[df.index[i]:, 'lower_vol_strat'] = lower_vol
                df.loc[df.index[i]:, 'upper_vol_strat'] = upper_vol
                df.loc[df.index[i]:, 'step_up_strat'] = step_up
                df.loc[df.index[i]:, 'step_dn_strat'] = step_dn

                last_trend_change_idx = i
                df.loc[df.index[i]:, 'last_trend_change_index_strat'] = last_trend_change_idx
            else:
                 df.loc[df.index[i], ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'last_trend_change_index_strat']] = \
                     df.loc[df.index[i-1], ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'last_trend_change_index_strat']]
        elif i > 0 :
             df.loc[df.index[i], ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'last_trend_change_index_strat']] = \
                 df.loc[df.index[i-1], ['upper_strat', 'lower_strat', 'lower_vol_strat', 'upper_vol_strat', 'step_up_strat', 'step_dn_strat', 'last_trend_change_index_strat']]

    # Calculate normalized volume and volume steps
    df['percentile_vol_strat'] = df['volume'].rolling(window=volume_percentile_lookback, min_periods=1).max()
    df['vol_norm_strat'] = np.where(df['percentile_vol_strat'] != 0, (df['volume'] / df['percentile_vol_strat'] * 100), 0).fillna(0)

    df['vol_up_step_strat'] = df['step_up_strat'] * df['vol_norm_strat']
    df['vol_dn_step_strat'] = df['step_dn_strat'] * df['vol_norm_strat']

    df['vol_trend_up_level_strat'] = df['lower_strat'] + df['vol_up_step_strat']
    df['vol_trend_dn_level_strat'] = df['upper_strat'] - df['vol_dn_step_strat']

    # Calculate cumulative volume delta since last trend change
    df['volume_delta_strat'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    df['volume_total_strat'] = df['volume']

    trend_block = df['trend_changed_strat'].cumsum()
    df['cum_vol_delta_since_change_strat'] = df.groupby(trend_block)['volume_delta_strat'].cumsum()
    df['cum_vol_total_since_change_strat'] = df.groupby(trend_block)['volume_total_strat'].cumsum()

    lg.info("Volumatic Trend Levels calculation complete.")
    return df

def calculate_pivot_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """Identifies pivot highs/lows for Order Blocks."""
    lg = logger
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    left_h = config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    left_l = config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    lg.info(f"Calculating Pivot Order Blocks (Source: {source})...")

    high_col = df['high'] if source == "Wicks" else df['close']
    low_col = df['low'] if source == "Wicks" else df['open'] # Original volbot used open for low pivot body source
    open_col = df['open']
    close_col = df['close']

    df['ph_strat'] = np.nan
    df['pl_strat'] = np.nan

    # Calculate Pivot Highs
    for i in range(left_h, len(df) - right_h):
        is_pivot_high = True
        pivot_high_val = high_col.iloc[i]
        # Check left side
        for j in range(1, left_h + 1):
            if high_col.iloc[i-j] >= pivot_high_val:
                is_pivot_high = False
                break
        if not is_pivot_high: continue
        # Check right side
        for j in range(1, right_h + 1):
             if high_col.iloc[i+j] > pivot_high_val:
                is_pivot_high = False
                break
        if is_pivot_high:
            df.loc[df.index[i], 'ph_strat'] = pivot_high_val

    # Calculate Pivot Lows
    for i in range(left_l, len(df) - right_l):
        is_pivot_low = True
        pivot_low_val = low_col.iloc[i]
        # Check left side
        for j in range(1, left_l + 1):
            if low_col.iloc[i-j] <= pivot_low_val:
                is_pivot_low = False
                break
        if not is_pivot_low: continue
        # Check right side
        for j in range(1, right_l + 1):
             if low_col.iloc[i+j] < pivot_low_val:
                is_pivot_low = False
                break
        if is_pivot_low:
            df.loc[df.index[i], 'pl_strat'] = pivot_low_val

    lg.info("Pivot Order Block calculation complete.")
    return df

def manage_order_blocks(df: pd.DataFrame, config: Dict[str, Any], logger: logging.Logger) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """Creates, manages, and tracks Order Block states."""
    lg = logger
    lg.info("Managing Order Block Boxes...")
    source = config.get("volbot_ob_source", DEFAULT_VOLBOT_OB_SOURCE)
    right_h = config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H)
    right_l = config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L)
    max_boxes = config.get("volbot_max_boxes", DEFAULT_VOLBOT_MAX_BOXES)

    bull_boxes = [] # Stores active bullish OBs: [ {id, start_idx, end_idx, top, bottom, state} ]
    bear_boxes = [] # Stores active bearish OBs: [ {id, start_idx, end_idx, top, bottom, state} ]
    box_counter = 0

    # Add columns to store OB info for each bar
    df['active_bull_ob_strat'] = None # Store ref to active bull OB if price is within one
    df['active_bear_ob_strat'] = None # Store ref to active bear OB if price is within one

    for i in range(len(df)):
        current_idx = df.index[i]
        current_close = df.loc[current_idx, 'close']

        # --- Create new boxes ---
        if pd.notna(df.loc[current_idx, 'ph_strat']):
            ob_candle_idx_num = i - right_h
            if ob_candle_idx_num >= 0:
                ob_candle_idx = df.index[ob_candle_idx_num]
                top_price, bottom_price = np.nan, np.nan
                if source == "Bodys":
                    top_price = df.loc[ob_candle_idx, 'close']
                    bottom_price = df.loc[ob_candle_idx, 'open']
                else: # Wicks
                    top_price = df.loc[ob_candle_idx, 'high']
                    bottom_price = df.loc[ob_candle_idx, 'close']

                if pd.notna(top_price) and pd.notna(bottom_price):
                    if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                    box_counter += 1
                    new_box = {'id': f'BearOB_{box_counter}', 'type': 'bear', 'start_idx': ob_candle_idx,
                               'end_idx': current_idx, 'top': top_price, 'bottom': bottom_price, 'state': 'active'}
                    bear_boxes.append(new_box)

        if pd.notna(df.loc[current_idx, 'pl_strat']):
            ob_candle_idx_num = i - right_l
            if ob_candle_idx_num >= 0:
                ob_candle_idx = df.index[ob_candle_idx_num]
                top_price, bottom_price = np.nan, np.nan
                if source == "Bodys":
                     top_price = df.loc[ob_candle_idx, 'close']
                     bottom_price = df.loc[ob_candle_idx, 'open']
                else: # Wicks
                     top_price = df.loc[ob_candle_idx, 'open']
                     bottom_price = df.loc[ob_candle_idx, 'low']

                if pd.notna(top_price) and pd.notna(bottom_price):
                    if bottom_price > top_price: top_price, bottom_price = bottom_price, top_price
                    box_counter += 1
                    new_box = {'id': f'BullOB_{box_counter}', 'type': 'bull', 'start_idx': ob_candle_idx,
                               'end_idx': current_idx, 'top': top_price, 'bottom': bottom_price, 'state': 'active'}
                    bull_boxes.append(new_box)

        # --- Manage existing boxes ---
        active_bull_ref = None
        for box in bull_boxes:
            if box['state'] == 'active':
                if current_close < box['bottom']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx
                else:
                    box['end_idx'] = current_idx
                    if box['bottom'] <= current_close <= box['top']:
                         active_bull_ref = box

        active_bear_ref = None
        for box in bear_boxes:
            if box['state'] == 'active':
                if current_close > box['top']:
                    box['state'] = 'closed'
                    box['end_idx'] = current_idx
                else:
                    box['end_idx'] = current_idx
                    if box['bottom'] <= current_close <= box['top']:
                         active_bear_ref = box

        # Store references using direct assignment (might be slow on huge DFs, but simpler)
        df.at[current_idx, 'active_bull_ob_strat'] = active_bull_ref
        df.at[current_idx, 'active_bear_ob_strat'] = active_bear_ref

        # --- Clean up old boxes ---
        active_bull_boxes = [b for b in bull_boxes if b['state'] == 'active']
        if len(active_bull_boxes) > max_boxes:
            num_to_remove = len(active_bull_boxes) - max_boxes
            # Find the indices of the oldest *active* boxes to mark them for removal conceptually
            oldest_active_indices = sorted(
                [idx for idx, box in enumerate(bull_boxes) if box['state'] == 'active'],
                key=lambda k: bull_boxes[k]['start_idx']
            )[:num_to_remove]
            # Rebuild list excluding the oldest active ones (simplest way)
            bull_boxes = [box for idx, box in enumerate(bull_boxes) if box['state'] == 'closed' or idx not in oldest_active_indices]

        active_bear_boxes = [b for b in bear_boxes if b['state'] == 'active']
        if len(active_bear_boxes) > max_boxes:
            num_to_remove = len(active_bear_boxes) - max_boxes
            oldest_active_indices = sorted(
                [idx for idx, box in enumerate(bear_boxes) if box['state'] == 'active'],
                key=lambda k: bear_boxes[k]['start_idx']
            )[:num_to_remove]
            bear_boxes = [box for idx, box in enumerate(bear_boxes) if box['state'] == 'closed' or idx not in oldest_active_indices]

    lg.info("Order Block management complete.")
    # Return df and the final lists of boxes (active and closed)
    return df, bull_boxes, bear_boxes

# --- Trading Analyzer Class (Modified for Volbot Strategy) ---
class TradingAnalyzer:
    """Analyzes trading data using Volbot strategy and generates signals."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any],
    ) -> None:
        self.df_raw = df # Store raw klines
        self.df_processed = pd.DataFrame() # Store df after strategy calculations
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN_INTERVAL")

        # Stores latest relevant state values from the strategy
        self.strategy_state: Dict[str, Any] = {}
        # Stores active OBs (references to dicts from manage_order_blocks)
        self.latest_active_bull_ob: Optional[Dict] = None
        self.latest_active_bear_ob: Optional[Dict] = None
        # Keep track of all boxes for potential analysis/plotting later (optional)
        self.all_bull_boxes: List[Dict] = []
        self.all_bear_boxes: List[Dict] = []

        # Calculate indicators immediately on initialization
        self._calculate_strategy_indicators()
        # Update latest state immediately after calculation
        self._update_latest_strategy_state()

    def _calculate_strategy_indicators(self):
        """Calculates all indicators needed for the Volbot strategy and Risk Management."""
        if self.df_raw.empty:
            self.logger.warning(f"{NEON_YELLOW}Raw DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check for sufficient data length based on strategy and risk params
        strat_lookbacks = [
            self.config.get("volbot_length", DEFAULT_VOLBOT_LENGTH),
            self.config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
            self.config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
            self.config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + self.config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H),
            self.config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + self.config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L),
        ]
        risk_lookback = self.config.get("atr_period", DEFAULT_ATR_PERIOD) # For SL/TP ATR
        min_required_data = max(strat_lookbacks + [risk_lookback]) + 50 # Add buffer

        if len(self.df_raw) < min_required_data:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df_raw)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate or NaN.{RESET}")
            # Continue calculation, but expect NaNs

        try:
            df_calc = self.df_raw.copy() # Work on a copy

            # --- Calculate Risk Management ATR ---
            # This ATR is used for SL/TP/BE calculations, separate from strategy ATR
            atr_period_risk = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc['atr_risk'] = ta.atr(df_calc['high'], df_calc['low'], df_calc['close'], length=atr_period_risk)
            self.logger.debug(f"Calculated Risk Management ATR (Length: {atr_period_risk})")

            # --- Calculate Volbot Strategy Indicators ---
            if self.config.get("volbot_enabled", True):
                df_calc = calculate_volatility_levels(df_calc, self.config, self.logger)
                df_calc = calculate_pivot_order_blocks(df_calc, self.config, self.logger)
                df_calc, self.all_bull_boxes, self.all_bear_boxes = manage_order_blocks(df_calc, self.config, self.logger)
            else:
                 self.logger.info("Volbot strategy calculation skipped (disabled in config).")

            # Assign the df with calculated indicators back
            self.df_processed = df_calc
            self.logger.debug(f"Finished indicator calculations for {self.symbol}. Processed DF columns: {self.df_processed.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators for {self.symbol}: {e}{RESET}", exc_info=True)
            # df_processed remains empty or incomplete


    def _update_latest_strategy_state(self):
        """Updates the strategy_state dict with the latest values from self.df_processed."""
        if self.df_processed.empty:
            self.logger.warning(f"Cannot update latest state: Processed DataFrame is empty for {self.symbol}.")
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None
            return

        try:
            latest = self.df_processed.iloc[-1]
            # Check if the last row contains any non-NaN values before proceeding
            if latest.isnull().all():
                self.logger.warning(f"{NEON_YELLOW}Cannot update latest state: Last row of processed DataFrame contains all NaNs for {self.symbol}.{RESET}")
                self.strategy_state = {}
                self.latest_active_bull_ob = None
                self.latest_active_bear_ob = None
                return

            updated_state = {}

            # --- Extract Core Price/Volume Data ---
            for col in ['open', 'high', 'low', 'close', 'volume']:
                 updated_state[col] = latest.get(col, np.nan)

            # --- Extract Risk Management ATR ---
            updated_state['atr_risk'] = latest.get('atr_risk', np.nan)

            # --- Extract Volbot Strategy State ---
            if self.config.get("volbot_enabled", True):
                volbot_cols = [
                    'trend_up_strat', 'trend_changed_strat',
                    'vol_trend_up_level_strat', 'vol_trend_dn_level_strat',
                    'lower_strat', 'upper_strat', # Include base levels too
                    'cum_vol_delta_since_change_strat', 'cum_vol_total_since_change_strat',
                    'last_trend_change_index_strat',
                    'ema1_strat', 'ema2_strat', 'atr_strat' # Include base indicators
                    # OB references are handled separately below
                ]
                for col in volbot_cols:
                    updated_state[col] = latest.get(col, np.nan)

                # Get the latest active OB references (these are dicts or None)
                self.latest_active_bull_ob = latest.get('active_bull_ob_strat', None)
                self.latest_active_bear_ob = latest.get('active_bear_ob_strat', None)
                # Add flags to state for easier checking
                updated_state['is_in_active_bull_ob'] = self.latest_active_bull_ob is not None
                updated_state['is_in_active_bear_ob'] = self.latest_active_bear_ob is not None

            self.strategy_state = updated_state
            # Filter out NaN/None for debug log brevity
            valid_state = {k: f"{v:.5f}" if isinstance(v, (float, np.float64)) else v
                           for k, v in self.strategy_state.items() if pd.notna(v)}
            self.logger.debug(f"Latest strategy state updated for {self.symbol}: {valid_state}")
            if self.latest_active_bull_ob: self.logger.debug(f"  Latest Active Bull OB: {self.latest_active_bull_ob['id']}")
            if self.latest_active_bear_ob: self.logger.debug(f"  Latest Active Bear OB: {self.latest_active_bear_ob['id']}")


        except IndexError:
            self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. Processed DataFrame might be unexpectedly empty.")
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None
        except Exception as e:
            self.logger.error(f"Unexpected error updating latest strategy state for {self.symbol}: {e}", exc_info=True)
            self.strategy_state = {}
            self.latest_active_bull_ob = None
            self.latest_active_bear_ob = None


    # --- Utility Functions (from livexy, adapted) ---
    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info, with fallbacks."""
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                if isinstance(price_precision_val, int): return price_precision_val
                elif isinstance(price_precision_val, (float, str)):
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0: return abs(tick_size.normalize().as_tuple().exponent)

            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_price_tick = Decimal(str(min_price_val))
                if min_price_tick > 0: return abs(min_price_tick.normalize().as_tuple().exponent)

            last_close = self.strategy_state.get("close")
            if last_close and pd.notna(last_close) and last_close > 0:
                s_close = format(Decimal(str(last_close)), 'f')
                if '.' in s_close: return len(s_close.split('.')[-1])
        except Exception: pass
        default_precision = 4
        self.logger.warning(f"Using default price precision {default_precision} for {self.symbol}.")
        return default_precision

    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info."""
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if price_precision_val is not None:
                if isinstance(price_precision_val, (float, str)):
                    tick_size = Decimal(str(price_precision_val))
                    if tick_size > 0: return tick_size
                elif isinstance(price_precision_val, int):
                    return Decimal('1e-' + str(price_precision_val))

            limits_info = self.market_info.get('limits', {})
            price_limits = limits_info.get('price', {})
            min_price_val = price_limits.get('min')
            if min_price_val is not None:
                min_tick_from_limit = Decimal(str(min_price_val))
                if min_tick_from_limit > 0: return min_tick_from_limit
        except Exception: pass
        price_precision_places = self.get_price_precision()
        fallback_tick = Decimal('1e-' + str(price_precision_places))
        self.logger.debug(f"Using fallback tick size based on precision places for {self.symbol}: {fallback_tick}")
        return fallback_tick


    # --- Signal Generation (Based on Volbot Rules) ---
    def generate_trading_signal(self) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on Volbot strategy rules."""
        if not self.strategy_state or not self.config.get("volbot_enabled", True):
            self.logger.debug("Cannot generate Volbot signal: State is empty or strategy disabled.")
            return "HOLD"

        # --- Extract latest state ---
        is_trend_up = self.strategy_state.get('trend_up_strat', None) # Boolean or None
        trend_changed = self.strategy_state.get('trend_changed_strat', False)
        is_in_bull_ob = self.strategy_state.get('is_in_active_bull_ob', False)
        is_in_bear_ob = self.strategy_state.get('is_in_active_bear_ob', False)
        # Need previous state to detect *entry* into OB
        # Get previous OB state (requires storing or accessing previous row)
        # Simple approach: Use current state and assume signal persists while inside
        # More complex: Check df_processed.iloc[-2] for previous OB state
        # Let's use the simpler approach for now: Signal if currently inside OB + trend matches
        # Consider adding check for `df_processed.iloc[-2]['active_bull/bear_ob_strat'] is None` later if needed

        # --- Get config flags ---
        signal_on_flip = self.config.get("volbot_signal_on_trend_flip", True)
        signal_on_ob = self.config.get("volbot_signal_on_ob_entry", True)

        # --- Determine Signal ---
        final_signal = "HOLD"

        if is_trend_up is None:
            self.logger.debug("Volbot signal: HOLD (Trend state undetermined)")
            return "HOLD"

        # 1. Trend Flip Signal
        if signal_on_flip and trend_changed:
            if is_trend_up:
                final_signal = "BUY"
                self.logger.info(f"{COLOR_UP}Volbot Signal: BUY (Trend flipped UP){RESET}")
            else:
                final_signal = "SELL"
                self.logger.info(f"{COLOR_DN}Volbot Signal: SELL (Trend flipped DOWN){RESET}")
            return final_signal # Prioritize trend flip signal

        # 2. Order Block Entry Signal (if no trend flip)
        if signal_on_ob:
            if is_trend_up and is_in_bull_ob:
                final_signal = "BUY"
                ob_id = self.latest_active_bull_ob['id'] if self.latest_active_bull_ob else 'N/A'
                self.logger.info(f"{COLOR_BULL_BOX}Volbot Signal: BUY (Price entered Bull OB {ob_id} during Uptrend){RESET}")
                return final_signal
            elif not is_trend_up and is_in_bear_ob:
                final_signal = "SELL"
                ob_id = self.latest_active_bear_ob['id'] if self.latest_active_bear_ob else 'N/A'
                self.logger.info(f"{COLOR_BEAR_BOX}Volbot Signal: SELL (Price entered Bear OB {ob_id} during Downtrend){RESET}")
                return final_signal

        # 3. Default to HOLD
        if final_signal == "HOLD":
             self.logger.info(f"Volbot Signal: HOLD (Trend: {'UP' if is_trend_up else 'DOWN'}, In Bull OB: {is_in_bull_ob}, In Bear OB: {is_in_bear_ob})")

        return final_signal


    # --- Risk Management Calculations (Using Risk ATR) ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculates potential take profit (TP) and initial stop loss (SL) levels
        based on entry price, RISK ATR, and configured multipliers. Uses Decimal precision.
        Returns (entry_price, take_profit, stop_loss), all as Decimal or None.
        """
        if signal not in ["BUY", "SELL"]:
            return entry_price, None, None

        # Use atr_risk from the strategy state
        atr_val_float = self.strategy_state.get("atr_risk")
        if atr_val_float is None or pd.isna(atr_val_float) or atr_val_float <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Risk ATR is invalid ({atr_val_float}).{RESET}")
            return entry_price, None, None
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Provided entry price is invalid ({entry_price}).{RESET}")
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_val_float))
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            price_precision = self.get_price_precision()
            min_tick = self.get_min_tick_size()

            take_profit, stop_loss = None, None

            if signal == "BUY":
                tp_offset, sl_offset = atr * tp_multiple, atr * sl_multiple
                take_profit_raw, stop_loss_raw = entry_price + tp_offset, entry_price - sl_offset
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
            elif signal == "SELL":
                tp_offset, sl_offset = atr * tp_multiple, atr * sl_multiple
                take_profit_raw, stop_loss_raw = entry_price - tp_offset, entry_price + sl_offset
                take_profit = (take_profit_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick
                stop_loss = (stop_loss_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick

            # Validation (as in livexy)
            min_sl_distance = min_tick
            if signal == "BUY" and stop_loss >= entry_price:
                adjusted_sl = (entry_price - min_sl_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}BUY SL ({stop_loss}) too close/above entry ({entry_price}). Adjusting to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl
            elif signal == "SELL" and stop_loss <= entry_price:
                adjusted_sl = (entry_price + min_sl_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}SELL SL ({stop_loss}) too close/below entry ({entry_price}). Adjusting to {adjusted_sl}.{RESET}")
                stop_loss = adjusted_sl

            min_tp_distance = min_tick
            if signal == "BUY" and take_profit <= entry_price:
                adjusted_tp = (entry_price + min_tp_distance).quantize(min_tick, rounding=ROUND_UP)
                self.logger.warning(f"{NEON_YELLOW}BUY TP ({take_profit}) non-profitable. Adjusting to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp
            elif signal == "SELL" and take_profit >= entry_price:
                adjusted_tp = (entry_price - min_tp_distance).quantize(min_tick, rounding=ROUND_DOWN)
                self.logger.warning(f"{NEON_YELLOW}SELL TP ({take_profit}) non-profitable. Adjusting to {adjusted_tp}.{RESET}")
                take_profit = adjusted_tp

            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(f"{NEON_RED}SL calculation resulted in zero/negative price ({stop_loss}). Cannot set SL.{RESET}")
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(f"{NEON_RED}TP calculation resulted in zero/negative price ({take_profit}). Cannot set TP.{RESET}")
                take_profit = None

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal} using Risk ATR ({atr:.{price_precision+1}f}): Entry={entry_price:.{price_precision}f}, TP={take_profit}, SL={stop_loss}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
            return entry_price, None, None

# --- Trading Logic Helper Functions (Mostly from livexy, unchanged unless noted) ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency, handling Bybit V5 structures."""
    # (Code from livexy - seems robust)
    lg = logger
    try:
        balance_info = None
        account_types_to_try = ['CONTRACT', 'UNIFIED']

        for acc_type in account_types_to_try:
            try:
                lg.debug(f"Fetching balance using params={{'type': '{acc_type}'}} for {currency}...")
                balance_info = exchange.fetch_balance(params={'type': acc_type})
                if currency in balance_info:
                    if balance_info[currency].get('free') is not None: break
                    elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                        balance_list = balance_info['info']['result']['list']
                        if isinstance(balance_list, list):
                            for account in balance_list:
                                if account.get('accountType') == acc_type:
                                    coin_list = account.get('coin')
                                    if isinstance(coin_list, list):
                                        if any(coin_data.get('coin') == currency for coin_data in coin_list): break # Found nested
                            if any(coin_data.get('coin') == currency for coin_data in coin_list): break # Break outer loop too
                    lg.debug(f"Currency '{currency}' found (type '{acc_type}'), but missing 'free' or V5 nested. Trying next.")
                    balance_info = None

                elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
                     balance_list = balance_info['info']['result']['list']
                     if isinstance(balance_list, list):
                         for account in balance_list:
                             if account.get('accountType') == acc_type:
                                 coin_list = account.get('coin')
                                 if isinstance(coin_list, list):
                                     if any(coin_data.get('coin') == currency for coin_data in coin_list): break
                         if any(coin_data.get('coin') == currency for coin_data in coin_list): break # Break outer loop too
                lg.debug(f"Currency '{currency}' not found in balance structure (type '{acc_type}'). Trying next.")
                balance_info = None

            except ccxt.ExchangeError as e:
                if "account type not support" in str(e).lower() or "invalid account type" in str(e).lower(): continue
                else: lg.warning(f"Exchange error fetching balance (type {acc_type}): {e}. Trying next."); continue
            except Exception as e: lg.warning(f"Unexpected error fetching balance (type {acc_type}): {e}. Trying next."); continue

        if not balance_info:
            lg.debug(f"Fetching balance using default parameters for {currency}...")
            try: balance_info = exchange.fetch_balance()
            except Exception as e: lg.error(f"{NEON_RED}Failed to fetch balance (default): {e}{RESET}"); return None

        available_balance_str = None
        if currency in balance_info and 'free' in balance_info[currency] and balance_info[currency]['free'] is not None:
            available_balance_str = str(balance_info[currency]['free'])
            lg.debug(f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}")
        elif 'info' in balance_info and 'result' in balance_info['info'] and isinstance(balance_info['info']['result'].get('list'), list):
            balance_list = balance_info['info']['result']['list']
            successful_acc_type = balance_info.get('params',{}).get('type')
            for account in balance_list:
                current_account_type = account.get('accountType')
                if successful_acc_type is None or current_account_type == successful_acc_type:
                    coin_list = account.get('coin')
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                            if coin_data.get('coin') == currency:
                                free = coin_data.get('availableToWithdraw', coin_data.get('availableBalance', coin_data.get('walletBalance')))
                                if free is not None: available_balance_str = str(free); break
                        if available_balance_str is not None: break
            if available_balance_str: lg.debug(f"Found balance via Bybit V5 nested: {available_balance_str} {currency} (Account: {current_account_type or 'N/A'})")
            else: lg.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]'.")
        elif 'free' in balance_info and currency in balance_info['free'] and balance_info['free'][currency] is not None:
             available_balance_str = str(balance_info['free'][currency])
             lg.debug(f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency}")

        if available_balance_str is None:
            total_balance = balance_info.get(currency, {}).get('total')
            if total_balance is not None:
                lg.warning(f"{NEON_YELLOW}Could not find 'free' balance for {currency}. Using 'total' ({total_balance}) as fallback.{RESET}")
                available_balance_str = str(total_balance)
            else:
                lg.error(f"{NEON_RED}Could not determine any balance for {currency}.{RESET}")
                lg.debug(f"Full balance_info: {balance_info}")
                return None

        try:
            final_balance = Decimal(available_balance_str)
            if final_balance >= 0:
                lg.info(f"Available {currency} balance: {final_balance:.4f}")
                return final_balance
            else: lg.error(f"Parsed balance for {currency} negative ({final_balance})."); return None
        except Exception as e: lg.error(f"Failed to convert balance '{available_balance_str}' to Decimal: {e}"); return None

    except ccxt.AuthenticationError as e: lg.error(f"{NEON_RED}Auth error fetching balance: {e}.{RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}"); return None
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True); return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision, limits, contract type, ensuring markets are loaded."""
    # (Code from livexy)
    lg = logger
    try:
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded/missing, reloading markets...")
            exchange.load_markets(reload=True)

        if symbol not in exchange.markets:
            lg.error(f"{NEON_RED}Market {symbol} still not found after reloading. Check symbol.{RESET}")
            return None

        market = exchange.market(symbol)
        if market:
            market_type = market.get('type', 'unknown')
            contract_type = "Linear" if market.get('linear') else "Inverse" if market.get('inverse') else "N/A"
            lg.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, Contract={contract_type}, "
                f"Precision(Price/Amount/Tick): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}/{market.get('precision', {}).get('tick', 'N/A')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            market['is_contract'] = market.get('contract', False) or market_type in ['swap', 'future']
            return market
        else:
            lg.error(f"{NEON_RED}Market dictionary not found for {symbol}.{RESET}")
            return None
    except ccxt.BadSymbol as e: lg.error(f"{NEON_RED}Symbol '{symbol}' not supported by {exchange.id}: {e}{RESET}"); return None
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error loading markets for {symbol}: {e}{RESET}"); return None
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error loading markets for {symbol}: {e}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error getting market info for {symbol}: {e}{RESET}", exc_info=True); return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    exchange: ccxt.Exchange,
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, SL distance, balance, and market constraints."""
    # (Code from livexy - should work as is)
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN_SYMBOL')
    quote_currency = market_info.get('quote', QUOTE_CURRENCY)
    base_currency = market_info.get('base', 'BASE')
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else base_currency

    if balance is None or balance <= 0: lg.error(f"Pos sizing failed ({symbol}): Invalid balance ({balance})."); return None
    if not (0 < risk_per_trade < 1): lg.error(f"Pos sizing failed ({symbol}): Invalid risk ({risk_per_trade})."); return None
    if initial_stop_loss_price is None or entry_price is None or entry_price <= 0: lg.error(f"Pos sizing failed ({symbol}): Invalid entry/SL."); return None
    if initial_stop_loss_price == entry_price: lg.error(f"Pos sizing failed ({symbol}): SL price equals entry price."); return None
    if initial_stop_loss_price <= 0: lg.warning(f"Pos sizing ({symbol}): Calculated SL price ({initial_stop_loss_price}) is zero/negative.")

    if 'limits' not in market_info or 'precision' not in market_info: lg.error(f"Pos sizing failed ({symbol}): Missing market limits/precision."); return None

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0: lg.error(f"Pos sizing failed ({symbol}): SL distance zero/negative."); return None

        contract_size_str = market_info.get('contractSize', '1')
        try: contract_size = Decimal(str(contract_size_str)); assert contract_size > 0
        except: lg.warning(f"Could not parse contract size '{contract_size_str}'. Defaulting to 1."); contract_size = Decimal('1')

        calculated_size = Decimal('0')
        if market_info.get('linear', True) or not is_contract:
            calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)
        else: # Inverse (Simplified Placeholder - Needs Verification for specific exchange)
            lg.warning(f"{NEON_YELLOW}Inverse contract {symbol}. Verify sizing logic! Assuming contractSize={contract_size} is Quote value.{RESET}")
            if entry_price > 0:
                 risk_per_contract_quote = sl_distance_per_unit * contract_size / entry_price
                 if risk_per_contract_quote > 0: calculated_size = risk_amount_quote / risk_per_contract_quote
                 else: lg.error(f"Pos sizing failed (inverse {symbol}): Risk per contract zero/negative."); return None
            else: lg.error(f"Pos sizing failed (inverse {symbol}): Entry price zero."); return None

        lg.info(f"Position Sizing ({symbol}): Balance={balance:.2f}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Dist={sl_distance_per_unit}")
        lg.info(f"  ContractSize={contract_size}, Initial Calc Size = {calculated_size:.8f} {size_unit}")

        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision_val = precision.get('amount')

        min_amount_str, max_amount_str = amount_limits.get('min'), amount_limits.get('max')
        min_cost_str, max_cost_str = cost_limits.get('min'), cost_limits.get('max')
        min_amount = Decimal(str(min_amount_str)) if min_amount_str is not None else Decimal('0')
        max_amount = Decimal(str(max_amount_str)) if max_amount_str is not None else Decimal('inf')
        min_cost = Decimal(str(min_cost_str)) if min_cost_str is not None else Decimal('0')
        max_cost = Decimal(str(max_cost_str)) if max_cost_str is not None else Decimal('inf')

        adjusted_size = calculated_size
        if adjusted_size < min_amount: lg.warning(f"{NEON_YELLOW}Calc size {adjusted_size:.8f} < min {min_amount}. Adjusting.{RESET}"); adjusted_size = min_amount
        elif adjusted_size > max_amount: lg.warning(f"{NEON_YELLOW}Calc size {adjusted_size:.8f} > max {max_amount}. Capping.{RESET}"); adjusted_size = max_amount

        current_cost = Decimal('0')
        if market_info.get('linear', True) or not is_contract: current_cost = adjusted_size * entry_price * contract_size
        else: contract_value_quote = contract_size; current_cost = adjusted_size * contract_value_quote
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Est Cost={current_cost:.4f}")

        if min_cost > 0 and current_cost < min_cost :
            lg.warning(f"{NEON_YELLOW}Est cost {current_cost:.4f} < min cost {min_cost}. Increasing size.{RESET}")
            required_size_for_min_cost = Decimal('0')
            if market_info.get('linear', True) or not is_contract:
                if entry_price > 0 and contract_size > 0: required_size_for_min_cost = min_cost / (entry_price * contract_size)
                else: lg.error("Cannot calc size for min cost (linear)."); return None
            else:
                contract_value_quote = contract_size
                if contract_value_quote > 0: required_size_for_min_cost = min_cost / contract_value_quote
                else: lg.error("Cannot calc size for min cost (inverse)."); return None
            lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
            if required_size_for_min_cost > max_amount: lg.error(f"{NEON_RED}Cannot meet min cost without exceeding max amount. Aborted.{RESET}"); return None
            elif required_size_for_min_cost < min_amount: lg.error(f"{NEON_RED}Conflicting limits: Min cost requires size < min amount. Aborted.{RESET}"); return None
            else: lg.info(f"  Adjusting size to meet min cost: {required_size_for_min_cost:.8f}"); adjusted_size = required_size_for_min_cost

        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(f"{NEON_YELLOW}Est cost {current_cost:.4f} > max cost {max_cost}. Reducing size.{RESET}")
            adjusted_size_for_max_cost = Decimal('0')
            if market_info.get('linear', True) or not is_contract:
                if entry_price > 0 and contract_size > 0: adjusted_size_for_max_cost = max_cost / (entry_price * contract_size)
                else: lg.error("Cannot calc size for max cost (linear)."); return None
            else:
                contract_value_quote = contract_size
                if contract_value_quote > 0: adjusted_size_for_max_cost = max_cost / contract_value_quote
                else: lg.error("Cannot calc size for max cost (inverse)."); return None
            lg.info(f"  Reduced size to meet max cost: {adjusted_size_for_max_cost:.8f}")
            if adjusted_size_for_max_cost < min_amount: lg.error(f"{NEON_RED}Reduced size {adjusted_size_for_max_cost} < min amount {min_amount}. Conflicting limits. Aborted.{RESET}"); return None
            else: adjusted_size = adjusted_size_for_max_cost

        final_size = Decimal('0')
        try:
            formatted_size_str = exchange.amount_to_precision(symbol, float(adjusted_size), padding_mode=exchange.TRUNCATE)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision (Truncated): {adjusted_size:.8f} -> {final_size} {size_unit}")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding.{RESET}")
            if amount_precision_val is not None:
                num_decimals, step_size = None, None
                if isinstance(amount_precision_val, int): num_decimals = amount_precision_val
                elif isinstance(amount_precision_val, (float, str)):
                    try: step_size = Decimal(str(amount_precision_val)); assert step_size > 0; num_decimals = abs(step_size.normalize().as_tuple().exponent)
                    except: pass
                if step_size is not None and step_size > 0: final_size = (adjusted_size // step_size) * step_size; lg.info(f"Applied manual step size ({step_size}): {adjusted_size:.8f} -> {final_size}")
                elif num_decimals is not None and num_decimals >= 0: rounding_factor = Decimal('1e-' + str(num_decimals)); final_size = adjusted_size.quantize(rounding_factor, rounding=ROUND_DOWN); lg.info(f"Applied manual precision ({num_decimals} decimals): {adjusted_size:.8f} -> {final_size}")
                else: lg.warning(f"Amount precision value ('{amount_precision_val}') invalid. Using limit-adjusted size: {adjusted_size:.8f}"); final_size = adjusted_size
            else: lg.warning(f"Amount precision undefined. Using limit-adjusted size: {adjusted_size:.8f}"); final_size = adjusted_size

        if final_size <= 0: lg.error(f"{NEON_RED}Pos size zero/negative ({final_size}) after adjustments. Aborted.{RESET}"); return None
        if final_size < min_amount and not math.isclose(float(final_size), float(min_amount), rel_tol=1e-9): lg.error(f"{NEON_RED}Final size {final_size} < min amount {min_amount}. Aborted.{RESET}"); return None

        final_cost = Decimal('0')
        if market_info.get('linear', True) or not is_contract: final_cost = final_size * entry_price * contract_size
        else: contract_value_quote = contract_size; final_cost = final_size * contract_value_quote
        if min_cost > 0 and final_cost < min_cost and not math.isclose(float(final_cost), float(min_cost), rel_tol=1e-6): lg.error(f"{NEON_RED}Final cost {final_cost:.4f} < min cost {min_cost}. Aborted.{RESET}"); return None

        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}")
        return final_size

    except KeyError as e: lg.error(f"{NEON_RED}Pos sizing error ({symbol}): Missing market key {e}. Market: {market_info}{RESET}"); return None
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error calculating position size for {symbol}: {e}{RESET}", exc_info=True); return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks for an open position, returning enhanced position dictionary or None."""
    # (Code from livexy - enhanced to parse SL/TP/TSL from 'info')
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: List[Dict] = []
        fetch_all = False
        try:
            params = {}
            if 'bybit' in exchange.id.lower():
                 market = None
                 try: market = exchange.market(symbol)
                 except: market = {'linear': True} # Assume linear on error
                 category = 'linear' if market.get('linear', True) else 'inverse'
                 params['category'] = category
                 lg.debug(f"Using params for fetch_positions: {params}")
            positions = exchange.fetch_positions(symbols=[symbol], params=params)
        except ccxt.ArgumentsRequired: fetch_all = True
        except ccxt.ExchangeError as e:
            no_pos_codes_v5 = [110025, 110021]
            if "symbol not found" in str(e).lower() or "instrument not found" in str(e).lower(): lg.warning(f"Symbol {symbol} not found fetching position: {e}. Assuming no position."); return None
            if hasattr(e, 'code') and e.code in no_pos_codes_v5: lg.info(f"No position found for {symbol} (Exchange code: {e.code})."); return None
            lg.error(f"Exchange error fetching single position for {symbol}: {e}", exc_info=True); return None
        except Exception as e: lg.error(f"Error fetching single position for {symbol}: {e}", exc_info=True); return None

        if fetch_all:
            try:
                all_positions = exchange.fetch_positions()
                positions = [p for p in all_positions if p.get('symbol') == symbol]
                lg.debug(f"Fetched {len(all_positions)} total, found {len(positions)} matching {symbol}.")
            except Exception as e: lg.error(f"Error fetching all positions for {symbol}: {e}", exc_info=True); return None

        active_position = None
        for pos in positions:
            pos_size_str = pos.get('contracts', pos.get('contractSize', pos.get('info', {}).get('size', pos.get('info', {}).get('positionAmt'))))
            if pos_size_str is None: continue
            try:
                position_size = Decimal(pos_size_str)
                size_threshold = Decimal('1e-9')
                if abs(position_size) > size_threshold:
                    active_position = pos; break
            except Exception: continue

        if active_position:
            side = active_position.get('side')
            size_decimal = Decimal('0')
            try: size_str_for_side = active_position.get('contracts', active_position.get('info',{}).get('size', '0')); size_decimal = Decimal(str(size_str_for_side))
            except: pass
            size_threshold = Decimal('1e-9')
            if side not in ['long', 'short']:
                info_side = active_position.get('info', {}).get('side', 'None')
                if info_side == 'Buy': side = 'long'
                elif info_side == 'Sell': side = 'short'
                elif size_decimal > size_threshold: side = 'long'
                elif size_decimal < -size_threshold: side = 'short'
                else: lg.warning(f"Position size {size_decimal} near zero, cannot determine side for {symbol}."); return None
                active_position['side'] = side

            info_dict = active_position.get('info', {})
            def get_valid_price_from_info(key: str) -> Optional[str]:
                val_str = info_dict.get(key)
                if val_str and str(val_str).strip() and Decimal(str(val_str).strip()) != 0:
                    try:
                        if Decimal(str(val_str)) > 0: return str(val_str)
                    except: pass
                return None

            if active_position.get('stopLossPrice') is None:
                sl_val = get_valid_price_from_info('stopLoss'); active_position['stopLossPrice'] = sl_val
            if active_position.get('takeProfitPrice') is None:
                tp_val = get_valid_price_from_info('takeProfit'); active_position['takeProfitPrice'] = tp_val

            active_position['trailingStopLoss'] = info_dict.get('trailingStop', '0')
            active_position['tslActivationPrice'] = info_dict.get('activePrice', '0')

            def format_log_price(key, price_val, precision=6):
                if price_val is None or price_val == '': return 'N/A'
                try:
                    d_price = Decimal(str(price_val))
                    if d_price > 0 or (d_price == 0 and ('trailingStop' in key or 'tslActivation' in key)): return f"{d_price:.{precision}f}"
                    elif d_price == 0 and key not in ['trailingStopLoss', 'tslActivationPrice']: return 'N/A'
                    else: return 'Invalid'
                except: return str(price_val)

            market_for_log = exchange.market(symbol) if symbol in exchange.markets else None
            log_precision = market_for_log['precision']['price'] if market_for_log else 6
            entry_price_str = active_position.get('entryPrice', info_dict.get('avgPrice'))
            entry_price = format_log_price('entryPrice', entry_price_str, log_precision)
            contracts_str = active_position.get('contracts', info_dict.get('size'))
            contracts = format_log_price('contracts', contracts_str, 8)
            liq_price = format_log_price('liquidationPrice', active_position.get('liquidationPrice'), log_precision)
            leverage_str = active_position.get('leverage', info_dict.get('leverage'))
            leverage = f"{Decimal(leverage_str):.1f}x" if leverage_str is not None else 'N/A'
            pnl_str = active_position.get('unrealizedPnl'); pnl = format_log_price('unrealizedPnl', pnl_str, 4)
            sl_price = format_log_price('stopLossPrice', active_position.get('stopLossPrice'), log_precision)
            tp_price = format_log_price('takeProfitPrice', active_position.get('takeProfitPrice'), log_precision)
            tsl_dist = format_log_price('trailingStopLoss', active_position.get('trailingStopLoss'), log_precision)
            tsl_act = format_log_price('tslActivationPrice', active_position.get('tslActivationPrice'), log_precision)
            is_tsl_active_log = False
            try:
                if Decimal(str(active_position.get('trailingStopLoss', '0'))) > 0: is_tsl_active_log = True
            except: pass

            logger.info(f"{NEON_GREEN}Active {side.upper()} position found for {symbol}:{RESET} "
                        f"Size={contracts}, Entry={entry_price}, Liq={liq_price}, "
                        f"Leverage={leverage}, PnL={pnl}, SL={sl_price}, TP={tp_price}, "
                        f"TSL Active: {is_tsl_active_log} (Dist={tsl_dist}/Act={tsl_act})")
            logger.debug(f"Full position details for {symbol}: {active_position}")
            return active_position
        else:
            logger.info(f"No active open position found for {symbol}.")
            return None

    except ccxt.AuthenticationError as e: lg.error(f"{NEON_RED}Auth error fetching positions: {e}{RESET}")
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error fetching positions: {e}{RESET}")
    except ccxt.ExchangeError as e:
        no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
        no_pos_codes = [110025, 110021]; err_str = str(e).lower(); err_code = getattr(e, 'code', None)
        if any(msg in err_str for msg in no_pos_msgs) or (err_code in no_pos_codes): lg.info(f"No open position found for {symbol} (Exchange error: {e})."); return None
        lg.error(f"{NEON_RED}Unhandled Exchange error fetching positions: {e} (Code: {err_code}){RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error fetching positions: {e}{RESET}", exc_info=True)
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    # (Code from livexy)
    lg = logger
    is_contract = market_info.get('is_contract', False)
    if not is_contract: lg.info(f"Leverage setting skipped for {symbol} (Not contract)."); return True
    if leverage <= 0: lg.warning(f"Leverage setting skipped ({symbol}): Invalid leverage ({leverage})."); return False
    if not exchange.has.get('setLeverage'): lg.error(f"{NEON_RED}Exchange {exchange.id} does not support set_leverage.{RESET}"); return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower():
            margin_mode = None # Can't easily detect mode without position here
            params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
            category = 'linear' if market_info.get('linear', True) else 'inverse'; params['category'] = category
            lg.debug(f"Using Bybit V5 params for set_leverage: {params}")

        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        verified = False
        if response is not None:
            if isinstance(response, dict):
                 ret_code = response.get('retCode', response.get('info', {}).get('retCode'))
                 if ret_code == 0: verified = True
                 elif ret_code == 110045: lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (Code {ret_code}).{RESET}"); verified = True
                 elif ret_code is not None: lg.warning(f"Set leverage returned non-zero retCode {ret_code}. Failure."); verified = False
                 else: verified = True # No retCode, assume success
            else: verified = True # Non-dict response, assume success
        else: lg.warning(f"Set leverage returned None. Assuming success, but verification uncertain."); verified = True

        if verified: lg.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}"); return True
        else: lg.error(f"{NEON_RED}Leverage setting failed for {symbol} based on response analysis.{RESET}"); return False

    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error setting leverage: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower(); bybit_code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Exchange error setting leverage: {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110045 or "leverage not modified" in err_str: lg.info(f"{NEON_YELLOW}Leverage already set to {leverage}x (Code {bybit_code}).{RESET}"); return True
        elif bybit_code == 110028 or "set margin mode first" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Set Margin Mode (Isolated/Cross) first.{RESET}")
        elif bybit_code == 110044 or "risk limit" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Leverage exceeds risk limit tier.{RESET}")
        elif bybit_code == 110009 or "position is in cross margin mode" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Cannot set leverage if symbol uses Cross Margin.{RESET}")
        elif bybit_code == 110013 or "parameter error" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Leverage value invalid for this symbol.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting leverage: {e}{RESET}", exc_info=True)
    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal, # Size calculated previously (Decimal)
    market_info: Dict,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """Places a market order using CCXT. Returns order dict on success, None on failure."""
    # (Code from livexy)
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'
    is_contract = market_info.get('is_contract', False)
    size_unit = "Contracts" if is_contract else market_info.get('base', '')

    try: amount_float = float(position_size); assert amount_float > 0
    except: lg.error(f"Trade aborted ({symbol} {side}): Invalid size {position_size}."); return None

    params = {'positionIdx': 0, 'reduceOnly': False}
    if 'bybit' in exchange.id.lower():
        category = 'linear' if market_info.get('linear', True) else 'inverse'; params['category'] = category

    lg.info(f"Attempting to place {side.upper()} {order_type} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {size_unit} | Params: {params}")

    try:
        order = exchange.create_order(symbol=symbol, type=order_type, side=side, amount=amount_float, price=None, params=params)
        order_id = order.get('id', 'N/A'); order_status = order.get('status', 'N/A')
        lg.info(f"{NEON_GREEN}Trade Placed Successfully! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response ({symbol} {side}): {order}")
        return order

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order: {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        try: balance = fetch_balance(exchange, QUOTE_CURRENCY, lg); lg.error(f"  Available {QUOTE_CURRENCY}: {balance}")
        except: pass
        if bybit_code == 110007: lg.error(f"{NEON_YELLOW} >> Hint (110007): Check balance, leverage. Cost ~ Size * Price / Leverage.{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters: {e}{RESET}")
        bybit_code = getattr(e, 'code', None)
        lg.error(f"  Size: {amount_float}, Type: {order_type}, Side: {side}, Params: {params}")
        lg.error(f"  Limits: Amount={market_info.get('limits',{}).get('amount')}, Cost={market_info.get('limits',{}).get('cost')}")
        lg.error(f"  Precision: Amount={market_info.get('precision',{}).get('amount')}, Price={market_info.get('precision',{}).get('price')}")
        if bybit_code == 10001 and "parameter error" in str(e).lower(): lg.error(f"{NEON_YELLOW} >> Hint (10001): Check precision/limits.{RESET}")
        elif bybit_code == 110017: lg.error(f"{NEON_YELLOW} >> Hint (110017): Size exceeds quantity limit.{RESET}")
        elif bybit_code == 110040: lg.error(f"{NEON_YELLOW} >> Hint (110040): Size below minimum.{RESET}")
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error placing order: {e}{RESET}")
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, 'code', None); err_str = str(e).lower()
        lg.error(f"{NEON_RED}Exchange error placing order: {e} (Code: {bybit_code}){RESET}")
        if bybit_code == 110007: lg.error(f"{NEON_YELLOW} >> Hint (110007): Insufficient balance/margin.{RESET}")
        elif bybit_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint (110043): Order cost exceeds limit/balance.{RESET}")
        elif bybit_code == 110044: lg.error(f"{NEON_YELLOW} >> Hint (110044): Position size exceeds risk limit.{RESET}")
        elif bybit_code == 110014: lg.error(f"{NEON_YELLOW} >> Hint (110014): Reduce-only failed (should be False).{RESET}")
        elif bybit_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint (110055): Position Index mismatch (should be 0 for One-Way).{RESET}")
        elif bybit_code == 10005 or "order link id exists" in err_str: lg.warning(f"{NEON_YELLOW}Duplicate order ID (Code {bybit_code}). Check position manually!{RESET}"); return None
        elif "risk limit can't be place order" in err_str: lg.error(f"{NEON_YELLOW} >> Hint: Blocked by risk limits.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error placing order: {e}{RESET}", exc_info=True)
    return None

def _set_position_protection(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict, logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None, take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, tsl_activation_price: Optional[Decimal] = None
) -> bool:
    """Internal helper to set SL/TP/TSL using Bybit V5 API via private_post."""
    # (Code from livexy - uses private_post for Bybit V5)
    lg = logger
    is_contract = market_info.get('is_contract', False)
    if not is_contract: lg.warning(f"Protection skipped ({symbol}): Not contract market."); return False
    if not position_info: lg.error(f"Cannot set protection ({symbol}): Missing position info."); return False

    pos_side = position_info.get('side')
    pos_size_str = position_info.get('contracts', position_info.get('info', {}).get('size'))
    if pos_side not in ['long', 'short'] or pos_size_str is None: lg.error(f"Cannot set protection ({symbol}): Invalid pos side/size."); return False

    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and isinstance(tsl_activation_price, Decimal) and tsl_activation_price >= 0
    if not has_sl and not has_tp and not has_tsl: lg.info(f"No valid protection params for {symbol}."); return True

    category = 'linear' if market_info.get('linear', True) else 'inverse'
    position_idx = 0
    try: pos_idx_val = position_info.get('info', {}).get('positionIdx'); position_idx = int(pos_idx_val) if pos_idx_val is not None else 0
    except: lg.warning(f"Could not parse positionIdx. Defaulting to {position_idx}.")

    params = {'category': category, 'symbol': market_info['id'], 'tpslMode': 'Full',
              'slTriggerBy': 'LastPrice', 'tpTriggerBy': 'LastPrice',
              'slOrderType': 'Market', 'tpOrderType': 'Market', 'positionIdx': position_idx}
    log_parts = [f"Attempting to set protection for {symbol} ({pos_side.upper()} Pos Idx: {position_idx}):"]
    tsl_added, sl_added, tp_added = False, False, False

    try:
        def format_price(price_decimal: Optional[Decimal]) -> Optional[str]:
            if price_decimal is None or price_decimal <= 0: return None
            try: return exchange.price_to_precision(symbol, float(price_decimal))
            except Exception as e: lg.warning(f"Price format failed: {e}. Skipping."); return None
        def format_distance(dist_decimal: Optional[Decimal]) -> Optional[str]:
             if dist_decimal is None or dist_decimal <= 0: return None
             try: return exchange.price_to_precision(symbol, float(abs(dist_decimal)))
             except Exception as e: lg.warning(f"Distance format failed: {e}. Skipping."); return None

        if has_tsl:
            formatted_tsl_dist = format_distance(trailing_stop_distance)
            formatted_act_price = '0' if tsl_activation_price == 0 else format_price(tsl_activation_price)
            if formatted_tsl_dist and formatted_act_price is not None:
                params['trailingStop'] = formatted_tsl_dist; params['activePrice'] = formatted_act_price
                log_parts.append(f"  TSL: Dist={formatted_tsl_dist}, Act={formatted_act_price}")
                tsl_added = True
            else: lg.error(f"Failed to format TSL params. Cannot set TSL."); has_tsl = False
        if has_sl and not tsl_added:
            formatted_sl = format_price(stop_loss_price)
            if formatted_sl: params['stopLoss'] = formatted_sl; log_parts.append(f"  Fixed SL: {formatted_sl}"); sl_added = True
            else: has_sl = False
        elif has_sl and tsl_added: lg.warning(f"TSL provided, ignoring fixed SL for {symbol}."); has_sl = False
        if has_tp:
            formatted_tp = format_price(take_profit_price)
            if formatted_tp: params['takeProfit'] = formatted_tp; log_parts.append(f"  Fixed TP: {formatted_tp}"); tp_added = True
            else: has_tp = False
    except Exception as fmt_err: lg.error(f"Error formatting protection params: {fmt_err}", exc_info=True); return False

    if not sl_added and not tp_added and not tsl_added:
        lg.warning(f"No valid protection params formatted for {symbol}. No API call."); return not (has_sl or has_tp or has_tsl)

    lg.info("\n".join(log_parts)); lg.debug(f"  API Call: private_post('/v5/position/set-trading-stop', params={params})")

    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', ''); ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            no_change_msg = "stoplosstakeprofittrailingstopwerenotmodified"
            processed_ret_msg = ret_msg.lower().replace(",", "").replace(".", "").replace("and", "").replace(" ", "")
            if processed_ret_msg == no_change_msg: lg.info(f"{NEON_YELLOW}Protection already set to target values (Exchange confirmation).{RESET}")
            elif "not modified" in ret_msg.lower(): lg.info(f"{NEON_YELLOW}Protection partially modified/already set. Resp: {ret_msg}{RESET}")
            else: lg.info(f"{NEON_GREEN}Protection set/updated successfully for {symbol}.{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed to set protection: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            if ret_code == 110043: lg.error(f"{NEON_YELLOW} >> Hint(110043): Check tpslMode, trigger prices? Ext: {ret_ext}.{RESET}")
            elif ret_code == 110025: lg.error(f"{NEON_YELLOW} >> Hint(110025): Position closed/changed? positionIdx mismatch?{RESET}")
            elif ret_code == 110055: lg.error(f"{NEON_YELLOW} >> Hint(110055): positionIdx mismatch with Position Mode?{RESET}")
            elif ret_code == 110013: lg.error(f"{NEON_YELLOW} >> Hint(110013): Parameter error. SL/TP/TSL value/tick? ActivePrice? Wrong side?{RESET}")
            elif ret_code == 110036: lg.error(f"{NEON_YELLOW} >> Hint(110036): TSL Activation price invalid?{RESET}")
            elif ret_code == 110086: lg.error(f"{NEON_YELLOW} >> Hint(110086): SL price cannot be same as TP price.{RESET}")
            elif "trailing stop value invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: TSL distance invalid?{RESET}")
            elif "sl price invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: Fixed SL invalid?{RESET}")
            elif "tp price invalid" in ret_msg.lower(): lg.error(f"{NEON_YELLOW} >> Hint: Fixed TP invalid?{RESET}")
            return False
    except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Network error setting protection: {e}{RESET}")
    except ccxt.ExchangeError as e: lg.error(f"{NEON_RED}Exchange error setting protection: {e}{RESET}")
    except KeyError as e: lg.error(f"{NEON_RED}Error setting protection: Missing key {e}.{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting protection: {e}{RESET}", exc_info=True)
    return False

def set_trailing_stop_loss(
    exchange: ccxt.Exchange, symbol: str, market_info: Dict, position_info: Dict,
    config: Dict[str, Any], logger: logging.Logger, take_profit_price: Optional[Decimal] = None
) -> bool:
    """Calculates TSL params and calls helper to set TSL (and optional TP)."""
    # (Code from livexy)
    lg = logger
    if not config.get("enable_trailing_stop", False): lg.info(f"TSL disabled in config ({symbol})."); return False

    try:
        callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005)))
        activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003)))
    except Exception as e: lg.error(f"{NEON_RED}Invalid TSL config format ({symbol}): {e}."); return False
    if callback_rate <= 0: lg.error(f"{NEON_RED}Invalid callback_rate ({callback_rate})."); return False
    if activation_percentage < 0: lg.error(f"{NEON_RED}Invalid activation_percentage ({activation_percentage})."); return False

    try:
        entry_price_str = position_info.get('entryPrice', position_info.get('info', {}).get('avgPrice'))
        side = position_info.get('side')
        if entry_price_str is None or side not in ['long', 'short']: raise ValueError("Missing entryPrice or side")
        entry_price = Decimal(str(entry_price_str)); assert entry_price > 0
    except Exception as e: lg.error(f"{NEON_RED}Error parsing position info for TSL ({symbol}): {e}."); return False

    try:
        # Use a temporary analyzer instance for utility methods
        dummy_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']); dummy_df.index.name = 'timestamp'
        temp_analyzer = TradingAnalyzer(df=dummy_df, logger=lg, config=config, market_info=market_info)
        price_precision = temp_analyzer.get_price_precision()
        min_tick_size = temp_analyzer.get_min_tick_size()

        activation_price = None
        if activation_percentage > 0:
            activation_offset = entry_price * activation_percentage
            if side == 'long':
                raw_activation = entry_price + activation_offset
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
                if activation_price <= entry_price: activation_price = (entry_price + min_tick_size).quantize(min_tick_size, rounding=ROUND_UP)
            else:
                raw_activation = entry_price - activation_offset
                activation_price = (raw_activation / min_tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * min_tick_size
                if activation_price >= entry_price: activation_price = (entry_price - min_tick_size).quantize(min_tick_size, rounding=ROUND_DOWN)
        else: activation_price = Decimal('0')
        if activation_price is None or activation_price < 0: lg.error(f"{NEON_RED}Invalid TSL activation price ({activation_price})."); return False

        trailing_distance_raw = entry_price * callback_rate
        trailing_distance = (trailing_distance_raw / min_tick_size).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick_size
        if trailing_distance < min_tick_size: lg.warning(f"{NEON_YELLOW}TSL dist {trailing_distance} < min tick {min_tick_size}. Adjusting."); trailing_distance = min_tick_size
        elif trailing_distance <= 0: lg.error(f"{NEON_RED}Invalid TSL distance ({trailing_distance})."); return False

        lg.info(f"Calculated TSL Params for {symbol} ({side.upper()}):")
        lg.info(f"  Entry: {entry_price:.{price_precision}f}, Callback: {callback_rate:.3%}, Activation Pct: {activation_percentage:.3%}")
        lg.info(f"  => Activation Price (API): {activation_price:.{price_precision}f if activation_price > 0 else '0 (Immediate)'}")
        lg.info(f"  => Trailing Distance (API): {trailing_distance:.{price_precision}f}")
        if take_profit_price and take_profit_price > 0: lg.info(f"  Take Profit Price: {take_profit_price:.{price_precision}f} (Will be set)")

        return _set_position_protection(
            exchange=exchange, symbol=symbol, market_info=market_info, position_info=position_info, logger=lg,
            stop_loss_price=None, # Don't set fixed SL with TSL
            take_profit_price=take_profit_price if isinstance(take_profit_price, Decimal) and take_profit_price > 0 else None,
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error calculating/setting TSL: {e}{RESET}", exc_info=True); return False

# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a single symbol using Volbot strategy and executes/manages trades."""

    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info & Data ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info: lg.error(f"Failed to get market info for {symbol}. Skipping cycle."); return

    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval: lg.error(f"Invalid interval '{config['interval']}'."); return

    # Determine required kline history for Volbot + Risk ATR
    strat_lookbacks = [
        config.get("volbot_length", DEFAULT_VOLBOT_LENGTH),
        config.get("volbot_atr_length", DEFAULT_VOLBOT_ATR_LENGTH),
        config.get("volbot_volume_percentile_lookback", DEFAULT_VOLBOT_VOLUME_PERCENTILE_LOOKBACK),
        config.get("volbot_pivot_left_len_h", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_H) + config.get("volbot_pivot_right_len_h", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_H),
        config.get("volbot_pivot_left_len_l", DEFAULT_VOLBOT_PIVOT_LEFT_LEN_L) + config.get("volbot_pivot_right_len_l", DEFAULT_VOLBOT_PIVOT_RIGHT_LEN_L),
    ]
    risk_lookback = config.get("atr_period", DEFAULT_ATR_PERIOD)
    kline_limit = max(strat_lookbacks + [risk_lookback]) + 100 # Add buffer

    lg.info(f"Fetching {kline_limit} klines for {symbol} ({ccxt_interval})...")
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg)
    if klines_df.empty or len(klines_df) < 50: # Need a reasonable minimum history
        lg.error(f"Failed to fetch sufficient kline data ({len(klines_df)}). Skipping cycle."); return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"Failed to fetch current price. Using last close from klines.")
        try:
            last_close_val = klines_df['close'].iloc[-1]
            if pd.notna(last_close_val) and last_close_val > 0: current_price = Decimal(str(last_close_val))
            else: lg.error(f"Last close price ({last_close_val}) invalid. Cannot proceed."); return
        except: lg.error(f"Error getting last close. Cannot proceed."); return

    # --- 2. Analyze Data & Generate Signal ---
    analyzer = TradingAnalyzer(df=klines_df, logger=lg, config=config, market_info=market_info)
    if not analyzer.strategy_state: lg.error(f"Indicator/Strategy calculation failed. Skipping cycle."); return

    signal = analyzer.generate_trading_signal() # Uses Volbot rules
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(current_price, signal) # Uses Risk ATR
    price_precision = analyzer.get_price_precision()
    min_tick_size = analyzer.get_min_tick_size()
    current_risk_atr_float = analyzer.strategy_state.get("atr_risk")
    current_strat_atr_float = analyzer.strategy_state.get("atr_strat") # Get strategy ATR too

    # --- 3. Log Analysis Summary ---
    # Signal logging happens inside generate_trading_signal
    lg.info(f"Current Price: {current_price:.{price_precision}f}")
    lg.info(f"Risk ATR ({config.get('atr_period')}): {f'{current_risk_atr_float:.{price_precision+1}f}' if current_risk_atr_float and pd.notna(current_risk_atr_float) else 'N/A'}")
    lg.info(f"Strategy ATR ({config.get('volbot_atr_length')}): {f'{current_strat_atr_float:.{price_precision+1}f}' if current_strat_atr_float and pd.notna(current_strat_atr_float) else 'N/A'}")
    # Log key Volbot levels
    if analyzer.config.get("volbot_enabled"):
        trend_state = "UP" if analyzer.strategy_state.get('trend_up_strat') else "DOWN" if analyzer.strategy_state.get('trend_up_strat') is False else "N/A"
        lg.info(f"Volbot Trend: {trend_state}")
        if trend_state == "UP": lg.info(f"  Vol-Adj Support: {analyzer.strategy_state.get('vol_trend_up_level_strat', 'N/A'):.{price_precision}f}")
        if trend_state == "DOWN": lg.info(f"  Vol-Adj Resist: {analyzer.strategy_state.get('vol_trend_dn_level_strat', 'N/A'):.{price_precision}f}")
        if analyzer.latest_active_bull_ob: lg.info(f"  In Active Bull OB: {analyzer.latest_active_bull_ob['id']} ({analyzer.latest_active_bull_ob['bottom']:.{price_precision}f} - {analyzer.latest_active_bull_ob['top']:.{price_precision}f})")
        if analyzer.latest_active_bear_ob: lg.info(f"  In Active Bear OB: {analyzer.latest_active_bear_ob['id']} ({analyzer.latest_active_bear_ob['bottom']:.{price_precision}f} - {analyzer.latest_active_bear_ob['top']:.{price_precision}f})")

    lg.info(f"Potential Initial SL (for new trade): {sl_potential if sl_potential else 'N/A'}")
    lg.info(f"Potential Initial TP (for new trade): {tp_potential if tp_potential else 'N/A'}")
    tsl_enabled = config.get('enable_trailing_stop'); be_enabled = config.get('enable_break_even')
    lg.info(f"Trailing Stop: {'Enabled' if tsl_enabled else 'Disabled'} | Break Even: {'Enabled' if be_enabled else 'Disabled'}")

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.debug(f"Trading disabled. Analysis complete."); cycle_end_time = time.monotonic(); lg.debug(f"---== Cycle End ({cycle_end_time - cycle_start_time:.2f}s) ==---"); return

    open_position = get_open_position(exchange, symbol, lg)

    # --- Scenario 1: No Open Position ---
    if open_position is None:
        if signal in ["BUY", "SELL"]:
            lg.info(f"*** {signal} Signal & No Open Position: Initiating Trade Sequence ***")
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            if balance is None: lg.error(f"Trade Aborted ({signal}): Failed fetch balance."); return
            if balance <= 0: lg.error(f"Trade Aborted ({signal}): Insufficient balance ({balance})."); return
            if sl_potential is None: lg.error(f"Trade Aborted ({signal}): Potential SL calculation failed (Risk ATR valid?)."); return

            if market_info.get('is_contract', False):
                leverage = int(config.get("leverage", 1))
                if leverage > 0:
                    if not set_leverage_ccxt(exchange, symbol, leverage, market_info, lg):
                        lg.error(f"Trade Aborted ({signal}): Failed set leverage {leverage}x."); return
                else: lg.warning(f"Leverage setting skipped: Config leverage <= 0.")
            else: lg.info(f"Leverage setting skipped (Spot).")

            position_size = calculate_position_size(balance, config["risk_per_trade"], sl_potential, current_price, market_info, exchange, lg)
            if position_size is None or position_size <= 0: lg.error(f"Trade Aborted ({signal}): Invalid position size ({position_size})."); return

            lg.info(f"==> Placing {signal} market order | Size: {position_size} <==")
            trade_order = place_trade(exchange, symbol, signal, position_size, market_info, lg)

            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                lg.info(f"Order {order_id} placed. Waiting for position confirmation...")
                position_confirm_delay = 10; time.sleep(position_confirm_delay)
                lg.info(f"Confirming position after order {order_id}...")
                confirmed_position = get_open_position(exchange, symbol, lg)

                if confirmed_position:
                    try:
                        entry_price_actual_str = confirmed_position.get('entryPrice', confirmed_position.get('info', {}).get('avgPrice'))
                        pos_size_actual_str = confirmed_position.get('contracts', confirmed_position.get('info', {}).get('size'))
                        entry_price_actual, pos_size_actual = Decimal('0'), Decimal('0'); valid_entry = False
                        if entry_price_actual_str and pos_size_actual_str:
                            try:
                                entry_price_actual = Decimal(str(entry_price_actual_str)); pos_size_actual = Decimal(str(pos_size_actual_str))
                                if entry_price_actual > 0 and abs(pos_size_actual) > 0: valid_entry = True
                                else: lg.error(f"Confirmed position invalid entry/size: {entry_price_actual}/{pos_size_actual}.")
                            except Exception as parse_err: lg.error(f"Error parsing confirmed entry/size: {parse_err}")
                        else: lg.error("Confirmed position missing entry/size.")

                        if valid_entry:
                            lg.info(f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_price_actual:.{price_precision}f}, Size: {pos_size_actual}{RESET}")
                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(entry_price_actual, signal) # Recalc SL/TP
                            protection_set_success = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(f"Setting TSL (TP target: {tp_actual})...")
                                protection_set_success = set_trailing_stop_loss(exchange, symbol, market_info, confirmed_position, config, lg, tp_actual)
                            else:
                                lg.info(f"Setting Fixed SL ({sl_actual}) and TP ({tp_actual})...")
                                if sl_actual or tp_actual:
                                    protection_set_success = _set_position_protection(exchange, symbol, market_info, confirmed_position, lg, sl_actual, tp_actual)
                                else: lg.warning(f"Fixed SL/TP calc failed. No fixed protection set.")

                            if protection_set_success: lg.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({signal}) ===")
                            else: lg.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION ({signal}) ==="); lg.warning(f"{NEON_YELLOW}Position open without protection! Manual check needed!{RESET}")
                        else: lg.error(f"{NEON_RED}Trade placed, but confirmed position data invalid. No protection set.{RESET}"); lg.warning(f"{NEON_YELLOW}Manual check needed!{RESET}")
                    except Exception as post_trade_err: lg.error(f"{NEON_RED}Error during post-trade processing: {post_trade_err}{RESET}", exc_info=True); lg.warning(f"{NEON_YELLOW}Manual check needed!{RESET}")
                else:
                    lg.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM position after {position_confirm_delay}s!{RESET}")
                    lg.warning(f"{NEON_YELLOW}Order might have failed/rejected/delayed. No protection set. Manual investigation needed!{RESET}")
                    try: order_status = exchange.fetch_order(order_id, symbol); lg.info(f"Status of order {order_id}: {order_status}")
                    except Exception as fetch_order_err: lg.warning(f"Could not fetch status for order {order_id}: {fetch_order_err}")
            else: lg.error(f"{NEON_RED}=== TRADE EXECUTION FAILED ({signal}). See logs. ===")
        else: lg.info(f"Signal is HOLD, no position. No action.")

    # --- Scenario 2: Existing Open Position Found ---
    else:
        pos_side = open_position.get('side', 'unknown')
        pos_size_str = open_position.get('contracts', open_position.get('info',{}).get('size', 'N/A'))
        entry_price_str = open_position.get('entryPrice', open_position.get('info', {}).get('avgPrice', 'N/A'))
        current_sl_str = open_position.get('stopLossPrice'); current_tp_str = open_position.get('takeProfitPrice')
        tsl_distance_str = open_position.get('trailingStopLoss'); is_tsl_active = False
        if tsl_distance_str:
            try:
                 if Decimal(str(tsl_distance_str)) > 0: is_tsl_active = True
            except: pass
        lg.info(f"Existing {pos_side.upper()} position found. Size: {pos_size_str}, Entry: {entry_price_str}, SL: {current_sl_str or 'N/A'}, TP: {current_tp_str or 'N/A'}, TSL Active: {is_tsl_active}")

        exit_signal_triggered = (pos_side == 'long' and signal == "SELL") or (pos_side == 'short' and signal == "BUY")
        if exit_signal_triggered:
            lg.warning(f"{NEON_YELLOW}*** EXIT Signal ({signal}) Opposes Existing {pos_side} Position. Closing Position. ***{RESET}")
            try:
                close_side = 'sell' if pos_side == 'long' else 'buy'
                size_to_close_str = open_position.get('contracts', open_position.get('info',{}).get('size'))
                if size_to_close_str is None: raise ValueError("Cannot determine size to close.")
                size_to_close = abs(Decimal(str(size_to_close_str))); assert size_to_close > 0
                close_amount_float = float(size_to_close)
                close_params = {'positionIdx': open_position.get('info', {}).get('positionIdx', 0), 'reduceOnly': True}
                if 'bybit' in exchange.id.lower(): category = 'linear' if market_info.get('linear', True) else 'inverse'; close_params['category'] = category
                lg.info(f"Placing {close_side.upper()} MARKET order (reduceOnly=True) for {close_amount_float} {symbol}...")
                lg.debug("Skipping explicit SL/TP cancellation before closing...")
                close_order = exchange.create_order(symbol=symbol, type='market', side=close_side, amount=close_amount_float, params=close_params)
                order_id = close_order.get('id', 'N/A'); lg.info(f"{NEON_GREEN}Position CLOSE order placed successfully. Order ID: {order_id}{RESET}")
            except ValueError as ve: lg.error(f"{NEON_RED}Error preparing close order: {ve}{RESET}")
            except ccxt.InsufficientFunds as e: lg.error(f"{NEON_RED}Error closing (InsufficientFunds?): {e}{RESET}")
            except ccxt.InvalidOrder as e: lg.error(f"{NEON_RED}Error closing (InvalidOrder): {e}{RESET}"); bybit_code = getattr(e, 'code', None); if bybit_code == 110014: lg.error(f"{NEON_YELLOW} >> Hint(110014): Reduce-only failed. Position changed?{RESET}")
            except ccxt.NetworkError as e: lg.error(f"{NEON_RED}Error closing (NetworkError): {e}. Manual check!{RESET}")
            except ccxt.ExchangeError as e: bybit_code = getattr(e, 'code', None); lg.error(f"{NEON_RED}Error closing (ExchangeError): {e} (Code: {bybit_code}){RESET}"); if bybit_code == 110025: lg.warning(f"{NEON_YELLOW} >> Hint(110025): Position already closed?{RESET}")
            except Exception as e: lg.error(f"{NEON_RED}Unexpected error closing position: {e}{RESET}", exc_info=True)

        elif signal == "HOLD" or (signal == "BUY" and pos_side == 'long') or (signal == "SELL" and pos_side == 'short'):
            lg.info(f"Signal ({signal}) allows holding existing {pos_side} position.")
            if config.get("enable_break_even", False) and not is_tsl_active:
                lg.debug(f"Checking Break-Even conditions...")
                try:
                    if not entry_price_str or entry_price_str == 'N/A': raise ValueError("Missing entry price")
                    entry_price = Decimal(str(entry_price_str)); assert entry_price > 0
                    if current_risk_atr_float is None or pd.isna(current_risk_atr_float) or current_risk_atr_float <= 0: raise ValueError("Invalid Risk ATR")
                    current_atr_decimal = Decimal(str(current_risk_atr_float))
                    profit_target_atr_multiple = Decimal(str(config.get("break_even_trigger_atr_multiple", 1.0)))
                    offset_ticks = int(config.get("break_even_offset_ticks", 2))
                    price_diff = (current_price - entry_price) if pos_side == 'long' else (entry_price - current_price)
                    profit_target_price_diff = profit_target_atr_multiple * current_atr_decimal
                    lg.debug(f"BE Check: Price Diff={price_diff:.{price_precision}f}, Target Diff={profit_target_price_diff:.{price_precision}f}")

                    if price_diff >= profit_target_price_diff:
                        be_stop_price = None; tick_offset = min_tick_size * offset_ticks
                        if pos_side == 'long': be_stop_price = (entry_price + tick_offset).quantize(min_tick_size, rounding=ROUND_UP)
                        else: be_stop_price = (entry_price - tick_offset).quantize(min_tick_size, rounding=ROUND_DOWN)

                        current_sl_price = None
                        if current_sl_str and str(current_sl_str).strip() not in ['0', '0.0', 'N/A']:
                            try: current_sl_price = Decimal(str(current_sl_str)); assert current_sl_price > 0
                            except: current_sl_price = None
                        update_be = False
                        if be_stop_price is not None and be_stop_price > 0:
                            if current_sl_price is None: update_be = True; lg.info(f"BE Triggered. No current SL.")
                            elif pos_side == 'long' and be_stop_price > current_sl_price: update_be = True; lg.info(f"BE Triggered. Current SL {current_sl_price} < Target {be_stop_price}.")
                            elif pos_side == 'short' and be_stop_price < current_sl_price: update_be = True; lg.info(f"BE Triggered. Current SL {current_sl_price} > Target {be_stop_price}.")
                            else: lg.debug(f"BE Triggered, but current SL ({current_sl_price}) already >= target ({be_stop_price}).")
                        else: lg.warning(f"Calculated BE Stop Price ({be_stop_price}) invalid.")

                        if update_be:
                            lg.warning(f"{NEON_PURPLE}*** Moving Stop Loss to Break-Even at {be_stop_price} ***{RESET}")
                            current_tp_price = None
                            if current_tp_str and str(current_tp_str).strip() not in ['0', '0.0', 'N/A']:
                                try: current_tp_price_dec = Decimal(str(current_tp_str)); assert current_tp_price_dec > 0; current_tp_price = current_tp_price_dec
                                except: pass
                            success = _set_position_protection(exchange, symbol, market_info, open_position, lg, be_stop_price, current_tp_price)
                            if success: lg.info(f"{NEON_GREEN}Break-Even SL set/updated.{RESET}")
                            else: lg.error(f"{NEON_RED}Failed to set Break-Even SL.{RESET}")
                    else: lg.debug(f"BE profit target not reached.")
                except ValueError as ve: lg.warning(f"Skipping BE check: {ve}")
                except Exception as be_err: lg.error(f"{NEON_RED}Error during break-even check: {be_err}{RESET}", exc_info=True)
            elif is_tsl_active: lg.info(f"BE check skipped: TSL active.")
            elif not config.get("enable_break_even", False): lg.debug(f"BE check skipped: Disabled.")
            lg.debug(f"No other position management actions taken.")

    cycle_end_time = time.monotonic()
    lg.debug(f"---== Cycle End ({cycle_end_time - cycle_start_time:.2f}s) ==---")


def main() -> None:
    """Main function to initialize the Volbot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY, console_log_level

    setup_logger("init"); init_logger = logging.getLogger("init")
    console_handler = None
    for handler in init_logger.handlers:
        if isinstance(handler, logging.StreamHandler): console_handler = handler; break
    if console_handler: console_handler.setLevel(console_log_level)

    init_logger.info(f"--- Starting Volbot Trading Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    CONFIG = load_config(CONFIG_FILE)
    QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")
    init_logger.info(f"Config loaded. Quote Currency: {QUOTE_CURRENCY}")
    init_logger.info(f"CCXT Version: {ccxt.__version__}, Pandas: {pd.__version__}, Pandas TA: {ta.version() if callable(ta.version) else getattr(ta, '__version__', 'N/A')}")

    if CONFIG.get("enable_trading"):
        init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
        if CONFIG.get("use_sandbox"): init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet).{RESET}")
        else: init_logger.warning(f"{NEON_RED}!!! USING REAL MONEY ENVIRONMENT !!!{RESET}")
        init_logger.warning(f"Settings: Risk={CONFIG.get('risk_per_trade', 0)*100:.2f}%, Lev={CONFIG.get('leverage', 0)}x, TSL={'ON' if CONFIG.get('enable_trailing_stop') else 'OFF'}, BE={'ON' if CONFIG.get('enable_break_even') else 'OFF'}")
        try: input(f">>> Review settings. Press Enter to continue, Ctrl+C to abort... "); init_logger.info("User acknowledged.")
        except KeyboardInterrupt: init_logger.info("Aborted."); return
    else: init_logger.info(f"{NEON_YELLOW}Trading is disabled. Analysis-only mode.{RESET}")

    init_logger.info("Initializing exchange...")
    exchange = initialize_exchange(init_logger)
    if not exchange: init_logger.critical(f"Failed to initialize exchange. Exiting."); return
    init_logger.info(f"Exchange {exchange.id} initialized.")

    target_symbol = None; market_info = None
    while True:
        try:
            symbol_input_raw = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT): {RESET}").strip()
            if not symbol_input_raw: continue
            symbol_input = symbol_input_raw.upper().replace('-', '/')
            init_logger.info(f"Validating symbol '{symbol_input}'...")
            market_info = get_market_info(exchange, symbol_input, init_logger)
            if market_info: target_symbol = market_info['symbol']; break
            else:
                variations = []; base_curr = ""; quote_curr = QUOTE_CURRENCY
                if '/' in symbol_input: parts = symbol_input.split('/'); base_curr = parts[0]; if len(parts) > 1: quote_curr = parts[1].split(':')[0]
                else: if symbol_input.endswith(quote_curr): base_curr = symbol_input[:-len(quote_curr)]; else: base_curr = symbol_input
                if base_curr: variations.extend([f"{base_curr}/{quote_curr}", f"{base_curr}/{quote_curr}:{quote_curr}"])
                found_variation = False
                if variations:
                    init_logger.info(f"Trying variations: {variations}")
                    for sym_var in variations:
                        market_info = get_market_info(exchange, sym_var, init_logger)
                        if market_info: target_symbol = market_info['symbol']; found_variation = True; break
                if found_variation: break
                else: init_logger.error(f"{NEON_RED}Symbol '{symbol_input_raw}' and variations not validated.{RESET}")
        except Exception as e: init_logger.error(f"Error validating symbol: {e}", exc_info=True)

    selected_interval = None
    while True:
        interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}] (default: {CONFIG['interval']}): {RESET}").strip()
        if not interval_input: interval_input = CONFIG['interval']; init_logger.info(f"Using default interval: {interval_input}")
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input; CONFIG["interval"] = selected_interval; ccxt_tf = CCXT_INTERVAL_MAP[selected_interval]
            init_logger.info(f"Using interval: {selected_interval} (CCXT: {ccxt_tf})"); break
        else: init_logger.error(f"{NEON_RED}Invalid interval: '{interval_input}'.{RESET}")

    symbol_logger = setup_logger(target_symbol)
    if console_handler:
        for handler in symbol_logger.handlers:
             if isinstance(handler, logging.StreamHandler): handler.setLevel(console_log_level); break

    symbol_logger.info(f"---=== Starting Trading Loop for {target_symbol} ({CONFIG['interval']}) ===---")
    symbol_logger.info(f"Config: Risk={CONFIG['risk_per_trade']:.2%}, Lev={CONFIG['leverage']}x, TSL={'ON' if CONFIG['enable_trailing_stop'] else 'OFF'}, BE={'ON' if CONFIG['enable_break_even'] else 'OFF'}, Trading={'ENABLED' if CONFIG['enable_trading'] else 'DISABLED'}")

    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f">>> New Loop Cycle Starting")
            try:
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger)
            except ccxt.RateLimitExceeded as e:
                wait_time = int(e.args[0].split(' ')[-2]) if 'try again in' in e.args[0] else 60; symbol_logger.warning(f"Rate limit exceeded: {e}. Waiting {wait_time}s..."); time.sleep(wait_time)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e: symbol_logger.error(f"{NEON_RED}Network error: {e}. Waiting {RETRY_DELAY_SECONDS*3}s..."); time.sleep(RETRY_DELAY_SECONDS * 3)
            except ccxt.AuthenticationError as e: symbol_logger.critical(f"{NEON_RED}CRITICAL: Auth Error: {e}. Stopping bot."); break
            except ccxt.ExchangeNotAvailable as e: symbol_logger.error(f"Exchange unavailable: {e}. Waiting 60s..."); time.sleep(60)
            except ccxt.OnMaintenance as e: symbol_logger.error(f"Exchange maintenance: {e}. Waiting 5 mins..."); time.sleep(300)
            except Exception as loop_error: symbol_logger.error(f"{NEON_RED}Uncaught error in loop: {loop_error}{RESET}", exc_info=True); symbol_logger.info("Continuing after 15s delay..."); time.sleep(15)

            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.debug(f"<<< Loop cycle finished in {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s...")
            if sleep_time > 0: time.sleep(sleep_time)
    except KeyboardInterrupt: symbol_logger.info("Keyboard interrupt. Shutting down...")
    except Exception as critical_error: init_logger.critical(f"CRITICAL unhandled error: {critical_error}", exc_info=True); if 'symbol_logger' in locals(): symbol_logger.critical(f"CRITICAL unhandled error: {critical_error}", exc_info=True)
    finally:
        shutdown_msg = f"--- Volbot for {target_symbol if target_symbol else 'N/A'} Stopping ---"
        init_logger.info(shutdown_msg);
        if 'symbol_logger' in locals(): symbol_logger.info(shutdown_msg)
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try: init_logger.info("Closing exchange connection..."); exchange.close(); init_logger.info("Connection closed.")
            except Exception as close_err: init_logger.error(f"Error closing connection: {close_err}")
        final_msg = "Bot stopped."
        init_logger.info(final_msg);
        if 'symbol_logger' in locals(): symbol_logger.info(final_msg)
        logging.shutdown()

if __name__ == "__main__":
    # This block writes the script content to 'volbot.py'
    output_filename = "volbot.py" # Set the desired output filename
    try:
        with open(__file__, 'r', encoding='utf-8') as current_file:
            script_content = current_file.read()
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            header = f"# {output_filename}\n# Incorporates Volumatic Trend + Order Block strategy into the livexy trading framework.\n\n"
            import re
            # Remove old filename headers if they exist
            script_content = re.sub(r'^# (livexy|volbot)\.py.*\n(# .*\n)*', '', script_content, flags=re.MULTILINE)
            output_file.write(header + script_content)
        print(f"Strategy incorporated. Full script written to {output_filename}")
        # Now run the main logic from the newly written file (or directly)
        main()
    except Exception as e:
        print(f"Error writing script to {output_filename} or running main: {e}")
        # Attempt to run main anyway if writing failed
        try: main()
        except Exception as main_e: print(f"Error running main after file write failure: {main_e}")

