# lwx.py
# Enhanced version of livewire.py with Trailing Stop Loss functionality

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

# Neon Color Scheme
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com") # Keep for direct API calls if needed
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # CCXT uses '1m', '5m' etc. Need mapping later
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
# Default periods (can be overridden by config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14 # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 12     # Window for underlying RSI in StochRSI
DEFAULT_K_WINDOW = 3          # K period for StochRSI
DEFAULT_D_WINDOW = 3          # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
LOOP_DELAY_SECONDS = 15 # Increased delay slightly
# QUOTE_CURRENCY defined in load_config

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found."""
    default_config = {
        "interval": "5m", # Default to ccxt compatible interval
        "retry_delay": 5,
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_window": DEFAULT_CCI_WINDOW,
        "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_window": DEFAULT_MFI_WINDOW,
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW,
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW,
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,
        "scalping_signal_threshold": 2.5,
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": True, # SAFETY FIRST: Default to False
        "use_sandbox": False,     # SAFETY FIRST: Default to True
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage
        "max_concurrent_positions": 2, # Limit open positions for this symbol
        "quote_currency": "USDT", # Currency for balance check and sizing
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to False
        "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance (as decimal)
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry
        # --- End Trailing Stop Loss Config ---
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
        },
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2,
            }
        },
    }

    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added
            if updated_config != config_from_file:
                 with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(updated_config, f, indent=4)
                 print(f"{NEON_YELLOW}Updated config file with default keys: {filepath}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config: {e}. Using default config.{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensure all keys from default_config are in config."""
    updated_config = config.copy()
    for key, value in default_config.items():
        if key not in updated_config:
            updated_config[key] = value
        elif isinstance(value, dict) and isinstance(updated_config.get(key), dict):
            updated_config[key] = _ensure_config_keys(updated_config[key], value)
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    logger_name = f"livewire_bot_{symbol.replace('/','_')}" # Unique logger per symbol instance potentially
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times if logger already exists
    if logger.hasHandlers():
        # Optional: clear existing handlers if config changes drastically between runs
        # logger.handlers.clear()
        return logger # Return existing logger

    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Assume linear contracts, adjust if needed
                'adjustForTimeDifference': True,
                # Bybit V5 Unified Margin requires 'unifiedMargin': True sometimes
                # 'unifiedMargin': False, # Explicitly set if NOT using Unified Margin Account
                # 'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress warning if needed
             }
        })
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE{RESET}")
            exchange.set_sandbox_mode(True)

        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error: {e}{RESET}")
        logger.error(f"{NEON_RED}Please check your API keys, permissions, and IP whitelist.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error initializing: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error initializing: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
    return None


# --- API Request Functions (kept for potential non-ccxt direct V5 calls) ---
# --- CCXT Data Fetching --- (Keep existing functions: fetch_current_price_ccxt, fetch_klines_ccxt, fetch_orderbook_ccxt)
def fetch_current_price_ccxt(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol using CCXT."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            return Decimal(str(ticker['last']))
        else:
            logger.warning(f"{NEON_YELLOW}Could not fetch last price for {symbol} via fetch_ticker.{RESET}")
            trades = exchange.fetch_trades(symbol, limit=1)
            if trades:
                return Decimal(str(trades[0]['price']))
            else:
                 logger.error(f"{NEON_RED}Failed to fetch current price for {symbol} via ticker and trades.{RESET}")
                 return None
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT."""
    lg = logger or logging.getLogger(__name__)
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Ensure correct dtypes for pandas_ta
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing.{RESET}")

        return df

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame()

def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries."""
    attempts = 0
    while attempts <= MAX_API_RETRIES:
        try:
            orderbook = exchange.fetch_order_book(symbol, limit=limit)
            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                return orderbook
            else:
                logger.warning(f"{NEON_YELLOW}Empty or invalid orderbook response for {symbol}. Attempt {attempts + 1}{RESET}")

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"{NEON_YELLOW}Orderbook fetch error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}){RESET}")

        except ccxt.ExchangeError as e:
            logger.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            return None
        except Exception as e:
            logger.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None

# --- Trading Analyzer Class (Using pandas_ta) ---
class TradingAnalyzer:
    """Analyze trading data and generate scalping signals using pandas_ta."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        symbol: str,
        interval: str,
    ) -> None:
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.symbol = symbol
        self.interval = interval
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(interval, "1m")
        self.indicator_values: Dict[str, float] = {}
        self.scalping_signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.weights = config["weight_sets"]["scalping"]
        self.fib_levels_data: Dict[str, Decimal] = {}
        self.ta_column_names = {} # To store actual column names generated by pandas_ta

        self._calculate_all_indicators() # Calculate indicators on initialization

    def _get_ta_col_name(self, base_name: str, *args) -> str:
        """Helper to construct the default column name used by pandas_ta."""
        # Simple default naming convention (e.g., ATR_14, RSI_14, EMA_9)
        # More complex ones like BBands, StochRSI, PSAR have specific formats
        if not args:
            return base_name.upper() # e.g., VWAP
        params_str = '_'.join(map(str, args))
        return f"{base_name.upper()}_{params_str}"

    def _calculate_all_indicators(self):
        """Calculate all enabled indicators using pandas_ta."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Check for sufficient data (rough estimate)
        periods = [
            self.config.get(k, v) for k, v in {
                "atr_period": DEFAULT_ATR_PERIOD, "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
                "rsi_period": DEFAULT_RSI_WINDOW, "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
                "cci_window": DEFAULT_CCI_WINDOW, "williams_r_window": DEFAULT_WILLIAMS_R_WINDOW,
                "mfi_window": DEFAULT_MFI_WINDOW, "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW,
                "sma_10_window": DEFAULT_SMA_10_WINDOW, "momentum_period": DEFAULT_MOMENTUM_PERIOD,
                "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD, "fibonacci_window": DEFAULT_FIB_WINDOW
            }.items()
        ]
        min_required_data = max(periods) + 20 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate.{RESET}")

        try:
            # --- Calculate indicators using pandas_ta ---
            # ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            self.df.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = f"ATRr_{atr_period}"

            # EMAs
            ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
            ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
            self.df.ta.ema(length=ema_short, append=True)
            self.ta_column_names["EMA_Short"] = f"EMA_{ema_short}"
            self.df.ta.ema(length=ema_long, append=True)
            self.ta_column_names["EMA_Long"] = f"EMA_{ema_long}"

            # Momentum (MOM)
            mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
            self.df.ta.mom(length=mom_period, append=True)
            self.ta_column_names["Momentum"] = f"MOM_{mom_period}"

            # CCI
            cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
            self.df.ta.cci(length=cci_period, append=True)
            self.ta_column_names["CCI"] = f"CCI_{cci_period}_0.015" # Default constant

            # Williams %R
            wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
            self.df.ta.willr(length=wr_period, append=True)
            self.ta_column_names["Williams_R"] = f"WILLR_{wr_period}"

            # MFI
            mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
            self.df.ta.mfi(length=mfi_period, append=True)
            self.ta_column_names["MFI"] = f"MFI_{mfi_period}"

            # VWAP
            self.df.ta.vwap(append=True)
            self.ta_column_names["VWAP"] = "VWAP"

            # PSAR
            psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
            psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
            self.df.ta.psar(af=psar_af, max_af=psar_max_af, append=True)
            psar_base = f"{psar_af}_{psar_max_af}"
            self.ta_column_names["PSAR_long"] = f"PSARl_{psar_base}"
            self.ta_column_names["PSAR_short"] = f"PSARs_{psar_base}"
            self.ta_column_names["PSAR_reversal"] = f"PSARr_{psar_base}"

            # SMA 10
            sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
            self.df.ta.sma(length=sma10_period, append=True)
            self.ta_column_names["SMA10"] = f"SMA_{sma10_period}"

            # StochRSI
            stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
            stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
            stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
            stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
            self.df.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d, append=True)
            stochrsi_base = f"_{stoch_rsi_len}_{stoch_rsi_rsi_len}_{stoch_rsi_k}_{stoch_rsi_d}"
            self.ta_column_names["StochRSI_K"] = f"STOCHRSIk{stochrsi_base}"
            self.ta_column_names["StochRSI_D"] = f"STOCHRSId{stochrsi_base}"

            # RSI
            rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
            self.df.ta.rsi(length=rsi_period, append=True)
            self.ta_column_names["RSI"] = f"RSI_{rsi_period}"

            # Bollinger Bands
            bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
            bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
            self.df.ta.bbands(length=bb_period, std=bb_std, append=True)
            bb_base = f"{bb_period}_{bb_std}" # Note: pandas_ta uses float like 2.0 in name
            self.ta_column_names["BB_Lower"] = f"BBL_{bb_period}_{float(bb_std):.1f}" # Format std dev
            self.ta_column_names["BB_Middle"] = f"BBM_{bb_period}_{float(bb_std):.1f}"
            self.ta_column_names["BB_Upper"] = f"BBU_{bb_period}_{float(bb_std):.1f}"


            # Volume MA
            vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
            self.df.ta.sma(close='volume', length=vol_ma_period, append=True) # Calculate SMA on volume
            self.ta_column_names["Volume_MA"] = f"SMA_{vol_ma_period}_volume"

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta: {e}{RESET}", exc_info=True)
            self.df = pd.DataFrame() # Make df empty to signal failure

        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _update_latest_indicator_values(self):
        """Update the indicator_values dictionary with the latest calculated values."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: DataFrame empty or last row NaN.{RESET}")
            self.indicator_values = {k: np.nan for k in self.ta_column_names.keys()}
            return

        latest = self.df.iloc[-1]
        updated_values = {}

        mapping = {
            "ATR": self.ta_column_names.get("ATR"),
            "EMA_Short": self.ta_column_names.get("EMA_Short"),
            "EMA_Long": self.ta_column_names.get("EMA_Long"),
            "Momentum": self.ta_column_names.get("Momentum"),
            "CCI": self.ta_column_names.get("CCI"),
            "Williams_R": self.ta_column_names.get("Williams_R"),
            "MFI": self.ta_column_names.get("MFI"),
            "VWAP": self.ta_column_names.get("VWAP"),
            "PSAR_long": self.ta_column_names.get("PSAR_long"),
            "PSAR_short": self.ta_column_names.get("PSAR_short"),
            "PSAR_reversal": self.ta_column_names.get("PSAR_reversal"),
            "SMA10": self.ta_column_names.get("SMA10"),
            "StochRSI_K": self.ta_column_names.get("StochRSI_K"),
            "StochRSI_D": self.ta_column_names.get("StochRSI_D"),
            "RSI": self.ta_column_names.get("RSI"),
            "BB_Upper": self.ta_column_names.get("BB_Upper"),
            "BB_Middle": self.ta_column_names.get("BB_Middle"),
            "BB_Lower": self.ta_column_names.get("BB_Lower"),
            "Volume_MA": self.ta_column_names.get("Volume_MA"),
        }

        for key, col_name in mapping.items():
            if col_name and col_name in latest and not pd.isna(latest[col_name]):
                try:
                    updated_values[key] = float(latest[col_name])
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert value for {key} ({latest[col_name]}) to float.")
                    updated_values[key] = np.nan
            else:
                 if not col_name:
                    self.logger.debug(f"Internal key '{key}' not found in ta_column_names mapping.")
                 elif col_name not in latest:
                     self.logger.debug(f"Column '{col_name}' for indicator '{key}' not found in DataFrame.")
                 # else: # Value is NaN
                 #    self.logger.debug(f"Value for {key} ({col_name}) is NaN.")
                 updated_values[key] = np.nan # Ensure key exists even if missing/NaN

        self.indicator_values = updated_values
        self.indicator_values["Close"] = float(latest.get('close', np.nan))
        self.indicator_values["Volume"] = float(latest.get('volume', np.nan))
        self.indicator_values["High"] = float(latest.get('high', np.nan)) # Add High/Low for TSL calcs
        self.indicator_values["Low"] = float(latest.get('low', np.nan))


    # --- Fibonacci Calculation (Keep Custom) ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.warning(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}).")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Explicitly cast to string before Decimal for safety
            high = Decimal(str(df_slice["high"].max()))
            low = Decimal(str(df_slice["low"].min()))
            diff = high - low

            levels = {}
            if diff > 0:
                 price_precision_power = Decimal('1e-' + str(self.get_price_precision())) # Get precision dynamically if possible
                 for level in FIB_LEVELS:
                    level_name = f"Fib_{level * 100:.1f}%"
                    # Apply rounding based on market precision
                    levels[level_name] = (high - (diff * Decimal(str(level)))).quantize(price_precision_power, rounding=ROUND_DOWN)
            self.fib_levels_data = levels
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """ Estimate price precision from last close price. Crude fallback. """
        # TODO: Ideally get this from market info passed during init
        last_close = self.indicator_values.get("Close")
        if last_close and not pd.isna(last_close):
             s = str(Decimal(str(last_close)))
             if '.' in s:
                 return len(s.split('.')[-1])
        return 6 # Default fallback precision

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Find nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
             self.calculate_fibonacci_levels()
             if not self.fib_levels_data:
                  return []

        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price):
            self.logger.warning("Invalid current price for Fibonacci comparison.")
            return []

        try:
            level_distances = [
                (name, level, abs(current_price - level))
                for name, level in self.fib_levels_data.items()
                if isinstance(level, Decimal) # Ensure level is Decimal
            ]
            level_distances.sort(key=lambda x: x[2])
            return [(name, level) for name, level, _ in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation (Keep Custom) ---
    def calculate_ema_alignment(self) -> float:
        """Calculate EMA alignment score based on latest values."""
        ema_short = self.indicator_values.get("EMA_Short", np.nan)
        ema_long = self.indicator_values.get("EMA_Long", np.nan)
        current_price = self.indicator_values.get("Close", np.nan)

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            return 0.0

        if current_price > ema_short > ema_long: return 1.0
        elif current_price < ema_short < ema_long: return -1.0
        else: return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generate trading signal based on weighted indicator scores."""
        self.scalping_signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        signal_score = 0.0
        active_indicator_count = 0
        nan_indicator_count = 0

        if not self.indicator_values or all(pd.isna(v) for k,v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low']):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal: Indicator values not calculated or all NaN.{RESET}")
             return "HOLD"
        if pd.isna(current_price):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal: Missing current price.{RESET}")
             return "HOLD"

        # Iterate through configured indicators and add weighted scores
        for indicator, enabled in self.config.get("indicators", {}).items():
            if enabled and indicator in self.weights:
                score_contribution = 0.0
                weight = self.weights[indicator]
                check_method_name = f"_check_{indicator}"
                if hasattr(self, check_method_name):
                    method = getattr(self, check_method_name)
                    try:
                        # Pass current price if needed by the check method (e.g., for checks relative to price)
                        score_contribution = method() * weight # Most checks use self.indicator_values now
                    except Exception as e:
                        self.logger.error(f"Error calling check method {check_method_name}: {e}", exc_info=True)
                        score_contribution = np.nan

                    if not pd.isna(score_contribution):
                        signal_score += score_contribution
                        active_indicator_count += 1
                        # self.logger.debug(f"Indicator {indicator}: value used, score_contrib {score_contribution:.4f}")
                    else:
                        nan_indicator_count += 1
                        self.logger.debug(f"Indicator {indicator}: NaN score contribution")
                else:
                    self.logger.warning(f"No check method found for indicator: {indicator}")

        # Add orderbook score if available
        if orderbook_data:
             orderbook_score = self._check_orderbook(orderbook_data, current_price)
             orderbook_weight = 0.15 # Configurable?
             if not pd.isna(orderbook_score):
                 signal_score += orderbook_score * orderbook_weight
                 # self.logger.debug(f"Orderbook score: {orderbook_score:.4f}, contribution: {orderbook_score * orderbook_weight:.4f}")


        threshold = self.config.get("scalping_signal_threshold", 2.5)
        self.logger.info(f"Signal Score: {signal_score:.4f} (Threshold: +/-{threshold}) from {active_indicator_count} indicators ({nan_indicator_count} NaN)")

        if signal_score >= threshold:
            self.scalping_signals["BUY"] = 1
            return "BUY"
        elif signal_score <= -threshold:
            self.scalping_signals["SELL"] = 1
            return "SELL"
        else:
            self.scalping_signals["HOLD"] = 1
            return "HOLD"


    # --- Check methods using self.indicator_values ---
    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment()

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        return 1.0 if momentum > 0.1 else -1.0 if momentum < -0.1 else 0.0

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = self.config.get("volume_confirmation_multiplier", 1.5)

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma == 0:
            return np.nan

        try:
            if current_volume > volume_ma * multiplier: return 1.0
            elif current_volume < volume_ma / multiplier: return -0.5
            else: return 0.0
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)

        if k < oversold and d < oversold: return 1.0
        if k > overbought and d > overbought: return -1.0
        if k > d: return 0.5
        if k < d: return -0.5
        return 0.0

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan
        if rsi < 30: return 1.0
        if rsi > 70: return -1.0
        if rsi > 55: return 0.5
        if rsi < 45: return -0.5
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        if cci < -100: return 1.0
        if cci > 100: return -1.0
        if cci > 0: return 0.3
        if cci < 0: return -0.3
        return 0.0

    def _check_wr(self) -> float: # Williams %R
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        if wr <= -80: return 1.0
        if wr >= -20: return -1.0
        return 0.0

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get("PSAR_long", np.nan)
        psar_s = self.indicator_values.get("PSAR_short", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)

        if pd.isna(last_close): return np.nan

        is_uptrend = not pd.isna(psar_l) and pd.isna(psar_s)
        is_downtrend = not pd.isna(psar_s) and pd.isna(psar_l)

        if is_uptrend:
            return 1.0 if last_close > psar_l else -1.0 # Consider price relation
        elif is_downtrend:
            return -1.0 if last_close < psar_s else 1.0
        else:
            return 0.0 # Neutral or unknown state

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan
        return 1.0 if last_close > sma_10 else -1.0 if last_close < sma_10 else 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan
        # Be careful with daily reset of VWAP in pandas_ta
        # A simple check might not be reliable across day boundaries
        # Consider VWAP relative to recent price action or other indicators
        return 1.0 if last_close > vwap else -1.0 if last_close < vwap else 0.0 # Simple check for now

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        if mfi < 20: return 1.0
        if mfi > 80: return -1.0
        if mfi > 55: return 0.5
        if mfi < 45: return -0.5
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(last_close): return np.nan

        if last_close < bb_lower: return 1.0
        if last_close > bb_upper: return -1.0
        return 0.0

    def _check_orderbook(self, orderbook_data: Dict, current_price: Decimal) -> float:
        """Analyze order book depth for immediate pressure."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            if not bids or not asks: return 0.0

            # Use a small fixed range around the price (e.g., 0.1%)
            price_range_percent = Decimal("0.001") # 0.1%
            price_range = current_price * price_range_percent

            relevant_bid_volume = sum(Decimal(str(bid[1])) for bid in bids if Decimal(str(bid[0])) >= current_price - price_range)
            relevant_ask_volume = sum(Decimal(str(ask[1])) for ask in asks if Decimal(str(ask[0])) <= current_price + price_range)

            if relevant_bid_volume == 0 and relevant_ask_volume == 0: return 0.0
            if relevant_ask_volume == 0: return 1.0 # Infinite bid pressure (avoid div by zero)
            if relevant_bid_volume == 0: return -1.0 # Infinite ask pressure

            # Order Book Imbalance (OBI)
            obi = (relevant_bid_volume - relevant_ask_volume) / (relevant_bid_volume + relevant_ask_volume)

            # Scale OBI to [-1, 1] score
            if obi > 0.3: return 1.0   # Strong buy pressure
            elif obi > 0.1: return 0.5 # Moderate buy pressure
            elif obi < -0.3: return -1.0 # Strong sell pressure
            elif obi < -0.1: return -0.5 # Moderate sell pressure
            else: return 0.0 # Balanced

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed: {e}{RESET}")
            return 0.0

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, current_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, take profit, and **initial** stop loss levels based on ATR."""
        atr = self.indicator_values.get("ATR")
        if atr is None or pd.isna(atr) or atr <= 0 or current_price is None or pd.isna(current_price):
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL: Missing ATR or Price. ATR: {atr}, Price: {current_price}{RESET}")
            return None, None, None

        try:
            atr_decimal = Decimal(str(atr))
            entry = current_price
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            if signal == "BUY":
                take_profit = entry + (atr_decimal * tp_multiple)
                stop_loss = entry - (atr_decimal * sl_multiple)
            elif signal == "SELL":
                take_profit = entry - (atr_decimal * tp_multiple)
                stop_loss = entry + (atr_decimal * sl_multiple)
            else: # HOLD signal
                return entry, None, None

            # Basic validation
            if signal == "BUY" and (stop_loss >= entry or take_profit <= entry):
                 self.logger.warning(f"BUY signal TP/SL calculation invalid: Entry={entry:.5f}, TP={take_profit:.5f}, SL={stop_loss:.5f}, ATR={atr_decimal:.5f}")
                 return entry, None, None # Return invalid SL/TP
            if signal == "SELL" and (stop_loss <= entry or take_profit >= entry):
                 self.logger.warning(f"SELL signal TP/SL calculation invalid: Entry={entry:.5f}, TP={take_profit:.5f}, SL={stop_loss:.5f}, ATR={atr_decimal:.5f}")
                 return entry, None, None

            # Return calculated values even if TSL is enabled; initial SL is used for sizing
            return entry, take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating TP/SL: {e}{RESET}", exc_info=True)
             return None, None, None


# --- Trading Logic ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency."""
    try:
        # Use fetch_balance({'type': 'swap'}) or {'type': 'linear'} for specific account types if needed
        # Default fetch_balance might cover it for Bybit unified/linear
        balance_info = exchange.fetch_balance()

        # Try standard 'free' first
        free_balance = balance_info.get(currency, {}).get('free')

        # Fallback for different structures (e.g., top-level 'free' dict)
        if free_balance is None:
            free_balance = balance_info.get('free', {}).get(currency)

        # Fallback for Bybit V5 'walletBalance' or 'availableToWithdraw' under the currency
        if free_balance is None:
            coin_info = balance_info.get('info', {}).get('result', {}).get('list', [{}])[0].get('coin', [])
            for coin_data in coin_info:
                 if coin_data.get('coin') == currency:
                     # Prefer availableToWithdraw as it implies usable balance
                     free_balance = coin_data.get('availableToWithdraw')
                     if free_balance is None:
                         free_balance = coin_data.get('walletBalance') # Total balance if available is missing
                     break

        # Final fallback: Check total balance if free is still missing
        if free_balance is None:
             total_balance = balance_info.get(currency, {}).get('total')
             if total_balance is not None:
                  logger.warning(f"{NEON_YELLOW}Could not find 'free' balance for {currency}, using 'total' balance ({total_balance}) as fallback. Use with caution.{RESET}")
                  free_balance = total_balance
             else:
                  logger.error(f"{NEON_RED}Could not determine balance for {currency}. Balance info: {balance_info}{RESET}")
                  return None

        return Decimal(str(free_balance))

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
        return None
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
        return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision and limits."""
    try:
        if not exchange.markets or symbol not in exchange.markets:
             logger.info(f"Market info for {symbol} not loaded or missing, reloading markets...")
             exchange.load_markets(reload=True)

        market = exchange.market(symbol)
        if market:
            # Log key details for confirmation
            # logger.debug(f"Market Info for {symbol}: Type={market.get('type')}, Contract={market.get('contract', False)}, Precision={market.get('precision')}, Limits={market.get('limits')}")
            return market
        else:
             logger.error(f"{NEON_RED}Market {symbol} still not found after reload.{RESET}")
             return None
    except ccxt.BadSymbol as e:
         logger.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id}: {e}{RESET}")
         return None
    except ccxt.NetworkError as e:
         logger.error(f"{NEON_RED}Network error loading markets for {symbol}: {e}{RESET}")
         return None
    except ccxt.ExchangeError as e:
         logger.error(f"{NEON_RED}Exchange error loading markets for {symbol}: {e}{RESET}")
         return None
    except Exception as e:
        logger.error(f"{NEON_RED}Error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    stop_loss_price: Decimal, # Initial SL price
    entry_price: Decimal,
    market_info: Dict,
    leverage: int = 1,
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, INITIAL SL distance, leverage, and market limits."""
    lg = logger or logging.getLogger(__name__)

    if balance is None or balance <= 0 or risk_per_trade <= 0 or stop_loss_price is None or entry_price is None:
        lg.warning(f"{NEON_YELLOW}Invalid inputs for position sizing: Balance={balance}, Risk={risk_per_trade}, Entry={entry_price}, SL={stop_loss_price}{RESET}")
        return None

    if stop_loss_price == entry_price:
         lg.warning(f"{NEON_YELLOW}Stop loss price cannot be the same as entry price.{RESET}")
         return None

    try:
        risk_amount = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - stop_loss_price)

        if sl_distance_per_unit <= 0:
             lg.warning(f"{NEON_YELLOW}Stop loss distance is zero or negative ({sl_distance_per_unit}), cannot calculate position size.{RESET}")
             return None

        # Size in base currency units (e.g., BTC for BTC/USDT)
        # Note: Leverage doesn't directly change this risk-based size calculation,
        # but it affects the required margin and potential liquidation price.
        position_size = risk_amount / sl_distance_per_unit
        lg.info(f"Risk Amount: {risk_amount:.4f} {market_info.get('quote','?')}, SL Distance: {sl_distance_per_unit:.5f}, Initial Size: {position_size:.8f} {market_info.get('base','?')}")

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        precision = market_info.get('precision', {})
        amount_precision = precision.get('amount') # Number of decimal places for amount

        min_amount = Decimal(str(amount_limits.get('min', '0'))) if amount_limits.get('min') is not None else Decimal('0')
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')

        cost_limits = limits.get('cost', {})
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')

        # Adjust size based on limits
        adjusted_size = position_size
        if adjusted_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} is below minimum amount {min_amount:.8f}. Setting to minimum.{RESET}")
            adjusted_size = min_amount

        if adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} exceeds maximum amount {max_amount:.8f}. Capping size.{RESET}")
             adjusted_size = max_amount

        # Check cost limits with the potentially adjusted size
        cost = adjusted_size * entry_price
        if cost < min_cost:
             lg.warning(f"{NEON_YELLOW}Cost {cost:.4f} (Size: {adjusted_size:.8f}) is below minimum cost {min_cost:.4f}. Cannot place trade.{RESET}")
             # Check if minimum amount *also* violates min cost
             min_amount_cost = min_amount * entry_price
             if min_amount_cost < min_cost:
                 lg.error(f"{NEON_RED}Minimum order size {min_amount} also violates minimum cost {min_cost}. Market constraints prevent trading.{RESET}")
             return None # Cannot meet min cost

        if cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Cost {cost:.4f} exceeds maximum cost {max_cost:.4f}. Reducing size based on max cost.{RESET}")
             adjusted_size = max_cost / entry_price # Recalculate size based on max cost
             # Re-check min amount after reducing for max cost
             if adjusted_size < min_amount:
                  lg.warning(f"{NEON_YELLOW}Size reduced for max cost ({adjusted_size:.8f}) is now below minimum amount {min_amount:.8f}. Cannot place trade.{RESET}")
                  return None

        # Apply amount precision (rounding down for safety, though exchange might round differently)
        if amount_precision is not None:
            try:
                # Use ccxt's formatter if possible
                formatted_size_str = exchange.amount_to_precision(market_info['symbol'], adjusted_size)
                final_size = Decimal(formatted_size_str)
                lg.info(f"Applied amount precision ({amount_precision} decimals): {adjusted_size:.8f} -> {final_size:.8f}")
            except Exception as fmt_err:
                lg.warning(f"Could not use exchange.amount_to_precision ({fmt_err}). Using manual rounding.")
                rounding_factor = Decimal('1e-' + str(int(amount_precision)))
                final_size = adjusted_size.quantize(rounding_factor, rounding=ROUND_DOWN)
                lg.info(f"Applied manual amount precision ({amount_precision} decimals): {adjusted_size:.8f} -> {final_size:.8f}")

        else:
             lg.warning(f"{NEON_YELLOW}Amount precision not defined for {market_info['symbol']}. Using unrounded size: {adjusted_size:.8f}{RESET}")
             final_size = adjusted_size # Use the size adjusted for limits only

        # Final checks after rounding
        if final_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero or negative after adjustments/rounding.{RESET}")
             return None
        if final_size < min_amount:
             # If rounding took it below min_amount, check if it's negligibly small difference or truly invalid
             if (min_amount - final_size) / min_amount < Decimal('0.01'): # Allow 1% tolerance below min due to rounding
                  lg.warning(f"{NEON_YELLOW}Rounded size {final_size:.8f} is slightly below minimum {min_amount:.8f} due to precision. Allowing.{RESET}")
             else:
                  lg.warning(f"{NEON_YELLOW}Rounded size {final_size:.8f} is significantly below minimum {min_amount:.8f}. Checking min cost again...{RESET}")
                  final_cost = final_size * entry_price
                  if final_cost < min_cost:
                      lg.error(f"{NEON_RED}Rounded size {final_size:.8f} violates min amount {min_amount:.8f} AND min cost {min_cost:.4f}. No trade.{RESET}")
                      return None
                  else:
                       lg.warning(f"{NEON_YELLOW}...Cost {final_cost:.4f} meets min cost {min_cost:.4f}. Allowing trade despite size slightly below min_amount.{RESET}")


        lg.info(f"Final calculated position size: {final_size:.8f} {market_info.get('base', '')}")
        return final_size

    except KeyError as e:
         lg.error(f"{NEON_RED}Position sizing error: Missing market info key {e}. Market: {market_info}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Error calculating position size: {e}{RESET}", exc_info=True)
        return None


def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Returns the unified position dictionary if found, otherwise None.
    """
    try:
        # Bybit V5 requires 'symbol' param for fetch_positions if not fetching all
        positions: List[Dict] = exchange.fetch_positions([symbol])

        # CCXT aims to standardize, but fields can vary. Check common ones.
        for pos in positions:
            # 1. Check standardized fields (prefer 'contracts' for derivatives)
            pos_size_str = pos.get('contracts')      # Standard for futures/swaps
            if pos_size_str is None: pos_size_str = pos.get('contractSize') # Alternative standard
            side = pos.get('side')                  # Standard 'long' or 'short'

            # 2. Fallback to less standard or exchange-specific fields in 'info'
            if pos_size_str is None: pos_size_str = pos.get('info', {}).get('size') # Common in Bybit V5 info
            if side is None: side = pos.get('info', {}).get('side', '').lower() # Bybit V5 info side ('Buy'/'Sell')

            # 3. Handle potential zero size represented differently
            if pos_size_str is not None:
                try:
                    position_size = Decimal(str(pos_size_str))
                    # Use a small tolerance for floating point inaccuracies
                    if abs(position_size) > Decimal('1e-12'):
                         # Determine side if not explicitly 'long'/'short'
                         if side not in ['long', 'short']:
                             if position_size > 0: side = 'long'
                             elif position_size < 0: side = 'short' # Some exchanges use negative size for shorts

                         # Add inferred side back to dict if needed
                         if 'side' not in pos or pos['side'] is None:
                              pos['side'] = side

                         logger.info(f"Found open {side} position for {symbol}: Size={position_size} (Raw: {pos.get('info')})")
                         # Return the unified position dict from CCXT
                         return pos
                except (ValueError, TypeError) as e:
                     logger.warning(f"Could not parse position size '{pos_size_str}' for {symbol}: {e}")
                     continue # Skip this entry if size is invalid

        # If loop completes without finding a non-zero position
        logger.info(f"No active open position found for {symbol}.")
        return None

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle errors that indicate no position exists gracefully (e.g., Bybit V5 returns success with empty list)
        no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
        if any(msg in str(e).lower() for msg in no_pos_msgs):
             logger.info(f"No open position found for {symbol} (Exchange message: {e}).")
             return None
        # Bybit V5 might return success (retCode 0) but an empty list if no position
        # This is handled by the loop completing without finding a position.
        logger.error(f"{NEON_RED}Exchange error fetching positions for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit specifics."""
    if not market_info.get('contract', False):
        logger.info(f"Leverage setting skipped (Not a contract market: {symbol}).")
        return True # Consider success if not applicable

    # Check if leverage is already set to the desired value (reduces unnecessary API calls)
    try:
        # Note: Fetching positions just to check leverage might be slow/rate-limited
        # Consider fetching only if logic depends heavily on pre-set leverage confirmation
        # current_positions = exchange.fetch_positions([symbol])
        # if current_positions:
        #     current_leverage = current_positions[0].get('leverage') or current_positions[0].get('info', {}).get('leverage')
        #     if current_leverage and int(float(current_leverage)) == leverage:
        #         logger.info(f"Leverage for {symbol} already set to {leverage}x.")
        #         return True
        pass # Skip pre-check for now to simplify
    except Exception as e:
         logger.warning(f"Could not pre-check current leverage for {symbol}: {e}")


    if not hasattr(exchange, 'set_leverage'):
         logger.warning(f"{NEON_YELLOW}Exchange {exchange.id} does not support standard 'set_leverage'. Attempting Bybit V5 specific method if needed.{RESET}")
         # TODO: Implement direct Bybit API call if standard method fails / is missing
         return False

    try:
        logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # Bybit V5 Unified/Linear often requires setting buy and sell leverage separately via params
        # The standard `set_leverage` might handle this, but explicit params are safer
        params = {
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage),
            # 'leverage': str(leverage) # Sometimes only this is needed, depends on account type/API version
        }
        response = exchange.set_leverage(leverage, symbol, params=params)

        # Log response for debugging, but don't rely solely on it for confirmation
        logger.debug(f"Set leverage raw response for {symbol}: {response}")

        # Basic check: if no exception was raised, assume success for now.
        # More robust: Fetch position *after* setting and check the leverage value.
        logger.info(f"Leverage for {symbol} set/requested to {leverage}x. Verify on exchange if needed.")
        return True

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Common "already set" or "no modification needed" messages
        if "leverage not modified" in err_str or "same leverage" in err_str:
            logger.warning(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x or no change needed.{RESET}")
            return True # Treat as success
        elif "set margin mode first" in err_str or "switch margin mode" in err_str:
             logger.error(f"{NEON_RED}Exchange error setting leverage: {e}. >> Hint: Ensure Margin Mode (Isolated/Cross) is set correctly for {symbol} *before* setting leverage.{RESET}")
        elif "available balance not enough" in err_str:
             logger.error(f"{NEON_RED}Exchange error setting leverage: {e}. >> Hint: Insufficient available balance in the account ({QUOTE_CURRENCY}) to support this leverage setting, especially for Isolated Margin.{RESET}")
        else:
             logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    signal: str, # "BUY" or "SELL"
    size: Decimal,
    market_info: Dict,
    tp_price: Optional[Decimal] = None,
    sl_price: Optional[Decimal] = None, # Initial SL
    enable_tsl: bool = False, # Flag to indicate if TSL will be set later
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """
    Places a market order using CCXT.
    Includes initial TP/SL if provided AND Trailing SL is NOT enabled.
    If TSL is enabled, the initial SL is primarily for risk calculation and might be omitted
    or immediately replaced by the TSL setting call.
    """
    lg = logger or logging.getLogger(__name__)
    global current_entry_price # Access the global entry price for validation

    side = 'buy' if signal == "BUY" else 'sell'
    order_type = 'market'
    params = {}

    # --- Prepare TP/SL parameters (Only if TSL is NOT enabled) ---
    # If TSL is enabled, we will set it *after* the position is opened.
    # We might still pass the initial SL here for immediate protection,
    # knowing the set_trailing_stop_loss call will overwrite it.
    # Let's include initial SL/TP even if TSL is enabled, for initial safety net.

    price_precision = market_info.get('precision', {}).get('price')
    if price_precision is None:
         lg.error(f"{NEON_RED}Price precision not found for {symbol}, cannot format TP/SL prices accurately.{RESET}")
         # Decide whether to abort or proceed without SL/TP formatting
         # return None # Abort if precision is critical
         lg.warning(f"{NEON_YELLOW}Proceeding without precise SL/TP formatting due to missing precision info.{RESET}")

    try:
        def format_price(price):
            if price is None: return None
            try:
                # Use exchange's formatting function
                return exchange.price_to_precision(symbol, price)
            except Exception as fmt_e:
                lg.warning(f"Could not format price {price} using exchange.price_to_precision: {fmt_e}. Using basic float conversion.")
                # Fallback to basic float conversion, might lack precision
                return float(price)

        formatted_sl = format_price(sl_price)
        formatted_tp = format_price(tp_price)

        # Basic Validation (ensure SL/TP are on the correct side of entry)
        # Use the globally stored entry price for this check
        entry_for_check = current_entry_price or (Decimal(str(market_info.get('last'))) if market_info.get('last') else None)

        if sl_price is not None and entry_for_check is not None:
             if (side == 'buy' and sl_price >= entry_for_check) or \
                (side == 'sell' and sl_price <= entry_for_check):
                  lg.warning(f"{NEON_YELLOW}Initial SL price ({sl_price}) is on the wrong side of entry ({entry_for_check}) for {side} order. SL may not be set correctly by initial order.{RESET}")
                  # Don't necessarily abort, but log the warning. The TSL setup later should fix it if enabled.
                  # formatted_sl = None # Optionally clear invalid SL for initial order

        if tp_price is not None and entry_for_check is not None:
             if (side == 'buy' and tp_price <= entry_for_check) or \
                (side == 'sell' and tp_price >= entry_for_check):
                  lg.warning(f"{NEON_YELLOW}Initial TP price ({tp_price}) is on the wrong side of entry ({entry_for_check}) for {side} order. TP may not be set correctly.{RESET}")
                  # formatted_tp = None # Optionally clear invalid TP

        # Add SL/TP to params for Bybit V5 (linear/inverse)
        # These might be overwritten later if TSL is enabled and set successfully
        if formatted_sl:
            params['stopLoss'] = formatted_sl
            params['slTriggerBy'] = 'LastPrice' # Common default, others: MarkPrice, IndexPrice
        if formatted_tp:
            params['takeProfit'] = formatted_tp
            params['tpTriggerBy'] = 'LastPrice'

        # Bybit V5 position index (0 for one-way, 1 buy hedge, 2 sell hedge)
        # Assume One-Way mode is enabled on the account settings
        params['positionIdx'] = 0 # For One-Way mode

        # Convert size to float for ccxt amount
        amount_float = float(size)

        lg.info(f"Attempting to place {side.upper()} {order_type} order for {amount_float:.8f} {market_info.get('base','')} of {symbol}...")
        if formatted_tp: lg.info(f"  Initial TP: {formatted_tp}")
        if formatted_sl: lg.info(f"  Initial SL: {formatted_sl}")
        if enable_tsl: lg.info(f"  Trailing SL: Will be set after position confirmation.")
        if params: lg.info(f"  Params: {params}")

        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order
            params=params
        )
        lg.info(f"{NEON_GREEN}Trade Placed Successfully! Order ID: {order.get('id')}{RESET}")
        lg.debug(f"Order details: {order}") # Log the full order response

        # IMPORTANT: Wait briefly for the order to likely fill and position to update on the exchange side
        time.sleep(3) # Small delay - adjust as needed, may need more sophisticated fill check

        return order

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order for {symbol}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
        lg.error(f"  Size: {amount_float}, Params: {params}, Market Limits: {market_info.get('limits')}, Precision: {market_info.get('precision')}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing order for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error placing order for {symbol}: {e}{RESET}")
        err_str = str(e).lower()
        # Provide specific hints for common Bybit V5 errors
        if "order cost not available" in err_str or "insufficient margin balance" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Check available balance ({QUOTE_CURRENCY}) covers the order cost (size * price / leverage + fees). Consider margin mode (Isolated/Cross) and available collateral.{RESET}")
        elif "leverage not match order" in err_str or "position size is zero" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage/position size issue. Ensure leverage is set correctly *before* ordering and position size is valid.{RESET}")
        elif "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Position size might exceed Bybit's risk limit tier for the current leverage. Check Bybit's risk limit documentation.{RESET}")
        elif "set margin mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Ensure margin mode (Isolated/Cross) is compatible with the trade and set beforehand.{RESET}")
        elif "position idx not match position mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Mismatch between positionIdx (should be 0 for One-Way) and account's Position Mode (must be One-Way, not Hedge). Check Bybit account settings.{RESET}")
        elif "repeated order id" in err_str or "order id exists" in err_str:
             lg.warning(f"{NEON_YELLOW}Duplicate order ID detected. May indicate a previous attempt succeeded or a network issue causing retry.{RESET}")
             # Consider adding logic to check if the position was actually opened
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if order failed


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Pass the confirmed position info
    config: Dict[str, Any],
    logger: logging.Logger
) -> bool:
    """
    Sets an exchange-side trailing stop loss for an open position using Bybit's V5 API.
    Requires the position details (side, size, entry price).
    """
    lg = logger
    if not market_info.get('contract', False):
        lg.warning(f"Trailing stop loss is typically for contract markets. Skipping for {symbol}.")
        return False # TSL usually not applicable to spot

    # --- Get TSL parameters from config ---
    callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005))) # e.g., 0.5%
    activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003))) # e.g., 0.3% move

    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_callback_rate ({callback_rate}). Must be positive.{RESET}")
        return False

    # --- Extract position details ---
    try:
        entry_price_str = position_info.get('entryPrice') or position_info.get('info', {}).get('avgPrice')
        position_size_str = position_info.get('contracts') or position_info.get('info', {}).get('size')
        side = position_info.get('side') # Should be 'long' or 'short' from get_open_position

        if not entry_price_str or not position_size_str or not side:
            lg.error(f"{NEON_RED}Missing required position info (entryPrice, size, side) to set TSL. Position: {position_info}{RESET}")
            return False

        entry_price = Decimal(str(entry_price_str))
        position_size = Decimal(str(position_size_str)) # Note: Bybit size might be negative for shorts in some contexts

    except (TypeError, ValueError, KeyError) as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL setup: {e}. Position: {position_info}{RESET}")
        return False

    # --- Calculate TSL parameters for Bybit API ---
    # Bybit V5 `/v5/position/set-trading-stop` parameters:
    # - trailingStop: The trail distance *in price points*.
    # - activePrice: The price at which the TSL becomes active.
    # - tpslMode: 'Full' or 'Partial' (use 'Full' for position TSL)
    # - slOrderType: 'Market' (usually) or 'Limit'

    try:
        price_precision = market_info.get('precision', {}).get('price')
        if price_precision is None:
            lg.error(f"{NEON_RED}Cannot calculate TSL parameters accurately: Missing price precision for {symbol}.{RESET}")
            return False
        price_rounding = Decimal('1e-' + str(int(price_precision)))

        # 1. Calculate Activation Price
        activation_offset = entry_price * activation_percentage
        if side == 'long':
            # Activate when price moves UP by the percentage
            activation_price = (entry_price + activation_offset).quantize(price_rounding, rounding=ROUND_UP)
            # Ensure activation is strictly above entry
            if activation_price <= entry_price: activation_price = (entry_price + price_rounding).quantize(price_rounding, rounding=ROUND_UP)
        elif side == 'short':
            # Activate when price moves DOWN by the percentage
            activation_price = (entry_price - activation_offset).quantize(price_rounding, rounding=ROUND_DOWN)
             # Ensure activation is strictly below entry
            if activation_price >= entry_price: activation_price = (entry_price - price_rounding).quantize(price_rounding, rounding=ROUND_DOWN)
        else:
            lg.error(f"Invalid position side '{side}' for TSL calculation.")
            return False

        # 2. Calculate Trailing Stop Distance (in price points)
        # This is the distance the SL will trail *behind* the best price achieved after activation.
        # Calculated based on the activation price and callback rate.
        trailing_distance = (activation_price * callback_rate).quantize(price_rounding, rounding=ROUND_UP) # Round distance up
        # Ensure minimum distance based on price tick size
        tick_size = market_info.get('precision', {}).get('price') # Tick size is the smallest price increment
        if tick_size is not None and trailing_distance < Decimal(str(tick_size)):
             lg.warning(f"Calculated TSL distance {trailing_distance} is smaller than tick size {tick_size}. Adjusting to tick size.")
             trailing_distance = Decimal(str(tick_size))
        elif trailing_distance <= 0:
             lg.error(f"Calculated TSL distance is zero or negative ({trailing_distance}). Check callback rate and activation price.")
             return False


        # 3. Prepare API Parameters
        params = {
            'category': market_info.get('type', 'linear'), # 'linear', 'inverse', 'spot'(not applicable for TSL)
            'symbol': market_info['id'], # Use exchange-specific ID (e.g., BTCUSDT)
            'tpslMode': 'Full', # Apply to the whole position
            # --- Trailing Stop Parameters ---
            'trailingStop': exchange.price_to_precision(symbol, trailing_distance), # Format distance string
            'activePrice': exchange.price_to_precision(symbol, activation_price), # Format activation price string
            # --- Optional: Keep existing Take Profit? ---
            # If you want to keep the TP set by the initial order, fetch it and include it here.
            # current_tp = position_info.get('takeProfit') or position_info.get('info', {}).get('takeProfit')
            # if current_tp and Decimal(str(current_tp)) > 0:
            #     params['takeProfit'] = exchange.price_to_precision(symbol, current_tp)
            # --- Stop Loss Type ---
            'slOrderType': 'Market', # Or 'Limit' if desired, requires slLimitPrice
            # Add positionIdx for Bybit V5 (0 for one-way)
            'positionIdx': 0 # position_info.get('info', {}).get('positionIdx', 0) # Get from position if available
        }

        lg.info(f"Attempting to set Trailing Stop Loss for {symbol} ({side}):")
        lg.info(f"  Entry Price: {entry_price:.{price_precision}f}")
        lg.info(f"  Activation Price: {params['activePrice']}")
        lg.info(f"  Trailing Distance: {params['trailingStop']} ({callback_rate:.3%})")
        lg.info(f"  Parameters: {params}")

        # --- Call Bybit V5 API Endpoint ---
        # Use private_post for endpoints not wrapped by standard CCXT methods
        response = exchange.private_post('/v5/position/set-trading-stop', params)

        lg.debug(f"Set Trailing Stop response: {response}")

        # --- Check Response ---
        if response and response.get('retCode') == 0:
            lg.info(f"{NEON_GREEN}Trailing Stop Loss set successfully for {symbol}.{RESET}")
            return True
        else:
            ret_code = response.get('retCode')
            ret_msg = response.get('retMsg', 'Unknown Bybit Error')
            ret_ext = response.get('retExtInfo', {})
            lg.error(f"{NEON_RED}Failed to set Trailing Stop Loss for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            # Log specific error hints
            if ret_code == 110043: # tpslMode=Partial needed?
                lg.error(f"{NEON_YELLOW} >> Hint: Try setting 'tpslMode': 'Partial' if not managing the full position size? Or check position mode (One-way/Hedge).{RESET}")
            elif ret_code == 110025: # Position size error
                lg.error(f"{NEON_YELLOW} >> Hint: Position size might have changed or issue matching position.{RESET}")
            elif "active price" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: Activation price ({params['activePrice']}) might be invalid (e.g., too close to current price, wrong side of market).{RESET}")
            elif "trailing stop" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: Trailing stop distance ({params['trailingStop']}) might be invalid (e.g., too small, exceeds limits).{RESET}")
            return False

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting trailing stop for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error setting trailing stop for {symbol}: {e}{RESET}")
    except KeyError as e:
        lg.error(f"{NEON_RED}Error setting TSL: Missing expected key {e} in market/position info.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting trailing stop for {symbol}: {e}{RESET}", exc_info=True)

    return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a symbol and places trades based on signals and risk management."""
    global current_entry_price # Use global to store entry price for checks

    logger.info(f"--- Analyzing {symbol} ({config['interval']}) ---")

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         logger.error(f"Invalid interval '{config['interval']}' provided. Cannot map to CCXT timeframe.")
         return

    kline_limit = 500 # Ensure enough data for indicators
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=logger)
    if klines_df.empty or len(klines_df) < 50:
        logger.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis.{RESET}")
        return

    orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], logger)
    current_price = fetch_current_price_ccxt(exchange, symbol, logger)

    if current_price is None:
         # Fallback to last close price
         current_price = Decimal(str(klines_df['close'].iloc[-1])) if not klines_df.empty else None
         if current_price:
             logger.warning(f"{NEON_YELLOW}Using last close price ({current_price}) as current price fetch failed.{RESET}")
         else:
             logger.error(f"{NEON_RED}Failed to get current price or last close for {symbol}. Skipping analysis.{RESET}")
             return

    # Store globally for potential use in place_trade validation
    current_entry_price = current_price

    # --- 2. Analyze Data ---
    analyzer = TradingAnalyzer(
        klines_df.copy(), logger, config, symbol, config["interval"]
    )
    if not analyzer.indicator_values or all(pd.isna(v) for k,v in analyzer.indicator_values.items() if k not in ['Close','Volume','High','Low']):
         logger.error(f"{NEON_RED}Indicator calculation failed or produced all NaNs for {symbol}. Skipping signal generation.{RESET}")
         return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    entry_calc, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal) # Calculates INITIAL SL
    fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)

    # --- 3. Log Analysis Results ---
    indicator_log = ""
    log_indicators = ["RSI", "StochRSI_K", "StochRSI_D", "MFI", "CCI", "Williams_R", "ATR"]
    for ind_key in log_indicators:
        val = analyzer.indicator_values.get(ind_key)
        precision = 5 if ind_key == "ATR" else 2 # More precision for ATR
        indicator_log += f"{ind_key}: {val:.{precision}f} " if val is not None and not pd.isna(val) else f"{ind_key}: NaN "

    output = (
        f"\n{NEON_BLUE}--- Analysis Results for {symbol} ({config['interval']}) ---{RESET}\n"
        f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Current Price: {NEON_GREEN}{current_price:.5f}{RESET}\n"
        f"Generated Signal: {NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW}{signal}{RESET}\n"
        f"Potential Entry: {NEON_YELLOW}{entry_calc:.5f}{RESET} (based on current price)\n"
        f"Potential Take Profit: {(NEON_GREEN + str(round(tp_calc, 5)) + RESET) if tp_calc else 'N/A'}\n"
        f"Initial Stop Loss: {(NEON_RED + str(round(sl_calc, 5)) + RESET) if sl_calc else 'N/A'} (ATR Multiple: {config.get('stop_loss_multiple')})\n"
        f"{indicator_log.strip()}\n"
        # f"Nearest Fibonacci Levels:\n" + "\n".join(
        #     f"  {name}: {NEON_PURPLE}{level:.5f}{RESET}" for name, level in fib_levels[:3]
        # )
    )
    if config.get('enable_trailing_stop'):
         output += f"Trailing Stop: {NEON_CYAN}Enabled (Rate: {config.get('trailing_stop_callback_rate', 0):.3%}, Activation: {config.get('trailing_stop_activation_percentage', 0):.3%}){RESET}\n"

    logger.info(output)

    # --- 4. Execute Trade (If Enabled and Signal is Buy/Sell) ---
    if not config.get("enable_trading", False):
        # logger.info(f"{NEON_YELLOW}Trading is disabled in config. No trade placed.{RESET}") # Reduce noise
        return

    if signal in ["BUY", "SELL"]:
        logger.info(f"*** {signal} Signal Triggered - Proceeding to Trade Execution ***")

        # --- 4a. Pre-Trade Checks ---
        market_info = get_market_info(exchange, symbol, logger)
        if not market_info:
             logger.error(f"{NEON_RED}Cannot proceed: Failed to get market info for {symbol}.{RESET}")
             return

        # Check for existing position
        open_position = get_open_position(exchange, symbol, logger)
        max_pos = config.get("max_concurrent_positions", 1)
        if max_pos <= 1 and open_position:
            pos_size_str = open_position.get('contracts', open_position.get('info',{}).get('size', '0'))
            pos_side = open_position.get('side', 'unknown')
            try: pos_size = Decimal(pos_size_str)
            except: pos_size = 0
            logger.warning(f"{NEON_YELLOW}Existing {pos_side} position found (Size: {pos_size}). Skipping new trade based on max_concurrent_positions={max_pos}.{RESET}")
            # --- TODO: Add logic here to MANAGE the existing position? ---
            # E.g., Check if the existing position's SL needs updating based on TSL logic?
            # This requires storing the *initial* TSL parameters when the position was opened.
            # For now, we just prevent opening a new one.
            return

        # Fetch available balance
        balance = fetch_balance(exchange, QUOTE_CURRENCY, logger)
        if balance is None or balance <= 0:
            logger.error(f"{NEON_RED}Cannot proceed: Failed to fetch sufficient balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
            return

        # Ensure initial SL is valid for risk calculation
        if sl_calc is None:
             logger.error(f"{NEON_RED}Cannot proceed: Initial Stop Loss calculation failed. Risk cannot be determined.{RESET}")
             return

        # --- 4b. Set Leverage ---
        leverage = int(config.get("leverage", 1))
        if market_info.get('contract', False) and leverage > 0: # Only set leverage for contracts
            if not set_leverage_ccxt(exchange, symbol, leverage, market_info, logger):
                 logger.warning(f"{NEON_YELLOW}Failed to confirm leverage set to {leverage}x for {symbol}. Proceeding with caution (may use default/previous leverage).{RESET}")
                 # Consider aborting if leverage setting is critical and failed: return
        else:
             logger.info(f"Leverage setting skipped (Spot market or leverage <= 0).")


        # --- 4c. Calculate Position Size ---
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=config["risk_per_trade"],
            stop_loss_price=sl_calc, # Use the *initial* calculated SL for sizing
            entry_price=current_price, # Use current price as entry estimate
            market_info=market_info,
            leverage=leverage,
            logger=logger
        )

        if position_size is None or position_size <= 0:
            logger.error(f"{NEON_RED}Trade aborted: Invalid position size calculated ({position_size}). Check balance, risk, SL, and market limits.{RESET}")
            return

        # --- 4d. Place Initial Market Order ---
        # Calculate the TP string representation beforehand
tp_string = f'{tp_calc:.5f}' if tp_calc else 'N/A'

# Use the pre-calculated string in the log message
logger.info(f"Attempting to place {signal} trade for {symbol} | Size: {position_size} | Entry: ~{current_price:.5f} | Initial TP: {tp_string} | Initial SL: {sl_calc:.5f}")
        trade_result = place_trade(
            exchange=exchange,
            symbol=symbol,
            signal=signal,
            size=position_size,
            market_info=market_info,
            tp_price=tp_calc,    # Pass potential initial TP
            sl_price=sl_calc,    # Pass potential initial SL
            enable_tsl=enable_tsl_flag, # Inform place_trade about TSL intent
            logger=logger
        )

        # --- 4e. Set Trailing Stop Loss (If Enabled and Trade Placed) ---
        if trade_result and enable_tsl_flag:
            logger.info(f"Initial order placed for {symbol}. Now attempting to set Trailing Stop Loss.")

            # Fetch the confirmed position details again to ensure we have accurate entry price etc.
            # This is crucial as the market order might have filled at a slightly different price.
            time.sleep(2) # Allow extra time for position update
            confirmed_position = get_open_position(exchange, symbol, logger)

            if confirmed_position:
                # Extract necessary details like accurate entry price from the confirmed position
                set_tsl_success = set_trailing_stop_loss(
                    exchange=exchange,
                    symbol=symbol,
                    market_info=market_info,
                    position_info=confirmed_position, # Use confirmed details
                    config=config,
                    logger=logger
                )
                if set_tsl_success:
                     logger.info(f"{NEON_GREEN}=== TRADE and TSL SETUP COMPLETE for {symbol} ===")
                else:
                     logger.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET TSL for {symbol} ===")
                     # Consider contingency: Maybe try setting a fixed SL again? or alert user.
            else:
                logger.error(f"{NEON_RED}Trade order {trade_result.get('id')} placed, but failed to fetch confirmed position details for {symbol} shortly after. Cannot set TSL.{RESET}")
                logger.error(f"{NEON_YELLOW}Position might still exist with only initial SL/TP (if set) or no SL. Manual check recommended.{RESET}")

        elif trade_result and not enable_tsl_flag:
             logger.info(f"{NEON_GREEN}=== TRADE EXECUTED SUCCESSFULLY (Fixed SL/TP) for {symbol} ===")
        elif not trade_result:
            logger.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ===")
        else:
             # Should not happen if trade_result is None/False
             pass

    else: # HOLD Signal
        logger.info(f"Signal is HOLD for {symbol}. Checking existing position status...")
        # --- Check and potentially manage existing position based on TSL (Advanced) ---
        open_position = get_open_position(exchange, symbol, logger)
        if open_position:
             # If a position exists, we could log its current PnL, SL, TP, TSL status here.
             pos_side = open_position.get('side', 'unknown')
             pos_size = open_position.get('contracts', 'N/A')
             entry_price = open_position.get('entryPrice', 'N/A')
             liq_price = open_position.get('liquidationPrice', 'N/A')
             sl_price = open_position.get('stopLossPrice', open_position.get('info',{}).get('stopLoss', 'N/A'))
             tp_price = open_position.get('takeProfitPrice', open_position.get('info',{}).get('takeProfit', 'N/A'))
             tsl_active = open_position.get('info',{}).get('trailingStop', 'N/A') # Check info for TSL details

             logger.info(f"Existing {pos_side} position details: Size={pos_size}, Entry={entry_price}, Liq={liq_price}, SL={sl_price}, TP={tp_price}, TSL Active={tsl_active}")
             # TODO: Add logic here if needed to monitor/update TSL or SL based on new analysis,
             # e.g., if price moved significantly against the TSL activation.
             # This part requires careful state management and is more complex.
        else:
             logger.info(f"No action taken (HOLD signal and no open position found for {symbol}).")


def main() -> None:
    """Main function to run the bot."""
    global CONFIG
    # Use a generic logger for setup, then specific ones per symbol
    setup_logger("init")
    init_logger = logging.getLogger("init")

    init_logger.info("Starting Livewire Trading Bot (lwx.py - TSL Enhanced)...")
    init_logger.info(f"Loaded configuration from {CONFIG_FILE}")
    init_logger.info(f"Using pandas_ta version: {ta.version}")

    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX environment.{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}Using REAL MONEY environment. TRIPLE CHECK CONFIG AND RISK!{RESET}")
         try:
             confirm = input("Press Enter to acknowledge live trading and continue, or Ctrl+C to abort...")
         except KeyboardInterrupt:
              init_logger.info("Aborted by user.")
              return
    else:
         init_logger.info("Trading is disabled. Running in analysis-only mode.")


    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return

    # --- Symbol and Interval Input ---
    while True:
        symbol_input = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT, ETH/USDT): {RESET}").upper().strip()
        if not symbol_input: continue
        # Ensure standard CCXT format (BASE/QUOTE)
        symbol_ccxt = symbol_input if '/' in symbol_input else f"{symbol_input}/{QUOTE_CURRENCY}"

        # Validate symbol with exchange
        market_info_check = get_market_info(exchange, symbol_ccxt, init_logger)
        if market_info_check:
            market_type = "Contract" if market_info_check.get('contract', False) else "Spot"
            init_logger.info(f"Symbol {symbol_ccxt} validated. Market Type: {market_type}")
            break
        else:
            init_logger.error(f"{NEON_RED}Symbol {symbol_ccxt} not found or invalid on {exchange.id}. Please try again.{RESET}")
            # Display available markets (optional, can be long)
            # try:
            #     available_symbols = list(exchange.markets.keys())
            #     init_logger.info(f"Available symbols (sample): {available_symbols[:20]}")
            # except: pass # Ignore errors listing symbols

    while True:
        interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}]: {RESET}").strip()
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            break
        else:
            init_logger.error(f"{NEON_RED}Invalid interval: '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")

    # Update config with user input for this run (or could manage per-symbol configs)
    CONFIG["interval"] = interval_input
    init_logger.info(f"Using Symbol: {symbol_ccxt}, Interval: {interval_input} ({CCXT_INTERVAL_MAP[interval_input]})")

    # Get a logger specific to this symbol for the main loop
    symbol_logger = setup_logger(symbol_ccxt)


    # --- Main Loop ---
    try:
        while True:
            start_time = time.time()
            try:
                # Reload config each loop to allow dynamic changes? (Optional)
                # CONFIG = load_config(CONFIG_FILE)

                analyze_and_trade_symbol(exchange, symbol_ccxt, CONFIG, symbol_logger)

            except ccxt.RateLimitExceeded as e:
                 symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except ccxt.NetworkError as e:
                 symbol_logger.error(f"{NEON_RED}Network error in main loop: {e}. Waiting 30s...{RESET}")
                 time.sleep(30)
            except ccxt.AuthenticationError as e:
                 symbol_logger.critical(f"{NEON_RED}Authentication Error in loop: {e}. Check API keys/permissions. Stopping bot.{RESET}")
                 break # Stop loop on auth errors
            except ccxt.ExchangeNotAvailable as e:
                 symbol_logger.error(f"{NEON_RED}Exchange not available: {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except ccxt.OnMaintenance as e:
                 symbol_logger.error(f"{NEON_RED}Exchange is under maintenance: {e}. Waiting 5 minutes...{RESET}")
                 time.sleep(300)
            except Exception as loop_error:
                 symbol_logger.error(f"{NEON_RED}An uncaught error occurred in the main loop: {loop_error}{RESET}", exc_info=True)
                 symbol_logger.info("Attempting to continue after 15s delay...")
                 time.sleep(15)

            # Calculate sleep time ensuring loop runs roughly every LOOP_DELAY_SECONDS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.info(f"Cycle took {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
         symbol_logger.critical(f"{NEON_RED}A critical unhandled error occurred outside the main loop: {e}{RESET}", exc_info=True)
    finally:
        symbol_logger.info("Livewire Trading Bot (lwx.py) stopped.")
        if exchange and hasattr(exchange, 'close'):
            try:
                exchange.close()
                symbol_logger.info("Exchange connection closed.")
            except Exception as close_err:
                 symbol_logger.error(f"Error closing exchange connection: {close_err}")


if __name__ == "__main__":
    # Store entry price globally for use in place_trade validation and TSL calculation context.
    # A more robust solution might pass this explicitly or use a class for state.
    current_entry_price: Optional[Decimal] = None
    main()

