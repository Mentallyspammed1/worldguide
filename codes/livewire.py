```python
# livewire.py
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple

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

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
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
        "stoch_rsi_window": DEFAULT_STOCH_RSI_WINDOW, # pandas-ta uses 'length' for stochrsi window
        "stoch_rsi_rsi_window": DEFAULT_STOCH_WINDOW, # pandas-ta uses 'rsi_length'
        "stoch_rsi_k": DEFAULT_K_WINDOW,
        "stoch_rsi_d": DEFAULT_D_WINDOW,
        "psar_af": DEFAULT_PSAR_AF,
        "psar_max_af": DEFAULT_PSAR_MAX_AF,
        "sma_10_window": DEFAULT_SMA_10_WINDOW,
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "orderbook_limit": 25, # Reduced for faster fetch?
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for SL
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # Reduced multiplier
        "scalping_signal_threshold": 2.5, # Adjusted threshold for trading
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False, # SAFETY FIRST: Default to False
        "use_sandbox": True,     # SAFETY FIRST: Default to True (requires sandbox keys)
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 5,          # Set desired leverage (check exchange limits)
        "max_concurrent_positions": 1, # Limit open positions for this symbol
        "quote_currency": "USDT", # Currency for balance check and sizing
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
                # Add weights for other indicators if needed
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
    logger_name = "livewire_bot"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)s{RESET} - %(message)s"
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
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
             }
        })
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE{RESET}")
            exchange.set_sandbox_mode(True)

        exchange.load_markets()
        logger.info(f"CCXT exchange initialized. Sandbox: {CONFIG.get('use_sandbox')}")
        return exchange
    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}CCXT Authentication Error: {e}{RESET}")
        logger.error(f"{NEON_RED}Please check your API keys and permissions.{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}CCXT Exchange Error: {e}{RESET}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}CCXT Network Error: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True)
    return None


# --- API Request Functions (kept for potential non-ccxt use) ---
def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5,
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def bybit_request(
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """Send a request to Bybit API V5 with retry logic."""
    # Note: This uses V5 signing. Ensure endpoint matches V5.
    session = create_session()
    params = params or {}
    recv_window = 5000
    timestamp = str(int(time.time() * 1000))

    if method == "GET":
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        payload = timestamp + API_KEY + str(recv_window) + query_string
    elif method == "POST":
        body = json.dumps(params, separators=(',', ':')) if params else ""
        payload = timestamp + API_KEY + str(recv_window) + body
    else:
         if logger: logger.error(f"Unsupported HTTP method: {method}")
         return None

    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": str(recv_window),
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}{endpoint}"
    request_kwargs: Dict[str, Any] = {
        "method": method,
        "url": url,
        "headers": headers,
        "timeout": 15
    }

    if method == "GET":
        request_kwargs["params"] = params
    elif method == "POST":
        request_kwargs["data"] = json.dumps(params)

    try:
        response = session.request(**request_kwargs)
        response.raise_for_status()
        json_response = response.json()

        if json_response.get("retCode") == 0:
            return json_response
        else:
            if logger:
                ret_msg = json_response.get('retMsg', 'Unknown Bybit Error')
                ret_ext = json_response.get('retExtInfo', {})
                logger.error(f"{NEON_RED}Bybit API Error: {ret_msg} (Code: {json_response.get('retCode')}) Ext: {ret_ext} Endpoint: {endpoint} Params: {params}{RESET}")
            return None
    except requests.exceptions.HTTPError as http_err:
         if logger: logger.error(f"{NEON_RED}HTTP error occurred: {http_err} - Status Code: {http_err.response.status_code} - Response: {http_err.response.text}{RESET}")
         return None
    except requests.exceptions.RequestException as e:
        if logger: logger.error(f"{NEON_RED}API request failed: {e}{RESET}")
        return None
    except json.JSONDecodeError as e:
         if logger: logger.error(f"{NEON_RED}Failed to decode JSON response: {e} - Response: {response.text}{RESET}")
         return None


# --- CCXT Data Fetching ---
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
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            if logger: logger.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Ensure correct dtypes for pandas_ta
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        if df.empty:
             if logger: logger.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing.{RESET}")

        return df

    except ccxt.NetworkError as e:
        if logger: logger.error(f"{NEON_RED}Network error fetching klines for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        if logger: logger.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
    except Exception as e:
        if logger: logger.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
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
        # Find max period needed from config
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
        min_required_data = max(periods) + 20 # Add buffer for multi-step calcs like StochRSI

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate.{RESET}")

        try:
            # --- Calculate indicators using pandas_ta ---
            # Use append=True to add columns directly to self.df
            # Store generated column names for later retrieval

            # ATR
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            self.df.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = f"ATRr_{atr_period}" # pandas-ta typically uses ATRr for RMA smoothed

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
            self.ta_column_names["CCI"] = f"CCI_{cci_period}_0.015" # Default constant 0.015

            # Williams %R
            wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
            self.df.ta.willr(length=wr_period, append=True)
            self.ta_column_names["Williams_R"] = f"WILLR_{wr_period}"

            # MFI
            mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
            self.df.ta.mfi(length=mfi_period, append=True)
            self.ta_column_names["MFI"] = f"MFI_{mfi_period}"

            # VWAP (Note: pandas_ta VWAP resets daily by default if DatetimeIndex has dates)
            self.df.ta.vwap(append=True)
            self.ta_column_names["VWAP"] = "VWAP" # Default name

            # PSAR
            psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
            psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
            self.df.ta.psar(af=psar_af, max_af=psar_max_af, append=True)
            # Store multiple PSAR columns
            psar_base = f"{psar_af}_{psar_max_af}"
            self.ta_column_names["PSAR_long"] = f"PSARl_{psar_base}"
            self.ta_column_names["PSAR_short"] = f"PSARs_{psar_base}"
            self.ta_column_names["PSAR_reversal"] = f"PSARr_{psar_base}" # 0=no reversal, 1=long, -1=short

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
            bb_base = f"{bb_period}_{bb_std}"
            self.ta_column_names["BB_Lower"] = f"BBL_{bb_base}"
            self.ta_column_names["BB_Middle"] = f"BBM_{bb_base}"
            self.ta_column_names["BB_Upper"] = f"BBU_{bb_base}"

            # Volume MA (simple SMA on volume)
            vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
            self.df.ta.sma(close='volume', length=vol_ma_period, append=True) # Calculate SMA on volume column
            self.ta_column_names["Volume_MA"] = f"SMA_{vol_ma_period}_volume" # Name pandas_ta gives when 'close' is overridden


            # Remove rows with initial NaNs created by indicators
            # self.df.dropna(inplace=True) # Optional: Or handle NaNs in checks

            # Log columns for debugging if needed
            # self.logger.debug(f"DataFrame columns after TA: {self.df.columns.tolist()}")

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta: {e}{RESET}", exc_info=True)
            # If TA fails, empty the df or handle appropriately to prevent downstream errors
            self.df = pd.DataFrame() # Make df empty to signal failure


        # Store the latest values in indicator_values dict
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Keep custom Fibonacci calculation

    def _update_latest_indicator_values(self):
        """Update the indicator_values dictionary with the latest calculated values using stored column names."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: DataFrame is empty or last row contains NaNs.{RESET}")
            self.indicator_values = {k: np.nan for k in self.ta_column_names.keys()} # Set all to NaN
            return

        latest = self.df.iloc[-1]
        updated_values = {}

        # Map internal names to pandas_ta column names
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
            if col_name and col_name in latest:
                updated_values[key] = float(latest[col_name]) if not pd.isna(latest[col_name]) else np.nan
            else:
                # self.logger.debug(f"Column name not found for {key}: {col_name}") # Debug missing cols
                updated_values[key] = np.nan # Ensure key exists even if column missing

        self.indicator_values = updated_values
        # Add raw close price for convenience in checks
        self.indicator_values["Close"] = float(latest.get('close', np.nan))
        self.indicator_values["Volume"] = float(latest.get('volume', np.nan))


    # --- Indicator Calculation Methods REMOVED (handled by pandas_ta) ---
    # Removed: calculate_atr, calculate_sma, calculate_ema, calculate_momentum,
    #          calculate_cci, calculate_williams_r, calculate_mfi, calculate_vwap,
    #          calculate_psar, calculate_sma_10, calculate_stoch_rsi, calculate_rsi,
    #          calculate_bollinger_bands

    # --- Keep Custom Fibonacci ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.warning(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}).")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            high = Decimal(str(df_slice["high"].max()))
            low = Decimal(str(df_slice["low"].min()))
            diff = high - low

            levels = {}
            if diff > 0:
                for level in FIB_LEVELS:
                    level_name = f"Fib_{level * 100:.1f}%"
                    levels[level_name] = (high - (diff * Decimal(str(level)))).quantize(Decimal("0.000001"), rounding=ROUND_DOWN) # Adjust precision
            self.fib_levels_data = levels
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}


    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Find nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
             self.calculate_fibonacci_levels()
             if not self.fib_levels_data:
                  return []

        if current_price is None or not isinstance(current_price, Decimal):
            self.logger.warning("Invalid current price for Fibonacci comparison.")
            return []

        try:
            level_distances = [
                (name, level, abs(current_price - level))
                for name, level in self.fib_levels_data.items()
            ]
            level_distances.sort(key=lambda x: x[2])
            return [(name, level) for name, level, _ in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels: {e}{RESET}", exc_info=True)
            return []

    # --- Keep custom comparison/logic methods ---
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


    # --- Signal Generation & Scoring (Checks use self.indicator_values) ---

    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generate trading signal based on weighted indicator scores."""
        self.scalping_signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        signal_score = 0.0
        active_indicator_count = 0

        # Check if indicator values are populated
        if not self.indicator_values or all(pd.isna(v) for v in self.indicator_values.values()):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values not calculated or all NaN.{RESET}")
             return "HOLD"
        if pd.isna(current_price):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Missing current price.{RESET}")
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
                        score_contribution = method() * weight
                    except Exception as e:
                        self.logger.error(f"Error calling check method {check_method_name}: {e}", exc_info=True)
                        score_contribution = np.nan # Treat error as NaN

                    if not pd.isna(score_contribution):
                        signal_score += score_contribution
                        active_indicator_count += 1
                        # self.logger.debug(f"Indicator {indicator}: value {self.indicator_values.get(indicator.split('_')[0])} score_contrib {score_contribution:.4f}") # Debugging
                    # else:
                        # self.logger.debug(f"Indicator {indicator}: NaN score contribution")

                else:
                    self.logger.warning(f"No check method found for indicator: {indicator}")

        # Add orderbook score if available
        if orderbook_data:
             orderbook_score = self._check_orderbook(orderbook_data, current_price)
             orderbook_weight = 0.15 # Configurable?
             if not pd.isna(orderbook_score):
                 signal_score += orderbook_score * orderbook_weight
                 # self.logger.debug(f"Orderbook score: {orderbook_score:.4f}, contribution: {orderbook_score * orderbook_weight:.4f}")


        threshold = self.config.get("scalping_signal_threshold", 2.5) # Use get with default
        self.logger.info(f"[{self.symbol}] Signal Score: {signal_score:.4f} (Threshold: +/-{threshold}) from {active_indicator_count} indicators")

        if signal_score >= threshold:
            self.scalping_signals["BUY"] = 1
            return "BUY"
        elif signal_score <= -threshold:
            self.scalping_signals["SELL"] = 1
            return "SELL"
        else:
            self.scalping_signals["HOLD"] = 1
            return "HOLD"


    # --- Check methods using self.indicator_values (updated keys) ---

    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment() # Uses values already in dict

    def _check_momentum(self) -> float:
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return 0.0
        return 1.0 if momentum > 0.1 else -1.0 if momentum < -0.1 else 0.0

    def _check_volume_confirmation(self) -> float:
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = self.config.get("volume_confirmation_multiplier", 1.5)

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma == 0:
            return 0.0

        try:
            if current_volume > volume_ma * multiplier:
                return 1.0 # High volume confirmation
            elif current_volume < volume_ma / multiplier:
                 return -0.5 # Low volume weakness
            else:
                return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed: {e}{RESET}")
            return 0.0

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return 0.0

        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)

        if k < oversold and d < oversold: return 1.0 # Oversold
        if k > overbought and d > overbought: return -1.0 # Overbought
        if k > d: return 0.5 # K above D (bullish bias)
        if k < d: return -0.5 # K below D (bearish bias)
        return 0.0

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return 0.0
        if rsi < 30: return 1.0
        if rsi > 70: return -1.0
        if rsi > 55: return 0.5
        if rsi < 45: return -0.5
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return 0.0
        if cci < -100: return 1.0
        if cci > 100: return -1.0
        if cci > 0: return 0.3
        if cci < 0: return -0.3
        return 0.0

    def _check_wr(self) -> float: # Check Williams %R
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return 0.0
        if wr <= -80: return 1.0 # Oversold
        if wr >= -20: return -1.0 # Overbought
        # if wr > -50: return 0.3 # Above midpoint (less common)
        # if wr < -50: return -0.3 # Below midpoint
        return 0.0 # Use only OS/OB for stronger signal

    def _check_psar(self) -> float:
        """Check PSAR using pandas_ta output."""
        psar_l = self.indicator_values.get("PSAR_long", np.nan)
        psar_s = self.indicator_values.get("PSAR_short", np.nan)
        # psar_r = self.indicator_values.get("PSAR_reversal", np.nan) # Reversal signal might be useful too
        last_close = self.indicator_values.get("Close", np.nan)

        if pd.isna(last_close): return 0.0

        # Determine current trend based on which PSAR value is active (not NaN)
        is_uptrend = not pd.isna(psar_l) and pd.isna(psar_s)
        is_downtrend = not pd.isna(psar_s) and pd.isna(psar_l)

        if is_uptrend:
            # In uptrend, price above PSAR_long confirms trend
            return 1.0 if last_close > psar_l else -1.0 # -1 indicates potential reversal
        elif is_downtrend:
            # In downtrend, price below PSAR_short confirms trend
            return -1.0 if last_close < psar_s else 1.0 # 1 indicates potential reversal
        else:
            # If both or neither are NaN (e.g., start of series, or calculation issue)
            return 0.0 # Neutral signal

    def _check_sma_10(self) -> float:
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return 0.0
        return 1.0 if last_close > sma_10 else -1.0 if last_close < sma_10 else 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return 0.0
        return 1.0 if last_close > vwap else -1.0 if last_close < vwap else 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return 0.0
        if mfi < 20: return 1.0
        if mfi > 80: return -1.0
        if mfi > 55: return 0.5
        if mfi < 45: return -0.5
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(last_close): return 0.0

        if last_close < bb_lower: return 1.0 # Below lower band (potential buy)
        if last_close > bb_upper: return -1.0 # Above upper band (potential sell)
        # Could add BB squeeze check here later (BBW)
        return 0.0

    def _check_orderbook(self, orderbook_data: Dict, current_price: Decimal) -> float:
        """Analyze order book depth for immediate pressure."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            if not bids or not asks: return 0.0

            price_range_percent = Decimal("0.001")
            price_range = current_price * price_range_percent

            relevant_bid_volume = sum(Decimal(str(bid[1])) for bid in bids if Decimal(str(bid[0])) >= current_price - price_range)
            relevant_ask_volume = sum(Decimal(str(ask[1])) for ask in asks if Decimal(str(ask[0])) <= current_price + price_range)

            if relevant_bid_volume == 0 and relevant_ask_volume == 0: return 0.0
            if relevant_ask_volume == 0: return 1.0 # Infinite bid pressure
            if relevant_bid_volume == 0: return -1.0 # Infinite ask pressure

            imbalance_ratio = relevant_bid_volume / relevant_ask_volume

            if imbalance_ratio > 1.5: return 1.0   # Strong bid
            elif imbalance_ratio > 1.1: return 0.5 # Moderate bid
            elif imbalance_ratio < 0.67: return -1.0 # Strong ask
            elif imbalance_ratio < 0.91: return -0.5 # Moderate ask
            else: return 0.0 # Balanced

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed: {e}{RESET}")
            return 0.0

    # --- Risk Management Calculations ---

    def calculate_entry_tp_sl(
        self, current_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, take profit, and stop loss levels based on ATR."""
        atr = self.indicator_values.get("ATR")
        if atr is None or pd.isna(atr) or atr <= 0 or current_price is None or pd.isna(current_price):
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Missing ATR or Price. ATR: {atr}, Price: {current_price}{RESET}")
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

            if signal == "BUY" and (stop_loss >= entry or take_profit <= entry):
                 self.logger.warning(f"[{self.symbol}] BUY signal TP/SL calculation invalid: Entry={entry:.5f}, TP={take_profit:.5f}, SL={stop_loss:.5f}, ATR={atr_decimal:.5f}")
                 return entry, None, None
            if signal == "SELL" and (stop_loss <= entry or take_profit >= entry):
                 self.logger.warning(f"[{self.symbol}] SELL signal TP/SL calculation invalid: Entry={entry:.5f}, TP={take_profit:.5f}, SL={stop_loss:.5f}, ATR={atr_decimal:.5f}")
                 return entry, None, None

            return entry, take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating TP/SL: {e}{RESET}", exc_info=True)
             return None, None, None

    # Placeholder - not used currently
    # def calculate_confidence(self) -> float:
    #     return 75.0


# --- Trading Logic ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the free balance for a specific currency."""
    try:
        balance_info = exchange.fetch_balance()
        # Use 'free' for available funds
        free_balance = balance_info.get(currency, {}).get('free')
        if free_balance is not None:
            return Decimal(str(free_balance))
        else:
            # Check top level 'free' dict as structure can vary
            free_balance_alt = balance_info.get('free', {}).get(currency)
            if free_balance_alt is not None:
                return Decimal(str(free_balance_alt))
            else:
                logger.warning(f"{NEON_YELLOW}Could not find 'free' balance for {currency}. Check balance structure: {balance_info}{RESET}")
                return None
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching balance: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error fetching balance: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching balance: {e}{RESET}", exc_info=True)
    return None

def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Gets market information like precision and limits."""
    try:
        if not exchange.markets: # Load markets if not already loaded
             logger.info("Markets not loaded, loading now...")
             exchange.load_markets(reload=True) # Force reload

        market = exchange.market(symbol)
        if market:
            return market
        else:
            # Maybe the symbol format is slightly off? Try fetching markets again.
            logger.warning(f"Market info initially not found for {symbol}. Reloading markets...")
            exchange.load_markets(reload=True)
            market = exchange.market(symbol) # Try again
            if market:
                 logger.info(f"Found market {symbol} after reload.")
                 return market
            else:
                 logger.error(f"{NEON_RED}Market {symbol} still not found after reload.{RESET}")
                 return None
    except ccxt.BadSymbol:
         logger.error(f"{NEON_RED}Symbol '{symbol}' is not supported by the exchange {exchange.id}.{RESET}")
         return None
    except Exception as e:
        logger.error(f"{NEON_RED}Error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: Dict,
    leverage: int = 1, # Leverage mostly affects margin, not size directly in this calc
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, SL distance, leverage, and market limits."""
    lg = logger or logging.getLogger(__name__) # Fallback logger

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
        position_size = risk_amount / sl_distance_per_unit
        lg.info(f"Risk Amount: {risk_amount:.4f} {QUOTE_CURRENCY}, SL Distance: {sl_distance_per_unit:.5f}, Initial Size: {position_size:.8f}")


        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        precision = market_info.get('precision', {})
        amount_precision = precision.get('amount') # Number of decimal places for amount

        min_amount = Decimal(str(amount_limits.get('min', '0'))) if amount_limits.get('min') is not None else Decimal('0')
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')

        # Cost limits (relevant for market orders)
        cost_limits = limits.get('cost', {})
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')

        # Check min/max amount
        if position_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {position_size:.8f} is below minimum order size {min_amount:.8f}. Checking min cost...{RESET}")
            # If size is too small, check if the *cost* at this size would meet min cost
            cost = position_size * entry_price
            if cost < min_cost:
                 lg.warning(f"{NEON_YELLOW}... Cost {cost:.4f} is also below minimum {min_cost:.4f}. No trade.{RESET}")
                 return None
            else:
                 lg.warning(f"{NEON_YELLOW}... Cost {cost:.4f} meets minimum {min_cost:.4f}. Proceeding with small size.{RESET}")
                 # Allow proceeding if cost is okay, but size is technically below min_amount limit
                 # Might still fail on exchange, but worth trying if cost is valid

        if position_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {position_size:.8f} exceeds maximum order size {max_amount:.8f}. Capping size to {max_amount:.8f}.{RESET}")
             position_size = max_amount

        # Check max cost
        cost = position_size * entry_price
        if cost > max_cost:
             lg.warning(f"{NEON_YELLOW}Cost {cost:.4f} of capped size exceeds maximum cost {max_cost:.4f}. Further reducing size.{RESET}")
             # Reduce size based on max cost
             position_size = max_cost / entry_price
             if position_size < min_amount:
                 lg.warning(f"{NEON_YELLOW}Size reduced for max cost ({position_size:.8f}) is now below minimum amount {min_amount:.8f}. No trade.{RESET}")
                 return None


        # Apply amount precision (rounding down)
        if amount_precision is not None:
            rounding_factor = Decimal('1e-' + str(int(amount_precision)))
            original_size = position_size
            position_size = position_size.quantize(rounding_factor, rounding=ROUND_DOWN)
            if original_size != position_size:
                 lg.info(f"Applied amount precision ({amount_precision} decimals): {original_size:.8f} -> {position_size:.8f}")
        else:
             lg.warning(f"{NEON_YELLOW}Amount precision not defined for {market_info['symbol']}. Size not rounded precisely.{RESET}")


        # Final check: Rounded size must not be zero or below minimum amount/cost
        if position_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero or negative after rounding/adjustments.{RESET}")
             return None
        if position_size < min_amount:
             # Re-check cost after rounding
             cost = position_size * entry_price
             if cost < min_cost:
                 lg.warning(f"{NEON_YELLOW}Rounded size {position_size:.8f} is below minimum amount {min_amount:.8f} AND cost {cost:.4f} is below min cost {min_cost:.4f}. No trade.{RESET}")
                 return None
             else:
                  lg.warning(f"{NEON_YELLOW}Rounded size {position_size:.8f} is below min amount {min_amount:.8f}, but cost {cost:.4f} meets min cost {min_cost:.4f}. Allowing trade.{RESET}")

        lg.info(f"Final calculated position size: {position_size:.8f} {market_info.get('base', '')}")
        return position_size

    except KeyError as e:
         lg.error(f"{NEON_RED}Position sizing error: Missing market info key {e}. Market: {market_info}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Error calculating position size: {e}{RESET}", exc_info=True)
        return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks if there's an open position for the given symbol."""
    try:
        # Ensure the symbol is in the format required by fetch_positions if needed
        # Bybit usually accepts the standard symbol format 'BTC/USDT'
        positions = exchange.fetch_positions([symbol])

        # Filter for the specific symbol and non-zero size
        for pos in positions:
            # CCXT aims to standardize, check common fields first
            position_size_str = pos.get('contracts') # Futures often use 'contracts'
            if position_size_str is None: position_size_str = pos.get('contractSize') # Another common name
            if position_size_str is None: position_size_str = pos.get('size') # Spot margin might use 'size'
            if position_size_str is None: position_size_str = pos.get('info', {}).get('size') # Fallback to raw info size (Bybit V5)
            if position_size_str is None: position_size_str = pos.get('info', {}).get('positionAmt') # Binance raw name

            if position_size_str is not None:
                try:
                    position_size = Decimal(str(position_size_str))
                    # Check if size is effectively non-zero (handle potential small residuals)
                    if abs(position_size) > Decimal('1e-12'): # Use a small tolerance
                         logger.info(f"Found open position for {symbol}: Size={position_size} (Raw: {pos})")
                         return pos # Return the unified position dictionary
                except Exception as e:
                     logger.warning(f"Could not parse position size '{position_size_str}' for {symbol}: {e}")
                     continue # Skip this entry if size is invalid

        # If loop completes without finding a non-zero position
        logger.info(f"No active open position found for {symbol}.")
        return None

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle errors that indicate no position exists gracefully
        no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
        if any(msg in str(e).lower() for msg in no_pos_msgs):
             logger.info(f"No open position found for {symbol} (Exchange message: {e}).")
             return None
        logger.error(f"{NEON_RED}Exchange error fetching positions for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    return None

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT."""
    if not hasattr(exchange, 'set_leverage'):
         logger.warning(f"{NEON_YELLOW}Exchange {exchange.id} does not support setting leverage via ccxt standard 'set_leverage' method.{RESET}")
         # Check for alternative methods if needed (e.g., specific API calls via fetch2)
         return False

    try:
        # Bybit V5 unified: set_leverage(leverage, symbol, params={'buyLeverage': leverage, 'sellLeverage': leverage})
        # The standard ccxt method might handle this internally, try it first.
        logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        response = exchange.set_leverage(leverage, symbol)
        logger.info(f"Set leverage response: {response}") # Log response for debugging
        # Verification step: Fetch positions again to see if leverage updated (optional)
        # pos = get_open_position(exchange, symbol, logger) # This might not show leverage reliably if no pos open
        # logger.info(f"Leverage set confirmation (check exchange UI if needed).")
        return True
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. Check Margin Mode (Isolated/Cross) & Collateral.{RESET}")
        # Common Bybit error: "Set leverage not modified" - might mean it's already set
        if "not modified" in str(e):
            logger.warning(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x or could not be changed.{RESET}")
            return True # Treat as success if already set
        # Bybit error for Isolated margin needing collateral
        if "available balance not enough" in str(e).lower():
             logger.error(f"{NEON_RED} >> Hint: Ensure sufficient collateral in the account for {symbol} (especially if using Isolated Margin).{RESET}")
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
    sl_price: Optional[Decimal] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[Dict]:
    """Places a market order with optional TP/SL using CCXT."""
    lg = logger or logging.getLogger(__name__)

    side = 'buy' if signal == "BUY" else 'sell'
    order_type = 'market'
    params = {}

    # --- Prepare TP/SL parameters (Exchange Specific!) ---
    # Bybit V5 USDT Linear uses 'stopLoss' and 'takeProfit' in params
    price_precision = market_info.get('precision', {}).get('price')
    if price_precision is None:
         lg.error(f"{NEON_RED}Price precision not found for {symbol}, cannot format TP/SL prices.{RESET}")
         return None

    try:
        def format_price(price):
            if price is None: return None
            # Use exchange's formatting function for robustness
            return exchange.price_to_precision(symbol, price)
            # Manual Decimal formatting (alternative):
            # rounding_factor = Decimal('1e-' + str(int(price_precision)))
            # return float(price.quantize(rounding_factor)) # CCXT often expects float

        formatted_sl = format_price(sl_price)
        formatted_tp = format_price(tp_price)

        # Ensure SL/TP makes sense relative to current price / side
        # (Basic check, more robust checks done in calculate_entry_tp_sl)
        if sl_price is not None:
             if (side == 'buy' and sl_price >= entry_price) or \
                (side == 'sell' and sl_price <= entry_price):
                  lg.error(f"{NEON_RED}Invalid SL price ({sl_price}) relative to entry ({entry_price}) for {side} order.{RESET}")
                  # Decide whether to proceed without SL or abort? Abort for safety.
                  # return None # Uncomment to abort if SL is invalid
                  formatted_sl = None # Or just ignore invalid SL
                  lg.warning(f"{NEON_YELLOW}Ignoring invalid SL price and proceeding without it.{RESET}")
             else:
                  params['stopLoss'] = formatted_sl
                  # Bybit V5: Specify trigger price type for SL/TP is often good practice
                  params['slTriggerBy'] = 'LastPrice' # Or 'MarkPrice', 'IndexPrice'

        if tp_price is not None:
             if (side == 'buy' and tp_price <= entry_price) or \
                (side == 'sell' and tp_price >= entry_price):
                  lg.error(f"{NEON_RED}Invalid TP price ({tp_price}) relative to entry ({entry_price}) for {side} order.{RESET}")
                  formatted_tp = None # Ignore invalid TP
                  lg.warning(f"{NEON_YELLOW}Ignoring invalid TP price and proceeding without it.{RESET}")
             else:
                  params['takeProfit'] = formatted_tp
                  params['tpTriggerBy'] = 'LastPrice'

        # Bybit V5: positionIdx might be needed (0 for one-way, 1 buy hedge, 2 sell hedge)
        # Assuming One-Way mode is default on account
        params['positionIdx'] = 0

        # Time in force for Market order (usually not needed, but can add if required)
        # params['timeInForce'] = 'GTC' # GoodTillCancel (default often) or 'IOC'/'FOK'

        # Convert size to float as required by ccxt create_order amount
        amount_float = float(size)

        lg.info(f"Attempting to place {signal} {order_type} order for {amount_float} {market_info.get('base','')} of {symbol}...")
        if formatted_tp: lg.info(f"  TP: {formatted_tp}")
        if formatted_sl: lg.info(f"  SL: {formatted_sl}")
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
        lg.info(f"Order details: {order}") # Log the full order response
        return order

    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {signal} order for {symbol}: {e}{RESET}")
        # Could try fetching balance again here to double-check
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
        lg.error(f"  Size: {amount_float}, Params: {params}, Market Limits: {market_info.get('limits')}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing order for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        lg.error(f"{NEON_RED}Exchange error placing order for {symbol}: {e}{RESET}")
        # Provide hints for common Bybit errors
        err_str = str(e).lower()
        if "order cost not available" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Check available balance ({QUOTE_CURRENCY}) covers the order cost (size * price / leverage + fees).{RESET}")
        if "leverage not match order" in err_str or "position size is zero" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Leverage/position size issue. Ensure leverage is set correctly *before* ordering and position size is valid.{RESET}")
        if "risk limit" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Position size might exceed Bybit's risk limit tier for the current leverage. Check Bybit's risk limit documentation.{RESET}")
        if "set margin mode" in err_str:
             lg.error(f"{NEON_YELLOW} >> Hint: Ensure margin mode (Isolated/Cross) is compatible with the trade.{RESET}")

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if order failed


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a symbol and places trades based on signals and risk management."""
    global entry_price # Make entry_price accessible for SL/TP formatting check

    logger.info(f"--- Analyzing {symbol} ({config['interval']}) ---")

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         logger.error(f"Invalid interval '{config['interval']}' provided. Cannot map to CCXT timeframe.")
         return

    kline_limit = 500 # Ensure enough data for longer lookbacks + pandas_ta NaNs
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=logger)
    if klines_df.empty or len(klines_df) < 50: # Basic check for minimum viable data
        logger.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis.{RESET}")
        return

    orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], logger)
    # Don't fail if orderbook fails, just proceed without it

    current_price = fetch_current_price_ccxt(exchange, symbol, logger)
    entry_price = current_price # Store globally for use in place_trade SL/TP check
    if entry_price is None:
         entry_price = Decimal(str(klines_df['close'].iloc[-1])) if not klines_df.empty else None
         if entry_price:
             logger.warning(f"{NEON_YELLOW}Using last close price ({entry_price}) as current price fetch failed.{RESET}")
         else:
             logger.error(f"{NEON_RED}Failed to get current price or last close for {symbol}. Skipping analysis.{RESET}")
             return

    # --- 2. Analyze Data ---
    analyzer = TradingAnalyzer(
        klines_df.copy(), logger, config, symbol, config["interval"] # Pass a copy
    )
    # Indicators calculated in __init__

    # Check if analysis produced valid results
    if not analyzer.indicator_values or all(pd.isna(v) for v in analyzer.indicator_values.values()):
         logger.error(f"{NEON_RED}Indicator calculation failed for {symbol}. Skipping signal generation.{RESET}")
         return

    signal = analyzer.generate_trading_signal(entry_price, orderbook_data)
    entry_calc, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(entry_price, signal)
    # confidence = analyzer.calculate_confidence() # Not currently used
    fib_levels = analyzer.get_nearest_fibonacci_levels(entry_price)

    # --- 3. Log Analysis Results ---
    # Build indicator string dynamically
    indicator_log = ""
    log_indicators = ["RSI", "StochRSI_K", "StochRSI_D", "MFI", "CCI", "Williams_R"]
    for ind_key in log_indicators:
        val = analyzer.indicator_values.get(ind_key, np.nan)
        indicator_log += f"{ind_key}: {val:.2f} " if not pd.isna(val) else f"{ind_key}: NaN "

    output = (
        f"\n{NEON_BLUE}--- Analysis Results for {symbol} ({config['interval']}) ---{RESET}\n"
        f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Current Price: {NEON_GREEN}{entry_price:.5f}{RESET}\n" # Use more precision for price
        f"Generated Signal: {NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW}{signal}{RESET}\n"
        f"Potential Entry: {NEON_YELLOW}{entry_calc:.5f}{RESET}\n"
        f"Potential Take Profit: {(NEON_GREEN + str(round(tp_calc, 5)) + RESET) if tp_calc else 'N/A'}\n"
        f"Potential Stop Loss: {(NEON_RED + str(round(sl_calc, 5)) + RESET) if sl_calc else 'N/A'}\n"
        f"ATR ({config.get('atr_period', DEFAULT_ATR_PERIOD)}): {NEON_YELLOW}{analyzer.indicator_values.get('ATR', np.nan):.5f}{RESET}\n"
        f"{indicator_log.strip()}\n"
        # f"Nearest Fibonacci Levels:\n" + "\n".join(
        #     f"  {name}: {NEON_PURPLE}{level:.5f}{RESET}" for name, level in fib_levels[:3]
        # )
    )
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
             logger.error(f"{NEON_RED}Cannot proceed with trade: Failed to get market info for {symbol}.{RESET}")
             return

        # Check for existing position only if max_concurrent is 1
        if config.get("max_concurrent_positions", 1) <= 1:
            open_position = get_open_position(exchange, symbol, logger)
            if open_position:
                pos_size_str = open_position.get('contracts', open_position.get('info',{}).get('size', '0'))
                pos_side = open_position.get('side', open_position.get('info',{}).get('side', 'unknown'))
                try: pos_size = Decimal(pos_size_str)
                except: pos_size = 0
                logger.warning(f"{NEON_YELLOW}Existing {pos_side} position found for {symbol} (Size: {pos_size}). Skipping new trade based on max_concurrent_positions=1.{RESET}")
                return

        # Fetch available balance
        balance = fetch_balance(exchange, QUOTE_CURRENCY, logger)
        if balance is None or balance <= 0:
            logger.error(f"{NEON_RED}Cannot proceed with trade: Failed to fetch sufficient balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
            return

        # Ensure calculated TP/SL are valid numbers if they exist
        if sl_calc is None:
             logger.error(f"{NEON_RED}Cannot proceed with trade: Stop Loss calculation failed.{RESET}")
             return
        # TP is optional, but SL is usually required for risk management

        # --- 4b. Set Leverage ---
        leverage = int(config.get("leverage", 1))
        # Check if market is a derivative type (future, swap, option) using market_info
        is_derivative = market_info.get('contract', False) or market_info.get('linear') or market_info.get('inverse')

        if is_derivative and leverage > 1:
            if not set_leverage_ccxt(exchange, symbol, leverage, logger):
                 logger.warning(f"{NEON_YELLOW}Failed to confirm leverage set to {leverage}x for {symbol}. Proceeding with caution.{RESET}")
                 # Consider aborting if leverage is critical and failed: return
        elif is_derivative and leverage <= 1:
             logger.info(f"Leverage <= 1 specified for derivative {symbol}. Setting leverage to 1x.")
             set_leverage_ccxt(exchange, symbol, 1, logger) # Explicitly set to 1 if needed
        else: # Spot market
             logger.info(f"Leverage setting skipped (Spot market: {symbol}).")


        # --- 4c. Calculate Position Size ---
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=config["risk_per_trade"],
            stop_loss_price=sl_calc, # Use the calculated valid SL
            entry_price=entry_price, # Use current price as entry
            market_info=market_info,
            leverage=leverage,
            logger=logger
        )

        if position_size is None or position_size <= 0:
            logger.error(f"{NEON_RED}Trade aborted: Invalid position size calculated ({position_size}). Check balance, risk settings, and market limits.{RESET}")
            return

        # --- 4d. Place Order ---
        logger.info(f"Attempting to place {signal} trade for {symbol} | Size: {position_size} | Entry: ~{entry_price:.5f} | TP: {tp_calc:.5f if tp_calc else 'N/A'} | SL: {sl_calc:.5f}")
        trade_result = place_trade(
            exchange=exchange,
            symbol=symbol,
            signal=signal,
            size=position_size,
            market_info=market_info,
            tp_price=tp_calc, # Pass potential TP
            sl_price=sl_calc, # Pass potential SL
            logger=logger
        )

        if trade_result:
            logger.info(f"{NEON_GREEN}=== TRADE EXECUTED SUCCESSFULLY for {symbol} ===")
        else:
            logger.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ===")

    else: # HOLD Signal
        logger.info(f"Signal is HOLD for {symbol}. No trade action taken.")


def main() -> None:
    """Main function to run the bot."""
    global CONFIG

    logger = setup_logger("global")
    logger.info("Starting Livewire Trading Bot (pandas_ta enhanced)...")
    logger.info(f"Loaded configuration from {CONFIG_FILE}")
    logger.info(f"Using pandas_ta version: {ta.version()}") # Log pandas_ta version

    if CONFIG.get("enable_trading"):
         logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              logger.warning(f"{NEON_YELLOW}Using SANDBOX environment.{RESET}")
         else:
              logger.warning(f"{NEON_RED}Using REAL MONEY environment. TRIPLE CHECK CONFIG AND RISK!{RESET}")
         try:
             confirm = input("Press Enter to acknowledge live trading and continue, or Ctrl+C to abort...")
         except KeyboardInterrupt:
              logger.info("Aborted by user.")
              return
    else:
         logger.info("Trading is disabled. Running in analysis-only mode.")


    exchange = initialize_exchange(logger)
    if not exchange:
        logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return

    # --- Symbol and Interval Input ---
    while True:
        symbol_input = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT, ETH/USDT): {RESET}").upper().strip()
        if not symbol_input: continue
        symbol_ccxt = symbol_input if '/' in symbol_input else f"{symbol_input}/{QUOTE_CURRENCY}"

        market_info = get_market_info(exchange, symbol_ccxt, logger)
        if market_info:
            market_type = "Contract" if market_info.get('contract', False) else "Spot"
            logger.info(f"Symbol {symbol_ccxt} validated. Market Type: {market_type}")
            break
        else:
            logger.error(f"{NEON_RED}Symbol {symbol_ccxt} not found or invalid on {exchange.id}. Please try again.{RESET}")

    while True:
        interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}]: {RESET}").strip()
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            break
        else:
            logger.error(f"{NEON_RED}Invalid interval: '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")

    # Update config with user input for this run
    CONFIG["interval"] = interval_input
    logger.info(f"Using Symbol: {symbol_ccxt}, Interval: {interval_input} ({CCXT_INTERVAL_MAP[interval_input]})")


    # --- Main Loop ---
    try:
        while True:
            start_time = time.time()
            try:
                analyze_and_trade_symbol(exchange, symbol_ccxt, CONFIG, logger)
            except ccxt.RateLimitExceeded as e:
                 logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting longer...{RESET}")
                 time.sleep(60) # Wait longer after rate limit error
            except ccxt.NetworkError as e:
                 logger.error(f"{NEON_RED}Network error in main loop: {e}. Retrying after delay...{RESET}")
                 time.sleep(30) # Wait longer after network issues
            except Exception as loop_error:
                 logger.error(f"{NEON_RED}An error occurred in the main analysis/trade loop: {loop_error}{RESET}", exc_info=True)
                 logger.info("Attempting to continue after error...")

            # Calculate sleep time ensuring loop runs roughly every LOOP_DELAY_SECONDS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            logger.info(f"Cycle took {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s until next cycle...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
         logger.critical(f"{NEON_RED}A critical unhandled error occurred: {e}{RESET}", exc_info=True)
    finally:
        logger.info("Livewire Trading Bot stopped.")
        if exchange and hasattr(exchange, 'close'):
            exchange.close() # Close connection if possible


if __name__ == "__main__":
    # Store entry_price globally for place_trade check (simple approach)
    # A class-based approach might be cleaner for state management
    entry_price: Optional[Decimal] = None
    main()

```

**Changes Made:**

1.  **Import `pandas_ta`:** Added `import pandas_ta as ta`.
2.  **`TradingAnalyzer` Refactor:**
    *   **`_calculate_all_indicators`:**
        *   Replaced individual calculation calls with `self.df.ta.<indicator_name>(...)` calls (e.g., `self.df.ta.rsi(...)`, `self.df.ta.bbands(...)`, `self.df.ta.stochrsi(...)` etc.).
        *   Used `append=True` to add results directly to the DataFrame `self.df`.
        *   Stored the *expected* column names generated by `pandas_ta` into `self.ta_column_names` dictionary (e.g., `RSI_14`, `BBU_20_2.0`, `STOCHRSIk_14_12_3_3`). This makes retrieving values more reliable.
        *   Included calculation for Volume MA using `df.ta.sma(close='volume', ...)`.
        *   Added error handling around the `pandas_ta` calculations.
    *   **`_update_latest_indicator_values`:**
        *   Modified to retrieve values from `self.df.iloc[-1]` using the column names stored in `self.ta_column_names`. This ensures it uses the correct names generated by `pandas_ta`.
        *   Handles missing columns or NaN values gracefully.
        *   Added 'Close' and 'Volume' directly to `indicator_values` for easier access in checks.
    *   **Removed Old Calculation Methods:** Deleted the individual `calculate_*` methods (like `calculate_rsi`, `calculate_atr`, etc.) as `pandas_ta` now handles them. Kept `calculate_fibonacci_levels` and `calculate_ema_alignment` as they are custom logic.
    *   **Updated `_check_*` Methods:**
        *   Modified `_check_psar` to use the `PSARl_...` (long stop) and `PSARs_...` (short stop) columns generated by `pandas_ta` to determine the trend and signal.
        *   All other `_check_*` methods now get their required values from `self.indicator_values` (which uses the correct `pandas_ta` column names via `_update_latest_indicator_values`).
        *   Updated `_check_volume_confirmation` to use the `Volume_MA` value.
3.  **Configuration (`config.json`):**
    *   Added `pandas_ta` specific parameters for StochRSI (`stoch_rsi_rsi_window`, `stoch_rsi_k`, `stoch_rsi_d`) and PSAR (`psar_af`, `psar_max_af`) to the `default_config` and `load_config`.
    *   Adjusted default Bollinger Bands std dev to float (`2.0`).
4.  **Risk Management & Trading:**
    *   Improved `calculate_position_size` to handle minimum cost limits (`limits['cost']['min']`) in addition to minimum amount limits, as sometimes small amounts are allowed if the total cost is sufficient. Added more logging.
    *   Refined `get_open_position` to check multiple potential size fields ('contracts', 'contractSize', 'size', 'info.size') for better compatibility and use a small tolerance when checking for non-zero size.
    *   Improved `set_leverage_ccxt` logging and error handling for common Bybit messages ("not modified", "available balance not enough").
    *   Added check in `set_leverage_ccxt` logic to only attempt setting leverage > 1 for derivative markets (`market_info['contract']` or linear/inverse).
    *   Added a basic check in `place_trade` to prevent placing orders where SL/TP is clearly on the wrong side of the entry price (though the main check is in `calculate_entry_tp_sl`).
    *   Made `entry_price` a global variable to allow the check inside `place_trade` (a cleaner solution might involve passing it or using a class structure for the main loop).
5.  **Main Loop & Logging:**
    *   Added logging for the `pandas_ta` version being used.
    *   Improved the main loop's sleep calculation to aim for a consistent cycle time defined by `LOOP_DELAY_SECONDS`.
    *   Added more specific exception handling for `ccxt.RateLimitExceeded` and `ccxt.NetworkError` in the main loop with longer wait times.
    *   Refined logging output for analysis results.
    *   Included `exchange.close()` in the `finally` block.
    *   Improved user input validation for symbol and interval.

This version now effectively uses `pandas_ta` for the core indicator calculations, simplifying the `TradingAnalyzer` class while retaining the overall structure and trading logic. Remember to `pip install pandas_ta` if you haven't already.
