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
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 12
DEFAULT_K_WINDOW = 3
DEFAULT_D_WINDOW = 3
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
LOOP_DELAY_SECONDS = 15 # Increased delay slightly
QUOTE_CURRENCY = "USDT" # Assuming USDT pairs

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        # Basic redaction, might need refinement for more complex cases
        if API_KEY:
            msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found."""
    default_config = {
        "interval": "1m", # Default to ccxt compatible interval
        "analysis_interval": 5, # This seems unused, review needed
        "retry_delay": 5,
        "momentum_period": 7,
        "volume_ma_period": 15,
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "orderbook_limit": 25, # Reduced for faster fetch?
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for SL
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5, # Reduced multiplier
        "scalping_signal_threshold": 2.5, # Adjusted threshold for trading
        "fibonacci_window": 50,
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
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, # Added mfi weight
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
    # Use a generic logger name unless symbol-specific logs are strictly needed per run
    logger_name = "livewire_bot"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log") # Consolidated log file
    logger = logging.getLogger(logger_name)

    # Prevent adding multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
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
            'enableRateLimit': True, # Essential for respecting API limits
            'options': {
                'defaultType': 'linear', # Or 'inverse' or 'spot' depending on target market
                'adjustForTimeDifference': True,
             }
        })
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE{RESET}")
            exchange.set_sandbox_mode(True) # Use Bybit's testnet

        # Load markets to get symbol info, limits, precision etc.
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


# --- API Request Functions (Keep original for non-trading or fallback) ---
def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5, # Shorter backoff for quicker retries
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
    # V5 uses recvWindow, not timestamp directly in signature for GET
    recv_window = 5000 # Milliseconds the request is valid
    timestamp = str(int(time.time() * 1000))

    # Prepare signature string based on method
    if method == "GET":
        query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        payload = timestamp + API_KEY + str(recv_window) + query_string
    elif method == "POST":
        body = json.dumps(params, separators=(',', ':')) if params else ""
        payload = timestamp + API_KEY + str(recv_window) + body
    else:
         if logger: logger.error(f"Unsupported HTTP method: {method}")
         return None

    # Generate signature
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
        "timeout": 15 # Slightly longer timeout
    }

    if method == "GET":
        request_kwargs["params"] = params
    elif method == "POST":
        request_kwargs["data"] = json.dumps(params) # POST uses data, not json for V5 signing

    try:
        response = session.request(**request_kwargs)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_response = response.json()

        # Check Bybit's specific return code
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
        # Use fetch_ticker for current price info
        ticker = exchange.fetch_ticker(symbol)
        if ticker and 'last' in ticker and ticker['last'] is not None:
            return Decimal(str(ticker['last']))
        else:
            logger.warning(f"{NEON_YELLOW}Could not fetch last price for {symbol} via fetch_ticker.{RESET}")
            # Fallback: fetch recent trades if ticker fails
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
        # Fetch OHLCV data
        # CCXT returns data in format: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        if not ohlcv:
            if logger: logger.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True) # Set timestamp as index for easier TA lib usage

        # Convert columns to numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True) # Drop rows with parsing errors

        if df.empty:
             if logger: logger.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing.{RESET}")

        return df

    except ccxt.NetworkError as e:
        if logger: logger.error(f"{NEON_RED}Network error fetching klines for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        if logger: logger.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
    except Exception as e:
        if logger: logger.error(f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}{RESET}", exc_info=True)
    return pd.DataFrame() # Return empty DataFrame on error


def fetch_orderbook_ccxt(exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger) -> Optional[Dict]:
    """Fetch orderbook data using ccxt with retries."""
    # CCXT's built-in rate limiter handles retries for 429, but we can add retries for other network issues
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
            return None # Don't retry on exchange errors usually
        except Exception as e:
            logger.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            # Maybe retry unexpected errors? Or maybe not. Let's not retry for now.
            return None

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None

# --- Trading Analyzer Class (largely unchanged, but ensure column names match DataFrame) ---
class TradingAnalyzer:
    """Analyze trading data and generate scalping signals."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        symbol: str,
        interval: str,
    ) -> None:
        self.df = df # Expects DataFrame with index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.symbol = symbol
        self.interval = interval # This is the original interval ('1', '5', etc.)
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(interval, "1m") # CCXT format interval
        self.indicator_values: Dict[str, float] = {}
        self.scalping_signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.weights = config["weight_sets"]["scalping"]
        self.fib_levels_data: Dict[str, Decimal] = {}
        self._calculate_all_indicators() # Calculate indicators on initialization

    def _calculate_all_indicators(self):
        """Helper to calculate all enabled indicators."""
        if self.df.empty:
            self.logger.warning(f"{NEON_YELLOW}DataFrame is empty, cannot calculate indicators for {self.symbol}.{RESET}")
            return

        # Make sure we have enough data points for the largest window required
        min_required_data = max(
            self.config.get("atr_period", DEFAULT_ATR_PERIOD),
            self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
            self.config.get("rsi_period", DEFAULT_RSI_WINDOW),
            self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD),
            self.config.get("cci_window", DEFAULT_CCI_WINDOW),
            self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW),
            self.config.get("mfi_window", DEFAULT_MFI_WINDOW),
            DEFAULT_STOCH_WINDOW + DEFAULT_K_WINDOW + DEFAULT_D_WINDOW, # Stoch RSI needs more history
            DEFAULT_SMA_10_WINDOW,
            self.config.get("momentum_period", 7) + 1, # Momentum needs shift
            self.config.get("volume_ma_period", 15),
            self.config.get("fibonacci_window", 50)
        )

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min required: {min_required_data}). Results may be inaccurate.{RESET}")
             # Attempt calculations anyway, but they might return NaN

        # Calculate indicators (store series on df for potential reuse/debugging)
        self.df['atr'] = self.calculate_atr()
        self.df['ema_short'] = self.calculate_ema(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
        self.df['ema_long'] = self.calculate_ema(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        self.df['momentum'] = self.calculate_momentum()
        self.df['cci'] = self.calculate_cci()
        self.df['wr'] = self.calculate_williams_r()
        self.df['mfi'] = self.calculate_mfi()
        self.df['vwap'] = self.calculate_vwap()
        # self.df['psar'] = self.calculate_psar() # PSAR calculation needs review/external lib
        self.df['sma_10'] = self.calculate_sma_10()
        stoch_rsi_df = self.calculate_stoch_rsi()
        if not stoch_rsi_df.empty:
            self.df = self.df.join(stoch_rsi_df) # Join K and D lines
        self.df['rsi'] = self.calculate_rsi()
        bbands_df = self.calculate_bollinger_bands()
        if not bbands_df.empty:
            self.df = self.df.join(bbands_df) # Join BBands

        # Store the latest values in indicator_values dict
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels

    def _update_latest_indicator_values(self):
        """Update the indicator_values dictionary with the latest calculated values."""
        if self.df.empty: return
        latest = self.df.iloc[-1]
        self.indicator_values["ATR"] = float(latest.get('atr', np.nan))
        self.indicator_values["EMA_Short"] = float(latest.get('ema_short', np.nan))
        self.indicator_values["EMA_Long"] = float(latest.get('ema_long', np.nan))
        self.indicator_values["Momentum"] = float(latest.get('momentum', np.nan))
        self.indicator_values["CCI"] = float(latest.get('cci', np.nan))
        self.indicator_values["Williams_R"] = float(latest.get('wr', np.nan))
        self.indicator_values["MFI"] = float(latest.get('mfi', np.nan))
        self.indicator_values["VWAP"] = float(latest.get('vwap', np.nan))
        # self.indicator_values["PSAR"] = float(latest.get('psar', np.nan))
        self.indicator_values["SMA10"] = float(latest.get('sma_10', np.nan))
        self.indicator_values["StochRSI_K"] = float(latest.get('k', np.nan))
        self.indicator_values["StochRSI_D"] = float(latest.get('d', np.nan))
        self.indicator_values["RSI"] = float(latest.get('rsi', np.nan))
        self.indicator_values["BB_Upper"] = float(latest.get('bb_upper', np.nan))
        self.indicator_values["BB_Middle"] = float(latest.get('bb_mid', np.nan))
        self.indicator_values["BB_Lower"] = float(latest.get('bb_lower', np.nan))
        # Add others if calculated


    # --- Indicator Calculation Methods (Using DataFrame index and columns) ---
    # Ensure these methods handle potential NaNs gracefully, especially at the start.

    def calculate_atr(self, period: Optional[int] = None) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        period = period or self.config.get("atr_period", DEFAULT_ATR_PERIOD)
        try:
            if not all(col in self.df.columns for col in ['high', 'low', 'close']):
                 raise KeyError("Missing HLC columns for ATR")
            high_low = self.df["high"] - self.df["low"]
            high_close = np.abs(self.df["high"] - self.df["close"].shift())
            low_close = np.abs(self.df["low"] - self.df["close"].shift())

            # Use .fillna(0) for the first row's shift result
            tr = pd.concat([high_low, high_close.fillna(0), low_close.fillna(0)], axis=1).max(axis=1, skipna=False)
            atr = tr.ewm(alpha=1/period, adjust=False).mean() # Smoothed ATR (common method)
            # atr = tr.rolling(window=period).mean() # Simple Moving Average ATR
            return atr
        except KeyError as e:
            self.logger.error(f"{NEON_RED}ATR error: Missing column {e}{RESET}")
        except Exception as e:
             self.logger.error(f"{NEON_RED}ATR calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index) # Return empty series on error


    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        window = window or self.config.get("fibonacci_window", 50)
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
            if diff > 0: # Avoid division by zero if high == low
                for level in FIB_LEVELS:
                    level_name = f"Fib_{level * 100:.1f}%"
                    # Calculation depends on trend direction, assume recent trend for simplicity
                    # Here, calculate levels down from high
                    levels[level_name] = (high - (diff * Decimal(str(level)))).quantize(Decimal("0.0001"), rounding=ROUND_DOWN) # Adjust precision as needed
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
             # Try calculating if not already done
             self.calculate_fibonacci_levels()
             if not self.fib_levels_data:
                  return [] # Return empty if calculation failed

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


    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        try:
             if 'close' not in self.df.columns: raise KeyError("'close' column missing")
             return self.df["close"].rolling(window=window, min_periods=window).mean()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}SMA error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}SMA calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        try:
            if 'close' not in self.df.columns: raise KeyError("'close' column missing")
            return self.df["close"].ewm(span=window, adjust=False, min_periods=window).mean()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}EMA error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}EMA calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)

    def calculate_ema_alignment(self) -> float:
        """Calculate EMA alignment score based on latest values."""
        ema_short = self.indicator_values.get("EMA_Short")
        ema_long = self.indicator_values.get("EMA_Long")
        current_price = self.df["close"].iloc[-1] if not self.df.empty else None

        if ema_short is None or ema_long is None or current_price is None or pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            return 0.0

        # Bullish: price > short EMA > long EMA
        if current_price > ema_short > ema_long:
            return 1.0
        # Bearish: price < short EMA < long EMA
        elif current_price < ema_short < ema_long:
            return -1.0
        # Other cases (choppy, crossing)
        else:
            return 0.0


    def calculate_momentum(self, period: Optional[int] = None) -> pd.Series:
        """Calculate Momentum."""
        period = period or self.config.get("momentum_period", 7)
        try:
            if 'close' not in self.df.columns: raise KeyError("'close' column missing")
            # Avoid division by zero if previous close was 0
            shifted_close = self.df["close"].shift(period)
            # Use np.where to handle potential division by zero or NaN
            momentum = np.where(
                shifted_close.isna() | (shifted_close == 0),
                0.0, # Assign 0 if shifted value is NaN or zero
                (self.df["close"] - shifted_close) / shifted_close * 100
            )
            return pd.Series(momentum, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Momentum error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}Momentum calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)

    def calculate_cci(self, window: Optional[int] = None) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)."""
        window = window or self.config.get("cci_window", DEFAULT_CCI_WINDOW)
        try:
            if not all(c in self.df.columns for c in ['high', 'low', 'close']):
                raise KeyError("Missing HLC columns for CCI")

            tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            sma_tp = tp.rolling(window=window, min_periods=window).mean()
            # Calculate mean deviation using rolling apply
            mean_dev = tp.rolling(window=window, min_periods=window).apply(
                lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
            )

            # Handle potential division by zero in mean deviation
            cci = np.where(
                mean_dev == 0,
                0.0, # Assign 0 if mean deviation is zero
                (tp - sma_tp) / (0.015 * mean_dev)
            )
            return pd.Series(cci, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}CCI error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}CCI calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)


    def calculate_williams_r(self, window: Optional[int] = None) -> pd.Series:
        """Calculate Williams %R."""
        window = window or self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
        try:
            if not all(c in self.df.columns for c in ['high', 'low', 'close']):
                 raise KeyError("Missing HLC columns for Williams %R")

            highest_high = self.df["high"].rolling(window=window, min_periods=window).max()
            lowest_low = self.df["low"].rolling(window=window, min_periods=window).min()
            range_ = highest_high - lowest_low

            # Handle potential division by zero if highest_high == lowest_low
            wr = np.where(
                range_ == 0,
                -50.0, # Assign a neutral value like -50 if range is zero
                (highest_high - self.df["close"]) / range_ * -100
            )
            return pd.Series(wr, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Williams %R error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}Williams %R calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)

    def calculate_mfi(self, window: Optional[int] = None) -> pd.Series:
        """Calculate Money Flow Index (MFI)."""
        window = window or self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
        try:
            if not all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 raise KeyError("Missing HLCV columns for MFI")

            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            raw_money_flow = typical_price * self.df['volume']
            typical_price_diff = typical_price.diff()

            positive_flow = raw_money_flow.where(typical_price_diff > 0, 0).rolling(window=window, min_periods=window).sum()
            negative_flow = raw_money_flow.where(typical_price_diff < 0, 0).rolling(window=window, min_periods=window).sum()

            # Handle potential division by zero if negative_flow is 0
            mfi = np.where(
                negative_flow == 0,
                100.0, # MFI is 100 if negative flow is zero
                100 - (100 / (1 + positive_flow / negative_flow))
             )
             # Handle case where both are zero (assign neutral 50)
             mfi = np.where((positive_flow == 0) & (negative_flow == 0), 50.0, mfi)

            return pd.Series(mfi, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}MFI error: Missing columns - {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}MFI calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype='float64', index=self.df.index)

    def calculate_vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        # VWAP is typically calculated daily, but can be approximated over the loaded data window
        try:
             if not all(c in self.df.columns for c in ['high', 'low', 'close', 'volume']):
                 raise KeyError("Missing HLCV columns for VWAP")

             typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
             # Calculate cumulative VWAP over the loaded period
             # Note: A true daily VWAP requires resetting at the start of each day
             vwap = (self.df["volume"] * typical_price).cumsum() / self.df["volume"].cumsum()
             return vwap.fillna(method='bfill') # Backfill initial NaNs if any

        except KeyError as e:
             self.logger.error(f"{NEON_RED}VWAP error: Missing columns - {e}{RESET}")
        except ZeroDivisionError:
            self.logger.warning(f"{NEON_YELLOW}VWAP calculation warning: Cumulative volume is zero at start.{RESET}")
            return pd.Series(dtype='float64', index=self.df.index) # Return empty series if volume is zero
        except Exception as e:
            self.logger.error(f"{NEON_RED}VWAP calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype='float64', index=self.df.index)

    def calculate_psar(self, acceleration: float = 0.02, max_acceleration: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR (PSAR). Basic implementation."""
        # WARNING: This is a simplified PSAR. For accuracy, consider ta-lib or pandas-ta.
        # This implementation might differ from standard versions.
        self.logger.warning(f"{NEON_YELLOW}Using simplified PSAR calculation. Consider a dedicated TA library for accuracy.{RESET}")
        try:
            if not all(c in self.df.columns for c in ['high', 'low']):
                 raise KeyError("Missing HL columns for PSAR")
            if len(self.df) < 2: return pd.Series(dtype='float64', index=self.df.index)

            psar = pd.Series(index=self.df.index, dtype="float64")
            trend = pd.Series(index=self.df.index, dtype="int64") # 1 for long, -1 for short
            af = pd.Series(index=self.df.index, dtype="float64")
            ep = pd.Series(index=self.df.index, dtype="float64") # Extreme Point

            # Initial values (guess trend based on first two closes)
            trend.iloc[0] = 1 if self.df['close'].iloc[1] > self.df['close'].iloc[0] else -1
            psar.iloc[0] = self.df['high'].iloc[0] if trend.iloc[0] == -1 else self.df['low'].iloc[0]
            af.iloc[0] = acceleration
            ep.iloc[0] = self.df['high'].iloc[0] if trend.iloc[0] == 1 else self.df['low'].iloc[0]

            for i in range(1, len(self.df)):
                # Calculate current SAR
                psar_i = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])

                # Check for trend reversal
                if trend.iloc[i-1] == 1: # Uptrend
                    psar_i = min(psar_i, self.df['low'].iloc[i-1], self.df['low'].iloc[i-2] if i > 1 else self.df['low'].iloc[i-1])
                    if self.df['low'].iloc[i] < psar_i: # Reversal to downtrend
                        trend.iloc[i] = -1
                        psar.iloc[i] = ep.iloc[i-1] # SAR is the prior EP
                        ep.iloc[i] = self.df['low'].iloc[i]
                        af.iloc[i] = acceleration
                    else: # Continue uptrend
                        trend.iloc[i] = 1
                        psar.iloc[i] = psar_i
                        if self.df['high'].iloc[i] > ep.iloc[i-1]:
                            ep.iloc[i] = self.df['high'].iloc[i]
                            af.iloc[i] = min(af.iloc[i-1] + acceleration, max_acceleration)
                        else:
                            ep.iloc[i] = ep.iloc[i-1]
                            af.iloc[i] = af.iloc[i-1]
                else: # Downtrend (trend == -1)
                    psar_i = max(psar_i, self.df['high'].iloc[i-1], self.df['high'].iloc[i-2] if i > 1 else self.df['high'].iloc[i-1])
                    if self.df['high'].iloc[i] > psar_i: # Reversal to uptrend
                        trend.iloc[i] = 1
                        psar.iloc[i] = ep.iloc[i-1] # SAR is the prior EP
                        ep.iloc[i] = self.df['high'].iloc[i]
                        af.iloc[i] = acceleration
                    else: # Continue downtrend
                        trend.iloc[i] = -1
                        psar.iloc[i] = psar_i
                        if self.df['low'].iloc[i] < ep.iloc[i-1]:
                            ep.iloc[i] = self.df['low'].iloc[i]
                            af.iloc[i] = min(af.iloc[i-1] + acceleration, max_acceleration)
                        else:
                            ep.iloc[i] = ep.iloc[i-1]
                            af.iloc[i] = af.iloc[i-1]

            self.indicator_values["PSAR"] = float(psar.iloc[-1]) if not psar.empty else np.nan
            return psar

        except KeyError as e:
             self.logger.error(f"{NEON_RED}PSAR error: Missing columns - {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}PSAR calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype='float64', index=self.df.index)


    def calculate_sma_10(self) -> pd.Series:
        """Calculate SMA with 10-period window."""
        return self.calculate_sma(DEFAULT_SMA_10_WINDOW)

    def calculate_stoch_rsi(self) -> pd.DataFrame:
        """Calculate Stochastic RSI K and D lines."""
        rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
        stoch_window = DEFAULT_STOCH_WINDOW
        k_window = DEFAULT_K_WINDOW
        d_window = DEFAULT_D_WINDOW
        min_periods = stoch_window + k_window + d_window - 2 # Min periods needed

        try:
            if 'close' not in self.df.columns: raise KeyError("'close' column missing for RSI base")
            rsi = self.calculate_rsi(window=rsi_period) # Calculate RSI first
            if rsi.isna().all():
                self.logger.warning(f"{NEON_YELLOW}RSI calculation failed or resulted in NaNs, cannot calculate StochRSI.{RESET}")
                return pd.DataFrame(index=self.df.index)

            # Calculate StochRSI
            min_rsi = rsi.rolling(window=stoch_window, min_periods=stoch_window).min()
            max_rsi = rsi.rolling(window=stoch_window, min_periods=stoch_window).max()
            stoch_rsi = np.where(
                max_rsi == min_rsi,
                 0.0, # Avoid division by zero, assign 0
                (rsi - min_rsi) / (max_rsi - min_rsi) * 100 # Scale to 0-100
            )
            stoch_rsi_series = pd.Series(stoch_rsi, index=self.df.index)

            # Calculate %K and %D
            k_line = stoch_rsi_series.rolling(window=k_window, min_periods=k_window).mean()
            d_line = k_line.rolling(window=d_window, min_periods=d_window).mean()

            return pd.DataFrame({"k": k_line, "d": d_line}, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}Stoch RSI error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}Stoch RSI calculation failed: {e}{RESET}", exc_info=True)
        return pd.DataFrame(index=self.df.index) # Return empty dataframe on error

    def calculate_rsi(self, window: Optional[int] = None) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        window = window or self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
        try:
            if 'close' not in self.df.columns: raise KeyError("'close' column missing")
            delta = self.df["close"].diff()
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1/window, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/window, adjust=False).mean()

            # Avoid division by zero
            rs = np.where(loss == 0, np.inf, gain / loss) # If loss is 0, RS is infinite
            rsi = 100 - (100 / (1 + rs))
            rsi = np.where(loss == 0, 100.0, rsi) # If loss is 0, RSI is 100

            return pd.Series(rsi, index=self.df.index)

        except KeyError as e:
            self.logger.error(f"{NEON_RED}RSI error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}RSI calculation failed: {e}{RESET}", exc_info=True)
        return pd.Series(dtype="float64", index=self.df.index)

    def calculate_bollinger_bands(self) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
        std_dev = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
        try:
            if 'close' not in self.df.columns: raise KeyError("'close' column missing")
            rolling_mean = self.df["close"].rolling(window=period, min_periods=period).mean()
            rolling_std = self.df["close"].rolling(window=period, min_periods=period).std()
            bb_upper = rolling_mean + (rolling_std * std_dev)
            bb_lower = rolling_mean - (rolling_std * std_dev)

            return pd.DataFrame({
                "bb_upper": bb_upper,
                "bb_mid": rolling_mean,
                "bb_lower": bb_lower,
            }, index=self.df.index)
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Bollinger Bands error: {e}{RESET}")
        except Exception as e:
            self.logger.error(f"{NEON_RED}Bollinger Bands calculation failed: {e}{RESET}", exc_info=True)
        return pd.DataFrame(index=self.df.index)

    # --- Signal Generation & Scoring ---

    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generate trading signal based on weighted indicator scores."""
        self.scalping_signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        signal_score = 0.0
        active_indicator_count = 0

        if self.df.empty or pd.isna(current_price):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Missing data or current price.{RESET}")
             return "HOLD"

        # Update latest values just in case
        self._update_latest_indicator_values()

        # Iterate through configured indicators and add weighted scores
        for indicator, enabled in self.config.get("indicators", {}).items():
            if enabled and indicator in self.weights:
                score_contribution = 0.0
                weight = self.weights[indicator]
                # Get score contribution from specific check methods
                check_method_name = f"_check_{indicator}"
                if hasattr(self, check_method_name):
                    method = getattr(self, check_method_name)
                    score_contribution = method() * weight
                    if not pd.isna(score_contribution): # Only add if not NaN
                        signal_score += score_contribution
                        active_indicator_count += 1
                        # self.logger.debug(f"Indicator {indicator}: contribution {score_contribution:.4f}") # Debug logging
                    # else:
                        # self.logger.debug(f"Indicator {indicator}: NaN score contribution") # Debug logging

                else:
                    self.logger.warning(f"No check method found for indicator: {indicator}")

        # Add orderbook score if available
        if orderbook_data:
             orderbook_score = self._check_orderbook(orderbook_data, current_price)
             # Give orderbook a fixed weight or make it configurable? Let's use a small fixed weight for now.
             orderbook_weight = 0.15
             signal_score += orderbook_score * orderbook_weight
             # self.logger.debug(f"Orderbook score: {orderbook_score:.4f}, contribution: {orderbook_score * orderbook_weight:.4f}") # Debug


        # Normalize score? Or use absolute threshold? Use threshold for now.
        threshold = self.config["scalping_signal_threshold"]
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


    def _check_ema_alignment(self) -> float:
        # This calls the method that uses the latest indicator_values
        return self.calculate_ema_alignment()

    def _check_momentum(self) -> float:
        """Check momentum for signal scoring."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return 0.0
        # Simple: positive momentum = bullish, negative = bearish
        return 1.0 if momentum > 0.1 else -1.0 if momentum < -0.1 else 0.0 # Add small threshold

    def _check_volume_confirmation(self) -> float:
        """Check volume confirmation for signal scoring."""
        if self.df.empty or 'volume' not in self.df.columns: return 0.0
        try:
            ma_period = self.config.get("volume_ma_period", 15)
            if len(self.df) < ma_period: return 0.0 # Not enough data

            volume_ma = self.df["volume"].rolling(window=ma_period).mean().iloc[-1]
            current_volume = self.df["volume"].iloc[-1]
            multiplier = self.config.get("volume_confirmation_multiplier", 1.5)

            if pd.isna(volume_ma) or pd.isna(current_volume) or volume_ma == 0:
                return 0.0

            if current_volume > volume_ma * multiplier:
                return 1.0 # High volume confirmation
            elif current_volume < volume_ma / multiplier:
                 return -0.5 # Low volume suggests weakness (less penalty than high confirmation)
            else:
                return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed: {e}{RESET}")
            return 0.0

    def _check_stoch_rsi(self) -> float:
        """Check Stochastic RSI for signal scoring."""
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return 0.0

        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)

        # Bullish signal: K crosses above D in oversold territory
        # Bearish signal: K crosses below D in overbought territory
        # Simple state check for now:
        if k < oversold and d < oversold:
            return 1.0 # Oversold
        elif k > overbought and d > overbought:
            return -1.0 # Overbought
        elif k > d:
            return 0.5 # K above D (generally bullish bias)
        elif k < d:
             return -0.5 # K below D (generally bearish bias)
        else:
             return 0.0


    def _check_rsi(self) -> float:
        """Check RSI for signal scoring."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return 0.0
        # Standard overbought/oversold
        if rsi < 30: return 1.0
        if rsi > 70: return -1.0
        # Trend indication (above/below 50)
        if rsi > 55: return 0.5 # Mild bullish
        if rsi < 45: return -0.5 # Mild bearish
        return 0.0 # Neutral range

    def _check_cci(self) -> float:
        """Check CCI for signal scoring."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return 0.0
        if cci < -100: return 1.0 # Oversold
        if cci > 100: return -1.0 # Overbought
        # Trend indication (crossing zero) - could add lookback here
        # Simple check based on current value vs 0
        if cci > 0: return 0.3
        if cci < 0: return -0.3
        return 0.0

    def _check_wr(self) -> float: # Renamed from _check_williams_r
        """Check Williams %R for signal scoring."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return 0.0
        if wr < -80: return 1.0 # Oversold
        if wr > -20: return -1.0 # Overbought
        # Mid-range check (less common for WR)
        if wr < -50: return 0.3 # Below midpoint (bearish bias)
        if wr > -50: return -0.3 # Above midpoint (bullish bias)
        return 0.0

    def _check_psar(self) -> float:
        """Check PSAR for signal scoring."""
        # Note: PSAR calculation needs verification
        psar = self.indicator_values.get("PSAR", np.nan)
        if pd.isna(psar) or self.df.empty: return 0.0
        last_close = self.df["close"].iloc[-1]
        if pd.isna(last_close): return 0.0

        # Basic: Price above SAR = Bullish, Price below SAR = Bearish
        return 1.0 if last_close > psar else -1.0 if last_close < psar else 0.0

    def _check_sma_10(self) -> float:
        """Check SMA_10 for signal scoring."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        if pd.isna(sma_10) or self.df.empty: return 0.0
        last_close = self.df["close"].iloc[-1]
        if pd.isna(last_close): return 0.0
        # Price vs SMA
        return 1.0 if last_close > sma_10 else -1.0 if last_close < sma_10 else 0.0

    def _check_vwap(self) -> float:
        """Check VWAP for signal scoring."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        if pd.isna(vwap) or self.df.empty: return 0.0
        last_close = self.df["close"].iloc[-1]
        if pd.isna(last_close): return 0.0
        # Price vs VWAP is often used for intraday bias
        return 1.0 if last_close > vwap else -1.0 if last_close < vwap else 0.0

    def _check_mfi(self) -> float:
        """Check MFI for signal scoring."""
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return 0.0
        # Similar to RSI, but volume-weighted
        if mfi < 20: return 1.0 # Oversold
        if mfi > 80: return -1.0 # Overbought
        if mfi > 55: return 0.5 # Mild bullish
        if mfi < 45: return -0.5 # Mild bearish
        return 0.0

    def _check_bollinger_bands(self) -> float:
        """Check Bollinger Bands for signal scoring."""
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or self.df.empty: return 0.0
        last_close = self.df["close"].iloc[-1]
        if pd.isna(last_close): return 0.0

        # Mean reversion signal: price touches/exceeds bands
        if last_close < bb_lower: return 1.0 # Potential bounce (buy)
        if last_close > bb_upper: return -1.0 # Potential pullback (sell)
        # Could add checks for BB width (squeeze) later
        return 0.0

    def _check_orderbook(self, orderbook_data: Dict, current_price: Decimal) -> float:
        """Analyze order book depth for immediate pressure."""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            if not bids or not asks: return 0.0

            # Look at volume imbalance within a small % range of the current price
            price_range_percent = Decimal("0.001") # 0.1% range around current price
            price_range = current_price * price_range_percent

            relevant_bid_volume = sum(Decimal(str(bid[1])) for bid in bids if Decimal(str(bid[0])) >= current_price - price_range)
            relevant_ask_volume = sum(Decimal(str(ask[1])) for ask in asks if Decimal(str(ask[0])) <= current_price + price_range)

            if relevant_bid_volume == 0 and relevant_ask_volume == 0: return 0.0 # Avoid division by zero
            if relevant_ask_volume == 0: return 1.0 # Infinite bid pressure (strong buy)
            if relevant_bid_volume == 0: return -1.0 # Infinite ask pressure (strong sell)

            imbalance_ratio = relevant_bid_volume / relevant_ask_volume

            # Score based on imbalance
            if imbalance_ratio > 1.5: return 1.0   # Strong bid support
            elif imbalance_ratio > 1.1: return 0.5 # Moderate bid support
            elif imbalance_ratio < 0.67: return -1.0 # Strong ask pressure (1 / 1.5)
            elif imbalance_ratio < 0.91: return -0.5 # Moderate ask pressure (1 / 1.1)
            else: return 0.0 # Relatively balanced

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed: {e}{RESET}")
            return 0.0

    # --- Risk Management Calculations ---

    def calculate_entry_tp_sl(
        self, current_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """Calculate entry, take profit, and stop loss levels based on ATR."""
        atr = self.indicator_values.get("ATR")
        if atr is None or pd.isna(atr) or atr == 0 or current_price is None or pd.isna(current_price):
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Missing ATR or Price. ATR: {atr}, Price: {current_price}{RESET}")
            return None, None, None

        try:
            atr_decimal = Decimal(str(atr))
            entry = current_price # Market order entry assumption
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            if signal == "BUY":
                take_profit = entry + (atr_decimal * tp_multiple)
                stop_loss = entry - (atr_decimal * sl_multiple)
            elif signal == "SELL":
                take_profit = entry - (atr_decimal * tp_multiple)
                stop_loss = entry + (atr_decimal * sl_multiple)
            else: # HOLD signal
                return entry, None, None # No TP/SL needed for hold

            # Basic validation: SL should not cross entry for BUY, TP should be profitable
            if signal == "BUY" and (stop_loss >= entry or take_profit <= entry):
                 self.logger.warning(f"[{self.symbol}] BUY signal TP/SL calculation invalid: Entry={entry:.4f}, TP={take_profit:.4f}, SL={stop_loss:.4f}, ATR={atr_decimal:.4f}")
                 return entry, None, None # Invalidate if nonsensical
            if signal == "SELL" and (stop_loss <= entry or take_profit >= entry):
                 self.logger.warning(f"[{self.symbol}] SELL signal TP/SL calculation invalid: Entry={entry:.4f}, TP={take_profit:.4f}, SL={stop_loss:.4f}, ATR={atr_decimal:.4f}")
                 return entry, None, None # Invalidate if nonsensical

            return entry, take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating TP/SL: {e}{RESET}", exc_info=True)
             return None, None, None

    def calculate_confidence(self) -> float:
        """Calculate confidence score (placeholder). Could use score magnitude."""
        # Placeholder - could be linked to signal_score magnitude relative to threshold
        return 75.0 # Keep simple for now


# --- Trading Logic ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the free balance for a specific currency."""
    try:
        balance_info = exchange.fetch_balance()
        # Structure depends on defaultType (linear, inverse, spot)
        # For linear (USDT margined):
        if currency in balance_info['total']:
             # Use 'free' balance for placing new orders
             free_balance = balance_info.get(currency, {}).get('free')
             if free_balance is not None:
                  return Decimal(str(free_balance))
             else:
                  # Sometimes balance might be under the top level keys directly
                  free_balance_alt = balance_info['free'].get(currency)
                  if free_balance_alt is not None:
                       return Decimal(str(free_balance_alt))
                  else:
                       logger.warning(f"{NEON_YELLOW}Could not find 'free' balance for {currency}. Check balance structure: {balance_info}{RESET}")
                       return None
        else:
            logger.warning(f"{NEON_YELLOW}Currency {currency} not found in balance info: {balance_info.keys()}{RESET}")
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
        market = exchange.market(symbol)
        if market:
            return market
        else:
            logger.error(f"{NEON_RED}Market info not found for {symbol}. Ensure markets are loaded.{RESET}")
            # Attempt to reload markets just in case
            exchange.load_markets(reload=True)
            market = exchange.market(symbol)
            if market:
                 logger.info(f"Reloaded markets and found {symbol}.")
                 return market
            else:
                 logger.error(f"{NEON_RED}Market {symbol} still not found after reload.{RESET}")
                 return None
    except ccxt.BadSymbol:
         logger.error(f"{NEON_RED}Symbol '{symbol}' is not supported by the exchange.{RESET}")
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
    leverage: int = 1,
    logger: logging.Logger = None
) -> Optional[Decimal]:
    """Calculates position size based on risk, SL distance, leverage, and market limits."""
    if balance is None or balance <= 0 or stop_loss_price is None or entry_price is None or stop_loss_price == entry_price:
        if logger: logger.warning(f"{NEON_YELLOW}Invalid inputs for position sizing: Balance={balance}, Entry={entry_price}, SL={stop_loss_price}{RESET}")
        return None

    try:
        risk_amount = balance * Decimal(str(risk_per_trade))
        sl_distance = abs(entry_price - stop_loss_price)
        if sl_distance <= 0:
             if logger: logger.warning(f"{NEON_YELLOW}Stop loss distance is zero or negative, cannot calculate position size.{RESET}")
             return None

        # Size in base currency units (e.g., BTC for BTC/USDT)
        base_size_unadjusted = risk_amount / sl_distance

        # Apply leverage (size in quote currency / price = size in base currency)
        # For linear contracts (USDT margined), the size is usually in the base currency (e.g., BTC amount).
        # Leverage multiplies the position value, not directly the size usually.
        # The risk calculation already accounts for the loss potential.
        # Let's assume 'size' refers to the contract quantity (usually in base currency).
        position_size = base_size_unadjusted

        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        precision = market_info.get('precision', {})
        amount_precision = precision.get('amount') # Number of decimal places for amount

        min_amount = Decimal(str(amount_limits.get('min', '0'))) if amount_limits.get('min') is not None else Decimal('0')
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')

        if position_size < min_amount:
            if logger: logger.warning(f"{NEON_YELLOW}Calculated size {position_size:.8f} is below minimum order size {min_amount:.8f}. No trade.{RESET}")
            return None
        if position_size > max_amount:
             if logger: logger.warning(f"{NEON_YELLOW}Calculated size {position_size:.8f} exceeds maximum order size {max_amount:.8f}. Capping size.{RESET}")
             position_size = max_amount

        # Apply amount precision (rounding down to avoid exceeding balance/risk)
        if amount_precision is not None:
            # Use Decimal quantize for precise rounding
            rounding_factor = Decimal('1e-' + str(int(amount_precision))) # e.g., 0.001 for 3 decimal places
            position_size = position_size.quantize(rounding_factor, rounding=ROUND_DOWN)
        else:
             # Fallback if precision not defined (less likely with ccxt)
             position_size = Decimal(math.floor(float(position_size) * 10**8) / 10**8) # Approx rounding down

        # Final check if rounded size is still above minimum
        if position_size < min_amount:
            if logger: logger.warning(f"{NEON_YELLOW}Rounded size {position_size:.8f} is below minimum order size {min_amount:.8f}. No trade.{RESET}")
            return None

        if logger: logger.info(f"Calculated position size: {position_size:.8f} {market_info.get('base', '')}")
        return position_size

    except KeyError as e:
         if logger: logger.error(f"{NEON_RED}Position sizing error: Missing market info key {e}. Market: {market_info}{RESET}")
         return None
    except Exception as e:
        if logger: logger.error(f"{NEON_RED}Error calculating position size: {e}{RESET}", exc_info=True)
        return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """Checks if there's an open position for the given symbol."""
    try:
        # fetch_positions requires the symbol list usually
        positions = exchange.fetch_positions([symbol])
        # Filter for the specific symbol and non-zero size
        for pos in positions:
            # Check if 'contracts' or 'size' is present and non-zero
            # The key might vary ('contracts', 'size', 'contractSize', 'positionAmt')
            size = pos.get('contracts') # Common for futures
            if size is None: size = pos.get('contractSize')
            if size is None: size = pos.get('size') # Common for spot margin?
            if size is None: size = pos.get('positionAmt') # Binance specific?

            # info often contains the raw exchange data
            info = pos.get('info', {})
            if size is None: size = info.get('size') # Check raw info size
            if size is None: size = info.get('positionValue') # Another potential key


            # Ensure size is treated as Decimal and check if > 0
            if size is not None and abs(Decimal(str(size))) > 0:
                 logger.info(f"Found open position for {symbol}: Size={size}")
                 # Return the raw position data which might be useful
                 return pos
        # No position found with non-zero size
        return None
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Some exchanges might throw error if no positions, handle gracefully
        if 'position idx not exist' in str(e).lower() or 'no position found' in str(e).lower():
             logger.info(f"No open position found for {symbol} (Exchange message).")
             return None
        logger.error(f"{NEON_RED}Exchange error fetching positions for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    return None # Return None on error

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT."""
    if not hasattr(exchange, 'set_leverage'):
         logger.warning(f"{NEON_YELLOW}Exchange {exchange.id} does not support setting leverage via ccxt standard method.{RESET}")
         return False
    try:
        # Some exchanges require setting for both long/short or buy/sell
        # Bybit V5 seems unified, try setting it once.
        exchange.set_leverage(leverage, symbol)
        logger.info(f"Set leverage for {symbol} to {leverage}x")
        return True
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. Check if symbol requires margin mode (isolated/cross) first.{RESET}")
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
    logger: logging.Logger = None
) -> Optional[Dict]:
    """Places a market order with optional TP/SL using CCXT."""
    if logger is None: logger = logging.getLogger(__name__) # Basic fallback logger

    side = 'buy' if signal == "BUY" else 'sell'
    order_type = 'market'
    params = {}

    # --- Prepare TP/SL parameters (Exchange Specific!) ---
    # Bybit V5 USDT Linear uses 'stopLoss' and 'takeProfit' in params for market orders
    # Important: Price precision MUST match the market's precision
    price_precision = market_info.get('precision', {}).get('price')
    if price_precision is None:
         logger.error(f"Price precision not found for {symbol}, cannot format TP/SL prices.")
         return None

    def format_price(price):
         return exchange.price_to_precision(symbol, price)
         # Alternatively using Decimal:
         # rounding_factor = Decimal('1e-' + str(int(price_precision)))
         # return float(price.quantize(rounding_factor)) # CCXT usually wants float prices

    if sl_price is not None:
        params['stopLoss'] = format_price(sl_price)
        # Bybit might also need trigger price type ('triggerPrice') e.g., 'LastPrice', 'MarkPrice', 'IndexPrice'
        # params['slTriggerBy'] = 'LastPrice' # Default usually works

    if tp_price is not None:
        params['takeProfit'] = format_price(tp_price)
        # params['tpTriggerBy'] = 'LastPrice'

    # Bybit might need position index (posIdx) for hedge mode (0 for one-way, 1 for buy hedge, 2 for sell hedge)
    # Assuming one-way mode:
    # params['positionIdx'] = 0 # Default for one-way

    # Convert size to float as required by ccxt create_order amount
    amount_float = float(size)

    logger.info(f"Attempting to place {signal} {order_type} order for {amount_float} {market_info.get('base','')} of {symbol}...")
    if tp_price: logger.info(f"  TP: {params['takeProfit']}")
    if sl_price: logger.info(f"  SL: {params['stopLoss']}")
    if params: logger.info(f"  Params: {params}")

    try:
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order doesn't need price
            params=params
        )
        logger.info(f"{NEON_GREEN}Trade Placed Successfully! Order ID: {order.get('id')}{RESET}")
        logger.info(f"Order details: {order}") # Log the full order response
        return order
    except ccxt.InsufficientFunds as e:
        logger.error(f"{NEON_RED}Insufficient funds to place {signal} order for {symbol}: {e}{RESET}")
    except ccxt.InvalidOrder as e:
        logger.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
        logger.error(f"  Size: {amount_float}, Params: {params}")
    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error placing order for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Specific Bybit errors can be caught here based on 'e' content
        logger.error(f"{NEON_RED}Exchange error placing order for {symbol}: {e}{RESET}")
        if "leverage not match orderSL" in str(e):
             logger.error(f"{NEON_YELLOW} >> Hint: Leverage setting might conflict with SL/TP or margin mode.{RESET}")
        if "Order cost not available" in str(e):
             logger.error(f"{NEON_YELLOW} >> Hint: Check if balance covers margin requirement including fees.{RESET}")

    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if order failed


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a symbol and places trades based on signals and risk management."""
    logger.info(f"--- Analyzing {symbol} ({config['interval']}) ---")

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         logger.error(f"Invalid interval '{config['interval']}' provided. Cannot map to CCXT timeframe.")
         return

    # Increase limit slightly to ensure enough data for indicators
    kline_limit = 300 # Increase if needed for long lookback indicators
    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=logger)
    if klines_df.empty:
        logger.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol}. Skipping analysis.{RESET}")
        return

    orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], logger)
    if not orderbook_data:
        logger.warning(f"{NEON_YELLOW}Orderbook fetch failed or incomplete for {symbol}, proceeding without it.{RESET}")

    current_price = fetch_current_price_ccxt(exchange, symbol, logger)
    if current_price is None:
         # Fallback to last close if price fetch fails
         current_price = Decimal(str(klines_df['close'].iloc[-1])) if not klines_df.empty else None
         if current_price:
             logger.warning(f"{NEON_YELLOW}Using last close price ({current_price}) as current price fetch failed.{RESET}")
         else:
             logger.error(f"{NEON_RED}Failed to get current price or last close for {symbol}. Skipping analysis.{RESET}")
             return

    # --- 2. Analyze Data ---
    analyzer = TradingAnalyzer(
        klines_df.copy(), logger, config, symbol, config["interval"]
    )
    # Indicators are calculated in __init__

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    entry_calc, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal) # These are potential levels
    confidence = analyzer.calculate_confidence() # Placeholder
    fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)

    # --- 3. Log Analysis Results ---
    output = (
        f"\n{NEON_BLUE}--- Scalping Analysis Results for {symbol} ({config['interval']}) ---{RESET}\n"
        f"Timestamp: {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Current Price: {NEON_GREEN}{current_price:.4f}{RESET}\n"
        f"Generated Signal: {NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW}{signal}{RESET}\n"
        f"Potential Entry: {NEON_YELLOW}{entry_calc:.4f}{RESET}\n"
        f"Potential Take Profit: {(NEON_GREEN + str(round(tp_calc, 4)) + RESET) if tp_calc else 'N/A'}\n"
        f"Potential Stop Loss: {(NEON_RED + str(round(sl_calc, 4)) + RESET) if sl_calc else 'N/A'}\n"
        # f"Confidence: {NEON_CYAN}{confidence:.2f}%{RESET}\n"
        f"ATR ({config.get('atr_period', DEFAULT_ATR_PERIOD)}): {NEON_YELLOW}{analyzer.indicator_values.get('ATR', np.nan):.4f}{RESET}\n"
        # Optional: Log key indicators
        f"RSI ({config.get('rsi_period', DEFAULT_RSI_WINDOW)}): {analyzer.indicator_values.get('RSI', np.nan):.2f} "
        f"StochK: {analyzer.indicator_values.get('StochRSI_K', np.nan):.2f} "
        f"StochD: {analyzer.indicator_values.get('StochRSI_D', np.nan):.2f} "
        f"MFI: {analyzer.indicator_values.get('MFI', np.nan):.2f}\n"
        # f"EMA Short/Long: {analyzer.indicator_values.get('EMA_Short', np.nan):.4f} / {analyzer.indicator_values.get('EMA_Long', np.nan):.4f}\n"
        # f"BB Lower/Upper: {analyzer.indicator_values.get('BB_Lower', np.nan):.4f} / {analyzer.indicator_values.get('BB_Upper', np.nan):.4f}\n"
        # f"Nearest Fibonacci Levels:\n" + "\n".join(
        #     f"  {name}: {NEON_PURPLE}{level:.4f}{RESET}" for name, level in fib_levels[:3] # Show top 3
        # )
    )
    logger.info(output)

    # --- 4. Execute Trade (If Enabled and Signal is Buy/Sell) ---
    if not config.get("enable_trading", False):
        logger.info(f"{NEON_YELLOW}Trading is disabled in config. No trade placed.{RESET}")
        return

    if signal in ["BUY", "SELL"]:
        logger.info(f"*** {signal} Signal Triggered - Proceeding to Trade Execution ***")

        # --- 4a. Pre-Trade Checks ---
        market_info = get_market_info(exchange, symbol, logger)
        if not market_info:
             logger.error(f"{NEON_RED}Cannot proceed with trade: Failed to get market info for {symbol}.{RESET}")
             return

        # Check for existing position
        open_position = get_open_position(exchange, symbol, logger)
        if open_position:
            # Simple logic: Don't open a new position if one already exists for this symbol
            # More complex logic could involve averaging down, closing, etc.
            pos_size = open_position.get('contracts', open_position.get('info',{}).get('size', 0))
            pos_side = open_position.get('side', open_position.get('info',{}).get('side', 'unknown'))
            logger.warning(f"{NEON_YELLOW}Existing {pos_side} position found for {symbol} (Size: {pos_size}). Skipping new trade based on max_concurrent_positions.{RESET}")
            return # Adhere to max_concurrent_positions = 1

        # Fetch available balance
        balance = fetch_balance(exchange, QUOTE_CURRENCY, logger)
        if balance is None or balance <= 0:
            logger.error(f"{NEON_RED}Cannot proceed with trade: Failed to fetch sufficient balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
            return

        # Ensure calculated TP/SL are valid
        if tp_calc is None or sl_calc is None:
             logger.error(f"{NEON_RED}Cannot proceed with trade: Invalid TP/SL calculated. TP={tp_calc}, SL={sl_calc}{RESET}")
             return

        # --- 4b. Set Leverage ---
        # Set leverage before calculating size if margin depends on it (usually does)
        leverage = int(config.get("leverage", 1))
        if market_info.get('contract') and leverage > 1: # Only set for derivatives and if > 1
            if not set_leverage_ccxt(exchange, symbol, leverage, logger):
                 logger.warning(f"{NEON_YELLOW}Failed to set leverage {leverage}x for {symbol}. Proceeding with caution (exchange default or previous setting might be used).{RESET}")
                 # Decide whether to abort if leverage fails? For now, continue.
        else:
             logger.info(f"Leverage setting skipped (Spot market or leverage <= 1).")


        # --- 4c. Calculate Position Size ---
        position_size = calculate_position_size(
            balance=balance,
            risk_per_trade=config["risk_per_trade"],
            stop_loss_price=sl_calc,
            entry_price=current_price, # Use current price as assumed entry for market order
            market_info=market_info,
            leverage=leverage, # Leverage is now mostly for margin calculation, risk calc handles stop-loss distance
            logger=logger
        )

        if position_size is None or position_size <= 0:
            logger.error(f"{NEON_RED}Trade aborted: Invalid position size calculated ({position_size}).{RESET}")
            return

        # --- 4d. Place Order ---
        logger.info(f"Attempting to place {signal} trade for {symbol} | Size: {position_size} | Entry: ~{current_price:.4f} | TP: {tp_calc:.4f} | SL: {sl_calc:.4f}")
        trade_result = place_trade(
            exchange=exchange,
            symbol=symbol,
            signal=signal,
            size=position_size,
            market_info=market_info,
            tp_price=tp_calc,
            sl_price=sl_calc,
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
    global CONFIG # Allow modifying config based on input if needed

    # --- Initial Setup ---
    logger = setup_logger("global") # Setup global logger first
    logger.info("Starting Livewire Trading Bot...")
    logger.info(f"Loaded configuration from {CONFIG_FILE}")
    if CONFIG.get("enable_trading"):
         logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              logger.warning(f"{NEON_YELLOW}Using SANDBOX environment.{RESET}")
         else:
              logger.warning(f"{NEON_RED}Using REAL MONEY environment. Be careful!{RESET}")
         input("Press Enter to acknowledge live trading and continue, or Ctrl+C to abort...")
    else:
         logger.info("Trading is disabled. Running in analysis-only mode.")


    exchange = initialize_exchange(logger)
    if not exchange:
        logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return

    # --- Symbol and Interval Input ---
    # Ensure symbol format matches CCXT (e.g., BTC/USDT)
    symbol_input = input(f"{NEON_YELLOW}Enter symbol (e.g., BTC/USDT, ETH/USDT): {RESET}").upper()
    # Basic validation/correction for common formats
    if '/' not in symbol_input and QUOTE_CURRENCY in symbol_input:
         base = symbol_input.replace(QUOTE_CURRENCY,"")
         symbol_ccxt = f"{base}/{QUOTE_CURRENCY}"
         logger.info(f"Assuming symbol format: {symbol_ccxt}")
    else:
         symbol_ccxt = symbol_input

    # Validate symbol exists on exchange
    market_info = get_market_info(exchange, symbol_ccxt, logger)
    if not market_info:
        logger.critical(f"{NEON_RED}Symbol {symbol_ccxt} not found or invalid on {exchange.id}. Exiting.{RESET}")
        return
    else:
         logger.info(f"Symbol {symbol_ccxt} validated. Market Type: {'Contract' if market_info.get('contract') else 'Spot'}")


    interval_input = input(f"{NEON_YELLOW}Enter interval [{'/'.join(VALID_INTERVALS)}]: {RESET}") # Show available keys
    if interval_input not in VALID_INTERVALS:
         logger.critical(f"{NEON_RED}Invalid interval: {interval_input}. Please choose from {VALID_INTERVALS}. Exiting.{RESET}")
         return
    if interval_input not in CCXT_INTERVAL_MAP:
         logger.critical(f"{NEON_RED}Interval '{interval_input}' not mapped to CCXT format. Exiting.{RESET}")
         return

    # Update config with user input for this run
    CONFIG["interval"] = interval_input
    logger.info(f"Using Symbol: {symbol_ccxt}, Interval: {interval_input} ({CCXT_INTERVAL_MAP[interval_input]})")


    # --- Main Loop ---
    try:
        while True:
            try:
                analyze_and_trade_symbol(exchange, symbol_ccxt, CONFIG, logger)
            except Exception as loop_error:
                 # Catch errors within the loop to prevent the bot from crashing
                 logger.error(f"{NEON_RED}An error occurred in the main loop: {loop_error}{RESET}", exc_info=True)
                 logger.info("Attempting to continue after error...")

            logger.info(f"Waiting {LOOP_DELAY_SECONDS} seconds until next cycle...")
            time.sleep(LOOP_DELAY_SECONDS)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
         logger.critical(f"{NEON_RED}A critical error occurred: {e}{RESET}", exc_info=True)
    finally:
        # Clean up resources if needed (e.g., close websocket connections if added later)
        logger.info("Livewire Trading Bot stopped.")


if __name__ == "__main__":
    main()
```

**Explanation of Changes and Key Concepts:**

1.  **CCXT Integration:**
    *   Added `import ccxt`.
    *   `initialize_exchange()`: Creates the `ccxt.bybit` exchange object using API keys from `.env`. It handles sandbox mode (`use_sandbox` in config) and loads market data (`exchange.load_markets()`) which is crucial for getting trading rules (precision, limits).
    *   Replaced API calls for price, klines, and orderbook with their `ccxt` equivalents (`fetch_current_price_ccxt`, `fetch_klines_ccxt`, `fetch_orderbook_ccxt`). This standardizes interaction.
    *   Kept the original `bybit_request` function as it might be useful for specific V5 endpoints not covered easily by `ccxt`'s standard methods, but it's not used in the current trading flow.
    *   Added `CCXT_INTERVAL_MAP` to translate the script's interval format ("1", "5") to `ccxt`'s format ("1m", "5m").

2.  **Configuration (`config.json`):**
    *   Added new keys:
        *   `enable_trading`: **Crucial Safety Feature**. Set to `true` only when you intend to place real orders. **Defaults to `false`**.
        *   `use_sandbox`: Set to `true` to use Bybit's testnet (requires separate sandbox API keys). **Defaults to `true`**.
        *   `risk_per_trade`: The fraction of your available balance to risk on a single trade (e.g., 0.01 = 1%).
        *   `leverage`: The desired leverage for derivative trades (ignored for spot). **Set carefully based on risk tolerance and exchange limits.**
        *   `max_concurrent_positions`: Limits how many trades the bot can have open simultaneously *for the specific symbol being run*. Currently set to 1 (no new trade if one exists).
        *   `quote_currency`: The currency used for balance checking and calculations (e.g., "USDT").
    *   `load_config()` now ensures these new keys exist and adds defaults if missing.

3.  **Risk Management Implementation:**
    *   **Stop-Loss (SL) and Take-Profit (TP):**
        *   `calculate_entry_tp_sl` still calculates the *levels* based on ATR multiples from the config (`stop_loss_multiple`, `take_profit_multiple`).
        *   These levels are now passed to the `place_trade` function.
    *   **Position Sizing:**
        *   `fetch_balance()`: Gets the available balance for the specified `quote_currency` using `exchange.fetch_balance()`.
        *   `get_market_info()`: Retrieves crucial details about the trading pair (min/max order size, amount/price precision) using `exchange.market()`.
        *   `calculate_position_size()`: This is the core risk calculation.
            *   Calculates the amount to risk (`balance * risk_per_trade`).
            *   Determines the distance between entry and stop-loss.
            *   Calculates the initial size based on `risk_amount / sl_distance`.
            *   Adjusts the size based on the market's minimum order size (`limits['amount']['min']`) and rounds it *down* to the required precision (`precision['amount']`) to avoid errors.
            *   Returns `None` if the calculated size is invalid or below the minimum.
    *   **Leverage:**
        *   `set_leverage_ccxt()`: Attempts to set the leverage for the symbol using `exchange.set_leverage()`. This is typically needed *before* placing an order on margin/futures accounts. Called within `analyze_and_trade_symbol`.
    *   **Position Check:**
        *   `get_open_position()`: Uses `exchange.fetch_positions()` to check if a position is already open for the symbol. This prevents opening multiple trades if `max_concurrent_positions` is 1.

4.  **Trade Execution:**
    *   `place_trade()`:
        *   Takes the signal ("BUY" or "SELL"), calculated size, market info, and TP/SL levels.
        *   Determines `side` ('buy' or 'sell') and `order_type` ('market').
        *   Formats TP/SL prices according to the market's required price precision using `exchange.price_to_precision()`.
        *   Uses `exchange.create_order()` to place the trade.
        *   Crucially, it passes the TP/SL prices in the `params` dictionary. **The exact `params` keys (`stopLoss`, `takeProfit`) are specific to the exchange (Bybit V5 linear in this case) and order type.** You might need to adjust these based on Bybit's documentation or `ccxt`'s unified API specifics if trading different contract types (inverse, options) or using limit orders.
        *   Includes robust error handling for common `ccxt` exceptions (`InsufficientFunds`, `InvalidOrder`, etc.).

5.  **Main Loop (`analyze_and_trade_symbol`, `main`):**
    *   The `main` function now initializes `ccxt`, gets user input for symbol/interval, validates them, and then calls `analyze_and_trade_symbol` in a loop.
    *   `analyze_and_trade_symbol` orchestrates the process: fetch data -> analyze -> log results -> check trading enabled -> perform pre-trade checks (position, balance, market info) -> set leverage -> calculate size -> place trade.
    *   Added a check for `config['enable_trading']` before attempting any trade execution steps.
    *   Added user input validation for symbol and interval.
    *   Symbol input now tries to standardize to `BASE/QUOTE` format.
    *   Added a critical warning and confirmation step if live trading is enabled.

6.  **Logging and Precision:**
    *   Enhanced logging around trading actions, balance fetching, position sizing, and errors.
    *   Increased `Decimal` precision (`getcontext().prec = 18`) for potentially more accurate calculations involving prices and sizes.
    *   Used `Decimal.quantize` for precise rounding according to market rules.

**To Run This Code:**

1.  **Install Libraries:**
    ```bash
    pip install ccxt python-dotenv requests colorama numpy pandas pytz # Use pytz if zoneinfo not available (< py 3.9)
    # or pip install ccxt python-dotenv requests colorama numpy pandas zoneinfo
    ```
2.  **Create `.env` file:** In the same directory as the script, create a file named `.env` with your Bybit API keys:
    ```dotenv
    BYBIT_API_KEY=YOUR_API_KEY_HERE
    BYBIT_API_SECRET=YOUR_API_SECRET_HERE
    # Optional: Specify base URL if needed (e.g., for specific region)
    # BYBIT_BASE_URL=https://api.bytick.com
    ```
    *   **IMPORTANT:** If you set `use_sandbox: true` in `config.json`, you need **Sandbox API keys** in the `.env` file. Get these from the Bybit testnet website.
3.  **Create/Review `config.json`:** The script will create a default `config.json` if it doesn't exist. Review the settings, especially:
    *   `enable_trading`: **Keep `false` initially for testing.**
    *   `use_sandbox`: **Keep `true` initially for testing.**
    *   `risk_per_trade`: Adjust risk percentage.
    *   `leverage`: Set desired leverage.
    *   `quote_currency`: Ensure it matches your account (usually `USDT`).
    *   Indicator settings and weights.
4.  **Run the script:**
    ```bash
    python livewire.py
    ```
5.  **Enter Symbol and Interval:** Provide the trading pair (e.g., `BTC/USDT`) and interval (e.g., `5`).
6.  **Monitor Output:** Observe the logs for analysis results, signal generation, and (if enabled) trade execution attempts and outcomes.

Remember to start with `enable_trading: false` and `use_sandbox: true` to thoroughly test the logic before risking real capital.
