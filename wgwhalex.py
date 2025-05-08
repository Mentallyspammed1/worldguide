# wgwhalex.py

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, getcontext, ROUND_DOWN
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize colorama and set decimal precision
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
os.makedirs(LOG_DIRECTORY, exist_ok=True) # Ensure log directory exists

TIMEZONE = ZoneInfo("America/Chicago") # User-configurable timezone if needed
MAX_API_RETRIES = 5 # Increased retries
RETRY_DELAY_SECONDS = 7 # Slightly longer delay
REQUEST_TIMEOUT = 15 # Increased timeout

# Technical Analysis Defaults (configurable via config.json)
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 14 # Often same as Stoch RSI period
DEFAULT_K_WINDOW = 3
DEFAULT_D_WINDOW = 3
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0 # Ensure float
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_SMA_50_WINDOW = 50 # Added longer SMA for trend context
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_PSAR_ACCELERATION = 0.02
DEFAULT_PSAR_MAX_ACCELERATION = 0.2
DEFAULT_MOMENTUM_PERIOD = 10
DEFAULT_VOLUME_MA_PERIOD = 20
DEFAULT_FIBONACCI_WINDOW = 60 # Increased default lookback

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
LOOP_DELAY_SECONDS = 15 # Slightly longer loop delay to avoid rate limits

# --- Configuration Management ---

def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found."""
    default_config = {
        "symbol": "BTCUSDT", # Default symbol if not provided at runtime
        "interval": "5m", # Default interval if not provided at runtime
        "analysis_interval": 5, # Not currently used, potential future feature
        "retry_delay": RETRY_DELAY_SECONDS,
        "orderbook_limit": 50,
        "signal_score_threshold": 1.8, # Slightly higher default threshold
        "stop_loss_atr_multiple": 1.5, # Renamed for clarity
        "take_profit_atr_multiple": 1.0, # Renamed for clarity
        "volume_confirmation_multiplier": 1.5, # Adjusted multiplier
        "scalping_signal_threshold": 2.5, # Adjusted threshold
        "loop_delay": LOOP_DELAY_SECONDS,
        # Indicator Periods
        "momentum_period": DEFAULT_MOMENTUM_PERIOD,
        "volume_ma_period": DEFAULT_VOLUME_MA_PERIOD,
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "stoch_rsi_period": DEFAULT_STOCH_RSI_WINDOW,
        "stoch_k_period": DEFAULT_K_WINDOW,
        "stoch_d_period": DEFAULT_D_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "cci_period": DEFAULT_CCI_WINDOW,
        "williams_r_period": DEFAULT_WILLIAMS_R_WINDOW,
        "mfi_period": DEFAULT_MFI_WINDOW,
        "psar_acceleration": DEFAULT_PSAR_ACCELERATION,
        "psar_max_acceleration": DEFAULT_PSAR_MAX_ACCELERATION,
        "sma10_period": DEFAULT_SMA_10_WINDOW,
        "sma50_period": DEFAULT_SMA_50_WINDOW, # Added SMA50
        "fibonacci_window": DEFAULT_FIBONACCI_WINDOW,
        # Indicator Thresholds
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stoch_rsi_oversold": 20,
        "stoch_rsi_overbought": 80,
        "cci_oversold": -100,
        "cci_overbought": 100,
        "williams_r_oversold": -80,
        "williams_r_overbought": -20,
        "mfi_oversold": 20,
        "mfi_overbought": 80,
        # Active Indicators (Toggle analysis components)
        "indicators": {
            "ema_alignment": True,
            "sma_trend_filter": True, # Added longer SMA filter
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True, # Williams %R
            "psar": True,
            "sma_10": True,
            "mfi": True,
            "orderbook_imbalance": True,
            "fibonacci_levels": True, # Track Fibonacci levels
        },
        # Weight Sets (Define influence of each indicator)
        "weight_sets": {
            "default_scalping": {
                "ema_alignment": 0.25,
                "sma_trend_filter": 0.3, # Give trend filter reasonable weight
                "momentum": 0.2,
                "volume_confirmation": 0.15,
                "stoch_rsi": 0.4, # StochRSI often key for scalping reversals
                "rsi": 0.2,
                "bollinger_bands": 0.3, # BB touches/breaks are important
                "vwap": 0.2, # VWAP cross/bounce
                "cci": 0.15,
                "wr": 0.15,
                "psar": 0.2, # PSAR flips
                "sma_10": 0.1, # Short-term MA cross
                "mfi": 0.2, # Money flow divergence/extremes
                "orderbook_imbalance": 0.1, # Small weight for basic OB check
                # Fibonacci weight is implicitly handled by TP/SL adjustment or context
            }
        },
    }

    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print(f"{NEON_YELLOW}Configuration file not found. Created default config at {filepath}{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}Error creating default config file: {e}{RESET}")
            return default_config # Return default if creation fails

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
            # Ensure all keys from default_config are present, adding missing ones
            _ensure_config_keys(config, default_config)
            # Save back the potentially updated config
            with open(filepath, "w", encoding="utf-8") as f_write:
                 json.dump(config, f_write, indent=4)
            return config
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"{NEON_RED}Error loading or parsing config file {filepath}: {e}. Using default config.{RESET}")
        # Attempt to save default config if loading failed badly
        try:
            with open(filepath, "w", encoding="utf-8") as f_default:
                json.dump(default_config, f_default, indent=4)
        except IOError as e_save:
             print(f"{NEON_RED}Could not save default config either: {e_save}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> None:
    """Recursively ensure all keys from default_config are in config."""
    for key, default_value in default_config.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(config.get(key), dict):
            # Check nested dictionaries recursively
             _ensure_config_keys(config[key], default_value)
        # Optional: Add type checking/conversion here if needed
        # Example: if isinstance(default_value, int) and not isinstance(config[key], int):
        #     try: config[key] = int(config[key])
        #     except (ValueError, TypeError): config[key] = default_value

# --- Logging Setup ---

class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information (API keys) from logs."""
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if API_KEY:
            msg = msg.replace(API_KEY, f"{API_KEY[:4]}***{API_KEY[-4:]}") # Show first/last 4 chars
        if API_SECRET:
             msg = msg.replace(API_SECRET, "***") # Don't show any part of secret
        return msg

def setup_logger(symbol: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger for the given symbol with file and stream handlers."""
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"wgwhalex_{symbol}_{timestamp}.log")

    logger = logging.getLogger(f"wgwhalex_{symbol}")
    logger.setLevel(log_level)
    # Prevent adding multiple handlers if function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (Console)
    stream_handler = logging.StreamHandler()
    # Use simpler format for console, include color
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)s{RESET} - %(message)s",
        datefmt=date_format
    )
    stream_handler.setFormatter(stream_formatter)
    # Ensure console logs also respect the main log level
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # Set timezone for logging timestamps
    logging.Formatter.converter = lambda *args: datetime.now(TIMEZONE).timetuple()

    return logger

# --- API Interaction ---

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.8, # Slightly more aggressive backoff
        status_forcelist=[429, 500, 502, 503, 504], # Standard retry codes
        allowed_methods=["GET", "POST"], # Retry POST requests too
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def bybit_request(
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    is_signed: bool = True,
    logger: Optional[logging.Logger] = None,
    session: Optional[requests.Session] = None
) -> Optional[Dict[str, Any]]:
    """Send a request to Bybit API with optional signing and retry logic."""
    effective_session = session if session else create_session()
    params = params or {}
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    payload_str = ""

    if is_signed:
        if not API_KEY or not API_SECRET:
             if logger: logger.error(f"{NEON_RED}API Key or Secret not configured for signed request.{RESET}")
             return None

        timestamp = str(int(time.time() * 1000))
        recv_window = "20000" # Bybit recommended recv_window

        # Prepare signature string based on method
        if method == "GET":
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            payload_str = f"{timestamp}{API_KEY}{recv_window}{query_string}"
        elif method == "POST":
            # Important: Bybit V5 POST uses JSON body for signature
            body_str = json.dumps(params, separators=(',', ':')) if params else ""
            payload_str = f"{timestamp}{API_KEY}{recv_window}{body_str}"

        signature = hmac.new(
            API_SECRET.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers.update({
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        })

    request_kwargs = {
        "method": method,
        "url": url,
        "headers": headers,
        "timeout": REQUEST_TIMEOUT,
    }

    if method == "GET":
        request_kwargs["params"] = params
    elif method == "POST":
        # POST requests send data as JSON body
        request_kwargs["json"] = params # Use json= directly

    if logger:
        log_params = {k: (f"{v[:4]}...{v[-4:]}" if k == "api_key" else "***" if k == "secret" else v) for k, v in params.items()}
        # logger.debug(f"Requesting {method} {endpoint} with params: {log_params}")
        logger.debug(f"Requesting {method} {endpoint}") # Avoid logging params directly


    try:
        response = effective_session.request(**request_kwargs)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        json_response = response.json()

        # Check Bybit's specific return code
        if json_response.get("retCode") == 0:
            if logger: logger.debug(f"API call successful: {method} {endpoint}")
            return json_response
        else:
            if logger:
                err_code = json_response.get("retCode")
                err_msg = json_response.get("retMsg", "Unknown Bybit error")
                logger.error(f"{NEON_RED}API Error {err_code}: {err_msg} | Endpoint: {endpoint} | Params: {params if method == 'GET' else 'See Body'}{RESET}")
            return None # Indicate failure

    except requests.exceptions.Timeout as e:
         if logger: logger.error(f"{NEON_RED}Request timed out: {e}{RESET}")
         return None
    except requests.exceptions.ConnectionError as e:
         if logger: logger.error(f"{NEON_RED}Connection error: {e}{RESET}")
         return None
    except requests.exceptions.HTTPError as e:
        # Log HTTP errors (like 401, 403, 404, 5xx already caught by raise_for_status)
        if logger:
            # Try to get more details from response if available
            error_details = ""
            try:
                error_content = response.json()
                error_details = f" | Response: {error_content}"
            except json.JSONDecodeError:
                 error_details = f" | Response: {response.text}" # Fallback to text
            logger.error(f"{NEON_RED}HTTP Error: {e}{error_details}{RESET}")
        return None
    except requests.exceptions.RequestException as e:
        if logger: logger.error(f"{NEON_RED}An unexpected error occurred during API request: {e}{RESET}")
        return None
    except json.JSONDecodeError as e:
         if logger: logger.error(f"{NEON_RED}Failed to decode JSON response: {e} | Response Text: {response.text[:200]}...{RESET}")
         return None

def fetch_current_price(symbol: str, logger: logging.Logger, session: requests.Session) -> Optional[Decimal]:
    """Fetch the current mark or last price of a trading symbol."""
    endpoint = "/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    # Use mark price preferentially for derivatives, fallback to last price
    response = bybit_request("GET", endpoint, params, is_signed=False, logger=logger, session=session)

    if response and response.get("result") and response["result"].get("list"):
        tickers = response["result"]["list"]
        if tickers:
            # Find the specific symbol in the list (should usually be the only one if symbol specified)
            ticker_data = next((item for item in tickers if item.get("symbol") == symbol), None)
            if ticker_data:
                price_str = ticker_data.get("markPrice") or ticker_data.get("lastPrice")
                if price_str:
                    try:
                        return Decimal(price_str)
                    except Exception as e:
                         logger.error(f"{NEON_RED}Error parsing price '{price_str}' for {symbol}: {e}{RESET}")
                         return None
                else:
                    logger.warning(f"{NEON_YELLOW}Mark price and Last price missing for {symbol} in ticker response.{RESET}")
            else:
                 logger.warning(f"{NEON_YELLOW}Symbol {symbol} not found in tickers response list.{RESET}")
        else:
            logger.warning(f"{NEON_YELLOW}Empty list in tickers response for {symbol}.{RESET}")
    else:
        # Error already logged by bybit_request
        logger.error(f"{NEON_RED}Failed to fetch ticker data for {symbol} (or unexpected response structure).{RESET}")

    return None


def fetch_klines(
    symbol: str, interval: str, limit: int = 200, logger: Optional[logging.Logger] = None, session: Optional[requests.Session] = None
) -> pd.DataFrame:
    """Fetch kline/OHLCV data for a symbol and interval."""
    endpoint = "/v5/market/kline"
    # Ensure limit is within Bybit's allowed range (e.g., max 1000 or 200)
    limit = min(limit, 1000) # Adjust based on Bybit V5 documentation if needed
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear", # Assuming linear perpetual futures
    }
    response = bybit_request("GET", endpoint, params, is_signed=False, logger=logger, session=session)

    if response and response.get("result") and response["result"].get("list"):
        data = response["result"]["list"]
        if not data:
            if logger: logger.warning(f"{NEON_YELLOW}No kline data returned for {symbol} interval {interval}.{RESET}")
            return pd.DataFrame()

        # Bybit V5 kline format: [startTime, open, high, low, close, volume, turnover]
        columns = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(data, columns=columns)

        # Data type conversions
        df['start_time'] = pd.to_datetime(df['start_time'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, coercing errors to NaN

        # Reverse dataframe so newest data is last (standard for TA libraries)
        df = df.iloc[::-1].reset_index(drop=True)
        if logger: logger.debug(f"Fetched {len(df)} klines for {symbol} ({interval})")
        return df
    else:
        if logger: logger.error(f"{NEON_RED}Failed to fetch or parse klines for {symbol}. Response: {response}{RESET}")
        return pd.DataFrame()


def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger, session: requests.Session) -> Optional[Dict]:
    """Fetch orderbook data using Bybit API V5."""
    endpoint = "/v5/market/orderbook"
    params = {"category": "linear", "symbol": symbol, "limit": min(limit, 50)} # V5 limit is 1-50 for L1, up to 200/500 for L2? Check docs. Let's use 50.
    response = bybit_request("GET", endpoint, params, is_signed=False, logger=logger, session=session)

    if response and response.get("result"):
        result = response["result"]
        # V5 format: result.a = asks [[price, size], ...], result.b = bids [[price, size], ...]
        orderbook = {
            "symbol": result.get("s"),
            "bids": [[Decimal(p), Decimal(s)] for p, s in result.get("b", [])],
            "asks": [[Decimal(p), Decimal(s)] for p, s in result.get("a", [])],
            "timestamp": int(result.get("ts", 0)), # Timestamp of update
            "update_id": int(result.get("u", 0)), # Update ID
        }
        if logger: logger.debug(f"Fetched orderbook for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks.")
        return orderbook
    else:
        if logger: logger.error(f"{NEON_RED}Failed to fetch or parse orderbook for {symbol}. Response: {response}{RESET}")
        return None

# --- Trading Analysis ---

class TradingAnalyzer:
    """
    Analyzes trading data (OHLCV) and generates trading signals based on
    a configurable set of technical indicators and strategy rules.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        logger: logging.Logger,
        symbol: str,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            df: Pandas DataFrame with OHLCV data (must include 'high', 'low', 'close', 'volume').
            config: Dictionary containing configuration parameters.
            logger: Logger instance for logging messages.
            symbol: The trading symbol being analyzed.
        """
        if df.empty or not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
             raise ValueError("DataFrame must not be empty and contain 'high', 'low', 'close', 'volume' columns.")

        self.df = df.copy() # Work on a copy to avoid modifying original
        self.config = config
        self.logger = logger
        self.symbol = symbol
        self.indicator_values: Dict[str, Union[float, str]] = {} # Store latest indicator values
        self.fib_levels: Dict[str, Decimal] = {} # Store calculated Fibonacci levels
        self.weights = config.get("weight_sets", {}).get("default_scalping", {}) # Use default scalping weights

        # Pre-calculate indicators if enabled and data is sufficient
        self._calculate_all_indicators()
        if self.config["indicators"].get("fibonacci_levels", False):
             self.calculate_fibonacci_levels()


    def _safe_calculate(self, func: callable, name: str, *args, **kwargs) -> Optional[pd.Series]:
        """Safely executes an indicator calculation function, logging errors."""
        min_required_periods = kwargs.get('window', kwargs.get('period', 1)) + 5 # Estimate minimum data needed
        if len(self.df) < min_required_periods:
             self.logger.warning(f"{NEON_YELLOW}Skipping {name}: Insufficient data ({len(self.df)} < {min_required_periods}){RESET}")
             self.indicator_values[name] = "Insufficient Data"
             return None
        try:
             result = func(*args, **kwargs)
             if result is not None and not result.empty:
                 # Store the *last* calculated value
                 last_val = result.iloc[-1]
                 self.indicator_values[name] = float(last_val) if pd.notna(last_val) else np.nan
                 return result
             else:
                  self.indicator_values[name] = np.nan # Indicate calculation didn't produce a result
                  return None
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Calculation Error ({name}): Missing column {e}{RESET}")
            self.indicator_values[name] = "Error"
            return None
        except ZeroDivisionError as e:
            self.logger.warning(f"{NEON_YELLOW}Calculation Warning ({name}): Division by zero encountered.{RESET}")
            self.indicator_values[name] = np.nan # Often happens in initial periods
            # Attempt to return the series even with NaNs
            try: return func(*args, **kwargs)
            except: return None
        except Exception as e:
            self.logger.error(f"{NEON_RED}Calculation Error ({name}): {type(e).__name__} - {e}{RESET}", exc_info=False) # Set exc_info=True for full traceback if needed
            self.indicator_values[name] = "Error"
            return None

    def _calculate_all_indicators(self) -> None:
        """Calculate all enabled technical indicators."""
        self.logger.debug("Calculating technical indicators...")
        cfg = self.config

        # --- Moving Averages & Trend ---
        if cfg["indicators"].get("sma_10", False):
            self.df['sma10'] = self._safe_calculate(self.calculate_sma, "SMA10", window=cfg["sma10_period"])
        if cfg["indicators"].get("sma_50", False) or cfg["indicators"].get("sma_trend_filter", False): # Calculate if used directly or for filter
             self.df['sma50'] = self._safe_calculate(self.calculate_sma, "SMA50", window=cfg["sma50_period"])
        if cfg["indicators"].get("ema_short", False) or cfg["indicators"].get("ema_alignment", False):
             self.df['ema_short'] = self._safe_calculate(self.calculate_ema, f"EMA{cfg['ema_short_period']}", window=cfg["ema_short_period"])
        if cfg["indicators"].get("ema_long", False) or cfg["indicators"].get("ema_alignment", False):
             self.df['ema_long'] = self._safe_calculate(self.calculate_ema, f"EMA{cfg['ema_long_period']}", window=cfg["ema_long_period"])
        if cfg["indicators"].get("vwap", False):
             self.df['vwap'] = self._safe_calculate(self.calculate_vwap, "VWAP")

        # --- Oscillators ---
        if cfg["indicators"].get("rsi", False):
             self.df['rsi'] = self._safe_calculate(self.calculate_rsi, "RSI", window=cfg["rsi_period"])
        if cfg["indicators"].get("stoch_rsi", False):
            stoch_rsi_df = self._safe_calculate(self.calculate_stoch_rsi, "StochRSI", rsi_window=cfg["rsi_period"], stoch_window=cfg["stoch_rsi_period"], k_window=cfg["stoch_k_period"], d_window=cfg["stoch_d_period"])
            if stoch_rsi_df is not None:
                self.df['stoch_rsi_k'] = stoch_rsi_df['k']
                self.df['stoch_rsi_d'] = stoch_rsi_df['d']
                # Store K and D separately
                self.indicator_values["StochRSI_K"] = float(stoch_rsi_df['k'].iloc[-1]) if pd.notna(stoch_rsi_df['k'].iloc[-1]) else np.nan
                self.indicator_values["StochRSI_D"] = float(stoch_rsi_df['d'].iloc[-1]) if pd.notna(stoch_rsi_df['d'].iloc[-1]) else np.nan
                # Remove the composite name if K/D are stored
                if "StochRSI" in self.indicator_values: del self.indicator_values["StochRSI"]
        if cfg["indicators"].get("cci", False):
             self.df['cci'] = self._safe_calculate(self.calculate_cci, "CCI", window=cfg["cci_period"])
        if cfg["indicators"].get("wr", False):
             self.df['wr'] = self._safe_calculate(self.calculate_williams_r, "Williams_R", window=cfg["williams_r_period"])
        if cfg["indicators"].get("mfi", False):
             self.df['mfi'] = self._safe_calculate(self.calculate_mfi, "MFI", window=cfg["mfi_period"])

        # --- Volatility & Momentum ---
        if cfg["indicators"].get("atr", True): # ATR is often needed for SL/TP
            self.df['atr'] = self._safe_calculate(self.calculate_atr, "ATR", period=cfg["atr_period"])
        if cfg["indicators"].get("bollinger_bands", False):
            bb_df = self._safe_calculate(self.calculate_bollinger_bands, "BollingerBands", period=cfg["bollinger_bands_period"], std_dev=cfg["bollinger_bands_std_dev"])
            if bb_df is not None:
                 self.df['bb_upper'] = bb_df['bb_upper']
                 self.df['bb_mid'] = bb_df['bb_mid']
                 self.df['bb_lower'] = bb_df['bb_lower']
                 # Store individual band values
                 self.indicator_values["BB_Upper"] = float(bb_df['bb_upper'].iloc[-1]) if pd.notna(bb_df['bb_upper'].iloc[-1]) else np.nan
                 self.indicator_values["BB_Mid"] = float(bb_df['bb_mid'].iloc[-1]) if pd.notna(bb_df['bb_mid'].iloc[-1]) else np.nan
                 self.indicator_values["BB_Lower"] = float(bb_df['bb_lower'].iloc[-1]) if pd.notna(bb_df['bb_lower'].iloc[-1]) else np.nan
                 if "BollingerBands" in self.indicator_values: del self.indicator_values["BollingerBands"] # Remove composite name
        if cfg["indicators"].get("momentum", False):
             self.df['momentum'] = self._safe_calculate(self.calculate_momentum, "Momentum", period=cfg["momentum_period"])
        if cfg["indicators"].get("psar", False):
             self.df['psar'] = self._safe_calculate(self.calculate_psar, "PSAR", acceleration=cfg["psar_acceleration"], max_acceleration=cfg["psar_max_acceleration"])

        # --- Volume ---
        if cfg["indicators"].get("volume_confirmation", False):
             self.df['volume_ma'] = self._safe_calculate(self.calculate_sma, "VolumeMA", series=self.df['volume'], window=cfg["volume_ma_period"])


    # --- Individual Indicator Calculations ---

    def calculate_atr(self, period: int = DEFAULT_ATR_PERIOD) -> Optional[pd.Series]:
        """Calculate Average True Range (ATR)."""
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
        atr = tr.ewm(alpha=1/period, adjust=False).mean() # Use Exponential Moving Average for ATR (common method)
        # atr = tr.rolling(window=period).mean() # Alternative: Simple Moving Average
        return atr

    def calculate_sma(self, window: int, series: Optional[pd.Series] = None) -> Optional[pd.Series]:
        """Calculate Simple Moving Average (SMA)."""
        target_series = series if series is not None else self.df['close']
        return target_series.rolling(window=window).mean()

    def calculate_ema(self, window: int, series: Optional[pd.Series] = None) -> Optional[pd.Series]:
        """Calculate Exponential Moving Average (EMA)."""
        target_series = series if series is not None else self.df['close']
        return target_series.ewm(span=window, adjust=False).mean()

    def calculate_momentum(self, period: int = DEFAULT_MOMENTUM_PERIOD) -> Optional[pd.Series]:
        """Calculate Momentum."""
        # Avoid division by zero or issues with initial values
        shifted_close = self.df['close'].shift(period)
        # Use numpy where for safe division
        momentum = np.where(
            shifted_close.notna() & (shifted_close != 0),
            ((self.df['close'] - shifted_close) / shifted_close) * 100,
            np.nan # Assign NaN where calculation is not possible
        )
        return pd.Series(momentum, index=self.df.index)


    def calculate_cci(self, window: int = DEFAULT_CCI_WINDOW) -> Optional[pd.Series]:
        """Calculate Commodity Channel Index (CCI)."""
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma_tp = tp.rolling(window=window).mean()
        # Calculate Mean Absolute Deviation (MAD) correctly
        mad_tp = tp.rolling(window=window).apply(lambda x: pd.Series(x).mad(), raw=True)
        # Handle potential division by zero in MAD
        cci = np.where(
             mad_tp.notna() & (mad_tp != 0),
             (tp - sma_tp) / (0.015 * mad_tp),
             np.nan # Assign NaN if MAD is zero or NaN
         )
        return pd.Series(cci, index=self.df.index)


    def calculate_williams_r(self, window: int = DEFAULT_WILLIAMS_R_WINDOW) -> Optional[pd.Series]:
        """Calculate Williams %R."""
        highest_high = self.df['high'].rolling(window=window).max()
        lowest_low = self.df['low'].rolling(window=window).min()
        range_high_low = highest_high - lowest_low
        # Avoid division by zero
        wr = np.where(
             range_high_low != 0,
             ((highest_high - self.df['close']) / range_high_low) * -100,
             -50.0 # Or np.nan or 0, depending on desired behavior when range is zero
         )
        return pd.Series(wr, index=self.df.index)

    def calculate_mfi(self, window: int = DEFAULT_MFI_WINDOW) -> Optional[pd.Series]:
        """Calculate Money Flow Index (MFI)."""
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_money_flow = typical_price * self.df['volume']

        # Calculate price change
        delta_tp = typical_price.diff()

        # Initialize Money Flow Series with NaNs
        positive_mf = pd.Series(np.nan, index=self.df.index)
        negative_mf = pd.Series(np.nan, index=self.df.index)

        # Assign money flow based on price change direction
        positive_mf[delta_tp > 0] = raw_money_flow[delta_tp > 0]
        negative_mf[delta_tp < 0] = raw_money_flow[delta_tp < 0]

        # Sum flows over the window, filling initial NaNs with 0 for rolling sum
        positive_mf_sum = positive_mf.fillna(0).rolling(window=window).sum()
        negative_mf_sum = negative_mf.fillna(0).rolling(window=window).sum()

        # Calculate Money Flow Ratio, handle division by zero
        money_ratio = np.where(negative_mf_sum != 0, positive_mf_sum / negative_mf_sum, np.inf)

        # Calculate MFI
        mfi = np.where(
            money_ratio == np.inf,
            100, # If negative flow is zero, MFI is 100
            100 - (100 / (1 + money_ratio))
        )
        # Set initial periods to NaN where window isn't filled
        mfi[:window-1] = np.nan

        return pd.Series(mfi, index=self.df.index)

    def calculate_vwap(self) -> Optional[pd.Series]:
        """Calculate Volume Weighted Average Price (VWAP) - typically reset daily,
           but here calculated cumulatively over the provided DataFrame length."""
        # Warning: Standard VWAP resets daily. This is a cumulative VWAP over the DF.
        # For true daily VWAP, data needs to be grouped by day.
        cumulative_pv = (self.df['close'] * self.df['volume']).cumsum()
        cumulative_volume = self.df['volume'].cumsum()
        vwap = np.where(cumulative_volume != 0, cumulative_pv / cumulative_volume, np.nan)
        return pd.Series(vwap, index=self.df.index)


    def calculate_psar(self, acceleration: float = DEFAULT_PSAR_ACCELERATION, max_acceleration: float = DEFAULT_PSAR_MAX_ACCELERATION) -> Optional[pd.Series]:
        """Calculate Parabolic SAR (PSAR)."""
        high, low = self.df['high'], self.df['low']
        psar = pd.Series(index=self.df.index, dtype='float64')
        trend = pd.Series(index=self.df.index, dtype='int') # 1 for long, -1 for short
        af = pd.Series(index=self.df.index, dtype='float64') # Acceleration Factor
        ep = pd.Series(index=self.df.index, dtype='float64') # Extreme Point

        # Initial values (handle potential NaNs in early data)
        first_valid_index = self.df[['high', 'low', 'close']].dropna().index.min()
        if first_valid_index is None or first_valid_index >= len(self.df) - 1:
             return pd.Series(np.nan, index=self.df.index) # Not enough data

        # Start at the second available index
        start_index = first_valid_index + 1
        if start_index >= len(self.df): return pd.Series(np.nan, index=self.df.index)


        # Determine initial trend based on first two periods
        if self.df['close'].iloc[start_index] > self.df['close'].iloc[first_valid_index]:
            trend.iloc[start_index] = 1
            psar.iloc[start_index] = low.iloc[first_valid_index] # Start below first low
            ep.iloc[start_index] = high.iloc[start_index]
        else:
            trend.iloc[start_index] = -1
            psar.iloc[start_index] = high.iloc[first_valid_index] # Start above first high
            ep.iloc[start_index] = low.iloc[start_index]

        af.iloc[start_index] = acceleration

        # Iterate from the third data point onwards
        for i in range(start_index + 1, len(self.df)):
            # Previous values
            prev_psar = psar.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            prev_af = af.iloc[i-1]
            prev_ep = ep.iloc[i-1]

            # Calculate current PSAR prediction
            current_psar = prev_psar + prev_af * (prev_ep - prev_psar)

            # Adjust PSAR if it penetrates previous periods' prices
            if prev_trend == 1: # Uptrend
                 current_psar = min(current_psar, low.iloc[i-1], low.iloc[i-2] if i > start_index + 1 else low.iloc[i-1])
            else: # Downtrend
                 current_psar = max(current_psar, high.iloc[i-1], high.iloc[i-2] if i > start_index + 1 else high.iloc[i-1])

            # Check for trend reversal
            reverse = False
            if prev_trend == 1 and low.iloc[i] < current_psar: # Uptrend reverses
                 trend.iloc[i] = -1
                 current_psar = prev_ep # SAR becomes the recent extreme high
                 ep.iloc[i] = low.iloc[i] # New EP is the low of the reversal bar
                 af.iloc[i] = acceleration
                 reverse = True
            elif prev_trend == -1 and high.iloc[i] > current_psar: # Downtrend reverses
                 trend.iloc[i] = 1
                 current_psar = prev_ep # SAR becomes the recent extreme low
                 ep.iloc[i] = high.iloc[i] # New EP is the high of the reversal bar
                 af.iloc[i] = acceleration
                 reverse = True

            # If no reversal, continue the trend
            if not reverse:
                trend.iloc[i] = prev_trend
                af.iloc[i] = prev_af
                ep.iloc[i] = prev_ep

                # Update EP and AF if new extreme is made
                if trend.iloc[i] == 1 and high.iloc[i] > prev_ep:
                     ep.iloc[i] = high.iloc[i]
                     af.iloc[i] = min(prev_af + acceleration, max_acceleration)
                elif trend.iloc[i] == -1 and low.iloc[i] < prev_ep:
                     ep.iloc[i] = low.iloc[i]
                     af.iloc[i] = min(prev_af + acceleration, max_acceleration)

                 # PSAR for the current period is the calculated value (before potential reversal adjustments)
                 # No, PSAR for the current period *is* current_psar, potentially adjusted
                 # Need to recalculate psar for the *next* period based on *current* state if no reversal
                # If no reversal, the PSAR value *applied* to bar 'i' is 'current_psar'
                psar.iloc[i] = current_psar
                # Now update the AF and EP for the *next* calculation (which will happen in the next iteration for i+1)
                # The state (af.iloc[i], ep.iloc[i]) is already set correctly above
            else:
                # On reversal, the PSAR value *applied* to bar 'i' is the reversed value
                psar.iloc[i] = current_psar
                # State (af, ep) for next calculation already reset above

        return psar


    def calculate_rsi(self, window: int = DEFAULT_RSI_WINDOW) -> Optional[pd.Series]:
        """Calculate Relative Strength Index (RSI)."""
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/window, adjust=False).mean()

        rs = np.where(loss == 0, np.inf, gain / loss) # Use np.inf where loss is 0
        rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))

        return pd.Series(rsi, index=self.df.index)

    def calculate_stoch_rsi(self, rsi_window: int = DEFAULT_RSI_WINDOW, stoch_window: int = DEFAULT_STOCH_WINDOW, k_window: int = DEFAULT_K_WINDOW, d_window: int = DEFAULT_D_WINDOW) -> Optional[pd.DataFrame]:
        """Calculate Stochastic RSI (%K and %D)."""
        rsi = self.calculate_rsi(window=rsi_window)
        if rsi is None or rsi.isna().all():
            return pd.DataFrame({'k': pd.Series(np.nan, index=self.df.index), 'd': pd.Series(np.nan, index=self.df.index)})

        min_rsi = rsi.rolling(window=stoch_window).min()
        max_rsi = rsi.rolling(window=stoch_window).max()
        rsi_range = max_rsi - min_rsi

        stoch_rsi = np.where(
            rsi_range == 0,
            50.0, # Or 0 or 100 or np.nan depending on preference when range is zero
            ((rsi - min_rsi) / rsi_range) * 100
        )
        stoch_rsi_series = pd.Series(stoch_rsi, index=self.df.index)

        k_line = stoch_rsi_series.rolling(window=k_window).mean()
        d_line = k_line.rolling(window=d_window).mean()

        return pd.DataFrame({'k': k_line, 'd': d_line})


    def calculate_bollinger_bands(self, period: int = DEFAULT_BOLLINGER_BANDS_PERIOD, std_dev: float = DEFAULT_BOLLINGER_BANDS_STD_DEV) -> Optional[pd.DataFrame]:
        """Calculate Bollinger Bands."""
        rolling_mean = self.df['close'].rolling(window=period).mean()
        rolling_std = self.df['close'].rolling(window=period).std()
        bb_upper = rolling_mean + (rolling_std * std_dev)
        bb_lower = rolling_mean - (rolling_std * std_dev)
        return pd.DataFrame({'bb_upper': bb_upper, 'bb_mid': rolling_mean, 'bb_lower': bb_lower})

    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        if window is None:
            window = self.config.get("fibonacci_window", DEFAULT_FIBONACCI_WINDOW)

        if len(self.df) < window:
            self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)}) for Fibonacci window ({window}). Skipping.{RESET}")
            self.fib_levels = {}
            return {}

        df_slice = self.df.tail(window)
        # Use Decimal for precision, ensure conversion from potentially float
        high = Decimal(str(df_slice['high'].max()))
        low = Decimal(str(df_slice['low'].min()))
        diff = high - low

        if diff <= 0:
            self.logger.warning(f"{NEON_YELLOW}Fibonacci range is zero or negative (High: {high}, Low: {low}). Skipping.{RESET}")
            self.fib_levels = {}
            return {}

        levels = {}
        # Standard Retracement Levels (from high)
        for level_pct in FIB_LEVELS:
             level_val = high - (diff * Decimal(str(level_pct)))
             levels[f"Retr_{level_pct*100:.1f}%"] = level_val.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN) # Adjust precision as needed

        # Optional: Add extension levels if needed (e.g., 1.618)
        # level_ext = high + (diff * Decimal('0.618')) # Example extension above high
        # levels["Ext_161.8%"] = level_ext.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)

        self.fib_levels = levels
        self.logger.debug(f"Calculated Fibonacci Levels over last {window} periods: { {k: f'{v:.4f}' for k, v in levels.items()} }") # Log calculated levels
        return levels


    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 3) -> List[Tuple[str, Decimal, str]]:
        """Find nearest Fibonacci levels (support & resistance) to the current price."""
        if not self.fib_levels:
             self.logger.debug("Fibonacci levels not calculated yet.")
             # Attempt calculation if not done
             self.calculate_fibonacci_levels()
             if not self.fib_levels: return [] # Return empty if still no levels

        support_levels = sorted(
            [(name, level) for name, level in self.fib_levels.items() if level <= current_price],
            key=lambda x: current_price - x[1] # Closest below
        )
        resistance_levels = sorted(
            [(name, level) for name, level in self.fib_levels.items() if level > current_price],
            key=lambda x: x[1] - current_price # Closest above
        )

        nearest = []
        if support_levels:
            nearest.append((support_levels[0][0], support_levels[0][1], "Support"))
        if resistance_levels:
             nearest.append((resistance_levels[0][0], resistance_levels[0][1], "Resistance"))

        # Add more levels if needed, alternating between support and resistance
        s_idx, r_idx = 1, 1
        while len(nearest) < num_levels:
            next_s_dist = current_price - support_levels[s_idx][1] if s_idx < len(support_levels) else Decimal('inf')
            next_r_dist = resistance_levels[r_idx][1] - current_price if r_idx < len(resistance_levels) else Decimal('inf')

            if next_s_dist <= next_r_dist and s_idx < len(support_levels):
                nearest.append((support_levels[s_idx][0], support_levels[s_idx][1], "Support"))
                s_idx += 1
            elif r_idx < len(resistance_levels):
                 nearest.append((resistance_levels[r_idx][0], resistance_levels[r_idx][1], "Resistance"))
                 r_idx += 1
            else:
                 break # No more levels to add

        return nearest[:num_levels]

    # --- Signal Generation Logic ---

    def _get_indicator_value(self, name: str) -> float:
        """Safely retrieves the latest calculated indicator value."""
        val = self.indicator_values.get(name)
        if isinstance(val, (float, int)) and pd.notna(val):
            return float(val)
        elif isinstance(val, str): # Handle "Error" or "Insufficient Data"
             # self.logger.debug(f"Indicator '{name}' has non-numeric value: {val}")
             pass
        # else: self.logger.debug(f"Indicator '{name}' not found or is NaN.")
        return np.nan # Return NaN if not available or not numeric


    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> Tuple[str, float]:
        """
        Generate a trading signal (BUY, SELL, HOLD) based on the confluence
        of enabled indicator signals and their assigned weights.

        Returns:
            A tuple containing the signal ("BUY", "SELL", "HOLD") and the raw score.
        """
        signal_score = 0.0
        active_indicators = self.config["indicators"]
        weights = self.weights
        last_close = self._get_indicator_value("Close") # Use last close from stored df
        if pd.isna(last_close):
             last_close = float(self.df['close'].iloc[-1]) # Fallback

        # --- EMA Alignment ---
        if active_indicators.get("ema_alignment", False) and 'ema_short' in self.df.columns and 'ema_long' in self.df.columns:
            ema_short = self._get_indicator_value(f"EMA{self.config['ema_short_period']}")
            ema_long = self._get_indicator_value(f"EMA{self.config['ema_long_period']}")
            if not pd.isna(ema_short) and not pd.isna(ema_long):
                if ema_short > ema_long and last_close > ema_short: # Bullish alignment
                    signal_score += weights.get("ema_alignment", 0.0)
                elif ema_short < ema_long and last_close < ema_short: # Bearish alignment
                     signal_score -= weights.get("ema_alignment", 0.0)

        # --- SMA Trend Filter ---
        if active_indicators.get("sma_trend_filter", False) and 'sma50' in self.df.columns:
             sma50 = self._get_indicator_value("SMA50")
             if not pd.isna(sma50):
                 if last_close > sma50: # Price above longer-term MA (uptrend confirmation)
                      signal_score += weights.get("sma_trend_filter", 0.0) * 0.5 # Half weight for confirmation
                 elif last_close < sma50: # Price below longer-term MA (downtrend confirmation)
                      signal_score -= weights.get("sma_trend_filter", 0.0) * 0.5 # Half weight negative confirmation
                 # We slightly increase score towards trend, decrease against it

        # --- Momentum ---
        if active_indicators.get("momentum", False):
            momentum = self._get_indicator_value("Momentum")
            if not pd.isna(momentum):
                 if momentum > 0: signal_score += weights.get("momentum", 0.0)
                 elif momentum < 0: signal_score -= weights.get("momentum", 0.0)

        # --- Volume Confirmation ---
        if active_indicators.get("volume_confirmation", False) and 'volume_ma' in self.df.columns:
             volume = float(self.df['volume'].iloc[-1])
             volume_ma = self._get_indicator_value("VolumeMA")
             multiplier = self.config.get("volume_confirmation_multiplier", 1.5)
             if not pd.isna(volume_ma) and volume_ma > 0:
                 if volume > volume_ma * multiplier: # High volume confirmation
                      signal_score += weights.get("volume_confirmation", 0.0)
                 elif volume < volume_ma / multiplier: # Low volume might indicate weak move
                      signal_score -= weights.get("volume_confirmation", 0.0) * 0.5 # Penalize low volume slightly

        # --- Stochastic RSI ---
        if active_indicators.get("stoch_rsi", False):
            k = self._get_indicator_value("StochRSI_K")
            d = self._get_indicator_value("StochRSI_D")
            oversold = self.config.get("stoch_rsi_oversold", 20)
            overbought = self.config.get("stoch_rsi_overbought", 80)
            if not pd.isna(k) and not pd.isna(d):
                 if k > d and k < oversold: # Bullish cross below oversold
                      signal_score += weights.get("stoch_rsi", 0.0)
                 elif k < d and k > overbought: # Bearish cross above overbought
                      signal_score -= weights.get("stoch_rsi", 0.0)
                 elif k < oversold and d < oversold: # Deeply oversold
                      signal_score += weights.get("stoch_rsi", 0.0) * 0.7 # Slightly less weight than cross
                 elif k > overbought and d > overbought: # Deeply overbought
                      signal_score -= weights.get("stoch_rsi", 0.0) * 0.7 # Slightly less weight than cross

        # --- RSI ---
        if active_indicators.get("rsi", False):
            rsi = self._get_indicator_value("RSI")
            oversold = self.config.get("rsi_oversold", 30)
            overbought = self.config.get("rsi_overbought", 70)
            if not pd.isna(rsi):
                 if rsi < oversold: signal_score += weights.get("rsi", 0.0)
                 elif rsi > overbought: signal_score -= weights.get("rsi", 0.0)

        # --- CCI ---
        if active_indicators.get("cci", False):
             cci = self._get_indicator_value("CCI")
             oversold = self.config.get("cci_oversold", -100)
             overbought = self.config.get("cci_overbought", 100)
             if not pd.isna(cci):
                 if cci < oversold: signal_score += weights.get("cci", 0.0)
                 elif cci > overbought: signal_score -= weights.get("cci", 0.0)

        # --- Williams %R ---
        if active_indicators.get("wr", False):
             wr = self._get_indicator_value("Williams_R")
             oversold = self.config.get("williams_r_oversold", -80)
             overbought = self.config.get("williams_r_overbought", -20)
             if not pd.isna(wr):
                  if wr < oversold: signal_score += weights.get("wr", 0.0)
                  elif wr > overbought: signal_score -= weights.get("wr", 0.0)

        # --- MFI ---
        if active_indicators.get("mfi", False):
             mfi = self._get_indicator_value("MFI")
             oversold = self.config.get("mfi_oversold", 20)
             overbought = self.config.get("mfi_overbought", 80)
             if not pd.isna(mfi):
                  if mfi < oversold: signal_score += weights.get("mfi", 0.0)
                  elif mfi > overbought: signal_score -= weights.get("mfi", 0.0)

        # --- Bollinger Bands ---
        if active_indicators.get("bollinger_bands", False):
             bb_lower = self._get_indicator_value("BB_Lower")
             bb_upper = self._get_indicator_value("BB_Upper")
             if not pd.isna(bb_lower) and not pd.isna(bb_upper):
                 if last_close < bb_lower: # Price below lower band (potential reversal buy)
                      signal_score += weights.get("bollinger_bands", 0.0)
                 elif last_close > bb_upper: # Price above upper band (potential reversal sell)
                      signal_score -= weights.get("bollinger_bands", 0.0)

        # --- PSAR ---
        if active_indicators.get("psar", False):
             psar = self._get_indicator_value("PSAR")
             # Need previous PSAR to detect flip robustly
             prev_psar = float(self.df['psar'].iloc[-2]) if len(self.df) > 1 and pd.notna(self.df['psar'].iloc[-2]) else np.nan
             prev_close = float(self.df['close'].iloc[-2]) if len(self.df) > 1 else np.nan

             if not pd.isna(psar) and not pd.isna(prev_psar) and not pd.isna(prev_close):
                 # Check for flip: PSAR moves from above price to below price (Buy signal)
                 if prev_close > prev_psar and last_close > psar:
                     signal_score += weights.get("psar", 0.0)
                 # Check for flip: PSAR moves from below price to above price (Sell signal)
                 elif prev_close < prev_psar and last_close < psar:
                      signal_score -= weights.get("psar", 0.0)
                 # If no flip, check current position relative to PSAR
                 elif last_close > psar: # Price above PSAR (bullish continuation)
                      signal_score += weights.get("psar", 0.0) * 0.3 # Smaller weight for continuation
                 elif last_close < psar: # Price below PSAR (bearish continuation)
                      signal_score -= weights.get("psar", 0.0) * 0.3 # Smaller weight for continuation


        # --- SMA 10 Cross (Simple) ---
        if active_indicators.get("sma_10", False):
            sma10 = self._get_indicator_value("SMA10")
            if not pd.isna(sma10):
                 if last_close > sma10: signal_score += weights.get("sma_10", 0.0)
                 elif last_close < sma10: signal_score -= weights.get("sma_10", 0.0)

        # --- VWAP ---
        if active_indicators.get("vwap", False):
             vwap = self._get_indicator_value("VWAP")
             if not pd.isna(vwap):
                  if last_close > vwap: signal_score += weights.get("vwap", 0.0)
                  elif last_close < vwap: signal_score -= weights.get("vwap", 0.0)

        # --- Order Book Imbalance (Simple) ---
        if active_indicators.get("orderbook_imbalance", False) and orderbook_data:
             signal_score += self._check_orderbook(orderbook_data)


        # --- Final Signal Determination ---
        threshold = self.config.get("signal_score_threshold", 1.5)
        final_signal = "HOLD"
        if signal_score >= threshold:
             # Trend filter: Only take longs if price is above SMA50 (if enabled)
            if active_indicators.get("sma_trend_filter", False):
                 sma50 = self._get_indicator_value("SMA50")
                 if not pd.isna(sma50) and last_close < sma50:
                     final_signal = "HOLD (Buy Signal Filtered by Trend)"
                     self.logger.info(f"{NEON_YELLOW}Buy signal ({signal_score:.2f}) ignored, price below SMA50.{RESET}")
                     signal_score = 0 # Reset score to reflect filtering
                 else:
                     final_signal = "BUY"
            else:
                 final_signal = "BUY"

        elif signal_score <= -threshold:
             # Trend filter: Only take shorts if price is below SMA50 (if enabled)
             if active_indicators.get("sma_trend_filter", False):
                 sma50 = self._get_indicator_value("SMA50")
                 if not pd.isna(sma50) and last_close > sma50:
                     final_signal = "HOLD (Sell Signal Filtered by Trend)"
                     self.logger.info(f"{NEON_YELLOW}Sell signal ({signal_score:.2f}) ignored, price above SMA50.{RESET}")
                     signal_score = 0 # Reset score to reflect filtering
                 else:
                      final_signal = "SELL"
             else:
                 final_signal = "SELL"

        # Log the final score and decision reason
        self.logger.debug(f"Final Signal Score: {signal_score:.4f} (Threshold: {threshold}) -> {final_signal}")
        self.indicator_values["SignalScore"] = round(signal_score, 4) # Store the score

        return final_signal, signal_score


    def _check_orderbook(self, orderbook_data: Dict) -> float:
        """Analyze top levels of the order book for simple imbalance."""
        try:
            # Look at top N levels (e.g., 5)
            num_levels = 5
            bids = orderbook_data.get('bids', [])[:num_levels]
            asks = orderbook_data.get('asks', [])[:num_levels]

            if not bids or not asks: return 0.0

            bid_volume = sum(size for _, size in bids)
            ask_volume = sum(size for _, size in asks)

            if bid_volume == 0 and ask_volume == 0: return 0.0

            # Simple imbalance ratio (consider more sophisticated metrics later)
            imbalance_ratio = bid_volume / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0.5

            weight = self.weights.get("orderbook_imbalance", 0.0)
            # Give score based on skewness
            if imbalance_ratio > 0.65: # Significantly more bid volume -> bullish pressure
                 return weight
            elif imbalance_ratio < 0.35: # Significantly more ask volume -> bearish pressure
                 return -weight
            else:
                 return 0.0 # Relatively balanced

        except Exception as e:
            self.logger.error(f"{NEON_RED}Error analyzing order book: {e}{RESET}")
            return 0.0


    def calculate_entry_tp_sl(
        self, current_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculate entry price, take profit (TP), and stop loss (SL) levels
        based on the signal, current price, and ATR.
        """
        entry_price = current_price # Entry assumed at current market price
        atr_val = self._get_indicator_value("ATR")

        if pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}ATR is not available or invalid ({atr_val}). Cannot calculate TP/SL.{RESET}")
            return entry_price, None, None

        atr_decimal = Decimal(str(atr_val)) # Convert ATR float to Decimal
        tp_multiple = Decimal(str(self.config.get("take_profit_atr_multiple", 1.0)))
        sl_multiple = Decimal(str(self.config.get("stop_loss_atr_multiple", 1.5)))

        tp_distance = atr_decimal * tp_multiple
        sl_distance = atr_decimal * sl_multiple

        if signal == "BUY":
            take_profit = entry_price + tp_distance
            stop_loss = entry_price - sl_distance
        elif signal == "SELL":
            take_profit = entry_price - tp_distance
            stop_loss = entry_price + sl_distance
        else: # HOLD signal
            return entry_price, None, None

        # Optional: Adjust TP/SL to nearest Fibonacci level if close? (More complex logic)
        # nearest_fibs = self.get_nearest_fibonacci_levels(current_price, num_levels=5)
        # ... logic to check if calculated TP/SL is near a fib and snap to it ...

        # Ensure TP/SL are positive
        if take_profit <= 0: take_profit = None # Or set to a minimum value
        if stop_loss <= 0: stop_loss = None

        # Quantize to appropriate precision if needed (e.g., based on symbol tick size)
        # For now, return as calculated
        return entry_price, take_profit, stop_loss


    def calculate_confidence(self, signal_score: float) -> float:
        """Calculate a confidence score based on the signal strength."""
        threshold = self.config.get("signal_score_threshold", 1.5)
        if threshold == 0: return 50.0 # Avoid division by zero

        # Simple confidence: scales from 50% at threshold to 100%
        # Score needs to be at least threshold to generate signal
        if abs(signal_score) < threshold:
            return 0.0 # No confidence if below threshold

        # How far beyond the threshold did the score reach?
        excess_ratio = (abs(signal_score) / threshold) - 1.0
        # Scale confidence: e.g., 50% base + 50% * excess (capped at 100)
        # Adjust the scaling factor (e.g., 50) to control sensitivity
        confidence = 50.0 + (excess_ratio * 50.0)

        return min(max(confidence, 0.0), 100.0) # Clamp between 0 and 100

# --- Main Execution Logic ---

def analyze_symbol(symbol: str, config: Dict[str, Any], session: requests.Session) -> None:
    """Fetches data, analyzes a symbol, and logs the results."""
    logger = setup_logger(symbol) # Setup logger for this specific run/symbol
    logger.info(f"--- Starting Analysis Cycle for {symbol} ({config['interval']}) ---")

    # 1. Fetch Data
    logger.debug("Fetching klines...")
    klines = fetch_klines(symbol, config["interval"], limit=250, logger=logger, session=session) # Fetch enough for indicators
    if klines.empty:
        logger.error(f"{NEON_RED}Failed to fetch klines for {symbol}. Skipping analysis cycle.{RESET}")
        return

    logger.debug("Fetching orderbook...")
    orderbook_data = None
    if config["indicators"].get("orderbook_imbalance"):
        orderbook_data = fetch_orderbook(symbol, config["orderbook_limit"], logger=logger, session=session)
        if not orderbook_data:
             logger.warning(f"{NEON_YELLOW}Could not fetch orderbook data. Orderbook analysis will be skipped.{RESET}")

    logger.debug("Fetching current price...")
    current_price = fetch_current_price(symbol, logger=logger, session=session)
    if current_price is None:
         # Fallback to last close price from klines if ticker fails
         last_close_kline = klines['close'].iloc[-1]
         if pd.notna(last_close_kline):
             current_price = Decimal(str(last_close_kline))
             logger.warning(f"{NEON_YELLOW}Failed to fetch current ticker price, using last kline close price: {current_price}{RESET}")
         else:
              logger.error(f"{NEON_RED}Failed to fetch current price and kline data is invalid. Cannot proceed.{RESET}")
              return

    # 2. Analyze Data
    logger.debug("Initializing TradingAnalyzer...")
    try:
        analyzer = TradingAnalyzer(klines, config, logger, symbol)
    except ValueError as e:
        logger.error(f"{NEON_RED}Failed to initialize analyzer: {e}{RESET}")
        return

    # 3. Generate Signal
    logger.debug("Generating trading signal...")
    signal, score = analyzer.generate_trading_signal(current_price, orderbook_data)

    # 4. Calculate TP/SL and Confidence
    entry, tp, sl = analyzer.calculate_entry_tp_sl(current_price, signal)
    confidence = analyzer.calculate_confidence(score)

    # 5. Get Fibonacci Context
    nearest_fibs = []
    if config["indicators"].get("fibonacci_levels", False):
        nearest_fibs = analyzer.get_nearest_fibonacci_levels(current_price)

    # 6. Log Output
    output_lines = [
        f"\n{NEON_BLUE}---=== Analysis Result for {symbol} ({config['interval']}) ===---{RESET}",
        f"Timestamp:       {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Current Price:   {NEON_GREEN}{current_price:.4f}{RESET}", # Adjust precision as needed
        "-"*30,
        f"Signal:          {(NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW)}{signal}{RESET}",
        f"Signal Score:    {NEON_CYAN}{score:.4f}{RESET} (Threshold: {config['signal_score_threshold']})",
        f"Confidence:      {NEON_PURPLE}{confidence:.2f}%{RESET}",
        "-"*30,
        f"Entry Price:     {NEON_YELLOW}{entry:.4f}{RESET}" if entry else "Entry Price:     N/A",
        f"Take Profit:     {NEON_GREEN}{tp:.4f}{RESET}" if tp else "Take Profit:     N/A",
        f"Stop Loss:       {NEON_RED}{sl:.4f}{RESET}" if sl else "Stop Loss:       N/A",
        "-"*30,
        "Key Indicators:",
    ]
    # Format and add indicator values
    for k, v in analyzer.indicator_values.items():
        if isinstance(v, (float, int)) and pd.notna(v):
             output_lines.append(f"  {k:<15}: {NEON_YELLOW}{v:.4f}{RESET}")
        else:
             output_lines.append(f"  {k:<15}: {NEON_YELLOW}{v}{RESET}") # Show "Error" or "N/A"

    if nearest_fibs:
        output_lines.append("-"*30)
        output_lines.append("Nearest Fibonacci Levels:")
        for name, level, level_type in nearest_fibs:
            color = NEON_GREEN if level_type == "Support" else NEON_RED
            output_lines.append(f"  {name:<15}: {color}{level:.4f}{RESET} ({level_type})")

    output_lines.append(f"{NEON_BLUE}---=== End Analysis ===---{RESET}\n")
    logger.info("\n".join(output_lines))


def main() -> None:
    """Main function to run the scalping analysis loop."""
    # Load configuration
    config = load_config(CONFIG_FILE)

    # Get symbol and interval from config or user input
    symbol_input = input(f"{NEON_YELLOW}Enter symbol (e.g., BTCUSDT) or press Enter for default ({config.get('symbol', 'BTCUSDT')}): {RESET}").strip().upper()
    symbol = symbol_input if symbol_input else config.get('symbol', 'BTCUSDT')

    interval_input = input(f"{NEON_YELLOW}Enter interval (e.g., 1m, 5m, 1h) or press Enter for default ({config.get('interval', '5m')}): {RESET}").strip().lower()
    # Basic validation (can be expanded)
    # Add Bybit specific valid intervals if needed
    valid_intervals_approx = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]
    # Map user input like '1h' to Bybit's '60' if necessary (V5 API uses both, check preference)
    interval_mapping = {
        "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
        "1d": "D", "1w": "W", "1M": "M"
    }
    # Use mapping if input matches keys, otherwise use input directly
    interval = interval_mapping.get(interval_input, interval_input) if interval_input else config.get("interval", "5m")

    # Check if the final interval seems valid (simple check)
    # A more robust check would query exchange capabilities if possible
    if not interval or interval not in (list(interval_mapping.values()) + list(interval_mapping.keys())):
         print(f"{NEON_RED}Invalid interval '{interval}'. Exiting.{RESET}")
         return

    # Update config with runtime values
    config['symbol'] = symbol
    config['interval'] = interval
    loop_delay = config.get("loop_delay", LOOP_DELAY_SECONDS)

    # Create a persistent session
    shared_session = create_session()
    main_logger = setup_logger("main_loop") # Logger for the main loop itself

    main_logger.info(f"Starting WGWHalex Analysis Bot for {symbol} on {interval} interval.")
    main_logger.info(f"Using config file: {CONFIG_FILE}")
    main_logger.info(f"Loop delay set to {loop_delay} seconds.")

    try:
        while True:
            start_time = time.monotonic()
            analyze_symbol(symbol, config, shared_session)
            end_time = time.monotonic()

            elapsed = end_time - start_time
            wait_time = max(0, loop_delay - elapsed)

            if wait_time > 0:
                main_logger.debug(f"Analysis took {elapsed:.2f}s. Waiting {wait_time:.2f}s for next cycle...")
                time.sleep(wait_time)
            else:
                main_logger.warning(f"{NEON_YELLOW}Analysis cycle ({elapsed:.2f}s) exceeded loop delay ({loop_delay}s). Running next cycle immediately.{RESET}")

    except KeyboardInterrupt:
        main_logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        main_logger.error(f"{NEON_RED}An unexpected error occurred in the main loop: {e}{RESET}", exc_info=True)
    finally:
        shared_session.close() # Close the session
        main_logger.info("WGWHalex Analysis Bot stopped.")
        print(f"{NEON_PURPLE}WGWHalex stopped.{RESET}")


if __name__ == "__main__":
    main()

```
**Key Enhancements and Strategy Updates:**

1.  **Configuration Overhaul (`load_config`, `_ensure_config_keys`):**
    *   More robust loading: Handles file not found, JSON errors, and IO errors gracefully.
    *   **Automatic Defaulting:** If the config file exists but is missing keys (e.g., after an update), it now adds the missing keys and their default values from `default_config` *recursively* for nested dictionaries (like `indicators` and `weight_sets`). It then saves the updated config back.
    *   Added more configurable parameters (indicator periods, thresholds like RSI overbought/sold, SMA50 period, PSAR settings, loop delay).
    *   Renamed `stop_loss_multiple` -> `stop_loss_atr_multiple` and `take_profit_multiple` -> `take_profit_atr_multiple` for clarity.

2.  **Improved API Interaction (`bybit_request`):**
    *   Uses `time.time()` for timestamps (standard).
    *   Includes `X-BAPI-RECV-WINDOW` (recommended by Bybit).
    *   Correctly handles V5 signature generation for both GET (query params) and POST (JSON body).
    *   Uses a persistent `requests.Session` passed down through functions for efficiency (connection pooling, retries).
    *   More specific error handling (Timeout, ConnectionError, HTTPError, JSONDecodeError).
    *   Logs more detailed error messages, including the Bybit `retCode` and `retMsg`.
    *   Redacted sensitive info from debug logs.

3.  **Data Fetching Refinements (`fetch_klines`, `fetch_orderbook`, `fetch_current_price`):**
    *   `fetch_klines`: Reverses DataFrame so newest data is last (standard for TA). Converts types more carefully using `pd.to_numeric` with `errors='coerce'`. Handles empty data returns better.
    *   `fetch_orderbook`: Switched to use Bybit V5 API endpoint directly instead of `ccxt` for consistency with other API calls. Parses the V5 structure (`result.a`, `result.b`). Converts prices/sizes to `Decimal`.
    *   `fetch_current_price`: Prefers `markPrice` for derivatives (more relevant than `lastPrice` for funding etc.), falls back to `lastPrice`. Handles cases where the symbol isn't found or prices are missing. Includes a fallback to use the latest kline close price if the ticker fetch fails entirely.

4.  **`TradingAnalyzer` Enhancements:**
    *   **Initialization:** Takes the DataFrame and copies it to prevent modification of the original. Checks for required columns.
    *   **Safe Indicator Calculation (`_safe_calculate`):** Wraps all indicator calculations. Checks for sufficient data length, handles `KeyError`, `ZeroDivisionError`, and other exceptions gracefully, logs issues, and stores appropriate values (`float`, `np.nan`, `"Error"`, `"Insufficient Data"`) in `indicator_values`.
    *   **Indicator Calculation (`_calculate_all_indicators`, individual methods):**
        *   Calculations triggered based on `config["indicators"]` flags.
        *   Uses `.ewm()` for EMA and standard ATR calculation (more common than SMA for ATR).
        *   Improved safety in calculations involving division (e.g., Momentum, CCI, Williams %R, MFI, RSI, StochRSI, VWAP) using `np.where` or explicit checks.
        *   PSAR calculation refined for better handling of initial conditions and reversals.
        *   StochRSI calculation separated K and D lines clearly and stores them individually.
        *   Bollinger Bands calculation stores Upper, Mid, Lower bands individually.
        *   **SMA50 Trend Filter:** Added calculation for SMA50.
        *   **Fibonacci:** `calculate_fibonacci_levels` uses `Decimal` for better precision and handles zero range. `get_nearest_fibonacci_levels` now correctly identifies nearest support/resistance and returns them labeled.
    *   **Volume Confirmation:** Calculation uses `volume_ma` if available.
    *   **Indicator Value Retrieval (`_get_indicator_value`):** Safely gets the latest calculated value, returning `np.nan` if unavailable or non-numeric.

5.  **Strategy Update (`generate_trading_signal`):**
    *   **Scoring Logic:** More clearly iterates through enabled indicators and adds/subtracts weights from `config["weight_sets"]["default_scalping"]`.
    *   **Indicator Signal Logic Refined:**
        *   *EMA Alignment:* Checks price relative to short EMA *and* short EMA relative to long EMA.
        *   *SMA Trend Filter:* If enabled, adds a *small* score bonus if the price is aligned with the SMA50 trend (price > SMA50 for potential longs, price < SMA50 for potential shorts). More importantly, it **filters** signals against the trend (e.g., ignores a BUY signal if price < SMA50).
        *   *StochRSI:* Checks for bullish/bearish crosses in oversold/overbought zones *and* for being deeply in those zones (slightly different weights).
        *   *Volume:* Adds weight for high volume, slightly penalizes very low volume.
        *   *PSAR:* Checks for SAR flips (stronger signal) and price position relative to SAR (weaker continuation signal).
        *   *Orderbook:* Basic imbalance check using top 5 levels.
    *   **Final Signal Decision:** Compares the total score against `signal_score_threshold`. Applies the SMA Trend Filter *after* the initial score calculation. Logs the reason if a signal is filtered.

6.  **TP/SL Calculation (`calculate_entry_tp_sl`):**
    *   Uses `Decimal` for calculations involving price and ATR for better precision.
    *   Handles cases where ATR is invalid.
    *   Returns `None` for TP/SL if the signal is "HOLD" or ATR is unavailable.
    *   Ensures TP/SL don't go below zero.

7.  **Confidence Score (`calculate_confidence`):**
    *   Now dynamically calculated based on how much the `signal_score` exceeds the `threshold`. Score at threshold = 50% confidence, increasing towards 100%. Returns 0% if below threshold.

8.  **Logging and Output (`setup_logger`, `analyze_symbol`):**
    *   **Sensitive Formatter:** Redacts API Key/Secret more effectively (shows first/last 4 of key).
    *   **Logger Setup:** Configures timezone correctly. Clears existing handlers to prevent duplication if called multiple times. Sets level on stream handler too.
    *   **Analysis Output:** More structured and readable console output using Colorama. Clearly displays current price, signal, score, confidence, TP/SL, key indicator values (formatted), and nearest Fibonacci levels with support/resistance labels.

9.  **Main Loop (`main`):**
    *   Allows user input for symbol/interval, falling back to config defaults.
    *   Includes basic interval validation and mapping (e.g., '1h' to '60').
    *   Uses the persistent `requests.Session`.
    *   Calculates loop timing accurately and waits dynamically. Handles cases where analysis takes longer than the loop delay.
    *   Includes top-level exception handling and graceful shutdown on `KeyboardInterrupt`.

This enhanced version provides a more robust, configurable, and informative analysis framework. The strategy incorporates a trend filter and refined logic for several indicators, while the code itself is more resilient to errors and edge cases.
