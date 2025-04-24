Okay, Seeker of Digital Insights! Pyrmethus shall reforge this script, infusing it with deeper analytical capabilities, especially concerning the vital signs of volume, and enhancing its visual enchantments. We shall summon the power of `pandas_ta` to streamline and expand our arsenal of indicators.

Prepare your Termux!

**Enhancements Overview:**

1.  **Dependency:** Integrated `pandas_ta` for robust and simplified indicator calculations. (You'll need to `pip install pandas_ta`).
2.  **New Indicators Added:**
    *   **VWAP (Volume Weighted Average Price):** Crucial for intraday volume context.
    *   **Chaikin Money Flow (CMF):** Volume-weighted accumulation/distribution indicator.
    *   **Ichimoku Cloud:** Comprehensive trend, momentum, and S/R indicator.
    *   **Stochastic Oscillator:** Classic momentum oscillator.
    *   **Awesome Oscillator (AO):** Measures market momentum.
3.  **Enhanced Volume Analysis:**
    *   **Volume MA Comparison:** Explicitly shows if the last candle's volume is above/below its MA.
    *   **Volume Spike Detection:** Highlights candles with volume significantly exceeding the MA.
    *   **VWAP Context:** Shows price relative to VWAP.
    *   **CMF Interpretation:** Added interpretation for CMF signal.
    *   **Volume Profile (Approximation):** Added a simple volume profile histogram based on kline data to identify high-volume price zones within the fetched data.
4.  **Improved Output:**
    *   **Structured Sections:** Output is now divided into logical sections (Market Info, Key Stats, Oscillators, Volume Analysis, Trend/Momentum, Levels, Order Book, Volume Profile).
    *   **Clearer Interpretations:** Indicator interpretations now often include the actual value for context (e.g., `RSI (72.5) > 70`).
    *   **Enhanced Colorization:** More consistent and meaningful use of `NEON_` colors and `Style.BRIGHT` for emphasis and clarity. Error messages are consistently highlighted.
    *   **Last 3 Values:** Displaying the last 3 values for key indicators to show recent changes.
5.  **Code Refinements:**
    *   Refactored indicator calculations using `pandas_ta`.
    *   Updated `config.json` defaults to include new indicator parameters.
    *   Improved error handling and logging messages with better colorization.
    *   Added Stochastic RSI and EMA to the analysis output.

**Pre-requisites:**

Make sure you have the necessary libraries installed in Termux:

```bash
pip install pandas numpy requests colorama python-dotenv ccxt pandas_ta pytz
```

**(Note:** `pytz` is often needed by `pandas_ta` for timezone handling).

**Updated `config.json` (Default Structure):**

If you let the script create a new `config.json`, it will look like this. Add these new sections/keys if using an existing one:

```json
{
  "interval": "15",
  "analysis_interval": 30,
  "retry_delay": 5,
  "momentum_period": 10,
  "momentum_ma_short": 12,
  "momentum_ma_long": 26,
  "volume_ma_period": 20,
  "atr_period": 14,
  "trend_strength_threshold": 0.4,
  "sideways_atr_multiplier": 1.5,
  "indicators": {
    "ema_alignment": true, // Controls EMA trend check
    "momentum": true,      // Controls Momentum MA trend check
    "volume_confirmation": true, // Placeholder, more specific checks added
    "divergence": true,    // Controls MACD Divergence check
    "stoch_rsi": true,
    "rsi": true,
    "macd": true,          // Enable full MACD analysis
    "bollinger_bands": true,
    "vwap": true,
    "cmf": true,
    "ichimoku": true,
    "stoch": true,
    "ao": true
  },
  "weight_sets": { // Placeholder - not actively used for signal generation in this script
    "low_volatility": {
      "ema_alignment": 0.4,
      "momentum": 0.3,
      "volume_confirmation": 0.2,
      "divergence": 0.1,
      "stoch_rsi": 0.7,
      "rsi": 0.0,
      "macd": 0.0,
      "bollinger_bands": 0.0
    }
  },
  "rsi_period": 14,
  "rsi_long_period": 100, // Added separate config for long RSI
  "stoch_rsi_period": 14,
  "stoch_rsi_stoch_window": 14, // Clarified Stoch RSI params
  "stoch_rsi_k": 3,
  "stoch_rsi_d": 3,
  "stoch_k": 14,            // Stochastic Oscillator params
  "stoch_d": 3,
  "stoch_smooth_k": 3,
  "bollinger_bands_period": 20,
  "bollinger_bands_std_dev": 2,
  "macd_fast": 12,         // MACD params
  "macd_slow": 26,
  "macd_signal": 9,
  "cmf_period": 20,        // CMF param
  "ao_fast": 5,            // Awesome Oscillator params
  "ao_slow": 34,
  "ichimoku_tenkan": 9,    // Ichimoku params
  "ichimoku_kijun": 26,
  "ichimoku_senkou": 52,
  "orderbook_limit": 100,
  "orderbook_cluster_threshold": 1000,
  "volume_spike_multiplier": 2.5, // Multiplier for detecting volume spikes vs MA
  "volume_profile_bins": 15     // Number of bins for Volume Profile approximation
}
```

**The Reforged Script (`neonta.py`):**

```python
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, getcontext
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta # Summoning the power of pandas_ta
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration & Constants ---
getcontext().prec = 10 # Precision for Decimal
init(autoreset=True) # Initialize Colorama
load_dotenv()        # Load secrets from .env

# Bybit Credentials & Settings
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use bright red for critical errors
    print(f"{Fore.LIGHTRED_EX}{Style.BRIGHT}FATAL: BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env{Style.RESET_ALL}")
    raise ValueError("API Keys not found in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago") # Or your preferred timezone
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
# Extended valid intervals slightly based on Bybit V5 API doc examples
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # HTTP codes to trigger retries

# Colorama Colors - The Wizard's Palette
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_WHITE = Fore.WHITE
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT

os.makedirs(LOG_DIRECTORY, exist_ok=True) # Ensure log directory exists

# --- Helper Classes & Functions ---

class SensitiveFormatter(logging.Formatter):
    """Hides API keys in log records."""
    def format(self, record):
        msg = super().format(record)
        # Use placeholders consistently
        if API_KEY: msg = msg.replace(API_KEY, "***API_KEY***")
        if API_SECRET: msg = msg.replace(API_SECRET, "***API_SECRET***")
        return msg

def load_config(filepath: str) -> dict:
    """Loads configuration from JSON, creates default if missing."""
    # Updated defaults including new indicators
    default_config = {
        "interval": "15",
        "analysis_interval": 30,
        "retry_delay": 5,
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "sideways_atr_multiplier": 1.5,
        "indicators": {
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "divergence": True, "stoch_rsi": True, "rsi": True, "macd": True,
            "bollinger_bands": True, "vwap": True, "cmf": True, "ichimoku": True,
            "stoch": True, "ao": True
        },
        "weight_sets": { # Placeholder - not used for signal generation here
            "low_volatility": {
              "ema_alignment": 0.4, "momentum": 0.3, "volume_confirmation": 0.2,
              "divergence": 0.1, "stoch_rsi": 0.7, "rsi": 0.0, "macd": 0.0,
              "bollinger_bands": 0.0
            }
        },
        "rsi_period": 14,
        "rsi_long_period": 100,
        "stoch_rsi_period": 14,
        "stoch_rsi_stoch_window": 14,
        "stoch_rsi_k": 3,
        "stoch_rsi_d": 3,
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth_k": 3,
        "bollinger_bands_period": 20,
        "bollinger_bands_std_dev": 2,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "cmf_period": 20,
        "ao_fast": 5,
        "ao_slow": 34,
        "ichimoku_tenkan": 9,
        "ichimoku_kijun": 26,
        "ichimoku_senkou": 52,
        "orderbook_limit": 100,
        "orderbook_cluster_threshold": 1000,
        "volume_spike_multiplier": 2.5,
        "volume_profile_bins": 15
    }

    if not os.path.exists(filepath):
        print(f"{NEON_YELLOW}{BRIGHT}# Conjuring default configuration at {filepath}...{RESET}")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            print(f"{NEON_GREEN}{BRIGHT}Created new config file '{filepath}' with defaults.{RESET}")
            return default_config
        except IOError as e:
            print(f"{NEON_RED}{BRIGHT}Error creating config file:{RESET} {e}")
            print(f"{NEON_YELLOW}{BRIGHT}Loading embedded defaults instead.{RESET}")
            return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            loaded_config = json.load(f)
            # Simple validation/merge: ensure all default keys exist
            for key, value in default_config.items():
                if key not in loaded_config:
                    print(f"{NEON_YELLOW}Config key '{key}' missing, adding default.{RESET}")
                    loaded_config[key] = value
                elif isinstance(value, dict): # Check nested dicts like 'indicators'
                     for sub_key, sub_value in value.items():
                         if isinstance(loaded_config[key], dict) and sub_key not in loaded_config[key]:
                             print(f"{NEON_YELLOW}Config sub-key '{key}.{sub_key}' missing, adding default.{RESET}")
                             loaded_config[key][sub_key] = sub_value
            return loaded_config
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"{NEON_RED}{BRIGHT}Error loading or parsing '{filepath}':{RESET} {e}")
        print(f"{NEON_YELLOW}{BRIGHT}Loading embedded defaults instead.{RESET}")
        return default_config


CONFIG = load_config(CONFIG_FILE)

def create_session() -> requests.Session:
    """Creates a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5, # Shorter delay between retries
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"Content-Type": "application/json"}) # Default header
    return session

def setup_logger(symbol: str) -> logging.Logger:
    """Configures logging to file and console with color."""
    timestamp = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (no color, sensitive formatting)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream Handler (with color, sensitive formatting)
    stream_handler = logging.StreamHandler()
    # Improved color formatting for levelname
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} [{NEON_YELLOW}%(levelname)s{RESET}] %(message)s",
        datefmt='%H:%M:%S' # Shorter time format for console
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False # Prevent root logger from handling messages
    return logger

def bybit_request(
    method: str,
    endpoint: str,
    params: dict | None = None,
    logger: logging.Logger | None = None,
    session: requests.Session = None # Allow passing session
) -> dict | None:
    """Sends an authenticated request to the Bybit API."""
    if not session:
        session = create_session() # Create session if not provided
    try:
        params = params or {}
        # Timestamp MUST be UTC milliseconds as string
        timestamp = str(int(time.time() * 1000))
        # Signature generation depends on GET/POST
        query_string = ""
        request_body = ""

        # Prepare parameters based on method
        signature_params = params.copy()
        if method == 'GET':
            query_string = "&".join(f"{k}={v}" for k, v in sorted(signature_params.items()))
        elif method == 'POST':
             # Ensure body is JSON encoded string for signature if not empty
            if signature_params:
                 request_body = json.dumps(signature_params, separators=(',', ':'))

        # Construct signature string: timestamp + api_key + recv_window + (query_string or request_body)
        # Bybit V5 uses recv_window=5000 by default if not specified. Let's include it for clarity.
        recv_window = "5000"
        param_str = timestamp + API_KEY + recv_window + (query_string if method == 'GET' else request_body)

        signature = hmac.new(
            API_SECRET.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            # Content-Type set by session or request kwargs
        }
        url = f"{BASE_URL}{endpoint}"
        request_kwargs = {
            "method": method,
            "url": url,
            "headers": headers,
            "timeout": 10, # Standard timeout
        }

        # Add params/data based on method
        if method == "GET":
            request_kwargs["params"] = params
        elif method == "POST":
            request_kwargs["data"] = request_body # Send as string for POST
            headers["Content-Type"] = "application/json" # Ensure correct content type for POST

        response = session.request(**request_kwargs)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        json_response = response.json()

        # Check Bybit's specific return code
        if json_response and json_response.get("retCode") == 0:
            return json_response
        else:
            if logger:
                ret_code = json_response.get('retCode', 'N/A')
                ret_msg = json_response.get('retMsg', 'No message')
                logger.error(
                    f"{NEON_RED}{BRIGHT}Bybit API Error:{RESET} Code {NEON_YELLOW}{ret_code}{RESET}, Msg: {NEON_YELLOW}{ret_msg}{RESET} (Endpoint: {endpoint}, Params: {params})"
                )
            return None

    except requests.exceptions.HTTPError as http_err:
        if logger:
            # Log detailed HTTP error, including response body if available
            response_text = http_err.response.text if http_err.response else "No response body"
            logger.error(f"{NEON_RED}{BRIGHT}HTTP Error during Bybit request:{RESET} {http_err} - Status: {http_err.response.status_code if http_err.response else 'N/A'} - Body: {response_text}")
        return None
    except requests.exceptions.RequestException as req_err:
        if logger:
            logger.error(f"{NEON_RED}{BRIGHT}Network/Request Error during Bybit request:{RESET} {req_err}")
        return None
    except Exception as e:
        if logger:
            logger.exception(f"{NEON_RED}{BRIGHT}Unexpected error during Bybit request:{RESET} {e}")
        return None


def fetch_current_price(symbol: str, logger: logging.Logger, session: requests.Session) -> Decimal | None:
    """Fetches the last traded price for a symbol."""
    endpoint = "/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    response = bybit_request("GET", endpoint, params, logger, session)
    try:
        if not response or response.get("retCode") != 0 or not response.get("result"):
            logger.error(f"{NEON_RED}{BRIGHT}Failed to fetch ticker data:{RESET} Response: {response}")
            return None
        tickers = response["result"].get("list", [])
        if not tickers:
             logger.error(f"{NEON_RED}{BRIGHT}Ticker list empty for {symbol}.{RESET}")
             return None
        # Assume the first ticker in the list is the correct one when requesting a single symbol
        ticker = tickers[0]
        if ticker.get("symbol") == symbol:
            last_price_str = ticker.get("lastPrice")
            if not last_price_str:
                logger.error(f"{NEON_RED}{BRIGHT}No 'lastPrice' found in ticker data for {symbol}.{RESET}")
                return None
            return Decimal(last_price_str)
        else:
             logger.error(f"{NEON_RED}{BRIGHT}Symbol mismatch in ticker data: Expected {symbol}, got {ticker.get('symbol')}.{RESET}")
             return None

    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.error(f"{NEON_RED}{BRIGHT}Error processing ticker data for {symbol}:{RESET} {e} - Data: {response}")
        return None
    except Exception as e:
        logger.exception(f"{NEON_RED}{BRIGHT}Unexpected error fetching current price:{RESET} {e}")
        return None


def fetch_klines(
    symbol: str, interval: str, limit: int = 200, logger: logging.Logger = None, session: requests.Session = None
) -> pd.DataFrame:
    """Fetches historical kline data and returns a pandas DataFrame."""
    endpoint = "/v5/market/kline"
    # Bybit expects `limit` as string or int, API handles it
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    response = bybit_request("GET", endpoint, params, logger, session)
    try:
        if (
            response
            and response.get("retCode") == 0
            and response.get("result")
            and response["result"].get("list")
        ):
            data = response["result"]["list"]
            if not data:
                logger.warning(f"{NEON_YELLOW}No kline data returned for {symbol} interval {interval}.{RESET}")
                return pd.DataFrame()

            # Kline format: [timestamp, open, high, low, close, volume, turnover]
            columns = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
            df = pd.DataFrame(data, columns=columns)

            # Convert timestamp to datetime (UTC)
            df["start_time"] = pd.to_numeric(df["start_time"])
            df["timestamp"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)
            df = df.set_index('timestamp') # Set datetime index

            # Convert OHLCV to numeric, coercing errors
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Ensure required columns exist and are numeric
            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns) or df[list(required_cols)].isnull().any().any():
                missing = required_cols - set(df.columns)
                null_cols = df[list(required_cols)].isnull().any()
                null_cols_str = ", ".join(null_cols[null_cols].index)
                if logger:
                    logger.error(
                        f"{NEON_RED}{BRIGHT}Kline data issue:{RESET} Missing: {missing if missing else 'None'}. Contains Nulls: {null_cols_str if null_cols_str else 'None'}."
                    )
                return pd.DataFrame()

            df = df.sort_index() # Ensure data is time-ordered
            # Add standard 'datetime' column for compatibility if needed elsewhere
            # df['datetime'] = df.index
            return df[["open", "high", "low", "close", "volume"]] # Return standard OHLCV

        if logger:
            logger.error(f"{NEON_RED}{BRIGHT}Failed to fetch klines:{RESET} Response: {response}")
        return pd.DataFrame()

    except (KeyError, ValueError, TypeError) as e:
        if logger:
            logger.error(f"{NEON_RED}{BRIGHT}Error processing kline data:{RESET} {e} - Data: {response}")
        return pd.DataFrame()
    except Exception as e:
        if logger:
            logger.exception(f"{NEON_RED}{BRIGHT}Unexpected error fetching klines:{RESET} {e}")
        return pd.DataFrame()


def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger) -> dict | None:
    """Fetches order book data using ccxt with retries."""
    retry_count = 0
    # Initialize ccxt exchange instance inside the function
    try:
        exchange = ccxt.bybit({'apiKey': API_KEY, 'secret': API_SECRET})
        # exchange.set_sandbox_mode(True) # Enable if using Bybit testnet
    except Exception as e:
         logger.error(f"{NEON_RED}{BRIGHT}Failed to initialize ccxt:{RESET} {e}")
         return None

    while retry_count <= MAX_API_RETRIES:
        try:
            # Fetch L2 order book data
            orderbook_data = exchange.fetch_l2_order_book(symbol, limit=limit)
            if orderbook_data and 'bids' in orderbook_data and 'asks' in orderbook_data:
                # Convert bid/ask tuples to lists of dicts for easier processing later if needed
                # orderbook_data['bids'] = [{'price': p, 'size': s} for p, s in orderbook_data['bids']]
                # orderbook_data['asks'] = [{'price': p, 'size': s} for p, s in orderbook_data['asks']]
                return orderbook_data
            else:
                logger.error(
                    f"{NEON_RED}{BRIGHT}Failed to fetch valid orderbook data (retCode or structure error). Attempt {retry_count + 1}/{MAX_API_RETRIES + 1}{RESET}"
                )

        except ccxt.RateLimitExceeded as e:
             logger.warning(f"{NEON_YELLOW}ccxt Rate Limit Exceeded fetching orderbook: {e}. Retrying...{RESET}")
             time.sleep(exchange.rateLimit / 1000 * (retry_count + 1)) # Backoff based on ccxt info
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.error(
                f"{NEON_RED}{BRIGHT}ccxt Error fetching orderbook:{RESET} {type(e).__name__}: {e}. Attempt {retry_count + 1}/{MAX_API_RETRIES + 1}. Retrying in {RETRY_DELAY_SECONDS}s...{RESET}"
            )
        except Exception as e:
            logger.exception(
                f"{NEON_RED}{BRIGHT}Unexpected error fetching orderbook with ccxt:{RESET} {e}. Attempt {retry_count + 1}/{MAX_API_RETRIES + 1}. Retrying...{RESET}"
            )

        time.sleep(RETRY_DELAY_SECONDS * (retry_count + 1)) # Exponential backoff
        retry_count += 1

    logger.error(
        f"{NEON_RED}{BRIGHT}Max retries reached ({MAX_API_RETRIES}) for orderbook fetch using ccxt. Aborting.{RESET}"
    )
    return None


# --- Main Analysis Class ---

class TradingAnalyzer:
    """Encapsulates trading analysis logic and indicator calculations."""
    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict,
        symbol: str,
        interval: str,
    ):
        if df.empty:
            logger.error(f"{NEON_RED}{BRIGHT}TradingAnalyzer initialized with empty DataFrame.{RESET}")
            raise ValueError("DataFrame cannot be empty for analysis.")

        self.df = df.copy() # Work on a copy
        self.logger = logger
        self.config = config
        self.symbol = symbol
        self.interval = interval
        self.levels = {}       # Combined S/R levels (Fib, Pivot, etc.)
        self.fib_levels = {}   # Store calculated Fib levels separately if needed
        self.indicators = {}   # Store calculated indicator series/values
        self.last_analysis_output = "" # Store formatted output

        # --- Calculate Indicators on Initialization ---
        self.logger.info(f"{NEON_BLUE}# Calculating indicators for {self.symbol} ({self.interval})...{RESET}")
        self._calculate_all_indicators()


    def _calculate_all_indicators(self):
        """Calculate all technical indicators using pandas_ta."""
        if self.df.empty:
            self.logger.error(f"{NEON_RED}{BRIGHT}Cannot calculate indicators on empty DataFrame.{RESET}")
            return

        try:
            # Add indicators using pandas_ta strategy (more efficient)
            self.df.ta.strategy(ta.Strategy(
                name="ComprehensiveTA",
                ta=[
                    {"kind": "sma", "length": 10, "col_names": "SMA_10"},
                    {"kind": "sma", "length": 50, "col_names": "SMA_50"},
                    {"kind": "sma", "length": 200, "col_names": "SMA_200"},
                    {"kind": "ema", "length": 12, "col_names": "EMA_12"},
                    {"kind": "ema", "length": 26, "col_names": "EMA_26"},
                    {"kind": "ema", "length": 50, "col_names": "EMA_50"},
                    {"kind": "ema", "length": 200, "col_names": "EMA_200"},
                    {"kind": "rsi", "length": self.config["rsi_period"], "col_names": "RSI"},
                    {"kind": "rsi", "length": self.config["rsi_long_period"], "col_names": "RSI_long"},
                    {"kind": "stochrsi", "rsi_length": self.config["stoch_rsi_period"],
                     "k": self.config["stoch_rsi_k"], "d": self.config["stoch_rsi_d"],
                     "length": self.config["stoch_rsi_stoch_window"], "col_names": ("STOCHRSIk", "STOCHRSId")},
                    {"kind": "stoch", "k": self.config["stoch_k"], "d": self.config["stoch_d"],
                     "smooth_k": self.config["stoch_smooth_k"], "col_names": ("STOCHk", "STOCHd")},
                    {"kind": "macd", "fast": self.config["macd_fast"], "slow": self.config["macd_slow"],
                     "signal": self.config["macd_signal"], "col_names": ("MACD", "MACDh", "MACDs")},
                    {"kind": "bbands", "length": self.config["bollinger_bands_period"],
                     "std": self.config["bollinger_bands_std_dev"], "col_names": ("BBL", "BBM", "BBU", "BBB", "BBP")},
                    {"kind": "atr", "length": self.config["atr_period"], "col_names": "ATR"},
                    {"kind": "obv", "col_names": "OBV"},
                    {"kind": "adx", "length": self.config["atr_period"], "col_names": ("ADX", "DMP", "DMN")}, # atr_period often used for ADX length
                    {"kind": "cci", "length": 20, "col_names": "CCI"}, # Default 20 period for CCI
                    {"kind": "mfi", "length": 14, "col_names": "MFI"}, # Default 14 period for MFI
                    {"kind": "willr", "length": 14, "col_names": "WILLR"}, # Default 14 for Williams %R
                    {"kind": "vwap", "col_names": "VWAP"},
                    {"kind": "cmf", "length": self.config["cmf_period"], "col_names": "CMF"},
                    {"kind": "ichimoku", "tenkan": self.config["ichimoku_tenkan"],
                     "kijun": self.config["ichimoku_kijun"], "senkou": self.config["ichimoku_senkou"],
                     "col_names": ("ISA", "ISB", "ITS", "IKS", "ICS")},
                     # ISA: Senkou A, ISB: Senkou B, ITS: Tenkan, IKS: Kijun, ICS: Chikou
                    {"kind": "ao", "fast": self.config["ao_fast"], "slow": self.config["ao_slow"], "col_names": "AO"},
                    # Basic Volume MA
                    {"kind": "sma", "close": "volume", "length": self.config["volume_ma_period"], "col_names": "VOL_MA"},
                ]
            ))

            # --- Additional Custom Calculations ---
            # Momentum (Price Rate of Change)
            self.df['Momentum'] = self.df.ta.roc(length=self.config["momentum_period"])

            # Volume Spike Detection
            spike_threshold = self.df['VOL_MA'] * self.config['volume_spike_multiplier']
            self.df['Vol_Spike'] = self.df['volume'] > spike_threshold

            # Simple volume trend (is volume increasing/decreasing?) - looking back 1 period
            self.df['Vol_Trend_Up'] = self.df['volume'] > self.df['volume'].shift(1)

            # Store results for easy access in analyze()
            self.indicators = self.df.iloc[-3:].to_dict('list') # Get last 3 rows as dict of lists
            self.indicators['full_df'] = self.df # Keep full df if needed for other calcs

            self.logger.info(f"{NEON_GREEN}Indicators calculated successfully.{RESET}")

        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Error calculating indicators:{RESET} {e}")
            # Fill indicators dict with Nones or empty lists to prevent errors later
            self.indicators = {col: [None] * 3 for col in self.df.columns}
            self.indicators['full_df'] = pd.DataFrame() # Empty df


    def calculate_fibonacci_retracement(self, high: float, low: float, current_price: float):
        """Calculates Fibonacci retracement levels based on provided high/low."""
        try:
            diff = high - low
            if diff == 0:
                 self.logger.warning(f"{NEON_YELLOW}High and Low are identical ({high}), cannot calculate Fibonacci levels.{RESET}")
                 return

            # Standard retracement levels
            levels_data = {
                "Fib 0.0% (High)": high,
                "Fib 23.6%": high - diff * 0.236,
                "Fib 38.2%": high - diff * 0.382,
                "Fib 50.0%": high - diff * 0.5,
                "Fib 61.8%": high - diff * 0.618,
                "Fib 78.6%": high - diff * 0.786,
                # "Fib 88.6%": high - diff * 0.886, # Less common, can add if needed
                "Fib 100% (Low)": low,
                # Potential Extension Levels (optional)
                # "Fib Ext 127.2%": high + diff * 0.272,
                # "Fib Ext 161.8%": high + diff * 0.618,
            }
            self.fib_levels = levels_data

            # Add to main S/R levels dictionary
            if "Support" not in self.levels: self.levels["Support"] = {}
            if "Resistance" not in self.levels: self.levels["Resistance"] = {}

            for label, value in levels_data.items():
                if value < current_price:
                    self.levels["Support"][label] = value
                elif value > current_price:
                    self.levels["Resistance"][label] = value
                # else: level is very close to current price

        except ZeroDivisionError:
            self.logger.error(f"{NEON_RED}{BRIGHT}Fibonacci calculation error: Division by zero (diff is zero).{RESET}")
        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Unexpected Fibonacci calculation error:{RESET} {e}")


    def calculate_pivot_points(self, high: float, low: float, close: float):
        """Calculates Standard Pivot Points."""
        try:
            # Ensure high >= low
            if high < low:
                self.logger.warning(f"{NEON_YELLOW}Pivot Point Warning: High ({high}) < Low ({low}). Swapping.{RESET}")
                high, low = low, high

            pivot = (high + low + close) / 3
            # Support Levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            # Resistance Levels
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)

            pivot_data = {"S3": s3, "S2": s2, "S1": s1, "Pivot": pivot, "R1": r1, "R2": r2, "R3": r3}

            # Add to main S/R levels dictionary
            if "Support" not in self.levels: self.levels["Support"] = {}
            if "Resistance" not in self.levels: self.levels["Resistance"] = {}

            current_price = close # Use close price for pivot S/R classification relative to calculation time
            for label, value in pivot_data.items():
                 # Decide if it's support or resistance relative to the pivot itself
                 # Simpler: store all under a 'Pivots' key
                 # Or classify based on current price like Fibs
                key_prefix = "Pivot "
                if value < current_price:
                    self.levels["Support"][key_prefix + label] = value
                elif value > current_price:
                     self.levels["Resistance"][key_prefix + label] = value
                else: # Add pivot itself if close
                    if label == "Pivot":
                        self.levels["Pivot Point"] = value


        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Pivot point calculation error:{RESET} {e}")


    def find_nearest_levels(
        self, current_price: float, num_levels: int = 5
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Finds the N nearest support and resistance levels from the combined levels dict."""
        try:
            all_supports = []
            all_resistances = []

            # Flatten the levels dictionary
            for level_type, type_levels in self.levels.items():
                if isinstance(type_levels, dict):
                    for label, value in type_levels.items():
                        if isinstance(value, (float, Decimal, int)) and value != current_price:
                            if value < current_price:
                                all_supports.append((label, float(value)))
                            else:
                                all_resistances.append((label, float(value)))
                elif isinstance(type_levels, (float, Decimal, int)) and type_levels != current_price: # Handle top-level entries like 'Pivot Point'
                     if type_levels < current_price:
                         all_supports.append((level_type, float(type_levels)))
                     else:
                         all_resistances.append((level_type, float(type_levels)))


            # Sort by distance to current price
            all_supports.sort(key=lambda x: current_price - x[1]) # Nearest below
            all_resistances.sort(key=lambda x: x[1] - current_price) # Nearest above

            return all_supports[:num_levels], all_resistances[:num_levels]

        except (KeyError, TypeError) as e:
            self.logger.error(f"{NEON_RED}{BRIGHT}Error finding nearest levels:{RESET} {e}")
            return [], []
        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Unexpected error finding nearest levels:{RESET} {e}")
            return [], []

    def _format_indicator_value(self, value, precision=2):
        """Helper to format indicator values, handling NaN/None."""
        if value is None or pd.isna(value):
            return f"{NEON_YELLOW}N/A{RESET}"
        try:
            # Use Decimal for precise rounding if needed, but float is often fine for display
            return f"{float(value):.{precision}f}"
        except (ValueError, TypeError):
            return f"{NEON_YELLOW}Err{RESET}"

    def interpret_indicator(self, indicator_name: str, values: list) -> str:
        """Provides textual interpretation of the latest indicator value with color."""
        if values is None or not values or all(pd.isna(v) for v in values):
            return f"{indicator_name.upper():<15}: {NEON_YELLOW}No data{RESET}"

        # Use the most recent non-NaN value for interpretation
        last_valid_value = next((v for v in reversed(values) if pd.notna(v)), None)
        if last_valid_value is None:
             return f"{indicator_name.upper():<15}: {NEON_YELLOW}No valid recent data{RESET}"

        # Display last 1-3 values for context
        display_values = " | ".join(self._format_indicator_value(v) for v in values[-3:]) # Show last 3

        # --- Interpretations ---
        try:
            name_upper = indicator_name.upper()
            # Add specific formatting precision where needed
            precision = 4 if name_upper in ["ATR", "ISA", "ISB", "ITS", "IKS", "ICS", "PSAR"] else 2 # More precision for ATR/Ichimoku/PSAR
            last_val_fmt = self._format_indicator_value(last_valid_value, precision)
            display_values_fmt = " | ".join(self._format_indicator_value(v, precision) for v in values[-3:])

            prefix = f"{name_upper:<15}: {NEON_WHITE}{display_values_fmt}{RESET}"

            if name_upper == "RSI":
                if last_valid_value > 70: return f"{prefix} - {NEON_RED}Overbought (>70){RESET}"
                elif last_valid_value < 30: return f"{prefix} - {NEON_GREEN}Oversold (<30){RESET}"
                else: return f"{prefix} - {NEON_YELLOW}Neutral (30-70){RESET}"
            elif name_upper == "RSI_LONG":
                if last_valid_value > 65: return f"{prefix} - {NEON_RED}High Momentum (>65){RESET}" # Different thresholds for long RSI
                elif last_valid_value < 35: return f"{prefix} - {NEON_GREEN}Low Momentum (<35){RESET}"
                else: return f"{prefix} - {NEON_YELLOW}Neutral (35-65){RESET}"
            elif name_upper.startswith("STOCHRSI"): # Combined K and D
                 k, d = values[-1] if isinstance(values[-1], (list, tuple)) and len(values[-1])==2 else (last_valid_value, None)
                 k_fmt = self._format_indicator_value(k)
                 d_fmt = self._format_indicator_value(d)
                 prefix = f"{'STOCHRSI K/D':<15}: {NEON_WHITE}{k_fmt} / {d_fmt}{RESET}"
                 if k is None or d is None: return f"{prefix} - {NEON_YELLOW}Data Missing{RESET}"
                 cross = ""
                 if k > d and values[-2][0] <= values[-2][1]: cross = f" - {NEON_GREEN}Bullish Cross{RESET}"
                 elif k < d and values[-2][0] >= values[-2][1]: cross = f" - {NEON_RED}Bearish Cross{RESET}"
                 if k > 80: return f"{prefix} - {NEON_RED}Overbought (>80){cross}{RESET}"
                 elif k < 20: return f"{prefix} - {NEON_GREEN}Oversold (<20){cross}{RESET}"
                 else: return f"{prefix} - {NEON_YELLOW}Neutral (20-80){cross}{RESET}"

            elif name_upper.startswith("STOCH"): # Stochastic K/D
                 k, d = values[-1] if isinstance(values[-1], (list, tuple)) and len(values[-1])==2 else (last_valid_value, None)
                 k_fmt = self._format_indicator_value(k)
                 d_fmt = self._format_indicator_value(d)
                 prefix = f"{'STOCH K/D':<15}: {NEON_WHITE}{k_fmt} / {d_fmt}{RESET}"
                 if k is None or d is None: return f"{prefix} - {NEON_YELLOW}Data Missing{RESET}"
                 cross = ""
                 if k > d and values[-2][0] <= values[-2][1]: cross = f" - {NEON_GREEN}Bullish Cross{RESET}"
                 elif k < d and values[-2][0] >= values[-2][1]: cross = f" - {NEON_RED}Bearish Cross{RESET}"
                 if k > 80: return f"{prefix} - {NEON_RED}Overbought (>80){cross}{RESET}"
                 elif k < 20: return f"{prefix} - {NEON_GREEN}Oversold (<20){cross}{RESET}"
                 else: return f"{prefix} - {NEON_YELLOW}Neutral (20-80){cross}{RESET}"

            elif name_upper == "MACD": # Full MACD interpretation
                macd_line, hist, signal_line = values[-1] # Order from pandas_ta: MACD, Histogram, Signal
                macd_fmt = self._format_indicator_value(macd_line)
                hist_fmt = self._format_indicator_value(hist)
                sig_fmt = self._format_indicator_value(signal_line)
                prefix = f"{'MACD(M/H/S)':<15}: {NEON_WHITE}{macd_fmt} / {hist_fmt} / {sig_fmt}{RESET}"
                status = ""
                # Crossover Check
                prev_macd, prev_hist, prev_sig = values[-2]
                if pd.notna(prev_macd) and pd.notna(prev_sig):
                    if macd_line > signal_line and prev_macd <= prev_sig: status += f" - {NEON_GREEN}Bull Cross{RESET}"
                    elif macd_line < signal_line and prev_macd >= prev_sig: status += f" - {NEON_RED}Bear Cross{RESET}"
                # Histogram Check
                if hist > 0: status += f" - {NEON_GREEN}Hist Positive{RESET}"
                elif hist < 0: status += f" - {NEON_RED}Hist Negative{RESET}"
                # Divergence (Simple Check - requires price data) - TODO: Implement properly if needed
                # divergence = self.detect_macd_divergence() ...
                return f"{prefix}{status if status else ' - Neutral'}"

            elif name_upper == "MFI":
                if last_valid_value > 80: return f"{prefix} - {NEON_RED}Overbought (>80){RESET}"
                elif last_valid_value < 20: return f"{prefix} - {NEON_GREEN}Oversold (<20){RESET}"
                else: return f"{prefix} - {NEON_YELLOW}Neutral (20-80){RESET}"
            elif name_upper == "CCI":
                if last_valid_value > 100: return f"{prefix} - {NEON_RED}Overbought (>100){RESET}"
                elif last_valid_value < -100: return f"{prefix} - {NEON_GREEN}Oversold (<-100){RESET}"
                else: return f"{prefix} - {NEON_YELLOW}Neutral (-100 to 100){RESET}"
            elif name_upper == "WILLR": # Williams %R
                if last_valid_value > -20: return f"{prefix} - {NEON_RED}Overbought (>-20){RESET}"
                elif last_valid_value < -80: return f"{prefix} - {NEON_GREEN}Oversold (<-80){RESET}"
                else: return f"{prefix} - {NEON_YELLOW}Neutral (-80 to -20){RESET}"
            elif name_upper == "ADX":
                adx_val, dmp, dmn = values[-1] # ADX, +DI, -DI
                adx_fmt = self._format_indicator_value(adx_val)
                dmp_fmt = self._format_indicator_value(dmp)
                dmn_fmt = self._format_indicator_value(dmn)
                prefix = f"{'ADX(+DI/-DI)':<15}: {NEON_WHITE}{adx_fmt} ({dmp_fmt}/{dmn_fmt}){RESET}"
                trend_status = ""
                if adx_val > 25: trend_status = f"{NEON_GREEN}Trending{RESET}"
                elif adx_val < 20: trend_status = f"{NEON_YELLOW}Weak/Ranging{RESET}"
                else: trend_status = f"{NEON_YELLOW}Developing{RESET}"
                direction = ""
                if dmp > dmn: direction = f" ({NEON_GREEN}Bullish Bias{RESET})"
                elif dmn > dmp: direction = f" ({NEON_RED}Bearish Bias{RESET})"
                return f"{prefix} - {trend_status}{direction}"
            elif name_upper == "OBV":
                trend = f"{NEON_YELLOW}Neutral{RESET}"
                if len(values) > 1 and pd.notna(values[-2]):
                     if last_valid_value > values[-2]: trend = f"{NEON_GREEN}Rising (Bullish){RESET}"
                     elif last_valid_value < values[-2]: trend = f"{NEON_RED}Falling (Bearish){RESET}"
                return f"{prefix} - {trend}"
            elif name_upper == "CMF": # Chaikin Money Flow
                 if last_valid_value > 0.05: return f"{prefix} - {NEON_GREEN}Bullish (>+0.05){RESET}" # Common thresholds
                 elif last_valid_value < -0.05: return f"{prefix} - {NEON_RED}Bearish (<-0.05){RESET}"
                 else: return f"{prefix} - {NEON_YELLOW}Neutral (-0.05 to +0.05){RESET}"
            elif name_upper == "AO": # Awesome Oscillator
                prefix = f"{'AO':<15}: {NEON_WHITE}{last_val_fmt}{RESET}"
                color = NEON_GREEN if last_valid_value > 0 else NEON_RED
                trend = ""
                if len(values) > 1 and pd.notna(values[-2]):
                    if last_valid_value > 0 and values[-2] < 0: trend = f" - {NEON_GREEN}Bullish Zero Cross{RESET}"
                    elif last_valid_value < 0 and values[-2] > 0: trend = f" - {NEON_RED}Bearish Zero Cross{RESET}"
                    elif last_valid_value > values[-2]: trend += f" - {color}Increasing{RESET}"
                    elif last_valid_value < values[-2]: trend += f" - {color}Decreasing{RESET}"
                return f"{prefix}{trend if trend else ' - Neutral'}"

            # --- Non-Interpreted Values (just display) ---
            elif name_upper == "ATR":
                 return f"{'ATR':<15}: {NEON_WHITE}{last_val_fmt}{RESET} (Avg Volatility)"
            elif name_upper == "MOMENTUM":
                 color = NEON_GREEN if last_valid_value > 0 else NEON_RED if last_valid_value < 0 else NEON_YELLOW
                 return f"{prefix} - {color}{'Positive' if last_valid_value > 0 else 'Negative' if last_valid_value < 0 else 'Zero'}{RESET}"
            elif name_upper == "VWAP":
                price = self.indicators.get('close', [None])[-1]
                if pd.notna(price) and pd.notna(last_valid_value):
                    relation = f"{NEON_GREEN}Above{RESET}" if price > last_valid_value else f"{NEON_RED}Below{RESET}" if price < last_valid_value else f"{NEON_YELLOW}At{RESET}"
                    return f"{prefix} - Price {relation} VWAP"
                else:
                    return f"{prefix} - {NEON_YELLOW}Price/VWAP N/A{RESET}"
            elif name_upper.startswith("SMA") or name_upper.startswith("EMA"):
                # Simple display, potential trend indication could be added
                price = self.indicators.get('close', [None])[-1]
                relation = ""
                if pd.notna(price) and pd.notna(last_valid_value):
                     relation = f" ({NEON_GREEN}Price Above{RESET})" if price > last_valid_value else f" ({NEON_RED}Price Below{RESET})" if price < last_valid_value else f" ({NEON_YELLOW}Price At MA{RESET})"
                return f"{prefix}{relation}"
            elif name_upper.startswith("BB"): # Bollinger Bands (BBM middle band shown)
                bbl, bbm, bbu, bbb, bbp = values[-1]
                bbl_f = self._format_indicator_value(bbl)
                bbm_f = self._format_indicator_value(bbm)
                bbu_f = self._format_indicator_value(bbu)
                bbp_f = self._format_indicator_value(bbp, 3) # %B higher precision
                prefix = f"{'BBands(L/M/U)':<15}: {NEON_WHITE}{bbl_f}/{bbm_f}/{bbu_f}{RESET}"
                price = self.indicators.get('close', [None])[-1]
                status = ""
                if pd.notna(price):
                    if price > bbu: status = f" - {NEON_RED}Price Above Upper{RESET}"
                    elif price < bbl: status = f" - {NEON_GREEN}Price Below Lower{RESET}"
                    else: status = f" - {NEON_YELLOW}Price Within Bands{RESET}"
                if pd.notna(bbp): status += f" (%B: {bbp_f})"
                return f"{prefix}{status}"

            elif name_upper.startswith("I"): # Ichimoku - Just display values for now
                isa, isb, its, iks, ics = values[-1] # SenkouA, SenkouB, Tenkan, Kijun, Chikou
                isa_f = self._format_indicator_value(isa, 4)
                isb_f = self._format_indicator_value(isb, 4)
                its_f = self._format_indicator_value(its, 4)
                iks_f = self._format_indicator_value(iks, 4)
                ics_f = self._format_indicator_value(ics, 4) # Chikou is price shifted back
                # Basic Cloud relation
                price = self.indicators.get('close', [None])[-1]
                cloud_status = ""
                if pd.notna(price) and pd.notna(isa) and pd.notna(isb):
                    if price > max(isa, isb): cloud_status = f" ({NEON_GREEN}Above Cloud{RESET})"
                    elif price < min(isa, isb): cloud_status = f" ({NEON_RED}Below Cloud{RESET})"
                    else: cloud_status = f" ({NEON_YELLOW}Inside Cloud{RESET})"

                return f"{'Ichimoku Cloud':<15}: A:{isa_f} B:{isb_f} T:{its_f} K:{iks_f} C:{ics_f}{cloud_status}"


            # Default return if no specific interpretation matches
            return f"{name_upper:<15}: {NEON_WHITE}{display_values_fmt}{RESET}"

        except (TypeError, IndexError, KeyError) as e:
            self.logger.error(f"{NEON_RED}{BRIGHT}Error interpreting {indicator_name}:{RESET} {e} - Values: {values}")
            return f"{indicator_name.upper():<15}: {NEON_RED}Interpretation Error{RESET}"
        except Exception as e:
             self.logger.exception(f"{NEON_RED}{BRIGHT}Unexpected error interpreting {indicator_name}:{RESET} {e}")
             return f"{indicator_name.upper():<15}: {NEON_RED}Unexpected Error{RESET}"


    def analyze_orderbook_levels(self, orderbook: dict, current_price: Decimal) -> str:
        """Analyzes order book for clusters near calculated S/R levels."""
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return f"  {NEON_YELLOW}Orderbook data not available or invalid.{RESET}"

        try:
            # Ensure data is usable (list of lists/tuples [price, size])
            bids_raw = orderbook.get('bids', [])
            asks_raw = orderbook.get('asks', [])
            if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                 self.logger.error(f"{NEON_RED}Orderbook bids/asks are not lists.{RESET}")
                 return f"  {NEON_RED}Invalid orderbook format.{RESET}"

            # Convert to DataFrame for easier filtering, handle potential errors
            try:
                 bids = pd.DataFrame(bids_raw, columns=["price", "size"], dtype=float)
                 asks = pd.DataFrame(asks_raw, columns=["price", "size"], dtype=float)
            except ValueError as ve:
                 self.logger.error(f"{NEON_RED}Could not convert orderbook data to DataFrame: {ve}{RESET}")
                 return f"  {NEON_RED}Error processing orderbook data.{RESET}"


            analysis_output = []
            cluster_threshold = Decimal(self.config.get("orderbook_cluster_threshold", 1000))
            price_proximity_percent = Decimal("0.001") # Check within 0.1% of the level price

            def check_cluster_at_level(level_label, level_price_float, bids_df, asks_df):
                level_price = Decimal(str(level_price_float)) # Use Decimal for comparisons
                price_delta = level_price * price_proximity_percent
                lower_bound = level_price - price_delta
                upper_bound = level_price + price_delta

                # Filter bids/asks near the level
                nearby_bids = bids_df[(bids_df["price"] >= float(lower_bound)) & (bids_df["price"] <= float(upper_bound))]
                nearby_asks = asks_df[(asks_df["price"] >= float(lower_bound)) & (asks_df["price"] <= float(upper_bound))]

                bid_volume_near_level = Decimal(str(nearby_bids["size"].sum()))
                ask_volume_near_level = Decimal(str(nearby_asks["size"].sum()))

                results = []
                if bid_volume_near_level > cluster_threshold:
                    results.append(f"  {NEON_GREEN}Bid Cluster{RESET} ({bid_volume_near_level:,.0f}) near {NEON_YELLOW}{level_label}{RESET} @ ${level_price:.4f}")
                if ask_volume_near_level > cluster_threshold:
                    results.append(f"  {NEON_RED}Ask Cluster{RESET} ({ask_volume_near_level:,.0f}) near {NEON_YELLOW}{level_label}{RESET} @ ${level_price:.4f}")
                return results

            processed_levels = set() # Avoid checking the exact same price level twice

            # Iterate through combined S/R levels
            combined_levels_list = []
            for level_type, type_levels in self.levels.items():
                 if isinstance(type_levels, dict):
                     for label, value in type_levels.items():
                         if isinstance(value, (float, Decimal, int)):
                             combined_levels_list.append((label, float(value)))
                 elif isinstance(type_levels, (float, Decimal, int)): # Handle top-level (e.g., Pivot Point)
                     combined_levels_list.append((level_type, float(type_levels)))


            # Sort levels by price for potentially clearer output (optional)
            combined_levels_list.sort(key=lambda x: x[1])

            for label, value in combined_levels_list:
                 if round(value, 5) not in processed_levels: # Check rounded value to avoid floating point issues
                     cluster_results = check_cluster_at_level(label, value, bids, asks)
                     analysis_output.extend(cluster_results)
                     processed_levels.add(round(value, 5))


            if not analysis_output:
                return f"  {NEON_YELLOW}No significant orderbook clusters detected near calculated S/R levels.{RESET}"

            return "\n".join(analysis_output)

        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Error analyzing orderbook levels:{RESET} {e}")
            return f"  {NEON_RED}Error during orderbook analysis.{RESET}"


    def approximate_volume_profile(self, num_bins: int) -> str:
        """Creates a simple text-based volume profile histogram from kline data."""
        if 'full_df' not in self.indicators or self.indicators['full_df'].empty:
            return f"  {NEON_YELLOW}Volume profile requires kline data.{RESET}"
        df = self.indicators['full_df']
        try:
            low_price = df['low'].min()
            high_price = df['high'].max()
            if pd.isna(low_price) or pd.isna(high_price) or low_price == high_price:
                return f"  {NEON_YELLOW}Insufficient price range for volume profile.{RESET}"

            price_range = high_price - low_price
            bin_size = price_range / num_bins
            if bin_size <= 0:
                 return f"  {NEON_YELLOW}Invalid bin size for volume profile.{RESET}"

            bins = np.arange(low_price, high_price + bin_size, bin_size)
            df['price_bin'] = pd.cut(df['close'], bins=bins, labels=bins[:-1], right=False)

            # Group by price bin and sum volume
            profile = df.groupby('price_bin')['volume'].sum().sort_index(ascending=False) # Display high to low

            max_volume = profile.max()
            if max_volume == 0:
                return f"  {NEON_YELLOW}No volume data for profile.{RESET}"

            # Create histogram string
            profile_str = f"  {NEON_BLUE}{BRIGHT}Volume Profile (Approx. {num_bins} bins):{RESET}\n"
            poc_level = profile.idxmax() # Price level with highest volume (Point of Control)
            poc_volume = profile.max()

            # Normalize bar width for terminal display (e.g., max 50 chars)
            max_bar_width = 50
            for price, volume in profile.items():
                bar_width = int((volume / max_volume) * max_bar_width) if max_volume > 0 else 0
                bar = '#' * bar_width
                price_fmt = f"{price:.4f}" # Format price level
                vol_fmt = f"{volume:,.0f}"   # Format volume
                is_poc = price == poc_level
                line_color = NEON_GREEN if is_poc else NEON_WHITE
                poc_indicator = f"{NEON_GREEN}{BRIGHT} <POC>{RESET}" if is_poc else ""
                profile_str += f"  {line_color}{price_fmt:<10}{RESET} | {NEON_YELLOW}{bar:<{max_bar_width}}{RESET} | {NEON_BLUE}{vol_fmt}{poc_indicator}{RESET}\n"

            profile_str += f"  {NEON_GREEN}Point of Control (POC): ~${poc_level:.4f} (Vol: {poc_volume:,.0f}){RESET}"
            return profile_str.strip()

        except Exception as e:
            self.logger.exception(f"{NEON_RED}{BRIGHT}Error generating volume profile:{RESET} {e}")
            return f"  {NEON_RED}Error during volume profile generation.{RESET}"


    def generate_analysis_output(self, current_price: Decimal, timestamp_str: str) -> str:
        """Generates the full formatted analysis output string."""

        if not self.indicators or 'close' not in self.indicators or not self.indicators['close']:
             return f"{NEON_RED}{BRIGHT}Insufficient data for analysis.{RESET}"

        # --- Prepare Data ---
        last_close = self.indicators['close'][-1]
        last_volume = self.indicators['volume'][-1]
        last_vol_ma = self.indicators.get('VOL_MA', [None])[-1]
        last_vol_spike = self.indicators.get('Vol_Spike', [False])[-1]
        last_vol_trend_up = self.indicators.get('Vol_Trend_Up', [False])[-1]

        # Calculate S/R levels based on recent data range
        df_recent = self.indicators.get('full_df', pd.DataFrame())
        if not df_recent.empty:
            recent_high = df_recent['high'].max()
            recent_low = df_recent['low'].min()
            self.calculate_fibonacci_retracement(recent_high, recent_low, float(current_price))
            self.calculate_pivot_points(df_recent['high'].iloc[-1], df_recent['low'].iloc[-1], df_recent['close'].iloc[-1]) # Use last candle for pivots
        else:
             self.logger.warning(f"{NEON_YELLOW}Full DataFrame missing, cannot calculate Fib/Pivots accurately.{RESET}")


        nearest_supports, nearest_resistances = self.find_nearest_levels(float(current_price))

        # --- Build Output String ---
        output = f"\n--- {NEON_PURPLE}{BRIGHT}Neonta Analysis{RESET} --- {NEON_BLUE}({self.symbol} @ {self.interval}){RESET} ---\n"
        output += f"{NEON_BLUE}{BRIGHT}{'Timestamp:':<15}{RESET} {NEON_WHITE}{timestamp_str}{RESET}\n"
        output += f"{NEON_BLUE}{BRIGHT}{'Current Price:':<15}{RESET} {NEON_WHITE}{BRIGHT}${current_price:.4f}{RESET}\n" # More precision for price
        output += f"{NEON_BLUE}{BRIGHT}{'Last Close:':<15}{RESET} {NEON_WHITE}{self._format_indicator_value(last_close, 4)}{RESET}\n"
        output += f"{NEON_BLUE}{BRIGHT}{'Last Volume:':<15}{RESET} {NEON_WHITE}{self._format_indicator_value(last_volume, 0)}{RESET}\n"

        # --- Section: Key Stats ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Key Stats ---{RESET}\n"
        output += self.interpret_indicator("ATR", self.indicators.get('ATR', [None])) + "\n"
        output += self.interpret_indicator("VWAP", self.indicators.get('VWAP', [None])) + "\n"
        # Display key MAs
        output += self.interpret_indicator("SMA_10", self.indicators.get('SMA_10', [None])) + "\n"
        output += self.interpret_indicator("EMA_12", self.indicators.get('EMA_12', [None])) + "\n"
        output += self.interpret_indicator("EMA_26", self.indicators.get('EMA_26', [None])) + "\n"
        output += self.interpret_indicator("SMA_50", self.indicators.get('SMA_50', [None])) + "\n"
        output += self.interpret_indicator("EMA_50", self.indicators.get('EMA_50', [None])) + "\n"
        output += self.interpret_indicator("SMA_200", self.indicators.get('SMA_200', [None])) + "\n"
        output += self.interpret_indicator("EMA_200", self.indicators.get('EMA_200', [None])) + "\n"


        # --- Section: Oscillators ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Oscillators ---{RESET}\n"
        output += self.interpret_indicator("RSI", self.indicators.get('RSI', [None])) + "\n"
        output += self.interpret_indicator("RSI_long", self.indicators.get('RSI_long', [None])) + "\n"
        # Combine Stoch RSI K/D display
        stoch_rsi_k = self.indicators.get('STOCHRSIk', [None])
        stoch_rsi_d = self.indicators.get('STOCHRSId', [None])
        if stoch_rsi_k and stoch_rsi_d:
            output += self.interpret_indicator("STOCHRSI K/D", list(zip(stoch_rsi_k, stoch_rsi_d))) + "\n"
        # Combine Stoch K/D display
        stoch_k = self.indicators.get('STOCHk', [None])
        stoch_d = self.indicators.get('STOCHd', [None])
        if stoch_k and stoch_d:
             output += self.interpret_indicator("STOCH K/D", list(zip(stoch_k, stoch_d))) + "\n"
        output += self.interpret_indicator("CCI", self.indicators.get('CCI', [None])) + "\n"
        output += self.interpret_indicator("MFI", self.indicators.get('MFI', [None])) + "\n"
        output += self.interpret_indicator("WILLR", self.indicators.get('WILLR', [None])) + "\n"

        # --- Section: Volume Analysis ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Volume Analysis ---{RESET}\n"
        vol_ma_str = self._format_indicator_value(last_vol_ma, 0)
        vol_vs_ma = ""
        if pd.notna(last_volume) and pd.notna(last_vol_ma):
             if last_volume > last_vol_ma:
                 vol_vs_ma = f"({NEON_GREEN}Above MA: {vol_ma_str}{RESET})"
             else:
                 vol_vs_ma = f"({NEON_RED}Below MA: {vol_ma_str}{RESET})"
        output += f"{'Volume MA Comp':<15}: {NEON_WHITE}{self._format_indicator_value(last_volume, 0)}{RESET} {vol_vs_ma}\n"
        output += f"{'Volume Trend':<15}: {NEON_GREEN}Increasing{RESET}" if last_vol_trend_up else f"{NEON_RED}Decreasing{RESET}" if pd.notna(last_vol_trend_up) else f"{NEON_YELLOW}N/A{RESET}"
        if last_vol_spike:
            output += f" {NEON_RED}{BRIGHT}** Volume Spike Detected **{RESET}"
        output += "\n"
        output += self.interpret_indicator("OBV", self.indicators.get('OBV', [None])) + "\n"
        output += self.interpret_indicator("CMF", self.indicators.get('CMF', [None])) + "\n"

        # --- Section: Trend & Momentum ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Trend & Momentum ---{RESET}\n"
        output += self.interpret_indicator("ADX", list(zip(self.indicators.get('ADX', [None]), self.indicators.get('DMP', [None]), self.indicators.get('DMN', [None])))) + "\n"
        # Combine MACD components
        macd_m = self.indicators.get('MACD', [None])
        macd_h = self.indicators.get('MACDh', [None])
        macd_s = self.indicators.get('MACDs', [None])
        if macd_m and macd_h and macd_s:
            output += self.interpret_indicator("MACD", list(zip(macd_m, macd_h, macd_s))) + "\n"
        output += self.interpret_indicator("AO", self.indicators.get('AO', [None])) + "\n"
        output += self.interpret_indicator("Momentum", self.indicators.get('Momentum', [None])) + "\n"
        # Ichimoku (basic interpretation or just values)
        ichi_a = self.indicators.get('ISA', [None])
        ichi_b = self.indicators.get('ISB', [None])
        ichi_t = self.indicators.get('ITS', [None])
        ichi_k = self.indicators.get('IKS', [None])
        ichi_c = self.indicators.get('ICS', [None])
        if ichi_a and ichi_b and ichi_t and ichi_k and ichi_c:
            output += self.interpret_indicator("ICHIMOKU", list(zip(ichi_a, ichi_b, ichi_t, ichi_k, ichi_c))) + "\n"


        # --- Section: Levels ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Support & Resistance Levels ---{RESET}\n"
        if nearest_supports:
            output += f"{NEON_GREEN}Nearest Support:{RESET}\n"
            for s_label, s_val in nearest_supports:
                output += f"  - {s_label:<20}: ${s_val:.4f}\n"
        else:
            output += f"  {NEON_YELLOW}No distinct support levels found nearby.{RESET}\n"

        if nearest_resistances:
            output += f"{NEON_RED}Nearest Resistance:{RESET}\n"
            for r_label, r_val in nearest_resistances:
                output += f"  - {r_label:<20}: ${r_val:.4f}\n"
        else:
             output += f"  {NEON_YELLOW}No distinct resistance levels found nearby.{RESET}\n"


        # --- Section: Order Book ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Order Book Analysis ---{RESET}\n"
        orderbook_data = fetch_orderbook(self.symbol, self.config["orderbook_limit"], self.logger)
        orderbook_analysis_str = self.analyze_orderbook_levels(orderbook_data, current_price)
        output += orderbook_analysis_str + "\n"

        # --- Section: Volume Profile ---
        output += f"\n{NEON_PURPLE}{BRIGHT}--- Volume Profile (Approx. from Kline Data) ---{RESET}\n"
        vp_str = self.approximate_volume_profile(self.config['volume_profile_bins'])
        output += vp_str + "\n"


        output += f"\n--- {NEON_PURPLE}{BRIGHT}End of Analysis{RESET} ---\n"
        self.last_analysis_output = output # Store for potential reuse/diff
        return output


    def analyze(self, current_price: Decimal, timestamp_str: str):
        """Performs the analysis and prints the formatted output."""
        # Most calculations are now done in __init__ or helper methods
        # Ensure indicators were calculated
        if not self.indicators or 'close' not in self.indicators:
             self.logger.error(f"{NEON_RED}{BRIGHT}Indicators not calculated, cannot perform analysis.{RESET}")
             print(f"{NEON_RED}{BRIGHT}Error: Analysis aborted due to missing indicator data.{RESET}")
             return

        # Generate the formatted output
        analysis_output_str = self.generate_analysis_output(current_price, timestamp_str)

        # Print to console
        print(analysis_output_str)

        # Log the raw (uncolored) output or a summary
        # For cleaner logs, remove color codes before logging to file
        # clean_output = re.sub(r'\x1b\[[0-9;]*m', '', analysis_output_str)
        # self.logger.info(f"Analysis Results:\n{clean_output}")
        # Or log just key values:
        self.logger.info(f"Analysis completed for {self.symbol} at {timestamp_str}. Price: {current_price:.4f}, RSI: {self._format_indicator_value(self.indicators.get('RSI', [None])[-1])}")


# --- Main Execution Logic ---

def main():
    """Main function to run the analysis loop."""
    symbol = ""
    while True:
        prompt = f"{NEON_BLUE}{BRIGHT}Enter trading symbol (e.g., BTCUSDT): {RESET}"
        symbol_input = input(prompt).upper().strip()
        if symbol_input:
            symbol = symbol_input
            break
        print(f"{NEON_RED}Symbol cannot be empty. Please try again.{RESET}")

    interval = ""
    while True:
        valid_intervals_str = ', '.join(VALID_INTERVALS)
        prompt = f"{NEON_BLUE}{BRIGHT}Enter timeframe [{valid_intervals_str}]: {RESET}"
        interval_input = input(prompt).strip()
        if not interval_input:
            interval = CONFIG.get("interval", "15") # Default from config
            print(f"{NEON_YELLOW}No interval provided. Using default: {interval}{RESET}")
            break
        if interval_input in VALID_INTERVALS:
            interval = interval_input
            break
        print(f"{NEON_RED}Invalid interval '{interval_input}'. Please choose from the list.{RESET}")

    logger = setup_logger(symbol)
    logger.info(f"{NEON_GREEN}{BRIGHT}--- Neonta Analysis Bot Starting ---{RESET}")
    logger.info(f"Symbol: {symbol}, Interval: {interval}")
    logger.info(f"Using config: {CONFIG_FILE}")

    analysis_interval_seconds = CONFIG.get("analysis_interval", 30)
    retry_delay = CONFIG.get("retry_delay", 5)
    kline_limit = 250 # Fetch slightly more data for indicator calculations (e.g., for 200 EMA)

    # Create persistent session
    api_session = create_session()

    while True:
        try:
            start_time = time.time()
            now_local = datetime.now(TIMEZONE)
            timestamp_log_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
            logger.info(f"{NEON_BLUE}--- Starting Analysis Cycle at {timestamp_log_str} ---{RESET}")

            # 1. Fetch Current Price
            current_price = fetch_current_price(symbol, logger, api_session)
            if current_price is None:
                logger.error(f"{NEON_RED}{BRIGHT}Failed to fetch current price. Retrying...{RESET}")
                time.sleep(retry_delay)
                continue # Skip this cycle

            # 2. Fetch Kline Data
            df_klines = fetch_klines(symbol, interval, limit=kline_limit, logger=logger, session=api_session)
            if df_klines.empty:
                logger.error(f"{NEON_RED}{BRIGHT}Failed to fetch kline data. Retrying...{RESET}")
                time.sleep(retry_delay)
                continue # Skip this cycle

            # 3. Perform Analysis
            try:
                analyzer = TradingAnalyzer(df_klines, logger, CONFIG, symbol, interval)
                analyzer.analyze(current_price, timestamp_log_str)
            except ValueError as ve: # Catch init errors in Analyzer (like empty DF)
                 logger.error(f"{NEON_RED}{BRIGHT}Analysis skipped due to error: {ve}{RESET}")
                 time.sleep(retry_delay)
                 continue


            # 4. Wait for the next cycle
            elapsed_time = time.time() - start_time
            wait_time = max(0, analysis_interval_seconds - elapsed_time)
            logger.info(f"Analysis cycle took {elapsed_time:.2f}s. Waiting {wait_time:.2f}s for next cycle.")
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            logger.error(f"{NEON_RED}{BRIGHT}Network error occurred in main loop:{RESET} {e}. Retrying after {retry_delay}s...")
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            logger.info(f"{NEON_YELLOW}{BRIGHT}Ctrl+C detected. Stopping analysis bot gracefully...{RESET}")
            print(f"\n{NEON_YELLOW}Analysis stopped by user.{RESET}")
            break
        except Exception as e:
            logger.exception(f"{NEON_RED}{BRIGHT}An unexpected error occurred in the main loop:{RESET} {e}. Retrying after {retry_delay}s...")
            time.sleep(retry_delay) # Wait before retrying on unexpected errors


if __name__ == "__main__":
    main()
```

**Explanation of Key Changes:**

1.  **`pandas_ta` Integration:** The core change is using `df.ta.strategy()` within `_calculate_all_indicators`. This function efficiently calculates multiple indicators defined in the `ta.Strategy` object and adds them directly as columns to the DataFrame. This significantly simplifies and standardizes indicator calculation.
2.  **New Indicator Calculations:** VWAP, CMF, Ichimoku, Stochastic, and AO are now calculated via `pandas_ta`. Their parameters are fetched from the `CONFIG`.
3.  **Configuration Updates:** `load_config` now includes default settings for the new indicators and slightly more robust merging logic.
4.  **Volume Analysis Enhancements:**
    *   `VOL_MA` is calculated using `pandas_ta`.
    *   `Vol_Spike` boolean column identifies candles where volume > `VOL_MA * volume_spike_multiplier`.
    *   `Vol_Trend_Up` boolean column shows if volume increased from the previous candle.
    *   `approximate_volume_profile` function creates a text histogram.
    *   The output section explicitly shows volume vs. MA, trend, and spikes.
5.  **Refactored `TradingAnalyzer`:**
    *   Calculations are centralized in `_calculate_all_indicators`.
    *   `analyze` method now focuses on orchestrating the analysis flow and calling the `generate_analysis_output` method.
    *   `interpret_indicator` handles interpretation logic for *all* indicators, including the new ones, providing more context and better colorization.
    *   `generate_analysis_output` structures the final text output into clear, color-coded sections.
6.  **Output Formatting (`generate_analysis_output` & `interpret_indicator`):**
    *   Uses f-strings extensively for cleaner formatting.
    *   Employs `NEON_` colors and `BRIGHT` style more effectively.
    *   Uses fixed-width formatting (`{:<15}`) for better alignment of indicator names.
    *   Includes actual indicator values alongside interpretations.
    *   Separated into logical sections using headers.
7.  **Error Handling and Logging:**
    *   More specific error messages with colors.
    *   Improved logging format and added UTC timezone info.
    *   Handles potential `NaN` values more gracefully during interpretation and formatting.
    *   `fetch_orderbook` uses `fetch_l2_order_book` from `ccxt`.
    *   API request/response errors are logged more clearly.
8.  **API Request Signing:** Updated `bybit_request` signature generation to match Bybit V5 API requirements more closely (using `recv_window`, handling GET/POST parameter string construction).
9.  **Main Loop:** Uses a persistent `requests.Session` for potentially better performance and connection reuse.

Remember to install `pandas_ta` and `pytz`, ensure your `.env` file is correct, and let the digital currents guide your analysis!
