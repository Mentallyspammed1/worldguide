```python
# livex.py
# Enhanced version of lwx.py focusing on robust order placement and TSL integration.

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
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # CCXT uses '1m', '5m' etc. Need mapping later
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # Add relevant error codes
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
        "signal_score_threshold": 1.5, # Lowered slightly from scalping
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,
        "scalping_signal_threshold": 2.5, # Keep separate threshold for scalping preset
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": True, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": False,     # SAFETY FIRST: Default to True, disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage
        "max_concurrent_positions": 1, # Limit open positions for this symbol (common strategy)
        "quote_currency": "USDT", # Currency for balance check and sizing
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL
        "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance (as decimal)
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry
        # --- End Trailing Stop Loss Config ---
        "indicators": { # Control which indicators are calculated and used
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
        },
        "weight_sets": { # Define different weighting strategies
            "scalping": { # Example weighting for a scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2,
            },
            "default": { # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2,
            }
        },
        "active_weight_set": "default" # Choose which weight set to use
    }

    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added
            if updated_config != config_from_file:
                 with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(updated_config, f, indent=4)
                 print(f"{NEON_YELLOW}Updated config file with default keys: {filepath}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config: {e}. Using default config.{RESET}")
        # Create default if loading failed badly
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensure all keys from default_config are in config."""
    updated_config = config.copy()
    for key, value in default_config.items():
        if key not in updated_config:
            updated_config[key] = value
        elif isinstance(value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], value)
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    logger_name = f"livex_bot_{symbol.replace('/','_').replace(':','-')}" # Use symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO) # Set base level to INFO

    # File Handler (writes INFO and above)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO) # Log INFO and higher to file
    logger.addHandler(file_handler)

    # Stream Handler (console, writes INFO and above by default)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO) # Log INFO and higher to console
    logger.addHandler(stream_handler)

    # Set propagate to False to prevent logs going to the root logger if it's configured
    logger.propagate = False

    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT margined)
                'adjustForTimeDifference': True,
                # Bybit V5 Unified Margin Account setting
                # Check your account type on Bybit website. Unified is common now.
                # 'brokerId': 'YOUR_BROKER_ID', # If using via a broker program
                # 'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress warning if needed
                # Test connection on initialization
                'fetchTickerTimeout': 10000, # 10 seconds timeout
                'fetchBalanceTimeout': 15000, # 15 seconds timeout
            }
        }
        # Handle potential sandbox URL override from .env
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE{RESET}")
            # Some exchanges have specific sandbox URLs, ccxt handles Bybit via set_sandbox_mode
            # exchange_options['urls'] = {'api': 'https://api-testnet.bybit.com'} # Manual override if needed

        exchange = ccxt.bybit(exchange_options)

        if CONFIG.get('use_sandbox'):
            exchange.set_sandbox_mode(True)

        # Test connection by fetching markets
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Optional: Test API credentials by fetching balance
        try:
            balance = exchange.fetch_balance({'type': 'swap'}) # Try fetching swap balance
            logger.info(f"Successfully connected and fetched balance (Example: {QUOTE_CURRENCY} available: {balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')})")
        except ccxt.AuthenticationError as auth_err:
             logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Trade, Read), and IP whitelist is set if required.{RESET}")
             return None # Exit if authentication fails early
        except Exception as balance_err:
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check permissions if trading fails.{RESET}")

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
    """Fetch the current price of a trading symbol using CCXT ticker."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        # Prefer 'last' price, fallback to 'bid' or 'ask' if 'last' is unavailable/stale
        price = ticker.get('last')
        if price is None:
            logger.warning(f"Ticker 'last' price unavailable for {symbol}. Trying bid/ask average.")
            bid = ticker.get('bid')
            ask = ticker.get('ask')
            if bid and ask:
                price = (Decimal(str(bid)) + Decimal(str(ask))) / 2
            elif ask: # Only ask available
                price = ask
                logger.warning(f"Only ask price ({ask}) available for {symbol}.")
            elif bid: # Only bid available
                price = bid
                logger.warning(f"Only bid price ({bid}) available for {symbol}.")

        if price is not None:
            return Decimal(str(price))
        else:
            logger.error(f"{NEON_RED}Failed to fetch current price for {symbol} via ticker (last/bid/ask). Ticker: {ticker}{RESET}")
            # Optional: Try fetch_trades as a last resort (less ideal for current price)
            # trades = exchange.fetch_trades(symbol, limit=1)
            # if trades: return Decimal(str(trades[0]['price']))
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
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    try:
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
             return pd.DataFrame()

        # Fetch data with retries for network issues
        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                  break # Success
             except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"Max retries reached fetching klines for {symbol} after network errors.")
                      raise e # Re-raise the last error
             except ccxt.ExchangeError as e:
                 # Some exchange errors might be retryable (e.g., rate limits if not handled by ccxt)
                 # Check e.http_status or specific error messages if needed
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 raise e # Re-raise non-network errors immediately

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Ensure correct dtypes for pandas_ta and calculations
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True) # Drop rows where essential price data is missing

        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/dropna.{RESET}")

        lg.debug(f"Fetched {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e: # Catch error if retries fail
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after retries: {e}{RESET}")
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
            if not exchange.has['fetchOrderBook']:
                 logger.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                 return None
            orderbook = exchange.fetch_order_book(symbol, limit=limit)
            # Basic validation
            if orderbook and isinstance(orderbook.get('bids'), list) and isinstance(orderbook.get('asks'), list):
                return orderbook
            else:
                logger.warning(f"{NEON_YELLOW}Empty or invalid orderbook response for {symbol}. Attempt {attempts + 1}/{MAX_API_RETRIES + 1}. Response: {orderbook}{RESET}")

        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")

        except ccxt.ExchangeError as e:
            logger.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            # Don't retry on definitive exchange errors (e.g., bad symbol) unless specifically handled
            return None
        except Exception as e:
            logger.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None

# --- Trading Analyzer Class (Using pandas_ta) ---
class TradingAnalyzer:
    """Analyze trading data and generate signals using pandas_ta."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass market info for precision etc.
    ) -> None:
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN')
        self.interval = config.get("interval", "UNKNOWN")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN") # Map to ccxt format
        self.indicator_values: Dict[str, float] = {} # Stores latest indicator values
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0} # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config["weight_sets"].get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels
        self.ta_column_names = {} # Stores actual column names generated by pandas_ta

        if not self.weights:
             logger.error(f"Active weight set '{self.active_weight_set_name}' not found in config. Using empty weights.")

        self._calculate_all_indicators() # Calculate indicators on initialization

    def _get_ta_col_name(self, base_name: str, *args) -> str:
        """Helper to construct the default column name used by pandas_ta. (May need adjustment based on pandas_ta version)"""
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

        # Check for sufficient data (crude check, can be improved)
        # Collect periods from config, using defaults if missing
        periods_needed = []
        cfg = self.config
        periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD))
        periods_needed.append(cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        periods_needed.append(cfg.get("rsi_period", DEFAULT_RSI_WINDOW))
        periods_needed.append(cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        periods_needed.append(cfg.get("cci_window", DEFAULT_CCI_WINDOW))
        periods_needed.append(cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        periods_needed.append(cfg.get("mfi_window", DEFAULT_MFI_WINDOW))
        periods_needed.append(cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)) # StochRSI needs combined length
        periods_needed.append(cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        periods_needed.append(cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        periods_needed.append(cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        periods_needed.append(cfg.get("fibonacci_window", DEFAULT_FIB_WINDOW))
        # Add a buffer, e.g., 20 candles, as some indicators need more history than just their period
        min_required_data = max(periods_needed) + 20 if periods_needed else 50

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate.{RESET}")
             # Decide whether to proceed or return early
             # return # Option: Stop if not enough data

        try:
            # Create a temporary copy for calculations to avoid modifying original df directly if needed elsewhere
            df_calc = self.df.copy()

            # --- Calculate indicators using pandas_ta ---
            # Always calculate ATR as it's used for SL/TP sizing
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = f"ATRr_{atr_period}" # pandas_ta usually appends 'r' for RMA smoothed ATR

            # Calculate other indicators based on config flags
            indicators_config = self.config.get("indicators", {})

            if indicators_config.get("ema_alignment", True): # EMA Alignment uses EMAs
                ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                df_calc.ta.ema(length=ema_short, append=True)
                self.ta_column_names["EMA_Short"] = f"EMA_{ema_short}"
                df_calc.ta.ema(length=ema_long, append=True)
                self.ta_column_names["EMA_Long"] = f"EMA_{ema_long}"

            if indicators_config.get("momentum", False):
                mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                df_calc.ta.mom(length=mom_period, append=True)
                self.ta_column_names["Momentum"] = f"MOM_{mom_period}"

            if indicators_config.get("cci", False):
                cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
                df_calc.ta.cci(length=cci_period, append=True)
                self.ta_column_names["CCI"] = f"CCI_{cci_period}_0.015" # Default constant

            if indicators_config.get("wr", False): # Williams %R
                wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                df_calc.ta.willr(length=wr_period, append=True)
                self.ta_column_names["Williams_R"] = f"WILLR_{wr_period}"

            if indicators_config.get("mfi", False):
                mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
                df_calc.ta.mfi(length=mfi_period, append=True)
                self.ta_column_names["MFI"] = f"MFI_{mfi_period}"

            if indicators_config.get("vwap", False):
                df_calc.ta.vwap(append=True) # Note: VWAP resets daily usually
                self.ta_column_names["VWAP"] = "VWAP"

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # Ensure psar returns separate columns for long/short/reversal
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                df_calc = pd.concat([df_calc, psar_result], axis=1) # Merge results back
                psar_base = f"{psar_af}_{psar_max_af}"
                self.ta_column_names["PSAR_long"] = f"PSARl_{psar_base}"
                self.ta_column_names["PSAR_short"] = f"PSARs_{psar_base}"
                # self.ta_column_names["PSAR_reversal"] = f"PSARr_{psar_base}" # Check if needed

            if indicators_config.get("sma_10", False): # Example SMA
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = f"SMA_{sma10_period}"

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                # Ensure correct params passed to ta.stochrsi
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                df_calc = pd.concat([df_calc, stochrsi_result], axis=1) # Merge results back
                stochrsi_base = f"_{stoch_rsi_len}_{stoch_rsi_rsi_len}_{stoch_rsi_k}_{stoch_rsi_d}"
                self.ta_column_names["StochRSI_K"] = f"STOCHRSIk{stochrsi_base}"
                self.ta_column_names["StochRSI_D"] = f"STOCHRSId{stochrsi_base}"

            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = f"RSI_{rsi_period}"

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                # Ensure std dev is float for pandas_ta
                bb_std_float = float(bb_std)
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std_float)
                df_calc = pd.concat([df_calc, bbands_result], axis=1) # Merge results back
                # Note: pandas_ta uses float like 2.0 in name
                bb_base = f"{bb_period}_{bb_std_float:.1f}" # Format std dev correctly
                self.ta_column_names["BB_Lower"] = f"BBL_{bb_base}"
                self.ta_column_names["BB_Middle"] = f"BBM_{bb_base}"
                self.ta_column_names["BB_Upper"] = f"BBU_{bb_base}"

            if indicators_config.get("volume_confirmation", False): # Relies on volume MA
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                df_calc.ta.sma(close='volume', length=vol_ma_period, append=True) # Calculate SMA on volume column
                # Store the generated column name
                vol_ma_col = f"SMA_{vol_ma_period}_volume" # Adjust if pandas_ta names it differently
                df_calc.rename(columns={f"SMA_{vol_ma_period}": vol_ma_col}, inplace=True) # Rename if needed
                self.ta_column_names["Volume_MA"] = vol_ma_col


            # Assign the calculated df back to self.df
            self.df = df_calc
            self.logger.debug(f"Calculated indicators for {self.symbol}. DataFrame columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             # Handle cases where a specific ta function might be missing or named differently
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators (check pandas_ta method name?): {e}{RESET}", exc_info=True)
             # self.df remains the original data without calculated indicators
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta: {e}{RESET}", exc_info=True)
            # Optionally clear df or handle state appropriately
            # self.df = pd.DataFrame() # Or keep original df?

        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels after indicators


    def _update_latest_indicator_values(self):
        """Update the indicator_values dictionary with the latest calculated values from self.df."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: DataFrame empty or last row all NaN for {self.symbol}.{RESET}")
            # Initialize keys with NaN if calculation failed or df is empty
            self.indicator_values = {k: np.nan for k in self.ta_column_names.keys()}
            # Add essential price/volume keys too
            for k in ["Close", "Volume", "High", "Low", "ATR"]: self.indicator_values.setdefault(k, np.nan)
            return

        try:
            latest = self.df.iloc[-1]
            updated_values = {}

            # Map internal keys to actual DataFrame column names stored in ta_column_names
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
                # "PSAR_reversal": self.ta_column_names.get("PSAR_reversal"), # If needed
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
                if col_name and col_name in latest.index: # Check if column exists
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            updated_values[key] = float(value)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ({value}) to float.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan # Value is NaN in DataFrame
                else:
                    # Key is expected but column name missing or column doesn't exist
                    if not col_name:
                         # Only log if the indicator was expected based on config
                         if self.config.get("indicators",{}).get(key.lower(), False) or key=="ATR":
                              self.logger.debug(f"Internal key '{key}' not found in ta_column_names mapping for {self.symbol}.")
                    elif col_name not in latest.index:
                        self.logger.debug(f"Column '{col_name}' for indicator '{key}' not found in DataFrame for {self.symbol}.")
                    updated_values[key] = np.nan # Ensure key exists even if missing/NaN

            # Add essential price/volume data from the latest candle
            updated_values["Close"] = float(latest.get('close', np.nan))
            updated_values["Volume"] = float(latest.get('volume', np.nan))
            updated_values["High"] = float(latest.get('high', np.nan))
            updated_values["Low"] = float(latest.get('low', np.nan))

            self.indicator_values = updated_values
            self.logger.debug(f"Latest indicator values updated for {self.symbol}: { {k: v for k, v in self.indicator_values.items() if pd.notna(v)} }")

        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty.")
             self.indicator_values = {k: np.nan for k in self.ta_column_names.keys()} # Reset values
        except Exception as e:
             self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
             self.indicator_values = {k: np.nan for k in self.ta_column_names.keys()} # Reset values


    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.warning(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure we handle potential NaNs in the slice before finding max/min
            high_price = df_slice["high"].dropna().max()
            low_price = df_slice["low"].dropna().min()

            if pd.isna(high_price) or pd.isna(low_price):
                 self.logger.warning(f"Could not find valid high/low in the last {window} periods for Fibonacci.")
                 self.fib_levels_data = {}
                 return {}

            high = Decimal(str(high_price))
            low = Decimal(str(low_price))
            diff = high - low

            levels = {}
            if diff > 0:
                # Use market price precision for formatting levels
                price_precision = self.get_price_precision()
                rounding_factor = Decimal('1e-' + str(price_precision))

                for level in FIB_LEVELS:
                    level_name = f"Fib_{level * 100:.1f}%"
                    # Calculate and quantize the level
                    level_price = (high - (diff * Decimal(str(level)))).quantize(rounding_factor, rounding=ROUND_DOWN)
                    levels[level_name] = level_price
            else:
                 self.logger.warning(f"Fibonacci range is zero or negative (High={high}, Low={low}) for {self.symbol}.")

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {levels}")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Get price precision (number of decimal places) from market info."""
        try:
            precision = int(self.market_info.get('precision', {}).get('price'))
            if precision is not None:
                 # CCXT sometimes returns precision as 1 / tick size (e.g., 0.01)
                 # We need the number of decimal places.
                 # If precision is a float < 1, calculate decimals.
                 if isinstance(precision, float) and 0 < precision < 1:
                      return abs(Decimal(str(precision)).normalize().as_tuple().exponent)
                 # If precision is an integer (usually number of decimals directly)
                 elif isinstance(precision, int):
                      return precision
            # Fallback if market info doesn't provide it easily
            last_close = self.indicator_values.get("Close")
            if last_close and not pd.isna(last_close):
                 s = str(Decimal(str(last_close)))
                 if '.' in s:
                     return len(s.split('.')[-1])
        except Exception as e:
            self.logger.warning(f"Could not reliably determine price precision for {self.symbol}: {e}. Falling back to default.")
        # Default fallback precision (adjust as needed)
        return self.market_info.get('precision', {}).get('base', 6) # Often 4-8 for crypto


    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Find nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
             self.calculate_fibonacci_levels() # Attempt calculation if not already done
             if not self.fib_levels_data:
                  return [] # Return empty if calculation failed

        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price):
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level in self.fib_levels_data.items():
                if isinstance(level, Decimal): # Ensure level is Decimal
                    distance = abs(current_price - level)
                    level_distances.append((name, level, distance))
                else:
                     self.logger.warning(f"Non-decimal value found in fib_levels_data: {name}={level}")

            level_distances.sort(key=lambda x: x[2]) # Sort by distance
            return [(name, level) for name, level, _ in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculate EMA alignment score based on latest values."""
        # Required indicators: EMA_Short, EMA_Long, Close
        ema_short = self.indicator_values.get("EMA_Short", np.nan)
        ema_long = self.indicator_values.get("EMA_Long", np.nan)
        current_price = self.indicator_values.get("Close", np.nan)

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            # self.logger.debug("EMA alignment check skipped: Missing required values.")
            return np.nan # Return NaN if data is missing

        # Bullish alignment: Price > Short EMA > Long EMA
        if current_price > ema_short > ema_long: return 1.0
        # Bearish alignment: Price < Short EMA < Long EMA
        elif current_price < ema_short < ema_long: return -1.0
        # Neutral or mixed alignment
        else: return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generate trading signal based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0} # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0

        # Check if essential data is present
        if not self.indicator_values or all(pd.isna(v) for k,v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low']):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values not calculated or all NaN.{RESET}")
             return "HOLD"
        if pd.isna(current_price):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Missing current price.{RESET}")
             return "HOLD"

        # Iterate through configured indicators and calculate weighted scores
        for indicator_key, enabled in self.config.get("indicators", {}).items():
            # Check if indicator is enabled AND has a weight defined in the active set
            if enabled and indicator_key in self.weights:
                weight = Decimal(str(self.weights[indicator_key]))
                if weight == 0: continue # Skip if weight is zero

                # Find the corresponding check method (e.g., _check_rsi for "rsi")
                check_method_name = f"_check_{indicator_key}"
                if hasattr(self, check_method_name):
                    method = getattr(self, check_method_name)
                    try:
                        # Call the check method - most use self.indicator_values now
                        indicator_score = method() # Returns score between -1.0 and 1.0, or np.nan
                    except Exception as e:
                        self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                        indicator_score = np.nan

                    # Process the score
                    if pd.notna(indicator_score):
                        score_contribution = Decimal(str(indicator_score)) * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                        self.logger.debug(f"  Indicator {indicator_key:<15}: Score={indicator_score:.2f}, Weight={weight:.2f}, Contrib={score_contribution:.3f}")
                    else:
                        nan_indicator_count += 1
                        self.logger.debug(f"  Indicator {indicator_key:<15}: Score=NaN (Skipped)")
                else:
                    self.logger.warning(f"No check method found for enabled/weighted indicator: {indicator_key}")

        # --- Add Order Book Score (Optional, configurable weight?) ---
        # Consider adding a specific weight for orderbook in config?
        orderbook_weight = Decimal("0.15") # Example weight
        if orderbook_data and orderbook_weight > 0:
             orderbook_score = self._check_orderbook(orderbook_data, current_price)
             if pd.notna(orderbook_score):
                 score_contribution = Decimal(str(orderbook_score)) * orderbook_weight
                 final_signal_score += score_contribution
                 total_weight_applied += orderbook_weight # Add to total weight if used
                 self.logger.debug(f"  Indicator {'Orderbook':<15}: Score={orderbook_score:.2f}, Weight={orderbook_weight:.2f}, Contrib={score_contribution:.3f}")
             else:
                 self.logger.debug(f"  Indicator {'Orderbook':<15}: Score=NaN (Skipped)")


        # --- Determine Final Signal ---
        # Normalize score? Optional: Divide by total_weight_applied if not all indicators contributed
        # normalized_score = final_signal_score / total_weight_applied if total_weight_applied > 0 else Decimal("0.0")
        # Using raw score against threshold is simpler:
        threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))

        log_msg = (
            f"Signal Calculation for {self.symbol}:\n"
            f"  Active Weight Set: {self.active_weight_set_name}\n"
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Score: {final_signal_score:.4f}\n"
            f"  Threshold: +/- {threshold:.3f}"
        )
        self.logger.info(log_msg)


        if final_signal_score >= threshold:
            self.signals["BUY"] = 1
            return "BUY"
        elif final_signal_score <= -threshold:
            self.signals["SELL"] = 1
            return "SELL"
        else:
            self.signals["HOLD"] = 1
            return "HOLD"


    # --- Check methods using self.indicator_values ---
    # Each method should return a score between -1.0 and 1.0, or np.nan if data invalid

    def _check_ema_alignment(self) -> float:
        """Check based on calculate_ema_alignment_score."""
        return self.calculate_ema_alignment_score() # Returns score or NaN

    def _check_momentum(self) -> float:
        """Check Momentum indicator."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        # Simple thresholding (can be refined)
        if momentum > 0.1: return 1.0 # Strong positive momentum
        if momentum < -0.1: return -1.0 # Strong negative momentum
        if momentum > 0: return 0.3 # Weak positive
        if momentum < 0: return -0.3 # Weak negative
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Check if current volume confirms price direction (relative to MA)."""
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma == 0:
            return np.nan

        try:
            # High volume suggests confirmation/strength
            if Decimal(str(current_volume)) > Decimal(str(volume_ma)) * multiplier:
                 # Here, we don't know the price trend from volume alone.
                 # This check works best when combined with trend indicators.
                 # Return a positive score indicating *potential* strength, let other indicators determine direction.
                 return 0.5 # Positive score for high volume
            # Low volume might indicate lack of conviction or potential reversal
            elif Decimal(str(current_volume)) < Decimal(str(volume_ma)) / multiplier:
                 return -0.2 # Slight negative score for low volume
            else:
                 return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed for {self.symbol}: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Check Stochastic RSI K and D lines."""
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)

        # Strong signals at extremes
        if k < oversold and d < oversold: return 1.0 # Oversold, potential bounce
        if k > overbought and d > overbought: return -1.0 # Overbought, potential drop

        # Crossover signals (less strong)
        # Need previous values ideally, but approximate with current K vs D
        if k > d: return 0.5 # K crossed above D (or is above) - Bullish momentum
        if k < d: return -0.5 # K crossed below D (or is below) - Bearish momentum
        return 0.0 # Lines might be equal or crossing exactly

    def _check_rsi(self) -> float:
        """Check RSI indicator."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan

        if rsi < 30: return 1.0 # Oversold
        if rsi > 70: return -1.0 # Overbought
        if rsi > 55: return -0.3 # Leaning bearish
        if rsi < 45: return 0.3 # Leaning bullish
        return 0.0 # Neutral zone (45-55)

    def _check_cci(self) -> float:
        """Check CCI indicator."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan

        if cci < -100: return 1.0 # Oversold / potential reversal up
        if cci > 100: return -1.0 # Overbought / potential reversal down
        if cci > 50: return -0.4 # Trending somewhat down
        if cci < -50: return 0.4 # Trending somewhat up
        return 0.0 # Near zero line

    def _check_wr(self) -> float: # Williams %R
        """Check Williams %R indicator."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # Note: WR ranges from -100 to 0.
        if wr <= -80: return 1.0 # Oversold
        if wr >= -20: return -1.0 # Overbought
        if wr > -50: return -0.4 # In upper half (more overbought)
        if wr < -50: return 0.4 # In lower half (more oversold)
        return 0.0

    def _check_psar(self) -> float:
        """Check Parabolic SAR indicator relative to price."""
        psar_l = self.indicator_values.get("PSAR_long", np.nan) # Value when PSAR is below price (support)
        psar_s = self.indicator_values.get("PSAR_short", np.nan) # Value when PSAR is above price (resistance)
        last_close = self.indicator_values.get("Close", np.nan)

        if pd.isna(last_close): return np.nan

        # Determine current trend based on which PSAR value is NOT NaN
        # (pandas_ta outputs NaN for the inactive side)
        is_uptrend = pd.notna(psar_l) and pd.isna(psar_s)
        is_downtrend = pd.notna(psar_s) and pd.isna(psar_l)

        if is_uptrend:
            # PSAR is below price, indicating uptrend
            return 1.0 # Strong bullish signal
        elif is_downtrend:
            # PSAR is above price, indicating downtrend
            return -1.0 # Strong bearish signal
        else:
            # Could be a reversal point or NaN values
            # Check if price crossed PSAR (requires previous state, harder here)
            return 0.0 # Neutral or unknown state

    def _check_sma_10(self) -> float: # Example using SMA10
        """Check price relative to SMA10."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(sma_10) or pd.isna(last_close): return np.nan

        # Simple crossover logic
        if last_close > sma_10: return 0.5 # Price above SMA (weak bullish)
        if last_close < sma_10: return -0.5 # Price below SMA (weak bearish)
        return 0.0

    def _check_vwap(self) -> float:
        """Check price relative to VWAP."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(vwap) or pd.isna(last_close): return np.nan
        # Note: Daily reset of VWAP can cause jumps. Use with caution, maybe alongside EMAs.
        if last_close > vwap: return 0.6 # Price above VWAP (bullish sentiment)
        if last_close < vwap: return -0.6 # Price below VWAP (bearish sentiment)
        return 0.0

    def _check_mfi(self) -> float:
        """Check Money Flow Index."""
        mfi = self.indicator_values.get("MFI", np.nan)
        if pd.isna(mfi): return np.nan
        # MFI combines price and volume
        if mfi < 20: return 1.0 # Oversold (potential buying pressure incoming)
        if mfi > 80: return -1.0 # Overbought (potential selling pressure incoming)
        if mfi > 60: return -0.4 # Leaning overbought/distribution
        if mfi < 40: return 0.4 # Leaning oversold/accumulation
        return 0.0

    def _check_bollinger_bands(self) -> float:
        """Check price relative to Bollinger Bands."""
        bb_lower = self.indicator_values.get("BB_Lower", np.nan)
        bb_upper = self.indicator_values.get("BB_Upper", np.nan)
        # bb_middle = self.indicator_values.get("BB_Middle", np.nan) # Middle band (SMA)
        last_close = self.indicator_values.get("Close", np.nan)
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(last_close): return np.nan

        # Price touching or outside bands suggests potential reversal or continuation
        if last_close < bb_lower: return 1.0 # Below lower band (potential mean reversion buy)
        if last_close > bb_upper: return -1.0 # Above upper band (potential mean reversion sell)
        # Could add checks relative to middle band for trend context
        # Example: Price between middle and upper -> weak bullish (0.3)
        # Example: Price between middle and lower -> weak bearish (-0.3)
        return 0.0 # Price within the bands, not touching extremes

    def _check_orderbook(self, orderbook_data: Dict, current_price: Decimal) -> float:
        """Analyze order book depth for immediate pressure. Returns score -1.0 to 1.0 or NaN."""
        try:
            bids = orderbook_data.get('bids', []) # List of [price, volume]
            asks = orderbook_data.get('asks', []) # List of [price, volume]
            if not bids or not asks:
                self.logger.debug("Orderbook check skipped: Missing bids or asks.")
                return np.nan

            # --- Simple Order Book Imbalance (OBI) ---
            # Consider volume within a certain % range of the current price
            price_range_percent = Decimal("0.001") # 0.1% range around current price
            price_range = current_price * price_range_percent

            # Sum volume of bids within range below current price
            relevant_bid_volume = sum(Decimal(str(bid[1])) for bid in bids if Decimal(str(bid[0])) >= current_price - price_range)
            # Sum volume of asks within range above current price
            relevant_ask_volume = sum(Decimal(str(ask[1])) for ask in asks if Decimal(str(ask[0])) <= current_price + price_range)

            total_volume_in_range = relevant_bid_volume + relevant_ask_volume
            if total_volume_in_range == 0:
                self.logger.debug("Orderbook check: No volume within defined price range.")
                return 0.0 # Neutral if no volume in range

            # Calculate Order Book Imbalance (OBI) ratio
            obi = (relevant_bid_volume - relevant_ask_volume) / total_volume_in_range

            # Scale OBI to a score between -1 and 1
            # Clamp the score to [-1, 1] range
            score = float(max(Decimal("-1.0"), min(Decimal("1.0"), obi * 2))) # Simple scaling, multiply by 2 for wider range? Adjust multiplier as needed.

            # Example: OBI=0.5 (strong bid) -> score=1.0
            # Example: OBI=-0.5 (strong ask) -> score=-1.0
            # Example: OBI=0.1 (weak bid) -> score=0.2

            self.logger.debug(f"Orderbook check: BidVol={relevant_bid_volume:.4f}, AskVol={relevant_ask_volume:.4f}, OBI={obi:.4f}, Score={score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Orderbook analysis failed for {self.symbol}: {e}{RESET}", exc_info=True)
            return np.nan # Return NaN on error


    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, current_price: Decimal, signal: str
    ) -> Tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
        """
        Calculate potential entry, take profit (TP), and initial stop loss (SL) levels.
        The initial SL is primarily used for position sizing, even if TSL is enabled.
        """
        atr_val = self.indicator_values.get("ATR")
        if atr_val is None or pd.isna(atr_val) or atr_val <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: ATR is invalid ({atr_val}).{RESET}")
            return None, None, None
        if current_price is None or pd.isna(current_price) or current_price <= 0:
            self.logger.warning(f"{NEON_YELLOW}Cannot calculate TP/SL for {self.symbol}: Current price is invalid ({current_price}).{RESET}")
            return None, None, None

        try:
            atr = Decimal(str(atr_val))
            entry_price = current_price # Use current price as potential entry

            # Get multipliers from config
            tp_multiple = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_multiple = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))

            # Get market precision for rounding
            price_precision = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_precision))

            take_profit = None
            stop_loss = None

            if signal == "BUY":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                # Round TP up, SL down for safety/better fills
                take_profit = (entry_price + tp_offset).quantize(rounding_factor, rounding=ROUND_UP)
                stop_loss = (entry_price - sl_offset).quantize(rounding_factor, rounding=ROUND_DOWN)
            elif signal == "SELL":
                tp_offset = atr * tp_multiple
                sl_offset = atr * sl_multiple
                # Round TP down, SL up for safety/better fills
                take_profit = (entry_price - tp_offset).quantize(rounding_factor, rounding=ROUND_DOWN)
                stop_loss = (entry_price + sl_offset).quantize(rounding_factor, rounding=ROUND_UP)
            else: # HOLD signal
                return entry_price, None, None # No TP/SL needed for HOLD

            # --- Validation ---
            # Ensure SL is actually beyond entry
            if signal == "BUY" and stop_loss >= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}BUY signal SL calculation invalid: SL ({stop_loss}) >= Entry ({entry_price}). ATR={atr:.5f}, Mult={sl_multiple}. Adjusting SL slightly.{RESET}")
                 stop_loss = (entry_price - rounding_factor).quantize(rounding_factor, rounding=ROUND_DOWN) # Place SL 1 tick below entry
            if signal == "SELL" and stop_loss <= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}SELL signal SL calculation invalid: SL ({stop_loss}) <= Entry ({entry_price}). ATR={atr:.5f}, Mult={sl_multiple}. Adjusting SL slightly.{RESET}")
                 stop_loss = (entry_price + rounding_factor).quantize(rounding_factor, rounding=ROUND_UP) # Place SL 1 tick above entry

             # Ensure TP is profitable relative to entry
            if signal == "BUY" and take_profit <= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}BUY signal TP calculation resulted in non-profitable level: TP ({take_profit}) <= Entry ({entry_price}). ATR={atr:.5f}, Mult={tp_multiple}. Setting TP to None.{RESET}")
                 take_profit = None # Invalidate TP if it's not profitable
            if signal == "SELL" and take_profit >= entry_price:
                 self.logger.warning(f"{NEON_YELLOW}SELL signal TP calculation resulted in non-profitable level: TP ({take_profit}) >= Entry ({entry_price}). ATR={atr:.5f}, Mult={tp_multiple}. Setting TP to None.{RESET}")
                 take_profit = None

            self.logger.debug(f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:.{price_precision}f}, TP={take_profit}, SL={stop_loss}, ATR={atr:.5f}")
            return entry_price, take_profit, stop_loss

        except Exception as e:
             self.logger.error(f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}", exc_info=True)
             return None, None, None


# --- Trading Logic ---

def fetch_balance(exchange: ccxt.Exchange, currency: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the available balance for a specific currency (usually quote currency like USDT)."""
    try:
        # Bybit V5: Need to specify account type if not default
        # Common types: 'CONTRACT' (for derivatives), 'UNIFIED', 'SPOT'
        # Fetching without params might get default (often SPOT or UNIFIED main)
        # Let's try fetching the 'UNIFIED' or 'CONTRACT' balance first
        balance_info = None
        account_types_to_try = ['UNIFIED', 'CONTRACT', 'TRADE'] # TRADE is another potential Bybit V5 term

        for acc_type in account_types_to_try:
             try:
                 # Use params for specific account type if supported by ccxt version/exchange mapping
                 # balance_info = exchange.fetch_balance(params={'accountType': acc_type}) # Older V3 style?
                 balance_info = exchange.fetch_balance(params={'type': acc_type}) # More standard CCXT style
                 # Check if the desired currency is present in this balance info
                 if currency in balance_info:
                      logger.debug(f"Fetched balance using account type: {acc_type}")
                      break # Found balance info containing the currency
                 else:
                      balance_info = None # Reset if currency not found in this account type
             except ccxt.ExchangeError as e:
                  # Ignore errors like 'account type not supported' and try the next one
                  if "account type" in str(e).lower():
                       logger.debug(f"Account type '{acc_type}' not found or error fetching: {e}. Trying next.")
                       continue
                  else:
                       raise e # Re-raise other exchange errors
             except Exception as e:
                 logger.warning(f"Error fetching balance for account type {acc_type}: {e}. Trying next.")
                 continue


        # If specific account types failed, try default fetch_balance
        if not balance_info:
             logger.debug("Fetching balance using default parameters.")
             balance_info = exchange.fetch_balance()

        # --- Parse the balance_info ---
        # Structure can vary greatly between exchanges and account types within Bybit V5
        available_balance = None

        # 1. Standard CCXT structure: balance_info[currency]['free']
        if currency in balance_info and 'free' in balance_info[currency]:
            available_balance = balance_info[currency]['free']
            logger.debug(f"Found balance via standard ccxt structure: {available_balance} {currency}")

        # 2. Bybit V5 structure (often nested in 'info'): Check 'result' list
        elif 'info' in balance_info and 'result' in balance_info['info'] and 'list' in balance_info['info']['result']:
            balance_list = balance_info['info']['result']['list']
            if isinstance(balance_list, list):
                for account in balance_list:
                    # V5 Unified/Contract often uses accountType ('UNIFIED', 'CONTRACT')
                    # Check account type if relevant (though we might have fetched specific type already)
                    # account_type = account.get('accountType')

                    coin_list = account.get('coin')
                    if isinstance(coin_list, list):
                        for coin_data in coin_list:
                             if coin_data.get('coin') == currency:
                                 # Prefer 'availableToWithdraw' or 'availableBalance' as free margin
                                 # These fields might vary based on exact V5 endpoint/account
                                 free = coin_data.get('availableToWithdraw')
                                 if free is None:
                                     free = coin_data.get('availableBalance') # Bybit's term for usable margin
                                 if free is None:
                                      free = coin_data.get('walletBalance') # Fallback to total balance if 'available' missing

                                 if free is not None:
                                     available_balance = free
                                     logger.debug(f"Found balance via Bybit V5 info structure: {available_balance} {currency}")
                                     break # Found the currency
                        if available_balance is not None: break # Stop searching accounts
                if available_balance is None:
                     logger.warning(f"{currency} not found within Bybit V5 'info.result.list[].coin[]' structure.")

        # 3. Fallback: Check top-level 'free' dictionary if present
        elif 'free' in balance_info and currency in balance_info['free']:
             available_balance = balance_info['free'][currency]
             logger.debug(f"Found balance via top-level 'free' dictionary: {available_balance} {currency}")

        # 4. Last Resort: Check total balance if free is still missing
        if available_balance is None:
             total_balance = balance_info.get(currency, {}).get('total')
             if total_balance is not None:
                  logger.warning(f"{NEON_YELLOW}Could not determine 'free'/'available' balance for {currency}. Using 'total' balance ({total_balance}) as fallback. This might include collateral not usable for new trades.{RESET}")
                  available_balance = total_balance
             else:
                  logger.error(f"{NEON_RED}Could not determine balance for {currency}. Balance info structure not recognized or currency missing.{RESET}")
                  logger.debug(f"Full balance_info: {balance_info}") # Log structure for debugging
                  return None

        # Ensure it's a Decimal
        if available_balance is not None:
             return Decimal(str(available_balance))
        else:
             # Should have been caught earlier, but safeguard
             logger.error(f"{NEON_RED}Balance parsing failed unexpectedly for {currency}.{RESET}")
             return None

    except ccxt.AuthenticationError as e:
        logger.error(f"{NEON_RED}Authentication error fetching balance: {e}. Check API key permissions.{RESET}")
        return None
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
    """Gets market information like precision, limits, contract type, etc."""
    try:
        # Ensure markets are loaded
        if not exchange.markets or symbol not in exchange.markets:
             logger.info(f"Market info for {symbol} not loaded or missing, reloading markets...")
             exchange.load_markets(reload=True) # Force reload if symbol missing

        market = exchange.market(symbol)
        if market:
            # Log key details for confirmation and debugging
            market_type = "Contract" if market.get('contract', False) else "Spot"
            logger.debug(
                f"Market Info for {symbol}: ID={market.get('id')}, Type={market_type}, "
                f"Precision(Price/Amount): {market.get('precision', {}).get('price')}/{market.get('precision', {}).get('amount')}, "
                f"Limits(Amount Min/Max): {market.get('limits', {}).get('amount', {}).get('min')}/{market.get('limits', {}).get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {market.get('limits', {}).get('cost', {}).get('min')}/{market.get('limits', {}).get('cost', {}).get('max')}"
            )
            # Add Bybit specific info if useful, e.g., contract size, taker/maker fees
            # logger.debug(f"  Contract Size: {market.get('contractSize')}, Taker Fee: {market.get('taker')}, Maker Fee: {market.get('maker')}")
            return market
        else:
             logger.error(f"{NEON_RED}Market {symbol} still not found after reloading markets.{RESET}")
             return None
    except ccxt.BadSymbol as e:
         logger.error(f"{NEON_RED}Symbol '{symbol}' is not supported by {exchange.id} or has been delisted: {e}{RESET}")
         return None
    except ccxt.NetworkError as e:
         logger.error(f"{NEON_RED}Network error loading markets info for {symbol}: {e}{RESET}")
         return None
    except ccxt.ExchangeError as e:
         logger.error(f"{NEON_RED}Exchange error loading markets info for {symbol}: {e}{RESET}")
         return None
    except Exception as e:
        logger.error(f"{NEON_RED}Error getting market info for {symbol}: {e}{RESET}", exc_info=True)
        return None

def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal, # The calculated initial SL price
    entry_price: Decimal, # Estimated entry price (e.g., current market price)
    market_info: Dict,
    exchange: ccxt.Exchange, # Pass exchange object for formatting
    logger: Optional[logging.Logger] = None
) -> Optional[Decimal]:
    """
    Calculates position size based on risk percentage, initial SL distance,
    and market constraints (min/max size, precision).
    Leverage impacts margin required, not this risk-based size calculation directly.
    """
    lg = logger or logging.getLogger(__name__)
    symbol = market_info.get('symbol', 'UNKNOWN')

    # --- Input Validation ---
    if balance is None or balance <= 0:
        lg.error(f"Position sizing failed for {symbol}: Invalid or zero balance ({balance}).")
        return None
    if not (0 < risk_per_trade < 1):
         lg.error(f"Position sizing failed for {symbol}: Invalid risk_per_trade ({risk_per_trade}). Must be between 0 and 1.")
         return None
    if initial_stop_loss_price is None or entry_price is None:
        lg.error(f"Position sizing failed for {symbol}: Missing entry_price ({entry_price}) or initial_stop_loss_price ({initial_stop_loss_price}).")
        return None
    if initial_stop_loss_price == entry_price:
         lg.error(f"Position sizing failed for {symbol}: Stop loss price cannot be the same as entry price.")
         return None
    if 'limits' not in market_info or 'precision' not in market_info:
         lg.error(f"Position sizing failed for {symbol}: Market info missing 'limits' or 'precision'. Market: {market_info}")
         return None

    try:
        # --- Calculate Risk Amount and Initial Size ---
        risk_amount_quote = balance * Decimal(str(risk_per_trade)) # Risk amount in quote currency (e.g., USDT)
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price) # Price difference per unit of base currency

        if sl_distance_per_unit <= 0:
             lg.error(f"Position sizing failed for {symbol}: Stop loss distance is zero or negative ({sl_distance_per_unit}).")
             return None

        # Size in base currency units (e.g., BTC for BTC/USDT)
        # For inverse contracts (e.g., BTC/USD), sizing is different (contracts * value / price)
        # Assuming linear/spot for this calculation: Size = Risk Amount / (SL Distance * Value per unit)
        # For linear (like BTC/USDT), value per unit is 1.
        # Need contract size for futures
        contract_size = Decimal(str(market_info.get('contractSize', '1'))) # Default to 1 for spot/linear

        # Size = Risk Amount (Quote) / ( SL Distance (Quote/Base) * Contract Size (Base/Contract) )
        # Resulting size is in number of contracts (for futures) or base units (for spot)
        calculated_size = risk_amount_quote / (sl_distance_per_unit * contract_size)

        lg.info(f"Position Sizing for {symbol}: Balance={balance:.2f} {market_info.get('quote','?')}, Risk={risk_per_trade:.2%}, Risk Amount={risk_amount_quote:.4f} {market_info.get('quote','?')}")
        lg.info(f"  Entry={entry_price}, SL={initial_stop_loss_price}, SL Distance={sl_distance_per_unit}")
        lg.info(f"  Contract Size={contract_size}, Initial Calculated Size = {calculated_size:.8f} {market_info.get('base','?') if contract_size == 1 else 'Contracts'}")


        # --- Apply Market Limits and Precision ---
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        precision = market_info.get('precision', {})
        amount_precision = precision.get('amount') # Number of decimal places for amount/size

        # Get min/max amount limits (size limits)
        min_amount = Decimal(str(amount_limits.get('min', '0'))) if amount_limits.get('min') is not None else Decimal('0')
        max_amount = Decimal(str(amount_limits.get('max', 'inf'))) if amount_limits.get('max') is not None else Decimal('inf')

        # Get min/max cost limits (value limits in quote currency)
        min_cost = Decimal(str(cost_limits.get('min', '0'))) if cost_limits.get('min') is not None else Decimal('0')
        max_cost = Decimal(str(cost_limits.get('max', 'inf'))) if cost_limits.get('max') is not None else Decimal('inf')

        # Adjust size based on amount limits
        adjusted_size = calculated_size
        if adjusted_size < min_amount:
            lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} is below minimum amount {min_amount:.8f}. Adjusting to minimum.{RESET}")
            adjusted_size = min_amount
        elif adjusted_size > max_amount:
             lg.warning(f"{NEON_YELLOW}Calculated size {adjusted_size:.8f} exceeds maximum amount {max_amount:.8f}. Capping at maximum.{RESET}")
             adjusted_size = max_amount

        # Check cost limits with the amount-adjusted size
        # Cost = Size * Entry Price * Contract Size (for futures, contractSize=1 for spot/linear)
        current_cost = adjusted_size * entry_price * contract_size
        lg.debug(f"  Cost Check: Adjusted Size={adjusted_size:.8f}, Cost={current_cost:.4f} {market_info.get('quote','?')}")

        if current_cost < min_cost and min_cost > 0:
             lg.warning(f"{NEON_YELLOW}Cost {current_cost:.4f} (Size: {adjusted_size:.8f}) is below minimum cost {min_cost:.4f}. Attempting to increase size to meet min cost.{RESET}")
             # Calculate required size to meet min cost
             required_size_for_min_cost = min_cost / (entry_price * contract_size)
             lg.info(f"  Required size for min cost: {required_size_for_min_cost:.8f}")
             # Only increase size if it doesn't exceed max amount
             if required_size_for_min_cost <= max_amount:
                  adjusted_size = required_size_for_min_cost
                  lg.info(f"  Adjusted size to meet min cost: {adjusted_size:.8f}")
                  # Re-verify that this new size isn't below the absolute min amount (edge case)
                  if adjusted_size < min_amount:
                       lg.error(f"{NEON_RED}Size adjusted for min cost ({adjusted_size:.8f}) is STILL below min amount ({min_amount:.8f}). Cannot meet conflicting limits. Trade aborted.{RESET}")
                       return None
             else:
                  lg.error(f"{NEON_RED}Cannot meet minimum cost {min_cost:.4f} without exceeding maximum amount {max_amount:.8f}. Trade aborted.{RESET}")
                  return None

        elif current_cost > max_cost and max_cost > 0:
             lg.warning(f"{NEON_YELLOW}Cost {current_cost:.4f} exceeds maximum cost {max_cost:.4f}. Reducing size to meet max cost.{RESET}")
             adjusted_size = max_cost / (entry_price * contract_size)
             lg.info(f"  Reduced size to meet max cost: {adjusted_size:.8f}")
             # Re-check if this reduced size is now below min amount
             if adjusted_size < min_amount:
                  lg.error(f"{NEON_RED}Size reduced for max cost ({adjusted_size:.8f}) is now below minimum amount {min_amount:.8f}. Cannot meet conflicting limits. Trade aborted.{RESET}")
                  return None

        # --- Apply Amount Precision ---
        # Use ccxt's amount_to_precision for accurate formatting based on market rules
        try:
            formatted_size_str = exchange.amount_to_precision(symbol, adjusted_size)
            final_size = Decimal(formatted_size_str)
            lg.info(f"Applied amount precision: {adjusted_size:.8f} -> {final_size:.8f} using exchange.amount_to_precision")
        except Exception as fmt_err:
            lg.warning(f"{NEON_YELLOW}Could not use exchange.amount_to_precision for {symbol} ({fmt_err}). Using manual rounding (ROUND_DOWN).{RESET}")
            if amount_precision is not None and amount_precision >= 0:
                rounding_factor = Decimal('1e-' + str(int(amount_precision)))
                final_size = adjusted_size.quantize(rounding_factor, rounding=ROUND_DOWN) # Round down when manually formatting
                lg.info(f"Applied manual amount precision ({amount_precision} decimals): {adjusted_size:.8f} -> {final_size:.8f}")
            else:
                 lg.warning(f"{NEON_YELLOW}Amount precision not defined or invalid for {symbol}. Using size adjusted only for limits: {adjusted_size:.8f}{RESET}")
                 final_size = adjusted_size # Use the size adjusted for limits only


        # --- Final Validation ---
        if final_size <= 0:
             lg.error(f"{NEON_RED}Position size became zero or negative after adjustments/rounding for {symbol}. Trade aborted.{RESET}")
             return None

        # Final check against min amount after formatting/rounding
        if final_size < min_amount:
            # Check if it's *significantly* below min_amount, allowing for small rounding discrepancies
            if (min_amount - final_size) / min_amount > Decimal('0.01'): # Allow 1% tolerance below min due to rounding? Be careful.
                 lg.error(f"{NEON_RED}Final formatted size {final_size:.8f} is significantly below minimum amount {min_amount:.8f} for {symbol}. Trade aborted.{RESET}")
                 return None
            else:
                  lg.warning(f"{NEON_YELLOW}Final formatted size {final_size:.8f} is slightly below minimum {min_amount:.8f} due to rounding for {symbol}. Allowing trade.{RESET}")

        # Final check against min cost
        final_cost = final_size * entry_price * contract_size
        if final_cost < min_cost and min_cost > 0:
            # If min cost was the limiting factor initially, rounding down might violate it again
            lg.error(f"{NEON_RED}Final formatted size {final_size:.8f} results in cost {final_cost:.4f} which is below minimum cost {min_cost:.4f} for {symbol}. Trade aborted.{RESET}")
            return None


        lg.info(f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size:.8f} {market_info.get('base','') if contract_size == 1 else 'Contracts'}{RESET}")
        return final_size

    except KeyError as e:
         lg.error(f"{NEON_RED}Position sizing error for {symbol}: Missing market info key {e}. Market: {market_info}{RESET}")
         return None
    except Exception as e:
        lg.error(f"{NEON_RED}Error calculating position size for {symbol}: {e}{RESET}", exc_info=True)
        return None

def get_open_position(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> Optional[Dict]:
    """
    Checks for an open position for the given symbol using fetch_positions.
    Handles variations in position reporting (size > 0, side field).
    Returns the unified position dictionary from CCXT if an active position exists.
    """
    try:
        # Bybit V5 requires 'symbol' param for fetch_positions when not fetching all
        # Use try-except as some exchanges might error if symbol *must* be provided
        positions: List[Dict] = []
        try:
             positions = exchange.fetch_positions([symbol]) # Fetch only for the target symbol
        except ccxt.ArgumentsRequired as e:
             logger.warning(f"Exchange requires fetching all positions? Fetching all... ({e})")
             all_positions = exchange.fetch_positions()
             # Filter for the specific symbol
             positions = [p for p in all_positions if p.get('symbol') == symbol]
        except Exception as e:
             # Handle other potential errors during fetch
             logger.error(f"Error fetching positions for {symbol}: {e}", exc_info=True)
             return None # Cannot determine position state

        # Iterate through returned positions (usually 0 or 1 for a specific symbol in non-hedge mode)
        active_position = None
        for pos in positions:
            # Standardized fields: 'contracts' (preferred), 'contractSize'
            pos_size_str = pos.get('contracts')
            if pos_size_str is None: pos_size_str = pos.get('contractSize') # Less common standard field for size
            # Non-standard fallback (check 'info' dict)
            if pos_size_str is None: pos_size_str = pos.get('info', {}).get('size') # Common in Bybit V5 info

            # Check if size is valid and non-zero
            if pos_size_str is not None:
                try:
                    position_size = Decimal(str(pos_size_str))
                    # Use a small tolerance to account for potential float noise if exchange returns float strings
                    if abs(position_size) > Decimal('1e-12'):
                        # Found an active position
                        active_position = pos
                        break # Stop after finding the first active position for the symbol
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse position size '{pos_size_str}' for {symbol}. Skipping this position entry.")
                    continue # Skip this entry if size is invalid

        # Process the found active position (if any)
        if active_position:
            # --- Determine Side ---
            side = active_position.get('side') # Standard 'long' or 'short'

            # Infer side from size if 'side' field is missing or ambiguous
            if side not in ['long', 'short']:
                size_decimal = Decimal(str(active_position.get('contracts', active_position.get('info',{}).get('size', '0'))))
                if size_decimal > 0:
                    side = 'long'
                elif size_decimal < 0:
                    # Some exchanges (like Bybit in some modes/endpoints) use negative size for short
                    side = 'short'
                    # Optional: Store the absolute size in a standard field if needed later
                    # active_position['abs_contracts'] = abs(size_decimal)
                else:
                    side = 'unknown' # Should not happen if size check passed, but safeguard

                # Add inferred side back to the dictionary for consistency
                active_position['side'] = side

            # Log details of the found position
            entry_price = active_position.get('entryPrice', active_position.get('info', {}).get('avgPrice', 'N/A'))
            liq_price = active_position.get('liquidationPrice', 'N/A')
            margin = active_position.get('initialMargin', 'N/A')
            leverage = active_position.get('leverage', active_position.get('info', {}).get('leverage', 'N/A'))

            logger.info(f"Found active {side} position for {symbol}: Size={position_size}, Entry={entry_price}, Liq={liq_price}, Leverage={leverage}")
            logger.debug(f"Full position details: {active_position}")
            return active_position
        else:
            # If loop completes without finding a non-zero position
            logger.info(f"No active open position found for {symbol}.")
            return None

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching positions for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        # Handle errors that definitively mean no position exists gracefully (e.g., Bybit V5 retCode != 0)
        # Example specific error codes/messages if known
        no_pos_msgs = ['position idx not exist', 'no position found', 'position does not exist']
        err_str = str(e).lower()
        if any(msg in err_str for msg in no_pos_msgs) or (hasattr(e, 'code') and e.code == 110025): # 110025: Position not found/zero
             logger.info(f"No open position found for {symbol} (Confirmed by exchange error: {e}).")
             return None
        # Bybit V5 might return success (retCode 0) but an empty list if no position - handled by the loop logic above.
        logger.error(f"{NEON_RED}Exchange error fetching positions for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching positions for {symbol}: {e}{RESET}", exc_info=True)
    return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: Dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    if not market_info.get('contract', False):
        logger.info(f"Leverage setting skipped for {symbol} (Not a contract market).")
        return True # Success if not applicable

    if leverage <= 0:
        logger.warning(f"Leverage setting skipped for {symbol}: Invalid leverage value ({leverage}).")
        return False

    # --- Check if leverage is already set ---
    # Fetching position just to check leverage adds latency and API calls.
    # Consider skipping this check unless absolutely necessary or if set_leverage fails often.
    # try:
    #     current_position = get_open_position(exchange, symbol, logger) # Use our existing function
    #     if current_position:
    #         current_leverage_str = current_position.get('leverage', current_position.get('info', {}).get('leverage'))
    #         if current_leverage_str:
    #             current_leverage = int(float(current_leverage_str))
    #             if current_leverage == leverage:
    #                 logger.info(f"Leverage for {symbol} already set to {leverage}x. No action needed.")
    #                 return True
    # except Exception as e:
    #     logger.warning(f"Could not pre-check current leverage for {symbol}: {e}. Proceeding with setting leverage.")
    # --- End pre-check ---


    if not hasattr(exchange, 'set_leverage') and not exchange.has['setLeverage']:
         logger.error(f"{NEON_RED}Exchange {exchange.id} does not support standard set_leverage method via CCXT.{RESET}")
         # TODO: Implement direct Bybit API call using private_post if standard method is missing/fails?
         # Example: exchange.private_post('/v5/position/set-leverage', params={...}) - requires knowing exact params
         return False

    try:
        logger.info(f"Attempting to set leverage for {symbol} to {leverage}x...")

        # --- Prepare Bybit V5 specific parameters ---
        # Bybit V5 often requires setting buy and sell leverage, especially for Isolated Margin.
        # For Cross Margin, setting just one might suffice, but setting both is safer.
        # Check if Isolated Margin is used (might need account settings info)
        # Assume we need to set both for robustness.
        params = {
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage),
            # 'market': symbol # Redundant? symbol is main arg
        }

        # --- Call set_leverage ---
        response = exchange.set_leverage(leverage=leverage, symbol=symbol, params=params)

        # Log response for debugging, but success isn't guaranteed by non-exception response
        logger.debug(f"Set leverage raw response for {symbol}: {response}")

        # --- Post-Setting Verification (Recommended) ---
        # Wait a short moment for the change to propagate on the exchange side
        time.sleep(1)
        verified = False
        try:
             # Fetch position/leverage info *after* setting it
             # Method 1: Fetch positions again (might be slow)
             # pos_after = get_open_position(exchange, symbol, logger)
             # if pos_after: lev_after = pos_after.get('leverage')

             # Method 2: Use fetch_leverage_tiers if available (often faster)
             if exchange.has.get('fetchLeverageTiers'):
                  tiers = exchange.fetch_leverage_tiers([symbol])
                  if symbol in tiers:
                       # Find the tier matching the symbol (structure varies)
                       symbol_tier_info = tiers[symbol]
                       # Look for leverage within the tier info (path varies by exchange)
                       # Example: check tier['leverage'], tier['maxLeverage'] etc.
                       # This needs specific inspection of fetchLeverageTiers response for Bybit
                       # For Bybit V5, this might return risk limit tiers, need to check leverage there
                       # Example structure check:
                       if isinstance(symbol_tier_info, list) and len(symbol_tier_info) > 0:
                           # Bybit V5 fetchLeverageTiers returns list of tiers
                           # Check the leverage associated with the tiers (might not directly confirm *set* leverage)
                           # Let's assume for now if the call succeeded, it worked.
                           logger.debug(f"Leverage tiers info fetched for {symbol}: {symbol_tier_info}")
                           # A more robust check would be needed here based on exact response structure.
                           # For now, we rely on the API call not erroring significantly.
                           verified = True # Tentative verification
             else:
                  logger.info("fetchLeverageTiers not available, relying on set_leverage call success.")
                  verified = True # Assume success if no error

             # Or use fetch_position_leverage if available and specific
             # elif exchange.has.get('fetchPositionLeverage'): ...

        except Exception as verify_err:
             logger.warning(f"Could not verify leverage setting for {symbol} after API call: {verify_err}")
             # Proceed cautiously, assuming it might have worked if no error from set_leverage

        if verified:
            logger.info(f"{NEON_GREEN}Leverage for {symbol} successfully set/requested to {leverage}x.{RESET}")
            return True
        else:
             # If verification step exists and failed
             logger.warning(f"{NEON_YELLOW}Leverage set call for {symbol} seemed to succeed, but verification failed. Check exchange manually.{RESET}")
             return True # Return True cautiously, as the set call itself didn't error


    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error setting leverage for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        err_str = str(e).lower()
        # Common "already set" or "no modification needed" messages
        # Bybit V5 retCode=110045: Leverage not modified
        if "leverage not modified" in err_str or "same leverage" in err_str or (hasattr(e, 'code') and e.code == 110045):
            logger.info(f"{NEON_YELLOW}Leverage for {symbol} already set to {leverage}x (Exchange confirmation).{RESET}")
            return True # Treat as success
        # Bybit V5 retCode=110028: Set margin mode first
        elif "set margin mode first" in err_str or "switch margin mode" in err_str or (hasattr(e, 'code') and e.code == 110028):
             logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. >> Hint: Ensure Margin Mode (Isolated/Cross) is set correctly for {symbol} *before* setting leverage. Check Bybit Account Settings.{RESET}")
        # Bybit V5 retCode=110044: Leverage exceeds risk limit
        elif "risk limit" in err_str or (hasattr(e, 'code') and e.code == 110044):
             logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. >> Hint: Leverage {leverage}x might exceed the risk limit for your account tier. Check Bybit Risk Limit documentation.{RESET}")
        # Bybit V5 retCode=110009: Position is in cross margin mode (cannot set isolated leverage?)
        elif "position is in cross margin mode" in err_str or (hasattr(e, 'code') and e.code == 110009):
             logger.error(f"{NEON_RED}Exchange error setting leverage for {symbol}: {e}. >> Hint: Cannot set leverage individually if symbol is using Cross Margin. Switch to Isolated first or set cross leverage for the whole account.{RESET}")
        elif "available balance not enough" in err_str: # Less common for setting leverage itself, more for orders
             logger.error(f"{NEON_RED}Exchange error setting leverage: {e}. >> Hint: May indicate insufficient available balance if using Isolated Margin and increasing leverage significantly requires more margin allocation possibility.{RESET}")
        else:
             # Log the raw error code if available (useful for debugging Bybit V5)
             bybit_code = getattr(e, 'code', 'N/A')
             logger.error(f"{NEON_RED}Unhandled Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error setting leverage for {symbol}: {e}{RESET}", exc_info=True)

    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str, # "BUY" or "SELL"
    position_size: Decimal,
    market_info: Dict,
    logger: Optional[logging.Logger] = None,
    # Removed sl_price, tp_price - SL/TP set *after* position confirmed
    # Removed enable_tsl flag - handled in main loop after trade
) -> Optional[Dict]:
    """
    Places a market order using CCXT.
    SL/TP/TSL should be set *after* this function confirms the order placement
    and the position is verified.
    """
    lg = logger or logging.getLogger(__name__)
    side = 'buy' if trade_signal == "BUY" else 'sell'
    order_type = 'market'

    # --- Prepare Basic Order Parameters ---
    base_currency = market_info.get('base', '')
    amount_float = float(position_size) # CCXT usually expects float amount

    # Bybit V5: Need to specify position index for One-Way vs Hedge mode
    # Assuming One-Way mode is standard/configured on the account.
    # positionIdx: 0 for One-Way mode; 1 for Buy in Hedge mode; 2 for Sell in Hedge mode
    params = {
        'positionIdx': 0,  # Set for One-Way Mode
        # 'timeInForce': 'GTC', # Good Till Cancelled (default usually) or 'IOC'/'FOK'
        # 'reduceOnly': False, # Ensure it's not a reduce-only order unless intended
        # 'closeOnTrigger': False, # Ensure not closing on trigger unless intended
    }

    lg.info(f"Attempting to place {side.upper()} {order_type} order for {symbol}:")
    lg.info(f"  Size: {amount_float:.8f} {base_currency}")
    lg.info(f"  Params: {params}")

    try:
        # --- Execute Market Order ---
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None, # Market order doesn't need price
            params=params
        )

        # --- Log Success and Order Details ---
        order_id = order.get('id', 'N/A')
        order_status = order.get('status', 'N/A') # Might be 'open' initially for market orders
        lg.info(f"{NEON_GREEN}Trade Placed Successfully! Order ID: {order_id}, Initial Status: {order_status}{RESET}")
        lg.debug(f"Raw order response: {order}") # Log the full order response

        # IMPORTANT: Market orders might not fill instantly or exactly at the last price.
        # The calling function MUST wait and verify the position using get_open_position
        # to get the actual entry price and confirm the size before setting SL/TP/TSL.
        return order # Return the order dictionary

    # --- Error Handling ---
    except ccxt.InsufficientFunds as e:
        lg.error(f"{NEON_RED}Insufficient funds to place {side} order for {symbol}: {e}{RESET}")
        # Log balance info if possible
        try:
            balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
            lg.error(f"  Available {QUOTE_CURRENCY} balance: {balance}")
        except Exception as bal_err:
            lg.error(f"  Could not fetch balance for context: {bal_err}")
    except ccxt.InvalidOrder as e:
        lg.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
        lg.error(f"  Size: {amount_float}, Params: {params}")
        lg.error(f"  Market Limits: Amount={market_info.get('limits',{}).get('amount')}, Cost={market_info.get('limits',{}).get('cost')}")
        lg.error(f"  Market Precision: Amount={market_info.get('precision',{}).get('amount')}, Price={market_info.get('precision',{}).get('price')}")
    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error placing order for {symbol}: {e}{RESET}")
        # Consider adding retry logic here if appropriate for order placement
    except ccxt.ExchangeError as e:
        # Handle specific Bybit V5 error codes for better diagnostics
        bybit_code = getattr(e, 'code', None) # CCXT often maps retCode to e.code
        err_str = str(e).lower()
        lg.error(f"{NEON_RED}Exchange error placing order for {symbol}: {e} (Code: {bybit_code}){RESET}")

        # --- Bybit V5 Specific Error Hints ---
        if bybit_code == 110007 or "insufficient margin balance" in err_str: # Insufficient margin/balance
             lg.error(f"{NEON_YELLOW} >> Hint: Check available balance ({QUOTE_CURRENCY}), leverage, and margin mode (Isolated/Cross). Ensure balance covers Cost = (Size * Price / Leverage).{RESET}")
        elif bybit_code == 110043: # Order cost not available / exceeds limit
             lg.error(f"{NEON_YELLOW} >> Hint: Order cost likely exceeds available balance or risk limits. Check calculation: Cost = (Size * Price / Leverage).{RESET}")
        elif bybit_code == 110044: # Position size has exceeded the risk limit
             lg.error(f"{NEON_YELLOW} >> Hint: Position size + existing position might exceed Bybit's risk limit tier for the current leverage. Check Bybit risk limit docs or reduce size/leverage.{RESET}")
        elif bybit_code == 110014: # Reduce-only order failed
             lg.error(f"{NEON_YELLOW} >> Hint: Ensure 'reduceOnly' is not incorrectly set to True if opening/increasing a position.{RESET}")
        elif bybit_code == 110017: # Order quantity exceeds open limit
             lg.error(f"{NEON_YELLOW} >> Hint: Might violate quantity limits per order or total open orders. Check market limits.{RESET}")
        elif bybit_code == 110025: # Position size is zero (when trying to modify?) - less likely for market order
             lg.error(f"{NEON_YELLOW} >> Hint: Issue related to existing position state? Should not occur for initial market order.{RESET}")
        elif bybit_code == 110055: # Position idx not match position mode
             lg.error(f"{NEON_YELLOW} >> Hint: Mismatch between 'positionIdx' parameter (should be 0 for One-Way) and account's Position Mode (must be One-Way, not Hedge). Check Bybit account trade settings.{RESET}")
        elif "leverage not match order" in err_str: # Less common V5 message?
             lg.error(f"{NEON_YELLOW} >> Hint: Ensure leverage is set correctly *before* placing the order.{RESET}")
        elif "repeated order id" in err_str or "order link id exists" in err_str or bybit_code == 10005: # Duplicate Order ID
             lg.warning(f"{NEON_YELLOW}Duplicate order ID detected. May indicate a previous attempt succeeded or a network issue causing retry. Check position status manually.{RESET}")
             # Consider not treating this as a hard failure, but needs investigation
             return None # Or return a specific status indicating potential duplicate

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)

    return None # Return None if order failed


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict,
    logger: logging.Logger,
    stop_loss_price: Optional[Decimal] = None,
    take_profit_price: Optional[Decimal] = None,
    trailing_stop_distance: Optional[Decimal] = None, # Trailing distance in price points
    tsl_activation_price: Optional[Decimal] = None, # Price to activate TSL
) -> bool:
    """
    Internal helper function to set SL, TP, or TSL for an existing position
    using Bybit's V5 `/v5/position/set-trading-stop` endpoint via CCXT's private_post.
    Requires confirmed position info.
    """
    lg = logger
    if not market_info.get('contract', False):
        lg.warning(f"Position protection (SL/TP/TSL) is typically for contract markets. Skipping for {symbol}.")
        return False

    # --- Validate Inputs ---
    if not position_info:
        lg.error(f"Cannot set protection for {symbol}: Missing position information.")
        return False

    pos_side = position_info.get('side') # Should be 'long' or 'short'
    if pos_side not in ['long', 'short']:
         lg.error(f"Cannot set protection for {symbol}: Invalid or missing position side ('{pos_side}') in position_info.")
         return False

    # At least one protection mechanism must be provided
    if stop_loss_price is None and take_profit_price is None and trailing_stop_distance is None:
         lg.warning(f"No protection parameters (SL, TP, TSL) provided for {symbol}. No action taken.")
         # This isn't strictly an error, but no API call will be made.
         return True # Return True as no failure occurred, just no action.

    # --- Prepare API Parameters ---
    params = {
        'category': market_info.get('type', 'linear'), # 'linear' or 'inverse'
        'symbol': market_info['id'], # Use exchange-specific ID (e.g., BTCUSDT)
        'tpslMode': 'Full', # Apply to the whole position ('Partial' is for partial SL/TP size)
        'slOrderType': 'Market', # Stop loss triggers a market order (most common)
        'tpOrderType': 'Market', # Take profit triggers a market order
        # --- Get positionIdx from position_info if available, default to 0 (One-Way) ---
        'positionIdx': position_info.get('info', {}).get('positionIdx', 0)
    }

    log_parts = [f"Attempting to set protection for {symbol} ({pos_side} position):"]

    # --- Format and Add Parameters ---
    try:
        price_precision_formatter = lambda p: exchange.price_to_precision(symbol, p) if p is not None else None

        # Fixed Stop Loss
        if stop_loss_price is not None:
            formatted_sl = price_precision_formatter(stop_loss_price)
            params['stopLoss'] = formatted_sl
            params['slTriggerBy'] = 'LastPrice' # Or MarkPrice, IndexPrice
            log_parts.append(f"  Fixed SL: {formatted_sl}")

        # Fixed Take Profit
        if take_profit_price is not None:
            formatted_tp = price_precision_formatter(take_profit_price)
            params['takeProfit'] = formatted_tp
            params['tpTriggerBy'] = 'LastPrice'
            log_parts.append(f"  Fixed TP: {formatted_tp}")

        # Trailing Stop Loss (Requires both distance and activation price)
        if trailing_stop_distance is not None and tsl_activation_price is not None:
            # TSL distance (trailingStop) is also treated as a price for formatting
            formatted_tsl_distance = price_precision_formatter(trailing_stop_distance)
            formatted_activation_price = price_precision_formatter(tsl_activation_price)

            if Decimal(formatted_tsl_distance) <= 0:
                 lg.error(f"Invalid TSL distance ({formatted_tsl_distance}) for {symbol}. Must be positive.")
                 return False # Cannot proceed with invalid TSL

            params['trailingStop'] = formatted_tsl_distance
            params['activePrice'] = formatted_activation_price
            # TSL also uses slTriggerBy and slOrderType defined earlier
            log_parts.append(f"  Trailing SL: Distance={formatted_tsl_distance}, Activation={formatted_activation_price}")
            # If TSL is set, Bybit might ignore the 'stopLoss' field if provided simultaneously.
            # It's usually better to set either fixed SL *or* TSL.
            if 'stopLoss' in params:
                 lg.warning(f"Both 'stopLoss' and 'trailingStop' provided for {symbol}. Exchange might prioritize TSL. Removing fixed SL parameter.")
                 del params['stopLoss']

        elif trailing_stop_distance is not None or tsl_activation_price is not None:
            # Error if only one TSL param is provided
            lg.error(f"Cannot set TSL for {symbol}: Both 'trailing_stop_distance' and 'tsl_activation_price' must be provided.")
            return False

    except Exception as fmt_err:
         lg.error(f"Error formatting protection parameters for {symbol}: {fmt_err}", exc_info=True)
         return False

    # Log the attempt
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Params: {params}")

    # --- Call Bybit V5 API Endpoint ---
    try:
        response = exchange.private_post('/v5/position/set-trading-stop', params)
        lg.debug(f"Set protection raw response: {response}")

        # --- Check Response ---
        ret_code = response.get('retCode')
        ret_msg = response.get('retMsg', 'Unknown Error')
        ret_ext = response.get('retExtInfo', {})

        if ret_code == 0:
            lg.info(f"{NEON_GREEN}Position protection (SL/TP/TSL) set successfully for {symbol}.{RESET}")
            return True
        else:
            # Log specific error hints based on Bybit V5 codes
            lg.error(f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}")
            if ret_code == 110043: # tpslMode=Partial needed? Or order cost issue?
                lg.error(f"{NEON_YELLOW} >> Hint: Ensure 'tpslMode' is correct ('Full'/'Partial'). May also relate to order cost if TP/SL creates an order.{RESET}")
            elif ret_code == 110025: # Position size error / not found
                lg.error(f"{NEON_YELLOW} >> Hint: Position might have closed or changed size unexpectedly before protection was set.{RESET}")
            elif ret_code == 110044: # Risk limit exceeded (less likely here, more for orders)
                lg.error(f"{NEON_YELLOW} >> Hint: Protection settings might conflict with risk limits? Unlikely but possible.{RESET}")
            elif ret_code == 110055: # Position idx mismatch
                 lg.error(f"{NEON_YELLOW} >> Hint: Ensure 'positionIdx' ({params.get('positionIdx')}) matches account's Position Mode (One-Way/Hedge).{RESET}")
            elif "active price" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: TSL Activation price ({params.get('activePrice')}) might be invalid (e.g., too close to current price, wrong side of market, already passed).{RESET}")
            elif "trailing stop" in ret_msg.lower() or "trail price" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: TSL distance ({params.get('trailingStop')}) might be invalid (e.g., too small, too large, violates tick size).{RESET}")
            elif "stop loss price is invalid" in ret_msg.lower() or "sl price" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: Fixed SL price ({params.get('stopLoss')}) might be invalid (wrong side of entry, too close).{RESET}")
            elif "take profit price is invalid" in ret_msg.lower() or "tp price" in ret_msg.lower():
                 lg.error(f"{NEON_YELLOW} >> Hint: Fixed TP price ({params.get('takeProfit')}) might be invalid (wrong side of entry, too close).{RESET}")
            # Add more specific codes as encountered
            return False

    except ccxt.NetworkError as e:
        lg.error(f"{NEON_RED}Network error setting protection for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e: # Catch potential errors from private_post call itself
        lg.error(f"{NEON_RED}Exchange error during protection API call for {symbol}: {e}{RESET}")
    except KeyError as e:
        lg.error(f"{NEON_RED}Error setting protection for {symbol}: Missing expected key {e} in market/position info.{RESET}")
    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error setting protection for {symbol}: {e}{RESET}", exc_info=True)

    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: Dict,
    position_info: Dict, # Pass the confirmed position info
    config: Dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Optional[Decimal] = None # Allow passing TP price
) -> bool:
    """
    Calculates TSL parameters (activation price, distance) and calls the internal
    `_set_position_protection` helper function to set TSL (and optionally TP).
    Requires confirmed position details (side, entry price).
    """
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"Trailing Stop Loss is disabled in config for {symbol}. Skipping.")
        return False # Not an error, just disabled

    # --- Get TSL parameters from config ---
    callback_rate = Decimal(str(config.get("trailing_stop_callback_rate", 0.005))) # e.g., 0.5%
    activation_percentage = Decimal(str(config.get("trailing_stop_activation_percentage", 0.003))) # e.g., 0.3% move

    if callback_rate <= 0:
        lg.error(f"{NEON_RED}Invalid trailing_stop_callback_rate ({callback_rate}) in config. Must be positive.{RESET}")
        return False
    # Activation percentage can be zero if TSL should be active immediately (use entry price)
    if activation_percentage < 0:
         lg.error(f"{NEON_RED}Invalid trailing_stop_activation_percentage ({activation_percentage}) in config. Must be non-negative.{RESET}")
         return False

    # --- Extract position details ---
    try:
        # Use reliable fields from CCXT unified position structure
        entry_price_str = position_info.get('entryPrice')
        if entry_price_str is None: entry_price_str = position_info.get('info', {}).get('avgPrice') # Fallback to info

        side = position_info.get('side') # Should be 'long' or 'short'

        if entry_price_str is None or side not in ['long', 'short']:
            lg.error(f"{NEON_RED}Missing required position info (entryPrice, side) to calculate TSL for {symbol}. Position: {position_info}{RESET}")
            return False

        entry_price = Decimal(str(entry_price_str))
        if entry_price <= 0:
             lg.error(f"{NEON_RED}Invalid entry price ({entry_price}) from position info for {symbol}. Cannot calculate TSL.{RESET}")
             return False

    except (TypeError, ValueError, KeyError) as e:
        lg.error(f"{NEON_RED}Error parsing position info for TSL calculation ({symbol}): {e}. Position: {position_info}{RESET}")
        return False

    # --- Calculate TSL parameters for Bybit API ---
    try:
        # Get price precision and tick size
        price_precision = int(market_info.get('precision', {}).get('price', 6)) # Default if missing
        price_rounding = Decimal('1e-' + str(price_precision))
        # Tick size is the smallest price increment (often same as price_rounding factor)
        tick_size = market_info.get('limits', {}).get('price', {}).get('min')
        min_price_increment = price_rounding
        if tick_size is not None:
             min_price_increment = Decimal(str(tick_size))
             # Ensure price_rounding isn't smaller than tick_size if possible
             price_rounding = max(price_rounding, min_price_increment)


        # 1. Calculate Activation Price
        if activation_percentage > 0:
            activation_offset = entry_price * activation_percentage
            if side == 'long':
                # Activate when price moves UP by the percentage
                activation_price = (entry_price + activation_offset).quantize(price_rounding, rounding=ROUND_UP)
                # Ensure activation is strictly above entry
                if activation_price <= entry_price: activation_price = entry_price + min_price_increment
            else: # side == 'short'
                # Activate when price moves DOWN by the percentage
                activation_price = (entry_price - activation_offset).quantize(price_rounding, rounding=ROUND_DOWN)
                 # Ensure activation is strictly below entry
                if activation_price >= entry_price: activation_price = entry_price - min_price_increment
        else:
             # Activate immediately (use entry price or slightly adjusted based on side)
             # Bybit might require activation slightly away from entry, use entry for now and see errors
             activation_price = entry_price
             lg.info(f"TSL activation percentage is zero, attempting immediate activation near entry price {entry_price} for {symbol}.")


        # 2. Calculate Trailing Stop Distance (in price points)
        # Trail distance is based on the callback rate applied to a reference price.
        # Using entry_price as reference is simple, but activation_price might be more logical? Test needed.
        # Let's use activation_price as the reference for distance calc.
        trailing_distance = (activation_price * callback_rate).quantize(price_rounding, rounding=ROUND_UP) # Round distance up

        # Ensure minimum distance respects tick size
        if trailing_distance < min_price_increment:
             lg.warning(f"{NEON_YELLOW}Calculated TSL distance {trailing_distance} is smaller than min price increment {min_price_increment} for {symbol}. Adjusting to min increment.{RESET}")
             trailing_distance = min_price_increment
        elif trailing_distance <= 0:
             lg.error(f"{NEON_RED}Calculated TSL distance is zero or negative ({trailing_distance}) for {symbol}. Check callback rate ({callback_rate}) and activation price ({activation_price}).{RESET}")
             return False

        lg.info(f"Calculated TSL Params for {symbol} ({side}):")
        lg.info(f"  Entry Price: {entry_price:.{price_precision}f}")
        lg.info(f"  Callback Rate: {callback_rate:.3%}")
        lg.info(f"  Activation Price: {activation_price:.{price_precision}f} (Based on {activation_percentage:.3%} move)")
        lg.info(f"  Trailing Distance: {trailing_distance:.{price_precision}f}")
        if take_profit_price:
             lg.info(f"  Take Profit Price: {take_profit_price:.{price_precision}f} (Passed from main logic)")


        # 3. Call the helper function to set TSL (and TP if provided)
        return _set_position_protection(
            exchange=exchange,
            symbol=symbol,
            market_info=market_info,
            position_info=position_info,
            logger=lg,
            stop_loss_price=None, # Do not set fixed SL when setting TSL
            take_profit_price=take_profit_price, # Pass TP if provided
            trailing_stop_distance=trailing_distance,
            tsl_activation_price=activation_price
        )

    except Exception as e:
        lg.error(f"{NEON_RED}Unexpected error calculating TSL parameters for {symbol}: {e}{RESET}", exc_info=True)
        return False


# --- Main Analysis and Trading Loop ---

def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: Dict[str, Any], logger: logging.Logger) -> None:
    """Analyzes a symbol and places/manages trades based on signals and config."""

    logger.info(f"--- Analyzing {symbol} ({config['interval']}) ---")

    # --- 1. Fetch Data ---
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
         logger.error(f"Invalid interval '{config['interval']}' in config. Cannot map to CCXT timeframe.")
         return

    market_info = get_market_info(exchange, symbol, logger)
    if not market_info:
        logger.error(f"{NEON_RED}Failed to get market info for {symbol}. Skipping analysis cycle.{RESET}")
        return

    # Determine required kline history based on longest indicator period
    kline_limit = 500 # Default, ensure enough for most indicators + buffer
    # Could refine kline_limit based on max period in config if needed

    klines_df = fetch_klines_ccxt(exchange, symbol, ccxt_interval, limit=kline_limit, logger=logger)
    if klines_df.empty or len(klines_df) < 50: # Need a reasonable minimum
        logger.error(f"{NEON_RED}Failed to fetch sufficient kline data for {symbol} (fetched {len(klines_df)}). Skipping analysis cycle.{RESET}")
        return

    # Fetch other data as needed
    orderbook_data = None
    if config.get("indicators",{}).get("orderbook", False): # Check if orderbook analysis is enabled in config
         orderbook_data = fetch_orderbook_ccxt(exchange, symbol, config["orderbook_limit"], logger)

    current_price = fetch_current_price_ccxt(exchange, symbol, logger)
    if current_price is None:
         # Fallback to last close price from klines
         try:
             current_price = Decimal(str(klines_df['close'].iloc[-1]))
             logger.warning(f"{NEON_YELLOW}Using last close price ({current_price}) as current price fetch failed for {symbol}.{RESET}")
         except (IndexError, ValueError):
             logger.error(f"{NEON_RED}Failed to get current price or last close for {symbol}. Skipping analysis cycle.{RESET}")
             return


    # --- 2. Analyze Data ---
    analyzer = TradingAnalyzer(
        df=klines_df.copy(), # Pass a copy to avoid modification issues
        logger=logger,
        config=config,
        market_info=market_info # Pass market info to analyzer
    )
    if not analyzer.indicator_values or all(pd.isna(v) for k,v in analyzer.indicator_values.items() if k not in ['Close','Volume','High','Low']):
         logger.error(f"{NEON_RED}Indicator calculation failed or produced all NaNs for {symbol}. Skipping signal generation.{RESET}")
         return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    # Calculate potential initial SL/TP based on current price and ATR (used for sizing and maybe fixed SL/TP)
    _, tp_calc, sl_calc = analyzer.calculate_entry_tp_sl(current_price, signal)
    fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)
    price_precision = analyzer.get_price_precision()


    # --- 3. Log Analysis Results ---
    indicator_log = ""
    log_indicators = ["RSI", "StochRSI_K", "StochRSI_D", "MFI", "CCI", "WR", "ATR"] # Key indicators to log
    for ind_key in log_indicators:
        val = analyzer.indicator_values.get(ind_key)
        precision_ind = 5 if ind_key == "ATR" else 2 # More precision for ATR
        indicator_log += f"{ind_key}: {val:.{precision_ind}f} " if val is not None and pd.notna(val) else f"{ind_key}: NaN "

    tp_str = f"{tp_calc:.{price_precision}f}" if tp_calc else 'N/A'
    sl_str = f"{sl_calc:.{price_precision}f}" if sl_calc else 'N/A'
    output = (
        f"\n{NEON_BLUE}--- Analysis Results: {symbol} ({config['interval']}) @ {datetime.now(TIMEZONE).strftime('%H:%M:%S')} ---{RESET}\n"
        f"Current Price: {NEON_YELLOW}{current_price:.{price_precision}f}{RESET}\n"
        f"Signal: {NEON_GREEN if signal == 'BUY' else NEON_RED if signal == 'SELL' else NEON_YELLOW}{signal}{RESET} "
        f"(Weight Set: {analyzer.active_weight_set_name})\n"
        f"Calculated Initial SL: {sl_str} (ATR: {analyzer.indicator_values.get('ATR', 0):.5f})\n"
        f"Calculated TP: {tp_str}\n"
        f"Indicators: {indicator_log.strip()}\n"
        # f"Nearest Fib Levels: " + ", ".join([f"{name}={level:.{price_precision}f}" for name, level in fib_levels[:3]])
    )
    if config.get('enable_trailing_stop'):
         output += f"Trailing Stop: {NEON_CYAN}Enabled (Rate: {config.get('trailing_stop_callback_rate', 0):.3%}, Activation: {config.get('trailing_stop_activation_percentage', 0):.3%}){RESET}\n"
    else:
         output += f"Trailing Stop: {NEON_RED}Disabled{RESET}\n"

    logger.info(output)

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        # logger.info(f"{NEON_YELLOW}Trading is disabled in config. Analysis complete.{RESET}") # Reduce log noise
        return

    # --- Get Current Position Status ---
    open_position = get_open_position(exchange, symbol, logger)

    # --- Scenario 1: No Open Position ---
    if not open_position:
        if signal in ["BUY", "SELL"]:
            logger.info(f"*** {signal} Signal Triggered & No Open Position: Preparing Trade ***")

            # --- Pre-Trade Checks ---
            balance = fetch_balance(exchange, QUOTE_CURRENCY, logger)
            if balance is None or balance <= 0:
                logger.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot proceed, failed to fetch sufficient balance ({balance}) for {QUOTE_CURRENCY}.{RESET}")
                return
            if sl_calc is None: # Essential for risk sizing
                 logger.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Cannot proceed, Initial Stop Loss calculation failed. Risk cannot be determined.{RESET}")
                 return

            # --- Set Leverage ---
            leverage = int(config.get("leverage", 1))
            if market_info.get('contract', False) and leverage > 0:
                if not set_leverage_ccxt(exchange, symbol, leverage, market_info, logger):
                     logger.warning(f"{NEON_YELLOW}Failed to confirm leverage set to {leverage}x for {symbol}. Proceeding with caution (may use default/previous leverage).{RESET}")
                     # Consider aborting if leverage is critical: return
            else:
                 logger.info(f"Leverage setting skipped (Spot market or leverage <= 0).")


            # --- Calculate Position Size ---
            position_size = calculate_position_size(
                balance=balance,
                risk_per_trade=config["risk_per_trade"],
                initial_stop_loss_price=sl_calc, # Use initial calculated SL for sizing
                entry_price=current_price, # Use current price as entry estimate for sizing
                market_info=market_info,
                exchange=exchange, # Pass exchange for formatting
                logger=logger
            )

            if position_size is None or position_size <= 0:
                logger.error(f"{NEON_RED}Trade Aborted ({symbol} {signal}): Invalid position size calculated ({position_size}). Check balance, risk, SL, market limits, and logs.{RESET}")
                return

            # --- Place Initial Market Order ---
            logger.info(f"Placing {signal} market order for {symbol} | Size: {position_size}")
            trade_order = place_trade(
                exchange=exchange,
                symbol=symbol,
                trade_signal=signal,
                position_size=position_size,
                market_info=market_info,
                logger=logger
                # SL/TP/TSL are NOT set here
            )

            # --- Post-Order: Verify Position and Set Protection ---
            if trade_order and trade_order.get('id'):
                order_id = trade_order['id']
                logger.info(f"Order {order_id} placed for {symbol}. Waiting briefly to verify position...")
                time.sleep(5) # Wait for exchange to process order and update position state

                # Fetch the confirmed position details
                confirmed_position = get_open_position(exchange, symbol, logger)

                if confirmed_position:
                    entry_price_actual = Decimal(str(confirmed_position.get('entryPrice', current_price))) # Use actual entry if available
                    logger.info(f"{NEON_GREEN}Position Confirmed for {symbol}! Actual Entry: {entry_price_actual:.{price_precision}f}{RESET}")

                    # Now set protection based on config (TSL or Fixed SL/TP)
                    protection_set = False
                    if config.get("enable_trailing_stop", False):
                         # Recalculate TP based on *actual* entry price if needed for TSL function
                         # Or just pass the original tp_calc? Let's recalculate TP slightly
                         _, tp_recalc, _ = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)
                         logger.info(f"Setting Trailing Stop Loss for {symbol} (and TP: {tp_recalc})...")
                         protection_set = set_trailing_stop_loss(
                             exchange=exchange,
                             symbol=symbol,
                             market_info=market_info,
                             position_info=confirmed_position,
                             config=config,
                             logger=logger,
                             take_profit_price=tp_recalc # Pass recalculated TP
                         )
                    else:
                         # Set Fixed SL and TP using the helper function
                         # Recalculate SL/TP based on actual entry for accuracy
                         _, tp_recalc, sl_recalc = analyzer.calculate_entry_tp_sl(entry_price_actual, signal)
                         logger.info(f"Setting Fixed Stop Loss ({sl_recalc}) and Take Profit ({tp_recalc}) for {symbol}...")
                         if sl_recalc or tp_recalc: # Only call if at least one is valid
                             protection_set = _set_position_protection(
                                 exchange=exchange,
                                 symbol=symbol,
                                 market_info=market_info,
                                 position_info=confirmed_position,
                                 logger=logger,
                                 stop_loss_price=sl_recalc,
                                 take_profit_price=tp_recalc
                             )
                         else:
                              logger.warning(f"Fixed SL/TP calculation based on actual entry failed for {symbol}. No protection set.")

                    if protection_set:
                         logger.info(f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE for {symbol} ({signal}) ===")
                    else:
                         logger.error(f"{NEON_RED}=== TRADE placed BUT FAILED TO SET PROTECTION (SL/TP/TSL) for {symbol} ({signal}) ===")
                         logger.warning(f"{NEON_YELLOW}Position is open without automated protection. Manual check required!{RESET}")

                else:
                    # Position not found after placing order - potentially critical issue
                    logger.error(f"{NEON_RED}Trade order {order_id} placed, but FAILED TO CONFIRM open position for {symbol} shortly after!{RESET}")
                    logger.error(f"{NEON_YELLOW}Order might have failed to fill, filled partially, or there's a delay/API issue. Manual investigation required!{RESET}")
                    # Could try fetching the order status itself: exchange.fetch_order(order_id, symbol)
            else:
                # place_trade failed
                logger.error(f"{NEON_RED}=== TRADE EXECUTION FAILED for {symbol} ({signal}) ===")
        else:
            # No open position, signal is HOLD
            logger.info(f"Signal is HOLD and no open position for {symbol}. No trade action taken.")


    # --- Scenario 2: Existing Open Position ---
    else: # open_position is not None
        pos_side = open_position.get('side', 'unknown')
        pos_size = open_position.get('contracts', 'N/A')
        entry_price = open_position.get('entryPrice', 'N/A')
        logger.info(f"Existing {pos_side} position found for {symbol}. Size: {pos_size}, Entry: {entry_price}")

        # Check if signal opposes the current position (potential exit signal)
        exit_signal = False
        if (pos_side == 'long' and signal == "SELL") or \
           (pos_side == 'short' and signal == "BUY"):
            exit_signal = True
            logger.warning(f"{NEON_YELLOW}New signal ({signal}) opposes existing {pos_side} position.{RESET}")

            # --- Exit Logic ---
            # Simple exit: Close the position with a market order
            # More complex: Could check PnL, adjust SL first, etc.
            # For now, let's implement a simple market close on opposing signal.
            logger.info(f"Attempting to close {pos_side} position for {symbol} due to opposing signal...")
            close_order = None
            try:
                # Determine close side
                close_side = 'sell' if pos_side == 'long' else 'buy'
                # Get actual size to close (might be negative for shorts in some API versions)
                size_to_close_str = open_position.get('contracts', open_position.get('info',{}).get('size'))
                if size_to_close_str is None:
                     raise ValueError("Could not determine size of existing position to close.")
                size_to_close = abs(Decimal(str(size_to_close_str))) # Use absolute size for closing order amount
                amount_float = float(size_to_close)

                close_params = {
                     'positionIdx': open_position.get('info', {}).get('positionIdx', 0), # Match position index
                     'reduceOnly': True # CRITICAL: Ensure this only closes the position
                }
                logger.info(f"Placing {close_side} market order (reduceOnly) for {amount_float} {market_info.get('base','')}...")

                close_order = exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=close_side,
                    amount=amount_float,
                    params=close_params
                )
                order_id = close_order.get('id', 'N/A')
                logger.info(f"{NEON_GREEN}Position close order placed successfully for {symbol}. Order ID: {order_id}{RESET}")
                # We assume the reduceOnly market order will close the position.
                # Further checks could verify the position is actually gone after a delay.
                # Optional: Cancel existing SL/TP orders if possible/necessary before closing.

            except ccxt.InsufficientFunds as e: # Should not happen for reduceOnly usually
                 logger.error(f"{NEON_RED}Error closing position (InsufficientFunds?): {e}{RESET}")
            except ccxt.InvalidOrder as e:
                 logger.error(f"{NEON_RED}Error closing position (InvalidOrder): {e}{RESET}")
            except ccxt.NetworkError as e:
                 logger.error(f"{NEON_RED}Error closing position (NetworkError): {e}{RESET}")
            except ccxt.ExchangeError as e:
                 bybit_code = getattr(e, 'code', None)
                 logger.error(f"{NEON_RED}Error closing position (ExchangeError): {e} (Code: {bybit_code}){RESET}")
                 if bybit_code == 110014 and "reduce-only" in str(e).lower():
                      logger.error(f"{NEON_YELLOW} >> Hint: Reduce-only order failed. Size might be incorrect or position already closed/closing?{RESET}")
                 elif bybit_code == 110025: # Position not found/zero
                      logger.warning(f"{NEON_YELLOW} >> Hint: Position might have been closed already by SL/TP or manually.{RESET}")
            except Exception as e:
                 logger.error(f"{NEON_RED}Unexpected error closing position {symbol}: {e}{RESET}", exc_info=True)

        elif signal == "HOLD":
            # Existing position and HOLD signal - just log status, maybe check SL/TP
            sl_price = open_position.get('stopLossPrice', open_position.get('info',{}).get('stopLoss', 'N/A'))
            tp_price = open_position.get('takeProfitPrice', open_position.get('info',{}).get('takeProfit', 'N/A'))
            tsl_active = open_position.get('info',{}).get('trailingStop', 'N/A') # Check info for TSL details (might be '0' or distance value)
            liq_price = open_position.get('liquidationPrice', 'N/A')
            unrealizedPnl = open_position.get('unrealizedPnl', 'N/A')

            log_status = (
                f"HOLD signal with active {pos_side} position ({symbol}):\n"
                f"  Size: {pos_size}, Entry: {entry_price}, Liq: {liq_price}\n"
                f"  Unrealized PnL: {unrealizedPnl}\n"
                f"  SL: {sl_price}, TP: {tp_price}, TSL Active: {tsl_active}"
            )
            logger.info(log_status)
            # --- Advanced: Add TSL monitoring/adjustment logic here if needed ---
            # e.g., Check if TSL failed to set previously, check if price is near SL, etc.
            # This requires more state management (knowing if TSL *should* be active).

        else: # Signal matches existing position direction (e.g., BUY signal on existing LONG)
             logger.info(f"Signal ({signal}) matches existing {pos_side} position direction for {symbol}. Holding position.")
             # Optional: Could add logic here to potentially *add* to the position (pyramiding),
             # but this requires careful risk management adjustments. Not implemented here.
             # Or, update SL/TP based on new analysis? (e.g., move SL to break-even)


def main() -> None:
    """Main function to run the bot."""
    global CONFIG # Allow main loop to potentially reload config

    # Use a generic logger for initial setup, then specific logger per symbol
    setup_logger("init") # Create handler for init logger
    init_logger = logging.getLogger("init") # Get the logger instance

    init_logger.info(f"--- Starting LiveX Trading Bot ({datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}) ---")
    init_logger.info(f"Loading configuration from {CONFIG_FILE}...")
    CONFIG = load_config(CONFIG_FILE) # Load/create config file
    init_logger.info(f"Configuration loaded.")
    init_logger.info(f"Using pandas_ta version: {getattr(ta, 'version', 'N/A')}")
    init_logger.info(f"Quote Currency: {QUOTE_CURRENCY}")

    # --- Safety Checks & User Confirmation ---
    if CONFIG.get("enable_trading"):
         init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED in {CONFIG_FILE} !!!{RESET}")
         if CONFIG.get("use_sandbox"):
              init_logger.warning(f"{NEON_YELLOW}Using SANDBOX environment (Testnet).{RESET}")
         else:
              init_logger.warning(f"{NEON_RED}Using REAL MONEY environment. Ensure configuration and risk settings are correct!{RESET}")
         init_logger.warning(f"Risk Per Trade: {CONFIG.get('risk_per_trade', 0)*100:.2f}%, Leverage: {CONFIG.get('leverage', 0)}x")
         init_logger.warning(f"Trailing Stop: {'Enabled' if CONFIG.get('enable_trailing_stop') else 'Disabled'}")

         try:
             confirm = input(">>> Press Enter to acknowledge live trading settings and continue, or Ctrl+C to abort... ")
             init_logger.info("User acknowledged live trading settings. Proceeding...")
         except KeyboardInterrupt:
              init_logger.info("User aborted startup.")
              return
    else:
         init_logger.info("Trading is disabled ('enable_trading': false in config). Running in analysis-only mode.")


    # --- Initialize Exchange ---
    init_logger.info("Initializing exchange connection...")
    exchange = initialize_exchange(init_logger)
    if not exchange:
        init_logger.critical(f"{NEON_RED}Failed to initialize exchange. Exiting.{RESET}")
        return
    init_logger.info(f"Exchange {exchange.id} initialized successfully.")

    # --- Symbol and Interval Selection ---
    target_symbol = None
    while True:
        symbol_input = input(f"{NEON_YELLOW}Enter symbol to trade (e.g., BTC/USDT, ETH/USDT:USDT): {RESET}").upper().strip()
        if not symbol_input: continue

        # Allow different formats like BTC/USDT or BTCUSDT
        # CCXT generally prefers BASE/QUOTE or BASE/QUOTE:QUOTE for linear swaps
        # Let's try to find a valid market matching the input
        potential_symbols = [
            symbol_input, # User input directly
            symbol_input.replace('/', '') + f"/{QUOTE_CURRENCY}", # e.g., BTC -> BTC/USDT
            symbol_input if '/' in symbol_input else f"{symbol_input}/{QUOTE_CURRENCY}", # e.g., BTCUSDT -> BTC/USDT
            f"{symbol_input}:{QUOTE_CURRENCY}" if '/' not in symbol_input and ':' not in symbol_input else symbol_input, # e.g., BTCUSDT -> BTCUSDT:USDT
            symbol_input.replace('/', '') + f":{QUOTE_CURRENCY}" # e.g., BTC -> BTC:USDT (less common)
        ]
        potential_symbols = list(dict.fromkeys(potential_symbols)) # Remove duplicates

        validated_market = None
        for sym in potential_symbols:
            market_check = get_market_info(exchange, sym, init_logger)
            if market_check:
                target_symbol = market_check['symbol'] # Use the exact symbol found by ccxt
                market_type = "Contract" if market_check.get('contract', False) else "Spot"
                init_logger.info(f"Validated symbol: {NEON_GREEN}{target_symbol}{RESET} (Type: {market_type})")
                validated_market = market_check
                break # Stop searching once found

        if validated_market:
            break
        else:
            init_logger.error(f"{NEON_RED}Symbol '{symbol_input}' could not be validated on {exchange.id} using common formats. Please check the symbol and try again.{RESET}")
            # Optional: List some available markets for guidance
            # try:
            #     markets = exchange.load_markets()
            #     available_symbols = list(markets.keys())
            #     init_logger.info(f"Available symbols sample: {available_symbols[:15]}")
            # except Exception as e:
            #     init_logger.warning(f"Could not fetch available symbols: {e}")

    selected_interval = None
    while True:
        interval_input = input(f"{NEON_YELLOW}Enter analysis interval [{'/'.join(VALID_INTERVALS)}]: {RESET}").strip()
        if interval_input in VALID_INTERVALS and interval_input in CCXT_INTERVAL_MAP:
            selected_interval = interval_input
            # Update config for this run (doesn't save back to file unless intended)
            CONFIG["interval"] = selected_interval
            init_logger.info(f"Using analysis interval: {selected_interval} ({CCXT_INTERVAL_MAP[selected_interval]})")
            break
        else:
            init_logger.error(f"{NEON_RED}Invalid interval: '{interval_input}'. Please choose from {VALID_INTERVALS}.{RESET}")


    # --- Setup Logger for the specific symbol ---
    symbol_logger = setup_logger(target_symbol) # Get logger instance for this symbol
    symbol_logger.info(f"--- Starting Analysis Loop for {target_symbol} ({CONFIG['interval']}) ---")


    # --- Main Execution Loop ---
    try:
        while True:
            loop_start_time = time.time()
            symbol_logger.debug(f"--- New analysis cycle started at {datetime.now(TIMEZONE)} ---")

            try:
                # --- Optional: Reload config each loop? ---
                # Useful for dynamic adjustments without restarting the bot
                # CONFIG = load_config(CONFIG_FILE)
                # symbol_logger.debug("Config reloaded.")
                # --- End Optional Reload ---

                # Perform analysis and potentially trade/manage position
                analyze_and_trade_symbol(exchange, target_symbol, CONFIG, symbol_logger)

            # --- Handle Specific CCXT/Network Errors ---
            except ccxt.RateLimitExceeded as e:
                 symbol_logger.warning(f"{NEON_YELLOW}Rate limit exceeded: {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                 symbol_logger.error(f"{NEON_RED}Network error during main loop: {e}. Waiting {RETRY_DELAY_SECONDS*2}s...{RESET}")
                 time.sleep(RETRY_DELAY_SECONDS * 2)
            except ccxt.AuthenticationError as e:
                 symbol_logger.critical(f"{NEON_RED}CRITICAL: Authentication Error in loop: {e}. API keys may be invalid, expired, or permissions revoked. Stopping bot.{RESET}")
                 break # Stop the bot on authentication errors
            except ccxt.ExchangeNotAvailable as e:
                 symbol_logger.error(f"{NEON_RED}Exchange not available (e.g., temporary outage): {e}. Waiting 60s...{RESET}")
                 time.sleep(60)
            except ccxt.OnMaintenance as e:
                 symbol_logger.error(f"{NEON_RED}Exchange is under maintenance: {e}. Waiting 5 minutes...{RESET}")
                 time.sleep(300)
            # --- Handle Generic Errors ---
            except Exception as loop_error:
                 symbol_logger.error(f"{NEON_RED}An uncaught error occurred in the main analysis loop: {loop_error}{RESET}", exc_info=True)
                 symbol_logger.info("Attempting to continue after 15s delay...")
                 time.sleep(15)

            # --- Loop Delay Calculation ---
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, LOOP_DELAY_SECONDS - elapsed_time)
            symbol_logger.debug(f"Analysis cycle took {elapsed_time:.2f}s. Waiting {sleep_time:.2f}s for next cycle...")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        symbol_logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as critical_error:
         # Catch errors outside the main try/except loop (e.g., during setup)
         symbol_logger.critical(f"{NEON_RED}A critical unhandled error occurred: {critical_error}{RESET}", exc_info=True)
    finally:
        symbol_logger.info("--- LiveX Trading Bot stopping ---")
        # --- Close Exchange Connection ---
        if exchange and hasattr(exchange, 'close'):
            try:
                symbol_logger.info("Closing exchange connection...")
                exchange.close()
                symbol_logger.info("Exchange connection closed.")
            except Exception as close_err:
                 symbol_logger.error(f"Error closing exchange connection: {close_err}")
        # --- Final Log Message ---
        symbol_logger.info("Bot stopped.")
        logging.shutdown() # Ensure all handlers are flushed


if __name__ == "__main__":
    main()
```

**Explanation of Key Changes and Enhancements:**

1.  **File Name:** Saved as `livex.py` as requested.
2.  **Configuration (`load_config`, `_ensure_config_keys`):**
    *   Recursively checks and adds missing default keys (including nested ones like `weight_sets` and `indicators`).
    *   Added an `active_weight_set` key to select which indicator weighting strategy to use.
    *   Added a `default` weight set example.
    *   Creates default config if loading fails badly.
3.  **Logging (`setup_logger`):**
    *   Uses the actual traded symbol in the logger name for clarity (`livex_bot_BTC-USDT.log`).
    *   Sets `propagate = False` to prevent duplicate logging if a root logger is configured elsewhere.
    *   Distinct log levels for file (INFO) and stream (INFO) by default. Debug messages go only to file unless stream level is changed.
4.  **Exchange Initialization (`initialize_exchange`):**
    *   More robust initialization options (`defaultType`, timeouts).
    *   Attempts an initial balance fetch after loading markets to test API key validity and permissions early. Logs specific hints if authentication fails.
5.  **Data Fetching (`fetch_klines_ccxt`, `fetch_orderbook_ccxt`):**
    *   Added basic retry logic for `fetch_ohlcv` to handle transient network errors.
    *   More explicit checks if the exchange supports the required methods (`has['fetchOHLCV']`, `has['fetchOrderBook']`).
    *   Improved validation of fetched data (checking for empty lists, NaN values).
6.  **Trading Analyzer (`TradingAnalyzer`):**
    *   Takes `market_info` during initialization to access precision details directly.
    *   Calculates indicators only if they are enabled in the `config['indicators']` section *and* have a weight defined in the `config['active_weight_set']`.
    *   Improved logic for determining required data length based on configured indicator periods.
    *   More robust handling of potential `AttributeError` if pandas_ta methods change/are missing.
    *   More robust `_update_latest_indicator_values` checking if columns exist and handling NaNs.
    *   `get_price_precision` now primarily uses `market_info`.
    *   `generate_trading_signal` uses the `active_weight_set` and applies weights only to enabled indicators. Calculates `total_weight_applied` for potential normalization (though not used by default).
    *   Indicator check methods (`_check_*`) consistently return scores between -1.0 and 1.0 or `np.nan`. Added more detailed logic/thresholds to several checks.
    *   `_check_volume_confirmation` score is now direction-neutral (indicates strength, not trend).
    *   `_check_psar` logic simplified to use the presence/absence of `PSARl` vs `PSARs`.
    *   `_check_orderbook` returns `np.nan` on error/missing data. Implemented Order Book Imbalance (OBI) calculation.
    *   `calculate_entry_tp_sl` uses market precision for rounding and adds validation to ensure SL/TP are on the correct side of entry and TP is potentially profitable.
7.  **Balance Fetching (`fetch_balance`):**
    *   Significantly enhanced to handle Bybit V5's different account types (UNIFIED, CONTRACT) and complex nested balance structures often found in `info['result']['list']`.
    *   Tries specific account types first, then falls back to default `fetch_balance`.
    *   Prioritizes `availableToWithdraw` or `availableBalance` over `walletBalance`.
8.  **Position Sizing (`calculate_position_size`):**
    *   More robust input validation.
    *   Explicitly uses `contract_size` from `market_info` (important for futures).
    *   Improved logic for handling `min_cost` and `max_cost` limits, attempting to adjust size to meet them where possible without violating other limits.
    *   Uses `exchange.amount_to_precision()` for accurate formatting, with a manual rounding fallback.
    *   Performs final validation checks on the formatted size against limits.
9.  **Position Check (`get_open_position`):**
    *   Handles cases where fetching positions requires fetching *all* positions if fetching by symbol isn't supported/required.
    *   More robustly checks standard (`contracts`) and non-standard (`info['size']`) fields for position size.
    *   Correctly infers `side` ('long'/'short') from the sign of the position size if the `side` field is missing/ambiguous.
    *   Improved logging of found position details.
    *   Better handling of specific "no position found" exchange errors.
10. **Leverage Setting (`set_leverage_ccxt`):**
    *   Removed the potentially slow pre-check for leverage.
    *   Uses standard `set_leverage` but includes Bybit V5 specific `params` (`buyLeverage`, `sellLeverage`) for robustness.
    *   Added placeholder for post-setting verification using `fetchLeverageTiers` (needs inspection of Bybit's response structure for full implementation).
    *   Improved error handling with hints for common Bybit V5 leverage-related error codes (e.g., margin mode, risk limits).
11. **Order Placement (`place_trade`):**
    *   **Simplified:** This function now *only* places the initial market order. It no longer accepts or tries to set SL/TP.
    *   Includes `positionIdx: 0` in `params` for Bybit V5 One-Way mode compatibility.
    *   Emphasizes in comments and logs that the calling function *must* verify the position before setting protection.
    *   Enhanced error handling with hints for common Bybit V5 order placement error codes.
12. **Position Protection (`_set_position_protection`, `set_trailing_stop_loss`):**
    *   **New Helper `_set_position_protection`:** Created this internal function to centralize the logic for calling the Bybit V5 `/v5/position/set-trading-stop` endpoint. It takes optional SL, TP, and TSL parameters, formats them, builds the correct `params` dictionary, and makes the API call. It includes detailed error handling for response codes related to SL/TP/TSL setting.
    *   **Refactored `set_trailing_stop_loss`:** This function now focuses on calculating the TSL `activation_price` and `trailing_distance` based on config and position info. It validates these parameters (e.g., positive distance, respecting tick size) and then calls `_set_position_protection` with the calculated TSL parameters (and the optional `take_profit_price`). It no longer makes the direct API call itself.
    *   TSL distance calculation now respects the minimum price increment (tick size).
13. **Main Loop (`analyze_and_trade_symbol`):**
    *   **Revised Order Flow:**
        *   Places the market order using the simplified `place_trade`.
        *   **Waits** using `time.sleep()` (simple delay, could be improved with fill checking).
        *   **Confirms** the position using `get_open_position`.
        *   **If position is confirmed:**
            *   Checks `config['enable_trailing_stop']`.
            *   If True: Calls `set_trailing_stop_loss`, passing the `confirmed_position`, `config`, and the originally calculated `tp_calc`.
            *   If False: Calls the helper `_set_position_protection` directly, passing the `confirmed_position` and the recalculated `sl_recalc`, `tp_recalc` (based on actual entry price).
        *   Handles the case where the position *cannot* be confirmed after placing the order (critical error).
    *   **Existing Position Management:** Added basic logic to close an existing position if the new signal directly opposes it (using a `reduceOnly` market order). Logs status if signal is HOLD or matches position.
    *   Removed the global `current_entry_price`. Uses fetched `current_price` for analysis/sizing and `confirmed_position['entryPrice']` for protection setting.
    *   More detailed logging throughout the decision-making process.
14. **Main Function (`main`):**
    *   Improved user interaction for symbol validation, trying multiple common formats (e.g., `BTC/USDT`, `BTCUSDT`, `BTCUSDT:USDT`).
    *   Enhanced safety checks and user confirmation for live trading.
    *   More specific error handling in the main loop (e.g., `AuthenticationError`, `ExchangeNotAvailable`, `OnMaintenance`).
    *   Graceful shutdown including closing the exchange connection.

This revised structure makes the order placement and protection setting more robust and aligns better with how exchanges typically handle these operations, especially for market orders where the exact entry price isn't known until after the fill.```
```python
# livex.py
# Enhanced version of lwx.py focusing on robust order placement and TSL integration.

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

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago") # e.g., "America/New_York", "Europe/London", "Asia/Tokyo"
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"] # CCXT uses '1m', '5m' etc. Need mapping later
CCXT_INTERVAL_MAP = { # Map our intervals to ccxt's expected format
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"
}
RETRY_ERROR_CODES = [429, 500, 502, 503, 504] # Add relevant error codes
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
LOOP_DELAY_SECONDS = 15 # Time between analysis cycles
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
        "signal_score_threshold": 1.5, # Lowered slightly from scalping
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8, # ATR multiple for initial SL (used for sizing)
        "take_profit_multiple": 0.7, # ATR multiple for TP
        "volume_confirmation_multiplier": 1.5,
        "scalping_signal_threshold": 2.5, # Keep separate threshold for scalping preset
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        "enable_trading": False, # SAFETY FIRST: Default to False, enable consciously
        "use_sandbox": True,     # SAFETY FIRST: Default to True, disable consciously
        "risk_per_trade": 0.01, # Risk 1% of account balance per trade
        "leverage": 20,          # Set desired leverage
        "max_concurrent_positions": 1, # Limit open positions for this symbol (common strategy)
        "quote_currency": "USDT", # Currency for balance check and sizing
        # --- Trailing Stop Loss Config ---
        "enable_trailing_stop": True, # Default to enabling TSL
        "trailing_stop_callback_rate": 0.005, # e.g., 0.5% trail distance (as decimal)
        "trailing_stop_activation_percentage": 0.003, # e.g., Activate TSL when price moves 0.3% in favor from entry
        # --- End Trailing Stop Loss Config ---
        "indicators": { # Control which indicators are calculated and used
            "ema_alignment": True, "momentum": True, "volume_confirmation": True,
            "stoch_rsi": True, "rsi": True, "bollinger_bands": True, "vwap": True,
            "cci": True, "wr": True, "psar": True, "sma_10": True, "mfi": True,
            "orderbook": True, # Add flag to enable orderbook analysis
        },
        "weight_sets": { # Define different weighting strategies
            "scalping": { # Example weighting for a scalping strategy
                "ema_alignment": 0.2, "momentum": 0.3, "volume_confirmation": 0.2,
                "stoch_rsi": 0.6, "rsi": 0.2, "bollinger_bands": 0.3, "vwap": 0.4,
                "cci": 0.3, "wr": 0.3, "psar": 0.2, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.15,
            },
            "default": { # A more balanced weighting strategy
                "ema_alignment": 0.3, "momentum": 0.2, "volume_confirmation": 0.1,
                "stoch_rsi": 0.4, "rsi": 0.3, "bollinger_bands": 0.2, "vwap": 0.3,
                "cci": 0.2, "wr": 0.2, "psar": 0.3, "sma_10": 0.1, "mfi": 0.2, "orderbook": 0.1,
            }
        },
        "active_weight_set": "default" # Choose which weight set to use
    }

    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        return default_config

    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
            # Ensure all default keys exist in the loaded config recursively
            updated_config = _ensure_config_keys(config_from_file, default_config)
            # Save back if keys were added
            if updated_config != config_from_file:
                 with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(updated_config, f, indent=4)
                 print(f"{NEON_YELLOW}Updated config file with default keys: {filepath}{RESET}")
            return updated_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_RED}Error loading config: {e}. Using default config.{RESET}")
        # Create default if loading failed badly
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created default config file: {filepath}{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensure all keys from default_config are in config."""
    updated_config = config.copy()
    for key, value in default_config.items():
        if key not in updated_config:
            updated_config[key] = value
        elif isinstance(value, dict) and isinstance(updated_config.get(key), dict):
            # Recursively check nested dictionaries
            updated_config[key] = _ensure_config_keys(updated_config[key], value)
    return updated_config

CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT") # Get quote currency from config

# --- Logger Setup ---
def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    # Clean symbol for filename (replace / and : )
    safe_symbol = symbol.replace('/', '_').replace(':', '-')
    logger_name = f"livex_bot_{safe_symbol}" # Use safe symbol in logger name
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG) # Set base level to DEBUG to capture all messages

    # File Handler (writes DEBUG and above)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_formatter = SensitiveFormatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s") # Add line number
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG) # Log DEBUG and higher to file
    logger.addHandler(file_handler)

    # Stream Handler (console, writes INFO and above by default)
    stream_handler = logging.StreamHandler()
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    # Set console level (e.g., INFO for normal operation, DEBUG for detailed)
    console_log_level = logging.INFO # Or set via config/env variable
    stream_handler.setLevel(console_log_level)
    logger.addHandler(stream_handler)

    # Set propagate to False to prevent logs going to the root logger if it's configured
    logger.propagate = False

    return logger

# --- CCXT Exchange Setup ---
def initialize_exchange(logger: logging.Logger) -> Optional[ccxt.Exchange]:
    """Initializes the CCXT Bybit exchange object."""
    try:
        exchange_options = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear', # Assume linear contracts (USDT margined)
                'adjustForTimeDifference': True,
                # Bybit V5 Unified Margin Account setting
                # Check your account type on Bybit website. Unified is common now.
                # 'brokerId': 'YOUR_BROKER_ID', # If using via a broker program
                # 'warnOnFetchOpenOrdersWithoutSymbol': False, # Suppress warning if needed
                # Test connection on initialization
                'fetchTickerTimeout': 10000, # 10 seconds timeout
                'fetchBalanceTimeout': 15000, # 15 seconds timeout
            }
        }
        # Handle potential sandbox URL override from .env
        if CONFIG.get('use_sandbox'):
            logger.warning(f"{NEON_YELLOW}USING SANDBOX MODE{RESET}")
            # Some exchanges have specific sandbox URLs, ccxt handles Bybit via set_sandbox_mode
            # exchange_options['urls'] = {'api': 'https://api-testnet.bybit.com'} # Manual override if needed

        exchange = ccxt.bybit(exchange_options)

        if CONFIG.get('use_sandbox'):
            exchange.set_sandbox_mode(True)

        # Test connection by fetching markets
        exchange.load_markets()
        logger.info(f"CCXT exchange initialized ({exchange.id}). Sandbox: {CONFIG.get('use_sandbox')}")

        # Optional: Test API credentials by fetching balance
        try:
            # Fetch balance appropriate for linear contracts
            balance = exchange.fetch_balance(params={'type': 'CONTRACT'}) # Try 'CONTRACT' or 'UNIFIED'
            logger.info(f"Successfully connected and fetched initial balance (Example: {QUOTE_CURRENCY} available: {balance.get(QUOTE_CURRENCY, {}).get('free', 'N/A')})")
        except ccxt.AuthenticationError as auth_err:
             logger.error(f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}")
             logger.error(f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), and IP whitelist is set if required.{RESET}")
             return None # Exit if authentication fails early
        except Exception as balance_err:
             logger.warning(f"{NEON_YELLOW}Could not perform initial balance fetch: {balance_err}. Continuing, but check permissions if trading fails.{RESET}")

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
    """Fetch the current price of a trading symbol using CCXT ticker."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        # Prefer 'last' price, fallback to 'bid' or 'ask' average if 'last' is unavailable/stale
        price = ticker.get('last')
        bid = ticker.get('bid')
        ask = ticker.get('ask')

        if price is None or price <= 0: # Check if last price is valid
            logger.warning(f"Ticker 'last' price unavailable or invalid ({price}) for {symbol}. Using bid/ask midpoint.")
            if bid and ask and bid > 0 and ask > 0:
                price = (Decimal(str(bid)) + Decimal(str(ask))) / 2
            elif ask and ask > 0: # Only ask available
                price = ask
                logger.warning(f"Using ask price ({ask}) as fallback for {symbol}.")
            elif bid and bid > 0: # Only bid available
                price = bid
                logger.warning(f"Using bid price ({bid}) as fallback for {symbol}.")
            else:
                 logger.error(f"{NEON_RED}Failed to get valid current price for {symbol} via ticker (last/bid/ask all invalid). Ticker: {ticker}{RESET}")
                 return None # All price sources invalid

        # Final check if price is valid Decimal > 0
        if price is not None:
             price_decimal = Decimal(str(price))
             if price_decimal > 0:
                  return price_decimal
             else:
                  logger.error(f"{NEON_RED}Fetched price ({price_decimal}) is not positive for {symbol}. Ticker: {ticker}{RESET}")
                  return None
        else:
            # Should have been caught earlier, but safeguard
            logger.error(f"{NEON_RED}Price variable is None after checks for {symbol}. Ticker: {ticker}{RESET}")
            return None

    except ccxt.NetworkError as e:
        logger.error(f"{NEON_RED}Network error fetching price for {symbol}: {e}{RESET}")
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error fetching price for {symbol}: {e}{RESET}")
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error fetching price for {symbol}: {e}{RESET}", exc_info=True)
    return None

def fetch_klines_ccxt(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 250, logger: logging.Logger = None) -> pd.DataFrame:
    """Fetch OHLCV kline data using CCXT with retries."""
    lg = logger or logging.getLogger(__name__) # Use provided logger or default
    try:
        if not exchange.has['fetchOHLCV']:
             lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
             return pd.DataFrame()

        # Fetch data with retries for network issues
        ohlcv = None
        for attempt in range(MAX_API_RETRIES + 1):
             try:
                  lg.debug(f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt+1})")
                  ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                  break # Success
             except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                  if attempt < MAX_API_RETRIES:
                      lg.warning(f"Network error fetching klines for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
                      time.sleep(RETRY_DELAY_SECONDS)
                  else:
                      lg.error(f"{NEON_RED}Max retries reached fetching klines for {symbol} after network errors.{RESET}")
                      raise e # Re-raise the last error
             except ccxt.ExchangeError as e:
                 # Some exchange errors might be retryable (e.g., rate limits if not handled by ccxt)
                 # Check e.http_status or specific error messages if needed
                 lg.error(f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}{RESET}")
                 raise e # Re-raise non-network errors immediately

        if not ohlcv:
            lg.warning(f"{NEON_YELLOW}No kline data returned for {symbol} {timeframe}.{RESET}")
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data DataFrame is empty for {symbol} {timeframe} immediately after creation.{RESET}")
             return df

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Ensure correct dtypes for pandas_ta and calculations
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce errors to NaN

        # Drop rows where essential price data is missing or zero (common data quality issue)
        initial_len = len(df)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        # Also remove rows where close price is non-positive
        df = df[df['close'] > 0]
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0:
             lg.debug(f"Dropped {rows_dropped} rows with NaN/invalid price data for {symbol}.")


        if df.empty:
             lg.warning(f"{NEON_YELLOW}Kline data for {symbol} {timeframe} was empty after processing/dropna.{RESET}")

        lg.info(f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}")
        return df

    except ccxt.NetworkError as e: # Catch error if retries fail
        lg.error(f"{NEON_RED}Network error fetching klines for {symbol} after retries: {e}{RESET}")
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
            if not exchange.has['fetchOrderBook']:
                 logger.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
                 return None
            lg.debug(f"Fetching order book for {symbol}, limit={limit} (Attempt {attempts+1})")
            orderbook = exchange.fetch_order_book(symbol, limit=limit)
            # Basic validation
            if orderbook and isinstance(orderbook.get('bids'), list) and isinstance(orderbook.get('asks'), list):
                # Further check: ensure bids/asks lists are not empty if expected
                # if not orderbook['bids'] and not orderbook['asks']:
                #     logger.warning(f"Orderbook received but bids and asks are both empty for {symbol}.")
                    # Decide if this is an error or valid state
                return orderbook
            else:
                logger.warning(f"{NEON_YELLOW}Empty or invalid orderbook structure received for {symbol}. Attempt {attempts + 1}/{MAX_API_RETRIES + 1}. Response: {orderbook}{RESET}")

        except (ccxt.NetworkError, ccxt.RequestTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.warning(f"{NEON_YELLOW}Orderbook fetch network error for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempts + 1}/{MAX_API_RETRIES + 1}){RESET}")

        except ccxt.ExchangeError as e:
            logger.error(f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}{RESET}")
            # Don't retry on definitive exchange errors (e.g., bad symbol) unless specifically handled
            return None
        except Exception as e:
            logger.error(f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}{RESET}", exc_info=True)
            return None # Don't retry on unexpected errors

        attempts += 1
        if attempts <= MAX_API_RETRIES:
             time.sleep(RETRY_DELAY_SECONDS)

    logger.error(f"{NEON_RED}Max retries reached fetching orderbook for {symbol}.{RESET}")
    return None

# --- Trading Analyzer Class (Using pandas_ta) ---
class TradingAnalyzer:
    """Analyze trading data and generate signals using pandas_ta."""

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict[str, Any],
        market_info: Dict[str, Any], # Pass market info for precision etc.
    ) -> None:
        self.df = df # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get('symbol', 'UNKNOWN')
        self.interval = config.get("interval", "UNKNOWN")
        self.ccxt_interval = CCXT_INTERVAL_MAP.get(self.interval, "UNKNOWN") # Map to ccxt format
        self.indicator_values: Dict[str, float] = {} # Stores latest indicator values
        self.signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0} # Simple signal state
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config["weight_sets"].get(self.active_weight_set_name, {})
        self.fib_levels_data: Dict[str, Decimal] = {} # Stores calculated fib levels
        self.ta_column_names = {} # Stores actual column names generated by pandas_ta

        if not self.weights:
             logger.error(f"Active weight set '{self.active_weight_set_name}' not found in config. Using empty weights.")

        self._calculate_all_indicators() # Calculate indicators on initialization

    def _get_ta_col_name(self, base_name: str, *args) -> str:
        """Helper to construct the default column name used by pandas_ta. (May need adjustment based on pandas_ta version)"""
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

        # Check for sufficient data
        periods_needed = []
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        if True: periods_needed.append(cfg.get("atr_period", DEFAULT_ATR_PERIOD)) # Always calc ATR
        if indi_cfg.get("ema_alignment"): periods_needed.append(cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if indi_cfg.get("momentum"): periods_needed.append(cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD))
        if indi_cfg.get("cci"): periods_needed.append(cfg.get("cci_window", DEFAULT_CCI_WINDOW))
        if indi_cfg.get("wr"): periods_needed.append(cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW))
        if indi_cfg.get("mfi"): periods_needed.append(cfg.get("mfi_window", DEFAULT_MFI_WINDOW))
        # VWAP doesn't have a period setting in the same way
        # PSAR uses initial/max AF, not period
        if indi_cfg.get("sma_10"): periods_needed.append(cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW))
        if indi_cfg.get("stoch_rsi"): periods_needed.append(cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW) + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW))
        if indi_cfg.get("rsi"): periods_needed.append(cfg.get("rsi_period", DEFAULT_RSI_WINDOW))
        if indi_cfg.get("bollinger_bands"): periods_needed.append(cfg.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD))
        if indi_cfg.get("volume_confirmation"): periods_needed.append(cfg.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD))
        # Add fibonacci window if needed elsewhere, though calculated separately
        # periods_needed.append(cfg.get("fibonacci_window", DEFAULT_FIB_WINDOW))

        min_required_data = max(periods_needed) + 20 if periods_needed else 50 # Add buffer

        if len(self.df) < min_required_data:
             self.logger.warning(f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators (min recommended: {min_required_data}). Results may be inaccurate.{RESET}")
             # Decide whether to proceed or return early
             # return # Option: Stop if not enough data

        try:
            # Use a temporary DataFrame for calculations to avoid modifying original df if needed
            df_calc = self.df.copy()

            # --- Calculate indicators using pandas_ta ---
            # Always calculate ATR as it's used for SL/TP sizing
            atr_period = self.config.get("atr_period", DEFAULT_ATR_PERIOD)
            df_calc.ta.atr(length=atr_period, append=True)
            self.ta_column_names["ATR"] = f"ATRr_{atr_period}" # pandas_ta usually appends 'r' for RMA smoothed ATR

            # Calculate other indicators based on config flags
            indicators_config = self.config.get("indicators", {})

            if indicators_config.get("ema_alignment", False): # EMA Alignment uses EMAs
                ema_short = self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD)
                ema_long = self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD)
                df_calc.ta.ema(length=ema_short, append=True)
                self.ta_column_names["EMA_Short"] = f"EMA_{ema_short}"
                df_calc.ta.ema(length=ema_long, append=True)
                self.ta_column_names["EMA_Long"] = f"EMA_{ema_long}"

            if indicators_config.get("momentum", False):
                mom_period = self.config.get("momentum_period", DEFAULT_MOMENTUM_PERIOD)
                df_calc.ta.mom(length=mom_period, append=True)
                self.ta_column_names["Momentum"] = f"MOM_{mom_period}"

            if indicators_config.get("cci", False):
                cci_period = self.config.get("cci_window", DEFAULT_CCI_WINDOW)
                df_calc.ta.cci(length=cci_period, append=True)
                self.ta_column_names["CCI"] = f"CCI_{cci_period}_0.015" # Default constant

            if indicators_config.get("wr", False): # Williams %R
                wr_period = self.config.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW)
                df_calc.ta.willr(length=wr_period, append=True)
                self.ta_column_names["Williams_R"] = f"WILLR_{wr_period}"

            if indicators_config.get("mfi", False):
                mfi_period = self.config.get("mfi_window", DEFAULT_MFI_WINDOW)
                df_calc.ta.mfi(length=mfi_period, append=True)
                self.ta_column_names["MFI"] = f"MFI_{mfi_period}"

            if indicators_config.get("vwap", False):
                df_calc.ta.vwap(append=True) # Note: VWAP resets daily usually
                self.ta_column_names["VWAP"] = "VWAP" # ta typically names it just VWAP

            if indicators_config.get("psar", False):
                psar_af = self.config.get("psar_af", DEFAULT_PSAR_AF)
                psar_max_af = self.config.get("psar_max_af", DEFAULT_PSAR_MAX_AF)
                # Ensure psar returns separate columns for long/short/reversal
                psar_result = df_calc.ta.psar(af=psar_af, max_af=psar_max_af)
                # Need to check exact column names returned by psar in your pandas_ta version
                # Common names: PSARl_0.02_0.2, PSARs_0.02_0.2, PSARaf_0.02_0.2, PSARr_0.02_0.2
                df_calc = pd.concat([df_calc, psar_result], axis=1) # Merge results back
                psar_base = f"{psar_af}_{psar_max_af}"
                self.ta_column_names["PSAR_long"] = next((col for col in psar_result.columns if col.startswith("PSARl")), None)
                self.ta_column_names["PSAR_short"] = next((col for col in psar_result.columns if col.startswith("PSARs")), None)
                # self.ta_column_names["PSAR_reversal"] = next((col for col in psar_result.columns if col.startswith("PSARr")), None) # Check if needed

            if indicators_config.get("sma_10", False): # Example SMA
                sma10_period = self.config.get("sma_10_window", DEFAULT_SMA_10_WINDOW)
                df_calc.ta.sma(length=sma10_period, append=True)
                self.ta_column_names["SMA10"] = f"SMA_{sma10_period}"

            if indicators_config.get("stoch_rsi", False):
                stoch_rsi_len = self.config.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                stoch_rsi_rsi_len = self.config.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
                stoch_rsi_k = self.config.get("stoch_rsi_k", DEFAULT_K_WINDOW)
                stoch_rsi_d = self.config.get("stoch_rsi_d", DEFAULT_D_WINDOW)
                # Ensure correct params passed to ta.stochrsi
                stochrsi_result = df_calc.ta.stochrsi(length=stoch_rsi_len, rsi_length=stoch_rsi_rsi_len, k=stoch_rsi_k, d=stoch_rsi_d)
                # Check exact column names (e.g., STOCHRSIk_14_14_3_3, STOCHRSId_14_14_3_3)
                df_calc = pd.concat([df_calc, stochrsi_result], axis=1) # Merge results back
                # Construct expected names based on defaults or find dynamically
                stochrsi_base = f"_{stoch_rsi_len}_{stoch_rsi_rsi_len}_{stoch_rsi_k}_{stoch_rsi_d}"
                self.ta_column_names["StochRSI_K"] = next((col for col in stochrsi_result.columns if col.startswith("STOCHRSIk")), None)
                self.ta_column_names["StochRSI_D"] = next((col for col in stochrsi_result.columns if col.startswith("STOCHRSId")), None)


            if indicators_config.get("rsi", False):
                rsi_period = self.config.get("rsi_period", DEFAULT_RSI_WINDOW)
                df_calc.ta.rsi(length=rsi_period, append=True)
                self.ta_column_names["RSI"] = f"RSI_{rsi_period}"

            if indicators_config.get("bollinger_bands", False):
                bb_period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
                bb_std = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
                # Ensure std dev is float for pandas_ta
                bb_std_float = float(bb_std)
                bbands_result = df_calc.ta.bbands(length=bb_period, std=bb_std_float)
                 # Check exact column names (e.g., BBL_20_2.0, BBM_20_2.0, BBU_20_2.0)
                df_calc = pd.concat([df_calc, bbands_result], axis=1) # Merge results back
                bb_base = f"{bb_period}_{bb_std_float:.1f}" # Format std dev correctly
                self.ta_column_names["BB_Lower"] = next((col for col in bbands_result.columns if col.startswith("BBL")), None)
                self.ta_column_names["BB_Middle"] = next((col for col in bbands_result.columns if col.startswith("BBM")), None)
                self.ta_column_names["BB_Upper"] = next((col for col in bbands_result.columns if col.startswith("BBU")), None)


            if indicators_config.get("volume_confirmation", False): # Relies on volume MA
                vol_ma_period = self.config.get("volume_ma_period", DEFAULT_VOLUME_MA_PERIOD)
                # Calculate SMA on volume column, explicitly naming the output
                vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"
                df_calc[vol_ma_col_name] = df_calc.ta.sma(close=df_calc['volume'], length=vol_ma_period, append=False)
                self.ta_column_names["Volume_MA"] = vol_ma_col_name


            # Assign the calculated df back to self.df
            self.df = df_calc
            self.logger.debug(f"Calculated indicators for {self.symbol}. DataFrame columns: {self.df.columns.tolist()}")

        except AttributeError as e:
             # Handle cases where a specific ta function might be missing or named differently
             self.logger.error(f"{NEON_RED}AttributeError calculating indicators for {self.symbol} (check pandas_ta method name?): {e}{RESET}", exc_info=True)
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}", exc_info=True)

        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels() # Calculate Fib levels after indicators


    def _update_latest_indicator_values(self):
        """Update the indicator_values dictionary with the latest calculated values from self.df."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            self.logger.warning(f"{NEON_YELLOW}Cannot update latest values: DataFrame empty or last row all NaN for {self.symbol}.{RESET}")
            # Initialize keys with NaN if calculation failed or df is empty
            all_potential_keys = list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]
            self.indicator_values = {k: np.nan for k in all_potential_keys}
            return

        try:
            latest = self.df.iloc[-1]
            updated_values = {}

            # Map internal keys to actual DataFrame column names stored in ta_column_names
            # Use the dynamically stored names
            mapping = self.ta_column_names.copy() # Use the names stored during calculation

            for key, col_name in mapping.items():
                if col_name and col_name in latest.index: # Check if column exists
                    value = latest[col_name]
                    if pd.notna(value):
                        try:
                            updated_values[key] = float(value)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Could not convert value for {key} ('{col_name}': {value}) to float for {self.symbol}.")
                            updated_values[key] = np.nan
                    else:
                        updated_values[key] = np.nan # Value is NaN in DataFrame
                else:
                    # Key is expected but column name missing or column doesn't exist
                    # Log if the indicator was expected to be calculated
                    indicator_short_key = key.split('_')[0] # Crude way to get base indicator name
                    if self.config.get("indicators",{}).get(indicator_short_key.lower(), False) or key=="ATR":
                         if not col_name:
                            self.logger.debug(f"Internal key '{key}' did not map to a column name during calculation for {self.symbol}.")
                         elif col_name not in latest.index:
                            self.logger.debug(f"Column '{col_name}' for indicator '{key}' not found in DataFrame for {self.symbol}.")
                    updated_values[key] = np.nan # Ensure key exists even if missing/NaN

            # Add essential price/volume data from the latest candle (original columns)
            for base_col in ['close', 'volume', 'high', 'low']:
                 value = latest.get(base_col, np.nan)
                 if pd.notna(value):
                      try:
                           updated_values[base_col.capitalize()] = float(value) # Store as Close, Volume etc.
                      except (ValueError, TypeError):
                           updated_values[base_col.capitalize()] = np.nan
                 else:
                      updated_values[base_col.capitalize()] = np.nan


            self.indicator_values = updated_values
            # Filter out NaN for debug log brevity
            valid_values = {k: v for k, v in self.indicator_values.items() if pd.notna(v)}
            self.logger.debug(f"Latest indicator values updated for {self.symbol}: {valid_values}")

        except IndexError:
             self.logger.error(f"Error accessing latest row (iloc[-1]) for {self.symbol}. DataFrame might be unexpectedly empty.")
             all_potential_keys = list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]
             self.indicator_values = {k: np.nan for k in all_potential_keys} # Reset values
        except Exception as e:
             self.logger.error(f"Unexpected error updating latest indicator values for {self.symbol}: {e}", exc_info=True)
             all_potential_keys = list(self.ta_column_names.keys()) + ["Close", "Volume", "High", "Low"]
             self.indicator_values = {k: np.nan for k in all_potential_keys} # Reset values


    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(self, window: Optional[int] = None) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.warning(f"Not enough data ({len(self.df)}) for Fibonacci window ({window}) on {self.symbol}.")
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            # Ensure we handle potential NaNs in the slice before finding max/min
            high_price = df_slice["high"].dropna().max()
            low_price = df_slice["low"].dropna().min()

            if pd.isna(high_price) or pd.isna(low_price):
                 self.logger.warning(f"Could not find valid high/low in the last {window} periods for Fibonacci on {self.symbol}.")
                 self.fib_levels_data = {}
                 return {}

            high = Decimal(str(high_price))
            low = Decimal(str(low_price))
            diff = high - low

            levels = {}
            if diff > 0:
                # Use market price precision for formatting levels
                price_precision = self.get_price_precision()
                rounding_factor = Decimal('1e-' + str(price_precision))

                for level in FIB_LEVELS:
                    level_name = f"Fib_{level * 100:.1f}%"
                    # Calculate and quantize the level
                    level_price = (high - (diff * Decimal(str(level)))).quantize(rounding_factor, rounding=ROUND_DOWN)
                    levels[level_name] = level_price
            else:
                 self.logger.debug(f"Fibonacci range is zero or negative (High={high}, Low={low}) for {self.symbol} in window {window}.")

            self.fib_levels_data = levels
            self.logger.debug(f"Calculated Fibonacci levels for {self.symbol}: {levels}")
            return levels
        except Exception as e:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}", exc_info=True)
            self.fib_levels_data = {}
            return {}

    def get_price_precision(self) -> int:
        """Get price precision (number of decimal places) from market info."""
        try:
            precision_val = self.market_info.get('precision', {}).get('price')
            if precision_val is not None:
                 # CCXT precision can be number of decimals (int) or tick size (float)
                 if isinstance(precision_val, int):
                      return precision_val
                 elif isinstance(precision_val, (float, str)) and Decimal(str(precision_val)) > 0:
                      # If it's the tick size (e.g., 0.01), calculate decimal places
                      return abs(Decimal(str(precision_val)).normalize().as_tuple().exponent)

            # Fallback if market info doesn't provide it easily: Use last close price format
            last_close = self.indicator_values.get("Close")
            if last_close and not pd.isna(last_close):
                 s_close = str(Decimal(str(last_close)))
                 if '.' in s_close:
                     return len(s_close.split('.')[-1])

        except Exception as e:
            self.logger.warning(f"Could not reliably determine price precision for {self.symbol} from market info/price: {e}. Falling back to default.")

        # Default fallback precision (adjust as needed for your exchange/market)
        return self.market_info.get('precision', {}).get('base', 6) # Or a fixed default like 6


    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 5
    ) -> list[Tuple[str, Decimal]]:
        """Find nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
             # self.calculate_fibonacci_levels() # Attempt calculation if not already done - maybe too slow here
             if not self.fib_levels_data: # Check again
                  return [] # Return empty if calculation failed or not run

        if current_price is None or not isinstance(current_price, Decimal) or pd.isna(current_price):
            self.logger.warning(f"Invalid current price ({current_price}) for Fibonacci comparison on {self.symbol}.")
            return []

        try:
            level_distances = []
            for name, level in self.fib_levels_data.items():
                if isinstance(level, Decimal): # Ensure level is Decimal
                    distance = abs(current_price - level)
                    level_distances.append((name, level, distance))
                else:
                     self.logger.warning(f"Non-decimal value found in fib_levels_data for {self.symbol}: {name}={level}")

            level_distances.sort(key=lambda x: x[2]) # Sort by distance
            return [(name, level) for name, level, _ in level_distances[:num_levels]]
        except Exception as e:
            self.logger.error(f"{NEON_RED}Error finding nearest Fibonacci levels for {self.symbol}: {e}{RESET}", exc_info=True)
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculate EMA alignment score based on latest values."""
        # Required indicators: EMA_Short, EMA_Long, Close
        ema_short = self.indicator_values.get("EMA_Short", np.nan)
        ema_long = self.indicator_values.get("EMA_Long", np.nan)
        current_price = self.indicator_values.get("Close", np.nan)

        if pd.isna(ema_short) or pd.isna(ema_long) or pd.isna(current_price):
            # self.logger.debug("EMA alignment check skipped: Missing required values.")
            return np.nan # Return NaN if data is missing

        # Bullish alignment: Price > Short EMA > Long EMA
        if current_price > ema_short > ema_long: return 1.0
        # Bearish alignment: Price < Short EMA < Long EMA
        elif current_price < ema_short < ema_long: return -1.0
        # Neutral or mixed alignment
        else: return 0.0

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: Optional[Dict]
    ) -> str:
        """Generate trading signal based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 0} # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0

        # Check if essential data is present
        if not self.indicator_values or all(pd.isna(v) for k,v in self.indicator_values.items() if k not in ['Close', 'Volume', 'High', 'Low']):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Indicator values not calculated or all NaN.{RESET}")
             return "HOLD"
        if pd.isna(current_price):
             self.logger.warning(f"{NEON_YELLOW}Cannot generate signal for {self.symbol}: Missing current price.{RESET}")
             return "HOLD"

        # Iterate through configured indicators and calculate weighted scores
        # Use the weights from the active set
        active_weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        if not active_weights:
             self.logger.error(f"Active weight set '{self.active_weight_set_name}' is empty or missing. Cannot generate signal.")
             return "HOLD"

        for indicator_key, enabled in self.config.get("indicators", {}).items():
            # Check if indicator is enabled AND has a weight defined in the active set
            if enabled and indicator_key in active_weights:
                weight = Decimal(str(active_weights[indicator_key]))
                if weight == 0: continue # Skip if weight is zero

                # Find the corresponding check method (e.g., _check_rsi for "rsi")
                check_method_name = f"_check_{indicator_key}"
                if hasattr(self, check_method_name):
                    method = getattr(self, check_method_name)
                    try:
                        # Call the check method - most use self.indicator_values now
                        # Pass orderbook data specifically if method needs it
                        if indicator_key == "orderbook":
                             indicator_score = method(orderbook_data, current_price) if orderbook_data else np.nan
                        else:
                             indicator_score = method() # Returns score between -1.0 and 1.0, or np.nan
                    except Exception as e:
                        self.logger.error(f"Error calling check method {check_method_name} for {self.symbol}: {e}", exc_info=True)
                        indicator_score = np.nan

                    # Process the score
                    if pd.notna(indicator_score):
                        score_contribution = Decimal(str(indicator_score)) * weight
                        final_signal_score += score_contribution
                        total_weight_applied += weight
                        active_indicator_count += 1
                        self.logger.debug(f"  Indicator {indicator_key:<20}: Score={indicator_score:+.2f}, Weight={weight:.2f}, Contrib={score_contribution:+.3f}")
                    else:
                        nan_indicator_count += 1
                        self.logger.debug(f"  Indicator {indicator_key:<20}: Score=NaN (Skipped)")
                else:
                    self.logger.warning(f"No check method found for enabled/weighted indicator: {indicator_key} ({self.symbol})")


        # --- Determine Final Signal ---
        if total_weight_applied == 0:
             self.logger.warning(f"No indicators contributed to the signal score for {self.symbol} (Total Weight Applied = 0). Defaulting to HOLD.")
             return "HOLD"

        # Normalize score? Optional: Divide by total_weight_applied
        # normalized_score = final_signal_score / total_weight_applied
        # Using raw score against threshold is simpler:
        threshold = Decimal(str(self.config.get("signal_score_threshold", 1.5)))

        log_msg = (
            f"Signal Calculation for {self.symbol}:\n"
            f"  Active Weight Set: {self.active_weight_set_name}\n"
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Weighted Score: {final_signal_score:.4f}\n"
            f"  Signal Threshold: +/- {threshold:.3f}"
        )
        # Use INFO level for the summary score and threshold
        self.logger.info(log_msg)


        if final_signal_score >= threshold:
            self.signals["BUY"] = 1
            return "BUY"
        elif final_signal_score <= -threshold:
            self.signals["SELL"] = 1
            return "SELL"
        else:
            self.signals["HOLD"] = 1
            return "HOLD"


    # --- Check methods using self.indicator_values ---
    # Each method should return a score between -1.0 and 1.0, or np.nan if data invalid

    def _check_ema_alignment(self) -> float:
        """Check based on calculate_ema_alignment_score."""
        # Ensure the indicator was calculated if this check is enabled
        if pd.isna(self.indicator_values.get("EMA_Short")) or pd.isna(self.indicator_values.get("EMA_Long")):
             self.logger.debug(f"EMA Alignment check skipped for {self.symbol}: EMAs not calculated.")
             return np.nan
        return self.calculate_ema_alignment_score() # Returns score or NaN

    def _check_momentum(self) -> float:
        """Check Momentum indicator."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        if pd.isna(momentum): return np.nan
        # Simple thresholding (can be refined based on typical values for the asset/interval)
        if momentum > 0.1: return 1.0 # Strong positive momentum
        if momentum < -0.1: return -1.0 # Strong negative momentum
        if momentum > 0.01: return 0.3 # Weak positive
        if momentum < -0.01: return -0.3 # Weak negative
        return 0.0

    def _check_volume_confirmation(self) -> float:
        """Check if current volume confirms price direction (relative to MA)."""
        current_volume = self.indicator_values.get("Volume", np.nan)
        volume_ma = self.indicator_values.get("Volume_MA", np.nan)
        multiplier = Decimal(str(self.config.get("volume_confirmation_multiplier", 1.5)))

        if pd.isna(current_volume) or pd.isna(volume_ma) or volume_ma <= 0: # Check MA > 0
            return np.nan

        try:
            # High volume suggests confirmation/strength
            dec_current_volume = Decimal(str(current_volume))
            dec_volume_ma = Decimal(str(volume_ma))

            if dec_current_volume > dec_volume_ma * multiplier:
                 # Positive score indicates *potential* strength or confirmation.
                 # The direction comes from other trend/momentum indicators.
                 return 0.6 # Positive score for high volume
            # Low volume might indicate lack of conviction or potential exhaustion
            elif dec_current_volume < dec_volume_ma / multiplier:
                 return -0.3 # Slight negative score for low volume
            else:
                 return 0.0 # Neutral volume
        except Exception as e:
            self.logger.warning(f"{NEON_YELLOW}Volume confirmation check failed for {self.symbol}: {e}{RESET}")
            return np.nan

    def _check_stoch_rsi(self) -> float:
        """Check Stochastic RSI K and D lines."""
        k = self.indicator_values.get("StochRSI_K", np.nan)
        d = self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d): return np.nan

        oversold = self.config.get("stoch_rsi_oversold_threshold", 25)
        overbought = self.config.get("stoch_rsi_overbought_threshold", 75)

        # Strong signals at extremes with confirmation (both lines in zone)
        if k < oversold and d < oversold: return 1.0 # Oversold, potential bounce UP
        if k > overbought and d > overbought: return -1.0 # Overbought, potential drop DOWN

        # Consider K vs D relationship for momentum within StochRSI
        if k > d: # K above D -> Bullish momentum / potential upward cross
            if k < 50: return 0.6 # Bullish momentum in lower half
            else: return 0.3 # Bullish momentum in upper half (less strong signal)
        elif k < d: # K below D -> Bearish momentum / potential downward cross
            if k > 50: return -0.6 # Bearish momentum in upper half
            else: return -0.3 # Bearish momentum in lower half (less strong signal)
        else:
            return 0.0 # Lines might be equal or crossing exactly

    def _check_rsi(self) -> float:
        """Check RSI indicator."""
        rsi = self.indicator_values.get("RSI", np.nan)
        if pd.isna(rsi): return np.nan
        # More granular scoring based on RSI level
        if rsi <= 30: return 1.0  # Strong Oversold -> Buy Signal
        if rsi >= 70: return -1.0 # Strong Overbought -> Sell Signal
        if rsi < 40: return 0.5  # Mildly Oversold / Leaning Bullish
        if rsi > 60: return -0.5 # Mildly Overbought / Leaning Bearish
        if 45 <= rsi <= 55: return 0.0 # Neutral Zone
        if 55 < rsi < 70: return -0.2 # Upper Neutral / Weak Bearish
        if 30 < rsi < 45: return 0.2 # Lower Neutral / Weak Bullish
        return 0.0 # Fallback neutral

    def _check_cci(self) -> float:
        """Check CCI indicator."""
        cci = self.indicator_values.get("CCI", np.nan)
        if pd.isna(cci): return np.nan
        # CCI extremes often indicate reversals
        if cci <= -100: return 1.0 # Extreme Oversold -> Buy Signal
        if cci >= 100: return -1.0 # Extreme Overbought -> Sell Signal
        if cci < -50: return 0.4 # Moderately Oversold / Leaning Bullish
        if cci > 50: return -0.4 # Moderately Overbought / Leaning Bearish
        if cci > 0: return -0.1 # Above zero line (slight bearish pressure if falling from high?)
        if cci < 0: return 0.1 # Below zero line (slight bullish pressure if rising from low?)
        return 0.0

    def _check_wr(self) -> float: # Williams %R
        """Check Williams %R indicator."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        if pd.isna(wr): return np.nan
        # Note: WR ranges from -100 (most oversold) to 0 (most overbought).
        if wr <= -80: return 1.0 # Oversold -> Buy Signal
        if wr >= -20: return -1.0 # Overbought -> Sell Signal
        if wr < -50: return 0.4 # Lower half (more oversold) -> Leaning Bullish
        if wr > -50: return -0.4 # Upper half (more overbought) -> Leaning Bearish
        return 0.0

    def _check_psar(self) -> float:
        """Check Parabolic SAR indicator relative to price."""
        psar_l = self.indicator_values.get("PSAR_long", np.nan) # Value when PSAR is below price (support)
        psar_s = self.indicator_values.get("PSAR_short", np.nan) # Value when PSAR is above price (resistance)
        last_close = self.indicator_values.get("Close", np.nan)

        if pd.isna(last_close): return np.nan

        # Determine current trend based on which PSAR value is NOT NaN
        is_uptrend = pd.notna(psar_l) and pd.isna(psar_s)
        is_downtrend = pd.notna(psar_s) and pd.isna(psar_l)

        if is_uptrend:
            # PSAR is below price, confirming uptrend
            return 1.0 # Strong bullish signal
        elif is_downtrend:
            # PSAR is above price, confirming downtrend
            return -1.0 # Strong bearish signal
        else:
            # Could be a reversal point or NaN values at start of data
            # A reversal is implicitly handled when the trend flips on the next candle
            return 0.0 # Neutral or undetermined state

    def _check_sma_10(self) -> float: # Example using SMA10
        """Check price relative to SMA10."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        
