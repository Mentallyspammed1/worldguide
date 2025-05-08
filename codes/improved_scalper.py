# livexx_improved.py
# Enhanced version focusing on robust order placement and TSL integration.
# Based on user-provided livexx.py, with cleanup and best practice adjustments.

import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo  # Modern timezone handling

# Third-Party Imports
import ccxt  # Exchange interaction library
import numpy as np  # Numerical operations, used for NaN and jitter
import pandas as pd  # Data manipulation and analysis
import pandas_ta as ta  # Technical analysis library built on pandas
import requests  # Used by ccxt for HTTP requests
from colorama import Fore, Style, init  # Colored terminal output
from dotenv import load_dotenv  # Loading environment variables
from requests.adapters import HTTPAdapter  # For retry logic in ccxt session
from urllib3.util.retry import Retry  # For retry logic in ccxt session

# --- Initialization ---
init(autoreset=True)  # Ensure colorama resets styles automatically
load_dotenv()  # Load environment variables from .env file (e.g., API keys)

# Set Decimal precision
# 18 is often sufficient for price/qty, adjust if needed for specific calcs
getcontext().prec = 18

# Neon Color Scheme (from livexx.py)
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Constants ---
# API Credentials (Load from .env)
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use logging if available, otherwise print before exiting
    logging.basicConfig(level=logging.ERROR)  # Basic config for critical error
    logging.critical("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")
    raise ValueError("API Keys not found in .env file.")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)  # Ensure log directory exists

# Timezone for logging and display (adjust in .env or config if needed)
# Defaulting to Chicago as seen in livexx.py
DEFAULT_TIMEZONE = "America/Chicago"
try:
    # Allow override from environment variable if needed
    TIMEZONE_STR = os.getenv("BOT_TIMEZONE", DEFAULT_TIMEZONE)
    TIMEZONE = ZoneInfo(TIMEZONE_STR)
except Exception:
    logging.warning(f"Could not load timezone '{TIMEZONE_STR}', using '{DEFAULT_TIMEZONE}'.")
    TIMEZONE = ZoneInfo(DEFAULT_TIMEZONE)

# Global config placeholder - recommend passing explicitly where possible
CONFIG: dict[str, Any] = {}
QUOTE_CURRENCY: str = "USDT"  # Default, updated after config load


# --- Logging Setup ---
class SensitiveFormatter(logging.Formatter):
    """Formatter that removes sensitive information in logs."""
    @staticmethod
    def _filter(log_record):
        # Basic redaction example, refine as needed
        if API_KEY: log_record.msg = str(log_record.msg).replace(API_KEY, "***API_KEY***")
        if API_SECRET: log_record.msg = str(log_record.msg).replace(API_SECRET, "***API_SECRET***")
        # Add other sensitive info redaction if necessary
        return log_record

    def format(self, record):
        record = self._filter(record)
        return super().format(record)


def setup_logger(name: str, level=logging.INFO, add_console=True) -> None:
    """Sets up file and optional console logging for a specific module/symbol."""
    log_file = os.path.join(LOG_DIRECTORY, f"{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs in root logger

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (UTC Timestamps)
    file_formatter = SensitiveFormatter(
        fmt='%(asctime)s.%(msecs)03dZ [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
    )
    file_formatter.converter = time.gmtime  # Use UTC for file logs
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler (Local Timestamps, optional)
    if add_console:
        console_formatter = SensitiveFormatter(
             # Use local time for console
            fmt=f"%(asctime)s [{NEON_BLUE}%(levelname)-8s{RESET}] [{NEON_PURPLE}{name}{RESET}] %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S'  # Local time format
        )
        # Note: Formatter doesn't directly know about TIMEZONE, asctime uses local system time by default.
        # For true timezone-aware console logging, manual formatting might be needed.
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        # Set console level potentially higher based on config?
        console_handler.setLevel(level)
        logger.addHandler(console_handler)


# --- Configuration Loading ---
def load_config(filepath: str) -> dict[str, Any]:
    """Loads configuration from JSON file, applies defaults, and merges."""
    logger = logging.getLogger("init")  # Use init logger during setup
    defaults = {
        "exchange_id": "bybit",
        "use_sandbox": True,
        "enable_trading": False,
        "symbols": ["BTC/USDT:USDT"],
        "quote_currency": "USDT",
        "logging": {
            "level": "INFO",
            "max_file_size_mb": 5,
            "backup_count": 5
        },
        "trading": {
            "interval": "1m",
            "ohlcv_limit": 200,
            "loop_interval_seconds": 10,
            "error_retry_delay": 60,
            "max_leverage": 10,
            "active_weight_set": "default",
            "max_order_retries": 3,
            "initial_order_delay": 0.5,  # Delay between placing order and setting protection
            "max_api_retries": 5,
            "api_initial_delay": 1,
            "api_backoff_factor": 2
        },
        "risk": {
            "risk_per_trade": 0.005,  # 0.5%
            "max_total_risk": 0.1,  # Max 10% of balance at risk across all positions (Example)
            "required_margin_buffer": 1.05  # 5% margin buffer
        },
        "protection": {  # Parameters for SL/TP/TSL/BE
            "use_position_protection": True,
            "sl_atr_multiplier": 1.5,
            "tp_atr_multiplier": 2.0,  # Fixed TP (Consider if TSL is active)
            "tpsl_mode": "Full",  # "Full" or "Partial"
            "tp_order_type": "Market",  # Only Market for Full mode
            "sl_order_type": "Market",
            "tp_trigger_by": "LastPrice",  # MarkPrice, IndexPrice, LastPrice
            "sl_trigger_by": "LastPrice",
            "enable_trailing_stop": False,
            "tsl_activation_atr_multiplier": 0.5,  # Activate TSL when price moves 0.5 * ATR in profit
            "tsl_distance_atr_multiplier": 1.0,  # Trail by 1.0 * ATR
            "tsl_trigger_by": "LastPrice",
            "enable_break_even": False,
            "be_activation_atr_multiplier": 1.0,  # Activate BE when price moves 1.0 * ATR in profit
            "be_offset_pips": 2  # Move SL slightly into profit (in quote currency units, adjust based on tick size)
        },
        "indicators": {  # Default parameters for indicators
             "atr": {"length": 14},
             "ema_short": {"length": 9},
             "ema_medium": {"length": 21},
             "ema_long": {"length": 50},
             "rsi": {"length": 14, "buy_threshold": 30, "sell_threshold": 70},
             "stochrsi": {"length": 14, "k": 3, "d": 3, "buy_threshold": 20, "sell_threshold": 80},
             "macd": {"fast": 12, "slow": 26, "signal": 9},
             "volume_profile": {"enabled": False, "atr_multiplier": 1.5},
             "orderbook_imbalance": {"enabled": False, "levels": 5, "threshold": 1.5}
        },
        "weight_sets": {  # Default scoring weights
            "default": {
                "ema_alignment_score": 0.25,
                "rsi_score": 0.15,
                "stochrsi_score": 0.15,
                "macd_score": 0.15,
                "volume_profile_score": 0.15,  # Requires enabled: true
                "orderbook_imbalance_score": 0.15,  # Requires enabled: true
                # Add weights for other indicators as implemented
                "buy_threshold": 0.6,  # Score threshold to enter long
                "sell_threshold": -0.6  # Score threshold to enter short
            }
        },
        "shutdown": {
             "close_open_positions": False,  # Default safety: do not close on exit
             "cancel_open_orders": True
        }
        # Add other sections as needed
    }

    try:
        with open(filepath, encoding='utf-8') as f:
            user_config = json.load(f)
        logger.info(f"Successfully loaded user config from {filepath}")

        # Basic recursive merge (user config overrides defaults)
        def merge_configs(default, user):
            if isinstance(default, dict) and isinstance(user, dict):
                merged = default.copy()
                for key, value in user.items():
                    if key in merged:
                        merged[key] = merge_configs(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            return user  # User value overrides if not a dict merge

        merged_config = merge_configs(defaults, user_config)
        # Potential: Add Pydantic validation here using merged_config
        return merged_config

    except FileNotFoundError:
        logger.warning(f"Config file not found at {filepath}. Using default settings.")
        return defaults
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}. Using default settings.")
        return defaults
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}. Using default settings.", exc_info=True)
        return defaults

# --- Utility / Helper Functions ---


def safe_api_call(func, *args, logger: logging.Logger, max_retries: int = 5, initial_delay: float = 1.0, backoff_factor: float = 2.0, **kwargs) -> Any | None:
    """Wraps an API call with retries and exponential backoff for specific errors."""
    retryable_errors = (
        ccxt.RequestTimeout,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.OnMaintenance,
        ccxt.RateLimitExceeded,
        ccxt.ExchangeError,  # Includes many potentially temporary issues
        requests.exceptions.RequestException,  # Catch potential underlying requests errors
    )
    non_retryable_errors = (
        ccxt.AuthenticationError,
        ccxt.PermissionDenied,
        ccxt.AccountSuspended,
        ccxt.InsufficientFunds,
        ccxt.InvalidOrder,
        ccxt.OrderNotFound,
        ccxt.NotSupported,
        ccxt.BadSymbol,
        # Add other errors that should NOT be retried
    )

    for attempt in range(max_retries):
        try:
            logger.debug(f"API Call Attempt {attempt + 1}/{max_retries}: {func.__name__} Args: {args} Kwargs: {kwargs}")
            result = func(*args, **kwargs)
            logger.debug(f"API Call Success: {func.__name__}")
            return result
        except non_retryable_errors as e:
            logger.error(f"{NEON_RED}Non-retryable API Error calling {func.__name__}: {type(e).__name__} - {e}{RESET}", exc_info=False)  # Less noise for expected fails like InsufficientFunds
            # Depending on severity, could raise, return None, or trigger an alert
            # Re-raise specific important ones if needed by caller
            if isinstance(e, (ccxt.AuthenticationError, ccxt.PermissionDenied, ccxt.AccountSuspended)):
                 raise e  # Fatal errors, should likely stop the bot
            return None  # For errors like InsufficientFunds, InvalidOrder, let caller handle None
        except retryable_errors as e:
            logger.warning(f"{NEON_YELLOW}Retryable API Error (Attempt {attempt + 1}/{max_retries}) calling {func.__name__}: {type(e).__name__} - {e}{RESET}")
            if attempt == max_retries - 1:
                logger.error(f"{NEON_RED}API Call failed after {max_retries} attempts: {func.__name__}{RESET}", exc_info=True)
                return None  # Failed after retries
            # Apply exponential backoff with jitter
            current_delay = initial_delay * (backoff_factor ** attempt)
            jitter = current_delay * 0.1  # Add up to +/- 10% jitter
            sleep_time = max(0.1, current_delay + np.random.uniform(-jitter, jitter))  # Ensure min sleep
            logger.info(f"Retrying {func.__name__} in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        except Exception as e:  # Catch any other unexpected CCXT or other errors
            logger.critical(f"{NEON_RED}Unexpected critical error in safe_api_call calling {func.__name__}: {type(e).__name__} - {e}{RESET}", exc_info=True)
            raise e  # Re-raise unknown critical errors

    return None  # Should only be reached if retries failed


# --- CCXT Exchange Initialization ---
def get_exchange_session(config: dict[str, Any]) -> requests.Session:
    """Creates a requests session with retry logic for CCXT."""
    session = requests.Session()
    retries = config.get('trading', {}).get('max_api_retries', 5)
    backoff_factor = config.get('trading', {}).get('api_backoff_factor', 2)
    status_forcelist = (500, 502, 503, 504)  # Status codes to retry on
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],  # Retry on all relevant methods
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def initialize_exchange(config: dict[str, Any]) -> ccxt.Exchange | None:
    """Initializes and configures the CCXT exchange instance."""
    logger = logging.getLogger("init")
    exchange_id = config.get("exchange_id", "bybit")
    use_sandbox = config.get("use_sandbox", True)
    logger.info(f"Initializing exchange: {exchange_id} (Sandbox: {use_sandbox})")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange_params = {
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,  # Let ccxt handle basic rate limiting
             # 'session': get_exchange_session(config), # Inject session with retry logic - uncomment if needed
            'options': {
                'defaultType': 'linear',  # Assume linear contracts unless overridden
                'adjustForTimeDifference': True,
            }
        }
        exchange = exchange_class(exchange_params)

        if use_sandbox:
            exchange.set_sandbox_mode(True)
            logger.info("Sandbox mode enabled.")

        # Test connection and load markets
        logger.info("Loading markets...")
        markets = safe_api_call(exchange.load_markets, logger=logger)
        if markets is None:
             logger.critical("Failed to load markets after retries.")
             return None
        logger.info(f"Markets loaded successfully for {exchange_id}.")

        # Check server time difference
        server_time = safe_api_call(exchange.fetch_time, logger=logger)
        if server_time:
            time_diff = abs(server_time - exchange.milliseconds())
            logger.info(f"Server time difference: {time_diff} ms")
            if time_diff > 5000:  # Warn if diff > 5 seconds
                logger.warning(f"{NEON_YELLOW}Server time difference is high ({time_diff} ms). Check system clock synchronization.{RESET}")
        else:
             logger.warning("Could not fetch server time to check difference.")

        # Set User-Agent
        exchange.options['user-agent'] = f"ScalpXX-Bot/1.0 ({exchange_id})"

        return exchange

    except AttributeError:
        logger.critical(f"Exchange '{exchange_id}' not found in CCXT.")
        return None
    except ccxt.AuthenticationError as e:
         logger.critical(f"Authentication failed: {e}. Check API Key/Secret.")
         return None
    except Exception as e:
        logger.critical(f"Error initializing exchange: {e}", exc_info=True)
        return None


# --- Data Handling ---
def create_dataframe(ohlcv_data: list[list[int | float]]) -> pd.DataFrame:
    """Converts CCXT OHLCV data to a Pandas DataFrame."""
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)  # Ensure UTC
    df = df.set_index('timestamp')
    # Convert columns to appropriate types (Decimal for price/volume)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].apply(lambda x: Decimal(str(x)) if x is not None else Decimal('NaN'))
    df = df.sort_index()  # Ensure data is sorted by time
    return df


# --- Core Logic Class ---
class TradingAnalyzer:
    """Calculates indicators and generates trading signals based on weighted scores."""
    def __init__(self, df: pd.DataFrame, config: dict[str, Any], market_info: dict, logger: logging.Logger) -> None:
        self.df = df.copy()  # Work on a copy
        self.config = config
        self.market_info = market_info
        self.logger = logger
        self._indicator_params = config.get("indicators", {})
        self._protection_params = config.get("protection", {})
        self.atr = Decimal('NaN')  # Store calculated ATR

    def _calculate_atr(self) -> None:
        """Calculates Average True Range (ATR)."""
        params = self._indicator_params.get("atr", {"length": 14})
        try:
            atr_series = ta.atr(self.df['high'].astype(float),  # pandas-ta often needs float
                                self.df['low'].astype(float),
                                self.df['close'].astype(float),
                                length=params.get("length", 14))
            if atr_series is not None and not atr_series.empty:
                self.df['ATR'] = atr_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                self.atr = self.df['ATR'].iloc[-1]
                self.logger.debug(f"Calculated ATR({params.get('length', 14)}): {self.atr:.{self.market_info.get('precision', {}).get('price', 2)}f}")
            else:
                 self.logger.warning("ATR calculation returned None or empty series.")
                 self.df['ATR'] = Decimal('NaN')
                 self.atr = Decimal('NaN')
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}", exc_info=True)
            self.df['ATR'] = Decimal('NaN')
            self.atr = Decimal('NaN')

    def _calculate_ema(self) -> None:
        """Calculates multiple EMAs based on config."""
        ema_configs = {k: v for k, v in self._indicator_params.items() if k.startswith("ema_")}
        for name, params in ema_configs.items():
            length = params.get("length")
            if length:
                try:
                    ema_series = ta.ema(self.df['close'].astype(float), length=length)
                    if ema_series is not None:
                        self.df[name.upper()] = ema_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                        self.logger.debug(f"Calculated {name.upper()}({length})")
                    else:
                         self.df[name.upper()] = Decimal('NaN')
                         self.logger.warning(f"{name.upper()} calculation returned None.")
                except Exception as e:
                    self.logger.error(f"Error calculating {name.upper()}: {e}", exc_info=True)
                    self.df[name.upper()] = Decimal('NaN')

    def _calculate_rsi(self) -> None:
        """Calculates Relative Strength Index (RSI)."""
        params = self._indicator_params.get("rsi", {})
        length = params.get("length", 14)
        try:
            rsi_series = ta.rsi(self.df['close'].astype(float), length=length)
            if rsi_series is not None:
                self.df['RSI'] = rsi_series.apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                self.logger.debug(f"Calculated RSI({length})")
            else:
                 self.df['RSI'] = Decimal('NaN')
                 self.logger.warning("RSI calculation returned None.")
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}", exc_info=True)
            self.df['RSI'] = Decimal('NaN')

    def _calculate_stochrsi(self) -> None:
        """Calculates Stochastic RSI."""
        params = self._indicator_params.get("stochrsi", {})
        length = params.get("length", 14)
        k = params.get("k", 3)
        d = params.get("d", 3)
        try:
            stochrsi_df = ta.stochrsi(self.df['close'].astype(float), length=length, rsi_length=length, k=k, d=d)  # Use same length for rsi_length for standard stochrsi
            if stochrsi_df is not None and not stochrsi_df.empty:
                # Column names might be like 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3'
                k_col = next((col for col in stochrsi_df.columns if 'k_' in col), None)
                d_col = next((col for col in stochrsi_df.columns if 'd_' in col), None)
                if k_col: self.df['STOCHRSI_K'] = stochrsi_df[k_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                if d_col: self.df['STOCHRSI_D'] = stochrsi_df[d_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                self.logger.debug(f"Calculated StochRSI({length},{k},{d})")
            else:
                 self.df['STOCHRSI_K'] = Decimal('NaN')
                 self.df['STOCHRSI_D'] = Decimal('NaN')
                 self.logger.warning("StochRSI calculation returned None or empty.")
        except Exception as e:
            self.logger.error(f"Error calculating StochRSI: {e}", exc_info=True)
            self.df['STOCHRSI_K'] = Decimal('NaN')
            self.df['STOCHRSI_D'] = Decimal('NaN')

    def _calculate_macd(self) -> None:
        """Calculates MACD."""
        params = self._indicator_params.get("macd", {})
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        try:
            macd_df = ta.macd(self.df['close'].astype(float), fast=fast, slow=slow, signal=signal)
            if macd_df is not None and not macd_df.empty:
                 # Columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                 macd_line_col = next((col for col in macd_df.columns if col.startswith('MACD_') and not col.endswith('h') and not col.endswith('s')), None)
                 macd_signal_col = next((col for col in macd_df.columns if col.endswith('s')), None)
                 macd_hist_col = next((col for col in macd_df.columns if col.endswith('h')), None)

                 if macd_line_col: self.df['MACD_line'] = macd_df[macd_line_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                 if macd_signal_col: self.df['MACD_signal'] = macd_df[macd_signal_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                 if macd_hist_col: self.df['MACD_hist'] = macd_df[macd_hist_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('NaN'))
                 self.logger.debug(f"Calculated MACD({fast},{slow},{signal})")
            else:
                 self.df['MACD_line'] = Decimal('NaN')
                 self.df['MACD_signal'] = Decimal('NaN')
                 self.df['MACD_hist'] = Decimal('NaN')
                 self.logger.warning("MACD calculation returned None or empty.")
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}", exc_info=True)
            self.df['MACD_line'] = Decimal('NaN')
            self.df['MACD_signal'] = Decimal('NaN')
            self.df['MACD_hist'] = Decimal('NaN')

    # --- Add other indicator calculation methods here (_calculate_volume_profile, _calculate_orderbook_imbalance, etc.) ---
    # Ensure they handle errors gracefully and assign Decimal('NaN') on failure.

    def _determine_final_signal(self, score: Decimal, weights: dict) -> str:
        """Determines final signal based on score and thresholds."""
        buy_threshold = Decimal(str(weights.get("buy_threshold", 0.6)))
        sell_threshold = Decimal(str(weights.get("sell_threshold", -0.6)))

        if score >= buy_threshold:
            return "LONG"
        elif score <= sell_threshold:
            return "SHORT"
        else:
            return "NEUTRAL"

    def generate_trading_signal(self, weight_set_name: str) -> tuple[str, float, dict]:
        """Calculates all indicators and generates a final weighted signal."""
        self.logger.debug(f"Generating signal using weight set: {weight_set_name}")

        # --- Calculate Indicators ---
        self._calculate_atr()
        self._calculate_ema()
        self._calculate_rsi()
        self._calculate_stochrsi()
        self._calculate_macd()
        # Call other indicator calculations here...

        # --- Scoring Logic ---
        if self.df.empty:
             self.logger.warning("DataFrame is empty, cannot generate signal.")
             return "NEUTRAL", 0.0, {}

        try:
            latest_data = self.df.iloc[-1]  # Get the most recent data row
            weights = self.config.get("weight_sets", {}).get(weight_set_name, {})
            if not weights:
                 self.logger.error(f"Weight set '{weight_set_name}' not found or empty in config.")
                 return "NEUTRAL", 0.0, {}

            total_score = Decimal('0.0')
            score_breakdown = {}  # For logging/debugging

            # EMA Alignment Score
            ema_alignment_weight = Decimal(str(weights.get("ema_alignment_score", 0.0)))
            if ema_alignment_weight != Decimal('0'):
                ema_short = latest_data.get('EMA_SHORT', Decimal('NaN'))
                ema_medium = latest_data.get('EMA_MEDIUM', Decimal('NaN'))
                ema_long = latest_data.get('EMA_LONG', Decimal('NaN'))
                contribution = Decimal('0.0')
                if not any(d.is_nan() for d in [ema_short, ema_medium, ema_long]):
                    if ema_short > ema_medium > ema_long:  # Bullish alignment
                        contribution = ema_alignment_weight
                    elif ema_short < ema_medium < ema_long:  # Bearish alignment
                        contribution = -ema_alignment_weight
                total_score += contribution
                score_breakdown['EMA'] = f"{contribution:.4f} (W:{ema_alignment_weight})"

            # RSI Score
            rsi_weight = Decimal(str(weights.get("rsi_score", 0.0)))
            if rsi_weight != Decimal('0'):
                rsi_value = latest_data.get('RSI', Decimal('NaN'))
                buy_thresh = Decimal(str(self._indicator_params.get("rsi", {}).get("buy_threshold", 30)))
                sell_thresh = Decimal(str(self._indicator_params.get("rsi", {}).get("sell_threshold", 70)))
                contribution = Decimal('0.0')
                if not rsi_value.is_nan():
                    if rsi_value < buy_thresh: contribution = rsi_weight  # Oversold -> bullish signal
                    elif rsi_value > sell_thresh: contribution = -rsi_weight  # Overbought -> bearish signal
                total_score += contribution
                score_breakdown['RSI'] = f"{contribution:.4f} (Val:{rsi_value:.2f}, W:{rsi_weight})"

            # StochRSI Score
            stochrsi_weight = Decimal(str(weights.get("stochrsi_score", 0.0)))
            if stochrsi_weight != Decimal('0'):
                k = latest_data.get('STOCHRSI_K', Decimal('NaN'))
                # d = latest_data.get('STOCHRSI_D', Decimal('NaN')) # Could use K, D, or crossover
                buy_thresh = Decimal(str(self._indicator_params.get("stochrsi", {}).get("buy_threshold", 20)))
                sell_thresh = Decimal(str(self._indicator_params.get("stochrsi", {}).get("sell_threshold", 80)))
                contribution = Decimal('0.0')
                if not k.is_nan():
                    if k < buy_thresh: contribution = stochrsi_weight  # Oversold -> bullish
                    elif k > sell_thresh: contribution = -stochrsi_weight  # Overbought -> bearish
                total_score += contribution
                score_breakdown['StochRSI_K'] = f"{contribution:.4f} (Val:{k:.2f}, W:{stochrsi_weight})"

            # MACD Score
            macd_weight = Decimal(str(weights.get("macd_score", 0.0)))
            if macd_weight != Decimal('0'):
                hist = latest_data.get('MACD_hist', Decimal('NaN'))
                # line = latest_data.get('MACD_line', Decimal('NaN'))
                # signal_line = latest_data.get('MACD_signal', Decimal('NaN'))
                contribution = Decimal('0.0')
                if not hist.is_nan():
                    if hist > 0: contribution = macd_weight  # Histogram positive -> bullish momentum
                    elif hist < 0: contribution = -macd_weight  # Histogram negative -> bearish momentum
                total_score += contribution
                score_breakdown['MACD_Hist'] = f"{contribution:.4f} (Val:{hist:.4f}, W:{macd_weight})"

            # Add scoring logic for other implemented indicators here...

            # --- Final Signal ---
            final_signal = self._determine_final_signal(total_score, weights)
            self.logger.debug(f"Score Breakdown ({weight_set_name}): {score_breakdown}")
            self.logger.info(f"Final Score ({weight_set_name}): {total_score:.4f}, Signal: {final_signal}")

            return final_signal, float(total_score), score_breakdown

        except Exception as e:
            self.logger.error(f"Error during signal generation: {e}", exc_info=True)
            return "NEUTRAL", 0.0, {}

# --- Position and Order Management ---


def get_market_info(exchange: ccxt.Exchange, symbol: str, logger: logging.Logger) -> dict[str, Any]:
    """Fetches and caches market information including precision and limits."""
    logger.debug(f"Getting market info for {symbol}")
    # Simple cache example (could use a more robust cache like cachetools)
    if not hasattr(get_market_info, "cache"):
        get_market_info.cache = {}  # Initialize cache if it doesn't exist

    if symbol in get_market_info.cache:
        logger.debug(f"Returning cached market info for {symbol}")
        return get_market_info.cache[symbol]

    try:
        market = exchange.market(symbol)
        # Ensure nested dictionaries exist before accessing
        precision = market.get('precision', {})
        limits = market.get('limits', {})
        amount_limits = limits.get('amount', {})
        cost_limits = limits.get('cost', {})
        price_limits = limits.get('price', {})

        info = {
            'id': market.get('id'),
            'symbol': market.get('symbol'),
            'type': market.get('type'),  # spot, swap, future
            'linear': market.get('linear'),  # True for linear contracts
            'inverse': market.get('inverse'),  # True for inverse contracts
            'settle': market.get('settle'),  # e.g., 'USDT'
            'precision': {
                'amount': int(-math.log10(precision['amount'])) if precision.get('amount') else 8,  # Use decimals for precision
                'price': int(-math.log10(precision['price'])) if precision.get('price') else 8,
            },
            'limits': {
                'amount': {
                    'min': Decimal(str(amount_limits['min'])) if amount_limits.get('min') else Decimal('0'),
                    'max': Decimal(str(amount_limits['max'])) if amount_limits.get('max') else Decimal('Infinity'),
                },
                'cost': {
                    'min': Decimal(str(cost_limits['min'])) if cost_limits.get('min') else Decimal('0'),
                    'max': Decimal(str(cost_limits['max'])) if cost_limits.get('max') else Decimal('Infinity'),
                },
                 'price': {
                    'min': Decimal(str(price_limits['min'])) if price_limits.get('min') else Decimal('0'),
                    'max': Decimal(str(price_limits['max'])) if price_limits.get('max') else Decimal('Infinity'),
                }
            },
            'contract_size': Decimal(str(market.get('contractSize', '1'))),  # Default to 1 if not specified
            'active': market.get('active', True),
            'tick_size': Decimal(str(precision['price'])) if precision.get('price') else Decimal('0.00000001')  # Smallest price increment
        }
        get_market_info.cache[symbol] = info
        logger.info(f"Market info fetched for {symbol}: Precision(Amt:{info['precision']['amount']}, Px:{info['precision']['price']}), Limits(AmtMin:{info['limits']['amount']['min']}, CostMin:{info['limits']['cost']['min']})")
        return info
    except Exception as e:
        logger.error(f"Could not fetch market info for {symbol}: {e}", exc_info=True)
        # Return a default structure on error to prevent crashes downstream
        return {
            'id': None, 'symbol': symbol, 'type': None, 'linear': None, 'inverse': None, 'settle': None,
            'precision': {'amount': 8, 'price': 8},
            'limits': {
                'amount': {'min': Decimal('0'), 'max': Decimal('Infinity')},
                'cost': {'min': Decimal('0'), 'max': Decimal('Infinity')},
                 'price': {'min': Decimal('0'), 'max': Decimal('Infinity')}
            },
            'contract_size': Decimal('1'), 'active': False, 'tick_size': Decimal('0.00000001')
        }


def fetch_balance(exchange: ccxt.Exchange, quote_currency: str, logger: logging.Logger) -> Decimal:
    """Fetches the free balance for the quote currency."""
    logger.debug(f"Fetching balance for {quote_currency}")
    try:
        balance_data = safe_api_call(exchange.fetch_balance, logger=logger)
        if balance_data and 'free' in balance_data and quote_currency in balance_data['total']:  # Check total exists first
            free_balance = Decimal(str(balance_data['free'].get(quote_currency, '0')))
            total_balance = Decimal(str(balance_data['total'].get(quote_currency, '0')))
            logger.info(f"Balance fetched: Total={total_balance:.2f} {quote_currency}, Free={free_balance:.2f} {quote_currency}")
            return free_balance
        else:
            logger.error(f"Could not fetch free balance for {quote_currency}. Response: {balance_data}")
            return Decimal('0')
    except Exception as e:
        logger.error(f"Error fetching balance: {e}", exc_info=True)
        return Decimal('0')


def quantize_value(value: Decimal, precision: int, rounding_mode=ROUND_DOWN) -> Decimal:
    """Quantizes a Decimal value to a specified number of decimal places."""
    if value.is_nan() or value.is_infinite():
         return value  # Don't quantize NaN or Inf
    # Use string formatting for precise quantization control
    quantizer = Decimal('1e-' + str(precision))
    return value.quantize(quantizer, rounding=rounding_mode)


def calculate_position_size(balance: Decimal, risk_per_trade: float, entry_price: Decimal, sl_price: Decimal,
                            market_info: dict, config: dict, logger: logging.Logger) -> Decimal | None:
    """Calculates position size based on risk, SL distance, and market constraints."""
    logger.debug("Calculating position size...")
    risk_config = config.get("risk", {})
    if not all([balance > 0, 0 < risk_per_trade < 1, entry_price > 0, sl_price > 0]):
        logger.error("Invalid input for position size calculation (balance, risk, prices must be positive).")
        return None
    if entry_price == sl_price:
        logger.error("Entry price cannot be equal to Stop Loss price.")
        return None

    risk_amount = balance * Decimal(str(risk_per_trade))
    sl_distance_per_unit = abs(entry_price - sl_price)
    contract_size = market_info.get('contract_size', Decimal('1'))

    if sl_distance_per_unit == Decimal('0'):
        logger.error("Stop loss distance is zero.")
        return None

    # Calculate raw size based on risk
    # Risk Amount = Size * SL Distance per Unit * Contract Size
    raw_size = risk_amount / (sl_distance_per_unit * contract_size)
    logger.debug(f"Risk Amount: {risk_amount:.4f} {QUOTE_CURRENCY}, SL Distance: {sl_distance_per_unit}, Contract Size: {contract_size}, Raw Size: {raw_size:.8f}")

    # Apply market precision
    amount_precision = market_info.get('precision', {}).get('amount', 8)
    quantized_size = quantize_value(raw_size, amount_precision, ROUND_DOWN)  # Round down to avoid exceeding risk

    # Check against amount limits
    min_amount = market_info.get('limits', {}).get('amount', {}).get('min', Decimal('0'))
    max_amount = market_info.get('limits', {}).get('amount', {}).get('max', Decimal('Infinity'))

    if quantized_size < min_amount:
        logger.warning(f"Calculated size {quantized_size:.{amount_precision}f} is below minimum order size {min_amount}. Cannot place trade.")
        return None
    if quantized_size > max_amount:
        logger.warning(f"Calculated size {quantized_size:.{amount_precision}f} exceeds maximum order size {max_amount}. Capping size.")
        quantized_size = quantize_value(max_amount, amount_precision, ROUND_DOWN)
        # Recalculate required cost after capping
        if quantized_size < min_amount:  # Check again after capping if max was below min
             logger.warning(f"Maximum size {max_amount} is still below minimum {min_amount}. Cannot place trade.")
             return None

    # Check against cost limits (Approximate check, actual cost depends on order type/slippage)
    estimated_cost = quantized_size * entry_price * contract_size
    min_cost = market_info.get('limits', {}).get('cost', {}).get('min', Decimal('0'))
    max_cost = market_info.get('limits', {}).get('cost', {}).get('max', Decimal('Infinity'))

    if min_cost is not None and estimated_cost < min_cost and min_cost > 0:  # Don't block if min_cost is 0
        logger.warning(f"Estimated cost {estimated_cost:.2f} is below minimum order cost {min_cost}. Cannot place trade.")
        return None
    if max_cost is not None and estimated_cost > max_cost:
        logger.warning(f"Estimated cost {estimated_cost:.2f} exceeds maximum order cost {max_cost}. Need to reduce size.")
        # Attempt to resize based on max cost - this might make it too small
        allowed_size_by_cost = max_cost / (entry_price * contract_size)
        quantized_size = quantize_value(allowed_size_by_cost, amount_precision, ROUND_DOWN)
        logger.warning(f"Size reduced to {quantized_size:.{amount_precision}f} based on max cost limit.")
        # Re-check against min amount limit after cost adjustment
        if quantized_size < min_amount:
             logger.warning(f"Size reduced by cost limit ({quantized_size}) is now below minimum amount {min_amount}. Cannot place trade.")
             return None

    # Check available balance (Simple check - more sophisticated margin check needed)
    # This check is basic; actual margin requirements depend on leverage and exchange rules.
    required_margin_buffer = Decimal(str(risk_config.get("required_margin_buffer", 1.05)))  # e.g. 5% buffer
    leverage = Decimal(str(config.get('trading', {}).get('max_leverage', 1)))
    required_margin = (quantized_size * entry_price * contract_size / leverage) * required_margin_buffer

    if required_margin > balance:
        logger.warning(f"Insufficient free balance ({balance:.2f}) for required margin ({required_margin:.2f}) with buffer. Cannot place trade.")
        # Potentially try to reduce size based on available balance? More complex.
        return None

    logger.info(f"Calculated Position Size: {quantized_size:.{amount_precision}f}")
    return quantized_size


def get_open_position(exchange: ccxt.Exchange, symbol: str, config: dict, logger: logging.Logger) -> dict | None:
    """Fetches the current open position for the symbol."""
    logger.debug(f"Fetching open position for {symbol}")
    try:
        params = {'category': 'linear'} if exchange.id == 'bybit' else {}  # Specify category for Bybit V5
        positions = safe_api_call(exchange.fetch_positions, [symbol], params=params, logger=logger)

        if positions:
            # fetch_positions returns a list, even if fetching for one symbol
            position = positions[0]
            # Check if position exists and has non-zero size
            # Position structure varies slightly between exchanges
            # Bybit V5 structure example: position['info']['size'] != '0' and position['info']['side'] != 'None'
            size_str = position.get('info', {}).get('size', '0')
            side_str = position.get('info', {}).get('side', 'None')
            entry_price_str = position.get('entryPrice') or position.get('info', {}).get('avgPrice')

            if size_str != '0' and side_str != 'None' and entry_price_str:
                pos_data = {
                    'symbol': position.get('symbol'),
                    'side': side_str.lower(),  # 'buy' (long) or 'sell' (short)
                    'size': Decimal(size_str),
                    'entry_price': Decimal(entry_price_str),
                    'liq_price': Decimal(str(position.get('liquidationPrice') or '0')),
                    'margin': Decimal(str(position.get('initialMargin') or '0')),  # Might be initial or maintenance
                    'pnl': Decimal(str(position.get('unrealizedPnl') or '0')),
                    'leverage': Decimal(str(position.get('leverage') or position.get('info', {}).get('leverage') or '1')),
                    'info': position.get('info', {})  # Keep original info dict
                }
                logger.info(f"Open position found: Side={pos_data['side']}, Size={pos_data['size']}, Entry={pos_data['entry_price']}")
                return pos_data
            else:
                logger.info(f"No active open position found for {symbol}.")
                return None
        else:
            logger.info(f"No position data returned for {symbol}.")
            return None
    except Exception as e:
        logger.error(f"Error fetching position for {symbol}: {e}", exc_info=True)
        return None


def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, config: dict, logger: logging.Logger) -> bool:
    """Sets leverage for the given symbol."""
    logger.info(f"Setting leverage for {symbol} to {leverage}x")
    try:
        response = safe_api_call(exchange.set_leverage, leverage, symbol, logger=logger)
        logger.debug(f"Set leverage response: {response}")
        logger.info(f"Leverage set successfully to {leverage}x for {symbol}.")
        return True
    except ccxt.DDoSProtection as e:
        logger.error(f"DDoS Protection triggered while setting leverage: {e}")
        return False
    except ccxt.RateLimitExceeded as e:
        logger.error(f"Rate Limit Exceeded while setting leverage: {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange Error setting leverage for {symbol} to {leverage}: {e}", exc_info=True)
        # Check if error indicates leverage is already set? Some exchanges might error.
        if "leverage not modify" in str(e).lower():
             logger.warning(f"Leverage for {symbol} already set to {leverage}x or cannot be modified.")
             return True  # Treat as success if already set
        return False
    except Exception as e:
        logger.error(f"Unexpected Error setting leverage for {symbol} to {leverage}: {e}", exc_info=True)
        return False


def place_trade(exchange: ccxt.Exchange, symbol: str, side: str, amount: Decimal, order_type: str,
                market_info: dict, config: dict, logger: logging.Logger, price: Decimal | None = None,
                params: dict | None = None) -> dict | None:
    """Places a trade order."""
    if params is None:
        params = {}
    logger.info(f"Placing {side.upper()} {order_type.upper()} order: {amount} {market_info.get('symbol')} at price {price if price else 'Market'}")
    amount_precision = market_info.get('precision', {}).get('amount', 8)
    price_precision = market_info.get('precision', {}).get('price', 8)

    # Format amount and price according to market precision
    formatted_amount = float(quantize_value(amount, amount_precision))  # CCXT often prefers float for amount
    formatted_price = float(quantize_value(price, price_precision)) if price else None

    # Ensure params is a dictionary
    order_params = params.copy() if params else {}

    # Add specific category for Bybit V5 if not already present
    if exchange.id == 'bybit' and 'category' not in order_params:
         order_params['category'] = 'linear'  # Assume linear, adjust if needed

    # Add reduceOnly parameter if closing an existing position
    # This logic might need refinement based on how exits are triggered
    if params.get('reduceOnly'):
        logger.info("Setting reduceOnly=True for closing order.")

    try:
        order = safe_api_call(
            exchange.create_order,
            symbol,
            order_type,
            side,
            formatted_amount,
            formatted_price,
            params=order_params,
            logger=logger
        )

        if order:
            logger.info(f"{side.upper()} {order_type.upper()} order placed successfully. Order ID: {order.get('id')}")
            logger.debug(f"Order details: {order}")
            return order
        else:
            logger.error(f"Order placement failed for {symbol}. safe_api_call returned None.")
            return None

    except ccxt.InsufficientFunds as e:
        logger.error(f"{NEON_RED}Insufficient funds to place {side} order for {amount} {symbol}: {e}{RESET}")
        return None
    except ccxt.InvalidOrder as e:
        logger.error(f"{NEON_RED}Invalid order parameters for {symbol}: {e}{RESET}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"{NEON_RED}Exchange error placing order for {symbol}: {e}{RESET}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"{NEON_RED}Unexpected error placing order for {symbol}: {e}{RESET}", exc_info=True)
        return None


def _set_position_protection(exchange: ccxt.Exchange, symbol: str, side: str,
                             tp_price: Decimal | None, sl_price: Decimal | None,
                             market_info: dict, config: dict, logger: logging.Logger) -> bool:
    """Sets Take Profit and Stop Loss for the position using exchange-native features (Bybit V5 example)."""
    protection_config = config.get("protection", {})
    if not protection_config.get("use_position_protection", True):
        logger.info("Position protection (TP/SL) is disabled in config.")
        return True  # Return True as no action is needed

    if not tp_price and not sl_price:
         logger.warning("Both TP and SL prices are None. Cannot set protection.")
         return False

    logger.info(f"Setting position protection for {symbol}: TP={tp_price}, SL={sl_price}")
    price_precision = market_info.get('precision', {}).get('price', 8)

    params = {
        'category': 'linear',
        'symbol': market_info.get('id'),  # Use exchange-specific ID
        'tpslMode': protection_config.get("tpsl_mode", "Full"),  # 'Full' or 'Partial'
        # Note: For Partial mode, tpSize/slSize would be needed
    }

    if tp_price:
        params['takeProfit'] = str(quantize_value(tp_price, price_precision, ROUND_UP if side == 'long' else ROUND_DOWN))  # Round TP away from entry
        params['tpTriggerBy'] = protection_config.get("tp_trigger_by", "LastPrice")
        params['tpOrderType'] = protection_config.get("tp_order_type", "Market")  # Only Market for Full mode

    if sl_price:
        params['stopLoss'] = str(quantize_value(sl_price, price_precision, ROUND_DOWN if side == 'long' else ROUND_UP))  # Round SL towards entry
        params['slTriggerBy'] = protection_config.get("sl_trigger_by", "LastPrice")
        params['slOrderType'] = protection_config.get("sl_order_type", "Market")

    logger.debug(f"Calling set_trading_stop with params: {params}")

    try:
        # Bybit V5 uses POST /v5/position/set-trading-stop
        # We need to call this via implicit methods or private API calls if not exposed directly in ccxt
        # Check if ccxt exposes this directly (e.g., exchange.set_trading_stop or similar)
        # As of some versions, direct support might be limited, requiring private API calls.
        # Example using private_post (adjust endpoint and method based on current ccxt/Bybit docs):
        if hasattr(exchange, 'private_post_v5_position_set_trading_stop'):
            response = safe_api_call(exchange.private_post_v5_position_set_trading_stop, params, logger=logger)
        elif hasattr(exchange, 'privatePost') or hasattr(exchange, 'v5_position_set_trading_stop'):  # Check older/alternative methods
             # This part is highly dependent on the CCXT version and exchange implementation details
             # You might need to find the correct implicit method name or structure for privatePost
             logger.warning("Direct V5 set_trading_stop method not found, attempting fallback (might fail)...")
             # Placeholder: replace with actual method if needed
             # response = safe_api_call(exchange.privatePost, 'v5/position/set-trading-stop', params, logger=logger)
             response = None  # Avoid executing potentially wrong call
             logger.error("Fallback for set_trading_stop not implemented. Cannot set TP/SL.")
             return False
        else:
             logger.error("Could not find method to set trading stop (TP/SL) for Bybit V5 in CCXT.")
             return False

        logger.debug(f"Set Trading Stop Response: {response}")
        # Check response for success indication (structure depends on Bybit API)
        if response and response.get('retCode') == 0:
            logger.info(f"Position protection (TP/SL) set successfully for {symbol}.")
            return True
        else:
            logger.error(f"Failed to set position protection for {symbol}. Response: {response}")
            return False

    except Exception as e:
        logger.error(f"Error setting position protection for {symbol}: {e}", exc_info=True)
        return False


def set_trailing_stop_loss(exchange: ccxt.Exchange, symbol: str, tsl_params: dict,
                            market_info: dict, config: dict, logger: logging.Logger) -> bool:
    """Sets or adjusts the Trailing Stop Loss for the position (Bybit V5 example)."""
    protection_config = config.get("protection", {})
    if not protection_config.get("enable_trailing_stop", False):
        logger.info("Trailing Stop Loss is disabled in config.")
        return True  # No action needed

    logger.info(f"Setting/Adjusting Trailing Stop Loss for {symbol} with params: {tsl_params}")

    # Combine base params with specific TSL params
    params = {
        'category': 'linear',
        'symbol': market_info.get('id'),  # Use exchange-specific ID
        'tpslMode': protection_config.get("tpsl_mode", "Full"),  # Assume TSL applies to full position for simplicity
        **tsl_params  # Merge in distance/activation price etc.
    }

    # Ensure numeric fields are strings if needed by API
    if 'trailingStop' in params: params['trailingStop'] = str(params['trailingStop'])
    if 'activePrice' in params: params['activePrice'] = str(params['activePrice'])
    # Set trigger price type if provided
    if 'triggerBy' in params: params['slTriggerBy'] = params.pop('triggerBy')  # Use slTriggerBy for TSL trigger

    logger.debug(f"Calling set_trading_stop for TSL with params: {params}")

    try:
        # Use the same mechanism as _set_position_protection
        if hasattr(exchange, 'private_post_v5_position_set_trading_stop'):
            response = safe_api_call(exchange.private_post_v5_position_set_trading_stop, params, logger=logger)
        # Add fallback logic as in _set_position_protection if needed
        else:
            logger.error("Could not find method to set trading stop (TSL) for Bybit V5 in CCXT.")
            return False

        logger.debug(f"Set Trailing Stop Response: {response}")
        if response and response.get('retCode') == 0:
            logger.info(f"Trailing Stop Loss set/adjusted successfully for {symbol}.")
            return True
        else:
            # Handle specific errors, e.g., TSL already active, invalid params
            ret_msg = response.get('retMsg', 'Unknown error')
            logger.error(f"Failed to set/adjust Trailing Stop Loss for {symbol}: {ret_msg} (Code: {response.get('retCode')}) Response: {response}")
            return False

    except Exception as e:
        logger.error(f"Error setting/adjusting Trailing Stop Loss for {symbol}: {e}", exc_info=True)
        return False

# --- Main Trading Logic ---


def analyze_and_trade_symbol(exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger) -> None:
    """Fetches data, analyzes, and executes trades for a single symbol."""
    try:
        logger.info(f"--- Starting Analysis Cycle for {symbol} ---")
        cycle_start_time = time.monotonic()

        # --- Config for this cycle ---
        trading_config = config.get("trading", {})
        protection_config = config.get("protection", {})
        risk_config = config.get("risk", {})
        interval = trading_config.get("interval", "1m")
        ohlcv_limit = trading_config.get("ohlcv_limit", 200)
        active_weight_set = trading_config.get("active_weight_set", "default")
        enable_trading = config.get("enable_trading", False)

        # --- Get Market Info (Cached) ---
        market_info = get_market_info(exchange, symbol, logger)
        if not market_info or not market_info.get('active'):
            logger.error(f"Market {symbol} not active or info not available. Skipping cycle.")
            return
        price_precision = market_info.get('precision', {}).get('price', 8)
        market_info.get('precision', {}).get('amount', 8)
        tick_size = market_info.get('tick_size', Decimal('0.00000001'))

        # --- Fetch Data ---
        logger.debug("Fetching OHLCV data...")
        ohlcv_data = safe_api_call(exchange.fetch_ohlcv, symbol, interval, limit=ohlcv_limit, logger=logger)
        if not ohlcv_data or len(ohlcv_data) < 50:  # Need sufficient data for indicators
            logger.warning(f"Insufficient OHLCV data fetched ({len(ohlcv_data) if ohlcv_data else 0}). Skipping analysis.")
            return

        df = create_dataframe(ohlcv_data)
        logger.debug(f"DataFrame created with {len(df)} rows. Last timestamp: {df.index[-1]}")

        # Fetch ticker for latest price (more reactive than close of last candle)
        ticker = safe_api_call(exchange.fetch_ticker, symbol, logger=logger)
        if not ticker or 'last' not in ticker:
            logger.warning("Could not fetch ticker for latest price. Using last close.")
            latest_price = df['close'].iloc[-1]
        else:
            latest_price = Decimal(str(ticker['last']))
        logger.debug(f"Latest Price: {latest_price:.{price_precision}f}")

        # --- Analyze Signal ---
        analyzer = TradingAnalyzer(df, config, market_info, logger)
        signal, score, score_breakdown = analyzer.generate_trading_signal(active_weight_set)
        calculated_atr = analyzer.atr  # Get ATR calculated during analysis

        # --- Get Position & Balance ---
        current_position = get_open_position(exchange, symbol, config, logger)
        free_balance = fetch_balance(exchange, QUOTE_CURRENCY, logger)

        # --- State Variables (for TSL/BE within this cycle) ---
        # In a real implementation, these might need to be loaded/saved externally
        tsl_active_for_position = False  # Track if TSL is currently active
        be_active_for_position = False  # Track if BreakEven is active

        # --- Decision Logic ---
        if not enable_trading:
            logger.info(f"Trading disabled. Signal: {signal}, Score: {score:.4f}. No actions taken.")

        # == ENTRY LOGIC ==
        elif signal != "NEUTRAL" and not current_position:
            target_side = "buy" if signal == "LONG" else "sell"
            logger.info(f"Potential ENTRY signal: {signal} (Score: {score:.4f})")

            if calculated_atr.is_nan() or calculated_atr <= 0:
                 logger.warning("ATR is invalid, cannot calculate SL/TP or size. Skipping entry.")
                 return

            # Calculate SL & TP Prices
            sl_atr_mult = Decimal(str(protection_config.get("sl_atr_multiplier", 1.5)))
            tp_atr_mult = Decimal(str(protection_config.get("tp_atr_multiplier", 2.0)))

            sl_distance = calculated_atr * sl_atr_mult
            tp_distance = calculated_atr * tp_atr_mult

            entry_price = latest_price  # Use latest price for calculation basis

            if signal == "LONG":
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:  # SHORT
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance

            # Ensure SL/TP respect min/max price if available
            # min_allowable_price = market_info.get('limits',{}).get('price',{}).get('min', Decimal('-Infinity'))
            # max_allowable_price = market_info.get('limits',{}).get('price',{}).get('max', Decimal('Infinity'))
            # sl_price = max(sl_price, min_allowable_price) if signal == "LONG" else min(sl_price, max_allowable_price)
            # tp_price = min(tp_price, max_allowable_price) if signal == "LONG" else max(tp_price, min_allowable_price)

            # Ensure TP is further than SL from entry
            if (signal == "LONG" and tp_price <= sl_price) or \
               (signal == "SHORT" and tp_price >= sl_price):
                logger.warning(f"Calculated TP price ({tp_price}) is not beyond SL price ({sl_price}). Adjusting TP.")
                # Simple adjustment: place TP further out based on SL distance, or skip TP
                if signal == "LONG": tp_price = sl_price + sl_distance  # Example: Mirror distance
                else: tp_price = sl_price - sl_distance
                logger.warning(f"Adjusted TP price to: {tp_price}")

            logger.info(f"Calculated Entry Parameters: Entry={entry_price:.{price_precision}f}, SL={sl_price:.{price_precision}f}, TP={tp_price:.{price_precision}f} (ATR={calculated_atr:.{price_precision}f})")

            # Calculate Position Size
            risk_per_trade = risk_config.get("risk_per_trade", 0.005)
            position_size = calculate_position_size(free_balance, risk_per_trade, entry_price, sl_price, market_info, config, logger)

            if position_size and position_size > 0:
                # Set Leverage
                leverage = int(config.get('trading', {}).get('max_leverage', 10))
                leverage_set = set_leverage_ccxt(exchange, symbol, leverage, config, logger)
                if not leverage_set:
                     logger.warning("Failed to set leverage. Proceeding with entry, but leverage might be incorrect.")
                     # Decide whether to proceed or stop based on risk tolerance

                # Place Entry Order (Market order assumed for scalping)
                order_params = {}  # Add any specific params if needed
                entry_order = place_trade(exchange, symbol, target_side, position_size, "market", market_info, config, logger, params=order_params)

                if entry_order and entry_order.get('id'):
                    # Wait briefly for order to likely fill and position to update
                    initial_delay = trading_config.get("initial_order_delay", 0.5)
                    time.sleep(initial_delay)

                    # Set Protection (TP/SL) for the newly opened position
                    # Re-fetch position to confirm it opened (optional but safer)
                    # confirmed_position = get_open_position(exchange, symbol, config, logger)
                    # if confirmed_position and confirmed_position['side'] == target_side:
                    protection_set = _set_position_protection(exchange, symbol, target_side, tp_price, sl_price, market_info, config, logger)
                    if not protection_set:
                         logger.error("!!! CRITICAL: Failed to set TP/SL protection after entry! Manual intervention may be required. !!!")
                         # Consider attempting to close the position immediately if protection fails? Risky.
                    # else:
                    #    logger.warning("Position not confirmed after entry order. Cannot set protection.")
                else:
                    logger.error("Entry order placement failed or did not return an ID.")
            else:
                logger.info("Position size calculation resulted in zero or invalid size. No entry order placed.")

        # == EXIT LOGIC ==
        elif current_position:
            position_side = current_position['side']  # 'buy' or 'sell'
            exit_reason = None

            # Check for signal reversal
            if (signal == "SHORT" and position_side == "buy") or \
               (signal == "LONG" and position_side == "sell") or \
               (signal == "NEUTRAL"):  # Exit on neutral signal if configured?
                exit_reason = f"Signal changed to {signal}"

            # Add checks for external TP/SL hits if protection placement fails or isn't used
            # Note: This relies on latest_price and is less reliable than exchange-native orders
            # if protection_config.get("use_position_protection", True) is False:
            #     # Manual checks (less reliable) - Example only
            #     entry_price_pos = current_position['entry_price']
            #     sl_price_pos = # Need to fetch or store SL associated with position
            #     tp_price_pos = # Need to fetch or store TP associated with position
            #     if position_side == 'buy' and (latest_price <= sl_price_pos or latest_price >= tp_price_pos):
            #         exit_reason = "External TP/SL Check"
            #     elif position_side == 'sell' and (latest_price >= sl_price_pos or latest_price <= tp_price_pos):
            #         exit_reason = "External TP/SL Check"

            if exit_reason:
                logger.info(f"Exit condition met for {position_side.upper()} position: {exit_reason}")
                if enable_trading:
                    close_side = "sell" if position_side == "buy" else "buy"
                    position_size = current_position['size']
                    # Place closing order (Market order, reduceOnly)
                    order_params = {'reduceOnly': True}
                    exit_order = place_trade(exchange, symbol, close_side, position_size, "market", market_info, config, logger, params=order_params)
                    if not exit_order:
                        logger.error("!!! Failed to place closing order! Manual intervention may be required. !!!")
                else:
                    logger.info("Trading disabled. Close order not placed.")
            else:
                 logger.info(f"Holding {position_side.upper()} position. Signal: {signal}")
                 # --- Trailing Stop & Break Even Logic (Only if holding position) ---
                 if enable_trading and protection_config.get("use_position_protection", True):
                      entry_price_pos = current_position['entry_price']
                      (latest_price - entry_price_pos) / entry_price_pos if position_side == 'buy' else (entry_price_pos - latest_price) / entry_price_pos

                      # ** Break Even Logic **
                      if protection_config.get("enable_break_even", False) and not be_active_for_position:
                           be_activation_atr_mult = Decimal(str(protection_config.get("be_activation_atr_multiplier", 1.0)))
                           be_target_profit_price = entry_price_pos + (calculated_atr * be_activation_atr_mult) if position_side == 'buy' else entry_price_pos - (calculated_atr * be_activation_atr_mult)

                           if (position_side == 'buy' and latest_price >= be_target_profit_price) or \
                              (position_side == 'sell' and latest_price <= be_target_profit_price):
                                logger.info(f"Break-even condition met (Price: {latest_price}, Target: {be_target_profit_price}). Adjusting SL.")
                                be_offset_pips = Decimal(str(protection_config.get("be_offset_pips", 2)))
                                be_sl_price = entry_price_pos + (tick_size * be_offset_pips) if position_side == 'buy' else entry_price_pos - (tick_size * be_offset_pips)
                                # Adjust existing SL using set_trading_stop (only modify SL)
                                {'stopLoss': str(quantize_value(be_sl_price, price_precision))}
                                be_success = _set_position_protection(exchange, symbol, position_side, None, be_sl_price, market_info, config, logger)  # Pass None for TP
                                if be_success:
                                     be_active_for_position = True  # Mark BE as active for this cycle
                                     logger.info(f"Break-even Stop Loss set to {be_sl_price:.{price_precision}f}")
                                else:
                                     logger.error("Failed to set break-even stop loss.")

                      # ** Trailing Stop Logic **
                      if protection_config.get("enable_trailing_stop", False) and not be_active_for_position:  # Don't trail if BE is active? Or allow? Configurable.
                          # Check if TSL needs activation
                          if not tsl_active_for_position:
                               tsl_activation_atr_mult = Decimal(str(protection_config.get("tsl_activation_atr_multiplier", 0.5)))
                               tsl_activation_target_price = entry_price_pos + (calculated_atr * tsl_activation_atr_mult) if position_side == 'buy' else entry_price_pos - (calculated_atr * tsl_activation_atr_mult)

                               if (position_side == 'buy' and latest_price >= tsl_activation_target_price) or \
                                  (position_side == 'sell' and latest_price <= tsl_activation_target_price):
                                    logger.info(f"Trailing Stop Loss activation condition met (Price: {latest_price}, Target: {tsl_activation_target_price}).")
                                    tsl_distance_atr_mult = Decimal(str(protection_config.get("tsl_distance_atr_multiplier", 1.0)))
                                    tsl_distance = calculated_atr * tsl_distance_atr_mult
                                    tsl_distance = quantize_value(tsl_distance, price_precision)  # Quantize distance

                                    # Set initial TSL using set_trading_stop with 'trailingStop' parameter
                                    tsl_params_set = {
                                         'trailingStop': str(tsl_distance),  # Distance value
                                         'activePrice': str(quantize_value(latest_price, price_precision)),  # Activate based on current price
                                         'slTriggerBy': protection_config.get("tsl_trigger_by", "LastPrice")  # Reuse SL trigger
                                         # Ensure any existing fixed SL/TP are removed or compatible
                                         # Setting TSL might implicitly cancel existing SL/TP on some exchanges/modes
                                         # Or explicitly set tpPrice=0, slPrice=0 if required by API when setting TSL
                                         # 'takeProfit': '0',
                                         # 'stopLoss': '0',
                                    }
                                    tsl_set_success = set_trailing_stop_loss(exchange, symbol, tsl_params_set, market_info, config, logger)
                                    if tsl_set_success:
                                         tsl_active_for_position = True
                                         # Need to store/track the TSL state if bot restarts
                                    else:
                                         logger.error("Failed to activate Trailing Stop Loss.")
                          # Note: If TSL is active via exchange-native, no further client-side adjustment is needed.
                          # The exchange handles trailing based on the set distance.

        # == NO POSITION STATE ==
        elif signal == "NEUTRAL":
            logger.info("Signal is NEUTRAL. No position open. Holding.")
        else:  # LONG or SHORT signal, but position already exists (should not happen if entry logic is correct)
             logger.warning(f"Signal is {signal}, but position already exists. Holding.")

    except ccxt.AuthenticationError as e:
         logger.critical(f"CRITICAL: Authentication Error in trading loop for {symbol}: {e}. Stopping bot.", exc_info=True)
         raise  # Re-raise critical error to stop main loop
    except Exception as e:
        logger.error(f"{NEON_RED}Unhandled error in trading loop for {symbol}: {e}{RESET}", exc_info=True)
        # Continue loop after error? Or break? Depends on severity.
    finally:
        cycle_end_time = time.monotonic()
        logger.info(f"--- Finished Analysis Cycle for {symbol} ({cycle_end_time - cycle_start_time:.2f}s) ---")


# --- Main Execution ---
def main() -> None:
    """Main function to initialize the bot and run the analysis loop."""
    global CONFIG, QUOTE_CURRENCY  # Allow modification of globals (consider refactoring to avoid)

    # Use a general logger for initial setup
    # Logger setup moved here to ensure it runs after potential config load errors
    setup_logger("init", level=logging.INFO)  # Use INFO level for init
    init_logger = logging.getLogger("init")

    start_time_str = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
    init_logger.info(f"--- Starting Bot ({start_time_str}) ---")

    try:
        # Load/Update config at start
        temp_config = load_config(CONFIG_FILE)
        # Update global CONFIG - avoid if possible by passing config explicitly
        CONFIG = temp_config
        QUOTE_CURRENCY = CONFIG.get("quote_currency", "USDT")

        # Set root logger level based on config *after* loading config
        log_level_str = CONFIG.get("logging", {}).get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)  # Set root level
        init_logger.setLevel(log_level)  # Adjust init logger level too
        init_logger.info(f"Logging level set to: {log_level_str}")

        init_logger.info(f"Config loaded from {CONFIG_FILE}. Quote Currency: {QUOTE_CURRENCY}")
        init_logger.info(f"Versions: CCXT={ccxt.__version__}, Pandas={pd.__version__}, PandasTA={ta.version if hasattr(ta, 'version') else 'N/A'}")

        # --- Trading Enabled Warning ---
        if CONFIG.get("enable_trading"):
            init_logger.warning(f"{NEON_YELLOW}!!! LIVE TRADING IS ENABLED !!!{RESET}")
            if CONFIG.get("use_sandbox"):
                init_logger.warning(f"{NEON_YELLOW}Using SANDBOX (Testnet) Environment.{RESET}")
            else:
                init_logger.warning(f"{NEON_RED}!!! CAUTION: USING REAL MONEY ENVIRONMENT !!!{RESET}")
        else:
             init_logger.info("Live trading is DISABLED.")

        # --- Initialize Exchange ---
        exchange = initialize_exchange(CONFIG)
        if not exchange:
            init_logger.critical("Exiting due to exchange initialization failure.")
            return

        # --- Main Loop ---
        symbols_to_trade = CONFIG.get("symbols", [])
        if not symbols_to_trade:
            init_logger.critical("No symbols configured to trade in 'config.json'. Exiting.")
            if hasattr(exchange, 'close'): await exchange.close()  # Close if async
            return

        init_logger.info(f"Starting trading loop for symbols: {', '.join(symbols_to_trade)}")
        loop_interval = CONFIG.get("trading", {}).get("loop_interval_seconds", 10)

        # Setup loggers for each symbol
        for symbol in symbols_to_trade:
            setup_logger(symbol.replace("/", "_").replace(":", "_"), level=log_level)  # Create logger for each symbol

        while True:
            # In a single-threaded model, iterate through symbols
            # In an async model, tasks would run concurrently (see previous examples)
            for symbol in symbols_to_trade:
                symbol_logger = logging.getLogger(symbol.replace("/", "_").replace(":", "_"))
                try:
                    analyze_and_trade_symbol(exchange, symbol, CONFIG, symbol_logger)
                except ccxt.AuthenticationError:
                    # Raised by analyze_and_trade_symbol on critical auth error
                    init_logger.critical(f"Authentication Error for {symbol}. Stopping bot.")
                    raise  # Re-raise to break the outer loop
                except Exception as symbol_err:
                     # Catch errors from the symbol analysis function itself if not caught internally
                     init_logger.error(f"{NEON_RED}Error processing symbol {symbol} in main loop: {symbol_err}{RESET}", exc_info=True)
                     # Decide if one symbol failing should stop the bot or just log and continue

            # Delay before next cycle through all symbols
            init_logger.debug(f"Completed cycle for all symbols. Waiting {loop_interval} seconds...")
            time.sleep(loop_interval)

    except KeyboardInterrupt:
        init_logger.info("KeyboardInterrupt received. Initiating graceful shutdown...")
        # Add enhanced shutdown logic here (cancel orders, close positions if configured)
        shutdown_config = CONFIG.get("shutdown", {})
        if 'exchange' in locals() and exchange:  # Check if exchange was initialized
             if shutdown_config.get("cancel_open_orders", True):
                 init_logger.info("Attempting to cancel open orders...")
                 # Add logic to fetch/cancel orders for all symbols
             if shutdown_config.get("close_open_positions", False):
                 init_logger.warning("Attempting to close open positions...")
                 # Add logic to fetch/close positions for all symbols

    except Exception as startup_err:
        # Catch errors during initial setup before the main loop
        # Use basic print if logger failed, otherwise use init_logger
        msg = f"CRITICAL error during startup/main loop: {startup_err}"
        try: init_logger.critical(msg, exc_info=True)
        except: traceback.print_exc()

    finally:
        # --- Final Shutdown Steps ---
        end_time_str = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')
        shutdown_msg = f"--- Bot Shutdown ({end_time_str}) ---"

        # Ensure exchange connection closed if initialized
        if 'exchange' in locals() and exchange and hasattr(exchange, 'close'):
            try:
                # Use basic print here as loggers might be shut down
                # If using async, this needs await exchange.close() in async context
                exchange.close()
            except Exception:
                 pass

        # Ensure logs are flushed
        logging.shutdown()


if __name__ == "__main__":
    # Removed the self-writing block.
    # Standard execution: python livexx_improved.py
    try:
         main()
    except Exception:
         # Catch any exception that might escape main() during critical failure
         # Optionally print traceback for debugging critical startup/shutdown errors
         import traceback
         traceback.print_exc()
