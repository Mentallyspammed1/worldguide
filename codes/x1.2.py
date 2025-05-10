import argparse  # For command-line arguments
import contextlib
import json
import logging
import math
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo  # Preferred over pytz for modern Python

# Third-party libraries - alphabetized
import ccxt
import pandas as pd
import pandas_ta as ta
from colorama import Fore, Style, init
from dotenv import load_dotenv

# Initialize colorama and set decimal precision
getcontext().prec = 28  # Set precision for Decimal calculations
init(autoreset=True)
load_dotenv()  # Load environment variables from .env file

# --- Neon Color Scheme ---
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# --- Environment Variable Loading and Validation ---
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    # Use print as logger might not be set up yet
    exit(1)  # Exit if keys are missing

# --- Configuration File and Constants ---
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Timezone for logging and display
try:
    TZ_NAME = os.getenv("BOT_TIMEZONE", "America/Chicago")
    TIMEZONE = ZoneInfo(TZ_NAME)
except Exception:
    TIMEZONE = ZoneInfo("UTC")

# --- API Interaction Constants ---
MAX_API_RETRIES = 3  # Max retries for recoverable API errors
RETRY_DELAY_SECONDS = 5  # Base delay between retries
# HTTP status codes considered retryable (Network/Server issues) - Not explicitly used here, ccxt handles some
# RETRYABLE_HTTP_STATUS = [429, 500, 502, 503, 504]
# Intervals supported by the bot's internal logic (ensure config matches)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
# Map bot intervals to ccxt's expected timeframe format
CCXT_INTERVAL_MAP = {
    "1": "1m",
    "3": "3m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "120": "2h",
    "240": "4h",
    "D": "1d",
    "W": "1w",
    "M": "1M",
}

# --- Default Indicator/Strategy Parameters (can be overridden by config.json) ---
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14  # Window for Stoch RSI calculation itself
DEFAULT_STOCH_WINDOW = 12  # Window for underlying RSI in StochRSI
DEFAULT_K_WINDOW = 3  # K period for StochRSI
DEFAULT_D_WINDOW = 3  # D period for StochRSI
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2.0  # Ensure float for calculations
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
DEFAULT_MOMENTUM_PERIOD = 7
DEFAULT_VOLUME_MA_PERIOD = 15
DEFAULT_FIB_WINDOW = 50
DEFAULT_PSAR_AF = 0.02
DEFAULT_PSAR_MAX_AF = 0.2
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]  # Standard Fibonacci levels

# --- Bot Timing and Delays ---
LOOP_DELAY_SECONDS = (
    15  # Time between the end of one cycle and the start of the next (configurable)
)
POSITION_CONFIRM_DELAY = (
    10  # Seconds to wait after placing order before checking position status
)

QUOTE_CURRENCY = "USDT"  # Default, will be updated from loaded config


# --- Logger Setup ---
class SensitiveFormatter(logging.Formatter):
    """Custom formatter to redact sensitive information (API keys) from logs."""

    def format(self, record: logging.LogRecord) -> str:
        original_message = super().format(record)
        redacted_message = original_message
        if API_KEY:
            redacted_message = redacted_message.replace(API_KEY, "***API_KEY***")
        if API_SECRET:
            redacted_message = redacted_message.replace(API_SECRET, "***API_SECRET***")
        return redacted_message


def setup_logger(symbol: str, console_level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger for the given symbol with rotating file and console handlers.

    Args:
        symbol: The trading symbol (used for naming log files).
        console_level: The logging level for the console output.

    Returns:
        The configured logger instance.
    """
    safe_symbol = symbol.replace("/", "_").replace(":", "-")
    logger_name = f"livebot_{safe_symbol}"
    log_filename = os.path.join(LOG_DIRECTORY, f"{logger_name}.log")
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        # Update existing console handler level if needed
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)
        logger.debug(
            f"Logger '{logger_name}' already configured. Console level set to {logging.getLevelName(console_level)}."
        )
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels

    # --- File Handler ---
    try:
        file_handler = RotatingFileHandler(
            log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_formatter = SensitiveFormatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # UTC timestamp in file
        )
        # Use UTC for file logs
        logging.Formatter.converter = time.gmtime
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # Log everything to the file
        logger.addHandler(file_handler)
    except Exception:
        pass

    # --- Stream Handler (Console) ---
    stream_handler = logging.StreamHandler()
    # Use local timezone for console timestamps
    stream_formatter = SensitiveFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",  # Timestamp format for console
    )
    # Apply the local timezone to the console formatter's asctime

    class LocalTimeFormatter(SensitiveFormatter):
        converter = lambda *args: datetime.now(TIMEZONE).timetuple()

    stream_formatter = LocalTimeFormatter(
        f"{NEON_BLUE}%(asctime)s{RESET} - {NEON_YELLOW}%(levelname)-8s{RESET} - {NEON_PURPLE}[%(name)s]{RESET} - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(console_level)  # Set level from argument
    logger.addHandler(stream_handler)

    logger.propagate = False

    logger.info(
        f"Logger '{logger_name}' initialized. File: '{log_filename}', Console Level: {logging.getLevelName(console_level)}"
    )
    return logger


# --- Configuration Management ---
def _ensure_config_keys(
    config: dict[str, Any], default_config: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Recursively ensures all keys from the default config are present in the loaded config."""
    updated_config = config.copy()
    keys_added = False
    for key, default_value in default_config.items():
        if key not in updated_config:
            updated_config[key] = default_value
            keys_added = True
        elif isinstance(default_value, dict) and isinstance(
            updated_config.get(key), dict
        ):
            nested_updated_config, nested_keys_added = _ensure_config_keys(
                updated_config[key], default_value
            )
            if nested_keys_added:
                updated_config[key] = nested_updated_config
                keys_added = True
        # Optional: Type mismatch check
        elif (
            type(default_value) != type(updated_config.get(key))
            and updated_config.get(key) is not None
        ):
            # Be careful with float vs int comparisons if precision matters
            # Allow int to be loaded if float is default, but warn
            if not (
                isinstance(default_value, float)
                and isinstance(updated_config.get(key), int)
            ):
                pass

    return updated_config, keys_added


def load_config(filepath: str) -> dict[str, Any]:
    """Load configuration from JSON file, create default if missing, ensure keys exist, save updates."""
    default_config = {
        "interval": "3",  # Default interval (ensure it's in VALID_INTERVALS)
        "retry_delay": RETRY_DELAY_SECONDS,
        "loop_delay": LOOP_DELAY_SECONDS,
        "quote_currency": "USDT",
        "enable_trading": False,  # SAFETY FIRST: Default to False
        "use_sandbox": True,  # SAFETY FIRST: Default to True (testnet)
        "risk_per_trade": 0.01,  # Risk 1% (as float)
        "leverage": 20,  # Desired leverage (check exchange limits!)
        "max_concurrent_positions": 1,  # Informational, not enforced by this script version
        # --- Indicator Periods & Settings ---
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
        "fibonacci_window": DEFAULT_FIB_WINDOW,
        # --- Signal Generation & Thresholds ---
        "orderbook_limit": 25,
        "signal_score_threshold": 1.5,  # Use float
        "stoch_rsi_oversold_threshold": 25.0,  # Use float
        "stoch_rsi_overbought_threshold": 75.0,  # Use float
        "volume_confirmation_multiplier": 1.5,  # Use float
        "scalping_signal_threshold": 2.5,  # Use float
        # --- Risk Management Multipliers (based on ATR) ---
        "stop_loss_multiple": 1.8,  # Use float
        "take_profit_multiple": 0.7,  # Use float
        # --- Exit Strategies ---
        "enable_ma_cross_exit": True,
        # --- Trailing Stop Loss Config (Exchange-based TSL) ---
        "enable_trailing_stop": True,
        "trailing_stop_callback_rate": 0.005,  # 0.5% trail distance (as float)
        "trailing_stop_activation_percentage": 0.003,  # 0.3% activation profit (as float)
        # --- Break-Even Stop Config ---
        "enable_break_even": True,
        "break_even_trigger_atr_multiple": 1.0,  # Use float
        "break_even_offset_ticks": 2,  # Use integer
        # --- Indicator Enable/Disable Flags ---
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True,
            "psar": True,
            "sma_10": True,
            "mfi": True,
            "orderbook": True,
        },
        # --- Indicator Weighting Sets ---
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "stoch_rsi": 0.6,
                "rsi": 0.2,
                "bollinger_bands": 0.3,
                "vwap": 0.4,
                "cci": 0.3,
                "wr": 0.3,
                "psar": 0.2,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.15,
            },
            "default": {
                "ema_alignment": 0.3,
                "momentum": 0.2,
                "volume_confirmation": 0.1,
                "stoch_rsi": 0.4,
                "rsi": 0.3,
                "bollinger_bands": 0.2,
                "vwap": 0.3,
                "cci": 0.2,
                "wr": 0.2,
                "psar": 0.3,
                "sma_10": 0.1,
                "mfi": 0.2,
                "orderbook": 0.1,
            },
        },
        "active_weight_set": "default",
    }

    # --- File Handling ---
    if not os.path.exists(filepath):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
            return default_config
        except OSError:
            return default_config

    # --- Load Existing Config and Merge Defaults ---
    try:
        with open(filepath, encoding="utf-8") as f:
            config_from_file = json.load(f)
        updated_config, keys_added = _ensure_config_keys(
            config_from_file, default_config
        )
        if keys_added:
            try:
                with open(filepath, "w", encoding="utf-8") as f_write:
                    json.dump(updated_config, f_write, indent=4, sort_keys=True)
            except OSError:
                pass
        return updated_config
    except (FileNotFoundError, json.JSONDecodeError):
        try:  # Attempt to recreate default if loading failed
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, sort_keys=True)
        except OSError:
            pass
        return default_config
    except Exception:
        return default_config


# Load configuration globally after defining the function
CONFIG = load_config(CONFIG_FILE)
QUOTE_CURRENCY = CONFIG.get(
    "quote_currency", "USDT"
)  # Get quote currency for global use


# --- CCXT Exchange Setup ---
def initialize_exchange(
    config: dict[str, Any], logger: logging.Logger
) -> ccxt.Exchange | None:
    """Initializes the CCXT Bybit exchange object with V5 settings, error handling, and tests.

    Args:
        config: The loaded configuration dictionary.
        logger: The logger instance.

    Returns:
        An initialized ccxt.Exchange object, or None if initialization fails.
    """
    lg = logger
    try:
        exchange_options = {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {
                "defaultType": "linear",  # Assume linear contracts
                "adjustForTimeDifference": True,
                # Timeouts (milliseconds)
                "fetchTickerTimeout": 10000,
                "fetchBalanceTimeout": 15000,
                "createOrderTimeout": 20000,
                "fetchOrderTimeout": 15000,
                "fetchPositionsTimeout": 15000,
                "cancelOrderTimeout": 15000,
                # Bybit V5 Specific Options
                "default_options": {
                    "adjustForTimeDifference": True,
                    "warnOnFetchOpenOrdersWithoutSymbol": False,
                    "recvWindow": 10000,
                    # Explicitly request V5 API for key endpoints
                    "fetchPositions": "v5",
                    "fetchBalance": "v5",
                    "createOrder": "v5",
                    "fetchOrder": "v5",
                    "cancelOrder": "v5",
                    "setLeverage": "v5",
                    "private_post_v5_position_trading_stop": "v5",  # For SL/TP/TSL
                    # Add others as needed: fetchOHLCV, fetchTicker, etc. might default ok
                },
                # Map ccxt market types to Bybit V5 account types for balance fetching etc.
                "accounts": {
                    "future": {
                        "linear": "CONTRACT",
                        "inverse": "CONTRACT",
                    },  # Futures use CONTRACT
                    "swap": {
                        "linear": "CONTRACT",
                        "inverse": "CONTRACT",
                    },  # Swaps use CONTRACT
                    "option": {"unified": "OPTION"},  # Options use OPTION
                    "spot": {
                        "unified": "UNIFIED",
                        "spot": "SPOT",
                    },  # Spot can use UNIFIED or SPOT
                    "margin": {"unified": "UNIFIED"},  # Margin likely UNIFIED
                    # Default or Unified account mapping
                    "unified": {"unified": "UNIFIED"},
                    "contract": {"contract": "CONTRACT"},  # Explicit contract type
                },
                # Optional: Add a broker ID for tracking
                "brokerId": "EnhancedWhale71",
            },
        }

        exchange_id = "bybit"
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(exchange_options)

        if config.get("use_sandbox", True):
            lg.warning(f"{NEON_YELLOW}USING SANDBOX MODE (Testnet){RESET}")
            exchange.set_sandbox_mode(True)
        else:
            lg.warning(f"{NEON_RED}USING LIVE TRADING MODE (Real Money){RESET}")

        # --- Test Connection & Load Markets ---
        lg.info(
            f"Connecting to {exchange.id} (Sandbox: {config.get('use_sandbox', True)})..."
        )
        lg.info(f"Loading markets for {exchange.id}...")
        try:
            exchange.load_markets()
            lg.info(f"Markets loaded successfully for {exchange.id}.")
        except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
            lg.error(
                f"{NEON_RED}Error loading markets: {e}. Check connection/API settings.{RESET}",
                exc_info=True,
            )
            return None

        lg.info(
            f"CCXT exchange initialized ({exchange.id}). CCXT Version: {ccxt.__version__}"
        )

        # --- Test API Credentials & Permissions (Fetch Balance) ---
        quote_currency = config.get("quote_currency", "USDT")
        lg.info(f"Attempting initial balance fetch for {quote_currency}...")
        try:
            balance_decimal = fetch_balance(
                exchange, quote_currency, lg
            )  # Use the enhanced fetch_balance

            if balance_decimal is not None:
                lg.info(
                    f"{NEON_GREEN}Successfully connected and fetched initial balance.{RESET} ({quote_currency} available: {balance_decimal:.4f})"
                )
            else:
                # fetch_balance logs specific errors, add a general warning here
                lg.warning(
                    f"{NEON_YELLOW}Initial balance fetch returned None or failed. Check logs. Ensure API keys have 'Read' permissions and correct account type (CONTRACT/UNIFIED/SPOT) is accessible for {quote_currency}.{RESET}"
                )
                if config.get("enable_trading"):
                    lg.error(
                        f"{NEON_RED}Cannot verify balance. Trading is enabled, aborting initialization for safety.{RESET}"
                    )
                    return None
                else:
                    lg.warning(
                        "Continuing in non-trading mode despite balance fetch issue."
                    )

        except ccxt.AuthenticationError as auth_err:
            lg.error(
                f"{NEON_RED}CCXT Authentication Error during initial balance fetch: {auth_err}{RESET}"
            )
            lg.error(
                f"{NEON_RED}>> Ensure API keys are correct, have necessary permissions (Read, Trade), match the account type (Real/Testnet), and IP whitelist is correctly set if enabled on Bybit.{RESET}"
            )
            return None
        except Exception as balance_err:
            lg.warning(
                f"{NEON_YELLOW}Warning during initial balance fetch: {balance_err}. Continuing, but check API permissions/account type if trading fails.{RESET}"
            )
            if config.get("enable_trading"):
                lg.error(
                    f"{NEON_RED}Aborting initialization due to balance fetch error in trading mode.{RESET}"
                )
                return None

        return exchange

    except (
        ccxt.AuthenticationError,
        ccxt.ExchangeError,
        ccxt.NetworkError,
        Exception,
    ) as e:
        lg.error(
            f"{NEON_RED}Failed to initialize CCXT exchange: {e}{RESET}", exc_info=True
        )
        return None


# --- CCXT Data Fetching with Retries ---
def _handle_rate_limit(error: ccxt.RateLimitExceeded, logger: logging.Logger) -> int:
    """Parses rate limit error message to find suggested wait time."""
    default_wait = RETRY_DELAY_SECONDS * 3  # Longer default for rate limits
    try:
        error_msg = str(error).lower()
        import re

        # Look for patterns like "try again in Xms" or "retry after Xs"
        match_ms = re.search(r"(?:try again in|retry after)\s*(\d+)\s*ms", error_msg)
        match_s = re.search(r"(?:try again in|retry after)\s*(\d+)\s*s", error_msg)
        wait_sec = default_wait

        if match_ms:
            wait_ms = int(match_ms.group(1))
            wait_sec = max(1, math.ceil(wait_ms / 1000) + 1)  # Add buffer, min 1s
            logger.warning(f"Rate limit suggests waiting {wait_sec}s (from ms).")
        elif match_s:
            wait_sec = max(1, int(match_s.group(1)) + 1)  # Add buffer, min 1s
            logger.warning(f"Rate limit suggests waiting {wait_sec}s (from s).")
        else:  # Generic message, look for digits
            match_num = re.search(r"(\d+)", error_msg)
            if match_num:  # Assume seconds if unit missing
                wait_sec = max(1, int(match_num.group(1)) + 1)
                logger.warning(f"Rate limit suggests waiting {wait_sec}s (inferred).")
            else:
                logger.warning(
                    f"Could not parse rate limit wait time from '{error}'. Using default {default_wait}s."
                )
                wait_sec = default_wait
        return wait_sec
    except Exception as parse_err:
        logger.warning(
            f"Error parsing rate limit wait time from '{error}': {parse_err}. Using default {default_wait}s."
        )
    return default_wait


def fetch_current_price_ccxt(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> Decimal | None:
    """Fetches the current price using CCXT ticker with retries and fallbacks (Decimal)."""
    lg = logger
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(
                f"Fetching ticker for {symbol} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})..."
            )
            # Add category param for Bybit V5 if needed
            params = {}
            if "bybit" in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    # Infer category based on type (spot/linear/inverse)
                    market_type = market.get(
                        "type", "linear"
                    )  # Default to linear if type missing
                    category = (
                        "spot"
                        if market_type == "spot"
                        else ("linear" if market.get("linear", True) else "inverse")
                    )
                except Exception:
                    category = (
                        "linear" if "USDT" in symbol else "inverse"
                    )  # Fallback guess
                params["category"] = category
                lg.debug(f"Using params for fetch_ticker: {params}")

            ticker = exchange.fetch_ticker(symbol, params=params)
            lg.debug(f"Raw ticker data for {symbol}: {ticker}")

            # Helper to safely convert to Decimal and check positivity
            def safe_decimal(value: Any) -> Decimal | None:
                if value is None:
                    return None
                try:
                    d_val = Decimal(str(value))
                    return d_val if d_val > 0 else None
                except (InvalidOperation, ValueError, TypeError):
                    return None

            # Prioritize: last > midpoint > ask > bid
            price = safe_decimal(ticker.get("last"))
            if price:
                lg.debug(f"Using 'last' price: {price}")
                return price

            bid = safe_decimal(ticker.get("bid"))
            ask = safe_decimal(ticker.get("ask"))
            if bid and ask and bid <= ask:
                price = (bid + ask) / Decimal("2")
                lg.debug(f"Using bid/ask midpoint: {price} (Bid: {bid}, Ask: {ask})")
                return price

            if ask:
                lg.warning(f"Using 'ask' price as fallback: {ask}")
                return ask
            if bid:
                lg.warning(f"Using 'bid' price as last resort: {bid}")
                return bid

            lg.error(
                f"{NEON_RED}Failed to extract a valid positive price from ticker for {symbol}.{RESET}"
            )
            return None  # No valid price found

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(
                f"Rate limit hit fetching ticker for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})"
            )
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(
                f"Network error fetching ticker for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(
                f"{NEON_RED}Exchange error fetching ticker for {symbol}: {e}. Not retrying.{RESET}"
            )
            return None
        except Exception as e:
            lg.error(
                f"{NEON_RED}Unexpected error fetching ticker for {symbol}: {e}. Not retrying.{RESET}",
                exc_info=True,
            )
            return None

    lg.error(
        f"{NEON_RED}Max retries reached fetching ticker for {symbol}. Last error: {last_exception}{RESET}"
    )
    return None


def fetch_klines_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int = 250,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetches OHLCV kline data using CCXT with retries and basic validation."""
    lg = logger or logging.getLogger(__name__)
    if not exchange.has["fetchOHLCV"]:
        lg.error(f"Exchange {exchange.id} does not support fetchOHLCV.")
        return pd.DataFrame()

    ohlcv = None
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(
                f"Fetching klines for {symbol}, {timeframe}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})"
            )
            params = {}
            if "bybit" in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    market_type = market.get("type", "linear")
                    category = (
                        "spot"
                        if market_type == "spot"
                        else ("linear" if market.get("linear", True) else "inverse")
                    )
                except Exception:
                    category = (
                        "linear" if "USDT" in symbol else "inverse"
                    )  # Fallback guess
                params["category"] = category
                lg.debug(f"Using params for fetch_ohlcv: {params}")

            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit, params=params
            )

            if ohlcv is not None and len(ohlcv) > 0:
                try:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    if df.empty:
                        lg.warning(f"Kline DataFrame empty for {symbol} {timeframe}.")
                        return pd.DataFrame()

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit="ms", errors="coerce"
                    )
                    df.dropna(subset=["timestamp"], inplace=True)
                    if df.empty:
                        lg.warning(
                            f"Kline DataFrame empty after timestamp conversion for {symbol}."
                        )
                        return pd.DataFrame()
                    df.set_index("timestamp", inplace=True)

                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    initial_len = len(df)
                    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                    df = df[df["close"] > 0]  # Require positive close price

                    rows_dropped = initial_len - len(df)
                    if rows_dropped > 0:
                        lg.debug(
                            f"Dropped {rows_dropped} rows with NaN/invalid data for {symbol}."
                        )
                    if df.empty:
                        lg.warning(f"Kline data empty after cleaning for {symbol}.")
                        return pd.DataFrame()

                    df.sort_index(inplace=True)
                    lg.info(
                        f"Successfully fetched and processed {len(df)} klines for {symbol} {timeframe}"
                    )
                    return df

                except Exception as proc_err:
                    lg.error(
                        f"{NEON_RED}Error processing kline data for {symbol}: {proc_err}. Not retrying.{RESET}",
                        exc_info=True,
                    )
                    return pd.DataFrame()
            else:
                lg.warning(
                    f"fetch_ohlcv returned no data for {symbol} (Attempt {attempt + 1}). Retrying..."
                )
                last_exception = ValueError("API returned no kline data")
                time.sleep(1)

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(
                f"Rate limit hit fetching klines for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})"
            )
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(
                f"Network error fetching klines for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(
                f"{NEON_RED}Exchange error fetching klines for {symbol}: {e}. Not retrying.{RESET}"
            )
            return pd.DataFrame()
        except Exception as e:
            lg.error(
                f"{NEON_RED}Unexpected error fetching klines for {symbol}: {e}. Not retrying.{RESET}",
                exc_info=True,
            )
            return pd.DataFrame()

    lg.error(
        f"{NEON_RED}Max retries reached fetching klines for {symbol}. Last error: {last_exception}{RESET}"
    )
    return pd.DataFrame()


def fetch_orderbook_ccxt(
    exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger
) -> dict | None:
    """Fetches orderbook data using ccxt with retries and basic validation."""
    lg = logger
    if not exchange.has["fetchOrderBook"]:
        lg.error(f"Exchange {exchange.id} does not support fetchOrderBook.")
        return None

    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            lg.debug(
                f"Fetching order book for {symbol}, limit={limit} (Attempt {attempt + 1}/{MAX_API_RETRIES + 1})"
            )
            params = {}
            if "bybit" in exchange.id.lower():
                try:
                    market = exchange.market(symbol)
                    market_type = market.get("type", "linear")
                    category = (
                        "spot"
                        if market_type == "spot"
                        else ("linear" if market.get("linear", True) else "inverse")
                    )
                except Exception:
                    category = (
                        "linear" if "USDT" in symbol else "inverse"
                    )  # Fallback guess
                params["category"] = category
                lg.debug(f"Using params for fetch_order_book: {params}")

            orderbook = exchange.fetch_order_book(symbol, limit=limit, params=params)

            if not orderbook:
                lg.warning(
                    f"fetch_order_book returned None/empty for {symbol} (Attempt {attempt + 1}). Retrying..."
                )
                last_exception = ValueError("API returned no orderbook data")
                time.sleep(1)
                continue
            elif not isinstance(orderbook.get("bids"), list) or not isinstance(
                orderbook.get("asks"), list
            ):
                lg.warning(
                    f"{NEON_YELLOW}Invalid orderbook structure for {symbol}. Attempt {attempt + 1}. Response: {orderbook}{RESET}"
                )
                last_exception = TypeError("Invalid orderbook structure")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            elif not orderbook["bids"] and not orderbook["asks"]:
                lg.warning(
                    f"{NEON_YELLOW}Orderbook received but bids/asks lists are empty for {symbol}.{RESET}"
                )
                return orderbook  # Return empty book
            else:
                lg.debug(
                    f"Successfully fetched orderbook for {symbol} with {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks."
                )
                return orderbook

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(
                f"Rate limit hit fetching orderbook for {symbol}. Retrying in {wait_time}s... (Attempt {attempt + 1})"
            )
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(
                f"Network error fetching orderbook for {symbol}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.ExchangeError as e:
            lg.error(
                f"{NEON_RED}Exchange error fetching orderbook for {symbol}: {e}. Not retrying.{RESET}"
            )
            return None
        except Exception as e:
            lg.error(
                f"{NEON_RED}Unexpected error fetching orderbook for {symbol}: {e}. Not retrying.{RESET}",
                exc_info=True,
            )
            return None

    lg.error(
        f"{NEON_RED}Max retries reached fetching orderbook for {symbol}. Last error: {last_exception}{RESET}"
    )
    return None


# --- Trading Analyzer Class ---
class TradingAnalyzer:
    """Analyzes trading data, generates signals, and provides risk management helpers.
    Uses Decimal for precision and market info for quantization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: dict[str, Any],
        market_info: dict[str, Any],  # Pass market info
    ) -> None:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("TradingAnalyzer requires a non-empty pandas DataFrame.")
        if not market_info or not isinstance(market_info, dict):
            raise ValueError("TradingAnalyzer requires a valid market_info dictionary.")

        self.df = df  # Expects index 'timestamp' and columns 'open', 'high', 'low', 'close', 'volume'
        self.logger = logger
        self.config = config
        self.market_info = market_info
        self.symbol = market_info.get("symbol", "UNKNOWN_SYMBOL")
        self.interval = config.get("interval", "UNKNOWN_INTERVAL")

        self.indicator_values: dict[
            str, float
        ] = {}  # Stores latest indicator float values
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = config.get("active_weight_set", "default")
        self.weights = config.get("weight_sets", {}).get(
            self.active_weight_set_name, {}
        )
        self.fib_levels_data: dict[
            str, Decimal
        ] = {}  # Stores calculated fib levels (Decimal)
        self.ta_column_names: dict[
            str, str | None
        ] = {}  # Stores actual TA column names
        self.break_even_triggered = False  # Track BE status per cycle

        if not self.weights:
            logger.error(
                f"{NEON_RED}Active weight set '{self.active_weight_set_name}' not found or empty in config for {self.symbol}. Weighting disabled.{RESET}"
            )

        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> str | None:
        """Helper to find the actual column name generated by pandas_ta."""
        cfg = self.config
        try:  # Safe float conversion for std dev
            bb_std_dev = float(
                cfg.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
            )
        except (ValueError, TypeError):
            bb_std_dev = DEFAULT_BOLLINGER_BANDS_STD_DEV

        expected_patterns = {  # Define expected patterns based on config values
            "ATR": [
                f"ATRr_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}",
                f"ATR_{cfg.get('atr_period', DEFAULT_ATR_PERIOD)}",
            ],
            "EMA_Short": [
                f"EMA_{cfg.get('ema_short_period', DEFAULT_EMA_SHORT_PERIOD)}"
            ],
            "EMA_Long": [f"EMA_{cfg.get('ema_long_period', DEFAULT_EMA_LONG_PERIOD)}"],
            "Momentum": [f"MOM_{cfg.get('momentum_period', DEFAULT_MOMENTUM_PERIOD)}"],
            "CCI": [
                f"CCI_{cfg.get('cci_window', DEFAULT_CCI_WINDOW)}_0.015"
            ],  # Default const suffix
            "Williams_R": [
                f"WILLR_{cfg.get('williams_r_window', DEFAULT_WILLIAMS_R_WINDOW)}"
            ],
            "MFI": [f"MFI_{cfg.get('mfi_window', DEFAULT_MFI_WINDOW)}"],
            "VWAP": ["VWAP", "VWAP_D"],  # Handle potential suffix
            "PSAR_long": [
                f"PSARl_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"
            ],
            "PSAR_short": [
                f"PSARs_{cfg.get('psar_af', DEFAULT_PSAR_AF)}_{cfg.get('psar_max_af', DEFAULT_PSAR_MAX_AF)}"
            ],
            "SMA10": [f"SMA_{cfg.get('sma_10_window', DEFAULT_SMA_10_WINDOW)}"],
            "StochRSI_K": [
                f"STOCHRSIk_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"
            ],
            "StochRSI_D": [
                f"STOCHRSId_{cfg.get('stoch_rsi_window', DEFAULT_STOCH_RSI_WINDOW)}_{cfg.get('stoch_rsi_rsi_window', DEFAULT_STOCH_WINDOW)}_{cfg.get('stoch_rsi_k', DEFAULT_K_WINDOW)}_{cfg.get('stoch_rsi_d', DEFAULT_D_WINDOW)}"
            ],
            "RSI": [f"RSI_{cfg.get('rsi_period', DEFAULT_RSI_WINDOW)}"],
            "BB_Lower": [
                f"BBL_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"
            ],
            "BB_Middle": [
                f"BBM_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"
            ],
            "BB_Upper": [
                f"BBU_{cfg.get('bollinger_bands_period', DEFAULT_BOLLINGER_BANDS_PERIOD)}_{bb_std_dev:.1f}"
            ],
            "Volume_MA": [
                f"VOL_SMA_{cfg.get('volume_ma_period', DEFAULT_VOLUME_MA_PERIOD)}"
            ],
        }

        patterns_to_check = expected_patterns.get(base_name, [])
        available_columns = result_df.columns.tolist()

        for pattern in patterns_to_check:  # Check exact patterns first
            if pattern in available_columns:
                return pattern

        # Fallback: Check base name prefix (case-insensitive)
        base_lower = base_name.lower()
        for col in available_columns:
            if col.lower().startswith(base_lower + "_"):
                return col
        # Fallback: Check if base name is present (less specific)
        for col in available_columns:
            if base_lower in col.lower():
                return col

        self.logger.warning(
            f"{NEON_YELLOW}Could not find column for indicator '{base_name}' in DF columns: {available_columns}{RESET}"
        )
        return None

    def _calculate_all_indicators(self) -> None:
        """Calculates all enabled indicators using pandas_ta and stores column names."""
        if self.df.empty:
            self.logger.warning(
                f"DataFrame empty, cannot calculate indicators for {self.symbol}."
            )
            return

        # Check for sufficient data length
        cfg = self.config
        indi_cfg = cfg.get("indicators", {})
        periods_needed = [
            cfg.get("atr_period", DEFAULT_ATR_PERIOD)
        ]  # ATR always needed
        if indi_cfg.get("ema_alignment", False) or cfg.get(
            "enable_ma_cross_exit", False
        ):
            periods_needed.extend(
                [
                    cfg.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD),
                    cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
                ]
            )
        for key, default in [
            ("momentum", DEFAULT_MOMENTUM_PERIOD),
            ("cci", DEFAULT_CCI_WINDOW),
            ("wr", DEFAULT_WILLIAMS_R_WINDOW),
            ("mfi", DEFAULT_MFI_WINDOW),
            ("sma_10", DEFAULT_SMA_10_WINDOW),
            ("rsi", DEFAULT_RSI_WINDOW),
            ("bollinger_bands", DEFAULT_BOLLINGER_BANDS_PERIOD),
            ("volume_confirmation", DEFAULT_VOLUME_MA_PERIOD),
        ]:
            if indi_cfg.get(key):
                periods_needed.append(
                    cfg.get(f"{key}_period", default)
                    if "period" in key
                    else cfg.get(f"{key}_window", default)
                )
        if indi_cfg.get("stoch_rsi"):
            periods_needed.append(
                cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW)
                + cfg.get("stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW)
            )

        min_required_data = max(periods_needed) + 20 if periods_needed else 50
        if len(self.df) < min_required_data:
            self.logger.warning(
                f"{NEON_YELLOW}Insufficient data ({len(self.df)} points) for {self.symbol} to calculate all indicators reliably (min recommended: {min_required_data}). Results may be inaccurate.{RESET}"
            )

        try:
            df_calc = self.df.copy()  # Work on a copy
            # Always calculate ATR
            df_calc.ta.atr(
                length=cfg.get("atr_period", DEFAULT_ATR_PERIOD), append=True
            )
            self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # Calculate other indicators based on flags
            if indi_cfg.get("ema_alignment", False) or cfg.get(
                "enable_ma_cross_exit", False
            ):
                df_calc.ta.ema(
                    length=cfg.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD),
                    append=True,
                )
                self.ta_column_names["EMA_Short"] = self._get_ta_col_name(
                    "EMA_Short", df_calc
                )
                df_calc.ta.ema(
                    length=cfg.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD),
                    append=True,
                )
                self.ta_column_names["EMA_Long"] = self._get_ta_col_name(
                    "EMA_Long", df_calc
                )
            if indi_cfg.get("momentum", False):
                df_calc.ta.mom(
                    length=cfg.get("momentum_period", DEFAULT_MOMENTUM_PERIOD),
                    append=True,
                )
                self.ta_column_names["Momentum"] = self._get_ta_col_name(
                    "Momentum", df_calc
                )
            if indi_cfg.get("cci", False):
                df_calc.ta.cci(
                    length=cfg.get("cci_window", DEFAULT_CCI_WINDOW), append=True
                )
                self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)
            if indi_cfg.get("wr", False):
                df_calc.ta.willr(
                    length=cfg.get("williams_r_window", DEFAULT_WILLIAMS_R_WINDOW),
                    append=True,
                )
                self.ta_column_names["Williams_R"] = self._get_ta_col_name(
                    "Williams_R", df_calc
                )
            if indi_cfg.get("mfi", False):
                df_calc.ta.mfi(
                    length=cfg.get("mfi_window", DEFAULT_MFI_WINDOW), append=True
                )
                self.ta_column_names["MFI"] = self._get_ta_col_name("MFI", df_calc)
            if indi_cfg.get("vwap", False):
                df_calc.ta.vwap(append=True)
                self.ta_column_names["VWAP"] = self._get_ta_col_name("VWAP", df_calc)
            if indi_cfg.get("psar", False):
                try:
                    psar_result = df_calc.ta.psar(
                        af=cfg.get("psar_af", DEFAULT_PSAR_AF),
                        max_af=cfg.get("psar_max_af", DEFAULT_PSAR_MAX_AF),
                    )
                    if psar_result is not None and not psar_result.empty:
                        for col in psar_result.columns:
                            df_calc[col] = psar_result[col]  # Append/Overwrite safely
                        self.ta_column_names["PSAR_long"] = self._get_ta_col_name(
                            "PSAR_long", df_calc
                        )
                        self.ta_column_names["PSAR_short"] = self._get_ta_col_name(
                            "PSAR_short", df_calc
                        )
                except Exception as e:
                    self.logger.error(f"Error calculating PSAR for {self.symbol}: {e}")
            if indi_cfg.get("sma_10", False):
                df_calc.ta.sma(
                    length=cfg.get("sma_10_window", DEFAULT_SMA_10_WINDOW), append=True
                )
                self.ta_column_names["SMA10"] = self._get_ta_col_name("SMA10", df_calc)
            if indi_cfg.get("stoch_rsi", False):
                try:
                    stochrsi_result = df_calc.ta.stochrsi(
                        length=cfg.get("stoch_rsi_window", DEFAULT_STOCH_RSI_WINDOW),
                        rsi_length=cfg.get(
                            "stoch_rsi_rsi_window", DEFAULT_STOCH_WINDOW
                        ),
                        k=cfg.get("stoch_rsi_k", DEFAULT_K_WINDOW),
                        d=cfg.get("stoch_rsi_d", DEFAULT_D_WINDOW),
                    )
                    if stochrsi_result is not None and not stochrsi_result.empty:
                        for col in stochrsi_result.columns:
                            df_calc[col] = stochrsi_result[col]
                        self.ta_column_names["StochRSI_K"] = self._get_ta_col_name(
                            "StochRSI_K", df_calc
                        )
                        self.ta_column_names["StochRSI_D"] = self._get_ta_col_name(
                            "StochRSI_D", df_calc
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error calculating StochRSI for {self.symbol}: {e}"
                    )
            if indi_cfg.get("rsi", False):
                df_calc.ta.rsi(
                    length=cfg.get("rsi_period", DEFAULT_RSI_WINDOW), append=True
                )
                self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)
            if indi_cfg.get("bollinger_bands", False):
                try:
                    bb_std_float = float(
                        cfg.get(
                            "bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV
                        )
                    )
                    bbands_result = df_calc.ta.bbands(
                        length=cfg.get(
                            "bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD
                        ),
                        std=bb_std_float,
                    )
                    if bbands_result is not None and not bbands_result.empty:
                        for col in bbands_result.columns:
                            df_calc[col] = bbands_result[col]
                        self.ta_column_names["BB_Lower"] = self._get_ta_col_name(
                            "BB_Lower", df_calc
                        )
                        self.ta_column_names["BB_Middle"] = self._get_ta_col_name(
                            "BB_Middle", df_calc
                        )
                        self.ta_column_names["BB_Upper"] = self._get_ta_col_name(
                            "BB_Upper", df_calc
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error calculating Bollinger Bands for {self.symbol}: {e}"
                    )
            if indi_cfg.get("volume_confirmation", False):
                try:
                    vol_ma_period = cfg.get(
                        "volume_ma_period", DEFAULT_VOLUME_MA_PERIOD
                    )
                    vol_ma_col_name = f"VOL_SMA_{vol_ma_period}"  # Custom name
                    df_calc[vol_ma_col_name] = ta.sma(
                        df_calc["volume"].fillna(0), length=vol_ma_period
                    )
                    self.ta_column_names["Volume_MA"] = vol_ma_col_name
                except Exception as e:
                    self.logger.error(
                        f"Error calculating Volume MA for {self.symbol}: {e}"
                    )

            self.df = df_calc  # Assign back the DF with indicators
            self.logger.debug(
                f"Finished indicator calculations for {self.symbol}. DF columns: {self.df.columns.tolist()}"
            )

        except (AttributeError, Exception) as e:
            self.logger.error(
                f"{NEON_RED}Error calculating indicators with pandas_ta for {self.symbol}: {e}{RESET}",
                exc_info=True,
            )

    def _update_latest_indicator_values(self) -> None:
        """Updates the indicator_values dict with the latest float values from self.df."""
        if self.df.empty or len(self.df) == 0:
            self.logger.warning(
                f"Cannot update latest values: DataFrame empty/short for {self.symbol}."
            )
            self.indicator_values = {}
            return

        try:
            latest = self.df.iloc[-1]
            if latest.isnull().all():
                self.logger.warning(
                    f"Cannot update latest values: Last row is all NaNs for {self.symbol}."
                )
                self.indicator_values = {}
                return

            updated_values = {}
            # Populate from calculated indicators
            for key, col_name in self.ta_column_names.items():
                if col_name and col_name in latest.index:
                    value = latest[col_name]
                    updated_values[key] = (
                        float(value) if pd.notna(value) else float("nan")
                    )
                else:
                    updated_values[key] = float(
                        "nan"
                    )  # Store NaN if column missing/invalid

            # Add essential price/volume data
            for base_col in ["open", "high", "low", "close", "volume"]:
                value = latest.get(base_col)
                updated_values[base_col.capitalize()] = (
                    float(value) if pd.notna(value) else float("nan")
                )

            self.indicator_values = updated_values
            valid_values = {
                k: f"{v:.5f}" if isinstance(v, float) and pd.notna(v) else v
                for k, v in self.indicator_values.items()
            }
            self.logger.debug(
                f"Latest indicator float values updated for {self.symbol}: {valid_values}"
            )

        except (IndexError, Exception) as e:
            self.logger.error(
                f"Error updating latest indicator values for {self.symbol}: {e}",
                exc_info=True,
            )
            self.indicator_values = {}

    # --- Precision and Market Info Helpers ---
    def get_min_tick_size(self) -> Decimal:
        """Gets the minimum price increment (tick size) from market info as Decimal."""
        try:
            # Prefer Bybit specific 'info.tickSize' if available
            tick_str = self.market_info.get("info", {}).get("tickSize")
            if tick_str:
                return Decimal(str(tick_str))

            # Fallback to standard 'precision.price' if it looks like a step size
            price_prec = self.market_info.get("precision", {}).get("price")
            if isinstance(price_prec, (float, str)):
                tick_dec = Decimal(str(price_prec))
                # The original error occurred here: Decimals don't have .is_integer()
                # Correct check: Use modulo or compare quantized value
                # if tick_dec > 0 and not (tick_dec.is_integer() and tick_dec > 1):
                if tick_dec > 0 and not (
                    tick_dec % 1 == 0 and tick_dec > 1
                ):  # Correct check
                    return tick_dec
            # Fallback to 'limits.price.min' if plausible
            min_price = self.market_info.get("limits", {}).get("price", {}).get("min")
            if min_price:
                min_tick = Decimal(str(min_price))
                if 0 < min_tick < 100:
                    return min_tick  # Heuristic check

        except (InvalidOperation, ValueError, TypeError, KeyError, Exception) as e:
            # Log the *actual* error now if it occurs, not the previous AttributeError
            self.logger.warning(
                f"Error determining tick size for {self.symbol}: {e}. Using fallback."
            )

        fallback_tick = Decimal("1E-8")  # Keep the original fallback value
        self.logger.warning(
            f"Using fallback tick size {fallback_tick} for {self.symbol}."
        )
        return fallback_tick

    def get_price_precision(self) -> int:
        """Gets price precision (number of decimal places) from market info."""
        try:
            min_tick = self.get_min_tick_size()
            if min_tick > 0:
                return abs(min_tick.normalize().as_tuple().exponent)
        except Exception:
            pass

        # Fallback: Use 'precision.price' if it's an integer
        try:
            price_precision_val = self.market_info.get("precision", {}).get("price")
            if isinstance(price_precision_val, int) and price_precision_val >= 0:
                return price_precision_val
        except Exception:
            pass

        default_precision = 4
        self.logger.warning(
            f"Using default price precision {default_precision} for {self.symbol}."
        )
        return default_precision

    # --- Fibonacci Calculation ---
    def calculate_fibonacci_levels(
        self, window: int | None = None
    ) -> dict[str, Decimal]:
        """Calculates Fibonacci retracement levels over a window using Decimal."""
        window = window or self.config.get("fibonacci_window", DEFAULT_FIB_WINDOW)
        if len(self.df) < window:
            self.logger.debug(
                f"Not enough data ({len(self.df)}) for Fib window ({window}) on {self.symbol}."
            )
            self.fib_levels_data = {}
            return {}

        df_slice = self.df.tail(window)
        try:
            high_raw = df_slice["high"].dropna().max()
            low_raw = df_slice["low"].dropna().min()
            if pd.isna(high_raw) or pd.isna(low_raw):
                self.logger.warning(
                    f"No valid high/low in last {window} periods for Fib on {self.symbol}."
                )
                return {}

            high = Decimal(str(high_raw))
            low = Decimal(str(low_raw))
            diff = high - low
            levels = {}
            min_tick = self.get_min_tick_size()

            if diff > 0 and min_tick > 0:
                for level_pct in FIB_LEVELS:
                    level_price_raw = high - (diff * Decimal(str(level_pct)))
                    # Quantize DOWN for upper levels, UP for lower levels (relative to high)
                    rounding = ROUND_DOWN if level_pct < 0.5 else ROUND_UP
                    level_price = (level_price_raw / min_tick).quantize(
                        Decimal("1"), rounding=rounding
                    ) * min_tick
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price
            elif min_tick > 0:  # Handle zero range
                level_price = (high / min_tick).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * min_tick
                for level_pct in FIB_LEVELS:
                    levels[f"Fib_{level_pct * 100:.1f}%"] = level_price
            else:  # Invalid tick size
                self.logger.warning(
                    f"Invalid min_tick_size for Fib quantization on {self.symbol}."
                )
                return {}

            self.fib_levels_data = levels
            log_levels = {
                k: f"{v:.{self.get_price_precision()}f}" for k, v in levels.items()
            }
            self.logger.debug(
                f"Calculated Fibonacci levels for {self.symbol}: {log_levels}"
            )
            return levels
        except Exception as e:
            self.logger.error(
                f"{NEON_RED}Fibonacci calculation error for {self.symbol}: {e}{RESET}",
                exc_info=True,
            )
            self.fib_levels_data = {}
            return {}

    def get_nearest_fibonacci_levels(
        self, current_price: Decimal, num_levels: int = 3
    ) -> list[tuple[str, Decimal]]:
        """Finds the N nearest Fibonacci levels (name, price) to the current price."""
        if not self.fib_levels_data or current_price is None or current_price <= 0:
            return []
        try:
            level_distances = [
                {"name": name, "level": level, "distance": abs(current_price - level)}
                for name, level in self.fib_levels_data.items()
                if isinstance(level, Decimal)
            ]
            level_distances.sort(key=lambda x: x["distance"])
            return [
                (item["name"], item["level"]) for item in level_distances[:num_levels]
            ]
        except Exception as e:
            self.logger.error(
                f"Error finding nearest Fib levels for {self.symbol}: {e}",
                exc_info=True,
            )
            return []

    # --- EMA Alignment Calculation ---
    def calculate_ema_alignment_score(self) -> float:
        """Calculates EMA alignment score. Returns float score or NaN."""
        ema_short = self.indicator_values.get("EMA_Short", float("nan"))
        ema_long = self.indicator_values.get("EMA_Long", float("nan"))
        current_price = self.indicator_values.get("Close", float("nan"))
        if math.isnan(ema_short) or math.isnan(ema_long) or math.isnan(current_price):
            return float("nan")

        if current_price > ema_short > ema_long:
            return 1.0  # Bullish
        elif current_price < ema_short < ema_long:
            return -1.0  # Bearish
        else:
            return 0.0  # Neutral/Mixed

    # --- Signal Generation & Scoring ---
    def generate_trading_signal(
        self, current_price: Decimal, orderbook_data: dict | None
    ) -> str:
        """Generates a trading signal (BUY/SELL/HOLD) based on weighted indicator scores."""
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Reset signals
        final_signal_score = Decimal("0.0")
        total_weight_applied = Decimal("0.0")
        active_indicator_count = 0
        nan_indicator_count = 0
        debug_scores = {}

        if not self.indicator_values or current_price is None or current_price <= 0:
            self.logger.warning(
                f"Cannot generate signal for {self.symbol}: Invalid inputs (indicators empty or price invalid)."
            )
            return "HOLD"

        active_weights = self.config.get("weight_sets", {}).get(
            self.active_weight_set_name
        )
        if not active_weights:
            self.logger.error(
                f"Weight set '{self.active_weight_set_name}' missing/empty for {self.symbol}. HOLDING."
            )
            return "HOLD"

        for indicator_key, enabled in self.config.get("indicators", {}).items():
            if not enabled:
                continue
            weight_str = active_weights.get(indicator_key)
            if weight_str is None:
                continue
            try:
                weight = Decimal(str(weight_str))
                assert weight != 0
            except (InvalidOperation, ValueError, TypeError, AssertionError):
                continue  # Skip invalid/zero weights

            check_method_name = f"_check_{indicator_key}"
            if hasattr(self, check_method_name) and callable(
                getattr(self, check_method_name)
            ):
                method = getattr(self, check_method_name)
                indicator_score_float = float("nan")
                try:
                    if indicator_key == "orderbook":
                        indicator_score_float = (
                            method(orderbook_data, current_price)
                            if orderbook_data
                            else float("nan")
                        )
                    else:
                        indicator_score_float = method()
                except Exception as e:
                    self.logger.error(
                        f"Error in check method {check_method_name} for {self.symbol}: {e}",
                        exc_info=True,
                    )

                debug_scores[indicator_key] = (
                    f"{indicator_score_float:.2f}"
                    if not math.isnan(indicator_score_float)
                    else "NaN"
                )
                if not math.isnan(indicator_score_float):
                    try:
                        clamped_score = max(-1.0, min(1.0, indicator_score_float))
                        final_signal_score += Decimal(str(clamped_score)) * weight
                        total_weight_applied += weight
                        active_indicator_count += 1
                    except (InvalidOperation, ValueError, TypeError) as calc_err:
                        self.logger.error(
                            f"Error processing score for {indicator_key}: {calc_err}"
                        )
                        nan_indicator_count += 1
                else:
                    nan_indicator_count += 1
            else:
                self.logger.warning(
                    f"Check method '{check_method_name}' not found for enabled indicator: {indicator_key}"
                )

        if total_weight_applied == 0:
            self.logger.warning(
                f"No indicators contributed to signal score for {self.symbol}. HOLDING."
            )
            final_signal = "HOLD"
        else:
            try:
                threshold_key = (
                    "scalping_signal_threshold"
                    if self.active_weight_set_name == "scalping"
                    else "signal_score_threshold"
                )
                default_threshold = (
                    2.5 if self.active_weight_set_name == "scalping" else 1.5
                )
                threshold = Decimal(
                    str(self.config.get(threshold_key, default_threshold))
                )
            except (InvalidOperation, ValueError, TypeError):
                threshold = Decimal(str(default_threshold))

            if final_signal_score >= threshold:
                final_signal = "BUY"
            elif final_signal_score <= -threshold:
                final_signal = "SELL"
            else:
                final_signal = "HOLD"

        # --- Log Summary ---
        price_precision = self.get_price_precision()
        score_details = ", ".join([f"{k}: {v}" for k, v in debug_scores.items()])
        log_msg = (
            f"Signal Calculation Summary ({self.symbol} @ {current_price:.{price_precision}f}):\n"
            f"  Weight Set: {self.active_weight_set_name}\n"
            f"  Indicators Used: {active_indicator_count} ({nan_indicator_count} NaN)\n"
            f"  Total Weight Applied: {total_weight_applied:.3f}\n"
            f"  Final Weighted Score: {final_signal_score:.4f}\n"
            f"  Signal Threshold: +/- {threshold:.3f}\n"
            f"  ==> Final Signal: {NEON_GREEN if final_signal == 'BUY' else NEON_RED if final_signal == 'SELL' else NEON_YELLOW}{final_signal}{RESET}"
        )
        self.logger.info(log_msg)
        self.logger.debug(f"  Detailed Scores: {score_details}")

        if final_signal in self.signals:
            self.signals[final_signal] = 1
            if final_signal != "HOLD":
                self.signals["HOLD"] = 0

        return final_signal

    # --- Indicator Check Methods (Return float score -1.0 to 1.0, or NaN) ---
    def _check_ema_alignment(self) -> float:
        return self.calculate_ema_alignment_score()

    def _check_momentum(self) -> float:
        mom = self.indicator_values.get("Momentum", float("nan"))
        if math.isnan(mom):
            return float("nan")
        # Simple scaling (adjust scale_factor based on typical range for asset/TF)
        scale_factor = 0.1  # Example: assumes typical mom range is +/- 10
        return max(-1.0, min(1.0, mom * scale_factor))

    def _check_volume_confirmation(self) -> float:
        vol = self.indicator_values.get("Volume", float("nan"))
        vol_ma = self.indicator_values.get("Volume_MA", float("nan"))
        if math.isnan(vol) or math.isnan(vol_ma) or vol_ma <= 0:
            return float("nan")
        try:
            mult = float(self.config.get("volume_confirmation_multiplier", 1.5))
        except:
            mult = 1.5
        if vol > vol_ma * mult:
            return 0.7  # High volume confirmation
        if vol < vol_ma / mult:
            return -0.4  # Low volume lack of confirmation
        return 0.0  # Neutral

    def _check_stoch_rsi(self) -> float:
        k = self.indicator_values.get("StochRSI_K", float("nan"))
        d = self.indicator_values.get("StochRSI_D", float("nan"))
        if math.isnan(k) or math.isnan(d):
            return float("nan")
        try:
            oversold = float(self.config.get("stoch_rsi_oversold_threshold", 25.0))
            overbought = float(self.config.get("stoch_rsi_overbought_threshold", 75.0))
        except:
            oversold, overbought = 25.0, 75.0
        score = 0.0
        if k < oversold and d < oversold:
            score = 1.0
        elif k > overbought and d > overbought:
            score = -1.0
        elif k > d:
            score = max(score, 0.5)  # K over D bullish bias
        elif k < d:
            score = min(score, -0.5)  # K under D bearish bias
        return max(-1.0, min(1.0, score))

    def _check_rsi(self) -> float:
        rsi = self.indicator_values.get("RSI", float("nan"))
        if math.isnan(rsi):
            return float("nan")
        if rsi <= 30:
            return 1.0
        if rsi >= 70:
            return -1.0
        if 30 < rsi < 70:
            return 1.0 - (rsi - 30.0) * (2.0 / 40.0)  # Linear scale between levels
        return 0.0

    def _check_cci(self) -> float:
        cci = self.indicator_values.get("CCI", float("nan"))
        if math.isnan(cci):
            return float("nan")
        if cci <= -150:
            return 1.0
        if cci >= 150:
            return -1.0
        if cci <= -100:
            return 0.6
        if cci >= 100:
            return -0.6
        if -100 < cci < 100:
            return -(cci / 100.0) * 0.6  # Linear scale between levels
        return 0.0

    def _check_wr(self) -> float:  # Williams %R
        wr = self.indicator_values.get("Williams_R", float("nan"))
        if math.isnan(wr):
            return float("nan")
        if wr <= -80:
            return 1.0
        if wr >= -20:
            return -1.0
        if -80 < wr < -20:
            return 1.0 - (wr - (-80.0)) * (2.0 / 60.0)  # Linear scale
        return 0.0

    def _check_psar(self) -> float:
        psar_l = self.indicator_values.get(
            "PSAR_long", float("nan")
        )  # Below price if active
        psar_s = self.indicator_values.get(
            "PSAR_short", float("nan")
        )  # Above price if active
        if not math.isnan(psar_l) and math.isnan(psar_s):
            return 1.0  # Uptrend
        if math.isnan(psar_l) and not math.isnan(psar_s):
            return -1.0  # Downtrend
        return 0.0  # Neutral/Ambiguous

    def _check_sma_10(self) -> float:
        sma = self.indicator_values.get("SMA10", float("nan"))
        close = self.indicator_values.get("Close", float("nan"))
        if math.isnan(sma) or math.isnan(close):
            return float("nan")
        if close > sma:
            return 0.6
        if close < sma:
            return -0.6
        return 0.0

    def _check_vwap(self) -> float:
        vwap = self.indicator_values.get("VWAP", float("nan"))
        close = self.indicator_values.get("Close", float("nan"))
        if math.isnan(vwap) or math.isnan(close):
            return float("nan")
        if close > vwap:
            return 0.7
        if close < vwap:
            return -0.7
        return 0.0

    def _check_mfi(self) -> float:
        mfi = self.indicator_values.get("MFI", float("nan"))
        if math.isnan(mfi):
            return float("nan")
        if mfi <= 20:
            return 1.0
        if mfi >= 80:
            return -1.0
        if 20 < mfi < 80:
            return 1.0 - (mfi - 20.0) * (2.0 / 60.0)  # Linear scale
        return 0.0

    def _check_bollinger_bands(self) -> float:
        bbl = self.indicator_values.get("BB_Lower", float("nan"))
        bbu = self.indicator_values.get("BB_Upper", float("nan"))
        close = self.indicator_values.get("Close", float("nan"))
        if math.isnan(bbl) or math.isnan(bbu) or math.isnan(close):
            return float("nan")
        if close <= bbl:
            return 1.0  # Touch/below lower -> Bullish mean revert
        if close >= bbu:
            return -1.0  # Touch/above upper -> Bearish mean revert
        # Optional: Add score based on proximity to bands or middle line
        return 0.0  # Inside bands

    def _check_orderbook(
        self, orderbook_data: dict | None, current_price: Decimal
    ) -> float:
        """Analyzes Order Book Imbalance (OBI)."""
        if not orderbook_data:
            return float("nan")
        try:
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])
            if not bids or not asks:
                return float("nan")
            levels = 10  # Check top N levels
            bid_vol = sum(Decimal(str(b[1])) for b in bids[:levels])
            ask_vol = sum(Decimal(str(a[1])) for a in asks[:levels])
            total_vol = bid_vol + ask_vol
            if total_vol <= 0:
                return 0.0  # Neutral if no volume
            obi = (bid_vol - ask_vol) / total_vol  # OBI range: -1 to +1
            self.logger.debug(
                f"OBI ({levels} levels): BidVol={bid_vol:.4f}, AskVol={ask_vol:.4f}, OBI={obi:.4f}"
            )
            return float(obi)  # Return OBI as score
        except (InvalidOperation, ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Error calculating OBI for {self.symbol}: {e}")
            return float("nan")
        except Exception as e:
            self.logger.error(f"Unexpected OBI error: {e}")
            return float("nan")

    # --- Risk Management Calculations ---
    def calculate_entry_tp_sl(
        self, entry_price: Decimal, signal: str
    ) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates potential TP/SL based on entry, ATR, config, quantized to tick size."""
        if signal not in ["BUY", "SELL"] or entry_price is None or entry_price <= 0:
            return entry_price, None, None
        atr_float = self.indicator_values.get("ATR")
        if atr_float is None or math.isnan(atr_float) or atr_float <= 0:
            self.logger.warning(
                f"Cannot calculate TP/SL for {self.symbol}: Invalid ATR ({atr_float})."
            )
            return entry_price, None, None

        try:
            atr = Decimal(str(atr_float))
            tp_mult = Decimal(str(self.config.get("take_profit_multiple", 1.0)))
            sl_mult = Decimal(str(self.config.get("stop_loss_multiple", 1.5)))
            min_tick = self.get_min_tick_size()
            if min_tick <= 0:
                raise ValueError("Invalid min tick size for quantization")

            tp_offset = atr * tp_mult
            sl_offset = atr * sl_mult
            take_profit, stop_loss = None, None

            if signal == "BUY":
                tp_raw = entry_price + tp_offset
                sl_raw = entry_price - sl_offset
                # Quantize TP UP, SL DOWN (away from entry)
                take_profit = (tp_raw / min_tick).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * min_tick
                stop_loss = (sl_raw / min_tick).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * min_tick
            elif signal == "SELL":
                tp_raw = entry_price - tp_offset
                sl_raw = entry_price + sl_offset
                # Quantize TP DOWN, SL UP (away from entry)
                take_profit = (tp_raw / min_tick).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) * min_tick
                stop_loss = (sl_raw / min_tick).quantize(
                    Decimal("1"), rounding=ROUND_UP
                ) * min_tick

            # Validation: Ensure SL/TP are strictly beyond entry by >= 1 tick and positive
            if signal == "BUY":
                if stop_loss >= entry_price:
                    stop_loss = (entry_price - min_tick).quantize(
                        min_tick, rounding=ROUND_DOWN
                    )
                if take_profit <= entry_price:
                    take_profit = (entry_price + min_tick).quantize(
                        min_tick, rounding=ROUND_UP
                    )
            elif signal == "SELL":
                if stop_loss <= entry_price:
                    stop_loss = (entry_price + min_tick).quantize(
                        min_tick, rounding=ROUND_UP
                    )
                if take_profit >= entry_price:
                    take_profit = (entry_price - min_tick).quantize(
                        min_tick, rounding=ROUND_DOWN
                    )

            if stop_loss is not None and stop_loss <= 0:
                self.logger.error(
                    f"Calculated SL is zero/negative ({stop_loss}). Setting SL to None."
                )
                stop_loss = None
            if take_profit is not None and take_profit <= 0:
                self.logger.error(
                    f"Calculated TP is zero/negative ({take_profit}). Setting TP to None."
                )
                take_profit = None

            prec = self.get_price_precision()
            tp_log = f"{take_profit:.{prec}f}" if take_profit else "N/A"
            sl_log = f"{stop_loss:.{prec}f}" if stop_loss else "N/A"
            self.logger.debug(
                f"Calculated TP/SL for {self.symbol} {signal}: Entry={entry_price:.{prec}f}, TP={tp_log}, SL={sl_log} (ATR={atr:.{prec + 1}f})"
            )
            return entry_price, take_profit, stop_loss

        except (InvalidOperation, ValueError, TypeError, Exception) as e:
            self.logger.error(
                f"{NEON_RED}Error calculating TP/SL for {self.symbol}: {e}{RESET}",
                exc_info=True,
            )
            return entry_price, None, None


# --- Trading Logic Helper Functions ---


def fetch_balance(
    exchange: ccxt.Exchange, currency: str, logger: logging.Logger
) -> Decimal | None:
    """Fetches available balance for a currency, handling Bybit V5 structures/retries."""
    lg = logger
    last_exception = None
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            balance_info = None
            # Bybit V5: Try specific account types relevant to futures/spot
            # Order matters: Try most likely first (CONTRACT for futures, UNIFIED/SPOT for spot)
            # Determine likely type from exchange settings if possible? Hard without market context here.
            account_types_to_try = ["CONTRACT", "UNIFIED", "SPOT"]  # Common types
            successful_acc_type = None

            for acc_type in account_types_to_try:
                try:
                    lg.debug(
                        f"Fetching balance using params={{'accountType': '{acc_type}'}} for {currency} (Attempt {attempt + 1})"
                    )
                    balance_info = exchange.fetch_balance(
                        params={"accountType": acc_type}
                    )
                    successful_acc_type = acc_type
                    # Basic check if the response seems valid for this account type
                    if balance_info and (
                        "info" in balance_info or currency in balance_info
                    ):
                        lg.debug(
                            f"Received balance structure using accountType '{acc_type}'."
                        )
                        break  # Proceed to parse this structure
                    else:
                        lg.debug(
                            f"Balance structure for accountType '{acc_type}' seems empty. Trying next."
                        )
                        balance_info = None  # Reset to try next type
                except ccxt.ExchangeError as e:
                    # Ignore "account type not support" errors and try the next type
                    ignore_msgs = [
                        "account type not support",
                        "invalid account type",
                        "accounttype invalid",
                        "10001",
                    ]  # Added 10001
                    if (
                        any(msg in str(e).lower() for msg in ignore_msgs)
                        or getattr(e, "code", None) == 10001
                    ):
                        lg.debug(
                            f"Account type '{acc_type}' not supported or error fetching: {e}. Trying next."
                        )
                        continue
                    else:
                        raise e  # Re-raise other exchange errors
                # Let outer try-except handle NetworkError, RateLimitExceeded etc.

            # If specific types failed, try default fetch (might work for some exchanges/setups)
            if not balance_info:
                lg.debug(
                    f"Fetching balance using default parameters for {currency} (Attempt {attempt + 1})..."
                )
                try:
                    balance_info = exchange.fetch_balance()
                    successful_acc_type = "Default"  # Mark as default fetch
                except ccxt.ExchangeError as e_default:
                    lg.warning(f"Default balance fetch also failed: {e_default}")
                    last_exception = e_default
                    continue  # Go to next retry attempt

            # --- Parse the balance_info ---
            if not balance_info:
                lg.warning(
                    f"Failed to fetch any balance information (Attempt {attempt + 1})."
                )
                last_exception = ValueError("API returned no balance information")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            available_balance_str = None
            # 1. Standard CCXT: balance_info[currency]['free']
            if (
                currency in balance_info
                and isinstance(balance_info.get(currency), dict)
                and balance_info[currency].get("free") is not None
            ):
                available_balance_str = str(balance_info[currency]["free"])
                lg.debug(
                    f"Found balance via standard ccxt structure ['{currency}']['free']: {available_balance_str}"
                )
            # 2. Bybit V5 Nested: info -> result -> list -> coin[] (check relevant account type)
            elif "info" in balance_info and isinstance(
                balance_info["info"].get("result", {}).get("list"), list
            ):
                balance_list = balance_info["info"]["result"]["list"]
                for account in balance_list:
                    current_account_type = account.get("accountType")
                    # Match the successful account type OR check all if default fetch was used
                    if (
                        successful_acc_type == "Default"
                        or current_account_type == successful_acc_type
                    ):
                        coin_list = account.get("coin")
                        if isinstance(coin_list, list):
                            for coin_data in coin_list:
                                if coin_data.get("coin") == currency:
                                    # Prefer 'availableToWithdraw' or 'availableBalance'
                                    free = coin_data.get(
                                        "availableToWithdraw",
                                        coin_data.get("availableBalance"),
                                    )
                                    if free is None:
                                        free = coin_data.get(
                                            "walletBalance"
                                        )  # Fallback
                                    if free is not None:
                                        available_balance_str = str(free)
                                        lg.debug(
                                            f"Found balance via Bybit V5 nested structure: {available_balance_str} {currency} (Account: {current_account_type or 'Unknown'})"
                                        )
                                        break
                            if available_balance_str is not None:
                                break
                if available_balance_str is None:
                    lg.warning(
                        f"{currency} not found within Bybit V5 'info.result.list[].coin[]' for account type '{successful_acc_type}'."
                    )
            # 3. Fallback: Top-level 'free' dictionary
            elif (
                "free" in balance_info
                and isinstance(balance_info.get("free"), dict)
                and currency in balance_info["free"]
            ):
                available_balance_str = str(balance_info["free"][currency])
                lg.debug(
                    f"Found balance via top-level 'free' dictionary: {available_balance_str} {currency}"
                )

            # 4. Last Resort: Check 'total' balance if 'free' still missing
            if available_balance_str is None:
                # Try standard total
                total_str = (
                    str(balance_info.get(currency, {}).get("total", ""))
                    if currency in balance_info
                    else ""
                )
                # Try Bybit V5 nested total ('walletBalance')
                if (
                    not total_str
                    and "info" in balance_info
                    and isinstance(
                        balance_info["info"].get("result", {}).get("list"), list
                    )
                ):
                    balance_list = balance_info["info"]["result"]["list"]
                    for account in balance_list:
                        current_account_type = account.get("accountType")
                        if (
                            successful_acc_type == "Default"
                            or current_account_type == successful_acc_type
                        ):
                            coin_list = account.get("coin")
                            if isinstance(coin_list, list):
                                for coin_data in coin_list:
                                    if coin_data.get("coin") == currency:
                                        total = coin_data.get("walletBalance")
                                        if total is not None:
                                            total_str = str(total)
                                            break
                                if total_str:
                                    break
                        if total_str:
                            break

                if total_str:
                    lg.warning(
                        f"{NEON_YELLOW}Using 'total' balance ({total_str}) as fallback for {currency}.{RESET}"
                    )
                    available_balance_str = total_str
                else:
                    lg.error(
                        f"{NEON_RED}Could not determine any balance for {currency}. Structure: {balance_info}{RESET}"
                    )
                    return None  # Critical failure

            # --- Convert to Decimal ---
            try:
                final_balance = Decimal(available_balance_str)
                if final_balance >= 0:
                    lg.info(f"Available {currency} balance: {final_balance:.4f}")
                    return final_balance
                else:
                    lg.error(
                        f"Parsed balance for {currency} is negative ({final_balance})."
                    )
                    return None
            except (InvalidOperation, ValueError, TypeError) as e:
                lg.error(
                    f"Failed to convert balance '{available_balance_str}' to Decimal for {currency}: {e}."
                )
                return None

        except ccxt.RateLimitExceeded as e:
            last_exception = e
            wait_time = _handle_rate_limit(e, lg)
            lg.warning(
                f"Rate limit hit fetching balance for {currency}. Retrying in {wait_time}s... (Attempt {attempt + 1})"
            )
            time.sleep(wait_time)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exception = e
            lg.warning(
                f"Network error fetching balance for {currency}: {e}. Retrying in {RETRY_DELAY_SECONDS}s... (Attempt {attempt + 1})"
            )
            time.sleep(RETRY_DELAY_SECONDS)
        except ccxt.AuthenticationError as e:
            lg.error(
                f"Authentication error fetching balance: {e}. Check API keys/perms. Not retrying."
            )
            raise e
        except ccxt.ExchangeError as e:
            lg.error(f"Exchange error fetching balance: {e}. Not retrying.")
            last_exception = e
            return None
        except Exception as e:
            lg.error(
                f"Unexpected error fetching balance: {e}. Not retrying.", exc_info=True
            )
            last_exception = e
            return None

    lg.error(
        f"{NEON_RED}Max retries reached fetching balance for {currency}. Last error: {last_exception}{RESET}"
    )
    return None


def get_market_info(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> dict | None:
    """Gets market information, ensuring markets are loaded, enhances with useful flags."""
    lg = logger
    try:
        if not exchange.markets or symbol not in exchange.markets:
            lg.info(f"Market info for {symbol} not loaded, reloading markets...")
            exchange.load_markets(reload=True)

        if not exchange.markets or symbol not in exchange.markets:
            lg.error(
                f"{NEON_RED}Market {symbol} not found after reloading markets.{RESET}"
            )
            return None

        market = exchange.market(symbol)
        if market:
            market_type = market.get("type", "unknown")  # spot, swap, future
            market["is_contract"] = market.get("contract", False) or market_type in [
                "swap",
                "future",
            ]
            market["is_linear"] = market.get(
                "linear",
                market_type == "linear"
                or (market_type == "swap" and market.get("quote", "").endswith("USDT")),
            )  # Heuristic for linear swap
            market["is_inverse"] = market.get(
                "inverse",
                market_type == "inverse"
                or (market_type == "swap" and not market["is_linear"]),
            )
            market["contract_type"] = (
                "Linear"
                if market["is_linear"]
                else "Inverse"
                if market["is_inverse"]
                else "Spot/Other"
            )
            # Ensure market ID is present
            if "id" not in market or not market["id"]:
                market["id"] = market.get("info", {}).get(
                    "symbol", symbol.replace("/", "").split(":")[0]
                )  # Fallback ID

            precision = market.get("precision", {})
            limits = market.get("limits", {})
            lg.debug(
                f"Market Info ({symbol}): ID={market.get('id')}, Type={market_type}, Contract={market['is_contract']} ({market['contract_type']}), "
                f"Precision(Price/Amount/Tick): {precision.get('price')}/{precision.get('amount')}/{market.get('info', {}).get('tickSize', 'N/A')}, "
                f"Limits(Amount Min/Max): {limits.get('amount', {}).get('min')}/{limits.get('amount', {}).get('max')}, "
                f"Limits(Cost Min/Max): {limits.get('cost', {}).get('min')}/{limits.get('cost', {}).get('max')}, "
                f"Contract Size: {market.get('contractSize', 'N/A')}"
            )
            return market
        else:
            lg.error(f"Market dictionary not found for {symbol} even after check.")
            return None
    except (ccxt.BadSymbol, ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
        lg.error(
            f"{NEON_RED}Error getting market info for {symbol}: {e}{RESET}",
            exc_info=True,
        )
    return None


def _manual_amount_rounding(
    size: Decimal, precision_val: Any, lg: logging.Logger, symbol: str
) -> Decimal:
    """Manual fallback for rounding amount DOWN based on precision (step or decimals)."""
    final_size = size
    step_size = None
    num_decimals = None
    try:
        if isinstance(precision_val, int) and precision_val >= 0:
            num_decimals = precision_val
        elif precision_val is not None:
            step_size = Decimal(str(precision_val))
            assert step_size > 0
    except:
        step_size = None
        num_decimals = None  # Invalidate on error

    if step_size is not None:
        final_size = (size // step_size) * step_size  # Floor division
        lg.debug(
            f"Applied manual amount step ({step_size}), rounded down: {size:.8f} -> {final_size}"
        )
    elif num_decimals is not None:
        rounding_factor = Decimal("1e-" + str(num_decimals))
        final_size = size.quantize(rounding_factor, rounding=ROUND_DOWN)
        lg.debug(
            f"Applied manual amount precision ({num_decimals} decimals), rounded down: {size:.8f} -> {final_size}"
        )
    else:
        lg.warning(
            f"Amount precision '{precision_val}' invalid for manual rounding ({symbol}). Using unrounded: {size:.8f}"
        )
    return final_size


def calculate_position_size(
    balance: Decimal,
    risk_per_trade: float,
    initial_stop_loss_price: Decimal,
    entry_price: Decimal,
    market_info: dict,
    exchange: ccxt.Exchange,
    logger: logging.Logger,
) -> Decimal | None:
    """Calculates position size based on risk, SL, balance, market constraints (Decimal)."""
    lg = logger
    symbol = market_info.get("symbol", "???")
    quote = market_info.get("quote", QUOTE_CURRENCY)
    base = market_info.get("base", "BASE")
    is_contract = market_info.get("is_contract", False)
    is_linear = market_info.get("is_linear", True)
    is_inverse = market_info.get("is_inverse", False)
    size_unit = "Contracts" if is_contract else base

    # --- Input Validation ---
    if (
        balance <= 0
        or not (0 < risk_per_trade < 1)
        or initial_stop_loss_price is None
        or entry_price <= 0
        or initial_stop_loss_price == entry_price
    ):
        lg.error(
            f"Position sizing failed ({symbol}): Invalid inputs (Balance={balance}, Risk={risk_per_trade}, SL={initial_stop_loss_price}, Entry={entry_price})."
        )
        return None
    if initial_stop_loss_price <= 0:
        lg.warning(
            f"Position sizing ({symbol}): Initial SL price ({initial_stop_loss_price}) is zero/negative."
        )

    try:
        risk_amount_quote = balance * Decimal(str(risk_per_trade))
        sl_distance_per_unit = abs(entry_price - initial_stop_loss_price)
        if sl_distance_per_unit <= 0:
            lg.error(
                f"Position sizing failed ({symbol}): SL distance zero/negative ({sl_distance_per_unit})."
            )
            return None

        contract_size_str = market_info.get("contractSize", "1")
        contract_size = Decimal("1")
        try:
            contract_size = Decimal(str(contract_size_str))
            assert contract_size > 0
        except:
            lg.warning(
                f"Invalid contract size '{contract_size_str}' for {symbol}. Defaulting to 1."
            )
            contract_size = Decimal("1")

        # --- Calculate Initial Size ---
        calculated_size = Decimal("0")
        if is_linear or not is_contract:  # Spot or Linear Contract
            # Size (Base/Contracts) = Risk Amount (Quote) / (SL Distance (Quote/Base) * Contract Size (Base/Contract))
            risk_per_contract_quote = sl_distance_per_unit * contract_size
            if risk_per_contract_quote <= 0:
                raise ValueError("Risk per contract is zero/negative")
            calculated_size = risk_amount_quote / risk_per_contract_quote
        elif is_inverse:  # Inverse Contract
            # Size (Contracts) = Risk Amount (Quote) / Risk per Contract (Quote)
            # Risk per Contract (Quote) = SL Distance (Quote/Base) * Value of 1 Contract (Base)
            # Value of 1 Contract (Base) = Contract Size (Quote/Contract) / Entry Price (Quote/Base)
            value_of_1_contract_base = (
                contract_size / entry_price
            )  # Assuming contract_size is Quote value (e.g., 1 USD)
            risk_per_contract_quote = sl_distance_per_unit * value_of_1_contract_base
            if risk_per_contract_quote <= 0:
                raise ValueError("Inverse risk per contract is zero/negative")
            calculated_size = risk_amount_quote / risk_per_contract_quote
        else:
            raise ValueError("Unknown contract type for sizing")

        lg.info(
            f"Position Sizing ({symbol}): Balance={balance:.2f} {quote}, Risk={risk_per_trade:.2%}, RiskAmt={risk_amount_quote:.4f} {quote}"
        )
        lg.info(
            f"  Entry={entry_price}, SL={initial_stop_loss_price}, SLDist={sl_distance_per_unit}"
        )
        lg.info(
            f"  ContractSize={contract_size}, Initial Calc. Size = {calculated_size:.8f} {size_unit}"
        )

        # --- Apply Market Limits and Precision ---
        limits = market_info.get("limits", {})
        amount_limits = limits.get("amount", {})
        cost_limits = limits.get("cost", {})
        precision = market_info.get("precision", {})
        amount_precision_val = precision.get("amount")

        def get_limit(d: dict, k: str, default: Decimal) -> Decimal:
            v = d.get(k)
            return Decimal(str(v)) if v is not None else default

        min_amount = get_limit(amount_limits, "min", Decimal("0"))
        max_amount = get_limit(amount_limits, "max", Decimal("inf"))
        min_cost = get_limit(cost_limits, "min", Decimal("0"))
        max_cost = get_limit(cost_limits, "max", Decimal("inf"))

        # 1. Adjust for MIN/MAX AMOUNT
        adjusted_size = max(min_amount, min(calculated_size, max_amount))
        if adjusted_size != calculated_size:
            lg.warning(
                f"Size adjusted due to amount limits: {calculated_size:.8f} -> {adjusted_size:.8f}"
            )

        # 2. Check COST limits
        cost_adjusted_size = adjusted_size
        cost_limit_adjusted = False
        current_cost = Decimal("0")
        if is_linear or not is_contract:
            current_cost = adjusted_size * entry_price * contract_size
        elif is_inverse:
            current_cost = (
                adjusted_size * contract_size
            )  # Assumes contract_size is Quote value

        lg.debug(
            f"  Cost Check: Adj. Size={adjusted_size:.8f}, Est. Cost={current_cost:.4f} (Min={min_cost}, Max={max_cost})"
        )
        if (
            min_cost > 0
            and current_cost < min_cost
            and not math.isclose(float(current_cost), float(min_cost), rel_tol=1e-6)
        ):
            lg.warning(
                f"Estimated cost {current_cost:.4f} < min cost {min_cost:.4f}. Attempting size increase."
            )
            required_size = Decimal("0")
            if is_linear or not is_contract:
                required_size = min_cost / (entry_price * contract_size)
            elif is_inverse:
                required_size = min_cost / contract_size
            if required_size > max_amount:
                lg.error(
                    f"Cannot meet min cost ({min_cost}) without exceeding max amount ({max_amount}). ABORT."
                )
                return None
            cost_adjusted_size = max(
                required_size, min_amount
            )  # Ensure still meets min amount
            lg.info(f"  Adjusting size to meet min cost: {cost_adjusted_size:.8f}")
            cost_limit_adjusted = True
        elif max_cost > 0 and current_cost > max_cost:
            lg.warning(
                f"Estimated cost {current_cost:.4f} > max cost {max_cost:.4f}. Reducing size."
            )
            allowed_size = Decimal("0")
            if is_linear or not is_contract:
                allowed_size = max_cost / (entry_price * contract_size)
            elif is_inverse:
                allowed_size = max_cost / contract_size
            if allowed_size < min_amount:
                lg.error(
                    f"Cannot meet max cost ({max_cost}) without going below min amount ({min_amount}). ABORT."
                )
                return None
            cost_adjusted_size = allowed_size
            lg.info(f"  Reducing size to meet max cost: {cost_adjusted_size:.8f}")
            cost_limit_adjusted = True

        # 3. Apply Amount Precision (rounding DOWN)
        final_size = Decimal("0")
        try:  # Use ccxt helper first
            formatted_size_str = exchange.amount_to_precision(
                symbol, float(cost_adjusted_size), padding_mode=exchange.TRUNCATE
            )
            final_size = Decimal(formatted_size_str)
            lg.info(
                f"Applied amount precision (Truncated): {cost_adjusted_size:.8f} -> {final_size} {size_unit}"
            )
        except (ccxt.BaseError, ValueError, TypeError, Exception) as fmt_err:
            lg.warning(
                f"Using manual rounding due to amount_to_precision error: {fmt_err}"
            )
            final_size = _manual_amount_rounding(
                cost_adjusted_size, amount_precision_val, lg, symbol
            )

        # --- Final Validation ---
        if final_size <= 0:
            lg.error(f"Final size is zero/negative ({final_size}). ABORT.")
            return None
        if final_size < min_amount and not math.isclose(
            float(final_size), float(min_amount), rel_tol=1e-9
        ):
            lg.error(f"Final size {final_size} < min amount {min_amount}. ABORT.")
            return None
        # Re-check min cost if size was reduced by rounding (and not by cost limits initially)
        if not cost_limit_adjusted and min_cost > 0:
            final_cost = Decimal("0")
            if is_linear or not is_contract:
                final_cost = final_size * entry_price * contract_size
            elif is_inverse:
                final_cost = final_size * contract_size
            if final_cost < min_cost and not math.isclose(
                float(final_cost), float(min_cost), rel_tol=1e-6
            ):
                lg.error(
                    f"Final size {final_size} causes cost {final_cost:.4f} < min cost {min_cost}. ABORT."
                )
                return None

        lg.info(
            f"{NEON_GREEN}Final calculated position size for {symbol}: {final_size} {size_unit}{RESET}"
        )
        return final_size

    except (
        KeyError,
        InvalidOperation,
        ValueError,
        TypeError,
        ZeroDivisionError,
        Exception,
    ) as e:
        lg.error(
            f"{NEON_RED}Position sizing error ({symbol}): {e}{RESET}", exc_info=True
        )
    return None


def get_open_position(
    exchange: ccxt.Exchange, symbol: str, logger: logging.Logger
) -> dict | None:
    """Checks for an open position, returning enhanced position dict or None."""
    lg = logger
    try:
        lg.debug(f"Fetching positions for symbol: {symbol}")
        positions: list[dict] = []
        market = get_market_info(exchange, symbol, lg)  # Need market for context/params
        if not market:
            raise ValueError("Failed to get market info for position fetch")

        params = {}
        market_id = market["id"]
        if "bybit" in exchange.id.lower():
            category = "linear" if market.get("is_linear", True) else "inverse"
            params = {
                "category": category,
                "symbol": market_id,
            }  # V5 requires symbol for single fetch
            lg.debug(f"Using params for fetch_positions: {params}")

        try:  # Bybit V5: fetch_positions with symbol param fetches that specific symbol
            positions = exchange.fetch_positions(symbols=None, params=params)
        except ccxt.ArgumentsRequired:  # Fallback if exchange needs fetching all
            lg.warning("Fetching all positions (fallback)...")
            all_pos = exchange.fetch_positions()
            positions = [p for p in all_pos if p.get("symbol") == symbol]
        except ccxt.ExchangeError as e:
            # Handle "no position" errors gracefully (e.g., Bybit 110021)
            no_pos_codes = [110021]
            no_pos_msgs = ["position idx not exist", "no position found"]
            err_code = getattr(e, "code", None)
            if err_code in no_pos_codes or any(
                msg in str(e).lower() for msg in no_pos_msgs
            ):
                lg.info(f"No position found for {symbol} (Exchange: {e}).")
                return None
            lg.error(
                f"Exchange error fetching position for {symbol}: {e}", exc_info=True
            )
            return None

        active_position = None
        if not positions:
            lg.info(f"No position entries returned for {symbol}.")
            return None

        # Determine threshold slightly above zero noise based on amount precision
        min_size_threshold = Decimal("1e-9")
        try:
            amount_prec_val = market.get("precision", {}).get("amount")
            if amount_prec_val:
                min_size_threshold = Decimal(str(amount_prec_val)) * Decimal(
                    "0.1"
                )  # 10% of step size
        except:
            pass

        for pos in positions:
            size_str = pos.get("contracts", pos.get("info", {}).get("size"))
            if size_str is None:
                continue
            try:
                pos_size = Decimal(str(size_str))
                if abs(pos_size) > min_size_threshold:
                    active_position = pos
                    break
            except (InvalidOperation, ValueError, TypeError):
                continue

        if not active_position:
            lg.info(f"No active open position found for {symbol}.")
            return None

        # --- Post-Process the Found Active Position ---
        # Determine side (Standard 'side' or Bybit 'info.side')
        pos_side = active_position.get("side")
        if pos_side not in ["long", "short"]:
            info_side = active_position.get("info", {}).get("side", "").lower()
            if info_side == "buy":
                pos_side = "long"
            elif info_side == "sell":
                pos_side = "short"
            else:  # Infer from size as last resort
                try:
                    size_dec = Decimal(str(active_position.get("contracts", "0")))
                except:
                    size_dec = Decimal("0")
                if size_dec > min_size_threshold:
                    pos_side = "long"
                elif size_dec < -min_size_threshold:
                    pos_side = "short"
                else:
                    lg.warning(f"Could not determine side for position {symbol}.")
                    return None
            active_position["side"] = pos_side  # Add inferred side

        # Ensure 'contracts' holds absolute size
        with contextlib.suppress(Exception):
            active_position["contracts"] = abs(
                Decimal(str(active_position.get("contracts", "0")))
            )

        # Enhance with Decimal SL/TP/TSL info
        info = active_position.get("info", {})

        def get_dec(std_key: str, info_key: str) -> Decimal | None:
            v = active_position.get(std_key, info.get(info_key))
            if v is not None and str(v).strip() not in ["", "0", "0.0"]:
                try:
                    d = Decimal(str(v))
                    return d if d > 0 else None
                except:
                    pass
            return None

        active_position["stopLossPriceDecimal"] = get_dec("stopLossPrice", "stopLoss")
        active_position["takeProfitPriceDecimal"] = get_dec(
            "takeProfitPrice", "takeProfit"
        )
        # TSL distance/activation can be '0'

        def get_tsl_dec(key: str) -> Decimal:
            v = info.get(key, "0")
            try:
                d = Decimal(str(v))
                return d if d >= 0 else Decimal("0")
            except:
                return Decimal("0")

        active_position["trailingStopLossDistanceDecimal"] = get_tsl_dec("trailingStop")
        active_position["tslActivationPriceDecimal"] = get_tsl_dec("activePrice")
        active_position["is_tsl_active"] = (
            active_position["trailingStopLossDistanceDecimal"] > 0
        )

        # Log Position Details
        prec = 4
        amt_prec = 4  # Defaults
        try:  # Use analyzer helpers for precision
            analyzer = TradingAnalyzer(
                pd.DataFrame({"close": [1]}), lg, {}, market
            )  # Dummy analyzer
            prec = analyzer.get_price_precision()
            amt_prec_val = market.get("precision", {}).get("amount")
            if amt_prec_val:
                amt_prec = (
                    abs(Decimal(str(amt_prec_val)).normalize().as_tuple().exponent)
                    if not isinstance(amt_prec_val, int)
                    else amt_prec_val
                )
        except:
            pass

        def fmt(v: Any, p: int, zero_ok: bool = False) -> str:
            if v is None:
                return "N/A"
            try:
                d = Decimal(str(v))
                return f"{d:.{p}f}" if d > 0 or (d == 0 and zero_ok) else "N/A"
            except:
                return str(v)

        entry = fmt(active_position.get("entryPrice", info.get("avgPrice")), prec)
        size = fmt(active_position.get("contracts", info.get("size")), amt_prec)
        liq = fmt(active_position.get("liquidationPrice"), prec)
        lev = (
            fmt(active_position.get("leverage", info.get("leverage")), 1) + "x"
            if active_position.get("leverage")
            else "N/A"
        )
        pnl = fmt(active_position.get("unrealizedPnl"), 4)
        sl = fmt(active_position.get("stopLossPriceDecimal"), prec)
        tp = fmt(active_position.get("takeProfitPriceDecimal"), prec)
        tsl_dist = fmt(
            active_position.get("trailingStopLossDistanceDecimal"), prec, True
        )
        tsl_act = fmt(active_position.get("tslActivationPriceDecimal"), prec, True)

        logger.info(
            f"{NEON_GREEN}Active {pos_side.upper()} position found for {symbol}:{RESET} "
            f"Size={size}, Entry={entry}, Liq={liq}, Lev={lev}, PnL={pnl}, "
            f"SL={sl}, TP={tp}, TSL Active: {active_position['is_tsl_active']} (Dist={tsl_dist}/Act={tsl_act})"
        )
        logger.debug(f"Full position details for {symbol}: {active_position}")
        return active_position

    except (
        ccxt.AuthenticationError,
        ccxt.NetworkError,
        ccxt.ExchangeError,
        Exception,
    ) as e:
        lg.error(
            f"{NEON_RED}Error fetching/processing positions for {symbol}: {e}{RESET}",
            exc_info=True,
        )
    return None


def set_leverage_ccxt(
    exchange: ccxt.Exchange,
    symbol: str,
    leverage: int,
    market_info: dict,
    logger: logging.Logger,
) -> bool:
    """Sets leverage using CCXT, handling Bybit V5 specifics and verification."""
    lg = logger
    if not market_info or not market_info.get("is_contract", False):
        lg.info(f"Leverage setting skipped ({symbol}): Not a contract market.")
        return True
    if leverage <= 0:
        lg.warning(
            f"Leverage setting skipped ({symbol}): Invalid leverage ({leverage})."
        )
        return False

    try:
        lg.info(f"Attempting to set leverage for {symbol} to {leverage}x...")
        params = {}
        market_id = market_info.get("id")
        if not market_id:
            lg.error(f"Leverage setting failed ({symbol}): Missing market ID.")
            return False

        if "bybit" in exchange.id.lower():
            category = "linear" if market_info.get("is_linear", True) else "inverse"
            params = {
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
                "category": category,
            }
            lg.debug(f"Using Bybit V5 specific params for set_leverage: {params}")

        response = exchange.set_leverage(
            leverage=leverage, symbol=symbol, params=params
        )
        lg.debug(f"Set leverage raw response for {symbol}: {response}")

        # Verification (Bybit V5 often returns None or dict with retCode)
        verified = False
        if response is None:
            verified = True  # Assume success if no error and None response
        elif isinstance(response, dict):
            ret_code = response.get("retCode", response.get("info", {}).get("retCode"))
            if ret_code == 0:
                verified = True
            elif ret_code == 110045:
                lg.info(f"Leverage already set to {leverage}x (Code {ret_code}).")
                verified = True
            else:
                lg.warning(
                    f"Set leverage returned non-zero retCode {ret_code} ({response.get('retMsg', 'N/A')}). Failed."
                )
        else:
            verified = True  # Assume success if non-dict/None and no error

        if verified:
            lg.info(
                f"{NEON_GREEN}Leverage for {symbol} successfully set/confirmed at {leverage}x.{RESET}"
            )
            return True
        else:
            lg.error(f"Leverage setting failed verification for {symbol}.")
            return False

    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, "code", None)
        lg.error(
            f"{NEON_RED}Exchange error setting leverage for {symbol}: {e} (Code: {bybit_code}){RESET}"
        )
        if bybit_code == 110045:
            lg.info(f"Leverage already set to {leverage}x.")
            return True  # Treat as success
        # Add hints for common errors
        if bybit_code in [110028, 110009]:
            lg.error(f" >> Hint: Check Margin Mode (Isolated/Cross) for {symbol}.")
        elif bybit_code == 110044:
            lg.error(f" >> Hint: Leverage {leverage}x may exceed risk limit tier.")
        elif bybit_code == 110013:
            lg.error(f" >> Hint: Leverage value {leverage}x might be invalid.")
    except (ccxt.NetworkError, Exception) as e:
        lg.error(
            f"{NEON_RED}Error setting leverage for {symbol}: {e}{RESET}", exc_info=True
        )
    return False


def place_trade(
    exchange: ccxt.Exchange,
    symbol: str,
    trade_signal: str,
    position_size: Decimal,
    market_info: dict,
    logger: logging.Logger,
    reduce_only: bool = False,
) -> dict | None:
    """Places a market order using CCXT, handling Bybit V5 specifics."""
    lg = logger
    if not market_info:
        lg.error(f"Trade aborted ({symbol}): Missing market_info.")
        return None

    side = "buy" if trade_signal == "BUY" else "sell"
    order_type = "market"
    is_contract = market_info.get("is_contract", False)
    size_unit = "Contracts" if is_contract else market_info.get("base", "")

    try:
        amount_float = float(abs(position_size))
        assert amount_float > 0
    except:
        lg.error(f"Trade aborted ({symbol}): Invalid size {position_size}.")
        return None

    params = {"positionIdx": 0, "reduceOnly": reduce_only}  # Default to One-Way mode
    market_id = market_info.get("id")
    if not market_id:
        lg.error(f"Trade aborted ({symbol}): Missing market ID.")
        return None
    if "bybit" in exchange.id.lower():
        category = "linear" if market_info.get("is_linear", True) else "inverse"
        params["category"] = category

    action = "Closing" if reduce_only else "Opening"
    lg.info(
        f"Attempting to place {side.upper()} {order_type} order ({action}) for {symbol}:"
    )
    lg.info(f"  Size: {amount_float:.8f} {size_unit} | Params: {params}")

    try:
        order = exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount_float,
            price=None,
            params=params,
        )
        order_id = order.get("id", "N/A")
        status = order.get("status", "N/A")
        lg.info(
            f"{NEON_GREEN}Trade Order Placed ({action})! ID: {order_id}, Status: {status}{RESET}"
        )
        lg.debug(f"Raw order response ({symbol} {side} reduce={reduce_only}): {order}")
        return order

    except ccxt.InsufficientFunds as e:
        lg.error(
            f"{NEON_RED}Insufficient funds ({action} {symbol}): {e}{RESET}"
        )  # Log balance hint
    except ccxt.InvalidOrder as e:
        lg.error(
            f"{NEON_RED}Invalid order ({action} {symbol}): {e}{RESET}"
        )  # Log hints
    except ccxt.NetworkError as e:
        lg.error(
            f"{NEON_RED}Network error placing order ({action} {symbol}): {e}{RESET}"
        )
    except ccxt.ExchangeError as e:
        bybit_code = getattr(e, "code", None)
        lg.error(
            f"{NEON_RED}Exchange error placing order ({action} {symbol}): {e} (Code: {bybit_code}){RESET}"
        )
        if bybit_code == 110025 and reduce_only:  # Position not found on close attempt
            lg.warning(
                f"{NEON_YELLOW} >> Hint (110025): Position not found for closing. Already closed?{RESET}"
            )
            return {
                "id": "N/A",
                "status": "closed",
                "info": {"retMsg": "Position not found on close attempt"},
            }  # Simulate success
        # Add other hints
    except Exception as e:
        lg.error(
            f"{NEON_RED}Unexpected error placing order ({action} {symbol}): {e}{RESET}",
            exc_info=True,
        )
    return None


def _set_position_protection(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,
    logger: logging.Logger,
    stop_loss_price: Decimal | str | None = None,
    take_profit_price: Decimal | str | None = None,
    trailing_stop_distance: Decimal | str | None = None,
    tsl_activation_price: Decimal | str | None = None,
) -> bool:
    """Internal helper to set SL/TP/TSL via Bybit V5 `/v5/position/set-trading-stop`."""
    lg = logger
    if (
        not market_info
        or not position_info
        or not market_info.get("is_contract", False)
    ):
        lg.error(
            f"Cannot set protection ({symbol}): Invalid input (market/pos info or not contract)."
        )
        return False
    pos_side = position_info.get("side")
    market_id = market_info.get("id")
    if pos_side not in ["long", "short"] or not market_id:
        lg.error(
            f"Cannot set protection ({symbol}): Invalid pos side ('{pos_side}') or market ID."
        )
        return False

    position_idx = 0  # Default One-Way
    try:  # Get positionIdx from Bybit V5 info
        idx_val = position_info.get(
            "id", position_info.get("info", {}).get("positionIdx")
        )
        if idx_val is not None and int(str(idx_val)) in [0, 1, 2]:
            position_idx = int(str(idx_val))
    except:
        pass

    def is_valid(val: Decimal | str | None) -> bool:
        return val is not None and (
            (isinstance(val, str) and val == "0")
            or (isinstance(val, Decimal) and val >= 0)
        )

    has_sl = stop_loss_price is not None
    has_tp = take_profit_price is not None
    has_tsl = trailing_stop_distance is not None
    valid_sl = has_sl and is_valid(stop_loss_price)
    valid_tp = has_tp and is_valid(take_profit_price)
    valid_tsl = (
        has_tsl
        and is_valid(trailing_stop_distance)
        and (tsl_activation_price is None or is_valid(tsl_activation_price))
    )

    if not valid_sl and not valid_tp and not valid_tsl:
        lg.info(f"No valid protection parameters provided for {symbol}. No action.")
        return True

    category = "linear" if market_info.get("is_linear", True) else "inverse"
    params = {
        "category": category,
        "symbol": market_id,
        "tpslMode": "Full",
        "slTriggerBy": "LastPrice",
        "tpTriggerBy": "LastPrice",
        "slOrderType": "Market",
        "tpOrderType": "Market",
        "positionIdx": position_idx,
    }
    log_parts = [
        f"Setting protection for {symbol} ({pos_side.upper()}, Idx: {position_idx}):"
    ]
    params_to_send = {}

    try:  # Format parameters using exchange helpers

        def fmt(val: Decimal | str) -> str:
            if isinstance(val, str) and val == "0":
                return "0"
            if isinstance(val, Decimal):
                if val < 0:
                    raise ValueError(f"Negative value {val} invalid")
                if val == 0:
                    return "0"
                try:
                    return exchange.price_to_precision(symbol, float(val))
                except Exception as e:
                    raise ValueError(f"Formatting failed for {val}: {e}")
            raise TypeError(f"Invalid type {type(val)}")

        tsl_distance_fmt = None
        if valid_tsl:
            tsl_distance_fmt = fmt(trailing_stop_distance)
            if tsl_distance_fmt != "0":
                params_to_send["trailingStop"] = tsl_distance_fmt
                act_price = (
                    tsl_activation_price
                    if tsl_activation_price is not None
                    else Decimal("0")
                )
                params_to_send["activePrice"] = fmt(act_price)
                log_parts.append(
                    f"  TSL: Dist={params_to_send['trailingStop']}, Act={params_to_send['activePrice']}"
                )
            else:  # Explicitly cancelling TSL
                params_to_send["trailingStop"] = "0"
                params_to_send["activePrice"] = "0"
                log_parts.append("  TSL: Cancelling")

        # Only set fixed SL if TSL is not being actively set OR is being cancelled
        if valid_sl and (tsl_distance_fmt is None or tsl_distance_fmt == "0"):
            params_to_send["stopLoss"] = fmt(stop_loss_price)
            log_parts.append(f"  Fixed SL: {params_to_send['stopLoss']}")
        elif has_sl and tsl_distance_fmt != "0":
            lg.warning(f"Ignoring Fixed SL for {symbol} as active TSL is being set.")

        if valid_tp:
            params_to_send["takeProfit"] = fmt(take_profit_price)
            log_parts.append(f"  Fixed TP: {params_to_send['takeProfit']}")

    except (ValueError, TypeError, InvalidOperation) as fmt_err:
        lg.error(
            f"{NEON_RED}Error formatting protection params for {symbol}: {fmt_err}. ABORTED.{RESET}"
        )
        return False
    except Exception as e:
        lg.error(f"Unexpected error preparing protection params: {e}", exc_info=True)
        return False

    if not params_to_send:
        lg.info(f"No protection parameters to set/modify for {symbol}. No API call.")
        return True

    params.update(params_to_send)
    lg.info("\n".join(log_parts))
    lg.debug(f"  API Call Params: {params}")

    try:  # Make the API Call
        method_name = "private_post_v5_position_trading_stop"
        if not hasattr(exchange, method_name):
            raise NotImplementedError(f"Exchange missing '{method_name}'. Update CCXT?")
        response = getattr(exchange, method_name)(params)
        lg.debug(f"Set protection raw response for {symbol}: {response}")

        ret_code = response.get("retCode")
        ret_msg = response.get("retMsg", "N/A")
        ret_ext = response.get("retExtInfo", {})
        if ret_code == 0:
            no_change = any(
                msg in ret_msg.lower() for msg in ["not modified", "same tpsl"]
            )
            if no_change:
                lg.info(
                    f"{NEON_YELLOW}Protection already set for {symbol} (Exchange: {ret_msg}).{RESET}"
                )
            else:
                lg.info(
                    f"{NEON_GREEN}Position protection set/updated successfully for {symbol}.{RESET}"
                )
            return True
        else:
            lg.error(
                f"{NEON_RED}Failed to set protection for {symbol}: {ret_msg} (Code: {ret_code}) Ext: {ret_ext}{RESET}"
            )
            # Add hints based on codes
            if ret_code == 110043:
                lg.error(" >> Hint: Check trigger prices, tpslMode.")
            elif ret_code == 110025:
                lg.error(" >> Hint: Position not found/zero size?")
            elif ret_code == 110013:
                lg.error(
                    " >> Hint: Parameter error (invalid SL/TP/TSL value? tick size? wrong side?)."
                )
            elif ret_code == 110036:
                lg.error(
                    f" >> Hint: TSL Activation price ({params.get('activePrice')}) invalid?"
                )
            return False
    except (
        NotImplementedError,
        ccxt.NetworkError,
        ccxt.ExchangeError,
        ccxt.BaseError,
        KeyError,
        Exception,
    ) as e:
        lg.error(
            f"{NEON_RED}Error during protection API call for {symbol}: {e}{RESET}",
            exc_info=True,
        )
    return False


def set_trailing_stop_loss(
    exchange: ccxt.Exchange,
    symbol: str,
    market_info: dict,
    position_info: dict,
    config: dict[str, Any],
    logger: logging.Logger,
    take_profit_price: Decimal | None = None,
) -> bool:
    """Calculates and sets TSL (and optionally TP), cancelling fixed SL."""
    lg = logger
    if not config.get("enable_trailing_stop", False):
        lg.info(f"TSL disabled in config for {symbol}. Skipping.")
        return False  # Not set, but not an error
    if not market_info or not position_info:
        lg.error(f"Cannot set TSL ({symbol}): Missing market/position info.")
        return False

    try:  # Get config params
        callback_rate = Decimal(str(config["trailing_stop_callback_rate"]))
        assert callback_rate > 0
        activation_perc = Decimal(str(config["trailing_stop_activation_percentage"]))
        assert activation_perc >= 0
    except (KeyError, InvalidOperation, ValueError, TypeError, AssertionError) as e:
        lg.error(f"Invalid TSL config for {symbol}: {e}. Cannot set TSL.")
        return False

    try:  # Get position details
        entry_str = position_info.get(
            "entryPrice", position_info.get("info", {}).get("avgPrice")
        )
        pos_side = position_info.get("side")
        if entry_str is None or pos_side not in ["long", "short"]:
            raise ValueError("Missing entryPrice/side")
        entry_price = Decimal(str(entry_str))
        assert entry_price > 0
    except (TypeError, ValueError, KeyError, InvalidOperation, AssertionError) as e:
        lg.error(
            f"Error parsing position info for TSL ({symbol}): {e}. Pos: {position_info}"
        )
        return False

    try:  # Calculate TSL params for Bybit API
        analyzer = TradingAnalyzer(
            pd.DataFrame({"close": [1]}), lg, config, market_info
        )  # Dummy analyzer for helpers
        prec = analyzer.get_price_precision()
        min_tick = analyzer.get_min_tick_size()
        if min_tick <= 0:
            raise ValueError(f"Invalid min tick size ({min_tick})")

        # 1. Activation Price
        activation_price = Decimal("0")  # Default immediate
        if activation_perc > 0:
            offset = entry_price * activation_perc
            rounding = ROUND_UP if pos_side == "long" else ROUND_DOWN
            raw_act = (
                entry_price + offset if pos_side == "long" else entry_price - offset
            )
            activation_price = (raw_act / min_tick).quantize(
                Decimal("1"), rounding=rounding
            ) * min_tick
            # Ensure strictly away from entry
            if (pos_side == "long" and activation_price <= entry_price) or (
                pos_side == "short" and activation_price >= entry_price
            ):
                activation_price = (
                    entry_price + (min_tick if pos_side == "long" else -min_tick)
                ).quantize(min_tick, rounding=rounding)
            if activation_price <= 0:
                lg.warning(
                    f"TSL activation price invalid ({activation_price}). Defaulting to immediate."
                )
                activation_price = Decimal("0")
        else:
            lg.info(f"TSL immediate activation for {symbol}.")

        # 2. Trailing Distance (price points)
        dist_raw = entry_price * callback_rate
        # Round distance UP to nearest tick, min 1 tick
        trail_dist = max(
            min_tick,
            (dist_raw / min_tick).quantize(Decimal("1"), rounding=ROUND_UP) * min_tick,
        )
        if trail_dist <= 0:
            raise ValueError(f"TSL distance invalid ({trail_dist})")

        act_log = (
            f"{activation_price:.{prec}f}" if activation_price > 0 else "0 (Immediate)"
        )
        tp_log = f"{take_profit_price:.{prec}f}" if take_profit_price else "None"
        lg.info(
            f"Calculated TSL Params ({symbol} {pos_side.upper()}): Activation={act_log}, Distance={trail_dist:.{prec}f}, TP={tp_log}"
        )

        # Call helper to set TSL & TP, cancelling fixed SL
        return _set_position_protection(
            exchange,
            symbol,
            market_info,
            position_info,
            lg,
            stop_loss_price="0",  # <<< Cancel fixed SL
            take_profit_price=take_profit_price
            if isinstance(take_profit_price, Decimal) and take_profit_price > 0
            else None,
            trailing_stop_distance=trail_dist,
            tsl_activation_price=activation_price,
        )
    except (ValueError, InvalidOperation, TypeError, Exception) as e:
        lg.error(
            f"{NEON_RED}Error calculating/setting TSL for {symbol}: {e}{RESET}",
            exc_info=True,
        )
        return False


# --- Main Analysis and Trading Loop ---
def analyze_and_trade_symbol(
    exchange: ccxt.Exchange, symbol: str, config: dict[str, Any], logger: logging.Logger
) -> None:
    """Analyzes symbol, executes/manages trades based on signals and config."""
    lg = logger
    lg.info(f"---== Analyzing {symbol} ({config['interval']}) Cycle Start ==---")
    cycle_start_time = time.monotonic()

    # --- 1. Fetch Market Info & Data ---
    market_info = get_market_info(exchange, symbol, lg)
    if not market_info:
        lg.error(f"Failed to get market info for {symbol}. Skipping cycle.")
        return
    ccxt_interval = CCXT_INTERVAL_MAP.get(config["interval"])
    if not ccxt_interval:
        lg.error(f"Invalid interval '{config['interval']}'. Skipping cycle.")
        return

    kline_limit = 500  # Ensure enough history
    klines_df = fetch_klines_ccxt(
        exchange, symbol, ccxt_interval, limit=kline_limit, logger=lg
    )
    if klines_df.empty or len(klines_df) < 50:
        lg.error(f"Failed to fetch sufficient kline data for {symbol}. Skipping cycle.")
        return

    current_price = fetch_current_price_ccxt(exchange, symbol, lg)
    if current_price is None:
        lg.warning(f"Using last close price for {symbol} due to ticker fetch failure.")
        try:
            last_close = Decimal(str(klines_df["close"].iloc[-1]))
            assert last_close > 0
            current_price = last_close
        except:
            lg.error(
                f"Failed to get valid last close price for {symbol}. Cannot proceed."
            )
            return

    orderbook_data = None
    active_weights = config.get("weight_sets", {}).get(
        config.get("active_weight_set", "default"), {}
    )
    if (
        config.get("indicators", {}).get("orderbook", False)
        and Decimal(str(active_weights.get("orderbook", 0))) != 0
    ):
        orderbook_data = fetch_orderbook_ccxt(
            exchange, symbol, config.get("orderbook_limit", 25), lg
        )

    # --- 2. Analyze Data & Generate Signal ---
    try:
        analyzer = TradingAnalyzer(klines_df.copy(), lg, config, market_info)
    except (ValueError, Exception) as e:
        lg.error(f"Failed to init TradingAnalyzer for {symbol}: {e}. Skipping cycle.")
        return
    if not analyzer.indicator_values:
        lg.error(f"Indicator calculation failed for {symbol}. Skipping cycle.")
        return

    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    _, tp_potential, sl_potential = analyzer.calculate_entry_tp_sl(
        current_price, signal
    )
    prec = analyzer.get_price_precision()
    min_tick = analyzer.get_min_tick_size()
    atr_float = analyzer.indicator_values.get("ATR", float("nan"))

    # --- 3. Log Analysis Summary ---
    atr_log = f"{atr_float:.{prec + 1}f}" if not math.isnan(atr_float) else "N/A"
    sl_pot_log = f"{sl_potential:.{prec}f}" if sl_potential else "N/A"
    tp_pot_log = f"{tp_potential:.{prec}f}" if tp_potential else "N/A"
    lg.info(
        f"Current ATR: {atr_log} | Potential Initial SL/TP: {sl_pot_log} / {tp_pot_log}"
    )
    tsl_on = config.get("enable_trailing_stop")
    be_on = config.get("enable_break_even")
    ma_exit_on = config.get("enable_ma_cross_exit")
    lg.info(
        f"Configured Protections: TSL={'On' if tsl_on else 'Off'} | BE={'On' if be_on else 'Off'} | MA Exit={'On' if ma_exit_on else 'Off'}"
    )

    # --- 4. Check Position & Execute/Manage ---
    if not config.get("enable_trading", False):
        lg.info(
            f"{NEON_YELLOW}Trading disabled. Analysis complete, no actions taken.{RESET}"
        )
    else:
        open_position = get_open_position(exchange, symbol, lg)

        # --- Scenario 1: No Open Position ---
        if open_position is None:
            if signal in ["BUY", "SELL"]:
                lg.info(
                    f"*** {signal} Signal & No Position: Initiating Trade Sequence for {symbol} ***"
                )
                balance = fetch_balance(exchange, QUOTE_CURRENCY, lg)
                if balance is None or balance <= 0:
                    lg.error(f"Trade Aborted ({signal}): Invalid balance.")
                    return
                if sl_potential is None:
                    lg.error(
                        f"Trade Aborted ({signal}): Cannot size trade, SL calculation failed."
                    )
                    return

                if market_info.get("is_contract", False):
                    leverage = int(config.get("leverage", 1))
                    if leverage > 0 and not set_leverage_ccxt(
                        exchange, symbol, leverage, market_info, lg
                    ):
                        lg.error(
                            f"Trade Aborted ({signal}): Failed to set/confirm leverage."
                        )
                        return
                position_size = calculate_position_size(
                    balance,
                    config["risk_per_trade"],
                    sl_potential,
                    current_price,
                    market_info,
                    exchange,
                    lg,
                )
                if position_size is None or position_size <= 0:
                    lg.error(
                        f"Trade Aborted ({signal}): Invalid position size calculated."
                    )
                    return

                lg.info(
                    f"==> Placing {signal} market order | Size: {position_size} <=="
                )
                trade_order = place_trade(
                    exchange,
                    symbol,
                    signal,
                    position_size,
                    market_info,
                    lg,
                    reduce_only=False,
                )

                if trade_order and trade_order.get("id"):
                    lg.info(
                        f"Order {trade_order['id']} placed. Waiting {POSITION_CONFIRM_DELAY}s for confirmation..."
                    )
                    time.sleep(POSITION_CONFIRM_DELAY)
                    confirmed_position = get_open_position(exchange, symbol, lg)

                    if confirmed_position:
                        try:
                            entry_actual_str = confirmed_position.get(
                                "entryPrice",
                                confirmed_position.get("info", {}).get("avgPrice"),
                            )
                            entry_actual = (
                                Decimal(str(entry_actual_str))
                                if entry_actual_str
                                else None
                            )
                            if not entry_actual or entry_actual <= 0:
                                raise ValueError("Invalid confirmed entry price")
                            lg.info(
                                f"{NEON_GREEN}Position Confirmed! Actual Entry: ~{entry_actual:.{prec}f}{RESET}"
                            )

                            _, tp_actual, sl_actual = analyzer.calculate_entry_tp_sl(
                                entry_actual, signal
                            )
                            protection_set = False
                            if config.get("enable_trailing_stop", False):
                                lg.info(
                                    f"Setting Trailing Stop Loss (TP target: {tp_actual})..."
                                )
                                protection_set = set_trailing_stop_loss(
                                    exchange,
                                    symbol,
                                    market_info,
                                    confirmed_position,
                                    config,
                                    lg,
                                    tp_actual,
                                )
                            else:  # Fixed SL/TP
                                lg.info(
                                    f"Setting Fixed SL ({sl_actual}) and TP ({tp_actual})..."
                                )
                                if sl_actual or tp_actual:
                                    protection_set = _set_position_protection(
                                        exchange,
                                        symbol,
                                        market_info,
                                        confirmed_position,
                                        lg,
                                        sl_actual,
                                        tp_actual,
                                        "0",
                                        "0",
                                    )
                                else:
                                    lg.warning(
                                        "Fixed SL/TP calculation failed. No fixed protection set."
                                    )
                                    protection_set = True

                            if protection_set:
                                lg.info(
                                    f"{NEON_GREEN}=== TRADE ENTRY & PROTECTION SETUP COMPLETE ({signal}) ==="
                                )
                            else:
                                lg.error(
                                    f"{NEON_RED}=== TRADE placed BUT FAILED TO SET/CONFIRM PROTECTION ({signal}) ==="
                                )
                        except (
                            ValueError,
                            TypeError,
                            InvalidOperation,
                            KeyError,
                        ) as post_err:
                            lg.error(
                                f"Error during post-trade processing for {symbol}: {post_err}. Manual check needed!",
                                exc_info=True,
                            )
                    else:  # Position not confirmed
                        lg.error(
                            f"Trade order {trade_order['id']} placed, but FAILED TO CONFIRM open position! Manual investigation required!"
                        )
                else:
                    lg.error(f"=== TRADE EXECUTION FAILED ({signal}). See logs. ===")
            else:
                lg.info("Signal is HOLD and no open position. No action.")

        # --- Scenario 2: Existing Open Position ---
        else:  # open_position is not None
            pos_side = open_position.get("side", "unknown")
            current_sl_dec = open_position.get("stopLossPriceDecimal")
            current_tp_dec = open_position.get("takeProfitPriceDecimal")
            is_tsl_active = open_position.get("is_tsl_active", False)

            # --- Check Exit Conditions ---
            exit_signal = (pos_side == "long" and signal == "SELL") or (
                pos_side == "short" and signal == "BUY"
            )
            ma_cross_exit = False
            if not exit_signal and config.get("enable_ma_cross_exit", False):
                ema_short = analyzer.indicator_values.get("EMA_Short")
                ema_long = analyzer.indicator_values.get("EMA_Long")
                if not math.isnan(ema_short) and not math.isnan(ema_long):
                    if (pos_side == "long" and ema_short < ema_long) or (
                        pos_side == "short" and ema_short > ema_long
                    ):
                        ma_cross_exit = True
                        lg.warning(
                            f"{NEON_YELLOW}*** MA CROSS EXIT Triggered for {pos_side.upper()} position. ***{RESET}"
                        )

            # --- Execute Position Close ---
            if exit_signal or ma_cross_exit:
                reason = (
                    f"Opposing Signal ({signal})" if exit_signal else "MA Cross Exit"
                )
                lg.warning(
                    f"*** EXITING {pos_side.upper()} position due to: {reason}. ***"
                )
                try:
                    close_signal = "SELL" if pos_side == "long" else "BUY"
                    size_str = open_position.get(
                        "contracts", open_position.get("info", {}).get("size")
                    )
                    size_to_close = abs(Decimal(str(size_str))) if size_str else None
                    if not size_to_close or size_to_close <= 0:
                        raise ValueError("Invalid size to close")

                    close_order = place_trade(
                        exchange,
                        symbol,
                        close_signal,
                        size_to_close,
                        market_info,
                        lg,
                        reduce_only=True,
                    )
                    if close_order:
                        if (
                            close_order.get("info", {}).get("retMsg")
                            == "Position not found on close attempt"
                        ):
                            lg.info(
                                f"{NEON_GREEN}Position for {symbol} confirmed already closed.{RESET}"
                            )
                        else:
                            lg.info(
                                f"Closing order {close_order.get('id', 'N/A')} placed. Waiting {POSITION_CONFIRM_DELAY}s..."
                            )
                            time.sleep(POSITION_CONFIRM_DELAY)
                            final_pos = get_open_position(exchange, symbol, lg)
                            if final_pos is None:
                                lg.info(
                                    f"{NEON_GREEN}=== POSITION successfully closed. ==="
                                )
                            else:
                                lg.error(
                                    f"{NEON_RED}*** POSITION CLOSE FAILED verification. Position still detected: {final_pos}{RESET}"
                                )
                    else:
                        lg.error(
                            f"{NEON_RED}Failed to place closing order. Manual intervention required.{RESET}"
                        )
                except (
                    ValueError,
                    InvalidOperation,
                    TypeError,
                    Exception,
                ) as close_err:
                    lg.error(
                        f"Error closing position: {close_err}. Manual intervention required!",
                        exc_info=True,
                    )

            # --- Check Break-Even (Only if NOT Exiting) ---
            elif (
                config.get("enable_break_even", False)
                and not analyzer.break_even_triggered
                and not is_tsl_active
            ):  # Don't move SL if TSL is active
                try:
                    entry_dec = Decimal(str(open_position.get("entryPrice", "0")))
                    atr_dec = (
                        Decimal(str(atr_float)) if not math.isnan(atr_float) else None
                    )
                    if entry_dec > 0 and atr_dec and atr_dec > 0 and min_tick > 0:
                        trigger_mult = Decimal(
                            str(config["break_even_trigger_atr_multiple"])
                        )
                        offset_ticks = int(config["break_even_offset_ticks"])
                        profit_target = atr_dec * trigger_mult
                        be_stop_price = None
                        trigger_met = False
                        trigger_price = None

                        if pos_side == "long":
                            trigger_price = entry_dec + profit_target
                            if current_price >= trigger_price:
                                trigger_met = True
                                be_stop_raw = entry_dec + (min_tick * offset_ticks)
                                be_stop_price = max(
                                    entry_dec + min_tick,
                                    (be_stop_raw / min_tick).quantize(
                                        Decimal("1"), rounding=ROUND_UP
                                    )
                                    * min_tick,
                                )
                        elif pos_side == "short":
                            trigger_price = entry_dec - profit_target
                            if current_price <= trigger_price:
                                trigger_met = True
                                be_stop_raw = entry_dec - (min_tick * offset_ticks)
                                be_stop_price = min(
                                    entry_dec - min_tick,
                                    (be_stop_raw / min_tick).quantize(
                                        Decimal("1"), rounding=ROUND_DOWN
                                    )
                                    * min_tick,
                                )

                        if trigger_met and be_stop_price and be_stop_price > 0:
                            sl_log = (
                                f"{current_sl_dec:.{prec}f}"
                                if current_sl_dec
                                else "N/A"
                            )
                            lg.warning(
                                f"{NEON_PURPLE}*** BREAK-EVEN Triggered for {pos_side.upper()}! ***"
                            )
                            lg.info(
                                f"  Current: {current_price:.{prec}f}, Trigger: {trigger_price:.{prec}f}, Target BE SL: {be_stop_price:.{prec}f}"
                            )
                            modify_needed = True
                            if current_sl_dec:  # Check if current SL is already better
                                if (
                                    pos_side == "long"
                                    and current_sl_dec >= be_stop_price
                                ) or (
                                    pos_side == "short"
                                    and current_sl_dec <= be_stop_price
                                ):
                                    lg.info(
                                        f"  Current SL ({sl_log}) already at/beyond BE target. No change."
                                    )
                                    modify_needed = False
                                    analyzer.break_even_triggered = True

                            if modify_needed:
                                lg.info(
                                    f"  Modifying SL to {be_stop_price:.{prec}f}, keeping TP {fmt(current_tp_dec, prec)}."
                                )
                                be_success = _set_position_protection(
                                    exchange,
                                    symbol,
                                    market_info,
                                    open_position,
                                    lg,
                                    be_stop_price,
                                    current_tp_dec,
                                    "0",
                                    "0",
                                )
                                if be_success:
                                    lg.info(f"{NEON_GREEN}Break-even SL set.{RESET}")
                                    analyzer.break_even_triggered = True
                                else:
                                    lg.error(
                                        f"{NEON_RED}Failed to set break-even SL.{RESET}"
                                    )
                        elif trigger_met:
                            lg.error(
                                f"BE triggered, but calculated BE stop ({be_stop_price}) invalid."
                            )
                except (InvalidOperation, ValueError, TypeError, KeyError) as be_err:
                    lg.error(f"Error in break-even logic: {be_err}", exc_info=False)

            # --- Log HOLD if no other action ---
            elif not exit_signal and not ma_cross_exit:
                lg.info(
                    f"Signal is {signal}. Holding existing {pos_side.upper()} position. No management action."
                )

    # --- End of Trading Logic ---
    cycle_end_time = time.monotonic()
    lg.info(
        f"---== Analysis Cycle End ({cycle_end_time - cycle_start_time:.2f}s) ==---"
    )


# --- Main Function ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Bybit Trading Bot")
    parser.add_argument(
        "symbol", help="Trading symbol (e.g., BTC/USDT:USDT or ETH/USDT)"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable DEBUG console logging"
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()  # Ensure uppercase symbol

    console_log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logger(symbol, console_log_level)  # Initialize logger first

    logger.info(
        f"---=== {NEON_GREEN}Whale 2.0 Enhanced Trading Bot Initializing{RESET} ===---"
    )
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Config File: {CONFIG_FILE}")
    logger.info(f"Log Directory: {LOG_DIRECTORY}")
    logger.info(f"Timezone: {TIMEZONE.key}")
    logger.info(
        f"Trading Enabled: {NEON_RED if CONFIG.get('enable_trading') else NEON_YELLOW}{CONFIG.get('enable_trading', False)}{RESET}"
    )
    logger.info(
        f"Sandbox Mode: {NEON_YELLOW if CONFIG.get('use_sandbox') else NEON_RED}{CONFIG.get('use_sandbox', True)}{RESET}"
    )
    logger.info(f"Quote Currency: {QUOTE_CURRENCY}")
    logger.info(f"Risk Per Trade: {CONFIG.get('risk_per_trade', 0.01):.2%}")
    logger.info(f"Leverage: {CONFIG.get('leverage', 1)}x")
    logger.info(f"Interval: {CONFIG.get('interval')}")

    if CONFIG.get("interval") not in VALID_INTERVALS:
        logger.critical(
            f"Invalid 'interval' ({CONFIG.get('interval')}) in config. Must be one of {VALID_INTERVALS}. Exiting."
        )
        return

    exchange = initialize_exchange(CONFIG, logger)
    if not exchange:
        logger.critical("Failed to initialize exchange. Bot cannot start.")
        return

    market_info = get_market_info(exchange, symbol, logger)
    if not market_info:
        logger.critical(
            f"Symbol {symbol} not found or invalid on {exchange.id}. Exiting."
        )
        return
    try:  # Log key details from market_info
        tick_size = market_info.get("info", {}).get("tickSize", "N/A")
        min_amt = market_info.get("limits", {}).get("amount", {}).get("min", "N/A")
        min_cost = market_info.get("limits", {}).get("cost", {}).get("min", "N/A")
        logger.info(
            f"Symbol Details: Contract={market_info.get('is_contract')}, TickSize={tick_size}, MinAmount={min_amt}, MinCost={min_cost}"
        )
    except Exception as e:
        logger.warning(f"Could not log detailed market info: {e}")

    # --- Bot Main Loop ---
    logger.info(
        f"{NEON_GREEN}Initialization complete. Starting main trading loop for {symbol}...{RESET}"
    )
    loop_count = 0
    while True:
        loop_start_utc = datetime.now(ZoneInfo("UTC"))
        loop_start_local = loop_start_utc.astimezone(TIMEZONE)
        loop_count += 1
        logger.debug(
            f"--- Loop Cycle {loop_count} starting at {loop_start_local.strftime('%Y-%m-%d %H:%M:%S %Z')} ---"
        )

        try:
            analyze_and_trade_symbol(exchange, symbol, CONFIG, logger)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            break
        except ccxt.AuthenticationError as e:
            logger.critical(f"CRITICAL: Auth Error: {e}. Bot stopped.")
            break
        except ccxt.NetworkError as e:
            logger.error(f"Network Error: {e}. Retrying after delay...")
            time.sleep(RETRY_DELAY_SECONDS * 6)
        except ccxt.RateLimitExceeded as e:
            wait = _handle_rate_limit(e, logger)
            logger.warning(f"Rate Limit. Waiting {wait}s...")
            time.sleep(wait)
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange Not Available: {e}. Waiting longer...")
            time.sleep(config.get("loop_delay", LOOP_DELAY_SECONDS) * 10)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange Error: {e} (Code: {getattr(e, 'code', 'N/A')})")
            if "maintenance" in str(e).lower():
                logger.warning("Maintenance? Waiting longer...")
                time.sleep(config.get("loop_delay", LOOP_DELAY_SECONDS) * 20)
            else:
                time.sleep(
                    RETRY_DELAY_SECONDS * 3
                )  # Moderate delay for other exchange errors
        except Exception as e:
            logger.error(f"UNEXPECTED CRITICAL ERROR: {e}", exc_info=True)
            logger.critical("Bot stopping due to unexpected error.")
            break

        try:
            loop_delay = max(1, int(config.get("loop_delay", LOOP_DELAY_SECONDS)))
        except:
            loop_delay = LOOP_DELAY_SECONDS
        logger.debug(f"Loop cycle finished. Sleeping for {loop_delay} seconds...")
        time.sleep(loop_delay)

    logger.info(f"---=== {NEON_RED}Trading Bot for {symbol} has stopped.{RESET} ===---")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:  # Attempt to log if possible
            root_logger = logging.getLogger()
            if root_logger and root_logger.hasHandlers():
                root_logger.critical(
                    "Unhandled exception caused script termination.", exc_info=True
                )
            else:
                import traceback

                traceback.print_exc()
        except Exception:
            pass
    finally:
        pass
