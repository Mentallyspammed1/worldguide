# ██████╗ ██╗   ██╗███████╗███╗   ███╗███████╗████████╗██╗   ██╗██╗   ██╗███████╗
# ██╔══██╗╚██╗ ██╔╝██╔════╝████╗ ████║██╔════╝╚══██╔══╝██║   ██║██║   ██║██╔════╝
# ██████╔╝ ╚████╔╝ ███████╗██╔████╔██║███████╗   ██║   ██║   ██║██║   ██║███████╗
# ██╔═══╝   ╚██╔╝  ╚════██║██║╚██╔╝██║╚════██║   ██║   ██║   ██║██║   ██║╚════██║
# ██║        ██║   ███████║██║ ╚═╝ ██║███████║   ██║   ╚██████╔╝╚██████╔╝███████║
# ╚═╝        ╚═╝   ╚══════╝╚═╝     ╚═╝╚══════╝   ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝
# Pyrmethus - Enhanced Termux Trading Spell (v4.0.0 - Merged & Refined)
# Merges concepts from previous versions, focusing on robustness, precision (Decimal),
# advanced position management (SL/TP/TSL/BE), CCXT, pandas_ta, and Termux usage.

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from decimal import ROUND_DOWN, ROUND_UP, Decimal, InvalidOperation, getcontext
from logging.handlers import RotatingFileHandler
from typing import Any
from zoneinfo import ZoneInfo  # Requires Python 3.9+

# --- Attempt Imports ---
try:
    import ccxt

    # Cryptography might be needed by some ccxt dependencies
    import cryptography
    import numpy as np
    import pandas as pd
    import pandas_ta as ta
    import requests  # Used by ccxt
    from colorama import Back, Fore, Style, init
    from dotenv import load_dotenv
    from requests.adapters import HTTPAdapter
    from tabulate import tabulate
    from urllib3.util.retry import Retry
except ImportError as e:
    init(autoreset=True)
    missing_pkg = e.name
    sys.exit(1)

# --- Core Setup ---
init(autoreset=True)
getcontext().prec = 30  # Set Decimal precision
load_dotenv()

# --- Logging Setup ---
TRADE_LEVEL_NUM = logging.INFO + 5
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")


def trade(self, message, *args, **kws) -> None:
    if self.isEnabledFor(TRADE_LEVEL_NUM): self._log(TRADE_LEVEL_NUM, message, args, **kws)


logging.Logger.trade = trade


class SensitiveFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        if api_key: msg = msg.replace(api_key, "***API_KEY***")
        if api_secret: msg = msg.replace(api_secret, "***API_SECRET***")
        return msg


log_formatter = SensitiveFormatter(
    Fore.CYAN + "%(asctime)s " + Style.BRIGHT + "[%(levelname)-8s] " +
    Fore.WHITE + "(%(filename)s:%(lineno)d) " + Style.RESET_ALL + Fore.WHITE + "%(message)s"
)
logger = logging.getLogger("PyrmethusBotV4")
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

LOG_DIRECTORY = "bot_logs"
os.makedirs(LOG_DIRECTORY, exist_ok=True)
log_file_path = os.path.join(LOG_DIRECTORY, "pyrmethus_bot_v4.log")
file_handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
logger.propagate = False


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and stores trading configuration."""
    # Define defaults for all expected parameters
    DEFAULT_CONFIG = {
        "SYMBOL": "BTC/USDT:USDT", "MARKET_TYPE": "linear", "INTERVAL": "5",
        "USE_TESTNET": True, "ENABLE_TRADING": False, "LOG_LEVEL": "INFO",
        "RISK_PERCENTAGE": "0.01", "LEVERAGE": "10",
        "SL_ATR_MULTIPLIER": "1.5", "TP_ATR_MULTIPLIER": "2.0",
        "ENABLE_TRAILING_STOP": True, "TSL_ACTIVATION_ATR_MULTIPLIER": "1.0",
        "TRAILING_STOP_DISTANCE_ATR_MULTIPLIER": "1.2",
        "TRAILING_STOP_CALLBACK_RATE": None,  # Default to None, prioritize ATR method
        "ENABLE_BREAK_EVEN": True, "BREAK_EVEN_TRIGGER_ATR_MULTIPLE": "0.8",
        "BREAK_EVEN_OFFSET_TICKS": "3",
        "MAX_CONCURRENT_POSITIONS": "1", "SL_TRIGGER_BY": "LastPrice",
        "TP_TRIGGER_BY": "LastPrice", "TIME_BASED_EXIT_MINUTES": None,
        "ACTIVE_WEIGHT_SET": "default", "OHLCV_LIMIT": "500",
        "INDICATOR_ATR_PERIOD": "14", "INDICATOR_EMA_SHORT": "9",
        "INDICATOR_EMA_LONG": "21", "INDICATOR_TREND_EMA": "50",
        "INDICATOR_STOCH_RSI_WINDOW": "14", "INDICATOR_STOCH_RSI_K": "3",
        "INDICATOR_STOCH_RSI_D": "3", "INDICATOR_RSI_WINDOW": "14",
        "INDICATOR_CCI_WINDOW": "20", "TRADE_ONLY_WITH_TREND": True,
        "LOOP_SLEEP_SECONDS": "15", "ORDER_CHECK_DELAY_SECONDS": "3",
        "ORDER_CHECK_TIMEOUT_SECONDS": "15", "MAX_FETCH_RETRIES": "3",
        "QUOTE_CURRENCY": "USDT"  # Auto-derived later
    }
    # Type and validation rules (type, min_val_or_list, max_val)
    TYPE_VALIDATIONS = {
        "SYMBOL": str, "MARKET_TYPE": (str, ['linear', 'inverse']), "INTERVAL": (str, ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]),
        "USE_TESTNET": bool, "ENABLE_TRADING": bool, "LOG_LEVEL": (str, list(logging._nameToLevel.keys())),
        "RISK_PERCENTAGE": (Decimal, Decimal("0.0001"), Decimal("0.5")), "LEVERAGE": (int, 1, 200),
        "SL_ATR_MULTIPLIER": (Decimal, Decimal("0.1"), Decimal("10.0")), "TP_ATR_MULTIPLIER": (Decimal, Decimal("0.1"), Decimal("20.0")),
        "ENABLE_TRAILING_STOP": bool, "TSL_ACTIVATION_ATR_MULTIPLIER": (Decimal, Decimal("0.0"), Decimal("10.0")),
        "TRAILING_STOP_DISTANCE_ATR_MULTIPLIER": (Decimal, Decimal("0.1"), Decimal("10.0")),
        "TRAILING_STOP_CALLBACK_RATE": (Decimal, Decimal("0.0001"), Decimal("0.2")),  # Optional
        "ENABLE_BREAK_EVEN": bool, "BREAK_EVEN_TRIGGER_ATR_MULTIPLE": (Decimal, Decimal("0.0"), Decimal("10.0")),
        "BREAK_EVEN_OFFSET_TICKS": (int, 0, 100),
        "MAX_CONCURRENT_POSITIONS": (int, 1, 10),
        "SL_TRIGGER_BY": (str, ["LastPrice", "MarkPrice", "IndexPrice"]),
        "TP_TRIGGER_BY": (str, ["LastPrice", "MarkPrice", "IndexPrice"]),
        "TIME_BASED_EXIT_MINUTES": (int, 1, 10080),  # Optional
        "ACTIVE_WEIGHT_SET": str, "OHLCV_LIMIT": (int, 50, 1000),
        "INDICATOR_ATR_PERIOD": (int, 2, 500), "INDICATOR_EMA_SHORT": (int, 2, 500),
        "INDICATOR_EMA_LONG": (int, 3, 1000), "INDICATOR_TREND_EMA": (int, 5, 1000),
        "INDICATOR_STOCH_RSI_WINDOW": (int, 5, 500), "INDICATOR_STOCH_RSI_K": (int, 1, 500),
        "INDICATOR_STOCH_RSI_D": (int, 1, 500), "INDICATOR_RSI_WINDOW": (int, 2, 500),
        "INDICATOR_CCI_WINDOW": (int, 2, 500), "TRADE_ONLY_WITH_TREND": bool,
        "LOOP_SLEEP_SECONDS": (int, 5, 600), "ORDER_CHECK_DELAY_SECONDS": (int, 1, 30),
        "ORDER_CHECK_TIMEOUT_SECONDS": (int, 5, 120), "MAX_FETCH_RETRIES": (int, 1, 10),
        "QUOTE_CURRENCY": str
    }
    # Mapping for interval conversion
    CCXT_INTERVAL_MAP = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
                         "60": "1h", "120": "2h", "240": "4h", "D": "1d", "W": "1w", "M": "1M"}

    def __init__(self) -> None:
        logger.info("Loading and validating configuration from environment...")
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        if not self.api_key or not self.api_secret:
            logger.critical("BYBIT_API_KEY or BYBIT_API_SECRET not found! Halting.")
            sys.exit(1)

        for key, default_val in self.DEFAULT_CONFIG.items():
            setattr(self, key.lower(), self._get_and_validate_env(key, default_val))

        # Post-load processing and validation
        self._process_symbol_and_quote()
        self._validate_dependent_params()
        self.ccxt_interval = self.CCXT_INTERVAL_MAP.get(self.interval)
        if not self.ccxt_interval:
            logger.critical(f"Invalid INTERVAL '{self.interval}'. Valid: {list(self.CCXT_INTERVAL_MAP.keys())}. Halting.")
            sys.exit(1)

        logger.info("Configuration loaded and validated successfully.")
        self._log_key_parameters()  # Log summary after processing

    def _get_and_validate_env(self, key: str, default: Any) -> Any:
        value_str = os.getenv(key)
        is_default = False
        if value_str is None or value_str == "":
            value_str = str(default) if default is not None else None
            is_default = True
            if default is not None: logger.debug(f"Using default for {key}: '{default}'")

        if value_str is None and default is None:  # Optional parameter not set
             return None

        if value_str is None and default is not None:  # Required parameter missing, fatal
             logger.critical(f"Required configuration key {key} not found in .env and has no default! Halting.")
             sys.exit(1)

        validation_info = self.TYPE_VALIDATIONS.get(key)
        if not validation_info: logger.warning(f"No validation defined for '{key}'."); return value_str  # Treat as string

        target_type = validation_info[0] if isinstance(validation_info, tuple) else validation_info

        # Type Casting
        casted_value = None
        try:
            if target_type == bool: casted_value = value_str.lower() in ['true', '1', 'yes', 'y', 'on']
            elif target_type == Decimal: casted_value = Decimal(value_str)
            elif target_type == int: casted_value = int(value_str)
            else: casted_value = str(value_str)
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(f"Cast Error for {key}='{value_str}' to {target_type.__name__}: {e}. Using default: '{default}'.")
            # Re-cast default
            try:
                if default is None: return None
                if target_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                if target_type == Decimal: return Decimal(str(default))
                return target_type(default)
            except: logger.critical(f"Default '{default}' for {key} also fails casting. Halting."); sys.exit(1)

        # Value Validation
        validation_passed = True
        if isinstance(validation_info, tuple):
            # Min/Max or Allowed List
            constraint = validation_info[1] if len(validation_info) > 1 else None
            max_val = validation_info[2] if len(validation_info) > 2 else None

            if isinstance(constraint, list):  # Allowed values
                if casted_value not in constraint: validation_passed = False; logger.error(f"{key} '{casted_value}' not in allowed list {constraint}.")
            elif constraint is not None:  # Min value
                if casted_value < constraint: validation_passed = False; logger.error(f"{key} '{casted_value}' < min '{constraint}'.")
            if max_val is not None:  # Max value
                if casted_value > max_val: validation_passed = False; logger.error(f"{key} '{casted_value}' > max '{max_val}'.")

        if not validation_passed:
             logger.warning(f"Validation failed for {key}. Using default: '{default}'.")
             # Return typed default
             try:
                 if default is None: return None
                 if target_type == bool: return str(default).lower() in ['true', '1', 'yes', 'y', 'on']
                 if target_type == Decimal: return Decimal(str(default))
                 return target_type(default)
             except: logger.critical(f"Default '{default}' for {key} also fails casting. Halting."); sys.exit(1)

        if not is_default and "SECRET" not in key and "KEY" not in key:
             logger.info(f"Loaded {key}: '{casted_value}'")
        elif not is_default:
             logger.info(f"Loaded {key}: '****'")  # Mask secrets

        return casted_value

    def _process_symbol_and_quote(self) -> None:
        """Derives quote currency from the symbol."""
        try:
            # Format: BASE/QUOTE:SETTLE or BASE/QUOTE
            parts = self.symbol.split(':')
            base_quote = parts[0].split('/')
            if len(base_quote) == 2:
                self.base_currency = base_quote[0]
                self.quote_currency = base_quote[1]
                self.settle_currency = parts[1] if len(parts) > 1 else self.quote_currency
                logger.info(f"Parsed Symbol '{self.symbol}': Base={self.base_currency}, Quote={self.quote_currency}, Settle={self.settle_currency}")
                # Override default QUOTE_CURRENCY with parsed value
                self.default_quote_currency = self.quote_currency  # Keep original default if needed
                self.quote_currency = self.settle_currency  # Use settle currency for balance/futures
            else: raise ValueError("Incorrect symbol format")
        except Exception as e:
             logger.warning(f"Could not parse Base/Quote/Settle from SYMBOL '{self.symbol}': {e}. Using default Quote: {self.quote_currency}.")
             # Attempt basic parsing for base/quote at least
             try:
                 base_quote = self.symbol.split('/')
                 self.base_currency = base_quote[0]
                 self.quote_currency = base_quote[1] if len(base_quote) > 1 else self.DEFAULT_CONFIG["QUOTE_CURRENCY"]
                 self.settle_currency = self.quote_currency
                 logger.info(f"Partial Parse '{self.symbol}': Base={self.base_currency}, Quote/Settle={self.quote_currency}")
             except:
                 logger.error("Could not parse base/quote. Using defaults.")
                 self.base_currency = "BASE"
                 self.quote_currency = self.DEFAULT_CONFIG["QUOTE_CURRENCY"]
                 self.settle_currency = self.quote_currency

    def _validate_dependent_params(self) -> None:
        """Validates relationships between parameters."""
        # TSL Method Priority
        if self.trailing_stop_distance_atr_multiplier is not None and self.trailing_stop_callback_rate is not None:
             logger.warning("Both TSL ATR Multiplier and Callback Rate are set. Prioritizing ATR Multiplier.")
             self.trailing_stop_callback_rate = None
        elif self.trailing_stop_distance_atr_multiplier is None and self.trailing_stop_callback_rate is None and self.enable_trailing_stop:
             logger.error("TSL is enabled, but neither ATR Multiplier nor Callback Rate is set for distance. Disabling TSL.")
             self.enable_trailing_stop = False

        # EMA Periods
        if self.indicator_ema_long <= self.indicator_ema_short:
             logger.warning(f"EMA Long ({self.indicator_ema_long}) <= EMA Short ({self.indicator_ema_short}). Setting Long = Short + 1.")
             self.indicator_ema_long = self.indicator_ema_short + 1
        if self.indicator_trend_ema <= self.indicator_ema_long:
             logger.warning(f"Trend EMA ({self.indicator_trend_ema}) <= EMA Long ({self.indicator_ema_long}). Trend filter might be less effective.")

    def _log_key_parameters(self) -> None:
        """Logs important configuration parameters."""
        logger.info("--- Trading Configuration Summary ---")
        logger.info(f"Symbol: {self.symbol} ({self.market_type.capitalize()}) | Timeframe: {self.interval} ({self.ccxt_interval})")
        logger.info(f"Trading: {'LIVE' if self.enable_trading else 'DISABLED'} | Environment: {'REAL MONEY' if not self.use_testnet else 'SANDBOX (Testnet)'}")
        logger.info(f"Risk: {self.risk_percentage * 100:.2f}% | Leverage: {self.leverage}x | Quote: {self.quote_currency}")
        logger.info(f"Initial SL/TP (ATR Mult): {self.sl_atr_multiplier} / {self.tp_atr_multiplier}")
        tsl_mode = "DISABLED"
        if self.enable_trailing_stop:
            dist_method = ""
            if self.trailing_stop_distance_atr_multiplier: dist_method = f"Dist:{self.trailing_stop_distance_atr_multiplier}*ATR"
            elif self.trailing_stop_callback_rate: dist_method = f"Callback:{self.trailing_stop_callback_rate:.3%}"
            else: dist_method = "ERROR:NoDistanceMethod"
            tsl_mode = f"ENABLED ({dist_method}, Act:{self.tsl_activation_atr_multiplier}*ATR)"
        logger.info(f"Trailing SL: {tsl_mode}")
        be_mode = f"ENABLED (Trig:{self.break_even_trigger_atr_multiple}*ATR, Offset:{self.break_even_offset_ticks} ticks)" if self.enable_break_even else "DISABLED"
        logger.info(f"Break-Even SL: {be_mode}")
        logger.info(f"Trend Filter (EMA {self.indicator_trend_ema}): {'ON' if self.trade_only_with_trend else 'OFF'}")
        logger.info(f"Active Weight Set: '{self.active_weight_set}'")
        logger.info("-----------------------------------")


CONFIG = TradingConfig()
EXCHANGE: ccxt.Exchange | None = None
MARKET_INFO: dict | None = None
TIMEZONE = ZoneInfo("America/Chicago")  # Default, adjust if needed

# Global state for active protection details fetched from exchange
active_protection_state: dict[str, Decimal | None] = {
    "stopLossPrice": None, "takeProfitPrice": None,
    "trailingStopValue": None, "trailingStopActivationPrice": None
}


# --- Termux Notification ---
def termux_notify(title: str, content: str) -> None:
    if not os.getenv("TERMUX_VERSION"): return
    try:
        safe_title = title.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '')
        safe_content = content.replace('"', "'").replace('`', "'").replace('$', '').replace('\\', '').replace(';', '')
        full_message = f"{safe_title}: {safe_content}"[:150]
        cmd_list = ['termux-toast', '-s', full_message]
        subprocess.run(cmd_list, capture_output=True, text=True, check=False, timeout=5)
    except Exception as e: logger.warning(f"Termux notification failed: {e}")


# --- Precision Formatting ---
def format_price(symbol: str, price: Any) -> str | None:
    global MARKET_INFO, EXCHANGE
    if not EXCHANGE or not MARKET_INFO or price is None: return str(price)
    try: return EXCHANGE.price_to_precision(symbol, float(price))
    except Exception: return str(price)  # Fallback


def format_amount(symbol: str, amount: Any, rounding_mode=ROUND_DOWN) -> str | None:
    global MARKET_INFO, EXCHANGE
    if not EXCHANGE or not MARKET_INFO or amount is None: return str(amount)
    try:
        ccxt_round = ccxt.TRUNCATE if rounding_mode == ROUND_DOWN else ccxt.ROUND
        return EXCHANGE.amount_to_precision(symbol, float(amount), rounding_mode=ccxt_round)
    except Exception: return str(amount)  # Fallback


# --- Robust Fetching Wrapper ---
def fetch_with_retries(fetch_function, *args, **kwargs) -> Any:
    global EXCHANGE
    if not EXCHANGE: logger.critical("Exchange object is None!"); return None
    last_exception = None
    # Add Bybit V5 category param if applicable
    if 'bybit' in EXCHANGE.id.lower():
        params = kwargs.get('params', {})
        if 'category' not in params: params['category'] = CONFIG.market_type
        kwargs['params'] = params

    for attempt in range(CONFIG.max_fetch_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1}: Calling {fetch_function.__name__}...")
            return fetch_function(*args, **kwargs)  # Success
        except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, requests.exceptions.ReadTimeout) as e:
            last_exception = e; wait_time = 2 ** attempt
            logger.warning(f"{fetch_function.__name__}: Network issue (Attempt {attempt + 1}). Retrying in {wait_time}s... Error: {e}")
        except ccxt.RateLimitExceeded as e:
             last_exception = e; wait_time = 5 * (attempt + 1)
             logger.warning(f"{fetch_function.__name__}: Rate limit hit. Retrying in {wait_time}s... Error: {e}")
        except ccxt.ExchangeNotAvailable as e: last_exception = e; logger.error(f"{fetch_function.__name__}: Exchange unavailable: {e}. Stop."); break
        except ccxt.AuthenticationError as e: last_exception = e; logger.critical(f"{fetch_function.__name__}: Auth error: {e}. Halt."); sys.exit(1)
        except (ccxt.OrderNotFound, ccxt.InsufficientFunds, ccxt.InvalidOrder) as e: last_exception = e; logger.warning(f"{fetch_function.__name__}: Encountered {type(e).__name__}: {e}. Propagating."); raise e
        except ccxt.ExchangeError as e:
            last_exception = e; error_code = getattr(e, 'code', None)
            non_retryable_codes = [110012, 110007, 110043, 10001, 110025, 110014]  # Bybit V5 examples
            if error_code in non_retryable_codes: logger.error(f"{fetch_function.__name__}: Non-retryable error (Code:{error_code}): {e}. Stop."); break
            else: wait_time = 2 * (attempt + 1); logger.warning(f"{fetch_function.__name__}: Exchange error (Code:{error_code}). Retrying in {wait_time}s... Error: {e}")
        except Exception as e: last_exception = e; logger.error(f"{fetch_function.__name__}: Unexpected error: {e}", exc_info=True); break

        if attempt < CONFIG.max_fetch_retries: time.sleep(wait_time)
        else: logger.error(f"{fetch_function.__name__} failed after {CONFIG.max_fetch_retries + 1} attempts. Last exception: {last_exception}"); break

    return None  # Indicate failure


# --- Core Data Fetching ---
def fetch_market_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame | None:
    global EXCHANGE
    logger.info(f"Fetching market data for {symbol} ({timeframe})...")
    if not EXCHANGE: return None
    try:
        ohlcv = fetch_with_retries(EXCHANGE.fetch_ohlcv, symbol, timeframe, limit=limit)
        if not isinstance(ohlcv, list) or not ohlcv: return None
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)
        for col in ["open", "high", "low", "close", "volume"]: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df = df[df['close'] > 0]
        if df.empty: return None
        df = df.set_index("timestamp")
        if not df.index.is_monotonic_increasing: df.sort_index(inplace=True)
        if df.index.duplicated().any(): df = df[~df.index.duplicated(keep='last')]
        logger.info(f"Successfully fetched {len(df)} candles for {symbol}.")
        return df
    except Exception as e: logger.error(f"Error processing market data: {e}", exc_info=True); return None


def fetch_balance() -> Decimal | None:
    global EXCHANGE, CONFIG
    currency = CONFIG.quote_currency
    logger.info(f"Fetching balance for {currency}...")
    if not EXCHANGE: return None
    try:
        balance_data = fetch_with_retries(EXCHANGE.fetch_balance)  # Let fetch_with_retries handle params if needed
        if not balance_data: return None
        balance_val_str = None
        # Prioritize equity > total > free for risk calculation basis
        if currency in balance_data:
             bal_details = balance_data[currency]
             equity = bal_details.get('equity')  # Check for explicit equity field first
             total = bal_details.get('total')
             free = bal_details.get('free')
             if equity is not None: balance_val_str = str(equity); logger.debug("Using 'equity' for balance.")
             elif total is not None: balance_val_str = str(total); logger.debug("Using 'total' for balance.")
             elif free is not None: balance_val_str = str(free); logger.debug("Using 'free' for balance.")

        # Add Bybit V5 parsing as fallback
        elif 'info' in balance_data and isinstance(balance_data['info'].get('result', {}).get('list'), list):
             for account in balance_data['info']['result']['list']:
                 if account.get('accountType') == 'CONTRACT' or account.get('accountType') == 'UNIFIED':  # Look in relevant account types
                      equity = account.get('equity')  # V5 equity
                      wallet = account.get('walletBalance')
                      available = account.get('availableToWithdraw')
                      if equity is not None: balance_val_str = str(equity); logger.debug("Using V5 'equity' for balance."); break
                      if wallet is not None: balance_val_str = str(wallet); logger.debug("Using V5 'walletBalance' for balance."); break
                      if available is not None: balance_val_str = str(available); logger.debug("Using V5 'availableToWithdraw' for balance."); break
             if balance_val_str: pass  # Found it
             else: logger.warning(f"Could not find {currency} balance in Bybit V5 structure.")

        if balance_val_str:
             balance_dec = Decimal(balance_val_str)
             logger.info(f"Balance ({currency}): {balance_dec:.4f}")
             return balance_dec
        else: logger.error(f"Could not parse balance for {currency}."); return None
    except Exception as e: logger.error(f"Error fetching balance: {e}", exc_info=True); return None


def get_market_info(symbol: str) -> dict | None:
    global EXCHANGE, MARKET_INFO
    logger.info(f"Fetching market info for {symbol}...")
    if not EXCHANGE: return None
    try:
        if not EXCHANGE.markets or symbol not in EXCHANGE.markets:
            logger.info("Markets not loaded or symbol missing, reloading...")
            EXCHANGE.load_markets(reload=True)
        if symbol not in EXCHANGE.markets:
             logger.error(f"Symbol {symbol} not found after reload."); return None
        MARKET_INFO = EXCHANGE.market(symbol)
        MARKET_INFO['is_contract'] = MARKET_INFO.get('contract', False) or MARKET_INFO.get('type') in ['swap', 'future']
        logger.info(f"Market info loaded for {symbol}. Type: {MARKET_INFO.get('type')}, Contract: {MARKET_INFO['is_contract']}")
        logger.debug(f"Market Details: {MARKET_INFO}")
        return MARKET_INFO
    except Exception as e: logger.error(f"Error fetching market info: {e}", exc_info=True); return None


def get_current_position(symbol: str) -> dict | None:
    """Fetches and processes the current position, including protection state."""
    global EXCHANGE, MARKET_INFO, active_protection_state
    logger.info(f"Fetching current position for {symbol}...")
    if not EXCHANGE: return None

    # Clear previous protection state before fetch
    active_protection_state = {"stopLossPrice": None, "takeProfitPrice": None, "trailingStopValue": None, "trailingStopActivationPrice": None}

    try:
        positions = fetch_with_retries(EXCHANGE.fetch_positions, [symbol])
        if positions is None: return None
        if not isinstance(positions, list): logger.error("fetch_positions returned non-list."); return None

        filtered_positions = [p for p in positions if p.get('symbol') == symbol]
        if not filtered_positions: logger.info(f"No open position found for {symbol}."); return None

        pos = filtered_positions[0]  # Assume one position per symbol
        standardized_pos = {'info': pos.get('info', {})}  # Start with raw info
        info_dict = standardized_pos['info']

        # Symbol
        standardized_pos['symbol'] = pos.get('symbol')

        # Size (Decimal)
        size_str = pos.get('contracts') or info_dict.get('size')
        if size_str is None: logger.error("Could not determine position size."); return None
        standardized_pos['size'] = Decimal(str(size_str))

        # Side ('long' or 'short')
        side = pos.get('side')
        if side not in ['long', 'short']:
            if standardized_pos['size'] > Decimal('1e-9'): side = 'long'
            elif standardized_pos['size'] < Decimal('-1e-9'): side = 'short'
            else: logger.error("Position size near zero, cannot determine side."); return None
        standardized_pos['side'] = side

        # Entry Price (Decimal)
        entry_price_str = pos.get('entryPrice') or info_dict.get('avgPrice')
        if entry_price_str: standardized_pos['entryPrice'] = Decimal(str(entry_price_str))
        else: logger.warning("Missing entry price."); standardized_pos['entryPrice'] = None

        # Other fields (Decimal or None)
        for key, info_key in [('unrealizedPnl', 'unrealisedPnl'), ('leverage', 'leverage'), ('liquidationPrice', 'liqPrice')]:
             val_str = pos.get(key) or info_dict.get(info_key)
             standardized_pos[key] = Decimal(str(val_str)) if val_str is not None else None

        # --- Protection State (Crucial) ---
        sl_str = pos.get('stopLossPrice') or info_dict.get('stopLoss')
        tp_str = pos.get('takeProfitPrice') or info_dict.get('takeProfit')
        tsl_val_str = info_dict.get('trailingStop')  # Distance
        tsl_act_str = info_dict.get('activePrice')  # Activation

        if sl_str and str(sl_str) != '0': active_protection_state['stopLossPrice'] = Decimal(str(sl_str))
        if tp_str and str(tp_str) != '0': active_protection_state['takeProfitPrice'] = Decimal(str(tp_str))
        if tsl_val_str and str(tsl_val_str) != '0': active_protection_state['trailingStopValue'] = Decimal(str(tsl_val_str))
        if tsl_act_str and str(tsl_act_str) != '0': active_protection_state['trailingStopActivationPrice'] = Decimal(str(tsl_act_str))
        standardized_pos['protectionState'] = active_protection_state.copy()
        logger.debug(f"Fetched Protection State: {active_protection_state}")

        # Timestamp & Index
        standardized_pos['timestamp_ms'] = pos.get('timestamp') or info_dict.get('updatedTime')
        standardized_pos['positionIdx'] = int(info_dict.get('positionIdx', 0))

        # Margin Mode
        standardized_pos['marginMode'] = str(info_dict.get('tradeMode', 'cross')).lower()

        if standardized_pos['entryPrice'] is None: logger.error("Processed position lacks valid entry price."); return None

        logger.info(f"Active {side.upper()} position found for {symbol}. Size: {standardized_pos['size']}")
        return standardized_pos

    except Exception as e: logger.error(f"Error fetching/processing position: {e}", exc_info=True); return None


# --- Indicator Analysis Class ---
class TradingAnalyzer:
    """Analyzes trading data using pandas_ta and generates weighted signals."""
    def __init__(self, df: pd.DataFrame, logger: logging.Logger, config_dict: dict[str, Any], market_info: dict) -> None:
        self.df = df
        self.logger = logger
        self.config = config_dict  # Use the passed config dict directly
        self.market_info = market_info
        self.symbol = market_info.get('symbol', '???')
        self.interval = self.config.get("interval", "???")
        self.indicator_values: dict[str, Any] = {}  # Stores latest Decimal/float values
        self.signals: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 1}
        self.active_weight_set_name = self.config.get("active_weight_set", "default")
        self.weights = self.config.get("weight_sets", {}).get(self.active_weight_set_name, {})
        self.fib_levels_data: dict[str, Decimal] = {}
        self.ta_column_names: dict[str, str | None] = {}

        if not self.weights: logger.error(f"Weight set '{self.active_weight_set_name}' empty/missing.")
        self._calculate_all_indicators()
        self._update_latest_indicator_values()
        self.calculate_fibonacci_levels()

    def get_price_precision(self) -> int:
        """Gets price precision (decimal places) from market info."""
        try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if isinstance(price_precision_val, int) and price_precision_val >= 0: return price_precision_val
            if isinstance(price_precision_val, (float, str)):
                tick = Decimal(str(price_precision_val))
                if tick > 0: return abs(tick.normalize().as_tuple().exponent)
            # Add other fallbacks if needed (e.g., limits.price.min)
        except Exception as e: logger.warning(f"Error getting price precision: {e}. Using default.")
        return 4  # Default

    def get_min_tick_size(self) -> Decimal:
         """Gets minimum price tick size from market info."""
         try:
            precision_info = self.market_info.get('precision', {})
            price_precision_val = precision_info.get('price')
            if isinstance(price_precision_val, (float, str)):
                 tick = Decimal(str(price_precision_val))
                 if tick > 0: return tick
            if isinstance(price_precision_val, int) and price_precision_val >= 0:
                 return Decimal('1e-' + str(price_precision_val))
            # Add other fallbacks
         except Exception as e: logger.warning(f"Error getting tick size: {e}. Using default.")
         return Decimal('1e-' + str(self.get_price_precision()))  # Fallback based on precision

    def _get_ta_col_name(self, base_name: str, result_df: pd.DataFrame) -> str | None:
        """Helper to find the actual column name generated by pandas_ta."""
        # Expected patterns (needs refinement based on actual pandas_ta output)
        expected_patterns = {
            "ATR": [f"ATRr_{self.config.get('indicator_atr_period')}"],
            "EMA_Short": [f"EMA_{self.config.get('indicator_ema_short')}"],
            "EMA_Long": [f"EMA_{self.config.get('indicator_ema_long')}"],
            "Trend_EMA": [f"EMA_{self.config.get('indicator_trend_ema')}"],  # Specific key for trend EMA
            "StochRSI_K": [f"STOCHRSIk_{self.config.get('indicator_stoch_rsi_window')}_"],  # Partial pattern
            "StochRSI_D": [f"STOCHRSId_{self.config.get('indicator_stoch_rsi_window')}_"],  # Partial pattern
            "RSI": [f"RSI_{self.config.get('indicator_rsi_window')}"],
            "CCI": [f"CCI_{self.config.get('indicator_cci_window')}_"],  # Partial pattern
            # Add patterns for other indicators if used (MFI, WR, BBands, PSAR, SMA10, MOM, VWAP)
        }
        patterns = expected_patterns.get(base_name, [])
        for col in result_df.columns:
            for pattern in patterns:
                if col.startswith(pattern): return col  # Use startswith for flexibility
        # Basic fallback
        for col in result_df.columns:
            if base_name.lower() in col.lower(): return col
        logger.warning(f"Could not map TA col name for '{base_name}'"); return None

    def _calculate_all_indicators(self) -> None:
        if self.df.empty: logger.warning("DataFrame empty, skipping indicator calc."); return
        # Check data length vs required periods (simplified check)
        required_len = self.config.get('ohlcv_limit', 200) // 2  # Rough estimate
        if len(self.df) < required_len: logger.warning(f"Data length {len(self.df)} < {required_len}, indicator accuracy may suffer.")

        try:
            df_calc = self.df.copy()
            cfg = self.config  # Shortcut for config access

            # Always calculate ATR (needed for risk/SL/TP)
            atr_p = cfg['indicator_atr_period']; df_calc.ta.atr(length=atr_p, append=True); self.ta_column_names["ATR"] = self._get_ta_col_name("ATR", df_calc)

            # EMA Alignment & Trend
            ema_s = cfg['indicator_ema_short']; df_calc.ta.ema(length=ema_s, append=True); self.ta_column_names["EMA_Short"] = self._get_ta_col_name("EMA_Short", df_calc)
            ema_l = cfg['indicator_ema_long']; df_calc.ta.ema(length=ema_l, append=True); self.ta_column_names["EMA_Long"] = self._get_ta_col_name("EMA_Long", df_calc)
            ema_t = cfg['indicator_trend_ema']; df_calc.ta.ema(length=ema_t, append=True); self.ta_column_names["Trend_EMA"] = self._get_ta_col_name("Trend_EMA", df_calc)

            # Stochastic RSI
            st_win = cfg['indicator_stoch_rsi_window']; st_k = cfg['indicator_stoch_rsi_k']; st_d = cfg['indicator_stoch_rsi_d']
            stochrsi_df = df_calc.ta.stochrsi(length=st_win, rsi_length=cfg['indicator_rsi_window'], k=st_k, d=st_d)  # Use rsi_window here
            if stochrsi_df is not None: df_calc = pd.concat([df_calc, stochrsi_df], axis=1)
            self.ta_column_names["StochRSI_K"] = self._get_ta_col_name("StochRSI_K", df_calc)
            self.ta_column_names["StochRSI_D"] = self._get_ta_col_name("StochRSI_D", df_calc)

            # Standard RSI
            rsi_p = cfg['indicator_rsi_window']; df_calc.ta.rsi(length=rsi_p, append=True); self.ta_column_names["RSI"] = self._get_ta_col_name("RSI", df_calc)

            # CCI
            cci_p = cfg['indicator_cci_window']; df_calc.ta.cci(length=cci_p, append=True); self.ta_column_names["CCI"] = self._get_ta_col_name("CCI", df_calc)

            # --- Add calculations for other indicators from weight sets if needed ---
            # Example: MFI, WR, BBands, PSAR, SMA10, Momentum, VWAP, Volume MA
            # if self.weights.get("mfi", 0) != 0: ... df_calc.ta.mfi(...) ... self.ta_column_names["MFI"] = ...
            # ... etc ...

            self.df = df_calc  # Update instance df
            logger.debug("Indicator calculations complete.")

        except Exception as e: logger.error(f"Error calculating indicators: {e}", exc_info=True)

    def _update_latest_indicator_values(self) -> None:
        """Updates indicator_values dict with latest values, handling Decimal/float."""
        if self.df.empty or self.df.iloc[-1].isnull().all():
            logger.warning("Cannot update latest indicator values: DataFrame empty or last row is all NaN.")
            self.indicator_values = dict.fromkeys(self.ta_column_names)  # Set all to None
            return

        latest = self.df.iloc[-1]
        updated_values = {}
        for key, col_name in self.ta_column_names.items():
            if col_name and col_name in latest.index and pd.notna(latest[col_name]):
                value = latest[col_name]
                try:
                    # Store ATR and price-based indicators as Decimal, others as float
                    if key in ["ATR", "EMA_Short", "EMA_Long", "Trend_EMA", "BB_Lower", "BB_Middle", "BB_Upper", "SMA10", "VWAP", "PSAR_long", "PSAR_short"]:
                        updated_values[key] = Decimal(str(value))
                    else:  # Stoch, RSI, CCI, WR, MFI, Momentum etc. as float
                        updated_values[key] = float(value)
                except (ValueError, TypeError, InvalidOperation):
                    updated_values[key] = None  # Store None if conversion fails
            else:
                updated_values[key] = None  # Store None if column missing or value is NaN

        # Add essential OHLCV as Decimal
        for base_col in ['open', 'high', 'low', 'close', 'volume']:
            if base_col in latest.index and pd.notna(latest[base_col]):
                try: updated_values[base_col.capitalize()] = Decimal(str(latest[base_col]))
                except: updated_values[base_col.capitalize()] = None
            else: updated_values[base_col.capitalize()] = None

        self.indicator_values = updated_values
        log_vals = {k: f"{v:.4f}" if isinstance(v, Decimal) else f"{v:.2f}" if isinstance(v, float) else v for k, v in updated_values.items() if v is not None}
        logger.debug(f"Latest indicator values updated: {log_vals}")

    def calculate_fibonacci_levels(self, window: int | None = None) -> dict[str, Decimal]:
        window = window or self.config.get("fibonacci_window", 50)
        if len(self.df) < window: return {}
        df_slice = self.df.tail(window)
        try:
            high = Decimal(str(df_slice["high"].dropna().max()))
            low = Decimal(str(df_slice["low"].dropna().min()))
            diff = high - low
            levels = {}
            if diff > 0:
                 rounding_factor = Decimal('1e-' + str(self.get_price_precision()))
                 for level_pct in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
                     level_name = f"Fib_{level_pct * 100:.1f}%"
                     level_price = (high - (diff * Decimal(str(level_pct))))
                     levels[level_name] = level_price.quantize(rounding_factor, rounding=ROUND_DOWN)
            else:  # Handle no range
                 levels = {f"Fib_{pct * 100:.1f}%": high for pct in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}
            self.fib_levels_data = levels
            logger.debug(f"Fibonacci Levels: { {k:f'{v:.{self.get_price_precision()}f}' for k, v in levels.items()} }")
            return levels
        except Exception as e: logger.error(f"Fibonacci error: {e}"); self.fib_levels_data = {}; return {}

    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 3) -> list[tuple[str, Decimal]]:
         if not self.fib_levels_data or not isinstance(current_price, Decimal) or current_price <= 0: return []
         try:
             dists = [{'n': n, 'l': l, 'd': abs(current_price - l)} for n, l in self.fib_levels_data.items() if isinstance(l, Decimal)]
             dists.sort(key=lambda x: x['d'])
             return [(i['n'], i['l']) for i in dists[:num_levels]]
         except Exception as e: logger.error(f"Error finding nearest Fibs: {e}"); return []

    def calculate_entry_tp_sl(self, entry_price_estimate: Decimal, signal: str) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
        """Calculates TP and initial SL based on entry, signal, and ATR. Returns Decimals."""
        if signal not in ["BUY", "SELL"]: return entry_price_estimate, None, None
        atr_val = self.indicator_values.get("ATR")  # Decimal or None
        if not isinstance(atr_val, Decimal) or atr_val <= 0: logger.warning("Cannot calc TP/SL: Invalid ATR."); return entry_price_estimate, None, None
        if not isinstance(entry_price_estimate, Decimal) or entry_price_estimate <= 0: logger.warning("Cannot calc TP/SL: Invalid entry estimate."); return entry_price_estimate, None, None

        try:
            tp_mult = Decimal(str(self.config.get("tp_atr_multiplier", "2.0")))
            sl_mult = Decimal(str(self.config.get("sl_atr_multiplier", "1.5")))
            price_prec = self.get_price_precision()
            rounding_factor = Decimal('1e-' + str(price_prec))
            min_tick = self.get_min_tick_size()

            tp, sl = None, None
            if signal == "BUY":
                 tp = (entry_price_estimate + atr_val * tp_mult).quantize(rounding_factor, rounding=ROUND_UP)
                 sl = (entry_price_estimate - atr_val * sl_mult).quantize(rounding_factor, rounding=ROUND_DOWN)
                 # Validation
                 if sl >= entry_price_estimate: sl = (entry_price_estimate - min_tick).quantize(rounding_factor, rounding=ROUND_DOWN)
                 if tp <= entry_price_estimate: tp = None  # Invalidate non-profitable TP
            else:  # SELL
                 tp = (entry_price_estimate - atr_val * tp_mult).quantize(rounding_factor, rounding=ROUND_DOWN)
                 sl = (entry_price_estimate + atr_val * sl_mult).quantize(rounding_factor, rounding=ROUND_UP)
                 # Validation
                 if sl <= entry_price_estimate: sl = (entry_price_estimate + min_tick).quantize(rounding_factor, rounding=ROUND_UP)
                 if tp >= entry_price_estimate: tp = None

            if sl is not None and sl <= 0: logger.error("Stop loss calc resulted in non-positive price!"); sl = None
            if tp is not None and tp <= 0: logger.warning("Take profit calc resulted in non-positive price!"); tp = None

            logger.debug(f"Calculated TP/SL: Entry={entry_price_estimate:.{price_prec}f}, TP={tp}, SL={sl}, ATR={atr_val:.{price_prec + 1}f}")
            return entry_price_estimate, tp, sl
        except Exception as e: logger.error(f"Error calculating TP/SL: {e}", exc_info=True); return entry_price_estimate, None, None

    # --- Signal Generation (Simplified Example - Requires _check methods) ---
    def generate_trading_signal(self, current_price: Decimal, orderbook_data: dict | None) -> str:
        """Generates BUY/SELL/HOLD signal based on simple EMA cross and Trend Filter."""
        # This is a placeholder - the full weighted scoring requires implementing all _check_ methods
        # like in livexy.py or wb_sig_v10.py if that complexity is desired.
        self.signals = {"BUY": 0, "SELL": 0, "HOLD": 1}  # Default HOLD
        ema_s = self.indicator_values.get("EMA_Short")  # Decimal or None
        ema_l = self.indicator_values.get("EMA_Long")   # Decimal or None
        trend_ema = self.indicator_values.get("Trend_EMA")  # Decimal or None

        if not all(isinstance(v, Decimal) for v in [ema_s, ema_l, trend_ema, current_price]):
             logger.warning("Cannot generate signal: Missing indicator values.")
             return "HOLD"

        # Basic EMA Cross Logic
        bullish_cross = ema_s > ema_l
        bearish_cross = ema_s < ema_l

        # Trend Filter
        price_above_trend = current_price > trend_ema
        price_below_trend = current_price < trend_ema

        final_signal = "HOLD"
        reason = "No signal condition met"

        if bullish_cross:
            if self.config.get('trade_only_with_trend', True):
                 if price_above_trend: final_signal = "BUY"; reason = "EMA Bull Cross + Above Trend"
                 else: reason = "Long Blocked: Below Trend"
            else: final_signal = "BUY"; reason = "EMA Bull Cross (Trend Filter OFF)"
        elif bearish_cross:
             if self.config.get('trade_only_with_trend', True):
                  if price_below_trend: final_signal = "SELL"; reason = "EMA Bear Cross + Below Trend"
                  else: reason = "Short Blocked: Above Trend"
             else: final_signal = "SELL"; reason = "EMA Bear Cross (Trend Filter OFF)"

        logger.info(f"Signal Check ({self.symbol} @ {current_price:.{self.get_price_precision()}f}): {reason} ==> {final_signal}")

        if final_signal != "HOLD": self.signals[final_signal] = 1
        return final_signal


# --- Position Management & Trading Functions ---
# (calculate_position_size, place_trade, _set_position_protection, set_trailing_stop_loss, set_breakeven_stop_loss)
# These are complex and critical. Using the robust versions developed in the previous merge step.

def set_leverage_ccxt(exchange: ccxt.Exchange, symbol: str, leverage: int, market_info: dict, logger: logging.Logger) -> bool:
    """Sets leverage for a symbol using CCXT, handling Bybit V5 specifics."""
    lg = logger
    is_contract = market_info.get('is_contract', False)
    if not is_contract: lg.info(f"Leverage skipped for {symbol} (Spot)."); return True
    if leverage <= 0: lg.warning(f"Leverage skipped: Invalid value ({leverage})."); return False
    if not exchange.has.get('setLeverage'): lg.error(f"{exchange.id} does not support setLeverage."); return False

    try:
        lg.info(f"Setting leverage for {symbol} to {leverage}x...")
        params = {}
        if 'bybit' in exchange.id.lower():
             params = {'buyLeverage': str(leverage), 'sellLeverage': str(leverage)}
             lg.debug(f"Using Bybit V5 params: {params}")

        response = fetch_with_retries(exchange.set_leverage, leverage, symbol, params=params)
        lg.debug(f"Set leverage raw response: {response}")

        # Basic success check (non-exception implies success for setLeverage often)
        lg.info(f"{NEON_GREEN}Leverage set/requested to {leverage}x for {symbol}.{RESET}")
        return True

    except ccxt.ExchangeError as e:
        err_str = str(e).lower(); code = getattr(e, 'code', None)
        lg.error(f"{NEON_RED}Leverage Error ({symbol}): {e} (Code:{code}){RESET}")
        if code == 110045 or "not modified" in err_str: lg.info("Leverage already set."); return True  # Treat as success
        # Add hints for other codes if needed (risk limit, margin mode etc.)
    except Exception as e: lg.error(f"{NEON_RED}Unexpected leverage error ({symbol}): {e}{RESET}", exc_info=True)
    return False


def calculate_position_size(
    balance: Decimal, risk_per_trade: Decimal, initial_stop_loss_price: Decimal,
    entry_price: Decimal, market_info: dict, exchange: ccxt.Exchange, logger: logging.Logger
) -> Decimal | None:
    """Calculates position size based on risk, SL, balance, and market constraints."""
    lg = logger; symbol = market_info.get('symbol', '???'); quote = market_info.get('settle', 'QUOTE')
    is_contract = market_info.get('is_contract', False); base = market_info.get('base', 'BASE')
    size_unit = "Contracts" if is_contract else base

    if not all(isinstance(v, Decimal) and v > 0 for v in [balance, entry_price, initial_stop_loss_price]) \
       or not (0 < risk_per_trade < 1) or initial_stop_loss_price == entry_price:
        lg.error(f"Sizing fail ({symbol}): Invalid inputs."); return None
    if 'limits' not in market_info or 'precision' not in market_info:
        lg.error(f"Sizing fail ({symbol}): Market info missing."); return None

    try:
        risk_amount = balance * risk_per_trade
        sl_dist = abs(entry_price - initial_stop_loss_price)
        if sl_dist <= 0: lg.error(f"Sizing fail ({symbol}): SL dist zero/neg."); return None

        contract_size = Decimal(str(market_info.get('contractSize', '1')))
        if contract_size <= 0: contract_size = Decimal('1')

        # Simplified: Assuming Linear/Spot logic primarily
        is_linear = market_info.get('linear', not market_info.get('inverse', False))
        if is_linear or not is_contract:
             denominator = sl_dist * contract_size
             if denominator <= 0: lg.error(f"Sizing fail ({symbol}): Zero denominator."); return None
             calculated_size = risk_amount / denominator
        else:  # Placeholder for Inverse - needs specific exchange logic
             lg.error(f"Inverse contract sizing logic required but not fully implemented for {symbol}. Cannot size."); return None

        lg.info(f"Sizing ({symbol}): RiskAmt={risk_amount:.4f} {quote}, SLDist={sl_dist:.{MARKET_INFO['precision'].get('price', 4)}f}, InitialSize={calculated_size:.8f} {size_unit}")

        # Apply Limits & Precision
        limits = market_info['limits']['amount']
        min_amount = Decimal(str(limits.get('min', '0')))
        max_amount = Decimal(str(limits.get('max', 'inf')))
        cost_limits = market_info['limits'].get('cost', {})
        min_cost = Decimal(str(cost_limits.get('min', '0')))
        max_cost = Decimal(str(cost_limits.get('max', 'inf')))

        adj_size = max(min_amount, min(calculated_size, max_amount))
        if adj_size != calculated_size: lg.warning(f"Size adjusted by Amount Limits: {calculated_size:.8f} -> {adj_size:.8f}")

        # Cost Check (Simplified for Linear/Spot)
        est_cost = adj_size * entry_price * contract_size
        if min_cost > 0 and est_cost < min_cost: lg.error(f"Cost {est_cost:.4f} < MinCost {min_cost}. Cannot meet limit."); return None
        if max_cost > 0 and est_cost > max_cost:
            # Reduce size to meet max cost
            if entry_price * contract_size > 0:
                 adj_size = max_cost / (entry_price * contract_size)
                 lg.warning(f"Size reduced by Max Cost Limit: -> {adj_size:.8f}")
                 if adj_size < min_amount: lg.error("Size reduced below Min Amount by Max Cost."); return None
            else: lg.error("Cannot adjust for Max Cost (zero price/contract size)."); return None

        # Apply Amount Precision (Format and convert back)
        final_size_str = format_amount(symbol, adj_size, ROUND_DOWN)
        if final_size_str is None: lg.error("Failed to format amount."); return None
        final_size = Decimal(final_size_str)

        # Final Validations
        if final_size <= 0: lg.error("Final size zero/negative."); return None
        if final_size < min_amount: lg.error(f"Final size {final_size} < Min Amount {min_amount}."); return None
        final_cost = final_size * entry_price * contract_size  # Recheck cost
        if min_cost > 0 and final_cost < min_cost: lg.error(f"Final cost {final_cost:.4f} < Min Cost {min_cost}."); return None

        lg.info(f"{NEON_GREEN}Final Size ({symbol}): {final_size} {size_unit}{RESET}")
        return final_size
    except Exception as e: lg.error(f"Unexpected sizing error ({symbol}): {e}", exc_info=True); return None


def place_trade(
    exchange: ccxt.Exchange, symbol: str, trade_signal: str, position_size: Decimal,
    market_info: dict, logger: logging.Logger, order_type: str = 'market',
    limit_price: Decimal | None = None, reduce_only: bool = False, params: dict | None = None
) -> dict | None:
    """Places market or limit order using CCXT."""
    lg = logger; side = 'buy' if trade_signal == "BUY" else 'sell'
    action = "Close/Reduce" if reduce_only else "Open/Increase"

    try:  # Size validation
        amount_float = float(position_size)
        min_amount = Decimal(str(market_info['limits']['amount'].get('min', '0')))
        if amount_float <= 0 or position_size < min_amount: raise ValueError(f"Invalid size {position_size}")
    except Exception as e: lg.error(f"Trade Aborted ({symbol}): Invalid size {e}"); return None
    if order_type == 'limit' and (not isinstance(limit_price, Decimal) or limit_price <= 0):
        lg.error(f"Trade Aborted ({symbol}): Invalid limit price {limit_price}"); return None

    # Prepare Params
    order_params = {'reduceOnly': reduce_only}
    if params: order_params.update(params)
    order_params['reduceOnly'] = reduce_only  # Ensure override
    if reduce_only and order_type == 'market': order_params['timeInForce'] = 'IOC'
    if 'bybit' in exchange.id.lower(): order_params['positionIdx'] = 0  # One-Way

    lg.info(f"Placing {action} {side.upper()} {order_type.upper()} order for {symbol}:")
    lg.info(f"  Size: {position_size} ({amount_float:.8f})")
    if order_type == 'limit': lg.info(f"  Price: {format_price(symbol, limit_price)}")
    lg.info(f"  Params: {order_params}")

    try:
        order_fn = exchange.create_market_order if order_type == 'market' else exchange.create_limit_order
        price_arg = None if order_type == 'market' else float(limit_price)
        order = fetch_with_retries(order_fn, symbol, side, amount_float, price_arg, order_params)

        if order:
            lg.info(f"{NEON_GREEN}Order Request Sent ({symbol}): ID={order.get('id', 'N/A')}, Status={order.get('status', 'N/A')}{RESET}")
            lg.debug(f"Raw Order Response: {order}")
            return order
        else: lg.error(f"{NEON_RED}Order placement failed after retries ({symbol} {side}).{RESET}"); return None
    except (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.ExchangeError) as e:
        lg.error(f"{NEON_RED}Trade Placement Error ({symbol} {side}): {type(e).__name__} - {e}{RESET}")
    except Exception as e: lg.error(f"{NEON_RED}Unexpected trade error ({symbol} {side}): {e}{RESET}", exc_info=True)
    return None


def _set_position_protection(  # Internal helper for Bybit V5 protection endpoint
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    logger: logging.Logger, stop_loss_price: Decimal | None = None,
    take_profit_price: Decimal | None = None, trailing_stop_distance: Decimal | None = None,
    tsl_activation_price: Decimal | None = None
) -> bool:
    lg = logger
    if 'bybit' not in exchange.id.lower(): lg.error("Protection setting currently Bybit V5 specific."); return False
    if not market_info.get('is_contract', False): lg.warning("Protection skipped (not contract)."); return False
    if not position_info: lg.error("Missing position info."); return False

    pos_side = position_info.get('side'); pos_idx = position_info.get('positionIdx', 0)
    if pos_side not in ['long', 'short']: lg.error(f"Invalid pos side '{pos_side}'."); return False

    has_sl = isinstance(stop_loss_price, Decimal) and stop_loss_price > 0
    has_tp = isinstance(take_profit_price, Decimal) and take_profit_price > 0
    has_tsl = isinstance(trailing_stop_distance, Decimal) and trailing_stop_distance > 0 and \
              isinstance(tsl_activation_price, Decimal) and tsl_activation_price > 0
    if not has_sl and not has_tp and not has_tsl: lg.info("No valid protection provided."); return True

    params = {'category': CONFIG.market_type, 'symbol': market_info['id'], 'tpslMode': 'Full', 'positionIdx': pos_idx}
    log_parts = [f"Setting protection ({symbol} {pos_side.upper()} Idx:{pos_idx}):"]

    try:  # Formatting
        def fmt_p(p): return format_price(symbol, p) if p and p > 0 else None

        def fmt_tsl_d(d) -> str | None:  # Format TSL distance based on tick precision
            if not d or d <= 0: return None
            try:
                tick = Decimal(str(market_info['precision']['price']))
                num_decimals = abs(tick.normalize().as_tuple().exponent)
                return f"{d:.{num_decimals}f}"
            except: return None

        if has_tsl:
            fmt_d = fmt_tsl_d(trailing_stop_distance); fmt_a = fmt_p(tsl_activation_price)
            if fmt_d and fmt_a:
                params['trailingStop'] = fmt_d; params['activePrice'] = fmt_a
                params['slTriggerBy'] = CONFIG.sl_trigger_by  # Use SL trigger for TSL
                log_parts.append(f"  TSL: Dist={fmt_d}, Act={fmt_a}")
                has_sl = False  # TSL overrides SL on Bybit
            else: lg.error("Failed to format TSL params."); has_tsl = False

        if has_sl:
            fmt_sl = fmt_p(stop_loss_price)
            if fmt_sl: params['stopLoss'] = fmt_sl; params['slTriggerBy'] = CONFIG.sl_trigger_by; log_parts.append(f"  SL: {fmt_sl}")
            else: has_sl = False

        if has_tp:
            fmt_tp = fmt_p(take_profit_price)
            if fmt_tp: params['takeProfit'] = fmt_tp; params['tpTriggerBy'] = CONFIG.tp_trigger_by; log_parts.append(f"  TP: {fmt_tp}")
            else: has_tp = False

        if not params.get('stopLoss') and not params.get('takeProfit') and not params.get('trailingStop'):
            lg.error("No valid protection params remained after formatting."); return False

    except Exception as fmt_err: lg.error(f"Error formatting protection params: {fmt_err}", exc_info=True); return False

    # API Call
    lg.info("\n".join(log_parts)); lg.debug(f"  API Params: {params}")
    try:
        response = fetch_with_retries(exchange.private_post, '/v5/position/set-trading-stop', params=params)
        if response is None: return False  # Fetch failed
        ret_code = response.get('retCode'); ret_msg = response.get('retMsg', 'Unknown')
        if ret_code == 0:
            if "not modified" in ret_msg.lower(): lg.info(f"{NEON_YELLOW}Protection already set/partially modified ({symbol}): {ret_msg}{RESET}")
            else: lg.info(f"{NEON_GREEN}Protection set/updated successfully ({symbol}).{RESET}")
            return True
        else:
            lg.error(f"{NEON_RED}Failed to set protection ({symbol}): {ret_msg} (Code:{ret_code}){RESET}")
            # Add hints based on codes if needed
            return False
    except Exception as e: lg.error(f"{NEON_RED}Unexpected error setting protection ({symbol}): {e}{RESET}", exc_info=True); return False


def set_trailing_stop_loss(  # Calculates parameters based on config, calls internal helper
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    config_dict: dict[str, Any], analyzer: TradingAnalyzer, logger: logging.Logger,
    take_profit_price: Decimal | None = None
) -> bool:
    lg = logger; config = config_dict  # Shortcut
    if not config.get("enable_trailing_stop", False): return False

    try:  # Get config params
        act_atr_mult = Decimal(str(config.get("tsl_activation_atr_multiplier", "1.0")))
        dist_atr_mult = config.get("trailing_stop_distance_atr_multiplier")  # Optional Decimal
        callback_rate = config.get("trailing_stop_callback_rate")  # Optional Decimal
    except Exception as e: lg.error(f"Invalid TSL config params: {e}"); return False

    try:  # Get position/analysis data
        entry_price = position_info.get('entryPrice')
        side = position_info.get('side')
        current_atr = analyzer.indicator_values.get("ATR")  # Decimal
        if not isinstance(entry_price, Decimal) or entry_price <= 0 or side not in ['long', 'short']: raise ValueError("Invalid position info")
        if not isinstance(current_atr, Decimal) or current_atr <= 0: raise ValueError(f"Invalid ATR {current_atr}")

        price_prec = analyzer.get_price_precision()
        price_rounding = Decimal('1e-' + str(price_prec))
        min_tick = analyzer.get_min_tick_size()

        # Calculate Activation Price (based on ATR multiplier)
        act_offset = current_atr * act_atr_mult
        act_price = (entry_price + act_offset if side == 'long' else entry_price - act_offset)
        act_price = act_price.quantize(price_rounding, rounding=ROUND_UP if side == 'long' else ROUND_DOWN)
        if side == 'long' and act_price <= entry_price: act_price = (entry_price + min_tick).quantize(price_rounding, ROUND_UP)
        if side == 'short' and act_price >= entry_price: act_price = (entry_price - min_tick).quantize(price_rounding, ROUND_DOWN)
        if act_price <= 0: raise ValueError(f"Invalid activation price {act_price}")

        # Calculate Trailing Distance
        trail_dist: Decimal | None = None
        if dist_atr_mult is not None:  # Prioritize ATR distance
             if not isinstance(dist_atr_mult, Decimal): dist_atr_mult = Decimal(str(dist_atr_mult))  # Ensure Decimal
             trail_dist_raw = current_atr * dist_atr_mult
             lg.debug(f"TSL Dist Method: ATR Multiplier ({dist_atr_mult})")
        elif callback_rate is not None:  # Fallback to callback rate
             if not isinstance(callback_rate, Decimal): callback_rate = Decimal(str(callback_rate))  # Ensure Decimal
             if callback_rate <= 0: raise ValueError("Invalid callback rate")
             trail_dist_raw = act_price * callback_rate  # Distance based on activation price
             lg.debug(f"TSL Dist Method: Callback Rate ({callback_rate:.3%})")
        else: raise ValueError("No TSL distance method configured")

        # Round distance UP to nearest tick, ensure > min_tick
        trail_dist = (trail_dist_raw / min_tick).quantize(Decimal('1'), rounding=ROUND_UP) * min_tick
        if trail_dist < min_tick: trail_dist = min_tick
        if trail_dist <= 0: raise ValueError(f"Invalid trailing distance {trail_dist}")

        lg.info(f"Calculated TSL: Activate @ {act_price:.{price_prec}f}, Trail Distance ≈ {trail_dist:.{price_prec}f}")

        # Call internal helper
        return _set_position_protection(
            exchange, symbol, market_info, position_info, lg,
            stop_loss_price=None,  # TSL overrides fixed SL
            take_profit_price=take_profit_price,
            trailing_stop_distance=trail_dist,
            tsl_activation_price=act_price
        )
    except Exception as e: lg.error(f"Error calculating/setting TSL ({symbol}): {e}", exc_info=True); return False


def set_breakeven_stop_loss(  # Checks conditions, calls internal helper
    exchange: ccxt.Exchange, symbol: str, market_info: dict, position_info: dict,
    config_dict: dict[str, Any], analyzer: TradingAnalyzer, logger: logging.Logger
) -> bool:
    lg = logger; config = config_dict
    if not config.get("enable_break_even", False): return False  # Disabled
    # Check if TSL is already active - BE should usually only trigger before TSL
    if active_protection_state.get("trailingStopValue") is not None:
        lg.debug("BE check skipped: Trailing Stop is already active."); return False

    try:  # Get position and analysis data
        entry_price = position_info.get('entryPrice')
        side = position_info.get('side')
        current_sl = active_protection_state.get('stopLossPrice')  # Get current SL from global state
        current_atr = analyzer.indicator_values.get("ATR")
        current_price = analyzer.indicator_values.get("Close")  # Current close price
        if not all(isinstance(v, Decimal) and v > 0 for v in [entry_price, current_atr, current_price]) or side not in ['long', 'short']:
            raise ValueError(f"Invalid inputs for BE check (entry={entry_price}, atr={current_atr}, price={current_price}, side={side})")
    except Exception as e: lg.error(f"BE Check Error ({symbol}): Invalid data - {e}"); return False

    try:  # Check trigger condition
        trigger_atr_mult = Decimal(str(config.get("break_even_trigger_atr_multiple", "1.0")))
        profit_target_points = current_atr * trigger_atr_mult
        current_profit_points = (current_price - entry_price) if side == 'long' else (entry_price - current_price)
        price_prec = analyzer.get_price_precision()
        lg.debug(f"BE Check ({symbol}): Profit Pts={current_profit_points:.{price_prec}f}, Target Pts={profit_target_points:.{price_prec}f}")

        if current_profit_points >= profit_target_points:
            # Calculate BE stop price
            offset_ticks = int(config.get("break_even_offset_ticks", 3))
            min_tick = analyzer.get_min_tick_size()
            tick_offset = min_tick * offset_ticks
            be_stop_price = (entry_price + tick_offset if side == 'long' else entry_price - tick_offset)
            be_stop_price = be_stop_price.quantize(min_tick, rounding=ROUND_UP if side == 'long' else ROUND_DOWN)
            if be_stop_price <= 0: raise ValueError("Invalid BE stop price calculated")

            # Determine if update needed
            update_needed = False
            if current_sl is None: update_needed = True; lg.info("BE trigger: No current SL.")
            elif side == 'long' and be_stop_price > current_sl: update_needed = True; lg.info(f"BE trigger: Moving SL UP {current_sl} -> {be_stop_price}.")
            elif side == 'short' and be_stop_price < current_sl: update_needed = True; lg.info(f"BE trigger: Moving SL DOWN {current_sl} -> {be_stop_price}.")
            else: lg.debug("BE Triggered, but current SL already better."); return False  # No action needed

            if update_needed:
                lg.warning(f"{NEON_PURPLE}*** Moving SL to Break-Even ({symbol}) at {be_stop_price} ***{RESET}")
                existing_tp = active_protection_state.get('takeProfitPrice')  # Preserve existing TP
                success = _set_position_protection(
                    exchange, symbol, market_info, position_info, lg,
                    stop_loss_price=be_stop_price, take_profit_price=existing_tp
                )
                if success: lg.info(f"{NEON_GREEN}BE SL set successfully.{RESET}")
                else: lg.error(f"{NEON_RED}Failed to set BE SL.{RESET}")
                return success  # Return result of setting protection
        else: lg.debug("BE trigger condition not met."); return False  # Condition not met
    except Exception as e: lg.error(f"Error during BE check/set ({symbol}): {e}", exc_info=True); return False


# --- Status Panel ---
def print_status_panel(  # Uses tabulate for better formatting
    cycle: int, timestamp: datetime | None, analyzer: TradingAnalyzer,
    position: dict | None, equity: Decimal | None, signal: str, market_info: dict
) -> None:
    global active_protection_state
    Fore.MAGENTA + Style.BRIGHT; section_color = Fore.CYAN; value_color = Fore.WHITE
    reset = Style.RESET_ALL

    timestamp.strftime('%Y-%m-%d %H:%M:%S %Z') if timestamp else f"{Fore.YELLOW}N/A"
    quote = market_info.get('settle', 'QUOTE')
    f"{equity:.4f} {quote}" if isinstance(equity, Decimal) else f"{Fore.YELLOW}N/A"

    # Market & Indicators
    indicators = analyzer.indicator_values if analyzer else {}
    price_prec = analyzer.get_price_precision() if analyzer else 4
    price = indicators.get("Close")  # Decimal or None
    price_str = f"{price:.{price_prec}f}" if isinstance(price, Decimal) else f"{Fore.YELLOW}N/A"
    atr = indicators.get('ATR')
    atr_str = f"{atr:.{price_prec + 1}f}" if isinstance(atr, Decimal) else f"{Fore.YELLOW}N/A"
    trend_ema_key = f"EMA_{CONFIG.indicator_trend_ema}"  # Construct key
    trend_ema = indicators.get(trend_ema_key)
    trend_ema_str = f"{trend_ema:.{price_prec}f}" if isinstance(trend_ema, Decimal) else f"{Fore.YELLOW}N/A"

    price_color, trend_desc = Fore.WHITE, f"{Fore.YELLOW}Trend N/A"
    if isinstance(price, Decimal) and isinstance(trend_ema, Decimal):
        if price > trend_ema: price_color, trend_desc = Fore.GREEN, f"{Fore.GREEN}(Above Trend)"
        elif price < trend_ema: price_color, trend_desc = Fore.RED, f"{Fore.RED}(Below Trend)"
        else: price_color, trend_desc = Fore.YELLOW, f"{Fore.YELLOW}(At Trend)"

    stoch_k = indicators.get('StochRSI_K'); stoch_d = indicators.get('StochRSI_D')  # Floats or None
    stoch_k_str = f"{stoch_k:.2f}" if pd.notna(stoch_k) else f"{Fore.YELLOW}N/A"
    stoch_d_str = f"{stoch_d:.2f}" if pd.notna(stoch_d) else f"{Fore.YELLOW}N/A"
    stoch_color, stoch_desc = Fore.YELLOW, f"{Fore.YELLOW}Stoch N/A"
    if pd.notna(stoch_k):
         if stoch_k < 25: stoch_color, stoch_desc = Fore.GREEN, f"{Fore.GREEN}Oversold"
         elif stoch_k > 75: stoch_color, stoch_desc = Fore.RED, f"{Fore.RED}Overbought"
         else: stoch_color, stoch_desc = Fore.YELLOW, f"{Fore.YELLOW}Neutral"

    [
        [section_color + "Market", value_color + market_info.get('symbol', 'N/A'), f"{price_color}{price_str}"],
        [section_color + f"Trend EMA ({CONFIG.indicator_trend_ema})", f"{value_color}{trend_ema_str}", trend_desc],
        [section_color + f"ATR ({CONFIG.indicator_atr_period})", f"{value_color}{atr_str}", ""],
        [section_color + "Stoch K/D", f"{stoch_color}{stoch_k_str} / {stoch_d_str}", stoch_desc],
    ]

    # Positions & Protection
    pos_side = position.get('side', 'None').upper() if position else 'None'
    pos_size = position.get('size', Decimal('0')) if position else Decimal('0')
    pos_entry = position.get('entryPrice') if position else None  # Decimal or None
    pos_pnl = position.get('unrealizedPnl') if position else None  # Decimal or None
    pos_liq = position.get('liquidationPrice') if position else None  # Decimal or None

    sl = active_protection_state['stopLossPrice']
    tp = active_protection_state['takeProfitPrice']
    tsl_dist = active_protection_state['trailingStopValue']
    tsl_act = active_protection_state['trailingStopActivationPrice']

    def fmt_prot(val, prefix="") -> str: return f"{prefix}{val:.{price_prec}f}" if val else f"{Fore.RED}NONE{reset}"
    tsl_status = f"{Fore.GREEN}ACTIVE{reset} (Dist:{fmt_prot(tsl_dist, '')}, Act:{fmt_prot(tsl_act)})" if tsl_dist else f"{Fore.YELLOW}INACTIVE{reset}"

    pos_size_str = f"{pos_size:.8f}".rstrip('0').rstrip('.') if pos_size != Decimal(0) else "0"
    pos_entry_str = f"{pos_entry:.{price_prec}f}" if pos_entry else "-"
    pnl_color = Fore.GREEN if pos_pnl is not None and pos_pnl >= 0 else Fore.RED
    pos_pnl_str = f"{pnl_color}{pos_pnl:+.{price_prec}f}{value_color}" if pos_pnl is not None else "-"
    pos_liq_str = f"{Fore.RED}{pos_liq:.{price_prec}f}{value_color}" if pos_liq else "-"

    [
        [section_color + "Position Status", value_color + pos_side],
        [section_color + "Size", value_color + pos_size_str],
        [section_color + "Entry", value_color + pos_entry_str],
        [section_color + "Unrealized PnL", pos_pnl_str],
        [section_color + "Liq. Price", pos_liq_str],
        [section_color + "Fixed SL", fmt_prot(sl)],
        [section_color + "Fixed TP", fmt_prot(tp)],
        [section_color + "Trailing SL", tsl_status],
    ]

    # Signals
    Fore.GREEN + Style.BRIGHT if signal == "BUY" else Fore.WHITE
    Fore.RED + Style.BRIGHT if signal == "SELL" else Fore.WHITE


# --- Main Trading Cycle ---
def trading_spell_cycle(cycle_count: int) -> None:
    """Executes one cycle of the trading spell: data -> analysis -> state -> signal -> manage/trade."""
    global EXCHANGE, MARKET_INFO, CONFIG  # Use globals

    logger.info(Fore.MAGENTA + Style.BRIGHT + f"\n--- Starting Cycle {cycle_count} ---")
    start_time = time.monotonic()
    if not EXCHANGE or not MARKET_INFO: logger.critical("Exchange/Market Info missing!"); return

    symbol = CONFIG.symbol
    ccxt_interval = CONFIG.ccxt_interval

    # 1. Fetch Data
    df = fetch_market_data(symbol, ccxt_interval, CONFIG.ohlcv_limit)
    if df is None or df.empty or len(df) < 50: logger.error("Market data failed. Skip cycle."); return
    df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now(TIMEZONE)

    # 2. Analyze Data
    analyzer = TradingAnalyzer(df.copy(), logger, CONFIG.__dict__, MARKET_INFO)
    if not analyzer.indicator_values: logger.error("Indicator calc failed. Skip cycle."); return
    current_price = analyzer.indicator_values.get("Close")  # Decimal
    current_atr = analyzer.indicator_values.get("ATR")  # Decimal
    if not isinstance(current_price, Decimal) or current_price <= 0: logger.error("Invalid price. Skip cycle."); return

    # Fetch OB if needed
    orderbook_data = None
    # Placeholder: Fetch if indicator enabled/weighted (logic omitted for brevity)

    # 3. Fetch State
    current_equity = fetch_balance()  # Uses CONFIG.quote_currency
    current_position = get_current_position(symbol)  # Includes protection state
    position_snapshot = current_position.copy() if current_position else None  # Snapshot for panel

    # 4. Generate Signal
    signal_action = analyzer.generate_trading_signal(current_price, orderbook_data)

    # 5. Display Status (BEFORE acting)
    print_status_panel(cycle_count, datetime.now(TIMEZONE), current_price, analyzer, position_snapshot, current_equity, signal_action, MARKET_INFO)

    # 6. Execute Logic if Trading Enabled
    if not CONFIG.enable_trading: logger.info("Trading disabled."); return

    # --- Position Management & Entry ---
    if current_position:
        pos_side = current_position.get('side')
        pos_size = current_position.get('size')
        # a. Check Exit Signal
        if (pos_side == 'long' and signal_action == "SELL") or (pos_side == 'short' and signal_action == "BUY"):
            logger.warning(f"*** EXIT Signal ({signal_action}) for {pos_side} position. Closing... ***")
            if pos_size: place_trade(EXCHANGE, symbol, "SELL" if pos_side == 'long' else "BUY", abs(pos_size), MARKET_INFO, logger, 'market', reduce_only=True)
            return  # Exit cycle after closing attempt
        # b. Check Time Exit
        if CONFIG.time_based_exit_minutes and current_position.get('timestamp_ms'):
            elapsed_mins = (time.time() * 1000 - current_position['timestamp_ms']) / 60000
            if elapsed_mins >= CONFIG.time_based_exit_minutes:
                logger.warning(f"*** TIME EXIT triggered ({elapsed_mins:.1f} >= {CONFIG.time_based_exit_minutes}m). Closing... ***")
                if pos_size: place_trade(EXCHANGE, symbol, "SELL" if pos_side == 'long' else "BUY", abs(pos_size), MARKET_INFO, logger, 'market', reduce_only=True)
                return  # Exit cycle
        # c. Manage Protection (BE first, then TSL if applicable)
        if not active_protection_state.get("trailingStopValue"):  # Only check BE if TSL inactive
            set_breakeven_stop_loss(EXCHANGE, symbol, MARKET_INFO, current_position, CONFIG.__dict__, analyzer, logger)
        # Attempt to set TSL (function checks config and if already active)
        # Pass current TP from fetched state to preserve it if TSL activates
        set_trailing_stop_loss(EXCHANGE, symbol, MARKET_INFO, current_position, CONFIG.__dict__, analyzer, logger, active_protection_state.get("takeProfitPrice"))

    elif signal_action in ["BUY", "SELL"]:  # No Position & Entry Signal
        logger.info(f"*** {signal_action} Signal & No Position: Initiating Entry for {symbol} ***")
        if current_equity is None or current_equity <= 0: logger.error("Cannot enter: Invalid equity."); return
        if not isinstance(current_atr, Decimal) or current_atr <= 0: logger.error("Cannot enter: Invalid ATR."); return

        # Set Leverage
        if MARKET_INFO.get('is_contract', False):
            if not set_leverage_ccxt(EXCHANGE, symbol, CONFIG.leverage, MARKET_INFO, logger): logger.error("Entry aborted: Leverage set failed."); return

        # Calculate SL for sizing
        _, _, initial_sl_price = analyzer.calculate_entry_tp_sl(current_price, signal_action)
        if not initial_sl_price: logger.error("Entry aborted: Initial SL calc failed."); return

        # Calculate Size
        pos_size = calculate_position_size(current_equity, CONFIG.risk_percentage, initial_sl_price, current_price, MARKET_INFO, EXCHANGE, logger)
        if not pos_size: logger.error("Entry aborted: Sizing failed."); return

        # Place Entry Order
        entry_order = place_trade(EXCHANGE, symbol, signal_action, pos_size, MARKET_INFO, logger, order_type='market')

        if entry_order and entry_order.get('id'):
            order_id = entry_order['id']
            logger.info(f"Entry order {order_id} placed. Waiting {CONFIG.order_check_delay_seconds}s for confirmation...")
            time.sleep(CONFIG.order_check_delay_seconds)

            # Confirm Position & Set Protection
            confirmed_position = get_current_position(symbol)
            if confirmed_position:
                entry_price_actual = confirmed_position.get('entryPrice') or current_price  # Use actual or estimate
                logger.info(f"Position Confirmed! Entry: ~{entry_price_actual:.{analyzer.get_price_precision()}f}")
                _, tp_final, sl_final = analyzer.calculate_entry_tp_sl(entry_price_actual, signal_action)

                protection_set = False
                if CONFIG.enable_trailing_stop:
                     logger.info(f"Setting Initial Trailing Stop Loss (TP target: {tp_final})...")
                     protection_set = set_trailing_stop_loss(EXCHANGE, symbol, MARKET_INFO, confirmed_position, CONFIG.__dict__, analyzer, logger, tp_final)
                else:
                     logger.info(f"Setting Initial Fixed SL ({sl_final}) and TP ({tp_final})...")
                     if sl_final or tp_final: protection_set = _set_position_protection(EXCHANGE, symbol, MARKET_INFO, confirmed_position, logger, sl_final, tp_final)
                     else: logger.warning("Initial SL/TP invalid, no fixed protection set.")

                if protection_set: logger.info(f"{NEON_GREEN}=== ENTRY & INITIAL PROTECTION COMPLETE ({symbol} {signal_action}) ===")
                else: logger.error(f"{NEON_RED}=== ENTRY OK BUT PROTECTION FAILED! ({symbol} {signal_action}) ==="); termux_notify("PROTECTION FAILED!", f"{symbol} entry OK, SL/TP/TSL set failed!")
            else:
                logger.error(f"{NEON_RED}Entry order {order_id} placed, but FAILED TO CONFIRM position! Manual check needed!{RESET}")
                termux_notify("CONFIRM FAILED!", f"{symbol} entry {order_id} status unknown!")
        else: logger.error(f"{NEON_RED}=== ENTRY FAILED ({symbol} {signal_action}). Order placement issue. ===")

    else:  # HOLD signal, no position
        logger.info(f"Signal is HOLD and no position for {symbol}. No action.")

    cycle_end_time = time.monotonic()
    logger.info(f"---== Cycle {cycle_count} End ({symbol}, Duration: {cycle_end_time - start_time:.2f}s) ==---")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("*** Pyrmethus Bot v4.0.0 Initializing ***")

    EXCHANGE = initialize_exchange(logger)
    if not EXCHANGE: logger.critical("Exchange init failed. Exiting."); sys.exit(1)

    MARKET_INFO = get_market_info(CONFIG.symbol, logger)
    if not MARKET_INFO: logger.critical(f"Market info load failed for {CONFIG.symbol}. Exiting."); sys.exit(1)

    CONFIG._log_key_parameters()  # Log effective config

    if CONFIG.enable_trading:
         logger.warning(f"{Fore.YELLOW}!!! LIVE TRADING ENABLED !!!{Style.RESET_ALL}")
         if CONFIG.use_testnet: logger.warning(f"{Fore.YELLOW}Using SANDBOX (Testnet).{Style.RESET_ALL}")
         else: logger.warning(f"{Fore.RED}!!! USING REAL MONEY !!!{Style.RESET_ALL}"); time.sleep(5)
    else: logger.info(f"{Fore.YELLOW}Trading is DISABLED. Analysis only.{Style.RESET_ALL}")

    logger.info(f"Starting trading cycles for {CONFIG.symbol}...")

    cycle = 0
    try:
        while True:
            cycle += 1
            try: trading_spell_cycle(cycle)
            except Exception as cycle_err: logger.critical(f"!! UNHANDLED CYCLE {cycle} ERROR !!: {cycle_err}", exc_info=True); time.sleep(60)  # Pause after crash
            logger.info(f"Cycle {cycle} finished. Sleeping {CONFIG.loop_sleep_seconds}s...")
            time.sleep(CONFIG.loop_sleep_seconds)
    except KeyboardInterrupt: logger.warning("\nCtrl+C detected! Shutting down...")
    except Exception as main_err: logger.critical(f"!! FATAL MAIN LOOP ERROR !!: {main_err}", exc_info=True)
    finally:
        logger.info("Closing exchange connection...")
        if EXCHANGE and hasattr(EXCHANGE, 'close'):
             try: EXCHANGE.close()
             except Exception as e: logger.error(f"Error closing connection: {e}")
        logger.info("--- Pyrmethus Bot Deactivated ---")
        logging.shutdown()
