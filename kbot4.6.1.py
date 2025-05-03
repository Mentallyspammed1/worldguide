# -*- coding: utf-8 -*-
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods, too-many-lines
# Note: Some pylint disables are kept where refactoring significantly alters structure or standard patterns (e.g., config attributes).
#       Others like line-too-long, wrong-import-order/position, unnecessary items have been addressed.
"""
Pyrmethus - Termux Trading Spell (v4.5.8 - Enhanced Edition)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in this version (4.5.8):
- Comprehensive code analysis and upgrade based on v4.5.7.
- Improved Readability: PEP8 compliance, clearer names, better comments/docstrings, type hinting.
- Improved Maintainability: Reduced pylint disables, cleaner structure, enhanced configuration validation.
- Improved Efficiency: Minor optimizations in data handling and calculations.
- Improved Error Handling: More specific exceptions, refined retry logic, robust NaN/API response handling.
- Improved Security: Standard practices maintained (API keys via env).
- Modern Python Practices: pathlib, f-strings, logging best practices, enhanced type hinting.
- Added Graceful Shutdown handler.
- Added basic Trade Journaling implementation.
- Added core trading loop logic (`run_bot`, `main`).
- Added `close_position` and `check_and_manage_trailing_stop` methods.
- Refined V5 API interactions (balance parsing, position parsing, protection setting).
- Integrated and refined previous snippets (Config, Retry, Indicator Conversion, V5 Param, Verification).
- Corrected colorama usage and refined console output.
"""

# Standard Library Imports
import csv
import logging
import os
import platform
import signal
import subprocess
import sys
import textwrap
import time
import types # For signal frame type hint
from datetime import datetime, timezone
from decimal import (
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UP,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    getcontext,
)
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, cast

# Third-Party Imports
try:
    import ccxt
    import numpy as np
    import pandas as pd
    import requests
    from colorama import Fore, Style, init as colorama_init
    from dotenv import load_dotenv
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # Initialize Colorama & Rich Console early for potential import errors
    colorama_init(autoreset=True)
    console = Console(log_path=False) # Disable Rich logging interception

except ImportError as e:
    # Attempt basic colorama init for the error message itself if possible
    try:
        colorama_init(autoreset=True)
        missing_pkg = e.name
        print(
            f"{Fore.RED}{Style.BRIGHT}Missing essential spell component: {missing_pkg}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}To conjure it, cast: {Style.BRIGHT}pip install {missing_pkg}{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}Or, to ensure all scrolls are present, cast:")
        COMMON_PACKAGES = [
            "ccxt", "python-dotenv", "pandas", "numpy", "rich", "colorama", "requests"
        ]
        if os.getenv("TERMUX_VERSION"):
            print(
                f"{Style.BRIGHT}pkg install python python-pandas python-numpy && pip install {' '.join([p for p in COMMON_PACKAGES if p not in ['pandas', 'numpy']])}{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Note: pandas and numpy often installed via pkg in Termux.{Style.RESET_ALL}")
        else:
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
    except Exception: # Fallback to plain print if colorama init failed
         print(f"Missing essential spell component: {e.name}")
         print(f"To conjure it, cast: pip install {e.name}")
         print("
Or, to ensure all scrolls are present, cast:")
         print(f"pip install ccxt python-dotenv pandas numpy rich colorama requests")
    sys.exit(1)


# --- Constants ---
# Precision & Thresholds
DECIMAL_PRECISION = 50
POSITION_QTY_EPSILON = Decimal("1E-12")  # Threshold for considering a position 'flat'
DEFAULT_PRICE_DP = 4
DEFAULT_AMOUNT_DP = 6

# API & Timing Defaults
DEFAULT_OHLCV_LIMIT = 200
DEFAULT_LOOP_SLEEP = 15 # seconds
DEFAULT_RETRY_DELAY = 3 # seconds
DEFAULT_MAX_RETRIES = 3
TERMUX_NOTIFY_TIMEOUT = 10 # seconds

# Strategy Defaults
DEFAULT_RISK_PERCENT = Decimal("0.01") # 1%
DEFAULT_SL_MULT = Decimal("1.5")
DEFAULT_TP_MULT = Decimal("3.0")
DEFAULT_TSL_ACT_MULT = Decimal("1.0")
DEFAULT_TSL_PERCENT = Decimal("0.5") # 0.5%
DEFAULT_STOCH_OVERSOLD = Decimal("30")
DEFAULT_STOCH_OVERBOUGHT = Decimal("70")
DEFAULT_MIN_ADX = Decimal("20")

# Bybit V5 Specifics
V5_UNIFIED_ACCOUNT_TYPE = "UNIFIED"
V5_HEDGE_MODE_POSITION_IDX = 0 # Default index (0=One-Way, 1=Buy Hedge, 2=Sell Hedge)
V5_TPSL_MODE_FULL = "Full" # Apply SL/TP to entire position
V5_SUCCESS_RETCODE = 0

# File Paths
DEFAULT_JOURNAL_FILE = "pyrmethus_trading_journal.csv"
DEFAULT_ENV_FILE = ".env"

# Set Decimal precision context
getcontext().prec = DECIMAL_PRECISION

# --- Logging Setup ---
# Custom logging level for trade actions
TRADE_LEVEL_NUM = 25  # Between INFO and WARNING
logging.addLevelName(TRADE_LEVEL_NUM, "TRADE")

def trade_log(self: logging.Logger, message: str, *args: Any, **kws: Any) -> None:
    """Logs a message with level TRADE."""
    if self.isEnabledFor(TRADE_LEVEL_NUM):
        # pylint: disable=protected-access
        self._log(TRADE_LEVEL_NUM, message, args, **kws)

# Add the custom level method to the Logger class if it doesn't exist
if not hasattr(logging.Logger, "trade"):
    logging.Logger.trade = trade_log # type: ignore[attr-defined]

# Base logger configuration
logger = logging.getLogger(__name__)
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] (%(filename)s:%(lineno)d) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(log_level)

# Ensure handler is added only once to prevent duplicate logs
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
logger.propagate = False # Prevent passing logs to the root logger


# --- Utility Functions ---
def safe_decimal(
    value: Any, default: Decimal = Decimal("NaN")
) -> Decimal:
    """Safely converts a value to Decimal, handling None, empty strings, and invalid formats."""
    if value is None:
        return default
    try:
        # Convert potential floats or numeric types to string first for precise Decimal
        str_value = str(value).strip()
        if not str_value:  # Handle empty string after stripping
            return default
        # Handle common non-numeric strings that might appear in APIs
        if str_value.lower() in ["nan", "none", "null", ""]:
            return default
        return Decimal(str_value)
    except (InvalidOperation, ValueError, TypeError):
        # logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}") # Optional debug
        return default

def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available."""
    # Check if running in Termux environment more reliably
    if "com.termux" in os.environ.get("PREFIX", ""):
        try:
            # termux-toast expects only the content argument
            result = subprocess.run(
                ["termux-toast", content],
                check=False, # Don't raise exception on non-zero exit
                timeout=TERMUX_NOTIFY_TIMEOUT, # Use constant for timeout
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Log stderr if available, otherwise stdout
                error_output = result.stderr or result.stdout
                logger.warning(
                    f"Termux toast command failed (code {result.returncode}): {error_output.strip()}"
                )
            # logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning(
                "Termux notify failed: 'termux-toast' command not found."
            )
        except subprocess.TimeoutExpired:
            # Log the timeout specifically
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
        except Exception as e:
            logger.warning(f"Termux notify failed unexpectedly: {e}")
    # else: logger.debug("Not in Termux, skipping notification.") # Optional debug

def fetch_with_retries(
    fetch_function: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: int = DEFAULT_RETRY_DELAY,
    retry_on_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
        ccxt.NetworkError, ccxt.RateLimitExceeded, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout, # Added ReadTimeout
    ),
    fatal_exceptions: Tuple[Type[Exception], ...] = (
        ccxt.AuthenticationError, ccxt.PermissionDenied
    ),
    fail_fast_exceptions: Tuple[Type[Exception], ...] = (
         ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound
    ),
    **kwargs: Any,
) -> Any:
    """Wraps a function call with enhanced retry logic and error handling."""
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1):
        try:
            result = fetch_function(*args, **kwargs)
            # Log success after previous failures
            if attempt > 0:
                logger.info(f"{Fore.GREEN}{Style.BRIGHT}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1} after previous failures.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting immediately.{Style.RESET_ALL}", exc_info=False)
            raise e # Re-raise critical error
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying.{Style.RESET_ALL}")
            last_exception = e
            break # Break loop, don't retry
        except retry_on_exceptions as e:
            last_exception = e
            retry_msg = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {str(e)[:150]}.{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(f"{retry_msg} Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name} after retryable error. Last error: {e}{Style.RESET_ALL}")
                # Loop ends, last_exception raised below
        except ccxt.ExchangeError as e: # Catch other exchange errors
            last_exception = e
            logger.error(f"{Fore.RED}Exchange error during {func_name}: {e}{Style.RESET_ALL}")
            # Decide if specific ExchangeErrors are retryable - here we retry generic ones
            if attempt < max_retries:
                logger.warning(f"{Fore.YELLOW}Retrying generic exchange error in {delay_seconds}s...{Style.RESET_ALL}")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries reached after exchange error for {func_name}.{Style.RESET_ALL}")
                break
        except Exception as e: # Catch truly unexpected errors
            last_exception = e
            logger.error(f"{Fore.RED}Unexpected error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True)
            # For unexpected errors, maybe retry once? Or fail fast? Current: Fail fast.
            break # Don't retry unknown errors

    # If loop finished without returning, raise the last captured exception
    if last_exception:
        raise last_exception
    else:
        # This path should ideally not be hit if logic is correct
        raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without specific exception.")


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from .env."""

    # pylint: disable=too-many-instance-attributes # Acceptable for a config object
    def __init__(self, env_file: Union[str, Path] = DEFAULT_ENV_FILE):
        logger.debug(f"Loading configuration from environment variables / {env_file}...")
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True) # Override allows env vars to take precedence
            logger.info(f"Loaded configuration from {env_path}")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying on system environment variables.")

        # Core Trading Parameters
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env(
            "MARKET_TYPE",
            "linear",
            Style.DIM,
            allowed_values=["linear", "inverse"], # Removed 'swap' as it maps to linear/inverse
        ).lower()
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)

        # Financial Parameters (Decimal)
        self.risk_percentage: Decimal = self._get_env(
            "RISK_PERCENTAGE",
            DEFAULT_RISK_PERCENT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.00001"), # Allow very small risk
            max_val=Decimal("0.5"), # Cap risk at 50%
        )
        self.sl_atr_multiplier: Decimal = self._get_env(
            "SL_ATR_MULTIPLIER",
            DEFAULT_SL_MULT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )
        self.tp_atr_multiplier: Decimal = self._get_env(
            "TP_ATR_MULTIPLIER",
            DEFAULT_TP_MULT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.0"), # Allow TP=0 to disable ATR-based TP
            max_val=Decimal("50.0"),
        )
        self.tsl_activation_atr_multiplier: Decimal = self._get_env(
            "TSL_ACTIVATION_ATR_MULTIPLIER",
            DEFAULT_TSL_ACT_MULT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.1"),
            max_val=Decimal("20.0"),
        )
        self.trailing_stop_percent: Decimal = self._get_env(
            "TRAILING_STOP_PERCENT",
            DEFAULT_TSL_PERCENT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.01"), # Minimum 0.01% TSL
            max_val=Decimal("10.0"), # Maximum 10% TSL
        )

        # V5 Position Stop Parameters
        self.sl_trigger_by: str = self._get_env(
            "SL_TRIGGER_BY",
            "LastPrice",
            Style.DIM,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        self.tsl_trigger_by: str = self._get_env(
            "TSL_TRIGGER_BY",
            "LastPrice",
            Style.DIM,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        # Hedge mode position index (0: One-Way, 1: Buy Hedge, 2: Sell Hedge)
        self.position_idx: int = self._get_env(
            "POSITION_IDX",
            V5_HEDGE_MODE_POSITION_IDX,
            Style.DIM,
            cast_type=int,
            allowed_values=[0, 1, 2],
        )

        # Indicator Periods (int)
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)

        # Signal Logic Thresholds (Decimal)
        self.stoch_oversold_threshold: Decimal = self._get_env("STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")) # 0 disables filter
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90"))

        # API Keys (Secrets)
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)

        # Operational Parameters
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1) # Allow 1s min sleep
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5) # Timeout waiting for fill confirmation (used in verification)
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10) # Allow 0 retries
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)

        # Journaling
        self.journal_file_path: Path = Path(self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM))
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)

        # Final Checks
        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found in environment. Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        self._validate_config()
        logger.debug("Configuration loaded and validated successfully.")

    def _determine_v5_category(self) -> str:
        """Determines the Bybit V5 API category based on symbol and market type."""
        try:
            # Symbol format like BASE/QUOTE:SETTLE (e.g., BTC/USDT:USDT)
            if ":" not in self.symbol:
                raise ValueError(
                    f"Symbol format '{self.symbol}' must include settle currency (e.g., BTC/USDT:USDT)"
                )

            # V5 categories: linear, inverse, spot, option
            if self.market_type == "inverse":
                category = "inverse" # e.g., BTC/USD:BTC
            elif self.market_type == "linear":
                # Linear includes USDT/USDC perpetuals/futures
                category = "linear"
            else:
                # This shouldn't happen due to _get_env validation, but as a safeguard:
                raise ValueError(f"Unsupported MARKET_TYPE '{self.market_type}' for V5 category determination.")

            logger.info(
                f"Determined Bybit V5 API category: '{category}' for symbol '{self.symbol}' and type '{self.market_type}'"
            )
            return category
        except ValueError as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Could not determine V5 category: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _validate_config(self) -> None:
        """Performs post-load validation of configuration parameters."""
        if self.fast_ema_period >= self.slow_ema_period:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Validation failed: FAST_EMA ({self.fast_ema_period}) must be < SLOW_EMA ({self.slow_ema_period}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.trend_ema_period <= self.slow_ema_period:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TREND_EMA ({self.trend_ema_period}) <= SLOW_EMA ({self.slow_ema_period}). Trend filter might lag short-term EMA signals.{Style.RESET_ALL}"
            )
        if self.stoch_oversold_threshold >= self.stoch_overbought_threshold:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Validation failed: STOCH_OVERSOLD ({self.stoch_oversold_threshold.normalize()}) must be < STOCH_OVERBOUGHT ({self.stoch_overbought_threshold.normalize()}). Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        if self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is reached.{Style.RESET_ALL}"
            )
        # Check TP vs SL only if TP is enabled (multiplier > 0)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). This implies a poor Risk:Reward setup (< 1:1).{Style.RESET_ALL}"
            )
        # Ensure journal file directory exists if journaling enabled
        if self.enable_journaling:
            try:
                self.journal_file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory for journal file {self.journal_file_path}: {e}. Journaling disabled.")
                self.enable_journaling = False

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Helper to cast string value to target type, returning default on failure."""
        val_to_cast = value_str.strip()
        if not val_to_cast: # Handle empty string after strip
            logger.warning(f"Empty value string for {key}. Using default '{default}'.")
            return default
        try:
            if cast_type == bool:
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                # Check for common non-numeric strings before casting
                if val_to_cast.lower() in ["nan", "none", "null"]:
                    raise ValueError("Non-numeric string cannot be cast to Decimal")
                return Decimal(val_to_cast)
            elif cast_type == int:
                # Use Decimal intermediary for robustness (handles "10.0")
                dec_val = Decimal(val_to_cast)
                # Check if it has a fractional part or is non-finite
                if dec_val.as_tuple().exponent < 0 or not dec_val.is_finite():
                    raise ValueError("Non-integer Decimal cannot be cast to int")
                return int(dec_val)
            # Add other specific casts if needed (e.g., float)
            else: # Includes str type
                return cast_type(val_to_cast) # Use constructor directly
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str}' -> {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}"
            )
            return default

    def _validate_value(self, key: str, value: Any, min_val: Optional[Union[int, float, Decimal]], max_val: Optional[Union[int, float, Decimal]], allowed_values: Optional[List[Any]]) -> bool:
        """Helper to validate value against constraints. Logs and returns False on failure."""
        # Type check for numeric comparison
        is_numeric = isinstance(value, (int, float, Decimal))
        # Ensure Decimal values are finite for comparisons
        if isinstance(value, Decimal) and not value.is_finite():
             logger.error(f"Validation failed for {key}: Non-finite Decimal value '{value}'.")
             return False

        if (min_val is not None or max_val is not None) and not is_numeric:
            logger.error(f"Validation failed for {key}: Non-numeric value '{value}' ({type(value).__name__}) cannot be compared with min/max.")
            return False

        # Min/Max checks
        if min_val is not None and is_numeric and value < min_val:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}Validation failed for {key}: Value '{value}' < minimum '{min_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1) # Critical failure
        if max_val is not None and is_numeric and value > max_val:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}Validation failed for {key}: Value '{value}' > maximum '{max_val}'. Halting.{Style.RESET_ALL}")
            sys.exit(1) # Critical failure

        # Allowed values check
        if allowed_values:
            # Handle case-insensitive string comparison
            comp_value = str(value).lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Validation failed for {key}: Invalid value '{value}'. Allowed: {allowed_values}.{Style.RESET_ALL}")
                return False

        return True # All checks passed

    def _get_env(
        self,
        key: str,
        default: Any,
        color: str,
        cast_type: Type = str,
        min_val: Optional[Union[int, float, Decimal]] = None,
        max_val: Optional[Union[int, float, Decimal]] = None,
        allowed_values: Optional[List[Any]] = None,
        is_secret: bool = False
    ) -> Any:
        """Streamlined fetching, casting, validating, and defaulting environment variables."""
        value_str = os.getenv(key)
        source = "environment"
        use_default = False

        if value_str is None or value_str.strip() == "":
            if default is None and not is_secret: # Allow secrets to be None initially, checked later
                 log_msg = f"{Fore.RED}{Style.BRIGHT}Required configuration '{key}' not found and no default provided. Halting.{Style.RESET_ALL}"
                 logger.critical(log_msg)
                 sys.exit(1)
            elif default is None and is_secret:
                 # Secret is missing, will be checked finally
                 value_str_to_process = "" # Process empty string for secret
                 log_value_display = "[Not Set]"
                 source = "missing"
            else:
                 use_default = True
                 value_str_to_process = str(default) # Use string representation of default for casting/logging
                 source = f"default ({default})"
                 log_value_display = default # Display original default
        else:
            value_str_to_process = value_str
            log_value_display = "****" if is_secret else value_str_to_process

        # Log the found/default value
        log_method = logger.warning if use_default or (is_secret and value_str is None) else logger.info
        log_method(f"{color}Using {key}: {log_value_display} (from {source}){Style.RESET_ALL}")

        # Handle case where secret is missing (value_str_to_process is "")
        if is_secret and not value_str_to_process:
             return None # Return None for missing secrets, final check happens in __init__

        # Attempt to cast the value string (from env or default)
        casted_value = self._cast_value(key, value_str_to_process, cast_type, default)

        # Validate the casted value
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            # Validation failed (min/max would have exited). This usually means allowed_values failed or type error.
            # Revert to the original default value provided to the function.
            logger.warning(
                f"{color}Reverting {key} to original default '{default}' due to validation failure of value '{casted_value}'.{Style.RESET_ALL}"
            )
            casted_value = default # Use the original default value

            # Critical: Re-validate the original default value itself
            if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                logger.critical(
                    f"{Fore.RED}{Style.BRIGHT}FATAL: Default value '{default}' for {key} failed validation. Halting.{Style.RESET_ALL}"
                )
                sys.exit(1)

        return casted_value


# --- Exchange Manager Class ---
class ExchangeManager:
    """Handles CCXT exchange interactions, data fetching, and formatting."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange()
        if self.exchange: # Load markets only if exchange initialized
             self.market_info = self._load_market_info()
        # Critical failure already logged in _initialize_exchange or _load_market_info, exit handled there

    def _initialize_exchange(self) -> None:
        """Initializes the CCXT exchange instance."""
        logger.info(f"Initializing Bybit exchange interface (V5 {self.config.market_type})...")
        try:
            exchange_params = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    "defaultType": self.config.market_type, # Set default market type
                    "adjustForTimeDifference": True, # Auto-adjust time difference
                    "recvWindow": 10000, # Optional: Increase recvWindow if needed
                    "brokerId": "PyrmethusNeonV5", # Custom ID for tracking via Bybit referrals
                    "createMarketBuyOrderRequiresPrice": False, # V5 specific option
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled default
                },
                # Example: Handle sandbox endpoint if needed via environment variable
                # 'urls': {'api': 'https://api-testnet.bybit.com'} if os.getenv("USE_SANDBOX", "false").lower() == "true" else {}
            }

            self.exchange = ccxt.bybit(exchange_params)
            # Test connectivity (optional but recommended)
            logger.debug("Testing exchange connection...")
            self.exchange.fetch_time()
            logger.info(
                f"{Fore.GREEN}{Style.BRIGHT}Bybit V5 interface initialized and connection tested successfully.{Style.RESET_ALL}"
            )

        except ccxt.AuthenticationError as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Authentication failed: {e}. Check API keys/permissions. Halting.{Style.RESET_ALL}",
                exc_info=False, # Don't need full traceback for auth errors
            )
            sys.exit(1)
        except (ccxt.NetworkError, requests.exceptions.RequestException) as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Network error initializing exchange: {e}. Check connection. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and caches market information for the configured symbol."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot load market info.")
            # Initialization should have already exited if failed critically
            return None
        try:
            logger.info(f"Loading market info for {self.config.symbol}...")
            # Force reload to ensure up-to-date info
            self.exchange.load_markets(True)
            market = self.exchange.market(self.config.symbol)
            if not market:
                # This shouldn't happen if load_markets succeeded, but check anyway
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found on exchange after loading markets."
                )

            # Safely extract precision and convert to integer decimal places (dp)
            amount_precision_raw = market.get("precision", {}).get("amount")
            price_precision_raw = market.get("precision", {}).get("price")

            # Precision might be step size (e.g., 0.001) or decimal places (e.g., 3)
            # We want integer decimal places (dp)
            def get_dp_from_precision(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val)
                if prec_dec.is_nan(): return default_dp
                if prec_dec == 0: return 0 # Handle case where precision is exactly 0

                # Check if it's likely a step size (e.g., 0.01, 1e-2)
                if 0 < prec_dec < 1:
                     exponent = prec_dec.normalize().as_tuple().exponent
                     # Ensure exponent is integer before abs()
                     return abs(int(exponent)) if isinstance(exponent, int) else default_dp
                # Check if it's likely number of decimal places directly (integer >= 0)
                elif prec_dec >= 1 and prec_dec.as_tuple().exponent >= 0:
                    try:
                        return int(prec_dec)
                    except (ValueError, TypeError):
                        return default_dp
                # Handle other cases (e.g., precision > 1 but not integer like 1.5?) - use default
                else:
                    logger.warning(f"Unexpected precision format '{precision_val}', using default {default_dp}dp.")
                    return default_dp


            amount_dp = get_dp_from_precision(amount_precision_raw, DEFAULT_AMOUNT_DP)
            price_dp = get_dp_from_precision(price_precision_raw, DEFAULT_PRICE_DP)

            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}

            # Store minimum tick size as Decimal for calculations
            # Tick size is the smallest price increment (usually price precision)
            market["tick_size"] = Decimal("1e-" + str(price_dp))

            # Store minimum order size (base currency amount) as Decimal
            min_amount_raw = market.get("limits", {}).get("amount", {}).get("min")
            market["min_order_size"] = safe_decimal(min_amount_raw, default=Decimal("NaN"))

            # Store contract size (important for inverse contracts, usually 1 for linear)
            market["contract_size"] = safe_decimal(market.get("contractSize", "1"), default=Decimal("1"))

            min_amt_str = (
                market["min_order_size"].normalize()
                if not market["min_order_size"].is_nan()
                else "[dim]N/A[/]" # Use Rich markup for display if needed later
            )
            contract_size_str = market["contract_size"].normalize()

            logger.info(
                f"Market info loaded: ID={market.get('id')}, "
                f"Precision(AmtDP={amount_dp}, PriceDP={price_dp}), "
                f"Limits(MinAmt={min_amt_str}), "
                f"ContractSize={contract_size_str}"
            )
            return market
        except (ccxt.ExchangeError, KeyError, ValueError, TypeError, Exception) as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Failed to load or parse market info for {self.config.symbol}: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def get_market_precision(self, precision_type: str) -> int:
        """Gets the number of decimal places for 'price' or 'amount'."""
        default_dp = DEFAULT_PRICE_DP if precision_type == "price" else DEFAULT_AMOUNT_DP
        if self.market_info and "precision_dp" in self.market_info:
            return self.market_info["precision_dp"].get(precision_type, default_dp)
        return default_dp

    def format_price(
        self, price: Union[Decimal, str, float, int]
    ) -> str:
        """Formats price according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            return "NaN" # Return NaN string if input was bad

        precision = self.get_market_precision("price")

        try:
            # Quantizer is 10^-precision (e.g., 0.01 for 2 dp)
            quantizer = Decimal("1e-" + str(precision))
            # ROUND_HALF_EVEN is standard for financial rounding
            formatted_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            # Ensure the string representation matches the precision exactly
            return f"{formatted_price:.{precision}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting price {price_decimal} to {precision}dp: {e}")
             return "ERR" # Indicate formatting error

    def format_amount(
        self,
        amount: Union[Decimal, str, float, int],
        rounding_mode=ROUND_DOWN, # Default to ROUND_DOWN for order quantities
    ) -> str:
        """Formats amount (quantity) according to market precision using specified rounding."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            return "NaN" # Return NaN string if input was bad

        precision = self.get_market_precision("amount")

        try:
            quantizer = Decimal("1e-" + str(precision))
            formatted_amount = amount_decimal.quantize(quantizer, rounding=rounding_mode)
            # Ensure the string representation matches the precision exactly
            return f"{formatted_amount:.{precision}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting amount {amount_decimal} to {precision}dp: {e}")
             return "ERR" # Indicate formatting error

    def _format_v5_param(
            self,
            value: Optional[Union[Decimal, str, float, int]],
            param_type: str = "price", # 'price', 'amount', or 'distance' (uses price precision)
            allow_zero: bool = False # Allow "0" string as a valid parameter
        ) -> Optional[str]:
        """Formats a value as a string suitable for Bybit V5 API parameters.
           Returns None if the value is invalid or cannot be formatted positively (unless allow_zero=True).
        """
        if value is None:
            # logger.debug(f"V5 Param Formatting: Input value is None.")
            return None # None input results in None output

        # Convert to Decimal for unified handling, default to NaN if conversion fails
        decimal_value = safe_decimal(value, default=Decimal("NaN"))

        if decimal_value.is_nan():
            logger.warning(f"V5 Param Formatting: Input '{value}' resulted in NaN Decimal.")
            return None # Cannot format NaN

        # Handle zero based on allow_zero flag
        is_zero = decimal_value.is_zero()
        if is_zero and allow_zero:
            # Format "0" using the appropriate precision
            precision = self.get_market_precision("price" if param_type in ["price", "distance"] else "amount")
            # Return "0" potentially with decimal places if required by API (e.g., "0.00")
            formatted_zero = f"{Decimal('0'):.{precision}f}"
            # logger.debug(f"V5 Param Formatting: Formatting zero for {param_type} with {precision}dp: {formatted_zero}")
            return formatted_zero
        elif is_zero and not allow_zero:
            # logger.debug(f"V5 Param Formatting: Input value '{value}' is zero, but zero is not allowed.")
            return None # Zero not allowed, return None
        elif decimal_value < 0:
            logger.warning(f"V5 Param Formatting: Input value '{value}' is negative, which is usually invalid.")
            return None # Negative values usually invalid for price/amount/distance

        # Select appropriate formatter based on type
        formatter: Callable[[Union[Decimal, str, float, int]], str]
        if param_type == "price" or param_type == "distance":
            formatter = self.format_price
        elif param_type == "amount":
            # Use ROUND_DOWN for amounts typically
            formatter = self.format_amount # format_amount default is ROUND_DOWN
        else:
            logger.error(f"V5 Param Formatting: Unknown param_type '{param_type}'.")
            return None

        # Format the positive Decimal value
        formatted_str = formatter(decimal_value)

        # Final check: ensure the formatted string isn't an error/NaN indicator
        if formatted_str in ["ERR", "NaN"]:
            logger.error(f"V5 Param Formatting: Failed to format valid string for '{value}' (type: {param_type}). Result: {formatted_str}")
            return None

        # logger.debug(f"V5 Param Formatted: Input='{value}', Type='{param_type}', Output='{formatted_str}'")
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries and converts to DataFrame."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching {self.config.ohlcv_limit} candles for {self.config.symbol} ({self.config.interval})..."
        )
        try:
            # Pass fetch function and args/kwargs to retry wrapper
            ohlcv = fetch_with_retries(
                self.exchange.fetch_ohlcv, # Pass the method itself
                symbol=self.config.symbol,
                timeframe=self.config.interval,
                limit=self.config.ohlcv_limit,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            if not ohlcv:
                logger.error(f"fetch_ohlcv for {self.config.symbol} returned empty list.")
                return None
            if len(ohlcv) < 10: # Check for suspiciously small amount of data
                 logger.warning(f"Fetched only {len(ohlcv)} candles, which might be insufficient for indicators.")

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            # Convert timestamp to UTC datetime and set as index
            df["timestamp"] = pd.to_datetime(
                df["timestamp"], unit="ms", utc=True
            )
            df.set_index("timestamp", inplace=True)

            # Convert OHLCV columns to Decimal robustly using safe_decimal
            nan_cols = []
            for col in ["open", "high", "low", "close", "volume"]:
                # Use safe_decimal utility via map for efficiency
                df[col] = df[col].map(safe_decimal)
                # Check if any conversion resulted in NaN, which indicates bad data
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     nan_cols.append(col)

            if nan_cols:
                 logger.warning(f"Column(s) '{', '.join(nan_cols)}' contain NaN values after conversion. Check API data.")

            # Drop rows where essential price data is NaN after conversion
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            rows_dropped = initial_len - len(df)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows with NaN OHLC values.")

            if df.empty:
                 logger.error("DataFrame became empty after processing OHLCV data.")
                 return None

            logger.debug(
                f"Fetched and processed {len(df)} candles. Last timestamp: {df.index[-1]}"
            )
            return df
        except Exception as e:
            # Catch exceptions from fetch_with_retries or DataFrame processing
            logger.error(f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency using V5 API."""
        if not self.exchange or not self.market_info:
            logger.error(
                "Exchange or market info not available, cannot fetch balance."
            )
            return None, None

        # Determine the currency to look for in the balance response
        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error(
                "Settle currency not found in market info. Cannot determine balance currency."
            )
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})..."
        )
        try:
            # V5 balance requires accountType. Category might be needed depending on context/endpoint used by CCXT.
            # CCXT fetch_balance usually handles mapping to the correct V5 endpoint.
            params = {
                "accountType": V5_UNIFIED_ACCOUNT_TYPE,
                "coin": settle_currency, # Request specific coin balance if API supports it
            }
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance, # Pass function
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw balance data: {balance_data}") # Verbose

            total_equity = Decimal("NaN")
            available_balance = Decimal("NaN")

            # --- Parse V5 Response Structure (Unified Account) ---
            # CCXT aims to unify the response, but sometimes details are in 'info'.
            # Primary check: Top-level 'free' and 'total' dicts for the settle currency.
            if settle_currency in balance_data.get("total", {}):
                total_equity = safe_decimal(balance_data["total"].get(settle_currency))
            if settle_currency in balance_data.get("free", {}):
                available_balance = safe_decimal(balance_data["free"].get(settle_currency))

            # Secondary check: Look inside `info` for more detailed V5 structure if primary failed.
            # Bybit V5 /v5/account/wallet-balance structure: result.list[0].totalEquity, result.list[0].coin[N].availableToWithdraw
            if (total_equity.is_nan() or available_balance.is_nan()) and "info" in balance_data:
                logger.debug("Parsing balance from 'info' field as fallback...")
                info_result = balance_data.get("info", {}).get("result", {})
                account_list = info_result.get("list", [])
                if account_list and isinstance(account_list, list):
                    # Find the Unified account details (usually the first item for unified)
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)
                    if unified_acc_info:
                        if total_equity.is_nan():
                            total_equity = safe_decimal(unified_acc_info.get("totalEquity"))
                        # Available balance might be totalAvailableBalance or specific coin balance
                        if available_balance.is_nan():
                             available_balance = safe_decimal(unified_acc_info.get("totalAvailableBalance"))

                        # If still NaN, check the specific coin details within the account
                        if available_balance.is_nan() and "coin" in unified_acc_info:
                             coin_list = unified_acc_info.get("coin", [])
                             if coin_list and isinstance(coin_list, list):
                                 settle_coin_info = next((c for c in coin_list if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      # Use availableToWithdraw or walletBalance? availableToWithdraw seems safer.
                                      available_balance = safe_decimal(settle_coin_info.get("availableToWithdraw"))
                                      # If total equity was also NaN, try coin equity
                                      if total_equity.is_nan():
                                          total_equity = safe_decimal(settle_coin_info.get("equity"))

            # Final Validation and Logging
            if total_equity.is_nan():
                logger.error(
                    f"Could not extract valid total equity for {settle_currency}. Balance data might be incomplete or unexpected format. Raw snippet: {str(balance_data)[:500]}"
                )
                # Return available balance if found, otherwise 0
                return None, available_balance if not available_balance.is_nan() else Decimal("0")

            if available_balance.is_nan():
                logger.warning(
                    f"Could not extract valid available balance for {settle_currency}. Defaulting to 0. Check raw balance data if issues persist."
                )
                available_balance = Decimal("0")

            logger.debug(
                f"Balance Fetched: Equity={total_equity.normalize()}, Available={available_balance.normalize()} {settle_currency}"
            )
            return total_equity, available_balance
        except Exception as e:
            logger.error(f"Failed to fetch or parse balance: {e}", exc_info=True)
            return None, None

    def get_current_position(
        self,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetches current position details for the symbol using V5 API.
           Returns a dict like {'long': {...}, 'short': {...}} where only one side
           (or neither if flat) will contain position details.
        """
        if not self.exchange or not self.market_info:
            logger.error(
                "Exchange or market info not available, cannot fetch position."
            )
            return None

        market_id = self.market_info.get("id")
        if not market_id:
            logger.error("Market ID not found in market info. Cannot fetch position.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Category: {self.config.bybit_v5_category}, PosIdx: {self.config.position_idx})..."
        )
        # Initialize structure to return
        positions_dict: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}

        try:
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id, # Filter by symbol server-side if possible
                # "positionIdx": self.config.position_idx, # Filter by index if API supports it (check docs)
            }
            # Use fetch_positions (plural) as it's the standard CCXT method
            # It should return a list, potentially empty or with multiple entries if hedge mode is used differently
            fetched_positions_list = fetch_with_retries(
                self.exchange.fetch_positions, # Pass function
                symbols=[self.config.symbol], # Specify symbol, CCXT might use this
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw position data list: {fetched_positions_list}") # Verbose

            if not fetched_positions_list:
                logger.debug("No position data returned from fetch_positions.")
                return positions_dict # Return empty dict, means flat

            # --- Find the specific position matching symbol AND index from the list ---\
            # V5 /v5/position/list response structure: result.list[] contains positions
            target_pos_info = None
            for pos_ccxt in fetched_positions_list:
                # CCXT standardizes some fields, but V5 specifics are often in 'info'
                info = pos_ccxt.get("info", {})
                pos_symbol_info = info.get("symbol")
                pos_idx_str = info.get("positionIdx") # V5 field name
                try:
                    pos_idx = int(pos_idx_str) if pos_idx_str is not None else -1
                except ValueError:
                    pos_idx = -1 # Handle non-integer index string

                symbol_matches = (pos_symbol_info == market_id)
                index_matches = (pos_idx == self.config.position_idx)

                if symbol_matches and index_matches:
                    target_pos_info = info
                    logger.debug(f"Found matching position info in list: {target_pos_info}")
                    break # Found the one we need

            if not target_pos_info:
                logger.debug(
                    f"No position found matching symbol {market_id} and positionIdx {self.config.position_idx} in the returned data. Assuming flat."
                )
                return positions_dict # Return empty dict, means flat

            # --- Parse the found position details using safe_decimal ---\
            qty = safe_decimal(target_pos_info.get("size", "0"))
            side_api = target_pos_info.get("side", "None").lower() # 'Buy' -> 'buy', 'Sell' -> 'sell', 'None' -> 'none'
            entry_price = safe_decimal(target_pos_info.get("avgPrice", "0"))
            liq_price_raw = safe_decimal(target_pos_info.get("liqPrice", "0"))
            unrealized_pnl = safe_decimal(target_pos_info.get("unrealisedPnl", "0"))
            sl_price_raw = safe_decimal(target_pos_info.get("stopLoss", "0"))
            tp_price_raw = safe_decimal(target_pos_info.get("takeProfit", "0"))
            tsl_trigger_price_raw = safe_decimal(target_pos_info.get("trailingStop", "0")) # This is the distance for V5 TSL
            tsl_activation_price_raw = safe_decimal(target_pos_info.get("activePrice", "0")) # Activation price for V5 TSL

            # --- Validate and clean up parsed values ---\
            qty_abs = qty.copy_abs() if not qty.is_nan() else Decimal("0")
            entry_price = entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN")
            liq_price = liq_price_raw if not liq_price_raw.is_nan() and liq_price_raw > 0 else Decimal("NaN")
            sl_price = sl_price_raw if not sl_price_raw.is_nan() and sl_price_raw > 0 else None
            tp_price = tp_price_raw if not tp_price_raw.is_nan() and tp_price_raw > 0 else None

            # V5 TSL: 'trailingStop' is distance, 'activePrice' is trigger price
            is_tsl_active = (not tsl_trigger_price_raw.is_nan() and tsl_trigger_price_raw > 0) or \
                            (not tsl_activation_price_raw.is_nan() and tsl_activation_price_raw > 0)
            tsl_distance = tsl_trigger_price_raw if not tsl_trigger_price_raw.is_nan() and tsl_trigger_price_raw > 0 else None
            tsl_activation_price = tsl_activation_price_raw if not tsl_activation_price_raw.is_nan() and tsl_activation_price_raw > 0 else None

            is_position_open = qty_abs >= POSITION_QTY_EPSILON

            if not is_position_open:
                logger.debug(f"Position size {qty} is negligible or zero. Considered flat.")
                return positions_dict # Return empty dict

            # Determine position side ('long' or 'short') based on API 'side' field
            # V5: 'Buy' means long, 'Sell' means short, 'None' means flat (already handled by qty check)
            position_side_key: Optional[str] = None
            if side_api == "buy": position_side_key = "long"
            elif side_api == "sell": position_side_key = "short"
            # If side is 'None' but qty > 0 and posIdx indicates hedge, infer side (less reliable)
            elif side_api == "none" and self.config.position_idx == 1: position_side_key = "long"
            elif side_api == "none" and self.config.position_idx == 2: position_side_key = "short"

            if position_side_key:
                position_details = {
                    "qty": qty_abs, # Store absolute quantity
                    "entry_price": entry_price,
                    "liq_price": liq_price,
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "side_api": side_api, # Store original API side ('buy'/'sell'/'None')
                    "info": target_pos_info, # Store raw info for debugging
                    "stop_loss_price": sl_price,
                    "take_profit_price": tp_price,
                    "is_tsl_active": is_tsl_active,
                    "tsl_distance": tsl_distance, # Store the TSL distance if active
                    "tsl_activation_price": tsl_activation_price, # Store the TSL activation price if set
                }
                positions_dict[position_side_key] = position_details
                entry_str = entry_price.normalize() if not entry_price.is_nan() else "[dim]N/A[/]"
                logger.debug(
                    f"Found {position_side_key.upper()} position: Qty={qty_abs.normalize()}, Entry={entry_str}"
                )
            else:
                 logger.warning(f"Position found with size {qty} but side is '{side_api}' and posIdx is {self.config.position_idx}. Could not determine long/short state reliably. Treating as flat for safety.")
                 return positions_dict # Return empty

            return positions_dict

        except Exception as e:
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True
            )
            return None


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators needed for the strategy."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_indicators(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates EMAs, Stochastic, ATR, ADX from OHLCV data with improved type handling."""
        logger.info(
            f"{Fore.CYAN}# Weaving indicator patterns (EMA, Stoch, ATR, ADX)...{Style.RESET_ALL}"
        )
        if df is None or df.empty:
            logger.error(f"{Fore.RED}No DataFrame provided for indicators.{Style.RESET_ALL}")
            return None

        required_cols = ["open", "high", "low", "close", "volume"] # Include volume for ADX/ATR context
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(f"{Fore.RED}DataFrame missing required columns for indicators: {missing}{Style.RESET_ALL}")
            return None

        try:
            # Work with a copy, select only necessary columns
            df_calc = df[required_cols].copy()

            # Robust conversion to numeric (float) for calculations, handling Decimal, strings, None, etc.
            for col in required_cols:
                if df_calc[col].empty:
                    logger.warning(f"Column {col} is empty, ensuring float type.")
                    df_calc[col] = pd.Series(dtype=float)
                    continue

                # Define a safe conversion function
                def safe_to_float(x: Any) -> float:
                    if isinstance(x, (float, int)):
                        return float(x)
                    if isinstance(x, Decimal):
                        return float('nan') if x.is_nan() or not x.is_finite() else float(x)
                    if isinstance(x, str):
                        try:
                            # Try direct float conversion first
                            return float(x)
                        except ValueError:
                            # Handle common non-numeric strings
                            if x.strip().lower() in ["nan", "none", "null", ""]:
                                return float('nan')
                            # If it's something else, log and return NaN
                            logger.debug(f"Could not convert string '{x}' in column '{col}' to float.")
                            return float('nan')
                    if x is None:
                        return float('nan')
                    # For other unexpected types, log and return NaN
                    logger.warning(f"Unexpected type {type(x)} in column '{col}', converting to NaN.")
                    return float('nan')

                # Apply the safe conversion using map for efficiency
                df_calc[col] = df_calc[col].map(safe_to_float)

                # Ensure the final dtype is float after conversion
                df_calc[col] = df_calc[col].astype(float)

            # Drop rows with NaN in essential OHLC columns *after* conversion
            initial_len = len(df_calc)
            ohlc_cols = ["open", "high", "low", "close"]
            df_calc.dropna(subset=ohlc_cols, inplace=True)
            rows_dropped = initial_len - len(df_calc)
            if rows_dropped > 0:
                 logger.debug(f"Dropped {rows_dropped} rows with NaN in OHLC after float conversion.")

            if df_calc.empty:
                logger.error(f"{Fore.RED}DataFrame empty after NaN drop during indicator calculation.{Style.RESET_ALL}")
                return None

            # --- Check Data Length ---\
            # ADX needs roughly 2*period for smoothing warmup
            # Stoch needs period + smooth_k + smooth_d
            max_period = max(
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d,
                self.config.atr_period, self.config.adx_period * 2,
            )
            min_required_len = max_period + 20 # Increased buffer for stability
            if len(df_calc) < min_required_len:
                logger.error(f"{Fore.RED}Insufficient data ({len(df_calc)} rows < required ~{min_required_len}) for indicators.{Style.RESET_ALL}")
                return None

            # --- Indicator Calculations (using df_calc with float types) ---\
            close_s = df_calc["close"]
            high_s = df_calc["high"]
            low_s = df_calc["low"]

            # EMAs
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator (%K, %D)
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            # Avoid division by zero if range is flat
            stoch_k_raw = np.where(stoch_range > 1e-12, 100 * (close_s - low_min) / stoch_range, 50.0)
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=df_calc.index).fillna(50) # Fill initial NaNs with 50
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # ATR
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
            # Use Wilder's smoothing (EMA with alpha=1/period) for ATR
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, atr_s, self.config.adx_period
            )

            # --- Extract Latest Values & Convert back to Decimal ---\
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                """Safely get the last valid (non-NaN, finite) value from a Series and convert to Decimal."""
                # Drop NaN and infinite values before selecting the last one
                valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
                if valid_series.empty: return Decimal("NaN")
                last_valid_float = valid_series.iloc[-1]
                # Use safe_decimal for robust conversion from the float
                dec_val = safe_decimal(str(last_valid_float)) # Convert float to string first
                if dec_val.is_nan():
                     logger.error(f"Failed converting latest {name} value '{last_valid_float}' (type: {type(last_valid_float).__name__}) to Decimal.")
                return dec_val

            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal(atr_s, "atr"),
                "atr_period": self.config.atr_period, # Store period used for ATR
                "adx": get_latest_decimal(adx_s, "adx"),
                "pdi": get_latest_decimal(pdi_s, "pdi"),
                "mdi": get_latest_decimal(mdi_s, "mdi"),
            }

            # --- Calculate Previous Stochastic K for Exit Logic ---\
            stoch_k_valid = stoch_k_s.replace([np.inf, -np.inf], np.nan).dropna()
            k_prev = Decimal("NaN")
            if len(stoch_k_valid) >= 2:
                k_prev = safe_decimal(str(stoch_k_valid.iloc[-2])) # Convert float to string first
            indicators_out["stoch_k_prev"] = k_prev # Add previous K to output

            # --- Calculate Stochastic Cross Signals ---\
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]
            d_prev = Decimal("NaN")
            stoch_d_valid = stoch_d_s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(stoch_d_valid) >= 2:
                 d_prev = safe_decimal(str(stoch_d_valid.iloc[-2]))

            stoch_kd_bullish = False
            stoch_kd_bearish = False
            # Check if all required stoch values are valid Decimals
            if not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev]):
                crossed_above = (k_last > d_last) and (k_prev <= d_prev)
                crossed_below = (k_last < d_last) and (k_prev >= d_prev)
                # Check if previous K or D was in the zone
                prev_in_oversold = (k_prev <= self.config.stoch_oversold_threshold) or \
                                   (d_prev <= self.config.stoch_oversold_threshold)
                prev_in_overbought = (k_prev >= self.config.stoch_overbought_threshold) or \
                                     (d_prev >= self.config.stoch_overbought_threshold)

                if crossed_above and prev_in_oversold: stoch_kd_bullish = True
                if crossed_below and prev_in_overbought: stoch_kd_bearish = True

            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish

            # Final check for critical NaNs
            critical_keys = [
                "fast_ema", "slow_ema", "trend_ema", "atr",
                "stoch_k", "stoch_d", "stoch_k_prev", # Added prev K here
                "adx", "pdi", "mdi",
            ]
            failed_indicators = [
                k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                logger.error(
                    f"{Fore.RED}Critical indicators calculated as NaN: {', '.join(failed_indicators)}. This may prevent signal generation.{Style.RESET_ALL}"
                )
                # ATR is essential for risk calculation, treat its absence as critical
                if indicators_out.get("atr", Decimal("NaN")).is_nan():
                     logger.error(f"{Fore.RED}ATR is NaN, cannot proceed with risk calculations. Aborting indicator calc.{Style.RESET_ALL}")
                     return None

            logger.info(f"{Fore.GREEN}{Style.BRIGHT}Indicator patterns woven successfully.{Style.RESET_ALL}")
            return cast(Dict[str, Union[Decimal, bool, int]], indicators_out) # Cast for type checker

        except Exception as e:
            logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_adx(
        self,
        high_s: pd.Series,
        low_s: pd.Series,
        atr_s: pd.Series, # Pre-calculated ATR series
        period: int,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA)."""
        if period <= 0:
            raise ValueError("ADX period must be positive")
        # Create NaN series for early return
        nan_series = pd.Series(np.nan, index=high_s.index)
        if atr_s.empty or atr_s.isnull().all(): # Check if ATR is empty or all NaN
             logger.error("ATR series is empty or all NaN, cannot calculate ADX.")
             return nan_series, nan_series, nan_series

        # Calculate Directional Movement (+DM, -DM)
        move_up = high_s.diff()
        move_down = -low_s.diff() # Note: diff is current - previous

        # +DM is UpMove if UpMove > DownMove and UpMove > 0, else 0
        plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        # -DM is DownMove if DownMove > UpMove and DownMove > 0, else 0
        minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

        # Smoothed +DM, -DM using Wilder's method (equivalent to EMA with alpha=1/period)
        alpha = 1 / period
        # Ensure initial NaNs from diff() are handled, fillna(0) before ewm
        plus_dm_s = pd.Series(plus_dm, index=high_s.index).fillna(0).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=high_s.index).fillna(0).ewm(alpha=alpha, adjust=False).mean()

        # Calculate Directional Indicators (+DI, -DI)
        # Avoid division by zero or NaN ATR
        safe_atr_s = atr_s.replace(0, np.nan) # Replace 0 ATR with NaN to avoid division error
        pdi_s_raw = np.where(safe_atr_s.notnull(), 100 * plus_dm_s / safe_atr_s, 0.0)
        mdi_s_raw = np.where(safe_atr_s.notnull(), 100 * minus_dm_s / safe_atr_s, 0.0)
        pdi_s = pd.Series(pdi_s_raw, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s_raw, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        di_diff_abs = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
        # Avoid division by zero if sum is zero
        dx_s_raw = np.where(di_sum > 1e-12, 100 * di_diff_abs / di_sum, 0.0)
        dx_s = pd.Series(dx_s_raw, index=high_s.index).fillna(0)

        # Calculate Average Directional Index (ADX - Smoothed DX using Wilder's EMA)
        adx_s = dx_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        return adx_s, pdi_s, mdi_s


# --- Signal Generator Class ---
class SignalGenerator:
    """Generates trading signals based on indicator conditions."""

    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_signals(
        self,
        df_last_candles: pd.DataFrame,
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Dict[str, Union[bool, str]]:
        """Generates 'long'/'short' entry signals and provides a reason string."""
        result: Dict[str, Union[bool, str]] = {
            "long": False,
            "short": False,
            "reason": "Initializing", # Default reason
        }

        # --- Pre-checks ---\
        if not indicators:
            result["reason"] = "No Signal: Indicators missing"
            logger.debug(result["reason"])
            return result
        if df_last_candles is None or len(df_last_candles) < 2:
            candle_count = len(df_last_candles) if df_last_candles is not None else 0
            reason = f"No Signal: Insufficient candle data (<2, got {candle_count})"
            result["reason"] = reason
            logger.debug(reason)
            return result

        try:
            # --- Extract Data & Indicators Safely ---\
            latest_candle = df_last_candles.iloc[-1]
            prev_candle = df_last_candles.iloc[-2]
            current_price = safe_decimal(latest_candle["close"])
            prev_close = safe_decimal(prev_candle["close"])

            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price})"
                logger.warning(result["reason"])
                return result

            required_indicator_keys = [
                "stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"
            ]
            ind_values: Dict[str, Decimal] = {} # Store valid Decimal indicators
            nan_keys = []
            for key in required_indicator_keys:
                val = indicators.get(key)
                # Ensure value is a valid, finite Decimal
                if isinstance(val, Decimal) and not val.is_nan() and val.is_finite():
                    ind_values[key] = val
                else:
                    nan_keys.append(key)

            if nan_keys:
                result["reason"] = f"No Signal: Required indicator(s) NaN/Missing/Infinite: {', '.join(nan_keys)}"
                logger.warning(result["reason"])
                return result

            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            # Get boolean flags (default to False if missing)
            kd_bull = indicators.get("stoch_kd_bullish", False) is True
            kd_bear = indicators.get("stoch_kd_bearish", False) is True

            # --- Define Conditions ---\
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema

            trend_buffer = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
            price_above_trend_ema = current_price > (trend_ema + trend_buffer) # Price clearly above trend EMA
            price_below_trend_ema = current_price < (trend_ema - trend_buffer) # Price clearly below trend EMA
            trend_allows_long = price_above_trend_ema if self.config.trade_only_with_trend else True
            trend_allows_short = price_below_trend_ema if self.config.trade_only_with_trend else True
            trend_reason = f"Trend(P:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{trend_ema:.{DEFAULT_PRICE_DP}f} +/- {trend_buffer:.{DEFAULT_PRICE_DP}f})" if self.config.trade_only_with_trend else "TrendFilter OFF"

            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            stoch_reason = f"Stoch(K:{k:.1f} {'BullX' if kd_bull else ''}{'BearX' if kd_bear else ''})"

            significant_move = True
            atr_reason = "ATR Filter OFF"
            if self.config.atr_move_filter_multiplier > 0:
                if atr.is_nan() or atr <= 0:
                    atr_reason = f"ATR Filter Skipped (Invalid ATR: {atr})"
                    significant_move = False # Cannot evaluate if ATR is invalid
                elif prev_close.is_nan():
                    atr_reason = "ATR Filter Skipped (Prev Close NaN)"
                    significant_move = False # Cannot evaluate if prev close is invalid
                else:
                    atr_move_threshold = atr * self.config.atr_move_filter_multiplier
                    price_move = (current_price - prev_close).copy_abs()
                    significant_move = price_move > atr_move_threshold
                    atr_reason = f"ATR Move({price_move:.{DEFAULT_PRICE_DP}f}) {'OK' if significant_move else 'LOW'} vs Thr({atr_move_threshold:.{DEFAULT_PRICE_DP}f})"

            adx_is_trending = adx > self.config.min_adx_level
            adx_long_direction = pdi > mdi
            adx_short_direction = mdi > pdi
            adx_allows_long = adx_is_trending and adx_long_direction
            adx_allows_short = adx_is_trending and adx_short_direction
            adx_reason = f"ADX({adx:.1f}) {'OK' if adx_is_trending else 'LOW'} vs {self.config.min_adx_level:.1f} | Dir({'+DI' if adx_long_direction else '-DI' if adx_short_direction else 'NONE'})"

            base_long_signal = ema_bullish_cross and stoch_long_cond
            base_short_signal = ema_bearish_cross and stoch_short_cond

            final_long_signal = base_long_signal and trend_allows_long and significant_move and adx_allows_long
            final_short_signal = base_short_signal and trend_allows_short and significant_move and adx_allows_short

            # --- Build Detailed Reason String ---\
            if final_long_signal:
                result["long"] = True
                result["reason"] = f"Long Signal: EMA Bull | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            elif final_short_signal:
                result["short"] = True
                result["reason"] = f"Short Signal: EMA Bear | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            else:
                # Provide reason why signal was not generated or blocked
                reason_parts = ["No Signal:"]
                if not base_long_signal and not base_short_signal:
                     reason_parts.append(f"Base EMA({'>' if ema_bullish_cross else '<' if ema_bearish_cross else '='}) Stoch({'L OK' if stoch_long_cond else 'L !OK'}/{'S OK' if stoch_short_cond else 'S !OK'})")
                elif base_long_signal and not trend_allows_long:
                    reason_parts.append(f"Long Blocked (Trend Filter: {trend_reason})")
                elif base_short_signal and not trend_allows_short:
                     reason_parts.append(f"Short Blocked (Trend Filter: {trend_reason})")
                elif (base_long_signal or base_short_signal) and not significant_move:
                     reason_parts.append(f"Blocked (ATR Filter: {atr_reason})")
                elif base_long_signal and significant_move and not adx_allows_long:
                     reason_parts.append(f"Long Blocked (ADX Filter: {adx_reason})")
                elif base_short_signal and significant_move and not adx_allows_short:
                     reason_parts.append(f"Short Blocked (ADX Filter: {adx_reason})")
                else: # Should cover all blocking cases, add generic fallback
                    reason_parts.append(f"Conditions unmet (EMA:{ema_bullish_cross}/{ema_bearish_cross}, Stoch:{stoch_long_cond}/{stoch_short_cond}, Trend:{trend_allows_long}/{trend_allows_short}, ATR:{significant_move}, ADX:{adx_allows_long}/{adx_allows_short})")
                result["reason"] = " | ".join(reason_parts)

            # Log signal generation result
            log_level_sig = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_sig, f"Signal Check: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating signals: {e}{Style.RESET_ALL}", exc_info=True)
            result["reason"] = f"No Signal: Exception ({type(e).__name__})"
            result["long"] = False
            result["short"] = False

        return result

    def check_exit_signals(
        self,
        position_side: str, # 'long' or 'short'
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Optional[str]:
        """Checks for signal-based exits (EMA cross, Stoch reversal). Returns reason string or None."""
        if not indicators:
            logger.warning("Cannot check exit signals: indicators missing.")
            return None

        # --- Extract indicators safely ---
        fast_ema = indicators.get("fast_ema")
        slow_ema = indicators.get("slow_ema")
        stoch_k = indicators.get("stoch_k")
        stoch_k_prev = indicators.get("stoch_k_prev") # Previous K value is crucial

        # Check for missing or invalid (NaN/Infinite) required indicators
        required_vals = [fast_ema, slow_ema, stoch_k, stoch_k_prev]
        if any(not isinstance(v, Decimal) or v.is_nan() or not v.is_finite() for v in required_vals):
            logger.warning(
                "Cannot check exit signals due to missing/invalid indicators (EMA/Stoch/StochPrev)."
            )
            return None # Cannot proceed without valid indicator values

        # Cast to Decimal after validation for type checker
        fast_ema = cast(Decimal, fast_ema)
        slow_ema = cast(Decimal, slow_ema)
        stoch_k = cast(Decimal, stoch_k)
        stoch_k_prev = cast(Decimal, stoch_k_prev)

        # --- Calculate EMA cross state ---
        ema_bullish_cross = fast_ema > slow_ema
        ema_bearish_cross = fast_ema < slow_ema

        exit_reason: Optional[str] = None

        # --- Define Overbought/Oversold levels from config ---
        oversold_level = self.config.stoch_oversold_threshold
        overbought_level = self.config.stoch_overbought_threshold

        # --- Evaluate Exit Conditions based on Position Side ---
        if position_side == "long":
            # Priority 1: EMA Bearish Cross
            if ema_bearish_cross:
                exit_reason = "Exit Signal: EMA Bearish Cross"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal Confirmation from Overbought
            # Check if previous K was >= OB and current K crossed below OB
            elif stoch_k_prev >= overbought_level and stoch_k < overbought_level:
                exit_reason = f"Exit Signal: Stoch Reversal Confirmation (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed below {overbought_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}")

            # Informational Logging: Stochastic is in the zone but hasn't reversed yet
            elif stoch_k >= overbought_level:
                logger.debug(f"Exit Check (Long): Stoch K ({stoch_k:.1f}) >= Overbought ({overbought_level}), awaiting cross below for exit signal.")

        elif position_side == "short":
            # Priority 1: EMA Bullish Cross
            if ema_bullish_cross:
                exit_reason = "Exit Signal: EMA Bullish Cross"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal Confirmation from Oversold
            # Check if previous K was <= OS and current K crossed above OS
            elif stoch_k_prev <= oversold_level and stoch_k > oversold_level:
                exit_reason = f"Exit Signal: Stoch Reversal Confirmation (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed above {oversold_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}")

            # Informational Logging: Stochastic is in the zone but hasn't reversed yet
            elif stoch_k <= oversold_level:
                logger.debug(f"Exit Check (Short): Stoch K ({stoch_k:.1f}) <= Oversold ({oversold_level}), awaiting cross above for exit signal.")

        # Return the reason string if an exit condition was met, otherwise None
        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL), and closing using V5 API."""

    def __init__(
        self, config: TradingConfig, exchange_manager: ExchangeManager
    ):
        self.config = config
        self.exchange_manager = exchange_manager
        # Ensure exchange instance exists before assigning
        if not exchange_manager or not exchange_manager.exchange:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}OrderManager cannot initialize: Exchange instance missing.{Style.RESET_ALL}")
            raise ValueError("OrderManager requires a valid Exchange instance.")
        self.exchange = exchange_manager.exchange # Convenience accessor
        self.market_info = exchange_manager.market_info # Convenience accessor
        # Tracks active protection STATUS (not order IDs) for V5 position stops
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Optional[Dict[str, Optional[Decimal]]]: # Note: tp_price can be None
        """Calculates SL price, TP price, quantity, and TSL distance based on risk and ATR."""
        # --- Input Validation ---\
        if atr.is_nan() or atr <= 0 or not atr.is_finite():
            logger.error(f"Invalid ATR ({atr}) for parameter calculation.")
            return None
        if total_equity.is_nan() or total_equity <= 0 or not total_equity.is_finite():
            logger.error(f"Invalid equity ({total_equity}) for parameter calculation.")
            return None
        if current_price.is_nan() or current_price <= 0 or not current_price.is_finite():
            logger.error(f"Invalid current price ({current_price}) for parameter calculation.")
            return None
        if not self.market_info or "tick_size" not in self.market_info or "contract_size" not in self.market_info or "min_order_size" not in self.market_info:
             logger.error("Market info (tick_size, contract_size, min_order_size) missing for parameter calculation.")
             return None
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side '{side}' for calculation.")
            return None

        try:
            risk_amount_per_trade = total_equity * self.config.risk_percentage
            sl_distance_atr = atr * self.config.sl_atr_multiplier
            sl_price: Decimal
            if side == "buy": sl_price = current_price - sl_distance_atr
            else: sl_price = current_price + sl_distance_atr

            # Ensure SL price is positive
            if sl_price <= 0:
                logger.error(f"Calculated SL price ({self.exchange_manager.format_price(sl_price)}) is invalid (<=0). Cannot calculate position size.")
                return None

            # Ensure SL distance respects tick size
            sl_distance_price = (current_price - sl_price).copy_abs()
            min_tick_size = self.market_info['tick_size']
            if sl_distance_price < min_tick_size:
                sl_distance_price = min_tick_size
                if side == "buy": sl_price = current_price - sl_distance_price
                else: sl_price = current_price + sl_distance_price
                logger.warning(f"Initial SL distance < tick size. Adjusted SL price to {self.exchange_manager.format_price(sl_price)} (dist {sl_distance_price}).")
                # Re-check if adjusted SL is still valid
                if sl_price <= 0:
                     logger.error(f"Adjusted SL price ({self.exchange_manager.format_price(sl_price)}) is still invalid (<=0).")
                     return None

            # Should not happen with checks above, but safeguard
            if sl_distance_price <= 0:
                logger.error(f"Calculated SL distance ({sl_distance_price}) is invalid (<=0).")
                return None

            # --- Quantity Calculation ---\
            contract_size = self.market_info['contract_size']
            value_per_point: Decimal
            if self.config.market_type == "inverse":
                 # For inverse, value per point changes with price
                 # Risk per contract = SL distance (in price) * Contract Size / Entry Price (approx)
                 # Qty = Risk Amount / (SL Distance * Contract Size / Entry Price)
                 # Qty = (Risk Amount * Entry Price) / (SL Distance * Contract Size)
                 if current_price <= 0: # Already checked, but defensive
                      logger.error("Cannot calculate inverse contract quantity: Current price is zero/negative.")
                      return None
                 risk_per_contract_in_quote = sl_distance_price # Risk in quote currency if 1 contract moves 1 point
                 # Convert risk amount to base currency value at current price
                 # This calculation is complex. Simpler: Calculate Qty in Base, then convert if needed.
                 # Qty (in Base) = Risk Amount (Quote) / (SL Distance (Quote) * Value per point (Base/Quote))
                 # Value per point (Base/Quote) = Contract Size (Base) / Price (Quote/Base) ? No.
                 # For inverse BTC/USD: Contract size is 1 USD. Value is 1 / Price (BTC/USD). Risk is in BTC.
                 # Risk Amount (BTC) = Risk Amount (USD) / Price (USD/BTC)
                 # Qty (Contracts=USD) = Risk Amount (BTC) / (SL Distance (BTC) * Value per contract (BTC/USD))
                 # Let's stick to CCXT's approach: calculate size in base currency first.
                 # Risk per unit of base currency = sl_distance_price
                 # Quantity (base) = risk_amount_per_trade (quote) / sl_distance_price (quote/base)
                 # This seems wrong for inverse.

                 # Let's use the definition: Qty * ContractSize * (1/ExitPrice - 1/EntryPrice) = PnL (Base)
                 # Risk Amount (Base) = Risk Amount (Quote) / Entry Price (Quote/Base)
                 # Qty (Contracts) * ContractSize (Quote/Contract) * (1/SLPrice - 1/EntryPrice) = -Risk Amount (Base)
                 # Qty = -Risk Amount (Base) / [ ContractSize * (1/SLPrice - 1/EntryPrice) ]
                 # Qty = - (Risk Amount (Quote) / EntryPrice) / [ ContractSize * ( (EntryPrice - SLPrice) / (SLPrice * EntryPrice) ) ]
                 # Qty = - Risk Amount (Quote) / [ ContractSize * (EntryPrice - SLPrice) / SLPrice ]
                 # Qty = Risk Amount (Quote) * SLPrice / [ ContractSize * SL Distance Price ]
                 # This seems plausible. Qty is in contracts (e.g., USD for BTC/USD)
                 if contract_size <= 0:
                      logger.error("Cannot calculate inverse quantity: Invalid contract size.")
                      return None
                 quantity = (risk_amount_per_trade * sl_price) / (contract_size * sl_distance_price)

            else: # Linear/Swap (e.g., BTC/USDT:USDT)
                 # Value per point = Contract Size (usually 1 for USDT perpetuals)
                 # Risk per contract (in Quote) = SL Distance (Quote) * Contract Size (Base/Contract) * Price (Quote/Base) ? No.
                 # Risk per contract (in Quote) = SL Distance (Quote) * Contract Size (Base/Contract) * 1 (if linear)
                 # Let's assume ContractSize for linear is effectively 1 base unit (e.g., 1 BTC for BTC/USDT)
                 # Risk per base unit = SL Distance (Quote)
                 # Quantity (Base) = Risk Amount (Quote) / SL Distance (Quote)
                 # If ContractSize != 1 base unit, adjust:
                 # Quantity (Base) = Risk Amount (Quote) / (SL Distance (Quote) * Contract Size (Base/Contract)) ? No.
                 # Let's use: Risk per Contract = SL Distance Price * Value per Contract Point
                 # Value per Contract Point = Contract Size (e.g., 0.001 BTC for some contracts)
                 # Quantity (Contracts) = Risk Amount (Quote) / (SL Distance Price (Quote) * Contract Size (Base/Contract))
                 # If contract size is 1 (meaning 1 contract = 1 base unit like BTC), then:
                 # Quantity (Base = Contracts) = Risk Amount (Quote) / SL Distance Price (Quote)
                 value_per_point = contract_size # For linear, contract size is in base currency (e.g., 1 BTC)
                 risk_per_contract = sl_distance_price * value_per_point
                 if risk_per_contract <= 0:
                     logger.error(f"Calculated zero or negative risk per contract ({risk_per_contract}) for linear market.")
                     return None
                 quantity = risk_amount_per_trade / risk_per_contract


            # Format quantity according to market rules
            quantity_str = self.exchange_manager.format_amount(quantity, rounding_mode=ROUND_DOWN)
            quantity_decimal = safe_decimal(quantity_str)

            if quantity_decimal.is_nan() or quantity_decimal <= 0:
                 logger.error(f"Calculated quantity ({quantity_str}) is invalid or zero.")
                 return None

            # Check against minimum order size
            min_order_size = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_size.is_nan() and quantity_decimal < min_order_size:
                logger.error(f"Calculated qty {quantity_decimal.normalize()} < minimum {min_order_size.normalize()}. Increase risk or check config.")
                return None

            # --- Take Profit Calculation ---\
            tp_price: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_distance_atr = atr * self.config.tp_atr_multiplier
                if side == "buy": tp_price = current_price + tp_distance_atr
                else: tp_price = current_price - tp_distance_atr
                if tp_price <= 0:
                    logger.warning(f"Calculated TP price ({self.exchange_manager.format_price(tp_price)}) invalid (<=0). Disabling TP.")
                    tp_price = None

            # --- Trailing Stop Calculation (Distance) ---\
            # TSL distance is calculated as a percentage of the current price
            tsl_distance_price = current_price * (self.config.trailing_stop_percent / 100)
            # Ensure TSL distance respects tick size
            if tsl_distance_price < min_tick_size:
                 tsl_distance_price = min_tick_size
            # Format distance like a price (it's a price difference)
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance_price)
            tsl_distance_decimal = safe_decimal(tsl_distance_str)
            if tsl_distance_decimal.is_nan() or tsl_distance_decimal <= 0:
                 logger.warning(f"Calculated invalid TSL distance ({tsl_distance_str}). TSL might fail.")
                 tsl_distance_decimal = Decimal('NaN') # Mark as invalid

            # --- Format Final SL/TP Prices ---\
            sl_price_str = self.exchange_manager.format_price(sl_price)
            sl_price_decimal = safe_decimal(sl_price_str)
            tp_price_decimal: Optional[Decimal] = None
            if tp_price is not None:
                 tp_price_str = self.exchange_manager.format_price(tp_price)
                 tp_price_decimal = safe_decimal(tp_price_str)
                 if tp_price_decimal.is_nan() or tp_price_decimal <= 0:
                      logger.warning(f"Failed to format valid TP price ({tp_price_str}). Disabling TP.")
                      tp_price_decimal = None

            # --- Calculate TSL Activation Price ---
            tsl_activation_price: Optional[Decimal] = None
            if self.config.tsl_activation_atr_multiplier > 0:
                 tsl_activation_distance = atr * self.config.tsl_activation_atr_multiplier
                 if side == "buy":
                     tsl_activation_price = current_price + tsl_activation_distance
                 else: # side == "sell"
                     tsl_activation_price = current_price - tsl_activation_distance

                 if tsl_activation_price <= 0:
                      logger.warning(f"Calculated TSL activation price ({self.exchange_manager.format_price(tsl_activation_price)}) invalid (<=0). TSL might not activate correctly.")
                      tsl_activation_price = None # Mark as invalid
                 else:
                      # Format activation price
                      tsl_act_price_str = self.exchange_manager.format_price(tsl_activation_price)
                      tsl_activation_price = safe_decimal(tsl_act_price_str)
                      if tsl_activation_price.is_nan() or tsl_activation_price <= 0:
                           logger.warning(f"Failed to format valid TSL activation price ({tsl_act_price_str}). TSL might not activate correctly.")
                           tsl_activation_price = None


            params_out: Dict[str, Optional[Decimal]] = {
                "qty": quantity_decimal,
                "sl_price": sl_price_decimal if not sl_price_decimal.is_nan() else None,
                "tp_price": tp_price_decimal, # Will be None if disabled or invalid
                "tsl_distance": tsl_distance_decimal if not tsl_distance_decimal.is_nan() else None,
                "tsl_activation_price": tsl_activation_price, # Will be None if calc failed or multiplier is 0
            }

            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "None"
            log_tsl_dist_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] else "Invalid"
            log_tsl_act_str = f"{params_out['tsl_activation_price'].normalize()}" if params_out['tsl_activation_price'] else "None"
            logger.info(
                f"Trade Params Calculated: Side={side.upper()}, "
                f"Qty={params_out['qty'].normalize()}, "
                f"Entry~={current_price.normalize()}, "
                f"SL={params_out['sl_price'].normalize() if params_out['sl_price'] else 'ERR'}, "
                f"TP={log_tp_str}, "
                f"TSL(Dist~={log_tsl_dist_str}, ActPx~={log_tsl_act_str}), "
                f"RiskAmt={risk_amount_per_trade.normalize():.{DEFAULT_PRICE_DP}f}, ATR={atr.normalize():.{DEFAULT_PRICE_DP}f}"
            )
            return params_out

        except (InvalidOperation, DivisionByZero, TypeError, Exception) as e:
            logger.error(f"Error calculating trade parameters for {side} side: {e}", exc_info=True)
            return None

    def _execute_market_order(
        self, side: str, qty_decimal: Decimal
    ) -> Optional[Dict]:
        """Executes a market order with retries and basic confirmation."""
        if not self.exchange or not self.market_info:
            logger.error("Cannot execute market order: Exchange/Market info missing.")
            return None

        symbol = self.config.symbol
        # Format quantity according to market rules before sending
        qty_str = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_decimal = safe_decimal(qty_str)

        if final_qty_decimal.is_nan() or final_qty_decimal <= 0:
            logger.error(f"Attempted market order with zero/invalid formatted quantity: {qty_str} (Original: {qty_decimal})")
            return None

        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} order: {final_qty_decimal.normalize()} {symbol}...{Style.RESET_ALL}"
        )
        try:
            params = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx, # Specify hedge mode index
                "timeInForce": "ImmediateOrCancel", # Use IOC for market orders to avoid partial fills hanging
                # "reduceOnly": False, # Ensure it's not accidentally set for entry
            }
            # CCXT expects amount as float
            amount_float = float(final_qty_decimal)

            order = fetch_with_retries(
                self.exchange.create_market_order,
                symbol=symbol,
                side=side,
                amount=amount_float,
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
                # Fail fast on insufficient funds or invalid order params
                fail_fast_exceptions=(ccxt.InsufficientFunds, ccxt.InvalidOrder),
            )

            if order is None:
                # fetch_with_retries would have raised if retries failed, so this means fail-fast or unexpected None
                logger.error(f"{Fore.RED}Market order submission failed (returned None). Check previous logs for fail-fast reasons.{Style.RESET_ALL}")
                return None

            # --- Parse Order Response ---
            order_id = order.get("id", "[N/A]")
            order_status = order.get("status", "[unknown]") # e.g., 'open', 'closed', 'canceled', 'rejected'
            filled_qty_str = order.get("filled", "0")
            avg_fill_price_str = order.get("average", "0")
            filled_qty = safe_decimal(filled_qty_str)
            avg_fill_price = safe_decimal(avg_fill_price_str)

            avg_price_log_str = '[N/A]'
            if not avg_fill_price.is_nan() and avg_fill_price > 0:
                try:
                    # Use format_price for consistent display
                    avg_price_log_str = self.exchange_manager.format_price(avg_fill_price)
                except (InvalidOperation, ValueError):
                    logger.warning(f"Could not format avg_fill_price {avg_fill_price} for logging.")
                    avg_price_log_str = str(avg_fill_price)

            log_color = Fore.GREEN if order_status in ['closed', 'filled'] else Fore.YELLOW if order_status == 'open' else Fore.RED
            logger.trade(
                f"{log_color}{Style.BRIGHT}Market order submitted: ID {order_id}, Side {side.upper()}, Qty {final_qty_decimal.normalize()}, Status: {order_status}, Filled: {filled_qty.normalize()}, AvgPx: {avg_price_log_str}{Style.RESET_ALL}"
            )
            termux_notify(
                f"{symbol} Order Submitted", f"Market {side.upper()} {final_qty_decimal.normalize()} ID:{order_id} Status:{order_status}"
            )

            # Check for immediate rejection or cancellation
            if order_status in ["rejected", "canceled", "expired"]:
                 reason = order.get("info", {}).get("rejectReason", "Unknown reason") # Check info for details
                 logger.error(f"{Fore.RED}Market order {order_id} was {order_status}. Reason: {reason}. Check exchange info: {order.get('info')}{Style.RESET_ALL}")
                 return None # Order failed immediately

            # If status is 'open' or 'closed' (partially filled IOC might close), proceed.
            # Verification step later will confirm the position state.

            # Short delay before verification
            logger.debug(f"Waiting short delay ({self.config.order_check_delay_seconds}s) after order {order_id} submission...")
            time.sleep(self.config.order_check_delay_seconds)

            return order # Return submitted order info

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # These are caught by fail_fast in fetch_with_retries, but catch again for specific logging
            logger.error(f"{Fore.RED}Order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{symbol} Order FAILED", f"Market {side.upper()} failed: {str(e)[:50]}")
            return None
        except Exception as e:
            # Includes errors raised by fetch_with_retries after max retries
            logger.error(f"{Fore.RED}Unexpected error or max retries reached placing market order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol} Order ERROR", f"Market {side.upper()} error.")
            return None

    def _set_position_protection(
        self,
        position_side: str, # 'long' or 'short'
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False,
        tsl_distance: Optional[Decimal] = None, # Price distance
        tsl_activation_price: Optional[Decimal] = None,
    ) -> bool:
        """Sets SL, TP, or TSL for a position using V5 setTradingStop (via private call).
           Ensures numeric parameters are passed as formatted strings. Clears other stops when activating TSL.
           Updates internal protection_tracker on success.
        """
        if not self.exchange: logger.error("Cannot set protection: Exchange missing."); return False
        if not self.market_info: logger.error("Cannot set protection: Market info missing."); return False
        market_id = self.market_info.get("id")
        if not market_id: logger.error("Cannot set protection: Market ID missing."); return False

        tracker_key = position_side.lower()
        if tracker_key not in self.protection_tracker:
             logger.error(f"Invalid position_side '{position_side}' for protection tracker."); return False

        # Use the parameter formatting helper (_format_v5_param)
        # Allow zero for SL/TP as "0" means cancel that specific stop
        sl_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True)
        tp_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True)
        # TSL distance requires formatting as price difference, zero is invalid for setting TSL distance
        tsl_distance_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False)
        # TSL activation price needs price formatting, zero is invalid for setting activation price
        tsl_activation_price_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False)

        # --- Prepare Base Parameters for V5 private_post_position_set_trading_stop ---\
        # Default all stops to "0" (meaning cancel/do not set)
        params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL, # Or "Partial" if needed
            "stopLoss": "0",
            "takeProfit": "0",
            "trailingStop": "0", # This is the distance value for V5 TSL
            "activePrice": "0", # This is the activation price for V5 TSL
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by, # Use same trigger for TP as SL (configurable if needed)
            "tslTriggerBy": self.config.tsl_trigger_by,
        }

        action_desc = ""
        new_tracker_state: Optional[str] = None

        # --- Logic Branch: Activate TSL ---\
        # Requires both distance and activation price for V5 TSL via this endpoint
        if is_tsl and tsl_distance_str and tsl_activation_price_str:
            params["trailingStop"] = tsl_distance_str
            params["activePrice"] = tsl_activation_price_str
            params["stopLoss"] = "0" # Explicitly clear fixed SL/TP when setting TSL
            params["takeProfit"] = "0"
            action_desc = f"ACTIVATE TSL (Dist: {tsl_distance_str}, ActPx: {tsl_activation_price_str})"
            new_tracker_state = "ACTIVE_TSL"
            logger.debug(f"TSL Params Prepared: {params}")

        # --- Logic Branch: Set Fixed SL/TP ---\
        # Only set if not activating TSL and at least one of SL/TP is provided
        elif not is_tsl and (sl_str or tp_str):
            params["stopLoss"] = sl_str if sl_str else "0" # Use formatted value or "0" to cancel
            params["takeProfit"] = tp_str if tp_str else "0"
            params["trailingStop"] = "0" # Ensure TSL is cleared
            params["activePrice"] = "0"
            action_desc = f"SET SL={params['stopLoss']} TP={params['takeProfit']}"
            new_tracker_state = "ACTIVE_SLTP" if params["stopLoss"] != "0" or params["takeProfit"] != "0" else None

        # --- Logic Branch: Clear All Stops ---\
        else:
            # This case handles clearing all stops explicitly, or if invalid inputs were provided
            # The default params already have all stops set to "0"
            action_desc = "CLEAR SL/TP/TSL"
            new_tracker_state = None
            logger.debug(f"Clearing all stops for {position_side.upper()} position.")

        # --- Execute API Call ---\
        symbol = self.config.symbol # Use the display symbol for logging
        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol}...{Style.RESET_ALL}")

        # --- Corrected Private Method Call ---\
        # Use the snake_case private method name corresponding to POST /v5/position/set-trading-stop
        private_method_name = "private_post_position_set_trading_stop"

        if not hasattr(self.exchange, private_method_name):
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Private method '{private_method_name}' not found in CCXT instance. Cannot set position protection. Check CCXT version/implementation.{Style.RESET_ALL}"
            )
            return False

        method_to_call = getattr(self.exchange, private_method_name)
        logger.debug(f"Calling private V5 method '{private_method_name}' with params: {params}")

        try:
            response = fetch_with_retries(
                method_to_call, # Pass the bound private method
                params=params, # Pass V5 params dict
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
                # Fail fast on invalid order type errors etc. if applicable
                fail_fast_exceptions=(ccxt.InvalidOrder,),
            )
            # logger.debug(f"SetTradingStop Response: {response}") # Verbose

            # --- Process Response ---\
            # V5 success code is 0
            if response and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Fore.GREEN}{Style.BRIGHT}{action_desc} successful for {position_side.upper()} {symbol}.{Style.RESET_ALL}")
                termux_notify(f"{symbol} Protection Set", f"{action_desc} {position_side.upper()}")
                self.protection_tracker[tracker_key] = new_tracker_state
                return True
            else:
                ret_code = response.get("retCode", "[N/A]") if response else "[No Response]"
                ret_msg = response.get("retMsg", "[No error message]") if response else "[No Response]"
                logger.error(f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol}. API Response: Code={ret_code}, Msg='{ret_msg}'{Style.RESET_ALL}")
                logger.debug(f"Failed SetTradingStop Full Response: {response}")
                termux_notify(f"{symbol} Protection FAILED", f"{action_desc[:30]} {position_side.upper()} failed: {ret_msg[:50]}\")")
                # Should we reset the tracker state on failure? Maybe safer not to assume.
                # self.protection_tracker[tracker_key] = None # Or keep previous state?
                return False
        except ccxt.InvalidOrder as e:
             logger.error(f"{Fore.RED}Invalid order parameters during {action_desc} for {position_side.upper()} {symbol}: {e}{Style.RESET_ALL}")
             termux_notify(f"{symbol} Protection FAILED", f"{action_desc[:30]} {position_side.upper()} invalid params.")
             return False
        except Exception as e:
            # Includes errors raised by fetch_with_retries after max retries
            logger.error(f"{Fore.RED}Unexpected error or max retries reached during {action_desc} for {position_side.upper()} {symbol}: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol} Protection ERROR", f"{action_desc[:30]} {position_side.upper()} error.")
            return False

    def _verify_position_state(
            self,
            expected_side: Optional[str], # 'long', 'short', or None (for expecting flat)
            expected_qty_min: Decimal = POSITION_QTY_EPSILON, # Minimum absolute qty if expecting a position
            max_attempts: int = 4, # Default attempts for verification
            delay_seconds: float = 1.5, # Shorter delay for verification
            action_context: str = "Verification" # e.g., "Post-Entry", "Post-Close"
        ) -> Tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """Fetches position state repeatedly to verify expected state after an action.
           Returns (verification_success: bool, final_position_state: Optional[Dict])
        """
        logger.debug(f"Verifying position state ({action_context}): Expecting '{expected_side}' (MinQty: {expected_qty_min if expected_side else 'N/A'})...")
        last_known_position_state: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(f"Verification attempt {attempt + 1}/{max_attempts}...")
            current_positions = self.exchange_manager.get_current_position()
            last_known_position_state = current_positions # Store the latest fetched state

            if current_positions is None:
                logger.warning(f"{Fore.YELLOW}{action_context} Verification Warning: Failed to fetch position state on attempt {attempt + 1}.{Style.RESET_ALL}")
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)
                    continue
                else:
                    logger.error(f"{Fore.RED}{action_context} Verification Failed: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}")
                    return False, last_known_position_state # Verification failed, return last attempt state (which is None here)

            # --- Check against expected state ---\
            is_currently_flat = True
            actual_side: Optional[str] = None
            actual_qty = Decimal("0")
            active_pos_data: Optional[Dict[str, Any]] = None

            long_pos_data = current_positions.get("long")
            short_pos_data = current_positions.get("short")

            # Check if long position exists and has significant quantity
            if long_pos_data:
                 long_qty = safe_decimal(long_pos_data.get("qty", "0"))
                 if not long_qty.is_nan() and long_qty.copy_abs() >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "long"
                     actual_qty = long_qty
                     active_pos_data = long_pos_data

            # Check if short position exists and has significant quantity (only if not already identified as long)
            if not active_pos_data and short_pos_data:
                 short_qty = safe_decimal(short_pos_data.get("qty", "0"))
                 if not short_qty.is_nan() and short_qty.copy_abs() >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "short"
                     actual_qty = short_qty
                     active_pos_data = short_pos_data

            # --- Evaluate success based on expectation ---\
            verification_met = False
            log_msg = ""
            if expected_side is None: # Expecting flat
                verification_met = is_currently_flat
                log_msg = f"Expected FLAT, Actual: {'FLAT' if is_currently_flat else f'{actual_side.upper()} Qty={actual_qty.normalize()}'}"
            elif actual_side == expected_side: # Expecting a specific side, and it matches
                # Check if quantity meets the minimum expectation
                qty_met = actual_qty.copy_abs() >= expected_qty_min
                verification_met = qty_met
                log_msg = f"Expected {expected_side.upper()} (MinQty~{expected_qty_min.normalize()}), Actual: {actual_side.upper()} Qty={actual_qty.normalize()} ({'QTY OK' if qty_met else 'QTY MISMATCH'})"
            else: # Expecting a specific side, but actual side is different or flat
                 verification_met = False
                 log_msg = f"Expected {expected_side.upper()}, Actual: {'FLAT' if is_currently_flat else (actual_side.upper() + ' Qty=' + actual_qty.normalize()) if actual_side else 'UNKNOWN'} (SIDE MISMATCH)"


            logger.debug(f"{action_context} Check {attempt + 1}: {log_msg}")

            if verification_met:
                logger.info(f"{Fore.GREEN}{Style.BRIGHT}{action_context} Verification SUCCESSFUL on attempt {attempt + 1}.{Style.RESET_ALL}")
                return True, current_positions # Verification succeeded, return final state

            # Verification not met, wait for next attempt if any remain
            if attempt < max_attempts - 1:
                logger.debug(f"State not as expected, waiting {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                 # Max attempts reached, verification failed
                 logger.error(f"{Fore.RED}{action_context} Verification FAILED after {max_attempts} attempts. Final state check: {log_msg}{Style.RESET_ALL}")
                 return False, current_positions # Verification failed, return last known state

        # Should not be reached if max_attempts >= 1
        logger.error(f"{action_context} Verification loop ended unexpectedly.")
        return False, last_known_position_state

    def place_risked_market_order(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Tuple[bool, Optional[Dict]]:
        """Calculates parameters, places market order, verifies position, sets initial SL/TP.
           Returns (success: bool, final_position_data: Optional[Dict])
        """
        if not self.exchange or not self.market_info: return False, None
        if side not in ["buy", "sell"]: logger.error(f"Invalid side '{side}'."); return False, None
        if atr.is_nan() or atr <= 0 or not atr.is_finite(): logger.error("Entry Aborted: Invalid ATR."); return False, None
        if total_equity is None or total_equity.is_nan() or total_equity <= 0 or not total_equity.is_finite(): logger.error("Entry Aborted: Invalid Equity."); return False, None
        if current_price.is_nan() or current_price <= 0 or not current_price.is_finite(): logger.error("Entry Aborted: Invalid Price."); return False, None

        position_side = "long" if side == "buy" else "short"
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Initiating Entry Sequence for {position_side.upper()} ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters
        logger.debug("Calculating trade parameters...")
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not trade_params.get("qty") or not trade_params.get("sl_price"):
            logger.error("Entry Aborted: Failed to calculate valid trade parameters (qty or sl_price missing/invalid?).")
            return False, None

        qty_to_order = cast(Decimal, trade_params["qty"])
        initial_sl_price = cast(Decimal, trade_params["sl_price"])
        initial_tp_price = trade_params.get("tp_price") # Can be None
        initial_tsl_distance = trade_params.get("tsl_distance") # Can be None
        initial_tsl_activation_price = trade_params.get("tsl_activation_price") # Can be None

        # 2. Execute Market Order
        logger.debug(f"Executing market {side} order for {qty_to_order.normalize()}...")
        order_info = self._execute_market_order(side, qty_to_order)
        if not order_info:
            logger.error("Entry Aborted: Market order execution failed or rejected.")
            # No position to clean up if order failed immediately
            return False, None
        order_id = order_info.get("id", "[N/A]")

        # 3. Verify Position Establishment using _verify_position_state
        logger.info(f"Verifying position establishment after market order {order_id}...")
        # Define verification parameters
        # Expect at least a small fraction of the intended quantity, allowing for fees/slippage
        min_expected_qty = qty_to_order * Decimal("0.5") # Be conservative: expect at least 50%? Adjust as needed.
        verification_ok, final_pos_state = self._verify_position_state(
            expected_side=position_side,
            expected_qty_min=min_expected_qty,
            max_attempts=5, # More attempts for verification
            delay_seconds=self.config.order_check_delay_seconds + 0.5, # Use configured delay + buffer
            action_context=f"Post-{position_side.upper()}-Entry"
        )

        if not verification_ok or not final_pos_state:
            logger.error(f"{Fore.RED}Entry Failed: Position verification FAILED after market order {order_id}. Manual check required! Attempting cleanup...{Style.RESET_ALL}")
            self._handle_entry_failure(position_side, qty_to_order, final_pos_state) # Attempt cleanup
            return False, None

        # Position verified, extract details from the final verified state
        active_pos_data = final_pos_state.get(position_side)
        if not active_pos_data: # Should not happen if verification_ok is True, but safeguard
            logger.error(f"{Fore.RED}Internal Error: Position verified OK but data missing for {position_side} in final state {final_pos_state}. Halting entry.{Style.RESET_ALL}")
            self._handle_entry_failure(position_side, qty_to_order, final_pos_state)
            return False, None

        filled_qty = safe_decimal(active_pos_data.get("qty", "0"))
        avg_entry_price = safe_decimal(active_pos_data.get("entry_price", "NaN"))

        # Log confirmation with verified details
        entry_px_str = self.exchange_manager.format_price(avg_entry_price) if not avg_entry_price.is_nan() else "[N/A]"
        logger.info(
            f"{Fore.GREEN}{Style.BRIGHT}Entry Confirmed: {position_side.upper()} position entered. "
            f"Qty: {filled_qty.normalize()}, Avg Entry Price: {entry_px_str}{Style.RESET_ALL}"
        )

        # 4. Set Initial Position Protection (SL/TP)
        logger.info(f"Setting initial protection: SL={initial_sl_price.normalize()}, TP={initial_tp_price.normalize() if initial_tp_price else 'None'}...")
        protection_set = self._set_position_protection(
            position_side=position_side,
            sl_price=initial_sl_price,
            tp_price=initial_tp_price,
            is_tsl=False # Set fixed SL/TP initially
        )

        if not protection_set:
            logger.warning(f"{Fore.YELLOW}Failed to set initial SL/TP after entry. Position is unprotected! Manual intervention may be needed.{Style.RESET_ALL}")
            # Decide whether to proceed or try to close? For now, proceed but log warning.
            # Optionally: Could attempt to close the position here if protection fails.
            # self.close_position(position_side, "Failed to set initial protection")

        # 5. Store potential TSL parameters for later activation check
        # We don't activate TSL immediately, but store the calculated values if valid
        active_pos_data["initial_tsl_distance"] = initial_tsl_distance
        active_pos_data["initial_tsl_activation_price"] = initial_tsl_activation_price

        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Entry Sequence for {position_side.upper()} Completed ---{Style.RESET_ALL}")
        return True, active_pos_data # Return success and the verified position data

    def _handle_entry_failure(self, expected_side: str, ordered_qty: Decimal, last_pos_state: Optional[Dict]) -> None:
        """Attempts to clean up if entry verification fails (e.g., close unexpected position)."""
        logger.warning(f"Handling entry failure for expected {expected_side.upper()}...")
        if last_pos_state:
            # Check if an unexpected position exists
            current_side: Optional[str] = None
            current_qty = Decimal("0")
            if last_pos_state.get("long"):
                current_side = "long"
                current_qty = safe_decimal(last_pos_state["long"].get("qty", "0"))
            elif last_pos_state.get("short"):
                current_side = "short"
                current_qty = safe_decimal(last_pos_state["short"].get("qty", "0"))

            if current_side and current_qty.copy_abs() >= POSITION_QTY_EPSILON:
                logger.warning(f"Unexpected {current_side.upper()} position found (Qty: {current_qty.normalize()}) after failed {expected_side.upper()} entry. Attempting to close.")
                self.close_position(current_side, f"Cleanup after failed {expected_side.upper()} entry")
            else:
                logger.info("No significant position found during entry failure cleanup.")
        else:
            logger.warning("Could not get position state during entry failure cleanup.")

    def close_position(self, position_side: str, reason: str) -> bool:
        """Closes the specified position with a market order and verifies closure."""
        if position_side not in ["long", "short"]:
            logger.error(f"Invalid side '{position_side}' for closing position.")
            return False
        if not self.exchange or not self.market_info:
            logger.error("Cannot close position: Exchange/Market info missing.")
            return False

        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Initiating Close Sequence for {position_side.upper()} (Reason: {reason}) ---{Style.RESET_ALL}")

        # 1. Get current position details to find quantity to close
        logger.debug("Fetching current position details for closure...")
        current_positions = self.exchange_manager.get_current_position()
        if not current_positions:
            logger.warning(f"Attempted to close {position_side} position, but failed to fetch current state. Assuming flat.")
            return True # Assume already flat if fetch fails

        pos_data = current_positions.get(position_side)
        if not pos_data:
            logger.info(f"Attempted to close {position_side} position, but no active position found. Already flat.")
            # Ensure protection tracker is cleared if state is flat
            self.protection_tracker[position_side] = None
            return True # Already flat

        qty_to_close = safe_decimal(pos_data.get("qty", "0"))
        if qty_to_close.is_nan() or qty_to_close.copy_abs() < POSITION_QTY_EPSILON:
            logger.info(f"Attempted to close {position_side} position, but quantity is negligible ({qty_to_close}). Already flat.")
            self.protection_tracker[position_side] = None
            return True # Effectively flat

        # 2. Cancel existing SL/TP/TSL before closing (optional but good practice)
        logger.debug(f"Clearing any existing protection for {position_side.upper()} before closing...")
        # Use is_tsl=False and provide no prices to clear all
        self._set_position_protection(position_side=position_side)
        # Small delay after clearing stops
        time.sleep(1)

        # 3. Execute closing market order
        close_side = "sell" if position_side == "long" else "buy"
        logger.debug(f"Executing market {close_side} order to close {qty_to_close.normalize()}...")
        # Use create_order with reduceOnly=True for V5 if possible, or just opposite market order
        params = {
            "category": self.config.bybit_v5_category,
            "positionIdx": self.config.position_idx,
            "reduceOnly": True, # Crucial for closing orders
            "timeInForce": "ImmediateOrCancel", # Try to fill immediately
        }
        try:
            close_order = fetch_with_retries(
                self.exchange.create_order,
                symbol=self.config.symbol,
                type='market',
                side=close_side,
                amount=float(qty_to_close), # CCXT expects float
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
                fail_fast_exceptions=(ccxt.InsufficientFunds, ccxt.InvalidOrder),
            )
            if not close_order:
                 logger.error(f"{Fore.RED}Market close order submission failed (returned None). Position might still be open! Manual check required.{Style.RESET_ALL}")
                 return False # Closure failed at order submission

            order_id = close_order.get("id", "[N/A]")
            order_status = close_order.get("status", "[unknown]")
            logger.trade(f"{Fore.YELLOW}Market close order submitted: ID {order_id}, Side {close_side.upper()}, Qty {qty_to_close.normalize()}, Status: {order_status}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Close Submitted", f"Market {close_side.upper()} {qty_to_close.normalize()} ID:{order_id}")

            if order_status in ["rejected", "canceled", "expired"]:
                 reason_api = close_order.get("info", {}).get("rejectReason", "Unknown reason")
                 logger.error(f"{Fore.RED}Market close order {order_id} was {order_status}. Reason: {reason_api}. Position might still be open! Manual check required.{Style.RESET_ALL}")
                 return False # Closure failed

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            logger.error(f"{Fore.RED}Close order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Close FAILED", f"Market {close_side.upper()} failed: {str(e)[:50]}")
            return False # Closure failed
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error or max retries reached placing close order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{self.config.symbol} Close ERROR", f"Market {close_side.upper()} error.")
            return False # Closure failed

        # 4. Verify Position Closure
        logger.info(f"Verifying position closure after market order {order_id}...")
        verification_ok, _ = self._verify_position_state(
            expected_side=None, # Expect flat
            max_attempts=5,
            delay_seconds=self.config.order_check_delay_seconds + 0.5,
            action_context=f"Post-{position_side.upper()}-Close"
        )

        if verification_ok:
            logger.info(f"{Fore.GREEN}{Style.BRIGHT}Position {position_side.upper()} successfully closed.{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Closed", f"{position_side.upper()} position closed.")
            self.protection_tracker[position_side] = None # Clear tracker on successful close
            logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Close Sequence for {position_side.upper()} Completed ---{Style.RESET_ALL}")
            return True
        else:
            logger.error(f"{Fore.RED}Position closure verification FAILED for {position_side.upper()}. Manual check required!{Style.RESET_ALL}")
            termux_notify(f"{self.config.symbol} Close Verify FAILED", f"{position_side.upper()} close verify failed!")
            # Don't clear tracker if verification failed
            return False

    def check_and_manage_trailing_stop(
        self,
        position_data: Dict[str, Any],
        position_side: str, # 'long' or 'short'
        current_price: Decimal,
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> None:
        """Checks if TSL should be activated based on ATR multiplier and current price."""
        if position_side not in ["long", "short"]: return
        if current_price.is_nan() or current_price <= 0: return
        if self.protection_tracker.get(position_side) == "ACTIVE_TSL":
            # logger.debug(f"TSL already active for {position_side.upper()} position.")
            return # TSL already active

        # Check if TSL activation parameters were calculated and stored during entry
        tsl_activation_price = position_data.get("initial_tsl_activation_price")
        tsl_distance = position_data.get("initial_tsl_distance")

        if not isinstance(tsl_activation_price, Decimal) or tsl_activation_price.is_nan() or tsl_activation_price <= 0:
            # logger.debug(f"No valid TSL activation price found for {position_side.upper()}. Cannot activate TSL.")
            return
        if not isinstance(tsl_distance, Decimal) or tsl_distance.is_nan() or tsl_distance <= 0:
            logger.debug(f"No valid TSL distance found for {position_side.upper()}. Cannot activate TSL.")
            return

        # Check activation condition
        should_activate_tsl = False
        if position_side == "long" and current_price >= tsl_activation_price:
            should_activate_tsl = True
        elif position_side == "short" and current_price <= tsl_activation_price:
            should_activate_tsl = True

        if should_activate_tsl:
            logger.trade(
                f"{Fore.CYAN}TSL Activation Condition Met for {position_side.upper()}: "
                f"Current Price ({self.exchange_manager.format_price(current_price)}) "
                f"{'>=' if position_side == 'long' else '<='} "
                f"Activation Price ({self.exchange_manager.format_price(tsl_activation_price)}). "
                f"Attempting to activate TSL...{Style.RESET_ALL}"
            )

            # Activate TSL using _set_position_protection
            success = self._set_position_protection(
                position_side=position_side,
                is_tsl=True,
                tsl_distance=tsl_distance,
                tsl_activation_price=tsl_activation_price # Pass activation price for V5
            )

            if success:
                logger.trade(f"{Fore.GREEN}{Style.BRIGHT}Trailing Stop Loss activated successfully for {position_side.upper()} position.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} TSL Activated", f"{position_side.upper()} TSL activated (Dist: {tsl_distance.normalize()})")
                # Tracker state is updated within _set_position_protection
            else:
                logger.error(f"{Fore.RED}Failed to activate Trailing Stop Loss for {position_side.upper()} position.{Style.RESET_ALL}")
                termux_notify(f"{self.config.symbol} TSL Activation FAILED", f"{position_side.upper()} TSL activation failed!")
                # Protection tracker state remains unchanged (likely ACTIVE_SLTP or None)


# --- Trade Journaler Class ---
class TradeJournaler:
    """Handles logging trade events to a CSV file."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.file_path = config.journal_file_path
        self.enabled = config.enable_journaling
        self.file_exists = self.file_path.exists()
        self._initialize_file()

    def _initialize_file(self) -> None:
        """Creates the journal file and writes header if it doesn't exist."""
        if not self.enabled:
            logger.info("Trade journaling is disabled.")
            return
        try:
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            # Write header if file is new
            if not self.file_exists:
                with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "TimestampUTC", "Symbol", "Action", "Side", "Quantity",
                        "EntryPrice", "ExitPrice", "StopLoss", "TakeProfit",
                        "PnL", "Reason", "OrderID", "Notes"
                    ])
                logger.info(f"Created new trade journal file: {self.file_path}")
                self.file_exists = True # Update status
        except IOError as e:
            logger.error(f"Failed to initialize journal file {self.file_path}: {e}. Journaling disabled.")
            self.enabled = False
        except Exception as e:
            logger.error(f"Unexpected error initializing journal file: {e}. Journaling disabled.", exc_info=True)
            self.enabled = False

    def log_trade(
        self,
        action: str, # e.g., "ENTRY", "EXIT", "SL_HIT", "TP_HIT", "TSL_HIT", "SIGNAL_EXIT"
        side: Optional[str] = None, # 'long' or 'short'
        quantity: Optional[Decimal] = None,
        entry_price: Optional[Decimal] = None,
        exit_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        pnl: Optional[Decimal] = None,
        reason: Optional[str] = None,
        order_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> None:
        """Appends a trade record to the CSV journal."""
        if not self.enabled:
            return
        if not self.file_exists:
            logger.warning("Journal file does not exist, attempting re-initialization...")
            self._initialize_file()
            if not self.enabled or not self.file_exists: # Check again after re-init attempt
                 logger.error("Cannot log trade: Journal file unavailable.")
                 return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        symbol = self.config.symbol

        # Helper to format Decimals or return empty string
        def fmt(val: Optional[Decimal], precision: int = 8) -> str:
            if isinstance(val, Decimal) and not val.is_nan() and val.is_finite():
                # Basic normalization and formatting
                return f"{val.normalize():.{precision}f}"
            return ""

        try:
            with open(self.file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    symbol,
                    action or "",
                    side or "",
                    fmt(quantity, self.config.ohlcv_limit), # Use amount precision
                    fmt(entry_price, self.config.ohlcv_limit), # Use price precision
                    fmt(exit_price, self.config.ohlcv_limit),
                    fmt(stop_loss, self.config.ohlcv_limit),
                    fmt(take_profit, self.config.ohlcv_limit),
                    fmt(pnl, 8), # PnL precision
                    reason or "",
                    order_id or "",
                    notes or ""
                ])
            # logger.debug(f"Trade journal entry added: {action} {side} {symbol}")
        except IOError as e:
            logger.error(f"Error writing to journal file {self.file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing to journal: {e}", exc_info=True)


# --- Graceful Shutdown Handler ---
class GracefulShutdown:
    """Handles SIGINT and SIGTERM for graceful shutdown."""
    shutdown_requested = False
    def __init__(self) -> None:
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)

    def request_shutdown(self, signum: int, frame: Optional[types.FrameType]) -> None:
        """Sets the shutdown flag."""
        signal_name = signal.Signals(signum).name
        logger.warning(f"
{Fore.YELLOW}{Style.BRIGHT}Shutdown requested (Signal: {signal_name}). Finishing current cycle...{Style.RESET_ALL}")
        self.shutdown_requested = True

    def is_shutdown_requested(self) -> bool:
        """Checks if shutdown has been requested."""
        return self.shutdown_requested


# --- Main Trading Bot Logic ---
def run_bot() -> None:
    """Main function to run the trading bot loop."""
    console.print(Panel(
        Text(
            # ASCII Art (Optional, keep if desired)
            textwrap.dedent("""
               ____        _       _   _                  _            _         _
              |  _ \ _   _| |_ ___| | | | __ ___   ____ _| |_ ___  ___| |_ _ __ | |__   ___ _ __ ___  _ __
              | |_) | | | | __/ _ \ | | |/ _` |\ V / _` | __/ _ \/ __| __| '_ \| '_ \ / _ \ '_ ` _ \| '_ \
              |  __/| |_| | ||  __/ | | | (_| | \ V / (_| | ||  __/\__ \ |_| |_) | | | |  __/ | | | | | |_) |
              |_|   \__,_|\__\___|_|_|_|\__,_|  \_/ \__,_|\__\___||___/\__| .__/|_| |_|\___|_| |_| |_| .__/
                                                                         |_|                       |_|
            Pyrmethus - Termux Trading Spell (v4.5.8 - Enhanced)
            """),
            style="bold cyan"
        ),
        title="[bold green]Pyrmethus Neon Nexus[/]",
        border_style="blue"
    ))

    # --- Initialization ---
    shutdown_handler = GracefulShutdown()
    try:
        config = TradingConfig()
        exchange_manager = ExchangeManager(config)
        indicator_calculator = IndicatorCalculator(config)
        signal_generator = SignalGenerator(config)
        order_manager = OrderManager(config, exchange_manager)
        journaler = TradeJournaler(config)

        # Check initial connection and market data
        if not exchange_manager.exchange or not exchange_manager.market_info:
            logger.critical("Initialization failed: Exchange or Market Info unavailable. Halting.")
            sys.exit(1)

        logger.info(f"{Fore.GREEN}Initialization complete. Starting trading loop for {config.symbol}...{Style.RESET_ALL}")
        termux_notify("Pyrmethus Started", f"Monitoring {config.symbol} on {config.interval}")

    except Exception as e:
        logger.critical(f"Critical error during initialization: {e}", exc_info=True)
        termux_notify("Pyrmethus FAILED", f"Initialization error: {str(e)[:50]}")
        sys.exit(1)

    # --- Main Loop ---
    while not shutdown_handler.is_shutdown_requested():
        loop_start_time = time.monotonic()
        try:
            logger.info(f"{Fore.BLUE}--- Starting Trading Cycle @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC ---{Style.RESET_ALL}")

            # 1. Fetch Data
            logger.debug("Fetching latest OHLCV data...")
            ohlcv_df = exchange_manager.fetch_ohlcv()
            if ohlcv_df is None or ohlcv_df.empty:
                logger.warning("Could not fetch valid OHLCV data. Skipping cycle.")
                time.sleep(config.loop_sleep_seconds)
                continue

            # Ensure we have at least 2 candles for price comparison etc.
            if len(ohlcv_df) < 2:
                 logger.warning(f"Insufficient candle data ({len(ohlcv_df)}). Skipping cycle.")
                 time.sleep(config.loop_sleep_seconds)
                 continue

            latest_candle = ohlcv_df.iloc[-1]
            current_price = safe_decimal(latest_candle["close"])
            if current_price.is_nan() or current_price <= 0:
                 logger.warning(f"Invalid current price ({current_price}) from latest candle. Skipping cycle.")
                 time.sleep(config.loop_sleep_seconds)
                 continue

            # 2. Calculate Indicators
            logger.debug("Calculating indicators...")
            indicators = indicator_calculator.calculate_indicators(ohlcv_df)
            if not indicators:
                logger.warning("Could not calculate indicators. Skipping cycle.")
                time.sleep(config.loop_sleep_seconds)
                continue

            # Display current status (optional)
            # display_status(current_price, indicators, ...) # Implement a display function if needed

            # 3. Check Position Status
            logger.debug("Checking current position status...")
            current_positions = exchange_manager.get_current_position()
            if current_positions is None:
                logger.warning("Could not fetch current position status. Skipping cycle.")
                time.sleep(config.loop_sleep_seconds)
                continue

            active_position_side: Optional[str] = None
            active_position_data: Optional[Dict[str, Any]] = None
            if current_positions.get("long"):
                active_position_side = "long"
                active_position_data = current_positions["long"]
            elif current_positions.get("short"):
                active_position_side = "short"
                active_position_data = current_positions["short"]

            logger.info(f"Current State: Price={exchange_manager.format_price(current_price)}, Position={'FLAT' if not active_position_side else active_position_side.upper()}")

            # --- Decision Logic ---
            if active_position_side and active_position_data:
                # --- In Position ---
                logger.debug(f"Currently in {active_position_side.upper()} position. Checking exit conditions...")

                # a. Check for Signal-Based Exit
                exit_signal_reason = signal_generator.check_exit_signals(active_position_side, indicators)
                if exit_signal_reason:
                    logger.trade(f"{Fore.YELLOW}Attempting to close {active_position_side.upper()} position due to: {exit_signal_reason}{Style.RESET_ALL}")
                    close_success = order_manager.close_position(active_position_side, exit_signal_reason)
                    if close_success:
                        # Log exit to journal
                        journaler.log_trade(
                            action="SIGNAL_EXIT", side=active_position_side,
                            entry_price=active_position_data.get("entry_price"),
                            exit_price=current_price, # Approximate exit price
                            reason=exit_signal_reason
                        )
                    # Continue to next cycle after attempting close
                    time.sleep(config.loop_sleep_seconds)
                    continue

                # b. Check and Manage Trailing Stop Loss
                # Only manage TSL if not exited by signal
                logger.debug(f"Checking TSL for {active_position_side.upper()} position...")
                order_manager.check_and_manage_trailing_stop(
                    position_data=active_position_data,
                    position_side=active_position_side,
                    current_price=current_price,
                    indicators=indicators # Pass indicators if needed for TSL logic
                )
                # Note: SL/TP hits are handled by the exchange, bot only detects the resulting flat state in the next cycle.
                # Could add checks here to see if SL/TP prices were breached by current_price, but exchange state is more reliable.

            else:
                # --- Flat ---
                logger.debug("Currently flat. Checking for entry signals...")

                # a. Generate Entry Signals
                entry_signals = signal_generator.generate_signals(ohlcv_df.tail(2), indicators) # Pass last 2 candles

                # b. Check Balance and Calculate Parameters if Signal Found
                entry_side: Optional[str] = None
                if entry_signals.get("long"):
                    entry_side = "buy"
                elif entry_signals.get("short"):
                    entry_side = "sell"

                if entry_side:
                    logger.trade(f"{Fore.CYAN}Entry signal detected: {entry_side.upper()}. Reason: {entry_signals.get('reason')}{Style.RESET_ALL}")
                    logger.debug("Fetching balance before entry...")
                    total_equity, available_balance = exchange_manager.get_balance()

                    if total_equity is None or total_equity.is_nan() or total_equity <= 0:
                        logger.error("Cannot place entry order: Invalid or zero equity detected.")
                    elif available_balance is None or available_balance.is_nan():
                         logger.error("Cannot place entry order: Failed to fetch available balance.")
                    else:
                        # Fetch ATR from indicators (ensure it's valid)
                        atr = indicators.get("atr")
                        if not isinstance(atr, Decimal) or atr.is_nan() or atr <= 0 or not atr.is_finite():
                             logger.error(f"Cannot place entry order: Invalid ATR value ({atr}).")
                        else:
                            # c. Place Risked Market Order
                            entry_success, final_pos_data = order_manager.place_risked_market_order(
                                side=entry_side,
                                atr=atr,
                                total_equity=total_equity,
                                current_price=current_price
                            )

                            if entry_success and final_pos_data:
                                # Log entry to journal
                                journaler.log_trade(
                                    action="ENTRY",
                                    side="long" if entry_side == "buy" else "short",
                                    quantity=final_pos_data.get("qty"),
                                    entry_price=final_pos_data.get("entry_price"),
                                    stop_loss=final_pos_data.get("stop_loss_price"),
                                    take_profit=final_pos_data.get("take_profit_price"),
                                    reason=entry_signals.get("reason")
                                )
                            else:
                                logger.error(f"Entry attempt for {entry_side.upper()} failed.")
                                # Failure handling is done within place_risked_market_order

                else:
                    logger.debug(f"No entry signal generated. Reason: {entry_signals.get('reason')}")

            # --- End of Cycle ---
            logger.info(f"{Fore.BLUE}--- Trading Cycle Ended ---{Style.RESET_ALL}")

        except ccxt.AuthenticationError as e:
            logger.critical(f"Authentication Error during cycle: {e}. Halting.", exc_info=False)
            termux_notify("Pyrmethus HALTED", "Authentication Error")
            break # Exit loop
        except ccxt.NetworkError as e:
            logger.error(f"Network Error during cycle: {e}. Continuing...", exc_info=False)
            # Allow loop to continue, rely on fetch_with_retries for subsequent calls
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange Error during cycle: {e}. Continuing...", exc_info=True)
            # Allow loop to continue
        except Exception as e:
            logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
            # Allow loop to continue, but log the error
            # Consider adding a counter for consecutive errors to halt if too many occur

        finally:
            # Calculate time elapsed and sleep if necessary
            elapsed_time = time.monotonic() - loop_start_time
            sleep_duration = max(0, config.loop_sleep_seconds - elapsed_time)
            if not shutdown_handler.is_shutdown_requested():
                logger.debug(f"Cycle took {elapsed_time:.2f}s. Sleeping for {sleep_duration:.2f}s...")
                time.sleep(sleep_duration)
            else:
                 logger.info("Shutdown requested, skipping sleep.")

    # --- Shutdown Sequence ---
    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}Pyrmethus shutting down...{Style.RESET_ALL}")
    # Add any cleanup tasks here (e.g., ensuring positions are closed if configured)
    # Example: Check if still in position and close if desired on shutdown
    # current_positions = exchange_manager.get_current_position()
    # if current_positions:
    #     # Check and close logic...
    termux_notify("Pyrmethus Stopped", "Trading loop terminated.")
    logger.info("Shutdown complete. Farewell!")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        run_bot()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting.")
    except Exception as main_exc:
        logger.critical(f"Critical unhandled exception at top level: {main_exc}", exc_info=True)
        termux_notify("Pyrmethus CRASHED", f"Fatal error: {str(main_exc)[:50]}")
        sys.exit(1)
    finally:
        # Ensure colorama is reset on exit
        print(Style.RESET_ALL)