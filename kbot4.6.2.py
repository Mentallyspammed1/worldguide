# -*- coding: utf-8 -*-
# pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-locals, too-many-public-methods
# Note: Some pylint disables are kept where refactoring significantly alters structure or standard patterns
#       (e.g., config attributes, complex methods like run_bot). Others like line-too-long,
#       wrong-import-order/position, unnecessary items, missing-final-newline have been addressed.
# Future considerations: Break down large classes/methods further if complexity increases.
"""
Pyrmethus - Termux Trading Spell (v4.5.8 - Enhanced Edition mk II)

Conjures market insights and executes trades on Bybit Futures using the
V5 Unified Account API via CCXT. Refactored into classes for better structure
and utilizing V5 position-based stop-loss/take-profit/trailing-stop features.

Enhancements in this version (4.5.8 - Enhanced mk II):
- Comprehensive code analysis and upgrade based on v4.5.8.
- Improved Readability: Enhanced PEP8 compliance, clearer names, improved comments/docstrings, extensive type hinting.
- Improved Maintainability: Further reduced pylint disables where practical, refined class responsibilities, enhanced configuration validation, improved helper functions.
- Improved Efficiency: Minor optimizations in data handling, calculations, and string formatting (f-strings).
- Improved Error Handling: More specific exceptions, refined retry logic, robust NaN/API response handling, clearer logging on errors. Added specific checks for division by zero and invalid inputs.
- Improved Security: Standard practices maintained (API keys via env). Secrets are masked in logs.
- Modern Python Practices: pathlib, f-strings, logging best practices, extensive type hinting (`typing`, `cast`), `from __future__ import annotations`.
- Refined Graceful Shutdown handler.
- Refined Trade Journaling: Uses market precision for logging prices/quantities.
- Refined Core Trading Loop: Clearer state management, improved logging, robust checks for data validity.
- Refined `close_position` and `check_and_manage_trailing_stop` methods, including pre-emptive stop cancellation.
- Refined V5 API interactions: Robust balance parsing, position parsing (handling hedge modes, side 'None'), protection setting (using private methods safely), parameter formatting helper.
- Integrated and refined previous snippets (Config, Retry, Indicator Conversion, V5 Param, Verification).
- Corrected colorama usage and refined console output using Rich.
- Added more specific checks for sufficient data length for indicators.
- Refined quantity calculation for linear/inverse contracts.
- Improved verification logic (`_verify_position_state`) with clearer reporting.
- Addressed potential division-by-zero errors in calculations.
- Added basic status display placeholder in main loop.
"""
from __future__ import annotations # Enables postponed evaluation of annotations

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
    # Disable Rich logging interception to use standard Python logging configuration
    console = Console(log_path=False)

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
                f"{Style.BRIGHT}pkg install python python-pandas python-numpy && "
                f"pip install {' '.join([p for p in COMMON_PACKAGES if p not in ['pandas', 'numpy']])}"
                f"{Style.RESET_ALL}"
            )
            print(
                f"{Fore.YELLOW}Note: pandas and numpy often installed via pkg in Termux.{Style.RESET_ALL}")
        else:
            print(
                f"{Style.BRIGHT}pip install {' '.join(COMMON_PACKAGES)}{Style.RESET_ALL}")
    except Exception: # Fallback to plain print if colorama init failed
         print(f"Missing essential spell component: {e.name}")
         print(f"To conjure it, cast: pip install {e.name}")
         print("Or, to ensure all scrolls are present, cast:")
         print("pip install ccxt python-dotenv pandas numpy rich colorama requests")
    sys.exit(1)


# --- Constants ---
# Precision & Thresholds
DECIMAL_PRECISION = 50 # Set precision for Decimal context
POSITION_QTY_EPSILON = Decimal("1E-12")  # Threshold for considering a position 'flat'
DEFAULT_PRICE_DP = 4 # Default decimal places for price formatting if market info fails
DEFAULT_AMOUNT_DP = 6 # Default decimal places for amount formatting if market info fails

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
    # Use typing.cast for complex type scenarios or ignore for simple assignments
    # logging.Logger.trade = trade_log # type: ignore[attr-defined]
    setattr(logging.Logger, "trade", trade_log)


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
        dec_val = Decimal(str_value)
        # Explicitly check for non-finite values (NaN, Infinity) created by Decimal
        if not dec_val.is_finite():
            return default
        return dec_val
    except (InvalidOperation, ValueError, TypeError):
        # logger.debug(f"Could not convert '{value}' (type: {type(value).__name__}) to Decimal, using default {default}") # Optional debug
        return default

def termux_notify(title: str, content: str) -> None:
    """Sends a notification via Termux API (toast), if available."""
    # Check if running in Termux environment more reliably
    if "com.termux" in os.environ.get("PREFIX", ""):
        try:
            # termux-toast expects only the content argument
            # Use subprocess.run for better control and error handling
            result = subprocess.run(
                ["termux-toast", content],
                check=False, # Don't raise exception on non-zero exit
                timeout=TERMUX_NOTIFY_TIMEOUT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # Log stderr if available, otherwise stdout for error info
                error_output = result.stderr or result.stdout
                logger.warning(
                    f"Termux toast command failed (code {result.returncode}): {error_output.strip()}"
                )
            # logger.debug(f"Termux toast sent: '{content}' (Title '{title}' ignored by toast)")
        except FileNotFoundError:
            logger.warning(
                "Termux notify failed: 'termux-toast' command not found. Is Termux:API installed and configured?"
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Termux notify failed: command timed out after {TERMUX_NOTIFY_TIMEOUT} seconds.")
        except Exception as e:
            logger.warning(f"Termux notify failed unexpectedly: {e}", exc_info=True)
    # else: logger.debug("Not in Termux environment, skipping notification.") # Optional debug for non-Termux

def fetch_with_retries(
    fetch_function: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay_seconds: int = DEFAULT_RETRY_DELAY,
    retry_on_exceptions: tuple[Type[Exception], ...] = (
        ccxt.DDoSProtection, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable,
        ccxt.NetworkError, ccxt.RateLimitExceeded, requests.exceptions.ConnectionError,
        requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout, # Added ReadTimeout
    ),
    fatal_exceptions: tuple[Type[Exception], ...] = (
        ccxt.AuthenticationError, ccxt.PermissionDenied
    ),
    fail_fast_exceptions: tuple[Type[Exception], ...] = (
         ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.OrderNotFound
    ),
    **kwargs: Any,
) -> Any:
    """Wraps a function call with enhanced retry logic and error handling.

    Args:
        fetch_function: The function/method to call.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retry attempts (0 means one initial attempt).
        delay_seconds: Delay between retries in seconds.
        retry_on_exceptions: Tuple of exception types to retry on.
        fatal_exceptions: Tuple of exception types that should halt immediately.
        fail_fast_exceptions: Tuple of exception types that should not be retried but not halt the bot.
        **kwargs: Keyword arguments for the function.

    Returns:
        The result of the fetch_function call.

    Raises:
        Exception: The last exception encountered if all retries fail or a fatal/fail-fast exception occurs.
    """
    last_exception: Optional[Exception] = None
    func_name = getattr(fetch_function, "__name__", "Unnamed function")

    for attempt in range(max_retries + 1):
        try:
            result = fetch_function(*args, **kwargs)
            # Log success only if there were previous failures
            if attempt > 0:
                logger.info(f"{Fore.GREEN}{Style.BRIGHT}Successfully executed {func_name} on attempt {attempt + 1}/{max_retries + 1}.{Style.RESET_ALL}")
            return result
        except fatal_exceptions as e:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}Fatal error ({type(e).__name__}) executing {func_name}: {e}. Halting immediately.{Style.RESET_ALL}", exc_info=False)
            raise e # Re-raise critical error to stop the bot
        except fail_fast_exceptions as e:
            logger.error(f"{Fore.RED}Fail-fast error ({type(e).__name__}) executing {func_name}: {e}. Not retrying.{Style.RESET_ALL}")
            last_exception = e
            break # Break loop, don't retry, exception raised below
        except retry_on_exceptions as e:
            last_exception = e
            # Truncate potentially long error messages
            error_str = str(e)
            error_display = (error_str[:150] + '...') if len(error_str) > 150 else error_str
            retry_msg = f"{Fore.YELLOW}Retryable error ({type(e).__name__}) on attempt {attempt + 1}/{max_retries + 1} for {func_name}: {error_display}.{Style.RESET_ALL}"
            if attempt < max_retries:
                logger.warning(f"{retry_msg} Retrying in {delay_seconds}s...")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries ({max_retries + 1}) reached for {func_name} after retryable error. Last error: {e}{Style.RESET_ALL}")
                # Loop ends, last_exception raised below
        except ccxt.ExchangeError as e: # Catch other specific exchange errors
            last_exception = e
            # Decide if specific ExchangeErrors are retryable - here we retry generic ones
            # You could add specific ExchangeError subclasses to retry_on_exceptions if needed
            error_str = str(e)
            error_display = (error_str[:150] + '...') if len(error_str) > 150 else error_str
            logger.error(f"{Fore.RED}Unhandled Exchange error during {func_name}: {error_display}{Style.RESET_ALL}")
            if attempt < max_retries:
                logger.warning(f"{Fore.YELLOW}Retrying generic exchange error in {delay_seconds}s...{Style.RESET_ALL}")
                time.sleep(delay_seconds)
            else:
                logger.error(f"{Fore.RED}Max retries reached after exchange error for {func_name}.{Style.RESET_ALL}")
                break
        except Exception as e: # Catch truly unexpected errors
            last_exception = e
            logger.error(f"{Fore.RED}Unexpected error during {func_name}: {e}{Style.RESET_ALL}", exc_info=True)
            # For unexpected errors, it's usually safer to fail fast.
            break # Don't retry unknown errors

    # If loop finished without returning, raise the last captured exception
    if last_exception:
        raise last_exception
    else:
        # This path should ideally not be hit if max_retries >= 0 and logic is correct
        # It might occur if fetch_function returns None unexpectedly without raising.
        raise RuntimeError(f"Function {func_name} failed after {max_retries + 1} attempts without raising a recognized exception.")


# --- Configuration Class ---
class TradingConfig:
    """Loads, validates, and holds trading configuration parameters from environment variables or .env file."""

    # pylint: disable=too-many-instance-attributes # Acceptable for a config object holding many params
    def __init__(self, env_file: Union[str, Path] = DEFAULT_ENV_FILE) -> None:
        """Initializes configuration by loading and validating parameters."""
        logger.debug(f"Loading configuration from environment variables / {env_file}...")
        env_path = Path(env_file)
        if env_path.is_file():
            load_dotenv(dotenv_path=env_path, override=True) # Env vars take precedence
            logger.info(f"Loaded configuration settings from '{env_path}'")
        else:
            logger.warning(f"Environment file '{env_path}' not found. Relying solely on system environment variables.")

        # --- Load Core Trading Parameters ---
        self.symbol: str = self._get_env("SYMBOL", "BTC/USDT:USDT", Style.DIM)
        self.market_type: str = self._get_env(
            "MARKET_TYPE",
            "linear", # Default to linear as it's common (USDT perps)
            Style.DIM,
            allowed_values=["linear", "inverse"],
        ).lower()
        # V5 Category is determined after market_type and symbol are loaded
        self.bybit_v5_category: str = self._determine_v5_category()
        self.interval: str = self._get_env("INTERVAL", "1m", Style.DIM)

        # --- Load Financial Parameters (Decimal) ---
        self.risk_percentage: Decimal = self._get_env(
            "RISK_PERCENTAGE",
            DEFAULT_RISK_PERCENT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.00001"), # Allow very small risk
            max_val=Decimal("0.5"), # Cap risk at 50% (high!)
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
            min_val=Decimal("0.0"), # Allow 0 to disable ATR-based TSL activation
            max_val=Decimal("20.0"),
        )
        self.trailing_stop_percent: Decimal = self._get_env(
            "TRAILING_STOP_PERCENT",
            DEFAULT_TSL_PERCENT,
            Fore.YELLOW,
            cast_type=Decimal,
            min_val=Decimal("0.01"), # Minimum 0.01% TSL distance
            max_val=Decimal("10.0"), # Maximum 10% TSL distance
        )

        # --- Load V5 Position Stop Parameters ---
        self.sl_trigger_by: str = self._get_env(
            "SL_TRIGGER_BY",
            "LastPrice", # Bybit default
            Style.DIM,
            allowed_values=["LastPrice", "MarkPrice", "IndexPrice"],
        )
        self.tsl_trigger_by: str = self._get_env(
            "TSL_TRIGGER_BY",
            "LastPrice", # Bybit default
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

        # --- Load Indicator Periods (int) ---
        self.trend_ema_period: int = self._get_env("TREND_EMA_PERIOD", 12, Style.DIM, cast_type=int, min_val=5, max_val=500)
        self.fast_ema_period: int = self._get_env("FAST_EMA_PERIOD", 9, Style.DIM, cast_type=int, min_val=1, max_val=200)
        self.slow_ema_period: int = self._get_env("SLOW_EMA_PERIOD", 21, Style.DIM, cast_type=int, min_val=2, max_val=500)
        self.stoch_period: int = self._get_env("STOCH_PERIOD", 7, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.stoch_smooth_k: int = self._get_env("STOCH_SMOOTH_K", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.stoch_smooth_d: int = self._get_env("STOCH_SMOOTH_D", 3, Style.DIM, cast_type=int, min_val=1, max_val=10)
        self.atr_period: int = self._get_env("ATR_PERIOD", 5, Style.DIM, cast_type=int, min_val=1, max_val=100)
        self.adx_period: int = self._get_env("ADX_PERIOD", 14, Style.DIM, cast_type=int, min_val=2, max_val=100)

        # --- Load Signal Logic Thresholds (Decimal) ---
        self.stoch_oversold_threshold: Decimal = self._get_env("STOCH_OVERSOLD_THRESHOLD", DEFAULT_STOCH_OVERSOLD, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("45"))
        self.stoch_overbought_threshold: Decimal = self._get_env("STOCH_OVERBOUGHT_THRESHOLD", DEFAULT_STOCH_OVERBOUGHT, Fore.CYAN, cast_type=Decimal, min_val=Decimal("55"), max_val=Decimal("100"))
        self.trend_filter_buffer_percent: Decimal = self._get_env("TREND_FILTER_BUFFER_PERCENT", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5"))
        self.atr_move_filter_multiplier: Decimal = self._get_env("ATR_MOVE_FILTER_MULTIPLIER", Decimal("0.5"), Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("5")) # 0 disables filter
        self.min_adx_level: Decimal = self._get_env("MIN_ADX_LEVEL", DEFAULT_MIN_ADX, Fore.CYAN, cast_type=Decimal, min_val=Decimal("0"), max_val=Decimal("90"))

        # --- Load API Keys (Secrets) ---
        self.api_key: str = self._get_env("BYBIT_API_KEY", None, Fore.RED, is_secret=True)
        self.api_secret: str = self._get_env("BYBIT_API_SECRET", None, Fore.RED, is_secret=True)

        # --- Load Operational Parameters ---
        self.ohlcv_limit: int = self._get_env("OHLCV_LIMIT", DEFAULT_OHLCV_LIMIT, Style.DIM, cast_type=int, min_val=50, max_val=1000)
        self.loop_sleep_seconds: int = self._get_env("LOOP_SLEEP_SECONDS", DEFAULT_LOOP_SLEEP, Style.DIM, cast_type=int, min_val=1) # Allow 1s min sleep
        self.order_check_delay_seconds: int = self._get_env("ORDER_CHECK_DELAY_SECONDS", 2, Style.DIM, cast_type=int, min_val=1)
        self.order_fill_timeout_seconds: int = self._get_env("ORDER_FILL_TIMEOUT_SECONDS", 20, Style.DIM, cast_type=int, min_val=5) # Used in verification attempts implicitly
        self.max_fetch_retries: int = self._get_env("MAX_FETCH_RETRIES", DEFAULT_MAX_RETRIES, Style.DIM, cast_type=int, min_val=0, max_val=10) # Allow 0 retries
        self.retry_delay_seconds: int = self._get_env("RETRY_DELAY_SECONDS", DEFAULT_RETRY_DELAY, Style.DIM, cast_type=int, min_val=1)
        self.trade_only_with_trend: bool = self._get_env("TRADE_ONLY_WITH_TREND", True, Style.DIM, cast_type=bool)

        # --- Load Journaling Parameters ---
        self.journal_file_path: Path = Path(self._get_env("JOURNAL_FILE_PATH", DEFAULT_JOURNAL_FILE, Style.DIM))
        self.enable_journaling: bool = self._get_env("ENABLE_JOURNALING", True, Style.DIM, cast_type=bool)

        # --- Final Validation Checks ---
        if not self.api_key or not self.api_secret:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}BYBIT_API_KEY or BYBIT_API_SECRET not found in environment or .env file. Halting.{Style.RESET_ALL}"
            )
            sys.exit(1)
        # Run post-load validation checks
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
                category = "linear" # e.g., BTC/USDT:USDT
            # Add 'spot' or 'option' if needed in the future
            # elif self.market_type == "spot": category = "spot"
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
                exc_info=True, # Log traceback for value errors during determination
            )
            sys.exit(1)

    def _validate_config(self) -> None:
        """Performs post-load validation of configuration parameters."""
        # Check relationships between parameters
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
        # Only warn if TSL activation is enabled (multiplier > 0) and less than SL
        if self.tsl_activation_atr_multiplier > 0 and self.tsl_activation_atr_multiplier < self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TSL_ACT_MULT ({self.tsl_activation_atr_multiplier.normalize()}) < SL_MULT ({self.sl_atr_multiplier.normalize()}). TSL may activate before initial SL distance is reached.{Style.RESET_ALL}"
            )
        # Check TP vs SL only if TP is enabled (multiplier > 0)
        if self.tp_atr_multiplier > Decimal("0") and self.tp_atr_multiplier <= self.sl_atr_multiplier:
            logger.warning(
                f"{Fore.YELLOW}Config Warning: TP_MULT ({self.tp_atr_multiplier.normalize()}) <= SL_MULT ({self.sl_atr_multiplier.normalize()}). This implies a poor Risk:Reward setup (<= 1:1).{Style.RESET_ALL}"
            )
        # Ensure journal file directory exists if journaling enabled
        if self.enable_journaling:
            try:
                self.journal_file_path.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured journal directory exists: {self.journal_file_path.parent}")
            except OSError as e:
                logger.error(f"Failed to create directory for journal file {self.journal_file_path}: {e}. Journaling disabled.")
                self.enable_journaling = False

    def _cast_value(self, key: str, value_str: str, cast_type: Type, default: Any) -> Any:
        """Helper to cast string value to target type, returning default on failure."""
        val_to_cast = value_str.strip()
        if not val_to_cast: # Handle empty string after strip
            logger.warning(f"Empty value string provided for {key}. Using default '{default}'.")
            return default
        try:
            if cast_type == bool:
                # Robust boolean casting
                return val_to_cast.lower() in ["true", "1", "yes", "y", "on"]
            elif cast_type == Decimal:
                # Use safe_decimal for robust Decimal conversion, check for NaN result
                dec_val = safe_decimal(val_to_cast, default=Decimal("NaN"))
                if dec_val.is_nan():
                    raise ValueError("Non-numeric or invalid string cannot be cast to Decimal")
                return dec_val
            elif cast_type == int:
                # Use Decimal intermediary for robustness (handles "10.0") then check integrity
                dec_val = Decimal(val_to_cast)
                # Check if it has a fractional part or is non-finite before casting to int
                if dec_val.as_tuple().exponent < 0 or not dec_val.is_finite():
                    raise ValueError("Non-integer Decimal value cannot be cast to int")
                return int(dec_val)
            elif cast_type == float:
                # Handle potential non-numeric strings for float as well
                if val_to_cast.lower() in ["nan", "none", "null"]:
                     raise ValueError("Non-numeric string cannot be cast to float")
                return float(val_to_cast)
            else: # Includes str type
                return cast_type(val_to_cast) # Use constructor directly for other types
        except (ValueError, TypeError, InvalidOperation) as e:
            logger.error(
                f"{Fore.RED}Cast failed for {key} ('{value_str}' -> {cast_type.__name__}): {e}. Using default '{default}'.{Style.RESET_ALL}"
            )
            return default

    def _validate_value(
            self,
            key: str,
            value: Any,
            min_val: Optional[Union[int, float, Decimal]],
            max_val: Optional[Union[int, float, Decimal]],
            allowed_values: Optional[List[Any]]
        ) -> bool:
        """Helper to validate value against constraints. Logs and returns False on failure.
           Critical min/max violations cause exit.
        """
        # Check for non-finite Decimal values before comparisons
        if isinstance(value, Decimal) and not value.is_finite():
             logger.error(f"Validation failed for {key}: Non-finite Decimal value '{value}'. Cannot perform checks.")
             return False # Treat non-finite Decimals as invalid for range/allowed checks

        is_numeric = isinstance(value, (int, float, Decimal))

        # Min/Max checks (only if value is numeric)
        if is_numeric:
            # Use Decimal for comparison if min/max are provided, converting them if needed
            min_dec = safe_decimal(min_val) if min_val is not None else None
            max_dec = safe_decimal(max_val) if max_val is not None else None
            value_dec = safe_decimal(value) # Value is already numeric, convert for consistent comparison

            if min_dec is not None and not min_dec.is_nan() and value_dec < min_dec:
                logger.critical(f"{Fore.RED}{Style.BRIGHT}Validation failed for {key}: Value '{value}' < minimum '{min_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1) # Critical failure
            if max_dec is not None and not max_dec.is_nan() and value_dec > max_dec:
                logger.critical(f"{Fore.RED}{Style.BRIGHT}Validation failed for {key}: Value '{value}' > maximum '{max_val}'. Halting.{Style.RESET_ALL}")
                sys.exit(1) # Critical failure
        elif (min_val is not None or max_val is not None):
             # If min/max is set, but value isn't numeric, log an error (should have been caught by cast usually)
             logger.error(f"Validation failed for {key}: Non-numeric value '{value}' ({type(value).__name__}) cannot be compared with min/max.")
             return False

        # Allowed values check
        if allowed_values:
            # Handle case-insensitive string comparison if value is string
            comp_value = value.lower() if isinstance(value, str) else value
            lower_allowed = [str(v).lower() if isinstance(v, str) else v for v in allowed_values]
            if comp_value not in lower_allowed:
                logger.error(f"{Fore.RED}Validation failed for {key}: Invalid value '{value}'. Allowed values: {allowed_values}.{Style.RESET_ALL}")
                return False # Non-critical failure (will revert to default later)

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
            if default is None and not is_secret: # Required non-secret is missing
                 log_msg = f"{Fore.RED}{Style.BRIGHT}Required configuration '{key}' not found in environment or .env, and no default provided. Halting.{Style.RESET_ALL}"
                 logger.critical(log_msg)
                 sys.exit(1)
            elif default is None and is_secret:
                 # Secret is missing, will be checked finally in __init__
                 value_str_to_process = "" # Process empty string for secret logic
                 log_value_display = "[Not Set - CRITICAL]"
                 source = "missing"
                 # Log as warning initially, critical check later
                 log_method = logger.warning
            else:
                 # Use default value
                 use_default = True
                 value_str_to_process = str(default) # Use string representation of default for casting/logging
                 source = f"default ('{default}')"
                 log_value_display = default # Display original default type/value
                 log_method = logger.info # Log using default is usually INFO level unless it's a sensitive default
        else:
            # Value found in environment
            value_str_to_process = value_str
            log_value_display = "****" if is_secret else value_str_to_process
            log_method = logger.info

        # Log the found/default value before casting/validation
        log_method(f"{color}Using {key}: {log_value_display} (Source: {source}){Style.RESET_ALL}")

        # Handle case where secret is missing (value_str_to_process is "")
        if is_secret and not value_str_to_process:
             # Return None for missing secrets; the final check happens in __init__
             return None

        # Attempt to cast the value string (from env or default)
        casted_value = self._cast_value(key, value_str_to_process, cast_type, default)

        # Validate the casted value
        if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
            # Validation failed (min/max would have exited). This usually means allowed_values failed or type error.
            # If the value came from the environment, revert to the original default.
            # If the default itself failed casting/validation, we have a bigger problem.
            if not use_default:
                logger.warning(
                    f"{color}Reverting {key} to default '{default}' due to validation failure of environment value '{value_str_to_process}' (casted: '{casted_value}').{Style.RESET_ALL}"
                )
                # Recast the original default value
                casted_value = self._cast_value(key, str(default), cast_type, default)
                # Re-validate the default value itself - critical if this fails
                if not self._validate_value(key, casted_value, min_val, max_val, allowed_values):
                    logger.critical(
                        f"{Fore.RED}{Style.BRIGHT}FATAL: Default value '{default}' for {key} also failed validation. Halting.{Style.RESET_ALL}"
                    )
                    sys.exit(1)
            else:
                 # This means the default value itself failed the _validate_value check (but not min/max exit)
                 # This case should be rare if defaults are set correctly, but handle defensively.
                 logger.critical(
                     f"{Fore.RED}{Style.BRIGHT}FATAL: Default value '{default}' for {key} failed validation (e.g., allowed_values check). Halting.{Style.RESET_ALL}"
                 )
                 sys.exit(1)

        return casted_value


# --- Exchange Manager Class ---
class ExchangeManager:
    """Handles CCXT exchange interactions, data fetching, formatting, and market info caching."""

    def __init__(self, config: TradingConfig) -> None:
        """Initializes the exchange manager, connects to Bybit, and loads market info."""
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.market_info: Optional[Dict[str, Any]] = None
        self._initialize_exchange()
        # Load market info only if exchange initialization was successful
        if self.exchange:
             self.market_info = self._load_market_info()
        # Critical failures are logged and sys.exit is called within the init/load methods

    def _initialize_exchange(self) -> None:
        """Initializes the CCXT Bybit exchange instance with V5 configuration."""
        logger.info(f"Initializing Bybit exchange interface (V5 {self.config.market_type} / {self.config.bybit_v5_category})...")
        try:
            # Check if ccxt.bybit exists
            if not hasattr(ccxt, 'bybit'):
                logger.critical(f"{Fore.RED}{Style.BRIGHT}CCXT 'bybit' exchange class not found. Update CCXT? Halting.{Style.RESET_ALL}")
                sys.exit(1)

            exchange_params = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "options": {
                    "defaultType": self.config.market_type, # 'linear' or 'inverse'
                    "adjustForTimeDifference": True, # Auto-adjust time difference
                    "recvWindow": 10000, # Optional: Increase recvWindow if needed (default 5000)
                    "brokerId": "PyrmethusNeonV5", # Custom ID for tracking via Bybit referrals
                    "createMarketBuyOrderRequiresPrice": False, # V5 market orders don't need price
                    "defaultTimeInForce": "GTC", # Good-Till-Cancelled default for limit orders
                    # Ensure V5 API is targeted if CCXT doesn't default correctly
                    'api-expires': str(int(time.time() * 1000 + 10000)), # Required for V5 signatures? Check CCXT handling. Usually handled internally.
                    'enableRateLimit': True, # Use CCXT's built-in rate limiter
                    # Add category to options if needed for specific endpoints CCXT doesn't handle via defaultType
                    # 'defaultCategory': self.config.bybit_v5_category, # Potentially useful
                },
                # Handle sandbox endpoint via environment variable
                'urls': {'api': 'https://api-testnet.bybit.com'} if os.getenv("USE_SANDBOX", "false").lower() == "true" else {}
            }
            if exchange_params['urls']:
                logger.warning(f"{Fore.YELLOW}Using Bybit Testnet API endpoint.{Style.RESET_ALL}")

            self.exchange = ccxt.bybit(exchange_params)

            # Test connectivity by fetching server time
            logger.debug("Testing exchange connection by fetching server time...")
            server_time = self.exchange.fetch_time()
            logger.debug(f"Exchange server time fetched: {datetime.fromtimestamp(server_time / 1000, tz=timezone.utc)}")
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
                exc_info=True, # Log traceback for network issues
            )
            sys.exit(1)
        except Exception as e:
            logger.critical(
                f"{Fore.RED}{Style.BRIGHT}Unexpected error initializing exchange: {e}. Halting.{Style.RESET_ALL}",
                exc_info=True,
            )
            sys.exit(1)

    def _load_market_info(self) -> Optional[Dict[str, Any]]:
        """Loads and caches market information for the configured symbol, calculating precision details."""
        if not self.exchange:
            # This case should ideally not be reached if _initialize_exchange fails critically
            logger.error("Exchange not initialized, cannot load market info.")
            return None
        try:
            logger.info(f"Loading market info for {self.config.symbol}...")
            # Force reload to ensure up-to-date info, especially precision and limits
            self.exchange.load_markets(reload=True) # Use reload=True
            market = self.exchange.market(self.config.symbol)

            if not market:
                # This shouldn't happen if load_markets succeeded and symbol is valid, but check anyway
                raise ccxt.ExchangeError(
                    f"Market {self.config.symbol} not found on exchange after loading markets. Check SYMBOL config."
                )

            # Safely extract precision and convert to integer decimal places (dp)
            amount_precision_raw = market.get("precision", {}).get("amount")
            price_precision_raw = market.get("precision", {}).get("price")

            # --- Helper to determine Decimal Places (DP) from precision value ---
            def get_dp_from_precision(precision_val: Optional[Union[str, float, int]], default_dp: int) -> int:
                """Converts CCXT precision (step size like 0.01 or DP like 2) to integer DP."""
                if precision_val is None: return default_dp
                prec_dec = safe_decimal(precision_val)
                if prec_dec.is_nan() or not prec_dec.is_finite(): return default_dp
                if prec_dec == 0: return 0 # Handle case where precision is exactly 0

                # Check if it's likely a step size (e.g., 0.01, 1e-2) -> convert to DP
                if 0 < prec_dec <= 1:
                     # Use string formatting to find decimal places for small numbers robustly
                     prec_str = f"{prec_dec.normalize():f}" # Format to plain string
                     if '.' in prec_str:
                         return len(prec_str.split('.')[-1])
                     else:
                         return 0 # e.g., precision = 1
                # Check if it's likely number of decimal places directly (integer >= 0)
                elif prec_dec >= 1 and prec_dec.as_tuple().exponent >= 0:
                    try:
                        return int(prec_dec)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert precision value {prec_dec} to integer DP, using default {default_dp}dp.")
                        return default_dp
                # Handle other cases (e.g., precision < 0 ?) - use default
                else:
                    logger.warning(f"Unexpected precision format '{precision_val}', using default {default_dp}dp.")
                    return default_dp

            amount_dp = get_dp_from_precision(amount_precision_raw, DEFAULT_AMOUNT_DP)
            price_dp = get_dp_from_precision(price_precision_raw, DEFAULT_PRICE_DP)

            # Store calculated DPs back into market info for easy access
            market["precision_dp"] = {"amount": amount_dp, "price": price_dp}

            # Store minimum tick size (price increment) as Decimal
            tick_size_raw = market.get("precision", {}).get("price") # Often same as price precision step
            market["tick_size"] = safe_decimal(tick_size_raw, default=Decimal("1e-" + str(price_dp)))
            if market["tick_size"].is_nan() or market["tick_size"] <= 0:
                 market["tick_size"] = Decimal("1e-" + str(price_dp)) # Fallback based on DP
                 logger.warning(f"Using calculated tick size based on price DP: {market['tick_size']}")

            # Store minimum order size (base currency amount) as Decimal
            min_amount_raw = market.get("limits", {}).get("amount", {}).get("min")
            market["min_order_size"] = safe_decimal(min_amount_raw, default=Decimal("NaN"))

            # Store contract size (important for inverse contracts, usually 1 for linear)
            # Default to Decimal '1' if not specified or invalid
            market["contract_size"] = safe_decimal(market.get("contractSize", "1"), default=Decimal("1"))
            if market["contract_size"].is_nan() or market["contract_size"] <= 0:
                logger.warning(f"Invalid contract size found ('{market.get('contractSize')}'). Defaulting to 1.")
                market["contract_size"] = Decimal("1")

            # Log loaded info
            min_amt_str = (
                market["min_order_size"].normalize()
                if not market["min_order_size"].is_nan()
                else "[N/A]"
            )
            contract_size_str = market["contract_size"].normalize()
            tick_size_str = market["tick_size"].normalize()

            logger.info(
                f"Market info loaded for {market.get('symbol', '[N/A]')}: "
                f"ID={market.get('id', '[N/A]')}, "
                f"Type={market.get('type', '[N/A]')}, "
                f"Settle={market.get('settle', '[N/A]')}, "
                f"Precision(AmtDP={amount_dp}, PriceDP={price_dp}), "
                f"TickSize={tick_size_str}, "
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
        """Gets the number of decimal places for 'price' or 'amount' from cached market info."""
        default_dp = DEFAULT_PRICE_DP if precision_type == "price" else DEFAULT_AMOUNT_DP
        if self.market_info and "precision_dp" in self.market_info:
            return self.market_info["precision_dp"].get(precision_type, default_dp)
        logger.warning(f"Market info or precision_dp missing, using default DP {default_dp} for '{precision_type}'.")
        return default_dp

    def format_price(
        self, price: Union[Decimal, str, float, int]
    ) -> str:
        """Formats price according to market precision using ROUND_HALF_EVEN."""
        price_decimal = safe_decimal(price)
        if price_decimal.is_nan():
            logger.warning(f"Cannot format invalid price input: {price}")
            return "NaN" # Return NaN string if input was bad

        precision = self.get_market_precision("price")

        try:
            # Quantizer is 10^-precision (e.g., 0.0001 for 4 dp)
            quantizer = Decimal("1e-" + str(precision))
            # ROUND_HALF_EVEN is standard for financial rounding ("round to nearest, ties to even")
            formatted_price = price_decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN)
            # Ensure the string representation matches the precision exactly using f-string formatting
            return f"{formatted_price:.{precision}f}"
        except (InvalidOperation, ValueError) as e:
             logger.error(f"Error formatting price {price_decimal} to {precision}dp: {e}")
             return "ERR" # Indicate formatting error

    def format_amount(
        self,
        amount: Union[Decimal, str, float, int],
        rounding_mode: str = ROUND_DOWN, # Default to ROUND_DOWN for order quantities
    ) -> str:
        """Formats amount (quantity) according to market precision using specified rounding."""
        amount_decimal = safe_decimal(amount)
        if amount_decimal.is_nan():
            logger.warning(f"Cannot format invalid amount input: {amount}")
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
            allow_zero: bool = False # Allow "0" or "0.0..." string as a valid parameter?
        ) -> Optional[str]:
        """Formats a value as a string suitable for Bybit V5 API parameters (SL/TP/Qty etc.).
           Returns None if the value is invalid, negative, or cannot be formatted positively (unless allow_zero=True).
        """
        if value is None:
            # logger.debug(f"V5 Param Formatting: Input value is None for param_type '{param_type}'.")
            return None # None input results in None output

        # Convert to Decimal for unified handling, default to NaN if conversion fails
        decimal_value = safe_decimal(value, default=Decimal("NaN"))

        # Check for NaN or Infinity
        if decimal_value.is_nan() or not decimal_value.is_finite():
            logger.warning(f"V5 Param Formatting: Input '{value}' resulted in non-finite Decimal for param_type '{param_type}'.")
            return None # Cannot format NaN or Infinity

        # Handle zero based on allow_zero flag
        is_zero = decimal_value.is_zero()
        if is_zero:
            if allow_zero:
                # Format "0" using the appropriate precision
                precision_dp = self.get_market_precision("price" if param_type in ["price", "distance"] else "amount")
                formatted_zero = f"{Decimal('0'):.{precision_dp}f}"
                # logger.debug(f"V5 Param Formatting: Formatting zero for {param_type} with {precision_dp}dp: {formatted_zero}")
                return formatted_zero
            else:
                # logger.debug(f"V5 Param Formatting: Input value '{value}' is zero, but zero is not allowed for param_type '{param_type}'.")
                return None # Zero not allowed, return None
        elif decimal_value < 0:
            logger.warning(f"V5 Param Formatting: Input value '{value}' is negative, which is usually invalid for param_type '{param_type}'.")
            return None # Negative values usually invalid for price/amount/distance

        # Select appropriate formatter based on type
        formatter: Callable[[Union[Decimal, str, float, int]], str]
        if param_type == "price" or param_type == "distance":
            # Price/Distance uses standard price formatting (ROUND_HALF_EVEN)
            formatter = self.format_price
        elif param_type == "amount":
            # Amount uses specific amount formatting (default ROUND_DOWN)
            formatter = self.format_amount
        else:
            logger.error(f"V5 Param Formatting: Unknown param_type '{param_type}'.")
            return None

        # Format the positive, finite Decimal value
        formatted_str = formatter(decimal_value)

        # Final check: ensure the formatted string isn't an error/NaN indicator
        if formatted_str in ["ERR", "NaN"]:
            logger.error(f"V5 Param Formatting: Failed to format valid string for '{value}' (type: {param_type}). Result: {formatted_str}")
            return None

        # logger.debug(f"V5 Param Formatted: Input='{value}', Type='{param_type}', Output='{formatted_str}'")
        return formatted_str

    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data with retries, converts to DataFrame with Decimal types."""
        if not self.exchange:
            logger.error("Exchange not initialized, cannot fetch OHLCV.")
            return None
        logger.debug(
            f"Fetching {self.config.ohlcv_limit} candles for {self.config.symbol} ({self.config.interval})..."
        )
        try:
            # Pass fetch function and args/kwargs to retry wrapper
            ohlcv_list = fetch_with_retries(
                self.exchange.fetch_ohlcv, # Pass the method itself
                symbol=self.config.symbol,
                timeframe=self.config.interval,
                limit=self.config.ohlcv_limit,
                # Pass retry config from TradingConfig if desired, or use defaults
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )

            if not isinstance(ohlcv_list, list) or not ohlcv_list:
                logger.error(f"fetch_ohlcv for {self.config.symbol} returned invalid data: {ohlcv_list}")
                return None
            if len(ohlcv_list) < 10: # Check for suspiciously small amount of data
                 logger.warning(f"Fetched only {len(ohlcv_list)} candles, which might be insufficient for some indicators.")

            # Define columns based on CCXT standard response
            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame(ohlcv_list, columns=columns)

            # Convert timestamp to UTC datetime and set as index
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors='coerce')
                df.set_index("timestamp", inplace=True)
                # Drop rows where timestamp conversion failed
                rows_before_ts_drop = len(df)
                df.dropna(subset=[df.index.name], inplace=True)
                if len(df) < rows_before_ts_drop:
                    logger.warning(f"Dropped {rows_before_ts_drop - len(df)} rows with invalid timestamps.")
            except Exception as e:
                logger.error(f"Error processing timestamps in OHLCV data: {e}", exc_info=True)
                return None

            if df.empty:
                logger.error("DataFrame is empty after timestamp processing.")
                return None

            # Convert OHLCV columns to Decimal robustly using safe_decimal
            nan_cols = []
            for col in ["open", "high", "low", "close", "volume"]:
                # Use safe_decimal utility via map for potentially better performance than apply
                df[col] = df[col].map(safe_decimal)
                # Check if any conversion resulted in NaN, which indicates bad data in that column
                if df[col].apply(lambda x: isinstance(x, Decimal) and x.is_nan()).any():
                     nan_cols.append(col)

            if nan_cols:
                 logger.warning(f"Column(s) '{', '.join(nan_cols)}' contain NaN values after Decimal conversion. Check API data source.")

            # Drop rows where essential price data (OHLC) is NaN after conversion
            initial_len = len(df)
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            rows_dropped = initial_len - len(df)
            if rows_dropped > 0:
                logger.warning(f"Dropped {rows_dropped} rows with NaN OHLC Decimal values.")

            if df.empty:
                 logger.error("DataFrame became empty after processing and dropping NaN OHLCV data.")
                 return None

            logger.debug(
                f"Fetched and processed {len(df)} candles. Last timestamp: {df.index[-1]}"
            )
            return df
        except Exception as e:
            # Catch exceptions from fetch_with_retries or DataFrame processing
            logger.error(f"Failed to fetch or process OHLCV data for {self.config.symbol}: {e}", exc_info=True)
            return None

    def get_balance(self) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Fetches total equity and available balance for the settlement currency using V5 API.

        Returns:
            tuple[Optional[Decimal], Optional[Decimal]]: (total_equity, available_balance).
            Returns (None, None) or (value, None) etc. if parts of the balance cannot be determined.
        """
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch balance.")
            return None, None

        # Determine the currency to look for in the balance response (e.g., USDT, BTC)
        settle_currency = self.market_info.get("settle")
        if not settle_currency:
            logger.error("Settle currency not found in market info. Cannot determine balance currency.")
            return None, None

        logger.debug(
            f"Fetching balance for {settle_currency} (Account: {V5_UNIFIED_ACCOUNT_TYPE}, Category: {self.config.bybit_v5_category})..."
        )
        total_equity: Optional[Decimal] = None
        available_balance: Optional[Decimal] = None

        try:
            # V5 balance requires accountType. CCXT fetch_balance should handle mapping.
            # Requesting specific coin balance might be supported via params.
            params = {
                "accountType": V5_UNIFIED_ACCOUNT_TYPE,
                "coin": settle_currency, # Request specific coin balance if API supports it directly
            }
            balance_data = fetch_with_retries(
                self.exchange.fetch_balance, # Pass function
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw balance data from fetch_balance: {balance_data}") # Very verbose

            # --- Parse V5 Response Structure (Unified Account) ---
            # CCXT aims to unify the response ('total', 'free'), but V5 specifics are often in 'info'.

            # Primary check: Top-level 'total' and 'free' dicts for the settle currency.
            # These usually represent the overall account balance in that currency.
            if settle_currency in balance_data.get("total", {}):
                total_equity_raw = balance_data["total"].get(settle_currency)
                total_equity = safe_decimal(total_equity_raw)
                if total_equity.is_nan():
                    logger.warning(f"Parsed NaN total equity from top-level balance for {settle_currency}. Value: '{total_equity_raw}'")

            if settle_currency in balance_data.get("free", {}):
                available_balance_raw = balance_data["free"].get(settle_currency)
                available_balance = safe_decimal(available_balance_raw)
                if available_balance.is_nan():
                     logger.warning(f"Parsed NaN available balance from top-level balance for {settle_currency}. Value: '{available_balance_raw}'")

            # Secondary check: Look inside `info` for more detailed V5 structure if primary failed or seems off.
            # Bybit V5 /v5/account/wallet-balance structure: result.list[0].totalEquity, result.list[0].coin[N].availableToWithdraw
            needs_info_check = (total_equity is None or total_equity.is_nan()) or \
                               (available_balance is None or available_balance.is_nan())

            if needs_info_check and "info" in balance_data and isinstance(balance_data["info"], dict):
                logger.debug("Parsing balance from 'info' field as fallback or refinement...")
                info_result = balance_data["info"].get("result", {})
                account_list = info_result.get("list", [])
                if account_list and isinstance(account_list, list):
                    # Find the Unified account details (usually the first item for unified type)
                    unified_acc_info = next((item for item in account_list if item.get("accountType") == V5_UNIFIED_ACCOUNT_TYPE), None)

                    if unified_acc_info:
                        # Try totalEquity from the unified account info if primary failed
                        if total_equity is None or total_equity.is_nan():
                            total_equity_info_raw = unified_acc_info.get("totalEquity")
                            total_equity_info = safe_decimal(total_equity_info_raw)
                            if not total_equity_info.is_nan():
                                total_equity = total_equity_info
                                logger.debug(f"Using totalEquity from info: {total_equity.normalize()}")
                            else:
                                logger.warning(f"Parsed NaN total equity from info field. Value: '{total_equity_info_raw}'")

                        # Try available balance from the specific coin details within the account info
                        if available_balance is None or available_balance.is_nan():
                             coin_list = unified_acc_info.get("coin", [])
                             if coin_list and isinstance(coin_list, list):
                                 settle_coin_info = next((c for c in coin_list if c.get("coin") == settle_currency), None)
                                 if settle_coin_info:
                                      # Use availableToWithdraw or walletBalance? availableToWithdraw seems safer.
                                      available_info_raw = settle_coin_info.get("availableToWithdraw")
                                      available_info = safe_decimal(available_info_raw)
                                      if not available_info.is_nan():
                                          available_balance = available_info
                                          logger.debug(f"Using availableToWithdraw from info coin list: {available_balance.normalize()}")
                                      else:
                                           logger.warning(f"Parsed NaN available balance from info coin list. Value: '{available_info_raw}'")
                                      # Optionally refine total_equity using coin equity if still needed
                                      if (total_equity is None or total_equity.is_nan()) and 'equity' in settle_coin_info:
                                          coin_equity_raw = settle_coin_info.get("equity")
                                          coin_equity = safe_decimal(coin_equity_raw)
                                          if not coin_equity.is_nan():
                                              total_equity = coin_equity # Use specific coin equity if overall failed
                                              logger.debug(f"Refined total equity using info coin equity: {total_equity.normalize()}")
                                          else:
                                              logger.warning(f"Parsed NaN equity from info coin list. Value: '{coin_equity_raw}'")

            # Final Validation and Logging
            if total_equity is None or total_equity.is_nan():
                logger.error(
                    f"Could not determine a valid total equity for {settle_currency}. Balance data might be incomplete or in an unexpected format. Raw info snippet: {str(balance_data.get('info', {}))[:300]}"
                )
                # Cannot proceed without equity, return None for it
                total_equity = None

            if available_balance is None or available_balance.is_nan():
                logger.warning(
                    f"Could not determine a valid available balance for {settle_currency}. Defaulting available balance to 0. Raw info snippet: {str(balance_data.get('info', {}))[:300]}"
                )
                available_balance = Decimal("0") # Default to 0 if unavailable

            equity_str = total_equity.normalize() if total_equity is not None else "[N/A]"
            avail_str = available_balance.normalize() if available_balance is not None else "[N/A]" # Should be Decimal(0) if defaulted
            logger.info(
                f"Balance Fetched: Equity={equity_str}, Available={avail_str} {settle_currency}"
            )
            return total_equity, available_balance # Return potentially None equity, but Decimal available

        except Exception as e:
            logger.error(f"Failed to fetch or parse balance: {e}", exc_info=True)
            return None, None

    def get_current_position(
        self,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetches current position details for the symbol using V5 API via CCXT.

        Handles one-way and hedge mode (using positionIdx from config).

        Returns:
            Optional[Dict[str, Dict[str, Any]]]: A dict like {'long': {...}, 'short': {...}}
            where only one side (or neither if flat) contains position details.
            Returns None if the fetch or parsing fails critically.
        """
        if not self.exchange or not self.market_info:
            logger.error("Exchange or market info not available, cannot fetch position.")
            return None

        market_id = self.market_info.get("id") # Use the exchange's specific market ID
        if not market_id:
            logger.error("Market ID not found in market info. Cannot fetch position.")
            return None

        logger.debug(
            f"Fetching position for {self.config.symbol} (ID: {market_id}, Category: {self.config.bybit_v5_category}, PosIdx: {self.config.position_idx})..."
        )
        # Initialize structure to return, assuming flat initially
        positions_dict: Dict[str, Dict[str, Any]] = {"long": {}, "short": {}}

        try:
            # Parameters for V5 fetch_positions
            # CCXT might automatically use defaultCategory, but explicit is safer
            params = {
                "category": self.config.bybit_v5_category,
                "symbol": market_id, # Filter by symbol server-side if possible
                # Include positionIdx in params if CCXT fetch_positions passes it through for filtering
                # Check CCXT Bybit implementation details if filtering isn't working as expected.
                # "positionIdx": self.config.position_idx, # May not be directly supported by fetch_positions params
            }
            # Use fetch_positions (plural) as it's the standard CCXT method
            # It should return a list of position dicts.
            fetched_positions_list = fetch_with_retries(
                self.exchange.fetch_positions, # Pass the bound method
                symbols=[self.config.symbol], # Pass symbol list as per CCXT standard
                params=params,
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
            )
            # logger.debug(f"Raw position data list from fetch_positions: {fetched_positions_list}") # Very verbose

            if not isinstance(fetched_positions_list, list):
                logger.error(f"fetch_positions returned unexpected data type: {type(fetched_positions_list)}. Expected list.")
                return None # Treat as failure

            if not fetched_positions_list:
                logger.debug("No position data returned from fetch_positions (list is empty). Assuming flat.")
                return positions_dict # Return empty dict, means flat

            # --- Find the specific position matching symbol AND positionIdx from the list ---
            # V5 /v5/position/list response structure: result.list[] contains positions
            # We need to iterate through the list returned by CCXT and find the one matching our criteria.
            target_pos_info = None
            for pos_ccxt in fetched_positions_list:
                # CCXT standardizes some fields, but V5 specifics are often in 'info'
                info = pos_ccxt.get("info", {})
                if not isinstance(info, dict): continue # Skip if info is not a dict

                pos_symbol_info = info.get("symbol")
                pos_idx_str = info.get("positionIdx") # V5 field name
                pos_category_info = info.get("category") # Check category matches too

                # Basic validation of retrieved info
                if pos_symbol_info != market_id: continue # Symbol doesn't match
                if pos_category_info != self.config.bybit_v5_category: continue # Category doesn't match

                try:
                    pos_idx = int(pos_idx_str) if pos_idx_str is not None else -1 # Default if missing
                except (ValueError, TypeError):
                    logger.warning(f"Non-integer positionIdx '{pos_idx_str}' found in position data: {info}")
                    pos_idx = -1 # Treat as invalid index

                # Check if the position index matches the configured one
                if pos_idx == self.config.position_idx:
                    target_pos_info = info
                    logger.debug(f"Found matching position info in list (Symbol: {pos_symbol_info}, Idx: {pos_idx}): {target_pos_info}")
                    break # Found the one we need

            if not target_pos_info:
                logger.debug(
                    f"No position found matching Symbol '{market_id}', Category '{self.config.bybit_v5_category}', and positionIdx {self.config.position_idx} in the returned list. Assuming flat."
                )
                return positions_dict # Return empty dict, means flat

            # --- Parse the found position details using safe_decimal ---
            qty_raw = target_pos_info.get("size", "0")
            side_api = str(target_pos_info.get("side", "None")).lower() # 'Buy' -> 'buy', 'Sell' -> 'sell', 'None' -> 'none'
            entry_price_raw = target_pos_info.get("avgPrice", "0")
            liq_price_raw = target_pos_info.get("liqPrice", "0") # Note: Liq price might be 0 if no risk
            unrealized_pnl_raw = target_pos_info.get("unrealisedPnl", "0")
            sl_price_raw = target_pos_info.get("stopLoss", "0") # This is the price level
            tp_price_raw = target_pos_info.get("takeProfit", "0") # This is the price level
            # V5 TSL fields: trailingStop (distance), activePrice (activation trigger price)
            tsl_distance_raw = target_pos_info.get("trailingStop", "0")
            tsl_activation_price_raw = target_pos_info.get("activePrice", "0")

            # --- Validate and clean up parsed values ---
            qty = safe_decimal(qty_raw)
            entry_price = safe_decimal(entry_price_raw)
            liq_price = safe_decimal(liq_price_raw)
            unrealized_pnl = safe_decimal(unrealized_pnl_raw)
            sl_price = safe_decimal(sl_price_raw)
            tp_price = safe_decimal(tp_price_raw)
            tsl_distance = safe_decimal(tsl_distance_raw)
            tsl_activation_price = safe_decimal(tsl_activation_price_raw)

            # Use absolute quantity for size check, ensure it's valid
            qty_abs = qty.copy_abs() if not qty.is_nan() else Decimal("0")
            is_position_open = qty_abs >= POSITION_QTY_EPSILON

            if not is_position_open:
                logger.debug(f"Position size {qty_raw} is negligible or zero after parsing. Considered flat.")
                return positions_dict # Return empty dict

            # --- Refine extracted values ---
            # Entry price should be positive if position is open
            entry_price = entry_price if not entry_price.is_nan() and entry_price > 0 else Decimal("NaN")
            # Liq price can be 0/NaN if no liquidation risk (e.g., low leverage or isolated margin)
            liq_price = liq_price if not liq_price.is_nan() and liq_price > 0 else Decimal("NaN")
            # SL/TP prices: Treat 0 or NaN as not set
            sl_price_active = sl_price if not sl_price.is_nan() and sl_price > 0 else None
            tp_price_active = tp_price if not tp_price.is_nan() and tp_price > 0 else None
            # TSL: Active if distance > 0 OR activation price > 0 (distance might be set but not yet triggered)
            is_tsl_active = (not tsl_distance.is_nan() and tsl_distance > 0) or \
                            (not tsl_activation_price.is_nan() and tsl_activation_price > 0)
            tsl_distance_active = tsl_distance if not tsl_distance.is_nan() and tsl_distance > 0 else None
            tsl_activation_price_active = tsl_activation_price if not tsl_activation_price.is_nan() and tsl_activation_price > 0 else None

            # Determine position side ('long' or 'short') based on API 'side' field
            # V5: 'Buy' means long, 'Sell' means short. 'None' might mean flat or could occur in hedge mode before first trade?
            position_side_key: Optional[str] = None
            if side_api == "buy":
                position_side_key = "long"
            elif side_api == "sell":
                position_side_key = "short"
            elif side_api == "none":
                # If side is None but qty > 0, try to infer from posIdx (less reliable)
                # This might happen in hedge mode if only one side has been entered.
                if self.config.position_idx == 1: # Hedge Buy Pos Index
                    position_side_key = "long"
                    logger.debug("Inferred LONG position from posIdx=1 despite API side='None'.")
                elif self.config.position_idx == 2: # Hedge Sell Pos Index
                    position_side_key = "short"
                    logger.debug("Inferred SHORT position from posIdx=2 despite API side='None'.")
                else: # One-Way mode (posIdx=0) or unknown state
                     logger.warning(f"Position found with size {qty_abs} but side is 'None' and posIdx is {self.config.position_idx}. Could not determine long/short state reliably. Treating as flat for safety.")
                     return positions_dict # Return empty dict if side cannot be determined

            if position_side_key:
                position_details = {
                    "qty": qty_abs, # Store absolute quantity
                    "entry_price": entry_price, # Store potentially NaN entry price if parsing failed
                    "liq_price": liq_price, # Store potentially NaN liq price
                    "unrealized_pnl": unrealized_pnl if not unrealized_pnl.is_nan() else Decimal("0"),
                    "side_api": side_api, # Store original API side ('buy'/'sell'/'None')
                    "info": target_pos_info, # Store raw info for debugging
                    "stop_loss_price": sl_price_active, # Store None if not active
                    "take_profit_price": tp_price_active, # Store None if not active
                    "is_tsl_active": is_tsl_active, # Boolean flag
                    "tsl_distance": tsl_distance_active, # Store distance if active/set, else None
                    "tsl_activation_price": tsl_activation_price_active, # Store activation price if set, else None
                }
                positions_dict[position_side_key] = position_details
                entry_str = self.format_price(entry_price) if not entry_price.is_nan() else "[N/A]"
                logger.info(
                    f"Found active {position_side_key.upper()} position: Qty={qty_abs.normalize()}, Entry={entry_str}"
                )
            # else: The case where position_side_key remained None is handled above

            return positions_dict

        except Exception as e:
            logger.error(
                f"Failed to fetch or parse positions for {self.config.symbol}: {e}", exc_info=True
            )
            return None # Return None on critical failure


# --- Indicator Calculator Class ---
class IndicatorCalculator:
    """Calculates technical indicators needed for the trading strategy."""

    def __init__(self, config: TradingConfig) -> None:
        """Initializes the indicator calculator with strategy configuration."""
        self.config = config

    def calculate_indicators(
        self, df: pd.DataFrame
    ) -> Optional[Dict[str, Union[Decimal, bool, int]]]:
        """Calculates EMAs, Stochastic, ATR, ADX from OHLCV data.

        Args:
            df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
                               and a DatetimeIndex. Values should be Decimal or convertible.

        Returns:
            Optional[Dict[str, Union[Decimal, bool, int]]]: Dictionary of the latest indicator values,
            or None if calculation fails or data is insufficient.
            Includes: fast_ema, slow_ema, trend_ema, stoch_k, stoch_d, stoch_k_prev,
                      atr, atr_period, adx, pdi, mdi, stoch_kd_bullish, stoch_kd_bearish.
        """
        func_name = "Indicator Calculation" # For logging context
        logger.info(
            f"{Fore.CYAN}# Weaving indicator patterns (EMA, Stoch, ATR, ADX)...{Style.RESET_ALL}"
        )
        if df is None or df.empty:
            logger.error(f"{Fore.RED}{func_name} failed: No DataFrame provided.{Style.RESET_ALL}")
            return None

        required_cols = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(f"{Fore.RED}{func_name} failed: DataFrame missing required columns: {missing}{Style.RESET_ALL}")
            return None

        try:
            # Work with a copy, select only necessary columns to avoid modifying original
            df_calc = df[required_cols].copy()

            # --- Robust conversion to numeric (float) for calculations ---
            # numpy/pandas work best with floats. Convert Decimal columns safely.
            for col in required_cols:
                if df_calc[col].empty:
                    logger.warning(f"{func_name}: Column '{col}' is empty, ensuring float type.")
                    df_calc[col] = pd.Series(dtype=float) # Ensure column exists with float type
                    continue

                # Define a safe conversion function from Any (including Decimal) to float
                def safe_to_float(x: Any) -> float:
                    dec_val = safe_decimal(x, default=Decimal('NaN')) # Use safe_decimal first
                    if dec_val.is_nan() or not dec_val.is_finite():
                        return np.nan # Convert non-finite Decimals to np.nan for pandas/numpy
                    try:
                        return float(dec_val)
                    except (ValueError, TypeError, OverflowError):
                         # Log specific conversion issues if they occur after safe_decimal
                         logger.debug(f"Could not convert Decimal '{dec_val}' to float in column '{col}'.")
                         return np.nan

                # Apply the safe conversion using map for efficiency
                df_calc[col] = df_calc[col].map(safe_to_float)

                # Ensure the final dtype is float after conversion (map might return object if NaNs present)
                try:
                    df_calc[col] = df_calc[col].astype(float)
                except ValueError as e:
                     logger.error(f"{func_name}: Could not cast column '{col}' to float after mapping: {e}. Contains problematic values.")
                     return None # Cannot proceed if casting fails

            # --- Drop rows with NaN in essential OHLC columns *after* float conversion ---
            initial_len = len(df_calc)
            ohlc_cols = ["open", "high", "low", "close"]
            df_calc.dropna(subset=ohlc_cols, inplace=True)
            rows_dropped = initial_len - len(df_calc)
            if rows_dropped > 0:
                 logger.debug(f"{func_name}: Dropped {rows_dropped} rows with NaN in OHLC after float conversion.")

            if df_calc.empty:
                logger.error(f"{Fore.RED}{func_name} failed: DataFrame empty after NaN drop during processing.{Style.RESET_ALL}")
                return None

            # --- Check Data Length Requirements ---
            # Estimate minimum length needed based on largest period + buffer for smoothing/convergence
            # ADX needs roughly 2*period for smoothing warmup
            # Stoch needs period + smooth_k + smooth_d
            periods = [
                self.config.slow_ema_period, self.config.trend_ema_period,
                self.config.stoch_period + self.config.stoch_smooth_k + self.config.stoch_smooth_d -1, # Rolling window needs this many points
                self.config.atr_period,
                self.config.adx_period * 2 # Heuristic for ADX warmup
            ]
            min_required_len = max(periods) + 20 # Add a buffer for stability/convergence
            if len(df_calc) < min_required_len:
                logger.error(
                    f"{Fore.RED}{func_name} failed: Insufficient data rows ({len(df_calc)} found) "
                    f"for reliable calculation (estimated requirement: ~{min_required_len} rows based on periods).{Style.RESET_ALL}"
                )
                return None

            # --- Indicator Calculations (using df_calc with float types) ---
            close_s = df_calc["close"]
            high_s = df_calc["high"]
            low_s = df_calc["low"]

            # EMAs (Exponential Moving Averages)
            # Using adjust=False makes EMA behave more like traditional platform calculations
            fast_ema_s = close_s.ewm(span=self.config.fast_ema_period, adjust=False).mean()
            slow_ema_s = close_s.ewm(span=self.config.slow_ema_period, adjust=False).mean()
            trend_ema_s = close_s.ewm(span=self.config.trend_ema_period, adjust=False).mean()

            # Stochastic Oscillator (%K, %D)
            low_min = low_s.rolling(window=self.config.stoch_period).min()
            high_max = high_s.rolling(window=self.config.stoch_period).max()
            stoch_range = high_max - low_min
            # Avoid division by zero if range is flat (high_max == low_min)
            # Use np.where for safe division; default to 50.0 if range is zero or NaN
            stoch_k_raw = np.where(
                (stoch_range > 1e-12) & stoch_range.notna(), # Check range > 0 and not NaN
                100 * (close_s - low_min) / stoch_range,
                50.0 # Default value in flat market or NaN range
            )
            # Fill initial NaNs from rolling min/max with 50 before smoothing
            stoch_k_raw_s = pd.Series(stoch_k_raw, index=df_calc.index).fillna(50)
            # Smooth %K (becomes the final %K)
            stoch_k_s = stoch_k_raw_s.rolling(window=self.config.stoch_smooth_k).mean().fillna(50)
            # Smooth %K again to get %D
            stoch_d_s = stoch_k_s.rolling(window=self.config.stoch_smooth_d).mean().fillna(50)

            # ATR (Average True Range)
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            # Calculate True Range, fill initial NaN with 0
            tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
            # Use Wilder's smoothing (equivalent to EMA with alpha=1/period) for ATR
            atr_s = tr_s.ewm(alpha=1 / self.config.atr_period, adjust=False).mean()

            # ADX, +DI, -DI (Average Directional Index)
            # Pass the float series to the helper method
            adx_s, pdi_s, mdi_s = self._calculate_adx(
                high_s, low_s, close_s, self.config.adx_period # Pass close_s for ADX calc
            )

            # --- Extract Latest Values & Convert back to Decimal ---
            def get_latest_decimal(series: pd.Series, name: str) -> Decimal:
                """Safely get the last valid (non-NaN, finite) float value from a Series and convert to Decimal."""
                # Drop NaN and infinite values before selecting the last one
                valid_series = series.replace([np.inf, -np.inf], np.nan).dropna()
                if valid_series.empty:
                    logger.warning(f"{func_name}: No valid finite values found for indicator '{name}'.")
                    return Decimal("NaN")
                # Get the last valid float value
                last_valid_float = valid_series.iloc[-1]
                # Use safe_decimal for robust conversion from the float back to Decimal
                # Convert float to string first for precision with Decimal
                dec_val = safe_decimal(str(last_valid_float), default=Decimal("NaN"))
                if dec_val.is_nan():
                     logger.error(f"{func_name}: Failed converting latest float value '{last_valid_float}' for '{name}' back to Decimal.")
                return dec_val

            # Store results in a dictionary
            indicators_out: Dict[str, Union[Decimal, bool, int]] = {
                "fast_ema": get_latest_decimal(fast_ema_s, "fast_ema"),
                "slow_ema": get_latest_decimal(slow_ema_s, "slow_ema"),
                "trend_ema": get_latest_decimal(trend_ema_s, "trend_ema"),
                "stoch_k": get_latest_decimal(stoch_k_s, "stoch_k"),
                "stoch_d": get_latest_decimal(stoch_d_s, "stoch_d"),
                "atr": get_latest_decimal(atr_s, "atr"),
                "atr_period": self.config.atr_period, # Store period used for ATR context
                "adx": get_latest_decimal(adx_s, "adx"),
                "pdi": get_latest_decimal(pdi_s, "pdi"), # Positive Directional Indicator (+DI)
                "mdi": get_latest_decimal(mdi_s, "mdi"), # Negative Directional Indicator (-DI)
            }

            # --- Calculate Previous Stochastic K for Exit Logic ---
            stoch_k_valid_floats = stoch_k_s.replace([np.inf, -np.inf], np.nan).dropna()
            k_prev = Decimal("NaN")
            if len(stoch_k_valid_floats) >= 2:
                # Get the second to last valid float value and convert to Decimal
                k_prev_float = stoch_k_valid_floats.iloc[-2]
                k_prev = safe_decimal(str(k_prev_float), default=Decimal("NaN"))
            indicators_out["stoch_k_prev"] = k_prev # Add previous K to output

            # --- Calculate Stochastic Cross Signals ---
            k_last = indicators_out["stoch_k"]
            d_last = indicators_out["stoch_d"]
            # Need previous D value as well for standard K/D cross
            d_prev = Decimal("NaN")
            stoch_d_valid_floats = stoch_d_s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(stoch_d_valid_floats) >= 2:
                 d_prev_float = stoch_d_valid_floats.iloc[-2]
                 d_prev = safe_decimal(str(d_prev_float), default=Decimal("NaN"))

            stoch_kd_bullish = False # K crosses above D in oversold zone
            stoch_kd_bearish = False # K crosses below D in overbought zone

            # Check if all required stoch values are valid Decimals before calculating crosses
            stoch_values_valid = not any(v.is_nan() for v in [k_last, d_last, k_prev, d_prev])

            if stoch_values_valid:
                # Check for cross conditions
                crossed_above = (k_prev <= d_prev) and (k_last > d_last)
                crossed_below = (k_prev >= d_prev) and (k_last < d_last)

                # Check if the cross occurred within the relevant zone (using previous values)
                # Bullish cross needs to happen below or crossing up through oversold threshold
                prev_in_oversold_zone = k_prev <= self.config.stoch_oversold_threshold

                # Bearish cross needs to happen above or crossing down through overbought threshold
                prev_in_overbought_zone = k_prev >= self.config.stoch_overbought_threshold

                if crossed_above and prev_in_oversold_zone:
                    stoch_kd_bullish = True
                if crossed_below and prev_in_overbought_zone:
                    stoch_kd_bearish = True

            indicators_out["stoch_kd_bullish"] = stoch_kd_bullish
            indicators_out["stoch_kd_bearish"] = stoch_kd_bearish

            # --- Final Check for Critical NaNs in Output ---
            # ATR is essential for risk calculation, treat its absence as critical failure.
            # Other indicators might be recoverable or strategy might adapt.
            if indicators_out.get("atr", Decimal("NaN")).is_nan():
                 logger.error(f"{Fore.RED}{func_name} failed: ATR calculation resulted in NaN. Cannot proceed with risk assessment.{Style.RESET_ALL}")
                 return None # Fail calculation if ATR is NaN

            # Log warnings for other missing indicators
            critical_keys = [
                "fast_ema", "slow_ema", "trend_ema", "stoch_k", "stoch_d",
                "stoch_k_prev", "adx", "pdi", "mdi",
            ]
            failed_indicators = [
                k for k in critical_keys if indicators_out.get(k, Decimal("NaN")).is_nan()
            ]
            if failed_indicators:
                logger.warning(
                    f"{Fore.YELLOW}{func_name}: Some non-critical indicators calculated as NaN: {', '.join(failed_indicators)}. Signal generation might be affected.{Style.RESET_ALL}"
                )

            logger.info(f"{Fore.GREEN}{Style.BRIGHT}Indicator patterns woven successfully.{Style.RESET_ALL}")
            # Cast the dictionary to the expected return type for static analysis
            return cast(Dict[str, Union[Decimal, bool, int]], indicators_out)

        except Exception as e:
            logger.error(f"{Fore.RED}Error weaving indicator patterns: {e}{Style.RESET_ALL}", exc_info=True)
            return None

    def _calculate_adx(
        self,
        high_s: pd.Series,
        low_s: pd.Series,
        close_s: pd.Series, # Close is needed for TR calculation within ADX
        period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Helper to calculate ADX, +DI, -DI using Wilder's smoothing (EMA).
           Takes pandas Series (float type) as input.
           Returns tuple of (ADX, +DI, -DI) Series (float type).
        """
        if period <= 0:
            raise ValueError("ADX period must be positive")
        if high_s.empty or low_s.empty or close_s.empty:
            logger.error("ADX Calculation: Input Series (H, L, C) are empty.")
            nan_series = pd.Series(np.nan, index=high_s.index)
            return nan_series, nan_series, nan_series

        # Calculate True Range (TR) - needed for DI calculation
        prev_close = close_s.shift(1)
        tr1 = high_s - low_s
        tr2 = (high_s - prev_close).abs()
        tr3 = (low_s - prev_close).abs()
        tr_s = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(0)
        # Calculate Smoothed TR (ATR using Wilder's method)
        alpha = 1 / period
        atr_s = tr_s.ewm(alpha=alpha, adjust=False).mean()

        # Calculate Directional Movement (+DM, -DM)
        move_up = high_s.diff()
        move_down = -low_s.diff() # Note: diff is current - previous

        # +DM is UpMove if UpMove > DownMove and UpMove > 0, else 0
        plus_dm = np.where((move_up > move_down) & (move_up > 0), move_up, 0.0)
        # -DM is DownMove if DownMove > UpMove and DownMove > 0, else 0
        minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

        # Smoothed +DM, -DM using Wilder's method (EMA)
        # Ensure initial NaNs from diff() are handled, fillna(0) before ewm
        plus_dm_s = pd.Series(plus_dm, index=high_s.index).fillna(0).ewm(alpha=alpha, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=high_s.index).fillna(0).ewm(alpha=alpha, adjust=False).mean()

        # Calculate Directional Indicators (+DI, -DI)
        # Avoid division by zero or NaN ATR: replace 0 ATR with NaN before division
        safe_atr_s = atr_s.replace(0, np.nan)
        # Use np.where for safe division, default to 0 if ATR is NaN
        pdi_s_raw = np.where(safe_atr_s.notnull(), 100 * plus_dm_s / safe_atr_s, 0.0)
        mdi_s_raw = np.where(safe_atr_s.notnull(), 100 * minus_dm_s / safe_atr_s, 0.0)
        # Fill any resulting NaNs (e.g., from initial ATR NaNs) with 0
        pdi_s = pd.Series(pdi_s_raw, index=high_s.index).fillna(0)
        mdi_s = pd.Series(mdi_s_raw, index=high_s.index).fillna(0)

        # Calculate Directional Movement Index (DX)
        di_diff_abs = (pdi_s - mdi_s).abs()
        di_sum = pdi_s + mdi_s
        # Avoid division by zero if sum is zero (PDI and MDI are both 0)
        dx_s_raw = np.where(di_sum > 1e-12, 100 * di_diff_abs / di_sum, 0.0)
        # Fill initial NaNs with 0
        dx_s = pd.Series(dx_s_raw, index=high_s.index).fillna(0)

        # Calculate Average Directional Index (ADX - Smoothed DX using Wilder's EMA)
        # Fill initial NaNs with 0
        adx_s = dx_s.ewm(alpha=alpha, adjust=False).mean().fillna(0)

        return adx_s, pdi_s, mdi_s


# --- Signal Generator Class ---
class SignalGenerator:
    """Generates trading entry and exit signals based on indicator conditions."""

    def __init__(self, config: TradingConfig) -> None:
        """Initializes the signal generator with strategy configuration."""
        self.config = config

    def generate_signals(
        self,
        df_last_candles: pd.DataFrame,
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Dict[str, Union[bool, str]]:
        """Generates 'long'/'short' entry signals based on EMA, Stoch, ADX, and other filters.

        Args:
            df_last_candles (pd.DataFrame): DataFrame containing at least the last 2 candles.
            indicators (Dict): Dictionary of calculated indicator values for the latest candle.

        Returns:
            Dict[str, Union[bool, str]]: Dictionary with 'long' (bool), 'short' (bool),
                                         and 'reason' (str) keys.
        """
        result: Dict[str, Union[bool, str]] = {
            "long": False,
            "short": False,
            "reason": "Initializing", # Default reason
        }
        func_name = "Signal Generation" # Context for logging

        # --- Pre-checks ---
        if not indicators:
            result["reason"] = "No Signal: Indicators dictionary missing"
            logger.debug(f"{func_name}: {result['reason']}")
            return result
        if df_last_candles is None or len(df_last_candles) < 2:
            candle_count = len(df_last_candles) if df_last_candles is not None else 0
            reason = f"No Signal: Insufficient candle data (<2 rows, got {candle_count})"
            result["reason"] = reason
            logger.debug(f"{func_name}: {reason}")
            return result

        try:
            # --- Extract Data & Indicators Safely ---
            try:
                latest_candle = df_last_candles.iloc[-1]
                prev_candle = df_last_candles.iloc[-2]
                # Ensure close prices are valid Decimals
                current_price = safe_decimal(latest_candle["close"])
                prev_close = safe_decimal(prev_candle["close"])
            except (IndexError, KeyError) as e:
                 result["reason"] = f"No Signal: Error accessing candle data ({e})"
                 logger.error(f"{func_name}: {result['reason']}", exc_info=True)
                 return result

            if current_price.is_nan() or current_price <= 0:
                result["reason"] = f"No Signal: Invalid current price ({current_price})"
                logger.warning(f"{func_name}: {result['reason']}")
                return result

            # Extract required indicators, checking for valid Decimal type and finite value
            required_indicator_keys = [
                "stoch_k", "fast_ema", "slow_ema", "trend_ema", "atr", "adx", "pdi", "mdi"
            ]
            ind_values: Dict[str, Decimal] = {} # Store validated Decimal indicators
            missing_or_invalid_keys = []
            for key in required_indicator_keys:
                val = indicators.get(key)
                # Ensure value is a valid, finite Decimal
                if isinstance(val, Decimal) and not val.is_nan() and val.is_finite():
                    ind_values[key] = val
                else:
                    missing_or_invalid_keys.append(f"{key}({val})") # Include value for context

            if missing_or_invalid_keys:
                reason = f"No Signal: Required indicator(s) missing, NaN, or infinite: {', '.join(missing_or_invalid_keys)}"
                result["reason"] = reason
                logger.warning(f"{func_name}: {reason}")
                return result

            # Assign validated indicators to variables
            k, fast_ema, slow_ema, trend_ema, atr, adx, pdi, mdi = (
                ind_values["stoch_k"], ind_values["fast_ema"], ind_values["slow_ema"],
                ind_values["trend_ema"], ind_values["atr"], ind_values["adx"],
                ind_values["pdi"], ind_values["mdi"]
            )
            # Get boolean Stochastic cross flags (default to False if missing/not bool)
            kd_bull = indicators.get("stoch_kd_bullish", False) if isinstance(indicators.get("stoch_kd_bullish"), bool) else False
            kd_bear = indicators.get("stoch_kd_bearish", False) if isinstance(indicators.get("stoch_kd_bearish"), bool) else False

            # --- Define Conditions ---
            # 1. EMA Cross
            ema_bullish_cross = fast_ema > slow_ema
            ema_bearish_cross = fast_ema < slow_ema
            ema_reason = f"EMA({fast_ema:.{DEFAULT_PRICE_DP}f} {' > ' if ema_bullish_cross else ' < ' if ema_bearish_cross else ' = '} {slow_ema:.{DEFAULT_PRICE_DP}f})"

            # 2. Trend Filter (Optional)
            trend_allows_long = True
            trend_allows_short = True
            trend_reason = "TrendFilter OFF"
            if self.config.trade_only_with_trend:
                trend_buffer = trend_ema.copy_abs() * (self.config.trend_filter_buffer_percent / 100)
                price_above_trend_ema = current_price > (trend_ema + trend_buffer) # Price clearly above trend EMA
                price_below_trend_ema = current_price < (trend_ema - trend_buffer) # Price clearly below trend EMA
                trend_allows_long = price_above_trend_ema
                trend_allows_short = price_below_trend_ema
                trend_reason = f"Trend(P:{current_price:.{DEFAULT_PRICE_DP}f} vs EMA:{trend_ema:.{DEFAULT_PRICE_DP}f} +/- {trend_buffer:.{DEFAULT_PRICE_DP}f} | Long:{trend_allows_long} Short:{trend_allows_short})"

            # 3. Stochastic Condition (Oversold/Overbought OR K/D Cross)
            stoch_long_cond = (k < self.config.stoch_oversold_threshold) or kd_bull
            stoch_short_cond = (k > self.config.stoch_overbought_threshold) or kd_bear
            stoch_reason = f"Stoch(K:{k:.1f} {'[OS]' if k < self.config.stoch_oversold_threshold else ''}{'[OB]' if k > self.config.stoch_overbought_threshold else ''} | {'BullX' if kd_bull else ''}{'BearX' if kd_bear else ''})"

            # 4. ATR Move Filter (Optional)
            significant_move = True # Assume true if filter disabled
            atr_reason = "ATR Filter OFF"
            if self.config.atr_move_filter_multiplier > 0:
                # ATR validity already checked in ind_values extraction
                if prev_close.is_nan() or prev_close <= 0:
                    atr_reason = "ATR Filter Skipped (Prev Close Invalid)"
                    significant_move = False # Cannot evaluate if prev close is invalid
                else:
                    atr_move_threshold = atr * self.config.atr_move_filter_multiplier
                    price_move = (current_price - prev_close).copy_abs()
                    significant_move = price_move > atr_move_threshold
                    atr_reason = f"ATR Move({price_move:.{DEFAULT_PRICE_DP}f}) {'OK' if significant_move else 'LOW'} vs Thr({atr_move_threshold:.{DEFAULT_PRICE_DP}f})"

            # 5. ADX Trend Strength and Direction Filter
            adx_is_trending = adx > self.config.min_adx_level
            adx_long_direction = pdi > mdi
            adx_short_direction = mdi > pdi
            # Signal requires ADX to be trending AND direction to align
            adx_allows_long = adx_is_trending and adx_long_direction
            adx_allows_short = adx_is_trending and adx_short_direction
            adx_reason = f"ADX({adx:.1f}) {'OK' if adx_is_trending else 'LOW'} vs {self.config.min_adx_level:.1f} | Dir({'+DI' if adx_long_direction else '-DI' if adx_short_direction else 'NONE'}) | Long:{adx_allows_long} Short:{adx_allows_short}"

            # --- Combine Conditions for Final Signal ---
            # Base signal: EMA cross aligns with Stochastic condition
            base_long_signal = ema_bullish_cross and stoch_long_cond
            base_short_signal = ema_bearish_cross and stoch_short_cond

            # Apply filters
            final_long_signal = base_long_signal and trend_allows_long and significant_move and adx_allows_long
            final_short_signal = base_short_signal and trend_allows_short and significant_move and adx_allows_short

            # --- Build Detailed Reason String ---
            if final_long_signal:
                result["long"] = True
                result["reason"] = f"Long Signal: {ema_reason} | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            elif final_short_signal:
                result["short"] = True
                result["reason"] = f"Short Signal: {ema_reason} | {stoch_reason} | {trend_reason} | {atr_reason} | {adx_reason}"
            else:
                # Provide reason why signal was not generated or was blocked
                reason_parts = ["No Signal:"]
                # Explain base signal failure
                if not base_long_signal and not base_short_signal:
                     reason_parts.append(f"BaseCond Fail ({ema_reason}, {stoch_reason})")
                # Explain filter blocks if base signal was present
                elif base_long_signal: # Base long was true, check blocks
                    if not trend_allows_long: reason_parts.append(f"Long Blocked ({trend_reason})")
                    elif not significant_move: reason_parts.append(f"Long Blocked ({atr_reason})")
                    elif not adx_allows_long: reason_parts.append(f"Long Blocked ({adx_reason})")
                    else: reason_parts.append("Unknown Long Block") # Fallback
                elif base_short_signal: # Base short was true, check blocks
                    if not trend_allows_short: reason_parts.append(f"Short Blocked ({trend_reason})")
                    elif not significant_move: reason_parts.append(f"Short Blocked ({atr_reason})")
                    elif not adx_allows_short: reason_parts.append(f"Short Blocked ({adx_reason})")
                    else: reason_parts.append("Unknown Short Block") # Fallback
                result["reason"] = " | ".join(reason_parts)

            # Log signal generation result (Info for signals/blocks, Debug otherwise)
            log_level_sig = logging.INFO if result["long"] or result["short"] or "Blocked" in result["reason"] else logging.DEBUG
            logger.log(log_level_sig, f"{func_name}: {result['reason']}")

        except Exception as e:
            logger.error(f"{Fore.RED}{func_name}: Error generating signals: {e}{Style.RESET_ALL}", exc_info=True)
            result["reason"] = f"No Signal: Exception ({type(e).__name__})"
            result["long"] = False
            result["short"] = False

        return result

    def check_exit_signals(
        self,
        position_side: str, # 'long' or 'short'
        indicators: Dict[str, Union[Decimal, bool, int]],
    ) -> Optional[str]:
        """Checks for signal-based position exits (e.g., EMA cross against position, Stoch reversal).

        Args:
            position_side (str): The side of the current active position ('long' or 'short').
            indicators (Dict): Dictionary of calculated indicator values.

        Returns:
            Optional[str]: A reason string if an exit signal is generated, otherwise None.
        """
        func_name = "Exit Signal Check" # Context for logging
        if not indicators:
            logger.warning(f"{func_name}: Cannot check exit signals: indicators dictionary missing.")
            return None
        if position_side not in ["long", "short"]:
            logger.error(f"{func_name}: Invalid position_side '{position_side}'.")
            return None

        # --- Extract indicators safely, checking for valid Decimal type ---
        fast_ema = indicators.get("fast_ema")
        slow_ema = indicators.get("slow_ema")
        stoch_k = indicators.get("stoch_k")
        stoch_k_prev = indicators.get("stoch_k_prev") # Previous K value is crucial for reversal

        # Validate required indicators are finite Decimals
        required_vals = {"fast_ema": fast_ema, "slow_ema": slow_ema, "stoch_k": stoch_k, "stoch_k_prev": stoch_k_prev}
        invalid_inds = [k for k, v in required_vals.items() if not isinstance(v, Decimal) or v.is_nan() or not v.is_finite()]

        if invalid_inds:
            logger.warning(
                f"{func_name}: Cannot check exit signals due to missing/invalid indicators: {', '.join(invalid_inds)}."
            )
            return None # Cannot proceed without valid indicator values

        # Cast to Decimal after validation for type checker clarity (already validated)
        fast_ema = cast(Decimal, fast_ema)
        slow_ema = cast(Decimal, slow_ema)
        stoch_k = cast(Decimal, stoch_k)
        stoch_k_prev = cast(Decimal, stoch_k_prev)

        # --- Calculate EMA cross state ---
        ema_cross_against_long = fast_ema < slow_ema # Bearish cross for exiting long
        ema_cross_against_short = fast_ema > slow_ema # Bullish cross for exiting short

        exit_reason: Optional[str] = None

        # --- Define Overbought/Oversold levels from config ---
        oversold_level = self.config.stoch_oversold_threshold
        overbought_level = self.config.stoch_overbought_threshold

        # --- Evaluate Exit Conditions based on Position Side ---
        if position_side == "long":
            # Priority 1: EMA Bearish Cross (Fast < Slow)
            if ema_cross_against_long:
                exit_reason = f"Exit Signal: EMA Bearish Cross ({fast_ema:.{DEFAULT_PRICE_DP}f} < {slow_ema:.{DEFAULT_PRICE_DP}f})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal from Overbought
            # Check if previous K was >= OB and current K crossed below OB
            elif stoch_k_prev >= overbought_level and stoch_k < overbought_level:
                exit_reason = f"Exit Signal: Stoch Reversal from OB (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed below {overbought_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for LONG position.{Style.RESET_ALL}")

            # Informational Logging: Stochastic is in the zone but hasn't confirmed reversal
            elif stoch_k >= overbought_level:
                logger.debug(f"{func_name} (Long): Stoch K ({stoch_k:.1f}) >= Overbought ({overbought_level}), awaiting cross below for exit signal.")

        elif position_side == "short":
            # Priority 1: EMA Bullish Cross (Fast > Slow)
            if ema_cross_against_short:
                exit_reason = f"Exit Signal: EMA Bullish Cross ({fast_ema:.{DEFAULT_PRICE_DP}f} > {slow_ema:.{DEFAULT_PRICE_DP}f})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}")

            # Priority 2: Stochastic Reversal from Oversold
            # Check if previous K was <= OS and current K crossed above OS
            elif stoch_k_prev <= oversold_level and stoch_k > oversold_level:
                exit_reason = f"Exit Signal: Stoch Reversal from OS (K {stoch_k_prev:.1f} -> {stoch_k:.1f} crossed above {oversold_level})"
                logger.trade(f"{Fore.YELLOW}{exit_reason} detected for SHORT position.{Style.RESET_ALL}")

            # Informational Logging: Stochastic is in the zone but hasn't confirmed reversal
            elif stoch_k <= oversold_level:
                logger.debug(f"{func_name} (Short): Stoch K ({stoch_k:.1f}) <= Oversold ({oversold_level}), awaiting cross above for exit signal.")

        # Return the reason string if an exit condition was met, otherwise None
        return exit_reason


# --- Order Manager Class ---
class OrderManager:
    """Handles order placement, position protection (SL/TP/TSL), closing, and verification using V5 API."""

    def __init__(
        self, config: TradingConfig, exchange_manager: ExchangeManager
    ) -> None:
        """Initializes the order manager with configuration and exchange access."""
        self.config = config
        self.exchange_manager = exchange_manager
        # Ensure exchange instance exists before assigning
        if not exchange_manager or not exchange_manager.exchange:
            logger.critical(f"{Fore.RED}{Style.BRIGHT}OrderManager cannot initialize: ExchangeManager or CCXT Exchange instance missing.{Style.RESET_ALL}")
            # Raise ValueError to prevent instantiation with invalid state
            raise ValueError("OrderManager requires a valid ExchangeManager with an initialized CCXT exchange.")
        self.exchange = exchange_manager.exchange # Convenience accessor
        self.market_info = exchange_manager.market_info # Convenience accessor
        # Tracks active protection STATUS (e.g., 'ACTIVE_SLTP', 'ACTIVE_TSL') for V5 position stops per side
        # Necessary because V5 stops are attached to the position, not separate orders.
        self.protection_tracker: Dict[str, Optional[str]] = {"long": None, "short": None}

    def _calculate_trade_parameters(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> Optional[Dict[str, Optional[Decimal]]]:
        """Calculates SL price, TP price, quantity, and TSL parameters based on risk, ATR, and market info.

        Args:
            side: 'buy' or 'sell'.
            atr: Current ATR value (as Decimal).
            total_equity: Total account equity (as Decimal).
            current_price: Current market price for calculation basis (as Decimal).

        Returns:
            Optional dictionary containing calculated parameters (all Decimal or None):
            {'qty': Decimal, 'sl_price': Decimal, 'tp_price': Optional[Decimal],
             'tsl_distance': Optional[Decimal], 'tsl_activation_price': Optional[Decimal]}
            Returns None if calculation fails due to invalid inputs or market constraints.
        """
        func_name = "Trade Parameter Calculation" # Context for logging
        # --- Input Validation ---
        if atr.is_nan() or atr <= 0 or not atr.is_finite():
            logger.error(f"{func_name} failed: Invalid ATR value ({atr}).")
            return None
        if total_equity.is_nan() or total_equity <= 0 or not total_equity.is_finite():
            logger.error(f"{func_name} failed: Invalid total equity value ({total_equity}).")
            return None
        if current_price.is_nan() or current_price <= 0 or not current_price.is_finite():
            logger.error(f"{func_name} failed: Invalid current price value ({current_price}).")
            return None
        if not self.market_info or not all(k in self.market_info for k in ['tick_size', 'contract_size', 'min_order_size', 'precision_dp']):
             logger.error(f"{func_name} failed: Market info missing required keys (tick_size, contract_size, min_order_size, precision_dp).")
             return None
        if side not in ["buy", "sell"]:
            logger.error(f"{func_name} failed: Invalid side '{side}'.")
            return None

        try:
            # --- Risk Amount ---
            risk_amount_per_trade = total_equity * self.config.risk_percentage
            logger.debug(f"{func_name}: Risk Amount = {risk_amount_per_trade.normalize()} (Equity {total_equity.normalize()} * Risk {self.config.risk_percentage * 100}%)")

            # --- Stop Loss Calculation ---
            sl_distance_atr = atr * self.config.sl_atr_multiplier
            sl_price_raw: Decimal
            if side == "buy": sl_price_raw = current_price - sl_distance_atr
            else: sl_price_raw = current_price + sl_distance_atr

            # Format SL price according to market precision
            sl_price_str = self.exchange_manager.format_price(sl_price_raw)
            sl_price = safe_decimal(sl_price_str)

            if sl_price.is_nan() or sl_price <= 0:
                logger.error(f"{func_name} failed: Calculated SL price ({sl_price_str}) is invalid (<=0 or NaN).")
                return None

            # Calculate SL distance in price terms AFTER formatting SL price
            sl_distance_price = (current_price - sl_price).copy_abs()

            # Ensure SL distance respects minimum tick size
            min_tick_size = self.market_info['tick_size']
            if sl_distance_price < min_tick_size:
                sl_distance_price = min_tick_size
                # Recalculate SL price based on minimum tick distance
                if side == "buy": sl_price = current_price - sl_distance_price
                else: sl_price = current_price + sl_distance_price
                # Re-format the adjusted SL price
                sl_price_str = self.exchange_manager.format_price(sl_price)
                sl_price = safe_decimal(sl_price_str)
                logger.warning(f"{func_name}: Initial SL distance < tick size. Adjusted SL price to {sl_price_str} (distance = {sl_distance_price.normalize()}).")
                # Final check if adjusted SL is still valid
                if sl_price.is_nan() or sl_price <= 0:
                     logger.error(f"{func_name} failed: Adjusted SL price ({sl_price_str}) is still invalid (<=0 or NaN).")
                     return None

            # Should not happen with checks above, but safeguard against zero distance
            if sl_distance_price <= 0:
                logger.error(f"{func_name} failed: Calculated SL distance ({sl_distance_price}) is zero or negative.")
                return None

            # --- Quantity Calculation (Handles Linear vs Inverse) ---
            contract_size = self.market_info['contract_size']
            quantity: Optional[Decimal] = None

            try:
                if self.config.market_type == "inverse":
                    # Formula for Inverse Contracts (e.g., BTC/USD settled in BTC)
                    # Qty (in Contracts, e.g., USD) = Risk Amount (Quote, e.g., USD) * SL Price / (Contract Size * SL Distance Price)
                    # Note: Risk amount is in Quote currency (USD), but needs to be based on Equity (which might be Base or Quote). Assuming equity is in Quote here.
                    # If equity is in Base (BTC), convert risk amount: Risk Amount (Quote) = Risk Amount (Base) * Entry Price
                    # Let's assume total_equity is already the Quote currency value of the account.
                    if contract_size <= 0: raise ValueError("Invalid contract size for inverse calc.")
                    if sl_price <= 0: raise ValueError("Invalid SL price for inverse calc.") # Already checked but defensive
                    quantity = (risk_amount_per_trade * sl_price) / (contract_size * sl_distance_price)
                    logger.debug(f"{func_name}: Inverse Qty Calc: ({risk_amount_per_trade} * {sl_price}) / ({contract_size} * {sl_distance_price}) = {quantity}")

                else: # Linear Contracts (e.g., BTC/USDT settled in USDT)
                    # Formula for Linear Contracts
                    # Qty (in Base, e.g., BTC) = Risk Amount (Quote, e.g., USDT) / SL Distance Price (Quote)
                    # Assumes Contract Size = 1 Base Unit (e.g., 1 contract = 1 BTC). If not, adjust.
                    # Bybit V5 Linear Perps usually have contractSize = 1 (meaning 1 contract = 1 Base unit)
                    if contract_size != Decimal(1):
                        logger.warning(f"{func_name}: Linear contract size is {contract_size}, not 1. Quantity calculation assumes 1 contract = 1 base unit. Verify market details if order size seems wrong.")
                    # Value per point = contract_size (in Base units)
                    # Risk per base unit = sl_distance_price (in Quote units)
                    quantity = risk_amount_per_trade / sl_distance_price
                    logger.debug(f"{func_name}: Linear Qty Calc: {risk_amount_per_trade} / {sl_distance_price} = {quantity}")

            except (DivisionByZero, ValueError, InvalidOperation) as calc_err:
                 logger.error(f"{func_name} failed during quantity calculation: {calc_err}", exc_info=True)
                 return None

            if quantity is None or quantity.is_nan() or quantity <= 0:
                 logger.error(f"{func_name} failed: Calculated quantity ({quantity}) is invalid or zero.")
                 return None

            # Format quantity according to market rules (ROUND_DOWN)
            quantity_str = self.exchange_manager.format_amount(quantity, rounding_mode=ROUND_DOWN)
            quantity_decimal = safe_decimal(quantity_str)

            if quantity_decimal.is_nan() or quantity_decimal <= 0:
                 logger.error(f"{func_name} failed: Formatted quantity ({quantity_str}) is invalid or zero.")
                 return None

            # Check against minimum order size
            min_order_size = self.market_info.get('min_order_size', Decimal('NaN'))
            if not min_order_size.is_nan() and quantity_decimal < min_order_size:
                logger.error(
                    f"{func_name} failed: Calculated quantity {quantity_decimal.normalize()} "
                    f"is less than minimum order size {min_order_size.normalize()}. "
                    f"Increase risk % or check available capital."
                )
                return None

            # --- Take Profit Calculation (Optional) ---
            tp_price: Optional[Decimal] = None
            if self.config.tp_atr_multiplier > 0:
                tp_distance_atr = atr * self.config.tp_atr_multiplier
                tp_price_raw: Decimal
                if side == "buy": tp_price_raw = current_price + tp_distance_atr
                else: tp_price_raw = current_price - tp_distance_atr

                # Format TP price
                tp_price_str = self.exchange_manager.format_price(tp_price_raw)
                tp_price_check = safe_decimal(tp_price_str)

                if tp_price_check.is_nan() or tp_price_check <= 0:
                    logger.warning(f"{func_name}: Calculated TP price ({tp_price_str}) invalid (<=0 or NaN). Disabling TP for this trade.")
                    tp_price = None
                else:
                    tp_price = tp_price_check # Use the validated, formatted TP price
            else:
                logger.debug(f"{func_name}: TP multiplier is 0, TP disabled.")

            # --- Trailing Stop Calculation (Distance and Activation Price) ---
            tsl_distance: Optional[Decimal] = None
            tsl_activation_price: Optional[Decimal] = None

            # Calculate TSL distance based on percentage of current price
            tsl_distance_raw = current_price * (self.config.trailing_stop_percent / 100)
            # Format distance like a price difference (using price precision)
            tsl_distance_str = self.exchange_manager.format_price(tsl_distance_raw)
            tsl_distance_check = safe_decimal(tsl_distance_str)

            if tsl_distance_check.is_nan() or tsl_distance_check <= min_tick_size: # Ensure distance > tick size
                 logger.warning(f"{func_name}: Calculated TSL distance ({tsl_distance_str}) is invalid or <= tick size. Using tick size ({min_tick_size}) as minimum TSL distance.")
                 # Use minimum tick size as the fallback distance
                 tsl_distance = min_tick_size
            else:
                 tsl_distance = tsl_distance_check

            # Calculate TSL activation price based on ATR multiplier (optional)
            if self.config.tsl_activation_atr_multiplier > 0:
                 tsl_activation_distance = atr * self.config.tsl_activation_atr_multiplier
                 tsl_activation_price_raw: Decimal
                 if side == "buy":
                     tsl_activation_price_raw = current_price + tsl_activation_distance
                 else: # side == "sell"
                     tsl_activation_price_raw = current_price - tsl_activation_distance

                 # Format activation price
                 tsl_act_price_str = self.exchange_manager.format_price(tsl_activation_price_raw)
                 tsl_activation_price_check = safe_decimal(tsl_act_price_str)

                 if tsl_activation_price_check.is_nan() or tsl_activation_price_check <= 0:
                      logger.warning(f"{func_name}: Calculated TSL activation price ({tsl_act_price_str}) invalid (<=0 or NaN). TSL may not activate via price.")
                      tsl_activation_price = None # Mark as invalid
                 else:
                      tsl_activation_price = tsl_activation_price_check # Use valid, formatted price
            else:
                logger.debug(f"{func_name}: TSL activation multiplier is 0, TSL activation price not set.")


            # --- Consolidate Results ---
            params_out: Dict[str, Optional[Decimal]] = {
                "qty": quantity_decimal,
                "sl_price": sl_price, # Already validated and formatted
                "tp_price": tp_price, # None if disabled or invalid
                "tsl_distance": tsl_distance, # Validated distance
                "tsl_activation_price": tsl_activation_price, # None if disabled or invalid
            }

            # Log calculated parameters clearly
            log_tp_str = f"{params_out['tp_price'].normalize()}" if params_out['tp_price'] else "None"
            log_tsl_dist_str = f"{params_out['tsl_distance'].normalize()}" if params_out['tsl_distance'] else "Invalid/MinTick"
            log_tsl_act_str = f"{params_out['tsl_activation_price'].normalize()}" if params_out['tsl_activation_price'] else "None"
            logger.info(
                f"{func_name} Results: Side={side.upper()}, "
                f"Qty={params_out['qty'].normalize()}, "
                f"EntryRefPx={current_price.normalize()}, "
                f"SL={params_out['sl_price'].normalize()}, "
                f"TP={log_tp_str}, "
                f"TSL(Dist={log_tsl_dist_str}, ActPx={log_tsl_act_str}), "
                f"RiskAmt={risk_amount_per_trade.normalize()}, ATR={atr.normalize()}"
            )
            return params_out

        except Exception as e:
            logger.error(f"{func_name}: Unexpected error: {e}", exc_info=True)
            return None

    def _execute_market_order(
        self, side: str, qty_decimal: Decimal, reduce_only: bool = False
    ) -> Optional[Dict]:
        """Executes a market order with retries and basic parsing of the response.

        Args:
            side (str): 'buy' or 'sell'.
            qty_decimal (Decimal): The quantity to order (positive Decimal).
            reduce_only (bool): Whether to set the reduceOnly flag (for closing orders).

        Returns:
            Optional[Dict]: The parsed order dictionary from CCXT if submission seems successful,
                           otherwise None.
        """
        if not self.exchange or not self.market_info:
            logger.error("Cannot execute market order: Exchange/Market info missing.")
            return None
        if qty_decimal.is_nan() or qty_decimal <= 0:
             logger.error(f"Attempted market order with invalid quantity: {qty_decimal}")
             return None

        symbol = self.config.symbol
        # Format quantity according to market rules (ROUND_DOWN) before sending
        qty_str = self.exchange_manager.format_amount(qty_decimal, rounding_mode=ROUND_DOWN)
        final_qty_decimal = safe_decimal(qty_str)

        if final_qty_decimal.is_nan() or final_qty_decimal <= 0:
            logger.error(f"Market order aborted: Zero/invalid formatted quantity '{qty_str}' from original '{qty_decimal}'.")
            return None

        # Ensure quantity meets minimum order size AFTER formatting
        min_order_size = self.market_info.get('min_order_size', Decimal('NaN'))
        if not min_order_size.is_nan() and final_qty_decimal < min_order_size:
            logger.error(
                f"Market order aborted: Formatted quantity {final_qty_decimal.normalize()} "
                f"< minimum {min_order_size.normalize()}."
            )
            return None

        order_type_desc = "CLOSE" if reduce_only else "ENTRY"
        logger.trade(
            f"{Fore.CYAN}Attempting MARKET {side.upper()} {order_type_desc} order: {final_qty_decimal.normalize()} {symbol}...{Style.RESET_ALL}"
        )
        try:
            # V5 parameters for create_order
            params: Dict[str, Any] = {
                "category": self.config.bybit_v5_category,
                "positionIdx": self.config.position_idx, # Specify hedge mode index
                "timeInForce": "ImmediateOrCancel", # Use IOC for market orders to avoid partial fills hanging
                # reduceOnly should only be True for closing orders
                "reduceOnly": reduce_only,
            }
            # CCXT's create_market_order often calls create_order internally
            # Using create_order directly gives more control over params like reduceOnly
            # CCXT expects amount as float
            amount_float = float(final_qty_decimal)

            order = fetch_with_retries(
                self.exchange.create_order,
                symbol=symbol,
                type='market', # Explicitly specify market type
                side=side,
                amount=amount_float,
                params=params,
                # Retry config
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
                # Fail fast on critical order errors
                fail_fast_exceptions=(ccxt.InsufficientFunds, ccxt.InvalidOrder),
            )

            if order is None:
                # fetch_with_retries would have raised if retries failed normally.
                # This likely means a fail-fast exception occurred or an unexpected None return.
                logger.error(f"{Fore.RED}Market {side.upper()} {order_type_desc} order submission failed (returned None). Check previous logs for fail-fast reasons (e.g., Insufficient Funds).{Style.RESET_ALL}")
                return None

            # --- Basic Order Response Parsing ---
            order_id = order.get("id", "[N/A]")
            order_status = order.get("status", "[unknown]") # e.g., 'open', 'closed', 'canceled', 'rejected', 'expired'
            filled_qty_str = order.get("filled", "0") # Amount filled
            avg_fill_price_str = order.get("average", "0") # Average fill price
            # Safely convert parsed values
            filled_qty = safe_decimal(filled_qty_str)
            avg_fill_price = safe_decimal(avg_fill_price_str)

            # Format average price for logging
            avg_price_log_str = '[N/A]'
            if not avg_fill_price.is_nan() and avg_fill_price > 0:
                avg_price_log_str = self.exchange_manager.format_price(avg_fill_price)

            log_color = Fore.GREEN if order_status in ['closed', 'filled'] else Fore.YELLOW if order_status == 'open' else Fore.RED
            logger.trade(
                f"{log_color}{Style.BRIGHT}Market {order_type_desc} order submitted: ID {order_id}, Side {side.upper()}, "
                f"ReqQty {final_qty_decimal.normalize()}, Status: {order_status}, FilledQty: {filled_qty.normalize()}, "
                f"AvgPx: {avg_price_log_str}{Style.RESET_ALL}"
            )
            termux_notify(
                f"{symbol} Order Submitted", f"Market {side.upper()} {order_type_desc} {final_qty_decimal.normalize()} ID:{order_id} Status:{order_status}"
            )

            # Check for immediate failure status
            if order_status in ["rejected", "canceled", "expired"]:
                 # Try to get reason from V5 response structure within 'info'
                 reason = order.get("info", {}).get("rejectReason", "Unknown reason") # V5 field? Check Bybit docs/CCXT source
                 if not reason or reason == "Unknown reason":
                      reason = order.get("info", {}).get("retMsg", "Unknown reason") # Fallback V5 field

                 logger.error(f"{Fore.RED}Market {order_type_desc} order {order_id} was {order_status}. Reason: {reason}. Check exchange info: {order.get('info')}{Style.RESET_ALL}")
                 return None # Order failed immediately

            # If status is 'open' (e.g., IOC partially filled but not fully) or 'closed' (fully filled IOC), proceed.
            # The verification step later will confirm the actual position state.

            # Short delay before verification allows exchange state to potentially update
            wait_time = self.config.order_check_delay_seconds
            logger.debug(f"Waiting short delay ({wait_time}s) after order {order_id} submission before verification...")
            time.sleep(wait_time)

            return order # Return the submitted order info dictionary for potential use

        except (ccxt.InsufficientFunds, ccxt.InvalidOrder) as e:
            # These are caught by fail_fast in fetch_with_retries, but catch again for specific logging
            logger.error(f"{Fore.RED}Market {order_type_desc} order placement failed ({type(e).__name__}): {e}{Style.RESET_ALL}")
            termux_notify(f"{symbol} Order FAILED", f"Market {side.upper()} {order_type_desc} failed: {str(e)[:50]}")
            return None
        except Exception as e:
            # Includes errors raised by fetch_with_retries after max retries fail
            logger.error(f"{Fore.RED}Unexpected error or max retries reached placing market {order_type_desc} order: {e}{Style.RESET_ALL}", exc_info=True)
            termux_notify(f"{symbol} Order ERROR", f"Market {side.upper()} {order_type_desc} error.")
            return None

    def _set_position_protection(
        self,
        position_side: str, # 'long' or 'short'
        sl_price: Optional[Decimal] = None,
        tp_price: Optional[Decimal] = None,
        is_tsl: bool = False, # Flag to indicate if TSL parameters should be used
        tsl_distance: Optional[Decimal] = None, # TSL distance (price difference)
        tsl_activation_price: Optional[Decimal] = None, # TSL activation price level
    ) -> bool:
        """Sets SL, TP, or TSL for a position using the V5 setTradingStop endpoint via a private CCXT call.

        Handles formatting parameters and updating internal protection state tracker.
        Clears existing SL/TP when activating TSL, and clears TSL when setting fixed SL/TP.
        Setting a parameter to 0 via the API cancels that specific stop type.

        Args:
            position_side: 'long' or 'short'.
            sl_price: Fixed stop loss price level (Decimal).
            tp_price: Fixed take profit price level (Decimal).
            is_tsl: If True, attempts to set TSL using tsl_distance and tsl_activation_price.
            tsl_distance: Trailing stop distance (positive Decimal price difference). Required if is_tsl is True.
            tsl_activation_price: Trailing stop activation price level (Decimal). Required if is_tsl is True for V5.

        Returns:
            bool: True if the API call seems successful (retCode 0), False otherwise.
        """
        if not self.exchange: logger.error("Cannot set protection: Exchange missing."); return False
        if not self.market_info: logger.error("Cannot set protection: Market info missing."); return False
        market_id = self.market_info.get("id")
        if not market_id: logger.error("Cannot set protection: Market ID missing."); return False

        tracker_key = position_side.lower()
        if tracker_key not in self.protection_tracker:
             logger.error(f"Invalid position_side '{position_side}' for protection tracker."); return False

        # --- Format Parameters using Helper ---
        # Allow zero for SL/TP as "0" means cancel that specific stop type in the API call.
        sl_str = self.exchange_manager._format_v5_param(sl_price, "price", allow_zero=True)
        tp_str = self.exchange_manager._format_v5_param(tp_price, "price", allow_zero=True)
        # TSL distance requires formatting as price difference; zero is invalid for setting *new* TSL distance.
        tsl_distance_str = self.exchange_manager._format_v5_param(tsl_distance, "distance", allow_zero=False)
        # TSL activation price needs price formatting; zero is invalid for setting *new* activation price.
        tsl_activation_price_str = self.exchange_manager._format_v5_param(tsl_activation_price, "price", allow_zero=False)

        # --- Prepare Base Parameters for V5 private_post_position_set_trading_stop ---
        # Default all stops to "0" (meaning cancel/do not set/leave unchanged if already 0)
        params: Dict[str, Any] = {
            "category": self.config.bybit_v5_category,
            "symbol": market_id,
            "positionIdx": self.config.position_idx,
            "tpslMode": V5_TPSL_MODE_FULL, # Assume full position protection
            "stopLoss": "0",
            "takeProfit": "0",
            "trailingStop": "0", # This is the TSL distance value for V5
            "activePrice": "0", # This is the TSL activation price for V5
            # Trigger prices based on config
            "slTriggerBy": self.config.sl_trigger_by,
            "tpTriggerBy": self.config.sl_trigger_by, # Use same trigger for TP as SL (common practice)
            "tslTriggerBy": self.config.tsl_trigger_by,
            # V5 also has slOrderType, tpOrderType (default Market) - usually okay as default
        }

        action_desc = ""
        new_tracker_state: Optional[str] = None

        # --- Logic Branch: Activate/Update TSL ---
        # Requires both distance and activation price for V5 TSL via this endpoint for initial activation.
        # To modify an active TSL, you might only need to send the distance again? Check API docs.
        # Assuming here we always set both when `is_tsl` is True.
        if is_tsl:
            if tsl_distance_str and tsl_activation_price_str:
                params["trailingStop"] = tsl_distance_str
                params["activePrice"] = tsl_activation_price_str
                # Explicitly clear fixed SL/TP when setting/activating TSL as per V5 logic
                params["stopLoss"] = "0"
                params["takeProfit"] = "0"
                action_desc = f"ACTIVATE/UPDATE TSL (Dist: {tsl_distance_str}, ActPx: {tsl_activation_price_str})"
                new_tracker_state = "ACTIVE_TSL"
                logger.debug(f"Attempting TSL activation/update with params: {params}")
            else:
                logger.error(f"Cannot activate TSL for {position_side.upper()}: Missing or invalid formatted TSL distance ('{tsl_distance_str}') or activation price ('{tsl_activation_price_str}').")
                return False # Cannot proceed if required TSL params are missing

        # --- Logic Branch: Set/Update Fixed SL/TP ---
        # Only proceeds if not activating TSL (`is_tsl` is False)
        # Allows setting only SL, only TP, or both. Sending "0" cancels that specific type.
        elif not is_tsl:
            has_sltp_action = False
            if sl_str is not None: # If sl_price was provided (even if formatted to "0")
                params["stopLoss"] = sl_str
                has_sltp_action = True
            if tp_str is not None: # If tp_price was provided (even if formatted to "0")
                params["takeProfit"] = tp_str
                has_sltp_action = True

            if has_sltp_action:
                # Explicitly clear TSL parameters when setting fixed SL/TP
                params["trailingStop"] = "0"
                params["activePrice"] = "0"
                action_desc = f"SET/UPDATE SL={params['stopLoss']} TP={params['takeProfit']}"
                # Update tracker state only if SL or TP is actually set (not "0")
                if params["stopLoss"] != "0" or params["takeProfit"] != "0":
                    new_tracker_state = "ACTIVE_SLTP"
                else:
                    new_tracker_state = None # All stops cleared
                logger.debug(f"Attempting SL/TP update with params: {params}")
            else:
                # No SL or TP price provided, and not setting TSL. This implies clearing all stops.
                action_desc = "CLEAR ALL STOPS (SL/TP/TSL)"
                new_tracker_state = None
                logger.debug(f"Clearing all stops for {position_side.upper()} position (no valid SL/TP/TSL provided). Params: {params}")


        # --- Execute API Call ---
        # Use the snake_case private method name corresponding to POST /v5/position/set-trading-stop
        # Found via CCXT source or documentation for Bybit V5.
        private_method_name = "private_post_position_set_trading_stop"

        if not hasattr(self.exchange, private_method_name):
            logger.error(
                f"{Fore.RED}{Style.BRIGHT}Private method '{private_method_name}' not found in CCXT instance. "
                f"Cannot set position protection. Check CCXT version/Bybit implementation.{Style.RESET_ALL}"
            )
            return False

        # Get the method from the exchange instance
        method_to_call = getattr(self.exchange, private_method_name)
        symbol_log = self.config.symbol # Use the display symbol for logging

        logger.trade(f"{Fore.CYAN}Attempting to {action_desc} for {position_side.upper()} {symbol_log}...{Style.RESET_ALL}")

        try:
            response = fetch_with_retries(
                method_to_call, # Pass the bound private method
                params=params, # Pass the prepared V5 parameters dictionary
                max_retries=self.config.max_fetch_retries,
                delay_seconds=self.config.retry_delay_seconds,
                # Fail fast on errors indicating invalid parameters if identifiable
                fail_fast_exceptions=(ccxt.InvalidOrder,), # Example, check Bybit errors
            )
            # logger.debug(f"SetTradingStop Raw Response: {response}") # Very verbose

            # --- Process Response ---
            # V5 success code is typically 0
            if isinstance(response, dict) and response.get("retCode") == V5_SUCCESS_RETCODE:
                logger.trade(f"{Fore.GREEN}{Style.BRIGHT}{action_desc} successful for {position_side.upper()} {symbol_log}.{Style.RESET_ALL}")
                termux_notify(f"{symbol_log} Protection Set", f"{action_desc} {position_side.upper()}")
                # Update internal tracker on success
                self.protection_tracker[tracker_key] = new_tracker_state
                logger.debug(f"Protection tracker state for {tracker_key} updated to: {new_tracker_state}")
                return True
            else:
                # Handle API error response
                ret_code = response.get("retCode", "[N/A]") if isinstance(response, dict) else "[No Response Dict]"
                ret_msg = response.get("retMsg", "[No error message]") if isinstance(response, dict) else str(response)
                logger.error(
                    f"{Fore.RED}{action_desc} failed for {position_side.upper()} {symbol_log}. "
                    f"API Response: Code={ret_code}, Msg='{ret_msg}'{Style.RESET_ALL}"
                )
                logger.debug(f"Failed SetTradingStop Full Response: {response}")
                termux_notify(f"{symbol_log} Protection FAILED", f"{action_desc[:30]}... failed: {ret_msg[:50]}...")
                # Should we reset the tracker state on failure? Safer not to assume the state.
                # Keep previous state until confirmed otherwise.
                return False
        except ccxt.InvalidOrder as e:
             # Specific handling for fail-fast exception if caught by fetch_with_retries
             logger.error(f"{Fore.RED}Invalid parameters during {action_desc} for {position_side.upper()} {symbol_log}: {e}{Style.RESET_ALL}")
             termux_notify(f"{symbol_log} Protection FAILED", f"{action_desc[:30]}... invalid params.")
             return False
        except Exception as e:
            # Includes errors raised by fetch_with_retries after max retries fail
            logger.error(
                f"{Fore.RED}Unexpected error or max retries reached during {action_desc} "
                f"for {position_side.upper()} {symbol_log}: {e}{Style.RESET_ALL}", exc_info=True
            )
            termux_notify(f"{symbol_log} Protection ERROR", f"{action_desc[:30]}... error.")
            return False

    def _verify_position_state(
            self,
            expected_side: Optional[str], # 'long', 'short', or None (for expecting flat)
            expected_qty_min: Decimal = POSITION_QTY_EPSILON, # Min absolute qty if expecting position
            max_attempts: int = 4, # Number of verification attempts
            delay_seconds: float = 1.5, # Delay between attempts
            action_context: str = "Verification" # Logging context (e.g., "Post-Entry", "Post-Close")
        ) -> tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
        """Fetches position state repeatedly to verify the expected state after an action.

        Args:
            expected_side: 'long', 'short', or None (if expecting flat).
            expected_qty_min: Minimum absolute quantity expected if side is not None.
            max_attempts: How many times to check the position state.
            delay_seconds: Time to wait between checks.
            action_context: String describing the action being verified (for logging).

        Returns:
            tuple[bool, Optional[Dict[str, Dict[str, Any]]]]:
                - verification_success (bool): True if the expected state was observed.
                - final_position_state (Optional[Dict]): The last fetched position state dictionary,
                  or None if fetching failed completely.
        """
        logger.debug(f"Verifying position state ({action_context}): Expecting '{expected_side or 'FLAT'}' (MinQty: {expected_qty_min.normalize() if expected_side else 'N/A'}). Will attempt {max_attempts} times...")
        last_known_position_state: Optional[Dict[str, Dict[str, Any]]] = None

        for attempt in range(max_attempts):
            logger.debug(f"{action_context} verification attempt {attempt + 1}/{max_attempts}...")
            current_positions = self.exchange_manager.get_current_position()
            last_known_position_state = current_positions # Store the latest fetched state regardless of validity

            if current_positions is None:
                logger.warning(f"{Fore.YELLOW}{action_context} Verification Warning: Failed to fetch position state on attempt {attempt + 1}.{Style.RESET_ALL}")
                if attempt < max_attempts - 1:
                    time.sleep(delay_seconds)
                    continue
                else:
                    logger.error(f"{Fore.RED}{action_context} Verification Failed: Could not fetch position state after {max_attempts} attempts.{Style.RESET_ALL}")
                    # Verification failed, return the last state (which is None here)
                    return False, None

            # --- Check current state against expected state ---
            is_currently_flat = True
            actual_side: Optional[str] = None
            actual_qty = Decimal("0")
            active_pos_data: Optional[Dict[str, Any]] = None

            long_pos_data = current_positions.get("long")
            short_pos_data = current_positions.get("short")

            # Check if long position exists and has significant quantity
            if long_pos_data and isinstance(long_pos_data.get("qty"), Decimal):
                 long_qty = long_pos_data["qty"].copy_abs()
                 if long_qty >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "long"
                     actual_qty = long_qty
                     active_pos_data = long_pos_data

            # Check if short position exists and has significant quantity (only if not already long)
            if not active_pos_data and short_pos_data and isinstance(short_pos_data.get("qty"), Decimal):
                 short_qty = short_pos_data["qty"].copy_abs()
                 if short_qty >= POSITION_QTY_EPSILON:
                     is_currently_flat = False
                     actual_side = "short"
                     actual_qty = short_qty
                     active_pos_data = short_pos_data

            # --- Evaluate success based on expectation ---
            verification_met = False
            log_msg_suffix = ""
            if expected_side is None: # Expecting flat
                verification_met = is_currently_flat
                log_msg_suffix = f"Expected FLAT, Actual: {'FLAT' if is_currently_flat else f'{actual_side.upper()} Qty={actual_qty.normalize()}'}"
            elif actual_side == expected_side: # Expecting a specific side, and it matches
                # Check if quantity meets the minimum expectation
                qty_met = actual_qty >= expected_qty_min
                verification_met = qty_met # Success only if side AND quantity match expectation
                log_msg_suffix = (f"Expected {expected_side.upper()} (MinQty~{expected_qty_min.normalize()}), "
                                  f"Actual: {actual_side.upper()} Qty={actual_qty.normalize()} "
                                  f"({'QTY OK' if qty_met else 'QTY TOO LOW'})")
            else: # Expecting a specific side, but actual side is different or flat
                 verification_met = False
                 log_msg_suffix = (f"Expected {expected_side.upper()}, "
                                   f"Actual: {'FLAT' if is_currently_flat else (actual_side.upper() + ' Qty=' + actual_qty.normalize()) if actual_side else 'UNKNOWN'} "
                                   f"(SIDE MISMATCH)")

            logger.debug(f"{action_context} Check {attempt + 1}: {log_msg_suffix}")

            if verification_met:
                logger.info(f"{Fore.GREEN}{Style.BRIGHT}{action_context} Verification SUCCESSFUL on attempt {attempt + 1}.{Style.RESET_ALL}")
                return True, current_positions # Verification succeeded, return the confirmed state

            # Verification not met, wait for next attempt if any remain
            if attempt < max_attempts - 1:
                logger.debug(f"State not as expected, waiting {delay_seconds}s before next check...")
                time.sleep(delay_seconds)
            else:
                 # Max attempts reached, verification failed
                 logger.error(f"{Fore.RED}{action_context} Verification FAILED after {max_attempts} attempts. Final state check: {log_msg_suffix}{Style.RESET_ALL}")
                 # Return failure and the last known state (which might show the discrepancy)
                 return False, current_positions

        # This part should not be reached if max_attempts >= 1
        logger.error(f"{action_context} verification loop ended unexpectedly.")
        return False, last_known_position_state # Return last known state even if loop exits unexpectedly

    def place_risked_market_order(
        self,
        side: str, # 'buy' or 'sell'
        atr: Decimal,
        total_equity: Decimal,
        current_price: Decimal,
    ) -> tuple[bool, Optional[Dict]]:
        """Calculates parameters, places market order, verifies position, and sets initial SL/TP.

        This is the main entry function.

        Args:
            side: 'buy' or 'sell'.
            atr: Validated ATR Decimal value.
            total_equity: Validated total equity Decimal value.
            current_price: Validated current price Decimal value.

        Returns:
            tuple[bool, Optional[Dict]]:
                - success (bool): True if the entire entry sequence was successful.
                - final_position_data (Optional[Dict]): The dictionary containing details of the
                  successfully entered position (e.g., from `get_current_position`), or None if failed.
        """
        if not self.exchange or not self.market_info:
             logger.error("Entry Aborted: OrderManager not properly initialized.")
             return False, None
        if side not in ["buy", "sell"]: logger.error(f"Entry Aborted: Invalid side '{side}'."); return False, None
        # Basic validation of inputs (should be pre-validated, but double-check)
        if atr.is_nan() or atr <= 0: logger.error("Entry Aborted: Invalid ATR provided."); return False, None
        if total_equity.is_nan() or total_equity <= 0: logger.error("Entry Aborted: Invalid Equity provided."); return False, None
        if current_price.is_nan() or current_price <= 0: logger.error("Entry Aborted: Invalid Price provided."); return False, None

        position_side = "long" if side == "buy" else "short"
        logger.info(f"{Fore.MAGENTA}{Style.BRIGHT}--- Initiating Entry Sequence for {position_side.upper()} ---{Style.RESET_ALL}")

        # 1. Calculate Trade Parameters
        logger.debug("Calculating trade parameters...")
        trade_params = self._calculate_trade_parameters(side, atr, total_equity, current_price)
        if not trade_params or not isinstance(trade_params.get("qty"), Decimal) or not isinstance(trade_params.get("sl_price"), Decimal):
            logger.error("Entry Aborted: Failed to calculate valid trade parameters (qty or sl_price missing/invalid).")
            return False, None # Critical calculation failure

        # Extract calculated parameters, ensuring types
        qty_to_order = cast(Decimal, trade_params["qty"])
        initial_sl_price = cast(Decimal, trade_params["sl_price"])
        initial_tp_price = trade_params.get("tp_price") # Can be None
        initial_tsl_distance = trade_params.get("tsl_distance") # Can be None
        initial_tsl_activation_price = trade_params.get("tsl_activation_price") # Can be None

        # 2. Execute Market Order (Entry)
        logger.debug(f"Executing market {side} order for {qty_to_order.normalize()}...")
        order_info = self._execute_market_order(side, qty_to_order, reduce_only=False)
        if not order_info:
            logger.error("Entry Aborted: Market order execution failed or was rejected.")
            # No position cleanup needed here as order submission itself failed
            return False, None
        order_id = order_info.get("id", "[N/A]")

        # 3. Verify Position Establishment
        # We need to confirm that a position of the expected side and approximate quantity now exists.
        logger.info(f"Verifying position establishment after market order {order_id}...")
        # Define verification parameters: Expect at least a fraction of the intended quantity,
        # allowing for fees/slippage impacting the final position size slightly.
        # Using 90% as a threshold - adjust if needed based on typical fee/slippage impact.
        min_expected_qty = qty_to_order * Decimal("0.90")
        verification_ok, final_pos_state_dict = self._verify_position_state(
            expected_side=position_side,
            expected_qty_min=min_expected_qty,
            max_attempts=5, # Allow more attempts for state propagation
            delay_seconds=self.config.order_check_delay_seconds + 0.5, # Use configured delay + buffer
            action_context=f"Post-{position_side.upper()}-Entry"
        )

        if not verification_ok:
            logger.error(f"{Fore.RED}Entry Failed: Position verification FAILED after market order {order_id}. State may be inconsistent. Manual check required! Attempting cleanup...{Style.RESET_ALL}")
            # Attempt to handle the failure (e.g., close unexpected position)
            self._handle_entry_failure(position_side, qty_to_order, final_pos_state_dict)
            return False, None # Entry failed verification

        # Position verified, extract details from the final verified state dictionary
        # The structure is {'long': {...}, 'short': {...}}
        if not final_pos_state_dict or not final_pos_state_dict.get(position_side):
            # This should not happen if verification_ok is True, but safeguard
            logger.error(f"{Fore.RED}Internal Error: Position verified OK but data missing for '{position_side}' in final state dict {final_pos_state_dict}. Halting entry.{Style.RESET_ALL}")
            self._handle_entry_failure(position_side, qty_to_order, final_pos_state_dict)
            return False, None

        # Get the dictionary for the specific side that was verified
        active_pos_data = final_pos_state_dict[position_side]

        # Extract